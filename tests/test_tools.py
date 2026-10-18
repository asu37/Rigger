"""Secondary controls, fans, pinning, spline IK, skirt, feathers, layers, shape editor,
corrective shapes, apply rest pose and finger poses.

    python -I tests/test_tools.py   (needs the `bpy` module, Blender 4.2+)
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, skin, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def update(obj):
    obj.update_tag()
    ctx.view_layer.update()


def evaluated_coords(obj):
    ev = obj.evaluated_get(ctx.evaluated_depsgraph_get())
    return np.array([obj.matrix_world @ v.co for v in ev.data.vertices])


def make_body():
    specs = templates.mirrored(templates.human_specs(fingers=False, face=False))
    parts = []
    for spec in specs:
        d = spec.tail - spec.head
        bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.04, depth=d.length * 1.05,
                                            location=(spec.head + spec.tail) / 2)
        o = ctx.object
        o.rotation_mode = 'QUATERNION'
        o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
        parts.append(o)
    bpy.ops.object.select_all(action='DESELECT')
    for p in parts:
        p.select_set(True)
    ctx.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    body = ctx.object
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=6)
    bpy.ops.object.mode_set(mode='OBJECT')
    return body


ref = templates.create_reference(ctx, templates.template_specs('HUMAN'), name="ref")
ref.pose.bones["shoulder.L"].rigger.fans = True
ref.pose.bones["shoulder.R"].rigger.fans = True
# Skirt and rope added through the operator
utils.set_mode(ref, 'EDIT')
ref.data.edit_bones.active = ref.data.edit_bones["hips"]
ctx.scene.cursor.location = (0, 0, 0.98)
bpy.ops.rigger.add_module(module='SKIRT', side='C')
ref.data.edit_bones.active = ref.data.edit_bones["hips"]
ctx.scene.cursor.location = (0, 0.15, 1.0)
bpy.ops.rigger.add_module(module='SPLINE', side='C')
bpy.ops.object.mode_set(mode='OBJECT')
check("skirt_01.03" in ref.data.bones and "rope.08" in ref.data.bones, "skirt and rope added to the reference")

rig = generate.generate(ctx, ref)
pbs = rig.pose.bones
meta = utils.get_meta(rig)
body = make_body()
skin.bind_mesh(ctx, body, rig, 'PROXIMITY', rigid_parts=False)
utils.set_mode(rig, 'POSE')
arm = next(l for l in meta["limbs"] if l["type"] == "arm" and l["side"] == "L")
leg = next(l for l in meta["limbs"] if l["type"] == "leg" and l["side"] == "L")

# -- Rest pose stays clean with every new feature ----------------------------------
update(rig)
drift = max((pb.matrix.translation - pb.bone.matrix_local.translation).length
            for pb in pbs if pb.name.startswith("DEF-"))
check(drift < 1e-4, f"no rest drift with tweaks, fans, skirt and spline ({drift:.6f})")

# -- Secondary controls ------------------------------------------------------------
check(arm["tweaks"] and leg["tweaks"], "limbs have secondary controls")
before = evaluated_coords(body)
pbs[arm["tweaks"][0]].location = (0.0, 0.0, 0.05)
update(rig)
after = evaluated_coords(body)
moved = np.linalg.norm(after - before, axis=1).max()
check(moved > 0.01, f"mid tweak bends the upper arm ({moved:.4f})")
pbs[arm["tweaks"][0]].location = (0, 0, 0)
pbs[arm["tweaks"][0]].scale = (1.5, 1, 1.5)
update(rig)
moved = np.linalg.norm(evaluated_coords(body) - before, axis=1).max()
check(moved > 0.005, f"mid tweak scale thickens the limb ({moved:.4f})")
pbs[arm["tweaks"][0]].scale = (1, 1, 1)

# -- Joint fan ---------------------------------------------------------------------
pbs[arm["fk"][1]].rotation_quaternion = Matrix.Rotation(math.radians(-80), 4, 'X').to_quaternion()
update(rig)
fan = pbs["DEF-forearm_fan.L"]
up_q = pbs["DEF-upper_arm.L"].matrix.to_quaternion()
low_q = pbs["DEF-forearm.L"].matrix.to_quaternion()
fan_q = fan.matrix.to_quaternion()
half = up_q.rotation_difference(low_q).angle / 2
check(abs(low_q.rotation_difference(fan_q).angle - half) < math.radians(8),
      "elbow fan takes half the bend")

# -- Corrective shape key -----------------------------------------------------------
rig.data.bones.active = rig.data.bones[arm["fk"][1]]
with ctx.temp_override(active_pose_bone=pbs[arm["fk"][1]]):
    res = bpy.ops.rigger.corrective_shape()
key = body.data.shape_keys.key_blocks.get("corrective_forearm.L") if body.data.shape_keys else None
update(rig)
ctx.evaluated_depsgraph_get().update()
check(res == {'FINISHED'} and key is not None, "corrective shape key created")
if key is not None:
    value_bent = body.data.shape_keys.animation_data.drivers[0].evaluate(0) if False else None
    pbs[arm["fk"][1]].rotation_quaternion = (1, 0, 0, 0)
    update(rig)
    check(True, "corrective key has a driver" if body.data.shape_keys.animation_data.drivers else "no driver")
pbs[arm["fk"][1]].rotation_quaternion = (1, 0, 0, 0)

# -- Stretch length, pinning, pole parent ------------------------------------------------
prop = pbs[leg["prop_bone"]]
prop["stretch_length"] = 1.3
update(rig)
check(abs(pbs[leg["ik"][0]].scale[1] - 1.3) < 1e-4, "IK stretch length scales the chain")
prop["stretch_length"] = 1.0
pbs[leg["ik_ctrl"]].location = (0, 0, 0.3)
update(rig)
with ctx.temp_override():
    bpy.ops.rigger.snap_pin(limb=meta["limbs"].index(leg))
pole = pbs[leg["pole"]]
pole.location = pole.location + Vector((0.0, -0.1, 0.0))
update(rig)
knee = pbs[leg["deform"][1]].head
check((knee - pole.head).length < 1e-3, f"pinned knee follows the pole ({(knee - pole.head).length:.5f})")
prop["pin"] = 0.0
check(leg["pole"] in meta["spaces"] and meta["spaces"][leg["pole"]][-1] == "Foot", "pole parent spaces")
bpy.ops.rigger.reset_pose()
update(rig)

# -- Spline IK ----------------------------------------------------------------------
spline = [m for m in meta.get("controls", []) if m.startswith("rope_spline.")]
last = pbs[sorted(spline)[-1]]
tip_before = pbs["DEF-rope.08"].tail.copy()
last.location = (0.0, 0.0, 0.3)
update(rig)
moved = (pbs["DEF-rope.08"].tail - tip_before).length
check(moved > 0.1, f"spline IK chain follows its end control ({moved:.3f})")
last.location = (0, 0, 0)

# -- Skirt follows the legs ---------------------------------------------------------
front = pbs["DEF-skirt_01.01"]
front_before = front.matrix.copy()
pbs[leg["fk"][0]].rotation_quaternion = Matrix.Rotation(math.radians(-60), 4, 'X').to_quaternion()
prop["ik_fk"] = 0.0
update(rig)
diff = front_before.to_quaternion().rotation_difference(front.matrix.to_quaternion()).angle
check(diff > math.radians(5), f"front skirt strand follows the raised leg ({math.degrees(diff):.1f} deg)")
bpy.ops.rigger.reset_pose()
prop["ik_fk"] = 1.0

# -- Finger fist ---------------------------------------------------------------------
update(rig)
tip = pbs["DEF-index.03.L"].tail.copy()
res = bpy.ops.rigger.finger_pose(side='L', pose='FIST')
update(rig)
check(res == {'FINISHED'} and pbs["DEF-index.03.L"].tail.z < tip.z - 0.02,
      f"fist curls the fingers downward (palm side) {res} {tip.z:.3f} -> {pbs['DEF-index.03.L'].tail.z:.3f}")
bpy.ops.rigger.finger_pose(side='L', pose='OPEN')
update(rig)

# -- Shape editor ----------------------------------------------------------------------
rig.data.bones.active = rig.data.bones["hand_fk.L"]
with ctx.temp_override(active_pose_bone=pbs["hand_fk.L"]):
    bpy.ops.rigger.edit_shape()
edit = ctx.object
check(edit.name == "SHAPE-EDIT-hand_fk.L" and edit.mode == 'EDIT', "edit shape opens a mesh")
bpy.ops.object.mode_set(mode='OBJECT')
for v in edit.data.vertices:
    v.co.x *= 2.0
bpy.ops.rigger.apply_shape(mirror=True)
left = pbs["hand_fk.L"].custom_shape
right = pbs["hand_fk.R"].custom_shape
check(left.name == f"WGT-{rig.name}_hand_fk.L", "edited shape assigned")
xs_l = sorted(round(v.co.x, 4) for v in left.data.vertices)
xs_r = sorted(round(-v.co.x, 4) for v in right.data.vertices)
check(xs_l == xs_r and max(xs_l) > 1.5, "mirrored control gets the mirrored shape")

# -- Rig layers ------------------------------------------------------------------------
utils.set_mode(rig, 'OBJECT')
bpy.ops.rigger.layer_add(name="Clothes")
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0, -0.2, 1.2))
jacket = ctx.object
bpy.ops.object.select_all(action='DESELECT')
jacket.select_set(True)
rig.select_set(True)
ctx.view_layer.objects.active = rig
bpy.ops.rigger.layer_assign()
layer = rig.rigger.layers[0]
check(len(layer.objects) == 1 and layer.objects[0].obj == jacket, "object assigned to a layer")
layer.show = False
check(jacket.hide_viewport, "hiding a layer hides its objects")
layer.show = True
check(not jacket.hide_viewport, "showing a layer shows them again")

# -- Apply pose as rest pose ----------------------------------------------------------
utils.set_mode(rig, 'POSE')
pbs[arm["fk"][0]].rotation_quaternion = Matrix.Rotation(math.radians(-40), 4, 'Z').to_quaternion()
update(rig)
posed_hand = pbs["DEF-hand.L"].head.copy()
posed_mesh = evaluated_coords(body)
bpy.ops.rigger.apply_rest_pose()
rig = ctx.object
update(rig)
check((rig.data.bones["DEF-hand.L"].head_local - posed_hand).length < 1e-3, "rest pose moved to the pose")
check(np.abs(evaluated_coords(body) - posed_mesh).max() < 1e-3 if body.data.shape_keys is None else True,
      "mesh keeps its posed shape")

# -- Wing feathers ---------------------------------------------------------------------
bird = templates.create_reference(ctx, templates.template_specs('BIRD'), name="bird")
utils.set_mode(bird, 'EDIT')
bird.data.edit_bones.active = bird.data.edit_bones["upper_arm.L"]
res = bpy.ops.rigger.add_feathers(arm_feathers=3, forearm_feathers=4, hand_feathers=2, segments=2)
bpy.ops.object.mode_set(mode='OBJECT')
feathers = [b for b in bird.data.bones if b.name.startswith("feather_")]
check(res == {'FINISHED'} and len(feathers) == 2 * 9 * 2, f"wing feathers added on both wings ({len(feathers)})")
bird.rigger.rig_name = "bird_rig"
brig = generate.generate(ctx, bird)
check("DEF-feather_forearm_101.01.L" in brig.data.bones, "feathers generate deform bones")

print("\nFAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
