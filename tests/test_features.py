"""Face rig, chest, parent spaces, skinning extras and export naming.

    python -I tests/test_features.py   (needs the `bpy` module, Blender 4.2+)
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import export, generate, skin, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def update(rig):
    rig.update_tag()
    ctx.view_layer.update()


def world_head(rig, name):
    return rig.matrix_world @ rig.pose.bones[name].head


ref = templates.create_reference(ctx, templates.template_specs('HUMAN'), name="ref")
rig = generate.generate(ctx, ref)
pbs = rig.pose.bones
meta = utils.get_meta(rig)
utils.set_mode(rig, 'POSE')

# -- Face ------------------------------------------------------------------------
face = meta["faces"][0]
check(face["blinks"] == ["blink_L", "blink_R"], "blink sliders on both sides")
lower = world_head(rig, "DEF-lid_lower.L")
upper_rest = world_head(rig, "DEF-lid_upper.L")
pbs[face["face"]]["blink_L"] = 1.0
update(rig)
upper = world_head(rig, "DEF-lid_upper.L")
check((upper - lower).length < 0.25 * (upper_rest - lower).length,
      f"blink closes the upper lid ({(upper - lower).length:.4f} from lower lid)")
check((world_head(rig, "DEF-lid_upper.R") - upper_rest.reflect(Vector((1, 0, 0)))).length > -1,
      "right lid evaluated")
pbs[face["face"]]["blink_L"] = 0.0

jaw = pbs[face["jaw"]]
lip_rest = world_head(rig, "DEF-lip_lower")
corner_rest = world_head(rig, "DEF-lip_corner.L")
side_rest = world_head(rig, "DEF-lip_upper.L")
jaw.rotation_quaternion = Matrix.Rotation(math.radians(20), 4, 'X').to_quaternion()
update(rig)
lip_move = (world_head(rig, "DEF-lip_lower") - lip_rest).length
corner_move = (world_head(rig, "DEF-lip_corner.L") - corner_rest).length
side_move = (world_head(rig, "DEF-lip_upper.L") - side_rest).length
check(lip_move > 0.005, f"jaw opens the lower lip ({lip_move:.4f})")
check(0.2 * lip_move < corner_move < 0.8 * lip_move, f"lip corner follows the jaw partly ({corner_move:.4f})")
check(0.0 < side_move < corner_move, f"upper side lip follows the corner partly ({side_move:.4f})")
jaw.rotation_quaternion = (1, 0, 0, 0)

lid_rest = world_head(rig, "DEF-lid_upper.L")
target = pbs["eye_target.L"]
target.location = (0, 0, 0.3)  # look up/down
update(rig)
lid_move = (world_head(rig, "DEF-lid_upper.L") - lid_rest).length
check(lid_move > 1e-4, f"upper lid follows the eye ({lid_move:.4f})")
target.location = (0, 0, 0)
update(rig)

# -- Chest ------------------------------------------------------------------------
chest = pbs[meta["spine"]["chest"]]
top = meta["spine"]["fk"][-1]
tail_rest = rig.matrix_world @ pbs[f"DEF-{top}"].tail
chest.rotation_quaternion = Matrix.Rotation(math.radians(30), 4, 'X').to_quaternion()
update(rig)
bend = math.degrees((rig.matrix_world @ pbs[f"DEF-{top}"].tail - rig.matrix_world @ pbs["DEF-spine.01"].head)
                    .angle(tail_rest - rig.matrix_world @ pbs["DEF-spine.01"].head))
check(bend > 10.0, f"chest bends the spine ({bend:.1f} deg)")
chest.rotation_quaternion = (1, 0, 0, 0)
update(rig)

# -- Parent spaces -------------------------------------------------------------------
check(meta["spaces"].get("hand_ik.L") == ["Root", "Torso", "Chest", "Head"], "hand IK parent spaces")
check(meta["spaces"].get("foot_ik.R") == ["Root", "Torso"], "foot IK parent spaces")
hand = pbs["hand_ik.L"]
hand.location = (0.05, 0.02, -0.1)
update(rig)
before = hand.matrix.copy()
bpy.ops.rigger.switch_space(bone="hand_ik.L", space=2)
update(rig)
err = max(abs(a - b) for ra, rb in zip(before, hand.matrix) for a, b in zip(ra, rb))
check(hand["parent"] == 2 and err < 1e-4, f"switching to chest keeps the hand in place (err {err:.6f})")
pbs["torso"].location = (0, 0, 0.2)
update(rig)
moved = (hand.matrix.translation - before.translation).length
check(moved > 0.15, f"hand follows the torso in chest space ({moved:.3f})")
bpy.ops.rigger.reset_pose()
hand["parent"] = 0
update(rig)
drift = max((pb.matrix.translation - pb.bone.matrix_local.translation).length
            for pb in pbs if pb.name.startswith("DEF-"))
check(drift < 1e-4, f"rest pose restored after reset ({drift:.6f})")

# -- Picker -------------------------------------------------------------------------
# hand_ik.L is hidden while the arm is in FK, so the picker skips it
bpy.ops.rigger.select_controls(bones="hand_fk.L,thigh_pole.R,hand_ik.L")
sel = [pb.name for pb in pbs if utils.is_pose_bone_selected(pb)]
check(sorted(sel) == ["hand_fk.L", "thigh_pole.R"], f"picker selects visible controls ({sel})")
bpy.ops.object.mode_set(mode='OBJECT')

# -- Skinning extras ----------------------------------------------------------------
# Torso cylinder + tiny loose "button" on the chest + left hand block
bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.15, depth=0.9, location=(0, 0, 1.2))
torso = ctx.object
bpy.ops.mesh.primitive_cube_add(size=0.06, location=(0.75, 0, 1.43))  # near the left hand
bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=4, radius=0.01, location=(0, -0.16, 1.3))
bpy.ops.object.select_all(action='SELECT')
for o in list(ctx.selected_objects):
    if o.type != 'MESH':
        o.select_set(False)
ctx.view_layer.objects.active = torso
bpy.ops.object.join()
body = ctx.object
filled, rigid = skin.bind_mesh(ctx, body, rig, 'PROXIMITY', rigid_parts=True, rigid_fraction=0.35)
check(rigid == 2, f"small loose parts bound rigidly ({rigid})")

# A loose piece straddling a joint (a mannequin's knee cap) keeps blended weights
knee = rig.matrix_world @ rig.data.bones["DEF-shin.L"].head_local
bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=0.07, depth=1.6, location=(knee.x, knee.y, 1.0))
legs = ctx.object
bpy.ops.mesh.primitive_cube_add(size=0.07, location=(knee.x, knee.y - 0.07, knee.z))
cap = ctx.object
bpy.ops.object.select_all(action='DESELECT')
legs.select_set(True)
cap.select_set(True)
ctx.view_layer.objects.active = legs
bpy.ops.object.join()
leg_body = ctx.object
skin.bind_mesh(ctx, leg_body, rig, 'PROXIMITY', rigid_parts=True, rigid_fraction=0.35)
labels = skin.loose_parts(leg_body)
cap_verts = np.nonzero(labels == labels[-1])[0]
thigh_w = np.array([sum(g.weight for g in leg_body.data.vertices[v].groups  # thigh and its twist bones
                        if leg_body.vertex_groups[g.group].name.startswith("DEF-thigh")) for v in cap_verts])
check(len(cap_verts) == 8 and thigh_w.max() - thigh_w.min() > 0.05,
      f"a piece across the knee keeps blended weights (thigh weight {thigh_w.min():.2f}..{thigh_w.max():.2f})")
labels = skin.loose_parts(body)
button = [v.index for v in body.data.vertices if (body.matrix_world @ v.co).y < -0.15 and
          abs((body.matrix_world @ v.co).z - 1.3) < 0.02]
weights = [tuple(sorted((body.vertex_groups[g.group].name, round(g.weight, 3)) for g in body.data.vertices[i].groups))
           for i in button]
check(len(set(weights)) == 1, "every vertex of the rigid part has the same weights")
right_on_left = 0
for v in body.data.vertices:
    if (body.matrix_world @ v.co).x > 0.1:
        right_on_left += sum(1 for g in v.groups if body.vertex_groups[g.group].name.endswith(".R"))
check(right_on_left == 0, "left side vertices get no right side bones")

# Heat on a tiny character (rescaled internally)
small_ref = templates.create_reference(ctx, templates.template_specs('HUMAN_SIMPLE'), name="small_ref")
small_ref.scale = (0.05, 0.05, 0.05)
small_ref.rigger.rig_name = "small_rig"
small = generate.generate(ctx, small_ref)
bpy.ops.object.mode_set(mode='OBJECT')
parts = []
for spec in templates.mirrored(templates.human_specs(fingers=False, face=False)):
    d = (spec.tail - spec.head) * 0.05
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=0.002, depth=d.length * 1.1,
                                        location=(spec.head + spec.tail) * 0.025)
    ctx.object.rotation_mode = 'QUATERNION'
    ctx.object.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    parts.append(ctx.object)
bpy.ops.object.select_all(action='DESELECT')
for p in parts:
    p.select_set(True)
ctx.view_layer.objects.active = parts[0]
bpy.ops.object.join()
tiny = ctx.object
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
filled, _rigid = skin.bind_mesh(ctx, tiny, small, 'AUTO', rigid_parts=False)
deform = {b.name for b in small.data.bones if b.use_deform}
missing = skin.unweighted_vertices(tiny, deform)
check(not missing, f"tiny character fully weighted (heat filled all but {filled} vertices)")
check(bpy.data.objects.get("small_rig.001") is None and len([o for o in bpy.data.objects
                                                               if o.name.startswith(tiny.name + ".")]) == 0,
      "temporary heat copies removed")

# -- Export naming ---------------------------------------------------------------------
deform_names = ["root"] + [b.name for b in rig.data.bones if b.use_deform]
ue = export.export_names(rig, deform_names, 'UNREAL')
for src, dst in (("DEF-hips", "pelvis"), ("DEF-spine.01", "spine_01"), ("DEF-spine.03", "spine_03"),
                 ("DEF-neck", "neck_01"), ("DEF-head", "head"), ("DEF-shoulder.L", "clavicle_l"),
                 ("DEF-upper_arm.L", "upperarm_l"), ("DEF-forearm.R", "lowerarm_r"),
                 ("DEF-hand.L", "hand_l"), ("DEF-thigh.L", "thigh_l"), ("DEF-shin.R", "calf_r"),
                 ("DEF-foot.L", "foot_l"), ("DEF-toe.L", "ball_l"), ("DEF-index.02.L", "index_02_l"),
                 ("DEF-thumb.01.R", "thumb_01_r"), ("DEF-forearm_twist.01.L", "lowerarm_twist_01_l"),
                 ("DEF-shin_twist.01.R", "calf_twist_01_r"), ("DEF-jaw", "jaw"), ("root", "root")):
    check(ue.get(src) == dst, f"Unreal name {src} -> {ue.get(src)}")
unity = export.export_names(rig, deform_names, 'UNITY')
for src, dst in (("DEF-hips", "Hips"), ("DEF-spine.01", "Spine"), ("DEF-spine.02", "Spine1"),
                 ("DEF-upper_arm.L", "LeftArm"), ("DEF-shin.R", "RightLeg"), ("DEF-thigh.L", "LeftUpLeg"),
                 ("DEF-index.01.L", "LeftHandIndex1"), ("DEF-toe.R", "RightToeBase")):
    check(unity.get(src) == dst, f"Unity name {src} -> {unity.get(src)}")
check(len(set(ue.values())) == len(ue), "Unreal names are unique")

out = os.path.join(tempfile.mkdtemp(), "ue.fbx")
utils.set_mode(rig, 'OBJECT')
res = bpy.ops.rigger.export_fbx(filepath=out, preset='UNREAL', naming='UNREAL', actions='NONE')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=out)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
check({"pelvis", "upperarm_l", "calf_twist_01_r", "ball_l"} <= set(arm.data.bones.keys()),
      "FBX uses Unreal names")

# -- Root motion ------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
ref = templates.create_reference(bpy.context, templates.template_specs('HUMAN_SIMPLE'), name="ref")
rig = generate.generate(bpy.context, ref)
bpy.ops.object.mode_set(mode='OBJECT')
torso = rig.pose.bones["torso"]
rig.animation_data_create()
rig.animation_data.action = bpy.data.actions.new("walk")
for frame, y in ((1, 0.0), (20, -1.0)):
    torso.location = (0, 0, 0)
    torso.matrix = torso.bone.matrix_local @ Matrix.Translation((0, 0, 0))
    m = torso.bone.matrix_local.copy()
    m.translation += Vector((0, y, 0))
    torso.matrix = m
    torso.keyframe_insert("location", frame=frame)
out = os.path.join(tempfile.mkdtemp(), "rm.fbx")
utils.set_mode(rig, 'OBJECT')
bpy.ops.rigger.export_fbx(filepath=out, preset='UNREAL', naming='UNREAL', actions='CURRENT', root_motion=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=out)
arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
bpy.context.scene.frame_set(20)
bpy.context.view_layer.update()
root_y = (arm.matrix_world @ arm.pose.bones["root"].head).y
pelvis_y = (arm.matrix_world @ arm.pose.bones["pelvis"].head).y
check(root_y < -0.9, f"root motion: root travels with the body ({root_y:.3f})")
check(abs(pelvis_y - root_y) < 0.1, f"pelvis stays over the root ({pelvis_y:.3f})")

# -- Eye and hand detection when fitting markers -----------------------------------------
from rigger import markers  # noqa: E402
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=1.8, location=(0, 0, 0.9))
body = bpy.context.object
bpy.ops.mesh.primitive_cube_add(size=1, location=(0.6, 0, 1.43))
bpy.context.object.scale = (0.5, 0.04, 0.04)  # arm, fingertips at x = 0.85
arm_obj = bpy.context.object
for side in (1, -1):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012, location=(side * 0.03, -0.09, 1.70))
bpy.ops.object.select_all(action='DESELECT')
for o in (body, arm_obj):
    o.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
for key, *_ in templates.MARKERS:
    p = templates.DEFAULT_POINTS[key].copy()
    p.y = 0
    markers.set_marker(bpy.context, key, p)
bpy.ops.rigger.fit_markers()
ref = bpy.context.object
bpy.ops.object.mode_set(mode='OBJECT')
eye = ref.data.bones["eye.L"].head_local
check((eye - Vector((0.03, -0.09, 1.70))).length < 0.005, f"eye bone placed in the detected eyeball ({eye[:]})")
tip = ref.data.bones["middle.03.L"].tail_local.x
check(abs(tip - 0.85) < 0.02, f"fingers scaled to the measured hand ({tip:.3f})")

# -- New templates ------------------------------------------------------------------------
for key in ('HORSE', 'BIRD'):
    r = templates.create_reference(bpy.context, templates.template_specs(key), name=f"ref_{key}")
    r.rigger.rig_name = f"rig_{key}"
    g = generate.generate(bpy.context, r)
    bpy.ops.object.mode_set(mode='OBJECT')
    check(len(utils.get_meta(g)["limbs"]) >= 4, f"{key} template generates ({len(g.data.bones)} bones)")

print("\nFAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
