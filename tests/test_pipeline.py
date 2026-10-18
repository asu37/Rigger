"""End-to-end test of the add-on through its operators.

Run with a Blender Python that provides `bpy` (Blender 4.2+), e.g.:
    blender --background --python tests/test_pipeline.py
or  python -I tests/test_pipeline.py   (with the `bpy` wheel installed)
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import markers, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
scene = ctx.scene
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# -- 1. Smart markers -> reference -------------------------------------------
for key, *_ in templates.MARKERS:
    markers.set_marker(ctx, key, templates.DEFAULT_POINTS[key] * 1.1)
# A body mesh made of capsules along the default skeleton
body_specs = templates.mirrored(templates.human_specs(fingers=False, face=False))
parts = []
for spec in body_specs:
    d = spec.tail - spec.head
    radius = 0.06 if "spine" in spec.name or spec.name in ("hips", "head") else 0.035
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=d.length * 1.15,
                                        location=(spec.head + spec.tail) / 2)
    obj = ctx.object
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    parts.append(obj)
bpy.ops.object.select_all(action='DESELECT')
for p in parts:
    p.select_set(True)
ctx.view_layer.objects.active = parts[0]
bpy.ops.object.join()
body = ctx.object
body.name = "body"
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
body.scale = (1.1, 1.1, 1.1)
bpy.ops.object.transform_apply(scale=True)

res = bpy.ops.rigger.fit_markers()
ref = ctx.object
check(res == {'FINISHED'} and utils.is_reference(ref), "fit reference to markers")
check(abs(ref.data.bones["head"].tail_local.z - body.dimensions.z) < 0.05,
      "head top matches mesh height")

# -- 2. Add module + mirror ----------------------------------------------------
bpy.ops.object.mode_set(mode='EDIT')
ref.data.edit_bones.active = ref.data.edit_bones["hips"]
scene.cursor.location = (0, 0.2, 1.0)
bpy.ops.rigger.add_module(module='CHAIN', side='C')
check("tail.01" in ref.data.edit_bones, "add tail module")
bpy.ops.object.mode_set(mode='OBJECT')
ref.pose.bones["thumb.01.L"].rigger.curl = False
bpy.ops.rigger.mirror_reference()
check(ref.pose.bones["thumb.01.R"].rigger.curl is False, "mirror copies module settings")

# -- 3. Generate -----------------------------------------------------------------
utils.set_mode(ref, 'OBJECT')
res = bpy.ops.rigger.generate()
rig = ctx.object
check(res == {'FINISHED'} and utils.is_rig(rig), "generate rig")
check("tail_master" in rig.pose.bones and "DEF-tail.04" in rig.pose.bones, "tail generated")
meta = utils.get_meta(rig)
check(len(meta["limbs"]) == 4, "four limbs in metadata")

# -- 4. Skin -----------------------------------------------------------------------
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
rig.select_set(True)
ctx.view_layer.objects.active = rig
res = bpy.ops.rigger.bind(method='AUTO')
deform = {b.name for b in rig.data.bones if b.use_deform}
from rigger.skin import unweighted_vertices  # noqa: E402
check(res == {'FINISHED'} and not unweighted_vertices(body, deform), "every vertex weighted")
mod = next(m for m in body.modifiers if m.type == 'ARMATURE')
check(mod.object == rig, "armature modifier points to rig")

# Proximity method on its own
bpy.ops.object.select_all(action='DESELECT')
body.select_set(True)
rig.select_set(True)
ctx.view_layer.objects.active = rig
bpy.ops.rigger.bind(method='PROXIMITY')
check(not unweighted_vertices(body, deform), "proximity weights cover all vertices")

# -- 5. IK/FK snapping ---------------------------------------------------------
utils.set_mode(rig, 'POSE')
limb = next(i for i, l in enumerate(meta["limbs"]) if l["type"] == "arm" and l["side"] == "L")
arm = meta["limbs"][limb]
pbs = rig.pose.bones
pbs[arm["prop_bone"]]["ik_fk"] = 0.0
pbs[arm["fk"][0]].rotation_quaternion = Matrix.Rotation(math.radians(-50), 4, 'X').to_quaternion()
pbs[arm["fk"][1]].rotation_quaternion = Matrix.Rotation(math.radians(-40), 4, 'X').to_quaternion()
ctx.view_layer.update()
hand_before = pbs["DEF-hand.L"].matrix.copy()
bpy.ops.rigger.snap_ik_to_fk(limb=limb)
ctx.view_layer.update()
err = max(abs(a - b) for ra, rb in zip(hand_before, pbs["DEF-hand.L"].matrix) for a, b in zip(ra, rb))
check(err < 1e-3, f"IK snapped to FK keeps the hand in place (err {err:.5f})")
bpy.ops.rigger.reset_pose()
bpy.ops.rigger.snap_fk_to_ik(limb=limb)
ctx.view_layer.update()
check(pbs[arm["prop_bone"]]["ik_fk"] == 0.0, "FK snap switches to FK")

leg_i = next(i for i, l in enumerate(meta["limbs"]) if l["type"] == "leg" and l["side"] == "R")
leg = meta["limbs"][leg_i]
pbs[leg["prop_bone"]]["ik_fk"] = 0.0
pbs[leg["fk"][0]].rotation_quaternion = Matrix.Rotation(math.radians(40), 4, 'X').to_quaternion()
pbs[leg["fk"][1]].rotation_quaternion = Matrix.Rotation(math.radians(-60), 4, 'X').to_quaternion()
ctx.view_layer.update()
foot_before = pbs["DEF-foot.R"].matrix.copy()
bpy.ops.rigger.snap_ik_to_fk(limb=leg_i)
ctx.view_layer.update()
err = max(abs(a - b) for ra, rb in zip(foot_before, pbs["DEF-foot.R"].matrix) for a, b in zip(ra, rb))
check(err < 1e-3, f"leg IK snapped to FK keeps the foot in place (err {err:.5f})")
bpy.ops.rigger.reset_pose()
for l in meta["limbs"]:
    pbs[l["prop_bone"]]["ik_fk"] = 1.0 if l["type"] == "leg" else 0.0
bpy.ops.object.mode_set(mode='OBJECT')

# -- 6. Retarget from a Mixamo-style skeleton --------------------------------------
MIXAMO = {
    "hips": "Hips", "spine.01": "Spine", "spine.02": "Spine1", "spine.03": "Spine2",
    "neck": "Neck", "head": "Head",
}
for s in "LR":
    side = "Left" if s == "L" else "Right"
    MIXAMO.update({f"shoulder.{s}": f"{side}Shoulder", f"upper_arm.{s}": f"{side}Arm",
                   f"forearm.{s}": f"{side}ForeArm", f"hand.{s}": f"{side}Hand",
                   f"thigh.{s}": f"{side}UpLeg", f"shin.{s}": f"{side}Leg",
                   f"foot.{s}": f"{side}Foot", f"toe.{s}": f"{side}ToeBase"})
src_specs = [s for s in templates.mirrored(templates.human_specs(fingers=False, face=False))]
src_data = bpy.data.armatures.new("mixamo")
src = bpy.data.objects.new("mixamo", src_data)
scene.collection.objects.link(src)
src.location = (1.5, 0, 0)
utils.set_mode(src, 'EDIT')
for spec in src_specs:
    eb = src_data.edit_bones.new("mixamorig:" + MIXAMO[spec.name])
    eb.head, eb.tail = spec.head * 0.9, spec.tail * 0.9
    eb.roll = 0.7  # different bone rolls than the target
for spec in src_specs:
    if spec.parent:
        src_data.edit_bones["mixamorig:" + MIXAMO[spec.name]].parent = \
            src_data.edit_bones["mixamorig:" + MIXAMO[spec.parent]]
bpy.ops.object.mode_set(mode='POSE')
sp = src.pose.bones
for pb in sp:
    pb.rotation_mode = 'QUATERNION'
src.animation_data_create()
src.animation_data.action = bpy.data.actions.new("walk")
for frame, angle, fwd in ((1, 0, 0.0), (10, 60, 0.3)):
    world_rot = Matrix.Rotation(math.radians(angle), 3, 'Y')  # rotate left arm down
    pb = sp["mixamorig:LeftArm"]
    rest = pb.bone.matrix_local.to_3x3()
    pb.rotation_quaternion = (rest.inverted() @ world_rot @ rest).to_quaternion()
    pb.keyframe_insert("rotation_quaternion", frame=frame)
    hips = sp["mixamorig:Hips"]
    hips.location = hips.bone.matrix_local.to_3x3().inverted() @ Vector((0, -fwd, 0))
    hips.keyframe_insert("location", frame=frame)
    knee = sp["mixamorig:RightLeg"]
    knee.rotation_quaternion = Matrix.Rotation(math.radians(-angle), 4, 'X').to_quaternion()
    knee.keyframe_insert("rotation_quaternion", frame=frame)
bpy.ops.object.mode_set(mode='OBJECT')

remap = scene.rigger_remap
remap.source, remap.target = src, rig
res = bpy.ops.rigger.remap_auto()
pairs = {i.source: i.target for i in remap.items}
check(pairs.get("mixamorig:LeftArm") == "upper_arm_fk.L", "auto map arm")
check(pairs.get("mixamorig:RightFoot") == "foot_ik.R", "auto map foot to IK")
check(pairs.get("mixamorig:Hips") == "torso", "auto map hips to torso")
res = bpy.ops.rigger.remap_bake()
check(res == {'FINISHED'}, "retarget bake")
scene.frame_set(10)
ctx.view_layer.update()
src_dir = (src.matrix_world @ sp["mixamorig:LeftForeArm"].tail) - (src.matrix_world @ sp["mixamorig:LeftArm"].head)
tgt_dir = (rig.matrix_world @ rig.pose.bones["DEF-forearm.L"].tail) - \
          (rig.matrix_world @ rig.pose.bones["DEF-upper_arm.L"].head)
angle = math.degrees(src_dir.angle(tgt_dir))
check(angle < 5.0, f"retargeted arm follows source direction ({angle:.2f} deg)")
hips_move = rig.pose.bones["DEF-hips"].head - rig.data.bones["DEF-hips"].head_local
check(abs(hips_move.y + 0.3 / 0.9 * 1.0) < 0.05 or abs(hips_move.y + 0.3) < 0.08,
      f"hips moved forward, scaled ({hips_move.y:.3f})")
src_knee = math.degrees((sp["mixamorig:RightLeg"].tail - sp["mixamorig:RightLeg"].head).angle(
    sp["mixamorig:RightUpLeg"].tail - sp["mixamorig:RightUpLeg"].head))
tgt_knee = math.degrees((rig.pose.bones["DEF-shin.R"].tail - rig.pose.bones["DEF-shin.R"].head).angle(
    rig.pose.bones["DEF-thigh.R"].tail - rig.pose.bones["DEF-thigh.R"].head))
check(abs(src_knee - tgt_knee) < 6.0, f"IK knee bend matches ({src_knee:.1f} vs {tgt_knee:.1f})")

# -- 6b. Auto-hide drivers and panel drawing ----------------------------------------
pbs[leg["prop_bone"]]["ik_fk"] = 0.0
rig.update_tag()
ctx.view_layer.update()
check(not rig.data.bones[leg["fk"][0]].hide and rig.data.bones[leg["ik_ctrl"]].hide,
      "auto-hide shows FK / hides IK in FK mode")
pbs[leg["prop_bone"]]["ik_fk"] = 1.0
rig.update_tag()
ctx.view_layer.update()
check(rig.data.bones[leg["fk"][0]].hide and not rig.data.bones[leg["ik_ctrl"]].hide,
      "auto-hide hides FK in IK mode")


class _Layout:
    """Accepts any layout call, to run panel draw code without a UI."""

    def __getattr__(self, _name):
        return lambda *a, **k: self

    def __setattr__(self, name, value):
        pass


from rigger import ui  # noqa: E402

for obj in (rig, rig.rigger.reference, body):
    utils.set_mode(obj, 'OBJECT')
    for panel in [c for c in ui.classes if issubclass(c, bpy.types.Panel)][1:]:
        if hasattr(panel, "poll") and not panel.poll(ctx):
            continue
        try:
            panel.draw(type("P", (), {"layout": _Layout()})(), ctx)
        except Exception as exc:  # noqa: BLE001
            check(False, f"{panel.__name__} draw on {obj.name}: {exc!r}")
check(True, "panels draw")

# -- 7. Export --------------------------------------------------------------------
out = os.path.join(tempfile.mkdtemp(), "character.fbx")
utils.set_mode(rig, 'OBJECT')
objects_before = set(bpy.data.objects.keys())
res = bpy.ops.rigger.export_fbx(filepath=out, preset='UNREAL', actions='ALL')
check(res == {'FINISHED'} and os.path.getsize(out) > 1000, "FBX written")
check(set(bpy.data.objects.keys()) == objects_before, "export leaves the scene untouched")
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=out)
arms = [o for o in bpy.data.objects if o.type == 'ARMATURE']
names = set(arms[0].data.bones.keys()) if arms else set()
check(len(arms) == 1 and "root" in names and "upper_arm.L" in names, "FBX skeleton has clean names")
check(not any(n.startswith(("MCH-", "DEF-")) or n.endswith("_fk.L") for n in names),
      "FBX skeleton has no control/mechanism bones")
check(len(bpy.data.actions) >= 1, f"FBX contains animation ({len(bpy.data.actions)} action(s))")
check(any(o.type == 'MESH' for o in bpy.data.objects), "FBX contains the mesh")
if arms and arms[0].animation_data and arms[0].animation_data.action:
    imported = arms[0]
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()
    m1 = imported.pose.bones["upper_arm.L"].matrix.copy()
    bpy.context.scene.frame_set(10)
    bpy.context.view_layer.update()
    m10 = imported.pose.bones["upper_arm.L"].matrix.copy()
    moved = max(abs(a - b) for ra, rb in zip(m1, m10) for a, b in zip(ra, rb))
    check(moved > 0.1, f"exported animation moves the arm ({moved:.3f})")

rigger.unregister()
print("\nFAILURES:", failures if failures else "none")
sys.exit(1 if failures else 0)
