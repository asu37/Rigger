"""Remap options and Quick Rig.

    blender -b --factory-startup --python-exit-code 1 --python tests/test_remap_quick.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, quick_rig, retarget, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
scene = ctx.scene
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


MIXAMO = {
    "hips": "Hips", "spine.01": "Spine", "spine.02": "Spine1", "spine.03": "Spine2",
    "neck": "Neck", "head": "Head",
}
FINGER_MIXAMO = {"thumb": "Thumb", "index": "Index", "middle": "Middle", "ring": "Ring", "pinky": "Pinky"}
for s in "LR":
    side = "Left" if s == "L" else "Right"
    MIXAMO.update({f"shoulder.{s}": f"{side}Shoulder", f"upper_arm.{s}": f"{side}Arm",
                   f"forearm.{s}": f"{side}ForeArm", f"hand.{s}": f"{side}Hand",
                   f"thigh.{s}": f"{side}UpLeg", f"shin.{s}": f"{side}Leg",
                   f"foot.{s}": f"{side}Foot", f"toe.{s}": f"{side}ToeBase"})
    for f, mf in FINGER_MIXAMO.items():
        for i in (1, 2, 3):
            MIXAMO[f"{f}.{i:02d}.{s}"] = f"{side}Hand{mf}{i}"


def mixamo_skeleton(name="mixamo", prefix="mixamorig:", scale=1.0, location=(1.5, 0, 0), extras=False,
                    fingers=False):
    """Mixamo-style skeleton. scale=0.01 mimics an FBX import (object scale 0.01, bones x100)."""
    specs = templates.mirrored(templates.human_specs(fingers=fingers, face=False))
    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.scale = (scale,) * 3
    utils.set_mode(obj, 'EDIT')
    k = 0.9 / scale
    root = None
    if extras:
        root = data.edit_bones.new("Root")
        root.head, root.tail = Vector((0, 0, 0)), Vector((0, 0.1 * k, 0))
    for spec in specs:
        eb = data.edit_bones.new(prefix + MIXAMO[spec.name])
        eb.head, eb.tail = spec.head * k, spec.tail * k
        eb.roll = 0.7  # different bone rolls than the target
    for spec in specs:
        eb = data.edit_bones[prefix + MIXAMO[spec.name]]
        if spec.parent:
            eb.parent = data.edit_bones[prefix + MIXAMO[spec.parent]]
        elif root is not None:
            eb.parent = root
    if extras:
        head = data.edit_bones[prefix + "Head"]
        end = data.edit_bones.new(prefix + "HeadTop_End")
        end.head, end.tail = head.tail.copy(), head.tail + Vector((0, 0, 0.05 * k))
        end.parent = head
        arm = data.edit_bones[prefix + "LeftArm"]
        twist = data.edit_bones.new(prefix + "LeftArmTwist")
        twist.head, twist.tail = arm.head.lerp(arm.tail, 0.5), arm.tail.copy()
        twist.parent = arm
        hips = data.edit_bones[prefix + "Hips"]
        tail = data.edit_bones.new(prefix + "Tail")
        tail.head, tail.tail = hips.head + Vector((0, 0.1 * k, 0)), hips.head + Vector((0, 0.3 * k, -0.1 * k))
        tail.parent = hips
    bpy.ops.object.mode_set(mode='POSE')
    for pb in obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def key_walk(obj, prefix="mixamorig:", name="walk", travel=0.3, angle=60):
    sp = obj.pose.bones
    action = bpy.data.actions.new(name)
    utils.assign_action(obj, action)
    unit = 1.0 / obj.scale[0]
    for frame, a, fwd in ((1, 0, 0.0), (10, angle, travel)):
        world_rot = Matrix.Rotation(math.radians(a), 3, 'Y')  # rotate the left arm down
        pb = sp[prefix + "LeftArm"]
        rest = pb.bone.matrix_local.to_3x3()
        pb.rotation_quaternion = (rest.inverted() @ world_rot @ rest).to_quaternion()
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        pb = sp[prefix + "LeftForeArm"]
        pb.rotation_quaternion = Matrix.Rotation(math.radians(a * 0.5), 4, 'Z').to_quaternion()
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        hips = sp[prefix + "Hips"]
        hips.location = hips.bone.matrix_local.to_3x3().inverted() @ Vector((0, -fwd * unit, 0))
        hips.keyframe_insert("location", frame=frame)
        knee = sp[prefix + "RightLeg"]
        knee.rotation_quaternion = Matrix.Rotation(math.radians(-a), 4, 'X').to_quaternion()
        knee.keyframe_insert("rotation_quaternion", frame=frame)
    return action


def human_rig():
    ref = templates.create_reference(ctx, templates.template_specs("HUMAN", options=None), name="reference")
    return generate.generate(ctx, ref)


def arm_angle(src, rig, prefix="mixamorig:"):
    sp = src.pose.bones
    a = (src.matrix_world @ sp[prefix + "LeftForeArm"].tail) - (src.matrix_world @ sp[prefix + "LeftArm"].head)
    # the wrist (hand head), not the forearm deform bone's tail: twist bones split the forearm
    b = (rig.matrix_world @ rig.pose.bones["DEF-hand.L"].head) - \
        (rig.matrix_world @ rig.pose.bones["DEF-upper_arm.L"].head)
    return math.degrees(a.angle(b))


def fcurve_frames(action, path):
    return sorted({round(k.co.x) for fc in utils.action_fcurves(action) if fc.data_path == path
                   for k in fc.keyframe_points})


# =============================================================================
# 1. Remap options
# =============================================================================
rig = human_rig()
src = mixamo_skeleton()
walk = key_walk(src)
walk.use_frame_range = True
walk.frame_start, walk.frame_end = 1, 10
walk.use_cyclic = True
remap = scene.rigger_remap
remap.source, remap.target = src, rig
remap.legs_ik = True
bpy.ops.rigger.remap_auto()
count = len(remap.items)
check(count > 10, f"auto map ({count} pairs)")

res = bpy.ops.rigger.remap_bake()
baked = rig.animation_data.action
check(res == {'FINISHED'} and baked.use_fake_user, "baked action has a fake user")
check(baked.use_frame_range and tuple(baked.frame_range) == (1.0, 10.0) and baked.use_cyclic,
      "frame range and cyclic copied")
scene.frame_set(10)
check(arm_angle(src, rig) < 5.0, f"arm follows ({arm_angle(src, rig):.2f} deg)")
torso_moving = rig.pose.bones["torso"].matrix.translation.copy()
scene.frame_set(1)
torso_start = rig.pose.bones["torso"].matrix.translation.copy()
travel = (torso_moving - torso_start).length
check(travel > 0.2, f"torso travels without In Place ({travel:.3f})")

# In place
remap.in_place = True
bpy.ops.rigger.remap_bake()
scene.frame_set(10)
moved = rig.pose.bones["torso"].matrix.translation.copy()
scene.frame_set(1)
drift = (Vector((moved.x, moved.y, 0)) - Vector((torso_start.x, torso_start.y, 0))).length
check(drift < 0.01, f"In Place removes the travel ({drift:.4f})")
remap.in_place = False

# Bake only existing keyframes
remap.existing_keys = True
bpy.ops.rigger.remap_bake()
frames = fcurve_frames(rig.animation_data.action, 'pose.bones["upper_arm_fk.L"].rotation_quaternion')
check(frames == [1, 10], f"only the source keyframes are baked ({frames})")
remap.existing_keys = False

# Pole modes and clean poles
pole_positions = {}
for mode in ('ABSOLUTE', 'TARGET', 'CHAIN'):
    remap.pole_mode = mode
    remap.smooth_poles = mode == 'TARGET'
    res = bpy.ops.rigger.remap_bake()
    scene.frame_set(10)
    leg = next(l for l in utils.get_meta(rig)["limbs"] if l["type"] == "leg" and l["side"] == "R")
    pole_positions[mode] = (rig.matrix_world @ rig.pose.bones[leg["pole"]].head).copy()
    sp = src.pose.bones
    src_knee = math.degrees((sp["mixamorig:RightLeg"].tail - sp["mixamorig:RightLeg"].head).angle(
        sp["mixamorig:RightUpLeg"].tail - sp["mixamorig:RightUpLeg"].head))
    pbs = rig.pose.bones
    tgt_knee = math.degrees((pbs["DEF-shin.R"].tail - pbs["DEF-shin.R"].head).angle(
        pbs["DEF-thigh.R"].tail - pbs["DEF-thigh.R"].head))
    frames = fcurve_frames(rig.animation_data.action, f'pose.bones["{leg["pole"]}"].location')
    check(res == {'FINISHED'} and len(frames) == 10, f"pole keyed with mode {mode}")
    check(abs(src_knee - tgt_knee) < 8.0, f"knee bend with pole mode {mode} ({src_knee:.1f} vs {tgt_knee:.1f})")
    # The knee points forward (-Y) like the pole
    knee_fwd = (rig.matrix_world @ pbs["DEF-shin.R"].head).y < (rig.matrix_world @ pbs["DEF-thigh.R"].head).y
    check(knee_fwd, f"knee points forward with pole mode {mode}")
check((pole_positions['TARGET'] - pole_positions['ABSOLUTE']).length > 1e-4, "pole modes differ")
remap.pole_mode = 'ABSOLUTE'
remap.smooth_poles = False
remap.legs_ik = False

# Clean FK rotation: forearm FK only bends on X
remap.clean_fk = True
bpy.ops.rigger.remap_bake()
scene.frame_set(10)
q = rig.pose.bones["forearm_fk.L"].rotation_quaternion
check(abs(q.y) < 1e-4 and abs(q.z) < 1e-4, f"clean FK keeps the hinge axis only ({q.y:.4f}, {q.z:.4f})")
remap.clean_fk = False

# Extract root motion
remap.root_motion = True
remap.legs_ik = True
bpy.ops.rigger.remap_bake()
root_frames = fcurve_frames(rig.animation_data.action, 'pose.bones["root"].location')
check(len(root_frames) >= 2, f"root motion extracted ({len(root_frames)} keys)")
remap.root_motion = False

# Rotation offset, source reference frame, current target pose
arm_pair = next(i for i in remap.items if i.source == "mixamorig:LeftArm")
arm_pair.rot_offset = (0.0, 0.0, math.radians(30))
bpy.ops.rigger.remap_bake()
scene.frame_set(1)
angle = math.degrees(rig.pose.bones["upper_arm_fk.L"].rotation_quaternion.angle)
check(abs(angle - 30.0) < 0.5, f"rotation offset on a pair ({angle:.2f} deg at the reference pose)")
arm_pair.rot_offset = (0.0, 0.0, 0.0)
remap.source_frame = 10
bpy.ops.rigger.remap_bake()
scene.frame_set(10)
angle = math.degrees(rig.pose.bones["upper_arm_fk.L"].rotation_quaternion.angle)
check(angle < 0.5, f"source reference frame: frame 10 is the reference pose ({angle:.2f} deg)")
remap.source_frame = -1
rig.animation_data.action = None
for pb in rig.pose.bones:
    pb.matrix_basis = Matrix()
posed = Matrix.Rotation(math.radians(35), 4, 'Y').to_quaternion()
rig.pose.bones["upper_arm_fk.L"].rotation_quaternion = posed
ctx.view_layer.update()
remap.use_target_pose = True
bpy.ops.rigger.remap_bake()
scene.frame_set(1)
diff = math.degrees(rig.pose.bones["upper_arm_fk.L"].rotation_quaternion.rotation_difference(posed).angle)
check(diff < 0.5, f"use current target pose as the reference ({diff:.2f} deg)")
remap.use_target_pose = False
for pb in rig.pose.bones:
    pb.matrix_basis = Matrix()

# Mirror the bone list
for i in reversed(range(len(remap.items))):
    if remap.items[i].source.startswith("mixamorig:Right"):
        remap.items.remove(i)
left = len(remap.items)
bpy.ops.rigger.remap_mirror()
pairs = {i.source: i.target for i in remap.items}
check(len(remap.items) > left and pairs.get("mixamorig:RightArm") == "upper_arm_fk.R",
      f"mirror bone list ({left} -> {len(remap.items)})")
check(retarget.mirror_any("LeftHand") == "RightHand" and retarget.mirror_any("hand_l") == "hand_r"
      and retarget.mirror_any("L_hand") == "R_hand", "mirror names with any convention")

# Select from the viewport
utils.set_mode(rig, 'POSE')
rig.data.bones.active = rig.data.bones["upper_arm_fk.R"]
bpy.ops.rigger.remap_select()
check(remap.items[remap.active_index].target == "upper_arm_fk.R", "select pair from the active bone")
bpy.ops.object.mode_set(mode='OBJECT')

# Local location + multiplier
neck = next(i for i in remap.items if i.source == "mixamorig:Neck")
neck.location, neck.location_mode, neck.loc_multiplier = True, 'LOCAL', 2.0
sp = src.pose.bones
sp["mixamorig:Neck"].location = (0, 0, 0)
sp["mixamorig:Neck"].keyframe_insert("location", frame=1)
sp["mixamorig:Neck"].location = (0, 0.05, 0)
sp["mixamorig:Neck"].keyframe_insert("location", frame=10)
bpy.ops.rigger.remap_bake()
scene.frame_set(10)
loc = rig.pose.bones["neck"].location
check(abs(loc.y - 0.1) < 1e-4, f"local location x multiplier ({loc.y:.4f})")
neck.location, neck.location_mode, neck.loc_multiplier = False, 'WORLD', 1.0

# Multiple source animations + filter
run = key_walk(src, name="run", travel=0.6, angle=30)
idle = bpy.data.actions.new("idle_pose")
utils.assign_action(src, walk)
remap.multiple = True
remap.action_filter = ""
bpy.ops.rigger.remap_actions_select(state=True)
idle.rigger_remap = True
before = set(bpy.data.actions.keys())
bpy.ops.rigger.remap_bake()
new = set(bpy.data.actions.keys()) - before
check(any(n.startswith("walk_remap") for n in new) and any(n.startswith("run_remap") for n in new)
      and not any(n.startswith("idle") for n in new), f"multiple source actions ({sorted(new)})")
check(src.animation_data.action == walk, "source action restored")
remap.action_filter = "run"
before = set(bpy.data.actions.keys())
bpy.ops.rigger.remap_bake()
new = set(bpy.data.actions.keys()) - before
check(len(new) == 1 and next(iter(new)).startswith("run_remap"), f"action filter ({sorted(new)})")
run.rigger_remap = False
remap.action_filter = ""
before = set(bpy.data.actions.keys())
bpy.ops.rigger.remap_bake()
new = set(bpy.data.actions.keys()) - before
check(len(new) == 1 and next(iter(new)).startswith("walk_remap"), f"unticked actions are skipped ({sorted(new)})")
remap.multiple = False
remap.action_filter = ""

# Namespace: the bone map is for 'mixamorig:', the skeleton uses 'mixamorig1:'
src2 = mixamo_skeleton("mixamo1", prefix="mixamorig1:", location=(3, 0, 0))
key_walk(src2, prefix="mixamorig1:")
remap.source = src2
remap.namespace_from, remap.namespace_to = "mixamorig:", "mixamorig1:"
res = bpy.ops.rigger.remap_bake()
scene.frame_set(10)
check(res == {'FINISHED'} and arm_angle(src2, rig, "mixamorig1:") < 5.0, "bake with a namespace replace")
bpy.ops.rigger.remap_namespace()
check(all(not i.source.startswith("mixamorig:") for i in remap.items)
      and any(i.source == "mixamorig1:LeftArm" for i in remap.items), "replace namespace in the list")

# Save / load presets with Clear off
path = os.path.join(bpy.app.tempdir or "/tmp", "rigger_map.json")
bpy.ops.rigger.remap_save(filepath=path)
first = remap.items[0].source
remap.items.remove(0)
extra = remap.items.add()
extra.source, extra.target = "custom", "head"
bpy.ops.rigger.remap_load(filepath=path, clear=False)
names = [i.source for i in remap.items]
check(first in names and "custom" in names, "load bone map without clearing")
bpy.ops.rigger.remap_load(filepath=path, clear=True)
check("custom" not in [i.source for i in remap.items], "load bone map with clearing")

# Bind / unbind preview
rig.animation_data.action = None
utils.assign_action(src2, src2.animation_data.action)
res = bpy.ops.rigger.remap_preview()
check(res == {'FINISHED'} and retarget.preview_active(), "bind preview")
scene.frame_set(10)
check(arm_angle(src2, rig, "mixamorig1:") < 5.0, "preview follows the source on frame change")
check(not rig.animation_data.action, "preview does not key")
bpy.ops.rigger.remap_preview()
check(not retarget.preview_active() and retarget._preview_handler not in bpy.app.handlers.frame_change_post,
      "unbind preview")
rot = rig.pose.bones["upper_arm_fk.L"].rotation_quaternion
check(abs(rot.w - 1.0) < 1e-5, "unbind resets the target pose")

# =============================================================================
# 2. Quick Rig
# =============================================================================


def skinned_mesh(armature, name):
    """A vertex on every bone (head, middle, tail) weighted to that bone; twist bones share the arm."""
    mesh = bpy.data.meshes.new(name)
    verts, groups = [], []
    for bone in armature.data.bones:
        if bone.name.endswith("_End") or bone.name == "Root":
            continue
        for t in (0.0, 0.5, 1.0):
            verts.append(armature.matrix_world @ bone.head_local.lerp(bone.tail_local, t))
            groups.append(bone.name)
    mesh.from_pydata([tuple(v) for v in verts], [], [])
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    for index, bone_name in enumerate(groups):
        vg = obj.vertex_groups.get(bone_name) or obj.vertex_groups.new(name=bone_name)
        if bone_name.endswith("LeftArmTwist"):
            vg.add([index], 0.5, 'REPLACE')
            arm = obj.vertex_groups.get(bone_name.replace("Twist", "")) or \
                obj.vertex_groups.new(name=bone_name.replace("Twist", ""))
            arm.add([index], 0.5, 'REPLACE')
        else:
            vg.add([index], 1.0, 'REPLACE')
    mod = obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = armature
    obj.parent = armature
    obj.matrix_parent_inverse = armature.matrix_world.inverted()
    return obj


def evaluated(obj):
    ctx.view_layer.update()
    eval_obj = obj.evaluated_get(ctx.evaluated_depsgraph_get())
    return [eval_obj.matrix_world @ v.co for v in eval_obj.data.vertices]


def bone_worlds(arm, names):
    return {n: (arm.matrix_world @ arm.pose.bones[n].head, arm.matrix_world @ arm.pose.bones[n].tail)
            for n in names}


for mode in ('PRESERVE', 'CONVERT'):
    qs = scene.rigger_quick
    qs.limbs.clear()
    qs.overrides.clear()
    skel = mixamo_skeleton(f"fbx_{mode}", scale=0.01, location=(-2, 0, 0), extras=True, fingers=True)
    mesh = skinned_mesh(skel, f"body_{mode}")
    action = key_walk(skel, name=f"take_{mode}")
    scene.frame_set(10)
    watched = ["mixamorig:LeftHand", "mixamorig:LeftForeArm", "mixamorig:RightFoot", "mixamorig:Head",
               "mixamorig:LeftHandIndex3"]
    expected_bones = bone_worlds(skel, watched)
    expected_mesh = evaluated(mesh)
    scene.frame_set(1)
    rest_mesh = evaluated(mesh)
    qs.armature = skel
    qs.mode = mode
    qs.fingers_ik = mode == 'CONVERT'
    bpy.ops.rigger.quick_rig_guess()
    kinds = sorted(l.kind for l in qs.limbs)
    check(kinds.count('ARM') == 2 and kinds.count('LEG') == 2 and kinds.count('FINGER') == 10
          and 'SPINE' in kinds and 'NECK' in kinds, f"{mode}: guess limbs ({len(kinds)})")
    spine = next(l for l in qs.limbs if l.kind == 'SPINE')
    check(quick_rig.bone_list(spine)[0] == "mixamorig:Hips" and len(quick_rig.bone_list(spine)) == 4,
          f"{mode}: spine starts at the hips")
    if mode == 'CONVERT':
        o = qs.overrides.add()
        o.source, o.target = "mixamorig:LeftArmTwist", "forearm.L"
        qs.orphans = True
    res = bpy.ops.rigger.quick_rig()
    check(res == {'FINISHED'}, f"{mode}: quick rig")
    rig_q = next(o for o in bpy.data.objects if o.get("rigger_quick_source") == f"fbx_{mode}")
    meta = utils.get_meta(rig_q)
    check(len(meta["limbs"]) == 4 and len(meta.get("fingers", [])) == 10, f"{mode}: rig limbs generated")
    check("Tail" in meta["deform_of"] and "HeadTop_End" not in meta["deform_of"],
          f"{mode}: orphan bone added, end bone skipped ({sorted(k for k in meta['deform_of'] if '.' not in k)})")
    check("Root" not in meta["deform_of"], f"{mode}: root bone ignored")
    check((rig_q.matrix_world.translation - Vector((-2, 0, 0))).length < 1e-5
          and abs(rig_q.matrix_world.to_scale().x - 1.0) < 1e-5, f"{mode}: same origin, scale 1")
    ref_q = rig_q.rigger.reference
    ref_head = ref_q.matrix_world @ ref_q.data.bones["forearm.L"].head_local
    src_head = Vector((-2, 0, 0)) + Vector(templates.DEFAULT_POINTS["shoulder"]).lerp(
        templates.DEFAULT_POINTS["wrist"], 0.5) * 0.9
    check(abs(ref_head.x - src_head.x) < 0.01 and abs(ref_head.z - src_head.z) < 0.01,
          f"{mode}: reference bones on the skeleton")
    baked_action = rig_q.animation_data.action if rig_q.animation_data else None
    check(baked_action is not None and baked_action.name.startswith(f"take_{mode}_rig"),
          f"{mode}: animation baked to the control rig")
    scene.frame_set(1)
    rest_err = max((a - b).length for a, b in zip(rest_mesh, evaluated(mesh)))
    check(rest_err < 1e-3, f"{mode}: mesh unchanged at frame 1 ({rest_err:.5f})")
    scene.frame_set(10)
    twist = {i for i, v in enumerate(mesh.data.vertices) if len(v.groups) == 2}
    err = max((a - b).length for i, (a, b) in enumerate(zip(expected_mesh, evaluated(mesh))) if i not in twist)
    check(err < 0.02, f"{mode}: baked animation deforms the mesh like before ({err:.4f})")
    if mode == 'PRESERVE':
        check(skel.animation_data.action is None and action.use_fake_user, "skeleton action unassigned, kept")
        got = bone_worlds(skel, watched)
        err = max(max((expected_bones[n][0] - got[n][0]).length, (expected_bones[n][1] - got[n][1]).length)
                  for n in watched)
        check(err < 0.02, f"skeleton follows the baked rig ({err:.4f})")
        con = skel.pose.bones["mixamorig:LeftArm"].constraints
        check([c.type for c in con] == ['COPY_LOCATION', 'COPY_ROTATION', 'COPY_SCALE'],
              "skeleton bones constrained to the rig")
        check(all(rig_q.data.bones.get("QR-" + n) for n in watched) and
              not rig_q.data.collections[quick_rig.QR_COLLECTION].is_visible, "hidden helper bones")
        check(skel.hide_get(), "base armature hidden")
        # Posing a control moves the skeleton
        rig_q.animation_data.action = None
        for pb in rig_q.pose.bones:
            pb.matrix_basis = Matrix()
        ctx.view_layer.update()
        pb = rig_q.pose.bones["forearm_fk.L"]
        pb.rotation_quaternion = Matrix.Rotation(math.radians(50), 4, 'X').to_quaternion()
        ctx.view_layer.update()
        hand = skel.matrix_world @ skel.pose.bones["mixamorig:LeftHand"].head
        def_hand = rig_q.matrix_world @ rig_q.pose.bones["DEF-hand.L"].head
        check((hand - def_hand).length < 1e-3, f"skeleton follows a posed control ({(hand - def_hand).length:.5f})")
        scale = skel.pose.bones["mixamorig:LeftHand"].matrix.to_scale()
        check(abs(scale.x - 1.0) < 1e-3, f"skeleton bones keep their scale ({scale.x:.4f})")
        # Regenerating keeps the helper bones
        generate.generate(ctx, ref_q)
        check(rig_q.data.bones.get("QR-mixamorig:LeftHand") is not None, "helper bones survive regeneration")
        pb = rig_q.pose.bones["forearm_fk.L"]
        pb.rotation_quaternion = Matrix.Rotation(math.radians(50), 4, 'X').to_quaternion()
        ctx.view_layer.update()
        hand = skel.matrix_world @ skel.pose.bones["mixamorig:LeftHand"].head
        def_hand = rig_q.matrix_world @ rig_q.pose.bones["DEF-hand.L"].head
        check((hand - def_hand).length < 1e-3, "skeleton still follows after regeneration")
    else:
        check(f"fbx_{mode}" not in bpy.data.objects, "Convert deletes the source armature")
        groups = {vg.name for vg in mesh.vertex_groups}
        check("DEF-upper_arm.L" in groups and not any(g.startswith("mixamorig:") for g in groups),
              "vertex groups renamed to deform bones")
        check("DEF-Tail" in groups and "DEF-LeftArmTwist" not in groups, "orphan and override weights")
        fore = mesh.vertex_groups["DEF-forearm.L"].index
        merged = [v for v in mesh.data.vertices if len([g for g in v.groups if g.group == fore]) and
                  len(v.groups) == 2]
        check(len(merged) == 3, f"override merged the twist weights into the forearm ({len(merged)})")
        check(mesh.parent == rig_q and mesh.modifiers["Armature"].object == rig_q, "mesh moved to the rig")
        fingers = [f for f in meta.get("fingers", []) if f.get("ik_ctrl")]
        check(len(fingers) == 10, "finger IK option")

# Presets
qs = scene.rigger_quick
path = os.path.join(bpy.app.tempdir or "/tmp", "rigger_quick.json")
qs.mode = 'CONVERT'
bpy.ops.rigger.quick_rig_preset_export(filepath=path)
saved = len(qs.limbs)
qs.limbs.clear()
qs.overrides.clear()
qs.mode = 'PRESERVE'
bpy.ops.rigger.quick_rig_preset_import(filepath=path)
check(len(qs.limbs) == saved and qs.mode == 'CONVERT' and len(qs.overrides) == 1, "Quick Rig presets")

# Manual limb from selected bones
skel = mixamo_skeleton("manual", location=(4, 0, 0), extras=True)
qs.armature = skel
qs.limbs.clear()
bpy.ops.rigger.quick_rig_limb_add()
qs.limbs[0].kind = 'CHAIN'
qs.limbs[0].label = "tail"
utils.set_mode(skel, 'POSE')
for pb in skel.pose.bones:
    utils.select_pose_bone(pb, pb.name == "mixamorig:Tail")
bpy.ops.rigger.quick_rig_from_selection()
check(qs.limbs[0].bones == "mixamorig:Tail", "limb from selected bones")
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.rigger.quick_rig_limb_remove()
check(len(qs.limbs) == 0, "remove limb")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("\nFAILURES: none")
