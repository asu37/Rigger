"""Rig tools: pose blends (fist, extend, blink, wings fold), set pose, mirror pose, Child Of switcher,
root motion, rotation conversion, snaps and bakes, character name, colour theme, shapes, custom
bones and limbs, rig data, presets, export tags, shape key drivers, rig layers, rest pose with
animation, limb duplication, turn model, IK lines, finger detection and init scale.

    blender -b --factory-startup --python tests/test_rig_tools.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, markers, poses, reference, rig_data, rig_tools, templates, tools, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []
TMP = tempfile.mkdtemp()


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def update(obj):
    obj.update_tag()
    ctx.view_layer.update()


def world(obj, name, tail=False):
    pb = obj.pose.bones[name]
    return obj.matrix_world @ (pb.tail if tail else pb.head)


def q(angle, axis):
    return Matrix.Rotation(math.radians(angle), 4, axis).to_quaternion()


def select_only(rig, names):
    for pb in rig.pose.bones:
        utils.select_pose_bone(pb, pb.name in names)
    if names:
        rig.data.bones.active = rig.data.bones[names[0]]


def current_rig():
    return ref.rigger.rig


ref = templates.create_reference(ctx, templates.template_specs('HUMAN', options=None), name="ref")
ref.pose.bones["hips"].rigger.reversed_spine = True
ref.pose.bones["index.01.L"].rigger.finger_ik = True
rig = generate.generate(ctx, ref)
ref.hide_set(True)
utils.set_mode(rig, 'POSE')

# -- Fist and extended hand ------------------------------------------------------------------
tip_rest = world(rig, "DEF-index.03.L", tail=True)
bpy.ops.rigger.finger_pose(side='L', pose='FIST')
select_only(rig, ["hand_ik.L"])
bpy.ops.rigger.add_pose_blend(kind='FIST', side='L')
rig = current_rig()
check("hand_fist.L" in rig.pose.bones, "fist control created")
check(all(rig.pose.bones[n].matrix_basis == Matrix() for n in ("index.01.L", "index_master.L")),
      "the rig is back in rest pose after storing the fist")
fist = rig.pose.bones["hand_fist.L"]
fist.scale = (1, 0.5, 1)
update(rig)
check(world(rig, "DEF-index.03.L", tail=True).z < tip_rest.z - 0.02, "scaling the fist control down closes the hand")
fist.scale = (1, 1, 1)
update(rig)
check((world(rig, "DEF-index.03.L", tail=True) - tip_rest).length < 1e-4, "fist control at 1 = rest pose")
rig.pose.bones["index.01.L"].rotation_quaternion = q(25, 'Z')
rig.pose.bones["pinky.01.L"].rotation_quaternion = q(-25, 'Z')
bpy.ops.rigger.add_pose_blend(kind='EXTEND', side='L')
rig = current_rig()
rig.pose.bones["hand_fist.L"].scale = (1, 1.5, 1)
update(rig)
check((world(rig, "DEF-index.03.L", tail=True) - tip_rest).length > 0.01, "scaling it up blends the extended pose")
rig.pose.bones["hand_fist.L"].scale = (1, 1, 1)
utils.set_mode(rig, 'POSE')
bpy.ops.rigger.mirror_pose_blend(name="fist.L")
rig = current_rig()
right_rest = world(rig, "DEF-index.03.R", tail=True)
rig.pose.bones["hand_fist.R"].scale = (1, 0.5, 1)
update(rig)
check(world(rig, "DEF-index.03.R", tail=True).z < right_rest.z - 0.02, "mirrored fist works on the right hand")
rig.pose.bones["hand_fist.R"].scale = (1, 1, 1)
update(rig)

# -- Blink pose --------------------------------------------------------------------------------
utils.set_mode(rig, 'POSE')
face = utils.get_meta(rig)["faces"][0]["face"]
gap = lambda r: (world(r, "DEF-lid_upper.L") - world(r, "DEF-lid_lower.L")).length  # noqa: E731
open_gap = gap(rig)
select_only(rig, ["lid_upper.L"])
bpy.ops.rigger.default_blink_pose(side='L')
check(gap(rig) < 0.3 * open_gap, "default blink pose puts the lid controls in the closed pose")
bpy.ops.rigger.add_pose_blend(kind='BLINK', side='L', amount=1.0)
rig = current_rig()
check(gap(rig) > 0.9 * open_gap, "lids open after storing the blink pose")
fc = rig.animation_data.drivers.find('pose.bones["MCH-lid_upper_blink.L"].rotation_euler', index=0)
check(fc is None, "the stored blink pose replaces the default blink rotation")
rig.pose.bones[face]["blink_L"] = 1.0
update(rig)
check(gap(rig) < 0.3 * open_gap, f"blink slider blends the stored blink pose ({gap(rig):.4f})")
rig.pose.bones[face]["blink_L"] = 0.0
update(rig)
utils.set_mode(rig, 'POSE')
rig.pose.bones["lid_upper.L"].location = (0, 0, -0.002)
bpy.ops.rigger.add_pose_blend(kind='BLINK', side='L', amount=0.5)
rig = current_rig()
blend = next(b for b in poses.load_blends(ref) if b["name"] == "blink.L")
check(sorted(blend["keys"]) == ["10", "5"], f"in-between blink pose stored ({sorted(blend['keys'])})")
utils.set_mode(rig, 'POSE')
bpy.ops.rigger.remove_pose_blend(name="blink.L", inbetweens_only=True)
rig = current_rig()
blend = next(b for b in poses.load_blends(ref) if b["name"] == "blink.L")
check(list(blend["keys"]) == ["10"], "in-between poses removed")
utils.set_mode(rig, 'POSE')
bpy.ops.rigger.remove_pose_blend(name="blink.L")
rig = current_rig()
check(rig.animation_data.drivers.find('pose.bones["MCH-lid_upper_blink.L"].rotation_euler', index=0) is not None,
      "removing the blink pose brings back the default blink")
pbs = rig.pose.bones

# -- Set pose and mirror pose ----------------------------------------------------------------------
utils.set_mode(rig, 'POSE')
bpy.ops.rigger.set_pose(pose='A')
d = (world(rig, "DEF-upper_arm.L", tail=True) - world(rig, "DEF-upper_arm.L")).normalized()
check(abs(math.degrees(math.asin(-d.z)) - 45) < 2, f"A-pose: arms 45 degrees down ({math.degrees(math.asin(-d.z)):.1f})")
bpy.ops.rigger.set_pose(pose='T')
d = (world(rig, "DEF-forearm.L", tail=True) - world(rig, "DEF-upper_arm.L")).normalized()
check(abs(d.z) < 0.02 and d.x > 0.99, "T-pose: arms straight out")
bpy.ops.rigger.reset_pose()
pbs["upper_arm_fk.L"].rotation_quaternion = q(30, 'X') @ q(20, 'Z')
pbs["hand_ik.L"]["ik_fk"] = 0.0
pbs["hand_ik.R"]["ik_fk"] = 0.0
update(rig)
select_only(rig, ["upper_arm_fk.L"])
bpy.ops.rigger.mirror_pose()
update(rig)
left, right = world(rig, "DEF-forearm.L"), world(rig, "DEF-forearm.R")
check((left - Vector((-right.x, right.y, right.z))).length < 1e-3, "mirror pose copies the pose to the other side")
bpy.ops.rigger.reset_pose()

# -- Child Of switcher --------------------------------------------------------------------------------
targets = []
for i, loc in enumerate(((0.5, -0.5, 1.2), (1.0, 0.5, 1.0))):
    empty = bpy.data.objects.new(f"target_{i}", None)
    ctx.collection.objects.link(empty)
    empty.location = loc
    targets.append(empty)
update(rig)
hand = pbs["hand_ik.L"]
for empty in targets:
    con = hand.constraints.new('CHILD_OF')
    con.target = empty
    con.inverse_matrix = empty.matrix_world.inverted()
hand.constraints[-1].influence = 0.0
update(rig)
before = hand.matrix.copy()
select_only(rig, ["hand_ik.L"])
bpy.ops.rigger.snap_child_of(index=1)
check((hand.matrix.translation - before.translation).length < 1e-4, "Snap Child Of keeps the control in place")
targets[1].location.z += 0.2
update(rig)
check(hand.matrix.translation.z > before.translation.z + 0.1, "the control now follows the second target")
for con in list(hand.constraints):
    if con.type == 'CHILD_OF':
        hand.constraints.remove(con)
for empty in targets:
    bpy.data.objects.remove(empty)
bpy.ops.rigger.reset_pose()

# -- Root motion -------------------------------------------------------------------------------------
scene = ctx.scene
scene.frame_start, scene.frame_end = 1, 10
torso = pbs[utils.get_meta(rig)["spine"]["torso"]]
rig.animation_data_create()
rig.animation_data.action = bpy.data.actions.new("walk")
for f, y in ((1, 0.0), (10, -1.0)):
    torso.location = (0, 0, y)  # the torso's local Z is world -Y (forward)
    torso.keyframe_insert("location", frame=f)
    pbs["foot_ik.L"].location = (0, 0, 0)
    pbs["foot_ik.L"].keyframe_insert("location", frame=f)
scene.frame_set(10)
hips_before = world(rig, "DEF-hips").copy()
foot_before = world(rig, "DEF-foot.L").copy()
bpy.ops.rigger.extract_root_motion(frame_start=1, frame_end=10)
scene.frame_set(10)
root_moved = (rig.matrix_world @ pbs["root"].head - rig.matrix_world @ rig.data.bones["root"].head_local)
check(abs(root_moved.length - abs(hips_before.y - rig.data.bones["DEF-hips"].head_local.y)) < 0.02,
      f"root follows the pelvis over the ground ({root_moved.length:.3f})")
check((world(rig, "DEF-hips") - hips_before).length < 1e-3 and (world(rig, "DEF-foot.L") - foot_before).length < 1e-3,
      "the animation stays where it was")
rig.animation_data.action = None
bpy.ops.rigger.reset_pose()
scene.frame_set(1)

# -- Rotation mode conversion -----------------------------------------------------------------------------
act = bpy.data.actions.new("conv")
rig.animation_data.action = act
fk = pbs["forearm_fk.L"]
for f, a in ((1, 0), (5, 50)):
    fk.rotation_quaternion = q(a, 'X')
    fk.keyframe_insert("rotation_quaternion", frame=f)
scene.frame_set(5)
before = world(rig, "DEF-hand.L").copy()
select_only(rig, ["forearm_fk.L"])
bpy.ops.rigger.convert_rotation(mode='XYZ')
scene.frame_set(1)
scene.frame_set(5)
paths = {fc.data_path for fc in utils.action_fcurves(act)}
check(fk.rotation_mode == 'XYZ' and 'pose.bones["forearm_fk.L"].rotation_euler' in paths
      and 'pose.bones["forearm_fk.L"].rotation_quaternion' not in paths, "rotation keys converted to Euler")
check((world(rig, "DEF-hand.L") - before).length < 1e-4, "converted animation looks the same")
rig.animation_data.action = None
fk.rotation_mode = 'QUATERNION'
bpy.ops.rigger.reset_pose()

# -- Snaps: head lock, IK/FK bake, reversed spine, fingers ------------------------------------------------------
head = utils.get_meta(rig)["neck"]["head"]
pbs["neck"].rotation_quaternion = q(30, 'X')
update(rig)
head_before = pbs[head].matrix.copy()
bpy.ops.rigger.snap_property(bone=head, prop="head_isolate", value=1.0)
check(pbs[head]["head_isolate"] == 1.0 and (pbs[head].matrix.to_quaternion().rotation_difference(
    head_before.to_quaternion()).angle < 1e-4), "snap head lock keeps the head rotation")
bpy.ops.rigger.reset_pose()
pbs[head]["head_isolate"] = 0.0

act = bpy.data.actions.new("arm_fk")
rig.animation_data.action = act
pbs["hand_ik.L"]["ik_fk"] = 0.0
for f, a in ((1, 0), (5, 60)):
    pbs["upper_arm_fk.L"].rotation_quaternion = q(a, 'X')
    pbs["upper_arm_fk.L"].keyframe_insert("rotation_quaternion", frame=f)
scene.frame_set(3)
fk_hand = world(rig, "DEF-hand.L").copy()
limbs = utils.get_meta(rig)["limbs"]
arm_index = next(i for i, l in enumerate(limbs) if l["type"] == "arm" and l["side"] == "L")
bpy.ops.rigger.bake_ikfk(limb=arm_index, to_ik=True, frame_start=1, frame_end=5)
paths = {fc.data_path for fc in utils.action_fcurves(act)}
scene.frame_set(3)
pbs["hand_ik.L"]["ik_fk"] = 1.0
update(rig)
check('pose.bones["hand_ik.L"].location' in paths and (world(rig, "DEF-hand.L") - fk_hand).length < 2e-3,
      f"bake FK > IK keys the IK hand on every frame ({(world(rig, 'DEF-hand.L') - fk_hand).length:.4f})")
rig.animation_data.action = None
bpy.ops.rigger.reset_pose()
pbs["hand_ik.L"]["ik_fk"] = 0.0

pbs["spine.02"].rotation_quaternion = q(20, 'X')
update(rig)
spine_before = world(rig, "DEF-spine.03", tail=True).copy()
bpy.ops.rigger.snap_reverse_spine()
update(rig)
check(pbs[utils.get_meta(rig)["spine"]["torso"]]["spine_reverse"] == 1.0
      and (world(rig, "DEF-spine.03", tail=True) - spine_before).length < 1e-3, "snap to the reversed spine keeps the shape")
bpy.ops.rigger.snap_reverse_spine()
update(rig)
check((world(rig, "DEF-spine.03", tail=True) - spine_before).length < 1e-3, "and back to the forward spine")
bpy.ops.rigger.reset_pose()
pbs[utils.get_meta(rig)["spine"]["torso"]]["spine_reverse"] = 0.0

pbs["index.02.L"].rotation_quaternion = q(-40, 'X')
update(rig)
finger_before = world(rig, "DEF-index.03.L", tail=True).copy()
bpy.ops.rigger.snap_fingers(finger="index_ik.L", to_ik=True)
check(pbs["index_ik.L"]["ik_fk"] == 1.0 and (world(rig, "DEF-index.03.L", tail=True) - finger_before).length < 2e-3,
      "snap finger to IK keeps its pose")
bpy.ops.rigger.snap_fingers(finger="index_ik.L", to_ik=False)
check((world(rig, "DEF-index.03.L", tail=True) - finger_before).length < 2e-3, "and back to FK")
bpy.ops.rigger.toggle_finger_parents(side='L')
check(pbs["index_ik.L"]["parent"] == 1, "finger IK parents toggled")
bpy.ops.rigger.reset_pose()

# -- Custom bones survive regeneration; custom limbs ---------------------------------------------------------------
utils.set_mode(rig, 'EDIT')
eb = rig.data.edit_bones.new("cc_hat")
eb.head, eb.tail = Vector((0, 0, 1.78)), Vector((0, 0, 1.88))
eb.parent = rig.data.edit_bones["DEF-head"]
bpy.ops.object.mode_set(mode='POSE')
hat = rig.pose.bones["cc_hat"]
hat["wobble"] = 0.25
con = hat.constraints.new('COPY_ROTATION')
con.target, con.subtarget, con.influence = rig, "neck", 0.3
utils.set_mode(rig, 'OBJECT')
rig = generate.generate(ctx, ref)
hat = rig.pose.bones.get("cc_hat")
check(hat is not None and hat.parent.name == "DEF-head" and hat.get("wobble") == 0.25
      and hat.constraints and hat.constraints[0].subtarget == "neck" and abs(hat.constraints[0].influence - 0.3) < 1e-6,
      "custom bone kept with parent, property and constraint after regenerating")
pbs = rig.pose.bones
utils.set_mode(rig, 'POSE')
select_only(rig, ["cc_hat"])
limb_dir = rig_data.library_dir("limbs")
bpy.ops.rigger.save_custom_limb(name="test_hat")
check(os.path.exists(os.path.join(limb_dir, "test_hat.json")), "custom limb saved")
bpy.ops.rigger.add_custom_limb(limb="test_hat")
check("cc_hat_2" in rig.pose.bones and rig.pose.bones["cc_hat_2"].constraints, "custom limb added (renamed)")
bpy.ops.rigger.remove_custom_limb(limb="test_hat")
check(not os.path.exists(os.path.join(limb_dir, "test_hat.json")), "custom limb removed from the library")

# -- Export tags ----------------------------------------------------------------------------------------------
select_only(rig, ["DEF-jaw"])
bpy.ops.rigger.tag_bones(tag='EXCLUDE')
tags = rig_data.export_tags(rig)
check(tags.get("cc_hat") == 'CUSTOM' and tags.get("DEF-jaw") == 'EXCLUDE', f"export tags ({tags})")

# -- Rig data, presets ------------------------------------------------------------------------------------------
path = os.path.join(TMP, "rig.json")
utils.set_mode(rig, 'OBJECT')
ctx.view_layer.objects.active = rig
bpy.ops.rigger.export_rig_data(filepath=path)
data = json.load(open(path))
check({"reference", "shapes", "layers", "custom"} <= set(data) and data["custom"], "rig data exported")
ref.hide_set(False)
utils.set_mode(ref, 'EDIT')
ref.data.edit_bones["hand.L"].tail.z += 0.1
bpy.ops.object.mode_set(mode='OBJECT')
saved_twist = ref.pose.bones["shoulder.L"].rigger.twist
ref.pose.bones["shoulder.L"].rigger.twist = saved_twist + 3
rig_data.import_rig_data(ctx, ref, data)
rig = current_rig()
check(abs(ref.data.bones["hand.L"].tail_local.z - data["reference"]["bones"][[b["name"] for b in data["reference"]
      ["bones"]].index("hand.L")]["tail"][2]) < 1e-5 and ref.pose.bones["shoulder.L"].rigger.twist == saved_twist,
      "rig data import restores reference bones and settings")
utils.set_mode(rig, 'OBJECT')
ref.hide_set(False)
ctx.view_layer.objects.active = ref
bpy.ops.rigger.save_preset(name="test_preset")
bpy.ops.rigger.add_preset(preset="test_preset")
new_ref = ctx.object
check(utils.is_reference(new_ref) and new_ref != ref and set(new_ref.data.bones.keys()) == set(ref.data.bones.keys())
      and new_ref.pose.bones["shoulder.L"].rigger.module == 'ARM', "armature preset adds an identical reference")
bpy.ops.rigger.remove_preset(preset="test_preset")
bpy.data.objects.remove(new_ref)
ref.hide_set(True)

# -- Colour theme, shapes, character name -------------------------------------------------------------------------
ref.rigger.color_left = (0.0, 1.0, 0.0)
ctx.view_layer.objects.active = rig
bpy.ops.rigger.color_theme()
check(tuple(round(c, 3) for c in rig.pose.bones["hand_ik.L"].color.custom.normal) == (0.0, 1.0, 0.0),
      "colour theme assigned")
rig = generate.generate(ctx, ref)
check(tuple(round(c, 3) for c in rig.pose.bones["hand_ik.L"].color.custom.normal) == (0.0, 1.0, 0.0),
      "colour theme kept when regenerating")
utils.set_mode(rig, 'POSE')
select_only(rig, ["chest"])
bpy.ops.rigger.set_shape(shape='CUBE', scale=0.5)
check(rig.pose.bones["chest"].custom_shape.name.startswith("WGT-rigger_cube"), "set custom shape from the library")
pb = rig.pose.bones["chest"]
pb.custom_shape_translation = (0, 0.05, 0.02)
pb.custom_shape_rotation_euler = (0.3, 0, 0)
before = [tuple(tools._shape_space(rig, pb) @ v.co) for v in pb.custom_shape.data.vertices]
bpy.ops.rigger.apply_shape_transforms()
after = [tuple(tools._shape_space(rig, pb) @ v.co) for v in pb.custom_shape.data.vertices]
check(tuple(pb.custom_shape_translation) == (0, 0, 0) and
      max((Vector(a) - Vector(b)).length for a, b in zip(before, after)) < 1e-5,
      "apply shape transforms keeps the shape where it was")
shapes_path = os.path.join(TMP, "shapes.json")
bpy.ops.rigger.export_shapes(filepath=shapes_path)
pb.custom_shape = None
bpy.ops.rigger.import_shapes(filepath=shapes_path)
check(rig.pose.bones["chest"].custom_shape is not None, "shapes export / import")
utils.set_mode(rig, 'OBJECT')
bpy.ops.rigger.set_character_name(name="matt")
check(rig.name == "matt" and {"matt", "matt_rig", "matt_cs"} <= set(bpy.data.collections.keys())
      and rig.name in bpy.data.collections["matt_rig"].objects and ref.name == "matt_reference"
      and any(o.name.startswith("WGT-matt_") for o in bpy.data.collections["matt_cs"].objects),
      "set character name renames the rig and organises collections")

# -- Shape key drivers -----------------------------------------------------------------------------------------------
bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0, -0.1, 1.6))
mesh = ctx.object
mesh.modifiers.new("Armature", 'ARMATURE').object = rig
mesh.shape_key_add(name="Basis")
smile = mesh.shape_key_add(name="smile_L")
smile.data[0].co.x += 0.05
mesh.active_shape_key_index = 1
bpy.ops.rigger.shape_key_driver(bone="lip_corner.L", channel='LOC_X', factor=2.0)
rig.pose.bones["lip_corner.L"].location = (0.1, 0, 0)
update(rig)
mesh.data.shape_keys.update_tag()
ctx.view_layer.update()
check(abs(smile.value - 0.2) < 1e-4, f"shape key driven by the bone ({smile.value:.3f})")
rig.pose.bones["lip_corner.L"].location = (0.05, 0, 0)
update(rig)
bpy.ops.rigger.shape_key_driver_key(value='1')
rig.pose.bones["lip_corner.L"].location = (0.0, 0, 0)
update(rig)
bpy.ops.rigger.shape_key_driver_key(value='0')
rig.pose.bones["lip_corner.L"].location = (0.05, 0, 0)
update(rig)
ctx.view_layer.update()
check(abs(smile.value - 1.0) < 1e-3, f"driver key 1 at the current pose ({smile.value:.3f})")
bpy.ops.rigger.mirror_shape_keys()
right = mesh.data.shape_keys.key_blocks.get("smile_R")
fc_r = mesh.data.shape_keys.animation_data.drivers.find('key_blocks["smile_R"].value')
check(right is not None and right.data[0].co.x < right.data[1].co.x + 1 and fc_r is not None
      and fc_r.driver.variables[0].targets[0].bone_target == "lip_corner.R", "mirror shape keys with drivers")
mesh.active_shape_key_index = 1
bpy.ops.rigger.remove_shape_driver()
check(mesh.data.shape_keys.animation_data.drivers.find('key_blocks["smile_L"].value') is None,
      "shape key driver removed")
rig.pose.bones["lip_corner.L"].location = (0, 0, 0)

# -- Rig layers: animated, select, export ------------------------------------------------------------------------------
layer = rig.rigger.layers.add()
layer.name = "Props"
layer.objects.add().obj = mesh
rig.rigger.animated_layers = True
layer.show = True
rig.keyframe_insert('rigger.layers[0].show', frame=1)
layer.show = False
rig.keyframe_insert('rigger.layers[0].show', frame=5)
scene.frame_set(1)
visible1 = not mesh.hide_viewport
scene.frame_set(5)
check(visible1 and mesh.hide_viewport, "animated rig layers hide objects during playback")
rig.animation_data.action = None
layer.show = True
lpath = os.path.join(TMP, "layers.json")
ctx.view_layer.objects.active = rig
bpy.ops.rigger.layers_export(filepath=lpath)
rig.rigger.layers.clear()
bpy.ops.rigger.layers_import(filepath=lpath)
check(len(rig.rigger.layers) == 1 and rig.rigger.layers[0].objects[0].obj == mesh, "rig layers export / import")
utils.set_mode(rig, 'POSE')
rig.rigger.layer_index = 0
select_only(rig, ["chest"])
bpy.ops.rigger.layer_assign()
select_only(rig, [])
bpy.ops.rigger.layer_select()
check(utils.is_pose_bone_selected(rig.pose.bones["chest"]), "select the bones of a rig layer")
bpy.data.objects.remove(mesh)
rig.rigger.layers.clear()

# -- Apply rest pose, keeping the animation ------------------------------------------------------------------------------
utils.set_mode(rig, 'POSE')
bpy.ops.rigger.reset_pose()
anim = bpy.data.actions.new("wave")
rig.animation_data.action = anim
pbs = rig.pose.bones
pbs["hand_ik.L"]["ik_fk"] = 0.0
for f, a in ((1, 0), (10, 70)):
    pbs["forearm_fk.L"].rotation_quaternion = q(a, 'X')
    pbs["forearm_fk.L"].keyframe_insert("rotation_quaternion", frame=f)
scene.frame_set(10)
hand_10 = world(rig, "DEF-hand.L").copy()
scene.frame_set(1)
rig.animation_data.action = None
pbs["upper_arm_fk.L"].rotation_quaternion = q(-30, 'Y')
update(rig)
bpy.ops.rigger.apply_rest_pose(preserve_anim=True)
rig = current_rig()
utils.assign_action(rig, anim)
scene.frame_set(10)
check((world(rig, "DEF-hand.L") - hand_10).length < 2e-3,
      f"apply pose as rest pose keeps the animation ({(world(rig, 'DEF-hand.L') - hand_10).length:.4f})")
rig.animation_data.action = None
scene.frame_set(1)

# -- Reference tools: duplicate, remove limb, IK lines, turn model -----------------------------------------------------
ref.hide_set(False)
utils.set_mode(ref, 'EDIT')
ref.data.edit_bones.active = ref.data.edit_bones["tail_test"] if "tail_test" in ref.data.edit_bones else \
    ref.data.edit_bones["upper_arm.L"]
bpy.ops.rigger.duplicate_limb(offset=(0, 0.3, 0))
dup = [n for n in ref.data.edit_bones.keys() if n.startswith("shoulder_2")]
check(dup and ref.pose.bones[dup[0]].rigger.module == 'ARM', f"duplicate limb ({dup})")
ref.data.edit_bones.active = ref.data.edit_bones[dup[0]]
bpy.ops.rigger.remove_limb()
check(not any(n.startswith(("shoulder_2", "upper_arm_2")) for n in ref.data.edit_bones.keys()), "remove limb")
lines = reference.ik_lines(ref)
check(len(lines) == 4, f"IK direction lines for 2 arms and 2 legs ({len(lines)})")
bpy.ops.object.mode_set(mode='OBJECT')
ref.hide_set(True)
bpy.ops.mesh.primitive_cube_add(location=(2, 0, 0))
cube = ctx.object
bpy.ops.rigger.turn_model(angle=90)
check(abs(cube.rotation_euler.z - math.radians(90)) < 1e-4, "turn model rotates by 90 degrees")
bpy.data.objects.remove(cube)

# -- Finger detection -------------------------------------------------------------------------------------------------
specs = {s.name: s for s in templates.mirrored(templates.human_specs(face=False))}
parts = []
stretch = 1.2
for name, spec in specs.items():
    if not name.endswith(".L") or not any(name.startswith(f) for f in ("index", "middle", "ring", "pinky", "hand")):
        continue
    head, tail = spec.head.copy(), spec.tail.copy()
    if not name.startswith("hand"):
        base = specs[name.split(".")[0] + ".01.L"].head
        head, tail = base + (head - base) * stretch, base + (tail - base) * stretch
    d = tail - head
    radius = 0.022 if name.startswith("hand") else 0.006
    bpy.ops.mesh.primitive_cylinder_add(vertices=10, radius=radius, depth=d.length * 1.02, location=(head + tail) / 2)
    o = ctx.object
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    parts.append(o)
bpy.ops.object.select_all(action='DESELECT')
for o in parts:
    o.select_set(True)
ctx.view_layer.objects.active = parts[0]
bpy.ops.object.join()
hand_mesh = ctx.object
fref = templates.create_reference(ctx, templates.human_specs(face=False), name="fingers_ref")
count = markers.detect_fingers(ctx, fref, [hand_mesh], "L", gap=0.003 * 1.8)
tip = specs["middle.03.L"].tail
base = specs["middle.01.L"].head
expected = base + (tip - base) * stretch
got = fref.data.bones["middle.03.L"].tail_local
check(count >= 4 and (got - expected).length < 0.015, f"detect fingers places the finger tips ({count} fingers, "
                                                       f"{(got - expected).length:.4f} from the tip)")
bpy.data.objects.remove(hand_mesh)
bpy.data.objects.remove(fref)

# -- Init scale -------------------------------------------------------------------------------------------------------
sref = templates.create_reference(ctx, templates.template_specs('QUADRUPED'), name="scaled_ref")
sref.scale = (2, 2, 2)
sref.rigger.rig_name = "scaled"
srig = generate.generate(ctx, sref)
check(abs(sref.scale.x - 1) < 1e-6 and abs(srig.scale.x - 1) < 1e-6
      and abs(srig.data.bones["DEF-hips"].head_local.z - 1.24) < 1e-3, "init scale applies the reference scale")

shutil.rmtree(TMP, ignore_errors=True)
print("FAILURES:", failures or "none")
if failures:
    sys.exit(1)
