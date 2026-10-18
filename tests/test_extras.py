"""Metacarpals, metatarsals, pinky auto, jaw Y unlock, eyelid borders, forced rest pose, UE-style Quick Rig.

    blender -b --factory-startup --python-exit-code 1 --python tests/test_extras.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import export, generate, quick_rig, retarget, skin, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
scene = ctx.scene
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# -- Classification -------------------------------------------------------------------
check(retarget.classify("DEF-index_meta.L")[:3] == ("metacarpal", "L", "index"), "metacarpal role")
check(retarget.classify("index_metacarpal_l")[:3] == ("metacarpal", "L", "index"), "UE metacarpal role")
check(retarget.classify("toe_thumb.01.L")[0] == "toe_digit", "individual toes are not fingers")
check(retarget.classify("thumb.01.L")[0] == "finger", "fingers still fingers")

# -- Template options: metacarpals, metatarsals, toes parent foot -----------------------
opts = dict(templates.DEFAULT_OPTIONS, metacarpals=True, toes=True, toes_metatarsal=True, toes_parent_foot=True)
specs = {sp.name: sp for sp in templates.human_specs(options=opts)}
check("index_meta.L" in specs and "thumb_meta.L" not in specs and specs["index.01.L"].parent == "index_meta.L",
      "metacarpal bones in the palm")
check(specs["toe_pinky_meta.L"].parent == "foot.L" and specs["toe_pinky.01.L"].parent == "toe_pinky_meta.L",
      "metatarsals parented to the foot")
ref = templates.create_reference(ctx, templates.template_specs("HUMAN", options=opts), name="reference")
face_root = next(pb for pb in ref.pose.bones if pb.rigger.module == 'FACE')
face_root.rigger.jaw_translate = True
face_root.rigger.jaw_unlock_y = True
rig = generate.generate(ctx, ref)
pbs = rig.pose.bones
meta = utils.get_meta(rig)
check({"index_meta.L", "pinky_meta.R", "toe_ring_meta.L"} <= set(meta["controls"]), "metacarpal controls generated")
fingers = {f["label"]: f for f in meta.get("fingers", [])}
check("index L" in fingers, "fingers under metacarpals stay fingers")
arm = next(l for l in meta["limbs"] if l["type"] == "arm" and l["side"] == "L")
# Grasp still drives the fingers through the hand
pbs[arm["prop_bone"]]["fingers_grasp"] = 1.0
ctx.view_layer.update()
rest = rig.data.bones["DEF-index.02.L"]
moved = (pbs["DEF-index.02.L"].tail - rest.tail_local).length
check(moved > 0.005, f"grasp curls fingers placed on metacarpals ({moved:.4f})")
pbs[arm["prop_bone"]]["fingers_grasp"] = 0.0
# Pinky auto
cons = [c for c in pbs["ring_meta.L"].constraints if c.name == "Pinky Auto"]
check(len(cons) == 1 and cons[0].subtarget == "pinky_meta.L", "pinky auto on the ring metacarpal")
pbs["pinky_meta.L"].rotation_mode = 'XYZ'
pbs["pinky_meta.L"].rotation_euler = (0.4, 0, 0)
ctx.view_layer.update()
ring_angle = pbs["DEF-ring_meta.L"].matrix.to_3x3().col[1].angle(rig.data.bones["DEF-ring_meta.L"].matrix_local.to_3x3().col[1])
middle_angle = pbs["DEF-middle_meta.L"].matrix.to_3x3().col[1].angle(
    rig.data.bones["DEF-middle_meta.L"].matrix_local.to_3x3().col[1])
check(ring_angle > middle_angle > 0.01, f"pinky auto cups the palm (ring {ring_angle:.3f} > middle {middle_angle:.3f})")
pbs["pinky_meta.L"].rotation_euler = (0, 0, 0)
cons = [c for c in pbs["toe_ring_meta.L"].constraints if c.name == "Pinky Auto"]
check(len(cons) == 1, "pinky auto on the metatarsals")

# Jaw unlock Y
jaw_ctrl = next(f for f in meta["faces"])["jaw"] if meta.get("faces") and meta["faces"][0].get("jaw") else "jaw"
jaw_def = rig.pose.bones["DEF-jaw"]
before = jaw_def.head.copy()
pbs[jaw_ctrl].location = (0, 0.02, 0)
ctx.view_layer.update()
slide = (jaw_def.head - before).length
check(slide > 0.015, f"jaw slides with the unlocked Y location ({slide:.4f})")
pbs[jaw_ctrl].location = (0, 0, 0)
ctx.view_layer.update()

# -- Export: UE metacarpal names, humanoid without metacarpals -------------------------
deform = [b.name for b in rig.data.bones if b.use_deform]
ue = export.export_names(rig, deform, 'UNREAL')
check(ue.get("DEF-index_meta.L") == "index_metacarpal_l" and ue.get("DEF-index.01.L") == "index_01_l",
      f"Unreal metacarpal names ({ue.get('DEF-index_meta.L')}, {ue.get('DEF-index.01.L')})")
check(ue.get("DEF-thumb.01.L") == "thumb_01_l", "Unreal thumb names unchanged")
check("thumb_01_l" not in [ue.get(n) for n in deform if n.startswith("DEF-toe_")], "toes do not take finger names")
settings = export._Defaults(skeleton='HUMANOID', metacarpals=False)
keep, merged, _tags = export.exported_bones(rig, settings)
check("DEF-index_meta.L" not in keep and "DEF-index_meta.L" in merged and "DEF-index.01.L" in keep,
      "humanoid without metacarpals")
check(export.export_parent(rig, "DEF-index.01.L", set(keep)) == "DEF-hand.L", "fingers re-parented to the hand")
keep, merged, _tags = export.exported_bones(rig, export._Defaults(skeleton='HUMANOID', metacarpals=True))
check("DEF-index_meta.L" in keep, "humanoid with metacarpals")

# Force rest pose: a helper mesh bound to every bone when no mesh is exported
s = scene.rigger_export
s.include_meshes, s.force_rest_pose, s.bake_actions = False, True, False
seen = {}
_orig = export._write


def spy(context, filepath, game, meshes, baked, settings, rig_name):
    seen["meshes"] = [m.name for m in meshes]
    seen["groups"] = {vg.name for vg in meshes[0].vertex_groups} if meshes else set()
    seen["bones"] = {b.name for b in game.data.bones}
    _orig(context, filepath, game, meshes, baked, settings, rig_name)


export._write = spy
out = os.path.join(tempfile.mkdtemp(), "rest.fbx")
utils.set_mode(rig, 'OBJECT')
bpy.ops.rigger.export_fbx(filepath=out)
export._write = _orig
check(seen.get("meshes") == ["rest_pose"] and seen["groups"] == seen["bones"], "force rest pose export")
check(not any(o.name.startswith("rest_pose") for o in bpy.data.objects), "rest pose helper removed")
s.include_meshes, s.force_rest_pose, s.bake_actions = True, False, True

# -- Eyelid borders ----------------------------------------------------------------------
eye = rig.matrix_world @ rig.data.bones["DEF-eye.L"].head_local
bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=0.02, location=eye + Vector((0, -0.01, 0)))
lids = ctx.object
lids.name = "face_skin"
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bpy.ops.object.mode_set(mode='OBJECT')
border = [v.index for v in lids.data.vertices if (lids.matrix_world @ v.co).y < eye.y - 0.02]
for i in border:
    lids.data.vertices[i].select = True
bpy.ops.object.mode_set(mode='EDIT')
res = bpy.ops.rigger.eyelid_border(side='L')
bpy.ops.object.mode_set(mode='OBJECT')
check(res == {'FINISHED'} and skin.BORDER_GROUP + ".L" in lids.vertex_groups, "set left eyelid border")
skin.bind_mesh(ctx, lids, rig, 'PROXIMITY', rigid_parts=False)
lid_names = {vg.index: vg.name for vg in lids.vertex_groups}
ok = True
for i in border:
    v = lids.data.vertices[i]
    groups = [(lid_names[g.group], g.weight) for g in v.groups if lid_names[g.group].startswith("DEF-")]
    up = (lids.matrix_world @ v.co).z >= eye.z
    want = "DEF-lid_upper" if up else "DEF-lid_lower"
    if len(groups) != 1 or not groups[0][0].startswith(want) or abs(groups[0][1] - 1.0) > 1e-4:
        ok = False
check(border and ok, f"eyelid border vertices follow the lids ({len(border)} vertices)")
ctx.view_layer.objects.active = lids
bpy.ops.rigger.eyelid_border(side='CLEAR')
check(skin.BORDER_GROUP + ".L" not in lids.vertex_groups, "clear eyelid borders")

# -- Quick Rig: UE-style skeleton with metacarpals ---------------------------------------
UE = {"hips": "pelvis", "spine.01": "spine_01", "spine.02": "spine_02", "spine.03": "spine_03",
      "neck": "neck_01", "head": "head"}
for side in "LR":
    s_ = side.lower()
    UE.update({f"shoulder.{side}": f"clavicle_{s_}", f"upper_arm.{side}": f"upperarm_{s_}",
               f"forearm.{side}": f"lowerarm_{s_}", f"hand.{side}": f"hand_{s_}",
               f"thigh.{side}": f"thigh_{s_}", f"shin.{side}": f"calf_{s_}", f"foot.{side}": f"foot_{s_}",
               f"toe.{side}": f"ball_{s_}"})
    for f in templates.FINGER_ORDER:
        if f != "thumb":
            UE[f"{f}_meta.{side}"] = f"{f}_metacarpal_{s_}"
        for i in (1, 2, 3):
            UE[f"{f}.{i:02d}.{side}"] = f"{f}_{i:02d}_{s_}"
src_specs = templates.mirrored(templates.human_specs(face=False, options=dict(templates.DEFAULT_OPTIONS,
                                                                                 metacarpals=True)))
data = bpy.data.armatures.new("ue_skel")
skel = bpy.data.objects.new("ue_skel", data)
scene.collection.objects.link(skel)
skel.location = (3, 0, 0)
utils.set_mode(skel, 'EDIT')
for sp in src_specs:
    eb = data.edit_bones.new(UE[sp.name])
    eb.head, eb.tail = sp.head, sp.tail
for sp in src_specs:
    if sp.parent:
        data.edit_bones[UE[sp.name]].parent = data.edit_bones[UE[sp.parent]]
bpy.ops.object.mode_set(mode='OBJECT')
qs = scene.rigger_quick
qs.armature = skel
qs.mode = 'PRESERVE'
qs.bake = False
bpy.ops.rigger.quick_rig_guess()
index = next(l for l in qs.limbs if l.kind == 'FINGER' and l.label == "index" and l.side == 'L')
check(index.metacarpal and quick_rig.bone_list(index)[0] == "index_metacarpal_l", "guess finds the metacarpal")
thumb = next(l for l in qs.limbs if l.kind == 'FINGER' and l.label == "thumb" and l.side == 'L')
check(not thumb.metacarpal, "thumb has no metacarpal")
res = bpy.ops.rigger.quick_rig()
rig_q = next(o for o in bpy.data.objects if o.get("rigger_quick_source") == "ue_skel")
ref_q = rig_q.rigger.reference
check(res == {'FINISHED'} and ref_q.pose.bones["index_meta.L"].rigger.module == 'SIMPLE'
      and ref_q.pose.bones["index.01.L"].rigger.module == 'FINGER', "Quick Rig metacarpal modules")
pb = rig_q.pose.bones["index_meta.L"]
pb.rotation_mode = 'XYZ'
pb.rotation_euler = (0.3, 0, 0)
ctx.view_layer.update()
a = skel.matrix_world @ skel.pose.bones["index_01_l"].head
b = rig_q.matrix_world @ rig_q.pose.bones["DEF-index.01.L"].head
check((a - b).length < 1e-3 and (a - (skel.matrix_world @ skel.data.bones["index_01_l"].head_local)).length > 1e-3,
      "skeleton finger follows the metacarpal control")

# -- Parent fallback, init scale, eye target distance, wings fold, soft lips and brows ----------------
opts = dict(templates.DEFAULT_OPTIONS, lips=3, brows=4)
ref6 = templates.create_reference(ctx, templates.template_specs("HUMAN", options=opts), name="ref6")
ref6.location = (6, 0, 0)
utils.set_mode(ref6, 'EDIT')
hat = ref6.data.edit_bones.new("hat")
hat.head, hat.tail = Vector((0, 0, 1.85)), Vector((0, 0, 1.95))
bpy.ops.object.mode_set(mode='OBJECT')
ref6.pose.bones["hat"].rigger.module = 'SIMPLE'
ref6.pose.bones["hat"].rigger.parent_fallback = "DEF-head"
face6 = next(pb for pb in ref6.pose.bones if pb.rigger.module == 'FACE').rigger
face6.brow_master = True
ref6.pose.bones["shoulder.L"].rigger.wings_fold = True
head_z = ref6.data.bones["head"].head_local.z
ref6.scale = (2.0, 2.0, 2.0)
rig6 = generate.generate(ctx, ref6)
check(rig6.data.bones["hat"].parent.name == "DEF-head", "parent fallback: an unparented module follows the head")
check(abs(ref6.scale.x - 1.0) < 1e-6 and abs(rig6.matrix_world.to_scale().x - 1.0) < 1e-6
      and abs(rig6.data.bones["DEF-head"].head_local.z - 2 * head_z) < 1e-4, "init scale applies the reference scale")


def eye_target_distance(rig):
    return (rig.data.bones["eye_target.L"].head_local - rig.data.bones["DEF-eye.L"].head_local).length


near = eye_target_distance(rig6)
for side in "LR":
    ref6.pose.bones[f"eye.{side}"].rigger.eye_distance = 24.0
rig6 = generate.generate(ctx, ref6)
far = eye_target_distance(rig6)
check(abs(far / near - 2.0) < 0.05, f"eye target distance ({near:.3f} -> {far:.3f})")

pbs6 = rig6.pose.bones
fold = next(l for l in utils.get_meta(rig6)["limbs"] if l["type"] == "arm" and l["side"] == "L")["wings_fold"]
check(fold == "hand_wings_fold.L", "wings fold control")
fold_action = bpy.data.actions.new("rig_wings_fold")
utils.assign_action(rig6, fold_action)
pbs6["forearm_fk.L"].rotation_quaternion = (1, 0, 0, 0)
pbs6["forearm_fk.L"].keyframe_insert("rotation_quaternion", frame=0)
pbs6["forearm_fk.L"].rotation_quaternion = Matrix.Rotation(math.radians(-80), 4, 'X').to_quaternion()
pbs6["forearm_fk.L"].keyframe_insert("rotation_quaternion", frame=10)
rig6.animation_data.action = None
pbs6["forearm_fk.L"].rotation_quaternion = (1, 0, 0, 0)
rig6 = generate.generate(ctx, ref6)
pbs6 = rig6.pose.bones
arm6 = next(l for l in utils.get_meta(rig6)["limbs"] if l["type"] == "arm" and l["side"] == "L")
pbs6[arm6["prop_bone"]]["ik_fk"] = 0.0
ctx.view_layer.update()
rest_tip = (pbs6["DEF-hand.L"].head).copy()
pbs6[fold].scale = (1, 0.0, 1)
ctx.view_layer.update()
folded = (pbs6["DEF-hand.L"].head - rest_tip).length
check(folded > 0.05, f"scaling the wings fold control plays the fold action ({folded:.3f})")
pbs6[fold].scale = (1, 1, 1)

# Soft lips: lips near the corner follow it more than the middle ones
lips = sorted((n for n in pbs6.keys() if n.startswith("lip_upper.") and n.endswith(".L")),
              key=lambda n: abs(rig6.data.bones[n].head_local.x))
defs = ["DEF-" + n for n in lips]
before = [pbs6[d].head.copy() for d in defs]
pbs6["lip_corner.L"].location = (0, 0, 0.01)
ctx.view_layer.update()
moves = [(pbs6[d].head - b).length for d, b in zip(defs, before)]
check(moves[0] < moves[1] < moves[2], f"soft lips fall off toward the middle ({[round(m, 4) for m in moves]})")
pbs6["lip_corner.L"].location = (0, 0, 0)

# Soft brows: the brow master moves the bones less toward the tip
brows = sorted((n for n in pbs6.keys() if n.startswith("brow.") and n.endswith(".L")))
before = [pbs6["DEF-" + n].head.copy() for n in brows]
pbs6["brow_master.L"].location = (0, 0, 0.01)
ctx.view_layer.update()
moves = [(pbs6["DEF-" + n].head - b).length for n, b in zip(brows, before)]
check(moves[0] > 0.001 and len(set(round(m, 5) for m in moves)) > 1, f"soft eyebrows ({[round(m, 4) for m in moves]})")
pbs6["brow_master.L"].location = (0, 0, 0)

# Bone collections and colours
colls = {c.name: c.is_visible for c in rig6.data.collections}
check(all(colls.get(n) for n in ("Root", "Torso", "Head", "Face", "Arm FK", "Arm IK", "Leg FK", "Leg IK", "Fingers"))
      and colls.get("Deform") is False and colls.get("Mechanism") is False, f"bone collections ({colls})")
left, right = pbs6["upper_arm_fk.L"].color.custom.normal, pbs6["upper_arm_fk.R"].color.custom.normal
check(pbs6["upper_arm_fk.L"].color.palette == 'CUSTOM' and tuple(left) != tuple(right), "left/right control colours")

# Toes with IK chains (finger IK on a toe)
opts = dict(templates.DEFAULT_OPTIONS, toes=True)
ref8 = templates.create_reference(ctx, templates.template_specs("HUMAN", options=opts), name="ref8")
ref8.location = (9, 0, 0)
ref8.pose.bones["toe_index.01.L"].rigger.finger_ik = True
ref8.rigger.rig_name = "toes_rig"
rig8 = generate.generate(ctx, ref8)
p8 = rig8.pose.bones
check("toe_index_ik.L" in p8, "toe IK control")
p8["toe_index_ik.L"]["ik_fk"] = 1.0
tip = p8["DEF-toe_index.02.L"].tail.copy()
p8["toe_index_ik.L"].location = (0, 0, 0.01)
ctx.view_layer.update()
check((p8["DEF-toe_index.02.L"].tail - tip).length > 0.003, "toe follows its IK control")

# Snap Switch baked over a frame range (head lock)
head8 = utils.get_meta(rig8)["neck"]["head"]
act = bpy.data.actions.new("look")
utils.assign_action(rig8, act)
for frame, angle in ((1, 0), (5, 40)):
    p8["neck"].rotation_quaternion = Matrix.Rotation(math.radians(angle), 4, 'X').to_quaternion()
    p8["neck"].keyframe_insert("rotation_quaternion", frame=frame)
worlds = {}
for frame in range(1, 6):
    scene.frame_set(frame)
    worlds[frame] = (rig8.matrix_world @ p8[head8].matrix).to_quaternion()
utils.set_mode(rig8, 'POSE')
rig8.data.bones.active = rig8.data.bones[head8]
res = bpy.ops.rigger.snap_property(bone=head8, prop="head_isolate", value=1.0, bake=True, frame_start=1, frame_end=5)
err = 0.0
for frame in range(1, 6):
    scene.frame_set(frame)
    err = max(err, math.degrees((rig8.matrix_world @ p8[head8].matrix).to_quaternion()
                                .rotation_difference(worlds[frame]).angle))
check(res == {'FINISHED'} and p8[head8]["head_isolate"] == 1.0 and err < 0.5,
      f"head lock switched and baked over frames 1-5 without moving the head ({err:.3f} deg)")
bpy.ops.object.mode_set(mode='OBJECT')

# Child Of switcher baked over a frame range
targets = []
for i, x in enumerate((0.0, 0.5)):
    empty = bpy.data.objects.new(f"holder_{i}", None)
    scene.collection.objects.link(empty)
    targets.append(empty)
targets[1].animation_data_create()
targets[1].animation_data.action = bpy.data.actions.new("holder_move")
for frame, z in ((1, 0.0), (5, 0.4)):
    targets[1].location = (0.5, 0, z)
    targets[1].keyframe_insert("location", frame=frame)
hand8 = p8["hand_ik.L"]
for i, empty in enumerate(targets):
    con = hand8.constraints.new('CHILD_OF')
    con.target = empty
    con.influence = 1.0 if i == 0 else 0.0
    con.inverse_matrix = empty.matrix_world.inverted()
ctx.view_layer.update()
before = {}
for frame in range(1, 6):
    scene.frame_set(frame)
    before[frame] = hand8.matrix.copy()
utils.set_mode(rig8, 'POSE')
rig8.data.bones.active = rig8.data.bones["hand_ik.L"]
res = bpy.ops.rigger.snap_child_of(index=1, bake=True, frame_start=1, frame_end=5)
err = 0.0
for frame in range(1, 6):
    scene.frame_set(frame)
    err = max(err, (hand8.matrix.translation - before[frame].translation).length)
cons = [c for c in hand8.constraints if c.type == 'CHILD_OF']
check(res == {'FINISHED'} and cons[1].influence == 1.0 and cons[0].influence == 0.0 and err < 1e-3,
      f"Child Of switch baked over frames 1-5 keeps the control in place ({err:.5f})")
bpy.ops.object.mode_set(mode='OBJECT')

# Spline IK/FK snap keeps the shape
ref7 = templates.create_reference(ctx, templates.spline_chunk(), name="rope_ref")
ref7.location = (-6, 0, 0)
root7 = next(pb for pb in ref7.pose.bones if pb.rigger.module == 'SPLINE')
root7.rigger.spline_fk = True
ref7.rigger.rig_name = "rope"
rope = generate.generate(ctx, ref7)
spline = utils.get_meta(rope)["splines"][0]
rp = rope.pose.bones
rp[spline["controls"][-1]].location = (0.1, 0, 0.05)
ctx.view_layer.update()
deforms = [b.name for b in rope.data.bones if b.use_deform]
shape = {n: rp[n].tail.copy() for n in deforms}
utils.set_mode(rope, 'POSE')
res = bpy.ops.rigger.snap_spline(master=spline["master"])
ctx.view_layer.update()
err = max((rp[n].tail - shape[n]).length for n in deforms)
check(res == {'FINISHED'} and rp[spline["master"]]["ik_fk"] < 0.5 and err < 2.5e-3,
      f"snap spline IK to FK keeps the shape ({err:.5f})")
bpy.ops.object.mode_set(mode='OBJECT')

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("\nFAILURES: none")
