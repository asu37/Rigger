"""Skinning options and game-engine export options.

    blender -b --factory-startup --python-exit-code 1 --python tests/test_skin_export.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import export, generate, rig_data, skin, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
scene = ctx.scene
failures = []
OUT = tempfile.mkdtemp()


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


ref = templates.create_reference(ctx, templates.template_specs("HUMAN", options=None), name="reference")
rig = generate.generate(ctx, ref)
ref.hide_set(True)
DEFORM = sorted(b.name for b in rig.data.bones if b.use_deform)


def capsule_body(name="body"):
    parts = []
    for spec in templates.mirrored(templates.human_specs(fingers=False, face=False)):
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
    obj = ctx.object
    obj.name = name
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.subdivide(number_cuts=4)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def weights(obj, name):
    vg = obj.vertex_groups.get(name)
    w = np.zeros(len(obj.data.vertices))
    if vg is None:
        return w
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == vg.index:
                w[v.index] = g.weight
    return w


def points(obj):
    return np.array([obj.matrix_world @ v.co for v in obj.data.vertices])


def all_weighted(obj):
    return not skin.unweighted_vertices(obj, set(DEFORM))


def world_head(name):
    return rig.matrix_world @ rig.data.bones[name].head_local


# =============================================================================
# Skin
# =============================================================================
body = capsule_body()

# Voxelized binding
vertex_count = len(body.data.vertices)
filled, rigid = skin.bind_mesh(ctx, body, rig, 'VOXEL', voxel_resolution=48)
check(all_weighted(body) and weights(body, "DEF-upper_arm.L").max() > 0.5, "voxelized bind")
check(not any(o.name.endswith("_rigger_proxy") for o in bpy.data.objects)
      and len(body.data.vertices) == vertex_count, "no proxy left behind, mesh unchanged")

# Split parts
skin.bind_mesh(ctx, body, rig, 'HEAT', split_parts=True)
check(all_weighted(body), "split parts bind")
leftover = [o.name for o in bpy.data.objects if o.type == 'MESH' and not o.name.startswith("WGT-")]
check(leftover == ["body"], f"split parts cleaned up ({leftover})")

# Optimize high-res (decimated proxy), shape keys mix, scale fix
body.shape_key_add(name="Basis")
key = body.shape_key_add(name="fat")
for p in key.data:
    p.co *= 1.02
key.value = 1.0
skin.bind_mesh(ctx, body, rig, 'AUTO', optimize=100)
check(all_weighted(body), "optimize high-res bind (decimated proxy)")
skin.bind_mesh(ctx, body, rig, 'AUTO', apply_shape_keys=True)
check(all_weighted(body) and body.data.shape_keys is not None and len(body.data.shape_keys.key_blocks) == 2,
      "bind with the shape key mix, shape keys kept")
body.shape_key_clear()
skin.bind_mesh(ctx, body, rig, 'AUTO', scale_fix=True)
check(all_weighted(body), "scale fix bind")

# Selected bones only
for pb in rig.pose.bones:
    utils.select_pose_bone(pb, pb.name in ("DEF-upper_arm.L", "DEF-forearm.L", "DEF-forearm_twist.01.L"))
for vg in list(body.vertex_groups):
    body.vertex_groups.remove(vg)
skin.bind_mesh(ctx, body, rig, 'PROXIMITY', selected_bones=True, rigid_parts=False)
groups = {vg.name for vg in body.vertex_groups if weights(body, vg.name).max() > 0}
check(groups <= {"DEF-upper_arm.L", "DEF-forearm.L", "DEF-forearm_twist.01.L"} and groups,
      f"selected bones only ({sorted(groups)})")
check(all(rig.data.bones[n].use_deform for n in DEFORM), "deform flags restored")
for pb in rig.pose.bones:
    utils.select_pose_bone(pb, False)

# Selected vertices only + locked groups
skin.bind_mesh(ctx, body, rig, 'PROXIMITY', rigid_parts=False)
pts = points(body)
selected = pts[:, 0] > 0.3  # left arm
for v in body.data.vertices:
    v.select = bool(selected[v.index])
before_spine = weights(body, "DEF-spine.02")
locked = body.vertex_groups.new(name="custom_locked")
locked.add([0, 1, 2], 0.25, 'REPLACE')
locked.lock_weight = True
for vg in body.vertex_groups:
    if vg.name.startswith("DEF-"):
        vg.remove([v.index for v in body.data.vertices if v.select])
skin.bind_mesh(ctx, body, rig, 'PROXIMITY', selected_verts=True, rigid_parts=False)
after_spine = weights(body, "DEF-spine.02")
check(np.allclose(before_spine[~selected], after_spine[~selected], atol=1e-3),
      "unselected vertices keep their weights")
arm_w = sum(weights(body, n) for n in ("DEF-upper_arm.L", "DEF-upper_arm_twist.01.L", "DEF-forearm.L",
                                         "DEF-forearm_twist.01.L", "DEF-forearm_twist.02.L"))
check(arm_w[selected].max() > 0.5, "selected vertices weighted")
lw = weights(body, "custom_locked")
check(np.allclose(lw[:3], 0.25, atol=1e-3) and lw[3:].max() == 0 and body.vertex_groups["custom_locked"].lock_weight,
      "locked vertex group untouched")
body.vertex_groups.remove(body.vertex_groups["custom_locked"])
for v in body.data.vertices:
    v.select = False

# Smooth twist weights: the twist bone takes over towards the wrist
skin.bind_mesh(ctx, body, rig, 'AUTO')
skin.smooth_twist_weights(body, rig)
pts = points(body)
chain = next(c for c in skin._chains(rig) if c[0] == "DEF-forearm.L")
elbow = world_head(chain[0])
wrist = max((rig.matrix_world @ rig.data.bones[n].tail_local for n in chain), key=lambda p: (p - elbow).length)
axis = np.array(wrist - elbow)
t = ((pts - np.array(elbow)) @ axis) / (axis @ axis)
main, twist = weights(body, "DEF-forearm.L"), sum(weights(body, n) for n in chain[1:])
if "DEF-forearm_fan.L" in body.vertex_groups:  # the elbow fan shares the skin at the elbow end
    main = main + weights(body, "DEF-forearm_fan.L")
near_elbow = (t > 0.1) & (t < 0.3) & ((main + twist) > 0.9)
near_wrist = (t > 0.7) & (t < 0.9) & ((main + twist) > 0.9)
check(near_elbow.any() and near_wrist.any() and main[near_elbow].mean() > twist[near_elbow].mean()
      and twist[near_wrist].mean() > main[near_wrist].mean(),
      f"smooth twist weights gradient (elbow {main[near_elbow].mean():.2f}/{twist[near_elbow].mean():.2f}, "
      f"wrist {main[near_wrist].mean():.2f}/{twist[near_wrist].mean():.2f})")

# Tightened fans: fan weight only within the band around the joint
fan_names = [b.name for b in rig.data.bones if b.use_deform and "_fan" in b.name]
skin.tighten_fan_weights(body, rig)
pts = points(body)
h = skin.rig_height(rig)
far_fan = 0.0
for name in fan_names:
    d = np.linalg.norm(pts - np.array(world_head(name)), axis=1)
    if name in body.vertex_groups:
        far_fan = max(far_fan, weights(body, name)[d > skin.FAN_BAND[1] * h + 1e-6].max(initial=0.0))
check(fan_names and far_fan < 1e-3, f"fan weights stay near their joint ({len(fan_names)} fans, max far {far_fan:.4f})")


# Improve hips: no thigh weight above the hip joint
thigh_z = world_head("DEF-thigh.L").z
skin.improve_hips_weights(body, rig)
high = pts[:, 2] > thigh_z + 0.08
check(weights(body, "DEF-thigh.L")[high].max() < 1e-3, "improve hips weights")

# Improve heels: below the ankle follows the foot
ankle = world_head("DEF-foot.L")
skin.improve_heel_weights(body, rig)
heel = (pts[:, 2] < ankle.z - 0.02) & (np.abs(pts[:, 0] - ankle.x) < 0.05) & (pts[:, 1] > ankle.y - 0.02)
shin = weights(body, "DEF-shin.L") + weights(body, "DEF-shin_twist.01.L")
check(heel.any() and shin[heel].max() < 1e-3, "improve heel weights")

# Refine head: above the head base belongs to the head
head_z = world_head("DEF-head").z
skin.refine_head_weights(body, rig)
top = pts[:, 2] > head_z + 0.03
neck_w, head_w = weights(body, "DEF-neck")[top], weights(body, "DEF-head")[top]
check(top.any() and neck_w.max() < 1e-3 and head_w.min() > 0.5,
      f"refine head weights (neck max {neck_w.max():.3f}, head min {head_w.min():.3f})")


def set_weights(v, values):
    for vg in body.vertex_groups:
        vg.remove([int(v)])
    for name, value in values.items():
        (body.vertex_groups.get(name) or body.vertex_groups.new(name=name)).add([int(v)], value, 'REPLACE')


# Head and shoulders stop sharing weights
head_v = int(np.nonzero(top)[0][0])
low_v = int(np.argmin(pts[:, 2]))  # far below the head: only the shoulder rule applies
set_weights(head_v, {"DEF-head": 0.8, "DEF-hand.L": 0.2})
set_weights(low_v, {"DEF-shoulder.L": 0.7, "DEF-head": 0.3})
skin.refine_head_weights(body, rig)
check(weights(body, "DEF-hand.L")[head_v] == 0 and abs(weights(body, "DEF-head")[head_v] - 1.0) < 1e-2,
      "head skin drops arm weights")
check(weights(body, "DEF-head")[low_v] == 0 and abs(weights(body, "DEF-neck")[low_v] - 0.3) < 1e-2,
      "shoulder skin gives its head weight to the neck")

# Fingers stop sticking together
tip, web, mid = 1, 2, 3
set_weights(tip, {"DEF-index.03.L": 0.7, "DEF-middle.03.L": 0.3})
set_weights(web, {"DEF-index.01.L": 0.6, "DEF-middle.01.L": 0.4})
set_weights(mid, {"DEF-index.02.L": 0.8, "DEF-hand.L": 0.2})
skin.clean_finger_weights(body, rig)
check(weights(body, "DEF-middle.03.L")[tip] == 0 and abs(weights(body, "DEF-index.03.L")[tip] - 1.0) < 1e-2,
      "finger tip drops the next finger's weight")
check(abs(weights(body, "DEF-middle.01.L")[web] - 0.4) < 1e-2, "web between base segments keeps both fingers")
check(weights(body, "DEF-hand.L")[mid] == 0, "middle phalanx drops the hand weight")
defaults = {k: bpy.ops.rigger.bind.get_rna_type().properties[k].default
            for k in ("improve_hips", "refine_head", "clean_fingers")}
check(defaults == {"improve_hips": False, "refine_head": True, "clean_fingers": True},
      f"bind defaults: hips cut off, head refine and finger cleanup on ({defaults})")

# Facial objects, preserve volume and parent options through the operator
eye_l = world_head("DEF-eye.L")
bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=6, radius=0.012, location=eye_l)
eye_obj = ctx.object
eye_obj.name = "eyeball_L"
bpy.ops.mesh.primitive_uv_sphere_add(segments=8, ring_count=6, radius=0.012, location=world_head("DEF-eye.R"))
eye_obj_r = ctx.object
eye_obj_r.name = "eyeball_R"
bpy.ops.mesh.primitive_cube_add(size=0.02, location=world_head("DEF-teeth_upper"))
teeth_up = ctx.object
teeth_up.name = "teeth_up"
bpy.ops.mesh.primitive_cube_add(size=0.02, location=world_head("DEF-teeth_lower"))
teeth_low = ctx.object
teeth_low.name = "teeth_low"
t0, t1 = world_head("DEF-tongue.01"), rig.matrix_world @ rig.data.bones["DEF-tongue.03"].tail_local
bpy.ops.mesh.primitive_cube_add(size=1, location=(t0 + t1) / 2)
tongue = ctx.object
tongue.name = "tongue"
tongue.scale = (0.01, max((t1 - t0).length / 2, 0.01), 0.005)
bpy.ops.object.transform_apply(scale=True)
bpy.ops.object.select_all(action='DESELECT')
for o in (eye_obj, eye_obj_r, teeth_up, teeth_low, tongue):
    o.select_set(True)
rig.select_set(True)
ctx.view_layer.objects.active = rig
res = bpy.ops.rigger.bind(method='PROXIMITY', eyes="eyeball_L, eyeball_R", teeth="teeth_up, teeth_low",
                          tongue="tongue", preserve_volume=False, parent=False)
check(res == {'FINISHED'}, "bind facial objects")
check(weights(eye_obj, "DEF-eye.L").min() == 1.0 and weights(eye_obj_r, "DEF-eye.R").min() == 1.0,
      "eyeballs bound to the eyes")
check(weights(teeth_up, "DEF-teeth_upper").min() == 1.0 and weights(teeth_low, "DEF-teeth_lower").min() == 1.0,
      "teeth bound")
tongue_groups = {vg.name for vg in tongue.vertex_groups if weights(tongue, vg.name).max() > 0}
check(tongue_groups and all(n.startswith("DEF-tongue") for n in tongue_groups) and len(tongue_groups) >= 2,
      f"tongue bound to the tongue bones ({sorted(tongue_groups)})")
mod = next(m for m in eye_obj.modifiers if m.type == 'ARMATURE')
check(mod.object == rig and not mod.use_deform_preserve_volume and eye_obj.parent is None,
      "preserve volume off, not parented")
for o in (eye_obj, eye_obj_r, teeth_up, teeth_low, tongue):
    o.parent = rig
    o.matrix_parent_inverse = rig.matrix_world.inverted()

# Warnings
bpy.ops.mesh.primitive_cube_add(size=0.1, location=(0.7, 0, 1.43))
prop = ctx.object
prop.name = "bracelet"
prop.delta_location = (0.01, 0, 0)
prop.scale = (-1, 1, 1)
ctx.view_layer.update()
msgs = skin.warnings_for(prop)
check(any("delta" in m for m in msgs) and any("negative" in m for m in msgs), "delta / negative scale warnings")
prop.delta_location = (0, 0, 0)
prop.scale = (1, 1, 1)
skin.bind_mesh(ctx, prop, rig, 'PROXIMITY')

# =============================================================================
# Export
# =============================================================================
captured = []
_orig_write = export._write


def spy(context, filepath, game, meshes, baked, settings, rig_name):
    info = {
        "path": filepath, "name": game.name,
        "parents": {b.name: (b.parent.name if b.parent else None) for b in game.data.bones},
        "deform": {b.name for b in game.data.bones if b.use_deform},
        "lengths": {b.name: game.matrix_world.to_scale().x * b.length for b in game.data.bones},
        "root_dir": tuple(game.data.bones[0].vector.normalized()) if game.data.bones else None,
        "meshes": [m.name for m in meshes],
        "groups": {m.name: {vg.name for vg in m.vertex_groups} for m in meshes},
        "unit": context.scene.unit_settings.scale_length,
        "actions": {},
    }
    for action in baked:
        curves = {}
        for fc in utils.action_fcurves(action):
            curves[(fc.data_path, fc.array_index)] = [(k.co.x, k.co.y, k.interpolation) for k in fc.keyframe_points]
        info["actions"][action.get("rigger_take", action.name)] = {"range": tuple(action.frame_range), "curves": curves}
    _orig_write(context, filepath, game, meshes, baked, settings, rig_name)
    info["tracks"] = [t.name for t in game.animation_data.nla_tracks] if game.animation_data else []
    captured.append(info)


export._write = spy
s = scene.rigger_export


def run_export(name, select=(), **options):
    """Export with the given settings (reset to defaults first). Returns the captured info list."""
    for key in export._setting_keys():
        prop = s.bl_rna.properties[key]
        if getattr(prop, "is_array", False):
            continue
        try:
            setattr(s, key, prop.default if prop.type != 'ENUM' else prop.default)
        except (TypeError, AttributeError):
            pass
    for key, value in options.items():
        setattr(s, key, value)
    captured.clear()
    utils.set_mode(rig, 'OBJECT')
    for obj in select:
        obj.select_set(True)
    path = os.path.join(OUT, name)
    res = bpy.ops.rigger.export_fbx(filepath=path)
    return res, list(captured), path


def curve(info, action, bone, prop, index):
    return info["actions"][action]["curves"].get((f'pose.bones["{bone}"].{prop}', index), [])


# Animations
pbs = rig.pose.bones
walk = bpy.data.actions.new("walk")
utils.assign_action(rig, walk)
for frame in range(1, 11):
    pbs["upper_arm_fk.L"].rotation_quaternion = Matrix.Rotation(frame * 0.08, 4, 'X').to_quaternion()
    pbs["upper_arm_fk.L"].keyframe_insert("rotation_quaternion", frame=frame)
    pbs["torso"].location = (0, 0.05 * frame, 0)
    pbs["torso"].keyframe_insert("location", frame=frame)
walk.pose_markers.new("start").frame = 3
walk.pose_markers.new("end").frame = 8
run = bpy.data.actions.new("run")
utils.assign_action(rig, run)
for frame in (1, 6):
    pbs["torso"].location = (0, 0.2 * frame, 0)
    pbs["torso"].keyframe_insert("location", frame=frame)
idle = bpy.data.actions.new("idle")
utils.assign_action(rig, idle)
pbs["torso"].location = (0, 0, 0)
pbs["torso"].keyframe_insert("location", frame=1)
pbs["torso"].keyframe_insert("location", frame=4)
utils.assign_action(rig, walk)
for pb in pbs:
    pb.matrix_basis = Matrix()

# Universal vs humanoid, facial bones
res, info, path = run_export("universal.fbx", naming='RIGGER')
bones = set(info[0]["parents"])
check(res == {'FINISHED'} and {"root", "hips", "lip_upper", "tongue.01", "forearm_twist.01.L"} <= bones,
      "universal: every deform bone")
check(info[0]["parents"]["hips"] == "root" and info[0]["parents"]["forearm.L"] == "upper_arm.L",
      "deform hierarchy")
res, info, path = run_export("humanoid.fbx", skeleton='HUMANOID', full_facial=False, bake_actions=False)
bones = set(info[0]["parents"])
check({"hips", "head", "jaw", "eye.L", "index.03.L", "forearm_twist.01.L"} <= bones and "lip_upper" not in bones
      and "tongue.01" not in bones, "humanoid with basic facial")
groups = info[0]["groups"]["body_export"]
check("DEF-lip_upper" not in groups and "lip_upper" not in groups, "weights of skipped bones merged")
res, info, path = run_export("humanoid_face.fbx", skeleton='HUMANOID', full_facial=True, bake_actions=False)
check("lip_upper" in info[0]["parents"], "humanoid with full facial")

# No twist
res, info, path = run_export("notwist.fbx", export_twist=False, bake_actions=False)
check("forearm_twist.01.L" not in info[0]["parents"] and "forearm_twist.01.L" not in info[0]["groups"]["body_export"],
      "twist bones not exported, weights merged")

# Custom bone, exclude, soft-link, constant tags
utils.set_mode(rig, 'EDIT')
eb = rig.data.edit_bones.new("cc_sword")
hand = rig.data.edit_bones["hand_fk.L"]
eb.head, eb.tail = hand.tail.copy(), hand.tail + Vector((0, 0, 0.3))
eb.parent = hand
eb.use_deform = False
eb = rig.data.edit_bones.new("weapon")
eb.head, eb.tail = Vector((0.5, 0, 1)), Vector((0.5, 0, 1.2))
eb.parent = rig.data.edit_bones["root"]
eb.use_deform = False
bpy.ops.object.mode_set(mode='POSE')
pbs = rig.pose.bones
pbs["weapon"]["custom_bone"] = 1
for pb in pbs:
    utils.select_pose_bone(pb, pb.name == "DEF-spine.02")
bpy.ops.rigger.tag_bones(tag='EXCLUDE')
for pb in pbs:
    utils.select_pose_bone(pb, pb.name == "DEF-head")
bpy.ops.rigger.tag_bones(tag='SOFTLINK')
for pb in pbs:
    utils.select_pose_bone(pb, pb.name == "DEF-upper_arm.L")
bpy.ops.rigger.tag_bones(tag='CONST')
for pb in pbs:
    utils.select_pose_bone(pb, False)
tags = rig_data.export_tags(rig)
check(tags.get("cc_sword") == 'CUSTOM' and tags.get("weapon") == 'CUSTOM' and tags.get("DEF-spine.02") == 'EXCLUDE',
      "export tags (prefix, property, button)")
bpy.ops.object.mode_set(mode='OBJECT')
game, mapping, merge_to, _tags = export.build_game_rig(ctx, rig, 'RIGGER', "probe", settings=s)
cons = [c.type for c in game.pose.bones["head"].constraints]
check(cons == ['COPY_LOCATION', 'COPY_ROTATION'], f"soft-link bone ({cons})")
check(game.data.bones["cc_sword"].parent.name == "hand.L" and game.data.bones["cc_sword"].use_deform,
      "custom bone exported under the deform hand")
check("spine.02" not in game.data.bones and game.data.bones["spine.03"].parent.name == "spine.01",
      "excluded bone skipped, children re-parented")
check(merge_to.get("DEF-spine.02") == "spine.01", "excluded bone weights go to the parent")
bpy.data.objects.remove(game)
res, info, path = run_export("tags.fbx", only_active=True)
interp = {k[2] for k in curve(info[0], "walk", "upper_arm.L", "rotation_quaternion", 0)}
check(interp == {'CONSTANT'}, f"constant interpolation tag ({interp})")
check({k[2] for k in curve(info[0], "walk", "forearm.L", "rotation_quaternion", 0)} != {'CONSTANT'},
      "other bones keep their interpolation")
utils.set_mode(rig, 'POSE')
for pb in pbs:
    utils.select_pose_bone(pb, pb.name in ("DEF-spine.02", "DEF-head", "DEF-upper_arm.L"))
bpy.ops.rigger.tag_bones(tag='NONE')
for pb in pbs:
    utils.select_pose_bone(pb, False)
bpy.ops.object.mode_set(mode='OBJECT')

# No parents, rename file, rig name, custom script
rename = os.path.join(OUT, "rename.txt")
with open(rename, "w") as f:
    f.write("# old = new\nhips = Pelvis_X\n")
script = bpy.data.texts.new("export_script")
script.write("rig_export['from_script'] = 1\nrig_export.data.bones['weapon'].name = 'weapon_scripted'\n")
res, info, path = run_export("flat.fbx", no_parents=True, rename_bones=rename, rig_name="Hero",
                             custom_script="export_script", bake_actions=False)
parents = info[0]["parents"]
check(all(p == "root" for n, p in parents.items() if n != "root"), "no parents: flat hierarchy")
check("Pelvis_X" in parents and "hips" not in parents, "rename bones from a file")
check(info[0]["name"] == "Hero", "rig name")
check("weapon_scripted" in parents, "custom export script ran")
# Running the script can be refused; Quick Export only runs scripts allowed in this session
captured.clear()
bpy.ops.rigger.export_fbx(filepath=os.path.join(OUT, "flat_noscript.fbx"), run_script=False)
check("weapon_scripted" not in captured[0]["parents"], "export without running the custom script")
export.APPROVED_SCRIPTS.clear()
captured.clear()
utils.set_mode(rig, 'OBJECT')
bpy.ops.rigger.quick_export()
check("weapon_scripted" not in captured[0]["parents"], "quick export skips a script nobody allowed")
export.APPROVED_SCRIPTS.add("export_script")
captured.clear()
utils.set_mode(rig, 'OBJECT')
bpy.ops.rigger.quick_export()
check("weapon_scripted" in captured[0]["parents"], "quick export runs an allowed script")
export.APPROVED_SCRIPTS.clear()
before = set(bpy.data.objects.keys())
bpy.ops.import_scene.fbx(filepath=path)
imported = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
arm = next(o for o in imported if o.type == 'ARMATURE')
check(arm.name.startswith("Hero") and "Pelvis_X" in arm.data.bones and "weapon_scripted" in arm.data.bones,
      "FBX file has the renamed bones and the rig name")
for o in imported:
    bpy.data.objects.remove(o)

# Units x100, UE IK bones, Godot names
res, info, path = run_export("ue.fbx", naming='UNREAL', units_x100=True, ue_ik_bones=True)
check(abs(info[0]["lengths"]["upperarm_l"] / rig.data.bones["DEF-upper_arm.L"].length - 100.0) < 0.5
      and abs(info[0]["unit"] - 0.01) < 1e-9, "units x100")
check(abs(scene.unit_settings.scale_length - 1.0) < 1e-9, "scene unit scale restored")
check({"ik_foot_root", "ik_foot_l", "ik_hand_gun", "ik_hand_r"} <= set(info[0]["parents"])
      and info[0]["parents"]["ik_hand_l"] == "ik_hand_gun", "UE IK bones")
res, info, path = run_export("godot.fbx", engine='GODOT', naming='GODOT', root_y_up=True, bake_actions=False)
parents = info[0]["parents"]
check({"Hips", "LeftUpperArm", "RightLowerLeg", "LeftIndexProximal", "Chest"} <= set(parents),
      "Godot humanoid names")
check(abs(info[0]["root_dir"][1] - 1.0) < 1e-4, f"Godot root Y up ({info[0]['root_dir']})")

# Selected objects / bones only
res, info, path = run_export("selected.fbx", select=[body], selected_objects=True, bake_actions=False)
check(info[0]["meshes"] == ["body_export"], f"selected objects only ({info[0]['meshes']})")
utils.set_mode(rig, 'POSE')
for pb in rig.pose.bones:
    utils.select_pose_bone(pb, pb.name in ("DEF-hips", "DEF-spine.01", "DEF-thigh.L"))
bpy.ops.object.mode_set(mode='OBJECT')
res, info, path = run_export("selected_bones.fbx", selected_bones=True, bake_actions=False)
check(set(info[0]["parents"]) == {"root", "hips", "spine.01", "thigh.L"},
      f"selected bones only ({sorted(info[0]['parents'])})")
for pb in rig.pose.bones:
    utils.select_pose_bone(pb, False)

# Dummy mesh
res, info, path = run_export("dummy.fbx", include_meshes=False, dummy_mesh=True, bake_actions=False)
check(info[0]["meshes"] == ["dummy"], "dummy mesh")

# Actions: only active, list with checkboxes, search, select all
res, info, path = run_export("active.fbx", only_active=True)
check(list(info[0]["actions"]) == ["walk"], "only the active action")
idle.rigger_export = False
res, info, path = run_export("all.fbx", only_active=False)
check(set(info[0]["actions"]) == {"walk", "run"}, f"ticked actions ({sorted(info[0]['actions'])})")
check(set(info[0]["tracks"]) == {"rig|walk", "rig|run"},
      f"take names rig|action ({info[0]['tracks']})")
res, info, path = run_export("search.fbx", only_active=False, action_search="ru", action_names='ACTION')
check(list(info[0]["actions"]) == ["run"] and info[0]["tracks"] == ["run"], "action search, action-only names")
res, info, path = run_export("sep.fbx", only_active=True, name_separator='_')
check(info[0]["tracks"] == ["rig_walk"], "name separator")
s.action_search = ""
bpy.ops.rigger.export_actions_select(state=True)
check(idle.rigger_export and walk.rigger_export, "select all actions")
idle.rigger_export = False

# Reset transforms between actions: 'run' does not key the arm
res, info, path = run_export("reset.fbx", only_active=False, reset_transforms=True)
arm_run = curve(info[0], "run", "upper_arm.L", "rotation_quaternion", 1)
check(arm_run and max(abs(k[1]) for k in arm_run) < 1e-4, "reset transforms between actions")

# Frame range: markers and manual, sampling rate
res, info, path = run_export("markers.fbx", only_active=True, frame_range='MARKERS')
check(info[0]["actions"]["walk"]["range"] == (3.0, 8.0), f"markers frame range ({info[0]['actions']['walk']['range']})")
res, info, path = run_export("manual.fbx", only_active=True, frame_range='MANUAL', frame_start=2, frame_end=5)
check(info[0]["actions"]["walk"]["range"] == (2.0, 5.0), "manual frame range")
res, info, path = run_export("sampling.fbx", only_active=True, sampling=2.0)
frames = [k[0] for k in curve(info[0], "walk", "upper_arm.L", "rotation_quaternion", 1)]
check(frames == [1.0, 3.0, 5.0, 7.0, 9.0], f"sampling rate ({frames})")

# One file per action, actions manager with lists
res, info, path = run_export("per_action.fbx", only_active=False, one_file_per='ACTION')
files = sorted(os.path.basename(i["path"]) for i in info)
check(files == ["per_action_run.fbx", "per_action_walk.fbx"] and all(os.path.exists(i["path"]) for i in info),
      f"one file per action ({files})")
for lst in list(s.action_lists):
    s.action_lists.remove(0)
bpy.ops.rigger.action_list_edit(mode='ADD_LIST')
s.action_search = "walk"
utils.set_mode(rig, 'OBJECT')
bpy.ops.rigger.action_list_edit(mode='ADD_ACTIONS')
bpy.ops.rigger.action_list_edit(mode='ADD_LIST')
s.action_lists[1].actions.add().action = run
s.action_lists[1].actions.add().action = idle
s.action_lists[0].name, s.action_lists[1].name = "locomotion", "misc"
check([i.action.name for i in s.action_lists[0].actions] == ["walk"], "action list from search")
saved_lists = [(l.name, [i.action for i in l.actions]) for l in s.action_lists]


def restore_lists():
    s.action_lists.clear()
    for name, acts in saved_lists:
        lst = s.action_lists.add()
        lst.name = name
        for a in acts:
            lst.actions.add().action = a


captured.clear()
s.use_actions_manager, s.one_file_per, s.action_search = True, 'LIST', ""
s.only_active = False
bpy.ops.rigger.export_fbx(filepath=os.path.join(OUT, "lists.fbx"))
files = sorted(os.path.basename(i["path"]) for i in captured)
check(files == ["lists_locomotion.fbx", "lists_misc.fbx"], f"one file per actions list ({files})")
misc = next(i for i in captured if i["path"].endswith("misc.fbx"))
check(set(misc["actions"]) == {"run", "idle"}, "actions manager exports the list's actions")
s.action_lists[1].enabled = False
captured.clear()
bpy.ops.rigger.export_fbx(filepath=os.path.join(OUT, "lists.fbx"))
check(len(captured) == 1, "disabled lists are skipped")
s.use_actions_manager = False
s.one_file_per = 'NONE'

# NLA export
rig.animation_data.action = None
track = rig.animation_data.nla_tracks.new()
track.strips.new("walk", 1, walk)
scene.frame_start, scene.frame_end = 1, 10
res, info, path = run_export("nla.fbx", anim_type='NLA')
check(list(info[0]["actions"]) == ["NLA"] and info[0]["actions"]["NLA"]["range"] == (1.0, 10.0), "NLA export")
rig.animation_data.nla_tracks.remove(track)
utils.assign_action(rig, walk)

# Actions linker: while 'walk' bakes, the prop rig plays 'prop_walk'
prop_data = bpy.data.armatures.new("prop_rig")
prop_rig = bpy.data.objects.new("prop_rig", prop_data)
scene.collection.objects.link(prop_rig)
utils.set_mode(prop_rig, 'EDIT')
grip = prop_data.edit_bones.new("grip")
grip.head, grip.tail = Vector((0.5, 0, 1)), Vector((0.5, 0, 1.1))
bpy.ops.object.mode_set(mode='OBJECT')
prop_walk = bpy.data.actions.new("prop_walk")
utils.assign_action(prop_rig, prop_walk)
for frame, x in ((1, 0.0), (10, 1.0)):
    prop_rig.pose.bones["grip"].location = (x, 0, 0)
    prop_rig.pose.bones["grip"].keyframe_insert("location", frame=frame)
prop_rig.animation_data.action = None
prop_rig.pose.bones["grip"].location = (0, 0, 0)
con = rig.pose.bones["weapon"].constraints.new('COPY_LOCATION')
con.target, con.subtarget = prop_rig, "grip"
s.links.clear()
link = s.links.add()
link.rig_a, link.action_a, link.rig_b, link.action_b = rig, walk, prop_rig, prop_walk
res, info, path = run_export("linked.fbx", only_active=True)
xs = curve(info[0], "walk", "weapon", "location", 0) + curve(info[0], "walk", "weapon", "location", 1) + \
    curve(info[0], "walk", "weapon", "location", 2)
check(xs and max(abs(k[1]) for k in xs) > 0.5, "actions linker plays the linked action while baking")
check(prop_rig.animation_data.action is None, "linked rig's action restored")
s.links.clear()
rig.pose.bones["weapon"].constraints.remove(con)

# glTF / GLB
res, info, path = run_export("hero.glb", file_format='GLB', only_active=True)
check(res == {'FINISHED'} and os.path.exists(os.path.join(OUT, "hero.glb")), "GLB export")
before = set(bpy.data.objects.keys())
bpy.ops.import_scene.gltf(filepath=os.path.join(OUT, "hero.glb"))
imported = [bpy.data.objects[n] for n in set(bpy.data.objects.keys()) - before]
arm = next((o for o in imported if o.type == 'ARMATURE'), None)
check(arm is not None and "upper_arm.L" in arm.data.bones, "GLB has the skeleton")
anims = {a.name for a in bpy.data.actions if "walk" in a.name and a not in (walk,)}
check(any(n.startswith("rig|walk") for n in anims), f"GLB has the take ({anims})")
for o in imported:
    bpy.data.objects.remove(o)
res, info, path = run_export("hero.gltf", file_format='GLTF_SEPARATE', bake_actions=False)
check(os.path.exists(os.path.join(OUT, "hero.gltf")) and os.path.exists(os.path.join(OUT, "hero.bin")),
      "glTF separate export")

# Mesh options pass through (smoothing, triangulate, modifiers, colors, textures, axes)
res, info, path = run_export("mesh_opts.fbx", smoothing='EDGE', triangulate=True, apply_modifiers=False,
                             vertex_colors='LINEAR', embed_textures=True, init_rotation=True, primary_axis='X',
                             secondary_axis='Y', simplify=1.0, bake_actions=False)
check(res == {'FINISHED'} and os.path.getsize(path) > 1000, "FBX mesh and axis options")

# Presets and quick export
s.engine, s.naming, s.units_x100 = 'UNITY', 'UNITY', True
bpy.ops.rigger.export_preset_save(name="unity_test")
s.engine, s.naming, s.units_x100 = 'UNREAL', 'RIGGER', False
bpy.ops.rigger.export_preset_load(preset="unity_test")
check(s.engine == 'UNITY' and s.naming == 'UNITY' and s.units_x100, "export preset load")
bpy.ops.rigger.export_preset_remove(preset="unity_test")
check(not os.path.exists(os.path.join(export._presets_dir(), "unity_test.json")), "export preset remove")
s.units_x100 = False
s.last_filepath = os.path.join(OUT, "quick.fbx")
captured.clear()
res = bpy.ops.rigger.quick_export()
check(res == {'FINISHED'} and os.path.exists(os.path.join(OUT, "quick.fbx")), "quick export")

# Check / fix rig
meta = utils.get_meta(rig)
prop_bone = rig.pose.bones[meta["limbs"][0]["prop_bone"]]
prop_bone["stretch_length"] = 1.3
body_mod = next(m for m in body.modifiers if m.type == 'ARMATURE')
body_mod.use_deform_preserve_volume = True
utils.set_mode(rig, 'OBJECT')
issues = export.check_rig(ctx, rig, s)
check(any("Stretch Length" in m for _l, m in issues) and any("Preserve Volume" in m for _l, m in issues),
      "check rig finds issues")
res = bpy.ops.rigger.check_rig(fix=True)
check(abs(prop_bone["stretch_length"] - 1.0) < 1e-6 and not body_mod.use_deform_preserve_volume, "fix rig")
check(not any("Stretch Length" in m for _l, m in export.check_rig(ctx, rig, s)), "no issue left after fixing")

export._write = _orig_write
# Shoulder volume: preserve volume everywhere but less around the shoulders
vol_body = capsule_body("vol_body")


def bind_vol(**kw):
    utils.set_mode(rig, 'OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    vol_body.select_set(True)
    rig.select_set(True)
    ctx.view_layer.objects.active = rig
    return bpy.ops.rigger.bind(method='PROXIMITY', **kw)


bind_vol()
mods = [m for m in vol_body.modifiers if m.type == 'ARMATURE']
vg = vol_body.vertex_groups.get(skin.VOLUME_GROUP)
check(len(mods) == 2 and not mods[0].use_deform_preserve_volume and mods[1].use_deform_preserve_volume
      and mods[1].use_multi_modifier and mods[1].vertex_group == skin.VOLUME_GROUP and vg is not None,
      "default bind mixes plain and volume-preserving deformation")
vp = points(vol_body)
shoulder = np.array(world_head("DEF-upper_arm.L"))
near = int(np.argmin(np.linalg.norm(vp - shoulder, axis=1)))
far = int(np.argmin(vp[:, 2]))
wv = weights(vol_body, skin.VOLUME_GROUP)
check(wv[near] < 0.5 and abs(wv[far] - 1.0) < 1e-3,
      f"shoulders preserve less volume ({wv[near]:.2f} at the shoulder, {wv[far]:.2f} at the feet)")
bind_vol(shoulder_volume=1.0)
mods = [m for m in vol_body.modifiers if m.type == 'ARMATURE']
check(len(mods) == 1 and mods[0].use_deform_preserve_volume and skin.VOLUME_GROUP not in vol_body.vertex_groups,
      "shoulder volume 1 gives the classic single modifier")
bind_vol()
bpy.ops.object.select_all(action='DESELECT')
vol_body.select_set(True)
ctx.view_layer.objects.active = vol_body
bpy.ops.rigger.unbind()
check(not any(m.type == 'ARMATURE' for m in vol_body.modifiers) and skin.VOLUME_GROUP not in vol_body.vertex_groups,
      "unbind removes both modifiers and the volume group")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("\nFAILURES: none")
