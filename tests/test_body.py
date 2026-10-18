"""Spine, neck, head, eyes, face, spline IK, skirt, bendy chain and template options.

    blender -b --factory-startup --python tests/test_body.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, templates, utils  # noqa: E402

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


def world(obj, name, tail=False):
    pb = obj.pose.bones[name]
    return obj.matrix_world @ (pb.tail if tail else pb.head)


def rot(obj, name):
    pb = obj.pose.bones[name]
    a = pb.matrix.to_quaternion().rotation_difference(obj.data.bones[name].matrix_local.to_quaternion()).angle
    return math.degrees(min(a, 2 * math.pi - a))


def rest_drift(obj):
    update(obj)
    worst, where = 0.0, None
    for b in obj.data.bones:
        if b.use_deform:
            pm = obj.pose.bones[b.name].matrix
            d = max((pm.translation - b.matrix_local.translation).length, math.radians(rot(obj, b.name)) * 0.1)
            if d > worst:
                worst, where = d, b.name
    return worst, where


def reset(obj):
    for pb in obj.pose.bones:
        pb.matrix_basis = Matrix()
    update(obj)


def q(angle, axis):
    return Matrix.Rotation(math.radians(angle), 4, axis).to_quaternion()


# -- Template options --------------------------------------------------------------------
opts = ctx.scene.rigger_template
opts.spine_count, opts.neck_count, opts.fingers, opts.toes = 5, 2, 4, True
opts.lips, opts.lids, opts.brows, opts.ears = 3, 3, 5, True
ref = templates.create_reference(ctx, templates.template_specs('HUMAN'), name="ref")
names = set(ref.data.bones.keys())
check({"spine.05", "neck.02", "toe_thumb.01.L", "lip_upper.03.L", "lid_upper.03.L", "brow.05.L", "ear.02.L"}
      <= names and "pinky.01.L" not in names, "template options: spine/neck counts, 4 fingers, toes, face counts")
for key in ("spine_count", "neck_count", "fingers", "toes", "lips", "lids", "brows", "ears"):
    opts.property_unset(key)

st = lambda n: ref.pose.bones[n].rigger  # noqa: E731
st("hips").spine_master = st("hips").bottom = st("hips").reversed_spine = True
st("neck.01").skulls = True
f = st("face")
f.muzzle = f.lips_offset = f.sticky_lips = f.lips_roll = f.soft_cheeks = f.brow_master = True
rig = generate.generate(ctx, ref)
pbs = rig.pose.bones
meta = utils.get_meta(rig)
drift, where = rest_drift(rig)
check(drift < 1e-4, f"no rest drift with every body/face option on ({drift:.6f} {where})")
new = {"spine_master", "bottom.L", "bottom.R", "DEF-bottom.L", "spine.05_rev", "skull_01", "skull_03",
       "muzzle", "lips_offset", "lips_roll_upper", "lips_roll_lower", "brow_master.L", "lid_upper_main.L",
       "lid_lower_main.R"}
check(new <= set(pbs.keys()), f"option bones generated (missing: {sorted(new - set(pbs.keys()))})")

# -- Spine master -------------------------------------------------------------------------------
top_rest = world(rig, "DEF-spine.05", tail=True)
pbs["spine_master"].rotation_quaternion = q(20, 'X')
update(rig)
check((world(rig, "DEF-spine.05", tail=True) - top_rest).length > 0.05, "spine master bends the spine")
reset(rig)
pbs["spine_master"].scale = (1, 1.3, 1)
update(rig)
check(world(rig, "DEF-spine.05", tail=True).z > top_rest.z + 0.05, "spine master stretches the spine")
check(pbs["DEF-spine.02"].matrix.to_scale().x < 0.95, "stretching squashes the spine")
reset(rig)

# -- Bottom bones ------------------------------------------------------------------------------------
legs = {(l["type"], l["side"]): l for l in meta["limbs"]}
leg_l = legs[("leg", "L")]
pbs[leg_l["prop_bone"]]["ik_fk"] = 0.0
pbs[leg_l["fk"][0]].rotation_quaternion = q(-60, 'X')
update(rig)
bottom = rot(rig, "DEF-bottom.L")
check(10 < bottom < 35, f"bottom bone follows the thigh partly ({bottom:.1f} deg)")
pbs[leg_l["fk"][0]].matrix_basis = Matrix()
reset(rig)

# -- Reversed spine -------------------------------------------------------------------------------------
torso = pbs[meta["spine"]["torso"]]
torso["spine_reverse"] = 1.0
update(rig)
check(rest_drift(rig)[0] < 1e-4, "reversed spine has no rest drift")
chest_rest, hips_rest = world(rig, "DEF-spine.05", tail=True), world(rig, "DEF-hips")
pbs["spine.05_rev"].rotation_quaternion = q(25, 'X')
update(rig)
check((world(rig, "DEF-spine.05", tail=True) - chest_rest).length < 1e-4
      and (world(rig, "DEF-hips") - hips_rest).length > 0.05, "reversed spine rotates from the chest down")
torso["spine_reverse"] = 0.0
reset(rig)

# -- Neck twist ----------------------------------------------------------------------------------------
head = meta["neck"]["head"]
pbs[head].rotation_quaternion = q(60, 'Y')
update(rig)
n1, n2 = rot(rig, "DEF-neck.01"), rot(rig, "DEF-neck.02")
check(10 < n1 < n2 < 50, f"neck bones twist with the head ({n1:.1f}, {n2:.1f})")
reset(rig)

# -- Skulls ---------------------------------------------------------------------------------------------
jaw_rest, eye_rest = world(rig, "DEF-jaw"), world(rig, "DEF-eye.L")
pbs["skull_01"].location = (0, 0, 0.02)
update(rig)
check((world(rig, "DEF-jaw") - jaw_rest).length > 0.01 and (world(rig, "DEF-eye.L") - eye_rest).length < 1e-4,
      "lower skull moves the jaw, not the eyes")
reset(rig)
pbs["skull_02"].location = (0, 0, 0.02)
update(rig)
check((world(rig, "DEF-eye.L") - eye_rest).length > 0.01, "middle skull moves the eyes")
reset(rig)

# -- Eyes -------------------------------------------------------------------------------------------------
eye = pbs["eye.L"]
eye["aim"] = 0.0
eye.rotation_quaternion = q(25, 'Z')
update(rig)
check(rot(rig, "DEF-eye.L") > 20, "eye aim 0: the eye rotates freely (FK)")
eye["aim"] = 1.0
update(rig)
check(rot(rig, "DEF-eye.L") < 5, "eye aim 1: the eye looks at its target")
reset(rig)
target = pbs["eyes_target"]
target["parent"] = 1  # Root
update(rig)
target_rest = world(rig, "eyes_target")
pbs[head].rotation_quaternion = q(30, 'Z')
update(rig)
check((world(rig, "eyes_target") - target_rest).length < 1e-4, "eye target can stay in place when the head turns")
target["parent"] = 0
reset(rig)

# -- Mouth offsets ------------------------------------------------------------------------------------------
lips = ["DEF-lip_upper", "DEF-lip_upper.02.L", "DEF-lip_lower.03.L", "DEF-lip_corner.L"]
rest = {n: world(rig, n) for n in lips + ["DEF-nose", "DEF-lid_upper.02.L"]}
pbs["lips_offset"].location = (0, 0, 0.01)
update(rig)
moves = [(world(rig, n) - rest[n]).length for n in lips]
check(all(abs(m - 0.01) < 0.002 for m in moves), f"lips offset moves every lip once ({[f'{m:.4f}' for m in moves]})")
check((world(rig, "DEF-nose") - rest["DEF-nose"]).length < 1e-5, "lips offset leaves the nose")
reset(rig)
pbs["muzzle"].location = (0, 0, 0.01)
update(rig)
moves = [(world(rig, n) - rest[n]).length for n in lips + ["DEF-nose"]]
check(all(abs(m - 0.01) < 0.002 for m in moves), f"muzzle moves mouth and nose once ({[f'{m:.4f}' for m in moves]})")
check((world(rig, "DEF-lid_upper.02.L") - rest["DEF-lid_upper.02.L"]).length < 1e-5, "muzzle leaves the eyes")
reset(rig)

# -- Lips retain and sticky lips -----------------------------------------------------------------------
jaw = pbs[meta["faces"][0]["jaw"]]
gap_rest = (world(rig, "DEF-lip_upper") - world(rig, "DEF-lip_lower")).length
jaw.rotation_quaternion = q(-20, 'X')  # -X opens the jaw (the chin moves down)
update(rig)
check(world(rig, "DEF-jaw", tail=True).z < (rig.matrix_world @ rig.data.bones["DEF-jaw"].tail_local).z,
      "negative jaw rotation opens the mouth")
gap_open = (world(rig, "DEF-lip_upper") - world(rig, "DEF-lip_lower")).length
jaw["lips_retain"] = 1.0
update(rig)
gap_retain = (world(rig, "DEF-lip_upper") - world(rig, "DEF-lip_lower")).length
check(gap_open > gap_rest + 0.005 and gap_retain < gap_rest + 0.002,
      f"lips retain keeps the lips sealed ({gap_rest:.4f} -> {gap_open:.4f} / {gap_retain:.4f})")
jaw["lips_retain"] = 0.0
up_rest = world(rig, "DEF-lip_upper")
jaw.rotation_quaternion = q(8, 'X')
update(rig)
check((world(rig, "DEF-lip_upper") - up_rest).length > 0.001, "sticky lips: closing the jaw further pushes the upper lip")
reset(rig)

# -- Lips roll, soft cheeks, eyebrow master --------------------------------------------------------------
pbs["lips_roll_upper"].rotation_quaternion = q(30, 'X')
update(rig)
check(rot(rig, "DEF-lip_upper") > 20 and rot(rig, "DEF-lip_corner.L") < 1, "lips roll rolls the upper lip")
reset(rig)
cheek_rest = world(rig, "DEF-cheek.L")
pbs["lip_corner.L"].location = (0, 0, 0.01)
update(rig)
check((world(rig, "DEF-cheek.L") - cheek_rest).length > 0.002, "soft cheeks follow the lip corner")
reset(rig)
brow = {n: world(rig, n) for n in ("DEF-brow.01.L", "DEF-brow.05.L")}
pbs["brow_master.L"].location = (0, 0, 0.01)
update(rig)
inner, outer = [(world(rig, n) - brow[n]).length for n in ("DEF-brow.01.L", "DEF-brow.05.L")]
check(inner > outer > 0.002, f"soft eyebrows: the master moves the inner end more ({inner:.4f}, {outer:.4f})")
reset(rig)

# -- Eyelids with main controls -----------------------------------------------------------------------------
face_ctrl = pbs[meta["faces"][0]["face"]]
gap = lambda i: (world(rig, f"DEF-lid_upper.0{i}.L") - world(rig, f"DEF-lid_lower.0{i}.L")).length  # noqa: E731
mid_rest, corner_rest = gap(2), gap(1)
face_ctrl["blink_L"] = 1.0
update(rig)
check(gap(2) < 0.3 * mid_rest, f"blink closes the middle lid ({mid_rest:.4f} -> {gap(2):.4f})")
check(corner_rest - gap(1) < mid_rest - gap(2), "lid corners move less than the middle")
face_ctrl["blink_L"] = 0.0
face_ctrl["auto_lid_upper"] = 0.0
eye.rotation_quaternion = q(20, 'X')
eye["aim"] = 0.0
update(rig)
lid0 = world(rig, "DEF-lid_upper.02.L").copy()
face_ctrl["auto_lid_upper"] = 0.6
update(rig)
check((world(rig, "DEF-lid_upper.02.L") - lid0).length > 1e-4, "auto eyelid amount is adjustable")
eye["aim"] = 1.0
reset(rig)

# -- Jaw from translation --------------------------------------------------------------------------------------
ref2 = templates.create_reference(ctx, templates.template_specs('HUMAN', options=None), name="ref2")
ref2.location = (2, 0, 0)
ref2.pose.bones["face"].rigger.jaw_translate = True
ref2.rigger.rig_name = "rig2"
rig2 = generate.generate(ctx, ref2)
check(rest_drift(rig2)[0] < 1e-4, "jaw by translation has no rest drift")
chin_rest = world(rig2, "DEF-jaw", tail=True)
rig2.pose.bones["jaw"].location = (0, 0, -0.02)
update(rig2)
check(world(rig2, "DEF-jaw", tail=True).z < chin_rest.z - 0.005, "moving the jaw control down opens the jaw")
rig2.pose.bones["jaw"]["jaw_speed"] = 2.0
update(rig2)
check(world(rig2, "DEF-jaw", tail=True).z < chin_rest.z - 0.012, "jaw speed opens it further")

# -- Spline IK: FK chain, twist and tail bone ---------------------------------------------------------------------
ref3 = templates.create_reference(ctx, templates.spline_chunk(), name="rope_ref")
ref3.location = (-2, 0, 1)
rs = ref3.pose.bones["rope.01"].rigger
rs.spline_fk = rs.spline_twist = rs.spline_tail = True
rs.spline_smooth = 3
ref3.rigger.rig_name = "rope"
rope = generate.generate(ctx, ref3)
rp = rope.pose.bones
check(rest_drift(rope)[0] < 1e-4, f"spline IK with FK, twist and tail has no rest drift {rest_drift(rope)}")
spline = utils.get_meta(rope)["splines"][0]
tip = spline["controls"][-1]
rp[tip].rotation_quaternion = q(90, 'Y')
update(rope)
first, last = rot(rope, "DEF-rope.01"), rot(rope, "DEF-rope.07")
check(first < 20 and last > 70, f"rotating the tip twists the chain progressively ({first:.1f}, {last:.1f})")
rp[tip].matrix_basis = Matrix()
rp[tip].location = (0, 0, 0.3)
update(rope)
check(abs(rp["DEF-rope.08"].matrix.to_scale().y - 1.0) < 1e-3, "tail bone is not stretched by the curve")
check((world(rope, "DEF-rope.08") - world(rope, "DEF-rope.07", tail=True)).length < 1e-3,
      "tail bone stays at the tip of the chain")
reset(rope)
rp[spline["master"]]["ik_fk"] = 0.0
rp[spline["fk"][0]].rotation_quaternion = q(30, 'X')
update(rope)
check(rot(rope, "DEF-rope.03") > 25, "spline FK chain drives the deform bones in FK")
check(spline["fk"] and rope.data.bones[spline["controls"][0]].hide, "IK controls hidden in FK")

# -- Skirt: collision, column and row masters -----------------------------------------------------------------------
ref4 = templates.create_reference(ctx, templates.template_specs('HUMAN', options=None), name="skirt_ref")
ref4.location = (4, 0, 0)
utils.set_mode(ref4, 'EDIT')
ref4.data.edit_bones.active = ref4.data.edit_bones["hips"]
ctx.scene.cursor.location = (4, 0, 0.98)
bpy.ops.rigger.add_module(module='SKIRT', side='C')
bpy.ops.object.mode_set(mode='OBJECT')
ks = ref4.pose.bones["skirt"].rigger
ks.skirt_masters = ks.skirt_row_masters = True
ref4.rigger.rig_name = "skirt_rig"
srig = generate.generate(ctx, ref4)
sp = srig.pose.bones
check(rest_drift(srig)[0] < 1e-4, f"skirt with collision and masters has no rest drift {rest_drift(srig)}")
smeta = utils.get_meta(srig)
skirt = smeta["skirts"][0]
front = skirt["strands"][0] if world(srig, skirt["strands"][0][0]).y < 0 else None
fronts = sorted(skirt["strands"], key=lambda s: world(srig, s[-1], tail=True).y)
front = fronts[0]
leg = next(l for l in smeta["limbs"] if l["type"] == "leg" and l["side"] == ("L" if world(srig, front[0]).x > 0 else "R"))
sp[leg["prop_bone"]]["ik_fk"] = 0.0
sp[leg["fk"][0]].rotation_quaternion = q(-75, 'X')
update(srig)
pushed = world(srig, front[-1], tail=True).y
sp[skirt["master"]]["collide"] = 0.0
update(srig)
free = world(srig, front[-1], tail=True).y
check(pushed < free - 0.01, f"thigh pushes the front strand forward ({free:.3f} -> {pushed:.3f})")
reset(srig)
sp[skirt["master"]]["collide"] = 1.0
strand_tip = world(srig, front[-1], tail=True)
sp[skirt["row_masters"][1]].rotation_quaternion = q(25, 'X')
update(srig)
check((world(srig, front[-1], tail=True) - strand_tip).length > 0.02, "row master rotates the row of strands")
reset(srig)
sp[skirt["col_masters"][0]].rotation_quaternion = q(25, 'X')
update(srig)
moved = sum((world(srig, s[-1], tail=True) - (srig.matrix_world @ srig.data.bones[s[-1]].tail_local)).length > 0.01
            for s in skirt["strands"])
check(2 <= moved < len(skirt["strands"]), f"column master drags its neighbours ({moved} strands)")

# -- Bendy chain and multi-ped torso ---------------------------------------------------------------------------------
ref5 = templates.create_reference(ctx, templates.template_specs('QUADRUPED'), name="dog_ref")
ref5.location = (-4, 0, 0)
ref5.pose.bones["tail.01"].rigger.module = 'BBONE'
ref5.rigger.multiped = True
ref5.rigger.rig_name = "dog"
dog = generate.generate(ctx, ref5)
dp = dog.pose.bones
check(rest_drift(dog)[0] < 1e-4, "bendy chain has no rest drift")
bendy = utils.get_meta(dog)["bendys"][0]
tail_rest = world(dog, "DEF-tail.04", tail=True)
dp[bendy["controls"][-1]].location = (0.05, 0, 0)
update(dog)
check((world(dog, "DEF-tail.04", tail=True) - tail_rest).length > 0.03 and dog.data.bones["DEF-tail.02"].bbone_segments > 1,
      "bendy chain follows its controls")
torso_dir = (dog.data.bones["torso"].tail_local - dog.data.bones["torso"].head_local).normalized()
check(abs(torso_dir.z) < 0.5, f"multi-ped torso follows the spine ({tuple(round(v, 2) for v in torso_dir)})")

print("FAILURES:", failures or "none")
if failures:
    sys.exit(1)
