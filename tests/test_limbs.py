"""Limb options: foot roll pivots and roll break, IK offset and pivot controls, upper twist,
secondary modes, joint fans with bulge, pole distance, soft IK, free roll, FK lock, smart pole,
3 bones legs, toes pivot, finger IK, grasp, rotate from scale, auto spread and IK/FK snapping.

    blender -b --factory-startup --python tests/test_limbs.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, rig_tools, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def update():
    rig.update_tag()
    ctx.view_layer.update()


def head(name):
    return rig.matrix_world @ pbs[name].head


def tail(name):
    return rig.matrix_world @ pbs[name].tail


def reset():
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix()
    for limb in meta["limbs"]:
        pbs[limb["prop_bone"]]["ik_fk"] = 1.0 if limb["type"] == "leg" else 0.0
    update()


def turned(pb):
    """World rotation of a pose bone since its rest pose."""
    return pb.matrix.to_quaternion() @ pb.bone.matrix_local.to_quaternion().inverted()


def angle_between(qa, qb):
    """Rotation angle between two orientations, folded into 0..pi."""
    a = qa.rotation_difference(qb).angle
    return min(a, 2 * math.pi - a)


def rest_drift():
    update()
    worst = 0.0
    for b in rig.data.bones:
        if b.use_deform:
            pm = pbs[b.name].matrix
            angle = pm.to_quaternion().rotation_difference(b.matrix_local.to_quaternion()).angle
            worst = max(worst, (pm.translation - b.matrix_local.translation).length,
                        min(angle, 2 * math.pi - angle) * 0.1)
    return worst


ref = templates.create_reference(ctx, templates.template_specs('HUMAN'), name="ref")
st = lambda name: ref.pose.bones[name].rigger  # noqa: E731
arm_l, arm_r, leg_l, leg_r = st("shoulder.L"), st("shoulder.R"), st("thigh.L"), st("thigh.R")
arm_l.twist_upper = 2
arm_l.secondary = False
arm_l.fans = arm_l.fans_root = arm_l.fans_end = True
arm_l.fan_count = 2
arm_l.fan_bulge = 0.5
arm_l.ik_offset = arm_l.ik_pivot = arm_l.fk_lock = True
arm_l.pole_distance = 1.0
arm_l.rot_from_scale = 'ALL'
arm_l.auto_spread = True
arm_r.soft_ik = True
arm_r.soft_ik_amount = 0.1
arm_r.twist_upper = 2  # with bendy secondaries, the default
st("index.01.L").finger_ik = True
st("middle.01.L").finger_shape = 'BOX'
leg_l.roll_break = leg_l.toes_pivot = leg_l.smart_pole = leg_l.ik_offset = leg_l.ik_pivot = True
leg_l.twist_upper = 1
leg_l.secondary_type = 'TWIST'
leg_r.secondary = False
leg_r.auto_ik_roll = False

rig = generate.generate(ctx, ref)
pbs = rig.pose.bones
meta = utils.get_meta(rig)
limbs = {(l["type"], l["side"]): l for l in meta["limbs"]}
arm, leg = limbs[("arm", "L")], limbs[("leg", "L")]
check(rest_drift() < 1e-4, f"no rest drift with every limb option on ({rest_drift():.6f})")

# -- Generated bones ------------------------------------------------------------------
names = set(pbs.keys())
expected = {"DEF-upper_arm_twist.01.L", "DEF-upper_arm_twist.02.L", "DEF-forearm_fan.01.L", "DEF-forearm_fan.02.L",
            "DEF-upper_arm_fan.01.L", "DEF-hand_fan.01.L", "hand_ik_offset.L", "hand_ik_pivot.L",
            "foot_ball.L", "foot_toes_pivot.L", "foot_roll.L", "foot_ik_offset.L", "foot_ik_pivot.L",
            "thigh_tweak.L", "thigh_tweak.01.L", "shin_tweak.L", "index_ik.L", "index_pole.L",
            "MCH-upper_arm_pole_line.L"}
check(expected <= names, f"option bones generated (missing: {sorted(expected - names)})")
check(not any(n.startswith(("thigh_tweak", "shin_tweak", "shin_bend")) and n.endswith(".R") for n in names),
      "no secondary controls with secondary off")
check("Smart" in meta["spaces"]["thigh_pole.L"], f"smart pole space ({meta['spaces']['thigh_pole.L']})")
check(rig.data.bones["MCH-upper_arm_pole_line.L"].hide_select, "pole line is not selectable")

# -- Pole distance ---------------------------------------------------------------------
ua, fa = ref.data.bones["upper_arm.L"], ref.data.bones["forearm.L"]
limb_len = ua.length + fa.length
pole_dist = (rig.data.bones["upper_arm_pole.L"].head_local - fa.head_local).length
check(abs(pole_dist - limb_len) < 0.02 * limb_len, f"IK pole distance follows the option ({pole_dist:.3f})")

# -- Free roll ----------------------------------------------------------------------------
ref_z = (ref.matrix_world.to_3x3() @ ref.data.bones["thigh.R"].matrix_local.to_3x3().col[2]).normalized()
def_z = rig.data.bones["DEF-thigh.R"].matrix_local.to_3x3().col[2].normalized()
check(ref_z.dot(def_z) > 0.999, "Auto IK Roll off keeps the reference roll")

# -- Foot roll ----------------------------------------------------------------------------
reset()
toe_tip, ball_rest, ankle_rest = tail("DEF-toe.L"), tail("DEF-foot.L"), head("DEF-foot.L")
foot_len = (ball_rest - ankle_rest).length
roll = pbs[leg["foot_roll"]]
roll.location = (0, -0.25 * foot_len, 0)
update()
check(tail("DEF-toe.L").z > toe_tip.z + 0.01, "roll cursor back lifts the toes (heel roll)")
check(tail("DEF-toe.L").z - toe_tip.z > 2 * (head("DEF-foot.L").z - ankle_rest.z),
      "heel roll pivots behind the ankle")
roll.location = (0, 0.15 * foot_len, 0)  # below the break point
update()
heel_up = head("DEF-foot.L").z - ankle_rest.z
check(heel_up > 0.005, f"roll forward lifts the heel ({heel_up:.4f})")
check((tail("DEF-foot.L") - ball_rest).length < 0.01, "before the break the foot rolls on the ball")
check(abs(tail("DEF-toe.L").z - toe_tip.z) < 0.005, "toes stay planted before the break")
roll.location = (0, 0.8 * foot_len, 0)  # past the break point
update()
check(tail("DEF-foot.L").z > ball_rest.z + 0.01, "past the break the ball lifts (tip toe)")
for x, pivot_name in ((0.3, "MCH-foot_bank_in.L"), (-0.3, "MCH-foot_bank_out.L")):
    roll.location = (x * foot_len, 0, 0)  # the cursor's X axis points inward on the left foot
    update()
    lift = head("DEF-foot.L").z - ankle_rest.z
    check(lift > 0.002 and abs(pbs[pivot_name].rotation_euler[1]) > 0.1,
          f"bank rolls the foot on its {'inner' if x > 0 else 'outer'} edge ({lift:.4f})")
roll.location = (0, 0, 0)
update()

# Ball control and toes pivot
pbs[leg["ball"]].rotation_quaternion = Matrix.Rotation(math.radians(25), 4, 'X').to_quaternion()
update()
check(head("DEF-foot.L").z > ankle_rest.z + 0.005 and (tail("DEF-foot.L") - ball_rest).length < 0.005,
      "ball control lifts the heel around the ball")
pbs[leg["ball"]].matrix_basis = Matrix()
pbs[leg["toes_pivot"]].rotation_quaternion = Matrix.Rotation(math.radians(-25), 4, 'X').to_quaternion()
update()
check((tail("DEF-foot.L") - ball_rest).length < 0.005 and (tail("DEF-toe.L") - toe_tip).length > 0.005,
      "toes pivot rotates the whole foot, toes included, from the ball")
pbs[leg["toes_pivot"]].matrix_basis = Matrix()

# -- IK pivot and offset ---------------------------------------------------------------------
reset()
pivot = pbs[leg["ik_pivot"]]
pivot.location = (0, 0.3, 0)  # along the IK control's Y axis (forward)
update()
check((head("DEF-foot.L") - ankle_rest).length < 1e-4, "moving the pivot alone does not move the foot")
pivot_world = (rig.matrix_world @ pivot.matrix).translation
pivot.rotation_quaternion = Matrix.Rotation(math.radians(30), 4, 'X').to_quaternion()
update()
expected_ankle = pivot_world + Matrix.Rotation(math.radians(30), 3, (rig.matrix_world.to_3x3()
                                                                    @ pivot.matrix.to_3x3()).col[0]) @ (ankle_rest - pivot_world)
check((head(leg["ik_target"]) - expected_ankle).length < 0.005, "rotating the pivot turns the foot around the pivot")
pivot.matrix_basis = Matrix()
pbs[leg["ik_offset"]].location = (0, 0, 0.05)
update()
check((head("DEF-foot.L") - ankle_rest).length > 0.02, "IK offset control moves the foot")
reset()

# -- Smart IK pole -----------------------------------------------------------------------------
pole = pbs[leg["pole"]]
pole["parent"] = 0  # Smart
update()
pole_rest = head(leg["pole"])
foot = pbs[leg["ik_ctrl"]]
foot.rotation_quaternion = Matrix.Rotation(math.radians(40), 4, 'X').to_quaternion()
update()
check((head(leg["pole"]) - pole_rest).length < 1e-3, "smart pole ignores foot tilt")
foot.matrix_basis = Matrix()
foot.rotation_quaternion = Matrix.Rotation(math.radians(40), 4, 'Z').to_quaternion()
update()
check((head(leg["pole"]) - pole_rest).length > 0.02, "smart pole follows the foot heading")
reset()

# -- Upper twist --------------------------------------------------------------------------------
reset()
fk_upper = pbs[arm["fk"][0]]
fk_upper.rotation_quaternion = Matrix.Rotation(math.radians(80), 4, 'Y').to_quaternion()
update()


def twist(name):
    rest = rig.data.bones[name].matrix_local.to_3x3()
    now = pbs[name].matrix.to_3x3()
    return math.degrees(rest.col[0].angle(now.col[0]))


t0, t1, t2 = twist("DEF-upper_arm.L"), twist("DEF-upper_arm_twist.01.L"), twist("DEF-upper_arm_twist.02.L")
check(t0 < t1 < t2 < 80.5, f"upper arm twist grows toward the elbow ({t0:.1f}, {t1:.1f}, {t2:.1f})")
check(t0 < 2.0 and t2 > 78.0, f"upper arm twist: none at the shoulder piece, all of it at the elbow piece "
                               f"({t0:.1f}, {t2:.1f} of 80)")

# Lower twist: from none at the elbow piece to all of the hand's twist at the wrist piece
reset()
pbs[arm["fk"][2]].rotation_quaternion = Matrix.Rotation(math.radians(90), 4, 'Y').to_quaternion()
update()
lower = ["DEF-forearm.L"] + sorted(n for n in pbs.keys() if n.startswith("DEF-forearm_twist.") and n.endswith(".L"))
angles = [twist(n) for n in lower]
check(angles[0] < 2.0 and abs(angles[-1] - 90.0) < 2.0 and all(a < b for a, b in zip(angles, angles[1:])),
      f"forearm twist spreads from 0 to the full hand twist ({', '.join(f'{a:.0f}' for a in angles)})")
reset()

# -- Twist secondary mode ---------------------------------------------------------------------------
before = tail("DEF-thigh.L").copy()
shin_before = head("DEF-shin.L").copy()
pbs["thigh_tweak.L"].location = (0.03, 0, 0)
update()
check((tail("DEF-thigh.L") - before).length > 0.01, "twist tweak moves its deform bone")
check((head("DEF-shin.L") - shin_before).length < 1e-4, "twist tweak does not move the next segment")
reset()

# -- Joint fans with bulge ---------------------------------------------------------------------------
fk_lower = pbs[arm["fk"][1]]
fk_lower.rotation_quaternion = Matrix.Rotation(math.radians(90), 4, 'X').to_quaternion()
update()
fans = [rig.pose.bones["DEF-forearm_fan.01.L"], rig.pose.bones["DEF-forearm_fan.02.L"]]
up_q, low_q = pbs["DEF-upper_arm_twist.02.L"].matrix.to_quaternion(), pbs["DEF-forearm.L"].matrix.to_quaternion()
bend = up_q.rotation_difference(low_q).angle
shares = [up_q.rotation_difference(f.matrix.to_quaternion()).angle / bend for f in fans]
check(abs(shares[0] - 1 / 3) < 0.12 and abs(shares[1] - 2 / 3) < 0.12,
      f"two fans share the elbow bend ({shares[0]:.2f}, {shares[1]:.2f})")
moved = (fans[0].matrix.translation - (pbs["DEF-forearm.L"].matrix.translation)).length
check(moved > 0.005, f"fan bulge pushes the fans out ({moved:.4f})")
reset()

# -- Upper twist bones with bendy secondaries -----------------------------------------------------
check("DEF-upper_arm_twist.02.R" in pbs and "MCH-forearm_joint_rel.R" in pbs,
      "bendy arm keeps its upper twist bones")
fk_up = pbs[limbs[("arm", "R")]["fk"][0]]
fk_up.rotation_mode = 'QUATERNION'
fk_up.rotation_quaternion = Matrix.Rotation(math.radians(60), 4, 'Y').to_quaternion()
update()
twisted = [angle_between(turned(pbs[n]), turned(fk_up)) for n in
           ("DEF-upper_arm.R", "DEF-upper_arm_twist.01.R", "DEF-upper_arm_twist.02.R")]
check(twisted[0] > twisted[1] > twisted[2],
      "bendy upper twist: the pieces near the shoulder twist less "
      f"({', '.join(f'{math.degrees(a):.0f}' for a in twisted)} deg left)")
reset()

# -- Bendy joint: sharp by default, rounded on demand -----------------------------------------------
bend_ctrl = limbs[("arm", "R")]["tweaks"][1]
upper_piece = "DEF-upper_arm_twist.02.R"
check(rig.data.bones[upper_piece].bbone_easeout == 0.0 and rig.data.bones["DEF-forearm.R"].bbone_easein == 0.0,
      "no bendy easing at the elbow at rest")
pbs[bend_ctrl].scale = (1, 2, 1)
update()
check(abs(pbs[upper_piece].bbone_easeout - 1.0) < 1e-4 and abs(pbs["DEF-forearm.R"].bbone_easein - 1.0) < 1e-4,
      "scaling the joint tweak rounds the elbow")
reset()

# -- FK lock -------------------------------------------------------------------------------------------
torso = pbs[meta["spine"]["torso"]]
fk_upper = pbs[arm["fk"][0]]
fk_upper["fk_lock"] = 0.0
update()
rest_dir = (tail(arm["fk"][0]) - head(arm["fk"][0])).normalized()
torso.rotation_quaternion = Matrix.Rotation(math.radians(30), 4, 'Z').to_quaternion()
update()
free_dir = (tail(arm["fk"][0]) - head(arm["fk"][0])).normalized()
check(math.degrees(rest_dir.angle(free_dir)) < 1.0, "fk_lock 0: the FK arm keeps its world rotation")
fk_upper["fk_lock"] = 1.0
update()
locked_dir = (tail(arm["fk"][0]) - head(arm["fk"][0])).normalized()
check(math.degrees(rest_dir.angle(locked_dir)) > 20.0, "fk_lock 1: the FK arm follows the body")
reset()

# -- Soft IK -------------------------------------------------------------------------------------------
arm_r_meta = limbs[("arm", "R")]
pbs[arm_r_meta["prop_bone"]]["ik_fk"] = 1.0
hand = pbs[arm_r_meta["ik_ctrl"]]
shoulder = head(arm_r_meta["blend"][0])
reach = sum(rig.data.bones[n].length for n in arm_r_meta["fk"][:2])
target_dir = (head(arm_r_meta["ik_ctrl"]) - shoulder).normalized()
mat = hand.matrix.copy()
mat.translation = rig.matrix_world.inverted() @ (shoulder + target_dir * reach * 0.995)
hand.matrix = mat
update()
elbow_angle = math.degrees((head(arm_r_meta["blend"][1]) - shoulder).angle(
    tail(arm_r_meta["blend"][1]) - head(arm_r_meta["blend"][1])))
reached = (tail(arm_r_meta["blend"][1]) - head(arm_r_meta["ik_ctrl"])).length
check(elbow_angle > 3.0, f"soft IK keeps the elbow bent near full reach ({elbow_angle:.1f} deg)")
check(reached < 0.01 * reach, f"soft IK still reaches the hand control ({reached:.4f})")
reset()

# -- Fingers --------------------------------------------------------------------------------------------
tip_rest = tail("DEF-index.03.L").copy()
pbs[arm["prop_bone"]]["fingers_grasp"] = 1.0
update()
check(tail("DEF-index.03.L").z < tip_rest.z - 0.01, "fingers grasp curls the fingers toward the palm")
pbs[arm["prop_bone"]]["fingers_grasp"] = 0.0
pbs["middle.01.L"].scale = (1.0, 0.5, 1.0)
update()
check(tail("DEF-middle.03.L").z < tail("DEF-index.03.L").z - 0.005, "scaling the first phalange curls the finger")
check(abs(pbs["DEF-middle.01.L"].matrix.to_scale().y - 1.0) < 1e-3, "rotate from scale keeps the bone length")
pbs["middle.01.L"].scale = (1, 1, 1)
pbs["pinky.01.L"].rotation_quaternion = Matrix.Rotation(math.radians(30), 4, 'Z').to_quaternion()
update()
ring_rot = math.degrees(pbs["ring.01.L"].matrix.to_quaternion().rotation_difference(
    (rig.data.bones["ring.01.L"].matrix_local).to_quaternion()).angle)
check(15.0 < ring_rot < 25.0, f"auto spread rotates the ring finger along ({ring_rot:.1f} deg)")
reset()
check(rig.pose.bones["middle.01.L"].custom_shape.name.startswith("WGT-rigger_box"), "box finger shapes")

# Finger IK
ik_ctrl = pbs["index_ik.L"]
ik_ctrl["ik_fk"] = 1.0
ik_ctrl.location = (0, 0, -0.02)
update()
check((tail("DEF-index.03.L") - head("index_ik.L")).length < 0.004, "finger IK reaches the IK control")
ik_ctrl.matrix_basis = Matrix()
ik_ctrl["ik_fk"] = 0.0
update()

# -- IK/FK snapping round trip with every option -------------------------------------------------------
reset()
pbs[leg["prop_bone"]]["ik_fk"] = 1.0
foot = pbs[leg["ik_ctrl"]]
foot.location = (0.05, -0.1, 0.2)
foot.rotation_quaternion = Matrix.Rotation(math.radians(20), 4, 'X').to_quaternion()
update()
before = {n: head(n).copy() for n in ("DEF-foot.L", "DEF-shin.L", "DEF-toe.L")}
rig_tools.snap_fk_to_ik(ctx, rig, leg)
update()
err = max((head(n) - p).length for n, p in before.items())
check(pbs[leg["prop_bone"]]["ik_fk"] == 0.0 and err < 1e-3, f"snap FK to IK keeps the pose ({err:.5f})")
foot.matrix_basis = Matrix()
rig_tools.snap_ik_to_fk(ctx, rig, leg)
update()
err = max((head(n) - p).length for n, p in before.items())
check(pbs[leg["prop_bone"]]["ik_fk"] == 1.0 and err < 2e-3, f"snap IK to FK keeps the pose ({err:.5f})")
reset()

# -- 3 bones leg (horse hind legs) ----------------------------------------------------------------------
horse = templates.create_reference(ctx, templates.template_specs('HORSE'), name="horse_ref")
horse.location = (3, 0, 0)
for side in "LR":
    horse.pose.bones[f"thigh.{side}"].rigger.three_bones = True
horse.rigger.rig_name = "horse"
hrig = generate.generate(ctx, horse)
hmeta = {(l["type"], l["side"], l["ik_ctrl"]): l for l in utils.get_meta(hrig)["limbs"]}
hind = next(l for k, l in hmeta.items() if k[0] == "leg" and k[1] == "L" and l.get("three"))
check(len(hind["fk"]) == 4 and hind["ik_rot"] == "thigh_ik.L", f"3 bones leg chain ({hind['fk']})")
hp = hrig.pose.bones
hp[hind["prop_bone"]]["ik_fk"] = 1.0
hrig.update_tag()
ctx.view_layer.update()
drift = max((hp[n].matrix.translation - hrig.data.bones[n].matrix_local.translation).length
            for n in hind["deform"])
check(drift < 1e-4, f"3 bones leg has no rest drift in IK ({drift:.6f})")
hp[hind["ik_ctrl"]].location = (0, 0, 0.15)
for three in (0.0, 1.0):
    hp[hind["prop_bone"]]["three_ik"] = three
    hrig.update_tag()
    ctx.view_layer.update()
    end = hrig.matrix_world @ hp[hind["deform"][-2]].tail
    goal = hrig.matrix_world @ hp[hind["ik_target"]].head
    check((end - goal).length < 0.01, f"3 bones leg reaches the foot with three_ik={three} "
                                      f"({(end - goal).length:.4f})")
hp[hind["ik_rot"]].rotation_quaternion = Matrix.Rotation(math.radians(20), 4, 'X').to_quaternion()
hp[hind["prop_bone"]]["three_ik"] = 0.0
hrig.update_tag()
ctx.view_layer.update()
thigh_b_moved = (hp[hind["deform"][0]].matrix.to_quaternion().rotation_difference(
    hrig.data.bones[hind["deform"][0]].matrix_local.to_quaternion()).angle)
check(thigh_b_moved > math.radians(10), "the IK rotation control turns the upper bone")

print("FAILURES:", failures or "none")
if failures:
    sys.exit(1)
