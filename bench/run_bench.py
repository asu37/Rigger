"""Deformation benchmark: rig real human bodies with Rigger, pose them, measure.

    python bench/fetch_assets.py          # once, downloads the CC0 bodies
    blender -b --factory-startup --python bench/run_bench.py -- [options]

Options (after the "--"):
    --bodies a,b        subset of bench/bodies.py BODIES (default: all)
    --configs a,b       subset of CONFIGS below (default: all)
    --poses a,b         subset of POSES below (default: all)
    --render            write a heat-map render per pose (slow: Cycles on the CPU)
    --out DIR           output folder (default: bench/results/latest)
    --compare FILE      results.json of an earlier run: adds a before/after table

Each body gets the frozen Smart Marker positions from bench/landmarks.json, so
only rig generation, skinning and deformation are measured. Poses are applied
to the FK controls of the character's left side. See bench/README.md for what
every number means.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import markers, utils  # noqa: E402

import bodies  # noqa: E402
import metrics  # noqa: E402

FWD, BACK, UP = Vector((0, -1, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
DOWN, OUT, IN = Vector((0, 0, -1)), Vector((1, 0, 0)), Vector((-1, 0, 0))  # OUT: away from the body (left side)


def _dir(*vs):
    v = sum(vs, Vector())
    return v.normalized()

# Rig and bind settings to compare. "ref" values go on the reference's arm and
# leg modules before generating; "bind" values are passed to rigger.bind.
QUALITY = {"twist": 3, "twist_upper": 2, "fans": True, "fans_root": True, "fans_end": True}
CONFIGS = {
    # Out of the box: bendy-bone secondaries, 2 lower and 1 upper twist bones, Preserve Volume on
    # (30% at the shoulders).
    "default": {"ref": {}, "bind": {}},
    # Joints from template proportions only (before mesh-based joint fitting).
    "no_joint_fit": {"ref": {}, "bind": {}, "fit_joints": False},
    # Weight corrections as they were: hip cut on, no head/shoulder or finger cleanup.
    "old_weight_defaults": {"ref": {}, "bind": {"improve_hips": True, "refine_head": False,
                                                "clean_fingers": False}},
    # Same rig, plain linear blend skinning: shows what Preserve Volume hides.
    "no_preserve_volume": {"ref": {}, "bind": {"preserve_volume": False}},
    # Without elbow/knee fans, and with fans weighted by heat alone (not tightened).
    "no_fans": {"ref": {"fans": False}, "bind": {}},
    # Markers guessed by the AI pose model (needs RIGGER_AI_DIR, see ai_markers_eval.py).
    "ai_markers": {"ref": {}, "bind": {}, "ai_markers": True},
    "loose_fans": {"ref": {}, "bind": {"tighten_fans": False}},
    # Preserve Volume at full strength on the shoulders too (no per-vertex mix).
    "full_preserve_volume": {"ref": {}, "bind": {"shoulder_volume": 1.0}},
    # Default with twist secondaries instead of bendy bones (no fans): separates
    # the effect of bendy bones from the effect of fans.
    "twist_secondary": {"ref": {"secondary_type": 'TWIST'}, "bind": {}},
    # More twist bones and fans. With bendy secondaries Rigger ignores twist_upper.
    "quality_bendy": {"ref": dict(QUALITY), "bind": {"refine_head": True}},
    # Same with twist secondaries, which keep the upper twist bones.
    "quality_twist": {"ref": dict(QUALITY, secondary_type='TWIST'), "bind": {"refine_head": True}},
}

# Measuring skeleton: every rig is measured against the same anatomy, taken from
# the mesh (frozen markers + elbow, knee and hip found by rigger.fitting), so moving
# the rig's own bones does not move the yardstick. Each measuring point follows one
# DEF bone rigidly when posed.
#
# Joint name -> (parent point, joint point, child point) as (anatomy point, DEF bone
# stem it follows), and the region radius as a fraction of body height.
JOINTS = {
    "elbow": (("shoulder", "upper_arm"), ("elbow", "upper_arm"), ("wrist", "hand"), 0.07),
    "wrist": (("elbow", "forearm"), ("wrist", "forearm"), ("hand_end", "hand"), 0.045),
    "shoulder": (("neck_side", "shoulder"), ("shoulder", "shoulder"), ("elbow", "upper_arm"), 0.10),
    "hip": (("pelvis", "hips"), ("hip", "hips"), ("knee", "thigh"), 0.11),
    "knee": (("hip", "thigh"), ("knee", "thigh"), ("ankle", "foot"), 0.08),
    "waist": (("pelvis", "hips"), ("mid_spine", "spine.02"), ("neck", "spine.03"), 0.12),
    "ankle": (("knee", "shin"), ("ankle", "shin"), ("toe", "foot"), 0.05),
    "neck": (("mid_spine", "spine.03"), ("neck", "neck"), ("chin", "head"), 0.07),
}

# Twist checks: limb axis (start, end) in anatomy points
TWISTS = {
    "forearm": ("elbow", "wrist"),
    "upper_arm": ("shoulder", "elbow"),
    "neck": ("neck", "chin"),
    "spine": ("pelvis", "neck"),
}

# Pose -> (steps, measured joints, measured twist, render view)
# A step bends a control toward a direction (bend) or turns it about a limb axis (twist).
POSES = {
    "elbow_90": ([("bend", "forearm_fk", "forearm", FWD, 90)], ["elbow"], None, "side"),
    "elbow_135": ([("bend", "forearm_fk", "forearm", FWD, 135)], ["elbow"], None, "side"),
    "wrist_60": ([("bend", "hand_fk", "hand", FWD, 60)], ["wrist"], None, "side"),
    "shoulder_raise_90": ([("bend", "upper_arm_fk", "upper_arm", UP, 90)], ["shoulder"], None, "front"),
    "shoulder_raise_135": ([("bend", "upper_arm_fk", "upper_arm", UP, 135)], ["shoulder"], None, "front"),
    "shoulder_forward_90": ([("bend", "upper_arm_fk", "upper_arm", FWD, 90)], ["shoulder"], None, "side"),
    "hip_forward_90": ([("bend", "thigh_fk", "thigh", FWD, 90)], ["hip"], None, "side_cut"),
    "knee_90": ([("bend", "shin_fk", "shin", BACK, 90)], ["knee"], None, "side_cut"),
    "knee_135": ([("bend", "shin_fk", "shin", BACK, 135)], ["knee"], None, "side_cut"),
    "squat": ([("bend", "thigh_fk", "thigh", FWD, 100), ("bend", "shin_fk", "shin", BACK, 130)],
              ["hip", "knee"], None, "side_cut"),
    "spine_forward_45": ([("bend", "spine.01", "spine.01", FWD, 15), ("bend", "spine.02", "spine.02", FWD, 15),
                          ("bend", "spine.03", "spine.03", FWD, 15)], ["waist"], None, "side"),
    "forearm_twist_90": ([("twist", "hand_fk", "forearm", None, 90)], [], "forearm", "front"),
    "upper_arm_twist_60": ([("twist", "upper_arm_fk", "upper_arm", None, 60)], [], "upper_arm", "front"),
    "neck_turn_70": ([("twist", "head", "neck", None, 70)], [], "neck", "front"),
}


def _bend(ctrl, limb, toward, deg):
    return ("bend", ctrl, limb, toward, deg)


def _twist(ctrl, limb, deg):
    return ("twist", ctrl, limb, None, deg)


# 3D sweep: every joint bent in several directions and amounts, plus combined poses.
SWEEP = {}
for label, toward, angles in (("up", UP, (60, 120)), ("fwd", FWD, (60, 120)), ("back", BACK, (30, 45)),
                              ("out", OUT, (45, 90)), ("across", _dir(IN, FWD), (45, 80)),
                              ("up_fwd", _dir(UP, FWD), (60, 120)), ("up_back", _dir(UP, BACK), (45, 90)),
                              ("out_fwd", _dir(OUT, FWD), (60, 110))):
    for a in angles:
        SWEEP[f"shoulder_{label}_{a}"] = ([_bend("upper_arm_fk", "upper_arm", toward, a)], ["shoulder"], None, "front")
SWEEP["elbow_45"] = ([_bend("forearm_fk", "forearm", FWD, 45)], ["elbow"], None, "side")
SWEEP["elbow_150"] = ([_bend("forearm_fk", "forearm", FWD, 150)], ["elbow"], None, "side")
SWEEP["elbow_90_twist_90"] = ([_bend("forearm_fk", "forearm", FWD, 90), _twist("hand_fk", "forearm", 90)],
                              ["elbow"], None, "side")  # the twist measure assumes an unbent limb
SWEEP["wrist_back_60"] = ([_bend("hand_fk", "hand", BACK, 60)], ["wrist"], None, "side")
SWEEP["wrist_out_30"] = ([_bend("hand_fk", "hand", OUT, 30)], ["wrist"], None, "front")
SWEEP["wrist_in_30"] = ([_bend("hand_fk", "hand", IN, 30)], ["wrist"], None, "front")
for label, toward, a in (("fwd_45", FWD, 45), ("fwd_120", FWD, 120), ("back_30", BACK, 30), ("out_45", OUT, 45),
                         ("in_20", IN, 20), ("out_fwd_70", _dir(OUT, FWD), 70)):
    SWEEP[f"hip_{label}"] = ([_bend("thigh_fk", "thigh", toward, a)], ["hip"], None, "side_cut")
SWEEP["knee_45"] = ([_bend("shin_fk", "shin", BACK, 45)], ["knee"], None, "side_cut")
SWEEP["knee_150"] = ([_bend("shin_fk", "shin", BACK, 150)], ["knee"], None, "side_cut")
for label, toward, a in (("up_20", UP, 20), ("down_40", DOWN, 40), ("in_20", IN, 20), ("out_15", OUT, 15)):
    SWEEP[f"ankle_{label}"] = ([_bend("foot_fk", "foot", toward, a)], ["ankle"], None, "side_cut")
SWEEP["spine_back_30"] = ([_bend(f"spine.0{i}", f"spine.0{i}", BACK, 10) for i in (1, 2, 3)], ["waist"], None, "side")
SWEEP["spine_side_30"] = ([_bend(f"spine.0{i}", f"spine.0{i}", OUT, 10) for i in (1, 2, 3)], ["waist"], None, "front")
SWEEP["spine_twist_45"] = ([_twist(f"spine.0{i}", f"spine.0{i}", 15) for i in (1, 2, 3)], ["waist"], "spine", "front")
SWEEP["neck_nod_40"] = ([_bend("head", "head", FWD, 40)], ["neck"], None, "side")
SWEEP["neck_back_30"] = ([_bend("head", "head", BACK, 30)], ["neck"], None, "side")
SWEEP["neck_tilt_30"] = ([_bend("head", "head", OUT, 30)], ["neck"], None, "front")
SWEEP["reach_up_fwd"] = ([_bend("upper_arm_fk", "upper_arm", _dir(UP, FWD), 110), _bend("forearm_fk", "forearm", FWD, 30)],
                         ["shoulder", "elbow"], None, "side")
SWEEP["arm_across"] = ([_bend("upper_arm_fk", "upper_arm", _dir(IN, FWD), 80), _bend("forearm_fk", "forearm", FWD, 100)],
                       ["shoulder", "elbow"], None, "front")
SWEEP["squat_twist"] = ([_bend("thigh_fk", "thigh", FWD, 100), _bend("shin_fk", "shin", BACK, 130)]
                        + [_twist(f"spine.0{i}", f"spine.0{i}", 10) for i in (1, 2, 3)],
                        ["hip", "knee", "waist"], None, "side_cut")
SWEEP["crouch_reach"] = ([_bend("thigh_fk", "thigh", FWD, 100), _bend("shin_fk", "shin", BACK, 120),
                          _bend("upper_arm_fk", "upper_arm", FWD, 90)], ["hip", "knee", "shoulder"], None, "side_cut")
POSES.update(SWEEP)

# ---------------------------------------------------------------------------
# Scene setup
# ---------------------------------------------------------------------------

def build(name, config):
    """Load a body, fit the reference to its markers, generate and bind. Returns (body, rig, bind seconds)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ctx = bpy.context
    body, eyes = bodies.load_body(name)
    for eye in eyes:  # the benchmark measures the body skin only
        bpy.data.objects.remove(eye)
    if config.get("ai_markers"):  # markers guessed by the pose model instead of the frozen landmarks
        from rigger import ai_markers
        placed, _conf, _kp = ai_markers.guess(ctx, [body])
        for key, point in placed.items():
            markers.set_marker(ctx, key, point)
    else:
        for key, point in bodies.landmarks(name).items():
            markers.set_marker(ctx, key, Vector(point))
    utils.deselect_all(ctx)
    body.select_set(True)
    ctx.view_layer.objects.active = body
    fit = {"fit_joints": config["fit_joints"]} if "fit_joints" in config else {}
    bpy.ops.rigger.fit_markers(fingers=True, face=False, **fit)
    ref = ctx.object
    for pb in ref.pose.bones:
        if pb.rigger.module in {'ARM', 'LEG'}:
            for key, value in config["ref"].items():
                setattr(pb.rigger, key, value)
    utils.set_mode(ref, 'OBJECT')
    if bpy.ops.rigger.generate() != {'FINISHED'}:
        raise RuntimeError("generate failed")
    rig = ctx.object
    utils.set_mode(rig, 'OBJECT')
    if config.get("post"):  # experiments: adjust the generated rig before binding
        config["post"](rig)
    utils.deselect_all(ctx)
    body.select_set(True)
    rig.select_set(True)
    ctx.view_layer.objects.active = rig
    start = time.time()
    if bpy.ops.rigger.bind(**config["bind"]) != {'FINISHED'}:
        raise RuntimeError("bind failed")
    seconds = time.time() - start
    if config.get("post_bind"):  # experiments: adjust the binding
        config["post_bind"](body, rig)
    for limb in utils.get_meta(rig)["limbs"]:
        rig.pose.bones[limb["prop_bone"]]["ik_fk"] = 0.0  # FK everywhere
    ctx.view_layer.update()
    return body, rig, seconds


def reset_pose(rig):
    for pb in rig.pose.bones:
        pb.location = (0, 0, 0)
        pb.rotation_quaternion = (1, 0, 0, 0)
        pb.rotation_euler = (0, 0, 0)
        pb.rotation_axis_angle = (0, 0, 1, 0)
        pb.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def def_name(rig, stem, side="L"):
    for name in (f"DEF-{stem}.{side}", f"DEF-{stem}"):
        if name in rig.data.bones:
            return name
    raise KeyError(stem)


def ctrl_name(rig, stem, side="L"):
    for name in (f"{stem}.{side}", stem):
        if name in rig.pose.bones:
            return name
    raise KeyError(stem)


def point(rig, ref, posed):
    stem, end = ref
    name = def_name(rig, stem)
    if posed:
        pb = rig.pose.bones[name]
        v = pb.head if end == "head" else pb.tail
    else:
        b = rig.data.bones[name]
        v = b.head_local if end == "head" else b.tail_local
    return np.array(rig.matrix_world @ v)


def apply_step(rig, step):
    kind, ctrl, limb_stem, toward, degrees = step
    pb = rig.pose.bones[ctrl_name(rig, ctrl)]
    bpy.context.view_layer.update()
    deform = rig.pose.bones[def_name(rig, limb_stem)]
    direction = (deform.tail - deform.head).normalized()
    if kind == "bend":
        axis = direction.cross(toward)
        if axis.length < 1e-6:
            axis = direction.orthogonal()
        axis.normalize()
    else:
        axis = direction
    angle = math.radians(degrees)
    head = pb.matrix.translation.copy()
    pb.matrix = Matrix.Translation(head) @ Matrix.Rotation(angle, 4, axis) @ Matrix.Translation(-head) @ pb.matrix
    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------

def vertices(body, evaluated):
    me = body.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh() if evaluated else body.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    if evaluated:
        body.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
    mw = np.array(body.matrix_world)
    return co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]


def deform_segments(rig):
    names = [b.name for b in rig.data.bones if b.use_deform]
    mw = rig.matrix_world
    segs = [(np.array(mw @ rig.data.bones[n].head_local), np.array(mw @ rig.data.bones[n].tail_local))
            for n in names]
    return names, segs


def deform_parents(rig, names):
    """Nearest deform ancestor of each deform bone. Rigs that hang deform bones from
    control or mechanism bones (other rigging tools) still get a connected deform hierarchy."""
    out = {}
    for n in names:
        p = rig.data.bones[n].parent
        while p is not None and not p.use_deform:
            p = p.parent
        out[n] = p.name if p is not None else None
    orphans = [n for n in names if out[n] is None]
    if len(orphans) > 1:
        # no deform ancestor (deform chains hung from controls): link each orphan to the
        # deform bone whose segment passes closest to its head, so the hops stay meaningful
        segs = {n: (rig.data.bones[n].head_local, rig.data.bones[n].tail_local) for n in names}
        for n in orphans[1:] if out[orphans[0]] is None else orphans:
            head = segs[n][0]
            best, best_d = None, None
            for m in names:
                if m == n:
                    continue
                a, b = segs[m]
                ab = b - a
                t = max(0.0, min(1.0, (head - a).dot(ab) / max(ab.length_squared, 1e-12)))
                d = (a + ab * t - head).length
                if best_d is None or d < best_d:
                    best, best_d = m, d
            out[n] = best
    return out


def weight_matrix(body, names):
    index = {vg.index: names.index(vg.name) for vg in body.vertex_groups if vg.name in names}
    w = np.zeros((len(body.data.vertices), len(names)))
    for v in body.data.vertices:
        for g in v.groups:
            col = index.get(g.group)
            if col is not None:
                w[v.index, col] = g.weight
    return w


def angle_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b)))))


def relative_bend(A, J, B, Ap, Jp, Bp):
    """Degrees the child segment turned relative to its parent segment."""
    undo = Vector(Jp - Ap).rotation_difference(Vector(J - A)).to_matrix()
    child = np.array(undo @ Vector(Bp - Jp))
    return angle_between(B - J, child)


def measure_placement(rig, body, height):
    """How well the elbow and knee sit in the limb: distance from the centre of
    the cross-section through the joint, as a share of the limb radius (0 =
    centred), and where the elbow falls between shoulder and wrist."""
    from rigger import fitting
    sl = fitting.Slicer([body])
    out = {}
    for joint, (a_ref, j_ref, b_ref) in {"elbow": (("upper_arm", "head"), ("forearm", "head"), ("hand", "head")),
                                         "knee": (("thigh", "head"), ("shin", "head"), ("foot", "head"))}.items():
        A, J, B = (point(rig, r, False) for r in (a_ref, j_ref, b_ref))
        normal = (J - A) / np.linalg.norm(J - A) + (B - J) / np.linalg.norm(B - J)
        loops = [(fitting.centroid(lp), lp) for lp in sl.section(J, normal)]
        loops = [(c, lp) for c, lp in loops if np.linalg.norm(c - J) < 0.08 * height]
        if not loops:
            out[joint] = {"off_center": None}
            continue
        c, lp = min(loops, key=lambda x: np.linalg.norm(x[0] - J))
        out[joint] = {"off_center": float(np.linalg.norm(c - J) / fitting.radius(lp, c)),
                      "along": float(np.linalg.norm(J - A) / (np.linalg.norm(J - A) + np.linalg.norm(B - J)))}
    return out


def anatomy(body, name, height):
    """Measuring skeleton of a body from its frozen markers and the mesh (left side)."""
    from rigger import fitting
    lm = {k: np.array(v) for k, v in bodies.landmarks(name).items()}
    sl = fitting.Slicer([body])
    shoulder, wrist, ankle, pelvis, neck = lm["shoulder"], lm["wrist"], lm["ankle"], lm["pelvis"], lm["neck"]
    elbow = fitting.elbow(sl, shoulder, wrist, height)
    if elbow is None:
        elbow = shoulder + (wrist - shoulder) * fitting.ELBOW_PRIOR
    hip_guess = np.array([max(abs(ankle[0]), shoulder[0] * 0.45), pelvis[1], pelvis[2] - 0.02 * height])
    knee, hip = fitting.leg(sl, ankle, hip_guess, height)
    hip = hip_guess if hip is None else hip
    knee = (hip + ankle) / 2 if knee is None else knee
    forearm = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    pts = {"shoulder": shoulder, "elbow": elbow, "wrist": wrist, "hand_end": wrist + forearm * 0.05 * height,
           "neck_side": np.array([0.012 * height, neck[1], shoulder[2]]), "pelvis": pelvis, "hip": hip,
           "knee": knee, "ankle": ankle, "neck": neck, "chin": lm["chin"],
           "mid_spine": (pelvis + neck) / 2}
    tip = fitting.foot_tip(sl, ankle, 0.0, height)
    pts["toe"] = tip if tip is not None else ankle + np.array([0.0, -0.08 * height, -0.035 * height])
    radii = {}
    for joint, (a, j, b, _r) in JOINTS.items():
        A, J, B = pts[a[0]], pts[j[0]], pts[b[0]]
        normal = (J - A) / np.linalg.norm(J - A) + (B - J) / np.linalg.norm(B - J)
        loops = [(fitting.centroid(lp), lp) for lp in sl.section(J, normal)]
        loops = [(c, lp) for c, lp in loops if np.linalg.norm(c - J) < 0.15 * height]
        radii[joint] = fitting.radius(*min(loops, key=lambda x: np.linalg.norm(x[0] - J))[::-1])             if loops else 0.05 * height
    for twist, (a, b) in TWISTS.items():
        mid = (pts[a] + pts[b]) / 2
        loops = [(fitting.centroid(lp), lp) for lp in sl.section(mid, pts[b] - pts[a])]
        loops = [(c, lp) for c, lp in loops if np.linalg.norm(c - mid) < 0.1 * height]
        radii["twist_" + twist] = fitting.radius(*min(loops, key=lambda x: np.linalg.norm(x[0] - mid))[::-1])             if loops else 0.04 * height
    return pts, radii


def follow(rig, stem, p):
    """Where a rest point attached to a DEF bone is in the current pose."""
    name = def_name(rig, stem)
    pb, bone = rig.pose.bones[name], rig.data.bones[name]
    m = rig.matrix_world @ pb.matrix @ bone.matrix_local.inverted() @ rig.matrix_world.inverted()
    return np.array(m @ Vector(p))


def limb_region(rest, segments, center, radius, limb_radius):
    """Skin within 'radius' of the joint and close to the limb's segments (geometry only)."""
    d = np.min([metrics.point_segment_distance(rest, a, b) for a, b in segments], axis=0)
    return (np.linalg.norm(rest - center, axis=1) < radius) & (d < 1.6 * limb_radius)


def measure_joint(rig, joint, rest, posed, tris, anat, height):
    pts, radii = anat
    (a, sa), (j, sj), (b, sb), frac = JOINTS[joint]
    A, J, B = pts[a], pts[j], pts[b]
    Ap, Jp, Bp = follow(rig, sa, A), follow(rig, sj, J), follow(rig, sb, B)
    region = limb_region(rest, [(A, J), (J, B)], J, frac * height, radii[joint])
    thick = metrics.thickness_ratio(rest, posed, region, [(A, J), (J, B)], [(Ap, Jp), (Jp, Bp)])
    out = {"vertices": int(region.sum()),
           "bend_deg": round(relative_bend(A, J, B, Ap, Jp, Bp), 1),
           "thickness": metrics.summary(thick)}
    out.update(metrics.area_distortion(rest, posed, tris, region))
    return out, region


def measure_twist(twist, degrees, rest, posed, anat):
    pts, radii = anat
    a, b = TWISTS[twist]
    head, tail = pts[a], pts[b]
    axis = (tail - head) / np.linalg.norm(tail - head)
    s = (rest - head) @ axis
    region = (s >= 0) & (s <= np.linalg.norm(tail - head)) &         (metrics.point_segment_distance(rest, head, tail) < 1.6 * radii["twist_" + twist])
    out = metrics.twist_profile(rest, posed, region, head, tail, math.radians(degrees))
    out["angles"] = [None if a is None else round(math.degrees(a), 1) for a in out["angles"]]
    return out, region


def run_body(name, config_name, pose_names, render_dir, config=None):
    config = config or CONFIGS[config_name]
    body, rig, bind_seconds = build(name, config)
    names, segs = deform_segments(rig)
    rest = vertices(body, False)
    height = float(rest[:, 2].max() - rest[:, 2].min())
    tris = metrics.triangles([list(p.vertices) for p in body.data.polygons])
    anat = anatomy(body, name, height)
    volume_rest = metrics.signed_volume(rest, tris)
    result = {
        "vertices": len(rest), "deform_bones": len(names), "bind_seconds": round(bind_seconds, 2),
        "placement": measure_placement(rig, body, height),
        "weights": metrics.weight_report(weight_matrix(body, names), names, deform_parents(rig, names)),
        "poses": {},
    }
    for pose in pose_names:
        steps, joints, twist, view = POSES[pose]
        reset_pose(rig)
        for step in steps:
            apply_step(rig, step)
        posed = vertices(body, True)
        entry = {"volume_change": round(metrics.signed_volume(posed, tris) / volume_rest - 1.0, 4)}
        focus = None
        for joint in joints:
            entry[joint], focus = measure_joint(rig, joint, rest, posed, tris, anat, height)
        if twist:
            degrees = sum(st[4] for st in steps if st[0] == "twist")
            entry["twist"], focus = measure_twist(twist, degrees, rest, posed, anat)
        if render_dir:
            entry["image"] = render_pose(body, rest, posed, tris, focus, height, view,
                                         os.path.join(render_dir, f"{config_name}_{name}_{pose}.png"))
        result["poses"][pose] = entry
        print(f"  {config_name:8s} {name:17s} {pose:20s} {json.dumps(headline(entry))}", flush=True)
    reset_pose(rig)
    return result


def headline(entry):
    """The few numbers worth printing for a pose."""
    out = {"vol": entry["volume_change"]}
    for key, val in entry.items():
        if isinstance(val, dict) and "thickness" in val:
            out[key] = {"p05": r3(val["thickness"]["p05"]), "p95": r3(val["thickness"]["p95"]),
                        "crushed": r3(val["crushed"]), "bend": val["bend_deg"]}
        if key == "twist":
            out["twist"] = {"rms": r3(val["rms_error"]), "min_r": r3(val["min_radius"]),
                            "end": r3(val["end_fraction"])}
    return out


def r3(v):
    return None if v is None else round(v, 3)


# ---------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------

def render_pose(body, rest, posed, tris, focus, height, view, path):
    """Posed body coloured by surface stretch: blue = crushed, red = stretched."""
    import render
    ratio = metrics.triangle_areas(posed, tris) / np.maximum(metrics.triangle_areas(rest, tris), 1e-12)
    log = np.clip(np.log2(np.maximum(ratio, 1e-6)), -1.5, 1.5) / 1.5
    per_vert = np.zeros(len(rest))
    count = np.zeros(len(rest))
    for k in range(3):
        np.add.at(per_vert, tris[:, k], log)
        np.add.at(count, tris[:, k], 1)
    per_vert /= np.maximum(count, 1)
    colors = np.ones((len(rest), 4))
    colors[:, 0] = np.where(per_vert > 0, 1.0, 1.0 + per_vert)
    colors[:, 1] = 1.0 - np.abs(per_vert)
    colors[:, 2] = np.where(per_vert < 0, 1.0, 1.0 - per_vert)
    me = body.data
    attr = me.color_attributes.get("bench_heat") or me.color_attributes.new("bench_heat", 'FLOAT_COLOR', 'POINT')
    attr.data.foreach_set("color", colors.ravel())
    mat = bpy.data.materials.get("bench_heat")
    if mat is None:
        mat = bpy.data.materials.new("bench_heat")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        node = nodes.new("ShaderNodeAttribute")
        node.attribute_name = "bench_heat"
        mat.node_tree.links.new(node.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    me.materials.clear()
    me.materials.append(mat)
    if focus is not None and focus.any():
        center = Vector(posed[focus].mean(axis=0))
        size = max(float(np.ptp(posed[focus], axis=0).max()) * 1.8, 0.35 * height)
    else:
        center, size = None, None
    cut = None
    if view == "side_cut":  # leg poses: hide the hanging arm in front of the leg
        view, cut = "side", 0.07 * height
    render.render([body], path, view, center, size, cut)
    try:
        return os.path.relpath(path, HERE)
    except ValueError:  # output on another drive
        return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def mean_of(results, config, pose, getter):
    vals = []
    for body in results[config]:
        entry = results[config][body]["poses"].get(pose)
        try:
            v = getter(entry)
        except (KeyError, TypeError):
            v = None
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


POSE_COLUMNS = [
    ("collapse p05", lambda e, j: e[j]["thickness"]["p05"], "lower = more volume lost (1 = none)"),
    ("bulge p95", lambda e, j: e[j]["thickness"]["p95"], "higher = more bulging (1 = none)"),
    ("crushed", lambda e, j: e[j]["crushed"], "share of skin triangles squashed below half area"),
    ("stretch rms", lambda e, j: e[j]["rms_log"], "RMS of log2 area change (0 = none)"),
]
TWIST_COLUMNS = [
    ("twist error", lambda e: e["twist"]["rms_error"], "RMS deviation from an even twist, fraction of the angle"),
    ("min radius", lambda e: e["twist"]["min_radius"], "thinnest slice / rest (candy-wrapper, 1 = none)"),
    ("end twist", lambda e: e["twist"]["end_fraction"], "twist reached in the last slice / ideal there (1 = full)"),
]


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def write_report(results, out_dir, compare=None):
    lines = ["# Deformation benchmark", "",
             "Averages over the bodies. See `bench/README.md` for definitions.", ""]
    configs = list(results)
    bodies_list = sorted({b for c in results.values() for b in c})
    lines += ["## Weights", "",
              "| config | body | bleed | finger bleed | cross side | unweighted | max influences | bind s |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in configs:
        for b in bodies_list:
            r = results[c].get(b)
            if not r:
                continue
            w = r["weights"]
            lines.append(f"| {c} | {b} | {fmt(w['bleed'], True)} | {fmt(w.get('finger_bleed'), True)} | {fmt(w['cross_side'], True)} | "
                         f"{w['unweighted']} | {w['max_influences']} | {r['bind_seconds']} |")
    lines += ["", "## Joint placement", "",
              "Distance of the joint from the centre of the limb section through it, as a share of the "
              "limb radius (0 = centred); elbow position between shoulder and wrist.", "",
              "| config | body | elbow off-centre | knee off-centre | elbow along arm |",
              "| --- | --- | --- | --- | --- |"]
    for c in configs:
        for b in bodies_list:
            r = results[c].get(b)
            pl = (r or {}).get("placement")
            if not pl:
                continue
            lines.append(f"| {c} | {b} | {fmt(pl['elbow'].get('off_center'))} | {fmt(pl['knee'].get('off_center'))} | "
                         f"{fmt(pl['elbow'].get('along'))} |")
    lines += ["", "Worst bleed pairs (main bone <- far bone: vertices):", ""]
    for c in configs:
        for b in bodies_list:
            r = results[c].get(b)
            if r and r["weights"]["bleed_pairs"]:
                pairs = ", ".join(f"`{k}`: {v}" for k, v in list(r["weights"]["bleed_pairs"].items())[:4])
                lines.append(f"* {c} / {b}: {pairs}")
    poses = list(next(iter(next(iter(results.values())).values()))["poses"])
    lines += ["", "## Joints", "",
              "| pose | joint | config | " + " | ".join(c for c, _g, _d in POSE_COLUMNS) + " | volume |",
              "| --- | --- | --- | " + " | ".join("---" for _ in POSE_COLUMNS) + " | --- |"]
    for pose in poses:
        _s, joints, twist, _v = POSES[pose]
        for j in joints:
            for c in configs:
                cells = [fmt(mean_of(results, c, pose, lambda e, g=g: g(e, j)), pct=(name == "crushed"))
                         for name, g, _d in POSE_COLUMNS]
                vol = mean_of(results, c, pose, lambda e: e["volume_change"])
                lines.append(f"| {pose} | {j} | {c} | " + " | ".join(cells) + f" | {fmt(vol, True)} |")
    lines += ["", "## Twist", "",
              "| pose | config | " + " | ".join(c for c, _g, _d in TWIST_COLUMNS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in TWIST_COLUMNS) + " |"]
    for pose in poses:
        if not POSES[pose][2]:
            continue
        for c in configs:
            cells = [fmt(mean_of(results, c, pose, g)) for _n, g, _d in TWIST_COLUMNS]
            lines.append(f"| {pose} | {c} | " + " | ".join(cells) + " |")
    lines += ["", "Column meanings:", ""]
    lines += [f"* **{n}**: {d}" for n, _g, d in POSE_COLUMNS + TWIST_COLUMNS]
    if compare:
        lines += ["", "## Compared with " + compare["_label"], "",
                  "| config | pose | measure | before | after |", "| --- | --- | --- | --- | --- |"]
        for c in configs:
            if c not in compare:
                continue
            for pose in poses:
                _s, joints, twist, _v = POSES[pose]
                for j in joints:
                    for name, g, _d in POSE_COLUMNS[:3]:
                        before = mean_of(compare, c, pose, lambda e, g=g: g(e, j))
                        after = mean_of(results, c, pose, lambda e, g=g: g(e, j))
                        if before is not None and after is not None and abs(after - before) > 0.005:
                            lines.append(f"| {c} | {pose} | {j} {name} | {fmt(before)} | {fmt(after)} |")
                if twist:
                    for name, g, _d in TWIST_COLUMNS:
                        before, after = mean_of(compare, c, pose, g), mean_of(results, c, pose, g)
                        if before is not None and after is not None and abs(after - before) > 0.005:
                            lines.append(f"| {c} | {pose} | {name} | {fmt(before)} | {fmt(after)} |")
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def write_gallery(results, out_dir):
    """gallery.html: per pose, one row per body and one column per config, with the headline numbers."""
    import html
    configs = list(results)
    bodies_list = sorted({b for c in results.values() for b in c})
    poses = list(next(iter(next(iter(results.values())).values()))["poses"])
    parts = ["<!doctype html><meta charset='utf-8'><title>Deformation gallery</title>",
             "<style>body{font:14px system-ui,sans-serif;background:#1d1d1f;color:#ddd;margin:16px}"
             "table{border-collapse:collapse;margin-bottom:28px}td,th{padding:4px;vertical-align:top}"
             "img{width:220px;display:block}small{color:#aaa;font-family:monospace;white-space:pre}"
             "h2{margin:24px 0 6px}nav a{color:#9cf;margin-right:10px}</style>",
             "<h1>Deformation gallery</h1><p>White = unchanged skin, blue = squashed, red = stretched.</p><nav>"]
    parts += [f"<a href='#{p}'>{p}</a>" for p in poses]
    parts.append("</nav>")
    for pose in poses:
        parts.append(f"<h2 id='{pose}'>{pose}</h2><table><tr><th></th>")
        parts += [f"<th>{html.escape(c)}</th>" for c in configs]
        parts.append("</tr>")
        for b in bodies_list:
            parts.append(f"<tr><th>{html.escape(b)}</th>")
            for c in configs:
                entry = results[c].get(b, {}).get("poses", {}).get(pose)
                if not entry or "image" not in entry:
                    parts.append("<td>-</td>")
                    continue
                img = os.path.relpath(os.path.join(HERE, entry["image"]), out_dir).replace(os.sep, "/")
                nums = json.dumps(headline(entry), indent=None).replace("{", "").replace("}", "")
                parts.append(f"<td><img src='{html.escape(img)}' loading='lazy'>"
                             f"<small>{html.escape(nums.replace(', ', chr(10)))}</small></td>")
            parts.append("</tr>")
        parts.append("</table>")
    path = os.path.join(out_dir, "gallery.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(parts))
    return path


def arg(flag, default=None):
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if flag in argv:
        i = argv.index(flag)
        return True if i + 1 >= len(argv) or argv[i + 1].startswith("--") else argv[i + 1]
    return default


def main():
    pick = lambda flag, allv: [s for s in str(arg(flag)).split(",")] if arg(flag) else list(allv)  # noqa: E731
    body_names = pick("--bodies", bodies.BODIES)
    if body_names == ["all"]:
        body_names = list(bodies.ALL_BODIES)
    elif body_names == ["heldout"]:
        body_names = list(bodies.HELD_OUT)
    config_names = pick("--configs", CONFIGS)
    pose_names = pick("--poses", POSES)
    out_dir = os.path.abspath(arg("--out", os.path.join(HERE, "results", "latest")))
    render_dir = os.path.join(out_dir, "images") if arg("--render") else None
    os.makedirs(out_dir, exist_ok=True)
    rigger.register()
    results = {}
    failures = []
    for config in config_names:
        results[config] = {}
        for name in body_names:
            try:
                results[config][name] = run_body(name, config, pose_names, render_dir)
            except Exception as exc:  # keep benchmarking the other bodies
                import traceback
                traceback.print_exc()
                failures.append(f"{config}/{name}: {exc}")
    with open(os.path.join(os.path.dirname(HERE), "rigger", "blender_manifest.toml"), encoding="utf-8") as fh:
        version = next((ln.split("=")[1].strip().strip('"') for ln in fh if ln.startswith("version")), None)
    meta = {"blender": bpy.app.version_string, "rigger": version,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "failures": failures}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"_meta": meta, **results}, fh, indent=1)
    compare = None
    if arg("--compare"):
        with open(arg("--compare"), encoding="utf-8") as fh:
            compare = json.load(fh)
        compare.pop("_meta", None)
        compare["_label"] = os.path.relpath(arg("--compare"), out_dir)
    done = {c: r for c, r in results.items() if r}
    print("report:", write_report(done, out_dir, compare))
    if render_dir and done:
        print("gallery:", write_gallery(done, out_dir))
    if failures:
        print("FAILURES:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
