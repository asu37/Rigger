"""Golden clips for video motion capture: a benchmark body moves through a
scripted performance (arm raises, reaching at the camera, squat, twist, kick,
turn and step, finger curls), is rendered to an MP4 with Cycles, and the
video-to-animation pipeline is scored against the exact motion it came from.

    blender -b --factory-startup --python bench/video_bench.py -- make --body realistic_male [--out DIR]
    blender -b --factory-startup --python bench/video_bench.py -- eval --body realistic_male [--focal 26] [--out DIR]

eval --two scores the two-camera solve (needs both clips). make writes DIR/clip.mp4 and DIR/truth.npz (with --view side: clip_side.mp4, truth_side.npz, the
second camera of two-camera runs); eval writes DIR/report_<focal>.json/.md.
Scores (lower is better): joint error with the root aligned (MPJPE) and after
the best rigid alignment (PA-MPJPE), limb angle error, root path error, jitter
against the true motion, foot sliding during true contacts, finger curl error,
and the pose model's own 2D error.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.argv = sys.argv[:1]

import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402

import rigger  # noqa: E402
from rigger.mocap import bake, pipeline, solver  # noqa: E402
from rigger.mocap.skeleton import from_rig, matrix_to_rotvec, rotation_between, rotvec_to_matrix  # noqa: E402

FPS = 24
FRAMES = 120
RES = (432, 768)
FOCAL_MM = 35.0
CAM_POS = (0.0, -3.7, 1.0)
CAM_PITCH = 6.0     # degrees down
SIDE_ANGLE = 70.0   # the second camera of two-camera clips: turned this far around the character (to its left)
VIEW = "front"      # "side" renders the second camera (make --view side)
TWO = False         # eval --two: solve from both cameras

# Beats of the performance: frame -> parameters (degrees, metres). Unlisted parameters are 0.
# arm: elevation (-90 down .. 90 up), azimuth (0 out to the side, 90 forward); thigh: pitch forward, abduction
NEUTRAL = {"arm_el.L": -70, "arm_el.R": -70, "arm_az.L": 10, "arm_az.R": 10, "elbow.L": 15, "elbow.R": 15}
BEATS = [
    (0, {}),
    (16, {"arm_el.L": 0, "arm_el.R": 0, "arm_az.L": 0, "arm_az.R": 0, "elbow.L": 20, "elbow.R": 20, "curl.L": -5,
          "curl.R": -5}),
    (32, {"arm_el.L": 5, "arm_el.R": 10, "arm_az.L": 85, "arm_az.R": 80, "elbow.L": 5, "elbow.R": 5, "head_pitch": 10}),
    (46, {"arm_el.L": -60, "arm_az.L": -20, "elbow.L": 100, "arm_el.R": 75, "arm_az.R": 20, "elbow.R": 10,
          "spine_side": 14, "curl.L": 70, "curl.R": 70}),
    (62, {"thigh_p.L": 75, "thigh_p.R": 75, "thigh_a.L": 12, "thigh_a.R": 12, "knee.L": 115, "knee.R": 115,
          "spine_bend": 25, "arm_el.L": -5, "arm_el.R": -5, "arm_az.L": 80, "arm_az.R": 80, "elbow.L": 10,
          "elbow.R": 10}),
    (78, {"spine_twist": 35, "head_yaw": -40, "arm_el.L": -40, "arm_el.R": -40, "arm_az.L": 40, "arm_az.R": -30,
          "elbow.L": 60, "elbow.R": 40, "curl.L": 35, "curl.R": 35}),
    (92, {"thigh_p.L": 50, "knee.L": 20, "arm_el.L": -10, "arm_el.R": -10, "arm_az.L": -10, "arm_az.R": -10,
          "elbow.L": 10, "elbow.R": 10, "spine_bend": -5}),
    (106, {"yaw": 55, "x": 0.35, "arm_el.R": 40, "arm_az.R": 10, "elbow.R": 95, "head_yaw": -25,
           "thigh_a.R": 10, "knee.L": 10}),
    (119, {"yaw": 10, "x": 0.45}),
]


def params_at(frame):
    keys = [b[0] for b in BEATS]
    i = max(k for k in range(len(BEATS)) if keys[k] <= frame)
    j = min(i + 1, len(BEATS) - 1)
    a, b = BEATS[i], BEATS[j]
    s = 0.0 if j == i else (frame - a[0]) / (b[0] - a[0])
    s = s * s * (3 - 2 * s)
    names = set(NEUTRAL) | set(a[1]) | set(b[1])
    pa = {k: a[1].get(k, NEUTRAL.get(k, 0.0)) for k in names}
    pb = {k: b[1].get(k, NEUTRAL.get(k, 0.0)) for k in names}
    return {k: pa[k] * (1 - s) + pb[k] * s for k in names}


def performance(sk):
    """-> theta (F,J,3) bone-frame rotvecs, root_R (F,3,3) world, root_t (F,3) world (feet on the floor)."""
    ix = sk.index
    J = len(sk.joints)
    F = FRAMES
    theta = np.zeros((F, J, 3))
    root_R = np.zeros((F, 3, 3))
    root_t = np.zeros((F, 3))
    rad = np.radians
    for f in range(F):
        p = params_at(f)
        G = {}
        L = {j: np.eye(3) for j in range(J)}
        yaw = rad(p.get("yaw", 0))
        Ry = rotvec_to_matrix(np.array((0, 0, yaw)))
        root = ix("pelvis")
        G[root] = Ry
        spine = [j for j in range(J) if sk.roles[j] == "spine"]
        n = len(spine)
        for j in spine:
            L[j] = (rotvec_to_matrix(np.array((rad(p.get("spine_bend", 0)) / n, 0, 0)))
                    @ rotvec_to_matrix(np.array((0, rad(-p.get("spine_side", 0)) / n, 0)))
                    @ rotvec_to_matrix(np.array((0, 0, rad(p.get("spine_twist", 0)) / n))))
        if "head" in sk.joints:
            L[ix("head")] = rotvec_to_matrix(np.array((0, 0, rad(p.get("head_yaw", 0))))) @ \
                rotvec_to_matrix(np.array((rad(p.get("head_pitch", 0)), 0, 0)))

        def world(j):
            if j not in G:
                G[j] = world(sk.parents[j]) @ L[j]
            return G[j]

        def aim(j, direction):
            par = world(sk.parents[j])
            rest_dir = sk.end_point(j) - sk.rest[j]
            L[j] = rotation_between(rest_dir, par.T @ direction)
            G.pop(j, None)

        def bend(j, angle):
            axis = sk.frames[j][:, 0] * sk.hinge[j]
            L[j] = rotvec_to_matrix(axis * rad(angle))
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            e, a = rad(p.get(f"arm_el.{side}", 0)), rad(p.get(f"arm_az.{side}", 0))
            world_dir = Ry @ np.array((sgn * np.cos(e) * np.cos(a), -np.cos(e) * np.sin(a), np.sin(e)))
            aim(ix(f"upper_arm.{side}"), world_dir)
            bend(ix(f"forearm.{side}"), p.get(f"elbow.{side}", 0))
            tp, ta = rad(p.get(f"thigh_p.{side}", 0)), rad(p.get(f"thigh_a.{side}", 0))
            world_dir = Ry @ np.array((sgn * np.sin(ta), -np.sin(tp) * np.cos(ta), -np.cos(tp) * np.cos(ta)))
            aim(ix(f"thigh.{side}"), world_dir)
            bend(ix(f"shin.{side}"), p.get(f"knee.{side}", 0))
            G.clear()
            G[root] = Ry
            foot = ix(f"foot.{side}")
            L[foot] = world(sk.parents[foot]).T @ Ry  # feet stay flat
            G.clear()
            G[root] = Ry
            for finger in ("index", "middle", "ring", "pinky"):
                for i in (1, 2, 3):
                    name = f"{finger}.{i:02d}.{side}"
                    if name in sk.joints:
                        bend(ix(name), p.get(f"curl.{side}", 0) * (0.8 if i == 1 else 1.0))
        G.clear()
        G[root] = Ry
        for j in range(J):
            if j != root:
                B = sk.frames[j]
                theta[f, j] = matrix_to_rotvec(B.T @ L[j] @ B)
        root_R[f] = Ry
        root_t[f] = sk.rest[root] + np.array((p.get("x", 0.0), 0, 0))
    # feet on the floor: the lowest foot point at z = 0 (or above, when one foot is lifted it is the other)
    _G, _P, S = solver.pose_points(sk, root_R, root_t, theta)
    feet = np.stack([S[n] for n in S if n.split(".")[0] in ("heel", "big_toe", "small_toe", "toe_end")], axis=1)
    root_t[:, 2] -= feet[:, :, 2].min(axis=1)
    return theta, root_R, root_t


def camera_matrix():
    pitch = np.radians(90 - CAM_PITCH)
    R = np.array(Matrix.Rotation(pitch, 3, 'X'))
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = CAM_POS
    if VIEW == "side":
        turn = np.eye(4)
        turn[:3, :3] = np.array(Matrix.Rotation(np.radians(SIDE_ANGLE), 3, 'Z'))
        M = turn @ M
    return M


def suffix():
    return "" if VIEW == "front" else "_" + VIEW


def to_camera(points, M):
    """World points -> pinhole camera coordinates (x right, y down, z forward)."""
    R, c = M[:3, :3], M[:3, 3]
    local = (points - c) @ R             # Blender camera axes: x right, y up, -z forward
    return local * np.array((1, -1, -1))


def build_body(name):
    import run_bench as rb
    config = rb.CONFIGS[next(iter(rb.CONFIGS))]
    body, rig, _ = rb.build(name, config)
    return body, rig


def make(name, out):
    os.makedirs(out, exist_ok=True)
    body, rig = build_body(name)
    sk = from_rig(rig)
    theta, root_R, root_t = performance(sk)
    world = {"root_t": root_t, "root_R": root_R, "theta": theta, "valid": np.ones(FRAMES, bool),
             "contacts": np.zeros((FRAMES, 2), bool), "W": np.eye(3), "cam_pos": np.zeros(3)}
    arrays = solver.pack(sk, world)
    meta = {"fps": FPS, "frames": FRAMES, "image_size": list(RES), "face": []}
    ctx = bpy.context
    bake.bake(ctx, rig, meta, sk.to_dict(), arrays, start=1, camera=False)
    scene = ctx.scene
    scene.frame_start, scene.frame_end = 1, FRAMES
    # look: skin-coloured body on a grey floor, sky light and a sun
    mat = bpy.data.materials.new("skin")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.75, 0.52, 0.42, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    body.data.materials.clear()
    body.data.materials.append(mat)
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    floor = ctx.object
    fmat = bpy.data.materials.new("floor")
    fmat.use_nodes = True
    fmat.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (0.32, 0.33, 0.35, 1)
    floor.data.materials.append(fmat)
    world_bg = bpy.data.worlds.new("sky")
    world_bg.color = (0.55, 0.6, 0.68)
    scene.world = world_bg
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN'))
    sun.data.energy = 3.0
    sun.rotation_euler = (0.7, 0.2, -0.5)
    scene.collection.objects.link(sun)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    cam.data.lens = FOCAL_MM
    cam.data.sensor_width = 36.0
    cam.data.sensor_fit = 'AUTO'
    M = camera_matrix()
    cam.matrix_world = Matrix(M.tolist())
    scene.collection.objects.link(cam)
    scene.camera = cam
    r = scene.render
    r.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 8
    scene.cycles.use_denoising = True
    r.resolution_x, r.resolution_y = RES
    r.resolution_percentage = 100
    r.fps = FPS
    scene.view_settings.view_transform = 'Standard'
    frames_dir = os.path.join(out, "frames" + suffix())
    os.makedirs(frames_dir, exist_ok=True)
    r.image_settings.file_format = 'PNG'
    r.filepath = os.path.join(frames_dir, "f_")
    start = time.time()
    bpy.ops.render.render(animation=True)
    print(f"rendered {FRAMES} frames in {time.time() - start:.0f} s")
    # truth
    G, P, S = solver.pose_points(sk, root_R, root_t, theta)
    f_px = FOCAL_MM / 36.0 * max(RES)
    sites_px = {}
    for n, pts in S.items():
        c = to_camera(pts, M)
        sites_px[n] = np.stack([f_px * c[:, 0] / c[:, 2] + RES[0] / 2, f_px * c[:, 1] / c[:, 2] + RES[1] / 2], axis=1)
    np.savez(os.path.join(out, f"truth{suffix()}.npz"), theta=theta, root_R=root_R, root_t=root_t, P=P, cam=M,
             skeleton=np.array(sk.dumps()), site_names=np.array(list(sites_px)),
             sites_px=np.stack(list(sites_px.values()), axis=1))
    encode(frames_dir, os.path.join(out, f"clip{suffix()}.mp4"))


def encode(frames_dir, path):
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    scene = bpy.data.scenes.new("encode")
    se = scene.sequence_editor_create()
    strips = se.strips if hasattr(se, "strips") else se.sequences
    strip = strips.new_image("frames", os.path.join(frames_dir, files[0]), 1, 1)
    for f in files[1:]:
        strip.elements.append(f)
    r = scene.render
    r.resolution_x, r.resolution_y = RES
    r.resolution_percentage = 100
    r.fps = FPS
    scene.frame_start, scene.frame_end = 1, len(files)
    scene.view_settings.view_transform = 'Standard'
    if hasattr(r.image_settings, "media_type"):
        r.image_settings.media_type = 'VIDEO'
    r.image_settings.file_format = 'FFMPEG'
    r.ffmpeg.format = 'MPEG4'
    r.ffmpeg.codec = 'H264'
    r.ffmpeg.constant_rate_factor = 'HIGH'
    r.filepath = path
    bpy.ops.render.render(animation=True, scene=scene.name)
    print("video:", path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def fk_world(sk, local_q, root_q, root_t):
    G = bake.world_rotations(sk.parents, local_q, root_q)
    F, J = G.shape[:2]
    P = np.zeros((F, J, 3))
    for j, p in enumerate(sk.parents):
        P[:, j] = root_t if p < 0 else P[:, p] + G[:, p] @ (sk.rest[j] - sk.rest[p])
    return G, P


def rigid(a, b):
    a0, b0 = a - a.mean(0), b - b.mean(0)
    U, _s, Vt = np.linalg.svd(a0.T @ b0)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt, a.mean(0), b.mean(0)


def score(sk, truth, meta, arrays, seen):
    fps = meta["fps"]
    _G, P = fk_world(sk, arrays["local_q"], arrays["root_q"], arrays["root_t"])
    Pt = truth["P"]
    F = min(len(P), len(Pt))
    P, Pt = P[:F], Pt[:F]
    body = [j for j in range(len(sk.joints)) if sk.roles[j] not in ("finger1", "finger", "thumb")]
    root = sk.parents.index(-1)
    rel = (P[:, body] - P[:, [root]]) - (Pt[:, body] - Pt[:, [root]])
    mpjpe = np.linalg.norm(rel, axis=2).mean()
    pa, aligned = [], np.zeros_like(P)
    for f in range(F):
        R, ma, mb = rigid(P[f, body], Pt[f, body])
        aligned[f] = (P[f] - ma) @ R + mb
        pa.append(np.linalg.norm(aligned[f, body] - Pt[f, body], axis=1).mean())
    limbs = {}
    for name in ("upper_arm", "forearm", "thigh", "shin", "spine2", "head"):
        js = [j for j in body if sk.joints[j].split(".")[0] == name]
        errs = []
        for j in js:
            end = [k for k in sk.children(j) if k in body]
            if not end:
                continue
            k = end[0]
            a = aligned[:, k] - aligned[:, j]
            b = Pt[:, k] - Pt[:, j]
            cos = np.sum(a * b, axis=1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            errs.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        if errs:
            limbs[name] = float(np.mean(errs))
    rt, rtt = arrays["root_t"][:F], truth["root_t"][:F]
    path = np.linalg.norm((rt[:, :2] - rt[0, :2]) - (rtt[:, :2] - rtt[0, :2]), axis=1).mean()
    height = np.abs(rt[:, 2] - rtt[:, 2]).mean()
    acc = lambda X: np.linalg.norm(np.diff(X[:, body], 2, axis=0), axis=2).mean() * fps * fps  # noqa: E731
    # feet: true contacts = foot site low and slow
    slide, fsum = [], 0
    names = list(truth["site_names"])
    _Gt, _Pt, St = solver.pose_points(sk, truth["root_R"], truth["root_t"], truth["theta"])
    _Ge, _Pe, Se = solver.pose_points(sk, bake.quat_to_matrix(arrays["root_q"]), arrays["root_t"],
                                      local_theta(sk, arrays["local_q"]))
    for side in ("L", "R"):
        heel_t = St[f"heel.{side}"][:F]
        speed_t = np.r_[0, np.linalg.norm(np.diff(heel_t[:, :2], axis=0), axis=1)] * fps
        contact = (heel_t[:, 2] < 0.03) & (speed_t < 0.1)
        heel_e = Se[f"heel.{side}"][:F]
        speed_e = np.r_[0, np.linalg.norm(np.diff(heel_e[:, :2], axis=0), axis=1)] * fps
        slide += list(speed_e[contact])
        fsum += contact.sum()
    th_e = local_theta(sk, arrays["local_q"])[:F]
    fingers = [j for j in range(len(sk.joints)) if sk.roles[j] in ("finger1", "finger")]
    curl = np.degrees(np.abs(th_e[:, fingers, 0] - truth["theta"][:F, fingers, 0])).mean()
    # the pose model alone: 2D error against the true projections
    kp_err = []
    site_kp = {v: k for k, v in __import__("rigger.mocap.skeleton", fromlist=["kp_sites"]).kp_sites().items()}
    person_h = np.ptp(truth["sites_px"][:, :, 1], axis=1).mean()
    for i, n in enumerate(names):
        if n in site_kp and site_kp[n] < 23:
            k = site_kp[n]
            ok = seen["scores"][:F, k] > 0.3
            kp_err += list(np.linalg.norm(seen["kp"][:F, k] - truth["sites_px"][:F, i], axis=1)[ok])
    return {
        "MPJPE_mm": round(float(mpjpe) * 1000, 1),
        "PA_MPJPE_mm": round(float(np.mean(pa)) * 1000, 1),
        "limb_angle_deg": {k: round(v, 1) for k, v in limbs.items()},
        "root_path_mm": round(float(path) * 1000, 1),
        "root_height_mm": round(float(height) * 1000, 1),
        "jitter_ratio": round(float(acc(P) / max(acc(Pt), 1e-9)), 2),
        "foot_slide_cm_s": round(float(np.mean(slide)) * 100, 1) if slide else None,
        "contact_frames": int(fsum),
        "finger_curl_deg": round(float(curl), 1),
        "keypoint_2d_err_pct_height": round(float(np.mean(kp_err)) / person_h * 100, 2),
        "found": meta.get("found"),
        "solve_seconds": meta.get("solve_seconds"),
    }


def calibration_error(truth, truth_side, meta):
    """Solved second-camera pose vs the true one: rotation (deg) and position (m) of camera 2 seen from camera 1."""
    D = np.diag([1, -1, -1])
    Ma, Mb = truth["cam"], truth_side["cam"]
    R_true = D @ Mb[:3, :3].T @ Ma[:3, :3] @ D
    t_true = D @ Mb[:3, :3].T @ (Ma[:3, 3] - Mb[:3, 3])
    cal = meta.get("calibration", {})
    true_angle = float(np.degrees(np.arccos(np.clip((np.trace(R_true) - 1) / 2, -1, 1))))
    true_base = float(np.linalg.norm(-R_true.T @ t_true))
    return {"angle_solved": cal.get("angle_deg"), "angle_true": round(true_angle, 2),
            "baseline_solved": cal.get("baseline_m"), "baseline_true": round(true_base, 3),
            "agreement": cal.get("agreement")}


def local_theta(sk, local_q):
    L = bake.quat_to_matrix(local_q)
    B = sk.frames
    return matrix_to_rotvec(np.swapaxes(B, -1, -2) @ L @ B)


def evaluate(name, out, focal, settings):
    from rigger.mocap.skeleton import Skeleton
    truth = dict(np.load(os.path.join(out, "truth.npz")))
    sk = Skeleton.from_dict(json.loads(str(truth["skeleton"])))
    _body, rig = build_body(name)
    sk_solve = from_rig(rig)   # today's site model; the joints are the truth's
    t0 = time.time()
    seen = pipeline.perceive(os.path.join(out, "clip.mp4"), cache_dir=os.path.join(out, "cache"),
                             depth=bool(settings.get("use_depth", False)) and not TWO)
    t_kp = time.time() - t0
    second = None
    if TWO:
        seen2 = pipeline.perceive(os.path.join(out, "clip_side.mp4"), cache_dir=os.path.join(out, "cache"))
        second = {"seen": seen2, "focal_mm": focal, "offset": 0.0, "source": "clip_side.mp4"}
    meta, arrays = pipeline.solve(sk_solve, seen, focal_mm=focal, settings=settings, second=second)
    result = score(sk, truth, meta, arrays, seen)
    if TWO:
        result["calibration"] = calibration_error(truth, dict(np.load(os.path.join(out, "truth_side.npz"))), meta)
    result["keypoint_seconds"] = round(t_kp, 1) if not seen.get("cached") else "cached"
    result["focal_mm"] = focal
    result["settings"] = settings
    tag = ("two_" if TWO else "") + f"{focal:g}" + ("_" + "_".join(f"{k}{v}" for k, v in sorted(settings.items())) if settings else "")
    with open(os.path.join(out, f"report_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    print("RESULT", json.dumps(result))
    return result


def arg(flag, default=None):
    return ARGS[ARGS.index(flag) + 1] if flag in ARGS else default


def main():
    os.environ.setdefault("RIGGER_AI_DIR", os.path.join(HERE, "assets", "ai"))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rigger.register()
    global VIEW, TWO
    VIEW = arg("--view", "front")
    TWO = "--two" in ARGS
    name = arg("--body", "realistic_male")
    out = os.path.abspath(arg("--out", os.path.join(HERE, "results", "video", name)))
    if ARGS and ARGS[0] == "make":
        make(name, out)
    else:
        settings = json.loads(arg("--settings", "{}"))
        evaluate(name, out, float(arg("--focal", pipeline.DEFAULT_FOCAL_MM)), settings)


main()
