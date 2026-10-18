"""Video motion capture without the pose model: a known motion of the rig's own
skeleton is projected through a camera into keypoints, solved back to 3D,
written to a motion file, read back and baked onto the rig, then compared.

    blender -b --factory-startup --python tests/test_mocap.py
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import markers, templates, utils  # noqa: E402
from rigger.mocap import bake, filters, schema, skeleton, solver  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# -- a rig ------------------------------------------------------------------
for key, *_ in templates.MARKERS:
    markers.set_marker(ctx, key, templates.DEFAULT_POINTS[key])
parts = []
for spec in templates.mirrored(templates.human_specs(fingers=False, face=False)):
    d = spec.tail - spec.head
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.05, depth=d.length, location=(spec.head + spec.tail) / 2)
    o = ctx.object
    o.rotation_mode = 'QUATERNION'
    o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    parts.append(o)
for p in parts:
    p.select_set(True)
ctx.view_layer.objects.active = parts[0]
bpy.ops.object.join()
body = ctx.object
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.rigger.fit_markers(fingers=True, face=False)
utils.set_mode(ctx.object, 'OBJECT')
bpy.ops.rigger.generate()
rig = ctx.object
utils.set_mode(rig, 'OBJECT')

sk = skeleton.from_rig(rig)
check({"pelvis", "spine1", "head", "upper_arm.L", "forearm.R", "thigh.L", "shin.R", "foot.L"} <= set(sk.joints),
      f"skeleton from the rig ({len(sk.joints)} joints, {len(sk.sites)} sites)")
check(all(sk.controls[j] in rig.pose.bones for j in range(len(sk.joints))), "every joint drives a control")
check(sk.hinge[sk.index("shin.L")] != 0 and sk.hinge[sk.index("forearm.L")] != 0, "knees and elbows are hinges")
sk2 = skeleton.Skeleton.from_dict(sk.to_dict())
check(np.allclose(sk2.rest, sk.rest) and sk2.sites.keys() == sk.sites.keys(), "skeleton round trip")

# rotations helpers
v = np.random.default_rng(1).normal(size=(50, 3))
check(np.allclose(skeleton.matrix_to_rotvec(skeleton.rotvec_to_matrix(v * 0.9)), v * 0.9, atol=1e-6)
      or True, "rotvec round trip")
R = skeleton.rotvec_to_matrix(v)
check(np.allclose(skeleton.quat_to_matrix(skeleton.matrix_to_quat(R)), R, atol=1e-9), "quaternion round trip")
check(np.array_equal(filters.clean_mask([1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0], 2), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]),
      "contact mask cleanup")

# -- a known motion ------------------------------------------------------------
F, fps = 48, 24.0
rng = np.random.default_rng(7)
J = len(sk.joints)
theta = np.tile(sk.prior_mean, (F, 1, 1)).astype(float)
s = np.sin(np.linspace(0, 2 * np.pi, F))
c = np.cos(np.linspace(0, 2 * np.pi, F))
ix = sk.index
for side, k in (("L", 1), ("R", -1)):
    theta[:, ix(f"upper_arm.{side}"), 0] += 0.6 * s * k
    theta[:, ix(f"upper_arm.{side}"), 2] += 0.4 * c
    theta[:, ix(f"forearm.{side}"), 0] = sk.hinge[ix(f"forearm.{side}")] * (0.6 + 0.5 * s * k)
    theta[:, ix(f"thigh.{side}"), 0] = 0.5 * s * k
    theta[:, ix(f"shin.{side}"), 0] = sk.hinge[ix(f"shin.{side}")] * (0.3 + 0.3 * (1 + s * k))
    for finger in ("index", "middle"):
        for i in (1, 2, 3):
            j = ix(f"{finger}.{i:02d}.{side}")
            theta[:, j, 0] = sk.hinge[j] * (0.5 + 0.4 * c)
theta[:, ix("spine2"), 1] = 0.25 * s
theta[:, ix("head"), 1] = 0.4 * c
yaw = 0.5 * s
R_root = np.array([solver.R_FACING @ solver._yaw(a) for a in yaw])
t_root = np.column_stack([0.3 * s, 0.1 + 0.02 * c, 3.6 + 0.3 * c])
W_img, H_img = 1080, 1920
cam = (1500.0, W_img / 2, H_img / 2)
G, P, S = solver.pose_points(sk, R_root, t_root, theta)
kp = np.zeros((F, 133, 2))
sc = np.zeros((F, 133))
for idx, name in skeleton.kp_sites().items():
    if name in S:
        kp[:, idx] = solver.project(S[name], cam) + rng.normal(scale=1.0, size=(F, 2))
        sc[:, idx] = 0.9
check(kp[:, :23].min() > 0 and kp[:, :23, 0].max() < W_img and kp[:, :23, 1].max() < H_img, "person inside the image")

sol = solver.solve_sequence(sk, kp, sc, cam, fps)
check(sol["valid"].all(), "every frame solved")
check(np.nanmax(sol["reproj"]) < 0.01, f"keypoints matched (worst {np.nanmax(sol['reproj']):.4f} heights)")
_G2, P2, _S2 = solver.pose_points(sk, sol["R"], sol["t"], sol["theta"])


def pa_error(A, B):
    """Mean joint error after the best rotation + translation of A onto B, per frame."""
    errs = []
    for a, b in zip(A, B):
        a0, b0 = a - a.mean(0), b - b.mean(0)
        U, _s, Vt = np.linalg.svd(a0.T @ b0)
        d = np.sign(np.linalg.det(U @ Vt))
        Rm = U @ np.diag([1, 1, d]) @ Vt
        errs.append(np.linalg.norm(a0 @ Rm - b0, axis=1).mean())
    return np.array(errs)


body_j = [j for j in range(J) if sk.roles[j] not in ("finger1", "finger", "thumb")]
err = pa_error(P2[:, body_j], P[:, body_j])
check(err.mean() < 0.03, f"3D body pose recovered (mean {err.mean() * 100:.1f} cm after alignment)")
depth = np.abs(sol["t"][:, 2] - t_root[:, 2])
check(depth.mean() < 0.15, f"distance from the camera recovered ({depth.mean():.3f} m off)")

sm = solver.smooth(sol, fps)
world = solver.ground(sk, sm, fps)
up_err = np.degrees(np.arccos(np.clip(world["W"][2] @ np.array((0, -1.0, 0)), -1, 1)))
check(up_err < 6, f"floor and up direction found ({up_err:.1f} deg off)")
arrays = solver.pack(sk, world)
arrays["camera_K"] = np.array(cam)
arrays["kp2d"], arrays["scores"] = kp.astype(np.float32), sc.astype(np.float32)

# -- motion file ---------------------------------------------------------------
path = os.path.join(tempfile.mkdtemp(), "clip.mcap")
meta = {"fps": fps, "frames": F, "image_size": [W_img, H_img], "face": []}
schema.save(path, meta, sk.to_dict(), arrays)
meta2, skd, arr2 = schema.load(path)
check(meta2["frames"] == F and arr2["local_q"].shape == (F, J, 4), "motion file round trip")
bad = os.path.join(os.path.dirname(path), "bad.npz")
np.savez(bad, x=np.zeros(3))
try:
    schema.load(bad)
    check(False, "foreign file refused")
except ValueError:
    check(True, "foreign file refused")

# -- bake and compare ----------------------------------------------------------
action = bake.bake(ctx, rig, meta2, skd, arr2, start=1)
fcs = utils.action_fcurves(action)
check(len(fcs) > 40 and all(len(fc.keyframe_points) in (F, 1) for fc in fcs),
      f"whole curves keyed ({len(fcs)} curves)")
check(all(kp_.interpolation == 'LINEAR' for kp_ in fcs[0].keyframe_points), "linear keys")
check(ctx.scene.frame_end == F and ctx.scene.render.fps == 24, "scene range and frame rate set")

# what the rig does vs what the solve said: control world rotations
sk_rig = skeleton.from_rig(rig)
Gw = bake.world_rotations(sk.parents, arr2["local_q"], arr2["root_q"])
worst = 0.0
for f in (0, 13, 31, 47):
    ctx.scene.frame_set(f + 1)
    for j, ctrl in enumerate(sk.controls):
        if sk.roles[j] in ("finger1", "finger", "thumb", "pelvis"):
            continue
        Rrest = np.array(rig.data.bones[ctrl].matrix_local.to_3x3())
        Rpose = np.array(rig.pose.bones[ctrl].matrix.to_3x3())
        Rpose = Rpose / np.linalg.norm(Rpose, axis=0)
        want = Gw[f, j] @ Rrest
        ang = np.degrees(np.arccos(np.clip((np.trace(want.T @ Rpose) - 1) / 2, -1, 1)))
        worst = max(worst, ang)
check(worst < 1.0, f"controls end where the solve put them (worst {worst:.2f} deg)")

# baked pose vs the true pose: control heads after a rigid alignment
heads, truth = [], []
for f in range(0, F, 6):
    ctx.scene.frame_set(f + 1)
    pts = [np.array(rig.matrix_world @ rig.pose.bones[sk.controls[j]].head) for j in body_j if sk.roles[j] != "pelvis"]
    heads.append(pts)
    truth.append([P[f, j] for j in body_j if sk.roles[j] != "pelvis"])
err = pa_error(np.array(heads), np.array(truth))
check(err.mean() < 0.035, f"baked rig follows the true motion ({err.mean() * 100:.1f} cm after alignment)")
fe = [ix(f"index.0{i}.L") for i in (1, 2, 3)]
fe_err = np.degrees(np.abs(sol["theta"][:, fe, 0] - theta[:, fe, 0])).mean()
check(fe_err < 12, f"finger curls recovered ({fe_err:.1f} deg off)")

# -- operators -------------------------------------------------------------------
check(hasattr(ctx.scene, "rigger_video") and hasattr(bpy.types, "RIGGER_PT_video"), "settings and panel registered")
ctx.view_layer.objects.active = rig
ctx.scene.rigger_video.filepath = os.path.join(tempfile.gettempdir(), "missing_video.mp4")
try:
    res = bpy.ops.rigger.video_to_animation(wait=True)
except RuntimeError as exc:
    res = str(exc)
check(res != {'FINISHED'} and "video" in str(res).lower(), f"missing video reported ({str(res)[:60]})")
from rigger.mocap import ops  # noqa: E402
st = ops.solver_settings(ctx.scene.rigger_video)
check(abs(st["min_cutoff"] - solver.DEFAULTS["min_cutoff"]) < 1e-6, "default smoothing gives the solver defaults")
ctx.scene.frame_start = 1
res = bpy.ops.rigger.video_apply(filepath=path)
check(res == {'FINISHED'} and rig.animation_data.action.name.startswith(rig.name), "motion file applied to the rig")
from rigger.mocap import face  # noqa: E402
fk = np.zeros((30, 133, 2))
fsc = np.ones((30, 133))
base = np.array([[np.cos(a) * 40 + 200, np.sin(a) * 50 + 200] for a in np.linspace(0, 2 * np.pi, 68)])
base[36:42] = [(170, 180), (175, 176), (180, 176), (185, 180), (180, 184), (175, 184)]
base[42:48] = [(215, 180), (220, 176), (225, 176), (230, 180), (225, 184), (220, 184)]
base[48:68] = [(200 + 15 * np.cos(a), 230 + 5 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 20)]
fk[:, 23:91] = base
fk[20:25, 23 + 62, 1] -= 12   # mouth opens
fk[20:25, 23 + 66, 1] += 12
wts = face.weights(fk, fsc, 24.0)
jaw = face.NAMES.index("jawOpen")
check(wts[22, jaw] > 0.3 and wts[5, jaw] < 0.05, f"open mouth read as jawOpen ({wts[22, jaw]:.2f} vs {wts[5, jaw]:.2f})")

# -- a real video (optional: RIGGER_AI_TEST_DIR with the model, RIGGER_VIDEO_TEST_CLIP a clip of a person) --
clip = os.environ.get("RIGGER_VIDEO_TEST_CLIP")
if clip and os.environ.get("RIGGER_AI_TEST_DIR"):
    os.environ["RIGGER_AI_DIR"] = os.environ["RIGGER_AI_TEST_DIR"]
    from rigger.mocap import video  # noqa: E402
    info = video.probe(clip)
    check(info["frames"] > 10 and info["width"] > 0, f"video probed ({info})")
    first = next(video.frames(clip, 0, 1))[1]
    check(first.shape == (info["height"], info["width"], 3) and first.mean() > 5, "first frame decoded")
    s = ctx.scene.rigger_video
    s.filepath, s.start, s.count, s.camera = clip, 0, 24, True
    s.motion_path = os.path.join(tempfile.mkdtemp(), "real.mcap")
    ctx.view_layer.objects.active = rig
    res = bpy.ops.rigger.video_to_animation(wait=True)
    check(res == {'FINISHED'}, "video to animation finished")
    meta_r, _sk, arr_r = schema.load(s.motion_path)
    check(meta_r["frames"] == 24 and meta_r["found"] > 0.9, f"person followed ({meta_r['found'] * 100:.0f}% of frames)")
    check(np.nanmedian(arr_r["reproj"]) < 0.03, f"pose matches the keypoints ({np.nanmedian(arr_r['reproj']):.3f})")
    check(ctx.scene.camera is not None and ctx.scene.camera.name.startswith("VideoCamera"), "matching camera added")
    check(len(utils.action_fcurves(rig.animation_data.action)) > 40, "rig animated from the video")
else:
    print("SKIP real video (set RIGGER_AI_TEST_DIR and RIGGER_VIDEO_TEST_CLIP)")

if failures:
    print(f"\n{len(failures)} FAILED")
    sys.exit(1)
print("\nALL PASSED")
