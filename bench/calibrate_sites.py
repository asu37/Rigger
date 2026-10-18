"""Calibrate where the pose model's keypoints sit on the skeleton.

For every site, the offset (in its joint's frame, in body heights) that best
explains the model's detections on a golden clip, given the true pose:

    blender -b --factory-startup --python bench/calibrate_sites.py -- --body realistic_male

Prints the offsets relative to the current site model, as (inward, forward, up)
at rest, ready to fold into SITE_SHIFT in rigger/mocap/skeleton.py. Calibrate on
one body and check on another (video_bench.py eval).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
sys.argv = sys.argv[:1]

import bpy  # noqa: E402

import rigger  # noqa: E402
from rigger.mocap import solver  # noqa: E402
from rigger.mocap.skeleton import Skeleton, from_rig, kp_sites  # noqa: E402


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rigger.register()
    name = ARGS[ARGS.index("--body") + 1] if "--body" in ARGS else "realistic_male"
    out = os.path.join(HERE, "results", "video", name)
    truth = dict(np.load(os.path.join(out, "truth.npz")))
    cache = os.path.join(out, "cache")
    seen = np.load(os.path.join(cache, sorted(os.listdir(cache))[0]))
    import run_bench as rb
    _body, rig, _sec = rb.build(name, rb.CONFIGS[next(iter(rb.CONFIGS))])
    sk = from_rig(rig)
    sk_truth = Skeleton.from_dict(json.loads(str(truth["skeleton"])))
    assert sk.joints == sk_truth.joints
    M = truth["cam"]
    Rwc = np.diag([1, -1, -1]) @ M[:3, :3].T
    R = np.einsum('ij,fjk->fik', Rwc, truth["root_R"])
    t = (truth["root_t"] - M[:3, 3]) @ Rwc.T
    G, P, _S = solver.pose_points(sk, R, t, truth["theta"])
    f_px = 35.0 / 36.0 * 768
    cam = (f_px, 432 / 2, 768 / 2)
    H = sk.height
    result = {}
    for k, site in kp_sites().items():
        if site not in sk.sites or k >= 91:
            continue
        j, pos = sk.sites[site]
        ok = seen["scores"][:, k] > 0.3
        if ok.sum() < 10:
            continue
        uv = seen["kp"][ok, k]
        Gj, Pj = G[ok, j], P[ok, j]
        base = pos - sk.rest[j]

        def residual(X):
            off = base[None] + X * H
            pts = Pj[None] + np.einsum('fab,nb->nfa', Gj, off)
            return (solver.project(pts, cam) - uv[None]).reshape(len(X), -1)
        x, cost = solver.levenberg_marquardt(residual, np.zeros(3), 20)
        before = np.abs(residual(np.zeros((1, 3)))[0]).reshape(-1, 2)
        after = np.abs(residual(x[None])[0]).reshape(-1, 2)
        inward = -1.0 if pos[0] > 0 else 1.0 if pos[0] < 0 else 0.0
        result[site] = (round(float(x[0] * inward), 4), round(float(-x[1]), 4), round(float(x[2]), 4))
        print(f"{site:14s} shift (in, fwd, up) {result[site]}  px err {np.linalg.norm(before, axis=1).mean():.1f}"
              f" -> {np.linalg.norm(after, axis=1).mean():.1f}")
    with open(os.path.join(out, "site_calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)


main()
