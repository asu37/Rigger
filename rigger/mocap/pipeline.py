# SPDX-License-Identifier: GPL-3.0-or-later
"""Video file -> motion file: decode, keypoints, face, 3D solve, floor, contacts.

Keypoints are cached per video (content hash) so re-solving with other
settings or for another character skips the slow part.
"""

import json
import os
import time

import numpy as np

from . import face, schema, solver
from .skeleton import Skeleton

DEFAULT_FOCAL_MM = 26.0   # 35 mm-equivalent focal length of a typical phone camera
LICENSES = {"decoder": "Blender FFmpeg (GPL)", "keypoints": "DWPose (Apache-2.0)",
            "solver": "Rigger (GPL-3.0-or-later)"}


def focal_pixels(width, height, focal_mm):
    """Focal length in pixels for a 35 mm-equivalent focal length (36 mm across the long side)."""
    return focal_mm / 36.0 * max(width, height)


def perceive(path, start=0, count=None, max_side=1280, cache_dir=None, progress=None):
    """-> dict(kp, scores, boxes, fps, width, height, hash). Cached by content hash."""
    from . import perceive as perc
    from . import video
    from .. import ai_markers
    info = video.probe(path)
    total = info["frames"] - start if count is None else min(count, info["frames"] - start)
    if total < 1:
        raise ValueError("no frames in the chosen range")
    scale = min(1.0, max_side / max(info["width"], info["height"]))
    w, h = int(round(info["width"] * scale)), int(round(info["height"] * scale))
    key = f"{video.file_hash(path)}_{start}_{total}_{max_side}_{ai_markers.MODEL_FILE.split('.')[0]}"
    cache = os.path.join(cache_dir, key + ".npz") if cache_dir else None
    if cache and os.path.isfile(cache):
        with np.load(cache) as d:
            return {"kp": d["kp"], "scores": d["scores"], "boxes": d["boxes"], "fps": info["fps"],
                    "width": w, "height": h, "hash": key, "cached": True}
    report = (lambda d, t: progress("keypoints", d, t)) if progress else None
    kp, sc, boxes = perc.track(video.frames(path, start, total, max_side), total, report)
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache, kp=kp, scores=sc, boxes=boxes)
    return {"kp": kp, "scores": sc, "boxes": boxes, "fps": info["fps"], "width": w, "height": h,
            "hash": key, "cached": False}


def solve(sk, seen, focal_mm=DEFAULT_FOCAL_MM, settings=None, progress=None, source=""):
    """Keypoints (from perceive) -> (meta, arrays) of a motion file."""
    settings = dict(settings or {})
    fps = float(seen["fps"])
    w, h = seen["width"], seen["height"]
    cam = (focal_pixels(w, h, focal_mm), w / 2.0, h / 2.0)
    kp, sc = np.asarray(seen["kp"], float), np.asarray(seen["scores"], float)
    report = (lambda d, t: progress("solve", d, t)) if progress else None
    t0 = time.time()
    sol = solver.solve_sequence(sk, kp, sc, cam, fps, settings, report)
    sm = solver.smooth(sol, fps, settings)
    world = solver.ground(sk, sm, fps, settings)
    if settings.get("floor_pass", solver.DEFAULTS["floor_pass"]) and world["floor_cam"]["contacts"].any():
        report2 = (lambda d, t: progress("floor", d, t)) if progress else None
        sol = solver.solve_sequence(sk, kp, sc, cam, fps, settings, report2, floor=world["floor_cam"])
        sm = solver.smooth(sol, fps, settings)
        world = solver.ground(sk, sm, fps, settings)
    arrays = solver.pack(sk, world)
    arrays.update({"camera_K": np.array(cam), "kp2d": kp.astype(np.float32), "scores": sc.astype(np.float32),
                   "reproj": sol["reproj"], "face": face.weights(kp, sc, fps)})
    meta = {"fps": fps, "frames": int(len(kp)), "image_size": [w, h], "source": os.path.basename(source),
            "hash": seen.get("hash", ""), "face": list(face.NAMES), "focal_mm": focal_mm,
            "settings": settings, "licenses": LICENSES, "solve_seconds": round(time.time() - t0, 2),
            "found": float(np.mean(sol["valid"]))}
    return meta, arrays


def run(path, sk, out=None, focal_mm=DEFAULT_FOCAL_MM, start=0, count=None, max_side=1280,
        cache_dir=None, settings=None, progress=None):
    """The whole video -> motion pipeline. Saves to 'out' when given; returns (meta, skeleton dict, arrays)."""
    if isinstance(sk, dict):
        sk = Skeleton.from_dict(sk)
    seen = perceive(path, start, count, max_side, cache_dir, progress)
    if not np.any(seen["scores"][:, :17].mean(axis=1) >= 0.3):
        raise ValueError("no person found in the video")
    meta, arrays = solve(sk, seen, focal_mm, settings, progress, source=path)
    meta["start"] = start
    if out:
        schema.save(out, meta, sk.to_dict(), arrays)
    return meta, sk.to_dict(), arrays


def summary(meta):
    return json.dumps({k: meta[k] for k in ("frames", "fps", "found", "solve_seconds") if k in meta})
