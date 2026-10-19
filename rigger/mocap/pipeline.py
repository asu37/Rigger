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
            "depth": "Depth Anything V2 Small (Apache-2.0), when installed",
            "solver": "Rigger (GPL-3.0-or-later)"}


def focal_pixels(width, height, focal_mm):
    """Focal length in pixels for a 35 mm-equivalent focal length (36 mm across the long side)."""
    return focal_mm / 36.0 * max(width, height)


def perceive(path, start=0, count=None, max_side=1280, cache_dir=None, progress=None, depth=False):
    """-> dict(kp, scores, boxes, fps, width, height, hash[, depth]). Cached by content hash.

    depth: also read the depth model at every keypoint (see depth.py), cached separately."""
    seen = _perceive(path, start, count, max_side, cache_dir, progress)
    if depth:
        from . import depth as depth_cues
        from . import video
        if depth_cues.available():
            cache = os.path.join(cache_dir, seen["hash"] + "_depth.npz") if cache_dir else None
            if cache and os.path.isfile(cache):
                with np.load(cache) as d:
                    seen["depth"] = d["depth"]
            else:
                total = len(seen["kp"])
                report = (lambda d, t: progress("depth", d, t)) if progress else None
                seen["depth"] = depth_cues.track(video.frames(path, start, total, max_side), seen["boxes"],
                                                 np.asarray(seen["kp"], float), total, report)
                if cache:
                    np.savez_compressed(cache, depth=seen["depth"])
    return seen


def _perceive(path, start=0, count=None, max_side=1280, cache_dir=None, progress=None):
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


def solve(sk, seen, focal_mm=DEFAULT_FOCAL_MM, settings=None, progress=None, source="", second=None):
    """Keypoints (from perceive) -> (meta, arrays) of a motion file.

    second: optional second camera of the same take: dict(seen=perceive result, focal_mm,
    offset=seconds it lags the first video). Both are then solved together."""
    settings = dict(settings or {})
    fps = float(seen["fps"])
    w, h = seen["width"], seen["height"]
    cam = (focal_pixels(w, h, focal_mm), w / 2.0, h / 2.0)
    kp, sc = np.asarray(seen["kp"], float), np.asarray(seen["scores"], float)
    t0 = time.time()
    views, info = None, {}
    depth = None
    if seen.get("depth") is not None and settings.get("use_depth", False):
        from . import depth as depth_cues
        depth = depth_cues.pairs_for(seen["depth"], sc, settings)
    if second:
        from . import multiview
        s2 = second["seen"]
        cam2 = (focal_pixels(s2["width"], s2["height"], second.get("focal_mm", focal_mm)),
                s2["width"] / 2.0, s2["height"] / 2.0)
        kp2, sc2 = multiview.align(np.asarray(s2["kp"], float), np.asarray(s2["scores"], float),
                                   fps, float(s2["fps"]), len(kp), second.get("offset", 0.0))
        sol, views, (kp_used, sc_used), info, cam = multiview.solve(sk, kp, sc, cam, kp2, sc2, cam2, fps, settings,
                                                                    progress)
        if "focal_mm" in info:
            focal_mm = info["focal_mm"]
    else:
        kp_used, sc_used = kp, sc
        report = (lambda d, t: progress("solve", d, t)) if progress else None
        sol = solver.solve_sequence(sk, kp, sc, cam, fps, settings, report, depth=depth)
    sm = solver.smooth(sol, fps, settings)
    world = solver.ground(sk, sm, fps, settings, views=views)
    if settings.get("floor_pass", solver.DEFAULTS["floor_pass"]) and world["floor_cam"]["contacts"].any():
        report2 = (lambda d, t: progress("floor", d, t)) if progress else None
        sol = solver.solve_sequence(sk, kp_used, sc_used, cam, fps, settings, report2, floor=world["floor_cam"],
                                    views=views, depth=None if views else depth)
        sm = solver.smooth(sol, fps, settings)
        world = solver.ground(sk, sm, fps, settings, views=views)
    arrays = solver.pack(sk, world)
    arrays.update({"camera_K": np.array(cam), "kp2d": kp.astype(np.float32), "scores": sc.astype(np.float32),
                   "reproj": sol["reproj"], "face": face.weights(kp, sc, fps)})
    if views:
        arrays.update({"camera2_K": np.array(views[0][0]), "camera2_W": world["views_W"][0],
                       "camera2_pos": world["views_pos"][0]})
    meta = {"fps": fps, "frames": int(len(kp)), "image_size": [w, h], "source": os.path.basename(source),
            "hash": seen.get("hash", ""), "face": list(face.NAMES), "focal_mm": focal_mm,
            "settings": settings, "licenses": LICENSES, "solve_seconds": round(time.time() - t0, 2),
            "found": float(np.mean(sol["valid"]))}
    if second:
        meta.update({"cameras": 2, "image_size2": [second["seen"]["width"], second["seen"]["height"]],
                     "source2": os.path.basename(second.get("source", "")), "offset2": second.get("offset", 0.0),
                     "calibration": {k: round(float(v), 4) for k, v in info.items() if np.ndim(v) == 0}})
    if depth is not None:
        meta["depth_cues"] = True
    return meta, arrays


def run(path, sk, out=None, focal_mm=DEFAULT_FOCAL_MM, start=0, count=None, max_side=1280,
        cache_dir=None, settings=None, progress=None, video2=None, offset2=None):
    """The whole video -> motion pipeline. Saves to 'out' when given; returns (meta, skeleton dict, arrays).

    video2: optional second camera of the same take. offset2: seconds it lags the first video;
    None finds it from the sound of both files (0 when they have none)."""
    if isinstance(sk, dict):
        sk = Skeleton.from_dict(sk)
    use_depth = bool((settings or {}).get("use_depth", False)) and not video2
    seen = perceive(path, start, count, max_side, cache_dir, progress, depth=use_depth)
    if not np.any(seen["scores"][:, :17].mean(axis=1) >= 0.3):
        raise ValueError("no person found in the video")
    second = None
    if video2:
        from . import multiview
        if offset2 is None:
            offset2 = multiview.audio_offset(path, video2)
            if offset2 is None:
                offset2 = 0.0
        report2 = (lambda stage, d, t: progress("keypoints 2", d, t)) if progress else None
        seen2 = perceive(video2, 0, None, max_side, cache_dir, report2)
        # the second clip is used from the moment matching the first clip's start frame
        second = {"seen": seen2, "focal_mm": focal_mm, "source": video2,
                  "offset": float(offset2) + start / float(seen["fps"])}
    meta, arrays = solve(sk, seen, focal_mm, settings, progress, source=path, second=second)
    meta["start"] = start
    if out:
        schema.save(out, meta, sk.to_dict(), arrays)
    return meta, sk.to_dict(), arrays


def summary(meta):
    return json.dumps({k: meta[k] for k in ("frames", "fps", "found", "solve_seconds") if k in meta})
