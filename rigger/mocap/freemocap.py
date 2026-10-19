# SPDX-License-Identifier: GPL-3.0-or-later
"""Import FreeMoCap recordings (free, open-source multi-camera motion capture).

FreeMoCap triangulates MediaPipe landmarks from several webcams and writes them
to its recording's output_data folder: mediapipe_body_3d_xyz (33 points) and
mediapipe_left/right_hand_3d_xyz (21 points each), in millimetres, as .npy
arrays or CSV files (wide 'body_nose_x' columns in FreeMoCap 1.x, long
frame/keypoint/x/y/z rows in 2.x).

The 3D points are projected into two virtual, perfectly known cameras (front
and side) and handed to the two-camera solver, so the character's own skeleton
is fitted to them with the same limits, smoothing, floor and foot planting as
video capture, and the result bakes like any motion file.
"""

import csv
import os

import numpy as np

# MediaPipe Pose landmark order
BODY = ("nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
        "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_heel", "right_heel", "left_foot_index", "right_foot_index")
HAND = ("wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_finger_mcp", "index_finger_pip",
        "index_finger_dip", "index_finger_tip", "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip",
        "middle_finger_tip", "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
        "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip")
# MediaPipe body landmark -> COCO-WholeBody keypoint (the solver's input layout)
BODY_TO_COCO = {"nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4, "left_shoulder": 5,
                "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
                "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14, "left_ankle": 15,
                "right_ankle": 16, "left_foot_index": 17, "left_heel": 19, "right_foot_index": 20, "right_heel": 22}
HAND_BASE = {"left": 91, "right": 112}   # the 21 hand points share COCO-WholeBody's hand order
FILES = {"body": "mediapipe_body_3d_xyz", "left": "mediapipe_left_hand_3d_xyz", "right": "mediapipe_right_hand_3d_xyz"}


def find_output_folder(path):
    """The folder holding the mediapipe_*_3d_xyz files, from a recording folder, its output_data or a file."""
    if os.path.isfile(path):
        path = os.path.dirname(path)
    for folder in (path, os.path.join(path, "output_data")):
        if any(os.path.isfile(os.path.join(folder, FILES["body"] + ext)) for ext in (".npy", ".csv")):
            return folder
    raise ValueError(f"no FreeMoCap body data (mediapipe_body_3d_xyz.npy/.csv) in {path}")


def _read_csv(path, names, prefix):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip() for h in rows[0]]
    body = [r for r in rows[1:] if r]
    low = [h.lower() for h in header]
    if {"x", "y", "z"} <= set(low) and ("keypoint" in low or "landmark" in low):   # long format (2.x)
        fi = low.index("frame") if "frame" in low else None
        ki = low.index("keypoint") if "keypoint" in low else low.index("landmark")
        xi, yi, zi = low.index("x"), low.index("y"), low.index("z")
        frames = sorted({int(float(r[fi])) for r in body}) if fi is not None else None
        index = {f: n for n, f in enumerate(frames)} if frames else {}
        out = np.full((len(frames) if frames else len(body) // len(names), len(names), 3), np.nan)
        for n, r in enumerate(body):
            name = r[ki].strip().lower().replace(prefix + "_", "")
            if name not in names:
                continue
            f = index[int(float(r[fi]))] if frames else n // len(names)
            out[f, names.index(name)] = [_num(r[xi]), _num(r[yi]), _num(r[zi])]
        return out
    out = np.full((len(body), len(names), 3), np.nan)           # wide format (1.x)
    for j, name in enumerate(names):
        for a, axis in enumerate("xyz"):
            col = f"{prefix}_{name}_{axis}"
            if col in header:
                c = header.index(col)
                out[:, j, a] = [_num(r[c]) for r in body]
    return out


def _num(text):
    try:
        return float(text)
    except ValueError:
        return np.nan


def load(path):
    """-> dict(body (F,33,3), left (F,21,3) or None, right (F,21,3) or None) in metres."""
    folder = find_output_folder(path)
    data = {}
    for key, stem in FILES.items():
        names, prefix = (BODY, "body") if key == "body" else (HAND, f"{key}_hand")
        npy, csv_path = os.path.join(folder, stem + ".npy"), os.path.join(folder, stem + ".csv")
        arr = None
        if os.path.isfile(npy):
            arr = np.load(npy, allow_pickle=False).astype(float)
            arr = arr.reshape(arr.shape[0], -1, 3)[:, :len(names)]
        elif os.path.isfile(csv_path):
            arr = _read_csv(csv_path, names, prefix)
        data[key] = None if arr is None else arr / 1000.0
    if data["body"] is None or not np.isfinite(data["body"]).any():
        raise ValueError("the FreeMoCap body data is empty")
    F = len(data["body"])
    for key in ("left", "right"):
        if data[key] is not None and len(data[key]) != F:
            data[key] = None
    return data


def world_frame(body):
    """Rotation taking FreeMoCap coordinates to Z-up with the person facing -Y at the start."""
    i = {n: k for k, n in enumerate(BODY)}
    mid_ankle = np.nanmean(body[:, [i["left_ankle"], i["right_ankle"]]], axis=1)
    mid_sh = np.nanmean(body[:, [i["left_shoulder"], i["right_shoulder"]]], axis=1)
    up = np.nanmedian(mid_sh - mid_ankle, axis=0)
    up /= np.linalg.norm(up)
    left = np.nanmedian(body[:30, i["left_shoulder"]] - body[:30, i["right_shoulder"]], axis=0)
    left -= up * np.dot(left, up)
    left /= np.linalg.norm(left)          # the person's left is world +X when facing -Y
    fwd = np.cross(left, up)              # the direction the person faces (left x up)
    return np.stack([left, -fwd, up])     # rows: world X, Y, Z in FreeMoCap coordinates


def to_views(data, distance=8.0, focal=4000.0, size=2000):
    """Project the 3D points into a front and a side virtual camera.

    Returns (kp_a, sc_a, cam, kp_b, sc_b, R_b, t_b) in the solver's camera conventions."""
    body = data["body"]
    F = len(body)
    W = world_frame(body)
    pts = np.full((F, 133, 3), np.nan)
    for name, k in BODY_TO_COCO.items():
        pts[:, k] = body[:, BODY.index(name)]
    for side in ("left", "right"):
        if data[side] is not None:
            pts[:, HAND_BASE[side]:HAND_BASE[side] + 21] = data[side]
    pts = pts @ W.T                                   # Z up, facing -Y
    centre = np.nanmedian(pts[:, [11, 12]].reshape(-1, 3), axis=0)
    pts = pts - centre
    # camera 1 in front (on -Y looking +Y), camera 2 at the person's left (on +X looking -X)
    to_cam_a = np.array(((1.0, 0, 0), (0, 0, -1), (0, 1, 0)))      # world -> camera (x right, y down, z fwd)
    to_cam_b = np.array(((0, 1.0, 0), (0, 0, -1), (-1, 0, 0)))
    off = np.array((0, 0, distance))
    cam = (focal, size / 2.0, size / 2.0)

    def project(R):
        X = pts @ R.T + off
        z = np.where(X[..., 2] > 0.1, X[..., 2], np.nan)
        kp = np.stack([focal * X[..., 0] / z + size / 2, focal * X[..., 1] / z + size / 2], axis=-1)
        ok = np.isfinite(kp).all(axis=-1)
        return np.nan_to_num(kp), np.where(ok, 0.95, 0.0)
    kp_a, sc_a = project(to_cam_a)
    kp_b, sc_b = project(to_cam_b)
    R_b = to_cam_b @ to_cam_a.T                        # camera-1 coords -> camera-2 coords
    t_b = off - R_b @ off
    return kp_a, sc_a, cam, kp_b, sc_b, R_b, t_b


def solve(sk, path, fps=30.0, settings=None, progress=None):
    """FreeMoCap recording -> (meta, arrays) of a motion file."""
    from . import pipeline, solver
    from . import face
    data = load(path)
    kp_a, sc_a, cam, kp_b, sc_b, R_b, t_b = to_views(data)
    settings = dict(settings or {})
    views = [(cam, R_b, t_b)]
    kp = np.stack([kp_a, kp_b], axis=1)
    sc = np.stack([sc_a, sc_b], axis=1)
    report = (lambda d, t: progress("solve", d, t)) if progress else None
    sol = solver.solve_sequence(sk, kp, sc, cam, fps, settings, report, views=views)
    sm = solver.smooth(sol, fps, settings)
    world = solver.ground(sk, sm, fps, dict(settings, level_camera=True, camera_tilt=0.0), views=views)
    if settings.get("floor_pass", solver.DEFAULTS["floor_pass"]) and world["floor_cam"]["contacts"].any():
        sol = solver.solve_sequence(sk, kp, sc, cam, fps, settings, report, floor=world["floor_cam"], views=views)
        sm = solver.smooth(sol, fps, settings)
        world = solver.ground(sk, sm, fps, dict(settings, level_camera=True, camera_tilt=0.0), views=views)
    arrays = solver.pack(sk, world)
    arrays.update({"kp2d": kp_a.astype(np.float32), "scores": sc_a.astype(np.float32), "reproj": sol["reproj"],
                   "face": np.zeros((len(kp_a), len(face.NAMES)))})
    meta = {"fps": float(fps), "frames": int(len(kp_a)), "image_size": [int(cam[1] * 2), int(cam[2] * 2)],
            "source": os.path.basename(os.path.normpath(path)), "face": list(face.NAMES),
            "settings": settings, "licenses": dict(pipeline.LICENSES, capture="FreeMoCap recording"),
            "found": float(np.mean(sol["valid"])), "freemocap": True}
    return meta, arrays
