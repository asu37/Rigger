# SPDX-License-Identifier: GPL-3.0-or-later
"""The .mcap motion file: everything between the video and the rig (numpy .npz).

    meta      JSON: format, version, fps, frame count, image size, source file and hash,
              the backends that produced it and their licenses
    skeleton  JSON: the skeleton the motion was solved on (skeleton.Skeleton.to_dict)
    root_t    (F,3) world position of the root joint, metres, Z up, floor at 0
    root_q    (F,4) world rotation of the root, wxyz, rest-aligned
    local_q   (F,J,4) every joint's rotation relative to its parent, wxyz, rest-aligned
    valid     (F,) a person was found in the frame
    contacts  (F,2) left / right foot on the floor
    face      (F,C) expression weights 0..1, names in meta["face"] (ARKit names)
    camera    f, cx, cy (pixels), W (camera->world rotation), cam_pos (world); camera2_* the same for
              the second camera of a two-camera solve
    kp2d, scores  (F,133,2), (F,133): the raw COCO-WholeBody keypoints, kept for
              re-solving with other settings and for checking the result
"""

import json

import numpy as np

FORMAT = "rigger-mcap"
VERSION = 1
ARRAYS = ("root_t", "root_q", "local_q", "valid", "contacts", "face", "camera_W", "camera_pos", "camera_K",
          "kp2d", "scores", "reproj", "camera2_W", "camera2_pos", "camera2_K", "depth")


def save(path, meta, skeleton, arrays):
    meta = dict(meta, format=FORMAT, version=VERSION)
    payload = {"meta": np.array(json.dumps(meta)), "skeleton": np.array(json.dumps(skeleton))}
    for key, value in arrays.items():
        if key not in ARRAYS:
            raise KeyError(f"unknown mcap array '{key}'")
        if value is not None:
            payload[key] = np.asarray(value)
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **payload)


def load(path):
    """-> (meta dict, skeleton dict, arrays dict). Raises ValueError on a foreign or broken file."""
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data.files:
            raise ValueError(f"{path}: not a Rigger motion file")
        meta = json.loads(str(data["meta"]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path}: not a Rigger motion file")
        if meta.get("version", 0) > VERSION:
            raise ValueError(f"{path}: made by a newer Rigger (format {meta['version']})")
        skeleton = json.loads(str(data["skeleton"]))
        arrays = {k: data[k] for k in data.files if k in ARRAYS}
    validate(meta, skeleton, arrays)
    return meta, skeleton, arrays


def validate(meta, skeleton, arrays):
    F = int(meta["frames"])
    J = len(skeleton["joints"])
    shapes = {"root_t": (F, 3), "root_q": (F, 4), "local_q": (F, J, 4), "valid": (F,)}
    for key, shape in shapes.items():
        if key not in arrays:
            raise ValueError(f"motion file lacks '{key}'")
        if tuple(arrays[key].shape) != shape:
            raise ValueError(f"motion file: '{key}' has shape {arrays[key].shape}, expected {shape}")
    if "face" in arrays and arrays["face"].shape[0] != F:
        raise ValueError("motion file: face weights do not match the frame count")
    for key in ("root_t", "root_q", "local_q"):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"motion file: '{key}' has invalid numbers")
