# SPDX-License-Identifier: GPL-3.0-or-later
"""Facial expression weights from the 68 face landmarks of the pose model.

Measured in eye distances so the face's size and distance do not matter, and
relative to the clip's own resting face (the median), so a neutral face reads
as zero. Output names follow ARKit, so they drive matching shape keys.
"""

import numpy as np

from . import filters

NAMES = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight",
         "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "mouthPucker")
FACE0 = 23          # first face landmark in COCO-WholeBody
MIN_SCORE = 0.4


def _d(a, b):
    return np.linalg.norm(a - b, axis=-1)


def measures(p):
    """p (F,68,2) -> dict of per-frame measurements (in eye distances)."""
    eye_r, eye_l = p[:, 36:42].mean(axis=1), p[:, 42:48].mean(axis=1)
    iod = np.maximum(_d(eye_l, eye_r), 1e-3)
    mouth = p[:, 48:68].mean(axis=1)
    up = eye_l / 2 + eye_r / 2 - mouth
    up /= np.maximum(np.linalg.norm(up, axis=-1, keepdims=True), 1e-6)

    def height(a, b):  # how far a is above b along the face's up direction
        return np.sum((a - b) * up, axis=-1) / iod

    def ear(e):  # eye opening / eye width
        return (_d(e[:, 1], e[:, 5]) + _d(e[:, 2], e[:, 4])) / (2 * np.maximum(_d(e[:, 0], e[:, 3]), 1e-3))
    return {
        "ear_l": ear(p[:, 42:48]), "ear_r": ear(p[:, 36:42]),
        "mouth_open": _d(p[:, 62], p[:, 66]) / iod,
        "mouth_w": _d(p[:, 48], p[:, 54]) / iod,
        "lift_l": height(p[:, 54], mouth), "lift_r": height(p[:, 48], mouth),
        "brow_in": (height(p[:, 22], eye_l) + height(p[:, 21], eye_r)) / 2,
        "brow_out_l": height(p[:, 26], eye_l), "brow_out_r": height(p[:, 17], eye_r),
    }


def weights(kp, sc, fps):
    """kp (F,133,2), sc (F,133) -> (F, len(NAMES)) weights 0..1 (zero where the face is not seen)."""
    p = np.asarray(kp[:, FACE0:FACE0 + 68], float)
    seen = np.asarray(sc[:, FACE0:FACE0 + 68]).mean(axis=1) >= MIN_SCORE
    F = len(p)
    out = np.zeros((F, len(NAMES)))
    if seen.sum() < 3:
        return out
    m = measures(p)
    ref = {k: np.median(v[seen]) for k, v in m.items()}
    open_l, open_r = np.percentile(m["ear_l"][seen], 85), np.percentile(m["ear_r"][seen], 85)
    clip = lambda x: np.clip(x, 0.0, 1.0)  # noqa: E731
    smile_w = (m["mouth_w"] - ref["mouth_w"]) / 0.3
    out[:, 0] = clip((open_l - m["ear_l"]) / (0.6 * open_l))
    out[:, 1] = clip((open_r - m["ear_r"]) / (0.6 * open_r))
    out[:, 2] = clip((m["mouth_open"] - ref["mouth_open"]) / 0.55)
    out[:, 3] = clip(0.5 * smile_w + (m["lift_l"] - ref["lift_l"]) / 0.12)
    out[:, 4] = clip(0.5 * smile_w + (m["lift_r"] - ref["lift_r"]) / 0.12)
    out[:, 5] = clip((m["brow_in"] - ref["brow_in"]) / 0.12)
    out[:, 6] = clip((m["brow_out_l"] - ref["brow_out_l"]) / 0.12)
    out[:, 7] = clip((m["brow_out_r"] - ref["brow_out_r"]) / 0.12)
    out[:, 8] = clip((ref["mouth_w"] - m["mouth_w"]) / 0.25)
    out[~seen] = 0.0
    return np.clip(filters.one_euro_zero_lag(out, fps, 2.0, 0.5), 0.0, 1.0)
