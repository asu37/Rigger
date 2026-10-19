# SPDX-License-Identifier: GPL-3.0-or-later
"""Which body part is nearer the camera, read from the image itself.

A single photo does not say whether a hand is in front of or behind the body,
but its shading, texture and perspective do, and a monocular depth model has
learned to read them. Depth Anything V2 Small (Apache-2.0, run with
onnxruntime like the pose model) gives a relative depth map of the person's
crop; at every keypoint it is sampled, and for limb pairs whose depth clearly
differs (wrist vs shoulder, knee vs hip...) the solver is asked to keep that
order. The map is only relative, so only the order is used, never distances.

Optional and experimental (Video to Animation > Depth Cues): the model file must be
in the AI folder (MODEL_FILE). Measured on the rendered golden clips, its
near/far calls were right only 56-73% of the time (legs worst: the floor's
perspective reads as depth) and the pose error got slightly worse, so it is off
by default; real footage, which the model was trained on, may fare better.
"""

import os

import numpy as np

from .. import ai_markers
from . import perceive as perc

MODEL_FILE = "depth_anything_v2_small_fp16.onnx"
MODEL_URL = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_fp16.onnx"
MODEL_SHA256 = "2df6223f206b5164e21f664ace61dabeb9bb6a49b8b5a3e00510b4807d0f5b04"
IN_W, IN_H = 252, 336          # multiples of 14, same aspect as the pose model's crop
MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)
# keypoint pairs whose depth order settles the usual front/back ambiguities (COCO-WholeBody indices)
PAIRS = ((9, 5), (7, 5), (10, 6), (8, 6), (9, 7), (10, 8), (13, 11), (14, 12), (15, 13), (16, 14),
         (15, 11), (16, 12), (9, 11), (10, 12), (9, 0), (10, 0), (7, 11), (8, 12))
MIN_SCORE = 0.5
THRESHOLD = 0.12               # share of the person's depth range a pair must differ by

_session = {}


def model_path():
    return os.path.join(ai_markers.ai_dir(), MODEL_FILE)


def available():
    return ai_markers.load_runtime() is not None and os.path.isfile(model_path())


def session():
    path = model_path()
    if _session.get("path") != path:
        ort = ai_markers.load_runtime()
        if ort is None or not os.path.isfile(path):
            raise RuntimeError(f"depth model not found: {path}")
        _session["path"] = path
        _session["session"] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return _session["session"]


def estimate(rgb):
    """Relative inverse depth (larger = nearer) of an RGB crop, at (IN_H, IN_W)."""
    s = session()
    x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    return s.run(None, {s.get_inputs()[0].name: x})[0][0]


def sample(img, box, kp):
    """Depth at each keypoint of one frame. img: the frame, box: (cx, cy, h) crop, kp (133,2) pixels."""
    cx, cy, h = box
    crop = perc.crop(img, cx, cy, h)                       # (H_IN, W_IN) pose-model crop
    small = _resize(crop, IN_W, IN_H)
    disp = estimate(small)
    s = h / perc.H_IN
    u = (kp[:, 0] - cx) / s + perc.W_IN / 2
    v = (kp[:, 1] - cy) / s + perc.H_IN / 2
    u = np.clip(u * IN_W / perc.W_IN, 0, IN_W - 1)
    v = np.clip(v * IN_H / perc.H_IN, 0, IN_H - 1)
    # 3x3 median around each keypoint, robust to edges
    out = np.empty(len(kp), np.float32)
    for i, (x, y) in enumerate(zip(u.astype(int), v.astype(int))):
        out[i] = np.median(disp[max(0, y - 1):y + 2, max(0, x - 1):x + 2])
    return out


def _resize(img, w, h):
    H, W = img.shape[:2]
    ys = (np.arange(h) + 0.5) * H / h - 0.5
    xs = (np.arange(w) + 0.5) * W / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None, None]
    fx = np.clip(xs - x0, 0, 1)[None, :, None]
    top = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    bot = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return top * (1 - fy) + bot * fy


def track(frame_iter, boxes, kp, total, progress=None):
    """Depth at the keypoints of every frame (F,133), using the crop boxes the tracker used."""
    out = np.zeros((total, kp.shape[1]), np.float32)
    for n, (_i, img) in enumerate(frame_iter):
        if n >= total:
            break
        if boxes[n][2] > 0:
            out[n] = sample(img.astype(np.float32), boxes[n], kp[n])
        if progress:
            progress(n + 1, total)
    return out


def pairs_for(depth, scores, settings=None):
    """Per frame: list of (kp_a, kp_b, sign, weight) where sign=+1 means kp_a is farther than kp_b."""
    settings = settings or {}
    thr = settings.get("depth_threshold", THRESHOLD)
    out = []
    body = list(range(23))
    for d, sc in zip(depth, scores):
        good = [k for k in body if sc[k] >= MIN_SCORE]
        if len(good) < 6 or not np.any(d[good]):
            out.append(None)
            continue
        lo, hi = np.percentile(d[good], 5), np.percentile(d[good], 95)
        span = max(hi - lo, 1e-6)
        frame = []
        for a, b in PAIRS:
            if sc[a] < MIN_SCORE or sc[b] < MIN_SCORE:
                continue
            diff = (d[a] - d[b]) / span            # > 0: a nearer
            if abs(diff) < thr:
                continue
            frame.append((a, b, -1.0 if diff > 0 else 1.0, float(min(1.0, (abs(diff) - thr) / thr))))
        out.append(frame or None)
    return out
