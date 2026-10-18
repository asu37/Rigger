# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-frame whole-body keypoints of the person in a video (numpy + onnxruntime).

Uses the DWPose model of AI Markers. The person is followed with a crop box
taken from the previous frame's keypoints, steadied over time; each frame is
detected twice (predicted box, then the box of that result) so fast moves stay
in the crop. When the person is lost, the whole frame and its thirds are
searched again.
"""

import numpy as np

from .. import ai_markers

W_IN, H_IN = ai_markers.INPUT_W, ai_markers.INPUT_H
ASPECT = W_IN / H_IN
TRACK_SCORE = 0.35       # mean body keypoint score below which the person counts as lost


def crop(img, cx, cy, h):
    """Bilinear crop of a (H_IN, W_IN) window of height h centred on (cx, cy); zeros outside."""
    s = h / H_IN
    xs = cx + (np.arange(W_IN) + 0.5 - W_IN / 2) * s - 0.5
    ys = cy + (np.arange(H_IN) + 0.5 - H_IN / 2) * s - 0.5
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = (xs - x0)[None, :, None]
    fy = (ys - y0)[:, None, None]
    H, W = img.shape[:2]
    pad = np.zeros((H + 2, W + 2, 3), np.float32)
    pad[1:-1, 1:-1] = img
    xa = np.clip(x0 + 1, 0, W + 1)
    xb = np.clip(x0 + 2, 0, W + 1)
    ya = np.clip(y0 + 1, 0, H + 1)
    yb = np.clip(y0 + 2, 0, H + 1)
    top = pad[ya][:, xa] * (1 - fx) + pad[ya][:, xb] * fx
    bot = pad[yb][:, xa] * (1 - fx) + pad[yb][:, xb] * fx
    return top * (1 - fy) + bot * fy


def detect_box(img, box):
    cx, cy, h = box
    pts, sc = ai_markers.detect(crop(img, cx, cy, h), subpixel=True)
    s = h / H_IN
    out = np.empty_like(pts)
    out[:, 0] = cx + (pts[:, 0] - W_IN / 2) * s
    out[:, 1] = cy + (pts[:, 1] - H_IN / 2) * s
    return out, sc


def box_from(pts, sc, min_score=0.3):
    """Crop box (cx, cy, h) around the confident body, face and feet keypoints."""
    use = np.concatenate([np.arange(0, 23), np.arange(23, 91, 4)])
    good = pts[use][sc[use] >= min_score]
    if len(good) < 4:
        return None
    lo, hi = good.min(axis=0), good.max(axis=0)
    w, h = hi - lo
    size = max(h, w / ASPECT) * ai_markers.PADDING
    centre = (lo + hi) / 2
    return np.array((centre[0], centre[1], max(size, 32.0)))


def _body_score(sc):
    return float(np.mean(sc[:17]))


def search(img):
    """Find the person without a previous box: the whole frame, then its thirds."""
    H, W = img.shape[:2]
    full = (W / 2, H / 2, max(H, W / ASPECT))
    boxes = [full]
    if W > H:
        third = max(H, W / 3 / ASPECT)
        boxes += [(W * f, H / 2, third) for f in (1 / 6, 1 / 2, 5 / 6)]
    best = None
    for b in boxes:
        pts, sc = detect_box(img, b)
        nb = box_from(pts, sc)
        if nb is None:
            continue
        pts, sc = detect_box(img, nb)
        if best is None or _body_score(sc) > _body_score(best[1]):
            best = (pts, sc, box_from(pts, sc) if box_from(pts, sc) is not None else nb)
    return best


class Tracker:
    def __init__(self, smooth_centre=0.6, smooth_size=0.35):
        self.box = None
        self.vel = np.zeros(2)
        self.a_c, self.a_s = smooth_centre, smooth_size

    def step(self, img):
        """-> (keypoints (133,2), scores (133,), box used)."""
        if self.box is None:
            found = search(img)
            if found is None:
                return np.zeros((133, 2)), np.zeros(133), None
            pts, sc, box = found
            self.box = box
            return pts, sc, box
        pred = self.box.copy()
        pred[:2] += self.vel
        pts, sc = detect_box(img, pred)
        nb = box_from(pts, sc)
        if nb is None or _body_score(sc) < TRACK_SCORE:
            self.box = None
            self.vel[:] = 0
            return self.step(img)
        # second pass on the box of the first result catches fast moves
        pts2, sc2 = detect_box(img, nb)
        if _body_score(sc2) >= _body_score(sc):
            pts, sc = pts2, sc2
            nb = box_from(pts, sc) if box_from(pts, sc) is not None else nb
        new = self.box.copy()
        new[:2] = self.box[:2] + self.a_c * (nb[:2] - self.box[:2])
        new[2] = self.box[2] + self.a_s * (nb[2] - self.box[2])
        new[2] = max(new[2], nb[2] * 0.9)  # never shrink below the person
        self.vel = 0.5 * self.vel + 0.5 * (new[:2] - self.box[:2])
        self.box = new
        return pts, sc, nb


def track(frame_iter, total, progress=None):
    """-> kp (F,133,2) image pixels, scores (F,133), boxes (F,3)."""
    kp = np.zeros((total, 133, 2), np.float32)
    sc = np.zeros((total, 133), np.float32)
    boxes = np.zeros((total, 3), np.float32)
    tracker = Tracker()
    for n, (i, img) in enumerate(frame_iter):
        pts, s, box = tracker.step(img.astype(np.float32))
        k = n if n < total else total - 1
        kp[k], sc[k] = pts, s
        if box is not None:
            boxes[k] = box
        if progress:
            progress(n + 1, total)
        del i
    return kp, sc, boxes
