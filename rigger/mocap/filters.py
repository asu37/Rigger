# SPDX-License-Identifier: GPL-3.0-or-later
"""Signal filters for solved motion (numpy only).

One-Euro (Casiez et al. 2012): a low-pass filter whose cutoff rises with speed,
so slow motion loses its jitter while fast motion keeps up. Run forwards and
backwards and averaged, it has no lag, which an offline solve can afford.
Rotations are filtered as sign-continuous quaternions.
"""

import numpy as np

from .skeleton import matrix_to_quat, quat_to_matrix


def _alpha(cutoff, fps):
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau * fps)


def one_euro(x, fps, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
    """Causal One-Euro over axis 0 of x (F, ...)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return x.copy()
    out = np.empty_like(x)
    out[0] = x[0]
    dx = np.zeros_like(x[0])
    a_d = _alpha(d_cutoff, fps)
    for i in range(1, len(x)):
        raw_d = (x[i] - out[i - 1]) * fps
        dx = dx + a_d * (raw_d - dx)
        cutoff = min_cutoff + beta * np.abs(dx)
        a = _alpha(cutoff, fps)
        out[i] = out[i - 1] + a * (x[i] - out[i - 1])
    return out


def one_euro_zero_lag(x, fps, min_cutoff=1.0, beta=0.0):
    """Forward and backward One-Euro, averaged: no lag, ends untouched by the start-up transient."""
    x = np.asarray(x, float)
    fwd = one_euro(x, fps, min_cutoff, beta)
    bwd = one_euro(x[::-1], fps, min_cutoff, beta)[::-1]
    return 0.5 * (fwd + bwd)


def continuous_quats(q):
    """Flip quaternion signs so consecutive ones are on the same side (F, ..., 4)."""
    q = np.array(q, float)
    for i in range(1, len(q)):
        flip = np.sum(q[i] * q[i - 1], axis=-1) < 0
        q[i] = np.where(flip[..., None], -q[i], q[i])
    return q


def smooth_rotations(R, fps, min_cutoff=1.0, beta=0.0):
    q = continuous_quats(matrix_to_quat(R))
    q = one_euro_zero_lag(q, fps, min_cutoff, beta)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return quat_to_matrix(q)


def clean_mask(mask, min_run=2):
    """Drop True runs and fill False gaps shorter than min_run."""
    m = np.asarray(mask, bool).copy()
    for value in (False, True):  # fill short gaps first, then drop short runs
        i = 0
        while i < len(m):
            if m[i] == value:
                j = i
                while j < len(m) and m[j] == value:
                    j += 1
                inner = i > 0 and j < len(m)
                if (j - i) < min_run and (inner or value):
                    m[i:j] = not value
                i = j
            else:
                i += 1
    return m


def rdp(values, tolerance):
    """Indices kept by Ramer-Douglas-Peucker on a (F, C) curve sampled once per frame."""
    values = np.asarray(values, float)
    F = len(values)
    if F < 3:
        return np.arange(F)
    keep = np.zeros(F, bool)
    keep[0] = keep[-1] = True
    stack = [(0, F - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        s = (np.arange(a + 1, b) - a) / (b - a)
        line = values[a] + (values[b] - values[a]) * s[:, None]
        err = np.abs(values[a + 1:b] - line).max(axis=1)
        k = int(np.argmax(err))
        if err[k] > tolerance:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]
