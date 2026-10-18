"""Deformation quality measures. Pure numpy: rest and posed vertex arrays in,
numbers out, so they can be checked without Blender.

Conventions: points are (N, 3) world coordinates; a segment is a (head, tail)
pair; "region" is a boolean mask over vertices.
"""
import re

import numpy as np


def point_segment_distance(points, head, tail):
    seg = tail - head
    t = np.clip(((points - head) @ seg) / max(seg @ seg, 1e-12), 0.0, 1.0)
    return np.linalg.norm(points - (head + t[:, None] * seg), axis=1)


def nearest_segment(points, segments):
    """Index of the closest segment for every point."""
    d = np.stack([point_segment_distance(points, h, t) for h, t in segments], axis=1)
    return np.argmin(d, axis=1)


def summary(values):
    if len(values) == 0:
        return {"mean": None, "p05": None, "p95": None}
    return {"mean": float(np.mean(values)), "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95))}


def thickness_ratio(rest, posed, region, rest_segs, posed_segs):
    """How far each vertex stays from the limb's bones, posed / rest.

    Below 1 the joint collapses (lost volume), above 1 it bulges. Each vertex
    is measured against the segment it is closest to at rest, posed and
    unposed, so a perfectly rigid bend gives 1 everywhere, even at angles where
    the other segment swings close to it.
    """
    if not region.any():
        return np.empty(0)
    r_all = np.stack([point_segment_distance(rest[region], h, t) for h, t in rest_segs], axis=1)
    p_all = np.stack([point_segment_distance(posed[region], h, t) for h, t in posed_segs], axis=1)
    own = np.argmin(r_all, axis=1)
    rows = np.arange(len(own))
    r, p = r_all[rows, own], p_all[rows, own]
    keep = r > 1e-6
    return p[keep] / r[keep]


def triangles(faces):
    """Fan-triangulate polygons given as lists of vertex indices."""
    tris = []
    for f in faces:
        for i in range(1, len(f) - 1):
            tris.append((f[0], f[i], f[i + 1]))
    return np.array(tris, dtype=np.int64)


def triangle_areas(points, tris):
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def area_distortion(rest, posed, tris, region):
    """Surface stretch of the triangles inside the region.

    Returns the share of triangles crushed below half their area, stretched
    above 1.5x, and the RMS of log2(area ratio) (0 = no distortion).
    """
    inside = region[tris].all(axis=1)
    if not inside.any():
        return {"crushed": None, "stretched": None, "rms_log": None}
    ar = triangle_areas(rest, tris[inside])
    ap = triangle_areas(posed, tris[inside])
    ok = ar > 1e-12
    ratio = ap[ok] / ar[ok]
    log = np.log2(np.maximum(ratio, 1e-6))
    return {"crushed": float((ratio < 0.5).mean()), "stretched": float((ratio > 1.5).mean()),
            "rms_log": float(np.sqrt((log ** 2).mean()))}


def signed_volume(points, tris):
    a, b, c = points[tris[:, 0]], points[tris[:, 1]], points[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def twist_profile(rest, posed, region, head, tail, angle, slices=10):
    """How a twist of 'angle' (radians) about the head->tail axis spreads along it.

    The ideal skin twists linearly from 0 at the head to 'angle' at the tail.
    Returns the per-slice measured angle, the RMS error against the ideal as a
    fraction of 'angle', and the thinnest slice radius ratio (candy-wrapper
    pinch: 1 = none, lower = the limb gets thinner when twisted).
    """
    axis = tail - head
    length = np.linalg.norm(axis)
    axis = axis / length
    pts_r, pts_p = rest[region], posed[region]
    s = ((pts_r - head) @ axis) / length

    def radial(pts):
        rel = pts - head
        return rel - np.outer(rel @ axis, axis)

    vr, vp = radial(pts_r), radial(pts_p)
    rr, rp = np.linalg.norm(vr, axis=1), np.linalg.norm(vp, axis=1)
    ok = (rr > 1e-6) & (rp > 1e-6)
    cross = np.cross(vr, vp) @ axis
    dot = np.einsum("ij,ij->i", vr, vp)
    theta = np.arctan2(cross, dot)
    measured, radius, centers = [], [], []
    edges = np.linspace(0.0, 1.0, slices + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (s >= lo) & (s < hi)
        if m.sum() < 3:
            measured.append(None)
            radius.append(None)
            centers.append(None)
            continue
        # circular mean of the angles in the slice
        measured.append(float(np.arctan2(np.sin(theta[m]).mean(), np.cos(theta[m]).mean())))
        radius.append(float(rp[m].mean() / rr[m].mean()))
        centers.append(float(s[m].mean()))
    pairs = [(c, a) for c, a in zip(centers, measured) if a is not None]
    if not pairs or abs(angle) < 1e-9:
        return {"angles": measured, "rms_error": None, "min_radius": None, "end_fraction": None}
    err = np.array([a - angle * c for c, a in pairs])
    radii = [r for r in radius if r is not None]
    # end_fraction: twist reached in the last slice, against the ideal there
    end_c, end_a = pairs[-1]
    return {"angles": measured, "rms_error": float(np.sqrt((err ** 2).mean()) / abs(angle)),
            "min_radius": float(min(radii)), "end_fraction": float(end_a / (angle * end_c))}


HELPER = re.compile(r"_(twist\.\d+|fan\d*)")
FINGER = re.compile(r"(thumb|index|middle|ring|pinky)")


def bone_group(name):
    """Twist and fan bones count as part of their limb bone: DEF-forearm_twist.01.L -> DEF-forearm.L."""
    return HELPER.sub("", name)


def bone_graph_distance(names, parents):
    """Hop distance between bones in the deform hierarchy (twist/fan bones merged
    into their limb bone). parents: bone name -> parent bone name or None."""
    groups = [bone_group(n) for n in names]
    keys = sorted(set(groups))
    index = {k: i for i, k in enumerate(keys)}
    adj = [set() for _ in keys]
    for n, g in zip(names, groups):
        p = parents.get(n)
        while p is not None and bone_group(p) == g:
            p = parents.get(p)
        if p is not None and bone_group(p) in index:
            a, b = index[g], index[bone_group(p)]
            adj[a].add(b)
            adj[b].add(a)
    n = len(keys)
    dist = np.full((n, n), 99, dtype=np.int64)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s, v] == 99:
                        dist[s, v] = dist[s, u] + 1
                        nxt.append(v)
            frontier = nxt
    col = np.array([index[g] for g in groups])
    return dist[np.ix_(col, col)]


def weight_report(weights, names, parents, bleed_weight=0.05, bleed_hops=3):
    """Sanity of a (vertices x bones) weight matrix.

    bleed: share of vertices carrying at least 'bleed_weight' on a bone that is
    'bleed_hops' or more bones away from the vertex's main bone in the deform
    hierarchy (a hand bone in the hip, an arm bone in the head...).
    cross_side: share of vertices weighted to both a .L and a .R bone.
    """
    total = weights.sum(axis=1)
    count = (weights > 1e-4).sum(axis=1)
    main = np.argmax(weights, axis=1)
    hops = bone_graph_distance(names, parents)
    far = hops[main] >= bleed_hops                      # (V, B)
    finger = np.array([bool(FINGER.search(n)) for n in names])
    # finger to finger leaks (fingers stuck together) are reported on their own
    both_fingers = finger[main][:, None] & finger[None, :]
    leak = (weights >= bleed_weight) & far
    bleeding = (leak & ~both_fingers).any(axis=1) & (total > 0)
    finger_bleeding = (leak & both_fingers).any(axis=1) & (total > 0)
    pairs, finger_pairs = {}, {}
    for v in np.nonzero(bleeding)[0][:20000]:
        for b in np.nonzero(leak[v] & ~both_fingers[v])[0]:
            key = f"{names[main[v]]} <- {names[b]}"
            pairs[key] = pairs.get(key, 0) + 1
    for v in np.nonzero(finger_bleeding)[0][:20000]:
        for b in np.nonzero(leak[v] & both_fingers[v])[0]:
            key = f"{names[main[v]]} <- {names[b]}"
            finger_pairs[key] = finger_pairs.get(key, 0) + 1
    left = np.array([n.endswith(".L") for n in names])
    right = np.array([n.endswith(".R") for n in names])
    cross = ((weights[:, left] > bleed_weight).any(axis=1) & (weights[:, right] > bleed_weight).any(axis=1))
    return {
        "unweighted": int((total <= 1e-6).sum()),
        "not_normalized": int((np.abs(total - 1.0) > 0.01).sum()),
        "max_influences": int(count.max()) if len(count) else 0,
        "over_4_influences": float((count > 4).mean()),
        "bleed": float(bleeding.mean()),
        "finger_bleed": float(finger_bleeding.mean()),
        "bleed_pairs": dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:8]),
        "finger_bleed_pairs": dict(sorted(finger_pairs.items(), key=lambda kv: -kv[1])[:8]),
        "cross_side": float(cross.mean()),
    }
