# SPDX-License-Identifier: GPL-3.0-or-later
"""Place reference joints from the character mesh.

The six Smart Markers give the ends of the limbs; this module looks at the
mesh between them. It cuts the mesh with planes (the outline where a plane
crosses the mesh, split into separate loops), traces the centre line of each
limb and places:

* the elbow at the bend of the arm, or its thinnest section when straight
* the knee at the narrowing between thigh and calf
* the hip joint on the thigh's centre line, at pelvis height
* the spine joints at a fixed share of the torso's depth (closer to the back)
* the ball and tip of the foot from the foot's real length

Every function returns None when the mesh does not give a clear answer, and the
caller then keeps the template proportions.
"""

import numpy as np

ELBOW_PRIOR = 0.54      # elbow position along shoulder -> wrist on a human skeleton
MIN_BEND_DEG = 8.0      # below this the arm counts as straight
SPINE_DEPTH = 0.6       # spine joints: share of the torso depth from the front


class Slicer:
    """Cross-sections of one or more mesh objects, in world space."""

    def __init__(self, objects, depsgraph=None):
        pts, edges, loop_edge, loop_face = [], [], [], []
        v_off = e_off = f_off = 0
        for obj in objects:
            ev = obj.evaluated_get(depsgraph) if depsgraph else obj
            me = ev.to_mesh() if depsgraph else obj.data
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            mw = np.array(obj.matrix_world)
            pts.append(co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3])
            e = np.empty(len(me.edges) * 2, dtype=np.int64)
            me.edges.foreach_get("vertices", e)
            edges.append(e.reshape(-1, 2) + v_off)
            le = np.empty(len(me.loops), dtype=np.int64)
            me.loops.foreach_get("edge_index", le)
            loop_edge.append(le + e_off)
            totals = np.empty(len(me.polygons), dtype=np.int64)
            me.polygons.foreach_get("loop_total", totals)
            loop_face.append(np.repeat(np.arange(len(totals)), totals) + f_off)
            v_off += len(me.vertices)
            e_off += len(me.edges)
            f_off += len(me.polygons)
            if depsgraph:
                ev.to_mesh_clear()
        self.p = np.concatenate(pts) if pts else np.empty((0, 3))
        self.edges = np.concatenate(edges) if edges else np.empty((0, 2), dtype=np.int64)
        self.loop_edge = np.concatenate(loop_edge) if loop_edge else np.empty(0, dtype=np.int64)
        self.loop_face = np.concatenate(loop_face) if loop_face else np.empty(0, dtype=np.int64)

    def section(self, origin, normal):
        """Loops where the plane cuts the mesh, each an (n, 2, 3) array of segments."""
        normal = np.asarray(normal, float)
        normal = normal / np.linalg.norm(normal)
        d = (self.p - np.asarray(origin, float)) @ normal
        # vertices exactly on the plane would make edges touch it without crossing
        d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        a, b = self.edges[:, 0], self.edges[:, 1]
        crossing = (d[a] * d[b]) < 0
        f = d[a] / np.where(crossing, d[a] - d[b], 1.0)
        hit = self.p[a] + (self.p[b] - self.p[a]) * f[:, None]
        on = crossing[self.loop_edge]
        faces, e = self.loop_face[on], self.loop_edge[on]
        order = np.argsort(faces, kind="stable")
        faces, e = faces[order], e[order]
        parent = {int(i): int(i) for i in np.unique(e)}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        segs = []
        for i in range(len(faces) - 1):
            if faces[i] == faces[i + 1]:
                ra, rb = find(int(e[i])), find(int(e[i + 1]))
                parent[ra] = rb
                segs.append((int(e[i]), int(e[i + 1])))
        loops = {}
        for ea, eb in segs:
            loops.setdefault(find(ea), []).append((hit[ea], hit[eb]))
        return [np.array(v) for v in loops.values() if len(v) >= 3]


def centroid(loop):
    mid = loop.mean(axis=1)
    length = np.linalg.norm(loop[:, 1] - loop[:, 0], axis=1)
    return (mid * length[:, None]).sum(axis=0) / max(length.sum(), 1e-12)


def radius(loop, center):
    return float(np.linalg.norm(loop.mean(axis=1) - center, axis=1).mean())


def trace(slicer, start, end, height, steps=40):
    """Centre line of a limb from start to end.

    Steps along the current heading, cuts the mesh across it and moves to the
    centre of the loop closest to the guess. Returns (points, radii, arc) where
    radii[i] is None where no loop of the limb was found (e.g. where an arm
    merges with the torso) and arc is the fraction of the path length.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    total = np.linalg.norm(end - start)
    if total < 1e-6:
        return None
    step = total / steps
    pts, radii = [start], [None]
    cur, heading = start.copy(), (end - start) / total
    for _ in range(1, steps):
        guess = cur + heading * step
        if np.linalg.norm(end - guess) < step:
            break
        loops = [(centroid(lp), lp) for lp in slicer.section(guess, heading)]
        loops = [(c, lp) for c, lp in loops if np.linalg.norm(c - guess) < 0.05 * height]
        if not loops:
            cur = guess
            pts.append(cur)
            radii.append(None)
            continue
        c, lp = min(loops, key=lambda x: np.linalg.norm(x[0] - guess))
        new = c - cur
        if np.linalg.norm(new) > 1e-9:
            heading = 0.5 * heading + 0.5 * new / np.linalg.norm(new)
            heading /= np.linalg.norm(heading)
        cur = c
        pts.append(c)
        radii.append(radius(lp, c))
    pts.append(end)
    radii.append(None)
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-12)
    return pts, radii, arc


def sharpest_bend(pts, arc, lo, hi, span=4):
    """(index, degrees) of the sharpest turn of the centre line between arc lo..hi."""
    best = (None, 0.0)
    for i in range(1, len(pts) - 1):
        if not lo <= arc[i] <= hi:
            continue
        a = pts[i] - pts[max(0, i - span)]
        b = pts[min(len(pts) - 1, i + span)] - pts[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        ang = float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1.0, 1.0))))
        if ang > best[1]:
            best = (i, ang)
    return best


def thinnest(radii, arc, lo, hi):
    cand = [(r, i) for i, r in enumerate(radii) if r is not None and lo <= arc[i] <= hi]
    return min(cand)[1] if len(cand) >= 3 else None


def at_arc(pts, arc, t):
    """Point at arc fraction t along the polyline."""
    i = int(np.searchsorted(arc, t))
    i = min(max(i, 1), len(arc) - 1)
    f = (t - arc[i - 1]) / max(arc[i] - arc[i - 1], 1e-12)
    return pts[i - 1] + (pts[i] - pts[i - 1]) * f


def elbow(slicer, shoulder, wrist, height):
    path = trace(slicer, shoulder, wrist, height)
    if path is None:
        return None
    pts, radii, arc = path
    if sum(r is not None for r in radii) < len(radii) // 3:
        return None
    i, angle = sharpest_bend(pts, arc, 0.35, 0.7)
    if i is not None and angle >= MIN_BEND_DEG:
        return pts[i]
    i = thinnest(radii, arc, 0.38, 0.62)
    if i is not None:
        return pts[i]
    return at_arc(pts, arc, ELBOW_PRIOR)


def leg(slicer, ankle, hip_guess, height):
    """(knee, hip joint) from the leg traced upward from the ankle (hands hang
    beside the thighs, never beside the ankles)."""
    path = trace(slicer, ankle, hip_guess, height)
    if path is None:
        return None, None
    pts, radii, arc = path
    pts, radii, arc = pts[::-1], radii[::-1], 1.0 - arc[::-1]
    if sum(r is not None for r in radii) < len(radii) // 3:
        return None, None
    i = thinnest(radii, arc, 0.38, 0.62)
    knee = pts[i] if i is not None else None
    # Thigh centre line: fit through its loops and extend it to pelvis height
    thigh = [p for p, r, a in zip(pts, radii, arc) if r is not None and 0.12 <= a <= 0.4]
    hip = None
    if len(thigh) >= 4:
        thigh = np.array(thigh)
        c = thigh.mean(axis=0)
        direction = np.linalg.svd(thigh - c)[2][0]
        if abs(direction[2]) > 0.3:
            hip = c + direction * ((hip_guess[2] - c[2]) / direction[2])
            # only correct sideways and in depth; never move the hip far from the guess
            if np.linalg.norm(hip - hip_guess) > 0.08 * height:
                hip = None
    return knee, hip


def spine_depth(slicer, z, height):
    """Depth (Y) of the spine at height z: a share of the torso from the front."""
    for loop in slicer.section((1e-4 * height, 0.0, z), (0, 0, 1)):
        xs = loop[:, :, 0]
        if xs.min() < 0 < xs.max():
            ys = _crossings(loop, 0, 1e-4 * height, 1)
            if len(ys) >= 2:
                front, back = ys.min(), ys.max()
                return float(front + (back - front) * SPINE_DEPTH)
    return None


def foot_tip(slicer, ankle, floor, height):
    """Front-most point of the foot under the ankle, or None."""
    p = slicer.p
    near = (np.abs(p[:, 0] - ankle[0]) < 0.06 * height) & (p[:, 2] < ankle[2]) & (p[:, 2] >= floor - 1e-6)
    ahead = near & (p[:, 1] < ankle[1])
    if ahead.sum() < 3:
        return None
    tip = p[ahead][np.argmin(p[ahead, 1])]
    length = ankle[1] - tip[1]
    if not 0.05 * height < length < 0.2 * height:
        return None
    return tip


def _crossings(segs, axis, value, other):
    a, b = segs[:, 0], segs[:, 1]
    da, db = a[:, axis] - value, b[:, axis] - value
    m = (da * db) <= 0
    t = da[m] / np.where(da[m] - db[m] == 0, 1.0, da[m] - db[m])
    return a[m, other] + (b[m, other] - a[m, other]) * t
