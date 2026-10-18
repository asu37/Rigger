"""Derive the six Smart Marker positions of each benchmark body and freeze them.

    blender -b --factory-startup --python bench/derive_landmarks.py [-- --render]

Writes bench/landmarks.json. The benchmark reads the frozen file, so it
measures skinning and deformation, not marker placement. Run this again only
when bodies are added; check the renders in bench/results/landmarks/.

The bodies face -Y, Z up, feet at Z = 0, in an A-pose. The character's left
side is +X. Every point sits in the middle of the mesh volume, like a marker
clicked in Rigger.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy  # noqa: E402

import bodies  # noqa: E402


class Slicer:
    """Cross-sections of a mesh: the outline where a plane cuts it, split into
    its separate loops (torso, arm, leg...)."""

    def __init__(self, obj):
        me = obj.data
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        mw = np.array(obj.matrix_world)
        self.p = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
        edges = np.empty(len(me.edges) * 2, dtype=np.int64)
        me.edges.foreach_get("vertices", edges)
        self.edges = edges.reshape(-1, 2)
        loop_edge = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("edge_index", loop_edge)
        starts = np.empty(len(me.polygons), dtype=np.int64)
        me.polygons.foreach_get("loop_start", starts)
        totals = np.empty(len(me.polygons), dtype=np.int64)
        me.polygons.foreach_get("loop_total", totals)
        self.loop_edge = loop_edge
        self.loop_face = np.repeat(np.arange(len(starts)), totals)

    def section(self, origin, normal):
        """List of loops, each an (n, 2, 3) array of outline segments."""
        normal = np.asarray(normal, float) / np.linalg.norm(normal)
        d = (self.p - np.asarray(origin, float)) @ normal
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

    def horizontal(self, z):
        return self.section((0, 0, z), (0, 0, 1))


def hits(segs, axis, value, other):
    """Coordinate 'other' where the outline segments cross 'axis' = value."""
    a, b = segs[:, 0], segs[:, 1]
    da, db = a[:, axis] - value, b[:, axis] - value
    m = (da * db) <= 0
    t = da[m] / np.where(da[m] - db[m] == 0, 1.0, da[m] - db[m])
    return a[m, other] + (b[m, other] - a[m, other]) * t


def centroid(segs):
    mid = segs.mean(axis=1)
    length = np.linalg.norm(segs[:, 1] - segs[:, 0], axis=1)
    return (mid * length[:, None]).sum(axis=0) / length.sum()


def derive(sl):
    p = sl.p
    floor, top = p[:, 2].min(), p[:, 2].max()
    h = top - floor

    def mid_y(x, z):
        """Middle depth of the mesh at (x, z): halfway between front and back."""
        x = x if x else 1e-4 * h
        ys = np.concatenate([hits(loop, 0, x, 1) for loop in sl.horizontal(z)] or [np.empty(0)])
        return float((ys.min() + ys.max()) / 2)

    def loops_at(z):
        """Horizontal loops at height z as (centroid, min x, max x)."""
        out = []
        for loop in sl.horizontal(z):
            xs = loop[:, :, 0]
            out.append((centroid(loop), xs.min(), xs.max(), loop))
        return out

    # Pelvis: the crotch is the lowest height where the two legs merge into one
    # loop crossing the midline; the pelvis marker sits a little above it.
    crotch = next(z for z in np.linspace(floor + 0.3 * h, floor + 0.7 * h, 400)
                  if any(lo < 0 < hi for _c, lo, hi, _l in loops_at(z)))
    pelvis_z = crotch + 0.045 * h
    pelvis = [0.0, mid_y(0.0, pelvis_z), float(pelvis_z)]

    # Front profile of the midline: the most forward point at each height.
    midline = sl.section((1e-4 * h, 0, 0), (1, 0, 0))  # just off the exact midline vertices
    zs = np.linspace(floor + 0.74 * h, floor + 0.95 * h, 211)

    def front_y(z):
        ys = np.concatenate([hits(loop, 2, z, 1) for loop in midline] or [np.empty(0)])
        return ys.min() if len(ys) else np.nan

    profile = np.array([front_y(z) for z in zs])
    # Neck base: the jugular notch, the most set-back point of the throat
    # between the chest and the chin.
    upper = zs > floor + 0.78 * h
    lower = zs < floor + 0.88 * h
    i_notch = int(np.nanargmax(np.where(upper & lower, profile, -np.inf)))
    notch_y = profile[i_notch]
    # Chin: going up from the notch, the first height where the face
    # overhangs the throat.
    chin_z = next(z for z, y in zip(zs[i_notch:], profile[i_notch:]) if y < notch_y - 0.025 * h)
    chin = [0.0, mid_y(0.0, chin_z), float(chin_z)]
    # Short (stylized) necks: keep the neck marker below the chin.
    neck_z = min(zs[i_notch] + 0.01 * h, chin_z - 0.02 * h)
    neck = [0.0, mid_y(0.0, neck_z), float(neck_z)]

    # Shoulder: the armpit is the highest slice where the left arm is a loop of
    # its own. The joint lies on the upper arm's centre line above it.
    def arm_loop(z):
        arms = [(c, lo) for c, lo, hi, _l in loops_at(z) if lo > 0.02 * h and c[0] < 0.4 * h]
        return min(arms, key=lambda a: a[1])[0] if arms else None

    armpit = next(z for z in np.linspace(neck_z, pelvis_z, 400) if arm_loop(z) is not None)
    a_hi = arm_loop(armpit - 0.01 * h)
    a_lo = arm_loop(armpit - 0.08 * h)
    up = (a_hi - a_lo) / np.linalg.norm(a_hi - a_lo)
    shoulder_pt = a_hi + up * ((0.05 * h) / up[2])
    shoulder = [float(v) for v in shoulder_pt]

    # Wrist: follow the arm's centre line out to the fingertips (rigger.fitting's
    # limb trace, which follows bent arms), cutting across the arm at each step.
    # The wrist is the shortest outline before the cut splits into fingers; outline
    # length, not mean radius, because a flat palm can have a smaller mean radius.
    from rigger import fitting
    left_pts = p[(p[:, 0] > shoulder_pt[0]) & (p[:, 2] > pelvis_z - 0.1 * h)]
    tip = left_pts[np.argmax(np.linalg.norm(left_pts - shoulder_pt, axis=1))]
    pts, radii, arc = fitting.trace(sl, shoulder_pt, tip, h, steps=60)
    best = None
    for k in range(1, len(pts) - 1):
        if not 0.55 <= arc[k] <= 0.95 or radii[k] is None:
            continue
        o = pts[k]
        direction = pts[k + 1] - pts[k - 1]
        loops = [(centroid(lp), lp) for lp in sl.section(o, direction)]
        loops = [(c, lp) for c, lp in loops if np.linalg.norm(c - o) < 0.06 * h]
        if not loops:
            continue
        c, lp = min(loops, key=lambda x: np.linalg.norm(x[0] - o))
        if sum(np.linalg.norm(c2 - c) < 0.035 * h for c2, _l in loops) >= 2:
            break  # separate fingers: past the wrist
        r = np.linalg.norm(lp[:, 1] - lp[:, 0], axis=1).sum()
        if best is None or r < best[0]:
            best = (r, c)
    wrist = [float(v) for v in best[1]]

    # Ankle: the lower shin's centre, at ankle joint height.
    shin = next(c for c, lo, hi, _l in loops_at(floor + 0.09 * h) if lo > 0)
    ankle = [float(shin[0]), float(shin[1]), float(floor + 0.045 * h)]

    return {"chin": chin, "neck": neck, "shoulder": shoulder, "wrist": wrist,
            "pelvis": pelvis, "ankle": ankle}


def main():
    do_render = "--render" in sys.argv
    out = {}
    for name in bodies.BODIES:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        body, eyes = bodies.load_body(name)
        marks = derive(Slicer(body))
        out[name] = {k: [round(v, 4) for v in pt] for k, pt in marks.items()}
        print(name, out[name], flush=True)
        if do_render:
            import render
            objs = [body]
            mat = bpy.data.materials.new("marker")
            mat.diffuse_color = (1, 0, 0, 1)
            for key, pt in marks.items():
                for x in ({pt[0], -pt[0]}):
                    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.012, location=(x, pt[1], pt[2]))
                    sphere = bpy.context.object
                    sphere.show_in_front = True
                    objs.append(sphere)
            body.display_type = 'WIRE'
            # make the body see-through so markers inside show
            mat_body = bpy.data.materials.new("glass")
            mat_body.use_nodes = True
            bsdf = mat_body.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Alpha"].default_value = 0.35
            body.data.materials.clear()
            body.data.materials.append(mat_body)
            for o in objs[1:]:
                o.data.materials.append(mat)
                mat.use_nodes = True
                mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (1, 0, 0, 1)
                mat.node_tree.nodes["Principled BSDF"].inputs["Emission Color"].default_value = (1, 0, 0, 1)
                mat.node_tree.nodes["Principled BSDF"].inputs["Emission Strength"].default_value = 3.0
            for view in ("front", "side"):
                render.render(objs, os.path.join(bodies.HERE, "results", "landmarks", f"{name}_{view}.png"), view)
    with open(bodies.LANDMARKS, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote", bodies.LANDMARKS)


main()
