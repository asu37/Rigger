# SPDX-License-Identifier: GPL-3.0-or-later
"""Control shapes and colours.

Shapes are wireframe meshes in bone space: Y runs along the bone, X/Z across.
They are drawn scaled by the bone length, so modules give a *world radius* and
the scale is computed from it. Most shapes have a radius of 1 in shape units.
A shape can be shifted along the bone with ``y`` (0 = head, 1 = tail).
"""

import math

import bpy

WIDGET_COLLECTION = "Rigger Widgets"

# (normal, select, active) colours
COLORS = {
    "L": ((0.10, 0.42, 1.00), (0.55, 0.78, 1.00), (0.85, 0.95, 1.00)),
    "R": ((1.00, 0.16, 0.16), (1.00, 0.60, 0.55), (1.00, 0.88, 0.85)),
    "C": ((1.00, 0.72, 0.02), (1.00, 0.90, 0.45), (1.00, 0.97, 0.80)),
    "ROOT": ((0.15, 0.85, 0.30), (0.55, 1.00, 0.60), (0.85, 1.00, 0.88)),
    "FACE": ((0.95, 0.45, 0.85), (1.00, 0.75, 0.95), (1.00, 0.92, 1.00)),
    "TWEAK": ((0.10, 0.85, 0.75), (0.55, 1.00, 0.92), (0.85, 1.00, 0.97)),
}


def set_color(pose_bone, key):
    normal, select, active = COLORS.get(key, COLORS["C"])
    color = pose_bone.color
    color.palette = 'CUSTOM'
    color.custom.normal = normal
    color.custom.select = select
    color.custom.active = active
    color.custom.show_colored_constraints = False


# ---------------------------------------------------------------------------
# Shape geometry
# ---------------------------------------------------------------------------

def _circle(radius=1.0, segments=32, y=0.0, plane='XZ', arc=(0.0, 2.0 * math.pi), closed=True):
    verts = []
    count = segments if closed else segments + 1
    for i in range(count):
        a = arc[0] + (arc[1] - arc[0]) * i / segments
        c, s = math.cos(a) * radius, math.sin(a) * radius
        if plane == 'XZ':
            verts.append((c, y, s))
        elif plane == 'XY':
            verts.append((c, s + y, 0.0))
        else:  # YZ
            verts.append((0.0, c + y, s))
    edges = [(i, i + 1) for i in range(count - 1)]
    if closed:
        edges.append((count - 1, 0))
    return verts, edges


def _poly(points, closed=True):
    count = len(points)
    edges = [(i, i + 1) for i in range(count - 1)]
    if closed:
        edges.append((count - 1, 0))
    return [tuple(p) for p in points], edges


def _merge(*parts):
    verts, edges = [], []
    for pv, pe in parts:
        offset = len(verts)
        verts.extend(pv)
        edges.extend((a + offset, b + offset) for a, b in pe)
    return verts, edges


def _squircle(rx, rz, y=0.0, power=4.0, segments=40):
    pts = []
    for i in range(segments):
        a = 2.0 * math.pi * i / segments
        c, s = math.cos(a), math.sin(a)
        x = math.copysign(abs(c) ** (2.0 / power), c) * rx
        z = math.copysign(abs(s) ** (2.0 / power), s) * rz
        pts.append((x, y, z))
    return _poly(pts)


def _rounded_rect(x0, x1, y0, y1, z, radius, segments=6):
    pts = []
    corners = ((x1 - radius, y1 - radius, 0.0), (x0 + radius, y1 - radius, 0.5 * math.pi),
               (x0 + radius, y0 + radius, math.pi), (x1 - radius, y0 + radius, 1.5 * math.pi))
    for cx, cy, start in corners:
        for i in range(segments + 1):
            a = start + 0.5 * math.pi * i / segments
            pts.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius, z))
    return _poly(pts)


def _shape_data(kind):
    if kind == 'CIRCLE':        # ring around the bone axis
        return _circle(1.0)
    if kind == 'TORSO':
        return _squircle(1.25, 0.95)
    if kind == 'CHEST':         # chest plate with a notch at the front
        verts, edges = _squircle(1.15, 0.9)
        notch = _poly([(-0.25, 0, -0.9), (0, 0, -1.15), (0.25, 0, -0.9)], closed=False)
        return _merge((verts, edges), notch)
    if kind == 'HIPS':          # bowl: ring with a dip at the front
        return _merge(_circle(1.0, 40), _circle(0.35, 16, plane='YZ', arc=(math.pi, 2 * math.pi),
                                                    closed=False))
    if kind == 'SHOULDER':      # arc over the shoulder
        return _circle(1.0, 16, plane='XZ', arc=(0.15 * math.pi, 0.85 * math.pi), closed=False)
    if kind == 'BOX':
        s = 1.0
        v = [(x, y, z) for x in (-s, s) for y in (0.0, 1.0) for z in (-s * 0.6, s * 0.6)]
        e = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
        return v, e
    if kind == 'POLE':          # small diamond
        r = 1.0
        v = [(0, 0, r), (r, 0, 0), (0, 0, -r), (-r, 0, 0), (0, r, 0), (0, -r, 0)]
        e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 0), (4, 1), (4, 2), (4, 3),
             (5, 0), (5, 1), (5, 2), (5, 3)]
        return v, e
    if kind == 'ROOT':
        parts = [_circle(1.0, 48)]
        for i in range(4):
            a = i * math.pi / 2
            c, s = math.cos(a), math.sin(a)
            tip = (c * 1.3, 0, s * 1.3)
            left = (c * 1.08 - s * 0.12, 0, s * 1.08 + c * 0.12)
            right = (c * 1.08 + s * 0.12, 0, s * 1.08 - c * 0.12)
            parts.append(_poly([left, tip, right]))
        return _merge(*parts)
    if kind == 'ROLL':          # bold curved arrow behind the heel
        arc = [(0.0, -math.cos(a), math.sin(a)) for a in
               (math.radians(d) for d in range(-20, 100, 8))]
        inner = [(0.0, -math.cos(a) * 0.82, math.sin(a) * 0.82) for a in
                 (math.radians(d) for d in range(-20, 100, 8))]
        n = len(arc)
        head = (0.0, -math.cos(math.radians(112)) * 0.91, math.sin(math.radians(112)) * 0.91)
        verts = arc + inner + [head]
        edges = [(i, i + 1) for i in range(n - 1)] + [(n + i, n + i + 1) for i in range(n - 1)]
        edges += [(0, n), (n - 1, 2 * n), (2 * n - 1, 2 * n)]
        return verts, edges
    if kind == 'MASTER':        # lollipop above a finger / chain
        stick = _poly([(0, 0.0, 0.25), (0, 0.75, 0.25)], closed=False)
        return _merge(stick, _circle(0.12, 16, y=0.87, plane='YZ'),
                      _poly([(0, 0.87, 0.25)], closed=False))
    if kind == 'EYES':          # capsule around both eye targets
        r = 0.45
        top = [(-1.0 + r + math.cos(a) * r, 0, math.sin(a) * r) for a in
               (math.pi / 2 + math.pi * i / 12 for i in range(13))]
        bottom = [(1.0 - r + math.cos(a) * r, 0, math.sin(a) * r) for a in
                  (-math.pi / 2 + math.pi * i / 12 for i in range(13))]
        return _poly(top + bottom)
    if kind == 'ARC':           # gentle arc (brows, upper lids)
        pts = [(x, 0.0, 0.35 * (1.0 - x * x)) for x in (i / 6.0 - 1.0 for i in range(13))]
        return _poly(pts, closed=False)
    if kind == 'ARC_DOWN':      # lower lids
        pts = [(x, 0.0, -0.35 * (1.0 - x * x)) for x in (i / 6.0 - 1.0 for i in range(13))]
        return _poly(pts, closed=False)
    if kind == 'JAW':           # U around the chin
        return _circle(1.0, 20, y=0.0, plane='XY', arc=(math.pi, 2.0 * math.pi), closed=False)
    if kind == 'DIAMOND':
        v = [(0, 0, 1), (0.6, 0, 0), (0, 0, -1), (-0.6, 0, 0)]
        return _poly(v)
    if kind == 'LINE':          # head to tail (pole lines)
        return [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1)]
    if kind == 'SPHERE':        # three rings (pivots)
        return _merge(_circle(1.0, 24), _circle(1.0, 24, plane='XY'), _circle(1.0, 24, plane='YZ'))
    if kind == 'CROSS':         # 3D plus (cursors, offsets)
        return ([(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)],
                [(0, 1), (2, 3), (4, 5)])
    if kind == 'CUBE':          # cube centred on the bone head
        v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        e = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
        return v, e
    if kind == 'SQUARE':        # flat square around the bone axis
        return _poly([(-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1)])
    if kind == 'ARROWS':        # four arrows on the ground plane (roll cursor)
        parts = [_circle(0.35, 16, plane='XY')]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tip = (dx, dy, 0.0)
            px, py = -dy * 0.25, dx * 0.25
            base = (dx * 0.6, dy * 0.6, 0.0)
            parts.append(_poly([(base[0] + px, base[1] + py, 0), tip, (base[0] - px, base[1] - py, 0)]))
            parts.append(_poly([(dx * 0.35, dy * 0.35, 0), base], closed=False))
        return _merge(*parts)
    raise ValueError(kind)


SHAPE_KINDS = ('CIRCLE', 'TORSO', 'CHEST', 'HIPS', 'SHOULDER', 'BOX', 'POLE', 'ROOT', 'ROLL', 'MASTER',
               'EYES', 'ARC', 'ARC_DOWN', 'JAW', 'DIAMOND', 'LINE', 'SPHERE', 'CROSS', 'CUBE', 'SQUARE',
               'ARROWS')


def _widget_collection():
    coll = bpy.data.collections.get(WIDGET_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(WIDGET_COLLECTION)
        bpy.context.scene.collection.children.link(coll)
    coll.hide_render = True
    coll.hide_viewport = True
    return coll


def _make_object(name, verts, edges):
    obj = bpy.data.objects.get(name)
    if obj is not None and obj.type == 'MESH':
        mesh = obj.data
        mesh.clear_geometry()
    else:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        _widget_collection().objects.link(obj)
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    return obj


# Shapes with depth along the bone are scaled uniformly; flat ones only across the bone.
UNIFORM = {'POLE', 'ROLL', 'JAW', 'MASTER', 'DIAMOND', 'ROOT', 'SPHERE', 'CROSS', 'CUBE', 'ARROWS'}


def get_widget(kind, y=0.0):
    name = f"WGT-rigger_{kind.lower()}" + (f"_y{y:.4f}" if y else "")
    obj = bpy.data.objects.get(name)
    if obj is None:
        verts, edges = _shape_data(kind)
        verts = [(x, vy + y, z) for x, vy, z in verts]
        obj = _make_object(name, verts, edges)
    return obj


def foot_widget(rig_name, bone_name, verts):
    """Per-bone footprint widget. verts: corners (x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)."""
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    z = verts[0][2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    radius = 0.45 * min(x1 - x0, y1 - y0)
    return _make_object(f"WGT-{rig_name}_{bone_name}", *_rounded_rect(x0, x1, y0, y1, z, radius))


def apply_widget(pose_bone, kind, scale=1.0, y=0.0, wire_width=2.0):
    """scale: size across the bone (in bone lengths). y: offset along the bone (0..1)."""
    if kind in UNIFORM:
        pose_bone.custom_shape = get_widget(kind, y / scale if scale else 0.0)
        pose_bone.custom_shape_scale_xyz = (scale, scale, scale)
    else:
        pose_bone.custom_shape = get_widget(kind, y)
        pose_bone.custom_shape_scale_xyz = (scale, 1.0, scale)
    if hasattr(pose_bone, "custom_shape_wire_width"):
        pose_bone.custom_shape_wire_width = wire_width
