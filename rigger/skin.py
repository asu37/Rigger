# SPDX-License-Identifier: GPL-3.0-or-later
"""Bind meshes to a generated rig.

Methods:
* HEAT      - Blender's bone heat weighting
* PROXIMITY - distance-to-bone weights computed by Rigger (works on any mesh,
              including non-manifold and multi-part meshes where heat fails)
* AUTO      - heat weighting, then proximity weights for every vertex heat
              could not weight

Extras applied after weighting:
* small loose parts (eyes, buttons, buckles...) are bound rigidly to the bones
  nearest their centre, so they never stretch
* left side vertices never get right side bones and vice versa
* heat weighting runs on a temporarily rescaled copy for very small or very
  large characters, where Blender's heat solver is unreliable
* optional limit of bone influences per vertex for game engines

Options (``bind_mesh(..., **options)``):
* engines: VOXEL (heat weighting on a watertight voxel remesh, transferred back),
  split_parts (heat per loose part), optimize (heat on a decimated copy of dense
  meshes), scale_fix (always work on a rescaled copy)
* selected_bones (only the selected bones), selected_verts (only the selected
  vertices), apply_shape_keys (weights for the current shape key mix); locked
  vertex groups are never changed
* smooth_twist, improve_hips, improve_heels, refine_head, clean_fingers:
  weight corrections
* eyes / teeth / tongue: objects bound to the eye, teeth and tongue bones
* preserve_volume, parent (parent the mesh to the rig)
"""

import re

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Matrix, Vector

from . import utils

HEAT_HEIGHT_RANGE = (0.5, 5.0)  # character heights where heat weighting is reliable


def deform_segments(rig):
    names, heads, tails = [], [], []
    mw = rig.matrix_world
    for bone in rig.data.bones:
        if bone.use_deform:
            names.append(bone.name)
            heads.append(mw @ bone.head_local)
            tails.append(mw @ bone.tail_local)
    return names, np.array(heads, dtype=np.float64), np.array(tails, dtype=np.float64)


def world_vertices(obj):
    count = len(obj.data.vertices)
    co = np.empty(count * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(count, 3)
    mw = np.array(obj.matrix_world, dtype=np.float64)
    return co @ mw[:3, :3].T + mw[:3, 3]


def rig_local_x(rig, points):
    """X coordinate of world points in the rig's space (character left is +X)."""
    inv = np.array(rig.matrix_world.inverted(), dtype=np.float64)
    return points @ inv[0, :3] + inv[0, 3]


def rig_height(rig):
    zs = [v.z for b in rig.data.bones if b.use_deform for v in (b.head_local, b.tail_local)]
    scale = rig.matrix_world.to_scale().z
    return (max(zs) - min(zs)) * scale if zs else 1.0


def segment_distances(points, heads, tails):
    """Distance from every point (N,3) to every segment (B,3) -> (N,B)."""
    seg = tails - heads                                   # (B,3)
    seg_len2 = np.maximum((seg * seg).sum(axis=1), 1e-12)  # (B,)
    rel = points[:, None, :] - heads[None, :, :]           # (N,B,3)
    t = np.clip((rel * seg[None]).sum(axis=2) / seg_len2, 0.0, 1.0)
    closest = heads[None] + t[..., None] * seg[None]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def proximity_weights(points, heads, tails, influences=3, falloff=4.0, chunk=4096,
                      point_side=None, bone_side=None):
    """Return (indices (N,k), weights (N,k)) of the closest bones per vertex.

    point_side / bone_side: optional arrays of -1 (right), 0 (centre), 1 (left).
    Bones of the opposite side are pushed away so they never win.
    """
    k = min(influences, len(heads))
    all_idx = np.empty((len(points), k), dtype=np.int64)
    all_w = np.empty((len(points), k), dtype=np.float64)
    lengths = np.linalg.norm(tails - heads, axis=1)
    for start in range(0, len(points), chunk):
        d = segment_distances(points[start:start + chunk], heads, tails)
        # Normalise by bone length so long bones do not steal small ones' vertices.
        d = d / np.maximum(lengths, 1e-6)[None, :] ** 0.25
        if point_side is not None and bone_side is not None:
            ps = point_side[start:start + chunk]
            opposite = (ps[:, None] * bone_side[None, :]) < 0
            d = np.where(opposite, d * 1e3 + 1e3, d)
        idx = np.argpartition(d, k - 1, axis=1)[:, :k] if k < d.shape[1] else \
            np.tile(np.arange(d.shape[1]), (d.shape[0], 1))
        dist = np.take_along_axis(d, idx, axis=1)
        w = 1.0 / np.maximum(dist, 1e-6) ** falloff
        w /= w.sum(axis=1, keepdims=True)
        w[w < 0.05] = 0.0
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        all_idx[start:start + chunk] = idx
        all_w[start:start + chunk] = w
    return all_idx, all_w


def _sides(rig, names, points):
    bone_side = np.array([{"L": 1, "R": -1}.get(utils.side_letter(n), 0) for n in names])
    tol = 0.01 * rig_height(rig)
    x = rig_local_x(rig, points)
    point_side = np.where(x > tol, 1, np.where(x < -tol, -1, 0))
    return point_side, bone_side


VOLUME_GROUP = "rigger_volume"
VOLUME_MODIFIER = "Armature Preserve Volume"
VOLUME_RADIUS = 0.12  # in body heights, around each shoulder joint


def volume_weights(obj, rig, shoulder_volume):
    """How much each vertex preserves volume: shoulder_volume at the shoulder
    joints, rising to 1 over VOLUME_RADIUS body heights."""
    points = world_vertices(obj)
    w = np.ones(len(points))
    h = rig_height(rig)
    for side in (".L", ".R"):
        bone = rig.data.bones.get(f"DEF-upper_arm{side}")
        if bone is None:
            continue
        d = np.linalg.norm(points - np.array(rig.matrix_world @ bone.head_local), axis=1)
        w = np.minimum(w, shoulder_volume + (1.0 - shoulder_volume) * np.clip(d / (VOLUME_RADIUS * h), 0.0, 1.0))
    return w


def ensure_armature_modifier(obj, rig, preserve_volume=True, shoulder_volume=1.0):
    """Armature modifier first in the stack.

    With Preserve Volume and shoulder_volume below 1, the first modifier deforms
    without preserving volume and a second one (Multi Modifier) preserves it,
    mixed per vertex by the VOLUME_GROUP weights: full volume preservation
    everywhere except around the shoulders, where it makes a raised arm bulge.
    """
    mods = [m for m in obj.modifiers if m.type == 'ARMATURE']
    mod = next((m for m in mods if m.name != VOLUME_MODIFIER), None)
    if mod is None:
        mod = obj.modifiers.new("Armature", 'ARMATURE')
        # Armature deform should come first in the stack.
        obj.modifiers.move(obj.modifiers.find(mod.name), 0)
    mod.object = rig
    mixed = preserve_volume and shoulder_volume < 1.0 and any(
        b in rig.data.bones for b in ("DEF-upper_arm.L", "DEF-upper_arm.R"))
    mod.use_deform_preserve_volume = preserve_volume and not mixed
    vol = obj.modifiers.get(VOLUME_MODIFIER)
    if not mixed:
        if vol is not None:
            obj.modifiers.remove(vol)
        if VOLUME_GROUP in obj.vertex_groups:
            obj.vertex_groups.remove(obj.vertex_groups[VOLUME_GROUP])
        return mod
    vg = obj.vertex_groups.get(VOLUME_GROUP) or obj.vertex_groups.new(name=VOLUME_GROUP)
    w = np.round(volume_weights(obj, rig, shoulder_volume), 3)
    for value in np.unique(w):
        vg.add(np.nonzero(w == value)[0].tolist(), float(value), 'REPLACE')
    if vol is None:
        vol = obj.modifiers.new(VOLUME_MODIFIER, 'ARMATURE')
    vol.object = rig
    vol.use_deform_preserve_volume = True
    vol.use_multi_modifier = True
    vol.vertex_group = VOLUME_GROUP
    obj.modifiers.move(obj.modifiers.find(vol.name), obj.modifiers.find(mod.name) + 1)
    return mod


def parent_keep_transform(obj, rig):
    mw = obj.matrix_world.copy()
    obj.parent = rig
    obj.parent_type = 'OBJECT'
    obj.matrix_world = mw


def unweighted_vertices(obj, deform_names):
    groups = {vg.index for vg in obj.vertex_groups if vg.name in deform_names}
    missing = []
    for v in obj.data.vertices:
        if not any(g.group in groups and g.weight > 1e-4 for g in v.groups):
            missing.append(v.index)
    return missing


def _clear_deform_groups(obj, names):
    for name in names:
        vg = obj.vertex_groups.get(name)
        if vg is not None:
            obj.vertex_groups.remove(vg)


def _write_weights(obj, names, vertex_indices, idx, w, replace=False):
    """Write per-vertex (bone index, weight) pairs into vertex groups."""
    groups = {name: obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name) for name in names}
    if replace:
        verts = vertex_indices.tolist()
        for vg in groups.values():
            vg.remove(verts)
    for bone_i, name in enumerate(names):
        mask = (idx == bone_i) & (w > 0)
        rows = np.nonzero(mask.any(axis=1))[0]
        if len(rows) == 0:
            continue
        # Quantise weights so vertices sharing a weight are added in one call.
        weights = np.round((w * mask).sum(axis=1)[rows] * 1024.0) / 1024.0
        verts = vertex_indices[rows]
        vg = groups[name]
        for value in np.unique(weights):
            vg.add(verts[weights == value].tolist(), float(value), 'REPLACE')


def apply_proximity(obj, rig, vertex_indices=None, influences=3, falloff=4.0):
    names, heads, tails = deform_segments(rig)
    if not names:
        return 0
    points = world_vertices(obj)
    if vertex_indices is None:
        vertex_indices = np.arange(len(points))
        _clear_deform_groups(obj, names)
    else:
        vertex_indices = np.asarray(vertex_indices, dtype=np.int64)
    if len(vertex_indices) == 0:
        return 0
    sel = points[vertex_indices]
    point_side, bone_side = _sides(rig, names, sel)
    idx, w = proximity_weights(sel, heads, tails, influences, falloff,
                               point_side=point_side, bone_side=bone_side)
    _write_weights(obj, names, vertex_indices, idx, w)
    return len(vertex_indices)


def loose_parts(obj):
    """Connected components of the mesh as a label per vertex."""
    count = len(obj.data.vertices)
    parent = np.arange(count)
    edges = np.empty(len(obj.data.edges) * 2, dtype=np.int64)
    obj.data.edges.foreach_get("vertices", edges)
    edges = edges.reshape(-1, 2)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    for a, b in edges.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(count)])


RIGID_MAX_SIZE = 0.06    # loose parts larger than this (body heights) keep their skin weights
RIGID_TINY = 0.02        # parts smaller than this are always rigid (eyes, buttons)
RIGID_DOMINANCE = 0.75   # otherwise rigid only when one bone already holds this share of the part


def bind_rigid_parts(obj, rig, max_fraction=0.02, influences=2, falloff=4.0):
    """Give each small loose part one set of weights taken at its centre.

    Only props: a part must be small, and either tiny or already weighted mostly to
    one bone. Pieces that straddle a joint (a mannequin's knee cap) keep their
    blended weights instead of swinging with one bone."""
    labels = loose_parts(obj)
    total = len(labels)
    if total == 0:
        return 0
    names, heads, tails = deform_segments(rig)
    if not names:
        return 0
    points = world_vertices(obj)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        return 0
    small = unique[(counts <= max(total * max_fraction, 1)) & (counts < counts.max())]
    h = rig_height(rig)
    # a hand and its fingers count as one owner: a prop held in the hand stays rigid
    groups = [FINGER.sub(lambda m: "DEF-hand" + (m.group(3) or ""), _limb_group(n)) if FINGER.match(n)
              else _limb_group(n) for n in names]
    group_index = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    current = read_weights(obj, names)
    grouped = np.zeros((len(points), len(group_index)))
    for col, g in enumerate(groups):
        grouped[:, group_index[g]] += current[:, col]
    count = 0
    for label in small:
        verts = np.nonzero(labels == label)[0]
        center = points[verts].mean(axis=0, keepdims=True)
        radius = np.linalg.norm(points[verts] - center, axis=1).max()
        size = 2 * radius
        part = grouped[verts].sum(axis=0)
        dominance = part.max() / part.sum() if part.sum() > 1e-9 else 1.0
        if size > RIGID_MAX_SIZE * h or (size > RIGID_TINY * h and dominance < RIGID_DOMINANCE):
            continue
        # A bone starting inside the part (e.g. the eye bone in an eyeball) owns it.
        head_dist = np.linalg.norm(heads - center, axis=1)
        inside = np.nonzero(head_dist <= radius * 1.2)[0]
        if len(inside):
            idx = np.array([[inside[np.argmin(head_dist[inside])]]])
            w = np.ones((1, 1))
        else:
            point_side, bone_side = _sides(rig, names, center)
            idx, w = proximity_weights(center, heads, tails, influences, falloff,
                                       point_side=point_side, bone_side=bone_side)
        _write_weights(obj, names, verts, np.repeat(idx, len(verts), axis=0),
                       np.repeat(w, len(verts), axis=0), replace=True)
        count += 1
    return count


def _parent_auto(context, obj, rig):
    utils.deselect_all(context)
    obj.select_set(True)
    rig.select_set(True)
    context.view_layer.objects.active = rig
    with context.temp_override(active_object=rig, object=rig, selected_objects=[obj, rig],
                               selected_editable_objects=[obj, rig]):
        bpy.ops.object.parent_set(type='ARMATURE_AUTO', keep_transform=True)


def heat_weights(context, obj, rig, force_rescale=False):
    """Bone heat weighting; rescales temporary copies when the character size is extreme."""
    deform = [b.name for b in rig.data.bones if b.use_deform]
    _clear_deform_groups(obj, deform)
    height = rig_height(rig)
    low, high = HEAT_HEIGHT_RANGE
    if low <= height <= high and not force_rescale:
        _parent_auto(context, obj, rig)
        return
    factor = 2.0 / max(height, 1e-6)
    scale = Matrix.Scale(factor, 4)
    tmp_rig = rig.copy()
    tmp_rig.data = rig.data.copy()
    tmp_rig.animation_data_clear()
    tmp_obj = obj.copy()
    tmp_obj.data = obj.data.copy()
    tmp_obj.parent = None
    tmp_obj.modifiers.clear()
    for o in (tmp_rig, tmp_obj):
        context.scene.collection.objects.link(o)
    tmp_rig.matrix_world = scale @ rig.matrix_world
    tmp_obj.matrix_world = scale @ obj.matrix_world
    try:
        _parent_auto(context, tmp_obj, tmp_rig)
        lookup = {vg.index: vg.name for vg in tmp_obj.vertex_groups if vg.name in deform}
        weights = {}
        for v in tmp_obj.data.vertices:
            for g in v.groups:
                name = lookup.get(g.group)
                if name and g.weight > 0:
                    weights.setdefault((name, round(g.weight, 4)), []).append(v.index)
        groups = {}
        for (name, weight), verts in weights.items():
            vg = groups.get(name) or obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
            groups[name] = vg
            vg.add(verts, weight, 'REPLACE')
    finally:
        mesh, arm = tmp_obj.data, tmp_rig.data
        bpy.data.objects.remove(tmp_obj)
        bpy.data.objects.remove(tmp_rig)
        bpy.data.meshes.remove(mesh)
        bpy.data.armatures.remove(arm)
    parent_keep_transform(obj, rig)


# ---------------------------------------------------------------------------
# Weight arrays
# ---------------------------------------------------------------------------

def read_weights(obj, names):
    """(N, len(names)) array of the weights of the named vertex groups."""
    order = {obj.vertex_groups[n].index: names.index(n) for n in names if n in obj.vertex_groups}
    w = np.zeros((len(obj.data.vertices), len(names)))
    if not order:
        return w
    for v in obj.data.vertices:
        for g in v.groups:
            col = order.get(g.group)
            if col is not None:
                w[v.index, col] = g.weight
    return w


def write_weights(obj, names, weights, vertices=None):
    """Write weights (N, len(names)) back; zero weights are removed."""
    vertices = np.arange(len(obj.data.vertices)) if vertices is None else np.asarray(vertices)
    for col, name in enumerate(names):
        vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        values = weights[vertices, col]
        zero = vertices[values <= 1e-6]
        if len(zero):
            vg.remove(zero.tolist())
        keep = values > 1e-6
        rounded = np.round(values[keep] * 1024.0) / 1024.0
        verts = vertices[keep]
        for value in np.unique(rounded):
            vg.add(verts[rounded == value].tolist(), float(value), 'REPLACE')


def _vertex_snapshot(obj, vertices=None):
    """{group name: {vertex: weight}} for the given vertices (all if None)."""
    wanted = None if vertices is None else set(int(v) for v in vertices)
    names = {vg.index: vg.name for vg in obj.vertex_groups}
    snap = {}
    for v in obj.data.vertices:
        if wanted is not None and v.index not in wanted:
            continue
        for g in v.groups:
            snap.setdefault(names[g.group], {})[v.index] = g.weight
    return snap


def _restore_snapshot(obj, snap, vertices, groups=None):
    """Put back the weights of 'vertices' for 'groups' (all groups if None) from a snapshot."""
    verts = [int(v) for v in vertices]
    for vg in obj.vertex_groups:
        if groups is not None and vg.name not in groups:
            continue
        vg.remove(verts)
    for name, values in snap.items():
        if groups is not None and name not in groups:
            continue
        vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        wanted = set(verts)
        for v, w in values.items():
            if v in wanted:
                vg.add([v], w, 'REPLACE')


# ---------------------------------------------------------------------------
# Proxy meshes: voxel, split parts, decimate, shape key mix
# ---------------------------------------------------------------------------

def _proxy(context, obj, kind, rig, voxel_resolution=64, ratio=0.5):
    """Temporary copy of obj (world space, no modifiers) for heat weighting."""
    depsgraph = context.evaluated_depsgraph_get()
    # Current shape key mix, without the modifiers
    states = [(m, m.show_viewport) for m in obj.modifiers]
    for m, _v in states:
        m.show_viewport = False
    depsgraph.update()
    mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    for m, v in states:
        m.show_viewport = v
    proxy = bpy.data.objects.new(obj.name + "_rigger_proxy", mesh)
    context.scene.collection.objects.link(proxy)
    proxy.matrix_world = obj.matrix_world.copy()
    for vg in list(proxy.vertex_groups):
        proxy.vertex_groups.remove(vg)
    if kind == 'VOXEL':
        mod = proxy.modifiers.new("remesh", 'REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = max(rig_height(rig) / max(voxel_resolution, 8), 1e-4)
        mod.use_remove_disconnected = False
        _apply_modifiers(context, proxy)
    elif kind == 'DECIMATE':
        mod = proxy.modifiers.new("decimate", 'DECIMATE')
        mod.ratio = max(0.01, min(1.0, ratio))
        _apply_modifiers(context, proxy)
    return proxy


def _apply_modifiers(context, obj):
    with context.temp_override(object=obj, active_object=obj, selected_objects=[obj],
                               selected_editable_objects=[obj]):
        for mod in list(obj.modifiers):
            bpy.ops.object.modifier_apply(modifier=mod.name)


def transfer_weights(context, src, dst, names, nearest_vertex=False):
    """Copy the named vertex groups from src to dst by surface proximity (Data Transfer)."""
    for name in names:
        if name in src.vertex_groups and name not in dst.vertex_groups:
            dst.vertex_groups.new(name=name)
    mod = dst.modifiers.new("rigger_transfer", 'DATA_TRANSFER')
    mod.object = src
    mod.use_vert_data = True
    mod.data_types_verts = {'VGROUP_WEIGHTS'}
    mod.vert_mapping = 'NEAREST' if nearest_vertex else 'POLYINTERP_NEAREST'
    mod.layers_vgroup_select_src = 'ALL'
    mod.layers_vgroup_select_dst = 'NAME'
    mod.mix_mode = 'REPLACE'
    dst.modifiers.move(dst.modifiers.find(mod.name), 0)
    with context.temp_override(object=dst, active_object=dst, selected_objects=[dst],
                               selected_editable_objects=[dst]):
        bpy.ops.object.modifier_apply(modifier=mod.name)


def _remove_object(obj):
    mesh = obj.data
    bpy.data.objects.remove(obj)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def heat_by_proxy(context, obj, rig, kind, voxel_resolution=64, ratio=0.5):
    """Heat weighting on a proxy, transferred to obj."""
    deform = [b.name for b in rig.data.bones if b.use_deform]
    proxy = _proxy(context, obj, kind, rig, voxel_resolution, ratio)
    try:
        heat_weights(context, proxy, rig)
        _clear_deform_groups(obj, deform)
        transfer_weights(context, proxy, obj, deform, nearest_vertex=(kind == 'MIX'))
    finally:
        _remove_object(proxy)
    parent_keep_transform(obj, rig)


def heat_split_parts(context, obj, rig):
    """Heat weighting of every loose part on its own (clothes, props made of many pieces)."""
    deform = [b.name for b in rig.data.bones if b.use_deform]
    proxy = _proxy(context, obj, 'MIX', rig)
    names = [proxy.name]
    try:
        utils.set_mode(proxy, 'EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
        parts = [o for o in context.selected_objects if o.type == 'MESH']
        names = [p.name for p in parts]
        for part in parts:
            heat_weights(context, part, rig)
            part.parent = None
            part.matrix_world = obj.matrix_world.copy()
        # Join the weighted parts back to transfer them in one go
        utils.deselect_all(context)
        for part in parts:
            part.select_set(True)
        context.view_layer.objects.active = parts[0]
        if len(parts) > 1:
            bpy.ops.object.join()
        joined = context.view_layer.objects.active
        names = [joined.name]
        _clear_deform_groups(obj, deform)
        transfer_weights(context, joined, obj, deform, nearest_vertex=True)
    finally:
        for name in names:
            part = bpy.data.objects.get(name)
            if part is not None:
                _remove_object(part)
    parent_keep_transform(obj, rig)


# ---------------------------------------------------------------------------
# Weight corrections
# ---------------------------------------------------------------------------

def _chains(rig):
    """Deform segments of each limb part: [[DEF-upper, twists...], [DEF-lower, twists...], ...]."""
    chains = []
    names = [b.name for b in rig.data.bones if b.use_deform]
    for limb in utils.get_meta(rig).get("limbs", []):
        for main in limb["deform"][-3:-1]:
            stem, suffix = utils.split_side(main)
            twists = sorted(n for n in names if n.startswith(f"{stem}_twist.") and n.endswith(suffix))
            if twists:
                chains.append([main] + twists)
    return chains


def smooth_twist_weights(obj, rig):
    """Spread each limb segment's weight over its twist bones with a linear gradient."""
    mw = rig.matrix_world
    points = world_vertices(obj)
    count = 0
    for chain in _chains(rig):
        w = read_weights(obj, chain)
        total = w.sum(axis=1)
        mask = total > 1e-6
        if not mask.any():
            continue
        bones = [rig.data.bones[n] for n in chain]
        head = np.array(mw @ bones[0].head_local)
        tail = np.array(mw @ max(bones, key=lambda b: (b.tail_local - bones[0].head_local).length).tail_local)
        axis = tail - head
        t = np.clip(((points - head) @ axis) / max(axis @ axis, 1e-12), 0.0, 1.0)
        n = len(chain)
        # Bone i carries the twist of the point i / (n - 1) along the segment (the first none,
        # the last all of it): tent functions between those points spread the twist evenly
        order = np.argsort([(rig.data.bones[c].head_local - bones[0].head_local).length for c in chain])
        centers = np.arange(n) / (n - 1)
        tent = np.clip(1.0 - np.abs(t[:, None] - centers[None, :]) * (n - 1), 0.0, 1.0)
        tent[t <= centers[0], 0] = 1.0
        tent[t >= centers[-1], n - 1] = 1.0
        tent /= np.maximum(tent.sum(axis=1, keepdims=True), 1e-12)
        new = np.zeros_like(w)
        new[:, order] = tent * total[:, None]
        w[mask] = new[mask]
        write_weights(obj, chain, w, np.nonzero(mask)[0])
        count += 1
    return count


def _move_weight(obj, names_from, name_to, factor):
    """Move factor (per vertex, 0..1) of the weights of names_from onto name_to."""
    names = list(names_from) + [name_to]
    w = read_weights(obj, names)
    moved = w[:, :-1] * factor[:, None]
    w[:, :-1] -= moved
    w[:, -1] += moved.sum(axis=1)
    rows = np.nonzero(moved.sum(axis=1) > 1e-6)[0]
    if len(rows):
        write_weights(obj, names, w, rows)
    return len(rows)


def _deform_chain(rig, main):
    stem, suffix = utils.split_side(main)
    return [main] + sorted(b.name for b in rig.data.bones if b.use_deform and
                           b.name.startswith(f"{stem}_twist.") and b.name.endswith(suffix))


def improve_hips_weights(obj, rig, offset=0.0, ramp=0.04, tilt=0.0):
    """Thigh weights should not climb above the hip joint.

    The cut is a plane through the hip joint raised by 'offset' (in body heights),
    tilted 'tilt' height units per unit of depth (positive: higher at the back),
    blending over 'ramp' body heights."""
    meta = utils.get_meta(rig)
    hips = meta.get("deform_of", {}).get("hips")
    if not hips or hips not in rig.data.bones:
        return 0
    points = world_vertices(obj)
    h = rig_height(rig)
    count = 0
    for limb in meta.get("limbs", []):
        if limb["type"] != "leg":
            continue
        thigh = limb["deform"][-3]
        joint = rig.matrix_world @ rig.data.bones[thigh].head_local
        cut = joint.z + offset * h + tilt * (points[:, 1] - joint.y)
        factor = np.clip((points[:, 2] - cut) / (ramp * h), 0.0, 1.0)
        count += _move_weight(obj, _deform_chain(rig, thigh), hips, factor)
    return count


def improve_heel_weights(obj, rig):
    """The heel (below and behind the ankle) follows the foot, not the shin."""
    points = world_vertices(obj)
    h = rig_height(rig)
    count = 0
    for limb in utils.get_meta(rig).get("limbs", []):
        if limb["type"] != "leg":
            continue
        foot = limb["deform"][-1]
        ankle = rig.matrix_world @ rig.data.bones[foot].head_local
        below = np.clip((ankle.z - points[:, 2]) / (0.015 * h) + 0.5, 0.0, 1.0)
        count += _move_weight(obj, _deform_chain(rig, limb["deform"][-2]), foot, below)
    return count


HEAD_BLEND = 0.08  # neck-to-head weight hand-off, in body heights


def refine_head_weights(obj, rig):
    """Everything above the bottom of the head belongs to the head, with a short blend."""
    meta = utils.get_meta(rig)
    neck = meta.get("neck") or {}
    head = meta.get("deform_of", {}).get(neck.get("head", ""))
    if not head or head not in rig.data.bones:
        return 0
    necks = [meta["deform_of"][n] for n in neck.get("fk", []) if n in meta.get("deform_of", {})]
    if not necks:
        return 0
    points = world_vertices(obj)
    z0 = (rig.matrix_world @ rig.data.bones[head].head_local).z
    factor = np.clip((points[:, 2] - z0) / (HEAD_BLEND * rig_height(rig)) + 0.5, 0.0, 1.0)
    count = _move_weight(obj, necks, head, factor)
    # Head and shoulders never share skin: the head's skin drops shoulder, arm and
    # hand weights, and the shoulders' skin (trapezius, collarbones) drops head weight.
    names = [b.name for b in rig.data.bones if b.use_deform]
    arm_like = [n for n in names if ARM_PART.search(n)]
    if not arm_like:
        return count
    w = read_weights(obj, names)
    main = np.argmax(w, axis=1)
    hi, ni = names.index(head), names.index(necks[-1])
    arm_cols = [names.index(n) for n in arm_like]
    on_head = (main == hi) & (w[:, arm_cols].sum(axis=1) > 0)
    w[on_head, hi] += w[on_head][:, arm_cols].sum(axis=1)
    w[np.ix_(on_head, arm_cols)] = 0.0
    on_shoulders = np.isin(main, arm_cols) & (w[:, hi] > 0)
    w[on_shoulders, ni] += w[on_shoulders, hi]
    w[on_shoulders, hi] = 0.0
    rows = np.nonzero(on_head | on_shoulders)[0]
    if len(rows):
        write_weights(obj, names, w, rows)
    return count + len(rows)


ARM_PART = re.compile(r"^DEF-(shoulder|upper_arm|forearm|hand|thumb|index|middle|ring|pinky)")
FINGER = re.compile(r"^DEF-(thumb|index|middle|ring|pinky)\.(\d+)(\.[LR])?$")


def _limb_group(name):
    """Twist and fan bones belong to their limb bone."""
    return re.sub(r"_(twist\.\d+|fan[^.]*(\.\d+)?)", "", name)


FAN_BAND = (0.01, 0.025)  # fan weights fade out between these distances from the joint (body heights)


def tighten_fan_weights(obj, rig, band=FAN_BAND):
    """Fans only weigh the skin right around their joint.

    Heat weighting lets a fan reach far along the thigh or upper arm, where the
    skin then turns only part of the way and the joint collapses. Fan weight
    beyond the band goes to the vertex's strongest other bone."""
    names = [b.name for b in rig.data.bones if b.use_deform]
    fans = [i for i, n in enumerate(names) if "_fan" in n]
    if not fans:
        return 0
    w = read_weights(obj, names)
    points = world_vertices(obj)
    h = rig_height(rig)
    others = np.array([i for i in range(len(names)) if i not in fans])
    inner, outer = band
    changed = set()
    for f in fans:
        joint = np.array(rig.matrix_world @ rig.data.bones[names[f]].head_local)
        d = np.linalg.norm(points - joint, axis=1)
        keep = np.clip((outer * h - d) / ((outer - inner) * h), 0.0, 1.0)
        moved = w[:, f] * (1.0 - keep)
        for v in np.nonzero(moved > 1e-6)[0]:
            target = others[np.argmax(w[v, others])]
            w[v, target] += moved[v]
            w[v, f] -= moved[v]
            changed.add(int(v))
    if changed:
        write_weights(obj, names, w, np.array(sorted(changed)))
    return len(changed)


def clean_finger_weights(obj, rig):
    """Fingers do not stick together: skin of a finger drops the weights of other
    fingers (except between neighbouring base segments, the web of the hand) and,
    past the first segment, of the hand."""
    names = [b.name for b in rig.data.bones if b.use_deform]
    info = []
    for n in names:
        m = FINGER.match(n)
        info.append((m.group(1), int(m.group(2)), m.group(3) or "") if m else None)
    if not any(info):
        return 0
    w = read_weights(obj, names)
    main = np.argmax(w, axis=1)
    hands = {side: names.index(f"DEF-hand{side}") for side in (".L", ".R") if f"DEF-hand{side}" in names}
    rows = []
    for v in np.nonzero(w.sum(axis=1) > 1e-6)[0]:
        own = info[main[v]]
        if own is None:
            continue
        finger, seg, side = own
        drop = []
        for c in np.nonzero(w[v] > 0)[0]:
            other = info[c]
            if other is not None and other[2] == side and other[0] != finger:
                if not (seg == 1 and other[1] == 1):
                    drop.append(c)
        if seg >= 2 and side in hands and w[v, hands[side]] > 0:
            drop.append(hands[side])
        if drop:
            w[v, main[v]] += w[v, drop].sum()
            w[v, drop] = 0.0
            rows.append(v)
    if rows:
        write_weights(obj, names, w, np.array(rows))
    return len(rows)


def bind_facial_objects(rig, eyes=(), teeth=(), tongue=()):
    """Eyeballs to the eye bones, teeth to the jaw / upper teeth, tongue to the tongue bones."""
    deform = {b.name for b in rig.data.bones if b.use_deform}
    count = 0
    for obj in eyes:
        points = world_vertices(obj)
        eyes_def = [n for n in deform if utils.split_side(n)[0] == "DEF-eye"]
        if not eyes_def:
            continue
        names = sorted(eyes_def)
        heads = np.array([rig.matrix_world @ rig.data.bones[n].head_local for n in names])
        nearest = np.argmin(np.linalg.norm(points[:, None, :] - heads[None], axis=2), axis=1)
        _clear_deform_groups(obj, list(deform))
        for i, name in enumerate(names):
            verts = np.nonzero(nearest == i)[0].tolist()
            if verts:
                (obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)).add(verts, 1.0, 'REPLACE')
        count += 1
    for obj in teeth:
        lower = "low" in obj.name.lower() or "bot" in obj.name.lower()
        candidates = ["DEF-teeth_lower", "DEF-jaw"] if lower else ["DEF-teeth_upper", "DEF-face", "DEF-head"]
        name = next((c for c in candidates if c in deform), None)
        if name:
            _clear_deform_groups(obj, list(deform))
            (obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)).add(
                list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
            count += 1
    tongue_bones = sorted(n for n in deform if n.startswith("DEF-tongue"))
    for obj in tongue:
        if not tongue_bones:
            continue
        _clear_deform_groups(obj, list(deform))
        names, heads, tails = deform_segments(rig)
        keep = [i for i, n in enumerate(names) if n in tongue_bones]
        idx, w = proximity_weights(world_vertices(obj), heads[keep], tails[keep], 2, 4.0)
        _write_weights(obj, [names[i] for i in keep], np.arange(len(obj.data.vertices)), idx, w)
        count += 1
    return count


BORDER_GROUP = "rigger_lid_border"


def apply_eyelid_borders(obj, rig):
    """Vertices stored with 'Set Left/Right' follow the closest upper or lower eyelid bone fully."""
    deform = [b.name for b in rig.data.bones if b.use_deform]
    count = 0
    for suffix in (".L", ".R"):
        vg = obj.vertex_groups.get(BORDER_GROUP + suffix)
        eye = f"DEF-eye{suffix}"
        if vg is None or eye not in rig.data.bones:
            continue
        verts = [v.index for v in obj.data.vertices if any(g.group == vg.index and g.weight > 0 for g in v.groups)]
        if not verts:
            continue
        rows = {row: [n for n in deform if n.startswith(f"DEF-lid_{row}") and n.endswith(suffix)]
                for row in ("upper", "lower")}
        if not rows["upper"] or not rows["lower"]:
            continue
        center = rig.matrix_world @ rig.data.bones[eye].head_local
        pts = world_vertices(obj)[verts]
        names, heads, tails = deform_segments(rig)
        for vgroup in obj.vertex_groups:
            if vgroup.name in deform:
                vgroup.remove(verts)
        above = pts[:, 2] >= center.z
        for row, mask in (("upper", above), ("lower", ~above)):
            if not mask.any():
                continue
            index = [names.index(n) for n in rows[row]]
            dist = segment_distances(pts[mask], heads[index], tails[index])
            nearest = np.argmin(dist, axis=1)
            chosen = np.array(verts)[mask]
            for i, name in enumerate(rows[row]):
                ids = chosen[nearest == i].tolist()
                if ids:
                    (obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)).add(ids, 1.0, 'REPLACE')
                    count += len(ids)
    return count


def open_edges(obj):
    """Edges used by a single face: holes in the mesh, where heat weighting can fail."""
    me = obj.data
    if not len(me.polygons):
        return 0
    edge_of_loop = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("edge_index", edge_of_loop)
    return int((np.bincount(edge_of_loop, minlength=len(me.edges)) == 1).sum())


def warnings_for(obj, rig=None, method='AUTO', optimize=0):
    """Problems worth knowing before binding, as short sentences."""
    msgs = []
    if any(abs(v) > 1e-6 for v in (*obj.delta_location, *obj.delta_rotation_euler)) or \
            any(abs(v - 1.0) > 1e-6 for v in obj.delta_scale):
        msgs.append(f"{obj.name} has delta transforms (Object > Apply > ... or clear them)")
    if obj.matrix_world.determinant() < 0 or obj.scale[0] * obj.scale[1] * obj.scale[2] < 0:
        msgs.append(f"{obj.name} has a negative scale: apply the scale first")
    elif max(obj.scale) - min(obj.scale) > 1e-4:
        msgs.append(f"{obj.name} has a non-uniform scale: apply it (Ctrl A > Scale) for even weights")
    others = {m.object.name for m in obj.modifiers if m.type == 'ARMATURE' and m.object and m.object != rig}
    if others:
        msgs.append(f"{obj.name} already has an Armature modifier for {', '.join(sorted(others))}")
    if method in {'AUTO', 'HEAT'}:
        holes = open_edges(obj)
        if holes:
            msgs.append(f"{obj.name} has {holes} open edges (holes): if some parts get bad weights, "
                        "try the Voxelized method")
        if not optimize and len(obj.data.polygons) > 100000:
            msgs.append(f"{obj.name} has {len(obj.data.polygons)} faces: Optimize High Res binds faster")
    if rig is not None:
        corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
        lo, hi = Vector(map(min, *corners)), Vector(map(max, *corners))
        bones = [rig.matrix_world @ p for b in rig.data.bones if b.use_deform for p in (b.head_local, b.tail_local)]
        if bones:
            blo, bhi = Vector(map(min, *bones)), Vector(map(max, *bones))
            if any(hi[i] < blo[i] or lo[i] > bhi[i] for i in range(3)):
                msgs.append(f"{obj.name} does not overlap the rig: wrong object, or the rig is elsewhere")
    return msgs


# ---------------------------------------------------------------------------
# Bind
# ---------------------------------------------------------------------------

def bind_mesh(context, obj, rig, method='AUTO', influences=3, falloff=4.0, rigid_parts=True,
              rigid_fraction=0.02, selected_bones=False, selected_verts=False, apply_shape_keys=False,
              split_parts=False, optimize=0, voxel_resolution=64, scale_fix=False, smooth_twist=False,
              improve_hips=False, improve_heels=False, refine_head=False, clean_fingers=False,
              tighten_fans=False, preserve_volume=True, shoulder_volume=1.0,
              parent=True):
    """Weight obj to rig. Returns (vertices filled by proximity, rigid parts)."""
    # Locked groups are never changed; with Selected Vertices Only the others keep their weights.
    locked = [vg.name for vg in obj.vertex_groups if vg.lock_weight]
    locked_snap = {n: w for n, w in _vertex_snapshot(obj).items() if n in locked}
    for vg in obj.vertex_groups:
        vg.lock_weight = False
    keep_verts = None
    if selected_verts:
        keep_verts = [v.index for v in obj.data.vertices if not v.select]
        keep_snap = _vertex_snapshot(obj, keep_verts)
    toggled = []
    if selected_bones:
        for bone in rig.data.bones:
            pb = rig.pose.bones[bone.name]
            if bone.use_deform and not utils.is_pose_bone_selected(pb):
                bone.use_deform = False
                toggled.append(bone)
    try:
        deform = {b.name for b in rig.data.bones if b.use_deform}
        filled = 0
        dense = optimize and len(obj.data.polygons) > optimize
        if method == 'VOXEL':
            heat_by_proxy(context, obj, rig, 'VOXEL', voxel_resolution)
        elif method in {'HEAT', 'AUTO'}:
            if split_parts:
                heat_split_parts(context, obj, rig)
            elif dense:
                heat_by_proxy(context, obj, rig, 'DECIMATE', ratio=optimize / len(obj.data.polygons))
            elif apply_shape_keys and obj.data.shape_keys is not None:
                heat_by_proxy(context, obj, rig, 'MIX')
            else:
                heat_weights(context, obj, rig, force_rescale=scale_fix)
        if method in {'AUTO', 'VOXEL'}:
            missing = unweighted_vertices(obj, deform)
            if missing:
                filled = apply_proximity(obj, rig, missing, influences, falloff)
        if method == 'PROXIMITY':
            apply_proximity(obj, rig, None, influences, falloff)
            if parent:
                parent_keep_transform(obj, rig)
        rigid = bind_rigid_parts(obj, rig, rigid_fraction) if rigid_parts else 0
    finally:
        for bone in toggled:
            bone.use_deform = True
    if smooth_twist:
        smooth_twist_weights(obj, rig)
    if improve_hips:
        improve_hips_weights(obj, rig)
    if improve_heels:
        improve_heel_weights(obj, rig)
    if refine_head:
        refine_head_weights(obj, rig)
    if clean_fingers:
        clean_finger_weights(obj, rig)
    if tighten_fans:
        tighten_fan_weights(obj, rig)
    apply_eyelid_borders(obj, rig)
    if keep_verts is not None:
        _restore_snapshot(obj, keep_snap, keep_verts)
    for name, values in locked_snap.items():
        vg = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
        vg.remove(list(range(len(obj.data.vertices))))
        for v, w in values.items():
            vg.add([v], w, 'REPLACE')
    for name in locked:
        if name in obj.vertex_groups:
            obj.vertex_groups[name].lock_weight = True
    if not parent and obj.parent == rig:
        mw = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = mw
    ensure_armature_modifier(obj, rig, preserve_volume, shoulder_volume)
    return filled, rigid


def smooth_weights(context, obj, factor, repeat):
    with context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.vertex_group_smooth(group_select_mode='BONE_DEFORM', factor=factor,
                                           repeat=repeat, expand=0.0)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode='BONE_DEFORM',
                                                  lock_active=False)


def limit_influences(context, obj, limit):
    with context.temp_override(object=obj, active_object=obj):
        bpy.ops.object.vertex_group_limit_total(group_select_mode='BONE_DEFORM', limit=limit)
        bpy.ops.object.vertex_group_normalize_all(group_select_mode='BONE_DEFORM',
                                                  lock_active=False)


class RIGGER_OT_bind(bpy.types.Operator):
    """Bind the selected meshes to the rig (select meshes, then the rig last)"""
    bl_idname = "rigger.bind"
    bl_label = "Bind Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    method: EnumProperty(name="Method", default='AUTO', items=[
        ('AUTO', "Heat + Fallback", "Bone heat weighting, vertices it misses get proximity weights"),
        ('HEAT', "Heat", "Blender bone heat weighting only"),
        ('VOXEL', "Voxelized", "Heat weighting on a watertight voxel copy, for layered clothes and "
                               "meshes that are not closed"),
        ('PROXIMITY', "Proximity", "Distance based weights, robust on any mesh"),
    ])
    voxel_resolution: IntProperty(name="Voxel Resolution", default=64, min=16, max=512,
                                  description="Voxels over the character's height (Voxelized)")
    split_parts: BoolProperty(name="Split Parts", default=True,
                              description="Heat weighting on each loose part separately")
    optimize: IntProperty(name="Optimize High Res", default=0, min=0,
                          description="Heat weighting on a decimated copy above this many faces (0 = off)")
    scale_fix: BoolProperty(name="Scale Fix", default=False,
                            description="Work on a rescaled copy (very small or dense meshes)")
    selected_bones: BoolProperty(name="Selected Bones Only", default=False,
                                 description="Only weight to the bones selected in the rig")
    selected_verts: BoolProperty(name="Selected Vertices Only", default=False,
                                 description="Only change the weights of the selected vertices")
    apply_shape_keys: BoolProperty(name="Apply Shape Keys", default=False,
                                   description="Weight the mesh as deformed by its current shape keys")
    smooth_twist: BoolProperty(name="Smooth Twist Weights", default=True,
                               description="Gradient of the limb weights over the twist bones")
    improve_hips: BoolProperty(name="Improve Hips Weights", default=False,
                               description="Thigh weights stop at the hip joint. Off by default: the cut "
                                           "creases the groin and stretches the buttock when the leg lifts")
    improve_heels: BoolProperty(name="Improve Heel Weights", default=False,
                                description="The heel follows the foot")
    refine_head: BoolProperty(name="Refine Head Weights", default=True,
                              description="Everything above the bottom of the head follows the head, "
                                          "and the head and shoulders stop sharing weights")
    tighten_fans: BoolProperty(
        name="Tighten Fan Weights", default=True,
        description="Joint fans only weigh the skin right around their joint, so bent elbows and knees "
                    "keep their volume")
    clean_fingers: BoolProperty(
        name="Clean Finger Weights", default=True,
        description="Fingers stop sharing weights with other fingers and, past the first segment, "
                    "with the hand")
    eyes: StringProperty(name="Eyeballs", default="", description="Comma separated eyeball objects")
    teeth: StringProperty(name="Teeth", default="", description="Comma separated teeth objects "
                                                                 "('lower' in the name = lower teeth)")
    tongue: StringProperty(name="Tongue", default="", description="Comma separated tongue objects")
    preserve_volume: BoolProperty(name="Preserve Volume", default=True,
                                  description="Dual quaternion skinning (off for game engines)")
    shoulder_volume: FloatProperty(
        name="Shoulder Volume", default=0.3, min=0.0, max=1.0, subtype='FACTOR',
        description="How much the shoulders preserve volume (1 = like the rest of the body). Lower values "
                    "stop a raised arm from ballooning the shoulder; twisting limbs keep full volume")
    parent: BoolProperty(name="Parent to Rig", default=True)
    influences: IntProperty(name="Max Influences", default=3, min=1, max=8,
                            description="Bones per vertex for proximity weights")
    falloff: FloatProperty(name="Falloff", default=4.0, min=1.0, max=12.0,
                           description="Higher = sharper transitions between bones")
    rigid_parts: BoolProperty(
        name="Rigid Small Parts", default=True,
        description="Bind small loose parts (eyes, buttons, buckles) rigidly so they never stretch")
    rigid_size: FloatProperty(
        name="Small Part Size", default=2.0, min=0.0, max=50.0, subtype='PERCENTAGE',
        description="Loose parts with at most this share of the mesh's vertices are bound rigidly")
    smooth: BoolProperty(name="Smooth Weights", default=False)
    smooth_factor: FloatProperty(name="Smooth Factor", default=0.5, min=0.0, max=1.0)
    smooth_repeat: IntProperty(name="Smooth Iterations", default=2, min=1, max=20)
    limit: IntProperty(name="Limit Influences", default=0, min=0, max=8,
                       description="Maximum bones per vertex (game engines often use 4). 0 = off")

    @classmethod
    def poll(cls, context):
        rig = utils.find_rig_for(context.object)
        return rig is not None and any(o.type == 'MESH' for o in context.selected_objects)

    def invoke(self, context, event):
        rig = utils.find_rig_for(context.object)
        self._warnings = [w for o in context.selected_objects if o.type == 'MESH'
                          for w in warnings_for(o, rig, self.method, self.optimize)]
        return context.window_manager.invoke_props_dialog(self, width=380)

    def draw(self, context):
        layout = self.layout
        for i, warning in enumerate(getattr(self, "_warnings", [])):
            layout.label(text=warning, icon='ERROR' if i == 0 else 'BLANK1')
        layout.prop(self, "method")
        col = layout.column(heading="Engine")
        if self.method == 'VOXEL':
            col.prop(self, "voxel_resolution")
        if self.method in {'AUTO', 'HEAT'}:
            col.prop(self, "split_parts")
            col.prop(self, "optimize")
            col.prop(self, "apply_shape_keys")
            col.prop(self, "scale_fix")
        col.prop(self, "influences")
        col.prop(self, "falloff")
        col.prop(self, "rigid_parts")
        if self.rigid_parts:
            col.prop(self, "rigid_size")
        col = layout.column(heading="Only")
        col.prop(self, "selected_bones")
        col.prop(self, "selected_verts")
        col = layout.column(heading="Improve")
        col.prop(self, "smooth_twist")
        col.prop(self, "improve_hips")
        col.prop(self, "improve_heels")
        col.prop(self, "refine_head")
        col.prop(self, "clean_fingers")
        col.prop(self, "tighten_fans")
        col.prop(self, "smooth")
        if self.smooth:
            row = col.row(align=True)
            row.prop(self, "smooth_factor")
            row.prop(self, "smooth_repeat")
        col.prop(self, "limit")
        col = layout.column(heading="Facial Features")
        col.prop(self, "eyes")
        col.prop(self, "teeth")
        col.prop(self, "tongue")
        col = layout.column(heading="Result")
        col.prop(self, "preserve_volume")
        if self.preserve_volume:
            col.prop(self, "shoulder_volume")
        col.prop(self, "parent")

    def execute(self, context):
        rig = utils.find_rig_for(context.object)
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        report = []
        named = lambda text: [bpy.data.objects[n.strip()] for n in text.split(",")  # noqa: E731
                              if n.strip() in bpy.data.objects]
        special = {"eyes": named(self.eyes), "teeth": named(self.teeth), "tongue": named(self.tongue)}
        special_objs = {o for objs in special.values() for o in objs}
        for obj in meshes:
            report.extend(warnings_for(obj, rig, self.method, self.optimize))
        for obj in [o for o in meshes if o not in special_objs]:
            filled, rigid = bind_mesh(context, obj, rig, self.method, self.influences, self.falloff,
                                      self.rigid_parts, self.rigid_size / 100.0,
                                      selected_bones=self.selected_bones, selected_verts=self.selected_verts,
                                      apply_shape_keys=self.apply_shape_keys, split_parts=self.split_parts,
                                      optimize=self.optimize, voxel_resolution=self.voxel_resolution,
                                      scale_fix=self.scale_fix, smooth_twist=self.smooth_twist,
                                      improve_hips=self.improve_hips, improve_heels=self.improve_heels,
                                      refine_head=self.refine_head, clean_fingers=self.clean_fingers,
                                      tighten_fans=self.tighten_fans,
                                      preserve_volume=self.preserve_volume, shoulder_volume=self.shoulder_volume,
                                      parent=self.parent)
            if self.smooth:
                smooth_weights(context, obj, self.smooth_factor, self.smooth_repeat)
            if self.limit:
                limit_influences(context, obj, self.limit)
            if filled:
                report.append(f"{obj.name}: {filled} vertices weighted by proximity")
            if rigid:
                report.append(f"{obj.name}: {rigid} small part(s) bound rigidly")
        if special_objs:
            bind_facial_objects(rig, **special)
            for obj in special_objs:
                if self.parent:
                    parent_keep_transform(obj, rig)
                ensure_armature_modifier(obj, rig, self.preserve_volume)
        from .shapes import fit_shapes
        fitted = fit_shapes(context, rig)
        utils.set_mode(rig, 'OBJECT')
        msg = f"Bound {len(meshes)} mesh(es) to '{rig.name}', fitted {fitted} control shapes"
        if report:
            msg += ". " + "; ".join(report)
        self.report({'INFO'}, msg)
        return {'FINISHED'}


class RIGGER_OT_eyelid_border(bpy.types.Operator):
    """Store the selected vertices (Edit Mode) as the eyelid border of one side: when binding, they follow the
    eyelid bones fully"""
    bl_idname = "rigger.eyelid_border"
    bl_label = "Set Eyelid Border"
    bl_options = {'REGISTER', 'UNDO'}

    side: EnumProperty(name="Side", items=[('L', "Left", ""), ('R', "Right", ""), ('CLEAR', "Clear", "")])

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == 'MESH'

    def execute(self, context):
        obj = context.object
        mode = obj.mode
        if mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        if self.side == 'CLEAR':
            for suffix in (".L", ".R"):
                vg = obj.vertex_groups.get(BORDER_GROUP + suffix)
                if vg is not None:
                    obj.vertex_groups.remove(vg)
        else:
            verts = [v.index for v in obj.data.vertices if v.select]
            if not verts:
                if mode == 'EDIT':
                    bpy.ops.object.mode_set(mode='EDIT')
                self.report({'WARNING'}, "Select the vertices around the eyelid first (Edit Mode)")
                return {'CANCELLED'}
            name = f"{BORDER_GROUP}.{self.side}"
            vg = obj.vertex_groups.get(name)
            if vg is not None:
                obj.vertex_groups.remove(vg)
            obj.vertex_groups.new(name=name).add(verts, 1.0, 'REPLACE')
            self.report({'INFO'}, f"Eyelid border {self.side}: {len(verts)} vertices")
        if mode == 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        return {'FINISHED'}


class RIGGER_OT_unbind(bpy.types.Operator):
    """Remove the rig binding (armature modifier, parent and rig vertex groups)"""
    bl_idname = "rigger.unbind"
    bl_label = "Unbind Meshes"
    bl_options = {'REGISTER', 'UNDO'}

    keep_weights: BoolProperty(name="Keep Weights", default=False)

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            rig = utils.find_rig_for(obj)
            for mod in list(obj.modifiers):
                if mod.type == 'ARMATURE' and (rig is None or mod.object == rig):
                    obj.modifiers.remove(mod)
            if VOLUME_GROUP in obj.vertex_groups and not any(m.type == 'ARMATURE' for m in obj.modifiers):
                obj.vertex_groups.remove(obj.vertex_groups[VOLUME_GROUP])
            if rig is not None:
                if not self.keep_weights:
                    for bone in rig.data.bones:
                        vg = obj.vertex_groups.get(bone.name)
                        if vg is not None:
                            obj.vertex_groups.remove(vg)
                if obj.parent == rig:
                    mw = obj.matrix_world.copy()
                    obj.parent = None
                    obj.matrix_world = mw
        return {'FINISHED'}


classes = (RIGGER_OT_bind, RIGGER_OT_eyelid_border, RIGGER_OT_unbind)
