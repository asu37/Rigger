# SPDX-License-Identifier: GPL-3.0-or-later
"""Resize control shapes to the skinned mesh, so rings hug the body."""

import bpy
import numpy as np

from . import utils


def _skinned_meshes(context, rig):
    return [o for o in context.scene.objects if o.type == 'MESH' and utils.find_rig_for(o) == rig]


def _vertex_weights(obj, bone_names):
    """World positions (N,3) and weight per requested bone {name: (N,) array}."""
    from .skin import world_vertices
    points = world_vertices(obj)
    index = {vg.index: vg.name for vg in obj.vertex_groups if vg.name in bone_names}
    weights = {name: np.zeros(len(points)) for name in bone_names}
    if index:
        for v in obj.data.vertices:
            for g in v.groups:
                name = index.get(g.group)
                if name is not None:
                    weights[name][v.index] = g.weight
    return points, weights


def fit_shapes(context, rig, meshes=None):
    """Scale every fittable control so its shape wraps the mesh around it.

    Returns the number of controls resized.
    """
    meta = utils.get_meta(rig)
    fits = meta.get("fit", {})
    if not fits:
        return 0
    meshes = meshes if meshes is not None else _skinned_meshes(context, rig)
    needed = {b for spec in fits.values() for b in spec["bones"]}
    data = [_vertex_weights(obj, needed) for obj in meshes]
    if not data:
        return 0
    mw = rig.matrix_world
    zs = [v.z for b in rig.data.bones for v in (b.head_local, b.tail_local)]
    height = max(zs) - min(zs)
    count = 0
    for ctrl, spec in fits.items():
        pb = rig.pose.bones.get(ctrl)
        if pb is None or pb.custom_shape is None:
            continue
        bone = rig.data.bones[ctrl]
        head = np.array(mw @ bone.head_local)
        tail = np.array(mw @ bone.tail_local)
        axis = tail - head
        length = np.linalg.norm(axis)
        if length < 1e-8:
            continue
        axis /= length
        center = head + (tail - head) * spec["y"]
        window = max(0.15 * length, 0.012 * height * mw.to_scale().z)
        dists = []
        for points, weights in data:
            w = sum(weights[b] for b in spec["bones"] if b in weights)
            rel = points - center
            along = rel @ axis
            mask = (w >= 0.35) & (np.abs(along) <= window)
            if not mask.any():
                continue
            perp = rel[mask] - np.outer(along[mask], axis)
            dists.append(np.linalg.norm(perp, axis=1))
        if not dists:
            continue
        radius = float(np.percentile(np.concatenate(dists), 85))
        if radius <= 1e-6:
            continue
        scale = radius * spec["margin"] / length
        _x, y_scale, _z = pb.custom_shape_scale_xyz
        pb.custom_shape_scale_xyz = (scale, y_scale, scale)
        count += 1
    return count


class RIGGER_OT_fit_shapes(bpy.types.Operator):
    """Resize the control shapes to the skinned mesh so they wrap around the body"""
    bl_idname = "rigger.fit_shapes"
    bl_label = "Fit Shapes to Mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def execute(self, context):
        rig = utils.find_rig_for(context.object)
        count = fit_shapes(context, rig)
        if not count:
            self.report({'WARNING'}, "No skinned mesh found: bind a mesh first")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Fitted {count} control shapes")
        return {'FINISHED'}


classes = (RIGGER_OT_fit_shapes,)
