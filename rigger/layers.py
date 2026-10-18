# SPDX-License-Identifier: GPL-3.0-or-later
"""Rig layers: named groups of bone collections and objects (clothes, hair, props)
that are shown or hidden together."""

import json

import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import utils


def _layer_rig(layer):
    return layer.id_data


def apply_layer(layer):
    """Show or hide everything in a layer."""
    rig = _layer_rig(layer)
    for item in layer.objects:
        if item.obj is not None:
            item.obj.hide_viewport = not layer.show
            item.obj.hide_render = not layer.show
    if rig is not None and rig.type == 'ARMATURE':
        for name in filter(None, (n.strip() for n in layer.bone_collections.split(","))):
            coll = rig.data.collections.get(name)
            if coll is not None:
                coll.is_visible = layer.show


def _rig(context):
    return utils.find_rig_for(context.object)


class RIGGER_OT_layer_add(bpy.types.Operator):
    """Add a rig layer"""
    bl_idname = "rigger.layer_add"
    bl_label = "Add Layer"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(name="Name", default="Layer")

    @classmethod
    def poll(cls, context):
        return _rig(context) is not None

    def execute(self, context):
        rig = _rig(context)
        layer = rig.rigger.layers.add()
        layer.name = self.name
        rig.rigger.layer_index = len(rig.rigger.layers) - 1
        return {'FINISHED'}


class RIGGER_OT_layer_remove(bpy.types.Operator):
    """Remove the active rig layer (its objects and bones are shown again)"""
    bl_idname = "rigger.layer_remove"
    bl_label = "Remove Layer"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        rig = _rig(context)
        return rig is not None and len(rig.rigger.layers) > 0

    def execute(self, context):
        settings = _rig(context).rigger
        index = settings.layer_index
        if 0 <= index < len(settings.layers):
            layer = settings.layers[index]
            layer.show = True
            settings.layers.remove(index)
            settings.layer_index = max(0, index - 1)
        return {'FINISHED'}


class RIGGER_OT_layer_assign(bpy.types.Operator):
    """Put the selected objects (Object Mode) or the selected bones (Pose Mode) in the active layer"""
    bl_idname = "rigger.layer_assign"
    bl_label = "Assign to Layer"
    bl_options = {'REGISTER', 'UNDO'}

    remove: BoolProperty(name="Remove", default=False, description="Take them out of the layer instead")

    @classmethod
    def poll(cls, context):
        rig = _rig(context)
        return rig is not None and 0 <= rig.rigger.layer_index < len(rig.rigger.layers)

    def execute(self, context):
        rig = _rig(context)
        layer = rig.rigger.layers[rig.rigger.layer_index]
        if context.object.mode == 'POSE' and context.object == rig:
            coll_name = f"Layer: {layer.name}"
            coll = rig.data.collections.get(coll_name) or rig.data.collections.new(coll_name)
            for pb in rig.pose.bones:
                if utils.is_pose_bone_selected(pb):
                    if self.remove:
                        coll.unassign(pb.bone)
                    else:
                        coll.assign(pb.bone)
            names = [n for n in layer.bone_collections.split(",") if n]
            if coll_name not in names:
                names.append(coll_name)
            layer.bone_collections = ",".join(names)
        else:
            current = {item.obj for item in layer.objects}
            for obj in context.selected_objects:
                if obj == rig:
                    continue
                if self.remove:
                    for i, item in enumerate(layer.objects):
                        if item.obj == obj:
                            layer.objects.remove(i)
                            break
                elif obj not in current:
                    layer.objects.add().obj = obj
        return {'FINISHED'}


def layer_bones(rig, layer):
    names = set()
    for coll_name in filter(None, (n.strip() for n in layer.bone_collections.split(","))):
        coll = rig.data.collections.get(coll_name)
        if coll is not None:
            names.update(b.name for b in coll.bones)
    return names


class RIGGER_OT_layer_select(bpy.types.Operator):
    """Select the bones of the active rig layer (Shift: add to the selection)"""
    bl_idname = "rigger.layer_select"
    bl_label = "Select Layer Bones"
    bl_options = {'REGISTER', 'UNDO'}

    extend: BoolProperty(name="Extend", default=False, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        rig = _rig(context)
        return (rig is not None and context.object == rig and rig.mode == 'POSE'
                and 0 <= rig.rigger.layer_index < len(rig.rigger.layers))

    def invoke(self, context, event):
        self.extend = event.shift
        return self.execute(context)

    def execute(self, context):
        rig = _rig(context)
        names = layer_bones(rig, rig.rigger.layers[rig.rigger.layer_index])
        for pb in rig.pose.bones:
            if pb.name in names:
                utils.select_pose_bone(pb, True)
            elif not self.extend:
                utils.select_pose_bone(pb, False)
        return {'FINISHED'}


class RIGGER_OT_layers_export(ExportHelper, bpy.types.Operator):
    """Save the rig layers to a file"""
    bl_idname = "rigger.layers_export"
    bl_label = "Export Rig Layers"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _rig(context) is not None

    def execute(self, context):
        from .rig_data import layers_data
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "layers": layers_data(_rig(context))}, f, indent=1)
        return {'FINISHED'}


class RIGGER_OT_layers_import(ImportHelper, bpy.types.Operator):
    """Load rig layers from a file (objects are matched by name)"""
    bl_idname = "rigger.layers_import"
    bl_label = "Import Rig Layers"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _rig(context) is not None

    def execute(self, context):
        from .rig_data import apply_layers
        with open(self.filepath, encoding="utf-8") as f:
            apply_layers(_rig(context), json.load(f)["layers"])
        return {'FINISHED'}


@persistent
def animated_layers_handler(scene, _depsgraph=None):
    """Apply keyframed layer visibility during playback and scrubbing."""
    for obj in scene.objects:
        if obj.type != 'ARMATURE' or not obj.rigger.animated_layers:
            continue
        for layer in obj.rigger.layers:
            apply_layer(layer)


def register_handlers():
    if animated_layers_handler not in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.append(animated_layers_handler)


def unregister_handlers():
    if animated_layers_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(animated_layers_handler)


class RIGGER_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "name", text="", emboss=False)
        count = len(item.objects)
        if count:
            row.label(text=str(count), icon='OBJECT_DATA')
        row.prop(item, "show", text="", emboss=False, icon='HIDE_OFF' if item.show else 'HIDE_ON')


classes = (RIGGER_OT_layer_add, RIGGER_OT_layer_remove, RIGGER_OT_layer_assign, RIGGER_OT_layer_select,
           RIGGER_OT_layers_export, RIGGER_OT_layers_import, RIGGER_UL_layers)
