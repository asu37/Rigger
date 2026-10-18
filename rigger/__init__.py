# SPDX-License-Identifier: GPL-3.0-or-later
#
# Rigger - modular auto-rigging for Blender
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version. See the LICENSE file for details.

bl_info = {
    "name": "Rigger",
    "author": "Rigger contributors",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Rigger",
    "description": "Modular auto-rigging: reference, generate, skin, retarget and game export",
    "category": "Rigging",
}

if "bpy" in locals():
    import importlib
    for _mod in (safety, utils, widgets, templates, properties, generate, reference, markers, ai_markers, skin,
                 shapes, layers, tools, rig_tools, poses, rig_data, retarget, quick_rig, export, picker, ui, mocap_ops):
        importlib.reload(_mod)

import bpy  # noqa: E402

from . import (ai_markers, export, generate, markers, picker, poses, properties, quick_rig, reference, retarget,  # noqa: E402
               layers, rig_data, rig_tools, safety, shapes, skin, templates, tools, ui, utils, widgets)
from .mocap import ops as mocap_ops  # noqa: E402

_CLASS_MODULES = (generate, reference, markers, ai_markers, skin, shapes, layers, tools, rig_tools, poses, rig_data,
                  retarget, quick_rig, export, picker, ui, mocap_ops)


def _menu_export(self, _context):
    self.layout.operator("rigger.export_fbx", text="Rigger Game Rig (.fbx, .glb)")


def register():
    properties.register()
    for module in _CLASS_MODULES:
        # every operator gets the safety net: readable errors, scene state restored
        safety.guard_operators(module.classes)
        for cls in module.classes:
            bpy.utils.register_class(cls)
    mocap_ops.register_props()
    bpy.types.TOPBAR_MT_file_export.append(_menu_export)
    layers.register_handlers()


def unregister():
    picker.stop()
    reference.unregister_overlay()
    retarget.preview_stop()
    layers.unregister_handlers()
    bpy.types.TOPBAR_MT_file_export.remove(_menu_export)
    mocap_ops.unregister_props()
    for module in reversed(_CLASS_MODULES):
        for cls in reversed(module.classes):
            bpy.utils.unregister_class(cls)
    properties.unregister()
