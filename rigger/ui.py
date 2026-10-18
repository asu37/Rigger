# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar panels (View3D > Sidebar > Rigger)."""

import bpy
from bpy.props import StringProperty

from . import templates, utils
from .modules import MODULES

# Module options shown for the active reference bone: (title, [(property, condition)])
_LIMB = [
    ("Limb", ["twist", "twist_upper", "twist_factor", "start_ik", "pole_distance", "auto_ik_roll",
              "fk_lock", "parent_fallback"]),
    ("IK", ["soft_ik", "soft_ik_amount", "ik_offset", "ik_pivot", "smart_pole"]),
    ("Secondary & Fans", ["secondary", "secondary_type", "bbone_segments", "bbone_scale", "fans", "fans_root",
                          "fans_end", "fan_count", "fan_bulge", "fan_bulge_mode"]),
    ("Digits", ["rot_from_scale", "auto_spread", "pinky_auto"]),
]
MODULE_OPTIONS = {
    'ARM': _LIMB + [("Arm", ["wings_fold"])],
    'LEG': _LIMB + [("Foot", ["roll_break", "toes_pivot", "three_bones"])],
    'SPINE': [("Spine", ["spine_master", "spine_master_space", "bottom", "reversed_spine", "parent_fallback"])],
    'NECK': [("Neck", ["neck_twist", "neck_bendy", "skulls", "parent_fallback"])],
    'FINGER': [("Finger", ["curl", "finger_ik", "finger_pole_distance", "finger_shape", "collection"])],
    'CHAIN': [("Chain", ["curl", "collection", "parent_fallback"])],
    'BBONE': [("Bendy Bones", ["bbone_segments", "bbone_scale", "collection"])],
    'SPLINE': [("Spline IK", ["spline_controls", "spline_stretch", "spline_smooth", "spline_fk", "spline_twist",
                              "spline_tail"])],
    'SKIRT': [("Skirt", ["skirt_collide", "skirt_distance", "skirt_masters", "skirt_master_freq",
                         "skirt_row_masters"])],
    'FACE': [("Eyes & Lids", ["lid_masters"]),
             ("Brows & Cheeks", ["brow_master", "soft_brows", "soft_cheeks"]),
             ("Lips", ["lips_offset", "muzzle", "sticky_lips", "lips_roll", "soft_lips", "soft_lips_linear",
                       "lips_masters_freq"]),
             ("Jaw", ["jaw_translate", "jaw_unlock_y"])],
    'EYE': [("Eye", ["eye_distance"])],
    'SIMPLE': [("Bone", ["collection", "parent_fallback"])],
}


def _section(layout, title):
    box = layout.box()
    box.label(text=title)
    return box.column(align=True)


def draw_module_settings(layout, settings):
    for title, props in MODULE_OPTIONS.get(settings.module, []):
        col = _section(layout, title)
        for prop in props:
            if hasattr(settings, prop):
                col.prop(settings, prop)


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

class RIGGER_UL_remap(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        s = context.scene.rigger_remap
        row = layout.row(align=True)
        if s.source:
            row.prop_search(item, "source", s.source.data, "bones", text="")
        else:
            row.prop(item, "source", text="")
        row.label(icon='FORWARD')
        if s.target:
            row.prop_search(item, "target", s.target.data, "bones", text="")
        else:
            row.prop(item, "target", text="")
        row.prop(item, "location", text="", icon='ORIENTATION_GLOBAL' if item.location else 'ORIENTATION_LOCAL')


class _ActionFilter:
    """Lists bpy.data.actions that animate a given armature."""

    def armature(self, context):
        return None

    def draw_filter(self, context, layout):
        layout.prop(self, "filter_name", text="")

    def filter_items(self, context, data, propname):
        from .export import _action_fits
        actions = getattr(data, propname)
        armature = self.armature(context)
        text = self.filter_name.lower()
        flags = []
        for action in actions:
            ok = armature is not None and _action_fits(action, armature) and text in action.name.lower()
            flags.append(self.bitflag_filter_item if ok else 0)
        return flags, []


class RIGGER_UL_export_actions(_ActionFilter, bpy.types.UIList):
    def armature(self, context):
        return utils.find_rig_for(context.object)

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "rigger_export", text="")
        row.label(text=item.name, icon='ACTION')


class RIGGER_UL_remap_actions(_ActionFilter, bpy.types.UIList):
    def armature(self, context):
        return context.scene.rigger_remap.source

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "rigger_remap", text="")
        row.label(text=item.name, icon='ACTION')


class RIGGER_UL_action_lists(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "name", text="", emboss=False)
        row.label(text=str(len(item.actions)))


class RIGGER_UL_action_items(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        layout.prop(item, "action", text="")


class RIGGER_UL_links(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"{item.rig_a.name if item.rig_a else '?'}: {item.action_a.name if item.action_a else '?'}")
        row.label(icon='LINKED')
        row.label(text=f"{item.rig_b.name if item.rig_b else '?'}: {item.action_b.name if item.action_b else '?'}")


class RIGGER_UL_quick_limbs(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.label(text=f"{item.kind.title()} {item.side if item.side != 'C' else ''} {item.label}".strip())
        row.label(text=f"{len([b for b in item.bones.split(',') if b.strip()])} bones")


class RIGGER_UL_quick_overrides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "source", text="")
        row.label(icon='FORWARD')
        row.prop(item, "target", text="")


# ---------------------------------------------------------------------------
# Small operators used by the panels
# ---------------------------------------------------------------------------

class RIGGER_OT_pin_property(bpy.types.Operator):
    """Pin or unpin a bone property: pinned properties are always shown in the Animate panel"""
    bl_idname = "rigger.pin_property"
    bl_label = "Pin Property"
    bl_options = {'REGISTER', 'UNDO'}

    bone: StringProperty()
    prop: StringProperty()

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        pins = pinned(rig)
        key = f"{self.bone}:{self.prop}"
        if key in pins:
            pins.remove(key)
        else:
            pins.append(key)
        rig.rigger.pinned_props = ",".join(pins)
        return {'FINISHED'}


def pinned(rig):
    return [p.strip() for p in rig.rigger.pinned_props.split(",") if p.strip()]


def custom_props(pb):
    """Animatable custom properties of a pose bone."""
    return [k for k in pb.keys() if not k.startswith("_") and isinstance(pb[k], (int, float))]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Rigger"


def _setup_poll(context):
    # Setup tools stay out of the way while animating
    obj = context.object
    return not (utils.is_rig(obj) and obj.mode == 'POSE')


class RIGGER_PT_main(_Panel, bpy.types.Panel):
    bl_label = "Rigger"

    def draw(self, context):
        layout = self.layout
        obj = context.object
        rig = utils.find_rig_for(obj)
        col = layout.column(align=True)
        if utils.is_reference(obj):
            col.label(text=f"Reference: {obj.name}", icon='ARMATURE_DATA')
            if obj.rigger.rig:
                col.label(text=f"Rig: {obj.rigger.rig.name}", icon='OUTLINER_OB_ARMATURE')
        elif rig:
            col.label(text=f"Rig: {rig.name}", icon='OUTLINER_OB_ARMATURE')
        else:
            col.label(text="1. Add a reference or place markers", icon='INFO')


class RIGGER_PT_reference(_Panel, bpy.types.Panel):
    bl_label = "1. Reference"
    bl_parent_id = "RIGGER_PT_main"

    @classmethod
    def poll(cls, context):
        return _setup_poll(context)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        col = layout.column(align=True)
        col.label(text="New reference from template:")
        for key, (label, _desc, _fn) in templates.TEMPLATES.items():
            col.operator("rigger.add_reference", text=label, icon='ARMATURE_DATA').template = key
        row = col.row(align=True)
        row.operator_menu_enum("rigger.add_preset", "preset", text="From Preset", icon='PRESET')
        row.operator_menu_enum("rigger.remove_preset", "preset", text="", icon='TRASH')
        if utils.is_reference(obj):
            col.operator("rigger.save_preset", icon='FILE_TICK')

        box = layout.box()
        box.label(text="Smart Markers (front view)", icon='PIVOT_CURSOR')
        row = box.row(align=True)
        row.operator("rigger.turn_model", text="Turn -90").angle = -90.0
        row.operator("rigger.turn_model", text="Turn +90").angle = 90.0
        grid = box.grid_flow(columns=2, align=True)
        markers = {k: bpy.data.objects.get("RGM-" + k) for k, *_ in templates.MARKERS}
        for key, label, _desc, _sym in templates.MARKERS:
            op = grid.operator("rigger.place_marker", text=label,
                               icon='CHECKMARK' if markers[key] else 'DOT')
            op.marker = key
        from . import ai_markers
        ready, _message = ai_markers.status()
        row = box.row(align=True)
        sub = row.row(align=True)
        sub.enabled = ready
        sub.operator("rigger.guess_markers", icon='SHADERFX')
        if not ready:
            row.operator("rigger.install_ai", text="", icon='IMPORT')
        row = box.row(align=True)
        row.operator("rigger.fit_markers", icon='MOD_ARMATURE')
        row.operator("rigger.clear_markers", text="", icon='TRASH')
        if utils.is_reference(obj):
            box.operator("rigger.detect_fingers", icon='VIEW_PAN')

        if not utils.is_reference(obj):
            return
        box = layout.box()
        box.label(text="Modules", icon='GROUP_BONE')
        row = box.row(align=True)
        row.operator_menu_enum("rigger.add_module", "module", text="Add Module", icon='ADD')
        row = box.row(align=True)
        row.operator("rigger.mirror_reference", icon='MOD_MIRROR')
        row.operator("rigger.show_ik_lines", text="", icon='CON_KINEMATIC', depress=_ik_lines_shown())
        row = box.row(align=True)
        row.operator("rigger.duplicate_limb", icon='DUPLICATE').mirror = False
        row.operator("rigger.duplicate_limb", text="Mirror", icon='MOD_MIRROR').mirror = True
        row.operator("rigger.remove_limb", text="", icon='X')

        bone_name = None
        if obj.mode == 'EDIT' and obj.data.edit_bones.active:
            bone_name = obj.data.edit_bones.active.name
        elif obj.data.bones.active:
            bone_name = obj.data.bones.active.name
        if bone_name and bone_name in obj.pose.bones:
            settings = obj.pose.bones[bone_name].rigger
            sub = box.column()
            sub.label(text=f"Bone: {bone_name}")
            sub.prop(settings, "module")
            module = MODULES.get(settings.module)
            if module is not None:
                sub.label(text=module.description, icon='INFO')
                if settings.module == 'ARM':
                    sub.operator("rigger.add_feathers", icon='MOD_PARTICLES')
                if settings.module == 'FACE':
                    sub.label(text="Child names set behaviour: jaw, lip_*, lid_*, tongue…")
                draw_module_settings(sub, settings)


def _ik_lines_shown():
    from .reference import ik_lines_shown
    return ik_lines_shown()


class RIGGER_PT_template_options(_Panel, bpy.types.Panel):
    bl_label = "Template Options"
    bl_parent_id = "RIGGER_PT_reference"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        t = context.scene.rigger_template
        col = self.layout.column(align=True)
        for prop in ("spine_count", "neck_count", "twist", "fingers", "metacarpals", "toes", "toes_metatarsal",
                     "toes_parent_foot"):
            col.prop(t, prop)
        col = _section(self.layout, "Face")
        for prop in ("lips", "lids", "brows"):
            col.prop(t, prop)
        grid = col.grid_flow(columns=2, align=True)
        for prop in ("eyes", "eyebrows", "nose", "mouth", "tongue", "teeth", "chin", "cheeks", "ears"):
            grid.prop(t, prop, toggle=True)


class RIGGER_PT_quick_rig(_Panel, bpy.types.Panel):
    bl_label = "Quick Rig"
    bl_parent_id = "RIGGER_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return _setup_poll(context)

    def draw(self, context):
        layout = self.layout
        s = context.scene.rigger_quick
        col = layout.column()
        col.prop(s, "armature")
        col.row().prop(s, "mode", expand=True)
        row = col.row(align=True)
        row.operator("rigger.quick_rig_guess", icon='AUTO')
        row.operator("rigger.quick_rig_preset_import", text="", icon='FILE_FOLDER')
        row.operator("rigger.quick_rig_preset_export", text="", icon='FILE_TICK')
        row = layout.row()
        row.template_list("RIGGER_UL_quick_limbs", "", s, "limbs", s, "active_index", rows=5)
        side = row.column(align=True)
        side.operator("rigger.quick_rig_limb_add", text="", icon='ADD')
        side.operator("rigger.quick_rig_limb_remove", text="", icon='REMOVE')
        if 0 <= s.active_index < len(s.limbs):
            limb = s.limbs[s.active_index]
            box = layout.box()
            row = box.row(align=True)
            row.prop(limb, "kind", text="")
            row.prop(limb, "side", text="")
            box.prop(limb, "label")
            box.prop(limb, "bones", text="")
            if limb.kind == 'FINGER':
                box.prop(limb, "metacarpal")
            box.operator("rigger.quick_rig_from_selection", icon='RESTRICT_SELECT_OFF')
        col = _section(layout, "Options")
        for prop in ("orphans", "ignore_root", "same_origin", "bake", "twist", "secondary", "fingers_ik"):
            col.prop(s, prop)
        col.prop(s, "hide_base" if s.mode == 'PRESERVE' else "delete_source")
        if s.mode == 'CONVERT':
            col = _section(layout, "Weights Overrides")
            row = col.row()
            row.template_list("RIGGER_UL_quick_overrides", "", s, "overrides", s, "override_index", rows=2)
            side = row.column(align=True)
            side.operator("rigger.quick_rig_override_add", text="", icon='ADD')
            side.operator("rigger.quick_rig_override_remove", text="", icon='REMOVE')
        row = layout.row()
        row.scale_y = 1.4
        row.operator("rigger.quick_rig", icon='OUTLINER_OB_ARMATURE')


class RIGGER_PT_generate(_Panel, bpy.types.Panel):
    bl_label = "2. Generate"
    bl_parent_id = "RIGGER_PT_main"

    @classmethod
    def poll(cls, context):
        return _setup_poll(context)

    def draw(self, context):
        layout = self.layout
        obj = context.object
        rig = utils.find_rig_for(obj)
        ref = obj if utils.is_reference(obj) else (rig.rigger.reference if rig else None)
        if ref is None:
            layout.label(text="Select a reference armature")
            return
        col = layout.column()
        col.prop(ref.rigger, "rig_name")
        col.prop(ref.rigger, "widget_scale")
        col.prop(ref.rigger, "auto_hide")
        col.prop(ref.rigger, "init_scale")
        col.prop(ref.rigger, "multiped")
        box = col.box()
        box.prop(ref.rigger, "use_theme")
        if ref.rigger.use_theme:
            row = box.row(align=True)
            row.prop(ref.rigger, "color_right", text="")
            row.prop(ref.rigger, "color_center", text="")
            row.prop(ref.rigger, "color_left", text="")
            if rig:
                row = box.row(align=True)
                row.operator("rigger.color_theme", text="Assign").reset = False
                row.operator("rigger.color_theme", text="Reset").reset = True
        row = col.row()
        row.scale_y = 1.4
        row.operator("rigger.generate", text="Update Rig" if ref.rigger.rig else "Generate Rig",
                     icon='OUTLINER_OB_ARMATURE')
        if utils.is_rig(obj):
            col.operator("rigger.edit_reference", icon='EDITMODE_HLT')
        row = col.row(align=True)
        row.operator("rigger.export_rig_data", text="Export Data", icon='EXPORT')
        row.operator("rigger.import_rig_data", text="Import Data", icon='IMPORT')
        if rig:
            col.operator("rigger.set_character_name", icon='SORTALPHA')


class RIGGER_PT_skin(_Panel, bpy.types.Panel):
    bl_label = "3. Skin"
    bl_parent_id = "RIGGER_PT_main"

    @classmethod
    def poll(cls, context):
        return _setup_poll(context)

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Select meshes, then the rig")
        col.operator_context = 'INVOKE_DEFAULT'
        for method, text, icon in (('AUTO', "Bind (Heat + Fallback)", 'MOD_ARMATURE'),
                                   ('VOXEL', "Bind (Voxelized)", 'MOD_REMESH'),
                                   ('HEAT', "Bind (Heat)", 'OUTLINER_OB_FORCE_FIELD'),
                                   ('PROXIMITY', "Bind (Proximity)", 'MOD_VERTEX_WEIGHT')):
            col.operator("rigger.bind", text=text, icon=icon).method = method
        col.operator_context = 'EXEC_DEFAULT'
        col.operator("rigger.unbind", icon='X')
        obj = context.object
        if obj is not None and obj.type == 'MESH':
            box = layout.box()
            box.label(text="Eyelid Borders (Edit Mode selection)")
            row = box.row(align=True)
            row.operator("rigger.eyelid_border", text="Set Left").side = 'L'
            row.operator("rigger.eyelid_border", text="Set Right").side = 'R'
            row.operator("rigger.eyelid_border", text="", icon='X').side = 'CLEAR'
        col = layout.column(align=True)
        col.operator("rigger.fit_shapes", icon='FULLSCREEN_ENTER')


class RIGGER_MT_space(bpy.types.Menu):
    bl_label = "Parent Space"

    def draw(self, context):
        pb = getattr(context, "rigger_space_bone", None)
        if pb is None:
            return
        labels = utils.get_meta(pb.id_data).get("spaces", {}).get(pb.name, [])
        for i, label in enumerate(labels):
            op = self.layout.operator("rigger.switch_space", text=label,
                                      icon='RADIOBUT_ON' if int(pb.get("parent", 0)) == i else 'RADIOBUT_OFF')
            op.bone, op.space = pb.name, i


def _space_row(layout, rig, bone, text):
    labels = utils.get_meta(rig).get("spaces", {}).get(bone)
    pb = rig.pose.bones.get(bone)
    if not labels or pb is None or "parent" not in pb:
        return
    row = layout.row(align=True)
    row.label(text=text)
    row.context_pointer_set("rigger_space_bone", pb)
    row.menu("RIGGER_MT_space", text=labels[min(int(pb["parent"]), len(labels) - 1)], icon='CON_CHILDOF')


def _limb_of(rig, bone_name):
    for index, limb in enumerate(utils.get_meta(rig).get("limbs", [])):
        names = set(limb["fk"]) | set(limb["ik"]) | set(limb.get("tweaks") or [])
        names |= {limb.get(k) for k in ("ik_ctrl", "pole", "foot_roll", "toe_fk", "toe_ik", "shoulder",
                                         "prop_bone", "ik_offset", "ik_pivot", "toes_pivot", "ball")}
        if bone_name in names:
            return index, limb
    return None, None


def _prop_row(layout, pb, key, text=None, pin=True):
    row = layout.row(align=True)
    row.prop(pb, f'["{key}"]', text=text or key.replace("_", " ").title(), slider=True)
    if pin:
        rig = pb.id_data
        op = row.operator("rigger.pin_property", text="",
                          icon='PINNED' if f"{pb.name}:{key}" in pinned(rig) else 'UNPINNED', emboss=False)
        op.bone, op.prop = pb.name, key


class RIGGER_PT_controls(_Panel, bpy.types.Panel):
    bl_label = "4. Animate"
    bl_parent_id = "RIGGER_PT_main"

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object)

    def draw(self, context):
        layout = self.layout
        rig = context.object
        meta = utils.get_meta(rig)
        pbs = rig.pose.bones
        active = context.active_pose_bone if rig.mode == 'POSE' else None

        # Pinned properties
        pins = [p.split(":", 1) for p in pinned(rig)]
        pins = [(b, k) for b, k in pins if b in pbs and k in pbs[b]]
        if pins:
            col = _section(layout, "Pinned")
            for bone, key in pins:
                _prop_row(col, pbs[bone], key, f"{bone}: {key}")

        # Settings of the limb the active control belongs to
        index, limb = _limb_of(rig, active.name) if active else (None, None)
        box = layout.box()
        if limb is None:
            box.label(text="Select an arm or leg control", icon='INFO')
        else:
            prop = pbs[limb["prop_bone"]]
            is_ik = prop.get("ik_fk", 0.0) > 0.5
            box.label(text=f"{limb.get('label', 'Limb')} ({'IK' if is_ik else 'FK'})", icon='CON_KINEMATIC')
            row = box.row(align=True)
            row.scale_y = 1.3
            op = row.operator("rigger.snap_ik_to_fk" if not is_ik else "rigger.snap_fk_to_ik",
                              text="Switch to IK" if not is_ik else "Switch to FK", icon='ARROW_LEFTRIGHT')
            op.limb = index
            row = box.row(align=True)
            op = row.operator("rigger.bake_ikfk", text="Bake to IK")
            op.limb, op.to_ik = index, True
            op = row.operator("rigger.bake_ikfk", text="Bake to FK")
            op.limb, op.to_ik = index, False
            col = box.column(align=True)
            col.prop(prop, '["ik_fk"]', text="IK-FK Blend", slider=True)
            if is_ik:
                for key in ("stretch_length", "stretch", "soft_ik", "three_ik", "roll_speed", "roll_break",
                            "roll_overshoot", "roll_unfold", "pole_follow"):
                    if key in prop:
                        _prop_row(col, prop, key)
                col = box.column(align=True)
                _space_row(col, rig, limb["ik_ctrl"], "IK Parent")
                _space_row(col, rig, limb["pole"], "Pole Parent")
                col = box.column(align=True)
                if "pin" in prop:
                    _prop_row(col, prop, "pin", "Pinning")
                op = col.operator("rigger.snap_pin", text="Snap Pinning", icon='PINNED')
                op.limb = index
            fk_lock = limb.get("fk_lock")
            if fk_lock and fk_lock in pbs and "fk_lock" in pbs[fk_lock]:
                _prop_row(box.column(), pbs[fk_lock], "fk_lock", "FK Lock")

        spine = meta.get("spine") or {}
        torso = pbs.get(spine.get("torso") or "")
        if torso is not None and "spine_reverse" in torso:
            col = _section(layout, "Spine")
            _prop_row(col, torso, "spine_reverse", "Reversed Spine")
            col.operator("rigger.snap_reverse_spine", icon='ARROW_LEFTRIGHT')

        neck = meta.get("neck")
        face = (meta.get("faces") or [None])[0]
        if (neck and neck["head"] in pbs) or face:
            col = _section(layout, "Head & Face")
            if neck and neck["head"] in pbs:
                for key in custom_props(pbs[neck["head"]]):
                    _prop_row(col, pbs[neck["head"]], key)
            if face and face["face"] in pbs:
                for key in custom_props(pbs[face["face"]]):
                    _prop_row(col, pbs[face["face"]], key)

        col = _section(layout, "Fingers")
        row = col.row(align=True)
        for side in ("R", "L"):
            op = row.operator("rigger.finger_pose", text=f"Fist {side}")
            op.side, op.pose = side, 'FIST'
        op = row.operator("rigger.finger_pose", text="Open")
        op.side, op.pose = 'BOTH', 'OPEN'
        if any(f.get("ik_ctrl") for f in meta.get("fingers", [])):
            row = col.row(align=True)
            for side in ("R", "L"):
                op = row.operator("rigger.snap_fingers", text=f"IK {side}")
                op.side, op.to_ik, op.finger = side, True, ""
                op = row.operator("rigger.snap_fingers", text=f"FK {side}")
                op.side, op.to_ik, op.finger = side, False, ""
            row = col.row(align=True)
            for side in ("R", "L"):
                row.operator("rigger.toggle_finger_parents", text=f"IK Parents {side}").side = side
        for spline in meta.get("splines", []):
            if spline.get("fk"):
                col.operator("rigger.snap_spline", text=f"Snap {spline.get('master', 'Spline')}").master = \
                    spline.get("master") or ""

        # Every animatable property of the active bone
        if active is not None:
            keys = custom_props(active)
            if keys or "parent" in active:
                col = _section(layout, f"Active: {active.name}")
                _space_row(col, rig, active.name, "Parent")
                for key in keys:
                    if key != "parent":
                        _prop_row(col, active, key)
                if keys:
                    op = col.operator("rigger.snap_property", text="Snap Switch", icon='SNAP_ON')
                    op.bone, op.prop = active.name, keys[0]

        box = layout.box()
        box.label(text="Rig Layers", icon='RENDERLAYERS')
        row = box.row()
        row.template_list("RIGGER_UL_layers", "", rig.rigger, "layers", rig.rigger, "layer_index", rows=3)
        side = row.column(align=True)
        side.operator("rigger.layer_add", text="", icon='ADD')
        side.operator("rigger.layer_remove", text="", icon='REMOVE')
        side.separator()
        side.operator("rigger.layer_assign", text="", icon='IMPORT')
        side.operator("rigger.layer_select", text="", icon='RESTRICT_SELECT_OFF')
        row = box.row(align=True)
        row.prop(rig.rigger, "animated_layers")
        row.operator("rigger.layers_export", text="", icon='EXPORT')
        row.operator("rigger.layers_import", text="", icon='IMPORT')
        sub = box.column(align=True)
        sub.label(text="Bone collections:")
        grid = sub.grid_flow(columns=2, align=True)
        for coll in rig.data.collections:
            if coll.name.startswith("Layer: "):
                continue
            grid.prop(coll, "is_visible", text=coll.name, toggle=True)

        col = layout.column(align=True)
        col.operator("rigger.reset_pose", icon='LOOP_BACK')


class RIGGER_PT_tools(_Panel, bpy.types.Panel):
    bl_label = "Rig Tools"
    bl_parent_id = "RIGGER_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_rig(obj) or (obj is not None and obj.name.startswith("SHAPE-EDIT-"))

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        if context.object.name.startswith("SHAPE-EDIT-"):
            col.scale_y = 1.4
            col.operator("rigger.apply_shape", icon='CHECKMARK')
            return
        rig = context.object

        col = _section(layout, "Pose")
        row = col.row(align=True)
        for pose, text in (('A', "A-Pose"), ('A_UE5', "A UE5"), ('T', "T-Pose")):
            row.operator("rigger.set_pose", text=text).pose = pose
        col.operator("rigger.mirror_pose", icon='MOD_MIRROR')
        col.operator("rigger.apply_rest_pose", icon='ARMATURE_DATA')
        col.operator_menu_enum("rigger.convert_rotation", "mode", text="Convert Rotation", icon='ORIENTATION_GIMBAL')
        col.operator("rigger.extract_root_motion", icon='ANIM')
        col.operator("rigger.snap_child_of", icon='CON_CHILDOF')

        col = _section(layout, "Pose Blends")
        row = col.row(align=True)
        row.operator("rigger.add_pose_blend", text="Fist").kind = 'FIST'
        row.operator("rigger.add_pose_blend", text="Extend").kind = 'EXTEND'
        row.operator("rigger.add_pose_blend", text="Blink").kind = 'BLINK'
        row = col.row(align=True)
        row.operator("rigger.default_blink_pose", text="Default Blink L").side = 'L'
        row.operator("rigger.default_blink_pose", text="R").side = 'R'
        from .poses import load_blends
        for blend in load_blends(rig.rigger.reference):
            row = col.row(align=True)
            row.label(text=f"{blend['name']} ({len(blend['keys'])} pose(s))")
            row.operator("rigger.mirror_pose_blend", text="", icon='MOD_MIRROR').name = blend["name"]
            row.operator("rigger.remove_pose_blend", text="", icon='X').name = blend["name"]

        col = _section(layout, "Control Shapes")
        col.operator("rigger.edit_shape", icon='MESH_DATA')
        row = col.row(align=True)
        row.operator_menu_enum("rigger.set_shape", "shape", text="Set Shape", icon='MESH_CIRCLE')
        row.operator("rigger.apply_shape_transforms", text="", icon='CHECKMARK')
        row = col.row(align=True)
        row.operator("rigger.export_shapes", text="Export", icon='EXPORT')
        row.operator("rigger.import_shapes", text="Import", icon='IMPORT')
        col.operator("rigger.fit_shapes", icon='FULLSCREEN_ENTER')

        col = _section(layout, "Shape Keys")
        col.operator("rigger.corrective_shape", icon='SHAPEKEY_DATA')
        col.operator("rigger.shape_key_driver", icon='DRIVER')
        row = col.row(align=True)
        for value in ('0', '1', 'RESET'):
            row.operator("rigger.shape_key_driver_key", text=value.title()).value = value
        row = col.row(align=True)
        row.operator("rigger.remove_shape_driver", icon='X')
        row.operator("rigger.mirror_shape_keys", icon='MOD_MIRROR')

        col = _section(layout, "Custom Limbs (your own bones)")
        row = col.row(align=True)
        row.operator_menu_enum("rigger.add_custom_limb", "limb", text="Add", icon='IMPORT')
        row.operator("rigger.save_custom_limb", text="Save Selected", icon='FILE_TICK')
        row.operator_menu_enum("rigger.remove_custom_limb", "limb", text="", icon='TRASH')

        col = _section(layout, "Export Tags (selected bones)")
        row = col.row(align=True)
        for tag, text in (('CUSTOM', "Custom"), ('EXCLUDE', "Exclude"), ('SOFTLINK', "Soft-Link"),
                          ('CONST', "Constant"), ('NONE', "Clear")):
            row.operator("rigger.tag_bones", text=text).tag = tag


def _pick(layout, rig, text, names, icon='NONE'):
    names = [n for n in names if n and n in rig.pose.bones]
    sub = layout.row(align=True)
    sub.enabled = bool(names)
    op = sub.operator("rigger.select_controls", text=text, icon=icon)
    op.bones = ",".join(names)


class RIGGER_PT_picker(_Panel, bpy.types.Panel):
    bl_label = "Picker"
    bl_parent_id = "RIGGER_PT_main"

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def draw(self, context):
        layout = self.layout
        rig = context.object
        meta = utils.get_meta(rig)
        from . import picker
        row = layout.row(align=True)
        row.operator("rigger.viewport_picker", text="Viewport Picker", icon='RESTRICT_SELECT_OFF',
                     depress=picker.is_running())
        row.prop(context.scene, "rigger_picker_size", text="Size")
        col = layout.column(align=True)
        neck = meta.get("neck") or {}
        spine = meta.get("spine") or {}
        faces = meta.get("faces", [])
        row = col.row(align=True)
        _pick(row, rig, "Head", [neck.get("head")])
        _pick(row, rig, "Neck", neck.get("fk", []))
        if faces:
            _pick(row, rig, "Face", faces[0].get("controls", []))
        limbs = {(l["type"], l["side"]): l for l in meta.get("limbs", [])}
        for kind in ("arm", "leg"):
            row = col.row(align=True)
            for side in ("R", "L"):
                limb = limbs.get((kind, side))
                if not limb:
                    continue
                ik = [limb["ik_ctrl"], limb["pole"], limb.get("foot_roll"), limb.get("toe_ik"),
                      limb.get("ik_offset"), limb.get("ik_pivot")]
                fk = list(limb["fk"]) + [limb.get("toe_fk"), limb.get("shoulder")]
                _pick(row, rig, f"{kind.title()} IK {side}", ik)
                _pick(row, rig, f"FK {side}", fk)
            if kind == "arm":
                row = col.row(align=True)
                _pick(row, rig, "Chest", [spine.get("chest")])
                _pick(row, rig, "Spine", spine.get("fk", []))
                row = col.row(align=True)
                _pick(row, rig, "Torso", [spine.get("torso")])
                _pick(row, rig, "Hips", [spine.get("hips")])
        row = col.row(align=True)
        for side in ("R", "L"):
            fingers = [n for n in meta.get("controls", [])
                       if n.endswith("." + side) and any(f in n for f in ("thumb", "index", "middle",
                                                                          "ring", "pinky"))]
            _pick(row, rig, f"Fingers {side}", fingers)
        row = col.row(align=True)
        _pick(row, rig, "Root", ["root"])
        _pick(row, rig, "All", meta.get("controls", []), icon='SELECT_SET')


class RIGGER_PT_remap(_Panel, bpy.types.Panel):
    bl_label = "5. Retarget"
    bl_parent_id = "RIGGER_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.rigger_remap
        col = layout.column()
        col.prop(s, "source")
        col.prop(s, "target")
        row = col.row(align=True)
        row.prop(s, "legs_ik", toggle=True)
        row.prop(s, "arms_ik", toggle=True)
        col.operator("rigger.remap_auto", icon='AUTO')
        row = layout.row()
        row.template_list("RIGGER_UL_remap", "", s, "items", s, "active_index", rows=6)
        side = row.column(align=True)
        side.operator("rigger.remap_add", text="", icon='ADD')
        side.operator("rigger.remap_remove", text="", icon='REMOVE')
        side.separator()
        side.operator("rigger.remap_mirror", text="", icon='MOD_MIRROR')
        side.operator("rigger.remap_select", text="", icon='RESTRICT_SELECT_OFF')
        side.separator()
        side.operator("rigger.remap_save", text="", icon='FILE_TICK')
        side.operator("rigger.remap_load", text="", icon='FILE_FOLDER')
        if 0 <= s.active_index < len(s.items):
            item = s.items[s.active_index]
            box = layout.box()
            box.prop(item, "rot_offset")
            if item.location:
                box.prop(item, "location_mode")
                if item.location_mode == 'LOCAL':
                    box.prop(item, "loc_multiplier")
                else:
                    box.prop(item, "loc_offset")
        box = layout.box()
        row = box.row(align=True)
        row.prop(s, "namespace_from")
        row.prop(s, "namespace_to")
        box.operator("rigger.remap_namespace", icon='SORTALPHA')

        col = _section(layout, "Options")
        col.prop(s, "use_target_pose")
        col.prop(s, "source_frame")
        col.prop(s, "scale")
        col.prop(s, "in_place")
        col.prop(s, "pole_mode")
        col.prop(s, "smooth_poles")
        col.prop(s, "clean_fk")
        col.prop(s, "existing_keys")
        col.prop(s, "root_motion")
        row = col.row(align=True)
        row.prop(s, "frame_start")
        row.prop(s, "frame_end")
        col = _section(layout, "Source Animations")
        col.prop(s, "multiple")
        if s.multiple:
            col.prop(s, "action_filter")
            if s.source:
                col.template_list("RIGGER_UL_remap_actions", "", bpy.data, "actions", s, "action_index", rows=3)
                row = col.row(align=True)
                row.operator("rigger.remap_actions_select", text="All").state = True
                row.operator("rigger.remap_actions_select", text="None").state = False
        else:
            col.prop(s, "action_name")
        row = layout.row(align=True)
        row.scale_y = 1.3
        from .retarget import preview_active
        row.operator("rigger.remap_preview", text="Unbind" if preview_active() else "Bind Preview",
                     icon='UNLINKED' if preview_active() else 'LINKED', depress=preview_active())
        row.operator("rigger.remap_bake", icon='ACTION')


def draw_export_settings(layout, context):
    """Game export settings (sidebar panel and the export file browser)."""
    s = context.scene.rigger_export
    row = layout.row(align=True)
    row.prop(s, "engine", text="")
    row.prop(s, "file_format", text="")
    row = layout.row(align=True)
    row.menu("RIGGER_MT_export_presets", text="Presets", icon='PRESET')
    row.operator("rigger.export_preset_save", text="", icon='ADD')
    layout.row().prop(s, "ui_tab", expand=True)
    col = layout.column()
    if s.ui_tab == 'RIG':
        col.prop(s, "skeleton")
        if s.skeleton == 'HUMANOID':
            col.prop(s, "full_facial")
            col.prop(s, "metacarpals")
        col.prop(s, "naming")
        if s.naming == 'UNREAL':
            col.prop(s, "ue_ik_bones")
        if s.engine == 'GODOT':
            col.prop(s, "root_y_up")
        col.prop(s, "rig_name")
        col.prop(s, "export_twist")
        col.prop(s, "no_parents")
        col.prop(s, "selected_objects")
        col.prop(s, "selected_bones")
        col.prop(s, "rename_bones")
        col.prop(s, "custom_script")
    elif s.ui_tab == 'ANIM':
        col.prop(s, "bake_actions")
        if s.bake_actions:
            col.row().prop(s, "anim_type", expand=True)
            if s.anim_type == 'ACTIONS':
                col.prop(s, "use_actions_manager")
                if s.use_actions_manager:
                    row = col.row()
                    row.template_list("RIGGER_UL_action_lists", "", s, "action_lists", s, "action_list_index", rows=2)
                    side = row.column(align=True)
                    side.operator("rigger.action_list_edit", text="", icon='ADD').mode = 'ADD_LIST'
                    side.operator("rigger.action_list_edit", text="", icon='REMOVE').mode = 'REMOVE_LIST'
                    if 0 <= s.action_list_index < len(s.action_lists):
                        lst = s.action_lists[s.action_list_index]
                        col.template_list("RIGGER_UL_action_items", "", lst, "actions", s, "action_index", rows=3)
                        row = col.row(align=True)
                        row.operator("rigger.action_list_edit", text="Add Actions").mode = 'ADD_ACTIONS'
                        row.operator("rigger.action_list_edit", text="Clear").mode = 'CLEAR'
                else:
                    col.prop(s, "only_active")
                    if not s.only_active and utils.find_rig_for(context.object):
                        col.template_list("RIGGER_UL_export_actions", "", bpy.data, "actions", s, "action_index",
                                          rows=4)
                        row = col.row(align=True)
                        row.operator("rigger.export_actions_select", text="All").state = True
                        row.operator("rigger.export_actions_select", text="None").state = False
                col.prop(s, "action_search")
                col.prop(s, "one_file_per")
                col.prop(s, "frame_range")
                if s.frame_range == 'MANUAL':
                    row = col.row(align=True)
                    row.prop(s, "frame_start")
                    row.prop(s, "frame_end")
                row = col.row(align=True)
                row.prop(s, "action_names", text="")
                row.prop(s, "name_separator", text="")
            col.prop(s, "root_motion")
            col.prop(s, "reset_transforms")
            col.prop(s, "sampling")
            col.prop(s, "simplify")
            box = col.box()
            box.label(text="Actions Linker")
            row = box.row()
            row.template_list("RIGGER_UL_links", "", s, "links", s, "link_index", rows=2)
            side = row.column(align=True)
            side.operator("rigger.action_link_edit", text="", icon='ADD').remove = False
            side.operator("rigger.action_link_edit", text="", icon='REMOVE').remove = True
            if 0 <= s.link_index < len(s.links):
                link = s.links[s.link_index]
                for prop in ("rig_a", "action_a", "rig_b", "action_b"):
                    box.prop(link, prop)
    else:
        col.prop(s, "include_meshes")
        col.prop(s, "dummy_mesh")
        col.prop(s, "force_rest_pose")
        col.prop(s, "apply_modifiers")
        col.prop(s, "apply_subsurf")
        col.prop(s, "triangulate")
        col.prop(s, "smoothing")
        col.prop(s, "vertex_colors")
        col.prop(s, "embed_textures")
        col.prop(s, "units_x100")
        col.prop(s, "global_scale")
        col.prop(s, "init_rotation")
        row = col.row(align=True)
        row.prop(s, "primary_axis", text="")
        row.prop(s, "secondary_axis", text="")


class RIGGER_MT_export_presets(bpy.types.Menu):
    bl_label = "Export Presets"

    def draw(self, context):
        from .export import _preset_items
        layout = self.layout
        for key, label, _desc in _preset_items(None, context):
            if key == 'NONE':
                layout.label(text=label)
                continue
            row = layout.row(align=True)
            row.operator("rigger.export_preset_load", text=label).preset = key
            row.operator("rigger.export_preset_remove", text="", icon='X').preset = key


class RIGGER_PT_export(_Panel, bpy.types.Panel):
    bl_label = "6. Export"
    bl_parent_id = "RIGGER_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        draw_export_settings(layout, context)
        col = layout.column(align=True)
        row = col.row(align=True)
        row.operator("rigger.check_rig", text="Check Rig", icon='VIEWZOOM').fix = False
        row.operator("rigger.check_rig", text="Fix Rig", icon='TOOL_SETTINGS').fix = True
        row = col.row(align=True)
        row.scale_y = 1.4
        row.operator("rigger.export_fbx", icon='EXPORT')
        row.operator("rigger.quick_export", text="", icon='FILE_REFRESH')


classes = (RIGGER_UL_remap, RIGGER_UL_export_actions, RIGGER_UL_remap_actions, RIGGER_UL_action_lists,
           RIGGER_UL_action_items, RIGGER_UL_links, RIGGER_UL_quick_limbs, RIGGER_UL_quick_overrides,
           RIGGER_OT_pin_property, RIGGER_MT_space, RIGGER_MT_export_presets, RIGGER_PT_main, RIGGER_PT_reference,
           RIGGER_PT_template_options, RIGGER_PT_quick_rig, RIGGER_PT_generate, RIGGER_PT_skin, RIGGER_PT_controls,
           RIGGER_PT_picker, RIGGER_PT_tools, RIGGER_PT_remap, RIGGER_PT_export)
