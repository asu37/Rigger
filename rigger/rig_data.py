# SPDX-License-Identifier: GPL-3.0-or-later
"""Rig data tools.

* Custom bones (bones you add to a generated rig) are kept when the rig is
  regenerated, with their shapes, constraints, drivers and properties.
* Custom limbs: save selected custom bones to a library file, add them to any rig.
* Import / export rig data: reference bones, module settings, control shapes,
  colours, pose blends and rig layers, to rebuild a broken rig or reuse a setup.
* Armature presets: save a reference as a template preset.
* Set Character Name, Color Theme, Set Custom Shape, Apply Shape Transforms,
  import / export of control shapes, export tags for game export.
"""

import json
import os

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Euler, Matrix, Vector

from . import utils, widgets

RIG_TOKEN = "__rig__"
TAGS_KEY = "rigger_export_tags"
EXPORT_TAGS = [
    ('CUSTOM', "Custom Bone", "Exported with the deform skeleton (props, hair, weapons...)"),
    ('EXCLUDE', "Exclude", "Not exported"),
    ('SOFTLINK', "Soft-Link", "Exported without scale: stretched position, scale 1"),
    ('CONST', "Constant", "Exported keys use constant interpolation"),
]


def library_dir(kind):
    """User folder for Rigger libraries (custom limbs, presets)."""
    base = bpy.utils.user_resource('CONFIG', path="rigger", create=True)
    path = os.path.join(base, kind)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Generic serialisation of bones, constraints and drivers
# ---------------------------------------------------------------------------

_SKIP = {"rna_type", "type", "is_valid", "error_location", "error_rotation", "is_override_data",
         "show_expanded", "active", "targets", "action_slot", "action_slot_handle", "last_slot_identifier",
         "action_suitable_slots"}


def _value(v):
    if isinstance(v, (bool, int, float, str)):
        return v
    if hasattr(v, "__len__") and not isinstance(v, str):
        try:
            return [_value(x) for x in v]
        except TypeError:
            return None
    return None


def serialize_constraint(rig, con):
    data = {"type": con.type, "props": {}}
    for prop in con.bl_rna.properties:
        key = prop.identifier
        if key in _SKIP or prop.is_readonly:
            continue
        value = getattr(con, key)
        if prop.type == 'POINTER':
            if value is None:
                continue
            data["props"][key] = {"ref": RIG_TOKEN if value == rig else value.name, "kind": type(value).__name__}
        elif prop.type == 'COLLECTION':
            continue
        else:
            v = _value(value)
            if v is not None:
                data["props"][key] = v
    return data


def _lookup(ref, rig):
    if ref["ref"] == RIG_TOKEN:
        return rig
    kind = ref.get("kind", "Object")
    store = {"Action": bpy.data.actions, "Text": bpy.data.texts}.get(kind, bpy.data.objects)
    return store.get(ref["ref"])


def restore_constraint(rig, pb, data):
    con = pb.constraints.new(data["type"])
    pointers = {k: v for k, v in data["props"].items() if isinstance(v, dict)}
    for key, ref in pointers.items():
        target = _lookup(ref, rig)
        if target is not None:
            try:
                setattr(con, key, target)
            except (AttributeError, TypeError, ValueError):
                pass
    for key, value in data["props"].items():
        if isinstance(value, dict):
            continue
        try:
            setattr(con, key, value)
        except (AttributeError, TypeError, ValueError):
            pass
    return con


def serialize_driver(rig, fcurve):
    driver = fcurve.driver
    data = {"path": fcurve.data_path, "index": fcurve.array_index, "type": driver.type,
            "expression": driver.expression, "variables": []}
    for var in driver.variables:
        targets = []
        for t in var.targets:
            targets.append({"id": (RIG_TOKEN if t.id == rig else (t.id.name if t.id else None)),
                            "id_type": t.id_type, "data_path": t.data_path, "bone": t.bone_target,
                            "transform_type": t.transform_type, "transform_space": t.transform_space,
                            "rotation_mode": t.rotation_mode})
        data["variables"].append({"name": var.name, "type": var.type, "targets": targets})
    return data


def restore_driver(rig, owner, data, rename=None):
    rename = rename or {}
    path = data["path"]
    for old, new in rename.items():
        path = path.replace(f'"{old}"', f'"{new}"')
    try:
        fcurve = owner.driver_add(path, data["index"]) if data["index"] >= 0 else owner.driver_add(path)
    except TypeError:
        fcurve = owner.driver_add(path)
    except (AttributeError, RuntimeError):
        return None
    driver = fcurve.driver
    driver.type = data["type"]
    for var in list(driver.variables):
        driver.variables.remove(var)
    for vdata in data["variables"]:
        var = driver.variables.new()
        var.name = vdata["name"]
        var.type = vdata["type"]
        for t, tdata in zip(var.targets, vdata["targets"]):
            if vdata["type"] == 'SINGLE_PROP':
                t.id_type = tdata["id_type"]
            if tdata["id"] == RIG_TOKEN:
                t.id = rig
            elif tdata["id"]:
                t.id = bpy.data.objects.get(tdata["id"])
            t.data_path = tdata["data_path"]
            t.bone_target = rename.get(tdata["bone"], tdata["bone"])
            t.transform_type = tdata["transform_type"]
            t.transform_space = tdata["transform_space"]
            t.rotation_mode = tdata["rotation_mode"]
    driver.expression = data["expression"]
    for mod in list(fcurve.modifiers):
        fcurve.modifiers.remove(mod)
    return fcurve


def _shape_data(obj):
    if obj is None or obj.type != 'MESH':
        return None
    return {"name": obj.name, "verts": [list(v.co) for v in obj.data.vertices],
            "edges": [list(e.vertices) for e in obj.data.edges]}


def serialize_bone(rig, name, with_shape_mesh=False):
    """Edit, pose and driver data of one bone of 'rig' (call in Object or Pose mode)."""
    bone = rig.data.bones[name]
    pb = rig.pose.bones[name]
    m = bone.matrix_local
    data = {
        "name": name, "head": list(bone.head_local), "tail": list(bone.tail_local),
        "z": list(m.to_3x3().col[2]), "parent": bone.parent.name if bone.parent else None,
        "connect": bone.use_connect, "deform": bone.use_deform, "bbone_segments": bone.bbone_segments,
        "collections": [c.name for c in bone.collections], "inherit_scale": bone.inherit_scale,
        "use_inherit_rotation": bone.use_inherit_rotation, "hide": bone.hide,
        "rotation_mode": pb.rotation_mode,
        "lock_location": list(pb.lock_location), "lock_rotation": list(pb.lock_rotation),
        "lock_rotation_w": pb.lock_rotation_w, "lock_scale": list(pb.lock_scale),
        "shape": pb.custom_shape.name if pb.custom_shape else None,
        "shape_scale": list(pb.custom_shape_scale_xyz),
        "shape_translation": list(pb.custom_shape_translation),
        "shape_rotation": list(pb.custom_shape_rotation_euler),
        "shape_bone_size": pb.use_custom_shape_bone_size,
        "wire": getattr(pb, "custom_shape_wire_width", 1.0),
        "color": [list(pb.color.custom.normal), list(pb.color.custom.select), list(pb.color.custom.active)]
        if pb.color.palette == 'CUSTOM' else pb.color.palette,
        "props": {k: _value(pb[k]) for k in pb.keys() if _value(pb[k]) is not None},
        "bone_props": {k: _value(bone[k]) for k in bone.keys() if _value(bone[k]) is not None},
        "constraints": [serialize_constraint(rig, c) for c in pb.constraints],
        "drivers": [],
    }
    if with_shape_mesh:
        data["shape_mesh"] = _shape_data(pb.custom_shape)
    anim = rig.animation_data
    if anim:
        prefix = f'pose.bones["{name}"]'
        data["drivers"] = [serialize_driver(rig, fc) for fc in anim.drivers if fc.data_path.startswith(prefix)]
    return data


def restore_bones(context, rig, bones, rename=None, offset=None):
    """Create bones from serialised data. rename: {old: new} applied to names, parents and targets."""
    rename = dict(rename or {})
    offset = Vector(offset) if offset else Vector()
    mode = rig.mode
    utils.set_mode(rig, 'EDIT')
    ebones = rig.data.edit_bones
    created = []
    for data in bones:
        name = rename.get(data["name"], data["name"])
        if name in ebones:
            continue
        eb = ebones.new(name)
        eb.head, eb.tail = Vector(data["head"]) + offset, Vector(data["tail"]) + offset
        eb.align_roll(Vector(data["z"]))
        eb.use_deform = data["deform"]
        eb.bbone_segments = data.get("bbone_segments", 1)
        eb.inherit_scale = data.get("inherit_scale", 'FULL')
        eb.use_inherit_rotation = data.get("use_inherit_rotation", True)
        for coll_name in data["collections"]:
            coll = rig.data.collections.get(coll_name) or rig.data.collections.new(coll_name)
            coll.assign(eb)
        created.append((name, data))
    for name, data in created:
        parent = rename.get(data["parent"], data["parent"]) if data["parent"] else None
        if parent and parent in ebones:
            ebones[name].parent = ebones[parent]
            ebones[name].use_connect = data["connect"]
    bpy.ops.object.mode_set(mode='POSE')
    pbs = rig.pose.bones
    for name, data in created:
        pb = pbs[name]
        pb.rotation_mode = data["rotation_mode"]
        pb.lock_location = data["lock_location"]
        pb.lock_rotation = data["lock_rotation"]
        pb.lock_rotation_w = data["lock_rotation_w"]
        pb.lock_scale = data["lock_scale"]
        rig.data.bones[name].hide = data.get("hide", False)
        shape = data.get("shape_mesh")
        if shape:
            pb.custom_shape = widgets._make_object(f"WGT-{rig.name}_{name}", shape["verts"], shape["edges"])
        elif data["shape"]:
            pb.custom_shape = bpy.data.objects.get(data["shape"])
        pb.custom_shape_scale_xyz = data["shape_scale"]
        pb.custom_shape_translation = data["shape_translation"]
        pb.custom_shape_rotation_euler = data["shape_rotation"]
        pb.use_custom_shape_bone_size = data["shape_bone_size"]
        if hasattr(pb, "custom_shape_wire_width"):
            pb.custom_shape_wire_width = data["wire"]
        if isinstance(data["color"], list):
            pb.color.palette = 'CUSTOM'
            pb.color.custom.normal, pb.color.custom.select, pb.color.custom.active = data["color"]
        else:
            pb.color.palette = data["color"]
        for key, value in data["props"].items():
            pb[key] = value
        for key, value in data.get("bone_props", {}).items():
            rig.data.bones[name][key] = value
    for name, data in created:
        pb = pbs[name]
        for cdata in data["constraints"]:
            cdata = json.loads(json.dumps(cdata))
            sub = cdata["props"].get("subtarget")
            if isinstance(sub, str):
                cdata["props"]["subtarget"] = rename.get(sub, sub)
            restore_constraint(rig, pb, cdata)
        for ddata in data["drivers"]:
            restore_driver(rig, rig, ddata, rename)
    bpy.ops.object.mode_set(mode=mode if mode in ('POSE', 'EDIT') else 'OBJECT')
    return [n for n, _d in created]


# ---------------------------------------------------------------------------
# Custom bones kept across regeneration
# ---------------------------------------------------------------------------

def custom_bone_names(rig):
    generated = set(utils.get_meta(rig).get("generated", []))
    if not generated:
        return []
    return [b.name for b in rig.data.bones if b.name not in generated]


def capture_custom(rig):
    """Data needed to rebuild the user's custom bones and collection visibility after regeneration."""
    names = custom_bone_names(rig) if rig.data.bones else []
    # Parents first
    names.sort(key=lambda n: len(rig.data.bones[n].parent_recursive))
    bones = [serialize_bone(rig, n) for n in names]
    builtin = {n for n, _v in utils.RIG_COLLECTIONS}
    visibility = {c.name: c.is_visible for c in rig.data.collections_all if c.name not in builtin} \
        if hasattr(rig.data, "collections_all") else {}
    return {"bones": bones, "visibility": visibility}


def restore_custom(context, rig, captured):
    if captured["bones"]:
        restore_bones(context, rig, captured["bones"])
    for name, visible in captured["visibility"].items():
        coll = rig.data.collections.get(name)
        if coll is not None:
            coll.is_visible = visible


# ---------------------------------------------------------------------------
# Custom limbs
# ---------------------------------------------------------------------------

def _limb_items(_self, _context):
    folder = library_dir("limbs")
    items = [(f[:-5], f[:-5], "") for f in sorted(os.listdir(folder)) if f.endswith(".json")]
    return items or [('NONE', "(no custom limbs saved)", "")]


class RIGGER_OT_save_custom_limb(bpy.types.Operator):
    """Save the selected custom bones (with shapes, constraints, drivers and properties) as a custom limb"""
    bl_idname = "rigger.save_custom_limb"
    bl_label = "Save Custom Limb"

    name: StringProperty(name="Name", default="custom_limb")

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        rig = context.object
        names = [pb.name for pb in rig.pose.bones if utils.is_pose_bone_selected(pb)]
        if not names:
            self.report({'ERROR'}, "Select the bones to save")
            return {'CANCELLED'}
        names.sort(key=lambda n: len(rig.data.bones[n].parent_recursive))
        bones = [serialize_bone(rig, n, with_shape_mesh=True) for n in names]
        path = os.path.join(library_dir("limbs"), bpy.path.clean_name(self.name) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "bones": bones}, f, indent=1)
        self.report({'INFO'}, f"Saved {len(bones)} bones to {path}")
        return {'FINISHED'}


class RIGGER_OT_add_custom_limb(bpy.types.Operator):
    """Add a saved custom limb to the rig. Bones whose parent is missing are parented to the active bone"""
    bl_idname = "rigger.add_custom_limb"
    bl_label = "Add Custom Limb"
    bl_options = {'REGISTER', 'UNDO'}

    limb: EnumProperty(name="Limb", items=_limb_items)

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        path = os.path.join(library_dir("limbs"), self.limb + ".json")
        if not os.path.exists(path):
            return {'CANCELLED'}
        with open(path, encoding="utf-8") as f:
            bones = json.load(f)["bones"]
        existing = set(rig.data.bones.keys())
        rename = {}
        for data in bones:
            rename[data["name"]] = utils.unique_name(existing, data["name"])
            existing.add(rename[data["name"]])
        active = rig.data.bones.active.name if rig.data.bones.active else "root"
        names = {d["name"] for d in bones}
        for data in bones:
            if data["parent"] not in names and data["parent"] not in rig.data.bones:
                data["parent"] = active
        created = restore_bones(context, rig, bones, rename)
        self.report({'INFO'}, f"Added {len(created)} bones")
        return {'FINISHED'}


class RIGGER_OT_remove_custom_limb(bpy.types.Operator):
    """Delete a saved custom limb from the library"""
    bl_idname = "rigger.remove_custom_limb"
    bl_label = "Remove Custom Limb"

    limb: EnumProperty(name="Limb", items=_limb_items)

    def execute(self, context):
        path = os.path.join(library_dir("limbs"), self.limb + ".json")
        if os.path.exists(path):
            os.remove(path)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Rig data import / export, presets
# ---------------------------------------------------------------------------

def reference_data(ref):
    bones = []
    for bone in ref.data.bones:
        settings = ref.pose.bones[bone.name].rigger
        values = {}
        for prop in settings.bl_rna.properties:
            if prop.identifier in ("rna_type", "name") or prop.is_readonly:
                continue
            values[prop.identifier] = _value(getattr(settings, prop.identifier))
        bones.append({"name": bone.name, "head": list(bone.head_local), "tail": list(bone.tail_local),
                      "z": list(bone.matrix_local.to_3x3().col[2]),
                      "parent": bone.parent.name if bone.parent else None, "connect": bone.use_connect,
                      "settings": values})
    obj = {k: _value(getattr(ref.rigger, k)) for k in ("rig_name", "widget_scale", "auto_hide", "multiped",
                                                       "init_scale", "use_theme", "color_left", "color_right",
                                                       "color_center") if hasattr(ref.rigger, k)}
    from .poses import POSES_KEY
    return {"bones": bones, "object": obj, "poses": ref.get(POSES_KEY, "[]")}


def apply_reference_data(context, ref, data, transforms=True, settings=True, selected_only=False):
    utils.set_mode(ref, 'EDIT')
    ebones = ref.data.edit_bones
    selected = {eb.name for eb in ebones if eb.select} if selected_only else None
    if transforms:
        for b in data["bones"]:
            if selected is not None and b["name"] not in selected:
                continue
            eb = ebones.get(b["name"]) or ebones.new(b["name"])
            eb.head, eb.tail = Vector(b["head"]), Vector(b["tail"])
            eb.align_roll(Vector(b["z"]))
        for b in data["bones"]:
            if (selected is not None and b["name"] not in selected) or b["name"] not in ebones:
                continue
            eb = ebones[b["name"]]
            eb.parent = ebones.get(b["parent"]) if b["parent"] else None
            eb.use_connect = b["connect"] and eb.parent is not None
    bpy.ops.object.mode_set(mode='OBJECT')
    if settings:
        for b in data["bones"]:
            pb = ref.pose.bones.get(b["name"])
            if pb is None or (selected is not None and b["name"] not in selected):
                continue
            for key, value in b["settings"].items():
                try:
                    setattr(pb.rigger, key, value)
                except (AttributeError, TypeError, ValueError):
                    pass
        for key, value in data.get("object", {}).items():
            try:
                setattr(ref.rigger, key, value)
            except (AttributeError, TypeError, ValueError):
                pass
        from .poses import POSES_KEY
        if data.get("poses"):
            ref[POSES_KEY] = data["poses"]


def shapes_data(rig, names=None):
    shapes = {}
    for pb in rig.pose.bones:
        if pb.custom_shape is None or (names is not None and pb.name not in names):
            continue
        shapes[pb.name] = {"mesh": _shape_data(pb.custom_shape), "scale": list(pb.custom_shape_scale_xyz),
                           "translation": list(pb.custom_shape_translation),
                           "rotation": list(pb.custom_shape_rotation_euler),
                           "bone_size": pb.use_custom_shape_bone_size,
                           "wire": getattr(pb, "custom_shape_wire_width", 1.0),
                           "color": [list(pb.color.custom.normal), list(pb.color.custom.select),
                                     list(pb.color.custom.active)] if pb.color.palette == 'CUSTOM' else None}
    return shapes


def apply_shapes(rig, shapes):
    count = 0
    for name, s in shapes.items():
        pb = rig.pose.bones.get(name)
        if pb is None or not s.get("mesh"):
            continue
        pb.custom_shape = widgets._make_object(f"WGT-{rig.name}_{name}", s["mesh"]["verts"], s["mesh"]["edges"])
        pb.custom_shape_scale_xyz = s["scale"]
        pb.custom_shape_translation = s["translation"]
        pb.custom_shape_rotation_euler = s["rotation"]
        pb.use_custom_shape_bone_size = s["bone_size"]
        if hasattr(pb, "custom_shape_wire_width"):
            pb.custom_shape_wire_width = s["wire"]
        if s.get("color"):
            pb.color.palette = 'CUSTOM'
            pb.color.custom.normal, pb.color.custom.select, pb.color.custom.active = s["color"]
        count += 1
    return count


def layers_data(rig):
    return [{"name": l.name, "show": l.show, "objects": [i.obj.name for i in l.objects if i.obj],
             "bone_collections": l.bone_collections} for l in rig.rigger.layers]


def apply_layers(rig, layers):
    for data in layers:
        layer = next((l for l in rig.rigger.layers if l.name == data["name"]), None) or rig.rigger.layers.add()
        layer.name = data["name"]
        layer.bone_collections = data["bone_collections"]
        names = {i.obj.name for i in layer.objects if i.obj}
        for obj_name in data["objects"]:
            if obj_name in bpy.data.objects and obj_name not in names:
                layer.objects.add().obj = bpy.data.objects[obj_name]
        layer.show = data["show"]


def _ref_and_rig(context):
    obj = context.object
    rig = utils.find_rig_for(obj)
    ref = obj if utils.is_reference(obj) else (rig.rigger.reference if rig else None)
    return ref, rig


class RIGGER_OT_export_rig_data(ExportHelper, bpy.types.Operator):
    """Save the reference bones, module settings, control shapes, pose blends and rig layers to a file"""
    bl_idname = "rigger.export_rig_data"
    bl_label = "Export Rig Data"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _ref_and_rig(context)[0] is not None

    def execute(self, context):
        ref, rig = _ref_and_rig(context)
        data = {"version": 1, "reference": reference_data(ref)}
        if rig is not None:
            data["shapes"] = shapes_data(rig)
            data["layers"] = layers_data(rig)
            data["custom"] = [serialize_bone(rig, n, True) for n in sorted(
                custom_bone_names(rig), key=lambda n: len(rig.data.bones[n].parent_recursive))]
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        self.report({'INFO'}, f"Saved rig data to {self.filepath}")
        return {'FINISHED'}


class RIGGER_OT_import_rig_data(ImportHelper, bpy.types.Operator):
    """Load rig data saved with Export Rig Data into the active reference (and regenerate the rig)"""
    bl_idname = "rigger.import_rig_data"
    bl_label = "Import Rig Data"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    transforms: BoolProperty(name="Reference Bones", default=True)
    settings: BoolProperty(name="Module Settings", default=True)
    shapes: BoolProperty(name="Control Shapes", default=True)
    layers: BoolProperty(name="Rig Layers", default=True)
    custom: BoolProperty(name="Custom Bones", default=True)
    selected_only: BoolProperty(name="Selected Bones Only", default=False)

    @classmethod
    def poll(cls, context):
        return _ref_and_rig(context)[0] is not None

    def execute(self, context):
        with open(self.filepath, encoding="utf-8") as f:
            data = json.load(f)
        ref, rig = _ref_and_rig(context)
        import_rig_data(context, ref, data, self.transforms, self.settings, self.shapes, self.layers, self.custom,
                        self.selected_only)
        return {'FINISHED'}


def import_rig_data(context, ref, data, transforms=True, settings=True, shapes=True, layers=True, custom=True,
                    selected_only=False):
    from .generate import generate
    apply_reference_data(context, ref, data["reference"], transforms, settings, selected_only)
    utils.set_mode(ref, 'OBJECT')
    rig = generate(context, ref)
    if shapes and data.get("shapes"):
        apply_shapes(rig, data["shapes"])
    if layers and data.get("layers"):
        apply_layers(rig, data["layers"])
    if custom and data.get("custom"):
        restore_bones(context, rig, data["custom"])
    ref.hide_set(True)
    utils.set_mode(rig, 'OBJECT')
    return rig


def _preset_items(_self, _context):
    folder = library_dir("presets")
    items = [(f[:-5], f[:-5], "") for f in sorted(os.listdir(folder)) if f.endswith(".json")]
    return items or [('NONE', "(no presets saved)", "")]


class RIGGER_OT_save_preset(bpy.types.Operator):
    """Save the active reference armature as a preset for Add Reference"""
    bl_idname = "rigger.save_preset"
    bl_label = "Save Armature Preset"

    name: StringProperty(name="Name", default="my_character")

    @classmethod
    def poll(cls, context):
        return _ref_and_rig(context)[0] is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        ref, _rig = _ref_and_rig(context)
        if ref.mode == 'EDIT':
            bpy.ops.object.mode_set(mode='OBJECT')
        path = os.path.join(library_dir("presets"), bpy.path.clean_name(self.name) + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "reference": reference_data(ref)}, f, indent=1)
        self.report({'INFO'}, f"Saved preset {path}")
        return {'FINISHED'}


class RIGGER_OT_add_preset(bpy.types.Operator):
    """Add a reference armature from a saved preset"""
    bl_idname = "rigger.add_preset"
    bl_label = "Add Reference from Preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(name="Preset", items=_preset_items)

    def execute(self, context):
        path = os.path.join(library_dir("presets"), self.preset + ".json")
        if not os.path.exists(path):
            return {'CANCELLED'}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        arm = bpy.data.armatures.new(self.preset)
        ref = bpy.data.objects.new(self.preset, arm)
        context.collection.objects.link(ref)
        ref.location = context.scene.cursor.location
        ref[utils.REF_KEY] = True
        ref.show_in_front = True
        apply_reference_data(context, ref, data["reference"])
        utils.set_mode(ref, 'OBJECT')
        return {'FINISHED'}


class RIGGER_OT_remove_preset(bpy.types.Operator):
    """Delete a saved armature preset"""
    bl_idname = "rigger.remove_preset"
    bl_label = "Remove Armature Preset"

    preset: EnumProperty(name="Preset", items=_preset_items)

    def execute(self, context):
        path = os.path.join(library_dir("presets"), self.preset + ".json")
        if os.path.exists(path):
            os.remove(path)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Character name, colour theme, shapes
# ---------------------------------------------------------------------------

def set_character_name(context, rig, name):
    """Rename the rig, its reference, curves, widgets and collections after the character."""
    ref = rig.rigger.reference
    old = rig.name
    main = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if main.name not in context.scene.collection.children:
        context.scene.collection.children.link(main)
    rig_coll = bpy.data.collections.get(f"{name}_rig") or bpy.data.collections.new(f"{name}_rig")
    if rig_coll.name not in main.children:
        main.children.link(rig_coll)
    cs_coll = bpy.data.collections.get(f"{name}_cs") or bpy.data.collections.new(f"{name}_cs")
    if cs_coll.name not in main.children:
        main.children.link(cs_coll)
    cs_coll.hide_viewport = cs_coll.hide_render = True

    def move(obj, coll):
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        coll.objects.link(obj)
    rig.name = name
    for obj in [rig, ref] + [o for o in bpy.data.objects if o.parent == rig and o.type == 'CURVE']:
        if obj is not None:
            move(obj, rig_coll)
    if ref is not None:
        ref.name = f"{name}_reference"
        ref.rigger.rig_name = name
    for obj in list(bpy.data.objects):
        if obj.name.startswith(f"WGT-{old}_"):
            obj.name = f"WGT-{name}_" + obj.name[len(f"WGT-{old}_"):]
            move(obj, cs_coll)
        elif obj.type == 'CURVE' and obj.name.startswith(f"{old}_"):
            obj.name = f"{name}_" + obj.name[len(old) + 1:]
    for pb in rig.pose.bones:
        if pb.custom_shape is not None and pb.custom_shape.name.startswith("WGT-rigger_"):
            if pb.custom_shape.name not in cs_coll.objects:
                cs_coll.objects.link(pb.custom_shape)
    for obj in context.scene.objects:
        if obj.type == 'MESH' and utils.find_rig_for(obj) == rig and obj.name not in main.objects:
            move(obj, main)
    return main


class RIGGER_OT_set_character_name(bpy.types.Operator):
    """Rename the rig and organise it in '<name>', '<name>_rig' and '<name>_cs' collections"""
    bl_idname = "rigger.set_character_name"
    bl_label = "Set Character Name"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(name="Name", default="character")

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        rig = utils.find_rig_for(context.object)
        if context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        set_character_name(context, rig, self.name)
        return {'FINISHED'}


def _brighter(color, k):
    return tuple(min(1.0, c + (1.0 - c) * k) for c in color)


def apply_theme(rig, left, right, center):
    """Recolour every control by side: .L, .R and the rest."""
    count = 0
    for pb in rig.pose.bones:
        if pb.name.startswith(("DEF-", "MCH-")) or pb.custom_shape is None:
            continue
        side = utils.side_letter(pb.name)
        if side not in ("L", "R") and pb.color.palette == 'CUSTOM' and tuple(pb.color.custom.normal) in (
                widgets.COLORS["TWEAK"][0], widgets.COLORS["FACE"][0], widgets.COLORS["ROOT"][0]):
            continue
        color = left if side == "L" else right if side == "R" else center
        pb.color.palette = 'CUSTOM'
        pb.color.custom.normal = color
        pb.color.custom.select = _brighter(color, 0.5)
        pb.color.custom.active = _brighter(color, 0.8)
        count += 1
    return count


class RIGGER_OT_color_theme(bpy.types.Operator):
    """Colour the left, right and middle controls (kept when the rig is regenerated)"""
    bl_idname = "rigger.color_theme"
    bl_label = "Assign Color Theme"
    bl_options = {'REGISTER', 'UNDO'}

    reset: BoolProperty(name="Reset", default=False, description="Go back to the default colours")

    @classmethod
    def poll(cls, context):
        ref, rig = _ref_and_rig(context)
        return ref is not None and rig is not None

    def execute(self, context):
        ref, rig = _ref_and_rig(context)
        s = ref.rigger
        if self.reset:
            s.use_theme = False
            from .generate import generate
            mode = context.object.mode
            utils.set_mode(rig, 'OBJECT')
            rig = generate(context, ref)
            ref.hide_set(True)
            utils.set_mode(rig, mode if mode in ('POSE',) else 'OBJECT')
            return {'FINISHED'}
        s.use_theme = True
        count = apply_theme(rig, tuple(s.color_left), tuple(s.color_right), tuple(s.color_center))
        self.report({'INFO'}, f"Coloured {count} controls")
        return {'FINISHED'}


def _shape_items(_self, _context):
    items = [(k, k.replace("_", " ").title(), "Built-in shape") for k in widgets.SHAPE_KINDS]
    coll = bpy.data.collections.get(widgets.WIDGET_COLLECTION)
    if coll:
        for obj in coll.objects:
            if obj.type == 'MESH' and not obj.name.startswith("WGT-rigger_"):
                items.append(("OBJ:" + obj.name, obj.name, "Shape object of this file"))
    return items


class RIGGER_OT_set_shape(bpy.types.Operator):
    """Give the selected controls a shape from the library"""
    bl_idname = "rigger.set_shape"
    bl_label = "Set Custom Shape"
    bl_options = {'REGISTER', 'UNDO'}

    shape: EnumProperty(name="Shape", items=_shape_items)
    scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=100.0,
                         description="Size relative to the bone length")

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def execute(self, context):
        rig = context.object
        count = 0
        for pb in rig.pose.bones:
            if not utils.is_pose_bone_selected(pb):
                continue
            if self.shape.startswith("OBJ:"):
                pb.custom_shape = bpy.data.objects.get(self.shape[4:])
                pb.custom_shape_scale_xyz = (self.scale,) * 3
            else:
                widgets.apply_widget(pb, self.shape, self.scale)
            count += 1
        self.report({'INFO'}, f"Set the shape of {count} controls")
        return {'FINISHED'} if count else {'CANCELLED'}


def apply_shape_transforms(rig, pb):
    """Bake the custom shape translation/rotation/scale into the shape mesh, then reset them."""
    if pb.custom_shape is None or pb.custom_shape.type != 'MESH':
        return False
    length = pb.bone.length if pb.use_custom_shape_bone_size else 1.0
    t = Vector(pb.custom_shape_translation)
    r = Euler(pb.custom_shape_rotation_euler).to_matrix().to_4x4()
    s = Vector(pb.custom_shape_scale_xyz) * length
    if t.length < 1e-9 and r == Matrix() and (s - Vector((length,) * 3)).length < 1e-9:
        return False
    # Drawn: bone @ T @ R @ S(length * scale). After: bone @ S(length) @ v'
    to_new = Matrix.Diagonal((1 / length, 1 / length, 1 / length, 1.0)) @ Matrix.Translation(t) @ r \
        @ Matrix.Diagonal((*s, 1.0))
    verts = [tuple(to_new @ v.co) for v in pb.custom_shape.data.vertices]
    edges = [tuple(e.vertices) for e in pb.custom_shape.data.edges]
    pb.custom_shape = widgets._make_object(f"WGT-{rig.name}_{pb.name}", verts, edges)
    pb.custom_shape_translation = (0, 0, 0)
    pb.custom_shape_rotation_euler = (0, 0, 0)
    pb.custom_shape_scale_xyz = (1, 1, 1)
    return True


class RIGGER_OT_apply_shape_transforms(bpy.types.Operator):
    """Reset the shape offset/rotation/scale of the selected controls, keeping how the shapes look"""
    bl_idname = "rigger.apply_shape_transforms"
    bl_label = "Apply Shape Transforms"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def execute(self, context):
        rig = context.object
        count = sum(apply_shape_transforms(rig, pb) for pb in rig.pose.bones if utils.is_pose_bone_selected(pb))
        self.report({'INFO'}, f"Applied {count} shape transforms")
        return {'FINISHED'}


class RIGGER_OT_export_shapes(ExportHelper, bpy.types.Operator):
    """Save the control shapes (selected controls, or all) to a file"""
    bl_idname = "rigger.export_shapes"
    bl_label = "Export Shapes"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        names = [pb.name for pb in rig.pose.bones if utils.is_pose_bone_selected(pb)] or None
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "shapes": shapes_data(rig, names)}, f)
        return {'FINISHED'}


class RIGGER_OT_import_shapes(ImportHelper, bpy.types.Operator):
    """Load control shapes saved with Export Shapes"""
    bl_idname = "rigger.import_shapes"
    bl_label = "Import Shapes"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object)

    def execute(self, context):
        with open(self.filepath, encoding="utf-8") as f:
            count = apply_shapes(context.object, json.load(f)["shapes"])
        self.report({'INFO'}, f"Loaded {count} shapes")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Export tags
# ---------------------------------------------------------------------------

def export_tags(rig):
    """{bone: tag} from the tag store, 'cc_' names and 'cc' / 'custom_bone' properties."""
    try:
        tags = json.loads(rig.get(TAGS_KEY, "{}"))
    except (TypeError, ValueError):
        tags = {}
    tags = {b: t for b, t in tags.items() if b in rig.data.bones}
    for bone in rig.data.bones:
        pb = rig.pose.bones[bone.name]
        if bone.name.startswith("cc_") or any(k in bone or k in pb for k in ("cc", "custom_bone")):
            tags.setdefault(bone.name, 'CUSTOM')
    return tags


class RIGGER_OT_tag_bones(bpy.types.Operator):
    """Tag the selected bones for game export (custom bone, exclude, soft-link, constant)"""
    bl_idname = "rigger.tag_bones"
    bl_label = "Tag Bones for Export"
    bl_options = {'REGISTER', 'UNDO'}

    tag: EnumProperty(name="Tag", items=EXPORT_TAGS + [('NONE', "Clear", "Remove the tag")])

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def execute(self, context):
        rig = context.object
        try:
            tags = json.loads(rig.get(TAGS_KEY, "{}"))
        except (TypeError, ValueError):
            tags = {}
        count = 0
        for pb in rig.pose.bones:
            if utils.is_pose_bone_selected(pb):
                if self.tag == 'NONE':
                    tags.pop(pb.name, None)
                else:
                    tags[pb.name] = self.tag
                count += 1
        rig[TAGS_KEY] = json.dumps(tags)
        self.report({'INFO'}, f"Tagged {count} bones")
        return {'FINISHED'}


classes = (RIGGER_OT_save_custom_limb, RIGGER_OT_add_custom_limb, RIGGER_OT_remove_custom_limb,
           RIGGER_OT_export_rig_data, RIGGER_OT_import_rig_data, RIGGER_OT_save_preset, RIGGER_OT_add_preset,
           RIGGER_OT_remove_preset, RIGGER_OT_set_character_name, RIGGER_OT_color_theme, RIGGER_OT_set_shape,
           RIGGER_OT_apply_shape_transforms, RIGGER_OT_export_shapes, RIGGER_OT_import_shapes,
           RIGGER_OT_tag_bones)

