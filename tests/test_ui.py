"""Draw every Rigger panel, menu and list in many contexts with a strict layout that checks each
property, operator (and operator property), list and menu the UI refers to.

    blender -b --factory-startup --python-exit-code 1 --python tests/test_ui.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, templates, ui, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
scene = ctx.scene
failures = []
errors = []
counts = {"prop": 0, "operator": 0, "list": 0}


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def _has_prop(data, name):
    if data is None:
        return False
    if name.startswith('["') and name.endswith('"]'):
        return name[2:-2] in data
    return name in data.bl_rna.properties


class _Op:
    def __init__(self, idname, where):
        module, name = idname.split(".")
        props = None
        try:
            props = {p.identifier for p in getattr(getattr(bpy.ops, module), name).get_rna_type().properties}
        except (AttributeError, KeyError):
            errors.append(f"{where}: unknown operator {idname}")
        object.__setattr__(self, "_props", props)
        object.__setattr__(self, "_where", f"{where} {idname}")

    def __setattr__(self, key, value):
        if self._props is not None and key not in self._props:
            errors.append(f"{self._where}: no operator property '{key}'")


class StrictLayout:
    def __init__(self, where):
        object.__setattr__(self, "where", where)

    def __setattr__(self, key, value):
        pass  # scale_y, enabled, operator_context...

    def _child(self, *_a, **_k):
        return self

    row = column = box = split = grid_flow = column_flow = _child

    def label(self, *a, **k):
        return None

    def separator(self, *a, **k):
        return None

    def context_pointer_set(self, *a, **k):
        return None

    def prop(self, data, name, *a, **k):
        counts["prop"] += 1
        if not _has_prop(data, name):
            errors.append(f"{self.where}: {type(data).__name__} has no property '{name}'")

    def prop_search(self, data, name, search_data, search_name, *a, **k):
        self.prop(data, name)
        self.prop(search_data, search_name)

    def operator(self, idname, *a, **k):
        counts["operator"] += 1
        return _Op(idname, self.where)

    def operator_menu_enum(self, idname, prop, *a, **k):
        op = _Op(idname, self.where)
        if op._props is not None and prop not in op._props:
            errors.append(f"{self.where}: {idname} has no enum '{prop}'")

    def menu(self, name, *a, **k):
        if not hasattr(bpy.types, name):
            errors.append(f"{self.where}: unknown menu {name}")

    def template_list(self, list_type, list_id, data, prop, active_data, active_prop, *a, **k):
        counts["list"] += 1
        if not hasattr(bpy.types, list_type):
            errors.append(f"{self.where}: unregistered list {list_type}")
        self.prop(data, prop)
        self.prop(active_data, active_prop)


def draw_all(label):
    for cls in ui.classes:
        where = f"{cls.__name__} [{label}]"
        if issubclass(cls, bpy.types.Panel):
            if hasattr(cls, "poll") and not cls.poll(ctx):
                continue
            try:
                cls.draw(type("P", (), {"layout": StrictLayout(where)})(), ctx)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{where}: {exc!r}")
        elif issubclass(cls, bpy.types.Menu) and cls is not ui.RIGGER_MT_space:
            try:
                cls.draw(type("M", (), {"layout": StrictLayout(where)})(), ctx)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{where}: {exc!r}")


class _OpView:
    """Stands in for an operator instance: its properties at their defaults, plus a layout."""

    def __init__(self, rna, layout):
        self.bl_rna = rna
        self.layout = layout
        for p in rna.properties:
            if p.identifier != "rna_type":
                setattr(self, p.identifier, getattr(p, "default", None))


def draw_operator(idname, label):
    """Run the draw() of an operator that has its own dialog layout."""
    module, name = idname.split(".")
    cls = getattr(bpy.types, f"{module.upper()}_OT_{name}")
    rna = getattr(getattr(bpy.ops, module), name).get_rna_type()
    try:
        cls.draw(_OpView(rna, StrictLayout(f"{idname} [{label}]")), ctx)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{idname} draw: {exc!r}")


# -- The strict layout catches mistakes
probe = StrictLayout("probe")
probe.prop(scene.rigger_export, "no_such_setting")
probe.operator("rigger.no_such_operator")
probe.operator("rigger.generate").no_such_property = 1
probe.template_list("RIGGER_UL_missing", "", scene.rigger_remap, "items", scene.rigger_remap, "active_index")
check(len(errors) == 4, f"strict layout reports mistakes ({len(errors)})")
errors.clear()

# -- Nothing selected
draw_all("empty")

# -- Reference: every module type as the active bone
opts = dict(templates.DEFAULT_OPTIONS, metacarpals=True, toes=True)
ref = templates.create_reference(ctx, templates.template_specs("HUMAN", options=opts), name="reference")
utils.set_mode(ref, 'OBJECT')
modules = {}
for pb in ref.pose.bones:
    modules.setdefault(pb.rigger.module, pb.name)
for module, bone in sorted(modules.items()):
    ref.data.bones.active = ref.data.bones[bone]
    draw_all(f"reference {module}")
for module in ('SPLINE', 'SKIRT', 'BBONE', 'EYE'):
    pb = ref.pose.bones["head"]
    old = pb.rigger.module
    ref.data.bones.active = ref.data.bones["head"]
    pb.rigger.module = module
    draw_all(f"reference {module}")
    pb.rigger.module = old
utils.set_mode(ref, 'EDIT')
draw_all("reference edit")
bpy.ops.object.mode_set(mode='OBJECT')

# -- Rig: object mode and pose mode with many active controls
rig = generate.generate(ctx, ref)
ref.hide_set(True)
ref.rigger.use_theme = True
draw_all("rig object")
utils.set_mode(rig, 'POSE')
meta = utils.get_meta(rig)
actives = set()
for limb in meta["limbs"]:
    actives.update([limb["ik_ctrl"], limb["fk"][0], limb["prop_bone"]])
for key in ("torso", "chest", "hips"):
    actives.add(meta["spine"].get(key))
actives.add(meta["neck"]["head"])
actives.update(f["ik_ctrl"] or f["fk"][0] for f in meta.get("fingers", []))
actives.add(meta["faces"][0]["face"] if meta.get("faces") else None)
for name in sorted(n for n in actives if n):
    rig.data.bones.active = rig.data.bones[name]
    draw_all(f"pose {name}")
# Pinned properties and IK mode
leg = next(l for l in meta["limbs"] if l["type"] == "leg")
rig.rigger.pinned_props = f"{leg['prop_bone']}:ik_fk"
rig.data.bones.active = rig.data.bones[leg["ik_ctrl"]]
rig.pose.bones[leg["prop_bone"]]["ik_fk"] = 1.0
draw_all("pose IK pinned")
res = bpy.ops.rigger.pin_property(bone=leg["prop_bone"], prop="ik_fk")
check(res == {'FINISHED'} and rig.rigger.pinned_props == "", "unpin property")
bpy.ops.rigger.pin_property(bone=leg["prop_bone"], prop="stretch")
check(rig.rigger.pinned_props == f"{leg['prop_bone']}:stretch", "pin property")
bpy.ops.object.mode_set(mode='OBJECT')

# -- Export tabs and options
s = scene.rigger_export
for tab in ('RIG', 'ANIM', 'MISC'):
    s.ui_tab = tab
    for combo in ({}, {"skeleton": 'HUMANOID', "naming": 'UNREAL', "engine": 'GODOT'},
                  {"only_active": False}, {"use_actions_manager": True}, {"anim_type": 'NLA'},
                  {"frame_range": 'MANUAL'}):
        for key, value in combo.items():
            setattr(s, key, value)
        if combo.get("use_actions_manager"):
            bpy.ops.rigger.action_list_edit(mode='ADD_LIST')
            bpy.ops.rigger.action_link_edit(remove=False)
        draw_all(f"export {tab} {combo}")
        ui.draw_export_settings(StrictLayout(f"draw_export_settings {tab} {combo}"), ctx)
s.use_actions_manager = False

# -- Retarget and Quick Rig with items
src = bpy.data.objects.new("src", bpy.data.armatures.new("src"))
scene.collection.objects.link(src)
utils.set_mode(src, 'EDIT')
eb = src.data.edit_bones.new("Hips")
eb.tail = (0, 0, 0.2)
bpy.ops.object.mode_set(mode='OBJECT')
remap = scene.rigger_remap
remap.source, remap.target = src, rig
item = remap.items.add()
item.source, item.target, item.location = "Hips", "torso", True
for mode in ('WORLD', 'LOCAL'):
    item.location_mode = mode
    for multiple in (False, True):
        remap.multiple = multiple
        draw_all(f"remap {mode} {multiple}")
qs = scene.rigger_quick
qs.armature = src
qs.limbs.add().kind = 'FINGER'
for mode in ('PRESERVE', 'CONVERT'):
    qs.mode = mode
    utils.set_mode(src, 'OBJECT')
    draw_all(f"quick rig {mode}")

# -- Mesh active (skin panel, eyelid borders)
bpy.ops.mesh.primitive_cube_add()
mesh = ctx.object
draw_all("mesh")

# -- UI lists draw their items
lists = {
    ui.RIGGER_UL_remap: (remap, remap.items[0]),
    ui.RIGGER_UL_quick_limbs: (qs, qs.limbs[0]),
    ui.RIGGER_UL_links: (s, s.links[0]),
    ui.RIGGER_UL_action_lists: (s, s.action_lists[0]),
}
qs.overrides.add()
lists[ui.RIGGER_UL_quick_overrides] = (qs, qs.overrides[0])
s.action_lists[0].actions.add()
lists[ui.RIGGER_UL_action_items] = (s.action_lists[0], s.action_lists[0].actions[0])
action = bpy.data.actions.new("walk")
lists[ui.RIGGER_UL_export_actions] = (bpy.data, action)
lists[ui.RIGGER_UL_remap_actions] = (bpy.data, action)
for cls, (data, item) in lists.items():
    try:
        cls.draw_item(cls, ctx, StrictLayout(cls.__name__), data, item, 0, data, "", 0)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{cls.__name__}: {exc!r}")

# -- Bind dialog
utils.set_mode(rig, 'OBJECT')
draw_operator("rigger.bind", "dialog")

for e in errors:
    print("UI ERROR", e)
check(not errors, f"panels, menus and lists only use existing properties and operators "
                  f"({counts['prop']} props, {counts['operator']} buttons, {counts['list']} lists, "
                  f"{len(errors)} errors)")

if failures:
    print(f"\n{len(failures)} FAILURES:")
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("\nFAILURES: none")
