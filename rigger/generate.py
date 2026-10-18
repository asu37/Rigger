# SPDX-License-Identifier: GPL-3.0-or-later
"""Build a control rig from a reference armature."""

from collections import deque

import bpy
from mathutils import Vector

from . import utils, widgets
from .modules import MODULES
from .modules.chain import SimpleModule
from .safety import RiggerError  # noqa: F401 - re-exported, raised by the generator


class RefBone:
    __slots__ = ("name", "head", "tail", "z_axis", "x_axis", "parent", "children",
                 "connected", "length", "depth")

    def __init__(self, bone):
        mat = bone.matrix_local.to_3x3()
        self.name = bone.name
        self.head = bone.head_local.copy()
        self.tail = bone.tail_local.copy()
        self.x_axis = mat.col[0].normalized()
        self.z_axis = mat.col[2].normalized()
        self.parent = bone.parent.name if bone.parent else None
        self.children = sorted(c.name for c in bone.children)
        self.connected = bone.use_connect
        self.length = bone.length
        self.depth = len(bone.parent_recursive)


class Generator:
    def __init__(self, ref, rig=None):
        self.ref = ref
        self.rig = rig
        self.settings = ref.rigger
        self.widget_scale = self.settings.widget_scale
        self.auto_hide = self.settings.auto_hide
        self.bones = {b.name: RefBone(b) for b in ref.data.bones}
        if not self.bones:
            raise RiggerError("The reference armature has no bones")
        zs = [v.z for b in self.bones.values() for v in (b.head, b.tail)]
        ys = [v.y for b in self.bones.values() for v in (b.head, b.tail)]
        self.scale = max(max(zs) - min(zs), max(ys) - min(ys), 0.01)
        self.deform_of = {}
        self.shared = {}
        # Bones other modules can use as parent spaces, e.g. {"Torso": "torso"}
        self.spaces = {"Root": "root"}
        # Callables run after every module's edit pass (still in Edit Mode)
        self.post_edit = []
        self.space_switches = []
        # Controls resized to the skinned mesh after binding: {ctrl: {...}}
        self.fit = {}
        self.modules = []
        self.edit_bones = None
        from . import poses
        self.pose_blends = poses.load_blends(ref)

    # -- reference helpers ----------------------------------------------------
    def module_type(self, name):
        return self.ref.pose.bones[name].rigger.module

    def chain_from(self, root):
        chain = [root]
        current = root
        while True:
            kids = [c for c in self.bones[current].children if self.module_type(c) == 'NONE']
            if not kids:
                return chain
            connected = [c for c in kids if self.bones[c].connected]
            current = (connected or kids)[0]
            chain.append(current)

    def subtree(self, root):
        """Root plus every descendant without a module of its own, parents first."""
        result, queue = [root], deque(self.bones[root].children)
        while queue:
            name = queue.popleft()
            if self.module_type(name) != 'NONE':
                continue
            result.append(name)
            queue.extend(self.bones[name].children)
        return result

    def add_space_switch(self, ctrl, options):
        """Let 'ctrl' follow one of several parents, chosen by its 'parent' property.

        options: list of (label, bone) pairs; the first one is the default.
        Must be called in Edit Mode, after every module's edit pass.
        """
        options = [(label, bone) for label, bone in options if bone and bone in self.edit_bones]
        if len(options) < 2:
            return
        eb = self.edit_bones[ctrl]
        stem, suffix = utils.split_side(ctrl)
        space = self.new_bone(f"MCH-{stem}_space{suffix}", eb.head, eb.tail, eb.z_axis,
                              parent=options[0][1], collection="Mechanism")
        followers = []
        for label, bone in options:
            followers.append(self.new_bone(f"MCH-{stem}_in_{label.lower()}{suffix}", eb.head, eb.tail,
                                           eb.z_axis, parent=bone, collection="Mechanism"))
        eb.parent = self.edit_bones[space]
        self.space_switches.append((ctrl, space, [label for label, _b in options], followers))

    def _pose_space_switches(self):
        rig = self.rig
        for ctrl, space, labels, followers in self.space_switches:
            pb = rig.pose.bones[ctrl]
            pb["parent"] = 0
            pb.id_properties_ui("parent").update(
                min=0, max=len(labels) - 1,
                description="Parent space: " + ", ".join(f"{i} = {l}" for i, l in enumerate(labels)))
            for i, follower in enumerate(followers):
                con = utils.add_constraint(rig.pose.bones[space], 'COPY_TRANSFORMS', rig, follower)
                utils.add_driver(con, "influence", rig, ctrl, "parent", f"1.0 if var == {i} else 0.0")

    def parent_deform(self, ref_name):
        parent = self.bones[ref_name].parent
        if parent is None:
            fallback = self.ref.pose.bones[ref_name].rigger.parent_fallback
            if not fallback:
                return None
            if self.edit_bones is not None and fallback not in self.edit_bones:
                return "root"
            return fallback
        try:
            return self.deform_of[parent]
        except KeyError:
            raise RiggerError(f"Parent '{parent}' of '{ref_name}' was not generated before it")

    def collect_modules(self):
        ordered = sorted(self.bones.values(), key=lambda b: (b.depth, b.name))
        modules = []
        for bone in ordered:
            module_type = self.module_type(bone.name)
            if module_type != 'NONE':
                modules.append(MODULES[module_type](self, bone.name))
        claimed = {name for m in modules for name in m.chain}
        # Bones that are not part of any module become simple FK bones.
        queue = deque(b.name for b in ordered if b.name not in claimed)
        while queue:
            name = queue.popleft()
            if name in claimed:
                continue
            module = SimpleModule(self, name)
            modules.append(module)
            claimed.add(name)
        for module in modules:
            error = module.validate()
            if error:
                raise RiggerError(error)
        modules.sort(key=lambda m: (self._late(m.root), self.bones[m.root].depth, m.root))
        self.modules = modules

    def _late(self, name):
        """1 for bones under a top-level bone whose Parent Fallback is another rig bone: they are
        built after the other modules, so the fallback bone exists."""
        while self.bones[name].parent is not None:
            name = self.bones[name].parent
        fallback = self.ref.pose.bones[name].rigger.parent_fallback
        return 1 if fallback and fallback != "root" else 0

    # -- bone creation ----------------------------------------------------------
    def new_bone(self, name, head, tail, z_axis=None, parent=None, connect=False,
                 deform=False, collection=None):
        if name in self.edit_bones:
            raise RiggerError(f"Two modules tried to create the bone '{name}'. "
                              "Rename one of the reference bones")
        head, tail = Vector(head), Vector(tail)
        if (tail - head).length < 1e-5:
            tail = head + Vector((0, 0, max(self.scale * 0.01, 1e-3)))
        eb = self.edit_bones.new(name)
        eb.head, eb.tail = head, tail
        if z_axis is not None:
            eb.align_roll(z_axis)
        eb.use_deform = deform
        if parent:
            eb.parent = self.edit_bones[parent]
            eb.use_connect = connect and (eb.parent.tail - eb.head).length < 1e-5
        if collection:
            utils.assign_collection(self.rig.data, eb, collection)
        return eb.name

    # -- main -----------------------------------------------------------------
    def _prepare_rig(self, context):
        name = self.settings.rig_name or "rig"
        rig = self.rig or self.settings.rig
        if rig is None or rig.name not in context.view_layer.objects:
            data = bpy.data.armatures.new(name)
            rig = bpy.data.objects.new(name, data)
            for coll in self.ref.users_collection:
                coll.objects.link(rig)
                break
            else:
                context.collection.objects.link(rig)
        rig.matrix_world = self.ref.matrix_world.copy()
        utils.clear_drivers(rig)
        utils.clear_drivers(rig.data)
        rig.show_in_front = True
        rig.data.display_type = 'OCTAHEDRAL'
        self.rig = rig
        rig.rigger.reference = self.ref
        self.settings.rig = rig

    def generate(self, context):
        from . import poses, rig_data
        self.collect_modules()
        self._prepare_rig(context)
        rig = self.rig
        if rig.mode != 'OBJECT':
            utils.set_mode(rig, 'OBJECT')
        custom = rig_data.capture_custom(rig)

        utils.set_mode(rig, 'EDIT')
        self.edit_bones = rig.data.edit_bones
        for eb in list(self.edit_bones):
            self.edit_bones.remove(eb)
        for coll in list(getattr(rig.data, "collections_all", rig.data.collections)):
            rig.data.collections.remove(coll)
        utils.ensure_collections(rig.data)
        self.new_bone("root", Vector((0, 0, 0)), Vector((0, 0, self.scale * 0.1)),
                      Vector((0, -1, 0)), collection="Root")
        for module in self.modules:
            module.edit()
        for callback in self.post_edit:
            callback()
        poses.edit_hook(self)
        self.edit_bones = None

        bpy.ops.object.mode_set(mode='POSE')
        # Bones re-created with the same name keep their pose channel: start them clean.
        for pb in rig.pose.bones:
            for con in list(pb.constraints):
                pb.constraints.remove(con)
            for key in list(pb.keys()):
                del pb[key]
            pb.custom_shape = None
            pb.custom_shape_scale_xyz = (1.0, 1.0, 1.0)
            pb.custom_shape_translation = (0.0, 0.0, 0.0)
            pb.custom_shape_rotation_euler = (0.0, 0.0, 0.0)
            pb.use_custom_shape_bone_size = True
            pb.lock_location = pb.lock_rotation = pb.lock_scale = (False, False, False)
            pb.lock_rotation_w = False
            pb.color.palette = 'DEFAULT'
            pb.rotation_mode = 'QUATERNION'
            pb.matrix_basis.identity()
        root = rig.pose.bones["root"]
        widgets.apply_widget(root, 'ROOT', 2.2 * self.widget_scale, wire_width=2.5)
        widgets.set_color(root, 'ROOT')
        for module in self.modules:
            module.pose()
        self._pose_space_switches()
        poses.pose_hook(self)
        if self.settings.use_theme:
            rig_data.apply_theme(rig, tuple(self.settings.color_left), tuple(self.settings.color_right),
                                 tuple(self.settings.color_center))
        bpy.ops.object.mode_set(mode='OBJECT')

        for name, visible in utils.RIG_COLLECTIONS:
            coll = rig.data.collections.get(name)
            if coll is not None:
                coll.is_visible = visible
        # Remove empty collections
        for coll in list(rig.data.collections):
            if not coll.bones:
                rig.data.collections.remove(coll)

        controls = [b.name for b in rig.data.bones
                    if not b.name.startswith(("DEF-", "MCH-"))]
        deform = [b.name for b in rig.data.bones if b.use_deform]
        meta = {"version": 2, "controls": controls, "deform": deform, "limbs": [],
                "deform_of": self.deform_of, "faces": [],
                "spaces": {ctrl: labels for ctrl, _s, labels, _f in self.space_switches},
                "fit": self.fit, "generated": [b.name for b in rig.data.bones]}
        for module in self.modules:
            kind = module.meta.get("type")
            if kind in ("arm", "leg"):
                meta["limbs"].append(module.meta)
            elif kind in ("spine", "neck"):
                meta[kind] = module.meta
            elif kind == "face":
                meta["faces"].append(module.meta)
            elif kind:
                meta.setdefault(kind + "s", []).append(module.meta)
        utils.set_meta(rig, meta)
        rig_data.restore_custom(context, rig, custom)
        utils.set_mode(rig, 'OBJECT')
        return rig


def init_scale(context, ref):
    """Apply the reference object's scale so the generated rig has a scale of 1."""
    if not ref.rigger.init_scale:
        return
    context.view_layer.update()
    scale = ref.matrix_world.to_scale()
    if (scale - Vector((1, 1, 1))).length < 1e-6:
        return
    utils.set_mode(ref, 'OBJECT')
    with context.temp_override(selected_editable_objects=[ref], selected_objects=[ref], active_object=ref,
                               object=ref):
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


def generate(context, ref):
    init_scale(context, ref)
    gen = Generator(ref)
    return gen.generate(context)


class RIGGER_OT_generate(bpy.types.Operator):
    """Generate (or update) the control rig from the reference armature"""
    bl_idname = "rigger.generate"
    bl_label = "Generate Rig"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_reference(obj) or (utils.is_rig(obj) and obj.rigger.reference)

    def execute(self, context):
        obj = context.object
        ref = obj if utils.is_reference(obj) else obj.rigger.reference
        if context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            rig = generate(context, ref)
        except RiggerError as exc:
            utils.set_mode(ref, 'OBJECT')
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        ref.hide_set(True)
        utils.set_mode(rig, 'POSE')
        self.report({'INFO'}, f"Generated '{rig.name}' with {len(rig.data.bones)} bones")
        return {'FINISHED'}


class RIGGER_OT_edit_reference(bpy.types.Operator):
    """Show the reference armature and enter Edit Mode on it"""
    bl_idname = "rigger.edit_reference"
    bl_label = "Edit Reference"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_rig(obj) and obj.rigger.reference is not None

    def execute(self, context):
        ref = context.object.rigger.reference
        utils.set_mode(ref, 'EDIT')
        return {'FINISHED'}


classes = (RIGGER_OT_generate, RIGGER_OT_edit_reference)
