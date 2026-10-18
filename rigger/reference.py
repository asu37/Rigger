# SPDX-License-Identifier: GPL-3.0-or-later
"""Operators that create and edit reference armatures."""

import bpy
from bpy.props import BoolProperty, EnumProperty
from mathutils import Vector

from . import templates, utils
from .modules import MODULES


def _template_items(_self, _context):
    return [(key, label, desc) for key, (label, desc, _fn) in templates.TEMPLATES.items()]


class RIGGER_OT_add_reference(bpy.types.Operator):
    """Add a reference armature from a template. Edit its bones to fit your character"""
    bl_idname = "rigger.add_reference"
    bl_label = "Add Reference"
    bl_options = {'REGISTER', 'UNDO'}

    template: EnumProperty(name="Template", items=_template_items)

    def execute(self, context):
        specs = templates.template_specs(self.template)
        obj = templates.create_reference(context, specs, name="reference",
                                         location=context.scene.cursor.location.copy())
        obj.rigger.multiped = self.template in ('QUADRUPED', 'HORSE')
        utils.set_mode(obj, 'EDIT')
        return {'FINISHED'}


class RIGGER_OT_add_module(bpy.types.Operator):
    """Add a module to the reference armature, parented to the active bone"""
    bl_idname = "rigger.add_module"
    bl_label = "Add Module"
    bl_options = {'REGISTER', 'UNDO'}

    module: EnumProperty(name="Module", items=[(k, c.label, c.description) for k, c in MODULES.items()])
    side: EnumProperty(name="Side", default='L', items=[
        ('L', "Left", "Left side (.L)"),
        ('R', "Right", "Right side (.R)"),
        ('BOTH', "Both", "Left and right"),
        ('C', "Center", "No side suffix"),
    ])

    @classmethod
    def poll(cls, context):
        return utils.is_reference(context.object)

    def _chunk(self):
        if self.module in templates.CHUNK_FUNCTIONS:
            return templates.CHUNK_FUNCTIONS[self.module]()
        template, names = templates.MODULE_CHUNKS[self.module]
        specs = {s.name: s for s in templates.template_specs(template, options=None)}
        return [specs[n] for n in names]

    def execute(self, context):
        obj = context.object
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        ebones = obj.data.edit_bones
        active = ebones.active
        cursor = obj.matrix_world.inverted() @ context.scene.cursor.location
        anchor = active.tail.copy() if active else cursor
        if active and (context.scene.cursor.location - obj.matrix_world.translation).length > 1e-6:
            anchor = cursor

        chunk = self._chunk()
        offset = anchor - chunk[0].head
        sides = {'L': [".L"], 'R': [".R"], 'BOTH': [".L", ".R"], 'C': [""]}[self.side]
        if self.module in {'FACE', 'SKIRT'}:
            # The face is symmetrical by itself: keep its own names, both sides.
            chunk = templates.mirrored(chunk)
            sides = [None]
        created = []
        for suffix in sides:
            existing = set(ebones.keys())
            rename = {}
            for spec in chunk:
                stem = utils.split_side(spec.name)[0]
                wanted = spec.name if suffix is None else stem + suffix
                rename[spec.name] = utils.unique_name(existing, wanted)
                existing.add(rename[spec.name])
            mirror = suffix == ".R"
            for spec in chunk:
                head, tail = spec.head + offset, spec.tail + offset
                z = spec.z.copy() if spec.z is not None else None
                if mirror:
                    head, tail = Vector((-head.x, head.y, head.z)), Vector((-tail.x, tail.y, tail.z))
                    if z is not None:
                        z.x = -z.x
                eb = ebones.new(rename[spec.name])
                eb.head, eb.tail = head, tail
                if z is not None:
                    eb.align_roll(z)
                if spec.parent in rename:
                    eb.parent = ebones[rename[spec.parent]]
                    eb.use_connect = spec.connect
                elif active is not None:
                    parent_name = active.name
                    if mirror and utils.mirror_name(parent_name) in ebones:
                        parent_name = utils.mirror_name(parent_name)
                    eb.parent = ebones[parent_name]
                created.append((rename[spec.name], spec))
        bpy.ops.object.mode_set(mode='OBJECT')
        roots = {rename_root for rename_root in (c[0] for c in created)
                 if obj.data.bones[rename_root].parent is None
                 or obj.data.bones[rename_root].parent.name not in {c[0] for c in created}}
        for name, spec in created:
            module = self.module if name in roots else spec.module
            templates.apply_settings(obj.pose.bones[name], module, spec.params)
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"Added {MODULES[self.module].label}: {len(created)} bones")
        return {'FINISHED'}


class RIGGER_OT_mirror_reference(bpy.types.Operator):
    """Copy every left (.L) reference bone and its module settings to the right side (.R)"""
    bl_idname = "rigger.mirror_reference"
    bl_label = "Mirror Left to Right"
    bl_options = {'REGISTER', 'UNDO'}

    right_to_left: BoolProperty(name="Right to Left", default=False)

    @classmethod
    def poll(cls, context):
        return utils.is_reference(context.object)

    def execute(self, context):
        obj = context.object
        src_letter = "R" if self.right_to_left else "L"
        mode = obj.mode
        utils.set_mode(obj, 'EDIT')
        ebones = obj.data.edit_bones
        sources = [eb for eb in ebones if utils.side_letter(eb.name) == src_letter]
        pairs = []
        for eb in sources:
            name = utils.mirror_name(eb.name)
            dst = ebones.get(name) or ebones.new(name)
            dst.head = Vector((-eb.head.x, eb.head.y, eb.head.z))
            dst.tail = Vector((-eb.tail.x, eb.tail.y, eb.tail.z))
            z = eb.z_axis.copy()
            z.x = -z.x
            dst.align_roll(z)
            pairs.append((eb.name, name))
        for src_name, dst_name in pairs:
            src, dst = ebones[src_name], ebones[dst_name]
            if src.parent:
                parent = utils.mirror_name(src.parent.name)
                dst.parent = ebones.get(parent) or src.parent
                dst.use_connect = src.use_connect
            else:
                dst.parent = None
        bpy.ops.object.mode_set(mode='OBJECT')
        from .properties import copy_settings
        for src_name, dst_name in pairs:
            copy_settings(obj.pose.bones[src_name].rigger, obj.pose.bones[dst_name].rigger)
        bpy.ops.object.mode_set(mode=mode if mode in {'EDIT', 'POSE'} else 'OBJECT')
        self.report({'INFO'}, f"Mirrored {len(pairs)} bones")
        return {'FINISHED'}


class RIGGER_OT_add_feathers(bpy.types.Operator):
    """Add wing feathers along the active arm chain (upper arm, forearm, hand) of the reference"""
    bl_idname = "rigger.add_feathers"
    bl_label = "Add Wing Feathers"
    bl_options = {'REGISTER', 'UNDO'}

    arm_feathers: bpy.props.IntProperty(name="Arm Feathers", default=4, min=0, max=20)
    forearm_feathers: bpy.props.IntProperty(name="Forearm Feathers", default=5, min=0, max=20)
    hand_feathers: bpy.props.IntProperty(name="Hand Feathers", default=3, min=0, max=20)
    segments: bpy.props.IntProperty(name="Feather Segments", default=2, min=1, max=6)
    layers: bpy.props.IntProperty(name="Feather Layers", default=1, min=1, max=4)
    length: bpy.props.FloatProperty(name="Feather Length", default=1.0, min=0.1, max=5.0,
                                    description="Relative to the wing segment length")
    mirror: BoolProperty(name="Both Sides", default=True)

    @classmethod
    def poll(cls, context):
        return utils.is_reference(context.object)

    def execute(self, context):
        obj = context.object
        utils.set_mode(obj, 'EDIT')
        ebones = obj.data.edit_bones
        active = ebones.active
        if active is None:
            self.report({'ERROR'}, "Select a bone of the wing (arm) chain")
            return {'CANCELLED'}
        active_name = active.name
        side = utils.split_side(active_name)[1]
        names = [n + side for n in ("upper_arm", "forearm", "hand")]
        if not all(n in ebones for n in names):
            # Fall back to the active bone's chain
            chain = [active]
            while chain[-1].children:
                chain.append(chain[-1].children[0])
            names = [b.name for b in chain[:3]]
        counts = (self.arm_feathers, self.forearm_feathers, self.hand_feathers)
        created = []
        for seg_name, count in zip(names, counts):
            seg = ebones[seg_name]
            direction = (seg.tail - seg.head).normalized()
            # Feathers point backwards (+Y) and slightly outward, layered upward
            back = Vector((0, 1, 0))
            back = (back - direction * direction.dot(back)).normalized()
            for layer in range(self.layers):
                for i in range(count):
                    t = (i + 0.5) / max(count, 1)
                    root = seg.head.lerp(seg.tail, t) + Vector((0, 0, 0.004 * layer))
                    tip_dir = (back + direction * (t * 0.6 if seg_name == names[-1] else 0.15)).normalized()
                    length = seg.length * self.length * (0.8 + 0.4 * t) / (1 + 0.3 * layer)
                    prev, parent = root, seg.name
                    stem = f"feather_{utils.split_side(seg_name)[0]}_{layer + 1}{i + 1:02d}"
                    for j in range(self.segments):
                        nxt = prev + tip_dir * length / self.segments
                        name = utils.unique_name(set(ebones.keys()), f"{stem}.{j + 1:02d}{side}")
                        eb = ebones.new(name)
                        eb.head, eb.tail = prev, nxt
                        eb.align_roll(Vector((0, 0, 1)))
                        eb.parent = ebones[parent]
                        eb.use_connect = j > 0
                        created.append((name, j == 0))
                        prev, parent = nxt, name
        bpy.ops.object.mode_set(mode='OBJECT')
        for name, first in created:
            settings = obj.pose.bones[name].rigger
            settings.module = 'CHAIN' if first else 'NONE'
            settings.curl = False
            settings.collection = "Fingers"
        if self.mirror and side:
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.rigger.mirror_reference(right_to_left=utils.side_letter(active_name) == "R")
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, f"Added {len(created)} feather bones")
        return {'FINISHED'}


def module_root(ref, name):
    """First bone at or above 'name' that starts a module."""
    bone = ref.data.bones.get(name)
    while bone is not None:
        if ref.pose.bones[bone.name].rigger.module != 'NONE':
            return bone.name
        bone = bone.parent
    return None


def module_bones(ref, root):
    """Root plus its descendants that do not start a module of their own (parents first)."""
    result, queue = [root], list(ref.data.bones[root].children)
    while queue:
        bone = queue.pop(0)
        if ref.pose.bones[bone.name].rigger.module != 'NONE':
            continue
        result.append(bone.name)
        queue.extend(bone.children)
    return result


def _active_name(obj):
    if obj.mode == 'EDIT':
        return obj.data.edit_bones.active.name if obj.data.edit_bones.active else None
    return obj.data.bones.active.name if obj.data.bones.active else None


class RIGGER_OT_duplicate_limb(bpy.types.Operator):
    """Duplicate the module of the active reference bone (optionally mirrored to the other side)"""
    bl_idname = "rigger.duplicate_limb"
    bl_label = "Duplicate Limb"
    bl_options = {'REGISTER', 'UNDO'}

    mirror: BoolProperty(name="Mirror", default=False, description="Mirror the copy to the other side")
    offset: bpy.props.FloatVectorProperty(name="Offset", size=3, default=(0.0, 0.0, 0.0), subtype='TRANSLATION')

    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_reference(obj) and _active_name(obj) is not None

    def execute(self, context):
        from .properties import copy_settings
        obj = context.object
        mode = obj.mode
        root = module_root(obj, _active_name(obj))
        if root is None:
            self.report({'ERROR'}, "The active bone is not part of a module")
            return {'CANCELLED'}
        bones = module_bones(obj, root)
        utils.set_mode(obj, 'EDIT')
        ebones = obj.data.edit_bones
        existing = set(ebones.keys())
        rename = {}
        for name in bones:
            wanted = utils.mirror_name(name) if self.mirror else name
            rename[name] = utils.unique_name(existing, wanted)
            existing.add(rename[name])
        offset = Vector(self.offset)
        for name in bones:
            src = ebones[name]
            dst = ebones.new(rename[name])
            head, tail, z = src.head.copy(), src.tail.copy(), src.z_axis.copy()
            if self.mirror:
                head.x, tail.x, z.x = -head.x, -tail.x, -z.x
            dst.head, dst.tail = head + offset, tail + offset
            dst.align_roll(z)
        for name in bones:
            src, dst = ebones[name], ebones[rename[name]]
            if src.parent is None:
                continue
            parent = rename.get(src.parent.name)
            if parent is None:
                parent = utils.mirror_name(src.parent.name) if self.mirror else src.parent.name
                parent = parent if parent in ebones else src.parent.name
            dst.parent = ebones[parent]
            dst.use_connect = src.use_connect
        bpy.ops.object.mode_set(mode='OBJECT')
        for name in bones:
            copy_settings(obj.pose.bones[name].rigger, obj.pose.bones[rename[name]].rigger)
        bpy.ops.object.mode_set(mode=mode if mode in ('EDIT', 'POSE') else 'OBJECT')
        self.report({'INFO'}, f"Duplicated {len(bones)} bones")
        return {'FINISHED'}


class RIGGER_OT_remove_limb(bpy.types.Operator):
    """Delete the module of the active reference bone. Child modules are kept and re-parented"""
    bl_idname = "rigger.remove_limb"
    bl_label = "Remove Limb"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_reference(obj) and _active_name(obj) is not None

    def execute(self, context):
        obj = context.object
        mode = obj.mode
        root = module_root(obj, _active_name(obj))
        if root is None:
            return {'CANCELLED'}
        bones = set(module_bones(obj, root))
        utils.set_mode(obj, 'EDIT')
        ebones = obj.data.edit_bones
        parent = ebones[root].parent
        for eb in list(ebones):
            if eb.name not in bones and eb.parent is not None and eb.parent.name in bones:
                eb.use_connect = False
                eb.parent = parent
        for name in bones:
            ebones.remove(ebones[name])
        bpy.ops.object.mode_set(mode=mode if mode in ('EDIT', 'POSE') else 'OBJECT')
        self.report({'INFO'}, f"Removed {len(bones)} bones")
        return {'FINISHED'}


class RIGGER_OT_turn_model(bpy.types.Operator):
    """Turn the selected meshes around the vertical axis by 90 degree steps (characters must face -Y)"""
    bl_idname = "rigger.turn_model"
    bl_label = "Turn Model"
    bl_options = {'REGISTER', 'UNDO'}

    angle: bpy.props.FloatProperty(name="Angle", default=90.0, subtype='NONE')

    @classmethod
    def poll(cls, context):
        return any(o.type == 'MESH' for o in context.selected_objects)

    def execute(self, context):
        import math
        from mathutils import Matrix
        meshes = [o for o in context.selected_objects if o.type == 'MESH']
        pivot = sum((o.matrix_world.translation for o in meshes), Vector()) / len(meshes)
        pivot.z = 0.0
        turn = Matrix.Translation(pivot) @ Matrix.Rotation(math.radians(self.angle), 4, 'Z') @ \
            Matrix.Translation(-pivot)
        for obj in meshes:
            if obj.parent in meshes:
                continue
            obj.matrix_world = turn @ obj.matrix_world
        return {'FINISHED'}


# -- IK direction overlay -------------------------------------------------------------------------
_ik_handle = None


def ik_lines(ref):
    """(joint, pole point) pairs for every arm / leg module of a reference, in world space."""
    lines = []
    bones = ref.data.edit_bones if ref.mode == 'EDIT' else ref.data.bones
    get = (lambda b: (b.head, b.tail)) if ref.mode == 'EDIT' else (lambda b: (b.head_local, b.tail_local))
    for pb in ref.pose.bones:
        module = pb.rigger.module
        if module not in ('ARM', 'LEG'):
            continue
        chain = [pb.name]
        bone = bones.get(pb.name)
        while bone is not None and len(chain) < 4:
            kids = [c for c in bone.children if ref.pose.bones[c.name].rigger.module == 'NONE']
            if not kids:
                break
            bone = next((c for c in kids if c.use_connect), kids[0])
            chain.append(bone.name)
        if module == 'ARM' and len(chain) == 4:
            chain = chain[1:]
        if len(chain) < 3:
            continue
        (a, _), (b, c) = get(bones[chain[0]]), get(bones[chain[1]])
        fallback = Vector((0, 1, 0)) if module == 'ARM' else Vector((0, -1, 0))
        bend = utils.bend_direction(a, b, c, fallback)
        length = (b - a).length + (c - b).length
        mw = ref.matrix_world
        lines.append((mw @ b, mw @ (b + bend * length * pb.rigger.pole_distance)))
    return lines


def _draw_ik_lines():
    import gpu
    from gpu_extras.batch import batch_for_shader
    obj = bpy.context.object
    if not utils.is_reference(obj):
        return
    lines = ik_lines(obj)
    if not lines:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    coords = [p for line in lines for p in line]
    gpu.state.line_width_set(2.0)
    gpu.state.point_size_set(8.0)
    shader.uniform_float("color", (1.0, 0.85, 0.1, 1.0))
    batch_for_shader(shader, 'LINES', {"pos": coords}).draw(shader)
    batch_for_shader(shader, 'POINTS', {"pos": [line[1] for line in lines]}).draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.point_size_set(1.0)


class RIGGER_OT_show_ik_lines(bpy.types.Operator):
    """Show or hide the IK direction of arms and legs: a line from the elbow/knee to the future pole"""
    bl_idname = "rigger.show_ik_lines"
    bl_label = "Show IK Directions"

    def execute(self, context):
        global _ik_handle
        if _ik_handle is None:
            _ik_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_ik_lines, (), 'WINDOW', 'POST_VIEW')
        else:
            bpy.types.SpaceView3D.draw_handler_remove(_ik_handle, 'WINDOW')
            _ik_handle = None
        for area in context.screen.areas if context.screen else []:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        return {'FINISHED'}


def ik_lines_shown():
    return _ik_handle is not None


def unregister_overlay():
    global _ik_handle
    if _ik_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_ik_handle, 'WINDOW')
        _ik_handle = None


classes = (RIGGER_OT_add_reference, RIGGER_OT_add_module, RIGGER_OT_mirror_reference,
           RIGGER_OT_add_feathers, RIGGER_OT_duplicate_limb, RIGGER_OT_remove_limb, RIGGER_OT_turn_model,
           RIGGER_OT_show_ik_lines)
