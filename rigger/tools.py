# SPDX-License-Identifier: GPL-3.0-or-later
"""Rigging tools: shape editor, corrective shape keys, apply pose as rest pose,
finger poses and pinning."""

import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from mathutils import Euler, Matrix

from . import utils

SHAPE_EDIT_PREFIX = "SHAPE-EDIT-"


# ---------------------------------------------------------------------------
# Shape editor
# ---------------------------------------------------------------------------

def _shape_space(rig, pb):
    """World matrix of a control's custom-shape space (offset, rotation and scale included)."""
    length = pb.bone.length if pb.use_custom_shape_bone_size else 1.0
    sx, sy, sz = pb.custom_shape_scale_xyz
    scale = Matrix.Diagonal((sx * length, sy * length, sz * length, 1.0))
    offset = Matrix.Translation(pb.custom_shape_translation) @ \
        Euler(pb.custom_shape_rotation_euler).to_matrix().to_4x4()
    return rig.matrix_world @ pb.matrix @ offset @ scale


class RIGGER_OT_edit_shape(bpy.types.Operator):
    """Edit the shape of the active control as a mesh. Use Apply Shape when done"""
    bl_idname = "rigger.edit_shape"
    bl_label = "Edit Shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (utils.is_rig(obj) and obj.mode == 'POSE' and context.active_pose_bone is not None
                and context.active_pose_bone.custom_shape is not None)

    def execute(self, context):
        rig, pb = context.object, context.active_pose_bone
        mesh = pb.custom_shape.data.copy()
        name = SHAPE_EDIT_PREFIX + pb.name
        old = bpy.data.objects.get(name)
        if old is not None:
            bpy.data.objects.remove(old)
        obj = bpy.data.objects.new(name, mesh)
        context.scene.collection.objects.link(obj)
        obj.matrix_world = _shape_space(rig, pb)
        obj["rigger_shape_rig"] = rig.name
        obj["rigger_shape_bone"] = pb.name
        obj.show_in_front = True
        utils.set_mode(obj, 'EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        return {'FINISHED'}


class RIGGER_OT_apply_shape(bpy.types.Operator):
    """Use the edited mesh as the control's shape (and the mirrored control's)"""
    bl_idname = "rigger.apply_shape"
    bl_label = "Apply Shape"
    bl_options = {'REGISTER', 'UNDO'}

    mirror: BoolProperty(name="Mirror", default=True, description="Also update the other side's control")

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.name.startswith(SHAPE_EDIT_PREFIX)

    def execute(self, context):
        obj = context.object
        bpy.ops.object.mode_set(mode='OBJECT')
        rig = bpy.data.objects.get(obj.get("rigger_shape_rig", ""))
        bone = obj.get("rigger_shape_bone", "")
        if rig is None or bone not in rig.pose.bones:
            self.report({'ERROR'}, "The rig or bone of this shape no longer exists")
            return {'CANCELLED'}
        from .widgets import _make_object
        pb = rig.pose.bones[bone]
        # The edit object lives in shape space, so its mesh is already in shape coordinates,
        # but it may have been moved: bake its transform relative to the shape space.
        to_shape = _shape_space(rig, pb).inverted() @ obj.matrix_world
        verts = [tuple(to_shape @ v.co) for v in obj.data.vertices]
        edges = [tuple(e.vertices) for e in obj.data.edges]
        pb.custom_shape = _make_object(f"WGT-{rig.name}_{bone}", verts, edges)
        transforms = (tuple(pb.custom_shape_translation), tuple(pb.custom_shape_rotation_euler))
        targets = 1
        other = utils.mirror_name(bone)
        if self.mirror and other != bone and other in rig.pose.bones:
            opb = rig.pose.bones[other]
            opb.custom_shape = _make_object(f"WGT-{rig.name}_{other}",
                                            [(-x, y, z) for x, y, z in verts], edges)
            opb.custom_shape_scale_xyz = pb.custom_shape_scale_xyz
            t, r = transforms
            opb.custom_shape_translation = (-t[0], t[1], t[2])
            opb.custom_shape_rotation_euler = (r[0], -r[1], -r[2])
            targets += 1
        mesh = obj.data
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
        utils.set_mode(rig, 'POSE')
        rig.data.bones.active = rig.data.bones[bone]
        self.report({'INFO'}, f"Updated {targets} control shape(s)")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Corrective shape keys
# ---------------------------------------------------------------------------

def _deform_pair(rig, pose_bone):
    """(child DEF bone, parent DEF bone) for a control, or None."""
    meta = utils.get_meta(rig)
    name = pose_bone.name
    candidates = [name if name.startswith("DEF-") else None]
    for limb in meta.get("limbs", []):
        for group in (limb["fk"], limb["ik"]):
            if name in group:
                candidates.append(limb["deform"][group.index(name)])
    candidates.append("DEF-" + name)
    stem, suffix = utils.split_side(name)
    for part in ("_fk", "_ik"):
        if stem.endswith(part):
            candidates.append("DEF-" + stem[: -len(part)] + suffix)
    for cand in candidates:
        if cand and cand in rig.data.bones:
            bone = rig.data.bones[cand]
            parent = bone.parent
            while parent is not None and not parent.use_deform:
                parent = parent.parent
            if parent is not None:
                return bone.name, parent.name
    return None


def rotation_angle(rig, child, parent, rest=False):
    if rest:
        a = rig.data.bones[child].matrix_local.to_quaternion()
        b = rig.data.bones[parent].matrix_local.to_quaternion()
    else:
        a = rig.pose.bones[child].matrix.to_quaternion()
        b = rig.pose.bones[parent].matrix.to_quaternion()
    return a.rotation_difference(b).angle


class RIGGER_OT_corrective_shape(bpy.types.Operator):
    """Create a corrective shape key on the selected meshes, driven by the active control's
    joint: 0 at the rest angle, 1 at the current pose. Sculpt the key in this pose"""
    bl_idname = "rigger.corrective_shape"
    bl_label = "Add Corrective Shape Key"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(name="Name", default="")

    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_rig(obj) and obj.mode == 'POSE' and context.active_pose_bone is not None

    def execute(self, context):
        rig = context.object
        pair = _deform_pair(rig, context.active_pose_bone)
        if pair is None:
            self.report({'ERROR'}, "No deforming joint found for this control")
            return {'CANCELLED'}
        child, parent = pair
        rest = rotation_angle(rig, child, parent, rest=True)
        now = rotation_angle(rig, child, parent)
        if abs(now - rest) < math.radians(5):
            self.report({'ERROR'}, "Bend the joint first (at least 5 degrees)")
            return {'CANCELLED'}
        meshes = [o for o in context.scene.objects if o.type == 'MESH' and utils.find_rig_for(o) == rig
                  and child in o.vertex_groups]
        if not meshes:
            self.report({'ERROR'}, "No mesh skinned to this joint")
            return {'CANCELLED'}
        key_name = self.name or f"corrective_{child[4:]}"
        for obj in meshes:
            if obj.data.shape_keys is None:
                obj.shape_key_add(name="Basis", from_mix=False)
            key = obj.data.shape_keys.key_blocks.get(key_name) or obj.shape_key_add(name=key_name, from_mix=False)
            key.slider_max = 1.0
            fcurve = key.driver_add("value")
            driver = fcurve.driver
            driver.type = 'SCRIPTED'
            for var in list(driver.variables):
                driver.variables.remove(var)
            var = driver.variables.new()
            var.name = "angle"
            var.type = 'ROTATION_DIFF'
            var.targets[0].id = rig
            var.targets[0].bone_target = child
            var.targets[1].id = rig
            var.targets[1].bone_target = parent
            driver.expression = f"max(0.0, min(1.0, (angle - {rest:.5f}) / {now - rest:.5f}))"
            obj.active_shape_key_index = list(obj.data.shape_keys.key_blocks).index(key)
            obj.show_only_shape_key = False
            obj.use_shape_key_edit_mode = True
        self.report({'INFO'}, f"Added '{key_name}' on {len(meshes)} mesh(es). Edit it in this pose "
                              "(Edit Mode with the armature modifier shown on cage)")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Shape key drivers
# ---------------------------------------------------------------------------

CHANNELS = [(c, c.replace("_", " ").title(), "") for c in
            ('LOC_X', 'LOC_Y', 'LOC_Z', 'ROT_X', 'ROT_Y', 'ROT_Z', 'SCALE_X', 'SCALE_Y', 'SCALE_Z')]


def _active_key(context):
    obj = context.object
    if obj is None or obj.type != 'MESH' or obj.data.shape_keys is None:
        return None, None
    key = obj.active_shape_key
    if key is None or key == obj.data.shape_keys.reference_key:
        return obj, None
    return obj, key


def _key_fcurve(obj, key):
    anim = obj.data.shape_keys.animation_data
    if anim is None:
        return None
    return anim.drivers.find(f'key_blocks["{key.name}"].value')


def _rig_for_mesh(context, obj):
    rig = utils.find_rig_for(obj)
    if rig is None:
        rig = next((o for o in context.selected_objects if o.type == 'ARMATURE'), None)
    return rig


class RIGGER_OT_shape_key_driver(bpy.types.Operator):
    """Drive the active shape key of the active mesh with a bone's location, rotation or scale"""
    bl_idname = "rigger.shape_key_driver"
    bl_label = "Create Shape Key Driver"
    bl_options = {'REGISTER', 'UNDO'}

    bone: StringProperty(name="Bone")
    channel: EnumProperty(name="Channel", items=CHANNELS, default='LOC_Z')
    factor: FloatProperty(name="Factor", default=1.0, description="The shape key value is the channel times this")

    @classmethod
    def poll(cls, context):
        return _active_key(context)[1] is not None

    def execute(self, context):
        obj, key = _active_key(context)
        rig = _rig_for_mesh(context, obj)
        if rig is None or self.bone not in rig.pose.bones:
            self.report({'ERROR'}, "Bind the mesh to a rig (or select the rig) and enter a bone of it")
            return {'CANCELLED'}
        fcurve = key.driver_add("value")
        driver = fcurve.driver
        driver.type = 'SCRIPTED'
        for var in list(driver.variables):
            driver.variables.remove(var)
        var = driver.variables.new()
        var.name = "var"
        var.type = 'TRANSFORMS'
        var.targets[0].id = rig
        var.targets[0].bone_target = self.bone
        var.targets[0].transform_type = self.channel
        var.targets[0].transform_space = 'LOCAL_SPACE'
        driver.expression = "var" if abs(self.factor - 1.0) < 1e-9 else f"var * {self.factor:g}"
        self.report({'INFO'}, f"'{key.name}' is driven by {self.bone} {self.channel}")
        return {'FINISHED'}


def _driver_input(fcurve):
    """Current value of the driver expression, before the F-curve keys remap it."""
    import math as _math
    driver = fcurve.driver
    values = {}
    for var in driver.variables:
        t = var.targets
        if var.type == 'TRANSFORMS' and t[0].id is not None:
            pb = t[0].id.pose.bones.get(t[0].bone_target)
            if pb is None:
                return None
            loc, rot, scale = pb.matrix_basis.decompose()
            euler = rot.to_euler()
            values[var.name] = {'LOC_X': loc.x, 'LOC_Y': loc.y, 'LOC_Z': loc.z, 'ROT_X': euler.x,
                                'ROT_Y': euler.y, 'ROT_Z': euler.z, 'SCALE_X': scale.x, 'SCALE_Y': scale.y,
                                'SCALE_Z': scale.z}.get(t[0].transform_type, 0.0)
        elif var.type == 'ROTATION_DIFF' and t[0].id is not None and t[1].id is not None:
            a = t[0].id.pose.bones[t[0].bone_target].matrix.to_quaternion()
            b = t[1].id.pose.bones[t[1].bone_target].matrix.to_quaternion()
            values[var.name] = a.rotation_difference(b).angle
        elif var.type == 'SINGLE_PROP' and t[0].id is not None:
            try:
                values[var.name] = float(t[0].id.path_resolve(t[0].data_path))
            except (ValueError, TypeError):
                return None
        else:
            return None
    names = {k: getattr(_math, k) for k in ("sin", "cos", "tan", "sqrt", "pi", "exp", "radians", "degrees")}
    names.update(max=max, min=min, abs=abs)
    try:
        return float(eval(driver.expression, {"__builtins__": {}}, {**names, **values}))  # noqa: S307
    except Exception:  # noqa: BLE001 - any expression error means we cannot evaluate it
        return None


class RIGGER_OT_shape_key_driver_key(bpy.types.Operator):
    """Make the active shape key reach 0 or 1 at the current bone pose (or reset its curve)"""
    bl_idname = "rigger.shape_key_driver_key"
    bl_label = "Set Driver Key"
    bl_options = {'REGISTER', 'UNDO'}

    value: EnumProperty(name="Value", items=[('0', "0", ""), ('1', "1", ""), ('RESET', "Reset", "")])

    @classmethod
    def poll(cls, context):
        obj, key = _active_key(context)
        return key is not None and _key_fcurve(obj, key) is not None

    def execute(self, context):
        obj, key = _active_key(context)
        fcurve = _key_fcurve(obj, key)
        if self.value == 'RESET':
            fcurve.keyframe_points.clear()
            return {'FINISHED'}
        x = _driver_input(fcurve)
        if x is None:
            self.report({'ERROR'}, "Cannot evaluate this driver")
            return {'CANCELLED'}
        for mod in list(fcurve.modifiers):
            fcurve.modifiers.remove(mod)
        for point in list(fcurve.keyframe_points):
            if abs(point.co.x - x) < 1e-6:
                fcurve.keyframe_points.remove(point)
        point = fcurve.keyframe_points.insert(x, float(self.value))
        point.interpolation = 'LINEAR'
        fcurve.extrapolation = 'CONSTANT'
        fcurve.update()
        obj.data.shape_keys.update_tag()
        return {'FINISHED'}


class RIGGER_OT_remove_shape_driver(bpy.types.Operator):
    """Remove the driver of the active shape key (corrective or bone driver)"""
    bl_idname = "rigger.remove_shape_driver"
    bl_label = "Remove Shape Key Driver"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj, key = _active_key(context)
        return key is not None and _key_fcurve(obj, key) is not None

    def execute(self, context):
        _obj, key = _active_key(context)
        key.driver_remove("value")
        key.value = 0.0
        return {'FINISHED'}


def mirror_key_name(name):
    for a, b in (("_L", "_R"), (".L", ".R"), ("_l", "_r"), (".l", ".r"), ("Left", "Right"), ("left", "right")):
        if name.endswith(a):
            return name[: -len(a)] + b
        if name.endswith(b):
            return name[: -len(b)] + a
    return None


def mirror_shape_keys(context, obj):
    """Create the other side's shape key for every sided key that has none, with mirrored drivers."""
    keys = obj.data.shape_keys
    created = []
    for key in list(keys.key_blocks):
        other = mirror_key_name(key.name)
        if other is None or other in keys.key_blocks:
            continue
        new = obj.shape_key_add(name=other, from_mix=False)
        coords = [0.0] * (len(key.data) * 3)
        key.data.foreach_get("co", coords)
        new.data.foreach_set("co", coords)
        new.slider_min, new.slider_max = key.slider_min, key.slider_max
        new.relative_key = key.relative_key
        index = list(keys.key_blocks).index(new)
        obj.active_shape_key_index = index
        utils.set_mode(obj, 'OBJECT')
        with context.temp_override(object=obj, active_object=obj):
            bpy.ops.object.shape_key_mirror(use_topology=False)
        fcurve = _key_fcurve(obj, key)
        if fcurve is not None:
            from .rig_data import restore_driver, serialize_driver
            data = serialize_driver(None, fcurve)
            data["path"] = f'key_blocks["{other}"].value'
            flip = False
            for var in data["variables"]:
                for t in var["targets"]:
                    if t["bone"]:
                        t["bone"] = utils.mirror_name(t["bone"])
                    if var["type"] == 'TRANSFORMS' and t["transform_type"] in ('LOC_X', 'ROT_Y', 'ROT_Z'):
                        flip = True
            rig_obj = next((t.id for v in fcurve.driver.variables for t in v.targets if t.id), None)
            if flip:
                data["expression"] = data["expression"].replace("var", "(-var)")
            for var in data["variables"]:
                for t in var["targets"]:
                    if t["id"] is None and rig_obj is not None:
                        t["id"] = rig_obj.name
            new_fc = restore_driver(rig_obj, keys, data)
            if new_fc is not None:
                for point in fcurve.keyframe_points:
                    new_fc.keyframe_points.insert(point.co.x * (-1.0 if flip else 1.0), point.co.y)
        created.append(other)
    return created


class RIGGER_OT_mirror_shape_keys(bpy.types.Operator):
    """Create the missing other-side shape keys (_L <-> _R) by mirroring them, with mirrored drivers"""
    bl_idname = "rigger.mirror_shape_keys"
    bl_label = "Mirror All Shape Keys"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        return obj is not None and obj.type == 'MESH' and obj.data.shape_keys is not None

    def execute(self, context):
        created = mirror_shape_keys(context, context.object)
        self.report({'INFO'}, f"Created {len(created)} shape keys: {', '.join(created)}")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Apply pose as rest pose
# ---------------------------------------------------------------------------

class RIGGER_OT_apply_rest_pose(bpy.types.Operator):
    """Make the current pose the new rest pose: meshes keep their posed shape, the reference
    follows the pose and the rig is regenerated"""
    bl_idname = "rigger.apply_rest_pose"
    bl_label = "Apply Pose as Rest Pose"
    bl_options = {'REGISTER', 'UNDO'}

    preserve_anim: BoolProperty(
        name="Preserve Animation", default=True,
        description="Convert the keyframes of every action of the rig so the animation looks the same "
                    "with the new rest pose")
    every_frame: BoolProperty(name="Key Every Frame", default=False,
                              description="Off: convert the existing keyframes only")

    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_rig(obj) and obj.rigger.reference is not None

    def execute(self, context):
        rig = context.object
        ref = rig.rigger.reference
        meta = utils.get_meta(rig)
        deform_of = meta.get("deform_of", {})
        if context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        context.view_layer.update()
        recorded = record_animation(context, rig, self.every_frame) if self.preserve_anim else None
        meshes = [o for o in context.scene.objects if o.type == 'MESH' and utils.find_rig_for(o) == rig]
        skipped = []
        for obj in meshes:
            if obj.data.shape_keys is not None:
                skipped.append(obj.name)
                continue
            mod = next(m for m in obj.modifiers if m.type == 'ARMATURE' and m.object == rig)
            utils.set_mode(obj, 'OBJECT')
            with context.temp_override(object=obj, active_object=obj):
                bpy.ops.object.modifier_apply(modifier=mod.name)
        if skipped:
            self.report({'WARNING'}, "Meshes with shape keys keep their old shape: " + ", ".join(skipped))
        # Move the reference bones onto the posed deform bones.
        posed = {}
        for ref_name, def_name in deform_of.items():
            pb = rig.pose.bones.get(def_name)
            if pb is not None:
                posed[ref_name] = pb.matrix.copy()
        utils.set_mode(ref, 'EDIT')
        to_ref = ref.matrix_world.inverted() @ rig.matrix_world
        for ref_name, matrix in posed.items():
            eb = ref.data.edit_bones.get(ref_name)
            if eb is None:
                continue
            length = eb.length
            world = to_ref @ matrix
            y_axis = world.to_3x3().col[1].normalized()
            eb.use_connect = False
            eb.head = world.translation
            eb.tail = world.translation + y_axis * length
            eb.align_roll(world.to_3x3().col[2])
        bpy.ops.object.mode_set(mode='OBJECT')
        from .generate import generate
        rig = generate(context, ref)
        from .skin import ensure_armature_modifier
        for obj in meshes:
            if obj.name not in skipped:
                ensure_armature_modifier(obj, rig)
        ref.hide_set(True)
        utils.set_mode(rig, 'POSE')
        if recorded and recorded["actions"]:
            rekey_animation(context, rig, recorded)
        self.report({'INFO'}, "Applied the pose as the new rest pose")
        return {'FINISHED'}


def _rest_local(rig, name):
    bone = rig.data.bones[name]
    parent = bone.parent.matrix_local if bone.parent else Matrix()
    return parent.inverted() @ bone.matrix_local


def _numeric_props(pb):
    return {k: pb[k] for k in pb.keys() if isinstance(pb[k], (int, float))}


def record_animation(context, rig, every_frame=False):
    """For every action of the rig: on each keyed frame (or every frame), the pose channels of the
    animated controls, their parent's evaluated matrix and their rest matrix in the parent."""
    from .export import _action_fits
    scene = context.scene
    anim = rig.animation_data
    original = anim.action if anim else None
    names = utils.controls(rig)
    current = {n: (rig.pose.bones[n].matrix_basis.copy(), _numeric_props(rig.pose.bones[n])) for n in names}
    frame = scene.frame_current
    result = []
    for action in [a for a in bpy.data.actions if _action_fits(a, rig) and not a.get("rigger_pose")]:
        utils.assign_action(rig, action)
        if every_frame:
            start, end = (int(round(f)) for f in action.frame_range)
            frames = list(range(start, end + 1))
        else:
            frames = sorted({int(round(k.co.x)) for fc in utils.action_fcurves(action) for k in fc.keyframe_points})
        keyed = [n for n in names if any(fc.data_path.startswith(f'pose.bones["{n}"]')
                                         for fc in utils.action_fcurves(action))]
        # Bones without keys sit at the old rest pose in this action
        for n in names:
            rig.pose.bones[n].matrix_basis = Matrix()
        values = {}
        for f in frames:
            scene.frame_set(f)
            values[f] = {}
            for n in names:
                pb = rig.pose.bones[n]
                parent = pb.parent.matrix.copy() if pb.parent else Matrix()
                values[f][n] = (parent @ _rest_local(rig, n) @ pb.matrix_basis, _numeric_props(pb),
                                pb.matrix_basis.copy())
        result.append((action, values, keyed))
    if anim:
        utils.assign_action(rig, original)
    scene.frame_set(frame)
    for n, (matrix, props) in current.items():
        pb = rig.pose.bones[n]
        pb.matrix_basis = matrix
        for k, v in props.items():
            pb[k] = v
    context.view_layer.update()
    return {"actions": result, "props": {n: props for n, (_m, props) in current.items()}}


def rekey_animation(context, rig, recorded):
    """Re-key recorded actions for the new rest pose: every control gets the same pose (before its
    constraints) in armature space as before, whatever the new rest of its parents."""
    anim = rig.animation_data_create()
    original = anim.action
    for action, values, keyed in recorded["actions"]:
        for fc in list(utils.action_fcurves(action)):
            if fc.data_path.startswith('pose.bones["') and fc.data_path.split('"')[1] in keyed:
                utils.remove_fcurve(action, fc)
        utils.assign_action(rig, action)
        first = next(iter(values.values()), {})
        names = sorted((n for n in first if n in rig.pose.bones),
                       key=lambda n: len(rig.data.bones[n].parent_recursive))
        new = {}
        for frame, pose in values.items():
            context.scene.frame_set(frame)
            for name in names:
                if name in keyed:
                    pb = rig.pose.bones[name]
                    for k, v in pose[name][1].items():
                        if k in pb.keys():
                            pb[k] = v
            context.view_layer.update()
            new[frame] = {}
            for name in names:
                pb = rig.pose.bones[name]
                parent = pb.parent.matrix if pb.parent else Matrix()
                pb.matrix_basis = (parent @ _rest_local(rig, name)).inverted() @ pose[name][0]
                new[frame][name] = pb.matrix_basis.copy()
                rig.update_tag()
                context.view_layer.update()
        # Key the animated controls, and the others whose pose changed with the new rest
        changed = {n for n in names for f in values
                   if any(abs(a - b) > 1e-5 for ra, rb in zip(new[f][n], values[f][n][2]) for a, b in zip(ra, rb))}
        for frame, pose in values.items():
            for name in names:
                if name not in keyed and name not in changed:
                    continue
                pb = rig.pose.bones[name]
                pb.matrix_basis = new[frame][name]
                utils.key_bone(pb, frame)
                if name in keyed:
                    for k in pose[name][1]:
                        if k in pb.keys():
                            pb[k] = pose[name][1][k]
                            pb.keyframe_insert(f'["{k}"]', frame=frame, group=name)
    utils.assign_action(rig, original)
    for pb in rig.pose.bones:
        pb.matrix_basis = Matrix()
    # Switches that no action animates go back to their value before the rest change
    for name, props in recorded["props"].items():
        pb = rig.pose.bones.get(name)
        for k, v in props.items():
            if pb is not None and k in pb.keys():
                pb[k] = v
    context.view_layer.update()


# ---------------------------------------------------------------------------
# Fingers
# ---------------------------------------------------------------------------

def finger_masters(rig, side):
    return [n for n in utils.get_meta(rig).get("controls", [])
            if n.endswith("_master." + side) and any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))]


class RIGGER_OT_finger_pose(bpy.types.Operator):
    """Curl the fingers of one hand into a fist, or open them"""
    bl_idname = "rigger.finger_pose"
    bl_label = "Finger Pose"
    bl_options = {'REGISTER', 'UNDO'}

    side: EnumProperty(items=[('L', "Left", ""), ('R', "Right", ""), ('BOTH', "Both", "")], default='BOTH')
    pose: EnumProperty(items=[('FIST', "Fist", "Close the hand"), ('OPEN', "Open", "Reset the fingers")],
                       default='FIST')
    amount: FloatProperty(name="Amount", default=1.0, min=0.0, max=1.0)

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def execute(self, context):
        rig = context.object
        sides = ("L", "R") if self.side == 'BOTH' else (self.side,)
        count = 0
        for side in sides:
            for name in finger_masters(rig, side):
                pb = rig.pose.bones[name]
                if self.pose == 'OPEN':
                    pb.rotation_quaternion = (1, 0, 0, 0)
                else:
                    # Masters point along the finger with Z up: curling is a negative X rotation.
                    # The master's rotation is added at every joint, so this is a per-joint angle.
                    angle = -math.radians(20 if "thumb" in name else 32) * self.amount
                    pb.rotation_quaternion = Matrix.Rotation(angle, 4, 'X').to_quaternion()
                if context.scene.tool_settings.use_keyframe_insert_auto:
                    pb.keyframe_insert("rotation_quaternion", group=name)
                count += 1
        if not count:
            self.report({'WARNING'}, "No finger curl controls found")
            return {'CANCELLED'}
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Pinning
# ---------------------------------------------------------------------------

class RIGGER_OT_snap_pin(bpy.types.Operator):
    """Move the pole control onto the elbow/knee and turn pinning on (or off)"""
    bl_idname = "rigger.snap_pin"
    bl_label = "Snap Pinning"
    bl_options = {'REGISTER', 'UNDO'}

    limb: bpy.props.IntProperty(default=0)

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def execute(self, context):
        rig = context.object
        limbs = utils.get_meta(rig).get("limbs", [])
        if not 0 <= self.limb < len(limbs):
            return {'CANCELLED'}
        limb = limbs[self.limb]
        pbs = rig.pose.bones
        prop = pbs[limb["prop_bone"]]
        if prop.get("pin", 0.0) > 0.5:
            prop["pin"] = 0.0
        else:
            joint = pbs[limb["deform"][-2]].head.copy()
            pole = pbs[limb["pole"]]
            mat = pole.matrix.copy()
            mat.translation = joint
            pole.matrix = mat
            prop["pin"] = 1.0
        rig.update_tag()
        context.view_layer.update()
        return {'FINISHED'}


classes = (RIGGER_OT_edit_shape, RIGGER_OT_apply_shape, RIGGER_OT_corrective_shape,
           RIGGER_OT_shape_key_driver, RIGGER_OT_shape_key_driver_key, RIGGER_OT_remove_shape_driver,
           RIGGER_OT_mirror_shape_keys,
           RIGGER_OT_apply_rest_pose, RIGGER_OT_finger_pose, RIGGER_OT_snap_pin)
