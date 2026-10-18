# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared helpers: naming, bone collections, colors, constraints and drivers."""

import json

import bpy
from mathutils import Vector

SIDE_SUFFIXES = (".L", ".R", "_L", "_R", ".l", ".r", "_l", "_r")
MIRROR_SUFFIX = {
    ".L": ".R", ".R": ".L", "_L": "_R", "_R": "_L",
    ".l": ".r", ".r": ".l", "_l": "_r", "_r": "_l",
}

META_KEY = "rigger_meta"
REF_KEY = "rigger_reference"
RIG_KEY = "rigger_rig"

# Bone collections created on generated rigs: (name, visible by default)
RIG_COLLECTIONS = (
    ("Root", True),
    ("Torso", True),
    ("Head", True),
    ("Face", True),
    ("Arm IK", True),
    ("Arm FK", True),
    ("Leg IK", True),
    ("Leg FK", True),
    ("Fingers", True),
    ("Tweak", True),
    ("Extra", True),
    ("Deform", False),
    ("Mechanism", False),
)



def split_side(name):
    """Return (stem, suffix) where suffix is a side suffix like '.L' or ''."""
    for suffix in SIDE_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)], suffix
    return name, ""


def side_letter(name):
    suffix = split_side(name)[1]
    return suffix[-1].upper() if suffix else ""


def mirror_name(name):
    stem, suffix = split_side(name)
    if not suffix:
        return name
    return stem + MIRROR_SUFFIX[suffix]


def unique_name(existing, name):
    """Return 'name' or 'stem_2.L', 'stem_3.L', ... not present in 'existing'."""
    if name not in existing:
        return name
    stem, suffix = split_side(name)
    i = 2
    while f"{stem}_{i}{suffix}" in existing:
        i += 1
    return f"{stem}_{i}{suffix}"


# ---------------------------------------------------------------------------
# Object tagging
# ---------------------------------------------------------------------------

def is_reference(obj):
    return obj is not None and obj.type == 'ARMATURE' and obj.get(REF_KEY, False)


def is_rig(obj):
    return obj is not None and obj.type == 'ARMATURE' and META_KEY in obj


def get_meta(rig):
    try:
        return json.loads(rig.get(META_KEY, "{}"))
    except (TypeError, ValueError):
        return {}


def set_meta(rig, meta):
    rig[META_KEY] = json.dumps(meta)


def find_rig_for(obj):
    """Return the generated rig related to obj (rig itself, its reference or a bound mesh)."""
    if obj is None:
        return None
    if is_rig(obj):
        return obj
    if is_reference(obj):
        return obj.rigger.rig
    if obj.type == 'MESH':
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and is_rig(mod.object):
                return mod.object
        if is_rig(obj.parent):
            return obj.parent
    return None


def deselect_all(context=None):
    """Deselect every object (skips stale entries of objects removed from bpy.data)."""
    ctx = context or bpy.context
    for o in list(ctx.view_layer.objects.selected):
        if o is not None:
            o.select_set(False)


def set_mode(obj, mode):
    """Make obj the only selected, active object and switch it to 'mode'."""
    ctx = bpy.context
    if ctx.object and ctx.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    deselect_all(ctx)
    obj.hide_set(False)
    obj.hide_viewport = False
    obj.select_set(True)
    ctx.view_layer.objects.active = obj
    if mode != 'OBJECT':
        bpy.ops.object.mode_set(mode=mode)


# ---------------------------------------------------------------------------
# Bone collections & colors
# ---------------------------------------------------------------------------

def ensure_collections(arm):
    for name, visible in RIG_COLLECTIONS:
        coll = arm.collections.get(name)
        if coll is None:
            coll = arm.collections.new(name)
        coll.is_visible = visible
    return arm.collections


def assign_collection(arm, bone, name):
    coll = arm.collections.get(name) or arm.collections.new(name)
    coll.assign(bone)


# ---------------------------------------------------------------------------
# Constraints, properties and drivers
# ---------------------------------------------------------------------------

def add_constraint(pose_bone, ctype, rig=None, subtarget=None, **props):
    con = pose_bone.constraints.new(ctype)
    if rig is not None and hasattr(con, "target"):
        con.target = rig
        if subtarget:
            con.subtarget = subtarget
    for key, value in props.items():
        setattr(con, key, value)
    return con


def add_property(pose_bone, name, value, minimum=0.0, maximum=1.0, description=""):
    pose_bone[name] = value
    ui = pose_bone.id_properties_ui(name)
    if isinstance(value, float):
        ui.update(min=minimum, max=maximum, soft_min=minimum, soft_max=maximum,
                  description=description)
    else:
        ui.update(description=description)
    try:
        pose_bone.property_overridable_library_set(f'["{name}"]', True)
    except (AttributeError, TypeError):
        pass


def add_driver(struct, path, rig, bone, prop, expression="var", index=-1):
    """Drive struct.path from rig.pose.bones[bone][prop]."""
    fcurve = struct.driver_add(path, index) if index >= 0 else struct.driver_add(path)
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    driver.expression = expression
    for var in list(driver.variables):
        driver.variables.remove(var)
    var = driver.variables.new()
    var.name = "var"
    var.type = 'SINGLE_PROP'
    target = var.targets[0]
    target.id_type = 'OBJECT'
    target.id = rig
    target.data_path = f'pose.bones["{bone}"]["{prop}"]'
    return fcurve


def add_driver_vars(struct, path, rig, variables, expression, index=-1):
    """Driver with several custom-property variables: [(var_name, bone, prop), ...]."""
    fcurve = struct.driver_add(path, index) if index >= 0 else struct.driver_add(path)
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    driver.expression = expression
    for var in list(driver.variables):
        driver.variables.remove(var)
    for name, bone, prop in variables:
        var = driver.variables.new()
        var.name = name
        var.type = 'SINGLE_PROP'
        target = var.targets[0]
        target.id_type = 'OBJECT'
        target.id = rig
        target.data_path = f'pose.bones["{bone}"]["{prop}"]'
    return fcurve


def add_transform_driver(struct, path, rig, bone, transform_type, expression="var", index=-1):
    """Driver reading a bone's local transform channel (e.g. 'LOC_X', 'SCALE_Z')."""
    fcurve = struct.driver_add(path, index) if index >= 0 else struct.driver_add(path)
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    driver.expression = expression
    for var in list(driver.variables):
        driver.variables.remove(var)
    var = driver.variables.new()
    var.name = "var"
    var.type = 'TRANSFORMS'
    target = var.targets[0]
    target.id = rig
    target.bone_target = bone
    target.transform_type = transform_type
    target.transform_space = 'LOCAL_SPACE'
    return fcurve


def add_driver_mixed(struct, path, rig, variables, expression, index=-1):
    """Driver with any mix of variables. Each variable is a tuple:

    * ``(name, 'prop', bone, prop)``               custom property of a pose bone
    * ``(name, 'obj', data_path)``                 property of the rig object (e.g. 'scale[2]')
    * ``(name, 'transform', bone, channel, space)`` bone channel ('LOC_Y', 'SCALE_Y', ...);
      space is 'LOCAL_SPACE', 'TRANSFORM_SPACE' (raw channel, before constraints) or 'WORLD_SPACE'
    * ``(name, 'distance', bone_a, bone_b)``       world distance between two bone heads
    * ``(name, 'rotdiff', bone_a, bone_b)``        angle between two bones
    """
    fcurve = struct.driver_add(path, index) if index >= 0 else struct.driver_add(path)
    driver = fcurve.driver
    driver.type = 'SCRIPTED'
    for var in list(driver.variables):
        driver.variables.remove(var)
    for spec in variables:
        name, kind = spec[0], spec[1]
        var = driver.variables.new()
        var.name = name
        if kind == 'prop':
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'OBJECT'
            var.targets[0].id = rig
            var.targets[0].data_path = f'pose.bones["{spec[2]}"]["{spec[3]}"]'
        elif kind == 'obj':
            var.type = 'SINGLE_PROP'
            var.targets[0].id_type = 'OBJECT'
            var.targets[0].id = rig
            var.targets[0].data_path = spec[2]
        elif kind == 'transform':
            var.type = 'TRANSFORMS'
            target = var.targets[0]
            target.id = rig
            target.bone_target = spec[2]
            target.transform_type = spec[3]
            target.transform_space = spec[4] if len(spec) > 4 else 'LOCAL_SPACE'
        elif kind in ('distance', 'rotdiff'):
            var.type = 'LOC_DIFF' if kind == 'distance' else 'ROTATION_DIFF'
            for target, bone in zip(var.targets, spec[2:4]):
                target.id = rig
                target.bone_target = bone
        else:
            raise ValueError(kind)
    driver.expression = expression
    return fcurve


def clear_drivers(id_data):
    anim = id_data.animation_data
    if anim is None:
        return
    for fcurve in list(anim.drivers):
        anim.drivers.remove(fcurve)


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def signed_angle(v1, v2, normal):
    """Angle from v1 to v2 around 'normal', in radians."""
    if v1.length < 1e-8 or v2.length < 1e-8:
        return 0.0
    angle = v1.angle(v2)
    if v1.cross(v2).dot(normal) < 0:
        angle = -angle
    return angle


def pole_angle(base_head, base_tail, base_x_axis, chain_end, pole_location):
    """IK pole angle that keeps a 2-bone chain in its rest orientation."""
    axis = base_tail - base_head
    pole_normal = (chain_end - base_head).cross(pole_location - base_head)
    projected = pole_normal.cross(axis)
    return -signed_angle(base_x_axis, projected, axis)


def bend_direction(a, b, c, fallback):
    """Direction the middle joint b points away from the line a-c."""
    ac = c - a
    if ac.length < 1e-8:
        return fallback.normalized()
    t = (b - a).dot(ac) / ac.length_squared
    offset = b - (a + ac * t)
    if offset.length < 1e-4 * ac.length:
        return fallback.normalized()
    return offset.normalized()


def rotation_angle_rest(rig, bone_a, bone_b):
    """Angle between two bones in the rest pose, as a ROTATION_DIFF driver variable measures it."""
    qa = rig.data.bones[bone_a].matrix_local.to_quaternion()
    qb = rig.data.bones[bone_b].matrix_local.to_quaternion()
    return qa.rotation_difference(qb).angle


def flatten(vec):
    """Project a vector on the ground plane (Z up)."""
    flat = Vector((vec.x, vec.y, 0.0))
    return flat.normalized() if flat.length > 1e-8 else Vector((0.0, -1.0, 0.0))


def set_pose_matrix(context, pose_bone, matrix, iterations=6, tolerance=1e-6):
    """Give a pose bone this final (constrained) pose-space matrix.

    Assigning pose_bone.matrix ignores the bone's own constraints (Child Of...), so the result is
    measured and corrected until the evaluated matrix matches.
    """
    target = matrix.copy()
    pose_bone.matrix = target
    context.view_layer.update()
    for _ in range(iterations):
        actual = pose_bone.matrix
        if max(abs(a - b) for ra, rb in zip(actual, matrix) for a, b in zip(ra, rb)) < tolerance:
            break
        target = target @ actual.inverted() @ matrix
        pose_bone.matrix = target
        context.view_layer.update()


def select_pose_bone(pose_bone, state=True):
    """Select a pose bone (selection lives on PoseBone in Blender 5.0+, on Bone before)."""
    if hasattr(pose_bone, "select"):
        pose_bone.select = state
    else:
        pose_bone.bone.select = state


def is_pose_bone_selected(pose_bone):
    if hasattr(pose_bone, "select"):
        return pose_bone.select
    return pose_bone.bone.select




# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_fcurve_lists(action):
    """F-curve collections of an action (legacy actions and layered 4.4+ actions)."""
    if action is None:
        return []
    if not getattr(action, "is_action_layered", False) and hasattr(action, "fcurves"):
        return [action.fcurves]
    lists = []
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                lists.append(bag.fcurves)
    return lists


def action_fcurves(action):
    return [fc for fcurves in action_fcurve_lists(action) for fc in fcurves]


def remove_fcurve(action, fcurve):
    for fcurves in action_fcurve_lists(action):
        for fc in fcurves:
            if fc == fcurve:
                fcurves.remove(fc)
                return True
    return False


def action_bones(action):
    """Names of the pose bones an action animates."""
    names = set()
    for fc in action_fcurves(action):
        if fc.data_path.startswith('pose.bones["'):
            names.add(fc.data_path.split('"')[1])
    return names


def assign_action(obj, action):
    """Assign an action to an object, picking a suitable slot on layered actions."""
    anim = obj.animation_data_create()
    anim.action = action
    if action is not None and hasattr(anim, "action_slot") and anim.action_slot is None and action.slots:
        try:
            anim.action_slot = action.slots[0]
        except (TypeError, RuntimeError):
            pass


def key_bone(pb, frame, group=None):
    group = group or pb.name
    pb.keyframe_insert("location", frame=frame, group=group)
    if pb.rotation_mode == 'QUATERNION':
        pb.keyframe_insert("rotation_quaternion", frame=frame, group=group)
    elif pb.rotation_mode == 'AXIS_ANGLE':
        pb.keyframe_insert("rotation_axis_angle", frame=frame, group=group)
    else:
        pb.keyframe_insert("rotation_euler", frame=frame, group=group)
    pb.keyframe_insert("scale", frame=frame, group=group)


def controls(rig):
    """Animator controls of a generated rig, parents first."""
    names = [n for n in get_meta(rig).get("controls", []) if n in rig.pose.bones]
    return sorted(names, key=lambda n: len(rig.data.bones[n].parent_recursive))
