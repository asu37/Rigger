# SPDX-License-Identifier: GPL-3.0-or-later
"""Animation tools for generated rigs.

* Pose blends: fist / extended hand poses blended by scaling a control, blink
  poses (with in-betweens) blended by the blink sliders, wings fold. They are
  stored on the reference and rebuilt with Action constraints at generation.
* Set Pose (A-pose, UE5 A-pose, T-pose), Mirror Pose
* Snap Child Of (switch between Child Of constraints, with bake)
* Extract Root Motion
* Quaternion / Euler conversion with keyframes
* IK/FK bakes, head lock / FK lock snaps and bakes, reversed spine snap,
  finger and spline IK/FK snaps
"""

import json
import math

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from mathutils import Matrix, Quaternion, Vector

from . import utils

POSES_KEY = "rigger_poses"
SIDES = [('L', "Left", ""), ('R', "Right", "")]


def _update(context, rig=None):
    if rig is not None:
        rig.update_tag()
    context.view_layer.update()


def _side_of(context, default='L'):
    pb = context.active_pose_bone
    letter = utils.side_letter(pb.name) if pb else ""
    return letter if letter in ("L", "R") else default


def _frame_range(context, start, end):
    scene = context.scene
    if start < 0 or end < 0:
        if scene.use_preview_range:
            return scene.frame_preview_start, scene.frame_preview_end
        return scene.frame_start, scene.frame_end
    return start, end


# ---------------------------------------------------------------------------
# Pose blends
# ---------------------------------------------------------------------------

def load_blends(ref):
    try:
        return json.loads(ref.get(POSES_KEY, "[]")) if ref is not None else []
    except (TypeError, ValueError):
        return []


def save_blends(ref, blends):
    ref[POSES_KEY] = json.dumps(blends)


def _basis(pb):
    loc, rot, scale = pb.matrix_basis.decompose()
    return [list(loc), list(rot), list(scale)]


def _set_basis(pb, values):
    loc, quat, scale = Vector(values[0]), Quaternion(values[1]), Vector(values[2])
    pb.location = loc
    if pb.rotation_mode == 'QUATERNION':
        pb.rotation_quaternion = quat
    elif pb.rotation_mode == 'AXIS_ANGLE':
        axis, angle = quat.to_axis_angle()
        pb.rotation_axis_angle = (angle, *axis)
    else:
        pb.rotation_euler = quat.to_euler(pb.rotation_mode)
    pb.scale = scale


def build_action(rig, name, keys):
    """Action with the rest pose at frame 0 and keys = {frame: {bone: basis}}."""
    old = bpy.data.actions.get(name)
    if old is not None:
        bpy.data.actions.remove(old)
    action = bpy.data.actions.new(name)
    anim = rig.animation_data_create()
    previous = anim.action
    previous_slot = getattr(anim, "action_slot", None)
    bones = sorted({b for pose in keys.values() for b in pose if b in rig.pose.bones})
    saved = {b: rig.pose.bones[b].matrix_basis.copy() for b in bones}
    anim.action = action
    for b in bones:
        rig.pose.bones[b].matrix_basis = Matrix()
        utils.key_bone(rig.pose.bones[b], 0)
    for frame, pose in sorted(keys.items(), key=lambda kv: float(kv[0])):
        for b, values in pose.items():
            if b in rig.pose.bones:
                _set_basis(rig.pose.bones[b], values)
                utils.key_bone(rig.pose.bones[b], float(frame))
    anim.action = previous
    if previous is not None and previous_slot is not None and hasattr(anim, "action_slot"):
        try:
            anim.action_slot = previous_slot
        except (TypeError, RuntimeError):
            pass
    for b, matrix in saved.items():
        rig.pose.bones[b].matrix_basis = matrix
    action.use_fake_user = True
    action["rigger_pose"] = True
    return action


def _action_constraint(rig, pb, action, driver, frame_end=10):
    con = pb.constraints.new('ACTION')
    con.name = "Pose Blend"
    con.action = action
    if hasattr(con, "action_slot") and action.slots:
        con.action_slot = action.slots[0]
    con.frame_start, con.frame_end = 0, frame_end
    con.mix_mode = 'BEFORE_SPLIT'
    if driver["type"] == 'SCALE':
        con.target = rig
        con.subtarget = driver["bone"]
        con.transform_channel = 'SCALE_Y'
        con.target_space = 'LOCAL'
        con.min, con.max = driver["min"], driver["max"]
    else:
        con.use_eval_time = True
        utils.add_driver(con, "eval_time", rig, driver["bone"], driver["prop"])
    return con


def edit_hook(gen):
    """Create the controls of stored pose blends (generator, Edit Mode)."""
    for blend in gen.pose_blends:
        ctrl = blend.get("ctrl")
        if not ctrl or ctrl["name"] in gen.edit_bones:
            continue
        parent = ctrl["parent"] if ctrl.get("parent") in gen.edit_bones else "root"
        gen.new_bone(ctrl["name"], Vector(ctrl["head"]), Vector(ctrl["tail"]), Vector(ctrl["z"]), parent=parent,
                     collection=ctrl.get("collection", "Fingers"))


def pose_hook(gen):
    """Actions and Action constraints of stored pose blends and wings fold (generator, Pose Mode)."""
    from . import widgets
    rig = gen.rig
    pbs = rig.pose.bones
    done = set()
    for blend in gen.pose_blends:
        keys = {k: {b: v for b, v in pose.items() if b in pbs} for k, pose in blend["keys"].items()}
        if not any(keys.values()):
            continue
        action = build_action(rig, f"rig_pose_{rig.name}_{blend['name']}", keys)
        bones = sorted({b for pose in keys.values() for b in pose})
        for name in bones:
            _action_constraint(rig, pbs[name], action, blend["driver"])
        ctrl = blend.get("ctrl")
        if ctrl and ctrl["name"] in pbs and ctrl["name"] not in done:
            done.add(ctrl["name"])
            pb = pbs[ctrl["name"]]
            length = max(rig.data.bones[pb.name].length, 1e-6)
            widgets.apply_widget(pb, 'CUBE', gen.scale * 0.01 * gen.widget_scale / length)
            widgets.set_color(pb, utils.side_letter(pb.name) or "C")
            pb.lock_location = (True, True, True)
            pb.lock_rotation = (True, True, True)
            pb.lock_rotation_w = True
            utils.add_constraint(pb, 'LIMIT_SCALE', use_min_y=True, min_y=0.5, use_max_y=True, max_y=1.5,
                                 owner_space='LOCAL')
    # Wings fold: needs an action named like 'rig_wings_fold' (rest at frame 0, folded at 10)
    fold_action = next((a for a in bpy.data.actions if "rig_wings_fold" in a.name), None)
    for module in gen.modules:
        fold = getattr(module, "wings_fold", None)
        if not fold or fold_action is None:
            continue
        suffix = module.suffix
        for name in sorted(utils.action_bones(fold_action)):
            if name in pbs and utils.split_side(name)[1] == suffix and name != fold:
                _action_constraint(rig, pbs[name], fold_action, {"type": 'SCALE', "bone": fold, "min": 1.0, "max": 0.0})


def has_blend(gen, prop):
    return any(b["driver"].get("prop") == prop for b in gen.pose_blends)


def _finger_controls(rig, side):
    names = []
    for finger in utils.get_meta(rig).get("fingers", []):
        for name in finger["fk"] + [finger.get("master")]:
            if name and utils.side_letter(name) == side:
                names.append(name)
    return names


def _lid_controls(rig, side):
    face = (utils.get_meta(rig).get("faces") or [{}])[0]
    return [n for n in face.get("controls", []) if utils.split_side(n)[0].startswith("lid_")
            and utils.side_letter(n) == side]


def _regenerate(context, rig):
    from .generate import generate
    ref = rig.rigger.reference
    mode = rig.mode
    utils.set_mode(rig, 'OBJECT')
    rig = generate(context, ref)
    ref.hide_set(True)
    utils.set_mode(rig, 'POSE' if mode == 'POSE' else 'OBJECT')
    return rig


def _fist_ctrl(rig, side):
    limb = next((l for l in utils.get_meta(rig).get("limbs", []) if l["type"] == "arm" and l["side"] == side), None)
    if limb is None:
        return None
    hand = rig.data.bones[limb["deform"][-1]]
    length = hand.length
    y = (hand.tail_local - hand.head_local).normalized()
    z = hand.matrix_local.to_3x3().col[2].normalized()
    head = hand.tail_local + z * length * 0.4
    stem = utils.split_side(limb["fk"][-1])[0].replace("_fk", "")
    return {"name": f"{stem}_fist.{side}", "head": list(head), "tail": list(head + y * length * 0.4),
            "z": list(z), "parent": hand.name, "collection": "Fingers"}


class _RigPoseOperator:
    @classmethod
    def poll(cls, context):
        obj = context.object
        return utils.is_rig(obj) and obj.mode == 'POSE'


class RIGGER_OT_add_pose_blend(_RigPoseOperator, bpy.types.Operator):
    """Store the current pose of the fingers (fist / extended hand) or eyelids (blink) as a pose
    the rig blends into: scale the hand's fist control down (fist) or up (extended), or use
    the blink slider. The rig is regenerated"""
    bl_idname = "rigger.add_pose_blend"
    bl_label = "Add Pose Blend"
    bl_options = {'REGISTER', 'UNDO'}

    kind: EnumProperty(name="Pose", items=[
        ('FIST', "Fist", "Fingers pose reached when the fist control is scaled to 0.5"),
        ('EXTEND', "Extended", "Fingers pose reached when the fist control is scaled to 1.5"),
        ('BLINK', "Blink", "Eyelid pose reached when the blink slider is 1 (or the in-between value)"),
    ])
    side: EnumProperty(name="Side", items=SIDES)
    amount: FloatProperty(name="Blink Value", default=1.0, min=0.05, max=1.0,
                          description="Blink slider value of this pose (below 1 = in-between pose)")

    def invoke(self, context, event):
        self.side = _side_of(context)
        return self.execute(context)

    def execute(self, context):
        rig = context.object
        ref = rig.rigger.reference
        if ref is None:
            self.report({'ERROR'}, "The rig has no reference armature")
            return {'CANCELLED'}
        blends = load_blends(ref)
        side = self.side
        if self.kind in ('FIST', 'EXTEND'):
            bones = _finger_controls(rig, side)
            ctrl = _fist_ctrl(rig, side)
            if not bones or ctrl is None:
                self.report({'ERROR'}, f"No fingers on the {side} arm")
                return {'CANCELLED'}
            name = f"{'fist' if self.kind == 'FIST' else 'extend'}.{side}"
            driver = {"type": 'SCALE', "bone": ctrl["name"], "min": 1.0, "max": 0.5 if self.kind == 'FIST' else 1.5}
            frame = "10"
        else:
            bones = _lid_controls(rig, side)
            if not bones:
                self.report({'ERROR'}, f"No eyelid controls on the {side} side")
                return {'CANCELLED'}
            face = utils.get_meta(rig)["faces"][0]["face"]
            name, ctrl = f"blink.{side}", None
            driver = {"type": 'PROP', "bone": face, "prop": f"blink_{side}"}
            frame = str(int(round(10 * self.amount)))
        pose = {b: _basis(rig.pose.bones[b]) for b in bones}
        blend = next((b for b in blends if b["name"] == name), None)
        if blend is None:
            blend = {"name": name, "driver": driver, "ctrl": ctrl, "keys": {}}
            blends.append(blend)
        blend["keys"][frame] = pose
        save_blends(ref, blends)
        _regenerate(context, rig)
        self.report({'INFO'}, f"Stored pose blend '{name}' ({len(bones)} bones)")
        return {'FINISHED'}


class RIGGER_OT_remove_pose_blend(bpy.types.Operator):
    """Remove a stored pose blend (or only its in-between poses). The rig is regenerated"""
    bl_idname = "rigger.remove_pose_blend"
    bl_label = "Remove Pose Blend"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(name="Name")
    inbetweens_only: BoolProperty(name="In-Betweens Only", default=False)

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.rigger.reference is not None

    def execute(self, context):
        rig = context.object
        ref = rig.rigger.reference
        blends = load_blends(ref)
        kept = []
        for blend in blends:
            if blend["name"] != self.name:
                kept.append(blend)
            elif self.inbetweens_only:
                blend["keys"] = {k: v for k, v in blend["keys"].items() if k == "10"}
                kept.append(blend)
        save_blends(ref, kept)
        _regenerate(context, rig)
        return {'FINISHED'}


def mirror_basis(values):
    loc, quat, scale = values
    return [[-loc[0], loc[1], loc[2]], [quat[0], quat[1], -quat[2], -quat[3]], list(scale)]


class RIGGER_OT_mirror_pose_blend(bpy.types.Operator):
    """Copy a pose blend to the other side (fist.L -> fist.R). The rig is regenerated"""
    bl_idname = "rigger.mirror_pose_blend"
    bl_label = "Mirror Pose Blend"
    bl_options = {'REGISTER', 'UNDO'}

    name: StringProperty(name="Name")

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.rigger.reference is not None

    def execute(self, context):
        rig = context.object
        ref = rig.rigger.reference
        blends = load_blends(ref)
        src = next((b for b in blends if b["name"] == self.name), None)
        if src is None:
            return {'CANCELLED'}
        name = utils.mirror_name(src["name"])
        driver = dict(src["driver"])
        driver["bone"] = utils.mirror_name(driver["bone"])
        if "prop" in driver:
            prop = driver["prop"]
            driver["prop"] = prop[:-1] + {"L": "R", "R": "L"}.get(prop[-1], prop[-1])
        ctrl = None
        if src.get("ctrl"):
            c = src["ctrl"]
            ctrl = dict(c, name=utils.mirror_name(c["name"]), parent=utils.mirror_name(c["parent"]),
                        head=[-c["head"][0], *c["head"][1:]], tail=[-c["tail"][0], *c["tail"][1:]],
                        z=[-c["z"][0], *c["z"][1:]])
        keys = {f: {utils.mirror_name(b): mirror_basis(v) for b, v in pose.items()} for f, pose in src["keys"].items()}
        blends = [b for b in blends if b["name"] != name]
        blends.append({"name": name, "driver": driver, "ctrl": ctrl, "keys": keys})
        save_blends(ref, blends)
        _regenerate(context, rig)
        return {'FINISHED'}


class RIGGER_OT_default_blink_pose(_RigPoseOperator, bpy.types.Operator):
    """Put the eyelid controls of one side in the closed pose of the default blink"""
    bl_idname = "rigger.default_blink_pose"
    bl_label = "Default Blink Pose"
    bl_options = {'REGISTER', 'UNDO'}

    side: EnumProperty(name="Side", items=SIDES)

    def invoke(self, context, event):
        self.side = _side_of(context)
        return self.execute(context)

    def execute(self, context):
        rig = context.object
        faces = utils.get_meta(rig).get("faces") or []
        if not faces:
            return {'CANCELLED'}
        face = rig.pose.bones[faces[0]["face"]]
        prop = f"blink_{self.side}"
        if prop not in face:
            return {'CANCELLED'}
        bones = sorted(_lid_controls(rig, self.side), key=lambda n: len(rig.data.bones[n].parent_recursive))
        old = face[prop]
        face[prop] = 1.0
        _update(context, rig)
        closed = {b: rig.pose.bones[b].matrix.copy() for b in bones}
        face[prop] = 0.0
        _update(context, rig)
        for b in bones:
            rig.pose.bones[b].matrix = closed[b]
            _update(context, rig)
        face[prop] = old if old < 1e-6 else 0.0
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Set pose, mirror pose
# ---------------------------------------------------------------------------

def set_arm_pose(context, rig, pose='A'):
    """Arms straight out (T), 45 degrees down (A) or down and slightly forward (UE5 A-pose)."""
    from . import rig_tools
    pbs = rig.pose.bones
    angle = {'T': 0.0, 'A': 45.0, 'A_UE5': 40.0}[pose]
    forward = {'T': 0.0, 'A': 0.0, 'A_UE5': 12.0}[pose]
    count = 0
    for limb in utils.get_meta(rig).get("limbs", []):
        if limb["type"] != "arm" or limb["side"] not in ("L", "R"):
            continue
        if pbs[limb["prop_bone"]]["ik_fk"] > 0.5:
            rig_tools.snap_fk_to_ik(context, rig, limb)
        side = 1.0 if limb["side"] == "L" else -1.0
        a, f = math.radians(angle), math.radians(forward)
        target = Vector((side * math.cos(a) * math.cos(f), -math.sin(f), -math.sin(a))).normalized()
        target = (rig.matrix_world.inverted().to_3x3() @ target).normalized()
        for name in limb["fk"][-3:]:
            pb = pbs[name]
            current = (pb.tail - pb.head).normalized()
            delta = current.rotation_difference(target).to_matrix().to_4x4()
            head = pb.head.copy()
            matrix = Matrix.Translation(head) @ delta @ Matrix.Translation(-head) @ pb.matrix
            pb.matrix = matrix
            _update(context, rig)
        count += 1
    return count


class RIGGER_OT_set_pose(_RigPoseOperator, bpy.types.Operator):
    """Pose the arms in a standard pose (FK). Use Apply Pose as Rest Pose afterwards to make it the rest pose"""
    bl_idname = "rigger.set_pose"
    bl_label = "Set Pose"
    bl_options = {'REGISTER', 'UNDO'}

    pose: EnumProperty(name="Pose", items=[
        ('A', "A-Pose", "Arms 45 degrees down"),
        ('A_UE5', "A-Pose (UE5)", "Arms down and slightly forward, like the UE5 mannequin"),
        ('T', "T-Pose", "Arms straight out"),
    ])

    def execute(self, context):
        count = set_arm_pose(context, context.object, self.pose)
        if not count:
            self.report({'WARNING'}, "No arms found")
            return {'CANCELLED'}
        return {'FINISHED'}


def flip_basis(pb):
    loc, rot, scale = pb.matrix_basis.decompose()
    return Matrix.LocRotScale(Vector((-loc.x, loc.y, loc.z)), Quaternion((rot.w, rot.x, -rot.y, -rot.z)), scale)


def mirror_pose(context, rig, names):
    """Copy the pose of the given controls to the other side (center controls are flipped)."""
    pbs = rig.pose.bones
    values = {}
    for name in names:
        other = utils.mirror_name(name)
        if other not in pbs:
            continue
        pb = pbs[name]
        props = {k: pb[k] for k in pb.keys() if isinstance(pb[k], (int, float))}
        values[other] = (flip_basis(pb), props)
    for other, (matrix, props) in values.items():
        pb = pbs[other]
        pb.matrix_basis = matrix
        for k, v in props.items():
            if k in pb.keys():
                pb[k] = v
    _update(context, rig)
    return len(values)


class RIGGER_OT_mirror_pose(_RigPoseOperator, bpy.types.Operator):
    """Copy the pose of the selected controls to the other side (all controls if none are selected)"""
    bl_idname = "rigger.mirror_pose"
    bl_label = "Mirror Pose"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.object
        names = [pb.name for pb in rig.pose.bones if utils.is_pose_bone_selected(pb)] or utils.controls(rig)
        count = mirror_pose(context, rig, names)
        self.report({'INFO'}, f"Mirrored {count} controls")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Child Of switcher
# ---------------------------------------------------------------------------

def child_of_constraints(pb):
    return [c for c in pb.constraints if c.type == 'CHILD_OF' and c.enabled]


def snap_child_of(context, rig, pb, index):
    cons = child_of_constraints(pb)
    if not 0 <= index < len(cons):
        return False
    matrix = pb.matrix.copy()
    for i, con in enumerate(cons):
        con.influence = 1.0 if i == index else 0.0
    _update(context, rig)
    utils.set_pose_matrix(context, pb, matrix)
    return True


def _key_child_of(rig, pb, frame):
    for con in child_of_constraints(pb):
        con.keyframe_insert("influence", frame=frame)
    utils.key_bone(pb, frame)


class RIGGER_OT_snap_child_of(_RigPoseOperator, bpy.types.Operator):
    """Switch the active bone to another Child Of constraint without moving it (optionally baked over a range)"""
    bl_idname = "rigger.snap_child_of"
    bl_label = "Snap Child Of"
    bl_options = {'REGISTER', 'UNDO'}

    index: IntProperty(name="Constraint", default=0, min=0)
    bake: BoolProperty(name="Bake", default=False)
    frame_start: IntProperty(name="Start", default=-1)
    frame_end: IntProperty(name="End", default=-1)

    def execute(self, context):
        rig, pb = context.object, context.active_pose_bone
        if pb is None or not child_of_constraints(pb):
            self.report({'ERROR'}, "The active bone has no Child Of constraint")
            return {'CANCELLED'}
        scene = context.scene
        if not self.bake:
            snap_child_of(context, rig, pb, self.index)
            if scene.tool_settings.use_keyframe_insert_auto:
                _key_child_of(rig, pb, scene.frame_current)
            return {'FINISHED'}
        start, end = _frame_range(context, self.frame_start, self.frame_end)
        current = scene.frame_current
        worlds = {}
        for frame in range(start, end + 1):
            scene.frame_set(frame)
            worlds[frame] = pb.matrix.copy()
        for frame in range(start, end + 1):
            scene.frame_set(frame)
            for i, con in enumerate(child_of_constraints(pb)):
                con.influence = 1.0 if i == self.index else 0.0
            _update(context, rig)
            utils.set_pose_matrix(context, pb, worlds[frame])
            _key_child_of(rig, pb, frame)
        scene.frame_set(current)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Root motion
# ---------------------------------------------------------------------------

def _root_children(rig):
    """Controls whose first control ancestor is the root."""
    names = set(utils.controls(rig))
    result = []
    for name in names:
        if name == "root":
            continue
        parent = rig.data.bones[name].parent
        while parent is not None and parent.name not in names:
            parent = parent.parent
        if parent is not None and parent.name == "root":
            result.append(name)
    order = {n: i for i, n in enumerate(utils.controls(rig))}
    torso = utils.get_meta(rig).get("spine", {}).get("torso")
    return sorted(result, key=lambda n: (n != torso, order.get(n, 0)))


def extract_root_motion(context, rig, target=None, use_x=True, use_y=True, use_z=False, z_offset=True,
                        rotation=False, forward_axis='Z', frame_start=None, frame_end=None):
    """Bake the root control so it follows the pelvis over the ground, keeping the world pose."""
    scene = context.scene
    pbs = rig.pose.bones
    target = target or utils.get_meta(rig).get("spine", {}).get("torso")
    if target not in pbs or "root" not in pbs:
        return 0
    deform = utils.get_meta(rig).get("deform_of", {})
    hips_def = next((d for r, d in deform.items() if r == "hips"), None) or target
    start = frame_start if frame_start is not None else scene.frame_start
    end = frame_end if frame_end is not None else scene.frame_end
    current = scene.frame_current
    children = _root_children(rig)
    root = pbs["root"]
    first_z = None
    frames = {}
    for frame in range(start, end + 1):
        scene.frame_set(frame)
        frames[frame] = ({n: pbs[n].matrix.copy() for n in children}, pbs[target].matrix.copy(),
                         pbs[hips_def].matrix.copy())
    for frame in range(start, end + 1):
        scene.frame_set(frame)
        worlds, pelvis, hips = frames[frame]
        loc = pelvis.translation
        if first_z is None:
            first_z = loc.z
        rest = root.bone.matrix_local
        new = rest.copy()
        new.translation = Vector((loc.x if use_x else rest.translation.x, loc.y if use_y else rest.translation.y,
                                  (loc.z - (first_z if z_offset else 0.0)) if use_z else rest.translation.z))
        if rotation:
            axis = {'X': 0, 'Y': 1, 'Z': 2}[forward_axis.lstrip('-')]
            forward = hips.to_3x3().col[axis] * (-1.0 if forward_axis.startswith('-') else 1.0)
            flat = Vector((forward.x, forward.y, 0.0))
            if flat.length > 1e-6:
                rest_forward = rest.to_3x3().col[1]
                rest_flat = Vector((rest_forward.x, rest_forward.y, 0.0)).normalized()
                yaw = rest_flat.to_2d().angle_signed(flat.normalized().to_2d())
                new = Matrix.Translation(new.translation) @ Matrix.Rotation(-yaw, 4, 'Z') @ rest.to_3x3().to_4x4()
        root.matrix = new
        _update(context, rig)
        for name in children:
            pbs[name].matrix = worlds[name]
            _update(context, rig)
        for name in ["root"] + children:
            utils.key_bone(pbs[name], frame)
            for prop in ("parent",):
                if prop in pbs[name]:
                    pbs[name].keyframe_insert(f'["{prop}"]', frame=frame)
    scene.frame_set(current)
    return end - start + 1


class RIGGER_OT_extract_root_motion(_RigPoseOperator, bpy.types.Operator):
    """Bake the root control along the pelvis on the ground, keeping the animation in place
    (keys every frame on the root and on the controls it carries)"""
    bl_idname = "rigger.extract_root_motion"
    bl_label = "Extract Root Motion"
    bl_options = {'REGISTER', 'UNDO'}

    target: StringProperty(name="Track", default="", description="Bone the root follows (empty = torso)")
    use_x: BoolProperty(name="X", default=True)
    use_y: BoolProperty(name="Y", default=True)
    use_z: BoolProperty(name="Z", default=False)
    z_offset: BoolProperty(name="Z Offset", default=True, description="Keep the initial height difference")
    rotation: BoolProperty(name="Rotation", default=False, description="The root turns with the pelvis")
    forward_axis: EnumProperty(name="Pelvis Forward", default='-Y', items=[
        (a, a, "") for a in ('X', '-X', 'Y', '-Y', 'Z', '-Z')])
    frame_start: IntProperty(name="Start", default=-1)
    frame_end: IntProperty(name="End", default=-1)

    def execute(self, context):
        start, end = _frame_range(context, self.frame_start, self.frame_end)
        count = extract_root_motion(context, context.object, self.target or None, self.use_x, self.use_y,
                                    self.use_z, self.z_offset, self.rotation, self.forward_axis, start, end)
        if not count:
            self.report({'ERROR'}, "No torso / root control found")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Baked root motion over {count} frames")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Rotation mode conversion
# ---------------------------------------------------------------------------

def convert_rotation(context, rig, names, mode, existing_only=True):
    scene = context.scene
    action = rig.animation_data.action if rig.animation_data else None
    frames = {}
    for fc in utils.action_fcurves(action) if action else []:
        if not fc.data_path.startswith('pose.bones["'):
            continue
        bone = fc.data_path.split('"')[1]
        if bone in names and fc.data_path.endswith(("rotation_euler", "rotation_quaternion", "rotation_axis_angle")):
            frames.setdefault(bone, set()).update(int(round(k.co.x)) for k in fc.keyframe_points)
    if not existing_only and action:
        start, end = (int(round(f)) for f in action.frame_range)
        for bone in names:
            frames[bone] = set(range(start, end + 1))
    current = scene.frame_current
    values = {}
    for bone, fs in frames.items():
        for frame in sorted(fs):
            scene.frame_set(frame)
            values.setdefault(bone, []).append((frame, rig.pose.bones[bone].matrix_basis.to_quaternion()))
    for bone in names:
        pb = rig.pose.bones[bone]
        quat = pb.matrix_basis.to_quaternion()
        if action:
            for fc in list(utils.action_fcurves(action)):
                if fc.data_path in (f'pose.bones["{bone}"].rotation_euler', f'pose.bones["{bone}"].rotation_quaternion',
                                    f'pose.bones["{bone}"].rotation_axis_angle'):
                    utils.remove_fcurve(action, fc)
        pb.rotation_mode = mode
        _set_basis(pb, [list(pb.location), list(quat), list(pb.scale)])
        for frame, q in values.get(bone, []):
            _set_basis(pb, [list(pb.location), list(q), list(pb.scale)])
            path = "rotation_quaternion" if mode == 'QUATERNION' else "rotation_euler"
            pb.keyframe_insert(path, frame=frame, group=bone)
    scene.frame_set(current)
    return len(names)


class RIGGER_OT_convert_rotation(_RigPoseOperator, bpy.types.Operator):
    """Change the rotation mode of the selected controls, converting their rotation keyframes"""
    bl_idname = "rigger.convert_rotation"
    bl_label = "Convert Rotation Mode"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Mode", items=[(m, m, "") for m in ('QUATERNION', 'XYZ', 'XZY', 'YXZ', 'YZX',
                                                                    'ZXY', 'ZYX')], default='XYZ')
    existing_only: BoolProperty(name="Existing Keyframes Only", default=True,
                                description="Off: key every frame of the action")

    def execute(self, context):
        rig = context.object
        names = [pb.name for pb in rig.pose.bones if utils.is_pose_bone_selected(pb)]
        if not names:
            self.report({'ERROR'}, "Select the controls to convert")
            return {'CANCELLED'}
        convert_rotation(context, rig, names, self.mode, self.existing_only)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Snaps and bakes
# ---------------------------------------------------------------------------

def snap_property(context, rig, bone, prop, value, keep, worlds=None):
    """Change a switch property while the 'keep' controls keep their transforms (the current ones,
    or 'worlds' = {bone: pose matrix} recorded earlier)."""
    pbs = rig.pose.bones
    keep = sorted([k for k in keep if k in pbs], key=lambda n: len(rig.data.bones[n].parent_recursive))
    if worlds is None:
        worlds = {k: pbs[k].matrix.copy() for k in keep}
    pbs[bone][prop] = value
    _update(context, rig)
    for k in keep:
        utils.set_pose_matrix(context, pbs[k], worlds[k])
    return keep


class RIGGER_OT_snap_property(_RigPoseOperator, bpy.types.Operator):
    """Switch a setting (head lock, FK lock...) without moving the controls"""
    bl_idname = "rigger.snap_property"
    bl_label = "Snap Switch"
    bl_options = {'REGISTER', 'UNDO'}

    bone: StringProperty(name="Bone")
    prop: StringProperty(name="Property")
    value: FloatProperty(name="Value", default=-1.0, description="-1 = toggle between 0 and 1")
    keep: StringProperty(name="Keep", description="Comma separated controls that keep their world transform "
                                                  "(empty = the bone itself)")
    bake: BoolProperty(name="Bake", default=False, description="Apply over a frame range, keying every frame")
    frame_start: IntProperty(name="Start", default=-1)
    frame_end: IntProperty(name="End", default=-1)

    def execute(self, context):
        rig = context.object
        pb = rig.pose.bones.get(self.bone)
        if pb is None or self.prop not in pb:
            return {'CANCELLED'}
        value = self.value if self.value >= 0 else (0.0 if pb[self.prop] > 0.5 else 1.0)
        keep = [k for k in self.keep.split(",") if k] or [self.bone]
        scene = context.scene
        frames = range(*(lambda s, e: (s, e + 1))(*_frame_range(context, self.frame_start, self.frame_end))) \
            if self.bake else [scene.frame_current]
        current = scene.frame_current
        recorded = {}
        if self.bake:
            # Record the pose of every frame before the switch is keyed
            for frame in frames:
                scene.frame_set(frame)
                recorded[frame] = {k: rig.pose.bones[k].matrix.copy() for k in keep if k in rig.pose.bones}
        for frame in frames:
            if self.bake:
                scene.frame_set(frame)
            kept = snap_property(context, rig, self.bone, self.prop, value, keep, recorded.get(frame))
            if self.bake or scene.tool_settings.use_keyframe_insert_auto:
                pb.keyframe_insert(f'["{self.prop}"]', frame=frame)
                for k in kept:
                    utils.key_bone(rig.pose.bones[k], frame)
        if self.bake:
            scene.frame_set(current)
        return {'FINISHED'}


class RIGGER_OT_bake_ikfk(_RigPoseOperator, bpy.types.Operator):
    """Snap and key a limb to IK or FK on every frame of a range (convert an FK animation to IK or back)"""
    bl_idname = "rigger.bake_ikfk"
    bl_label = "Bake IK/FK"
    bl_options = {'REGISTER', 'UNDO'}

    limb: IntProperty(name="Limb", default=0)
    to_ik: BoolProperty(name="To IK", default=True)
    frame_start: IntProperty(name="Start", default=-1)
    frame_end: IntProperty(name="End", default=-1)

    def execute(self, context):
        from . import rig_tools
        rig = context.object
        limbs = utils.get_meta(rig).get("limbs", [])
        if not 0 <= self.limb < len(limbs):
            return {'CANCELLED'}
        limb = limbs[self.limb]
        scene = context.scene
        start, end = _frame_range(context, self.frame_start, self.frame_end)
        current = scene.frame_current
        auto = scene.tool_settings.use_keyframe_insert_auto
        scene.tool_settings.use_keyframe_insert_auto = True
        try:
            for frame in range(start, end + 1):
                scene.frame_set(frame)
                pbs = rig.pose.bones
                prop = pbs[limb["prop_bone"]]
                # Evaluate the source mode, then snap the other chain onto it
                prop["ik_fk"] = 0.0 if self.to_ik else 1.0
                _update(context, rig)
                if self.to_ik:
                    rig_tools.snap_ik_to_fk(context, rig, limb)
                else:
                    rig_tools.snap_fk_to_ik(context, rig, limb)
        finally:
            scene.tool_settings.use_keyframe_insert_auto = auto
            scene.frame_set(current)
        return {'FINISHED'}


def snap_reverse_spine(context, rig, reverse):
    """Switch the spine between forward and reversed FK, keeping its shape."""
    spine = utils.get_meta(rig).get("spine") or {}
    if not spine.get("rev"):
        return False
    pbs = rig.pose.bones
    deform = spine["deform"][1:]
    worlds = [pbs[d].matrix.copy() for d in deform]
    torso = pbs[spine["torso"]]
    if reverse:
        for ctrl, mch, world in reversed(list(zip(spine["rev"], spine["rev_mch"], worlds))):
            rel = rig.data.bones[ctrl].matrix_local.inverted() @ rig.data.bones[mch].matrix_local
            pbs[ctrl].matrix = world @ rel.inverted()
            _update(context, rig)
        torso["spine_reverse"] = 1.0
    else:
        for ctrl, world in zip(spine["fk"], worlds):
            pbs[ctrl].matrix = world
            _update(context, rig)
        torso["spine_reverse"] = 0.0
    _update(context, rig)
    return True


class RIGGER_OT_snap_reverse_spine(_RigPoseOperator, bpy.types.Operator):
    """Switch between the forward and the reversed spine, keeping the current spine shape"""
    bl_idname = "rigger.snap_reverse_spine"
    bl_label = "Snap Reversed Spine"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        rig = context.object
        spine = utils.get_meta(rig).get("spine") or {}
        if not spine.get("rev"):
            return {'CANCELLED'}
        reverse = rig.pose.bones[spine["torso"]].get("spine_reverse", 0.0) < 0.5
        snap_reverse_spine(context, rig, reverse)
        if context.scene.tool_settings.use_keyframe_insert_auto:
            frame = context.scene.frame_current
            rig.pose.bones[spine["torso"]].keyframe_insert('["spine_reverse"]', frame=frame)
            for name in spine["rev"] if reverse else spine["fk"]:
                utils.key_bone(rig.pose.bones[name], frame)
        return {'FINISHED'}


def snap_finger(context, rig, finger, to_ik):
    pbs = rig.pose.bones
    if not finger.get("ik"):
        return False
    ik_ctrl = pbs[finger["ik_ctrl"]]
    if to_ik:
        last = pbs[finger["fk"][-1]]
        mat = ik_ctrl.matrix.copy()
        mat.translation = last.tail.copy()
        ik_ctrl.matrix = mat
        _update(context, rig)
        fk = [pbs[n] for n in finger["fk"]]
        joint = fk[len(fk) // 2].head
        bend = utils.bend_direction(fk[0].head, joint, fk[-1].tail, fk[0].z_axis)
        pole = pbs[finger["pole"]]
        distance = (pole.bone.head_local - rig.data.bones[finger["fk"][len(fk) // 2]].head_local).length
        pmat = pole.matrix.copy()
        pmat.translation = joint + bend * distance
        pole.matrix = pmat
        ik_ctrl["ik_fk"] = 1.0
    else:
        worlds = [pbs[n].matrix.copy() for n in finger["ik"]]
        for name, world in zip(finger["fk"], worlds):
            pbs[name].matrix = world
            _update(context, rig)
        ik_ctrl["ik_fk"] = 0.0
    _update(context, rig)
    return True


class RIGGER_OT_snap_fingers(_RigPoseOperator, bpy.types.Operator):
    """Switch fingers between IK and FK without moving them"""
    bl_idname = "rigger.snap_fingers"
    bl_label = "Snap Fingers IK/FK"
    bl_options = {'REGISTER', 'UNDO'}

    finger: StringProperty(name="Finger", description="IK control of one finger (empty = all fingers of the side)")
    side: EnumProperty(name="Side", items=SIDES)
    to_ik: BoolProperty(name="To IK", default=True)

    def execute(self, context):
        rig = context.object
        fingers = [f for f in utils.get_meta(rig).get("fingers", []) if f.get("ik")]
        if self.finger:
            fingers = [f for f in fingers if f["ik_ctrl"] == self.finger]
        else:
            fingers = [f for f in fingers if utils.side_letter(f["ik_ctrl"]) == self.side]
        for finger in fingers:
            snap_finger(context, rig, finger, self.to_ik)
        return {'FINISHED'} if fingers else {'CANCELLED'}


class RIGGER_OT_toggle_finger_parents(_RigPoseOperator, bpy.types.Operator):
    """Free the finger IK controls from the hand, or lock them to it again, without moving them"""
    bl_idname = "rigger.toggle_finger_parents"
    bl_label = "Toggle Finger IK Parents"
    bl_options = {'REGISTER', 'UNDO'}

    side: EnumProperty(name="Side", items=SIDES)

    def execute(self, context):
        rig = context.object
        for finger in utils.get_meta(rig).get("fingers", []):
            ctrl = finger.get("ik_ctrl")
            if ctrl and utils.side_letter(ctrl) == self.side and "parent" in rig.pose.bones[ctrl]:
                pb = rig.pose.bones[ctrl]
                bpy.ops.rigger.switch_space(bone=ctrl, space=0 if pb["parent"] else 1)
        return {'FINISHED'}


def snap_spline(context, rig, spline, to_ik):
    """IK -> FK is exact; FK -> IK places the curve controls on the FK chain."""
    pbs = rig.pose.bones
    if not spline.get("fk"):
        return False
    master = pbs[spline["master"]]
    if not to_ik:
        worlds = [pbs[n].matrix.copy() for n in spline["mch"]]
        for name, world in zip(spline["fk"], worlds):
            pbs[name].matrix = world
            _update(context, rig)
        master["ik_fk"] = 0.0
    else:
        fk = [pbs[n] for n in spline["fk"]]
        points = [fk[0].head] + [b.tail for b in fk]
        lengths = [0.0]
        for a, b in zip(points, points[1:]):
            lengths.append(lengths[-1] + (b - a).length)
        ctrls = spline["controls"]
        for i, name in enumerate(ctrls):
            dist = lengths[-1] * i / max(len(ctrls) - 1, 1)
            j = next((k for k in range(len(points) - 1) if lengths[k + 1] >= dist - 1e-9), len(points) - 2)
            seg = lengths[j + 1] - lengths[j]
            pos = points[j].lerp(points[j + 1], (dist - lengths[j]) / seg if seg > 1e-9 else 0.0)
            mat = pbs[name].matrix.copy()
            mat.translation = pos
            pbs[name].matrix = mat
            _update(context, rig)
        master["ik_fk"] = 1.0
    _update(context, rig)
    return True


class RIGGER_OT_snap_spline(_RigPoseOperator, bpy.types.Operator):
    """Switch a spline IK chain between IK and FK, keeping its shape"""
    bl_idname = "rigger.snap_spline"
    bl_label = "Snap Spline IK/FK"
    bl_options = {'REGISTER', 'UNDO'}

    master: StringProperty(name="Master")

    def execute(self, context):
        rig = context.object
        spline = next((s for s in utils.get_meta(rig).get("splines", []) if s["master"] == self.master), None)
        if spline is None:
            return {'CANCELLED'}
        to_ik = rig.pose.bones[spline["master"]].get("ik_fk", 1.0) < 0.5
        snap_spline(context, rig, spline, to_ik)
        return {'FINISHED'}


classes = (RIGGER_OT_add_pose_blend, RIGGER_OT_remove_pose_blend, RIGGER_OT_mirror_pose_blend,
           RIGGER_OT_default_blink_pose, RIGGER_OT_set_pose, RIGGER_OT_mirror_pose, RIGGER_OT_snap_child_of,
           RIGGER_OT_extract_root_motion, RIGGER_OT_convert_rotation, RIGGER_OT_snap_property,
           RIGGER_OT_bake_ikfk, RIGGER_OT_snap_reverse_spine, RIGGER_OT_snap_fingers,
           RIGGER_OT_toggle_finger_parents, RIGGER_OT_snap_spline)
