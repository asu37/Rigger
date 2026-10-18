# SPDX-License-Identifier: GPL-3.0-or-later
"""Transfer animation from any armature (Mixamo, mocap/BVH, game skeletons...)
to a Rigger rig or any other armature.

Method: for every mapped pair the world-space rotation change of the source
bone (relative to its reference pose) is applied to the target bone's
reference pose. Root/hips and IK controls also receive the location change,
scaled by the height ratio of the two characters. Bones are solved parent
first so children see their parent's new pose, and the result is keyed.
"""

import json
import re

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Euler, Matrix, Quaternion, Vector

from . import utils

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")
_SEP = r"[._\- ]"
_ROLE_TABLE = (
    ("toe", ("toe", "ball")),
    ("foot", ("foot", "ankle")),
    ("hand", ("hand", "wrist")),
    ("forearm", ("forearm", "lowerarm", "elbow")),
    ("upper_arm", ("upperarm", "humerus")),
    ("shoulder", ("shoulder", "clavicle", "collar", "clav")),
    ("thigh", ("thigh", "upleg", "upperleg", "femur")),
    ("shin", ("shin", "calf", "lowerleg", "knee", "tibia")),
    ("neck", ("neck",)),
    ("head", ("head",)),
    ("spine", ("spine", "chest", "torso", "abdomen", "back")),
    ("hips", ("hips", "pelvis", "hip")),
)


def _side(raw):
    low = raw.lower()
    if "left" in low:
        return "L"
    if "right" in low:
        return "R"
    for letter in "lr":
        if re.search(rf"(^|{_SEP}){letter}({_SEP}|$)", low):
            return letter.upper()
    match = re.match(r"^([LR])(?=[A-Z])", raw)
    return match.group(1) if match else ""


def classify(name):
    """Guess (role, side, finger, segment) of a bone from its name.

    role is one of: hips, spine, neck, head, shoulder, upper_arm, forearm, hand,
    thigh, shin, foot, toe, finger, metacarpal, toe_digit, plus '<role>_ik' for IK controls.
    Returns None for unknown or helper bones.
    """
    raw = name.split(":")[-1].split("|")[-1]
    if raw.lower().startswith("def-"):
        raw = raw[4:]
    low = raw.lower()
    if low.startswith(("mch-", "org-", "wgt-")) or any(
            k in low for k in ("_master", "_pole", "_target", "_roll", "twist", "_socket")):
        return None
    if re.search(r"(end|nub|top|tip)(_?\d*)$", low):
        return None
    side = _side(raw)
    digits = re.findall(r"\d+", low)
    segment = int(digits[-1]) if digits else 0
    words = re.sub(r"^[LR](?=[A-Z])", "", raw).lower()
    words = re.sub(r"left|right", "", words)
    words = re.sub(rf"(^|{_SEP})[lr](?={_SEP}|$)", r"\1", words)
    tokens = re.findall(r"[a-z]+", words)
    stripped = "".join(tokens)
    ik = "ik" in tokens

    for finger in FINGER_NAMES:
        if finger in stripped or (finger == "pinky" and ("little" in stripped or "pinkie" in stripped)):
            if stripped.startswith("toe"):
                return ("toe_digit", side, finger, segment)  # individual toes, not fingers
            if "meta" in stripped:
                return ("metacarpal", side, finger, segment)
            return ("finger", side, finger, segment)
    role = None
    for candidate, keys in _ROLE_TABLE:
        if any(k in stripped for k in keys):
            role = candidate
            break
    if role is None and side:
        # Mixamo / mocap style: LeftArm = upper arm, LeftLeg = shin
        if re.fullmatch(r"arm(fk|ik)?", stripped):
            role = "upper_arm"
        elif re.fullmatch(r"leg(fk|ik)?", stripped):
            role = "shin"
    if role is None or (role == "hips" and side):  # e.g. CMU 'LHipJoint'
        return None
    if role in ("hips", "spine", "neck", "head"):
        side = ""
    if ik:
        role += "_ik"
    return (role, side, None, segment)


def _depth(armature, name):
    return len(armature.data.bones[name].parent_recursive)


def classify_armature(obj, names=None):
    """Return {(role, side, finger): [bone names ordered by depth]}."""
    groups = {}
    for name in (names if names is not None else obj.data.bones.keys()):
        info = classify(name)
        if info is None:
            continue
        role, side, finger, _seg = info
        groups.setdefault((role, side, finger), []).append(name)
    for key in groups:
        groups[key].sort(key=lambda n: (_depth(obj, n), n))
        if key[0] == "finger" and len(groups[key]) > 3:
            # Drop end/leaf bones such as Mixamo's 'LeftHandIndex4'
            groups[key] = [n for n in groups[key] if obj.data.bones[n].children] or groups[key]
    return groups


def _pair_lists(sources, targets):
    """Pair two ordered chains of different lengths, spreading the shorter one."""
    if not sources or not targets:
        return []
    if len(sources) == len(targets):
        return list(zip(sources, targets))
    if len(sources) < len(targets):
        if len(sources) == 1:
            return [(sources[0], targets[0])]
        step = (len(targets) - 1) / (len(sources) - 1)
        return [(s, targets[round(i * step)]) for i, s in enumerate(sources)]
    if len(targets) == 1:
        return [(sources[0], targets[0])]
    step = (len(sources) - 1) / (len(targets) - 1)
    return [(sources[round(i * step)], t) for i, t in enumerate(targets)]


def rigger_target_groups(rig, legs_ik, arms_ik):
    """Role groups for a Rigger rig, pointing at its animation controls."""
    meta = utils.get_meta(rig)
    groups = {}
    loc = set()
    spine = meta.get("spine")
    if spine:
        groups[("hips", "", None)] = [spine["torso"]]
        loc.add(spine["torso"])
        groups[("spine", "", None)] = list(spine["fk"])
    neck = meta.get("neck")
    if neck:
        if neck["fk"]:
            groups[("neck", "", None)] = list(neck["fk"])
        groups[("head", "", None)] = [neck["head"]]
    for limb in meta.get("limbs", []):
        side = limb["side"]
        use_ik = legs_ik if limb["type"] == "leg" else arms_ik
        if limb["type"] == "arm":
            roles = ("upper_arm", "forearm", "hand")
            if limb.get("shoulder"):
                groups[("shoulder", side, None)] = [limb["shoulder"]]
        else:
            roles = ("thigh", "shin", "foot")
            if groups.get(("thigh", side, None)):
                continue  # only the first pair of legs (quadrupeds keep their hind legs unmapped)
        for role, fk in zip(roles, limb["fk"]):
            groups[(role, side, None)] = [fk]
        if use_ik:
            groups[(roles[2], side, None)] = [limb["ik_ctrl"]]
            loc.add(limb["ik_ctrl"])
        if limb["type"] == "leg":
            toe = limb.get("toe_ik") if use_ik else limb.get("toe_fk")
            if toe:
                groups[("toe", side, None)] = [toe]
    controls = [n for n in meta.get("controls", []) if "_master" not in n]
    for key, names in classify_armature(rig, controls).items():
        if key[0] in ("finger", "metacarpal"):
            groups[key] = names
    return groups, loc


def auto_map(settings):
    source, target = settings.source, settings.target
    src_groups = classify_armature(source)
    if utils.is_rig(target):
        tgt_groups, loc = rigger_target_groups(target, settings.legs_ik, settings.arms_ik)
    else:
        tgt_groups = classify_armature(target)
        loc = set(tgt_groups.get(("hips", "", None), [])[:1])
    settings.items.clear()
    count = 0
    for key, targets in tgt_groups.items():
        sources = src_groups.get(key)
        if not sources:
            continue
        for src, tgt in _pair_lists(sources, targets):
            item = settings.items.add()
            item.source, item.target = src, tgt
            item.location = tgt in loc
            count += 1
    # Fallback for the hips: the top-most source bone
    if not any(i.location for i in settings.items) and ("hips", "", None) in tgt_groups:
        roots = [b.name for b in source.data.bones if b.parent is None]
        if roots:
            item = settings.items.add()
            item.source, item.target = roots[0], tgt_groups[("hips", "", None)][0]
            item.location = True
            count += 1
    return count


# ---------------------------------------------------------------------------
# Baking
# ---------------------------------------------------------------------------

def _world(obj, pb):
    return obj.matrix_world @ pb.matrix


def _rest_world(obj, name):
    return obj.matrix_world @ obj.data.bones[name].matrix_local


class Retargeter:
    """Per-frame transfer of the source animation to the target (keyed, or live for preview)."""

    def __init__(self, context, settings, action=None):
        self.context = context
        self.settings = settings
        source, target = settings.source, settings.target
        self.source, self.target = source, target
        self.items = [i for i in settings.items if self._src(i) in source.pose.bones and i.target in target.pose.bones]
        if not self.items:
            raise ValueError("Bone map is empty: use Auto Map or add pairs")
        if action is not None:
            utils.assign_action(source, action)
        self.action = source.animation_data.action if source.animation_data else None
        scene = context.scene
        if settings.frame_start >= 0 and settings.frame_end >= settings.frame_start:
            self.start, self.end = settings.frame_start, settings.frame_end
        elif self.action is not None:
            self.start, self.end = (int(round(f)) for f in self.action.frame_range)
        else:
            self.start, self.end = scene.frame_start, scene.frame_end
        frame_before = scene.frame_current
        items = self.items
        if settings.source_frame >= 0:
            scene.frame_set(settings.source_frame)
            self.src_ref = {self._src(i): _world(source, source.pose.bones[self._src(i)]) for i in items}
            self.src_local_ref = {self._src(i): source.pose.bones[self._src(i)].matrix_basis.copy() for i in items}
        else:
            self.src_ref = {self._src(i): _rest_world(source, self._src(i)) for i in items}
            self.src_local_ref = {self._src(i): Matrix() for i in items}
        if settings.use_target_pose:
            context.view_layer.update()
            self.tgt_ref = {i.target: _world(target, target.pose.bones[i.target]) for i in items}
        else:
            self.tgt_ref = {i.target: _rest_world(target, i.target) for i in items}
        scene.frame_set(frame_before)
        # Location scale from the hips height
        self.scale = 1.0
        loc_items = [i for i in items if i.location and i.location_mode == 'WORLD']
        self.hip = None
        if loc_items:
            hip = next((i for i in loc_items if (target.data.bones[i.target].parent is None
                                                 or target.data.bones[i.target].parent.name == "root")),
                       loc_items[0])
            self.hip = hip
            src_h = self.src_ref[self._src(hip)].translation.z - source.matrix_world.translation.z
            tgt_h = self.tgt_ref[hip.target].translation.z - target.matrix_world.translation.z
            if abs(src_h) > 1e-5:
                self.scale = tgt_h / src_h
        self.scale *= settings.scale
        # In place: remove the hips' straight travel between the first and last frame
        self.travel = None
        if settings.in_place and self.hip is not None:
            spb = source.pose.bones[self._src(self.hip)]
            scene.frame_set(self.start)
            a = _world(source, spb).translation.copy()
            scene.frame_set(self.end)
            b = _world(source, spb).translation.copy()
            scene.frame_set(frame_before)
            self.travel = Vector((b.x - a.x, b.y - a.y, 0.0))
        groups = {}
        for i in items:
            groups.setdefault(_depth(target, i.target), []).append(i)
        self.order = [groups[d] for d in sorted(groups)]
        self.tgt_inv = target.matrix_world.inverted()
        self.tgt_rot_inv = target.matrix_world.to_quaternion().inverted()
        self.prev_quats = {}
        self.poles = []
        self.limbs = utils.get_meta(target).get("limbs", []) if utils.is_rig(target) else []
        by_target = {i.target: self._src(i) for i in items}
        for limb in self.limbs:
            use_ik = settings.legs_ik if limb["type"] == "leg" else settings.arms_ik
            src_upper, src_lower = by_target.get(limb["fk"][-3]), by_target.get(limb["fk"][-2])
            if use_ik and src_upper and src_lower and limb["ik_ctrl"] in by_target:
                self.poles.append((limb, src_upper, src_lower))
        self.pole_track = {}

    def _src(self, item):
        s = self.settings
        if s.namespace_from:
            return item.source.replace(s.namespace_from, s.namespace_to)
        return item.source

    def reset_target(self):
        target = self.target
        if utils.is_rig(target):
            for ctrl in utils.get_meta(target).get("controls", []):
                pb = target.pose.bones.get(ctrl)
                if pb is not None:
                    pb.matrix_basis = Matrix()
        for i in self.items:
            target.pose.bones[i.target].matrix_basis = Matrix()
        for limb in self.limbs:
            use_ik = self.settings.legs_ik if limb["type"] == "leg" else self.settings.arms_ik
            target.pose.bones[limb["prop_bone"]]["ik_fk"] = 1.0 if use_ik else 0.0

    def frames(self):
        if self.settings.existing_keys and self.action is not None:
            keys = sorted({int(round(k.co.x)) for fc in utils.action_fcurves(self.action)
                           for k in fc.keyframe_points})
            return [f for f in keys if self.start <= f <= self.end] or [self.start]
        return list(range(self.start, self.end + 1))

    def _offset(self, frame):
        if self.travel is None or self.end == self.start:
            return Vector()
        return self.travel * ((frame - self.start) / (self.end - self.start))

    def apply(self, frame):
        """Pose the target for the current frame (the scene must be at 'frame')."""
        source, target, context = self.source, self.target, self.context
        offset = self._offset(frame)
        for level in self.order:
            for i in level:
                spb, tpb = source.pose.bones[self._src(i)], target.pose.bones[i.target]
                if i.location and i.location_mode == 'LOCAL':
                    rel = self.src_local_ref[self._src(i)].inverted() @ spb.matrix_basis
                    loc, rot, _scale = rel.decompose()
                    tpb.location = loc * i.loc_multiplier * self.settings.scale
                    if tpb.rotation_mode == 'QUATERNION':
                        tpb.rotation_quaternion = rot @ Euler(i.rot_offset).to_quaternion()
                    else:
                        tpb.rotation_euler = (rot @ Euler(i.rot_offset).to_quaternion()).to_euler(tpb.rotation_mode)
                    continue
                src_world = _world(source, spb)
                s_ref, t_ref = self.src_ref[self._src(i)], self.tgt_ref[i.target]
                delta = src_world.to_quaternion() @ s_ref.to_quaternion().inverted()
                rot = delta @ t_ref.to_quaternion() @ Euler(i.rot_offset).to_quaternion()
                if i.location:
                    loc = t_ref.translation + (src_world.translation - s_ref.translation - offset) * self.scale
                    loc += Vector(i.loc_offset)
                else:
                    loc = (target.matrix_world @ tpb.matrix).translation
                pose = Matrix.Translation(self.tgt_inv @ loc) @ (self.tgt_rot_inv @ rot).to_matrix().to_4x4()
                tpb.matrix = pose
                if not i.location:
                    tpb.location = (0.0, 0.0, 0.0)
            context.view_layer.update()
        for limb, src_upper, src_lower in self.poles:
            pole = target.pose.bones[limb["pole"]]
            ta = target.matrix_world @ target.pose.bones[limb["ik"][-3]].head
            tc = target.matrix_world @ target.pose.bones[limb["ik_target"]].head
            mode = self.settings.pole_mode
            if mode == 'TARGET':
                rest_rel = target.data.bones[limb["ik_target"]].matrix_local.inverted() @ pole.bone.matrix_local
                world_pos = (target.matrix_world @ target.pose.bones[limb["ik_target"]].matrix @ rest_rel).translation
            elif mode == 'CHAIN':
                s_ref = self.src_ref[src_upper]
                delta = _world(source, source.pose.bones[src_upper]).to_quaternion() @ \
                    s_ref.to_quaternion().inverted()
                rest_head = target.matrix_world @ target.data.bones[limb["ik"][-3]].head_local
                rest_pole = target.matrix_world @ pole.bone.head_local
                world_pos = ta + delta @ (rest_pole - rest_head)
            else:
                a = _world(source, source.pose.bones[src_upper]).translation
                b = _world(source, source.pose.bones[src_lower]).translation
                c = source.matrix_world @ source.pose.bones[src_lower].tail
                fallback = (target.matrix_world @ pole.head) - ta.lerp(tc, 0.5)
                bend = utils.bend_direction(a, b, c, fallback)
                length = sum(target.data.bones[n].length for n in limb["fk"][-3:-1])
                world_pos = ta.lerp(tc, 0.5) + bend * length * 0.6
            mat = pole.bone.matrix_local.copy()
            mat.translation = self.tgt_inv @ world_pos
            pole.matrix = mat
            self.pole_track.setdefault(limb["pole"], {})[frame] = pole.location.copy()
        if self.poles:
            context.view_layer.update()

    def key(self, frame):
        target = self.target
        for i in self.items:
            tpb = target.pose.bones[i.target]
            if tpb.rotation_mode == 'QUATERNION':
                q = tpb.rotation_quaternion.copy()
                prev = self.prev_quats.get(i.target)
                if prev is not None and prev.dot(q) < 0:
                    q.negate()
                    tpb.rotation_quaternion = q
                self.prev_quats[i.target] = q
                tpb.keyframe_insert("rotation_quaternion", frame=frame, group=i.target)
            elif tpb.rotation_mode == 'AXIS_ANGLE':
                tpb.keyframe_insert("rotation_axis_angle", frame=frame, group=i.target)
            else:
                tpb.keyframe_insert("rotation_euler", frame=frame, group=i.target)
            if i.location:
                tpb.keyframe_insert("location", frame=frame, group=i.target)
        for limb, _u, _l in self.poles:
            pole = target.pose.bones[limb["pole"]]
            pole.keyframe_insert("location", frame=frame, group=pole.name)

    def smooth_poles(self, action):
        """Moving average of the keyed pole locations."""
        for pole in self.pole_track:
            for fc in utils.action_fcurves(action):
                if fc.data_path != f'pose.bones["{pole}"].location':
                    continue
                values = {int(round(k.co.x)): k.co.y for k in fc.keyframe_points}
                for k in fc.keyframe_points:
                    f = int(round(k.co.x))
                    near = [values[g] for g in (f - 2, f - 1, f, f + 1, f + 2) if g in values]
                    k.co.y = sum(near) / len(near)
                fc.update()

    def clean_fk(self, action):
        """Forearm / shin FK controls only rotate around their X (hinge) axis."""
        hinges = {limb["fk"][-2] for limb in self.limbs}
        target = self.target
        for name in hinges:
            pb = target.pose.bones.get(name)
            if pb is None or pb.rotation_mode != 'QUATERNION':
                continue
            curves = {fc.array_index: fc for fc in utils.action_fcurves(action)
                      if fc.data_path == f'pose.bones["{name}"].rotation_quaternion'}
            if len(curves) != 4:
                continue
            frames = sorted({k.co.x for fc in curves.values() for k in fc.keyframe_points})
            for f in frames:
                q = Quaternion([curves[i].evaluate(f) for i in range(4)])
                hinge = Quaternion((q.w, q.x, 0.0, 0.0)).normalized()
                for i in range(4):
                    curves[i].keyframe_points.insert(f, hinge[i], options={'REPLACE'})

    def run(self, report=None):
        """Bake every frame into a new action on the target. Returns the action."""
        context, scene, settings = self.context, self.context.scene, self.settings
        target = self.target
        frame_before = scene.frame_current
        name = settings.action_name or ((self.action.name if self.action else "remap") + "_remap")
        if settings.multiple and self.action is not None:
            name = self.action.name + "_remap"
        new_action = bpy.data.actions.new(name)
        new_action.use_fake_user = True
        target.animation_data_create()
        utils.assign_action(target, new_action)
        self.reset_target()
        for limb in self.limbs:
            target.pose.bones[limb["prop_bone"]].keyframe_insert('["ik_fk"]', frame=self.start,
                                                                 group=limb["prop_bone"])
        for frame in self.frames():
            scene.frame_set(frame)
            self.apply(frame)
            self.key(frame)
        if settings.smooth_poles:
            self.smooth_poles(new_action)
        if settings.clean_fk:
            self.clean_fk(new_action)
        if self.action is not None:
            new_action.use_frame_range = self.action.use_frame_range
            if self.action.use_frame_range:
                new_action.frame_start, new_action.frame_end = self.action.frame_start, self.action.frame_end
            new_action.use_cyclic = self.action.use_cyclic
        utils.assign_action(target, new_action)
        if settings.root_motion and utils.is_rig(target):
            from .poses import extract_root_motion
            extract_root_motion(context, target, frame_start=self.start, frame_end=self.end)
        scene.frame_set(frame_before)
        if report:
            report({'INFO'}, f"Retargeted {len(self.items)} bones, frames {self.start}-{self.end} "
                             f"into '{new_action.name}'")
        return new_action


def remap_actions(settings):
    """Source actions to retarget with Multiple Source Animations."""
    source = settings.source
    names = set(source.pose.bones.keys())
    result = []
    for action in bpy.data.actions:
        if not action.rigger_remap or action.get("rigger_pose"):
            continue
        if settings.action_filter and settings.action_filter.lower() not in action.name.lower():
            continue
        if any(fc.data_path.startswith('pose.bones["') and fc.data_path.split('"')[1] in names
               for fc in utils.action_fcurves(action)):
            result.append(action)
    return result


def retarget(context, settings, report=None):
    """Retarget the source's current action (or every ticked action). Returns the last new action."""
    if settings.multiple:
        source = settings.source
        original = source.animation_data.action if source.animation_data else None
        result = None
        for action in remap_actions(settings):
            result = Retargeter(context, settings, action).run(report)
        if original is not None:
            utils.assign_action(source, original)
        if result is None:
            raise ValueError("No source action to retarget")
        return result
    return Retargeter(context, settings).run(report)


# -- Live preview (bind / unbind) -------------------------------------------------------------
_preview = {}


@bpy.app.handlers.persistent
def _preview_handler(scene, _depsgraph=None):
    retargeter = _preview.get("retargeter")
    if retargeter is None:
        return
    try:
        retargeter.apply(scene.frame_current)
    except ReferenceError:
        _preview.clear()


def preview_start(context, settings):
    preview_stop()
    retargeter = Retargeter(context, settings)
    retargeter.reset_target()
    _preview["retargeter"] = retargeter
    bpy.app.handlers.frame_change_post.append(_preview_handler)
    retargeter.apply(context.scene.frame_current)


def preview_stop():
    if _preview_handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_preview_handler)
    retargeter = _preview.pop("retargeter", None)
    if retargeter is not None:
        try:
            retargeter.reset_target()
        except ReferenceError:
            pass


def preview_active():
    return "retargeter" in _preview


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class _RemapPoll:
    @classmethod
    def poll(cls, context):
        s = context.scene.rigger_remap
        return s.source is not None and s.target is not None


class RIGGER_OT_remap_auto(_RemapPoll, bpy.types.Operator):
    """Build the bone map automatically from bone names"""
    bl_idname = "rigger.remap_auto"
    bl_label = "Auto Map"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = auto_map(context.scene.rigger_remap)
        self.report({'INFO'} if count else {'WARNING'}, f"Mapped {count} bones")
        return {'FINISHED'}


class RIGGER_OT_remap_add(_RemapPoll, bpy.types.Operator):
    """Add a pair to the bone map"""
    bl_idname = "rigger.remap_add"
    bl_label = "Add Pair"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_remap
        s.items.add()
        s.active_index = len(s.items) - 1
        return {'FINISHED'}


class RIGGER_OT_remap_remove(_RemapPoll, bpy.types.Operator):
    """Remove the active pair from the bone map"""
    bl_idname = "rigger.remap_remove"
    bl_label = "Remove Pair"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_remap
        if 0 <= s.active_index < len(s.items):
            s.items.remove(s.active_index)
            s.active_index = min(s.active_index, len(s.items) - 1)
        return {'FINISHED'}


class RIGGER_OT_remap_bake(_RemapPoll, bpy.types.Operator):
    """Transfer the source animation to the target and key it into a new action"""
    bl_idname = "rigger.remap_bake"
    bl_label = "Retarget"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_remap
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            retarget(context, s, self.report)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class RIGGER_OT_remap_save(ExportHelper, bpy.types.Operator):
    """Save the bone map as a JSON preset"""
    bl_idname = "rigger.remap_save"
    bl_label = "Save Bone Map"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        s = context.scene.rigger_remap
        data = [{"source": i.source, "target": i.target, "location": i.location,
                 "rot_offset": list(i.rot_offset), "loc_offset": list(i.loc_offset),
                 "location_mode": i.location_mode, "loc_multiplier": i.loc_multiplier} for i in s.items]
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump({"rigger_bone_map": 1, "pairs": data}, fh, indent=2)
        return {'FINISHED'}


class RIGGER_OT_remap_load(ImportHelper, bpy.types.Operator):
    """Load a bone map preset saved as JSON"""
    bl_idname = "rigger.remap_load"
    bl_label = "Load Bone Map"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})
    clear: BoolProperty(name="Clear Current List", default=True,
                        description="Off: keep the pairs of the current list that the preset does not have")

    def execute(self, context):
        try:
            with open(self.filepath, encoding="utf-8") as fh:
                data = json.load(fh)
            pairs = data["pairs"]
        except (OSError, ValueError, KeyError) as exc:
            self.report({'ERROR'}, f"Invalid bone map: {exc}")
            return {'CANCELLED'}
        s = context.scene.rigger_remap
        if self.clear:
            s.items.clear()
        existing = {i.source: i for i in s.items}
        for pair in pairs:
            source = str(pair.get("source", ""))
            if s.namespace_from:
                source = source.replace(s.namespace_from, s.namespace_to)
            item = existing.get(source) or s.items.add()
            item.source = source
            item.target = str(pair.get("target", ""))
            item.location = bool(pair.get("location", False))
            item.rot_offset = pair.get("rot_offset", (0.0, 0.0, 0.0))
            item.loc_offset = pair.get("loc_offset", (0.0, 0.0, 0.0))
            item.location_mode = pair.get("location_mode", 'WORLD')
            item.loc_multiplier = pair.get("loc_multiplier", 1.0)
        return {'FINISHED'}


class RIGGER_OT_remap_namespace(_RemapPoll, bpy.types.Operator):
    """Replace a prefix in the source names of the bone map (e.g. 'mixamorig:' -> 'mixamorig1:')"""
    bl_idname = "rigger.remap_namespace"
    bl_label = "Replace Namespace"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_remap
        if not s.namespace_from:
            return {'CANCELLED'}
        for item in s.items:
            item.source = item.source.replace(s.namespace_from, s.namespace_to)
        s.namespace_from = s.namespace_to = ""
        return {'FINISHED'}


class RIGGER_OT_remap_mirror(_RemapPoll, bpy.types.Operator):
    """Add the other side's pair for every sided pair of the bone map"""
    bl_idname = "rigger.remap_mirror"
    bl_label = "Mirror Bone Map"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_remap
        existing = {i.source for i in s.items}
        added = 0
        for item in list(s.items):
            src, tgt = mirror_any(item.source), mirror_any(item.target)
            if src == item.source or src in existing or src not in s.source.pose.bones \
                    or tgt not in s.target.pose.bones:
                continue
            new = s.items.add()
            new.source, new.target, new.location = src, tgt, item.location
            new.location_mode, new.loc_multiplier = item.location_mode, item.loc_multiplier
            r = item.rot_offset
            new.rot_offset = (r[0], -r[1], -r[2])
            existing.add(src)
            added += 1
        self.report({'INFO'}, f"Added {added} mirrored pairs")
        return {'FINISHED'}


def mirror_any(name):
    """Mirror a bone name with any side convention (Left/Right, .L/_l, L_ prefix...)."""
    swaps = (("Left", "Right"), ("left", "right"), ("LEFT", "RIGHT"))
    for a, b in swaps:
        if a in name:
            return name.replace(a, b)
        if b in name:
            return name.replace(b, a)
    mirrored = utils.mirror_name(name)
    if mirrored != name:
        return mirrored
    m = re.match(r"^([LRlr])([_.\- ])(.*)$", name)
    if m:
        return {"L": "R", "R": "L", "l": "r", "r": "l"}[m.group(1)] + m.group(2) + m.group(3)
    return name


class RIGGER_OT_remap_select(_RemapPoll, bpy.types.Operator):
    """Make the pair of the active bone (source or target, in Pose Mode) the active pair of the list"""
    bl_idname = "rigger.remap_select"
    bl_label = "Select in List"

    def execute(self, context):
        s = context.scene.rigger_remap
        pb = context.active_pose_bone
        if pb is None:
            return {'CANCELLED'}
        for index, item in enumerate(s.items):
            if (context.object == s.target and item.target == pb.name) or \
                    (context.object == s.source and item.source == pb.name):
                s.active_index = index
                return {'FINISHED'}
        self.report({'WARNING'}, f"{pb.name} is not in the bone map")
        return {'CANCELLED'}


class RIGGER_OT_remap_preview(_RemapPoll, bpy.types.Operator):
    """Bind the target to the source live, without keying, to check the bone map (click again to unbind)"""
    bl_idname = "rigger.remap_preview"
    bl_label = "Bind Preview"

    def execute(self, context):
        if preview_active():
            preview_stop()
            self.report({'INFO'}, "Unbound the preview")
            return {'FINISHED'}
        try:
            preview_start(context, context.scene.rigger_remap)
        except ValueError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class RIGGER_OT_remap_actions_select(_RemapPoll, bpy.types.Operator):
    """Tick or untick every source action for Multiple Source Animations"""
    bl_idname = "rigger.remap_actions_select"
    bl_label = "Select Source Actions"
    bl_options = {'REGISTER', 'UNDO'}

    state: BoolProperty(default=True)

    def execute(self, context):
        s = context.scene.rigger_remap
        names = set(s.source.pose.bones.keys())
        for action in bpy.data.actions:
            if any(fc.data_path.startswith('pose.bones["') and fc.data_path.split('"')[1] in names
                   for fc in utils.action_fcurves(action)):
                action.rigger_remap = self.state
        return {'FINISHED'}


classes = (RIGGER_OT_remap_auto, RIGGER_OT_remap_add, RIGGER_OT_remap_remove,
           RIGGER_OT_remap_bake, RIGGER_OT_remap_save, RIGGER_OT_remap_load, RIGGER_OT_remap_namespace,
           RIGGER_OT_remap_mirror, RIGGER_OT_remap_select, RIGGER_OT_remap_preview,
           RIGGER_OT_remap_actions_select)
