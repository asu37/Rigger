# SPDX-License-Identifier: GPL-3.0-or-later
"""Animator tools for generated rigs: IK/FK snapping and pose reset."""

import bpy
from bpy.props import BoolProperty, IntProperty, StringProperty
from mathutils import Matrix

from . import utils


def _update(context):
    context.view_layer.update()


def _autokey(context, rig, bones, prop_bone=None):
    if not context.scene.tool_settings.use_keyframe_insert_auto:
        return
    frame = context.scene.frame_current
    for name in bones:
        pb = rig.pose.bones[name]
        pb.keyframe_insert("location", frame=frame, group=name)
        if pb.rotation_mode == 'QUATERNION':
            pb.keyframe_insert("rotation_quaternion", frame=frame, group=name)
        else:
            pb.keyframe_insert("rotation_euler", frame=frame, group=name)
        pb.keyframe_insert("scale", frame=frame, group=name)
    if prop_bone:
        rig.pose.bones[prop_bone].keyframe_insert('["ik_fk"]', frame=frame, group=prop_bone)


def snap_fk_to_ik(context, rig, limb):
    """Match the FK controls to the current deformation and switch to FK."""
    pbs = rig.pose.bones
    sources = limb.get("blend") or limb["ik"]
    pairs = list(zip(limb["fk"], sources))
    if limb.get("toe_fk") and limb.get("toe_ik"):
        pairs.append((limb["toe_fk"], limb["toe_ik"]))
    matrices = [pbs[src].matrix.copy() for _fk, src in pairs]
    for (fk, _src), matrix in zip(pairs, matrices):
        pbs[fk].matrix = matrix
        _update(context)
    pbs[limb["prop_bone"]]["ik_fk"] = 0.0
    _update(context)
    _autokey(context, rig, [fk for fk, _src in pairs], limb["prop_bone"])


def snap_ik_to_fk(context, rig, limb):
    """Match the IK controls to the current FK pose and switch to IK."""
    pbs = rig.pose.bones
    fk = [pbs[n] for n in limb["fk"]]
    ik_ctrl = pbs[limb["ik_ctrl"]]
    target = pbs[limb["ik_target"]]
    for name in limb.get("ik_reset", []) + [limb.get("foot_roll")]:
        if name and name in pbs:
            pbs[name].matrix_basis = Matrix()
    if limb.get("ik_rot"):
        pbs[limb["ik_rot"]].matrix = fk[0].matrix.copy()
        _update(context)
    # Rest offset between the IK target bone and the IK control.
    offset = target.bone.matrix_local.inverted() @ ik_ctrl.bone.matrix_local
    ik_ctrl.matrix = fk[-1].matrix @ offset
    _update(context)
    if limb.get("toe_fk") and limb.get("toe_ik"):
        pbs[limb["toe_ik"]].matrix = pbs[limb["toe_fk"]].matrix.copy()
        _update(context)

    pole = pbs[limb["pole"]]
    a, b, c = fk[-3].head, fk[-2].head, fk[-2].tail
    rest_joint = fk[-2].bone.head_local
    distance = (pole.bone.head_local - rest_joint).length
    bend = utils.bend_direction(a, b, c, pole.head - b)
    mat = pole.bone.matrix_local.copy()
    mat.translation = b + bend * distance
    pole.matrix = mat
    pbs[limb["prop_bone"]]["ik_fk"] = 1.0
    _update(context)
    keyed = [limb["ik_ctrl"], limb["pole"]]
    keyed += [n for n in [limb.get("foot_roll"), limb.get("toe_ik"), limb.get("ik_rot")]
              + limb.get("ik_reset", []) if n]
    _autokey(context, rig, keyed, limb["prop_bone"])


class _LimbOperator:
    limb: IntProperty(name="Limb", default=0)

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def get_limb(self, context):
        limbs = utils.get_meta(context.object).get("limbs", [])
        if not 0 <= self.limb < len(limbs):
            return None
        return limbs[self.limb]


class RIGGER_OT_snap_fk_to_ik(_LimbOperator, bpy.types.Operator):
    """Match the FK controls to the current IK pose and switch to FK"""
    bl_idname = "rigger.snap_fk_to_ik"
    bl_label = "Snap FK to IK"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        limb = self.get_limb(context)
        if limb is None:
            return {'CANCELLED'}
        snap_fk_to_ik(context, context.object, limb)
        return {'FINISHED'}


class RIGGER_OT_snap_ik_to_fk(_LimbOperator, bpy.types.Operator):
    """Match the IK controls to the current FK pose and switch to IK"""
    bl_idname = "rigger.snap_ik_to_fk"
    bl_label = "Snap IK to FK"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        limb = self.get_limb(context)
        if limb is None:
            return {'CANCELLED'}
        snap_ik_to_fk(context, context.object, limb)
        return {'FINISHED'}


class RIGGER_OT_reset_pose(bpy.types.Operator):
    """Reset controls to their rest pose"""
    bl_idname = "rigger.reset_pose"
    bl_label = "Reset Pose"
    bl_options = {'REGISTER', 'UNDO'}

    selected_only: BoolProperty(name="Selected Only", default=False)

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object)

    def execute(self, context):
        rig = context.object
        controls = utils.get_meta(rig).get("controls", [])
        for name in controls:
            pb = rig.pose.bones.get(name)
            if pb is None:
                continue
            if self.selected_only and not utils.is_pose_bone_selected(pb):
                continue
            pb.matrix_basis = Matrix()
        return {'FINISHED'}


class RIGGER_OT_switch_space(bpy.types.Operator):
    """Change what this control follows without moving it"""
    bl_idname = "rigger.switch_space"
    bl_label = "Switch Parent"
    bl_options = {'REGISTER', 'UNDO'}

    bone: StringProperty(name="Bone")
    space: IntProperty(name="Space", default=0, min=0)

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def execute(self, context):
        rig = context.object
        pb = rig.pose.bones.get(self.bone)
        if pb is None or "parent" not in pb:
            return {'CANCELLED'}
        matrix = pb.matrix.copy()
        pb["parent"] = self.space
        rig.update_tag()
        _update(context)
        pb.matrix = matrix
        _update(context)
        if context.scene.tool_settings.use_keyframe_insert_auto:
            frame = context.scene.frame_current
            pb.keyframe_insert('["parent"]', frame=frame, group=pb.name)
            _autokey(context, rig, [pb.name])
        return {'FINISHED'}


class RIGGER_OT_select_controls(bpy.types.Operator):
    """Select rig controls (Shift: add to selection)"""
    bl_idname = "rigger.select_controls"
    bl_label = "Select Controls"
    bl_options = {'REGISTER', 'UNDO'}

    bones: StringProperty(name="Bones", description="Comma separated bone names")
    extend: BoolProperty(name="Extend", default=False, options={'SKIP_SAVE'})

    @classmethod
    def poll(cls, context):
        return utils.is_rig(context.object) and context.object.mode == 'POSE'

    def invoke(self, context, event):
        self.extend = event.shift
        return self.execute(context)

    def execute(self, context):
        rig = context.object
        names = [n for n in self.bones.split(",") if n in rig.pose.bones]
        if not self.extend:
            for pb in rig.pose.bones:
                utils.select_pose_bone(pb, False)
        for name in names:
            pb = rig.pose.bones[name]
            if pb.bone.hide:
                continue
            utils.select_pose_bone(pb, True)
            rig.data.bones.active = pb.bone
        return {'FINISHED'}


classes = (RIGGER_OT_snap_fk_to_ik, RIGGER_OT_snap_ik_to_fk, RIGGER_OT_reset_pose,
           RIGGER_OT_switch_space, RIGGER_OT_select_controls)
