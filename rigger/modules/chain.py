# SPDX-License-Identifier: GPL-3.0-or-later
"""Single bones, FK chains (tails, ears, hair, props) and fingers."""

import math

from .. import utils
from .base import RigModule


class SimpleModule(RigModule):
    type_id = 'SIMPLE'
    label = "Simple Bone"
    description = "One deforming bone with one FK control (jaw, props, accessories)"
    chain_length = 1

    def edit(self):
        bone = self.chain[0]
        parent = self.parent_deform
        self.ctrl = self.copy_ref(bone, bone, parent,
                                  collection=self.settings.collection or self.collection)
        self.make_deform(bone, parent)

    def pose(self):
        bone = self.chain[0]
        ref = self.gen.bones[bone]
        self.shape(self.ctrl, 'CIRCLE', ref.length * 0.35, y=0.5,
                   color="FACE" if self.settings.collection == "Face" else None,
                   fit=[self.gen.deform_of[bone]])
        self.copy_transforms(self.gen.deform_of[bone], self.ctrl)


class ChainModule(RigModule):
    type_id = 'CHAIN'
    label = "FK Chain"
    description = "Chain of FK controls with an optional curl master (tails, ears, hair)"
    min_length = 1

    margin = 1.3

    @property
    def collection(self):
        return self.settings.collection or "Extra"

    def ring_radius(self, ref_name):
        return max(self.gen.bones[ref_name].length * 0.3, self.h * 0.01)

    def shape_kind(self):
        return 'CIRCLE'

    def edit(self):
        gen = self.gen
        parent = self.parent_deform
        self.master = None
        if self.settings.curl and len(self.chain) > 1:
            first, last = gen.bones[self.chain[0]], gen.bones[self.chain[-1]]
            self.master = gen.new_bone(self.name(self.chain[0], "_master").replace(".01_", "_"),
                                       first.head, last.tail, first.z_axis,
                                       parent=parent, collection=self.collection)
        self.ctrls = []
        ctrl_parent = self.master or parent
        def_parent = parent
        for i, ref_name in enumerate(self.chain):
            connect = i > 0 and gen.bones[ref_name].connected
            ctrl = self.copy_ref(ref_name, ref_name, ctrl_parent,
                                 connect=connect, collection=self.collection)
            self.ctrls.append(ctrl)
            ctrl_parent = ctrl
            def_parent = self.make_deform(ref_name, def_parent, connect=connect)

    def pose(self):
        rig = self.gen.rig
        if self.master:
            pb = self.shape(self.master, 'MASTER', self.gen.rig.data.bones[self.master].length)
            pb.lock_location = (True, True, True)
        for i, (ref_name, ctrl) in enumerate(zip(self.chain, self.ctrls)):
            self.shape(ctrl, self.shape_kind(), self.ring_radius(ref_name), y=0.5,
                       fit=[self.gen.deform_of[ref_name]], margin=self.margin)
            if self.master and i > 0:
                # Rotating the master curls every segment.
                utils.add_constraint(rig.pose.bones[ctrl], 'COPY_ROTATION', rig, self.master,
                                     mix_mode='ADD', owner_space='LOCAL', target_space='LOCAL')
            self.copy_transforms(self.gen.deform_of[ref_name], ctrl)


class FingerModule(ChainModule):
    """Finger or toe: FK chain with a curl master, optional IK chain, grasp and
    rotate-from-scale driven by the limb it belongs to."""
    type_id = 'FINGER'
    label = "Finger"
    description = "Finger or toe chain with per-segment FK controls, a curl master and optional IK"
    margin = 1.5

    def ring_radius(self, ref_name):
        return self.h * 0.007

    @property
    def collection(self):
        return self.settings.collection or "Fingers"

    def _limb(self):
        from .limbs import LimbModule
        parent = self.gen.bones[self.root].parent
        if parent is not None and self.stem(parent).endswith("_meta"):
            parent = self.gen.bones[parent].parent  # finger under a metacarpal
        for module in self.gen.modules:
            if isinstance(module, LimbModule) and parent in (getattr(module, "end_ref", None),
                                                              getattr(module, "toe_ref", None)):
                return module
        return None

    def edit(self):
        super().edit()
        gen, s = self.gen, self.settings
        parent = self.parent_deform
        self.limb = self._limb()
        self.digit = self.stem(self.chain[0]).split(".")[0]
        stem, suffix = self.digit, self.suffix
        first = gen.bones[self.chain[0]]
        self.curl_mch = self.grasp_mch = None
        if self.limb is not None:
            self.curl_mch = gen.new_bone(f"MCH-{stem}_curl{suffix}", first.head, first.tail, first.z_axis,
                                         parent=parent, collection="Mechanism")
            if self.limb.settings.rot_from_scale == 'LAST_TWO':
                self.grasp_mch = gen.new_bone(f"MCH-{stem}_grasp{suffix}", first.head, first.tail, first.z_axis,
                                              parent=parent, collection="Mechanism")
        self.ik, self.ik_ctrl, self.pole = [], None, None
        if s.finger_ik and len(self.chain) >= 2:
            prev = parent
            for i, ref_name in enumerate(self.chain):
                prev = self.copy_ref(ref_name, "MCH-" + self.name(ref_name, "_ik"), prev,
                                     connect=i > 0 and gen.bones[ref_name].connected, collection="Mechanism")
                self.ik.append(prev)
            last = gen.bones[self.chain[-1]]
            tip_dir = (last.tail - last.head)
            self.ik_ctrl = gen.new_bone(f"{stem}_ik{suffix}", last.tail, last.tail + tip_dir * 0.6, last.z_axis,
                                        parent=parent, collection=self.collection)
            joint = gen.bones[self.chain[len(self.chain) // 2]].head
            bend = utils.bend_direction(first.head, joint, last.tail, first.z_axis)
            total = sum(gen.bones[n].length for n in self.chain)
            pole_pos = joint + bend * total * 0.5 * s.finger_pole_distance
            self.pole = gen.new_bone(f"{stem}_pole{suffix}", pole_pos, pole_pos + bend * total * 0.1,
                                     parent=self.ik_ctrl, collection=self.collection)
            base = gen.edit_bones[self.ik[0]]
            self.pole_angle = utils.pole_angle(base.head, base.tail, base.x_axis,
                                               gen.edit_bones[self.ik[-1]].tail, pole_pos)
            gen.post_edit.append(lambda: gen.add_space_switch(self.ik_ctrl, [
                ("Hand" if self.limb is None or self.limb.limb_type == "arm" else "Foot", parent),
                ("Root", "root")]))
        self.meta = {"type": "finger", "fk": list(self.ctrls), "ik": self.ik, "ik_ctrl": self.ik_ctrl,
                     "pole": self.pole, "prop_bone": self.ik_ctrl, "master": self.master,
                     "label": self.digit + (f" {self.side}" if self.side else "")}

    def shape_kind(self):
        return 'BOX' if self.settings.finger_shape == 'BOX' else 'CIRCLE'

    def _curl_pose(self):
        """Grasp (limb property) and rotate-from-scale curl every phalange around X."""
        if self.limb is None or self.curl_mch is None:
            return
        rig = self.gen.rig
        pbs = rig.pose.bones
        limb = self.limb
        prop_name = "fingers_grasp" if limb.limb_type == "arm" else "toes_grasp"
        prop = pbs[limb.prop_bone]
        if prop_name not in prop:
            utils.add_property(prop, prop_name, 0.0, -1.0, 1.0, "Curl every finger / toe (negative opens)")
        angle = math.radians(80.0) * (0.6 if self.digit.startswith("thumb") else 1.0)
        grasp = [("g", 'prop', limb.prop_bone, prop_name)]
        mode = limb.settings.rot_from_scale
        curl = pbs[self.curl_mch]
        curl.rotation_mode = 'XYZ'
        if mode == 'NONE':
            utils.add_driver_mixed(curl, "rotation_euler", rig, grasp, f"-g * {angle:.5f}", index=0)
        else:
            utils.add_driver_mixed(curl, "rotation_euler", rig,
                                   grasp + [("sy", 'obj', f'pose.bones["{self.ctrls[0]}"].scale[1]')],
                                   f"-(g * {angle:.5f} + (1.0 - sy) * {math.pi / 2:.5f})", index=0)
            utils.add_constraint(pbs[self.ctrls[0]], 'LIMIT_SCALE', use_min_y=True, min_y=1.0, use_max_y=True,
                                 max_y=1.0, owner_space='LOCAL')
        sources = [self.curl_mch] * len(self.ctrls)
        if self.grasp_mch:
            grasp_pb = pbs[self.grasp_mch]
            grasp_pb.rotation_mode = 'XYZ'
            utils.add_driver_mixed(grasp_pb, "rotation_euler", rig, grasp, f"-g * {angle:.5f}", index=0)
            sources[0] = self.grasp_mch
        for ctrl, source in zip(self.ctrls, sources):
            utils.add_constraint(pbs[ctrl], 'COPY_ROTATION', rig, source, mix_mode='ADD',
                                 owner_space='LOCAL', target_space='LOCAL')

    def pose(self):
        super().pose()
        self._curl_pose()
        if not self.ik:
            return
        rig = self.gen.rig
        pbs = rig.pose.bones
        utils.add_constraint(pbs[self.ik[-1]], 'IK', rig, self.ik_ctrl, pole_target=rig, pole_subtarget=self.pole,
                             pole_angle=self.pole_angle, chain_count=len(self.ik), use_tail=True,
                             iterations=2000)
        ctrl = self.shape(self.ik_ctrl, 'SPHERE', self.h * 0.006)
        ctrl.lock_scale = (True, True, True)
        utils.add_property(ctrl, "ik_fk", 0.0, description="0 = FK, 1 = IK")
        self.shape(self.pole, 'POLE', self.h * 0.004)
        for ref_name, ik in zip(self.chain, self.ik):
            con = self.copy_transforms(self.gen.deform_of[ref_name], ik)
            utils.add_driver(con, "influence", rig, self.ik_ctrl, "ik_fk")
        if self.gen.auto_hide:
            arm = rig.data
            for name in self.ctrls + ([self.master] if self.master else []):
                utils.add_driver(arm, f'bones["{name}"].hide', rig, self.ik_ctrl, "ik_fk", "var > 0.5")
            for name in (self.ik_ctrl, self.pole):
                utils.add_driver(arm, f'bones["{name}"].hide', rig, self.ik_ctrl, "ik_fk", "var < 0.5")


class BendyModule(RigModule):
    """Chain of bendy bones shaped by a control at every joint."""
    type_id = 'BBONE'
    label = "Bendy Chain"
    description = "Bendy-bone chain with a control at every joint (hair, snakes, noodles). Blender only"
    min_length = 1

    @property
    def collection(self):
        return self.settings.collection or "Extra"

    def edit(self):
        gen, s = self.gen, self.settings
        parent = self.parent_deform
        bones = [gen.bones[n] for n in self.chain]
        points = [bones[0].head] + [b.tail for b in bones]
        stem = self.stem(self.chain[0]).split(".")[0]
        first_dir = (points[1] - points[0]).normalized()
        self.master = gen.new_bone(f"{stem}_bendy_master{self.suffix}", points[0],
                                   points[0] + first_dir * bones[0].length * 0.5, bones[0].z_axis,
                                   parent=parent, collection=self.collection)
        self.ctrls = []
        for i, point in enumerate(points):
            ref = bones[min(i, len(bones) - 1)]
            direction = ((points[i + 1] - point) if i < len(bones) else (point - points[i - 1])).normalized()
            self.ctrls.append(gen.new_bone(f"{stem}_bendy.{i + 1:02d}{self.suffix}", point,
                                           point + direction * ref.length * 0.3, ref.z_axis,
                                           parent=self.master, collection=self.collection))
        prev = parent
        for i, ref_name in enumerate(self.chain):
            prev = self.make_deform(ref_name, prev, connect=i > 0 and gen.bones[ref_name].connected)
            eb = gen.edit_bones[prev]
            eb.bbone_segments = s.bbone_segments
            eb.bbone_handle_type_start = 'TANGENT'
            eb.bbone_custom_handle_start = gen.edit_bones[self.ctrls[i]]
            eb.bbone_handle_type_end = 'TANGENT'
            eb.bbone_custom_handle_end = gen.edit_bones[self.ctrls[i + 1]]
            for attr in ("bbone_handle_use_scale_start", "bbone_handle_use_scale_end"):
                if hasattr(eb, attr):
                    setattr(eb, attr, (True, False, True))
        self.meta = {"type": "bendy", "controls": self.ctrls, "master": self.master}

    def pose(self):
        rig = self.gen.rig
        pbs = rig.pose.bones
        scale = self.settings.bbone_scale
        for i, ref_name in enumerate(self.chain):
            deform = pbs[self.gen.deform_of[ref_name]]
            utils.add_constraint(deform, 'COPY_LOCATION', rig, self.ctrls[i])
            utils.add_constraint(deform, 'STRETCH_TO', rig, self.ctrls[i + 1],
                                 rest_length=rig.data.bones[deform.name].length, volume='NO_VOLUME')
        length = sum(self.gen.bones[n].length for n in self.chain)
        self.shape(self.master, 'CIRCLE', length * 0.1 * scale)
        for i, ctrl in enumerate(self.ctrls):
            ref = self.gen.bones[self.chain[min(i, len(self.chain) - 1)]]
            self.shape(ctrl, 'CIRCLE', ref.length * 0.3 * scale, fit=[self.gen.deform_of[ref.name]], margin=1.3)
