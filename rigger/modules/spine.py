# SPDX-License-Identifier: GPL-3.0-or-later
"""Spine (hips + FK spine with a torso control) and neck/head modules.

Spine options: spine master (move, rotate and stretch the whole spine), bottom
(buttock) bones following the thighs, and a reversed FK spine rotating from the
chest down, switched by the torso's ``spine_reverse`` slider.

Neck options: neck twist following the head, bendy neck bones and three skull
controls moving the lower (mouth), middle (eyes, nose) and top parts of the head.
"""

import math

from mathutils import Vector

from .. import utils
from .base import RigModule


class SpineModule(RigModule):
    type_id = 'SPINE'
    label = "Spine"
    description = "Hips bone followed by spine bones. Adds torso, hips and FK spine controls"
    min_length = 2
    collection = "Torso"

    def edit(self):
        gen, s = self.gen, self.settings
        parent = self.parent_deform
        hips_ref = gen.bones[self.chain[0]]
        size = hips_ref.length
        top_ref = gen.bones[self.chain[-1]]

        # Torso: moves the whole upper body, pivots at the pelvis. Multi-peds point it along the spine.
        if gen.ref.rigger.multiped:
            up = (top_ref.tail - hips_ref.head).normalized()
            torso_z = hips_ref.z_axis
        else:
            up, torso_z = Vector((0, 0, 1)), Vector((0, -1, 0))
        self.torso = gen.new_bone("torso", hips_ref.head, hips_ref.head + up * size * 1.5,
                                  torso_z, parent=parent, collection="Torso")
        # Hips control points down so the pelvis rotates around the waist.
        self.hips = gen.new_bone(self.stem(self.chain[0]), hips_ref.tail, hips_ref.head,
                                 hips_ref.z_axis, parent=self.torso, collection="Torso")
        self.hips_mch = self.copy_ref(self.chain[0], "MCH-" + self.chain[0], self.hips,
                                      collection="Mechanism")
        self.make_deform(self.chain[0], parent)

        # Spine master: moves, rotates and stretches every FK spine bone
        fk_parent = self.torso
        self.master = None
        if s.spine_master:
            first = gen.bones[self.chain[1]]
            self.master = gen.new_bone("spine_master", first.head, top_ref.tail, first.z_axis,
                                       parent=self.torso, collection="Torso")
            self.master_space = gen.new_bone("MCH-spine_master_space", first.head, top_ref.tail, first.z_axis,
                                             parent=self.torso, collection="Mechanism")
            self.master_squash = gen.new_bone("MCH-spine_master_squash", first.head, top_ref.tail, first.z_axis,
                                              parent=self.master_space, collection="Mechanism")
            fk_parent = self.master_squash

        self.ctrls = []
        ctrl_parent = fk_parent
        def_parent = gen.deform_of[self.chain[0]]
        self.deforms = [gen.deform_of[self.chain[0]]]
        for i, ref_name in enumerate(self.chain[1:]):
            connect = i > 0 and gen.bones[ref_name].connected
            ctrl = self.copy_ref(ref_name, ref_name, ctrl_parent, connect=connect)
            self.ctrls.append(ctrl)
            ctrl_parent = ctrl
            def_parent = self.make_deform(ref_name, def_parent,
                                          connect=gen.bones[ref_name].connected)
            self.deforms.append(def_parent)

        # Reversed spine: an FK chain from the chest down; the hips hang from its bottom.
        self.rev, self.rev_mch, self.hips_rev = [], [], None
        if s.reversed_spine:
            rev_parent = self.torso
            for ref_name in reversed(self.chain[1:]):
                ref = gen.bones[ref_name]
                stem, suffix = utils.split_side(ref_name)
                ctrl = gen.new_bone(f"{stem}_rev{suffix}", ref.tail, ref.head, ref.z_axis, parent=rev_parent,
                                    connect=rev_parent != self.torso, collection="Torso")
                self.rev.insert(0, ctrl)
                self.rev_mch.insert(0, gen.new_bone(f"MCH-{stem}_rev{suffix}", ref.head, ref.tail, ref.z_axis,
                                                    parent=ctrl, collection="Mechanism"))
                rev_parent = ctrl
            self.hips_rev = self.copy_ref(self.chain[0], f"MCH-{self.chain[0]}_rev", self.rev[0],
                                          collection="Mechanism")

        # Chest: rotating it bends the whole FK spine, spread over every segment.
        self.chest = gen.new_bone("chest", top_ref.head, top_ref.tail, top_ref.z_axis, parent=self.torso,
                                  collection="Torso")
        gen.spaces["Torso"] = self.torso
        gen.spaces["Chest"] = gen.deform_of[self.chain[-1]]
        self.bottoms = []
        if s.bottom:
            gen.post_edit.append(self._bottom_edit)
        self.meta = {"type": "spine", "torso": self.torso, "hips": self.hips,
                     "chest": self.chest, "fk": list(self.ctrls), "master": self.master,
                     "rev": self.rev, "rev_mch": self.rev_mch, "hips_rev": self.hips_rev,
                     "deform": self.deforms, "hips_mch": self.hips_mch, "bottoms": []}

    def _bottom_edit(self):
        """Buttock bones behind each thigh, under the hips; they follow the thighs a little."""
        gen = self.gen
        hips_def = gen.deform_of[self.chain[0]]
        thighs = [n for n in gen.edit_bones.keys() if n.startswith("DEF-thigh") and "twist" not in n
                  and "fan" not in n and utils.split_side(n)[1]]
        h = self.h
        for thigh in thighs:
            suffix = utils.split_side(thigh)[1]
            eb = gen.edit_bones[thigh]
            head = eb.head + Vector((0, 0.035 * h, -0.005 * h))
            tail = head + Vector((0, 0.04 * h, 0))
            space = gen.new_bone(f"MCH-bottom_space{suffix}", head, tail, Vector((0, 0, 1)), parent=hips_def,
                                 collection="Mechanism")
            follow = gen.new_bone(f"MCH-bottom_follow{suffix}", head, tail, Vector((0, 0, 1)), parent=thigh,
                                  collection="Mechanism")
            ctrl = gen.new_bone(f"bottom{suffix}", head, tail, Vector((0, 0, 1)), parent=space,
                                collection="Torso")
            deform = gen.new_bone(f"DEF-bottom{suffix}", head, tail, Vector((0, 0, 1)), parent=hips_def,
                                  deform=True, collection="Deform")
            self.bottoms.append((space, follow, ctrl, deform))
        self.meta["bottoms"] = [b[2] for b in self.bottoms]

    def pose(self):
        rig, s = self.gen.rig, self.settings
        pbs = rig.pose.bones
        h = self.h
        hips_def = self.gen.deform_of[self.chain[0]]
        torso = self.shape(self.torso, 'TORSO', h * 0.1, fit=[hips_def], margin=1.6, wire=2.5)
        torso.rotation_mode = 'QUATERNION'
        self.shape(self.hips, 'HIPS', h * 0.08, y=0.7, fit=[hips_def], margin=1.25)
        pbs[self.hips].lock_location = (True, True, True)
        self.copy_transforms(hips_def, self.hips_mch)
        top_def = self.gen.deform_of[self.chain[-1]]
        chest = self.shape(self.chest, 'CHEST', h * 0.09, y=0.6, fit=[top_def], margin=1.45)
        chest.lock_location = (True, True, True)
        share = 1.0 / len(self.ctrls)
        for ref_name, ctrl in zip(self.chain[1:], self.ctrls):
            utils.add_constraint(pbs[ctrl], 'COPY_ROTATION', rig, self.chest,
                                 mix_mode='ADD', owner_space='LOCAL',
                                 target_space='LOCAL_OWNER_ORIENT', influence=share)
            if self.master and s.spine_master_space == 'LOCAL':
                utils.add_constraint(pbs[ctrl], 'COPY_ROTATION', rig, self.master, mix_mode='ADD',
                                     owner_space='LOCAL', target_space='LOCAL_OWNER_ORIENT', influence=share)
            pb = self.shape(ctrl, 'CIRCLE', h * 0.075, y=0.5, fit=[self.gen.deform_of[ref_name]],
                            margin=1.15)
            pb.lock_location = (True, True, True)
            self.copy_transforms(self.gen.deform_of[ref_name], ctrl)

        if self.master:
            self.shape(self.master, 'SQUARE', h * 0.11, y=0.0, color="C", wire=2.5)
            space = pbs[self.master_space]
            local = dict(owner_space='LOCAL', target_space='LOCAL')
            utils.add_constraint(space, 'COPY_LOCATION', rig, self.master, **local)
            utils.add_constraint(space, 'COPY_SCALE', rig, self.master, **local)
            if s.spine_master_space == 'MASTER':
                utils.add_constraint(space, 'COPY_ROTATION', rig, self.master, **local)
            # Squash: the spine gets thinner as it stretches
            squash = pbs[self.master_squash]
            path = f'pose.bones["{self.master}"].scale[1]'
            for index in (0, 2):
                utils.add_driver_mixed(squash, "scale", rig, [("sy", 'obj', path)],
                                       "1.0 / sqrt(max(sy, 0.001))", index=index)

        if self.rev:
            utils.add_property(torso, "spine_reverse", 0.0,
                               description="0 = the spine rotates from the hips, 1 = from the chest down")
            for ref_name, ctrl, mch in zip(self.chain[1:], self.rev, self.rev_mch):
                pb = self.shape(ctrl, 'CIRCLE', h * 0.065, y=0.5, color="C")
                pb.lock_location = (True, True, True)
                con = self.copy_transforms(self.gen.deform_of[ref_name], mch)
                utils.add_driver(con, "influence", rig, self.torso, "spine_reverse")
            con = self.copy_transforms(hips_def, self.hips_rev)
            utils.add_driver(con, "influence", rig, self.torso, "spine_reverse")
            if self.gen.auto_hide:
                arm = rig.data
                for name in self.ctrls:
                    utils.add_driver(arm, f'bones["{name}"].hide', rig, self.torso, "spine_reverse", "var > 0.5")
                for name in self.rev:
                    utils.add_driver(arm, f'bones["{name}"].hide', rig, self.torso, "spine_reverse", "var < 0.5")

        for space, follow, ctrl, deform in self.bottoms:
            utils.add_constraint(pbs[space], 'COPY_TRANSFORMS', rig, follow, influence=0.35)
            self.shape(ctrl, 'CIRCLE', h * 0.04, y=0.5, color=utils.side_letter(ctrl) or "C", fit=[deform])
            self.copy_transforms(deform, ctrl)


class NeckModule(RigModule):
    type_id = 'NECK'
    label = "Neck & Head"
    description = "Neck bones followed by the head. The last bone of the chain is the head"
    min_length = 1
    collection = "Head"

    def edit(self):
        gen, s = self.gen, self.settings
        parent = self.parent_deform
        necks, head = self.chain[:-1], self.chain[-1]

        self.necks = []
        ctrl_parent = parent
        def_parent = parent
        self.neck_defs = []
        for i, ref_name in enumerate(necks):
            connect = i > 0 and gen.bones[ref_name].connected
            ctrl = self.copy_ref(ref_name, ref_name, ctrl_parent, connect=connect)
            self.necks.append(ctrl)
            ctrl_parent = ctrl
            def_parent = self.make_deform(ref_name, def_parent, connect=connect)
            self.neck_defs.append(def_parent)
            if s.neck_bendy > 1:
                gen.edit_bones[def_parent].bbone_segments = s.neck_bendy

        # Head isolation: blend between following the neck and keeping world rotation.
        self.head_space = self.copy_ref(head, "MCH-" + self.stem(head) + "_space" + self.suffix,
                                        ctrl_parent, collection="Mechanism")
        self.head_world = self.copy_ref(head, "MCH-" + self.stem(head) + "_world" + self.suffix,
                                        "root", collection="Mechanism")
        self.head = self.copy_ref(head, head, self.head_space)
        head_def = self.make_deform(head, def_parent, connect=gen.bones[head].connected and bool(necks))
        gen.spaces["Head"] = head_def
        if s.neck_bendy > 1 and self.neck_defs:
            eb = gen.edit_bones[self.neck_defs[-1]]
            eb.bbone_handle_type_end = 'TANGENT'
            eb.bbone_custom_handle_end = gen.edit_bones[head_def]

        # Skull controls: lower (mouth), middle (eyes, nose), top
        self.skulls = []
        if s.skulls:
            ref = gen.bones[head]
            bounds = ((0.0, 0.35), (0.35, 0.7), (0.7, 1.0))
            for i, (a, b) in enumerate(bounds, 1):
                h0, h1 = ref.head.lerp(ref.tail, a), ref.head.lerp(ref.tail, b)
                ctrl = gen.new_bone(f"skull_{i:02d}{self.suffix}", h0, h1, ref.z_axis, parent=self.head,
                                    collection="Head")
                deform = gen.new_bone(f"DEF-skull_{i:02d}{self.suffix}", h0, h1, ref.z_axis, parent=head_def,
                                      deform=True, collection="Deform")
                self.skulls.append((ctrl, deform))
            gen.post_edit.append(self._skulls_edit)
        self.meta = {"type": "neck", "head": self.head, "fk": list(self.necks),
                     "skulls": [c for c, _d in self.skulls]}

    def _skulls_edit(self):
        """Re-parent the face and head controls to the skull control of their region."""
        gen = self.gen
        head_def = gen.deform_of[self.chain[-1]]
        from .face import FaceModule
        face = next((m for m in gen.modules if isinstance(m, FaceModule)
                     and gen.bones[m.root].parent == self.chain[-1]), None)
        own = {self.head, self.head_space, self.head_world} | {n for pair in self.skulls for n in pair}
        holders = {head_def, self.head} | ({face.face} if face else set())
        ref = gen.bones[self.chain[-1]]
        axis = ref.tail - ref.head
        # With a face, the skulls follow the face root control so it still moves everything.
        if face:
            for ctrl, _d in self.skulls:
                gen.edit_bones[ctrl].parent = gen.edit_bones[face.face]
        for eb in list(gen.edit_bones):
            if eb.parent is None or eb.parent.name not in holders or eb.name in own:
                continue
            if eb.name.startswith("DEF-") or (face and eb.name == face.face):
                continue
            stem = utils.split_side(eb.name.replace("MCH-", ""))[0]
            if stem.startswith(("jaw", "chin", "lip", "teeth", "tongue", "mouth", "muzzle")):
                region = 0
            elif stem.startswith(("ear", "hair", "top")):
                region = 2
            else:
                t = (eb.head - ref.head).dot(axis) / max(axis.length_squared, 1e-9)
                region = 2 if t > 0.75 else 1
            eb.parent = gen.edit_bones[self.skulls[region][0]]

    def pose(self):
        rig, s = self.gen.rig, self.settings
        pbs = rig.pose.bones
        for ref_name, ctrl in zip(self.chain[:-1], self.necks):
            pb = self.shape(ctrl, 'CIRCLE', self.h * 0.035, y=0.5, fit=[self.gen.deform_of[ref_name]],
                            margin=1.3)
            pb.lock_location = (True, True, True)
            self.copy_transforms(self.gen.deform_of[ref_name], ctrl)

        head_def = self.gen.deform_of[self.chain[-1]]
        head = self.shape(self.head, 'CIRCLE', self.h * 0.06, y=0.72, fit=[head_def], margin=1.25)
        head.lock_location = (True, True, True)
        utils.add_property(head, "head_isolate", 0.0,
                           description="0 = head follows the neck, 1 = head keeps its world rotation")
        con = utils.add_constraint(pbs[self.head_space], 'COPY_ROTATION', rig,
                                   self.head_world, mix_mode='REPLACE',
                                   owner_space='WORLD', target_space='WORLD')
        utils.add_driver(con, "influence", rig, self.head, "head_isolate")
        self.copy_transforms(head_def, self.head)

        # Neck twist: each neck bone takes part of the head's twist
        if s.neck_twist and self.neck_defs:
            utils.add_property(head, "neck_twist", 1.0, 0.0, 1.0, "How much the neck twists with the head")
            count = len(self.neck_defs)
            for i, deform in enumerate(self.neck_defs):
                factor = (i + 1) / (count + 1)
                if s.neck_bendy > 1:
                    fcurve = utils.add_driver_mixed(pbs[deform], "bbone_rollout", rig, [
                        ("t", 'transform', self.head, 'ROT_Y', 'LOCAL_SPACE'),
                        ("k", 'prop', self.head, "neck_twist")], f"t * k * {factor * 2:.4f}")
                    fcurve.driver.variables["t"].targets[0].rotation_mode = 'SWING_TWIST_Y'
                    continue
                con = utils.add_constraint(
                    pbs[deform], 'TRANSFORM', rig, self.head, owner_space='LOCAL', target_space='LOCAL',
                    map_from='ROTATION', map_to='ROTATION', from_rotation_mode='SWING_TWIST_Y',
                    map_to_x_from='X', map_to_y_from='Y', map_to_z_from='Z', mix_mode_rot='ADD')
                con.from_min_y_rot, con.from_max_y_rot = -math.pi, math.pi
                con.to_min_y_rot, con.to_max_y_rot = -math.pi * factor, math.pi * factor
                utils.add_driver(con, "influence", rig, self.head, "neck_twist")

        for ctrl, deform in self.skulls:
            self.shape(ctrl, 'CIRCLE', self.h * 0.05, y=0.5, color="C", fit=[head_def], margin=1.1)
            self.copy_transforms(deform, ctrl)
