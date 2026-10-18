# SPDX-License-Identifier: GPL-3.0-or-later
"""Arm and leg modules: FK + IK chains blended onto deform bones.

Arm chain:  [shoulder,] upper_arm, forearm, hand
Leg chain:  [thigh_b (3 bones leg),] thigh, shin, foot[, toe]

Every limb gets:
* FK controls (optionally with an ``fk_lock`` parent space)
* an IK chain (mechanism bones) with an IK control, a pole control and stretch
* a blend chain (``MCH-*_blend``) mixing FK, IK and pinning, that the deform
  bones and their twist bones follow
* an ``ik_fk`` slider (0 = FK, 1 = IK) and stretch sliders on the IK control

Options (``pose_bone.rigger`` of the first reference bone): upper and lower
twist bones, secondary controls (bendy or per-twist tweaks), joint fans with
bulge, IK pole distance, soft IK, free roll, IK offset and pivot controls,
smart IK pole, 3 bones leg, foot roll break, toes pivot, digit spreading.
"""

import math

from mathutils import Vector

from .. import utils, widgets
from .base import RigModule

DEG = math.pi / 180.0


class LimbModule(RigModule):
    fallback_bend = Vector((0, 1, 0))
    ik_collection = "Arm IK"
    fk_collection = "Arm FK"
    limb_type = "arm"

    # -- edit helpers --------------------------------------------------------
    def _roll(self, ref):
        """Z axis of the limb bones: the bend plane, or the reference roll."""
        return self.bend if self.settings.auto_ik_roll else self.gen.bones[ref].z_axis

    def _chain(self, refs, kind, prefix, parent, collection):
        names = []
        for i, ref in enumerate(refs):
            is_end = ref == self.end_ref
            connect = i > 0 and (self.gen.bones[ref].connected if is_end else True)
            names.append(self.copy_ref(ref, self.name(ref, kind, prefix), parent, connect=connect,
                                       collection=collection, z_axis=None if is_end else self._roll(ref)))
            parent = names[-1]
        return names

    def _build(self, uppers, lower, end, socket_parent):
        """Build FK, IK, blend and deform chains for [uppers...], lower and end."""
        gen, s = self.gen, self.settings
        upper = uppers[-1]
        self.three = len(uppers) > 1
        self.end_ref = end
        self.seg = list(uppers) + [lower, end]
        a, b = gen.bones[upper].head, gen.bones[lower].head
        c = gen.bones[lower].tail
        bend = utils.bend_direction(a, b, c, self.fallback_bend)
        self.bend = bend
        limb_length = (b - a).length + (c - b).length
        self.limb_length = limb_length
        first = gen.bones[uppers[0]]

        self.socket = gen.new_bone(self.name(uppers[0], "_socket", "MCH-"), first.head,
                                   first.head + (first.tail - first.head).normalized() * limb_length * 0.1,
                                   first.z_axis, parent=socket_parent, collection="Mechanism")

        # FK chain, optionally in a space that can keep its world rotation
        fk_parent = self.socket
        self.fk_space = None
        if s.fk_lock:
            roll = self._roll(uppers[0])
            self.fk_space = gen.new_bone(self.name(uppers[0], "_fk_space", "MCH-"), first.head, first.tail,
                                         roll, parent=self.socket, collection="Mechanism")
            self.fk_world = gen.new_bone(self.name(uppers[0], "_fk_world", "MCH-"), first.head, first.tail,
                                         roll, parent="root", collection="Mechanism")
            fk_parent = self.fk_space
        self.fk = self._chain(self.seg, "_fk", "", fk_parent, self.fk_collection)

        # IK chains. 3 bones legs: chain B = rotation control + 2 bones IK, chain A = 3 bones IK.
        self.ik3 = []
        if self.three:
            self.ik_rot = self.copy_ref(uppers[0], self.name(uppers[0], "_ik"), self.socket,
                                        collection=self.ik_collection, z_axis=self._roll(uppers[0]))
            self.ik = [self.ik_rot] + self._chain(self.seg[1:], "_ik", "MCH-", self.ik_rot, "Mechanism")
            self.ik3 = self._chain(self.seg[:-1], "_ik3", "MCH-", self.socket, "Mechanism")
        else:
            self.ik_rot = None
            self.ik = self._chain(self.seg, "_ik", "MCH-", self.socket, "Mechanism")
        self.blend = self._chain(self.seg, "_blend", "MCH-", self.socket, "Mechanism")

        # Pole
        pole_pos = b + bend * limb_length * s.pole_distance
        self.pole = gen.new_bone(self.name(upper, "_pole"), pole_pos,
                                 pole_pos + bend * limb_length * 0.08, Vector((0, 0, 1)),
                                 parent="root", collection=self.ik_collection)
        ik_upper = gen.edit_bones[self.ik[-3]]
        self.pole_angle = utils.pole_angle(ik_upper.head, ik_upper.tail, ik_upper.x_axis,
                                           gen.edit_bones[self.ik[-2]].tail, pole_pos)
        if self.three:
            base = gen.edit_bones[self.ik3[0]]
            self.pole_angle3 = utils.pole_angle(base.head, base.tail, base.x_axis,
                                                gen.edit_bones[self.ik3[-1]].tail, pole_pos)
        # Line from the elbow/knee to the pole, drawn in IK mode
        self.pole_line = gen.new_bone(self.name(upper, "_pole_line", "MCH-"), b, pole_pos, Vector((0, 0, 1)),
                                      parent=self.blend[-2], collection=self.ik_collection)

        self._deform_edit(uppers, upper, lower, end, socket_parent)
        self._extras_edit(upper, lower, bend)

    def _deform_edit(self, uppers, upper, lower, end, socket_parent):
        gen, s = self.gen, self.settings
        mode = s.secondary_type if s.secondary else 'NONE'
        self.mode = mode
        nu = s.twist_upper
        nl = max(0, s.twist)
        self.def_sources = []      # (deform bone, bone it copies)
        self.upper_segments = []   # (MCH source, fraction along the upper bone, twist kept)
        self.lower_segments = []   # (MCH source, twist fraction)
        self.twist_tweaks = []
        blend = dict(zip(self.seg, self.blend))

        parent = socket_parent
        deform = []
        if self.three:
            parent = self.make_deform(uppers[0], socket_parent)
            deform.append(parent)
            self.def_sources.append((parent, blend[uppers[0]]))

        # Upper segment: the bones near the shoulder / hip keep less of the twist
        up = gen.bones[upper]
        roll = self._roll(upper)
        def_upper = self.make_deform(upper, parent, connect=self.three and up.connected,
                                     tail=up.head.lerp(up.tail, 1.0 / (nu + 1)))
        gen.edit_bones[def_upper].align_roll(roll)
        self.upper_twists = []
        self.swing = None
        if nu:
            self.swing = gen.new_bone(self.name(upper, "_swing", "MCH-"), up.head, up.tail, roll,
                                      parent=self.blend[0] if self.three else self.socket,
                                      collection="Mechanism")
        for i in range(nu + 1):
            head = up.head.lerp(up.tail, i / (nu + 1))
            tail = up.head.lerp(up.tail, (i + 1) / (nu + 1))
            if i == 0:
                name = def_upper
            else:
                name = gen.new_bone(f"DEF-{self.stem(upper)}_twist.{i:02d}{self.suffix}", head, tail, roll,
                                    parent=def_upper, deform=True, collection="Deform")
                self.upper_twists.append(name)
            source = blend[upper]
            if nu:
                source = gen.new_bone(self.name(upper, f"_seg.{i:02d}", "MCH-"), head, tail, roll,
                                      parent=self.socket, collection="Mechanism")
                # twist kept: none at the shoulder/hip piece, all of it at the elbow/knee piece;
                # the skin weights blend between pieces, so the twist spreads evenly in between
                self.upper_segments.append((source, i / (nu + 1), i / nu))
            self.def_sources.append((name, source))
        deform.append(def_upper)

        # Lower segment: twist bones follow the hand / foot twist more toward the end
        low = gen.bones[lower]
        roll = self._roll(lower)
        segments = nl + 1
        def_lower = self.make_deform(lower, def_upper, connect=True,
                                     tail=low.head.lerp(low.tail, 1.0 / segments))
        gen.edit_bones[def_lower].align_roll(roll)
        self.def_sources.append((def_lower, blend[lower]))
        self.twists = []
        for i in range(1, segments):
            head = low.head.lerp(low.tail, i / segments)
            tail = low.head.lerp(low.tail, (i + 1) / segments)
            name = gen.new_bone(f"DEF-{self.stem(lower)}_twist.{i:02d}{self.suffix}", head, tail,
                                roll, parent=def_lower, deform=True, collection="Deform")
            source = gen.new_bone(self.name(lower, f"_twist.{i:02d}", "MCH-"), head, tail, roll,
                                  parent=blend[lower], collection="Mechanism")
            self.twists.append((name, i / segments))
            # share of the hand/foot twist: the first piece (the lower deform bone) none, the
            # last all of it, evenly in between; the weights blend between pieces
            self.lower_segments.append((source, i / (segments - 1)))
            self.def_sources.append((name, source))
        deform.append(def_lower)
        def_end = self.make_deform(end, def_lower)
        self.def_sources.append((def_end, blend[end]))
        deform.append(def_end)
        self.deform = deform

        if mode == 'TWIST':
            # One tweak control per deform segment, child of what the segment follows
            sources = dict(self.def_sources)
            segs = [(def_upper, upper)] + [(n, upper) for n in self.upper_twists]
            segs += [(def_lower, lower)] + [(n, lower) for n, _f in self.twists]
            index = {}
            for name, ref in segs:
                i = index.get(ref, 0)
                index[ref] = i + 1
                eb = gen.edit_bones[name]
                tweak = gen.new_bone(self.name(ref, "_tweak" + (f".{i:02d}" if i else "")), eb.head, eb.tail,
                                     eb.z_axis, parent=sources[name], collection="Tweak")
                self.twist_tweaks.append(tweak)
                self.def_sources = [(d, tweak if d == name else src) for d, src in self.def_sources]

    # -- secondary controls, joint fans, pinning ------------------------------
    def _extras_edit(self, upper, lower, bend):
        gen, s = self.gen, self.settings
        up, low = gen.bones[upper], gen.bones[lower]
        joint = low.head
        up_dir, low_dir = (up.tail - up.head).normalized(), (low.tail - low.head).normalized()
        avg = (up_dir + low_dir).normalized() if (up_dir + low_dir).length > 1e-6 else low_dir
        small = (up.length + low.length) * 0.06
        def_upper, def_lower = self.deform[-3], self.deform[-2]
        # The upper piece touching the joint, and the lower segment's rotation relative
        # to it for the bendy joint handle. With upper twist bones the lower deform
        # bone's own parent is the untwisted first piece, so a helper child of the joint
        # piece follows the lower bone: its local rotation is the bend without the upper
        # arm / thigh twist.
        self.joint_upper = self.upper_twists[-1] if self.upper_twists else def_upper
        self.joint_rel = def_lower
        if self.upper_twists and self.mode == 'BENDY':
            eb = gen.edit_bones[def_lower]
            self.joint_rel = gen.new_bone(self.name(lower, "_joint_rel", "MCH-"), eb.head, eb.tail, eb.z_axis,
                                          parent=self.joint_upper, collection="Mechanism")

        self.tweaks = list(self.twist_tweaks)
        if self.mode == 'BENDY':
            # Bendy deform bones: a joint tweak sets the curve tangent at the elbow/knee,
            # mid tweaks bend and squash each segment.
            joint_parent = gen.new_bone(self.name(lower, "_tweak_parent", "MCH-"), joint,
                                        joint + avg * small, bend, parent=self.joint_upper,
                                        collection="Mechanism")
            self.joint_parent = joint_parent
            self.joint_tweak = gen.new_bone(self.name(lower, "_bend"), joint, joint + avg * small, bend,
                                            parent=joint_parent, collection="Tweak")
            # Each segment may be split into twist bones: the mid tweak bends the piece
            # holding the segment's middle, the joint handle shapes the pieces at the joint.
            upper_pieces = [def_upper] + self.upper_twists
            lower_pieces = [def_lower] + [n for n, _f in self.twists]
            mids, self.mid_bendy = [], []
            for ref, pieces in ((up, upper_pieces), (low, lower_pieces)):
                mid = ref.head.lerp(ref.tail, 0.5)
                direction = (ref.tail - ref.head).normalized()
                along = [(mid - gen.edit_bones[p].head).dot(direction) for p in pieces]
                holder = pieces[max(i for i, d in enumerate(along) if d >= -1e-6)] if any(
                    d >= -1e-6 for d in along) else pieces[0]
                mids.append(gen.new_bone(self.name(ref.name, "_tweak"), mid, mid + direction * small, bend,
                                         parent=holder, collection="Tweak"))
                self.mid_bendy.append(holder)
                for piece in pieces:
                    gen.edit_bones[piece].bbone_segments = 8
            self.mid_tweaks = mids
            up_eb, low_eb = gen.edit_bones[self.joint_upper], gen.edit_bones[def_lower]
            up_eb.bbone_handle_type_end = 'TANGENT'
            up_eb.bbone_custom_handle_end = gen.edit_bones[self.joint_tweak]
            low_eb.bbone_handle_type_start = 'TANGENT'
            low_eb.bbone_custom_handle_start = gen.edit_bones[self.joint_tweak]
            # No easing at the joint by default: a curve that has to pass through the
            # joint along the bisector bows outward and balloons a bent elbow/knee.
            # Scaling the joint tweak along Y rounds the joint on demand.
            up_eb.bbone_easeout = 0.0
            low_eb.bbone_easein = 0.0
            self.tweaks = [mids[0], self.joint_tweak, mids[1]]

        # Joint fans: helper deform bones that take part of the joint rotation.
        self.fans = []    # (fan bone, parent blend, child blend, child deform, influence)
        blend = dict(zip(self.seg, self.blend))
        joints = []
        if s.fans_root:
            parent_def = gen.edit_bones[self.deform[0]].parent
            if parent_def is not None:
                joints.append((upper, parent_def.name, def_upper, self.socket, blend[upper], up.head, up_dir))
        if s.fans:
            joints.append((lower, self.upper_twists[-1] if self.upper_twists else def_upper, def_lower,
                           blend[upper], blend[lower], joint, low_dir))
        if s.fans_end:
            end = gen.bones[self.end_ref]
            joints.append((self.end_ref, self.twists[-1][0] if self.twists else def_lower, self.deform[-1],
                           blend[lower], blend[self.end_ref], end.head, (end.tail - end.head).normalized()))
        count = s.fan_count
        for ref, parent_def, child_def, parent_blend, child_blend, head, direction in joints:
            for i in range(count):
                index = "" if count == 1 else f".{i + 1:02d}"
                eb_child = gen.edit_bones[child_def]
                name = gen.new_bone(f"DEF-{self.stem(ref)}_fan{index}{self.suffix}", head,
                                    head + direction * small * 1.5, eb_child.z_axis, parent=parent_def,
                                    deform=True, collection="Deform")
                self.fans.append((name, parent_blend, child_blend, child_def, (i + 1) / (count + 1)))

        # Pinning chain: upper bone aims at the pole, lower bone at the IK target.
        pin_parent = self.blend[0] if self.three else self.socket
        self.pin = [
            gen.new_bone(self.name(upper, "_pin", "MCH-"), up.head, up.tail, bend,
                         parent=pin_parent, collection="Mechanism"),
            None,
        ]
        self.pin[1] = gen.new_bone(self.name(lower, "_pin", "MCH-"), low.head, low.tail, bend,
                                   parent=self.pin[0], connect=True, collection="Mechanism")

    def _ik_target_chain(self, ik_ctrl, end_ref, kind):
        """IK control -> [offset] -> [pivot] ; returns the bone the rest of the chain hangs from."""
        gen, s = self.gen, self.settings
        eb = gen.edit_bones[ik_ctrl]
        head, tail, z = eb.head.copy(), eb.tail.copy(), eb.z_axis.copy()
        size = (tail - head) * 0.75
        self.ik_offset = self.pivot = None
        top = ik_ctrl
        if s.ik_offset:
            self.ik_offset = gen.new_bone(self.name(end_ref, "_ik_offset"), head, head + size, z,
                                          parent=top, collection=self.ik_collection)
            top = self.ik_offset
        if s.ik_pivot:
            short = head + (tail - head) * 0.4
            self.pivot = gen.new_bone(self.name(end_ref, "_ik_pivot"), head, short, z, parent=top,
                                      collection=self.ik_collection)
            self.pivot_a = gen.new_bone(self.name(end_ref, "_pivot_a", "MCH-"), head, short, z, parent=top,
                                        collection="Mechanism")
            self.pivot_b = gen.new_bone(self.name(end_ref, "_pivot_b", "MCH-"), head, short, z,
                                        parent=self.pivot_a, collection="Mechanism")
            top = self.pivot_b
        return top

    # -- pose helpers ---------------------------------------------------------
    def _pivot_pose(self):
        """T(p) R T(-p): the pivot control rotates the chain around its own position."""
        if not self.pivot:
            return
        rig = self.gen.rig
        pbs = rig.pose.bones
        local = dict(owner_space='LOCAL', target_space='LOCAL')
        utils.add_constraint(pbs[self.pivot_a], 'COPY_LOCATION', rig, self.pivot, **local)
        utils.add_constraint(pbs[self.pivot_a], 'COPY_ROTATION', rig, self.pivot, **local)
        utils.add_constraint(pbs[self.pivot_b], 'COPY_LOCATION', rig, self.pivot, invert_x=True, invert_y=True,
                             invert_z=True, **local)
        self.shape(self.pivot, 'SPHERE', self.h * 0.012, color="TWEAK")
        pbs[self.pivot].lock_scale = (True, True, True)

    def _extras_pose(self, prop_bone, ik_target):
        gen, rig = self.gen, self.gen.rig
        pbs = rig.pose.bones
        utils.add_property(pbs[prop_bone], "pin", 0.0, description="Pin the elbow/knee to the pole control")

        # Pinning
        utils.add_constraint(pbs[self.pin[0]], 'STRETCH_TO', rig, self.pole,
                             rest_length=rig.data.bones[self.pin[0]].length, volume='NO_VOLUME')
        utils.add_constraint(pbs[self.pin[1]], 'STRETCH_TO', rig, ik_target,
                             rest_length=rig.data.bones[self.pin[1]].length, volume='NO_VOLUME')
        for blend, pin in zip(self.blend[-3:-1], self.pin):
            con = self.copy_transforms(blend, pin)
            utils.add_driver_vars(con, "influence", rig, [("ik", prop_bone, "ik_fk"), ("pin", prop_bone, "pin")],
                                  "ik * pin")

        # Pole line
        line = pbs[self.pole_line]
        utils.add_constraint(line, 'STRETCH_TO', rig, self.pole, rest_length=rig.data.bones[self.pole_line].length,
                             volume='NO_VOLUME')
        widgets.apply_widget(line, 'LINE', 1.0, wire_width=1.0)
        widgets.set_color(line, "TWEAK")
        rig.data.bones[self.pole_line].hide_select = True

        if self.joint_rel != self.deform[-2]:
            self.copy_transforms(self.joint_rel, self.deform[-2])
        if self.mode == 'BENDY':
            joint_parent = pbs[self.joint_parent]
            def_upper, def_lower = self.deform[-3], self.deform[-2]
            utils.add_constraint(joint_parent, 'COPY_ROTATION', rig, self.joint_rel, mix_mode='REPLACE',
                                 owner_space='LOCAL', target_space='LOCAL_OWNER_ORIENT', influence=0.5)
            joint = self.shape(self.joint_tweak, 'CIRCLE', self.h * 0.02, y=0.0, color="TWEAK",
                               fit=[self.joint_upper, def_lower], margin=1.1, wire=1.5)
            joint.lock_location = (True, True, True)
            # Joint roundness: Y scale 1 = sharp joint, 2 = fully rounded curve
            utils.add_transform_driver(pbs[self.joint_upper], "bbone_easeout", rig, self.joint_tweak, "SCALE_Y",
                                       "max(0.0, var - 1.0)")
            utils.add_transform_driver(pbs[def_lower], "bbone_easein", rig, self.joint_tweak, "SCALE_Y",
                                       "max(0.0, var - 1.0)")
            for mid, deform in zip(self.mid_tweaks, self.mid_bendy):
                pb = self.shape(mid, 'CIRCLE', self.h * 0.02, color="TWEAK", fit=[deform], margin=1.1,
                                wire=1.5)
                pb.lock_rotation = (True, True, True)
                pb.lock_rotation_w = True
                dpb = pbs[deform]
                length = rig.data.bones[deform].length
                for channel, axis in (("x", "LOC_X"), ("z", "LOC_Z")):
                    for side in ("in", "out"):
                        attr = f"bbone_curve{side}{channel}"
                        if hasattr(dpb, attr):
                            utils.add_transform_driver(dpb, attr, rig, mid, axis,
                                                       f"var * {1.5 / max(length, 1e-6):.4f} * {length:.4f}")
                if hasattr(dpb, "bbone_scalein") and len(dpb.bbone_scalein) == 3:
                    for index, axis in ((0, "SCALE_X"), (2, "SCALE_Z")):
                        utils.add_transform_driver(dpb, "bbone_scalein", rig, mid, axis, index=index)
                        utils.add_transform_driver(dpb, "bbone_scaleout", rig, mid, axis, index=index)
        elif self.mode == 'TWIST':
            for tweak in self.twist_tweaks:
                self.shape(tweak, 'CIRCLE', self.h * 0.018, y=0.5, color="TWEAK",
                           fit=[d for d, src in self.def_sources if src == tweak], margin=1.08, wire=1.5)

        s = self.settings
        for fan, parent_blend, child_blend, child_def, influence in self.fans:
            pb = pbs[fan]
            utils.add_constraint(pb, 'COPY_ROTATION', rig, child_def, mix_mode='REPLACE',
                                 owner_space='WORLD', target_space='WORLD', influence=influence)
            if s.fan_bulge > 0:
                rest = utils.rotation_angle_rest(rig, parent_blend, child_blend)
                length = rig.data.bones[fan].length
                if s.fan_bulge_mode == 'LOCATION':
                    expr = f"{s.fan_bulge * length:.5f} * max(0.0, angle - {rest:.5f})"
                    path, index = "location", 1
                else:
                    expr = f"1.0 + {s.fan_bulge:.5f} * max(0.0, angle - {rest:.5f})"
                    path, index = "scale", 1
                utils.add_driver_mixed(pb, path, rig, [("angle", 'rotdiff', parent_blend, child_blend)], expr,
                                       index=index)

    def _fk_lock_pose(self):
        if not self.fk_space:
            return
        rig = self.gen.rig
        ctrl = rig.pose.bones[self.fk[0]]
        utils.add_property(ctrl, "fk_lock", 1.0,
                           description="1 = the FK chain follows the body, 0 = it keeps its world rotation")
        con = utils.add_constraint(rig.pose.bones[self.fk_space], 'COPY_ROTATION', rig, self.fk_world,
                                   owner_space='WORLD', target_space='WORLD')
        utils.add_driver(con, "influence", rig, self.fk[0], "fk_lock", "1.0 - var")

    def _pole_space(self, ik_ctrl, label, smart=None):
        gen = self.gen
        options = [("Root", "root"), ("Torso", gen.spaces.get("Torso")), (label, ik_ctrl)]
        if smart:
            options.insert(0, ("Smart", smart))
        gen.post_edit.append(lambda: gen.add_space_switch(self.pole, options))

    def _blend(self, prop_bone):
        """Blend bones follow FK, mixed toward IK by ik_fk (and 3 bones IK by three_ik)."""
        rig = self.gen.rig
        for i, (blend, fk, ik) in enumerate(zip(self.blend, self.fk, self.ik)):
            self.copy_transforms(blend, fk)
            con = self.copy_transforms(blend, ik)
            utils.add_driver(con, "influence", rig, prop_bone, "ik_fk")
            if i < len(self.ik3):
                con = self.copy_transforms(blend, self.ik3[i])
                utils.add_driver_vars(con, "influence", rig,
                                      [("ik", prop_bone, "ik_fk"), ("three", prop_bone, "three_ik")],
                                      "ik * three")

    def _deform_pose(self):
        rig = self.gen.rig
        pbs = rig.pose.bones
        blend = dict(zip(self.seg, self.blend))
        s = self.settings
        upper, lower, end = self.seg[-3], self.seg[-2], self.seg[-1]
        if self.swing:
            utils.add_constraint(pbs[self.swing], 'DAMPED_TRACK', rig, blend[lower], track_axis='TRACK_Y')
        for source, fraction, keep in self.upper_segments:
            utils.add_constraint(pbs[source], 'COPY_TRANSFORMS', rig, blend[upper], head_tail=fraction)
            utils.add_constraint(pbs[source], 'COPY_ROTATION', rig, self.swing, owner_space='WORLD',
                                 target_space='WORLD', influence=(1.0 - keep) * s.twist_factor)
        for source, factor in self.lower_segments:
            con = utils.add_constraint(
                pbs[source], 'TRANSFORM', rig, blend[end],
                owner_space='LOCAL', target_space='LOCAL',
                map_from='ROTATION', map_to='ROTATION', from_rotation_mode='SWING_TWIST_Y',
                map_to_x_from='X', map_to_y_from='Y', map_to_z_from='Z', mix_mode_rot='ADD')
            con.from_min_y_rot, con.from_max_y_rot = -math.pi, math.pi
            amount = math.pi * factor * s.twist_factor
            con.to_min_y_rot, con.to_max_y_rot = -amount, amount
        for deform, source in self.def_sources:
            self.copy_transforms(deform, source)

    def _ik_setup(self, ik_target, prop_bone):
        rig, s = self.gen.rig, self.settings
        pbs = rig.pose.bones
        utils.add_property(pbs[prop_bone], "stretch_length", 1.0, 0.1, 3.0, "Length of the IK chain")
        utils.add_constraint(pbs[self.ik[-2]], 'IK', rig, ik_target, pole_target=rig, pole_subtarget=self.pole,
                             pole_angle=self.pole_angle, chain_count=2, use_tail=True, use_stretch=True)
        stretchy = list(self.ik[-3:-1])
        if self.three:
            utils.add_property(pbs[prop_bone], "three_ik", 0.0,
                               description="0 = 2 bones IK with a rotation control at the root, 1 = 3 bones IK")
            utils.add_constraint(pbs[self.ik3[-1]], 'IK', rig, ik_target, pole_target=rig,
                                 pole_subtarget=self.pole, pole_angle=self.pole_angle3, chain_count=3,
                                 use_tail=True, use_stretch=True)
            stretchy += self.ik3
        length = sum(rig.data.bones[n].length for n in self.ik[-3:-1])
        for name in stretchy:
            ik_pb = pbs[name]
            ik_pb.ik_stretch = 0.0
            utils.add_driver(ik_pb, "ik_stretch", rig, prop_bone, "stretch", "var * 0.1")
            ik_pb.lock_ik_y = False
            if s.soft_ik and not self.three:
                # Scale the chain so the IK solves as if the target were softly pulled in.
                soft = s.soft_ik_amount * length
                expr = (f"sl * max(1.0, (d / sc) / ((d / sc) if (d / sc) < {length:.5f} * sl - {soft:.5f} "
                        f"else {length:.5f} * sl - {soft:.5f} * exp(({length:.5f} * sl - {soft:.5f} - d / sc) "
                        f"/ {soft:.5f})))")
                utils.add_driver_mixed(ik_pb, "scale", rig, [
                    ("sl", 'prop', prop_bone, "stretch_length"), ("d", 'distance', self.socket, ik_target),
                    ("sc", 'obj', 'scale[2]')], expr, index=1)
            else:
                utils.add_driver(ik_pb, "scale", rig, prop_bone, "stretch_length", index=1)

    def _auto_hide(self, prop_bone, fk_bones, ik_bones):
        if not self.gen.auto_hide:
            return
        arm = self.gen.rig.data
        for name in fk_bones:
            utils.add_driver(arm, f'bones["{name}"].hide', self.gen.rig, prop_bone, "ik_fk", "var > 0.5")
        for name in ik_bones:
            if name:
                utils.add_driver(arm, f'bones["{name}"].hide', self.gen.rig, prop_bone, "ik_fk", "var < 0.5")

    def _fk_controls(self):
        r = self.h * (0.032 if self.limb_type == "arm" else 0.045)
        twist = [n for n, _f in self.twists]
        fits = [[d] for d in self.deform[:-3]]
        fits += [[self.deform[-3]] + self.upper_twists, [self.deform[-2]] + twist, [self.deform[-1]]]
        for i, name in enumerate(self.fk):
            is_end = i == len(self.fk) - 1
            pb = self.shape(name, 'CIRCLE', r * (0.85 if is_end else 1.0), y=0.35 if is_end else 0.5,
                            fit=fits[i])
            if not is_end:
                pb.lock_location = (True, True, True)
        if self.ik_rot:
            pb = self.shape(self.ik_rot, 'CIRCLE', r * 1.1, y=0.5, fit=fits[0])
            pb.lock_location = (True, True, True)

    def _pole_shape(self):
        self.shape(self.pole, 'POLE', self.h * 0.015)

    def _under(self, ref_name, parents):
        """True when the bone's parent, or its metacarpal's parent, is one of 'parents'."""
        parent = self.gen.bones[ref_name].parent
        if parent in parents:
            return True
        return parent is not None and self.stem(parent).endswith("_meta") and self.gen.bones[parent].parent in parents

    def _digit_modules(self, parents):
        from .chain import FingerModule
        return [m for m in self.gen.modules if isinstance(m, FingerModule) and self._under(m.root, parents)]

    def _metacarpal_pose(self, parents):
        """Pinky auto: rotating the pinky metacarpal/metatarsal cups the palm (ring and middle follow)."""
        if not self.settings.pinky_auto:
            return
        from .chain import SimpleModule
        rig = self.gen.rig
        metas = {}
        for m in self.gen.modules:
            if isinstance(m, SimpleModule) and self.gen.bones[m.root].parent in parents:
                stem = self.stem(m.root)
                for kind in ("pinky", "ring", "middle"):
                    if stem.endswith(f"{kind}_meta"):
                        metas[kind] = m.ctrl
        pinky = metas.get("pinky")
        if not pinky:
            return
        others = [k for k in ("ring", "middle") if k in metas]
        for i, kind in enumerate(others):
            influence = (len(others) - i) / (len(others) + 1)
            con = utils.add_constraint(rig.pose.bones[metas[kind]], 'COPY_ROTATION', rig, pinky, mix_mode='ADD',
                                       owner_space='LOCAL', target_space='LOCAL', influence=influence)
            con.name = "Pinky Auto"

    def _spread_pose(self, parents):
        """Rotating the pinky's first bone sideways spreads the other digits."""
        if not self.settings.auto_spread:
            return
        rig = self.gen.rig
        digits = {}
        for m in self._digit_modules(parents):
            stem = self.stem(m.root)
            for kind in ("pinky", "ring", "middle", "index"):
                if stem.startswith(kind):
                    digits[kind] = m.ctrls[0]
        pinky = digits.get("pinky")
        if not pinky:
            return
        for kind, influence in (("ring", 0.66), ("middle", 0.33)):
            if kind in digits:
                utils.add_constraint(rig.pose.bones[digits[kind]], 'COPY_ROTATION', rig, pinky,
                                     use_x=False, use_y=False, use_z=True, mix_mode='ADD',
                                     owner_space='LOCAL', target_space='LOCAL', influence=influence)

    def _limb_meta(self, kind, prop_bone, ik_ctrl, ik_target, extra):
        meta = {
            "type": kind, "side": self.side, "label": f"{kind.title()} {self.side}".strip(),
            "prop_bone": prop_bone, "fk": self.fk, "ik": self.ik, "blend": self.blend,
            "ik_ctrl": ik_ctrl, "ik_target": ik_target, "pole": self.pole,
            "deform": self.deform, "ref": self.seg, "tweaks": self.tweaks,
            "ik_rot": self.ik_rot, "three": self.three, "fk_lock": self.fk[0] if self.fk_space else None,
            "ik_offset": self.ik_offset, "ik_pivot": self.pivot, "pole_line": self.pole_line,
        }
        meta.update(extra)
        return meta


class ArmModule(LimbModule):
    type_id = 'ARM'
    label = "Arm"
    description = "Shoulder (optional), upper arm, forearm and hand with IK/FK"
    chain_length = 4
    min_length = 3
    collection = "Arm FK"

    def edit(self):
        gen = self.gen
        parent = self.parent_deform
        chain = self.chain if len(self.chain) == 4 else [None] + self.chain[:3]
        shoulder, upper, lower, hand = chain
        self.shoulder = None
        socket_parent = parent
        if shoulder:
            self.shoulder = self.copy_ref(shoulder, shoulder, parent, collection="Arm FK")
            socket_parent = self.make_deform(shoulder, parent)
            self.shoulder_def = socket_parent
        self._build([upper], lower, hand, socket_parent)
        self.hand_ik = self.copy_ref(hand, self.name(hand, "_ik"), "root", collection=self.ik_collection)
        self.prop_bone = self.hand_ik
        top = self._ik_target_chain(self.hand_ik, hand, "hand")
        self.ik_target = self.copy_ref(hand, self.name(hand, "_target", "MCH-"), top, collection="Mechanism")
        self.wings_fold = None
        if self.settings.wings_fold:
            ref = gen.bones[hand]
            # Parented outside the arm: the fold action moves the arm, the control must not follow it
            self.wings_fold = gen.new_bone(self.name(hand, "_wings_fold"), ref.tail,
                                           ref.tail + (ref.tail - ref.head) * 0.5, ref.z_axis,
                                           parent=parent or "root", collection="Arm FK")
        gen.post_edit.append(lambda: gen.add_space_switch(self.hand_ik, [
            (label, gen.spaces.get(label)) for label in ("Root", "Torso", "Chest", "Head")]))
        self._pole_space(self.hand_ik, "Hand")
        self.meta = self._limb_meta("arm", self.hand_ik, self.hand_ik, self.ik_target, {
            "shoulder": self.shoulder, "wings_fold": self.wings_fold,
            "ik_reset": [n for n in (self.ik_offset, self.pivot) if n],
        })

    def pose(self):
        rig = self.gen.rig
        pbs = rig.pose.bones
        if self.shoulder:
            self.shape(self.shoulder, 'SHOULDER', self.h * 0.03, y=0.75,
                       fit=[self.shoulder_def], margin=1.4)
            self.copy_transforms(self.shoulder_def, self.shoulder)
        prop = pbs[self.hand_ik]
        utils.add_property(prop, "ik_fk", 1.0 if self.settings.start_ik else 0.0,
                           description="0 = FK, 1 = IK")
        utils.add_property(prop, "stretch", 0.0, description="IK auto-stretch amount")
        self.shape(self.hand_ik, 'BOX', self.h * 0.03, fit=[self.deform[-1]], margin=1.45)
        if self.ik_offset:
            self.shape(self.ik_offset, 'BOX', self.h * 0.022, fit=[self.deform[-1]], margin=1.2, color="TWEAK")
        self._pole_shape()
        self._fk_controls()
        self._pivot_pose()
        self._ik_setup(self.ik_target, self.hand_ik)
        mch_hand = pbs[self.ik[-1]]
        utils.add_constraint(mch_hand, 'COPY_ROTATION', rig, self.ik_target)
        utils.add_constraint(mch_hand, 'COPY_SCALE', rig, self.ik_target)
        self._blend(self.hand_ik)
        self._deform_pose()
        self._extras_pose(self.hand_ik, self.ik_target)
        self._fk_lock_pose()
        self._spread_pose({self.seg[-1]})
        self._metacarpal_pose({self.seg[-1]})
        if self.wings_fold:
            pb = self.shape(self.wings_fold, 'CUBE', self.h * 0.012)
            pb.lock_location = (True, True, True)
            utils.add_constraint(pb, 'LIMIT_SCALE', use_min_y=True, min_y=0.0, use_max_y=True, max_y=1.0,
                                 owner_space='LOCAL')
        hide_ik = [self.hand_ik, self.pole, self.ik_offset, self.pivot, self.pole_line]
        self._auto_hide(self.hand_ik, self.fk, hide_ik)


class LegModule(LimbModule):
    type_id = 'LEG'
    label = "Leg"
    description = "Thigh, shin, foot and toe (optional) with IK/FK and foot roll"
    chain_length = 4
    min_length = 3
    collection = "Leg FK"
    fallback_bend = Vector((0, -1, 0))
    ik_collection = "Leg IK"
    fk_collection = "Leg FK"
    limb_type = "leg"

    def __init__(self, gen, root):
        super().__init__(gen, root)
        if self.settings.three_bones:
            self.chain = gen.chain_from(root)[:5]
            self.min_length = 4

    def edit(self):
        gen, s = self.gen, self.settings
        parent = self.parent_deform
        n = 3 if s.three_bones else 2
        uppers = self.chain[:n - 1]
        shin, foot = self.chain[n - 1], self.chain[n]
        toe = self.chain[n + 1] if len(self.chain) > n + 1 else None
        self.toe_ref = toe
        self._build(uppers, shin, foot, parent)

        foot_ref = gen.bones[foot]
        ankle, ball = foot_ref.head, foot_ref.tail
        forward = utils.flatten(ball - ankle)
        foot_len = max((ball - ankle).length, 0.01)
        self.foot_len = foot_len
        up = Vector((0, 0, 1))
        self.foot_ik = gen.new_bone(self.name(foot, "_ik"), ankle, ankle + forward * foot_len, up,
                                    parent="root", collection=self.ik_collection)
        self.prop_bone = self.foot_ik
        gen.post_edit.append(lambda: gen.add_space_switch(self.foot_ik, [
            (label, gen.spaces.get(label)) for label in ("Root", "Torso")]))
        self.smart = None
        if s.smart_pole:
            self.smart = gen.new_bone(self.name(foot, "_smart_pole", "MCH-"), ankle, ankle + forward * foot_len,
                                      up, parent="root", collection="Mechanism")
        self._pole_space(self.foot_ik, "Foot", self.smart)
        top = self._ik_target_chain(self.foot_ik, foot, "foot")

        # Reverse foot: heel -> outer bank -> inner bank -> toe tip -> [toes pivot] -> ball -> foot
        toe_tail = gen.bones[toe].tail if toe else ball + forward * foot_len * 0.3
        ground = min(ball.z, toe_tail.z, ankle.z)
        flat = lambda v: Vector((v.x, v.y, ground))
        side = forward.cross(up)
        # The outer edge points away from the body centre (centre feet: any side).
        outer = -side if side.x * ankle.x < 0 else side
        width = foot_len * 0.3
        heel = flat(ankle) - forward * foot_len * 0.15
        tip = flat(toe_tail)
        self.forward, self.outer, self.side_axis = forward, outer, side
        piv = forward * foot_len * 0.2

        def pivot(name, pos, parent):
            return gen.new_bone(self.name(foot, name, "MCH-"), pos, pos + piv, up, parent=parent,
                                collection="Mechanism")
        self.heel = pivot("_heel", heel, top)
        self.bank_out = pivot("_bank_out", flat(ball) + outer * width, self.heel)
        self.bank_in = pivot("_bank_in", flat(ball) - outer * width, self.bank_out)
        self.tip = pivot("_tip", tip, self.bank_in)
        under_tip = self.tip
        self.toes_pivot = None
        if s.toes_pivot:
            self.toes_pivot = gen.new_bone(self.name(foot, "_toes_pivot"), ball, ball + forward * foot_len * 0.5,
                                           up, parent=self.tip, collection=self.ik_collection)
            under_tip = self.toes_pivot
        back = ball - forward * foot_len * 0.5
        self.ball = gen.new_bone(self.name(foot, "_ball"), ball, back, up, parent=under_tip,
                                 collection=self.ik_collection)
        self.ball_roll = gen.new_bone(self.name(foot, "_ball_roll", "MCH-"), ball, back, up, parent=self.ball,
                                      collection="Mechanism")
        self.foot_ref = self.copy_ref(foot, self.name(foot, "_target", "MCH-"), self.ball_roll,
                                      collection="Mechanism")
        # Roll cursor behind the heel: move it forward/back to roll, sideways to bank.
        cursor = heel - forward * foot_len * 0.35
        self.foot_roll = gen.new_bone(self.name(foot, "_roll"), cursor, cursor + forward * foot_len * 0.25, up,
                                      parent=top, collection=self.ik_collection)

        self.toe_fk = self.toe_ik = self.toe_roll = None
        if toe:
            self.toe_fk = self.copy_ref(toe, self.name(toe, "_fk"), self.fk[-1],
                                        connect=gen.bones[toe].connected, collection=self.fk_collection)
            self.toe_roll = gen.new_bone(self.name(toe, "_roll", "MCH-"), ball, ball + piv, up, parent=under_tip,
                                         collection="Mechanism")
            self.toe_ik = self.copy_ref(toe, self.name(toe, "_ik"), self.toe_roll, collection=self.ik_collection)
            self.toe_def = self.make_deform(toe, self.deform[-1], connect=gen.bones[toe].connected)

        # Footprint shape for the IK foot control, drawn on the ground.
        height = ankle.z - ground
        front = (toe_tail - ankle).dot(forward)
        width = foot_len * 0.3
        self.foot_shape = [(-width, -foot_len * 0.35, -height), (width, -foot_len * 0.35, -height),
                           (width, front, -height), (-width, front, -height)]
        reset = [n for n in (self.ik_offset, self.pivot, self.ball, self.toes_pivot, self.foot_roll) if n]
        self.meta = self._limb_meta("leg", self.foot_ik, self.foot_ik, self.foot_ref, {
            "foot_roll": self.foot_roll, "ball": self.ball, "toes_pivot": self.toes_pivot,
            "toe_fk": self.toe_fk, "toe_ik": self.toe_ik, "ik_reset": reset,
        })

    def _foot_roll_pose(self):
        """Drivers turning the roll cursor's location into heel, tip, ball and bank rotations."""
        rig, s = self.gen.rig, self.settings
        pbs = rig.pose.bones
        prop = pbs[self.foot_ik]
        utils.add_property(prop, "roll_speed", 1.0, 0.0, 5.0, "How fast the foot rolls when moving the roll cursor")
        k = 1.5 / self.foot_len
        cur = self.foot_roll
        u = [("u", 'transform', cur, 'LOC_Y', 'TRANSFORM_SPACE'), ("speed", 'prop', self.foot_ik, "roll_speed")]
        a = f"(u * {k:.5f} * speed)"
        for name in (self.heel, self.bank_out, self.bank_in, self.tip, self.ball_roll):
            pbs[name].rotation_mode = 'XYZ'
        # Rotations about the pivots' X axis (forward x up): positive raises the toes.
        utils.add_driver_mixed(pbs[self.heel], "rotation_euler", rig, u, f"max(0.0, -{a})", index=0)
        if s.roll_break:
            utils.add_property(prop, "roll_break", 30.0, 0.0, 90.0,
                               "Roll angle (degrees) where the foot starts rotating from the toe tip")
            utils.add_property(prop, "roll_overshoot", 0.0, 0.0, 2.0,
                               "How far the toes bend back when rolling onto the tip")
            utils.add_property(prop, "roll_unfold", 1.0, 0.01, 10.0,
                               "Higher = the ball rotation fades out slower after the break point")
            brk = u + [("b", 'prop', self.foot_ik, "roll_break"), ("lag", 'prop', self.foot_ik, "roll_unfold")]
            br = "(b * 0.0174533)"
            utils.add_driver_mixed(pbs[self.tip], "rotation_euler", rig, brk, f"-max(0.0, {a} - {br})", index=0)
            utils.add_driver_mixed(pbs[self.ball_roll], "rotation_euler", rig, brk,
                                   f"max(0.0, min({a}, {br}) - max(0.0, {a} - {br}) / max(lag, 0.01))", index=0)
            if self.toe_roll:
                pbs[self.toe_roll].rotation_mode = 'XYZ'
                over = brk + [("o", 'prop', self.foot_ik, "roll_overshoot")]
                utils.add_driver_mixed(pbs[self.toe_roll], "rotation_euler", rig, over,
                                       f"o * max(0.0, {a} - {br})", index=0)
        else:
            utils.add_driver_mixed(pbs[self.tip], "rotation_euler", rig, u, f"-max(0.0, {a})", index=0)
        # Bank: rotation about the pivots' Y axis (forward). The cursor's X axis is side_axis.
        v = [("v", 'transform', cur, 'LOC_X', 'TRANSFORM_SPACE'), ("speed", 'prop', self.foot_ik, "roll_speed")]
        toward_outer = 1.0 if self.side_axis.dot(self.outer) > 0 else -1.0
        b = f"(v * {toward_outer * k:.5f} * speed)"
        lift_out = 1.0 if self.forward.cross(-self.outer).z > 0 else -1.0
        lift_in = 1.0 if self.forward.cross(self.outer).z > 0 else -1.0
        utils.add_driver_mixed(pbs[self.bank_out], "rotation_euler", rig, v, f"{lift_out} * max(0.0, {b})", index=1)
        utils.add_driver_mixed(pbs[self.bank_in], "rotation_euler", rig, v, f"{lift_in} * max(0.0, -{b})", index=1)

    def pose(self):
        rig = self.gen.rig
        pbs = rig.pose.bones
        prop = pbs[self.foot_ik]
        utils.add_property(prop, "ik_fk", 1.0 if self.settings.start_ik else 0.0,
                           description="0 = FK, 1 = IK")
        utils.add_property(prop, "stretch", 0.0, description="IK auto-stretch amount")

        foot = self.shape(self.foot_ik, 'BOX', self.h * 0.03, wire=2.5)
        foot.custom_shape = widgets.foot_widget(rig.name, self.foot_ik, self.foot_shape)
        foot.use_custom_shape_bone_size = False
        foot.custom_shape_scale_xyz = (1.0, 1.0, 1.0)
        if self.ik_offset:
            pb = self.shape(self.ik_offset, 'BOX', self.h * 0.02, color="TWEAK")
            pb.custom_shape = foot.custom_shape
            pb.use_custom_shape_bone_size = False
            pb.custom_shape_scale_xyz = (0.8, 0.8, 0.8)
        roll = self.shape(self.foot_roll, 'ARROWS', self.h * 0.02)
        roll.lock_rotation = (True, True, True)
        roll.lock_rotation_w = True
        roll.lock_scale = (True, True, True)
        roll.lock_location = (False, False, True)
        ball = self.shape(self.ball, 'ROLL', self.h * 0.025, y=1.0)
        ball.lock_location = (True, True, True)
        if self.toes_pivot:
            pb = self.shape(self.toes_pivot, 'CIRCLE', self.h * 0.02, y=0.0, color="TWEAK")
            pb.lock_location = (True, True, True)
        if self.smart:
            utils.add_constraint(pbs[self.smart], 'COPY_LOCATION', rig, self.foot_ik)
            utils.add_constraint(pbs[self.smart], 'COPY_ROTATION', rig, self.foot_ik, use_x=False, use_y=False)
        self._pole_shape()
        self._fk_controls()
        self._pivot_pose()
        self._foot_roll_pose()
        self._ik_setup(self.foot_ref, self.foot_ik)
        mch_foot = pbs[self.ik[-1]]
        utils.add_constraint(mch_foot, 'COPY_ROTATION', rig, self.foot_ref)
        self._blend(self.foot_ik)
        self._deform_pose()
        self._extras_pose(self.foot_ik, self.foot_ref)
        self._fk_lock_pose()
        self._spread_pose({self.seg[-1], self.toe_ref})
        self._metacarpal_pose({self.seg[-1], self.toe_ref})

        hide_fk = list(self.fk)
        hide_ik = [self.foot_ik, self.foot_roll, self.pole, self.ball, self.ik_offset, self.pivot,
                   self.toes_pivot, self.ik_rot, self.pole_line]
        if self.toe_fk:
            pb = self.shape(self.toe_fk, 'CIRCLE', self.h * 0.025, y=0.5, fit=[self.toe_def])
            pb.lock_location = (True, True, True)
            pb = self.shape(self.toe_ik, 'CIRCLE', self.h * 0.025, y=0.5, fit=[self.toe_def])
            pb.lock_location = (True, True, True)
            self.copy_transforms(self.toe_def, self.toe_fk)
            con = self.copy_transforms(self.toe_def, self.toe_ik)
            utils.add_driver(con, "influence", rig, self.foot_ik, "ik_fk")
            hide_fk.append(self.toe_fk)
            hide_ik.append(self.toe_ik)
        self._auto_hide(self.foot_ik, hide_fk, hide_ik)
