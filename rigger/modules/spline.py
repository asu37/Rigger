# SPDX-License-Identifier: GPL-3.0-or-later
"""Spline IK chains (ropes, tentacles, tails, IK spines) and skirts."""

import math

import bpy
from mathutils import Matrix, Vector

from .. import utils
from .base import RigModule


class SplineModule(RigModule):
    """A chain that follows a curve shaped by a few controls."""
    type_id = 'SPLINE'
    label = "Spline IK"
    description = "Chain following a curve driven by a few controls (ropes, tentacles, IK tails)"
    min_length = 2

    @property
    def collection(self):
        return self.settings.collection or "Extra"

    def validate(self):
        if self.settings.spline_tail and len(self.chain) < 3:
            return f"'{self.root}' (Spline IK with a tail bone) needs at least 3 bones"
        return super().validate()

    def edit(self):
        gen, st = self.gen, self.settings
        parent = self.parent_deform
        self.tail_ref = None
        if st.spline_tail:
            self.chain, self.tail_ref = self.chain[:-1], self.chain[-1]
        bones = [gen.bones[n] for n in self.chain]
        points = [bones[0].head] + [b.tail for b in bones]
        lengths = [0.0]
        for a, b in zip(points, points[1:]):
            lengths.append(lengths[-1] + (b - a).length)
        total = lengths[-1]
        count = max(2, self.settings.spline_controls)

        def point_at(dist):
            for i in range(len(points) - 1):
                if lengths[i + 1] >= dist - 1e-9:
                    seg = lengths[i + 1] - lengths[i]
                    t = (dist - lengths[i]) / seg if seg > 1e-9 else 0.0
                    return points[i].lerp(points[i + 1], t), (points[i + 1] - points[i]).normalized()
            return points[-1], (points[-1] - points[-2]).normalized()

        size = total / (count - 1) * 0.3
        stem = utils.split_side(self.root)[0].split(".")[0]
        self.master = gen.new_bone(f"{stem}_spline_master{self.suffix}", points[0],
                                   points[0] + (points[1] - points[0]).normalized() * size * 1.5,
                                   bones[0].z_axis, parent=parent, collection=self.collection)
        self.ctrls, self.ctrl_points = [], []
        for i in range(count):
            pos, direction = point_at(total * i / (count - 1))
            self.ctrl_points.append(pos)
            self.ctrls.append(gen.new_bone(f"{stem}_spline.{i + 1:02d}{self.suffix}", pos, pos + direction * size,
                                           bones[0].z_axis, parent=self.master, collection=self.collection))
        # Mechanism chain solved by Spline IK, then deform bones copy it.
        self.mch = []
        mch_parent = self.master
        def_parent = parent
        for i, ref_name in enumerate(self.chain):
            name = self.copy_ref(ref_name, "MCH-" + ref_name, mch_parent, connect=i > 0,
                                 collection="Mechanism")
            self.mch.append(name)
            mch_parent = name
            def_parent = self.make_deform(ref_name, def_parent, connect=i > 0 and gen.bones[ref_name].connected)
        # Optional FK chain with an IK/FK switch
        self.fk = []
        if st.spline_fk:
            fk_parent = self.master
            for i, ref_name in enumerate(self.chain):
                fk_parent = self.copy_ref(ref_name, self.name(ref_name, "_fk"), fk_parent, connect=i > 0,
                                          collection=self.collection)
                self.fk.append(fk_parent)
        # Tail bone: follows the tip of the curve without its stretch and twist
        self.tail_mch = None
        if self.tail_ref:
            self.tail_mch = self.copy_ref(self.tail_ref, f"MCH-{self.tail_ref}_tail", self.ctrls[-1],
                                          collection="Mechanism")
            self.make_deform(self.tail_ref, def_parent, connect=gen.bones[self.tail_ref].connected)
        self.meta = {"type": "spline", "controls": self.ctrls, "master": self.master, "fk": self.fk,
                     "mch": self.mch, "prop_bone": self.master}

    def _curve(self):
        """Create (or rebuild) the curve object the chain follows, hooked to the controls."""
        gen, rig = self.gen, self.gen.rig
        name = f"{rig.name}_{self.root}_spline"
        obj = bpy.data.objects.get(name)
        if obj is None or obj.type != 'CURVE':
            curve = bpy.data.curves.new(name, 'CURVE')
            obj = bpy.data.objects.new(name, curve)
            for coll in rig.users_collection:
                coll.objects.link(obj)
                break
        curve = obj.data
        curve.splines.clear()
        curve.dimensions = '3D'
        spline = curve.splines.new('NURBS')
        spline.points.add(len(self.ctrl_points) - 1)
        for point, co in zip(spline.points, self.ctrl_points):
            point.co = (co.x, co.y, co.z, 1.0)
        spline.use_endpoint_u = True
        spline.order_u = min(self.settings.spline_smooth, len(self.ctrl_points))
        obj.parent = rig
        obj.matrix_parent_inverse = Matrix()
        obj.matrix_basis = Matrix()
        obj.hide_render = True
        obj.display_type = 'WIRE'
        obj.hide_select = True
        for mod in list(obj.modifiers):
            obj.modifiers.remove(mod)
        for i, ctrl in enumerate(self.ctrls):
            mod = obj.modifiers.new(f"hook_{ctrl}", 'HOOK')
            mod.object = rig
            mod.subtarget = ctrl
            mod.vertex_indices_set([i])
            # Hook matrices are relative to the curve object, which sits at the rig origin.
            mod.matrix_inverse = rig.data.bones[ctrl].matrix_local.inverted()
            mod.center = self.ctrl_points[i]
        return obj

    def pose(self):
        rig, st = self.gen.rig, self.settings
        pbs = rig.pose.bones
        curve = self._curve()
        pb = rig.pose.bones[self.mch[-1]]
        con = pb.constraints.new('SPLINE_IK')
        con.target = curve
        con.chain_count = len(self.mch)
        con.use_curve_radius = False
        con.y_scale_mode = 'FIT_CURVE' if self.settings.spline_stretch else 'BONE_ORIGINAL'
        con.xz_scale_mode = 'NONE'
        length = sum(self.gen.bones[n].length for n in self.chain)
        master = self.shape(self.master, 'CIRCLE', length * 0.12)
        if self.fk:
            utils.add_property(master, "ik_fk", 1.0,
                               description="0 = FK chain, 1 = spline IK")
        count = len(self.chain)
        for i, (ref_name, mch) in enumerate(zip(self.chain, self.mch)):
            deform = self.gen.deform_of[ref_name]
            if self.fk:
                self.copy_transforms(deform, self.fk[i])
                con = self.copy_transforms(deform, mch)
                utils.add_driver(con, "influence", rig, self.master, "ik_fk")
            else:
                self.copy_transforms(deform, mch)
            if st.spline_twist:
                # Rotating the tip control around the chain twists it progressively
                con = utils.add_constraint(
                    pbs[deform], 'TRANSFORM', rig, self.ctrls[-1], owner_space='LOCAL', target_space='LOCAL',
                    map_from='ROTATION', map_to='ROTATION', from_rotation_mode='SWING_TWIST_Y',
                    map_to_x_from='X', map_to_y_from='Y', map_to_z_from='Z', mix_mode_rot='ADD')
                factor = (i + 1) / count
                con.from_min_y_rot, con.from_max_y_rot = -math.pi, math.pi
                con.to_min_y_rot, con.to_max_y_rot = -math.pi * factor, math.pi * factor
                if self.fk:
                    utils.add_driver(con, "influence", rig, self.master, "ik_fk")
        for ctrl in self.ctrls:
            self.shape(ctrl, 'CIRCLE', length * 0.06, fit=[self.gen.deform_of[n] for n in self.chain],
                       margin=1.4)
        for name in self.fk:
            pb = self.shape(name, 'CIRCLE', length * 0.05, y=0.5, color="TWEAK")
            pb.lock_location = (True, True, True)
        if self.fk and self.gen.auto_hide:
            for name in self.fk:
                utils.add_driver(rig.data, f'bones["{name}"].hide', rig, self.master, "ik_fk", "var > 0.5")
            for name in self.ctrls:
                utils.add_driver(rig.data, f'bones["{name}"].hide', rig, self.master, "ik_fk", "var < 0.5")
        if self.tail_ref:
            deform = self.gen.deform_of[self.tail_ref]
            utils.add_constraint(pbs[deform], 'COPY_TRANSFORMS', rig, self.tail_mch)
            utils.add_constraint(pbs[deform], 'COPY_LOCATION', rig, self.mch[-1], head_tail=1.0)


class SkirtModule(RigModule):
    """A skirt root with strands hanging around it.

    Strands follow the nearest thigh partly and can be pushed out by the thighs
    (constrained collision with a Floor constraint in the thigh's space). Optional
    column masters drag neighbouring strands; row masters rotate a whole row outward.
    """
    type_id = 'SKIRT'
    label = "Skirt"
    description = ("Skirt root with FK strands (each child chain is a strand). "
                   "Strands follow the nearest thigh and collide with the legs")
    collection = "Extra"

    def __init__(self, gen, root):
        super().__init__(gen, root)
        self.chain = gen.subtree(root)
        self.strands = []
        for child in gen.bones[root].children:
            if gen.module_type(child) == 'NONE':
                self.strands.append(gen.chain_from(child))

    def _angle(self, strand):
        d = self.gen.bones[strand[0]].head - self.gen.bones[self.root].head
        return math.atan2(d.y, d.x)

    def edit(self):
        gen, st = self.gen, self.settings
        parent = self.parent_deform
        # Strands sorted around the skirt so masters and neighbours make sense
        self.strands.sort(key=self._angle)
        self.master = self.copy_ref(self.root, self.root, parent)
        root_def = self.make_deform(self.root, parent)
        self.strand_ctrls = []
        self.spaces = []
        center = gen.bones[self.root].head
        # Column masters every N strands
        self.col_masters = []
        if st.skirt_masters and len(self.strands) > 2:
            freq = st.skirt_master_freq
            for k in range(0, len(self.strands), freq):
                first = gen.bones[self.strands[k][0]]
                last = gen.bones[self.strands[k][-1]]
                name = gen.new_bone(f"{self.stem(self.root)}_master.{k // freq + 1:02d}", first.head, last.tail,
                                    first.z_axis, parent=self.master, collection=self.collection)
                self.col_masters.append((k, name))
        # Row masters: one per segment row
        self.row_masters = []
        rows = max((len(strand) for strand in self.strands), default=0)
        if st.skirt_row_masters:
            for j in range(rows):
                heads = [gen.bones[strand[j]].head for strand in self.strands if len(strand) > j]
                z = sum(v.z for v in heads) / len(heads)
                pos = Vector((center.x, center.y, z))
                name = gen.new_bone(f"{self.stem(self.root)}_row.{j + 1:02d}", pos, pos + Vector((0, 0, -0.05)),
                                    Vector((0, -1, 0)), parent=self.master, collection=self.collection)
                self.row_masters.append(name)
        self.master_follows = []
        for index, strand in enumerate(self.strands):
            first = gen.bones[strand[0]]
            # Each strand top hangs from a space that can follow the closest thigh.
            space = gen.new_bone("MCH-" + strand[0] + "_space", first.head, first.tail, first.z_axis,
                                 parent=self.master, collection="Mechanism")
            self.spaces.append((space, strand[0]))
            for k, master in self.col_masters:
                n = len(self.strands)
                d = min(abs(index - k), n - abs(index - k))
                weight = 1.0 - d / st.skirt_master_freq
                if weight > 0:
                    follow = gen.new_bone(f"MCH-{strand[0]}_{master}_follow", first.head, first.tail, first.z_axis,
                                          parent=master, collection="Mechanism")
                    self.master_follows.append((space, follow, weight))
            ctrl_parent, def_parent, ctrls = space, root_def, []
            if st.skirt_collide:
                # Collision rotates this child of the space, so masters and followers still apply
                ctrl_parent = gen.new_bone("MCH-" + strand[0] + "_collide_space", first.head, first.tail,
                                           first.z_axis, parent=space, collection="Mechanism")
            for i, name in enumerate(strand):
                connect = i > 0 and gen.bones[name].connected
                ctrl = self.copy_ref(name, name, ctrl_parent, connect=connect)
                ctrls.append(ctrl)
                ctrl_parent = ctrl
                def_parent = self.make_deform(name, def_parent, connect=connect)
            self.strand_ctrls.append(ctrls)
        gen.post_edit.append(self._followers)
        self.meta = {"type": "skirt", "master": self.master, "strands": self.strand_ctrls,
                     "col_masters": [m for _k, m in self.col_masters], "row_masters": self.row_masters}

    def _followers(self):
        """After every module is built: follow and collide with the nearest thigh."""
        gen, st = self.gen, self.settings
        thighs = [(n, b) for n, b in gen.edit_bones.items()
                  if n.startswith("DEF-") and "thigh" in n and "twist" not in n and "fan" not in n]
        self.followers = []
        self.colliders = []
        for space, first in self.spaces:
            eb = gen.edit_bones[space]
            if not thighs:
                break
            name, thigh = min(thighs, key=lambda t: (t[1].head - eb.head).length)
            # Strands in front/behind follow the leg more than strands on the sides.
            side = abs(eb.head.x - gen.bones[self.root].head.x)
            reach = max((gen.bones[s[0]].head - gen.bones[self.root].head).length for s in self.strands)
            influence = max(0.0, 0.6 * (1.0 - side / max(reach, 1e-6))) + 0.2
            follower = gen.new_bone("MCH-" + first + "_follow", eb.head, eb.tail, eb.z_axis,
                                    parent=name, collection="Mechanism")
            self.followers.append((space, follower, min(influence, 0.8)))
            if st.skirt_collide:
                # Collision target at the strand's tip, kept outside the thigh by a Floor constraint
                tip = eb.tail.copy()
                target = gen.new_bone(f"MCH-{first}_collide", tip, tip + (eb.tail - eb.head) * 0.2, eb.z_axis,
                                      parent=space, collection="Mechanism")
                local = thigh.matrix.inverted() @ tip
                axis = max(((abs(local.x), 'X', local.x), (abs(local.z), 'Z', local.z)))
                floor = ('FLOOR_' if axis[2] >= 0 else 'FLOOR_NEGATIVE_') + axis[1]
                self.colliders.append(("MCH-" + first + "_collide_space", target, name, floor, axis[0]))

    def pose(self):
        rig, st = self.gen.rig, self.settings
        pbs = rig.pose.bones
        self.copy_transforms(self.gen.deform_of[self.root], self.master)
        reach = max((self.gen.bones[s[0]].head - self.gen.bones[self.root].head).length
                    for s in self.strands) if self.strands else self.h * 0.1
        master = self.shape(self.master, 'CIRCLE', reach * 1.25, color="C")
        for strand, ctrls in zip(self.strands, self.strand_ctrls):
            for name, ctrl in zip(strand, ctrls):
                self.shape(ctrl, 'CIRCLE', self.gen.bones[name].length * 0.18, y=0.5,
                           color=utils.side_letter(name) or ("L" if self.gen.bones[name].head.x > 0 else "R"))
                self.copy_transforms(self.gen.deform_of[name], ctrl)
        for space, follower, influence in getattr(self, "followers", []):
            utils.add_constraint(pbs[space], 'COPY_TRANSFORMS', rig, follower, influence=influence)
        for space, follow, weight in self.master_follows:
            utils.add_constraint(pbs[space], 'COPY_TRANSFORMS', rig, follow, influence=weight)
        for _k, name in self.col_masters:
            self.shape(name, 'MASTER', rig.data.bones[name].length, color="TWEAK")
        for j, name in enumerate(self.row_masters):
            self.shape(name, 'CIRCLE', reach * (1.1 + 0.15 * j), color="TWEAK", wire=1.5)
            for ctrls in self.strand_ctrls:
                if len(ctrls) > j:
                    utils.add_constraint(pbs[ctrls[j]], 'COPY_ROTATION', rig, name, mix_mode='ADD',
                                         owner_space='LOCAL', target_space='LOCAL', use_y=False, use_z=False)
        if getattr(self, "colliders", None):
            utils.add_property(master, "collide", 1.0, description="Strands are pushed away by the thighs")
            utils.add_property(master, "collide_distance", st.skirt_distance, 0.0, 3.0,
                               "Distance kept from the thighs, relative to the rest distance")
            for space, target, thigh, floor, distance in self.colliders:
                con = utils.add_constraint(pbs[target], 'FLOOR', rig, thigh, floor_location=floor,
                                           use_rotation=True, offset=distance * min(st.skirt_distance, 1.0))
                utils.add_driver(con, "offset", rig, self.master, "collide_distance",
                                 f"var * {distance * 0.999:.5f}")
                con = utils.add_constraint(pbs[space], 'DAMPED_TRACK', rig, target, track_axis='TRACK_Y')
                utils.add_driver(con, "influence", rig, self.master, "collide")
