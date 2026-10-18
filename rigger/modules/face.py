# SPDX-License-Identifier: GPL-3.0-or-later
"""Eye and face modules."""

import math
import re

from mathutils import Matrix, Vector

from .. import utils
from .base import RigModule


class EyeModule(RigModule):
    type_id = 'EYE'
    label = "Eye"
    description = "Eye bone aiming at a target placed in front of the face"
    chain_length = 1
    collection = "Face"

    def _master_target(self):
        """Create the shared 'eyes_target' once, centred in front of all eyes."""
        gen = self.gen
        if "eyes_target" in gen.shared:
            return gen.shared["eyes_target"]
        eyes = [m for m in gen.modules if isinstance(m, EyeModule)]
        size = max(gen.bones[m.chain[0]].length for m in eyes)
        distance = max(size * self.settings.eye_distance, 0.2 * gen.scale * self.settings.eye_distance / 12.0)
        gen.shared["eyes_distance"] = distance
        targets = [self._target_pos(gen.bones[m.chain[0]]) for m in eyes]
        pos = sum(targets, Vector()) / len(targets)
        parent = self.parent_deform
        name = gen.new_bone("eyes_target", pos, pos + Vector((0, -size * 2.0, 0)),
                            Vector((0, 0, 1)), parent=parent, collection="Face")
        gen.shared["eyes_target"] = name
        # The target follows the head by default, or stays in place (root)
        gen.post_edit.append(lambda: gen.add_space_switch(name, [("Head", parent), ("Root", "root")]))
        return name

    def _target_pos(self, ref):
        """Point the eye looks at in its rest pose, along the eye bone."""
        direction = (ref.tail - ref.head).normalized()
        return ref.head + direction * self.gen.shared["eyes_distance"]

    def edit(self):
        gen = self.gen
        eye = self.chain[0]
        master = self._master_target()
        ref = gen.bones[eye]
        pos = self._target_pos(ref)
        self.target = gen.new_bone(self.name(eye, "_target"), pos,
                                   pos + Vector((0, -ref.length, 0)), Vector((0, 0, 1)),
                                   parent=master, collection="Face")
        self.ctrl = self.copy_ref(eye, eye, self.parent_deform)
        self.make_deform(eye, self.parent_deform)
        self.meta = {"type": "eye", "ctrl": self.ctrl, "target": self.target, "master": master}

    def pose(self):
        rig = self.gen.rig
        master = self.gen.shared["eyes_target"]
        eye_len = self.gen.bones[self.chain[0]].length
        if not self.gen.shared.get("eyes_target_done"):
            eyes = [m for m in self.gen.modules if isinstance(m, EyeModule)]
            spread = max((abs(self.gen.bones[m.chain[0]].head.x) for m in eyes), default=eye_len)
            self.shape(master, 'EYES', spread + eye_len * 1.4, color="C")
            self.gen.shared["eyes_target_done"] = True
        self.shape(self.target, 'CIRCLE', eye_len * 0.8)
        ctrl = self.shape(self.ctrl, 'CIRCLE', eye_len * 1.2)
        ctrl.lock_location = (True, True, True)
        utils.add_property(ctrl, "aim", 1.0, description="1 = the eye looks at its target, 0 = rotate it freely (FK)")
        con = utils.add_constraint(ctrl, 'DAMPED_TRACK', rig, self.target, track_axis='TRACK_Y')
        utils.add_driver(con, "influence", rig, self.ctrl, "aim")
        self.copy_transforms(self.gen.deform_of[self.chain[0]], self.ctrl)


def face_role(stem):
    """Behaviour of a face bone, from its name (side suffix removed)."""
    if stem.startswith("jaw"):
        return "jaw"
    if stem.startswith("lip_corner"):
        return "corner"
    if stem.startswith(("lip_upper", "lip_lower")):
        return "lip"
    if stem.startswith("lid_upper"):
        return "lid_upper"
    if stem.startswith("lid_lower"):
        return "lid_lower"
    if stem.startswith("tongue"):
        return "tongue"
    if stem.startswith("brow"):
        return "brow"
    if stem.startswith("cheek"):
        return "cheek"
    return "tweak"


MOUTH_PREFIXES = ("jaw", "chin", "lip", "teeth", "tongue", "nose", "mouth")
_INDEX = re.compile(r"^(.*?)\.(\d+)$")


def split_index(stem):
    """'lip_upper.02' -> ('lip_upper', 2); 'lip_upper' -> ('lip_upper', 0)."""
    m = _INDEX.match(stem)
    return (m.group(1), int(m.group(2))) if m else (stem, 0)


class FaceModule(RigModule):
    """Face rig built from the bones under a 'face' root bone.

    Bone names decide behaviour (side suffix .L/.R, optional index .01, .02...):
    * ``jaw``             - jaw control; bones parented to the jaw follow it
    * ``lip_corner``      - follows the jaw halfway
    * ``lip_upper/lip_lower`` - middle lip without side, side lips follow their corner
      softly (more toward the corner); ``lip_upper.01.L``... for several lips per side
    * ``lid_upper/lid_lower`` - pivot around the eye, follow its rotation a little
      (``auto_lid_*`` sliders); the face control gets ``blink_L`` / ``blink_R``
    * ``brow*``           - eyebrow bones, optionally driven by a master per side
    * ``cheek``           - optionally follows the lip corner
    * ``tongue.*``        - FK chain
    * anything else       - tweak control (nose, teeth...)

    Options add a lips offset control, a muzzle (mouth + nose), lips roll controls,
    sticky lips / lips retain, jaw by translation and eyelid main controls.
    """
    type_id = 'FACE'
    label = "Face"
    description = ("Face root: every child bone becomes a face control (jaw, lips, "
                   "lip corners, eyelids with blink, brows, cheeks, nose, teeth, tongue)")
    collection = "Face"
    LID_FOLLOW = {"lid_upper": 0.3, "lid_lower": 0.15}

    def __init__(self, gen, root):
        super().__init__(gen, root)
        self.chain = gen.subtree(root)
        freq = self.settings.lips_masters_freq
        order = {"jaw": 0, "corner": 1, "brow": 2}

        def key(n):
            stem = self.stem(n)
            role = face_role(stem)
            base, index = split_index(stem)
            master = role == "lip" and freq > 1 and index and index % freq == 0
            return (gen.bones[n].depth, order.get(role, 3), 0 if master else 1, n)
        self.chain = [root] + sorted(self.chain[1:], key=key)
        # Number of indexed lips / lids per (row, side)
        self.counts = {}
        for n in self.chain[1:]:
            stem, suffix = utils.split_side(n)
            base, index = split_index(stem)
            if index:
                self.counts[(base, suffix)] = max(self.counts.get((base, suffix), 0), index)

    # -- helpers ------------------------------------------------------------------------
    def _lip_t(self, name):
        """0 at the middle of the mouth, 1 at the corner."""
        stem, suffix = utils.split_side(name)
        if not suffix:
            return 0.0
        base, index = split_index(stem)
        count = self.counts.get((base, suffix), 0)
        return index / (count + 1) if index else 0.5

    def _soft(self, t):
        return t ** (2.0 ** (1.0 - self.settings.soft_lips_linear))

    def _follow(self, name, base, target, influence, tag="", driver=None):
        """Space bone under 'base' that blends toward following 'target'."""
        gen = self.gen
        ref = gen.bones[name]
        stem, suffix = utils.split_side(name)
        tag = f"_{tag}" if tag else ""
        space = gen.new_bone(f"MCH-{stem}{tag}_space{suffix}", ref.head, ref.tail, ref.z_axis,
                             parent=base, collection="Mechanism")
        follow = gen.new_bone(f"MCH-{stem}{tag}_follow{suffix}", ref.head, ref.tail, ref.z_axis,
                              parent=target, collection="Mechanism")
        self.follows.append((space, follow, influence, driver))
        return space

    def _offset_space(self, name, base, offset, influence):
        """Space bone that applies the offset control's local transform around its pivot."""
        gen = self.gen
        eb = gen.edit_bones[offset]
        stem, suffix = utils.split_side(name)
        space = gen.new_bone(f"MCH-{stem}_{offset}_ofs{suffix}", eb.head, eb.tail, eb.z_axis,
                             parent=base, collection="Mechanism")
        self.offsets.append((space, offset, influence))
        return space

    def _offset_applies(self, offset, name, role):
        stem = self.stem(name)
        if offset == self.muzzle:
            return stem.startswith(MOUTH_PREFIXES) or name in (self.lips_offset, self.roll_upper, self.roll_lower)
        if offset == self.lips_offset:
            return role in ("lip", "corner") or name in (self.roll_upper, self.roll_lower)
        return False

    def _chain_parent(self, name, role, base, follows):
        """Parent chain for a control: base -> follow spaces -> offset spaces."""
        carried = {o: self.carried.get(base, {}).get(o, 0.0) for o in self.offset_ctrls}
        parent = base
        for i, (target, influence, driver) in enumerate(follows):
            parent = self._follow(name, parent, target, influence, tag=f"f{i}" if i else "", driver=driver)
            f = 0.0 if driver else influence
            for o in self.offset_ctrls:
                carried[o] = (1 - f) * carried[o] + f * self.carried.get(target, {}).get(o, 0.0)
        own = {}
        for o in self.offset_ctrls:
            if o and o != name and self._offset_applies(o, name, role):
                influence = max(0.0, 1.0 - carried[o])
                if influence > 1e-3:
                    parent = self._offset_space(name, parent, o, influence)
                own[o] = 1.0
            else:
                own[o] = carried[o]
        return parent, own

    def _new_control(self, name, head, tail, z, base, role="tweak"):
        """Generated face control (not from a reference bone), with offsets applied."""
        gen = self.gen
        parent, carried = base, {}
        for o in self.offset_ctrls:
            carried[o] = self.carried.get(base, {}).get(o, 0.0)
            if self._offset_applies(o, name, role) and carried[o] < 1.0:
                eb = gen.edit_bones[o]
                parent = gen.new_bone(f"MCH-{name}_{o}_ofs", eb.head, eb.tail, eb.z_axis, parent=parent,
                                      collection="Mechanism")
                self.offsets.append((parent, o, 1.0 - carried[o]))
                carried[o] = 1.0
        ctrl = gen.new_bone(name, head, tail, z, parent=parent, collection="Face")
        self.carried[ctrl] = carried
        return ctrl

    # -- edit -------------------------------------------------------------------------------
    def edit(self):
        gen, s = self.gen, self.settings
        parent = self.parent_deform
        root = self.root
        self.follows, self.offsets, self.blinks, self.ctrls, self.carried = [], [], [], {}, {}
        self.lid_mains, self.lip_rolls, self.jaw_rot = {}, [], None
        self.face = self.copy_ref(root, root, parent)
        self.ctrls[root] = self.face
        self.make_deform(root, parent)
        self.jaw = None
        bones = gen.bones
        h = self.h
        forward = Vector((0, -1, 0))
        up = Vector((0, 0, 1))

        # Offset controls (created first so their pivots exist)
        lips_up = bones["lip_upper"] if "lip_upper" in self.chain else None
        lips_low = bones["lip_lower"] if "lip_lower" in self.chain else None
        jaw_ref = next((bones[n] for n in self.chain if face_role(self.stem(n)) == "jaw"), None)
        if lips_up and lips_low:
            mouth = (lips_up.head + lips_low.head) * 0.5
        elif jaw_ref:
            mouth = jaw_ref.tail.copy()
        else:
            mouth = None
        self.muzzle = self.lips_offset = self.roll_upper = self.roll_lower = None
        self.offset_ctrls = []
        size = h * 0.012
        if mouth is not None and s.muzzle:
            pos = mouth + forward * h * 0.03
            self.muzzle = self._new_control("muzzle", pos, pos + forward * size, up, self.face)
            self.offset_ctrls.append(self.muzzle)
        if mouth is not None and s.lips_offset:
            pos = mouth + forward * h * 0.02
            self.lips_offset = self._new_control("lips_offset", pos, pos + forward * size, up, self.face)
            self.offset_ctrls.append(self.lips_offset)

        for name in self.chain[1:]:
            ref = bones[name]
            stem, suffix = utils.split_side(name)
            role = face_role(stem)
            base = self.ctrls.get(ref.parent, self.face)
            follows = []
            t = self._lip_t(name) if role == "lip" else 0.0
            if role == "corner" and self.jaw:
                follows.append((self.jaw_children, 0.5, None))
            elif role == "lip" and suffix and s.soft_lips:
                corner = self.ctrls.get("lip_corner" + suffix)
                if corner:
                    follows.append((corner, self._soft(t), None))
            if role == "lip" and s.lips_masters_freq > 1 and suffix:
                lip_base, index = split_index(stem)
                freq = s.lips_masters_freq
                if index and index % freq:
                    master_index = int(round(index / freq)) * freq
                    master = self.ctrls.get(f"{lip_base}.{master_index:02d}{suffix}")
                    if master:
                        follows.append((master, max(0.0, 1.0 - abs(index - master_index) / freq), None))
            if role == "lip" and stem.startswith("lip_upper") and self.jaw:
                follows.append((self.jaw_children, None, ("retain", 1.0 - t)))
            if role == "brow" and s.brow_master:
                master = self._brow_master(suffix, base)
                if master:
                    follows.append((master, self._brow_weight(name, suffix) if s.soft_brows else 1.0, None))
            if role == "cheek" and s.soft_cheeks:
                corner = self.ctrls.get("lip_corner" + suffix)
                if corner:
                    follows.append((corner, 0.35, None))
            if role in ("lid_upper", "lid_lower"):
                base = self._lid_space(name, role, suffix, base)
                lid_base, index = split_index(stem)
                main = self.lid_mains.get((lid_base, suffix))
                if main:
                    follows.append((main, 1.0 - abs(index / (self.counts[(lid_base, suffix)] + 1) - 0.5), None))
            ctrl_parent, carried = self._chain_parent(name, role, base, follows)
            is_jaw = role == "jaw" and self.jaw is None
            if is_jaw and s.jaw_translate:
                # The control sits at the chin; moving it down rotates the jaw.
                tip = ref.tail
                ctrl = gen.new_bone(name, tip, tip + (ref.tail - ref.head).normalized() * h * 0.015, ref.z_axis,
                                    parent=ctrl_parent, collection="Face")
                self.jaw_rot = self.copy_ref(name, f"MCH-{name}_rot", ctrl_parent, collection="Mechanism")
            else:
                ctrl = self.copy_ref(name, name, ctrl_parent)
            self.ctrls[name] = ctrl
            self.carried[ctrl] = carried
            if is_jaw:
                self.jaw, self.jaw_ref = ctrl, name
                self.jaw_children = self.jaw_rot or ctrl
                self.ctrls[name] = self.jaw_children  # children follow the jaw
                self.carried[self.jaw_children] = carried
                self.jaw_open = self._jaw_open_sign(ref)
            self.make_deform(name, gen.deform_of[ref.parent],
                             connect=ref.connected and role == "tongue")
            if role == "lip" and s.lips_roll:
                self.lip_rolls.append((ctrl, "upper" if stem.startswith("lip_upper") else "lower", t))

        # Lips roll controls in front of the mouth
        if self.lip_rolls and mouth is not None:
            for row, offset in (("upper", 1), ("lower", -1)):
                base = self.face if row == "upper" else (self.jaw_children or self.face)
                pos = mouth + forward * h * 0.025 + up * offset * h * 0.012
                name = self._new_control(f"lips_roll_{row}", pos, pos + forward * size, up, base, role="lip")
                setattr(self, f"roll_{row}", name)
        extras = [self.jaw, self.muzzle, self.lips_offset, self.roll_upper, self.roll_lower]
        extras += list(self.lid_mains.values()) + list(self.brow_masters.values())
        controls = [c for c in list(self.ctrls.values()) + extras if c and not c.startswith("MCH-")]
        self.meta = {"type": "face", "face": self.face, "jaw": self.jaw,
                     "blinks": sorted({f"blink_{s_}" for _b, s_, _a in self.blinks}),
                     "controls": list(dict.fromkeys(controls)),
                     "muzzle": self.muzzle, "lips_offset": self.lips_offset,
                     "lids": list(self.lid_mains.values())}

    @property
    def brow_masters(self):
        if not hasattr(self, "_brow_masters"):
            self._brow_masters = {}
        return self._brow_masters

    def _brow_bones(self, suffix):
        return [n for n in self.chain if face_role(self.stem(n)) == "brow" and utils.split_side(n)[1] == suffix]

    def _brow_master(self, suffix, base):
        if suffix in self.brow_masters:
            return self.brow_masters[suffix]
        brows = self._brow_bones(suffix)
        if len(brows) < 2:
            return None
        gen = self.gen
        heads = [gen.bones[n].head for n in brows]
        inner = min(heads, key=lambda v: abs(v.x))
        outer = max(heads, key=lambda v: abs(v.x))
        lift = Vector((0, -1, 0)) * self.h * 0.006 + Vector((0, 0, 1)) * self.h * 0.006
        name = gen.new_bone(f"brow_master{suffix}", inner + lift, outer + lift, Vector((0, -1, 0)), parent=base,
                            collection="Face")
        self.brow_masters[suffix] = name
        self.carried[name] = self.carried.get(base, {})
        return name

    def _brow_weight(self, name, suffix):
        """1 at the inner end of the eyebrow, 0.4 at the outer end."""
        gen = self.gen
        xs = [abs(gen.bones[n].head.x) for n in self._brow_bones(suffix)]
        x = abs(gen.bones[name].head.x)
        span = max(xs) - min(xs)
        t = (x - min(xs)) / span if span > 1e-6 else 0.0
        return 1.0 - 0.6 * t

    def _jaw_open_sign(self, ref):
        """+1 if a positive rotation around the jaw's X axis lowers the chin, else -1."""
        y_axis = (ref.tail - ref.head).normalized()
        x_axis = y_axis.cross(ref.z_axis).normalized()
        rotated = Matrix.Rotation(0.1, 3, x_axis) @ (ref.tail - ref.head)
        return 1.0 if rotated.z < (ref.tail - ref.head).z else -1.0

    def _lid_space(self, name, role, suffix, base):
        """Eyelids pivot around the eye centre and follow the eye a little."""
        gen, s = self.gen, self.settings
        eye_name = next((n for n in gen.bones if utils.split_side(n)[1] == suffix
                         and utils.split_side(n)[0] == "eye"), None)
        if eye_name is None:
            return base
        stem = utils.split_side(name)[0]
        lid_base, index = split_index(stem)
        count = self.counts.get((lid_base, suffix), 0)
        eye_ref, lid_ref = gen.bones[eye_name], gen.bones[name]
        center = eye_ref.head
        if (lid_ref.head - center).length < 1e-5:
            return base
        if count > 1 and s.lid_masters:
            # Main control per eyelid, pivoting around the eye: it blinks and follows the eye.
            if (lid_base, suffix) not in self.lid_mains:
                lids = [n for n in self.chain if utils.split_side(n)[1] == suffix
                        and split_index(utils.split_side(n)[0])[0] == lid_base]
                mid = sum((gen.bones[n].head for n in lids), Vector()) / len(lids)
                pivot = self._pivot(f"{lid_base}_main", role, suffix, base, center, mid, eye_ref, eye_name,
                                    lid_base, index=None)
                main = gen.new_bone(f"{lid_base}_main{suffix}", mid, mid + (mid - center).normalized()
                                    * self.h * 0.01, eye_ref.z_axis, parent=pivot, collection="Face")
                self.lid_mains[(lid_base, suffix)] = main
                self.carried[main] = {}
            return gen.new_bone(f"MCH-{stem}_pivot{suffix}", center, lid_ref.head, eye_ref.z_axis,
                                parent=base, collection="Mechanism")
        return self._pivot(stem, role, suffix, base, center, lid_ref.head, eye_ref, eye_name, lid_base, index)

    def _pivot(self, stem, role, suffix, base, center, point, eye_ref, eye_name, lid_base, index):
        gen = self.gen
        space = gen.new_bone(f"MCH-{stem}_space{suffix}", center, point, eye_ref.z_axis,
                             parent=base, collection="Mechanism")
        if eye_name in gen.edit_bones:  # eye control of the Eye module
            follow = gen.new_bone(f"MCH-{stem}_follow{suffix}", center, point, eye_ref.z_axis,
                                  parent=eye_name, collection="Mechanism")
            prop = "auto_" + role
            self.follows.append((space, follow, self.LID_FOLLOW[role], ("prop", prop)))
        if role != "lid_upper":
            return space
        lower_name = "lid_lower" + (f".{index:02d}" if index else "") + suffix
        lower = gen.bones.get(lower_name)
        if index is None:
            lowers = [n for n in self.chain if utils.split_side(n)[1] == suffix
                      and split_index(utils.split_side(n)[0])[0] == "lid_lower"]
            if lowers:
                low_mid = sum((gen.bones[n].head for n in lowers), Vector()) / len(lowers)
                return self._blink(stem, suffix, space, center, point, low_mid, eye_ref)
            return space
        if lower is None:
            return space
        return self._blink(stem, suffix, space, center, point, lower.head, eye_ref)

    def _blink(self, stem, suffix, space, center, upper, lower, eye_ref):
        gen = self.gen
        up_dir, low_dir = upper - center, lower - center
        axis = up_dir.cross(low_dir)
        if axis.length < 1e-8:
            return space
        x_axis = axis.normalized()
        z_axis = x_axis.cross(up_dir.normalized())
        blink = gen.new_bone(f"MCH-{stem}_blink{suffix}", center, upper, z_axis,
                             parent=space, collection="Mechanism")
        self.blinks.append((blink, utils.side_letter(stem + suffix) or "C", up_dir.angle(low_dir)))
        return blink

    # -- pose -----------------------------------------------------------------------------------
    def pose(self):
        rig, s = self.gen.rig, self.settings
        pbs = rig.pose.bones
        h = self.h
        face = self.shape(self.face, 'DIAMOND', h * 0.012, y=1.0, color="FACE")
        face.lock_location = (True, True, True)
        utils.add_property(face, "auto_lid_upper", self.LID_FOLLOW["lid_upper"],
                           description="How much the upper eyelids follow the eyes")
        utils.add_property(face, "auto_lid_lower", self.LID_FOLLOW["lid_lower"],
                           description="How much the lower eyelids follow the eyes")
        for name, ctrl in self.ctrls.items():
            if ctrl == self.face:
                continue
            stem, suffix = utils.split_side(name)
            role = face_role(stem)
            color = utils.side_letter(name) if suffix else "FACE"
            if role == "jaw":
                ctrl = self.jaw
                self.shape(ctrl, 'JAW' if not self.jaw_rot else 'CUBE', h * (0.022 if not self.jaw_rot else 0.006),
                           y=1.0 if not self.jaw_rot else 0.0, color="FACE")
            elif role == "tongue":
                self.shape(ctrl, 'CIRCLE', h * 0.005, y=0.5, color="FACE")
            elif role == "lid_upper" or role == "brow":
                self.shape(ctrl, 'ARC', h * 0.009 * (0.6 if split_index(stem)[1] else 1.0), color=color)
            elif role == "lid_lower":
                self.shape(ctrl, 'ARC_DOWN', h * 0.009 * (0.6 if split_index(stem)[1] else 1.0), color=color)
            else:
                self.shape(ctrl, 'CIRCLE', h * 0.0035, color=color, wire=1.5)
        for main in self.lid_mains.values():
            self.shape(main, 'ARC' if "upper" in main else 'ARC_DOWN', h * 0.011, color=utils.side_letter(main))
        for master in self.brow_masters.values():
            pb = self.shape(master, 'SQUARE', h * 0.004, y=0.5, color=utils.side_letter(master))
            pb.custom_shape_scale_xyz = (pb.custom_shape_scale_xyz[0], 1.0, pb.custom_shape_scale_xyz[2] * 0.5)
        for extra, kind in ((self.muzzle, 'CUBE'), (self.lips_offset, 'DIAMOND'), (self.roll_upper, 'ARC'),
                            (self.roll_lower, 'ARC_DOWN')):
            if extra:
                self.shape(extra, kind, h * (0.008 if kind != 'CUBE' else 0.01), color="FACE")
        for ref_name, ctrl in self.ctrls.items():
            self.copy_transforms(self.gen.deform_of[ref_name], ctrl)

        jaw_props = self.jaw and any(d and d[0] == "retain" for _s, _f, _i, d in self.follows)
        if jaw_props:
            utils.add_property(pbs[self.jaw], "lips_retain", 0.0,
                               description="Keep the lips sealed when opening the jaw")
            if s.sticky_lips:
                utils.add_property(pbs[self.jaw], "sticky", 1.0,
                                   description="Lips stay sealed when the jaw closes past its rest pose")
        for space, follow, influence, driver in self.follows:
            con = utils.add_constraint(pbs[space], 'COPY_TRANSFORMS', rig, follow,
                                       influence=influence if influence is not None else 0.0)
            if driver is None:
                continue
            if driver[0] == "prop":
                utils.add_driver(con, "influence", rig, self.face, driver[1])
            elif driver[0] == "retain":
                weight = driver[1]
                variables = [("r", 'prop', self.jaw, "lips_retain")]
                expr = f"r * {weight:.4f}"
                if s.sticky_lips:
                    channel = 'LOC_Z' if self.jaw_rot else 'ROT_X'
                    sign = 1.0 if self.jaw_rot else self.jaw_open
                    variables += [("st", 'prop', self.jaw, "sticky"), ("j", 'transform', self.jaw, channel)]
                    # closing = moving the jaw control up, or rotating it opposite to opening
                    close = "j > 0.0" if self.jaw_rot else f"j * {sign} < 0.0"
                    expr = f"max(r, st if {close} else 0.0) * {weight:.4f}"
                utils.add_driver_mixed(con, "influence", rig, variables, expr)
        for space, offset, influence in self.offsets:
            utils.add_constraint(pbs[space], 'COPY_TRANSFORMS', rig, offset, owner_space='LOCAL',
                                 target_space='LOCAL', influence=influence)
        from ..poses import has_blend
        for bone, side, angle in self.blinks:
            prop = f"blink_{side}"
            if prop not in face:
                utils.add_property(face, prop, 0.0, description="Close the eyelids")
            if has_blend(self.gen, prop):
                continue  # a stored blink pose closes the eyelids instead
            pb = pbs[bone]
            pb.rotation_mode = 'XYZ'
            utils.add_driver(pb, "rotation_euler", rig, self.face, prop, f"var * {angle:.6f}", index=0)
        for ctrl, row, t in self.lip_rolls:
            roll = getattr(self, f"roll_{row}")
            if roll:
                utils.add_constraint(pbs[ctrl], 'COPY_ROTATION', rig, roll, use_x=True, use_y=False, use_z=False,
                                     mix_mode='ADD', owner_space='LOCAL', target_space='LOCAL',
                                     influence=max(0.0, 1.0 - t * t))
        if self.jaw_rot:
            jaw = pbs[self.jaw]
            utils.add_property(jaw, "jaw_speed", 1.0, 0.0, 5.0, "How far the jaw opens when moving the control")
            length = self.gen.bones[self.jaw_ref].length
            con = utils.add_constraint(pbs[self.jaw_rot], 'TRANSFORM', rig, self.jaw, owner_space='LOCAL',
                                       target_space='LOCAL', map_from='LOCATION', map_to='ROTATION',
                                       map_to_x_from='Z', map_to_y_from='X', map_to_z_from='Y')
            con.from_min_z, con.from_max_z = -length, length
            for attr, sign in (("to_min_x_rot", 1.0), ("to_max_x_rot", -1.0)):
                utils.add_driver(con, attr, rig, self.jaw, "jaw_speed",
                                 f"var * {sign * self.jaw_open * math.radians(60):.5f}")
            if self.settings.jaw_unlock_y:
                # Sliding the control along the jaw also slides the jaw
                utils.add_constraint(pbs[self.jaw_rot], 'COPY_LOCATION', rig, self.jaw, use_x=False, use_z=False,
                                     use_offset=True, owner_space='LOCAL', target_space='LOCAL')
