# SPDX-License-Identifier: GPL-3.0-or-later
"""Base class shared by every rig module.

A module is declared on the first bone of a chain in the reference armature
(``pose_bone.rigger.module``). The chain continues through child bones that
have no module of their own. Generation runs in two passes so modules can
reference each other's bones:

* ``edit()``  - create bones (armature in Edit Mode)
* ``pose()``  - constraints, drivers, custom properties and shapes (Pose Mode)

Each module registers the deform bone created for every reference bone it
owns in ``gen.deform_of`` so child modules know what to attach to.
"""

from .. import utils


class RigModule:
    type_id = ""
    label = ""
    description = ""
    # Number of chain bones the module uses (None = whole chain)
    chain_length = None
    # Minimum number of bones required
    min_length = 1
    collection = "Extra"

    def __init__(self, gen, root):
        self.gen = gen
        self.root = root
        self.settings = gen.ref.pose.bones[root].rigger
        chain = gen.chain_from(root)
        if self.chain_length is not None:
            chain = chain[: self.chain_length]
        self.chain = chain
        self.suffix = utils.split_side(root)[1]
        self.side = utils.side_letter(root)
        self.meta = {}

    # -- validation ---------------------------------------------------------
    def validate(self):
        if len(self.chain) < self.min_length:
            return (f"'{self.root}' ({self.label}) needs a chain of at least "
                    f"{self.min_length} bones, found {len(self.chain)}")
        return None

    # -- naming helpers -----------------------------------------------------
    def stem(self, ref_name):
        return utils.split_side(ref_name)[0]

    def name(self, ref_name, kind="", prefix=""):
        """'upper_arm.L' + kind '_fk' -> 'upper_arm_fk.L' (with optional prefix)."""
        return f"{prefix}{self.stem(ref_name)}{kind}{self.suffix}"

    def def_name(self, ref_name):
        return f"DEF-{ref_name}"

    @property
    def color(self):
        return self.side or "C"

    @property
    def h(self):
        """Character size, the unit for default control sizes."""
        return self.gen.scale

    @property
    def parent_deform(self):
        return self.gen.parent_deform(self.root)

    # -- shared building blocks --------------------------------------------
    def make_deform(self, ref_name, parent, connect=False, head=None, tail=None):
        ref = self.gen.bones[ref_name]
        name = self.gen.new_bone(
            self.def_name(ref_name),
            head if head is not None else ref.head,
            tail if tail is not None else ref.tail,
            ref.z_axis, parent=parent, connect=connect, deform=True,
            collection="Deform")
        self.gen.deform_of[ref_name] = name
        return name

    def copy_ref(self, ref_name, name, parent, connect=False, collection=None, z_axis=None):
        ref = self.gen.bones[ref_name]
        return self.gen.new_bone(name, ref.head, ref.tail,
                                 z_axis if z_axis is not None else ref.z_axis,
                                 parent=parent, connect=connect,
                                 collection=collection or self.collection)

    def shape(self, name, kind, radius, y=0.0, color=None, fit=None, margin=1.3, wire=2.0):
        """Give a control its shape.

        radius: world size of the shape (shapes have a radius of 1 in shape units)
        y:      position along the bone (0 = head, 1 = tail)
        fit:    deform bones whose skinned vertices size the shape after binding
        """
        from .. import widgets
        rig = self.gen.rig
        pb = rig.pose.bones[name]
        length = max(rig.data.bones[name].length, 1e-6)
        widgets.apply_widget(pb, kind, radius * self.gen.widget_scale / length, y, wire)
        widgets.set_color(pb, color or self.color)
        if fit:
            self.gen.fit[name] = {"bones": [b for b in fit if b], "y": y, "margin": margin}
        return pb

    def copy_transforms(self, owner, target, influence=1.0):
        pb = self.gen.rig.pose.bones[owner]
        return utils.add_constraint(pb, 'COPY_TRANSFORMS', self.gen.rig, target,
                                    influence=influence)

    # -- passes ---------------------------------------------------------------
    def edit(self):
        raise NotImplementedError

    def pose(self):
        raise NotImplementedError
