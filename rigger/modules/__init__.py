# SPDX-License-Identifier: GPL-3.0-or-later
"""Registry of rig module types."""

from .chain import BendyModule, ChainModule, FingerModule, SimpleModule
from .face import EyeModule, FaceModule
from .limbs import ArmModule, LegModule
from .spine import NeckModule, SpineModule
from .spline import SkirtModule, SplineModule

MODULES = {
    cls.type_id: cls
    for cls in (SpineModule, NeckModule, ArmModule, LegModule,
                FingerModule, ChainModule, BendyModule, SplineModule, SkirtModule, FaceModule, EyeModule,
                SimpleModule)
}


def module_items():
    items = [('NONE', "None", "Bone is part of its parent's chain")]
    items += [(cls.type_id, cls.label, cls.description) for cls in MODULES.values()]
    return items
