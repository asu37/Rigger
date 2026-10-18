"""Headless smoke test: python -I tests/test_generate.py (needs the `bpy` module)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402

from rigger import properties, templates, generate  # noqa: E402

properties.register()
bpy.ops.wm.read_factory_settings(use_empty=True)
ctx = bpy.context
for key in templates.TEMPLATES:
    ref = templates.create_reference(ctx, templates.template_specs(key), name=f"ref_{key}")
    ref.rigger.rig_name = f"rig_{key}"
    rig = generate.generate(ctx, ref)
    print(key, "bones:", len(rig.data.bones), "deform:", sum(b.use_deform for b in rig.data.bones))
    # regenerate in place
    rig2 = generate.generate(ctx, ref)
    assert rig2 == rig
print("OK")
