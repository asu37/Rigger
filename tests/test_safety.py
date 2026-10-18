"""Safety net around operators and the checks shown before binding.

    blender -b --factory-startup --python tests/test_safety.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, safety, skin, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


check(getattr(bpy.types.RIGGER_OT_generate.execute, "_rigger_guarded", False)
      and getattr(bpy.types.RIGGER_OT_bind.execute, "_rigger_guarded", False),
      "operators are wrapped by the safety net")
check(generate.RiggerError is safety.RiggerError, "one user-facing error type")

ref = templates.create_reference(ctx, templates.template_specs("HUMAN"), name="reference")
utils.set_mode(ref, 'EDIT')

# An unexpected failure inside an operator: readable message, state restored
real_generate = generate.generate


def boom(context, reference):
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    raise ValueError("boom")


generate.generate = boom
try:
    bpy.ops.rigger.generate()
    message = ""
except RuntimeError as exc:
    message = str(exc)
generate.generate = real_generate
check("failed: boom" in message and "Ctrl+Z" in message and "console" in message,
      f"unexpected error reported readably ({message.strip()[:90]})")
check(ctx.object == ref and ref.mode == 'EDIT' and ref.select_get(),
      f"active object, selection and mode restored ({ctx.object and ctx.object.name}, {ref.mode})")

# An expected problem keeps its own message
def refuse(context, reference):
    raise safety.RiggerError("The reference armature has no bones")


generate.generate = refuse
try:
    bpy.ops.rigger.generate()
    message = ""
except RuntimeError as exc:
    message = str(exc)
generate.generate = real_generate
check("no bones" in message and "failed" not in message, "user-facing errors keep their message")

# The real operator still works afterwards
utils.set_mode(ref, 'OBJECT')
check(bpy.ops.rigger.generate() == {'FINISHED'} and utils.is_rig(ctx.object), "generate works after a failure")
rig = ctx.object

# Checks before binding
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.mesh.primitive_plane_add(size=0.5, location=(0, 0, 1.0))  # open mesh
plane = ctx.object
plane.scale = (1.0, 2.0, 1.0)
msgs = skin.warnings_for(plane, rig)
check(any("open edges" in m for m in msgs), "open edges reported")
check(any("non-uniform scale" in m for m in msgs), "non-uniform scale reported")
check(not any("overlap" in m for m in msgs), "a mesh on the rig is not reported as away from it")
plane.location = (20, 0, 0)
ctx.view_layer.update()
check(any("does not overlap the rig" in m for m in skin.warnings_for(plane, rig)), "mesh away from the rig reported")
other = bpy.data.objects.new("other_rig", bpy.data.armatures.new("other"))
ctx.scene.collection.objects.link(other)
plane.modifiers.new("Armature", 'ARMATURE').object = other
check(any("other_rig" in m for m in skin.warnings_for(plane, rig)), "armature modifier for another rig reported")
check(not any("open edges" in m for m in skin.warnings_for(plane, rig, method='VOXEL')),
      "no open-edge advice when the voxel method is chosen")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
