"""Headless rig behaviour checks: python -I tests/test_behavior.py (needs `bpy`)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402
from mathutils import Matrix, Vector  # noqa: E402

from rigger import generate, properties, templates  # noqa: E402

properties.register()
bpy.ops.wm.read_factory_settings(use_empty=True)
ctx = bpy.context


def rest_drift(rig):
    ctx.view_layer.update()
    worst = (0.0, "")
    for pb in rig.pose.bones:
        if not pb.name.startswith("DEF-"):
            continue
        d = max(abs(a - b) for ra, rb in zip(pb.matrix, pb.bone.matrix_local) for a, b in zip(ra, rb))
        worst = max(worst, (d, pb.name))
    return worst


def head(rig, name):
    return rig.pose.bones[name].head.copy()


for key in templates.TEMPLATES:
    ref = templates.create_reference(ctx, templates.template_specs(key), name=f"ref_{key}")
    ref.rigger.rig_name = f"rig_{key}"
    rig = generate.generate(ctx, ref)
    drift, name = rest_drift(rig)
    print(f"{key}: rest drift {drift:.6f} ({name})")
    assert drift < 1e-4, (key, name, drift)

    # Switch all limbs to FK and back to IK: rest must stay put
    for value in (0.0, 1.0):
        for pb in rig.pose.bones:
            if "ik_fk" in pb:
                pb["ik_fk"] = value
        rig.update_tag()
        drift, name = rest_drift(rig)
        assert drift < 1e-4, (key, value, name, drift)

ref = bpy.data.objects["ref_HUMAN"]
rig = bpy.data.objects["rig_HUMAN"]
for side in "LR":
    pb = rig.pose.bones[f"foot_ik.{side}"]
    pb["ik_fk"] = 1.0
    knee_rest = head(rig, f"DEF-shin.{side}")
    pb.location = (0, 0, 0.3)  # local Z is up for the foot control
    rig.update_tag(); ctx.view_layer.update()
    knee = head(rig, f"DEF-shin.{side}")
    ankle = head(rig, f"DEF-foot.{side}")
    print(f"leg {side}: knee moved {knee - knee_rest}, ankle z {ankle.z:.3f}")
    assert knee.y < knee_rest.y - 0.05, "knee should bend forward"
    pb.location = (0, 0, 0)

    hand = rig.pose.bones[f"hand_ik.{side}"]
    hand["ik_fk"] = 1.0
    elbow_rest = head(rig, f"DEF-forearm.{side}")
    # Move the hand toward the shoulder in world space
    m = hand.matrix.copy()
    m.translation += Vector((-0.25 if side == "L" else 0.25, 0, 0))
    hand.matrix = m
    rig.update_tag(); ctx.view_layer.update()
    elbow = head(rig, f"DEF-forearm.{side}")
    print(f"arm {side}: elbow moved {elbow - elbow_rest}")
    assert elbow.y > elbow_rest.y + 0.05, "elbow should bend backward"
    hand.matrix_basis = Matrix()

# FK rotation propagates to deform bones
fk = rig.pose.bones["forearm_fk.L"]
rig.pose.bones["hand_ik.L"]["ik_fk"] = 0.0
fk.rotation_quaternion = (0.7071, 0.7071, 0, 0)
rig.update_tag(); ctx.view_layer.update()
assert (rig.pose.bones["DEF-hand.L"].head - rig.pose.bones["hand_fk.L"].head).length < 1e-4
print("OK")
