"""Viewport picker: layout built from the rig, hit testing and selection.

    blender -b --factory-startup --python tests/test_picker.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bpy  # noqa: E402

import rigger  # noqa: E402
from rigger import generate, picker, templates, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


ref = templates.create_reference(ctx, templates.template_specs("HUMAN"), name="reference")
rig = generate.generate(ctx, ref)
utils.set_mode(rig, 'POSE')
buttons = picker.build_buttons(rig)
keys = {b.key for b in buttons}
expected = {"face", "head", "neck", "chest", "spine", "hips", "torso", "root", "all"}
for side in ("L", "R"):
    expected |= {f"{k}.{side}" for k in ("shoulder", "upper_arm", "forearm", "hand", "hand_ik", "arm_pole",
                                         "fingers", "thigh", "shin", "foot", "foot_ik", "toe", "leg_pole")}
check(expected <= keys, f"every body part has a button (missing: {sorted(expected - keys)})")
check(all(b.bones and all(n in rig.pose.bones for n in b.bones) for b in buttons),
      "buttons only point at existing controls")
by_key = {b.key: b for b in buttons}
check(by_key["hand.L"].rect[0] > 0 and by_key["hand.R"].rect[2] < 0,
      "character's left on the right of the picker (facing the viewer)")
check(all(-0.5 <= b.rect[0] and b.rect[2] <= 0.5 and 0 <= b.rect[1] and b.rect[3] <= picker.HEIGHT
          for b in buttons), "buttons inside the picker panel")
overlaps = [(a.key, b.key) for i, a in enumerate(buttons) for b in buttons[i + 1:]
            if a.rect[0] < b.rect[2] and b.rect[0] < a.rect[2] and a.rect[1] < b.rect[3] and b.rect[1] < a.rect[3]]
check(not overlaps, f"buttons do not overlap ({overlaps[:4]})")

size = 300.0
hits = []
for b in buttons:
    cx, cy = (b.rect[0] + b.rect[2]) / 2, (b.rect[1] + b.rect[3]) / 2
    mx, my = picker.MARGIN + (cx + 0.5) * size, picker.MARGIN + cy * size
    hit = picker.button_at(buttons, mx, my, size)
    hits.append(hit is b)
check(all(hits), "clicking the centre of each button hits that button")
check(picker.button_at(buttons, picker.MARGIN + 2, picker.MARGIN + 1.3 * size, size) is None
      and picker.inside_panel(picker.MARGIN + 2, picker.MARGIN + 1.3 * size, size),
      "empty picker area hits nothing but still belongs to the picker")
check(not picker.inside_panel(picker.MARGIN + 1.2 * size, picker.MARGIN + 10, size), "outside the panel")

pbs = rig.pose.bones
head, hand = by_key["head"].bones, by_key["hand.L"].bones  # FK hand: arms start in FK, IK hidden
picker.select(rig, head)
check(all(utils.is_pose_bone_selected(pbs[n]) for n in head) and picker.is_lit(rig, by_key["head"])
      and not picker.is_lit(rig, by_key["hand.L"]), "click selects the part and lights its button")
picker.select(rig, hand, 'ADD')
check(all(utils.is_pose_bone_selected(pbs[n]) for n in head + hand), "shift-click adds to the selection")
picker.select(rig, head, 'REMOVE')
check(not any(utils.is_pose_bone_selected(pbs[n]) for n in head)
      and all(utils.is_pose_bone_selected(pbs[n]) for n in hand), "ctrl-click removes from the selection")
picker.select(rig, by_key["chest"].bones)
check(not any(utils.is_pose_bone_selected(pbs[n]) for n in hand), "plain click replaces the selection")

# Controls in hidden bone collections are not selected
hand_bone = rig.data.bones[hand[0]]
for coll in hand_bone.collections:
    coll.is_visible = False
picker.select(rig, hand)
check(not utils.is_pose_bone_selected(pbs[hand[0]]), "hidden controls are skipped")
for coll in hand_bone.collections:
    coll.is_visible = True

# Toggling the picker
check(bpy.ops.rigger.viewport_picker() == {'FINISHED'} and picker.is_running(), "picker switched on")
bpy.ops.rigger.viewport_picker()
check(not picker.is_running(), "picker switched off")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
