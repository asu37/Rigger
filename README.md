# Rigger

Modular auto-rigging for Blender 4.2+ (tested on Blender 5.2). Place a
reference skeleton, generate an animator-friendly control rig, skin the mesh,
retarget mocap/Mixamo animation and export a clean skeleton to game engines.

Rigger is original code released under the **GNU GPL v3 or later** (see
`LICENSE`), like Blender itself. You may sell it; buyers receive the same GPL
rights to the code.

## Features

| Step | What it does |
| --- | --- |
| **Reference** | Templates (Human with face, Human without fingers/face, Quadruped, Horse, Bird) with options (spine/neck/twist bone counts, fingers, metacarpals, toes, face parts). **Smart markers**: click 6 landmarks on the mesh (or let the optional AI pose model guess them) and the reference fits itself: elbows, knees, hips, spine and toes are placed from the mesh shape, eyes and fingers detected. Turn model, armature presets, add/duplicate/mirror/remove limbs, IK direction lines. |
| **Modules** | Spine, Neck & Head, Arm, Leg, Finger, FK Chain, Bendy Bones, **Spline IK**, **Skirt**, **Face**, Eye, Simple bone, plus your own **custom limbs**. |
| **Quick Rig** | Put a control rig on an existing skeleton (Mixamo, Unreal, any armature): limbs guessed from bone names, **Preserve** (skeleton follows the rig) or **Convert** (weights renamed) mode, orphan bones, weights overrides, animation baked to the controls, presets. |
| **Generate** | Deform (`DEF-`), mechanism (`MCH-`) and control bones, bone collections, colour themes, custom shapes. Regenerating updates the same rig object and keeps skinning, animation, custom bones, shapes and pose blends. Rig data import/export, character name. |
| **Arms & legs** | FK/IK with snapping and baking, poles (distance, smart pole, pinning), stretch, soft IK, auto IK roll, FK lock, IK offset and pivot controls, reverse foot (heel, tip, ball, bank, roll break), toes pivot, 3-bone legs, twist bones on both segments (up to 32), secondary controls (bendy or twist), joint fans, wings fold. |
| **Spine & head** | Torso, hips, FK spine, chest, spine master (stretch & squash), bottom bones, reversed spine, multi-ped torso, neck twist, bendy neck, skull controls, head isolation. |
| **Face** | Jaw (rotate or translate, Y unlock), lips (count, soft lips, roll, sticky lips, offset, muzzle, masters), eyelids (blink, auto lids, masters), brows (masters, soft), cheeks, nose, teeth, tongue; eye aim with parent spaces. |
| **Fingers & toes** | FK, curl master, grasp, rotate from scale, auto spread, finger IK with snapping, metacarpals with pinky auto, individual toes with metatarsals. |
| **Rig tools** | Pose blends (fist, extended hand, blink with in-betweens), A/T poses, mirror pose, apply pose as rest pose (animation kept), rotation mode conversion, root motion extraction, Child Of switcher, snap switches, shape editor and library, corrective shape keys, shape key drivers, rig layers (animated), pinned properties, sidebar picker and a clickable viewport picker. |
| **Skin** | Heat + fallback, **voxelized**, heat or proximity binding; split parts, decimated proxy, shape key mix, scale fix, selected bones/vertices only, locked groups kept, twist/hips/heel/head weight corrections, eyeballs/teeth/tongue objects, eyelid borders, preserve volume and parent toggles. |
| **Retarget** | Auto bone mapping (Mixamo, Unreal, Rigify, BVH/CMU, Biped), world-space transfer, IK with three pole modes, in place, multiple source actions, namespace replacement, local/world location with multiplier, mirror list, live bind preview, clean FK/poles, root motion, presets. |
| **Video to Animation** | Markerless motion capture from a video of a person, on your computer: body, fingers, face shape keys (ARKit names) and root motion, solved on the character's own skeleton, feet planted, matching camera with the video behind; runs in the background, cached, reusable `.mcap` motion files. |
| **Export** | FBX or glTF with root + deform skeleton, Universal or Humanoid, Rigger / Unreal / Unity / Godot names, UE IK bones, custom/exclude/soft-link/constant tags, no parents, rename file, custom script, x100 units, action lists and manager, actions linker, NLA, frame ranges and markers, one file per action, dummy mesh, forced rest pose, presets, quick export, check/fix rig. Your scene is left untouched. |

## Install

1. `python build.py` creates `dist/rigger-1.2.0.zip`.
2. In Blender: *Edit > Preferences > Get Extensions > ⌄ > Install from Disk…* and pick the zip.
3. Open the sidebar in the 3D Viewport (`N`) and choose the **Rigger** tab.

## Quick start

1. **Reference**: click *Human*, or select your mesh, go to Front view (`Numpad 1`)
   and place the six **Smart Markers**, then click *Fit Reference to Markers*.
   Tweak the bones in Edit Mode if needed. Already have a skeleton? Use
   **Quick Rig** instead of steps 1-3.
2. **Generate**: click *Generate Rig*.
3. **Skin**: select the mesh(es), then shift-select the rig and click *Bind*.
4. **Animate**: use the controls. The *Animate* panel holds IK/FK switching and
   baking, the selected control's properties, pins, rig layers and bone
   collections. *Rig Tools* has poses, pose blends, shapes and shape keys.
5. **Retarget** (optional): set the animated armature as *Source* and your rig as
   *Target*, click *Auto Map*, check the list and click *Retarget*.
6. **Export**: *File > Export > Rigger Game Rig* or the *Export* panel (FBX, glTF).

The full manual is in [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Development

Tests run headless with the `bpy` module (`pip install bpy`) or Blender itself:

```
python -I tests/test_generate.py
python -I tests/test_behavior.py
python -I tests/test_pipeline.py      # full pipeline: markers → rig → skin → retarget → FBX
python -I tests/test_features.py      # face, chest, parent spaces, picker, skin extras, export names
python -I tests/test_tools.py         # secondary controls, fans, pinning, spline IK, skirt, layers, tools
python -I tests/test_limbs.py         # reverse foot, pivots, offsets, twist, FK lock, soft IK, fingers
python -I tests/test_body.py          # spine, neck, face, eyes, spline IK and skirt options, templates
python -I tests/test_rig_tools.py     # pose blends, poses, root motion, rig data, shapes, shape keys
python -I tests/test_skin_export.py   # binding options, every export option (FBX and glTF re-imported)
python -I tests/test_remap_quick.py   # retarget options and Quick Rig (Preserve and Convert)
python -I tests/test_extras.py        # metacarpals, metatarsals, jaw Y, eyelid borders, rest pose export
python -I tests/test_ui.py            # every panel, list and button checked against the real properties
python -I tests/test_fitting.py       # joints fitted to a synthetic body with known elbow, knee and hips
python -I tests/test_safety.py        # operator safety net and the checks shown before binding
python -I tests/test_picker.py        # viewport picker layout, hit testing and selection
python -I tests/test_ai_markers.py    # AI markers mapping; with RIGGER_AI_TEST_DIR also a real body
```

With Blender itself: `blender -b --factory-startup --python-exit-code 1 --python tests/<test>.py`.

The GUI test drives the real interface on an X display (Xvfb works): it places
the markers with mouse clicks, confirms dialogs and the file browser with the
keyboard, scrubs frames with the arrow keys, screenshots every panel and reports
any panel draw error in `report.txt`.

```
Xvfb :99 -screen 0 1600x1000x24 &
DISPLAY=:99 RIGGER_SHOTS=/tmp/shots blender --factory-startup --python tests/gui_test.py
```

The deformation benchmark rigs four real human bodies (CC0), poses them and
measures joint volume loss, skin stretch, twist spread and weight bleed. See
[`bench/README.md`](bench/README.md).

```
python bench/fetch_assets.py
blender -b --factory-startup --python bench/run_bench.py -- --render
```
