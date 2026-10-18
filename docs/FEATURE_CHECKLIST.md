# Feature checklist

Every feature described in the reference documentation (Auto-Rig, Rig Features,
Game Engine Export, Remap, Quick Rig, FAQ and Updates Log pages), with its
status in Rigger. Rigger's code is original; only the feature list comes from
the reference.

Status: **done** (implemented and tested), **n/a** (does not apply, see the
note). The *Test* column names the test file that checks the feature
(`tests/<name>.py`). Everything in the sidebar is also checked by `test_ui`
(every property, button and list exists) and drawn in the real interface by
`gui_test` (Xvfb).

## Reference / Smart placement

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| Templates: human, human without fingers/face, quadruped/dog, horse, bird | done | test_generate | |
| Empty reference + add limbs one by one | done | test_pipeline | Add Module |
| Smart markers (neck, chin, shoulder, wrist, spine root, ankle) | done | test_pipeline, gui_test | markers placed with real mouse clicks in the GUI test |
| Eyeball / hand detection when fitting | done | test_features | |
| Mirror markers / X-mirror reference | done | test_pipeline | Mirror Left to Right |
| Turn buttons (rotate the model by 90°) | done | test_rig_tools | Turn -90 / +90 |
| Facial markers toggles (eyebrows, eyes, ears, nose, lips, tongue, teeth, chin) | done | test_body | Template Options face parts |
| Spine / neck / twist / finger counts, toes (template options) | done | test_body | |
| Metacarpals, toes metatarsal, toes parent foot | done | test_extras | |
| AI "Guess Markers" | done | test_ai_markers | DWPose whole-body model with onnxruntime, downloaded on request (Install AI Model) |
| AI "Guess Fingers" / "Guess Facial" | n/a | | the same model finds hand and face points; not wired yet |
| Voxel-centroid finger detection | done | test_rig_tools | Detect Fingers: slices of the hand clustered into fingers |
| Mac "Disable Smart FX" | n/a | | markers are plain empties already |
| Biped / Multi-ped setting (spine shape orientation) | done | test_body | on by default for the Quadruped and Horse templates |
| Duplicate limb / Duplicate Mirror / Disable limb | done | test_rig_tools | Duplicate Limb, Mirror, Remove Limb |
| Parent Fallback setting per limb | done | test_extras | |
| Show IK directions (pole line) | done | test_rig_tools | viewport overlay |
| Init Scale (apply rig object scale) | done | test_extras | |
| Armature presets: save selected armature as custom preset | done | test_rig_tools | |

## Limbs

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| Spine: count 1-64, torso, hips, FK spine, chest | done | test_features | |
| Spine master controller (stretch & squash) | done | test_body | |
| Spine master space: local / master | done | test_body | |
| Bottom (buttock) bones | done | test_body | |
| Reversed spine (switchable + snap) | done | test_body, test_rig_tools | |
| Align root master / bend controllers | n/a | | controls follow reference bones |
| Neck: count, twist bones, bendy bones, auto twist | done | test_body | |
| Head: head lock + snap head lock + bake | done | test_rig_tools, test_extras | head isolate + Snap Switch (bake option) |
| Skull controllers (3) | done | test_body | |
| Arm: FK/IK, switch, snap, pole, stretch, twist on forearm | done | test_pipeline | |
| Arm/thigh upper twist bones, multiple twist (up to 32) | done | test_limbs | |
| Tweak twist for arm/thigh | done | test_limbs | Secondary Controls: Twist |
| Arm FK lock-free / leg FK lock (snap + bake) | done | test_limbs, test_extras | FK Lock + Snap Switch |
| IK pole distance | done | test_limbs | |
| Soft IK | done | test_limbs | |
| Auto IK roll (free roll) | done | test_limbs | |
| Hand IK offset controller | done | test_limbs | |
| Hand / foot IK pivot controller | done | test_limbs | |
| Foot IK offset controller | done | test_limbs | |
| Foot roll cursor: heel, tip, ball, bank pivots | done | test_limbs | |
| Foot roll break (break point, speed, overshoot, unfold lag) | done | test_limbs | |
| Roll cursor distance / speed | done | test_limbs | `roll_speed` |
| Toes pivot controller | done | test_limbs | |
| Smart IK pole (pole steady when tilting foot) | done | test_limbs | |
| Pole parent snap (+ smart pole) | done | test_features | parent spaces |
| Elbow/knee pinning + Snap Pinning | done | test_tools | |
| Stretch length / auto stretch | done | test_tools, test_skin_export | |
| 3 bones leg (type 1 and 2) | done | test_limbs | 2/3-bone IK switch |
| Toes individual / metatarsal / parent foot / pinky auto | done | test_body, test_extras | |
| Toes IK-FK | done | test_extras | finger IK on toe chains |
| Toes rot from scale | done | test_limbs | |
| Fingers: FK + curl master | done | test_pipeline | |
| Fingers grasp (curl all) | done | test_limbs, test_extras | |
| Fingers rotate from scale | done | test_limbs | all or last two phalanges |
| Fingers auto-spread (pinky base) | done | test_limbs | |
| Fingers metacarpals + pinky auto (independent pinky option) | done | test_extras | |
| Fingers IK-FK (snap, snap all, toggle IK parents) | done | test_limbs, test_rig_tools | |
| Fingers shapes (box/circle) | done | test_limbs | |
| Joint fans (elbow/knee, count 1-32, bulge loc/scale) | done | test_limbs | |
| Ankle / wrist / thigh fans | done | test_limbs | Root Fans, End Fans |
| Wings: feathers per segment, subdivisions, layers | done | test_tools | |
| Wings fold controller | done | test_extras | plays an action named `rig_wings_fold` |
| Tail: count, master at root | done | test_pipeline | FK chain |
| Ears: count | done | test_body | FK chain |
| Spline IK: count, controls, stretch | done | test_tools | |
| Spline IK: IK-FK chain, twist, tail bone, smoothness | done | test_body, test_extras | snap IK/FK |
| Bendy-bones chain limb | done | test_body | |
| Kilt/skirt: strands, follow legs | done | test_tools | |
| Kilt: master columns/rows, collide with legs, collision distance | done | test_body | both legs collide |
| Secondary controllers: Bendy | done | test_tools | |
| Secondary controllers: Twist (exportable) / None | done | test_limbs | |
| Secondary controllers: Additive | n/a | | legacy mode in the reference |

## Face

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| Jaw, lips, corners, eyelids, blink, brows, cheeks, nose, teeth, tongue | done | test_features | |
| Eye targets (per-eye + master) | done | test_features | |
| Eye aim IK/FK switch, target follows head toggle | done | test_body | aim slider, Head / Root space |
| Eye target distance | done | test_extras | |
| Auto-eyelids amount (upper/lower) | done | test_body | |
| Eyelid tweak, amount, masters | done | test_body | eyelid bone count, Eyelid Masters |
| Eyebrow masters, soft eyebrows, middle eyebrow | done | test_body, test_extras | |
| Lips offset controller | done | test_body | |
| Muzzle controller | done | test_body | |
| Lips roll | done | test_body | |
| Lips amount / masters / soft lips / linear settings | done | test_body, test_extras | |
| Sticky lips / lips retain / lips follow | done | test_body | |
| Soft cheeks | done | test_body | |
| Jaw: rotate from translation, unlock Y loc, jaw speed | done | test_body, test_extras | |

## Rig UI and tools

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| Bone collections (main, secondary, deform, mechanism, colors) | done | test_extras | |
| Edit shape + mirror | done | test_tools | |
| Apply shape transforms | done | test_rig_tools | |
| Set custom shape from library | done | test_rig_tools | |
| Import/export custom shapes | done | test_rig_tools | |
| Picker panel | done | test_features, test_picker, gui_test | sidebar buttons and a clickable body diagram in the viewport |
| Color theme (left/middle/right, assign) | done | test_rig_tools | |
| Import/export rig data (reference, options, shapes) | done | test_rig_tools | |
| Save/load custom limbs (custom bones) | done | test_rig_tools | |
| Set character name (rig, collections) | done | test_rig_tools | |
| Rig layers (bones + objects) | done | test_tools | |
| Rig layers: animated, select toggle, import/export | done | test_rig_tools | |
| Bone custom properties shown in panel + pin | done | test_ui, gui_test | Animate panel |
| IK-FK switch + snap | done | test_pipeline, gui_test | |
| Bake IK>FK / FK>IK over a frame range | done | test_rig_tools | |
| Snap Child Of (ChildOf switcher) + bake | done | test_rig_tools, test_extras | |
| Extract root motion | done | test_rig_tools | |
| Mirror pose | done | test_rig_tools, gui_test | |
| Reset pose | done | test_pipeline | |
| Quaternion/Euler conversion (bake) | done | test_rig_tools | |
| Hand fist pose (add, default, mirror, extend) | done | test_rig_tools, gui_test | pose blends |
| Blink pose (add, default, in-between, mirror) | done | test_rig_tools | |
| Set pose (A-pose, A-pose UE5, T-pose) | done | test_rig_tools, gui_test | |
| Apply pose as rest pose | done | test_tools | |
| Apply pose as rest pose: preserve animation | done | test_rig_tools | |

## Skin

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| Heat bind, proximity fallback, rigid small parts, auto rescale | done | test_pipeline, test_features | |
| Bind options dialog | done | gui_test | confirmed with Enter in the GUI test |
| Voxelized engine | done | test_skin_export | |
| Voxel Heat Diffuse (third-party) | n/a | | separate add-on |
| Optimize high res | done | test_skin_export | decimated proxy |
| Split parts | done | test_skin_export | |
| Scale fix | done | test_skin_export | |
| Selected bones only / selected vertices only | done | test_skin_export | |
| Smooth twist weights | done | test_skin_export | |
| Improve hips / heel weights, refine head weights | done | test_skin_export | |
| Facial features (eyeball/teeth/tongue objects) | done | test_skin_export, gui_test | |
| Eyelid borders (Set Left / Set Right) | done | test_extras | |
| Apply shape keys before binding | done | test_skin_export | |
| Keep locked / modifier-used vertex groups | done | test_skin_export | |
| Preserve volume toggle, parent objects toggle | done | test_skin_export | |
| Corrective shape keys | done | test_tools | |
| Corrective driver: remove | done | test_rig_tools | |
| Shape key driver tool (bone transform → shape key, 0/1/reset) | done | test_rig_tools | |
| Mirror shape keys with drivers | done | test_rig_tools | |
| Warnings: delta transforms, negative scale | done | test_skin_export | |

## Game engine export

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| FBX: root + deform skeleton, baked actions, Unreal/Unity/generic | done | test_pipeline | |
| Unreal Mannequin / Unity Humanoid names | done | test_features | |
| Root motion | done | test_features | |
| Godot engine type, Godot names, Godot root axes | done | test_skin_export | |
| GLTF/GLB export | done | test_skin_export, gui_test | re-imported in the test |
| Humanoid vs Universal type, full/basic facial, metacarpals | done | test_skin_export, test_extras | |
| Custom bones (cc_ prefix / property / button) | done | test_skin_export | |
| Exclude tag | done | test_skin_export | |
| Soft-link bones | done | test_skin_export | |
| Constant interpolation bones | done | test_skin_export | |
| Check rig / Fix rig | done | test_skin_export | |
| No parents | done | test_skin_export | |
| Export twist / twist amount | done | test_skin_export, test_limbs | twist amount: per-limb Twist Amount |
| Rename bones from file / text block | done | test_skin_export | |
| Custom export script | done | test_skin_export | |
| Rig name | done | test_skin_export | |
| Units x100 | done | test_skin_export | |
| UE IK bones | done | test_skin_export | |
| Selected objects only / selected bones only | done | test_skin_export | |
| Actions: only active, list with checkboxes, search, select all | done | test_skin_export | |
| Actions manager (lists) / actions linker | done | test_skin_export | |
| NLA export | done | test_skin_export | |
| Frame range: action, manual, markers (start/end) | done | test_skin_export | |
| Action names format (separator, only action name) | done | test_skin_export | |
| One file per action / per actions list | done | test_skin_export | |
| Simplify factor, sampling rate, reset transforms between actions | done | test_skin_export | |
| Smoothing type, triangulate, apply modifiers/subsurf, vertex colors | done | test_skin_export | |
| Add dummy mesh | done | test_skin_export | |
| Force rest pose export | done | test_extras | helper mesh bound to every bone |
| Initialize armature/mesh rotation, bake axis conversion, bone axes | done | test_skin_export | Convert Axes, bone axes |
| Embed textures | done | test_skin_export | |
| Export presets, quick export | done | test_skin_export, gui_test | |
| Export file browser with the settings | done | gui_test | extension follows the format |
| Export by script API | done | test_skin_export | `scene.rigger_export` + `bpy.ops.rigger.export_fbx` |
| Fix Rotation / Fix Matrices / Push Additive | n/a | | workarounds for the reference's own bake and additive mode |

## Remap

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| Source/target, auto bone list, presets (Mixamo, UE, Rigify, BVH, Biped) | done | test_pipeline | |
| IK feet/hands with poles, root location | done | test_pipeline | |
| Redefine rest pose (use target pose / source frame) | done | test_remap_quick | |
| Per-bone rotation/location offsets | done | test_remap_quick | |
| Auto-scale | done | test_pipeline | height-based scale |
| In place | done | test_remap_quick | |
| Multiple source animations (+ filter) | done | test_remap_quick | |
| Replace namespace | done | test_remap_quick | |
| Location (local) per bone + location multiplier | done | test_remap_quick | |
| IK pole modes (absolute / relative target / relative chain) | done | test_remap_quick | |
| Mirror bones list | done | test_remap_quick | |
| Select bone from viewport in list | done | test_remap_quick | |
| Clean FK rotation, clean IK poles | done | test_remap_quick | |
| Fake user, copy frame range and cyclic | done | test_remap_quick | |
| Bake only existing keyframes | done | test_remap_quick | |
| Extract root motion while retargeting | done | test_remap_quick | |
| Bind / unbind only (preview) | done | test_remap_quick, gui_test | frames changed with the arrow keys in the GUI test |
| Load bone map without clearing the list | done | test_remap_quick | |

## Quick Rig

| Feature | Status | Test | Notes |
| --- | --- | --- | --- |
| Add limbs from an existing skeleton (auto-guess bones) | done | test_remap_quick, gui_test | Mixamo and Unreal names |
| Limbs definition by selection, 3 or 4 finger bones (metacarpals) | done | test_remap_quick, test_extras | |
| Convert mode (rename vertex groups) | done | test_remap_quick | |
| Preserve mode (constrain original skeleton) | done | test_remap_quick | |
| Bake existing animation to the control rig | done | test_remap_quick, gui_test | |
| Orphan bones, ignore root bone, same origin, hide base armature | done | test_remap_quick | |
| Weights overrides | done | test_remap_quick | |
| Presets import/export | done | test_remap_quick | |

## Not applicable

Smart FX on Mac, Check for Updates, Lite version, Legacy menu,
proxy linking (Blender 2.9x), Update Armature for old rigs, purchase/download
pages, picker add-on camera (Rigger's picker draws in the 3D view instead), the
reference's own bake workarounds (Fix Rotation, Fix Matrices) and its legacy
additive secondary mode.
