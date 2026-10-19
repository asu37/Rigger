# Rigger User Guide

All tools live in the 3D Viewport sidebar (`N`) under the **Rigger** tab. The
panels follow the workflow:

1. **Reference**: describe the character's skeleton (with *Template Options*).
2. **Quick Rig**: or start from a skeleton you already have.
3. **Generate**: build the control rig.
4. **Skin**: bind the meshes.
5. **Animate**, **Picker**, **Rig Tools**: animate.
6. **Retarget**: bring in animation from other skeletons, or **Video to Animation** from a video.
7. **Export**: write a clean game skeleton (FBX, glTF).

Characters should face **-Y** (Blender's front view) with **Z up**, and stand
on the ground at Z = 0. Apply the mesh's rotation and scale (`Ctrl A`) before
rigging, or use **Turn Model** if the character faces another way.

---

## 1. Reference

The reference is a plain armature that describes your character's proportions.
Each **module** starts on one bone and continues through its child bones.

### Templates and presets

| Template | Contents |
| --- | --- |
| Human | Spine, neck, head, face (jaw, lips, lids, brows, cheeks, nose, teeth, tongue), eyes, arms with shoulders, fingers, legs |
| Human (No Fingers/Face) | Same body without fingers, jaw and eyes, for game characters |
| Quadruped, Horse | Horizontal spine (Multi-ped torso), neck, head, jaw, eyes, tail, front and hind legs |
| Bird | Spine, long neck, beak, wings with feathers, legs with toes |

**Template Options** (sub-panel) apply to the Human templates and to *Fit
Reference to Markers*:

| Option | |
| --- | --- |
| Spine Bones, Neck Bones | UE4 mannequin: 3 and 1. UE5: 5 and 2 |
| Twist Bones | Twist bones on the forearm/shin (the upper arm/thigh get one less) |
| Fingers | 1 to 5 fingers per hand |
| Metacarpals | Palm bones before each finger (not the thumb) |
| Individual Toes, Toes Metatarsal, Toes Parent Foot | Five toes per foot, optional metatarsal bones, toes following the foot instead of the main toe bone |
| Face: Lips per Side, Eyelid Bones, Eyebrow Bones | Number of bones |
| Face parts | Eyes, eyebrows, nose, lips, tongue, teeth, chin, cheeks, ears on/off |

**From Preset** adds a reference saved with **Save Armature Preset** (your own
creatures). The trash icon deletes a preset.

The reference is added at the 3D cursor and opens in Edit Mode. Move the bones
to match your mesh. X-Axis Mirror (in the Edit Mode header) keeps both sides
symmetrical, or use **Mirror Left to Right** at any time.

### Smart Markers

1. Select the character mesh and switch to Front view (`Numpad 1`). **Turn
   -90 / +90** rotates the selected meshes if the character doesn't face -Y.
2. Click each marker button, then click on the mesh:
   * **Chin**: bottom of the chin
   * **Neck**: base of the neck
   * **Shoulder**: the character's left shoulder joint (screen right in front view)
   * **Wrist**: the character's left wrist
   * **Spine Root**: pelvis
   * **Ankle**: the character's left ankle
3. Click **Fit Reference to Markers**. Eyes and fingers are found on the mesh.

Markers sit in the middle of the mesh volume under the cursor, not on its
surface. Centre markers snap to X = 0 and side markers are mirrored. Mesh height
sets the floor and top of the head. Hold the middle mouse button to orbit or pan
while placing. Right-click or `Esc` cancels.

**Guess Markers (AI)** places the six markers for you: Rigger renders the
selected mesh from the front, a pose estimation model finds the joints in the
image, and each marker goes to the middle of the mesh behind the joint, as if
clicked. Check the markers (a warning names any the model was unsure of), move
them if needed, then click *Fit Reference to Markers*. The model (DWPose,
Apache-2.0, about 135 MB) and onnxruntime (about 15 MB) are not part of the
add-on: click the download button next to *Guess Markers* or *Install AI Model*
in the add-on preferences, with online access allowed in Blender. On the four
benchmark bodies the guessed markers land about 2 cm from hand-checked
positions (6 cm at most).

With **Fit Joints to Mesh** (on by default, in the operator's redo panel), the
fit also reads the mesh between the markers: it traces the centre of each arm
and leg and places the elbow at the arm's bend (or its thinnest point when the
arm is straight), the knee where the leg narrows between thigh and calf, the
hip joints on the thigh's centre line, the spine joints toward the back of the
torso and the toes at the real front of the foot. Where the mesh gives no clear
answer (a robe, a blocky prop) the standard proportions are used.

**Detect Fingers** (with the reference active and the mesh selected) slices the
hand, finds each finger and places the finger bones on it, mirrored to the
right hand.

### Modules

Select a bone (Edit or Pose Mode) to see its module settings. **Add Module**
inserts a pre-shaped module at the 3D cursor, or at the tail of the active bone.
**Duplicate Limb** copies the active bone's module (**Mirror**: to the other
side), **Remove Limb** deletes it (the button with the X). The kinematics icon
shows the **IK directions**: lines from each elbow/knee to where its pole will be.

| Module | Chain | Main options |
| --- | --- | --- |
| Spine | hips, spine bones… | Spine Master (stretch & squash, local or master space), Bottom Bones, Reversed Spine |
| Neck & Head | neck bones…, head (last bone) | Neck Twist, Neck Bendy Segments, Skull Controls |
| Arm | [shoulder], upper arm, forearm, hand | See *Limb options*, Wings Fold Control |
| Leg | thigh, shin, foot, [toe] | See *Limb options*, Foot Roll Break, Toes Pivot, 3 Bones Leg |
| Finger | any number of segments | Curl master, IK-FK with pole, box or circle shapes, bone collection |
| FK Chain | any number of bones | Curl master, bone collection |
| Bendy Bones | any number of bones | One control per joint shaping a smooth bendy chain |
| Spline IK | any number of bones | Controls, stretch, smoothness, IK-FK chain, twist, tail bone |
| Skirt | strands around the hips | Collide with legs (distance), column masters (frequency), row masters |
| Face | a `face` root bone and every bone below it | See *Face module* |
| Eye | one bone pointing forward | Eye Target Distance |
| Simple | one bone | Bone collection |

Every module also has **Parent Fallback**: the rig bone it follows when its
first reference bone has no parent (default `root`).

A chain continues through child bones whose module is *None*. It stops at a
child that starts its own module (fingers stop the arm chain at the hand).
Bones not covered by any module become *Simple* FK bones automatically.

#### Limb options (arms and legs)

| Option | |
| --- | --- |
| Twist Bones, Upper Twist Bones, Twist Amount | Twist bones on the lower and upper segments (up to 32, default 2 and 1). The ones near the shoulder/hip twist less |
| Start in IK | Initial IK/FK switch value |
| IK Pole Distance | How far the pole sits from the elbow/knee |
| Auto IK Roll | Align the bone rolls to the bend plane (off: keep your rolls) |
| FK Lock Setting | `fk_lock` slider: the arm/leg follows the body (1) or keeps its world rotation (0) |
| Soft IK, Softness | No pop near full extension (slight stretch) |
| IK Offset Control, IK Pivot Control | A second hand/foot control under the main one, and a movable pivot |
| Smart IK Pole | Pole space following the foot's position and heading only |
| Secondary Controls (Bendy Bones or Twist), Segments, Control Scale | Tweak controls that bend and squash the limb. *Twist* mode uses exportable twist bones instead of bendy bones. In bendy mode the elbow/knee stays sharp; scale its joint control along Y (up to 2) to round it |
| Joint Fans, Root Fans, End Fans, Fans per Joint, Bulge | Extra deform bones keeping volume at the elbow/knee (on by default), shoulder/hip and wrist/ankle |
| Rotate Digits from Scale | Scaling a finger's first bone curls it (all phalanges or the last two) |
| Auto Spread | Rotating the pinky sideways spreads the other fingers |
| Pinky Auto | Rotating the pinky metacarpal also turns the ring and middle ones (off: independent pinky) |

Legs have a **reverse foot**: `foot_ik.L` with heel, tip, ball and bank pivots
driven by the **roll cursor** `foot_roll.L` behind the heel (move it forward and
back to roll, sideways to bank). The `roll_speed` property sets how fast it
rolls. **Foot Roll Break** adds `roll_break` (degrees where the tip takes over),
`roll_overshoot` and `roll_unfold`. **3 Bones Leg** adds a bone at the leg's root
for digitigrade legs, with a `three_ik` slider.

### Face module

Every bone under the face root becomes a control. Its name sets how it behaves:

| Bone name starts with | Behaviour |
| --- | --- |
| `jaw` | Jaw control. Bones parented to it in the reference (lower lip, chin, lower teeth, tongue) follow it |
| `lip_corner.L/.R` | Follows the jaw halfway |
| `lip_upper…`, `lip_lower…` | Follow the lip corners and the jaw (smoothly with Soft Lips) |
| `lid_upper…`, `lid_lower…` | Pivot around the eye (needs an `eye.L/.R` Eye module) and follow it (`auto_lid_upper/lower` amounts). With both lids, the face control gets `blink_L` / `blink_R` sliders |
| `brow…` | Eyebrow controls |
| `tongue…` | FK chain |
| anything else | Free tweak control (cheeks, nose, teeth…) |

Face options: **Eyelid Masters**, **Eyebrow Masters** with **Soft Eyebrows**,
**Soft Cheeks**, **Lips Offset Control**, **Muzzle Control** (animals), **Sticky
Lips**, **Lips Roll**, **Soft Lips** (with a Linear falloff setting), **Lips
Masters Frequency**, **Translate to Rotate Jaw** (moving the jaw control down
opens the jaw, `jaw_speed` property) and **Unlock Jaw Y Loc** (sliding the jaw
forward and back).

**Bone roll**: for arms and legs, Rigger computes rolls from the bend of the
limb, so give elbows and knees a slight bend in the reference (elbows bend back,
knees bend forward).

---

## 2. Quick Rig

Puts a Rigger control rig on a skeleton you already have (Mixamo, a game rig,
any armature). The original bone names and weights stay usable.

1. Pick the **Armature** and click **Guess Limbs**: spine, neck & head, arms,
   legs and fingers are found from the bone names (Mixamo, Unreal, Unity,
   Rigify, Biped… names). Metacarpals are detected too.
2. Check the list. For each limb: type, side, name and the source bones from
   root to tip. **Set From Selected Bones** fills the active limb from the bones
   you selected on the armature. Add limbs (tails, ears) with **+**.
3. Choose a **Mode**:
   * **Preserve**: the skeleton keeps its names and weights and follows the
     rig's deform bones (through hidden `QR-` helper bones on the rig). Export
     the original skeleton as before. **Hide Base Armature** hides it.
   * **Convert**: the meshes' vertex groups are renamed to the rig's deform
     bones and moved to the rig. **Weights Overrides** send a vertex group to
     another bone (for example a twist bone's weights to the forearm). Groups of
     bones in no limb go to their closest mapped parent. **Delete Source
     Armature** removes the old skeleton.
4. Options: **Add Orphan Bones** (bones in no limb become single-bone
   controls), **Ignore Root Bone**, **Same Origin** (rig at the armature's
   location and rotation), **Bake Animation** (the armature's action is
   transferred to the controls), **Twist Bones**, **Secondary Controls**,
   **Finger IK**.
5. Click **Quick Rig**.

Presets (import/export icons) store the limb list, overrides and options as
JSON.

---

## 3. Generate

| Setting | Effect |
| --- | --- |
| Rig Name | Name of the generated armature object |
| Control Size | Scale of all control shapes |
| Auto-Hide IK/FK | Hide the inactive chain's controls |
| Init Scale | Apply the reference's object scale first, so the rig has a scale of 1 |
| Multi-ped | The torso control follows the spine direction (quadrupeds) |
| Use Color Theme | Left, middle and right colours. **Assign** colours the controls, **Reset** goes back to the defaults |

**Generate Rig** creates the rig and hides the reference. Clicking it again
(**Update Rig**) rebuilds the same object. Meshes bound to it, actions, stored
pose blends, custom shapes, colours and your own custom bones are kept. **Edit
Reference** brings the reference back.

**Export Data / Import Data** save the reference bones, module settings, control
shapes, pose blends and rig layers to a file and load them into another
character. **Set Character Name** renames the rig and puts it, its meshes and its
shapes in `<name>`, `<name>_rig` and `<name>_cs` collections.

### Bone naming

| Prefix | Purpose |
| --- | --- |
| `DEF-` | Deform bones. The only bones that move the mesh, in a clean hierarchy under `root` |
| `MCH-` | Mechanism bones (IK chains, sockets), hidden |
| none | Animation controls |

### Bone collections

Root, Torso, Head, Face, Arm IK, Arm FK, Leg IK, Leg FK, Fingers, Tweak and Extra
are visible. Deform and Mechanism are hidden. Toggle them in **Animate**.

---

## 4. Skin

Select the mesh(es), then shift-select the rig, and click one of the **Bind**
buttons. A dialog shows every option before binding. Its top lists anything
worth fixing first: holes in the mesh, a non-uniform or negative scale, delta
transforms, an Armature modifier for another rig, a mesh that doesn't overlap
the rig, or a very dense mesh.

| Engine | |
| --- | --- |
| Heat + Fallback | Blender's bone heat weighting. Vertices heat misses get proximity weights |
| Voxelized | Heat weighting on a watertight voxel copy, transferred back. For layered clothes and messy, non-manifold meshes (*Voxel Resolution*) |
| Heat | Heat weighting only |
| Proximity | Weights from the distance to each deform bone. Works on any mesh |

| Option | |
| --- | --- |
| Split Parts | Each loose part is weighted on its own (on by default: characters built from separate pieces deform far better, single meshes are unchanged) |
| Optimize High Res | Meshes with more faces than this are weighted on a decimated copy (0 = off) |
| Apply Shape Keys | Weight the current shape key mix |
| Scale Fix | Run heat weighting on a rescaled copy (tiny or huge characters; automatic under 0.5 m or over 5 m) |
| Max Influences, Falloff | Proximity weights |
| Rigid Small Parts, Small Part Size | Small loose parts (eyes, buttons, a prop in the hand) move rigidly with the bone that already holds them. Larger pieces and pieces that cross a joint (a mannequin's knee cap) keep blended weights |
| Selected Bones Only | Only the selected deform bones get weights (clothes bones…) |
| Selected Vertices Only | Only the selected vertices are changed |
| Smooth Twist Weights | Gradient along the twist bones |
| Improve Hips Weights | Thighs don't climb into the hips. Off by default: the cut creases the groin and stretches the buttock when the leg lifts |
| Improve Heel Weights | Heels follow the foot. Off by default: it stretches the ankle when the foot points down |
| Refine Head Weights | Everything above the base of the head follows the head, and the head and shoulders stop sharing weights (on by default) |
| Tighten Fan Weights | Joint fans only weigh the skin right around their joint (on by default). Heat weighting alone lets them reach along the limb, which makes bent joints collapse |
| Clean Finger Weights | Fingers stop sharing weights with other fingers and, past the first segment, with the hand (on by default) |
| Smooth Weights, Limit Influences | Softer weights, maximum bones per vertex (4 for most engines) |
| Facial Features | Eyeball, teeth and tongue objects (comma separated names) bound to the eyes, jaw / upper teeth and tongue bones |
| Preserve Volume, Parent to Rig | Armature modifier option and parenting |
| Shoulder Volume | How much the shoulders preserve volume (default 0.3, 1 = like the rest of the body). Full volume preservation balloons the shoulder when the arm is raised; a second Armature modifier mixes it per vertex (vertex group `rigger_volume`) |

Locked vertex groups (lock icon) are never changed, and groups that are not
bone names (used by modifiers) are kept. **Eyelid Borders**: in Edit Mode, select
the vertices around the left eyelid and click **Set Left** (same for the right).
When binding, they follow the upper or lower eyelid bones fully.

Binding warns about delta transforms and negative scales. All methods keep left
and right separate. **Unbind** removes the armature modifier, the parent and
(optionally) the weights. **Fit Shapes to Mesh** sizes the controls to the body.

---

## 5. Animate

| Control | Use |
| --- | --- |
| `root` | Moves the whole character. Used as root motion when exported |
| `torso` | Moves and rotates the body from the pelvis |
| `hips` | Rotates the pelvis around the waist |
| `spine.01…`, `chest` | FK spine. Rotating the chest bends the whole spine |
| `spine_master` | (option) Moves, rotates and stretches the whole spine |
| `neck`, `head`, `skull_01…03` | FK neck and head, optional skull controls |
| `shoulder.L` | Clavicle |
| `upper_arm_fk.L`, `forearm_fk.L`, `hand_fk.L` | FK arm |
| `hand_ik.L`, `upper_arm_pole.L` | IK arm and elbow pole (`hand_ik_offset.L`, `hand_ik_pivot.L` with the options) |
| `foot_ik.L`, `thigh_pole.L`, `foot_roll.L`, `foot_ball.L` | IK leg, knee pole, roll cursor, ball rotation |
| `toe_ik.L` / `toe_fk.L` | Toes |
| `index_master.L`, `index_ik.L`… | Finger curl and finger IK |
| `eyes_target`, `eye_target.L` | Eye aim |
| `face`, `jaw`, `lip_*`, `lid_*`, `brow_*`… | Face controls |

The **Animate** panel shows:

* **Pinned** properties. Click the pin next to any property to always show it.
* The **limb of the selected control**: *Switch to IK / FK* (snaps without
  moving the limb, keyed with auto-keying), **Bake to IK / Bake to FK** over the
  scene's frame range, the IK-FK blend, stretch, soft IK, 3-bone IK, roll
  settings, pinning (**Snap Pinning** moves the pole onto the elbow/knee), the
  parent spaces of the IK control and the pole, and FK Lock.
* **Spine**: reversed spine switch and **Snap Reversed Spine**.
* **Head & Face**: head isolate (head lock), neck twist, auto eyelids, blink,
  jaw speed, sticky lips…
* **Fingers**: fist/open, finger **IK / FK** snapping per hand and **IK
  Parents** (free the finger IK controls from the hand without moving them).
  Spline IK chains have a **Snap** button.
* Every property of the **active control**, its parent space menu and **Snap
  Switch**, which changes a setting (head lock, FK lock…) without moving the
  controls.
* **Rig Layers**: named layers that show or hide bones and objects together.
  Assign the selected bones (Pose Mode) or objects (Object Mode), select a
  layer's bones, export/import layers. **Animated Layers** plays keyed layer
  visibility back.
* Bone collection toggles and **Reset Pose**.

**Parent Spaces**: IK hands follow *Root*, *Torso*, *Chest* or *Head*, IK feet
*Root* or *Torso*, eye targets *Head* or *Root*. Switching keeps the control
in place.

**Picker** (Pose Mode): buttons that select the controls of a body part
(Shift-click adds to the selection).

**Viewport Picker**: the button at the top of the Picker panel draws a
clickable body diagram in the bottom-left corner of every 3D view (Pose Mode).
Click a part to select its controls, Shift-click to add them, Ctrl-click to
remove them. Parts light up when their controls are selected, hovering shows
the part's name, and parts whose controls are hidden (the IK hand while the arm
is in FK, for example) are faded. *Size* scales it. It is drawn by the add-on,
so nothing is added to the rig, and clicks on it never reach the viewport.

### Rig Tools

| Section | Tools |
| --- | --- |
| Pose | **A-Pose / A UE5 / T-Pose**, **Mirror Pose**, **Apply Pose as Rest Pose** (option: keep the animation looking the same), **Convert Rotation** (quaternion/Euler, keys converted), **Extract Root Motion** (root baked along the pelvis on the ground), **Snap Child Of** (switch between Child Of constraints without a jump, optionally baked) |
| Pose Blends | Pose the fingers and click **Fist** or **Extend**: scaling the hand's fist control down or up blends into that pose. **Blink**: pose the eyelids closed (or half-closed for an in-between) and store it for the blink slider. **Default Blink** poses the lids like the built-in blink. Stored blends can be mirrored or removed |
| Control Shapes | **Edit Shape** (then **Apply Shape**, mirrored automatically), **Set Shape** from the library, **Apply Shape Transforms**, export/import shapes, fit shapes |
| Shape Keys | **Add Corrective Shape Key** (driven by a joint's angle), **Create Shape Key Driver** (a bone's location/rotation/scale drives the active shape key), **Set Driver Key 0 / 1 / Reset**, **Remove Shape Key Driver**, **Mirror All Shape Keys** (with mirrored drivers) |
| Custom Limbs | Add your own bones to the rig (names starting with `cc_` or any bone you add), **Save Selected** as a custom limb with its shapes, constraints and drivers, and add it to other rigs. Custom bones survive regeneration |
| Export Tags | Tag the selected bones: **Custom** (exported with the skeleton), **Exclude**, **Soft-Link** (exported without scale) or **Constant** (constant interpolation) |

---

## 6. Retarget

Moves animation from any armature onto your rig (or onto any armature).

1. Import the animation (FBX from Mixamo, BVH, a game skeleton…).
2. Set **Source** (the animated armature) and **Target** (your rig).
3. Choose **Legs in IK** (feet stay planted) and/or **Arms in IK**.
4. Click **Auto Map**, then check the list (*source bone → target control*).
   The globe icon also transfers location. For the active pair: **Rotation
   Offset**, **Location Space** (*World* for the hips and IK controls, *Local* for
   bones animated by translation, such as facial bones, with a **Location
   Multiplier**) and **Location Offset**.
5. **Bind Preview** binds the target to the source live, without keying: play
   or scrub to check the map, click **Unbind** when done.
6. Click **Retarget**. A new action `<source action>_remap` (fake user, same
   frame range and cyclic setting as the source) is created.

| Option | |
| --- | --- |
| Mirror (list icon) | Add the other side's pairs |
| Select in List (arrow icon) | Make the pair of the active bone the active row |
| Save / Load Bone Map | JSON presets (Load can keep the current pairs) |
| Replace Namespace | Change a prefix in the source names (`mixamorig:` → `mixamorig1:`). It is also applied while retargeting |
| Use Current Target Pose, Source Reference Frame | Match different rest poses (T-pose vs A-pose) |
| Scale | Multiplier for the automatic height-based scale |
| In Place | Remove the travel over the ground (looping walks) |
| IK Pole Mode | **Absolute** (from the source limb's bend), **Relative: Target** (follows the hand/foot), **Relative: Chain** (follows the upper limb) |
| Clean IK Poles | Smooth the pole motion |
| Clean FK Rotation | Forearms and shins only bend on their hinge axis |
| Bake Only Existing Keyframes | Key only the source's keyed frames |
| Extract Root Motion | Bake the root along the pelvis afterwards (Rigger targets) |
| Start / End | Frame range (-1 = the action's range) |
| Multiple Source Animations | Retarget every ticked action of the source, filtered by name |

How it works: each target bone gets the same world-space rotation *change* its
source bone made from its reference pose, so different bone rolls and axis
conventions don't matter. Root motion is scaled by the ratio of the hip heights.

### Video to Animation

Animates the rig from an ordinary video of one person (phone clip, MP4, MOV...):
whole body, fingers, face shape keys and the path over the floor. Nothing is
uploaded: it runs on your computer with the same pose model as AI Markers
(install it once from the add-on preferences).

1. Select the rig (or a mesh bound to it) and open **Rigger > Video to Animation**.
2. Pick the video. Set **Focal Length** if you know it (35 mm-equivalent; phones
   are about 24-28 mm, the default 26 mm is fine when unsure).
3. Optionally limit **Start** / **Frames**, then click **Video to Animation**.
   It runs in a second, hidden Blender: keep working, watch the progress bar,
   press `Esc` to cancel.
4. A new action `<rig>_<video>` is keyed on the FK controls and the torso (your
   other actions are kept), the scene gets the video's frame rate and range, and
   with **Camera** on, a camera matching the video, with the video as its
   background, so you can compare.

| Option | |
| --- | --- |
| Camera Tilt | How far the camera looks down, in degrees. Used when the body alone cannot tell (low cameras): leave 0 for tripod and handheld shots |
| Smoothing | Steadier (higher) or quicker (lower) motion |
| Plant Feet | Feet that touch the floor stay still (leg IK), the body stays on the floor |
| Fingers / Face | Animate the fingers; drive shape keys named like ARKit (`eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft`, `browInnerUp`...) |
| Simplify | Remove keys that the tolerance can rebuild (0 keeps one key per frame) |
| Save | Also save the solved motion (`.mcap`); **Apply Motion File** puts it on any other Rigger rig |

**Two cameras (much more accurate).** Film the same take with a second phone
about 60-90 degrees around the person, and set it as **2nd**. Clap once at the
start: **Sync by Sound** lines the videos up from their sound (or turn it off and
type the **Offset** in seconds). No calibration board is needed: the cameras are
calibrated from the person, and the focal length is found automatically. Depth
no longer has to be guessed, so arms pointing at a camera and deep bends come out
right; a second **VideoCamera 2** with its own video is added for checking.

**FreeMoCap.** **Import FreeMoCap** reads a FreeMoCap recording (choose any file
in its `output_data` folder: the `mediapipe_*_3d_xyz` .npy or .csv files, version
1.x or 2.x; set its frame rate) and fits the rig to it the same way.

**Depth Cues (experimental, off).** With one camera, a depth model can judge
which limb is nearer (install it with the button next to the option). On our
test clips it did not help, so it is off by default.

Tips: the whole body should stay in the picture, the camera should not move,
and loose clothing hides the joints. Keypoints are cached per video, so a second
run (other settings, another character) only re-solves.

How it works: each frame, the character's own skeleton is posed so its joints,
seen through the camera, land on the keypoints the pose model found (bone
lengths give the depth; joint limits and the previous frame settle what one
view cannot show). A second pass holds planted feet on the floor found in the
first, then the motion is smoothed without lag, contacts are detected, and every
control gets its rotation change from rest. Single-camera capture cannot tell an
arm pointing straight at the camera from a bent one in every case: check such
moments and fix them with a few keys, or add a second camera. With two cameras
the second camera's position comes from the keypoints both see (epipolar
geometry, refined from a first guess) and its scale from the character's bone
lengths; the skeleton is then fitted to both videos at once.

---

## 7. Export

*File > Export > Rigger Game Rig*, or the **Export** panel. Rigger builds a
temporary skeleton with only `root`, the deform bones and your custom bones,
binds copies of the meshes to it, bakes the animation, writes the file and
deletes the temporary objects. Your scene is unchanged.

**Engine** (Unreal Engine, Unity, Godot, Others) and **Format** (FBX, glTF
Binary, glTF Separate) sit at the top, with **Presets** (save, load, remove).

**Rig tab**

| Option | |
| --- | --- |
| Skeleton | **Universal**: every deform bone (any creature). **Humanoid**: standard human bones only, other weights go to their parent. Options: **Full Facial** (off: jaw and eyes only), **Metacarpals** |
| Bone Names | **Rigger**, **Unreal Mannequin** (`pelvis`, `upperarm_l`, `lowerarm_twist_01_l`, `index_metacarpal_l`…) with optional **Add IK Bones** (`ik_foot_root`, `ik_hand_gun`…), **Unity Humanoid** (`Hips`, `LeftArm`…), **Godot Humanoid** (`Hips`, `LeftUpperArm`…), or **Keep** |
| Root Y Up | Godot: root bone pointing Y up for root motion |
| Rig Name | Name of the exported armature (Unreal default: `Armature`) |
| Export Twist | Off: twist weights go to the main limb bones |
| No Parents | Every bone directly under the root (stretchy rigs) |
| Selected Objects Only, Selected Bones Only | |
| Rename Bones From | A text block or file with `old = new` lines |
| Custom Script | A text block or Python file run on the export skeleton (`rig_export`) before baking. A script can do anything, and its name is saved in the .blend, so the Export dialog asks (*Run Custom Script*) once per session; Quick Export only runs scripts allowed that way. Exports started from Python run it |

**Animations tab**

| Option | |
| --- | --- |
| Bake Animations, Type | **Actions** (one take per action) or **NLA** (the scene as the NLA plays it) |
| Only Active Action / action list | Tick the actions to export, **All / None**, **Search** filter |
| Actions Manager | Named lists of actions, each with an export toggle (**Add Actions** adds the rig's actions matching the search) |
| Files | One file, **One File per Action** or **One File per Actions List** (`<file>_<name>.fbx`) |
| Frame Range | Each action's range, **Manual**, or **Markers** (`start` / `end` pose markers in the action) |
| Action Names | `Rig|Action` with a separator, or the action name only |
| Root Motion | The root carries the body's travel |
| Reset Transforms | Reset the controls between actions |
| Sampling Rate, Simplify | Frames between keys, FBX curve simplification |
| Actions Linker | While Rig A's action bakes, Rig B plays its linked action (a character holding a prop rig, two characters interacting) |

**Files tab**: Include Meshes, **Add Dummy Mesh**, **Force Rest Pose Export**
(bind pose written even without meshes), Apply Modifiers, Apply Subsurf,
Triangulate, Smoothing, Vertex Colors, Embed Textures, **Units x100**
(centimetres), Global Scale, **Convert Axes** (bake the axis conversion) and the
bone axes.

**Check Rig** lists what exports badly (stretch settings, moved tweak
controls, Preserve Volume, Soft IK, scale, a mesh named like the skeleton).
**Fix Rig** corrects what it can. **Quick Export** writes the last file again with
the current settings.

Scripted export:

```python
s = bpy.context.scene.rigger_export
s.engine, s.naming, s.units_x100 = 'UNREAL', 'UNREAL', True
bpy.ops.rigger.export_fbx(filepath="/tmp/hero.fbx")
```

---

## Troubleshooting

* **An operator fails with "… failed: …"**: Rigger puts the active object,
  selection and mode back and prints the full error in the system console
  (*Window > Toggle System Console* on Windows). One Ctrl+Z undoes anything the
  operator changed before failing. Please include the console text when
  reporting the problem.
* **Heat weighting failed / some vertices didn't move**: use *Heat + Fallback*
  (the default), *Voxelized* for messy meshes, or *Proximity*.
* **Knee or elbow bends the wrong way**: bend the joint slightly in the right
  direction in the reference (check with the IK directions button) and
  regenerate.
* **"Two modules tried to create the bone …"**: two reference bones produce the
  same control name. Rename one of them.
* **Retargeted arms are offset**: the reference poses differ (T vs A pose). See
  *Use Current Target Pose*.
* **Unreal adds an extra root bone**: keep the Unreal engine preset (armature
  named `Armature`).
