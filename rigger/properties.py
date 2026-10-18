# SPDX-License-Identifier: GPL-3.0-or-later
"""Property groups stored on pose bones, objects and the scene."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty, StringProperty)

from .modules import module_items


def _is_armature(_self, obj):
    return obj.type == 'ARMATURE'


_FINGER_ROT_ITEMS = [
    ('NONE', "None", "Scaling the first phalange does not rotate the finger"),
    ('ALL', "All Phalanges", "Scaling the first phalange curls the three phalanges"),
    ('LAST_TWO', "Last Two", "Scaling the first phalange curls the second and third phalanges"),
]


class RiggerBoneSettings(bpy.types.PropertyGroup):
    module: EnumProperty(
        name="Module", items=module_items(), default='NONE',
        description="Rig module that starts at this reference bone")
    parent_fallback: StringProperty(
        name="Parent Fallback", default="root",
        description="Parent of the module when its first reference bone has no parent "
                    "(a rig bone name, empty = no parent)")
    collection: StringProperty(
        name="Bone Collection", default="",
        description="Bone collection for the controls (empty = module default)")

    # -- arms and legs ------------------------------------------------------------
    twist: IntProperty(
        name="Twist Bones", default=2, min=0, max=32,
        description="Number of twist bones added to the lower segment (forearm / shin)")
    twist_upper: IntProperty(
        name="Upper Twist Bones", default=1, min=0, max=32,
        description="Number of twist bones added to the upper segment (upper arm / thigh). "
                    "The ones near the shoulder / hip twist less")
    twist_factor: FloatProperty(
        name="Twist Amount", default=1.0, min=0.0, max=1.0,
        description="How much the twist bones twist (1 = full)")
    start_ik: BoolProperty(
        name="Start in IK", default=False,
        description="Initial value of the IK/FK switch")
    secondary: BoolProperty(
        name="Secondary Controls", default=True,
        description="Tweak controls along the limb to bend and squash it")
    secondary_type: EnumProperty(
        name="Secondary Type", default='BENDY', items=[
            ('BENDY', "Bendy Bones", "Smooth curves with few controls (Blender only, not exported)"),
            ('TWIST', "Twist", "One tweak control per deform and twist bone (exported to game engines)"),
        ])
    fans: BoolProperty(
        name="Joint Fans", default=True,
        description="Extra deform bones at the elbow/knee that keep volume when bending "
                    "(exported to game engines)")
    fans_root: BoolProperty(name="Root Fans", default=False,
                            description="Joint fans at the shoulder / hip")
    fans_end: BoolProperty(name="End Fans", default=False,
                           description="Joint fans at the wrist / ankle")
    fan_count: IntProperty(name="Fans per Joint", default=1, min=1, max=32)
    fan_bulge: FloatProperty(
        name="Bulge", default=0.0, min=0.0, max=2.0,
        description="Fans push out (or stretch) as the joint bends. 0 = off")
    fan_bulge_mode: EnumProperty(name="Bulge Mode", default='LOCATION', items=[
        ('LOCATION', "Location", "Fans move along their length"),
        ('SCALE', "Scale", "Fans stretch along their length"),
    ])
    pole_distance: FloatProperty(
        name="IK Pole Distance", default=0.6, min=0.05, max=5.0,
        description="Distance of the pole control from the elbow/knee, relative to the limb length")
    soft_ik: BoolProperty(
        name="Soft IK", default=False,
        description="Avoid the elbow/knee pop near full extension. The chain stretches slightly; "
                    "not recommended for game export")
    soft_ik_amount: FloatProperty(name="Softness", default=0.05, min=0.001, max=0.5)
    auto_ik_roll: BoolProperty(
        name="Auto IK Roll", default=True,
        description="Align the limb bones' roll to the bend plane. Off: keep the reference bone roll")
    fk_lock: BoolProperty(
        name="FK Lock Setting", default=False,
        description="Add a 'fk_lock' slider to the first FK control: follow the body (1) "
                    "or keep the world rotation (0)")
    ik_offset: BoolProperty(name="IK Offset Control", default=False,
                            description="Second IK hand/foot control, child of the main one")
    ik_pivot: BoolProperty(name="IK Pivot Control", default=False,
                           description="Control to rotate the IK hand/foot around a custom point")
    roll_break: BoolProperty(
        name="Foot Roll Break", default=False,
        description="Rolling forward rotates the foot from the ball first, then from the toe tip")
    toes_pivot: BoolProperty(name="Toes Pivot Control", default=False,
                             description="Control rotating the whole foot from the ball")
    smart_pole: BoolProperty(
        name="Smart IK Pole", default=False,
        description="Add a pole space that follows the foot's position and heading only, "
                    "so tilting the foot does not flip the knee")
    three_bones: BoolProperty(
        name="3 Bones Leg", default=False,
        description="Extra leg bone at the root (digitigrade legs: dogs, horses, birds). "
                    "The IK chain switches between 2 and 3 bones")
    rot_from_scale: EnumProperty(
        name="Rotate Digits from Scale", default='NONE', items=_FINGER_ROT_ITEMS,
        description="Scaling the first phalange of a finger or toe curls it")
    auto_spread: BoolProperty(
        name="Auto Spread", default=False,
        description="Rotating the pinky's first bone sideways spreads the other fingers")
    pinky_auto: BoolProperty(
        name="Pinky Auto", default=True,
        description="Rotating the pinky metacarpal/metatarsal also rotates the ring and middle ones "
                    "(off = independent pinky)")
    wings_fold: BoolProperty(
        name="Wings Fold Control", default=False,
        description="Control that folds the wing by scaling it. Plays an action whose name "
                    "contains 'rig_wings_fold' (rest pose at frame 0, folded at frame 10)")

    # -- fingers and chains -----------------------------------------------------------
    curl: BoolProperty(
        name="Curl Master", default=True,
        description="Add a master control that curls the whole chain")
    finger_ik: BoolProperty(name="IK-FK", default=False,
                            description="Add an IK chain with an IK/FK switch")
    finger_pole_distance: FloatProperty(name="IK Pole Distance", default=1.0, min=0.1, max=5.0,
                                        description="Relative to the finger length")
    finger_shape: EnumProperty(name="Shape", default='CIRCLE', items=[
        ('CIRCLE', "Circle", ""), ('BOX', "Box", "")])
    bbone_segments: IntProperty(name="Segments", default=8, min=1, max=32,
                                description="Bendy-bone segments per bone")
    bbone_scale: FloatProperty(name="Control Scale", default=1.0, min=0.05, max=10.0)

    # -- spline IK --------------------------------------------------------------------
    spline_controls: IntProperty(
        name="Controls", default=3, min=2, max=64,
        description="Number of controls along the spline IK chain")
    spline_stretch: BoolProperty(
        name="Stretch", default=True, description="Let the chain stretch to follow the curve")
    spline_smooth: IntProperty(name="Smoothness", default=4, min=2, max=6,
                               description="Curve order: higher is smoother, 2 is linear")
    spline_fk: BoolProperty(name="IK-FK Chain", default=False,
                            description="Add an FK chain and an IK/FK switch")
    spline_twist: BoolProperty(name="Twist", default=True,
                               description="Rotating the tip control around the chain twists it")
    spline_tail: BoolProperty(
        name="Tail Bone", default=False,
        description="Extra deform bone at the tip that is not stretched or twisted by the curve, "
                    "to parent other modules to")

    # -- skirt ------------------------------------------------------------------------
    skirt_collide: BoolProperty(name="Collide with Legs", default=True,
                                description="Strands are pushed away by the thighs")
    skirt_distance: FloatProperty(name="Collision Distance", default=1.0, min=0.0, max=3.0,
                                  description="Relative to the rest distance between strand and thigh")
    skirt_masters: BoolProperty(name="Column Masters", default=False,
                                description="Master controls every few strands that drag their neighbours")
    skirt_master_freq: IntProperty(name="Master Frequency", default=3, min=2, max=16)
    skirt_row_masters: BoolProperty(name="Row Masters", default=False,
                                    description="One ring control per row that rotates the whole row outward")

    # -- spine, neck, head -------------------------------------------------------------
    spine_master: BoolProperty(name="Spine Master", default=False,
                               description="Control that moves, rotates and stretches all spine bones")
    spine_master_space: EnumProperty(name="Master Space", default='MASTER', items=[
        ('MASTER', "Master", "The spine rotates from the master control, at the bottom of the spine"),
        ('LOCAL', "Local", "Each spine bone rotates from its own pivot"),
    ])
    bottom: BoolProperty(name="Bottom Bones", default=False,
                         description="Buttock deform bones, following the thighs partly")
    reversed_spine: BoolProperty(
        name="Reversed Spine", default=False,
        description="Second FK spine rotating from the chest down, switched with 'spine_reverse'")
    neck_twist: BoolProperty(name="Neck Twist", default=True,
                             description="Neck bones twist along when the head twists")
    neck_bendy: IntProperty(name="Neck Bendy Segments", default=1, min=1, max=32,
                            description="Bendy-bone segments of the neck deform bones (1 = off)")
    skulls: BoolProperty(name="Skull Controls", default=False,
                         description="Three controls moving the lower (mouth), middle (eyes, nose) "
                                     "and top parts of the head")

    # -- eyes and face ------------------------------------------------------------------
    eye_distance: FloatProperty(name="Eye Target Distance", default=12.0, min=1.0, max=100.0,
                                description="Distance of the eye targets, in eye sizes")
    lips_offset: BoolProperty(name="Lips Offset Control", default=False,
                              description="Control moving all the lips at once")
    muzzle: BoolProperty(name="Muzzle Control", default=False,
                         description="Control moving the whole mouth and the nose (animals)")
    sticky_lips: BoolProperty(name="Sticky Lips", default=False,
                              description="Lips stay sealed when the jaw closes further")
    lips_roll: BoolProperty(name="Lips Roll", default=False,
                            description="Controls rolling the upper and lower lips")
    soft_lips: BoolProperty(name="Soft Lips", default=True,
                            description="Lips follow the corners and the jaw smoothly")
    soft_lips_linear: FloatProperty(
        name="Linear", default=0.0, min=-1.0, max=1.0,
        description="Shape of the soft lips falloff: -1 = very curved, 1 = straight")
    lips_masters_freq: IntProperty(name="Lips Masters Frequency", default=1, min=1, max=8,
                                   description="One lips master every N lips (1 = none)")
    soft_cheeks: BoolProperty(name="Soft Cheeks", default=False,
                              description="Cheeks follow the lip corners")
    brow_master: BoolProperty(name="Eyebrow Masters", default=False,
                              description="One control per eyebrow moving all its bones")
    soft_brows: BoolProperty(name="Soft Eyebrows", default=True,
                             description="Eyebrow bones follow the master less toward the tip")
    lid_masters: BoolProperty(name="Eyelid Masters", default=True,
                              description="With several eyelid bones, one main control per eyelid")
    jaw_unlock_y: BoolProperty(name="Unlock Jaw Y Loc", default=False,
                               description="Moving the jaw control forward/backward slides the jaw")
    jaw_translate: BoolProperty(name="Translate to Rotate Jaw", default=False,
                                description="Moving the jaw control down opens the jaw")


def copy_settings(src, dst):
    """Copy every module setting from one bone to another."""
    for prop in src.bl_rna.properties:
        if prop.identifier in ("rna_type", "name") or prop.is_readonly:
            continue
        setattr(dst, prop.identifier, getattr(src, prop.identifier))


def _update_layer(self, context):
    from .layers import apply_layer
    apply_layer(self)


class RiggerLayerObject(bpy.types.PropertyGroup):
    obj: PointerProperty(name="Object", type=bpy.types.Object)


class RiggerLayer(bpy.types.PropertyGroup):
    """A named group of bone collections and objects shown or hidden together."""
    name: StringProperty(name="Name", default="Layer")
    show: BoolProperty(name="Show", default=True, update=_update_layer)
    objects: CollectionProperty(type=RiggerLayerObject)
    bone_collections: StringProperty(name="Bone Collections", default="",
                                     description="Comma separated bone collection names")


class RiggerObjectSettings(bpy.types.PropertyGroup):
    rig: PointerProperty(
        name="Rig", type=bpy.types.Object, poll=_is_armature,
        description="Rig generated from this reference")
    reference: PointerProperty(
        name="Reference", type=bpy.types.Object, poll=_is_armature,
        description="Reference armature this rig was generated from")
    rig_name: StringProperty(name="Rig Name", default="rig")
    widget_scale: FloatProperty(
        name="Control Size", default=1.0, min=0.05, max=20.0,
        description="Global size multiplier for control shapes")
    auto_hide: BoolProperty(
        name="Auto-Hide IK/FK", default=True,
        description="Hide FK controls in IK mode and IK controls in FK mode")
    layers: CollectionProperty(type=RiggerLayer)
    layer_index: IntProperty()
    init_scale: BoolProperty(
        name="Init Scale", default=True,
        description="Apply the reference object's scale before generating, so the rig has a scale of 1")
    use_theme: BoolProperty(name="Use Color Theme", default=False)
    color_left: FloatVectorProperty(name="Left", subtype='COLOR', size=3, min=0.0, max=1.0,
                                    default=(0.10, 0.42, 1.00))
    color_right: FloatVectorProperty(name="Right", subtype='COLOR', size=3, min=0.0, max=1.0,
                                     default=(1.00, 0.16, 0.16))
    color_center: FloatVectorProperty(name="Middle", subtype='COLOR', size=3, min=0.0, max=1.0,
                                      default=(1.00, 0.72, 0.02))
    animated_layers: BoolProperty(
        name="Animated Layers", default=False,
        description="Update rig layer visibility on every frame, so keyframed 'show' values play back")
    pinned_props: StringProperty(name="Pinned Properties", default="",
                                 description="Comma separated bone:property pairs always shown in the panel")
    multiped: BoolProperty(
        name="Multi-ped", default=False,
        description="Quadrupeds and other multi-peds: the torso control follows the spine direction "
                    "instead of pointing up")


class RiggerRemapItem(bpy.types.PropertyGroup):
    source: StringProperty(name="Source Bone")
    target: StringProperty(name="Target Bone")
    location: BoolProperty(
        name="Location", default=False,
        description="Transfer location as well as rotation (hips, IK controls)")
    rot_offset: FloatVectorProperty(
        name="Rotation Offset", size=3, subtype='EULER', default=(0.0, 0.0, 0.0),
        description="Extra rotation added to the target bone, in its own axes")
    loc_offset: FloatVectorProperty(
        name="Location Offset", size=3, subtype='TRANSLATION', default=(0.0, 0.0, 0.0),
        description="Extra world-space offset for location pairs")
    location_mode: EnumProperty(name="Location Space", default='WORLD', items=[
        ('WORLD', "World (Root/IK)", "World-space travel scaled by the height ratio (hips, IK hands and feet)"),
        ('LOCAL', "Local", "Local translation relative to the parent, e.g. facial bones animated by location")])
    loc_multiplier: FloatProperty(name="Location Multiplier", default=1.0, min=-10.0, max=10.0,
                                  description="Scales the transferred location (Local mode)")


def _source_poll(self, obj):
    return obj.type == 'ARMATURE' and obj != self.target


def _target_poll(self, obj):
    return obj.type == 'ARMATURE' and obj != self.source


class RiggerRemapSettings(bpy.types.PropertyGroup):
    source: PointerProperty(name="Source", type=bpy.types.Object, poll=_source_poll,
                            description="Animated armature to copy motion from")
    target: PointerProperty(name="Target", type=bpy.types.Object, poll=_target_poll,
                            description="Armature that receives the motion")
    items: CollectionProperty(type=RiggerRemapItem)
    active_index: IntProperty()
    legs_ik: BoolProperty(name="Legs in IK", default=True,
                          description="Drive the IK foot controls (keeps feet planted)")
    arms_ik: BoolProperty(name="Arms in IK", default=False,
                          description="Drive the IK hand controls instead of the FK arm")
    use_target_pose: BoolProperty(
        name="Use Current Target Pose", default=False,
        description="Use the target's current pose (instead of its rest pose) as the pose that "
                    "matches the source rest pose. Pose the target into the source's rest "
                    "pose first (e.g. T-pose vs A-pose)")
    source_frame: IntProperty(
        name="Source Reference Frame", default=-1, min=-1,
        description="Frame used as the source reference pose (-1 = rest pose)")
    scale: FloatProperty(name="Scale", default=1.0, min=0.001,
                         description="Multiplier for the automatic height-based location scale")
    frame_start: IntProperty(name="Start", default=-1, description="-1 = action start")
    frame_end: IntProperty(name="End", default=-1, description="-1 = action end")
    action_name: StringProperty(name="Action Name", default="",
                                description="Name of the new action (empty = source action + '_remap')")
    in_place: BoolProperty(name="In Place", default=False,
                           description="Remove the travel over the ground (looping walks, runs)")
    multiple: BoolProperty(name="Multiple Source Animations", default=False,
                           description="Retarget every ticked action of the source (see the action list)")
    action_filter: StringProperty(name="Only Containing", default="")
    pole_mode: EnumProperty(name="IK Pole Mode", default='ABSOLUTE', items=[
        ('ABSOLUTE', "Absolute", "Pole from the plane of the source limb (best with clean animation)"),
        ('TARGET', "Relative: Target", "Pole follows the IK hand/foot like a child"),
        ('CHAIN', "Relative: Chain", "Pole follows the first bone of the limb like a child")])
    smooth_poles: BoolProperty(name="Clean IK Poles", default=False,
                               description="Smooth the pole positions over time to avoid flips")
    clean_fk: BoolProperty(name="Clean FK Rotation", default=False,
                           description="Forearm and shin FK controls only bend on their hinge axis")
    existing_keys: BoolProperty(name="Bake Only Existing Keyframes", default=False,
                                description="Key only the frames where the source action has keys")
    root_motion: BoolProperty(name="Extract Root Motion", default=False,
                              description="Rigger targets: bake the root along the pelvis afterwards")
    action_index: IntProperty()
    namespace_from: StringProperty(name="Replace", default="", description="Text to replace in the source names")
    namespace_to: StringProperty(name="With", default="")


class RiggerTemplateSettings(bpy.types.PropertyGroup):
    """Options of the human template and Smart markers."""
    spine_count: IntProperty(name="Spine Bones", default=3, min=1, max=64,
                             description="Spine bones above the hips (UE4: 3, UE5: 5)")
    neck_count: IntProperty(name="Neck Bones", default=1, min=1, max=16, description="UE4: 1, UE5: 2")
    twist: IntProperty(name="Twist Bones", default=2, min=0, max=32,
                       description="Twist bones of the forearm/shin; the upper arm/thigh get one less")
    fingers: IntProperty(name="Fingers", default=5, min=1, max=5)
    toes: BoolProperty(name="Individual Toes", default=False, description="Five toes on each foot")
    metacarpals: BoolProperty(name="Metacarpals", default=False,
                              description="Palm bones before the first phalange of each finger (not the thumb)")
    toes_metatarsal: BoolProperty(name="Toes Metatarsal", default=False,
                                  description="Root bone before the first phalange of each toe")
    toes_parent_foot: BoolProperty(name="Toes Parent Foot", default=False,
                                   description="Individual toes (or their metatarsals) follow the foot instead of "
                                               "the main toe bone")
    lips: IntProperty(name="Lips per Side", default=1, min=1, max=16,
                      description="Lip bones between the middle and the corner, per row and side")
    lids: IntProperty(name="Eyelid Bones", default=1, min=1, max=16, description="Bones per eyelid")
    brows: IntProperty(name="Eyebrow Bones", default=3, min=1, max=16)
    eyes: BoolProperty(name="Eyes", default=True)
    eyebrows: BoolProperty(name="Eyebrows", default=True)
    nose: BoolProperty(name="Nose", default=True)
    mouth: BoolProperty(name="Lips", default=True)
    tongue: BoolProperty(name="Tongue", default=True)
    teeth: BoolProperty(name="Teeth", default=True)
    chin: BoolProperty(name="Chin", default=True)
    cheeks: BoolProperty(name="Cheeks", default=True)
    ears: BoolProperty(name="Ears", default=False)


def _armature_poll(_self, obj):
    return obj.type == 'ARMATURE'


class RiggerActionItem(bpy.types.PropertyGroup):
    action: PointerProperty(name="Action", type=bpy.types.Action)


class RiggerActionList(bpy.types.PropertyGroup):
    """A named list of actions exported together (Actions Manager)."""
    name: StringProperty(name="Name", default="Actions")
    enabled: BoolProperty(name="Export", default=True)
    actions: CollectionProperty(type=RiggerActionItem)


class RiggerActionLink(bpy.types.PropertyGroup):
    """While Rig A's action is baked, Rig B plays its linked action (characters constrained together)."""
    rig_a: PointerProperty(name="Rig A", type=bpy.types.Object, poll=_armature_poll)
    action_a: PointerProperty(name="Action A", type=bpy.types.Action)
    rig_b: PointerProperty(name="Rig B", type=bpy.types.Object, poll=_armature_poll)
    action_b: PointerProperty(name="Action B", type=bpy.types.Action)


class RiggerExportSettings(bpy.types.PropertyGroup):
    engine: EnumProperty(name="Engine", default='UNREAL', items=[
        ('UNREAL', "Unreal Engine", ""), ('UNITY', "Unity", ""), ('GODOT', "Godot", ""),
        ('GENERIC', "Others", "Other engines and applications")])
    file_format: EnumProperty(name="Format", default='FBX', items=[
        ('FBX', "FBX", ""), ('GLB', "glTF Binary (.glb)", ""), ('GLTF_SEPARATE', "glTF Separate (.gltf)", "")])
    skeleton: EnumProperty(name="Skeleton", default='UNIVERSAL', items=[
        ('UNIVERSAL', "Universal", "Every deform bone: any creature"),
        ('HUMANOID', "Humanoid", "Standard human bones only (spine, limbs, fingers, head); the weights of "
                                 "the other bones go to their parent")])
    full_facial: BoolProperty(name="Full Facial", default=True,
                              description="Humanoid: export every face bone, otherwise only jaw and eyes")
    metacarpals: BoolProperty(name="Metacarpals", default=True,
                              description="Humanoid: export the palm bones; off: their weights go to the hand")
    force_rest_pose: BoolProperty(name="Force Rest Pose Export", default=False,
                                  description="Write the bind pose even when no mesh is exported (a hidden "
                                              "helper mesh bound to every bone is added to the file)")
    naming: EnumProperty(name="Bone Names", default='RIGGER', items=[
        ('RIGGER', "Rigger", "Deform bone names without 'DEF-'"),
        ('UNREAL', "Unreal Mannequin", "pelvis, upperarm_l, calf_twist_01_l..."),
        ('UNITY', "Unity Humanoid", "Hips, LeftArm, LeftUpLeg..."),
        ('GODOT', "Godot Humanoid", "Hips, LeftUpperArm, LeftLowerLeg... (SkeletonProfileHumanoid)"),
        ('KEEP', "Keep", "Keep the rig's names")])
    rename_bones: StringProperty(name="Rename Bones From", default="",
                                 description="Text file or text block with 'old = new' lines")
    custom_script: StringProperty(name="Custom Script", default="",
                                  description="Python file or text block run on the export skeleton "
                                              "(the active object) before baking")
    rig_name: StringProperty(name="Rig Name", default="",
                             description="Name of the exported skeleton (empty = engine default)")
    units_x100: BoolProperty(name="Units x100", default=False,
                             description="Centimetre units: scene unit scale 0.01 and objects 100 times larger")
    ue_ik_bones: BoolProperty(name="Add IK Bones", default=False,
                              description="Unreal: ik_foot_root, ik_foot_l/r, ik_hand_root, ik_hand_gun, ik_hand_l/r")
    export_twist: BoolProperty(name="Export Twist", default=True,
                               description="Off: twist bone weights go to the main limb bones")
    no_parents: BoolProperty(name="No Parents", default=False,
                             description="Flat hierarchy: every bone is a child of the root (stretchy rigs)")
    include_meshes: BoolProperty(name="Include Meshes", default=True)
    selected_objects: BoolProperty(name="Selected Objects Only", default=False)
    selected_bones: BoolProperty(name="Selected Bones Only", default=False)
    root_motion: BoolProperty(name="Root Motion", default=False,
                              description="The root bone carries the body's travel over the ground")
    root_y_up: BoolProperty(name="Root Y Up", default=False,
                            description="Root bone pointing Y up / Z forward in the file (Godot root motion)")
    dummy_mesh: BoolProperty(name="Add Dummy Mesh", default=False,
                             description="Add a tiny mesh when no mesh is exported (engines that need one)")
    bake_actions: BoolProperty(name="Bake Animations", default=True)
    anim_type: EnumProperty(name="Type", default='ACTIONS', items=[
        ('ACTIONS', "Actions", "One animation per action"),
        ('NLA', "NLA", "One animation of the scene as the NLA editor plays it")])
    only_active: BoolProperty(name="Only Active Action", default=True)
    action_search: StringProperty(name="Search", default="", description="Only actions containing this text")
    use_actions_manager: BoolProperty(name="Actions Manager", default=False,
                                      description="Export the actions of the enabled action lists")
    action_lists: CollectionProperty(type=RiggerActionList)
    action_list_index: IntProperty()
    links: CollectionProperty(type=RiggerActionLink)
    link_index: IntProperty()
    one_file_per: EnumProperty(name="Files", default='NONE', items=[
        ('NONE', "One File", ""), ('ACTION', "One File per Action", ""),
        ('LIST', "One File per Actions List", "")])
    frame_range: EnumProperty(name="Frame Range", default='ACTION', items=[
        ('ACTION', "Action", "Frame range of each action"),
        ('MANUAL', "Manual", ""),
        ('MARKERS', "Markers", "Between the action's 'start' and 'end' pose markers")])
    frame_start: IntProperty(name="Start", default=1)
    frame_end: IntProperty(name="End", default=250)
    action_names: EnumProperty(name="Action Names", default='RIG_ACTION', items=[
        ('RIG_ACTION', "Rig + Action", "RigName|ActionName"), ('ACTION', "Action Only", "")])
    name_separator: EnumProperty(name="Separator", default='|', items=[(c, c, "") for c in ('|', '_', '-')])
    simplify: FloatProperty(name="Simplify", default=0.0, min=0.0, max=10.0)
    sampling: FloatProperty(name="Sampling Rate", default=1.0, min=0.01, max=100.0,
                            description="Frames between baked keys (below 1 = subframes)")
    reset_transforms: BoolProperty(name="Reset Transforms", default=True,
                                   description="Reset the controls between actions so unkeyed ones do not "
                                               "carry the previous action's pose")
    smoothing: EnumProperty(name="Smoothing", default='FACE', items=[
        ('OFF', "Normals Only", ""), ('FACE', "Face", ""), ('EDGE', "Edge", "")])
    triangulate: BoolProperty(name="Triangulate", default=False)
    apply_modifiers: BoolProperty(name="Apply Modifiers", default=True)
    apply_subsurf: BoolProperty(name="Apply Subsurf", default=True)
    vertex_colors: EnumProperty(name="Vertex Colors", default='SRGB', items=[
        ('NONE', "None", ""), ('SRGB', "sRGB", ""), ('LINEAR', "Linear", "")])
    embed_textures: BoolProperty(name="Embed Textures", default=False)
    init_rotation: BoolProperty(name="Convert Axes", default=False,
                                description="Bake the axis conversion into the data (zeroed object rotations)")
    primary_axis: EnumProperty(name="Primary Bone Axis", default='Y', items=[
        (a, a, "") for a in ('X', 'Y', 'Z', '-X', '-Y', '-Z')])
    secondary_axis: EnumProperty(name="Secondary Bone Axis", default='X', items=[
        (a, a, "") for a in ('X', 'Y', 'Z', '-X', '-Y', '-Z')])
    global_scale: FloatProperty(name="Global Scale", default=1.0, min=0.001, max=1000.0)
    last_filepath: StringProperty(name="Last File", default="", subtype='FILE_PATH')
    action_index: IntProperty()
    ui_tab: EnumProperty(name="Tab", default='RIG', items=[
        ('RIG', "Rig", "Skeleton and bones"), ('ANIM', "Animations", "Actions and baking"),
        ('MISC', "Files", "Meshes, axes, units and files")])


QUICK_KINDS = [
    ('SPINE', "Spine", "Hips/pelvis first, then the spine bones"),
    ('NECK', "Neck & Head", "Neck bones, then the head"),
    ('ARM', "Arm", "Shoulder (optional), upper arm, forearm, hand"),
    ('LEG', "Leg", "Thigh, shin, foot, toe (optional)"),
    ('FINGER', "Finger", "Finger bones from the base to the tip"),
    ('CHAIN', "FK Chain", "Tail, ears, hair... any chain of bones"),
    ('SIMPLE', "Single Bone", "One bone (jaw, prop...)"),
]


class RiggerQuickLimb(bpy.types.PropertyGroup):
    kind: EnumProperty(name="Type", items=QUICK_KINDS, default='CHAIN')
    side: EnumProperty(name="Side", default='C', items=[
        ('L', "Left", ""), ('R', "Right", ""), ('C', "Center", "")])
    label: StringProperty(name="Name", default="",
                          description="Name of the reference bones (finger: thumb, index...; chain: tail...)")
    bones: StringProperty(name="Bones", default="",
                          description="Source bone names from root to tip, separated by commas")
    metacarpal: BoolProperty(name="Metacarpal First", default=False,
                             description="Finger: the first bone is a palm bone (4 bones per finger)")


class RiggerQuickOverride(bpy.types.PropertyGroup):
    source: StringProperty(name="Vertex Group", description="Source vertex group (bone name)")
    target: StringProperty(name="Bone", description="Reference or rig bone that receives the weights")


def _skeleton_poll(_self, obj):
    return obj.type == 'ARMATURE' and "rigger_meta" not in obj and not obj.get("rigger_reference")


class RiggerQuickRigSettings(bpy.types.PropertyGroup):
    armature: PointerProperty(name="Armature", type=bpy.types.Object, poll=_skeleton_poll,
                              description="Existing skeleton to rig (Mixamo, game rig, any armature)")
    mode: EnumProperty(name="Mode", default='PRESERVE', items=[
        ('PRESERVE', "Preserve", "Keep the skeleton and its weights: it follows the control rig "
                                 "(bone names stay compatible with game engines)"),
        ('CONVERT', "Convert", "Replace the skeleton: its vertex groups are renamed to the rig's deform bones")])
    limbs: CollectionProperty(type=RiggerQuickLimb)
    active_index: IntProperty(default=0)
    overrides: CollectionProperty(type=RiggerQuickOverride)
    override_index: IntProperty(default=0)
    orphans: BoolProperty(name="Add Orphan Bones", default=True,
                          description="Bones that are in no limb become single-bone controls")
    ignore_root: BoolProperty(name="Ignore Root Bone", default=True,
                              description="Skip the top-most bone when it is not the hips (e.g. 'root', "
                                          "'Armature'): the rig has its own root")
    same_origin: BoolProperty(name="Same Origin", default=True,
                              description="Put the rig at the location and rotation of the armature object")
    hide_base: BoolProperty(name="Hide Base Armature", default=True,
                            description="Preserve mode: hide the skeleton that follows the rig")
    bake: BoolProperty(name="Bake Animation", default=True,
                       description="Transfer the armature's current action to the control rig")
    delete_source: BoolProperty(name="Delete Source Armature", default=True,
                                description="Convert mode: remove the original armature")
    twist: IntProperty(name="Twist Bones", default=0, min=0, max=4,
                       description="Twist bones of the generated limbs (0 keeps the deform bones 1:1 "
                                   "with the source)")
    fingers_ik: BoolProperty(name="Finger IK", default=False)
    secondary: BoolProperty(name="Secondary Controls", default=False,
                            description="Bendy tweak controls on the limbs (curves the limbs, so the "
                                        "deformation differs from the original skeleton's)")


classes = (RiggerQuickLimb, RiggerQuickOverride, RiggerQuickRigSettings, RiggerTemplateSettings, RiggerActionItem, RiggerActionList, RiggerActionLink, RiggerExportSettings,
           RiggerBoneSettings, RiggerLayerObject, RiggerLayer, RiggerObjectSettings,
           RiggerRemapItem, RiggerRemapSettings)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.PoseBone.rigger = PointerProperty(type=RiggerBoneSettings)
    bpy.types.Object.rigger = PointerProperty(type=RiggerObjectSettings)
    bpy.types.Scene.rigger_remap = PointerProperty(type=RiggerRemapSettings)
    bpy.types.Scene.rigger_template = PointerProperty(type=RiggerTemplateSettings)
    bpy.types.Scene.rigger_export = PointerProperty(type=RiggerExportSettings)
    bpy.types.Scene.rigger_quick = PointerProperty(type=RiggerQuickRigSettings)
    bpy.types.Scene.rigger_picker_size = FloatProperty(
        name="Picker Size", default=1.0, min=0.4, max=3.0, subtype='FACTOR',
        description="Size of the viewport picker")
    bpy.types.Action.rigger_export = BoolProperty(name="Export", default=True,
                                                  description="Export this action to game engines")
    bpy.types.Action.rigger_remap = BoolProperty(name="Retarget", default=True,
                                                 description="Retarget this action (Multiple Source Animations)")


def unregister():
    del bpy.types.Action.rigger_remap
    del bpy.types.Action.rigger_export
    del bpy.types.Scene.rigger_picker_size
    del bpy.types.Scene.rigger_quick
    del bpy.types.Scene.rigger_export
    del bpy.types.Scene.rigger_template
    del bpy.types.Scene.rigger_remap
    del bpy.types.Object.rigger
    del bpy.types.PoseBone.rigger
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
