# SPDX-License-Identifier: GPL-3.0-or-later
"""Game-engine export.

Builds a temporary, clean skeleton that only contains the root, the deform bones
and the bones tagged as custom (no controls, no mechanism, no constraints in the
file), binds copies of the meshes to it, bakes the animation and writes FBX or
glTF files. The scene is left untouched afterwards.

Settings live on the scene (``scene.rigger_export``) so they can be scripted:

    s = bpy.context.scene.rigger_export
    s.engine, s.naming, s.units_x100 = 'UNREAL', 'UNREAL', True
    bpy.ops.rigger.export_fbx(filepath="/tmp/hero.fbx")
"""

import json
import os
import re

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Vector

from . import utils

NAMING_ITEMS = [
    ('RIGGER', "Rigger", "Deform bone names without the 'DEF-' prefix (upper_arm.L)"),
    ('UNREAL', "Unreal Mannequin", "Unreal Engine mannequin names (pelvis, upperarm_l, calf_twist_01_l...)"),
    ('UNITY', "Unity Humanoid", "Mixamo-style names that Unity's Humanoid avatar maps automatically "
                                "(Hips, LeftArm, LeftUpLeg...)"),
    ('GODOT', "Godot Humanoid", "Godot's SkeletonProfileHumanoid names (Hips, LeftUpperArm...)"),
    ('KEEP', "Keep", "Keep the rig's names, including the 'DEF-' prefix"),
]

_UNREAL = {"shoulder": "clavicle", "upper_arm": "upperarm", "forearm": "lowerarm", "hand": "hand",
           "thigh": "thigh", "shin": "calf", "foot": "foot", "toe": "ball"}
_UNITY = {"shoulder": "Shoulder", "upper_arm": "Arm", "forearm": "ForeArm", "hand": "Hand",
          "thigh": "UpLeg", "shin": "Leg", "foot": "Foot", "toe": "ToeBase"}
_GODOT = {"shoulder": "Shoulder", "upper_arm": "UpperArm", "forearm": "LowerArm", "hand": "Hand",
          "thigh": "UpperLeg", "shin": "LowerLeg", "foot": "Foot", "toe": "Toes"}
_GODOT_SPINE = ("Spine", "Chest", "UpperChest")
_GODOT_FINGER = {"thumb": ("Metacarpal", "Proximal", "Distal"), None: ("Proximal", "Intermediate", "Distal")}


def _role_name(scheme, role, side, finger, index):
    """Engine name for one bone (None = no standard name). index counts from 1 within its chain."""
    if role == "metacarpal":
        return f"{finger}_metacarpal_{side.lower()}" if scheme == 'UNREAL' and index == 1 else None
    if role not in _UNREAL and role not in ("hips", "spine", "neck", "head", "finger"):
        return None
    if scheme == 'UNREAL':
        s = side.lower()
        if role == "hips":
            return "pelvis"
        if role in ("spine", "neck"):
            return f"{role}_{index:02d}"
        if role == "head":
            return "head"
        if role == "finger":
            return f"{finger}_{index:02d}_{s}"
        return f"{_UNREAL[role]}_{s}"
    side_word = {"L": "Left", "R": "Right"}.get(side, "")
    if scheme == 'GODOT':
        if role == "hips":
            return "Hips"
        if role == "spine":
            return _GODOT_SPINE[index - 1] if index <= 3 else f"UpperChest{index - 3}"
        if role == "neck":
            return "Neck" if index == 1 else f"Neck{index}"
        if role == "head":
            return "Head"
        if role == "finger":
            parts = _GODOT_FINGER.get(finger, _GODOT_FINGER[None])
            return f"{side_word}{finger.title() if finger != 'pinky' else 'Little'}{parts[min(index, 3) - 1]}"
        return side_word + _GODOT[role]
    if role == "hips":
        return "Hips"
    if role in ("spine", "neck"):
        base = role.title()
        return base if index == 1 else f"{base}{index - 1}"
    if role == "head":
        return "Head"
    if role == "finger":
        return f"{side_word}Hand{finger.title()}{index}"
    return side_word + _UNITY[role]


def export_names(rig, bones, scheme):
    """Map rig bone names to exported names for a naming scheme."""
    from .retarget import classify_armature
    stripped = {n: (n[4:] if n.startswith("DEF-") and scheme != 'KEEP' else n) for n in bones}
    if scheme in ('RIGGER', 'KEEP'):
        return stripped
    names = dict(stripped)
    taken = set()
    groups = classify_armature(rig, [n for n in bones if n.startswith("DEF-")])
    single = {"hips", "head", "shoulder", "upper_arm", "forearm", "hand", "thigh", "shin", "foot", "toe"}
    for (role, side, finger), members in groups.items():
        if role.endswith("_ik") or (side == "" and role not in ("hips", "spine", "neck", "head", "finger")):
            continue
        if role in single:
            members = members[:1]  # e.g. a quadruped's front legs keep their own names
        for index, bone in enumerate(members, 1):
            new = _role_name(scheme, role, side, finger, index)
            if new is not None and new not in taken:
                names[bone] = new
                taken.add(new)
    # Twist bones follow their limb segment: DEF-forearm_twist.01.L -> lowerarm_twist_01_l
    for bone in bones:
        m = re.match(r"DEF-(.+)_twist\.(\d+)(\.[LR])?$", bone)
        if not m:
            continue
        base = names.get(f"DEF-{m.group(1)}{m.group(3) or ''}")
        if base is None:
            continue
        i = int(m.group(2))
        if scheme == 'UNREAL':
            stem, _, side = base.rpartition("_")
            new = f"{stem}_twist_{i:02d}_{side}" if side in ("l", "r") else f"{base}_twist_{i:02d}"
        else:
            new = f"{base}Twist{i}"
        if new not in taken:
            names[bone] = new
            taken.add(new)
    # Facial bones for the engine schemes
    if scheme in ('UNREAL', 'GODOT', 'UNITY'):
        for bone in bones:
            stem, suffix = utils.split_side(stripped[bone])
            side = utils.side_letter(stripped[bone])
            if stem == "jaw" and "jaw" not in taken:
                names[bone] = "jaw" if scheme == 'UNREAL' else "Jaw"
            elif stem == "eye" and side:
                names[bone] = f"eye_{side.lower()}" if scheme == 'UNREAL' else \
                    f"{'Left' if side == 'L' else 'Right'}Eye"
    # Keep names unique
    seen = set()
    for bone in bones:
        if names[bone] in seen:
            names[bone] = stripped[bone] if stripped[bone] not in seen else bone
        seen.add(names[bone])
    return names


# ---------------------------------------------------------------------------
# Bone selection
# ---------------------------------------------------------------------------

HUMANOID_ROLES = {"hips", "spine", "neck", "head", "shoulder", "upper_arm", "forearm", "hand", "thigh", "shin",
                  "foot", "toe", "finger", "metacarpal"}


def control_to_deform(rig, name):
    """Deform bone that a control drives ('forearm_fk.L' -> 'DEF-forearm.L')."""
    stem, suffix = utils.split_side(name)
    for cand in ("DEF-" + name, "DEF-" + stem.replace("_fk", "").replace("_ik", "") + suffix):
        if cand in rig.data.bones:
            return cand
    return None


def exported_bones(rig, settings):
    """Ordered names of the bones to export, the bones merged into others and the tags."""
    from .retarget import classify
    from .rig_data import export_tags
    tags = export_tags(rig)
    deform = [b.name for b in rig.data.bones if b.use_deform]
    keep = []
    for name in deform:
        if tags.get(name) == 'EXCLUDE':
            continue
        if "_twist." in name and not settings.export_twist:
            continue
        if settings.skeleton == 'HUMANOID':
            info = classify(name)
            stem = utils.split_side(name[4:] if name.startswith("DEF-") else name)[0]
            facial = stem in ("jaw", "eye") or (settings.full_facial and _is_face(rig, name))
            if "_twist." not in name and not facial and (info is None or info[0] not in HUMANOID_ROLES):
                continue
            if info is not None and info[0] == "metacarpal" and not getattr(settings, "metacarpals", True):
                continue
        keep.append(name)
    keep += [n for n, t in tags.items() if t == 'CUSTOM' and n not in keep and n in rig.data.bones]
    if settings.selected_bones:
        selected = {pb.name for pb in rig.pose.bones if utils.is_pose_bone_selected(pb)}
        if selected:
            keep = [n for n in keep if n in selected]
    if "root" in rig.data.bones:
        keep = ["root"] + [n for n in keep if n != "root"]
    keep.sort(key=lambda n: len(rig.data.bones[n].parent_recursive))
    merged = [n for n in deform if n not in keep]
    return keep, merged, tags


def _is_face(rig, name):
    """True for bones under the face module or the head (not the neck/spine)."""
    meta = utils.get_meta(rig)
    face_controls = set()
    for face in meta.get("faces", []):
        face_controls.update(face.get("controls", []))
    stem = name[4:] if name.startswith("DEF-") else name
    return stem in face_controls or stem.startswith(("skull", "eye", "ear"))


def export_parent(rig, name, kept):
    """Closest exported ancestor, retargeting controls to the deform bone they drive."""
    parent = rig.data.bones[name].parent
    while parent is not None:
        if parent.name in kept:
            return parent.name
        mapped = control_to_deform(rig, parent.name)
        if mapped and mapped in kept and mapped != name:
            return mapped
        parent = parent.parent
    return "root" if "root" in kept and name != "root" else None


def _rename_map(text):
    """'old = new' lines from a text block name or a file path."""
    if not text:
        return {}
    block = bpy.data.texts.get(text)
    if block is not None:
        content = block.as_string()
    else:
        path = bpy.path.abspath(text)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            content = f.read()
    mapping = {}
    for line in content.splitlines():
        line = line.split("#")[0].strip()
        if "=" in line:
            old, new = (part.strip() for part in line.split("=", 1))
            if old and new:
                mapping[old] = new
    return mapping


def _run_script(context, text, game):
    if not text:
        return
    block = bpy.data.texts.get(text)
    if block is not None:
        code = block.as_string()
    else:
        path = bpy.path.abspath(text)
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            code = f.read()
    context.view_layer.objects.active = game
    namespace = {"bpy": bpy, "rig_export": game, "__name__": "__rigger_export__"}
    exec(compile(code, text, "exec"), namespace)  # noqa: S102 - the user's own export script
    if context.object and context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


# ---------------------------------------------------------------------------
# Game skeleton
# ---------------------------------------------------------------------------

UE_IK_BONES = (("ik_foot_root", None, None), ("ik_foot_l", "ik_foot_root", "foot_l"),
               ("ik_foot_r", "ik_foot_root", "foot_r"), ("ik_hand_root", None, None),
               ("ik_hand_gun", "ik_hand_root", "hand_r"), ("ik_hand_l", "ik_hand_gun", "hand_l"),
               ("ik_hand_r", "ik_hand_gun", "hand_r"))


def build_game_rig(context, rig, naming='RIGGER', name="Armature", root_motion=False, settings=None):
    """Create an armature with root + deform (+ custom) bones following the rig.

    Returns (game rig, {rig bone: export bone}, {merged deform bone: export bone}, tags).
    """
    settings = settings or _Defaults(naming=naming, root_motion=root_motion)
    keep, merged, tags = exported_bones(rig, settings)
    data = bpy.data.armatures.new(name)
    game = bpy.data.objects.new(name, data)
    context.scene.collection.objects.link(game)
    game.matrix_world = rig.matrix_world.copy()

    utils.set_mode(game, 'EDIT')
    ebones = data.edit_bones
    mapping = {}
    names = export_names(rig, keep, settings.naming)
    renames = _rename_map(settings.rename_bones)
    for src_name in keep:
        src = rig.data.bones[src_name]
        new_name = renames.get(names[src_name], renames.get(src_name, names[src_name]))
        eb = ebones.new(new_name)
        eb.head, eb.tail = src.head_local, src.tail_local
        eb.align_roll(src.matrix_local.to_3x3().col[2])
        eb.use_deform = src.use_deform or tags.get(src_name) == 'CUSTOM'
        mapping[src_name] = eb.name
        if src_name == "root" and settings.root_y_up:
            eb.tail = eb.head + Vector((0, rig.data.bones["root"].length, 0))
            eb.align_roll(Vector((0, 0, 1)))
    for src_name in keep:
        if src_name == "root":
            continue
        parent = "root" if settings.no_parents else export_parent(rig, src_name, set(keep))
        if parent in mapping:
            ebones[mapping[src_name]].parent = ebones[mapping[parent]]
    ik = []
    if settings.ue_ik_bones and settings.naming == 'UNREAL':
        by_name = {v: k for k, v in mapping.items()}
        root_bone = ebones.get(mapping.get("root", ""))
        for ik_name, parent, follow in UE_IK_BONES:
            if follow is not None and follow not in by_name:
                continue
            eb = ebones.new(ik_name)
            src = rig.data.bones[by_name[follow]] if follow else None
            eb.head = src.head_local if src else Vector((0, 0, 0))
            eb.tail = src.tail_local if src else Vector((0, 0, rig.data.bones["root"].length * 0.5))
            if src:
                eb.align_roll(src.matrix_local.to_3x3().col[2])
            eb.parent = ebones.get(parent) if parent else root_bone
            eb.use_deform = False
            ik.append((ik_name, by_name[follow] if follow else None))
    bpy.ops.object.mode_set(mode='POSE')
    for src_name, dst_name in mapping.items():
        pb = game.pose.bones[dst_name]
        if tags.get(src_name) == 'SOFTLINK':
            utils.add_constraint(pb, 'COPY_LOCATION', rig, src_name)
            utils.add_constraint(pb, 'COPY_ROTATION', rig, src_name)
        else:
            utils.add_constraint(pb, 'COPY_TRANSFORMS', rig, src_name)
    for ik_name, follow in ik:
        if follow:
            utils.add_constraint(game.pose.bones[ik_name], 'COPY_TRANSFORMS', rig, follow)
    torso = utils.get_meta(rig).get("spine", {}).get("torso")
    if settings.root_motion and "root" in mapping and torso in rig.pose.bones:
        # The root carries the body's travel over the ground; deform bones keep
        # their world transforms, so the hierarchy compensates automatically.
        utils.add_constraint(game.pose.bones[mapping["root"]], 'COPY_LOCATION', rig, torso, use_z=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    merge_to = {}
    for name in merged:
        parent = export_parent(rig, name, set(keep))
        if parent:
            merge_to[name] = mapping[parent]
    return game, mapping, merge_to, tags


class _Defaults:
    """Export settings used by scripts that call the functions directly."""

    def __init__(self, **kwargs):
        defaults = dict(skeleton='UNIVERSAL', full_facial=True, metacarpals=True, naming='RIGGER', export_twist=True,
                        selected_bones=False, rename_bones="", no_parents=False, ue_ik_bones=False,
                        root_motion=False, root_y_up=False)
        defaults.update(kwargs)
        self.__dict__.update(defaults)


def copy_meshes(context, rig, game, mapping, merge_to=None, selected_only=False, apply_subsurf=True,
                triangulate=False, selected=None):
    copies = []
    selected = set(context.selected_objects) if selected is None else selected
    for obj in list(context.scene.objects):
        if obj.type != 'MESH' or utils.find_rig_for(obj) != rig:
            continue
        if selected_only and obj not in selected:
            continue
        dup = obj.copy()
        dup.data = obj.data.copy()
        context.scene.collection.objects.link(dup)
        dup.name = obj.name + "_export"
        for mod in dup.modifiers:
            if mod.type == 'ARMATURE' and mod.object == rig:
                mod.object = game
            if mod.type == 'SUBSURF' and not apply_subsurf:
                mod.show_viewport = False
        if triangulate:
            dup.modifiers.new("triangulate", 'TRIANGULATE')
        mw = obj.matrix_world.copy()
        dup.parent = game
        dup.matrix_world = mw
        # Weights of bones that are not exported go to their exported parent
        for src, dst in (merge_to or {}).items():
            vg = dup.vertex_groups.get(src)
            if vg is None:
                continue
            target = dup.vertex_groups.get(dst) or dup.vertex_groups.get(_src_of(mapping, dst)) or \
                dup.vertex_groups.new(name=dst)
            index = vg.index
            for v in dup.data.vertices:
                for g in v.groups:
                    if g.group == index and g.weight > 0:
                        target.add([v.index], g.weight, 'ADD')
            dup.vertex_groups.remove(vg)
        for vg in dup.vertex_groups:
            if vg.name in mapping:
                vg.name = mapping[vg.name]
        copies.append(dup)
    return copies


def _src_of(mapping, dst):
    return next((k for k, v in mapping.items() if v == dst), dst)


def _dummy_mesh(context, game):
    mesh = bpy.data.meshes.new("dummy")
    mesh.from_pydata([(0, 0, 0), (0.001, 0, 0), (0, 0.001, 0)], [], [(0, 1, 2)])
    obj = bpy.data.objects.new("dummy", mesh)
    context.scene.collection.objects.link(obj)
    obj.parent = game
    root = game.data.bones[0].name
    obj.vertex_groups.new(name=root).add([0, 1, 2], 1.0, 'REPLACE')
    obj.modifiers.new("Armature", 'ARMATURE').object = game
    return obj


def _rest_pose_mesh(context, game):
    """Tiny mesh with a degenerate triangle on every bone: the FBX gets every bone's bind pose."""
    verts, faces = [], []
    for bone in game.data.bones:
        h = bone.head_local
        i = len(verts)
        verts += [tuple(h), tuple(h + Vector((1e-4, 0, 0))), tuple(h + Vector((0, 1e-4, 0)))]
        faces.append((i, i + 1, i + 2))
    mesh = bpy.data.meshes.new("rest_pose")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("rest_pose", mesh)
    context.scene.collection.objects.link(obj)
    obj.parent = game
    for index, bone in enumerate(game.data.bones):
        obj.vertex_groups.new(name=bone.name).add([3 * index, 3 * index + 1, 3 * index + 2], 1.0, 'REPLACE')
    obj.modifiers.new("Armature", 'ARMATURE').object = game
    return obj


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _action_fits(action, rig):
    """True when the action animates bones of this rig (pose blend actions excluded)."""
    if action.get("rigger_pose"):
        return False
    names = set(rig.pose.bones.keys())
    for path in _action_paths(action):
        if path.startswith('pose.bones["'):
            bone = path.split('"')[1]
            if bone in names:
                return True
    return False


def _action_paths(action):
    """Data paths of every F-curve, for legacy and layered (4.4+) actions."""
    return [fc.data_path for fc in utils.action_fcurves(action)]


def export_actions(context, rig, settings):
    """Actions to export, grouped by file name suffix: [(suffix, [actions])]."""
    anim = rig.animation_data
    if settings.anim_type == 'NLA' or not settings.bake_actions:
        return []
    if settings.use_actions_manager:
        groups = []
        for lst in settings.action_lists:
            if lst.enabled:
                acts = [i.action for i in lst.actions if i.action is not None]
                groups.append((lst.name, acts))
        return groups
    if settings.only_active:
        acts = [anim.action] if anim and anim.action else []
    else:
        acts = [a for a in bpy.data.actions if _action_fits(a, rig) and a.rigger_export]
    if settings.action_search:
        acts = [a for a in acts if settings.action_search.lower() in a.name.lower()]
    return [("", acts)]


def _frame_range(action, settings):
    if settings.frame_range == 'MANUAL':
        return settings.frame_start, settings.frame_end
    if settings.frame_range == 'MARKERS':
        markers = {m.name.lower(): m.frame for m in action.pose_markers}
        if "start" in markers and "end" in markers:
            return markers["start"], markers["end"]
    start, end = action.frame_range
    return int(round(start)), int(round(end))


def bake_actions(context, rig, game, actions, settings=None, mapping=None):
    """Bake each rig action onto the game rig. Returns the baked actions."""
    settings = settings or _Defaults(reset_transforms=False, frame_range='ACTION', sampling=1.0, links=[])
    baked = []
    anim = rig.animation_data
    original = anim.action if anim else None
    links = list(getattr(settings, "links", []))
    utils.set_mode(game, 'POSE')
    for pb in game.pose.bones:
        utils.select_pose_bone(pb)
    for action in actions:
        if getattr(settings, "reset_transforms", False):
            for pb in rig.pose.bones:
                pb.matrix_basis.identity()
        utils.assign_action(rig, action)
        restore = []
        for link in links:
            if link.rig_a == rig and link.action_a == action and link.rig_b and link.action_b:
                b_anim = link.rig_b.animation_data_create()
                restore.append((link.rig_b, b_anim.action))
                utils.assign_action(link.rig_b, link.action_b)
        start, end = _frame_range(action, settings)
        game.animation_data_create()
        game.animation_data.action = None
        step = max(1, int(round(getattr(settings, "sampling", 1.0))))
        bpy.ops.nla.bake(frame_start=start, frame_end=end, step=step, only_selected=False, visual_keying=True,
                         clear_constraints=False, use_current_action=False, bake_types={'POSE'})
        result = game.animation_data.action
        if result is not None:
            result.name = action.name
            result["rigger_take"] = action.name  # the name may get a .001 suffix
            baked.append(result)
        for other, act in restore:
            utils.assign_action(other, act)
    if anim:
        utils.assign_action(rig, original)
    bpy.ops.object.mode_set(mode='OBJECT')
    return baked


def bake_nla(context, rig, game):
    """Bake the scene as the NLA plays it, as one action."""
    scene = context.scene
    utils.set_mode(game, 'POSE')
    for pb in game.pose.bones:
        utils.select_pose_bone(pb)
    game.animation_data_create()
    game.animation_data.action = None
    bpy.ops.nla.bake(frame_start=scene.frame_start, frame_end=scene.frame_end, only_selected=False,
                     visual_keying=True, clear_constraints=False, use_current_action=False, bake_types={'POSE'})
    result = game.animation_data.action
    bpy.ops.object.mode_set(mode='OBJECT')
    if result is not None:
        result.name = result["rigger_take"] = "NLA"
        return [result]
    return []


def _constant_bones(baked, names):
    for action in baked:
        for fc in utils.action_fcurves(action):
            if fc.data_path.startswith('pose.bones["') and fc.data_path.split('"')[1] in names:
                for point in fc.keyframe_points:
                    point.interpolation = 'CONSTANT'


def _scale_x100(context, game, meshes, baked):
    """Centimetres: objects and animated locations 100 times larger (scene unit scale 0.01 at export)."""
    for obj in [game] + meshes:
        obj.matrix_world = Matrix.Scale(100.0, 4) @ obj.matrix_world
    for obj in [game] + meshes:
        utils.set_mode(obj, 'OBJECT')
        with context.temp_override(selected_editable_objects=[obj], selected_objects=[obj], active_object=obj,
                                   object=obj):
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for action in baked:
        for fc in utils.action_fcurves(action):
            if fc.data_path.endswith(".location"):
                for point in fc.keyframe_points:
                    point.co.y *= 100.0
                    point.handle_left.y *= 100.0
                    point.handle_right.y *= 100.0


# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------

PRESETS = {
    'UNREAL': dict(apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE',
                   axis_forward='-Z', axis_up='Y'),
    'UNITY': dict(apply_unit_scale=True, apply_scale_options='FBX_SCALE_ALL',
                  axis_forward='-Z', axis_up='Y'),
    'GODOT': dict(apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE',
                  axis_forward='-Z', axis_up='Y'),
    'GENERIC': dict(apply_unit_scale=True, apply_scale_options='FBX_SCALE_NONE',
                    axis_forward='-Z', axis_up='Y'),
}


def _take_name(rig_name, action, settings):
    name = action.get("rigger_take", action.name)
    if settings.action_names == 'ACTION':
        return name
    return f"{rig_name}{settings.name_separator}{name}"


def _write(context, filepath, game, meshes, baked, settings, rig_name):
    utils.deselect_all(context)
    game.select_set(True)
    for obj in meshes:
        obj.select_set(True)
    context.view_layer.objects.active = game
    # Animations go through NLA strips named like the takes
    game.animation_data_create()
    anim = game.animation_data
    for track in list(anim.nla_tracks):
        anim.nla_tracks.remove(track)
    for action in baked:
        track = anim.nla_tracks.new()
        track.name = _take_name(rig_name, action, settings)
        strip = track.strips.new(track.name, int(action.frame_range[0]), action)
        strip.name = track.name
    anim.action = None
    if settings.file_format == 'FBX':
        options = dict(PRESETS.get(settings.engine, PRESETS['GENERIC']))
        bpy.ops.export_scene.fbx(
            filepath=filepath, use_selection=True, object_types={'ARMATURE', 'MESH'},
            global_scale=settings.global_scale, add_leaf_bones=False, use_armature_deform_only=False,
            bake_anim=bool(baked), bake_anim_use_all_actions=False, bake_anim_use_nla_strips=bool(baked),
            bake_anim_force_startend_keying=True, bake_anim_simplify_factor=settings.simplify,
            bake_anim_step=settings.sampling,
            mesh_smooth_type=settings.smoothing, use_mesh_modifiers=settings.apply_modifiers,
            use_triangles=settings.triangulate, colors_type=settings.vertex_colors,
            embed_textures=settings.embed_textures, path_mode='COPY' if settings.embed_textures else 'AUTO',
            bake_space_transform=settings.init_rotation, primary_bone_axis=settings.primary_axis,
            secondary_bone_axis=settings.secondary_axis, **options)
    else:
        bpy.ops.export_scene.gltf(
            filepath=filepath, export_format=settings.file_format, use_selection=True,
            export_apply=settings.apply_modifiers, export_animations=bool(baked),
            export_animation_mode='NLA_TRACKS', export_yup=True, export_def_bones=False,
            export_skins=True, export_morph=True, export_frame_step=max(1, int(settings.sampling)))


# Custom export scripts the user allowed in this Blender session (never saved in a file)
APPROVED_SCRIPTS = set()


def export_rig(context, rig, filepath, settings, run_script=True):
    """Export with scene-style settings. Returns the number of baked animations."""
    scene = context.scene
    frame = scene.frame_current
    unit_scale = scene.unit_settings.scale_length
    ext = ".fbx" if settings.file_format == 'FBX' else (".glb" if settings.file_format == 'GLB' else ".gltf")
    base, _old_ext = os.path.splitext(filepath)
    default_name = "Armature" if settings.engine == 'UNREAL' else rig.name + "_export"
    rig_name = settings.rig_name or default_name
    blocker = bpy.data.objects.get(rig_name)
    if blocker is not None:
        blocker.name = rig_name + "_rigger_tmp"
    groups = export_actions(context, rig, settings)
    selected = {o.name for o in context.selected_objects}
    if settings.one_file_per == 'ACTION':
        jobs = [(f"{base}_{bpy.path.clean_name(a.name)}{ext}", [a]) for _suffix, acts in groups for a in acts] \
            or [(base + ext, [])]
    elif settings.one_file_per == 'LIST':
        jobs = [(f"{base}_{suffix or 'actions'}{ext}", acts) for suffix, acts in groups] or [(base + ext, [])]
    else:
        jobs = [(base + ext, [a for _s, acts in groups for a in acts])]
    total = 0
    try:
        for path, actions in jobs:
            total += _export_one(context, rig, path, actions, settings, rig_name, selected, run_script)
    finally:
        if blocker is not None:
            blocker.name = rig_name
        scene.unit_settings.scale_length = unit_scale
        scene.frame_set(frame)
    return total


def _export_one(context, rig, filepath, actions, settings, rig_name, selected=None, run_script=True):
    scene = context.scene
    selected = {o.name for o in context.selected_objects} if selected is None else selected
    game, mapping, merge_to, tags = build_game_rig(context, rig, settings.naming, rig_name, settings.root_motion,
                                                   settings)
    meshes = copy_meshes(context, rig, game, mapping, merge_to, settings.selected_objects,
                         settings.apply_subsurf, settings.triangulate and settings.file_format != 'FBX',
                         {bpy.data.objects[n] for n in selected if n in bpy.data.objects}) \
        if settings.include_meshes else []
    if not meshes and settings.force_rest_pose:
        meshes = [_rest_pose_mesh(context, game)]
    elif not meshes and settings.dummy_mesh:
        meshes = [_dummy_mesh(context, game)]
    baked = []
    try:
        if run_script:
            _run_script(context, settings.custom_script, game)
        if settings.anim_type == 'NLA' and settings.bake_actions:
            baked = bake_nla(context, rig, game)
        elif actions:
            baked = bake_actions(context, rig, game, actions, settings, mapping)
        for pb in game.pose.bones:
            for con in list(pb.constraints):
                pb.constraints.remove(con)
        const = {mapping[b] for b, t in tags.items() if t == 'CONST' and b in mapping}
        if const:
            _constant_bones(baked, const)
        if settings.units_x100:
            _scale_x100(context, game, meshes, baked)
            scene.unit_settings.scale_length = 0.01
        _write(context, filepath, game, meshes, baked, settings, rig.name)
    finally:
        for obj in meshes:
            data = obj.data
            bpy.data.objects.remove(obj)
            if data is not None and data.users == 0:
                bpy.data.meshes.remove(data)
        game_data = game.data
        bpy.data.objects.remove(game)
        bpy.data.armatures.remove(game_data)
        for action in baked:
            bpy.data.actions.remove(action)
    return len(baked)


def export_fbx(context, rig, filepath, preset='UNREAL', naming='RIGGER', actions='CURRENT',
               scale=1.0, include_meshes=True, root_motion=False):
    """Script-friendly FBX export with the main options (other options: scene.rigger_export)."""
    s = context.scene.rigger_export
    saved = {k: getattr(s, k) for k in ("engine", "naming", "bake_actions", "only_active", "global_scale",
                                        "include_meshes", "root_motion", "file_format")}
    s.engine, s.naming, s.global_scale = preset, naming, scale
    s.include_meshes, s.root_motion, s.file_format = include_meshes, root_motion, 'FBX'
    s.bake_actions = actions != 'NONE'
    s.only_active = actions == 'CURRENT'
    try:
        return export_rig(context, rig, filepath, s)
    finally:
        for k, v in saved.items():
            setattr(s, k, v)


# ---------------------------------------------------------------------------
# Check / fix rig
# ---------------------------------------------------------------------------

def check_rig(context, rig, settings=None):
    """List of (severity, message) for things that export badly."""
    issues = []
    pbs = rig.pose.bones
    meta = utils.get_meta(rig)
    for limb in meta.get("limbs", []):
        prop = pbs.get(limb["prop_bone"])
        if prop is None:
            continue
        if abs(prop.get("stretch_length", 1.0) - 1.0) > 1e-4:
            issues.append(('WARNING', f"{limb['label']}: Stretch Length is not 1"))
        if prop.get("stretch", 0.0) > 1e-4:
            issues.append(('WARNING', f"{limb['label']}: Auto Stretch is on"))
        for tweak in limb.get("tweaks") or []:
            if tweak in pbs and pbs[tweak].location.length > 1e-5:
                issues.append(('WARNING', f"{tweak} is moved (stretches the deform bones)"))
    for obj in context.scene.objects:
        if obj.type == 'MESH' and utils.find_rig_for(obj) == rig:
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.use_deform_preserve_volume:
                    issues.append(('INFO', f"{obj.name}: Preserve Volume is on (game engines use linear skinning)"))
    ref = rig.rigger.reference
    if ref is not None and any(ref.pose.bones[n].rigger.soft_ik for n in ref.pose.bones.keys()
                               if ref.pose.bones[n].rigger.module in ('ARM', 'LEG')):
        issues.append(('WARNING', "Soft IK is on: the limbs are slightly stretched"))
    if (rig.matrix_world.to_scale() - Vector((1, 1, 1))).length > 1e-4:
        issues.append(('INFO', "The rig object is scaled (applied on the fly at export)"))
    if abs(context.scene.unit_settings.scale_length - 1.0) > 1e-6:
        issues.append(('WARNING', "Scene Unit Scale is not 1"))
    name = (settings.rig_name if settings else "") or ""
    if name and any(o.name == name and o.type == 'MESH' for o in context.scene.objects):
        issues.append(('ERROR', f"A mesh is named like the export skeleton '{name}'"))
    return issues


def fix_rig(context, rig):
    """Reset stretch settings and tweaks, disable Preserve Volume. Returns the number of changes."""
    pbs = rig.pose.bones
    count = 0
    for limb in utils.get_meta(rig).get("limbs", []):
        prop = pbs.get(limb["prop_bone"])
        if prop is None:
            continue
        if "stretch_length" in prop and abs(prop["stretch_length"] - 1.0) > 1e-4:
            prop["stretch_length"] = 1.0
            count += 1
        if prop.get("stretch", 0.0) > 1e-4:
            prop["stretch"] = 0.0
            count += 1
        for tweak in limb.get("tweaks") or []:
            if tweak in pbs and pbs[tweak].location.length > 1e-5:
                pbs[tweak].location = (0, 0, 0)
                count += 1
    for obj in context.scene.objects:
        if obj.type == 'MESH' and utils.find_rig_for(obj) == rig:
            for mod in obj.modifiers:
                if mod.type == 'ARMATURE' and mod.use_deform_preserve_volume:
                    mod.use_deform_preserve_volume = False
                    count += 1
    return count


class RIGGER_OT_check_rig(bpy.types.Operator):
    """Report what would export badly (stretch, preserve volume, scale...). With Fix: correct it"""
    bl_idname = "rigger.check_rig"
    bl_label = "Check Rig"
    bl_options = {'REGISTER', 'UNDO'}

    fix: BoolProperty(name="Fix", default=False)

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def execute(self, context):
        rig = utils.find_rig_for(context.object)
        if self.fix:
            count = fix_rig(context, rig)
            self.report({'INFO'}, f"Fixed {count} settings")
            return {'FINISHED'}
        issues = check_rig(context, rig, context.scene.rigger_export)
        for level, msg in issues:
            self.report({level}, msg)
        if not issues:
            self.report({'INFO'}, "No issues found")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _setting_keys():
    from .properties import RiggerExportSettings
    return [p for p in RiggerExportSettings.__annotations__ if p not in ("action_lists", "links", "last_filepath",
                                                                         "action_list_index", "link_index",
                                                                         "action_index", "ui_tab")]


class RIGGER_OT_export_fbx(ExportHelper, bpy.types.Operator):
    """Export the rig as a clean game skeleton (root + deform bones) with meshes and animation"""
    bl_idname = "rigger.export_fbx"
    bl_label = "Export Game Rig"
    filename_ext = ".fbx"
    filter_glob: StringProperty(default="*.fbx;*.glb;*.gltf", options={'HIDDEN'})

    preset: EnumProperty(name="Engine", default='UNREAL', items=[
        ('UNREAL', "Unreal Engine", "Armature named 'Armature' so Unreal does not add an extra root"),
        ('UNITY', "Unity", "Scale applied to all (no 100x scale on import)"),
        ('GODOT', "Godot", ""),
        ('GENERIC', "Generic", "Other engines and DCCs"),
    ], options={'SKIP_SAVE'})
    naming: EnumProperty(name="Bone Names", default='RIGGER', items=NAMING_ITEMS, options={'SKIP_SAVE'})
    actions: EnumProperty(name="Animation", default='CURRENT', items=[
        ('NONE', "None", "Skeleton and meshes only"),
        ('CURRENT', "Current Action", "Bake the rig's active action"),
        ('ALL', "All Actions", "Bake every action that animates this rig"),
    ], options={'SKIP_SAVE'})
    scale: FloatProperty(name="Scale", default=1.0, min=0.001, max=1000.0, options={'SKIP_SAVE'})
    include_meshes: BoolProperty(name="Include Meshes", default=True, options={'SKIP_SAVE'})
    root_motion: BoolProperty(name="Root Motion", default=False, options={'SKIP_SAVE'},
                              description="Move the root bone with the body over the ground (engine root motion)")
    run_script: BoolProperty(
        name="Run Custom Script", default=True, options={'SKIP_SAVE'},
        description="Run the custom export script named in the export settings. It can do anything Python "
                    "can: only allow scripts you trust (the export dialog asks every session)")

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def invoke(self, context, event):
        # A script name can come from a downloaded .blend: the dialog asks before running it
        script = context.scene.rigger_export.custom_script
        self.run_script = not script or script in APPROVED_SCRIPTS
        # The file browser uses the extension of the format chosen in the settings
        fmt = context.scene.rigger_export.file_format
        type(self).filename_ext = {'FBX': ".fbx", 'GLB': ".glb"}.get(fmt, ".gltf")
        return ExportHelper.invoke(self, context, event)

    def check(self, context):
        fmt = context.scene.rigger_export.file_format
        type(self).filename_ext = {'FBX': ".fbx", 'GLB': ".glb"}.get(fmt, ".gltf")
        return ExportHelper.check(self, context)

    def draw(self, context):
        from . import ui
        script = context.scene.rigger_export.custom_script
        if script:
            box = self.layout.box()
            box.label(text=f"This file runs a script on export: {script}", icon='ERROR')
            box.prop(self, "run_script")
        ui.draw_export_settings(self.layout, context)

    def execute(self, context):
        rig = utils.find_rig_for(context.object)
        if context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        s = context.scene.rigger_export
        props = self.properties
        if props.is_property_set("preset"):
            s.engine = self.preset
        if props.is_property_set("naming"):
            s.naming = self.naming
        if props.is_property_set("actions"):
            s.bake_actions = self.actions != 'NONE'
            s.only_active = self.actions == 'CURRENT'
        if props.is_property_set("scale"):
            s.global_scale = self.scale
        if props.is_property_set("include_meshes"):
            s.include_meshes = self.include_meshes
        if props.is_property_set("root_motion"):
            s.root_motion = self.root_motion
        ext = os.path.splitext(self.filepath)[1].lower()
        if ext in (".glb", ".gltf") and s.file_format == 'FBX':
            s.file_format = 'GLB' if ext == ".glb" else 'GLTF_SEPARATE'
        script = s.custom_script
        if script and self.run_script:
            APPROVED_SCRIPTS.add(script)
        count = export_rig(context, rig, self.filepath, s, run_script=self.run_script)
        s.last_filepath = self.filepath
        if script and not self.run_script:
            self.report({'WARNING'}, f"Custom export script '{script}' skipped")
        utils.set_mode(rig, 'OBJECT')
        self.report({'INFO'}, f"Exported '{rig.name}' with {count} animation(s) to {self.filepath}")
        return {'FINISHED'}


class RIGGER_OT_quick_export(bpy.types.Operator):
    """Export again to the last file with the current settings"""
    bl_idname = "rigger.quick_export"
    bl_label = "Quick Export"

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None and context.scene.rigger_export.last_filepath

    def execute(self, context):
        s = context.scene.rigger_export
        rig = utils.find_rig_for(context.object)
        if context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        allowed = not s.custom_script or s.custom_script in APPROVED_SCRIPTS
        count = export_rig(context, rig, bpy.path.abspath(s.last_filepath), s, run_script=allowed)
        self.report({'INFO'}, f"Exported {count} animation(s) to {s.last_filepath}")
        if not allowed:
            self.report({'WARNING'}, f"Custom export script '{s.custom_script}' skipped: allow it once in the "
                                     "Export dialog")
        return {'FINISHED'}


def _presets_dir():
    from .rig_data import library_dir
    return library_dir("export_presets")


def _preset_items(_self, _context):
    folder = _presets_dir()
    items = [(f[:-5], f[:-5], "") for f in sorted(os.listdir(folder)) if f.endswith(".json")]
    return items or [('NONE', "(no export presets)", "")]


class RIGGER_OT_export_preset_save(bpy.types.Operator):
    """Save the export settings as a preset"""
    bl_idname = "rigger.export_preset_save"
    bl_label = "Save Export Preset"

    name: StringProperty(name="Name", default="unreal")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        s = context.scene.rigger_export
        data = {k: getattr(s, k) for k in _setting_keys()}
        with open(os.path.join(_presets_dir(), bpy.path.clean_name(self.name) + ".json"), "w") as f:
            json.dump(data, f, indent=1)
        return {'FINISHED'}


class RIGGER_OT_export_preset_load(bpy.types.Operator):
    """Load export settings from a preset"""
    bl_idname = "rigger.export_preset_load"
    bl_label = "Load Export Preset"

    preset: EnumProperty(name="Preset", items=_preset_items)

    def execute(self, context):
        path = os.path.join(_presets_dir(), self.preset + ".json")
        if not os.path.exists(path):
            return {'CANCELLED'}
        with open(path) as f:
            data = json.load(f)
        s = context.scene.rigger_export
        for k, v in data.items():
            try:
                setattr(s, k, v)
            except (AttributeError, TypeError, ValueError):
                pass
        return {'FINISHED'}


class RIGGER_OT_export_preset_remove(bpy.types.Operator):
    """Delete an export preset"""
    bl_idname = "rigger.export_preset_remove"
    bl_label = "Remove Export Preset"

    preset: EnumProperty(name="Preset", items=_preset_items)

    def execute(self, context):
        path = os.path.join(_presets_dir(), self.preset + ".json")
        if os.path.exists(path):
            os.remove(path)
        return {'FINISHED'}


class RIGGER_OT_export_actions_select(bpy.types.Operator):
    """Tick or untick every action of the rig for export"""
    bl_idname = "rigger.export_actions_select"
    bl_label = "Select Actions"
    bl_options = {'REGISTER', 'UNDO'}

    state: BoolProperty(name="Export", default=True)

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def execute(self, context):
        rig = utils.find_rig_for(context.object)
        search = context.scene.rigger_export.action_search.lower()
        for action in bpy.data.actions:
            if _action_fits(action, rig) and search in action.name.lower():
                action.rigger_export = self.state
        return {'FINISHED'}


class RIGGER_OT_action_list_edit(bpy.types.Operator):
    """Add or remove action lists, or add the rig's actions to the active list"""
    bl_idname = "rigger.action_list_edit"
    bl_label = "Edit Action Lists"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(items=[('ADD_LIST', "Add List", ""), ('REMOVE_LIST', "Remove List", ""),
                              ('ADD_ACTIONS', "Add Actions", "Add the rig's actions matching the search"),
                              ('CLEAR', "Clear List", "")])

    def execute(self, context):
        s = context.scene.rigger_export
        if self.mode == 'ADD_LIST':
            lst = s.action_lists.add()
            lst.name = f"Actions {len(s.action_lists)}"
            s.action_list_index = len(s.action_lists) - 1
            return {'FINISHED'}
        if not 0 <= s.action_list_index < len(s.action_lists):
            return {'CANCELLED'}
        if self.mode == 'REMOVE_LIST':
            s.action_lists.remove(s.action_list_index)
            s.action_list_index = max(0, s.action_list_index - 1)
            return {'FINISHED'}
        lst = s.action_lists[s.action_list_index]
        if self.mode == 'CLEAR':
            lst.actions.clear()
            return {'FINISHED'}
        rig = utils.find_rig_for(context.object)
        if rig is None:
            return {'CANCELLED'}
        present = {i.action for i in lst.actions}
        for action in bpy.data.actions:
            if _action_fits(action, rig) and s.action_search.lower() in action.name.lower() and action not in present:
                lst.actions.add().action = action
        return {'FINISHED'}


class RIGGER_OT_action_link_edit(bpy.types.Operator):
    """Add or remove an Actions Linker relation"""
    bl_idname = "rigger.action_link_edit"
    bl_label = "Edit Action Links"
    bl_options = {'REGISTER', 'UNDO'}

    remove: BoolProperty(default=False)

    def execute(self, context):
        s = context.scene.rigger_export
        if self.remove:
            if 0 <= s.link_index < len(s.links):
                s.links.remove(s.link_index)
                s.link_index = max(0, s.link_index - 1)
        else:
            s.links.add()
            s.link_index = len(s.links) - 1
        return {'FINISHED'}


classes = (RIGGER_OT_export_fbx, RIGGER_OT_quick_export, RIGGER_OT_check_rig, RIGGER_OT_export_preset_save,
           RIGGER_OT_export_preset_load, RIGGER_OT_export_preset_remove, RIGGER_OT_export_actions_select,
           RIGGER_OT_action_list_edit, RIGGER_OT_action_link_edit)
