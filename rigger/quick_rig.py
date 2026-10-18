# SPDX-License-Identifier: GPL-3.0-or-later
"""Quick Rig: put a Rigger control rig on an existing skeleton (Mixamo, game rigs, any armature).

1. Guess the limbs from the bone names (or fill the list by hand from selected bones).
2. Build a reference armature whose bones sit on the skeleton's bones, and generate the rig.
3. Preserve mode: the skeleton keeps its names and weights and follows the rig's deform bones.
   Convert mode: the meshes' vertex groups are renamed to the rig's deform bones and moved to the rig.
4. Optionally bake the skeleton's action onto the control rig.
"""

import json
import re
from types import SimpleNamespace

import bpy
from bpy.props import StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Vector

from . import generate, templates, utils
from .retarget import FINGER_NAMES, Retargeter, classify, classify_armature

_END = re.compile(r"(end|nub|top|tip)(_?\d*)$", re.IGNORECASE)
SUFFIX = {"L": ".L", "R": ".R", "C": ""}
QR_PREFIX = "QR-"
QR_COLLECTION = "Quick Rig"
OPTION_KEYS = ("mode", "orphans", "ignore_root", "same_origin", "hide_base", "bake", "delete_source",
               "twist", "fingers_ik", "secondary")


def bone_list(limb):
    return [n.strip() for n in limb.bones.split(",") if n.strip()]


# ---------------------------------------------------------------------------
# Guessing
# ---------------------------------------------------------------------------

def guess_limbs(obj):
    """Return [(kind, side, label, [source bones], metacarpal first)] guessed from the bone names."""
    groups = classify_armature(obj)
    get = lambda role, side="", finger=None: groups.get((role, side, finger), [])
    limbs = []
    spine = get("hips")[:1] + get("spine")
    if len(spine) >= 2:
        limbs.append(('SPINE', 'C', "spine", spine))
    neck = get("neck") + get("head")[:1]
    if neck:
        limbs.append(('NECK', 'C', "neck", neck))
    for side in ("L", "R"):
        arm = get("shoulder", side)[:1] + get("upper_arm", side)[:1] + get("forearm", side)[:1] + \
            get("hand", side)[:1]
        if len(arm) >= 3 and get("hand", side):
            limbs.append(('ARM', side, "arm", arm))
        leg = get("thigh", side)[:1] + get("shin", side)[:1] + get("foot", side)[:1] + get("toe", side)[:1]
        if len(leg) >= 3 and get("foot", side):
            limbs.append(('LEG', side, "leg", leg))
        for finger in FINGER_NAMES:
            bones = get("finger", side, finger)
            meta = get("metacarpal", side, finger)[:1]
            if bones:
                limbs.append(('FINGER', side, finger, meta + bones, bool(meta)))
    return [limb if len(limb) == 5 else limb + (False,) for limb in limbs]


def fill_limbs(settings):
    settings.limbs.clear()
    for kind, side, label, bones, meta in guess_limbs(settings.armature):
        item = settings.limbs.add()
        item.kind, item.side, item.label, item.bones = kind, side, label, ", ".join(bones)
        item.metacarpal = meta
    return len(settings.limbs)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def _clean(name):
    """Source name without its namespace ('mixamorig:Hips' -> 'Hips')."""
    return name.split(":")[-1].split("|")[-1] or name


def ref_names(limb):
    """Reference bone names of a limb, matching its source bones one to one."""
    bones = bone_list(limb)
    sfx = SUFFIX[limb.side]
    count = len(bones)
    if limb.kind == 'SPINE':
        return ["hips"] + [f"spine.{i:02d}" for i in range(1, count)]
    if limb.kind == 'NECK':
        necks = count - 1
        names = ["neck"] if necks == 1 else [f"neck.{i:02d}" for i in range(1, necks + 1)]
        return names + ["head"]
    if limb.kind == 'ARM':
        names = ["shoulder", "upper_arm", "forearm", "hand"][4 - min(count, 4):]
        return [n + sfx for n in names]
    if limb.kind == 'LEG':
        return [n + sfx for n in ("thigh", "shin", "foot", "toe")[:count]]
    label = limb.label or _clean(bones[0]).rstrip("0123456789._- ") or "chain"
    if limb.kind == 'SIMPLE' or count == 1:
        return [label + sfx]
    if limb.kind == 'FINGER' and limb.metacarpal:
        return [f"{label}_meta{sfx}"] + [f"{label}.{i:02d}{sfx}" for i in range(1, count)]
    return [f"{label}.{i:02d}{sfx}" for i in range(1, count + 1)]


def _has_weights(meshes, name):
    for mesh in meshes:
        vg = mesh.vertex_groups.get(name)
        if vg is None:
            continue
        for v in mesh.data.vertices:
            for g in v.groups:
                if g.group == vg.index and g.weight > 1e-4:
                    return True
    return False


def skinned_meshes(armature):
    return [o for o in bpy.data.objects if o.type == 'MESH' and (
        o.parent == armature or any(m.type == 'ARMATURE' and m.object == armature for m in o.modifiers))]


class QuickRig:
    def __init__(self, context, settings):
        self.context = context
        self.s = settings
        self.src = settings.armature
        if self.src is None or self.src.type != 'ARMATURE':
            raise ValueError("Pick the armature to rig")
        self.meshes = skinned_meshes(self.src)
        self.limbs = [l for l in settings.limbs if bone_list(l)]
        if not self.limbs:
            raise ValueError("The limb list is empty: use Guess Limbs or add limbs")
        bones = self.src.data.bones
        self.map = {}       # source bone -> reference bone
        self.roots = {}     # reference root bone -> (module, params)
        self.chains = []    # [(ref names, source names, limb)]
        existing = set()
        for limb in self.limbs:
            sources = bone_list(limb)
            missing = [n for n in sources if n not in bones]
            if missing:
                raise ValueError(f"Bones not in '{self.src.name}': {', '.join(missing)}")
            if limb.kind == 'ARM' and len(sources) < 3:
                raise ValueError("An arm needs at least the upper arm, forearm and hand")
            if limb.kind == 'LEG' and len(sources) < 3:
                raise ValueError("A leg needs at least the thigh, shin and foot")
            if limb.kind == 'SPINE' and len(sources) < 2:
                raise ValueError("The spine needs the hips and at least one spine bone")
            names = [utils.unique_name(existing, n) for n in ref_names(limb)]
            existing.update(names)
            for src, ref in zip(sources, names):
                if src in self.map:
                    raise ValueError(f"'{src}' is in two limbs")
                self.map[src] = ref
            self.chains.append((names, sources, limb))
        # Root bone to ignore: top-most bone that is in no limb
        self.ignored = set()
        if settings.ignore_root:
            for bone in bones:
                if bone.parent is None and bone.name not in self.map:
                    self.ignored.add(bone.name)
        # Orphans: bones in no limb, with weights or with included children
        self.orphans = []
        if settings.orphans:
            for bone in sorted(bones, key=lambda b: len(b.parent_recursive)):
                name = bone.name
                if name in self.map or name in self.ignored:
                    continue
                leaf_end = not bone.children and (_END.search(_clean(name)) or name.lower().endswith("_end"))
                if leaf_end and not _has_weights(self.meshes, name):
                    continue
                if classify(name) is not None and not _has_weights(self.meshes, name):
                    continue  # e.g. an unused second hand bone
                ref = utils.unique_name(existing, _clean(name))
                existing.add(ref)
                self.map[name] = ref
                self.orphans.append(name)

    # -- reference -------------------------------------------------------------
    def _mapped_parent(self, name):
        bone = self.src.data.bones[name].parent
        while bone is not None:
            if bone.name in self.map:
                return self.map[bone.name]
            bone = bone.parent
        return None

    def _world(self, name):
        bone = self.src.data.bones[name]
        mw = self.src.matrix_world
        z = (mw.to_3x3() @ bone.z_axis).normalized()
        return mw @ bone.head_local, mw @ bone.tail_local, z

    def build_reference(self):
        context, src = self.context, self.src
        context.view_layer.update()
        data = bpy.data.armatures.new(f"{src.name}_reference")
        ref = bpy.data.objects.new(f"{src.name}_reference", data)
        context.collection.objects.link(ref)
        ref[utils.REF_KEY] = True
        ref.show_in_front = True
        if self.s.same_origin:
            loc, rot, _scale = src.matrix_world.decompose()
            ref.matrix_world = Matrix.LocRotScale(loc, rot, None)
        context.view_layer.update()
        inv = ref.matrix_world.inverted()
        inv3 = inv.to_3x3()
        utils.set_mode(ref, 'EDIT')
        ebones = data.edit_bones
        h = max((max(v.z for v in (self._world(n)[0], self._world(n)[1])) for n in self.map), default=1.0) - \
            min(self._world(n)[0].z for n in self.map)
        min_len = max(h, 0.1) * 0.01

        def add(ref_name, src_name, next_src=None):
            head, tail, z = self._world(src_name)
            if next_src is not None:
                tail = self._world(next_src)[0]
            else:
                end = next((c for c in self.src.data.bones[src_name].children if c.name not in self.map
                            and (self._world(c.name)[0] - head).length > min_len), None)
                if end is not None and not self.src.data.bones[src_name].children[1:]:
                    tail = self._world(end.name)[0]
            if (tail - head).length < min_len:
                parent = self.src.data.bones[src_name].parent
                direction = (head - self._world(parent.name)[0]).normalized() if parent else Vector((0, 0, 1))
                if direction.length < 0.5:
                    direction = Vector((0, 0, 1))
                tail = head + direction * min_len * 5
            eb = ebones.new(ref_name)
            eb.head, eb.tail = inv @ head, inv @ tail
            eb.align_roll(inv3 @ z)
            return eb

        for names, sources, limb in self.chains:
            for i, (ref_name, src_name) in enumerate(zip(names, sources)):
                nxt = sources[i + 1] if i + 1 < len(sources) else None
                if limb.kind == 'ARM' and i == 0 and len(sources) == 4:
                    nxt = None  # the clavicle keeps its own length
                add(ref_name, src_name, nxt)
        for name in self.orphans:
            add(self.map[name], name)
        # Parenting: chains in order, limb roots to the nearest mapped ancestor
        for names, sources, limb in self.chains:
            for i, ref_name in enumerate(names):
                eb = ebones[ref_name]
                parent = names[i - 1] if i else self._mapped_parent(sources[0])
                if parent is not None and parent in ebones:
                    eb.parent = ebones[parent]
                    eb.use_connect = i > 0 and (eb.parent.tail - eb.head).length < 1e-5
        for name in self.orphans:
            parent = self._mapped_parent(name)
            if parent is not None:
                ebones[self.map[name]].parent = ebones[parent]
        bpy.ops.object.mode_set(mode='OBJECT')
        twist = self.s.twist
        for names, sources, limb in self.chains:
            params = {}
            if limb.kind in ('ARM', 'LEG'):
                params = {"twist": twist, "twist_upper": max(0, twist - 1), "start_ik": limb.kind == 'LEG',
                          "fans": False, "secondary": self.s.secondary}
            elif limb.kind == 'FINGER':
                params = {"finger_ik": self.s.fingers_ik}
            module = 'SIMPLE' if limb.kind == 'CHAIN' and len(names) == 1 else limb.kind
            if limb.kind == 'FINGER' and limb.metacarpal and len(names) > 1:
                templates.apply_settings(ref.pose.bones[names[0]], 'SIMPLE', {"collection": "Fingers"})
                templates.apply_settings(ref.pose.bones[names[1]], 'FINGER', params)
                continue
            templates.apply_settings(ref.pose.bones[names[0]], module, params)
            if limb.kind == 'NECK':
                ref.pose.bones[names[0]].rigger.module = 'NECK'
        for name in self.orphans:
            templates.apply_settings(ref.pose.bones[self.map[name]], 'SIMPLE', {})
        ref.rigger.rig_name = f"{src.name}_rig"
        return ref

    # -- control rig -----------------------------------------------------------
    def control_of(self, rig):
        """Source bone -> (control, is location control) for baking."""
        meta = utils.get_meta(rig)
        controls = set(meta.get("controls", []))
        ref_ctrl = {}
        spine = meta.get("spine")
        neck = meta.get("neck")
        for names, _sources, limb in self.chains:
            if limb.kind == 'SPINE' and spine:
                ref_ctrl[names[0]] = (spine["torso"], True)
                for ref_name, ctrl in zip(names[1:], spine["fk"]):
                    ref_ctrl[ref_name] = (ctrl, False)
            elif limb.kind == 'NECK' and neck:
                for ref_name, ctrl in zip(names[:-1], neck["fk"]):
                    ref_ctrl[ref_name] = (ctrl, False)
                ref_ctrl[names[-1]] = (neck["head"], False)
        for limb in meta.get("limbs", []):
            for ref_name, ctrl in zip(limb.get("ref", []), limb["fk"]):
                ref_ctrl[ref_name] = (ctrl, False)
        for names, _sources, limb in self.chains:
            if limb.kind == 'ARM' and len(names) == 4:
                arm = next((m for m in meta.get("limbs", []) if names[1] in m.get("ref", [])), None)
                if arm and arm.get("shoulder"):
                    ref_ctrl[names[0]] = (arm["shoulder"], False)
            if limb.kind == 'LEG' and len(names) == 4:
                leg = next((m for m in meta.get("limbs", []) if names[0] in m.get("ref", [])), None)
                if leg and leg.get("toe_fk"):
                    ref_ctrl[names[3]] = (leg["toe_fk"], False)
        result = {}
        for src, ref in self.map.items():
            if ref in ref_ctrl:
                result[src] = ref_ctrl[ref]
            elif ref in controls:
                result[src] = (ref, False)
        return result

    def bake(self, rig):
        src = self.src
        action = src.animation_data.action if src.animation_data else None
        if action is None:
            return None
        items = [SimpleNamespace(source=s, target=c, location=loc, location_mode='WORLD', loc_multiplier=1.0,
                                 rot_offset=(0.0, 0.0, 0.0), loc_offset=(0.0, 0.0, 0.0))
                 for s, (c, loc) in self.control_of(rig).items()]
        settings = SimpleNamespace(
            source=src, target=rig, items=items, frame_start=-1, frame_end=-1, source_frame=-1,
            use_target_pose=False, scale=1.0, in_place=False, legs_ik=False, arms_ik=False,
            namespace_from="", namespace_to="", existing_keys=False, pole_mode='ABSOLUTE',
            smooth_poles=False, clean_fk=False, action_name=f"{action.name}_rig", multiple=False,
            root_motion=False)
        new_action = Retargeter(self.context, settings).run()
        action.use_fake_user = True
        return new_action

    def deform_map(self, rig):
        """Source vertex group -> rig deform bone (Convert mode)."""
        meta = utils.get_meta(rig)
        deform_of = meta.get("deform_of", {})
        result = {}
        for bone in self.src.data.bones:
            name = bone.name
            walk = bone
            while walk is not None and walk.name not in self.map:
                walk = walk.parent
            if walk is not None and self.map[walk.name] in deform_of:
                result[name] = deform_of[self.map[walk.name]]
        for item in self.s.overrides:
            if not item.source:
                continue
            target = deform_of.get(item.target, item.target)
            if target in rig.data.bones:
                result[item.source] = target
        return result

    def convert(self, rig):
        mapping = self.deform_map(rig)
        for mesh in self.meshes:
            merge_groups(mesh, mapping)
            for mod in mesh.modifiers:
                if mod.type == 'ARMATURE' and mod.object == self.src:
                    mod.object = rig
            if mesh.parent == self.src:
                world = mesh.matrix_world.copy()
                mesh.parent = rig
                mesh.parent_type = 'OBJECT'
                mesh.matrix_world = world
        return mapping

    def preserve(self, rig):
        """Helper bones on the rig with the skeleton's rest matrices; the skeleton copies them."""
        context, src = self.context, self.src
        meta = utils.get_meta(rig)
        deform_of = meta.get("deform_of", {})
        context.view_layer.update()
        to_rig = rig.matrix_world.inverted()
        followed = {s: deform_of[r] for s, r in self.map.items() if r in deform_of}
        utils.set_mode(rig, 'EDIT')
        ebones = rig.data.edit_bones
        coll = rig.data.collections.get(QR_COLLECTION) or rig.data.collections.new(QR_COLLECTION)
        for name, deform in followed.items():
            head, tail, z = self._world(name)
            helper = ebones.get(QR_PREFIX + name) or ebones.new(QR_PREFIX + name)
            helper.head, helper.tail = to_rig @ head, to_rig @ tail
            if (helper.tail - helper.head).length < 1e-5:
                helper.tail = helper.head + Vector((0, 0, 0.01))
            helper.align_roll(to_rig.to_3x3() @ z)
            helper.parent = ebones[deform]
            helper.use_deform = False
            coll.assign(helper)
        bpy.ops.object.mode_set(mode='OBJECT')
        coll.is_visible = False
        if src.animation_data and src.animation_data.action:
            src.animation_data.action.use_fake_user = True
            src.animation_data.action = None
        for name in followed:
            pb = src.pose.bones[name]
            for con in list(pb.constraints):
                if con.name.startswith("QuickRig"):
                    pb.constraints.remove(con)
            pb.matrix_basis = Matrix()
            target = QR_PREFIX + name
            con = pb.constraints.new('COPY_LOCATION')
            con.name, con.target, con.subtarget = "QuickRig Location", rig, target
            con = pb.constraints.new('COPY_ROTATION')
            con.name, con.target, con.subtarget = "QuickRig Rotation", rig, target
            con = pb.constraints.new('COPY_SCALE')
            con.name, con.target, con.subtarget = "QuickRig Scale", rig, target
            con.owner_space = con.target_space = 'POSE'
        src["rigger_quick_rig"] = rig.name
        if self.s.hide_base:
            src.hide_set(True)
        return followed

    def run(self, report=None):
        context = self.context
        frame = context.scene.frame_current
        ref = self.build_reference()
        rig = generate.generate(context, ref)
        ref.hide_set(True)
        rig["rigger_quick_source"] = self.src.name
        baked = self.bake(rig) if self.s.bake else None
        context.scene.frame_set(frame)
        if self.s.mode == 'CONVERT':
            self.convert(rig)
            if self.s.delete_source:
                bpy.data.objects.remove(self.src)
        else:
            self.preserve(rig)
        if baked is not None:
            utils.assign_action(rig, baked)
        context.scene.frame_set(frame)
        utils.set_mode(rig, 'POSE')
        if report:
            extra = f", baked '{baked.name}'" if baked else ""
            report({'INFO'}, f"Quick Rig: {len(self.map)} bones, {len(self.orphans)} orphans{extra}")
        return rig, ref


def merge_groups(mesh, mapping):
    """Rename vertex groups by 'mapping'; groups mapped to the same bone are added together."""
    groups = {vg.index: vg.name for vg in mesh.vertex_groups}
    if not any(n in mapping and mapping[n] != n for n in groups.values()):
        return
    sums = {}
    for v in mesh.data.vertices:
        for g in v.groups:
            target = mapping.get(groups.get(g.group))
            if target is not None and g.weight > 0.0:
                bucket = sums.setdefault(target, {})
                bucket[v.index] = bucket.get(v.index, 0.0) + g.weight
    for name in list(groups.values()):
        if name in mapping:
            mesh.vertex_groups.remove(mesh.vertex_groups[name])
    for target, weights in sums.items():
        vg = mesh.vertex_groups.get(target) or mesh.vertex_groups.new(name=target)
        for index, weight in weights.items():
            vg.add([index], min(weight, 1.0), 'ADD')


def quick_rig(context, settings, report=None):
    return QuickRig(context, settings).run(report)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def preset_data(settings):
    return {
        "limbs": [{"kind": l.kind, "side": l.side, "label": l.label, "bones": bone_list(l),
                   "metacarpal": l.metacarpal} for l in settings.limbs],
        "overrides": [{"source": o.source, "target": o.target} for o in settings.overrides],
        "options": {k: getattr(settings, k) for k in OPTION_KEYS},
    }


def load_preset(settings, data):
    settings.limbs.clear()
    for entry in data.get("limbs", []):
        item = settings.limbs.add()
        item.kind = entry.get("kind", 'CHAIN')
        item.side = entry.get("side", 'C')
        item.label = entry.get("label", "")
        item.metacarpal = bool(entry.get("metacarpal", False))
        bones = entry.get("bones", [])
        item.bones = ", ".join(bones) if isinstance(bones, list) else str(bones)
    settings.overrides.clear()
    for entry in data.get("overrides", []):
        item = settings.overrides.add()
        item.source, item.target = entry.get("source", ""), entry.get("target", "")
    for key, value in data.get("options", {}).items():
        if key in OPTION_KEYS:
            setattr(settings, key, value)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class _QuickPoll:
    @classmethod
    def poll(cls, context):
        s = context.scene.rigger_quick
        return s.armature is not None


class RIGGER_OT_quick_rig_guess(_QuickPoll, bpy.types.Operator):
    """Fill the limb list by guessing the spine, neck, arms, legs and fingers from the bone names"""
    bl_idname = "rigger.quick_rig_guess"
    bl_label = "Guess Limbs"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        count = fill_limbs(context.scene.rigger_quick)
        self.report({'INFO'} if count else {'WARNING'}, f"Found {count} limbs")
        return {'FINISHED'}


class RIGGER_OT_quick_rig_limb_add(bpy.types.Operator):
    """Add a limb to the Quick Rig list"""
    bl_idname = "rigger.quick_rig_limb_add"
    bl_label = "Add Limb"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_quick
        s.limbs.add()
        s.active_index = len(s.limbs) - 1
        return {'FINISHED'}


class RIGGER_OT_quick_rig_limb_remove(bpy.types.Operator):
    """Remove the active limb from the Quick Rig list"""
    bl_idname = "rigger.quick_rig_limb_remove"
    bl_label = "Remove Limb"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.rigger_quick.limbs) > 0

    def execute(self, context):
        s = context.scene.rigger_quick
        s.limbs.remove(s.active_index)
        s.active_index = max(0, min(s.active_index, len(s.limbs) - 1))
        return {'FINISHED'}


class RIGGER_OT_quick_rig_from_selection(_QuickPoll, bpy.types.Operator):
    """Set the bones of the active limb from the bones selected on the armature, ordered parent first"""
    bl_idname = "rigger.quick_rig_from_selection"
    bl_label = "Set From Selected Bones"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_quick
        arm = s.armature
        if not s.limbs:
            s.limbs.add()
            s.active_index = 0
        if arm.mode == 'EDIT':
            names = [b.name for b in arm.data.edit_bones if b.select]
        else:
            names = [pb.name for pb in arm.pose.bones if utils.is_pose_bone_selected(pb)]
        if not names:
            self.report({'WARNING'}, f"Select bones of '{arm.name}' first")
            return {'CANCELLED'}
        names.sort(key=lambda n: (len(arm.data.bones[n].parent_recursive), n))
        s.limbs[s.active_index].bones = ", ".join(names)
        return {'FINISHED'}


class RIGGER_OT_quick_rig_override_add(bpy.types.Operator):
    """Add a weights override: send a vertex group to another bone (Convert mode)"""
    bl_idname = "rigger.quick_rig_override_add"
    bl_label = "Add Weights Override"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_quick
        s.overrides.add()
        s.override_index = len(s.overrides) - 1
        return {'FINISHED'}


class RIGGER_OT_quick_rig_override_remove(bpy.types.Operator):
    """Remove the active weights override"""
    bl_idname = "rigger.quick_rig_override_remove"
    bl_label = "Remove Weights Override"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return len(context.scene.rigger_quick.overrides) > 0

    def execute(self, context):
        s = context.scene.rigger_quick
        s.overrides.remove(s.override_index)
        s.override_index = max(0, min(s.override_index, len(s.overrides) - 1))
        return {'FINISHED'}


class RIGGER_OT_quick_rig(_QuickPoll, bpy.types.Operator):
    """Build a reference on the skeleton, generate the control rig and connect the skeleton to it"""
    bl_idname = "rigger.quick_rig"
    bl_label = "Quick Rig"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.rigger_quick
        if not s.limbs:
            fill_limbs(s)
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            quick_rig(context, s, self.report)
        except (ValueError, generate.RiggerError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        if s.mode == 'CONVERT' and s.delete_source:
            s.armature = None
        return {'FINISHED'}


class RIGGER_OT_quick_rig_preset_export(ExportHelper, bpy.types.Operator):
    """Save the limb list, weights overrides and options as a Quick Rig preset"""
    bl_idname = "rigger.quick_rig_preset_export"
    bl_label = "Export Quick Rig Preset"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        with open(self.filepath, "w", encoding="utf-8") as handle:
            json.dump(preset_data(context.scene.rigger_quick), handle, indent=2)
        return {'FINISHED'}


class RIGGER_OT_quick_rig_preset_import(ImportHelper, bpy.types.Operator):
    """Load a Quick Rig preset"""
    bl_idname = "rigger.quick_rig_preset_import"
    bl_label = "Import Quick Rig Preset"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={'HIDDEN'})

    def execute(self, context):
        try:
            with open(self.filepath, encoding="utf-8") as handle:
                load_preset(context.scene.rigger_quick, json.load(handle))
        except (OSError, ValueError) as exc:
            self.report({'ERROR'}, f"Invalid preset: {exc}")
            return {'CANCELLED'}
        return {'FINISHED'}


classes = (RIGGER_OT_quick_rig_guess, RIGGER_OT_quick_rig_limb_add, RIGGER_OT_quick_rig_limb_remove,
           RIGGER_OT_quick_rig_from_selection, RIGGER_OT_quick_rig_override_add,
           RIGGER_OT_quick_rig_override_remove, RIGGER_OT_quick_rig, RIGGER_OT_quick_rig_preset_export,
           RIGGER_OT_quick_rig_preset_import)
