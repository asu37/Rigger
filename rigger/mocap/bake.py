# SPDX-License-Identifier: GPL-3.0-or-later
"""Motion file -> keyed action on a Rigger rig.

Retarget: every joint's world rotation change from rest (the motion is stored
rest-aligned) is applied to the rest orientation of the control the joint
drives. If the rig the motion was solved on differs from this one, each bone's
rest direction is first turned onto the other's. Controls are solved parent
first, each against its nearest keyed ancestor control:

    world_c = D_j @ C_j @ Rest_c
    basis_c = (Rest_p^-1 @ Rest_c)^-1 @ world_p^-1 @ world_c

which holds on Rigger rigs because the deform bones between two controls copy
the controls' transforms. The root moves through the torso control.

Bake: whole f-curves at once (keyframe_points.add + foreach_set, linear), no
frame stepping, on a new action so the existing animation is kept.
"""

import bpy
import numpy as np

from .. import utils
from . import filters
from .skeleton import Skeleton, from_rig, matrix_to_quat, quat_to_matrix, rotation_between

LINEAR = 1  # Keyframe.interpolation enum value for 'LINEAR'


def world_rotations(parents, local_q, root_q):
    """(F,J,4) parent-local + (F,4) root -> (F,J,3,3) world (rest-aligned)."""
    L = quat_to_matrix(local_q)
    G = np.empty_like(L)
    for j, p in enumerate(parents):
        G[:, j] = quat_to_matrix(root_q) if p < 0 else G[:, p] @ L[:, j]
    return G


def rest_alignment(src, tgt):
    """Per target joint: rotation taking its rest bone direction onto the source's (identity if equal)."""
    C = np.tile(np.eye(3), (len(tgt.joints), 1, 1))
    for j, name in enumerate(tgt.joints):
        if name not in src.joints or tgt.roles[j] == "pelvis":
            continue
        s = src.index(name)
        dt, ds = tgt.end_point(j), src.end_point(s)
        if dt is None or ds is None:
            continue
        a, b = dt - tgt.rest[j], ds - src.rest[s]
        if np.linalg.norm(a) > 1e-6 and np.linalg.norm(b) > 1e-6:
            C[j] = rotation_between(a, b)
    return C


def _copied_from(rig, name):
    """The bone whose transform this one copies outright, if any."""
    for con in rig.pose.bones[name].constraints:
        if (con.type == 'COPY_TRANSFORMS' and con.influence >= 0.999 and not con.mute
                and getattr(con, "target", None) == rig and con.subtarget in rig.data.bones):
            return con.subtarget
    return None


def _effective_parent(rig, name, keyed):
    """Nearest keyed control the bone follows, through parents and deform bones that copy a control."""
    b = rig.data.bones[name].parent
    seen = set()
    while b is not None and b.name not in keyed and b.name not in seen:
        seen.add(b.name)
        src = _copied_from(rig, b.name)
        b = rig.data.bones[src] if src else b.parent
    return b.name if b is not None and b.name in keyed else None


def _rest(rig, name):
    return np.array(rig.data.bones[name].matrix_local.to_3x3())


def solve_controls(rig, sk_src, arrays, sk_tgt=None):
    """-> ({control: (F,4) basis quaternions}, torso location (F,3) or None, torso name)."""
    tgt = sk_tgt or from_rig(rig)
    G = world_rotations(sk_src.parents, arrays["local_q"], arrays["root_q"])
    C = rest_alignment(sk_src, tgt)
    F = len(G)
    world = {}
    order = []
    for j, ctrl in enumerate(tgt.controls):
        if not ctrl or ctrl not in rig.pose.bones or tgt.joints[j] not in sk_src.joints:
            continue
        s = sk_src.index(tgt.joints[j])
        world[ctrl] = G[:, s] @ C[j] @ _rest(rig, ctrl)
        order.append(ctrl)
    order.sort(key=lambda n: len(rig.data.bones[n].parent_recursive))
    keyed = set(order)
    basis = {}
    posed = {}
    for ctrl in order:
        p = _effective_parent(rig, ctrl, keyed)
        Rc = _rest(rig, ctrl)
        if p is None:
            rel = Rc
            Wp = np.eye(3)
        else:
            Rp = _rest(rig, p)
            rel = Rp.T @ Rc
            Wp = posed[p]
        b = rel.T @ np.swapaxes(Wp, -1, -2) @ world[ctrl]
        basis[ctrl] = filters.continuous_quats(matrix_to_quat(b))
        posed[ctrl] = Wp @ rel @ quat_to_matrix(basis[ctrl])
    # root position through the torso control
    root_j = tgt.parents.index(-1)
    torso = tgt.controls[root_j]
    loc = None
    if torso in posed:
        src_root = sk_src.rest[sk_src.parents.index(-1)]
        scale = tgt.rest[root_j][2] / max(src_root[2], 1e-6)
        pelvis = arrays["root_t"] * scale + np.array((tgt.rest[root_j][0], tgt.rest[root_j][1], 0.0))
        head = np.array(rig.data.bones[torso].head_local)
        D = posed[torso] @ _rest(rig, torso).T          # world change of the torso
        pos = pelvis - np.einsum('fij,j->fi', D, tgt.rest[root_j] - head)
        loc = (pos - head) @ _rest(rig, torso)   # Rest_torso^T @ offset per frame (its parent stays at rest)
    return basis, loc, torso


def _fcurve(action, rig, path, index, group):
    if hasattr(action, "fcurve_ensure_for_datablock"):
        return action.fcurve_ensure_for_datablock(rig, path, index=index, group_name=group)
    fc = action.fcurves.find(path, index=index)
    return fc or action.fcurves.new(path, index=index, action_group=group)


def _write(fc, frames, values):
    fc.keyframe_points.clear()
    n = len(frames)
    fc.keyframe_points.add(n)
    co = np.empty(2 * n, np.float32)
    co[0::2] = frames
    co[1::2] = values
    fc.keyframe_points.foreach_set("co", co)
    fc.keyframe_points.foreach_set("interpolation", np.full(n, LINEAR, np.int32))
    fc.update()


def bake(context, rig, meta, skeleton, arrays, start=1, fingers=True, face=True, simplify=0.0,
         set_scene=True, camera=False, video=None, name=None, video2=None):
    """Key the motion on a new action. Returns the action."""
    sk_src = Skeleton.from_dict(skeleton)
    basis, loc, torso = solve_controls(rig, sk_src, arrays)
    if not fingers:
        finger_ctrls = {c for j, c in enumerate(sk_src.controls) if sk_src.roles[j] in ("finger1", "finger", "thumb")}
        basis = {c: q for c, q in basis.items() if c not in finger_ctrls}
    F = len(arrays["root_t"])
    frames = np.arange(F, dtype=np.float64) + start
    action = bpy.data.actions.new(name or f"{rig.name}_video")
    utils.assign_action(rig, action)
    for ctrl, q in basis.items():
        rig.pose.bones[ctrl].rotation_mode = 'QUATERNION'
        keep = filters.rdp(q, simplify) if simplify > 0 else np.arange(F)
        for i in range(4):
            _write(_fcurve(action, rig, f'pose.bones["{ctrl}"].rotation_quaternion', i, ctrl), frames[keep], q[keep, i])
    if loc is not None:
        keep = filters.rdp(loc, simplify * 0.5) if simplify > 0 else np.arange(F)
        for i in range(3):
            _write(_fcurve(action, rig, f'pose.bones["{torso}"].location', i, torso), frames[keep], loc[keep, i])
    for limb in utils.get_meta(rig).get("limbs", []):  # the motion drives the FK chains
        prop = limb.get("prop_bone")
        if prop and prop in rig.pose.bones and "ik_fk" in rig.pose.bones[prop]:
            rig.pose.bones[prop]["ik_fk"] = 0.0
            _write(_fcurve(action, rig, f'pose.bones["{prop}"]["ik_fk"]', 0, prop), frames[:1], np.zeros(1))
    if face and "face" in arrays and meta.get("face"):
        bake_face(rig, meta["face"], arrays["face"], frames)
    if set_scene:
        scene = context.scene
        fps = float(meta.get("fps", 30.0))
        scene.render.fps = max(1, int(round(fps)))
        scene.render.fps_base = scene.render.fps / fps
        scene.frame_start, scene.frame_end = int(start), int(start + F - 1)
    if camera and "camera_K" in arrays:
        make_camera(context, rig, meta, arrays, sk_src, start, video)
        if "camera2_K" in arrays:   # second camera of a two-camera solve
            make_camera(context, rig, meta, arrays, sk_src, start, video2, second=True)
    return action


# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------

def _norm(name):
    return name.lower().replace("_", "").replace(".", "").replace(" ", "")


def bake_face(rig, names, values, frames):
    """Expression weights onto shape keys of the rig's meshes whose names match (ARKit naming)."""
    wanted = {_norm(n): i for i, n in enumerate(names)}
    for alias, base in (("eyeblinkl", "eyeblinkleft"), ("eyeblinkr", "eyeblinkright"),
                        ("mouthsmilel", "mouthsmileleft"), ("mouthsmiler", "mouthsmileright")):
        if base in wanted:
            wanted[alias] = wanted[base]
    count = 0
    for obj in bpy.data.objects:
        if obj.type != 'MESH' or not obj.data.shape_keys:
            continue
        if not any(m.type == 'ARMATURE' and m.object == rig for m in obj.modifiers) and obj.parent != rig:
            continue
        key = obj.data.shape_keys
        act = None
        for kb in key.key_blocks[1:]:
            i = wanted.get(_norm(kb.name))
            if i is None:
                continue
            if act is None:
                act = bpy.data.actions.new(f"{obj.name}_video_face")
                utils.assign_action(key, act)
            _write(_fcurve(act, key, f'key_blocks["{kb.name}"].value', 0, "face"), frames, values[:, i])
            count += 1
    return count


# ---------------------------------------------------------------------------
# Camera matching the video
# ---------------------------------------------------------------------------

def make_camera(context, rig, meta, arrays, sk_src, start, video=None, second=False):
    key = "camera2" if second else "camera"
    f, cx, cy = (float(v) for v in arrays[key + "_K"])
    w, h = meta["image_size2" if second else "image_size"]
    W = np.asarray(arrays[key + "_W"])
    tgt = from_rig(rig)
    root_j = tgt.parents.index(-1)
    scale = tgt.rest[root_j][2] / max(sk_src.rest[sk_src.parents.index(-1)][2], 1e-6)
    pos = np.asarray(arrays[key + "_pos"]) * scale + np.array((tgt.rest[root_j][0], tgt.rest[root_j][1], 0.0))
    from mathutils import Matrix
    name = "VideoCamera 2" if second else "VideoCamera"
    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_fit = 'HORIZONTAL'
    cam_data.sensor_width = 36.0
    cam_data.lens = f * 36.0 / w
    cam_data.shift_x = (cx - w / 2) / w
    cam_data.shift_y = -(cy - h / 2) / w
    cam = bpy.data.objects.new(name, cam_data)
    context.scene.collection.objects.link(cam)
    axes = np.column_stack([W[:, 0], -W[:, 1], -W[:, 2]])
    m = Matrix.Identity(4)
    for r in range(3):
        for c in range(3):
            m[r][c] = axes[r, c]
        m[r][3] = pos[r]
    cam.matrix_world = rig.matrix_world @ m
    if not second:
        context.scene.render.resolution_x, context.scene.render.resolution_y = int(w), int(h)
        context.scene.camera = cam
    if video:
        try:
            clip = bpy.data.movieclips.load(video, check_existing=True)
            bg = cam_data.background_images.new()
            bg.source = 'MOVIE_CLIP'
            bg.clip = clip
            bg.alpha = 0.7
            clip.frame_start = int(start) - (int(round(meta.get("offset2", 0.0) * meta["fps"])) if second else 0)
            cam_data.show_background_images = True
        except RuntimeError:
            pass
    return cam
