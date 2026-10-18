# SPDX-License-Identifier: GPL-3.0-or-later
"""Reference armature templates.

Templates are lists of ``BoneSpec`` for the character's left side and centre.
Right side bones are mirrored automatically. Characters face -Y with Z up.

The human template is computed from a handful of landmark points so the same
code builds the default human and fits a reference to placed markers.
"""

import math
from dataclasses import dataclass, field, replace

import bpy
from mathutils import Vector

from . import utils

UP = Vector((0, 0, 1))
FORWARD = Vector((0, -1, 0))


@dataclass
class BoneSpec:
    name: str
    head: Vector
    tail: Vector
    parent: str = None
    connect: bool = False
    z: Vector = None
    module: str = 'NONE'
    params: dict = field(default_factory=dict)


# Default landmarks of a 1.8 m human in T-pose.
DEFAULT_POINTS = {
    "chin": Vector((0.0, -0.075, 1.585)),
    "neck": Vector((0.0, 0.0, 1.47)),
    "shoulder": Vector((0.185, 0.0, 1.43)),
    "wrist": Vector((0.70, 0.0, 1.43)),
    "pelvis": Vector((0.0, 0.0, 0.98)),
    "ankle": Vector((0.10, 0.03, 0.085)),
    "top": 1.80,
    "floor": 0.0,
}

MARKERS = (
    ("chin", "Chin", "Front view: click the bottom of the chin", False),
    ("neck", "Neck", "Front view: click the base of the neck", False),
    ("shoulder", "Shoulder", "Front view: click the left shoulder joint", True),
    ("wrist", "Wrist", "Front view: click the left wrist", True),
    ("pelvis", "Spine Root", "Front view: click the pelvis / root of the spine", False),
    ("ankle", "Ankle", "Front view: click the left ankle", True),
)

# (name, base offset along hand, spread, up) and length, all in units of height
FINGERS = (
    ("thumb", (0.012, 0.022, -0.008), 0.034),
    ("index", (0.055, 0.017, 0.0), 0.045),
    ("middle", (0.056, 0.004, 0.0), 0.049),
    ("ring", (0.054, -0.009, 0.0), 0.045),
    ("pinky", (0.050, -0.020, 0.0), 0.036),
)


DEFAULT_OPTIONS = {
    "spine_count": 3, "neck_count": 1, "twist": 2, "fingers": 5, "toes": False,
    "metacarpals": False, "toes_metatarsal": False, "toes_parent_foot": False,
    "lips": 1, "lids": 1, "brows": 3,
    "eyes": True, "eyebrows": True, "nose": True, "mouth": True, "tongue": True, "teeth": True,
    "chin": True, "cheeks": True, "ears": False,
}

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def human_specs(points=None, fingers=True, face=True, options=None):
    """Human reference. options: see DEFAULT_OPTIONS (bone counts and face parts)."""
    o = dict(DEFAULT_OPTIONS)
    if options:
        o.update(options)
    p = dict(DEFAULT_POINTS)
    if points:
        p.update(points)
    chin, neck, shoulder, wrist = p["chin"], p["neck"], p["shoulder"], p["wrist"]
    pelvis, ankle, top, floor = p["pelvis"], p["ankle"], p["top"], p["floor"]
    h = max(top - floor, 0.01)
    specs = []
    add = specs.append

    # Spine
    hips_head = Vector((0, pelvis.y, pelvis.z))
    hips_tail = hips_head + Vector((0, 0, 0.055 * h))
    add(BoneSpec("hips", hips_head, hips_tail, z=FORWARD, module='SPINE'))
    neck_base = Vector((0, neck.y, neck.z))
    prev, parent = hips_tail, "hips"
    count = max(1, int(o["spine_count"]))
    spine_y = p.get("spine_y") or {}  # measured depth of inner spine joints (fit to mesh)
    for i in range(1, count + 1):
        pt = hips_tail.lerp(neck_base, i / count)
        if i in spine_y:
            pt.y = spine_y[i]
        name = f"spine.{i:02d}"
        add(BoneSpec(name, prev, pt, parent, True, FORWARD))
        prev, parent = pt, name
    chest = parent

    # Neck & head
    neck_top = Vector((0, neck.y + (chin.y - neck.y) * 0.25, chin.z + 0.012 * h))
    necks = max(1, int(o["neck_count"]))
    prev, parent = neck_base, chest
    for i in range(necks):
        name = "neck" if necks == 1 else f"neck.{i + 1:02d}"
        pt = neck_base.lerp(neck_top, (i + 1) / necks)
        add(BoneSpec(name, prev, pt, parent, True, FORWARD, 'NECK' if i == 0 else 'NONE'))
        prev, parent = pt, name
    add(BoneSpec("head", neck_top, Vector((0, neck_top.y, top)), parent, True, FORWARD))
    if face:
        face_specs(add, chin, neck_top, top, h, p.get("face_front"), p.get("eye_front"), p.get("eye"), o)

    # Arm
    clav_head = Vector((0.012 * h, neck.y - 0.008 * h, shoulder.z + 0.004 * h))
    twist = int(o["twist"])
    add(BoneSpec("shoulder.L", clav_head, shoulder.copy(), chest, False, UP, 'ARM',
                 {"start_ik": False, "twist": twist, "twist_upper": max(0, twist - 1)}))
    elbow = p["elbow"].copy() if "elbow" in p else shoulder.lerp(wrist, 0.5) + Vector((0, 0.012 * h, 0))
    add(BoneSpec("upper_arm.L", shoulder.copy(), elbow, "shoulder.L", True, UP))
    add(BoneSpec("forearm.L", elbow, wrist.copy(), "upper_arm.L", True, UP))
    arm_dir = (wrist - shoulder).normalized()
    hand_up = (UP - arm_dir * arm_dir.dot(UP)).normalized()
    spread = arm_dir.cross(hand_up)  # points to the front of the hand for the left side
    k = p.get("hand_scale", 1.0)  # measured hand size relative to the default
    add(BoneSpec("hand.L", wrist.copy(), wrist + arm_dir * 0.052 * h * k, "forearm.L", True, UP))

    if fingers:
        keep = FINGER_ORDER[:max(1, min(5, int(o["fingers"])))]
        for fname, (along, side, up), length in FINGERS:
            if fname not in keep:
                continue
            base = wrist + (arm_dir * along + spread * side + hand_up * up) * h * k
            if fname == "thumb":
                direction = (arm_dir * 0.55 + spread * 0.7 - hand_up * 0.3).normalized()
            else:
                direction = arm_dir
            seg = (0.40, 0.33, 0.27)
            prev, parent = base, "hand.L"
            if o["metacarpals"] and fname != "thumb":
                # Palm bone from near the wrist to the knuckle
                meta_head = wrist + (arm_dir * 0.012 + spread * side * 0.6 + hand_up * up) * h * k
                add(BoneSpec(f"{fname}_meta.L", meta_head, base.copy(), "hand.L", False, hand_up, 'SIMPLE',
                             {"collection": "Fingers"}))
                parent = f"{fname}_meta.L"
            done = 0.0
            for i, s in enumerate(seg, 1):
                # Slight upward arc: IK chains need a bend to know which way to fold
                done += s
                arc = hand_up * length * h * k * 0.03 * math.sin(math.pi * done) if i < len(seg) else Vector()
                nxt = base + direction * length * h * done * k + arc
                name = f"{fname}.{i:02d}.L"
                add(BoneSpec(name, prev, nxt, parent, i > 1 or parent.endswith("_meta.L"), hand_up,
                             'FINGER' if i == 1 else 'NONE'))
                prev, parent = nxt, name

    # Leg
    hip_x = max(abs(ankle.x), abs(shoulder.x) * 0.45)
    thigh_head = p["thigh_head"].copy() if "thigh_head" in p else Vector((hip_x, pelvis.y, pelvis.z - 0.02 * h))
    ankle_pt = Vector((abs(ankle.x), ankle.y, ankle.z))
    knee = p["knee"].copy() if "knee" in p else thigh_head.lerp(ankle_pt, 0.5) + Vector((0, -0.01 * h, 0))
    if "toe_tip" in p:  # measured foot length: ball at 72% from the ankle to the tip
        tip = p["toe_tip"]
        ball = Vector((ankle_pt.x + (tip.x - ankle_pt.x) * 0.72, ankle_pt.y + (tip.y - ankle_pt.y) * 0.72,
                       floor + 0.012 * h))
        toe_tip = Vector((tip.x, tip.y, ball.z))
    else:
        ball = Vector((ankle_pt.x, ankle_pt.y - 0.075 * h, floor + 0.012 * h))
        toe_tip = ball + Vector((0, -0.035 * h, 0))
    add(BoneSpec("thigh.L", thigh_head, knee, "hips", False, FORWARD, 'LEG',
                 {"start_ik": True, "twist": twist, "twist_upper": max(0, twist - 1)}))
    add(BoneSpec("shin.L", knee, ankle_pt, "thigh.L", True, FORWARD))
    add(BoneSpec("foot.L", ankle_pt, ball, "shin.L", True, UP))
    add(BoneSpec("toe.L", ball, toe_tip, "foot.L", True, UP))
    if o["toes"]:
        # Individual toes on top of the main toe bone, big toe on the inner side
        width = 0.03 * h
        for i, tname in enumerate(FINGER_ORDER):
            x = ankle_pt.x - width * 0.5 + width * i / 4.0
            base = Vector((x, ball.y - 0.004 * h, ball.z))
            length = 0.03 * h * (1.0 - 0.12 * i)
            prev, parent = base, ("foot.L" if o["toes_parent_foot"] else "toe.L")
            if o["toes_metatarsal"]:
                meta_head = Vector((x * 0.6 + ankle_pt.x * 0.4, ankle_pt.y - 0.03 * h, (ankle_pt.z + ball.z) * 0.5))
                add(BoneSpec(f"toe_{tname}_meta.L", meta_head, base.copy(), parent, False, UP, 'SIMPLE',
                             {"collection": "Fingers"}))
                parent = f"toe_{tname}_meta.L"
            for j, seg in enumerate((0.5, 0.5) if i else (0.55, 0.45)):
                nxt = prev + Vector((0, -length * seg, -0.002 * h * j))
                name = f"toe_{tname}.{j + 1:02d}.L"
                add(BoneSpec(name, prev, nxt, parent, j > 0 or parent.endswith("_meta.L"), UP,
                             'FINGER' if j == 0 else 'NONE',
                             {"curl": False} if j == 0 else {}))
                prev, parent = nxt, name
    return specs


def face_specs(add, chin, neck_top, top, h, face_front=None, eye_front=None, eye=None, options=None):
    """Eyes plus a face module: jaw, lips, eyelids, brows, cheeks, nose, teeth, tongue.

    face_front / eye_front: Y of the face surface at the mouth / in front of the
    left eye when measured on the mesh; estimated from the chin otherwise.
    """
    eye_z = chin.z + (top - chin.z) * 0.47
    mouth_z = chin.z + 0.026 * h
    front = face_front + 0.003 * h if face_front is not None else chin.y - 0.016 * h
    eye_y = eye_front + 0.0075 * h if eye_front is not None else chin.y - 0.004 * h
    eye_x = 0.018 * h
    if eye is not None:  # centre of a detected left eyeball
        eye_x, eye_y, eye_z = abs(eye.x), eye.y, eye.z
    small = Vector((0, -0.007 * h, 0))  # face controls point forward
    o = dict(DEFAULT_OPTIONS)
    if options:
        o.update(options)

    def bone(name, pos, parent, module='NONE', params=None):
        add(BoneSpec(name, pos, pos + small, parent, False, UP, module, params or {}))

    eye_head = Vector((eye_x, eye_y, eye_z))
    if o["eyes"]:
        add(BoneSpec("eye.L", eye_head, eye_head + Vector((0, -0.012 * h, 0)), "head", False, UP, 'EYE'))
    face_head = Vector((0, neck_top.y, eye_z))
    add(BoneSpec("face", face_head, face_head + Vector((0, 0, 0.03 * h)), "head", False, FORWARD, 'FACE'))
    if o["mouth"] or o["chin"] or o["teeth"] or o["tongue"]:
        jaw_tip = Vector((0, front + 0.006 * h, chin.z + 0.004 * h))
        add(BoneSpec("jaw", Vector((0, jaw_tip.y + 0.05 * h, mouth_z + 0.01 * h)), jaw_tip, "face", False, UP))
    if o["chin"]:
        bone("chin", Vector((0, front + 0.002 * h, chin.z + 0.006 * h)), "jaw")
    if o["mouth"]:
        bone("lip_upper", Vector((0, front, mouth_z + 0.005 * h)), "face")
        bone("lip_lower", Vector((0, front, mouth_z - 0.005 * h)), "jaw")
        lips = max(1, int(o["lips"]))
        corner_x = 0.02 * h
        for i in range(1, lips + 1):
            t = i / (lips + 1)
            index = "" if lips == 1 else f".{i:02d}"
            for row, sign, parent in (("upper", 1.0, "face"), ("lower", -1.0, "jaw")):
                bone(f"lip_{row}{index}.L", Vector((corner_x * t, front + 0.008 * h * t * t,
                                                    mouth_z + sign * 0.005 * h * (1.0 - t * t) ** 0.5)), parent)
        bone("lip_corner.L", Vector((corner_x, front + 0.008 * h, mouth_z)), "face")
    if o["eyes"]:
        lids = max(1, int(o["lids"]))
        for i in range(1, lids + 1):
            u = (2.0 * i / (lids + 1) - 1.0) if lids > 1 else 0.0
            index = "" if lids == 1 else f".{i:02d}"
            x = eye_x + u * 0.009 * h
            y = eye_y - 0.008 * h * (1.0 - 0.5 * u * u)
            for row, sign in (("upper", 1.0), ("lower", -1.0)):
                bone(f"lid_{row}{index}.L", Vector((x, y, eye_z + sign * 0.005 * h * (1.0 - u * u) ** 0.5)), "face")
    if o["eyebrows"]:
        brows = int(o["brows"])
        if brows == 3:
            spots = (("brow_inner.L", 0.007, 0.021, 0.0), ("brow_mid.L", 0.019, 0.025, 0.002),
                     ("brow_outer.L", 0.031, 0.020, 0.010))
        else:
            spots = []
            for i in range(brows):
                t = i / max(brows - 1, 1)
                spots.append((f"brow.{i + 1:02d}.L", 0.007 + 0.024 * t, 0.021 + 0.004 * (1 - (2 * t - 1) ** 2), 0.01 * t))
        for name, x, z, y in spots:
            bone(name, Vector((x * h, eye_y - 0.008 * h + y * h, eye_z + z * h)), "face")
    if o["cheeks"]:
        bone("cheek.L", Vector((0.03 * h, front + 0.008 * h, mouth_z + 0.024 * h)), "face")
    if o["nose"]:
        bone("nose", Vector((0, front - 0.01 * h, eye_z - 0.022 * h)), "face")
    if o["teeth"]:
        bone("teeth_upper", Vector((0, front + 0.004 * h, mouth_z + 0.004 * h)), "face")
        bone("teeth_lower", Vector((0, front + 0.004 * h, mouth_z - 0.004 * h)), "jaw")
    if o["tongue"]:
        start = Vector((0, chin.y + 0.012 * h, mouth_z - 0.002 * h))
        end = Vector((0, front + 0.006 * h, mouth_z - 0.002 * h))
        for i in range(3):
            add(BoneSpec(f"tongue.{i + 1:02d}", start.lerp(end, i / 3), start.lerp(end, (i + 1) / 3),
                         "jaw" if i == 0 else f"tongue.{i:02d}", i > 0, UP))
    if o["ears"]:
        ear = Vector((0.075 * h * 0.92, neck_top.y + 0.004 * h, eye_z - 0.002 * h))
        mid = ear + Vector((0.008 * h, 0.006 * h, 0.012 * h))
        tip = mid + Vector((0.004 * h, 0.006 * h, 0.014 * h))
        add(BoneSpec("ear.01.L", ear, mid, "head", False, FORWARD, 'CHAIN', {"curl": False, "collection": "Face"}))
        add(BoneSpec("ear.02.L", mid, tip, "ear.01.L", True, FORWARD))


def quadruped_specs():
    V = Vector
    s = []
    add = s.append
    add(BoneSpec("hips", V((0, 0.30, 0.62)), V((0, 0.22, 0.64)), z=UP, module='SPINE'))
    pts = [V((0, 0.22, 0.64)), V((0, 0.08, 0.65)), V((0, -0.06, 0.65)), V((0, -0.18, 0.64))]
    for i in range(3):
        add(BoneSpec(f"spine.{i + 1:02d}", pts[i], pts[i + 1], "hips" if i == 0 else f"spine.{i:02d}",
                     True, UP))
    add(BoneSpec("neck", V((0, -0.18, 0.64)), V((0, -0.28, 0.78)), "spine.03", True, UP, 'NECK'))
    add(BoneSpec("head", V((0, -0.28, 0.78)), V((0, -0.46, 0.74)), "neck", True, UP))
    add(BoneSpec("jaw", V((0, -0.30, 0.74)), V((0, -0.44, 0.70)), "head", False, UP, 'SIMPLE',
                 {"collection": "Face"}))
    add(BoneSpec("eye.L", V((0.04, -0.36, 0.82)), V((0.04, -0.38, 0.82)), "head", False, UP, 'EYE'))
    tail = [V((0, 0.30, 0.62)), V((0, 0.40, 0.61)), V((0, 0.48, 0.57)), V((0, 0.55, 0.51)), V((0, 0.60, 0.44))]
    for i in range(4):
        add(BoneSpec(f"tail.{i + 1:02d}", tail[i], tail[i + 1], "hips" if i == 0 else f"tail.{i:02d}",
                     i > 0, UP, 'CHAIN' if i == 0 else 'NONE'))
    # Front legs
    add(BoneSpec("front_thigh.L", V((0.08, -0.16, 0.58)), V((0.08, -0.13, 0.36)), "spine.03", False,
                 FORWARD, 'LEG', {"start_ik": True, "twist": 0}))
    add(BoneSpec("front_shin.L", V((0.08, -0.13, 0.36)), V((0.08, -0.16, 0.10)), "front_thigh.L", True, FORWARD))
    add(BoneSpec("front_foot.L", V((0.08, -0.16, 0.10)), V((0.08, -0.21, 0.03)), "front_shin.L", True, UP))
    add(BoneSpec("front_toe.L", V((0.08, -0.21, 0.03)), V((0.08, -0.26, 0.02)), "front_foot.L", True, UP))
    # Hind legs
    add(BoneSpec("thigh.L", V((0.08, 0.26, 0.60)), V((0.08, 0.20, 0.40)), "hips", False, FORWARD, 'LEG',
                 {"start_ik": True, "twist": 0}))
    add(BoneSpec("shin.L", V((0.08, 0.20, 0.40)), V((0.08, 0.30, 0.20)), "thigh.L", True, FORWARD))
    add(BoneSpec("foot.L", V((0.08, 0.30, 0.20)), V((0.08, 0.27, 0.04)), "shin.L", True, FORWARD))
    add(BoneSpec("toe.L", V((0.08, 0.27, 0.04)), V((0.08, 0.22, 0.02)), "foot.L", True, UP))
    return s


def horse_specs():
    V = Vector
    s = []
    add = s.append
    add(BoneSpec("hips", V((0, 0.55, 1.35)), V((0, 0.40, 1.40)), z=UP, module='SPINE'))
    pts = [V((0, 0.40, 1.40)), V((0, 0.10, 1.42)), V((0, -0.20, 1.42)), V((0, -0.45, 1.45))]
    for i in range(3):
        add(BoneSpec(f"spine.{i + 1:02d}", pts[i], pts[i + 1], "hips" if i == 0 else f"spine.{i:02d}",
                     True, UP))
    neck = [V((0, -0.45, 1.45)), V((0, -0.62, 1.62)), V((0, -0.76, 1.82))]
    add(BoneSpec("neck.01", neck[0], neck[1], "spine.03", True, UP, 'NECK'))
    add(BoneSpec("neck.02", neck[1], neck[2], "neck.01", True, UP))
    add(BoneSpec("head", V((0, -0.76, 1.82)), V((0, -1.05, 1.55)), "neck.02", True, UP))
    add(BoneSpec("jaw", V((0, -0.82, 1.66)), V((0, -1.02, 1.47)), "head", False, UP, 'SIMPLE',
                 {"collection": "Face"}))
    add(BoneSpec("eye.L", V((0.09, -0.86, 1.74)), V((0.11, -0.89, 1.74)), "head", False, UP, 'EYE'))
    for i, (name, a, b) in enumerate((("ear.L", V((0.06, -0.78, 1.88)), V((0.08, -0.76, 2.00))),)):
        add(BoneSpec(name, a, b, "head", False, FORWARD, 'CHAIN', {"curl": False}))
    tail = [V((0, 0.58, 1.33)), V((0, 0.70, 1.20)), V((0, 0.76, 1.02)), V((0, 0.78, 0.84))]
    for i in range(3):
        add(BoneSpec(f"tail.{i + 1:02d}", tail[i], tail[i + 1], "hips" if i == 0 else f"tail.{i:02d}",
                     i > 0, UP, 'CHAIN' if i == 0 else 'NONE'))
    # Front legs: shoulder blade is part of the spine, leg = upper, forearm, cannon, hoof
    add(BoneSpec("front_thigh.L", V((0.16, -0.40, 1.20)), V((0.16, -0.36, 0.80)), "spine.03", False,
                 FORWARD, 'LEG', {"start_ik": True, "twist": 0}))
    add(BoneSpec("front_shin.L", V((0.16, -0.36, 0.80)), V((0.16, -0.38, 0.40)), "front_thigh.L", True, FORWARD))
    add(BoneSpec("front_foot.L", V((0.16, -0.38, 0.40)), V((0.16, -0.40, 0.10)), "front_shin.L", True, FORWARD))
    add(BoneSpec("front_toe.L", V((0.16, -0.40, 0.10)), V((0.16, -0.46, 0.02)), "front_foot.L", True, UP))
    # Hind legs: thigh, gaskin, cannon, hoof
    add(BoneSpec("thigh.L", V((0.17, 0.48, 1.22)), V((0.17, 0.36, 0.85)), "hips", False, FORWARD, 'LEG',
                 {"start_ik": True, "twist": 0}))
    add(BoneSpec("shin.L", V((0.17, 0.36, 0.85)), V((0.17, 0.52, 0.52)), "thigh.L", True, FORWARD))
    add(BoneSpec("foot.L", V((0.17, 0.52, 0.52)), V((0.17, 0.48, 0.10)), "shin.L", True, FORWARD))
    add(BoneSpec("toe.L", V((0.17, 0.48, 0.10)), V((0.17, 0.42, 0.02)), "foot.L", True, UP))
    return s


def bird_specs():
    V = Vector
    s = []
    add = s.append
    add(BoneSpec("hips", V((0, 0.10, 0.42)), V((0, 0.02, 0.46)), z=UP, module='SPINE'))
    add(BoneSpec("spine.01", V((0, 0.02, 0.46)), V((0, -0.06, 0.50)), "hips", True, UP))
    add(BoneSpec("spine.02", V((0, -0.06, 0.50)), V((0, -0.12, 0.55)), "spine.01", True, UP))
    neck = [V((0, -0.12, 0.55)), V((0, -0.14, 0.61)), V((0, -0.14, 0.67)), V((0, -0.15, 0.72))]
    add(BoneSpec("neck.01", neck[0], neck[1], "spine.02", True, FORWARD, 'NECK'))
    add(BoneSpec("neck.02", neck[1], neck[2], "neck.01", True, FORWARD))
    add(BoneSpec("head", neck[2], neck[3], "neck.02", True, FORWARD))
    add(BoneSpec("beak_upper", V((0, -0.17, 0.705)), V((0, -0.23, 0.70)), "head", False, UP, 'SIMPLE',
                 {"collection": "Face"}))
    add(BoneSpec("jaw", V((0, -0.165, 0.695)), V((0, -0.22, 0.688)), "head", False, UP, 'SIMPLE',
                 {"collection": "Face"}))
    add(BoneSpec("eye.L", V((0.022, -0.16, 0.715)), V((0.03, -0.165, 0.715)), "head", False, UP, 'EYE'))
    # Wings use the arm module: shoulder, upper wing, forearm, hand (primaries)
    add(BoneSpec("shoulder.L", V((0.02, -0.08, 0.53)), V((0.07, -0.07, 0.54)), "spine.02", False, UP, 'ARM',
                 {"start_ik": False, "twist": 1}))
    add(BoneSpec("upper_arm.L", V((0.07, -0.07, 0.54)), V((0.20, -0.06, 0.55)), "shoulder.L", True, UP))
    add(BoneSpec("forearm.L", V((0.20, -0.06, 0.55)), V((0.34, -0.065, 0.55)), "upper_arm.L", True, UP))
    add(BoneSpec("hand.L", V((0.34, -0.065, 0.55)), V((0.48, -0.06, 0.55)), "forearm.L", True, UP))
    add(BoneSpec("feathers.01.L", V((0.48, -0.06, 0.55)), V((0.56, -0.05, 0.55)), "hand.L", True, UP, 'CHAIN',
                 {"curl": False, "collection": "Fingers"}))
    tail = [V((0, 0.10, 0.42)), V((0, 0.18, 0.40)), V((0, 0.27, 0.38))]
    for i in range(2):
        add(BoneSpec(f"tail.{i + 1:02d}", tail[i], tail[i + 1], "hips" if i == 0 else f"tail.{i:02d}",
                     i > 0, UP, 'CHAIN' if i == 0 else 'NONE'))
    # Legs: thigh, shin (drumstick), tarsus, toes
    add(BoneSpec("thigh.L", V((0.05, 0.04, 0.40)), V((0.06, -0.02, 0.30)), "hips", False, FORWARD, 'LEG',
                 {"start_ik": True, "twist": 0}))
    add(BoneSpec("shin.L", V((0.06, -0.02, 0.30)), V((0.06, 0.05, 0.16)), "thigh.L", True, FORWARD))
    add(BoneSpec("foot.L", V((0.06, 0.05, 0.16)), V((0.06, 0.02, 0.025)), "shin.L", True, FORWARD))
    add(BoneSpec("toe.L", V((0.06, 0.02, 0.025)), V((0.06, -0.06, 0.01)), "foot.L", True, UP))
    return s


def scene_options():
    """Template options from the scene (Rigger > Reference > Template Options), if registered."""
    scene = getattr(bpy.context, "scene", None)
    settings = getattr(scene, "rigger_template", None) if scene else None
    if settings is None:
        return None
    return {key: getattr(settings, key) for key in DEFAULT_OPTIONS if hasattr(settings, key)}


TEMPLATES = {
    'HUMAN': ("Human", "Biped with spine, neck, head, face, arms, fingers and legs",
              lambda o=None: human_specs(options=o)),
    'HUMAN_SIMPLE': ("Human (No Fingers/Face)", "Game-friendly biped without fingers, jaw and eyes",
                     lambda o=None: human_specs(fingers=False, face=False, options=o)),
    'QUADRUPED': ("Quadruped", "Four-legged animal with tail", quadruped_specs),
    'HORSE': ("Horse", "Hoofed animal: long neck, ears, tail, four-segment legs", horse_specs),
    'BIRD': ("Bird", "Wings (arm modules with feathers), long neck, beak, tail and bird legs", bird_specs),
}

# Bones copied from templates when adding a single module
MODULE_CHUNKS = {
    'SPINE': ('HUMAN', ["hips", "spine.01", "spine.02", "spine.03"]),
    'NECK': ('HUMAN', ["neck", "head"]),
    'ARM': ('HUMAN', ["shoulder.L", "upper_arm.L", "forearm.L", "hand.L"]),
    'LEG': ('HUMAN', ["thigh.L", "shin.L", "foot.L", "toe.L"]),
    'FINGER': ('HUMAN', ["index.01.L", "index.02.L", "index.03.L"]),
    'CHAIN': ('QUADRUPED', ["tail.01", "tail.02", "tail.03", "tail.04"]),
    'EYE': ('HUMAN', ["eye.L"]),
    'FACE': ('HUMAN', ["face", "jaw", "chin", "lip_upper", "lip_lower", "lip_upper.L", "lip_lower.L",
                       "lip_corner.L", "lid_upper.L", "lid_lower.L", "brow_inner.L", "brow_mid.L",
                       "brow_outer.L", "cheek.L", "nose", "teeth_upper", "teeth_lower",
                       "tongue.01", "tongue.02", "tongue.03"]),
    'SIMPLE': ('HUMAN', ["jaw"]),
}


def spline_chunk():
    """Rope / tentacle: 8 bones along +Y."""
    pts = [Vector((0, 0.1 * i, 0)) for i in range(9)]
    return [BoneSpec(f"rope.{i + 1:02d}", pts[i], pts[i + 1], None if i == 0 else f"rope.{i:02d}",
                     i > 0, UP, 'SPLINE' if i == 0 else 'NONE') for i in range(8)]


def skirt_chunk(strands=10, segments=3, radius=0.15, length=0.4, flare=0.12):
    """Skirt root with strands hanging around it."""
    import math
    specs = [BoneSpec("skirt", Vector((0, 0, 0)), Vector((0, 0, 0.05)), None, False, FORWARD, 'SKIRT')]
    for k in range(strands):
        a = 2.0 * math.pi * k / strands - math.pi / 2  # first strand at the front
        out = Vector((math.cos(a), math.sin(a), 0))
        prev, parent = out * radius, "skirt"
        for j in range(segments):
            t = (j + 1) / segments
            nxt = out * (radius + flare * t) + Vector((0, 0, -length * t))
            name = f"skirt_{k + 1:02d}.{j + 1:02d}"
            specs.append(BoneSpec(name, prev, nxt, parent, j > 0, out))
            prev, parent = nxt, name
    return specs


CHUNK_FUNCTIONS = {'SPLINE': spline_chunk, 'SKIRT': skirt_chunk}


def mirrored(specs):
    """Return specs plus right-side copies of every left-side bone."""
    out = list(specs)
    for spec in specs:
        if utils.side_letter(spec.name) != "L":
            continue
        flip = lambda v: Vector((-v.x, v.y, v.z)) if v is not None else None
        parent = utils.mirror_name(spec.parent) if spec.parent else None
        out.append(replace(spec, name=utils.mirror_name(spec.name), head=flip(spec.head),
                           tail=flip(spec.tail), parent=parent, z=flip(spec.z),
                           params=dict(spec.params)))
    return out


def apply_settings(pose_bone, module, params):
    settings = pose_bone.rigger
    settings.module = module
    for key, value in params.items():
        setattr(settings, key, value)


def build_bones(obj, specs):
    """Create edit bones from specs on an armature already in Edit Mode."""
    ebones = obj.data.edit_bones
    for spec in specs:
        eb = ebones.new(spec.name)
        eb.head = spec.head
        eb.tail = spec.tail
        if spec.z is not None:
            eb.align_roll(spec.z)
    for spec in specs:
        if spec.parent and spec.parent in ebones:
            eb = ebones[spec.name]
            eb.parent = ebones[spec.parent]
            eb.use_connect = spec.connect and (eb.parent.tail - eb.head).length < 1e-5


def create_reference(context, specs, name="reference", location=None):
    arm = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, arm)
    context.collection.objects.link(obj)
    if location is not None:
        obj.location = location
    obj[utils.REF_KEY] = True
    obj.show_in_front = True
    arm.display_type = 'OCTAHEDRAL'
    specs = mirrored(specs)
    utils.set_mode(obj, 'EDIT')
    build_bones(obj, specs)
    bpy.ops.object.mode_set(mode='OBJECT')
    for spec in specs:
        apply_settings(obj.pose.bones[spec.name], spec.module, spec.params)
    obj.rigger.rig_name = "rig"
    return obj


def template_specs(key, options="scene"):
    """Bone specs of a template. Human templates take options ("scene" = the scene's
    Template Options, None = defaults)."""
    fn = TEMPLATES[key][2]
    if key.startswith("HUMAN"):
        return fn(scene_options() if options == "scene" else options)
    return fn()
