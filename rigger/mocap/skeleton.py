# SPDX-License-Identifier: GPL-3.0-or-later
"""The skeleton the video is solved on, built from the character's own rig.

Joints carry rest positions (armature space) and a bone frame (the rest
orientation of the control they drive: X is the bend axis, Y runs along the
bone). Poses are stored per joint as a rotation relative to the parent, in
"rest-aligned" coordinates: at rest every joint's world rotation is the
identity, so a joint's world rotation is directly the change from rest that
the retarget applies to its control.

Sites are the points the pose model sees (shoulders, toes, eyes, finger
tips...), each fixed to one joint. Their positions come from the rig, the head
and foot sites from head/foot proportions.

Plain data and numpy only; from_rig() needs bpy.
"""

import json

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# Prior per joint role, in the joint's bone frame: (mean rotvec, std per axis x/y/z in radians).
# x is the bend axis of Rigger limbs, y the twist along the bone.
PRIORS = {
    "pelvis": ((0, 0, 0), (9, 9, 9)),
    "spine": ((0, 0, 0), (0.25, 0.2, 0.2)),
    "neck": ((0, 0, 0), (0.45, 0.5, 0.35)),
    "head": ((0, 0, 0), (0.5, 0.6, 0.4)),
    "collar": ((0, 0, 0), (0.15, 0.15, 0.15)),
    "upper_arm": ((0, 0, 0), (1.3, 0.8, 1.3)),     # mean replaced by an arms-down pose in from_rig
    "forearm": ((0, 0, 0), (1.3, 0.6, 0.08)),
    "hand": ((0, 0, 0), (0.7, 0.3, 0.4)),
    "thigh": ((0, 0, 0), (0.9, 0.4, 0.45)),
    "shin": ((0, 0, 0), (1.2, 0.08, 0.05)),
    "foot": ((0, 0, 0), (0.45, 0.15, 0.25)),
    "toe": ((0, 0, 0), (0.4, 0.05, 0.05)),
    "finger1": ((0.15, 0, 0), (0.7, 0.05, 0.2)),
    "finger": ((0.15, 0, 0), (0.7, 0.04, 0.06)),
    "thumb": ((0.1, 0, 0), (0.5, 0.35, 0.45)),
}
# Joints that only bend one way around their X axis (sign found from the rest geometry)
HINGES = ("forearm", "shin", "finger1", "finger")

# COCO-WholeBody keypoint index -> site name (the person's left is "L")
BODY_KP = {0: "nose", 1: "eye.L", 2: "eye.R", 3: "ear.L", 4: "ear.R", 5: "shoulder.L", 6: "shoulder.R",
           7: "elbow.L", 8: "elbow.R", 9: "wrist.L", 10: "wrist.R", 11: "hip.L", 12: "hip.R", 13: "knee.L",
           14: "knee.R", 15: "ankle.L", 16: "ankle.R", 17: "big_toe.L", 18: "small_toe.L", 19: "heel.L",
           20: "big_toe.R", 21: "small_toe.R", 22: "heel.R", 23 + 8: "chin"}
HAND_BASE = {"L": 91, "R": 112}
HAND_ORDER = ("wrist",) + tuple(f"{f}{i}" for f in FINGERS for i in (1, 2, 3, 4))  # COCO-WholeBody hand order

# Head sites as fractions of the head bone length: (lateral +X, forward -Y, up +Z) from the head joint
HEAD_SITES = {"nose": (0.0, 0.62, 0.18), "eye.L": (0.2, 0.5, 0.32), "eye.R": (-0.2, 0.5, 0.32),
              "ear.L": (0.46, -0.02, 0.26), "ear.R": (-0.46, -0.02, 0.26), "chin": (0.0, 0.5, -0.22)}
# Where COCO-style keypoints sit relative to the joints, in body heights (inward to the midline,
# forward, up): shoulders are marked on top of the shoulder, hips above the hip joints, the face and
# feet a little higher than a skeleton's head and sole points. Measured on a rendered golden clip
# (bench/calibrate_sites.py); finer per-site fits did not carry over to another body.
SITE_SHIFT = {"shoulder": (0.012, 0.0, 0.036), "hip": (0.0, 0.0, 0.035), "nose": (0, 0, 0.022),
              "eye": (0.004, 0, 0.022), "ear": (-0.002, 0, 0.018), "chin": (0, 0, 0.02),
              "big_toe": (-0.011, 0, 0.011), "small_toe": (0.014, 0, 0.012), "heel": (0, 0, 0.012),
              "knee": (0, 0, 0.008)}


def kp_sites():
    """{COCO-WholeBody index: site name} for body, feet and both hands."""
    out = dict(BODY_KP)
    for side, base in HAND_BASE.items():
        for i, part in enumerate(HAND_ORDER):
            out[base + i] = f"{part}.{side}" if part != "wrist" else f"hand_wrist.{side}"
    return out


def rotvec_to_matrix(v):
    """Rodrigues, batched: (..., 3) -> (..., 3, 3)."""
    v = np.asarray(v, dtype=np.float64)
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    small = theta < 1e-8
    k = np.where(small, 0.0, v / np.where(small, 1.0, theta))
    x, y, z = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], axis=-1).reshape(v.shape[:-1] + (3, 3))
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    R = eye + s * K + (1 - c) * (K @ K)
    # second order for tiny angles keeps finite differences smooth
    Ks = np.stack([zero, -v[..., 2], v[..., 1], v[..., 2], zero, -v[..., 0], -v[..., 1], v[..., 0], zero],
                  axis=-1).reshape(K.shape)
    Rs = eye + Ks + 0.5 * (Ks @ Ks)
    return np.where(small[..., None], Rs, R)


def matrix_to_quat(R):
    """(..., 3, 3) -> (..., 4) wxyz with w >= 0."""
    R = np.asarray(R, dtype=np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q = np.empty(R.shape[:-2] + (4,))
    c0 = tr > 0
    c1 = (~c0) & (m00 >= m11) & (m00 >= m22)
    c2 = (~c0) & (~c1) & (m11 >= m22)
    c3 = ~(c0 | c1 | c2)
    s = np.sqrt(np.maximum(tr + 1, 1e-12)) * 2
    q[..., 0] = np.where(c0, 0.25 * s, 0)
    q[..., 1] = np.where(c0, (R[..., 2, 1] - R[..., 1, 2]) / s, 0)
    q[..., 2] = np.where(c0, (R[..., 0, 2] - R[..., 2, 0]) / s, 0)
    q[..., 3] = np.where(c0, (R[..., 1, 0] - R[..., 0, 1]) / s, 0)
    for cond, i, j, k in ((c1, 0, 1, 2), (c2, 1, 2, 0), (c3, 2, 0, 1)):
        s = np.sqrt(np.maximum(1 + R[..., i, i] - R[..., j, j] - R[..., k, k], 1e-12)) * 2
        w = (R[..., k, j] - R[..., j, k]) / s
        a = 0.25 * s
        b = (R[..., j, i] + R[..., i, j]) / s
        c = (R[..., k, i] + R[..., i, k]) / s
        vals = np.zeros(R.shape[:-2] + (4,))
        vals[..., 0] = w
        vals[..., 1 + i] = a
        vals[..., 1 + j] = b
        vals[..., 1 + k] = c
        q = np.where(cond[..., None], vals, q)
    q *= np.where(q[..., :1] < 0, -1.0, 1.0)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    axis=-1).reshape(q.shape[:-1] + (3, 3))


def matrix_to_rotvec(R):
    q = matrix_to_quat(R)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(n, q[..., :1])
    return np.where(n > 1e-10, v / np.where(n > 1e-10, n, 1) * angle, 2 * v)


def rotation_between(a, b):
    """Smallest rotation matrix taking direction a onto direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, (1, 0, 0) if abs(a[0]) < 0.9 else (0, 1, 0))
        return rotvec_to_matrix(perp / np.linalg.norm(perp) * np.pi)
    return rotvec_to_matrix(axis / s * np.arctan2(s, c))


class Skeleton:
    """Joints, bone frames, priors and sites of a character (see module docstring)."""

    def __init__(self):
        self.joints = []      # names
        self.parents = []     # parent index, -1 for the root
        self.roles = []       # PRIORS key
        self.controls = []    # control bone driven by the joint ("" for none)
        self.rest = []        # rest position (3,)
        self.frames = []      # bone frame (3, 3), columns X Y Z
        self.prior_mean = []
        self.prior_std = []
        self.hinge = []       # +1 / -1: sign of a natural bend around X, 0 for free joints
        self.sites = {}       # name -> (joint index, rest position)
        self.height = 1.0

    # -- building ----------------------------------------------------------
    def add(self, name, parent, role, pos, frame, control=""):
        self.joints.append(name)
        self.parents.append(self.joints.index(parent) if parent else -1)
        self.roles.append(role)
        self.controls.append(control)
        self.rest.append(np.asarray(pos, float))
        f = np.asarray(frame, float)
        self.frames.append(f)
        mean, std = PRIORS[role]
        self.prior_mean.append(np.asarray(mean, float))
        self.prior_std.append(np.asarray(std, float))
        self.hinge.append(0)
        return len(self.joints) - 1

    def site(self, name, joint, pos):
        self.sites[name] = (self.index(joint), np.asarray(pos, float))

    def index(self, name):
        return self.joints.index(name)

    def children(self, j):
        return [i for i, p in enumerate(self.parents) if p == j]

    def finish(self):
        """Arrays, hinge signs and the arms-down prior, once all joints and sites are in."""
        self.rest = np.array(self.rest)
        self.frames = np.array(self.frames)
        self.prior_mean = np.array(self.prior_mean)
        self.prior_std = np.array(self.prior_std)
        self.hinge = np.zeros(len(self.joints), int)
        for j, role in enumerate(self.roles):
            end = self.end_point(j)
            if end is None:
                continue
            bone = end - self.rest[j]
            if role in HINGES:
                # bend direction: knees and toes go backwards (+Y), elbows forwards (-Y),
                # fingers towards the palm (the hand's -Z at rest for a flat hand)
                if role == "shin":
                    want = np.array((0, 1.0, 0))
                elif role == "forearm":
                    want = np.array((0, -1.0, 0))
                else:
                    want = -self.frames[self.parents[j]][:, 2]
                axis = self.frames[j][:, 0]
                moved = np.cross(axis, bone)  # direction the end moves for a small +X rotation
                self.hinge[j] = 1 if np.dot(moved, want) >= 0 else -1
                self.prior_mean[j][0] *= self.hinge[j]
            if role == "upper_arm":
                side = 1.0 if self.rest[j][0] > 0 else -1.0
                down = np.array((0.25 * side, -0.15, -1.0))
                R = rotation_between(bone, down)
                B = self.frames[j]
                self.prior_mean[j] = matrix_to_rotvec(B.T @ R @ B)
        self.height = float(np.ptp(np.array([p for _j, p in self.sites.values()] + list(self.rest))[:, 2]))
        return self

    def end_point(self, j):
        kids = self.children(j)
        if kids:
            # spine-like joints: the child along the chain (the one closest to straight above)
            if len(kids) > 1:
                same = [k for k in kids if np.linalg.norm(self.rest[k] - self.rest[j]) > 1e-6]
                kids = sorted(same or kids, key=lambda k: abs(self.rest[k][0] - self.rest[j][0]))
            if np.linalg.norm(self.rest[kids[0]] - self.rest[j]) > 1e-6:
                return self.rest[kids[0]]
        owned = [(name, pos) for name, (owner, pos) in self.sites.items() if owner == j]
        ends = [pos for name, pos in owned if "end" in name or name[:-2].endswith("4")]
        if ends:
            return ends[0]
        return owned[0][1] if owned else None

    # -- serialization -----------------------------------------------------
    def to_dict(self):
        return {"joints": self.joints, "parents": self.parents, "roles": self.roles, "controls": self.controls,
                "rest": np.asarray(self.rest).tolist(), "frames": np.asarray(self.frames).tolist(),
                "prior_mean": np.asarray(self.prior_mean).tolist(), "prior_std": np.asarray(self.prior_std).tolist(),
                "hinge": np.asarray(self.hinge).tolist(), "height": self.height,
                "sites": {k: [j, np.asarray(p).tolist()] for k, (j, p) in self.sites.items()}}

    @classmethod
    def from_dict(cls, d):
        s = cls()
        s.joints, s.parents, s.roles, s.controls = list(d["joints"]), list(d["parents"]), list(d["roles"]), list(d["controls"])
        s.rest, s.frames = np.array(d["rest"], float), np.array(d["frames"], float)
        s.prior_mean, s.prior_std = np.array(d["prior_mean"], float), np.array(d["prior_std"], float)
        s.hinge = np.array(d["hinge"], int)
        s.height = float(d["height"])
        s.sites = {k: (int(j), np.array(p, float)) for k, (j, p) in d["sites"].items()}
        return s

    def dumps(self):
        return json.dumps(self.to_dict())

    def signature(self):
        """Short hash of the skeleton, part of the cache key."""
        import hashlib
        return hashlib.sha1(self.dumps().encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# From a Rigger rig
# ---------------------------------------------------------------------------

def from_rig(rig):
    """Skeleton of a generated Rigger rig (armature space, rest pose)."""
    from .. import utils
    meta = utils.get_meta(rig)
    bones = rig.data.bones
    if not meta or "spine" not in meta:
        raise ValueError(f"'{rig.name}' is not a Rigger rig with a spine")

    def head(n):
        return np.array(bones[n].head_local)

    def tail(n):
        return np.array(bones[n].tail_local)

    def frame(n):
        return np.array(bones[n].matrix_local.to_3x3())

    sk = Skeleton()
    spine = meta["spine"]
    fk = [n for n in spine["fk"] if n in bones]
    torso = spine["torso"]
    sk.add("pelvis", None, "pelvis", head(fk[0]), frame(torso), torso)
    parent = "pelvis"
    for i, n in enumerate(fk):
        sk.add(f"spine{i + 1}", parent, "spine", head(n), frame(n), n)
        parent = f"spine{i + 1}"
    chest = parent
    neck = meta.get("neck") or {}
    for i, n in enumerate(neck.get("fk", [])):
        sk.add(f"neck{i + 1}" if i else "neck", parent, "neck", head(n), frame(n), n)
        parent = sk.joints[-1]
    head_bone = neck.get("head")
    if head_bone:
        sk.add("head", parent, "head", head(head_bone), frame(head_bone), head_bone)
        hb, H = head(head_bone), float(np.linalg.norm(tail(head_bone) - head(head_bone)))
        sk.site("head_end", "head", tail(head_bone))
        for name, (lat, fwd, up) in HEAD_SITES.items():
            sk.site(name, "head", hb + H * np.array((lat, -fwd, up)))

    legs = {}
    for limb in meta.get("limbs", []):
        side = limb["side"]
        if limb["type"] == "arm":
            names = list(limb["fk"])
            par = chest
            if limb.get("shoulder"):
                sk.add(f"collar.{side}", chest, "collar", head(limb["shoulder"]), frame(limb["shoulder"]),
                       limb["shoulder"])
                par = f"collar.{side}"
            sk.add(f"upper_arm.{side}", par, "upper_arm", head(names[0]), frame(names[0]), names[0])
            sk.add(f"forearm.{side}", f"upper_arm.{side}", "forearm", head(names[1]), frame(names[1]), names[1])
            sk.add(f"hand.{side}", f"forearm.{side}", "hand", head(names[2]), frame(names[2]), names[2])
            # the shoulder keypoint marks the top of the shoulder, which moves with the collarbone
            sk.site(f"shoulder.{side}", par, head(names[0]))
            sk.site(f"elbow.{side}", f"forearm.{side}", head(names[1]))
            sk.site(f"wrist.{side}", f"hand.{side}", head(names[2]))
            sk.site(f"hand_wrist.{side}", f"hand.{side}", head(names[2]))
            sk.site(f"hand_end.{side}", f"hand.{side}", tail(names[2]))
            for finger in FINGERS:
                chain = [f"{finger}.{i:02d}.{side}" for i in (1, 2, 3)]
                if not all(c in bones for c in chain):
                    continue
                par = f"hand.{side}"
                for i, c in enumerate(chain):
                    role = "thumb" if finger == "thumb" else ("finger1" if i == 0 else "finger")
                    sk.add(c, par, role, head(c), frame(c), c)
                    sk.site(f"{finger}{i + 1}.{side}", par, head(c))  # a joint's head moves with its parent
                    par = c
                sk.site(f"{finger}4.{side}", chain[-1], tail(chain[-1]))
        elif limb["type"] == "leg" and side not in legs:
            legs[side] = limb
    for side, limb in legs.items():
        names = list(limb["fk"])
        sk.add(f"thigh.{side}", "pelvis", "thigh", head(names[0]), frame(names[0]), names[0])
        sk.add(f"shin.{side}", f"thigh.{side}", "shin", head(names[1]), frame(names[1]), names[1])
        sk.add(f"foot.{side}", f"shin.{side}", "foot", head(names[2]), frame(names[2]), names[2])
        sk.site(f"knee.{side}", f"shin.{side}", head(names[1]))
        sk.site(f"ankle.{side}", f"foot.{side}", head(names[2]))
        toe = limb.get("toe_fk")
        ankle = head(names[2])
        if toe and toe in bones:
            sk.add(f"toe.{side}", f"foot.{side}", "toe", head(toe), frame(toe), toe)
            toe_end = tail(toe)
            sk.site(f"toe_end.{side}", f"toe.{side}", toe_end)
            toe_owner = f"toe.{side}"
        else:
            toe_end = tail(names[2])
            toe_owner = f"foot.{side}"
        ground = min(toe_end[2], ankle[2])
        foot_len = float(np.linalg.norm((toe_end - ankle)[:2]))
        outward = 1.0 if ankle[0] > 0 else -1.0
        width = 0.35 * foot_len
        sk.site(f"big_toe.{side}", toe_owner, toe_end + np.array((-outward * 0.3 * width, 0, 0)))
        sk.site(f"small_toe.{side}", toe_owner, toe_end + np.array((outward * 0.7 * width, 0.15 * foot_len, 0)))
        sk.site(f"heel.{side}", f"foot.{side}", np.array((ankle[0], ankle[1] + 0.22 * foot_len,
                                                          ground + 0.03 * foot_len)))
    for side in ("L", "R"):
        if f"thigh.{side}" in sk.joints:
            sk.site(f"hip.{side}", "pelvis", sk.rest[sk.index(f"thigh.{side}")])
    height = float(np.ptp(np.array([p for _j, p in sk.sites.values()] + list(sk.rest))[:, 2]))
    for name, (j, pos) in list(sk.sites.items()):
        shift = SITE_SHIFT.get(name.split(".")[0])
        if shift:
            inward = -1.0 if pos[0] > 0 else 1.0 if pos[0] < 0 else 0.0
            sk.sites[name] = (j, pos + height * np.array((inward * shift[0], -shift[1], shift[2])))
    return sk.finish()
