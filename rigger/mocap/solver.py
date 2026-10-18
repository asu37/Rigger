# SPDX-License-Identifier: GPL-3.0-or-later
"""2D keypoints of a video -> 3D motion of a skeleton (numpy only).

Monocular lifting by model fitting: for every frame the character's skeleton
(skeleton.py: its own bone lengths, joint frames and limits) is posed so that
its sites, seen through a pinhole camera, land on the detected keypoints.
Depth comes from the known bone lengths; what the image cannot tell (an arm
pointing at or away from the camera) is settled by anatomical priors, one-way
hinges at knees, elbows and fingers, and continuity with the previous frame.

    solve_sequence -> per-frame camera-space pose (Levenberg-Marquardt per frame)
    smooth          -> zero-lag One-Euro filtering of the joint rotations
    ground          -> world frame from the body's up direction and the floor,
                       contacts, feet kept planted, root kept on the floor
"""

import numpy as np

from . import filters
from .skeleton import kp_sites, matrix_to_rotvec, rotation_between, rotvec_to_matrix

# Camera space: x right, y down, z forward. The rig faces -Y with Z up, so a
# character facing the camera has this root rotation (rig -> camera):
R_FACING = np.array(((1.0, 0, 0), (0, 0, -1), (0, 1, 0)))

DEFAULTS = {
    "back_thigh": 0.35,     # sine of the largest backward swing of the thighs (about 20 deg)...
    "back_arm": 0.6,        # ...and of the upper arms (about 37 deg, as in walking)
    "min_score": 0.3,       # keypoints below this are ignored
    "px_sigma": 0.012,      # reprojection noise, in person heights
    "cauchy": 2.5,          # robust loss scale, in px_sigma
    "prior": 1.0,           # weight of the pose prior
    "temporal_rot": 0.35,   # radians per frame at 30 fps between frames
    "temporal_pos": 0.12,   # metres per frame at 30 fps (times the height / 1.8)
    "floor_sigma": 0.012,   # planted feet: allowed distance from the floor, in heights
    "floor_pass": True,     # second pass with planted feet held on the floor
    "level_camera": "auto",  # True: camera level (tilted camera_tilt degrees down); False: up from the body;
    "camera_tilt": 0.0,      # "auto": from the body unless that is over max_tilt_error degrees off level
    "max_tilt_error": 20.0,
    "iterations": 14,
    "finger_iterations": 8,
    "min_cutoff": 1.2,      # One-Euro, Hz
    "beta": 0.4,
    "contact_height": 0.04,  # contacts: foot within this share of the height of the floor...
    "contact_speed": 0.35,   # ...and slower than this many heights per second
    "foot_lock": True,
}


def _settings(cfg):
    out = dict(DEFAULTS)
    out.update(cfg or {})
    return out


# ---------------------------------------------------------------------------
# Forward kinematics on a part of the skeleton
# ---------------------------------------------------------------------------

class Chain:
    """Joints below a base joint (whose world transform is given), plus the sites they carry."""

    def __init__(self, sk, base, joints, site_names):
        self.sk = sk
        self.base = base
        self.joints = list(joints)
        idx = {j: i for i, j in enumerate(self.joints)}
        self.par = []
        for j in self.joints:
            p = sk.parents[j]
            if p != base and p not in idx:
                raise ValueError(f"joint {sk.joints[j]}: parent outside the chain")
            self.par.append(idx.get(p, -1))
        self.off = np.array([sk.rest[j] - sk.rest[sk.parents[j]] for j in self.joints]).reshape(-1, 3)
        self.B = sk.frames[self.joints] if self.joints else np.zeros((0, 3, 3))
        self.site_names = list(site_names)
        owners = [sk.sites[s][0] for s in self.site_names]
        for s, o in zip(self.site_names, owners):
            if o != base and o not in idx:
                raise ValueError(f"site {s}: joint outside the chain")
        self.site_owner = np.array([idx[o] + 1 if o in idx else 0 for o in owners], int)
        self.site_off = np.array([sk.sites[s][1] - sk.rest[o] for s, o in zip(self.site_names, owners)]).reshape(-1, 3)

    def local(self, theta):
        return self.B @ rotvec_to_matrix(theta) @ np.swapaxes(self.B, -1, -2)

    def forward(self, Gb, pb, theta):
        """Gb (N,3,3), pb (N,3), theta (N,n,3) -> world rotations (N,n,3,3), positions (N,n,3), sites (N,s,3)."""
        N, n = theta.shape[0], len(self.joints)
        L = self.local(theta)
        G = np.empty((N, n + 1, 3, 3))
        P = np.empty((N, n + 1, 3))
        G[:, 0], P[:, 0] = Gb, pb
        for i in range(n):
            p = self.par[i] + 1
            P[:, i + 1] = P[:, p] + G[:, p] @ self.off[i]
            G[:, i + 1] = G[:, p] @ L[:, i]
        S = P[:, self.site_owner] + np.einsum('nkij,kj->nki', G[:, self.site_owner], self.site_off)
        return G[:, 1:], P[:, 1:], S


def project(X, cam):
    f, cx, cy = cam
    z = np.maximum(X[..., 2], 0.05)
    return np.stack([f * X[..., 0] / z + cx, f * X[..., 1] / z + cy], axis=-1)


def _yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1.0)))


# ---------------------------------------------------------------------------
# Levenberg-Marquardt with a batched finite-difference Jacobian
# ---------------------------------------------------------------------------

def levenberg_marquardt(residual, x0, iterations, eps=1e-5, reweight=None):
    """Minimise |residual(x)|^2; residual takes a batch (N, P) and returns (N, M)."""
    x = np.array(x0, float)
    P = x.size
    lam = 1e-2
    cost = None
    for _ in range(iterations):
        if reweight is not None:
            reweight(x)
        X = np.vstack([x, x + eps * np.eye(P)])
        R = residual(X)
        r0 = R[0]
        cost = float(r0 @ r0)
        Jt = (R[1:] - r0) / eps            # (P, M)
        H = Jt @ Jt.T
        g = Jt @ r0
        diag = np.diag(H).copy()
        improved = False
        for _try in range(6):
            try:
                step = np.linalg.solve(H + lam * (np.diag(diag) + 1e-6 * np.eye(P)), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x - step
            rn = residual(xn[None])[0]
            cn = float(rn @ rn)
            if cn < cost:
                x, cost = xn, cn
                lam = max(lam / 3, 1e-7)
                improved = True
                break
            lam *= 4
        if not improved or np.abs(step).max() < 1e-6:
            break
    return x, cost


# ---------------------------------------------------------------------------
# One frame
# ---------------------------------------------------------------------------

class FrameFitter:
    """Fits the body (root + all non-finger joints), then each hand's fingers."""

    def __init__(self, sk, cam, cfg=None):
        self.sk = sk
        self.cam = cam
        self.cfg = _settings(cfg)
        sites = kp_sites()
        self.site_kp = {name: k for k, name in sites.items()}
        finger_roles = ("finger1", "finger", "thumb")
        self.body_joints = [j for j in range(len(sk.joints)) if sk.parents[j] >= 0 and sk.roles[j] not in finger_roles]
        root = sk.parents.index(-1)
        self.root = root
        body_set = set(self.body_joints) | {root}
        body_sites = [s for s, (j, _p) in sk.sites.items() if j in body_set and s in self.site_kp]
        self.body = Chain(sk, root, self.body_joints, body_sites)
        self.hands = {}
        for side in ("L", "R"):
            hand = f"hand.{side}"
            if hand not in sk.joints:
                continue
            h = sk.index(hand)
            joints = [j for j in range(len(sk.joints)) if sk.roles[j] in finger_roles and sk.joints[j].endswith(side)]
            if not joints:
                continue
            fsites = [s for s, (j, _p) in sk.sites.items() if j in joints and s in self.site_kp]
            self.hands[side] = (h, Chain(sk, h, joints, fsites))
        self.body_kp = np.array([self.site_kp[s] for s in self.body.site_names])
        n = len(self.body_joints)
        self.mean = sk.prior_mean[self.body_joints]
        self.std = sk.prior_std[self.body_joints]
        self.hinge = sk.hinge[self.body_joints]
        self.n = n
        self.floor_now = None
        self.foot_sites = {side: [i for i, name in enumerate(self.body.site_names)
                                  if name.split(".")[0] in ("heel", "big_toe", "small_toe") and name.endswith(side)]
                           for side in ("L", "R")}
        # how far a limb may point backwards (+Y in its parent's rest-aligned frame): hips and
        # shoulders extend much less than they flex, which settles the image's front/back ambiguity
        self.reach = []
        for name, child, limit in (("thigh", "shin", self.cfg["back_thigh"]), ("upper_arm", "forearm", self.cfg["back_arm"])):
            for side in ("L", "R"):
                a, b = f"{name}.{side}", f"{child}.{side}"
                if a in sk.joints and b in sk.joints:
                    ia, ib = self.body_joints.index(sk.index(a)), self.body_joints.index(sk.index(b))
                    self.reach.append((ia, ib, self.body.par[ia], limit))

    # -- body ----------------------------------------------------------------
    def _body_residual(self, uv, w, sigma, prev, Rref):
        cfg = self.cfg
        n = self.n
        hinge_idx = np.nonzero(self.hinge)[0]
        hsign = self.hinge[hinge_idx]
        scale = self.sk.height / 1.8
        fps_k = 30.0 / max(cfg.get("fps", 30.0), 1.0)

        def residual(X):
            N = X.shape[0]
            t = X[:, 0:3]
            Rr = rotvec_to_matrix(X[:, 3:6]) @ Rref
            theta = X[:, 6:].reshape(N, n, 3)
            G, P, S = self.body.forward(Rr, t, theta)
            proj = (project(S, self.cam) - uv) * (np.sqrt(w)[:, None] / sigma)
            parts = [proj.reshape(N, -1), ((theta - self.mean) / self.std).reshape(N, -1) * cfg["prior"]]
            for ia, ib, par, limit in self.reach:
                Gp = Rr if par < 0 else G[:, par]
                d = P[:, ib] - P[:, ia]
                back = np.einsum('nij,ni->nj', Gp, d)[:, 1] / np.linalg.norm(d, axis=1)
                parts.append((np.maximum(back - limit, 0) / 0.04)[:, None])
            if len(hinge_idx):
                bend = theta[:, hinge_idx, 0] * hsign
                parts.append(np.maximum(-bend, 0) / 0.05)
            if self.floor_now is not None:  # planted feet on the floor plane found by the first pass
                up, level, idx = self.floor_now
                parts.append((S[:, idx] @ up - level) / (cfg["floor_sigma"] * self.sk.height))
            if prev is not None:
                parts.append(((theta - prev["theta_body"]) / (cfg["temporal_rot"] / fps_k)).reshape(N, -1))
                parts.append(X[:, 3:6] / (cfg["temporal_rot"] / fps_k))
                parts.append((t - prev["t"]) / (cfg["temporal_pos"] * scale / fps_k))
            return np.concatenate(parts, axis=1)
        return residual

    def fit_body(self, kp, sc, init, prev=None, iterations=None):
        """kp (133,2), sc (133,); init/prev: dict t, R, theta_body. Returns (state, cost per keypoint)."""
        cfg = self.cfg
        uv = kp[self.body_kp]
        conf = sc[self.body_kp]
        w0 = np.where(conf >= cfg["min_score"], conf, 0.0)
        good = kp[np.where(sc[:23] >= cfg["min_score"])[0]]
        person_h = max(np.ptp(good[:, 1]) if len(good) > 1 else 100.0, 30.0)
        sigma = cfg["px_sigma"] * person_h
        w = w0.copy()
        Rref = init["R"]
        residual_core = None

        def reweight(x):
            nonlocal residual_core
            _G, _P, S = self.body.forward((rotvec_to_matrix(x[3:6]) @ Rref)[None], x[None, :3],
                                          x[None, 6:].reshape(1, self.n, 3))
            err = np.linalg.norm(project(S[0], self.cam) - uv, axis=1) / sigma
            w[:] = w0 / (1 + (err / cfg["cauchy"]) ** 2)
            residual_core = self._body_residual(uv, w, sigma, prev, Rref)

        def residual(X):
            return residual_core(X)

        x0 = np.concatenate([init["t"], np.zeros(3), init["theta_body"].ravel()])
        x, cost = levenberg_marquardt(residual, x0, iterations or cfg["iterations"], reweight=reweight)
        state = {"t": x[:3], "R": rotvec_to_matrix(x[3:6]) @ Rref, "theta_body": x[6:].reshape(self.n, 3)}
        _G, _P, S = self.body.forward(state["R"][None], state["t"][None], state["theta_body"][None])
        err = np.linalg.norm(project(S[0], self.cam) - uv, axis=1)
        used = w0 > 0
        state["reproj"] = float(np.sum(err[used] * w0[used]) / max(np.sum(w0[used]), 1e-6) / person_h)
        state["sigma"] = sigma
        r = self._body_residual(uv, w, sigma, None, state["R"])(
            np.concatenate([state["t"], np.zeros(3), state["theta_body"].ravel()])[None])[0]
        state["fit_cost"] = float(r @ r)
        return state, cost

    def initial_guesses(self, kp, sc):
        """Root position from the torso's pixel size, four headings."""
        f, cx, cy = self.cam
        sk = self.sk
        ok = sc >= self.cfg["min_score"]

        def mid(a, b):
            if ok[a] and ok[b]:
                return (kp[a] + kp[b]) / 2
            return kp[a] if ok[a] else kp[b] if ok[b] else None
        sh, hp = mid(5, 6), mid(11, 12)
        rest_sh = (sk.sites["shoulder.L"][1] + sk.sites["shoulder.R"][1]) / 2
        rest_hp = (sk.sites["hip.L"][1] + sk.sites["hip.R"][1]) / 2
        torso_len = np.linalg.norm(rest_sh - rest_hp)
        if sh is not None and hp is not None and np.linalg.norm(sh - hp) > 5:
            z = f * torso_len / np.linalg.norm(sh - hp)
            centre = hp
        else:
            good = kp[ok]
            if len(good) < 2:
                return []
            z = f * sk.height / max(np.ptp(good[:, 1]), 20)
            centre = good.mean(axis=0)
        t = np.array(((centre[0] - cx) / f * z, (centre[1] - cy) / f * z, z))
        facing = 0.0
        if ok[5] and ok[6]:
            facing = 0.0 if kp[5][0] > kp[6][0] else np.pi
        theta = self.mean.copy()
        return [{"t": t.copy(), "R": R_FACING @ _yaw(facing + d), "theta_body": theta.copy()}
                for d in (0.0, 0.6, -0.6, np.pi)]

    # -- hands ---------------------------------------------------------------
    def fit_hand(self, side, kp, sc, Gb, pb, theta0, prev_theta, sigma):
        cfg = self.cfg
        h, chain = self.hands[side]
        kps = np.array([self.site_kp[s] for s in chain.site_names])
        uv, conf = kp[kps], sc[kps]
        w = np.where(conf >= cfg["min_score"], conf, 0.0)
        if w.sum() < 3:
            return theta0
        J = chain.joints
        mean, std, hinge = self.sk.prior_mean[J], self.sk.prior_std[J], self.sk.hinge[J]
        hidx = np.nonzero(hinge)[0]
        n = len(J)
        fps_k = 30.0 / max(cfg.get("fps", 30.0), 1.0)
        hsigma = sigma * 0.6

        def residual(X):
            N = X.shape[0]
            theta = X.reshape(N, n, 3)
            _G, _P, S = chain.forward(np.broadcast_to(Gb, (N, 3, 3)), np.broadcast_to(pb, (N, 3)), theta)
            parts = [((project(S, self.cam) - uv) * (np.sqrt(w)[:, None] / hsigma)).reshape(N, -1),
                     ((theta - mean) / std).reshape(N, -1)]
            if len(hidx):
                parts.append(np.maximum(-theta[:, hidx, 0] * hinge[hidx], 0) / 0.05)
            if prev_theta is not None:
                parts.append(((theta - prev_theta) / (cfg["temporal_rot"] * 1.5 / fps_k)).reshape(N, -1))
            return np.concatenate(parts, axis=1)
        x, _ = levenberg_marquardt(residual, theta0.ravel(), cfg["finger_iterations"])
        return x.reshape(n, 3)


# ---------------------------------------------------------------------------
# A whole clip
# ---------------------------------------------------------------------------

def solve_sequence(sk, kp, sc, cam, fps, cfg=None, progress=None, floor=None, reverse=False):
    """kp (F,133,2) pixels, sc (F,133) -> dict of camera-space results:
    t (F,3), R (F,3,3) root rotation rig->camera, theta (F,J,3) bone-frame rotvecs, valid (F,), reproj (F,).

    floor: optional dict(up, level, contacts (F,2)) in camera space from a first pass (see ground):
    feet in contact are then held on that plane, which settles leg depth and the root height.
    reverse: solve from the last frame back to the first (for experiments)."""
    cfg = _settings(cfg)
    cfg["fps"] = fps
    fitter = FrameFitter(sk, cam, cfg)
    F, J = len(kp), len(sk.joints)
    out_t = np.zeros((F, 3))
    out_R = np.tile(np.eye(3), (F, 1, 1))
    theta = np.tile(sk.prior_mean, (F, 1, 1)).astype(float)
    valid = np.zeros(F, bool)
    reproj = np.full(F, np.nan)
    fit_costs = np.full(F, np.inf)
    prev = None
    last = None
    bad_run = 0
    for n_done, f in enumerate(range(F - 1, -1, -1) if reverse else range(F)):
        enough = np.sum(sc[f, :23] >= cfg["min_score"]) >= 6
        if not enough:
            bad_run += 1
            if bad_run > fps:  # lost for over a second: start over when the person comes back
                prev = None
            if prev is not None:
                out_t[f], out_R[f] = prev["t"], prev["R"]
                theta[f] = theta[last]
            if progress:
                progress(n_done + 1, F)
            continue
        bad_run = 0
        fitter.floor_now = None
        if floor is not None:
            idx = [i for k, side in enumerate(("L", "R")) if floor["contacts"][f, k] for i in fitter.foot_sites[side]]
            if idx:
                fitter.floor_now = (floor["up"], floor["level"], idx)
        if prev is None:
            best = None
            for guess in fitter.initial_guesses(kp[f], sc[f]):
                st, cost = fitter.fit_body(kp[f], sc[f], guess, None, iterations=cfg["iterations"] * 2)
                if best is None or cost < best[1]:
                    best = (st, cost)
            state = best[0]
        else:
            state, cost = fitter.fit_body(kp[f], sc[f], prev, prev)
            if state["reproj"] > 0.05:  # lost the pose (a fast turn...): try fresh starts as well
                for guess in fitter.initial_guesses(kp[f], sc[f]):
                    guess["t"] = prev["t"].copy()
                    st, c2 = fitter.fit_body(kp[f], sc[f], guess, prev, iterations=cfg["iterations"] * 2)
                    if c2 < cost:
                        state, cost = st, c2
        out_t[f], out_R[f] = state["t"], state["R"]
        theta[f, fitter.body_joints] = state["theta_body"]
        if fitter.hands:
            G, P, _S = fitter.body.forward(state["R"][None], state["t"][None], state["theta_body"][None])
            for side, (h, chain) in fitter.hands.items():
                k = fitter.body_joints.index(h)
                had = last is not None and valid[last]
                t0 = theta[last, chain.joints] if had else sk.prior_mean[chain.joints]
                theta[f, chain.joints] = fitter.fit_hand(side, kp[f], sc[f], G[0, k], P[0, k], t0.copy(),
                                                         t0 if had else None, state["sigma"])
        valid[f] = True
        reproj[f] = state["reproj"]
        fit_costs[f] = state["fit_cost"]
        prev = dict(state)
        last = f
        if progress:
            progress(n_done + 1, F)
    _fill_gaps(valid, out_t, out_R, theta)
    return {"t": out_t, "R": out_R, "theta": theta, "valid": valid, "reproj": reproj, "cost": fit_costs}


def _fill_gaps(valid, t, R, theta):
    """Frames without a person: hold the nearest solved pose (interpolated between two)."""
    idx = np.nonzero(valid)[0]
    if len(idx) == 0:
        return
    F = len(valid)
    for f in range(F):
        if valid[f]:
            continue
        after = idx[idx > f]
        before = idx[idx < f]
        if len(before) and len(after):
            a, b = before[-1], after[0]
            s = (f - a) / (b - a)
            t[f] = t[a] * (1 - s) + t[b] * s
            theta[f] = theta[a] * (1 - s) + theta[b] * s
            R[f] = R[a] if s < 0.5 else R[b]
        else:
            src = before[-1] if len(before) else after[0]
            t[f], R[f], theta[f] = t[src], R[src], theta[src]


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def smooth(sol, fps, cfg=None):
    """Zero-lag One-Euro filtering of the root path, root rotation and joint rotations."""
    cfg = _settings(cfg)
    out = dict(sol)
    mc, beta = cfg["min_cutoff"], cfg["beta"]
    out["theta"] = filters.one_euro_zero_lag(sol["theta"], fps, mc, beta)
    out["t"] = filters.one_euro_zero_lag(sol["t"], fps, mc * 0.8, beta)
    out["R"] = filters.smooth_rotations(sol["R"], fps, mc, beta)
    return out


# ---------------------------------------------------------------------------
# World frame, floor, contacts
# ---------------------------------------------------------------------------

def full_chain(sk):
    root = sk.parents.index(-1)
    joints = [j for j in range(len(sk.joints)) if j != root]
    return Chain(sk, root, joints, list(sk.sites)), root


def pose_points(sk, R, t, theta):
    """Joint world rotations (F,J,3,3), positions (F,J,3) and site positions {name: (F,3)}."""
    chain, root = full_chain(sk)
    G, P, S = chain.forward(R, t, theta[:, chain.joints])
    F, J = len(t), len(sk.joints)
    Gall = np.empty((F, J, 3, 3))
    Pall = np.empty((F, J, 3))
    Gall[:, root], Pall[:, root] = R, t
    Gall[:, chain.joints], Pall[:, chain.joints] = G, P
    return Gall, Pall, {name: S[:, i] for i, name in enumerate(chain.site_names)}


def _foot_names(sk, side):
    return [n for n in (f"heel.{side}", f"big_toe.{side}", f"small_toe.{side}", f"toe_end.{side}") if n in sk.sites]


def ground(sk, sol, fps, cfg=None):
    """Camera-space solution -> world (Z up, camera looking along +Y, floor at Z=0).

    Returns dict: root_t (F,3) world pelvis position, root_R (F,3,3) world root rotation
    (rest-aligned), theta, W (camera->world rotation), cam_pos (3,), contacts (F,2) L/R,
    floor_offset, valid."""
    cfg = _settings(cfg)
    R, t, theta = sol["R"], sol["t"], sol["theta"]
    G, P, S = pose_points(sk, R, t, theta)
    j = sk.index
    # up: the body's own up direction while standing straight, roll assumed level
    head_j = j("head") if "head" in sk.joints else j("neck")
    up_vecs = []
    for f in range(len(t)):
        knees = [theta[f, j(f"shin.{s}"), 0] * sk.hinge[j(f"shin.{s}")] for s in ("L", "R")]
        if max(knees) < 0.35:
            ank = (P[f, j("foot.L")] + P[f, j("foot.R")]) / 2
            up_vecs.append(P[f, head_j] - ank)
    if len(up_vecs) < 3:
        up_vecs = [P[f, head_j] - P[f, j("pelvis")] for f in range(len(t))]
    up = np.mean([v / np.linalg.norm(v) for v in up_vecs], axis=0)
    up[0] = 0.0  # no camera roll
    up /= np.linalg.norm(up)
    # A body leaning toward or away from a low camera looks almost the same in the image, so the
    # body's up direction can be far off; then a level camera (tripod, phone held upright), tilted
    # by camera_tilt, is the steadier assumption. "auto" keeps the body's estimate when it is close.
    tilt = np.radians(cfg["camera_tilt"])
    level = np.array((0.0, -np.cos(tilt), -np.sin(tilt)))
    mode = cfg["level_camera"]
    if mode is True or (mode == "auto" and np.degrees(np.arccos(np.clip(up @ level, -1, 1))) > cfg["max_tilt_error"]):
        up = level
    fwd = np.array((0, 0, 1.0)) - up * up[2]
    fwd /= np.linalg.norm(fwd)
    x = np.cross(fwd, up)
    W = np.stack([x, fwd, up])            # rows: world axes in camera coordinates
    feet = {s: np.stack([S[n] for n in _foot_names(sk, s)], axis=1) for s in ("L", "R")}
    heights = {s: feet[s] @ up for s in feet}             # (F, k)
    lowest = np.minimum(heights["L"].min(axis=1), heights["R"].min(axis=1))
    floor = np.median(lowest)
    p0 = P[0, j("pelvis")]
    origin = p0 - (p0 @ up - floor) * up
    to_world = lambda v: (v - origin) @ W.T  # noqa: E731
    root_t = to_world(t)
    root_R = W @ R
    feet_w = {s: to_world(feet[s]) for s in feet}
    # contacts: a slow foot that is not clearly above the other one. Heights are judged
    # between the feet, not against a fixed floor, because depth errors move the whole body
    H = sk.height
    lows = np.stack([feet_w[s][:, :, 2].min(axis=1) for s in ("L", "R")], axis=1)
    contacts = np.zeros((len(t), 2), bool)
    for k, s in enumerate(("L", "R")):
        pos = feet_w[s].mean(axis=1)
        vel = np.zeros(len(t))
        if len(t) > 1:
            vel[1:] = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * fps
            vel[0] = vel[1]
        vel = filters.one_euro_zero_lag(vel[:, None], fps, 3.0, 0.0)[:, 0]
        above = lows[:, k] - lows.min(axis=1)
        contacts[:, k] = (above < cfg["contact_height"] * H) & (vel < cfg["contact_speed"] * H)
        contacts[:, k] = filters.clean_mask(contacts[:, k], min_run=max(2, int(fps * 0.08)))
    lows_cam = np.stack([heights[s].min(axis=1) for s in ("L", "R")], axis=1)
    level = float(np.median(lows_cam[contacts])) if contacts.any() else float(floor)
    out = {"root_t": root_t, "root_R": root_R, "theta": theta, "W": W, "cam_pos": to_world(np.zeros(3)),
           "floor_cam": {"up": up, "level": level, "contacts": contacts.copy()},
           "contacts": contacts, "valid": sol["valid"], "floor_offset": 0.0}
    if cfg["foot_lock"]:
        plant(sk, out, fps)
    return out


def plant(sk, world, fps):
    """Keep the root on the floor and stance feet from sliding by moving the root.

    Vertical: the lowest foot point of a planted foot is put on the floor. Horizontal:
    during a contact the foot is held at where it landed; the root takes the
    correction (the foot planted longest wins), blended smoothly in and out."""
    root_t, root_R, theta, contacts = world["root_t"], world["root_R"], world["theta"], world["contacts"]
    F = len(root_t)
    _G, _P, S = pose_points(sk, root_R, root_t, theta)
    feet = {s: np.stack([S[n] for n in _foot_names(sk, s)], axis=1) for s in ("L", "R")}
    # vertical: floor contact
    dz = np.full(F, np.nan)
    for f in range(F):
        lows = [feet[s][f, :, 2].min() for k, s in enumerate(("L", "R")) if contacts[f, k]]
        if lows:
            dz[f] = -min(lows)
    if np.isfinite(dz).any():
        idx = np.nonzero(np.isfinite(dz))[0]
        dz = np.interp(np.arange(F), idx, dz[idx])
        dz = filters.one_euro_zero_lag(dz[:, None], fps, 1.0, 0.0)[:, 0]
    else:
        dz = np.zeros(F)
    # horizontal: anchor stance feet
    corr = np.zeros((F, 2))
    weight = np.zeros(F)
    for k, s in enumerate(("L", "R")):
        pos = feet[s].mean(axis=1)[:, :2]
        f = 0
        while f < F:
            if not contacts[f, k]:
                f += 1
                continue
            e = f
            while e < F and contacts[e, k]:
                e += 1
            anchor = pos[f:e].mean(axis=0) if e - f < 3 else np.median(pos[f:e], axis=0)
            for g in range(f, e):
                w = float(e - f)
                if w > weight[g]:
                    weight[g] = w
                    corr[g] = anchor - pos[g]
            f = e
    has = weight > 0
    if has.any():
        # blend the correction through swing phases so the root never jumps
        idx = np.nonzero(has)[0]
        for a in range(2):
            corr[:, a] = np.interp(np.arange(F), idx, corr[idx, a])
        corr = filters.one_euro_zero_lag(corr, fps, 2.0, 0.0)
    delta = np.column_stack([corr, dz])
    world["root_t"] = root_t + delta
    world["floor_offset"] = float(np.mean(dz))
    # the camera moves with the average correction, so the body still lines up with the video
    world["cam_pos"] = world["cam_pos"] + delta.mean(axis=0)
    world["plant_shift"] = delta.mean(axis=0)
    lock_feet(sk, world, fps)


def _segments(mask):
    out, f = [], 0
    while f < len(mask):
        if mask[f]:
            e = f
            while e < len(mask) and mask[e]:
                e += 1
            out.append((f, e))
            f = e
        else:
            f += 1
    return out


def lock_feet(sk, world, fps, blend_seconds=0.12):
    """Hold each planted foot still: two-bone IK on the leg puts the ankle back where the
    contact began (its median position) and the foot keeps one orientation, faded in and
    out at the ends of each contact."""
    from .filters import continuous_quats
    from .skeleton import matrix_to_quat, quat_to_matrix
    root_R, root_t, theta, contacts = world["root_R"], world["root_t"], world["theta"], world["contacts"]
    G, P, _S = pose_points(sk, root_R, root_t, theta)
    B = sk.frames
    ramp = max(1, int(round(blend_seconds * fps)))
    for k, side in enumerate(("L", "R")):
        names = (f"thigh.{side}", f"shin.{side}", f"foot.{side}")
        if not all(n in sk.joints for n in names):
            continue
        h, kn, a = (sk.index(n) for n in names)
        l1 = np.linalg.norm(sk.rest[kn] - sk.rest[h])
        l2 = np.linalg.norm(sk.rest[a] - sk.rest[kn])
        for f0, f1 in _segments(contacts[:, k]):
            anchor = np.median(P[f0:f1, a], axis=0)
            q = continuous_quats(matrix_to_quat(G[f0:f1, a]))
            q_lock = q.mean(axis=0)
            q_lock /= np.linalg.norm(q_lock)
            R_lock = quat_to_matrix(q_lock)
            for f in range(max(0, f0 - ramp), min(len(theta), f1 + ramp)):
                w = 1.0 if f0 <= f < f1 else 1.0 - (f0 - f if f < f0 else f - f1 + 1) / (ramp + 1)
                if w <= 0:
                    continue
                H, K, A = P[f, h], P[f, kn], P[f, a]
                target = A + w * (anchor - A)
                d = target - H
                dist = np.clip(np.linalg.norm(d), abs(l1 - l2) + 1e-4, (l1 + l2) * 0.999)
                dn = d / max(np.linalg.norm(d), 1e-9)
                pole = (K - H) - dn * np.dot(K - H, dn)
                if np.linalg.norm(pole) < 1e-6:
                    pole = G[f, h] @ np.array((0, -1.0, 0))
                    pole -= dn * np.dot(pole, dn)
                pole /= np.linalg.norm(pole)
                cos_a = np.clip((l1 ** 2 + dist ** 2 - l2 ** 2) / (2 * l1 * dist), -1, 1)
                K2 = H + l1 * (cos_a * dn + np.sqrt(1 - cos_a ** 2) * pole)
                A2 = H + dn * dist
                D1 = rotation_between(K - H, K2 - H)
                D2 = rotation_between(D1 @ (A - K), A2 - K2)
                Gh = D1 @ G[f, h]
                Gk = D2 @ D1 @ G[f, kn]
                qa = matrix_to_quat(G[f, a])
                ql = q_lock if np.dot(qa, q_lock) >= 0 else -q_lock
                qf = (1 - w) * qa + w * ql
                Ga = quat_to_matrix(qf / np.linalg.norm(qf))
                par = sk.parents[h]
                for j, Gj, Gp in ((h, Gh, G[f, par]), (kn, Gk, Gh), (a, Ga, Gk)):
                    theta[f, j] = matrix_to_rotvec(B[j].T @ (Gp.T @ Gj) @ B[j])
        del R_lock
    world["theta"] = theta


def pack(sk, world):
    """World solution -> mcap motion arrays (root_t, root_q, local_q, valid, contacts)."""
    B = sk.frames
    L = B @ rotvec_to_matrix(world["theta"]) @ np.swapaxes(B, -1, -2)
    root = sk.parents.index(-1)
    L[:, root] = np.eye(3)
    from .skeleton import matrix_to_quat
    from .filters import continuous_quats
    return {"root_t": world["root_t"], "root_q": continuous_quats(matrix_to_quat(world["root_R"])),
            "local_q": continuous_quats(matrix_to_quat(L)), "valid": world["valid"], "contacts": world["contacts"],
            "camera_W": world["W"], "camera_pos": world["cam_pos"]}
