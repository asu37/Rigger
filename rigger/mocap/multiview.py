# SPDX-License-Identifier: GPL-3.0-or-later
"""Two cameras: what one camera cannot see in depth, the other sees across.

No calibration board is needed: the person is the calibration object. The
same keypoints seen by both cameras give hundreds of point correspondences, from
which the essential matrix (8-point algorithm, RANSAC) yields the second
camera's rotation and direction; triangulated bone lengths of the character set
the scale. (With too few shared keypoints, two single-camera solves are aligned
instead.) The solve then poses the skeleton on the keypoints of both videos at
once, and the camera pose is refined on them.

    align        -> the second video's keypoints on the first video's frames
    audio_offset -> time offset between two recordings of the same take, from their sound
    calibrate_epipolar -> second camera pose from shared keypoints (calibrate: from two solves)
    solve        -> joint two-camera solve (camera-1 coordinates)
"""

import numpy as np

from . import solver
from .skeleton import matrix_to_rotvec, rotvec_to_matrix


def align(kp_b, sc_b, fps_a, fps_b, frames_a, offset=0.0):
    """Second video's keypoints resampled onto the first video's frames.

    offset: seconds the second video lags the first (a frame at time T in the first video
    shows the same moment as time T + offset in the second)."""
    kp = np.zeros((frames_a,) + kp_b.shape[1:], kp_b.dtype)
    sc = np.zeros((frames_a,) + sc_b.shape[1:], sc_b.dtype)
    for f in range(frames_a):
        g = int(round((f / fps_a + offset) * fps_b))
        if 0 <= g < len(kp_b):
            kp[f], sc[f] = kp_b[g], sc_b[g]
    return kp, sc


def audio_offset(path_a, path_b, max_seconds=10.0, rate=2000):
    """Seconds the second recording lags the first, from their sound (None without sound)."""
    try:
        import aud
    except ImportError:
        return None

    def envelope(path):
        try:
            data = np.asarray(aud.Sound(path).resample(rate, False).data(), float)
        except Exception:  # no audio stream, unsupported codec...
            return None
        if data.size == 0:
            return None
        mono = np.abs(data.mean(axis=1) if data.ndim > 1 else data)
        hop = rate // 100   # 10 ms
        env = mono[:len(mono) // hop * hop].reshape(-1, hop).mean(axis=1)
        return (env - env.mean()) / (env.std() + 1e-9)
    a, b = envelope(path_a), envelope(path_b)
    if a is None or b is None or min(len(a), len(b)) < 50:
        return None
    n = 1 << int(np.ceil(np.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    lag_max = int(max_seconds * 100)
    lags = np.r_[np.arange(0, lag_max + 1), np.arange(-lag_max, 0)]
    vals = np.r_[corr[:lag_max + 1], corr[-lag_max:]]
    best = int(lags[np.argmax(vals)])
    if vals.max() < 3 * np.median(np.abs(vals)):   # no clear match
        return None
    return best / 100.0


def _rigid(a, b):
    """Rotation R and translation t with b ~ a @ R.T + t (Kabsch, no scale)."""
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    U, _s, Vt = np.linalg.svd((a - ma).T @ (b - mb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mb - R @ ma


def calibrate(sk, sol_a, sol_b, inlier=0.12, seed=0):
    """Second camera pose (R, t: X_b = R @ X_a + t) from two independent solves of the same take.

    A robust fit over frames: single-camera depth mistakes spoil some frames, which the
    consensus leaves out. Returns (R, t, share of frames that agreed)."""
    _G, Pa, _S = solver.pose_points(sk, sol_a["R"], sol_a["t"], sol_a["theta"])
    _G, Pb, _S = solver.pose_points(sk, sol_b["R"], sol_b["t"], sol_b["theta"])
    body = [j for j in range(len(sk.joints)) if sk.roles[j] not in ("finger1", "finger", "thumb")]
    ok = sol_a["valid"] & sol_b["valid"]
    frames = np.nonzero(ok)[0]
    if len(frames) < 5:
        raise ValueError("the person must be seen by both cameras for longer (fewer than 5 shared frames)")
    rng = np.random.default_rng(seed)
    tol = inlier * sk.height
    best = None
    for f in rng.choice(frames, size=min(60, len(frames)), replace=False):
        R, t = _rigid(Pa[f, body], Pb[f, body])
        err = np.array([np.median(np.linalg.norm(Pa[g, body] @ R.T + t - Pb[g, body], axis=1)) for g in frames])
        n = int(np.sum(err < tol))
        if best is None or n > best[0]:
            best = (n, err < tol)
    keep = frames[best[1]]
    R, t = _rigid(Pa[keep][:, body].reshape(-1, 3), Pb[keep][:, body].reshape(-1, 3))
    return R, t, float(best[0]) / len(frames)


def refine(sk, sol, kp_b, sc_b, cam_b, R, t, min_score=0.3, step=2, cauchy=2.5):
    """Refine the second camera's pose on its keypoints, with the body solved from both cameras."""
    chain, _root = solver.full_chain(sk)
    _G, _P, S = chain.forward(sol["R"], sol["t"], sol["theta"][:, chain.joints])
    names = chain.site_names
    from .skeleton import kp_sites
    site_kp = {v: k for k, v in kp_sites().items()}
    use = [i for i, n in enumerate(names) if n in site_kp and site_kp[n] < 23]
    kps = np.array([site_kp[names[i]] for i in use])
    frames = np.nonzero(sol["valid"])[0][::step]
    pts = S[frames][:, use]                         # (n, s, 3) camera-1 coordinates
    uv = kp_b[frames][:, kps]
    w0 = np.where(sc_b[frames][:, kps] >= min_score, sc_b[frames][:, kps], 0.0)
    if w0.sum() < 20:
        return R, t
    h = np.array([max(np.ptp(uv[i][w0[i] > 0][:, 1]) if (w0[i] > 0).sum() > 1 else 100.0, 30.0)
                  for i in range(len(frames))])
    sigma = (0.012 * h)[:, None]
    w = w0.copy()

    def project_with(x):
        Rn = rotvec_to_matrix(x[:3]) @ R
        return solver.project(pts @ Rn.T + (t + x[3:]), cam_b)

    def residual(X):
        out = []
        for x in X:
            out.append(((project_with(x) - uv) * (np.sqrt(w)[..., None] / sigma[..., None])).ravel())
        return np.array(out)

    def reweight(x):
        err = np.linalg.norm(project_with(x) - uv, axis=2) / sigma
        w[:] = w0 / (1 + (err / cauchy) ** 2)
    x, _ = solver.levenberg_marquardt(residual, np.zeros(6), 12, reweight=reweight)
    return rotvec_to_matrix(x[:3]) @ R, t + x[3:]


# ---------------------------------------------------------------------------
# Two-view geometry from keypoint correspondences
# ---------------------------------------------------------------------------

def _normalized(kp, cam):
    f, cx, cy = cam
    return np.column_stack([(kp[:, 0] - cx) / f, (kp[:, 1] - cy) / f, np.ones(len(kp))])


def _essential_8pt(xa, xb):
    """Essential matrix from >= 8 normalised correspondences (x_b^T E x_a = 0)."""
    A = np.column_stack([xb[:, [0]] * xa, xb[:, [1]] * xa, xa])
    _u, _s, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    U, _s, Vt = np.linalg.svd(E)
    return U @ np.diag([1.0, 1.0, 0.0]) @ Vt


def _sampson(E, xa, xb):
    Ex = xa @ E.T
    Etx = xb @ E
    num = np.sum(xb * Ex, axis=1) ** 2
    den = Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


def triangulate(xa, xb, R, t):
    """Linear (DLT) triangulation of normalised points; returns points in camera-1 coordinates."""
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t[:, None]])
    out = np.empty((len(xa), 3))
    for i, (a, b) in enumerate(zip(xa, xb)):
        A = np.stack([a[0] * P1[2] - P1[0], a[1] * P1[2] - P1[1], b[0] * P2[2] - P2[0], b[1] * P2[2] - P2[1]])
        _u, _s, Vt = np.linalg.svd(A)
        X = Vt[-1]
        out[i] = X[:3] / X[3] if abs(X[3]) > 1e-12 else X[:3] * 1e6
    return out


def _decompose(E, xa, xb):
    """The (R, t) of the four candidates that puts the most points in front of both cameras."""
    U, _s, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array(((0, -1.0, 0), (1, 0, 0), (0, 0, 1)))
    best = None
    for R in (U @ W @ Vt, U @ W.T @ Vt):
        for t in (U[:, 2], -U[:, 2]):
            X = triangulate(xa, xb, R, t)
            front = np.sum((X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0))
            if best is None or front > best[0]:
                best = (front, R, t)
    return best[1], best[2]


BONE_PAIRS = ((5, 7, "shoulder", "elbow"), (7, 9, "elbow", "wrist"), (6, 8, "shoulder", "elbow"),
              (8, 10, "elbow", "wrist"), (11, 13, "hip", "knee"), (13, 15, "knee", "ankle"),
              (12, 14, "hip", "knee"), (14, 16, "knee", "ankle"))


def _cross(t):
    return np.array(((0, -t[2], t[1]), (t[2], 0, -t[0]), (-t[1], t[0], 0)))


def shared_points(kp_a, sc_a, cam_a, kp_b, sc_b, cam_b, min_score=0.5):
    """Normalised image points of the body keypoints both cameras see confidently, frame by frame."""
    xa, xb, frames, kps = [], [], [], []
    for f in range(len(kp_a)):
        ok = (sc_a[f, :23] >= min_score) & (sc_b[f, :23] >= min_score)
        idx = np.nonzero(ok)[0]
        if len(idx):
            xa.append(_normalized(kp_a[f, idx], cam_a))
            xb.append(_normalized(kp_b[f, idx], cam_b))
            frames += [f] * len(idx)
            kps += list(idx)
    if not xa:
        return None
    return np.vstack(xa), np.vstack(xb), np.array(frames), np.array(kps)


def refine_epipolar(R, t, xa, xb, f_px, iterations=30, cauchy_px=2.0):
    """Camera-2 rotation and translation direction minimising the Sampson (epipolar) error of the
    shared keypoints, starting from R, t. A person a few metres away is nearly flat in depth, which
    makes closed-form essential-matrix solutions unstable; refining a rough start is not."""
    t = np.asarray(t, float)
    d = t / np.linalg.norm(t)
    # tangent basis around the starting direction for the 2 direction parameters
    e1 = np.cross(d, (1, 0, 0) if abs(d[0]) < 0.9 else (0, 1, 0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    w = np.ones(len(xa))

    def params_to_pose(p):
        Rn = rotvec_to_matrix(p[:3]) @ R
        dn = d + p[3] * e1 + p[4] * e2
        return Rn, dn / np.linalg.norm(dn)

    def residual(X):
        out = []
        for p in X:
            Rn, dn = params_to_pose(p)
            out.append(np.sqrt(_sampson(_cross(dn) @ Rn, xa, xb) * w) * f_px)
        return np.array(out)

    def reweight(p):
        Rn, dn = params_to_pose(p)
        err = np.sqrt(_sampson(_cross(dn) @ Rn, xa, xb)) * f_px
        w[:] = 1.0 / (1.0 + (err / cauchy_px) ** 2)
    p, _ = solver.levenberg_marquardt(residual, np.zeros(5), iterations, reweight=reweight)
    Rn, dn = params_to_pose(p)
    err = np.sqrt(_sampson(_cross(dn) @ Rn, xa, xb)) * f_px
    return Rn, dn, float(np.median(err))


def metric_scale(sk, R, d, kp_a, sc_a, cam_a, kp_b, sc_b, cam_b, min_score=0.5):
    """Scale of the camera baseline from the character's bone lengths between triangulated joints."""
    rest = {name: sk.sites[name][1] for name in sk.sites}
    ratios = []
    for f in range(len(kp_a)):
        for a, b, na, nb in BONE_PAIRS:
            if min(sc_a[f, a], sc_a[f, b], sc_b[f, a], sc_b[f, b]) < min_score:
                continue
            side = "L" if a % 2 else "R"
            sa, sb = f"{na}.{side}", f"{nb}.{side}"
            if sa not in rest or sb not in rest:
                continue
            P = triangulate(_normalized(kp_a[f, [a, b]], cam_a), _normalized(kp_b[f, [a, b]], cam_b), R, d)
            est = np.linalg.norm(P[0] - P[1])
            if est > 1e-9:
                ratios.append(np.linalg.norm(rest[sa] - rest[sb]) / est)
    return float(np.median(ratios)) if ratios else None


FOCAL_CANDIDATES = (18, 22, 26, 30, 35, 40, 46, 55)   # 35 mm-equivalent focal lengths tried by auto_focal


def _with_focal(cam, focal_mm):
    _f, cx, cy = cam
    return (focal_mm / 36.0 * 2 * max(cx, cy), cx, cy)


def solve(sk, kp_a, sc_a, cam_a, kp_b, sc_b, cam_b, fps, cfg=None, progress=None, rounds=None):
    """Joint two-camera solve. kp_b/sc_b already on the first video's frames (see align).

    Returns (solution in camera-1 coordinates, views list for further passes, keypoints used,
    info dict, camera-1 intrinsics actually used)."""
    def stage(name, share):
        if not progress:
            return None
        return lambda d, n: progress(name, d, n)
    # rough start: two single-camera solves aligned on the skeleton...
    sol_a = solver.solve_sequence(sk, kp_a, sc_a, cam_a, fps, cfg, stage("camera 1", 0))
    sol_b = solver.solve_sequence(sk, kp_b, sc_b, cam_b, fps, cfg, stage("camera 2", 0))
    R, t, agree = calibrate(sk, sol_a, sol_b)
    # ...then exact: epipolar refinement on the shared keypoints, metric scale from bone lengths.
    # The lens is found too when asked: the focal length whose calibration lets the skeleton
    # fit both videos best (two views of a known skeleton pin it down; one view cannot).
    method = "skeletons"
    epi_err = None
    focal_mm = None
    candidates = [None]
    if (cfg or {}).get("auto_focal", True):
        candidates = list(FOCAL_CANDIDATES)
    best = None
    focal_fits = {}
    for fmm in candidates:
        ca = cam_a if fmm is None else _with_focal(cam_a, fmm)
        cb = cam_b if fmm is None else _with_focal(cam_b, fmm)
        pts = shared_points(kp_a, sc_a, ca, kp_b, sc_b, cb)
        if pts is None or len(pts[0]) < 30:
            continue
        R2, d2, err = refine_epipolar(R, t, pts[0], pts[1], (ca[0] + cb[0]) / 2)
        scale = metric_scale(sk, R2, d2, kp_a, sc_a, ca, kp_b, sc_b, cb)
        if not scale or err > 6.0:
            continue
        fit = 0.0
        if len(candidates) > 1:
            sub = np.arange(0, len(kp_a), 4)
            probe = solver.solve_sequence(sk, np.stack([kp_a[sub], kp_b[sub]], 1), np.stack([sc_a[sub], sc_b[sub]], 1),
                                          ca, fps / 4, cfg, views=[(cb, R2, d2 * scale)])
            fit = float(np.nanmean(probe["reproj"]))
            focal_fits[fmm] = round(fit, 5)
        if best is None or fit < best[0]:
            best = (fit, fmm, ca, cb, R2, d2 * scale, err)
    if len(focal_fits) >= 4:
        # the probe fits are noisy; the bottom of a parabola through all of them (in log focal)
        # is steadier than the single lowest candidate
        x = np.log(np.array(list(focal_fits), float))
        y = np.array(list(focal_fits.values()))
        a, b, _c = np.polyfit(x, y, 2)
        if a > 0:
            fmm = float(np.clip(np.exp(-b / (2 * a)), FOCAL_CANDIDATES[0], FOCAL_CANDIDATES[-1]))
            ca, cb = _with_focal(cam_a, fmm), _with_focal(cam_b, fmm)
            pts = shared_points(kp_a, sc_a, ca, kp_b, sc_b, cb)
            R2, d2, err = refine_epipolar(best[4], best[5], pts[0], pts[1], (ca[0] + cb[0]) / 2)
            scale = metric_scale(sk, R2, d2, kp_a, sc_a, ca, kp_b, sc_b, cb)
            if scale and err <= 6.0:
                best = (0.0, round(fmm, 1), ca, cb, R2, d2 * scale, err)
    if best is not None:
        _fit, focal_mm, cam_a, cam_b, R, t, epi_err = best
        method = "epipolar"
    found = (R, t, agree)
    R, t, agree = found
    kp = np.stack([kp_a, kp_b], axis=1)
    sc = np.stack([sc_a, sc_b], axis=1)
    sol = None
    rounds = rounds or int((cfg or {}).get("calibration_rounds", 1 if method == "epipolar" else 2))
    for r in range(rounds):
        views = [(cam_b, R, t)]
        sol = solver.solve_sequence(sk, kp, sc, cam_a, fps, cfg, stage(f"both cameras {r + 1}", 0), views=views)
        if method != "epipolar":
            R, t = refine(sk, sol, kp_b, sc_b, cam_b, R, t)
    info = {"agreement": agree, "method_epipolar": float(method == "epipolar"), "epipolar_px": epi_err if epi_err is not None else -1.0, "R": R, "t": t, "angle_deg": float(np.degrees(np.linalg.norm(matrix_to_rotvec(R)))),
            "baseline_m": float(np.linalg.norm(-R.T @ t))}
    if focal_mm is not None:
        info["focal_mm"] = float(focal_mm)
    return sol, [(cam_b, R, t)], (kp, sc), info, cam_a
