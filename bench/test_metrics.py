"""Checks that the benchmark measures react the right way on synthetic shapes.

    python -I bench/test_metrics.py        (any Python with numpy)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402

failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def cylinder(radius=0.05, length=1.0, rings=41, sides=16):
    """Tube along +Z from 0 to length, as (points, quads)."""
    a = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    z = np.linspace(0, length, rings)
    pts = np.array([(radius * math.cos(t), radius * math.sin(t), zz) for zz in z for t in a])
    quads = []
    for r in range(rings - 1):
        for s in range(sides):
            n = (s + 1) % sides
            quads.append([r * sides + s, r * sides + n, (r + 1) * sides + n, (r + 1) * sides + s])
    return pts, quads


def rot_z(points, angles):
    c, s = np.cos(angles), np.sin(angles)
    out = points.copy()
    out[:, 0] = c * points[:, 0] - s * points[:, 1]
    out[:, 1] = s * points[:, 0] + c * points[:, 1]
    return out


pts, quads = cylinder()
tris = metrics.triangles(quads)
head, tail = np.zeros(3), np.array([0, 0, 1.0])
everything = np.ones(len(pts), bool)

# Rigid motion: no distortion at all
moved = pts + np.array([0.3, -0.2, 0.1])
d = metrics.area_distortion(pts, moved, tris, everything)
check(d["crushed"] == 0 and d["stretched"] == 0 and d["rms_log"] < 1e-9, "rigid motion: no area distortion")
seg = [(head, tail)]
seg_moved = [(head + [0.3, -0.2, 0.1], tail + [0.3, -0.2, 0.1])]
t = metrics.thickness_ratio(pts, moved, everything, seg, seg_moved)
check(np.allclose(t, 1.0), "rigid motion: thickness ratio 1")

# Rigid 135 degree bend of the top half around the middle: every vertex keeps its
# distance to its own bone, even where the other bone swings close to it
pivot = np.array([0, 0, 0.5])
c, s_ = math.cos(math.radians(135)), math.sin(math.radians(135))
rot = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
bent = pts.copy()
top = pts[:, 2] > 0.5
bent[top] = (pts[top] - pivot) @ rot.T + pivot
segs_rest = [(head, pivot), (pivot, tail)]
segs_bent = [(head, pivot), (pivot, pivot + rot @ (tail - pivot))]
t = metrics.thickness_ratio(pts, bent, everything, segs_rest, segs_bent)
check(np.allclose(t, 1.0, atol=1e-6), f"rigid 135 degree bend: thickness 1 (min {t.min():.3f})")

# Uniform squash towards the axis: thickness drops by the same factor
squashed = pts.copy()
squashed[:, :2] *= 0.5
t = metrics.thickness_ratio(pts, squashed, everything, seg, seg)
check(np.allclose(t, 0.5), "squash to half radius: thickness 0.5")
check(metrics.area_distortion(pts, squashed, tris, everything)["rms_log"] > 0.5, "squash shows area loss")
check(abs(metrics.signed_volume(squashed, tris) / metrics.signed_volume(pts, tris) - 0.25) < 0.02,
      "volume of half-radius tube is a quarter")

# Ideal twist: angle grows linearly along the tube
angle = math.radians(90)
ideal = rot_z(pts, angle * pts[:, 2])
p = metrics.twist_profile(pts, ideal, everything, head, tail, angle)
check(p["rms_error"] < 0.005 and abs(p["end_fraction"] - 1) < 0.01 and abs(p["min_radius"] - 1) < 1e-6,
      f"linear twist: error ~0, end ~1 ({p['rms_error']:.3f}, {p['end_fraction']:.3f})")

# Rigid twist of the whole tube: large error against the even spread
rigid = rot_z(pts, np.full(len(pts), angle))
p = metrics.twist_profile(pts, rigid, everything, head, tail, angle)
check(p["rms_error"] > 0.4, f"rigid twist: large error ({p['rms_error']:.3f})")

# Candy wrapper: linear blend of untwisted and twisted copies pinches the middle
w = pts[:, 2:3]
lbs = (1 - w) * pts + w * rigid
p = metrics.twist_profile(pts, lbs, everything, head, tail, angle)
check(p["min_radius"] < 0.8, f"linear blend twist pinches (min radius {p['min_radius']:.3f})")

# Weight report: bleed across the hierarchy, cross-side weights, finger leaks
names = ["DEF-hips", "DEF-spine", "DEF-head", "DEF-thigh.L", "DEF-thigh.R", "DEF-index.L", "DEF-middle.L",
         "DEF-hand.L", "DEF-forearm.L", "DEF-forearm_twist.01.L"]
parents = {"DEF-hips": None, "DEF-spine": "DEF-hips", "DEF-head": "DEF-spine", "DEF-thigh.L": "DEF-hips",
           "DEF-thigh.R": "DEF-hips", "DEF-forearm.L": "DEF-spine", "DEF-forearm_twist.01.L": "DEF-forearm.L",
           "DEF-hand.L": "DEF-forearm_twist.01.L", "DEF-index.L": "DEF-hand.L", "DEF-middle.L": "DEF-hand.L"}
W = np.zeros((5, len(names)))
W[0, 0] = 1.0                                   # clean hips vertex
W[1, 2], W[1, 7] = 0.8, 0.2                     # head vertex with a hand weight: bleed
W[2, 3], W[2, 4] = 0.6, 0.4                     # crotch vertex on both thighs: cross side
W[3, 5], W[3, 6] = 0.7, 0.3                     # index tip on the middle finger: finger leak
W[4, 8], W[4, 9] = 0.5, 0.5                     # forearm and its twist bone: same bone, fine
r = metrics.weight_report(W, names, parents)
check(abs(r["bleed"] - 0.2) < 1e-9, f"one of five vertices bleeds ({r['bleed']})")
check(abs(r["cross_side"] - 0.2) < 1e-9, "one of five vertices is cross side")
check(r["finger_bleed"] == 0.0, "index/middle are 2 hops apart: not a finger leak at 3 hops")
check(list(r["bleed_pairs"]) == ["DEF-head <- DEF-hand.L"], f"bleed pair named ({r['bleed_pairs']})")
check(metrics.bone_graph_distance(names, parents)[8, 9] == 0, "twist bone merged into its limb bone")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
