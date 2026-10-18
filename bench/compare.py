"""Head-to-head scoreboard between two benchmark runs.

    python bench/compare.py A/results.json:config B/results.json:config [--md out.md]

Every measure of every pose is averaged over the bodies of each set (tuned /
held out, see bodies.py), and the closer-to-ideal side wins it: volume, bulge
and thinnest slice ideally 1, squashed skin, stretch and twist error ideally 0.
Differences smaller than the noise margin count as ties. Lists the measures the
first run loses, worst first.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TUNED = ("realistic_male", "realistic_female", "stylized_male", "stylized_female")

MEASURES = [  # (label, getter on a joint entry, ideal: 'one' or 'zero')
    ("volume", lambda j: j["thickness"]["p05"], "one"),
    ("bulge", lambda j: j["thickness"]["p95"], "one"),
    ("squashed", lambda j: j["crushed"], "zero"),
    ("stretch", lambda j: j["rms_log"], "zero"),
]
TWIST = [("twist error", lambda t: t["rms_error"], "zero"), ("thinnest", lambda t: t["min_radius"], "one")]
MARGIN = 0.01  # absolute difference in distance-to-ideal below which it is a tie


def load(spec):
    path, _, config = spec.partition(":")
    data = json.load(open(path, encoding="utf-8"))
    config = config or next(k for k in data if not k.startswith("_"))
    return data[config], config


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rows(a, b, bodies):
    out = []
    poses = [p for p in next(iter(a.values()))["poses"]]
    for pose in poses:
        sample = a[bodies[0]]["poses"].get(pose) if bodies[0] in a else None
        if sample is None:
            continue
        items = []
        for key, val in sample.items():
            if isinstance(val, dict) and "thickness" in val:
                items += [(f"{key} {lab}", lambda e, k=key, g=g: g(e[k]), ideal) for lab, g, ideal in MEASURES]
            if key == "twist":
                items += [(lab, lambda e, g=g: g(e["twist"]), ideal) for lab, g, ideal in TWIST]
        for label, getter, ideal in items:
            va = mean([getter(a[n]["poses"][pose]) for n in bodies if n in a and pose in a[n]["poses"]])
            vb = mean([getter(b[n]["poses"][pose]) for n in bodies if n in b and pose in b[n]["poses"]])
            if va is None or vb is None:
                continue
            da, db = (abs(va - 1), abs(vb - 1)) if ideal == "one" else (va, vb)
            winner = "tie" if abs(da - db) < MARGIN else ("A" if da < db else "B")
            out.append((pose, label, va, vb, da - db, winner))
    return out


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
    a, na = load(args[0])
    b, nb = load(args[1])
    lines = [f"# {na} (A) vs {nb} (B)", ""]
    for label, names in (("tuned bodies", [n for n in TUNED if n in a and n in b]),
                         ("held-out bodies", [n for n in a if n not in TUNED and n in b])):
        if not names:
            continue
        r = rows(a, b, names)
        wins = {w: sum(1 for x in r if x[5] == w) for w in ("A", "B", "tie")}
        lines += [f"## {label} ({len(names)}): A wins {wins['A']}, B wins {wins['B']}, ties {wins['tie']}", ""]
        joints = {}
        for pose, lab, *_rest, w in r:
            joint = lab.split()[0] if lab.split()[0] in ("elbow", "knee", "shoulder", "hip", "wrist", "ankle",
                                                         "waist", "neck") else "twist"
            joints.setdefault(joint, {"A": 0, "B": 0, "tie": 0})[w] += 1
        lines += ["| joint | A wins | B wins | ties |", "| --- | --- | --- | --- |"]
        lines += [f"| {j} | {c['A']} | {c['B']} | {c['tie']} |" for j, c in sorted(joints.items())]
        lines += ["", f"A loses ({label}), worst first:", "", "| pose | measure | A | B | gap |",
                  "| --- | --- | --- | --- | --- |"]
        losses = sorted((x for x in r if x[5] == "B"), key=lambda x: -x[4])
        lines += [f"| {p} | {lab} | {va:.3f} | {vb:.3f} | {gap:.3f} |" for p, lab, va, vb, gap, _w in losses[:40]]
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if md:
        with open(md, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
