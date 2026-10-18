"""How close AI-guessed markers land to the frozen landmarks of each body.

    python bench/fetch_assets.py
    RIGGER_AI_DIR=<folder with dw-ll_ucoco_384.onnx and onnxruntime in site/> \\
    RIGGER_AI_ENGINE=CYCLES blender -b --factory-startup --python bench/ai_markers_eval.py

Prints the error of every marker in centimetres and as a share of body height,
and writes the rendered input with the detected keypoints to
bench/results/ai_markers/<body>.png.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import bpy  # noqa: E402

import bodies  # noqa: E402
import rigger  # noqa: E402
from rigger import ai_markers  # noqa: E402

OUT = os.path.join(HERE, "results", "ai_markers")


def save_overlay(rgb, points, scores, path):
    img = rgb.copy()
    for (x, y), s in zip(points, scores):
        if s < ai_markers.MIN_SCORE:
            continue
        xi, yi = int(round(x)), int(round(y))
        img[max(0, yi - 2):yi + 3, max(0, xi - 2):xi + 3] = (255, 40, 40)
    for name, i in ai_markers.KP.items():
        x, y = points[i]
        xi, yi = int(round(x)), int(round(y))
        img[max(0, yi - 3):yi + 4, max(0, xi - 3):xi + 4] = (40, 255, 80)
    h, w = img.shape[:2]
    im = bpy.data.images.new("overlay", w, h)
    rgba = np.concatenate([img[::-1] / 255.0, np.ones((h, w, 1))], axis=2)
    im.pixels[:] = rgba.ravel()
    im.filepath_raw = path
    im.file_format = 'PNG'
    im.save()
    bpy.data.images.remove(im)


def main():
    os.makedirs(OUT, exist_ok=True)
    rigger.register()
    report = {}
    for name in bodies.BODIES:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        body, eyes = bodies.load_body(name)
        for eye in eyes:
            bpy.data.objects.remove(eye)
        placed, conf, (points, scores, rgb) = ai_markers.guess(bpy.context, [body])
        save_overlay(rgb, points, scores, os.path.join(OUT, f"{name}.png"))
        truth = bodies.landmarks(name)
        zs = [v.co.z for v in body.data.vertices]
        h = max(zs) - min(zs)
        errs = {}
        for key, true in truth.items():
            if key not in placed:
                errs[key] = None
                continue
            d = np.linalg.norm(np.array(placed[key]) - np.array(true))
            errs[key] = round(float(d) * 100, 1)
        report[name] = {"error_cm": errs, "score": {k: round(v, 2) for k, v in conf.items()}, "height_m": round(h, 3)}
        print(name, json.dumps(report[name]), flush=True)
    allerr = [e for r in report.values() for e in r["error_cm"].values() if e is not None]
    print("MEAN ERROR cm", round(float(np.mean(allerr)), 1), "MAX", max(allerr), flush=True)
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


main()
