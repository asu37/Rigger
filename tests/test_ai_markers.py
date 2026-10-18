"""AI markers: keypoints to markers, image to world, and a clean message when the
model is not installed. With RIGGER_AI_TEST_DIR pointing at a folder holding the
model and onnxruntime (site/) and the benchmark bodies downloaded, also guesses
the markers of a real body.

    blender -b --factory-startup --python tests/test_ai_markers.py
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import bpy  # noqa: E402

import rigger  # noqa: E402
from rigger import ai_markers  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


# Keypoints -> markers (world X/Z already), character 1.8 m tall
pts = np.zeros((133, 2))
scores = np.ones(133)
K = ai_markers.KP
pts[K["l_shoulder"]], pts[K["r_shoulder"]] = (0.20, 1.40), (-0.18, 1.42)
pts[K["l_wrist"]], pts[K["r_wrist"]] = (0.40, 0.90), (-0.42, 0.92)
pts[K["l_hip"]], pts[K["r_hip"]] = (0.10, 0.90), (-0.10, 0.92)
pts[K["l_ankle"]], pts[K["r_ankle"]] = (0.11, 0.08), (-0.11, 0.08)
pts[K["chin"]] = (0.0, 1.56)
m, conf = ai_markers.markers_from_keypoints(pts, scores, 1.8)
check(set(m) == {"chin", "neck", "shoulder", "wrist", "pelvis", "ankle"}, f"six markers ({sorted(m)})")
check(np.allclose(m["shoulder"], (0.19, 1.41)), "left and right shoulders averaged on the left side")
check(np.allclose(m["pelvis"], (0.0, 0.91 + ai_markers.PELVIS_RAISE * 1.8)), "pelvis above the hip joints")
check(np.isclose(m["neck"][1], 1.41 + (1.56 - 1.41) * ai_markers.NECK_SHARE), "neck between shoulders and chin")
scores2 = scores.copy()
scores2[K["l_wrist"]] = 0.1
m2, conf2 = ai_markers.markers_from_keypoints(pts, scores2, 1.8)
check(np.allclose(m2["wrist"], (0.42, 0.92)) and conf2["wrist"] < 0.5,
      "an unsure wrist falls back to the other side and is flagged")
scores3 = scores.copy()
scores3[K["chin"]] = 0.1
m3, _c = ai_markers.markers_from_keypoints(pts, scores3, 1.8)
check("chin" not in m3 and "neck" not in m3, "no chin: no chin or neck marker")

# Image pixel -> world on the front view
view = (0.1, 0.9, 1.5, 2.0)
x, z = ai_markers.to_world((ai_markers.INPUT_W / 2, ai_markers.INPUT_H / 2), view)
check(abs(x - 0.1) < 1e-9 and abs(z - 0.9) < 1e-9, "image centre is the view centre")
x, z = ai_markers.to_world((ai_markers.INPUT_W, 0), view)
check(abs(x - 0.85) < 1e-9 and abs(z - 1.9) < 1e-9, "image top right is the view's top right")

# Not installed: clear status and a readable operator error
os.environ["RIGGER_AI_DIR"] = tempfile.mkdtemp(prefix="rigger_ai_empty_")
ready, message = ai_markers.status()
check(not ready and "missing" in message, f"status without the files ({message})")
bpy.ops.mesh.primitive_cube_add()
try:
    bpy.ops.rigger.guess_markers()
    error = ""
except RuntimeError as exc:
    error = str(exc)
check("not installed" in error and "Install AI Model" in error, f"guess explains what to install ({error.strip()[:80]})")

# Full run on a real body when the model and the benchmark assets are available
model_dir = os.environ.get("RIGGER_AI_TEST_DIR")
if model_dir and os.path.isfile(os.path.join(model_dir, ai_markers.MODEL_FILE)):
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    import bodies
    from fetch_assets import BLEND
    if os.path.isfile(BLEND):
        os.environ["RIGGER_AI_DIR"] = model_dir
        os.environ.setdefault("RIGGER_AI_ENGINE", "CYCLES")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        body, eyes = bodies.load_body("realistic_female")
        for eye in eyes:
            bpy.data.objects.remove(eye)
        bpy.ops.object.select_all(action='DESELECT')
        body.select_set(True)
        ctx.view_layer.objects.active = body
        res = bpy.ops.rigger.guess_markers()
        truth = bodies.landmarks("realistic_female")
        errors = {k: np.linalg.norm(np.array(bpy.data.objects["RGM-" + k].location) - np.array(v)) * 100
                  for k, v in truth.items() if bpy.data.objects.get("RGM-" + k)}
        check(res == {'FINISHED'} and len(errors) == 6 and max(errors.values()) < 6.0,
              "markers guessed on a real body within 6 cm ("
              + ", ".join(f"{k} {e:.1f}" for k, e in errors.items()) + ")")
else:
    print("SKIP real body (set RIGGER_AI_TEST_DIR to a folder with the model and onnxruntime)")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
