# SPDX-License-Identifier: GPL-3.0-or-later
"""Guess the Smart Markers with a pose estimation model.

The character is rendered from the front, a whole-body pose model (DWPose,
COCO-WholeBody keypoints, run with onnxruntime) finds the joints in the image,
and each joint is brought back to 3D: the image gives X and Z, and the depth is
the middle of the mesh along a ray from the front, like a marker clicked by
hand. The markers are placed, not fitted: check them, then Fit Reference to
Markers as usual.

The model and onnxruntime are not shipped with the add-on. They live in the AI
folder (see ai_dir), downloaded once from the add-on preferences or copied
there by hand.
"""

import os
import sys

import numpy as np

MODEL_FILE = "dw-ll_ucoco_384.onnx"
MODEL_URL = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
MODEL_SHA256 = "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"
INPUT_W, INPUT_H = 288, 384
SIMCC_RATIO = 2.0
PADDING = 1.25          # crop padding around the character, as the model was trained
MIN_SCORE = 0.3         # keypoints scored lower count as not found

# COCO-WholeBody indices (the person's left, which faces the viewer, is on the image's right)
KP = {"l_shoulder": 5, "r_shoulder": 6, "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12,
      "l_ankle": 15, "r_ankle": 16, "chin": 23 + 8}
NECK_SHARE = 0.5        # neck base: this share of the way from the shoulder line to the chin
PELVIS_RAISE = 0.02     # spine root above the hip joints, in body heights


def ai_dir():
    """Folder holding the model and onnxruntime (RIGGER_AI_DIR overrides it)."""
    override = os.environ.get("RIGGER_AI_DIR")
    if override:
        return override
    import bpy
    return bpy.utils.user_resource('DATAFILES', path=os.path.join("rigger", "ai"), create=True)


def model_path():
    return os.path.join(ai_dir(), MODEL_FILE)


def load_runtime():
    """onnxruntime from the AI folder (site/) or the Python environment, or None."""
    site = os.path.join(ai_dir(), "site")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.insert(0, site)
    try:
        import onnxruntime
    except ImportError:
        return None
    return onnxruntime


def status():
    """(ready, message) for the UI."""
    if load_runtime() is None:
        return False, "onnxruntime is missing"
    if not os.path.isfile(model_path()):
        return False, f"model file {MODEL_FILE} is missing"
    return True, "ready"


_session = {}


def session():
    ort = load_runtime()
    if ort is None:
        raise RuntimeError("onnxruntime is not installed (Rigger preferences > AI Markers)")
    path = model_path()
    if not os.path.isfile(path):
        raise RuntimeError(f"AI model not found: {path}")
    if _session.get("path") != path:
        _session["path"] = path
        _session["session"] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    return _session["session"]


# ---------------------------------------------------------------------------
# Image -> keypoints
# ---------------------------------------------------------------------------

MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _peak(simcc, subpixel):
    """Argmax per row, or with subpixel the weighted mean of the bins around it above half the peak."""
    idx = simcc.argmax(axis=1)
    if not subpixel:
        return idx.astype(np.float64)
    n = simcc.shape[1]
    offs = np.arange(-3, 4)
    cols = np.clip(idx[:, None] + offs, 0, n - 1)
    vals = np.take_along_axis(simcc, cols, axis=1)
    w = np.maximum(vals - 0.5 * vals.max(axis=1, keepdims=True), 0)
    return (w * cols).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)


def detect(rgb, subpixel=False):
    """Keypoints of an (INPUT_H, INPUT_W, 3) RGB image with values 0..255.

    Returns (points (133, 2) in image pixels, x right and y down; scores (133,))."""
    x = ((rgb.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    simcc_x, simcc_y = session().run(None, {"input": x})
    ix = _peak(simcc_x[0], subpixel)
    iy = _peak(simcc_y[0], subpixel)
    scores = np.minimum(simcc_x[0].max(axis=1), simcc_y[0].max(axis=1))
    points = np.stack([ix, iy], axis=1) / SIMCC_RATIO
    return points, scores


# ---------------------------------------------------------------------------
# Rendering the character
# ---------------------------------------------------------------------------

def frame(meshes):
    """Orthographic front view framing: (center x, center z, view width, view height, front y)."""
    from mathutils import Vector
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector(map(min, *pts))
    hi = Vector(map(max, *pts))
    h = (hi.z - lo.z) * PADDING
    w = max((hi.x - lo.x) * PADDING, h * INPUT_W / INPUT_H)
    h = w * INPUT_H / INPUT_W
    return (lo.x + hi.x) / 2, (lo.z + hi.z) / 2, w, h, lo.y


def render_front(context, meshes, engine=None):
    """Render the meshes from the front into an (INPUT_H, INPUT_W, 3) array."""
    import tempfile

    import bpy
    scene = context.scene
    cx, cz, w, h, front = frame(meshes)
    saved = {"camera": scene.camera, "engine": scene.render.engine, "x": scene.render.resolution_x,
             "y": scene.render.resolution_y, "pct": scene.render.resolution_percentage,
             "path": scene.render.filepath, "world": scene.world,
             "transform": scene.view_settings.view_transform, "film": scene.render.film_transparent}
    hidden = [o for o in scene.objects if o.type in {'MESH', 'CURVE', 'EMPTY', 'ARMATURE'}
              and o not in meshes and not o.hide_render]
    cam_data = bpy.data.cameras.new("rigger_ai_cam")
    cam = bpy.data.objects.new("rigger_ai_cam", cam_data)
    world = bpy.data.worlds.new("rigger_ai_world")
    tmp = os.path.join(tempfile.mkdtemp(prefix="rigger_ai_"), "front.png")
    try:
        scene.collection.objects.link(cam)
        cam_data.type = 'ORTHO'
        cam_data.ortho_scale = max(w, h)
        cam_data.sensor_fit = 'AUTO'
        cam_data.clip_end = 1000.0
        cam.location = (cx, front - h * 5, cz)
        cam.rotation_euler = (1.5707963, 0.0, 0.0)
        scene.camera = cam
        for o in hidden:
            o.hide_render = True
        world.color = (0.18, 0.18, 0.18)
        scene.world = world
        scene.render.film_transparent = False
        scene.view_settings.view_transform = 'Standard'
        scene.render.resolution_x, scene.render.resolution_y = INPUT_W, INPUT_H
        scene.render.resolution_percentage = 100
        scene.render.filepath = tmp
        scene.render.engine = engine or os.environ.get("RIGGER_AI_ENGINE", 'BLENDER_WORKBENCH')
        if scene.render.engine == 'BLENDER_WORKBENCH':
            shading = scene.display.shading
            shading.light, shading.color_type = 'STUDIO', 'SINGLE'
            shading.single_color = (0.8, 0.75, 0.7)
        elif scene.render.engine == 'CYCLES':
            scene.cycles.device, scene.cycles.samples = 'CPU', 8
            world.color = (0.5, 0.5, 0.5)
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(tmp)
        px = np.array(img.pixels[:], dtype=np.float32).reshape(img.size[1], img.size[0], img.channels)
        bpy.data.images.remove(img)
        rgb = np.clip(px[::-1, :, :3] * 255.0, 0, 255)   # Blender rows go bottom-up
    finally:
        for o in hidden:
            o.hide_render = False
        scene.camera = saved["camera"]
        scene.render.engine = saved["engine"]
        scene.render.resolution_x, scene.render.resolution_y = saved["x"], saved["y"]
        scene.render.resolution_percentage = saved["pct"]
        scene.render.filepath = saved["path"]
        scene.world = saved["world"]
        scene.view_settings.view_transform = saved["transform"]
        scene.render.film_transparent = saved["film"]
        bpy.data.objects.remove(cam)
        bpy.data.cameras.remove(cam_data)
        bpy.data.worlds.remove(world)
    return rgb, (cx, cz, max(w, h) * INPUT_W / max(INPUT_W, INPUT_H), max(w, h) * INPUT_H / max(INPUT_W, INPUT_H))


def to_world(point, view):
    """Image pixel -> world (x, z) on the front view."""
    cx, cz, w, h = view
    return cx + (point[0] / INPUT_W - 0.5) * w, cz + (0.5 - point[1] / INPUT_H) * h


# ---------------------------------------------------------------------------
# Keypoints -> markers
# ---------------------------------------------------------------------------

def markers_from_keypoints(points_xz, scores, height):
    """Six Smart Markers (x, z) on the front view from keypoints already in world X/Z.

    Left and right estimates are averaged (the markers are mirrored anyway).
    Returns ({key: (x, z)}, {key: score}); missing keys were not found."""
    def pick(name):
        i = KP[name]
        return (np.array(points_xz[i]), float(scores[i])) if scores[i] >= MIN_SCORE else (None, float(scores[i]))

    out, conf = {}, {}

    def side(key, left, right):
        (pl, sl), (pr, sr) = pick(left), pick(right)
        found = [(p, s) for p, s in ((pl, sl), (pr, sr)) if p is not None]
        if pl is not None and pr is not None:
            out[key] = np.array([(abs(pl[0]) + abs(pr[0])) / 2, (pl[1] + pr[1]) / 2])
        elif found:
            p = found[0][0]
            out[key] = np.array([abs(p[0]), p[1]])
        # the weaker side counts: one unsure side means the marker is worth checking
        conf[key] = min(sl, sr) if found else 0.0

    side("shoulder", "l_shoulder", "r_shoulder")
    side("wrist", "l_wrist", "r_wrist")
    side("ankle", "l_ankle", "r_ankle")
    (lh, slh), (rh, srh) = pick("l_hip"), pick("r_hip")
    if lh is not None and rh is not None:
        out["pelvis"] = np.array([0.0, (lh[1] + rh[1]) / 2 + PELVIS_RAISE * height])
    conf["pelvis"] = min(slh, srh)
    chin, sc = pick("chin")
    if chin is not None:
        out["chin"] = np.array([0.0, chin[1]])
    conf["chin"] = sc
    if "chin" in out and "shoulder" in out:
        z = out["shoulder"][1] + (out["chin"][1] - out["shoulder"][1]) * NECK_SHARE
        out["neck"] = np.array([0.0, z])
        conf["neck"] = min(conf["chin"], conf["shoulder"])
    return out, conf


def guess(context, meshes):
    """Run the model on the meshes. Returns ({marker: world point}, {marker: score}, keypoints)."""
    from mathutils import Vector

    from .markers import _volume_center
    rgb, view = render_front(context, meshes)
    points, scores = detect(rgb)
    world_xz = np.array([to_world(p, view) for p in points])
    zs = [(o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box]
    height = max(zs) - min(zs)
    flat, conf = markers_from_keypoints(world_xz, scores, height)
    depsgraph = context.evaluated_depsgraph_get()
    front = min((o.matrix_world @ Vector(c)).y for o in meshes for c in o.bound_box) - height
    placed = {}
    for key, (x, z) in flat.items():
        origin = Vector((x, front, z))
        hit = _volume_center(context, depsgraph, meshes, origin, Vector((0, 1, 0)))
        placed[key] = hit if hit is not None else Vector((x, 0.0, z))
    return placed, conf, (points, scores, rgb)


# ---------------------------------------------------------------------------
# Installing the optional files
# ---------------------------------------------------------------------------

ORT_VERSION = "1.31.0"


def _wheel_url():
    """URL and SHA-256 of the onnxruntime wheel for this Python and platform, from PyPI."""
    import json
    import platform
    import urllib.request
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    machine = platform.machine().lower()
    if sys.platform.startswith("win"):
        plat = ("win_amd64",)
    elif sys.platform == "darwin":
        plat = ("macosx",)
    else:
        plat = ("manylinux",) if machine in ("x86_64", "amd64") else ("aarch64",)
    with urllib.request.urlopen(f"https://pypi.org/pypi/onnxruntime/{ORT_VERSION}/json", timeout=30) as resp:
        files = json.load(resp)["urls"]
    for f in files:
        name = f["filename"]
        if f"-{tag}-" in name and any(p in name for p in plat) and \
                (sys.platform != "darwin" or "arm64" in name or "universal" in name or machine == "x86_64"):
            return f["url"], f["digests"]["sha256"]
    raise RuntimeError(f"No onnxruntime {ORT_VERSION} wheel for Python {tag} on {sys.platform}/{machine}")


def _download(url, path, sha256=None, progress=None):
    import hashlib
    import urllib.request
    h = hashlib.sha256()
    tmp = path + ".part"
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            h.update(block)
            done += len(block)
            if progress and total:
                progress(done / total)
    if sha256 and h.hexdigest() != sha256:
        os.remove(tmp)
        raise RuntimeError(f"Checksum mismatch for {os.path.basename(path)}")
    os.replace(tmp, path)


def install(progress=None):
    """Download onnxruntime (into site/) and the model into ai_dir(). Returns the folder."""
    import zipfile
    folder = ai_dir()
    os.makedirs(folder, exist_ok=True)
    if load_runtime() is None:
        url, digest = _wheel_url()
        wheel = os.path.join(folder, os.path.basename(url))
        _download(url, wheel, digest, progress)
        with zipfile.ZipFile(wheel) as zf:
            members = [m for m in zf.namelist() if m.startswith("onnxruntime/")]
            zf.extractall(os.path.join(folder, "site"), members)
        os.remove(wheel)
    if not os.path.isfile(model_path()):
        _download(MODEL_URL, model_path(), MODEL_SHA256, progress)
    _session.clear()
    return folder


# ---------------------------------------------------------------------------
# Operators and preferences
# ---------------------------------------------------------------------------

import bpy  # noqa: E402


class RIGGER_OT_guess_markers(bpy.types.Operator):
    """Place the six Smart Markers with an AI pose model (front view render of the selected mesh).
Check them, then Fit Reference to Markers"""
    bl_idname = "rigger.guess_markers"
    bl_label = "Guess Markers (AI)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .markers import set_marker
        from .safety import RiggerError
        meshes = [o for o in context.selected_objects if o.type == 'MESH'] or \
                 [o for o in context.visible_objects if o.type == 'MESH']
        if not meshes:
            raise RiggerError("Select the character mesh")
        ready, message = status()
        if not ready:
            raise RiggerError(f"AI markers are not installed ({message}): Rigger preferences > Install AI Model")
        if context.object and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        placed, conf, _kp = guess(context, meshes)
        if not placed:
            raise RiggerError("The AI found no body in the front view. Is the character facing -Y?")
        for key, point in placed.items():
            set_marker(context, key, point)
        weak = sorted(k for k, s in conf.items() if s < 0.5 or k not in placed)
        if weak:
            self.report({'WARNING'}, f"Placed {len(placed)} markers; check {', '.join(weak)}")
        else:
            self.report({'INFO'}, f"Placed {len(placed)} markers. Check them, then Fit Reference to Markers")
        return {'FINISHED'}


class RIGGER_OT_install_ai(bpy.types.Operator):
    """Download onnxruntime (about 15 MB) and the DWPose model (about 135 MB, Apache-2.0) for AI markers.
Blender waits until the download is done"""
    bl_idname = "rigger.install_ai"
    bl_label = "Install AI Model"

    def execute(self, context):
        from .safety import RiggerError
        if not getattr(bpy.app, "online_access", True):
            raise RiggerError("Online access is disabled (Preferences > System > Network). Allow it, or copy "
                              f"{MODEL_FILE} and onnxruntime into {ai_dir()}")
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            folder = install(lambda f: wm.progress_update(int(f * 100)))
        finally:
            wm.progress_end()
        ready, message = status()
        if not ready:
            raise RiggerError(f"Install finished but AI markers are not ready: {message}")
        self.report({'INFO'}, f"AI markers installed in {folder}")
        return {'FINISHED'}


class RiggerPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="AI Markers", icon='OUTLINER_OB_ARMATURE')
        ready, message = status()
        box.label(text=f"Status: {message}", icon='CHECKMARK' if ready else 'ERROR')
        box.label(text=f"Folder: {ai_dir()}")
        if not ready:
            box.operator("rigger.install_ai", icon='IMPORT')
        box.label(text="DWPose whole-body model (Apache-2.0) run with onnxruntime, both downloaded on request.")


classes = (RIGGER_OT_guess_markers, RIGGER_OT_install_ai, RiggerPreferences)
