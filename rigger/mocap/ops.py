# SPDX-License-Identifier: GPL-3.0-or-later
"""Video to Animation: operators, settings and panel.

The slow part (decoding, keypoints, solving) runs in a second, headless
Blender so this one stays usable; a modal operator follows its progress and
bakes the result onto the rig when it is done. Esc cancels.
"""

import json
import os
import queue
import subprocess
import tempfile
import threading

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, PointerProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from .. import ai_markers, utils
from ..safety import RiggerError
from . import bake, pipeline, schema, skeleton

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")


def cache_dir():
    return os.path.join(ai_markers.ai_dir(), "video_cache")


class RiggerVideoSettings(bpy.types.PropertyGroup):
    filepath: StringProperty(name="Video", subtype='FILE_PATH', description="Video of one person (MP4, MOV, ...)")
    motion_path: StringProperty(name="Motion File", subtype='FILE_PATH',
                                description="Also save the solved motion here (.mcap) to reuse it on other rigs")
    focal_mm: FloatProperty(name="Focal Length", default=pipeline.DEFAULT_FOCAL_MM, min=8, max=300, unit='CAMERA',
                            description="35 mm-equivalent focal length of the camera that filmed the video "
                                        "(phones: about 24-28 mm)")
    camera_tilt: FloatProperty(name="Camera Tilt", default=0.0, min=-60, max=60,
                               description="How far the camera looks down (degrees), if the body alone "
                                           "cannot tell; most tripod and handheld shots are close to 0")
    start: IntProperty(name="Start", default=0, min=0, description="First video frame to use")
    count: IntProperty(name="Frames", default=0, min=0, description="Number of video frames to use (0: all)")
    max_side: IntProperty(name="Max Resolution", default=1280, min=320, max=4096,
                          description="Frames are scaled down to this size for detection")
    smoothing: FloatProperty(name="Smoothing", default=0.5, min=0.0, max=1.0, subtype='FACTOR',
                             description="Steadier motion (higher) or quicker response (lower)")
    foot_lock: BoolProperty(name="Plant Feet", default=True,
                            description="Keep feet from sliding while they touch the floor and the body on the floor")
    fingers: BoolProperty(name="Fingers", default=True, description="Animate the fingers")
    face: BoolProperty(name="Face", default=True,
                       description="Drive shape keys with ARKit names (eyeBlinkLeft, jawOpen, ...)")
    camera: BoolProperty(name="Camera", default=True,
                         description="Add a camera matching the video, with the video as its background")
    simplify: FloatProperty(name="Simplify", default=0.0, min=0.0, max=0.05,
                            description="Drop keys that this tolerance can rebuild (0 keeps every frame)")
    status: StringProperty(name="Status", default="")
    progress: FloatProperty(name="Progress", default=0.0, min=0.0, max=1.0, subtype='FACTOR')


def solver_settings(s):
    """UI smoothing 0..1 -> One-Euro settings (0.5 gives the defaults)."""
    k = 2.0 ** (1.0 - 2.0 * s.smoothing)   # 0 -> 2x cutoff, 1 -> 0.5x
    return {"min_cutoff": round(1.2 * k, 3), "beta": round(0.4 * k, 3), "foot_lock": bool(s.foot_lock),
            "camera_tilt": float(s.camera_tilt)}


def _target_rig(context):
    rig = utils.find_rig_for(context.object)
    if rig is None:
        raise RiggerError("Select a Rigger rig (or a mesh bound to one)")
    return rig


def apply_motion(context, rig, path, s, video=None):
    meta, sk, arrays = schema.load(path)
    action = bake.bake(context, rig, meta, sk, arrays, start=context.scene.frame_start,
                       fingers=s.fingers, face=s.face, simplify=s.simplify, camera=s.camera, video=video,
                       name=f"{rig.name}_{os.path.splitext(os.path.basename(video or path))[0]}")
    return meta, action


class RIGGER_OT_video_to_animation(bpy.types.Operator):
    """Animate the rig from a video of a person: whole body, fingers and face"""
    bl_idname = "rigger.video_to_animation"
    bl_label = "Video to Animation"
    bl_options = {'REGISTER', 'UNDO'}

    wait: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'},
                       description="Run in this Blender and wait (scripts and background mode)")

    _proc = None
    _timer = None
    _lines = None

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def _job(self, context):
        s = context.scene.rigger_video
        video = bpy.path.abspath(s.filepath)
        if not video or not os.path.isfile(video):
            raise RiggerError("Choose a video file first")
        ready, message = ai_markers.status()
        if not ready:
            raise RiggerError(f"Pose model not ready: {message} (Rigger preferences > AI)")
        rig = _target_rig(context)
        sk = skeleton.from_rig(rig)
        out = bpy.path.abspath(s.motion_path) if s.motion_path else os.path.join(
            cache_dir(), f"{os.path.splitext(os.path.basename(video))[0]}_{sk.signature()}.mcap")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        return rig, {"video": video, "skeleton": sk.to_dict(), "out": out, "focal_mm": s.focal_mm,
                     "start": s.start, "count": s.count or None, "max_side": s.max_side, "cache_dir": cache_dir(),
                     "settings": solver_settings(s), "ai_dir": ai_markers.ai_dir()}

    def execute(self, context):
        rig, job = self._job(context)
        s = context.scene.rigger_video
        if self.wait or bpy.app.background:
            meta, _sk, _arr = pipeline.run(job["video"], job["skeleton"], out=job["out"], focal_mm=job["focal_mm"],
                                           start=job["start"], count=job["count"], max_side=job["max_side"],
                                           cache_dir=job["cache_dir"], settings=job["settings"])
            meta, action = apply_motion(context, rig, job["out"], s, job["video"])
            self.report({'INFO'}, f"Animated {meta['frames']} frames onto '{rig.name}' ({action.name})")
            return {'FINISHED'}
        fd, job_path = tempfile.mkstemp(suffix=".json", prefix="rigger_video_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(job, fh)
        self._job_data, self._job_path, self._rig_name = job, job_path, rig.name
        cmd = [bpy.app.binary_path, "-b", "--factory-startup", "--python", WORKER, "--", job_path]
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                      encoding="utf-8", errors="replace", creationflags=flags)
        self._lines = queue.Queue()
        threading.Thread(target=self._read, args=(self._proc, self._lines), daemon=True).start()
        self._timer = context.window_manager.event_timer_add(0.25, window=context.window)
        context.window_manager.modal_handler_add(self)
        s.status, s.progress = "Starting...", 0.0
        self._error = None
        return {'RUNNING_MODAL'}

    @staticmethod
    def _read(proc, lines):
        for line in proc.stdout:
            if line.startswith("RIGGER_JSON "):
                try:
                    lines.put(json.loads(line[len("RIGGER_JSON "):]))
                except ValueError:
                    pass
        lines.put({"exit": proc.wait()})

    def _finish(self, context):
        context.window_manager.event_timer_remove(self._timer)
        try:
            os.remove(self._job_path)
        except OSError:
            pass
        for area in context.screen.areas if context.screen else []:
            area.tag_redraw()

    def modal(self, context, event):
        s = context.scene.rigger_video
        if event.type == 'ESC':
            self._proc.terminate()
            self._finish(context)
            s.status, s.progress = "Cancelled", 0.0
            return {'CANCELLED'}
        if event.type != 'TIMER':
            return {'PASS_THROUGH'}
        done = None
        while not self._lines.empty():
            msg = self._lines.get()
            if "stage" in msg:
                share = {"keypoints": (0.0, 0.8), "solve": (0.8, 0.1), "floor": (0.9, 0.1)}.get(msg["stage"], (0, 1))
                s.progress = share[0] + share[1] * msg["done"] / max(msg["total"], 1)
                s.status = f"{'Reading the video' if msg['stage'] == 'keypoints' else 'Solving 3D'}: " \
                           f"{msg['done']}/{msg['total']}"
            elif "error" in msg:
                self._error = msg["error"]
            elif "done_file" in msg:
                done = msg["done_file"]
            elif "exit" in msg:
                done = done or ("" if msg["exit"] else self._job_data["out"])
        for area in context.screen.areas if context.screen else []:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
        if done is None:
            return {'PASS_THROUGH'}
        self._finish(context)
        if self._error or not done:
            s.status, s.progress = f"Failed: {self._error or 'the worker stopped'}", 0.0
            self.report({'ERROR'}, s.status)
            return {'CANCELLED'}
        rig = bpy.data.objects.get(self._rig_name)
        if rig is None:
            self.report({'ERROR'}, "The rig was removed")
            return {'CANCELLED'}
        try:
            meta, action = apply_motion(context, rig, done, s, self._job_data["video"])
        except (ValueError, RiggerError) as exc:
            s.status = f"Failed: {exc}"
            self.report({'ERROR'}, s.status)
            return {'CANCELLED'}
        s.status = f"Done: {meta['frames']} frames, person found in {meta['found'] * 100:.0f}%"
        s.progress = 1.0
        self.report({'INFO'}, f"Animated '{rig.name}' from the video ({action.name})")
        return {'FINISHED'}


class RIGGER_OT_video_apply(bpy.types.Operator, ImportHelper):
    """Bake a saved motion file (.mcap) onto the selected rig"""
    bl_idname = "rigger.video_apply"
    bl_label = "Apply Motion File"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".mcap"
    filter_glob: StringProperty(default="*.mcap", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def execute(self, context):
        rig = _target_rig(context)
        try:
            meta, action = apply_motion(context, rig, self.filepath, context.scene.rigger_video)
        except ValueError as exc:
            raise RiggerError(str(exc)) from None
        self.report({'INFO'}, f"Applied {meta['frames']} frames to '{rig.name}' ({action.name})")
        return {'FINISHED'}


class RIGGER_PT_video(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Rigger"
    bl_label = "Video to Animation"
    bl_parent_id = "RIGGER_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return utils.find_rig_for(context.object) is not None

    def draw(self, context):
        s = context.scene.rigger_video
        layout = self.layout
        layout.prop(s, "filepath", text="")
        col = layout.column(align=True)
        col.prop(s, "focal_mm")
        col.prop(s, "camera_tilt")
        row = col.row(align=True)
        row.prop(s, "start")
        row.prop(s, "count")
        col.prop(s, "smoothing", slider=True)
        row = layout.row(align=True)
        row.prop(s, "foot_lock", toggle=True)
        row.prop(s, "fingers", toggle=True)
        row.prop(s, "face", toggle=True)
        row.prop(s, "camera", toggle=True)
        sub = layout.column(align=True)
        sub.prop(s, "simplify")
        sub.prop(s, "motion_path", text="Save")
        ready, message = ai_markers.status()
        if not ready:
            layout.label(text=f"Pose model: {message}", icon='ERROR')
            layout.operator("rigger.install_ai", icon='IMPORT')
        layout.operator("rigger.video_to_animation", icon='FILE_MOVIE')
        if s.status:
            if 0 < s.progress < 1:
                layout.progress(factor=s.progress, text=s.status) if hasattr(layout, "progress") else \
                    layout.label(text=s.status)
                layout.label(text="Esc cancels", icon='INFO')
            else:
                layout.label(text=s.status, icon='INFO')
        layout.operator("rigger.video_apply", icon='IMPORT')


classes = (RiggerVideoSettings, RIGGER_OT_video_to_animation, RIGGER_OT_video_apply, RIGGER_PT_video)


def register_props():
    bpy.types.Scene.rigger_video = PointerProperty(type=RiggerVideoSettings)


def unregister_props():
    del bpy.types.Scene.rigger_video
