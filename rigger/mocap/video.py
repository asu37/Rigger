# SPDX-License-Identifier: GPL-3.0-or-later
"""Read video frames with Blender's own FFmpeg (no extra packages).

The movie is placed in the sequencer of a throw-away scene and rendered, a
chunk of frames at a time, to JPEG files that are read back as arrays. Colour
management is set to pass the pixels through unchanged. Blender applies the
file's rotation metadata itself.
"""

import os
import shutil
import tempfile

import bpy
import numpy as np


def probe(path):
    """-> dict(fps, frames, width, height). Raises ValueError for unreadable files."""
    if not os.path.isfile(path):
        raise ValueError(f"video not found: {path}")
    try:
        clip = bpy.data.movieclips.load(path)
    except RuntimeError as exc:
        raise ValueError(f"cannot read video {path}: {exc}") from None
    try:
        info = {"fps": float(clip.fps) or 30.0, "frames": int(clip.frame_duration),
                "width": int(clip.size[0]), "height": int(clip.size[1])}
    finally:
        bpy.data.movieclips.remove(clip)
    if info["frames"] < 1 or info["width"] < 8:
        raise ValueError(f"no frames in {path}")
    return info


def _strips(scene):
    se = scene.sequence_editor_create()
    return se.strips if hasattr(se, "strips") else se.sequences


def frames(path, start=0, count=None, max_side=1280, chunk=48):
    """Yield (index, RGB uint8 array (H, W, 3)) for frames start .. start+count-1."""
    info = probe(path)
    total = info["frames"] if count is None else min(count, info["frames"] - start)
    scale = min(1.0, max_side / max(info["width"], info["height"]))
    scene = bpy.data.scenes.new("rigger_video_decode")
    tmp = tempfile.mkdtemp(prefix="rigger_video_")
    try:
        strip = _strips(scene).new_movie("video", path, 1, 1)
        r = scene.render
        r.resolution_x, r.resolution_y = info["width"], info["height"]
        r.resolution_percentage = max(1, int(round(scale * 100)))
        r.fps = max(1, int(round(info["fps"])))
        r.fps_base = r.fps / info["fps"]
        r.use_sequencer = True
        r.use_compositing = False
        r.image_settings.file_format = 'JPEG'
        r.image_settings.quality = 95
        r.image_settings.color_mode = 'RGB'
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.sequencer_colorspace_settings.name = 'sRGB'
        del strip
        done = 0
        while done < total:
            n = min(chunk, total - done)
            first = start + done + 1
            scene.frame_start, scene.frame_end = first, first + n - 1
            r.filepath = os.path.join(tmp, "f_")
            bpy.ops.render.render(animation=True, scene=scene.name)
            for i in range(n):
                fpath = os.path.join(tmp, f"f_{first + i:04d}.jpg")
                if not os.path.isfile(fpath):
                    fpath = os.path.join(tmp, f"f_{first + i:05d}.jpg")
                yield start + done + i, read_image(fpath)
                os.remove(fpath)
            done += n
    finally:
        bpy.data.scenes.remove(scene)
        shutil.rmtree(tmp, ignore_errors=True)


def read_image(path):
    img = bpy.data.images.load(path)
    try:
        w, h = img.size
        px = np.empty(w * h * 4, np.float32)
        img.pixels.foreach_get(px)
    finally:
        bpy.data.images.remove(img)
    return (px.reshape(h, w, 4)[::-1, :, :3] * 255 + 0.5).astype(np.uint8)


def file_hash(path):
    """Content hash for the cache: size plus the first, middle and last MiB."""
    import hashlib
    h = hashlib.sha1()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as fh:
        for pos in (0, max(0, size // 2 - 2 ** 19), max(0, size - 2 ** 20)):
            fh.seek(pos)
            h.update(fh.read(2 ** 20))
    return h.hexdigest()[:16]
