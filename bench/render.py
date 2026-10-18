"""Small orthographic renders for the benchmark report.

Cycles on the CPU, so it also works in background mode on machines whose
OpenGL cannot render headless.
"""
import os

import bpy
from mathutils import Vector

VIEWS = {
    "front": (Vector((0, -1, 0)), (1.5708, 0.0, 0.0)),
    "side": (Vector((1, 0, 0)), (1.5708, 0.0, 1.5708)),
}


def setup(resolution=384, samples=8):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True  # OpenImageDenoise, runs on the CPU
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    # Plain colours: no filmic tone mapping, a bright sky and a key light
    scene.view_settings.view_transform = 'Standard'
    world = bpy.data.worlds.get("bench") or bpy.data.worlds.new("bench")
    world.color = (0.55, 0.55, 0.58)
    scene.world = world
    sun = bpy.data.objects.get("bench_sun")
    if sun is None:
        light = bpy.data.lights.new("bench_sun", 'SUN')
        light.energy = 2.0
        light.angle = 0.5
        sun = bpy.data.objects.new("bench_sun", light)
        scene.collection.objects.link(sun)
    sun.rotation_euler = (0.9, 0.3, -0.6)
    cam = bpy.data.objects.get("bench_cam")
    if cam is None:
        cam = bpy.data.objects.new("bench_cam", bpy.data.cameras.new("bench_cam"))
        scene.collection.objects.link(cam)
    cam.data.type = 'ORTHO'
    scene.camera = cam
    return cam


def bounds(objects):
    deps = bpy.context.evaluated_depsgraph_get()
    pts = []
    for obj in objects:
        ev = obj.evaluated_get(deps)
        pts += [ev.matrix_world @ Vector(c) for c in ev.bound_box]
    lo = Vector(map(min, *pts))
    hi = Vector(map(max, *pts))
    return lo, hi


def render(objects, path, view="front", center=None, size=None, cut=None):
    """Render with only 'objects' visible, framed on their bounds (or center/size).

    cut: hide everything nearer to the camera than this distance in front of
    the center (e.g. an arm hanging in front of the leg being looked at).
    """
    cam = setup()
    hidden = []
    for obj in bpy.context.scene.objects:
        if obj.type in {'MESH', 'ARMATURE', 'EMPTY'} and obj not in objects:
            if not obj.hide_render:
                obj.hide_render = True
                hidden.append(obj)
    lo, hi = bounds(objects)
    c = center if center is not None else (lo + hi) / 2
    s = size if size is not None else max(hi - lo) * 1.1
    direction, rotation = VIEWS[view]
    cam.data.ortho_scale = s
    cam.data.clip_end = s * 20
    cam.location = c + direction * s * 5
    cam.data.clip_start = s * 5 - cut if cut else 0.01
    cam.rotation_euler = rotation
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    for obj in hidden:
        obj.hide_render = False
    return path
