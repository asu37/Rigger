"""Benchmark bodies: loading them into the scene and their frozen marker positions."""
import json
import os

import bpy

from fetch_assets import BLEND, fetch

HERE = os.path.dirname(os.path.abspath(__file__))
LANDMARKS = os.path.join(HERE, "landmarks.json")

# Benchmark name -> object name in the Human Base Meshes bundle. Rigger's defaults were
# tuned on these four.
BODIES = {
    "realistic_male": "GEO-body_male_realistic",
    "realistic_female": "GEO-body_female_realistic",
    "stylized_male": "GEO-body_male_stylized",
    "stylized_female": "GEO-body_female_stylized",
}

# Held out: never used to tune Rigger, only to check that improvements carry over.
# Blockout mannequins made of ~50 separate low-poly pieces (a collection each),
# joined into one mesh when loaded.
HELD_OUT = {
    "primitive_male_realistic": "Body Male - Primitve (Realistic)",
    "primitive_female_realistic": "Body Female - Primitve (Realistic)",
    "primitive_male_stylized": "Body Male - Primitve (Stylized)",
    "primitive_female_stylized": "Body Female - Primitive (Stylized)",
}
ALL_BODIES = {**BODIES, **HELD_OUT}


def load_body(name):
    """Append one body and its eyeballs to the current scene. Returns (body, [eyes])."""
    fetch()
    if name in HELD_OUT:
        return _load_pieces(name), []
    base = BODIES[name]
    wanted = [base, base + ".eye.L", base + ".eye.R"]
    with bpy.data.libraries.load(BLEND, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n in wanted]
    scene = bpy.context.scene
    body, eyes = None, []
    for obj in dst.objects:
        scene.collection.objects.link(obj)
        if obj.name.startswith(base + ".eye"):
            eyes.append(obj)
        else:
            body = obj
    # Bench the base mesh: multires levels would only slow the solver down.
    for mod in list(body.modifiers):
        body.modifiers.remove(mod)
    # The bundle lays the bodies out side by side: centre this one on the origin
    # (eyes follow as children), then free the eyes as separate objects.
    body.location = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    for eye in eyes:
        mw = eye.matrix_world.copy()
        eye.parent = None
        eye.matrix_world = mw
    bpy.context.view_layer.update()
    body.name = name
    return body, sorted(eyes, key=lambda o: o.name)


def _load_pieces(name):
    """Join a collection of body pieces into one mesh, centred on X/Y with its feet at Z = 0."""
    with bpy.data.libraries.load(BLEND, link=False) as (src, dst):
        dst.collections = [HELD_OUT[name]]
    coll = dst.collections[0]
    scene = bpy.context.scene
    pieces = [o for o in coll.all_objects if o.type == 'MESH' and "eye" not in o.name.lower()]
    for o in pieces:
        scene.collection.objects.link(o)
    bpy.context.view_layer.update()
    for o in pieces:
        mw = o.matrix_world.copy()
        o.parent = None
        o.matrix_world = mw
        for mod in list(o.modifiers):
            o.modifiers.remove(mod)
        if o.data.users > 1:  # left/right pieces share a mirrored mesh
            o.data = o.data.copy()
    bpy.context.view_layer.update()
    ctx = bpy.context
    for o in ctx.view_layer.objects:
        o.select_set(o in pieces)
    ctx.view_layer.objects.active = pieces[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.join()
    body = ctx.view_layer.objects.active
    import numpy as np
    co = np.empty(len(body.data.vertices) * 3)
    body.data.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    shift = np.array([(co[:, 0].min() + co[:, 0].max()) / 2, (co[:, 1].min() + co[:, 1].max()) / 2, co[:, 2].min()])
    body.data.vertices.foreach_set("co", (co - shift).ravel())
    body.data.update()
    body.location = (0.0, 0.0, 0.0)
    body.name = name
    bpy.data.collections.remove(coll)
    bpy.context.view_layer.update()
    return body


def landmarks(name):
    with open(LANDMARKS, encoding="utf-8") as fh:
        return json.load(fh)[name]
