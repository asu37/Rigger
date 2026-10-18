"""Fit joints to mesh: elbow at the bend, knee at the narrowing, hips on the thigh
line, spine depth and foot length, on a synthetic body with known joints.

    blender -b --factory-startup --python tests/test_fitting.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import bmesh  # noqa: E402
import bpy  # noqa: E402
from mathutils import Vector  # noqa: E402

import rigger  # noqa: E402
from rigger import markers, utils  # noqa: E402

bpy.ops.wm.read_factory_settings(use_empty=True)
rigger.register()
ctx = bpy.context
failures = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


H = 1.8
# Known joints of the synthetic body (left side; mirrored for the right)
SHOULDER = Vector((0.19, 0.0, 1.43))
ELBOW = Vector((0.33, 0.03, 1.17))     # 58% of the way to the wrist, bent backward
WRIST = Vector((0.40, -0.04, 0.96))
HIP = Vector((0.10, 0.0, 0.92))
KNEE = Vector((0.11, -0.01, 0.55))     # 46% of the way down, where the leg is thinnest
ANKLE = Vector((0.11, 0.02, 0.08))
TOE = Vector((0.11, -0.17, 0.02))


def make_body():
    """Body built with the Skin modifier: radii make the knee the thinnest spot of the leg."""
    me = bpy.data.meshes.new("body")
    bm = bmesh.new()
    rad = bm.verts.layers.skin.verify()
    chains = []

    def chain(points):
        verts = []
        for p, r in points:
            v = bm.verts.new(p)
            v[rad].radius = (r, r)
            verts.append(v)
        for a, b in zip(verts, verts[1:]):
            bm.edges.new((a, b))
        chains.append(verts)
        return verts

    torso = chain([(Vector((0, 0, 0.88)), 0.13), (Vector((0, 0, 1.1)), 0.12), (Vector((0, 0, 1.3)), 0.13),
                   (Vector((0, 0, 1.45)), 0.06), (Vector((0, -0.01, 1.62)), 0.09), (Vector((0, 0, 1.8)), 0.02)])
    for sx in (1, -1):
        m = Vector((sx, 1, 1))

        def mir(v):
            return Vector((v.x * m.x, v.y, v.z))
        arm = chain([(mir(Vector((0.08, 0, 1.4))), 0.05), (mir(SHOULDER), 0.05),
                     (mir(SHOULDER.lerp(ELBOW, 0.5)), 0.045), (mir(ELBOW), 0.035),
                     (mir(ELBOW.lerp(WRIST, 0.4)), 0.04), (mir(WRIST), 0.025),
                     (mir(WRIST + (WRIST - ELBOW).normalized() * 0.1), 0.03)])
        bm.edges.new((torso[2], arm[0]))
        leg = chain([(mir(HIP), 0.08), (mir(HIP.lerp(KNEE, 0.5)), 0.07), (mir(KNEE), 0.045),
                     (mir(KNEE.lerp(ANKLE, 0.35)), 0.055), (mir(ANKLE), 0.035), (mir(TOE), 0.03)])
        bm.edges.new((torso[0], leg[0]))
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("body", me)
    ctx.scene.collection.objects.link(obj)
    skin = obj.modifiers.new("skin", 'SKIN')
    obj.modifiers.new("sub", 'SUBSURF').levels = 1
    obj.data.skin_vertices[0].data[0].use_root = True
    ctx.view_layer.objects.active = obj
    with ctx.temp_override(object=obj, active_object=obj):
        bpy.ops.object.modifier_apply(modifier=skin.name)
        bpy.ops.object.modifier_apply(modifier="sub")
    return obj


body = make_body()
z = [(body.matrix_world @ v.co).z for v in body.data.vertices]
check(max(z) > 1.7 and min(z) < 0.06, f"synthetic body built ({min(z):.2f}..{max(z):.2f} m)")

for key, point in (("chin", (0, -0.03, 1.55)), ("neck", (0, 0, 1.46)), ("shoulder", SHOULDER),
                   ("wrist", WRIST), ("pelvis", (0, 0, 0.95)), ("ankle", ANKLE)):
    markers.set_marker(ctx, key, Vector(point))


def fit(**kw):
    for o in list(bpy.data.objects):
        if o.type == 'ARMATURE':
            bpy.data.objects.remove(o)
    utils.deselect_all(ctx)
    body.select_set(True)
    ctx.view_layer.objects.active = body
    res = bpy.ops.rigger.fit_markers(fingers=False, face=False, **kw)
    ref = ctx.object
    bpy.ops.object.mode_set(mode='OBJECT')
    return res, ref


res, ref = fit()
bones = ref.data.bones
check(res == {'FINISHED'}, "fit with joints from the mesh")
elbow = bones["forearm.L"].head_local
knee = bones["shin.L"].head_local
check((elbow - ELBOW).length < 0.03, f"elbow found at the bend ({(elbow - ELBOW).length * 100:.1f} cm off)")
check((knee - KNEE).length < 0.04, f"knee found at the narrowing ({(knee - KNEE).length * 100:.1f} cm off)")
check(abs(bones["thigh.L"].head_local.x - HIP.x) < 0.03,
      f"hip joint on the thigh line (x {bones['thigh.L'].head_local.x:.3f}, expected {HIP.x})")
check((bones["forearm.R"].head_local - Vector((-elbow.x, elbow.y, elbow.z))).length < 1e-4, "right side mirrored")
toe = bones["toe.L"].tail_local
check(abs(toe.y - TOE.y) < 0.04, f"toe tip from the foot length (y {toe.y:.3f}, mesh {TOE.y - 0.03:.3f})")

res, ref = fit(fit_joints=False)
elbow_template = ref.data.bones["forearm.L"].head_local
check(res == {'FINISHED'} and (elbow_template - ELBOW).length > 0.05,
      f"without fitting the elbow uses template proportions ({(elbow_template - ELBOW).length * 100:.1f} cm off)")

# A mesh without limbs: fitting falls back to proportions instead of failing
bpy.data.objects.remove(body)
bpy.ops.mesh.primitive_cube_add(size=1.8, location=(0, 0, 0.9))
body = ctx.object
res, ref = fit()
check(res == {'FINISHED'} and "forearm.L" in ref.data.bones, "a shapeless mesh still gets a reference")

print("FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
