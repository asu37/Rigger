# SPDX-License-Identifier: GPL-3.0-or-later
"""Smart markers: click a few landmarks on the character, then fit a reference.

Markers are placed in the middle of the mesh volume under the mouse: the view
ray is cast through the mesh and the entry/exit points are averaged.
"""

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from . import templates, utils

MARKER_PREFIX = "RGM-"


def marker_name(key):
    return MARKER_PREFIX + key


def marker_objects():
    return {key: bpy.data.objects.get(marker_name(key)) for key, *_ in templates.MARKERS}


def _marker_collection(context):
    coll = bpy.data.collections.get("Rigger Markers")
    if coll is None:
        coll = bpy.data.collections.new("Rigger Markers")
        context.scene.collection.children.link(coll)
    return coll


def _volume_center(context, depsgraph, meshes, origin, direction):
    """Average of the first entry and exit points of the ray through the meshes."""
    hits = []
    for obj in meshes:
        inv = obj.matrix_world.inverted()
        o, d = inv @ origin, (inv.to_3x3() @ direction).normalized()
        evaluated = obj.evaluated_get(depsgraph)
        ok, loc, _normal, _idx = evaluated.ray_cast(o, d)
        if not ok:
            continue
        entry = obj.matrix_world @ loc
        exit_point = entry
        ok2, loc2, _n2, _i2 = evaluated.ray_cast(loc + d * 1e-4, d)
        if ok2:
            exit_point = obj.matrix_world @ loc2
        hits.append(((entry - origin).length, entry, exit_point))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    return (hits[0][1] + hits[0][2]) * 0.5


class RIGGER_OT_place_marker(bpy.types.Operator):
    """Click on the character (front view) to place this marker"""
    bl_idname = "rigger.place_marker"
    bl_label = "Place Marker"
    bl_options = {'REGISTER', 'UNDO'}

    marker: EnumProperty(items=[(k, label, desc) for k, label, desc, _sym in templates.MARKERS])

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        self.meshes = [o for o in context.selected_objects if o.type == 'MESH']
        if not self.meshes:
            self.meshes = [o for o in context.visible_objects if o.type == 'MESH']
        if not self.meshes:
            self.report({'ERROR'}, "No mesh to click on")
            return {'CANCELLED'}
        label = dict((k, (l, d)) for k, l, d, _s in templates.MARKERS)[self.marker]
        context.workspace.status_text_set(f"{label[1]}  |  LMB: place   RMB/Esc: cancel")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.workspace.status_text_set(None)
            return {'CANCELLED'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            region, rv3d = context.region, context.region_data
            coord = (event.mouse_region_x, event.mouse_region_y)
            origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
            direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
            point = _volume_center(context, context.evaluated_depsgraph_get(), self.meshes,
                                   origin, direction)
            if point is None:
                self.report({'WARNING'}, "Click on the mesh")
                return {'RUNNING_MODAL'}
            set_marker(context, self.marker, point)
            context.workspace.status_text_set(None)
            return {'FINISHED'}
        return {'PASS_THROUGH'} if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} \
            else {'RUNNING_MODAL'}


def set_marker(context, key, point):
    symmetric = {k: s for k, _l, _d, s in templates.MARKERS}[key]
    point = Vector(point)
    if symmetric:
        point.x = abs(point.x)
    else:
        point.x = 0.0
    obj = bpy.data.objects.get(marker_name(key))
    if obj is None:
        obj = bpy.data.objects.new(marker_name(key), None)
        obj.empty_display_type = 'SPHERE'
        obj.show_in_front = True
        obj.show_name = True
        _marker_collection(context).objects.link(obj)
    obj.location = point
    return obj


def surface_point(depsgraph, meshes, ref_matrix, x, z, start_y):
    """Local Y where a ray going backwards (+Y) from the front first hits the meshes."""
    origin = ref_matrix @ Vector((x, start_y, z))
    direction = (ref_matrix.to_3x3() @ Vector((0, 1, 0))).normalized()
    best = None
    for obj in meshes:
        inv = obj.matrix_world.inverted()
        o, d = inv @ origin, (inv.to_3x3() @ direction).normalized()
        ok, loc, _n, _i = obj.evaluated_get(depsgraph).ray_cast(o, d)
        if ok:
            world = obj.matrix_world @ loc
            dist = (world - origin).length
            if best is None or dist < best[0]:
                best = (dist, world)
    if best is None:
        return None
    return (ref_matrix.inverted() @ best[1]).y


def find_eyeball(context, meshes, to_local, points):
    """Centre of the left eyeball, from separate eye objects or small loose mesh parts."""
    from .skin import loose_parts, world_vertices
    import numpy as np
    h = points["top"] - points["floor"]
    chin = points["chin"]
    expect = Vector((0.018 * h, chin.y, chin.z + (points["top"] - chin.z) * 0.47))
    candidates = []
    for obj in context.view_layer.objects:
        if obj.type != 'MESH' or not obj.visible_get() or obj.name.startswith("WGT-"):
            continue
        corners = [to_local @ (obj.matrix_world @ Vector(c)) for c in obj.bound_box]
        size = (Vector(map(max, *corners)) - Vector(map(min, *corners))).length
        center = sum(corners, Vector()) / 8
        if obj not in meshes and size < 0.06 * h:
            candidates.append(center)
        elif obj in meshes and len(obj.data.vertices) < 300000:
            labels = loose_parts(obj)
            pts = world_vertices(obj)
            for label in np.unique(labels):
                part = pts[labels == label]
                if len(part) < 8 or len(part) > 0.2 * len(pts):
                    continue
                lo, hi = part.min(axis=0), part.max(axis=0)
                if np.linalg.norm(hi - lo) < 0.06 * h:
                    candidates.append(to_local @ Vector(((lo + hi) / 2).tolist()))
    good = [c for c in candidates if c.x > 0.004 * h and chin.z < c.z < points["top"]
            and (c - expect).length < 0.06 * h]
    if not good:
        return None
    return min(good, key=lambda c: (c - expect).length)


def measure_hand(depsgraph, meshes, ref_matrix, points):
    """Hand size relative to the template, from a ray shot back along the arm at the fingertips."""
    h = points["top"] - points["floor"]
    wrist, shoulder = points["wrist"], points["shoulder"]
    direction = (wrist - shoulder).normalized()
    origin = ref_matrix @ (wrist + direction * 0.35 * h)
    back = (ref_matrix.to_3x3() @ -direction).normalized()
    best = None
    for obj in meshes:
        inv = obj.matrix_world.inverted()
        ok, loc, _n, _i = obj.evaluated_get(depsgraph).ray_cast(inv @ origin, (inv.to_3x3() @ back).normalized())
        if ok:
            world = obj.matrix_world @ loc
            if best is None or (world - origin).length < (best - origin).length:
                best = world
    if best is None:
        return None
    reach = ((ref_matrix.inverted() @ best) - wrist).dot(direction)
    default = (0.056 + 0.049) * h  # middle finger base + length in the template
    if reach <= 0.02 * h:
        return None
    return max(0.5, min(1.6, reach / default))


def _hand_points(depsgraph, meshes, to_local, wrist, direction, reach, radius):
    """Points sampled along the edges of the mesh around the hand, in reference space."""
    pts = []
    for obj in meshes:
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        mw = to_local @ obj.matrix_world
        co = np.array([mw @ v.co for v in mesh.vertices]) if len(mesh.vertices) else np.zeros((0, 3))
        ev.to_mesh_clear()
        if not len(co):
            continue
        rel = co - np.array(wrist)
        along = rel @ np.array(direction)
        perp = np.linalg.norm(rel - np.outer(along, direction), axis=1)
        keep = (along > -0.05 * reach) & (along < reach) & (perp < radius)
        if not keep.any():
            continue
        mesh = obj.data
        edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
        if len(mesh.vertices) == len(co):
            edges = edges[keep[edges[:, 0]] & keep[edges[:, 1]]]
            # Dense samples along the edges so every slice crosses the whole finger outline
            t = np.linspace(0.0, 1.0, 16)[None, :, None]
            a, b = co[edges[:, 0]][:, None, :], co[edges[:, 1]][:, None, :]
            pts.append((a * (1 - t) + b * t).reshape(-1, 3))
        pts.append(co[keep])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def detect_fingers(context, ref, meshes, side="L", slices=48, gap=None):
    """Place the finger reference bones of one hand on the fingers of the mesh.

    The hand region is cut in slices across the arm direction; in each slice the
    points split into clusters (one per finger); clusters are chained from slice
    to slice, and each chain becomes a finger from its base to its tip.
    Returns the number of fingers placed.
    """
    bones = ref.data.bones
    hand = bones.get(f"hand.{side}")
    if hand is None:
        return 0
    fingers = [b for b in hand.children if ref.pose.bones[b.name].rigger.module == 'FINGER']
    if not fingers:
        return 0
    zs = [v.z for b in bones for v in (b.head_local, b.tail_local)]
    h = max(zs) - min(zs)
    wrist = hand.head_local.copy()
    direction = (hand.tail_local - hand.head_local).normalized()
    up = hand.matrix_local.to_3x3().col[2].normalized()
    up = (up - direction * direction.dot(up)).normalized()
    spread = direction.cross(up)
    reach, radius = 0.16 * h, 0.07 * h
    depsgraph = context.evaluated_depsgraph_get()
    pts = _hand_points(depsgraph, meshes, ref.matrix_world.inverted(), wrist, direction, reach, radius)
    if len(pts) < 50:
        return 0
    rel = pts - np.array(wrist)
    along = rel @ np.array(direction)
    sp = rel @ np.array(spread)
    upc = rel @ np.array(up)
    gap = gap or 0.003 * h
    finger_max = 0.013 * h
    start = float(np.percentile(along, 2))
    end = float(along.max())
    width = (end - start) / slices
    chains = []    # each: list of (along, spread, up)
    open_chains = []
    for i in range(slices):
        lo = start + i * width
        mask = (along >= lo) & (along < lo + width)
        if mask.sum() < 3:
            continue
        order = np.argsort(sp[mask])
        s_sorted, u_sorted = sp[mask][order], upc[mask][order]
        cuts = np.nonzero(np.diff(s_sorted) > gap)[0] + 1
        # A finger cross-section is small both ways; palm pieces are thicker
        clusters = [(float(seg.mean()), float(us.mean()))
                    for seg, us in zip(np.split(s_sorted, cuts), np.split(u_sorted, cuts))
                    if len(seg) >= 2 and np.ptp(seg) < finger_max and np.ptp(us) < finger_max]
        a = lo + width * 0.5
        if len(clusters) < 2:
            open_chains = []  # palm: fingers not separated yet
            continue
        new_open = []
        for s_c, u_c in clusters:
            best = min(open_chains, key=lambda c: abs(c[-1][1] - s_c), default=None)
            if best is not None and abs(best[-1][1] - s_c) < 3 * gap and all(best is not c for c in new_open):
                best.append((a, s_c, u_c))
                new_open.append(best)
            else:
                chain = [(a, s_c, u_c)]
                chains.append(chain)
                new_open.append(chain)
        open_chains = new_open
    # Join pieces of one finger (same spread position), then drop short bits (palm edges)
    merged = []
    for chain in sorted(chains, key=len, reverse=True):
        mean = np.mean([p[1] for p in chain])
        target = next((m for m in merged if abs(np.mean([p[1] for p in m]) - mean) < 1.5 * gap), None)
        if target is not None:
            target.extend(chain)
            target.sort(key=lambda p: p[0])
        else:
            merged.append(sorted(chain, key=lambda p: p[0]))
    extent = lambda c: c[-1][0] - c[0][0]  # noqa: E731
    longest = max((extent(c) for c in merged), default=0.0)
    chains = [c for c in merged if len(c) >= 3 and extent(c) >= 0.25 * longest]
    if not chains:
        return 0
    names = [f for f in ("thumb", "index", "middle", "ring", "pinky")
             if any(b.name.startswith(f) for b in fingers)]
    if len(chains) > len(names):
        chains = sorted(chains, key=len, reverse=True)[:len(names)]
    if len(chains) < len(names) and "thumb" in names:
        names.remove("thumb")  # a thumb lying against the hand does not separate
    chains.sort(key=lambda c: -np.mean([p[1] for p in c]))  # thumb / index side first
    by_name = dict(zip(names, chains))
    utils.set_mode(ref, 'EDIT')
    ebones = ref.data.edit_bones
    placed = 0
    to_vec = lambda p: wrist + direction * p[0] + spread * p[1] + up * p[2]  # noqa: E731
    for name, chain in by_name.items():
        seg_names = [f"{name}.{i:02d}.{side}" for i in range(1, 4)]
        if not all(n in ebones for n in seg_names):
            continue
        base, tip = to_vec(chain[0]), to_vec(chain[-1]) + direction * width * 0.5
        length = (tip - base).length
        base = base - (tip - base).normalized() * length * 0.25  # fingers separate past the knuckle
        points = [base]
        total = 0.0
        for frac in (0.40, 0.33, 0.27):
            total += frac
            target = base.lerp(tip, total)
            # Follow the chain's curve: take the centroid closest to this point along the arm
            a = (target - wrist).dot(direction)
            near = min(chain, key=lambda p: abs(p[0] - a))
            points.append(to_vec((a, near[1], near[2])) if total < 0.999 else tip)
        for i, n in enumerate(seg_names):
            eb = ebones[n]
            eb.head, eb.tail = points[i], points[i + 1]
            eb.align_roll(up)
        placed += 1
    bpy.ops.object.mode_set(mode='OBJECT')
    return placed


class RIGGER_OT_detect_fingers(bpy.types.Operator):
    """Find the fingers on the selected mesh and place the finger bones of the active reference on them
    (works best with fingers spread apart and the palm facing down)"""
    bl_idname = "rigger.detect_fingers"
    bl_label = "Detect Fingers"
    bl_options = {'REGISTER', 'UNDO'}

    mirror: BoolProperty(name="Mirror", default=True, description="Copy the left hand to the right")
    thickness: bpy.props.FloatProperty(name="Finger Gap", default=0.3, min=0.05, max=3.0,
                                       description="Smallest gap between two fingers, in percent of the height")

    @classmethod
    def poll(cls, context):
        return any(utils.is_reference(o) for o in context.selected_objects + [context.object])

    def execute(self, context):
        ref = context.object if utils.is_reference(context.object) else next(
            o for o in context.selected_objects if utils.is_reference(o))
        meshes = [o for o in context.selected_objects if o.type == 'MESH'] or \
                 [o for o in context.visible_objects if o.type == 'MESH']
        zs = [v.z for b in ref.data.bones for v in (b.head_local, b.tail_local)]
        count = detect_fingers(context, ref, meshes, "L", gap=(max(zs) - min(zs)) * self.thickness / 100.0)
        if not count:
            self.report({'ERROR'}, "No fingers found. Spread the fingers apart or place them by hand")
            return {'CANCELLED'}
        if self.mirror:
            utils.set_mode(ref, 'EDIT')
            bpy.ops.rigger.mirror_reference()
            bpy.ops.object.mode_set(mode='OBJECT')
        self.report({'INFO'}, f"Placed {count} fingers")
        return {'FINISHED'}


def fit_joints(depsgraph, meshes, points, ref_matrix, spine_count):
    """Elbow, knee, hip joint, spine depth and foot length measured on the mesh
    (see fitting.py), in the reference's local space. Missing keys keep the
    template proportions."""
    from . import fitting
    import numpy as np
    sl = fitting.Slicer(meshes, depsgraph)
    if not len(sl.p):
        return {}
    to_world = np.array(ref_matrix)
    to_local = np.array(ref_matrix.inverted())

    def world(v):
        return to_world[:3, :3] @ np.array(v) + to_world[:3, 3]

    def local(v):
        return Vector((to_local[:3, :3] @ np.asarray(v) + to_local[:3, 3]).tolist())

    h = points["top"] - points["floor"]
    hw = h * ref_matrix.to_scale().z
    out = {}
    shoulder, wrist, ankle, pelvis = (points[k] for k in ("shoulder", "wrist", "ankle", "pelvis"))
    e = fitting.elbow(sl, world(shoulder), world(wrist), hw)
    if e is not None:
        out["elbow"] = local(e)
    hip_x = max(abs(ankle.x), abs(shoulder.x) * 0.45)
    hip_guess = Vector((hip_x, pelvis.y, pelvis.z - 0.02 * h))
    ankle_pt = Vector((abs(ankle.x), ankle.y, ankle.z))
    knee, hip = fitting.leg(sl, world(ankle_pt), world(hip_guess), hw)
    if knee is not None:
        out["knee"] = local(knee)
    if hip is not None:
        out["thigh_head"] = local(hip)
    # Spine joints between the hips and the neck base
    neck = points["neck"]
    hips_tail_z = pelvis.z + 0.055 * h
    depths = {}
    for i in range(1, max(1, spine_count)):
        w = world(Vector((0.0, 0.0, hips_tail_z + (neck.z - hips_tail_z) * i / spine_count)))
        y = fitting.spine_depth(sl, w[2], hw)
        if y is not None:
            depths[i] = local((w[0], y, w[2])).y
    if depths:
        out["spine_y"] = depths
    tip = fitting.foot_tip(sl, world(ankle_pt), world(Vector((0, 0, points["floor"])))[2], hw)
    if tip is not None:
        out["toe_tip"] = local(tip)
    return out


def meshes_bounds(meshes):
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    if not pts:
        return None
    return (min(p.z for p in pts), max(p.z for p in pts))


class RIGGER_OT_fit_markers(bpy.types.Operator):
    """Create (or refit) a human reference from the placed markers"""
    bl_idname = "rigger.fit_markers"
    bl_label = "Fit Reference to Markers"
    bl_options = {'REGISTER', 'UNDO'}

    fingers: BoolProperty(name="Fingers", default=True)
    face: BoolProperty(name="Face (jaw, eyes)", default=True)
    fit_joints: BoolProperty(
        name="Fit Joints to Mesh", default=True,
        description="Place the elbows, knees, hips, spine and toes from the mesh shape "
                    "instead of standard proportions")

    def execute(self, context):
        markers = marker_objects()
        missing = [k for k, o in markers.items() if o is None]
        if missing:
            self.report({'ERROR'}, "Place these markers first: " + ", ".join(missing))
            return {'CANCELLED'}
        meshes = [o for o in context.selected_objects if o.type == 'MESH'] or \
                 [o for o in context.visible_objects if o.type == 'MESH']
        bounds = meshes_bounds(meshes)
        if bounds is None:
            self.report({'ERROR'}, "Select the character mesh")
            return {'CANCELLED'}
        ref = context.object if utils.is_reference(context.object) else None
        # Bones are built in the reference's local space.
        to_local = ref.matrix_world.inverted() if ref else Matrix()
        points = {k: to_local @ o.matrix_world.translation for k, o in markers.items()}
        floor, top = bounds
        points["floor"] = (to_local @ Vector((0, 0, floor))).z
        points["top"] = (to_local @ Vector((0, 0, top))).z
        # Depth of the face surface (markers sit in the middle of the head)
        h = points["top"] - points["floor"]
        chin = points["chin"]
        mouth_z = chin.z + 0.026 * h
        eye_z = chin.z + (points["top"] - chin.z) * 0.47
        depsgraph = context.evaluated_depsgraph_get()
        for key, x, z in (("face_front", 0.0, mouth_z), ("eye_front", 0.018 * h, eye_z)):
            hit = surface_point(depsgraph, meshes, ref.matrix_world if ref else Matrix(), x, z,
                                chin.y - h)
            if hit is not None and abs(hit - chin.y) < 0.1 * h:
                points[key] = hit
        inv = (ref.matrix_world if ref else Matrix()).inverted()
        eye = find_eyeball(context, meshes, inv, points)
        if eye is not None:
            points["eye"] = eye
        scale = measure_hand(depsgraph, meshes, ref.matrix_world if ref else Matrix(), points)
        if scale is not None:
            points["hand_scale"] = scale
        options = templates.scene_options()
        if self.fit_joints:
            try:
                points.update(fit_joints(depsgraph, meshes, points, ref.matrix_world if ref else Matrix(),
                                         int((options or templates.DEFAULT_OPTIONS)["spine_count"])))
            except Exception as exc:  # a measuring failure must never block fitting
                import traceback
                traceback.print_exc()
                self.report({'WARNING'}, f"Could not fit joints to the mesh ({exc}); using standard proportions")
        specs = templates.human_specs(points, fingers=self.fingers, face=self.face, options=options)

        if ref is None:
            ref = templates.create_reference(context, specs, name="reference")
        else:
            utils.set_mode(ref, 'EDIT')
            for eb in list(ref.data.edit_bones):
                ref.data.edit_bones.remove(eb)
            templates.build_bones(ref, templates.mirrored(specs))
            bpy.ops.object.mode_set(mode='OBJECT')
            for spec in templates.mirrored(specs):
                templates.apply_settings(ref.pose.bones[spec.name], spec.module, spec.params)
        for obj in markers.values():
            obj.hide_set(True)
        utils.set_mode(ref, 'EDIT')
        self.report({'INFO'}, "Reference fitted. Adjust bones if needed, then Generate Rig")
        return {'FINISHED'}


class RIGGER_OT_clear_markers(bpy.types.Operator):
    """Delete all placed markers"""
    bl_idname = "rigger.clear_markers"
    bl_label = "Clear Markers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for obj in marker_objects().values():
            if obj is not None:
                bpy.data.objects.remove(obj)
        return {'FINISHED'}


classes = (RIGGER_OT_place_marker, RIGGER_OT_fit_markers, RIGGER_OT_clear_markers, RIGGER_OT_detect_fingers)
