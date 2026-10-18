# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport picker: a clickable body diagram drawn in the corner of the 3D view.

The layout is built from the rig's metadata, so only the parts the rig has
appear. Click selects a part's controls, Shift adds them, Ctrl removes them.
Buttons light up when their controls are selected; hovering shows the part's
name. Drawing uses the gpu module; nothing is added to the rig.

Layout units: x from -0.5 to 0.5 (the character faces the viewer, so its left
side is on the right), y from 0 (feet) to HEIGHT (top of the head).
"""

import json

import bpy

from . import utils

HEIGHT = 1.62
BASE_SIZE = 210        # picker width in pixels at size 1.0 and UI scale 1.0
MARGIN = 24

SIDE_COLORS = {"L": (0.10, 0.42, 1.00), "R": (1.00, 0.16, 0.16), "C": (1.00, 0.72, 0.02),
               "ROOT": (0.15, 0.85, 0.30), "FACE": (0.95, 0.45, 0.85)}

_state = {"running": False, "handle": None, "hover": None, "cache": (None, None, [])}


class Button:
    __slots__ = ("key", "label", "bones", "rect", "side", "round")

    def __init__(self, key, label, bones, center, size, side, round_=False):
        self.key, self.label, self.bones, self.side, self.round = key, label, bones, side, round_
        (cx, cy), (w, h) = center, size
        self.rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def contains(self, x, y):
        x0, y0, x1, y1 = self.rect
        return x0 <= x <= x1 and y0 <= y <= y1


def build_buttons(rig):
    """Picker buttons for this rig, from its metadata (existing controls only)."""
    meta = utils.get_meta(rig)
    bones = rig.pose.bones
    out = []

    def add(key, label, names, center, size, side, round_=False):
        names = [n for n in names if n and n in bones]
        if names:
            out.append(Button(key, label, names, center, size, side, round_))

    neck = meta.get("neck") or {}
    spine = meta.get("spine") or {}
    faces = meta.get("faces") or []
    if faces:
        add("face", "Face", faces[0].get("controls", []), (0.0, 1.565), (0.13, 0.05), "FACE")
    add("head", "Head", [neck.get("head")], (0.0, 1.43), (0.15, 0.15), "C")
    add("neck", "Neck", neck.get("fk", []), (0.0, 1.31), (0.07, 0.06), "C")
    add("chest", "Chest", [spine.get("chest")], (0.0, 1.17), (0.22, 0.09), "C")
    add("spine", "Spine FK", spine.get("fk", []), (0.0, 1.04), (0.16, 0.08), "C")
    add("hips", "Hips", [spine.get("hips")], (0.0, 0.91), (0.22, 0.07), "C")
    add("torso", "Torso", [spine.get("torso")], (0.0, 0.82), (0.30, 0.05), "C")
    add("root", "Root", ["root"], (0.0, 0.025), (0.62, 0.04), "ROOT")
    add("all", "All Controls", meta.get("controls", []), (0.43, 0.025), (0.12, 0.04), "ROOT")

    limbs = {(l["type"], l["side"]): l for l in meta.get("limbs", [])}
    controls = meta.get("controls", [])
    for side, s in (("L", 1.0), ("R", -1.0)):
        arm = limbs.get(("arm", side))
        if arm:
            fk = list(arm["fk"]) + [None, None, None]
            add(f"shoulder.{side}", f"Shoulder {side}", [arm.get("shoulder")], (s * 0.17, 1.24), (0.11, 0.05), side)
            add(f"upper_arm.{side}", f"Upper Arm FK {side}", [fk[0]], (s * 0.265, 1.09), (0.07, 0.15), side)
            add(f"forearm.{side}", f"Forearm FK {side}", [fk[1]], (s * 0.325, 0.92), (0.065, 0.15), side)
            add(f"hand.{side}", f"Hand FK {side}", [fk[2]], (s * 0.375, 0.785), (0.075, 0.07), side)
            add(f"hand_ik.{side}", f"Hand IK {side}", [arm["ik_ctrl"], arm.get("ik_offset"), arm.get("ik_pivot")],
                (s * 0.40, 0.69), (0.10, 0.05), side)
            add(f"arm_pole.{side}", f"Elbow Pole {side}", [arm["pole"]], (s * 0.44, 1.01), (0.045, 0.045), side, True)
            fingers = [n for n in controls if n.endswith("." + side) and
                       any(f in n for f in ("thumb", "index", "middle", "ring", "pinky"))]
            add(f"fingers.{side}", f"Fingers {side}", fingers, (s * 0.42, 0.60), (0.10, 0.06), side)
        leg = limbs.get(("leg", side))
        if leg:
            fk = list(leg["fk"]) + [None, None, None]
            add(f"thigh.{side}", f"Thigh FK {side}", [fk[0]], (s * 0.085, 0.66), (0.09, 0.17), side)
            add(f"shin.{side}", f"Shin FK {side}", [fk[1]], (s * 0.09, 0.45), (0.08, 0.17), side)
            add(f"foot.{side}", f"Foot FK {side}", [fk[2]], (s * 0.10, 0.295), (0.09, 0.06), side)
            add(f"foot_ik.{side}", f"Foot IK {side}",
                [leg["ik_ctrl"], leg.get("foot_roll"), leg.get("ik_offset"), leg.get("ik_pivot")],
                (s * 0.11, 0.19), (0.13, 0.06), side)
            add(f"toe.{side}", f"Toes {side}", [leg.get("toe_fk"), leg.get("toe_ik")], (s * 0.11, 0.10), (0.09, 0.04), side)
            add(f"leg_pole.{side}", f"Knee Pole {side}", [leg["pole"]], (s * 0.20, 0.49), (0.045, 0.045), side, True)
    return out


def buttons_for(rig):
    """Cached build_buttons: rebuilt when the rig or its metadata changes."""
    key = (rig.name, rig.get(utils.META_KEY, ""))
    if _state["cache"][0] != key[0] or _state["cache"][1] != key[1]:
        _state["cache"] = (key[0], key[1], build_buttons(rig))
    return _state["cache"][2]


def picker_size(context):
    prefs = context.preferences.system
    return BASE_SIZE * prefs.ui_scale * context.scene.rigger_picker_size


def to_layout(mx, my, size):
    """Region pixel -> layout units (picker anchored at the bottom left)."""
    return (mx - MARGIN) / size - 0.5, (my - MARGIN) / size


def inside_panel(mx, my, size):
    x, y = to_layout(mx, my, size)
    return -0.5 <= x <= 0.5 and 0.0 <= y <= HEIGHT


def button_at(buttons, mx, my, size):
    x, y = to_layout(mx, my, size)
    return next((b for b in buttons if b.contains(x, y)), None)


def visible(pose_bone):
    bone = pose_bone.bone
    if bone.hide:
        return False
    collections = list(bone.collections)
    return not collections or any(getattr(c, "is_visible_effectively", c.is_visible) for c in collections)


def select(rig, names, mode='SET'):
    """mode: SET replaces the selection, ADD extends it, REMOVE deselects the names."""
    pbs = rig.pose.bones
    if mode == 'SET':
        for pb in pbs:
            utils.select_pose_bone(pb, False)
    chosen = [pbs[n] for n in names if n in pbs and visible(pbs[n])]
    for pb in chosen:
        utils.select_pose_bone(pb, mode != 'REMOVE')
    if chosen and mode != 'REMOVE':
        rig.data.bones.active = chosen[0].bone
    return len(chosen)


def is_lit(rig, button):
    pbs = rig.pose.bones
    return any(utils.is_pose_bone_selected(pbs[n]) for n in button.bones if n in pbs)


def _active_rig(context):
    obj = context.view_layer.objects.active if context.view_layer else None
    return obj if utils.is_rig(obj) and obj.mode == 'POSE' else None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _rect_tris(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]


def _circle(cx, cy, r, n=16):
    import math
    pts = [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return pts


def draw_callback():
    context = bpy.context
    rig = _active_rig(context)
    if not _state["running"] or rig is None:
        return
    import blf
    import gpu
    from gpu_extras.batch import batch_for_shader
    size = picker_size(context)
    ox, oy = MARGIN, MARGIN

    def px(x, y):
        return ox + (x + 0.5) * size, oy + y * size

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')

    def fill(points, color):
        batch = batch_for_shader(shader, 'TRIS', {"pos": points})
        shader.uniform_float("color", color)
        batch.draw(shader)

    def outline(points, color):
        batch = batch_for_shader(shader, 'LINE_LOOP', {"pos": points})
        shader.uniform_float("color", color)
        batch.draw(shader)

    x0, y0 = px(-0.5, 0.0)
    x1, y1 = px(0.5, HEIGHT)
    fill(_rect_tris(x0, y0, x1, y1), (0.08, 0.08, 0.09, 0.45))
    hover = _state["hover"]
    gpu.state.line_width_set(1.5)
    for b in buttons_for(rig):
        bx0, by0 = px(b.rect[0], b.rect[1])
        bx1, by1 = px(b.rect[2], b.rect[3])
        base = SIDE_COLORS.get(b.side, SIDE_COLORS["C"])
        lit = is_lit(rig, b)
        shown = any(visible(rig.pose.bones[n]) for n in b.bones if n in rig.pose.bones)
        # faded: every control of the part is hidden (e.g. the IK hand while the arm is in FK)
        color = (*base, 0.95 if lit else 0.45 if shown else 0.12)
        if b.round:
            cx, cy, r = (bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) / 2
            ring = _circle(cx, cy, r)
            tris = [p for i in range(len(ring)) for p in ((cx, cy), ring[i], ring[(i + 1) % len(ring)])]
            fill(tris, color)
            edge = ring
        else:
            fill(_rect_tris(bx0, by0, bx1, by1), color)
            edge = [(bx0, by0), (bx1, by0), (bx1, by1), (bx0, by1)]
        if lit or (hover == b.key):
            outline(edge, (1.0, 1.0, 1.0, 1.0 if lit else 0.7))
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')
    label = next((b.label for b in buttons_for(rig) if b.key == hover), "Picker")
    font = 0
    blf.size(font, 11 * context.preferences.system.ui_scale)
    blf.color(font, 1.0, 1.0, 1.0, 0.9)
    blf.position(font, x0 + 6, y1 - 16 * context.preferences.system.ui_scale, 0)
    blf.draw(font, label)


def _redraw(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def start(context):
    if _state["running"]:
        return
    _state["running"] = True
    if _state["handle"] is None:
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(draw_callback, (), 'WINDOW', 'POST_PIXEL')
    _redraw(context)


def stop(context=None):
    _state["running"] = False
    _state["hover"] = None
    if _state["handle"] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_state["handle"], 'WINDOW')
        _state["handle"] = None
    if context is not None:
        _redraw(context)


def is_running():
    return _state["running"]


def _region_under_mouse(context, event):
    for area in context.window.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for region in area.regions:
            if region.type == 'WINDOW' and region.x <= event.mouse_x < region.x + region.width \
                    and region.y <= event.mouse_y < region.y + region.height:
                return area, region
    return None, None


class RIGGER_OT_viewport_picker(bpy.types.Operator):
    """Show or hide the clickable body picker in the corner of the 3D view (Pose Mode).
Click: select, Shift: add, Ctrl: remove"""
    bl_idname = "rigger.viewport_picker"
    bl_label = "Viewport Picker"

    def invoke(self, context, event):
        if _state["running"]:
            stop(context)
            return {'FINISHED'}
        start(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        if _state["running"]:
            stop(context)
        else:
            start(context)
        return {'FINISHED'}

    def modal(self, context, event):
        if not _state["running"]:
            return {'CANCELLED'}
        rig = _active_rig(context)
        if rig is None or event.type not in {'MOUSEMOVE', 'LEFTMOUSE'}:
            return {'PASS_THROUGH'}
        area, region = _region_under_mouse(context, event)
        if region is None:
            return {'PASS_THROUGH'}
        mx, my = event.mouse_x - region.x, event.mouse_y - region.y
        size = picker_size(context)
        button = button_at(buttons_for(rig), mx, my, size)
        key = button.key if button else None
        if key != _state["hover"]:
            _state["hover"] = key
            area.tag_redraw()
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and inside_panel(mx, my, size):
            if button:
                mode = 'ADD' if event.shift else 'REMOVE' if event.ctrl else 'SET'
                select(rig, button.bones, mode)
                bpy.ops.ed.undo_push(message=f"Picker: {button.label}")
                _redraw(context)
            return {'RUNNING_MODAL'}  # clicks on the picker never reach the viewport
        return {'PASS_THROUGH'}


classes = (RIGGER_OT_viewport_picker,)
