"""Interactive GUI test: drives the real Blender UI on an X display.

Usage (headless, with Xvfb, xdotool and ImageMagick installed):

    Xvfb :99 -screen 0 1600x1000x24 &
    DISPLAY=:99 RIGGER_SHOTS=/tmp/shots blender --factory-startup --python tests/gui_test.py

The script loads Rigger from this repository, builds a character mesh, places
the smart markers with real mouse clicks (xdotool), runs the workflow through
the operators and the sidebar, and saves a screenshot after every step in
$RIGGER_SHOTS. A summary of checks is written to $RIGGER_SHOTS/report.txt.
"""
import math
import os
import subprocess
import sys
import traceback

import bpy
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO)

import rigger  # noqa: E402
from rigger import retarget, templates, ui, utils  # noqa: E402

# Every panel opens by default and reports draw errors raised in the real UI
PANELS = [c for c in ui.classes if issubclass(c, bpy.types.Panel)]
DRAWN, DRAW_ERRORS = {}, []


def _wrap(cls):
    original = cls.draw

    def draw(self, context):
        DRAWN[cls.__name__] = DRAWN.get(cls.__name__, 0) + 1
        try:
            original(self, context)
        except Exception:  # noqa: BLE001
            DRAW_ERRORS.append(f"{cls.__name__}: {traceback.format_exc().splitlines()[-1]}")
            raise
    cls.draw = draw
    cls.bl_options = set(getattr(cls, "bl_options", set())) - {'DEFAULT_CLOSED'}


for _cls in PANELS:
    _wrap(_cls)

SHOTS = os.environ.get("RIGGER_SHOTS", os.path.join(REPO, "gui_shots"))
DISPLAY = os.environ.get("DISPLAY", ":99")
os.makedirs(SHOTS, exist_ok=True)
REPORT = []


def log(msg):
    print("[gui]", msg)
    REPORT.append(msg)


def check(cond, msg):
    log(("PASS " if cond else "FAIL ") + msg)


def screenshot(name):
    path = os.path.join(SHOTS, f"{len(os.listdir(SHOTS)):02d}_{name}.png")
    subprocess.run(["import", "-display", DISPLAY, "-window", "root", path], check=False)


def xdo(*args):
    subprocess.Popen(["xdotool", *map(str, args)], env=dict(os.environ, DISPLAY=DISPLAY))


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def view3d():
    win = bpy.context.window_manager.windows[0]
    area = max((a for a in win.screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height)
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return win, area, region


def override():
    win, area, region = view3d()
    return bpy.context.temp_override(window=win, area=area, region=region,
                                     screen=win.screen, space_data=area.spaces.active)


def screen_xy(point):
    """Window pixel (top-left origin) of a 3D point in the main 3D view."""
    win, area, region = view3d()
    rv3d = area.spaces.active.region_3d
    co = view3d_utils.location_3d_to_region_2d(region, rv3d, point)
    return int(region.x + co.x), int(win.height - (region.y + co.y))


def setup_view(view='FRONT', frame=None):
    win, area, region = view3d()
    space = area.spaces.active
    space.show_region_ui = True
    space.overlay.show_bones = True
    space.shading.type = 'SOLID'
    with override():
        bpy.ops.view3d.view_axis(type=view)
        if frame is not None:
            for o in bpy.context.view_layer.objects.selected:
                o.select_set(False)
            frame.select_set(True)
            bpy.context.view_layer.objects.active = frame
            bpy.ops.view3d.view_selected()
    ui = next(r for r in area.regions if r.type == 'UI')
    try:
        ui.active_panel_category = "Rigger"
    except (AttributeError, TypeError):
        # Blender < 4.2: click the "Rigger" tab on the sidebar's tab strip
        STATE["click_tab"] = True


def zoom_to(points, view='FRONT'):
    """Frame a set of points by selecting a temporary empty cloud."""
    win, area, region = view3d()
    rv3d = area.spaces.active.region_3d
    center = sum(points, Vector()) / len(points)
    radius = max((p - center).length for p in points)
    with override():
        bpy.ops.view3d.view_axis(type=view)
    rv3d.view_location = center
    rv3d.view_distance = radius * 3.5


def build_character():
    """A smooth one-piece humanoid made from metaballs, plus loose eyeballs."""
    specs = templates.mirrored(templates.human_specs(fingers=False, face=False))
    mb = bpy.data.metaballs.new("body_mb")
    mb.resolution = 0.025
    mb.render_resolution = 0.025
    obj = bpy.data.objects.new("body_mb", mb)
    bpy.context.scene.collection.objects.link(obj)
    radius = {"hips": 0.13, "spine": 0.13, "neck": 0.06, "head": 0.1, "shoulder": 0.06,
              "upper_arm": 0.05, "forearm": 0.042, "hand": 0.04, "thigh": 0.08, "shin": 0.06,
              "foot": 0.045, "toe": 0.035}
    for spec in specs:
        stem = utils.split_side(spec.name)[0].split(".")[0]
        r = radius.get(stem, 0.05)
        d = spec.tail - spec.head
        el = mb.elements.new(type='CAPSULE')
        el.co = (spec.head + spec.tail) / 2
        el.radius = r
        el.size_x = d.length / 2
        el.rotation = Vector((1, 0, 0)).rotation_difference(d.normalized())
        el.stiffness = 2.0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.context.view_layer.update()
    with override():
        bpy.ops.object.convert(target='MESH')
    body = bpy.context.object
    body.name = "body"
    for side in (1, -1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.014,
                                             location=(side * 0.032, -0.085, 1.686))
        eye = bpy.context.object
        eye.name = f"eyeball_{'L' if side > 0 else 'R'}"
    return body


# ---------------------------------------------------------------------------
# Steps (run one by one from a timer so the UI redraws between them)
# ---------------------------------------------------------------------------

STATE = {}


def click_rigger_tab():
    """Click the "Rigger" tab: 4th tab (Item, Tool, View, Rigger) on the sidebar's right edge."""
    win, area, region = view3d()
    ui = next(r for r in area.regions if r.type == 'UI')
    x = ui.x + ui.width - 8
    top = win.height - (ui.y + ui.height)
    scale = bpy.context.preferences.system.ui_scale
    y = top + int(164 * scale)
    xdo("mousemove", x, y, "sleep", "0.2", "click", "1")


def step_start():
    xdo("key", "Escape")  # close the splash screen
    for name in ("Cube", "Camera", "Light"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj)
    rigger.register()
    body = build_character()
    STATE["body"] = body
    setup_view('FRONT', body)


def step_shot_start():
    if STATE.get("click_tab"):
        click_rigger_tab()
    screenshot("character")


def make_marker_step(key):
    def run():
        win, area, region = view3d()
        point = templates.DEFAULT_POINTS[key].copy()
        if isinstance(point, float):
            return
        x, y = screen_xy(point)
        bpy.ops.object.select_all(action='DESELECT')
        STATE["body"].select_set(True)
        bpy.context.view_layer.objects.active = STATE["body"]
        with override():
            res = bpy.ops.rigger.place_marker('INVOKE_DEFAULT', marker=key)
        STATE.setdefault("marker_invoke", []).append(res)
        xdo("mousemove", x, y, "sleep", "0.3", "click", "1")
    return run


def step_check_markers():
    missing = [k for k, *_ in templates.MARKERS if bpy.data.objects.get("RGM-" + k) is None]
    check(not missing, f"all markers placed by mouse clicks (missing: {missing})")
    errs = []
    for key, *_ in templates.MARKERS:
        obj = bpy.data.objects.get("RGM-" + key)
        if obj is None:
            continue
        expect = templates.DEFAULT_POINTS[key]
        errs.append((obj.location - Vector((abs(expect.x) if expect.x else 0, expect.y, expect.z))).length)
    log(f"marker distance to intended landmark: {['%.3f' % e for e in errs]}")
    screenshot("markers_placed")


def step_fit():
    bpy.ops.object.select_all(action='DESELECT')
    STATE["body"].select_set(True)
    bpy.context.view_layer.objects.active = STATE["body"]
    with override():
        res = bpy.ops.rigger.fit_markers()
    check(res == {'FINISHED'}, "fit reference to markers")
    STATE["ref"] = bpy.context.object


def step_shot_ref():
    screenshot("reference_edit_mode")


def step_generate():
    ref = STATE["ref"]
    with override():
        bpy.ops.object.mode_set(mode='OBJECT')
        res = bpy.ops.rigger.generate()
    check(res == {'FINISHED'}, "generate rig")
    STATE["rig"] = bpy.context.object
    setup_view('FRONT', STATE["body"])
    _ = ref


def step_shot_rig():
    screenshot("rig_pose_mode")


def step_bind():
    rig, body = STATE["rig"], STATE["body"]
    with override():
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for o in bpy.data.objects:
            if o.type == 'MESH' and (o == body or o.name.startswith("eyeball")):
                o.select_set(True)
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
    win, area, region = view3d()
    xdo("mousemove", region.x + region.width // 2, win.height - (region.y + region.height // 2))


def step_bind_dialog():
    with override():
        res = bpy.ops.rigger.bind('INVOKE_DEFAULT', method='AUTO', eyes="eyeball_L, eyeball_R")
    check(res == {'RUNNING_MODAL'}, f"bind options dialog opens ({res})")


def step_shot_bind_dialog():
    screenshot("bind_dialog")
    xdo("key", "Return")


def step_bind_check():
    rig, body = STATE["rig"], STATE["body"]
    deform = {b.name for b in rig.data.bones if b.use_deform}
    mod = next((m for m in body.modifiers if m.type == 'ARMATURE'), None)
    check(mod is not None and mod.object == rig, "bind dialog confirmed with Enter")
    from rigger.skin import unweighted_vertices
    check(not unweighted_vertices(body, deform), "every body vertex weighted")
    for side in "LR":
        eye = bpy.data.objects[f"eyeball_{side}"]
        groups = {eye.vertex_groups[g.group].name for v in eye.data.vertices for g in v.groups}
        check(groups == {f"DEF-eye.{side}"}, f"eyeball_{side} bound rigidly to its eye bone ({sorted(groups)})")
        center = sum((eye.matrix_world @ v.co for v in eye.data.vertices), Vector()) / len(eye.data.vertices)
        log(f"eyeball_{side} centre {center[:]}, DEF-eye head {rig.data.bones[f'DEF-eye.{side}'].head_local[:]}")
    with override():
        bpy.ops.object.mode_set(mode='POSE')


def step_pose():
    rig = STATE["rig"]
    pbs = rig.pose.bones
    meta = utils.get_meta(rig)
    for limb in meta["limbs"]:
        if limb["type"] == "arm":
            pbs[limb["fk"][0]].rotation_quaternion = Matrix.Rotation(
                math.radians(-45 if limb["side"] == "L" else 45), 4, 'Z').to_quaternion()
            pbs[limb["fk"][1]].rotation_quaternion = Matrix.Rotation(
                math.radians(-50), 4, 'X').to_quaternion()
        else:
            if limb["side"] == "L":
                pbs[limb["ik_ctrl"]].location = (0, 0, 0.25)
                pbs[limb["ik_ctrl"]].location = (0, -0.0, 0.25)
    pbs["torso"].location = (0, 0, -0.08)
    pbs[meta["spine"]["chest"]].rotation_quaternion = Matrix.Rotation(math.radians(15), 4, 'X').to_quaternion()
    pbs[meta["neck"]["head"]].rotation_quaternion = Matrix.Rotation(math.radians(20), 4, 'Z').to_quaternion()
    setup_view('FRONT', STATE["body"])


def step_shot_pose_front():
    screenshot("posed_front")


def step_view_side():
    with override():
        bpy.ops.view3d.view_axis(type='RIGHT')


def step_shot_pose_side():
    screenshot("posed_side")


def step_face():
    rig = STATE["rig"]
    meta = utils.get_meta(rig)
    face = meta["faces"][0] if meta.get("faces") else None
    if face:
        rig.pose.bones[face["face"]]["blink_L"] = 1.0
        rig.pose.bones[face["jaw"]].rotation_quaternion = Matrix.Rotation(
            math.radians(15), 4, 'X').to_quaternion()
        rig.update_tag()
    head = rig.matrix_world @ rig.pose.bones["DEF-head"].head
    zoom_to([head + Vector((0, 0, 0.12)), head - Vector((0, 0, 0.05)), head + Vector((0.12, 0, 0))])


def step_shot_face():
    screenshot("face_blink_jaw")


def step_picker():
    rig = STATE["rig"]
    bpy.context.view_layer.objects.active = rig
    with override():
        bpy.ops.rigger.reset_pose()
    setup_view('FRONT', STATE["body"])
    bpy.context.view_layer.objects.active = rig
    with override():
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.rigger.select_controls(bones="hand_fk.L,forearm_fk.L,upper_arm_fk.L")
    sel = sorted(pb.name for pb in rig.pose.bones if utils.is_pose_bone_selected(pb))
    check(sel == ["forearm_fk.L", "hand_fk.L", "upper_arm_fk.L"], f"picker selection in the GUI ({sel})")


def step_shot_ui():
    screenshot("sidebar_controls")


def step_scroll_ui():
    win, area, region = view3d()
    ui = next(r for r in area.regions if r.type == 'UI')
    x = ui.x + ui.width // 2
    y = win.height - (ui.y + ui.height // 2)
    xdo("mousemove", x, y, "click", "--repeat", "12", "5")


def step_shot_ui2():
    screenshot("sidebar_scrolled")


# -- Panels one by one ---------------------------------------------------------------

def show_only(*names):
    for cls in reversed(PANELS):
        if cls.is_registered:
            bpy.utils.unregister_class(cls)
    for cls in PANELS:
        if cls is ui.RIGGER_PT_main or cls.__name__ in names:
            bpy.utils.register_class(cls)
    for area in bpy.context.window_manager.windows[0].screen.areas:
        area.tag_redraw()


def show_all():
    for cls in PANELS:
        if not cls.is_registered:
            bpy.utils.register_class(cls)


def activate(obj, mode='OBJECT', bone=None):
    with override():
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for o in bpy.context.view_layer.objects.selected:
            o.select_set(False)
        obj.hide_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=mode)
    if bone is not None:
        obj.data.bones.active = obj.data.bones[bone]
        if obj.mode == 'POSE':
            for pb in obj.pose.bones:
                utils.select_pose_bone(pb, pb.name == bone)


def widen_sidebar():
    win, area, region = view3d()
    ui_region = next(r for r in area.regions if r.type == 'UI')
    x = ui_region.x + 2
    y = win.height - (ui_region.y + ui_region.height // 2)
    xdo("mousemove", x, y, "sleep", "0.2", "mousedown", "1", "sleep", "0.2", "mousemove_relative", "--", "-220", "0",
        "sleep", "0.2", "mouseup", "1")


def panel_steps(name, panels, prepare=None, scroll=0):
    def show():
        if prepare:
            prepare()
        show_only(*panels)

    def shot():
        screenshot(name)
        if scroll:
            win, area, region = view3d()
            ui_region = next(r for r in area.regions if r.type == 'UI')
            xdo("mousemove", ui_region.x + ui_region.width // 2, win.height - (ui_region.y + ui_region.height // 2),
                "click", "--repeat", str(scroll), "5")

    steps = [show, shot]
    if scroll:
        steps.append(lambda: screenshot(name + "_scrolled"))
    return steps


def prep_reference():
    ref = STATE["rig"].rigger.reference
    activate(ref, 'POSE', "upper_arm.L" if "upper_arm.L" in ref.data.bones else None)
    ref.data.bones.active = ref.data.bones["shoulder.L"]


def prep_rig_pose():
    rig = STATE["rig"]
    arm = next(l for l in utils.get_meta(rig)["limbs"] if l["type"] == "arm" and l["side"] == "L")
    activate(rig, 'POSE', arm["fk"][1])


def prep_rig_object():
    activate(STATE["rig"])


def prep_export(tab):
    def run():
        activate(STATE["rig"])
        bpy.context.scene.rigger_export.ui_tab = tab
        bpy.context.scene.rigger_export.only_active = False
    return run


# -- Features in the live session ------------------------------------------------------

def step_ik_fk():
    rig = STATE["rig"]
    meta = utils.get_meta(rig)
    index = next(i for i, l in enumerate(meta["limbs"]) if l["type"] == "arm" and l["side"] == "L")
    limb = meta["limbs"][index]
    activate(rig, 'POSE', limb["fk"][1])
    pbs = rig.pose.bones
    pbs[limb["prop_bone"]]["ik_fk"] = 0.0
    pbs[limb["fk"][0]].rotation_quaternion = Matrix.Rotation(math.radians(-40), 4, 'Z').to_quaternion()
    pbs[limb["fk"][1]].rotation_quaternion = Matrix.Rotation(math.radians(-60), 4, 'X').to_quaternion()
    bpy.context.view_layer.update()
    before = (rig.matrix_world @ pbs["DEF-hand.L"].head).copy()
    with override():
        res = bpy.ops.rigger.snap_ik_to_fk(limb=index)
    bpy.context.view_layer.update()
    after = rig.matrix_world @ pbs["DEF-hand.L"].head
    check(res == {'FINISHED'} and pbs[limb["prop_bone"]]["ik_fk"] == 1.0 and (after - before).length < 1e-3,
          f"switch the arm to IK in place ({(after - before).length:.5f})")
    with override():
        res = bpy.ops.rigger.pin_property(bone=limb["prop_bone"], prop="ik_fk")
    check(res == {'FINISHED'} and limb["prop_bone"] in rig.rigger.pinned_props, "pin the IK-FK slider")


def step_shot_ik():
    show_only("RIGGER_PT_controls")
    screenshot("animate_ik_pinned")


def step_tools():
    rig = STATE["rig"]
    activate(rig, 'POSE')
    with override():
        res = bpy.ops.rigger.reset_pose()
        res = bpy.ops.rigger.set_pose(pose='T')
    meta = utils.get_meta(rig)
    upper = rig.pose.bones["DEF-upper_arm.L"]
    direction = (upper.tail - upper.head).normalized()
    check(res == {'FINISHED'} and abs(direction.z) < 0.1, f"T-pose from the Rig Tools ({direction.z:.3f})")
    rig.pose.bones[meta["neck"]["head"]].rotation_quaternion = Matrix.Rotation(0.4, 4, 'Z').to_quaternion()
    activate(rig, 'POSE', meta["neck"]["head"])
    with override():
        res = bpy.ops.rigger.mirror_pose()
    check(res == {'FINISHED'}, "mirror pose")
    with override():
        bpy.ops.rigger.layer_add(name="Hands")
        bpy.ops.rigger.reset_pose()
    check(len(rig.rigger.layers) >= 1, "add a rig layer")
    show_only("RIGGER_PT_tools")


def step_shot_tools():
    screenshot("rig_tools_tpose")


def step_pose_blend():
    rig = STATE["rig"]
    meta = utils.get_meta(rig)
    fingers = [f for f in meta.get("fingers", []) if f["fk"][0].endswith(".L")]
    activate(rig, 'POSE', fingers[0]["fk"][0])
    for finger in fingers:
        for ctrl in finger["fk"]:
            rig.pose.bones[ctrl].rotation_quaternion = Matrix.Rotation(math.radians(-70), 4, 'X').to_quaternion()
    with override():
        res = bpy.ops.rigger.add_pose_blend(kind='FIST', side='L')
    from rigger.poses import load_blends
    rig = bpy.context.object
    STATE["rig"] = rig
    names = [b["name"] for b in load_blends(rig.rigger.reference)]
    check(res == {'FINISHED'} and "fist.L" in names, f"store the fist pose blend ({names})")
    with override():
        bpy.ops.rigger.reset_pose()
    show_only("RIGGER_PT_tools")


def step_shot_pose_blend():
    screenshot("pose_blend_stored")


MIXAMO = {"hips": "Hips", "spine.01": "Spine", "spine.02": "Spine1", "spine.03": "Spine2", "neck": "Neck",
          "head": "Head"}
for _s in "LR":
    _side = "Left" if _s == "L" else "Right"
    MIXAMO.update({f"shoulder.{_s}": f"{_side}Shoulder", f"upper_arm.{_s}": f"{_side}Arm",
                   f"forearm.{_s}": f"{_side}ForeArm", f"hand.{_s}": f"{_side}Hand",
                   f"thigh.{_s}": f"{_side}UpLeg", f"shin.{_s}": f"{_side}Leg", f"foot.{_s}": f"{_side}Foot",
                   f"toe.{_s}": f"{_side}ToeBase"})


def mixamo(name, location):
    specs = templates.mirrored(templates.human_specs(fingers=False, face=False))
    obj = bpy.data.objects.new(name, bpy.data.armatures.new(name))
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    activate(obj, 'EDIT')
    for spec in specs:
        eb = obj.data.edit_bones.new("mixamorig:" + MIXAMO[spec.name])
        eb.head, eb.tail = spec.head * 0.9, spec.tail * 0.9
    for spec in specs:
        if spec.parent:
            obj.data.edit_bones["mixamorig:" + MIXAMO[spec.name]].parent = \
                obj.data.edit_bones["mixamorig:" + MIXAMO[spec.parent]]
    activate(obj, 'OBJECT')
    action = bpy.data.actions.new(name + "_walk")
    utils.assign_action(obj, action)
    for pb in obj.pose.bones:
        pb.rotation_mode = 'QUATERNION'
    for frame, angle in ((1, 0), (10, 60)):
        pb = obj.pose.bones["mixamorig:LeftArm"]
        rest = pb.bone.matrix_local.to_3x3()
        pb.rotation_quaternion = (rest.inverted() @ Matrix.Rotation(math.radians(angle), 3, 'Y') @ rest).to_quaternion()
        pb.keyframe_insert("rotation_quaternion", frame=frame)
    return obj


def arm_error(src, rig):
    sp = src.pose.bones
    a = (src.matrix_world @ sp["mixamorig:LeftForeArm"].tail) - (src.matrix_world @ sp["mixamorig:LeftArm"].head)
    b = (rig.matrix_world @ rig.pose.bones["DEF-forearm.L"].tail) - \
        (rig.matrix_world @ rig.pose.bones["DEF-upper_arm.L"].head)
    return math.degrees(a.angle(b))


def step_retarget_setup():
    rig = STATE["rig"]
    rig.animation_data_clear()
    src = mixamo("mixamo", (1.2, 0, 0))
    STATE["src"] = src
    remap = bpy.context.scene.rigger_remap
    remap.source, remap.target = src, rig
    activate(rig)
    with override():
        bpy.ops.rigger.remap_auto()
        res = bpy.ops.rigger.remap_preview()
    check(res == {'FINISHED'} and retarget.preview_active(), "bind preview")
    bpy.context.scene.frame_set(1)
    win, area, region = view3d()
    xdo("mousemove", region.x + region.width // 2, win.height - (region.y + region.height // 2), "sleep", "0.3",
        "key", "--repeat", "9", "--delay", "80", "Right")
    show_only("RIGGER_PT_remap")


def step_retarget_check():
    scene = bpy.context.scene
    err = arm_error(STATE["src"], STATE["rig"])
    check(scene.frame_current == 10 and err < 5.0,
          f"arrow keys change frames, the preview follows (frame {scene.frame_current}, {err:.2f} deg)")
    screenshot("retarget_preview")


def step_retarget_bake():
    with override():
        bpy.ops.rigger.remap_preview()
        res = bpy.ops.rigger.remap_bake()
    rig = STATE["rig"]
    check(res == {'FINISHED'} and rig.animation_data.action is not None and not retarget.preview_active(),
          "unbind and bake the retarget")
    bpy.context.scene.frame_set(10)
    check(arm_error(STATE["src"], rig) < 5.0, "baked retarget follows the source")


def step_quick_rig():
    skel = mixamo("game_skeleton", (-1.2, 0, 0))
    qs = bpy.context.scene.rigger_quick
    qs.armature = skel
    activate(skel)
    with override():
        bpy.ops.rigger.quick_rig_guess()
        res = bpy.ops.rigger.quick_rig()
    rig_q = next((o for o in bpy.data.objects if o.get("rigger_quick_source") == "game_skeleton"), None)
    check(res == {'FINISHED'} and rig_q is not None, f"Quick Rig ({len(qs.limbs)} limbs)")
    STATE["rig_q"] = rig_q
    bpy.context.scene.frame_set(10)
    a = skel.matrix_world @ skel.pose.bones["mixamorig:LeftForeArm"].head
    check((a - (skel.matrix_world @ skel.data.bones["mixamorig:LeftForeArm"].head_local)).length > 0.05,
          "Quick Rig baked the skeleton's animation, the skeleton follows the rig")
    activate(skel)
    show_only("RIGGER_PT_quick_rig")
    activate(rig_q, 'POSE')


def step_shot_quick_rig():
    with override():
        bpy.ops.view3d.view_all()
    screenshot("quick_rig")


def step_export():
    import tempfile
    out_dir = tempfile.mkdtemp()
    STATE["out_dir"] = out_dir
    s = bpy.context.scene.rigger_export
    s.file_format, s.only_active, s.naming = 'GLB', True, 'UNREAL'
    activate(STATE["rig"])
    win, area, region = view3d()
    xdo("mousemove", region.x + region.width // 2, win.height - (region.y + region.height // 2))
    with override():
        res = bpy.ops.rigger.export_fbx('INVOKE_DEFAULT', filepath=os.path.join(out_dir, "gui.glb"))
    check(res == {'RUNNING_MODAL'}, f"export file browser opens ({res})")


def step_shot_export_browser():
    screenshot("export_file_browser")
    xdo("key", "Return")


def step_export_check():
    out = os.path.join(STATE["out_dir"], "gui.glb")
    check(os.path.exists(out) and os.path.getsize(out) > 1000, "GLB exported through the file browser")
    s = bpy.context.scene.rigger_export
    s.file_format = 'FBX'
    activate(STATE["rig"])
    with override():
        res = bpy.ops.rigger.export_fbx(filepath=os.path.join(STATE["out_dir"], "gui.fbx"))
        res2 = bpy.ops.rigger.quick_export()
    check(res == {'FINISHED'} and res2 == {'FINISHED'} and os.path.exists(os.path.join(STATE["out_dir"], "gui.fbx")),
          "FBX export and quick export")


def step_ui_report():
    show_all()
    missing = [c.__name__ for c in PANELS if not DRAWN.get(c.__name__)]
    check(not missing, f"every panel drawn in the real UI (missing {missing})")
    check(not DRAW_ERRORS, f"no draw errors in the real UI ({DRAW_ERRORS[:3]})")


def step_finish():
    with open(os.path.join(SHOTS, "report.txt"), "w") as fh:
        fh.write("\n".join(REPORT))
    bpy.ops.wm.quit_blender()


STEPS = ([step_start, step_shot_start]
         + [make_marker_step(k) for k, *_ in templates.MARKERS]
         + [step_check_markers, step_fit, step_shot_ref, step_generate, step_shot_rig, step_bind,
            step_bind_dialog, step_shot_bind_dialog, step_bind_check,
            step_pose, step_shot_pose_front, step_view_side, step_shot_pose_side, step_face,
            step_shot_face, step_picker, step_shot_ui, step_scroll_ui, step_shot_ui2, widen_sidebar]
         + panel_steps("panel_reference", ["RIGGER_PT_reference", "RIGGER_PT_template_options"], prep_reference, 8)
         + panel_steps("panel_generate_skin", ["RIGGER_PT_generate", "RIGGER_PT_skin"], prep_rig_object)
         + panel_steps("panel_animate", ["RIGGER_PT_controls", "RIGGER_PT_picker"], prep_rig_pose, 10)
         + panel_steps("panel_rig_tools", ["RIGGER_PT_tools"], prep_rig_pose, 8)
         + panel_steps("panel_export_rig", ["RIGGER_PT_export"], prep_export('RIG'))
         + panel_steps("panel_export_anim", ["RIGGER_PT_export"], prep_export('ANIM'))
         + panel_steps("panel_export_files", ["RIGGER_PT_export"], prep_export('MISC'))
         + [step_ik_fk, step_shot_ik, step_tools, step_shot_tools, step_pose_blend, step_shot_pose_blend,
            step_retarget_setup, step_retarget_check, step_retarget_bake, step_quick_rig, step_shot_quick_rig,
            step_export, step_shot_export_browser, step_export_check, step_ui_report, step_finish])


def tick():
    if not STEPS:
        return None
    step = STEPS.pop(0)
    try:
        step()
    except Exception:  # noqa: BLE001
        log(f"ERROR in {step.__name__}:\n{traceback.format_exc()}")
        screenshot(f"error_{step.__name__}")
    return 1.5


bpy.app.timers.register(tick, first_interval=3.0)
