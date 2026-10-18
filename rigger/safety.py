# SPDX-License-Identifier: GPL-3.0-or-later
"""A safety net around every Rigger operator.

If an operator fails part way, the user gets a readable message instead of a
Python traceback popup, the full traceback goes to the system console, and the
active object, selection and mode are put back. Operators that support undo
return FINISHED after an unexpected failure so that a single Ctrl+Z removes
whatever was changed before the error.
"""

import functools
import traceback

import bpy


class RiggerError(Exception):
    """An expected, user-facing problem (missing marker, wrong selection...)."""


def _snapshot(context):
    obj = context.view_layer.objects.active
    return {
        "active": obj.name if obj else None,
        "mode": obj.mode if obj else 'OBJECT',
        "selected": [o.name for o in context.view_layer.objects if o.select_get()],
    }


def _restore(context, state):
    """Best effort: every step is optional, the scene may have changed a lot."""
    try:
        obj = context.view_layer.objects.active
        if obj and obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:  # noqa: BLE001 - restoring must never raise
        pass
    try:
        for o in context.view_layer.objects:
            o.select_set(o.name in state["selected"])
        active = bpy.data.objects.get(state["active"]) if state["active"] else None
        if active and active.name in context.view_layer.objects:
            context.view_layer.objects.active = active
            if state["mode"] != 'OBJECT':
                bpy.ops.object.mode_set(mode=state["mode"])
    except Exception:  # noqa: BLE001
        pass


def guard(execute):
    """Wrap an operator's execute() with the safety net (idempotent)."""
    if getattr(execute, "_rigger_guarded", False):
        return execute

    @functools.wraps(execute)
    def wrapped(self, context):
        state = _snapshot(context)
        try:
            return execute(self, context)
        except RiggerError as exc:
            _restore(context, state)
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        except Exception as exc:  # noqa: BLE001 - the whole point is to catch anything
            traceback.print_exc()
            _restore(context, state)
            label = getattr(self, "bl_label", type(self).__name__)
            undo = 'UNDO' in getattr(self, "bl_options", set())
            self.report({'ERROR'}, f"{label} failed: {exc}. Details are in the system console"
                        + ("; Ctrl+Z undoes any partial change." if undo else "."))
            return {'FINISHED'} if undo else {'CANCELLED'}

    wrapped._rigger_guarded = True
    return wrapped


def guard_operators(classes):
    for cls in classes:
        if issubclass(cls, bpy.types.Operator) and "execute" in cls.__dict__:
            cls.execute = guard(cls.execute)
