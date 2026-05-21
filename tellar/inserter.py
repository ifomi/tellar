import os
import subprocess
import time
from typing import Optional

import Quartz
from AppKit import NSWorkspace, NSApplicationActivateIgnoringOtherApps, NSRunningApplication

from .logging_setup import get_logger

log = get_logger(__name__)


kVK_V = 0x09


def get_frontmost_app() -> Optional[NSRunningApplication]:
    """Get the frontmost app, excluding our own process and login window."""
    ws = NSWorkspace.sharedWorkspace()
    app = ws.frontmostApplication()
    if not app:
        return None
    bundle_id = app.bundleIdentifier() or ""
    if bundle_id in ("com.apple.loginwindow",):
        return None
    my_pid = os.getpid()
    if app.processIdentifier() == my_pid:
        return None
    return app


def _activate_app(app: NSRunningApplication) -> bool:
    return app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)


def _simulate_paste():
    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
    cmd_v_down = Quartz.CGEventCreateKeyboardEvent(src, kVK_V, True)
    Quartz.CGEventSetFlags(cmd_v_down, Quartz.kCGEventFlagMaskCommand)
    cmd_v_up = Quartz.CGEventCreateKeyboardEvent(src, kVK_V, False)
    Quartz.CGEventSetFlags(cmd_v_up, Quartz.kCGEventFlagMaskCommand)

    Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_v_down)
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_v_up)


def insert_text(text: str, fallback_app: Optional[NSRunningApplication] = None, auto_paste: bool = True) -> bool:
    """Copy text to clipboard and optionally paste into the active app.

    When auto_paste is True:
    1. Current frontmost app (user switched during recording)
    2. fallback_app (saved at recording start)
    When auto_paste is False: clipboard only.

    Returns True if paste was attempted, False if clipboard-only.
    """
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)

    if not auto_paste:
        log.debug("auto_paste=False, clipboard-only")
        return False

    target = get_frontmost_app() or fallback_app
    if target is None:
        log.warning("No target app for paste, clipboard-only")
        return False

    target_bundle = target.bundleIdentifier() or "?"
    if _activate_app(target):
        log.debug("Activated %s, simulating Cmd+V", target_bundle)
        time.sleep(0.1)
        _simulate_paste()
        return True

    log.warning("Could not activate %s, clipboard-only fallback", target_bundle)
    return False
