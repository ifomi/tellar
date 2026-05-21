import Quartz

from .logging_setup import get_logger

log = get_logger(__name__)


class HotkeyListener:
    """Global hotkey via Quartz Event Tap. Ctrl+Space toggle, Ctrl+Escape cancel."""

    def __init__(self, on_toggle, on_cancel, toggle_key=49, toggle_mods=0x40000, cancel_key=53, cancel_mods=0x40000):
        # toggle: Ctrl+Space (key_code=49, modifiers=Ctrl)
        # cancel: Ctrl+Escape (key_code=53, modifiers=Ctrl)
        self.on_toggle = on_toggle
        self.on_cancel = on_cancel
        self.toggle_key = toggle_key
        self.toggle_mods = toggle_mods
        self.cancel_key = cancel_key
        self.cancel_mods = cancel_mods
        self._tap = None

    def _callback(self, proxy, event_type, event, refcon):
        if event_type == Quartz.kCGEventKeyDown:
            code = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventKeycode)
            flags = Quartz.CGEventGetFlags(event) & 0x1F0000
            repeat = Quartz.CGEventGetIntegerValueField(event, Quartz.kCGKeyboardEventAutorepeat)
            if repeat:
                return event
            if code == self.toggle_key and flags == self.toggle_mods:
                self.on_toggle()
            elif code == self.cancel_key and flags == self.cancel_mods:
                self.on_cancel()
        return event

    def start(self):
        mask = Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        self._tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self._callback,
            None,
        )
        if not self._tap:
            log.error("CGEventTapCreate returned NULL — Accessibility permission missing")
            raise PermissionError(
                "Cannot create event tap. Grant Accessibility permission in "
                "System Settings → Privacy & Security → Accessibility"
            )
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        loop = Quartz.CFRunLoopGetCurrent()
        Quartz.CFRunLoopAddSource(loop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        log.info("Event tap created and enabled")
