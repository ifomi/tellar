"""Native NSStatusBar menu bar icon — replacement for QSystemTrayIcon.

QSystemTrayIcon in PyQt6 has known visibility issues on macOS Sonoma+ when
the host app uses LSUIElement (menu-bar-only). NSStatusBar is the native
AppKit API; it always works for menu-bar accessory apps.
"""
from typing import Callable, Optional

from AppKit import (
    NSStatusBar,
    NSMenu,
    NSMenuItem,
    NSImage,
    NSObject,
    NSFont,
    NSFontAttributeName,
    NSAlert,
)
from Foundation import NSData, NSAttributedString, NSBundle
import objc

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from .logging_setup import get_logger

log = get_logger(__name__)


NSVariableStatusItemLength = -1.0
NSSquareStatusItemLength = -2.0


def qpixmap_to_nsimage(pix: QPixmap) -> NSImage:
    buf = QByteArray()
    qbuf = QBuffer(buf)
    qbuf.open(QIODevice.OpenModeFlag.WriteOnly)
    pix.save(qbuf, "PNG")
    qbuf.close()
    nsdata = NSData.dataWithBytes_length_(bytes(buf), buf.size())
    img = NSImage.alloc().initWithData_(nsdata)
    if img and img.isValid():
        # We deliberately render the pixmap at exact retina pixel size
        # (44px tall for a 22pt menubar slot) without using Qt's
        # devicePixelRatio. So the conversion to logical points is just
        # divide-by-2. This guarantees integer logical sizes; the
        # devicePixelRatio path produced half-pixel widths (e.g. 30.5pt)
        # which made NSStatusBarButton render the icon clipped/tiny.
        logical_w = pix.width() / 2.0
        logical_h = pix.height() / 2.0
        img.setSize_((logical_w, logical_h))
    log.debug("qpixmap_to_nsimage: PNG bytes=%d, NSImage valid=%s, size=%s",
              buf.size(), img.isValid() if img else False,
              img.size() if img else None)
    return img


class _MenuController(NSObject):
    """ObjC-side target for menu-item actions."""

    def initWithCallbacks_(self, cb):
        self = objc.super(_MenuController, self).init()
        if self is None:
            return None
        self._cb = cb
        self._auto_paste_item = None
        self._model_name = cb.get("model_name", "?")
        return self

    def doToggle_(self, sender):
        fn = self._cb.get("toggle")
        if fn:
            fn()

    def doToggleAutoPaste_(self, sender):
        sender.setState_(0 if sender.state() else 1)
        fn = self._cb.get("auto_paste_changed")
        if fn:
            fn(bool(sender.state()))

    def doAbout_(self, sender):
        # Show a native NSAlert with build info. Version comes from the
        # bundle's CFBundleShortVersionString (no need to thread it through
        # in code — Info.plist is canonical). Model is whatever transcriber
        # is configured to load.
        bundle = NSBundle.mainBundle()
        info = bundle.infoDictionary() or {}
        version = info.get("CFBundleShortVersionString", "?")
        body = (
            "Local push-to-talk voice dictation for macOS.\n\n"
            "⌃Space  start / stop recording\n"
            "⌃Esc    cancel\n\n"
            "Result is pasted into the active app, or just copied to the "
            "clipboard when Auto Paste is off (toggle in the menubar menu).\n\n"
            f"Speech recognition: {self._model_name}\n"
            "Runs entirely on this Mac — no cloud, no network calls "
            "after the model is downloaded."
        )
        alert = NSAlert.alloc().init()
        alert.setMessageText_(f"Tellar {version}")
        alert.setInformativeText_(body)
        alert.runModal()

    def doQuit_(self, sender):
        # Quit via Qt, not NSApp.terminate(). In a hybrid PyQt + AppKit app,
        # NSApp.terminate routes through Qt's NSApplicationDelegate, which
        # answers NSTerminateLater while Qt drains its event loop. The handoff
        # back to AppKit is unreliable: aboutToQuit may not fire cleanly,
        # the lock file may not be released, and macOS can interpret the
        # abnormal exit as a crash and auto-relaunch the app. Asking Qt to
        # quit lets aboutToQuit run normally and the process exits cleanly.
        app = QApplication.instance()
        if app is not None:
            app.quit()


class MenuBarIcon:
    """Public Pythonic API around NSStatusItem + NSMenu."""

    def __init__(self, on_toggle: Callable[[], None],
                 on_auto_paste_changed: Optional[Callable[[bool], None]] = None,
                 model_name: str = "?"):
        self._bar = NSStatusBar.systemStatusBar()
        # Variable length so the item resizes to fit a wide title (e.g. "0:01"
        # timer text). Square length clipped multi-character titles.
        self._item = self._bar.statusItemWithLength_(NSVariableStatusItemLength)
        self._item.button().setTitle_("●")
        self._item.button().setToolTip_("Tellar")
        log.info("NSStatusItem length=%s, button=%s",
                 self._item.length(), self._item.button())

        self._controller = _MenuController.alloc().initWithCallbacks_({
            "toggle": on_toggle,
            "auto_paste_changed": on_auto_paste_changed,
            "model_name": model_name,
        })
        # Keep strong reference so ARC doesn't deallocate
        objc.setAssociatedObject(self._item, b"controller", self._controller, 0x301)

        menu = NSMenu.alloc().init()

        self._record_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Start Recording  (⌃Space)", "doToggle:", ""
        )
        self._record_item.setTarget_(self._controller)
        self._record_item.setEnabled_(False)
        menu.addItem_(self._record_item)

        menu.addItem_(NSMenuItem.separatorItem())

        self._auto_paste_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Auto Paste", "doToggleAutoPaste:", ""
        )
        self._auto_paste_item.setTarget_(self._controller)
        self._auto_paste_item.setState_(1)
        menu.addItem_(self._auto_paste_item)

        menu.addItem_(NSMenuItem.separatorItem())

        about_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "About Tellar…", "doAbout:", ""
        )
        about_item.setTarget_(self._controller)
        menu.addItem_(about_item)

        menu.addItem_(NSMenuItem.separatorItem())

        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit Tellar", "doQuit:", "q"
        )
        quit_item.setTarget_(self._controller)
        menu.addItem_(quit_item)

        self._item.setMenu_(menu)
        log.info("NSStatusItem created and attached to system menu bar")

    def set_icon_pixmap(self, pix: QPixmap):
        img = qpixmap_to_nsimage(pix)
        if img and img.isValid() and img.size().width > 0:
            self._item.button().setImage_(img)
            self._item.button().setTitle_("")
            log.info("Status icon set from pixmap (size=%s)", img.size())
        else:
            log.warning("Pixmap conversion produced invalid NSImage; keeping title fallback")
            self._item.button().setTitle_("●")

    def set_icon_title(self, text: str):
        """For showing live timer text instead of icon."""
        self._item.button().setImage_(None)
        # Tabular (monospaced) digits so the item width stays constant as the
        # timer ticks ("0:01" → "0:02" → "0:09" → "0:10"). With proportional
        # digits each second-tick changed pixel width, nudging adjacent menu
        # bar icons.
        font = NSFont.monospacedDigitSystemFontOfSize_weight_(0.0, 0.0)
        attr_title = NSAttributedString.alloc().initWithString_attributes_(
            text, {NSFontAttributeName: font}
        )
        self._item.button().setAttributedTitle_(attr_title)

    def set_record_action_text(self, text: str):
        self._record_item.setTitle_(text)

    def set_record_action_enabled(self, enabled: bool):
        self._record_item.setEnabled_(enabled)

    def is_auto_paste_enabled(self) -> bool:
        return bool(self._auto_paste_item.state())
