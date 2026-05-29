"""Tellar Studio — a side panel that collects dictations for editing.

When Studio mode is on (menu bar toggle), each dictation is appended into
this window's editor instead of being pasted into the active app. The user
edits the text by hand, then copies the finished result to the clipboard.

This is Phase 1: a plain editable text field plus an Undo / Redo / Clear /
Cut / Copy toolbar. LLM transformation presets, geometry persistence,
pin-on-top and the overlay badge are deliberately out of scope here — they
layer on later.

Undo/redo is managed by us, not by QTextEdit's built-in stack. The built-in
stack merges a programmatic append with the manual typing that follows it
(positional adjacency), so a single undo would wipe both the typed edit and
the dictation. We disable it and keep our own snapshot stack with explicit
step boundaries: each dictation is one step; a burst of manual typing is
coalesced into one step on a short pause (like a normal editor does).

Window shape: a normal, resizable top-level window meant to live along the
side of the screen. The app's accessory activation policy keeps it out of the
Dock and ⌘Tab; a normal (non-Tool) window means it's a first-class window in
Mission Control and can be raised by clicking it there.
"""
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import (
    QGuiApplication,
    QTextCursor,
    QShortcut,
    QKeySequence,
    QPixmap,
    QIcon,
)
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QApplication,
)

from .logging_setup import get_logger

log = get_logger(__name__)

# How long manual typing stays "open" before it's committed as one undo step.
TYPING_COMMIT_MS = 350
# Cap the history so a very long session can't grow snapshots unbounded.
MAX_HISTORY = 200

# Toolbar icon buttons: equal width, small monochrome SF Symbol glyphs.
ICON_BTN_WIDTH = 40
ICON_PT = 16


def sf_icon(name: str) -> QIcon | None:
    """Render an SF Symbol as a black monochrome QIcon, or None if unavailable.

    SF Symbols are template (tint-based) images; we draw the glyph and then
    fill black through it (sourceIn) to get a flat black icon, render at 2x for
    retina, and hand it to Qt as a PNG. Falls back to None on older macOS or any
    failure so callers can use a text label instead.
    """
    try:
        from AppKit import (
            NSImage, NSColor, NSBitmapImageRep, NSMakeRect,
            NSCompositingOperationSourceIn, NSCompositingOperationSourceOver,
            NSRectFillUsingOperation, NSBitmapImageFileTypePNG,
            NSImageSymbolConfiguration, NSFontWeightRegular,
        )
        base = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
        if base is None:
            return None
        # Regular weight — a middle ground (bold looked heavy, ultraLight faint).
        base = base.imageWithSymbolConfiguration_(
            NSImageSymbolConfiguration.configurationWithPointSize_weight_(ICON_PT, NSFontWeightRegular)
        )
        sz = base.size()
        scale = 2
        pw, ph = max(1, int(round(sz.width))) * scale, max(1, int(round(sz.height))) * scale
        out = NSImage.alloc().initWithSize_((pw, ph))
        out.lockFocus()
        base.drawInRect_fromRect_operation_fraction_(
            NSMakeRect(0, 0, pw, ph), NSMakeRect(0, 0, sz.width, sz.height),
            NSCompositingOperationSourceOver, 1.0)
        # Black fill — Qt fades it for the disabled state, so enabled vs disabled
        # stays clearly distinguishable (a mid-gray fill blurred that line).
        NSColor.blackColor().set()
        NSRectFillUsingOperation(NSMakeRect(0, 0, pw, ph), NSCompositingOperationSourceIn)
        out.unlockFocus()
        rep = NSBitmapImageRep.imageRepWithData_(out.TIFFRepresentation())
        png = bytes(rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {}))
        pm = QPixmap()
        if not pm.loadFromData(png, "PNG"):
            return None
        pm.setDevicePixelRatio(scale)
        return QIcon(pm)
    except Exception:
        log.exception("SF Symbol icon %r failed", name)
        return None


class StudioWindow(QWidget):
    DEFAULT_WIDTH = 360
    DEFAULT_HEIGHT = 640

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tellar Studio")
        # Called when the user closes the window via its title bar so the app
        # can keep the menu checkbox / Auto Paste in sync. Set by the owner.
        self.on_close = None
        # A normal top-level window so it participates in Mission Control and
        # can be raised by clicking its thumbnail there. We deliberately avoid
        # Qt.WindowType.Tool: Tool windows are utility panels that float with
        # the active app and aren't first-class in Mission Control. The app's
        # accessory activation policy (see app.py) is what keeps it out of the
        # Dock and ⌘Tab now, so the window type no longer has to.
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(280, 320)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

        # --- custom undo/redo state ---
        self._undo: list[str] = []   # snapshots to step back to
        self._redo: list[str] = []   # snapshots to step forward to
        self._baseline = ""          # last committed text
        self._applying = False       # True while WE mutate the editor (ignore textChanged)
        self._commit_timer = QTimer(self)
        self._commit_timer.setSingleShot(True)
        self._commit_timer.setInterval(TYPING_COMMIT_MS)
        self._commit_timer.timeout.connect(self._commit_pending)

        editor = QTextEdit()
        editor.setAcceptRichText(False)
        # We run our own undo stack (see module docstring); turn off Qt's so it
        # doesn't fight us and so ⌃Z reaches our shortcut instead of the editor.
        editor.setUndoRedoEnabled(False)
        editor.setPlaceholderText(
            "Dictations land here while Studio is on.\n"
            "Edit freely, then Copy."
        )
        self.editor = editor
        editor.textChanged.connect(self._on_text_changed)

        self._undo_btn = self._icon_button("arrow.counterclockwise", "↺", "Undo (⌘Z)")
        self._undo_btn.clicked.connect(self.undo)
        self._redo_btn = self._icon_button("arrow.clockwise", "↻", "Redo (⌘⇧Z)")
        self._redo_btn.clicked.connect(self.redo)

        self._clear_btn = self._icon_button("trash", "Clear", "Clear the panel")
        self._clear_btn.clicked.connect(self.clear_all)
        self._cut_btn = self._icon_button("scissors", "Cut", "Copy everything to the clipboard and clear the panel")
        self._cut_btn.clicked.connect(self.cut_to_clipboard)
        self._copy_btn = self._icon_button("doc.on.doc", "Copy", "Copy everything to the clipboard")
        self._copy_btn.clicked.connect(self.copy_to_clipboard)

        # Two visual blocks: undo/redo, then clear/cut/copy. Tight spacing
        # within blocks (4px), a wider gap between them.
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        toolbar.addWidget(self._undo_btn)
        toolbar.addWidget(self._redo_btn)
        toolbar.addSpacing(12)
        toolbar.addWidget(self._clear_btn)
        toolbar.addWidget(self._cut_btn)
        toolbar.addWidget(self._copy_btn)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(toolbar)
        layout.addWidget(self.editor)

        # Keyboard shortcuts — Qt's editor undo is disabled, so bind our own.
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self.undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self.redo)

        self._refresh_buttons()

    def _icon_button(self, symbol: str, fallback_text: str, tip: str) -> QPushButton:
        """Build a fixed-width toolbar button showing an SF Symbol icon, or the
        fallback text if the symbol can't be rendered."""
        btn = QPushButton()
        icon = sf_icon(symbol)
        if icon is not None:
            btn.setIcon(icon)
            btn.setIconSize(QSize(ICON_PT, ICON_PT))
        else:
            btn.setText(fallback_text)
        btn.setToolTip(tip)
        btn.setFixedWidth(ICON_BTN_WIDTH)
        return btn

    # --- dictation routing -------------------------------------------------

    def append_dictation(self, text: str):
        """Append a freshly transcribed chunk on its own line as one undo step.

        Any in-progress manual typing is committed first (so it stays a
        separate step), then the append is pushed as its own snapshot.
        """
        text = text.strip()
        if not text:
            return
        self._commit_pending()          # flush pending typing as its own step
        self._push_undo(self._baseline)  # snapshot before the append
        self._applying = True
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.editor.document().isEmpty():
            cursor.insertText("\n")
        cursor.insertText(text)
        self.editor.setTextCursor(cursor)
        self.editor.ensureCursorVisible()
        self._applying = False
        self._baseline = self.editor.toPlainText()
        self._refresh_buttons()
        log.info("Studio: appended %d chars", len(text))

    # --- undo / redo -------------------------------------------------------

    def _on_text_changed(self):
        if self._applying:
            return
        # A manual edit invalidates the redo branch and (re)starts the typing
        # coalesce timer; the burst commits as one step after a short pause.
        self._redo.clear()
        self._commit_timer.start()
        self._refresh_buttons()

    def _commit_pending(self):
        """Commit any uncommitted manual edits as a single undo step."""
        self._commit_timer.stop()
        current = self.editor.toPlainText()
        if current != self._baseline:
            self._push_undo(self._baseline)
            self._baseline = current
            self._refresh_buttons()

    def _push_undo(self, snapshot: str):
        self._undo.append(snapshot)
        if len(self._undo) > MAX_HISTORY:
            self._undo.pop(0)

    def undo(self):
        self._commit_pending()  # in-progress typing becomes its own step first
        if not self._undo:
            return
        self._redo.append(self._baseline)
        target = self._undo.pop()
        self._set_text(target)
        self._baseline = target
        self._refresh_buttons()

    def redo(self):
        if not self._redo:
            return
        self._push_undo(self._baseline)
        target = self._redo.pop()
        self._set_text(target)
        self._baseline = target
        self._refresh_buttons()

    def _set_text(self, text: str):
        """Replace editor content programmatically (ignored by undo tracking)."""
        self._applying = True
        self.editor.setPlainText(text)
        cursor = self.editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.editor.setTextCursor(cursor)
        self._applying = False
        self.editor.ensureCursorVisible()

    # --- toolbar actions ---------------------------------------------------

    def copy_to_clipboard(self):
        text = self.editor.toPlainText()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        log.info("Studio: copied %d chars to clipboard", len(text))

    def cut_to_clipboard(self):
        """Copy everything to the clipboard and clear the panel — one click for
        'grab the result and start fresh'."""
        text = self.editor.toPlainText()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        log.info("Studio: cut %d chars to clipboard", len(text))
        self.clear_all()

    def clear_all(self):
        """Empty the editor as one undo step (so a clear can be undone)."""
        self._commit_pending()
        if not self.editor.toPlainText():
            return
        self._push_undo(self._baseline)
        self._redo.clear()
        self._set_text("")
        self._baseline = ""
        self._refresh_buttons()

    # --- visibility --------------------------------------------------------

    def show_panel(self):
        """Show the panel and bring it forward. Called when a Studio-mode
        recording starts so the window itself signals where the dictation
        will land."""
        if not self.isVisible():
            self._place_default()
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event):
        """Closing the window leaves Studio routing: visibility is coupled to
        the menu toggle. Notify the owner so it unchecks the menu item and
        restores Auto Paste. We hide (no WA_DeleteOnClose) so the editor
        content survives and comes back when Studio is re-enabled."""
        if self.on_close:
            try:
                self.on_close()
            except Exception:
                log.exception("Studio on_close handler failed")
        event.accept()

    def _place_default(self):
        """Dock to the right edge of the primary screen on first show."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        x = area.right() - self.width() - 20
        y = area.top() + (area.height() - self.height()) // 2
        self.move(max(area.left(), x), max(area.top(), y))

    def _refresh_buttons(self):
        has_text = bool(self.editor.toPlainText())
        self._clear_btn.setEnabled(has_text)
        self._copy_btn.setEnabled(has_text)
        self._cut_btn.setEnabled(has_text)
        # Undo is available if there's a committed step OR an uncommitted edit
        # in flight; redo only when there's a forward branch.
        pending = self.editor.toPlainText() != self._baseline
        self._undo_btn.setEnabled(bool(self._undo) or pending)
        self._redo_btn.setEnabled(bool(self._redo))
