"""Tellar Studio — a two-pane editor that collects dictations and rewrites them.

When Studio mode is on (menu bar toggle), each dictation is appended into the
top pane instead of being pasted into the active app. The user edits it, runs a
preset (Polish for now) which rewrites the whole top pane through a local LLM
(studio_llm) into the bottom pane, then copies the finished result.

Layout — two panes in a vertical splitter:
  - top    = "было"  (source: dictations land here, hand-editable)
  - bottom = "стало" (result of the last transform, also editable)
The bottom pane is hidden until the first transform; once the split appears it
STAYS — so "before/after" is always visible — and only an explicit Close (×) on
the bottom pane folds it back to a single field. Clearing/cutting never auto-
collapses, and never crosses panes.

Controls:
  - Each pane has its OWN vertical button strip down its right side, acting on
    THAT pane only ([Copy] [Cut] [Clear]) — a top button never touches the
    bottom pane and vice versa. The bottom pane's strip also carries
    "Use as input ↑" (feed the result back up to chain another preset) and
    "Close ×" (fold the split back to a single field).
  - The top toolbar holds the controls common to both panes: Undo / Redo (they
    act on whichever pane has focus) and the preset button (Polish for now).

Undo/redo is managed by us per pane (_UndoController), not by QTextEdit's
built-in stack. The built-in stack merges a programmatic mutation (a dictation
append or a transform result) with the manual typing that follows it (positional
adjacency), so a single undo would wipe both. We disable it and keep our own
snapshot stack with explicit step boundaries: each programmatic change is one
step; a burst of manual typing coalesces into one step on a short pause.

Window shape: a normal, resizable top-level window. We deliberately avoid
Qt.WindowType.Tool (utility panels aren't first-class in Mission Control); the
app's accessory activation policy (app.py) is what keeps it out of the Dock and
⌘Tab, so the window type doesn't have to.
"""
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal
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
    QSplitter,
)

import threading

from . import studio_llm
from .logging_setup import get_logger

log = get_logger(__name__)

# How long manual typing stays "open" before it's committed as one undo step.
TYPING_COMMIT_MS = 350
# Cap the history so a very long session can't grow snapshots unbounded.
MAX_HISTORY = 200

# Toolbar icon buttons: square, small monochrome SF Symbol glyphs. Square (not
# wide rectangles) keeps the vertical per-pane strip looking like a clean column.
ICON_BTN_SIZE = 32
ICON_PT = 16

# Window frame margin, reused as the editor↔strip gap so the icon column is
# spaced evenly: window-edge→editor == editor→strip == strip→window-edge.
FRAME_MARGIN = 7

# macOS draws QPushButton with a rounded "lozenge" bezel that reads as a wide
# pill even at a fixed square size. Override it with a flat square style so the
# vertical strip is a tidy column of equal square icons.
_ICON_BTN_QSS = """
QPushButton {
    border: 1px solid #d6d6d6;
    border-radius: 6px;
    background: #ffffff;
}
QPushButton:hover:enabled { background: #f2f2f2; }
QPushButton:pressed:enabled { background: #e6e6e6; }
QPushButton:disabled { background: #f7f7f7; border-color: #ececec; }
"""

# QTextEdit defaults to a sunken bevelled frame, which renders thicker on the
# left and reads as an uneven gap next to the icon column. Replace it with a
# flat rounded border (matching the buttons / window) — crisp edge, rounded
# corners, even spacing.
_EDITOR_QSS = """
QTextEdit {
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    background: #ffffff;
    padding: 6px;
}
"""


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


def icon_button(symbol: str, fallback_text: str, tip: str) -> QPushButton:
    """A fixed-width toolbar button showing an SF Symbol icon (or fallback text).

    All Studio buttons are NoFocus: clicking one must NOT steal focus from the
    text pane, so the focused pane (which Undo/Redo act on) stays put.
    """
    btn = QPushButton()
    icon = sf_icon(symbol)
    if icon is not None:
        btn.setIcon(icon)
        btn.setIconSize(QSize(ICON_PT, ICON_PT))
    else:
        btn.setText(fallback_text)
    btn.setToolTip(tip)
    btn.setFixedSize(ICON_BTN_SIZE, ICON_BTN_SIZE)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setStyleSheet(_ICON_BTN_QSS)
    return btn


class _FocusTextEdit(QTextEdit):
    """A QTextEdit that announces when it gains focus, so the window can track
    which pane is active for the shared Undo/Redo buttons."""

    focused = pyqtSignal()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.focused.emit()


class _UndoController:
    """Custom undo/redo for one QTextEdit via a snapshot stack.

    Each programmatic change (dictation append, transform result, clear,
    use-as-input) is one undo step; a burst of manual typing coalesces into one
    step after TYPING_COMMIT_MS. `on_change` fires whenever undo/redo
    availability may have changed so the owner can refresh button states.
    """

    def __init__(self, editor: QTextEdit, on_change=None):
        self.editor = editor
        self._on_change = on_change
        # We run our own stack; turn off Qt's so it doesn't fight us and so ⌘Z
        # reaches our window shortcut instead of the editor's own handler.
        editor.setUndoRedoEnabled(False)
        self._undo: list[str] = []   # snapshots to step back to
        self._redo: list[str] = []   # snapshots to step forward to
        self._baseline = ""          # last committed text
        self._applying = False       # True while WE mutate the editor
        self._transient = False      # True while a non-committed placeholder shows
        self._timer = QTimer(editor)
        self._timer.setSingleShot(True)
        self._timer.setInterval(TYPING_COMMIT_MS)
        self._timer.timeout.connect(self.commit_pending)
        editor.textChanged.connect(self._on_text_changed)

    # --- programmatic mutations (each one undo step) -----------------------

    def append_line(self, text: str):
        """Append text on its own line as one step (dictation into the top)."""
        text = text.strip()
        if not text:
            return
        self.commit_pending()
        self._transient = False
        self._push_undo(self._baseline)
        self._redo.clear()
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
        self._changed()

    def replace_all(self, text: str):
        """Replace the whole content as one step (e.g. use-as-input)."""
        self.commit_pending()
        self._transient = False
        if self.editor.toPlainText() == text:
            return
        self._push_undo(self._baseline)
        self._redo.clear()
        self._set_raw(text)
        self._baseline = text
        self._changed()

    def set_result(self, text: str):
        """Commit a transform result as one undo step, discarding the transient
        placeholder. Assumes no pending manual edit (the pane was read-only
        during the run), so we deliberately do NOT commit_pending — that would
        push the 'Polishing…' placeholder as a bogus step."""
        self._timer.stop()
        self._transient = False
        self._push_undo(self._baseline)  # baseline = content before the run
        self._redo.clear()
        # A fresh result reads top-to-bottom, so land the view at the START
        # rather than scrolling to the end (which made a long result look empty
        # — you only saw its tail in a short pane).
        self._set_raw(text, to_end=False)
        self._baseline = text
        self._changed()

    def show_placeholder(self, text: str):
        """Show transient text (a 'Polishing…' note) without an undo step and
        without moving the baseline; the next set_result() undoes back to the
        content that preceded this placeholder."""
        self._transient = True
        self._set_raw(text)

    def clear(self):
        """Empty the editor as one undo step (so a clear can be undone)."""
        self.replace_all("")

    # --- manual typing -----------------------------------------------------

    def _on_text_changed(self):
        if self._applying:
            return
        # A manual edit invalidates the redo branch and (re)starts the coalesce
        # timer; the burst commits as one step after a short pause.
        self._transient = False
        self._redo.clear()
        self._timer.start()
        self._changed()

    def commit_pending(self):
        """Commit any uncommitted manual edits as a single undo step."""
        self._timer.stop()
        current = self.editor.toPlainText()
        if current != self._baseline:
            self._push_undo(self._baseline)
            self._baseline = current
            self._changed()

    # --- undo / redo -------------------------------------------------------

    def undo(self):
        self.commit_pending()  # in-progress typing becomes its own step first
        if not self._undo:
            return
        self._redo.append(self._baseline)
        target = self._undo.pop()
        self._set_raw(target)
        self._baseline = target
        self._changed()

    def redo(self):
        if not self._redo:
            return
        self._push_undo(self._baseline)
        target = self._redo.pop()
        self._set_raw(target)
        self._baseline = target
        self._changed()

    # --- helpers -----------------------------------------------------------

    def _push_undo(self, snapshot: str):
        self._undo.append(snapshot)
        if len(self._undo) > MAX_HISTORY:
            self._undo.pop(0)

    def _set_raw(self, text: str, to_end: bool = True):
        """Replace editor content programmatically (ignored by undo tracking).
        to_end=True lands the cursor/view at the end (latest dictation, edits);
        to_end=False lands it at the start (a fresh result reads top-down)."""
        self._applying = True
        self.editor.setPlainText(text)
        cursor = self.editor.textCursor()
        cursor.movePosition(
            QTextCursor.MoveOperation.End if to_end
            else QTextCursor.MoveOperation.Start)
        self.editor.setTextCursor(cursor)
        self._applying = False
        self.editor.ensureCursorVisible()

    def _changed(self):
        if self._on_change:
            self._on_change()

    @property
    def can_undo(self) -> bool:
        if self._transient:
            # A placeholder is showing; ignore the text/baseline mismatch.
            return bool(self._undo)
        return bool(self._undo) or self.editor.toPlainText() != self._baseline

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)


class _EditorPane(QWidget):
    """One editable text area with its own undo controller and a vertical strip
    of [Copy] [Cut] [Clear] buttons down its right side, acting on THIS pane
    only. Callers can pass extra_buttons that sit at the top of the strip
    (the bottom pane uses this for "Use as input ↑" and "Close ×")."""

    # A pane never crushes below this — the splitter can't squeeze a pane so
    # small its content vanishes with no room for a scrollbar.
    MIN_EDITOR_HEIGHT = 90

    def __init__(self, placeholder: str, on_focus, on_change, extra_buttons=()):
        super().__init__()

        self.editor = _FocusTextEdit()
        self.editor.setAcceptRichText(False)
        self.editor.setPlaceholderText(placeholder)
        # Wrap long lines and show a vertical scrollbar as soon as content
        # overflows — without this an over-tall pane could hide text with no
        # visible way to scroll to it.
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.editor.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.editor.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.editor.setMinimumHeight(self.MIN_EDITOR_HEIGHT)
        self.editor.setFrameShape(QTextEdit.Shape.NoFrame)  # QSS draws the border
        self.editor.setStyleSheet(_EDITOR_QSS)
        self.editor.focused.connect(lambda: on_focus(self))
        self.controller = _UndoController(self.editor, on_change=on_change)

        self._copy_btn = icon_button("doc.on.doc", "Copy",
                                     "Copy this pane to the clipboard")
        self._copy_btn.clicked.connect(self.copy)
        self._cut_btn = icon_button("scissors", "Cut",
                                     "Copy this pane to the clipboard and clear it")
        self._cut_btn.clicked.connect(self.cut)
        self._clear_btn = icon_button("trash", "Clear", "Clear this pane")
        self._clear_btn.clicked.connect(self.clear)

        # Vertical strip on the right, top-aligned: caller extras first, then
        # this pane's own edit actions.
        strip = QVBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(4)
        for b in extra_buttons:
            strip.addWidget(b)
        strip.addWidget(self._copy_btn)
        strip.addWidget(self._cut_btn)
        strip.addWidget(self._clear_btn)
        strip.addStretch(1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        # Match the window frame margin so the gap editor→strip equals the gap
        # strip→window edge (the strip sits evenly, not hugging either side).
        layout.setSpacing(FRAME_MARGIN)
        layout.addWidget(self.editor, 1)
        layout.addLayout(strip)
        self.refresh()

    # --- actions -----------------------------------------------------------

    def copy(self):
        text = self.editor.toPlainText()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        log.info("Studio: copied %d chars", len(text))

    def cut(self):
        """Copy this pane to the clipboard, then clear it — one click for
        'grab it and start fresh'. Affects this pane only."""
        text = self.editor.toPlainText()
        if not text:
            return
        QGuiApplication.clipboard().setText(text)
        log.info("Studio: cut %d chars", len(text))
        self.clear()

    def clear(self):
        """Clear this pane only (one undo step). Never touches the other pane
        and never collapses the split."""
        if not self.editor.toPlainText():
            return
        self.controller.clear()

    def append_line(self, text: str):
        self.controller.append_line(text)

    def refresh(self):
        has = bool(self.editor.toPlainText())
        self._clear_btn.setEnabled(has)
        self._cut_btn.setEnabled(has)
        self._copy_btn.setEnabled(has)


class StudioWindow(QWidget):
    # Opens short — a single dictation field (no result pane yet). On the first
    # transform the result pane appears and the window grows to TWO_PANE_HEIGHT
    # (grow-only — never shrinks a window the user already enlarged); after that
    # the user resizes freely.
    DEFAULT_WIDTH = 400
    DEFAULT_HEIGHT = 260
    TWO_PANE_HEIGHT = 620

    # Emitted from the transform worker thread back onto the Qt main thread
    # (cross-thread signals are delivered queued).
    _transform_done = pyqtSignal(str)
    _transform_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tellar Studio")
        # Called when the user closes the window via its title bar so the app
        # can keep the menu checkbox / Auto Paste in sync. Set by the owner.
        self.on_close = None
        # A normal top-level window so it participates in Mission Control and
        # can be raised by clicking its thumbnail there.
        self.setWindowFlags(Qt.WindowType.Window)
        self.setMinimumSize(320, 200)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

        self._transforming = False
        # Height the window had right before we grew it for the result pane;
        # restored on Close so collapsing frees the second pane's space.
        self._pre_grow_height = None

        # --- panes ---------------------------------------------------------
        self.top = _EditorPane(
            "Dictations land here while Studio is on.\n"
            "Edit freely, run a preset, then Copy.",
            on_focus=self._set_active,
            on_change=self._refresh,
        )
        # Bottom-pane-only controls: feed the result back up to chain another
        # preset, and explicitly fold the split back to a single field.
        self._use_btn = icon_button("arrow.up", "↑",
                                    "Use this result as the input above")
        self._use_btn.clicked.connect(self._use_as_input)
        self._close_btn = icon_button("xmark", "×",
                                      "Close the result pane (back to one field)")
        self._close_btn.clicked.connect(self._close_bottom)
        self.bottom = _EditorPane(
            "Transform result lands here.",
            on_focus=self._set_active,
            on_change=self._refresh,
            extra_buttons=(self._use_btn, self._close_btn),
        )
        self.bottom.hide()  # the split appears on the first transform
        self._active = self.top

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.top)
        split.addWidget(self.bottom)
        split.setChildrenCollapsible(False)
        self._split = split

        # --- window toolbar: shared undo/redo (focus-based) + presets ------
        self._undo_btn = icon_button("arrow.counterclockwise", "↺", "Undo (⌘Z)")
        self._undo_btn.clicked.connect(self._undo)
        self._redo_btn = icon_button("arrow.clockwise", "↻", "Redo (⌘⇧Z)")
        self._redo_btn.clicked.connect(self._redo)
        # Trash everything: a common action affecting BOTH panes, so it lives in
        # the top toolbar (not tied to one pane). Filled trash distinguishes it
        # from the outline per-pane Clear.
        self._clear_all_btn = icon_button("trash.fill", "Clr", "Clear both panes")
        self._clear_all_btn.clicked.connect(self._clear_all)

        # Phase 1 temporary control: a single hardcoded "Polish" preset that
        # proves the end-to-end slice. Phase 3 replaces this with the full
        # preset button row driven by studio_llm.PRESETS.
        self._polish_btn = QPushButton("Polish")
        self._polish_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._polish_btn.setToolTip("Rewrite the text above (LLM) into the pane below")
        self._polish_btn.clicked.connect(self._run_polish)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        toolbar.addWidget(self._undo_btn)
        toolbar.addWidget(self._redo_btn)
        toolbar.addWidget(self._clear_all_btn)
        toolbar.addSpacing(12)
        toolbar.addWidget(self._polish_btn)
        toolbar.addStretch(1)

        layout = QVBoxLayout(self)
        # Even frame: window margin == editor↔strip gap == strip↔edge gap, so
        # the icon column is spaced symmetrically (gap to the editor equals the
        # gap to the window border). Halved from the earlier wider frame.
        layout.setContentsMargins(FRAME_MARGIN, FRAME_MARGIN, FRAME_MARGIN, FRAME_MARGIN)
        layout.setSpacing(8)
        layout.addLayout(toolbar)
        layout.addWidget(split)

        self._transform_done.connect(self._on_transform_done)
        self._transform_failed.connect(self._on_transform_failed)

        # Qt's editor undo is disabled, so bind our own shortcuts (dispatch to
        # the focused pane).
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self._redo)

        self._refresh()

    # --- focus + shared undo/redo -----------------------------------------

    def _set_active(self, pane: _EditorPane):
        self._active = pane
        self._refresh()

    def _undo(self):
        self._active.controller.undo()

    def _redo(self):
        self._active.controller.redo()

    # --- dictation routing -------------------------------------------------

    def append_dictation(self, text: str):
        """Append a freshly transcribed chunk into the top pane. New dictation
        always lands in the source ("было"), never in the result."""
        self.top.append_line(text)
        log.info("Studio: appended dictation (%d chars)", len(text.strip()))

    # --- transform (Phase 1 temporary slice) -------------------------------

    def _run_polish(self):
        """Run the hardcoded Polish preset over the whole top pane → bottom.
        Blocking, so it runs on a worker thread; the result comes back via
        _transform_done on the main thread."""
        text = self.top.editor.toPlainText().strip()
        if not text or self._transforming:
            return
        self._transforming = True
        if self.bottom.isHidden():
            self.bottom.show()              # reveal the result pane
            self._grow_for_two_panes()      # grow the short window to fit two
            self._balance_split()
        self.bottom.editor.setReadOnly(True)
        self.bottom.controller.show_placeholder("Polishing…")
        self._refresh()
        threading.Thread(target=self._transform_worker, args=(text,), daemon=True).start()

    def _transform_worker(self, text: str):
        try:
            out = studio_llm.transform(text, studio_llm.POLISH)
            self._transform_done.emit(out)
        except Exception as e:
            log.exception("Studio transform failed")
            self._transform_failed.emit(str(e))

    def _on_transform_done(self, result: str):
        self._transforming = False
        self.bottom.editor.setReadOnly(False)
        self.bottom.controller.set_result(result)
        self._refresh()

    def _on_transform_failed(self, message: str):
        self._transforming = False
        self.bottom.editor.setReadOnly(False)
        self.bottom.controller.set_result(f"Transform failed: {message}")
        self._refresh()

    # --- chaining + explicit close ----------------------------------------

    def _use_as_input(self):
        """Feed the result back up so a second preset can run on it: replace
        the top with the bottom's text. We deliberately leave the bottom intact
        and keep the split open — "before/after" stays visible until the next
        run overwrites the bottom (or the user closes it)."""
        text = self.bottom.editor.toPlainText().strip()
        if not text or self._transforming:
            return
        self.top.controller.replace_all(text)
        self.top.editor.setFocus()
        self._set_active(self.top)

    def _close_bottom(self):
        """Explicitly fold the split back to a single field. The only thing
        that collapses the panes — clearing/cutting never does. Content is left
        as-is (hidden), so a later transform just overwrites it."""
        if self._transforming:
            return
        self.bottom.hide()
        if self._active is self.bottom:
            self._active = self.top
        # Undo the grow we did when the split appeared: shrink back to the
        # pre-split height so collapsing frees the second pane's space. Only if
        # we were the ones who grew it (a window the user enlarged is left be).
        if self._pre_grow_height is not None:
            self.resize(self.width(), self._pre_grow_height)
            self._pre_grow_height = None
        self._refresh()

    def _clear_all(self):
        """Trash everything: clear both panes at once. Each pane clears as its
        own undo step (focus-based Undo restores the focused pane). Like the
        per-pane Clear, this does NOT collapse the split — only Close (×) does."""
        if self._transforming:
            return
        self.top.controller.clear()
        self.bottom.controller.clear()
        self._refresh()

    def _grow_for_two_panes(self):
        """When the result pane first appears, grow the short single-pane window
        tall enough for two panes — but never shrink one the user enlarged.
        Remember the pre-grow height so Close can restore it."""
        if self.height() < self.TWO_PANE_HEIGHT:
            self._pre_grow_height = self.height()
            self.resize(self.width(), self.TWO_PANE_HEIGHT)

    def _balance_split(self):
        # Equal halves regardless of the exact pixel height — the splitter
        # normalises the ratio, so this is correct even right after a resize.
        self._split.setSizes([10_000, 10_000])

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
        restores Auto Paste. We hide (no WA_DeleteOnClose) so the content
        survives and comes back when Studio is re-enabled."""
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

    # --- button state ------------------------------------------------------

    def _refresh(self):
        # Shared undo/redo reflect the focused pane.
        c = self._active.controller
        self._undo_btn.setEnabled(c.can_undo)
        self._redo_btn.setEnabled(c.can_redo)
        # Polish runs the top pane; disabled while empty or mid-transform.
        self._polish_btn.setEnabled(
            bool(self.top.editor.toPlainText()) and not self._transforming)
        self._use_btn.setEnabled(
            bool(self.bottom.editor.toPlainText()) and not self._transforming)
        self._close_btn.setEnabled(not self._transforming)
        # Clear-all: enabled if either pane has anything to wipe.
        self._clear_all_btn.setEnabled(
            (bool(self.top.editor.toPlainText())
             or bool(self.bottom.editor.toPlainText()))
            and not self._transforming)
        self.top.refresh()
        self.bottom.refresh()
