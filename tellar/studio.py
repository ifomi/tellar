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
    QPlainTextEdit,
    QPushButton,
    QApplication,
    QSplitter,
)

import threading
from pathlib import Path

from . import diff_highlight, studio_llm
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

# Same square button, plus a clearly different background when checked so the
# diff toggle reads as ON at a glance. The accent blue echoes macOS native
# selection colour without leaning on a system palette query (which differs
# light/dark and would ship with no dark theme in v1 anyway).
_TOGGLE_BTN_QSS = _ICON_BTN_QSS + """
QPushButton:checked { background: #d8e9ff; border-color: #7aa7e0; }
QPushButton:checked:hover:enabled { background: #c8dffc; }
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

# Same border for the Custom prompt input (a QPlainTextEdit, not QTextEdit) so
# both fields visually match.
_PROMPT_QSS = """
QPlainTextEdit {
    border: 1px solid #d0d0d0;
    border-radius: 6px;
    background: #ffffff;
    padding: 6px;
}
"""

# Custom prompt history — bash-style recall of past instructions on ↑/↓.
# Lives outside the bundle so it survives rebuilds (same convention as
# vocabulary.txt and the transcription log).
CUSTOM_HISTORY_FILE = (
    Path.home() / "Library" / "Application Support" / "Tellar"
    / "custom_prompts_history.txt"
)
# Cap the history so it stays a useful "last few" stack rather than an
# archive — older entries silently fall off as new ones are added.
CUSTOM_HISTORY_CAP = 5


def _load_custom_history() -> list[str]:
    """Read persisted Custom prompts (newest first). Missing file → empty."""
    try:
        with open(CUSTOM_HISTORY_FILE, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.rstrip("\n")]
    except FileNotFoundError:
        return []
    except Exception:
        log.exception("Failed to read Custom history")
        return []


def _save_custom_history(entries: list[str]):
    """Write the history (newest first) capped at CUSTOM_HISTORY_CAP."""
    try:
        CUSTOM_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CUSTOM_HISTORY_FILE, "w", encoding="utf-8") as f:
            for line in entries[:CUSTOM_HISTORY_CAP]:
                f.write(line + "\n")
    except Exception:
        log.exception("Failed to save Custom history")


# Persisted toggle state for the Diff Highlighting button. One byte ("1"/"0")
# in the same Application Support directory as vocabulary.txt and the Custom
# history — outside the bundle so it survives rebuilds. First-launch default
# is ON: this is a comprehension feature; users discover it before they decide
# to silence it.
DIFF_SETTINGS_FILE = (
    Path.home() / "Library" / "Application Support" / "Tellar"
    / "studio_diff_enabled.txt"
)


def _load_diff_enabled() -> bool:
    try:
        return DIFF_SETTINGS_FILE.read_text().strip() == "1"
    except FileNotFoundError:
        return True
    except Exception:
        log.exception("Failed to read diff toggle state")
        return True


def _save_diff_enabled(enabled: bool):
    try:
        DIFF_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        DIFF_SETTINGS_FILE.write_text("1" if enabled else "0")
    except Exception:
        log.exception("Failed to save diff toggle state")


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


def icon_button(symbol: str, fallback_text: str, tip: str,
                checkable: bool = False) -> QPushButton:
    """A fixed-width toolbar button showing an SF Symbol icon (or fallback text).

    All Studio buttons are NoFocus: clicking one must NOT steal focus from the
    text pane, so the focused pane (which Undo/Redo act on) stays put. Pass
    `checkable=True` for a toggle button (e.g. the diff highlight switch); a
    different stylesheet draws an accent-blue background in the checked state.
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
    if checkable:
        btn.setCheckable(True)
        btn.setStyleSheet(_TOGGLE_BTN_QSS)
    else:
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
    `on_any_change` fires once per content change (manual OR programmatic) and
    is used by the Studio to recompute live diff highlights — purely
    formatting changes (the diff paint itself) are filtered out via
    `_painting` so the recompute can't loop.
    """

    def __init__(self, editor: QTextEdit, on_change=None, on_any_change=None):
        self.editor = editor
        self._on_change = on_change
        self._on_any_change = on_any_change
        # We run our own stack; turn off Qt's so it doesn't fight us and so ⌘Z
        # reaches our window shortcut instead of the editor's own handler.
        editor.setUndoRedoEnabled(False)
        self._undo: list[str] = []   # snapshots to step back to
        self._redo: list[str] = []   # snapshots to step forward to
        self._baseline = ""          # last committed text
        self._applying = False       # True while WE mutate the editor's content
        # True while WE merge a char-format (diff highlight paint). Qt6 fires
        # textChanged for format-only edits too, so without this guard a paint
        # would notify on_any_change and trigger another recompute → repaint
        # → recompute …  Filter format-only changes out entirely.
        self._painting = False
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
        # Format-only edits (a diff highlight repaint) must not trigger anything
        # — they don't change the text, and notifying on_any_change here would
        # loop right back into another recompute/repaint.
        if self._painting:
            return
        if self._applying:
            # Programmatic content change (set_result, append_line, replace_all).
            # Don't touch the user-edit bookkeeping, but DO notify the live diff
            # so it sees the new content and can recompute against it.
            if self._on_any_change:
                self._on_any_change()
            return
        # A manual edit invalidates the redo branch and (re)starts the coalesce
        # timer; the burst commits as one step after a short pause.
        self._transient = False
        self._redo.clear()
        self._timer.start()
        self._changed()
        if self._on_any_change:
            self._on_any_change()

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

    def apply_format(self, fn):
        """Run a format-only mutation (e.g. paint diff highlights) without
        triggering any of the change-tracking machinery. Suspends both the
        manual-edit and programmatic-change paths so the repaint stays a
        purely visual operation."""
        self._painting = True
        try:
            fn(self.editor)
        finally:
            self._painting = False

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

    def __init__(self, placeholder: str, on_focus, on_change,
                 on_any_change=None, extra_buttons=()):
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
        self.controller = _UndoController(
            self.editor, on_change=on_change, on_any_change=on_any_change,
        )

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

        # Set the pane's minimum height to what the button strip ACTUALLY
        # needs (each fixed-size button + 4px spacing between them) so the
        # QSplitter can't squeeze the bottom pane — which has more buttons —
        # below the height that fits its column. Without this, an even
        # 50/50 split could clip the lower buttons (Use/Close/Copy/Cut/Clear
        # don't all fit in ~150px).
        n_buttons = 3 + len(extra_buttons)  # 3 standard + caller's extras
        strip_min = (
            n_buttons * ICON_BTN_SIZE
            + max(0, n_buttons - 1) * 4
        )
        self.setMinimumHeight(max(strip_min, self.MIN_EDITOR_HEIGHT))

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


class _CustomPromptEdit(QPlainTextEdit):
    """Multi-line input for the Custom preset's free-form instruction.

    Two non-default behaviours layered on QPlainTextEdit:

      ⌘↵ submits — emits `submit`. Plain Enter still inserts a newline (the
      field is multi-line because real instructions can be long).

      ↑/↓ recall — bash-style. ↑ pressed when the cursor is at the very start
      of the text walks back through history (older entries); ↓ at the very
      end walks forward (newer); going past the newest restores the user's
      pre-recall draft. Anywhere in the middle, ↑/↓ move the cursor as usual.

    History is the persistent CUSTOM_HISTORY_FILE list; navigation never
    mutates it. Submitting calls remember(), which prepends the instruction
    and rewrites the file.
    """

    submit = pyqtSignal()
    escape = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._history: list[str] = _load_custom_history()
        # -1 means "the user's current draft" (not pointing at a history entry).
        self._history_idx: int = -1
        # Stash of the user's text the moment they first hit ↑, so ↓ past the
        # newest entry can restore exactly what they had typed.
        self._draft: str = ""

    def remember(self, instruction: str):
        """Persist a just-submitted instruction at the top of history."""
        instruction = instruction.strip()
        if not instruction:
            return
        # Dedupe: if it's already at the top, nothing to do; otherwise pull
        # any older copy out and re-insert at position 0.
        if self._history and self._history[0] == instruction:
            return
        self._history = [instruction] + [
            h for h in self._history if h != instruction
        ]
        _save_custom_history(self._history)
        self._history_idx = -1
        self._draft = ""

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        # ⌘↵ (or Ctrl+↵) — submit. Plain Enter still inserts a newline.
        is_submit_modifier = bool(mods & (
            Qt.KeyboardModifier.MetaModifier
            | Qt.KeyboardModifier.ControlModifier
        ))
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and is_submit_modifier:
            self.submit.emit()
            return

        # On macOS, the arrow keys arrive with Qt.KeyboardModifier.KeypadModifier
        # set, so a plain `not mods` test fails for them. Only treat the
        # "active" modifiers as blocking; the keypad flag is innocent
        # context and must be ignored. Same trap for Esc on some layouts.
        active_mods = (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.MetaModifier
            | Qt.KeyboardModifier.AltModifier
        )
        plain = not (mods & active_mods)

        # Esc — close the row (StudioWindow toggles us off and returns focus
        # to the top pane). Done via a signal so the field knows nothing
        # about the surrounding window layout. Plain Esc only — Ctrl+Esc is
        # the global "cancel recording" hotkey and must not also close us.
        if key == Qt.Key.Key_Escape and plain:
            self.escape.emit()
            return

        cursor = self.textCursor()

        # ↑ — at the first line of the doc, recall the previous (older)
        # history entry. The bash trick: we check the line (block) the
        # cursor is in, NOT its position-in-line. On a recalled single-line
        # entry the cursor lands at end-of-line, but it's still on block 0,
        # so the next ↑ keeps walking back through history. On multi-line
        # text, ↑ first walks the cursor up through lines and only paginates
        # history once it reaches the first line — same as bash.
        if (key == Qt.Key.Key_Up and plain
                and cursor.blockNumber() == 0
                and self._history_idx + 1 < len(self._history)):
            if self._history_idx == -1:
                # First step into history — stash the draft for restore.
                self._draft = self.toPlainText()
            self._history_idx += 1
            self._set_text(self._history[self._history_idx])
            return

        # ↓ — at the last line, recall the next (newer) entry, or restore
        # the stashed draft when stepping past the newest. Only intercepts
        # while we're already navigating (history_idx >= 0); otherwise the
        # default cursor-down behaviour applies.
        last_block = self.document().blockCount() - 1
        if (key == Qt.Key.Key_Down and plain
                and cursor.blockNumber() == last_block
                and self._history_idx >= 0):
            if self._history_idx > 0:
                self._history_idx -= 1
                self._set_text(self._history[self._history_idx])
            else:
                self._history_idx = -1
                self._set_text(self._draft)
            return

        super().keyPressEvent(event)

    def _set_text(self, text: str):
        """Replace content and land the cursor at the end (so the next ↓ at
        end of doc is correctly detected, and so a recalled instruction is
        ready to be edited at its tail)."""
        self.setPlainText(text)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)


class StudioWindow(QWidget):
    # Opens with the toolbar, the top pane, and the always-on prompt row at
    # the bottom; the result pane appears on the first transform and grows
    # the window to TWO_PANE_HEIGHT (grow-only — never shrinks one the user
    # already enlarged). Heights are sized so the top pane has a comfortable
    # ~150 px before any user resize, plus the prompt row's fixed 60 px and
    # the toolbar.
    DEFAULT_WIDTH = 400
    DEFAULT_HEIGHT = 330
    TWO_PANE_HEIGHT = 690

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
        self.setMinimumSize(320, 260)
        self.resize(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT)

        self._transforming = False
        # Height the window had right before we grew it for the result pane;
        # restored on Close so collapsing frees the second pane's space.
        self._pre_grow_height = None

        # --- live diff highlighting ---------------------------------------
        # Persisted toggle state (default ON). When on, the bottom pane's
        # additions get a green tint and the top pane's deletions get a red
        # tint, recomputed against the current text whenever either pane
        # changes — manual edits, dictation appends, transform results, all
        # take the same path.
        self._diff_enabled = _load_diff_enabled()
        # `show_diff` from the last preset that ran. Translate-style presets
        # produce wholly different text where every token reads as "changed",
        # so they opt out of highlighting via Preset.show_diff = False; we
        # remember the choice across the transform's async boundary.
        self._last_show_diff = True
        # Debounce timer: every textChanged restarts it; on timeout we
        # recompute. Without this, a long manual paragraph would re-tokenise
        # and re-paint on every keystroke. 150ms is short enough to feel
        # immediate yet long enough to coalesce a fast typist's bursts.
        self._diff_timer = QTimer(self)
        self._diff_timer.setSingleShot(True)
        self._diff_timer.setInterval(150)
        self._diff_timer.timeout.connect(self._recompute_diff_now)

        # --- panes ---------------------------------------------------------
        self.top = _EditorPane(
            "Dictations land here while Studio is on.\n"
            "Edit freely, run a preset, then Copy.",
            on_focus=self._set_active,
            on_change=self._refresh,
            on_any_change=self._request_diff_recompute,
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
            on_any_change=self._request_diff_recompute,
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
        # Diff highlighting toggle. Sits with the other pane-wide controls
        # because it acts on both panes at once. The eye glyph reads as
        # "show me what changed"; the QSS gives the checked state an accent
        # background so the on/off state is obvious.
        self._diff_btn = icon_button(
            "eye", "Diff",
            "Highlight differences between the panes (⌘D)",
            checkable=True,
        )
        self._diff_btn.setChecked(self._diff_enabled)
        self._diff_btn.toggled.connect(self._on_diff_toggled)

        # The single fixed preset for now (Phase 1). Custom prompt lives in
        # its own always-on row at the bottom of the window — there's no
        # toggle; the field and the pill coexist as two equally-available
        # entry points. The sparkles glyph mirrors what Apple Intelligence
        # uses for its Writing Tools (Polish / Rewrite / Make Friendly), so
        # it reads as "AI tidies up the text" at a glance.
        self._polish_btn = icon_button(
            "sparkles", "✨",
            "Polish — rewrite the text above (LLM) into the pane below",
        )
        self._polish_btn.clicked.connect(self._run_polish)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)
        toolbar.addWidget(self._undo_btn)
        toolbar.addWidget(self._redo_btn)
        toolbar.addWidget(self._clear_all_btn)
        toolbar.addWidget(self._diff_btn)
        toolbar.addWidget(self._polish_btn)
        toolbar.addStretch(1)

        # --- always-on Custom prompt row -----------------------------------
        # The prompt field is permanent: no toggle, no reveal animation, no
        # mode switch. It coexists with the preset pills above as the second
        # always-available entry point ("type and ⌘↵" vs "click a pill").
        # Dictation routes here whenever this field has focus; otherwise it
        # lands in the top pane (see append_dictation).
        self._custom_input = _CustomPromptEdit()
        self._custom_input.setPlaceholderText(
            "Prompt — ⌘↵ to run, ↑/↓ history"
        )
        # Height fits exactly two stacked icon buttons in the right-side
        # strip (Apply on top, Clear below): 2 × ICON_BTN_SIZE + 4 spacing.
        # Long instructions scroll internally rather than ballooning the row.
        self._custom_input.setFixedHeight(2 * ICON_BTN_SIZE + 4)
        self._custom_input.setFrameShape(QPlainTextEdit.Shape.NoFrame)
        self._custom_input.setStyleSheet(_PROMPT_QSS)
        self._custom_input.submit.connect(self._run_custom)
        # Esc no longer closes the row (it's permanent now); it just bounces
        # focus back to the top pane so the user can keep typing/dictating
        # there without grabbing the mouse.
        self._custom_input.escape.connect(self._focus_top)
        # Apply enables only when both source text and instruction are non-
        # empty — refresh on every keystroke so the button state tracks the
        # field live. The same refresh drives the prompt-clear button below.
        self._custom_input.textChanged.connect(self._refresh)

        self._apply_btn = icon_button(
            "play.fill", "▶", "Apply the Custom instruction (⌘↵)"
        )
        self._apply_btn.clicked.connect(self._run_custom)
        # Local "clear this prompt" — outline trash to match the per-pane
        # Clear glyph (filled trash stays the global clear-all). Enabled
        # only when there's something to wipe; clicking never grabs focus
        # (NoFocus on the button), so the keyboard caret stays where it was.
        self._clear_prompt_btn = icon_button(
            "trash", "Clr", "Clear the prompt field"
        )
        self._clear_prompt_btn.clicked.connect(self._custom_input.clear)

        self._custom_row = QWidget()
        custom_row_layout = QHBoxLayout(self._custom_row)
        custom_row_layout.setContentsMargins(0, 0, 0, 0)
        # Match the panes' editor↔strip spacing exactly so the prompt row's
        # input + button column lines up with the panes' editor + strip
        # column above — same FRAME_MARGIN gap, same 32 px button column on
        # the right edge, same right margin to the window border.
        custom_row_layout.setSpacing(FRAME_MARGIN)
        custom_row_layout.addWidget(self._custom_input, 1)
        # Vertical strip on the right: Apply on top, Clear below. Same 4 px
        # spacing as the per-pane strips so the visual rhythm matches across
        # the window (every right-edge button column reads as the same shape).
        prompt_strip = QVBoxLayout()
        prompt_strip.setContentsMargins(0, 0, 0, 0)
        prompt_strip.setSpacing(4)
        prompt_strip.addWidget(self._apply_btn)
        prompt_strip.addWidget(self._clear_prompt_btn)
        custom_row_layout.addLayout(prompt_strip)

        layout = QVBoxLayout(self)
        # Even frame: window margin == editor↔strip gap == strip↔edge gap, so
        # the icon column is spaced symmetrically (gap to the editor equals the
        # gap to the window border). Halved from the earlier wider frame.
        layout.setContentsMargins(FRAME_MARGIN, FRAME_MARGIN, FRAME_MARGIN, FRAME_MARGIN)
        layout.setSpacing(8)
        layout.addLayout(toolbar)
        # Prompt row sits directly under the toolbar — Polish (a pill) and
        # Custom (this field) are two equally-available entry points to the
        # same transform, so they share the top of the window. Putting the
        # field below the panes (chat-style) visually divorced it from the
        # pill above, hiding the unification.
        layout.addWidget(self._custom_row)
        # stretch=1 — the splitter must absorb ALL extra vertical space so
        # the panes grow with the window. Without this stretch, QVBoxLayout
        # only gives the splitter its sizeHint (≈ sum of pane minimums) and
        # the rest stays as empty space below.
        layout.addWidget(split, 1)

        self._transform_done.connect(self._on_transform_done)
        self._transform_failed.connect(self._on_transform_failed)

        # Qt's editor undo is disabled, so bind our own shortcuts (dispatch to
        # the focused pane).
        QShortcut(QKeySequence.StandardKey.Undo, self, activated=self._undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, activated=self._redo)
        # ⌘D toggles the diff highlighting. Calling .toggle() through the
        # button (not setting `_diff_enabled` directly) routes through the
        # toggled() signal so the checkmark, persistence, and recompute all
        # happen on one path.
        QShortcut(QKeySequence("Ctrl+D"), self, activated=self._diff_btn.toggle)

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
        """Append a freshly transcribed chunk into whichever field is active.
        Dictation lands in the Custom prompt field if it has focus (so the
        user can dictate the instruction itself); otherwise it goes to the
        top pane (the default flow)."""
        if self._custom_input.hasFocus():
            self._append_to_custom(text)
            log.info("Studio: appended dictation to Custom (%d chars)",
                     len(text.strip()))
        else:
            self.top.append_line(text)
            log.info("Studio: appended dictation (%d chars)", len(text.strip()))

    def _append_to_custom(self, text: str):
        """Insert a dictation chunk at the end of the Custom prompt field,
        joined to existing content with a single space (so multi-chunk
        dictations read as one continuous instruction)."""
        text = text.strip()
        if not text:
            return
        cursor = self._custom_input.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if self._custom_input.toPlainText():
            cursor.insertText(" " + text)
        else:
            cursor.insertText(text)
        self._custom_input.setTextCursor(cursor)
        self._custom_input.ensureCursorVisible()

    def _focus_top(self):
        """Move keyboard focus to the top pane. Used by Esc inside the
        prompt field to bounce back without grabbing the mouse."""
        self.top.editor.setFocus()

    # --- transform (Phase 1 temporary slice) -------------------------------

    def _start_transform(self, placeholder: str, show_diff: bool = True):
        """Common pre-transform setup shared by Polish and Custom: reveal the
        result pane, lock it, show a placeholder, refresh button states. The
        worker thread + signal plumbing is per-transform. `show_diff` is
        latched here so the live diff respects translate-style presets across
        the worker's async boundary (Preset.show_diff = False)."""
        self._transforming = True
        self._last_show_diff = show_diff
        if self.bottom.isHidden():
            self.bottom.show()              # reveal the result pane
            self._grow_for_two_panes()      # grow the short window to fit two
            self._balance_split()
        self.bottom.editor.setReadOnly(True)
        self.bottom.controller.show_placeholder(placeholder)
        self._refresh()

    def _run_polish(self):
        """Run the hardcoded Polish preset over the whole top pane → bottom.
        Blocking, so it runs on a worker thread; the result comes back via
        _transform_done on the main thread."""
        text = self.top.editor.toPlainText().strip()
        if not text or self._transforming:
            return
        self._start_transform("Polishing…", show_diff=studio_llm.POLISH.show_diff)
        threading.Thread(
            target=self._polish_worker, args=(text,), daemon=True
        ).start()

    def _polish_worker(self, text: str):
        try:
            out = studio_llm.transform(text, studio_llm.POLISH)
            self._transform_done.emit(out)
        except Exception as e:
            log.exception("Studio Polish failed")
            self._transform_failed.emit(str(e))

    def _run_custom(self):
        """Apply the Custom instruction (free-form prompt) to the top pane.
        Both the source text and the instruction must be non-empty; otherwise
        nothing happens (the Apply button is disabled in that state anyway,
        but ⌘↵ from the keyboard reaches us regardless)."""
        text = self.top.editor.toPlainText().strip()
        instruction = self._custom_input.toPlainText().strip()
        if not text or not instruction or self._transforming:
            return
        # Persist the instruction at the top of history before we kick off
        # the worker; if the run later fails, the instruction is still
        # recallable for a retry.
        self._custom_input.remember(instruction)
        self._custom_input.clear()
        # Custom is the user-driven escape hatch. We can't introspect what
        # they typed (translate? polish? rephrase?), so we leave the diff on
        # by default — if it's noisy for their intent, they toggle it off.
        self._start_transform("Applying…", show_diff=True)
        threading.Thread(
            target=self._custom_worker, args=(text, instruction), daemon=True
        ).start()

    def _custom_worker(self, text: str, instruction: str):
        try:
            out = studio_llm.transform_custom(text, instruction)
            self._transform_done.emit(out)
        except Exception as e:
            log.exception("Studio Custom failed")
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

    # --- live diff highlighting -------------------------------------------

    def _on_diff_toggled(self, checked: bool):
        """Slot for the toolbar toggle (and ⌘D shortcut, which calls toggle()
        on the same button). Persists the new state, then either repaints to
        reflect the current panes or clears any standing highlights."""
        self._diff_enabled = checked
        _save_diff_enabled(checked)
        if checked:
            # Don't wait the debounce — the user just asked to see the diff,
            # show it immediately.
            self._recompute_diff_now()
        else:
            self._clear_diff_highlights()
        log.info("Studio: diff highlighting %s", "on" if checked else "off")

    def _request_diff_recompute(self):
        """Schedule a diff repaint after a short debounce. Called from both
        panes' on_any_change so manual typing, programmatic appends, and
        transform results all converge on the same path. Cheap when nothing's
        to do — the heavy work happens in _recompute_diff_now after the
        timer fires."""
        if not self._diff_enabled:
            return
        # Skip while a placeholder ("Polishing…") is showing — the user
        # would otherwise see the placeholder text get diff-highlighted
        # against the real top pane for one frame.
        if self._transforming:
            return
        if not self._last_show_diff:
            return
        self._diff_timer.start()

    def _recompute_diff_now(self):
        """Repaint the live diff between the two panes against their CURRENT
        text. Fully idempotent — clears the old highlights first, then paints
        the fresh ones. Skipped when the comparison would be meaningless
        (single-pane mode, toggle off, translate-style preset)."""
        if not self._diff_enabled or self._transforming or not self._last_show_diff:
            return
        if self.bottom.isHidden():
            return
        before = self.top.editor.toPlainText()
        after = self.bottom.editor.toPlainText()
        self._apply_diff_highlights(before, after)

    def _apply_diff_highlights(self, before: str, after: str):
        """Compute spans and paint them. Always clears first so a shrinking
        diff (e.g. user just edited the panes closer together) doesn't leave
        stale tints behind."""
        self._clear_diff_highlights()
        if not before or not after:
            return
        deleted, inserted = diff_highlight.diff_spans(before, after)
        if deleted:
            self.top.controller.apply_format(
                lambda e: diff_highlight.paint_background(
                    e, deleted, diff_highlight.DELETED_BG
                )
            )
        if inserted:
            self.bottom.controller.apply_format(
                lambda e: diff_highlight.paint_background(
                    e, inserted, diff_highlight.INSERTED_BG
                )
            )

    def _clear_diff_highlights(self):
        """Remove any background tints from both panes. Wrapped in
        apply_format so the format-only edits don't trigger another
        recompute (loop guard)."""
        self.top.controller.apply_format(diff_highlight.clear_background)
        self.bottom.controller.apply_format(diff_highlight.clear_background)

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
        # Strip diff tints from BOTH panes — comparison is meaningless with
        # one pane, and an orphan red tint on the still-visible top would be
        # confusing without its green counterpart on screen.
        self._clear_diff_highlights()
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
        # Apply (Custom) needs BOTH a source text and a typed instruction.
        # While transforming, lock it like Polish.
        self._apply_btn.setEnabled(
            bool(self.top.editor.toPlainText())
            and bool(self._custom_input.toPlainText().strip())
            and not self._transforming)
        # Prompt-clear is enabled only when there's something to wipe;
        # never during a transform (the prompt is already cleared after a
        # run and re-enabling the button mid-run reads as a no-op).
        self._clear_prompt_btn.setEnabled(
            bool(self._custom_input.toPlainText())
            and not self._transforming)
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
