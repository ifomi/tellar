"""Word-level diff highlighting for the Studio two-pane editor.

Computes which tokens in `before` were removed and which tokens in `after` were
added, then paints the corresponding character spans onto a QTextEdit through
QTextCharFormat backgrounds. The text is never modified — only its formatting.

The diff lives in two layers:

  * `diff_spans(before, after)` — pure: tokenises each text with a Unicode-aware
    regex (`\\w+` words, `\\S+` punctuation runs, `\\s+` whitespace runs), runs
    `difflib.SequenceMatcher` on the parallel token lists and returns the
    character ranges to highlight on each side. Whitespace-only tokens are
    deliberately not highlighted (a changed run of spaces alone is noise).
  * `paint_background` / `clear_background` — Qt helpers that mutate a
    QTextEdit's char formats. The caller is expected to suppress its own undo
    bookkeeping while these run (formatting changes can fire textChanged in
    Qt6); the Studio uses `_UndoController.apply_format()` for that.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import QTextEdit


# Soft, light-mode-friendly tints. Saturated reds/greens at full opacity drown
# out the text; these are pale enough that black glyphs stay legible while the
# delta is unmistakable at a glance.
DELETED_BG = QColor("#ffd6d6")
INSERTED_BG = QColor("#d6f5d6")


# Tokeniser. Three alternatives cover the input exhaustively (no characters
# fall through), so re-joining tokens in order reconstructs the source byte for
# byte and the (start, end) offsets are accurate. `\w` is Unicode-aware by
# default in `re`, so Cyrillic words tokenise the same way English ones do.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)


def _tokenize(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Split text into tokens and their absolute character spans.

    Returns parallel lists: `tokens[i]` is the token string, `spans[i]` is
    its (start, end) offset in `text`. Whitespace runs are kept as their own
    tokens so the alignment in SequenceMatcher can match them against
    whitespace on the other side rather than colliding with adjacent words.
    """
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for m in _TOKEN_RE.finditer(text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def diff_spans(
    before: str, after: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Compute character ranges to mark deleted (in `before`) and inserted
    (in `after`).

    Whitespace-only tokens are skipped on both sides — a changed run of
    spaces alone reads as noise; we only highlight the words/punctuation that
    actually changed. SequenceMatcher's `autojunk` heuristic is disabled
    because it ignores frequent tokens (common short words, spaces) and would
    skew the alignment for long inputs.
    """
    before_toks, before_spans = _tokenize(before)
    after_toks, after_spans = _tokenize(after)

    sm = SequenceMatcher(a=before_toks, b=after_toks, autojunk=False)
    deleted: list[tuple[int, int]] = []
    inserted: list[tuple[int, int]] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            for k in range(i1, i2):
                if not before_toks[k].isspace():
                    deleted.append(before_spans[k])
        if tag in ("replace", "insert"):
            for k in range(j1, j2):
                if not after_toks[k].isspace():
                    inserted.append(after_spans[k])

    return deleted, inserted


def paint_background(
    editor: QTextEdit, spans: list[tuple[int, int]], color: QColor
) -> None:
    """Paint a background colour over each (start, end) range in `editor`.

    Wrapped in a single edit block so the whole repaint is one undoable
    operation at the QTextDocument level (irrelevant to our custom undo
    controller — that one filters formatting via `_applying`, but Qt may
    still group it for its own bookkeeping).
    """
    if not spans:
        return
    fmt = QTextCharFormat()
    fmt.setBackground(color)
    cursor = QTextCursor(editor.document())
    cursor.beginEditBlock()
    for start, end in spans:
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(fmt)
    cursor.endEditBlock()


def clear_background(editor: QTextEdit) -> None:
    """Strip any background colour previously painted by `paint_background`.

    Sets a transparent brush over the whole document via mergeCharFormat
    rather than setCharFormat, so foreground colour, weight, italic, etc.
    survive intact (only the background attribute is overwritten). Cheap
    enough to run unconditionally — no need to track which spans were painted.
    """
    fmt = QTextCharFormat()
    fmt.setBackground(QColor(0, 0, 0, 0))  # transparent — clears the tint
    cursor = QTextCursor(editor.document())
    cursor.beginEditBlock()
    cursor.select(QTextCursor.SelectionType.Document)
    cursor.mergeCharFormat(fmt)
    cursor.endEditBlock()
