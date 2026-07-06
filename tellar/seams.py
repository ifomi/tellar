"""Chunk-seam reconciliation for the chunked-transcription join.

When the VAD cuts mid-sentence (an intra-thought pause), the two chunks
straddling the cut are transcribed somewhat independently. Whisper's
decisions at the seam can disagree with the fact that it is ONE
continuous sentence. This module fixes the two CHEAP, language-agnostic
defect classes that the naive `" ".join` leaves behind:

  TERM — chunk N ends with a sentence terminator (.!?) it invented from
         the acoustic fade, while chunk N+1 — seeing N's tail as rolling
         prompt — correctly starts lowercase. The terminator is spurious.
         Fix: drop it (N+1 has more context, trust the continuation).

  DUP  — chunk N's last word reappears as chunk N+1's first word (the
         token is in the rolling prompt AND re-emitted at the next
         onset): "...происходит Происходит транскрибация".
         Fix: drop the duplicated leading word of N+1.

Seam CASING (a chunk starting with a false capital mid-sentence) is NO
longer handled here. A whitelist of "safe to lowercase" words cannot tell
a name from a sentence-start and does not scale to multiple languages;
that decision now lives in the seam-local P&C pass
(tellar/pnc.apply_seam_local), which lets the model decide per seam.

This module is dependency-light on purpose (only `re`) so both
pipeline.finalize() and the offline replay/validation tools can import
and apply the exact same TERM/DUP reconciliation.
"""

import re
from typing import List, Optional, Set

# A "word" — a run of Unicode letters, optionally hyphenated (so
# "mlx-whisper" / "по-моему" count as one word). Excludes digits and
# underscores so "v15" or stray markers don't masquerade as words.
_WORD = r"[^\W\d_]+(?:-[^\W\d_]+)*"
_FIRST_WORD_RE = re.compile(r"[\W\d_]*(" + _WORD + r")")
_WORD_RE = re.compile(_WORD)
_TERMINATORS = ".!?"
# Separators left dangling after a removed leading duplicate word.
_DANGLING = " \t,.;:!?—–-"

# Minimum word length for the DUP check — short function words ("и",
# "а", "в", "по") legitimately repeat across a pause ("...по по-моему")
# far more often than they are seam artifacts, so only treat words of
# 3+ letters as duplicates.
_MIN_DUP_LEN = 3


def _first_word(s: str) -> Optional[str]:
    m = _FIRST_WORD_RE.match(s)
    return m.group(1) if m else None


def _last_word(s: str) -> Optional[str]:
    words = _WORD_RE.findall(s)
    return words[-1] if words else None


def _strip_leading_word(s: str) -> str:
    """Remove the first word (and leading junk + trailing separators) from s."""
    m = _FIRST_WORD_RE.match(s)
    if not m:
        return s
    return s[m.end():].lstrip(_DANGLING)


def reconcile_seams(ordered: List[str], vocab: Optional[Set[str]] = None) -> List[str]:
    """Reconcile chunk-boundary defects (TERM + DUP) in a list of per-chunk
    texts. Returns a NEW list (does not mutate the input). Apply before
    joining with single spaces.

    `vocab` is accepted for backward compatibility with the diagnostic tools
    but is no longer used — seam casing is decided by the seam-local P&C pass.
    """
    out = list(ordered)
    for i in range(len(out) - 1):
        prev, curr = out[i], out[i + 1]
        if not prev or not curr:
            continue
        prev_s, curr_s = prev.rstrip(), curr.lstrip()
        if not prev_s or not curr_s:
            continue

        prev_last = _last_word(prev_s)
        curr_first = _first_word(curr_s)
        if not prev_last or not curr_first:
            continue
        prev_terminated = prev_s[-1] in _TERMINATORS

        # DUP — drop the duplicated leading word of the next chunk.
        if len(prev_last) >= _MIN_DUP_LEN and prev_last.lower() == curr_first.lower():
            curr_s = _strip_leading_word(curr_s)
            if not curr_s:
                out[i + 1] = curr_s
                continue
            curr_first = _first_word(curr_s)
            if not curr_first:
                out[i + 1] = curr_s
                continue

        # TERM — drop a spurious terminator when the next chunk continues in
        # lowercase (N+1 has more context; trust the continuation). A false
        # capital at the seam is handled downstream by seam-local P&C, not here.
        if prev_terminated and curr_s[0].islower():
            prev_s = prev_s[:-1].rstrip()

        out[i] = prev_s
        out[i + 1] = curr_s
    return out


def vocabulary_word_set() -> Set[str]:
    """Lowercased set of individual words across all vocabulary entries
    (multi-word phrases are split). Retained for the diagnostic tools
    (tools/scan_defects.py); no longer used by reconcile_seams. Never
    raises — empty set on any error."""
    try:
        from .vocabulary import read_vocabulary
        words: Set[str] = set()
        for entry in read_vocabulary():
            for w in _WORD_RE.findall(entry.lower()):
                words.add(w)
        return words
    except Exception:
        return set()
