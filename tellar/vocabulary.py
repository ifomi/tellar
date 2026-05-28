"""Custom vocabulary for whisper prompt biasing.

The user maintains a plain-text list of names, project terms, and proper
nouns they want whisper to recognize correctly. The contents are
appended to whisper's initial_prompt at transcribe time, biasing the
decoder toward these spellings when the audio sounds similar.

The file lives at ~/Library/Application Support/Tellar/vocabulary.txt
and the user edits it via "Edit Vocabulary…" in the menubar dropdown.
Re-read on every transcription — files are tiny, no caching needed.

Limitations
-----------

Prompt biasing is statistical, not substitutional. It works well for
proper nouns spoken in the same language as the input audio (e.g.
"Tellar" while speaking English will reliably come out as "Tellar"
instead of "Teller"). It does NOT cleanly handle cross-language
substitution (e.g. saying "хок" in Russian and wanting "HAWK" in the
English-letter output) — Russian audio biases the decoder toward
Cyrillic tokens regardless of the prompt language. Cross-language
brand-name substitution belongs to a separate post-process replacement
layer, intentionally out of scope here.
"""
from pathlib import Path
from typing import List

from .logging_setup import get_logger

log = get_logger(__name__)


VOCAB_PATH = (
    Path.home() / "Library" / "Application Support" / "Tellar" / "vocabulary.txt"
)

VOCAB_TEMPLATE = """\
# Tellar Custom Vocabulary
#
# One word or phrase per line. Helps Whisper recognize specific names
# you use often — project names, colleague names, technical terms.
#
# Whisper will be biased toward these spellings when the audio sounds
# similar. Works best for proper nouns in the same language you're
# speaking (e.g. "Tellar" while speaking English).
#
# Lines starting with # are comments and ignored.
# Empty lines are ignored. Multi-word phrases on one line are OK.
#
# Examples (uncomment to use):
# Tellar
# mlx-whisper
# Anand Undavia
"""


def ensure_file_exists() -> None:
    """Create the vocabulary file with a commented template if it doesn't
    exist yet. Idempotent — safe to call on every app start and before
    every "Edit Vocabulary…" click."""
    VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not VOCAB_PATH.exists():
        VOCAB_PATH.write_text(VOCAB_TEMPLATE, encoding="utf-8")
        log.info("Created vocabulary template at %s", VOCAB_PATH)


def read_vocabulary() -> List[str]:
    """Return the list of vocabulary entries, in file order, with comments
    and blank lines stripped. Returns [] if the file is missing, empty,
    or contains only comments. Never raises — read errors are logged
    and treated as empty vocabulary so transcription always proceeds."""
    if not VOCAB_PATH.exists():
        return []
    try:
        lines = VOCAB_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        log.exception("Failed to read vocabulary file %s", VOCAB_PATH)
        return []
    out: List[str] = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def format_vocabulary_suffix() -> str:
    """Build the prompt suffix to append after PUNCTUATION_PROMPT.

    Returns an empty string when the vocabulary is empty so callers
    don't have to special-case it — they can always concatenate.

    Format: each entry becomes its own period-terminated "sentence"
    ('Tellar. mlx-whisper. Anand Undavia.'). The trailing period after
    each entry is what carries the bias: whisper treats a token at the
    start of a sentence as a likely proper noun, which boosts the
    probability of the spelling we provided over generic alternatives
    (e.g. 'Tellar' over 'Taylor', 'Teller'). Earlier 'Names: a, b, c'
    label form was tested first but proved too weak — Whisper's prior
    on common English names easily outweighed a single comma-list.

    The format is generic (no per-word context like 'Working with X')
    so it works for any vocabulary entry — proper nouns, technical
    terms, multi-word phrases — without hardcoded scaffolding."""
    words = read_vocabulary()
    if not words:
        return ""
    return ". ".join(words) + "."
