"""Known-hallucination phrase filter for whisper output.

Whisper-large-v3-turbo's training data is heavy on YouTube/series
content where certain phrases mark "end of show", subtitle attribution
or sign-off cues. On long audio with mid-word cuts, unusual silence
patterns, or content that resembles "end of segment" acoustically,
the model emits these phrases as hallucinations — sometimes REPLACING
real content (the phrase appears WHERE the real content should be).

We can't prevent these at decode time without losing the benefits of
the prompt mechanism. Filter them post-hoc with a hand-curated
blocklist. False-positive cost (filtering a legitimate occurrence) is
low for our dictation use case — these phrases are very rare in real
speech.

Adding new phrases
------------------

When a new hallucination is observed in production logs, add the
phrase here. Match is case-insensitive and consumes trailing
sentence punctuation (.…!?) so we don't leave dangling dots after
removal. Whitespace around the match is collapsed to a single space.
"""

import re

from .logging_setup import get_logger

log = get_logger(__name__)


HALLUCINATION_PHRASES = [
    # Russian — series/segment end markers
    "Продолжение следует",
    "Продолжение в следующей серии",
    "Конец фильма",
    "Конец первой серии",
    # Russian — subtitle attribution that whisper sometimes inserts
    # (DimaTorzok is a real fansub author whose credits leaked into
    # whisper's training data; the model emits them on long Russian
    # audio especially when there's a sign-off-like pause)
    "Субтитры подготовил DimaTorzok",
    "Субтитры сделал DimaTorzok",
    "DimaTorzok",
    "Дима Торжок",
    "Корректор субтитров",
    # English — channel sign-offs / engagement hooks
    "Subscribe to my channel",
    "Subscribe to the channel",
    "Don't forget to subscribe",
    "Like and subscribe",
    "Thanks for watching",
    "Thank you for watching",
    "Bye-bye!",
    # English — subtitle attribution
    "Subtitles by the Amara.org community",
    "Captions by the Amara.org community",
    # Mixed — translations of segment markers we've seen leak through
    "To be continued",
]


# Compile once at import time. Match: optional leading whitespace,
# the phrase (any of), optional trailing sentence-ending punctuation.
# We replace with a single space so two real sentences flanking the
# match don't get glued together; the multi-space collapse below tidies up.
_PATTERN = re.compile(
    r'\s*(?:' + '|'.join(re.escape(p) for p in HALLUCINATION_PHRASES) + r')[.…!?]*',
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r'\s{2,}')


def remove_hallucinations(text: str) -> str:
    """Remove all known hallucination phrases from `text`. Whitespace
    around removed phrases is normalized so the cleaned output doesn't
    have visible double-gaps. Idempotent and order-independent for the
    phrases it matches.
    """
    cleaned, n = _PATTERN.subn(' ', text)
    if n:
        log.info("Removed %d known hallucination phrase(s)", n)
    cleaned = _MULTI_SPACE.sub(' ', cleaned)
    return cleaned.strip()
