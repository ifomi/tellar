"""Punctuation & Capitalization layer — variant C (full-replacement).

After chunks are stitched, whisper's per-chunk formatting carries seam defects
(false mid-sentence capitals, missing/extra periods, because each chunk is
formatted in isolation). This layer DISCARDS that formatting and re-derives
punctuation + case coherently over the WHOLE text with a trained P&C model, so
seam inconsistencies cannot exist by construction.

DEV build only: the model is loaded via the `punctuators` wrapper, which pulls
torch as a dependency. That is fine inside the venv but must NOT go into the
bundle — the production path will reimplement the ONNX + sentencepiece inference
without torch/punctuators.

Model: 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase, int8 quantized
(spike verdict 2026-07-03: ONNX/onnxruntime, ~266 MB, keeps words intact,
lowers wrong mid-sentence capitals; see Obsidian/Tellar/PnC Layer — предложение).

Safety contract: apply_pnc() NEVER changes the words themselves (guarded) and
NEVER raises into the caller — on any failure it returns the input unchanged,
so a dictation is never lost to a P&C problem.
"""
import glob
import os
import re
import threading

# The model is already cached (~/.cache/huggingface). Force offline at load:
# Apple corp net 403s HF, and we must not risk a network hang on the finalize
# path. Set before importing anything that touches huggingface_hub.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from .logging_setup import get_logger

log = get_logger(__name__)

_MODEL_ID = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
_model = None
_lock = threading.Lock()

_STRIP = re.compile(r"[^\w\s]", flags=re.UNICODE)


def _int8_path():
    """The quantized model produced during the spike, if present."""
    hits = glob.glob(
        os.path.expanduser("~/.cache/huggingface/**/model.int8.onnx"), recursive=True
    )
    return hits[0] if hits else None


def _load():
    from punctuators.models import PunctCapSegModelONNX
    import onnxruntime as ort

    m = PunctCapSegModelONNX.from_pretrained(_MODEL_ID)
    # Swap the fp32 ONNX session for the int8 one (4x smaller, ~same quality,
    # slightly faster) when the quantized file exists.
    int8 = _int8_path()
    if int8:
        m._ort_session = ort.InferenceSession(int8, providers=["CPUExecutionProvider"])
        log.info("P&C: loaded int8 model (%s)", int8)
    else:
        log.info("P&C: loaded fp32 model (int8 file not found)")
    return m


def _get():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = _load()
    return _model


def warm():
    """Load the model off the critical path — call in a daemon thread at
    startup so the first multi-chunk finalize does not stall a few seconds."""
    try:
        _get()
        log.info("P&C: model warmed")
    except Exception:
        log.exception("P&C: warm-up failed (will retry lazily on first use)")


def _lower_strip(text):
    """Variant C input: drop punctuation, lowercase → clean slate for the model."""
    return _STRIP.sub(" ", text).lower()


def _word_multiset(text):
    """Case/punct-insensitive sorted word list — for the word-integrity guard."""
    return sorted(_STRIP.sub(" ", text).lower().split())


def apply_pnc(text):
    """Re-punctuate and re-case `text` coherently. Returns the input unchanged
    on any failure or if the model would have altered the words themselves."""
    if not text or not text.strip():
        return text
    try:
        model = _get()
        out = model.infer([_lower_strip(text)])
        if not out or not isinstance(out[0], list):
            return text
        result = " ".join(out[0]).strip()
        if not result:
            return text
        # Guard: P&C may only add punctuation/case, never add/drop/alter words.
        if _word_multiset(result) != _word_multiset(text):
            log.warning("P&C word-integrity mismatch — keeping raw text")
            return text
        return result
    except Exception:
        log.exception("P&C failed — keeping raw text")
        return text


# --- Seam-local P&C -------------------------------------------------------
# Instead of reformatting the whole transcript (which damages the ~90% of text
# whisper already punctuated correctly INSIDE each chunk), fix ONLY the chunk
# boundaries. For each seam we ask the model, on a small lowercased window
# (tail of chunk A + head of chunk B), two things:
#   1. the casing of chunk B's first word in the model's output — so proper
#      nouns (Python, Tellar) stay capitalised because the MODEL decided so,
#      not a word list; a false sentence-cap (Перестали, Наш) gets lowered;
#   2. whether the model placed a terminator (. ? !) at the junction.
# We then set only chunk A's trailing punctuation and chunk B's first-letter
# case; every chunk interior stays exactly as whisper wrote it. No whitelist,
# language-agnostic, and the tiny windows never hit the 256-token limit.
_TAIL_WORDS, _HEAD_WORDS = 10, 6
_TERMS = ".?!"


def _set_first_letter(s, upper):
    for i, ch in enumerate(s):
        if ch.isalpha():
            return s[:i] + (ch.upper() if upper else ch.lower()) + s[i + 1:]
    return s


def _seam_decision(chunk_a, chunk_b):
    """(cap_b, term_char_or_None) for the A→B boundary, or (None, None) if the
    model output could not be aligned to the window (then leave the seam as-is)."""
    a = _lower_strip(chunk_a).split()[-_TAIL_WORDS:]
    b = _lower_strip(chunk_b).split()[:_HEAD_WORDS]
    if not a or not b:
        return None, None
    out = apply_pnc(" ".join(a + b))
    words = out.split()
    j = len(a)
    if len(words) != len(a) + len(b) or j >= len(words) or j == 0:
        return None, None
    b_first = words[j]
    cap_b = next((c.isupper() for c in b_first if c.isalpha()), False)
    prev = words[j - 1]
    term = prev[-1] if prev and prev[-1] in _TERMS else None
    return cap_b, term


def apply_seam_local(chunks):
    """Join `chunks` fixing only the boundaries (see block comment above).
    Never raises — falls back to a plain join on any error."""
    parts = [c for c in chunks if c and c.strip()]
    if len(parts) < 2:
        return " ".join(parts)
    try:
        out = list(parts)
        for i in range(len(out) - 1):
            cap_b, term = _seam_decision(out[i], out[i + 1])
            if cap_b is None:
                continue  # undecided → leave this seam untouched
            out[i] = out[i].rstrip().rstrip(".,;:!? ") + (term or "")
            out[i + 1] = _set_first_letter(out[i + 1].lstrip(), cap_b)
        return " ".join(out)
    except Exception:
        log.exception("seam-local P&C failed — plain join")
        return " ".join(parts)
