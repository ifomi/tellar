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
