"""Punctuation & Capitalization layer — seam-local, torch-free.

Runs the P&C ONNX model DIRECTLY on onnxruntime + sentencepiece (+ numpy) — NO
torch, NO `punctuators` package — so it ships in the bundle, which deliberately
has no torch. This reimplements what `punctuators.PunctCapSegModelONNX` does
(sentencepiece tokenize → window ≤256 tok with overlap → ONNX run → decode the
pre/post/cap heads into text); verified against it in tools/pnc_decode_probe.py.

Model: 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase, int8 quantized.

Architecture: SEAM-LOCAL — we do not reformat the whole transcript (that damages
text whisper already got right). For each chunk boundary we run the model on a
small lowercased window and apply only the terminator + first-letter-case
decision, keeping chunk interiors verbatim. No whitelist, language-agnostic.

Safety: never raises into the caller and never changes the words themselves
(word-integrity guard) — on any failure it returns the input unchanged, so a
dictation is never lost to a P&C problem.
"""
import glob
import os
import re
import threading

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

# Label space + window size are FIXED for this model (from its config.yaml).
# Hardcoded so the bundle depends on neither pyyaml nor a shipped config.yaml.
_MAX_LEN = 256
_PRE_LABELS = ["<NULL>", "¿"]
_POST_LABELS = ["<NULL>", "<ACRONYM>", ".", ",", "?", "？", "，", "。", "、",
                "・", "।", "؟", "،", ";", "።", "፣", "፧"]
_NULL = "<NULL>"
_ACRONYM = "<ACRONYM>"
_OVERLAP = 16

# The P&C model ships INSIDE the bundle (tellar/assets/pnc/): every multi-chunk
# dictation uses it, so every user needs it; and it is NOT on HuggingFace (we
# quantized the int8 locally), so bundling beats self-hosting + a download path.
# Dev falls back to the HF cache. (gemma/whisper stay download-on-demand — those
# are opt-in / huge and already on HF.)
_BUNDLED = os.path.join(os.path.dirname(__file__), "assets", "pnc")
_HF_CACHE = os.path.expanduser("~/.cache/huggingface")

_sp = None
_sess = None
_lock = threading.Lock()

_STRIP = re.compile(r"[^\w\s]", flags=re.UNICODE)
_TAIL_WORDS, _HEAD_WORDS = 10, 6
_TERMS = ".?!"


def _find(*names):
    for root in (_BUNDLED, _HF_CACHE):
        for name in names:
            hits = glob.glob(os.path.join(root, "**", name), recursive=True)
            if hits:
                return hits[0]
    return None


def _load():
    global _sp, _sess
    import onnxruntime as ort
    from sentencepiece import SentencePieceProcessor

    onnx = _find("model.int8.onnx", "model.onnx")
    spe = _find("sp.model")
    if not onnx or not spe:
        raise FileNotFoundError(
            "P&C model not found (bundle tellar/assets/pnc or HF cache)")
    _sp = SentencePieceProcessor(spe)
    _sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    log.info("P&C: loaded torch-free (%s)", os.path.basename(onnx))


def _ready():
    global _sp, _sess
    if _sp is None or _sess is None:
        with _lock:
            if _sp is None or _sess is None:
                _load()


def warm():
    """Load the model off the critical path (daemon thread at startup)."""
    try:
        _ready()
        log.info("P&C: model warmed")
    except Exception:
        log.exception("P&C: warm-up failed (will retry lazily on first use)")


def _windows(ids):
    """Split token ids into ≤(_MAX_LEN-2) windows with _OVERLAP carry, matching
    punctuators' TextInferenceDataset so window seams behave identically."""
    ml = _MAX_LEN - 2
    segs, start, idx = [], 0, 0
    while start < len(ids):
        adj = start - (0 if idx == 0 else _OVERLAP)
        segs.append(ids[adj:adj + ml])
        start = adj + ml
        idx += 1
    return segs


def _infer(text):
    """Punctuate + truecase `text` via the ONNX model. Torch-free."""
    _ready()
    ids = _sp.EncodeAsIds(text)
    if not ids:
        return text
    segs = _windows(ids)
    bos, eos = _sp.bos_id(), _sp.eos_id()
    m_ids, m_pre, m_post, m_cap = [], [], [], []
    for i, seg in enumerate(segs):
        arr = np.array([[bos] + seg + [eos]], dtype=np.int64)
        pre, post, cap, _seg = _sess.run(None, {"input_ids": arr})
        sl = slice(1, arr.shape[1] - 1)          # drop BOS/EOS
        pre_t = [None if _PRE_LABELS[x] == _NULL else _PRE_LABELS[x]
                 for x in pre[0][sl].tolist()]
        post_t = [None if _POST_LABELS[x] == _NULL else _POST_LABELS[x]
                  for x in post[0][sl].tolist()]
        cap_t = cap[0][sl].tolist()
        # Drop half the overlap on inner window edges (collector semantics).
        start = _OVERLAP // 2 if i > 0 else 0
        stop = len(seg) - (_OVERLAP // 2 if i < len(segs) - 1 else 0)
        m_ids += seg[start:stop]
        m_pre += pre_t[start:stop]
        m_post += post_t[start:stop]
        m_cap += cap_t[start:stop]
    return _render(m_ids, m_pre, m_post, m_cap)


def _render(ids, pre, post, cap):
    pieces = [_sp.IdToPiece(x) for x in ids]
    chars = []
    for ti, tok in enumerate(pieces):
        if tok.startswith("▁") and chars:
            chars.append(" ")
        cs = 1 if tok.startswith("▁") else 0
        for ci, ch in enumerate(tok[cs:], start=cs):
            if ci == cs and pre[ti]:
                chars.append(pre[ti])
            if ci < len(cap[ti]) and cap[ti][ci]:
                ch = ch.upper()
            chars.append(ch)
            lab = post[ti]
            if lab == _ACRONYM:
                chars.append(".")
            elif ci == len(tok) - 1 and lab:
                chars.append(lab)
    return "".join(chars).strip()


def _lower_strip(text):
    """Variant-C input: drop punctuation, lowercase → clean slate for the model."""
    return _STRIP.sub(" ", text).lower()


def _word_multiset(text):
    return sorted(_STRIP.sub(" ", text).lower().split())


def apply_pnc(text):
    """Re-punctuate and re-case `text`. Returns the input unchanged on any
    failure or if the model would have altered the words themselves."""
    if not text or not text.strip():
        return text
    try:
        out = _infer(_lower_strip(text)).strip()
        if not out:
            return text
        if _word_multiset(out) != _word_multiset(text):
            log.warning("P&C word-integrity mismatch — keeping raw text")
            return text
        return out
    except Exception:
        log.exception("P&C failed — keeping raw text")
        return text


# --- Seam-local P&C -------------------------------------------------------
# Fix ONLY chunk boundaries. Per seam, run the model on a small lowercased
# window (tail of chunk A + head of chunk B) and read two things: the casing
# of chunk B's first word (so proper nouns stay capital because the MODEL
# decided, not a list) and whether a terminator was placed at the junction.
# We set only chunk A's trailing punctuation and chunk B's first-letter case;
# every chunk interior stays exactly as whisper wrote it.


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
    """Join `chunks` fixing only the boundaries. Never raises — falls back to a
    plain join on any error."""
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
