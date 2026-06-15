"""Tellar Studio LLM — local text transformations via mlx-lm.

Studio presets ("Polish", "Email", "Formal"…) rewrite the text in the top
pane and drop the result in the bottom pane. This module is the engine: it
loads a local instruct model through mlx-lm and exposes transform(text, preset).

Decoupled from the UI — Studio calls transform() on a background thread and
gets the result via a Qt signal. The model is NOT loaded on import; it is
preloaded into RAM at app startup alongside the Whisper model (see
app.py:_preload_model), mirroring the Whisper flow. On a fresh install both
models download from HF Hub with the shared throttled progress UX
(hf_download), so the first launch shows two models downloading instead of one.

Model choice is a single constant: STUDIO_LLM_MODEL. The engine is model-
agnostic — it drives any instruct model that ships a chat template, so swapping
models (A/B comparison, see plans/studio-llm.md §2) is a one-line change.

This is Phase 1: lazy-download + load + transform() + one hardcoded "Polish"
preset. The full preset registry, Custom prompt, focus-based undo and the
two-pane UI layer on in later phases.
"""
import time
from dataclasses import dataclass

from . import hf_download
from .logging_setup import get_logger

log = get_logger(__name__)

# Single point of model choice. Swap to compare models (plans/studio-llm.md §2).
# Staying on 3B for now. We tried Qwen2.5-7B-Instruct-4bit (2026-05-31): better
# filler removal and cleaner single-language polish, BUT not a clean win — it
# still failed the same Russian declension ("Встреча перенесли"), and on mixed-
# language input it leaked Chinese into the output (Qwen multilingual quirk),
# whereas 3B silently dropped a chunk. Neither handles mixed-language in one
# pass; that case is out of scope for v1. The 7B cost (~4.5 GB, slower) didn't
# justify the partial gain, so we reverted. Revisit model choice in Phase 5.
# Other A/B candidates: Qwen3-4B-Instruct-2507-4bit, gemma-3-4b-it-qat-4bit.
STUDIO_LLM_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"


@dataclass(frozen=True)
class Preset:
    """A text transformation. Presets differ only by their system prompt
    (and optional decode params) — the engine is otherwise identical."""
    key: str           # 'polish'
    label: str         # 'Polish' — button text
    system: str        # instruction placed in the system message
    temperature: float = 0.2
    max_tokens: int = 1024
    # Whether the Studio should highlight word-level differences between the
    # source pane and this preset's result. True for in-place edits like Polish
    # (where every change is a meaningful copy-edit). Set False for presets
    # that produce a wholly different output — translate, summarise — where
    # almost every token is "different" and the highlighting is just noise.
    show_diff: bool = True


# Output-only discipline (a gotcha with small models): they love prepending
# "Sure! Here's the polished version:". We fight it twice — this line in every
# system prompt, plus _strip_preamble() post-processing.
_OUTPUT_ONLY = (
    " Output ONLY the transformed text. No preamble, no explanations, "
    "no quotes, no markdown fences. Keep the original language."
)

# Meta-template for the Custom preset (free-form runtime instruction). Differs
# structurally from fixed presets: in Polish/Email/Slack the system message IS
# the instruction; here it just frames the model as an assistant that follows
# whatever the user typed in the Custom field. The actual instruction travels
# in the user message together with the source text. Output discipline is the
# same; we drop the "keep the original language" rule because Custom is the
# escape hatch where the user might explicitly ask for a translation.
CUSTOM_SYSTEM = (
    "You are a writing assistant. Apply the user's instruction to their "
    "text. Output ONLY the resulting text — no preamble, no explanations, "
    "no quotes, no markdown fences."
)

POLISH = Preset(
    key="polish",
    label="Polish",
    system=(
        "You are an expert editor who turns rough speech-to-text dictation "
        "into clean, correct written text. Do all of the following:\n"
        "- Remove filler words, verbal tics, false starts and self-corrections "
        "(e.g. ну, вот, короче, это самое, как бы, типа, значит, э-э; um, uh, "
        "like, you know, I mean).\n"
        "- Fix grammar, spelling and punctuation. In Russian, fix case endings "
        "and subject-verb-object agreement (падежи и согласование).\n"
        "- Fix word order and split or merge sentences so the text reads "
        "naturally as writing, not as a transcript.\n"
        "- Preserve the original meaning and every fact. Do NOT add, invent or "
        "explain anything.\n"
        "- Keep the SAME language as the input. If the text mixes languages, "
        "polish each part in its own language." + _OUTPUT_ONLY
    ),
)


# V2 prompt — written to fix systematic failures observed across the v1/v2/v3
# A/B rounds on Llama-3.1-8B-Instruct-4bit:
#  * "я рассмотрел вашу проблему" — model slipped into chatbot/answer mode
#    instead of editing the author's text → explicit "you are not a chatbot"
#    instruction.
#  * Truncating long inputs (lost endings of multi-paragraph reports/emails)
#    → explicit "do not shorten or summarize; output length ≈ input length".
#  * Garbling proper nouns ("Анан" → "Анас") → explicit "verbatim" rule
#    with concrete examples from our domain.
#  * Formalizing a casual Slack message and vice versa → explicit
#    register-preservation rule.
# Temperature lowered 0.2 → 0.1 to reduce decode-noise grammar slips
# ("я с радостью обсудим их"). max_tokens unchanged.
POLISH_V2 = Preset(
    key="polish_v2",
    label="Polish v2",
    temperature=0.1,
    system=(
        "You are a copy-editor. The input is a rough speech-to-text "
        "dictation. Your output is the same text, rewritten as clean "
        "written prose.\n\n"
        "You are NOT a chatbot. Never address or reply to the author. "
        "Never describe what they said — restate it as they would have "
        "written it carefully themselves.\n\n"
        "Required edits:\n"
        "- Remove fillers and false starts (ну, вот, короче, это самое, "
        "как бы, типа, значит, э-э; um, uh, like, you know, I mean).\n"
        "- Fix grammar, punctuation; in Russian, fix падежи и "
        "согласование.\n"
        "- Fix word order; split or merge sentences for written flow.\n\n"
        "Preserve, never change:\n"
        "- Every fact and detail. Do NOT shorten, summarize, or skip any "
        "part. Output length ≈ input length.\n"
        "- Every proper noun and technical term verbatim (e.g. Anand, "
        "MyGold, edge case, chart-store, devops, UAT).\n"
        "- The author's register. A casual Slack message stays casual; "
        "a formal email stays formal. Do not formalize informal text. "
        "Do not casualize formal text.\n"
        "- The original language. Do not translate. If text mixes "
        "languages, polish each part in its own language." + _OUTPUT_ONLY
    ),
)

# Phase 1: just Polish. Phase 3 fills this out (Email / Formal / Casual /
# Bullets / Translate / Custom…). This list is the source of truth for both
# the preset buttons and the engine.
PRESETS: list[Preset] = [POLISH]


_model = None
_tokenizer = None
_loaded = False
_mlx_imported = False


def _release_mlx_scratch():
    """Drop MLX's free-cache pool after a generate() call. Same rationale
    as transcriber._release_mlx_scratch — the LLM's per-call KV cache and
    activation tensors otherwise stay resident under the cache pool. With
    a 4-bit Qwen-3B that's an extra ~500 MB-1 GB stranded after every
    transform; idle-unload (planned) will eventually drop the weights too,
    but releasing scratch is the cheap baseline."""
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def is_loaded() -> bool:
    """True once get_model() has finished loading the model into memory.

    UI code uses this to disable Polish / Apply while a lazy load is in
    flight: the buttons reflect "you can run a transform" rather than
    "you can request one" — clicking before load completes would block
    the worker thread on get_model() with no visible feedback.
    """
    return _loaded


def unload():
    """Drop the loaded LLM weights and clear MLX scratch.

    Called when the user turns Dictate to Studio off — without this the
    ~1.5-2 GB Qwen-3B-4bit weights stay resident for the rest of the
    process lifetime, which is what made memory climb sticky-up after
    every Studio session in a long-running daemon. After unload, the
    next transform() will trigger get_model() again (~12s reload from
    the warm HF cache); the user already paid that cost on toggle-on,
    so the symmetry is acceptable.

    Safe to call while a transform() is mid-inference — Python ref
    counting keeps the model object alive until generate() returns;
    we just clear the module-level handles so future calls reload."""
    global _model, _tokenizer, _loaded
    if not _loaded and _model is None:
        return
    log.info("Unloading Studio LLM (was loaded=%s)", _loaded)
    _model = None
    _tokenizer = None
    _loaded = False
    try:
        import gc
        gc.collect()
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        log.exception("Studio LLM unload cleanup failed (non-fatal)")


def preimport_mlx():
    """Pre-import mlx_lm in the background so the lazy get_model() doesn't
    pay the ~9 sec cold-import cost when the user toggles Dictate to Studio.

    Cheap (no weights allocated) and idempotent. Called from app.py once,
    on idle after the Whisper preload is done — that timing keeps the
    import off the critical path of first dictation while still being
    almost certainly finished by the time the user opens Studio.

    Logged so we can confirm in production that the warmup actually saves
    the import time it claimed to in development.
    """
    global _mlx_imported
    if _mlx_imported:
        return
    t0 = time.time()
    try:
        import mlx_lm  # noqa: F401 — side effect: cache the modules
    except Exception:
        log.exception("mlx_lm pre-import failed; lazy get_model will retry")
        return
    _mlx_imported = True
    log.info("mlx_lm pre-imported in %.2fs (lazy get_model will skip this)",
             time.time() - t0)


def get_model(on_download_progress: hf_download.ProgressCallback = None):
    """Ensure the model is downloaded and loaded into memory. Idempotent.

    Mirrors transcriber.get_model: download (with progress) only if the
    snapshot is missing, then load into RAM. Called from the startup preload
    thread; transform() also calls it defensively in case it runs first.

    Logs a per-phase timing breakdown so a slow load (~12s on Qwen-3B-4bit,
    likely ~22-28s on Llama-8B-8bit) can be attributed to the right cause:
    cold cache check, mlx_lm cold import, or the actual weight load.
    """
    global _model, _tokenizer, _loaded
    if _loaded:
        return
    t_cache = time.time()
    cached = hf_download.snapshot_exists(STUDIO_LLM_MODEL)
    log.info("Studio LLM cache check: %.2fs (cached=%s)",
             time.time() - t_cache, cached)
    if not cached:
        log.info("Studio LLM %s not in HF cache, downloading", STUDIO_LLM_MODEL)
        hf_download.set_hf_offline(False)
        hf_download.download_snapshot(STUDIO_LLM_MODEL, on_download_progress)
    hf_download.set_hf_offline(True)
    t_total = time.time()
    log.info("Loading Studio LLM %s into memory...", STUDIO_LLM_MODEL)
    t_import = time.time()
    from mlx_lm import load
    log.info("Studio LLM mlx_lm import: %.2fs", time.time() - t_import)
    t_load = time.time()
    _model, _tokenizer = load(STUDIO_LLM_MODEL)
    log.info("Studio LLM load(): %.2fs", time.time() - t_load)
    _loaded = True
    log.info("Studio LLM total %.2fs", time.time() - t_total)


def build_chat_prompt(tokenizer, system: str, user: str):
    """Build a chat prompt from a system + user message.

    Some chat templates (notably Gemma) have no `system` role and raise if one
    is passed. For those we fold the system instruction into the user turn —
    the model still sees it, just not as a separate role. Shared by transform()
    and the A/B harness so both handle every candidate model uniformly.
    """
    try:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            add_generation_prompt=True,
        )
    except Exception:
        merged = f"{system}\n\n{user}"
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": merged}],
            add_generation_prompt=True,
        )


def transform(text: str, preset: Preset) -> str:
    """Run the preset over the whole text and return the rewritten result.

    Blocking — call from a worker thread, not the Qt main thread. Assumes the
    model is loaded (preloaded at startup); loads on demand if not.
    """
    get_model()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = build_chat_prompt(_tokenizer, preset.system, text)
    sampler = make_sampler(temp=preset.temperature)
    t0 = time.time()
    out = generate(
        _model, _tokenizer, prompt,
        max_tokens=preset.max_tokens, sampler=sampler, verbose=False,
    )
    log.info(
        "Studio transform '%s': %d chars -> %d chars in %.2fs",
        preset.key, len(text), len(out), time.time() - t0,
    )
    _release_mlx_scratch()
    return _strip_preamble(out)


def transform_custom(text: str, instruction: str) -> str:
    """Apply a user-supplied runtime instruction to the text.

    Differs from transform(text, preset) in WHERE the instruction lives:
    fixed presets freeze it into the system message at import time; here it
    arrives at call time and rides in the user message alongside the text.
    Same model, same chat-template builder, same output cleanup.

    Blocking — call from a worker thread, not the Qt main thread.
    """
    get_model()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    user_msg = f"INSTRUCTION: {instruction}\n\nTEXT:\n{text}"
    prompt = build_chat_prompt(_tokenizer, CUSTOM_SYSTEM, user_msg)
    sampler = make_sampler(temp=0.2)
    t0 = time.time()
    out = generate(
        _model, _tokenizer, prompt,
        max_tokens=1024, sampler=sampler, verbose=False,
    )
    log.info(
        "Studio transform_custom: %d chars text + %d chars instruction "
        "-> %d chars in %.2fs",
        len(text), len(instruction), len(out), time.time() - t0,
    )
    _release_mlx_scratch()
    return _strip_preamble(out)


# Common conversational lead-ins small models emit despite instructions.
_PREAMBLE_STARTS = (
    "sure", "here's", "here is", "here you go", "certainly", "of course",
    "below is", "the polished", "the rewritten", "rewritten text",
)


def _strip_preamble(text: str) -> str:
    """Best-effort output-only cleanup: drop a leading 'Sure, here's…' line,
    surrounding quotes and ```fences```. Conservative — only strips when the
    pattern is unambiguous. Expanded in Phase 4 as real cases surface."""
    s = text.strip()

    # Some chat templates leak their end-of-turn / eos marker into the decoded
    # text (seen with Gemma-2: a trailing "<end_of_turn>"). Drop known special
    # tokens wherever they appear.
    for tok in ("<end_of_turn>", "<eos>", "<bos>", "<|im_end|>", "<|eot_id|>",
                "</s>", "<pad>"):
        s = s.replace(tok, "")
    s = s.strip()

    # Drop a single leading lead-in line ("Sure! Here's the polished text:").
    lines = s.split("\n", 1)
    if len(lines) == 2:
        first = lines[0].strip().lower()
        if any(first.startswith(p) for p in _PREAMBLE_STARTS) and first.endswith(":"):
            s = lines[1].strip()

    # Strip a wrapping ``` fence (with or without a language tag).
    if s.startswith("```"):
        s = s[3:]
        if "\n" in s:
            s = s.split("\n", 1)[1]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()

    # Strip matching surrounding quotes (straight and typographic pairs).
    _QUOTE_PAIRS = (('"', '"'), ("'", "'"), ("“", "”"), ("«", "»"))
    for open_q, close_q in _QUOTE_PAIRS:
        if len(s) >= 2 and s[0] == open_q and s[-1] == close_q:
            s = s[1:-1].strip()
            break

    return s
