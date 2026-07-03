import os
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from . import hf_download
from .hallucinations import remove_hallucinations
from .hf_download import ProgressCallback
from .logging_setup import get_logger
from .vocabulary import format_vocabulary_suffix

log = get_logger(__name__)

MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
MODEL_DIR = Path.home() / "Library" / "Application Support" / "Tellar" / "models"

# Cap on MLX's free-cache pool. MLX retains scratch buffers (mel spec,
# encoder hidden, decoder KV) between transcribe calls for reuse — without
# a cap, the pool grows to the high-water mark of the longest chunk ever
# seen and never shrinks for the life of the process. Over a multi-day
# uptime that pool was observed at 6+ GB, swapping the model weights out
# and dragging individual transcriptions to 12s on 7s of audio.
# 256 MB is enough to keep one chunk's scratch hot; everything beyond
# that is discarded on the next allocation. Pair with mx.clear_cache()
# in pipeline.py after each chunk for steady-state behaviour.
MLX_CACHE_LIMIT_BYTES = 256 * 1024 * 1024

# Feature flag for the chunked-transcription pipeline. When False (default),
# _stop_recording transcribes the full WAV in one mlx_whisper call — the
# pre-migration behavior. When True, the recording is sliced into 30s chunks
# transcribed in parallel with capture, then stitched on stop. See
# ~/context/tellar/plans/chunked-transcription.md for the full design.
#
# A single boolean here is the rollback story: flip to False, cp into the
# bundle, restart Tellar — pre-chunked behavior is restored. The Recorder
# keeps writing the full WAV in either path, so the fallback contract is
# always satisfied.
CHUNKED_TRANSCRIPTION = True

# Identifier for the current transcription configuration. Written into
# every transcription_log_chunked.jsonl row so we can group historical
# entries by behavior version without consulting timestamps + git log.
# Bump this whenever transcribe_chunk / finalize / chunk size /
# temperature schedule / prompt strategy changes in a way that could
# affect output text or per-chunk latency. Keep a comment-history below.
#
#   chunked_rolling_v1 — Phase 3 baseline. 30s chunks, rolling 200-char
#                        prompt, no PUNCTUATION_PROMPT prefix, default
#                        temperature schedule (0.0..1.0 with
#                        compression_ratio_threshold=2.0 retries),
#                        no_speech_threshold=0.5.
#   chunked_rolling_v2 — short-lived 2026-05-27 experiment: prepended
#                        PUNCTUATION_PROMPT before rolling prompt.
#                        Caused 3-5x per-chunk slowdown via temperature
#                        fallback cascade. Reverted same day.
#   chunked_rolling_v4_loose_threshold —
#                        v1 + compression_ratio_threshold raised from
#                        2.0 to 2.4 (whisper's documented default). The
#                        threshold-2.0 setting was inherited from the
#                        full-WAV transcribe_audio path; on chunked
#                        boundaries it triggered temperature retry
#                        cascade too aggressively (chunks cut mid-word
#                        produce borderline-degenerate output that
#                        looked "too dense" but didn't benefit from
#                        retries). Higher threshold should reduce
#                        retry triggers, smoothing out the 11-25s
#                        outlier chunks observed in v1. Tested
#                        2026-05-27: speed dramatically improved
#                        (median 2.0s, no >6s outliers), but degenerate
#                        chunks now visibly emit 5-40 chars and poison
#                        the rolling prompt of subsequent chunks, so
#                        whole recordings sometimes lose half the
#                        content.
#   chunked_rolling_v5_defensive —
#                        v4 + (a) skip rolling-prompt update when a
#                        full chunk returns < 20 chars — keeps the
#                        last known-good prompt for the next chunk,
#                        breaking the failure cascade; (b) on finalize,
#                        if any full chunk in the result set was
#                        degenerate (< 20 chars), fall back to
#                        transcribe_audio on the preserved full WAV.
#                        Trades occasional ~5s latency hits (when
#                        chunks fail) for guaranteed quality, while
#                        keeping v4's speed on the happy path.
#                        Tested 2026-05-27: 92% happy path, but
#                        fallback also fails on the same problematic
#                        recordings — root cause is whisper's internal
#                        prompt-poisoning across its 30s windows
#                        (condition_on_previous_text=True default).
#   chunked_rolling_v6_no_condition_fallback —
#                        v5 + transcribe_audio now passes
#                        condition_on_previous_text=False. This stops
#                        whisper's INTERNAL 30s windows from feeding
#                        each other prompts: a degenerate first window
#                        no longer poisons the second, the second no
#                        longer poisons the third, etc. Targets the
#                        observed pattern where bad audio at the start
#                        of a recording cascades through the rest.
#                        Trade-off: loses long-range proper-noun
#                        consistency on the fallback path. Acceptable —
#                        fallback is a rescue path, prioritize content
#                        recovery over style consistency.
#                        TESTED 2026-05-27: still saw catastrophic
#                        failures. Diagnostic on sample_..._120s.wav
#                        (real Russian speech about decree/children)
#                        revealed PUNCTUATION_PROMPT itself was the
#                        root cause — bilingual prompt + our threshold
#                        tweaks pushed long-audio decoder into either
#                        garbage output (one Cyrillic char + thousands
#                        of combining marks → cleanup zeroed it out)
#                        or English translation mode. Whisper defaults
#                        with NO prompt gave clean Russian output on
#                        the same file in 6s. v7 fixes this.
#   chunked_rolling_v7_fallback_defaults —
#                        v6 + new transcribe_audio_defaults function
#                        used by pipeline.finalize() fallback path.
#                        Calls mlx_whisper.transcribe with stock
#                        defaults — no PUNCTUATION_PROMPT, no custom
#                        threshold/conditioning. The original
#                        transcribe_audio (with PROMPT) stays for the
#                        flag-off path and any short-recording use
#                        case where prompt fixed cold-start
#                        punctuation. Fallback is only reached for
#                        recordings ≥ 30s with at least one degenerate
#                        chunk; on these the prompt-induced
#                        degenerations far outweigh its benefit.
#                        Tested 2026-05-27: helped some cases but a
#                        172s recording still came back as 228 chars.
#                        Diagnostic isolated the root cause to
#                        clean_hallucinations — whisper's raw output
#                        was 2040 chars of clean Russian, our cleanup
#                        regex saw "нет нет нет" mid-text and trimmed
#                        everything after it. Fixed in v8.
#   chunked_rolling_v8_clean_substitute —
#                        v7 + clean_hallucinations rewritten as a
#                        non-destructive substitute. Regex \1{3,}
#                        (4+ total instances) catches real whisper
#                        loops (typically 10-100+ repetitions) but not
#                        natural speech patterns ("нет нет нет",
#                        "да-да", "что-что"). Match is replaced with
#                        a single instance via re.sub; all text
#                        surrounding the run is preserved verbatim.
#                        Closes the loop on multiple cases that
#                        looked like whisper failures (i think i'm
#                        excited × 2 → 39 chars; 172s → 228 chars).
#                        Tested 2026-05-27: catastrophic failures
#                        gone, but punctuation regression appeared
#                        on second half of long recordings — root
#                        cause was rolling prompt carrying forward
#                        text from chunks that hit retry cascade
#                        (T=0.6+ outputs lacked sentence terminators
#                        → next chunk inherited "mid-thought" style).
#   chunked_rolling_v10_vad_smart_polish —
#                        v8 + three changes:
#                        (a) ChunkingBuffer cuts at natural pauses
#                            instead of fixed 30s. Looks for ≥150ms
#                            of silence (RMS < 0.01) starting at the
#                            30s mark, with a hard force-cut at 35s
#                            if no pause found. Removes mid-word
#                            audio splits — root architectural fix
#                            for the boundary-quality issues we've
#                            been working around since v3.
#                        (b) Smart rolling prompt: chunk text used
#                            as next chunk's prompt only if it ends
#                            with .!?…; otherwise reset to
#                            PUNCTUATION_PROMPT. Defensive layer for
#                            the residual case where whisper goes
#                            degenerate even on clean audio.
#                        (c) Strip leading whisper "..." continuation
#                            marker from chunk text before storing.
#                            With VAD-cut chunks this rarely fires,
#                            but it's belt-and-suspenders against
#                            the "ellipsis between joined chunks"
#                            artifact.
#                        + heavy telemetry: per-chunk durations,
#                        cut_reasons (vad/force_max/tail), per-chunk
#                        transcribe_secs, per-chunk chars, degenerate
#                        indices, rolling reset count, finalize path.
#                        TESTED 2026-05-27: VAD made things WORSE —
#                        chunk endings on natural pauses were
#                        interpreted by whisper as "end of show",
#                        triggering "Продолжение следует..."
#                        hallucination that REPLACED real chunk-tail
#                        content. Reverted in v11.
#                        Smart rolling logic also had a bug: "…" was
#                        in the terminator set, so "Продолжение
#                        следует..." (which ends with .) was treated
#                        as clean → the bad prompt cascaded.
#   chunked_rolling_v11_smart_prompt_no_vad —
#                        v8 chunking (fixed 30s cuts) + the v10
#                        improvements that actually worked: smart
#                        rolling prompt (now correctly excluding
#                        "..." and "…" endings as dirty), strip
#                        leading continuation marker, and full
#                        telemetry. Targeted fix for the v8 residual
#                        bug — when a chunk hit retry cascade and
#                        produced text without a sentence terminator,
#                        next chunk inherited "mid-thought" prompt
#                        and the whole second half lost punctuation.
#                        Now next chunk gets PUNCTUATION_PROMPT in
#                        that case. Doesn't change chunking, doesn't
#                        change whisper params — just smarter prompt
#                        handoff. Minimal change over best-known v8.
#   chunked_rolling_v12_hallucination_filter —
#                        v11 + post-hoc filter for known whisper
#                        hallucinations (tellar/hallucinations.py).
#                        Targets phrases the model emits when it
#                        interprets audio as end-of-segment markers
#                        — "Продолжение следует...", "DimaTorzok"
#                        subtitle attribution, English channel
#                        sign-offs etc. — REPLACING real content
#                        with these phrases. Hand-curated blocklist,
#                        applied after clean_hallucinations on every
#                        transcribe path (chunks + chunked-assembly +
#                        full-WAV fallback + flag-off path).
#   chunked_rolling_v13_temp0 —
#                        v12 + transcribe_chunk pins temperature=0.0,
#                        disabling whisper's temperature-fallback retry
#                        cascade. Telemetry on a single-speaker, stable
#                        setup showed bimodal per-chunk latency: ~0.1x
#                        realtime when the first decode passes thresholds,
#                        but ~1x (10x slower) when avg_logprob dips below
#                        logprob_threshold (-1.0) and whisper re-decodes
#                        the whole 30s window at rising temperatures up to
#                        6x. Content-dependent, hence the same speech
#                        varying wildly run-to-run. The pipeline already
#                        has its own degenerate detector + full-WAV
#                        fallback, so whisper's internal retry is largely
#                        redundant. EXPERIMENT — comparing latency/quality
#                        vs v12 via per-variant telemetry.
#   chunked_rolling_v14_silero_vad —
#                        v13 + chunking switched from fixed-30s cuts to
#                        silero-vad natural-pause cuts (chunking_vad.py).
#                        Re-attempt of the v10 idea that previously failed
#                        for two reasons we now address structurally:
#                        (1) v10 used RMS-threshold "silence" detection,
#                        which never fires under realistic background
#                        noise — silero is a small NN that classifies
#                        speech vs non-speech and ignores hum/breath/
#                        keyboard. (2) v10 left trailing silence inside
#                        the chunk → whisper interpreted it as "end of
#                        show" and emitted "Продолжение следует..."
#                        replacing the chunk's tail; we now trim trailing
#                        silence to last_speech_end + 100ms before
#                        emitting, so whisper sees a chunk that ends
#                        mid-utterance acoustically (no EOS cue). The
#                        v12 hallucinations.py filter remains as a
#                        belt-and-suspenders catch.
#                        Activated by VAD_CHUNKING=True below.
#   chunked_rolling_v15_suffix_punctuation —
#                        v14 + flips the prompt order in transcribe_chunk:
#                        rolling_text + ' ' + PUNCTUATION_PROMPT instead
#                        of PUNCTUATION_PROMPT + ' ' + rolling_text. The
#                        prefix layout from v14 didn't break the
#                        punctuation-loss cascade we'd been chasing —
#                        whisper's decoder weights the END of the prompt
#                        as the most recent style cue, and a
#                        terminator-less rolling tail at the end of the
#                        prompt would push subsequent chunks toward
#                        unpunctuated output regardless of the
#                        PUNCTUATION_PROMPT prefix. With the suffix
#                        layout, every chunk gets a period-terminated
#                        style anchor as the freshest token before
#                        decoding starts. Verified via offline replay
#                        harness (tools/replay_chunked.py) on 13 saved
#                        problem dictations: cascade eliminated on 9/10
#                        reproducible cases, zero regressions. Also
#                        bumped NATURAL_PAUSE_S from 0.5 → 0.6 in
#                        chunking_vad.py to reduce mid-thought VAD cuts.
#   chunked_rolling_v16_late_cut —
#                        v15 + raises MIN_SPEECH_S in chunking_vad.py from
#                        5s to 22s. Telemetry showed the v14/v15 chunker
#                        cut after the FIRST pause past 5s of speech →
#                        median chunk 9.5s, ~742 seams total, 80% of chunks
#                        <15s. That over-fragmentation was the root cause
#                        of the "рваность" and the whole class of chunk-
#                        seam artifacts (false capitals, boundary word
#                        duplication, stray punctuation) we'd been patching
#                        in finalize(). At 22s the policy becomes "fill the
#                        chunk to ~24-28s, then snap the cut to the next
#                        pause" — so ~67% of dictations (<28s) become a
#                        SINGLE seamless chunk, and long ones roughly halve
#                        their chunk count. seams.py reconcile (TERM/DUP/
#                        CAP) stays as a cheap belt-and-suspenders for the
#                        far-rarer remaining seams. No dictionary added.
VAD_CHUNKING = True
TRANSCRIPTION_VARIANT = (
    "chunked_rolling_v17_pnc_late_cut" if VAD_CHUNKING
    else "chunked_rolling_v13_temp0"
)

# Variant tag used by app.py when CHUNKED_TRANSCRIPTION=False, i.e. when
# the entire chunked pipeline is bypassed and transcribe_audio() is
# called on the full WAV. Bump if transcribe_audio's parameters change
# in a way that affects output text or latency on long recordings.
#
#   baseline_no_chunked_v1 — current transcribe_audio: PUNCTUATION_PROMPT
#                            + no_speech_threshold=0.5
#                            + compression_ratio_threshold=2.0
#                            + condition_on_previous_text=True (default).
#                            Same as the pre-Phase-0 production code.
BASELINE_VARIANT = "baseline_no_chunked_v1"

# Bias whisper-large-v3-turbo's decoder toward producing punctuation,
# capitalization, and proper sentence breaks. The turbo model is a distilled
# variant that drops these stylistic features when it can't infer sentence
# boundaries from the audio alone (fast unbroken speech, repeated words,
# nominal phrases). A short prompt with the target style nudges the decoder
# to continue in that style — a zero-cost fix that empirically takes our
# bad-output rate from ~50% to 0% on the diagnostic samples.
#
# initial_prompt does NOT lock the language — whisper still detects it from
# the audio. We use a bilingual RU/EN prompt because the user dictates in
# both languages (sometimes mixed within a single utterance). A single-
# language prompt biases cross-language outputs in subtle ways (transliteration
# artifacts, leakage of stop-words from the prompt language). Both languages
# present means whichever the decoder lands on, it has style context for it.
PUNCTUATION_PROMPT = (
    "Привет, как дела? Сегодня хороший день. "
    "Hello, how are you? Today is a great day."
)


def _with_vocabulary(base_prompt: str) -> str:
    """Append the user's custom-vocabulary suffix to a base prompt when
    the vocabulary file has any entries. Empty vocabulary → base prompt
    is returned unchanged. Used at every code path that constructs a
    fresh whisper initial_prompt (i.e. PUNCTUATION_PROMPT-based) — not
    on the chunked-pipeline rolling prompt, which has its own meaningful
    context and shouldn't be diluted by the vocab list.

    The vocab file is re-read on every call (cheap, < 1ms for typical
    sizes). That keeps the user's edits visible at the very next
    transcription without any cache-invalidation logic."""
    suffix = format_vocabulary_suffix()
    if not suffix:
        return base_prompt
    return f"{base_prompt} {suffix}"

_model_loaded = False


def get_model(on_download_progress: ProgressCallback = None):
    global _model_loaded
    if _model_loaded:
        return
    if not model_exists():
        log.info("Model %s not in HF cache, downloading", MODEL_NAME)
        _set_hf_offline(False)
        _download_model(on_download_progress)
    _set_hf_offline(True)
    t0 = time.time()
    log.info("Importing mlx_whisper.load_models...")
    from mlx_whisper.load_models import load_model
    log.info("mlx_whisper imported in %.2fs", time.time() - t0)
    t1 = time.time()
    log.info("Loading model %s into memory...", MODEL_NAME)
    load_model(MODEL_NAME)
    log.info("Model loaded in %.2fs", time.time() - t1)
    # Cap MLX's free-cache pool so scratch buffers don't accumulate over
    # the multi-day uptime of a daemon process. Idempotent — safe to call
    # before any mlx_whisper.transcribe(). See MLX_CACHE_LIMIT_BYTES.
    try:
        import mlx.core as mx
        prev = mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
        log.info("MLX cache limit set: %d -> %d bytes",
                 prev, MLX_CACHE_LIMIT_BYTES)
    except Exception:
        log.exception("Failed to set MLX cache limit (non-fatal)")
    _model_loaded = True


def _set_hf_offline(offline: bool):
    # Download progress + offline toggling live in the shared hf_download
    # module (single source of truth for both Whisper and the Studio LLM).
    hf_download.set_hf_offline(offline)


def _download_model(on_progress: ProgressCallback):
    hf_download.download_snapshot(MODEL_NAME, on_progress)


def model_exists() -> bool:
    return hf_download.snapshot_exists(MODEL_NAME)


def clean_hallucinations(text: str) -> str:
    """Collapse whisper-style repetition loops without destroying
    surrounding content. Earlier versions of this function trimmed
    everything from the first detected repetition onwards, which
    catastrophically over-fired on legitimate speech ("нет нет нет это
    не основная спальня" looked like a hallucination, and the entire
    rest of a 2400-char transcript was deleted, leaving 228 chars
    output for 172s of audio — diagnosed 2026-05-27).

    Real whisper hallucinations are runaway decoder loops emitting the
    same 3-50 char pattern 10-100+ times. Legitimate speech repetitions
    rarely exceed 3 consecutive. Pattern \\1{3,} requires 4+ total
    instances — well above natural repetition, well below loop length.
    On match, we substitute the run with a single instance of the
    pattern and continue with the rest of the text intact.
    """
    import re
    cleaned, n = re.subn(r'(.{3,50}?)\1{3,}', r'\1', text)
    if n:
        log.debug("Collapsed %d repetition run(s) in transcript", n)
    return cleaned


def _load_wav_mono16k(path: str) -> np.ndarray:
    # Recorder writes 16 kHz mono int16 PCM, so we can hand mlx_whisper a
    # ready-made waveform and skip its ffmpeg subprocess fallback entirely.
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _release_mlx_scratch():
    """Drop MLX's free-cache pool after a transcribe call. MLX retains
    scratch buffers (mel spec, encoder hidden, decoder KV) between calls
    for reuse — without explicit release the pool grows to the high-water
    mark of the largest chunk and never shrinks for the life of the
    process. Over multi-day uptimes this was observed at 6+ GB, swapping
    weights out under memory pressure and dragging individual chunks to
    12s on 7s of audio. Best-effort: wrapped so a non-Metal fallback path
    never breaks transcription."""
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def transcribe_audio(audio_path: str) -> str:
    """Full-WAV transcribe with the project's standard prompt + threshold
    tuning. Used by the flag-off code path (CHUNKED_TRANSCRIPTION=False)
    and historically by the chunked-pipeline fallback before v7.

    The bilingual PUNCTUATION_PROMPT fixes a cold-start regression on
    short recordings where ~20% of outputs would lose all punctuation
    (commit b062d46). It works there because the audio is one whisper
    window and the prompt is the only style hint the decoder gets.

    On LONG recordings (≥30s, multiple internal whisper windows) the
    same prompt occasionally pushes the decoder into degenerate output
    or English translation mode — see chunked_rolling_v6_no_condition_fallback
    comment for the diagnostic. The v7 pipeline fallback uses
    transcribe_audio_defaults() to avoid that failure mode while
    preserving the prompt-fix here for the flag-off path.
    """
    _set_hf_offline(True)
    import mlx_whisper
    audio = _load_wav_mono16k(audio_path)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_NAME,
        initial_prompt=_with_vocabulary(PUNCTUATION_PROMPT),
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
    )
    text = result.get("text", "").strip()
    _release_mlx_scratch()
    return remove_hallucinations(clean_hallucinations(text))


def transcribe_audio_defaults(audio_path: str) -> str:
    """Full-WAV transcribe with whisper's stock defaults — no prompt,
    no custom thresholds, condition_on_previous_text=True. Used by
    pipeline.finalize() as the chunked-path fallback.

    Why pull all the customizations: diagnostic on a real failing
    sample (sample_..._120s.wav, normal Russian speech) found that our
    PUNCTUATION_PROMPT was the trigger for whisper to either emit
    pure garbage (one Cyrillic char followed by thousands of combining
    marks, then cleanup zeroed it) or switch into English translation
    mode. Same file with whisper defaults returned clean Russian text
    in 6s. Fallback runs only on long recordings that already lost a
    chunk in the chunked path — keeping the prompt there can turn a
    partial loss into total loss.

    Cold-start punctuation regression (the reason PROMPT exists) only
    affects SHORT recordings, which never reach the fallback path
    (no full chunk to fail).
    """
    _set_hf_offline(True)
    import mlx_whisper
    audio = _load_wav_mono16k(audio_path)
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL_NAME)
    text = result.get("text", "").strip()
    _release_mlx_scratch()
    return remove_hallucinations(clean_hallucinations(text))


def transcribe_chunk(audio: np.ndarray, initial_prompt: Optional[str] = None) -> str:
    """Transcribe a single 16 kHz mono float32 audio chunk.

    For chunk 0 in a sequence pass initial_prompt=None to use the default
    bilingual style hint. For chunks 1+ the caller should pass the last
    ~200 chars of the previous chunk's text — that rolling context keeps
    proper-noun spelling, capitalization and punctuation style consistent
    across chunk boundaries (the standard whisper-streaming pattern).

    Prompt composition for rolling chunks:
        rolling_text + ' ' + PUNCTUATION_PROMPT

    Order matters: whisper's decoder treats the END of the prompt as
    the most recent style cue. By placing PUNCTUATION_PROMPT after the
    rolling tail, every chunk receives a fresh punctuation/capitalization
    anchor regardless of how the previous chunk's tail looked. The
    earlier prefix layout (PUNCTUATION_PROMPT first, then rolling) let
    a terminator-less rolling tail dominate as the freshest style cue,
    which cascaded "no punctuation" forward across all subsequent
    chunks once a single chunk drifted.

    A short-lived v2 experiment (2026-05-27) tried adding the
    PUNCTUATION_PROMPT to the rolling prompt and was reverted: the
    style-mismatched concatenation tripped whisper's
    compression_ratio_threshold + temperature-fallback retry cascade,
    causing 3-5x per-chunk slowdown AND worse output. v13 pinned
    temperature=0.0 — disabling that retry cascade — so the v2 failure
    mode no longer applies. v14 brought the prepend back as prefix; on
    13 saved problem dictations the prefix layout failed to break the
    cascade. v15 (2026-06-17) flips to suffix layout based on A/B
    replay results: cascade eliminated on 9/10 reproducible cases,
    zero regressions. See tools/replay_chunked.py for the harness.

    Returns the cleaned text (post hallucination filter). Returns "" for
    an empty ndarray (zero-length tail on a stop that landed exactly on
    a chunk boundary).
    """
    if len(audio) == 0:
        return ""
    _set_hf_offline(True)
    import mlx_whisper
    if initial_prompt:
        # Rolling context first, PUNCTUATION_PROMPT last. Order matters:
        # whisper's decoder weights the END of the prompt as the most
        # recent style cue. With PUNCTUATION_PROMPT in the suffix
        # position, every chunk gets a fresh, period-terminated style
        # anchor regardless of whether the rolling tail itself ended
        # with a terminator. The prefix-position layout used previously
        # let a terminator-less rolling tail dominate as the most
        # recent style cue, cascading "no punctuation" forward across
        # the rest of the recording.
        # Vocabulary skipped here — the rolling tail already carries
        # chunk-specific named entities through, and adding the vocab
        # list would crowd the ~224-token prompt budget.
        prompt = initial_prompt + ' ' + PUNCTUATION_PROMPT
    else:
        # Chunk 0: no rolling context, so vocab attaches to the style
        # hint to bias proper nouns from the user's vocabulary.txt.
        prompt = _with_vocabulary(PUNCTUATION_PROMPT)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_NAME,
        initial_prompt=prompt,
        no_speech_threshold=0.5,
        # v4 — raised from 2.0 to whisper's documented default 2.4.
        # Chunked audio cut mid-word looks borderline-degenerate by
        # the compression ratio metric, triggering temperature retries
        # that don't help (output stays poor) but cost 3-5x latency.
        # Looser threshold keeps the recovery path for genuinely broken
        # output (compression ratio >> 2.4 means actual repetition loops)
        # without firing on every awkward boundary.
        compression_ratio_threshold=2.4,
        # v13 — single temperature, no fallback cascade. The default
        # schedule re-decodes the whole window up to 6x when avg_logprob
        # dips below logprob_threshold, which on a stable single-speaker
        # setup caused ~10x run-to-run latency swings. The pipeline's own
        # degenerate detector + full-WAV fallback covers genuine failures.
        temperature=0.0,
    )
    text = result.get("text", "").strip()
    _release_mlx_scratch()
    return remove_hallucinations(clean_hallucinations(text))
