import os
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
MODEL_DIR = Path.home() / "Library" / "Application Support" / "Tellar" / "models"

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
TRANSCRIPTION_VARIANT = "chunked_rolling_v11_smart_prompt_no_vad"

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

_model_loaded = False

# (pct, mb_done, mb_total)
ProgressCallback = Optional[Callable[[int, int, int], None]]


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
    _model_loaded = True


def _set_hf_offline(offline: bool):
    """Toggle huggingface_hub's offline mode reliably.

    huggingface_hub reads HF_HUB_OFFLINE from the env exactly once — at
    module import — and caches it in `huggingface_hub.constants.HF_HUB_OFFLINE`.
    Subsequent env changes are ignored. To switch modes mid-process we have
    to patch both the env var (so any *future* fresh import of HF Hub picks
    up the new value) AND the cached module constant (so the *already*
    imported HF Hub honours the change for its `is_offline_mode()` checks).
    """
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        import huggingface_hub
        huggingface_hub.constants.HF_HUB_OFFLINE = offline
    except (ImportError, AttributeError):
        pass


def _download_model(on_progress: ProgressCallback):
    """Download the model snapshot from HF Hub into the default cache.
    Calls on_progress(pct, mb_done, mb_total) repeatedly as bytes arrive.
    on_progress may be None — download still happens, just without UI feedback.
    Aggregates byte-progress across all files in the snapshot via a tqdm
    subclass that shares state on the class (snapshot_download creates one
    tqdm per file plus an outer file-count tqdm which we filter by `unit`).

    Caller is responsible for ensuring offline mode is OFF before calling.
    """
    import huggingface_hub
    # Use huggingface_hub's tqdm wrapper, NOT tqdm.auto.tqdm directly:
    # HF Hub's _create_progress_bar passes a `name=` kwarg that the plain
    # tqdm.auto base class doesn't accept — only HF's wrapper pops it before
    # delegating to super().__init__. Inheriting from the wrapper means our
    # subclass gets instantiated correctly inside snapshot_download / xet_get.
    from huggingface_hub.utils.tqdm import tqdm as _hf_tqdm

    class _AggTqdm(_hf_tqdm):
        def __init__(self, *args, **kwargs):
            # Force enabled. Without this, tqdm auto-disables in non-TTY
            # contexts (our .app bundle has no real stderr/TTY), and its
            # early-return path skips setting self.unit / self.total —
            # which then makes our update() crash with AttributeError.
            # We don't care about tqdm's own terminal rendering anyway,
            # this subclass exists purely to relay byte counts to the UI.
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self._last_emit_pct = -1
            self._last_emit_time = 0.0

        def update(self, n=1):
            ret = super().update(n)
            # snapshot_download uses one outer aggregating instance of this
            # class as `bytes_progress`. Per-file tqdms (HF Hub's internal
            # `_AggregatedTqdm`, not ours) do `bytes_progress.total += total`
            # as they spin up, then call `bytes_progress.update(n)` for each
            # chunk of bytes received. So `self.total` and `self.n` here are
            # the live aggregated totals across all files in the snapshot.
            if on_progress and self.unit == "B" and self.total:
                try:
                    pct = min(100, int(self.n * 100 / self.total))
                    # Throttle signal emission. hf_xet downloads chunks in
                    # parallel and fires update() many times per second; if
                    # we relay every one as a Qt signal, the main thread's
                    # event queue chokes and stops servicing menubar clicks
                    # AND the permissions poll timer — exactly the "icon
                    # unresponsive, second permission never appears" symptom
                    # observed during first-launch model download. Emit on
                    # percent change OR every 0.5s, whichever first.
                    now = time.time()
                    if pct != self._last_emit_pct or (now - self._last_emit_time) > 0.5:
                        on_progress(
                            pct,
                            self.n // (1024 * 1024),
                            self.total // (1024 * 1024),
                        )
                        self._last_emit_pct = pct
                        self._last_emit_time = now
                except Exception:
                    log.exception("download progress callback failed")
            return ret

    t0 = time.time()
    huggingface_hub.snapshot_download(MODEL_NAME, tqdm_class=_AggTqdm)
    log.info("Model download complete in %.2fs", time.time() - t0)


def model_exists() -> bool:
    # local_files_only=True forces a cache-only lookup regardless of the
    # global offline state — no env manipulation needed here.
    try:
        import huggingface_hub
        huggingface_hub.snapshot_download(MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False


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
        initial_prompt=PUNCTUATION_PROMPT,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
    )
    text = result.get("text", "").strip()
    return clean_hallucinations(text)


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
    return clean_hallucinations(text)


def transcribe_chunk(audio: np.ndarray, initial_prompt: Optional[str] = None) -> str:
    """Transcribe a single 16 kHz mono float32 audio chunk.

    For chunk 0 in a sequence pass initial_prompt=None to use the default
    bilingual style hint. For chunks 1+ the caller should pass the last
    ~200 chars of the previous chunk's text — that rolling context keeps
    proper-noun spelling, capitalization and punctuation style consistent
    across chunk boundaries (the standard whisper-streaming pattern).

    Note: we tried prepending PUNCTUATION_PROMPT to the rolling prompt
    to restore punctuation bias on chunk boundaries, but it tripped
    whisper's compression_ratio_threshold retry cascade (the
    style-mismatched concatenation produced degenerate decoder output
    → retries at higher temperatures → 3-5x per-chunk slowdown AND
    worse final quality because the retries also lose punctuation).
    Rolling-only is the lesser evil.

    Returns the cleaned text (post hallucination filter). Returns "" for
    an empty ndarray (zero-length tail on a stop that landed exactly on
    a chunk boundary).
    """
    if len(audio) == 0:
        return ""
    _set_hf_offline(True)
    import mlx_whisper
    prompt = initial_prompt if initial_prompt else PUNCTUATION_PROMPT
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
    )
    text = result.get("text", "").strip()
    return clean_hallucinations(text)
