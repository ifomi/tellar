"""Silero-VAD-driven chunking buffer — drop-in replacement for the
fixed-30s ChunkingBuffer.

The old ChunkingBuffer cut the audio every 30 s on the wall clock,
which routinely landed mid-word and produced "..." artifacts at
chunk seams (visible in transcriber.py's _LEADING_CONTINUATION_RE
strip + the "просто-... ...просто" pattern users complained about).

This buffer uses silero-vad to find natural pauses in speech and
cut there. Crucially it differs from the failed v10 RMS-based
attempt in two ways:

  1. silero-vad classifies SPEECH vs non-speech with a small NN —
     it ignores background hum, keyboard, fan noise, breathing.
     The v10 RMS-threshold approach was foiled by any constant
     background, since "silence" at amplitude < 0.01 essentially
     never occurs in real environments.

  2. We TRIM trailing silence from the chunk before emitting it.
     v10 left the natural pause inside the chunk, and whisper
     interpreted that "speech then trailing pause" pattern as
     "end of show / segment", emitting "Продолжение следует..."
     hallucinations that REPLACED the chunk's tail content.
     Cutting at last_speech_end + 100ms padding gives whisper a
     chunk that ends mid-utterance acoustically — no EOS cue.

Cut policy
----------

  * Natural cut: ≥ NATURAL_PAUSE_S of continuous non-speech AFTER
    at least MIN_SPEECH_S of accumulated speech in the chunk. Cut
    point = last_speech_end + TRIM_PAD_S.
  * Hard cut: chunk hits HARD_CUT_S without a natural opportunity.
    Cut at the quietest window in the last few seconds (the spot
    where mid-word risk is locally minimized).
  * Tail: whatever's left in the buffer at flush() time, regardless
    of speech state.

Speech-state hysteresis: enter speech at p ≥ ENTER_THRESHOLD, leave
at p ≤ LEAVE_THRESHOLD. Without hysteresis a single low-confidence
frame in the middle of a sentence flips the state and resets
last_speech_end, defeating the trim.

Threading: same model as the old ChunkingBuffer — single-producer
recorder thread calls push(); flush() runs on main thread AFTER
recorder.join(). No locking.
"""

import audioop
from typing import List, Optional, Tuple

import numpy as np

from .logging_setup import get_logger
from .vad import SileroVAD, WINDOW_SAMPLES

log = get_logger(__name__)


TARGET_RATE = 16000
SAMPLE_WIDTH = 2  # int16

# Target window for natural / hard cuts. We aim for chunks under 30s
# because mlx-whisper's internal window is 30s — chunks longer than
# that get split internally with whatever boundary whisper picks
# (typically worse than ours because it doesn't know about silero).
HARD_CUT_S = 28.0

# Don't even try to find a natural cut until the chunk has at least
# this much accumulated SPEECH (silence doesn't count). Avoids cutting
# after the first phrase when more is clearly coming. Keeps chunks
# meaty enough that whisper's per-chunk overhead amortizes.
MIN_SPEECH_S = 5.0

# Continuous non-speech run that qualifies as a natural cut point.
# 500 ms — slightly above the typical intra-sentence pause (200-400 ms
# for emphasis or word-search) and below the typical inter-sentence
# pause (700+ ms). Cutting at 400 ms ripped logical sentences in half
# at intra-thought pauses; with 500 ms we should cut closer to actual
# sentence boundaries, while force_max picks up the slack when a
# single burst genuinely runs past 28s without any pause that long.
NATURAL_PAUSE_S = 0.5

# Padding kept after last_speech_end before the cut. Defends against
# clipping the final consonant of a word; small enough that whisper
# doesn't read it as "trailing silence → end of show" (the v10
# failure mode at ≥150ms RMS-detected silence).
TRIM_PAD_S = 0.1

# Hysteresis thresholds for the speech state machine. Hand-tuned
# starting points from silero's docs; can be exposed as runtime knobs
# if telemetry shows we're missing speech onsets or holding state too
# long.
ENTER_THRESHOLD = 0.5
LEAVE_THRESHOLD = 0.35

# When forced to hard-cut, scan the last HARD_CUT_SCAN_S seconds of
# windows for the quietest (lowest p_speech) one and cut there. This
# is the locally-best mid-word-risk minimization when no real pause
# was found.
HARD_CUT_SCAN_S = 5.0


CutReason = str  # "vad" | "force_max" | "tail"


def _samples_to_seconds(n: int) -> float:
    return n / TARGET_RATE


def _seconds_to_samples(s: float) -> int:
    return int(s * TARGET_RATE)


class ChunkingBufferVAD:
    """Same surface as ChunkingBuffer (push/flush returning the same
    types) so pipeline.py can swap implementations behind the
    VAD_CHUNKING flag without other changes."""

    def __init__(
        self,
        source_rate: int,
        target_rate: int = TARGET_RATE,
        chunk_seconds: float = HARD_CUT_S,
        channels: int = 1,
    ):
        self._source_rate = source_rate
        self._target_rate = target_rate
        self._channels = channels
        self._hard_cut_samples = _seconds_to_samples(chunk_seconds)
        self._min_speech_samples = _seconds_to_samples(MIN_SPEECH_S)
        self._natural_pause_samples = _seconds_to_samples(NATURAL_PAUSE_S)
        self._trim_pad_samples = _seconds_to_samples(TRIM_PAD_S)
        self._hard_cut_scan_samples = _seconds_to_samples(HARD_CUT_SCAN_S)

        self._ratecv_state = None
        self._vad = SileroVAD()

        # Per-chunk state — reset after every cut.
        self._chunk_audio: np.ndarray = np.empty(0, dtype=np.float32)
        self._window_probs: List[float] = []  # one entry per 512-sample window
        self._in_speech = False
        self._speech_samples = 0
        # Sample index (relative to chunk start) of the END of the most
        # recent window where we were in_speech. Used as the cut anchor
        # for natural cuts.
        self._last_speech_end_sample: Optional[int] = None
        # Length of the current non-speech run, in samples. Reset to 0
        # whenever we transition back into speech.
        self._silence_run_samples = 0

    def push(self, frame_bytes: bytes) -> List[Tuple[np.ndarray, CutReason]]:
        """Resample new bytes, run silero on every newly-completed
        32 ms window, and emit chunks at natural pauses or hard-cut
        boundaries. Typical PyAudio frame is ~21 ms so most calls
        produce zero chunks — we accumulate until silero says it's a
        good moment, or until HARD_CUT_S is reached."""
        if self._source_rate != self._target_rate:
            resampled, self._ratecv_state = audioop.ratecv(
                frame_bytes,
                SAMPLE_WIDTH,
                self._channels,
                self._source_rate,
                self._target_rate,
                self._ratecv_state,
            )
        else:
            resampled = frame_bytes

        new_audio_f32 = _pcm16_to_float32(bytes(resampled))
        self._chunk_audio = np.concatenate([self._chunk_audio, new_audio_f32])

        emitted: List[Tuple[np.ndarray, CutReason]] = []

        # Process newly-completed VAD windows. silero buffers internally
        # if we hand it less than 512 samples; it returns one window per
        # 512 it manages to assemble.
        for window in self._vad.push_audio(new_audio_f32):
            self._window_probs.append(window.p_speech)
            self._update_speech_state(window.p_speech, window.end_sample)

            cut = self._maybe_cut(window.end_sample)
            if cut is not None:
                cut_audio, cut_reason = cut
                emitted.append((cut_audio, cut_reason))

        return emitted

    def flush(self) -> np.ndarray:
        """Return whatever's in the chunk buffer right now as a tail.
        Caller (pipeline.stop_capture) labels it 'tail' for telemetry.
        Resets state so a re-use of the buffer starts clean (we don't
        currently do that, but cheap to be defensive)."""
        tail = self._chunk_audio
        self._reset_chunk_state()
        self._vad.reset()
        self._ratecv_state = None
        return tail

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_speech_state(self, p: float, window_end_sample: int):
        """Hysteresis state machine. Updates _in_speech, _speech_samples,
        _silence_run_samples, _last_speech_end_sample based on the new
        window's probability."""
        was_speech = self._in_speech
        if p >= ENTER_THRESHOLD:
            self._in_speech = True
        elif p <= LEAVE_THRESHOLD:
            self._in_speech = False
        # else: hold previous state (hysteresis)

        if self._in_speech:
            self._speech_samples += WINDOW_SAMPLES
            self._last_speech_end_sample = window_end_sample
            self._silence_run_samples = 0
        else:
            # Even if we were silent before, this is one more silent
            # window — extend the run.
            self._silence_run_samples += WINDOW_SAMPLES
            if was_speech:
                # Edge: speech → silence. last_speech_end is the END
                # of the last speech window, which is the start of
                # this silence run. Already set during prior window.
                pass

    def _maybe_cut(self, window_end_sample: int) -> Optional[Tuple[np.ndarray, CutReason]]:
        """Decide whether the chunk is ready to emit. Three branches:

        (a) Natural cut: enough speech accumulated AND we're in the
            middle of a long-enough silence run.
        (b) Hard cut: chunk has reached HARD_CUT_S without a natural
            opportunity. Cut at the quietest window in the last
            HARD_CUT_SCAN_S seconds.
        (c) No cut: keep accumulating.
        """
        chunk_len = len(self._chunk_audio)

        # (a) Natural cut.
        if (
            self._speech_samples >= self._min_speech_samples
            and self._silence_run_samples >= self._natural_pause_samples
            and self._last_speech_end_sample is not None
            and not self._in_speech
        ):
            cut_at = min(
                self._last_speech_end_sample + self._trim_pad_samples,
                chunk_len,
            )
            return self._emit_cut(cut_at, "vad")

        # (b) Hard cut.
        if chunk_len >= self._hard_cut_samples:
            cut_at = self._find_quietest_cut(chunk_len)
            return self._emit_cut(cut_at, "force_max")

        # (c) No cut yet.
        return None

    def _find_quietest_cut(self, chunk_len: int) -> int:
        """For hard cuts, locate the window with minimum p_speech in
        the last HARD_CUT_SCAN_S seconds. Cut at that window's END
        sample so the next chunk starts after it. Falls back to the
        chunk end if scan window is empty (shouldn't happen — chunk
        must be >= HARD_CUT_S to reach here, and that's > scan window
        by construction)."""
        scan_windows = self._hard_cut_scan_samples // WINDOW_SAMPLES
        if scan_windows == 0 or len(self._window_probs) == 0:
            return chunk_len

        tail_probs = self._window_probs[-int(scan_windows):]
        local_min_idx = int(np.argmin(tail_probs))
        # Translate window index back to a sample offset. Total
        # windows so far = len(self._window_probs). The chosen window
        # is at offset (len - scan + local_min_idx). Its END sample
        # is (offset + 1) * WINDOW_SAMPLES.
        global_window_idx = len(self._window_probs) - len(tail_probs) + local_min_idx
        cut_at = (global_window_idx + 1) * WINDOW_SAMPLES
        return min(cut_at, chunk_len)

    def _emit_cut(self, cut_at: int, reason: CutReason) -> Tuple[np.ndarray, CutReason]:
        """Slice the chunk at cut_at, keep the remainder for the next
        chunk, reset speech state + silero state. Returns (chunk_audio,
        reason). The remainder may contain trailing silence from the
        current chunk's tail; that's fine — silero re-classifies it
        from a clean state, and it becomes the leading silence of the
        next chunk (which whisper handles via no_speech detection)."""
        chunk = self._chunk_audio[:cut_at].copy()
        remainder = self._chunk_audio[cut_at:].copy()

        log.info(
            "[VAD-CHUNKER] cut at %.2fs (reason=%s, speech=%.2fs, "
            "silence_run=%.2fs, remainder=%.2fs kept for next chunk)",
            _samples_to_seconds(cut_at),
            reason,
            _samples_to_seconds(self._speech_samples),
            _samples_to_seconds(self._silence_run_samples),
            _samples_to_seconds(len(remainder)),
        )

        self._chunk_audio = remainder
        self._reset_chunk_state(keep_audio=True)
        # Reset silero — new chunk = clean acoustic context. The
        # remainder samples will be re-classified by silero on the
        # next push (if any) or stay as the next chunk's lead-in.
        self._vad.reset()
        # If the remainder is non-empty, run silero on it now so the
        # state machine is up to date for the next push() call. This
        # matters when the cut consumed only part of a frame and the
        # remainder already contains a window's worth of audio.
        if len(remainder) >= WINDOW_SAMPLES:
            for window in self._vad.push_audio(remainder):
                self._window_probs.append(window.p_speech)
                self._update_speech_state(window.p_speech, window.end_sample)

        return chunk, reason

    def _reset_chunk_state(self, keep_audio: bool = False):
        """Reset per-chunk counters. With keep_audio=True the
        chunk_audio buffer is preserved (caller already replaced it
        with the post-cut remainder)."""
        if not keep_audio:
            self._chunk_audio = np.empty(0, dtype=np.float32)
        self._window_probs = []
        self._in_speech = False
        self._speech_samples = 0
        self._last_speech_end_sample = None
        self._silence_run_samples = 0


def _pcm16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
