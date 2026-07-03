"""Transcription pipeline orchestrator.

v11: fixed-30s chunked end-to-end (VAD reverted from v10). ChunkingBuffer
emits 30-second chunks; a single worker thread transcribes them in
parallel with ongoing recording, threading a 200-char rolling prompt
between chunks ONLY when the previous chunk's text ends with a clean
sentence terminator (.!? but NOT "..." or "…" — those indicate
mid-thought continuation or whisper "show is over" hallucination).
Otherwise the next chunk receives PUNCTUATION_PROMPT — clean restart
that breaks the punctuation-loss cascade observed in v8.

On stop the tail is enqueued, the queue drains, and the per-chunk
texts are joined into the final output. Each chunk text is stripped
of leading whisper-continuation markers ("..."). The full WAV is
still written by the recorder so we can fall back to
transcribe_audio_defaults() if the drain times out or the worker
dropped a chunk.

Threading model
---------------

Three threads cooperate during a capture:

  * main UI thread — drives start(), stop_capture(), cancel(),
    finalize() runs here too (called from app.py's _stop_recording
    worker thread, but conceptually a single foreground caller)
  * recorder thread — owned by Recorder, runs read loop, calls our
    frame_callback for every PyAudio frame
  * worker thread — owned by this pipeline, drains a queue.Queue and
    runs transcribe_chunk on each item

ChunkingBuffer is touched only by the recorder thread (push) and by
the main thread AFTER recorder.stop() has joined the recorder thread
(flush). No locking needed — happens-before is enforced by the join.

queue.Queue is thread-safe by construction. The worker uses
references captured at thread creation, not self._queue / self._results
— so an orphaned worker after cancel() can't pollute a freshly
started pipeline (it writes to its own dead refs).

Telemetry
---------

start() resets diagnostic state. After finalize() the caller
(app.py) reads `last_run_stats` to enrich the JSONL log row with
per-chunk durations / cut reasons / transcribe times / char counts,
plus aggregate counters (rolling resets, finalize path).
"""

import os
import queue
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import pnc
from .chunking import ChunkingBuffer, TARGET_RATE
from .hallucinations import remove_hallucinations
from .recorder import Recorder
from .seams import reconcile_seams, vocabulary_word_set
from .transcriber import transcribe_audio_defaults, transcribe_chunk, clean_hallucinations
from .logging_setup import get_logger

log = get_logger(__name__)

# P&C layer toggle (dev): default ON. Set TELLAR_PNC=0 to compare against raw
# v16 stitching without the re-punctuation pass. Gated to multi-chunk dictations
# in finalize() — single-chunk output has no seams and is left as whisper made it.
PNC_ENABLED = os.environ.get("TELLAR_PNC", "1") != "0"

# Last N chars of chunk N's text become the initial_prompt for chunk N+1.
# Whisper's prompt budget is ~224 tokens (~600-1000 chars depending on
# language), so 200 chars leaves headroom and keeps the most recent
# context — which is what matters for next-chunk continuity.
ROLLING_PROMPT_CHARS = 200

# A full chunk that produces fewer than this many chars is treated as
# degenerate: the whisper output collapsed (lost language, returned a
# placeholder, hit no_speech for clearly speech-bearing audio). The
# pipeline (a) doesn't propagate the chunk's text as rolling prompt
# and (b) flags the run for fallback to a full-WAV transcribe in
# finalize(). Tail chunks are exempt — short tails can legitimately
# produce few chars.
DEGENERATE_CHAR_THRESHOLD = 20

# Worker drain timeout. Realistic worst case: a 5-minute recording
# produces ~10 × ~30s chunks (~25s of GPU work) + tail (~2s). All
# but the tail are typically already done by the time stop_capture
# runs, so the actual wait is dominated by tail transcribe time.
# 30s gives ~10x headroom; if we exceed it, something is genuinely
# stuck and the fallback path is faster anyway.
DRAIN_TIMEOUT_SEC = 30

# Sentence terminators that mark a chunk's text as "clean enough" to
# carry forward as the rolling prompt for the next chunk. Note that
# "..." (three dots) and "…" (single ellipsis char) are NOT in this
# set: they indicate mid-thought continuation OR a whisper "show is
# over" hallucination ("Продолжение следует..."). Carrying that
# forward as prompt poisons the next chunk.
_CLEAN_TERMINATORS = ('.', '!', '?')
_DIRTY_ENDINGS = ('...', '…')

# Whisper sometimes emits leading "..." or "…" at the start of a chunk
# whose audio begins mid-utterance — a visible "I am continuing"
# marker. Strip it defensively before storing chunk text.
_LEADING_CONTINUATION_RE = re.compile(r'^\s*[.…]{2,}\s*')

# Symmetric strip from the END of the chunk text. Whisper's internal
# segmenter marks the last segment of a chunk as "to be continued" and
# emits "..." even when the human didn't pause meaningfully — the
# acoustic finish of the chunk is enough of a cue for the model. With
# fixed-30s cuts it was masked by mid-word boundaries; with VAD cuts
# the chunk ends on a clean phrase, so the model emits the marker more
# often, and it shows up between joined chunks as bogus "fragment...
# next sentence". Apply ONLY to full chunks; tail (the last piece) is
# whatever the user actually said at the end of the dictation, and
# might legitimately end with "..." (rare but possible).
_TRAILING_CONTINUATION_RE = re.compile(r'\s*[.…]{2,}\s*$')

# Debug marker between joined chunks in the final output. When True,
# pipeline.finalize() inserts a visible "⟨✂N: Ds reason⟩" tag at every
# chunk boundary in the assembled text. Lets the user see in the
# pasted/Studio output where chunk seams actually fall — useful for
# diagnosing whether artifacts (e.g. "..." between sentences) are
# emitted at chunk seams or somewhere else entirely. Strip via a
# simple Find & Replace before sending the text anywhere it matters.
# Diagnostic only — flip back to False once the question is settled.
DEBUG_CHUNK_MARKERS = False


class TranscriptionPipeline:
    def __init__(self, recorder: Recorder):
        self._recorder = recorder
        self._wav_path: Optional[str] = None
        self._buffer: Optional[ChunkingBuffer] = None
        self._queue: Optional[queue.Queue] = None
        self._results: Dict[int, str] = {}
        self._degenerate_indices: set = set()
        self._chunk_idx: int = 0
        self._worker_thread: Optional[threading.Thread] = None
        self._last_fallback_used: bool = False

        # Telemetry — populated through start/_on_frame/stop_capture/
        # worker/finalize and read by app.py via last_run_stats.
        self._chunk_durations_s: List[float] = []
        self._chunk_cut_reasons: List[str] = []
        # transcribe_secs and rolling_resets are populated from the
        # worker thread; passed in args (orphan-safe pattern).
        self._chunk_transcribe_secs: Dict[int, float] = {}
        self._worker_counters: Dict[str, int] = {"rolling_resets": 0}
        self._finalize_path: str = ""
        # Whether the P&C layer ran on the last finalize (per-record telemetry).
        self._pnc_applied: bool = False

        # Warm the P&C model off the critical path so the first multi-chunk
        # finalize doesn't stall while a ~266 MB model loads. Idempotent.
        if PNC_ENABLED:
            threading.Thread(target=pnc.warm, daemon=True).start()

    @property
    def wav_path(self) -> Optional[str]:
        return self._wav_path

    @property
    def last_fallback_used(self) -> bool:
        return self._last_fallback_used

    @property
    def last_run_stats(self) -> Dict:
        """Snapshot for telemetry. Only meaningful after finalize().
        All lists are 1:1 with chunk indices except chunk_chars which
        derives from results."""
        chunk_texts = [self._results.get(i, "") for i in range(self._chunk_idx)]
        chunk_chars = [len(t) for t in chunk_texts]
        chunk_t = [round(self._chunk_transcribe_secs.get(i, 0.0), 3) for i in range(self._chunk_idx)]
        return {
            "n_chunks": self._chunk_idx,
            "chunk_durations_s": [round(d, 2) for d in self._chunk_durations_s],
            "chunk_cut_reasons": list(self._chunk_cut_reasons),
            "chunk_transcribe_secs": chunk_t,
            "chunk_chars": chunk_chars,
            # Per-chunk text BEFORE the finalize() join/terminator-drop —
            # the raw seam material. Lets offline tools locate chunk
            # boundaries in the final text and classify join defects
            # (boundary duplication, false capital, missing separator)
            # without re-running the WAV. Privacy-sensitive (it is the
            # transcript), so app.py drops it outside DIAGNOSTIC_MODE.
            "chunk_texts": chunk_texts,
            "degenerate_chunk_indices": sorted(self._degenerate_indices),
            "rolling_prompt_resets": self._worker_counters.get("rolling_resets", 0),
            "finalize_path": self._finalize_path,
            "pnc_applied": self._pnc_applied,
        }

    def start(self):
        self._wav_path = None
        self._buffer = None
        self._chunk_idx = 0
        self._results = {}
        self._degenerate_indices = set()
        self._last_fallback_used = False
        self._chunk_durations_s = []
        self._chunk_cut_reasons = []
        self._chunk_transcribe_secs = {}
        self._worker_counters = {"rolling_resets": 0}
        self._finalize_path = ""
        self._pnc_applied = False
        self._queue = queue.Queue()
        # Capture queue/results/degenerate/transcribe_secs/counters in args
        # so an orphaned worker (cancel + restart) writes to its own dead
        # refs, not the new pipeline's state.
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(self._queue, self._results, self._degenerate_indices,
                  self._chunk_transcribe_secs, self._worker_counters),
            daemon=True,
            name="TranscribePipeline.Worker",
        )
        self._worker_thread.start()
        self._recorder.start(frame_callback=self._on_frame)

    def stop_capture(self) -> str:
        # recorder.stop() joins the recorder thread, so by the time we
        # touch self._buffer below there are no concurrent push() calls.
        path = self._recorder.stop()
        if self._buffer is not None and self._queue is not None:
            tail = self._buffer.flush()
            tail_seconds = len(tail) / TARGET_RATE if len(tail) else 0.0
            if len(tail) > 0:
                # is_full_chunk=False — tail is exempt from the degenerate
                # check (a 3s tail can legitimately produce few chars).
                self._queue.put((self._chunk_idx, tail, False))
                self._chunk_durations_s.append(tail_seconds)
                self._chunk_cut_reasons.append("tail")
                log.info(
                    "Pipeline: enqueued tail chunk %d (%.2fs, qsize=%d)",
                    self._chunk_idx, tail_seconds, self._queue.qsize(),
                )
                self._chunk_idx += 1
            else:
                log.info("Pipeline: no tail (buffer empty on stop)")
            # Sentinel — worker drains remaining items in order, then exits.
            self._queue.put(None)
        self._wav_path = path or None
        return path or ""

    def finalize(self) -> str:
        """Wait for the worker to drain the queue, then assemble chunk
        results into the final text. Falls back to
        transcribe_audio_defaults on the preserved full WAV if the
        drain times out, the result dict has gaps, or any full chunk
        was flagged degenerate.
        """
        drain_succeeded = False
        if self._worker_thread is not None:
            t0 = time.time()
            self._worker_thread.join(timeout=DRAIN_TIMEOUT_SEC)
            drain_seconds = time.time() - t0
            if self._worker_thread.is_alive():
                log.warning(
                    "Pipeline drain timed out after %.1fs (chunks queued=%d, completed=%d)",
                    drain_seconds, self._chunk_idx, len(self._results),
                )
            else:
                log.info(
                    "Pipeline drain complete: %d chunks, drain_wait=%.2fs, result_keys=%s",
                    self._chunk_idx, drain_seconds, sorted(self._results.keys()),
                )
                drain_succeeded = True
            self._worker_thread = None

        if not self._wav_path:
            self._finalize_path = "silence_bail"
            return ""

        if drain_succeeded:
            expected = set(range(self._chunk_idx))
            actual = set(self._results.keys())
            if expected != actual:
                log.warning(
                    "Pipeline result gap (expected %s, got %s) — falling back",
                    sorted(expected), sorted(actual),
                )
            elif self._degenerate_indices:
                log.warning(
                    "Pipeline degenerate chunks %s — falling back",
                    sorted(self._degenerate_indices),
                )
            else:
                ordered = [self._results[i] for i in range(self._chunk_idx)]
                # Reconcile chunk-boundary inconsistencies before joining:
                # spurious terminators, words duplicated across a seam, and
                # false capitals where a mid-sentence pause was rendered as
                # a new sentence. See tellar/seams.py for the full rationale.
                ordered = reconcile_seams(ordered, vocabulary_word_set())
                if DEBUG_CHUNK_MARKERS:
                    parts: List[str] = []
                    for i, text in enumerate(ordered):
                        if text:
                            parts.append(text)
                        # Marker after every chunk except the last —
                        # the last chunk's end IS the end of the dictation.
                        if i < len(ordered) - 1:
                            duration = (
                                self._chunk_durations_s[i]
                                if i < len(self._chunk_durations_s) else 0.0
                            )
                            reason = (
                                self._chunk_cut_reasons[i]
                                if i < len(self._chunk_cut_reasons) else "?"
                            )
                            parts.append(f"⟨✂{i}: {duration:.2f}s {reason}⟩")
                    joined = " ".join(parts)
                else:
                    joined = " ".join(part for part in ordered if part)
                log.info("Pipeline finalize: assembled from %d chunks (%d chars)",
                         self._chunk_idx, len(joined))
                self._finalize_path = "assembled"
                result = remove_hallucinations(clean_hallucinations(joined))
                # P&C layer (variant C): re-derive punctuation/case over the
                # whole text so per-chunk seam formatting cannot survive.
                # Gated to multi-chunk (single chunk has no seams) and skipped
                # under DEBUG_CHUNK_MARKERS (the ⟨✂⟩ markers would confuse it).
                if PNC_ENABLED and not DEBUG_CHUNK_MARKERS and self._chunk_idx > 1:
                    result = pnc.apply_pnc(result)
                    self._pnc_applied = True
                return result

        log.info("Pipeline finalize: using fallback transcribe_audio_defaults")
        self._last_fallback_used = True
        self._finalize_path = "fallback_defaults"
        return transcribe_audio_defaults(self._wav_path)

    def cleanup(self):
        self._stop_worker_if_running()
        self._recorder.cleanup()
        self._wav_path = None

    def cancel(self):
        if self._recorder.is_recording:
            self._recorder.stop()
        self._stop_worker_if_running(drain_queue=True)
        self._recorder.cleanup()
        self._wav_path = None

    def _stop_worker_if_running(self, drain_queue: bool = False):
        if self._worker_thread is None:
            return
        if drain_queue and self._queue is not None:
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
        if self._queue is not None:
            self._queue.put(None)
        self._worker_thread.join(timeout=5)
        if self._worker_thread.is_alive():
            log.warning("Pipeline worker did not exit within 5s; daemonized")
        self._worker_thread = None

    def _on_frame(self, frame_bytes: bytes, source_rate: int):
        """Called from the recorder thread for every PyAudio frame.
        Lazy-init the ChunkingBuffer with the device rate (only known
        after the device is probed inside Recorder.start)."""
        if self._buffer is None:
            from .transcriber import VAD_CHUNKING
            if VAD_CHUNKING:
                from .chunking_vad import ChunkingBufferVAD
                self._buffer = ChunkingBufferVAD(source_rate=source_rate)
                log.info(
                    "[VAD-CHUNKER] Pipeline initialized at %d Hz — "
                    "silero-vad natural-pause cuts ACTIVE (v14)",
                    source_rate,
                )
            else:
                self._buffer = ChunkingBuffer(source_rate=source_rate)
                log.info(
                    "[fixed-30s] Pipeline initialized at %d Hz — "
                    "fixed 30s cuts (v13 baseline)",
                    source_rate,
                )
        for chunk_audio, cut_reason in self._buffer.push(frame_bytes):
            assert self._queue is not None
            duration = len(chunk_audio) / TARGET_RATE
            self._queue.put((self._chunk_idx, chunk_audio, True))
            self._chunk_durations_s.append(duration)
            self._chunk_cut_reasons.append(cut_reason)
            # Visible marker so live-test logs make it obvious which
            # chunker is running — VAD cuts (vad/force_max) vs fixed-30s
            # (fixed). cut_reason "tail" only comes from stop_capture,
            # not this loop, so we don't need to handle it here.
            marker = "VAD" if cut_reason in ("vad", "force_max") else "fixed-30s"
            log.info(
                "Pipeline [%s]: emitted chunk %d (%.2fs, cut=%s, qsize=%d)",
                marker, self._chunk_idx, duration, cut_reason, self._queue.qsize(),
            )
            self._chunk_idx += 1

    @staticmethod
    def _worker_loop(
        q: queue.Queue,
        results: Dict[int, str],
        degenerate_indices: set,
        transcribe_secs: Dict[int, float],
        counters: Dict[str, int],
    ):
        """Single-threaded — mlx-whisper is sequential on the GPU."""
        last_prompt: Optional[str] = None
        while True:
            item = q.get()
            if item is None:
                break
            idx, audio, is_full_chunk = item
            t0 = time.time()
            try:
                text = transcribe_chunk(audio, initial_prompt=last_prompt)
                # Strip whisper's leading "..." continuation marker. With
                # VAD-cut chunks this is rare — chunks start at clean
                # speech onset — but keep the strip as belt-and-suspenders.
                text = _LEADING_CONTINUATION_RE.sub('', text)
                # Symmetric trailing strip — only for full chunks, not
                # tail. Whisper's segmenter routinely emits "..." at the
                # END of the last segment in a chunk as a "to be
                # continued" cue, which on joined chunks shows up as
                # spurious mid-text ellipses (the artifact users see
                # between sentences after VAD cuts).
                if is_full_chunk:
                    text = _TRAILING_CONTINUATION_RE.sub('', text)
                results[idx] = text
                elapsed = time.time() - t0
                transcribe_secs[idx] = elapsed
                log.info(
                    "Pipeline: worker chunk %d done in %.2fs (%d chars)",
                    idx, elapsed, len(text),
                )
                if is_full_chunk and len(text) < DEGENERATE_CHAR_THRESHOLD:
                    degenerate_indices.add(idx)
                    log.warning(
                        "Pipeline: chunk %d degenerate (%d chars < %d) — "
                        "keeping previous rolling prompt, marking for fallback",
                        idx, len(text), DEGENERATE_CHAR_THRESHOLD,
                    )
                    continue
                # Rolling prompt: carry forward the tail of every
                # non-degenerate chunk, regardless of how it ended.
                # Earlier versions (v8-v13) only carried forward when
                # the chunk ended with .!? — defensive measure from
                # the era when temperature-fallback retry cascades
                # could produce poisonous prompts. With v12 hallucinations
                # filter, v13 temperature=0, v14 VAD-cut chunks (clean
                # speech onset) and the trailing "..." strip above,
                # degenerate-cascade risk is low enough that the cost
                # of resetting outweighs the benefit. Cost: when VAD
                # cuts mid-sentence (intra-thought pause), reset means
                # the next chunk has no context — whisper treats it as
                # a fresh utterance, capitalizes the first word, and
                # the joined output reads as two sentences instead of
                # one continuous thought. Carrying the tail through
                # gives whisper enough context to continue lower-case
                # and grammatically agreeing with the previous chunk.
                if text.strip():
                    last_prompt = text[-ROLLING_PROMPT_CHARS:]
                else:
                    last_prompt = None
                    counters["rolling_resets"] = counters.get("rolling_resets", 0) + 1
            except Exception:
                log.exception("Pipeline: worker chunk %d failed", idx)
                degenerate_indices.add(idx)
                transcribe_secs[idx] = time.time() - t0
