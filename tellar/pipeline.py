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

import queue
import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chunking import ChunkingBuffer, TARGET_RATE
from .hallucinations import remove_hallucinations
from .recorder import Recorder
from .transcriber import transcribe_audio_defaults, transcribe_chunk, clean_hallucinations
from .logging_setup import get_logger

log = get_logger(__name__)

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
        chunk_chars = [len(self._results.get(i, "")) for i in range(self._chunk_idx)]
        chunk_t = [round(self._chunk_transcribe_secs.get(i, 0.0), 3) for i in range(self._chunk_idx)]
        return {
            "n_chunks": self._chunk_idx,
            "chunk_durations_s": [round(d, 2) for d in self._chunk_durations_s],
            "chunk_cut_reasons": list(self._chunk_cut_reasons),
            "chunk_transcribe_secs": chunk_t,
            "chunk_chars": chunk_chars,
            "degenerate_chunk_indices": sorted(self._degenerate_indices),
            "rolling_prompt_resets": self._worker_counters.get("rolling_resets", 0),
            "finalize_path": self._finalize_path,
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
                joined = " ".join(part for part in ordered if part)
                log.info("Pipeline finalize: assembled from %d chunks (%d chars)",
                         self._chunk_idx, len(joined))
                self._finalize_path = "assembled"
                return remove_hallucinations(clean_hallucinations(joined))

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
            self._buffer = ChunkingBuffer(source_rate=source_rate)
            log.info("Pipeline: ChunkingBuffer initialized at %d Hz", source_rate)
        for chunk_audio, cut_reason in self._buffer.push(frame_bytes):
            assert self._queue is not None
            duration = len(chunk_audio) / TARGET_RATE
            self._queue.put((self._chunk_idx, chunk_audio, True))
            self._chunk_durations_s.append(duration)
            self._chunk_cut_reasons.append(cut_reason)
            log.info(
                "Pipeline: emitted chunk %d (%.2fs, cut=%s, qsize=%d)",
                self._chunk_idx, duration, cut_reason, self._queue.qsize(),
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
                # Smart rolling prompt: only carry forward if the chunk
                # text ends with a real sentence terminator AND is not a
                # whisper "..." / "…" hallucination. Otherwise reset so
                # the next chunk gets a clean PUNCTUATION_PROMPT instead
                # of inheriting mid-thought / hallucinated context.
                stripped = text.rstrip()
                clean_ending = (
                    stripped
                    and not stripped.endswith(_DIRTY_ENDINGS)
                    and stripped.endswith(_CLEAN_TERMINATORS)
                )
                if clean_ending:
                    last_prompt = text[-ROLLING_PROMPT_CHARS:]
                else:
                    if last_prompt is not None:
                        log.info(
                            "Pipeline: chunk %d ends without clean terminator — "
                            "resetting rolling prompt for chunk %d",
                            idx, idx + 1,
                        )
                    last_prompt = None
                    counters["rolling_resets"] = counters.get("rolling_resets", 0) + 1
            except Exception:
                log.exception("Pipeline: worker chunk %d failed", idx)
                degenerate_indices.add(idx)
                transcribe_secs[idx] = time.time() - t0
