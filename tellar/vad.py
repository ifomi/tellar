"""Streaming silero-vad wrapper.

Wraps the silero_vad_16k_op15.onnx model bundled next to this module.
Audio comes in as 16 kHz mono float32; the wrapper emits per-window
speech probabilities (one per 512-sample / 32 ms window) on demand.
ChunkingBufferVAD consumes these to decide when to cut chunks at
natural pauses without relying on RMS thresholds — the core fix for
the v10 RMS-VAD failure mode where any background noise prevented
silence detection.

ONNX graph signature (silero op15, 16 kHz only)
----------------------------------------------
inputs:
    input: float32 [1, 576]   (64-sample context + 512-sample window)
    state: float32 [2, 1, 128]  LSTM h+c, threaded across calls
    sr:    int64   scalar = 16000
outputs:
    out:   float32 [1, 1]   p(speech) for the window
    state: float32 [2, 1, 128]   updated LSTM state

The 64-sample context is the tail of the PREVIOUS window, prepended
so the model has acoustic continuity across window boundaries. For
the very first window after reset(), context is zeros — model still
works (silero is robust to that), but the first probability is less
reliable. Chunker should look at multiple windows before committing
to a state, not the first one.

Threading: ORT session is created with intra/inter_op=1. Whisper runs
on Metal GPU so it doesn't compete; silero CPU work is well under 1ms
per window, irrelevant in the recorder thread context.

Lazy load: import onnxruntime + create the InferenceSession on first
push_audio call. Loading takes ~50ms; doing it eagerly at app start
would slow Tellar's cold-start unnecessarily — VAD is only needed
during recording, and the first chunk doesn't need to land within the
first 32ms of capture.
"""

from pathlib import Path
from typing import List, NamedTuple, Optional

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)


SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512  # 32 ms at 16 kHz
CONTEXT_SAMPLES = 64  # silero op15 spec
STATE_SHAPE = (2, 1, 128)

MODEL_PATH = Path(__file__).parent / "silero_vad.onnx"


class VADWindow(NamedTuple):
    """One 32 ms (512-sample) decision window. start/end are sample
    offsets relative to the first audio pushed after reset()."""
    start_sample: int
    end_sample: int
    p_speech: float


class SileroVAD:
    """Streaming silero-vad. Push audio in arbitrary-sized chunks; get
    a list of completed 32 ms decisions back. State (LSTM + tail
    context) persists across pushes until reset()."""

    def __init__(self):
        self._session = None
        self._state: Optional[np.ndarray] = None
        self._context: Optional[np.ndarray] = None
        self._buffer = np.empty(0, dtype=np.float32)
        self._samples_processed = 0
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def _ensure_loaded(self):
        if self._session is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"silero VAD model not found at {MODEL_PATH} — "
                "build.sh should have copied it next to vad.py"
            )
        # Import here so a turn-off of VAD_CHUNKING never pays the
        # onnxruntime import cost, even if vad.py is imported transitively.
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        # CPUExecutionProvider explicitly — prevents future onnxruntime
        # builds from auto-binding to CoreML/Metal (would compete with
        # whisper). Silero on CPU is sub-millisecond, so no benefit.
        self._session = ort.InferenceSession(
            str(MODEL_PATH),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.reset()
        log.info("Silero VAD loaded from %s", MODEL_PATH)

    def reset(self):
        """Zero out LSTM state and rolling context. Call between
        unrelated audio streams (e.g. between recordings, or when
        starting a new chunk after a hard cut)."""
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self._buffer = np.empty(0, dtype=np.float32)
        self._samples_processed = 0

    def push_audio(self, audio: np.ndarray) -> List[VADWindow]:
        """Append audio (float32 mono 16 kHz) to the buffer and run
        silero on every complete 512-sample window. Trailing samples
        smaller than 512 are kept for the next push.
        """
        self._ensure_loaded()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self._buffer = np.concatenate([self._buffer, audio])

        windows: List[VADWindow] = []
        while len(self._buffer) >= WINDOW_SAMPLES:
            window = self._buffer[:WINDOW_SAMPLES]
            self._buffer = self._buffer[WINDOW_SAMPLES:]

            # silero expects 64-sample context prepended to the 512-sample
            # window for acoustic continuity. Reshape to (1, 576).
            inp = np.concatenate([self._context[0], window])[np.newaxis, :].astype(np.float32)
            ort_inputs = {
                "input": inp,
                "state": self._state,
                "sr": self._sr,
            }
            out, new_state = self._session.run(None, ort_inputs)
            p = float(out[0][0])
            self._state = new_state
            # Tail of THIS window becomes context for the next call.
            self._context = window[-CONTEXT_SAMPLES:][np.newaxis, :].copy()

            windows.append(VADWindow(
                start_sample=self._samples_processed,
                end_sample=self._samples_processed + WINDOW_SAMPLES,
                p_speech=p,
            ))
            self._samples_processed += WINDOW_SAMPLES

        return windows

    @property
    def samples_processed(self) -> int:
        """Total samples that have been classified since last reset().
        Equals (number of windows emitted) * 512."""
        return self._samples_processed

    @property
    def buffered_samples(self) -> int:
        """Samples received but not yet classified (<512 until the
        next push_audio crosses the threshold)."""
        return len(self._buffer)
