"""Resampling chunk buffer with fixed-duration cut points.

Accepts raw PyAudio frames at the device rate (typically 48 kHz int16
mono) and emits 30-second ndarray chunks at the target rate (16 kHz
float32 mono). Resampling happens on-the-fly with a persistent
audioop.ratecv state so chunk boundaries don't get IIR-filter restart
artifacts (each ratecv call without preserved state would otherwise
glitch the first few output samples).

v11 reverted to fixed-duration cuts after VAD experiment (v10) made
quality worse. With VAD the chunks ended at natural pauses, and
whisper interpreted that "speech + trailing pause" pattern as
"end of segment / show is over" — triggering a "Продолжение
следует..." hallucination that REPLACED the real content of the
chunk's tail. Fixed-duration cuts produce mid-word boundaries which
look ugly to the human eye but DON'T trigger this hallucination,
so they preserve more real content overall.

Single-threaded by design — recorder thread is the only producer.
flush() is called from the main thread on stop_capture, but only AFTER
the recorder thread has joined.

API note: push() returns List[(ndarray, cut_reason)] for telemetry
parity with the pipeline. cut_reason is always "fixed" for full
chunks; tails are tagged "tail" by the caller in stop_capture.
"""

import audioop
from typing import List, Tuple

import numpy as np

CHUNK_SECONDS = 30
TARGET_RATE = 16000
SAMPLE_WIDTH = 2  # int16

CutReason = str  # "fixed" | "tail"


class ChunkingBuffer:
    def __init__(
        self,
        source_rate: int,
        target_rate: int = TARGET_RATE,
        chunk_seconds: int = CHUNK_SECONDS,
        channels: int = 1,
    ):
        self._source_rate = source_rate
        self._target_rate = target_rate
        self._channels = channels
        self._chunk_threshold_bytes = chunk_seconds * target_rate * SAMPLE_WIDTH
        self._ratecv_state = None
        self._accum = bytearray()

    def push(self, frame_bytes: bytes) -> List[Tuple[np.ndarray, CutReason]]:
        """Resample and accumulate one PyAudio frame. Return any chunks
        that completed at the fixed threshold (typically 0 — frames are
        ~21 ms while chunks are 30 s).
        """
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
        self._accum.extend(resampled)

        chunks: List[Tuple[np.ndarray, CutReason]] = []
        while len(self._accum) >= self._chunk_threshold_bytes:
            chunk_bytes = bytes(self._accum[: self._chunk_threshold_bytes])
            del self._accum[: self._chunk_threshold_bytes]
            chunks.append((_pcm16_to_float32(chunk_bytes), "fixed"))
        return chunks

    def flush(self) -> np.ndarray:
        """Return whatever's left in the accumulator as a tail ndarray."""
        if not self._accum:
            return np.array([], dtype=np.float32)
        tail = _pcm16_to_float32(bytes(self._accum))
        self._accum.clear()
        return tail


def _pcm16_to_float32(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0
