import pyaudio
import wave
import tempfile
import threading
import struct
import audioop
from pathlib import Path
from typing import Callable, Optional

from .logging_setup import get_logger

log = get_logger(__name__)

SILENCE_THRESHOLD = 500
MIN_SPEECH_CHUNKS = 3
TARGET_RATE = 16000

# Per-frame callback signature: (frame_bytes, source_rate). Source rate is
# passed because it's only known after the device is probed inside start();
# the consumer (chunking pipeline) needs it to construct its resampler.
FrameCallback = Callable[[bytes, int], None]


class Recorder:
    def __init__(self):
        self.is_recording = False
        self._frames = []
        self._stream = None
        self._pa = None
        self._thread = None
        self._tmp_path = None
        self._device_rate = TARGET_RATE
        self.channels = 1
        self.chunk = 1024
        self._frame_callback: Optional[FrameCallback] = None

    def _find_input_device(self):
        info = self._pa.get_default_input_device_info()
        return int(info['index']), int(info['defaultSampleRate'])

    def start(self, frame_callback: Optional[FrameCallback] = None):
        if self.is_recording:
            return
        self.is_recording = True
        self._frames = []
        self._frame_callback = frame_callback
        self._pa = pyaudio.PyAudio()
        dev_index, self._device_rate = self._find_input_device()
        try:
            info = self._pa.get_device_info_by_index(dev_index)
            log.info("Audio input: %s (idx=%d, rate=%d Hz)",
                     info.get('name', '?'), dev_index, self._device_rate)
        except Exception:
            log.warning("Could not query audio device info for idx=%d", dev_index)
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self._device_rate,
            input=True,
            input_device_index=dev_index,
            frames_per_buffer=self.chunk,
        )
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()

    def _record_loop(self):
        while self.is_recording:
            try:
                data = self._stream.read(self.chunk, exception_on_overflow=False)
                self._frames.append(data)
                if self._frame_callback is not None:
                    try:
                        self._frame_callback(data, self._device_rate)
                    except Exception:
                        # A buggy callback must not kill capture — fallback
                        # path needs the WAV to keep accumulating.
                        log.exception("frame_callback raised")
                samples = struct.unpack(f"<{len(data)//2}h", data)
                self._amplitude = (sum(s * s for s in samples) / len(samples)) ** 0.5
            except Exception:
                log.exception("Recorder loop error, terminating capture")
                break

    @property
    def amplitude(self) -> float:
        return getattr(self, '_amplitude', 0.0)

    def _is_silent(self, frame: bytes) -> bool:
        samples = struct.unpack(f"<{len(frame)//2}h", frame)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return rms < SILENCE_THRESHOLD

    def _trim_silence(self):
        start = 0
        for i, frame in enumerate(self._frames):
            if not self._is_silent(frame):
                start = max(0, i - 1)
                break
        end = len(self._frames)
        for i in range(len(self._frames) - 1, -1, -1):
            if not self._is_silent(self._frames[i]):
                end = min(len(self._frames), i + 4)
                break
        self._frames = self._frames[start:end]

    def stop(self) -> str:
        self.is_recording = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()

        if not self._frames:
            log.info("Recorder.stop: no frames captured")
            return ""

        frames_before = len(self._frames)
        self._trim_silence()
        log.debug("Silence trim: %d → %d chunks", frames_before, len(self._frames))

        if len(self._frames) < MIN_SPEECH_CHUNKS:
            log.info("Recorder.stop: below MIN_SPEECH_CHUNKS after trim, dropping")
            self._frames = []
            return ""

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self._tmp_path = tmp.name
        raw = b"".join(self._frames)
        if self._device_rate != TARGET_RATE:
            raw, _ = audioop.ratecv(raw, 2, self.channels, self._device_rate, TARGET_RATE, None)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_RATE)
            wf.writeframes(raw)
        self._frames = []
        return tmp.name

    def cleanup(self):
        if self._tmp_path and Path(self._tmp_path).exists():
            Path(self._tmp_path).unlink()
            self._tmp_path = None
