import os
import time
import wave
from pathlib import Path

import numpy as np

from .logging_setup import get_logger

log = get_logger(__name__)

MODEL_NAME = "mlx-community/whisper-large-v3-turbo"
MODEL_DIR = Path.home() / "Library" / "Application Support" / "Tellar" / "models"

_model_loaded = False


def get_model():
    global _model_loaded
    if _model_loaded:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    t0 = time.time()
    log.info("Importing mlx_whisper.load_models...")
    from mlx_whisper.load_models import load_model
    log.info("mlx_whisper imported in %.2fs", time.time() - t0)
    t1 = time.time()
    log.info("Loading model %s into memory...", MODEL_NAME)
    load_model(MODEL_NAME)
    log.info("Model loaded in %.2fs", time.time() - t1)
    _model_loaded = True


def model_exists() -> bool:
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        import huggingface_hub
        huggingface_hub.snapshot_download(MODEL_NAME, local_files_only=True)
        return True
    except Exception:
        return False


def _clean_hallucinations(text: str) -> str:
    import re
    match = re.search(r'(.{3,50}?)\1{2,}', text)
    if match:
        log.debug("Trimmed hallucinated repetition at offset %d", match.start())
        text = text[:match.start()].rstrip()
    return text


def _load_wav_mono16k(path: str) -> np.ndarray:
    # Recorder writes 16 kHz mono int16 PCM, so we can hand mlx_whisper a
    # ready-made waveform and skip its ffmpeg subprocess fallback entirely.
    with wave.open(path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_audio(audio_path: str) -> str:
    os.environ["HF_HUB_OFFLINE"] = "1"
    import mlx_whisper
    audio = _load_wav_mono16k(audio_path)
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=MODEL_NAME,
        # Use whisper's default temperature schedule (0.0, 0.2, 0.4, 0.6, 0.8, 1.0).
        # The first pass is greedy at T=0 — same as fixing temperature=0.0 — but if
        # `compression_ratio_threshold` trips on a degenerate output (greedy
        # occasionally drops capitalization/punctuation and becomes repetitive),
        # whisper retries at the next temperature in the schedule. That retry is
        # the recovery path Wizper relies on. We previously pinned T=0.0 thinking
        # it would prevent quality drift, but it just disabled the recovery and
        # the bad first-pass output shipped as-is.
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
    )
    text = result.get("text", "").strip()
    return _clean_hallucinations(text)
