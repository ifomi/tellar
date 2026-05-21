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
    _warmup_model()
    _model_loaded = True


def _warmup_model():
    # Cold-start fix: the first transcribe call compiles Metal compute kernels
    # (mel spectrogram, encoder attention, decoder steps) on the GPU. JIT
    # compilation slightly perturbs numerical paths inside greedy decoding,
    # and we observed the first real transcription consistently dropping
    # capitalization and punctuation. Running one warmup pass on noise here
    # forces every kernel to compile before the user records anything.
    #
    # We feed low-amplitude white noise (not silence) and set
    # no_speech_threshold=1.0 so whisper never short-circuits as "no speech"
    # and the full decoder loop actually runs — otherwise decoder kernels
    # wouldn't compile and the warmup would be incomplete.
    t0 = time.time()
    try:
        import mlx_whisper
        rng = np.random.default_rng(0)
        warmup_audio = (rng.standard_normal(16000) * 0.05).astype(np.float32)
        mlx_whisper.transcribe(
            warmup_audio,
            path_or_hf_repo=MODEL_NAME,
            temperature=0.0,
            no_speech_threshold=1.0,
            compression_ratio_threshold=2.0,
            verbose=None,
        )
        log.info("Model warmup completed in %.2fs", time.time() - t0)
    except Exception as e:
        log.warning("Warmup transcribe failed (non-fatal): %s", e)


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
        # Greedy, deterministic decoding. The default schedule is a tuple
        # (0.0, 0.2, 0.4, 0.6, 0.8, 1.0): whisper retries with rising
        # temperature whenever compression_ratio_threshold trips, and the
        # higher-T outputs lose capitalization and punctuation. Pinning to 0
        # means same audio → same text, every run.
        temperature=0.0,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
    )
    text = result.get("text", "").strip()
    return _clean_hallucinations(text)
