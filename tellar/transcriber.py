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
                    on_progress(
                        pct,
                        self.n // (1024 * 1024),
                        self.total // (1024 * 1024),
                    )
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
    _set_hf_offline(True)
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
