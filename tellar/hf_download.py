"""Shared HuggingFace Hub snapshot download with throttled byte-progress.

Both the Whisper model (transcriber.py) and the Studio LLM (studio_llm.py)
download from HF Hub on first launch and want the same progress UX: a single
aggregated percent/MB readout relayed to the overlay, throttled so the Qt main
thread keeps servicing menubar clicks and the permissions poll timer.

This module is the single source of truth for that mechanism. It is model-
agnostic — callers pass a repo id. The subtle bits (the HF tqdm-wrapper
inheritance, the disable=False fix for non-TTY .app bundles, the emit throttle)
live here once instead of being copy-pasted per model.
"""
import os
import time
from typing import Callable, Optional

from .logging_setup import get_logger

log = get_logger(__name__)

# (percent 0..100, MB downloaded, MB total)
ProgressCallback = Optional[Callable[[int, int, int], None]]


def set_hf_offline(offline: bool):
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


def snapshot_exists(repo_id: str) -> bool:
    """True if the repo snapshot is fully present in the local HF cache."""
    # local_files_only=True forces a cache-only lookup regardless of the
    # global offline state — no env manipulation needed here.
    try:
        import huggingface_hub
        huggingface_hub.snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def download_snapshot(repo_id: str, on_progress: ProgressCallback = None):
    """Download the repo snapshot from HF Hub into the default cache.
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
    huggingface_hub.snapshot_download(repo_id, tqdm_class=_AggTqdm)
    log.info("Snapshot download complete for %s in %.2fs", repo_id, time.time() - t0)
