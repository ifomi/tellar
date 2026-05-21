"""Diagnostic dump for remote troubleshooting.

Usage: python -m tellar.app --diag
"""
import os
import sys
from pathlib import Path

from .logging_setup import LOG_FILE
from .transcriber import MODEL_NAME, MODEL_DIR


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("tellar")
    except Exception:
        return "unknown"


def _get_model_size() -> str:
    try:
        import huggingface_hub
        path = huggingface_hub.snapshot_download(MODEL_NAME, local_files_only=True)
        total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        return f"{total / 1024 / 1024:.1f} MB at {path}"
    except Exception as e:
        return f"not cached locally ({type(e).__name__}: {e})"


def _bundle_path() -> str:
    exe = Path(sys.executable).resolve()
    for parent in [exe, *exe.parents]:
        if parent.suffix == ".app":
            return str(parent)
    return "(not running from .app)"


def _tail_log(n: int = 100) -> str:
    if not LOG_FILE.exists():
        return f"(log file not found: {LOG_FILE})"
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception as e:
        return f"(could not read log: {e})"


def print_diag() -> None:
    print("=" * 60)
    print("Tellar diagnostic")
    print("=" * 60)
    print(f"Tellar version : {_get_version()}")
    print(f"Python         : {sys.version.split()[0]} at {sys.executable}")
    print(f"Platform       : {sys.platform}")
    print(f"PID            : {os.getpid()}")
    print(f"Bundle path    : {_bundle_path()}")
    print(f"Model name     : {MODEL_NAME}")
    print(f"Model dir cfg  : {MODEL_DIR}")
    print(f"Model cached   : {_get_model_size()}")
    print(f"Log file       : {LOG_FILE}")
    print()
    print(f"--- last 100 log lines ---")
    print(_tail_log(100))
    print("=" * 60)
