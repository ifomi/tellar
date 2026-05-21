import logging
import logging.handlers
import os
import sys
from pathlib import Path

LOG_DIR = Path.home() / "Library" / "Logs" / "Tellar"
LOG_FILE = LOG_DIR / "tellar.log"
MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 3

_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logger with rotating file + stderr handlers.

    Idempotent — safe to call multiple times.
    Returns the 'tellar' logger for convenience.
    """
    global _configured
    logger = logging.getLogger("tellar")
    if _configured:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)

    logger.info("=" * 60)
    logger.info("Tellar starting | pid=%s | python=%s", os.getpid(), sys.version.split()[0])
    logger.info("Log file: %s", LOG_FILE)

    _configured = True
    return logger


def get_logger(name: str = "tellar") -> logging.Logger:
    return logging.getLogger(name)
