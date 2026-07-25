"""Console + file logging for the ``eex`` CLI.

:func:`configure_logging` sends INFO to the console (as before) and, in addition, writes a timestamped
file under ``logs/`` (e.g. ``logs/eex_2026-07-25_143002.log``) so every run leaves a persistent,
greppable record - handy for the long backfill / tune / forecast commands whose console output scrolls
away. Logs older than ``LOG_RETENTION_DAYS`` are pruned on startup. File logging is best-effort (the
console still works if the directory can't be written) and can be turned off with ``EEX_LOG_TO_FILE=0``
(the test suite sets this, so runs don't litter ``logs/``). Idempotent - repeated calls don't stack
handlers, so the per-command Typer callback can call it freely.

The pure :func:`prune_old_logs` is unit-tested; :func:`configure_logging` is the thin wiring around it.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from eex_forecast.config import LOG_RETENTION_DAYS, LOGS_DIR

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_configured = False


def prune_old_logs(log_dir: Path, retention_days: float, *, now: datetime | None = None) -> int:
    """Delete ``*.log`` files under ``log_dir`` older than ``retention_days``; return how many were
    removed. A no-op for ``retention_days <= 0`` or a missing directory, and it never raises on a file
    that is locked by another process or already gone."""
    if retention_days <= 0 or not log_dir.exists():
        return 0
    cutoff = (now or datetime.now()).timestamp() - retention_days * 86_400
    removed = 0
    for entry in log_dir.glob("*.log"):
        try:
            if entry.is_file() and entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:  # locked by another process, already deleted, etc.
            continue
    return removed


def _file_logging_enabled() -> bool:
    return os.getenv("EEX_LOG_TO_FILE", "1").strip().lower() not in {"0", "false", "no", "off"}


def configure_logging(
    *,
    level: int = logging.INFO,
    log_dir: Path = LOGS_DIR,
    retention_days: float = LOG_RETENTION_DAYS,
) -> Path | None:
    """Attach a console handler and a timestamped file handler to the root logger, once.

    Returns the log file's path, or ``None`` when file logging is disabled (``EEX_LOG_TO_FILE=0``) or the
    file could not be opened. Safe to call from every CLI invocation - it configures only on the first.
    """
    global _configured
    if _configured:
        return None
    _configured = True

    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(console)

    if not _file_logging_enabled():
        return None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        prune_old_logs(log_dir, retention_days)
        log_path = log_dir / f"eex_{datetime.now():%Y-%m-%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")  # UTF-8: avoid Windows cp1252
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(file_handler)
    except OSError:  # file logging is best-effort; the console handler still works
        return None
    return log_path
