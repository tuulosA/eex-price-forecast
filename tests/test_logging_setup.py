"""Tests for the CLI logging setup (prune + configure)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from eex_forecast import logging_setup
from eex_forecast.logging_setup import configure_logging, prune_old_logs


def _write(path: Path, *, age_days: float) -> Path:
    path.write_text("x", encoding="utf-8")
    stamp = time.time() - age_days * 86_400
    os.utime(path, (stamp, stamp))
    return path


def test_prune_old_logs_removes_only_stale_log_files(tmp_path: Path) -> None:
    old = _write(tmp_path / "eex_old.log", age_days=30)
    fresh = _write(tmp_path / "eex_fresh.log", age_days=1)
    other = _write(tmp_path / "keep.txt", age_days=30)  # not a .log
    removed = prune_old_logs(tmp_path, retention_days=14)
    assert removed == 1
    assert not old.exists() and fresh.exists() and other.exists()


def test_prune_old_logs_disabled_for_nonpositive_retention(tmp_path: Path) -> None:
    stale = _write(tmp_path / "eex.log", age_days=100)
    assert prune_old_logs(tmp_path, retention_days=0) == 0 and stale.exists()


def test_prune_old_logs_no_directory_is_a_noop(tmp_path: Path) -> None:
    assert prune_old_logs(tmp_path / "missing", retention_days=14) == 0


def test_configure_logging_writes_a_file_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EEX_LOG_TO_FILE", "1")
    monkeypatch.setattr(logging_setup, "_configured", False)  # allow reconfigure in-test
    root = logging.getLogger()
    before = root.handlers[:]
    try:
        path = configure_logging(log_dir=tmp_path)
        assert path is not None and path.exists() and path.name.startswith("eex_")
    finally:  # release the file and restore the root logger for other tests
        for handler in root.handlers[:]:
            if handler not in before:
                handler.close()
                root.removeHandler(handler)


def test_configure_logging_can_disable_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EEX_LOG_TO_FILE", "0")
    monkeypatch.setattr(logging_setup, "_configured", False)
    root = logging.getLogger()
    before = root.handlers[:]
    try:
        assert configure_logging(log_dir=tmp_path) is None
        assert not any(p.suffix == ".log" for p in tmp_path.iterdir())
    finally:
        for handler in root.handlers[:]:
            if handler not in before:
                handler.close()
                root.removeHandler(handler)
