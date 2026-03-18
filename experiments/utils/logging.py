"""Shared loguru setup for experiment scripts."""

from __future__ import annotations

from pathlib import Path

from loguru import logger as lg


def setup_experiment_log(log_path: Path) -> int:
    """Add a file sink for an experiment run and return its sink id.

    The caller can later call ``lg.remove(sink_id)`` to detach the
    file logger once the experiment finishes.
    """
    return lg.add(
        str(log_path),
        level="DEBUG",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<7} | {message}",
    )
