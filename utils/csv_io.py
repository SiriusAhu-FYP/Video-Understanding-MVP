"""CSV read/write helpers shared across experiment scripts."""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path


def init_csv(csv_path: Path, headers: list[str]) -> None:
    """Write CSV header row, creating the file."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(headers)


def append_csv(csv_path: Path, row: list[object]) -> None:
    """Append a single data row to an existing CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def read_csv_dicts(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts (one per row)."""
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
