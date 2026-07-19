from __future__ import annotations

from pathlib import Path
from typing import Iterable


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_columns(dataframe, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
