"""Where downloaded and derived data lives. Override with HYDROLOCATOR_DATA."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    root = os.environ.get("HYDROLOCATOR_DATA")
    path = Path(root) if root else Path(__file__).resolve().parents[2] / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_path(*parts: str) -> Path:
    path = data_dir().joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
