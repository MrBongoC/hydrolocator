"""Small cached HTTP helpers. Everything public here is read in place and cached on disk."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

_TIMEOUT = httpx.Timeout(60.0, connect=20.0)


def download(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Stream url to dest (atomic rename). Skips when dest already exists."""
    if dest.exists() and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(f"{dest.suffix}.{os.getpid()}-{threading.get_ident()}.part")
    with httpx.stream("GET", url, timeout=_TIMEOUT, follow_redirects=True) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    return dest


def download_many(pairs: list[tuple[str, Path]], workers: int = 16) -> list[Path]:
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda p: download(*p), pairs))


def get_json(url: str, params: dict | None = None) -> dict:
    r = httpx.get(url, params=params, timeout=_TIMEOUT, follow_redirects=True)
    r.raise_for_status()
    return r.json()
