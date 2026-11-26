"""MBARI Pacific Sound: the MARS cabled hydrophone in Monterey Bay (891 m, since 2015).

The `pacific-sound-2khz` bucket holds one decimated 24-bit WAV per UTC day (518 MB), which
is exactly the rate the pipeline works at. Call shape copied from
probes/mbari_pacific_sound.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from ..ingest import lowrate_copy
from ..paths import cache_path

BUCKET = "https://pacific-sound-2khz.s3.amazonaws.com/"
LAT, LON, DEPTH_M = 36.7128, -122.186, 891.0


def day_url(day: date) -> str:
    return f"{BUCKET}{day.year}/{day.month:02d}/MARS-{day:%Y%m%d}T000000Z-2kHz.wav"


def lf_copy(day: date) -> Path:
    """The day's recording as a local 2 kHz FLAC (about a third of the WAV's size)."""
    return lowrate_copy(day_url(day), cache_path("mbari", "lf", f"MARS-{day:%Y%m%d}T000000Z.2k.flac"))
