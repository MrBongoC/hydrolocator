"""Live listening: follow a hydrophone stream and report the ship-noise level as it happens.

This is half of "what am I hearing right now". The other half, naming the ship, needs
live ship positions, and the free AIS this project uses (MarineCadastre) is published
months late. With a live AIS feed the same matcher applies; without one, this reports
that something is passing and how loud it is.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Iterator

import numpy as np

RATE = 2000
FRAME_S = 10.0


def listen(stream_url: str, seconds: float) -> Iterator[tuple[float, float]]:
    """Yield (wall-clock time, 20-900 Hz band level in dB, uncalibrated) every ten seconds."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to read the stream")
    cmd = ["ffmpeg", "-v", "error", "-i", stream_url, "-t", str(seconds), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    need = int(RATE * FRAME_S) * 4
    freqs = np.fft.rfftfreq(int(RATE * FRAME_S), 1.0 / RATE)
    band = (freqs >= 20.0) & (freqs < 900.0)
    try:
        while True:
            buf = proc.stdout.read(need)
            if len(buf) < need:
                break
            x = np.frombuffer(buf, dtype=np.float32)
            power = float((np.abs(np.fft.rfft(x * np.hanning(len(x)))[band]) ** 2).sum())
            yield time.time(), 10.0 * np.log10(max(power, 1e-30))
    finally:
        proc.kill()
        proc.wait()


def describe(levels: list[float]) -> str:
    """One word for what the last frame is doing against the session so far."""
    if len(levels) < 3:
        return "listening"
    floor = float(np.percentile(levels, 10))
    excess = levels[-1] - floor
    rising = levels[-1] - levels[-3]
    if excess < 4:
        return "quiet"
    if rising > 2:
        return "vessel noise rising"
    if rising < -2:
        return "vessel noise fading"
    return "vessel noise"
