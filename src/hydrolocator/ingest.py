"""Audio ingest: any file or URL ffmpeg can read -> a 2 kHz mono copy -> spectra.

Ship noise that matters for matching sits below 1 kHz, so everything downstream works
from a 2 kHz copy (a 24 h clip is about 340 MB of FLAC). ffmpeg reads remote objects in
place, so public archives never need to be downloaded at full rate.

Nothing here needs calibration: the matcher only uses level relative to the clip's own
floor, which is what makes arbitrary uploads workable.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

LF_RATE = 2000

_STAMPS = [
    # ONC: ICLISTENHF1251_20180101T000000.000Z.wav ; MARS: MARS-20210601T000000Z-2kHz.wav ;
    # SanctSound (newer): SanctSound_SB01_01_1678032935_20181113T002444Z.flac
    (re.compile(r"(\d{8}T\d{6})(?:\.\d+)?Z"), "%Y%m%dT%H%M%S"),
    (re.compile(r"(\d{8}_\d{6})"), "%Y%m%d_%H%M%S"),
    # SanctSound (older) and SoundTrap default: ..._200224055255.flac (yymmddHHMMSS)
    (re.compile(r"(?<!\d)(\d{12})(?!\d)"), "%y%m%d%H%M%S"),
    (re.compile(r"(\d{4}-\d{2}-\d{2}[T_ ]\d{2}[-:]\d{2}[-:]\d{2})"), None),
]


def start_time_from_name(name: str) -> float | None:
    """UTC start time encoded in a recording's filename, as POSIX seconds."""
    base = Path(name).name
    for pattern, fmt in _STAMPS:
        m = pattern.search(base)
        if not m:
            continue
        text = m.group(1)
        try:
            if fmt is None:
                text = re.sub(r"[T_ ]", "T", text, count=1)
                date, clock = text.split("T")
                dt = datetime.fromisoformat(f"{date}T{clock.replace('-', ':')}")
            else:
                dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if 1990 <= dt.year <= 2100:
            return dt.replace(tzinfo=UTC).timestamp()
    return None


def lowrate_copy(src: str | Path, dest: Path, rate: int = LF_RATE) -> Path:
    """Decode `src` (path or URL), mix to mono, resample to `rate`, write 24-bit FLAC."""
    if dest.exists():
        return dest
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to decode audio")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part.flac")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(rate),
           "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", str(tmp)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed on {src}: {done.stderr.strip()[:400]}")
    tmp.rename(dest)
    return dest


def read_audio(path: Path, start_s: float = 0.0, duration_s: float | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 samples and the sample rate."""
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        f.seek(int(start_s * sr))
        frames = -1 if duration_s is None else int(duration_s * sr)
        x = f.read(frames, dtype="float32", always_2d=True)
    return x.mean(axis=1) if x.shape[1] > 1 else x[:, 0], sr


@dataclass
class Spectra:
    """Averaged power spectra: one row per `dt` seconds, 1 Hz bins. Levels are dB relative
    to full scale, not calibrated."""
    t: np.ndarray            # frame start, POSIX s (or seconds from file start if unknown)
    freqs: np.ndarray
    psd_db: np.ndarray       # (T, F) float32
    dt: float

    def band(self, lo: float = 20.0, hi: float = 900.0) -> np.ndarray:
        m = (self.freqs >= lo) & (self.freqs < hi)
        return 10.0 * np.log10(np.maximum((10.0 ** (self.psd_db[:, m] / 10.0)).sum(axis=1), 1e-30))


def spectra(x: np.ndarray, sr: int, t_start: float = 0.0, *, dt: float = 10.0, fmax: float = 1000.0,
            align: bool = True) -> Spectra:
    """Welch spectra with 1 s Hann segments at 50 % overlap, averaged over `dt` seconds.
    With `align`, frames start on whole multiples of dt so files tile onto one grid."""
    if align:
        skip = (-t_start) % dt
        x = x[int(round(skip * sr)):]
        t_start = t_start + skip
    nseg, hop = sr, sr // 2
    per_frame = int(dt * sr)
    n_frames = len(x) // per_frame
    freqs = np.fft.rfftfreq(nseg, 1.0 / sr)
    keep = freqs <= fmax
    window = np.hanning(nseg).astype(np.float32)
    scale = 2.0 / (sr * float((window ** 2).sum()))
    out = np.empty((n_frames, int(keep.sum())), dtype=np.float32)
    segs_per_frame = (per_frame - nseg) // hop + 1
    idx = (np.arange(segs_per_frame) * hop)[:, None] + np.arange(nseg)[None, :]
    block = 64                                                   # frames per FFT batch
    for a in range(0, n_frames, block):
        b = min(n_frames, a + block)
        chunk = x[a * per_frame:b * per_frame].reshape(b - a, per_frame)
        seg = chunk[:, idx] * window                             # (frames, segs, nseg)
        power = (np.abs(np.fft.rfft(seg, axis=2)[:, :, keep]) ** 2).mean(axis=1) * scale
        out[a:b] = 10.0 * np.log10(np.maximum(power, 1e-30))
    return Spectra(t_start + dt * np.arange(n_frames), freqs[keep], out, dt)


def stft_db(x: np.ndarray, sr: int, *, nfft: int = 16384, hop: int = 2048, fmax: float = 600.0
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """High-resolution spectrogram for tonal tracking: (frame centre s, freqs Hz, dB (T, F)).
    The default gives 0.12 Hz bins and about 1 s hops at 2 kHz."""
    n = (len(x) - nfft) // hop + 1
    if n <= 0:
        return np.array([]), np.array([]), np.zeros((0, 0), dtype=np.float32)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    keep = freqs <= fmax
    window = np.hanning(nfft).astype(np.float32)
    out = np.empty((n, int(keep.sum())), dtype=np.float32)
    for a in range(0, n, 256):
        b = min(n, a + 256)
        idx = (np.arange(a, b) * hop)[:, None] + np.arange(nfft)[None, :]
        out[a:b] = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(x[idx] * window, axis=1)[:, keep]), 1e-12))
    return (np.arange(n) * hop + nfft / 2) / sr, freqs[keep], out
