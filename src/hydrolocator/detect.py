"""Passage detection: band-level time series -> ship passages.

Works on any spectrogram-like input (1-minute archive spectra or spectra computed from
audio). A passing ship lifts the 20-1,000 Hz level into a peak whose top sits near the
closest point of approach and whose width scales with CPA range over speed (r/v).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import find_peaks

from .events import Observed

BAND = (20.0, 1000.0)


def band_level(psd_db: np.ndarray, freqs: np.ndarray, lo: float = BAND[0], hi: float = BAND[1]) -> np.ndarray:
    """Integrate a PSD (dB re 1 uPa^2/Hz, shape (T, F)) over [lo, hi) -> band level in dB."""
    freqs = np.asarray(freqs, dtype=np.float64)
    edges = np.r_[freqs[0] - (freqs[1] - freqs[0]) / 2, (freqs[1:] + freqs[:-1]) / 2,
                  freqs[-1] + (freqs[-1] - freqs[-2]) / 2]
    m = (freqs >= lo) & (freqs < hi)
    power = (10.0 ** (np.asarray(psd_db, dtype=np.float64)[:, m] / 10.0) * np.diff(edges)[m]).sum(axis=1)
    return 10.0 * np.log10(np.maximum(power, 1e-30))


def detect_passages(t: np.ndarray, level_db: np.ndarray, *, min_prominence: float = 3.0,
                    smooth_s: float = 180.0, min_gap_s: float = 240.0,
                    baseline_s: float = 4 * 3600.0) -> Observed:
    """Find passages in a band-level series sampled at times t (POSIX s, uniform).

    Returns events with: t (centre of the top 3 dB of the peak), sigma (timing
    uncertainty, growing with the peak's width), quality (from prominence), and
    meta["tau"] (half-width of the top 3 dB, an estimate of r/v) and meta["prominence"].
    """
    t = np.asarray(t, dtype=np.float64)
    level = np.asarray(level_db, dtype=np.float64)
    dt = float(np.median(np.diff(t)))
    good = np.isfinite(level)
    if good.sum() < 10:
        return Observed(t[0], t[-1] + dt, np.array([]), np.array([]), meta={"tau": np.array([]), "prominence": np.array([])})
    level = np.where(good, level, np.nanmedian(level[good]))
    smooth = uniform_filter1d(level, max(1, int(round(smooth_s / dt))), mode="nearest")
    # Slow-moving floor (wind, distant shipping): a low percentile over a few hours.
    floor = percentile_filter(smooth, 10, size=max(3, int(round(baseline_s / dt))), mode="nearest")
    excess = smooth - floor
    peaks, props = find_peaks(excess, prominence=min_prominence, distance=max(1, int(round(min_gap_s / dt))))
    times, taus = [], []
    power = 10.0 ** (smooth / 10.0)
    for k in peaks:
        # Walk out to where the level has dropped 3 dB from the peak.
        lo = k
        while lo > 0 and smooth[lo - 1] > smooth[k] - 3.0 and smooth[lo - 1] <= smooth[k]:
            lo -= 1
        hi = k
        while hi < len(smooth) - 1 and smooth[hi + 1] > smooth[k] - 3.0 and smooth[hi + 1] <= smooth[k]:
            hi += 1
        w = power[lo:hi + 1] - 0.5 * power[k]
        w = np.maximum(w, 0.0)
        times.append(float((t[lo:hi + 1] * w).sum() / w.sum()) + dt / 2)
        taus.append(max(dt, (hi - lo + 1) * dt) / 2)
    times, taus = np.array(times), np.array(taus)
    prom = props["prominences"] if len(peaks) else np.array([])
    sigma = np.hypot(dt / 2 + 30.0, 0.3 * taus) if len(peaks) else np.array([])
    quality = 1.0 - np.exp(-prom / 4.0)
    return Observed(float(t[0]), float(t[-1] + dt), times, sigma, quality=quality,
                    meta={"tau": taus, "prominence": prom})
