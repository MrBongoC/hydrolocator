"""Passage characterisation from interference fringes (the "bathtub").

A passing ship's direct and reflected paths interfere, striping the spectrogram. Each
stripe sits at a frequency proportional to range, so through a straight pass every stripe
follows the same curve

    f(t) = f_cpa * sqrt(1 + ((t - t0) / tau)^2),        tau = r_cpa / v

On a log-frequency axis that is one shared vertical shift per instant. Undo the shift for
the right (t0, tau) and the stripes become horizontal lines; average over time and they
add up. For a wrong (t0, tau) they smear out. So we search (t0, tau) for the sharpest
time-averaged spectrum. This needs no tonals and no calibration, and it times the
closest approach from the pattern's symmetry instead of the flat top of the level peak.

Doppler on engine tonals would add speed on its own; that is not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

MIN_SHARPNESS = 1.6


@dataclass
class Fringe:
    t0: float              # closest approach, POSIX s
    tau: float             # r_cpa / v, seconds
    sharpness: float       # stripe contrast at the optimum over a typical misaligned stack
    t0_sigma: float        # half-width of the region within 10 % of the optimum, seconds
    reliable: bool         # False when the optimum sits on the edge of the search or is weak
    surface: np.ndarray    # contrast over (t0, tau), for inspection
    t0_grid: np.ndarray
    tau_grid: np.ndarray


def fit_fringes(t: np.ndarray, freqs: np.ndarray, psd_db: np.ndarray, t_guess: float, *,
                span_s: float = 420.0, step_s: float = 10.0, tau_range: tuple[float, float] = (90.0, 2400.0),
                band: tuple[float, float] = (40.0, 450.0), window_s: float = 900.0, n_log: int = 360) -> Fringe:
    """Estimate (t0, tau) for the passage nearest t_guess from spectra (T, F) with frame
    start times t and linear frequency bins."""
    dt = float(np.median(np.diff(t)))
    keep = (t + dt / 2 >= t_guess - window_s - span_s) & (t + dt / 2 <= t_guess + window_s + span_s)
    tc = t[keep] + dt / 2
    fsel = (freqs >= band[0] * 0.5) & (freqs <= band[1] * 1.6)
    f = freqs[fsel]
    spec = np.asarray(psd_db[keep][:, fsel], dtype=np.float64)
    spec = np.where(np.isfinite(spec), spec, np.nanmedian(spec))
    # Flatten the ship's own spectrum and the level swell so only the stripes remain:
    # remove each frame's smooth trend along frequency, then each bin's mean over time.
    spec = spec - uniform_filter1d(spec, size=max(5, len(f) // 8), axis=1, mode="nearest")
    spec = spec - spec.mean(axis=0, keepdims=True)
    u = np.linspace(np.log(band[0]), np.log(band[1]), n_log)
    logf = np.log(f)

    t0_grid = np.arange(t_guess - span_s, t_guess + span_s + 1e-6, step_s)
    tau_grid = np.geomspace(tau_range[0], tau_range[1], 28)
    surface = np.zeros((len(t0_grid), len(tau_grid)))
    for i, t0 in enumerate(t0_grid):
        near = np.abs(tc - t0) <= window_s
        frames = spec[near]
        x = tc[near] - t0
        for j, tau in enumerate(tau_grid):
            use = np.abs(x) <= min(window_s, 2.5 * tau)
            if use.sum() < 12:
                continue
            shift = 0.5 * np.log1p((x[use] / tau) ** 2)
            stacked = np.zeros(n_log)
            for row, s in zip(frames[use], shift):
                stacked += np.interp(u + s, logf, row)
            stacked /= use.sum()
            # Aligned stripes survive the average; noise averages down as 1/sqrt(n).
            surface[i, j] = stacked.std() * np.sqrt(use.sum())
    i, j = np.unravel_index(int(np.argmax(surface)), surface.shape)
    best = surface[i, j]
    close = np.nonzero(surface[:, j] >= best - 0.1 * (best - float(np.median(surface))))[0]
    half = max(step_s, (close.max() - close.min()) * step_s / 2) if len(close) else step_s
    # Contrast of the aligned stripes against a typical misaligned stack. Pure noise gives
    # about 1.2 however it is shifted; real fringes give well over that.
    sharpness = float(best / max(float(np.median(surface[surface > 0])) if (surface > 0).any() else 1.0, 1e-9))
    # With no warp at all (huge tau) steady lines in the noise also "align", so an optimum
    # on the boundary of the search says nothing. Only an interior, well-marked peak counts.
    interior = 0 < i < len(t0_grid) - 1 and 0 < j < len(tau_grid) - 1
    return Fringe(float(t0_grid[i]), float(tau_grid[j]), sharpness, float(half),
                  bool(interior and sharpness >= MIN_SHARPNESS), surface, t0_grid, tau_grid)
