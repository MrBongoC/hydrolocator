"""Predicted soundscape: the ship-noise level curve AIS says a position should have heard.

This is the ENF-style matcher. Instead of reducing a clip to discrete passages, compare
the whole observed level curve with the curve predicted for each candidate. Every vessel
contributes source power (a function of length and speed) spread over range; the sum,
over a noise floor, is the predicted excess level in dB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d

from .ais import Tracks
from .geo import KNOT


@dataclass
class LevelParams:
    """Defaults were fit on training days across all SanctSound regions (see blindtest.fit),
    starting from textbook values. They landed near them, which is some reassurance that
    the model is describing propagation and not just the training sites."""
    spreading: float = 19.1        # dB per decade of range
    absorption: float = 0.78       # extra dB per km (bottom loss in shallow water)
    length_db: float = 14.0        # dB per decade of vessel length
    speed_db: float = 19.3         # dB per decade of speed
    floor_km: float = 5.15         # a 50 m ship at 10 kn at this range equals the noise floor
    min_range: float = 150.0       # m; also stands in for water depth under the keel
    min_speed_kn: float = 1.5      # below this a vessel is treated as idling at this speed
    idle_db: float = -15.0         # level of a vessel that is not making way
    length_class_a: float = 40.0
    length_class_b: float = 12.0

    def to_dict(self) -> dict:
        return asdict(self)


def vessel_positions(tracks: Tracks, tgrid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate every vessel onto tgrid. Returns (vessel index, x, y, speed) with shape
    (V, T); positions are NaN where the vessel has no track (before, after, or in a gap)."""
    V, T = tracks.n_vessels, len(tgrid)
    x = np.full((V, T), np.nan)
    y = np.full((V, T), np.nan)
    spd = np.zeros((V, T))
    if tracks.n_legs == 0:
        return np.arange(V), x, y, spd
    bounds = np.r_[0, np.nonzero(np.diff(tracks.vid))[0] + 1, tracks.n_legs]
    for a, b in zip(bounds[:-1], bounds[1:]):
        v = tracks.vid[a]
        t0, t1 = tracks.t0[a:b], tracks.t1[a:b]
        k = np.searchsorted(t0, tgrid, side="right") - 1          # leg containing each grid time
        ok = (k >= 0) & (tgrid <= t1[np.clip(k, 0, b - a - 1)])
        kk = k[ok]
        u = (tgrid[ok] - t0[kk]) / (t1[kk] - t0[kk])
        x0, y0, x1, y1 = tracks.x0[a:b][kk], tracks.y0[a:b][kk], tracks.x1[a:b][kk], tracks.y1[a:b][kk]
        x[v, ok] = x0 + u * (x1 - x0)
        y[v, ok] = y0 + u * (y1 - y0)
        spd[v, ok] = np.hypot(x1 - x0, y1 - y0) / (t1[kk] - t0[kk])
    return np.arange(V), x, y, spd


def source_db(length: np.ndarray, tclass: np.ndarray, speed: np.ndarray, params: LevelParams) -> np.ndarray:
    """Relative source level (dB) for vessels (V,) at speeds (V, T). 0 dB = 50 m at 10 kn."""
    fallback = np.where(np.asarray(tclass) == "B", params.length_class_b, params.length_class_a)
    length = np.asarray(length, dtype=np.float64)
    length = np.where(np.isfinite(length) & (length > 2), length, fallback)
    moving = speed >= params.min_speed_kn * KNOT
    sp = params.speed_db * np.log10(np.maximum(speed, params.min_speed_kn * KNOT) / (10 * KNOT))
    return params.length_db * np.log10(length / 50.0)[:, None] + np.where(moving, sp, params.idle_db)


def ranges(positions, cx: float, cy: float, r_keep: float = 40_000.0
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal range (m) and speed (m/s) of every vessel that comes within r_keep of
    one candidate. Returns (vessel index, range (V', T), speed (V', T)) as float32."""
    vid, x, y, spd = positions
    r = np.hypot(x - cx, y - cy)
    with np.errstate(invalid="ignore"):
        near = np.nanmin(np.where(np.isfinite(r), r, np.inf), axis=1) < r_keep
    return vid[near], r[near].astype(np.float32), spd[near].astype(np.float32)


def excess_from_ranges(r: np.ndarray, spd: np.ndarray, length: np.ndarray, tclass: np.ndarray,
                       params: LevelParams) -> np.ndarray:
    """Predicted ship-noise excess over the floor (dB), shape (T,)."""
    if len(r) == 0:
        return np.zeros(r.shape[1] if r.ndim == 2 else 0)
    src = source_db(length, tclass, spd, params)
    rr = np.hypot(r, params.min_range)
    tl = params.spreading * np.log10(rr) + params.absorption * rr / 1000.0
    floor_tl = params.spreading * np.log10(params.floor_km * 1000.0) + params.absorption * params.floor_km
    power = np.nansum(10.0 ** ((src - tl + floor_tl) / 10.0), axis=0)
    return 10.0 * np.log10(1.0 + power)


def predicted_excess(tracks: Tracks, tgrid: np.ndarray, cx, cy, params: LevelParams,
                     *, positions=None) -> np.ndarray:
    """Predicted excess (dB) at each candidate, shape (C, T)."""
    cx = np.atleast_1d(np.asarray(cx, dtype=np.float64))
    cy = np.atleast_1d(np.asarray(cy, dtype=np.float64))
    positions = positions if positions is not None else vessel_positions(tracks, tgrid)
    out = np.zeros((len(cx), len(tgrid)))
    for c in range(len(cx)):
        vid, r, spd = ranges(positions, cx[c], cy[c])
        out[c] = excess_from_ranges(r, spd, tracks.length[vid], tracks.tclass[vid], params)
    return out


def observed_excess(level_db: np.ndarray, dt: float, *, smooth_s: float = 180.0,
                    baseline_s: float = 4 * 3600.0) -> np.ndarray:
    """Observed band level above its slow floor (dB). NaNs (gaps) are kept as NaN."""
    level = np.asarray(level_db, dtype=np.float64)
    good = np.isfinite(level)
    if good.sum() < 5:
        return np.full(len(level), np.nan)
    filled = np.where(good, level, np.nanmedian(level))
    smooth = uniform_filter1d(filled, max(1, int(round(smooth_s / dt))), mode="nearest")
    floor = percentile_filter(smooth, 10, size=max(3, int(round(baseline_s / dt))), mode="nearest")
    return np.where(good, np.maximum(smooth - floor, 0.0), np.nan)


def highpass(excess: np.ndarray, dt: float, baseline_s: float = 4 * 3600.0) -> np.ndarray:
    """Remove the slow floor from a predicted curve the same way as from an observed one,
    so hours-long plateaus (ships at anchor, distant traffic) do not count as passages."""
    floor = percentile_filter(excess, 10, size=max(3, int(round(baseline_s / dt))), mode="nearest")
    return np.maximum(excess - floor, 0.0)


def correlate(obs: np.ndarray, pred: np.ndarray, *, min_lag: int = 45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Correlate one observed curve with predicted curves for C candidates (same length).

    Returns (rho at zero lag (C,), z-score (C,), rho at every circular lag (C, T)). The
    z-score compares zero lag with the spread of correlations at lags beyond min_lag
    samples, where the curves are unrelated: an empirical null that needs no model of
    how peaky the curves are.
    """
    pred = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    n = pred.shape[1]
    good = np.isfinite(obs)
    a = np.where(good, obs, 0.0)
    a = np.where(good, a - a[good].mean() if good.any() else 0.0, 0.0)
    b = pred - pred.mean(axis=1, keepdims=True)
    sa, sb = np.sqrt((a * a).sum()), np.sqrt((b * b).sum(axis=1))
    rho = np.zeros_like(b)
    ok = (sb > 1e-9) & (sa > 1e-9)
    if ok.any():
        xc = np.fft.irfft(np.conj(np.fft.rfft(a))[None, :] * np.fft.rfft(b[ok], axis=1), n=n, axis=1)
        rho[ok] = xc / (sa * sb[ok, None])
    null = rho[:, min_lag:n - min_lag]
    z = np.where(ok, (rho[:, 0] - null.mean(axis=1)) / np.maximum(null.std(axis=1), 1e-6), 0.0)
    return rho[:, 0], z, rho
