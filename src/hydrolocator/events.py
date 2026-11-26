"""Observed passages: what the hydrophone heard, reduced to a list of timed events."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observed:
    """Ship passages heard in one clip.

    Times are POSIX seconds on the recording's own clock, which may be offset from UTC.
    Each event says "a ship was closest at t, give or take": `half` is a flat half-width
    (the whole audible period, for logs that only give start and end) and `sigma` is the
    Gaussian uncertainty beyond that. Marks are NaN when not measured.
    """
    t_start: float
    t_end: float
    t: np.ndarray
    sigma: np.ndarray
    half: np.ndarray = None
    quality: np.ndarray = None           # 0..1 confidence that this is a vessel
    speed: np.ndarray = None             # m/s, from Doppler
    range: np.ndarray = None             # m, CPA slant range from Doppler
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.t)
        self.t = np.asarray(self.t, dtype=np.float64)
        self.sigma = np.broadcast_to(np.asarray(self.sigma, dtype=np.float64), (n,)).copy()
        for name, fill in (("half", 0.0), ("quality", 1.0), ("speed", np.nan), ("range", np.nan)):
            value = getattr(self, name)
            setattr(self, name, np.full(n, fill) if value is None
                    else np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)).copy())

    def __len__(self) -> int:
        return len(self.t)

    @property
    def hours(self) -> float:
        return (self.t_end - self.t_start) / 3600.0

    @property
    def rate(self) -> float:
        """Events per second, floored at one event per clip so empty clips stay finite."""
        return max(len(self), 1) / max(self.t_end - self.t_start, 1.0)

    def _take(self, mask, t0, t1, shift=0.0) -> Observed:
        return Observed(t0, t1, self.t[mask] + shift, self.sigma[mask], self.half[mask],
                        self.quality[mask], self.speed[mask], self.range[mask], dict(self.meta))

    def window(self, t0: float, t1: float) -> Observed:
        return self._take((self.t >= t0) & (self.t < t1), t0, t1)

    def shifted(self, seconds: float) -> Observed:
        """The same clip with its clock moved by `seconds` (for clock-error tests)."""
        return self._take(np.ones(len(self), bool), self.t_start + seconds, self.t_end + seconds, seconds)


def from_intervals(spans: np.ndarray, t_start: float, t_end: float, *,
                   edge_sigma: float = 120.0) -> Observed:
    """Events from logged [start, end] detections (for example SanctSound ship logs).

    The log says when vessel noise was audible, not when the ship was closest, so the
    closest approach could be anywhere inside the span.
    """
    spans = np.asarray(spans, dtype=np.float64).reshape(-1, 2)
    mid = spans.mean(axis=1)
    keep = (mid >= t_start) & (mid < t_end)
    spans, mid = spans[keep], mid[keep]
    return Observed(t_start, t_end, mid, edge_sigma, half=(spans[:, 1] - spans[:, 0]) / 2)
