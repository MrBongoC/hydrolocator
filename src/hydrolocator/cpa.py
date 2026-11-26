"""Closest-point-of-approach prediction: for each candidate position, the list of ship
passages AIS says it should have heard. This is the hot path of Mode B.

Tracks are straight legs, so range to a fixed point is convex along each leg and the
closest approach per leg has a closed form. A passage is a local minimum of that per-leg
minimum along a vessel's track.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ais import Tracks

MIN_SPEED = 0.5        # m/s (about 1 kn); slower than this is drifting or moored, not a passage
MERGE_S = 900.0        # repeated minima of one vessel inside this window count as one passage


@dataclass
class Passages:
    """Predicted passages, flat. `cand` indexes the candidate, `vid` the vessel."""
    cand: np.ndarray
    vid: np.ndarray
    t: np.ndarray      # CPA time, POSIX s
    r: np.ndarray      # horizontal CPA range, m
    v: np.ndarray      # speed over ground at CPA, m/s

    def __len__(self) -> int:
        return len(self.t)

    def for_candidate(self, c: int) -> Passages:
        m = self.cand == c
        return Passages(self.cand[m], self.vid[m], self.t[m], self.r[m], self.v[m])


def predict(tracks: Tracks, cx, cy, r_max: float, *, chunk: int = 256) -> Passages:
    """Passages within r_max of each candidate (cx[i], cy[i]), metres in the track frame."""
    cx = np.atleast_1d(np.asarray(cx, dtype=np.float64))
    cy = np.atleast_1d(np.asarray(cy, dtype=np.float64))
    empty = Passages(*(np.array([], dtype=d) for d in (np.int64, np.int64, float, float, float)))
    if tracks.n_legs == 0 or len(cx) == 0:
        return empty

    # Only legs that come anywhere near the candidate set matter.
    pad = r_max
    near = ((np.maximum(tracks.x0, tracks.x1) >= cx.min() - pad) & (np.minimum(tracks.x0, tracks.x1) <= cx.max() + pad)
            & (np.maximum(tracks.y0, tracks.y1) >= cy.min() - pad) & (np.minimum(tracks.y0, tracks.y1) <= cy.max() + pad))
    # Keep whole vessels so that leg adjacency stays intact.
    keep = np.isin(tracks.vid, np.unique(tracks.vid[near]))
    if not keep.any():
        return empty
    vid, t0, t1 = tracks.vid[keep], tracks.t0[keep], tracks.t1[keep]
    x0, y0, x1, y1 = tracks.x0[keep], tracks.y0[keep], tracks.x1[keep], tracks.y1[keep]
    dx, dy = x1 - x0, y1 - y0
    dt = t1 - t0
    len2 = dx * dx + dy * dy
    speed = np.sqrt(len2) / dt
    # A track "piece" is a run of legs joined end to end (same vessel, no time gap).
    joined = np.zeros(len(vid), dtype=bool)
    joined[1:] = (vid[1:] == vid[:-1]) & (t0[1:] == t1[:-1])
    first = ~joined
    last = np.ones(len(vid), dtype=bool)
    last[:-1] = ~joined[1:]

    out = []
    for a in range(0, len(cx), chunk):
        ccx, ccy = cx[a:a + chunk, None], cy[a:a + chunk, None]
        wx, wy = ccx - x0, ccy - y0
        u = np.clip((wx * dx + wy * dy) / np.where(len2 > 0, len2, 1.0), 0.0, 1.0)
        d = np.hypot(wx - u * dx, wy - u * dy)                     # (C, K) closest range per leg
        left = np.empty_like(d, dtype=bool)
        left[:, 0] = True
        left[:, 1:] = (d[:, 1:] <= d[:, :-1]) | first[1:]
        right = np.empty_like(d, dtype=bool)
        right[:, -1] = True
        right[:, :-1] = (d[:, :-1] < d[:, 1:]) | last[:-1]
        # A minimum sitting exactly on the end of a piece is where the track starts or
        # stops, not a passage.
        edge = (first & (u == 0.0)) | (last & (u == 1.0))
        hit = left & right & ~edge & (d < r_max) & (speed >= MIN_SPEED)
        ci, ki = np.nonzero(hit)
        out.append((ci + a, vid[ki], t0[ki] + u[ci, ki] * dt[ki], d[ci, ki], speed[ki]))
    cand, v_id, t, r, v = (np.concatenate(c) for c in zip(*out))
    if len(t) == 0:
        return empty

    # Merge repeated minima (GPS jitter, loitering): keep the closest per burst.
    order = np.lexsort((t, v_id, cand))
    cand, v_id, t, r, v = cand[order], v_id[order], t[order], r[order], v[order]
    new = np.ones(len(t), dtype=bool)
    new[1:] = (cand[1:] != cand[:-1]) | (v_id[1:] != v_id[:-1]) | (t[1:] - t[:-1] > MERGE_S)
    group = np.cumsum(new) - 1
    best = np.lexsort((r, group))
    pick = best[np.r_[True, group[best][1:] != group[best][:-1]]]
    inside = (t[pick] >= tracks.t_start) & (t[pick] <= tracks.t_end)
    pick = pick[inside]
    return Passages(cand[pick], v_id[pick], t[pick], r[pick], v[pick])
