"""Ships heard at two hydrophones: link the two passages and infer the ship's track.

The public recorders are tens of kilometres apart with unsynchronised clocks, so there is
no triangulation. What two sites do share is a ship that passes both. The gap between the
two closest approaches fixes its speed along the way, and years of AIS say which way ships
actually go between those two points. A linked pair of passages therefore gets the route
of the AIS vessels that made the same crossing in the same time, rescaled to the heard
times. Linking uses the time gap and how alike the two passages sound.

For a ship on AIS this is redundant, which is what makes it testable: hide the ship,
infer from sound, compare. For a passage with no AIS it is the only track there is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from . import ais, cpa
from .geo import KNOT, haversine_m, to_latlon, to_xy

REACH_M = 10_000.0


@dataclass
class Site:
    id: str
    lat: float
    lon: float


@dataclass
class Crossing:
    """One AIS vessel passing both sites: a real route between them."""
    day: date
    mmsi: int
    name: str
    vessel_type: int
    length: float
    t_a: float
    t_b: float
    r_a: float
    r_b: float
    speed: float                       # m/s, mean over the crossing
    path: np.ndarray = field(repr=False, default=None)   # (n, 3): time, lat, lon

    @property
    def dt(self) -> float:
        return self.t_b - self.t_a


def crossings(a: Site, b: Site, day: date, *, reach_m: float = REACH_M) -> list[Crossing]:
    """AIS vessels that passed within reach_m of both sites on one UTC day."""
    from .blindtest import day_bounds

    d0, d1 = day_bounds(day)
    mid_lat, mid_lon = (a.lat + b.lat) / 2, (a.lon + b.lon) / 2
    span = float(haversine_m(a.lat, a.lon, b.lat, b.lon))
    tracks = ais.load_tracks(mid_lat, mid_lon, d0, d1, span / 2 + reach_m + 25_000.0)
    ax, ay = to_xy(a.lat, a.lon, mid_lat, mid_lon)
    bx, by = to_xy(b.lat, b.lon, mid_lat, mid_lon)
    pa = cpa.predict(tracks, float(ax), float(ay), reach_m)
    pb = cpa.predict(tracks, float(bx), float(by), reach_m)
    out = []
    for i in range(len(pa)):
        if pa.v[i] < 2 * KNOT:
            continue
        same = np.nonzero(pb.vid == pa.vid[i])[0]
        if len(same) == 0:
            continue
        j = same[np.argmin(np.abs(pb.t[same] - pa.t[i]))]
        gap = abs(pb.t[j] - pa.t[i])
        if gap < 600 or gap > 6 * 3600:
            continue
        vid = int(pa.vid[i])
        legs = np.nonzero(tracks.vid == vid)[0]
        lo, hi = min(pa.t[i], pb.t[j]) - 1800, max(pa.t[i], pb.t[j]) + 1800
        keep = legs[(tracks.t0[legs] >= lo) & (tracks.t0[legs] <= hi)]
        if len(keep) < 5:
            continue
        lat, lon = to_latlon(tracks.x0[keep], tracks.y0[keep], mid_lat, mid_lon)
        step = max(1, len(keep) // 120)
        path = np.stack([tracks.t0[keep], lat, lon], axis=1)[::step]
        out.append(Crossing(day, int(tracks.mmsi[vid]), str(tracks.name[vid]), int(tracks.vessel_type[vid]),
                            float(tracks.length[vid]), float(pa.t[i]), float(pb.t[j]), float(pa.r[i]), float(pb.r[j]),
                            float(span / gap * abs(np.cos(0.0))), path))
        # mean speed along the actual path, not the straight baseline
        seg = np.hypot(np.diff(tracks.x0[keep]), np.diff(tracks.y0[keep])).sum()
        out[-1].speed = float(seg / max(tracks.t0[keep][-1] - tracks.t0[keep][0], 1.0))
    return out


def route_library(a: Site, b: Site, days: list[date] | None = None) -> list[Crossing]:
    out = []
    for day in days if days is not None else ais.available_days():
        out += crossings(a, b, day)
    return out


@dataclass
class Inferred:
    """A track inferred for a pair of heard times, from the AIS vessels that made the same crossing."""
    t_a: float
    t_b: float
    speed_kn: float
    speed_spread_kn: float
    r_a_km: float
    r_b_km: float
    routes_used: int
    like: str                          # name of the closest historical crossing
    path: np.ndarray = field(repr=False, default=None)   # (n, 3): time, lat, lon, rescaled to the heard times


def infer(t_a: float, t_b: float, library: list[Crossing], *, tolerance: float = 0.15, exclude_day: date | None = None,
          exclude_mmsi: int | None = None) -> Inferred | None:
    """Track for a ship heard at A at t_a and at B at t_b."""
    dt = t_b - t_a
    pool = [c for c in library if c.day != exclude_day and c.mmsi != exclude_mmsi and np.sign(c.dt) == np.sign(dt)]
    if not pool or dt == 0:
        return None
    ratio = np.array([abs(c.dt / dt) for c in pool])
    close = [c for c, r in zip(pool, ratio) if abs(np.log(r)) <= np.log(1 + tolerance)]
    used = close or [pool[int(np.argmin(np.abs(np.log(ratio))))]]
    # Same path in a different time: speed scales with the ratio of the time gaps.
    speeds = np.array([c.speed * abs(c.dt / dt) for c in used])
    best = min(used, key=lambda c: abs(np.log(abs(c.dt / dt))))
    path = best.path.copy()
    path[:, 0] = t_a + (path[:, 0] - best.t_a) * (dt / best.dt)
    return Inferred(t_a, t_b, float(np.median(speeds) / KNOT), float(np.std(speeds) / KNOT) if len(speeds) > 1 else float("nan"),
                    float(np.median([c.r_a for c in used]) / 1000), float(np.median([c.r_b for c in used]) / 1000),
                    len(used), best.name or str(best.mmsi), path)


def plausible_gap(library: list[Crossing]) -> tuple[float, float]:
    """The range of time gaps real ships take between the two sites (seconds, unsigned)."""
    gaps = np.array([abs(c.dt) for c in library])
    if len(gaps) < 3:
        return 1800.0, 4 * 3600.0
    return float(np.percentile(gaps, 2) * 0.85), float(np.percentile(gaps, 98) * 1.15)


# ---------------------------------------------------------------------------- analysis

# Measured on Stellwagen SB01 and SB03 (36 km apart), 20 AIS days, by hiding the AIS of
# ships heard at both sites: 15 ships. Given the right pair of passages, speed came out
# within 0.6 kn and range within 0.3 km (medians). Picking the right pair from sound and
# loudness worked for 9 of the 15, so a link is a lead to check, not a finding.
VALIDATION = {"ships": 15, "speed_err_kn": 0.6, "range_err_km": 0.3, "links_right": 9}
LOUD_WEIGHT = 0.03


def _dark_events(clip, lat: float, lon: float):
    """Loud passages at one site that no AIS vessel explains: (time, prominence, signature)."""
    from .pipeline import who_passed
    from .voiceprint import line_spectra, signature

    report = who_passed(clip, lat, lon)
    lines = line_spectra(clip.freqs, clip.psd_db) if clip.psd_db is not None else None
    out = []
    for row in report.dark:
        sig = signature(lines, clip.t + clip.dt / 2, row.heard_time) if lines is not None else None
        out.append((row.heard_time, row.heard_prominence_db, sig))
    return report, out


def link(dark_a: list, dark_b: list, library: list[Crossing]) -> list[tuple[int, int, float]]:
    """Pair unexplained passages at A with ones at B: plausible time gap, mutual best by
    how alike they sound plus how loud each is. Returns (index at A, index at B, score)."""
    lo, hi = plausible_gap(library)

    def score(i: int, j: int) -> float | None:
        ta, pa, sa = dark_a[i]
        tb, pb, sb = dark_b[j]
        if not lo <= abs(tb - ta) <= hi:
            return None
        sim = float(sa @ sb) if sa is not None and sb is not None else 0.0
        return sim + LOUD_WEIGHT * (pa + pb) / 2

    table = {(i, j): s for i in range(len(dark_a)) for j in range(len(dark_b)) if (s := score(i, j)) is not None}
    pairs = []
    for (i, j), s in table.items():
        best_for_a = max((v for (x, _), v in table.items() if x == i), default=None)
        best_for_b = max((v for (_, y), v in table.items() if y == j), default=None)
        if s == best_for_a and s == best_for_b:
            pairs.append((i, j, s))
    return sorted(pairs, key=lambda p: -p[2])


def analyse(site_a: str, site_b: str, t0: float, t1: float) -> dict:
    """Everything the two-site view shows for one window: the AIS ships that passed both
    sites, and tracks inferred for linked passages that have no AIS at either."""
    from datetime import datetime, timezone

    from . import catalog
    from .pipeline import iso, vessel_type_name
    from .public import public_clip

    sa, sb = catalog.get(site_a), catalog.get(site_b)
    da, db = sa.active(t0, t1), sb.active(t0, t1)
    if da is None or db is None:
        raise LookupError("both sites must have been recording for the whole window")
    a, b = Site(sa.id, da.lat, da.lon), Site(sb.id, db.lat, db.lon)
    library = route_library(a, b)
    clip_a, clip_b = public_clip(sa.id, t0, t1), public_clip(sb.id, t0, t1)
    rep_a, dark_a = _dark_events(clip_a, a.lat, a.lon)
    rep_b, dark_b = _dark_events(clip_b, b.lat, b.lon)
    day = datetime.fromtimestamp(t0, timezone.utc).date()

    def path(p: np.ndarray) -> list[list[float]]:
        return [[round(float(lon), 5), round(float(lat), 5)] for _, lat, lon in p]

    def path_t(p: np.ndarray) -> list[int]:
        return [int(round(t - t0)) for t, _, _ in p]

    def heard_near(report, t: float, r: float, v: float) -> bool:
        return any(row.heard_time is not None and abs(row.heard_time - t) <= 180.0 + 0.5 * r / max(v, 0.5) for row in report.rows)

    on_ais = []
    for c in crossings(a, b, day):
        if not (t0 <= c.t_a <= t1 and t0 <= c.t_b <= t1):
            continue
        on_ais.append({
            "mmsi": c.mmsi, "name": c.name or None, "type": vessel_type_name(c.vessel_type),
            "length_m": round(c.length) if np.isfinite(c.length) else None,
            "t_a": iso(c.t_a), "t_b": iso(c.t_b), "gap_min": round(c.dt / 60), "speed_kn": round(c.speed / KNOT, 1),
            "range_a_km": round(c.r_a / 1000, 1), "range_b_km": round(c.r_b / 1000, 1),
            "heard_a": heard_near(rep_a, c.t_a, c.r_a, c.speed), "heard_b": heard_near(rep_b, c.t_b, c.r_b, c.speed),
            "path": path(c.path), "path_t": path_t(c.path),
        })
    inferred = []
    for i, j, s in link(dark_a, dark_b, library):
        guess = infer(dark_a[i][0], dark_b[j][0], library)
        if guess is None:
            continue
        inferred.append({
            "t_a": iso(guess.t_a), "t_b": iso(guess.t_b), "gap_min": round((guess.t_b - guess.t_a) / 60),
            "speed_kn": round(guess.speed_kn, 1),
            "speed_spread_kn": None if not np.isfinite(guess.speed_spread_kn) else round(guess.speed_spread_kn, 1),
            "range_a_km": round(guess.r_a_km, 1), "range_b_km": round(guess.r_b_km, 1),
            "routes_used": guess.routes_used, "route_like": guess.like, "link_score": round(s, 2),
            "peak_a_db": round(dark_a[i][1]), "peak_b_db": round(dark_b[j][1]),
            "path": path(guess.path), "path_t": path_t(guess.path),
        })
    return {
        "sites": [{"id": a.id, "name": sa.name, "lat": a.lat, "lon": a.lon}, {"id": b.id, "name": sb.name, "lat": b.lat, "lon": b.lon}],
        "baseline_km": round(float(haversine_m(a.lat, a.lon, b.lat, b.lon)) / 1000, 1),
        "library": {"crossings": len(library), "days": len(ais.available_days())},
        "unexplained": {"a": len(dark_a), "b": len(dark_b)},
        "on_ais": on_ais, "inferred": inferred, "validation": VALIDATION,
    }
