"""Who could an unexplained sound have been? Vessels whose AIS went silent on a course
that would have taken them past the hydrophone at about that time.

Tracks never bridge a gap of more than 15 minutes, so a ship that stops reporting simply
vanishes from the predicted passages. Here those gaps are the evidence: join the last fix
before the silence to the first one after it with a straight course at steady speed, and
see where that puts the ship when the sound was heard. This is dead reckoning, not proof.
Reception gaps far from shore look the same as a transponder switched off.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np

from . import ais
from .geo import KNOT, bbox_around, to_xy
from .regions import region_for

MIN_GAP_S = ais.MAX_GAP_S          # shorter silences are already bridged by the tracks
MAX_GAP_S = 6 * 3600.0
EDGE_M = 12_000.0                  # a "gap" that starts or ends this near the region box is just the box


@dataclass
class Gap:
    mmsi: int
    name: str
    vessel_type: int
    length: float
    silent_from: float
    silent_to: float
    bridge_time: float               # when the straight course is closest to the site
    bridge_range_m: float
    speed_kn: float                  # speed the straight course implies


def silent_courses(lat: float, lon: float, t0: float, t1: float, *, reach_m: float = 20_000.0,
                   search_m: float = 150_000.0) -> list[Gap]:
    """Every AIS silence of 15 min to 6 h whose straight-line course passes within reach_m
    of (lat, lon) between t0 and t1."""
    region = region_for(lat, lon)
    have, _ = ais.coverage(t0 - MAX_GAP_S, t1 + MAX_GAP_S)
    if region is None or not have:
        return []
    la0, la1, lo0, lo1 = bbox_around(lat, lon, search_m)
    files = ", ".join(f"'{ais.points_path(d)}'" for d in have)
    con = duckdb.connect()
    # Fixes of every vessel seen in the neighbourhood, wherever in the region they were, so
    # that leaving the neighbourhood is not mistaken for going silent.
    rows = con.execute(f"""
        WITH nearby AS (
            SELECT DISTINCT mmsi FROM read_parquet([{files}])
            WHERE lat BETWEEN {la0} AND {la1} AND lon BETWEEN {lo0} AND {lo1}
              AND t BETWEEN {t0 - MAX_GAP_S} AND {t1 + MAX_GAP_S}
        ), fixes AS (
            SELECT p.mmsi, p.t, p.lat, p.lon,
                   lead(p.t) OVER w AS t2, lead(p.lat) OVER w AS lat2, lead(p.lon) OVER w AS lon2
            FROM read_parquet([{files}]) p JOIN nearby USING (mmsi)
            WHERE p.t BETWEEN {t0 - MAX_GAP_S} AND {t1 + MAX_GAP_S}
            WINDOW w AS (PARTITION BY p.mmsi ORDER BY p.t)
        )
        SELECT mmsi, t, lat, lon, t2, lat2, lon2 FROM fixes
        WHERE t2 - t > {MIN_GAP_S} AND t2 - t <= {MAX_GAP_S}
    """).fetchnumpy()
    if len(rows["mmsi"]) == 0:
        con.close()
        return []
    mmsi = np.asarray(rows["mmsi"], dtype=np.int64)
    ta, tb = np.asarray(rows["t"], float), np.asarray(rows["t2"], float)
    lata, lona = np.asarray(rows["lat"], float), np.asarray(rows["lon"], float)
    latb, lonb = np.asarray(rows["lat2"], float), np.asarray(rows["lon2"], float)

    # Both ends must sit well inside the region box, or the "silence" is only the box edge.
    def inside(la, lo):
        dy = np.minimum(la - region.lat_min, region.lat_max - la) * 111_320.0
        dx = np.minimum(lo - region.lon_min, region.lon_max - lo) * 111_320.0 * np.cos(np.radians(lat))
        return np.minimum(dx, dy) > EDGE_M

    xa, ya = to_xy(lata, lona, lat, lon)
    xb, yb = to_xy(latb, lonb, lat, lon)
    dx, dy = xb - xa, yb - ya
    dist = np.hypot(dx, dy)
    dt = tb - ta
    speed = dist / dt
    u = np.clip(-(xa * dx + ya * dy) / np.where(dist > 0, dist ** 2, 1.0), 0.0, 1.0)
    rng = np.hypot(xa + u * dx, ya + u * dy)
    when = ta + u * dt
    ok = (inside(lata, lona) & inside(latb, lonb) & (speed >= 1.0) & (speed <= 23.0)
          & (u > 0.0) & (u < 1.0) & (rng <= reach_m) & (when >= t0) & (when <= t1))
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        con.close()
        return []
    ids = ", ".join(str(m) for m in sorted(set(mmsi[idx].tolist())))
    static = {int(r[0]): r[1:] for r in con.execute(f"""
        SELECT mmsi, coalesce(max(name), ''), coalesce(max(vessel_type), -1), coalesce(max(length), 'nan'::FLOAT)::DOUBLE
        FROM read_parquet([{files}]) WHERE mmsi IN ({ids}) GROUP BY mmsi""").fetchall()}
    con.close()
    gaps = [Gap(int(mmsi[i]), str(static[int(mmsi[i])][0]), int(static[int(mmsi[i])][1]), float(static[int(mmsi[i])][2]),
                float(ta[i]), float(tb[i]), float(when[i]), float(rng[i]), float(speed[i] / KNOT)) for i in idx]
    return sorted(gaps, key=lambda g: g.bridge_time)


def for_event(gaps: list[Gap], t_event: float, *, limit: int = 3) -> list[Gap]:
    """Silences that could explain a sound heard at t_event. The longer the silence, the
    less a straight course says about when the ship went by, so the window widens with it."""
    hits = []
    for g in gaps:
        window = max(600.0, 0.2 * (g.silent_to - g.silent_from))
        if abs(g.bridge_time - t_event) <= window:
            hits.append((abs(g.bridge_time - t_event) / window + g.bridge_range_m / 20_000.0, g))
    return [g for _, g in sorted(hits, key=lambda h: h[0])[:limit]]
