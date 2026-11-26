"""One day of ship noise over a whole region, ready to animate.

Everything here comes from data already held: AIS for the day, and spectra for the
hydrophones in the region. The browser draws the noise field itself from the vessel
positions and source levels, using the same propagation constants as the matcher, so the
picture and the score are the same model.

One built day ships with the package (`data/soundmaps/`), so the map plays straight after
a clone with nothing downloaded.
"""

from __future__ import annotations

import base64
import json
from datetime import date
from importlib import resources

import numpy as np

from . import ais, catalog, multisite
from .geo import KNOT, haversine_m, to_latlon
from .paths import cache_path
from .pipeline import MATCH_DT, iso, vessel_type_name, who_passed
from .public import public_clip
from .regions import REGIONS
from .shipdata import flag, is_government
from .soundscape import LevelParams, source_db, vessel_positions

STEP_S = 120.0
PAD_M = 48_000.0
MAX_VESSELS = 320
CLEAN_DB = 6.0           # a passage counts for the ranking when its ship out-shouts the rest by this much


def _sites_with_spectra(region_id: str, t0: float, t1: float) -> list[tuple[catalog.Site, catalog.Deployment, object]]:
    """Sites in the region whose spectra are published or already cached. Never streams audio."""
    out = []
    for site, dep in catalog.active_sites(t0, t1):
        if site.region != region_id:
            continue
        try:
            clip = public_clip(site.id, t0, t1, source="spectra")
        except (LookupError, OSError):
            continue
        out.append((site, dep, clip))
    return out


def _bundled(region_id: str):
    """Built days that ship with the package, as {iso day: resource}."""
    folder = resources.files("hydrolocator").joinpath("data/soundmaps")
    if not folder.is_dir():
        return {}
    prefix = f"{region_id}_"
    return {f.name[len(prefix):-5]: f for f in folder.iterdir() if f.name.startswith(prefix) and f.name.endswith(".json")}


def available(region_id: str) -> list[str]:
    """Days the map can play: every AIS day held, plus the days that ship with the package."""
    if region_id not in REGIONS:
        return []
    return sorted({d.isoformat() for d in ais.available_days()} | set(_bundled(region_id)))


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def build(region_id: str, day: date, *, refresh: bool = False) -> dict:
    cache = cache_path("soundmaps", f"{region_id}_{day.isoformat()}.json")
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    sample = _bundled(region_id).get(day.isoformat())
    if sample is not None and not refresh:
        return json.loads(sample.read_text())
    from .blindtest import day_bounds

    t0, t1 = day_bounds(day)
    if not ais.points_path(day).exists():
        raise LookupError(f"no AIS held for {day}")
    level = LevelParams()
    heard = _sites_with_spectra(region_id, t0, t1)
    region = REGIONS[region_id]
    if heard:
        lats, lons = [d.lat for _, d, _ in heard], [d.lon for _, d, _ in heard]
        lat0, lon0 = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
        half_ns = (max(lats) - min(lats)) / 2 * 111_320.0 + PAD_M
        half_ew = (max(lons) - min(lons)) / 2 * 111_320.0 * np.cos(np.radians(lat0)) + PAD_M * 1.5
    else:
        lat0, lon0 = region.center
        half_ns, half_ew = PAD_M, PAD_M * 1.5
    d_lat = half_ns / 111_320.0
    d_lon = half_ew / (111_320.0 * np.cos(np.radians(lat0)))
    bbox = [lon0 - d_lon, lat0 - d_lat, lon0 + d_lon, lat0 + d_lat]          # west, south, east, north

    # --- every vessel in view, on a two-minute grid
    steps = int(86400 / STEP_S)
    tgrid = t0 + STEP_S * (np.arange(steps) + 0.5)
    tracks = ais.load_tracks(lat0, lon0, t0, t1, float(np.hypot(half_ns, half_ew)) + 5_000.0)
    _, x, y, spd = vessel_positions(tracks, tgrid)
    lat, lon = to_latlon(x, y, lat0, lon0)
    inside = np.isfinite(x) & (lat >= bbox[1]) & (lat <= bbox[3]) & (lon >= bbox[0]) & (lon <= bbox[2])
    src = source_db(tracks.length, tracks.tclass, spd, level)
    moving = (np.where(inside, spd, 0.0).max(axis=1) >= 1.5 * KNOT) if len(inside) else np.zeros(0, bool)
    weight = np.where(inside, 10.0 ** (src / 10.0), 0.0).sum(axis=1) if len(inside) else np.zeros(0)
    keep = np.nonzero(moving)[0]
    keep = keep[np.argsort(-weight[keep])][:MAX_VESSELS]

    def quant(v, lo, hi):
        q = np.round((v - lo) / (hi - lo) * 65534.0) - 32767.0
        return q

    qlat = np.where(inside[keep], quant(lat[keep], bbox[1], bbox[3]), -32768).astype("<i2")
    qlon = np.where(inside[keep], quant(lon[keep], bbox[0], bbox[2]), -32768).astype("<i2")
    qsrc = np.clip(np.round(src[keep]), -100, 100).astype("i1")
    vessels = []
    for vid in keep:
        length = float(tracks.length[vid])
        vessels.append({
            "mmsi": int(tracks.mmsi[vid]), "name": str(tracks.name[vid] or "") or None,
            "type": vessel_type_name(int(tracks.vessel_type[vid])),
            "length_m": round(length) if np.isfinite(length) and length > 0 else None,
            "flag": flag(int(tracks.mmsi[vid])),
        })
    index = {int(tracks.mmsi[v]): k for k, v in enumerate(keep)}

    # --- each listening post: what it heard, what AIS predicts, which ship made each peak
    posts, ranking = [], []
    spreading = lambda r: level.spreading * np.log10(np.hypot(r, level.min_range)) + level.absorption * np.hypot(r, level.min_range) / 1000.0  # noqa: E731
    for site, dep, clip in heard:
        rep = who_passed(clip, dep.lat, dep.lon)
        calibrated = site.id.startswith("sanctsound:") and clip.dt == 60.0           # archive spectra are in dB re 1 uPa
        events = []
        for row in rep.rows:
            if row.heard_time is None:
                continue
            events.append({"t": int(round(row.heard_time - t0)), "db": round(row.heard_prominence_db or 0),
                           "vessel": index.get(row.vessel.mmsi) if row.vessel else None,
                           "name": (row.vessel.name or None) if row.vessel else None, "status": row.status})
            v = row.vessel
            if row.status != "matched" or not calibrated or v is None or row.cpa_range_m is None or row.cpa_range_m > 8_000.0:
                continue
            if is_government(v.vessel_type, v.name, callsign=v.callsign, mmsi=v.mmsi):
                continue
            # Only rank a passage when this ship clearly dominates what AIS predicts at that moment.
            k = int(np.clip((row.heard_time - t0) // STEP_S, 0, steps - 1))
            sx, sy = (dep.lon - lon0) * 111_320.0 * np.cos(np.radians(lat0)), (dep.lat - lat0) * 111_320.0
            rng = np.hypot(x[:, k] - sx, y[:, k] - sy)
            contrib = np.where(np.isfinite(rng), src[:, k] - spreading(np.nan_to_num(rng, nan=1e9)), -999.0)
            me = np.nonzero(tracks.mmsi == v.mmsi)[0]
            if len(me) == 0:
                continue
            others = np.delete(contrib, me[0])
            rest = 10.0 * np.log10(np.sum(10.0 ** (others / 10.0)) + 1e-30)
            if contrib[me[0]] - rest < CLEAN_DB:
                continue
            frame = int(np.clip((row.heard_time - clip.t_start) // clip.dt, 0, len(clip.level) - 1))
            received = float(np.nanmax(clip.level[max(0, frame - 2):frame + 3]))
            ranking.append({
                "mmsi": v.mmsi, "name": v.name or None, "type": vessel_type_name(v.vessel_type),
                "length_m": round(v.length) if v.length else None, "flag": flag(v.mmsi), "vessel": index.get(v.mmsi),
                "source_db": round(received + float(spreading(row.cpa_range_m))), "t": int(round(row.heard_time - t0)),
                "range_km": round(row.cpa_range_m / 1000, 1), "speed_kn": round(row.speed_kn, 1), "site": site.name,
            })
        posts.append({
            "id": site.id, "name": site.name, "lat": dep.lat, "lon": dep.lon,
            "heard_db": [None if not np.isfinite(v) else round(float(v), 1) for v in rep.observed_excess],
            "predicted_db": [round(float(v), 1) for v in rep.predicted_excess],
            "z": round(rep.z, 1), "events": events, "curve_dt_s": MATCH_DT,
        })
    best: dict[int, dict] = {}
    for r in sorted(ranking, key=lambda r: -r["source_db"]):
        best.setdefault(r["mmsi"], r)

    # --- ghosts: unexplained sounds linked across the two posts furthest apart
    ghosts = []
    if len(heard) >= 2:
        pairs = [(a, b) for i, a in enumerate(heard) for b in heard[i + 1:]]
        a, b = max(pairs, key=lambda p: haversine_m(p[0][1].lat, p[0][1].lon, p[1][1].lat, p[1][1].lon))
        try:
            found = multisite.analyse(a[0].id, b[0].id, t0, t1)
            for g in found["inferred"]:
                ghosts.append({k: g[k] for k in ("speed_kn", "route_like", "routes_used", "path", "path_t", "gap_min")}
                              | {"from": a[0].name if g["gap_min"] > 0 else b[0].name, "to": b[0].name if g["gap_min"] > 0 else a[0].name})
        except LookupError:
            pass

    out = {
        "region": {"id": region_id, "name": region.name}, "date": day.isoformat(), "t0": iso(t0), "step_s": STEP_S, "steps": steps,
        "bbox": [round(v, 5) for v in bbox], "center": [round(lon0, 5), round(lat0, 5)],
        "model": {"spreading": level.spreading, "absorption": level.absorption, "floor_km": level.floor_km, "min_range": level.min_range},
        "vessels": vessels, "lat": _b64(qlat), "lon": _b64(qlon), "src": _b64(qsrc),
        "posts": posts, "ranking": list(best.values())[:8], "ghosts": ghosts,
    }
    cache.write_text(json.dumps(out))
    return out
