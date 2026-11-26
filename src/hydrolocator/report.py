"""Turn results into plain dicts: the JSON the CLI prints and the API serves."""

from __future__ import annotations

import numpy as np

from . import catalog, shipdata
from .geo import to_latlon
from .pipeline import Clip, GridResult, LocateResult, SiteReport, iso, vessel_type_name


def _round(x, nd=1):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def passage_rows(report: SiteReport) -> list[dict]:
    rows = []
    for r in report.rows:
        v = r.vessel
        rows.append({
            "status": r.status,
            "mmsi": v.mmsi if v else None,
            "name": (v.name or None) if v else None,
            "type": vessel_type_name(v.vessel_type) if v else None,
            "length_m": _round(v.length, 0) if v else None,
            "ais_class": (v.tclass or None) if v else None,
            "flag": shipdata.flag(v.mmsi) if v else None,
            "imo": (v.imo or None) if v else None,
            "callsign": (v.callsign or None) if v else None,
            "beam_m": _round(v.width, 0) if v else None,
            "draught_m": _round(v.draft, 1) if v else None,
            "government": shipdata.is_government(v.vessel_type, v.name, callsign=v.callsign, mmsi=v.mmsi) if v else None,
            "links": shipdata.links(v.mmsi) if v else None,
            "record": None,                       # filled from Wikidata by enrich()
            "cpa_time": iso(r.cpa_time),
            "cpa_range_km": _round(None if r.cpa_range_m is None else r.cpa_range_m / 1000, 2),
            "speed_kn": _round(r.speed_kn),
            "detect_prob": _round(r.detect_prob, 2),
            "heard_time": iso(r.heard_time),
            "prominence_db": _round(r.heard_prominence_db),
            "dt_s": _round(r.dt_s, 0),
            "suspects": [{
                "mmsi": g.mmsi, "name": g.name or None, "type": vessel_type_name(g.vessel_type),
                "length_m": _round(g.length, 0), "flag": shipdata.flag(g.mmsi),
                "silent_from": iso(g.silent_from), "silent_to": iso(g.silent_to),
                "silent_min": round((g.silent_to - g.silent_from) / 60),
                "bridge_time": iso(g.bridge_time), "bridge_range_km": _round(g.bridge_range_m / 1000, 1),
                "speed_kn": _round(g.speed_kn),
            } for g in r.suspects],
            "sounds_like": [{
                "mmsi": h.mmsi, "name": h.name or None, "type": vessel_type_name(h.vessel_type),
                "similarity": _round(h.similarity, 2), "right_share": _round(h.trust, 2), "heard_at": h.heard_at,
            } for h in r.hints],
        })
    return rows


def enrich(rows: list[dict]) -> None:
    """Add public records (operator, photograph, article, year built) to vessel rows, in place."""
    found = shipdata.wikidata([r["imo"] for r in rows if r.get("imo")])
    for r in rows:
        rec = found.get(r.get("imo") or "")
        if rec:
            r["record"] = rec
            if rec.get("operator") and shipdata.is_government(0, operator=rec["operator"]):
                r["government"] = True


def tracks_geojson(report: SiteReport, max_range_m: float = 25_000.0) -> dict:
    """Tracks of the vessels in the table, clipped to the neighbourhood, for the map."""
    tracks = report.tracks
    listed = {r.vessel.mmsi: r.status for r in report.rows if r.vessel}
    features = []
    if tracks is None or tracks.n_legs == 0:
        return {"type": "FeatureCollection", "features": features}
    bounds = np.r_[0, np.nonzero(np.diff(tracks.vid))[0] + 1, tracks.n_legs]
    for a, b in zip(bounds[:-1], bounds[1:]):
        vid = int(tracks.vid[a])
        mmsi = int(tracks.mmsi[vid])
        if mmsi not in listed:
            continue
        idx = np.nonzero(np.hypot(tracks.x0[a:b], tracks.y0[a:b]) < max_range_m)[0]
        if len(idx) < 2:
            continue
        # One feature per unbroken stretch: a vessel that leaves the neighbourhood and
        # comes back, or drops off AIS, must not be joined up with a straight line.
        when_all = tracks.t0[a:b][idx]
        cuts = np.nonzero((np.diff(idx) > 1) | (np.diff(when_all) > 900.0))[0] + 1
        for piece in np.split(idx, cuts):
            if len(piece) < 2:
                continue
            piece = piece[::max(1, len(piece) // 240)]
            lat, lon = to_latlon(tracks.x0[a:b][piece], tracks.y0[a:b][piece], tracks.lat0, tracks.lon0)
            features.append({
                "type": "Feature",
                "properties": {"mmsi": mmsi, "name": str(tracks.name[vid] or ""), "status": listed[mmsi],
                               # seconds after the clip's (clock-corrected) start, one per vertex
                               "times": [int(round(t - (report.t_start - report.offset_s)))
                                         for t in tracks.t0[a:b][piece]]},
                "geometry": {"type": "LineString",
                             "coordinates": [[round(float(x), 5), round(float(y), 5)] for y, x in zip(lat, lon)]},
            })
    return {"type": "FeatureCollection", "features": features}


def site_report(report: SiteReport, *, with_curves: bool = True, with_tracks: bool = True) -> dict:
    rows = passage_rows(report)
    out = {
        "lat": report.lat, "lon": report.lon,
        # Times in this report are UTC with the clock offset already taken off.
        "start": iso(report.t_start - report.offset_s), "end": iso(report.t_end - report.offset_s),
        "clock_offset_s": report.offset_s,
        "match": {"rho": _round(report.rho, 3), "z": _round(report.z, 2)},
        "counts": {
            "heard": report.n_events,
            "ais_passages": report.n_predicted,
            "matched": sum(r["status"] == "matched" for r in rows),
            "dark": sum(r["status"] == "dark" for r in rows),
            "not_heard": sum(r["status"] == "not_heard" for r in rows),
        },
        "ais_days_missing": report.ais_days_missing,
        "passages": rows,
    }
    if with_curves and report.tgrid is not None:
        out["curves"] = {
            "t0": iso(float(report.tgrid[0] - 30.0 - report.offset_s)), "dt_s": 60,
            "observed_db": [_round(v, 2) for v in report.observed_excess],
            "predicted_db": [_round(v, 2) for v in report.predicted_excess],
        }
    if with_tracks:
        out["tracks"] = tracks_geojson(report)
    return out


def locate_result(result: LocateResult) -> dict:
    sites = {s.id: s for s in catalog.load()}
    ranking = []
    for s in result.ranking:
        site = sites.get(s.candidate.id)
        ranking.append({
            "id": s.candidate.id,
            "name": s.candidate.name or (site.name if site else s.candidate.id),
            "network": s.candidate.network or (site.network if site else None),
            "lat": s.candidate.lat, "lon": s.candidate.lon,
            "z": _round(s.z, 2), "rho": _round(s.rho, 3),
            "confidence": _round(s.posterior, 3),
            "clock_offset_s": s.offset_s,
            "ais_vessels_nearby": s.n_vessels,
            "ais_coverage": s.covered,
        })
    return {"verdict": result.verdict, "none_of_these": _round(result.none_posterior, 3),
            "heard": result.n_events, "clock_offsets_searched": result.offsets_searched,
            "ranking": ranking}


def grid_result(grid: GridResult) -> dict:
    return {"step_m": grid.step_m, "best": {"lat": grid.best[0], "lon": grid.best[1]},
            "cells": [[round(float(a), 5), round(float(b), 5), _round(c, 2)]
                      for a, b, c in zip(grid.lat, grid.lon, grid.z)]}


def spectrogram(clip: Clip, *, max_frames: int = 1440, fmax: float = 500.0, bins: int = 160) -> dict | None:
    """A display-sized spectrogram: dB above each band's median, on a log-ish frequency axis."""
    if clip.psd_db is None:
        return None
    freqs = np.asarray(clip.freqs, dtype=np.float64)
    edges = np.geomspace(10.0, fmax, bins + 1)
    power = 10.0 ** (np.asarray(clip.psd_db, dtype=np.float64) / 10.0)
    step = max(1, int(np.ceil(len(power) / max_frames)))
    n = len(power) // step
    with np.errstate(invalid="ignore"):
        power = np.nanmean(power[:n * step].reshape(n, step, -1), axis=1)
    out = np.full((n, bins), np.nan)
    for k in range(bins):
        m = (freqs >= edges[k]) & (freqs < edges[k + 1])
        if not m.any():
            m = np.zeros(len(freqs), bool)
            m[int(np.argmin(np.abs(freqs - np.sqrt(edges[k] * edges[k + 1]))))] = True
        out[:, k] = power[:, m].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(out)
    db = db - np.nanmedian(db, axis=0, keepdims=True)
    q = np.clip(np.nan_to_num(db, nan=0.0) * 8.0 + 32.0, 0, 255).astype(np.uint8)   # 1/8 dB steps
    import base64

    return {"t0": iso(clip.t_start), "dt_s": clip.dt * step, "frames": n, "bins": bins,
            "f_lo": 10.0, "f_hi": fmax, "scale": "u8 = dB_above_median * 8 + 32",
            "data": base64.b64encode(q.tobytes()).decode()}
