"""Blind site identification over the SanctSound catalog (the held-out site test).

For every site-day we hold both AIS and a level curve for, hide the site and ask which
of the deployments active that day recorded it. Training days fit the level model;
held-out days are the score that counts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from . import ais
from .detect import band_level
from .regions import region_for
from .soundscape import LevelParams, correlate, excess_from_ranges, highpass, observed_excess, ranges, vessel_positions
from .sources import sanctsound as ss

DT = 60.0
FIT_FIELDS = ("spreading", "absorption", "length_db", "speed_db", "floor_km")


def day_bounds(day: date) -> tuple[float, float]:
    t0 = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
    return t0, t0 + 86400.0


def active_deployments(day: date, deps: list[ss.Deployment] | None = None) -> list[ss.Deployment]:
    """Deployments recording for the whole UTC day, inside a region we hold AIS for."""
    t0, t1 = day_bounds(day)
    deps = deps if deps is not None else ss.fetch_deployments()
    return [d for d in deps
            if d.start.timestamp() <= t0 and d.end.timestamp() >= t1 and region_for(d.lat, d.lon)]


BAND = (20.0, 900.0)


def site_day_level(dep: ss.Deployment, day: date, band: tuple[float, float] = BAND) -> tuple[np.ndarray, str] | None:
    """One-minute band level for a site-day from whatever we already hold: archive
    spectra where published, otherwise cached spectra computed from audio."""
    if dep.site[:2] in ("sb", "gr"):
        try:
            sp = ss.hmd_spectra(dep, day)
        except Exception:
            sp = None
        if sp is not None and len(sp[0]) == 1440:
            return band_level(sp[2], sp[1], *band), "hmd"
    sp = ss.audio_spectra(dep, *day_bounds(day), dt=10.0, cached_only=True)
    if sp is None:
        return None
    level = sp.band(*band)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        per_min = 10.0 * np.log10(np.nanmean(10.0 ** (level.reshape(1440, 6) / 10.0), axis=1))
    return (per_min, "audio") if np.isfinite(per_min).mean() > 0.8 else None


@dataclass
class Case:
    label: str
    site: str
    source: str
    level: np.ndarray                 # one-minute band level for the day
    candidates: list[str]
    geoms: list[tuple]                # per candidate: (range (V, T), speed (V, T), length, tclass)
    truth: int
    start: int = 0                    # clip start, minutes into the day
    minutes: int = 1440


def build_cases(days: list[date], *, sites: tuple[str, ...] | None = None, log=None,
                band: tuple[float, float] = BAND) -> list[Case]:
    cases = []
    for day in days:
        d0, d1 = day_bounds(day)
        tgrid = d0 + DT * (np.arange(1440) + 0.5)
        active = active_deployments(day)
        levels = {}
        for dep in active:
            if sites and dep.site[:2] not in sites and dep.site not in sites:
                continue
            got = site_day_level(dep, day, band)
            if got is not None:
                levels[dep.site] = got
        if not levels:
            continue
        geoms = []
        for dep in active:
            tracks = ais.load_tracks(dep.lat, dep.lon, d0, d1, 45_000.0)
            vid, r, spd = ranges(vessel_positions(tracks, tgrid), 0.0, 0.0)
            geoms.append((r, spd, tracks.length[vid], tracks.tclass[vid]))
        names = [d.site for d in active]
        for site, (level, source) in levels.items():
            cases.append(Case(f"{site} {day}", site, source, level, names, geoms, names.index(site)))
        if log:
            log(f"{day}: {len(levels)} clips, {len(active)} candidates")
    return cases


def subclips(cases: list[Case], hours: float) -> list[Case]:
    """Cut each day into consecutive clips of the given length."""
    n = int(hours * 60)
    return [replace(c, label=f"{c.label} +{a // 60:02d}h", start=a, minutes=n)
            for c in cases for a in range(0, 1440 - n + 1, n)]


def scores(case: Case, params: LevelParams) -> tuple[np.ndarray, np.ndarray]:
    """(rho, z) per candidate. The observed floor is taken over the whole day so that a
    sub-clip is scored the way it would be if cut from a longer recording."""
    sl = slice(case.start, case.start + case.minutes)
    obs = observed_excess(case.level, DT)[sl]
    pred = np.stack([highpass(excess_from_ranges(*g, params), DT)[sl] for g in case.geoms])
    rho, z, _ = correlate(obs, pred, min_lag=min(45, max(5, case.minutes // 6)))
    return rho, z


def evaluate(cases: list[Case], params: LevelParams) -> dict:
    hits = hits3 = 0
    loss = 0.0
    rows = []
    for c in cases:
        rho, z = scores(c, params)
        order = np.argsort(-z)
        rank = int(np.nonzero(order == c.truth)[0][0]) + 1
        hits += int(rank == 1 and z[c.truth] > 0)
        hits3 += int(rank <= 3)
        loss -= z[c.truth] - logsumexp(np.r_[z, 0.0])
        rows.append({"label": c.label, "site": c.site, "source": c.source, "rho": float(rho[c.truth]),
                     "z": float(z[c.truth]), "rank": rank, "top": c.candidates[order[0]],
                     "top_z": float(z[order[0]]),
                     "runner_up_z": float(z[order[1]]) if len(order) > 1 else 0.0})
    n = max(len(cases), 1)
    return {"n": len(cases), "top1": hits, "top3": hits3, "top1_rate": hits / n, "loss": loss / n, "rows": rows}


def fit(cases: list[Case], start: LevelParams | None = None, *, maxiter: int = 300) -> LevelParams:
    start = start or LevelParams()
    x0 = np.array([getattr(start, f) for f in FIT_FIELDS])

    def unpack(x: np.ndarray) -> LevelParams:
        v = dict(zip(FIT_FIELDS, (float(a) for a in x)))
        v["floor_km"] = float(np.clip(v["floor_km"], 1.0, 60.0))
        v["absorption"] = max(v["absorption"], 0.0)
        return replace(start, **v)

    res = minimize(lambda x: evaluate(cases, unpack(x))["loss"], x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 0.05, "fatol": 1e-3})
    return unpack(res.x)
