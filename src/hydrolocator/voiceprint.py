"""Vessel voiceprints: the narrow lines a ship's machinery leaves in the spectrum.

A passage's signature is its spectrum at closest approach with the smooth continuum and
the site's standing lines removed, so mostly blade and engine tonals remain. Signatures
of AIS-matched passages form a library; an unexplained sound can then be compared with it.

Measured on Stellwagen with one-minute, 1 Hz archive spectra: repeat passages of the same
vessel are more alike than passages of different vessels (AUC 0.67), and the nearest
library signature names the right vessel about one time in four against 2 % by chance.
That makes this a hint to follow up, never an identification. Naval, coast guard and
law-enforcement vessels are left out of the library.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .paths import cache_path

GRID = np.arange(25.0, 451.0)          # Hz, the common axis every signature is put on
CONTINUUM_HZ = 21
WINDOW_S = 150.0


def line_spectra(freqs: np.ndarray, psd_db: np.ndarray) -> np.ndarray:
    """Lines above the smooth continuum for every frame, on GRID, with the clip's standing
    lines (mains hum, recorder noise) taken off. Shape (T, len(GRID))."""
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.asarray(psd_db, dtype=np.float64)
    fill = np.nanmedian(psd)
    on_grid = np.stack([np.interp(GRID, freqs, np.where(np.isfinite(row), row, fill)) for row in psd])
    lines = on_grid - median_filter(on_grid, (1, CONTINUUM_HZ), mode="nearest")
    return lines - np.median(lines, axis=0, keepdims=True)


def signature(lines: np.ndarray, t_frames: np.ndarray, t_event: float) -> np.ndarray | None:
    """Unit vector for the passage at t_event, or None when nothing stands out."""
    near = np.abs(np.asarray(t_frames) - t_event) <= WINDOW_S
    if near.sum() < 1:
        return None
    vec = uniform_filter1d(np.maximum(np.median(lines[near], axis=0), 0.0), 3)
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm > 1e-6 else None


@dataclass
class Library:
    vectors: np.ndarray          # (N, len(GRID)) unit vectors
    mmsi: np.ndarray
    name: np.ndarray
    vessel_type: np.ndarray
    site: np.ndarray
    waters: np.ndarray           # AIS region id of the site
    t: np.ndarray
    calibration: list[tuple[float, float]]   # (similarity threshold, share of nearest matches that were right)

    def __len__(self) -> int:
        return len(self.mmsi)

    def trust(self, similarity: float) -> float:
        """Share of nearest matches at or above this similarity that named the right vessel."""
        share = 0.0
        for threshold, right in self.calibration:
            if similarity >= threshold:
                share = right
        return share


@dataclass
class Hint:
    mmsi: int
    name: str
    vessel_type: int
    similarity: float
    trust: float                 # how often a match this close was right in the library itself
    heard_at: str                # site and date of the library passage it resembles


def library_path():
    return cache_path("voiceprints", "library.npz")


def load_library() -> Library | None:
    path = library_path()
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return Library(z["vectors"].astype(np.float32), z["mmsi"], z["name"], z["vessel_type"], z["site"], z["waters"], z["t"],
                   [tuple(x) for x in json.loads(str(z["calibration"]))])


def suggest(library: Library | None, sig: np.ndarray | None, *, waters: str | None = None,
            exclude_mmsi: int | None = None, limit: int = 2, min_similarity: float = 0.3) -> list[Hint]:
    """The library vessels whose signature is closest to `sig`, best first. `waters` (an
    AIS region id such as "stellwagen") keeps the comparison to ships recorded
    in the same waters, where the same vessels come back and the spectra are comparable."""
    if library is None or sig is None or len(library) == 0:
        return []
    sim = library.vectors @ sig
    if waters:
        sim = np.where(library.waters == waters, sim, -1.0)
    best: dict[int, int] = {}
    for i in np.argsort(-sim):
        m = int(library.mmsi[i])
        if m == exclude_mmsi or m in best:
            continue
        if sim[i] < min_similarity or len(best) >= limit:
            break
        best[m] = int(i)
    return [Hint(m, str(library.name[i]), int(library.vessel_type[i]), float(sim[i]), library.trust(float(sim[i])),
                 f"{library.site[i]} on {date.fromtimestamp(float(library.t[i])).isoformat()}") for m, i in best.items()]


def build_library(log=None) -> Library:
    """Signatures of every AIS-matched passage in the site-days we hold spectra for."""
    from . import ais, cpa, shipdata
    from . import blindtest as bt
    from .detect import band_level, detect_passages
    from .regions import region_for
    from .sources import sanctsound as ss

    vectors, meta = [], []
    for day in ais.available_days():
        d0, d1 = bt.day_bounds(day)
        for dep in bt.active_deployments(day):
            spectra = None
            if dep.site[:2] in ("sb", "gr"):
                try:
                    hmd = ss.hmd_spectra(dep, day)
                except Exception:  # noqa: BLE001 - a missing or broken archive file just means no spectra
                    hmd = None
                if hmd is not None and len(hmd[0]) == 1440:
                    spectra = (hmd[0] + 30.0, hmd[1], hmd[2], band_level(hmd[2], hmd[1], 20.0, 900.0))
            if spectra is None:
                sp = ss.audio_spectra(dep, d0, d1, dt=10.0, cached_only=True)
                if sp is None:
                    continue
                per_min = bt.site_day_level(dep, day)
                if per_min is None:
                    continue
                spectra = (sp.t + sp.dt / 2, sp.freqs, sp.psd_db, per_min[0])
            t_frames, freqs, psd, level = spectra
            events = detect_passages(d0 + 60.0 * np.arange(1440), level)
            if not len(events):
                continue
            lines = line_spectra(freqs, psd)
            tracks = ais.load_tracks(dep.lat, dep.lon, d0, d1, 30_000.0)
            passes = cpa.predict(tracks, 0.0, 0.0, 6_000.0)
            for tc, r, v, vid in zip(passes.t, passes.r, passes.v, passes.vid):
                if v < 1.0 or shipdata.is_government(int(tracks.vessel_type[vid]), str(tracks.name[vid]),
                                                      callsign=str(tracks.callsign[vid]), mmsi=int(tracks.mmsi[vid])):
                    continue
                j = int(np.argmin(np.abs(events.t - tc)))
                if abs(events.t[j] - tc) > 180.0 + 0.5 * r / v or events.meta["prominence"][j] < 5.0:
                    continue
                sig = signature(lines, t_frames, float(events.t[j]))
                if sig is not None:
                    vectors.append(sig)
                    meta.append((int(tracks.mmsi[vid]), str(tracks.name[vid]), int(tracks.vessel_type[vid]), dep.site,
                                 float(events.t[j]), region_for(dep.lat, dep.lon).id))
        if log:
            log(f"{day}: {len(vectors)} signatures")
    V = np.stack(vectors) if vectors else np.zeros((0, len(GRID)), dtype=np.float32)
    mmsi = np.array([m[0] for m in meta], dtype=np.int64)

    # How far to trust a nearest match, measured inside the library: leave each passage
    # out, find its nearest neighbour, and see how often that is the same vessel.
    calibration = []
    if len(mmsi) > 2:
        S = V @ V.T
        np.fill_diagonal(S, -1.0)
        waters = np.array([m[5] for m in meta])
        S[waters[:, None] != waters[None, :]] = -1.0          # compare within the same waters only
        nearest = np.argmax(S, axis=1)
        top = S[np.arange(len(mmsi)), nearest]
        repeat = np.array([((mmsi == m) & (waters == w)).sum() > 1 for m, w in zip(mmsi, waters)])
        right = mmsi[nearest] == mmsi
        for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
            sel = repeat & (top >= threshold)
            if sel.sum() >= 5:
                calibration.append((threshold, round(float(right[sel].mean()), 2)))
    lib = Library(V, mmsi, np.array([m[1] for m in meta], dtype=object), np.array([m[2] for m in meta], dtype=np.int64),
                  np.array([m[3] for m in meta], dtype=object), np.array([m[5] for m in meta], dtype=object),
                  np.array([m[4] for m in meta]), calibration)
    np.savez_compressed(library_path(), vectors=V.astype(np.float16), mmsi=lib.mmsi, name=lib.name,
                        vessel_type=lib.vessel_type, site=lib.site, waters=lib.waters, t=lib.t,
                        calibration=json.dumps(calibration))
    return lib
