"""NOAA SanctSound (2018-2022): 31 moored recorders in US marine sanctuaries.

Read from the public `noaa-passive-bioacoustic` bucket. Three products matter here:

- ship detections: start/end times of vessel noise events, one CSV per deployment
- hmd: daily NetCDF of 1-minute hybrid-millidecade spectra
- audio: 6 h FLAC files (about 600 MB each)

Call shapes are copied from probes/gcs_sanctsound.py.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..fetch import download, download_many, get_json
from ..paths import cache_path

API = "https://storage.googleapis.com/storage/v1/b/noaa-passive-bioacoustic/o"
OBJ = "https://storage.googleapis.com/noaa-passive-bioacoustic/"


def list_objects(*, prefix: str | None = None, match_glob: str | None = None) -> list[dict]:
    params = {"fields": "items(name,size),nextPageToken", "maxResults": 1000}
    if prefix:
        params["prefix"] = prefix
    if match_glob:
        params["matchGlob"] = match_glob
    items: list[dict] = []
    while True:
        page = get_json(API, params)
        items += page.get("items", [])
        if not page.get("nextPageToken"):
            return items
        params["pageToken"] = page["nextPageToken"]


@dataclass(frozen=True)
class Deployment:
    site: str            # "sb01"
    deployment: str      # "05"
    lat: float
    lon: float
    start: datetime
    end: datetime

    @property
    def key(self) -> str:
        return f"{self.site}_{self.deployment}"


def _parse_dt(text: str) -> datetime:
    return datetime.strptime(text, "%d-%b-%Y %H:%M:%S").replace(tzinfo=UTC)


def fetch_deployments(refresh: bool = False) -> list[Deployment]:
    """Every deployment that has a ship-detection product, with position and dates."""
    index = cache_path("sanctsound", "deployments.json")
    if index.exists() and not refresh:
        rows = json.loads(index.read_text())
    else:
        items = list_objects(
            match_glob="sanctsound/products/detections/*/*_ships/metadata/*-metadata.json")
        pairs = [(OBJ + it["name"], cache_path("sanctsound", "meta", Path(it["name"]).name))
                 for it in items]
        rows = []
        for path in download_many(pairs):
            meta = json.loads(path.read_text())
            lon, lat = map(float, re.match(r"POINT \(([-\d.]+) ([-\d.]+)\)", meta["SHAPE"]).groups())
            rows.append({
                "site": meta["SITE_NAME"].lower(), "deployment": meta["DEPLOYMENT_NAME"],
                "lat": lat, "lon": lon,
                "start": _parse_dt(meta["START_TIME"]).isoformat(),
                "end": _parse_dt(meta["END_TIME"]).isoformat(),
            })
        # The bucket holds one misfiled duplicate (hi01/sanctsound_hi03_03_ships), so key on
        # what the metadata says and keep one row per site and deployment.
        rows = list({(r["site"], r["deployment"]): r for r in rows}.values())
        rows.sort(key=lambda r: (r["site"], r["deployment"]))
        index.write_text(json.dumps(rows, indent=1))
    return [Deployment(r["site"], r["deployment"], r["lat"], r["lon"],
                       datetime.fromisoformat(r["start"]), datetime.fromisoformat(r["end"]))
            for r in rows]


def ships_csv(dep: Deployment) -> Path:
    name = f"SanctSound_{dep.site.upper()}_{dep.deployment}_ships"
    url = (f"{OBJ}sanctsound/products/detections/{dep.site}/"
           f"sanctsound_{dep.site}_{dep.deployment}_ships/data/{name}.csv")
    return download(url, cache_path("sanctsound", "ships", f"{name}.csv"))


def ship_detections(dep: Deployment) -> np.ndarray:
    """(n, 2) float64 array of [start, end] POSIX seconds for each logged vessel event."""
    rows = []
    with ships_csv(dep).open(encoding="utf-8-sig") as f:  # some files carry a BOM
        for row in csv.DictReader(f):
            if "ship" not in (row.get("Labels") or "").lower():
                continue
            a = datetime.fromisoformat(row["ISOStartTime"].replace("Z", "+00:00")).timestamp()
            b = datetime.fromisoformat(row["ISOEndTime"].replace("Z", "+00:00")).timestamp()
            rows.append((a, b))
    return np.array(rows, dtype=np.float64).reshape(-1, 2)


def hmd_file(dep: Deployment, day) -> Path | None:
    """Daily NetCDF of 1-minute hybrid-millidecade spectra, or None if not published.
    Only some sites (Stellwagen, Gray's Reef) have this product."""
    index = cache_path("sanctsound", "hmd", f"index_{dep.key}.json")
    if index.exists():
        names = json.loads(index.read_text())
    else:
        prefix = f"sanctsound/products/sound_level_metrics/{dep.site}/sanctsound_{dep.site}_{dep.deployment}_hmd/data/"
        names = [it["name"] for it in list_objects(prefix=prefix) if it["name"].endswith(".nc")]
        index.write_text(json.dumps(names))
    stamp = f"_{day:%Y%m%d}.nc"
    name = next((n for n in names if n.endswith(stamp)), None)
    return download(OBJ + name, cache_path("sanctsound", "hmd", Path(name).name)) if name else None


def hmd_spectra(dep: Deployment, day) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(t POSIX s, freqs Hz, psd dB re 1 uPa^2/Hz with shape (T, F)) for one UTC day."""
    import netCDF4

    path = hmd_file(dep, day)
    if path is None:
        return None
    with netCDF4.Dataset(path) as ds:
        freqs = np.asarray(ds["frequency"][:], dtype=np.float64)
        minutes = np.asarray(ds["time"][:], dtype=np.float64)
        psd = np.ma.filled(ds["psd"][:].astype(np.float64), np.nan)
    day0 = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
    return day0 + minutes * 60.0, freqs, psd


# Two filename styles exist: ..._20181113T002444Z.flac and ..._200131150000.flac (yymmddHHMMSS).
_AUDIO_STAMP = re.compile(r"_(?:(\d{8}T\d{6})Z?|(\d{12}))(?:_[\w-]+)?\.(?:flac|wav)$", re.IGNORECASE)


def audio_files(dep: Deployment) -> list[tuple[float, str, int]]:
    """(start POSIX s, object name, bytes) for every audio file of a deployment, time-sorted.
    Files are about six hours long; the UTC start time is in the filename."""
    index = cache_path("sanctsound", "audio", f"index_{dep.key}.json")
    if index.exists():
        rows = json.loads(index.read_text())
    else:
        prefix = f"sanctsound/audio/{dep.site}/sanctsound_{dep.site}_{dep.deployment}/audio/"
        rows = []
        for it in list_objects(prefix=prefix):
            m = _AUDIO_STAMP.search(it["name"])
            if m:
                start = (datetime.strptime(m.group(1), "%Y%m%dT%H%M%S") if m.group(1)
                         else datetime.strptime(m.group(2), "%y%m%d%H%M%S")).replace(tzinfo=UTC)
                rows.append([start.timestamp(), it["name"], int(it["size"])])
        rows.sort()
        index.write_text(json.dumps(rows))
    return [tuple(r) for r in rows]


def audio_for_window(dep: Deployment, t0: float, t1: float) -> list[tuple[float, str, int]]:
    """Audio files that overlap [t0, t1)."""
    files = audio_files(dep)
    out = []
    for k, (start, name, size) in enumerate(files):
        end = files[k + 1][0] if k + 1 < len(files) else start + 6 * 3600
        if start < t1 and end > t0:
            out.append((start, name, size))
    return out


def fetch_audio(name: str) -> Path:
    return download(OBJ + name, cache_path("sanctsound", "audio", Path(name).name))


def _lf_path(name: str) -> Path:
    return cache_path("sanctsound", "lf", Path(name).stem + ".2k.flac")


def lf_copies(dep: Deployment, t0: float, t1: float) -> list[tuple[float, Path]]:
    """2 kHz copies of the audio overlapping [t0, t1), streamed from the bucket by ffmpeg."""
    from ..ingest import lowrate_copy

    return [(start, lowrate_copy(OBJ + name, _lf_path(name)))
            for start, name, _ in audio_for_window(dep, t0, t1)]


def read_lf(dep: Deployment, t0: float, t1: float) -> tuple[np.ndarray, int]:
    """2 kHz samples for [t0, t1), stitched across files. Gaps are zero-filled."""
    from ..ingest import LF_RATE, read_audio

    out = np.zeros(int(round((t1 - t0) * LF_RATE)), dtype=np.float32)
    for start, path in lf_copies(dep, t0, t1):
        skip = max(0.0, t0 - start)
        x, sr = read_audio(path, skip, t1 - max(t0, start))
        a = int(round((max(t0, start) - t0) * sr))
        n = min(len(x), len(out) - a)
        if n > 0:
            out[a:a + n] = x[:n]
    return out, LF_RATE


def audio_spectra(dep: Deployment, t0: float, t1: float, dt: float = 10.0, *, keep_lf: bool = True,
                  cached_only: bool = False):
    """Spectra (1 Hz bins, `dt` s frames) for [t0, t1) on a uniform grid; gaps are NaN.
    Per-file spectra are cached, so the 2 kHz copy is only needed once (keep_lf=False
    deletes it afterwards). cached_only returns None rather than fetch anything."""
    from ..ingest import Spectra, lowrate_copy, read_audio, spectra

    t0 = np.ceil(t0 / dt) * dt
    n = int((t1 - t0) // dt)
    grid = t0 + dt * np.arange(n)
    psd = None
    freqs = None
    index = cache_path("sanctsound", "audio", f"index_{dep.key}.json")
    if cached_only and not index.exists():
        return None
    for start, name, _ in audio_for_window(dep, t0, t1):
        lf = _lf_path(name)
        cache = lf.with_suffix(f".psd{int(dt)}.npz")
        if cache.exists():
            z = np.load(cache)
            ft, freqs, fp = z["t"], z["freqs"], z["psd_db"].astype(np.float32)
        elif cached_only:
            return None
        else:
            had = lf.exists()
            x, sr = read_audio(lowrate_copy(OBJ + name, lf))
            sp = spectra(x, sr, start, dt=dt)
            ft, freqs, fp = sp.t, sp.freqs, sp.psd_db
            np.savez_compressed(cache, t=ft, freqs=freqs, psd_db=fp.astype(np.float16))
            if not keep_lf and not had:
                lf.unlink(missing_ok=True)
        if psd is None:
            psd = np.full((n, len(freqs)), np.nan, dtype=np.float32)
        k = np.round((ft - t0) / dt).astype(int)
        ok = (k >= 0) & (k < n)
        psd[k[ok]] = fp[ok]
    if psd is None:
        return None
    return Spectra(grid, freqs, psd, dt)
