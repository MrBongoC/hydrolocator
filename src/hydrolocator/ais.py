"""AIS store: MarineCadastre daily CSV -> one Parquet file per UTC day, clipped to the v1
regions. DuckDB does the CSV parsing and the bbox queries.

MarineCadastre points are thinned to about one per minute per vessel, so tracks are
piecewise linear with 60 s legs.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import numpy as np

from .fetch import download
from .geo import bbox_around, to_xy
from .paths import cache_path
from .regions import REGIONS

URL = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{y}/AIS_{y}_{m:02d}_{d:02d}.zip"

# Longest gap we bridge with a straight line. Class B and weak-signal vessels can drop
# out for several minutes; past this we do not guess where the ship went.
MAX_GAP_S = 900.0


def raw_zip(day: date) -> Path:
    return cache_path("ais", "raw", f"AIS_{day.year}_{day.month:02d}_{day.day:02d}.zip")


def points_path(day: date) -> Path:
    return cache_path("ais", "points", f"{day.isoformat()}.parquet")


def ingest_day(day: date, *, keep_zip: bool = True) -> Path:
    """Download (if needed) and convert one day to Parquet. Returns the Parquet path."""
    out = points_path(day)
    if out.exists() and has_identity(out):
        return out
    zpath = download(URL.format(y=day.year, m=day.month, d=day.day), raw_zip(day))
    tmp_csv = cache_path("ais", "tmp", f"{day.isoformat()}.csv")
    with zipfile.ZipFile(zpath) as z:
        member = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(member) as src, tmp_csv.open("wb") as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
    where = " OR ".join(
        f"(LAT BETWEEN {r.lat_min} AND {r.lat_max} AND LON BETWEEN {r.lon_min} AND {r.lon_max})"
        for r in REGIONS.values())
    tmp_out = out.with_suffix(".parquet.part")
    con = duckdb.connect()
    src = f"read_csv('{tmp_csv}', header=true, all_varchar=true, strict_mode=false, ignore_errors=true)"
    # Column names drift between years (2021 files spell it "TranscieverClass").
    have = {row[0].lower(): row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}

    def col(*names: str) -> str:
        for name in names:
            if name.lower() in have:
                return f'"{have[name.lower()]}"'
        return "NULL"

    con.execute(f"""
        COPY (
            WITH pts AS (
                SELECT CAST({col('MMSI')} AS BIGINT) AS mmsi,
                       epoch(CAST({col('BaseDateTime')} AS TIMESTAMP)) AS t,
                       TRY_CAST({col('LAT')} AS DOUBLE) AS LAT, TRY_CAST({col('LON')} AS DOUBLE) AS LON,
                       TRY_CAST({col('SOG')} AS FLOAT) AS sog, TRY_CAST({col('COG')} AS FLOAT) AS cog,
                       {col('VesselName')} AS name, TRY_CAST({col('VesselType')} AS INTEGER) AS vessel_type,
                       TRY_CAST({col('Length')} AS FLOAT) AS length, TRY_CAST({col('Width')} AS FLOAT) AS width,
                       TRY_CAST({col('Draft')} AS FLOAT) AS draft,
                       {col('IMO')} AS imo, {col('CallSign')} AS callsign,
                       TRY_CAST({col('Cargo')} AS INTEGER) AS cargo, TRY_CAST({col('Status')} AS INTEGER) AS status,
                       {col('TransceiverClass', 'TranscieverClass')} AS tclass
                FROM {src}
            )
            SELECT mmsi, t, LAT AS lat, LON AS lon, sog, cog, name, vessel_type, length, width, draft,
                   imo, callsign, cargo, status, tclass
            FROM pts WHERE {where}
            ORDER BY mmsi, t
        ) TO '{tmp_out}' (FORMAT parquet, COMPRESSION zstd)
    """)
    con.close()
    tmp_out.replace(out)
    tmp_csv.unlink()
    if not keep_zip:
        zpath.unlink()
    return out


def has_identity(path: Path) -> bool:
    """True when a day file carries the identity fields (IMO, call sign, width, cargo, status)."""
    con = duckdb.connect()
    cols = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM '{path}'").fetchall()}
    con.close()
    return {"imo", "callsign", "width", "cargo", "status"} <= cols


def available_days() -> list[date]:
    return sorted(date.fromisoformat(p.stem) for p in cache_path("ais", "points", "x").parent.glob("*.parquet"))


@dataclass
class Tracks:
    """Vessel tracks near one origin, as flat arrays of straight legs.

    Leg k runs from (x0, y0) at t0 to (x1, y1) at t1 and belongs to vessel `vid[k]`
    (an index into the static arrays). Legs of one vessel are contiguous and time-ordered.
    """
    lat0: float
    lon0: float
    t_start: float
    t_end: float
    vid: np.ndarray
    t0: np.ndarray
    t1: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    x1: np.ndarray
    y1: np.ndarray
    mmsi: np.ndarray
    name: np.ndarray
    vessel_type: np.ndarray
    length: np.ndarray
    tclass: np.ndarray
    imo: np.ndarray = None
    callsign: np.ndarray = None
    width: np.ndarray = None
    draft: np.ndarray = None

    @property
    def n_vessels(self) -> int:
        return len(self.mmsi)

    @property
    def n_legs(self) -> int:
        return len(self.vid)


def _days_between(t_start: float, t_end: float) -> list[date]:
    d = datetime.fromtimestamp(t_start, UTC).date()
    last = datetime.fromtimestamp(t_end, UTC).date()
    days = []
    while d <= last:
        days.append(d)
        d += timedelta(days=1)
    return days


def coverage(t_start: float, t_end: float) -> tuple[list[date], list[date]]:
    """(days we hold, days missing) for a time window."""
    days = _days_between(t_start, t_end)
    have = [d for d in days if points_path(d).exists()]
    return have, [d for d in days if d not in have]


def load_tracks(lat0: float, lon0: float, t_start: float, t_end: float,
                radius_m: float = 60_000.0, *, pad_s: float = 1800.0) -> Tracks:
    """All vessel legs within radius_m of (lat0, lon0) during [t_start, t_end] (+/- pad)."""
    have, _ = coverage(t_start - pad_s, t_end + pad_s)
    if not have:
        return build_tracks({"mmsi": [], "t": [], "lat": [], "lon": []}, {}, lat0, lon0, t_start, t_end)
    la0, la1, lo0, lo1 = bbox_around(lat0, lon0, radius_m)
    files = ", ".join(f"'{points_path(d)}'" for d in have)
    src = f"""read_parquet([{files}])
        WHERE lat BETWEEN {la0} AND {la1} AND lon BETWEEN {lo0} AND {lo1}
          AND t BETWEEN {t_start - pad_s} AND {t_end + pad_s}"""
    con = duckdb.connect()
    rows = con.execute(f"SELECT mmsi, t, lat, lon FROM {src} ORDER BY mmsi, t").fetchnumpy()
    static = con.execute(f"""
        SELECT mmsi, coalesce(max(name), '') AS name,
               coalesce(max(vessel_type), -1)::DOUBLE AS vessel_type,
               coalesce(max(length), 'nan'::FLOAT)::DOUBLE AS length,
               coalesce(max(tclass), '') AS tclass,
               coalesce(max(nullif(replace(imo, 'IMO', ''), '0000000')), '') AS imo,
               coalesce(max(callsign), '') AS callsign,
               coalesce(max(width), 'nan'::FLOAT)::DOUBLE AS width,
               coalesce(max(nullif(draft, 0)), 'nan'::FLOAT)::DOUBLE AS draft
        FROM {src} GROUP BY mmsi ORDER BY mmsi""").fetchnumpy()
    con.close()
    return build_tracks(rows, static, lat0, lon0, t_start, t_end)


def build_tracks(rows: dict, static: dict, lat0: float, lon0: float,
                 t_start: float, t_end: float) -> Tracks:
    """Turn time-sorted AIS fixes (grouped by mmsi) into legs.

    `static` holds per-vessel fields keyed by mmsi (name, vessel_type, length, tclass);
    anything missing is filled with a neutral default.
    """
    mmsi = np.asarray(rows["mmsi"], dtype=np.int64)
    t = np.asarray(rows["t"], dtype=np.float64)
    x, y = to_xy(np.asarray(rows["lat"], dtype=np.float64),
                 np.asarray(rows["lon"], dtype=np.float64), lat0, lon0)
    uniq, inverse = np.unique(mmsi, return_inverse=True)
    if len(mmsi) >= 2:
        dt = t[1:] - t[:-1]
        ok = (mmsi[1:] == mmsi[:-1]) & (dt > 0) & (dt <= MAX_GAP_S)
        # Drop position glitches: legs that imply more than 45 kn.
        ok &= np.hypot(x[1:] - x[:-1], y[1:] - y[:-1]) <= 23.0 * np.maximum(dt, 1.0)
        k = np.nonzero(ok)[0]
    else:
        k = np.array([], dtype=np.int64)

    def field(col, fill, dtype):
        out = np.full(len(uniq), fill, dtype=dtype)
        if col in static and len(static.get("mmsi", [])):
            pos = np.searchsorted(uniq, np.asarray(static["mmsi"], dtype=np.int64))
            vals = static[col]
            vals = np.ma.filled(vals, fill) if np.ma.isMaskedArray(vals) else np.asarray(vals)
            good = (pos < len(uniq))
            good[good] &= uniq[pos[good]] == np.asarray(static["mmsi"], dtype=np.int64)[good]
            out[pos[good]] = vals[good]
        return out

    return Tracks(
        lat0, lon0, t_start, t_end,
        vid=inverse[k].astype(np.int64), t0=t[k], t1=t[k + 1],
        x0=x[k], y0=y[k], x1=x[k + 1], y1=y[k + 1],
        mmsi=uniq,
        name=field("name", "", object),
        vessel_type=field("vessel_type", -1.0, np.float64),
        length=field("length", np.nan, np.float64),
        tclass=field("tclass", "", object),
        imo=field("imo", "", object),
        callsign=field("callsign", "", object),
        width=field("width", np.nan, np.float64),
        draft=field("draft", np.nan, np.float64),
    )
