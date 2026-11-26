"""JSON API. Also hosts a front-end build when one is present in web/dist.

Analyses run as background jobs. A finished result is written to data/results/<id>.json
and stays readable at /api/results/<id>, which is what a shared link points at. Uploaded
audio is deleted as soon as its analysis finishes; only the derived result is kept.
"""

from __future__ import annotations

import json
import secrets
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import ais, catalog, report, soundmap
from .paths import cache_path, data_dir
from .pipeline import Candidate, Clip, clip_from_audio, grid_search, iso, locate, who_passed
from .public import public_clip
from .regions import REGIONS

app = FastAPI(title="Hydrolocator", docs_url="/api/docs", openapi_url="/api/openapi.json")
_pool = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
MAX_UPLOAD_BYTES = 4 * 1024 ** 3
MAX_HOURS = 48.0


def _results_dir() -> Path:
    path = data_dir() / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_time(text: str) -> float:
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as err:
        raise HTTPException(422, f"could not read time {text!r}") from err
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def _set(job_id: str, **fields) -> None:
    with _lock:
        _jobs[job_id].update(fields)


# ------------------------------------------------------------------ reference data

@app.get("/api/catalog")
def get_catalog(date: str | None = None):
    t = _parse_time(date) if date else None
    days = {d.isoformat() for d in ais.available_days()}
    sites = []
    for s in catalog.load():
        dep = s.active(t, t + 86400.0) if t is not None else None
        sites.append({
            "id": s.id, "name": s.name, "network": s.network, "lat": s.lat, "lon": s.lon,
            "depth_m": s.depth_m, "region": s.region, "source": s.source,
            "recording": dep is not None if t is not None else None,
            "deployments": [{"start": iso(d.start), "end": iso(d.end)} for d in s.deployments],
        })
    return {"sites": sites, "ais_days": sorted(days), "soundmap_days": soundmap.available("stellwagen"),
            "regions": [{"id": r.id, "name": r.name, "bbox": [r.lon_min, r.lat_min, r.lon_max, r.lat_max]}
                        for r in REGIONS.values()]}


# ------------------------------------------------------------------ analysis jobs

def _analyse(job_id: str, mode: str, clip_fn, site_id: str | None, lat: float | None, lon: float | None,
             clock_hours: float, refine: bool, cleanup: Path | None) -> None:
    started = time.time()
    try:
        _set(job_id, status="running", step="Reading the recording")
        clip: Clip = clip_fn()
        events = clip.events()
        result: dict = {
            "id": job_id, "created": iso(started), "mode": mode,
            "clip": {"label": clip.label, "start": iso(clip.t_start), "end": iso(clip.t_end),
                     "hours": round(clip.hours, 2)},
            "events": [{"t": iso(float(t)), "prominence_db": round(float(p), 1)}
                       for t, p in zip(events.t, events.meta.get("prominence", []))],
            "spectrogram": report.spectrogram(clip),
            "truth": None, "locate": None, "site": None, "evidence": None, "grid": None,
        }
        offset = 0.0
        if mode == "where":
            _set(job_id, step="Scoring candidate sites against AIS")
            cands = [Candidate(s.id, d.lat, d.lon, s.name, s.network)
                     for s, d in catalog.active_sites(clip.t_start, clip.t_end)]
            found = locate(clip, cands, clock_uncertainty_s=clock_hours * 3600.0)
            result["locate"] = report.locate_result(found)
            if site_id:                                   # a public recording located blind
                truth = catalog.get(site_id)
                rank = next((k + 1 for k, s in enumerate(found.ranking) if s.candidate.id == truth.id), None)
                result["truth"] = {"id": truth.id, "name": truth.name, "rank": rank}
            best = found.best
            if best is not None and best.covered and best.z > 0:
                lat, lon, offset = best.candidate.lat, best.candidate.lon, best.offset_s
                result["site"] = {"id": best.candidate.id, "name": best.candidate.name, "lat": lat, "lon": lon}
        else:
            if site_id:
                s = catalog.get(site_id)
                dep = s.active(clip.t_start, clip.t_end)
                lat, lon = (dep.lat, dep.lon) if dep else (s.lat, s.lon)
                result["site"] = {"id": s.id, "name": s.name, "lat": lat, "lon": lon}
            else:
                result["site"] = {"id": None, "name": "Given position", "lat": lat, "lon": lon}
        if lat is not None and lon is not None:
            _set(job_id, step="Matching passages to vessels")
            result["evidence"] = report.site_report(who_passed(clip, lat, lon, offset_s=offset))
            _set(job_id, step="Looking up vessel records")
            report.enrich(result["evidence"]["passages"])
            if refine:
                _set(job_id, step="Searching the surrounding grid")
                result["grid"] = report.grid_result(grid_search(clip, lat, lon, offset_s=offset))
        result["seconds"] = round(time.time() - started, 1)
        (_results_dir() / f"{job_id}.json").write_text(json.dumps(result))
        _set(job_id, status="done", step="Done")
    except LookupError as err:
        _set(job_id, status="failed", error=str(err))
    except Exception as err:  # noqa: BLE001 - report anything to the client rather than hang the job
        traceback.print_exc()
        _set(job_id, status="failed", error=f"{type(err).__name__}: {err}")
    finally:
        if cleanup is not None:
            shutil.rmtree(cleanup, ignore_errors=True)


def _tracks(job_id: str, site_a: str, site_b: str, t0: float, hours: float) -> None:
    started = time.time()
    try:
        _set(job_id, status="running", step="Reading both recordings")
        from . import multisite

        sa, sb = catalog.get(site_a), catalog.get(site_b)
        found = multisite.analyse(sa.id, sb.id, t0, t0 + hours * 3600.0)
        result = {
            "id": job_id, "created": iso(started), "mode": "tracks",
            "clip": {"label": f"{sa.name} and {sb.name}", "start": iso(t0), "end": iso(t0 + hours * 3600.0), "hours": round(hours, 2)},
            "events": [], "spectrogram": None, "truth": None, "locate": None, "evidence": None, "grid": None,
            "site": {"id": sa.id, "name": f"{sa.name} and {sb.name}", "lat": sa.lat, "lon": sa.lon},
            "tracks": found, "seconds": round(time.time() - started, 1),
        }
        (_results_dir() / f"{job_id}.json").write_text(json.dumps(result))
        _set(job_id, status="done", step="Done")
    except (LookupError, KeyError) as err:
        _set(job_id, status="failed", error=str(err))
    except Exception as err:  # noqa: BLE001 - report anything to the client rather than hang the job
        traceback.print_exc()
        _set(job_id, status="failed", error=f"{type(err).__name__}: {err}")


@app.post("/api/analyze")
async def analyze(mode: str = Form(...), site: str | None = Form(None), start: str | None = Form(None),
                  hours: float = Form(24.0), lat: float | None = Form(None), lon: float | None = Form(None),
                  clock_hours: float = Form(0.0), refine: bool = Form(False), source: str = Form("auto"),
                  site_b: str | None = Form(None), audio: UploadFile | None = File(None)):
    if mode not in ("who", "where", "tracks"):
        raise HTTPException(422, "mode must be 'who', 'where' or 'tracks'")
    if not 0 < hours <= MAX_HOURS:
        raise HTTPException(422, f"hours must be between 0 and {MAX_HOURS:g}")
    if not 0 <= clock_hours <= 14:
        raise HTTPException(422, "clock_hours must be between 0 and 14")
    job_id = secrets.token_urlsafe(8)
    cleanup = None
    if mode == "tracks":
        if not (site and site_b and start) or site == site_b:
            raise HTTPException(422, "tracking needs two different sites and a start time")
        try:
            a, b = catalog.get(site), catalog.get(site_b)
        except KeyError as err:
            raise HTTPException(404, str(err)) from err
        if not a.region or a.region != b.region:
            raise HTTPException(422, "the two sites must be in the same AIS region")
        with _lock:
            _jobs[job_id] = {"id": job_id, "status": "queued", "step": "Queued", "error": None}
        _pool.submit(_tracks, job_id, site, site_b, _parse_time(start), hours)
        return {"id": job_id}
    if audio is not None:
        workdir = cache_path("uploads", job_id, "x").parent
        dest = workdir / Path(audio.filename or "upload").name
        size = 0
        with dest.open("wb") as f:
            while chunk := await audio.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    shutil.rmtree(workdir, ignore_errors=True)
                    raise HTTPException(413, "upload is larger than 4 GB")
                f.write(chunk)
        t_start = _parse_time(start) if start else None
        cleanup = workdir
        if mode == "who" and not site and (lat is None or lon is None):
            raise HTTPException(422, "Mode A needs a site or a position")

        def clip_fn():
            return clip_from_audio(dest, t_start, workdir=workdir, label=audio.filename or "Upload")
        site_for_truth = site if mode == "who" else None
    else:
        if not (site and start):
            raise HTTPException(422, "give an audio file, or a site and a start time")
        try:
            catalog.get(site)
        except KeyError as err:
            raise HTTPException(404, f"unknown site {site!r}") from err
        t0 = _parse_time(start)

        def clip_fn():
            return public_clip(site, t0, t0 + hours * 3600.0, source=source)
        site_for_truth = site
    with _lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "step": "Queued", "error": None}
    _pool.submit(_analyse, job_id, mode, clip_fn, site_for_truth, lat, lon, clock_hours, refine, cleanup)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    with _lock:
        state = dict(_jobs.get(job_id) or {})
    if not state:
        if (_results_dir() / f"{job_id}.json").exists():
            return {"id": job_id, "status": "done", "step": "Done", "error": None}
        raise HTTPException(404, "no such job")
    return state


@app.get("/api/results/{job_id}")
def result(job_id: str):
    path = _results_dir() / f"{Path(job_id).name}.json"
    if not path.exists():
        raise HTTPException(404, "no such result")
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/results")
def recent_results(limit: int = 12):
    rows = []
    for path in sorted(_results_dir().glob("*.json"), key=lambda p: -p.stat().st_mtime)[:limit]:
        r = json.loads(path.read_text())
        rows.append({"id": r["id"], "created": r["created"], "mode": r["mode"], "label": r["clip"]["label"],
                     "hours": r["clip"]["hours"], "site": (r.get("site") or {}).get("name"),
                     "verdict": (r.get("locate") or {}).get("verdict"),
                     "inferred": len((r.get("tracks") or {}).get("inferred", [])) if r["mode"] == "tracks" else None})
    return {"results": rows}


@app.get("/api/soundmap")
def soundmap_day(region: str = "stellwagen", date: str = "2019-08-10"):
    from datetime import date as _date

    if region not in REGIONS:
        raise HTTPException(404, f"unknown region {region!r}")
    try:
        return JSONResponse(soundmap.build(region, _date.fromisoformat(date)))
    except ValueError as err:
        raise HTTPException(422, "date must be YYYY-MM-DD") from err
    except LookupError as err:
        raise HTTPException(404, str(err)) from err


# ------------------------------------------------------------------ optional front-end build

_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        target = _DIST / path
        if path and target.is_file() and _DIST in target.resolve().parents:
            return FileResponse(target)
        return FileResponse(_DIST / "index.html")
