"""Resolve "site + time window" on a public archive into a Clip."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from . import catalog
from .ingest import read_audio, spectra
from .pipeline import Clip, clip_from_spectra


def _sanctsound_dep(site: catalog.Site, t0: float, t1: float):
    from .sources import sanctsound as ss

    code = site.id.split(":")[1]
    for dep in ss.fetch_deployments():
        if dep.site == code and dep.start.timestamp() <= t0 and dep.end.timestamp() >= t1:
            return dep
    raise LookupError(f"{site.id} was not recording for the whole window")


def public_clip(site_id: str, t0: float, t1: float, *, source: str = "auto") -> Clip:
    """A clip from a public recording. `source` is "auto", "spectra" (archive one-minute
    spectra, where published) or "audio" (streamed and decimated from the archive)."""
    site = catalog.get(site_id)
    label = f"{site.name}, {datetime.fromtimestamp(t0, UTC):%Y-%m-%d %H:%M} UTC"
    if site.id.startswith("sanctsound:"):
        from .sources import sanctsound as ss

        dep = _sanctsound_dep(site, t0, t1)
        if source in ("auto", "spectra"):
            parts = []
            day = datetime.fromtimestamp(t0, UTC).date()
            last = datetime.fromtimestamp(t1 - 1, UTC).date()
            while day <= last:
                try:
                    sp = ss.hmd_spectra(dep, day)
                except Exception:
                    sp = None
                if sp is None:
                    parts = None
                    break
                parts.append(sp)
                day += timedelta(days=1)
            if parts:
                t = np.concatenate([p[0] for p in parts])
                psd = np.concatenate([p[2] for p in parts])
                keep = (t >= t0) & (t < t1)
                f = parts[0][1]
                low = f <= 1000.0
                return clip_from_spectra(t[keep], f[low], psd[keep][:, low], 60.0, label)
            if source == "spectra":
                raise LookupError(f"{site.id} has no archive spectra for this window")
        sp = ss.audio_spectra(dep, t0, t1, dt=10.0)
        if sp is None:
            raise LookupError(f"{site.id} has no audio for this window")
        return Clip(sp.t, sp.band(20.0, 900.0), sp.dt, label, sp.freqs, sp.psd_db)
    if site.id == "mbari:mars":
        from .sources import mbari

        frames = []
        day = datetime.fromtimestamp(t0, UTC).date()
        last = datetime.fromtimestamp(t1 - 1, UTC).date()
        while day <= last:
            day0 = datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()
            a, b = max(t0, day0), min(t1, day0 + 86400.0)
            x, sr = read_audio(mbari.lf_copy(day), a - day0, b - a)
            frames.append(spectra(x, sr, a, dt=10.0))
            day += timedelta(days=1)
        t = np.concatenate([s.t for s in frames])
        psd = np.concatenate([s.psd_db for s in frames])
        clip = Clip(t, frames[0].band(20.0, 900.0) if len(frames) == 1 else
                    np.concatenate([s.band(20.0, 900.0) for s in frames]), 10.0, label, frames[0].freqs, psd)
        return clip
    raise LookupError(f"no reader for {site.id}")
