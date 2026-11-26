"""The public hydrophone catalog: every position Mode B is allowed to name.

Entries come from published deployment metadata only. The catalog ships with the package
(`data/catalog.json`) and can be rebuilt from source with `build()`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources

from .regions import region_for

SANCTUARIES = {
    "sb": "Stellwagen Bank", "ci": "Channel Islands", "mb": "Monterey Bay", "oc": "Olympic Coast",
    "fk": "Florida Keys", "gr": "Gray's Reef", "hi": "Hawaiian Islands", "pm": "Papahanaumokuakea",
}


@dataclass(frozen=True)
class Deployment:
    start: float
    end: float | None       # None: still recording
    lat: float
    lon: float
    label: str = ""


@dataclass(frozen=True)
class Site:
    id: str                 # "sanctsound:sb01"
    name: str
    network: str
    lat: float
    lon: float
    depth_m: float | None
    deployments: tuple[Deployment, ...]
    source: str

    @property
    def region(self) -> str | None:
        region = region_for(self.lat, self.lon)
        return region.id if region else None

    def active(self, t0: float, t1: float) -> Deployment | None:
        """The deployment that covers [t0, t1], if any."""
        for d in self.deployments:
            if d.start <= t0 and (d.end is None or d.end >= t1):
                return d
        return None


def load() -> list[Site]:
    raw = json.loads(resources.files("hydrolocator").joinpath("data/catalog.json").read_text())
    return [Site(s["id"], s["name"], s["network"], s["lat"], s["lon"], s.get("depth_m"),
                 tuple(Deployment(**d) for d in s["deployments"]), s["source"]) for s in raw]


def get(site_id: str) -> Site:
    for site in load():
        if site.id == site_id or site.id.split(":")[-1] == site_id:
            return site
    raise KeyError(f"unknown site {site_id!r}")


def active_sites(t0: float, t1: float, *, with_ais: bool = True) -> list[tuple[Site, Deployment]]:
    """Sites recording through [t0, t1]. with_ais keeps only those in a region we index."""
    out = []
    for site in load():
        dep = site.active(t0, t1)
        if dep and (site.region or not with_ais):
            out.append((site, dep))
    return out


def build() -> list[dict]:
    """Rebuild the catalog from source metadata (network access needed)."""
    from .sources import sanctsound as ss

    sites: dict[str, dict] = {}
    for d in ss.fetch_deployments():
        entry = sites.setdefault(d.site, {
            "id": f"sanctsound:{d.site}",
            "name": f"{SANCTUARIES[d.site[:2]]} {d.site.upper()}",
            "network": "NOAA SanctSound", "lat": d.lat, "lon": d.lon, "depth_m": None,
            "deployments": [],
            "source": "https://www.ncei.noaa.gov/products/passive-acoustic-data",
        })
        entry["deployments"].append({"start": d.start.timestamp(), "end": d.end.timestamp(),
                                     "lat": d.lat, "lon": d.lon, "label": d.deployment})
    out = list(sites.values())
    # MBARI MARS cabled observatory: position and depth from docs.mbari.org/pacific-sound.
    mars_start = datetime(2015, 7, 28, tzinfo=UTC).timestamp()
    out.append({
        "id": "mbari:mars", "name": "MARS cabled observatory", "network": "MBARI Pacific Sound",
        "lat": 36.7128, "lon": -122.186, "depth_m": 891.0,
        "deployments": [{"start": mars_start, "end": None, "lat": 36.7128, "lon": -122.186, "label": "MARS"}],
        "source": "https://docs.mbari.org/pacific-sound/",
    })
    return out


if __name__ == "__main__":
    from pathlib import Path

    path = Path(__file__).parent / "data" / "catalog.json"
    path.write_text(json.dumps(build(), indent=1))
    print(f"wrote {path}")
