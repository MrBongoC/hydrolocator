"""Search regions for v1: places where public hydrophones and free AIS overlap.

AIS is only kept inside these boxes, and Mode B only ever searches inside them. That is
also the guardrail: the candidate set is public civilian water, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    id: str
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


REGIONS: dict[str, Region] = {r.id: r for r in [
    Region("salish_sea", "Salish Sea and Olympic Coast", 46.7, 50.3, -126.2, -122.0),
    Region("monterey_bay", "Monterey Bay", 35.7, 37.6, -123.3, -121.4),
    Region("channel_islands", "Santa Barbara Channel", 32.8, 34.9, -121.4, -118.2),
    Region("stellwagen", "Stellwagen Bank and Massachusetts Bay", 41.5, 43.2, -71.3, -69.2),
    Region("grays_reef", "Gray's Reef, Georgia", 30.7, 32.1, -81.7, -80.0),
    Region("florida_keys", "Florida Keys", 23.9, 25.3, -82.8, -80.3),
    Region("hawaii", "Main Hawaiian Islands", 19.2, 22.9, -160.4, -155.2),
]}


def region_for(lat: float, lon: float) -> Region | None:
    for region in REGIONS.values():
        if region.contains(lat, lon):
            return region
    return None
