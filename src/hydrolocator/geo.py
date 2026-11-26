"""Flat-earth helpers. Everything downstream works in metres on a local tangent plane,
which is accurate to well under 0.1 % across the ~100 km regions we search."""

from __future__ import annotations

import numpy as np

EARTH_R = 6_371_008.8
KNOT = 0.514444  # m/s


def to_xy(lat, lon, lat0: float, lon0: float):
    """Degrees -> metres east/north of (lat0, lon0)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    x = np.radians(lon - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def to_latlon(x, y, lat0: float, lon0: float):
    lat = lat0 + np.degrees(np.asarray(y) / EARTH_R)
    lon = lon0 + np.degrees(np.asarray(x) / (EARTH_R * np.cos(np.radians(lat0))))
    return lat, lon


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def bbox_around(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) enclosing a circle of radius_m."""
    dlat = np.degrees(radius_m / EARTH_R)
    dlon = dlat / np.cos(np.radians(lat))
    return lat - dlat, lat + dlat, lon - dlon, lon + dlon
