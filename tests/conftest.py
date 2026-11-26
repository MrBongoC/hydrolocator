import numpy as np
import pytest

from hydrolocator.ais import build_tracks
from hydrolocator.geo import to_latlon

LAT0, LON0 = 42.0, -70.0


def make_tracks(vessels, t_start, t_end, step=60.0):
    """vessels: list of dicts with x0, y0 (m), vx, vy (m/s), t0, t1, length. Straight lines."""
    rows = {"mmsi": [], "t": [], "lat": [], "lon": []}
    static = {"mmsi": [], "name": [], "vessel_type": [], "length": [], "tclass": []}
    for k, v in enumerate(vessels):
        mmsi = 100000000 + k
        ts = np.arange(v.get("t0", t_start), v.get("t1", t_end) + 1, step)
        x = v["x0"] + v["vx"] * (ts - ts[0])
        y = v["y0"] + v["vy"] * (ts - ts[0])
        lat, lon = to_latlon(x, y, LAT0, LON0)
        rows["mmsi"] += [mmsi] * len(ts)
        rows["t"] += list(ts)
        rows["lat"] += list(lat)
        rows["lon"] += list(lon)
        static["mmsi"].append(mmsi)
        static["name"].append(v.get("name", f"SHIP {k}"))
        static["vessel_type"].append(70.0)
        static["length"].append(v.get("length", 150.0))
        static["tclass"].append("A")
    rows = {k: np.array(v) for k, v in rows.items()}
    static = {k: np.array(v, dtype=object if k in ("name", "tclass") else None) for k, v in static.items()}
    return build_tracks(rows, static, LAT0, LON0, t_start, t_end)


@pytest.fixture
def lane():
    """A day of traffic on an east-west lane 3 km north of the origin, mixed speeds."""
    rng = np.random.default_rng(7)
    t_start, t_end = 1_600_000_000.0, 1_600_000_000.0 + 86400.0
    vessels = []
    for k in range(14):
        speed = rng.uniform(4.0, 11.0) * rng.choice([-1, 1])
        depart = t_start + rng.uniform(-2, 22) * 3600
        vessels.append({"x0": -np.sign(speed) * 60_000.0, "y0": 3000.0 + rng.uniform(-800, 800),
                        "vx": speed, "vy": 0.0, "t0": depart, "t1": depart + 120_000.0 / abs(speed),
                        "length": float(rng.uniform(60, 300))})
    return make_tracks(vessels, t_start, t_end), t_start, t_end
