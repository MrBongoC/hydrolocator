"""Draw one moment of a sound-map day with matplotlib.

    uv run python examples/plot_soundmap.py --time 01:57 --out frame.png

It reads the day through the same function the API serves (`soundmap.build`), decodes the
vessel positions, and evaluates the level model on a grid. That is the arithmetic behind
the clip in the README: the glow is predicted from AIS, the rings are what the hydrophones
recorded. The default day ships with the package, so nothing is downloaded.
"""

from __future__ import annotations

import argparse
import base64
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from hydrolocator import soundmap  # noqa: E402

GROUND, GLOW, INK = "#14110f", "#ffb066", "#f3ede6"


def decode(day: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitude, longitude and source level for every vessel at every two-minute step."""
    shape = (len(day["vessels"]), day["steps"])
    west, south, east, north = day["bbox"]

    def unpack(key: str, lo: float, hi: float) -> np.ndarray:
        q = np.frombuffer(base64.b64decode(day[key]), "<i2").reshape(shape).astype(float)
        return np.where(q == -32768, np.nan, lo + (q + 32767.0) / 65534.0 * (hi - lo))       # -32768: out of view

    src = np.frombuffer(base64.b64decode(day["src"]), "i1").reshape(shape).astype(float)
    return unpack("lat", south, north), unpack("lon", west, east), src


def predicted_excess(day: dict, lat: np.ndarray, lon: np.ndarray, src: np.ndarray, cells: int = 360):
    """Predicted level above the noise floor, in dB, on a grid over the day's bounding box."""
    model = day["model"]
    west, south, east, north = day["bbox"]
    m_per_lat = 111_320.0
    m_per_lon = m_per_lat * np.cos(np.radians((south + north) / 2))
    gx, gy = np.meshgrid(np.linspace(west, east, cells), np.linspace(south, north, int(cells * 0.6)))
    floor_loss = model["spreading"] * np.log10(model["floor_km"] * 1000.0) + model["absorption"] * model["floor_km"]
    power = np.zeros_like(gx)
    for v in np.nonzero(np.isfinite(lat))[0]:
        r = np.hypot(np.hypot((gx - lon[v]) * m_per_lon, (gy - lat[v]) * m_per_lat), model["min_range"])
        loss = model["spreading"] * np.log10(r) + model["absorption"] * r / 1000.0
        power += 10.0 ** ((src[v] - loss + floor_loss) / 10.0)
    return 10.0 * np.log10(1.0 + power)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--date", default="2019-08-10", help="UTC day, YYYY-MM-DD")
    args.add_argument("--time", default="01:57", help="UTC time of day, HH:MM")
    args.add_argument("--out", default="sound-map.png")
    opts = args.parse_args()

    day = soundmap.build("stellwagen", date.fromisoformat(opts.date))
    hours, minutes = (int(part) for part in opts.time.split(":"))
    seconds = hours * 3600 + minutes * 60
    k = min(int(seconds // day["step_s"]), day["steps"] - 1)
    lat, lon, src = decode(day)
    west, south, east, north = day["bbox"]

    fig, ax = plt.subplots(figsize=(12, 7.2), facecolor=GROUND)
    ax.set_facecolor(GROUND)
    excess = predicted_excess(day, lat[:, k], lon[:, k], src[:, k])
    ax.imshow(excess, extent=(west, east, south, north), origin="lower", aspect="auto",
              cmap=LinearSegmentedColormap.from_list("glow", [GROUND, GLOW]), vmin=0.0, vmax=30.0)

    here = np.nonzero(np.isfinite(lat[:, k]))[0]
    ax.scatter(lon[here, k], lat[here, k], s=9, color=INK, linewidths=0)
    for v in here[np.argsort(-src[here, k])][:5]:                                  # name the five loudest
        if day["vessels"][v]["name"]:
            ax.annotate(day["vessels"][v]["name"], (lon[v, k], lat[v, k]), xytext=(7, -3), textcoords="offset points",
                        color=INK, fontsize=8)

    for post in day["posts"]:
        heard = post["heard_db"][min(int(seconds // post["curve_dt_s"]), len(post["heard_db"]) - 1)]
        label = post["name"].split()[-1] + (f"  heard {heard:+.0f} dB" if heard is not None else "  no data")
        ax.scatter([post["lon"]], [post["lat"]], s=110, facecolors="none", edgecolors=INK, linewidths=1.4)
        ax.annotate(label, (post["lon"], post["lat"]), xytext=(0, -18), textcoords="offset points", ha="center",
                    color=INK, fontsize=9, style="italic")

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1.0 / np.cos(np.radians((south + north) / 2)))
    ax.set_title(f"{day['region']['name']}, {opts.date} {opts.time} UTC: {len(here)} ships on AIS, predicted noise above the floor",
                 color=INK, fontsize=11, loc="left")
    ax.tick_params(colors="#8d857c", labelsize=8)
    for side in ax.spines.values():
        side.set_color("#3a342e")
    fig.savefig(opts.out, dpi=150, facecolor=GROUND, bbox_inches="tight")
    print(f"wrote {opts.out}: {len(here)} ships in view at {opts.time} UTC, loudest predicted excess {excess.max():.0f} dB")


if __name__ == "__main__":
    main()
