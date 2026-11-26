"""The two product modes, built from the pieces in this package.

Mode A, `who_passed`: clip + known position -> the AIS vessels that explain what was heard,
plus heard events nothing explains ("dark" detections).

Mode B, `locate`: clip + candidate positions -> which position's AIS traffic predicts the
level curve that was actually recorded.

Both work from a `Clip`, which is just a band-level time series with a start time. A clip
can come from an upload, a public archive recording, or archive spectra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from . import ais, cpa
from .detect import detect_passages
from .events import Observed
from .geo import KNOT, to_latlon
from .regions import region_for
from .score import Params, Predicted, assign, detect_prob, effective_length, passage_sigma
from .soundscape import LevelParams, excess_from_ranges, highpass, observed_excess, ranges, vessel_positions
from .suspects import for_event, silent_courses
from .voiceprint import line_spectra, load_library, signature, suggest

MATCH_DT = 60.0            # the matcher works on a one-minute grid
R_PASSAGE = 20_000.0       # passages beyond this are not listed
R_TRACKS = 45_000.0        # vessels beyond this never matter to the level curve
MIN_LAG = 45               # minutes; correlations at larger shifts form the null
DARK_MIN_DB = 6.0          # an unmatched peak must stand this far out to be listed as a dark detection


# --------------------------------------------------------------------------- clips

@dataclass
class Clip:
    """A recording reduced to what the matcher needs."""
    t: np.ndarray                       # frame start times, POSIX s, uniform
    level: np.ndarray                   # 20-1,000 Hz band level per frame, dB (uncalibrated is fine)
    dt: float
    label: str = ""
    freqs: np.ndarray | None = None     # optional spectra for display
    psd_db: np.ndarray | None = None
    lf_path: Path | None = None         # 2 kHz copy, when the clip came from audio

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1] + self.dt)

    @property
    def hours(self) -> float:
        return (self.t_end - self.t_start) / 3600.0

    def on_grid(self, dt: float = MATCH_DT) -> tuple[np.ndarray, np.ndarray]:
        """(grid times at bin centres, mean level per bin) on a coarser uniform grid."""
        n = int((self.t_end - self.t_start) // dt)
        k = np.floor((self.t - self.t_start) / dt).astype(int)
        ok = (k >= 0) & (k < n) & np.isfinite(self.level)
        power = np.bincount(k[ok], weights=10.0 ** (self.level[ok] / 10.0), minlength=n)
        count = np.bincount(k[ok], minlength=n)
        with np.errstate(divide="ignore", invalid="ignore"):
            level = np.where(count > 0, 10.0 * np.log10(power / np.maximum(count, 1)), np.nan)
        return self.t_start + dt * (np.arange(n) + 0.5), level

    def events(self) -> Observed:
        t, level = self.on_grid(min(MATCH_DT, max(self.dt, 30.0)))
        return detect_passages(t - (t[1] - t[0]) / 2, level)


def clip_from_spectra(t: np.ndarray, freqs: np.ndarray, psd_db: np.ndarray, dt: float,
                      label: str = "", band: tuple[float, float] = (20.0, 1000.0)) -> Clip:
    from .detect import band_level

    return Clip(np.asarray(t, dtype=np.float64), band_level(psd_db, freqs, *band), dt, label,
                np.asarray(freqs), psd_db)


def clip_from_audio(src: str | Path, t_start: float | None = None, *, workdir: Path | None = None,
                    label: str = "") -> Clip:
    """Decode an audio file (anything ffmpeg reads). The start time comes from `t_start`
    or, failing that, the filename."""
    from .ingest import lowrate_copy, read_audio, spectra, start_time_from_name
    from .paths import cache_path

    src_name = Path(str(src)).name
    if t_start is None:
        t_start = start_time_from_name(src_name)
    if t_start is None:
        raise ValueError("no start time given and none found in the filename")
    lf = (workdir or cache_path("uploads", "x").parent) / (Path(src_name).stem + ".2k.flac")
    x, sr = read_audio(lowrate_copy(src, lf))
    sp = spectra(x, sr, t_start, dt=10.0)
    return Clip(sp.t, sp.band(20.0, 900.0), sp.dt, label or src_name, sp.freqs, sp.psd_db, lf)


# --------------------------------------------------------------------------- shared

@dataclass
class Vessel:
    mmsi: int
    name: str
    vessel_type: int
    length: float | None
    tclass: str
    imo: str = ""
    callsign: str = ""
    width: float | None = None
    draft: float | None = None


def _num(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) and x > 0 else None


def _vessel(tracks: ais.Tracks, vid: int) -> Vessel:
    return Vessel(int(tracks.mmsi[vid]), str(tracks.name[vid] or ""), int(tracks.vessel_type[vid]),
                  _num(tracks.length[vid]), str(tracks.tclass[vid] or ""),
                  str(tracks.imo[vid] or "") if tracks.imo is not None else "",
                  str(tracks.callsign[vid] or "") if tracks.callsign is not None else "",
                  _num(tracks.width[vid]) if tracks.width is not None else None,
                  _num(tracks.draft[vid]) if tracks.draft is not None else None)


VESSEL_TYPES = {
    30: "Fishing", 31: "Tug", 32: "Tug", 35: "Military", 36: "Sailing", 37: "Pleasure craft",
    50: "Pilot", 51: "Search and rescue", 52: "Tug", 53: "Port tender", 55: "Law enforcement",
}


def vessel_type_name(code: int) -> str:
    if code in VESSEL_TYPES:
        return VESSEL_TYPES[code]
    if 60 <= code <= 69:
        return "Passenger"
    if 70 <= code <= 79:
        return "Cargo"
    if 80 <= code <= 89:
        return "Tanker"
    if 40 <= code <= 49:
        return "High-speed craft"
    if 1001 <= code <= 1002:
        return "Fishing"
    if 1012 <= code <= 1015:
        return "Passenger"
    if 1003 <= code <= 1004 or code == 1016:
        return "Cargo"
    if code == 1017 or code == 1024:
        return "Tanker"
    if code in (1023, 1025):
        return "Tug"
    if code == 1019:
        return "Pleasure craft"
    return "Other" if code > 0 else "Unknown"


def predicted_curve(tracks: ais.Tracks, tgrid: np.ndarray, cx: float, cy: float,
                    level: LevelParams) -> np.ndarray:
    vid, r, spd = ranges(vessel_positions(tracks, tgrid), cx, cy)
    return highpass(excess_from_ranges(r, spd, tracks.length[vid], tracks.tclass[vid], level), MATCH_DT)


def _corr_rows(obs: np.ndarray, windows: np.ndarray) -> np.ndarray:
    """Pearson correlation of obs (n,) with each row of windows (K, n), ignoring NaNs in obs."""
    good = np.isfinite(obs)
    if good.sum() < 20:
        return np.zeros(len(windows))
    a = obs[good] - obs[good].mean()
    b = windows[:, good]
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum() * (b * b).sum(axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 1e-9, (b @ a) / den, 0.0)


def circular_z(obs: np.ndarray, core: np.ndarray) -> tuple[float, float]:
    """Correlation of an observed excess curve with a predicted one of the same length,
    and its z-score against circular shifts of at least MIN_LAG samples. Shifted copies
    keep the predicted curve's own peakiness, so the null needs no model of it. The
    confidence thresholds below were calibrated on this statistic."""
    n = len(obs)
    good = np.isfinite(obs)
    lag = min(MIN_LAG, max(5, n // 6))
    if good.sum() < 40 or n < 2 * lag + 20:
        return 0.0, 0.0
    a = np.where(good, obs - obs[good].mean(), 0.0)
    b = core - core.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den < 1e-9:
        return 0.0, 0.0
    rho = np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n=n) / den
    null = rho[lag:n - lag]
    return float(rho[0]), float((rho[0] - null.mean()) / max(float(null.std()), 1e-6))


def sweep_offsets(obs: np.ndarray, pred_ext: np.ndarray, pad: int, k_unc: int) -> tuple[int, np.ndarray]:
    """Correlate obs against pred_ext (the clip plus `pad` samples each side) for every
    clock offset within +/- k_unc samples. Returns (best offset, rho per offset).

    A recording clock running k samples ahead means the clip really started k samples
    earlier, so clock offset = -shift along the predicted curve.
    """
    n = len(obs)
    windows = np.lib.stride_tricks.sliding_window_view(pred_ext, n)[pad - k_unc:pad + k_unc + 1]
    rho = _corr_rows(obs, windows)[::-1]
    return int(np.argmax(rho)) - k_unc, rho


def search_penalty(n_offsets: int) -> float:
    """Searching many clock offsets lifts the best chance correlation. Neighbouring
    offsets are correlated over roughly ten minutes, so count one trial per ten."""
    return float(np.sqrt(2.0 * np.log(max(1.0, n_offsets / 10.0))))


# --------------------------------------------------------------------------- Mode A

@dataclass
class PassageRow:
    """One line of the Mode A table."""
    vessel: Vessel | None            # None for a dark detection
    cpa_time: float | None           # from AIS
    cpa_range_m: float | None
    speed_kn: float | None
    detect_prob: float | None
    heard_time: float | None         # from the recording
    heard_prominence_db: float | None
    dt_s: float | None               # heard minus AIS CPA
    status: str                      # "matched" | "also_present" | "not_heard" | "dark"
    suspects: list = field(default_factory=list)   # for a dark row: vessels whose AIS was silent on a passing course
    hints: list = field(default_factory=list)      # for a dark row: library vessels it sounds most like


@dataclass
class SiteReport:
    lat: float
    lon: float
    t_start: float
    t_end: float
    rows: list[PassageRow]
    rho: float
    z: float
    offset_s: float
    n_events: int
    n_predicted: int
    ais_days_missing: list[str]
    tgrid: np.ndarray = field(repr=False, default=None)
    observed_excess: np.ndarray = field(repr=False, default=None)
    predicted_excess: np.ndarray = field(repr=False, default=None)
    tracks: ais.Tracks | None = field(repr=False, default=None)

    @property
    def matched(self) -> list[PassageRow]:
        return [r for r in self.rows if r.status == "matched"]

    @property
    def dark(self) -> list[PassageRow]:
        return [r for r in self.rows if r.status == "dark"]


def who_passed(clip: Clip, lat: float, lon: float, *, params: Params | None = None,
               level: LevelParams | None = None, offset_s: float = 0.0,
               list_range_m: float = R_PASSAGE) -> SiteReport:
    """Mode A: explain a clip recorded at a known position."""
    params = params or Params()
    level = level or LevelParams()
    tgrid, lvl = clip.on_grid()
    obs_curve = observed_excess(lvl, MATCH_DT)
    t0, t1 = clip.t_start - offset_s, clip.t_end - offset_s
    _, missing = ais.coverage(t0, t1)
    tracks = ais.load_tracks(lat, lon, t0, t1, R_TRACKS)
    pred_curve = predicted_curve(tracks, tgrid - offset_s, 0.0, 0.0, level)
    rho, z = circular_z(obs_curve, pred_curve)

    events = clip.events()
    pas = cpa.predict(tracks, 0.0, 0.0, list_range_m)
    length = effective_length(tracks.length[pas.vid], tracks.tclass[pas.vid], params)
    pred = Predicted(pas.t, passage_sigma(pas.r, pas.v, params),
                     detect_prob(pas.r, pas.v, length, params), pas.r, pas.v, pas.vid)
    matches, dark_idx, unheard_idx = assign(events, pred, params, offset_s)
    prom = events.meta.get("prominence", np.full(len(events), np.nan))

    rows: list[PassageRow] = []
    for m in matches:
        rows.append(PassageRow(_vessel(tracks, int(pred.vid[m.pred])), float(pred.t[m.pred]),
                               float(pred.r[m.pred]), float(pred.v[m.pred] / KNOT), float(pred.p[m.pred]),
                               float(events.t[m.obs] - offset_s), float(prom[m.obs]), m.dt, "matched"))
    # A second ship inside an already-claimed event is "also present", not "not heard".
    claimed = np.array([events.t[m.obs] for m in matches]) - offset_s
    claimed_sigma = np.array([events.sigma[m.obs] for m in matches])
    for i in unheard_idx:
        near = len(claimed) and bool(np.any(np.abs(claimed - pred.t[i]) < np.hypot(claimed_sigma, pred.sigma[i])))
        status = "also_present" if near else "not_heard"
        if pred.p[i] < 0.25:
            continue                                  # quiet or distant: nobody expected to hear it
        rows.append(PassageRow(_vessel(tracks, int(pred.vid[i])), float(pred.t[i]), float(pred.r[i]),
                               float(pred.v[i] / KNOT), float(pred.p[i]), None, None, None, status))
    loud = [j for j in dark_idx if prom[j] >= DARK_MIN_DB]   # a faint bump with no vessel is not a finding
    if loud:
        gaps = silent_courses(lat, lon, t0, t1)
        region = region_for(lat, lon)
        library = load_library()
        lines = (line_spectra(clip.freqs, clip.psd_db)
                 if library is not None and region is not None and clip.psd_db is not None else None)
        for j in loud:
            heard = float(events.t[j] - offset_s)
            hints = (suggest(library, signature(lines, clip.t + clip.dt / 2, float(events.t[j])), waters=region.id)
                     if lines is not None else [])
            rows.append(PassageRow(None, None, None, None, None, heard, float(prom[j]), None, "dark",
                                   suspects=for_event(gaps, heard), hints=hints))
    rows.sort(key=lambda r: r.cpa_time if r.cpa_time is not None else r.heard_time)
    n_heard = sum(r.heard_time is not None for r in rows)
    return SiteReport(lat, lon, clip.t_start, clip.t_end, rows, rho, z, offset_s, n_heard, len(pred),
                      [d.isoformat() for d in missing], tgrid, obs_curve, pred_curve, tracks)


# --------------------------------------------------------------------------- Mode B

@dataclass
class Candidate:
    id: str
    lat: float
    lon: float
    name: str = ""
    network: str = ""


@dataclass
class CandidateScore:
    candidate: Candidate
    rho: float
    z: float
    offset_s: float
    posterior: float = 0.0
    n_vessels: int = 0
    covered: bool = True             # False when we hold no AIS for this place and time


@dataclass
class LocateResult:
    ranking: list[CandidateScore]
    verdict: str                     # located | probable | ambiguous | not_enough_ships | no_ais_coverage
    none_posterior: float
    n_events: int
    offsets_searched: int

    @property
    def best(self) -> CandidateScore | None:
        return self.ranking[0] if self.ranking else None


# Confidence: softmax(BETA * z) over the candidates plus "none of these" at Z_NONE. On
# held-out site-days the top pick was right 12 times out of 12 at z >= 4, about 5 in 6 at
# z >= 3.5, and no better than a coin flip below that; these two constants encode that.
BETA = 2.5
Z_NONE = 3.6
MIN_EVENTS = 3


def locate(clip: Clip, candidates: list[Candidate], *, level: LevelParams | None = None,
           clock_uncertainty_s: float = 0.0) -> LocateResult:
    """Mode B over a list of candidate positions."""
    level = level or LevelParams()
    tgrid, lvl = clip.on_grid()
    obs_curve = observed_excess(lvl, MATCH_DT)
    n = len(tgrid)
    k_unc = int(round(clock_uncertainty_s / MATCH_DT))
    ext = clip.t_start + MATCH_DT * (np.arange(-k_unc, n + k_unc) + 0.5)
    have, _ = ais.coverage(clip.t_start, clip.t_end)
    n_events = len(clip.events())
    penalty = search_penalty(2 * k_unc + 1) if k_unc else 0.0

    scores = []
    for cand in candidates:
        if not have:
            scores.append(CandidateScore(cand, 0.0, 0.0, 0.0, covered=False))
            continue
        tracks = ais.load_tracks(cand.lat, cand.lon, float(ext[0]), float(ext[-1]), R_TRACKS, pad_s=600.0)
        pred_ext = predicted_curve(tracks, ext, 0.0, 0.0, level)
        off = sweep_offsets(obs_curve, pred_ext, k_unc, k_unc)[0] if k_unc else 0
        rho, z = circular_z(obs_curve, pred_ext[k_unc - off:k_unc - off + n])
        scores.append(CandidateScore(cand, rho, z - penalty, off * MATCH_DT, n_vessels=tracks.n_vessels))

    zs = np.array([s.z if s.covered else -np.inf for s in scores] + [Z_NONE])
    post = np.exp(BETA * zs - logsumexp(BETA * zs))
    for s, p in zip(scores, post):
        s.posterior = float(p)
    ranking = sorted(scores, key=lambda s: -s.z)
    if not have:
        verdict = "no_ais_coverage"
    elif n_events < MIN_EVENTS:
        verdict = "not_enough_ships"
    elif ranking and ranking[0].posterior >= 0.7:
        verdict = "located"
    elif ranking and ranking[0].posterior >= 0.35:
        verdict = "probable"
    else:
        verdict = "ambiguous"
    return LocateResult(ranking, verdict, float(post[-1]), n_events, 2 * k_unc + 1)


@dataclass
class GridResult:
    lat: np.ndarray
    lon: np.ndarray
    z: np.ndarray
    rho: np.ndarray
    best: tuple[float, float]
    step_m: float


def grid_search(clip: Clip, lat0: float, lon0: float, *, half_m: float = 8000.0, step_m: float = 500.0,
                level: LevelParams | None = None, offset_s: float = 0.0, chunk: int = 48) -> GridResult:
    """Mode B on a local grid: score every cell within half_m of (lat0, lon0)."""
    level = level or LevelParams()
    tgrid, lvl = clip.on_grid()
    obs_curve = observed_excess(lvl, MATCH_DT)
    axis = np.arange(-half_m, half_m + 1, step_m)
    gx, gy = (a.ravel() for a in np.meshgrid(axis, axis))
    tracks = ais.load_tracks(lat0, lon0, clip.t_start - offset_s, clip.t_end - offset_s, R_TRACKS + half_m)
    vid, x, y, spd = vessel_positions(tracks, tgrid - offset_s)
    # Vessels that never come near the grid cannot shape its curves.
    with np.errstate(invalid="ignore"):
        dmin = np.nanmin(np.where(np.isfinite(x), np.hypot(x, y), np.inf), axis=1)
    keep = dmin < R_TRACKS + half_m
    vid, x, y, spd = vid[keep], x[keep], y[keep], spd[keep]
    length, tclass = tracks.length[vid], tracks.tclass[vid]
    rho = np.zeros(len(gx))
    z = np.zeros(len(gx))
    for k in range(len(gx)):
        r = np.hypot(x - gx[k], y - gy[k]).astype(np.float32)
        core = highpass(excess_from_ranges(r, spd, length, tclass, level), MATCH_DT)
        rho[k], z[k] = circular_z(obs_curve, core)
    lat, lon = to_latlon(gx, gy, lat0, lon0)
    b = int(np.argmax(z))
    return GridResult(lat, lon, z, rho, (float(lat[b]), float(lon[b])), step_m)


def iso(t: float | None) -> str | None:
    return None if t is None else datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
