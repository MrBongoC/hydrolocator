"""Score how well the passages AIS predicts for a position explain the passages heard.

Each predicted passage i is treated as a trial. Either the hydrophone missed it
(probability 1 - p_i), or it was heard and one of the observed events sits on it. Against
the alternative that the observed events are unrelated to this position, arriving at
their own average rate lambda_obs, the likelihood ratio for passage i is

    LR_i = (1 - p_i) + p_i * sum_j K(t_j - t_i) / lambda_obs

and the score is sum_i log LR_i. K is the timing kernel: a Gaussian for a sharp level or
Doppler peak, flat-topped for a log entry that only gives start and end. Loud ships that
went unheard cost log(1 - p_i). Several ships inside one long detection each get credit,
which matters because real detectors merge overlapping passages. Heard events that no
predicted ship explains cost nothing here; they are reported as dark detections.

A Hungarian assignment on the same terms says which ship explains which event.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .events import Observed
from .geo import KNOT

SQRT_2PI = 2.5066282746310002


@dataclass
class Params:
    # Detection probability: logistic in range, vessel length and speed. Defaults were fit
    # on Stellwagen site-days (level peaks from one-minute spectra against AIS passages).
    det_bias: float = 0.3
    det_range: float = -3.8        # per decade of range, relative to 5 km
    det_length: float = 2.8        # per decade of length, relative to 50 m
    det_speed: float = -0.2        # per decade of speed, relative to 10 kn
    p_max: float = 0.9             # nothing is certain: masking, gaps, duty cycles
    # Timing: sigma^2 = observed^2 + ais^2 + (kappa * r / v)^2. Slow, distant passages
    # have broad level peaks that are also skewed toward the louder stern aspect.
    sigma_ais: float = 45.0
    kappa: float = 0.5
    # Used when the vessel does not broadcast a length.
    length_class_a: float = 40.0
    length_class_b: float = 12.0
    # Marks from Doppler, when the clip has them.
    speed_sigma_kn: float = 2.0
    speed_prior_kn: float = 30.0   # width of the "any vessel" speed spread

    def to_dict(self) -> dict:
        return asdict(self)


def effective_length(length: np.ndarray, tclass: np.ndarray, params: Params) -> np.ndarray:
    fallback = np.where(np.asarray(tclass) == "B", params.length_class_b, params.length_class_a)
    length = np.asarray(length, dtype=np.float64)
    return np.where(np.isfinite(length) & (length > 2), length, fallback)


def detect_prob(r: np.ndarray, v: np.ndarray, length: np.ndarray, params: Params) -> np.ndarray:
    z = (params.det_bias
         + params.det_range * np.log10(np.maximum(r, 100.0) / 5000.0)
         + params.det_length * np.log10(np.maximum(length, 3.0) / 50.0)
         + params.det_speed * np.log10(np.maximum(v, 0.25) / (10 * KNOT)))
    return params.p_max / (1.0 + np.exp(-z))


def passage_sigma(r: np.ndarray, v: np.ndarray, params: Params) -> np.ndarray:
    return np.hypot(params.sigma_ais, params.kappa * r / np.maximum(v, 0.25))


@dataclass
class Predicted:
    """Predicted passages at one candidate with everything the score needs."""
    t: np.ndarray
    sigma: np.ndarray
    p: np.ndarray
    r: np.ndarray
    v: np.ndarray
    vid: np.ndarray

    def __len__(self) -> int:
        return len(self.t)


def kernel(obs: Observed, pred: Predicted, delta: np.ndarray, params: Params) -> np.ndarray:
    """Density (per second) of observed event j landing where it did if it came from
    passage i. `delta` is heard minus predicted time, shape (..., J, I)."""
    s = np.hypot(obs.sigma[:, None], pred.sigma[None, :])
    half = obs.half[:, None]
    d = np.maximum(np.abs(delta) - half, 0.0)
    dens = np.exp(-0.5 * (d / s) ** 2) / (2 * half + s * SQRT_2PI)
    has_speed = np.isfinite(obs.speed)
    if has_speed.any():
        sv = params.speed_sigma_kn * KNOT
        ratio = (np.exp(-0.5 * ((obs.speed[:, None] - pred.v[None, :]) / sv) ** 2) / (sv * SQRT_2PI)
                 * (params.speed_prior_kn * KNOT))
        dens = dens * np.where(has_speed[:, None], ratio, 1.0)
    return dens


def llr(obs: Observed, pred: Predicted, params: Params, offsets: np.ndarray | float = 0.0,
        *, chunk: int = 128) -> np.ndarray:
    """Score for each clock offset (recording clock minus UTC, seconds)."""
    offsets = np.atleast_1d(np.asarray(offsets, dtype=np.float64))
    out = np.zeros(len(offsets))
    if len(pred) == 0:
        return out
    diff = obs.t[:, None] - pred.t[None, :]                       # (J, I)
    for a in range(0, len(offsets), chunk):
        off = offsets[a:a + chunk]
        # Only passages that fall inside the clip (after the shift) are expected at all.
        tp = pred.t[None, :] + off[:, None]
        inside = (tp >= obs.t_start) & (tp < obs.t_end)           # (O, I)
        if len(obs):
            claim = kernel(obs, pred, diff[None] - off[:, None, None], params).sum(axis=1) / obs.rate
        else:
            claim = np.zeros(inside.shape)
        lr = (1.0 - pred.p)[None, :] + pred.p[None, :] * claim
        out[a:a + chunk] = (np.log(lr) * inside).sum(axis=1)
    return out


@dataclass
class Match:
    obs: int           # index into the observed events
    pred: int          # index into the predicted passages
    dt: float          # heard minus predicted, s (after the clock offset)
    gain: float        # log-likelihood this pairing adds over "unrelated event"


def assign(obs: Observed, pred: Predicted, params: Params, offset: float = 0.0
           ) -> tuple[list[Match], np.ndarray, np.ndarray]:
    """One-to-one matching. Returns (matches, unmatched observed idx, unmatched predicted idx)."""
    J, I = len(obs), len(pred)
    if J == 0 or I == 0:
        return [], np.arange(J), np.arange(I)
    delta = obs.t[:, None] - pred.t[None, :] - offset
    dens = pred.p[None, :] * kernel(obs, pred, delta, params)
    gain = np.log(np.maximum(dens, 1e-300) / obs.rate)
    # Pad with one "unrelated" column per observed event (gain 0) so anything can stay unmatched.
    cost = np.zeros((J, I + J))
    cost[:, :I] = -np.maximum(gain, 0.0)
    rows, cols = linear_sum_assignment(cost)
    matches = [Match(int(j), int(i), float(delta[j, i]), float(gain[j, i]))
               for j, i in zip(rows, cols) if i < I and gain[j, i] > 0]
    used_o = {m.obs for m in matches}
    used_p = {m.pred for m in matches}
    return (matches, np.array([j for j in range(J) if j not in used_o], dtype=int),
            np.array([i for i in range(I) if i not in used_p], dtype=int))
