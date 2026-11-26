import shutil

import numpy as np
import pytest
from conftest import make_tracks

from hydrolocator import cpa
from hydrolocator.characterise import fit_fringes
from hydrolocator.detect import detect_passages
from hydrolocator.events import Observed
from hydrolocator.ingest import lowrate_copy, read_audio, spectra, start_time_from_name
from hydrolocator.pipeline import circular_z, predicted_curve, search_penalty, sweep_offsets
from hydrolocator.score import Params, Predicted, assign, llr
from hydrolocator.soundscape import LevelParams, observed_excess

T0 = 1_600_000_000.0


def test_cpa_of_a_straight_pass():
    # 5 m/s due east along y = 2000 m, crossing x = 0 exactly one hour in.
    tracks = make_tracks([{"x0": -18_000.0, "y0": 2000.0, "vx": 5.0, "vy": 0.0}], T0, T0 + 7200)
    p = cpa.predict(tracks, 0.0, 0.0, 10_000.0)
    assert len(p) == 1
    assert p.t[0] == pytest.approx(T0 + 3600, abs=1.0)
    assert p.r[0] == pytest.approx(2000.0, abs=1.0)
    assert p.v[0] == pytest.approx(5.0, abs=0.01)


def test_cpa_per_candidate_and_range_limit():
    tracks = make_tracks([{"x0": -18_000.0, "y0": 2000.0, "vx": 5.0, "vy": 0.0}], T0, T0 + 7200)
    p = cpa.predict(tracks, [0.0, 5000.0, 0.0], [0.0, 2000.0, 30_000.0], 10_000.0)
    assert sorted(p.cand.tolist()) == [0, 1]                    # the third candidate is out of range
    second = p.for_candidate(1)
    assert second.t[0] == pytest.approx(T0 + 3600 + 1000, abs=1.0)
    assert second.r[0] == pytest.approx(0.0, abs=1.0)


def test_moored_vessel_is_not_a_passage():
    tracks = make_tracks([{"x0": 500.0, "y0": 500.0, "vx": 0.0, "vy": 0.0}], T0, T0 + 7200)
    assert len(cpa.predict(tracks, 0.0, 0.0, 10_000.0)) == 0


def test_track_gap_is_not_bridged():
    a = {"x0": -18_000.0, "y0": 2000.0, "vx": 5.0, "vy": 0.0, "t0": T0, "t1": T0 + 3000}
    tracks = make_tracks([a], T0, T0 + 7200)
    # The track stops ten minutes before it would have reached CPA: no passage is invented.
    assert len(cpa.predict(tracks, 0.0, 0.0, 10_000.0)) == 0


def test_start_time_from_filenames():
    assert start_time_from_name("ICLISTENHF1251_20180101T000000.000Z.wav") == 1514764800.0
    assert start_time_from_name("MARS-20210601T000000Z-2kHz.wav") == 1622505600.0
    assert start_time_from_name("SanctSound_SB01_08_67678214_200224055255.flac") == 1582523575.0
    assert start_time_from_name("holiday.wav") is None


def test_detector_finds_bumps():
    t = T0 + 60.0 * np.arange(720)
    level = 100.0 + np.random.default_rng(1).normal(0, 0.4, len(t))
    for centre, height, tau in ((120, 12.0, 6.0), (400, 8.0, 10.0), (600, 15.0, 4.0)):
        level += height / (1.0 + ((np.arange(len(t)) - centre) / tau) ** 2)
    events = detect_passages(t, level)
    found = sorted(((events.t - T0) / 60.0).round().tolist())
    for centre in (120, 400, 600):
        assert any(abs(f - centre) <= 2 for f in found)
    assert len(events) <= 5


def test_assignment_pairs_events_with_passages():
    obs = Observed(T0, T0 + 86400, np.array([T0 + 3600 + 40, T0 + 40_000]), 90.0)
    pred = Predicted(t=np.array([T0 + 3600.0, T0 + 20_000.0]), sigma=np.array([60.0, 60.0]),
                     p=np.array([0.8, 0.8]), r=np.array([2000.0, 2000.0]), v=np.array([5.0, 5.0]),
                     vid=np.array([0, 1]))
    matches, dark, unheard = assign(obs, pred, Params())
    assert [(m.obs, m.pred) for m in matches] == [(0, 0)]
    assert dark.tolist() == [1] and unheard.tolist() == [1]
    assert matches[0].dt == pytest.approx(40.0)
    # The score prefers the true alignment to a clock that is off by an hour.
    scores = llr(obs, pred, Params(), np.array([0.0, 3600.0]))
    assert scores[0] > scores[1]


def test_level_curve_picks_the_right_position_and_clock(lane):
    tracks, t_start, t_end = lane
    params = LevelParams()
    tgrid = t_start + 60.0 * (np.arange(1440) + 0.5)
    rng = np.random.default_rng(3)
    truth = predicted_curve(tracks, tgrid, 0.0, 0.0, params)
    # What a hydrophone at the origin would log: the predicted excess over a noisy floor.
    observed = observed_excess(95.0 + truth + rng.normal(0, 1.0, len(tgrid)), 60.0)
    z_here = circular_z(observed, truth)[1]
    z_20km_east = circular_z(observed, predicted_curve(tracks, tgrid, 20_000.0, 0.0, params))[1]
    z_far_side = circular_z(observed, predicted_curve(tracks, tgrid, 0.0, 30_000.0, params))[1]
    assert z_here > 6.0
    # Along the lane every ship's passage time shifts by a different amount, so the match
    # collapses. Straight across the lane the times are the same and only the ranges
    # differ, so that direction is constrained far more weakly.
    assert z_here > z_20km_east + 5.0
    assert z_here > z_far_side + 1.0

    # A recorder whose clock runs 90 minutes fast stamps the clip 90 minutes late. The
    # sweep predicts over the stamped times and recovers the offset to the minute.
    k = 90
    stamped = t_start + k * 60.0 + 60.0 * (np.arange(-240, 1440 + 240) + 0.5)
    best, _ = sweep_offsets(observed, predicted_curve(tracks, stamped, 0.0, 0.0, params), 240, 240)
    assert best == k
    assert search_penalty(1) == 0.0 and search_penalty(1681) > 3.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_audio_ingest_keeps_a_tone(tmp_path):
    import soundfile as sf

    sr = 16_000
    t = np.arange(sr * 30) / sr
    x = 0.2 * np.sin(2 * np.pi * 137.0 * t) + 0.01 * np.random.default_rng(0).normal(size=len(t))
    src = tmp_path / "tone_20200224T120000Z.wav"
    sf.write(src, x.astype(np.float32), sr)
    lf = lowrate_copy(src, tmp_path / "tone.2k.flac")
    y, rate = read_audio(lf)
    assert rate == 2000 and abs(len(y) - 60_000) < 200
    sp = spectra(y, rate, start_time_from_name(src.name), dt=10.0)
    assert sp.freqs[int(np.argmax(sp.psd_db.mean(axis=0)))] == pytest.approx(137.0, abs=1.0)
    assert sp.t[0] % 10.0 == 0.0


def test_fringes_give_closest_approach_and_range_over_speed():
    # Lloyd's-mirror stripes for a pass with r/v = 10 min, closest at 30 min into the clip.
    rng = np.random.default_rng(5)
    t = T0 + 10.0 * np.arange(360)
    freqs = np.arange(0.0, 1001.0)
    t0, tau = T0 + 1800.0, 600.0
    spacing = 38.0 * np.sqrt(1.0 + ((t + 5.0 - t0) / tau) ** 2)
    psd = 60.0 + 6.0 * np.cos(2 * np.pi * freqs[None, :] / spacing[:, None]) + rng.normal(0, 3.0, (len(t), len(freqs)))
    fit = fit_fringes(t, freqs, psd, t0 + 150.0)        # start from a level peak 2.5 min late
    assert fit.reliable
    assert fit.t0 == pytest.approx(t0, abs=30.0)
    assert fit.tau == pytest.approx(tau, rel=0.2)

    # Featureless noise must not be reported as a passage.
    noise = 60.0 + rng.normal(0, 3.0, (len(t), len(freqs)))
    assert not fit_fringes(t, freqs, noise, t0).reliable
