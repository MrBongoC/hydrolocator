from datetime import date, datetime, timezone

import duckdb
import numpy as np
import pytest

from hydrolocator import multisite as ms
from hydrolocator import shipdata, suspects
from hydrolocator import voiceprint as vp
from hydrolocator.geo import KNOT


def test_flag_from_mmsi_and_government_call_signs():
    assert shipdata.flag(538002221) == "Marshall Islands"
    assert shipdata.flag(367866000) == "United States"
    assert shipdata.flag(111111111) is None and shipdata.flag(12345) is None
    # US government ships carry four-letter call signs starting with N; merchant ships do not.
    assert shipdata.is_government(80, "LEROY GRUMMAN", callsign="NNLG", mmsi=367866000)
    assert not shipdata.is_government(70, "MSC ZLATA R", "MSC Shipmanagement", callsign="D5GE5", mmsi=636016436)
    assert shipdata.is_government(35) and shipdata.is_government(70, operator="United States Navy")


def test_silent_course_past_the_site_is_found(tmp_path, monkeypatch):
    monkeypatch.setenv("HYDROLOCATOR_DATA", str(tmp_path))
    from hydrolocator import ais

    lat, lon = 42.40, -70.50                                  # inside the Stellwagen region
    day = date(2020, 3, 1)
    t0 = datetime(2020, 3, 1, tzinfo=timezone.utc).timestamp()
    rows = []
    # Vessel 1 reports every minute heading east along the site's latitude, falls silent for
    # two hours while it passes, then reports again. Vessel 2 reports all day, far to the north.
    for k in range(0, 60):
        rows.append((367000001, t0 + 3600 + 60 * k, lat, lon - 0.50 + 0.002 * k, "QUIET ONE", 70, 120.0))
    for k in range(0, 60):
        rows.append((367000001, t0 + 3600 + 3540 + 7200 + 60 * k, lat, lon + 0.20 + 0.002 * k, "QUIET ONE", 70, 120.0))
    for k in range(0, 600):
        rows.append((367000002, t0 + 60 * k, lat + 0.5, lon - 0.4 + 0.001 * k, "CHATTY", 70, 90.0))
    con = duckdb.connect()
    con.execute("CREATE TABLE p (mmsi BIGINT, t DOUBLE, lat DOUBLE, lon DOUBLE, name VARCHAR, vessel_type INTEGER, length FLOAT)")
    con.executemany("INSERT INTO p VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    con.execute(f"COPY p TO '{ais.points_path(day)}' (FORMAT parquet)")
    con.close()

    gaps = suspects.silent_courses(lat, lon, t0, t0 + 86400)
    assert [g.name for g in gaps] == ["QUIET ONE"]
    g = gaps[0]
    assert g.bridge_range_m < 1000.0
    assert 6.0 < g.speed_kn < 14.0
    assert suspects.for_event(gaps, g.bridge_time + 300) == [g]
    assert suspects.for_event(gaps, g.bridge_time + 4 * 3600) == []


def test_voiceprint_tells_two_line_spectra_apart():
    rng = np.random.default_rng(2)
    freqs = np.arange(0.0, 1001.0)
    t = 60.0 * np.arange(120)

    def clip(lines_hz, at):
        psd = 60.0 + rng.normal(0, 1.0, (len(t), len(freqs)))
        for f in lines_hz:
            psd[at - 2:at + 3, int(f)] += 14.0
        return psd

    a1 = vp.signature(vp.line_spectra(freqs, clip([62, 124, 186, 301], 40)), t, t[40])
    a2 = vp.signature(vp.line_spectra(freqs, clip([62, 124, 186, 301], 90)), t, t[90])
    b = vp.signature(vp.line_spectra(freqs, clip([47, 94, 141, 260], 60)), t, t[60])
    assert float(a1 @ a2) > 0.7
    assert float(a1 @ b) < 0.3


def test_track_inference_rescales_a_known_crossing():
    path = np.array([[1000.0 + 60 * k, 42.0 + 0.001 * k, -70.0] for k in range(100)])
    seen = ms.Crossing(date(2020, 1, 1), 1, "TEMPLATE", 70, 150.0, t_a=1000.0, t_b=7000.0, r_a=2000.0, r_b=3000.0,
                       speed=10 * KNOT, path=path)
    # The same crossing made in 3,000 s instead of 6,000 s is the same route at twice the speed.
    fast = ms.infer(50_000.0, 53_000.0, [seen])
    assert fast.speed_kn == pytest.approx(20.0, rel=1e-6)
    assert fast.r_a_km == pytest.approx(2.0) and fast.r_b_km == pytest.approx(3.0)
    assert fast.path[0, 0] == pytest.approx(50_000.0) and fast.like == "TEMPLATE"
    # A crossing in the other direction has no template to borrow from.
    assert ms.infer(53_000.0, 50_000.0, [seen]) is None
