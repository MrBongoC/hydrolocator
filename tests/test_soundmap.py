"""The sample day that ships with the package: the map has to play with nothing downloaded."""

from datetime import date

import pytest

from hydrolocator import soundmap

SAMPLE = date(2019, 8, 10)


@pytest.fixture
def empty_store(tmp_path, monkeypatch):
    """A data directory with no AIS, no audio and no caches, like a fresh clone."""
    monkeypatch.setenv("HYDROLOCATOR_DATA", str(tmp_path))
    return tmp_path


def test_sample_day_plays_from_an_empty_store(empty_store):
    assert SAMPLE.isoformat() in soundmap.available("stellwagen")
    day = soundmap.build("stellwagen", SAMPLE)
    assert day["date"] == "2019-08-10" and day["steps"] == 720
    assert len(day["vessels"]) == 289
    assert [p["id"] for p in day["posts"]] == ["sanctsound:sb01", "sanctsound:sb03"]
    assert not list(empty_store.rglob("*.json")), "reading the sample must not write a cache"


def test_a_day_with_no_data_says_so(empty_store):
    with pytest.raises(LookupError, match="no AIS held"):
        soundmap.build("stellwagen", date(2019, 8, 11))


def test_api_serves_the_sample_day(empty_store):
    from fastapi.testclient import TestClient

    from hydrolocator.api import app

    client = TestClient(app)
    catalog = client.get("/api/catalog").json()
    assert catalog["ais_days"] == [] and catalog["soundmap_days"] == ["2019-08-10"]
    assert len(catalog["sites"]) == 32
    assert client.get("/api/soundmap?region=stellwagen&date=2019-08-10").status_code == 200
    assert client.get("/api/soundmap?region=stellwagen&date=2019-08-11").status_code == 404
