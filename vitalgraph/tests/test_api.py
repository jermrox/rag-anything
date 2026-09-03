"""API surface tests.

The server module builds its store at import time from environment config, so
the environment is redirected to a temporary database *before* the import.
"""

import importlib
import os
import tempfile

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

_TMP = tempfile.mkdtemp(prefix="vitalgraph-test-")
os.environ["VITALGRAPH_DATA_DIR"] = _TMP
os.environ["VITALGRAPH_DB"] = os.path.join(_TMP, "test.db")

server = importlib.import_module("vitalgraph.api.server")
client = fastapi_testclient.TestClient(server.app)


@pytest.fixture(scope="module", autouse=True)
def seeded():
    client.post("/api/demo/seed", json={"nights": 9, "seed": 11})


def test_health_reports_stored_data():
    d = client.get("/api/health").json()
    assert d["status"] == "ok"
    assert d["samples"] > 0
    assert "rr_interval" in d["signals"]


def test_metrics_returns_nightly_periods_with_citations():
    d = client.get("/api/metrics?nights=9").json()
    periods = d["periods"]
    assert len(periods) >= 2
    for p in periods:
        assert p["citation"].startswith("biometrics://")
        assert "rmssd_ms" in p["metrics"]
        assert p["narrative"]


def test_metrics_builds_a_baseline_then_flags_a_decline():
    periods = client.get("/api/metrics?nights=9").json()["periods"]
    assert periods[0]["rmssd_z"] is None          # no baseline on night one
    assert any(p["rmssd_z"] is not None for p in periods)
    assert periods[-1]["metrics"]["rmssd_ms"] < periods[0]["metrics"]["rmssd_ms"]


def test_ingest_raw_gatt_notification():
    from vitalgraph.ble.gatt import encode_heart_rate_measurement

    payload = encode_heart_rate_measurement(66, [820.0, 830.0, 840.0])
    r = client.post("/api/ingest/gatt", json={"hex": payload.hex(), "ts_ms": 1772582400000})
    assert r.status_code == 200
    d = r.json()
    assert d["heart_rate"] == 66
    assert d["rr_count"] == 3
    assert d["inserted"] == 4  # 1 heart-rate sample + 3 RR intervals


def test_ingest_gatt_rejects_malformed_payload():
    assert client.post("/api/ingest/gatt", json={"hex": "zz"}).status_code == 422
    assert client.post("/api/ingest/gatt", json={"hex": "00"}).status_code == 422


def test_ingest_stream_accepts_decoded_samples():
    r = client.post("/api/ingest/stream", json={"samples": [
        {"ts_ms": 1772582400000, "signal": "rr_interval", "value": 812.0},
        {"ts_ms": 1772582401000, "signal": "heart_rate", "value": 74.0},
    ]})
    assert r.status_code == 200
    assert r.json()["received"] == 2


def test_ingest_stream_rejects_implausible_values():
    r = client.post("/api/ingest/stream", json={"samples": [
        {"ts_ms": 1772582400000, "signal": "spo2", "value": 140.0},
    ]})
    assert r.status_code == 422


def test_protocol_derivable_answers_what_hardware_can_measure():
    d = client.get("/api/protocol/derivable?uuids=0x2A37,0x9999").json()
    assert "rmssd" in d["derivable"]["0x2A37"]
    assert d["unrecognised"] == ["0x9999"]


def test_rag_endpoints_degrade_clearly_without_raganything():
    """Analytics must stay usable when the RAG stack is absent."""
    r = client.post("/api/query", json={"question": "how did I sleep?"})
    assert r.status_code == 503
    assert "raganything" in r.json()["detail"].lower()


def test_ui_is_served_at_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "VitalGraph" in r.text
    assert "navigator.bluetooth" in r.text


def test_partial_night_is_omitted_rather_than_plotted_as_zero():
    """Regression: a stray notification landing in a fresh day created a
    window with a handful of beats. Reporting it produced a 0 ms RMSSD point,
    and missing data drawn as zero reads as 'no variability'."""
    from vitalgraph.ble.gatt import encode_heart_rate_measurement

    before = len(client.get("/api/metrics?nights=3650").json()["periods"])

    # One notification, timestamped well past the seeded nights.
    payload = encode_heart_rate_measurement(64, [840.0])
    far_future = 2_000_000_000_000  # 2033, far beyond the seeded window
    client.post("/api/ingest/gatt", json={"hex": payload.hex(), "ts_ms": far_future})

    periods = client.get("/api/metrics?nights=3650").json()["periods"]
    assert len(periods) == before
    assert all(p["metrics"]["n_beats"] > 0 for p in periods)
    assert all(p["metrics"]["rmssd_ms"] > 0 for p in periods)
