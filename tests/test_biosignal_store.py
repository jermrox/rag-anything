"""Tests for the durable report store."""

import json

import pytest

from raganything.biosignal import narrative
from raganything.biosignal.schema import (
    Modality,
    Provenance,
    Session,
    SourceKind,
    make_stream,
)
from raganything.biosignal.store import (
    REPORT_SCHEMA_VERSION,
    ReportRecord,
    ReportSchemaError,
    ReportStore,
    content_hash_for,
)

T0 = 1_700_000_000.0
DAY = 86400.0


def provenance(source_id="strap"):
    return Provenance(
        source_id=source_id,
        kind=SourceKind.BLE,
        device="chest strap",
        latency_s=0.05,
        nominal_hz=1.0,
    )


def build_report(session_id="ride-01", start=T0, minutes=10, amplitude=20.0):
    n = minutes * 60
    rr = make_stream(
        Modality.RR_INTERVAL,
        [(start + i, 1000.0 + (amplitude if i % 2 else -amplitude)) for i in range(n)],
        provenance("strap-rr"),
    )
    hr = make_stream(
        Modality.HEART_RATE,
        [(start + i, 150.0) for i in range(n)],
        provenance("strap-hr"),
    )
    session = Session(
        session_id, start, start + n, streams=[rr, hr], labels={"sport": "cycling"}
    )
    return narrative.analyze_session(session)


class TestRoundTrip:
    def test_put_then_get(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        record = store.get("ride-01")

        assert record is not None
        assert record.session_id == "ride-01"
        assert record.subject_id == "self"
        assert "cycling" == record.labels["sport"]
        assert record.schema_version == REPORT_SCHEMA_VERSION

    def test_atomic_write_leaves_no_temp_file(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        assert list((tmp_path / "reports").glob("*.tmp")) == []

    def test_overwrites_by_session_id(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report(amplitude=20.0))
        first = store.get("ride-01").metric("hrv_rmssd")
        store.put(build_report(amplitude=40.0))
        second = store.get("ride-01").metric("hrv_rmssd")

        # Unlike the knowledge graph, a re-analysed session replaces the old
        # one. That is what keeps arithmetic correct against a stale graph.
        assert len(store) == 1
        assert second != first

    def test_missing_session_returns_none(self, tmp_path):
        assert ReportStore(tmp_path).get("nope") is None

    def test_ids_needing_sanitising_do_not_collide(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report(session_id="ride/01"))
        store.put(build_report(session_id="ride:01"))
        assert len(store) == 2

    def test_delete(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        assert store.delete("ride-01") is True
        assert store.delete("ride-01") is False
        assert len(store) == 0


class TestReadModel:
    def test_metric_returns_none_for_withheld(self, tmp_path):
        store = ReportStore(tmp_path)
        report = build_report()
        report.metrics["hrv_rmssd"] = 40.0
        report.withheld["hrv_rmssd"] = "withheld for testing"
        store.put(report)

        record = store.get("ride-01")
        # Present in metrics AND withheld: withholding must win, or the whole
        # guarantee is defeated by a dictionary lookup.
        assert record.metric("hrv_rmssd") is None
        assert record.is_withheld("hrv_rmssd")
        assert record.withheld_reason("hrv_rmssd") == "withheld for testing"

    def test_modality_quality_uses_the_recorded_gate(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        record = store.get("ride-01")
        assert record.quality_for("rr_interval") is not None
        assert record.quality_for("heart_rate") is not None
        assert record.quality_for("power") is None

    def test_day_and_iso_date(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        record = store.get("ride-01")
        assert record.iso_date() == record.day().isoformat()

    def test_record_dict_round_trip(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        record = store.get("ride-01")
        assert ReportRecord.from_dict(record.to_dict()) == record


class TestListing:
    def _three(self, tmp_path):
        store = ReportStore(tmp_path)
        for i in range(3):
            store.put(build_report(session_id=f"ride-{i}", start=T0 + i * DAY))
        return store

    def test_sorted_oldest_first(self, tmp_path):
        store = self._three(tmp_path)
        assert [r.session_id for r in store.list()] == ["ride-0", "ride-1", "ride-2"]

    def test_window_filter_is_half_open(self, tmp_path):
        store = self._three(tmp_path)
        selected = store.list(start=T0 + DAY, end=T0 + 2 * DAY)
        assert [r.session_id for r in selected] == ["ride-1"]

    def test_label_filter(self, tmp_path):
        store = self._three(tmp_path)
        assert len(store.list(labels={"sport": "cycling"})) == 3
        assert store.list(labels={"sport": "swimming"}) == []

    def test_metric_filter_includes_withheld_sessions(self, tmp_path):
        store = ReportStore(tmp_path)
        report = build_report()
        report.metrics.pop("hrv_rmssd", None)
        report.withheld["hrv_rmssd"] = "withheld for testing"
        store.put(report)
        # A caller asking about RMSSD needs the nights it was withheld too.
        assert len(store.list(metric="hrv_rmssd")) == 1

    def test_span(self, tmp_path):
        store = self._three(tmp_path)
        low, high = store.span()
        assert low == T0
        assert high > T0 + 2 * DAY

    def test_empty_store(self, tmp_path):
        store = ReportStore(tmp_path)
        assert store.list() == []
        assert store.span() is None
        assert len(store) == 0
        assert "anything" not in store


class TestIndexAndResilience:
    def test_rebuild_index_matches_a_full_scan(self, tmp_path):
        store = ReportStore(tmp_path)
        for i in range(4):
            store.put(build_report(session_id=f"ride-{i}", start=T0 + i * DAY))
        assert store.rebuild_index() == 4

        lines = store.index_path.read_text().strip().split("\n")
        assert len(lines) == 4
        assert [json.loads(line)["session_id"] for line in lines] == [
            "ride-0",
            "ride-1",
            "ride-2",
            "ride-3",
        ]

    def test_corrupt_file_is_skipped_not_fatal(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report(session_id="good"))
        (tmp_path / "reports" / "broken.json").write_text("{not json")

        records = store.list()
        assert [r.session_id for r in records] == ["good"]
        assert "broken.json" in store.errors

    def test_future_schema_version_is_refused(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        path = next((tmp_path / "reports").glob("*.json"))
        envelope = json.loads(path.read_text())
        envelope["schema_version"] = REPORT_SCHEMA_VERSION + 1
        path.write_text(json.dumps(envelope))

        with pytest.raises(ReportSchemaError, match="Refusing to guess"):
            store.get("ride-01")

    def test_missing_migration_is_an_error(self, tmp_path):
        store = ReportStore(tmp_path)
        store.put(build_report())
        path = next((tmp_path / "reports").glob("*.json"))
        envelope = json.loads(path.read_text())
        envelope["schema_version"] = -1
        path.write_text(json.dumps(envelope))

        with pytest.raises(ReportSchemaError, match="no migration"):
            store.get("ride-01")


class TestConstruction:
    def test_for_rag_uses_working_dir(self, tmp_path):
        class FakeRAG:
            working_dir = str(tmp_path)

        store = ReportStore.for_rag(FakeRAG())
        assert store.root == tmp_path / "biosignal"

    def test_for_rag_falls_back_to_config(self, tmp_path):
        class Config:
            working_dir = str(tmp_path)

        class FakeRAG:
            config = Config()

        assert ReportStore.for_rag(FakeRAG()).root == tmp_path / "biosignal"

    def test_maybe_for_rag_returns_none_without_a_working_dir(self):
        assert ReportStore.maybe_for_rag(object()) is None

    def test_for_rag_raises_without_a_working_dir(self):
        with pytest.raises(ValueError, match="working_dir"):
            ReportStore.for_rag(object())


class TestContentHash:
    def test_stable_and_order_sensitive(self):
        a = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
        b = [{"type": "text", "text": "two"}, {"type": "text", "text": "one"}]
        assert content_hash_for(a) == content_hash_for(list(a))
        assert content_hash_for(a) != content_hash_for(b)
