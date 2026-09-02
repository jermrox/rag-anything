"""Tests for the schema, source normalisers, narrative rendering and indexing."""

import asyncio
from datetime import datetime, timezone

import pytest

from raganything.biosignal import narrative, sources
from raganything.biosignal.index import index_session
from raganything.biosignal.schema import (
    Evidence,
    Modality,
    Provenance,
    Sample,
    Session,
    SourceKind,
    Stream,
    make_stream,
)

T0 = 1_700_000_000.0


def ble_provenance(source_id="strap", **kw):
    defaults = dict(
        source_id=source_id,
        kind=SourceKind.BLE,
        device="chest strap",
        latency_s=0.05,
        nominal_hz=1.0,
    )
    defaults.update(kw)
    return Provenance(**defaults)


class TestSchema:
    def test_units_are_defined_for_every_modality(self):
        from raganything.biosignal.schema import UNITS

        for modality in Modality:
            assert modality in UNITS

    def test_stream_takes_its_unit_from_its_modality(self):
        stream = Stream(modality=Modality.POWER, provenance=ble_provenance())
        assert stream.unit == "W"

    def test_implausible_values_are_flagged_not_dropped(self):
        stream = make_stream(
            Modality.HEART_RATE,
            [(T0, 120.0), (T0 + 1, 900.0)],
            ble_provenance(),
        ).flag_implausible()
        assert len(stream) == 2
        assert stream.samples[1].flags == ("out_of_range",)
        assert stream.samples[1].confidence == 0.0

    def test_window_preserves_provenance(self):
        stream = make_stream(
            Modality.HEART_RATE,
            [(T0 + i, 120.0) for i in range(10)],
            ble_provenance(),
        )
        sliced = stream.window(T0 + 2, T0 + 5)
        assert len(sliced) == 3
        assert sliced.provenance is stream.provenance

    def test_session_prefers_measured_streams_over_vendor_scores(self):
        measured = make_stream(
            Modality.HEART_RATE, [(T0, 120.0)], ble_provenance("strap")
        )
        vendor = Stream(
            modality=Modality.HEART_RATE,
            provenance=Provenance(
                source_id="ring", kind=SourceKind.VENDOR_CLOUD, latency_s=21600
            ),
            samples=[Sample(t=T0, value=118.0, evidence=Evidence.VENDOR_DERIVED)],
        )
        session = Session("s", T0, T0 + 10, streams=[vendor, measured])
        assert session.first(Modality.HEART_RATE) is measured
        assert session.first(Modality.GLUCOSE) is None

    def test_sample_flags_deduplicate(self):
        sample = Sample(t=T0, value=1.0).with_flag("a").with_flag("a", "b")
        assert sample.flags == ("a", "b")

    def test_evidence_ranking_orders_measurement_first(self):
        ranks = [Evidence.rank(e) for e in Evidence]
        assert ranks == sorted(ranks)
        assert Evidence.rank(Evidence.MEASURED) < Evidence.rank(Evidence.IMPUTED)


class TestTimestampParsing:
    def test_iso_with_zulu(self):
        t = sources.parse_timestamp("2026-09-02T06:30:00Z")
        assert t == datetime(2026, 9, 2, 6, 30, tzinfo=timezone.utc).timestamp()

    def test_naive_iso_assumed_utc(self):
        assert sources.parse_timestamp(
            "2026-09-02T06:30:00"
        ) == sources.parse_timestamp("2026-09-02T06:30:00Z")

    def test_epoch_seconds_and_milliseconds(self):
        assert sources.parse_timestamp(1_700_000_000) == 1_700_000_000.0
        assert sources.parse_timestamp(1_700_000_000_000) == 1_700_000_000.0

    def test_unparseable_returns_none(self):
        assert sources.parse_timestamp("last tuesday") is None
        assert sources.parse_timestamp(None) is None


class TestVendorNormalisation:
    def test_oura_payload_stays_vendor_derived(self):
        payload = {
            "data": [
                {"timestamp": "2026-09-02T06:00:00Z", "bpm": 52, "score": 78},
                {"timestamp": "2026-09-02T06:05:00Z", "bpm": 54, "score": 78},
            ]
        }
        streams = sources.normalize_vendor_payload("oura", payload)
        by_modality = {s.modality: s for s in streams}

        hr = by_modality[Modality.HEART_RATE]
        assert [s.value for s in hr] == [52.0, 54.0]
        assert all(s.evidence is Evidence.VENDOR_DERIVED for s in hr)
        assert hr.provenance.kind is SourceKind.VENDOR_CLOUD
        assert hr.provenance.latency_s == 6 * 3600

        readiness = by_modality[Modality.READINESS]
        assert readiness.provenance.documented is False
        assert readiness.provenance.algorithm == "Oura readiness score"

    def test_whoop_rmssd_is_rescaled_to_milliseconds(self):
        payload = {
            "records": [
                {
                    "created_at": "2026-09-02T06:00:00Z",
                    "score": {"hrv_rmssd_milli": 0.065, "recovery_score": 61},
                }
            ]
        }
        streams = sources.normalize_vendor_payload("whoop", payload)
        rmssd = [s for s in streams if s.modality is Modality.HRV_RMSSD][0]
        assert rmssd.samples[0].value == pytest.approx(65.0)

    def test_records_without_timestamps_are_skipped(self):
        payload = {
            "data": [{"bpm": 52}, {"timestamp": "2026-09-02T06:00:00Z", "bpm": 60}]
        }
        streams = sources.normalize_vendor_payload("oura", payload)
        assert len(streams[0].samples) == 1

    def test_missing_records_path_yields_nothing(self):
        assert sources.normalize_vendor_payload("oura", {"unexpected": []}) == []

    def test_unknown_vendor_is_an_explicit_error(self):
        with pytest.raises(KeyError, match="no profile for vendor"):
            sources.normalize_vendor_payload("mystery", {})

    def test_fitbit_intraday_dotted_path(self):
        payload = {
            "activities-heart-intraday": {
                "dataset": [
                    {"time": "2026-09-02T06:00:00Z", "value": 58},
                    {"time": "2026-09-02T06:01:00Z", "value": 59},
                ]
            }
        }
        streams = sources.normalize_vendor_payload("fitbit", payload)
        assert [s.value for s in streams[0]] == [58.0, 59.0]


class TestHealthStoreNormalisation:
    def test_records_are_split_by_writing_app(self):
        records = [
            {
                "type": "StepsRecord",
                "startTime": "2026-09-02T06:00:00Z",
                "value": 500,
                "dataOrigin": "com.watch",
            },
            {
                "type": "StepsRecord",
                "startTime": "2026-09-02T06:00:00Z",
                "value": 480,
                "dataOrigin": "com.phone",
            },
        ]
        streams = sources.normalize_health_records(records)
        # Two apps counting the same steps must not become one series.
        assert len(streams) == 2
        assert {s.provenance.extra["writing_app"] for s in streams} == {
            "com.watch",
            "com.phone",
        }

    def test_healthkit_identifiers_are_mapped(self):
        records = [
            {
                "type": "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
                "startDate": "2026-09-02T06:00:00Z",
                "value": 55,
                "sourceName": "Apple Watch",
            }
        ]
        streams = sources.normalize_health_records(records, platform="healthkit")
        assert streams[0].modality is Modality.HRV_SDNN
        assert streams[0].provenance.kind is SourceKind.PHONE

    def test_unknown_types_and_bad_values_are_skipped(self):
        records = [
            {"type": "MysteryRecord", "startTime": "2026-09-02T06:00:00Z", "value": 1},
            {
                "type": "StepsRecord",
                "startTime": "2026-09-02T06:00:00Z",
                "value": "many",
            },
            {"type": "StepsRecord", "value": 100},
        ]
        assert sources.normalize_health_records(records) == []


def build_session(with_rr=True, minutes=30):
    n = minutes * 60
    streams = [
        make_stream(
            Modality.HEART_RATE,
            [(T0 + i, 150.0) for i in range(n)],
            ble_provenance("strap-hr"),
        ),
        make_stream(
            Modality.POWER,
            [(T0 + i, 220.0 + (10 if i % 2 else -10)) for i in range(n)],
            ble_provenance("meter-power", device="power meter"),
        ),
    ]
    if with_rr:
        streams.append(
            make_stream(
                Modality.RR_INTERVAL,
                [(T0 + i * 0.4, 400.0 + (8 if i % 2 else -8)) for i in range(n * 2)],
                ble_provenance("strap-rr"),
            )
        )
    return Session("ride-001", T0, T0 + n, streams=streams, labels={"sport": "cycling"})


class TestAnalyzeSession:
    def test_reports_metrics_it_can_support(self):
        report = narrative.analyze_session(build_session())
        assert "mean_hr" in report.metrics
        assert "normalized_power" in report.metrics
        assert report.metrics["mean_hr"] == pytest.approx(150.0)

    def test_trimp_is_withheld_without_measured_athlete_constants(self):
        report = narrative.analyze_session(build_session())
        assert "trimp" not in report.metrics
        assert "population estimate" in report.withheld["trimp"]

    def test_trimp_appears_once_constants_are_supplied(self):
        report = narrative.analyze_session(build_session(), rest_hr=48, max_hr=188)
        assert report.metrics["trimp"] > 0

    def test_hrv_is_not_faked_from_averaged_heart_rate(self):
        report = narrative.analyze_session(build_session(with_rr=False))
        assert report.hrv is None
        assert "only from RR intervals" in report.withheld["hrv_rmssd"]

    def test_hrv_computed_when_beat_intervals_exist(self):
        report = narrative.analyze_session(build_session())
        assert report.hrv is not None
        assert "hrv_rmssd" in report.metrics

    def test_intensity_factor_requires_a_measured_threshold(self):
        report = narrative.analyze_session(build_session())
        assert (
            "needs a measured functional threshold power"
            in report.withheld["intensity_factor"]
        )
        with_ftp = narrative.analyze_session(build_session(), threshold_power=250.0)
        assert with_ftp.metrics["intensity_factor"] == pytest.approx(0.88, abs=0.02)

    def test_conflicting_sources_surface_as_warnings(self):
        session = build_session()
        session.add(
            Stream(
                modality=Modality.HEART_RATE,
                provenance=Provenance(
                    source_id="watch",
                    kind=SourceKind.VENDOR_CLOUD,
                    device="wrist optical",
                    latency_s=300.0,
                    documented=False,
                ),
                samples=[
                    Sample(
                        t=T0 + i,
                        value=175.0,
                        evidence=Evidence.VENDOR_DERIVED,
                        confidence=0.6,
                    )
                    for i in range(1800)
                ],
            )
        )
        report = narrative.analyze_session(session)
        assert any("disagree on heart_rate" in w for w in report.warnings)
        assert report.fusion[Modality.HEART_RATE].chosen.provenance.source_id == (
            "strap-hr"
        )

    def test_dropout_lowers_quality_and_can_withhold_load(self):
        session = build_session()
        hr = session.of(Modality.HEART_RATE)[0]
        # Sensor dies 3 minutes in and never returns.
        hr.samples = [s for s in hr.samples if s.t < T0 + 180]
        report = narrative.analyze_session(session, rest_hr=48, max_hr=188)
        assert report.quality["strap-hr"].coverage < 0.2
        assert "trimp" in report.withheld


class TestContentList:
    def test_renders_text_and_tables(self):
        report = narrative.analyze_session(build_session())
        items = narrative.to_content_list(report)
        types = [i["type"] for i in items]
        assert types[0] == "text"
        assert types.count("table") >= 2
        assert all("page_idx" in i for i in items)
        assert [i["page_idx"] for i in items] == list(range(len(items)))

    def test_provenance_travels_with_the_numbers(self):
        report = narrative.analyze_session(build_session())
        items = narrative.to_content_list(report)
        inventory = [i for i in items if i["type"] == "table"][0]["table_body"]
        assert "chest strap" in inventory
        assert "measured" in inventory
        assert "coverage (%)" in inventory

    def test_withheld_metrics_are_written_down_as_withheld(self):
        report = narrative.analyze_session(build_session())
        items = narrative.to_content_list(report)
        ledger = [i for i in items if i["type"] == "table"][1]["table_body"]
        assert "withheld" in ledger
        assert "trimp" in ledger

    def test_caveats_section_appears_when_quality_drops(self):
        session = build_session()
        hr = session.of(Modality.HEART_RATE)[0]
        hr.samples = [s for s in hr.samples if s.t < T0 + 180]
        items = narrative.to_content_list(narrative.analyze_session(session))
        assert any("Data-quality caveats" in i.get("text", "") for i in items)

    def test_page_offset_is_respected(self):
        report = narrative.analyze_session(build_session())
        items = narrative.to_content_list(report, page_offset=10)
        assert items[0]["page_idx"] == 10


class _FakeRAG:
    def __init__(self):
        self.calls = []

    async def insert_content_list(self, content_list, file_path, doc_id, **kwargs):
        self.calls.append(
            {"content_list": content_list, "file_path": file_path, "doc_id": doc_id}
        )


class TestIndexing:
    def test_session_is_inserted_with_a_stable_document_id(self):
        rag = _FakeRAG()
        report = asyncio.run(index_session(rag, build_session()))

        assert len(rag.calls) == 1
        call = rag.calls[0]
        assert call["doc_id"] == "biosignal:ride-001"
        assert call["file_path"] == "ride-001.biosignal"
        assert call["content_list"]
        assert report.metrics["mean_hr"] == pytest.approx(150.0)

    def test_analysis_arguments_pass_through(self):
        rag = _FakeRAG()
        report = asyncio.run(
            index_session(rag, build_session(), rest_hr=48, max_hr=188)
        )
        assert "trimp" in report.metrics

    def test_empty_session_indexes_nothing_but_does_not_raise(self):
        rag = _FakeRAG()
        empty = Session("empty", T0, T0)
        report = asyncio.run(index_session(rag, empty))
        assert report.session.session_id == "empty"
        # An overview is always produced, so the insert still happens; what
        # matters is that no metric was invented for it.
        assert report.metrics == {}
