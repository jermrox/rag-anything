"""Tests for HRV, quality, load and cross-source fusion."""

import math

import pytest

from raganything.biosignal.analytics import fusion, hrv, load, quality
from raganything.biosignal.schema import (
    Evidence,
    Modality,
    Provenance,
    Sample,
    SourceKind,
    Stream,
    make_stream,
)

T0 = 1_700_000_000.0


def strap_provenance(source_id="strap", **kw):
    defaults = dict(
        source_id=source_id,
        kind=SourceKind.BLE,
        device="chest strap",
        transport="GATT notify 0x2A37",
        latency_s=0.05,
        nominal_hz=1.0,
    )
    defaults.update(kw)
    return Provenance(**defaults)


def steady_rr(n=400, base=1000.0, jitter=20.0):
    """A physiologically plausible beat series with mild alternation."""
    return [base + (jitter if i % 2 else -jitter) for i in range(n)]


class TestArtifactCorrection:
    def test_clean_series_keeps_everything(self):
        report = hrv.correct_rr(steady_rr(50))
        assert report.artifact_fraction == 0.0
        assert len(report.kept) == 50

    def test_malik_rejects_a_sudden_jump(self):
        rr = [1000.0] * 10 + [500.0] + [1000.0] * 10
        report = hrv.correct_rr(rr, method="malik")
        assert 10 in report.rejected_index
        assert report.reasons[10] == "malik_successive_deviation"

    def test_karlsson_uses_local_mean(self):
        rr = [1000.0] * 10 + [1400.0] + [1000.0] * 10
        report = hrv.correct_rr(rr, method="karlsson")
        assert 10 in report.rejected_index
        assert report.reasons[10] == "karlsson_local_mean_deviation"

    def test_physiologically_impossible_values_always_rejected(self):
        rr = [1000.0, 40.0, 1000.0, 9000.0, float("nan")]
        report = hrv.correct_rr(rr, method="none")
        assert set(report.rejected_index) == {1, 3, 4}
        for i in (1, 3, 4):
            assert report.reasons[i] == "outside_physiological_bounds"

    def test_contiguity_is_tracked_across_removals(self):
        rr = [1000.0] * 5 + [400.0] + [1000.0] * 5
        report = hrv.correct_rr(rr)
        # The pair spanning the removed beat must not be treated as adjacent.
        boundary = report.kept_index.index(4)
        assert not report.is_contiguous_pair(boundary)


class TestHRVMetrics:
    def test_rmssd_matches_hand_computation(self):
        rr = [1000.0, 1020.0, 1000.0, 1020.0, 1000.0]
        # successive differences are all +/-20 ms
        assert hrv.rmssd(rr) == pytest.approx(20.0)

    def test_sdnn_and_pnn50(self):
        rr = steady_rr(200)
        assert hrv.sdnn(rr) == pytest.approx(20.05, abs=0.1)
        # every successive difference is 40 ms, so all exceed 50 ms? no: 40 < 50
        assert hrv.pnn50(rr) == pytest.approx(0.0)

    def test_pnn50_counts_large_differences(self):
        rr = [1000.0 + (60.0 if i % 2 else -60.0) for i in range(100)]
        assert hrv.pnn50(rr) == pytest.approx(100.0)

    def test_successive_differences_never_span_a_rejected_beat(self):
        # Without contiguity tracking, the jump across the hole inflates RMSSD.
        rr = [1000.0] * 20 + [400.0] + [1200.0] * 20
        naive = math.sqrt(
            sum(
                (b - a) ** 2
                for a, b in zip(
                    [v for v in rr if v != 400.0], [v for v in rr if v != 400.0][1:]
                )
            )
            / 39
        )
        assert hrv.rmssd(rr) < naive

    def test_full_metrics_on_a_long_clean_window(self):
        result = hrv.hrv_metrics(steady_rr(400))
        assert result.usable
        assert set(result.metrics) == {"rmssd", "sdnn", "pnn50", "mean_rr", "mean_hr"}
        assert result.confidence == pytest.approx(1.0)
        assert result.metrics["mean_hr"] == pytest.approx(60.0, abs=0.5)

    def test_short_window_withholds_the_metrics_it_cannot_support(self):
        # 20 beats at 1 s is a 20 s window: below the floor for every metric
        # whose name implies a longer one.
        result = hrv.hrv_metrics(steady_rr(20))
        assert "sdnn" in result.withheld
        assert "rmssd" in result.withheld
        assert "120s minimum" in result.withheld["sdnn"]

    def test_sdnn_withheld_but_rmssd_reported_in_the_middle_band(self):
        result = hrv.hrv_metrics(steady_rr(60))  # 60 s window
        assert "rmssd" in result.metrics
        assert "sdnn" in result.withheld

    def test_heavy_artifact_load_reports_nothing(self):
        rr = [1000.0 if i % 3 else 400.0 for i in range(90)]
        result = hrv.hrv_metrics(rr)
        assert not result.usable
        assert result.metrics == {}
        assert result.confidence == 0.0
        assert "ceiling" in result.notes[0]

    def test_confidence_degrades_with_artifacts(self):
        clean = hrv.hrv_metrics(steady_rr(400))
        rr = steady_rr(400)
        for i in range(0, 400, 50):  # 2% of beats corrupted
            rr[i] = 1500.0
        dirty = hrv.hrv_metrics(rr)
        assert dirty.confidence < clean.confidence
        assert dirty.artifact_fraction > 0

    def test_empty_input_is_handled(self):
        result = hrv.hrv_metrics([])
        assert not result.usable
        assert result.n_beats_used == 0
        assert result.withheld["rmssd"] == "no beat intervals supplied"


class TestQuality:
    def test_continuous_stream_scores_high(self):
        stream = make_stream(
            Modality.HEART_RATE,
            [(T0 + i, 120.0) for i in range(100)],
            strap_provenance(),
        )
        report = quality.assess(stream)
        assert report.coverage == pytest.approx(1.0)
        assert report.score > 0.95
        assert report.reasons == ["continuous, unflagged, directly measured"]

    def test_dropout_is_detected_and_named(self):
        times = list(range(0, 50)) + list(range(80, 130))
        stream = make_stream(
            Modality.HEART_RATE,
            [(T0 + i, 120.0) for i in times],
            strap_provenance(),
        )
        report = quality.assess(stream)
        assert len(report.gaps) == 1
        assert report.longest_gap_s == pytest.approx(31.0)
        assert report.coverage < 1.0
        assert "no data" in report.reasons[0]

    def test_late_connection_detected_against_the_session_window(self):
        stream = make_stream(
            Modality.HEART_RATE,
            [(T0 + 600 + i, 120.0) for i in range(100)],
            strap_provenance(),
        )
        # The stream looks perfect on its own; only the session window reveals
        # that the sensor connected ten minutes late.
        alone = quality.assess(stream)
        windowed = quality.assess(stream, window=(T0, T0 + 700))
        assert alone.coverage == pytest.approx(1.0)
        assert windowed.coverage < 0.2

    def test_empty_stream_scores_zero(self):
        stream = Stream(modality=Modality.POWER, provenance=strap_provenance())
        report = quality.assess(stream, window=(T0, T0 + 60))
        assert report.score == 0.0
        assert report.reasons == ["stream contains no samples"]

    def test_flags_and_inference_reduce_the_score(self):
        stream = Stream(
            modality=Modality.HEART_RATE,
            provenance=strap_provenance(),
            samples=[
                Sample(
                    t=T0 + i,
                    value=120.0,
                    evidence=Evidence.VENDOR_DERIVED if i < 20 else Evidence.MEASURED,
                    flags=("no_contact",) if i < 10 else (),
                )
                for i in range(100)
            ],
        )
        report = quality.assess(stream)
        assert report.flagged_fraction == pytest.approx(0.10)
        assert report.inferred_fraction == pytest.approx(0.20)
        assert report.score < 0.9

    def test_provenance_flags_are_not_counted_as_defects(self):
        # "platform_store" records where a sample came from; that is already
        # captured by its evidence class and must not be charged twice.
        stream = Stream(
            modality=Modality.HEART_RATE,
            provenance=strap_provenance(),
            samples=[
                Sample(
                    t=T0 + i,
                    value=120.0,
                    evidence=Evidence.VENDOR_DERIVED,
                    flags=("platform_store",),
                )
                for i in range(100)
            ],
        )
        report = quality.assess(stream)
        assert report.flagged_fraction == 0.0
        assert report.inferred_fraction == 1.0
        assert report.score > 0.0

    def test_undocumented_high_latency_source_is_penalised(self):
        stream = make_stream(
            Modality.READINESS,
            [(T0 + i * 86400, 70.0) for i in range(5)],
            Provenance(
                source_id="oura:readiness",
                kind=SourceKind.VENDOR_CLOUD,
                device="ring",
                latency_s=6 * 3600,
                algorithm="vendor readiness",
                documented=False,
            ),
        )
        report = quality.assess(stream)
        assert report.score < 0.8
        assert any("undocumented" in r for r in report.reasons)

    def test_gate_suppresses_values_the_signal_cannot_support(self):
        stream = Stream(modality=Modality.POWER, provenance=strap_provenance())
        report = quality.assess(stream, window=(T0, T0 + 60))
        value, explanation = quality.gate(240.0, report, label="mean power")
        assert value is None
        assert "withheld" in explanation

    def test_gate_passes_a_good_signal_through_with_its_reasons(self):
        stream = make_stream(
            Modality.POWER, [(T0 + i, 240.0) for i in range(60)], strap_provenance()
        )
        report = quality.assess(stream)
        value, explanation = quality.gate(240.0, report, label="mean power")
        assert value == 240.0
        assert "signal quality" in explanation


class TestLoad:
    def test_trimp_requires_real_athlete_constants(self):
        series = [(T0 + i * 60, 150.0) for i in range(30)]
        assert load.trimp_banister(series, rest_hr=60, max_hr=60) is None
        assert load.trimp_banister([], rest_hr=50, max_hr=190) is None

    def test_trimp_rises_with_intensity(self):
        easy = [(T0 + i * 60, 120.0) for i in range(60)]
        hard = [(T0 + i * 60, 170.0) for i in range(60)]
        assert load.trimp_banister(hard, 50, 190) > load.trimp_banister(easy, 50, 190)

    def test_trimp_ignores_effort_below_resting(self):
        series = [(T0 + i * 60, 45.0) for i in range(30)]
        assert load.trimp_banister(series, 50, 190) == pytest.approx(0.0)

    def test_normalized_power_undefined_for_short_efforts(self):
        short = [(T0 + i, 250.0) for i in range(10)]
        assert load.normalized_power(short) is None

    def test_normalized_power_exceeds_mean_for_variable_efforts(self):
        steady = [(T0 + i, 200.0) for i in range(600)]
        surges = [(T0 + i, 400.0 if (i // 30) % 2 else 0.0) for i in range(600)]
        assert load.normalized_power(steady) == pytest.approx(200.0, abs=1.0)
        assert load.normalized_power(surges) > 200.0

    def test_intensity_factor_and_stress(self):
        assert load.intensity_factor(250.0, 250.0) == pytest.approx(1.0)
        assert load.intensity_factor(250.0, 0.0) is None
        # One hour exactly at threshold is the definition of 100.
        assert load.training_stress(3600.0, 250.0, 250.0) == pytest.approx(100.0)

    def test_aerobic_decoupling_detects_drift(self):
        power = [(T0 + i, 200.0) for i in range(3600)]
        hr = [(T0 + i, 140.0 if i < 1800 else 160.0) for i in range(3600)]
        drift = load.aerobic_decoupling(power, hr)
        assert drift is not None and drift > 10.0

    def test_aerobic_decoupling_needs_ten_minutes(self):
        power = [(T0 + i, 200.0) for i in range(120)]
        hr = [(T0 + i, 140.0) for i in range(120)]
        assert load.aerobic_decoupling(power, hr) is None

    def test_ewma_load_ratio(self):
        balance = load.ewma_load([50.0] * 60)
        assert balance.ratio == pytest.approx(1.0, abs=0.05)
        spike = load.ewma_load([50.0] * 50 + [200.0] * 7)
        assert spike.ratio > 1.3

    def test_ewma_load_flags_short_history(self):
        balance = load.ewma_load([50.0] * 10)
        assert any("not yet fully established" in n for n in balance.notes)

    def test_ewma_load_handles_no_history(self):
        balance = load.ewma_load([])
        assert balance.ratio is None


class TestFusion:
    def _two_heart_rates(self, offset=0.0):
        strap = make_stream(
            Modality.HEART_RATE,
            [(T0 + i, 150.0) for i in range(120)],
            strap_provenance("strap"),
        )
        watch = Stream(
            modality=Modality.HEART_RATE,
            provenance=Provenance(
                source_id="watch",
                kind=SourceKind.VENDOR_CLOUD,
                device="wrist optical",
                latency_s=300.0,
                algorithm="vendor PPG pipeline",
                documented=False,
                nominal_hz=1.0,
            ),
            samples=[
                Sample(
                    t=T0 + i,
                    value=150.0 + offset,
                    evidence=Evidence.VENDOR_DERIVED,
                    confidence=0.6,
                )
                for i in range(120)
            ],
        )
        return strap, watch

    def test_measurement_beats_vendor_derivation(self):
        strap, watch = self._two_heart_rates()
        result = fusion.reconcile([watch, strap])
        assert result.chosen.provenance.source_id == "strap"
        assert "measured" in result.chosen_reason

    def test_disagreement_is_reported_not_hidden(self):
        strap, watch = self._two_heart_rates(offset=25.0)
        result = fusion.reconcile([strap, watch])
        assert result.conflicts
        assert "disagree on heart_rate" in result.conflicts[0]
        assert result.agreements[0].bias == pytest.approx(-25.0)

    def test_agreeing_sources_produce_no_conflict(self):
        strap, watch = self._two_heart_rates(offset=1.0)
        result = fusion.reconcile([strap, watch])
        assert result.conflicts == []
        assert result.agreements[0].max_abs_difference == pytest.approx(1.0)

    def test_non_overlapping_sources_are_called_out(self):
        strap, watch = self._two_heart_rates()
        watch.samples = [Sample(t=s.t + 10_000, value=s.value) for s in watch.samples]
        result = fusion.reconcile([strap, watch])
        assert any("no overlapping samples" in c for c in result.conflicts)

    def test_compare_rejects_mismatched_modalities(self):
        strap, _ = self._two_heart_rates()
        power = make_stream(
            Modality.POWER, [(T0 + i, 200.0) for i in range(10)], strap_provenance("pm")
        )
        with pytest.raises(ValueError):
            fusion.compare(strap, power)

    def test_reconcile_requires_a_single_modality(self):
        strap, _ = self._two_heart_rates()
        power = make_stream(
            Modality.POWER, [(T0 + i, 200.0) for i in range(10)], strap_provenance("pm")
        )
        with pytest.raises(ValueError):
            fusion.reconcile([strap, power])
        with pytest.raises(ValueError):
            fusion.reconcile([])

    def test_single_source_still_reports_quality(self):
        strap, _ = self._two_heart_rates()
        result = fusion.reconcile([strap])
        assert result.chosen is strap
        assert "strap" in result.quality
        assert result.agreements == []
