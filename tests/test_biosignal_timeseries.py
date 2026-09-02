"""Tests for the deterministic statistics over stored reports."""

import pytest

from raganything.biosignal import timeseries as ts
from raganything.biosignal.narrative import analyze_session
from raganything.biosignal.schema import (
    Modality,
    Provenance,
    Session,
    SourceKind,
    make_stream,
)
from raganything.biosignal.store import ReportRecord

T0 = 1_700_000_000.0
DAY = 86400.0


def record(
    session_id,
    day,
    *,
    metrics=None,
    withheld=None,
    quality=1.0,
    modalities=("rr_interval", "heart_rate", "power"),
    hrv_confidence=1.0,
    labels=None,
):
    return ReportRecord(
        session_id=session_id,
        subject_id="self",
        start=T0 + day * DAY,
        end=T0 + day * DAY + 3600,
        duration_s=3600.0,
        labels=dict(labels or {}),
        metrics=dict(metrics or {}),
        withheld=dict(withheld or {}),
        modality_quality={m: quality for m in modalities},
        hrv_confidence=hrv_confidence,
    )


def linear_records(n=10, slope=-1.0, intercept=50.0, step_days=3, noise=None):
    out = []
    for i in range(n):
        value = intercept + slope * (i * step_days)
        if noise:
            value += noise[i % len(noise)]
        out.append(record(f"s{i}", i * step_days, metrics={"hrv_rmssd": value}))
    return out


class TestMetricTables:
    def test_every_emitted_metric_is_described(self):
        """Guard: analyze_session must never emit a metric the tables miss."""
        prov = Provenance(
            source_id="s",
            kind=SourceKind.BLE,
            device="d",
            latency_s=0.0,
            nominal_hz=1.0,
        )
        n = 1800
        session = Session(
            "cover",
            T0,
            T0 + n,
            streams=[
                make_stream(
                    Modality.RR_INTERVAL,
                    [
                        (T0 + i * 0.5, 500.0 + (8 if i % 2 else -8))
                        for i in range(n * 2)
                    ],
                    prov,
                ),
                make_stream(
                    Modality.HEART_RATE, [(T0 + i, 150.0) for i in range(n)], prov
                ),
                make_stream(
                    Modality.POWER,
                    [(T0 + i, 220.0 + (10 if i % 2 else -10)) for i in range(n)],
                    prov,
                ),
            ],
        )
        report = analyze_session(session, rest_hr=48, max_hr=188, threshold_power=250.0)
        emitted = set(report.metrics) | set(report.withheld)
        for name in emitted:
            assert name in ts.METRIC_SUPPORT, f"{name} missing from METRIC_SUPPORT"
            assert name in ts.METRIC_UNITS, f"{name} missing from METRIC_UNITS"
            assert name in ts.STATISTIC_VALIDITY, f"{name} missing from validity"

    def test_unknown_metric_raises_rather_than_guessing(self):
        with pytest.raises(ts.UnknownMetricError, match="recovery_score"):
            ts.collect([], "recovery_score")


class TestCollect:
    def test_gathers_usable_values(self):
        series = ts.collect(linear_records(5), "hrv_rmssd")
        assert len(series) == 5
        assert series.unit == "ms"
        assert series.excluded == {}

    def test_withheld_session_is_excluded_with_its_stored_reason(self):
        records = [
            record("good", 0, metrics={"hrv_rmssd": 40.0}),
            record(
                "bad", 1, withheld={"hrv_rmssd": "strap lost contact for 9 minutes"}
            ),
        ]
        series = ts.collect(records, "hrv_rmssd")
        assert len(series) == 1
        # The reason must survive verbatim -- it is what the user is told.
        assert series.excluded["bad"] == "strap lost contact for 9 minutes"

    def test_withheld_wins_even_when_a_value_is_present(self):
        """The gate-bypass regression.

        A metric can appear in both dicts (the HRV block still holds values the
        quality gate rejected). Reading past the withholding would silently
        undo it.
        """
        leaky = record(
            "leaky", 0, metrics={"hrv_rmssd": 99.0}, withheld={"hrv_rmssd": "rejected"}
        )
        series = ts.collect([leaky], "hrv_rmssd")
        assert len(series) == 0
        assert series.excluded["leaky"] == "rejected"

    def test_quality_gate_excludes_with_a_reason(self):
        records = [
            record("clean", 0, metrics={"hrv_rmssd": 40.0}, quality=0.9),
            record("noisy", 1, metrics={"hrv_rmssd": 41.0}, quality=0.2),
        ]
        series = ts.collect(records, "hrv_rmssd", min_quality=0.5)
        assert [s for s in series.session_ids] == ["clean"]
        assert "below the 0.50 threshold" in series.excluded["noisy"]

    def test_query_time_gate_can_only_remove(self):
        records = linear_records(6)
        lenient = ts.collect(records, "hrv_rmssd", min_quality=0.0)
        strict = ts.collect(records, "hrv_rmssd", min_quality=0.99)
        assert len(strict) <= len(lenient)

    def test_gate_cannot_un_withhold(self):
        withheld = [record("w", 0, withheld={"hrv_rmssd": "nope"})]
        assert len(ts.collect(withheld, "hrv_rmssd", min_quality=0.0)) == 0

    def test_unverifiable_quality_excludes_rather_than_admits(self):
        blind = record("blind", 0, metrics={"hrv_rmssd": 40.0}, modalities=())
        series = ts.collect([blind], "hrv_rmssd")
        assert len(series) == 0
        assert "cannot verify" in series.excluded["blind"]

    def test_hrv_confidence_gate(self):
        records = [record("low", 0, metrics={"hrv_rmssd": 40.0}, hrv_confidence=0.1)]
        series = ts.collect(records, "hrv_rmssd", min_hrv_confidence=0.5)
        assert len(series) == 0
        assert "HRV confidence" in series.excluded["low"]

    def test_out_of_window_is_skipped_silently_not_excluded(self):
        records = linear_records(5)
        series = ts.collect(records, "hrv_rmssd", start=T0, end=T0 + 4 * DAY)
        assert len(series) == 2
        assert series.excluded == {}

    def test_missing_metric_is_recorded(self):
        series = ts.collect([record("none", 0)], "hrv_rmssd")
        assert series.excluded["none"] == "metric not computed for this session"


class TestAggregate:
    def test_mean_and_median(self):
        records = [
            record(f"s{i}", i, metrics={"hrv_rmssd": v})
            for i, v in enumerate([10.0, 20.0, 60.0])
        ]
        series = ts.collect(records, "hrv_rmssd")
        assert ts.aggregate(series, "mean").value == pytest.approx(30.0)
        assert ts.aggregate(series, "median").value == pytest.approx(20.0)
        assert ts.aggregate(series, "min").value == 10.0
        assert ts.aggregate(series, "max").value == 60.0

    def test_count_always_available(self):
        series = ts.collect(linear_records(4), "hrv_rmssd")
        result = ts.aggregate(series, "count")
        assert result.value == 4.0
        assert result.unit == "sessions"

    def test_sum_refused_for_an_intensive_metric(self):
        series = ts.collect(linear_records(4), "hrv_rmssd")
        result = ts.aggregate(series, "sum")
        assert result.value is None
        assert "intensive" in result.withheld_reason

    def test_sum_allowed_for_an_extensive_metric(self):
        records = [record(f"s{i}", i, metrics={"trimp": 50.0}) for i in range(4)]
        series = ts.collect(records, "trimp")
        assert ts.aggregate(series, "sum").value == pytest.approx(200.0)

    def test_stdev_refused_below_three_points(self):
        series = ts.collect(linear_records(2), "hrv_rmssd")
        assert "not meaningful" in ts.aggregate(series, "stdev").withheld_reason

    def test_empty_series_refuses(self):
        result = ts.aggregate(ts.collect([], "hrv_rmssd"), "mean")
        assert result.value is None
        assert "no session" in result.withheld_reason

    def test_unknown_statistic_refuses(self):
        series = ts.collect(linear_records(4), "hrv_rmssd")
        assert "unknown statistic" in ts.aggregate(series, "vibe").withheld_reason

    def test_denominator_is_always_visible(self):
        records = linear_records(3) + [
            record("x", 99, withheld={"hrv_rmssd": "bad night"})
        ]
        result = ts.aggregate(ts.collect(records, "hrv_rmssd"), "mean")
        assert result.n == 3
        assert result.n_excluded == 1
        assert "1 excluded" in result.note

    def test_mostly_excluded_is_flagged_not_hidden(self):
        records = [record("ok", 0, metrics={"hrv_rmssd": 40.0})] + [
            record(f"bad{i}", i + 1, withheld={"hrv_rmssd": "noise"}) for i in range(5)
        ]
        result = ts.aggregate(ts.collect(records, "hrv_rmssd"), "mean")
        assert result.value == 40.0
        assert result.representative is False
        assert "caution" in result.note


class TestTrend:
    def test_slope_matches_hand_computation(self):
        # values 50, 47, 44 ... at 3-day spacing -> exactly -1.0 per day
        result = ts.trend(ts.collect(linear_records(10), "hrv_rmssd"))
        assert result.slope_per_day == pytest.approx(-1.0)
        assert result.intercept == pytest.approx(50.0)
        assert result.r2 == pytest.approx(1.0)
        assert result.direction == "falling"

    def test_rising_series(self):
        result = ts.trend(ts.collect(linear_records(10, slope=+0.5), "hrv_rmssd"))
        assert result.direction == "rising"
        assert result.slope_per_day == pytest.approx(0.5)

    def test_flat_when_the_interval_spans_zero(self):
        noisy = [
            record(f"s{i}", i * 3, metrics={"hrv_rmssd": v})
            for i, v in enumerate([40, 55, 38, 60, 35, 58, 42, 57])
        ]
        result = ts.trend(ts.collect(noisy, "hrv_rmssd"))
        assert result.direction == "flat"
        assert result.ci95_slope[0] < 0 < result.ci95_slope[1]

    def test_confidence_interval_brackets_the_slope(self):
        result = ts.trend(
            ts.collect(linear_records(8, noise=[0, 1.5, -1.0, 0.5]), "hrv_rmssd")
        )
        low, high = result.ci95_slope
        assert low <= result.slope_per_day <= high

    def test_single_outlier_does_not_manufacture_a_trend(self):
        """A flat history with one wild session must not read as a trend.

        The confidence interval is the primary guard here: one outlier inflates
        the residual variance enough that the interval spans zero. The
        Theil-Sen check below is the backstop for the rarer case where it does
        not.
        """
        values = [40.0] * 9 + [400.0]
        records = [
            record(f"s{i}", i * 3, metrics={"hrv_rmssd": v})
            for i, v in enumerate(values)
        ]
        result = ts.trend(ts.collect(records, "hrv_rmssd"))
        assert result.direction == "flat"
        assert result.ci95_slope[0] < 0 < result.ci95_slope[1]

    def test_theil_sen_is_robust_to_an_outlier(self):
        # Nine flat points and one extreme: least squares is dragged upward,
        # the median pairwise slope is not.
        xs = [float(i) for i in range(10)]
        ys = [40.0] * 9 + [400.0]
        assert ts._theil_sen(xs, ys) == pytest.approx(0.0)

    def test_theil_sen_tracks_a_genuine_slope(self):
        xs = [float(i) for i in range(10)]
        ys = [50.0 - 2.0 * x for x in xs]
        assert ts._theil_sen(xs, ys) == pytest.approx(-2.0)

    def test_sign_disagreement_downgrades_to_flat(self):
        """The backstop itself, exercised directly.

        Constructing data where a tight interval and the robust estimator
        genuinely disagree is difficult -- which is a good sign -- so the guard
        is verified on the branch rather than through a contrived series.
        """
        series = ts.collect(linear_records(10, slope=+0.5), "hrv_rmssd")
        rising = ts.trend(series)
        assert rising.direction == "rising"
        assert rising.theil_sen_slope > 0

        original = ts._theil_sen
        try:
            ts._theil_sen = lambda xs, ys: -1.0  # pretend the robust fit disagrees
            downgraded = ts.trend(series)
        finally:
            ts._theil_sen = original
        assert downgraded.direction == "flat"
        assert any("Theil-Sen" in n for n in downgraded.notes)

    def test_refuses_below_minimum_points(self):
        result = ts.trend(ts.collect(linear_records(3), "hrv_rmssd"), min_points=5)
        assert result.direction == "undetermined"
        assert "at least 5 usable sessions" in result.withheld_reason

    def test_refuses_below_minimum_span(self):
        records = [
            record(f"s{i}", 0, metrics={"hrv_rmssd": 40.0 + i}) for i in range(6)
        ]
        # All on the same instant -> zero span.
        result = ts.trend(ts.collect(records, "hrv_rmssd"))
        assert result.direction == "undetermined"

    def test_exact_fit_is_flagged_as_suspicious(self):
        result = ts.trend(ts.collect(linear_records(10), "hrv_rmssd"))
        assert any("exact" in note for note in result.notes)

    def test_excluded_sessions_are_carried_into_the_trend(self):
        records = linear_records(8) + [
            record("bad", 30, withheld={"hrv_rmssd": "lost contact"})
        ]
        result = ts.trend(ts.collect(records, "hrv_rmssd"))
        assert result.excluded["bad"] == "lost contact"
        assert any("excluded" in note for note in result.notes)


class TestComparisonAndGrouping:
    def test_compare_windows_detects_a_real_difference(self):
        records = [
            record(f"a{i}", i, metrics={"mean_power": 200.0 + i}) for i in range(5)
        ] + [
            record(f"b{i}", 40 + i, metrics={"mean_power": 300.0 + i}) for i in range(5)
        ]
        result = ts.compare_windows(
            records,
            "mean_power",
            (T0, T0 + 10 * DAY),
            (T0 + 39 * DAY, T0 + 50 * DAY),
        )
        assert result.direction == "lower"
        assert result.difference == pytest.approx(-100.0)

    def test_compare_windows_calls_a_small_gap_indistinguishable(self):
        records = [
            record(f"a{i}", i, metrics={"mean_power": 200.0 + (i % 3) * 10})
            for i in range(6)
        ] + [
            record(f"b{i}", 40 + i, metrics={"mean_power": 202.0 + (i % 3) * 10})
            for i in range(6)
        ]
        result = ts.compare_windows(
            records, "mean_power", (T0, T0 + 10 * DAY), (T0 + 39 * DAY, T0 + 50 * DAY)
        )
        assert result.direction == "indistinguishable"

    def test_group_by_label(self):
        records = [
            record("r1", 0, metrics={"mean_power": 200.0}, labels={"sport": "road"}),
            record("r2", 1, metrics={"mean_power": 220.0}, labels={"sport": "road"}),
            record("t1", 2, metrics={"mean_power": 180.0}, labels={"sport": "turbo"}),
        ]
        grouped = ts.group_by_label(records, "mean_power", "sport")
        assert grouped["road"].value == pytest.approx(210.0)
        assert grouped["turbo"].value == pytest.approx(180.0)

    def test_after_event(self):
        records = []
        for i in range(12):
            hard = i % 6 == 0
            records.append(
                record(
                    f"s{i}",
                    i * 2,
                    metrics={"trimp": 300.0 if hard else 80.0},
                    labels={"hard": hard},
                )
            )
        result = ts.after_event(
            records, "trimp", lambda r: bool(r.labels.get("hard")), lag_days=(1, 5)
        )
        assert result.a.n > 0
        assert result.metric == "trimp"


class TestDailyLoadSeries:
    def test_rest_days_are_zeros(self):
        records = [
            record("a", 0, metrics={"trimp": 100.0}),
            record("b", 3, metrics={"trimp": 50.0}),
        ]
        series = ts.daily_load_series(records, metric="trimp")
        assert series == [100.0, 0.0, 0.0, 50.0]

    def test_withheld_day_counts_as_rest_not_as_a_gap(self):
        records = [
            record("a", 0, metrics={"trimp": 100.0}),
            record("b", 1, withheld={"trimp": "heart rate dropped out"}),
            record("c", 2, metrics={"trimp": 60.0}),
        ]
        assert ts.daily_load_series(records, metric="trimp") == [100.0, 0.0, 60.0]

    def test_two_sessions_on_one_day_are_summed(self):
        records = [
            record("a", 0, metrics={"trimp": 40.0}),
            record("b", 0, metrics={"trimp": 60.0}),
        ]
        assert ts.daily_load_series(records, metric="trimp") == [100.0]

    def test_empty(self):
        assert ts.daily_load_series([], metric="trimp") == []

    def test_feeds_ewma_load(self):
        from raganything.biosignal.analytics import load

        records = [record(f"s{i}", i, metrics={"trimp": 50.0}) for i in range(30)]
        balance = load.ewma_load(ts.daily_load_series(records, metric="trimp"))
        assert balance.ratio == pytest.approx(1.0, abs=0.05)
