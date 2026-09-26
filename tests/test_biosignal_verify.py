"""Tests for answer verification -- the honesty guarantee."""

import pytest

from raganything.biosignal.router import classify_rules
from raganything.biosignal.store import ReportRecord
from raganything.biosignal.verify import (
    DEFAULT_POLICY,
    VerificationPolicy,
    ViolationKind,
    annotate,
    extract_claims,
    refusal,
    verify,
    worst_action,
)

T0 = 1_700_000_000.0
DAY = 86400.0


def record(session_id, day, *, metrics=None, withheld=None, quality=1.0):
    return ReportRecord(
        session_id=session_id,
        subject_id="self",
        start=T0 + day * DAY,
        end=T0 + day * DAY + 3600,
        duration_s=3600.0,
        metrics=dict(metrics or {}),
        withheld=dict(withheld or {}),
        modality_quality={
            "rr_interval": quality,
            "heart_rate": quality,
            "power": quality,
        },
        hrv_confidence=1.0,
    )


def plan_for(question="what was my average rmssd?", window=None):
    base = classify_rules(question, now=T0 + 60 * DAY)
    if window is None:
        return base
    from dataclasses import replace

    return replace(base, window=window)


class TestExtraction:
    def test_canonical_form(self):
        claims = extract_claims("Your hrv_rmssd = 42.1 ms across the window.")
        assert claims[0].metric == "hrv_rmssd"
        assert claims[0].value == pytest.approx(42.1)

    def test_natural_phrasing(self):
        claims = extract_claims("Your average heart rate was 148 bpm.")
        assert any(c.metric == "mean_hr" and c.value == 148.0 for c in claims)

    def test_alias_fans_out_to_both_hrv_metrics(self):
        metrics = {c.metric for c in extract_claims("Your HRV was 45 ms.")}
        assert metrics == {"hrv_rmssd", "hrv_sdnn"}

    def test_direction_without_a_number_is_a_trend_claim(self):
        claims = extract_claims("Your rmssd is declining.")
        assert claims[0].kind == "trend"
        assert claims[0].direction == "falling"

    def test_a_number_and_a_direction_yield_both_claims(self):
        """Regression: a sentence stating both must still be trend-checked."""
        kinds = {c.kind for c in extract_claims("rmssd = 55 ms and it is rising.")}
        assert "trend" in kinds
        assert "point" in kinds

    def test_claims_do_not_bind_across_sentences(self):
        claims = extract_claims("Your rmssd looks fine. Your power was 250 W.")
        rmssd = [c for c in claims if c.metric == "hrv_rmssd" and c.value is not None]
        assert rmssd == []

    def test_session_and_date_are_captured(self):
        claims = extract_claims("On 2026-07-14 your mean_hr = 150 bpm.")
        assert claims[0].date == "2026-07-14"

    def test_spelled_numbers(self):
        claims = extract_claims("Your rmssd was around forty.")
        assert any(c.value == 40.0 for c in claims)

    def test_unsupported_metric_is_flagged(self):
        claims = extract_claims("Your recovery score was 62.")
        assert claims[0].kind == "unsupported"

    def test_prose_without_claims_yields_nothing(self):
        assert extract_claims("You slept well and felt good.") == []


class TestWithheldCheck:
    def test_asserting_a_withheld_metric_is_caught(self):
        records = [record("s1", 0, withheld={"hrv_rmssd": "strap lost contact"})]
        verdict = verify("Your hrv_rmssd = 42.0 ms.", plan_for(), records)

        assert not verdict.ok
        kinds = [v.kind for v in verdict.violations]
        assert ViolationKind.WITHHELD_METRIC_ASSERTED in kinds

    def test_the_stored_reason_is_echoed_verbatim(self):
        reason = "withheld: RR stream quality 0.31 below 0.50 threshold"
        records = [record("s1", 0, withheld={"hrv_rmssd": reason})]
        verdict = verify("Your hrv_rmssd = 42.0 ms.", plan_for(), records)
        violation = next(
            v
            for v in verdict.violations
            if v.kind is ViolationKind.WITHHELD_METRIC_ASSERTED
        )
        assert violation.expected == reason

    def test_mixed_sessions_are_not_a_violation(self):
        records = [
            record("good", 0, metrics={"hrv_rmssd": 42.0}),
            record("bad", 1, withheld={"hrv_rmssd": "noise"}),
        ]
        verdict = verify("Your hrv_rmssd = 42.0 ms.", plan_for(), records)
        assert ViolationKind.WITHHELD_METRIC_ASSERTED not in [
            v.kind for v in verdict.violations
        ]

    def test_saying_it_was_withheld_is_not_a_violation(self):
        records = [record("s1", 0, withheld={"hrv_rmssd": "strap lost contact"})]
        verdict = verify(
            "Your RMSSD was withheld because the strap lost contact.",
            plan_for(),
            records,
        )
        assert verdict.ok


class TestTrendCheck:
    def _falling(self):
        return [
            record(f"s{i}", i * 3, metrics={"hrv_rmssd": 50.0 - i}) for i in range(10)
        ]

    def test_opposite_direction_is_caught(self):
        verdict = verify("Your rmssd is rising.", plan_for(), self._falling())
        assert ViolationKind.UNGATED_TREND_ASSERTED in [
            v.kind for v in verdict.violations
        ]

    def test_correct_direction_passes(self):
        verdict = verify("Your rmssd is declining.", plan_for(), self._falling())
        assert verdict.ok

    def test_trend_asserted_on_too_few_sessions_is_caught(self):
        records = [record("s0", 0, metrics={"hrv_rmssd": 50.0})]
        verdict = verify("Your rmssd is declining.", plan_for(), records)
        violation = next(
            v
            for v in verdict.violations
            if v.kind is ViolationKind.UNGATED_TREND_ASSERTED
        )
        assert "at least" in violation.expected

    def test_a_trend_is_checked_against_the_window_not_one_cited_date(self):
        """Regression: binding a trend to a cited date made the check unreachable."""
        verdict = verify(
            "On 2020-01-01 the picture changed and your rmssd is rising.",
            plan_for(),
            self._falling(),
        )
        assert ViolationKind.UNGATED_TREND_ASSERTED in [
            v.kind for v in verdict.violations
        ]


class TestValueCheck:
    def test_value_within_tolerance_passes(self):
        records = [record("s1", 0, metrics={"mean_hr": 148.4})]
        assert verify("Your mean_hr = 148.0 bpm.", plan_for(), records).ok

    def test_value_outside_tolerance_is_caught(self):
        records = [record("s1", 0, metrics={"mean_hr": 148.0})]
        verdict = verify("Your mean_hr = 172.0 bpm.", plan_for(), records)
        assert ViolationKind.VALUE_CONTRADICTS_COMPUTATION in [
            v.kind for v in verdict.violations
        ]

    def test_a_mean_across_sessions_is_accepted(self):
        records = [
            record("s1", 0, metrics={"mean_hr": 140.0}),
            record("s2", 1, metrics={"mean_hr": 160.0}),
        ]
        assert verify("Your average mean_hr = 150.0 bpm.", plan_for(), records).ok


class TestScopeAndUnsupported:
    def test_out_of_scope_date_is_reported(self):
        records = [record("s1", 0, metrics={"mean_hr": 148.0})]
        verdict = verify("On 2019-01-01 your mean_hr = 148 bpm.", plan_for(), records)
        assert ViolationKind.OUT_OF_SCOPE_SESSION_CITED in [
            v.kind for v in verdict.violations
        ]

    def test_unsupported_metric_name(self):
        records = [record("s1", 0, metrics={"mean_hr": 148.0})]
        verdict = verify("Your body battery was 62.", plan_for(), records)
        violation = next(
            v
            for v in verdict.violations
            if v.kind is ViolationKind.UNSUPPORTED_METRIC_NAME
        )
        assert "does not compute" in violation.expected

    def test_unverifiable_when_no_records_are_in_scope(self):
        verdict = verify("Your mean_hr = 148 bpm.", plan_for(), [])
        assert ViolationKind.UNVERIFIABLE in [v.kind for v in verdict.violations]


class TestPolicy:
    def _violation(self, kind):
        records = [record("s1", 0, withheld={"hrv_rmssd": "noise"})]
        verdict = verify("Your hrv_rmssd = 42.0 ms.", plan_for(), records)
        return verdict

    def test_severity_ordering(self):
        verdict = self._violation(ViolationKind.WITHHELD_METRIC_ASSERTED)
        assert worst_action(verdict.violations) is VerificationPolicy.REGENERATE

    def test_default_policy_covers_every_kind(self):
        for kind in ViolationKind:
            assert kind in DEFAULT_POLICY

    def test_annotate_appends_and_never_edits(self):
        verdict = self._violation(ViolationKind.WITHHELD_METRIC_ASSERTED)
        original = "Your hrv_rmssd = 42.0 ms."
        annotated = annotate(original, verdict.violations)
        # The model's own sentence must survive untouched: rewriting prose is
        # itself a way to manufacture a claim it never made.
        assert annotated.startswith(original)
        assert "Automated data check" in annotated

    def test_annotate_is_a_noop_without_violations(self):
        assert annotate("fine", []) == "fine"

    def test_refusal_reports_what_the_data_does_support(self):
        verdict = self._violation(ViolationKind.WITHHELD_METRIC_ASSERTED)
        text = refusal(verdict.violations, facts="mean_hr (mean) = 148.00 bpm")
        assert "can't answer" in text
        assert "mean_hr" in text

    def test_clean_answer_has_no_violations(self):
        records = [record("s1", 0, metrics={"mean_hr": 148.0})]
        verdict = verify("Your mean_hr = 148.0 bpm.", plan_for(), records)
        assert verdict.ok
        assert verdict.violations == ()
        assert worst_action(verdict.violations) is None
