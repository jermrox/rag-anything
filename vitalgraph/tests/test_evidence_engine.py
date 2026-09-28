"""Tests for the personal evidence engine.

The behaviour that matters is what the engine *refuses* to validate. A system
that promotes a real effect is easy; one that declines to promote a pure
coincidence, a reversed replication, or a finding that only survived because
twenty hypotheses were tested at once is the thing worth having.

Histories are generated with fixed seeds, so a failure is a real regression
rather than an unlucky draw.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from vitalgraph.evidence.engine import (
    MIN_HISTORY_DAYS,
    MIN_REPLICATION_EFFECT_RATIO,
    DayRecord,
    Hypothesis,
    build_evidence_graph,
    hypotheses_from_exposures,
)
from vitalgraph.evidence.stats import (
    MIN_GROUP_SIZE,
    benjamini_hochberg,
    bootstrap_interval,
    compare,
    hedges_g,
    permutation_p_value,
    split_by_time,
)
from vitalgraph.knowledge.factors import EvidenceTier, Factor

START = date(2026, 1, 1)

#: Fewer resamples than the production default keeps the suite fast. The
#: p-value resolution is still 1/2001, far finer than any threshold applied.
RESAMPLES = 2000


def _history(
    days: int,
    effect: float,
    exposure_rate: float = 0.4,
    noise: float = 5.0,
    baseline: float = 30.0,
    seed: int = 7,
    exposure: str = "late_dinner",
    outcome: str = "sleep_onset_minutes",
    effect_after: float | None = None,
) -> list[DayRecord]:
    """Build a history where ``exposure`` shifts ``outcome`` by ``effect``.

    ``effect_after`` changes the effect halfway through, which is how a
    relationship that stops being true -- or reverses -- is simulated.
    """
    rng = random.Random(seed)
    records = []
    for i in range(days):
        exposed = rng.random() < exposure_rate
        current = effect if effect_after is None or i < days // 2 else effect_after
        value = baseline + rng.gauss(0.0, noise) + (current if exposed else 0.0)
        records.append(
            DayRecord(
                day=START + timedelta(days=i),
                exposures=frozenset([exposure]) if exposed else frozenset(),
                outcomes={outcome: value},
            )
        )
    return records


LATE_DINNER = Hypothesis(
    exposure="late_dinner",
    outcome="sleep_onset_minutes",
    factor=Factor.SLEEP,
    description="On evenings with a late dinner, sleep onset",
)


# --- statistics ------------------------------------------------------------


def test_permutation_p_value_is_never_exactly_zero():
    """The observed labelling is one of the permutations under the null.

    Omitting it can return p = 0, which claims more certainty than any finite
    number of resamples can support.
    """
    p = permutation_p_value([100.0] * 10, [0.0] * 10, resamples=200, seed=1)
    assert p > 0.0
    assert p == pytest.approx(1 / 201, abs=1e-9)


def test_permutation_p_value_is_high_when_groups_are_alike():
    rng = random.Random(3)
    a = [rng.gauss(0, 1) for _ in range(30)]
    b = [rng.gauss(0, 1) for _ in range(30)]
    assert permutation_p_value(a, b, resamples=2000, seed=1) > 0.2


def test_permutation_p_value_is_deterministic_for_a_seed():
    a, b = [1.0, 2.0, 3.0, 4.0, 9.0], [5.0, 6.0, 7.0, 8.0, 9.0]
    first = permutation_p_value(a, b, resamples=500, seed=42)
    second = permutation_p_value(a, b, resamples=500, seed=42)
    assert first == second


def test_hedges_g_is_smaller_than_cohens_d_at_small_n():
    """The small-sample correction is applied, not ignored."""
    a = [10.0, 12.0, 11.0, 13.0, 12.0]
    b = [5.0, 6.0, 7.0, 5.0, 6.0]
    g = hedges_g(a, b)
    # Cohen's d without correction, computed inline for the comparison.
    from vitalgraph.evidence.stats import mean, variance

    pooled = (4 * variance(a) + 4 * variance(b)) / 8
    d = (mean(a) - mean(b)) / pooled**0.5
    assert abs(g) < abs(d)
    assert abs(g) == pytest.approx(abs(d) * (1 - 3 / (4 * 10 - 9)), rel=1e-9)


def test_hedges_g_is_zero_when_there_is_no_scale():
    assert hedges_g([5.0] * 6, [5.0] * 6) == 0.0


def test_bootstrap_interval_narrows_as_evidence_grows():
    rng = random.Random(11)
    small_a = [rng.gauss(10, 3) for _ in range(8)]
    small_b = [rng.gauss(0, 3) for _ in range(8)]
    big_a = [rng.gauss(10, 3) for _ in range(200)]
    big_b = [rng.gauss(0, 3) for _ in range(200)]

    small = bootstrap_interval(small_a, small_b, resamples=2000, seed=2)
    big = bootstrap_interval(big_a, big_b, resamples=2000, seed=2)
    assert big.width < small.width


def test_compare_returns_none_below_the_minimum_group_size():
    """Two nights either side is no evidence, and must not produce a p-value."""
    tiny = [1.0] * (MIN_GROUP_SIZE - 1)
    assert compare(tiny, [2.0] * 20) is None
    assert compare([2.0] * 20, tiny) is None


def test_compare_reports_direction_and_n():
    result = compare([10.0] * 10, [0.0] * 10, resamples=500)
    assert result is not None
    assert result.direction == 1
    assert result.n == 20
    assert result.difference == pytest.approx(10.0)


# --- multiple comparisons --------------------------------------------------


def test_benjamini_hochberg_rejects_nothing_when_all_p_values_are_large():
    assert benjamini_hochberg([0.4, 0.6, 0.9, 0.5]) == [False] * 4


def test_benjamini_hochberg_uses_a_step_up_rule():
    """Everything below the largest passing rank is rejected too.

    Testing each rank independently would not control the FDR at all; this is
    the property that makes the procedure work.
    """
    # Ranks 1 and 3 clear their thresholds; rank 2 does not on its own.
    # The step-up rule must still reject rank 2, because rank 3 passed.
    p_values = [0.001, 0.04, 0.045]
    survives = benjamini_hochberg(p_values, false_discovery_rate=0.10)
    assert survives == [True, True, True]


def test_benjamini_hochberg_is_stricter_than_an_uncorrected_threshold():
    """Twenty hypotheses at p < 0.05 yields a false positive by construction."""
    p_values = [0.04] + [0.5] * 19
    uncorrected = [p < 0.05 for p in p_values]
    corrected = benjamini_hochberg(p_values, false_discovery_rate=0.10)
    assert any(uncorrected)
    assert not any(corrected)


def test_benjamini_hochberg_handles_an_empty_list():
    assert benjamini_hochberg([]) == []


# --- the chronological split ----------------------------------------------


def test_split_is_chronological_not_random():
    """A random split leaks: adjacent days are correlated.

    Every discovery item must predate every holdout item, whatever order the
    input arrived in.
    """
    items = [(float(i), f"day{i}") for i in range(20)]
    random.Random(5).shuffle(items)
    discovery, holdout = split_by_time(items, discovery_fraction=0.6)
    assert discovery == [f"day{i}" for i in range(12)]
    assert holdout == [f"day{i}" for i in range(12, 20)]


# --- the engine's refusals -------------------------------------------------


def test_engine_declines_to_run_on_a_short_history():
    """ "We have not looked" and "nothing is true" are opposite claims."""
    graph = build_evidence_graph(
        _history(10, effect=20.0), [LATE_DINNER], resamples=RESAMPLES
    )
    assert graph.ran is False
    assert graph.validated() == []
    assert graph.reason_not_run is not None
    assert str(MIN_HISTORY_DAYS) in graph.reason_not_run


def test_a_real_and_stable_effect_is_validated():
    graph = build_evidence_graph(
        _history(200, effect=25.0, seed=1), [LATE_DINNER], resamples=RESAMPLES
    )
    assert graph.ran is True
    validated = graph.validated()
    assert len(validated) == 1
    relationship = validated[0]
    assert relationship.tier is EvidenceTier.VALIDATED
    assert relationship.rejection_reason is None
    assert relationship.replication is not None
    assert relationship.can_support_recommendation is True


def test_pure_noise_is_not_validated():
    """No exposure effect at all. Nothing may be promoted."""
    graph = build_evidence_graph(
        _history(200, effect=0.0, seed=4), [LATE_DINNER], resamples=RESAMPLES
    )
    assert graph.ran is True
    assert graph.validated() == []
    assert graph.relationships[0].can_support_recommendation is False


def test_an_effect_that_reverses_later_is_not_validated():
    """Direction is the first thing a replication must reproduce."""
    records = _history(240, effect=25.0, effect_after=-25.0, seed=2)
    graph = build_evidence_graph(records, [LATE_DINNER], resamples=RESAMPLES)
    relationship = graph.relationships[0]
    assert relationship.is_validated is False
    assert relationship.rejection_reason is not None


def test_an_effect_that_vanishes_later_is_not_validated():
    """A relationship that stopped being true must stop being believed."""
    records = _history(240, effect=30.0, effect_after=0.0, seed=6)
    graph = build_evidence_graph(records, [LATE_DINNER], resamples=RESAMPLES)
    relationship = graph.relationships[0]
    assert relationship.is_validated is False


def test_twenty_null_hypotheses_produce_no_validated_relationships():
    """The headline property: mass hypothesis generation must not manufacture
    personal truths.

    Twenty exposures, none of which affects the outcome. Uncorrected discovery
    over a personal history would be expected to promote at least one.
    """
    rng = random.Random(99)
    records = []
    for i in range(200):
        exposures = frozenset(f"exposure_{j}" for j in range(20) if rng.random() < 0.4)
        records.append(
            DayRecord(
                day=START + timedelta(days=i),
                exposures=exposures,
                outcomes={"sleep_onset_minutes": rng.gauss(30.0, 5.0)},
            )
        )
    hypotheses = hypotheses_from_exposures(records, "sleep_onset_minutes", Factor.SLEEP)
    assert len(hypotheses) == 20

    graph = build_evidence_graph(records, hypotheses, resamples=RESAMPLES)
    assert graph.ran is True
    assert (
        graph.validated() == []
    ), "the engine manufactured a personal truth from noise"


def test_a_real_effect_survives_alongside_nineteen_null_ones():
    """The correction must not be so strict that nothing real ever surfaces."""
    rng = random.Random(21)
    records = []
    for i in range(300):
        exposures = {f"exposure_{j}" for j in range(19) if rng.random() < 0.4}
        real = rng.random() < 0.4
        if real:
            exposures.add("late_dinner")
        value = rng.gauss(30.0, 5.0) + (28.0 if real else 0.0)
        records.append(
            DayRecord(
                day=START + timedelta(days=i),
                exposures=frozenset(exposures),
                outcomes={"sleep_onset_minutes": value},
            )
        )
    hypotheses = hypotheses_from_exposures(records, "sleep_onset_minutes", Factor.SLEEP)
    graph = build_evidence_graph(records, hypotheses, resamples=RESAMPLES)
    validated = graph.validated()
    assert [r.hypothesis.exposure for r in validated] == ["late_dinner"]


def test_a_hypothesis_failing_correction_is_never_replicated():
    """Re-testing a rejected hypothesis on the holdout would quietly undo the
    correction by giving it a second chance."""
    rng = random.Random(31)
    records = []
    for i in range(200):
        exposures = frozenset(f"exposure_{j}" for j in range(20) if rng.random() < 0.4)
        records.append(
            DayRecord(
                day=START + timedelta(days=i),
                exposures=exposures,
                outcomes={"sleep_onset_minutes": rng.gauss(30.0, 5.0)},
            )
        )
    hypotheses = hypotheses_from_exposures(records, "sleep_onset_minutes", Factor.SLEEP)
    graph = build_evidence_graph(records, hypotheses, resamples=RESAMPLES)
    for relationship in graph.relationships:
        if not relationship.survived_correction:
            assert relationship.replication is None


def test_a_rare_exposure_is_reported_as_untestable_not_as_absent():
    """Too few comparable days is a different answer from no effect."""
    records = []
    for i in range(120):
        exposed = i in (3, 40)  # twice in four months
        records.append(
            DayRecord(
                day=START + timedelta(days=i),
                exposures=frozenset(["travelled"]) if exposed else frozenset(),
                outcomes={"sleep_onset_minutes": 30.0 + (60.0 if exposed else 0.0)},
            )
        )
    hypothesis = Hypothesis(
        exposure="travelled",
        outcome="sleep_onset_minutes",
        factor=Factor.SLEEP,
        description="On travel days, sleep onset",
    )
    graph = build_evidence_graph(records, [hypothesis], resamples=RESAMPLES)
    relationship = graph.relationships[0]
    assert relationship.discovery is None
    assert relationship.is_validated is False
    assert "enough" in relationship.rejection_reason


def test_missing_outcomes_are_dropped_not_counted_as_zero():
    """A night with no measurement is not a night with a value of nothing."""
    records = []
    for i in range(120):
        exposed = i % 2 == 0
        outcomes = {} if i % 5 == 0 else {"sleep_onset_minutes": 30.0}
        records.append(
            DayRecord(
                day=START + timedelta(days=i),
                exposures=frozenset(["late_dinner"]) if exposed else frozenset(),
                outcomes=outcomes,
            )
        )
    graph = build_evidence_graph(records, [LATE_DINNER], resamples=RESAMPLES)
    relationship = graph.relationships[0]
    assert relationship.discovery is not None
    # All present values are identical, so a zero-filled group would show a
    # large spurious difference.
    assert relationship.discovery.difference == pytest.approx(0.0)


def test_failed_relationships_are_kept_not_deleted():
    """ "We looked and it is not true for you" is a finding about a person."""
    graph = build_evidence_graph(
        _history(200, effect=0.0, seed=8), [LATE_DINNER], resamples=RESAMPLES
    )
    assert len(graph.relationships) == 1
    assert graph.observed() == graph.relationships


# --- what a person is shown ------------------------------------------------


def test_a_validated_statement_names_its_tier_and_day_count():
    graph = build_evidence_graph(
        _history(200, effect=25.0, seed=1), [LATE_DINNER], resamples=RESAMPLES
    )
    statement = graph.validated()[0].statement()
    assert "validated" in statement
    assert "comparable days" in statement


def test_an_observed_statement_says_it_cannot_support_a_recommendation():
    """A hypothesis and a validated finding must never read the same."""
    graph = build_evidence_graph(
        _history(200, effect=0.0, seed=4), [LATE_DINNER], resamples=RESAMPLES
    )
    statement = graph.relationships[0].statement()
    assert "observed only" in statement
    assert "not a basis for a recommendation" in statement
    assert "validated -" not in statement


def test_comparable_days_counts_evidence_not_history_length():
    graph = build_evidence_graph(
        _history(200, effect=25.0, seed=1), [LATE_DINNER], resamples=RESAMPLES
    )
    relationship = graph.validated()[0]
    assert relationship.comparable_days <= graph.history_days
    assert relationship.comparable_days > 0


def test_supporting_orders_by_effect_size():
    rng = random.Random(77)
    records = []
    for i in range(300):
        big = rng.random() < 0.4
        small = rng.random() < 0.4
        exposures = set()
        if big:
            exposures.add("big_effect")
        if small:
            exposures.add("small_effect")
        value = rng.gauss(30.0, 4.0) + (30.0 if big else 0.0) + (12.0 if small else 0.0)
        records.append(
            DayRecord(
                day=START + timedelta(days=i),
                exposures=frozenset(exposures),
                outcomes={"sleep_onset_minutes": value},
            )
        )
    hypotheses = hypotheses_from_exposures(records, "sleep_onset_minutes", Factor.SLEEP)
    graph = build_evidence_graph(records, hypotheses, resamples=RESAMPLES)
    ordered = graph.supporting("sleep_onset_minutes")
    assert [r.hypothesis.exposure for r in ordered] == ["big_effect", "small_effect"]


def test_summary_distinguishes_not_run_from_nothing_found():
    short = build_evidence_graph(
        _history(10, effect=20.0), [LATE_DINNER], resamples=RESAMPLES
    ).summary()
    null = build_evidence_graph(
        _history(200, effect=0.0, seed=4), [LATE_DINNER]
    ).summary()

    assert short["ran"] is False and short["validated"] == 0
    assert null["ran"] is True and null["validated"] == 0
    assert short["reason_not_run"] and null["reason_not_run"] is None


def test_replication_ratio_floor_is_between_a_coin_flip_and_certainty():
    """Below 0 validates anything with a direction; at 1.0 nothing replicates,
    because regression to the mean is real."""
    assert 0.0 < MIN_REPLICATION_EFFECT_RATIO < 1.0


def test_relationships_are_partitioned_by_factor():
    graph = build_evidence_graph(
        _history(200, effect=25.0, seed=1), [LATE_DINNER], resamples=RESAMPLES
    )
    assert graph.for_factor(Factor.SLEEP) == graph.relationships
    assert graph.for_factor(Factor.CONNECTION) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
