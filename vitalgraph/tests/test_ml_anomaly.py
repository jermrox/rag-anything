"""Multivariate anomaly detection against a personal baseline."""

import pytest

from vitalgraph.biometrics.schema import utc
from vitalgraph.biometrics.store import BiometricStore
from vitalgraph.ble.simulator import simulate_period
from vitalgraph.bridge.summarizer import nightly_windows
from vitalgraph.ml.anomaly import (
    MAX_REPORTED_Z,
    MIN_TRAINING_NIGHTS,
    InsufficientBaseline,
    PersonalAnomalyDetector,
    detect_with_baseline,
    to_content_list,
)
from vitalgraph.ml.features import features_for_windows

pytest.importorskip("sklearn")

START = utc(1772582400)
#: Ten steady nights, then two clearly poor ones.
SCRIPT = [1.0, 0.95, 0.9, 1.0, 0.95, 0.9, 1.0, 0.95, 0.9, 1.0, 0.2, 0.15]


@pytest.fixture(scope="module")
def vectors():
    store = BiometricStore(":memory:")
    store.add(simulate_period(START, nights=12, recovery_by_night=SCRIPT, seed=11))
    out = features_for_windows(store, nightly_windows(START, 12))
    store.close()
    return out


def test_cold_start_refuses_rather_than_inventing_a_baseline(vectors):
    """With no history there is nothing to be anomalous against."""
    detector = PersonalAnomalyDetector()
    with pytest.raises(InsufficientBaseline):
        detector.fit(vectors[: MIN_TRAINING_NIGHTS - 1])
    assert not detector.is_fitted


def test_scoring_before_fitting_raises(vectors):
    with pytest.raises(InsufficientBaseline):
        PersonalAnomalyDetector().score(vectors[0])


def test_poor_nights_score_higher_than_steady_ones(vectors):
    results = {r.period_id: r for r in detect_with_baseline(vectors)}
    scored = list(results.values())
    worst = max(scored, key=lambda r: r.score)
    assert worst.period_id == vectors[-1].period_id


def test_the_scripted_decline_is_flagged(vectors):
    flagged = {r.period_id for r in detect_with_baseline(vectors) if r.is_anomalous}
    assert vectors[-1].period_id in flagged
    assert vectors[-2].period_id in flagged


def test_steady_nights_are_not_flagged(vectors):
    results = detect_with_baseline(vectors)
    steady = [
        r for r in results if r.period_id not in {v.period_id for v in vectors[-2:]}
    ]
    assert not any(r.is_anomalous for r in steady)


def test_flags_name_the_features_that_drove_them(vectors):
    """A bare score is not actionable; attribution is the product."""
    worst = max(detect_with_baseline(vectors), key=lambda r: r.score)
    names = {n for n, _ in worst.contributors}
    assert names & {"mean_hr_bpm", "rmssd_ms", "pnn50_pct", "awake_fraction"}
    assert "driven by" in worst.explain()


def test_reported_deviation_is_capped(vectors):
    """An unusually consistent baseline must not yield a fifty-sigma claim."""
    for r in detect_with_baseline(vectors):
        for _, z in r.contributors:
            assert abs(z) <= MAX_REPORTED_Z


def test_scoring_is_deterministic(vectors):
    a = PersonalAnomalyDetector(seed=3)
    a.fit(vectors[:10])
    b = PersonalAnomalyDetector(seed=3)
    b.fit(vectors[:10])
    assert a.score(vectors[-1]).score == b.score(vectors[-1]).score


def test_each_night_is_judged_only_against_its_past(vectors):
    """Fitting on the night being scored lets an anomaly teach the model that
    it is normal, which is how a slow decline becomes invisible."""
    results = detect_with_baseline(vectors)
    assert len(results) == len(vectors) - MIN_TRAINING_NIGHTS
    assert results[0].baseline_nights == MIN_TRAINING_NIGHTS


def test_fit_returns_a_card_marked_synthetic(vectors):
    card = PersonalAnomalyDetector().fit(vectors[:10])
    assert card.algorithm == "IsolationForest"
    assert card.is_synthetic
    assert card.n_training_samples == 10


def test_results_carry_a_scope_caveat(vectors):
    r = detect_with_baseline(vectors)[-1]
    assert "own history" in r.caveat
    assert "not a diagnosis" in r.caveat


def test_content_list_emits_only_flagged_periods(vectors):
    items = to_content_list(detect_with_baseline(vectors))
    assert [i["type"] for i in items] == ["text", "table"]
    body = items[1]["table_body"]
    assert vectors[-1].period_id in body
    assert vectors[0].period_id not in body
    assert "not a diagnosis" in items[1]["table_footnote"][0].lower()


def test_content_list_is_empty_when_nothing_is_flagged(vectors):
    assert to_content_list([]) == []


def test_result_serialises_for_the_api(vectors):
    d = detect_with_baseline(vectors)[-1].as_dict()
    assert set(d) >= {
        "period_id",
        "score",
        "is_anomalous",
        "contributors",
        "explanation",
    }
    assert isinstance(d["contributors"], list)
