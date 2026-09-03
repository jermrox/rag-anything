"""Epoch features and learned sleep staging.

The behaviour under test is as much methodological as functional: subject-level
splitting, refusal to evaluate on trained nights, and the model reporting its
own accuracy as implausible when it is.
"""

from datetime import timedelta

import pytest

from vitalgraph.biometrics.schema import utc
from vitalgraph.biometrics.store import BiometricStore
from vitalgraph.ble.simulator import simulate_night
from vitalgraph.ml.epochs import (
    EPOCH_FEATURE_NAMES,
    epoch_samples,
    labelled_only,
    stage_distribution,
)
from vitalgraph.ml.staging import (
    IMPLAUSIBLE_ACCURACY,
    MIN_TRAINING_NIGHTS,
    NotEnoughNights,
    SleepStager,
    subject_split,
    to_content_list,
)

pytest.importorskip("sklearn")

START = utc(1772582400)


@pytest.fixture(scope="module")
def samples():
    store = BiometricStore(":memory:")
    out = []
    for i in range(10):
        night_start = START + timedelta(days=i)
        store.add(
            simulate_night(night_start, hours=8, recovery=0.5 + 0.05 * i, seed=100 + i)
        )
        out += epoch_samples(
            store, night_start, night_start + timedelta(hours=8), f"night-{i}"
        )
    store.close()
    return out


# --- epoch features --------------------------------------------------------


def test_epochs_have_the_declared_feature_length(samples):
    assert all(len(s.values) == len(EPOCH_FEATURE_NAMES) for s in samples)


def test_epoch_vocabulary_has_no_duplicates():
    assert len(set(EPOCH_FEATURE_NAMES)) == len(EPOCH_FEATURE_NAMES)


def test_labels_are_attached_where_ground_truth_exists(samples):
    labelled = labelled_only(samples)
    assert labelled
    assert len(labelled) < len(samples)  # stages log per minute, epochs are 30 s


def test_all_four_stages_are_represented(samples):
    assert set(stage_distribution(samples)) == {"awake", "light", "deep", "rem"}


def test_empty_window_yields_no_epochs():
    empty = BiometricStore(":memory:")
    assert epoch_samples(empty, START, START + timedelta(hours=8), "n") == []


# --- subject-level splitting ----------------------------------------------


def test_split_never_puts_a_night_on_both_sides(samples):
    """Epoch-level splitting leaks: adjacent 30 s windows are near-identical,
    so the model would be scored on what it trained on."""
    train, test = subject_split(samples, test_fraction=0.3, seed=1)
    assert {s.night_id for s in train} & {s.night_id for s in test} == set()


def test_split_is_deterministic(samples):
    a = subject_split(samples, 0.3, seed=4)[1]
    b = subject_split(samples, 0.3, seed=4)[1]
    assert {s.night_id for s in a} == {s.night_id for s in b}


def test_split_requires_two_nights(samples):
    one_night = [s for s in samples if s.night_id == "night-0"]
    with pytest.raises(NotEnoughNights):
        subject_split(one_night)


def test_split_always_holds_out_at_least_one_night(samples):
    _, test = subject_split(samples, test_fraction=0.01, seed=0)
    assert len({s.night_id for s in test}) >= 1


# --- training and evaluation ----------------------------------------------


def test_training_requires_several_nights(samples):
    too_few = [s for s in samples if s.night_id in {"night-0", "night-1"}]
    with pytest.raises(NotEnoughNights):
        SleepStager().fit(too_few)
    assert MIN_TRAINING_NIGHTS >= 3


def test_predicting_before_fitting_raises(samples):
    with pytest.raises(RuntimeError):
        SleepStager().predict(samples[:5])


def test_evaluation_refuses_nights_the_model_trained_on(samples):
    """Scoring on training nights would make the number meaningless."""
    train, _ = subject_split(samples, 0.3, seed=1)
    stager = SleepStager(seed=0)
    stager.fit(train)
    with pytest.raises(NotEnoughNights):
        stager.evaluate(train)


def test_model_reports_its_own_accuracy_as_implausible(samples):
    """The central honesty check. The simulator's stage track is a
    deterministic function of elapsed time, so a model given elapsed time
    recovers it perfectly -- which is a finding about the data, not a good
    model, and the code must say so rather than celebrate."""
    train, test = subject_split(samples, 0.3, seed=1)
    stager = SleepStager(seed=0)
    stager.fit(train)
    evaluation = stager.evaluate(test)

    assert evaluation.accuracy >= IMPLAUSIBLE_ACCURACY
    assert evaluation.is_implausible
    note = evaluation.plausibility_note()
    assert "IMPLAUSIBLE" in note
    assert "Do not ship this as a capability." in note


def test_evaluation_carries_the_synthetic_caveat(samples):
    train, test = subject_split(samples, 0.3, seed=1)
    stager = SleepStager(seed=0)
    stager.fit(train)
    assert "simulated" in stager.evaluate(test).caveat


def test_fit_returns_a_card_marked_synthetic(samples):
    train, _ = subject_split(samples, 0.3, seed=1)
    card = SleepStager(seed=0).fit(train)
    assert card.is_synthetic
    assert card.algorithm == "RandomForestClassifier"
    assert card.feature_names == EPOCH_FEATURE_NAMES


def test_predictions_are_named_stages(samples):
    train, test = subject_split(samples, 0.3, seed=1)
    stager = SleepStager(seed=0)
    stager.fit(train)
    named = set(stager.predict_named(test[:200]))
    assert named <= {"awake", "light", "deep", "rem"}


def test_staging_is_deterministic(samples):
    train, test = subject_split(samples, 0.3, seed=1)
    a, b = SleepStager(seed=2), SleepStager(seed=2)
    a.fit(train)
    b.fit(train)
    assert a.predict(test[:100]) == b.predict(test[:100])


def test_content_list_surfaces_the_warning(samples):
    train, test = subject_split(samples, 0.3, seed=1)
    stager = SleepStager(seed=0)
    card = stager.fit(train)
    items = to_content_list(stager.evaluate(test), card)

    assert [i["type"] for i in items] == ["text", "table"]
    assert "IMPLAUSIBLE" in items[0]["text"]
    assert "simulator" in items[0]["text"].lower()
    assert "IMPLAUSIBLE" in items[1]["table_footnote"][0]
