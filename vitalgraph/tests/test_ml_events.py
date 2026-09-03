"""Detection of desaturation and irregularity episodes.

Measured against the simulator's ground-truth windows, which is what makes
this testable at all -- the events have to exist before they can be found.
"""

from datetime import timedelta

import pytest

from vitalgraph.biometrics.schema import SignalType, utc
from vitalgraph.biometrics.store import BiometricStore
from vitalgraph.ble.simulator import simulate_night
from vitalgraph.ml.events import (
    MIN_EVENT_SECONDS,
    SIGNATURE_OF_CAUSE,
    detect_desaturations,
    detect_events,
    detect_irregularity,
    score_against_truth,
    to_content_list,
)

START = utc(1772582400)
NIGHT = timedelta(hours=8)


def _night(seed, apnea=0, arrhythmia=0):
    store = BiometricStore(":memory:")
    samples, episodes = simulate_night(
        START,
        hours=8,
        seed=seed,
        apnea_events=apnea,
        arrhythmia_events=arrhythmia,
        return_episodes=True,
    )
    store.add(samples)
    return store, episodes


# --- the simulator must actually produce events ---------------------------


def test_simulator_injects_observable_desaturations():
    """A 1/min SpO2 cadence cannot see a 45 s event; the generator samples
    faster so detection is testable rather than impossible by construction."""
    plain, _ = _night(7)
    evented, _ = _night(7, apnea=3)

    clean_min = min(
        s.value for s in plain.samples_in(SignalType.SPO2, START, START + NIGHT)
    )
    event_min = min(
        s.value for s in evented.samples_in(SignalType.SPO2, START, START + NIGHT)
    )
    assert event_min < clean_min - 3.0


def test_event_injection_is_opt_in_and_backwards_compatible():
    """Existing callers get a plain list and unchanged output."""
    assert isinstance(simulate_night(START, hours=1, seed=3), list)
    a = simulate_night(START, hours=1, seed=3)
    b = simulate_night(START, hours=1, seed=3)
    assert [s.value for s in a] == [s.value for s in b]


def test_episodes_do_not_overlap():
    _, episodes = _night(2, apnea=4, arrhythmia=3)
    ordered = sorted(episodes, key=lambda e: e.start_s)
    for first, second in zip(ordered, ordered[1:]):
        assert first.end_s <= second.start_s


# --- detection -------------------------------------------------------------


def test_desaturations_are_detected():
    store, episodes = _night(7, apnea=4)
    score = score_against_truth(
        detect_events(store, START, START + NIGHT), episodes, START, "apnea"
    )
    assert score.recall == 1.0
    assert score.precision == 1.0


def test_irregularity_is_detected():
    store, episodes = _night(7, arrhythmia=3)
    score = score_against_truth(
        detect_events(store, START, START + NIGHT), episodes, START, "arrhythmia"
    )
    assert score.recall == 1.0


def test_both_kinds_are_found_together():
    store, episodes = _night(5, apnea=3, arrhythmia=2)
    found = detect_events(store, START, START + NIGHT)
    kinds = {e.kind for e in found}
    assert kinds == {"desaturation", "irregularity"}
    assert found == sorted(found, key=lambda e: e.start)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_clean_nights_produce_no_events(seed):
    """False positives on healthy nights are how a health product loses trust."""
    store, _ = _night(seed)
    assert detect_events(store, START, START + NIGHT) == []


def test_short_dips_are_not_events():
    assert MIN_EVENT_SECONDS >= 10.0


def test_detection_is_deterministic():
    store, _ = _night(7, apnea=3, arrhythmia=2)
    first = detect_events(store, START, START + NIGHT)
    second = detect_events(store, START, START + NIGHT)
    assert [e.as_dict() for e in first] == [e.as_dict() for e in second]


def test_empty_window_is_safe():
    empty = BiometricStore(":memory:")
    assert detect_desaturations(empty, START, START + NIGHT) == []
    assert detect_irregularity(empty, START, START + NIGHT) == []


# --- vocabulary and reporting ---------------------------------------------


def test_cause_and_signature_vocabularies_stay_distinct():
    """A detector reporting 'apnea' would claim a diagnosis it cannot support."""
    assert SIGNATURE_OF_CAUSE["apnea"] == "desaturation"
    assert SIGNATURE_OF_CAUSE["arrhythmia"] == "irregularity"
    store, _ = _night(5, apnea=2, arrhythmia=2)
    reported = {e.kind for e in detect_events(store, START, START + NIGHT)}
    assert not reported & set(SIGNATURE_OF_CAUSE)


def test_content_list_disclaims_diagnosis():
    store, _ = _night(7, apnea=3, arrhythmia=2)
    items = to_content_list(detect_events(store, START, START + NIGHT))
    assert [i["type"] for i in items] == ["text", "table"]
    footnote = items[1]["table_footnote"][0].lower()
    assert "not a diagnosis of atrial fibrillation" in footnote
    assert "polysomnography" in footnote


def test_content_list_is_empty_without_events():
    assert to_content_list([]) == []
