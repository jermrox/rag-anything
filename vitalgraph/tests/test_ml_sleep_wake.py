"""Tests for the two-class sleep/wake reduction and the per-task ceiling.

The reduction exists to answer a diagnostic question -- is the failure the
four-class task or the feature set -- so the tests are mostly about it staying
an honest narrower claim rather than a quiet rebranding of a broken model.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vitalgraph.biometrics.schema import SleepStage
from vitalgraph.ml.epochs import EpochSample
from vitalgraph.ml.metrics import (
    IMPLAUSIBLE_ACCURACY,
    IMPLAUSIBLE_SLEEP_WAKE_ACCURACY,
    SkillAssessment,
    assess_skill,
)
from vitalgraph.ml.staging import SLEEP_WAKE_NAMES, collapse_to_sleep_wake

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _epoch(index: int, label: float | None) -> EpochSample:
    return EpochSample(
        night_id="n1",
        index=index,
        start=START,
        values=(float(index),) * 12,
        label=label,
    )


# --- the reduction ---------------------------------------------------------


def test_every_sleep_stage_collapses_to_asleep():
    stages = [
        SleepStage.LIGHT.value,
        SleepStage.DEEP.value,
        SleepStage.REM.value,
    ]
    collapsed = collapse_to_sleep_wake([_epoch(i, s) for i, s in enumerate(stages)])
    assert [e.label for e in collapsed] == [1.0, 1.0, 1.0]


def test_awake_collapses_to_wake():
    collapsed = collapse_to_sleep_wake([_epoch(0, SleepStage.AWAKE.value)])
    assert collapsed[0].label == 0.0


def test_features_are_untouched_by_the_reduction():
    """Only the label changes. Altering features would make the two-class and
    four-class results incomparable, and comparing them is the entire point."""
    original = _epoch(3, SleepStage.DEEP.value)
    collapsed = collapse_to_sleep_wake([original])[0]
    assert collapsed.values == original.values
    assert collapsed.index == original.index
    assert collapsed.night_id == original.night_id
    assert collapsed.start == original.start


def test_unlabelled_epochs_stay_unlabelled():
    """An epoch with no ground truth must not acquire one.

    Defaulting it to "asleep" would fabricate a label for exactly the epochs
    where the truth is unknown.
    """
    collapsed = collapse_to_sleep_wake([_epoch(0, None)])
    assert collapsed[0].label is None


def test_the_reduction_does_not_mutate_its_input():
    samples = [_epoch(0, SleepStage.REM.value)]
    collapse_to_sleep_wake(samples)
    assert samples[0].label == SleepStage.REM.value


def test_only_two_class_names_exist():
    """A sleep/wake classifier cannot report deep sleep or REM.

    Naming only the two it can distinguish is what stops it being presented as
    though it could report more.
    """
    assert set(SLEEP_WAKE_NAMES.values()) == {"wake", "sleep"}
    assert set(SLEEP_WAKE_NAMES) == {0.0, 1.0}


# --- the per-task ceiling --------------------------------------------------


def test_the_sleep_wake_ceiling_is_higher_than_the_four_class_one():
    """Two-class is a genuinely easier task with a higher published range.

    Applying the four-class ceiling to it would condemn a real result as a
    leak; applying the two-class ceiling to four-class would excuse one.
    """
    assert IMPLAUSIBLE_SLEEP_WAKE_ACCURACY > IMPLAUSIBLE_ACCURACY


def test_a_result_can_be_a_leak_on_one_task_and_legitimate_on_the_other():
    """92% is implausible for four-class and unremarkable for sleep/wake."""
    four_class = SkillAssessment(
        accuracy=0.92,
        baseline_accuracy=0.60,
        kappa=0.7,
        balanced_accuracy=0.9,
        implausible_above=IMPLAUSIBLE_ACCURACY,
    )
    sleep_wake = SkillAssessment(
        accuracy=0.92,
        baseline_accuracy=0.60,
        kappa=0.7,
        balanced_accuracy=0.9,
        implausible_above=IMPLAUSIBLE_SLEEP_WAKE_ACCURACY,
    )
    assert four_class.is_implausible is True
    assert four_class.has_skill is False
    assert sleep_wake.is_implausible is False
    assert sleep_wake.has_skill is True


def test_the_ceiling_defaults_to_the_stricter_four_class_value():
    """A caller who forgets to say which task gets the conservative answer."""
    assert (
        assess_skill([1.0] * 10, [1.0] * 10, [1.0] * 10).implausible_above
        == IMPLAUSIBLE_ACCURACY
    )


def test_the_verdict_names_the_ceiling_it_applied():
    """A reader cannot judge "implausible" without knowing the bar."""
    skill = assess_skill(
        [1.0] * 100,
        [1.0] * 97 + [0.0] * 3,
        [1.0] * 50 + [0.0] * 50,
        implausible_above=IMPLAUSIBLE_SLEEP_WAKE_ACCURACY,
    )
    assert skill.is_implausible is True
    assert f"{IMPLAUSIBLE_SLEEP_WAKE_ACCURACY:.0%}" in skill.verdict()


# --- the finding this reduction produced -----------------------------------


def test_below_baseline_accuracy_with_positive_kappa_has_no_skill():
    """The shape of the real slpdb result, pinned as a behaviour.

    Several subjects scored positive kappa -- real agreement beyond chance --
    while landing below their majority baseline. That combination is a model
    with weak signal that trades away too much of the majority class, and it
    must not count as skill: predicting "asleep" throughout would have scored
    higher.
    """
    weak = SkillAssessment(
        accuracy=0.565,
        baseline_accuracy=0.892,
        kappa=0.105,
        balanced_accuracy=0.55,
        implausible_above=IMPLAUSIBLE_SLEEP_WAKE_ACCURACY,
    )
    assert weak.margin_over_baseline < 0
    assert weak.has_skill is False
    assert "No demonstrated skill" in weak.verdict()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
