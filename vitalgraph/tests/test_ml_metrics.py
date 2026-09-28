"""Tests for honest sleep-staging metrics.

The whole point of this module is catching a model that scores well without
having learned anything. A night is roughly 85% sleep, so the tests below are
mostly about the classifier that never wakes up: it must look good on accuracy
and be caught by everything else.
"""

from __future__ import annotations

import pytest

from vitalgraph.ml.metrics import (
    CHANCE_LEVEL_KAPPA,
    IMPLAUSIBLE_ACCURACY,
    MIN_SKILL_MARGIN,
    CrossValidationSummary,
    SkillAssessment,
    assess_skill,
    balanced_accuracy,
    cohens_kappa,
    majority_baseline_accuracy,
    missed_classes,
    per_class_recall,
)

#: A realistic night: mostly light sleep, a little deep and REM, less wake.
NIGHT = [1.0] * 60 + [2.0] * 15 + [3.0] * 15 + [0.0] * 10


# --- the baseline ----------------------------------------------------------


def test_majority_baseline_scores_high_on_an_imbalanced_night():
    """The number a real model has to beat, and often does not."""
    baseline = majority_baseline_accuracy(NIGHT, NIGHT)
    assert baseline == pytest.approx(0.60)


def test_majority_class_comes_from_training_not_test():
    """Taking it from test would let the baseline see the answers.

    That makes the baseline unbeatable and inverts the comparison it exists
    to provide.
    """
    train = [1.0] * 90 + [0.0] * 10  # majority is light sleep
    test = [0.0] * 90 + [1.0] * 10  # but the test night is mostly wake
    # Using the training majority scores badly here, which is correct.
    assert majority_baseline_accuracy(train, test) == pytest.approx(0.10)


def test_baseline_of_empty_input_is_zero_not_an_error():
    assert majority_baseline_accuracy([], [1.0]) == 0.0
    assert majority_baseline_accuracy([1.0], []) == 0.0


# --- kappa -----------------------------------------------------------------


def test_kappa_is_one_for_perfect_agreement():
    assert cohens_kappa(NIGHT, NIGHT) == pytest.approx(1.0)


def test_kappa_is_near_zero_for_a_model_that_always_guesses_the_majority():
    """The headline case. High accuracy, no agreement beyond chance."""
    always_light = [1.0] * len(NIGHT)
    accuracy = sum(1 for p, a in zip(always_light, NIGHT) if p == a) / len(NIGHT)
    assert accuracy == pytest.approx(0.60)
    assert cohens_kappa(always_light, NIGHT) == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_zero_when_chance_agreement_is_total():
    """One class, predicted every time. The correction is undefined.

    Returning 1.0 would claim perfect skill for a model that cannot be wrong.
    """
    assert cohens_kappa([1.0] * 20, [1.0] * 20) == 0.0


def test_kappa_handles_empty_and_mismatched_input():
    assert cohens_kappa([], []) == 0.0
    assert cohens_kappa([1.0], [1.0, 2.0]) == 0.0


# --- balanced accuracy and missed classes ----------------------------------


def test_balanced_accuracy_punishes_a_model_that_ignores_rare_stages():
    always_light = [1.0] * len(NIGHT)
    # Four stages occur; only one is ever recalled.
    assert balanced_accuracy(always_light, NIGHT) == pytest.approx(0.25)


def test_per_class_recall_only_reports_stages_that_occurred():
    """A stage the model invented, which never happened, has no recall.

    Inventing an entry for it would imply the night contained something it
    did not.
    """
    recalls = per_class_recall([9.0] * 10, [1.0] * 10)
    assert set(recalls) == {1.0}
    assert recalls[1.0] == 0.0


def test_missed_classes_names_the_stages_never_predicted():
    """The specific failure accuracy hides best."""
    assert missed_classes([1.0] * len(NIGHT), NIGHT) == [0.0, 2.0, 3.0]
    assert missed_classes(NIGHT, NIGHT) == []


# --- the skill verdict -----------------------------------------------------


def test_a_model_that_never_wakes_up_is_reported_as_having_no_skill():
    """The single most important assertion in this module.

    60% accuracy against a 60% baseline is not a stager, and must not be
    describable as one.
    """
    always_light = [1.0] * len(NIGHT)
    skill = assess_skill(always_light, NIGHT, NIGHT)

    assert skill.accuracy == pytest.approx(0.60)
    assert skill.baseline_accuracy == pytest.approx(0.60)
    assert skill.margin_over_baseline == pytest.approx(0.0)
    assert skill.has_skill is False
    assert "No demonstrated skill" in skill.verdict()
    assert "Do not describe this as staging" in skill.verdict()


def test_a_genuinely_good_model_is_credited():
    """A model in the band the literature actually reports, ~75%.

    Deliberately not higher. Four-class staging from these signals reaches
    roughly 70-80% against polysomnography, so a fixture scoring 90% would be
    modelling a leak rather than a good result.
    """
    predicted = list(NIGHT)
    for i in range(0, len(NIGHT), 4):  # ~25% wrong
        predicted[i] = 0.0 if NIGHT[i] != 0.0 else 1.0
    skill = assess_skill(predicted, NIGHT, NIGHT)

    assert 0.70 <= skill.accuracy < IMPLAUSIBLE_ACCURACY
    assert skill.has_skill is True
    assert skill.kappa > CHANCE_LEVEL_KAPPA
    assert "Beats the majority baseline" in skill.verdict()


def test_a_perfect_score_is_reported_as_a_leak_not_as_skill():
    """The most dangerous output this module could produce.

    A model scoring 100% clears the majority baseline by a mile, so a
    margin-only rule congratulates it. Against this simulator that is exactly
    what happens -- the stage track is a deterministic function of elapsed
    time -- and a reader who sees "beats the majority baseline" concludes the
    thing works.
    """
    skill = assess_skill(NIGHT, NIGHT, NIGHT)

    assert skill.accuracy == 1.0
    assert skill.margin_over_baseline > MIN_SKILL_MARGIN
    assert skill.is_implausible is True
    assert skill.has_skill is False, "a leak must never be reported as skill"

    verdict = skill.verdict()
    assert "IMPLAUSIBLE" in verdict
    assert "Do not ship this as staging" in verdict
    assert "Beats the majority baseline" not in verdict


def test_skill_requires_both_a_margin_and_agreement_beyond_chance():
    """Either failure alone is disqualifying.

    A model can clear the baseline on accuracy while agreeing no better than
    chance, so both conditions are checked.
    """
    clears_margin_only = SkillAssessment(
        accuracy=0.80,
        baseline_accuracy=0.60,
        kappa=CHANCE_LEVEL_KAPPA - 0.01,
        balanced_accuracy=0.5,
    )
    good_kappa_no_margin = SkillAssessment(
        accuracy=0.61,
        baseline_accuracy=0.60,
        kappa=0.9,
        balanced_accuracy=0.9,
    )
    assert clears_margin_only.has_skill is False
    assert good_kappa_no_margin.has_skill is False


def test_the_skill_margin_threshold_is_a_real_bar_not_zero():
    """At zero, beating the baseline by one epoch would count as skill."""
    assert MIN_SKILL_MARGIN > 0.0


def test_skill_serialises_with_its_verdict():
    skill = assess_skill([1.0] * len(NIGHT), NIGHT, NIGHT)
    payload = skill.as_dict()
    assert payload["has_skill"] is False
    assert "baseline_accuracy" in payload
    assert "kappa" in payload
    assert payload["verdict"]


# --- cross-validation summary ----------------------------------------------


def _assessment(accuracy: float, kappa: float) -> SkillAssessment:
    return SkillAssessment(
        accuracy=accuracy,
        baseline_accuracy=0.60,
        kappa=kappa,
        balanced_accuracy=accuracy,
    )


def test_summary_surfaces_the_worst_subject_not_just_the_mean():
    """A mean hides a model that fails completely for a minority of people,
    which for a health product is the case that matters most."""
    summary = CrossValidationSummary(
        per_subject=(
            ("night-a", _assessment(0.85, 0.7)),
            ("night-b", _assessment(0.83, 0.7)),
            ("night-c", _assessment(0.40, 0.05)),
        )
    )
    assert summary.mean_accuracy == pytest.approx(0.6933, abs=1e-3)
    assert summary.worst[0] == "night-c"
    assert summary.subjects_without_skill == ["night-c"]
    assert summary.works_for_everyone is False


def test_summary_reports_working_for_everyone_only_when_it_does():
    summary = CrossValidationSummary(
        per_subject=(
            ("night-a", _assessment(0.85, 0.7)),
            ("night-b", _assessment(0.83, 0.7)),
        )
    )
    assert summary.works_for_everyone is True
    assert summary.subjects_without_skill == []


def test_an_implausible_fold_disqualifies_the_run():
    """A leak must not be reported as universal success.

    Implausible folds are tracked apart from unskilled ones because the two
    need opposite responses: one says the model is too weak, the other says
    the experiment is wrong.
    """
    summary = CrossValidationSummary(
        per_subject=(
            ("night-a", _assessment(0.78, 0.7)),
            ("night-b", _assessment(1.00, 1.0)),
        )
    )
    assert summary.implausible_subjects == ["night-b"]
    assert summary.works_for_everyone is False
    assert summary.summary()["implausible_subjects"] == ["night-b"]


def test_an_empty_summary_does_not_claim_to_work_for_everyone():
    """Vacuous truth would be the wrong answer: nothing was evaluated."""
    empty = CrossValidationSummary(per_subject=())
    assert empty.works_for_everyone is False
    assert empty.worst is None
    assert empty.mean_accuracy == 0.0


def test_summary_serialises_per_subject():
    summary = CrossValidationSummary(per_subject=(("night-a", _assessment(0.85, 0.7)),))
    payload = summary.summary()
    assert payload["n_subjects"] == 1
    assert payload["worst_subject"] == "night-a"
    assert "night-a" in payload["per_subject"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
