"""Scoring a sleep stager honestly.

Accuracy is the wrong headline number for sleep staging, and the reason is
arithmetic rather than taste. A night is roughly 85-90% sleep, so a classifier
that predicts "asleep" for every epoch and never wakes up scores in the
high eighties. Reported alone, that number looks like a working product.

So every figure here exists to answer a question accuracy cannot:

**Majority baseline** -- what the same accuracy would be from a model that has
learned nothing and always guesses the commonest stage. A model must be
compared against this, not against zero. If it does not clear the baseline it
has no skill, however good the raw accuracy looks.

**Cohen's kappa** -- agreement corrected for what chance alone would produce.
The standard metric in the sleep-staging literature, precisely because of the
imbalance above. Kappa near zero means chance-level, whatever the accuracy.

**Balanced accuracy** -- mean recall across stages, so the rarest stage counts
as much as the commonest. This is what falls through the floor when a model
quietly stops predicting a class, which accuracy hides entirely.

**Per-subject spread** -- a mean over nights conceals a model that works
beautifully for eight people and fails completely for two. The worst night
matters more than the average one for a health product.

The approach follows what the wrist-staging literature does with the same
sensor set (motion and heart rate scored against polysomnography): leave one
subject out, score each held-out subject separately, and report the
distribution rather than a single number.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

#: Kappa below this is chance-level agreement whatever the accuracy says.
#: Landis and Koch call 0.00-0.20 "slight"; a stager in that band has not
#: learned the task.
CHANCE_LEVEL_KAPPA = 0.20

#: A model must beat the majority baseline by at least this much in absolute
#: accuracy to be described as having skill. Below it, the honest reading is
#: that predicting the commonest stage every epoch would do as well.
MIN_SKILL_MARGIN = 0.05

#: And an upper bound, which matters just as much.
#:
#: Four-class staging from RR intervals reaches roughly 70-80% agreement with
#: polysomnography in the published literature. A result far above that is not
#: a better model, it is a leak: synthetic training data, a split that put
#: adjacent epochs on both sides, or a feature that encodes the label.
#:
#: This is here rather than only in the stager because skill and implausibility
#: are the same judgement from opposite ends. Without it, a model scoring 100%
#: clears the baseline by a mile and gets congratulated by :meth:`verdict`,
#: which is the most dangerous output this module could produce -- a reader who
#: sees "beats the majority baseline" will conclude the thing works.
IMPLAUSIBLE_ACCURACY = 0.90


def majority_baseline_accuracy(
    train_labels: Sequence[float], test_labels: Sequence[float]
) -> float:
    """Accuracy of always predicting the commonest *training* stage.

    The majority class is taken from training data, never from test. Choosing
    it from the test set would let the baseline peek at the answers and make
    it unbeatable, which inverts the comparison it exists to provide.
    """
    if not train_labels or not test_labels:
        return 0.0
    majority = Counter(train_labels).most_common(1)[0][0]
    return sum(1 for label in test_labels if label == majority) / len(test_labels)


def cohens_kappa(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Agreement corrected for chance.

    Returns 0.0 when chance agreement is total -- one class present, predicted
    every time -- because the correction is then undefined and the honest
    answer is that the result carries no information, not that it is perfect.
    """
    if not predicted or len(predicted) != len(actual):
        return 0.0
    n = len(actual)
    observed = sum(1 for p, a in zip(predicted, actual) if p == a) / n

    predicted_counts = Counter(predicted)
    actual_counts = Counter(actual)
    expected = sum(
        (predicted_counts.get(label, 0) / n) * (actual_counts.get(label, 0) / n)
        for label in set(predicted_counts) | set(actual_counts)
    )
    if expected >= 1.0:
        return 0.0
    return (observed - expected) / (1.0 - expected)


def per_class_recall(
    predicted: Sequence[float], actual: Sequence[float]
) -> Dict[float, float]:
    """Recall for each stage that actually occurs in the reference labels.

    Keyed on stages present in ``actual`` only. A stage the model invented but
    which never occurred has no recall to report, and inventing an entry for it
    would imply the night contained something it did not.
    """
    recalls: Dict[float, float] = {}
    for label in sorted(set(actual)):
        support = sum(1 for a in actual if a == label)
        hits = sum(1 for p, a in zip(predicted, actual) if a == label and p == label)
        recalls[label] = hits / support if support else 0.0
    return recalls


def balanced_accuracy(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Mean recall across stages, so a rare stage counts as much as a common one."""
    recalls = per_class_recall(predicted, actual)
    if not recalls:
        return 0.0
    return sum(recalls.values()) / len(recalls)


def missed_classes(predicted: Sequence[float], actual: Sequence[float]) -> List[float]:
    """Stages that occurred but the model never once predicted.

    Reported separately because it is the specific failure raw accuracy hides
    best: a stager that has silently given up on the rarest stage still scores
    well, and would be shipped.
    """
    seen = set(predicted)
    return [label for label in sorted(set(actual)) if label not in seen]


@dataclass(frozen=True, slots=True)
class SkillAssessment:
    """Whether a model has skill, as distinct from a high accuracy."""

    accuracy: float
    baseline_accuracy: float
    kappa: float
    balanced_accuracy: float

    @property
    def margin_over_baseline(self) -> float:
        return self.accuracy - self.baseline_accuracy

    @property
    def is_implausible(self) -> bool:
        """Whether the result is too good to have come from real staging."""
        return self.accuracy >= IMPLAUSIBLE_ACCURACY

    @property
    def has_skill(self) -> bool:
        """True only when the model beats guessing *and* stays believable.

        Three conditions, and every one of them is a way this can fail. A model
        can clear the baseline on accuracy by a whisker while agreeing with the
        reference no better than chance. It can also clear it by so much that
        the only explanation is a leak -- which is not skill, and must not be
        reported as skill just because the arithmetic points the right way.
        """
        return (
            self.margin_over_baseline >= MIN_SKILL_MARGIN
            and self.kappa > CHANCE_LEVEL_KAPPA
            and not self.is_implausible
        )

    def verdict(self) -> str:
        """One line a reader can act on, naming the comparison."""
        if self.is_implausible:
            return (
                f"IMPLAUSIBLE: {self.accuracy:.1%} accuracy, kappa "
                f"{self.kappa:.2f}, against a {self.baseline_accuracy:.1%} "
                f"majority-class baseline. Published four-class staging from "
                f"these signals reaches roughly 70-80%, so a result this high "
                f"means the labels were recoverable from the features by "
                f"construction -- synthetic data, a leaky split, or a feature "
                f"encoding the label. Beating the baseline this far is the "
                f"symptom, not the achievement. Do not ship this as staging."
            )
        if not self.has_skill:
            return (
                f"No demonstrated skill: {self.accuracy:.1%} accuracy against a "
                f"{self.baseline_accuracy:.1%} majority-class baseline "
                f"(margin {self.margin_over_baseline:+.1%}), kappa {self.kappa:.2f}. "
                f"Predicting the commonest stage every epoch would do about as "
                f"well. Do not describe this as staging."
            )
        return (
            f"Beats the majority baseline: {self.accuracy:.1%} against "
            f"{self.baseline_accuracy:.1%} (margin {self.margin_over_baseline:+.1%}), "
            f"kappa {self.kappa:.2f}, balanced accuracy "
            f"{self.balanced_accuracy:.1%}."
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "accuracy": round(self.accuracy, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "margin_over_baseline": round(self.margin_over_baseline, 4),
            "kappa": round(self.kappa, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
            "has_skill": self.has_skill,
            "is_implausible": self.is_implausible,
            "verdict": self.verdict(),
        }


def assess_skill(
    predicted: Sequence[float],
    actual: Sequence[float],
    train_labels: Sequence[float],
) -> SkillAssessment:
    """Score a set of predictions against the baseline that makes them meaningful."""
    n = len(actual)
    accuracy = sum(1 for p, a in zip(predicted, actual) if p == a) / n if n else 0.0
    return SkillAssessment(
        accuracy=accuracy,
        baseline_accuracy=majority_baseline_accuracy(train_labels, actual),
        kappa=cohens_kappa(predicted, actual),
        balanced_accuracy=balanced_accuracy(predicted, actual),
    )


@dataclass(frozen=True, slots=True)
class CrossValidationSummary:
    """Per-subject results from leaving one subject out at a time.

    The distribution is the point. A single held-out night gives one number
    with no sense of whether it was lucky, and a mean over nights hides a model
    that fails completely for a minority of people -- which for a health
    product is the case that matters most.
    """

    per_subject: Tuple[Tuple[str, SkillAssessment], ...]

    @property
    def n_subjects(self) -> int:
        return len(self.per_subject)

    @property
    def mean_accuracy(self) -> float:
        if not self.per_subject:
            return 0.0
        return sum(a.accuracy for _, a in self.per_subject) / self.n_subjects

    @property
    def worst(self) -> Tuple[str, SkillAssessment] | None:
        """The subject the model did worst on. Reported because a health
        product is judged on its failures, not its average."""
        if not self.per_subject:
            return None
        return min(self.per_subject, key=lambda pair: pair[1].accuracy)

    @property
    def subjects_without_skill(self) -> List[str]:
        return [name for name, a in self.per_subject if not a.has_skill]

    @property
    def implausible_subjects(self) -> List[str]:
        """Subjects whose result is too good to be real.

        Tracked separately from a lack of skill because the two need opposite
        responses: one says the model is too weak, the other says the
        experiment is wrong.
        """
        return [name for name, a in self.per_subject if a.is_implausible]

    @property
    def works_for_everyone(self) -> bool:
        """True only when every subject shows believable skill.

        An implausible fold disqualifies the run rather than counting toward
        it, so a leak cannot be reported as universal success.
        """
        return (
            bool(self.per_subject)
            and not self.subjects_without_skill
            and not self.implausible_subjects
        )

    def summary(self) -> Dict[str, object]:
        worst = self.worst
        return {
            "n_subjects": self.n_subjects,
            "mean_accuracy": round(self.mean_accuracy, 4),
            "worst_subject": worst[0] if worst else None,
            "worst_accuracy": round(worst[1].accuracy, 4) if worst else None,
            "subjects_without_skill": self.subjects_without_skill,
            "implausible_subjects": self.implausible_subjects,
            "works_for_everyone": self.works_for_everyone,
            "per_subject": {
                name: assessment.as_dict() for name, assessment in self.per_subject
            },
        }
