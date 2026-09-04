"""Learned sleep staging from RR-interval epochs.

**Read this before trusting a number from this module.** A stager trained on
``ble/simulator.py`` output has learned the simulator's assumptions -- its
90-minute cycles, its stage-dependent RSA gains, its noise model -- and not
human physiology. It will score very well on held-out simulated nights, and
that accuracy says nothing about a real wrist. Models fitted here are stamped
``training_data="synthetic"`` and carry that caveat with every prediction.

What this module is genuinely for: proving the pipeline end to end -- epoching,
class balance, subject-level evaluation, persistence, and the path from a
prediction into the knowledge graph -- so that when real polysomnography data
arrives (Sleep-EDF, MESA, PhysioNet) only the data source changes.

The one methodological point that carries over unchanged is
:func:`subject_split`. Splitting epochs at random puts adjacent 30-second
windows of the *same* night on both sides of the split. Neighbouring epochs are
nearly identical, so the model is scored on data it has effectively seen, and
accuracy comes out far higher than it should. Splitting by night is the only
honest evaluation, and this module refuses to evaluate any other way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Sequence, Tuple

from ..biometrics.schema import SleepStage
from .epochs import EPOCH_FEATURE_NAMES, EpochSample, labelled_only
from .metrics import (
    IMPLAUSIBLE_ACCURACY,
    CrossValidationSummary,
    SkillAssessment,
    assess_skill,
)
from .registry import SYNTHETIC, ModelCard, require_sklearn

STAGE_NAMES = {
    SleepStage.AWAKE.value: "awake",
    SleepStage.LIGHT.value: "light",
    SleepStage.DEEP.value: "deep",
    SleepStage.REM.value: "rem",
}

#: Distinct nights required before training is allowed. Below this the model is
#: fitting one person's one night and cannot generalise even in principle.
MIN_TRAINING_NIGHTS = 3

#: The implausibility ceiling is imported from :mod:`vitalgraph.ml.metrics`
#: rather than defined here. It was defined in both, which is how two copies of
#: one threshold drift apart -- and it became a parameter once sleep/wake
#: needed a different value, so a second copy would have silently kept the old
#: one. Four-class staging from RR intervals reaches roughly 70-80% agreement
#: with polysomnography; a result far above that means the labels were
#: derivable from the features by construction, not that the model is good.

#: Published range for RR-only four-class staging, for context in reports.
PLAUSIBLE_ACCURACY_RANGE = (0.70, 0.80)


class NotEnoughNights(RuntimeError):
    """Raised when training or splitting has too few distinct nights."""


@dataclass(frozen=True, slots=True)
class StagingEvaluation:
    """Held-out performance, measured across nights the model never saw."""

    accuracy: float
    per_stage: Dict[str, Dict[str, float]]
    n_test_epochs: int
    test_nights: Tuple[str, ...]
    caveat: str
    skill: SkillAssessment | None = None
    """Accuracy against the majority-class baseline, with kappa.

    Optional only so an evaluation can still be constructed without training
    labels to hand; every evaluation produced by :meth:`SleepStager.evaluate`
    carries one. A high accuracy with no skill is the failure this exists to
    surface, and it is invisible in the accuracy alone."""

    @property
    def is_implausible(self) -> bool:
        """Whether the accuracy is too good to be real.

        See :data:`IMPLAUSIBLE_ACCURACY`. A True here is a finding about the
        experiment, not a compliment to the model.
        """
        return self.accuracy >= IMPLAUSIBLE_ACCURACY

    def plausibility_note(self) -> str:
        lo, hi = PLAUSIBLE_ACCURACY_RANGE
        if not self.is_implausible:
            return (
                f"Accuracy {self.accuracy:.1%} is within the range reported for "
                f"RR-only four-class staging ({lo:.0%}-{hi:.0%})."
            )
        return (
            f"Accuracy {self.accuracy:.1%} is IMPLAUSIBLE for RR-only four-class "
            f"staging, where published agreement with polysomnography is "
            f"{lo:.0%}-{hi:.0%}. This indicates the labels were recoverable from "
            f"the features by construction -- synthetic training data, a leaky "
            f"split, or a feature encoding the label -- not a superior model. "
            f"Do not ship this as a capability."
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "per_stage": self.per_stage,
            "n_test_epochs": self.n_test_epochs,
            "test_nights": list(self.test_nights),
            "caveat": self.caveat,
            "is_implausible": self.is_implausible,
            "plausibility": self.plausibility_note(),
            "skill": self.skill.as_dict() if self.skill else None,
        }


def subject_split(
    samples: Sequence[EpochSample], test_fraction: float = 0.3, seed: int = 0
) -> Tuple[List[EpochSample], List[EpochSample]]:
    """Split epochs by night, never within one.

    Random epoch-level splitting leaks: adjacent 30-second windows of the same
    night are nearly identical, so the model is tested on what it trained on
    and accuracy is inflated, often dramatically.

    Raises:
        NotEnoughNights: with fewer than two distinct nights, since no honest
            split exists.
    """
    nights = sorted({s.night_id for s in samples})
    if len(nights) < 2:
        raise NotEnoughNights(
            f"need at least 2 distinct nights to split by subject, got {len(nights)}"
        )

    rng = random.Random(seed)
    shuffled = list(nights)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_fraction))
    test_nights = set(shuffled[:n_test])

    train = [s for s in samples if s.night_id not in test_nights]
    test = [s for s in samples if s.night_id in test_nights]
    return train, test


class SleepStager:
    """Random-forest classifier over RR-derived epoch features.

    A forest suits this: the classes are imbalanced (light sleep dominates),
    the features interact non-linearly, and it needs no scaling.
    """

    def __init__(self, seed: int = 0, n_estimators: int = 200) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self._model: Any = None
        self._nights: Tuple[str, ...] = ()
        self._train_labels: Tuple[float, ...] = ()
        """Training label distribution, kept so the majority-class baseline
        can be computed from training data rather than from the test set.
        Taking the majority class from test would let the baseline see the
        answers and make it unbeatable, inverting the comparison."""

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(
        self, samples: Sequence[EpochSample], training_data: str = SYNTHETIC
    ) -> ModelCard:
        """Train on labelled epochs drawn from several nights."""
        labelled = labelled_only(samples)
        nights = sorted({s.night_id for s in labelled})
        if len(nights) < MIN_TRAINING_NIGHTS:
            raise NotEnoughNights(
                f"need epochs from at least {MIN_TRAINING_NIGHTS} distinct nights, "
                f"got {len(nights)}"
            )

        require_sklearn()
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier

        x = np.asarray([s.values for s in labelled], dtype=float)
        y = np.asarray([s.label for s in labelled], dtype=float)

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            class_weight="balanced",
            min_samples_leaf=3,
        ).fit(x, y)
        self._nights = tuple(nights)
        self._train_labels = tuple(float(s.label) for s in labelled)

        return ModelCard(
            name="sleep-stager",
            algorithm="RandomForestClassifier",
            training_data=training_data,
            seed=self.seed,
            feature_names=EPOCH_FEATURE_NAMES,
            n_training_samples=len(labelled),
            notes=(
                f"Trained on {len(nights)} night(s) of RR-derived 30 s epochs. "
                "Evaluate only with subject-level splits."
            ),
        )

    def predict(self, samples: Sequence[EpochSample]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("stager has not been fitted")
        import numpy as np

        x = np.asarray([s.values for s in samples], dtype=float)
        return [float(v) for v in self._model.predict(x)]

    def predict_named(self, samples: Sequence[EpochSample]) -> List[str]:
        return [STAGE_NAMES.get(v, "unknown") for v in self.predict(samples)]

    def evaluate(
        self, test: Sequence[EpochSample], training_data: str = SYNTHETIC
    ) -> StagingEvaluation:
        """Score against held-out nights.

        Refuses to score against nights the model was trained on, which would
        make the result meaningless.
        """
        labelled = labelled_only(test)
        if not labelled:
            raise ValueError("no labelled epochs to evaluate against")

        test_nights = sorted({s.night_id for s in labelled})
        overlap = set(test_nights) & set(self._nights)
        if overlap:
            raise NotEnoughNights(
                f"test set overlaps training nights {sorted(overlap)}; "
                "evaluation would be meaningless"
            )

        predicted = self.predict(labelled)
        actual = [s.label for s in labelled]
        correct = sum(1 for p, a in zip(predicted, actual) if p == a)

        per_stage: Dict[str, Dict[str, float]] = {}
        for value, name in STAGE_NAMES.items():
            true_positive = sum(
                1 for p, a in zip(predicted, actual) if p == value and a == value
            )
            predicted_count = sum(1 for p in predicted if p == value)
            actual_count = sum(1 for a in actual if a == value)
            if not actual_count and not predicted_count:
                continue
            per_stage[name] = {
                "precision": round(true_positive / predicted_count, 4)
                if predicted_count
                else 0.0,
                "recall": round(true_positive / actual_count, 4)
                if actual_count
                else 0.0,
                "support": actual_count,
            }

        caveat = (
            "Measured on held-out nights, but the training data is simulated: "
            "this figure reflects how well the model learned the simulator, not "
            "how it would perform on a real wrist."
            if training_data == SYNTHETIC
            else "Measured on held-out nights."
        )
        return StagingEvaluation(
            accuracy=correct / len(labelled),
            per_stage=per_stage,
            n_test_epochs=len(labelled),
            test_nights=tuple(test_nights),
            caveat=caveat,
            skill=assess_skill(predicted, actual, self._train_labels),
        )


#: Sleep and wake, the two-class reduction.
SLEEP_WAKE_NAMES = {0.0: "wake", 1.0: "sleep"}


def collapse_to_sleep_wake(samples: Sequence[EpochSample]) -> List[EpochSample]:
    """Relabel epochs as wake (0) or asleep (1), leaving features untouched.

    Four-class staging from RR intervals alone is a hard problem and our
    features do not solve it. Sleep/wake is a genuinely easier question and the
    one the wrist-staging literature reports usable numbers for, so asking it
    separately says whether the failure is the task or the feature set --
    which is worth knowing before adding features.

    This is a narrower claim, not a repaired model. A sleep/wake classifier
    cannot report time in deep sleep or REM, and must never be presented as
    though it could.
    """
    out: List[EpochSample] = []
    for sample in samples:
        if sample.label is None:
            out.append(sample)
            continue
        asleep = 0.0 if sample.label == SleepStage.AWAKE.value else 1.0
        out.append(replace(sample, label=asleep))
    return out


def leave_one_night_out(
    samples: Sequence[EpochSample],
    seed: int = 0,
    n_estimators: int = 200,
    implausible_above: float = IMPLAUSIBLE_ACCURACY,
) -> CrossValidationSummary:
    """Score every night by training on all the others.

    A single held-out night gives one number with no sense of whether it was
    lucky. Rotating the held-out night uses all the data and produces a
    distribution, which is the only way to see that a model works for most
    people and fails completely for some -- the case a mean conceals and the
    one a health product is actually judged on.

    This mirrors the leave-one-subject-out design used in the wrist-staging
    literature with the same sensor set. Nights that cannot be trained around,
    because removing them leaves too few, are skipped rather than scored on a
    model that could not be fitted.
    """
    labelled = labelled_only(samples)
    nights = sorted({s.night_id for s in labelled})
    if len(nights) <= MIN_TRAINING_NIGHTS:
        raise NotEnoughNights(
            f"leave-one-night-out needs more than {MIN_TRAINING_NIGHTS} nights "
            f"so each fold still has enough to train on, got {len(nights)}"
        )

    results: List[Tuple[str, SkillAssessment]] = []
    for held_out in nights:
        train = [s for s in labelled if s.night_id != held_out]
        test = [s for s in labelled if s.night_id == held_out]
        if not test:
            continue
        stager = SleepStager(seed=seed, n_estimators=n_estimators)
        stager.fit(train)
        predicted = stager.predict(test)
        actual = [s.label for s in test]
        results.append(
            (
                held_out,
                assess_skill(
                    predicted,
                    actual,
                    stager._train_labels,
                    implausible_above=implausible_above,
                ),
            )
        )

    return CrossValidationSummary(per_subject=tuple(results))


def to_content_list(
    evaluation: StagingEvaluation, card: ModelCard
) -> List[Dict[str, Any]]:
    """Render a staging model's performance for the knowledge graph."""
    rows = ["| Stage | Precision | Recall | Epochs |", "| --- | --- | --- | --- |"]
    for name, m in evaluation.per_stage.items():
        rows.append(
            f"| {name} | {m['precision']:.2f} | {m['recall']:.2f} | {int(m['support'])} |"
        )

    narrative = (
        f"Sleep staging model {card.qualified_name} ({card.algorithm}) reached "
        f"{evaluation.accuracy:.1%} accuracy across {evaluation.n_test_epochs} "
        f"held-out epochs from {len(evaluation.test_nights)} night(s) it never saw. "
        f"{card.caveat()} {evaluation.plausibility_note()}"
    )
    return [
        {"type": "text", "text": narrative, "page_idx": 0},
        {
            "type": "table",
            "table_body": "\n".join(rows),
            "table_caption": [f"Per-stage performance for {card.qualified_name}"],
            "table_footnote": [f"{evaluation.caveat} {evaluation.plausibility_note()}"],
            "page_idx": 0,
        },
    ]
