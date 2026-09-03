"""Multivariate anomaly detection against a personal baseline.

Single-metric thresholds miss the nights that matter. A resting heart rate of
62 is unremarkable, and an RMSSD of 40 ms is unremarkable, but the two together
in someone whose norm is 54 bpm and 55 ms is the shape of an oncoming illness.
This module scores the whole feature vector jointly.

Two properties matter more than the raw score:

* **Cold start is explicit.** Below :data:`MIN_TRAINING_NIGHTS` of personal
  history there is no baseline to be anomalous against, and the detector says
  so rather than inventing a number.
* **Every flag is attributable.** A score alone is not actionable, so each
  result names the features that drove it, as robust deviations from the
  person's own median.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .features import FEATURE_NAMES, FeatureVector
from .registry import SYNTHETIC, ModelCard, require_sklearn

#: Nights of personal history required before scoring means anything. Matches
#: the HRV baseline requirement so "enough history" is one idea in one product.
MIN_TRAINING_NIGHTS = 7

#: Robust z above which a feature is named as a contributor.
CONTRIBUTOR_Z = 1.5

#: Scale factor making MAD a consistent estimator of standard deviation for
#: normally distributed data.
MAD_TO_SIGMA = 1.4826

#: Absolute floor on the MAD-derived scale. Without it, a feature identical
#: across a short baseline produces an infinite deviation from a trivial change.
MIN_SCALE = 1e-6

#: Relative floor: the scale is never taken below this fraction of the
#: feature's own typical magnitude. Features arrive in wildly different units
#: (milliseconds, bpm, unit fractions), so a single absolute floor cannot serve
#: them all, whereas "2% of the median" is meaningful for every one.
#:
#: This is the same failure the HRV baseline had: an unusually consistent
#: stretch shrinks the spread until an ordinary difference reads as a
#: fifteen-sigma event. Tightly clustered baselines are common in practice --
#: and universal in simulated data.
MIN_RELATIVE_SCALE = 0.02

#: Reported deviations are capped here. Past this point the baseline spread is
#: too small to quantify further, and a larger number would imply a precision
#: the data does not support. Attribution is a hint about which features moved,
#: not a statistical claim about how improbable the movement was.
MAX_REPORTED_Z = 10.0


@dataclass(frozen=True, slots=True)
class AnomalyResult:
    """One period's anomaly assessment."""

    period_id: str
    score: float
    """0.0-1.0, higher is more anomalous."""
    is_anomalous: bool
    contributors: Tuple[Tuple[str, float], ...] = field(default_factory=tuple)
    """(feature name, robust z) pairs, largest deviation first."""
    baseline_nights: int = 0
    caveat: str = ""

    def explain(self) -> str:
        """Plain-language reason, suitable for insertion into the graph."""
        if not self.contributors:
            return f"Anomaly score {self.score:.2f}; no single feature dominates."
        parts = [
            f"{name} {'above' if z > 0 else 'below'} the personal median "
            f"({abs(z):.1f} robust SD)"
            for name, z in self.contributors[:4]
        ]
        return f"Anomaly score {self.score:.2f}, driven by " + ", ".join(parts) + "."

    def as_dict(self) -> Dict[str, Any]:
        return {
            "period_id": self.period_id,
            "score": round(self.score, 4),
            "is_anomalous": self.is_anomalous,
            "contributors": [
                {"feature": n, "robust_z": round(z, 3)} for n, z in self.contributors
            ],
            "baseline_nights": self.baseline_nights,
            "explanation": self.explain(),
            "caveat": self.caveat,
        }


class InsufficientBaseline(RuntimeError):
    """Raised when there is not enough personal history to fit a detector."""


class PersonalAnomalyDetector:
    """Isolation Forest over a person's own history.

    Isolation Forest suits this problem: it needs no labelled anomalies (there
    are none), it is robust to the mixed scales in the feature vector without
    standardisation, and it is cheap to refit as history grows.
    """

    def __init__(self, contamination: float = 0.1, seed: int = 0) -> None:
        self.contamination = contamination
        self.seed = seed
        self._model: Any = None
        self._median: List[float] = []
        self._scale: List[float] = []
        self._n_training = 0

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def fit(self, vectors: Sequence[FeatureVector]) -> ModelCard:
        """Fit against a person's own history.

        Raises:
            InsufficientBaseline: with fewer than MIN_TRAINING_NIGHTS vectors.
        """
        if len(vectors) < MIN_TRAINING_NIGHTS:
            raise InsufficientBaseline(
                f"need at least {MIN_TRAINING_NIGHTS} nights to establish a "
                f"personal baseline, got {len(vectors)}"
            )
        require_sklearn()
        import numpy as np
        from sklearn.ensemble import IsolationForest

        matrix = np.asarray([v.values for v in vectors], dtype=float)

        # Robust centre and scale, used for attribution rather than for the
        # model itself. Median and MAD resist the very outliers being hunted.
        self._median = np.median(matrix, axis=0).tolist()
        mad = np.median(np.abs(matrix - np.asarray(self._median)), axis=0)
        relative_floor = np.abs(np.asarray(self._median)) * MIN_RELATIVE_SCALE
        self._scale = np.maximum(
            np.maximum(mad * MAD_TO_SIGMA, relative_floor), MIN_SCALE
        ).tolist()

        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.seed,
            n_estimators=200,
        ).fit(matrix)
        self._n_training = len(vectors)

        return ModelCard(
            name="personal-anomaly",
            algorithm="IsolationForest",
            training_data=SYNTHETIC,
            seed=self.seed,
            n_training_samples=len(vectors),
            metrics={"contamination": self.contamination},
            notes=(
                "Fitted on one person's own history; scores are meaningful only "
                "relative to that person."
            ),
        )

    def _contributors(self, vector: FeatureVector) -> Tuple[Tuple[str, float], ...]:
        pairs: List[Tuple[str, float]] = []
        for name, value, med, scale in zip(
            FEATURE_NAMES, vector.values, self._median, self._scale
        ):
            z = (value - med) / scale
            z = max(-MAX_REPORTED_Z, min(MAX_REPORTED_Z, z))
            if abs(z) >= CONTRIBUTOR_Z:
                pairs.append((name, z))
        pairs.sort(key=lambda kv: -abs(kv[1]))
        return tuple(pairs)

    def score(self, vector: FeatureVector) -> AnomalyResult:
        """Score one period against the fitted baseline."""
        if not self.is_fitted:
            raise InsufficientBaseline("detector has not been fitted")
        import numpy as np

        row = np.asarray([vector.values], dtype=float)
        # decision_function is positive for inliers; invert and squash to 0-1
        # so "higher is more anomalous" reads the way a reader expects.
        raw = float(self._model.decision_function(row)[0])
        score = 1.0 / (1.0 + pow(2.718281828459045, 4.0 * raw))
        flagged = bool(self._model.predict(row)[0] == -1)

        return AnomalyResult(
            period_id=vector.period_id,
            score=score,
            is_anomalous=flagged,
            contributors=self._contributors(vector),
            baseline_nights=self._n_training,
            caveat=(
                "Relative to this person's own history only; not a population "
                "norm and not a diagnosis."
            ),
        )

    def score_many(self, vectors: Sequence[FeatureVector]) -> List[AnomalyResult]:
        return [self.score(v) for v in vectors]


def detect_with_baseline(
    vectors: Sequence[FeatureVector], seed: int = 0
) -> List[AnomalyResult]:
    """Score each period against the history preceding it.

    Fitting on all nights including the one being scored would let an anomaly
    teach the model that it is normal, which is how a slow decline becomes
    invisible. Each night is therefore judged only against its own past, and
    nights before a baseline exists are simply not scored.
    """
    results: List[AnomalyResult] = []
    for index in range(MIN_TRAINING_NIGHTS, len(vectors)):
        detector = PersonalAnomalyDetector(seed=seed)
        detector.fit(vectors[:index])
        results.append(detector.score(vectors[index]))
    return results


def to_content_list(results: Sequence[AnomalyResult]) -> List[Dict[str, Any]]:
    """Render anomaly findings as knowledge-graph content.

    Only flagged periods are emitted: a table of ordinary nights would dilute
    retrieval, and the value is in what stood out.
    """
    flagged = [r for r in results if r.is_anomalous]
    if not flagged:
        return []

    rows = ["| Period | Score | Driven by |", "| --- | --- | --- |"]
    for r in flagged:
        drivers = (
            ", ".join(f"{n} ({z:+.1f})" for n, z in r.contributors[:3])
            or "no single feature"
        )
        rows.append(f"| {r.period_id} | {r.score:.2f} | {drivers} |")

    narrative = "\n".join(
        f"Period {r.period_id} was flagged as anomalous. {r.explain()}" for r in flagged
    )

    return [
        {"type": "text", "text": narrative, "page_idx": 0},
        {
            "type": "table",
            "table_body": "\n".join(rows),
            "table_caption": [
                f"{len(flagged)} period(s) flagged against the personal baseline"
            ],
            "table_footnote": [
                "Scored against this person's own history, not a population norm. "
                "Robust z is deviation from their median in MAD-derived units. "
                "Wellness signal, not a diagnosis."
            ],
            "page_idx": 0,
        },
    ]
