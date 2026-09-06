"""Heart-rate variability from raw beat intervals, with honest confidence.

Every consumer wearable prints one HRV number a day. Almost none of them tell
you how many beats it was computed from, how many of those beats were rejected
as artifact, what correction was applied, or over what window -- and those four
facts change the answer more than any physiological difference between two
adjacent days does.

This module computes the standard time-domain metrics and returns them wrapped
in the evidence needed to judge them, refusing to emit a metric whose inputs
cannot support it rather than emitting a confident-looking wrong number.

References for the correction thresholds: Malik's 20% successive-difference
criterion, and the Karlsson local-mean criterion, both standard in the HRV
literature and both cheap enough to run on-device.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

__all__ = [
    "ArtifactReport",
    "HRVResult",
    "correct_rr",
    "hrv_metrics",
    "rmssd",
    "sdnn",
    "pnn50",
]

CorrectionMethod = Literal["malik", "karlsson", "none"]

#: Beat intervals outside this range are not plausible for a living human at
#: rest or in exercise and are rejected before any correction runs.
RR_HARD_BOUNDS_MS = (250.0, 3000.0)

#: Minimum window lengths, in seconds, below which a metric is not reported.
#: SDNN needs a longer window than RMSSD because it captures slower rhythms;
#: an "SDNN" from a 30-second window is a different quantity than the clinical
#: 5-minute one and comparing them is meaningless.
MIN_WINDOW_S: Dict[str, float] = {
    "rmssd": 30.0,
    "sdnn": 120.0,
    "pnn50": 60.0,
    "mean_rr": 10.0,
    "mean_hr": 10.0,
}

#: Above this fraction of rejected beats, nothing is reported at all.
MAX_ARTIFACT_FRACTION = 0.20


@dataclass
class ArtifactReport:
    """Which beats survived correction, and why the others did not."""

    kept: List[float]
    #: Index into the original series for every kept interval.
    kept_index: List[int]
    rejected_index: List[int]
    reasons: Dict[int, str] = field(default_factory=dict)
    method: CorrectionMethod = "malik"

    @property
    def n_total(self) -> int:
        return len(self.kept) + len(self.rejected_index)

    @property
    def artifact_fraction(self) -> float:
        return (len(self.rejected_index) / self.n_total) if self.n_total else 1.0

    def is_contiguous_pair(self, i: int) -> bool:
        """True when kept intervals ``i`` and ``i+1`` were adjacent originally.

        Successive-difference metrics (RMSSD, pNN50) are only defined across
        genuinely adjacent beats. Computing them across a gap where beats were
        removed manufactures variability that never happened -- a mistake that
        systematically inflates RMSSD exactly on the noisy nights when a user
        is most likely to act on it.
        """
        return (
            i + 1 < len(self.kept_index)
            and self.kept_index[i + 1] == self.kept_index[i] + 1
        )


@dataclass
class HRVResult:
    """Metrics plus everything needed to decide whether to believe them."""

    metrics: Dict[str, float]
    #: Metrics that were requested but withheld, mapped to the reason.
    withheld: Dict[str, str]
    artifact_fraction: float
    n_beats_used: int
    window_s: float
    confidence: float
    method: CorrectionMethod
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.metrics) and self.confidence > 0.0

    def get(self, name: str) -> Optional[float]:
        return self.metrics.get(name)

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": dict(self.metrics),
            "withheld": dict(self.withheld),
            "artifact_fraction": self.artifact_fraction,
            "n_beats_used": self.n_beats_used,
            "window_s": self.window_s,
            "confidence": self.confidence,
            "correction_method": self.method,
            "notes": list(self.notes),
        }


def correct_rr(
    rr_ms: Sequence[float],
    method: CorrectionMethod = "malik",
    threshold: float = 0.20,
) -> ArtifactReport:
    """Flag ectopic and artifact beats in a beat-interval series.

    Args:
        rr_ms: Beat intervals in milliseconds, in temporal order.
        method: ``"malik"`` compares each interval to its predecessor;
            ``"karlsson"`` compares it to the mean of its two neighbours, which
            is less likely to reject a whole run after a single bad beat;
            ``"none"`` applies only the hard physiological bounds.
        threshold: Relative deviation allowed before an interval is rejected.

    Returns:
        An :class:`ArtifactReport`. Nothing is interpolated: rejected beats are
        removed from the analysis set and recorded, so that downstream metrics
        can tell an adjacent pair from a pair that spans a hole.
    """
    rr = [float(v) for v in rr_ms]
    rejected: Dict[int, str] = {}
    low, high = RR_HARD_BOUNDS_MS

    for i, v in enumerate(rr):
        if not math.isfinite(v) or v < low or v > high:
            rejected[i] = "outside_physiological_bounds"

    if method == "malik":
        prev: Optional[float] = None
        for i, v in enumerate(rr):
            if i in rejected:
                continue
            if prev is not None and abs(v - prev) > threshold * prev:
                rejected[i] = "malik_successive_deviation"
            else:
                prev = v
    elif method == "karlsson":
        for i in range(1, len(rr) - 1):
            if i in rejected:
                continue
            a, b = rr[i - 1], rr[i + 1]
            if not (math.isfinite(a) and math.isfinite(b)):
                continue
            local_mean = (a + b) / 2.0
            if local_mean > 0 and abs(rr[i] - local_mean) > threshold * local_mean:
                rejected[i] = "karlsson_local_mean_deviation"

    kept_index = [i for i in range(len(rr)) if i not in rejected]
    return ArtifactReport(
        kept=[rr[i] for i in kept_index],
        kept_index=kept_index,
        rejected_index=sorted(rejected),
        reasons=rejected,
        method=method,
    )


def _successive_diffs(report: ArtifactReport) -> List[float]:
    """Differences between genuinely adjacent surviving beats only."""
    return [
        report.kept[i + 1] - report.kept[i]
        for i in range(len(report.kept) - 1)
        if report.is_contiguous_pair(i)
    ]


def rmssd(
    rr_ms: Sequence[float], method: CorrectionMethod = "malik"
) -> Optional[float]:
    """Root mean square of successive differences, in ms."""
    report = correct_rr(rr_ms, method=method)
    diffs = _successive_diffs(report)
    if len(diffs) < 2:
        return None
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def sdnn(rr_ms: Sequence[float], method: CorrectionMethod = "malik") -> Optional[float]:
    """Standard deviation of surviving beat intervals, in ms."""
    report = correct_rr(rr_ms, method=method)
    if len(report.kept) < 3:
        return None
    return statistics.stdev(report.kept)


def pnn50(
    rr_ms: Sequence[float], method: CorrectionMethod = "malik"
) -> Optional[float]:
    """Percentage of adjacent intervals differing by more than 50 ms."""
    report = correct_rr(rr_ms, method=method)
    diffs = _successive_diffs(report)
    if not diffs:
        return None
    return 100.0 * sum(1 for d in diffs if abs(d) > 50.0) / len(diffs)


def _confidence(
    artifact_fraction: float, window_s: float, n_beats: int
) -> Tuple[float, List[str]]:
    """Monotone trust weight in 0..1, with the reasons it was reduced.

    This is deliberately not a probability. It is a transparent penalty product
    over the three things that actually degrade an HRV estimate: rejected
    beats, a window too short for the rhythms being measured, and too few beats
    for the statistic to be stable.
    """
    notes: List[str] = []
    conf = 1.0

    if artifact_fraction > 0:
        # Linear down to zero at the hard rejection threshold.
        penalty = max(0.0, 1.0 - artifact_fraction / MAX_ARTIFACT_FRACTION)
        conf *= penalty
        notes.append(f"{artifact_fraction:.1%} of beats rejected as artifact")

    if window_s < 300.0:
        # The clinical short-term standard is 5 minutes; shorter windows are
        # usable but noisier, and the penalty says so instead of hiding it.
        conf *= max(0.35, window_s / 300.0)
        notes.append(
            f"window is {window_s:.0f}s, shorter than the 300s short-term standard"
        )

    if n_beats < 120:
        conf *= max(0.3, n_beats / 120.0)
        notes.append(f"only {n_beats} usable beats")

    return max(0.0, min(1.0, conf)), notes


def hrv_metrics(
    rr_ms: Sequence[float],
    method: CorrectionMethod = "malik",
    threshold: float = 0.20,
    max_artifact_fraction: float = MAX_ARTIFACT_FRACTION,
) -> HRVResult:
    """Compute time-domain HRV with per-metric gating.

    A metric is withheld -- returned in ``withheld`` with a reason rather than
    in ``metrics`` -- when the window is too short for it to mean what its name
    implies, or when too few beats survived. This is the behaviour that makes
    the output safe to hand to a language model: the model cannot narrate a
    number that was never produced.
    """
    report = correct_rr(rr_ms, method=method, threshold=threshold)
    window_s = sum(report.kept) / 1000.0
    n = len(report.kept)

    metrics: Dict[str, float] = {}
    withheld: Dict[str, str] = {}
    notes: List[str] = []

    if report.n_total == 0:
        return HRVResult(
            metrics={},
            withheld={k: "no beat intervals supplied" for k in MIN_WINDOW_S},
            artifact_fraction=1.0,
            n_beats_used=0,
            window_s=0.0,
            confidence=0.0,
            method=method,
            notes=["empty input"],
        )

    if report.artifact_fraction > max_artifact_fraction:
        reason = (
            f"{report.artifact_fraction:.1%} of beats rejected, above the "
            f"{max_artifact_fraction:.0%} ceiling"
        )
        return HRVResult(
            metrics={},
            withheld={k: reason for k in MIN_WINDOW_S},
            artifact_fraction=report.artifact_fraction,
            n_beats_used=n,
            window_s=window_s,
            confidence=0.0,
            method=method,
            notes=[reason],
        )

    def gate(name: str, value: Optional[float]) -> None:
        need = MIN_WINDOW_S[name]
        if value is None:
            withheld[name] = "not enough usable adjacent beats"
        elif window_s < need:
            withheld[name] = (
                f"window {window_s:.0f}s is below the {need:.0f}s minimum for {name}"
            )
        else:
            metrics[name] = value

    diffs = _successive_diffs(report)
    gate(
        "rmssd",
        math.sqrt(sum(d * d for d in diffs) / len(diffs)) if len(diffs) >= 2 else None,
    )
    gate("sdnn", statistics.stdev(report.kept) if n >= 3 else None)
    gate(
        "pnn50",
        (100.0 * sum(1 for d in diffs if abs(d) > 50.0) / len(diffs))
        if diffs
        else None,
    )
    mean_rr = statistics.fmean(report.kept) if n else None
    gate("mean_rr", mean_rr)
    gate("mean_hr", (60000.0 / mean_rr) if mean_rr else None)

    conf, conf_notes = _confidence(report.artifact_fraction, window_s, n)
    notes.extend(conf_notes)
    if len(diffs) < max(0, n - 1):
        notes.append(
            f"{n - 1 - len(diffs)} successive-difference pair(s) skipped because "
            "they spanned a rejected beat"
        )

    return HRVResult(
        metrics=metrics,
        withheld=withheld,
        artifact_fraction=report.artifact_fraction,
        n_beats_used=n,
        window_s=window_s,
        confidence=conf,
        method=method,
        notes=notes,
    )
