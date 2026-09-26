"""Signal quality assessment -- the layer every fitness app is missing.

A wearable that lost contact for eleven minutes and a wearable that recorded
continuously produce charts that look identical, because the gap is drawn as a
straight line and never mentioned again. Every conclusion built on top -- the
recovery score, the sleep stage, the trend arrow -- inherits an error nobody
downstream can see.

This module makes the invisible visible: it measures how much of a window was
actually observed, how badly the sampling drifted, how much was flagged at
decode time, and how stale the data is relative to the question being asked.
The result is a single trust weight plus the plain-language reasons behind it,
suitable both for gating analytics and for telling a user why their number is
missing today.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..schema import DEFECT_FLAGS, Evidence, Stream

__all__ = ["Gap", "QualityReport", "assess", "gate"]


@dataclass(frozen=True)
class Gap:
    """A stretch of time in which nothing was recorded."""

    start: float
    end: float

    @property
    def duration_s(self) -> float:
        return self.end - self.start


@dataclass
class QualityReport:
    """What was actually observed, versus what the chart would imply."""

    stream_modality: str
    n_samples: int
    window_s: float
    #: Fraction of the window covered by samples at the expected rate, 0..1.
    coverage: float
    gaps: List[Gap]
    longest_gap_s: float
    #: Coefficient of variation of inter-sample intervals. High values mean the
    #: transport, not the physiology, is shaping the series.
    jitter: float
    #: Fraction of samples flagged as degraded (lost contact, out of range).
    #: Provenance-only flags are excluded -- see :data:`~..schema.DEFECT_FLAGS`.
    flagged_fraction: float
    #: Fraction of samples that are not direct measurements.
    inferred_fraction: float
    staleness_s: float
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        return self.score >= 0.5

    def to_dict(self) -> Dict[str, object]:
        return {
            "modality": self.stream_modality,
            "n_samples": self.n_samples,
            "window_s": self.window_s,
            "coverage": self.coverage,
            "n_gaps": len(self.gaps),
            "longest_gap_s": self.longest_gap_s,
            "jitter": self.jitter,
            "flagged_fraction": self.flagged_fraction,
            "inferred_fraction": self.inferred_fraction,
            "staleness_s": self.staleness_s,
            "score": self.score,
            "reasons": list(self.reasons),
        }


def _expected_interval(stream: Stream, samples_t: List[float]) -> float:
    """Sampling interval to judge gaps against.

    Prefer what the source claims; otherwise use the median observed interval,
    which is robust to the gaps we are trying to find in the first place.
    """
    hz = stream.provenance.nominal_hz
    if hz:
        return 1.0 / hz
    if len(samples_t) < 3:
        return 1.0
    deltas = [b - a for a, b in zip(samples_t, samples_t[1:]) if b > a]
    return statistics.median(deltas) if deltas else 1.0


def assess(
    stream: Stream,
    window: Optional[Tuple[float, float]] = None,
    now: Optional[float] = None,
    gap_factor: float = 3.0,
) -> QualityReport:
    """Score one stream's fitness for analysis.

    Args:
        stream: The stream to assess.
        window: ``(start, end)`` the stream is supposed to cover. Defaults to
            the stream's own extent, which can only detect internal gaps -- pass
            the session window to also catch a sensor that connected late or
            dropped out before the end.
        now: Reference time for staleness. Defaults to the window end.
        gap_factor: An interval longer than this multiple of the expected
            sampling interval counts as a gap.
    """
    samples = sorted(stream.samples, key=lambda s: s.t)
    times = [s.t for s in samples]

    if window is not None:
        w_start, w_end = window
    elif times:
        w_start, w_end = times[0], times[-1]
    else:
        w_start = w_end = 0.0
    window_s = max(0.0, w_end - w_start)

    if not samples:
        return QualityReport(
            stream_modality=stream.modality.value,
            n_samples=0,
            window_s=window_s,
            coverage=0.0,
            gaps=[Gap(w_start, w_end)] if window_s > 0 else [],
            longest_gap_s=window_s,
            jitter=0.0,
            flagged_fraction=0.0,
            inferred_fraction=0.0,
            staleness_s=window_s,
            score=0.0,
            reasons=["stream contains no samples"],
        )

    expected = _expected_interval(stream, times)
    gap_threshold = expected * gap_factor

    gaps: List[Gap] = []
    if window is not None:
        if times[0] - w_start > gap_threshold:
            gaps.append(Gap(w_start, times[0]))
        if w_end - times[-1] > gap_threshold:
            gaps.append(Gap(times[-1], w_end))
    for a, b in zip(times, times[1:]):
        if b - a > gap_threshold:
            gaps.append(Gap(a, b))

    gap_total = sum(g.duration_s for g in gaps)
    longest_gap = max((g.duration_s for g in gaps), default=0.0)
    coverage = 1.0 if window_s <= 0 else max(0.0, 1.0 - gap_total / window_s)

    deltas = [b - a for a, b in zip(times, times[1:])]
    inliers = [d for d in deltas if 0 < d <= gap_threshold]
    if len(inliers) >= 2:
        mean_d = statistics.fmean(inliers)
        jitter = (statistics.pstdev(inliers) / mean_d) if mean_d > 0 else 0.0
    else:
        jitter = 0.0

    flagged = sum(1 for s in samples if DEFECT_FLAGS.intersection(s.flags)) / len(
        samples
    )
    inferred = sum(1 for s in samples if s.evidence is not Evidence.MEASURED) / len(
        samples
    )

    reference = now if now is not None else w_end
    staleness = max(0.0, reference - times[-1])

    reasons: List[str] = []
    score = 1.0

    if coverage < 1.0:
        score *= coverage
        reasons.append(
            f"{(1 - coverage):.1%} of the window has no data "
            f"({len(gaps)} gap(s), longest {longest_gap:.0f}s)"
        )
    if jitter > 0.5:
        score *= max(0.4, 1.0 - (jitter - 0.5))
        reasons.append(
            f"sampling interval varies by {jitter:.0%} -- transport jitter is "
            "shaping the series"
        )
    if flagged > 0:
        score *= max(0.0, 1.0 - flagged)
        reasons.append(
            f"{flagged:.1%} of samples were flagged as degraded by the decoder"
        )
    if inferred > 0:
        # Inference is not disqualifying, but it is not measurement either.
        score *= max(0.5, 1.0 - 0.5 * inferred)
        reasons.append(f"{inferred:.1%} of samples are inferred, not measured")
    if stream.provenance.latency_s > 3600:
        score *= 0.9
        reasons.append(
            f"source delivers with ~{stream.provenance.latency_s / 3600:.1f}h latency"
        )
    if not stream.provenance.documented:
        score *= 0.8
        reasons.append(
            f"derivation is undocumented (algorithm: {stream.provenance.algorithm})"
        )
    if not reasons:
        reasons.append("continuous, unflagged, directly measured")

    return QualityReport(
        stream_modality=stream.modality.value,
        n_samples=len(samples),
        window_s=window_s,
        coverage=coverage,
        gaps=gaps,
        longest_gap_s=longest_gap,
        jitter=jitter,
        flagged_fraction=flagged,
        inferred_fraction=inferred,
        staleness_s=staleness,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
    )


def gate(
    value: Optional[float],
    report: QualityReport,
    min_score: float = 0.5,
    label: str = "metric",
) -> Tuple[Optional[float], str]:
    """Return ``(value, explanation)``, suppressing the value if unsupported.

    The explanation is always populated, so a refusal is as informative as a
    result. This is what stops a downstream model from confidently narrating a
    trend that is really a dead battery.
    """
    if value is None:
        return None, f"{label} could not be computed from the available data"
    if report.score < min_score:
        return None, (
            f"{label} withheld: signal quality {report.score:.2f} is below the "
            f"{min_score:.2f} threshold ({'; '.join(report.reasons)})"
        )
    return value, (
        f"{label} computed at signal quality {report.score:.2f} "
        f"({'; '.join(report.reasons)})"
    )
