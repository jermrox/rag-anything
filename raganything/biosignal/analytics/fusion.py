"""Reconciling sources that disagree, instead of picking one and hiding it.

A person wearing a chest strap, a watch, and a ring has three heart rates. Every
platform resolves this with a fixed priority list and shows one number. The
disagreement itself is discarded -- even though it is often the most
informative signal available: two devices that agree to within 2 bpm all night
and then diverge by 25 bpm during a lift have told you exactly which one to
believe, and when.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..schema import Evidence, Modality, Stream
from .quality import QualityReport, assess

__all__ = ["Agreement", "FusionResult", "compare", "reconcile"]


@dataclass
class Agreement:
    """Pairwise comparison of two streams over their overlapping window."""

    a: str
    b: str
    n_pairs: int
    bias: float
    #: 95% limits of agreement (Bland-Altman), as ``(lower, upper)``.
    limits: Tuple[float, float]
    max_abs_difference: float
    overlap_s: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "a": self.a,
            "b": self.b,
            "n_pairs": self.n_pairs,
            "bias": self.bias,
            "limits_of_agreement": list(self.limits),
            "max_abs_difference": self.max_abs_difference,
            "overlap_s": self.overlap_s,
        }


@dataclass
class FusionResult:
    """The chosen stream, why it was chosen, and what the others said."""

    modality: Modality
    chosen: Optional[Stream]
    chosen_reason: str
    quality: Dict[str, QualityReport] = field(default_factory=dict)
    agreements: List[Agreement] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "modality": self.modality.value,
            "chosen": self.chosen.provenance.source_id if self.chosen else None,
            "chosen_reason": self.chosen_reason,
            "quality": {k: v.to_dict() for k, v in self.quality.items()},
            "agreements": [a.to_dict() for a in self.agreements],
            "conflicts": list(self.conflicts),
        }


def _nearest(times: Sequence[float], values: Sequence[float], t: float, tol: float):
    """Nearest value to ``t`` within ``tol`` seconds, or ``None``."""
    if not times:
        return None
    lo, hi = 0, len(times) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if times[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    best = None
    for i in (lo - 1, lo, lo + 1):
        if 0 <= i < len(times):
            d = abs(times[i] - t)
            if d <= tol and (best is None or d < best[0]):
                best = (d, values[i])
    return None if best is None else best[1]


def compare(a: Stream, b: Stream, tolerance_s: float = 2.0) -> Optional[Agreement]:
    """Bland-Altman style agreement between two streams of the same modality."""
    if a.modality != b.modality:
        raise ValueError("compare() requires two streams of the same modality")

    a_sorted = a.sorted()
    b_sorted = b.sorted()
    b_times = b_sorted.times()
    b_values = b_sorted.values()

    diffs: List[float] = []
    matched_times: List[float] = []
    for s in a_sorted.samples:
        other = _nearest(b_times, b_values, s.t, tolerance_s)
        if other is not None:
            diffs.append(s.value - other)
            matched_times.append(s.t)

    if len(diffs) < 3:
        return None

    bias = statistics.fmean(diffs)
    sd = statistics.pstdev(diffs)
    return Agreement(
        a=a.provenance.source_id,
        b=b.provenance.source_id,
        n_pairs=len(diffs),
        bias=bias,
        limits=(bias - 1.96 * sd, bias + 1.96 * sd),
        max_abs_difference=max(abs(d) for d in diffs),
        overlap_s=matched_times[-1] - matched_times[0],
    )


def reconcile(
    streams: Sequence[Stream],
    window: Optional[Tuple[float, float]] = None,
    tolerance_s: float = 2.0,
    conflict_threshold: float = 10.0,
) -> FusionResult:
    """Choose among competing streams and report the disagreement.

    Selection is by evidence class first (a measurement beats a vendor score),
    then by measured signal quality, then by latency. The alternatives are not
    thrown away: their quality reports and pairwise agreement with the winner
    are returned, and any pair whose limits of agreement exceed
    ``conflict_threshold`` is called out explicitly.
    """
    if not streams:
        raise ValueError("reconcile() requires at least one stream")
    modality = streams[0].modality
    if any(s.modality != modality for s in streams):
        raise ValueError("reconcile() requires streams of a single modality")

    quality = {s.provenance.source_id: assess(s, window=window) for s in streams}

    def key(stream: Stream):
        best_evidence = min(
            (Evidence.rank(s.evidence) for s in stream.samples),
            default=Evidence.rank(Evidence.IMPUTED),
        )
        return (
            best_evidence,
            -quality[stream.provenance.source_id].score,
            stream.provenance.latency_s,
        )

    ranked = sorted(streams, key=key)
    chosen = ranked[0]
    chosen_quality = quality[chosen.provenance.source_id]
    reason = (
        f"{chosen.provenance.source_id} ({chosen.provenance.device}) selected: "
        f"evidence {min((s.evidence.value for s in chosen.samples), default='none')}, "
        f"quality {chosen_quality.score:.2f}, "
        f"latency {chosen.provenance.latency_s:.0f}s"
    )

    agreements: List[Agreement] = []
    conflicts: List[str] = []
    for other in ranked[1:]:
        ag = compare(chosen, other, tolerance_s=tolerance_s)
        if ag is None:
            conflicts.append(
                f"{other.provenance.source_id} could not be compared with "
                f"{chosen.provenance.source_id}: no overlapping samples within "
                f"{tolerance_s:.0f}s"
            )
            continue
        agreements.append(ag)
        spread = ag.limits[1] - ag.limits[0]
        if abs(ag.bias) > conflict_threshold or spread > 2 * conflict_threshold:
            conflicts.append(
                f"{ag.a} and {ag.b} disagree on {modality.value}: bias "
                f"{ag.bias:+.1f}, limits of agreement "
                f"[{ag.limits[0]:+.1f}, {ag.limits[1]:+.1f}], worst case "
                f"{ag.max_abs_difference:.1f}"
            )

    return FusionResult(
        modality=modality,
        chosen=chosen,
        chosen_reason=reason,
        quality=quality,
        agreements=agreements,
        conflicts=conflicts,
    )
