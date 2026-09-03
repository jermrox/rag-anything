"""Per-epoch features for sleep staging.

Sleep is scored in 30-second epochs by convention, so staging works at that
resolution rather than per night. Features are derived from RR intervals alone,
which is deliberate: many wrist devices expose heart rate and RR through the
standard GATT characteristic but no usable accelerometer stream, so an RR-only
stager is the one that works on the widest hardware.

Pure standard library, like ``features.py`` and for the same reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

from ..biometrics.schema import SignalType, SleepStage
from ..biometrics.store import BiometricStore

#: Standard sleep-scoring epoch.
EPOCH_SECONDS = 30.0

#: Ordered epoch feature vocabulary. Append only, for the same reason as
#: ``features.FEATURE_NAMES``.
EPOCH_FEATURE_NAMES: Tuple[str, ...] = (
    "mean_rr_ms",
    "sd_rr_ms",
    "rmssd_ms",
    "min_rr_ms",
    "max_rr_ms",
    "rr_range_ms",
    "n_beats",
    # Context relative to the night, which is what separates deep from REM far
    # better than any absolute value: stage depends on where you are in the night.
    "rr_vs_night_median",
    "rmssd_vs_night_median",
    "elapsed_fraction",
    # Local dynamics
    "delta_prev_mean_rr",
    "delta_next_mean_rr",
)

#: Epochs with fewer beats than this are unscoreable and are dropped rather
#: than imputed.
MIN_BEATS_PER_EPOCH = 10


@dataclass(frozen=True, slots=True)
class EpochSample:
    """One 30-second epoch: its features and, when known, its true stage."""

    night_id: str
    index: int
    start: datetime
    values: Tuple[float, ...]
    label: float | None = None
    """SleepStage value when ground truth is available, else None."""

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(EPOCH_FEATURE_NAMES, self.values))


def _stats(rr: Sequence[float]) -> Tuple[float, float, float, float, float]:
    n = len(rr)
    mean = sum(rr) / n
    variance = sum((x - mean) ** 2 for x in rr) / (n - 1) if n > 1 else 0.0
    diffs = [rr[i + 1] - rr[i] for i in range(n - 1)]
    rmssd = math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else 0.0
    return mean, math.sqrt(variance), rmssd, min(rr), max(rr)


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def epoch_samples(
    store: BiometricStore, start: datetime, end: datetime, night_id: str
) -> List[EpochSample]:
    """Build labelled epoch samples for one night.

    Labels come from stored ``SLEEP_STAGE`` samples when present; epochs
    without one are returned with ``label=None`` so the same function serves
    both training and inference.
    """
    rr = store.samples_in(SignalType.RR_INTERVAL, start, end)
    if not rr:
        return []

    stage_samples = store.samples_in(SignalType.SLEEP_STAGE, start, end)
    stage_by_epoch: Dict[int, float] = {}
    for s in stage_samples:
        idx = int((s.ts - start).total_seconds() // EPOCH_SECONDS)
        stage_by_epoch.setdefault(idx, s.value)

    # Bucket beats by epoch.
    buckets: Dict[int, List[float]] = {}
    for sample in rr:
        idx = int((sample.ts - start).total_seconds() // EPOCH_SECONDS)
        buckets.setdefault(idx, []).append(sample.value)

    usable = {i: v for i, v in buckets.items() if len(v) >= MIN_BEATS_PER_EPOCH}
    if not usable:
        return []

    night_median_rr = _median([x for v in usable.values() for x in v])
    per_epoch_rmssd = {i: _stats(v)[2] for i, v in usable.items()}
    night_median_rmssd = _median(list(per_epoch_rmssd.values())) or 1.0
    total_epochs = max(usable) + 1

    ordered = sorted(usable)
    means = {i: sum(usable[i]) / len(usable[i]) for i in ordered}

    out: List[EpochSample] = []
    for position, idx in enumerate(ordered):
        beats = usable[idx]
        mean, sd, rmssd, lo, hi = _stats(beats)
        previous = means[ordered[position - 1]] if position > 0 else mean
        following = (
            means[ordered[position + 1]] if position + 1 < len(ordered) else mean
        )
        values = (
            mean,
            sd,
            rmssd,
            lo,
            hi,
            hi - lo,
            float(len(beats)),
            mean / night_median_rr if night_median_rr else 1.0,
            rmssd / night_median_rmssd,
            idx / total_epochs if total_epochs else 0.0,
            mean - previous,
            following - mean,
        )
        out.append(
            EpochSample(
                night_id=night_id,
                index=idx,
                start=start + timedelta(seconds=idx * EPOCH_SECONDS),
                values=values,
                label=stage_by_epoch.get(idx),
            )
        )
    return out


def labelled_only(samples: Sequence[EpochSample]) -> List[EpochSample]:
    return [s for s in samples if s.label is not None]


def stage_distribution(samples: Sequence[EpochSample]) -> Dict[str, int]:
    """Epoch counts per stage -- the class balance a classifier will face."""
    names = {
        SleepStage.AWAKE.value: "awake",
        SleepStage.LIGHT.value: "light",
        SleepStage.DEEP.value: "deep",
        SleepStage.REM.value: "rem",
    }
    counts: Dict[str, int] = {}
    for s in samples:
        if s.label is None:
            continue
        key = names.get(s.label, "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
