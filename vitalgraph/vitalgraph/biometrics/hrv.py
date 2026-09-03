"""Heart-rate-variability analytics computed from RR-interval series.

Everything here is derivable from the *standard* Bluetooth GATT Heart Rate
Measurement characteristic (0x2A37) when the sensor sets the
"RR-Interval present" flag -- no proprietary vendor protocol required. That is
the central competitive fact: a large part of what premium wearables sell is
computable from an open, published characteristic.

Pure standard library on purpose: a night of beats is ~30k floats, which does
not justify a NumPy dependency in the core analytics path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

# Malik rule: reject a beat differing more than this fraction from its
# predecessor. The standard artifact-correction threshold in HRV literature.
MALIK_THRESHOLD = 0.20

# Minimum accepted beats before frequency/time-domain output is meaningful.
MIN_BEATS = 20


@dataclass(frozen=True, slots=True)
class HRVMetrics:
    """Time-domain HRV summary for one window of beats."""

    n_beats: int
    n_rejected: int
    mean_rr_ms: float
    mean_hr_bpm: float
    rmssd_ms: float
    sdnn_ms: float
    pnn50_pct: float
    coverage: float
    """Fraction of submitted beats that survived artifact correction (0..1)."""

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def correct_artifacts(
    rr_ms: Sequence[float], threshold: float = MALIK_THRESHOLD
) -> tuple[List[float], int]:
    """Drop ectopic/spurious beats from an RR series.

    Applies the Malik rule: a beat is rejected when it deviates from the last
    *accepted* beat by more than ``threshold``. Comparing against the last
    accepted beat (rather than the immediate predecessor) stops a single
    artifact from cascading and rejecting the healthy beats after it.

    Returns ``(accepted, n_rejected)``.
    """
    accepted: List[float] = []
    rejected = 0
    reference: float | None = None

    for rr in rr_ms:
        if rr <= 0 or not math.isfinite(rr):
            rejected += 1
            continue
        if reference is None:
            accepted.append(rr)
            reference = rr
            continue
        if abs(rr - reference) / reference > threshold:
            rejected += 1
            continue
        accepted.append(rr)
        reference = rr

    return accepted, rejected


def rmssd(rr_ms: Sequence[float]) -> float:
    """Root mean square of successive differences (ms).

    The dominant short-term/parasympathetic HRV index and the basis of most
    commercial "recovery" scores.
    """
    if len(rr_ms) < 2:
        return 0.0
    diffs = [rr_ms[i + 1] - rr_ms[i] for i in range(len(rr_ms) - 1)]
    return math.sqrt(sum(d * d for d in diffs) / len(diffs))


def sdnn(rr_ms: Sequence[float]) -> float:
    """Standard deviation of NN intervals (ms) -- overall variability."""
    n = len(rr_ms)
    if n < 2:
        return 0.0
    mean = sum(rr_ms) / n
    # Sample standard deviation (n-1), matching HRV convention.
    return math.sqrt(sum((rr - mean) ** 2 for rr in rr_ms) / (n - 1))


def pnn50(rr_ms: Sequence[float]) -> float:
    """Percentage of successive RR differences greater than 50 ms."""
    if len(rr_ms) < 2:
        return 0.0
    diffs = [abs(rr_ms[i + 1] - rr_ms[i]) for i in range(len(rr_ms) - 1)]
    return 100.0 * sum(1 for d in diffs if d > 50.0) / len(diffs)


def analyze(rr_ms: Sequence[float], correct: bool = True) -> HRVMetrics:
    """Compute the full time-domain HRV summary for a window of beats."""
    raw_n = len(rr_ms)
    beats, rejected = correct_artifacts(rr_ms) if correct else (list(rr_ms), 0)

    if not beats:
        return HRVMetrics(0, rejected, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    mean_rr = sum(beats) / len(beats)
    return HRVMetrics(
        n_beats=len(beats),
        n_rejected=rejected,
        mean_rr_ms=mean_rr,
        mean_hr_bpm=60000.0 / mean_rr,
        rmssd_ms=rmssd(beats),
        sdnn_ms=sdnn(beats),
        pnn50_pct=pnn50(beats),
        coverage=len(beats) / raw_n if raw_n else 0.0,
    )


#: Nights of history required before a personal baseline is trusted. A short
#: baseline has almost no observed variance, so its standard deviation is a bad
#: scale estimate and z-scores explode to implausible magnitudes. Commercial
#: wearables use multi-week rolling windows for the same reason.
MIN_BASELINE_NIGHTS = 7

#: Floor on the baseline standard deviation, in ms. Guards the same failure
#: from the other direction: an unusually consistent stretch of nights must not
#: turn an ordinary dip into a ten-sigma event.
MIN_BASELINE_SD_MS = 3.0


def baseline_deviation(
    value: float,
    baseline: Sequence[float],
    min_samples: int = MIN_BASELINE_NIGHTS,
) -> float | None:
    """Z-score of ``value`` against a personal ``baseline`` history.

    Personal baselines are what make HRV actionable -- an RMSSD of 24 ms is
    unremarkable for one person and a red flag for another. Returns ``None``
    when there is not enough history, so callers must handle "unknown" rather
    than silently comparing against a fabricated norm.
    """
    if len(baseline) < min_samples:
        return None
    mean = sum(baseline) / len(baseline)
    sd = math.sqrt(sum((b - mean) ** 2 for b in baseline) / (len(baseline) - 1))
    sd = max(sd, MIN_BASELINE_SD_MS)
    return (value - mean) / sd
