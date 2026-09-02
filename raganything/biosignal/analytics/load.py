"""Training load and intensity, computed from what was measured.

The formulas here are the published ones -- Banister's TRIMP, Coggan-style
normalised power and intensity factor, exponentially weighted acute and chronic
load, and aerobic decoupling. None of them are secret. What differs from a
commercial implementation is that each one states its inputs, refuses to run on
insufficient data, and never substitutes a default for a value it does not have.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "LoadBalance",
    "aerobic_decoupling",
    "ewma_load",
    "intensity_factor",
    "normalized_power",
    "training_stress",
    "trimp_banister",
]


def trimp_banister(
    hr_bpm: Sequence[Tuple[float, float]],
    rest_hr: float,
    max_hr: float,
    sex: str = "unspecified",
) -> Optional[float]:
    """Banister TRIMP from a ``(timestamp, bpm)`` series.

    Args:
        hr_bpm: Heart-rate samples in temporal order.
        rest_hr: The athlete's true resting heart rate.
        max_hr: The athlete's true maximum heart rate. A population estimate
            such as ``220 - age`` has a standard deviation of roughly 10 bpm,
            which propagates directly into this score -- so it must be supplied
            deliberately rather than defaulted here.
        sex: ``"male"``, ``"female"``, or ``"unspecified"``. Selects the
            weighting exponent from Banister's original formulation; the
            unspecified case uses their mean, and the caller should treat the
            result as correspondingly less precise.

    Returns:
        TRIMP in arbitrary units, or ``None`` if the inputs cannot support it.
    """
    if len(hr_bpm) < 2 or max_hr <= rest_hr:
        return None

    coefficient = {"male": 1.92, "female": 1.67}.get(sex.lower(), 1.795)
    reserve = max_hr - rest_hr

    total = 0.0
    for (t0, hr0), (t1, hr1) in zip(hr_bpm, hr_bpm[1:]):
        dt_min = (t1 - t0) / 60.0
        if dt_min <= 0:
            continue
        hr_mean = (hr0 + hr1) / 2.0
        ratio = (hr_mean - rest_hr) / reserve
        if ratio <= 0:
            continue
        ratio = min(ratio, 1.5)
        total += dt_min * ratio * 0.64 * math.exp(coefficient * ratio)
    return total


def normalized_power(
    power: Sequence[Tuple[float, float]], window_s: float = 30.0
) -> Optional[float]:
    """Fourth-root of the mean of the fourth power of a 30 s rolling average.

    Returns ``None`` for efforts shorter than the rolling window, where the
    statistic is not defined -- rather than silently degrading to average power.
    """
    if len(power) < 2:
        return None
    t0, t1 = power[0][0], power[-1][0]
    if t1 - t0 < window_s:
        return None

    rolling: List[float] = []
    start = 0
    for i, (t, _) in enumerate(power):
        while power[start][0] < t - window_s:
            start += 1
        if t - power[start][0] < window_s * 0.9:
            continue
        rolling.append(statistics.fmean(v for _, v in power[start : i + 1]))
    if not rolling:
        return None
    return (statistics.fmean(p**4 for p in rolling)) ** 0.25


def intensity_factor(np_watts: float, threshold_watts: float) -> Optional[float]:
    """Normalised power as a fraction of functional threshold power."""
    if threshold_watts <= 0:
        return None
    return np_watts / threshold_watts


def training_stress(
    duration_s: float, np_watts: float, threshold_watts: float
) -> Optional[float]:
    """Duration-weighted stress score, 100 = one hour at threshold."""
    if threshold_watts <= 0 or duration_s <= 0:
        return None
    intensity = np_watts / threshold_watts
    return (duration_s * np_watts * intensity) / (threshold_watts * 3600.0) * 100.0


def aerobic_decoupling(
    power: Sequence[Tuple[float, float]],
    hr: Sequence[Tuple[float, float]],
) -> Optional[float]:
    """Percentage drift in output-per-heartbeat between the halves of an effort.

    A positive value means the same heart rate bought less power in the second
    half -- the classic aerobic durability signal, and one that no mainstream
    consumer app computes despite having every input for it.
    """
    if len(power) < 8 or len(hr) < 8:
        return None

    t_start = max(power[0][0], hr[0][0])
    t_end = min(power[-1][0], hr[-1][0])
    if t_end - t_start < 600:  # under ten minutes the ratio is noise
        return None
    mid = (t_start + t_end) / 2.0

    def ratio(lo: float, high: float) -> Optional[float]:
        p = [v for t, v in power if lo <= t < high]
        h = [v for t, v in hr if lo <= t < high and v > 0]
        if len(p) < 4 or len(h) < 4:
            return None
        mean_hr = statistics.fmean(h)
        if mean_hr <= 0:
            return None
        return statistics.fmean(p) / mean_hr

    first = ratio(t_start, mid)
    second = ratio(mid, t_end)
    if first is None or second is None or first == 0:
        return None
    return (first - second) / first * 100.0


@dataclass
class LoadBalance:
    """Acute versus chronic load, and the ratio between them."""

    acute: float
    chronic: float
    n_days: int
    notes: List[str] = field(default_factory=list)

    @property
    def ratio(self) -> Optional[float]:
        return self.acute / self.chronic if self.chronic > 0 else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "acute": self.acute,
            "chronic": self.chronic,
            "ratio": self.ratio,
            "n_days": self.n_days,
            "notes": list(self.notes),
        }


def ewma_load(
    daily_load: Sequence[float], acute_tau: float = 7.0, chronic_tau: float = 42.0
) -> LoadBalance:
    """Exponentially weighted acute and chronic load from a daily series.

    ``daily_load`` must be one entry per calendar day including zeros for rest
    days: dropping rest days inflates the acute figure, which is the most
    common way this metric is got wrong.
    """
    values = [float(v) for v in daily_load]
    notes: List[str] = []
    if not values:
        return LoadBalance(0.0, 0.0, 0, ["no daily load supplied"])
    if len(values) < chronic_tau:
        notes.append(
            f"only {len(values)} days of history; the {chronic_tau:.0f}-day chronic "
            "load is not yet fully established"
        )

    a_alpha = 2.0 / (acute_tau + 1.0)
    c_alpha = 2.0 / (chronic_tau + 1.0)
    acute = values[0]
    chronic = values[0]
    for v in values[1:]:
        acute = v * a_alpha + acute * (1 - a_alpha)
        chronic = v * c_alpha + chronic * (1 - c_alpha)
    return LoadBalance(acute=acute, chronic=chronic, n_days=len(values), notes=notes)
