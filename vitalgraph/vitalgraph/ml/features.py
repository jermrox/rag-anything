"""Feature extraction: a night of biometrics -> a fixed-length vector.

Deliberately pure standard library. Feature extraction is arithmetic over a few
thousand numbers, so making it depend on numpy would push the ``[ml]`` extra
into the core path for no benefit. Only the *models* need scikit-learn.

The single most important property here is :data:`FEATURE_NAMES`: a fixed,
ordered vocabulary. A model persisted today and loaded next month must receive
its features in the same order, so the order is data, not an implementation
detail, and the registry records it alongside every model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

from ..biometrics import hrv
from ..biometrics.schema import SignalType, SleepStage
from ..biometrics.store import BiometricStore

#: Ordered feature vocabulary. Append only -- reordering or removing an entry
#: silently invalidates every persisted model that was trained against it.
FEATURE_NAMES: Tuple[str, ...] = (
    # Heart-rate variability
    "rmssd_ms",
    "sdnn_ms",
    "pnn50_pct",
    "mean_hr_bpm",
    "hr_range_bpm",
    "artifact_coverage",
    # Oxygen saturation
    "spo2_mean",
    "spo2_min",
    "spo2_dips",
    # Temperature
    "skin_temp_mean",
    "skin_temp_range",
    # Sleep architecture
    "sleep_minutes",
    "deep_fraction",
    "rem_fraction",
    "light_fraction",
    "awake_fraction",
    "fragmentation",
    # Circadian placement, encoded cyclically so 23:00 and 01:00 are close
    "start_hour_sin",
    "start_hour_cos",
)

#: A window with fewer usable beats than this yields no feature vector at all.
#: Reuses the analytics threshold so "too little data" means the same thing
#: everywhere in the product.
MIN_BEATS = hrv.MIN_BEATS

#: A drop of at least this many points below the window mean counts as a
#: desaturation dip -- the coarse signal an apnea-like event would produce.
SPO2_DIP_POINTS = 3.0


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """One period's features, in :data:`FEATURE_NAMES` order."""

    period_id: str
    start: datetime
    end: datetime
    values: Tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FEATURE_NAMES)} features, got {len(self.values)}"
            )

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values))

    def get(self, name: str) -> float:
        return self.values[FEATURE_NAMES.index(name)]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_fraction(part: float, whole: float) -> float:
    return part / whole if whole else 0.0


def night_features(
    store: BiometricStore, start: datetime, end: datetime, period_id: str
) -> FeatureVector | None:
    """Extract features for one window.

    Returns ``None`` when the window holds too few usable beats. Returning a
    vector of zeros instead would be worse than useless: a model cannot tell
    "measured zero variability" from "no data", and the resulting prediction
    would look confident and be meaningless.
    """
    rr = store.rr_series(start, end)
    metrics = hrv.analyze(rr)
    if metrics.n_beats < MIN_BEATS:
        return None

    hr_samples = [s.value for s in store.samples_in(SignalType.HEART_RATE, start, end)]
    hr_range = max(hr_samples) - min(hr_samples) if hr_samples else 0.0

    spo2 = [s.value for s in store.samples_in(SignalType.SPO2, start, end)]
    spo2_mean = _mean(spo2)
    spo2_min = min(spo2) if spo2 else 0.0
    spo2_dips = float(sum(1 for v in spo2 if spo2_mean - v >= SPO2_DIP_POINTS))

    temps = [s.value for s in store.samples_in(SignalType.SKIN_TEMPERATURE, start, end)]
    temp_range = (max(temps) - min(temps)) if temps else 0.0

    stages = [s.value for s in store.samples_in(SignalType.SLEEP_STAGE, start, end)]
    # Stage samples are logged once per minute, so each counts for a minute.
    minutes = {
        "awake": float(sum(1 for v in stages if v == SleepStage.AWAKE.value)),
        "light": float(sum(1 for v in stages if v == SleepStage.LIGHT.value)),
        "deep": float(sum(1 for v in stages if v == SleepStage.DEEP.value)),
        "rem": float(sum(1 for v in stages if v == SleepStage.REM.value)),
    }
    asleep = minutes["light"] + minutes["deep"] + minutes["rem"]
    in_bed = asleep + minutes["awake"]

    # Count transitions into wake rather than total wake time: many brief
    # awakenings and one long one are physiologically different.
    transitions = sum(
        1
        for a, b in zip(stages, stages[1:])
        if a != SleepStage.AWAKE.value and b == SleepStage.AWAKE.value
    )

    hour = start.hour + start.minute / 60.0
    angle = 2 * math.pi * hour / 24.0

    values = (
        metrics.rmssd_ms,
        metrics.sdnn_ms,
        metrics.pnn50_pct,
        metrics.mean_hr_bpm,
        hr_range,
        metrics.coverage,
        spo2_mean,
        spo2_min,
        spo2_dips,
        _mean(temps),
        temp_range,
        asleep,
        _safe_fraction(minutes["deep"], asleep),
        _safe_fraction(minutes["rem"], asleep),
        _safe_fraction(minutes["light"], asleep),
        _safe_fraction(minutes["awake"], in_bed),
        _safe_fraction(float(transitions), asleep / 60.0) if asleep else 0.0,
        math.sin(angle),
        math.cos(angle),
    )
    return FeatureVector(period_id=period_id, start=start, end=end, values=values)


def features_for_windows(
    store: BiometricStore, windows: Sequence[Tuple[str, datetime, datetime]]
) -> List[FeatureVector]:
    """Extract features for many windows, skipping those with too little data."""
    out: List[FeatureVector] = []
    for period_id, start, end in windows:
        fv = night_features(store, start, end, period_id)
        if fv is not None:
            out.append(fv)
    return out


def to_matrix(vectors: Sequence[FeatureVector]) -> List[List[float]]:
    """Plain nested lists, so callers without numpy can still inspect them."""
    return [list(v.values) for v in vectors]
