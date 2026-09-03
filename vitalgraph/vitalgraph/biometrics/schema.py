"""Canonical biometric signal model.

Every sample that enters VitalGraph -- from a live BLE stream, a bulk vendor
export, or the simulator -- is normalised into a :class:`Sample`. Keeping one
shape means the analytics layer never has to know where data came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List


class SignalType(str, Enum):
    """Physiological quantities VitalGraph stores."""

    HEART_RATE = "heart_rate"          # bpm
    RR_INTERVAL = "rr_interval"        # ms, beat-to-beat
    SPO2 = "spo2"                      # percent
    SKIN_TEMPERATURE = "skin_temp"     # degrees C
    ACCEL_MAGNITUDE = "accel_mag"      # g
    SLEEP_STAGE = "sleep_stage"        # SleepStage value, as float


class SleepStage(float, Enum):
    AWAKE = 0.0
    LIGHT = 1.0
    DEEP = 2.0
    REM = 3.0


#: Physiologically plausible bounds. Samples outside these are rejected at the
#: door rather than being allowed to poison downstream aggregates.
VALID_RANGE: Dict[SignalType, tuple[float, float]] = {
    SignalType.HEART_RATE: (20.0, 240.0),
    SignalType.RR_INTERVAL: (250.0, 3000.0),
    SignalType.SPO2: (50.0, 100.0),
    SignalType.SKIN_TEMPERATURE: (20.0, 45.0),
    SignalType.ACCEL_MAGNITUDE: (0.0, 16.0),
    SignalType.SLEEP_STAGE: (0.0, 3.0),
}


class InvalidSample(ValueError):
    """Raised when a sample cannot be represented in the canonical model."""


@dataclass(frozen=True, slots=True)
class Sample:
    """A single timestamped measurement.

    ``source`` records provenance (``"simulator"``, ``"ble:0x2A37"``,
    ``"import:apple_health"``) so a later audit can tell measured data from
    synthetic data -- which matters once real and simulated sessions coexist.
    """

    ts: datetime
    signal: SignalType
    value: float
    source: str = "unknown"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise InvalidSample("Sample.ts must be timezone-aware (use UTC)")
        lo, hi = VALID_RANGE[self.signal]
        if not (lo <= self.value <= hi):
            raise InvalidSample(
                f"{self.signal.value}={self.value} outside plausible range [{lo}, {hi}]"
            )

    @property
    def epoch_ms(self) -> int:
        return int(self.ts.timestamp() * 1000)


def utc(ts: float) -> datetime:
    """Epoch seconds -> timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def rr_to_instantaneous_hr(rr_ms: Iterable[float]) -> List[float]:
    """Convert RR intervals (ms) to instantaneous heart rate (bpm)."""
    return [60000.0 / rr for rr in rr_ms if rr > 0]
