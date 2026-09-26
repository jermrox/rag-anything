"""Canonical schema for physiological and fitness signals.

The design goal of this module is one idea: **a number is never just a number.**

Every mainstream health app flattens three very different things into the same
chart: something a sensor actually measured, something a vendor's closed
algorithm inferred, and something that was interpolated to fill a hole. Once
they are flattened, no downstream consumer -- human or model -- can tell them
apart, and no honest confidence can be attached to any conclusion.

Here, every :class:`Sample` carries the evidence class it belongs to and the
confidence it was recorded with, and every :class:`Stream` carries the
:class:`Provenance` describing how it reached us. Analytics downstream refuse
to emit a metric that its evidence cannot support.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


class SourceKind(str, Enum):
    """Where a stream physically came from."""

    BLE = "ble"
    """Raw GATT notifications decoded on-device. Sub-second, unfiltered."""

    VENDOR_CLOUD = "vendor_cloud"
    """Oura / WHOOP / Garmin / Fitbit REST payloads. Delayed and pre-chewed."""

    PHONE = "phone"
    """HealthKit / Health Connect / handset IMU, barometer, GPS."""

    MANUAL = "manual"
    """Entered by a human (weight, RPE, soreness, meals)."""

    DERIVED = "derived"
    """Computed locally from other streams in this system."""


class Evidence(str, Enum):
    """How much of this value is observation versus inference.

    Ordered from strongest to weakest. :func:`Evidence.rank` gives the
    precedence used when reconciling disagreeing sources.
    """

    MEASURED = "measured"
    """A sensor reading, transported without reinterpretation."""

    LOCALLY_DERIVED = "locally_derived"
    """Computed here, from measured inputs, by auditable code in this repo."""

    VENDOR_DERIVED = "vendor_derived"
    """Produced by a closed vendor algorithm. Not reproducible, not auditable."""

    IMPUTED = "imputed"
    """Interpolated, back-filled, or otherwise invented to cover a gap."""

    @staticmethod
    def rank(evidence: "Evidence") -> int:
        return _EVIDENCE_RANK[evidence]


_EVIDENCE_RANK: Dict[Evidence, int] = {
    Evidence.MEASURED: 0,
    Evidence.LOCALLY_DERIVED: 1,
    Evidence.VENDOR_DERIVED: 2,
    Evidence.IMPUTED: 3,
}


class Modality(str, Enum):
    """What is being measured. Units are fixed per modality (see UNITS)."""

    HEART_RATE = "heart_rate"
    RR_INTERVAL = "rr_interval"
    POWER = "power"
    CADENCE = "cadence"
    SPEED = "speed"
    PACE = "pace"
    DISTANCE = "distance"
    STROKE_RATE = "stroke_rate"
    RESISTANCE = "resistance"
    INCLINE = "incline"
    ENERGY = "energy"
    ELEVATION = "elevation"
    FORCE = "force"
    TORQUE = "torque"
    PEDAL_BALANCE = "pedal_balance"
    STEPS = "steps"
    STRIDE_LENGTH = "stride_length"
    SPO2 = "spo2"
    RESPIRATION = "respiration"
    TEMPERATURE = "temperature"
    GLUCOSE = "glucose"
    WEIGHT = "weight"
    BODY_FAT = "body_fat"
    MUSCLE_MASS = "muscle_mass"
    BODY_WATER = "body_water"
    IMPEDANCE = "impedance"
    SLEEP_STAGE = "sleep_stage"
    HRV_RMSSD = "hrv_rmssd"
    HRV_SDNN = "hrv_sdnn"
    READINESS = "readiness"
    BATTERY = "battery"


UNITS: Dict[Modality, str] = {
    Modality.HEART_RATE: "bpm",
    Modality.RR_INTERVAL: "ms",
    Modality.POWER: "W",
    Modality.CADENCE: "rpm",
    Modality.SPEED: "m/s",
    Modality.PACE: "s/km",
    Modality.DISTANCE: "m",
    Modality.STROKE_RATE: "spm",
    Modality.RESISTANCE: "level",
    Modality.INCLINE: "%",
    Modality.ENERGY: "kcal",
    Modality.ELEVATION: "m",
    Modality.FORCE: "N",
    Modality.TORQUE: "Nm",
    Modality.PEDAL_BALANCE: "%",
    Modality.STEPS: "steps",
    Modality.STRIDE_LENGTH: "m",
    Modality.SPO2: "%",
    Modality.RESPIRATION: "brpm",
    Modality.TEMPERATURE: "degC",
    Modality.GLUCOSE: "mg/dL",
    Modality.WEIGHT: "kg",
    Modality.BODY_FAT: "%",
    Modality.MUSCLE_MASS: "kg",
    Modality.BODY_WATER: "kg",
    Modality.IMPEDANCE: "ohm",
    Modality.SLEEP_STAGE: "stage",
    Modality.HRV_RMSSD: "ms",
    Modality.HRV_SDNN: "ms",
    Modality.READINESS: "score",
    Modality.BATTERY: "%",
}

#: Physiologically plausible bounds, used to flag (never silently drop) outliers.
PLAUSIBLE_RANGE: Dict[Modality, Tuple[float, float]] = {
    Modality.HEART_RATE: (20.0, 240.0),
    Modality.RR_INTERVAL: (250.0, 3000.0),
    Modality.POWER: (0.0, 2500.0),
    Modality.CADENCE: (0.0, 250.0),
    Modality.SPEED: (0.0, 30.0),
    Modality.SPO2: (50.0, 100.0),
    Modality.RESPIRATION: (3.0, 60.0),
    Modality.TEMPERATURE: (25.0, 45.0),
    Modality.GLUCOSE: (20.0, 600.0),
    Modality.WEIGHT: (20.0, 300.0),
    Modality.BODY_FAT: (2.0, 70.0),
}


#: Flags that mean the sample itself is degraded, as opposed to flags that
#: merely record where it came from. Only these reduce a quality score --
#: "this arrived from a platform health store" is provenance, already captured
#: by :class:`Evidence`, and must not be double-counted as a defect.
DEFECT_FLAGS = frozenset(
    {
        "no_contact",
        "out_of_range",
        "sensor_fault",
        "low_battery",
        "motion_artifact",
    }
)


@dataclass(frozen=True)
class Provenance:
    """Everything needed to decide how much to trust a stream.

    ``documented`` is deliberately explicit: a vendor score whose derivation is
    a trade secret is not the same kind of object as an RR interval off a chest
    strap, and the difference must survive all the way to the answer a user
    reads.
    """

    source_id: str
    kind: SourceKind
    device: str = "unknown"
    #: Free-form transport note, e.g. "GATT notify 0x2A37" or "REST /v2/sleep".
    transport: str = ""
    #: Typical delay between the physical event and this record existing, in
    #: seconds. BLE notify is ~0.05; a vendor nightly summary can be 30_000.
    latency_s: float = 0.0
    #: Name of the closed algorithm that produced the value, when applicable.
    algorithm: Optional[str] = None
    #: Whether the derivation of the value is publicly specified.
    documented: bool = True
    #: Nominal sampling rate the source claims, in Hz. ``None`` if event-driven.
    nominal_hz: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind.value,
            "device": self.device,
            "transport": self.transport,
            "latency_s": self.latency_s,
            "algorithm": self.algorithm,
            "documented": self.documented,
            "nominal_hz": self.nominal_hz,
            **({"extra": dict(self.extra)} if self.extra else {}),
        }


@dataclass(frozen=True)
class Sample:
    """One observation on one stream.

    Attributes:
        t: POSIX timestamp in seconds (float, sub-second resolution kept).
        value: The measurement, in the modality's canonical unit.
        evidence: Observation vs. inference class.
        confidence: 0..1. Not a probability -- a monotone trust weight used to
            gate downstream metrics.
        flags: Machine-readable notes, e.g. ``("no_contact",)`` or
            ``("out_of_range",)``. Flags annotate, they never delete.
    """

    t: float
    value: float
    evidence: Evidence = Evidence.MEASURED
    confidence: float = 1.0
    flags: Tuple[str, ...] = ()

    def with_flag(self, *flags: str, confidence: Optional[float] = None) -> "Sample":
        merged = tuple(dict.fromkeys(self.flags + flags))
        return replace(
            self,
            flags=merged,
            confidence=self.confidence if confidence is None else confidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "t": self.t,
            "value": self.value,
            "evidence": self.evidence.value,
            "confidence": self.confidence,
        }
        if self.flags:
            d["flags"] = list(self.flags)
        return d


@dataclass
class Stream:
    """A time-ordered series of one modality from one source."""

    modality: Modality
    provenance: Provenance
    samples: List[Sample] = field(default_factory=list)
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if self.unit is None:
            self.unit = UNITS.get(self.modality, "")

    # -- construction ----------------------------------------------------

    def add(self, sample: Sample) -> "Stream":
        self.samples.append(sample)
        return self

    def extend(self, samples: Iterable[Sample]) -> "Stream":
        self.samples.extend(samples)
        return self

    def sorted(self) -> "Stream":
        """Return a copy with samples in non-decreasing time order."""
        return Stream(
            modality=self.modality,
            provenance=self.provenance,
            samples=sorted(self.samples, key=lambda s: s.t),
            unit=self.unit,
        )

    # -- access ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def times(self) -> List[float]:
        return [s.t for s in self.samples]

    def values(self) -> List[float]:
        return [s.value for s in self.samples]

    @property
    def start(self) -> Optional[float]:
        return min((s.t for s in self.samples), default=None)

    @property
    def end(self) -> Optional[float]:
        return max((s.t for s in self.samples), default=None)

    @property
    def duration_s(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return float(self.end - self.start)  # type: ignore[operator]

    def window(self, start: float, end: float) -> "Stream":
        """Half-open ``[start, end)`` slice, preserving provenance."""
        return Stream(
            modality=self.modality,
            provenance=self.provenance,
            samples=[s for s in self.samples if start <= s.t < end],
            unit=self.unit,
        )

    def flag_implausible(self) -> "Stream":
        """Flag (do not remove) samples outside physiological bounds."""
        bounds = PLAUSIBLE_RANGE.get(self.modality)
        if bounds is None:
            return self
        low, high = bounds
        out: List[Sample] = []
        for s in self.samples:
            if not math.isfinite(s.value) or s.value < low or s.value > high:
                out.append(s.with_flag("out_of_range", confidence=0.0))
            else:
                out.append(s)
        self.samples = out
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "unit": self.unit,
            "provenance": self.provenance.to_dict(),
            "n_samples": len(self.samples),
            "start": self.start,
            "end": self.end,
        }


@dataclass
class Session:
    """A bounded observation period: a workout, a night, a whole day."""

    session_id: str
    start: float
    end: float
    streams: List[Stream] = field(default_factory=list)
    #: Free-form context: sport, subjective RPE, altitude, illness, caffeine.
    labels: Dict[str, Any] = field(default_factory=dict)
    subject_id: str = "self"

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end - self.start))

    def add(self, stream: Stream) -> "Session":
        self.streams.append(stream)
        return self

    def of(self, modality: Modality) -> List[Stream]:
        """All streams of a modality -- there may be several, disagreeing."""
        return [s for s in self.streams if s.modality == modality]

    def first(self, modality: Modality) -> Optional[Stream]:
        """Highest-evidence stream of a modality, or ``None``.

        Ties break toward the lower-latency source, which in practice means a
        chest strap beats a nightly cloud summary for the same modality.
        """
        candidates = self.of(modality)
        if not candidates:
            return None

        def key(stream: Stream) -> Tuple[int, float]:
            best = min(
                (Evidence.rank(s.evidence) for s in stream.samples),
                default=_EVIDENCE_RANK[Evidence.IMPUTED],
            )
            return (best, stream.provenance.latency_s)

        return min(candidates, key=key)

    def modalities(self) -> List[Modality]:
        seen: Dict[Modality, None] = {}
        for s in self.streams:
            seen.setdefault(s.modality, None)
        return list(seen)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "start": self.start,
            "end": self.end,
            "duration_s": self.duration_s,
            "labels": dict(self.labels),
            "streams": [s.to_dict() for s in self.streams],
        }


def make_stream(
    modality: Modality,
    values: Sequence[Tuple[float, float]],
    provenance: Provenance,
    evidence: Evidence = Evidence.MEASURED,
    confidence: float = 1.0,
) -> Stream:
    """Convenience builder from ``(timestamp, value)`` pairs."""
    return Stream(
        modality=modality,
        provenance=provenance,
        samples=[
            Sample(t=t, value=float(v), evidence=evidence, confidence=confidence)
            for t, v in values
        ],
    )
