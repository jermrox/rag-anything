"""Normalising the non-BLE sources: vendor clouds and phone health stores.

BLE gives you raw physiology at millisecond resolution. Vendor clouds give you
the overnight analysis you cannot reproduce. Phone health stores give you the
passive baseline you never had to ask for. All three are worth having and all
three lie in different ways, so the job of this module is to bring them into
the canonical schema *without* laundering a vendor's opinion into a
measurement.

The rules applied here:

* A vendor's score (readiness, recovery, sleep stage) enters as
  :attr:`~.schema.Evidence.VENDOR_DERIVED` with ``documented=False`` and the
  algorithm named. It never becomes ``MEASURED``, however precise it looks.
* Each profile records the source's real latency and granularity, so a nightly
  summary is never silently compared against a live strap.
* Field maps are declarative and versioned, because vendor payload shapes
  change without notice and a decoding change should be a one-line diff, not an
  archaeology project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .schema import Evidence, Modality, Provenance, Sample, SourceKind, Stream

__all__ = [
    "FieldSpec",
    "VendorProfile",
    "PROFILES",
    "normalize_health_records",
    "normalize_vendor_payload",
    "parse_timestamp",
]


def parse_timestamp(value: Any) -> Optional[float]:
    """Best-effort POSIX seconds from the timestamp shapes vendors emit."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Milliseconds are common; anything past year 2286 in seconds is not.
        return float(value) / 1000.0 if float(value) > 1e11 else float(value)
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


def _dig(payload: Mapping[str, Any], path: str) -> Any:
    """Fetch a dotted path, tolerating missing intermediate keys."""
    cur: Any = payload
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


@dataclass(frozen=True)
class FieldSpec:
    """How one vendor field maps onto one canonical modality."""

    path: str
    modality: Modality
    #: Multiply the vendor value by this to reach the canonical unit.
    scale: float = 1.0
    offset: float = 0.0
    evidence: Evidence = Evidence.VENDOR_DERIVED
    confidence: float = 0.6
    algorithm: Optional[str] = None
    #: Dotted path to a per-record timestamp, if it differs from the record's.
    time_path: Optional[str] = None


@dataclass(frozen=True)
class VendorProfile:
    """Everything known about how a vendor delivers data."""

    vendor: str
    #: Dotted path to the list of records inside a response body.
    records_path: str
    #: Dotted path to each record's timestamp.
    time_path: str
    fields: Sequence[FieldSpec]
    #: Typical delay from physical event to availability, in seconds.
    latency_s: float
    #: Nominal reporting cadence, in Hz (1/60 for per-minute data).
    nominal_hz: Optional[float] = None
    transport: str = ""
    notes: str = ""


#: Profiles for the four vendor APIs most health apps integrate. Payload shapes
#: are the documented ones at time of writing; they are declarative on purpose
#: so that an API revision is a data change rather than a code change.
PROFILES: Dict[str, VendorProfile] = {
    "oura": VendorProfile(
        vendor="oura",
        records_path="data",
        time_path="timestamp",
        fields=(
            FieldSpec("bpm", Modality.HEART_RATE, evidence=Evidence.VENDOR_DERIVED),
            FieldSpec(
                "average_hrv",
                Modality.HRV_RMSSD,
                algorithm="Oura nightly RMSSD aggregate",
            ),
            FieldSpec("score", Modality.READINESS, algorithm="Oura readiness score"),
            FieldSpec("spo2_percentage.average", Modality.SPO2),
        ),
        latency_s=6 * 3600,
        nominal_hz=1 / 300.0,
        transport="REST /v2/usercollection",
        notes=(
            "Five-minute heart-rate resolution while asleep; readiness is a "
            "closed composite and cannot be recomputed from anything Oura returns."
        ),
    ),
    "whoop": VendorProfile(
        vendor="whoop",
        records_path="records",
        time_path="created_at",
        fields=(
            FieldSpec(
                "score.hrv_rmssd_milli",
                Modality.HRV_RMSSD,
                scale=1000.0,
                algorithm="WHOOP nightly RMSSD",
            ),
            FieldSpec("score.resting_heart_rate", Modality.HEART_RATE),
            FieldSpec(
                "score.recovery_score",
                Modality.READINESS,
                algorithm="WHOOP recovery score",
            ),
            FieldSpec("score.spo2_percentage", Modality.SPO2),
        ),
        latency_s=4 * 3600,
        transport="REST /v1/recovery, /v1/cycle",
        notes=(
            "One recovery value per physiological cycle, not per calendar day; "
            "aligning it to dates is itself an assumption."
        ),
    ),
    "garmin": VendorProfile(
        vendor="garmin",
        records_path="dailies",
        time_path="startTimeInSeconds",
        fields=(
            FieldSpec("averageHeartRateInBeatsPerMinute", Modality.HEART_RATE),
            FieldSpec("restingHeartRateInBeatsPerMinute", Modality.HEART_RATE),
            FieldSpec("steps", Modality.STEPS, evidence=Evidence.VENDOR_DERIVED),
            FieldSpec("activeKilocalories", Modality.ENERGY),
        ),
        latency_s=900,
        transport="Health API push / ping",
        notes=(
            "Push-based, so lower latency than its peers, but daily summaries "
            "are restated retroactively -- the same day fetched twice can differ."
        ),
    ),
    "fitbit": VendorProfile(
        vendor="fitbit",
        records_path="activities-heart-intraday.dataset",
        time_path="time",
        fields=(FieldSpec("value", Modality.HEART_RATE),),
        latency_s=3600,
        nominal_hz=1 / 60.0,
        transport="REST /1/user/-/activities/heart",
        notes=(
            "Intraday access is gated behind a separate application; the public "
            "tier gives daily rollups only, and rate limits cap history backfill."
        ),
    ),
}


def normalize_vendor_payload(
    vendor: str,
    payload: Mapping[str, Any],
    source_id: Optional[str] = None,
    device: Optional[str] = None,
    profile: Optional[VendorProfile] = None,
) -> List[Stream]:
    """Convert one vendor API response into canonical streams.

    Unknown fields are ignored rather than guessed at, and records without a
    parseable timestamp are skipped -- an undated physiological value cannot be
    placed on any timeline and is worse than absent.
    """
    prof = profile or PROFILES.get(vendor.lower())
    if prof is None:
        raise KeyError(
            f"no profile for vendor {vendor!r}; known: {sorted(PROFILES)}. "
            "Pass profile= to normalise a custom shape."
        )

    records = _dig(payload, prof.records_path)
    if records is None:
        return []
    if isinstance(records, Mapping):
        records = [records]
    if not isinstance(records, (list, tuple)):
        return []

    by_key: Dict[Any, Stream] = {}
    for spec in prof.fields:
        samples: List[Sample] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            raw = _dig(record, spec.path)
            if raw is None or isinstance(raw, (str, bool)):
                continue
            t = parse_timestamp(_dig(record, spec.time_path or prof.time_path))
            if t is None:
                continue
            samples.append(
                Sample(
                    t=t,
                    value=float(raw) * spec.scale + spec.offset,
                    evidence=spec.evidence,
                    confidence=spec.confidence,
                    flags=("vendor_aggregate",),
                )
            )
        if not samples:
            continue

        key = (spec.modality, spec.path)
        provenance = Provenance(
            source_id=source_id or f"{prof.vendor}:{spec.path}",
            kind=SourceKind.VENDOR_CLOUD,
            device=device or prof.vendor,
            transport=prof.transport,
            latency_s=prof.latency_s,
            algorithm=spec.algorithm,
            documented=spec.evidence is not Evidence.VENDOR_DERIVED,
            nominal_hz=prof.nominal_hz,
            extra={"vendor_field": spec.path, "vendor_notes": prof.notes},
        )
        by_key[key] = Stream(
            modality=spec.modality, provenance=provenance, samples=samples
        )
    return list(by_key.values())


#: HealthKit / Health Connect type names mapped onto canonical modalities.
HEALTH_TYPE_MAP: Dict[str, Modality] = {
    "HKQuantityTypeIdentifierHeartRate": Modality.HEART_RATE,
    "HKQuantityTypeIdentifierRestingHeartRate": Modality.HEART_RATE,
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": Modality.HRV_SDNN,
    "HKQuantityTypeIdentifierOxygenSaturation": Modality.SPO2,
    "HKQuantityTypeIdentifierRespiratoryRate": Modality.RESPIRATION,
    "HKQuantityTypeIdentifierStepCount": Modality.STEPS,
    "HKQuantityTypeIdentifierBodyMass": Modality.WEIGHT,
    "HKQuantityTypeIdentifierBodyFatPercentage": Modality.BODY_FAT,
    "HKQuantityTypeIdentifierBloodGlucose": Modality.GLUCOSE,
    "HeartRateRecord": Modality.HEART_RATE,
    "RestingHeartRateRecord": Modality.HEART_RATE,
    "HeartRateVariabilityRmssdRecord": Modality.HRV_RMSSD,
    "OxygenSaturationRecord": Modality.SPO2,
    "RespiratoryRateRecord": Modality.RESPIRATION,
    "StepsRecord": Modality.STEPS,
    "WeightRecord": Modality.WEIGHT,
    "BodyFatRecord": Modality.BODY_FAT,
    "BloodGlucoseRecord": Modality.GLUCOSE,
    "SleepSessionRecord": Modality.SLEEP_STAGE,
}


def normalize_health_records(
    records: Iterable[Mapping[str, Any]],
    source_id: str = "phone",
    device: str = "handset",
    platform: str = "health_connect",
) -> List[Stream]:
    """Normalise HealthKit / Health Connect record dictionaries.

    Accepts the shape both platform bridges converge on: a ``type``, a start
    time, a numeric ``value``, and an originating app. The originating app
    matters and is preserved -- a step count written by a watch and one written
    by the phone's own pedometer are different measurements that these stores
    happily interleave into a single series.
    """
    grouped: Dict[Any, List[Sample]] = {}
    origins: Dict[Any, str] = {}

    for record in records:
        raw_type = record.get("type") or record.get("recordType") or ""
        modality = HEALTH_TYPE_MAP.get(str(raw_type))
        if modality is None:
            continue
        t = parse_timestamp(
            record.get("startDate")
            or record.get("startTime")
            or record.get("time")
            or record.get("timestamp")
        )
        value = record.get("value")
        if t is None or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue

        origin = str(
            record.get("sourceName")
            or record.get("dataOrigin")
            or record.get("origin")
            or "unknown_app"
        )
        key = (modality, origin)
        origins[key] = origin
        grouped.setdefault(key, []).append(
            Sample(
                t=t,
                value=numeric,
                evidence=Evidence.VENDOR_DERIVED,
                confidence=0.65,
                flags=("platform_store",),
            )
        )

    streams: List[Stream] = []
    for (modality, origin), samples in grouped.items():
        streams.append(
            Stream(
                modality=modality,
                provenance=Provenance(
                    source_id=f"{source_id}:{origin}",
                    kind=SourceKind.PHONE,
                    device=device,
                    transport=platform,
                    # Phone stores update in near real time, but writers batch:
                    # step data lands no more than once a minute.
                    latency_s=60.0,
                    algorithm=f"{origin} on-device processing",
                    documented=False,
                    extra={"writing_app": origin, "platform": platform},
                ),
                samples=sorted(samples, key=lambda s: s.t),
            )
        )
    return streams
