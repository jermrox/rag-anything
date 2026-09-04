"""Turning real polysomnography into the same epochs the simulator produces.

The point of routing real records through :func:`vitalgraph.ml.epochs.
epoch_samples` rather than building features separately is that it makes the
comparison honest. If real and simulated nights took different paths into the
model, a difference in accuracy could be a difference in feature construction
rather than a difference in the data, and the whole exercise would prove
nothing.

So a PhysioNet record becomes exactly what a simulated night becomes: RR
intervals and stage labels in a :class:`BiometricStore`, epoched by the same
function, with ``source`` recording which is which.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Sequence, Tuple

from ..biometrics.schema import Sample, SignalType, SleepStage
from ..biometrics.store import BiometricStore
from ..ml.epochs import EPOCH_SECONDS, EpochSample, epoch_samples
from .ecg import BeatDetection, detect_beats
from .physionet import SLPDB_RECORDS, PolysomnographyRecord, load_record

#: Provenance marker distinguishing real records from simulated ones. The
#: registry already separates synthetic from real training data; this is what
#: makes that distinction true rather than declared.
PHYSIONET_SOURCE = "physionet:slpdb"

#: An arbitrary but fixed epoch start. Records carry a wall-clock start time,
#: but nothing downstream depends on the absolute date and pinning it keeps
#: runs reproducible.
EPOCH_ORIGIN = datetime(2020, 1, 1, tzinfo=timezone.utc)


def record_to_epochs(
    record: PolysomnographyRecord,
    detection: BeatDetection | None = None,
) -> Tuple[List[EpochSample], BeatDetection]:
    """Convert one record into labelled epochs, plus the beat detection.

    The detection is returned rather than discarded because its rejection rate
    is the honest caveat on everything derived from it: a night that threw away
    a fifth of its intervals produces HRV features that mean much less than the
    same features from a clean trace.
    """
    detection = detection or detect_beats(record.ecg, record.sampling_hz)

    store = BiometricStore(":memory:")
    try:
        samples: List[Sample] = [
            Sample(
                ts=EPOCH_ORIGIN + timedelta(seconds=t),
                signal=SignalType.RR_INTERVAL,
                value=rr,
                source=PHYSIONET_SOURCE,
            )
            for t, rr in zip(detection.rr_times_s, detection.rr_ms)
        ]
        samples.extend(
            Sample(
                # Placed at the centre of its epoch, so a stage cannot land on
                # a boundary and be bucketed into the neighbouring epoch by
                # floating-point rounding.
                ts=EPOCH_ORIGIN
                + timedelta(seconds=index * EPOCH_SECONDS + EPOCH_SECONDS / 2),
                signal=SignalType.SLEEP_STAGE,
                value=float(stage.value),
                source=PHYSIONET_SOURCE,
            )
            for index, stage in record.stages
        )
        store.add(samples)

        duration = len(record.ecg) / record.sampling_hz
        epochs = epoch_samples(
            store,
            EPOCH_ORIGIN,
            EPOCH_ORIGIN + timedelta(seconds=duration),
            night_id=record.name,
        )
    finally:
        store.close()

    return epochs, detection


def load_epochs(
    records: Sequence[str] = SLPDB_RECORDS,
    cache_dir=None,
    skip_implausible: bool = True,
    on_progress=None,
) -> Tuple[List[EpochSample], Dict[str, object]]:
    """Load several records into one labelled epoch set, with a report.

    Records whose beat detection is implausible are skipped by default. A
    detector failing on a bad lead produces confident nonsense rather than an
    error, and training on it would quietly poison the model with a night of
    fabricated intervals.
    """
    all_epochs: List[EpochSample] = []
    per_record: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for name in records:
        try:
            record = (
                load_record(name, cache_dir=cache_dir)
                if cache_dir is not None
                else load_record(name)
            )
        except Exception as exc:  # noqa: BLE001 - a bad record must not stop the run
            skipped.append({"record": name, "reason": f"load failed: {exc}"})
            if on_progress:
                on_progress(f"{name}: load failed: {exc}")
            continue

        epochs, detection = record_to_epochs(record)
        row = {
            "record": name,
            "hours": round(record.duration_hours, 2),
            "beats": detection.n_beats,
            "mean_hr": round(detection.mean_heart_rate, 1),
            "rejection_rate": round(detection.rejection_rate, 4),
            "labelled_epochs": sum(1 for e in epochs if e.label is not None),
        }

        if skip_implausible and not detection.is_plausible():
            row["reason"] = (
                f"implausible detection: {detection.mean_heart_rate:.0f} bpm, "
                f"{detection.rejection_rate:.1%} of intervals rejected"
            )
            skipped.append(row)
            if on_progress:
                on_progress(f"{name}: skipped, {row['reason']}")
            continue

        if not row["labelled_epochs"]:
            # A record with a clean ECG but no usable staging contributes
            # nothing and must not be counted as used. Two slpdb records
            # behave this way, and counting them inflated "records_used" to 18
            # for a run that actually cross-validated 16 -- a small lie, but
            # exactly the kind that makes a report untrustworthy.
            row["reason"] = "no epochs carry a sleep stage label"
            skipped.append(row)
            if on_progress:
                on_progress(f"{name}: skipped, {row['reason']}")
            continue

        all_epochs.extend(epochs)
        per_record.append(row)
        if on_progress:
            on_progress(
                f"{name}: {row['labelled_epochs']} labelled epochs, "
                f"{row['mean_hr']} bpm"
            )

    stage_counts: Dict[str, int] = {}
    for epoch in all_epochs:
        if epoch.label is None:
            continue
        name = SleepStage(epoch.label).name.lower()
        stage_counts[name] = stage_counts.get(name, 0) + 1

    report: Dict[str, object] = {
        "records_used": len(per_record),
        "records_skipped": len(skipped),
        "total_epochs": len(all_epochs),
        "labelled_epochs": sum(1 for e in all_epochs if e.label is not None),
        "stage_counts": stage_counts,
        "per_record": per_record,
        "skipped": skipped,
    }
    return all_epochs, report
