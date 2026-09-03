"""Detection of discrete physiological events within a night.

Two event classes, both derivable from what a standard BLE device exposes:

* **Desaturation** -- a sustained drop in SpO2 below the night's own baseline,
  the signature an apnea-like airway event leaves in the periphery.
* **Irregularity** -- a stretch where beat-to-beat intervals scatter far beyond
  the person's normal variability, which is what an arrhythmia looks like
  through an RR stream.

Both are detected against the night's *own* baseline rather than fixed
thresholds. An SpO2 of 93% is unremarkable at altitude and notable at sea
level; a relative drop is comparable across people and places.

Deliberately rule-based rather than learned. There is no labelled corpus of
real events here, and a classifier trained on the simulator's injected
episodes would learn the injection, exactly as the sleep stager does. Rules
built from the published scoring conventions -- a 3-4 point desaturation
sustained for ten seconds or more -- are honest about what they encode and can
be checked against those conventions by a clinician. Pure standard library.

Naming: these are *signal* events. "Irregularity" is not atrial fibrillation
and a desaturation is not a diagnosed apnea; scoring either requires
polysomnography. What is offered is a flag worth investigating.

On the perfect scores this achieves against the simulator: the injected
episodes are unambiguous by construction -- a clean 6-point desaturation, a
clean 28% scatter -- so recovering all of them says the rules are wired up
correctly and nothing more. Real events are shallower, noisier, and overlap
normal variation; expect materially lower recall and real false positives.
The rules are defensible because they encode published scoring conventions,
not because they scored 100% here.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple

from ..biometrics.schema import SignalType
from ..biometrics.store import BiometricStore

#: Points below the rolling baseline that count as a desaturation. Three to
#: four points is the conventional scoring threshold.
DESAT_DROP_POINTS = 3.0

#: Minimum duration, seconds, before a dip is an event rather than noise.
MIN_EVENT_SECONDS = 10.0

#: Gap under which two dips are merged into one event, seconds.
MERGE_GAP_SECONDS = 20.0

#: Window over which the SpO2 baseline is computed, seconds. Long enough that
#: an event cannot drag its own baseline down with it.
BASELINE_WINDOW_SECONDS = 600.0

#: Multiple of the night's median local variability above which a stretch of
#: beats counts as irregular.
IRREGULARITY_FACTOR = 2.5

#: Beats per window used to measure local variability.
IRREGULARITY_WINDOW_BEATS = 20

#: Maps a physiological cause to the signal signature it leaves. The simulator
#: injects causes ("apnea"); the detector observes signatures ("desaturation").
#: Keeping the two vocabularies distinct is deliberate -- a detector that
#: reported "apnea" would be claiming a diagnosis it cannot support -- so
#: scoring has to translate between them explicitly.
SIGNATURE_OF_CAUSE = {
    "apnea": "desaturation",
    "arrhythmia": "irregularity",
}


@dataclass(frozen=True, slots=True)
class DetectedEvent:
    """One detected episode."""

    kind: str
    """desaturation | irregularity"""
    start: datetime
    end: datetime
    severity: float
    """Depth of the excursion: SpO2 points lost, or ratio above normal scatter."""
    detail: str

    @property
    def duration_s(self) -> float:
        return (self.end - self.start).total_seconds()

    def overlaps(self, start: datetime, end: datetime) -> bool:
        return self.start < end and start < self.end

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "duration_s": round(self.duration_s, 1),
            "severity": round(self.severity, 2),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EventScore:
    """Detection performance against known ground truth."""

    detected: int
    truth: int
    matched: int

    @property
    def recall(self) -> float:
        return self.matched / self.truth if self.truth else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.detected if self.detected else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "truth": self.truth,
            "matched": self.matched,
            "recall": round(self.recall, 3),
            "precision": round(self.precision, 3),
        }


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _merge(
    runs: Sequence[Tuple[datetime, datetime, float]], kind: str, unit: str
) -> List[DetectedEvent]:
    """Merge adjacent runs and drop those too short to be events."""
    if not runs:
        return []

    merged: List[List[Any]] = [list(runs[0])]
    for start, end, severity in runs[1:]:
        if (start - merged[-1][1]).total_seconds() <= MERGE_GAP_SECONDS:
            merged[-1][1] = end
            merged[-1][2] = max(merged[-1][2], severity)
        else:
            merged.append([start, end, severity])

    events = []
    for start, end, severity in merged:
        duration = (end - start).total_seconds()
        if duration < MIN_EVENT_SECONDS:
            continue
        events.append(
            DetectedEvent(
                kind=kind,
                start=start,
                end=end,
                severity=severity,
                detail=(
                    f"{kind} lasting {duration:.0f}s, "
                    f"peak {severity:.1f} {unit} from baseline"
                ),
            )
        )
    return events


def detect_desaturations(
    store: BiometricStore, start: datetime, end: datetime
) -> List[DetectedEvent]:
    """Find sustained SpO2 drops below the night's rolling baseline."""
    samples = store.samples_in(SignalType.SPO2, start, end)
    if len(samples) < 4:
        return []

    values = [s.value for s in samples]
    times = [s.ts for s in samples]

    # A trailing-window median resists being pulled down by the event itself.
    runs: List[Tuple[datetime, datetime, float]] = []
    open_start: datetime | None = None
    open_peak = 0.0
    previous_time = times[0]

    for index, (ts, value) in enumerate(zip(times, values)):
        window = [
            v
            for t, v in zip(times[:index], values[:index])
            if (ts - t).total_seconds() <= BASELINE_WINDOW_SECONDS
        ]
        baseline = _median(window) if len(window) >= 3 else _median(values)
        drop = baseline - value

        if drop >= DESAT_DROP_POINTS:
            if open_start is None:
                open_start = ts
                open_peak = drop
            else:
                open_peak = max(open_peak, drop)
        elif open_start is not None:
            runs.append((open_start, previous_time, open_peak))
            open_start = None
        previous_time = ts

    if open_start is not None:
        runs.append((open_start, times[-1], open_peak))

    return _merge(runs, "desaturation", "points")


def detect_irregularity(
    store: BiometricStore, start: datetime, end: datetime
) -> List[DetectedEvent]:
    """Find stretches where beat-to-beat scatter far exceeds the night's norm."""
    samples = store.samples_in(SignalType.RR_INTERVAL, start, end)
    if len(samples) < IRREGULARITY_WINDOW_BEATS * 3:
        return []

    values = [s.value for s in samples]
    times = [s.ts for s in samples]

    # Local scatter: mean absolute successive difference, normalised by the
    # local mean so it is comparable across heart rates.
    scatter: List[float] = []
    for i in range(len(values) - IRREGULARITY_WINDOW_BEATS):
        window = values[i : i + IRREGULARITY_WINDOW_BEATS]
        diffs = [abs(window[j + 1] - window[j]) for j in range(len(window) - 1)]
        mean = sum(window) / len(window)
        scatter.append((sum(diffs) / len(diffs)) / mean if mean else 0.0)

    if not scatter:
        return []
    baseline = _median(scatter)
    if baseline <= 0:
        return []
    threshold = baseline * IRREGULARITY_FACTOR

    runs: List[Tuple[datetime, datetime, float]] = []
    open_start: datetime | None = None
    open_peak = 0.0
    for i, value in enumerate(scatter):
        if value >= threshold:
            if open_start is None:
                open_start = times[i]
                open_peak = value / baseline
            else:
                open_peak = max(open_peak, value / baseline)
        elif open_start is not None:
            runs.append((open_start, times[i], open_peak))
            open_start = None
    if open_start is not None:
        runs.append((open_start, times[len(scatter) - 1], open_peak))

    return _merge(runs, "irregularity", "x normal scatter")


def detect_events(
    store: BiometricStore, start: datetime, end: datetime
) -> List[DetectedEvent]:
    """All detectable events in a window, ordered by time."""
    events = detect_desaturations(store, start, end) + detect_irregularity(
        store, start, end
    )
    return sorted(events, key=lambda e: e.start)


def score_against_truth(
    detected: Sequence[DetectedEvent],
    episodes: Sequence[Any],
    night_start: datetime,
    kind: str,
    tolerance_s: float = 60.0,
) -> EventScore:
    """Measure detection against the simulator's ground-truth episodes.

    A detection counts as matched when it overlaps the true window, widened by
    ``tolerance_s`` -- a desaturation lags the airway event that caused it, so
    demanding exact alignment would understate real performance.
    """
    signature = SIGNATURE_OF_CAUSE.get(kind, kind)
    truth = [e for e in episodes if e.kind == kind]
    mine = [d for d in detected if d.kind == signature]

    matched = 0
    for episode in truth:
        window_start = night_start + timedelta(seconds=episode.start_s - tolerance_s)
        window_end = night_start + timedelta(seconds=episode.end_s + tolerance_s)
        if any(d.overlaps(window_start, window_end) for d in mine):
            matched += 1

    return EventScore(detected=len(mine), truth=len(truth), matched=matched)


def to_content_list(events: Sequence[DetectedEvent]) -> List[Dict[str, Any]]:
    """Render detected events as knowledge-graph content."""
    if not events:
        return []

    rows = ["| Time | Kind | Duration | Severity |", "| --- | --- | --- | --- |"]
    for e in events:
        rows.append(
            f"| {e.start:%Y-%m-%d %H:%M} | {e.kind} | {e.duration_s:.0f}s "
            f"| {e.severity:.1f} |"
        )

    by_kind: Dict[str, int] = {}
    for e in events:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    narrative = (
        "Detected "
        + ", ".join(
            f"{count} {kind} event(s)" for kind, count in sorted(by_kind.items())
        )
        + " during this period."
    )

    return [
        {"type": "text", "text": narrative, "page_idx": 0},
        {
            "type": "table",
            "table_body": "\n".join(rows),
            "table_caption": [f"{len(events)} signal event(s) detected"],
            "table_footnote": [
                "Signal events detected against this night's own baseline. "
                "An irregularity is not a diagnosis of atrial fibrillation and a "
                "desaturation is not a diagnosed apnea; scoring either requires "
                "polysomnography. These are flags worth investigating."
            ],
            "page_idx": 0,
        },
    ]
