"""BIDSLEEP: Apple Watch heart rate and motion, against expert-scored EEG.

The dataset this project needed and did not have. 47 people, 253 nights, worn
on the wrist, with 3-axis accelerometry alongside heart rate and sleep stages
scored from a Dreem 2 EEG headband and checked by a human expert. Open Data
Commons Attribution licence, no credentialing.

    https://physionet.org/content/bidsleep-dataset/1.0.0/

**Why this matters here.** Our staging work has been stuck at roughly chance on
MIT-BIH slpdb, and the stated reason was the missing accelerometer: every wrist
staging method worth copying uses motion alongside heart rate, and slpdb has
none. This dataset has it, on the wrist, at 50 Hz, on sixteen times as many
nights.

**Three honest differences from slpdb, which make this a separate evaluation
rather than a continuation of that one.**

The heart rate here is a PPG-derived rate at about 0.2 Hz -- roughly six
samples per 30-second epoch -- not beat-to-beat RR intervals. Every RR-derived
feature we built (RMSSD, the interval spread, the frequency bands) is simply
not computable from it. So this module defines its own feature set, and its
numbers are not comparable to the slpdb numbers by construction.

The labels come from an EEG headband rather than clinical polysomnography.
Expert and automatic scoring agree on about 93% of epochs in the first night,
which sets a practical ceiling: a model cannot be more right than its labels.

And the recordings are from healthy volunteers at home. slpdb subjects were
patients in a sleep lab, many with apnea. Easier population, more realistic
setting -- better numbers here would partly reflect that, not only better
features.

**Storage.** The motion file for one night is about 87 MB, and the whole
dataset is near 6 GB. Nothing here keeps raw motion: a night is streamed,
reduced to per-epoch features, and the raw file deleted. What persists is a
small JSON of features per night.
"""

from __future__ import annotations

import csv
import json
import math
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ..biometrics.schema import SleepStage
from ..ml.epochs import EPOCH_SECONDS, EpochSample

BASE_URL = "https://physionet.org/files/bidsleep-dataset/1.0.0"

#: Where reduced per-night features are kept. Raw signals are never cached:
#: the full motion set is ~6 GB and the features are a few hundred kilobytes.
DEFAULT_CACHE = (
    Path(os.environ.get("VITALGRAPH_CACHE", Path.home() / ".cache" / "vitalgraph"))
    / "bidsleep"
)

#: Dataset stage codes, mapped onto the four-class model the rest of the
#: project uses. N1 and N2 both become LIGHT, which is the standard collapse
#: and the one slpdb already uses, so the two datasets at least share a label
#: space even though they do not share a feature space.
#:
#: The mapping is asserted rather than assumed: :func:`check_label_mapping`
#: verifies it against sleep architecture (deep sleep concentrates early, REM
#: late), because a silently transposed code would not raise anything -- it
#: would just produce a model that looks mediocre for the wrong reason.
STAGE_CODES: Dict[int, SleepStage] = {
    0: SleepStage.AWAKE,
    1: SleepStage.LIGHT,  # N1
    2: SleepStage.LIGHT,  # N2
    3: SleepStage.DEEP,  # N3
    4: SleepStage.REM,
}

#: Accelerometer sample rate, samples per second. Used only to size buffers
#: and to sanity-check a night; the timestamps are authoritative.
NOMINAL_ACCEL_HZ = 50.0

#: Below this many heart-rate samples an epoch has no usable rate statistics.
#: Apple Watch sampling is irregular and drops out; two samples is the minimum
#: that supports any spread at all.
MIN_HR_SAMPLES_PER_EPOCH = 2

#: Below this many motion samples the epoch is a gap, not a still period.
#: A tenth of the nominal rate over 30 seconds.
MIN_MOTION_SAMPLES_PER_EPOCH = 150

#: Change in acceleration magnitude, in g, below which a sample pair counts as
#: stillness. Chosen just above the noise floor of a consumer MEMS
#: accelerometer at rest, so quiet breathing does not register as movement.
STILLNESS_THRESHOLD_G = 0.01

FEATURE_NAMES: Tuple[str, ...] = (
    # --- motion, the signal slpdb never had -------------------------------
    "accel_sd",
    "accel_range",
    "accel_mean_jerk",
    "accel_max_jerk",
    "accel_still_fraction",
    "posture_x",
    "posture_y",
    "posture_z",
    "posture_change",
    # --- heart rate, at 0.2 Hz rather than per beat -----------------------
    "hr_mean",
    "hr_sd",
    "hr_min",
    "hr_max",
    "hr_vs_night_median",
    "hr_delta_prev",
    # --- position in the night --------------------------------------------
    "elapsed_fraction",
)


class BidsleepError(RuntimeError):
    """Raised when a night cannot be turned into epochs."""


# --- fetching ---------------------------------------------------------------


def night_url(subject: str, night: int, filename: str) -> str:
    return f"{BASE_URL}/{subject}/{night}/{filename}"


def fetch(url: str, dest: Path, timeout: float = 600.0) -> Path:
    """Download one file. Existing files are reused rather than re-fetched."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        with partial.open("wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    # Rename only once complete, so an interrupted download is never mistaken
    # for a cached file on the next run.
    partial.replace(dest)
    return dest


def list_subjects(timeout: float = 60.0) -> List[str]:
    """Subject folder names, read from the dataset index."""
    import re

    with urllib.request.urlopen(BASE_URL + "/", timeout=timeout) as response:
        html = response.read().decode("utf-8", "replace")
    names = re.findall(r'href="(Bidslab\d+)/"', html)
    seen: List[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def list_nights(subject: str, timeout: float = 60.0) -> List[int]:
    import re

    with urllib.request.urlopen(f"{BASE_URL}/{subject}/", timeout=timeout) as response:
        html = response.read().decode("utf-8", "replace")
    return sorted({int(n) for n in re.findall(r'href="(\d+)/"', html)})


# --- reading one night ------------------------------------------------------


def read_labels(path: Path) -> Tuple[datetime, List[int]]:
    """Expert-verified stage codes and the recording start time, as written.

    The expert labels are used, never the automatic ones. They differ on about
    7% of epochs, and training against the automatic scoring would mean
    learning another algorithm's mistakes rather than sleep.

    ``recStart`` carries no timezone. It is returned here parsed as UTC and
    must be corrected by :func:`align_start` before it is used as an epoch
    origin -- see that function for why.
    """
    from scipy.io import loadmat

    mat = loadmat(str(path))
    if "expert_label" not in mat:
        raise BidsleepError(f"{path} has no expert_label array")
    labels = [int(v) for v in mat["expert_label"].ravel()]
    raw_start = str(mat["recStart"][0])
    start = datetime.strptime(raw_start, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    return start, labels


#: How far the inferred offset may sit from a timezone before the alignment is
#: rejected. Signals starting a minute or two either side of ``recStart`` is
#: normal lead-in; ten minutes is not, and lining those clocks up anyway would
#: shift every label half an epoch-block against its signal.
#:
#: This has to stay below half the rounding granularity or it is unreachable:
#: rounding to half hours already bounds the residual at 15 minutes, so a
#: larger limit would be a guard that can never fire. A test asserts that.
MAX_ALIGNMENT_RESIDUAL_S = 10 * 60


def align_start(nominal_start: datetime, first_signal_ts: float) -> Tuple[float, float]:
    """Recover the true unix time of epoch zero. Returns (start_ts, residual).

    ``recStart`` is local wall-clock time with no zone attached, while the
    signal timestamps are unix seconds. Reading the label clock as UTC puts
    the epoch grid hours away from the data, and the damage is quiet: epochs
    still get built, still carry labels, and are simply matched to the wrong
    part of the night.

    That bug cost this loader 64% of its epochs on the first night tried, and
    it looked like sensor dropout rather than an error. The fix is to infer
    the offset from the data: whatever whole number of half-hours brings
    ``recStart`` closest to where the signals actually begin is the timezone,
    and anything left over is real lead-in.
    """
    naive = nominal_start.timestamp()
    delta = first_signal_ts - naive
    # Half-hour granularity covers every real timezone, India and Newfoundland
    # included, without being loose enough to absorb a genuine misalignment.
    offset = round(delta / 1800.0) * 1800.0
    residual = delta - offset
    if abs(residual) > MAX_ALIGNMENT_RESIDUAL_S:
        raise BidsleepError(
            f"cannot align labels to signals: {residual / 60:.1f} min from the "
            f"nearest timezone offset, which is more than a timezone apart"
        )
    return naive + offset, residual


def read_heart_rate(path: Path) -> List[Tuple[float, float]]:
    """(unix timestamp, bpm) pairs. The file has no header."""
    out: List[Tuple[float, float]] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            try:
                out.append((float(row[0]), float(row[1])))
            except ValueError:
                continue  # a stray header line, if a future release adds one
    return out


def stream_motion(path: Path) -> Iterable[Tuple[float, float, float, float]]:
    """Yield (timestamp, x, y, z) without holding the file in memory.

    One night is over a million rows and ~87 MB. Streaming keeps the whole
    dataset processable on a machine that could not hold two nights at once.
    """
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header and header[0].lower().startswith("time"):
            pass  # header consumed
        elif header:
            try:
                yield (
                    float(header[0]),
                    float(header[1]),
                    float(header[2]),
                    float(header[3]),
                )
            except (ValueError, IndexError):
                pass
        for row in reader:
            if len(row) < 4:
                continue
            try:
                yield float(row[0]), float(row[1]), float(row[2]), float(row[3])
            except ValueError:
                continue


# --- per-epoch reduction ----------------------------------------------------


@dataclass
class MotionEpoch:
    """Motion summary for one 30-second epoch.

    ``samples`` is kept so a caller can tell a genuinely still epoch from a
    stretch where the watch stopped reporting. Those look identical in every
    other field and mean opposite things.
    """

    samples: int = 0
    sd: float = 0.0
    range_g: float = 0.0
    mean_jerk: float = 0.0
    max_jerk: float = 0.0
    still_fraction: float = 0.0
    posture: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def _reduce_motion(
    rows: Iterable[Tuple[float, float, float, float]],
    start_ts: float,
    n_epochs: int,
) -> List[MotionEpoch]:
    """Fold a night of motion into one summary per epoch, in a single pass."""
    sums = [[0.0, 0.0, 0.0, 0.0] for _ in range(n_epochs)]  # mag, x, y, z
    sq = [0.0] * n_epochs
    lo = [math.inf] * n_epochs
    hi = [-math.inf] * n_epochs
    jerk_sum = [0.0] * n_epochs
    jerk_max = [0.0] * n_epochs
    still = [0] * n_epochs
    count = [0] * n_epochs
    pairs = [0] * n_epochs

    previous_mag: float | None = None
    previous_index: int | None = None

    for ts, x, y, z in rows:
        index = int((ts - start_ts) // EPOCH_SECONDS)
        if index < 0 or index >= n_epochs:
            previous_mag, previous_index = None, None
            continue
        mag = math.sqrt(x * x + y * y + z * z)
        s = sums[index]
        s[0] += mag
        s[1] += x
        s[2] += y
        s[3] += z
        sq[index] += mag * mag
        if mag < lo[index]:
            lo[index] = mag
        if mag > hi[index]:
            hi[index] = mag
        count[index] += 1
        # Jerk is only meaningful between consecutive samples inside the same
        # epoch. Across an epoch boundary or a dropout it would measure the
        # gap, not the movement.
        if previous_mag is not None and previous_index == index:
            delta = abs(mag - previous_mag)
            jerk_sum[index] += delta
            if delta > jerk_max[index]:
                jerk_max[index] = delta
            if delta < STILLNESS_THRESHOLD_G:
                still[index] += 1
            pairs[index] += 1
        previous_mag, previous_index = mag, index

    out: List[MotionEpoch] = []
    for i in range(n_epochs):
        n = count[i]
        if n == 0:
            out.append(MotionEpoch())
            continue
        mean = sums[i][0] / n
        variance = max(0.0, sq[i] / n - mean * mean)
        out.append(
            MotionEpoch(
                samples=n,
                sd=math.sqrt(variance),
                range_g=(hi[i] - lo[i]) if n > 1 else 0.0,
                mean_jerk=(jerk_sum[i] / pairs[i]) if pairs[i] else 0.0,
                max_jerk=jerk_max[i],
                still_fraction=(still[i] / pairs[i]) if pairs[i] else 0.0,
                posture=(sums[i][1] / n, sums[i][2] / n, sums[i][3] / n),
            )
        )
    return out


def _bucket_heart_rate(
    samples: Sequence[Tuple[float, float]], start_ts: float, n_epochs: int
) -> List[List[float]]:
    buckets: List[List[float]] = [[] for _ in range(n_epochs)]
    for ts, bpm in samples:
        index = int((ts - start_ts) // EPOCH_SECONDS)
        # Implausible rates are dropped rather than clipped: a PPG dropout
        # reports a number, and letting it through would move the mean.
        if 0 <= index < n_epochs and 25.0 <= bpm <= 220.0:
            buckets[index].append(bpm)
    return buckets


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def build_epochs(
    night_id: str,
    start: datetime,
    labels: Sequence[int],
    heart_rate: Sequence[Tuple[float, float]],
    motion_rows: Iterable[Tuple[float, float, float, float]],
) -> Tuple[List[EpochSample], Dict[str, object]]:
    """Turn one night into labelled epochs, with a report on what was dropped.

    Epochs missing either signal are dropped, never imputed. A watch that
    stopped reporting is not a still, resting subject, and filling the gap with
    a plausible value is how a dataset quietly teaches a model to hallucinate.
    """
    if not heart_rate:
        raise BidsleepError(f"{night_id}: no heart rate samples")
    start_ts, residual = align_start(start, heart_rate[0][0])
    n = len(labels)
    motion = _reduce_motion(motion_rows, start_ts, n)
    hr_buckets = _bucket_heart_rate(heart_rate, start_ts, n)

    usable = [
        i
        for i in range(n)
        if motion[i].samples >= MIN_MOTION_SAMPLES_PER_EPOCH
        and len(hr_buckets[i]) >= MIN_HR_SAMPLES_PER_EPOCH
        and labels[i] in STAGE_CODES
    ]
    if not usable:
        raise BidsleepError(f"{night_id}: no epoch has both signals")

    night_median = _median([sum(hr_buckets[i]) / len(hr_buckets[i]) for i in usable])
    samples: List[EpochSample] = []
    previous_hr: float | None = None
    previous_posture: Tuple[float, float, float] | None = None

    for i in usable:
        m = motion[i]
        hr = hr_buckets[i]
        hr_mean = sum(hr) / len(hr)
        hr_sd = (
            math.sqrt(sum((v - hr_mean) ** 2 for v in hr) / (len(hr) - 1))
            if len(hr) > 1
            else 0.0
        )
        # Posture change is the angle between this epoch's mean orientation
        # and the previous one -- rolling over, which is one of the few things
        # a wrist can actually see happening during sleep.
        if previous_posture is None:
            posture_change = 0.0
        else:
            a, b = m.posture, previous_posture
            na = math.sqrt(sum(v * v for v in a)) or 1.0
            nb = math.sqrt(sum(v * v for v in b)) or 1.0
            cos = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))
            posture_change = math.degrees(math.acos(cos))

        values = (
            m.sd,
            m.range_g,
            m.mean_jerk,
            m.max_jerk,
            m.still_fraction,
            m.posture[0],
            m.posture[1],
            m.posture[2],
            posture_change,
            hr_mean,
            hr_sd,
            min(hr),
            max(hr),
            hr_mean - night_median,
            0.0 if previous_hr is None else hr_mean - previous_hr,
            i / max(1, n - 1),
        )
        samples.append(
            EpochSample(
                night_id=night_id,
                index=i,
                start=start,
                values=values,
                label=float(STAGE_CODES[labels[i]].value),
            )
        )
        previous_hr = hr_mean
        previous_posture = m.posture

    report = {
        "night": night_id,
        "scored_epochs": n,
        "usable_epochs": len(usable),
        "dropped_no_motion": sum(
            1 for i in range(n) if motion[i].samples < MIN_MOTION_SAMPLES_PER_EPOCH
        ),
        "dropped_no_heart_rate": sum(
            1 for i in range(n) if len(hr_buckets[i]) < MIN_HR_SAMPLES_PER_EPOCH
        ),
        "hours": round(n * EPOCH_SECONDS / 3600.0, 2),
        "alignment_residual_s": round(residual, 1),
        "usable_fraction": round(len(usable) / n, 3),
    }
    return samples, report


# --- structural check on the label mapping ---------------------------------


def check_label_mapping(samples: Sequence[EpochSample]) -> Dict[str, object]:
    """Verify the stage codes against sleep architecture, not against a README.

    Deep sleep concentrates in the first third of the night and REM in the
    last. If :data:`STAGE_CODES` were transposed, nothing would raise -- the
    model would just be mediocre, and we would spend a week blaming the
    features. So the mapping is checked against the one thing that is true of
    every human night.
    """
    early_deep = late_deep = early_rem = late_rem = 0
    for s in samples:
        if s.label is None:
            continue
        first_third = s.values[FEATURE_NAMES.index("elapsed_fraction")] < 0.34
        stage = SleepStage(int(s.label))
        if stage is SleepStage.DEEP:
            early_deep += first_third
            late_deep += not first_third
        elif stage is SleepStage.REM:
            early_rem += first_third
            late_rem += not first_third

    deep_early_share = early_deep / max(1, early_deep + late_deep)
    rem_late_share = late_rem / max(1, early_rem + late_rem)
    return {
        "deep_share_in_first_third": round(deep_early_share, 3),
        "rem_share_after_first_third": round(rem_late_share, 3),
        # Deep should be over-represented early relative to its 1/3 share of
        # the night, REM under-represented. Both holding is strong evidence
        # the mapping is right way up.
        "consistent_with_sleep_architecture": deep_early_share > 0.34
        and rem_late_share > 0.66,
    }


# --- the whole night, end to end -------------------------------------------


def load_night(
    subject: str,
    night: int,
    cache_dir: Path | None = None,
    keep_raw: bool = False,
) -> Tuple[List[EpochSample], Dict[str, object]]:
    """Download, reduce and cache one night.

    The raw motion file is deleted once reduced unless ``keep_raw``. At 87 MB
    a night the full dataset does not fit alongside itself on a modest disk,
    and the features are what everything downstream actually wants.
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    night_id = f"{subject}/{night}"
    reduced = cache_dir / "features" / f"{subject}_{night}.json"

    if reduced.exists():
        payload = json.loads(reduced.read_text())
        samples = [
            EpochSample(
                night_id=payload["night"],
                index=row["index"],
                start=datetime.fromisoformat(payload["start"]),
                values=tuple(row["values"]),
                label=row["label"],
            )
            for row in payload["epochs"]
        ]
        return samples, payload["report"]

    raw = cache_dir / "raw" / subject / str(night)
    labels_path = fetch(night_url(subject, night, "labels.mat"), raw / "labels.mat")
    hr_path = fetch(night_url(subject, night, "hr.csv"), raw / "hr.csv")
    motion_path = fetch(night_url(subject, night, "motion.csv"), raw / "motion.csv")

    try:
        start, labels = read_labels(labels_path)
        heart_rate = read_heart_rate(hr_path)
        samples, report = build_epochs(
            night_id, start, labels, heart_rate, stream_motion(motion_path)
        )
    finally:
        if not keep_raw and motion_path.exists():
            motion_path.unlink()

    reduced.parent.mkdir(parents=True, exist_ok=True)
    reduced.write_text(
        json.dumps(
            {
                "night": night_id,
                "start": start.isoformat(),
                "report": report,
                "epochs": [
                    {"index": s.index, "values": list(s.values), "label": s.label}
                    for s in samples
                ],
            }
        )
    )
    return samples, report
