"""R-peak detection: turning a real ECG trace into RR intervals.

This is the bridge between a polysomnography record and the stager, which
consumes RR intervals and nothing else. The simulator handed those over
directly; real data does not, and the quality of everything downstream --
every HRV metric, every epoch feature, the stager's own accuracy -- is capped
by the quality of this step.

The approach is Pan-Tompkins, which is the standard and is standard for good
reasons: bandpass to the frequency band where the QRS complex lives,
differentiate to emphasise its steep slope, square to make everything positive
and amplify the large deflections, integrate over roughly a QRS width so each
complex becomes one broad bump, then find the bumps.

Two details decide whether the output is usable:

**A refractory period.** The heart cannot beat twice in 200 ms, so peaks
closer than that are the same complex detected twice. Without the constraint,
a tall T-wave gets counted as a beat and halves the reported interval.

**Rejecting implausible intervals.** A missed beat produces one interval of
roughly double length; a doubled detection produces two of half. Both are
artifacts, not physiology, and passing them into an HRV calculation inflates
RMSSD dramatically -- the metric is a root-mean-square of successive
differences, so a single doubled interval contributes far more than a real
beat ever does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

#: The QRS complex has most of its energy between these frequencies. Below is
#: baseline wander and the T wave; above is muscle noise and mains hum.
QRS_BAND_HZ = (5.0, 15.0)

#: Minimum time between beats. 200 ms is the physiological refractory period,
#: an absolute ceiling of 300 bpm, and the standard Pan-Tompkins value.
REFRACTORY_S = 0.20

#: Integration window, about the width of a QRS complex.
INTEGRATION_WINDOW_S = 0.15

#: Plausible RR interval range for a sleeping adult, in milliseconds. A
#: 300 ms interval is 200 bpm and a 2000 ms interval is 30 bpm; outside this
#: band the far likelier explanation is a detection error than a real beat.
PLAUSIBLE_RR_MS = (300.0, 2000.0)

#: A detection threshold as a fraction of the integrated signal's spread.
#: Deliberately conservative: missing a beat costs one interval, while a false
#: detection corrupts two.
THRESHOLD_FRACTION = 0.35


@dataclass(frozen=True, slots=True)
class BeatDetection:
    """Detected beats and an honest account of what was discarded."""

    beat_samples: Tuple[int, ...]
    rr_ms: Tuple[float, ...]
    rr_times_s: Tuple[float, ...]
    """Time of the *second* beat of each interval, which is when the interval
    becomes known. Attributing an interval to the first beat would shift every
    epoch boundary by one beat."""

    rejected_intervals: int
    """Intervals discarded as implausible. Reported rather than hidden: a high
    count means the trace is poor, and every downstream number should be read
    with that in mind."""

    @property
    def n_beats(self) -> int:
        return len(self.beat_samples)

    @property
    def rejection_rate(self) -> float:
        total = len(self.rr_ms) + self.rejected_intervals
        return self.rejected_intervals / total if total else 0.0

    @property
    def mean_heart_rate(self) -> float:
        if not self.rr_ms:
            return 0.0
        return 60_000.0 / (sum(self.rr_ms) / len(self.rr_ms))

    def is_plausible(self) -> bool:
        """Whether the detection looks like a real sleeping heart.

        A sleeping adult sits roughly between 40 and 100 bpm, and a trace
        rejecting more than a fifth of its intervals is too noisy to trust.
        Checked because a detector failing on a bad lead produces confident
        nonsense rather than an error.
        """
        return (
            35.0 <= self.mean_heart_rate <= 110.0
            and self.rejection_rate < 0.20
            and self.n_beats > 0
        )


def detect_beats(
    ecg: Sequence[float],
    sampling_hz: float,
    threshold_fraction: float = THRESHOLD_FRACTION,
) -> BeatDetection:
    """Find R peaks in an ECG trace and derive RR intervals.

    Returns intervals in milliseconds alongside the time each was observed,
    plus a count of what was rejected as implausible.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt, find_peaks

    signal = np.asarray(ecg, dtype=float)
    if signal.size < int(sampling_hz):
        return BeatDetection((), (), (), 0)

    nyquist = sampling_hz / 2.0
    low, high = QRS_BAND_HZ
    # Guard the band against a low sampling rate, where 15 Hz could exceed
    # Nyquist and butter() would raise rather than return a usable filter.
    high = min(high, nyquist * 0.9)
    if low >= high:
        return BeatDetection((), (), (), 0)

    b, a = butter(2, [low / nyquist, high / nyquist], btype="band")
    filtered = filtfilt(b, a, signal)

    differentiated = np.diff(filtered, prepend=filtered[0])
    squared = differentiated**2

    window = max(1, int(INTEGRATION_WINDOW_S * sampling_hz))
    integrated = np.convolve(squared, np.ones(window) / window, mode="same")

    # A threshold from the spread rather than the maximum: one motion artifact
    # can be orders of magnitude larger than any QRS, and scaling to it would
    # push the threshold above every real beat and find nothing.
    threshold = integrated.mean() + threshold_fraction * integrated.std()
    peaks, _ = find_peaks(
        integrated,
        height=threshold,
        distance=max(1, int(REFRACTORY_S * sampling_hz)),
    )

    if peaks.size < 2:
        return BeatDetection(tuple(int(p) for p in peaks), (), (), 0)

    intervals_ms = np.diff(peaks) / sampling_hz * 1000.0
    times_s = peaks[1:] / sampling_hz

    low_ms, high_ms = PLAUSIBLE_RR_MS
    keep = (intervals_ms >= low_ms) & (intervals_ms <= high_ms)
    rejected = int((~keep).sum())

    return BeatDetection(
        beat_samples=tuple(int(p) for p in peaks),
        rr_ms=tuple(float(v) for v in intervals_ms[keep]),
        rr_times_s=tuple(float(t) for t in times_s[keep]),
        rejected_intervals=rejected,
    )
