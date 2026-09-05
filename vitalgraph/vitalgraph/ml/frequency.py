"""Frequency-domain HRV, computed over windows long enough to mean anything.

The other half of what the corpus recommended for sleep staging. Autonomic
balance shifts across sleep stages in ways the time-domain statistics only see
indirectly: high-frequency power tracks parasympathetic (vagal) activity,
which rises in deep sleep, while the low-frequency band carries a mix that
rises in REM and wake.

**The constraint that shapes this module.** These estimates need minutes, not
seconds. The harvested hrv-analysis documents its own frequency function as
being for "short term recordings, from 2 to 5 minutes"
(`hrv-analysis@4fd889a:hrvanalysis/extract_features.py#L205-L293`, read as
documented behaviour -- the repository is GPL and facts-only). A 30-second
epoch is far below that, so computing per-epoch would produce numbers that
look like spectral power and are mostly windowing artifact.

So each epoch gets features from a centred window of several minutes around
it. That satisfies the duration requirement and supplies temporal context in
the same step.

**VLF is deliberately absent.** The very-low-frequency band starts at
0.003 Hz, a period of over five minutes, so a five-minute window contains less
than one full cycle of its slowest component. An estimate from that is not a
weak measurement of VLF, it is not a measurement of VLF, and reporting one
would be exactly the kind of confident nonsense the signal model exists to
prevent.

**Interpolation is on the time axis.** RR intervals arrive one per beat, so
the series is irregularly sampled and must be resampled before a spectrum can
be taken. The resampling interpolates interval *duration against beat
timestamp* -- never against beat index. Against index, a stretch of slow beats
occupies the same axis length as a stretch of fast ones, which distorts every
frequency in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

#: Frequency bands, in Hz. VLF is omitted: see the module docstring.
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)

#: Rate the irregular RR series is resampled to before the spectrum is taken.
#: 4 Hz is the conventional choice and is comfortably above twice the top of
#: the HF band, so nothing in the bands of interest is aliased.
RESAMPLE_HZ = 4.0

#: Centred window, in seconds. Five minutes is the top of the range the
#: literature considers valid for short-term frequency analysis, and the
#: longest that still resolves a stage transition rather than averaging over
#: several of them.
WINDOW_SECONDS = 300.0

#: Below this many beats a window is not analysed at all. Roughly two minutes
#: at 60 bpm; less does not support an LF estimate, whose slowest component
#: has a 25-second period.
MIN_BEATS_FOR_SPECTRUM = 120

FREQUENCY_FEATURE_NAMES: Tuple[str, ...] = (
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "normalised_hf",
)


@dataclass(frozen=True, slots=True)
class BandPowers:
    """Spectral power in the two estimable bands.

    ``None`` throughout when the window was too short or too sparse to
    support an estimate, which is different from zero power and must stay
    distinguishable.
    """

    lf: float | None
    hf: float | None

    @property
    def ratio(self) -> float | None:
        """LF/HF. None when either band is missing or HF is zero.

        Often described as "sympathovagal balance"; that interpretation is
        contested and the domain catalogue already records it as such. It is
        included here as a discriminative feature, not as a physiological
        claim.
        """
        if self.lf is None or self.hf is None or self.hf <= 0.0:
            return None
        return self.lf / self.hf

    @property
    def normalised_hf(self) -> float | None:
        """HF as a fraction of LF+HF.

        Bounded in [0, 1] and free of the absolute scaling that makes raw
        power incomparable between people, which matters because the model is
        trained across subjects.
        """
        if self.lf is None or self.hf is None:
            return None
        total = self.lf + self.hf
        return self.hf / total if total > 0 else None

    def as_features(self) -> Tuple[float, float, float, float]:
        """Feature vector, with absent estimates rendered as zero.

        Zero is used only here, at the boundary with a model that cannot
        accept None, and only after ``None`` has done its work of keeping
        "not estimable" distinct everywhere upstream.
        """
        return (
            self.lf or 0.0,
            self.hf or 0.0,
            self.ratio or 0.0,
            self.normalised_hf or 0.0,
        )


def resample_rr(
    times_s: Sequence[float],
    rr_ms: Sequence[float],
    rate_hz: float = RESAMPLE_HZ,
) -> Tuple[List[float], List[float]]:
    """Resample an irregular RR series onto a uniform grid.

    Returns the grid times and the interpolated interval durations. Linear
    interpolation against the beat *timestamps*, which is the axis that makes
    a spectrum meaningful.
    """
    if len(times_s) < 2 or len(times_s) != len(rr_ms):
        return [], []

    start, end = times_s[0], times_s[-1]
    if end <= start:
        return [], []

    step = 1.0 / rate_hz
    grid: List[float] = []
    t = start
    while t <= end:
        grid.append(t)
        t += step

    values: List[float] = []
    cursor = 0
    for point in grid:
        while cursor + 2 < len(times_s) and times_s[cursor + 1] < point:
            cursor += 1
        t0, t1 = times_s[cursor], times_s[cursor + 1]
        v0, v1 = rr_ms[cursor], rr_ms[cursor + 1]
        if t1 <= t0:
            values.append(v0)
            continue
        fraction = (point - t0) / (t1 - t0)
        values.append(v0 + fraction * (v1 - v0))
    return grid, values


def band_powers(
    times_s: Sequence[float],
    rr_ms: Sequence[float],
    rate_hz: float = RESAMPLE_HZ,
) -> BandPowers:
    """Estimate LF and HF power for one window of RR intervals.

    Returns empty powers rather than numbers when the window cannot support
    an estimate, which is the case this module exists to handle correctly.
    """
    if len(rr_ms) < MIN_BEATS_FOR_SPECTRUM:
        return BandPowers(None, None)

    grid, values = resample_rr(times_s, rr_ms, rate_hz)
    if len(values) < 16:
        return BandPowers(None, None)

    import numpy as np
    from scipy.signal import welch

    series = np.asarray(values, dtype=float)
    # Detrend: a slow drift in mean RR across the window is a trend, not
    # oscillatory power, and leaks into the lowest bins if left in.
    series = series - series.mean()

    # Segment length is a quarter of the window, so Welch averages roughly
    # four overlapping segments -- enough to steady the estimate without
    # coarsening resolution past the LF band's lower edge.
    nperseg = min(len(series), max(64, len(series) // 4))
    freqs, power = welch(series, fs=rate_hz, nperseg=nperseg)

    def integrate(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            return 0.0
        return float(np.trapezoid(power[mask], freqs[mask]))

    return BandPowers(lf=integrate(*LF_BAND), hf=integrate(*HF_BAND))


def windowed_band_powers(
    times_s: Sequence[float],
    rr_ms: Sequence[float],
    epoch_starts: Sequence[float],
    window_seconds: float = WINDOW_SECONDS,
    rate_hz: float = RESAMPLE_HZ,
) -> List[BandPowers]:
    """Band powers for each epoch, from a window centred on it.

    ``epoch_starts`` are seconds from the same origin as ``times_s``. One
    result is returned per epoch start, in order.
    """
    half = window_seconds / 2.0
    results: List[BandPowers] = []

    for start in epoch_starts:
        low, high = start - half, start + half
        # The series is sorted, so a linear scan per epoch would be
        # quadratic; bisect keeps it linear overall.
        from bisect import bisect_left, bisect_right

        begin = bisect_left(times_s, low)
        end = bisect_right(times_s, high)
        results.append(band_powers(times_s[begin:end], rr_ms[begin:end], rate_hz))
    return results
