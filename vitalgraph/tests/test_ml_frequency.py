"""Tests for frequency-domain HRV.

The behaviour that matters most here is refusal: these estimates need minutes
of data, and a module that produces a number from thirty seconds produces
windowing artifact that looks exactly like spectral power.
"""

from __future__ import annotations

import math

import pytest

from vitalgraph.ml.frequency import (
    FREQUENCY_FEATURE_NAMES,
    HF_BAND,
    LF_BAND,
    MIN_BEATS_FOR_SPECTRUM,
    RESAMPLE_HZ,
    WINDOW_SECONDS,
    BandPowers,
    band_powers,
    resample_rr,
    windowed_band_powers,
)

pytest.importorskip("numpy")
pytest.importorskip("scipy")


def _oscillating_rr(
    n_beats: int,
    frequency_hz: float,
    amplitude_ms: float = 40.0,
    mean_ms: float = 1000.0,
):
    """An RR series modulated at one frequency, with beat times to match."""
    times = []
    rr = []
    t = 0.0
    for i in range(n_beats):
        interval = mean_ms + amplitude_ms * math.sin(2 * math.pi * frequency_hz * t)
        t += interval / 1000.0
        times.append(t)
        rr.append(interval)
    return times, rr


# --- the refusal -----------------------------------------------------------


def test_a_short_window_yields_no_estimate_rather_than_a_number():
    """Thirty seconds cannot support an LF estimate.

    The slowest LF component has a 25-second period, so a number from a short
    window is windowing artifact wearing the name of spectral power.
    """
    times, rr = _oscillating_rr(30, 0.25)
    powers = band_powers(times, rr)
    assert powers.lf is None
    assert powers.hf is None


def test_the_minimum_beat_count_covers_at_least_an_lf_cycle():
    """120 beats is about two minutes at 60 bpm; LF needs 25 seconds a cycle."""
    assert MIN_BEATS_FOR_SPECTRUM >= 120


def test_absent_powers_stay_distinguishable_from_zero_power():
    empty = BandPowers(None, None)
    zero = BandPowers(0.0, 0.0)
    assert empty.lf is None and zero.lf == 0.0
    assert empty.ratio is None and zero.ratio is None  # zero HF, undefined


def test_ratio_is_none_rather_than_infinite_when_hf_is_zero():
    """Dividing by zero HF would report infinite sympathovagal balance."""
    assert BandPowers(lf=5.0, hf=0.0).ratio is None


def test_normalised_hf_is_bounded_and_absent_when_undefined():
    assert BandPowers(lf=3.0, hf=1.0).normalised_hf == pytest.approx(0.25)
    assert BandPowers(lf=0.0, hf=0.0).normalised_hf is None
    assert BandPowers(lf=None, hf=1.0).normalised_hf is None


# --- vlf is deliberately absent --------------------------------------------


def test_no_vlf_band_is_reported():
    """VLF starts at 0.003 Hz -- over five minutes a cycle.

    A five-minute window holds less than one full cycle, so an estimate from
    it is not a weak VLF measurement, it is not a VLF measurement.
    """
    assert not any("vlf" in name for name in FREQUENCY_FEATURE_NAMES)
    assert LF_BAND[0] >= 0.04


def test_the_bands_do_not_overlap_and_are_the_conventional_ones():
    assert LF_BAND == (0.04, 0.15)
    assert HF_BAND == (0.15, 0.40)
    assert LF_BAND[1] == HF_BAND[0]


def test_the_resample_rate_does_not_alias_the_hf_band():
    """Nyquist must sit above the top of the highest band of interest."""
    assert RESAMPLE_HZ / 2 > HF_BAND[1]


# --- resampling ------------------------------------------------------------


def test_resampling_produces_a_uniform_grid():
    times, rr = _oscillating_rr(200, 0.25)
    grid, values = resample_rr(times, rr)
    assert len(grid) == len(values)
    steps = [b - a for a, b in zip(grid, grid[1:])]
    assert all(s == pytest.approx(1 / RESAMPLE_HZ) for s in steps)


def test_resampling_interpolates_against_time_not_beat_index():
    """Against index, slow and fast stretches occupy equal axis length.

    That distorts every frequency in the result, so the grid must track wall
    time: a series spanning T seconds yields about T * rate points regardless
    of how many beats it contains.
    """
    slow_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    slow_rr = [2000.0] * 6
    grid, _ = resample_rr(slow_times, slow_rr)
    assert len(grid) == pytest.approx(10.0 * RESAMPLE_HZ + 1, abs=1)


def test_degenerate_series_resample_to_nothing():
    assert resample_rr([], []) == ([], [])
    assert resample_rr([1.0], [1000.0]) == ([], [])
    assert resample_rr([5.0, 5.0], [1000.0, 1000.0]) == ([], [])


# --- the spectrum finds what is there --------------------------------------


def test_a_high_frequency_oscillation_lands_in_the_hf_band():
    """0.25 Hz is respiratory-rate modulation, squarely in HF."""
    times, rr = _oscillating_rr(400, 0.25)
    powers = band_powers(times, rr)
    assert powers.hf is not None
    assert powers.hf > powers.lf


def test_a_low_frequency_oscillation_lands_in_the_lf_band():
    """0.08 Hz is the baroreflex band."""
    times, rr = _oscillating_rr(400, 0.08)
    powers = band_powers(times, rr)
    assert powers.lf is not None
    assert powers.lf > powers.hf


def test_normalised_hf_separates_the_two_cases():
    """The bounded feature, which is what a cross-subject model can use."""
    hf_driven = band_powers(*_oscillating_rr(400, 0.25))
    lf_driven = band_powers(*_oscillating_rr(400, 0.08))
    assert hf_driven.normalised_hf > lf_driven.normalised_hf


def test_a_flat_series_has_no_meaningful_power_anywhere():
    times = [i * 1.0 for i in range(1, 401)]
    rr = [1000.0] * 400
    powers = band_powers(times, rr)
    assert powers.lf == pytest.approx(0.0, abs=1e-6)
    assert powers.hf == pytest.approx(0.0, abs=1e-6)


# --- windowing -------------------------------------------------------------


def test_one_result_is_returned_per_epoch_in_order():
    times, rr = _oscillating_rr(600, 0.25)
    starts = [0.0, 30.0, 60.0, 90.0]
    results = windowed_band_powers(times, rr, starts)
    assert len(results) == len(starts)
    assert all(isinstance(r, BandPowers) for r in results)


def test_an_epoch_with_too_little_surrounding_data_gets_no_estimate():
    """Sparse regions must not quietly produce numbers."""
    times, rr = _oscillating_rr(20, 0.25)
    results = windowed_band_powers(times, rr, [0.0])
    assert results[0].lf is None


def test_the_window_is_long_enough_for_the_lf_band():
    """Five minutes holds twelve cycles of the slowest LF component."""
    assert WINDOW_SECONDS >= 120.0
    assert WINDOW_SECONDS * LF_BAND[0] >= 4.0


def test_features_render_absent_estimates_as_zero_at_the_boundary():
    """Zero is used only where a model cannot accept None.

    This is the documented weakness of the feature set: it makes "not
    estimable" and "no power" identical to the model, which is a candidate
    explanation for why these features did not help.
    """
    assert BandPowers(None, None).as_features() == (0.0, 0.0, 0.0, 0.0)
    assert len(FREQUENCY_FEATURE_NAMES) == len(BandPowers(1.0, 1.0).as_features())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
