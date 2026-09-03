"""Trend projection and sustained-decline detection.

Pure standard library, so these run without the ML extra installed.
"""

import random

import pytest

from vitalgraph.biometrics.schema import utc
from vitalgraph.biometrics.store import BiometricStore
from vitalgraph.ble.simulator import simulate_period
from vitalgraph.bridge.summarizer import nightly_windows
from vitalgraph.ml.features import features_for_windows
from vitalgraph.ml.forecast import (
    MIN_HISTORY,
    SUSTAINED_PERIODS,
    InsufficientHistory,
    detect_sustained_decline,
    forecast_from_vectors,
    forecast_series,
    holt_linear,
    to_content_list,
)

START = utc(1772582400)


def test_recovers_the_slope_of_an_exact_line():
    f = forecast_series([10, 12, 14, 16, 18, 20, 22], "sdnn_ms", horizon=3)
    assert f.slope == pytest.approx(2.0, abs=0.05)
    assert f.predictions[0].value == pytest.approx(24.0, abs=0.2)


def test_projection_extends_the_trend():
    f = forecast_series([10, 12, 14, 16, 18, 20], "sdnn_ms", horizon=3)
    values = [p.value for p in f.predictions]
    assert values == sorted(values)  # monotonically continuing upward


def test_intervals_widen_with_the_horizon():
    """A seven-day projection must not look as confident as a one-day one."""
    noisy = [50, 53, 49, 55, 48, 54, 47, 56]
    f = forecast_series(noisy, "rmssd_ms", horizon=5)
    widths = [p.upper - p.lower for p in f.predictions]
    assert widths == sorted(widths)
    assert widths[-1] > widths[0]


def test_noise_is_not_reported_as_a_trend():
    rng = random.Random(1)
    flat = [50 + rng.gauss(0, 3) for _ in range(14)]
    assert forecast_series(flat, "rmssd_ms").direction == "stable"


def test_direction_respects_which_way_is_good():
    falling = [55, 52, 48, 44, 40, 36]
    rising = [55, 58, 62, 66, 70, 74]
    # Falling HRV is bad; falling resting heart rate is good.
    assert forecast_series(falling, "rmssd_ms").direction == "declining"
    assert forecast_series(falling, "mean_hr_bpm").direction == "improving"
    assert forecast_series(rising, "mean_hr_bpm").direction == "declining"


def test_short_history_raises_rather_than_guessing():
    with pytest.raises(InsufficientHistory):
        forecast_series([1.0] * (MIN_HISTORY - 1), "rmssd_ms")


def test_forecasting_is_deterministic():
    series = [55, 54, 52, 49, 45, 40, 34]
    assert (
        forecast_series(series, "rmssd_ms").as_dict()
        == forecast_series(series, "rmssd_ms").as_dict()
    )


def test_holt_on_a_constant_series_has_no_slope():
    _, slope, _ = holt_linear([7.0] * 10, alpha=0.3, beta=0.1)
    assert slope == pytest.approx(0.0, abs=1e-9)


# --- sustained decline -----------------------------------------------------


def test_sustained_run_is_detected():
    signal = detect_sustained_decline([55, 54, 52, 49, 45, 40], "rmssd_ms")
    assert signal.is_declining
    assert signal.consecutive >= SUSTAINED_PERIODS
    assert signal.change < 0


def test_one_bad_period_is_not_a_decline():
    """A single poor night follows a hard session; it is not a pattern."""
    assert not detect_sustained_decline([50, 51, 50, 52, 44], "rmssd_ms").is_declining


def test_decline_direction_is_metric_aware():
    rising_hr = [54, 57, 60, 64, 68]
    assert detect_sustained_decline(rising_hr, "mean_hr_bpm").is_declining
    assert not detect_sustained_decline(rising_hr, "rmssd_ms").is_declining


def test_decline_needs_history():
    assert not detect_sustained_decline([50, 49], "rmssd_ms").is_declining


# --- integration and rendering ---------------------------------------------


def test_forecast_from_simulated_nights():
    store = BiometricStore(":memory:")
    script = [1.0, 0.95, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15]
    store.add(simulate_period(START, nights=8, recovery_by_night=script, seed=5))
    vectors = features_for_windows(store, nightly_windows(START, 8))
    store.close()

    f = forecast_from_vectors(vectors, "rmssd_ms", horizon=3)
    assert f.direction == "declining"
    assert f.n_history == len(vectors)


def test_content_list_leads_with_the_actionable_finding():
    f = forecast_series([55, 54, 52, 49, 45, 40], "rmssd_ms", horizon=3)
    d = detect_sustained_decline([55, 54, 52, 49, 45, 40], "rmssd_ms")
    items = to_content_list([f], [d])

    assert items[0]["type"] == "text"
    assert "Sustained adverse trends" in items[0]["text"]
    assert items[1]["type"] == "table"
    assert "rmssd_ms" in items[1]["table_body"]
    assert "extrapolation" in items[1]["table_footnote"][0]


def test_content_list_is_empty_without_input():
    assert to_content_list([], []) == []
