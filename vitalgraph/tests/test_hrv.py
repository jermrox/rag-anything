"""HRV maths verified against hand-computable reference vectors."""

import math

from vitalgraph.biometrics import hrv


def test_constant_rr_has_zero_variability():
    m = hrv.analyze([800.0] * 50, correct=False)
    assert m.rmssd_ms == 0.0
    assert m.sdnn_ms == 0.0
    assert m.pnn50_pct == 0.0


def test_mean_hr_from_rr():
    # 800 ms between beats == 75 bpm
    assert hrv.analyze([800.0] * 10, correct=False).mean_hr_bpm == 75.0


def test_rmssd_known_vector():
    # Alternating 800/850 -> every successive difference is exactly 50 ms,
    # so RMSSD == 50 by definition.
    assert math.isclose(hrv.rmssd([800.0, 850.0] * 25), 50.0)


def test_pnn50_is_strictly_greater_than_50ms():
    assert hrv.pnn50([800.0, 850.0] * 10) == 0.0      # exactly 50 does not count
    assert hrv.pnn50([800.0, 860.0] * 10) == 100.0    # 60 ms does


def test_sdnn_uses_sample_standard_deviation():
    values = [800.0, 810.0, 790.0, 820.0]
    mean = sum(values) / len(values)
    expected = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    assert math.isclose(hrv.sdnn(values), expected)


def test_artifact_correction_drops_ectopic_beat():
    accepted, rejected = hrv.correct_artifacts([800, 800, 2000, 800, 800])
    assert rejected == 1
    assert accepted == [800, 800, 800, 800]


def test_artifact_correction_does_not_cascade():
    # A single outlier must not disqualify the healthy beats that follow it,
    # which is why the reference is the last *accepted* beat.
    accepted, rejected = hrv.correct_artifacts([800, 2500, 805, 810, 800])
    assert rejected == 1
    assert len(accepted) == 4


def test_empty_and_single_beat_are_safe():
    assert hrv.analyze([]).n_beats == 0
    assert hrv.analyze([800.0]).rmssd_ms == 0.0


def test_baseline_requires_enough_history():
    assert hrv.baseline_deviation(24.0, [50.0, 48.0, 52.0]) is None


def test_baseline_deviation_sign_and_floor():
    baseline = [50.0, 48.0, 52.0, 49.0, 51.0, 50.0, 49.0]
    z = hrv.baseline_deviation(24.0, baseline)
    assert z is not None and z < 0

    # A perfectly flat baseline has zero variance; the SD floor stops the
    # z-score diverging to infinity.
    flat = hrv.baseline_deviation(40.0, [50.0] * 7)
    assert math.isfinite(flat)
    assert flat == (40.0 - 50.0) / hrv.MIN_BASELINE_SD_MS


def test_coverage_reports_fraction_surviving_correction():
    m = hrv.analyze([800, 800, 3000, 800])
    assert m.coverage == 0.75
    assert m.n_rejected == 1
