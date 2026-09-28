"""Tests for the BIDSLEEP loader.

Two of these guard bugs that were real, found on the first night processed,
and silent in both cases: the loader produced epochs, carried labels, and
raised nothing.

The first was clock alignment. ``recStart`` is local wall time with no zone;
reading it as UTC put the epoch grid five hours from the data and threw away
64% of the night as apparent sensor dropout.

The second was the stage mapping. Nothing verifies that code 3 means deep
sleep, so a transposed mapping would have produced a mediocre model and a week
spent blaming the features.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from vitalgraph.biometrics.schema import SleepStage
from vitalgraph.data.bidsleep import (
    FEATURE_NAMES,
    MAX_ALIGNMENT_RESIDUAL_S,
    MIN_HR_SAMPLES_PER_EPOCH,
    MIN_MOTION_SAMPLES_PER_EPOCH,
    STAGE_CODES,
    BidsleepError,
    align_start,
    build_epochs,
    check_label_mapping,
    night_url,
)
from vitalgraph.ml.epochs import EPOCH_SECONDS

MIDNIGHT_LOCAL = datetime(2021, 12, 2, 23, 11, 25, tzinfo=timezone.utc)


# --- clock alignment: the bug that cost 64% of the first night -------------


def test_a_local_recording_start_is_pulled_onto_the_signal_clock():
    """Five hours out is a timezone, not a disagreement."""
    naive = MIDNIGHT_LOCAL.timestamp()
    first_signal = naive + 5 * 3600 - 92.0  # US Eastern, signals 92 s early
    start_ts, residual = align_start(MIDNIGHT_LOCAL, first_signal)
    assert start_ts == pytest.approx(naive + 5 * 3600)
    assert residual == pytest.approx(-92.0)


def test_half_hour_timezones_are_handled():
    """India and Newfoundland exist and would otherwise fail the residual check."""
    naive = MIDNIGHT_LOCAL.timestamp()
    start_ts, residual = align_start(MIDNIGHT_LOCAL, naive + 5.5 * 3600 + 30)
    assert start_ts == pytest.approx(naive + 5.5 * 3600)
    assert abs(residual) < 60


def test_an_offset_that_is_not_a_timezone_is_refused():
    """Silently aligning this would shift every label against its signal."""
    naive = MIDNIGHT_LOCAL.timestamp()
    with pytest.raises(BidsleepError, match="cannot align"):
        align_start(MIDNIGHT_LOCAL, naive + 3600 + MAX_ALIGNMENT_RESIDUAL_S + 60)


def test_the_alignment_guard_is_reachable_at_all():
    """Rounding to half hours bounds the residual at 15 minutes on its own.

    A limit above that would be a guard that can never fire -- which is what
    the first version of this was, and the reason this test exists.
    """
    assert MAX_ALIGNMENT_RESIDUAL_S < 900


def test_an_already_aligned_clock_is_left_alone():
    naive = MIDNIGHT_LOCAL.timestamp()
    start_ts, residual = align_start(MIDNIGHT_LOCAL, naive + 3.0)
    assert start_ts == pytest.approx(naive)
    assert residual == pytest.approx(3.0)


# --- the stage mapping -----------------------------------------------------


def _night(stages, start=MIDNIGHT_LOCAL, hz=50.0):
    """Build a synthetic night whose signals sit exactly on the epoch grid."""
    start_ts = start.timestamp()
    heart_rate = []
    motion = []
    for i, _ in enumerate(stages):
        base = start_ts + i * EPOCH_SECONDS
        for k in range(6):  # 0.2 Hz, as the Apple Watch reports
            heart_rate.append((base + k * 5.0, 60.0 + (i % 7)))
        step = 1.0 / hz
        for k in range(int(EPOCH_SECONDS * hz)):
            t = base + k * step
            wobble = 0.002 * ((i + k) % 3)
            motion.append((t, 0.01 + wobble, 0.02, -0.98 + wobble))
    return start, heart_rate, motion


def test_the_stage_codes_cover_the_five_aasm_stages():
    assert set(STAGE_CODES) == {0, 1, 2, 3, 4}
    assert STAGE_CODES[0] is SleepStage.AWAKE
    assert STAGE_CODES[3] is SleepStage.DEEP
    assert STAGE_CODES[4] is SleepStage.REM
    # N1 and N2 both collapse to light, matching the slpdb label space.
    assert STAGE_CODES[1] is STAGE_CODES[2] is SleepStage.LIGHT


def test_a_realistic_night_passes_the_architecture_check():
    """Deep early, REM late -- true of every human night."""
    stages = [3] * 40 + [2] * 30 + [4] * 50  # deep first third, REM last
    start, hr, motion = _night(stages)
    samples, _ = build_epochs("s/1", start, stages, hr, motion)
    check = check_label_mapping(samples)
    assert check["consistent_with_sleep_architecture"] is True


def test_a_transposed_mapping_is_caught_rather_than_producing_a_bad_model():
    """The whole reason the check exists: REM early and deep late is not sleep."""
    stages = [4] * 40 + [2] * 30 + [3] * 50  # REM first, deep last -- inverted
    start, hr, motion = _night(stages)
    samples, _ = build_epochs("s/1", start, stages, hr, motion)
    check = check_label_mapping(samples)
    assert check["consistent_with_sleep_architecture"] is False


# --- building epochs -------------------------------------------------------


def test_every_scored_epoch_with_both_signals_becomes_one_sample():
    stages = [0, 1, 2, 3, 4] * 4
    start, hr, motion = _night(stages)
    samples, report = build_epochs("s/1", start, stages, hr, motion)
    assert len(samples) == len(stages)
    assert report["usable_epochs"] == len(stages)
    assert report["usable_fraction"] == 1.0


def test_the_feature_vector_matches_the_declared_names():
    stages = [2] * 10
    start, hr, motion = _night(stages)
    samples, _ = build_epochs("s/1", start, stages, hr, motion)
    assert len(samples[0].values) == len(FEATURE_NAMES)


def test_an_epoch_missing_motion_is_dropped_not_imputed():
    """A watch that stopped reporting is not a still, resting subject."""
    stages = [2] * 6
    start, hr, motion = _night(stages)
    gap_start = start.timestamp() + 2 * EPOCH_SECONDS
    thinned = [
        row for row in motion if not (gap_start <= row[0] < gap_start + EPOCH_SECONDS)
    ]
    samples, report = build_epochs("s/1", start, stages, hr, thinned)
    assert [s.index for s in samples] == [0, 1, 3, 4, 5]
    assert report["dropped_no_motion"] == 1


def test_an_epoch_missing_heart_rate_is_dropped_not_imputed():
    stages = [2] * 6
    start, hr, motion = _night(stages)
    gap_start = start.timestamp() + 4 * EPOCH_SECONDS
    thinned = [s for s in hr if not (gap_start <= s[0] < gap_start + EPOCH_SECONDS)]
    samples, report = build_epochs("s/1", start, stages, thinned, motion)
    assert 4 not in [s.index for s in samples]
    assert report["dropped_no_heart_rate"] == 1


def test_a_night_with_no_usable_epoch_raises_rather_than_returning_nothing():
    stages = [2] * 4
    start, hr, motion = _night(stages)
    with pytest.raises(BidsleepError, match="no epoch has both signals"):
        build_epochs("s/1", start, stages, hr, motion[:10])


def test_implausible_heart_rates_are_dropped_before_the_mean():
    """A PPG dropout reports a number, and it would move the average."""
    stages = [2] * 3
    start, hr, motion = _night(stages)
    poisoned = list(hr) + [(start.timestamp() + 1.0, 400.0)]
    samples, _ = build_epochs("s/1", start, stages, poisoned, motion)
    hr_mean = samples[0].values[FEATURE_NAMES.index("hr_mean")]
    assert hr_mean < 100.0


def test_motion_thresholds_are_a_meaningful_fraction_of_the_sample_rate():
    """150 samples is a tenth of 30 s at 50 Hz -- a gap, not a still period."""
    assert MIN_MOTION_SAMPLES_PER_EPOCH >= 100
    assert MIN_HR_SAMPLES_PER_EPOCH >= 2


def test_posture_change_is_zero_for_the_first_epoch_and_small_when_still():
    stages = [2] * 5
    start, hr, motion = _night(stages)
    samples, _ = build_epochs("s/1", start, stages, hr, motion)
    idx = FEATURE_NAMES.index("posture_change")
    assert samples[0].values[idx] == 0.0
    assert all(s.values[idx] < 5.0 for s in samples)  # degrees, barely moving


def test_labels_outside_the_known_codes_are_dropped():
    stages = [2, 2, 9, 2]  # 9 is not an AASM stage
    start, hr, motion = _night(stages)
    samples, _ = build_epochs("s/1", start, stages, hr, motion)
    assert [s.index for s in samples] == [0, 1, 3]


def test_epoch_order_and_night_id_are_preserved():
    stages = [2] * 8
    start, hr, motion = _night(stages)
    samples, _ = build_epochs("subj/3", start, stages, hr, motion)
    assert [s.index for s in samples] == sorted(s.index for s in samples)
    assert all(s.night_id == "subj/3" for s in samples)


# --- urls ------------------------------------------------------------------


def test_night_urls_point_at_the_published_dataset():
    url = night_url("Bidslab00", 1, "hr.csv")
    assert url.startswith("https://physionet.org/files/bidsleep-dataset/1.0.0/")
    assert url.endswith("/Bidslab00/1/hr.csv")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
