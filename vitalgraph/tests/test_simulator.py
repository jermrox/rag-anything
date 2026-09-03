"""The simulator is both a dev fixture and the substitute for hardware,
so it must be deterministic and physiologically directional."""

from vitalgraph.biometrics import hrv
from vitalgraph.biometrics.schema import SignalType, utc
from vitalgraph.ble.simulator import simulate_night, simulate_period


def _rr(samples):
    return [s.value for s in samples if s.signal == SignalType.RR_INTERVAL]


def test_same_seed_produces_identical_output():
    a = simulate_night(utc(0), seed=7)
    b = simulate_night(utc(0), seed=7)
    assert [s.value for s in a] == [s.value for s in b]


def test_different_seeds_differ():
    a = simulate_night(utc(0), seed=1)
    b = simulate_night(utc(0), seed=2)
    assert [s.value for s in a] != [s.value for s in b]


def test_poor_recovery_suppresses_hrv_and_raises_heart_rate():
    good = hrv.analyze(_rr(simulate_night(utc(0), recovery=1.0, seed=3)))
    bad = hrv.analyze(_rr(simulate_night(utc(0), recovery=0.2, seed=3)))
    assert bad.rmssd_ms < good.rmssd_ms
    assert bad.mean_hr_bpm > good.mean_hr_bpm


def test_generated_beats_survive_artifact_correction():
    """Synthetic data must look clean. Low coverage would mean the generator
    is producing physiologically impossible beat-to-beat jumps."""
    m = hrv.analyze(_rr(simulate_night(utc(0), recovery=1.0, seed=5)))
    assert m.coverage > 0.85


def test_all_signal_channels_present():
    signals = {s.signal for s in simulate_night(utc(0), hours=1.0, seed=1)}
    assert signals == {
        SignalType.RR_INTERVAL, SignalType.HEART_RATE,
        SignalType.SLEEP_STAGE, SignalType.SPO2, SignalType.SKIN_TEMPERATURE,
    }


def test_night_covers_requested_duration():
    samples = simulate_night(utc(0), hours=2.0, seed=1)
    span = max(s.ts for s in samples) - min(s.ts for s in samples)
    assert 1.9 * 3600 <= span.total_seconds() <= 2.0 * 3600


def test_simulate_period_spaces_nights_one_day_apart():
    samples = simulate_period(utc(0), nights=3, seed=1)
    days = {s.ts.date() for s in samples}
    assert len(days) == 3


def test_recovery_list_length_is_validated():
    import pytest

    with pytest.raises(ValueError):
        simulate_period(utc(0), nights=3, recovery_by_night=[1.0])
