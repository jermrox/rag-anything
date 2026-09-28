"""The signal catalogue and its dependency graph."""

import pytest

from vitalgraph.biometrics.schema import SignalType
from vitalgraph.knowledge import signals as S


def test_catalogue_ids_are_unique():
    ids = [s.id for s in S.SIGNALS]
    assert len(ids) == len(set(ids))
    assert len(S.BY_ID) == len(S.SIGNALS)


def test_unknown_signal_raises():
    with pytest.raises(S.UnknownSignal):
        S.get("not_a_signal")


def test_every_derived_signal_names_real_parents():
    """A dangling parent id would silently break dependency resolution."""
    for signal in S.SIGNALS:
        for parent in signal.derived_from:
            assert parent in S.BY_ID, f"{signal.id} -> {parent}"


def test_measured_signals_have_no_parents():
    for signal in S.by_derivation(S.Derivation.MEASURED):
        assert signal.derived_from == ()


def test_derived_and_inferred_signals_have_parents():
    for derivation in (S.Derivation.DERIVED, S.Derivation.INFERRED):
        for signal in S.by_derivation(derivation):
            assert (
                signal.derived_from
            ), f"{signal.id} claims {derivation} with no source"


def test_dependency_resolution_reaches_the_root():
    """A domain asking for respiratory rate is really asking for RR intervals,
    and the sensor question has to be answered there."""
    assert S.root_signals(["respiratory_rate"]) == ["rr_interval"]
    assert S.root_signals(["sleep_stage"]) == ["acceleration", "rr_interval"]
    assert S.root_signals(["hrv_baseline_deviation"]) == ["rr_interval"]


def test_root_of_a_measured_signal_is_itself():
    assert S.root_signals(["spo2"]) == ["spo2"]


def test_dependency_resolution_terminates_on_cycles():
    """Guards against a future edit introducing a cycle and hanging."""
    for signal in S.SIGNALS:
        S.dependencies(signal.id)


def test_sampling_minimum_is_the_load_bearing_field():
    """Event detection needs a real rate; trend does not."""
    assert S.get("spo2").min_sampling_hz >= 0.2
    assert S.get("spo2").min_interval_s == pytest.approx(5.0)
    # Skin temperature is genuinely fine once a minute.
    assert S.get("skin_temperature").min_sampling_hz < 0.02


def test_episodic_signals_are_marked_non_continuous():
    for signal_id in ("blood_pressure", "cortisol", "body_mass", "training_load"):
        signal = S.get(signal_id)
        assert not signal.is_continuous
        assert signal.min_interval_s is None


def test_rhythm_grade_signals_demand_waveform_rates():
    """PPG-based rhythm claims are where wearables get into trouble."""
    assert S.get("ecg_waveform").min_sampling_hz >= 250.0
    assert S.get("qt_interval").derived_from == ("ecg_waveform",)


def test_ingested_mapping_matches_the_store_vocabulary():
    stored = {t.value for t in SignalType}
    for universe_id, stored_id in S.INGESTED_AS.items():
        assert universe_id in S.BY_ID
        assert stored_id in stored


def test_classes_partition_the_catalogue():
    counted = sum(len(S.by_class(c)) for c in S.SignalClass)
    assert counted == len(S.SIGNALS)
