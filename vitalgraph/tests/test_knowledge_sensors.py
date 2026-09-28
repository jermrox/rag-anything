"""The sensor taxonomy, and the adequacy arithmetic built on it."""

import pytest

from vitalgraph.knowledge import sensors as N
from vitalgraph.knowledge import signals as S


def test_taxonomy_ids_are_unique():
    ids = [s.id for s in N.SENSORS]
    assert len(ids) == len(set(ids))


def test_unknown_sensor_raises():
    with pytest.raises(N.UnknownSensor):
        N.get("not_a_sensor")


def test_sensors_cannot_declare_signals_outside_the_catalogue():
    """The two halves of the knowledge base must not drift apart."""
    with pytest.raises(S.UnknownSignal):
        N.Sensor(
            "bogus",
            "Bogus",
            N.Modality.INERTIAL,
            (N.Site.WRIST,),
            {"invented_signal": 1.0},
            N.WearBurden.PASSIVE,
        )


def test_every_sensor_delivers_something():
    for sensor in N.SENSORS:
        assert sensor.delivers, sensor.id


def test_wrist_ppg_spo2_is_inadequate_for_events():
    """The core of the signal-adequacy critique, as arithmetic rather than
    argument: wrist SpO2 is reported far below the rate event detection needs."""
    wrist = N.get("ppg_wrist")
    assert wrist.rate_for("spo2") < S.get("spo2").min_sampling_hz
    assert wrist.meets_minimum("spo2") is False


def test_finger_and_ear_ppg_do_clear_the_spo2_minimum():
    for sensor_id in ("ppg_finger", "ppg_ear", "ppg_forehead"):
        assert N.get(sensor_id).meets_minimum("spo2") is True


def test_not_delivered_is_distinct_from_delivered_too_slowly():
    """Collapsing these two answers into one is how a capability report
    becomes misleading: 'no sensor' and 'wrong sensor' need different fixes."""
    assert N.get("imu").meets_minimum("spo2") is None
    assert N.get("ppg_wrist").meets_minimum("spo2") is False


def test_beat_accurate_intervals_come_only_from_electrical_sensors():
    adequate = {s.id for s in N.providers_of("rr_interval", adequate_only=True)}
    assert adequate == {"ecg_chest_strap", "ecg_patch", "ecg_handheld"}
    for sensor_id in adequate:
        assert N.get(sensor_id).modality is N.Modality.ELECTRICAL_CARDIAC


def test_waveform_morphology_needs_a_patch_or_handheld():
    providers = {s.id for s in N.providers_of("ecg_waveform")}
    assert providers == {"ecg_patch", "ecg_handheld"}
    assert "ecg_chest_strap" not in providers  # intervals only, no morphology


def test_glucose_is_reachable_only_by_cgm():
    """No cardiac or optical wearable signal reaches metabolic data."""
    assert [s.id for s in N.providers_of("interstitial_glucose")] == ["cgm"]


def test_eeg_is_reachable_only_at_the_scalp():
    providers = N.providers_of("eeg")
    assert [s.id for s in providers] == ["eeg_headband"]
    assert N.Site.WRIST not in providers[0].sites


def test_coverage_takes_the_best_rate_across_a_stack():
    """A real product wears several devices, each contributing what it does best."""
    stack = N.coverage(["ppg_wrist", "ecg_patch"])
    # The patch's beat-accurate intervals beat the wrist's pulse intervals.
    assert stack["rr_interval"] >= S.get("rr_interval").min_sampling_hz
    assert stack["heart_rate"] >= N.get("ppg_wrist").rate_for("heart_rate")


def test_a_wrist_only_stack_covers_far_less_than_a_multi_site_one():
    wrist_only = N.coverage(["ppg_wrist", "imu", "temp_skin"])
    multi_site = N.coverage(
        ["ppg_finger", "ecg_patch", "cgm", "env_suite", "self_report", "bcg_mat"]
    )
    assert len(multi_site) > 2 * len(wrist_only)


def test_lookups_by_axis_are_consistent():
    assert N.by_modality(N.Modality.ELECTRICAL_CARDIAC)
    assert N.by_site(N.Site.WRIST)
    assert N.by_burden(N.WearBurden.INVASIVE)
    counted = sum(len(N.by_modality(m)) for m in N.Modality)
    assert counted == len(N.SENSORS)


def test_gatt_services_link_to_the_protocol_layer():
    """Sensors exposing standard services tie this taxonomy to mined facts."""
    assert "0x180D" in N.get("ecg_chest_strap").gatt_services
    assert "0x1809" in N.get("temp_skin").gatt_services


def test_self_report_is_modelled_as_a_first_class_input():
    """Context the sensors cannot supply -- medication, training load -- and
    without it a measured deviation cannot be attributed to a cause."""
    sensor = N.get("self_report")
    assert "medication" in sensor.delivers
    assert "training_load" in sensor.delivers
    assert sensor.sites == (N.Site.NONE,)
