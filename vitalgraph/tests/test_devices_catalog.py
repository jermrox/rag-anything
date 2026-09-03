"""The device catalogue: named products and honest reachability.

The central guarantee under test is that reachability claims never overstate
what a device actually exposes -- especially that a vendor's cloud API does
not upgrade the underlying sensor's physical limits.
"""

import pytest

from vitalgraph.devices.catalog import (
    DEVICES,
    AccessMode,
    DeviceProfile,
    UnknownDevice,
    by_access_mode,
    directly_connectable,
    get,
    requiring_vendor_api,
    supporting,
)
from vitalgraph.knowledge.sensors import UnknownSensor


def test_catalogue_loads_without_error():
    assert len(DEVICES) >= 10


def test_every_device_id_is_unique():
    ids = [d.id for d in DEVICES]
    assert len(ids) == len(set(ids))


def test_get_returns_the_right_profile():
    assert get("polar_h10").name == "Polar H10"


def test_unknown_device_raises():
    with pytest.raises(UnknownDevice):
        get("does_not_exist")


def test_open_ble_devices_declare_gatt_services():
    """A device claiming direct connectability with no service list would
    be an unfulfillable promise."""
    for d in by_access_mode(AccessMode.OPEN_BLE):
        assert d.gatt_services


def test_cloud_api_devices_declare_docs():
    for d in by_access_mode(AccessMode.CLOUD_API):
        assert d.api_docs_url


def test_open_ble_without_services_is_rejected():
    with pytest.raises(ValueError):
        DeviceProfile(
            "x", "X", "Vendor", AccessMode.OPEN_BLE, sensor_ids=("ppg_wrist",)
        )


def test_cloud_api_without_docs_is_rejected():
    with pytest.raises(ValueError):
        DeviceProfile(
            "x", "X", "Vendor", AccessMode.CLOUD_API, sensor_ids=("ppg_wrist",)
        )


def test_unknown_sensor_reference_is_rejected():
    with pytest.raises(UnknownSensor):
        DeviceProfile(
            "x",
            "X",
            "Vendor",
            AccessMode.OPEN_BLE,
            sensor_ids=("not_a_real_sensor",),
            gatt_services=("0x180D",),
        )


# --- the central guarantee --------------------------------------------------


def test_cloud_api_does_not_upgrade_the_underlying_sensor():
    """Whoop's API is reachable; wrist PPG's physical limits are not
    dissolved by having an API in front of them."""
    whoop = get("whoop_4")
    assert whoop.access_mode is AccessMode.CLOUD_API
    # Wrist PPG cannot clear SpO2's usable minimum -- neither can Whoop's API.
    assert whoop.meets_minimum("spo2") is False


def test_open_ble_chest_straps_support_rr_interval():
    for device_id in ("polar_h10", "wahoo_tickr", "garmin_hrm_strap"):
        assert get(device_id).meets_minimum("rr_interval") is True


def test_devices_that_do_not_deliver_a_signal_report_none_not_false():
    """None (absent) and False (present but too slow) must not collapse --
    they call for different fixes."""
    strap = get("polar_h10")
    assert strap.meets_minimum("core_temperature") is None


def test_directly_connectable_excludes_cloud_and_closed():
    ids = {d.id for d in directly_connectable()}
    assert "polar_h10" in ids
    assert "whoop_4" not in ids
    assert "whoop_generic_ble" not in ids


def test_requiring_vendor_api_is_exactly_the_cloud_class():
    assert {d.id for d in requiring_vendor_api()} == {
        d.id for d in DEVICES if d.access_mode is AccessMode.CLOUD_API
    }


def test_whoops_ble_link_is_separate_from_its_api():
    """The device-to-phone radio and the cloud API are different reachability
    facts about the same product and must not be conflated into one entry."""
    assert get("whoop_4").access_mode is AccessMode.CLOUD_API
    assert get("whoop_generic_ble").access_mode is AccessMode.CLOSED


def test_supporting_respects_the_minimum_flag():
    with_minimum = {d.id for d in supporting("spo2", minimum=True)}
    without_minimum = {d.id for d in supporting("spo2", minimum=False)}
    assert with_minimum <= without_minimum
    assert "whoop_4" in without_minimum
    assert "whoop_4" not in with_minimum


def test_signals_delivered_takes_the_best_rate_across_sensors():
    """A multi-sensor device (e.g. wrist PPG + on-demand ECG) should not be
    limited to its weakest sensor for a signal its other sensor covers."""
    watch = get("apple_watch")
    delivered = watch.signals_delivered()
    assert delivered["rr_interval"] > 0  # from the ECG sensor, not the PPG one
    assert watch.meets_minimum("rr_interval") is True


def test_medical_pulse_oximeter_clears_spo2_where_consumer_wrist_does_not():
    assert get("nonin_pulse_ox").meets_minimum("spo2") is True
    assert get("whoop_4").meets_minimum("spo2") is False
