"""Decoding a named device's raw BLE bytes into canonical Samples.

Every case here is really testing one guarantee: that decoding routes
through the device catalogue first, so bytes cannot be attributed to a
device that does not actually advertise the characteristic they came from.
"""

from datetime import datetime, timezone

import pytest

from vitalgraph.biometrics.schema import SignalType
from vitalgraph.ble.gatt import encode_heart_rate_measurement
from vitalgraph.devices.catalog import UnknownDevice
from vitalgraph.devices.connector import (
    DeviceNotDirectlyConnectable,
    UnsupportedCharacteristic,
    decode_heart_rate_notification,
    decode_pulse_oximeter_notification,
)

NOW = datetime(2026, 3, 4, tzinfo=timezone.utc)


def test_decodes_heart_rate_for_an_open_ble_device():
    payload = encode_heart_rate_measurement(58, [1020.0, 1030.0])
    decoded = decode_heart_rate_notification("polar_h10", payload, NOW)
    assert decoded.device_id == "polar_h10"
    hr = [s for s in decoded.samples if s.signal is SignalType.HEART_RATE]
    rr = [s for s in decoded.samples if s.signal is SignalType.RR_INTERVAL]
    assert hr[0].value == 58.0
    assert len(rr) == 2


def test_samples_are_attributed_to_the_named_device():
    payload = encode_heart_rate_measurement(60)
    decoded = decode_heart_rate_notification("wahoo_tickr", payload, NOW)
    assert all(s.source == "device:wahoo_tickr:0x2A37" for s in decoded.samples)


def test_rr_intervals_stamp_backwards_from_arrival():
    """RR intervals arrive oldest-first within a notification (per the GATT
    spec), so the last one in the list is closest to the arrival time and
    each earlier one is pushed further back by the accumulated offset."""
    payload = encode_heart_rate_measurement(60, [800.0, 850.0])
    decoded = decode_heart_rate_notification("polar_h10", payload, NOW)
    rr = [s for s in decoded.samples if s.signal is SignalType.RR_INTERVAL]
    assert all(s.ts < NOW for s in rr)
    assert rr[0].ts > rr[1].ts  # most recent beat (850ms, listed last) first


def test_cloud_api_device_is_refused_not_silently_decoded():
    """The central guarantee: a cloud-only device's bytes are not ours to
    decode, and asking must fail loudly rather than quietly succeeding."""
    payload = encode_heart_rate_measurement(60)
    with pytest.raises(DeviceNotDirectlyConnectable):
        decode_heart_rate_notification("whoop_4", payload)


def test_closed_device_is_refused():
    payload = encode_heart_rate_measurement(60)
    with pytest.raises(DeviceNotDirectlyConnectable):
        decode_heart_rate_notification("whoop_generic_ble", payload)


def test_unknown_device_raises_the_catalogue_error():
    payload = encode_heart_rate_measurement(60)
    with pytest.raises(UnknownDevice):
        decode_heart_rate_notification("not_a_real_device", payload)


def test_device_not_advertising_the_characteristic_is_refused():
    """A pulse oximeter does not advertise Heart Rate; decoding its bytes
    as one would misattribute the signal to a device that never sent it."""
    payload = encode_heart_rate_measurement(60)
    with pytest.raises(UnsupportedCharacteristic):
        decode_heart_rate_notification("nonin_pulse_ox", payload)


def test_malformed_payload_still_raises_the_gatt_decode_error():
    """Device validation must not swallow the underlying spec decoder's own
    errors -- a truncated payload is still a truncated payload."""
    from vitalgraph.ble.gatt import DecodeError

    with pytest.raises(DecodeError):
        decode_heart_rate_notification("polar_h10", bytes([0x00]))


def test_pulse_oximeter_decoding_for_the_device_that_supports_it():
    decoded = decode_pulse_oximeter_notification("nonin_pulse_ox", 96.5, NOW)
    assert decoded.samples[0].signal is SignalType.SPO2
    assert decoded.samples[0].value == 96.5


def test_pulse_oximeter_decoding_refused_for_a_device_without_the_service():
    with pytest.raises(UnsupportedCharacteristic):
        decode_pulse_oximeter_notification("polar_h10", 96.5)


def test_decoded_samples_are_the_canonical_type_the_store_accepts():
    """Decoded output must be indistinguishable from simulator or generic
    ingest output -- that interchangeability is the entire point."""
    from vitalgraph.biometrics.schema import Sample
    from vitalgraph.biometrics.store import BiometricStore

    payload = encode_heart_rate_measurement(60, [800.0])
    decoded = decode_heart_rate_notification("polar_h10", payload, NOW)
    assert all(isinstance(s, Sample) for s in decoded.samples)

    store = BiometricStore(":memory:")
    assert store.add(decoded.samples) == len(decoded.samples)
