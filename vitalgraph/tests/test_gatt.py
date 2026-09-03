"""GATT decoding must match the published Bluetooth SIG layouts exactly."""

import pytest

from vitalgraph.ble import gatt


def test_uint8_heart_rate_no_rr():
    m = gatt.decode_heart_rate_measurement(bytes([0x00, 60]))
    assert m.heart_rate_bpm == 60
    assert m.rr_intervals_ms == []
    assert not m.has_rr


def test_uint16_heart_rate():
    # flags bit0 set -> uint16 little endian
    m = gatt.decode_heart_rate_measurement(bytes([0x01, 0x2C, 0x01]))
    assert m.heart_rate_bpm == 300


def test_rr_intervals_use_1024ths_of_a_second():
    # 1024 raw units == exactly 1000 ms. Getting this scaling wrong is the
    # classic HRV integration bug, so pin it.
    payload = bytes([0x10, 60]) + (1024).to_bytes(2, "little")
    m = gatt.decode_heart_rate_measurement(payload)
    assert m.rr_intervals_ms == [1000.0]


def test_energy_expended_offsets_rr_block():
    # flags: energy present (0x08) + RR present (0x10)
    payload = (
        bytes([0x18, 60]) + (500).to_bytes(2, "little") + (512).to_bytes(2, "little")
    )
    m = gatt.decode_heart_rate_measurement(payload)
    assert m.energy_expended_kj == 500
    assert m.rr_intervals_ms == [500.0]


def test_sensor_contact_flags():
    m = gatt.decode_heart_rate_measurement(bytes([0x06, 60]))
    assert m.sensor_contact_supported
    assert m.sensor_contact_detected


@pytest.mark.parametrize("rr", [[800.0], [800.0, 850.0, 900.0], []])
def test_encode_decode_roundtrip(rr):
    decoded = gatt.decode_heart_rate_measurement(
        gatt.encode_heart_rate_measurement(72, rr)
    )
    assert decoded.heart_rate_bpm == 72
    assert len(decoded.rr_intervals_ms) == len(rr)
    # 1/1024 s quantisation means round-trip is exact only to ~1 ms.
    for got, want in zip(decoded.rr_intervals_ms, rr):
        assert abs(got - want) < 1.0


def test_truncated_payload_rejected():
    with pytest.raises(gatt.DecodeError):
        gatt.decode_heart_rate_measurement(bytes([0x00]))


def test_odd_rr_block_rejected():
    with pytest.raises(gatt.DecodeError):
        gatt.decode_heart_rate_measurement(bytes([0x10, 60, 0x00]))


def test_battery_and_location():
    assert gatt.decode_battery_level(bytes([87])) == 87
    assert gatt.decode_body_sensor_location(bytes([2])) == "Wrist"
    with pytest.raises(gatt.DecodeError):
        gatt.decode_battery_level(bytes([1, 2]))


def test_derivable_signals_reports_hrv_from_standard_characteristic():
    d = gatt.derivable_signals(["0x2a37", "0xFFFF"])
    assert "rmssd" in d["0x2A37"]
    assert "0xFFFF" not in d
