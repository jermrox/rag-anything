"""Tests for the multi-field GATT health measurements.

Payloads are built by hand from the specification rather than captured, so a
test that passes proves the decoder matches the spec rather than matching one
device's quirks.
"""

from __future__ import annotations

import struct
from datetime import datetime

import pytest

from vitalgraph.ble.gatt import DecodeError
from vitalgraph.ble.measurements import (
    BP_STATUS_FLAGS,
    decode_blood_pressure,
    decode_body_composition,
    decode_cycling_cadence,
    decode_datetime,
    decode_glucose,
    decode_running_cadence,
    decode_sfloat,
    decode_weight_measurement,
)


def _datetime_bytes(dt: datetime) -> bytes:
    return struct.pack(
        "<HBBBBB", dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
    )


# --- SFLOAT ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0x0078, 120.0),  # exponent 0, mantissa 120 -- a systolic reading
        (0x0050, 80.0),  # diastolic
        (0xF03C, 6.0),  # exponent -1, mantissa 60 -> 6.0
        (0x0FFF, -1.0),  # mantissa -1 via two's complement
    ],
)
def test_decode_sfloat_values(raw, expected):
    assert decode_sfloat(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", [0x07FF, 0x0800, 0x07FE, 0x0802])
def test_sfloat_exceptional_values_are_none_not_numbers(raw):
    """NaN, NRes and the infinities mean "no measurement".

    Mapping them onto numbers turns a failed cuff reading into a systolic of
    2047, which is worse than reporting nothing.
    """
    assert decode_sfloat(raw) is None


# --- Date Time -------------------------------------------------------------


def test_decode_datetime_roundtrip():
    dt = datetime(2026, 9, 3, 14, 30, 15)
    parsed, offset = decode_datetime(_datetime_bytes(dt))
    assert parsed == dt
    assert offset == 7


@pytest.mark.parametrize(
    "raw",
    [
        struct.pack("<HBBBBB", 0, 9, 3, 14, 30, 15),  # year unknown
        struct.pack("<HBBBBB", 2026, 0, 3, 14, 30, 15),  # month unknown
        struct.pack("<HBBBBB", 2026, 9, 0, 14, 30, 15),  # day unknown
    ],
)
def test_unknown_datetime_components_yield_none_not_a_fabricated_date(raw):
    parsed, offset = decode_datetime(raw)
    assert parsed is None
    assert offset == 7


def test_invalid_datetime_raises_rather_than_silently_shifting():
    with pytest.raises(DecodeError):
        decode_datetime(struct.pack("<HBBBBB", 2026, 13, 40, 14, 30, 15))


def test_truncated_datetime_raises():
    with pytest.raises(DecodeError, match="truncated"):
        decode_datetime(b"\x01\x02\x03")


# --- Body Composition (0x2A9C) ---------------------------------------------


def test_body_composition_minimal_payload():
    """Flags plus body fat percentage only -- every other field absent."""
    payload = struct.pack("<HH", 0x0000, 224)  # 22.4%
    bc = decode_body_composition(payload)
    assert bc.body_fat_percent == pytest.approx(22.4)
    assert bc.imperial is False
    assert bc.muscle_mass is None
    assert bc.weight is None


def test_body_composition_full_si_payload():
    flags = (
        (1 << 4)  # muscle percentage
        | (1 << 5)  # muscle mass
        | (1 << 6)  # fat free mass
        | (1 << 9)  # impedance
        | (1 << 10)  # weight
        | (1 << 11)  # height
    )
    payload = struct.pack(
        "<HHHHHHHH",
        flags,
        224,  # body fat 22.4%
        412,  # muscle 41.2%
        7000,  # muscle mass 35.000 kg
        12800,  # fat free mass 64.000 kg
        4850,  # impedance 485.0 ohm
        16240,  # weight 81.200 kg
        1780,  # height 1.780 m
    )
    bc = decode_body_composition(payload)
    assert bc.body_fat_percent == pytest.approx(22.4)
    assert bc.muscle_percent == pytest.approx(41.2)
    assert bc.muscle_mass == pytest.approx(35.0)
    assert bc.fat_free_mass == pytest.approx(64.0)
    assert bc.impedance_ohm == pytest.approx(485.0)
    assert bc.weight == pytest.approx(81.2)
    assert bc.height == pytest.approx(1.78)
    assert bc.mass_unit == "kg"


def test_body_composition_imperial_uses_different_mass_resolution():
    """Reading an Imperial payload with SI scaling is a silent 2x error."""
    si = struct.pack("<HHH", 0x0400, 224, 16240)
    imperial = struct.pack("<HHH", 0x0401, 224, 16240)

    assert decode_body_composition(si).weight == pytest.approx(81.2)
    assert decode_body_composition(imperial).weight == pytest.approx(162.4)
    assert decode_body_composition(imperial).mass_unit == "lb"


def test_body_composition_unknown_field_is_none_not_zero():
    """0xFFFF means "not measured". Zero body fat is a very different claim."""
    payload = struct.pack("<HH", 0x0000, 0xFFFF)
    assert decode_body_composition(payload).body_fat_percent is None


def test_body_composition_optional_fields_are_read_in_specification_order():
    """Field order is fixed and positional; reordering misassigns everything.

    Muscle mass and weight are both uint16 masses. If the walk skipped or
    reordered a flag, weight would be read from the muscle-mass bytes and the
    error would be invisible in the output.
    """
    flags = (1 << 5) | (1 << 10)  # muscle mass, then weight
    payload = struct.pack("<HHHH", flags, 224, 7000, 16240)
    bc = decode_body_composition(payload)
    assert bc.muscle_mass == pytest.approx(35.0)
    assert bc.weight == pytest.approx(81.2)


def test_body_composition_timestamp_and_user_id():
    dt = datetime(2026, 9, 3, 7, 15, 0)
    flags = (1 << 1) | (1 << 2)
    payload = struct.pack("<HH", flags, 224) + _datetime_bytes(dt) + bytes([3])
    bc = decode_body_composition(payload)
    assert bc.timestamp == dt
    assert bc.user_id == 3


def test_body_composition_lean_mass_prefers_the_measured_field():
    """A measured fat-free mass is a measurement; computing one is a derivation."""
    measured = decode_body_composition(struct.pack("<HHH", 1 << 6, 224, 12800))
    assert measured.lean_mass == pytest.approx(64.0)

    derived = decode_body_composition(struct.pack("<HHH", 1 << 10, 250, 16000))
    assert derived.lean_mass == pytest.approx(80.0 * 0.75)


def test_body_composition_truncated_payload_raises():
    with pytest.raises(DecodeError):
        decode_body_composition(struct.pack("<HH", 1 << 10, 224))


def test_body_composition_rejects_short_payload():
    with pytest.raises(DecodeError, match="too short"):
        decode_body_composition(b"\x00\x00")


# --- Weight Measurement (0x2A9D) -------------------------------------------


def test_weight_measurement_si():
    wm = decode_weight_measurement(bytes([0x00]) + struct.pack("<H", 16240))
    assert wm.weight == pytest.approx(81.2)
    assert wm.mass_unit == "kg"


def test_weight_measurement_with_bmi_and_height():
    flags = 1 << 3
    payload = bytes([flags]) + struct.pack("<HHH", 16240, 256, 1780)
    wm = decode_weight_measurement(payload)
    assert wm.weight == pytest.approx(81.2)
    assert wm.bmi == pytest.approx(25.6)
    assert wm.height == pytest.approx(1.78)


def test_weight_measurement_timestamp_then_user_id_order():
    dt = datetime(2026, 9, 3, 6, 0, 0)
    flags = (1 << 1) | (1 << 2)
    payload = bytes([flags]) + struct.pack("<H", 16240) + _datetime_bytes(dt)
    payload += bytes([7])
    wm = decode_weight_measurement(payload)
    assert wm.timestamp == dt
    assert wm.user_id == 7


# --- Blood Pressure (0x2A35) -----------------------------------------------


def test_blood_pressure_basic():
    payload = bytes([0x00]) + struct.pack("<HHH", 0x0078, 0x0050, 0x005D)
    bp = decode_blood_pressure(payload)
    assert bp.systolic == pytest.approx(120.0)
    assert bp.diastolic == pytest.approx(80.0)
    assert bp.mean_arterial_pressure == pytest.approx(93.0)
    assert bp.unit == "mmHg"
    assert bp.is_reliable is True
    assert bp.is_complete is True


def test_blood_pressure_kpa_flag_changes_the_unit():
    payload = bytes([0x01]) + struct.pack("<HHH", 0x0010, 0x000B, 0x000C)
    assert decode_blood_pressure(payload).unit == "kPa"


def test_blood_pressure_surfaces_device_reported_problems():
    """A cuff that detected motion still returns numbers.

    Dropping the status word would present an unreliable reading as a clean
    one, which is the failure mode that matters clinically.
    """
    flags = 1 << 4
    status = (1 << 0) | (1 << 2)  # body movement, irregular pulse
    payload = bytes([flags]) + struct.pack("<HHHH", 0x0078, 0x0050, 0x005D, status)
    bp = decode_blood_pressure(payload)
    assert bp.is_reliable is False
    assert "body movement detected" in bp.status_flags
    assert "irregular pulse detected" in bp.status_flags


def test_blood_pressure_nan_reading_is_incomplete_not_a_number():
    payload = bytes([0x00]) + struct.pack("<HHH", 0x07FF, 0x0050, 0x005D)
    bp = decode_blood_pressure(payload)
    assert bp.systolic is None
    assert bp.is_complete is False


def test_blood_pressure_pulse_rate_and_user_id():
    dt = datetime(2026, 9, 3, 8, 0, 0)
    flags = (1 << 1) | (1 << 2) | (1 << 3)
    payload = bytes([flags]) + struct.pack("<HHH", 0x0078, 0x0050, 0x005D)
    payload += _datetime_bytes(dt) + struct.pack("<H", 0x0040) + bytes([2])
    bp = decode_blood_pressure(payload)
    assert bp.timestamp == dt
    assert bp.pulse_rate == pytest.approx(64.0)
    assert bp.user_id == 2


def test_every_declared_status_flag_has_a_label():
    assert all(label and bit for bit, label in BP_STATUS_FLAGS)


# --- Glucose (0x2A18) ------------------------------------------------------


def test_glucose_mg_dl_conversion():
    """The wire carries kg/L; nobody reads glucose in kg/L."""
    dt = datetime(2026, 9, 3, 12, 0, 0)
    flags = 1 << 1  # concentration present, kg/L units
    # 0.001 kg/L -> 100 mg/dL. SFLOAT: mantissa 1, exponent -3.
    sfloat = ((-3 & 0x0F) << 12) | 1
    payload = bytes([flags]) + struct.pack("<H", 42) + _datetime_bytes(dt)
    payload += struct.pack("<H", sfloat) + bytes([0x11])  # capillary, finger
    g = decode_glucose(payload)
    assert g.sequence_number == 42
    assert g.base_time == dt
    assert g.concentration_mg_dl == pytest.approx(100.0)
    assert g.concentration_mmol_l is None
    assert g.sample_type == "capillary whole blood"
    assert g.sample_location == "finger"


def test_glucose_mmol_flag_selects_the_other_unit():
    dt = datetime(2026, 9, 3, 12, 0, 0)
    flags = (1 << 1) | (1 << 2)
    sfloat = ((-3 & 0x0F) << 12) | 5  # 0.005 mol/L -> 5 mmol/L
    payload = bytes([flags]) + struct.pack("<H", 1) + _datetime_bytes(dt)
    payload += struct.pack("<H", sfloat) + bytes([0x11])
    g = decode_glucose(payload)
    assert g.concentration_mmol_l == pytest.approx(5.0)
    assert g.concentration_mg_dl is None


def test_interstitial_glucose_is_labelled_as_a_different_measurand():
    """Interstitial glucose lags blood glucose and is not the same thing."""
    dt = datetime(2026, 9, 3, 12, 0, 0)
    flags = 1 << 1
    sfloat = ((-3 & 0x0F) << 12) | 1
    payload = bytes([flags]) + struct.pack("<H", 1) + _datetime_bytes(dt)
    payload += struct.pack("<H", sfloat) + bytes([0x09])  # type 9
    g = decode_glucose(payload)
    assert g.sample_type == "interstitial fluid"
    assert g.is_interstitial is True


def test_glucose_time_offset_is_signed():
    """A negative offset means the reading predates the base time."""
    dt = datetime(2026, 9, 3, 12, 0, 0)
    payload = bytes([1 << 0]) + struct.pack("<H", 1) + _datetime_bytes(dt)
    payload += struct.pack("<h", -90)
    assert decode_glucose(payload).time_offset_minutes == -90


# --- Running cadence (0x2A53) ----------------------------------------------


def test_running_cadence_speed_scaling():
    """Speed is 1/256 m/s, not m/s."""
    payload = bytes([0x00]) + struct.pack("<H", 768) + bytes([180])
    r = decode_running_cadence(payload)
    assert r.speed_m_s == pytest.approx(3.0)
    assert r.speed_kph == pytest.approx(10.8)
    assert r.cadence_spm == 180
    assert r.running is False


def test_running_cadence_optional_fields_and_running_flag():
    flags = (1 << 0) | (1 << 1) | (1 << 2)
    payload = bytes([flags]) + struct.pack("<H", 768) + bytes([180])
    payload += struct.pack("<H", 120) + struct.pack("<I", 52340)
    r = decode_running_cadence(payload)
    assert r.running is True
    assert r.stride_length_m == pytest.approx(1.20)
    assert r.total_distance_m == pytest.approx(5234.0)


def test_running_cadence_truncated_optional_field_raises():
    payload = bytes([1 << 1]) + struct.pack("<H", 768) + bytes([180]) + b"\x01"
    with pytest.raises(DecodeError, match="Total Distance"):
        decode_running_cadence(payload)


# --- Cycling cadence (0x2A5B) ----------------------------------------------


def test_cycling_cadence_wheel_and_crank():
    flags = (1 << 0) | (1 << 1)
    payload = bytes([flags]) + struct.pack("<IH", 12345, 1024)
    payload += struct.pack("<HH", 678, 2048)
    c = decode_cycling_cadence(payload)
    assert c.cumulative_wheel_revolutions == 12345
    assert c.last_wheel_event_time_s == pytest.approx(1.0)
    assert c.cumulative_crank_revolutions == 678
    assert c.last_crank_event_time_s == pytest.approx(2.0)


def test_cycling_cadence_absent_blocks_are_none():
    c = decode_cycling_cadence(bytes([0x00]))
    assert c.cumulative_wheel_revolutions is None
    assert c.cumulative_crank_revolutions is None


def test_cycling_cadence_crank_only_reads_from_the_right_offset():
    """With the wheel flag clear, crank data starts immediately after flags."""
    payload = bytes([1 << 1]) + struct.pack("<HH", 678, 2048)
    c = decode_cycling_cadence(payload)
    assert c.cumulative_crank_revolutions == 678
    assert c.cumulative_wheel_revolutions is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
