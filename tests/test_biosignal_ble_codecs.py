"""Byte-level tests for the BLE fitness decoders.

Every payload here is hand-assembled from the Bluetooth SIG characteristic
definitions, which is the whole point of keeping the decoders free of a
Bluetooth stack: the bit packing can be verified in CI with no radio present.
"""

import math
import struct

import pytest

from raganything.biosignal.ble import codecs, uuids
from raganything.biosignal.ble.codecs import (
    DecodeError,
    RevolutionTracker,
    decode,
    sfloat,
)
from raganything.biosignal.schema import Evidence, Modality

T0 = 1_700_000_000.0


def u16(v):
    return struct.pack("<H", v)


def s16(v):
    return struct.pack("<h", v)


def u24(v):
    return v.to_bytes(3, "little")


class TestSFloat:
    def test_positive_exponent_and_mantissa(self):
        # exponent 0, mantissa 98
        assert sfloat(u16(0x0062)) == pytest.approx(98.0)

    def test_negative_exponent(self):
        # exponent -5 (0xB), mantissa 100 -> 0.001
        assert sfloat(u16(0xB064)) == pytest.approx(0.001)

    def test_negative_mantissa(self):
        # exponent 0, mantissa -1 (0xFFF)
        assert sfloat(u16(0x0FFF)) == pytest.approx(-1.0)

    def test_reserved_values(self):
        assert math.isnan(sfloat(u16(0x07FF)))  # NaN
        assert math.isnan(sfloat(u16(0x0800)))  # NRes
        assert math.isnan(sfloat(u16(0x0801)))  # reserved
        assert sfloat(u16(0x07FE)) == math.inf
        assert sfloat(u16(0x0802)) == -math.inf

    def test_wrong_length_rejected(self):
        with pytest.raises(DecodeError):
            sfloat(b"\x01")


class TestHeartRateMeasurement:
    def test_uint8_rate_with_rr_intervals(self):
        # flags: RR present (0x10); hr 60; RR 1024/1024 s and 512/1024 s
        payload = b"\x10\x3c" + u16(1024) + u16(512)
        d = decode(uuids.CHR_HEART_RATE_MEASUREMENT, payload, T0)

        assert d.first(Modality.HEART_RATE) == 60.0
        assert d.raw["rr_intervals_ms"] == pytest.approx([1000.0, 500.0])

    def test_rr_intervals_are_back_dated_to_their_beats(self):
        payload = b"\x10\x3c" + u16(1024) + u16(512)
        d = decode(uuids.CHR_HEART_RATE_MEASUREMENT, payload, T0)

        rr = [s for m, s in d.readings if m == Modality.RR_INTERVAL]
        assert len(rr) == 2
        # The last interval ended at packet arrival; the one before it ended
        # 500 ms earlier. Stamping both at T0 would destroy the beat timeline.
        assert rr[1].t == pytest.approx(T0)
        assert rr[0].t == pytest.approx(T0 - 0.5)

    def test_uint16_rate_format(self):
        payload = b"\x01" + u16(200)
        d = decode(uuids.CHR_HEART_RATE_MEASUREMENT, payload, T0)
        assert d.first(Modality.HEART_RATE) == 200.0

    def test_energy_expended_field_does_not_shift_rr(self):
        # flags: energy (0x08) + RR (0x10)
        payload = b"\x18\x3c" + u16(500) + u16(1024)
        d = decode(uuids.CHR_HEART_RATE_MEASUREMENT, payload, T0)
        assert d.raw["energy_expended_kj"] == 500
        assert d.raw["rr_intervals_ms"] == pytest.approx([1000.0])

    def test_sensor_contact_lost_zeroes_confidence(self):
        # contact supported (0x04) but not detected (0x02 clear)
        payload = b"\x04\x3c"
        d = decode(uuids.CHR_HEART_RATE_MEASUREMENT, payload, T0)
        sample = d.readings[0][1]
        assert "no_contact" in sample.flags
        assert sample.confidence == 0.0

    def test_contact_detected_keeps_confidence(self):
        payload = b"\x06\x3c"  # supported and detected
        d = decode(uuids.CHR_HEART_RATE_MEASUREMENT, payload, T0)
        assert d.readings[0][1].flags == ()
        assert d.readings[0][1].confidence == 1.0

    def test_truncated_payload_raises(self):
        with pytest.raises(DecodeError):
            decode(uuids.CHR_HEART_RATE_MEASUREMENT, b"\x01\x2c", T0)


class TestIndoorBikeData:
    def test_speed_cadence_power(self):
        # bit0 clear -> speed present; bit2 cadence; bit6 power
        flags = 0x0044
        payload = u16(flags) + u16(3000) + u16(180) + s16(250)
        d = decode(uuids.CHR_INDOOR_BIKE_DATA, payload, T0)

        assert d.first(Modality.SPEED) == pytest.approx(30.0 / 3.6)
        assert d.first(Modality.CADENCE) == pytest.approx(90.0)
        assert d.first(Modality.POWER) == pytest.approx(250.0)

    def test_more_data_bit_is_inverted(self):
        # bit0 SET means instantaneous speed is ABSENT. Reading it as an
        # ordinary presence bit shifts every field that follows.
        flags = 0x0041
        payload = u16(flags) + s16(250)
        d = decode(uuids.CHR_INDOOR_BIKE_DATA, payload, T0)

        assert d.first(Modality.SPEED) is None
        assert d.first(Modality.POWER) == pytest.approx(250.0)

    def test_machine_relayed_heart_rate_is_not_a_measurement(self):
        flags = 0x0201  # no speed, heart rate present
        payload = u16(flags) + bytes([142])
        d = decode(uuids.CHR_INDOOR_BIKE_DATA, payload, T0)

        hr = [s for m, s in d.readings if m == Modality.HEART_RATE][0]
        assert hr.value == 142.0
        assert hr.evidence is Evidence.VENDOR_DERIVED
        assert "relayed_by_machine" in hr.flags

    def test_energy_block_consumes_five_bytes(self):
        flags = 0x0101 | 0x0200  # no speed, energy, heart rate
        payload = u16(flags) + u16(400) + u16(600) + bytes([10]) + bytes([150])
        d = decode(uuids.CHR_INDOOR_BIKE_DATA, payload, T0)

        assert d.raw["total_energy_kcal"] == 400
        assert d.raw["energy_per_hour_kcal"] == 600
        assert d.raw["energy_per_minute_kcal"] == 10
        assert d.raw["heart_rate_bpm"] == 150


class TestTreadmillAndRower:
    def test_treadmill_speed_and_incline(self):
        flags = 0x0008  # speed present (bit0 clear), inclination present
        payload = u16(flags) + u16(1200) + s16(55) + s16(30)
        d = decode(uuids.CHR_TREADMILL_DATA, payload, T0)

        assert d.first(Modality.SPEED) == pytest.approx(12.0 / 3.6)
        assert d.first(Modality.INCLINE) == pytest.approx(5.5)
        assert d.raw["ramp_angle_deg"] == pytest.approx(3.0)

    def test_treadmill_negative_incline(self):
        flags = 0x0009  # no speed, inclination present
        payload = u16(flags) + s16(-25) + s16(-15)
        d = decode(uuids.CHR_TREADMILL_DATA, payload, T0)
        assert d.first(Modality.INCLINE) == pytest.approx(-2.5)

    def test_rower_stroke_and_power(self):
        flags = 0x0020  # stroke data present (bit0 clear), power present
        payload = u16(flags) + bytes([44]) + u16(100) + s16(200)
        d = decode(uuids.CHR_ROWER_DATA, payload, T0)

        assert d.first(Modality.STROKE_RATE) == pytest.approx(22.0)
        assert d.raw["stroke_count"] == 100
        assert d.first(Modality.POWER) == pytest.approx(200.0)

    def test_rower_pace_normalised_to_seconds_per_km(self):
        flags = 0x0009  # no stroke data, instantaneous pace present
        payload = u16(flags) + u16(120)  # 120 s per 500 m
        d = decode(uuids.CHR_ROWER_DATA, payload, T0)
        assert d.first(Modality.PACE) == pytest.approx(240.0)


class TestCyclingPowerAndCadence:
    def test_power_with_pedal_balance_and_crank_data(self):
        flags = 0x0001 | 0x0020
        payload = u16(flags) + s16(240) + bytes([104]) + u16(1000) + u16(512)
        d = decode(uuids.CHR_CYCLING_POWER_MEASUREMENT, payload, T0)

        assert d.first(Modality.POWER) == pytest.approx(240.0)
        assert d.first(Modality.PEDAL_BALANCE) == pytest.approx(52.0)
        assert d.raw["cumulative_crank_revolutions"] == 1000
        assert d.raw["last_crank_event_time"] == pytest.approx(0.5)

    def test_extreme_angles_are_two_packed_twelve_bit_fields(self):
        flags = 0x0100
        # min angle 200, max angle 100 -> packed as (200 << 12) | 100
        payload = u16(flags) + s16(150) + u24((200 << 12) | 100)
        d = decode(uuids.CHR_CYCLING_POWER_MEASUREMENT, payload, T0)

        assert d.raw["max_angle_deg"] == 100
        assert d.raw["min_angle_deg"] == 200

    def test_negative_power_is_signed(self):
        payload = u16(0x0000) + s16(-5)
        d = decode(uuids.CHR_CYCLING_POWER_MEASUREMENT, payload, T0)
        assert d.first(Modality.POWER) == pytest.approx(-5.0)

    def test_csc_wheel_and_crank(self):
        payload = b"\x03" + struct.pack("<I", 5000) + u16(1024) + u16(700) + u16(512)
        d = decode(uuids.CHR_CSC_MEASUREMENT, payload, T0)

        assert d.raw["cumulative_wheel_revolutions"] == 5000
        assert d.raw["last_wheel_event_time"] == pytest.approx(1.0)
        assert d.raw["cumulative_crank_revolutions"] == 700
        assert d.raw["last_crank_event_time"] == pytest.approx(0.5)

    def test_rsc_measurement(self):
        payload = (
            b"\x03" + u16(768) + bytes([180]) + u16(120) + struct.pack("<I", 12345)
        )
        d = decode(uuids.CHR_RSC_MEASUREMENT, payload, T0)

        assert d.first(Modality.SPEED) == pytest.approx(3.0)
        assert d.first(Modality.CADENCE) == pytest.approx(180.0)
        assert d.first(Modality.STRIDE_LENGTH) == pytest.approx(1.2)
        assert d.first(Modality.DISTANCE) == pytest.approx(1234.5)


class TestBodyGlucoseOximetry:
    def test_body_composition_separates_impedance_from_estimate(self):
        flags = 0x0200 | 0x0400  # impedance and weight, SI units
        payload = u16(flags) + u16(155) + u16(5000) + u16(15000)
        d = decode(uuids.CHR_BODY_COMPOSITION_MEASUREMENT, payload, T0)

        impedance = [s for m, s in d.readings if m == Modality.IMPEDANCE][0]
        body_fat = [s for m, s in d.readings if m == Modality.BODY_FAT][0]

        assert impedance.value == pytest.approx(500.0)
        assert impedance.evidence is Evidence.MEASURED
        # The percentage is a proprietary regression over the impedance, and
        # must not inherit the impedance's evidence class.
        assert body_fat.value == pytest.approx(15.5)
        assert body_fat.evidence is Evidence.VENDOR_DERIVED
        assert d.first(Modality.WEIGHT) == pytest.approx(75.0)

    def test_body_composition_imperial_scaling(self):
        flags = 0x0001 | 0x0400  # imperial, weight present
        payload = u16(flags) + u16(155) + u16(15000)
        d = decode(uuids.CHR_BODY_COMPOSITION_MEASUREMENT, payload, T0)
        assert d.first(Modality.WEIGHT) == pytest.approx(150.0)  # lb

    def test_glucose_kg_per_litre_converted_to_mg_dl(self):
        flags = 0x02
        payload = (
            bytes([flags])
            + u16(7)
            + u16(2026)
            + bytes([9, 2, 6, 30, 0])
            + u16(0xB064)  # 0.001 kg/L
            + bytes([0x12])
        )
        d = decode(uuids.CHR_GLUCOSE_MEASUREMENT, payload, T0)

        assert d.first(Modality.GLUCOSE) == pytest.approx(100.0)
        assert d.raw["sequence_number"] == 7
        assert d.raw["base_time"]["year"] == 2026
        assert d.raw["glucose_type"] == 2
        assert d.raw["sample_location"] == 1

    def test_plx_continuous_measurement(self):
        payload = bytes([0x00]) + u16(0x0062) + u16(0x003C)
        d = decode(uuids.CHR_PLX_CONTINUOUS_MEASUREMENT, payload, T0)

        assert d.first(Modality.SPO2) == pytest.approx(98.0)
        assert d.first(Modality.HEART_RATE) == pytest.approx(60.0)

    def test_battery_level(self):
        d = decode(uuids.CHR_BATTERY_LEVEL, b"\x55", T0)
        assert d.first(Modality.BATTERY) == 85.0


class TestDispatch:
    def test_unknown_characteristic_names_itself(self):
        with pytest.raises(DecodeError, match="0x1234"):
            decode(0x1234, b"\x00", T0)

    def test_every_advertised_decoder_is_reachable(self):
        for char in uuids.DECODABLE:
            assert char in codecs._DECODERS


class TestRevolutionTracker:
    def test_first_packet_yields_nothing(self):
        assert RevolutionTracker().update(100, 1.0) is None

    def test_computes_rpm_from_successive_packets(self):
        tracker = RevolutionTracker()
        tracker.update(0, 0.0)
        assert tracker.update(10, 10.0) == pytest.approx(60.0)

    def test_event_time_rollover(self):
        tracker = RevolutionTracker(event_time_rollover=64.0)
        tracker.update(0, 60.0)
        # Event time wrapped from 60 s to 4 s: the real gap is 8 s, not -56 s.
        assert tracker.update(16, 4.0) == pytest.approx(120.0)

    def test_counter_rollover(self):
        tracker = RevolutionTracker(counter_bits=16)
        tracker.update(65530, 0.0)
        assert tracker.update(4, 1.0) == pytest.approx(600.0)

    def test_stalled_sensor_reports_nothing_rather_than_a_stale_value(self):
        tracker = RevolutionTracker()
        tracker.update(100, 5.0)
        assert tracker.update(100, 5.0) is None

    def test_no_revolutions_but_time_advanced_is_genuinely_zero(self):
        tracker = RevolutionTracker()
        tracker.update(100, 5.0)
        assert tracker.update(100, 10.0) == 0.0

    def test_reset_clears_history(self):
        tracker = RevolutionTracker()
        tracker.update(0, 0.0)
        tracker.reset()
        assert tracker.update(10, 10.0) is None
