"""Pure-Python decoders for standard Bluetooth SIG fitness characteristics.

These are deliberately free of any Bluetooth stack dependency: they take
``bytes`` and a timestamp and return canonical :class:`~..schema.Sample`
objects. That makes the hardest, most error-prone part of BLE fitness work --
the bit-level packing, the inverted flag bits, the mixed-endian medical float
formats, the 16-bit counters that silently wrap -- fully testable against
captured packets, with no radio in the loop.

Three things here are routinely got wrong in shipping apps and are handled
explicitly:

1. **RR intervals are back-dated.** A Heart Rate Measurement notification can
   carry up to nine beat intervals. Stamping them all with the packet arrival
   time destroys the beat timeline that HRV analysis depends on, so each
   interval is placed at the moment the beat actually occurred.
2. **FTMS "More Data" is inverted.** In Indoor Bike / Treadmill / Rower data,
   bit 0 set means Instantaneous Speed is *absent*. Reading it as a normal
   presence bit shifts every subsequent field.
3. **Cumulative counters wrap.** Wheel/crank revolution counters and their
   event-time companions roll over at 16 or 32 bits; :class:`RevolutionTracker`
   handles the wrap instead of emitting a negative cadence spike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..schema import Evidence, Modality, Sample
from . import uuids

__all__ = [
    "Decoded",
    "DecodeError",
    "RevolutionTracker",
    "decode",
    "decode_battery_level",
    "decode_body_composition_measurement",
    "decode_cycling_power_measurement",
    "decode_csc_measurement",
    "decode_glucose_measurement",
    "decode_heart_rate_measurement",
    "decode_indoor_bike_data",
    "decode_plx_continuous_measurement",
    "decode_rower_data",
    "decode_rsc_measurement",
    "decode_treadmill_data",
    "sfloat",
]


class DecodeError(ValueError):
    """Raised when a payload is too short or structurally impossible."""


@dataclass
class Decoded:
    """Result of decoding one notification.

    Attributes:
        characteristic: 16-bit assigned number of the source characteristic.
        t: Packet arrival timestamp (POSIX seconds).
        readings: ``(modality, sample)`` pairs ready to append to streams.
        raw: Every field the packet contained, including ones with no canonical
            modality (cumulative counters, status words). Nothing is discarded.
        flags: Packet-level notes such as ``"no_contact"``.
    """

    characteristic: int
    t: float
    readings: List[Tuple[Modality, Sample]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)
    flags: Tuple[str, ...] = ()

    def values(self, modality: Modality) -> List[float]:
        return [s.value for m, s in self.readings if m == modality]

    def first(self, modality: Modality) -> Optional[float]:
        vals = self.values(modality)
        return vals[0] if vals else None


# --------------------------------------------------------------------------
# primitive readers
# --------------------------------------------------------------------------


class _Cursor:
    """Sequential little-endian reader with bounds checking."""

    def __init__(self, data: bytes, what: str) -> None:
        self.data = bytes(data)
        self.i = 0
        self.what = what

    def _take(self, n: int) -> bytes:
        if self.i + n > len(self.data):
            raise DecodeError(
                f"{self.what}: payload truncated, needed {n} more byte(s) at "
                f"offset {self.i} of {len(self.data)}"
            )
        chunk = self.data[self.i : self.i + n]
        self.i += n
        return chunk

    def u8(self) -> int:
        return self._take(1)[0]

    def s8(self) -> int:
        return int.from_bytes(self._take(1), "little", signed=True)

    def u16(self) -> int:
        return int.from_bytes(self._take(2), "little")

    def s16(self) -> int:
        return int.from_bytes(self._take(2), "little", signed=True)

    def u24(self) -> int:
        return int.from_bytes(self._take(3), "little")

    def u32(self) -> int:
        return int.from_bytes(self._take(4), "little")

    def sfloat(self) -> float:
        return sfloat(self._take(2))

    def datetime7(self) -> Dict[str, int]:
        """The 7-byte SIG Date Time structure."""
        year = self.u16()
        return {
            "year": year,
            "month": self.u8(),
            "day": self.u8(),
            "hours": self.u8(),
            "minutes": self.u8(),
            "seconds": self.u8(),
        }

    @property
    def remaining(self) -> int:
        return len(self.data) - self.i


def sfloat(raw: bytes) -> float:
    """Decode an IEEE-11073 16-bit SFLOAT.

    Medical BLE characteristics (glucose, SpO2, temperature) use a 4-bit signed
    exponent plus a 12-bit signed mantissa, with five reserved mantissa values
    for NaN and the infinities. Treating these two bytes as a normal uint16 --
    a common shortcut -- yields values that are wrong by orders of magnitude.
    """
    if len(raw) != 2:
        raise DecodeError("sfloat requires exactly 2 bytes")
    word = int.from_bytes(raw, "little")
    mantissa = word & 0x0FFF
    exponent = (word >> 12) & 0x0F

    if mantissa == 0x07FF:
        return math.nan  # NaN
    if mantissa == 0x0800:
        return math.nan  # NRes -- not a valid result
    if mantissa == 0x07FE:
        return math.inf
    if mantissa == 0x0802:
        return -math.inf
    if mantissa == 0x0801:
        return math.nan  # reserved for future use

    if mantissa >= 0x0800:
        mantissa -= 0x1000
    if exponent >= 0x08:
        exponent -= 0x10
    return float(mantissa) * (10.0**exponent)


def _s(
    t: float, value: float, *, confidence: float = 1.0, flags: Tuple[str, ...] = ()
) -> Sample:
    return Sample(
        t=t,
        value=float(value),
        evidence=Evidence.MEASURED,
        confidence=confidence,
        flags=flags,
    )


# --------------------------------------------------------------------------
# Heart Rate Service
# --------------------------------------------------------------------------


def decode_heart_rate_measurement(data: bytes, t: float) -> Decoded:
    """Heart Rate Measurement, characteristic 0x2A37.

    RR intervals are transported at 1/1024 s resolution and are the single most
    valuable field on any consumer fitness device -- they are the only raw
    physiological timing signal most people own. They are also the field most
    apps throw away, keeping only the smoothed bpm.

    Each interval is timestamped at the beat it ended, reconstructed backwards
    from packet arrival, so a stream assembled from many notifications carries a
    coherent beat timeline rather than clumps at packet boundaries.
    """
    c = _Cursor(data, "Heart Rate Measurement")
    flags = c.u8()

    hr_16bit = bool(flags & 0x01)
    contact_supported = bool(flags & 0x04)
    contact_detected = bool(flags & 0x02)
    has_energy = bool(flags & 0x08)
    has_rr = bool(flags & 0x10)

    hr = c.u16() if hr_16bit else c.u8()

    packet_flags: List[str] = []
    confidence = 1.0
    if contact_supported and not contact_detected:
        # The strap is telling us it is not touching skin. Every app that
        # ignores this bit charts electrode noise as a heart rate.
        packet_flags.append("no_contact")
        confidence = 0.0

    raw: Dict[str, Any] = {"flags": flags, "heart_rate": hr}
    readings: List[Tuple[Modality, Sample]] = [
        (
            Modality.HEART_RATE,
            _s(t, hr, confidence=confidence, flags=tuple(packet_flags)),
        )
    ]

    if has_energy:
        raw["energy_expended_kj"] = c.u16()

    rr_ms: List[float] = []
    if has_rr:
        while c.remaining >= 2:
            rr_ms.append(c.u16() * 1000.0 / 1024.0)
    raw["rr_intervals_ms"] = rr_ms

    if rr_ms:
        # Place the last interval at packet arrival and walk backwards.
        offsets: List[float] = []
        running = 0.0
        for rr in reversed(rr_ms):
            offsets.append(running)
            running += rr / 1000.0
        offsets.reverse()
        for rr, back in zip(rr_ms, offsets):
            readings.append(
                (
                    Modality.RR_INTERVAL,
                    _s(t - back, rr, confidence=confidence, flags=tuple(packet_flags)),
                )
            )

    return Decoded(
        characteristic=uuids.CHR_HEART_RATE_MEASUREMENT,
        t=t,
        readings=readings,
        raw=raw,
        flags=tuple(packet_flags),
    )


# --------------------------------------------------------------------------
# Cycling Speed and Cadence / Running Speed and Cadence
# --------------------------------------------------------------------------


def decode_csc_measurement(data: bytes, t: float) -> Decoded:
    """CSC Measurement, characteristic 0x2A5B.

    Carries only cumulative counters; instantaneous speed and cadence must be
    differentiated from successive packets (see :class:`RevolutionTracker`).
    """
    c = _Cursor(data, "CSC Measurement")
    flags = c.u8()
    raw: Dict[str, Any] = {"flags": flags}

    if flags & 0x01:
        raw["cumulative_wheel_revolutions"] = c.u32()
        raw["last_wheel_event_time"] = c.u16() / 1024.0
    if flags & 0x02:
        raw["cumulative_crank_revolutions"] = c.u16()
        raw["last_crank_event_time"] = c.u16() / 1024.0

    return Decoded(characteristic=uuids.CHR_CSC_MEASUREMENT, t=t, raw=raw)


def decode_rsc_measurement(data: bytes, t: float) -> Decoded:
    """RSC Measurement, characteristic 0x2A53 (foot pod / running sensor)."""
    c = _Cursor(data, "RSC Measurement")
    flags = c.u8()

    readings: List[Tuple[Modality, Sample]] = []
    speed = c.u16() / 256.0  # m/s
    cadence = c.u8()  # steps per minute
    readings.append((Modality.SPEED, _s(t, speed)))
    readings.append((Modality.CADENCE, _s(t, cadence)))

    raw: Dict[str, Any] = {
        "flags": flags,
        "instantaneous_speed_mps": speed,
        "instantaneous_cadence_spm": cadence,
        "running": bool(flags & 0x04),
    }

    if flags & 0x01:
        stride = c.u16() / 100.0
        raw["stride_length_m"] = stride
        readings.append((Modality.STRIDE_LENGTH, _s(t, stride)))
    if flags & 0x02:
        distance = c.u32() / 10.0
        raw["total_distance_m"] = distance
        readings.append((Modality.DISTANCE, _s(t, distance)))

    return Decoded(
        characteristic=uuids.CHR_RSC_MEASUREMENT, t=t, readings=readings, raw=raw
    )


# --------------------------------------------------------------------------
# Cycling Power
# --------------------------------------------------------------------------


def decode_cycling_power_measurement(data: bytes, t: float) -> Decoded:
    """Cycling Power Measurement, characteristic 0x2A63.

    The optional blocks here (pedal balance, extreme force and torque
    magnitudes, dead-spot angles) describe *how* force is applied through the
    stroke. Essentially no consumer app surfaces them, and they are the raw
    material for asymmetry and technique analysis.
    """
    c = _Cursor(data, "Cycling Power Measurement")
    flags = c.u16()
    power = c.s16()

    readings: List[Tuple[Modality, Sample]] = [(Modality.POWER, _s(t, power))]
    raw: Dict[str, Any] = {"flags": flags, "instantaneous_power_w": power}

    if flags & 0x0001:
        balance = c.u8() / 2.0
        raw["pedal_power_balance_pct"] = balance
        raw["pedal_power_balance_reference_left"] = bool(flags & 0x0002)
        readings.append((Modality.PEDAL_BALANCE, _s(t, balance)))
    if flags & 0x0004:
        raw["accumulated_torque_nm"] = c.u16() / 32.0
        raw["accumulated_torque_source_crank"] = bool(flags & 0x0008)
    if flags & 0x0010:
        raw["cumulative_wheel_revolutions"] = c.u32()
        raw["last_wheel_event_time"] = c.u16() / 2048.0
    if flags & 0x0020:
        raw["cumulative_crank_revolutions"] = c.u16()
        raw["last_crank_event_time"] = c.u16() / 1024.0
    if flags & 0x0040:
        raw["max_force_magnitude_n"] = c.s16()
        raw["min_force_magnitude_n"] = c.s16()
    if flags & 0x0080:
        raw["max_torque_magnitude_nm"] = c.s16() / 32.0
        raw["min_torque_magnitude_nm"] = c.s16() / 32.0
    if flags & 0x0100:
        packed = c.u24()
        raw["max_angle_deg"] = packed & 0x0FFF
        raw["min_angle_deg"] = (packed >> 12) & 0x0FFF
    if flags & 0x0200:
        raw["top_dead_spot_angle_deg"] = c.u16()
    if flags & 0x0400:
        raw["bottom_dead_spot_angle_deg"] = c.u16()
    if flags & 0x0800:
        raw["accumulated_energy_kj"] = c.u16()
    raw["offset_compensation_indicator"] = bool(flags & 0x1000)

    return Decoded(
        characteristic=uuids.CHR_CYCLING_POWER_MEASUREMENT,
        t=t,
        readings=readings,
        raw=raw,
    )


# --------------------------------------------------------------------------
# Fitness Machine Service
# --------------------------------------------------------------------------


def _ftms_energy(c: _Cursor, raw: Dict[str, Any]) -> None:
    raw["total_energy_kcal"] = c.u16()
    raw["energy_per_hour_kcal"] = c.u16()
    raw["energy_per_minute_kcal"] = c.u8()


def decode_indoor_bike_data(data: bytes, t: float) -> Decoded:
    """Indoor Bike Data, characteristic 0x2AD2 (FTMS)."""
    c = _Cursor(data, "Indoor Bike Data")
    flags = c.u16()
    readings: List[Tuple[Modality, Sample]] = []
    raw: Dict[str, Any] = {"flags": flags}

    # Bit 0 is "More Data": when SET, instantaneous speed is ABSENT.
    if not (flags & 0x0001):
        speed = c.u16() * 0.01 / 3.6  # 0.01 km/h -> m/s
        raw["instantaneous_speed_mps"] = speed
        readings.append((Modality.SPEED, _s(t, speed)))
    if flags & 0x0002:
        raw["average_speed_mps"] = c.u16() * 0.01 / 3.6
    if flags & 0x0004:
        cadence = c.u16() * 0.5
        raw["instantaneous_cadence_rpm"] = cadence
        readings.append((Modality.CADENCE, _s(t, cadence)))
    if flags & 0x0008:
        raw["average_cadence_rpm"] = c.u16() * 0.5
    if flags & 0x0010:
        distance = float(c.u24())
        raw["total_distance_m"] = distance
        readings.append((Modality.DISTANCE, _s(t, distance)))
    if flags & 0x0020:
        resistance = float(c.s16())
        raw["resistance_level"] = resistance
        readings.append((Modality.RESISTANCE, _s(t, resistance)))
    if flags & 0x0040:
        power = float(c.s16())
        raw["instantaneous_power_w"] = power
        readings.append((Modality.POWER, _s(t, power)))
    if flags & 0x0080:
        raw["average_power_w"] = float(c.s16())
    if flags & 0x0100:
        _ftms_energy(c, raw)
        readings.append((Modality.ENERGY, _s(t, raw["total_energy_kcal"])))
    if flags & 0x0200:
        hr = c.u8()
        raw["heart_rate_bpm"] = hr
        # A machine-relayed heart rate has been through the machine's own
        # smoothing; it is not the strap's measurement even when it came from
        # one, so it is marked as vendor-derived rather than measured.
        readings.append(
            (
                Modality.HEART_RATE,
                Sample(
                    t=t,
                    value=float(hr),
                    evidence=Evidence.VENDOR_DERIVED,
                    confidence=0.7,
                    flags=("relayed_by_machine",),
                ),
            )
        )
    if flags & 0x0400:
        raw["metabolic_equivalent"] = c.u8() * 0.1
    if flags & 0x0800:
        raw["elapsed_time_s"] = c.u16()
    if flags & 0x1000:
        raw["remaining_time_s"] = c.u16()

    return Decoded(
        characteristic=uuids.CHR_INDOOR_BIKE_DATA, t=t, readings=readings, raw=raw
    )


def decode_treadmill_data(data: bytes, t: float) -> Decoded:
    """Treadmill Data, characteristic 0x2ACD (FTMS)."""
    c = _Cursor(data, "Treadmill Data")
    flags = c.u16()
    readings: List[Tuple[Modality, Sample]] = []
    raw: Dict[str, Any] = {"flags": flags}

    if not (flags & 0x0001):  # inverted "More Data" bit
        speed = c.u16() * 0.01 / 3.6
        raw["instantaneous_speed_mps"] = speed
        readings.append((Modality.SPEED, _s(t, speed)))
    if flags & 0x0002:
        raw["average_speed_mps"] = c.u16() * 0.01 / 3.6
    if flags & 0x0004:
        distance = float(c.u24())
        raw["total_distance_m"] = distance
        readings.append((Modality.DISTANCE, _s(t, distance)))
    if flags & 0x0008:
        incline = c.s16() * 0.1
        raw["inclination_pct"] = incline
        raw["ramp_angle_deg"] = c.s16() * 0.1
        readings.append((Modality.INCLINE, _s(t, incline)))
    if flags & 0x0010:
        raw["positive_elevation_gain_m"] = c.u16() * 0.1
        raw["negative_elevation_gain_m"] = c.u16() * 0.1
        readings.append((Modality.ELEVATION, _s(t, raw["positive_elevation_gain_m"])))
    if flags & 0x0020:
        raw["instantaneous_pace_km_per_min"] = c.u8() * 0.1
    if flags & 0x0040:
        raw["average_pace_km_per_min"] = c.u8() * 0.1
    if flags & 0x0080:
        _ftms_energy(c, raw)
        readings.append((Modality.ENERGY, _s(t, raw["total_energy_kcal"])))
    if flags & 0x0100:
        hr = c.u8()
        raw["heart_rate_bpm"] = hr
        readings.append(
            (
                Modality.HEART_RATE,
                Sample(
                    t=t,
                    value=float(hr),
                    evidence=Evidence.VENDOR_DERIVED,
                    confidence=0.7,
                    flags=("relayed_by_machine",),
                ),
            )
        )
    if flags & 0x0200:
        raw["metabolic_equivalent"] = c.u8() * 0.1
    if flags & 0x0400:
        raw["elapsed_time_s"] = c.u16()
    if flags & 0x0800:
        raw["remaining_time_s"] = c.u16()
    if flags & 0x1000:
        force = float(c.s16())
        raw["force_on_belt_n"] = force
        raw["power_output_w"] = float(c.s16())
        readings.append((Modality.FORCE, _s(t, force)))
        readings.append((Modality.POWER, _s(t, raw["power_output_w"])))

    return Decoded(
        characteristic=uuids.CHR_TREADMILL_DATA, t=t, readings=readings, raw=raw
    )


def decode_rower_data(data: bytes, t: float) -> Decoded:
    """Rower Data, characteristic 0x2AD1 (FTMS)."""
    c = _Cursor(data, "Rower Data")
    flags = c.u16()
    readings: List[Tuple[Modality, Sample]] = []
    raw: Dict[str, Any] = {"flags": flags}

    if not (flags & 0x0001):  # inverted "More Data" bit
        stroke_rate = c.u8() * 0.5
        raw["stroke_rate_spm"] = stroke_rate
        raw["stroke_count"] = c.u16()
        readings.append((Modality.STROKE_RATE, _s(t, stroke_rate)))
    if flags & 0x0002:
        raw["average_stroke_rate_spm"] = c.u8() * 0.5
    if flags & 0x0004:
        distance = float(c.u24())
        raw["total_distance_m"] = distance
        readings.append((Modality.DISTANCE, _s(t, distance)))
    if flags & 0x0008:
        pace = float(c.u16())  # seconds per 500 m
        raw["instantaneous_pace_s_per_500m"] = pace
        readings.append((Modality.PACE, _s(t, pace * 2.0)))  # canonical s/km
    if flags & 0x0010:
        raw["average_pace_s_per_500m"] = float(c.u16())
    if flags & 0x0020:
        power = float(c.s16())
        raw["instantaneous_power_w"] = power
        readings.append((Modality.POWER, _s(t, power)))
    if flags & 0x0040:
        raw["average_power_w"] = float(c.s16())
    if flags & 0x0080:
        resistance = float(c.s16())
        raw["resistance_level"] = resistance
        readings.append((Modality.RESISTANCE, _s(t, resistance)))
    if flags & 0x0100:
        _ftms_energy(c, raw)
        readings.append((Modality.ENERGY, _s(t, raw["total_energy_kcal"])))
    if flags & 0x0200:
        hr = c.u8()
        raw["heart_rate_bpm"] = hr
        readings.append(
            (
                Modality.HEART_RATE,
                Sample(
                    t=t,
                    value=float(hr),
                    evidence=Evidence.VENDOR_DERIVED,
                    confidence=0.7,
                    flags=("relayed_by_machine",),
                ),
            )
        )
    if flags & 0x0400:
        raw["metabolic_equivalent"] = c.u8() * 0.1
    if flags & 0x0800:
        raw["elapsed_time_s"] = c.u16()
    if flags & 0x1000:
        raw["remaining_time_s"] = c.u16()

    return Decoded(characteristic=uuids.CHR_ROWER_DATA, t=t, readings=readings, raw=raw)


# --------------------------------------------------------------------------
# Body Composition / Glucose / Pulse Oximetry / Battery
# --------------------------------------------------------------------------


def decode_body_composition_measurement(data: bytes, t: float) -> Decoded:
    """Body Composition Measurement, characteristic 0x2A9C.

    Smart scales expose raw bioimpedance here alongside the vendor's body-fat
    estimate. The impedance is the measurement; the percentage is a proprietary
    regression over it, and the two are labelled accordingly.
    """
    c = _Cursor(data, "Body Composition Measurement")
    flags = c.u16()
    imperial = bool(flags & 0x0001)
    mass_scale = 0.01 if imperial else 0.005
    length_scale = 0.1 if imperial else 0.001

    readings: List[Tuple[Modality, Sample]] = []
    raw: Dict[str, Any] = {"flags": flags, "imperial_units": imperial}

    body_fat = c.u16() * 0.1
    raw["body_fat_percentage"] = body_fat
    readings.append(
        (
            Modality.BODY_FAT,
            Sample(
                t=t,
                value=body_fat,
                evidence=Evidence.VENDOR_DERIVED,
                confidence=0.5,
                flags=("proprietary_impedance_model",),
            ),
        )
    )

    if flags & 0x0002:
        raw["timestamp"] = c.datetime7()
    if flags & 0x0004:
        raw["user_id"] = c.u8()
    if flags & 0x0008:
        raw["basal_metabolism_kj"] = c.u16()
    if flags & 0x0010:
        raw["muscle_percentage"] = c.u16() * 0.1
    if flags & 0x0020:
        muscle = c.u16() * mass_scale
        raw["muscle_mass"] = muscle
        readings.append((Modality.MUSCLE_MASS, _s(t, muscle)))
    if flags & 0x0040:
        raw["fat_free_mass"] = c.u16() * mass_scale
    if flags & 0x0080:
        raw["soft_lean_mass"] = c.u16() * mass_scale
    if flags & 0x0100:
        water = c.u16() * mass_scale
        raw["body_water_mass"] = water
        readings.append((Modality.BODY_WATER, _s(t, water)))
    if flags & 0x0200:
        impedance = c.u16() * 0.1
        raw["impedance_ohm"] = impedance
        readings.append((Modality.IMPEDANCE, _s(t, impedance)))
    if flags & 0x0400:
        weight = c.u16() * mass_scale
        raw["weight"] = weight
        readings.append((Modality.WEIGHT, _s(t, weight)))
    if flags & 0x0800:
        raw["height"] = c.u16() * length_scale
    raw["multiple_packet_measurement"] = bool(flags & 0x1000)

    return Decoded(
        characteristic=uuids.CHR_BODY_COMPOSITION_MEASUREMENT,
        t=t,
        readings=readings,
        raw=raw,
    )


def decode_glucose_measurement(data: bytes, t: float) -> Decoded:
    """Glucose Measurement, characteristic 0x2A18."""
    c = _Cursor(data, "Glucose Measurement")
    flags = c.u8()
    raw: Dict[str, Any] = {"flags": flags}
    readings: List[Tuple[Modality, Sample]] = []

    raw["sequence_number"] = c.u16()
    raw["base_time"] = c.datetime7()
    if flags & 0x01:
        raw["time_offset_min"] = c.s16()
    if flags & 0x02:
        concentration = c.sfloat()
        molar = bool(flags & 0x04)
        raw["concentration_raw"] = concentration
        raw["concentration_units"] = "mol/L" if molar else "kg/L"
        type_location = c.u8()
        raw["glucose_type"] = type_location & 0x0F
        raw["sample_location"] = (type_location >> 4) & 0x0F
        if not molar and math.isfinite(concentration):
            # kg/L -> mg/dL
            mg_dl = concentration * 100000.0
            raw["concentration_mg_dl"] = mg_dl
            readings.append((Modality.GLUCOSE, _s(t, mg_dl)))
        elif math.isfinite(concentration):
            # mol/L -> mg/dL, using glucose molar mass 180.156 g/mol
            mg_dl = concentration * 180.156 * 100.0
            raw["concentration_mg_dl"] = mg_dl
            readings.append((Modality.GLUCOSE, _s(t, mg_dl)))
    if flags & 0x08:
        raw["sensor_status_annunciation"] = c.u16()
    raw["context_information_follows"] = bool(flags & 0x10)

    return Decoded(
        characteristic=uuids.CHR_GLUCOSE_MEASUREMENT, t=t, readings=readings, raw=raw
    )


def decode_plx_continuous_measurement(data: bytes, t: float) -> Decoded:
    """PLX Continuous Measurement, characteristic 0x2A5F (pulse oximeter)."""
    c = _Cursor(data, "PLX Continuous Measurement")
    flags = c.u8()
    raw: Dict[str, Any] = {"flags": flags}
    readings: List[Tuple[Modality, Sample]] = []

    spo2 = c.sfloat()
    pulse = c.sfloat()
    raw["spo2_normal"] = spo2
    raw["pulse_rate_normal"] = pulse
    if math.isfinite(spo2):
        readings.append((Modality.SPO2, _s(t, spo2)))
    if math.isfinite(pulse):
        readings.append((Modality.HEART_RATE, _s(t, pulse)))

    if flags & 0x01:
        raw["spo2_fast"] = c.sfloat()
        raw["pulse_rate_fast"] = c.sfloat()
    if flags & 0x02:
        raw["spo2_slow"] = c.sfloat()
        raw["pulse_rate_slow"] = c.sfloat()
    if flags & 0x04:
        raw["measurement_status"] = c.u16()
    if flags & 0x08:
        raw["device_and_sensor_status"] = c.u24()
    if flags & 0x10:
        raw["pulse_amplitude_index"] = c.sfloat()

    return Decoded(
        characteristic=uuids.CHR_PLX_CONTINUOUS_MEASUREMENT,
        t=t,
        readings=readings,
        raw=raw,
    )


def decode_battery_level(data: bytes, t: float) -> Decoded:
    """Battery Level, characteristic 0x2A19.

    Trivial to decode and near-universally ignored, yet a strap at 4% is the
    single best predictor that the next hour of data will be garbage.
    """
    c = _Cursor(data, "Battery Level")
    level = c.u8()
    return Decoded(
        characteristic=uuids.CHR_BATTERY_LEVEL,
        t=t,
        readings=[(Modality.BATTERY, _s(t, level))],
        raw={"battery_level_pct": level},
    )


_DECODERS = {
    uuids.CHR_HEART_RATE_MEASUREMENT: decode_heart_rate_measurement,
    uuids.CHR_CSC_MEASUREMENT: decode_csc_measurement,
    uuids.CHR_RSC_MEASUREMENT: decode_rsc_measurement,
    uuids.CHR_CYCLING_POWER_MEASUREMENT: decode_cycling_power_measurement,
    uuids.CHR_INDOOR_BIKE_DATA: decode_indoor_bike_data,
    uuids.CHR_TREADMILL_DATA: decode_treadmill_data,
    uuids.CHR_ROWER_DATA: decode_rower_data,
    uuids.CHR_BODY_COMPOSITION_MEASUREMENT: decode_body_composition_measurement,
    uuids.CHR_GLUCOSE_MEASUREMENT: decode_glucose_measurement,
    uuids.CHR_PLX_CONTINUOUS_MEASUREMENT: decode_plx_continuous_measurement,
    uuids.CHR_BATTERY_LEVEL: decode_battery_level,
}


def decode(char_uuid16: int, data: bytes, t: float) -> Decoded:
    """Decode any supported characteristic by its 16-bit assigned number."""
    fn = _DECODERS.get(char_uuid16)
    if fn is None:
        raise DecodeError(
            f"no decoder for characteristic 0x{char_uuid16:04X} "
            f"({uuids.name_for(char_uuid16)})"
        )
    return fn(data, t)


# --------------------------------------------------------------------------
# cumulative counter differentiation
# --------------------------------------------------------------------------


class RevolutionTracker:
    """Turn cumulative revolution counters into instantaneous rates.

    CSC and Cycling Power report revolutions since power-on plus the timestamp
    of the last revolution event, both in wrapping unsigned fields. The device
    clock is the right time base -- far better than packet arrival, which is
    jittered by the radio -- but only if the wrap is handled.

    Args:
        event_time_rollover: Period of the event-time field, in seconds. CSC
            and crank fields wrap at 64 s (1/1024 s over 16 bits); the Cycling
            Power wheel field wraps at 32 s (1/2048 s over 16 bits).
        counter_bits: Width of the revolution counter (16 or 32).
    """

    def __init__(
        self, event_time_rollover: float = 64.0, counter_bits: int = 32
    ) -> None:
        self.event_time_rollover = event_time_rollover
        self.counter_modulo = 1 << counter_bits
        self._last_revs: Optional[int] = None
        self._last_event: Optional[float] = None

    def update(self, revolutions: int, event_time_s: float) -> Optional[float]:
        """Feed one packet; return revolutions per minute, or ``None``.

        ``None`` means "cannot be computed yet or would be a lie": the first
        packet, a stalled sensor repeating its last event, or a gap longer than
        the event-time field can unambiguously represent.
        """
        if self._last_revs is None or self._last_event is None:
            self._last_revs = revolutions
            self._last_event = event_time_s
            return None

        d_revs = (revolutions - self._last_revs) % self.counter_modulo
        d_time = event_time_s - self._last_event
        if d_time < 0:
            d_time += self.event_time_rollover

        self._last_revs = revolutions
        self._last_event = event_time_s

        if d_time <= 0:
            # No new revolution event since last packet: the sensor is idle or
            # simply re-notified. Reporting 0 rpm here would be an invention;
            # reporting the previous value would be a stale lie.
            return None
        if d_revs == 0:
            return 0.0
        return (d_revs / d_time) * 60.0

    def reset(self) -> None:
        self._last_revs = None
        self._last_event = None
