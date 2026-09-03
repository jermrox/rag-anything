"""Multi-field GATT health measurements: body composition, weight, blood
pressure, glucose, and running and cycling cadence.

These are the characteristics the Five Factor model needs and ``gatt.py`` did
not decode. The gap was found mechanically rather than guessed: the harvested
corpus carries the Bluetooth SIG assigned-numbers table, and cross-referencing
it against our own registry showed exactly which health characteristics a
device could advertise that we would have to ignore.

Each maps onto a factor:

============================  ========  ==============================
Characteristic                Factor    What it answers
============================  ========  ==============================
0x2A9C Body Composition       Nutrition Am I gaining lean mass or fat?
0x2A9D Weight Measurement     Nutrition Is body mass trending?
0x2A18 Glucose Measurement    Nutrition How am I metabolising fuel?
0x2A35 Blood Pressure         Medical   Context for every other factor
0x2A53 RSC Measurement        Movement  Running cadence and distance
0x2A5B CSC Measurement        Movement  Cycling cadence and wheel speed
============================  ========  ==============================

Blood pressure is deliberately *not* a factor of its own. It changes how the
other factors should be read -- a hypertensive reading reframes a training
recommendation -- which makes it context underneath the model rather than a
sixth dimension inside it.

Two primitives are shared and defined once here: IEEE 11073 SFLOAT, which the
medical characteristics use instead of plain integers, and the 7-byte Date
Time. Both have exceptional values that mean "no measurement", and both are
where a naive decoder silently invents data.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

from .gatt import DecodeError

# --- UUIDs, verified against the SIG assigned-numbers table in the corpus --

SERVICE_GLUCOSE = "0x1808"
SERVICE_BLOOD_PRESSURE = "0x1810"
SERVICE_RUNNING_SPEED_CADENCE = "0x1814"
SERVICE_CYCLING_SPEED_CADENCE = "0x1816"
SERVICE_BODY_COMPOSITION = "0x181B"
SERVICE_WEIGHT_SCALE = "0x181D"
SERVICE_CONTINUOUS_GLUCOSE = "0x181F"

CHAR_GLUCOSE_MEASUREMENT = "0x2A18"
CHAR_BLOOD_PRESSURE_MEASUREMENT = "0x2A35"
CHAR_RSC_MEASUREMENT = "0x2A53"
CHAR_CSC_MEASUREMENT = "0x2A5B"
CHAR_BODY_COMPOSITION_MEASUREMENT = "0x2A9C"
CHAR_WEIGHT_MEASUREMENT = "0x2A9D"

#: Wheel and crank event times, like RR intervals, are in 1/1024 second.
EVENT_TIME_UNITS_PER_SECOND = 1024.0

#: Instantaneous speed in RSC is in units of 1/256 m/s.
RSC_SPEED_UNITS_PER_MS = 256.0


# --- IEEE 11073 SFLOAT -----------------------------------------------------

#: SFLOAT mantissa values that are not measurements. Returning a number for
#: any of these is how "the cuff failed" becomes "your systolic was 2047".
_SFLOAT_NAN = 0x007FF
_SFLOAT_NRES = 0x00800
_SFLOAT_POSITIVE_INFINITY = 0x007FE
_SFLOAT_NEGATIVE_INFINITY = 0x00802

#: Mantissa is 12-bit two's complement, exponent 4-bit two's complement.
_MANTISSA_SIGN_BIT = 0x0800
_MANTISSA_RANGE = 0x1000
_EXPONENT_SIGN_BIT = 0x0008
_EXPONENT_RANGE = 0x0010


def decode_sfloat(raw: int) -> float | None:
    """Decode an IEEE 11073 16-bit SFLOAT.

    Returns None for NaN, NRes and the infinities. That is not a convenience:
    these codes mean "no valid measurement", and a decoder that maps them onto
    numbers reports a failed blood-pressure reading as a real one.
    """
    mantissa = raw & 0x0FFF
    exponent = (raw >> 12) & 0x000F

    if mantissa in (
        _SFLOAT_NAN,
        _SFLOAT_NRES,
        _SFLOAT_POSITIVE_INFINITY,
        _SFLOAT_NEGATIVE_INFINITY,
    ):
        return None

    if mantissa >= _MANTISSA_SIGN_BIT:
        mantissa -= _MANTISSA_RANGE
    if exponent >= _EXPONENT_SIGN_BIT:
        exponent -= _EXPONENT_RANGE
    return float(mantissa) * (10.0**exponent)


def _read_sfloat(data: bytes, offset: int, field: str) -> Tuple[float | None, int]:
    if offset + 2 > len(data):
        raise DecodeError(f"payload truncated before {field}")
    (raw,) = struct.unpack_from("<H", data, offset)
    return decode_sfloat(raw), offset + 2


# --- Date Time (7 bytes) ---------------------------------------------------


def decode_datetime(data: bytes, offset: int = 0) -> Tuple[datetime | None, int]:
    """Decode the 7-byte GATT Date Time, returning it and the next offset.

    Year 0 means "not known", and a zero month or day means the same for that
    component. Those cases return None rather than a fabricated date: a
    measurement whose time is unknown must not be silently stamped with one.
    """
    if offset + 7 > len(data):
        raise DecodeError("payload truncated before Date Time")
    year, month, day, hours, minutes, seconds = struct.unpack_from(
        "<HBBBBB", data, offset
    )
    offset += 7
    if year == 0 or month == 0 or day == 0:
        return None, offset
    try:
        return datetime(year, month, day, hours, minutes, seconds), offset
    except ValueError as exc:
        raise DecodeError(f"invalid Date Time in payload: {exc}") from exc


# --- Body Composition (0x2A9C) ---------------------------------------------

#: Mass and height resolutions differ by measurement system, and the flags
#: byte is the only thing that says which is in use. Reading an Imperial
#: payload with SI scaling is a silent 2.2x error in every mass field.
_MASS_KG_RESOLUTION = 0.005
_MASS_LB_RESOLUTION = 0.01
_HEIGHT_M_RESOLUTION = 0.001
_HEIGHT_IN_RESOLUTION = 0.1

_BC_FLAG_IMPERIAL = 1 << 0
_BC_FLAG_TIMESTAMP = 1 << 1
_BC_FLAG_USER_ID = 1 << 2
_BC_FLAG_BASAL_METABOLISM = 1 << 3
_BC_FLAG_MUSCLE_PERCENTAGE = 1 << 4
_BC_FLAG_MUSCLE_MASS = 1 << 5
_BC_FLAG_FAT_FREE_MASS = 1 << 6
_BC_FLAG_SOFT_LEAN_MASS = 1 << 7
_BC_FLAG_BODY_WATER_MASS = 1 << 8
_BC_FLAG_IMPEDANCE = 1 << 9
_BC_FLAG_WEIGHT = 1 << 10
_BC_FLAG_HEIGHT = 1 << 11

#: Body fat percentage and muscle percentage are both in units of 0.1%.
_PERCENT_RESOLUTION = 0.1
#: Impedance is in units of 0.1 ohm.
_IMPEDANCE_RESOLUTION = 0.1
#: An all-ones uint16 in a percentage or mass field means "not measured".
_UINT16_UNKNOWN = 0xFFFF


@dataclass(frozen=True, slots=True)
class BodyComposition:
    """A decoded 0x2A9C payload.

    Every optional field is None when the device did not send it. None means
    "this scale does not measure that", which is a different claim from zero
    and must stay distinguishable -- the same rule the signal model applies to
    a wearable that cannot deliver a signal at all.
    """

    body_fat_percent: float | None
    imperial: bool
    timestamp: datetime | None = None
    user_id: int | None = None
    basal_metabolism_kj: int | None = None
    muscle_percent: float | None = None
    muscle_mass: float | None = None
    fat_free_mass: float | None = None
    soft_lean_mass: float | None = None
    body_water_mass: float | None = None
    impedance_ohm: float | None = None
    weight: float | None = None
    height: float | None = None

    @property
    def mass_unit(self) -> str:
        return "lb" if self.imperial else "kg"

    @property
    def height_unit(self) -> str:
        return "in" if self.imperial else "m"

    @property
    def lean_mass(self) -> float | None:
        """Fat-free mass, preferring the measured field over the derived one.

        A scale that reports fat-free mass directly is stating a measurement.
        Computing it from weight and fat percentage instead is a derivation,
        and the two must not be conflated when the number reaches the evidence
        model.
        """
        if self.fat_free_mass is not None:
            return self.fat_free_mass
        if self.weight is not None and self.body_fat_percent is not None:
            return self.weight * (1.0 - self.body_fat_percent / 100.0)
        return None


def _read_uint16(data: bytes, offset: int, field: str) -> Tuple[int | None, int]:
    if offset + 2 > len(data):
        raise DecodeError(f"payload truncated before {field}")
    (value,) = struct.unpack_from("<H", data, offset)
    return (None if value == _UINT16_UNKNOWN else value), offset + 2


def _scaled(value: int | None, resolution: float) -> float | None:
    return None if value is None else value * resolution


def decode_body_composition(data: bytes) -> BodyComposition:
    """Decode Body Composition Measurement (0x2A9C).

    Field order is fixed by the specification and every field is optional, so
    the payload can only be parsed by walking the flags in order. Any
    reordering here silently misassigns every subsequent field.
    """
    if len(data) < 4:
        raise DecodeError(f"0x2A9C payload too short: {len(data)} bytes, need >= 4")
    (flags,) = struct.unpack_from("<H", data, 0)
    imperial = bool(flags & _BC_FLAG_IMPERIAL)
    mass_res = _MASS_LB_RESOLUTION if imperial else _MASS_KG_RESOLUTION
    height_res = _HEIGHT_IN_RESOLUTION if imperial else _HEIGHT_M_RESOLUTION

    offset = 2
    fat_raw, offset = _read_uint16(data, offset, "Body Fat Percentage")
    body_fat_percent = _scaled(fat_raw, _PERCENT_RESOLUTION)

    timestamp = None
    if flags & _BC_FLAG_TIMESTAMP:
        timestamp, offset = decode_datetime(data, offset)

    user_id = None
    if flags & _BC_FLAG_USER_ID:
        if offset + 1 > len(data):
            raise DecodeError("payload truncated before User ID")
        user_id = data[offset]
        offset += 1

    basal_metabolism_kj = None
    if flags & _BC_FLAG_BASAL_METABOLISM:
        basal_metabolism_kj, offset = _read_uint16(data, offset, "Basal Metabolism")

    muscle_percent = None
    if flags & _BC_FLAG_MUSCLE_PERCENTAGE:
        raw, offset = _read_uint16(data, offset, "Muscle Percentage")
        muscle_percent = _scaled(raw, _PERCENT_RESOLUTION)

    optional_masses: Dict[str, float | None] = {}
    for flag, name in (
        (_BC_FLAG_MUSCLE_MASS, "muscle_mass"),
        (_BC_FLAG_FAT_FREE_MASS, "fat_free_mass"),
        (_BC_FLAG_SOFT_LEAN_MASS, "soft_lean_mass"),
        (_BC_FLAG_BODY_WATER_MASS, "body_water_mass"),
    ):
        if flags & flag:
            raw, offset = _read_uint16(data, offset, name)
            optional_masses[name] = _scaled(raw, mass_res)

    impedance_ohm = None
    if flags & _BC_FLAG_IMPEDANCE:
        raw, offset = _read_uint16(data, offset, "Impedance")
        impedance_ohm = _scaled(raw, _IMPEDANCE_RESOLUTION)

    weight = None
    if flags & _BC_FLAG_WEIGHT:
        raw, offset = _read_uint16(data, offset, "Weight")
        weight = _scaled(raw, mass_res)

    height = None
    if flags & _BC_FLAG_HEIGHT:
        raw, offset = _read_uint16(data, offset, "Height")
        height = _scaled(raw, height_res)

    return BodyComposition(
        body_fat_percent=body_fat_percent,
        imperial=imperial,
        timestamp=timestamp,
        user_id=user_id,
        basal_metabolism_kj=basal_metabolism_kj,
        muscle_percent=muscle_percent,
        impedance_ohm=impedance_ohm,
        weight=weight,
        height=height,
        **optional_masses,
    )


# --- Weight Measurement (0x2A9D) -------------------------------------------

_WM_FLAG_IMPERIAL = 1 << 0
_WM_FLAG_TIMESTAMP = 1 << 1
_WM_FLAG_USER_ID = 1 << 2
_WM_FLAG_BMI_AND_HEIGHT = 1 << 3

#: BMI is in units of 0.1 kg/m^2.
_BMI_RESOLUTION = 0.1


@dataclass(frozen=True, slots=True)
class WeightMeasurement:
    """A decoded 0x2A9D payload."""

    weight: float | None
    imperial: bool
    timestamp: datetime | None = None
    user_id: int | None = None
    bmi: float | None = None
    height: float | None = None

    @property
    def mass_unit(self) -> str:
        return "lb" if self.imperial else "kg"


def decode_weight_measurement(data: bytes) -> WeightMeasurement:
    """Decode Weight Measurement (0x2A9D)."""
    if len(data) < 3:
        raise DecodeError(f"0x2A9D payload too short: {len(data)} bytes, need >= 3")
    flags = data[0]
    imperial = bool(flags & _WM_FLAG_IMPERIAL)
    mass_res = _MASS_LB_RESOLUTION if imperial else _MASS_KG_RESOLUTION
    height_res = _HEIGHT_IN_RESOLUTION if imperial else _HEIGHT_M_RESOLUTION

    offset = 1
    raw, offset = _read_uint16(data, offset, "Weight")
    weight = _scaled(raw, mass_res)

    timestamp = None
    if flags & _WM_FLAG_TIMESTAMP:
        timestamp, offset = decode_datetime(data, offset)

    user_id = None
    if flags & _WM_FLAG_USER_ID:
        if offset + 1 > len(data):
            raise DecodeError("payload truncated before User ID")
        user_id = data[offset]
        offset += 1

    bmi = height = None
    if flags & _WM_FLAG_BMI_AND_HEIGHT:
        raw_bmi, offset = _read_uint16(data, offset, "BMI")
        bmi = _scaled(raw_bmi, _BMI_RESOLUTION)
        raw_height, offset = _read_uint16(data, offset, "Height")
        height = _scaled(raw_height, height_res)

    return WeightMeasurement(
        weight=weight,
        imperial=imperial,
        timestamp=timestamp,
        user_id=user_id,
        bmi=bmi,
        height=height,
    )


# --- Blood Pressure Measurement (0x2A35) -----------------------------------

_BP_FLAG_KPA = 1 << 0
_BP_FLAG_TIMESTAMP = 1 << 1
_BP_FLAG_PULSE_RATE = 1 << 2
_BP_FLAG_USER_ID = 1 << 3
_BP_FLAG_MEASUREMENT_STATUS = 1 << 4

#: Measurement Status bits that mean the reading is not trustworthy. Surfaced
#: rather than swallowed: a cuff that detected motion produced a number, and
#: a number without its caveat is worse than no number.
BP_STATUS_FLAGS: Tuple[Tuple[int, str], ...] = (
    (1 << 0, "body movement detected"),
    (1 << 1, "cuff too loose"),
    (1 << 2, "irregular pulse detected"),
    (1 << 5, "improper measurement position"),
)


@dataclass(frozen=True, slots=True)
class BloodPressure:
    """A decoded 0x2A35 payload.

    Medical context, not a factor. A reading here changes how sleep, movement
    and mind should be interpreted; it is not a dimension of health alongside
    them.
    """

    systolic: float | None
    diastolic: float | None
    mean_arterial_pressure: float | None
    kilopascals: bool
    timestamp: datetime | None = None
    pulse_rate: float | None = None
    user_id: int | None = None
    status_flags: Tuple[str, ...] = ()

    @property
    def unit(self) -> str:
        return "kPa" if self.kilopascals else "mmHg"

    @property
    def is_reliable(self) -> bool:
        """False when the device itself flagged a problem with the reading."""
        return not self.status_flags

    @property
    def is_complete(self) -> bool:
        """False when any core pressure came back as a non-measurement."""
        return None not in (self.systolic, self.diastolic)


def decode_blood_pressure(data: bytes) -> BloodPressure:
    """Decode Blood Pressure Measurement (0x2A35)."""
    if len(data) < 7:
        raise DecodeError(f"0x2A35 payload too short: {len(data)} bytes, need >= 7")
    flags = data[0]
    offset = 1
    systolic, offset = _read_sfloat(data, offset, "Systolic")
    diastolic, offset = _read_sfloat(data, offset, "Diastolic")
    mean_arterial, offset = _read_sfloat(data, offset, "Mean Arterial Pressure")

    timestamp = None
    if flags & _BP_FLAG_TIMESTAMP:
        timestamp, offset = decode_datetime(data, offset)

    pulse_rate = None
    if flags & _BP_FLAG_PULSE_RATE:
        pulse_rate, offset = _read_sfloat(data, offset, "Pulse Rate")

    user_id = None
    if flags & _BP_FLAG_USER_ID:
        if offset + 1 > len(data):
            raise DecodeError("payload truncated before User ID")
        user_id = data[offset]
        offset += 1

    status: List[str] = []
    if flags & _BP_FLAG_MEASUREMENT_STATUS:
        if offset + 2 > len(data):
            raise DecodeError("payload truncated before Measurement Status")
        (raw_status,) = struct.unpack_from("<H", data, offset)
        offset += 2
        status = [label for bit, label in BP_STATUS_FLAGS if raw_status & bit]

    return BloodPressure(
        systolic=systolic,
        diastolic=diastolic,
        mean_arterial_pressure=mean_arterial,
        kilopascals=bool(flags & _BP_FLAG_KPA),
        timestamp=timestamp,
        pulse_rate=pulse_rate,
        user_id=user_id,
        status_flags=tuple(status),
    )


# --- Glucose Measurement (0x2A18) ------------------------------------------

_GL_FLAG_TIME_OFFSET = 1 << 0
_GL_FLAG_CONCENTRATION = 1 << 1
_GL_FLAG_MOL_PER_L = 1 << 2
_GL_FLAG_STATUS = 1 << 3

#: The specification transmits kg/L and mol/L, which are not the units anyone
#: reads a glucose result in. Converting here keeps the awkward scaling in one
#: place instead of at every call site.
_KG_PER_L_TO_MG_PER_DL = 100_000.0
_MOL_PER_L_TO_MMOL_PER_L = 1000.0

GLUCOSE_TYPES: Dict[int, str] = {
    1: "capillary whole blood",
    2: "capillary plasma",
    3: "venous whole blood",
    4: "venous plasma",
    5: "arterial whole blood",
    6: "arterial plasma",
    7: "undetermined whole blood",
    8: "undetermined plasma",
    9: "interstitial fluid",
    10: "control solution",
}

GLUCOSE_SAMPLE_LOCATIONS: Dict[int, str] = {
    1: "finger",
    2: "alternate site test",
    3: "earlobe",
    4: "control solution",
    15: "not available",
}


@dataclass(frozen=True, slots=True)
class GlucoseMeasurement:
    """A decoded 0x2A18 payload."""

    sequence_number: int
    base_time: datetime | None
    time_offset_minutes: int | None = None
    concentration_mg_dl: float | None = None
    concentration_mmol_l: float | None = None
    sample_type: str | None = None
    sample_location: str | None = None

    @property
    def is_interstitial(self) -> bool:
        """Whether this came from interstitial fluid rather than blood.

        A continuous sensor measures interstitial glucose, which lags blood
        glucose. Treating the two as the same measurand is the metabolic
        equivalent of reading wrist SpO2 as arterial saturation.
        """
        return self.sample_type == "interstitial fluid"


def decode_glucose(data: bytes) -> GlucoseMeasurement:
    """Decode Glucose Measurement (0x2A18)."""
    if len(data) < 10:
        raise DecodeError(f"0x2A18 payload too short: {len(data)} bytes, need >= 10")
    flags = data[0]
    (sequence_number,) = struct.unpack_from("<H", data, 1)
    base_time, offset = decode_datetime(data, 3)

    time_offset_minutes = None
    if flags & _GL_FLAG_TIME_OFFSET:
        if offset + 2 > len(data):
            raise DecodeError("payload truncated before Time Offset")
        (time_offset_minutes,) = struct.unpack_from("<h", data, offset)
        offset += 2

    mg_dl = mmol_l = None
    sample_type = sample_location = None
    if flags & _GL_FLAG_CONCENTRATION:
        concentration, offset = _read_sfloat(data, offset, "Glucose Concentration")
        if offset + 1 > len(data):
            raise DecodeError("payload truncated before Type/Sample Location")
        type_location = data[offset]
        offset += 1
        sample_type = GLUCOSE_TYPES.get(type_location & 0x0F)
        sample_location = GLUCOSE_SAMPLE_LOCATIONS.get((type_location >> 4) & 0x0F)
        if concentration is not None:
            if flags & _GL_FLAG_MOL_PER_L:
                mmol_l = concentration * _MOL_PER_L_TO_MMOL_PER_L
            else:
                mg_dl = concentration * _KG_PER_L_TO_MG_PER_DL

    return GlucoseMeasurement(
        sequence_number=sequence_number,
        base_time=base_time,
        time_offset_minutes=time_offset_minutes,
        concentration_mg_dl=mg_dl,
        concentration_mmol_l=mmol_l,
        sample_type=sample_type,
        sample_location=sample_location,
    )


# --- Running Speed and Cadence (0x2A53) ------------------------------------

_RSC_FLAG_STRIDE_LENGTH = 1 << 0
_RSC_FLAG_TOTAL_DISTANCE = 1 << 1
_RSC_FLAG_RUNNING = 1 << 2

_STRIDE_LENGTH_RESOLUTION = 0.01
_TOTAL_DISTANCE_RESOLUTION = 0.1


@dataclass(frozen=True, slots=True)
class RunningCadence:
    """A decoded 0x2A53 payload."""

    speed_m_s: float
    cadence_spm: int
    running: bool
    stride_length_m: float | None = None
    total_distance_m: float | None = None

    @property
    def speed_kph(self) -> float:
        return self.speed_m_s * 3.6


def decode_running_cadence(data: bytes) -> RunningCadence:
    """Decode RSC Measurement (0x2A53).

    Cadence here is steps per minute for a single foot pod, so a two-sided
    step rate is twice this. The specification's own wording is the reason
    this is stated rather than assumed.
    """
    if len(data) < 4:
        raise DecodeError(f"0x2A53 payload too short: {len(data)} bytes, need >= 4")
    flags = data[0]
    (speed_raw,) = struct.unpack_from("<H", data, 1)
    cadence = data[3]
    offset = 4

    stride_length_m = None
    if flags & _RSC_FLAG_STRIDE_LENGTH:
        if offset + 2 > len(data):
            raise DecodeError("payload truncated before Stride Length")
        (raw,) = struct.unpack_from("<H", data, offset)
        stride_length_m = raw * _STRIDE_LENGTH_RESOLUTION
        offset += 2

    total_distance_m = None
    if flags & _RSC_FLAG_TOTAL_DISTANCE:
        if offset + 4 > len(data):
            raise DecodeError("payload truncated before Total Distance")
        (raw32,) = struct.unpack_from("<I", data, offset)
        total_distance_m = raw32 * _TOTAL_DISTANCE_RESOLUTION
        offset += 4

    return RunningCadence(
        speed_m_s=speed_raw / RSC_SPEED_UNITS_PER_MS,
        cadence_spm=cadence,
        running=bool(flags & _RSC_FLAG_RUNNING),
        stride_length_m=stride_length_m,
        total_distance_m=total_distance_m,
    )


# --- Cycling Speed and Cadence (0x2A5B) ------------------------------------

_CSC_FLAG_WHEEL = 1 << 0
_CSC_FLAG_CRANK = 1 << 1


@dataclass(frozen=True, slots=True)
class CyclingCadence:
    """A decoded 0x2A5B payload.

    Both counters are cumulative and both event times wrap at 64 seconds, so
    a rate is only meaningful between two consecutive notifications. This type
    deliberately holds raw counters rather than a computed cadence: computing
    one from a single packet is impossible, and returning a number anyway
    would be an invention.
    """

    cumulative_wheel_revolutions: int | None = None
    last_wheel_event_time_s: float | None = None
    cumulative_crank_revolutions: int | None = None
    last_crank_event_time_s: float | None = None


def decode_cycling_cadence(data: bytes) -> CyclingCadence:
    """Decode CSC Measurement (0x2A5B)."""
    if len(data) < 1:
        raise DecodeError("0x2A5B payload is empty")
    flags = data[0]
    offset = 1

    wheel_revs = wheel_time = None
    if flags & _CSC_FLAG_WHEEL:
        if offset + 6 > len(data):
            raise DecodeError("payload truncated before wheel revolution data")
        wheel_revs, wheel_raw = struct.unpack_from("<IH", data, offset)
        wheel_time = wheel_raw / EVENT_TIME_UNITS_PER_SECOND
        offset += 6

    crank_revs = crank_time = None
    if flags & _CSC_FLAG_CRANK:
        if offset + 4 > len(data):
            raise DecodeError("payload truncated before crank revolution data")
        crank_revs, crank_raw = struct.unpack_from("<HH", data, offset)
        crank_time = crank_raw / EVENT_TIME_UNITS_PER_SECOND
        offset += 4

    return CyclingCadence(
        cumulative_wheel_revolutions=wheel_revs,
        last_wheel_event_time_s=wheel_time,
        cumulative_crank_revolutions=crank_revs,
        last_crank_event_time_s=crank_time,
    )


#: What each newly decodable characteristic contributes, in the same shape as
#: ``gatt.DERIVABLE_FROM`` so the two merge into one capability answer.
DERIVABLE_FROM_MEASUREMENTS: Dict[str, List[str]] = {
    CHAR_BODY_COMPOSITION_MEASUREMENT: [
        "body_fat_percentage",
        "skeletal_muscle_mass",
        "fat_free_mass",
        "body_water_mass",
        "bioimpedance",
        "lean_mass_trend (derived)",
    ],
    CHAR_WEIGHT_MEASUREMENT: ["body_mass", "bmi", "body_mass_trend (derived)"],
    CHAR_GLUCOSE_MEASUREMENT: [
        "blood_glucose",
        "interstitial_glucose",
        "post_prandial_response (derived)",
    ],
    CHAR_BLOOD_PRESSURE_MEASUREMENT: [
        "systolic_pressure",
        "diastolic_pressure",
        "mean_arterial_pressure",
        "pulse_rate",
    ],
    CHAR_RSC_MEASUREMENT: [
        "running_speed",
        "running_cadence",
        "stride_length",
        "distance",
    ],
    CHAR_CSC_MEASUREMENT: [
        "wheel_revolutions",
        "crank_revolutions",
        "cycling_cadence (between packets)",
    ],
}
