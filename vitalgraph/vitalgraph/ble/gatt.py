"""Bluetooth GATT decoding for standard health characteristics.

Implements the Bluetooth SIG *published* characteristic layouts. These are open
specifications, so decoding them is interoperability work -- no vendor
reverse-engineering required, and no license encumbrance on the result.

The important commercial fact encoded here: 0x2A37 optionally carries
beat-to-beat RR intervals. Every time-domain HRV metric, and therefore most of
what a premium recovery score is built on, follows from that one field.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Dict, List

# --- Standard 16-bit UUIDs -------------------------------------------------

SERVICE_HEART_RATE = "0x180D"
SERVICE_BATTERY = "0x180F"
SERVICE_HEALTH_THERMOMETER = "0x1809"
SERVICE_PULSE_OXIMETER = "0x1822"
SERVICE_DEVICE_INFORMATION = "0x180A"

CHAR_HEART_RATE_MEASUREMENT = "0x2A37"
CHAR_BODY_SENSOR_LOCATION = "0x2A38"
CHAR_BATTERY_LEVEL = "0x2A19"
CHAR_TEMPERATURE_MEASUREMENT = "0x2A1C"
CHAR_PLX_CONTINUOUS = "0x2A5F"

#: RR intervals are transmitted in units of 1/1024 second, not milliseconds.
#: Forgetting this scaling is the single most common HRV integration bug.
RR_UNITS_PER_SECOND = 1024.0

BODY_SENSOR_LOCATIONS = {
    0: "Other",
    1: "Chest",
    2: "Wrist",
    3: "Finger",
    4: "Hand",
    5: "Ear Lobe",
    6: "Foot",
}


class DecodeError(ValueError):
    """Raised when a characteristic payload does not match its specification."""


@dataclass(frozen=True, slots=True)
class HeartRateMeasurement:
    """Decoded 0x2A37 payload."""

    heart_rate_bpm: int
    rr_intervals_ms: List[float] = field(default_factory=list)
    energy_expended_kj: int | None = None
    sensor_contact_supported: bool = False
    sensor_contact_detected: bool = False

    @property
    def has_rr(self) -> bool:
        return bool(self.rr_intervals_ms)


def decode_heart_rate_measurement(data: bytes) -> HeartRateMeasurement:
    """Decode the GATT Heart Rate Measurement characteristic (0x2A37).

    Layout::

        byte 0      flags
                      bit 0    HR format: 0 = uint8, 1 = uint16
                      bit 1    sensor contact detected
                      bit 2    sensor contact supported
                      bit 3    energy expended present
                      bit 4    RR intervals present
        uint8|uint16  heart rate (little endian)
        uint16        energy expended, kJ      (only if bit 3)
        uint16[]      RR intervals, 1/1024 s   (only if bit 4, until end)
    """
    if len(data) < 2:
        raise DecodeError(f"0x2A37 payload too short: {len(data)} bytes, need >= 2")

    flags = data[0]
    hr_is_uint16 = bool(flags & 0x01)
    contact_detected = bool(flags & 0x02)
    contact_supported = bool(flags & 0x04)
    energy_present = bool(flags & 0x08)
    rr_present = bool(flags & 0x10)

    offset = 1
    if hr_is_uint16:
        if len(data) < offset + 2:
            raise DecodeError("0x2A37 declares uint16 HR but payload is truncated")
        heart_rate = struct.unpack_from("<H", data, offset)[0]
        offset += 2
    else:
        heart_rate = data[offset]
        offset += 1

    energy: int | None = None
    if energy_present:
        if len(data) < offset + 2:
            raise DecodeError(
                "0x2A37 declares energy expended but payload is truncated"
            )
        energy = struct.unpack_from("<H", data, offset)[0]
        offset += 2

    rr_intervals: List[float] = []
    if rr_present:
        remaining = len(data) - offset
        if remaining % 2 != 0:
            raise DecodeError(
                f"0x2A37 RR block has {remaining} trailing bytes; must be a multiple of 2"
            )
        for pos in range(offset, len(data), 2):
            raw = struct.unpack_from("<H", data, pos)[0]
            # 1/1024 s -> ms
            rr_intervals.append(raw * 1000.0 / RR_UNITS_PER_SECOND)

    return HeartRateMeasurement(
        heart_rate_bpm=heart_rate,
        rr_intervals_ms=rr_intervals,
        energy_expended_kj=energy,
        sensor_contact_supported=contact_supported,
        sensor_contact_detected=contact_detected,
    )


def encode_heart_rate_measurement(
    heart_rate_bpm: int, rr_intervals_ms: List[float] | None = None
) -> bytes:
    """Build a spec-compliant 0x2A37 payload.

    Used by the simulator and by round-trip tests -- if encode/decode disagree,
    one of them is wrong about the specification.
    """
    rr_intervals_ms = rr_intervals_ms or []
    flags = 0
    if heart_rate_bpm > 255:
        flags |= 0x01
    if rr_intervals_ms:
        flags |= 0x10

    out = bytearray([flags])
    if heart_rate_bpm > 255:
        out += struct.pack("<H", heart_rate_bpm)
    else:
        out.append(heart_rate_bpm)
    for rr in rr_intervals_ms:
        out += struct.pack("<H", round(rr * RR_UNITS_PER_SECOND / 1000.0))
    return bytes(out)


def decode_battery_level(data: bytes) -> int:
    """Decode Battery Level (0x2A19): a single uint8 percentage."""
    if len(data) != 1:
        raise DecodeError(f"0x2A19 must be exactly 1 byte, got {len(data)}")
    return data[0]


def decode_body_sensor_location(data: bytes) -> str:
    """Decode Body Sensor Location (0x2A38)."""
    if len(data) != 1:
        raise DecodeError(f"0x2A38 must be exactly 1 byte, got {len(data)}")
    return BODY_SENSOR_LOCATIONS.get(data[0], f"Reserved(0x{data[0]:02X})")


#: Which health signals become computable once a device exposes a given
#: characteristic. This is the M1 seed of the ``derivable`` protocol-registry
#: table: point it at a device's advertised characteristics and it answers
#: "what can we actually build from this hardware?"
DERIVABLE_FROM: Dict[str, List[str]] = {
    CHAR_HEART_RATE_MEASUREMENT: [
        "heart_rate",
        "rr_intervals (if flag bit 4 set)",
        "rmssd",
        "sdnn",
        "pnn50",
        "respiratory_rate (via RSA)",
        "readiness/recovery score",
        "sleep staging (with accelerometer)",
    ],
    CHAR_TEMPERATURE_MEASUREMENT: ["skin_temperature", "circadian phase estimate"],
    CHAR_PLX_CONTINUOUS: ["spo2", "perfusion index", "apnea event detection"],
    CHAR_BATTERY_LEVEL: ["device_battery"],
}


def derivable_signals(characteristic_uuids: List[str]) -> Dict[str, List[str]]:
    """Map advertised characteristics to the health signals they unlock."""
    out: Dict[str, List[str]] = {}
    for uuid in characteristic_uuids:
        key = uuid.upper().replace("0X", "0x")
        if key in DERIVABLE_FROM:
            out[key] = DERIVABLE_FROM[key]
    return out
