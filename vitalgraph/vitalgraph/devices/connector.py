"""Normalising a named device's BLE notifications into canonical Samples.

This is the piece that makes device-agnosticism real rather than aspirational:
every OPEN_BLE product in ``catalog.py`` decodes through the same path,
``ble/gatt.py``'s spec decoders, and lands in the same
:class:`~vitalgraph.biometrics.schema.Sample` shape that the simulator, the
store, the feature pipeline and the RAG bridge already consume. A Polar strap
and a Wahoo strap produce indistinguishable samples once decoded -- which is
correct, because at the standard-GATT layer they are indistinguishable.

Deliberately narrow. Only :class:`~vitalgraph.devices.catalog.AccessMode.OPEN_BLE`
devices are handled here, because those are the only ones whose bytes we are
entitled to decode ourselves. A CLOUD_API device's data arrives already
structured from the vendor's own API (not implemented here -- it needs a
credentialed OAuth flow this module has no business doing on spec), and a
CLOSED device has no path at all. Routing a cloud or closed device through
this decoder would silently claim a capability the product does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from ..ble import gatt
from ..biometrics.schema import Sample, SignalType
from .catalog import AccessMode, DeviceProfile, get


class UnsupportedCharacteristic(ValueError):
    """Raised when a characteristic has no known decoder.

    Only characteristics with a real decoder in ``ble/gatt.py`` are handled;
    anything else fails loudly rather than silently dropping data, since a
    silent drop looks identical to "the device sent nothing."
    """


class DeviceNotDirectlyConnectable(ValueError):
    """Raised when asked to decode bytes for a CLOUD_API or CLOSED device.

    Those devices' bytes are not decodable by us at all -- CLOUD_API data
    arrives pre-structured from the vendor's API, and CLOSED devices expose
    nothing we have a right to decode.
    """


@dataclass(frozen=True, slots=True)
class DecodedNotification:
    """One decoded BLE notification: the samples it yielded, from which
    device and characteristic."""

    device_id: str
    characteristic: str
    samples: List[Sample]


def _require_open_ble(profile: DeviceProfile) -> None:
    if profile.access_mode is not AccessMode.OPEN_BLE:
        raise DeviceNotDirectlyConnectable(
            f"{profile.name} is {profile.access_mode.value}, not directly "
            f"connectable over BLE; its data does not arrive as raw GATT "
            f"bytes for this module to decode"
        )


def decode_heart_rate_notification(
    device_id: str, raw: bytes, received_at: datetime | None = None
) -> DecodedNotification:
    """Decode a Heart Rate Measurement (0x2A37) notification for a named
    device, producing the same Sample shapes the simulator and the generic
    ``/api/ingest/gatt`` path produce.

    Args:
        device_id: catalogue id, e.g. ``"polar_h10"``. Validated against the
            catalogue so decoding a characteristic a device does not actually
            advertise is refused rather than silently accepted.
        raw: the raw characteristic value, exactly as delivered by the BLE
            stack.
        received_at: notification arrival time; defaults to now. RR
            intervals are stamped backwards from this instant, mirroring the
            generic ingest path.
    """
    profile = get(device_id)
    _require_open_ble(profile)
    if gatt.SERVICE_HEART_RATE not in profile.gatt_services:
        raise UnsupportedCharacteristic(
            f"{profile.name} does not advertise the standard Heart Rate "
            f"service; refusing to decode as one"
        )

    measurement = gatt.decode_heart_rate_measurement(raw)
    source = f"device:{device_id}:0x2A37"
    base = received_at or datetime.now(timezone.utc)

    samples = [
        Sample(
            ts=base,
            signal=SignalType.HEART_RATE,
            value=float(measurement.heart_rate_bpm),
            source=source,
        )
    ]
    offset = 0.0
    for rr in reversed(measurement.rr_intervals_ms):
        offset += rr
        samples.append(
            Sample(
                ts=base - timedelta(milliseconds=offset),
                signal=SignalType.RR_INTERVAL,
                value=rr,
                source=source,
            )
        )
    return DecodedNotification(
        device_id=device_id,
        characteristic=gatt.CHAR_HEART_RATE_MEASUREMENT,
        samples=samples,
    )


def decode_pulse_oximeter_notification(
    device_id: str, spo2_percent: float, received_at: datetime | None = None
) -> DecodedNotification:
    """Record a Pulse Oximeter Service (0x1822) reading for a named device.

    ``ble/gatt.py`` does not yet implement a byte-level PLX decoder (the
    spec's continuous-measurement characteristic carries more fields than
    VitalGraph currently uses), so this takes an already-parsed percentage.
    The device-catalogue and access-mode checks below are the part that
    matters: they are identical to the Heart Rate path, so adding a real byte
    decoder later slots in without touching this validation.
    """
    profile = get(device_id)
    _require_open_ble(profile)
    if gatt.SERVICE_PULSE_OXIMETER not in profile.gatt_services:
        raise UnsupportedCharacteristic(
            f"{profile.name} does not advertise the standard Pulse Oximeter "
            f"service; refusing to decode as one"
        )

    sample = Sample(
        ts=received_at or datetime.now(timezone.utc),
        signal=SignalType.SPO2,
        value=spo2_percent,
        source=f"device:{device_id}:0x2A5F",
    )
    return DecodedNotification(
        device_id=device_id, characteristic=gatt.CHAR_PLX_CONTINUOUS, samples=[sample]
    )
