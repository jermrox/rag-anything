"""The device landscape: named commercial products, and whether Vybe can
actually reach their sensors.

This is the module that makes "replace the device's app with ours" an
engineering claim instead of a slogan. Being device-agnostic only works where
a device's data is actually reachable, and reachability splits into three
structurally different cases that must never be collapsed into one:

* **OPEN_BLE** -- the device broadcasts standard Bluetooth SIG GATT services
  (Heart Rate 0x180D, Pulse Oximeter 0x1822, ...) that any app, including
  ours, can subscribe to directly with no vendor cooperation. This is the
  ``ble/gatt.py`` decoder's entire domain.
* **CLOUD_API** -- the device's own radio protocol is proprietary and closed;
  the *only* legitimate path to its data is the vendor's own cloud API, after
  the user authorizes it (OAuth). We do not, and cannot, decode these
  devices' BLE traffic ourselves -- claiming otherwise would be exactly the
  hand-wavey behaviour the product is built to avoid.
* **CLOSED** -- no third-party path exists at all today, direct or via API.

Precision here matters more than optimism. A catalogue that overstates
reachability produces the same failure the sensor/signal taxonomy exists to
prevent: a confident claim the underlying hardware cannot support. Every
entry below is sourced from what the vendor itself documents (public API
docs, developer terms) or from the Bluetooth SIG's own published service
list -- never from reverse-engineering a proprietary protocol we have not
verified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from ..ble import gatt
from ..knowledge.sensors import BY_ID as SENSORS_BY_ID
from ..knowledge.sensors import UnknownSensor


class AccessMode(str, Enum):
    """How -- if at all -- a third-party app can reach a device's data."""

    OPEN_BLE = "open_ble"
    """Standard GATT services, subscribable directly. No vendor involved."""

    CLOUD_API = "cloud_api"
    """Vendor-operated API, reachable only after the user grants OAuth
    consent. The radio protocol itself stays closed."""

    CLOSED = "closed"
    """No third-party path today, BLE or API."""


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """One commercial product: what it measures and how Vybe can reach it."""

    id: str
    name: str
    vendor: str
    access_mode: AccessMode
    sensor_ids: Tuple[str, ...]
    """Which entries in ``knowledge/sensors.py`` this product corresponds to."""
    gatt_services: Tuple[str, ...] = field(default_factory=tuple)
    """Standard service UUIDs actually advertised, when OPEN_BLE."""
    api_docs_url: str = ""
    """Where the vendor documents the integration path, when CLOUD_API."""
    notes: str = ""

    def __post_init__(self) -> None:
        for sensor_id in self.sensor_ids:
            if sensor_id not in SENSORS_BY_ID:
                raise UnknownSensor(
                    f"device {self.id!r} references unknown sensor {sensor_id!r}"
                )
        if self.access_mode is AccessMode.OPEN_BLE and not self.gatt_services:
            raise ValueError(
                f"device {self.id!r} is OPEN_BLE but declares no gatt_services"
            )
        if self.access_mode is AccessMode.CLOUD_API and not self.api_docs_url:
            raise ValueError(
                f"device {self.id!r} is CLOUD_API but declares no api_docs_url"
            )

    @property
    def is_directly_connectable(self) -> bool:
        """Whether Vybe's own Web Bluetooth session can subscribe to this
        device with no vendor cooperation at all."""
        return self.access_mode is AccessMode.OPEN_BLE

    def signals_delivered(self) -> Dict[str, float]:
        """Union of every signal this product's sensors deliver, at the
        fastest rate any of them achieves."""
        out: Dict[str, float] = {}
        for sensor_id in self.sensor_ids:
            for signal_id, rate in SENSORS_BY_ID[sensor_id].delivers.items():
                out[signal_id] = max(rate, out.get(signal_id, 0.0))
        return out

    def meets_minimum(self, signal_id: str) -> bool | None:
        """Whether *any* sensor on this product clears the signal's usable
        minimum. Mirrors ``Sensor.meets_minimum``'s None-for-absent contract."""
        best: bool | None = None
        for sensor_id in self.sensor_ids:
            result = SENSORS_BY_ID[sensor_id].meets_minimum(signal_id)
            if result is True:
                return True
            if result is False:
                best = False
        return best


def _device(*args, **kwargs) -> DeviceProfile:
    return DeviceProfile(*args, **kwargs)


DEVICES: Tuple[DeviceProfile, ...] = (
    # --- Open BLE: standard GATT, directly connectable today --------------
    _device(
        "polar_h10",
        "Polar H10",
        "Polar",
        AccessMode.OPEN_BLE,
        sensor_ids=("ecg_chest_strap",),
        gatt_services=(gatt.SERVICE_HEART_RATE,),
        notes=(
            "Chest ECG. Broadcasts standard Heart Rate service with RR "
            "intervals (flag bit 4 set); also exposes Polar's own proprietary "
            "PMD service for raw ECG waveform, which is outside GATT-standard "
            "reach and not claimed here. The RR stream alone is the cleanest "
            "signal this product line offers over open BLE."
        ),
    ),
    _device(
        "polar_verity_sense",
        "Polar Verity Sense",
        "Polar",
        AccessMode.OPEN_BLE,
        sensor_ids=("ppg_wrist",),
        gatt_services=(gatt.SERVICE_HEART_RATE,),
        notes=(
            "Arm/temple optical HR band. Standard Heart Rate service with "
            "RR intervals; mapped to the ppg_wrist sensor profile as the "
            "nearest match -- motion artifact and perfusion at this site are "
            "closer to wrist-class than finger-class. It advertises no "
            "Pulse Oximeter service, so no SpO2 claim is made for it here."
        ),
    ),
    _device(
        "wahoo_tickr",
        "Wahoo TICKR",
        "Wahoo",
        AccessMode.OPEN_BLE,
        sensor_ids=("ecg_chest_strap",),
        gatt_services=(gatt.SERVICE_HEART_RATE,),
        notes="Chest strap. Standard Heart Rate service, RR intervals present.",
    ),
    _device(
        "garmin_hrm_strap",
        "Garmin HRM-Pro / HRM-Dual",
        "Garmin",
        AccessMode.OPEN_BLE,
        sensor_ids=("ecg_chest_strap",),
        gatt_services=(gatt.SERVICE_HEART_RATE,),
        notes=(
            "Chest strap. Broadcasts standard Heart Rate over BLE regardless "
            "of Garmin watch pairing -- the strap itself is open even though "
            "Garmin's watches (below) are not."
        ),
    ),
    _device(
        "generic_ble_hr_strap",
        "Any Bluetooth SIG-compliant HR strap",
        "various",
        AccessMode.OPEN_BLE,
        sensor_ids=("ecg_chest_strap",),
        gatt_services=(gatt.SERVICE_HEART_RATE,),
        notes=(
            "Catch-all for the broad market of compliant straps (CooSpo, "
            "Movesense, Xoss, ...). Interchangeable at the protocol level: "
            "conformance to the SIG Heart Rate service is the actual "
            "requirement, not any specific brand."
        ),
    ),
    _device(
        "nonin_pulse_ox",
        "Nonin 3230/9560 Bluetooth pulse oximeter",
        "Nonin",
        AccessMode.OPEN_BLE,
        sensor_ids=("ppg_finger",),
        gatt_services=(gatt.SERVICE_PULSE_OXIMETER,),
        notes=(
            "Medical-grade fingertip oximeter with continuous BLE streaming "
            "via the standard Pulse Oximeter service. Mapped to the "
            "ppg_finger sensor profile as the closest match in the "
            "taxonomy; a real continuous-stream medical oximeter delivers "
            "SpO2 materially faster than a consumer ring reporting under "
            "that same sensor id, which is a taxonomy gap worth widening "
            "before this device is used to gate an apnea-adjacent claim -- "
            "see knowledge/domains.py sleep-disordered-breathing."
        ),
    ),
    # --- Cloud API: closed radio, open account-level integration ----------
    _device(
        "whoop_4",
        "Whoop 4.0 / Whoop MG",
        "Whoop",
        AccessMode.CLOUD_API,
        sensor_ids=("ppg_wrist",),
        api_docs_url="https://developer.whoop.com/",
        notes=(
            "Wrist PPG. The device's own BLE link to its phone app is "
            "proprietary and undocumented; Whoop does not publish a GATT "
            "profile for it. The only legitimate integration path is the "
            "Whoop Developer Platform, a REST API reached after the user "
            "authorizes access via OAuth -- daily/derived metrics, not raw "
            "beat-to-beat data. This is the device the investor's critique "
            "was made about: even with API access, the underlying sensor is "
            "still wrist PPG, so the signal-adequacy limits in "
            "knowledge/sensors.py apply exactly as they would to any other "
            "wrist-PPG product."
        ),
    ),
    _device(
        "oura_ring_4",
        "Oura Ring Gen 3 / Gen 4",
        "Oura",
        AccessMode.CLOUD_API,
        sensor_ids=("ppg_finger",),
        api_docs_url="https://cloud.ouraring.com/v2/docs",
        notes=(
            "Finger-adjacent PPG (worn at the base of the finger, closer in "
            "practice to ppg_finger than ppg_wrist). No public BLE GATT "
            "profile; the Oura API v2 exposes daily summaries and some "
            "session-level data, not a live beat stream."
        ),
    ),
    _device(
        "fitbit_sense_charge",
        "Fitbit Sense / Charge",
        "Fitbit (Google)",
        AccessMode.CLOUD_API,
        sensor_ids=("ppg_wrist",),
        api_docs_url="https://dev.fitbit.com/build/reference/web-api/",
        notes="Wrist PPG. Fitbit Web API, OAuth-gated, no public BLE profile.",
    ),
    _device(
        "garmin_watch",
        "Garmin wrist watches (Forerunner / Fenix / Venu, etc.)",
        "Garmin",
        AccessMode.CLOUD_API,
        sensor_ids=("ppg_wrist",),
        api_docs_url="https://developer.garmin.com/gc-developer-program/health-api/",
        notes=(
            "Wrist PPG on-watch. Garmin's own chest straps (above) are open "
            "BLE; the watch itself is not -- Garmin Connect's Health API is "
            "the only third-party path to what the watch records."
        ),
    ),
    _device(
        "apple_watch",
        "Apple Watch (all generations)",
        "Apple",
        AccessMode.CLOUD_API,
        sensor_ids=("ppg_wrist", "ecg_handheld"),
        api_docs_url="https://developer.apple.com/documentation/healthkit",
        notes=(
            "Wrist PPG plus an on-demand single-lead ECG. No third-party BLE "
            "access at all; HealthKit is an on-device data store an iOS app "
            "reads with user permission, not a network API -- so any "
            "integration has to ship as an iOS companion app, not a browser "
            "or server-side connector."
        ),
    ),
    _device(
        "samsung_galaxy_watch",
        "Samsung Galaxy Watch",
        "Samsung",
        AccessMode.CLOUD_API,
        sensor_ids=("ppg_wrist", "ecg_handheld"),
        api_docs_url="https://developer.samsung.com/health",
        notes="Wrist PPG plus on-demand ECG. Samsung Health platform, OAuth-gated.",
    ),
    # --- Closed: no third-party path today ---------------------------------
    _device(
        "whoop_generic_ble",
        "Whoop's own BLE link (device-to-phone)",
        "Whoop",
        AccessMode.CLOSED,
        sensor_ids=("ppg_wrist",),
        notes=(
            "Listed separately from whoop_4 above to keep the distinction "
            "explicit: the device-to-phone radio link itself is closed even "
            "though the vendor's cloud API is open. A scan will see the "
            "device advertise, but its service and characteristic UUIDs are "
            "proprietary and undocumented -- there is no standard GATT "
            "profile to subscribe to, and reverse-engineering an encrypted "
            "vendor link without authorization is out of scope for this "
            "product regardless of feasibility."
        ),
    ),
)

BY_ID: Dict[str, DeviceProfile] = {d.id: d for d in DEVICES}


class UnknownDevice(KeyError):
    """Raised when a device id is not in the catalogue."""


def get(device_id: str) -> DeviceProfile:
    try:
        return BY_ID[device_id]
    except KeyError as exc:
        raise UnknownDevice(f"no device profile for {device_id!r}") from exc


def by_access_mode(mode: AccessMode) -> List[DeviceProfile]:
    return [d for d in DEVICES if d.access_mode is mode]


def directly_connectable() -> List[DeviceProfile]:
    """Devices Vybe's own BLE session can subscribe to with zero vendor
    cooperation -- the actual "replace the app" surface area today."""
    return by_access_mode(AccessMode.OPEN_BLE)


def requiring_vendor_api() -> List[DeviceProfile]:
    return by_access_mode(AccessMode.CLOUD_API)


def supporting(signal_id: str, minimum: bool = True) -> List[DeviceProfile]:
    """Devices that deliver ``signal_id``.

    Args:
        minimum: when True (default), also require the delivered rate clear
            the signal's usable minimum -- not merely claim the signal.
    """
    out = [d for d in DEVICES if signal_id in d.signals_delivered()]
    if minimum:
        out = [d for d in out if d.meets_minimum(signal_id)]
    return out
