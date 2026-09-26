"""Bluetooth SIG assigned numbers for the fitness-relevant GATT surface.

Only the services and characteristics that actually carry usable physiology are
listed. The point of collecting them here is that this set -- all of it public,
all of it royalty-free, none of it requiring a partner agreement -- is
considerably wider than what any shipping fitness app reads.
"""

from __future__ import annotations

from typing import Dict

BASE_UUID_SUFFIX = "-0000-1000-8000-00805f9b34fb"


def uuid16(value: int) -> str:
    """Expand a 16-bit assigned number to its full 128-bit UUID string."""
    return f"0000{value:04x}{BASE_UUID_SUFFIX}"


# --- Services ---------------------------------------------------------------

SVC_DEVICE_INFORMATION = 0x180A
SVC_BATTERY = 0x180F
SVC_HEART_RATE = 0x180D
SVC_HEALTH_THERMOMETER = 0x1809
SVC_CYCLING_SPEED_CADENCE = 0x1816
SVC_CYCLING_POWER = 0x1818
SVC_RUNNING_SPEED_CADENCE = 0x1814
SVC_FITNESS_MACHINE = 0x1826
SVC_WEIGHT_SCALE = 0x181D
SVC_BODY_COMPOSITION = 0x181B
SVC_GLUCOSE = 0x1808
SVC_CONTINUOUS_GLUCOSE = 0x181F
SVC_PULSE_OXIMETER = 0x1822
SVC_BLOOD_PRESSURE = 0x1810
SVC_USER_DATA = 0x181C

# --- Characteristics --------------------------------------------------------

CHR_BATTERY_LEVEL = 0x2A19
CHR_HEART_RATE_MEASUREMENT = 0x2A37
CHR_BODY_SENSOR_LOCATION = 0x2A38
CHR_CSC_MEASUREMENT = 0x2A5B
CHR_RSC_MEASUREMENT = 0x2A53
CHR_CYCLING_POWER_MEASUREMENT = 0x2A63
CHR_CYCLING_POWER_VECTOR = 0x2A64
CHR_TREADMILL_DATA = 0x2ACD
CHR_ROWER_DATA = 0x2AD1
CHR_INDOOR_BIKE_DATA = 0x2AD2
CHR_CROSS_TRAINER_DATA = 0x2ACE
CHR_FITNESS_MACHINE_CONTROL_POINT = 0x2AD9
CHR_FITNESS_MACHINE_STATUS = 0x2ADA
CHR_FITNESS_MACHINE_FEATURE = 0x2ACC
CHR_WEIGHT_MEASUREMENT = 0x2A9D
CHR_BODY_COMPOSITION_MEASUREMENT = 0x2A9C
CHR_GLUCOSE_MEASUREMENT = 0x2A18
CHR_CGM_MEASUREMENT = 0x2AA7
CHR_PLX_CONTINUOUS_MEASUREMENT = 0x2A5F
CHR_PLX_SPOT_CHECK_MEASUREMENT = 0x2A5E
CHR_TEMPERATURE_MEASUREMENT = 0x2A1C
CHR_BLOOD_PRESSURE_MEASUREMENT = 0x2A35

#: Characteristics this package can decode, mapped to a human label.
DECODABLE: Dict[int, str] = {
    CHR_HEART_RATE_MEASUREMENT: "Heart Rate Measurement",
    CHR_CSC_MEASUREMENT: "CSC Measurement",
    CHR_RSC_MEASUREMENT: "RSC Measurement",
    CHR_CYCLING_POWER_MEASUREMENT: "Cycling Power Measurement",
    CHR_INDOOR_BIKE_DATA: "Indoor Bike Data",
    CHR_TREADMILL_DATA: "Treadmill Data",
    CHR_ROWER_DATA: "Rower Data",
    CHR_BODY_COMPOSITION_MEASUREMENT: "Body Composition Measurement",
    CHR_GLUCOSE_MEASUREMENT: "Glucose Measurement",
    CHR_PLX_CONTINUOUS_MEASUREMENT: "PLX Continuous Measurement",
    CHR_BATTERY_LEVEL: "Battery Level",
}


def name_for(char_uuid16: int) -> str:
    return DECODABLE.get(char_uuid16, f"Unknown characteristic 0x{char_uuid16:04X}")
