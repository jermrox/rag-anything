"""The sensor universe: what can be measured, from where, and how well.

Structured along two axes that matter independently:

* **Modality** -- the physical principle (optical, electrical, electrochemical,
  mechanical). Determines *which* signals are obtainable at all.
* **Site** -- where on (or off) the body it sits. Determines how *well*, and it
  matters far more than product marketing implies. The same PPG modality at
  the wrist and at the finger differ by roughly an order of magnitude in
  motion artifact and perfusion.

Every sensor declares the rate at which it actually delivers each signal.
Cross-referenced against ``signals.Signal.min_sampling_hz``, that turns
"can this device support X?" into arithmetic instead of argument.

Where a signal is exposed over Bluetooth, the standard GATT service is
recorded, which links this taxonomy to the protocol facts mined in
``protocol/extractor.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple

# Aliased deliberately: this module defines its own BY_ID for sensors, and
# an unaliased import would be shadowed by it -- silently passing validation
# at construction time (before the rebinding) then failing at runtime.
from .signals import BY_ID as SIGNALS_BY_ID
from .signals import UnknownSignal


class Modality(str, Enum):
    """Physical measurement principle."""

    OPTICAL_PPG = "optical_ppg"
    OPTICAL_NIRS = "optical_nirs"
    ELECTRICAL_CARDIAC = "electrical_cardiac"
    ELECTRICAL_NEURAL = "electrical_neural"
    ELECTRODERMAL = "electrodermal"
    BIOIMPEDANCE = "bioimpedance"
    INERTIAL = "inertial"
    THERMAL = "thermal"
    ACOUSTIC = "acoustic"
    PRESSURE = "pressure"
    ELECTROCHEMICAL = "electrochemical"
    RESPIRATORY_MECHANICAL = "respiratory_mechanical"
    ENVIRONMENTAL = "environmental"
    SELF_REPORT = "self_report"


class Site(str, Enum):
    """Where the sensor sits."""

    WRIST = "wrist"
    FINGER = "finger"
    EAR = "ear"
    CHEST = "chest"
    TORSO_PATCH = "torso_patch"
    UPPER_ARM = "upper_arm"
    FOREHEAD = "forehead"
    ANKLE = "ankle"
    SCALP = "scalp"
    INGESTIBLE = "ingestible"
    ORAL = "oral"
    SKIN_GENERIC = "skin_generic"
    BEDSIDE = "bedside"
    HANDHELD = "handheld"
    AMBIENT = "ambient"
    NONE = "none"
    """Not a physical sensor -- self-report or an external record."""


class WearBurden(str, Enum):
    """How much the user has to tolerate. Compliance is a signal constraint:
    a sensor nobody wears produces no data, however good it is."""

    PASSIVE = "passive"
    """Worn continuously without thought (wrist, ring)."""

    TOLERATED = "tolerated"
    """Worn for hours but noticed (chest strap, patch)."""

    EPISODIC = "episodic"
    """Used deliberately for a reading (cuff, scale, handheld ECG)."""

    INVASIVE = "invasive"
    """Breaks the skin or is swallowed (CGM, ingestible capsule)."""


@dataclass(frozen=True, slots=True)
class Sensor:
    """One sensor: its modality, its site, and what it actually delivers."""

    id: str
    name: str
    modality: Modality
    sites: Tuple[Site, ...]
    delivers: Dict[str, float]
    """signal id -> delivered sampling rate in Hz. 0.0 means episodic."""
    wear_burden: WearBurden
    strengths: Tuple[str, ...] = field(default_factory=tuple)
    limitations: Tuple[str, ...] = field(default_factory=tuple)
    gatt_services: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        for signal_id in self.delivers:
            if signal_id not in SIGNALS_BY_ID:
                raise UnknownSignal(
                    f"sensor {self.id!r} declares unknown signal {signal_id!r}"
                )

    @property
    def signal_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self.delivers))

    def rate_for(self, signal_id: str) -> float | None:
        return self.delivers.get(signal_id)

    def meets_minimum(self, signal_id: str) -> bool | None:
        """Whether the delivered rate clears the signal's usable minimum.

        ``None`` when this sensor does not deliver the signal at all -- which
        is a different answer from "delivers it too slowly" and must not be
        collapsed into one.
        """
        delivered = self.delivers.get(signal_id)
        if delivered is None:
            return None
        required = SIGNALS_BY_ID[signal_id].min_sampling_hz
        return delivered >= required


def _sensor(*args, **kwargs) -> Sensor:
    return Sensor(*args, **kwargs)


SENSORS: Tuple[Sensor, ...] = (
    # --- Optical PPG, by site ----------------------------------------------
    _sensor(
        "ppg_wrist",
        "Wrist PPG",
        Modality.OPTICAL_PPG,
        (Site.WRIST,),
        {
            "heart_rate": 1.0,
            "pulse_interval": 4.0,
            "spo2": 0.017,
            "perfusion_index": 1.0,
        },
        WearBurden.PASSIVE,
        strengths=(
            "worn continuously with no effort",
            "high compliance",
            "cheap and mature",
        ),
        limitations=(
            "motion artifact dominates during activity",
            "poor perfusion at the wrist relative to the digit",
            "SpO2 typically reported once per minute or slower, which is below "
            "the rate any event detection requires",
            "accuracy degrades on darker skin and with tattoos",
        ),
        gatt_services=("0x180D",),
        notes="The default consumer form factor, and the weakest optical site. "
        "Adequate for trend; the basis of most contested claims.",
    ),
    _sensor(
        "ppg_finger",
        "Finger PPG (ring)",
        Modality.OPTICAL_PPG,
        (Site.FINGER,),
        {
            "heart_rate": 1.0,
            "pulse_interval": 8.0,
            "spo2": 0.2,
            "perfusion_index": 1.0,
            "ppg_waveform": 50.0,
        },
        WearBurden.PASSIVE,
        strengths=(
            "far better perfusion than the wrist",
            "much less motion during sleep",
            "supports genuine overnight SpO2 sampling",
        ),
        limitations=(
            "small battery constrains duty cycle",
            "sizing and finger swelling affect contact",
            "not robust during hand-loaded exercise",
        ),
        gatt_services=("0x180D", "0x1822"),
    ),
    _sensor(
        "ppg_ear",
        "In-ear PPG",
        Modality.OPTICAL_PPG,
        (Site.EAR,),
        {
            "heart_rate": 1.0,
            "pulse_interval": 8.0,
            "spo2": 0.2,
            "ppg_waveform": 100.0,
            "core_temperature": 0.017,
        },
        WearBurden.TOLERATED,
        strengths=(
            "close to the carotid circulation, so low motion artifact",
            "tympanic proximity makes core temperature plausible",
            "shares a form factor people already wear",
        ),
        limitations=(
            "worn only in sessions, not overnight",
            "fit variability between ears",
        ),
    ),
    _sensor(
        "ppg_forehead",
        "Forehead PPG",
        Modality.OPTICAL_PPG,
        (Site.FOREHEAD,),
        {"heart_rate": 1.0, "spo2": 1.0, "ppg_waveform": 100.0},
        WearBurden.TOLERATED,
        strengths=(
            "least motion-corrupted optical site",
            "clinical-grade SpO2 sampling rates",
        ),
        limitations=("cosmetically unacceptable for daily wear",),
    ),
    # --- Electrical cardiac -------------------------------------------------
    _sensor(
        "ecg_chest_strap",
        "Chest strap ECG",
        Modality.ELECTRICAL_CARDIAC,
        (Site.CHEST,),
        {
            "rr_interval": 4.0,
            "heart_rate": 1.0,
            "respiratory_rate": 4.0,
            "ectopic_burden": 4.0,
            "acceleration": 25.0,
        },
        WearBurden.TOLERATED,
        strengths=(
            "beat-accurate RR intervals -- the reference standard for HRV",
            "roughly an order of magnitude cleaner than wrist PPG",
            "unaffected by skin tone or peripheral perfusion",
            "exposes RR over standard GATT 0x2A37, so no proprietary protocol",
        ),
        limitations=(
            "worn in sessions, rarely overnight",
            "needs electrode moisture",
            "delivers intervals, not waveform, so no rhythm morphology",
        ),
        gatt_services=("0x180D",),
        notes="The single highest-leverage upgrade over wrist PPG for anything "
        "autonomic. Moves HRV work from contested to defensible.",
    ),
    _sensor(
        "ecg_patch",
        "Adhesive single-lead ECG patch",
        Modality.ELECTRICAL_CARDIAC,
        (Site.TORSO_PATCH,),
        {
            "ecg_waveform": 250.0,
            "rr_interval": 4.0,
            "heart_rate": 1.0,
            "qt_interval": 250.0,
            "ectopic_burden": 4.0,
            "respiratory_effort": 4.0,
            "acceleration": 25.0,
            "posture": 1.0,
        },
        WearBurden.TOLERATED,
        strengths=(
            "waveform morphology, so rhythm classification is founded",
            "multi-day continuous wear including sleep",
            "posture and respiratory effort from the same device",
        ),
        limitations=(
            "adhesive irritation limits wear duration",
            "consumable cost per wear",
            "single lead cannot localise ischaemia",
        ),
        notes="The form factor that makes cardiac rhythm claims supportable "
        "rather than hand-wavey.",
    ),
    _sensor(
        "ecg_handheld",
        "Handheld / finger-to-finger ECG",
        Modality.ELECTRICAL_CARDIAC,
        (Site.HANDHELD, Site.FINGER),
        {"ecg_waveform": 250.0, "rr_interval": 4.0},
        WearBurden.EPISODIC,
        strengths=("waveform on demand", "no adhesive or consumable"),
        limitations=(
            "30-second spot checks only",
            "user must initiate, so paroxysmal events are missed",
        ),
    ),
    # --- Neural -------------------------------------------------------------
    _sensor(
        "eeg_headband",
        "EEG headband",
        Modality.ELECTRICAL_NEURAL,
        (Site.SCALP, Site.FOREHEAD),
        {"eeg": 128.0, "sleep_stage": 0.033, "acceleration": 25.0},
        WearBurden.TOLERATED,
        strengths=(
            "the only signal that stages sleep definitively",
            "the reference against which RR-based staging is validated",
        ),
        limitations=(
            "low compliance for nightly wear",
            "electrode contact degrades through the night",
        ),
    ),
    # --- Respiratory --------------------------------------------------------
    _sensor(
        "resp_band",
        "Respiratory inductance band",
        Modality.RESPIRATORY_MECHANICAL,
        (Site.CHEST,),
        {"respiratory_effort": 4.0, "respiratory_rate": 4.0, "posture": 1.0},
        WearBurden.TOLERATED,
        strengths=(
            "direct effort measurement",
            "distinguishes obstructive from central events, which "
            "RSA-derived rate cannot",
        ),
        limitations=("band tension and placement sensitive",),
    ),
    _sensor(
        "airflow_cannula",
        "Nasal airflow cannula",
        Modality.PRESSURE,
        (Site.ORAL,),
        {"airflow": 10.0, "respiratory_rate": 4.0},
        WearBurden.TOLERATED,
        strengths=("the reference signal for scoring apneas and hypopneas",),
        limitations=("clinical sleep-study context only", "poor tolerability"),
    ),
    # --- Thermal ------------------------------------------------------------
    _sensor(
        "temp_skin",
        "Skin thermistor",
        Modality.THERMAL,
        (Site.WRIST, Site.FINGER, Site.TORSO_PATCH, Site.SKIN_GENERIC),
        {"skin_temperature": 0.017},
        WearBurden.PASSIVE,
        strengths=(
            "negligible power cost",
            "sensitive to circadian and " "ovulatory shifts when measured nightly",
        ),
        limitations=(
            "ambient temperature and bedding confound it",
            "not a proxy for core temperature",
        ),
        gatt_services=("0x1809",),
    ),
    _sensor(
        "temp_core_ingestible",
        "Ingestible core temperature capsule",
        Modality.THERMAL,
        (Site.INGESTIBLE,),
        {"core_temperature": 0.017},
        WearBurden.INVASIVE,
        strengths=("true core temperature, the reference for heat strain",),
        limitations=(
            "single use",
            "transit time limits wear window",
            "unsuitable for consumer daily use",
        ),
    ),
    _sensor(
        "temp_dual_site",
        "Dual-site skin thermometry",
        Modality.THERMAL,
        (Site.SKIN_GENERIC,),
        {"skin_temperature": 0.017, "distal_proximal_gradient": 0.017},
        WearBurden.TOLERATED,
        strengths=("gradient tracks circadian phase far better than one site",),
        limitations=("two devices or one device spanning two sites",),
    ),
    # --- Inertial -----------------------------------------------------------
    _sensor(
        "imu",
        "Inertial measurement unit",
        Modality.INERTIAL,
        (Site.WRIST, Site.FINGER, Site.CHEST, Site.TORSO_PATCH, Site.ANKLE),
        {"acceleration": 50.0, "actigraphy_counts": 0.033, "posture": 1.0},
        WearBurden.PASSIVE,
        strengths=(
            "negligible cost and power",
            "present in essentially every " "wearable already",
            "the classical sleep/wake input",
        ),
        limitations=(
            "cannot distinguish quiet wakefulness from sleep alone",
            "site changes what the signal means entirely",
        ),
    ),
    # --- Electrodermal ------------------------------------------------------
    _sensor(
        "eda",
        "Electrodermal activity sensor",
        Modality.ELECTRODERMAL,
        (Site.WRIST, Site.FINGER),
        {"electrodermal_activity": 4.0},
        WearBurden.PASSIVE,
        strengths=("direct sympathetic measure, complementing vagal HRV",),
        limitations=(
            "sweat, hydration and ambient temperature dominate",
            "sparse validation outside the laboratory",
        ),
    ),
    # --- Bioimpedance -------------------------------------------------------
    _sensor(
        "bia",
        "Bioelectrical impedance analyser",
        Modality.BIOIMPEDANCE,
        (Site.WRIST, Site.HANDHELD, Site.BEDSIDE),
        {"body_impedance": 0.0, "body_fat_percentage": 0.0, "respiratory_effort": 4.0},
        WearBurden.EPISODIC,
        strengths=("cheap body composition", "can also sense respiration"),
        limitations=(
            "hydration state swamps composition estimates",
            "population equations, not direct measurement",
        ),
    ),
    # --- Electrochemical ----------------------------------------------------
    _sensor(
        "cgm",
        "Continuous glucose monitor",
        Modality.ELECTROCHEMICAL,
        (Site.UPPER_ARM, Site.TORSO_PATCH),
        {"interstitial_glucose": 0.0033, "glucose_variability": 0.0033},
        WearBurden.INVASIVE,
        strengths=(
            "the only route to continuous metabolic data",
            "no cardiac or optical wearable signal reaches it",
            "high user engagement -- immediate, legible feedback",
        ),
        limitations=(
            "filament insertion",
            "10-14 day consumable",
            "5-15 minute lag behind blood glucose",
            "prescription-gated in some markets",
        ),
    ),
    _sensor(
        "sweat_patch",
        "Sweat analyte patch",
        Modality.ELECTROCHEMICAL,
        (Site.SKIN_GENERIC,),
        {"sweat_electrolytes": 0.017, "sweat_lactate": 0.017},
        WearBurden.TOLERATED,
        strengths=("hydration and effort markers without blood",),
        limitations=("needs active sweating", "early-stage validation"),
    ),
    # --- Pressure and mechanical -------------------------------------------
    _sensor(
        "bp_cuff",
        "Oscillometric blood pressure cuff",
        Modality.PRESSURE,
        (Site.UPPER_ARM,),
        {"blood_pressure": 0.0, "heart_rate": 0.0},
        WearBurden.EPISODIC,
        strengths=("the only validated route to blood pressure", "clinically accepted"),
        limitations=(
            "episodic and effortful",
            "no overnight profile without " "an ambulatory unit",
        ),
        gatt_services=("0x1810",),
    ),
    _sensor(
        "bcg_mat",
        "Under-mattress ballistocardiography",
        Modality.PRESSURE,
        (Site.BEDSIDE,),
        {
            "ballistocardiogram": 100.0,
            "heart_rate": 1.0,
            "pulse_interval": 4.0,
            "respiratory_rate": 4.0,
            "posture": 1.0,
        },
        WearBurden.PASSIVE,
        strengths=(
            "nothing worn at all, so compliance is total",
            "captures every night without user action",
        ),
        limitations=(
            "bed only",
            "confounded by a partner or pet",
            "intervals less precise than ECG",
        ),
    ),
    _sensor(
        "scale",
        "Smart scale",
        Modality.BIOIMPEDANCE,
        (Site.BEDSIDE,),
        {"body_mass": 0.0, "body_impedance": 0.0, "body_fat_percentage": 0.0},
        WearBurden.EPISODIC,
        strengths=("trivially high compliance", "mass trend is a real signal"),
        limitations=("daily variance is mostly fluid",),
    ),
    # --- Acoustic -----------------------------------------------------------
    _sensor(
        "mic_bedside",
        "Bedside microphone",
        Modality.ACOUSTIC,
        (Site.BEDSIDE,),
        {"snoring": 100.0, "cough_events": 100.0},
        WearBurden.PASSIVE,
        strengths=(
            "snoring and cough without contact",
            "cheap, and a phone already has one",
        ),
        limitations=(
            "privacy-sensitive by nature",
            "cannot attribute sound to a person in a shared room",
        ),
    ),
    # --- NIRS ---------------------------------------------------------------
    _sensor(
        "nirs",
        "Near-infrared muscle oximeter",
        Modality.OPTICAL_NIRS,
        (Site.SKIN_GENERIC,),
        {"smo2": 1.0},
        WearBurden.TOLERATED,
        strengths=("local muscle oxygenation, unavailable any other way",),
        limitations=(
            "adipose thickness attenuates it",
            "informative only for the muscle measured",
        ),
    ),
    # --- Environmental ------------------------------------------------------
    _sensor(
        "env_suite",
        "Ambient environment sensors",
        Modality.ENVIRONMENTAL,
        (Site.AMBIENT, Site.WRIST),
        {
            "ambient_light": 0.017,
            "ambient_temperature": 0.017,
            "altitude": 0.017,
            "air_quality": 0.017,
        },
        WearBurden.PASSIVE,
        strengths=(
            "light exposure is the dominant circadian input",
            "altitude is required to interpret SpO2 at all",
            "nearly free to add",
        ),
        limitations=("wrist light sensing is occluded by sleeves and bedding",),
    ),
    # --- Non-sensor inputs --------------------------------------------------
    _sensor(
        "self_report",
        "Structured self-report",
        Modality.SELF_REPORT,
        (Site.NONE,),
        {
            "subjective_report": 0.0,
            "medication": 0.0,
            "training_load": 0.0,
            "menstrual_phase": 0.017,
        },
        WearBurden.EPISODIC,
        strengths=(
            "often the strongest single predictor of outcome",
            "the only source of medication, alcohol and illness context",
            "costs nothing to collect",
        ),
        limitations=("compliance decays quickly", "recall bias"),
        notes="Systematically under-used by device-led products because it is "
        "not sensor data. Without medication and training context, a "
        "measured deviation cannot be attributed to a cause.",
    ),
)

BY_ID: Dict[str, Sensor] = {s.id: s for s in SENSORS}


class UnknownSensor(KeyError):
    """Raised when a sensor id is not in the taxonomy."""


def get(sensor_id: str) -> Sensor:
    try:
        return BY_ID[sensor_id]
    except KeyError as exc:
        raise UnknownSensor(f"unknown sensor id {sensor_id!r}") from exc


def by_modality(modality: Modality) -> List[Sensor]:
    return [s for s in SENSORS if s.modality is modality]


def by_site(site: Site) -> List[Sensor]:
    return [s for s in SENSORS if site in s.sites]


def by_burden(burden: WearBurden) -> List[Sensor]:
    return [s for s in SENSORS if s.wear_burden is burden]


def providers_of(signal_id: str, adequate_only: bool = False) -> List[Sensor]:
    """Sensors delivering a signal.

    With ``adequate_only``, restricted to those clearing the signal's usable
    minimum rate -- which is what separates "the spec sheet lists it" from
    "you can build on it".
    """
    out = [s for s in SENSORS if signal_id in s.delivers]
    if adequate_only:
        out = [s for s in out if s.meets_minimum(signal_id)]
    return out


def coverage(sensor_ids: Sequence[str]) -> Dict[str, float]:
    """Best delivered rate per signal across a set of sensors.

    Models a real product: several devices worn together, each contributing
    what it does best.
    """
    best: Dict[str, float] = {}
    for sensor_id in sensor_ids:
        for signal_id, rate in get(sensor_id).delivers.items():
            if signal_id not in best or rate > best[signal_id]:
                best[signal_id] = rate
    return dict(sorted(best.items()))
