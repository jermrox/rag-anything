"""The signal universe: every physiological quantity Vybe reasons about.

This is the shared vocabulary. Sensors *emit* signals, decision domains
*require* signals, and both sides reference the ids defined here so the two
halves of the knowledge base cannot drift apart.

The load-bearing field is :attr:`Signal.min_sampling_hz` -- the slowest rate at
which a signal still supports the inferences built on it. It is what turns
"could this device do X?" from an argument into a lookup: a signal sampled
below its minimum is not a weaker version of that signal, it is a different and
usually useless one. Sampling SpO2 once a minute does not give you a coarse
apnea screen; it gives you nothing, because a 45-second desaturation lands
between two samples.

Distinct from ``biometrics/schema.py``'s ``SignalType``, which enumerates only
what the store currently ingests. This module is the full space of what is
knowable; that one is what is presently captured. :data:`INGESTED_AS` maps
between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple


class SignalClass(str, Enum):
    """Physiological system a signal belongs to."""

    CARDIAC = "cardiac"
    CARDIAC_ELECTRICAL = "cardiac_electrical"
    HEMODYNAMIC = "hemodynamic"
    RESPIRATORY = "respiratory"
    OXYGENATION = "oxygenation"
    AUTONOMIC = "autonomic"
    THERMAL = "thermal"
    METABOLIC = "metabolic"
    BIOCHEMICAL = "biochemical"
    MOVEMENT = "movement"
    SLEEP = "sleep"
    BODY_COMPOSITION = "body_composition"
    ACOUSTIC = "acoustic"
    NEURAL = "neural"
    ENVIRONMENTAL = "environmental"
    CONTEXTUAL = "contextual"


class Derivation(str, Enum):
    """How directly a signal is obtained."""

    MEASURED = "measured"
    """Read off a sensor essentially as-is."""

    DERIVED = "derived"
    """Computed from one or more measured signals by a defined algorithm."""

    INFERRED = "inferred"
    """Estimated by a model. Carries irreducible uncertainty; the weakest
    footing on which to base a decision."""


@dataclass(frozen=True, slots=True)
class Signal:
    """One physiological quantity."""

    id: str
    name: str
    signal_class: SignalClass
    unit: str
    derivation: Derivation
    min_sampling_hz: float
    """Slowest rate that still supports what is built on this signal.
    0.0 means episodic -- a single reading is meaningful on its own."""
    derived_from: Tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    @property
    def is_continuous(self) -> bool:
        return self.min_sampling_hz > 0.0

    @property
    def min_interval_s(self) -> float | None:
        return 1.0 / self.min_sampling_hz if self.min_sampling_hz > 0 else None


def _s(*args, **kwargs) -> Signal:
    return Signal(*args, **kwargs)


#: The catalogue. Sampling minima are the rates below which the inferences
#: named in ``notes`` stop being supportable, not vendor marketing numbers.
SIGNALS: Tuple[Signal, ...] = (
    # --- Cardiac, optical ---------------------------------------------------
    _s(
        "heart_rate",
        "Heart rate",
        SignalClass.CARDIAC,
        "bpm",
        Derivation.MEASURED,
        1.0,
        notes="Adequate at 1 Hz for trend; 0.017 Hz (1/min) loses all "
        "short-term structure.",
    ),
    _s(
        "pulse_interval",
        "Pulse-to-pulse interval (PPG)",
        SignalClass.CARDIAC,
        "ms",
        Derivation.MEASURED,
        4.0,
        notes="PPG analogue of RR. Motion-corrupted at the wrist; acceptable at "
        "the finger and ear where perfusion is better and movement less.",
    ),
    _s(
        "perfusion_index",
        "Perfusion index",
        SignalClass.HEMODYNAMIC,
        "%",
        Derivation.DERIVED,
        1.0,
        ("ppg_waveform",),
        notes="Pulsatile over non-pulsatile optical absorption. A signal-quality "
        "gate as much as a physiological one.",
    ),
    _s(
        "ppg_waveform",
        "Raw PPG waveform",
        SignalClass.CARDIAC,
        "a.u.",
        Derivation.MEASURED,
        25.0,
        notes="25 Hz is the floor for morphology; 100 Hz+ for pulse-wave "
        "analysis. Most consumer devices never expose this.",
    ),
    # --- Cardiac, electrical ------------------------------------------------
    _s(
        "rr_interval",
        "RR interval (ECG)",
        SignalClass.CARDIAC_ELECTRICAL,
        "ms",
        Derivation.MEASURED,
        4.0,
        notes="Beat-accurate. The reference standard for all HRV. Exposed by "
        "standard GATT 0x2A37 in 1/1024 s units when flag bit 4 is set.",
    ),
    _s(
        "ecg_waveform",
        "ECG waveform",
        SignalClass.CARDIAC_ELECTRICAL,
        "mV",
        Derivation.MEASURED,
        250.0,
        notes="250 Hz minimum for rhythm classification; 500 Hz for morphology "
        "and interval measurement. Below this, rhythm claims are unfounded.",
    ),
    _s(
        "qt_interval",
        "QT interval",
        SignalClass.CARDIAC_ELECTRICAL,
        "ms",
        Derivation.DERIVED,
        250.0,
        ("ecg_waveform",),
        notes="Requires waveform morphology; not obtainable from RR intervals.",
    ),
    _s(
        "ectopic_burden",
        "Ectopic beat burden",
        SignalClass.CARDIAC_ELECTRICAL,
        "%",
        Derivation.DERIVED,
        4.0,
        ("rr_interval",),
        notes="Detectable from intervals alone; classifying the ectopy needs "
        "the waveform.",
    ),
    # --- HRV ----------------------------------------------------------------
    _s(
        "rmssd",
        "RMSSD",
        SignalClass.AUTONOMIC,
        "ms",
        Derivation.DERIVED,
        4.0,
        ("rr_interval",),
        notes="Short-term, parasympathetically mediated. The basis of most "
        "commercial recovery scores.",
    ),
    _s(
        "sdnn",
        "SDNN",
        SignalClass.AUTONOMIC,
        "ms",
        Derivation.DERIVED,
        4.0,
        ("rr_interval",),
        notes="Overall variability; window-length dependent.",
    ),
    _s(
        "pnn50",
        "pNN50",
        SignalClass.AUTONOMIC,
        "%",
        Derivation.DERIVED,
        4.0,
        ("rr_interval",),
    ),
    _s(
        "lf_hf_ratio",
        "LF/HF ratio",
        SignalClass.AUTONOMIC,
        "ratio",
        Derivation.DERIVED,
        4.0,
        ("rr_interval",),
        notes="Needs Lomb-Scargle on the unevenly sampled RR series; "
        "resample-then-FFT introduces artifacts. Interpretation as "
        "'sympathovagal balance' is contested.",
    ),
    _s(
        "hrv_baseline_deviation",
        "HRV deviation from personal baseline",
        SignalClass.AUTONOMIC,
        "z",
        Derivation.DERIVED,
        4.0,
        ("rmssd",),
        notes="Needs weeks of the same person's history. Population norms are "
        "not a substitute.",
    ),
    # --- Respiratory --------------------------------------------------------
    _s(
        "respiratory_rate",
        "Respiratory rate",
        SignalClass.RESPIRATORY,
        "breaths/min",
        Derivation.DERIVED,
        4.0,
        ("rr_interval",),
        notes="Recoverable from respiratory sinus arrhythmia at rest. Fails "
        "during movement and at high heart rates where RSA is blunted.",
    ),
    _s(
        "respiratory_effort",
        "Respiratory effort",
        SignalClass.RESPIRATORY,
        "a.u.",
        Derivation.MEASURED,
        4.0,
        notes="Inductance plethysmography or a strain band. Distinguishes "
        "obstructive from central events -- RSA-derived rate cannot.",
    ),
    _s(
        "airflow",
        "Nasal/oral airflow",
        SignalClass.RESPIRATORY,
        "a.u.",
        Derivation.MEASURED,
        10.0,
        notes="Thermistor or pressure cannula. The reference signal for scoring "
        "apneas and hypopneas.",
    ),
    _s(
        "snoring",
        "Snoring intensity",
        SignalClass.ACOUSTIC,
        "dB",
        Derivation.MEASURED,
        100.0,
        notes="Acoustic; needs audio-rate sampling. A screening adjunct, not a "
        "diagnostic signal.",
    ),
    _s(
        "cough_events",
        "Cough events",
        SignalClass.ACOUSTIC,
        "count",
        Derivation.INFERRED,
        100.0,
        ("snoring",),
    ),
    # --- Oxygenation --------------------------------------------------------
    _s(
        "spo2",
        "Peripheral oxygen saturation",
        SignalClass.OXYGENATION,
        "%",
        Derivation.MEASURED,
        0.2,
        notes="0.2 Hz (one sample per 5 s) is the floor for event detection: a "
        "45 s desaturation is missed entirely at 1/min. Trend-only use "
        "tolerates slower. Wrist SpO2 is markedly less accurate than "
        "finger, and both degrade on darker skin.",
    ),
    _s(
        "desaturation_index",
        "Oxygen desaturation index (ODI)",
        SignalClass.OXYGENATION,
        "events/h",
        Derivation.DERIVED,
        0.2,
        ("spo2",),
        notes="Dips of 3-4 points per hour of sleep. The closest a wearable "
        "gets to an apnea-hypopnea index, and not equivalent to it.",
    ),
    _s(
        "smo2",
        "Muscle oxygen saturation",
        SignalClass.OXYGENATION,
        "%",
        Derivation.MEASURED,
        1.0,
        notes="NIRS. Local to the muscle measured.",
    ),
    # --- Hemodynamic --------------------------------------------------------
    _s(
        "blood_pressure",
        "Blood pressure",
        SignalClass.HEMODYNAMIC,
        "mmHg",
        Derivation.MEASURED,
        0.0,
        notes="Episodic by cuff. Cuffless optical estimation drifts and needs "
        "frequent recalibration; treat as inferred, not measured.",
    ),
    _s(
        "pulse_transit_time",
        "Pulse transit time",
        SignalClass.HEMODYNAMIC,
        "ms",
        Derivation.DERIVED,
        100.0,
        ("ecg_waveform", "ppg_waveform"),
        notes="Needs two synchronised sites. The physiological basis of most "
        "cuffless blood-pressure claims.",
    ),
    _s(
        "stroke_volume",
        "Stroke volume",
        SignalClass.HEMODYNAMIC,
        "mL",
        Derivation.INFERRED,
        100.0,
        ("ppg_waveform",),
    ),
    _s(
        "ballistocardiogram",
        "Ballistocardiogram",
        SignalClass.HEMODYNAMIC,
        "a.u.",
        Derivation.MEASURED,
        100.0,
        notes="Bed- or chair-mounted force sensing. Contact-free cardiac timing.",
    ),
    # --- Thermal ------------------------------------------------------------
    _s(
        "skin_temperature",
        "Skin temperature",
        SignalClass.THERMAL,
        "degC",
        Derivation.MEASURED,
        0.017,
        notes="Once a minute suffices. Confounded by ambient temperature and "
        "bedding; nocturnal values are the usable ones.",
    ),
    _s(
        "core_temperature",
        "Core body temperature",
        SignalClass.THERMAL,
        "degC",
        Derivation.MEASURED,
        0.017,
        notes="Ingestible capsule or tympanic. Skin temperature is not a "
        "substitute -- the offset is large and situation-dependent.",
    ),
    _s(
        "distal_proximal_gradient",
        "Distal-proximal temperature gradient",
        SignalClass.THERMAL,
        "degC",
        Derivation.DERIVED,
        0.017,
        ("skin_temperature",),
        notes="Two sites required. Tracks circadian phase.",
    ),
    _s(
        "heat_strain_index",
        "Heat strain",
        SignalClass.THERMAL,
        "index",
        Derivation.INFERRED,
        0.017,
        ("core_temperature", "heart_rate"),
    ),
    # --- Movement and sleep -------------------------------------------------
    _s(
        "acceleration",
        "Tri-axial acceleration",
        SignalClass.MOVEMENT,
        "g",
        Derivation.MEASURED,
        25.0,
        notes="25 Hz for activity classification; 50-100 Hz for gait and tremor.",
    ),
    _s(
        "actigraphy_counts",
        "Activity counts",
        SignalClass.MOVEMENT,
        "counts/epoch",
        Derivation.DERIVED,
        0.033,
        ("acceleration",),
        notes="The classical sleep/wake input. One value per 30 s epoch.",
    ),
    _s(
        "posture",
        "Body position",
        SignalClass.MOVEMENT,
        "category",
        Derivation.DERIVED,
        1.0,
        ("acceleration",),
        notes="Supine position matters: positional apnea is a real phenotype.",
    ),
    _s(
        "sleep_stage",
        "Sleep stage",
        SignalClass.SLEEP,
        "category",
        Derivation.INFERRED,
        0.033,
        ("rr_interval", "actigraphy_counts"),
        notes="Four-class staging from RR plus actigraphy agrees with "
        "polysomnography at roughly 70-80%. Claims materially above that "
        "indicate a leaky evaluation.",
    ),
    _s(
        "sleep_onset_latency",
        "Sleep onset latency",
        SignalClass.SLEEP,
        "min",
        Derivation.DERIVED,
        0.033,
        ("sleep_stage",),
    ),
    _s(
        "wake_after_sleep_onset",
        "Wake after sleep onset",
        SignalClass.SLEEP,
        "min",
        Derivation.DERIVED,
        0.033,
        ("sleep_stage",),
    ),
    # --- Autonomic, non-cardiac --------------------------------------------
    _s(
        "electrodermal_activity",
        "Electrodermal activity",
        SignalClass.AUTONOMIC,
        "microsiemens",
        Derivation.MEASURED,
        4.0,
        notes="Sympathetic only, and strongly site- and hydration-dependent.",
    ),
    # --- Metabolic and biochemical -----------------------------------------
    _s(
        "interstitial_glucose",
        "Interstitial glucose",
        SignalClass.METABOLIC,
        "mg/dL",
        Derivation.MEASURED,
        0.0033,
        notes="CGM, typically one sample per 5 min, lagging blood by 5-15 min. "
        "Not derivable from any cardiac or optical wearable signal.",
    ),
    _s(
        "glucose_variability",
        "Glucose variability",
        SignalClass.METABOLIC,
        "%",
        Derivation.DERIVED,
        0.0033,
        ("interstitial_glucose",),
    ),
    _s(
        "energy_expenditure",
        "Energy expenditure",
        SignalClass.METABOLIC,
        "kcal",
        Derivation.INFERRED,
        0.017,
        ("heart_rate", "acceleration"),
        notes="Estimated, and inaccurate at the individual level without "
        "calorimetry calibration.",
    ),
    _s(
        "vo2max",
        "Estimated VO2 max",
        SignalClass.METABOLIC,
        "mL/kg/min",
        Derivation.INFERRED,
        1.0,
        ("heart_rate", "acceleration"),
        notes="Requires submaximal exercise with known external load.",
    ),
    _s(
        "sweat_electrolytes",
        "Sweat electrolytes",
        SignalClass.BIOCHEMICAL,
        "mmol/L",
        Derivation.MEASURED,
        0.017,
    ),
    _s(
        "sweat_lactate",
        "Sweat lactate",
        SignalClass.BIOCHEMICAL,
        "mmol/L",
        Derivation.MEASURED,
        0.017,
    ),
    _s(
        "cortisol",
        "Cortisol",
        SignalClass.BIOCHEMICAL,
        "nmol/L",
        Derivation.MEASURED,
        0.0,
        notes="Saliva or blood, episodic. No wearable measures it continuously.",
    ),
    # --- Body composition ---------------------------------------------------
    _s(
        "body_impedance",
        "Bioelectrical impedance",
        SignalClass.BODY_COMPOSITION,
        "ohm",
        Derivation.MEASURED,
        0.0,
    ),
    _s(
        "body_fat_percentage",
        "Body fat percentage",
        SignalClass.BODY_COMPOSITION,
        "%",
        Derivation.INFERRED,
        0.0,
        ("body_impedance",),
        notes="Population-equation estimate; hydration-sensitive.",
    ),
    _s(
        "body_mass",
        "Body mass",
        SignalClass.BODY_COMPOSITION,
        "kg",
        Derivation.MEASURED,
        0.0,
    ),
    # --- Neural -------------------------------------------------------------
    _s(
        "eeg",
        "Electroencephalogram",
        SignalClass.NEURAL,
        "microvolt",
        Derivation.MEASURED,
        128.0,
        notes="The only signal that stages sleep definitively. Headband form "
        "factors exist; wrist and ring devices have no access to it.",
    ),
    # --- Environmental and contextual --------------------------------------
    _s(
        "ambient_light",
        "Ambient light",
        SignalClass.ENVIRONMENTAL,
        "lux",
        Derivation.MEASURED,
        0.017,
        notes="Light exposure is the dominant zeitgeber; without it circadian "
        "inference is guesswork.",
    ),
    _s(
        "ambient_temperature",
        "Ambient temperature",
        SignalClass.ENVIRONMENTAL,
        "degC",
        Derivation.MEASURED,
        0.017,
    ),
    _s(
        "altitude",
        "Altitude",
        SignalClass.ENVIRONMENTAL,
        "m",
        Derivation.MEASURED,
        0.017,
        notes="Necessary to interpret SpO2 at all: 94% at 2500 m is normal.",
    ),
    _s(
        "air_quality",
        "Air quality (PM2.5)",
        SignalClass.ENVIRONMENTAL,
        "ug/m3",
        Derivation.MEASURED,
        0.017,
    ),
    _s(
        "training_load",
        "External training load",
        SignalClass.CONTEXTUAL,
        "a.u.",
        Derivation.MEASURED,
        0.0,
        notes="Power, pace or logged sessions. Recovery is uninterpretable "
        "without knowing what it is recovery from.",
    ),
    _s(
        "subjective_report",
        "Self-reported state",
        SignalClass.CONTEXTUAL,
        "scale",
        Derivation.MEASURED,
        0.0,
        notes="Cheap, and often the strongest single predictor. Under-used by "
        "device-led products because it is not sensor data.",
    ),
    _s(
        "medication",
        "Medication and substance intake",
        SignalClass.CONTEXTUAL,
        "category",
        Derivation.MEASURED,
        0.0,
        notes="Beta blockers, alcohol and stimulants all move HRV; without this "
        "context a deviation is unattributable.",
    ),
    _s(
        "menstrual_phase",
        "Menstrual cycle phase",
        SignalClass.CONTEXTUAL,
        "category",
        Derivation.INFERRED,
        0.017,
        ("skin_temperature", "rmssd"),
        notes="Cycle phase shifts HRV and temperature enough to swamp a "
        "recovery signal if unmodelled.",
    ),
)

BY_ID: Dict[str, Signal] = {s.id: s for s in SIGNALS}

#: Bridge to what the store currently ingests (``biometrics/schema.py``).
INGESTED_AS: Dict[str, str] = {
    "heart_rate": "heart_rate",
    "rr_interval": "rr_interval",
    "spo2": "spo2",
    "skin_temperature": "skin_temp",
    "acceleration": "accel_mag",
    "sleep_stage": "sleep_stage",
}


class UnknownSignal(KeyError):
    """Raised when a signal id is not in the catalogue."""


def get(signal_id: str) -> Signal:
    try:
        return BY_ID[signal_id]
    except KeyError as exc:
        raise UnknownSignal(f"unknown signal id {signal_id!r}") from exc


def by_class(signal_class: SignalClass) -> List[Signal]:
    return [s for s in SIGNALS if s.signal_class is signal_class]


def by_derivation(derivation: Derivation) -> List[Signal]:
    return [s for s in SIGNALS if s.derivation is derivation]


def dependencies(signal_id: str, _seen: set[str] | None = None) -> List[str]:
    """Transitive signals a derived signal depends on.

    A domain requiring ``respiratory_rate`` really requires ``rr_interval``,
    and the sensor question has to be answered at the root.
    """
    seen = _seen if _seen is not None else set()
    signal = get(signal_id)
    out: List[str] = []
    for parent in signal.derived_from:
        if parent in seen:
            continue
        seen.add(parent)
        out.append(parent)
        out.extend(dependencies(parent, seen))
    return out


def root_signals(signal_ids: Sequence[str]) -> List[str]:
    """The measured signals ultimately needed for a set of requirements."""
    roots: set[str] = set()
    for signal_id in signal_ids:
        signal = get(signal_id)
        if signal.derivation is Derivation.MEASURED:
            roots.add(signal.id)
        for parent in dependencies(signal_id):
            if get(parent).derivation is Derivation.MEASURED:
                roots.add(parent)
    return sorted(roots)
