"""Decision domains: what Vybe is trying to answer, and what each answer needs.

Each domain is a self-contained area of inference with its own decisions, its
own required signals, and **its own RAG namespace**. Domains are partitioned
rather than pooled deliberately: the literature, protocol knowledge and
personal history relevant to sleep-disordered breathing has almost no overlap
with what is relevant to glycaemic response, and mixing them into one corpus
degrades retrieval for both. A query about apnea should never have to compete
with cardiology papers for context space.

Every :class:`Decision` names the signals it requires by id, the minimum
history it needs, and the strength of the evidence behind it. Requirements are
expressed against ``signals.py`` ids, so the sensor question is answerable by
composition rather than by judgement.

Evidence grades follow GRADE conventions. They describe the *literature*
supporting the inference, not the confidence of any particular output: a
HIGH-grade decision computed from an inadequate sensor is still unsupported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

from .signals import BY_ID as SIGNALS_BY_ID
from .signals import UnknownSignal, root_signals


class Evidence(str, Enum):
    """GRADE-style strength of the supporting literature."""

    HIGH = "high"
    """Consistent findings from controlled studies or meta-analysis."""

    MODERATE = "moderate"
    """Reproduced in several independent cohorts; effect sizes vary."""

    LOW = "low"
    """Plausible mechanism, limited or conflicting human evidence."""

    VERY_LOW = "very_low"
    """Mechanistic speculation or single small studies. Marketed far more
    confidently than it is evidenced."""


@dataclass(frozen=True, slots=True)
class Decision:
    """One question the product tries to answer."""

    id: str
    question: str
    requires: Tuple[str, ...]
    """Signal ids that must be present and adequate."""
    helpful: Tuple[str, ...] = field(default_factory=tuple)
    """Signals that materially improve the answer without being required."""
    min_history_days: int = 0
    evidence: Evidence = Evidence.LOW
    notes: str = ""

    def __post_init__(self) -> None:
        for signal_id in self.requires + self.helpful:
            if signal_id not in SIGNALS_BY_ID:
                raise UnknownSignal(
                    f"decision {self.id!r} references unknown signal {signal_id!r}"
                )

    @property
    def required_roots(self) -> List[str]:
        """Measured signals ultimately needed -- the sensor question."""
        return root_signals(self.requires)


@dataclass(frozen=True, slots=True)
class Domain:
    """One area of inference, with its own corpus."""

    id: str
    name: str
    summary: str
    decisions: Tuple[Decision, ...]
    corpus_topics: Tuple[str, ...] = field(default_factory=tuple)
    """What literature and protocol material this domain's RAG should hold."""
    notes: str = ""

    @property
    def namespace(self) -> str:
        """RAG namespace. Each domain retrieves only over its own corpus."""
        return f"vybe/{self.id}"

    @property
    def required_signals(self) -> List[str]:
        out: set[str] = set()
        for decision in self.decisions:
            out.update(decision.requires)
        return sorted(out)

    @property
    def required_roots(self) -> List[str]:
        return root_signals(self.required_signals)

    @property
    def best_evidence(self) -> Evidence:
        order = [Evidence.HIGH, Evidence.MODERATE, Evidence.LOW, Evidence.VERY_LOW]
        for grade in order:
            if any(d.evidence is grade for d in self.decisions):
                return grade
        return Evidence.VERY_LOW

    def decision(self, decision_id: str) -> Decision:
        for d in self.decisions:
            if d.id == decision_id:
                return d
        raise KeyError(f"domain {self.id!r} has no decision {decision_id!r}")


def _d(*args, **kwargs) -> Decision:
    return Decision(*args, **kwargs)


DOMAINS: Tuple[Domain, ...] = (
    Domain(
        "training_readiness",
        "Training readiness and recovery",
        "Whether the body is prepared for training load today, and how much.",
        (
            _d(
                "readiness_today",
                "Am I recovered enough to train hard today?",
                ("rmssd", "heart_rate", "sleep_stage"),
                ("training_load", "subjective_report", "respiratory_rate"),
                min_history_days=14,
                evidence=Evidence.MODERATE,
                notes="Requires clean overnight RR. Uninterpretable without "
                "knowing what is being recovered from, which is why "
                "training_load is listed as helpful and in practice is "
                "close to required.",
            ),
            _d(
                "acute_chronic_load",
                "Is my load ramping faster than I adapt?",
                ("training_load",),
                ("rmssd", "heart_rate"),
                min_history_days=28,
                evidence=Evidence.MODERATE,
            ),
            _d(
                "overreaching",
                "Am I drifting into non-functional overreaching?",
                ("rmssd", "heart_rate", "training_load"),
                ("subjective_report", "body_mass"),
                min_history_days=42,
                evidence=Evidence.LOW,
                notes="Distinguishing functional from non-functional "
                "overreaching is not established from wearable data alone.",
            ),
        ),
        corpus_topics=(
            "HRV-guided training",
            "acute:chronic workload ratio",
            "overtraining syndrome",
            "autonomic recovery kinetics",
        ),
        notes="The most crowded category. Whoop, Oura and Garmin all ship a "
        "readiness number; differentiation here is unlikely to come from "
        "the metric itself.",
    ),
    Domain(
        "illness_onset",
        "Illness onset and infection detection",
        "Pre-symptomatic detection of infection from multi-signal deviation.",
        (
            _d(
                "presymptomatic_infection",
                "Am I becoming ill before I feel it?",
                ("rmssd", "heart_rate", "skin_temperature"),
                ("respiratory_rate", "spo2", "subjective_report"),
                min_history_days=21,
                evidence=Evidence.MODERATE,
                notes="Resting HR elevation with HRV suppression and a "
                "temperature rise is reproducible across cohorts and "
                "typically precedes symptoms by 1-3 days. Multi-signal "
                "agreement is what makes it defensible; any single "
                "channel produces too many false positives.",
            ),
            _d(
                "recovery_from_illness",
                "Have I returned to baseline?",
                ("rmssd", "heart_rate"),
                ("skin_temperature", "respiratory_rate"),
                min_history_days=21,
                evidence=Evidence.MODERATE,
            ),
        ),
        corpus_topics=(
            "pre-symptomatic infection detection",
            "resting heart " "rate elevation and infection",
            "wearable fever " "detection",
            "COVID and influenza wearable cohorts",
        ),
        notes="Materially easier to defend than a readiness score: the outcome "
        "is objective and the effect sizes are large.",
    ),
    Domain(
        "sleep_architecture",
        "Sleep architecture and quality",
        "How long, how well, and in what stages a person slept.",
        (
            _d(
                "sleep_staging",
                "What were my sleep stages?",
                ("sleep_stage",),
                ("eeg", "acceleration", "rr_interval"),
                evidence=Evidence.MODERATE,
                notes="RR plus actigraphy agrees with polysomnography at "
                "roughly 70-80% for four classes. EEG is the only "
                "definitive route; claims well above that range indicate "
                "a leaky evaluation rather than a better model.",
            ),
            _d(
                "sleep_efficiency",
                "How much of my time in bed was sleep?",
                ("sleep_stage",),
                ("actigraphy_counts",),
                evidence=Evidence.HIGH,
            ),
            _d(
                "circadian_alignment",
                "Is my sleep timing aligned to my clock?",
                ("sleep_stage", "ambient_light"),
                ("distal_proximal_gradient", "skin_temperature"),
                min_history_days=14,
                evidence=Evidence.MODERATE,
                notes="Light exposure is the dominant zeitgeber; without it "
                "circadian inference is guesswork.",
            ),
        ),
        corpus_topics=(
            "polysomnography scoring criteria",
            "actigraphy " "validation",
            "HRV-based sleep staging",
            "circadian " "phase markers",
            "light exposure and entrainment",
        ),
    ),
    Domain(
        "sleep_disordered_breathing",
        "Sleep-disordered breathing",
        "Screening for apnea and hypopnea from overnight respiratory signals.",
        (
            _d(
                "desaturation_burden",
                "How often does my oxygen dip at night?",
                ("spo2", "desaturation_index"),
                ("posture", "snoring"),
                evidence=Evidence.MODERATE,
                notes="Requires SpO2 at 0.2 Hz or faster. Wrist SpO2 reported "
                "once per minute cannot support this at all -- a 45 "
                "second event falls between samples.",
            ),
            _d(
                "apnea_screening",
                "Should I be evaluated for sleep apnea?",
                ("spo2", "desaturation_index", "respiratory_effort"),
                ("airflow", "snoring", "posture", "body_mass"),
                evidence=Evidence.MODERATE,
                notes="ODI is not AHI. Distinguishing obstructive from central "
                "events needs respiratory effort, which RSA-derived rate "
                "cannot provide.",
            ),
            _d(
                "positional_component",
                "Is my breathing worse on my back?",
                ("posture", "desaturation_index"),
                (),
                evidence=Evidence.MODERATE,
                notes="Positional apnea is a real phenotype and actionable "
                "without a clinic visit.",
            ),
        ),
        corpus_topics=(
            "AASM scoring rules",
            "oxygen desaturation index",
            "home sleep apnea testing",
            "positional obstructive " "apnea",
            "pulse oximetry accuracy and skin tone",
        ),
        notes="The clearest case where wrist-only hardware cannot reach the "
        "decision, whatever the software does.",
    ),
    Domain(
        "cardiac_rhythm",
        "Cardiac rhythm and conduction",
        "Detection and characterisation of arrhythmia.",
        (
            _d(
                "irregularity_screening",
                "Is my rhythm irregular?",
                ("rr_interval", "ectopic_burden"),
                ("ppg_waveform",),
                evidence=Evidence.MODERATE,
                notes="Interval-based irregularity screening is established. "
                "It flags irregularity, not a named arrhythmia.",
            ),
            _d(
                "afib_detection",
                "Do I have atrial fibrillation?",
                ("ecg_waveform", "rr_interval"),
                ("ectopic_burden",),
                evidence=Evidence.HIGH,
                notes="Requires waveform morphology at 250 Hz or better. "
                "PPG-only AF claims are where consumer wearables attract "
                "regulatory attention.",
            ),
            _d(
                "qt_monitoring",
                "Is my QT interval prolonging?",
                ("qt_interval", "ecg_waveform"),
                ("medication",),
                evidence=Evidence.HIGH,
                notes="Drug-induced QT prolongation. Needs morphology and "
                "medication context together.",
            ),
        ),
        corpus_topics=(
            "atrial fibrillation detection algorithms",
            "photoplethysmography rhythm limitations",
            "single-lead ECG validation",
            "drug-induced QT " "prolongation",
        ),
    ),
    Domain(
        "cardiorespiratory_fitness",
        "Cardiorespiratory fitness",
        "Aerobic capacity and its change over time.",
        (
            _d(
                "vo2max_estimate",
                "What is my aerobic capacity?",
                ("vo2max", "heart_rate"),
                ("acceleration", "training_load"),
                min_history_days=28,
                evidence=Evidence.MODERATE,
                notes="Needs submaximal exercise at a known external load. "
                "Estimates without load are unreliable.",
            ),
            _d(
                "hr_recovery",
                "How fast does my heart rate recover after effort?",
                ("heart_rate",),
                ("rr_interval",),
                evidence=Evidence.HIGH,
                notes="One-minute heart-rate recovery is a strong, cheap "
                "prognostic marker and needs no exotic sensing.",
            ),
            _d(
                "efficiency_drift",
                "Is my cardiac cost at a given pace rising?",
                ("heart_rate", "training_load"),
                (),
                min_history_days=28,
                evidence=Evidence.MODERATE,
            ),
        ),
        corpus_topics=(
            "VO2max estimation from wearables",
            "heart rate " "recovery prognosis",
            "aerobic decoupling",
        ),
    ),
    Domain(
        "blood_pressure_hemodynamics",
        "Blood pressure and hemodynamics",
        "Arterial pressure and vascular function.",
        (
            _d(
                "bp_trend",
                "Is my blood pressure trending up?",
                ("blood_pressure",),
                ("heart_rate", "body_mass"),
                min_history_days=28,
                evidence=Evidence.HIGH,
                notes="Cuff measurement only. Cuffless optical estimation "
                "drifts and requires frequent recalibration.",
            ),
            _d(
                "nocturnal_dipping",
                "Does my pressure fall overnight as it should?",
                ("blood_pressure", "sleep_stage"),
                (),
                evidence=Evidence.HIGH,
                notes="Non-dipping is prognostically important and needs "
                "ambulatory overnight measurement, which consumer "
                "wearables do not provide.",
            ),
            _d(
                "arterial_stiffness",
                "How stiff are my arteries?",
                ("pulse_transit_time", "ppg_waveform"),
                ("blood_pressure",),
                evidence=Evidence.LOW,
                notes="Needs two synchronised measurement sites.",
            ),
        ),
        corpus_topics=(
            "cuffless blood pressure validation",
            "nocturnal blood " "pressure dipping",
            "pulse transit time",
            "arterial " "stiffness indices",
        ),
    ),
    Domain(
        "metabolic_glycemic",
        "Metabolic and glycaemic response",
        "Glucose handling, insulin sensitivity and metabolic flexibility.",
        (
            _d(
                "meal_response",
                "How did my body handle that meal?",
                ("interstitial_glucose",),
                ("acceleration", "heart_rate"),
                evidence=Evidence.HIGH,
                notes="Requires CGM. Not derivable from any cardiac or optical "
                "wearable signal, at any sampling rate.",
            ),
            _d(
                "glycemic_variability",
                "How stable is my glucose?",
                ("glucose_variability", "interstitial_glucose"),
                (),
                min_history_days=14,
                evidence=Evidence.HIGH,
            ),
            _d(
                "exercise_glucose_coupling",
                "How does training change my glucose response?",
                ("interstitial_glucose", "training_load"),
                ("heart_rate",),
                min_history_days=28,
                evidence=Evidence.MODERATE,
            ),
        ),
        corpus_topics=(
            "continuous glucose monitoring in non-diabetics",
            "glycaemic variability metrics",
            "exercise and insulin " "sensitivity",
            "postprandial response personalisation",
        ),
        notes="Structurally unreachable from wrist hardware. Either a CGM is "
        "in the stack or this domain is out of scope.",
    ),
    Domain(
        "stress_autonomic",
        "Stress and autonomic load",
        "Sympathetic and parasympathetic balance through the day.",
        (
            _d(
                "acute_stress",
                "Am I under acute stress right now?",
                ("rmssd", "heart_rate"),
                ("electrodermal_activity", "subjective_report"),
                evidence=Evidence.MODERATE,
                notes="Daytime HRV is heavily movement-confounded; posture and "
                "activity must be controlled for or the reading is noise.",
            ),
            _d(
                "autonomic_load",
                "How much cumulative strain am I carrying?",
                ("rmssd", "heart_rate"),
                ("electrodermal_activity", "sleep_stage"),
                min_history_days=14,
                evidence=Evidence.LOW,
            ),
        ),
        corpus_topics=(
            "HRV and psychological stress",
            "electrodermal activity " "validation",
            "allostatic load",
        ),
        notes="Popular and weakly evidenced. LF/HF as 'sympathovagal balance' "
        "is contested in the literature it is usually cited from.",
    ),
    Domain(
        "thermoregulation",
        "Thermoregulation and heat strain",
        "Thermal load, acclimatisation and heat risk.",
        (
            _d(
                "heat_strain",
                "Am I approaching dangerous heat strain?",
                ("core_temperature", "heart_rate"),
                ("ambient_temperature", "sweat_electrolytes"),
                evidence=Evidence.HIGH,
                notes="Needs true core temperature. Skin temperature is not a "
                "substitute -- the offset is large and situational.",
            ),
            _d(
                "acclimatisation",
                "Am I adapting to the heat?",
                ("core_temperature", "heart_rate", "ambient_temperature"),
                ("sweat_electrolytes",),
                min_history_days=14,
                evidence=Evidence.MODERATE,
            ),
        ),
        corpus_topics=(
            "heat strain index",
            "core temperature estimation",
            "heat acclimatisation protocols",
        ),
    ),
    Domain(
        "menstrual_hormonal",
        "Menstrual cycle and hormonal state",
        "Cycle phase, ovulation and hormonal effects on other signals.",
        (
            _d(
                "phase_tracking",
                "Where am I in my cycle?",
                ("skin_temperature", "menstrual_phase"),
                ("rmssd", "heart_rate"),
                min_history_days=60,
                evidence=Evidence.MODERATE,
                notes="Nocturnal skin temperature shift is the usable signal. "
                "Needs two or more cycles of history.",
            ),
            _d(
                "ovulation_window",
                "Am I ovulating?",
                ("skin_temperature",),
                ("core_temperature", "rmssd"),
                min_history_days=60,
                evidence=Evidence.MODERATE,
            ),
            _d(
                "cycle_confounding",
                "Is my cycle phase explaining this HRV change?",
                ("menstrual_phase", "rmssd"),
                (),
                min_history_days=60,
                evidence=Evidence.MODERATE,
                notes="Cycle phase shifts HRV and temperature enough to swamp a "
                "recovery signal. Unmodelled, it makes readiness scores "
                "systematically wrong for half the population.",
            ),
        ),
        corpus_topics=(
            "basal body temperature and ovulation",
            "menstrual " "cycle effects on HRV",
            "cycle-aware training",
        ),
        notes="An underserved differentiator: most recovery products treat "
        "cycle phase as noise rather than as a modelled covariate.",
    ),
    Domain(
        "hydration_electrolytes",
        "Hydration and electrolyte status",
        "Fluid balance and electrolyte loss.",
        (
            _d(
                "hydration_status",
                "Am I dehydrated?",
                ("body_mass", "sweat_electrolytes"),
                ("body_impedance", "heart_rate", "ambient_temperature"),
                evidence=Evidence.LOW,
                notes="Acute body-mass change is the most reliable field proxy. "
                "Bioimpedance hydration claims are weakly supported.",
            ),
            _d(
                "sweat_sodium_loss",
                "How much sodium am I losing?",
                ("sweat_electrolytes",),
                ("ambient_temperature", "training_load"),
                evidence=Evidence.MODERATE,
            ),
        ),
        corpus_topics=(
            "hydration assessment methods",
            "sweat sodium " "concentration variability",
            "bioimpedance hydration " "validity",
        ),
    ),
    Domain(
        "respiratory_health",
        "Respiratory health",
        "Airway and pulmonary status outside sleep.",
        (
            _d(
                "respiratory_infection",
                "Is my breathing pattern changing with illness?",
                ("respiratory_rate",),
                ("cough_events", "spo2", "skin_temperature"),
                min_history_days=21,
                evidence=Evidence.MODERATE,
            ),
            _d(
                "exercise_breathing",
                "Is my breathing efficiency changing?",
                ("respiratory_rate", "training_load"),
                ("smo2",),
                min_history_days=28,
                evidence=Evidence.LOW,
            ),
        ),
        corpus_topics=(
            "respiratory rate as a vital sign",
            "cough detection " "from audio",
            "wearable respiratory monitoring",
        ),
    ),
    Domain(
        "mental_health_mood",
        "Mood and mental health",
        "Affective state and its physiological correlates.",
        (
            _d(
                "mood_correlates",
                "Do my physiological signals track my mood?",
                ("subjective_report", "rmssd", "sleep_stage"),
                ("electrodermal_activity", "acceleration"),
                min_history_days=28,
                evidence=Evidence.LOW,
                notes="Self-report is the ground truth here, not an adjunct. "
                "Physiology-only mood inference is not supported.",
            ),
            _d(
                "behavioural_activation",
                "Has my activity and social rhythm changed?",
                ("acceleration", "sleep_stage"),
                ("subjective_report",),
                min_history_days=28,
                evidence=Evidence.MODERATE,
                notes="Circadian and activity disruption are reproducible "
                "correlates of depressive episodes.",
            ),
        ),
        corpus_topics=(
            "digital phenotyping",
            "actigraphy and depression",
            "HRV and affective disorders",
        ),
        notes="High potential for harm if overclaimed. Self-report must remain "
        "the anchor.",
    ),
    Domain(
        "musculoskeletal_load",
        "Musculoskeletal load and injury risk",
        "Mechanical loading, asymmetry and injury exposure.",
        (
            _d(
                "impact_load",
                "How much mechanical load did I absorb?",
                ("acceleration",),
                ("training_load", "posture"),
                evidence=Evidence.MODERATE,
                notes="Sensor site changes the meaning entirely: wrist "
                "acceleration is a poor proxy for ground reaction force.",
            ),
            _d(
                "asymmetry",
                "Am I loading one side more than the other?",
                ("acceleration",),
                (),
                evidence=Evidence.LOW,
                notes="Requires bilateral sensing; a single wrist cannot "
                "detect asymmetry in principle.",
            ),
        ),
        corpus_topics=(
            "accelerometry and impact load",
            "gait asymmetry",
            "training load and injury incidence",
        ),
    ),
    Domain(
        "cognitive_alertness",
        "Cognitive performance and alertness",
        "Vigilance, sleep pressure and time-of-day effects.",
        (
            _d(
                "sleep_pressure",
                "How impaired is my alertness right now?",
                ("sleep_stage", "sleep_onset_latency"),
                ("ambient_light", "subjective_report"),
                min_history_days=14,
                evidence=Evidence.MODERATE,
                notes="Sleep debt and circadian phase together predict "
                "vigilance well; both are needed.",
            ),
            _d(
                "optimal_window",
                "When should I do demanding work?",
                ("sleep_stage", "ambient_light"),
                ("distal_proximal_gradient",),
                min_history_days=21,
                evidence=Evidence.LOW,
            ),
        ),
        corpus_topics=(
            "two-process model of sleep regulation",
            "psychomotor vigilance and sleep debt",
            "circadian performance rhythms",
        ),
    ),
    Domain(
        "body_composition",
        "Body composition",
        "Mass, fat, lean tissue and their trends.",
        (
            _d(
                "composition_trend",
                "Is my body composition changing?",
                ("body_mass", "body_fat_percentage"),
                ("body_impedance",),
                min_history_days=28,
                evidence=Evidence.MODERATE,
                notes="Trend is meaningful; single readings are dominated by "
                "hydration.",
            ),
        ),
        corpus_topics=("bioimpedance validity", "body composition tracking " "methods"),
    ),
    Domain(
        "medication_response",
        "Medication and substance response",
        "How drugs, alcohol and stimulants move the other signals.",
        (
            _d(
                "substance_effect",
                "How is alcohol affecting my recovery?",
                ("medication", "rmssd", "heart_rate"),
                ("sleep_stage", "skin_temperature"),
                min_history_days=28,
                evidence=Evidence.MODERATE,
                notes="Alcohol's effect on nocturnal HRV and heart rate is "
                "large and reproducible -- one of the clearest "
                "signal-to-behaviour links available.",
            ),
            _d(
                "beta_blocker_confound",
                "Is my medication invalidating my HRV metrics?",
                ("medication", "heart_rate", "rmssd"),
                (),
                evidence=Evidence.HIGH,
                notes="Rate-control drugs make heart-rate-based scores "
                "uninterpretable. Without medication context the product "
                "will confidently mislead these users.",
            ),
        ),
        corpus_topics=(
            "alcohol and heart rate variability",
            "beta blockade " "and HRV",
            "stimulants and autonomic function",
        ),
        notes="Cheap to collect, high explanatory power, and almost universally "
        "ignored by device-led products.",
    ),
    Domain(
        "longevity_aging",
        "Longevity and biological ageing",
        "Long-horizon markers of physiological ageing.",
        (
            _d(
                "fitness_age",
                "How does my fitness compare to my age cohort?",
                ("vo2max",),
                ("heart_rate", "body_fat_percentage"),
                min_history_days=90,
                evidence=Evidence.MODERATE,
                notes="VO2max is among the strongest all-cause mortality "
                "predictors available from non-invasive measurement.",
            ),
            _d(
                "autonomic_age",
                "Is my autonomic function ageing well?",
                ("rmssd", "sdnn"),
                ("heart_rate",),
                min_history_days=180,
                evidence=Evidence.LOW,
                notes="HRV declines with age at the population level, but "
                "individual 'HRV age' figures are marketing, not "
                "measurement.",
            ),
        ),
        corpus_topics=(
            "cardiorespiratory fitness and mortality",
            "HRV and " "ageing",
            "biological age estimators",
        ),
    ),
    Domain(
        "environmental_exposure",
        "Environmental exposure",
        "Light, air, altitude and thermal environment as inputs to health.",
        (
            _d(
                "light_exposure",
                "Am I getting the light I need, when I need it?",
                ("ambient_light",),
                ("sleep_stage",),
                min_history_days=14,
                evidence=Evidence.HIGH,
                notes="Morning bright light and evening darkness are the most "
                "actionable circadian levers there are, and require only "
                "a cheap sensor.",
            ),
            _d(
                "air_quality_impact",
                "Is air quality affecting my breathing?",
                ("air_quality", "respiratory_rate"),
                ("cough_events", "spo2"),
                min_history_days=28,
                evidence=Evidence.MODERATE,
            ),
            _d(
                "altitude_adjustment",
                "Is my SpO2 normal for this altitude?",
                ("altitude", "spo2"),
                (),
                evidence=Evidence.HIGH,
                notes="Without altitude, an SpO2 reading cannot be interpreted "
                "at all: 94% at 2500 m is unremarkable.",
            ),
        ),
        corpus_topics=(
            "light exposure and circadian entrainment",
            "air " "pollution and respiratory outcomes",
            "altitude and " "oxygen saturation norms",
        ),
        notes="The cheapest sensors in the taxonomy and among the most "
        "actionable outputs. Routinely omitted.",
    ),
)

BY_ID: Dict[str, Domain] = {d.id: d for d in DOMAINS}


class UnknownDomain(KeyError):
    """Raised when a domain id is not registered."""


def get(domain_id: str) -> Domain:
    try:
        return BY_ID[domain_id]
    except KeyError as exc:
        raise UnknownDomain(f"unknown domain id {domain_id!r}") from exc


def namespaces() -> List[str]:
    """Every RAG namespace, one per domain."""
    return [d.namespace for d in DOMAINS]


def all_decisions() -> List[Tuple[str, Decision]]:
    return [(d.id, decision) for d in DOMAINS for decision in d.decisions]


def domains_requiring(signal_id: str) -> List[Domain]:
    """Which domains depend on a signal -- the blast radius of a sensor choice."""
    return [d for d in DOMAINS if signal_id in d.required_signals]


def by_evidence(grade: Evidence) -> List[Domain]:
    return [d for d in DOMAINS if d.best_evidence is grade]


def signal_demand() -> Dict[str, int]:
    """How many decisions require each signal.

    Reads as a sensor-priority list: the signals many decisions depend on are
    the ones worth spending hardware budget and board space on.
    """
    counts: Dict[str, int] = {}
    for _, decision in all_decisions():
        for signal_id in decision.requires:
            counts[signal_id] = counts.get(signal_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def root_signal_demand() -> Dict[str, int]:
    """The same tally resolved to measured signals -- the actual sensor question."""
    counts: Dict[str, int] = {}
    for _, decision in all_decisions():
        for signal_id in decision.required_roots:
            counts[signal_id] = counts.get(signal_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def summary() -> Dict[str, object]:
    return {
        "domains": len(DOMAINS),
        "decisions": len(all_decisions()),
        "namespaces": namespaces(),
        "by_evidence": {
            grade.value: [d.id for d in by_evidence(grade)] for grade in Evidence
        },
    }
