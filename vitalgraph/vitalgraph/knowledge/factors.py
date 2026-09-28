"""The Five Factor model, Medical Context, and the evidence tiers.

Five factors organise what a person's health *is*. Medical context is not a
sixth: labs, medications, diagnoses and blood pressure do not sit alongside
sleep and movement, they change how sleep and movement should be read. A
hypertensive reading reframes a training recommendation; an SSRI reframes an
HRV trend; a thyroid result reframes a temperature trend. Modelling that as a
layer beneath all five, rather than as a peer of them, is what lets an
interpretation say *why* it was adjusted.

The other half of this module is the evidence tier. Every value the system
holds is tagged with how it came to be known, and tiers never silently merge:

    Measured     an instrument reported it
    Derived      arithmetic on measurements, with no claim about the world
    Observed     a relationship seen in this person's data, once
    Validated    a relationship that held on data it was not found in
    Interpreted  a language model's reading of the above

"Weight: 81.2 kg" and "your training appears productive" are both things the
system can say, and the difference between them is the entire product. A
blended score destroys that difference, which is why there is no single number
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Sequence, Tuple


class Factor(str, Enum):
    """The five dimensions health is organised into."""

    SLEEP = "sleep"
    MOVEMENT = "movement"
    NUTRITION = "nutrition"
    MIND = "mind"
    CONNECTION = "connection"


class EvidenceTier(str, Enum):
    """How a statement came to be known. Ordered from strongest to weakest
    claim about the world, which is *not* the same as most to least certain --
    a measurement is certain about a number, an interpretation is uncertain
    about a meaning, and they are not comparable on one axis."""

    MEASURED = "measured"
    """An instrument reported it. Carries the instrument, not just the value."""

    DERIVED = "derived"
    """Arithmetic on measurements. Asserts nothing new about the world."""

    OBSERVED = "observed"
    """A relationship seen in this person's data. Not yet evidence."""

    VALIDATED = "validated"
    """An observed relationship that held on data it was not discovered in."""

    INTERPRETED = "interpreted"
    """A language model's reading. Never a substitute for the tiers above."""


#: Tiers that may support a recommendation on their own. An interpretation
#: resting only on an observation is a hypothesis wearing a conclusion's
#: clothes, which is the specific failure the tier system exists to prevent.
RECOMMENDATION_TIERS = frozenset(
    {EvidenceTier.MEASURED, EvidenceTier.DERIVED, EvidenceTier.VALIDATED}
)


class InputMode(str, Enum):
    """How a factor learns about itself."""

    PASSIVE = "passive"
    """Sensed continuously without the person doing anything."""

    ACTIVE = "active"
    """Requires the person: a survey, a log, an external scan, a test."""


@dataclass(frozen=True, slots=True)
class FactorDefinition:
    """One factor: its question, and honestly what can and cannot sense it."""

    factor: Factor
    core_question: str
    passive_inputs: Tuple[str, ...]
    active_inputs: Tuple[str, ...]
    wrist_sensing_note: str
    """What a wrist device genuinely contributes here. This exists because the
    honest answer differs enormously by factor, and a model that pretends
    otherwise is exactly the hand-waving the whole design is meant to avoid."""

    @property
    def is_wrist_observable(self) -> bool:
        """Whether passive wrist sensing meaningfully covers this factor."""
        return bool(self.passive_inputs)


FIVE_FACTORS: Dict[Factor, FactorDefinition] = {
    Factor.SLEEP: FactorDefinition(
        factor=Factor.SLEEP,
        core_question="Am I restoring?",
        passive_inputs=(
            "sleep_duration",
            "sleep_timing",
            "sleep_consistency",
            "hrv",
            "resting_heart_rate",
            "respiratory_rate",
            "skin_temperature",
            "circadian_phase",
        ),
        active_inputs=(
            "psqi",
            "subjective_sleep_quality",
            "fatigue_rating",
            "travel_and_jet_lag",
        ),
        wrist_sensing_note=(
            "The strongest factor for wrist sensing. Duration, timing and "
            "consistency are directly measurable, and HRV, resting heart rate "
            "and respiratory rate all follow from RR intervals alone. Staging "
            "is the weak part: it is inferred, not measured."
        ),
    ),
    Factor.MOVEMENT: FactorDefinition(
        factor=Factor.MOVEMENT,
        core_question="Am I physically capable and appropriately loaded?",
        passive_inputs=(
            "steps",
            "training_load",
            "heart_rate",
            "vo2max_estimate",
            "sedentary_time",
            "workout_detection",
        ),
        active_inputs=(
            "rpe",
            "soreness",
            "body_composition_scan",
            "strength_tests",
            "mobility",
            "external_training_apps",
        ),
        wrist_sensing_note=(
            "Well covered passively for volume and intensity. Capability -- "
            "strength, mobility, whether load is *appropriate* -- is not "
            "sensed at all and needs active input."
        ),
    ),
    Factor.NUTRITION: FactorDefinition(
        factor=Factor.NUTRITION,
        core_question="Am I adequately fuelled?",
        passive_inputs=(
            "body_mass",
            "body_composition",
            "glucose",
        ),
        active_inputs=(
            "food_logging",
            "hydration",
            "caffeine",
            "alcohol",
            "supplements",
            "meal_photos",
        ),
        wrist_sensing_note=(
            "The weakest factor for wrist sensing: a wrist cannot see food. "
            "The passive inputs listed here arrive from paired instruments -- "
            "a scale (0x2A9D), a body-composition device (0x2A9C), a glucose "
            "sensor (0x2A18) -- not from the band. Claiming otherwise is the "
            "clearest place this model could start hand-waving."
        ),
    ),
    Factor.MIND: FactorDefinition(
        factor=Factor.MIND,
        core_question="How am I functioning psychologically and cognitively?",
        passive_inputs=(
            "hrv",
            "resting_heart_rate",
            "sleep_derived_signals",
        ),
        active_inputs=(
            "mood",
            "perceived_stress",
            "cognitive_tests",
            "reaction_time",
            "burnout_survey",
            "journaling",
        ),
        wrist_sensing_note=(
            "Physiological arousal is measurable; psychological state is not. "
            "HRV is a stress *proxy* and a poor one at the individual level -- "
            "it moves with training load, alcohol, illness and position as "
            "readily as with stress. This factor leans on active input, and "
            "the passive signals mainly indicate when to ask."
        ),
    ),
    Factor.CONNECTION: FactorDefinition(
        factor=Factor.CONNECTION,
        core_question="How connected and supported is my life?",
        passive_inputs=(),
        active_inputs=(
            "loneliness_scale",
            "social_connectedness",
            "relationship_support",
            "time_with_others",
        ),
        wrist_sensing_note=(
            "No passive input, deliberately. Calendar and communication "
            "patterns can generate hypotheses -- and do, in the hypothesis "
            "engine -- but six meetings is not a social-cohesion score. "
            "Scheduled interaction is not connection, and turning one into "
            "the other is false precision. This factor is measured by asking."
        ),
    ),
}


class MedicalContextKind(str, Enum):
    """Categories of medical context. Not factors: modifiers of factors."""

    LAB_RESULT = "lab_result"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    HISTORY = "history"
    VITAL_SIGN = "vital_sign"
    BODY_COMPOSITION = "body_composition"


@dataclass(frozen=True, slots=True)
class MedicalContext:
    """One piece of medical context and which factors it reframes.

    ``modifies`` is the whole point. A beta blocker does not belong to a
    factor; it makes heart-rate-derived readings in Sleep, Movement and Mind
    mean something different, and an interpretation that does not say so is
    wrong in a way the user cannot detect.
    """

    kind: MedicalContextKind
    name: str
    modifies: Tuple[Factor, ...]
    interpretation_note: str
    source_characteristic: str | None = None
    """The GATT characteristic this arrives on, when it arrives from a device."""


MEDICAL_CONTEXTS: Tuple[MedicalContext, ...] = (
    MedicalContext(
        kind=MedicalContextKind.VITAL_SIGN,
        name="blood pressure",
        modifies=(Factor.MOVEMENT, Factor.SLEEP, Factor.MIND),
        interpretation_note=(
            "Elevated readings change what a training recommendation may say "
            "and make an unexplained resting heart rate rise more significant."
        ),
        source_characteristic="0x2A35",
    ),
    MedicalContext(
        kind=MedicalContextKind.BODY_COMPOSITION,
        name="body composition",
        modifies=(Factor.MOVEMENT, Factor.NUTRITION),
        interpretation_note=(
            "Distinguishes mass change that is lean tissue from mass change "
            "that is not, which is the difference between a training block "
            "working and a diet failing."
        ),
        source_characteristic="0x2A9C",
    ),
    MedicalContext(
        kind=MedicalContextKind.MEDICATION,
        name="beta blocker",
        modifies=(Factor.SLEEP, Factor.MOVEMENT, Factor.MIND),
        interpretation_note=(
            "Suppresses heart rate and blunts its response to load. Every "
            "heart-rate-derived reading -- resting heart rate, training zones, "
            "recovery -- must be read against this and never compared to an "
            "untreated population baseline."
        ),
    ),
    MedicalContext(
        kind=MedicalContextKind.DIAGNOSIS,
        name="atrial fibrillation",
        modifies=(Factor.SLEEP, Factor.MIND),
        interpretation_note=(
            "Time-domain HRV becomes uninterpretable: RMSSD is elevated by "
            "the arrhythmia itself, not by parasympathetic tone. Reporting a "
            "recovery score from it would invert the truth."
        ),
    ),
    MedicalContext(
        kind=MedicalContextKind.LAB_RESULT,
        name="thyroid function",
        modifies=(Factor.SLEEP, Factor.MOVEMENT, Factor.NUTRITION, Factor.MIND),
        interpretation_note=(
            "Shifts resting heart rate, temperature and energy availability "
            "together, so a simultaneous drift across several factors has a "
            "single explanation rather than four."
        ),
    ),
    MedicalContext(
        kind=MedicalContextKind.MEDICATION,
        name="SSRI",
        modifies=(Factor.SLEEP, Factor.MIND),
        interpretation_note=(
            "Alters sleep architecture and can reduce HRV independently of "
            "stress, so a declining trend is not evidence of worsening "
            "psychological state."
        ),
    ),
)


def contexts_modifying(factor: Factor) -> List[MedicalContext]:
    """Medical context that changes how ``factor`` should be read."""
    return [c for c in MEDICAL_CONTEXTS if factor in c.modifies]


# --- Mapping the device layer onto the model -------------------------------

#: Which factor each GATT characteristic feeds. Only characteristics we can
#: actually decode appear here: a mapping that lists a characteristic with no
#: decoder behind it describes an aspiration, not a capability.
CHARACTERISTIC_FACTORS: Dict[str, Factor] = {
    "0x2A37": Factor.SLEEP,  # heart rate + RR: also Movement and Mind
    "0x2A1C": Factor.SLEEP,  # temperature: circadian phase
    "0x2A5F": Factor.SLEEP,  # SpO2: respiratory events in sleep
    "0x2A53": Factor.MOVEMENT,
    "0x2A5B": Factor.MOVEMENT,
    "0x2A9C": Factor.NUTRITION,
    "0x2A9D": Factor.NUTRITION,
    "0x2A18": Factor.NUTRITION,
}

#: Characteristics that carry medical context rather than feeding a factor.
CONTEXT_CHARACTERISTICS: Dict[str, MedicalContextKind] = {
    "0x2A35": MedicalContextKind.VITAL_SIGN,
    "0x2A9C": MedicalContextKind.BODY_COMPOSITION,
}


@dataclass
class FactorCoverage:
    """What a given set of characteristics can and cannot cover."""

    covered: Dict[Factor, List[str]] = field(default_factory=dict)
    uncovered: List[Factor] = field(default_factory=list)
    context_available: List[MedicalContextKind] = field(default_factory=list)

    @property
    def factors_covered(self) -> int:
        return len(self.covered)

    def summary(self) -> Dict[str, object]:
        return {
            "factors_covered": sorted(f.value for f in self.covered),
            "factors_uncovered": sorted(f.value for f in self.uncovered),
            "medical_context_available": sorted(
                k.value for k in self.context_available
            ),
            "characteristics_by_factor": {
                f.value: sorted(uuids) for f, uuids in self.covered.items()
            },
        }


def coverage_for(characteristic_uuids: Sequence[str]) -> FactorCoverage:
    """Which factors a device's characteristics actually reach.

    Uncovered factors are reported explicitly rather than omitted. A device
    that cannot see Connection or Nutrition should say so: the gap is what
    active input exists to fill, and hiding it is how a product ends up
    inferring social cohesion from a calendar.
    """
    normalised = {u.strip().upper().replace("0X", "0x") for u in characteristic_uuids}
    coverage = FactorCoverage()
    for uuid in sorted(normalised):
        factor = CHARACTERISTIC_FACTORS.get(uuid)
        if factor is not None:
            coverage.covered.setdefault(factor, []).append(uuid)
        kind = CONTEXT_CHARACTERISTICS.get(uuid)
        if kind is not None and kind not in coverage.context_available:
            coverage.context_available.append(kind)
    coverage.uncovered = [f for f in Factor if f not in coverage.covered]
    return coverage
