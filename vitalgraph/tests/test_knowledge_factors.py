"""Tests for the Five Factor model, Medical Context and the evidence tiers.

The properties asserted here are the ones that stop the model degenerating
into a wellness score: that every factor states what a wrist genuinely cannot
sense, that medical context modifies factors rather than joining them, and
that an interpretation cannot pass itself off as a measurement.
"""

from __future__ import annotations

import pytest

from vitalgraph.knowledge.factors import (
    CHARACTERISTIC_FACTORS,
    FIVE_FACTORS,
    MEDICAL_CONTEXTS,
    RECOMMENDATION_TIERS,
    EvidenceTier,
    Factor,
    contexts_modifying,
    coverage_for,
)


# --- the model's shape -----------------------------------------------------


def test_there_are_exactly_five_factors_and_each_is_defined():
    assert len(Factor) == 5
    assert set(FIVE_FACTORS) == set(Factor)


def test_every_factor_states_its_core_question_and_sensing_limits():
    for factor, definition in FIVE_FACTORS.items():
        assert definition.core_question.endswith("?"), factor
        assert definition.active_inputs, f"{factor} has no active input"
        assert (
            len(definition.wrist_sensing_note) > 80
        ), f"{factor} does not honestly state what a wrist can sense"


def test_connection_has_no_passive_inputs_by_design():
    """Scheduled interaction is not connection.

    Deriving a social-cohesion number from calendar density is the false
    precision this model exists to refuse, so the factor carries no passive
    input at all and must be measured by asking.
    """
    connection = FIVE_FACTORS[Factor.CONNECTION]
    assert connection.passive_inputs == ()
    assert connection.is_wrist_observable is False


def test_nutrition_passive_inputs_come_from_paired_instruments_not_the_wrist():
    """A wrist cannot see food, and the note has to say so."""
    nutrition = FIVE_FACTORS[Factor.NUTRITION]
    note = nutrition.wrist_sensing_note.lower()
    assert "cannot see food" in note
    assert "0x2a9d" in note or "0x2a9c" in note


def test_sleep_is_the_strongest_passive_factor():
    passive_counts = {f: len(d.passive_inputs) for f, d in FIVE_FACTORS.items()}
    assert passive_counts[Factor.SLEEP] == max(passive_counts.values())


# --- medical context is a layer, not a factor ------------------------------


def test_medical_context_is_not_a_factor():
    factor_values = {f.value for f in Factor}
    assert "medical" not in factor_values
    assert "medical_context" not in factor_values


def test_every_medical_context_names_the_factors_it_reframes():
    for context in MEDICAL_CONTEXTS:
        assert context.modifies, f"{context.name} modifies nothing"
        assert all(m in Factor for m in context.modifies)
        assert len(context.interpretation_note) > 60, context.name


def test_a_beta_blocker_reframes_every_heart_rate_derived_factor():
    """The reading does not change; what it means does."""
    beta = next(c for c in MEDICAL_CONTEXTS if c.name == "beta blocker")
    assert set(beta.modifies) >= {Factor.SLEEP, Factor.MOVEMENT, Factor.MIND}


def test_atrial_fibrillation_invalidates_time_domain_hrv():
    af = next(c for c in MEDICAL_CONTEXTS if c.name == "atrial fibrillation")
    assert "rmssd" in af.interpretation_note.lower()
    assert Factor.MIND in af.modifies


@pytest.mark.parametrize("factor", list(Factor))
def test_contexts_modifying_returns_only_relevant_context(factor):
    for context in contexts_modifying(factor):
        assert factor in context.modifies


def test_connection_has_no_medical_context_and_that_is_not_a_bug():
    """Nothing in the medical layer changes how social connection is read.

    Asserted rather than left implicit so that adding one is a deliberate act.
    """
    assert contexts_modifying(Factor.CONNECTION) == []


# --- evidence tiers --------------------------------------------------------


def test_an_interpretation_cannot_support_a_recommendation_alone():
    """The whole point of the tiers: a model's reading is not evidence."""
    assert EvidenceTier.INTERPRETED not in RECOMMENDATION_TIERS
    assert EvidenceTier.OBSERVED not in RECOMMENDATION_TIERS


def test_validated_outranks_observed_for_supporting_a_recommendation():
    """An observation is a hypothesis until it holds on unseen data."""
    assert EvidenceTier.VALIDATED in RECOMMENDATION_TIERS
    assert EvidenceTier.OBSERVED not in RECOMMENDATION_TIERS


def test_measured_and_derived_are_distinct_tiers():
    """ "Weight 81.2 kg" and "lean mass 64.0 kg" are different kinds of claim."""
    assert EvidenceTier.MEASURED != EvidenceTier.DERIVED
    assert {EvidenceTier.MEASURED, EvidenceTier.DERIVED} <= RECOMMENDATION_TIERS


def test_there_is_no_blended_overall_score_tier():
    """A single number would destroy exactly the distinction the tiers make."""
    values = {t.value for t in EvidenceTier}
    assert not values & {"score", "overall", "composite", "index"}


# --- device coverage -------------------------------------------------------


def test_every_mapped_characteristic_has_a_decoder():
    """A mapping without a decoder behind it describes an aspiration."""
    from vitalgraph.ble import gatt, measurements

    decodable = {
        v
        for module in (gatt, measurements)
        for k, v in vars(module).items()
        if k.startswith("CHAR_") and isinstance(v, str)
    }
    assert set(CHARACTERISTIC_FACTORS) <= decodable


def test_coverage_reports_uncovered_factors_explicitly():
    """The gap is the point: it is what active input exists to fill."""
    coverage = coverage_for(["0x2A37", "0x2A1C"])
    assert Factor.SLEEP in coverage.covered
    assert Factor.CONNECTION in coverage.uncovered
    assert Factor.NUTRITION in coverage.uncovered

    summary = coverage.summary()
    assert "connection" in summary["factors_uncovered"]
    assert "nutrition" in summary["factors_uncovered"]


def test_a_band_alone_never_covers_connection():
    """No characteristic reaches Connection, whatever the device advertises."""
    coverage = coverage_for(sorted(CHARACTERISTIC_FACTORS))
    assert Factor.CONNECTION in coverage.uncovered


def test_paired_instruments_extend_coverage_into_nutrition():
    coverage = coverage_for(["0x2A37", "0x2A9C", "0x2A9D", "0x2A18"])
    assert Factor.NUTRITION in coverage.covered
    assert set(coverage.covered[Factor.NUTRITION]) == {"0x2A9C", "0x2A9D", "0x2A18"}


def test_blood_pressure_contributes_context_not_a_factor():
    coverage = coverage_for(["0x2A35"])
    assert coverage.covered == {}
    assert coverage.context_available
    assert "vital_sign" in coverage.summary()["medical_context_available"]


def test_coverage_normalises_uuid_casing():
    """Regression: 0X2A37 must not silently miss a registry keyed 0x2A37."""
    for variant in ("0x2A37", "0X2A37", "0x2a37"):
        assert Factor.SLEEP in coverage_for([variant]).covered


def test_unknown_characteristic_is_ignored_not_guessed():
    coverage = coverage_for(["0xFFFF", "0x2A37"])
    assert list(coverage.covered) == [Factor.SLEEP]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
