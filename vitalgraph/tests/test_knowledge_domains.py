"""Decision domains: their integrity, and the sensor questions they imply."""

import pytest

from vitalgraph.knowledge import domains as D
from vitalgraph.knowledge import sensors as N
from vitalgraph.knowledge import signals as S


def test_domain_ids_and_namespaces_are_unique():
    """Each domain retrieves over its own corpus; a collision would pool two."""
    ids = [d.id for d in D.DOMAINS]
    assert len(ids) == len(set(ids))
    assert len(D.namespaces()) == len(set(D.namespaces())) == len(D.DOMAINS)


def test_namespaces_are_domain_scoped():
    for domain in D.DOMAINS:
        assert domain.namespace == f"vybe/{domain.id}"


def test_decision_ids_are_unique_within_a_domain():
    for domain in D.DOMAINS:
        ids = [dec.id for dec in domain.decisions]
        assert len(ids) == len(set(ids)), domain.id


def test_every_domain_has_decisions_and_corpus_topics():
    for domain in D.DOMAINS:
        assert domain.decisions, domain.id
        assert domain.corpus_topics, domain.id


def test_decisions_cannot_reference_unknown_signals():
    """This caught a real typo during authoring -- a signal *class* name used
    where a signal id belonged."""
    with pytest.raises(S.UnknownSignal):
        D.Decision("bogus", "?", ("body_composition",))


def test_every_referenced_signal_exists():
    for _, decision in D.all_decisions():
        for signal_id in decision.requires + decision.helpful:
            assert signal_id in S.BY_ID


def test_every_decision_requires_something():
    for domain_id, decision in D.all_decisions():
        assert decision.requires, f"{domain_id}/{decision.id}"


def test_unknown_domain_and_decision_raise():
    with pytest.raises(D.UnknownDomain):
        D.get("not_a_domain")
    with pytest.raises(KeyError):
        D.get("training_readiness").decision("not_a_decision")


def test_requirements_resolve_to_measured_signals():
    """A domain asking for derived signals is really asking for sensors."""
    readiness = D.get("training_readiness")
    assert "rmssd" in readiness.required_signals
    assert "rr_interval" in readiness.required_roots
    # RMSSD is derived, so it must not appear as its own root.
    assert "rmssd" not in readiness.required_roots


def test_signal_demand_reads_as_a_sensor_priority_list():
    """The signals many decisions depend on are where hardware budget goes."""
    roots = D.root_signal_demand()
    assert next(iter(roots)) == "rr_interval"
    assert roots["rr_interval"] > roots.get("spo2", 0)


def test_blast_radius_of_a_signal_is_queryable():
    affected = {d.id for d in D.domains_requiring("rmssd")}
    assert "training_readiness" in affected
    assert "illness_onset" in affected
    assert "metabolic_glycemic" not in affected


def test_evidence_grades_are_assigned_and_not_uniformly_high():
    """A catalogue where everything is HIGH is a marketing document."""
    grades = {decision.evidence for _, decision in D.all_decisions()}
    assert len(grades) >= 3
    assert D.Evidence.LOW in grades


def test_glucose_domain_is_unreachable_without_a_cgm():
    """Structural, not incidental: no cardiac or optical signal reaches it."""
    roots = D.get("metabolic_glycemic").required_roots
    assert "interstitial_glucose" in roots
    providers = {s.id for s in N.providers_of("interstitial_glucose")}
    assert providers == {"cgm"}


def test_apnea_domain_needs_spo2_faster_than_wrist_delivers():
    """The clearest case where wrist-only hardware cannot reach a decision."""
    assert "spo2" in D.get("sleep_disordered_breathing").required_roots
    assert N.get("ppg_wrist").meets_minimum("spo2") is False
    assert N.get("ppg_finger").meets_minimum("spo2") is True


def test_afib_decision_needs_waveform_not_intervals():
    decision = D.get("cardiac_rhythm").decision("afib_detection")
    assert "ecg_waveform" in decision.requires
    # A chest strap gives beat-accurate intervals but no morphology.
    assert N.get("ecg_chest_strap").rate_for("ecg_waveform") is None
    assert N.get("ecg_patch").meets_minimum("ecg_waveform") is True


def test_medication_context_is_modelled_as_a_requirement():
    """Rate-control drugs make heart-rate-based scores uninterpretable."""
    decision = D.get("medication_response").decision("beta_blocker_confound")
    assert "medication" in decision.requires
    assert decision.evidence is D.Evidence.HIGH


def test_history_requirements_are_stated_where_baselines_are_needed():
    readiness = D.get("training_readiness").decision("readiness_today")
    assert readiness.min_history_days >= 14
    cycle = D.get("menstrual_hormonal").decision("phase_tracking")
    assert cycle.min_history_days >= 60  # two or more cycles


def test_best_evidence_reports_the_strongest_decision_in_a_domain():
    assert D.get("cardiac_rhythm").best_evidence is D.Evidence.HIGH
    assert D.get("stress_autonomic").best_evidence is D.Evidence.MODERATE


def test_summary_is_serialisable_for_the_api():
    s = D.summary()
    assert s["domains"] == len(D.DOMAINS)
    assert s["decisions"] == len(D.all_decisions())
    assert len(s["namespaces"]) == len(D.DOMAINS)
