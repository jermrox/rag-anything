"""Per-domain RAG: storage isolation, seed content, and question routing.

No LLM or raganything required -- the registry takes an injectable factory
precisely so the partitioning and routing are testable on their own.
"""

import pytest

from vitalgraph.knowledge import domains as D
from vitalgraph.knowledge import rag_domains as R


# --- storage isolation -----------------------------------------------------


def test_every_domain_gets_its_own_working_directory(tmp_path):
    """Twenty separate graphs, not one pooled corpus."""
    registry = R.DomainRAGRegistry(root=tmp_path, factory=lambda wd: {"wd": wd})
    layout = registry.layout()
    assert len(layout) == len(D.DOMAINS)
    assert len(set(layout.values())) == len(layout)


def test_working_dir_is_scoped_by_domain(tmp_path):
    path = R.working_dir_for("sleep_disordered_breathing", tmp_path)
    assert path.name == "sleep_disordered_breathing"
    assert path.parent == tmp_path


def test_instances_are_created_lazily(tmp_path):
    """Standing up twenty graphs when one is needed would be wasteful."""
    registry = R.DomainRAGRegistry(root=tmp_path, factory=lambda wd: {"wd": wd})
    assert registry.loaded == []
    registry.get("illness_onset")
    assert registry.loaded == ["illness_onset"]
    registry.get("illness_onset")
    assert registry.loaded == ["illness_onset"]  # reused, not rebuilt


def test_unknown_domain_is_rejected(tmp_path):
    registry = R.DomainRAGRegistry(root=tmp_path, factory=lambda wd: {"wd": wd})
    with pytest.raises(D.UnknownDomain):
        registry.get("not_a_domain")


# --- seed content ----------------------------------------------------------


def test_seed_content_matches_the_insert_contract():
    items = R.seed_content_list(D.get("training_readiness"))
    assert [i["type"] for i in items] == ["text", "table", "table"]
    assert set(items[0]) == {"type", "text", "page_idx"}
    for table in items[1:]:
        assert table["table_body"].startswith("| ")
        assert isinstance(table["table_caption"], list)
        assert isinstance(table["table_footnote"], list)


def test_seed_content_is_deterministic():
    """Stable output keeps the deterministic doc_id meaningful on re-seed."""
    domain = D.get("cardiac_rhythm")
    assert R.seed_content_list(domain) == R.seed_content_list(domain)


def test_seed_content_states_the_sensor_requirement():
    items = R.seed_content_list(D.get("metabolic_glycemic"))
    text = items[0]["text"]
    assert "interstitial_glucose" in text
    assert "vybe/metabolic_glycemic" in text


def test_adequacy_table_names_inadequate_sensors():
    """The table that answers 'can this hardware reach this decision'."""
    body = R.seed_content_list(D.get("sleep_disordered_breathing"))[2]["table_body"]
    assert "spo2" in body
    assert "ppg_wrist" in body  # listed as inadequate
    assert "ppg_finger" in body  # listed as adequate


def test_adequacy_footnote_preserves_the_two_failure_modes():
    items = R.seed_content_list(D.get("sleep_disordered_breathing"))
    footnote = items[2]["table_footnote"][0]
    assert "different from not" in footnote


def test_evidence_footnote_separates_literature_from_adequacy():
    items = R.seed_content_list(D.get("cardiac_rhythm"))
    footnote = items[1]["table_footnote"][0]
    assert "inadequate sensor is still unsupported" in footnote


def test_every_domain_produces_seed_content():
    for domain in D.DOMAINS:
        items = R.seed_content_list(domain)
        assert len(items) == 3
        assert items[0]["text"].strip()


# --- routing ---------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("why was my recovery bad last week", "training_readiness"),
        ("am I getting sick", "illness_onset"),
        ("do I stop breathing at night", "sleep_disordered_breathing"),
        ("how deep was my sleep", "sleep_architecture"),
        ("does alcohol wreck my sleep", "medication_response"),
        ("how did that meal affect my glucose", "metabolic_glycemic"),
        ("is my heart rhythm irregular", "cardiac_rhythm"),
        ("where am I in my cycle", "menstrual_hormonal"),
        ("is my blood pressure rising", "blood_pressure_hemodynamics"),
        ("is my VO2 max improving", "cardiorespiratory_fitness"),
        ("am I getting enough sunlight", "environmental_exposure"),
    ],
)
def test_questions_route_to_the_right_domain(question, expected):
    matches = R.route(question, limit=1)
    assert matches, question
    assert matches[0].domain_id == expected


def test_routing_is_deterministic():
    assert R.route("am I getting sick") == R.route("am I getting sick")


def test_routing_explains_itself():
    """When routing picks wrongly it has to be inspectable."""
    match = R.route("does alcohol wreck my sleep", limit=1)[0]
    assert "alcohol" in match.matched
    assert match.matched[0] == "alcohol"  # most discriminative term first


def test_lay_terms_are_understood():
    """Users type 'hungover', not 'substance effect on recovery'."""
    for question, expected in (
        ("I'm hungover", "medication_response"),
        ("I feel exhausted", "training_readiness"),
        ("when should I focus", "cognitive_alertness"),
    ):
        assert R.route(question, limit=1)[0].domain_id == expected


def test_function_words_do_not_drive_routing():
    """Rarity is not meaning: 'getting' appears in one domain, so raw IDF
    scored it as highly discriminative and hijacked the question."""
    assert R.term_weight("getting") == 0.0
    assert R.term_weight("alcohol") > R.term_weight("sleep") > 0.0


def test_ambiguous_questions_return_several_candidates():
    """Real questions span domains; forcing a single answer loses information."""
    matches = R.route("when should I do hard mental work", limit=2)
    assert {m.domain_id for m in matches} >= {"cognitive_alertness"}
    assert len(matches) == 2


def test_unmatched_question_returns_nothing_rather_than_a_guess():
    assert R.route("what is the capital of France") == []
    assert R.route("") == []


def test_route_respects_the_limit():
    assert len(R.route("sleep", limit=3)) <= 3


# --- corpus plan -----------------------------------------------------------


def test_corpus_plan_is_a_harvesting_work_list():
    plan = {row["domain"]: row for row in R.corpus_plan()}
    assert len(plan) == len(D.DOMAINS)
    apnea = plan["sleep_disordered_breathing"]
    assert apnea["topics"]
    assert "spo2" in apnea["required_roots"]
    assert apnea["namespace"] == "vybe/sleep_disordered_breathing"
    assert apnea["decisions"] >= 1
