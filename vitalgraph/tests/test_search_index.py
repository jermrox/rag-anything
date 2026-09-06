"""Tests for keyless lexical search over the harvested corpus.

The properties worth locking in are the ones that make this usable without a
language model: exact tokens survive tokenization, identifiers match the words
inside them, every hit carries a citation, and the licence gate is still
visible at query time.
"""

from __future__ import annotations

import pytest

from vitalgraph.search.index import (
    STOPWORDS,
    Document,
    SearchIndex,
    format_hits,
    tokenize,
)


def _doc(doc_id: str, text: str, **kwargs) -> Document:
    return Document(
        doc_id=doc_id,
        text=text,
        citation=kwargs.pop("citation", f"repo@abc1234:{doc_id}.py#L1-L10"),
        repo=kwargs.pop("repo", "repo"),
        kind=kwargs.pop("kind", "code"),
        **kwargs,
    )


# --- tokenization ----------------------------------------------------------


def test_hex_literals_survive_as_single_tokens():
    """The highest-value query in this corpus is a bare UUID.

    Splitting 0x2A37 into '0' and 'x2a37' would make it unanswerable.
    """
    assert "0x2a37" in tokenize("flags = 0x2A37")


def test_hex_literals_are_case_normalised():
    """0x2A37 and 0x2a37 must be the same term.

    The same normalisation failure once made a UUID registry lookup return an
    empty result rather than an error.
    """
    assert tokenize("0x2A37") == tokenize("0x2a37")


def test_snake_case_identifiers_match_their_words():
    terms = tokenize("def decode_heart_rate(payload):")
    assert "decode_heart_rate" in terms
    assert "heart" in terms
    assert "rate" in terms


def test_camel_case_identifiers_match_their_words():
    terms = tokenize("class HeartRateMeasurementCallback")
    assert "heart" in terms
    assert "rate" in terms
    assert "measurement" in terms


def test_acronym_runs_split_sensibly():
    """UUIDValue must not shatter into single letters."""
    terms = tokenize("CharacteristicUUIDValue")
    assert "uuid" in terms
    assert "value" in terms
    assert "u" not in terms


def test_stopwords_and_single_characters_are_dropped():
    terms = tokenize("the value is a b c")
    assert not (set(terms) & STOPWORDS)
    assert "b" not in terms


def test_tokenizing_empty_text_is_empty():
    assert tokenize("") == []


# --- ranking ---------------------------------------------------------------


def test_an_exact_uuid_query_finds_the_document_that_names_it():
    index = SearchIndex()
    index.add(_doc("hr", "Bluetooth characteristic 0x2A37 Heart Rate Measurement"))
    index.add(_doc("battery", "Bluetooth characteristic 0x2A19 Battery Level"))

    hits = index.search("0x2A37")
    assert len(hits) == 1
    assert hits[0].document.doc_id == "hr"
    assert hits[0].matched_terms == ("0x2a37",)


def test_a_rare_term_outranks_a_common_one():
    """IDF is what stops a word appearing everywhere from deciding a ranking."""
    index = SearchIndex()
    for i in range(20):
        index.add(_doc(f"common{i}", "bluetooth device connection handling"))
    index.add(_doc("rare", "bluetooth device impedance measurement"))

    hits = index.search("bluetooth impedance")
    assert hits[0].document.doc_id == "rare"


def test_length_normalisation_prefers_the_focused_document():
    """A long file mentioning a term in passing must not beat a short one
    that is about it."""
    index = SearchIndex()
    index.add(_doc("focused", "spo2 desaturation detection"))
    index.add(
        _doc(
            "sprawling",
            "spo2 " + " ".join(f"unrelated{i}" for i in range(400)),
        )
    )
    hits = index.search("spo2")
    assert hits[0].document.doc_id == "focused"


def test_idf_is_floored_so_a_ubiquitous_term_never_scores_negative():
    """Unfloored BM25 goes negative past 50% document frequency.

    A document could then improve its score by *not* matching a query term,
    which is incoherent.
    """
    index = SearchIndex()
    for i in range(10):
        index.add(_doc(f"d{i}", "ubiquitous term appears everywhere"))
    hits = index.search("ubiquitous")
    assert all(hit.score >= 0.0 for hit in hits)


def test_documents_without_any_query_term_are_not_returned():
    index = SearchIndex()
    index.add(_doc("a", "heart rate measurement"))
    index.add(_doc("b", "battery level"))
    assert [h.document.doc_id for h in index.search("heart")] == ["a"]


def test_search_is_deterministic_for_tied_scores():
    """Ties break on document id, so a ranking never reorders between runs."""
    index = SearchIndex()
    for name in ("c", "a", "b"):
        index.add(_doc(name, "identical text for every document"))
    first = [h.document.doc_id for h in index.search("identical")]
    assert first == sorted(first)


def test_empty_query_and_empty_index_return_nothing():
    assert SearchIndex().search("anything") == []
    index = SearchIndex()
    index.add(_doc("a", "text"))
    assert index.search("") == []


def test_limit_is_respected():
    index = SearchIndex()
    for i in range(20):
        index.add(_doc(f"d{i}", "heart rate measurement"))
    assert len(index.search("heart", limit=5)) == 5


# --- filters ---------------------------------------------------------------


def test_results_can_be_restricted_to_one_repository():
    index = SearchIndex()
    index.add(_doc("a", "heart rate", repo="bleak"))
    index.add(_doc("b", "heart rate", repo="NeuroKit"))
    hits = index.search("heart", repo="bleak")
    assert [h.document.repo for h in hits] == ["bleak"]


def test_results_can_be_restricted_to_one_kind():
    index = SearchIndex()
    index.add(_doc("a", "characteristic 0x2A37", kind="protocol_fact"))
    index.add(_doc("b", "characteristic 0x2A37", kind="code"))
    hits = index.search("0x2A37", kind="protocol_fact")
    assert [h.document.kind for h in hits] == ["protocol_fact"]


# --- citations and the licence gate ----------------------------------------


def test_every_hit_carries_a_citation():
    """A snippet without one is a claim; with one it is a reference."""
    index = SearchIndex()
    index.add(_doc("a", "heart rate measurement"))
    for hit in index.search("heart"):
        assert hit.document.citation
        assert "@" in hit.document.citation


def test_licence_policy_survives_into_results():
    """A caller must be able to tell a quotable hit from a describable one."""
    index = SearchIndex()
    index.add(_doc("permissive", "heart rate", policy="verbatim"))
    index.add(_doc("copyleft", "heart rate", policy="facts_only"))

    by_id = {h.document.doc_id: h.document for h in index.search("heart")}
    assert by_id["permissive"].may_quote is True
    assert by_id["copyleft"].may_quote is False


def test_facts_only_results_are_marked_in_rendered_output():
    index = SearchIndex()
    index.add(_doc("copyleft", "heart rate", policy="facts_only"))
    assert "[facts only]" in format_hits(index.search("heart"))


def test_rendered_output_does_not_waste_the_snippet_on_the_citation():
    """The Source line is printed above as the citation.

    Repeating it inside a truncated snippet spends the whole visible width on
    a path the reader has already seen.
    """
    index = SearchIndex()
    index.add(
        _doc(
            "a",
            "Source: repo@abc1234:a.py#L1-L10\nSymbol `decode` decodes the payload",
        )
    )
    rendered = format_hits(index.search("decode"))
    assert "Symbol `decode`" in rendered
    assert rendered.count("repo@abc1234:a.py#L1-L10") == 1


def test_no_matches_renders_as_a_statement_not_an_empty_string():
    assert format_hits([]) == "no matches"


# --- persistence -----------------------------------------------------------


def test_index_round_trips_through_disk(tmp_path):
    index = SearchIndex()
    index.add(_doc("a", "heart rate measurement 0x2A37"))
    index.add(_doc("b", "battery level 0x2A19", policy="facts_only"))
    path = index.save(tmp_path / "index.json")

    reloaded = SearchIndex.load(path)
    assert len(reloaded.documents) == 2
    assert [h.document.doc_id for h in reloaded.search("0x2A37")] == ["a"]
    assert reloaded.documents[1].may_quote is False


def test_saved_index_stores_documents_not_postings(tmp_path):
    """Postings rebuild deterministically; a serialised one would go stale
    the moment the tokenizer changed."""
    import json

    index = SearchIndex()
    index.add(_doc("a", "heart rate"))
    path = index.save(tmp_path / "index.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload) == {"version", "documents"}
    assert "postings" not in payload


def test_stats_describe_the_corpus():
    index = SearchIndex()
    index.add(_doc("a", "heart rate", repo="bleak", kind="code"))
    index.add(_doc("b", "0x2A37", repo="bleak", kind="protocol_fact"))
    stats = index.stats()
    assert stats["documents"] == 2
    assert stats["repos"] == ["bleak"]
    assert stats["kinds"] == ["code", "protocol_fact"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
