"""The bridge: time-series -> content_list. Shapes here must match what
RAGAnything.insert_content_list documents (raganything/processor.py:2232-2239),
because a mismatch fails silently at ingest rather than loudly at import."""

import pytest

from vitalgraph.biometrics.schema import utc
from vitalgraph.biometrics.store import BiometricStore
from vitalgraph.ble.simulator import simulate_period
from vitalgraph.bridge import summarizer as S

START = utc(1772582400)  # 2026-03-04T00:00:00Z


@pytest.fixture()
def seeded():
    """Seven good nights then two clearly bad ones."""
    store = BiometricStore(":memory:")
    recovery = [1.0, 0.95, 1.0, 0.9, 1.0, 0.95, 0.9, 0.2, 0.2]
    store.add(simulate_period(START, nights=9, recovery_by_night=recovery, seed=11))
    yield store
    store.close()


def _summaries(store, nights=9):
    baseline, out = [], []
    for pid, s, e in S.nightly_windows(START, nights):
        summ = S.summarize_period(store, s, e, pid, rmssd_baseline=baseline)
        if summ.metrics.n_beats:
            baseline.append(summ.metrics.rmssd_ms)
        out.append(summ)
    return out


def test_content_list_matches_insert_content_list_contract(seeded):
    items = S.to_content_list(_summaries(seeded)[-1])
    assert [i["type"] for i in items] == ["text", "table"]

    text_item = items[0]
    assert set(text_item) == {"type", "text", "page_idx"}
    assert text_item["text"].strip()

    table_item = items[1]
    assert "table_body" in table_item
    assert table_item["table_body"].startswith("| Metric | Value |")
    assert isinstance(table_item["table_caption"], list)
    assert isinstance(table_item["table_footnote"], list)


def test_doc_id_is_deterministic_so_reingest_updates(seeded):
    a = _summaries(seeded)[-1]
    b = _summaries(seeded)[-1]
    assert a.doc_id == b.doc_id
    assert a.doc_id.startswith("vg-")


def test_distinct_periods_get_distinct_doc_ids(seeded):
    ids = {s.doc_id for s in _summaries(seeded)}
    assert len(ids) == 9


def test_citation_is_a_resolvable_reference(seeded):
    summ = _summaries(seeded)[-1]
    assert summ.citation_ref.startswith("biometrics://default/night-")
    # The date in the citation is the window it actually summarises.
    assert summ.start.strftime("%Y-%m-%d") in summ.citation_ref


def test_bad_nights_are_flagged_against_the_personal_baseline(seeded):
    summaries = _summaries(seeded)
    assert summaries[-1].rmssd_z is not None
    assert summaries[-1].rmssd_z < 0
    assert summaries[-1].verdict in ("poor recovery", "below-average recovery")


def test_early_nights_have_no_baseline_rather_than_a_fabricated_one(seeded):
    first = _summaries(seeded)[0]
    assert first.rmssd_z is None
    assert first.verdict == "no baseline yet"


def test_narrative_states_measured_numbers(seeded):
    summ = _summaries(seeded)[-1]
    text = S.render_narrative(summ)
    assert f"{summ.metrics.rmssd_ms:.1f} ms" in text
    assert f"{summ.metrics.mean_hr_bpm:.1f} bpm" in text
    assert summ.verdict in text


def test_narrative_is_stable_across_runs(seeded):
    """Byte-stable output is what keeps doc_id meaningful as an update key."""
    summ = _summaries(seeded)[-1]
    assert S.render_narrative(summ) == S.render_narrative(summ)


def test_empty_window_degrades_gracefully():
    store = BiometricStore(":memory:")
    summ = S.summarize_period(store, START, utc(START.timestamp() + 3600), "empty")
    assert summ.metrics.n_beats == 0
    assert summ.verdict == "insufficient data"
    assert "No usable heart-rate data" in S.render_narrative(summ)
    items = S.to_content_list(summ)
    assert items[0]["type"] == "text"


def test_metrics_table_contains_real_values(seeded):
    summ = _summaries(seeded)[-1]
    table = S.render_metrics_table(summ)
    assert "RMSSD (ms)" in table
    assert f"{summ.metrics.rmssd_ms:.1f}" in table
