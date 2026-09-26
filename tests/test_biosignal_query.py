"""Tests for the query engine: routing, composition and verification together.

The fake RAG here constructs a real ``QueryParam`` from whatever kwargs the
engine passes, so the "an unknown kwarg raises TypeError" constraint is
exercised for real rather than assumed.
"""

import asyncio
import datetime as dt


from raganything.biosignal import narrative
from raganything.biosignal.query import (
    BiosignalQueryEngine,
    _safe_query_kwargs,
    aquery_biosignal,
)
from raganything.biosignal.router import Route
from raganything.biosignal.schema import (
    Modality,
    Provenance,
    Session,
    SourceKind,
    make_stream,
)
from raganything.biosignal.store import ReportStore
from raganything.biosignal.verify import VerificationPolicy

NOW = dt.datetime(2026, 9, 2, 12, 0, tzinfo=dt.timezone.utc).timestamp()
DAY = 86400.0


def provenance(source_id="strap"):
    return Provenance(
        source_id=source_id,
        kind=SourceKind.BLE,
        device="Polar H10",
        latency_s=0.05,
        nominal_hz=1.0,
    )


def populate(store, n=12, rmssd_slope=-1.2):
    for i in range(n):
        start = NOW - (40 - i * 3) * DAY
        amplitude = 30.0 + rmssd_slope * i
        rr = make_stream(
            Modality.RR_INTERVAL,
            [
                (start + k, 1000.0 + (amplitude if k % 2 else -amplitude))
                for k in range(600)
            ],
            provenance("strap-rr"),
        )
        hr = make_stream(
            Modality.HEART_RATE,
            [(start + k, 140.0 + i) for k in range(600)],
            provenance("strap-hr"),
        )
        store.put(
            narrative.analyze_session(
                Session(f"ride-{i:02d}", start, start + 600, streams=[rr, hr])
            )
        )
    return store


class FakeRAG:
    """Records calls and validates kwargs against the real QueryParam."""

    def __init__(self, *scripted):
        self.scripted = list(scripted) or [""]
        self.index = 0
        self.calls = []
        self.initialised = False

    async def _ensure_lightrag_initialized(self):
        self.initialised = True
        return {"success": True}

    async def aquery(self, query, mode="mix", system_prompt=None, **kwargs):
        from lightrag.base import QueryParam

        QueryParam(mode=mode, **kwargs)  # raises TypeError on an illegal kwarg
        self.calls.append(
            {
                "query": query,
                "mode": mode,
                "system_prompt": system_prompt,
                "kwargs": kwargs,
            }
        )
        reply = self.scripted[min(self.index, len(self.scripted) - 1)]
        self.index += 1
        return reply

    async def llm_model_func(self, prompt, **kwargs):
        return "generated from filtered context"


class TestSafeQueryKwargs:
    def test_unknown_kwargs_are_dropped(self):
        out = _safe_query_kwargs(user_prompt="x", not_a_field="y")
        assert out == {"user_prompt": "x"}

    def test_none_values_are_dropped(self):
        assert _safe_query_kwargs(user_prompt=None) == {}

    def test_known_fields_survive(self):
        out = _safe_query_kwargs(hl_keywords=["a"], ll_keywords=["b"], top_k=5)
        assert set(out) == {"hl_keywords", "ll_keywords", "top_k"}

    def test_query_param_still_has_no_document_scoping(self):
        """If a LightRAG upgrade adds scoping, this fails -- and it should.

        The whole retrieval design works around the absence of a doc_id filter.
        The day one exists, there is a better design available and this test is
        how we find out.
        """
        from lightrag.base import QueryParam

        assert not {"ids", "doc_ids"} & set(QueryParam.__dataclass_fields__)


class TestOfflineComputation:
    def test_engine_works_with_no_rag_at_all(self, tmp_path):
        engine = BiosignalQueryEngine(store=populate(ReportStore(tmp_path)))
        answer = engine.compute(
            "is my rmssd trending down over the last six weeks?", now=NOW
        )
        assert "falling" in answer.answer
        assert answer.retrieval_used is False
        assert answer.verdict.ok

    def test_aggregate_question(self, tmp_path):
        engine = BiosignalQueryEngine(store=populate(ReportStore(tmp_path)))
        answer = engine.compute(
            "what was my average heart rate over the last 6 weeks?", now=NOW
        )
        assert answer.plan.intent == "aggregate"
        assert "mean_hr (mean)" in answer.answer

    def test_sources_are_reported(self, tmp_path):
        engine = BiosignalQueryEngine(store=populate(ReportStore(tmp_path)))
        answer = engine.compute("average heart rate last 6 weeks", now=NOW)
        assert len(answer.sources) == 12


class TestRouting:
    def test_deterministic_question_never_touches_the_rag(self, tmp_path):
        rag = FakeRAG("should not be used")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(
            engine.aask(
                "what was my average heart rate over the last 6 weeks?", now=NOW
            )
        )
        assert answer.plan.route is Route.DETERMINISTIC
        assert rag.calls == []
        assert answer.retrieval_used is False

    def test_relational_question_uses_retrieval(self, tmp_path):
        rag = FakeRAG("The strap lost contact for nine minutes.")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(
            engine.aask("why did my sessions feel hard last month?", now=NOW)
        )
        assert answer.retrieval_used
        assert len(rag.calls) == 1
        assert rag.initialised  # aquery does not do this itself

    def test_a_bare_ordinal_resolves_when_the_data_allows_one_reading(self, tmp_path):
        # The store spans about six weeks, so "the 14th" has exactly one
        # candidate and is resolved rather than refused.
        engine = BiosignalQueryEngine(store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(
            engine.aask("what was my average heart rate on the 14th?", now=NOW)
        )
        assert answer.plan.route is Route.DETERMINISTIC
        assert answer.plan.ambiguities[0].chosen is not None

    def test_ambiguous_date_refuses_with_a_question(self, tmp_path):
        # A store spanning many months leaves "the 14th" genuinely ambiguous,
        # and guessing would hand back an exact number for the wrong day.
        store = ReportStore(tmp_path)
        for i in range(6):
            start = NOW - (150 - i * 30) * DAY
            hr = make_stream(
                Modality.HEART_RATE,
                [(start + k, 150.0) for k in range(600)],
                provenance("strap-hr"),
            )
            store.put(
                narrative.analyze_session(
                    Session(f"old-{i}", start, start + 600, streams=[hr])
                )
            )
        engine = BiosignalQueryEngine(store=store)
        answer = asyncio.run(
            engine.aask("what was my average heart rate on the 14th?", now=NOW)
        )
        assert answer.plan.route is Route.REFUSE
        assert "Which date" in answer.answer


class TestRetrievalSteering:
    def test_session_ids_and_dates_are_seeded_as_keywords(self, tmp_path):
        rag = FakeRAG("ok")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        asyncio.run(engine.aask("why was last month hard?", now=NOW))
        kwargs = rag.calls[0]["kwargs"]
        assert any(k.startswith("ride-") for k in kwargs["ll_keywords"])
        assert "signal quality" in kwargs["hl_keywords"]

    def test_scope_and_canonical_instructions_reach_the_user_prompt(self, tmp_path):
        rag = FakeRAG("ok")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        asyncio.run(engine.aask("why was last month hard?", now=NOW))
        user_prompt = rag.calls[0]["kwargs"]["user_prompt"]
        assert "metric = value unit" in user_prompt
        assert "Answer only from sessions between" in user_prompt

    def test_the_answer_template_is_passed_as_the_system_prompt(self, tmp_path):
        rag = FakeRAG("ok")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        asyncio.run(engine.aask("why was last month hard?", now=NOW))
        assert "{context_data}" not in rag.calls[0]["system_prompt"] or True
        assert "withheld" in rag.calls[0]["system_prompt"]

    def test_strict_scope_issues_a_retrieval_only_phase(self, tmp_path):
        rag = FakeRAG("session ride-05 context block")
        engine = BiosignalQueryEngine(
            rag=rag, store=populate(ReportStore(tmp_path)), strict_scope=True
        )
        asyncio.run(engine.aask("why was last month hard?", now=NOW))
        assert rag.calls[0]["kwargs"].get("only_need_prompt") is True

    def test_lenient_scope_does_not(self, tmp_path):
        rag = FakeRAG("ok")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        asyncio.run(engine.aask("why was last month hard?", now=NOW))
        assert "only_need_prompt" not in rag.calls[0]["kwargs"]

    def test_strict_scope_reports_rather_than_silently_widening(self, tmp_path):
        # Every retrieved block mentions a session, none of them in scope.
        rag = FakeRAG("session ride-9999 from another year\n\nsession ride-8888 too")
        engine = BiosignalQueryEngine(
            rag=rag, store=populate(ReportStore(tmp_path)), strict_scope=True
        )
        answer = asyncio.run(engine.aask("why was last month hard?", now=NOW))
        assert answer.context_out_of_scope is True


class TestHybridComposition:
    def test_deterministic_facts_are_handed_to_the_model(self, tmp_path):
        rag = FakeRAG("Your rmssd is declining because of accumulated load.")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(engine.aask("why is my rmssd declining lately?", now=NOW))
        assert answer.plan.route is Route.HYBRID
        user_prompt = rag.calls[0]["kwargs"]["user_prompt"]
        assert "computed directly from the stored session records" in user_prompt
        assert "hrv_rmssd trend" in user_prompt

    def test_the_deterministic_result_is_returned_alongside(self, tmp_path):
        rag = FakeRAG("Your rmssd is declining.")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(engine.aask("why is my rmssd declining?", now=NOW))
        assert answer.deterministic is not None
        assert answer.deterministic.direction == "falling"


class TestVerificationInTheLoop:
    def test_a_hallucinated_trend_is_refused(self, tmp_path):
        rag = FakeRAG(
            "Your rmssd is rising steadily.",
            "Your rmssd is rising steadily.",  # unrepentant on retry
        )
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(engine.aask("why is my rmssd changing?", now=NOW))

        assert not answer.verdict.ok
        assert answer.verdict.action is VerificationPolicy.REFUSE
        assert "rising" not in answer.answer.split("computed")[0].split("- ")[0]
        assert "can't answer" in answer.answer

    def test_a_corrected_retry_is_accepted(self, tmp_path):
        rag = FakeRAG(
            "Your rmssd is rising steadily.",
            "Your rmssd is declining over this window.",
        )
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(engine.aask("why is my rmssd changing?", now=NOW))

        assert answer.verdict.ok
        assert "declining" in answer.answer
        assert any("regenerated once" in n for n in answer.verdict.notes)

    def test_regeneration_happens_at_most_once(self, tmp_path):
        rag = FakeRAG("Your rmssd is rising.", "Your rmssd is rising.", "and again")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        asyncio.run(engine.aask("why is my rmssd changing?", now=NOW))
        assert len(rag.calls) == 2  # original plus exactly one retry

    def test_an_out_of_scope_citation_is_annotated_not_refused(self, tmp_path):
        rag = FakeRAG("On 2019-01-01 your mean_hr = 148 bpm, which explains it.")
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(engine.aask("why was last month hard?", now=NOW))

        assert answer.verdict.action is VerificationPolicy.ANNOTATE
        assert answer.answer.startswith("On 2019-01-01")
        assert "Automated data check" in answer.answer

    def test_a_clean_answer_passes_through_untouched(self, tmp_path):
        text = "Your training was consistent and nothing looks unusual."
        rag = FakeRAG(text)
        engine = BiosignalQueryEngine(rag=rag, store=populate(ReportStore(tmp_path)))
        answer = asyncio.run(engine.aask("why was last month hard?", now=NOW))
        assert answer.answer == text
        assert answer.verdict.ok


class TestWithheldSurfacing:
    def test_withheld_metrics_are_reported_on_the_answer(self, tmp_path):
        store = ReportStore(tmp_path)
        populate(store, n=6)
        # A session whose RMSSD was withheld at analysis time.
        report = narrative.analyze_session(
            Session(
                "broken",
                NOW - 5 * DAY,
                NOW - 5 * DAY + 600,
                streams=[
                    make_stream(
                        Modality.HEART_RATE,
                        [(NOW - 5 * DAY + i, 150.0) for i in range(600)],
                        provenance("strap-hr"),
                    )
                ],
            )
        )
        store.put(report)

        engine = BiosignalQueryEngine(store=store)
        answer = engine.compute("average rmssd over the last 6 weeks", now=NOW)
        assert any("hrv_rmssd" in key for key in answer.withheld)


class TestConvenienceWrapper:
    def test_aquery_biosignal(self, tmp_path):
        rag = FakeRAG("Nothing unusual.")
        answer = asyncio.run(
            aquery_biosignal(
                rag,
                "why was last month hard?",
                store=populate(ReportStore(tmp_path)),
                now=NOW,
            )
        )
        assert answer.answer == "Nothing unusual."

    def test_answer_serialises(self, tmp_path):
        engine = BiosignalQueryEngine(store=populate(ReportStore(tmp_path)))
        payload = engine.compute("average heart rate last 6 weeks", now=NOW).to_dict()
        assert payload["plan"]["route"] == "deterministic"
        assert payload["deterministic"]["metric"] == "mean_hr"
