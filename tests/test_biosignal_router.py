"""Tests for question classification and date parsing."""

import asyncio
import datetime as dt

import pytest

from raganything.biosignal.router import (
    METRIC_ALIASES,
    QueryPlan,
    Route,
    classify_rules,
    parse_window,
    route,
)
from raganything.biosignal.timeseries import METRIC_SUPPORT

NOW = dt.datetime(2026, 9, 2, 12, 0, tzinfo=dt.timezone.utc).timestamp()
DAY = 86400.0


def iso(value):
    return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc).date().isoformat()


class TestLexiconCoverage:
    def test_every_metric_is_reachable_by_some_alias(self):
        reachable = {m for metrics in METRIC_ALIASES.values() for m in metrics}
        assert set(METRIC_SUPPORT) <= reachable

    def test_aliases_only_name_real_metrics(self):
        for alias, metrics in METRIC_ALIASES.items():
            for metric in metrics:
                assert metric in METRIC_SUPPORT, f"{alias} -> unknown {metric}"


class TestRuleClassification:
    @pytest.mark.parametrize(
        "question,expected_route,expected_intent,expected_metrics",
        [
            (
                "is my RMSSD trending down over the last six weeks?",
                Route.DETERMINISTIC,
                "trend",
                ("hrv_rmssd",),
            ),
            (
                "what was my average power in July?",
                Route.DETERMINISTIC,
                "aggregate",
                ("mean_power",),
            ),
            (
                "what was my average heart rate over the last 6 weeks?",
                Route.DETERMINISTIC,
                "aggregate",
                ("mean_hr",),
            ),
            (
                "what is my peak heart rate this month",
                Route.DETERMINISTIC,
                "aggregate",
                ("max_hr_observed",),
            ),
            (
                "why was my recovery poor after that block?",
                Route.RETRIEVAL,
                "explain",
                (),
            ),
            (
                "which device disagrees during lifting?",
                Route.RETRIEVAL,
                "explain",
                (),
            ),
            (
                "why is my rmssd declining?",
                Route.HYBRID,
                "trend",
                ("hrv_rmssd",),
            ),
            (
                "compare my normalized power in July against August",
                Route.DETERMINISTIC,
                "compare",
                ("normalized_power",),
            ),
        ],
    )
    def test_table(self, question, expected_route, expected_intent, expected_metrics):
        plan = classify_rules(question, now=NOW)
        assert plan.route is expected_route
        assert plan.intent == expected_intent
        assert plan.metrics == expected_metrics

    def test_hrv_fans_out_rather_than_being_ambiguous(self):
        plan = classify_rules("what is my average hrv?", now=NOW)
        assert plan.metrics == ("hrv_rmssd", "hrv_sdnn")
        assert plan.ambiguities == ()

    def test_longest_alias_wins(self):
        plan = classify_rules("what was my max heart rate", now=NOW)
        assert plan.metrics == ("max_hr_observed",)

    def test_unrecognised_question_falls_back_to_hybrid(self):
        plan = classify_rules("tell me about the thing", now=NOW)
        assert plan.route is Route.HYBRID
        assert plan.confidence < 0.5

    def test_high_level_keywords_always_seed_evidence_terms(self):
        plan = classify_rules("average power in July", now=NOW)
        assert "signal quality" in plan.hl_keywords
        assert "withheld" in plan.hl_keywords


class TestWindowParsing:
    def test_explicit_iso_date(self):
        window, _, _ = parse_window("what happened on 2026-07-14?", now=NOW)
        assert iso(window[0]) == "2026-07-14"
        assert iso(window[1]) == "2026-07-15"

    def test_explicit_range(self):
        window, _, _ = parse_window("between 2026-07-01 and 2026-07-31", now=NOW)
        assert iso(window[0]) == "2026-07-01"
        assert iso(window[1]) == "2026-08-01"

    @pytest.mark.parametrize(
        "phrase,days",
        [
            ("last 7 days", 7),
            ("past 3 weeks", 21),
            ("last six weeks", 42),
            ("previous 2 months", 60),
        ],
    )
    def test_relative_windows(self, phrase, days):
        window, _, _ = parse_window(f"my rmssd over the {phrase}", now=NOW)
        assert window[1] == pytest.approx(NOW)
        assert window[0] == pytest.approx(NOW - days * DAY)

    def test_yesterday_and_today(self):
        window, _, _ = parse_window("what about yesterday", now=NOW)
        assert iso(window[0]) == "2026-09-01"
        window, _, _ = parse_window("how about today", now=NOW)
        assert iso(window[0]) == "2026-09-02"

    def test_named_month_assumes_the_current_year_and_says_so(self):
        window, ambiguities, _ = parse_window("average power in July", now=NOW)
        assert iso(window[0]) == "2026-07-01"
        assert iso(window[1]) == "2026-08-01"
        assert ambiguities[0].chosen == "2026"
        assert "named no year" in ambiguities[0].reason

    def test_named_month_with_year(self):
        window, ambiguities, _ = parse_window("average power in July 2025", now=NOW)
        assert iso(window[0]) == "2025-07-01"
        assert ambiguities == []

    def test_last_month(self):
        window, _, _ = parse_window("what about last month", now=NOW)
        assert iso(window[0]) == "2026-08-01"
        assert iso(window[1]) == "2026-09-01"

    def test_no_period_named(self):
        window, _, _ = parse_window("what is my average power", now=NOW)
        assert window is None

    def test_ordinal_resolves_when_the_data_allows_one_reading(self):
        span = (
            dt.datetime(2026, 8, 20, tzinfo=dt.timezone.utc).timestamp(),
            dt.datetime(2026, 8, 30, tzinfo=dt.timezone.utc).timestamp(),
        )
        window, ambiguities, _ = parse_window(
            "why was the 25th bad?", now=NOW, span=span
        )
        assert iso(window[0]) == "2026-08-25"
        assert ambiguities[0].chosen == "2026-08-25"

    def test_ordinal_stays_ambiguous_across_many_months(self):
        window, ambiguities, _ = parse_window("why was the 14th bad?", now=NOW)
        assert window is None
        assert ambiguities[0].chosen is None
        assert len(ambiguities[0].candidates) > 1


class TestAmbiguityHandling:
    def test_unresolvable_date_refuses_rather_than_guessing(self):
        plan = classify_rules("what was my average power on the 14th?", now=NOW)
        assert plan.route is Route.REFUSE
        assert plan.clarifying_question
        assert "Guessing" in plan.clarifying_question

    def test_a_relational_question_with_an_open_date_still_retrieves(self):
        # Refusing is only right when an exact number was requested.
        plan = classify_rules("why was the 14th a bad day?", now=NOW)
        assert plan.route is not Route.REFUSE


class _SpyLLM:
    def __init__(self, reply="{}"):
        self.reply = reply
        self.calls = 0

    async def __call__(self, prompt, **kwargs):
        self.calls += 1
        return self.reply


class TestLLMClassification:
    def test_not_called_when_the_rules_are_confident(self):
        spy = _SpyLLM()
        plan = asyncio.run(
            route("what was my average power in July?", now=NOW, llm_func=spy)
        )
        assert spy.calls == 0
        assert plan.route is Route.DETERMINISTIC

    def test_called_when_the_rules_are_unsure(self):
        spy = _SpyLLM('{"route": "retrieval", "intent": "explain", "metrics": []}')
        plan = asyncio.run(route("how did last week go?", now=NOW, llm_func=spy))
        assert spy.calls == 1
        assert plan.route is Route.RETRIEVAL

    def test_invented_metric_names_are_rejected(self):
        spy = _SpyLLM(
            '{"route": "deterministic", "intent": "aggregate", '
            '"metrics": ["body_battery", "hrv_rmssd"]}'
        )
        plan = asyncio.run(route("how did last week go?", now=NOW, llm_func=spy))
        assert "body_battery" not in plan.metrics
        assert "hrv_rmssd" in plan.metrics
        assert any("invented" in r for r in plan.reasons)

    def test_garbage_output_keeps_the_rule_classification(self):
        base = classify_rules("how did last week go?", now=NOW)
        plan = asyncio.run(
            route("how did last week go?", now=NOW, llm_func=_SpyLLM("not json"))
        )
        assert plan.route is base.route

    def test_a_failing_llm_is_not_fatal(self):
        async def explode(prompt, **kwargs):
            raise RuntimeError("model down")

        plan = asyncio.run(route("how did last week go?", now=NOW, llm_func=explode))
        assert isinstance(plan, QueryPlan)

    def test_llm_cannot_force_a_refusal(self):
        spy = _SpyLLM('{"route": "refuse"}')
        plan = asyncio.run(route("how did last week go?", now=NOW, llm_func=spy))
        assert plan.route is not Route.REFUSE


class TestKeywordSeeding:
    def test_session_ids_and_dates_are_seeded(self):
        plan = asyncio.run(
            route(
                "why was my power low in July?",
                now=NOW,
                session_ids=["ride-01", "ride-02"],
                dates=["2026-07-14"],
            )
        )
        assert "ride-01" in plan.ll_keywords
        assert "2026-07-14" in plan.ll_keywords
