"""Asking questions about your own physiology, and getting honest answers.

The engine here routes a question, answers the arithmetic part by computation
and the explanatory part by retrieval, and then checks what came back against
what the data supports.

The composition rule is the important one: **every number the user reads comes
from arithmetic; generation is confined to explanation.** For a hybrid question
the deterministic results are computed first and handed to the model as fixed
facts it must explain around rather than reproduce. That is the strongest
honesty guarantee available given that the retrieval layer underneath cannot be
scoped to a date range at all.

The engine works with no ``rag`` at all. Given only a
:class:`~.store.ReportStore`, every arithmetic question is answerable with no
language model and no network -- which is a genuinely useful offline mode, and
not incidentally the bulk of the test surface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from . import timeseries as ts
from .prompts import BIOSIGNAL_PROMPTS
from .router import QueryPlan, Route, route as route_question
from .store import ReportRecord, ReportStore
from .verify import (
    VerificationPolicy,
    Verdict,
    Violation,
    ViolationKind,
    annotate,
    refusal,
    verify,
    worst_action,
)

logger = logging.getLogger(__name__)

__all__ = ["BiosignalAnswer", "BiosignalQueryEngine", "aquery_biosignal"]

#: Fallback allowlist used only if the installed LightRAG cannot be inspected.
_KNOWN_QUERY_PARAM_FIELDS = frozenset(
    {
        "mode",
        "only_need_context",
        "only_need_prompt",
        "response_type",
        "stream",
        "top_k",
        "chunk_top_k",
        "max_entity_tokens",
        "max_relation_tokens",
        "max_total_tokens",
        "hl_keywords",
        "ll_keywords",
        "conversation_history",
        "history_turns",
        "model_func",
        "user_prompt",
        "enable_rerank",
        "include_references",
    }
)


def _query_param_fields() -> frozenset:
    """Fields the installed LightRAG's QueryParam actually accepts.

    Anything else raises ``TypeError`` at query time, so kwargs are filtered
    rather than passed hopefully. Read from the installed class so that a
    LightRAG upgrade widens what we can use without a code change here.
    """
    try:
        from lightrag.base import QueryParam  # noqa: PLC0415 - optional at import

        return frozenset(QueryParam.__dataclass_fields__)
    except Exception:  # noqa: BLE001 - LightRAG absent or restructured
        return _KNOWN_QUERY_PARAM_FIELDS


def _safe_query_kwargs(**kwargs: Any) -> Dict[str, Any]:
    allowed = _query_param_fields()
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in allowed:
            out[key] = value
        else:
            logger.debug("dropping query kwarg %r: not a QueryParam field", key)
    return out


@dataclass
class BiosignalAnswer:
    """An answer, the plan that produced it, and the check that vetted it."""

    question: str
    plan: QueryPlan
    answer: str
    deterministic: Any = None
    retrieval_used: bool = False
    context_out_of_scope: bool = False
    verdict: Optional[Verdict] = None
    withheld: Dict[str, str] = field(default_factory=dict)
    excluded_sessions: Dict[str, str] = field(default_factory=dict)
    sources: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "plan": self.plan.to_dict(),
            "answer": self.answer,
            "deterministic": (
                self.deterministic.to_dict()
                if hasattr(self.deterministic, "to_dict")
                else self.deterministic
            ),
            "retrieval_used": self.retrieval_used,
            "context_out_of_scope": self.context_out_of_scope,
            "verdict": self.verdict.to_dict() if self.verdict else None,
            "withheld": dict(self.withheld),
            "excluded_sessions": dict(self.excluded_sessions),
            "sources": list(self.sources),
        }


def _render_series(result: Any) -> str:
    """Deterministic results as plain text a model can be handed verbatim."""
    lines: List[str] = []
    if isinstance(result, ts.Aggregate):
        if result.value is None:
            lines.append(
                f"{result.metric} ({result.statistic}): not reported -- "
                f"{result.withheld_reason}"
            )
        else:
            lines.append(
                f"{result.metric} ({result.statistic}) = {result.value:.2f} "
                f"{result.unit or ''}".rstrip()
            )
            lines.append(f"  {result.note}")
    elif isinstance(result, ts.Trend):
        if result.withheld_reason:
            lines.append(
                f"{result.metric} trend: not determined -- {result.withheld_reason}"
            )
        else:
            lines.append(
                f"{result.metric} trend over {result.span_days:.0f} days "
                f"({result.n} sessions): {result.direction}"
            )
            if result.slope_per_day is not None:
                lines.append(
                    f"  slope = {result.slope_per_day:+.3f} {result.unit or ''} per day"
                )
            if result.ci95_slope is not None:
                lines.append(
                    f"  95% confidence interval on the slope: "
                    f"[{result.ci95_slope[0]:+.3f}, {result.ci95_slope[1]:+.3f}]"
                )
            if result.first_value is not None and result.last_value is not None:
                lines.append(
                    f"  first = {result.first_value:.2f}, last = {result.last_value:.2f}"
                )
        for note in result.notes:
            lines.append(f"  note: {note}")
    elif isinstance(result, ts.Comparison):
        lines.append(f"{result.metric}: {result.label_a} vs {result.label_b}")
        lines.append("  " + _render_series(result.a).replace("\n", "\n  "))
        lines.append("  " + _render_series(result.b).replace("\n", "\n  "))
        if result.difference is not None:
            lines.append(
                f"  difference = {result.difference:+.2f} ({result.direction})"
            )
    elif isinstance(result, dict):
        for key, value in result.items():
            lines.append(f"{key}:")
            lines.append("  " + _render_series(value).replace("\n", "\n  "))
    elif result is not None:
        lines.append(str(result))

    excluded = getattr(result, "excluded", None)
    if excluded:
        lines.append(f"  {len(excluded)} session(s) excluded:")
        for session_id, reason in list(excluded.items())[:8]:
            lines.append(f"    - {session_id}: {reason}")
    return "\n".join(lines)


class BiosignalQueryEngine:
    """Routes, computes, retrieves and verifies."""

    def __init__(
        self,
        rag: Any = None,
        store: Optional[ReportStore] = None,
        *,
        llm_func: Optional[Callable[..., Awaitable[str]]] = None,
        min_quality: float = 0.5,
        policy: Optional[Dict[ViolationKind, VerificationPolicy]] = None,
        strict_scope: bool = False,
        mode: str = "mix",
    ) -> None:
        if store is None and rag is not None:
            store = ReportStore.maybe_for_rag(rag)
        self.rag = rag
        self.store = store
        self.llm_func = llm_func
        self.min_quality = min_quality
        self.policy = policy
        self.strict_scope = strict_scope
        self.mode = mode

    # -- planning --------------------------------------------------------

    def _records(self, plan: QueryPlan) -> List[ReportRecord]:
        if self.store is None:
            return []
        if plan.window:
            return self.store.list(start=plan.window[0], end=plan.window[1])
        return self.store.list()

    # -- deterministic ---------------------------------------------------

    def _compute(self, plan: QueryPlan, records: Sequence[ReportRecord]) -> Any:
        if not plan.metrics:
            return None
        results: Dict[str, Any] = {}
        for metric in plan.metrics:
            series = ts.collect(
                records,
                metric,
                start=plan.window[0] if plan.window else None,
                end=plan.window[1] if plan.window else None,
                min_quality=self.min_quality,
            )
            if plan.intent == "trend":
                results[metric] = ts.trend(series)
            else:
                results[metric] = ts.aggregate(series, plan.statistic or "mean")
        if len(results) == 1:
            return next(iter(results.values()))
        return results

    def compute(self, question: str, now: Optional[float] = None) -> BiosignalAnswer:
        """Answer arithmetically, with no language model and no network.

        Available whenever a store exists. This is the path that cannot
        hallucinate, because nothing generative is involved at any point.
        """
        from .router import classify_rules

        span = self.store.span() if self.store is not None else None
        plan = classify_rules(question, now=now, span=span)
        records = self._records(plan)
        result = self._compute(plan, records)
        text = (
            _render_series(result)
            if result is not None
            else (
                "That question needs more than arithmetic over the stored sessions, "
                "and no retrieval backend was supplied."
            )
        )
        return BiosignalAnswer(
            question=question,
            plan=plan,
            answer=text,
            deterministic=result,
            retrieval_used=False,
            verdict=Verdict(ok=True, final_answer=text),
            withheld=_collect_withheld(records, plan.metrics),
            excluded_sessions=getattr(result, "excluded", {}) or {},
            sources=tuple(r.session_id for r in records),
        )

    # -- retrieval -------------------------------------------------------

    async def _retrieve(
        self, plan: QueryPlan, facts: str, records: Sequence[ReportRecord]
    ) -> Tuple[str, bool]:
        if self.rag is None:
            return "", False

        ensure = getattr(self.rag, "_ensure_lightrag_initialized", None)
        if ensure is not None:
            # aquery, unlike the other entry points, does not do this itself.
            await ensure()

        user_prompt_parts = [BIOSIGNAL_PROMPTS["CANONICAL_NUMBER_INSTRUCTION"]]
        if plan.window:
            import datetime as _dt

            user_prompt_parts.append(
                BIOSIGNAL_PROMPTS["SCOPE_USER_PROMPT"].format(
                    start=_dt.datetime.fromtimestamp(
                        plan.window[0], tz=_dt.timezone.utc
                    ).date(),
                    end=_dt.datetime.fromtimestamp(
                        plan.window[1], tz=_dt.timezone.utc
                    ).date(),
                )
            )
        if facts:
            user_prompt_parts.append(
                BIOSIGNAL_PROMPTS["DETERMINISTIC_FACTS_HEADER"] + "\n\n" + facts
            )

        kwargs = _safe_query_kwargs(
            user_prompt="\n\n".join(user_prompt_parts),
            hl_keywords=list(plan.hl_keywords) or None,
            ll_keywords=list(plan.ll_keywords) or None,
        )

        if self.strict_scope:
            return await self._retrieve_strict(plan, kwargs, records)

        answer = await self.rag.aquery(
            plan.raw_question,
            mode=plan.mode or self.mode,
            system_prompt=BIOSIGNAL_PROMPTS["ANSWER_SYSTEM_PROMPT"],
            **kwargs,
        )
        return answer, False

    async def _retrieve_strict(
        self, plan: QueryPlan, kwargs: Dict[str, Any], records: Sequence[ReportRecord]
    ) -> Tuple[str, bool]:
        """Two-phase: retrieve, drop out-of-window fragments, then generate.

        The pipeline underneath offers no way to scope retrieval to a document
        set, so the only real enforcement is to take the assembled prompt back
        and filter it here. The cost is a coupling to that prompt's shape, which
        is why this is opt-in rather than the default.
        """
        prompt = await self.rag.aquery(
            plan.raw_question,
            mode=plan.mode or self.mode,
            system_prompt=BIOSIGNAL_PROMPTS["ANSWER_SYSTEM_PROMPT"],
            **_safe_query_kwargs(only_need_prompt=True, **kwargs),
        )
        if not isinstance(prompt, str):
            return "", True

        in_scope = {r.session_id for r in records} | {r.iso_date() for r in records}
        blocks = prompt.split("\n\n")
        kept = [
            block
            for block in blocks
            if not _mentions_a_session(block) or any(k in block for k in in_scope)
        ]
        if not kept:
            # Never silently fall back to unfiltered context: an unscoped answer
            # presented as a scoped one is the failure this mode exists to stop.
            return "", True

        model = getattr(self.rag, "llm_model_func", None)
        if model is None:
            logger.warning(
                "strict_scope needs rag.llm_model_func to generate from filtered "
                "context; falling back to reporting the filtered context itself"
            )
            return "\n\n".join(kept), False
        answer = await model("\n\n".join(kept))
        return answer, False

    # -- the public entry point ------------------------------------------

    async def aask(
        self, question: str, *, now: Optional[float] = None
    ) -> BiosignalAnswer:
        """Route, answer, and verify."""
        span = self.store.span() if self.store is not None else None
        plan = await route_question(
            question, now=now, span=span, llm_func=self.llm_func
        )
        records = self._records(plan)
        if records:
            from dataclasses import replace

            plan = replace(
                plan,
                ll_keywords=tuple(
                    dict.fromkeys(
                        list(plan.ll_keywords)
                        + [r.session_id for r in records][:24]
                        + sorted({r.iso_date() for r in records})[:24]
                    )
                ),
            )

        if plan.route is Route.REFUSE:
            text = plan.clarifying_question or (
                "I need more detail before I can answer that precisely."
            )
            return BiosignalAnswer(
                question=question,
                plan=plan,
                answer=text,
                verdict=Verdict(ok=True, final_answer=text),
            )

        result = None
        facts = ""
        if plan.route in (Route.DETERMINISTIC, Route.HYBRID):
            result = self._compute(plan, records)
            facts = _render_series(result) if result is not None else ""

        if plan.route is Route.DETERMINISTIC:
            text = facts or "No stored session in that window produced a usable value."
            return BiosignalAnswer(
                question=question,
                plan=plan,
                answer=text,
                deterministic=result,
                verdict=Verdict(ok=True, final_answer=text),
                withheld=_collect_withheld(records, plan.metrics),
                excluded_sessions=getattr(result, "excluded", {}) or {},
                sources=tuple(r.session_id for r in records),
            )

        generated, out_of_scope = await self._retrieve(plan, facts, records)
        if not generated:
            text = facts or (
                "No retrieval backend is available and this question needs more "
                "than arithmetic over the stored sessions."
            )
            return BiosignalAnswer(
                question=question,
                plan=plan,
                answer=text,
                deterministic=result,
                retrieval_used=self.rag is not None,
                context_out_of_scope=out_of_scope,
                verdict=Verdict(ok=True, final_answer=text),
                withheld=_collect_withheld(records, plan.metrics),
                sources=tuple(r.session_id for r in records),
            )

        verdict = verify(
            generated,
            plan,
            records,
            deterministic=result,
            min_quality=self.min_quality,
            policy=self.policy,
        )
        final = await self._apply_policy(verdict, plan, records, facts, result)

        return BiosignalAnswer(
            question=question,
            plan=plan,
            answer=final,
            deterministic=result,
            retrieval_used=True,
            context_out_of_scope=out_of_scope,
            verdict=verdict,
            withheld=_collect_withheld(records, plan.metrics),
            excluded_sessions=getattr(result, "excluded", {}) or {},
            sources=tuple(r.session_id for r in records),
        )

    async def _apply_policy(
        self,
        verdict: Verdict,
        plan: QueryPlan,
        records: Sequence[ReportRecord],
        facts: str,
        result: Any,
    ) -> str:
        """Answer a violation according to its severity. At most one retry."""
        if verdict.ok:
            verdict.action = None
            return verdict.final_answer

        action = worst_action(verdict.violations, self.policy)
        verdict.action = action

        if action is VerificationPolicy.ANNOTATE:
            verdict.final_answer = annotate(verdict.final_answer, verdict.violations)
            return verdict.final_answer

        # Both REGENERATE and REFUSE get exactly one retry with the violations
        # and the computed facts fed back. Refusing outright without trying is
        # strictly worse for the user and no safer: the re-check below is what
        # actually enforces the guarantee, so a corrected answer is accepted on
        # its merits and an uncorrected one still refuses.
        if self.rag is not None:
            retry = await self._regenerate(plan, verdict.violations, facts)
            if retry:
                recheck = verify(
                    retry,
                    plan,
                    records,
                    deterministic=result,
                    min_quality=self.min_quality,
                    policy=self.policy,
                )
                verdict.notes.append(
                    f"regenerated once after {len(verdict.violations)} violation(s)"
                )
                if recheck.ok:
                    verdict.ok = True
                    verdict.violations = ()
                    verdict.final_answer = retry
                    return retry
                verdict.violations = recheck.violations
                verdict.notes.append("the regenerated answer still overstated the data")

        verdict.final_answer = refusal(verdict.violations, facts)
        verdict.action = VerificationPolicy.REFUSE
        return verdict.final_answer

    async def _regenerate(
        self, plan: QueryPlan, violations: Sequence[Violation], facts: str
    ) -> Optional[str]:
        described = "\n".join(f"- {v.describe()}" for v in violations)
        user_prompt = BIOSIGNAL_PROMPTS["REGENERATE_USER_PROMPT"].format(
            violations=described
        )
        if facts:
            user_prompt += (
                "\n\n"
                + BIOSIGNAL_PROMPTS["DETERMINISTIC_FACTS_HEADER"]
                + "\n\n"
                + facts
            )
        try:
            return await self.rag.aquery(
                plan.raw_question,
                mode=plan.mode or self.mode,
                system_prompt=BIOSIGNAL_PROMPTS["ANSWER_SYSTEM_PROMPT"],
                **_safe_query_kwargs(
                    user_prompt=user_prompt,
                    hl_keywords=list(plan.hl_keywords) or None,
                    ll_keywords=list(plan.ll_keywords) or None,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a retry failure falls through
            logger.warning("regeneration failed: %s", exc)
            return None


def _collect_withheld(
    records: Sequence[ReportRecord], metrics: Sequence[str]
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for record in records:
        for metric in metrics or record.withheld:
            reason = record.withheld_reason(metric)
            if reason:
                out.setdefault(f"{record.session_id}:{metric}", reason)
    return out


def _mentions_a_session(block: str) -> bool:
    lowered = block.lower()
    return "session" in lowered or "biosignal" in lowered


async def aquery_biosignal(rag: Any, question: str, **kwargs: Any) -> BiosignalAnswer:
    """One-shot convenience wrapper around :class:`BiosignalQueryEngine`."""
    now = kwargs.pop("now", None)
    engine = BiosignalQueryEngine(rag, **kwargs)
    return await engine.aask(question, now=now)
