"""Deciding how a question should be answered.

Some questions about your own body are arithmetic ("what was my average power
in July?"). Some are relational ("why was recovery poor on the 14th?"). Sending
the first kind to a retriever produces a plausible number pulled out of a
retrieved table; sending the second to a statistics function produces nothing
at all. So the question is classified first.

Classification is rules-first and deliberately boring: compiled lexicons over
metric names, statistic words, trend words and relational words, plus a date
parser. That makes it free, instant, and exhaustively testable. An optional
language model runs only when the rules are unsure, and everything it returns
is validated against the known metric names -- it is allowed to choose, never
to invent.

The fallback when nothing is certain is ``HYBRID``, not one of the specific
routes. Defaulting to deterministic would invent a scope; defaulting to
retrieval would throw away the arithmetic. Hybrid runs both and lets the
verifier arbitrate.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

from .timeseries import METRIC_SUPPORT, STATISTICS

logger = logging.getLogger(__name__)

__all__ = [
    "Ambiguity",
    "QueryPlan",
    "Route",
    "classify_llm",
    "classify_rules",
    "parse_window",
    "route",
]

DAY = 86400.0


class Route(str, Enum):
    DETERMINISTIC = "deterministic"
    RETRIEVAL = "retrieval"
    HYBRID = "hybrid"
    REFUSE = "refuse"


Intent = Literal[
    "aggregate", "trend", "compare", "lookup", "explain", "relate", "unknown"
]


@dataclass(frozen=True)
class Ambiguity:
    """Something the question left open, and what was done about it."""

    field: str
    candidates: Tuple[str, ...]
    chosen: Optional[str]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "candidates": list(self.candidates),
            "chosen": self.chosen,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QueryPlan:
    """How one question will be answered."""

    raw_question: str
    route: Route
    intent: Intent = "unknown"
    metrics: Tuple[str, ...] = ()
    statistic: Optional[str] = None
    window: Optional[Tuple[float, float]] = None
    group_by: Optional[str] = None
    hl_keywords: Tuple[str, ...] = ()
    ll_keywords: Tuple[str, ...] = ()
    mode: str = "mix"
    confidence: float = 0.0
    ambiguities: Tuple[Ambiguity, ...] = ()
    reasons: Tuple[str, ...] = ()
    clarifying_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.raw_question,
            "route": self.route.value,
            "intent": self.intent,
            "metrics": list(self.metrics),
            "statistic": self.statistic,
            "window": list(self.window) if self.window else None,
            "group_by": self.group_by,
            "hl_keywords": list(self.hl_keywords),
            "ll_keywords": list(self.ll_keywords),
            "mode": self.mode,
            "confidence": self.confidence,
            "ambiguities": [a.to_dict() for a in self.ambiguities],
            "reasons": list(self.reasons),
            "clarifying_question": self.clarifying_question,
        }


# --------------------------------------------------------------------------
# lexicons
# --------------------------------------------------------------------------

#: Phrases a person actually uses -> canonical metric names. A phrase may map
#: to several metrics, in which case the question fans out rather than being
#: treated as ambiguous.
METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "rmssd": ("hrv_rmssd",),
    "sdnn": ("hrv_sdnn",),
    "pnn50": ("hrv_pnn50",),
    "mean rr": ("hrv_mean_rr",),
    "rr interval": ("hrv_mean_rr",),
    "beat interval": ("hrv_mean_rr",),
    "beat-to-beat heart rate": ("hrv_mean_hr",),
    "hrv heart rate": ("hrv_mean_hr",),
    "hrv": ("hrv_rmssd", "hrv_sdnn"),
    "heart rate variability": ("hrv_rmssd", "hrv_sdnn"),
    "resting heart rate": ("mean_hr",),
    "average heart rate": ("mean_hr",),
    "mean heart rate": ("mean_hr",),
    "heart rate": ("mean_hr",),
    "max heart rate": ("max_hr_observed",),
    "maximum heart rate": ("max_hr_observed",),
    "peak heart rate": ("max_hr_observed",),
    "highest heart rate": ("max_hr_observed",),
    "hr": ("mean_hr",),
    "bpm": ("mean_hr",),
    "trimp": ("trimp",),
    "training load": ("trimp",),
    "training impulse": ("trimp",),
    "normalised power": ("normalized_power",),
    "normalized power": ("normalized_power",),
    "np": ("normalized_power",),
    "average power": ("mean_power",),
    "mean power": ("mean_power",),
    "power": ("mean_power",),
    "watts": ("mean_power",),
    "intensity factor": ("intensity_factor",),
    "if": ("intensity_factor",),
    "training stress": ("training_stress",),
    "tss": ("training_stress",),
    "decoupling": ("aerobic_decoupling_pct",),
    "aerobic decoupling": ("aerobic_decoupling_pct",),
    "cardiac drift": ("aerobic_decoupling_pct",),
}

_STATISTIC_MARKERS: Dict[str, Tuple[str, ...]] = {
    "mean": ("average", "mean", "typical", "on average"),
    "median": ("median",),
    "max": ("max", "maximum", "highest", "peak", "best"),
    "min": ("min", "minimum", "lowest", "worst"),
    "sum": ("total", "sum", "altogether", "cumulative"),
    "count": ("how many", "number of", "count"),
    "stdev": ("variance", "variability of", "spread", "standard deviation"),
    "last": ("most recent", "latest", "last one"),
    "first": ("earliest", "first one"),
}

_TREND_MARKERS = (
    "trend",
    "trending",
    "going up",
    "going down",
    "improving",
    "declining",
    "getting worse",
    "getting better",
    "worsening",
    "week over week",
    "progressing",
    "drifting",
    "rising",
    "falling",
    "dropping",
)
# "over the last N weeks" is deliberately absent: it states a window, not a
# direction. Treating it as a trend marker turned "what was my average heart
# rate over the last six weeks?" into a slope fit instead of a mean.

_RELATIONAL_MARKERS = (
    "why",
    "because",
    "explain",
    "cause",
    "reason",
    "what happened",
    "how come",
    "affect",
    "impact",
    "relate",
    "relationship",
    "correlate",
    "after",
    "before",
    "which device",
    "disagree",
    "wrong with",
    "should i",
)

_COMPARE_MARKERS = ("compare", "versus", " vs ", "against", "difference between")

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


def _compile_alias_pattern() -> re.Pattern:
    # Longest first so "max heart rate" wins over "heart rate".
    alternatives = sorted(METRIC_ALIASES, key=len, reverse=True)
    return re.compile(
        r"\b(" + "|".join(re.escape(a) for a in alternatives) + r")\b", re.I
    )


_ALIAS_RE = _compile_alias_pattern()
_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_LAST_N_RE = re.compile(
    r"\b(?:last|past|previous)\s+(\d+|a|an|one|two|three|four|five|six)\s+"
    r"(day|days|week|weeks|month|months|year|years)\b",
    re.I,
)
_ORDINAL_RE = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.I)
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b(?:\s+(\d{4}))?", re.I)

_WORD_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def _utc(dt: _dt.datetime) -> float:
    return dt.replace(tzinfo=_dt.timezone.utc).timestamp()


def _day_bounds(date: _dt.date) -> Tuple[float, float]:
    start = _dt.datetime(date.year, date.month, date.day)
    return _utc(start), _utc(start + _dt.timedelta(days=1))


# --------------------------------------------------------------------------
# date parsing
# --------------------------------------------------------------------------


def parse_window(
    question: str,
    now: Optional[float] = None,
    span: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[Tuple[float, float]], List[Ambiguity], List[str]]:
    """Resolve a time window from natural language.

    ``now`` is injectable so relative phrases resolve deterministically in
    tests. ``span`` is the extent of the stored data, used to disambiguate a
    bare ordinal like "the 14th": a date only counts if there is data near it.

    Returns ``(window, ambiguities, reasons)``. A ``None`` window means the
    question named no period, which is not an error -- it usually means "all of
    it".
    """
    text = question.lower()
    reference = (
        now if now is not None else _dt.datetime.now(tz=_dt.timezone.utc).timestamp()
    )
    ref_dt = _dt.datetime.fromtimestamp(reference, tz=_dt.timezone.utc)
    ambiguities: List[Ambiguity] = []
    reasons: List[str] = []

    isos = _ISO_RE.findall(question)
    if len(isos) >= 2:
        first = _dt.date(*map(int, isos[0]))
        second = _dt.date(*map(int, isos[1]))
        lo, _ = _day_bounds(min(first, second))
        _, hi = _day_bounds(max(first, second))
        reasons.append(f"explicit date range {first} to {second}")
        return (lo, hi), ambiguities, reasons
    if len(isos) == 1:
        date = _dt.date(*map(int, isos[0]))
        reasons.append(f"explicit date {date}")
        return _day_bounds(date), ambiguities, reasons

    match = _LAST_N_RE.search(text)
    if match:
        raw, unit = match.group(1), match.group(2).rstrip("s")
        count = _WORD_NUMBERS.get(raw, None)
        if count is None:
            count = int(raw)
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * count
        reasons.append(f"relative window: the last {count} {unit}(s)")
        return (reference - days * DAY, reference), ambiguities, reasons

    if "yesterday" in text:
        date = (ref_dt - _dt.timedelta(days=1)).date()
        reasons.append("yesterday")
        return _day_bounds(date), ambiguities, reasons
    if "today" in text:
        reasons.append("today")
        return _day_bounds(ref_dt.date()), ambiguities, reasons
    if "this week" in text:
        start = ref_dt.date() - _dt.timedelta(days=ref_dt.weekday())
        reasons.append("this week")
        return (_day_bounds(start)[0], reference), ambiguities, reasons
    if "last week" in text:
        this_monday = ref_dt.date() - _dt.timedelta(days=ref_dt.weekday())
        start = this_monday - _dt.timedelta(days=7)
        reasons.append("last week")
        return (
            (_day_bounds(start)[0], _day_bounds(this_monday)[0]),
            ambiguities,
            reasons,
        )
    if "this month" in text:
        start = ref_dt.date().replace(day=1)
        reasons.append("this month")
        return (_day_bounds(start)[0], reference), ambiguities, reasons
    if "last month" in text:
        first_this = ref_dt.date().replace(day=1)
        last_month_end = first_this
        start = (first_this - _dt.timedelta(days=1)).replace(day=1)
        reasons.append("last month")
        return (
            (_day_bounds(start)[0], _day_bounds(last_month_end)[0]),
            ambiguities,
            reasons,
        )

    month_match = _MONTH_RE.search(text)
    if month_match:
        month = _MONTHS[month_match.group(1).lower()]
        year = int(month_match.group(2)) if month_match.group(2) else ref_dt.year
        start = _dt.date(year, month, 1)
        end = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
        if month_match.group(2) is None:
            ambiguities.append(
                Ambiguity(
                    field="window",
                    candidates=(str(year), str(year - 1)),
                    chosen=str(year),
                    reason=(f"'{month_match.group(1)}' named no year; assuming {year}"),
                )
            )
        reasons.append(f"month window {start} to {end}")
        return (_day_bounds(start)[0], _day_bounds(end)[0]), ambiguities, reasons

    ordinal = _ORDINAL_RE.search(text)
    if ordinal:
        day = int(ordinal.group(1))
        candidates: List[_dt.date] = []
        probe = ref_dt.date().replace(day=1)
        for _ in range(12):
            try:
                candidate = probe.replace(day=day)
            except ValueError:
                candidate = None
            if candidate is not None and candidate <= ref_dt.date():
                if span is None:
                    candidates.append(candidate)
                else:
                    lo, hi = _day_bounds(candidate)
                    if lo <= span[1] and hi >= span[0]:
                        candidates.append(candidate)
            probe = (probe - _dt.timedelta(days=1)).replace(day=1)
        if len(candidates) == 1:
            chosen = candidates[0]
            ambiguities.append(
                Ambiguity(
                    field="window",
                    candidates=(str(chosen),),
                    chosen=str(chosen),
                    reason=(
                        f"'the {day}th' matched exactly one date with data: {chosen}"
                    ),
                )
            )
            reasons.append(f"resolved 'the {day}th' to {chosen}")
            return _day_bounds(chosen), ambiguities, reasons
        if len(candidates) > 1:
            ambiguities.append(
                Ambiguity(
                    field="window",
                    candidates=tuple(str(c) for c in candidates[:6]),
                    chosen=None,
                    reason=(
                        f"'the {day}th' could mean any of "
                        f"{', '.join(str(c) for c in candidates[:6])}"
                    ),
                )
            )
            return None, ambiguities, reasons

    return None, ambiguities, reasons


# --------------------------------------------------------------------------
# rule classification
# --------------------------------------------------------------------------


def _find_metrics(question: str) -> Tuple[Tuple[str, ...], List[str]]:
    found: List[str] = []
    reasons: List[str] = []
    for match in _ALIAS_RE.finditer(question):
        alias = match.group(1).lower()
        for metric in METRIC_ALIASES[alias]:
            if metric not in found:
                found.append(metric)
        reasons.append(f"matched '{alias}'")
    return tuple(found), reasons


def _find_statistic(text: str) -> Optional[str]:
    for statistic, markers in _STATISTIC_MARKERS.items():
        if any(marker in text for marker in markers):
            return statistic
    return None


def classify_rules(
    question: str,
    now: Optional[float] = None,
    span: Optional[Tuple[float, float]] = None,
) -> QueryPlan:
    """Classify without a language model. Free, instant, deterministic."""
    text = question.lower()
    metrics, metric_reasons = _find_metrics(question)
    statistic = _find_statistic(text)
    is_trend = any(marker in text for marker in _TREND_MARKERS)
    is_relational = any(marker in text for marker in _RELATIONAL_MARKERS)
    is_compare = any(marker in text for marker in _COMPARE_MARKERS)
    window, ambiguities, window_reasons = parse_window(question, now=now, span=span)

    reasons = metric_reasons + window_reasons
    numeric = bool(metrics) and (statistic is not None or is_trend or is_compare)

    if is_trend:
        intent: Intent = "trend"
    elif is_compare:
        intent = "compare"
    elif statistic is not None:
        intent = "aggregate"
    elif is_relational:
        intent = "explain"
    elif metrics:
        intent = "lookup"
    else:
        intent = "unknown"

    if numeric and not is_relational:
        route, confidence = Route.DETERMINISTIC, 0.9
    elif numeric and is_relational:
        route, confidence = Route.HYBRID, 0.8
    elif metrics and is_relational:
        route, confidence = Route.HYBRID, 0.7
    elif metrics:
        route, confidence = Route.DETERMINISTIC, 0.6
    elif is_relational:
        route, confidence = Route.RETRIEVAL, 0.7
    elif is_trend or statistic is not None:
        route, confidence = Route.HYBRID, 0.4
        reasons.append("a statistic was asked for but no metric was named")
    else:
        route, confidence = Route.HYBRID, 0.2
        reasons.append("no metric, statistic or relational marker matched")

    unresolved = [a for a in ambiguities if a.chosen is None]
    clarifying = None
    if unresolved and route is Route.DETERMINISTIC:
        route = Route.REFUSE
        confidence = 0.0
        clarifying = (
            "Which date did you mean? "
            + ", ".join(unresolved[0].candidates)
            + ". Guessing would give you an exact-looking number for the wrong day."
        )

    hl = tuple(
        dict.fromkeys(list(metrics) + ["signal quality", "withheld", "provenance"])
    )
    ll: Tuple[str, ...] = ()

    return QueryPlan(
        raw_question=question,
        route=route,
        intent=intent,
        metrics=metrics,
        statistic=statistic or ("mean" if intent == "aggregate" else None),
        window=window,
        hl_keywords=hl,
        ll_keywords=ll,
        mode="mix",
        confidence=confidence,
        ambiguities=tuple(ambiguities),
        reasons=tuple(reasons),
        clarifying_question=clarifying,
    )


# --------------------------------------------------------------------------
# optional LLM classification
# --------------------------------------------------------------------------


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """First balanced JSON object in a model reply, or ``None``."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


async def classify_llm(
    question: str,
    llm_func: Callable[..., Awaitable[str]],
    base: QueryPlan,
) -> QueryPlan:
    """Refine a low-confidence plan with a language model.

    Every field the model returns is validated against the known enums and
    metric names. Anything unrecognised is dropped rather than accepted: the
    model is allowed to choose among the metrics this system computes, never to
    name one it does not.
    """
    from .prompts import BIOSIGNAL_PROMPTS

    prompt = BIOSIGNAL_PROMPTS["ROUTER_CLASSIFY"].format(
        question=question, metrics=", ".join(sorted(METRIC_SUPPORT))
    )
    try:
        raw = await llm_func(prompt)
    except Exception as exc:  # noqa: BLE001 - a router failure must not be fatal
        logger.warning("router LLM call failed, keeping rule classification: %s", exc)
        return base

    parsed = _extract_json(raw or "")
    if not parsed:
        logger.debug("router LLM returned no parseable JSON; keeping rules")
        return base

    reasons = list(base.reasons)
    route = base.route
    raw_route = str(parsed.get("route", "")).lower()
    if raw_route in {r.value for r in Route} and raw_route != Route.REFUSE.value:
        route = Route(raw_route)
        reasons.append(f"language model chose route '{raw_route}'")

    intent = base.intent
    raw_intent = str(parsed.get("intent", "")).lower()
    if raw_intent in {
        "aggregate",
        "trend",
        "compare",
        "lookup",
        "explain",
        "relate",
    }:
        intent = raw_intent  # type: ignore[assignment]

    metrics = base.metrics
    raw_metrics = parsed.get("metrics")
    if isinstance(raw_metrics, list):
        validated = tuple(
            m for m in raw_metrics if isinstance(m, str) and m in METRIC_SUPPORT
        )
        rejected = [
            m for m in raw_metrics if isinstance(m, str) and m not in METRIC_SUPPORT
        ]
        if rejected:
            reasons.append(
                "ignored metric name(s) the model invented: " + ", ".join(rejected)
            )
        if validated:
            metrics = tuple(dict.fromkeys(base.metrics + validated))

    statistic = base.statistic
    raw_statistic = parsed.get("statistic")
    if isinstance(raw_statistic, str) and raw_statistic in STATISTICS:
        statistic = raw_statistic

    hl = tuple(
        dict.fromkeys(list(metrics) + ["signal quality", "withheld", "provenance"])
    )

    return QueryPlan(
        raw_question=base.raw_question,
        route=route,
        intent=intent,
        metrics=metrics,
        statistic=statistic,
        window=base.window,
        group_by=base.group_by,
        hl_keywords=hl,
        ll_keywords=base.ll_keywords,
        mode=base.mode,
        confidence=max(base.confidence, 0.65),
        ambiguities=base.ambiguities,
        reasons=tuple(reasons),
        clarifying_question=base.clarifying_question,
    )


async def route(
    question: str,
    *,
    now: Optional[float] = None,
    span: Optional[Tuple[float, float]] = None,
    llm_func: Optional[Callable[..., Awaitable[str]]] = None,
    llm_threshold: float = 0.5,
    session_ids: Optional[Sequence[str]] = None,
    dates: Optional[Sequence[str]] = None,
) -> QueryPlan:
    """Full classification: rules, then optionally a model, then keyword seeding."""
    plan = classify_rules(question, now=now, span=span)
    if plan.route is not Route.REFUSE and plan.confidence < llm_threshold and llm_func:
        plan = await classify_llm(question, llm_func, plan)

    if session_ids or dates:
        # Retrieval cannot be scoped to a document set, so the session ids and
        # dates in range are seeded as low-level keywords instead. It is
        # steering, not filtering -- the verifier is what actually enforces
        # scope.
        ll = tuple(
            dict.fromkeys(
                list(plan.ll_keywords) + list(session_ids or ()) + list(dates or ())
            )
        )
        plan = replace(plan, ll_keywords=ll)
    return plan
