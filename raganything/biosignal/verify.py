"""Checking a generated answer against what the data actually supports.

The rest of this subsystem is careful to record, for every metric it could not
compute to a standard worth reporting, that it was withheld and why. All of
that care is undone if a language model reads a number out of a retrieved table
and states it anyway.

So answers are checked before they are returned. The check is deliberately
mechanical -- regular expressions over a closed lexicon of metric names, not a
second model -- for one reason: **a verifier that can hallucinate is not a
verifier.** This one can only ever find claims it can match, which makes its
misses predictable (documented below) rather than arbitrary.

What it cannot do is catch every paraphrase. Two things bound that gap: the
answering prompt instructs the model to state numbers in canonical
``metric = value unit`` form, and any numeric claim the checker cannot resolve
is reported as :attr:`ViolationKind.UNVERIFIABLE` rather than passing silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .router import METRIC_ALIASES, QueryPlan
from .store import ReportRecord
from .timeseries import METRIC_SUPPORT, METRIC_UNITS, Trend, collect, trend

__all__ = [
    "Claim",
    "VerificationPolicy",
    "Verdict",
    "Violation",
    "ViolationKind",
    "extract_claims",
    "verify",
]


class ViolationKind(str, Enum):
    WITHHELD_METRIC_ASSERTED = "withheld_metric_asserted"
    UNGATED_TREND_ASSERTED = "ungated_trend_asserted"
    VALUE_CONTRADICTS_COMPUTATION = "value_contradicts_computation"
    OUT_OF_SCOPE_SESSION_CITED = "out_of_scope_session_cited"
    UNSUPPORTED_METRIC_NAME = "unsupported_metric_name"
    UNVERIFIABLE = "unverifiable"


class VerificationPolicy(str, Enum):
    ANNOTATE = "annotate"
    REGENERATE = "regenerate"
    REFUSE = "refuse"


#: How each kind of violation is answered. Withheld values and contradicted
#: numbers are worth another generation attempt; a cited out-of-scope session is
#: a note, not a reason to throw the answer away.
DEFAULT_POLICY: Dict[ViolationKind, VerificationPolicy] = {
    ViolationKind.WITHHELD_METRIC_ASSERTED: VerificationPolicy.REGENERATE,
    ViolationKind.VALUE_CONTRADICTS_COMPUTATION: VerificationPolicy.REGENERATE,
    ViolationKind.UNGATED_TREND_ASSERTED: VerificationPolicy.REFUSE,
    ViolationKind.UNSUPPORTED_METRIC_NAME: VerificationPolicy.ANNOTATE,
    ViolationKind.OUT_OF_SCOPE_SESSION_CITED: VerificationPolicy.ANNOTATE,
    ViolationKind.UNVERIFIABLE: VerificationPolicy.ANNOTATE,
}

#: (absolute, relative) tolerance when comparing a stated number to a computed
#: one. Generous enough to permit sensible rounding, tight enough that a
#: different number is caught.
ROUNDING_TOLERANCE: Dict[str, Tuple[float, float]] = {
    "hrv_rmssd": (0.05, 0.01),
    "hrv_sdnn": (0.05, 0.01),
    "hrv_pnn50": (0.05, 0.01),
    "hrv_mean_rr": (0.5, 0.01),
    "hrv_mean_hr": (0.5, 0.01),
    "mean_hr": (0.5, 0.01),
    "max_hr_observed": (0.5, 0.01),
    "trimp": (0.5, 0.02),
    "mean_power": (0.5, 0.01),
    "normalized_power": (0.5, 0.01),
    "intensity_factor": (0.005, 0.01),
    "training_stress": (0.5, 0.02),
    "aerobic_decoupling_pct": (0.05, 0.02),
}
_DEFAULT_TOLERANCE = (0.5, 0.02)

#: Metrics people ask about that this system does not compute. Asserting a
#: value for one of these is a fabrication, not a retrieval.
UNSUPPORTED_METRICS = (
    "recovery score",
    "readiness",
    "readiness score",
    "body battery",
    "sleep score",
    "strain",
    "fitness age",
    "vo2 max",
    "vo2max",
    "stress score",
    "training readiness",
)

_RISING = (
    "rising",
    "increasing",
    "improving",
    "going up",
    "climbing",
    "higher",
    "up by",
)
_FALLING = (
    "falling",
    "declining",
    "decreasing",
    "dropping",
    "worsening",
    "going down",
    "lower",
    "down by",
    "deteriorating",
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SESSION_ID_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*[-_][A-Za-z0-9][A-Za-z0-9._-]*\b")

_SPELLED = {
    "zero": 0.0,
    "ten": 10.0,
    "twenty": 20.0,
    "thirty": 30.0,
    "forty": 40.0,
    "fifty": 50.0,
    "sixty": 60.0,
    "seventy": 70.0,
    "eighty": 80.0,
    "ninety": 90.0,
    "a hundred": 100.0,
    "one hundred": 100.0,
}

_CANONICAL_NAMES = {m: m for m in METRIC_SUPPORT}
_ALL_ALIASES: Dict[str, Tuple[str, ...]] = {
    **{alias: metrics for alias, metrics in METRIC_ALIASES.items()},
    **{name: (name,) for name in METRIC_SUPPORT},
}
_ALIAS_RE = re.compile(
    r"\b("
    + "|".join(re.escape(a) for a in sorted(_ALL_ALIASES, key=len, reverse=True))
    + r")\b",
    re.I,
)
_UNSUPPORTED_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in UNSUPPORTED_METRICS) + r")\b", re.I
)


@dataclass(frozen=True)
class Claim:
    """One assertion found in an answer."""

    metric: str
    value: Optional[float]
    unit: Optional[str]
    kind: str  # "point" | "aggregate" | "trend"
    direction: Optional[str] = None
    session_id: Optional[str] = None
    date: Optional[str] = None
    sentence: str = ""
    span: Tuple[int, int] = (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "kind": self.kind,
            "direction": self.direction,
            "session_id": self.session_id,
            "date": self.date,
            "sentence": self.sentence,
        }


@dataclass(frozen=True)
class Violation:
    kind: ViolationKind
    claim: Claim
    expected: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "claim": self.claim.to_dict(),
            "expected": self.expected,
            "detail": self.detail,
        }

    def describe(self) -> str:
        return f"[{self.kind.value}] {self.detail} Expected: {self.expected}"


@dataclass
class Verdict:
    ok: bool
    violations: Tuple[Violation, ...] = ()
    claims: Tuple[Claim, ...] = ()
    final_answer: str = ""
    action: Optional[VerificationPolicy] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
            "claims": [c.to_dict() for c in self.claims],
            "action": self.action.value if self.action else None,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def _sentences(text: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    offset = 0
    for part in _SENTENCE_RE.split(text):
        if part.strip():
            index = text.find(part, offset)
            out.append((part, index if index >= 0 else offset))
            offset = (index if index >= 0 else offset) + len(part)
    return out


def _direction_in(sentence: str) -> Optional[str]:
    lowered = sentence.lower()
    if any(marker in lowered for marker in _FALLING):
        return "falling"
    if any(marker in lowered for marker in _RISING):
        return "rising"
    return None


def extract_claims(answer: str) -> List[Claim]:
    """Find every checkable assertion in an answer.

    Works sentence by sentence so that a metric named in one sentence can never
    bind to a number that appears in another -- a cross-sentence match would
    manufacture claims the model never made, and a verifier that invents
    violations is as bad as one that misses them.
    """
    claims: List[Claim] = []

    for sentence, offset in _sentences(answer):
        lowered = sentence.lower()
        date_match = _ISO_DATE_RE.search(sentence)
        date = date_match.group(0) if date_match else None
        direction = _direction_in(sentence)

        session_id = None
        for candidate in _SESSION_ID_RE.findall(sentence):
            if candidate.lower() not in _ALL_ALIASES and not _ISO_DATE_RE.match(
                candidate
            ):
                session_id = candidate
                break

        for unsupported in _UNSUPPORTED_RE.finditer(sentence):
            numbers = _NUMBER_RE.findall(sentence)
            if numbers:
                claims.append(
                    Claim(
                        metric=unsupported.group(1).lower(),
                        value=float(numbers[0]),
                        unit=None,
                        kind="unsupported",
                        session_id=session_id,
                        date=date,
                        sentence=sentence,
                        span=(offset, offset + len(sentence)),
                    )
                )

        for match in _ALIAS_RE.finditer(sentence):
            alias = match.group(1).lower()
            metrics = _ALL_ALIASES[alias]
            local_start = match.end()
            window = sentence[max(0, match.start() - 30) : local_start + 60]

            value: Optional[float] = None
            numbers = _NUMBER_RE.findall(window)
            if numbers:
                value = float(numbers[0])
            else:
                for word, spelled in _SPELLED.items():
                    if word in window.lower():
                        value = spelled
                        break

            for metric in metrics:
                if value is not None:
                    kind = "aggregate" if _is_aggregate(lowered) else "point"
                    claims.append(
                        Claim(
                            metric=metric,
                            value=value,
                            unit=METRIC_UNITS.get(metric),
                            kind=kind,
                            direction=direction,
                            session_id=session_id,
                            date=date,
                            sentence=sentence,
                            span=(offset, offset + len(sentence)),
                        )
                    )
                if direction is not None:
                    claims.append(
                        Claim(
                            metric=metric,
                            value=None,
                            unit=METRIC_UNITS.get(metric),
                            kind="trend",
                            direction=direction,
                            session_id=session_id,
                            date=date,
                            sentence=sentence,
                            span=(offset, offset + len(sentence)),
                        )
                    )
    return claims


def _is_aggregate(lowered: str) -> bool:
    return any(
        word in lowered
        for word in ("average", "mean", "median", "typical", "across", "overall")
    )


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------


def _within_tolerance(metric: str, stated: float, computed: float) -> bool:
    atol, rtol = ROUNDING_TOLERANCE.get(metric, _DEFAULT_TOLERANCE)
    return abs(stated - computed) <= max(atol, rtol * abs(computed))


def verify(
    answer: str,
    plan: QueryPlan,
    records: Sequence[ReportRecord],
    *,
    deterministic: Any = None,
    min_quality: float = 0.5,
    policy: Optional[Dict[ViolationKind, VerificationPolicy]] = None,
) -> Verdict:
    """Check an answer against the records it was supposed to be drawn from."""
    policy = policy or DEFAULT_POLICY
    claims = extract_claims(answer)
    violations: List[Violation] = []
    notes: List[str] = []

    in_scope = list(records)
    scope_ids = {r.session_id for r in in_scope}
    scope_dates = {r.iso_date() for r in in_scope}

    for claim in claims:
        if claim.kind == "unsupported":
            violations.append(
                Violation(
                    kind=ViolationKind.UNSUPPORTED_METRIC_NAME,
                    claim=claim,
                    expected=(
                        "this system does not compute "
                        f"'{claim.metric}'; it computes "
                        f"{', '.join(sorted(METRIC_SUPPORT))}"
                    ),
                    detail=(
                        f"the answer states a value for '{claim.metric}', which is "
                        "not a metric this system produces"
                    ),
                )
            )
            continue

        # Which records could this claim be about?
        if claim.kind == "trend":
            # A trend is an assertion about the whole window, so it is checked
            # against every in-scope session regardless of what date the
            # sentence happens to mention.
            candidates = in_scope
        elif claim.session_id and claim.session_id in scope_ids:
            candidates = [r for r in in_scope if r.session_id == claim.session_id]
        elif claim.date:
            candidates = [r for r in in_scope if r.iso_date() == claim.date]
        else:
            candidates = in_scope

        if claim.session_id and claim.session_id not in scope_ids:
            violations.append(
                Violation(
                    kind=ViolationKind.OUT_OF_SCOPE_SESSION_CITED,
                    claim=claim,
                    expected=f"sessions in scope: {', '.join(sorted(scope_ids)) or 'none'}",
                    detail=(
                        f"the answer cites session '{claim.session_id}', which is "
                        "not in the window this question was scoped to"
                    ),
                )
            )
        elif claim.date and claim.date not in scope_dates and scope_dates:
            violations.append(
                Violation(
                    kind=ViolationKind.OUT_OF_SCOPE_SESSION_CITED,
                    claim=claim,
                    expected=f"dates in scope: {', '.join(sorted(scope_dates))}",
                    detail=(
                        f"the answer cites {claim.date}, which has no session in "
                        "the window this question was scoped to"
                    ),
                )
            )

        if not candidates:
            violations.append(
                Violation(
                    kind=ViolationKind.UNVERIFIABLE,
                    claim=claim,
                    expected="a session this claim could be checked against",
                    detail=(
                        f"the answer states a {claim.metric} value with no session "
                        "in scope to check it against"
                    ),
                )
            )
            continue

        withheld = [r for r in candidates if r.is_withheld(claim.metric)]
        reported = [r for r in candidates if r.metric(claim.metric) is not None]

        if withheld and not reported:
            violations.append(
                Violation(
                    kind=ViolationKind.WITHHELD_METRIC_ASSERTED,
                    claim=claim,
                    expected=withheld[0].withheld_reason(claim.metric) or "withheld",
                    detail=(
                        f"the answer asserts {claim.metric}"
                        + (f" = {claim.value}" if claim.value is not None else "")
                        + ", but it was withheld for every session in scope"
                    ),
                )
            )
            continue

        if claim.kind == "trend" and claim.direction:
            series = collect(
                candidates,
                claim.metric,
                start=plan.window[0] if plan.window else None,
                end=plan.window[1] if plan.window else None,
                min_quality=min_quality,
            )
            computed = trend(series)
            if computed.direction != claim.direction:
                violations.append(
                    Violation(
                        kind=ViolationKind.UNGATED_TREND_ASSERTED,
                        claim=claim,
                        expected=_describe_trend(computed),
                        detail=(
                            f"the answer says {claim.metric} is {claim.direction}, "
                            f"but the computed trend is '{computed.direction}'"
                        ),
                    )
                )
            continue

        if claim.value is not None and reported:
            values = [r.metric(claim.metric) for r in reported]
            values = [v for v in values if v is not None]
            matches_a_session = any(
                _within_tolerance(claim.metric, claim.value, v) for v in values
            )
            mean_value = sum(values) / len(values)
            matches_mean = _within_tolerance(claim.metric, claim.value, mean_value)
            if not matches_a_session and not matches_mean:
                violations.append(
                    Violation(
                        kind=ViolationKind.VALUE_CONTRADICTS_COMPUTATION,
                        claim=claim,
                        expected=(
                            f"{claim.metric} across the sessions in scope is "
                            + ", ".join(f"{v:.2f}" for v in values[:6])
                            + (f" (mean {mean_value:.2f})" if len(values) > 1 else "")
                        ),
                        detail=(
                            f"the answer states {claim.metric} = {claim.value}, "
                            "which matches no session in scope nor their mean"
                        ),
                    )
                )

    ok = not violations
    return Verdict(
        ok=ok,
        violations=tuple(violations),
        claims=tuple(claims),
        final_answer=answer,
        notes=notes,
    )


def _describe_trend(computed: Trend) -> str:
    if computed.withheld_reason:
        return computed.withheld_reason
    parts = [f"the computed direction is '{computed.direction}'"]
    if computed.slope_per_day is not None:
        parts.append(f"slope {computed.slope_per_day:+.3f} per day")
    if computed.ci95_slope is not None:
        parts.append(
            f"95% CI [{computed.ci95_slope[0]:+.3f}, {computed.ci95_slope[1]:+.3f}]"
        )
    return "; ".join(parts)


def worst_action(
    violations: Iterable[Violation],
    policy: Optional[Dict[ViolationKind, VerificationPolicy]] = None,
) -> Optional[VerificationPolicy]:
    """The most severe response any violation calls for."""
    policy = policy or DEFAULT_POLICY
    order = [
        VerificationPolicy.ANNOTATE,
        VerificationPolicy.REGENERATE,
        VerificationPolicy.REFUSE,
    ]
    worst: Optional[VerificationPolicy] = None
    for violation in violations:
        action = policy.get(violation.kind, VerificationPolicy.ANNOTATE)
        if worst is None or order.index(action) > order.index(worst):
            worst = action
    return worst


def annotate(answer: str, violations: Sequence[Violation]) -> str:
    """Append a correction block without touching the model's own sentences.

    Rewriting the prose would be a way to manufacture a claim the model never
    made, so corrections are always additive and clearly attributed.
    """
    if not violations:
        return answer
    lines = [
        "",
        "---",
        "**Automated data check.** The following statements above are not "
        "supported by the recorded data:",
        "",
    ]
    for violation in violations:
        lines.append(f"- {violation.detail}. {violation.expected}")
    return answer + "\n".join(lines)


def refusal(violations: Sequence[Violation], facts: str = "") -> str:
    """Replace an unsupportable answer with what the data does support."""
    lines = [
        "I can't answer that from the recorded data without overstating it.",
        "",
    ]
    for violation in violations:
        lines.append(f"- {violation.detail}. {violation.expected}")
    if facts:
        lines.extend(["", "What the data does support:", "", facts])
    return "\n".join(lines)
