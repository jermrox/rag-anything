"""The personal evidence engine: discovering relationships, and refusing to
believe them until they hold on data they were not found in.

This is the part that makes a recommendation defensible rather than plausible.
Anything can find a correlation in one person's history; the question is
whether it survives being tested on nights that had no say in proposing it.

The pipeline, and why each stage exists:

1. **Hypothesise.** A candidate is an exposure (a condition on a day) paired
   with an outcome (a signal on that day or the next). Candidates come from
   anywhere -- a calendar pattern, a logged behaviour, a model's suggestion.
   Their source is irrelevant here, because the test is the same regardless.
2. **Split chronologically.** Older data proposes; newer data judges. Never a
   random split: adjacent nights are correlated, so a random holdout confirms
   a relationship against itself.
3. **Test on discovery data.** Permutation test, bootstrap interval, effect
   size. Groups smaller than the minimum are not tested at all.
4. **Correct for multiple comparisons.** Twenty hypotheses at p < 0.05 yields
   one false positive by construction.
5. **Replicate on the holdout.** Same direction, effect at least a floor
   fraction of the discovered one, and an interval that excludes zero.
   Direction alone is a coin flip; magnitude alone can be noise.
6. **Promote, or do not.** Surviving all of it makes a relationship VALIDATED.
   Everything else stays OBSERVED, which cannot support a recommendation.

A relationship that fails is kept, not deleted. "We looked and it is not true
for you" is a genuine finding about a person, and it is what stops the same
hypothesis being re-proposed every week.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, Iterable, List, Sequence

from ..knowledge.factors import EvidenceTier, Factor
from .stats import (
    DEFAULT_RESAMPLES,
    MIN_GROUP_SIZE,
    ComparisonResult,
    benjamini_hochberg,
    compare,
    split_by_time,
)

#: A replication must reproduce at least this fraction of the discovered
#: effect. Set below 1.0 because regression to the mean is expected and real:
#: an effect discovered at its most extreme will almost always look smaller on
#: replication. Set well above 0.0 because "same sign, any magnitude" is close
#: to a coin flip and would validate almost anything with a direction.
MIN_REPLICATION_EFFECT_RATIO = 0.5

#: False discovery rate tolerated during discovery.
DISCOVERY_FDR = 0.10

#: Days of history below which the engine declines to run at all. Distinct
#: from :data:`~vitalgraph.evidence.stats.MIN_GROUP_SIZE`, which governs one
#: comparison: this governs whether a chronological split is meaningful.
MIN_HISTORY_DAYS = 30


@dataclass(frozen=True, slots=True)
class DayRecord:
    """One day of a person's history: what happened, and what followed.

    ``exposures`` are conditions that were true of the day -- "late dinner",
    "travelled", "three or more meetings". ``outcomes`` are the measured or
    derived numbers attached to it.
    """

    day: date
    exposures: frozenset[str]
    outcomes: Dict[str, float]

    def has(self, exposure: str) -> bool:
        return exposure in self.exposures


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A candidate relationship, before anything is known about it."""

    exposure: str
    outcome: str
    factor: Factor
    description: str
    """Plain-language statement of what is being tested, shown to the person.
    Required, because a relationship nobody can state is one nobody can
    disagree with."""

    @property
    def key(self) -> str:
        return f"{self.exposure}->{self.outcome}"


@dataclass
class Relationship:
    """A hypothesis after testing: what was found, and how far it got.

    Both results are kept. A relationship that replicated is more credible than
    one that merely looked strong once, and keeping the discovery result makes
    the difference between the two auditable rather than asserted.
    """

    hypothesis: Hypothesis
    discovery: ComparisonResult | None
    replication: ComparisonResult | None = None
    tier: EvidenceTier = EvidenceTier.OBSERVED
    survived_correction: bool = False
    rejection_reason: str | None = None
    """Why this did not reach VALIDATED. None once it has."""

    @property
    def is_validated(self) -> bool:
        return self.tier is EvidenceTier.VALIDATED

    @property
    def can_support_recommendation(self) -> bool:
        return self.is_validated

    @property
    def comparable_days(self) -> int:
        """Total days behind this relationship, discovery and replication.

        This is the "11 comparable evenings" a person is shown. It is the
        honest denominator, not the count of days in their history.
        """
        total = 0
        for result in (self.discovery, self.replication):
            if result is not None:
                total += result.n
        return total

    def statement(self) -> str:
        """One line stating the finding and its standing, for a person to read.

        The tier is always named. A validated relationship and a hypothesis
        must never read the same way, and the difference has to survive being
        quoted out of context.
        """
        if self.discovery is None:
            return (
                f"{self.hypothesis.description}: not enough comparable days to "
                f"test (need at least {MIN_GROUP_SIZE} either side)."
            )
        best = self.replication or self.discovery
        direction = "higher" if best.difference > 0 else "lower"
        magnitude = abs(best.difference)
        if self.is_validated:
            return (
                f"{self.hypothesis.description}: {magnitude:.1f} {direction} "
                f"on average, held up on later data. "
                f"Evidence: validated - {self.comparable_days} comparable days."
            )
        reason = self.rejection_reason or "not yet replicated"
        return (
            f"{self.hypothesis.description}: {magnitude:.1f} {direction} "
            f"on average, but {reason}. "
            f"Evidence: observed only - {self.comparable_days} comparable days, "
            f"not a basis for a recommendation."
        )


@dataclass
class EvidenceGraph:
    """Everything known about how one person's life affects their physiology."""

    relationships: List[Relationship] = field(default_factory=list)
    history_days: int = 0
    ran: bool = False
    """False when the engine declined to run. Distinguishes "nothing is true
    for this person" from "we have not looked yet", which are opposite claims
    and must never render the same."""

    reason_not_run: str | None = None

    def validated(self) -> List[Relationship]:
        return [r for r in self.relationships if r.is_validated]

    def observed(self) -> List[Relationship]:
        return [r for r in self.relationships if not r.is_validated]

    def for_factor(self, factor: Factor) -> List[Relationship]:
        return [r for r in self.relationships if r.hypothesis.factor is factor]

    def supporting(self, outcome: str) -> List[Relationship]:
        """Validated relationships bearing on an outcome, strongest first."""
        matches = [
            r
            for r in self.validated()
            if r.hypothesis.outcome == outcome and r.replication is not None
        ]
        return sorted(
            matches, key=lambda r: abs(r.replication.effect_size), reverse=True
        )

    def summary(self) -> Dict[str, object]:
        return {
            "ran": self.ran,
            "reason_not_run": self.reason_not_run,
            "history_days": self.history_days,
            "hypotheses_tested": len(self.relationships),
            "validated": len(self.validated()),
            "observed_only": len(self.observed()),
            "validated_relationships": [
                {
                    "exposure": r.hypothesis.exposure,
                    "outcome": r.hypothesis.outcome,
                    "factor": r.hypothesis.factor.value,
                    "difference": r.replication.difference if r.replication else None,
                    "comparable_days": r.comparable_days,
                }
                for r in self.validated()
            ],
        }


def _group(
    records: Sequence[DayRecord], exposure: str, outcome: str
) -> tuple[List[float], List[float]]:
    """Split records into exposed and unexposed outcome values.

    A record missing the outcome is dropped from both groups rather than
    counted as zero -- a night with no measurement is not a night with a value
    of nothing.
    """
    exposed: List[float] = []
    unexposed: List[float] = []
    for record in records:
        value = record.outcomes.get(outcome)
        if value is None:
            continue
        (exposed if record.has(exposure) else unexposed).append(value)
    return exposed, unexposed


def _replicates(
    discovery: ComparisonResult, replication: ComparisonResult | None
) -> tuple[bool, str | None]:
    """Whether a holdout result reproduces a discovered effect."""
    if replication is None:
        return False, "not enough later days to re-test it"
    if replication.direction != discovery.direction:
        return False, "the effect went the other way on later data"
    if abs(replication.effect_size) < abs(discovery.effect_size) * (
        MIN_REPLICATION_EFFECT_RATIO
    ):
        return False, "the effect was much weaker on later data"
    if not replication.interval.excludes_zero:
        return False, "later data could not rule out no effect at all"
    return True, None


def build_evidence_graph(
    records: Sequence[DayRecord],
    hypotheses: Sequence[Hypothesis],
    discovery_fraction: float = 0.6,
    false_discovery_rate: float = DISCOVERY_FDR,
    min_history_days: int = MIN_HISTORY_DAYS,
    seed: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
    on_progress: Callable[[str], None] | None = None,
) -> EvidenceGraph:
    """Test every hypothesis against a person's history and promote what holds.

    ``resamples`` trades precision for time. The default is right for a real
    run; a caller sweeping many hypotheses interactively can lower it, at the
    cost of coarsening the p-value's own resolution, which is 1/resamples.

    Returns a graph even when nothing could be tested. ``ran`` distinguishes
    "we found nothing" from "we did not look", which are opposite claims about
    a person and must never be rendered identically.
    """
    graph = EvidenceGraph(history_days=len(records))

    if len(records) < min_history_days:
        graph.reason_not_run = (
            f"{len(records)} days of history; at least {min_history_days} are "
            "needed before a chronological split means anything"
        )
        return graph

    discovery_records, holdout_records = split_by_time(
        [(record.day.toordinal(), record) for record in records],
        discovery_fraction=discovery_fraction,
    )
    graph.ran = True

    # Stage one: test everything on discovery data only.
    tested: List[Relationship] = []
    for hypothesis in hypotheses:
        exposed, unexposed = _group(
            discovery_records, hypothesis.exposure, hypothesis.outcome
        )
        result = compare(exposed, unexposed, seed=seed, resamples=resamples)
        relationship = Relationship(hypothesis=hypothesis, discovery=result)
        if result is None:
            relationship.rejection_reason = "not enough comparable days to test it"
        tested.append(relationship)
        if on_progress:
            on_progress(f"tested {hypothesis.key} on discovery data")

    # Stage two: correct across everything that produced a p-value at all.
    testable = [r for r in tested if r.discovery is not None]
    survives = benjamini_hochberg(
        [r.discovery.p_value for r in testable], false_discovery_rate
    )
    for relationship, survived in zip(testable, survives):
        relationship.survived_correction = survived
        if not survived:
            relationship.rejection_reason = (
                "did not survive correction for testing many hypotheses at once"
            )

    # Stage three: replicate the survivors on data that had no say in finding
    # them. Anything that failed correction is not re-tested: doing so would
    # give a rejected hypothesis a second chance at the holdout and quietly
    # undo the correction.
    for relationship in tested:
        if relationship.discovery is None or not relationship.survived_correction:
            continue
        exposed, unexposed = _group(
            holdout_records,
            relationship.hypothesis.exposure,
            relationship.hypothesis.outcome,
        )
        relationship.replication = compare(
            exposed, unexposed, seed=seed, resamples=resamples
        )
        held, reason = _replicates(relationship.discovery, relationship.replication)
        if held:
            relationship.tier = EvidenceTier.VALIDATED
            relationship.rejection_reason = None
        else:
            relationship.rejection_reason = reason
        if on_progress:
            on_progress(
                f"replication of {relationship.hypothesis.key}: "
                f"{'held' if held else reason}"
            )

    graph.relationships = tested
    return graph


def hypotheses_from_exposures(
    records: Iterable[DayRecord],
    outcome: str,
    factor: Factor,
    describe: Callable[[str, str], str] | None = None,
) -> List[Hypothesis]:
    """Propose one hypothesis per exposure seen in a history.

    Generating candidates mechanically is exactly why the correction and the
    holdout exist. Proposing is cheap and should be; believing is expensive
    and is where the cost belongs.
    """
    exposures = sorted({e for record in records for e in record.exposures})

    def default_description(exposure: str, out: str) -> str:
        return f"On days with {exposure.replace('_', ' ')}, {out.replace('_', ' ')}"

    describe = describe or default_description
    return [
        Hypothesis(
            exposure=exposure,
            outcome=outcome,
            factor=factor,
            description=describe(exposure, outcome),
        )
        for exposure in exposures
    ]
