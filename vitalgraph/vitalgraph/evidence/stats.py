"""Statistics for the personal evidence engine. Standard library only.

Three deliberate choices, each because the usual alternative is wrong for this
data rather than because a dependency was unavailable:

**Permutation tests, not t-tests.** A person has tens of nights, not thousands.
Biometric distributions are skewed, bounded and autocorrelated, and a t-test
assumes none of that. A permutation test assumes only exchangeability under
the null, computes an exact or near-exact p-value at any sample size, and needs
no distribution table. Seeded, so a run is reproducible.

**Bootstrap intervals, not standard errors.** The interval is what says whether
an effect is worth acting on. "Sleep onset was 31 minutes later" means one
thing with an interval of 24 to 38 and another entirely with an interval of
-5 to 67, and only the second is honest about a handful of nights.

**Benjamini-Hochberg, not raw p-values.** The engine tests many hypotheses
against the same history. At twenty hypotheses and p < 0.05, one false positive
is the expected outcome, not a surprise. Uncorrected discovery over a personal
history is a machine for manufacturing spurious personal truths.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

#: Resamples for permutation and bootstrap. Enough that the p-value's own
#: resolution (1/N) is far finer than any threshold applied to it, and small
#: enough to stay instant on a personal history.
DEFAULT_RESAMPLES = 10_000

#: Below this many observations per group, no test is run at all. Two nights
#: either side is not weak evidence of a relationship, it is no evidence, and
#: reporting a p-value for it invites treating noise as signal.
MIN_GROUP_SIZE = 5


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval, with the level it was computed at."""

    low: float
    high: float
    level: float = 0.95

    @property
    def excludes_zero(self) -> bool:
        return (self.low > 0.0) or (self.high < 0.0)

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The outcome of comparing an exposed group against an unexposed one."""

    difference: float
    """Exposed mean minus unexposed mean, in the outcome's own units."""

    interval: Interval
    p_value: float
    n_exposed: int
    n_unexposed: int
    effect_size: float
    """Hedges' g: the difference in pooled standard deviations, bias-corrected
    for small samples. Unitless, so effects on different outcomes compare."""

    @property
    def n(self) -> int:
        return self.n_exposed + self.n_unexposed

    @property
    def direction(self) -> int:
        """+1, -1, or 0. The sign is what a replication must reproduce."""
        if self.difference > 0:
            return 1
        if self.difference < 0:
            return -1
        return 0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    """Sample variance with Bessel's correction. Zero for a single value."""
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return sum((v - mu) ** 2 for v in values) / (len(values) - 1)


def hedges_g(exposed: Sequence[float], unexposed: Sequence[float]) -> float:
    """Standardised mean difference, corrected for small-sample bias.

    Cohen's d overstates the effect at the sample sizes a personal history
    provides -- by around 4% at n=20 -- so the correction is applied rather
    than ignored. Returns 0.0 when both groups are constant, since there is
    then no scale to standardise against.
    """
    n1, n2 = len(exposed), len(unexposed)
    if n1 < 2 or n2 < 2:
        return 0.0
    pooled_var = ((n1 - 1) * variance(exposed) + (n2 - 1) * variance(unexposed)) / (
        n1 + n2 - 2
    )
    if pooled_var <= 0.0:
        return 0.0
    d = (mean(exposed) - mean(unexposed)) / math.sqrt(pooled_var)
    correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    return d * correction


def permutation_p_value(
    exposed: Sequence[float],
    unexposed: Sequence[float],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> float:
    """Two-sided p-value for a difference in means, by label permutation.

    The null is that the exposure label carries no information, so the labels
    are reshuffled and the observed difference compared against the resulting
    distribution.

    The count is initialised at one rather than zero -- the observed labelling
    is itself one of the permutations under the null, and omitting it can
    return a p-value of exactly zero, which claims more certainty than any
    finite number of resamples can support.
    """
    if len(exposed) < 1 or len(unexposed) < 1:
        return 1.0
    observed = abs(mean(exposed) - mean(unexposed))
    pool = list(exposed) + list(unexposed)
    n_exposed = len(exposed)
    rng = random.Random(seed)

    at_least_as_extreme = 1
    for _ in range(resamples):
        rng.shuffle(pool)
        diff = abs(mean(pool[:n_exposed]) - mean(pool[n_exposed:]))
        if diff >= observed:
            at_least_as_extreme += 1
    return at_least_as_extreme / (resamples + 1)


def bootstrap_interval(
    exposed: Sequence[float],
    unexposed: Sequence[float],
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for the difference in means."""
    if not exposed or not unexposed:
        return Interval(0.0, 0.0, level)
    rng = random.Random(seed)
    differences: List[float] = []
    for _ in range(resamples):
        a = [exposed[rng.randrange(len(exposed))] for _ in exposed]
        b = [unexposed[rng.randrange(len(unexposed))] for _ in unexposed]
        differences.append(mean(a) - mean(b))
    differences.sort()
    tail = (1.0 - level) / 2.0
    low_index = max(0, int(tail * len(differences)))
    high_index = min(len(differences) - 1, int((1.0 - tail) * len(differences)))
    return Interval(differences[low_index], differences[high_index], level)


def compare(
    exposed: Sequence[float],
    unexposed: Sequence[float],
    level: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> ComparisonResult | None:
    """Compare two groups, or return None when there is too little to compare.

    None rather than a weak result: below :data:`MIN_GROUP_SIZE` the honest
    answer is that nothing was tested, and a p-value would only invite someone
    to act on it.
    """
    if len(exposed) < MIN_GROUP_SIZE or len(unexposed) < MIN_GROUP_SIZE:
        return None
    return ComparisonResult(
        difference=mean(exposed) - mean(unexposed),
        interval=bootstrap_interval(exposed, unexposed, level, resamples, seed),
        p_value=permutation_p_value(exposed, unexposed, resamples, seed),
        n_exposed=len(exposed),
        n_unexposed=len(unexposed),
        effect_size=hedges_g(exposed, unexposed),
    )


def benjamini_hochberg(
    p_values: Sequence[float], false_discovery_rate: float = 0.10
) -> List[bool]:
    """Which hypotheses survive control of the false discovery rate.

    Returns a list parallel to ``p_values``. Controlling the FDR rather than
    the family-wise error rate is the right trade for discovery: the engine is
    generating candidate relationships to be replicated, so tolerating a known
    proportion of false leads is better than a criterion so strict that real
    personal effects never surface.

    The step-up rule matters. Finding the largest rank whose p-value clears its
    threshold and rejecting *everything* below it -- rather than testing each
    rank independently -- is what makes this control the FDR at all.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    largest_passing_rank = -1
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= false_discovery_rate * rank / n:
            largest_passing_rank = rank
    survives = [False] * n
    for rank, index in enumerate(order, start=1):
        if rank <= largest_passing_rank:
            survives[index] = True
    return survives


def split_by_time(
    items: Sequence[Tuple[float, object]], discovery_fraction: float = 0.6
) -> Tuple[List[object], List[object]]:
    """Split chronologically into a discovery set and a later holdout set.

    Chronological, never random. A random split leaks: adjacent nights are
    correlated, so a relationship discovered on Tuesday and "confirmed" on the
    Wednesday beside it has been confirmed against itself. Requiring a pattern
    found in older data to hold in newer data is the only split that tests what
    matters -- whether the relationship is still true.
    """
    ordered = sorted(items, key=lambda pair: pair[0])
    cut = int(len(ordered) * discovery_fraction)
    return (
        [payload for _, payload in ordered[:cut]],
        [payload for _, payload in ordered[cut:]],
    )
