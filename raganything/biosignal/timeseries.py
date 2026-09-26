"""Deterministic statistics over stored session reports.

This is the half of the query layer that does not involve a language model at
all. A question like *"is my RMSSD trending down over six weeks?"* is answered
here, by arithmetic over :class:`~.store.ReportRecord`s, and the answer is
exact, reproducible, and identical on every run.

Three rules hold throughout:

1. **A withheld metric is never silently replaced.** A session whose RMSSD was
   withheld is excluded and its stored reason is carried through verbatim. No
   related metric is substituted, and the diagnostic ``hrv`` block -- which
   still holds the rejected values -- is never consulted.
2. **The denominator is always visible.** Every result reports how many sessions
   contributed and how many were excluded, with reasons. A mean over three of
   twenty sessions still computes, but it says so.
3. **A direction is only claimed when the statistics support one.** A trend
   reports "rising" or "falling" only when the 95% confidence interval on the
   slope excludes zero *and* a robust estimator agrees on the sign.
"""

from __future__ import annotations

import datetime as _dt
import math
import statistics
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

from .schema import Modality
from .store import ReportRecord

__all__ = [
    "Aggregate",
    "Comparison",
    "METRIC_SUPPORT",
    "METRIC_UNITS",
    "STATISTIC_VALIDITY",
    "STATISTICS",
    "Series",
    "Trend",
    "UnknownMetricError",
    "aggregate",
    "collect",
    "compare_windows",
    "daily_load_series",
    "group_by_label",
    "trend",
]


class UnknownMetricError(KeyError):
    """A metric this system does not compute. Never guessed at."""


#: metric -> the modalities whose signal quality governs it. A metric derived
#: from two signals is gated on the worse of them.
METRIC_SUPPORT: Dict[str, Tuple[Modality, ...]] = {
    "hrv_rmssd": (Modality.RR_INTERVAL,),
    "hrv_sdnn": (Modality.RR_INTERVAL,),
    "hrv_pnn50": (Modality.RR_INTERVAL,),
    "hrv_mean_rr": (Modality.RR_INTERVAL,),
    "hrv_mean_hr": (Modality.RR_INTERVAL,),
    "mean_hr": (Modality.HEART_RATE,),
    "max_hr_observed": (Modality.HEART_RATE,),
    "trimp": (Modality.HEART_RATE,),
    "mean_power": (Modality.POWER,),
    "normalized_power": (Modality.POWER,),
    "intensity_factor": (Modality.POWER,),
    "training_stress": (Modality.POWER,),
    "aerobic_decoupling_pct": (Modality.POWER, Modality.HEART_RATE),
}

METRIC_UNITS: Dict[str, str] = {
    "hrv_rmssd": "ms",
    "hrv_sdnn": "ms",
    "hrv_pnn50": "%",
    "hrv_mean_rr": "ms",
    "hrv_mean_hr": "bpm",
    "mean_hr": "bpm",
    "max_hr_observed": "bpm",
    "trimp": "au",
    "mean_power": "W",
    "normalized_power": "W",
    "intensity_factor": "ratio",
    "training_stress": "au",
    "aerobic_decoupling_pct": "%",
}

STATISTICS = (
    "mean",
    "median",
    "min",
    "max",
    "sum",
    "count",
    "stdev",
    "p25",
    "p75",
    "first",
    "last",
)

#: Summing an intensive quantity is meaningless -- the total of ten sessions'
#: mean heart rates is not a heart rate. Extensive quantities (accumulated
#: training load) may legitimately be summed.
_EXTENSIVE = frozenset({"trimp", "training_stress"})

STATISTIC_VALIDITY: Dict[str, frozenset] = {
    metric: frozenset(
        STATISTICS
        if metric in _EXTENSIVE
        else tuple(s for s in STATISTICS if s != "sum")
    )
    for metric in METRIC_SUPPORT
}

#: Two-sided 95% critical values of Student's t by degrees of freedom. Hardcoded
#: so the confidence intervals need no third-party dependency and can be checked
#: against any statistics textbook.
_T95: Dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _t_critical(df: int) -> float:
    if df < 1:
        return float("inf")
    if df in _T95:
        return _T95[df]
    return 1.96  # the normal limit, approached from above


def _check_metric(metric: str) -> None:
    if metric not in METRIC_SUPPORT:
        raise UnknownMetricError(
            f"unknown metric {metric!r}; this system computes: "
            f"{', '.join(sorted(METRIC_SUPPORT))}"
        )


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Series:
    """The usable values of one metric over a window, and what was dropped."""

    metric: str
    unit: Optional[str]
    points: List[Tuple[float, float]] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    #: session_id -> why it did not contribute. Withheld reasons verbatim.
    excluded: Dict[str, str] = field(default_factory=dict)
    window: Optional[Tuple[float, float]] = None
    gates: Dict[str, Any] = field(default_factory=dict)

    def values(self) -> List[float]:
        return [v for _, v in self.points]

    def __len__(self) -> int:
        return len(self.points)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "n": len(self.points),
            "n_excluded": len(self.excluded),
            "session_ids": list(self.session_ids),
            "excluded": dict(self.excluded),
            "window": list(self.window) if self.window else None,
            "gates": dict(self.gates),
        }


@dataclass(frozen=True)
class Aggregate:
    """One statistic over a series, or a refusal with its reason."""

    metric: str
    statistic: str
    value: Optional[float]
    unit: Optional[str]
    n: int
    n_excluded: int
    representative: bool
    window: Optional[Tuple[float, float]] = None
    excluded: Dict[str, str] = field(default_factory=dict)
    withheld_reason: Optional[str] = None
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "statistic": self.statistic,
            "value": self.value,
            "unit": self.unit,
            "n": self.n,
            "n_excluded": self.n_excluded,
            "representative": self.representative,
            "window": list(self.window) if self.window else None,
            "excluded": dict(self.excluded),
            "withheld_reason": self.withheld_reason,
            "note": self.note,
        }


@dataclass(frozen=True)
class Trend:
    """A fitted trend, with the evidence for or against calling a direction."""

    metric: str
    n: int
    span_days: float
    slope_per_day: Optional[float] = None
    ci95_slope: Optional[Tuple[float, float]] = None
    intercept: Optional[float] = None
    r2: Optional[float] = None
    theil_sen_slope: Optional[float] = None
    direction: Literal["rising", "falling", "flat", "undetermined"] = "undetermined"
    first_value: Optional[float] = None
    last_value: Optional[float] = None
    change: Optional[float] = None
    change_pct: Optional[float] = None
    unit: Optional[str] = None
    excluded: Dict[str, str] = field(default_factory=dict)
    withheld_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "n": self.n,
            "span_days": self.span_days,
            "slope_per_day": self.slope_per_day,
            "ci95_slope": list(self.ci95_slope) if self.ci95_slope else None,
            "intercept": self.intercept,
            "r2": self.r2,
            "theil_sen_slope": self.theil_sen_slope,
            "direction": self.direction,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "change": self.change,
            "change_pct": self.change_pct,
            "unit": self.unit,
            "excluded": dict(self.excluded),
            "withheld_reason": self.withheld_reason,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Comparison:
    """Two groups of sessions set against each other."""

    metric: str
    a: Aggregate
    b: Aggregate
    label_a: str = "A"
    label_b: str = "B"
    difference: Optional[float] = None
    ci95_difference: Optional[Tuple[float, float]] = None
    direction: Literal["higher", "lower", "indistinguishable", "undetermined"] = (
        "undetermined"
    )
    withheld_reason: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "label_a": self.label_a,
            "label_b": self.label_b,
            "a": self.a.to_dict(),
            "b": self.b.to_dict(),
            "difference": self.difference,
            "ci95_difference": list(self.ci95_difference)
            if self.ci95_difference
            else None,
            "direction": self.direction,
            "withheld_reason": self.withheld_reason,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------


def collect(
    records: Iterable[ReportRecord],
    metric: str,
    *,
    start: Optional[float] = None,
    end: Optional[float] = None,
    min_quality: float = 0.5,
    min_hrv_confidence: float = 0.0,
) -> Series:
    """Gather one metric's usable values, recording every exclusion.

    A session outside the window is skipped silently -- it was never in scope.
    A session *in* scope that cannot contribute is an **exclusion**, and is
    reported, because the difference between "you have no data" and "you have
    data I refused to use" is the whole point of this subsystem.

    A query-time ``min_quality`` can only ever remove sessions. It cannot
    un-withhold anything: withholding decisions were made at analysis time and
    are never recomputed here.
    """
    _check_metric(metric)
    modalities = METRIC_SUPPORT[metric]

    points: List[Tuple[float, float]] = []
    session_ids: List[str] = []
    excluded: Dict[str, str] = {}

    for record in sorted(records, key=lambda r: r.start):
        if start is not None and record.start < start:
            continue
        if end is not None and record.start >= end:
            continue

        if record.is_withheld(metric):
            excluded[record.session_id] = record.withheld_reason(metric) or "withheld"
            continue

        value = record.metric(metric)
        if value is None:
            excluded[record.session_id] = "metric not computed for this session"
            continue

        gated = False
        for modality in modalities:
            score = record.quality_for(modality.value)
            if score is None:
                excluded[record.session_id] = (
                    f"cannot verify the quality of the {modality.value} stream "
                    f"that {metric} derives from"
                )
                gated = True
                break
            if score < min_quality:
                excluded[record.session_id] = (
                    f"{modality.value} signal quality {score:.2f} is below the "
                    f"{min_quality:.2f} threshold applied to this question"
                )
                gated = True
                break
        if gated:
            continue

        if metric.startswith("hrv_") and min_hrv_confidence > 0:
            confidence = record.hrv_confidence
            if confidence is None or confidence < min_hrv_confidence:
                excluded[record.session_id] = (
                    f"HRV confidence {confidence if confidence is not None else 'unknown'} "
                    f"is below the {min_hrv_confidence:.2f} threshold"
                )
                continue

        points.append((record.start, value))
        session_ids.append(record.session_id)

    return Series(
        metric=metric,
        unit=METRIC_UNITS.get(metric),
        points=points,
        session_ids=session_ids,
        excluded=excluded,
        window=(start, end) if start is not None and end is not None else None,
        gates={"min_quality": min_quality, "min_hrv_confidence": min_hrv_confidence},
    )


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def aggregate(series: Series, statistic: str = "mean") -> Aggregate:
    """Reduce a series to one number, or refuse and say why."""
    metric = series.metric
    _check_metric(metric)
    values = series.values()
    n = len(values)
    n_excluded = len(series.excluded)

    def refuse(reason: str) -> Aggregate:
        return Aggregate(
            metric=metric,
            statistic=statistic,
            value=None,
            unit=series.unit,
            n=n,
            n_excluded=n_excluded,
            representative=False,
            window=series.window,
            excluded=dict(series.excluded),
            withheld_reason=reason,
        )

    if statistic not in STATISTICS:
        return refuse(
            f"unknown statistic {statistic!r}; supported: {', '.join(STATISTICS)}"
        )
    if statistic not in STATISTIC_VALIDITY[metric]:
        return refuse(
            f"{statistic} is not a meaningful statistic for {metric}: summing an "
            "intensive quantity does not produce a quantity of the same kind"
        )
    if statistic == "count":
        return Aggregate(
            metric=metric,
            statistic=statistic,
            value=float(n),
            unit="sessions",
            n=n,
            n_excluded=n_excluded,
            representative=True,
            window=series.window,
            excluded=dict(series.excluded),
            note=f"{n} session(s) contributed, {n_excluded} excluded",
        )
    if n == 0:
        return refuse(
            "no session in this window produced a usable value"
            + (f" ({n_excluded} excluded)" if n_excluded else "")
        )
    if statistic == "stdev" and n < 3:
        return refuse(f"a standard deviation over {n} session(s) is not meaningful")

    computed = {
        "mean": lambda: statistics.fmean(values),
        "median": lambda: statistics.median(values),
        "min": lambda: min(values),
        "max": lambda: max(values),
        "sum": lambda: sum(values),
        "stdev": lambda: statistics.stdev(values),
        "p25": lambda: _percentile(values, 0.25),
        "p75": lambda: _percentile(values, 0.75),
        "first": lambda: values[0],
        "last": lambda: values[-1],
    }[statistic]()

    representative = n_excluded <= n
    note = f"{n} session(s) contributed"
    if n_excluded:
        note += f", {n_excluded} excluded"
        if not representative:
            note += " -- more sessions were excluded than used, treat with caution"

    return Aggregate(
        metric=metric,
        statistic=statistic,
        value=float(computed),
        unit=series.unit,
        n=n,
        n_excluded=n_excluded,
        representative=representative,
        window=series.window,
        excluded=dict(series.excluded),
        note=note,
    )


# --------------------------------------------------------------------------
# trend
# --------------------------------------------------------------------------


def _theil_sen(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Median pairwise slope -- robust to a single wild session."""
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    return statistics.median(slopes) if slopes else None


def trend(series: Series, *, min_points: int = 5, min_span_days: float = 7.0) -> Trend:
    """Fit a trend and decide, conservatively, whether to call a direction."""
    metric = series.metric
    unit = series.unit
    values = series.values()
    n = len(values)

    if n == 0:
        span_days = 0.0
    else:
        span_days = (series.points[-1][0] - series.points[0][0]) / 86400.0

    def undetermined(reason: str, notes: Optional[List[str]] = None) -> Trend:
        return Trend(
            metric=metric,
            n=n,
            span_days=span_days,
            unit=unit,
            direction="undetermined",
            excluded=dict(series.excluded),
            withheld_reason=reason,
            notes=notes or [],
            first_value=values[0] if values else None,
            last_value=values[-1] if values else None,
        )

    if n < min_points:
        return undetermined(
            f"a trend needs at least {min_points} usable sessions; this window has "
            f"{n}" + (f" ({len(series.excluded)} excluded)" if series.excluded else "")
        )
    if span_days < min_span_days:
        return undetermined(
            f"a trend needs at least {min_span_days:.0f} days of span; these "
            f"sessions cover {span_days:.1f}"
        )

    t0 = series.points[0][0]
    xs = [(t - t0) / 86400.0 for t, _ in series.points]
    ys = list(values)

    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        return undetermined("every session falls on the same instant")
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    fitted = [intercept + slope * x for x in xs]
    sse = sum((y - f) ** 2 for y, f in zip(ys, fitted))
    sst = sum((y - mean_y) ** 2 for y in ys)
    r2 = (1.0 - sse / sst) if sst > 0 else None

    df = n - 2
    notes: List[str] = []
    if df >= 1:
        se_slope = math.sqrt((sse / df) / sxx)
        margin = _t_critical(df) * se_slope
        ci = (slope - margin, slope + margin)
        if sse == 0:
            # Zero residual variance collapses the interval onto the slope
            # itself. That is the correct reading -- under the fitted model
            # there is no uncertainty left -- but in real physiology it means
            # the input is synthetic or degenerate, so say so.
            notes.append(
                "the fit is exact (zero residuals), which real measurements "
                "essentially never are -- check the input is not synthetic"
            )
    else:
        ci = None
        notes.append(
            "fewer than three points leaves no residual degrees of freedom, so "
            "no confidence interval can be computed"
        )

    ts_slope = _theil_sen(xs, ys)

    if ci is None:
        direction = "flat"
    elif ci[0] > 0:
        direction = "rising"
    elif ci[1] < 0:
        direction = "falling"
    else:
        direction = "flat"

    if (
        direction in ("rising", "falling")
        and ts_slope is not None
        and ts_slope != 0
        and (ts_slope > 0) != (slope > 0)
    ):
        notes.append(
            "a robust (Theil-Sen) slope disagrees in sign with the least-squares "
            "fit, which usually means one session is driving the apparent trend"
        )
        direction = "flat"

    first_value, last_value = ys[0], ys[-1]
    change = last_value - first_value
    change_pct = (change / first_value * 100.0) if first_value else None
    if series.excluded:
        notes.append(f"{len(series.excluded)} session(s) in this window were excluded")

    return Trend(
        metric=metric,
        n=n,
        span_days=span_days,
        slope_per_day=slope,
        ci95_slope=ci,
        intercept=intercept,
        r2=r2,
        theil_sen_slope=ts_slope,
        direction=direction,
        first_value=first_value,
        last_value=last_value,
        change=change,
        change_pct=change_pct,
        unit=unit,
        excluded=dict(series.excluded),
        notes=notes,
    )


# --------------------------------------------------------------------------
# comparison and grouping
# --------------------------------------------------------------------------


def _compare_aggregates(
    metric: str,
    a: Aggregate,
    b: Aggregate,
    values_a: Sequence[float],
    values_b: Sequence[float],
    label_a: str,
    label_b: str,
) -> Comparison:
    if a.value is None or b.value is None:
        return Comparison(
            metric=metric,
            a=a,
            b=b,
            label_a=label_a,
            label_b=label_b,
            withheld_reason=(
                a.withheld_reason or b.withheld_reason or "one side has no usable data"
            ),
        )
    difference = a.value - b.value
    if len(values_a) < 2 or len(values_b) < 2:
        return Comparison(
            metric=metric,
            a=a,
            b=b,
            label_a=label_a,
            label_b=label_b,
            difference=difference,
            direction="undetermined",
            notes=["too few sessions on one side to say whether the gap is real"],
        )

    var_a = statistics.variance(values_a) / len(values_a)
    var_b = statistics.variance(values_b) / len(values_b)
    se = math.sqrt(var_a + var_b)
    df = min(len(values_a), len(values_b)) - 1  # conservative
    margin = _t_critical(df) * se
    ci = (difference - margin, difference + margin)

    if ci[0] > 0:
        direction = "higher"
    elif ci[1] < 0:
        direction = "lower"
    else:
        direction = "indistinguishable"

    return Comparison(
        metric=metric,
        a=a,
        b=b,
        label_a=label_a,
        label_b=label_b,
        difference=difference,
        ci95_difference=ci,
        direction=direction,
    )


def compare_windows(
    records: Iterable[ReportRecord],
    metric: str,
    window_a: Tuple[float, float],
    window_b: Tuple[float, float],
    *,
    statistic: str = "mean",
    **gates: Any,
) -> Comparison:
    """Compare one metric across two time windows."""
    records = list(records)
    series_a = collect(records, metric, start=window_a[0], end=window_a[1], **gates)
    series_b = collect(records, metric, start=window_b[0], end=window_b[1], **gates)
    return _compare_aggregates(
        metric,
        aggregate(series_a, statistic),
        aggregate(series_b, statistic),
        series_a.values(),
        series_b.values(),
        "window A",
        "window B",
    )


def group_by_label(
    records: Iterable[ReportRecord],
    metric: str,
    label_key: str,
    *,
    statistic: str = "mean",
    **gates: Any,
) -> Dict[str, Aggregate]:
    """Aggregate a metric separately for each value of a session label."""
    _check_metric(metric)
    buckets: Dict[str, List[ReportRecord]] = {}
    for record in records:
        value = record.labels.get(label_key)
        if value is None:
            continue
        buckets.setdefault(str(value), []).append(record)
    return {
        key: aggregate(collect(group, metric, **gates), statistic)
        for key, group in sorted(buckets.items())
    }


def after_event(
    records: Iterable[ReportRecord],
    metric: str,
    predicate: Callable[[ReportRecord], bool],
    *,
    lag_days: Tuple[float, float] = (1.0, 7.0),
    statistic: str = "mean",
    **gates: Any,
) -> Comparison:
    """Compare sessions following an event against all the others.

    The deterministic half of "does my sleep degrade after hard blocks?" --
    the qualitative half still belongs to retrieval.
    """
    records = sorted(records, key=lambda r: r.start)
    events = [r.start for r in records if predicate(r)]

    def follows_event(record: ReportRecord) -> bool:
        return any(
            lag_days[0] * 86400.0 <= (record.start - event) <= lag_days[1] * 86400.0
            for event in events
        )

    after = [r for r in records if follows_event(r) and not predicate(r)]
    other = [r for r in records if not follows_event(r) and not predicate(r)]

    series_after = collect(after, metric, **gates)
    series_other = collect(other, metric, **gates)
    return _compare_aggregates(
        metric,
        aggregate(series_after, statistic),
        aggregate(series_other, statistic),
        series_after.values(),
        series_other.values(),
        f"{lag_days[0]:.0f}-{lag_days[1]:.0f} days after event",
        "all other sessions",
    )


def daily_load_series(
    records: Iterable[ReportRecord],
    *,
    metric: str = "trimp",
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> List[float]:
    """One entry per calendar day, **including zeros for rest days**.

    ``analytics.load.ewma_load`` documents that dropping rest days inflates the
    acute figure, and until now nothing in the subsystem could supply a
    calendar-complete series. A day whose only session had the metric withheld
    contributes zero and is not silently skipped, because a missing day and a
    rest day are the same thing to an exponentially weighted average.
    """
    _check_metric(metric)
    records = sorted(records, key=lambda r: r.start)
    if not records:
        return []

    lo = start if start is not None else records[0].start
    hi = end if end is not None else records[-1].start

    by_day: Dict[_dt.date, float] = {}
    for record in records:
        if record.start < lo or record.start > hi:
            continue
        value = record.metric(metric)
        if value is None:
            continue
        by_day[record.day()] = by_day.get(record.day(), 0.0) + value

    first_day = _dt.datetime.fromtimestamp(lo, tz=_dt.timezone.utc).date()
    last_day = _dt.datetime.fromtimestamp(hi, tz=_dt.timezone.utc).date()

    out: List[float] = []
    day = first_day
    while day <= last_day:
        out.append(by_day.get(day, 0.0))
        day += _dt.timedelta(days=1)
    return out
