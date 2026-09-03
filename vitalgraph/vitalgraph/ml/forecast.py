"""Trend projection and early detection of sustained decline.

Uses Holt's linear exponential smoothing, fitted by a deterministic grid search
over the smoothing constants. That is a considered choice, not a shortcut: a
person has tens of nights of history, not thousands, and a gradient-boosted
model on 30 samples with 19 features memorises the sample and projects
confident nonsense. Holt estimates exactly two things -- a level and a slope --
which is about what this much data can honestly support.

A consequence worth having: this module needs no third-party package at all.
It lives under ``ml/`` because it is predictive, but it runs wherever the
analytics core runs.

The forecast is secondary to :func:`detect_sustained_decline`. Knowing RMSSD
will be 41 ms on Thursday is mildly interesting; knowing it has fallen for five
consecutive days is what should reach a person.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from .features import FeatureVector

#: Minimum history before a trend can be estimated at all. Two points define a
#: line through any noise; a handful is the least that says anything.
MIN_HISTORY = 5

#: Consecutive declining periods before a drift is called sustained. Chosen so
#: that one poor night after a hard workout does not trigger it.
SUSTAINED_PERIODS = 3

#: Grid searched for the smoothing constants. Coarse on purpose -- finer
#: resolution over this little data fits noise rather than signal.
_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
_BETAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)

#: Metrics where a falling value is the bad direction.
HIGHER_IS_BETTER = {
    "rmssd_ms",
    "sdnn_ms",
    "pnn50_pct",
    "sleep_minutes",
    "deep_fraction",
    "rem_fraction",
    "spo2_mean",
    "spo2_min",
    "artifact_coverage",
}


@dataclass(frozen=True, slots=True)
class Prediction:
    """One step of a projection."""

    step: int
    value: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class Forecast:
    """A projection of one metric, with its uncertainty."""

    metric: str
    level: float
    slope: float
    """Change per period, in the metric's own units."""
    predictions: Tuple[Prediction, ...]
    residual_sd: float
    n_history: int
    direction: str
    """improving | stable | declining"""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "level": round(self.level, 4),
            "slope_per_period": round(self.slope, 4),
            "direction": self.direction,
            "n_history": self.n_history,
            "residual_sd": round(self.residual_sd, 4),
            "predictions": [
                {
                    "step": p.step,
                    "value": round(p.value, 3),
                    "lower": round(p.lower, 3),
                    "upper": round(p.upper, 3),
                }
                for p in self.predictions
            ],
        }

    def summary(self) -> str:
        if not self.predictions:
            return f"{self.metric}: insufficient history to project."
        last = self.predictions[-1]
        return (
            f"{self.metric} is {self.direction} at {self.slope:+.2f} per period; "
            f"projected {last.value:.1f} in {last.step} period(s) "
            f"(range {last.lower:.1f}-{last.upper:.1f})."
        )


@dataclass(frozen=True, slots=True)
class DeclineSignal:
    """Whether a metric is in a sustained adverse drift."""

    metric: str
    is_declining: bool
    consecutive: int
    change: float
    """Total change across the declining run, in the metric's units."""
    detail: str


class InsufficientHistory(RuntimeError):
    """Raised when there is too little history to project anything."""


def holt_linear(
    series: Sequence[float], alpha: float, beta: float
) -> Tuple[float, float, List[float]]:
    """Holt's linear method. Returns ``(level, slope, one_step_fitted)``."""
    level = float(series[0])
    slope = float(series[1] - series[0])
    fitted: List[float] = []

    for value in series[1:]:
        prediction = level + slope
        fitted.append(prediction)
        previous_level = level
        level = alpha * value + (1 - alpha) * prediction
        slope = beta * (level - previous_level) + (1 - beta) * slope

    return level, slope, fitted


def _fit_parameters(series: Sequence[float]) -> Tuple[float, float]:
    """Grid-search the smoothing constants by one-step squared error.

    Deterministic, so a forecast is reproducible from the same history.
    """
    best: Tuple[float, float] | None = None
    best_sse = math.inf
    for alpha in _ALPHAS:
        for beta in _BETAS:
            _, _, fitted = holt_linear(series, alpha, beta)
            sse = sum((a - b) ** 2 for a, b in zip(series[1:], fitted))
            if sse < best_sse:
                best_sse, best = sse, (alpha, beta)
    return best or (0.3, 0.1)


def forecast_series(series: Sequence[float], metric: str, horizon: int = 7) -> Forecast:
    """Project a metric forward.

    Prediction intervals widen with the square root of the horizon, which is
    the right shape for accumulating uncertainty and stops a seven-day
    projection being presented as confidently as a one-day one.
    """
    if len(series) < MIN_HISTORY:
        raise InsufficientHistory(
            f"need at least {MIN_HISTORY} periods to project {metric}, "
            f"got {len(series)}"
        )

    alpha, beta = _fit_parameters(series)
    level, slope, fitted = holt_linear(series, alpha, beta)

    residuals = [a - b for a, b in zip(series[1:], fitted)]
    residual_sd = (
        math.sqrt(sum(r * r for r in residuals) / len(residuals)) if residuals else 0.0
    )

    predictions = []
    for step in range(1, horizon + 1):
        value = level + step * slope
        spread = 1.96 * residual_sd * math.sqrt(step)
        predictions.append(Prediction(step, value, value - spread, value + spread))

    # A slope smaller than the noise is not a trend.
    if abs(slope) < residual_sd * 0.25:
        direction = "stable"
    elif (slope > 0) == (metric in HIGHER_IS_BETTER):
        direction = "improving"
    else:
        direction = "declining"

    return Forecast(
        metric=metric,
        level=level,
        slope=slope,
        predictions=tuple(predictions),
        residual_sd=residual_sd,
        n_history=len(series),
        direction=direction,
    )


def detect_sustained_decline(
    series: Sequence[float], metric: str, periods: int = SUSTAINED_PERIODS
) -> DeclineSignal:
    """Detect a run of consecutive adverse movements.

    This is the signal worth surfacing. A single poor night follows a hard
    session or a late meal; several in a row is a pattern, and catching it on
    day three rather than day ten is the whole value.
    """
    if len(series) < periods + 1:
        return DeclineSignal(metric, False, 0, 0.0, "not enough history")

    adverse_is_down = metric in HIGHER_IS_BETTER
    consecutive = 0
    for previous, current in zip(reversed(series[:-1]), reversed(series[1:])):
        moved_down = current < previous
        if moved_down == adverse_is_down and current != previous:
            consecutive += 1
        else:
            break

    if consecutive < periods:
        return DeclineSignal(
            metric,
            False,
            consecutive,
            0.0,
            f"{consecutive} consecutive adverse period(s); "
            f"{periods} required to call it sustained",
        )

    change = series[-1] - series[-1 - consecutive]
    return DeclineSignal(
        metric,
        True,
        consecutive,
        change,
        f"{metric} moved adversely for {consecutive} consecutive periods "
        f"({change:+.1f} overall)",
    )


def forecast_from_vectors(
    vectors: Sequence[FeatureVector], metric: str, horizon: int = 7
) -> Forecast:
    """Project a named feature from a run of periods."""
    return forecast_series([v.get(metric) for v in vectors], metric, horizon)


def to_content_list(
    forecasts: Sequence[Forecast], declines: Sequence[DeclineSignal] = ()
) -> List[Dict[str, Any]]:
    """Render projections as knowledge-graph content.

    Sustained declines lead, because they are the actionable part; the
    projections follow as supporting detail.
    """
    if not forecasts and not declines:
        return []

    lines = []
    active = [d for d in declines if d.is_declining]
    if active:
        lines.append(
            "Sustained adverse trends detected: "
            + "; ".join(d.detail for d in active)
            + "."
        )
    lines.extend(f.summary() for f in forecasts)

    rows = [
        "| Metric | Direction | Per period | Projected | Range |",
        "| --- | --- | --- | --- | --- |",
    ]
    for f in forecasts:
        if not f.predictions:
            continue
        last = f.predictions[-1]
        rows.append(
            f"| {f.metric} | {f.direction} | {f.slope:+.2f} | {last.value:.1f} "
            f"| {last.lower:.1f}-{last.upper:.1f} |"
        )

    items: List[Dict[str, Any]] = [
        {"type": "text", "text": "\n".join(lines), "page_idx": 0}
    ]
    if len(rows) > 2:
        items.append(
            {
                "type": "table",
                "table_body": "\n".join(rows),
                "table_caption": ["Projected trends from recent history"],
                "table_footnote": [
                    "Holt linear smoothing over a short personal history. Intervals "
                    "widen with the square root of the horizon; a projection is an "
                    "extrapolation of recent trend, not a prediction of the future."
                ],
                "page_idx": 0,
            }
        )
    return items
