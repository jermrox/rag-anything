"""Bridge: biometric time-series -> retrievable knowledge-graph content.

This is the piece that makes VitalGraph a RAG product rather than a dashboard.
A knowledge graph cannot retrieve over 20,000 raw RR intervals, but it can
retrieve over "the night of 2026-03-04, RMSSD 23 ms, 41% below baseline,
fragmented sleep". The summariser performs that reduction.

Output is a ``content_list`` in the exact shape
``RAGAnything.insert_content_list`` documents (raganything/processor.py:2216),
so personal biometrics enter the same graph as harvested papers and protocol
specs -- and a single query can traverse all three.

**Narratives are generated deterministically from the computed numbers rather
than by an LLM.** Every sentence is a rendering of a value that was actually
measured, so nothing hallucinated can enter the corpus at ingest time, the
output is byte-stable across runs (which keeps ``doc_id`` stable), and
summarising a year of history costs no tokens. The LLM does its reasoning at
query time, over facts it can trust.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence

from ..biometrics import hrv
from ..biometrics.schema import SignalType, SleepStage
from ..biometrics.store import BiometricStore

#: URI scheme used as the ``file_path`` citation reference for personal data.
#: Answers cite e.g. ``biometrics://default/night-2026-03-04``, which resolves
#: to a real, inspectable time window rather than an opaque chunk id.
BIOMETRIC_SCHEME = "biometrics"

STAGE_NAMES = {
    SleepStage.AWAKE.value: "awake",
    SleepStage.LIGHT.value: "light",
    SleepStage.DEEP.value: "deep",
    SleepStage.REM.value: "REM",
}


@dataclass(frozen=True, slots=True)
class PeriodSummary:
    """A single summarised period (typically one night)."""

    period_id: str
    user: str
    start: datetime
    end: datetime
    metrics: hrv.HRVMetrics
    stage_minutes: Dict[str, float]
    mean_spo2: float | None
    mean_skin_temp: float | None
    rmssd_baseline: float | None
    rmssd_z: float | None
    verdict: str

    @property
    def citation_ref(self) -> str:
        return f"{BIOMETRIC_SCHEME}://{self.user}/{self.period_id}"

    @property
    def doc_id(self) -> str:
        """Stable id derived from the citation reference.

        Deterministic so that re-summarising a period *updates* rather than
        appending a near-duplicate -- the main defence against flooding the
        graph with redundant nightly summaries and destroying retrieval
        precision.
        """
        digest = hashlib.sha256(self.citation_ref.encode()).hexdigest()[:16]
        return f"vg-{digest}"

    @property
    def sleep_minutes(self) -> float:
        return sum(v for k, v in self.stage_minutes.items() if k != "awake")


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _classify(rmssd_z: float | None, coverage: float) -> str:
    """Turn a baseline deviation into a plain-language verdict."""
    if coverage < 0.5:
        return "insufficient data"
    if rmssd_z is None:
        return "no baseline yet"
    if rmssd_z <= -1.5:
        return "poor recovery"
    if rmssd_z <= -0.5:
        return "below-average recovery"
    if rmssd_z >= 1.5:
        return "strong recovery"
    if rmssd_z >= 0.5:
        return "above-average recovery"
    return "typical recovery"


def summarize_period(
    store: BiometricStore,
    start: datetime,
    end: datetime,
    period_id: str,
    user: str = "default",
    rmssd_baseline: Sequence[float] | None = None,
) -> PeriodSummary:
    """Reduce one time window to a summary object."""
    rr = store.rr_series(start, end)
    metrics = hrv.analyze(rr)

    stage_samples = store.samples_in(SignalType.SLEEP_STAGE, start, end)
    # Stage samples are logged once per minute, so each counts for one minute.
    stage_minutes: Dict[str, float] = {}
    for s in stage_samples:
        name = STAGE_NAMES.get(s.value, "unknown")
        stage_minutes[name] = stage_minutes.get(name, 0.0) + 1.0

    spo2 = _mean([s.value for s in store.samples_in(SignalType.SPO2, start, end)])
    temp = _mean(
        [s.value for s in store.samples_in(SignalType.SKIN_TEMPERATURE, start, end)]
    )

    baseline = list(rmssd_baseline or [])
    z = hrv.baseline_deviation(metrics.rmssd_ms, baseline) if baseline else None
    mean_baseline = _mean(baseline)

    return PeriodSummary(
        period_id=period_id,
        user=user,
        start=start,
        end=end,
        metrics=metrics,
        stage_minutes=stage_minutes,
        mean_spo2=spo2,
        mean_skin_temp=temp,
        rmssd_baseline=mean_baseline,
        rmssd_z=z,
        verdict=_classify(z, metrics.coverage),
    )


def render_narrative(summary: PeriodSummary) -> str:
    """Render a summary as grounded prose.

    Every clause restates a measured value. This text is what gets embedded and
    graph-extracted, so it names entities explicitly ("RMSSD", "deep sleep",
    the date) to give the extractor something to link.
    """
    m = summary.metrics
    date = summary.start.strftime("%Y-%m-%d")
    lines = [
        f"Biometric summary for {summary.user} covering {summary.period_id} "
        f"({date}, {summary.start:%H:%M} to {summary.end:%H:%M} UTC).",
        f"Overall assessment: {summary.verdict}.",
    ]

    if m.n_beats == 0:
        lines.append("No usable heart-rate data was recorded for this period.")
        return "\n".join(lines)

    lines.append(
        f"Heart rate averaged {m.mean_hr_bpm:.1f} bpm across {m.n_beats} accepted "
        f"beats ({m.coverage:.0%} of beats passed artifact correction; "
        f"{m.n_rejected} were rejected as ectopic or spurious)."
    )
    lines.append(
        f"Heart-rate variability: RMSSD {m.rmssd_ms:.1f} ms, SDNN {m.sdnn_ms:.1f} ms, "
        f"pNN50 {m.pnn50_pct:.1f}%."
    )

    if summary.rmssd_z is not None and summary.rmssd_baseline:
        delta = (m.rmssd_ms - summary.rmssd_baseline) / summary.rmssd_baseline * 100.0
        direction = "below" if delta < 0 else "above"
        lines.append(
            f"That RMSSD is {abs(delta):.0f}% {direction} the personal baseline of "
            f"{summary.rmssd_baseline:.1f} ms (z = {summary.rmssd_z:+.2f}), computed "
            f"from this user's own recent history rather than a population norm."
        )

    if summary.stage_minutes:
        total = summary.sleep_minutes
        parts = ", ".join(
            f"{name} {mins:.0f} min"
            for name, mins in sorted(
                summary.stage_minutes.items(), key=lambda kv: -kv[1]
            )
        )
        lines.append(
            f"Sleep totalled {total:.0f} minutes ({total / 60:.1f} h): {parts}."
        )
        awake = summary.stage_minutes.get("awake", 0.0)
        if total and awake / (total + awake) > 0.10:
            lines.append(
                f"Sleep was fragmented: {awake:.0f} minutes awake, "
                f"{awake / (total + awake):.0%} of time in bed."
            )

    if summary.mean_spo2 is not None:
        lines.append(f"Mean blood oxygen saturation was {summary.mean_spo2:.1f}%.")
    if summary.mean_skin_temp is not None:
        lines.append(f"Mean skin temperature was {summary.mean_skin_temp:.2f} C.")

    if summary.verdict in ("poor recovery", "below-average recovery"):
        lines.append(
            "Suppressed RMSSD alongside elevated heart rate is the classic "
            "signature of incomplete parasympathetic recovery, commonly "
            "following training load, illness, alcohol, or short sleep."
        )

    return "\n".join(lines)


def render_metrics_table(summary: PeriodSummary) -> str:
    """Markdown metrics table, so real numbers land in the graph."""
    m = summary.metrics
    rows = [
        ("Period", summary.period_id),
        ("Date", summary.start.strftime("%Y-%m-%d")),
        ("Verdict", summary.verdict),
        ("Mean HR (bpm)", f"{m.mean_hr_bpm:.1f}"),
        ("RMSSD (ms)", f"{m.rmssd_ms:.1f}"),
        ("SDNN (ms)", f"{m.sdnn_ms:.1f}"),
        ("pNN50 (%)", f"{m.pnn50_pct:.1f}"),
        ("Accepted beats", str(m.n_beats)),
        ("Artifact coverage", f"{m.coverage:.1%}"),
    ]
    if summary.rmssd_baseline is not None:
        rows.append(("RMSSD baseline (ms)", f"{summary.rmssd_baseline:.1f}"))
    if summary.rmssd_z is not None:
        rows.append(("RMSSD z-score", f"{summary.rmssd_z:+.2f}"))
    if summary.mean_spo2 is not None:
        rows.append(("Mean SpO2 (%)", f"{summary.mean_spo2:.1f}"))
    if summary.mean_skin_temp is not None:
        rows.append(("Mean skin temp (C)", f"{summary.mean_skin_temp:.2f}"))
    for name, mins in sorted(summary.stage_minutes.items(), key=lambda kv: -kv[1]):
        rows.append((f"Sleep: {name} (min)", f"{mins:.0f}"))

    body = "| Metric | Value |\n| --- | --- |\n"
    body += "\n".join(f"| {k} | {v} |" for k, v in rows)
    return body


def to_content_list(summary: PeriodSummary) -> List[Dict[str, Any]]:
    """Build the ``content_list`` payload for ``insert_content_list``.

    Shapes follow raganything/processor.py:2232-2239 exactly: a ``text`` item
    routes to the text pipeline, a ``table`` item routes to
    ``TableModalProcessor``.
    """
    return [
        {
            "type": "text",
            "text": render_narrative(summary),
            "page_idx": 0,
        },
        {
            "type": "table",
            "table_body": render_metrics_table(summary),
            "table_caption": [
                f"Biometric metrics for {summary.period_id} ({summary.verdict})"
            ],
            "table_footnote": [
                "Derived from BLE GATT Heart Rate Measurement (0x2A37) "
                "RR-interval data. Wellness metrics, not a medical diagnosis."
            ],
            "page_idx": 0,
        },
    ]


def nightly_windows(
    start: datetime, nights: int, hour: int = 0, duration_h: float = 8.0
) -> List[tuple[str, datetime, datetime]]:
    """Generate ``(period_id, start, end)`` windows for consecutive nights."""
    out = []
    for i in range(nights):
        s = (start + timedelta(days=i)).replace(
            hour=hour, minute=0, second=0, microsecond=0
        )
        e = s + timedelta(hours=duration_h)
        out.append((f"night-{s:%Y-%m-%d}", s, e))
    return out
