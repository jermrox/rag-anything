"""Turning signals into something a language model can reason over honestly.

This is the join between the two halves of the system. Numbers are useless to a
retrieval-augmented model unless they arrive as text that carries their own
caveats; and text is dangerous unless the caveats are inseparable from the
values. So every table produced here puts provenance and confidence in the same
row as the number, and every metric that was withheld is written down *as
withheld*, with its reason.

The practical effect: when the model is later asked "why was my recovery poor on
the 14th?", the retrieved context contains not just the recovery figure but the
fact that the strap lost contact for nine minutes and the RMSSD was computed
from 41 beats. A model given that context declines to over-claim. A model given
only the number does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .analytics import fusion, hrv, load, quality
from .schema import Modality, Session, Stream

__all__ = ["SessionReport", "analyze_session", "to_content_list"]


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat(timespec="seconds")


def _fmt(value: Optional[float], digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


@dataclass
class SessionReport:
    """Everything computed for one session, with its evidence attached."""

    session: Session
    quality: Dict[str, quality.QualityReport] = field(default_factory=dict)
    fusion: Dict[Modality, fusion.FusionResult] = field(default_factory=dict)
    hrv: Optional[hrv.HRVResult] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    withheld: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session.to_dict(),
            "quality": {k: v.to_dict() for k, v in self.quality.items()},
            "fusion": {k.value: v.to_dict() for k, v in self.fusion.items()},
            "hrv": self.hrv.to_dict() if self.hrv else None,
            "metrics": dict(self.metrics),
            "withheld": dict(self.withheld),
            "warnings": list(self.warnings),
        }


def analyze_session(
    session: Session,
    rest_hr: Optional[float] = None,
    max_hr: Optional[float] = None,
    threshold_power: Optional[float] = None,
    sex: str = "unspecified",
    min_quality: float = 0.5,
) -> SessionReport:
    """Run the full analysis chain over a session.

    Athlete constants (``rest_hr``, ``max_hr``, ``threshold_power``) have no
    defaults on purpose. Substituting a population estimate would let a
    load score appear that is really a guess about the person wearing the
    device, which is precisely the class of quiet fabrication this system
    exists to avoid: without them, the dependent metrics are reported as
    withheld and say what they need.
    """
    report = SessionReport(session=session)
    window = (session.start, session.end)

    for stream in session.streams:
        report.quality[stream.provenance.source_id] = quality.assess(
            stream, window=window
        )

    for modality in session.modalities():
        streams = session.of(modality)
        if len(streams) > 1:
            result = fusion.reconcile(streams, window=window)
            report.fusion[modality] = result
            report.warnings.extend(result.conflicts)

    def best(modality: Modality) -> Optional[Stream]:
        if modality in report.fusion:
            return report.fusion[modality].chosen
        return session.first(modality)

    def q(stream: Optional[Stream]) -> Optional[quality.QualityReport]:
        if stream is None:
            return None
        return report.quality.get(stream.provenance.source_id)

    # --- HRV ------------------------------------------------------------
    rr_stream = best(Modality.RR_INTERVAL)
    if rr_stream is not None and len(rr_stream) >= 4:
        rr_quality = q(rr_stream)
        usable = [
            s.value for s in rr_stream.sorted().samples if "no_contact" not in s.flags
        ]
        result = hrv.hrv_metrics(usable)
        report.hrv = result
        if rr_quality is not None and rr_quality.score < min_quality:
            for name in list(result.metrics):
                report.withheld[f"hrv_{name}"] = (
                    f"withheld: RR stream quality {rr_quality.score:.2f} below "
                    f"{min_quality:.2f} ({'; '.join(rr_quality.reasons)})"
                )
        else:
            for name, value in result.metrics.items():
                report.metrics[f"hrv_{name}"] = value
        for name, reason in result.withheld.items():
            report.withheld.setdefault(f"hrv_{name}", reason)
    else:
        report.withheld["hrv_rmssd"] = (
            "no beat-interval stream in this session -- HRV cannot be computed "
            "from averaged heart rate, only from RR intervals"
        )

    # --- heart-rate load -------------------------------------------------
    hr_stream = best(Modality.HEART_RATE)
    if hr_stream is not None and len(hr_stream) >= 2:
        hr_pairs = [
            (s.t, s.value) for s in hr_stream.sorted().samples if s.confidence > 0
        ]
        if hr_pairs:
            report.metrics["mean_hr"] = sum(v for _, v in hr_pairs) / len(hr_pairs)
            report.metrics["max_hr_observed"] = max(v for _, v in hr_pairs)
        if rest_hr is None or max_hr is None:
            report.withheld["trimp"] = (
                "withheld: TRIMP needs the athlete's measured resting and maximum "
                "heart rate; a population estimate would make the score a guess "
                "about the person, not a measure of the session"
            )
        else:
            value = load.trimp_banister(hr_pairs, rest_hr, max_hr, sex=sex)
            gated, explanation = quality.gate(
                value,
                report.quality[hr_stream.provenance.source_id],
                min_quality,
                "TRIMP",
            )
            if gated is None:
                report.withheld["trimp"] = explanation
            else:
                report.metrics["trimp"] = gated

    # --- power -----------------------------------------------------------
    power_stream = best(Modality.POWER)
    if power_stream is not None and len(power_stream) >= 2:
        pw_pairs = [(s.t, s.value) for s in power_stream.sorted().samples]
        report.metrics["mean_power"] = sum(v for _, v in pw_pairs) / len(pw_pairs)
        np_value = load.normalized_power(pw_pairs)
        if np_value is None:
            report.withheld["normalized_power"] = (
                "withheld: effort is shorter than the 30 s rolling window the "
                "statistic is defined over"
            )
        else:
            report.metrics["normalized_power"] = np_value
            if threshold_power:
                report.metrics["intensity_factor"] = (
                    load.intensity_factor(np_value, threshold_power) or 0.0
                )
                tss = load.training_stress(
                    session.duration_s, np_value, threshold_power
                )
                if tss is not None:
                    report.metrics["training_stress"] = tss
            else:
                report.withheld["intensity_factor"] = (
                    "withheld: needs a measured functional threshold power"
                )

        if hr_stream is not None:
            decoupling = load.aerobic_decoupling(
                pw_pairs, [(s.t, s.value) for s in hr_stream.sorted().samples]
            )
            if decoupling is None:
                report.withheld["aerobic_decoupling"] = (
                    "withheld: needs at least ten minutes of overlapping power and "
                    "heart rate"
                )
            else:
                report.metrics["aerobic_decoupling_pct"] = decoupling

    return report


def to_content_list(
    report: SessionReport, page_offset: int = 0
) -> List[Dict[str, Any]]:
    """Render a report as a RAG-Anything content list.

    The output is deliberately plain: narrative text plus markdown tables, in an
    order that reads top-down. Retrieval works on the text, and every retrieved
    fragment carries the provenance and confidence needed to interpret it
    without going back to the raw signal.
    """
    session = report.session
    page = page_offset
    items: List[Dict[str, Any]] = []

    labels = ", ".join(
        f"{k}={v}" for k, v in session.labels.items() if k != "undecoded_notifications"
    )
    overview = (
        f"Biosignal session {session.session_id} for subject {session.subject_id}. "
        f"Recorded from {_iso(session.start)} to {_iso(session.end)}, a duration of "
        f"{session.duration_s / 60.0:.1f} minutes. "
        f"Modalities captured: {', '.join(m.value for m in session.modalities()) or 'none'}. "
        f"Sources: {', '.join(sorted({s.provenance.device for s in session.streams})) or 'none'}."
    )
    if labels:
        overview += f" Context supplied with the session: {labels}."
    items.append({"type": "text", "text": overview, "page_idx": page})
    page += 1

    # --- stream inventory with provenance -------------------------------
    rows: List[List[Any]] = []
    for stream in session.streams:
        qr = report.quality.get(stream.provenance.source_id)
        evidence = sorted({s.evidence.value for s in stream.samples}) or ["none"]
        rows.append(
            [
                stream.modality.value,
                stream.unit,
                stream.provenance.device,
                stream.provenance.kind.value,
                "/".join(evidence),
                "yes" if stream.provenance.documented else "no",
                f"{stream.provenance.latency_s:.0f}",
                len(stream),
                _fmt(qr.coverage * 100 if qr else None, 0),
                _fmt(qr.score if qr else None, 2),
            ]
        )
    if rows:
        items.append(
            {
                "type": "table",
                "table_body": _table(
                    [
                        "modality",
                        "unit",
                        "device",
                        "source",
                        "evidence",
                        "documented",
                        "latency (s)",
                        "samples",
                        "coverage (%)",
                        "quality",
                    ],
                    rows,
                ),
                "table_caption": [
                    f"Stream inventory and provenance for session {session.session_id}"
                ],
                "table_footnote": [
                    "Evidence 'measured' is a sensor reading; 'vendor_derived' is a "
                    "closed algorithm's output; 'imputed' was invented to fill a gap. "
                    "Quality is the product of coverage, jitter, flag and provenance "
                    "penalties."
                ],
                "page_idx": page,
            }
        )
        page += 1

    # --- the evidence ledger --------------------------------------------
    metric_rows = [
        [name, _fmt(value, 2), "reported", ""]
        for name, value in sorted(report.metrics.items())
    ]
    metric_rows += [
        [name, "-", "withheld", reason]
        for name, reason in sorted(report.withheld.items())
    ]
    if metric_rows:
        items.append(
            {
                "type": "table",
                "table_body": _table(
                    ["metric", "value", "status", "reason if withheld"], metric_rows
                ),
                "table_caption": [
                    f"Derived metrics and withholding decisions for {session.session_id}"
                ],
                "table_footnote": [
                    "A withheld metric was not computable to a standard worth "
                    "reporting. Do not infer a value for it from other rows."
                ],
                "page_idx": page,
            }
        )
        page += 1

    # --- HRV detail ------------------------------------------------------
    if report.hrv is not None:
        h = report.hrv
        text = (
            f"Heart-rate variability for {session.session_id} was computed from "
            f"{h.n_beats_used} usable beat intervals spanning {h.window_s:.0f} seconds, "
            f"using {h.method} artifact correction, which rejected "
            f"{h.artifact_fraction:.1%} of beats. Overall confidence in these HRV "
            f"figures is {h.confidence:.2f} on a 0-1 scale."
        )
        if h.metrics:
            text += (
                " Reported: "
                + ", ".join(f"{k} {v:.1f}" for k, v in sorted(h.metrics.items()))
                + "."
            )
        if h.withheld:
            text += (
                " Withheld: "
                + "; ".join(f"{k} ({why})" for k, why in sorted(h.withheld.items()))
                + "."
            )
        if h.notes:
            text += " Notes: " + "; ".join(h.notes) + "."
        items.append({"type": "text", "text": text, "page_idx": page})
        page += 1

    # --- disagreement between sources ------------------------------------
    if report.fusion:
        lines: List[str] = []
        for modality, result in report.fusion.items():
            lines.append(f"For {modality.value}, {result.chosen_reason}.")
            for ag in result.agreements:
                lines.append(
                    f"Against {ag.b}, bias was {ag.bias:+.1f} {UNIT_HINT.get(modality, '')}"
                    f" over {ag.n_pairs} matched samples, with 95% limits of agreement "
                    f"[{ag.limits[0]:+.1f}, {ag.limits[1]:+.1f}] and a worst-case "
                    f"difference of {ag.max_abs_difference:.1f}."
                )
            for conflict in result.conflicts:
                lines.append(f"Conflict: {conflict}.")
        items.append(
            {
                "type": "text",
                "text": (
                    "Cross-source reconciliation. Where several devices measured the "
                    "same thing, one was selected and the rest were compared against "
                    "it rather than discarded.\n" + "\n".join(lines)
                ),
                "page_idx": page,
            }
        )
        page += 1

    # --- caveats ----------------------------------------------------------
    caveats: List[str] = list(report.warnings)
    for source_id, qr in report.quality.items():
        if qr.score < 0.8:
            caveats.append(
                f"{source_id} ({qr.stream_modality}) scored {qr.score:.2f}: "
                + "; ".join(qr.reasons)
            )
    if caveats:
        items.append(
            {
                "type": "text",
                "text": (
                    f"Data-quality caveats for session {session.session_id}. Any "
                    "conclusion drawn from this session must be qualified by the "
                    "following:\n- " + "\n- ".join(caveats)
                ),
                "page_idx": page,
            }
        )
        page += 1

    return items


UNIT_HINT: Dict[Modality, str] = {
    Modality.HEART_RATE: "bpm",
    Modality.POWER: "W",
    Modality.CADENCE: "rpm",
    Modality.SPEED: "m/s",
}
