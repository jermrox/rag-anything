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
    #: Diagnostic detail only -- correction method, beat counts, artifact
    #: fraction. **Never read values from ``hrv.metrics``**: a metric withheld
    #: by the quality gate still appears there, so consulting it silently
    #: defeats the withholding. ``metrics`` and ``withheld`` are authoritative.
    hrv: Optional[hrv.HRVResult] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    withheld: Dict[str, str] = field(default_factory=dict)
    #: modality value -> quality score of the stream chosen for it.
    modality_quality: Dict[str, float] = field(default_factory=dict)
    #: metric name -> the quality score its gate was actually evaluated
    #: against. Recorded at gate time so no consumer has to reconstruct the
    #: join between metrics, modalities and streams and risk disagreeing
    #: with the decision that was made here.
    metric_quality: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session": self.session.to_dict(),
            "quality": {k: v.to_dict() for k, v in self.quality.items()},
            "fusion": {k.value: v.to_dict() for k, v in self.fusion.items()},
            "hrv": self.hrv.to_dict() if self.hrv else None,
            "metrics": dict(self.metrics),
            "withheld": dict(self.withheld),
            "modality_quality": dict(self.modality_quality),
            "metric_quality": dict(self.metric_quality),
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

    # Map each modality to the stream actually chosen for it, so that a metric
    # can be gated against the quality of the signal that produced it rather
    # than against nothing at all.
    chosen: Dict[Modality, Stream] = {}
    for modality in session.modalities():
        stream = best(modality)
        if stream is not None:
            chosen[modality] = stream
            stream_quality = report.quality.get(stream.provenance.source_id)
            if stream_quality is not None:
                report.modality_quality[modality.value] = stream_quality.score

    def governing_quality(
        *modalities: Modality,
    ) -> Optional[quality.QualityReport]:
        """Weakest quality report among the streams supporting a metric.

        A metric derived from two signals is only as trustworthy as the worse
        of them: aerobic decoupling computed from clean power and a heart-rate
        stream that dropped out for a third of the session is not a clean
        measurement of anything.
        """
        reports: List[quality.QualityReport] = []
        for modality in modalities:
            stream = chosen.get(modality)
            if stream is None:
                return None
            report_for_stream = report.quality.get(stream.provenance.source_id)
            if report_for_stream is None:
                return None
            reports.append(report_for_stream)
        if not reports:
            return None
        return min(reports, key=lambda r: r.score)

    def record(
        name: str,
        value: Optional[float],
        *modalities: Modality,
        label: Optional[str] = None,
    ) -> None:
        """Report a metric only if the signal behind it can support it.

        Every metric goes through this. An earlier version gated only TRIMP,
        which meant a mean heart rate computed over a stream with twelve
        percent coverage was reported as an ordinary number -- the precise
        failure this module exists to prevent, committed by the module itself.
        """
        if value is None:
            return
        governing = governing_quality(*modalities)
        if governing is None:
            report.withheld[name] = (
                f"withheld: cannot identify which stream supports {name}, so "
                "its quality cannot be verified"
            )
            return
        report.metric_quality[name] = governing.score
        gated, explanation = quality.gate(value, governing, min_quality, label or name)
        if gated is None:
            report.withheld[name] = explanation
        else:
            report.metrics[name] = gated

    # --- HRV ------------------------------------------------------------
    rr_stream = best(Modality.RR_INTERVAL)
    if rr_stream is not None and len(rr_stream) >= 4:
        usable = [
            s.value for s in rr_stream.sorted().samples if "no_contact" not in s.flags
        ]
        result = hrv.hrv_metrics(usable)
        report.hrv = result
        for name, value in result.metrics.items():
            record(
                f"hrv_{name}",
                value,
                Modality.RR_INTERVAL,
                label=f"HRV {name}",
            )
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
            record(
                "mean_hr",
                sum(v for _, v in hr_pairs) / len(hr_pairs),
                Modality.HEART_RATE,
            )
            record(
                "max_hr_observed",
                max(v for _, v in hr_pairs),
                Modality.HEART_RATE,
            )
        if rest_hr is None or max_hr is None:
            report.withheld["trimp"] = (
                "withheld: TRIMP needs the athlete's measured resting and maximum "
                "heart rate; a population estimate would make the score a guess "
                "about the person, not a measure of the session"
            )
        else:
            record(
                "trimp",
                load.trimp_banister(hr_pairs, rest_hr, max_hr, sex=sex),
                Modality.HEART_RATE,
                label="TRIMP",
            )

    # --- power -----------------------------------------------------------
    power_stream = best(Modality.POWER)
    if power_stream is not None and len(power_stream) >= 2:
        pw_pairs = [(s.t, s.value) for s in power_stream.sorted().samples]
        record(
            "mean_power",
            sum(v for _, v in pw_pairs) / len(pw_pairs),
            Modality.POWER,
        )
        np_value = load.normalized_power(pw_pairs)
        if np_value is None:
            report.withheld["normalized_power"] = (
                "withheld: effort is shorter than the 30 s rolling window the "
                "statistic is defined over"
            )
        else:
            record("normalized_power", np_value, Modality.POWER)
            if threshold_power:
                intensity = load.intensity_factor(np_value, threshold_power)
                if intensity is None:
                    # intensity_factor refuses on a non-positive threshold. An
                    # earlier `or 0.0` here turned that refusal into a reported
                    # value of zero, which is exactly the quiet fabrication this
                    # subsystem exists to prevent.
                    report.withheld["intensity_factor"] = (
                        "withheld: threshold power must be positive for an "
                        f"intensity factor to mean anything (got {threshold_power})"
                    )
                else:
                    record("intensity_factor", intensity, Modality.POWER)
                record(
                    "training_stress",
                    load.training_stress(session.duration_s, np_value, threshold_power),
                    Modality.POWER,
                )
            else:
                report.withheld["intensity_factor"] = (
                    "withheld: needs a measured functional threshold power"
                )

        if hr_stream is not None:
            decoupling = load.aerobic_decoupling(
                pw_pairs, [(s.t, s.value) for s in hr_stream.sorted().samples]
            )
            if decoupling is None:
                # Keyed to match the metric name exactly: a withheld entry whose
                # key differs from the metric it withholds is invisible to any
                # consumer looking the metric up.
                report.withheld["aerobic_decoupling_pct"] = (
                    "withheld: needs at least ten minutes of overlapping power and "
                    "heart rate"
                )
            else:
                record(
                    "aerobic_decoupling_pct",
                    decoupling,
                    Modality.POWER,
                    Modality.HEART_RATE,
                )

    return report


def _caveat_lines(report: SessionReport) -> List[str]:
    """Every reason a reader should hesitate before believing this session."""
    caveats: List[str] = list(report.warnings)
    for source_id, qr in report.quality.items():
        if qr.score < 0.8:
            caveats.append(
                f"{source_id} ({qr.stream_modality}) scored {qr.score:.2f}: "
                + "; ".join(qr.reasons)
            )
    if report.hrv is not None and report.hrv.notes:
        caveats.extend(f"HRV: {note}" for note in report.hrv.notes)
    return caveats


def to_content_list(
    report: SessionReport, page_offset: int = 0
) -> List[Dict[str, Any]]:
    """Render a report as a RAG-Anything content list.

    The structure here is dictated by how the ingestion pipeline chunks. Every
    ``text`` item in a content list is concatenated with the others into a
    single document and re-split on a token budget, so two facts written as
    separate text items can end up in different chunks and be retrieved apart.
    A ``table`` item, by contrast, becomes its own chunk with its caption,
    body and footnotes inlined.

    So anything that must never be read without its caveat is written as a
    table, and the caveats go in that table's footnote. An earlier version put
    the caveats in a text block of their own, which meant a retrieval hit on
    the metrics could arrive with the qualifications stripped off -- precisely
    the failure this subsystem exists to prevent.
    """
    session = report.session
    page = page_offset
    items: List[Dict[str, Any]] = []
    stamp = f"session {session.session_id} on {_iso(session.start)[:10]}"
    caveats = _caveat_lines(report)

    labels = ", ".join(
        f"{k}={v}" for k, v in session.labels.items() if k != "undecoded_notifications"
    )
    overview = (
        f"Biosignal {stamp} for subject {session.subject_id}. "
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
                "table_caption": [f"Stream inventory and provenance for {stamp}"],
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
        [
            name,
            _fmt(value, 2),
            _fmt(report.metric_quality.get(name), 2),
            "reported",
            "",
        ]
        for name, value in sorted(report.metrics.items())
    ]
    metric_rows += [
        [
            name,
            "-",
            _fmt(report.metric_quality.get(name), 2),
            "withheld",
            reason,
        ]
        for name, reason in sorted(report.withheld.items())
    ]
    if metric_rows:
        footnote = [
            "A withheld metric was not computable to a standard worth reporting. "
            "Do not infer a value for it from other rows. The quality column is "
            "the score of the stream the metric was gated against."
        ]
        if caveats:
            footnote.append(
                "Caveats that qualify every number in this table: " + "; ".join(caveats)
            )
        items.append(
            {
                "type": "table",
                "table_body": _table(
                    ["metric", "value", "quality", "status", "reason if withheld"],
                    metric_rows,
                ),
                "table_caption": [
                    f"Derived metrics and withholding decisions for {stamp}"
                ],
                "table_footnote": footnote,
                "page_idx": page,
            }
        )
        page += 1

    # --- HRV detail ------------------------------------------------------
    if report.hrv is not None:
        h = report.hrv
        hrv_rows = [
            [name, _fmt(value, 1), "reported", ""]
            for name, value in sorted(h.metrics.items())
        ]
        hrv_rows += [
            [name, "-", "withheld", why] for name, why in sorted(h.withheld.items())
        ]
        hrv_footnote = [
            f"Computed from {h.n_beats_used} usable beat intervals spanning "
            f"{h.window_s:.0f} seconds, using {h.method} artifact correction, which "
            f"rejected {h.artifact_fraction:.1%} of beats. Overall confidence "
            f"{h.confidence:.2f} on a 0-1 scale."
        ]
        if h.notes:
            hrv_footnote.append("Notes: " + "; ".join(h.notes) + ".")
        hrv_footnote.append(
            "These are the raw HRV computations. Where the session's evidence "
            "ledger marks one of them withheld, that decision governs and the "
            "value here must not be quoted."
        )
        if hrv_rows:
            items.append(
                {
                    "type": "table",
                    "table_body": _table(
                        [
                            "hrv metric",
                            "value (ms or %)",
                            "status",
                            "reason if withheld",
                        ],
                        hrv_rows,
                    ),
                    "table_caption": [f"Heart-rate variability detail for {stamp}"],
                    "table_footnote": hrv_footnote,
                    "page_idx": page,
                }
            )
            page += 1

    # --- disagreement between sources ------------------------------------
    if report.fusion:
        fusion_rows: List[List[Any]] = []
        chosen_lines: List[str] = []
        conflict_lines: List[str] = []
        for modality, result in report.fusion.items():
            chosen_lines.append(f"For {modality.value}, {result.chosen_reason}")
            for ag in result.agreements:
                fusion_rows.append(
                    [
                        modality.value,
                        ag.a,
                        ag.b,
                        ag.n_pairs,
                        f"{ag.bias:+.1f}",
                        f"{ag.limits[0]:+.1f}",
                        f"{ag.limits[1]:+.1f}",
                        f"{ag.max_abs_difference:.1f}",
                        UNIT_HINT.get(modality, ""),
                    ]
                )
            conflict_lines.extend(result.conflicts)
        if fusion_rows:
            items.append(
                {
                    "type": "table",
                    "table_body": _table(
                        [
                            "modality",
                            "chosen source",
                            "compared against",
                            "matched samples",
                            "bias",
                            "LoA lower",
                            "LoA upper",
                            "worst case",
                            "unit",
                        ],
                        fusion_rows,
                    ),
                    "table_caption": [f"Cross-source agreement for {stamp}"],
                    "table_footnote": [
                        "Where several devices measured the same thing, one was "
                        "selected and the rest compared against it rather than "
                        "discarded. " + " ".join(chosen_lines)
                    ]
                    + (
                        ["Conflicts: " + "; ".join(conflict_lines)]
                        if conflict_lines
                        else []
                    ),
                    "page_idx": page,
                }
            )
            page += 1
        elif conflict_lines or chosen_lines:
            items.append(
                {
                    "type": "text",
                    "text": (
                        f"Cross-source reconciliation for {stamp}. "
                        + " ".join(chosen_lines)
                        + (
                            ""
                            if not conflict_lines
                            else " Conflicts: " + "; ".join(conflict_lines) + "."
                        )
                    ),
                    "page_idx": page,
                }
            )
            page += 1

    # --- caveats ----------------------------------------------------------
    # Retained so the knowledge graph extracts entities and relations from the
    # caveats too. Nothing load-bearing lives only here: every caveat above is
    # also inlined into the metrics table's footnote, which cannot be split
    # away from the numbers it qualifies.
    if caveats:
        items.append(
            {
                "type": "text",
                "text": (
                    f"Data-quality caveats for {stamp}. Any "
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
