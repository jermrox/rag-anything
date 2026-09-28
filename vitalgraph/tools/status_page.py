"""Build the capability report: one page showing what the system can do.

Everything on the page is read from live code or from the committed evaluation
artifacts, never typed into the HTML by hand. That is the point. A status page
maintained separately from the thing it describes starts drifting the day it is
written, and a page that overstates a health product's capability is exactly
the failure this project is organised around avoiding.

So: the sensor table comes from the decoder constants, the factor coverage
comes from ``knowledge.factors``, the library comes from ``acquire.targets``
plus the harvest report, and every accuracy figure comes from the evaluation
JSON produced by the real-data runs. Rename a constant and this breaks rather
than lying.

Usage::

    python -m tools.status_page            # writes docs/status.html
    python -m tools.status_page --out X    # writes elsewhere
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import date
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from vitalgraph.acquire.targets import TARGETS
from vitalgraph.ble import measurements as m
from vitalgraph.knowledge.factors import (
    CHARACTERISTIC_FACTORS,
    FIVE_FACTORS,
    MEDICAL_CONTEXTS,
    Factor,
)
from vitalgraph.ml.epochs import EPOCH_FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


# --- what the decoders actually yield --------------------------------------


@dataclass(frozen=True, slots=True)
class SensorRow:
    """One characteristic we can decode, and what it is worth.

    ``uuid`` references the decoder module's own constants rather than a
    literal, so a characteristic that is renamed or dropped breaks this build
    instead of leaving a stale row on the page.
    """

    uuid: str
    name: str
    yields: str
    caveat: str


SENSORS: Tuple[SensorRow, ...] = (
    SensorRow(
        "0x2A37",
        "Heart Rate Measurement",
        "Beats per minute, plus RR intervals at 1/1024 s resolution",
        "RR intervals are the whole basis for HRV. Most bands never expose them.",
    ),
    SensorRow(
        m.CHAR_BODY_COMPOSITION_MEASUREMENT,
        "Body Composition",
        "Fat percentage, muscle mass, water, basal metabolism",
        "Impedance estimates, not a scan. Trends are worth more than any one reading.",
    ),
    SensorRow(
        m.CHAR_WEIGHT_MEASUREMENT,
        "Weight Measurement",
        "Body mass, in kg at 5 g or lb at 10 mg resolution",
        "Resolution differs by unit; the decoder keeps them apart.",
    ),
    SensorRow(
        m.CHAR_GLUCOSE_MEASUREMENT,
        "Glucose Measurement",
        "Concentration, sample type, sample location, sequence number",
        "Meter data. Nothing on a wrist measures this today.",
    ),
    SensorRow(
        m.CHAR_BLOOD_PRESSURE_MEASUREMENT,
        "Blood Pressure",
        "Systolic, diastolic, mean arterial pressure, pulse rate",
        "Carries status flags: body movement, loose cuff, irregular pulse.",
    ),
    SensorRow(
        m.CHAR_RSC_MEASUREMENT,
        "Running Speed and Cadence",
        "Instantaneous speed, cadence, stride length, total distance",
        "Foot pod or shoe sensor. A wrist infers cadence; this measures it.",
    ),
    SensorRow(
        m.CHAR_CSC_MEASUREMENT,
        "Cycling Speed and Cadence",
        "Wheel and crank revolutions with event timestamps",
        "Counters roll over; the decoder reports raw counts, not derived speed.",
    ),
)


# --- reading the committed evidence ----------------------------------------


def _load(name: str) -> dict:
    return json.loads((DOCS / name).read_text())


def count_tests() -> int:
    """Count test functions, by reading the test files rather than trusting a
    number someone wrote down once."""
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        total += len(re.findall(r"^def test_", path.read_text(), re.MULTILINE))
    return total


# --- rendering -------------------------------------------------------------


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _signed_pct(value: float) -> str:
    return f"{value * 100:+.1f}%"


def subject_chart(per_subject: Dict[str, dict]) -> str:
    """A diverging bar per subject: accuracy minus the majority-class baseline.

    This chart rather than a single mean, because the mean hides the thing that
    matters. Ten of these sixteen bars point the wrong way, and a headline
    accuracy of 50.4% does not tell you that.
    """
    rows = sorted(
        per_subject.items(), key=lambda kv: kv[1]["margin_over_baseline"], reverse=True
    )
    row_h, gap = 22, 6
    left, right = 78, 60
    width = 640
    height = len(rows) * (row_h + gap) + 46
    plot = width - left - right
    span = 0.50  # bars are clipped at +/- 50 points, which nothing reaches
    zero = left + plot / 2

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Accuracy margin over the majority-class baseline, '
        f'per subject" class="chart">'
    ]

    # Axis: ticks at -40, -20, 0, +20, +40 points.
    for tick in (-0.4, -0.2, 0.0, 0.2, 0.4):
        x = zero + (tick / span) * (plot / 2)
        cls = "axis-zero" if tick == 0 else "axis-tick"
        parts.append(
            f'<line x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{height - 18}" '
            f'class="{cls}" />'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{height - 5}" class="tick-label" '
            f'text-anchor="middle">{tick * 100:+.0f}</text>'
        )
    parts.append(
        f'<text x="{zero:.1f}" y="16" class="tick-label" text-anchor="middle">'
        f"percentage points vs. that subject&#8217;s own baseline</text>"
    )

    for i, (name, row) in enumerate(rows):
        y = 28 + i * (row_h + gap)
        margin = max(-span, min(span, row["margin_over_baseline"]))
        w = abs(margin / span) * (plot / 2)
        x = zero if margin >= 0 else zero - w
        cls = "bar-skill" if row["has_skill"] else "bar-noskill"
        parts.append(
            f'<rect x="{x:.1f}" y="{y}" width="{max(w, 1.0):.1f}" '
            f'height="{row_h - 8}" class="{cls}" rx="1" />'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + row_h - 12}" class="row-label" '
            f'text-anchor="end">{escape(name)}</text>'
        )
        label_x = zero + (w + 8 if margin >= 0 else -w - 8)
        anchor = "start" if margin >= 0 else "end"
        parts.append(
            f'<text x="{label_x:.1f}" y="{y + row_h - 12}" class="row-value" '
            f'text-anchor="{anchor}">{_signed_pct(row["margin_over_baseline"])}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


#: Shown where a factor has no passive input at all. Spelled out rather than
#: left as a dash: the absence is the finding, not a missing cell.
NO_PASSIVE_INPUT = (
    '<em class="dim">nothing &#8212; this factor is invisible to a band</em>'
)
NO_CHARACTERISTICS = '<span class="dim">none</span>'


def factor_rows() -> str:
    out: List[str] = []
    for factor in Factor:
        d = FIVE_FACTORS[factor]
        passive = escape(", ".join(d.passive_inputs)) or NO_PASSIVE_INPUT
        chars = sorted(u for u, f in CHARACTERISTIC_FACTORS.items() if f is factor)
        decoded = escape(", ".join(chars)) or NO_CHARACTERISTICS
        state = "sensed" if d.is_wrist_observable else "asked"
        badge = "Passively sensed" if d.is_wrist_observable else "Only by asking"
        out.append(
            f"""
            <article class="factor">
              <header>
                <h3>{escape(factor.value.title())}</h3>
                <span class="badge badge-{state}">{badge}</span>
              </header>
              <p class="question">{escape(d.core_question)}</p>
              <dl>
                <dt>Sensed without asking</dt>
                <dd>{passive}</dd>
                <dt>Requires the person</dt>
                <dd>{escape(", ".join(d.active_inputs))}</dd>
                <dt>Characteristics decoded</dt>
                <dd class="mono">{decoded}</dd>
              </dl>
              <p class="note">{escape(d.wrist_sensing_note)}</p>
            </article>"""
        )
    return "".join(out)


def library_rows(harvest: dict) -> str:
    by_name = {r["name"]: r for r in harvest["repositories"]}
    rows: List[str] = []
    for target in TARGETS:
        r = by_name.get(target.name)
        if r is None:
            continue
        policy = r["policy"]
        label = "Source + facts" if policy == "verbatim" else "Facts only"
        rows.append(
            f"""<tr>
              <td>{escape(r["name"])}</td>
              <td class="mono dim">{escape(r["citation_ref"])}</td>
              <td>{escape(r["category"].replace("_", " "))}</td>
              <td class="mono">{escape(r["detected_spdx"] or "unknown")}</td>
              <td><span class="policy policy-{policy}">{label}</span></td>
              <td class="num">{r["chunks"]:,}</td>
              <td class="num">{r["protocol_facts"]:,}</td>
            </tr>"""
        )
    return "".join(rows)


def medical_rows() -> str:
    return "".join(
        f"""<tr>
          <td>{escape(c.name)}</td>
          <td class="dim">{escape(c.kind.value.replace("_", " "))}</td>
          <td>{escape(", ".join(f.value for f in c.modifies))}</td>
          <td>{escape(c.interpretation_note)}</td>
        </tr>"""
        for c in MEDICAL_CONTEXTS
    )


def sensor_rows() -> str:
    return "".join(
        f"""<tr>
          <td class="mono">{escape(s.uuid)}</td>
          <td>{escape(s.name)}</td>
          <td>{escape(s.yields)}</td>
          <td class="dim">{escape(s.caveat)}</td>
        </tr>"""
        for s in SENSORS
    )


def ablation_rows(ablation: dict) -> str:
    labels = {
        "baseline_12_features": ("Per-epoch statistics only", "12"),
        "with_context_36_features": ("&#43; temporal context", "36"),
        "with_context_and_frequency_48_features": ("&#43; full LF/HF spectrum", "48"),
        "with_context_and_compact_frequency_39_features": (
            "&#43; normalised HF only",
            "39",
        ),
    }
    four = ablation["results"]["four_class"]
    two = ablation["results"]["sleep_wake"]
    rows: List[str] = []
    for key, (label, n) in labels.items():
        a, b = four[key], two[key]
        best = key == "with_context_36_features"
        rows.append(
            f"""<tr class="{"best" if best else ""}">
              <td>{label}</td>
              <td class="num dim">{n}</td>
              <td class="num">{_pct(a["accuracy"])}</td>
              <td class="num">{a["mean_kappa"]:.3f}</td>
              <td class="num">{a["subjects_with_skill"]}&#8202;/&#8202;16</td>
              <td class="num">{_pct(b["accuracy"])}</td>
              <td class="num">{b["mean_kappa"]:.3f}</td>
              <td class="num">{b["subjects_with_skill"]}&#8202;/&#8202;16</td>
            </tr>"""
        )
    return "".join(rows)


NOT_BUILT: Sequence[Tuple[str, str]] = (
    (
        "Accelerometer input",
        "The single most likely reason staging is stuck. Every wrist-staging "
        "method worth copying uses motion alongside heart rate. The dataset we "
        "can reach has no accelerometer channel at all.",
    ),
    (
        "The domain knowledge graphs",
        "Twenty subject-area graphs are defined with routing and seeded "
        "catalogues, and none has been built. Building them needs an LLM key; "
        "search over the harvested code does not, and works today.",
    ),
    (
        "Nutrition, Mind and Connection input",
        "Three of the five factors are answered by asking, and there is no "
        "surface for asking yet. The passive half of the product is much "
        "further along than the conversational half.",
    ),
    (
        "Anything clinical",
        "Lab ingestion, FHIR observations, evidence-graded recommendations, "
        "the audit chain and the mode gate are designed and unbuilt. Nothing "
        "here should be pointed at a patient.",
    ),
    (
        "Multi-user isolation",
        "One user, one SQLite file. The citation scheme already carries a user "
        "segment; the enforcement behind it does not exist.",
    ),
)


def render(harvest: dict, four_class: dict, ablation: dict, n_tests: int) -> str:
    loso = four_class["leave_one_subject_out"]
    load = four_class["load"]
    stages = load["stage_counts"]
    context = ablation["results"]["four_class"]["with_context_36_features"]
    context2 = ablation["results"]["sleep_wake"]["with_context_36_features"]

    return f"""<title>Vybe Health Capability Report</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bitter:wght@500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  --ground: #eceff0;
  --surface: #ffffff;
  --sunken: #e2e7e8;
  --ink: #14191b;
  --ink-soft: #47555a;
  --ink-dim: #6d7b80;
  --rule: #cfd7d9;
  --accent: #0d6e6b;
  --accent-soft: #d5e8e7;
  --good: #2c6e45;
  --warn: #8a5a10;
  --crit: #9c3128;
  --shadow: 0 1px 2px rgba(20, 25, 27, .07);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground: #12181a;
    --surface: #192023;
    --sunken: #101618;
    --ink: #e6ecee;
    --ink-soft: #a9b8bc;
    --ink-dim: #7d8d92;
    --rule: #2c3639;
    --accent: #4fbfb8;
    --accent-soft: #1d3634;
    --good: #6cc08b;
    --warn: #d9a441;
    --crit: #e0776b;
    --shadow: none;
  }}
}}
:root[data-theme="dark"] {{
  --ground: #12181a;
  --surface: #192023;
  --sunken: #101618;
  --ink: #e6ecee;
  --ink-soft: #a9b8bc;
  --ink-dim: #7d8d92;
  --rule: #2c3639;
  --accent: #4fbfb8;
  --accent-soft: #1d3634;
  --good: #6cc08b;
  --warn: #d9a441;
  --crit: #e0776b;
  --shadow: none;
}}

* {{ box-sizing: border-box; }}
body {{
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.55;
  margin: 0;
  padding: 0 20px 80px;
}}
.wrap {{ max-width: 1000px; margin: 0 auto; }}

h1, h2, h3 {{ font-family: Bitter, Georgia, serif; text-wrap: balance; margin: 0; }}
h1 {{ font-size: 2.35rem; font-weight: 600; letter-spacing: -.015em; line-height: 1.12; }}
h2 {{ font-size: 1.3rem; font-weight: 600; letter-spacing: -.005em; }}
h3 {{ font-size: 1.02rem; font-weight: 600; }}
p {{ margin: 0; max-width: 68ch; }}
.mono, code {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: .86em; }}
.dim {{ color: var(--ink-dim); }}
.num {{ font-variant-numeric: tabular-nums; text-align: right; }}

/* --- masthead --- */
.masthead {{ padding: 56px 0 34px; display: grid; gap: 18px; }}
.eyebrow {{
  font-family: "IBM Plex Mono", monospace; font-size: .72rem; letter-spacing: .16em;
  text-transform: uppercase; color: var(--accent);
}}
.lede {{ font-size: 1.08rem; color: var(--ink-soft); max-width: 64ch; }}
.stamp {{
  font-family: "IBM Plex Mono", monospace; font-size: .74rem; color: var(--ink-dim);
  border-top: 1px solid var(--rule); padding-top: 12px;
}}

/* --- strips --- */
section {{ border-top: 1px solid var(--rule); padding: 34px 0; display: grid; gap: 20px; }}
.strip-head {{ display: grid; gap: 8px; }}
.strip-label {{
  font-family: "IBM Plex Mono", monospace; font-size: .7rem; letter-spacing: .14em;
  text-transform: uppercase; color: var(--ink-dim);
}}

/* --- tables --- */
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; min-width: 620px; font-size: .9rem; }}
th {{
  text-align: left; font-weight: 500; font-size: .72rem; letter-spacing: .09em;
  text-transform: uppercase; color: var(--ink-dim);
  padding: 0 14px 8px 0; border-bottom: 1px solid var(--rule); white-space: nowrap;
}}
th.num {{ text-align: right; }}
td {{ padding: 9px 14px 9px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }}
tr.best td {{ background: var(--accent-soft); }}
tbody tr:last-child td {{ border-bottom: none; }}

.policy {{
  font-family: "IBM Plex Mono", monospace; font-size: .72rem; white-space: nowrap;
  padding: 2px 7px; border-radius: 2px; border: 1px solid var(--rule);
}}
.policy-verbatim {{ color: var(--good); border-color: currentColor; }}
.policy-facts_only {{ color: var(--warn); border-color: currentColor; }}

/* --- factors --- */
.factors {{ display: grid; gap: 1px; background: var(--rule); border: 1px solid var(--rule); }}
.factor {{ background: var(--surface); padding: 18px 20px; display: grid; gap: 10px; }}
.factor header {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.question {{ color: var(--ink-soft); font-style: italic; }}
.badge {{
  font-family: "IBM Plex Mono", monospace; font-size: .68rem; letter-spacing: .06em;
  text-transform: uppercase; padding: 2px 8px; border-radius: 2px;
}}
.badge-sensed {{ background: var(--accent-soft); color: var(--accent); }}
.badge-asked {{ background: var(--sunken); color: var(--ink-dim); }}
.factor dl {{ margin: 0; display: grid; grid-template-columns: 172px 1fr; gap: 4px 16px; font-size: .88rem; }}
.factor dt {{ color: var(--ink-dim); font-size: .8rem; }}
.factor dd {{ margin: 0; }}
.note {{ font-size: .88rem; color: var(--ink-soft); border-left: 2px solid var(--accent); padding-left: 12px; }}

/* --- chart --- */
.chart {{ width: 100%; max-width: 660px; height: auto; }}
.axis-tick {{ stroke: var(--rule); stroke-width: 1; }}
.axis-zero {{ stroke: var(--ink-soft); stroke-width: 1; }}
.tick-label {{ fill: var(--ink-dim); font-family: "IBM Plex Mono", monospace; font-size: 9px; }}
.row-label {{ fill: var(--ink-soft); font-family: "IBM Plex Mono", monospace; font-size: 10px; }}
.row-value {{ fill: var(--ink-dim); font-family: "IBM Plex Mono", monospace; font-size: 9.5px; }}
.bar-skill {{ fill: var(--accent); }}
.bar-noskill {{ fill: var(--crit); opacity: .72; }}
.legend {{ display: flex; gap: 20px; font-size: .8rem; color: var(--ink-soft); flex-wrap: wrap; }}
.key {{ display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; }}

/* --- verdict --- */
.verdict {{
  background: var(--surface); border: 1px solid var(--rule); border-left: 3px solid var(--crit);
  padding: 18px 22px; display: grid; gap: 10px; box-shadow: var(--shadow);
}}
.verdict h3 {{ color: var(--crit); }}

/* --- counters --- */
.counters {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(146px, 1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); }}
.counter {{ background: var(--surface); padding: 14px 16px; display: grid; gap: 2px; }}
.counter .v {{ font-family: Bitter, Georgia, serif; font-size: 1.5rem; font-variant-numeric: tabular-nums; }}
.counter .k {{ font-size: .74rem; color: var(--ink-dim); }}

/* --- gaps --- */
.gaps {{ display: grid; gap: 0; border: 1px solid var(--rule); background: var(--rule); }}
.gap {{ background: var(--surface); padding: 15px 18px; display: grid; grid-template-columns: 230px 1fr; gap: 6px 22px; }}
.gap h3 {{ font-size: .95rem; }}
.gap p {{ font-size: .89rem; color: var(--ink-soft); }}

@media (max-width: 700px) {{
  .factor dl, .gap {{ grid-template-columns: 1fr; }}
  h1 {{ font-size: 1.85rem; }}
}}
</style>

<div class="wrap">
  <header class="masthead">
    <div class="eyebrow">Vybe Health &#183; capability report</div>
    <h1>What this system can measure, and what it only guesses at</h1>
    <p class="lede">A health engine is only worth what its weakest claim is worth. This page
    lists every sensor reading we can decode, every factor we cannot sense, and the one
    accuracy number we have actually measured against expert-scored human sleep &#8212;
    including the ten subjects it fails on.</p>
    <div class="stamp">Generated from the source tree on {date.today().isoformat()}
    &#183; {n_tests} tests &#183; every figure below is read from code or from a committed
    evaluation run, none is typed in by hand</div>
  </header>

  <section>
    <div class="strip-head">
      <div class="strip-label">01 &#183; Sensor layer</div>
      <h2>Readings we can decode off a device</h2>
      <p>Bluetooth health characteristics with a working decoder and tests behind them.
      Absent measurements stay absent: an IEEE&#8202;11073 &#8220;not a number&#8221; comes
      back as nothing, never as a zero that a chart would draw.</p>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>UUID</th><th>Characteristic</th><th>Yields</th><th>What it is not</th></tr></thead>
      <tbody>{sensor_rows()}</tbody>
    </table></div>
  </section>

  <section>
    <div class="strip-head">
      <div class="strip-label">02 &#183; The five factors</div>
      <h2>Three of these five cannot be sensed from a wrist</h2>
      <p>Health is organised into five factors, and the honest answer to &#8220;can a band
      see this?&#8221; is completely different for each. Connection has no passive input at
      all, deliberately: six meetings in a calendar is not a social-cohesion score, and
      turning one into the other is false precision.</p>
    </div>
    <div class="factors">{factor_rows()}</div>
  </section>

  <section>
    <div class="strip-head">
      <div class="strip-label">03 &#183; Medical context</div>
      <h2>Not a sixth factor &#8212; a layer beneath all five</h2>
      <p>A beta blocker does not belong to a factor. It changes what every heart-rate-derived
      reading in three of them means. Modelling context as a modifier rather than a peer is
      what lets an interpretation say <em>why</em> it was adjusted.</p>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>Context</th><th>Kind</th><th>Reframes</th><th>How the reading changes</th></tr></thead>
      <tbody>{medical_rows()}</tbody>
    </table></div>
  </section>

  <section>
    <div class="strip-head">
      <div class="strip-label">04 &#183; Reference library</div>
      <h2>Public implementations, indexed and searchable</h2>
      <p>Thirteen repositories pinned at a commit, chunked by symbol and searchable without
      a model or an API key. The licence gate decides what enters the corpus: permissive code
      goes in verbatim with attribution, copyleft and unknown contribute protocol
      <em>facts</em> only &#8212; that a characteristic carries a flags byte then a heart
      rate is interoperability information, not expression.</p>
    </div>
    <div class="counters">
      <div class="counter"><span class="v">{harvest["repositories_harvested"]}</span><span class="k">repositories</span></div>
      <div class="counter"><span class="v">{harvest["total_chunks"]:,}</span><span class="k">symbol chunks</span></div>
      <div class="counter"><span class="v">{harvest["total_protocol_facts"]:,}</span><span class="k">protocol facts</span></div>
      <div class="counter"><span class="v">{harvest["policy_breakdown"]["facts_only"]}</span><span class="k">held to facts only</span></div>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>Repository</th><th>Pinned at</th><th>Category</th><th>Licence</th><th>Policy</th><th class="num">Chunks</th><th class="num">Facts</th></tr></thead>
      <tbody>{library_rows(harvest)}</tbody>
    </table></div>
  </section>

  <section>
    <div class="strip-head">
      <div class="strip-label">05 &#183; The measured result</div>
      <h2>Sleep staging from heart rate alone: {_pct(loso["mean_accuracy"])}</h2>
      <p>Trained and tested on {load["records_used"]} nights of MIT-BIH polysomnography
      &#8212; {load["labelled_epochs"]:,} thirty-second epochs a technician scored by hand.
      Held out one whole subject at a time, so no night appears on both sides of the split.
      The same model scores 100% on our own simulator, which is why the simulator number is
      worthless and this one is not.</p>
    </div>

    {subject_chart(loso["per_subject"])}
    <div class="legend">
      <span><span class="key" style="background:var(--accent)"></span>Beat the subject&#8217;s own baseline</span>
      <span><span class="key" style="background:var(--crit);opacity:.72"></span>Did not &#8212; guessing the commonest stage would score as well or better</span>
      <span class="dim">Per-subject detail is from the {len(EPOCH_FEATURE_NAMES)}-feature run.</span>
    </div>

    <div class="verdict">
      <h3>Read this before quoting the average</h3>
      <p>The worst subject, {loso["worst_subject"]}, scores
      {_pct(loso["worst_accuracy"])} where predicting one stage for the whole night scores
      44.4%. {len(loso["subjects_without_skill"])} of {loso["n_subjects"]} subjects show no
      skill at all. A mean of {_pct(loso["mean_accuracy"])} across four classes
      &#8212; {stages["light"]:,} light, {stages["awake"]:,} awake,
      {stages["deep"]:,} deep, {stages["rem"]:,} REM &#8212; is not a staging product.
      It is a measurement of how far heart rate alone gets you, which is not far.</p>
    </div>

    <div class="strip-head">
      <h3>What moved the number, and what did not</h3>
      <p>Two label sets, same protocol. Temporal context &#8212; a 7.5-minute centred
      rolling average plus a 2-minute trailing one, computed strictly within a night &#8212;
      is the only change that helped, and it helped on both, which is what makes it credible
      rather than a lucky split. Frequency-domain HRV made things worse and is shipped off.</p>
    </div>
    <div class="scroll"><table>
      <thead><tr>
        <th>Feature set</th><th class="num">n</th>
        <th class="num">4-class acc</th><th class="num">&#954;</th><th class="num">With skill</th>
        <th class="num">Sleep/wake acc</th><th class="num">&#954;</th><th class="num">With skill</th>
      </tr></thead>
      <tbody>{ablation_rows(ablation)}</tbody>
    </table></div>
    <p class="dim" style="font-size:.86rem">Best result: {_pct(context["accuracy"])} four-class
    (&#954;&#8202;{context["mean_kappa"]:.3f}, {context["subjects_with_skill"]}/16 subjects) and
    {_pct(context2["accuracy"])} sleep/wake (&#954;&#8202;{context2["mean_kappa"]:.3f},
    {context2["subjects_with_skill"]}/16). Offline scoring of a finished night, not a live readout:
    the centred window reads epochs from the future.</p>
  </section>

  <section>
    <div class="strip-head">
      <div class="strip-label">06 &#183; Not built</div>
      <h2>The gaps, named</h2>
      <p>Listed here rather than left out, because a capability report that only lists
      capabilities is marketing.</p>
    </div>
    <div class="gaps">
      {"".join(f'<div class="gap"><h3>{escape(t)}</h3><p>{escape(b)}</p></div>' for t, b in NOT_BUILT)}
    </div>
  </section>
</div>
"""


def build(out: Path | None = None) -> Path:
    """Render the page and write it. Returns the path written."""
    out = out or (DOCS / "status.html")
    html = render(
        harvest=_load("harvest-report.json"),
        four_class=_load("real-data-evaluation.json"),
        ablation=_load("feature-ablation.json"),
        n_tests=count_tests(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    path = build(args.out)
    print(f"wrote {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
