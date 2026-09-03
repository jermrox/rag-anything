# VitalGraph

Health analytics and BLE knowledge RAG.

A **self-contained subproject**. It imports `raganything` as a library and never
modifies it — no file under `raganything/`, `tests/`, `examples/`, `docs/` or
`reproduce/` is touched by this project.

---

## What it is

Commercial BLE wearables sell an interpretation layer on top of a data stream
that is, in large part, **openly specified**. VitalGraph is built to know that
stream better than they do, and to make a person's own data answerable in
natural language.

Three things sit in one knowledge graph:

1. **Your own biometrics** — every night reduced to a narrative + metric table
   and inserted as a citable document.
2. **BLE/GATT protocol knowledge** — what each characteristic exposes and what
   is computable from it.
3. **Health & sports-science literature** — so recommendations cite evidence.

Because all three land in the same graph, one question can traverse all three:

> *"Why was my recovery bad last week?"*
> → your RMSSD for those nights, the personal baseline it fell below, and the
> literature that explains the mechanism — each cited.

## Why RAG, and not just a dashboard

A knowledge graph cannot retrieve over 30,000 raw RR intervals. It can retrieve
over *"the night of 2026-03-09, RMSSD 23 ms, 51% below baseline, fragmented
sleep."* Reducing time-series to that form is what
`vitalgraph/bridge/summarizer.py` does, and it is the core of the product.

Summaries are **generated deterministically from measured numbers, not by an
LLM**. Every sentence restates a value that was actually computed, so nothing
hallucinated enters the corpus at ingest time, output is byte-stable (which
keeps `doc_id` stable across re-ingest), and summarising a year of history costs
no tokens. The LLM reasons at query time, over facts it can trust.

## The BLE fact this is built on

The standard GATT **Heart Rate Measurement** characteristic (`0x2A37`)
optionally carries beat-to-beat RR intervals, in units of 1/1024 s. From that
one published field you can compute RMSSD, SDNN, pNN50, respiratory rate (via
respiratory sinus arrhythmia), and — with an accelerometer — sleep staging and a
recovery score.

That is most of what a premium recovery score is made of, from an open
specification. `vitalgraph/ble/gatt.py` implements the decoder and maps
characteristics to the signals they unlock:

```bash
curl 'http://127.0.0.1:8770/api/protocol/derivable?uuids=0x2A37'
```

## Quick start

```bash
pip install -e "vitalgraph[api,dev]"
vitalgraph-server                     # http://127.0.0.1:8770
```

Then click **Seed demo data** — nine simulated nights, the last two scripted as
poor recovery — and the dashboard fills in. **Connect BLE device** streams from
real hardware via Web Bluetooth (Chrome/Edge, over HTTPS or localhost).

RAG answers additionally need `raganything` and an LLM key:

```bash
pip install -e ".[all]"                # from the repository root
export LLM_BINDING_API_KEY=sk-...
curl -X POST localhost:8770/api/rag/ingest
```

Without those, every analytics feature still works and the RAG endpoints return
a clear `503` rather than failing obscurely.

## Reading code at scale

VitalGraph ingests source code as *symbols*, not as flat text. RAG-Anything is
document-centric and has no notion of a function, so `ingest/code_chunker.py`
fills that gap: Python is parsed with the standard library `ast` (exact), other
languages fall back to brace matching (approximate, and **marked as such** —
a symbol graph that hides uneven extraction quality produces confidently wrong
cross-repo answers).

```bash
curl -X POST localhost:8770/api/code/ingest \
  -H 'Content-Type: application/json' \
  -d '{"path": "/path/to/checkout", "repo": "vendor/sdk", "ref": "a1b2c3d"}'

curl 'localhost:8770/api/code/search?q=decode'
curl 'localhost:8770/api/protocol/facts?uuid=0x2A37'
```

Answers cite `repo@sha:path#L66-L128` — real upstream lines, not opaque chunks.

### The license gate

Every repository's license is identified before a byte of it is indexed, and
that class decides what may enter the corpus:

| License class | Source verbatim | Protocol / behavioural facts |
|---|---|---|
| permissive | ✅ with attribution | ✅ |
| copyleft | ❌ interface + docs only | ✅ |
| proprietary / **unknown** | ❌ | ✅ |

Facts — a UUID, a byte layout, which bit is the RR-present flag — are
interoperability information, not creative expression, and stay usable no
matter where they were learned. Verbatim copyleft *source* is what stays out,
because a RAG that ingests GPL code can emit it into a proprietary product.
An unlicensed repository is gated as strictly as a proprietary one: absence of
a license is absence of permission.

So a GPL repository still contributes this, and nothing more:

```
Symbol `decode_heart_rate_measurement` (function) in ble/gatt.py, lines 66-128.
Signature: def decode_heart_rate_measurement(data: bytes) -> HeartRateMeasurement
Documented behaviour: byte 0 flags; bit 4 = RR intervals present; uint16 LE …
Source text withheld under the licensing policy.
```

### Mining protocol facts

`protocol/extractor.py` recovers wire formats from implementations: UUIDs
(including SIG UUIDs written in 128-bit form, and vendor-proprietary ones),
`struct` formats, byte offsets and bit masks, each attributed to the nearest
preceding UUID. Run against this repository's own `ble/gatt.py` it recovers
`0x2A37`'s layout unaided — byte 0 is flags, masks `0x01`–`0x10`, `<H`
little-endian reads. Point it at a vendor SDK and it does the same to theirs.

Facts corroborated across two or more independent repositories are marked as
such: one project's quirk is not a protocol fact; two agreeing is evidence.

## Layout

| Path | Role |
|---|---|
| `vitalgraph/biometrics/` | canonical signal model, SQLite store, HRV maths |
| `vitalgraph/ble/` | GATT decoders + deterministic hardware simulator |
| `vitalgraph/bridge/` | **time-series → `content_list` → knowledge graph** |
| `vitalgraph/rag.py` | RAGAnything facade + health system prompt |
| `vitalgraph/api/` | FastAPI backend |
| `vitalgraph/web/` | single-file, buildless UI with Web Bluetooth |
| `vitalgraph/acquire/` | provenance, SPDX detection, **the license gate** |
| `vitalgraph/ingest/` | symbol-aware code chunking, symbol graph, pipeline |
| `vitalgraph/protocol/` | BLE protocol-fact mining and registry |

The analytics core depends on **nothing outside the standard library**, so it
runs and tests anywhere. FastAPI, RAGAnything and the harvesters are extras.

## Tests

```bash
cd vitalgraph && pytest tests -q      # 143 tests, no network, no LLM, no hardware
```

Run per-file if memory is tight — the whole suite in one process can exhaust a
small container.

`vitalgraph/ble/simulator.py` substitutes for hardware: seeded and deterministic,
with a `recovery` dial that jointly suppresses HRV, raises resting heart rate and
fragments sleep — so the whole pipeline is testable end to end without a device.

The root suite is unaffected: the repository root pins `testpaths = ["tests"]`,
which resolves relative to the root and never reaches `vitalgraph/tests/`.

## Roadmap

- **M1 — done.** Signal model, store, HRV with artifact correction, GATT
  decoder, simulator, summariser bridge, API, web UI.
- **M2 — done.** Symbol-aware code chunking, cross-repository symbol graph,
  provenance and SPDX license gate, BLE protocol-fact mining, and the API
  surface for all of it.
- **M3 — network harvesters.** GitHub code search, Bluetooth SIG specs,
  arXiv/PubMed, feeding the same gated pipeline.
- **M4 — machine learning** (`[ml]` extra: numpy + scikit-learn). Multivariate
  anomaly detection against a personal baseline, recovery forecasting, learned
  sleep staging and event classification. Note: a stager trained on
  `ble/simulator.py` learns the simulator, not physiology — it proves the
  pipeline, and real capability needs Sleep-EDF or MESA with subject-level
  splits.
- **M5 — clinical.** Lab/imaging/discharge ingestion, LOINC-coded FHIR
  observations, evidence-graded decision support, with mode gating, a
  hash-chained audit log, red-flag escalation and a PHI consent gate.

## Scope note

VitalGraph reports **wellness metrics, not medical diagnoses**. That framing is
enforced in the system prompt (`vitalgraph/rag.py`), because presenting these
outputs as diagnosis would move the product into medical-device regulatory
scope.
