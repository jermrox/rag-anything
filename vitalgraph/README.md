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

## Layout

| Path | Role |
|---|---|
| `vitalgraph/biometrics/` | canonical signal model, SQLite store, HRV maths |
| `vitalgraph/ble/` | GATT decoders + deterministic hardware simulator |
| `vitalgraph/bridge/` | **time-series → `content_list` → knowledge graph** |
| `vitalgraph/rag.py` | RAGAnything facade + health system prompt |
| `vitalgraph/api/` | FastAPI backend |
| `vitalgraph/web/` | single-file, buildless UI with Web Bluetooth |
| `vitalgraph/{acquire,ingest,protocol}/` | reserved for M2/M3 (see Roadmap) |

The analytics core depends on **nothing outside the standard library**, so it
runs and tests anywhere. FastAPI, RAGAnything and the harvesters are extras.

## Tests

```bash
cd vitalgraph && pytest tests -q      # 60 tests, no network, no LLM, no hardware
```

`vitalgraph/ble/simulator.py` substitutes for hardware: seeded and deterministic,
with a `recovery` dial that jointly suppresses HRV, raises resting heart rate and
fragments sleep — so the whole pipeline is testable end to end without a device.

The root suite is unaffected: the repository root pins `testpaths = ["tests"]`,
which resolves relative to the root and never reaches `vitalgraph/tests/`.

## Roadmap

- **M1 — done.** Signal model, store, HRV with artifact correction, GATT
  decoder, simulator, summariser bridge, API, web UI.
- **M2 — acquisition.** Harvest public BLE/wearable code, Bluetooth SIG specs
  and literature. Every artifact carries provenance and an SPDX license class;
  the license gate keeps copyleft *source* out of the corpus while keeping
  protocol *facts* (UUIDs, byte layouts, algorithms) usable from every source —
  those are interoperability information, not creative expression. Plus an
  AST-aware code chunker, since RAG-Anything is document-centric.
- **M3 — protocol registry.** Mine UUIDs, services, byte-layout decoders and
  vendor framings into a queryable SQLite registry whose `derivable` table
  answers "what can this hardware actually measure?" mechanically.
- **M4 — depth.** Lomb-Scargle LF/HF (correct for unevenly sampled RR series),
  respiratory rate, sleep staging, readiness, cohort norms, bulk import.

## Scope note

VitalGraph reports **wellness metrics, not medical diagnoses**. That framing is
enforced in the system prompt (`vitalgraph/rag.py`), because presenting these
outputs as diagnosis would move the product into medical-device regulatory
scope.
