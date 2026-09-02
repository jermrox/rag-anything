# The biosignal query layer

*How questions about your own physiology get answered without overstating what
the data supports.*

This document covers `raganything/biosignal/{store,timeseries,router,verify,query}.py`.
For why the subsystem exists at all, see
[ble_fitness_gap_analysis.md](ble_fitness_gap_analysis.md).

---

## 1. The problem with asking a retriever for a number

Ingestion records, for every metric it could not compute to a standard worth
reporting, that it was **withheld** and why. That care is undone the moment a
language model reads a figure out of a retrieved table and states it anyway.

And retrieval cannot do arithmetic at all. *"Is my RMSSD trending down over six
weeks?"* is not a search problem — it is a least-squares fit over eleven
sessions, four of which should be excluded. Similarity search returns the six
chunks that mention RMSSD and invites the model to eyeball a trend.

So the layer splits the work:

```
question
   │
   ├─ router.py ─────────► deterministic? relational? both?
   │
   ├─ timeseries.py ─────► arithmetic over stored reports   (exact, no LLM)
   │
   ├─ retrieval ─────────► explanation, with the computed facts pinned
   │
   └─ verify.py ─────────► every claim checked before it is returned
```

**Every number the user reads comes from arithmetic. Generation is confined to
explanation.** For a hybrid question the deterministic results are computed
first and handed to the model as fixed facts to explain *around*, never to
reproduce.

---

## 2. The report store

`SessionReport`s were previously computed and discarded after indexing. They are
now persisted under `<working_dir>/biosignal/`:

```
reports/<safe_session_id>.json   source of truth, atomic write
index.jsonl                      flattened cache, rebuildable, last-wins
```

Two properties matter more than the format:

**It overwrites by session id.** The knowledge graph will not update a document
under a `doc_id` it already knows — an insert is discarded, not applied. So a
re-analysed session may never reach the graph while remaining perfectly correct
in the store. The store is therefore the numeric source of truth, and
`index_report` writes to it even when it skips the graph insert.

**It exposes values only through `metrics` and `withheld`.** `ReportRecord.metric()`
returns `None` for a withheld metric exactly as it does for an absent one. The
diagnostic `hrv` block — which still holds values the quality gate rejected —
is deliberately not part of the read model. Reading past a withholding decision
should require deciding to, not merely picking the wrong attribute.

---

## 3. Deterministic statistics

`timeseries.py` computes over records with no LLM and no network. Three rules
hold throughout.

**A withheld metric is never silently replaced.** The session is excluded and
its stored reason travels verbatim into the result. No related metric is
substituted.

**The denominator is always visible.** Every result reports how many sessions
contributed and how many were excluded, with reasons. A mean over three of
twenty sessions still computes — refusing there would be over-strict — but
`representative` goes false and the note says so.

**A direction is claimed only when the statistics support one.** `trend()`
reports `rising` or `falling` only when the 95% confidence interval on the OLS
slope excludes zero *and* a Theil–Sen slope agrees on the sign. The interval
uses a hardcoded t-table (no scipy), so it is exactly reproducible. The robust
check is the backstop for the single-outlier "trend" that a six-week question
invites.

A query-time `min_quality` can only ever *remove* sessions. It cannot
un-withhold anything: withholding decisions were made at analysis time and are
never recomputed.

`daily_load_series()` emits one entry per calendar day **including zeros for
rest days** — which `analytics.load.ewma_load` documents as a requirement, and
which nothing in the subsystem could previously supply.

---

## 4. Routing

Rules first: compiled lexicons over metric aliases, statistic words, trend words
and relational words, plus a stdlib date parser with an injectable `now`. Free,
instant, exhaustively testable.

A language model runs only when rule confidence is low, and everything it
returns is validated against the known metric names — it may choose among the
metrics this system computes, never name one it does not.

The fallback is `HYBRID`, not a specific route. Defaulting to deterministic
would invent a scope; defaulting to retrieval would throw away the arithmetic.

Ambiguity is resolved where the data allows and refused where it does not. Asked
*"what was my average power on the 14th?"* with six months of history, the
router returns `REFUSE` and a clarifying question rather than picking a month —
because guessing produces an exact-looking number for the wrong day, which is
worse than asking.

---

## 5. Retrieval, and why scope is checked rather than enforced

`QueryParam` in LightRAG 1.4.16 has no `ids` or `doc_ids` field, and
RAG-Anything adds no filtering. **Retrieval genuinely cannot be scoped to a date
range.** Four levers, in order:

1. **Keyword pre-seeding** — resolved session ids and ISO dates go into
   `ll_keywords`, metric names and `signal quality` / `withheld` / `provenance`
   into `hl_keywords`. This also skips LightRAG's own keyword-extraction call.
2. **A window instruction** in `QueryParam.user_prompt`. An instruction is not
   enforcement, which is exactly why check 4 of the verifier exists.
3. **`strict_scope=True`** — two-phase: `only_need_prompt=True` returns the
   assembled prompt, out-of-window fragments are dropped, and the model is
   called directly. Real scoping, at the cost of coupling to LightRAG's prompt
   shape, which is why it is opt-in. If filtering leaves nothing, the engine
   sets `context_out_of_scope` rather than silently widening.
4. **The verifier**, below, as enforcement of last resort.

A rejected alternative: a separate `working_dir` per epoch would give real
scoping, but it fragments the graph and defeats the point of putting physiology
alongside plans and literature. It stays documented only as a multi-subject
escape hatch.

---

## 6. Verification

Claim extraction is **regex over a closed lexicon, sentence by sentence** — not
a second model. A verifier that can hallucinate is not a verifier. Working
per-sentence means a metric named in one sentence can never bind to a number in
another, which would manufacture violations the model never committed.

| Check | Catches |
|---|---|
| `WITHHELD_METRIC_ASSERTED` | a value stated for a metric withheld in every session in scope; echoes the stored reason verbatim |
| `UNGATED_TREND_ASSERTED` | a direction the recomputed `trend()` does not support |
| `VALUE_CONTRADICTS_COMPUTATION` | a number matching no session in scope nor their mean, beyond a per-metric tolerance |
| `OUT_OF_SCOPE_SESSION_CITED` | a session or date outside the window — the backstop for §5 |
| `UNSUPPORTED_METRIC_NAME` | a value for something never computed ("recovery score", "body battery") |
| `UNVERIFIABLE` | a numeric claim with no record to check it against |

**Response.** Withheld values, contradicted numbers and ungated trends get
exactly one regeneration with the violations and computed facts fed back; if the
re-check still fails, the answer is replaced by what the data does support.
Out-of-scope citations are **annotated** — a correction block is appended and
the model's own sentences are never edited, because rewriting prose is itself a
way to manufacture a claim.

**Documented limitation.** Regex cannot catch every paraphrase. Two things bound
the gap: the answering prompt requires canonical `metric = value unit` phrasing,
and any numeric claim the checker cannot resolve becomes `UNVERIFIABLE` rather
than passing silently.

---

## 7. Using it

```python
from raganything.biosignal import BiosignalQueryEngine, ReportStore

# Offline: exact arithmetic, no LLM, no network.
engine = BiosignalQueryEngine(store=ReportStore("./rag_storage/biosignal"))
print(engine.compute("is my RMSSD trending down over the last six weeks?").answer)

# With retrieval, for questions arithmetic cannot answer.
engine = BiosignalQueryEngine(rag=rag)          # store resolved from working_dir
answer = await engine.aask("why was recovery poor on the 14th?")
print(answer.answer)
print(answer.verdict.ok, [v.kind.value for v in answer.verdict.violations])
```

`rag` is optional throughout. Given only a store, every arithmetic question is
answerable with nothing generative involved at any point — which is a useful
offline mode, and the reason most of this layer is testable without a model.

---

## 8. Prompts

Biosignal prompts live in `prompts.BIOSIGNAL_PROMPTS`, a module-local dict, and
deliberately **not** in `raganything.prompt.PROMPTS`. Two reasons:
`set_prompt_language()` rebuilds the shared registry from an import-time
snapshot and would erase externally added keys, and
`test_chinese_prompts_have_all_keys` fails on any key without a Chinese
counterpart. A regression test asserts the shared registry stays unpolluted.

The answer template is validated at import, because the one catastrophic way to
get it wrong — omitting `{context_data}` — fails *silently*: the query runs, the
model answers, and every retrieved chunk was discarded.
