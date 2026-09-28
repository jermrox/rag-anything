"""Ask the corpus a fixed set of questions and record what comes back.

A search index is easy to build and easy to fool yourself about. It returns
ten results for any string, and if nobody reads them the index looks like it
works forever. So this file is a standing exam: real questions with known
right answers, run against the live index, written to a committed file.

The questions are chosen to fail loudly. Each one names a behaviour that
exists in exactly one place in the corpus, so a hit from the wrong repository
is visible as a wrong answer rather than a plausible-looking result. Two of
them exist to catch the failure that already happened once here -- a query for
"sleep staging" returning a thread-sleep helper -- where the top hit matches
the words and answers nothing.

``expect_repo`` is the check. It is not a filter: the search runs unrestricted
and the result records whether the expected repository appeared, so a
regression shows up as a changed answer rather than a silent reordering.

Usage::

    python -m tools.probe_corpus                   # writes docs/corpus-probe.json
    python -m tools.probe_corpus --show            # print the hits too
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from vitalgraph.search.build import DEFAULT_INDEX_PATH
from vitalgraph.search.index import SearchIndex

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class Probe:
    """One question, and where its answer should come from."""

    question: str
    expect_repo: str
    why: str
    """What this question is testing. A probe nobody can justify is a probe
    that will be quietly deleted the first time it fails."""

    known_gap: bool = False
    """This probe is expected to fail, and is kept because it fails.

    Deleting a failing probe to make the run green is the exact dishonesty
    this file exists to prevent. A known gap stays in, is reported
    separately, and is the standing description of what the index cannot do.
    """


PROBES: Tuple[Probe, ...] = (
    # --- the medical-device half of the corpus ---------------------------
    Probe(
        "Dexcom transmitter authentication challenge",
        "xdrip",
        "Pairing with a CGM transmitter is a challenge-response exchange. "
        "xDrip is the only place in the corpus that implements it, so a hit "
        "from anywhere else is a wrong answer, not a near miss.",
    ),
    Probe(
        "calibration slope intercept glucose",
        "xdrip",
        "Turning a raw sensor current into a glucose value is the single most "
        "safety-critical arithmetic in the corpus.",
    ),
    Probe(
        "glucose reading is stale and should not be displayed",
        "xdrip",
        "The refusal this project is built around, implemented against a "
        "device people dose insulin from. If the corpus cannot find it, the "
        "corpus is not earning its place.",
    ),
    Probe(
        "polar ECG stream start",
        "polar-ble-sdk",
        "A vendor exposing raw ECG rather than a heart rate number. The "
        "capability that separates a research strap from a consumer band.",
    ),
    Probe(
        "polar measurement data frame parsing",
        "polar-ble-sdk",
        "Polar's PMD frame format. Expected bleakheart on the first run, on "
        "the theory that the independent reimplementation would be the "
        "clearer read; the SDK's own parser and its tests won instead, and "
        "on inspection they are the fuller answer. The expectation was "
        "wrong, not the ranking.",
    ),
    Probe(
        "electrodermal activity EDA skin conductance",
        "NeuroKit",
        "Expected EmotiBit here, which was a category error: EmotiBit is the "
        "sensor, NeuroKit is the analysis, and the question asks about the "
        "signal. A corpus that answered with firmware would be answering a "
        "question nobody asked.",
    ),
    Probe(
        "refuse to dose when data is missing or old",
        "oref0",
        "The most carefully argued 'we do not know enough' code in "
        "open-source health software, and the model for our own tiers. It "
        "is also the probe that shows the index's limit: this is a question "
        "about intent, and every content word in it is common. BM25 matches "
        "words, so it returns test files containing 'data', 'missing' and "
        "'when'. Answering this needs embeddings or a model reading the "
        "code; it is not a ranking bug to be tuned away.",
        known_gap=True,
    ),
    # --- the questions that catch a word-matching index -------------------
    Probe(
        "heart rate measurement flags byte RR interval",
        "Android-BLE-Library",
        "The canonical 0x2A37 decode. Expected bleak, which was simply "
        "wrong: bleak is transport and carries no heart-rate parser at all, "
        "while Nordic's library has the flags-byte implementation with its "
        "tests. The index knew the corpus better than I did.",
    ),
    Probe(
        "sleep stage classification from epochs",
        "yasa",
        "The probe that already failed once: this returned a thread-sleep "
        "helper and a smart-bulb example before the corpus had any staging "
        "code in it at all.",
    ),
    Probe(
        "R peak detection QRS",
        "wfdb-python",
        "Beat detection. Expected heartrate_analysis_python; wfdb-python's "
        "qrs.py carries the XQRS and GQRS implementations and is the better "
        "answer. Another wrong expectation, kept here with its correction "
        "rather than quietly rewritten.",
    ),
)


def run(index: SearchIndex, limit: int = 5) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for probe in PROBES:
        hits = index.search(probe.question, limit=limit)
        repos = [h.document.repo for h in hits]
        rank = (
            repos.index(probe.expect_repo) + 1 if probe.expect_repo in repos else None
        )
        results.append(
            {
                "question": probe.question,
                "expect_repo": probe.expect_repo,
                "why": probe.why,
                "found_at_rank": rank,
                "known_gap": probe.known_gap,
                "top_repos": repos,
                "hits": [
                    {
                        "rank": i,
                        "citation": h.document.citation,
                        "repo": h.document.repo,
                        "policy": h.document.policy,
                        "score": round(h.score, 2),
                        "matched": sorted(h.matched_terms)[:6],
                    }
                    for i, h in enumerate(hits, 1)
                ],
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--show", action="store_true", help="print every hit")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "corpus-probe.json")
    args = parser.parse_args()

    if not args.index.exists():
        raise SystemExit(
            f"no index at {args.index}; build it with "
            "python -m vitalgraph.search.build --rebuild"
        )
    index = SearchIndex.load(args.index)
    results = run(index, limit=args.limit)

    graded = [r for r in results if not r["known_gap"]]
    answered = sum(1 for r in graded if r["found_at_rank"] is not None)
    first = sum(1 for r in graded if r["found_at_rank"] == 1)
    gaps = [r for r in results if r["known_gap"]]

    for r in results:
        rank = r["found_at_rank"]
        if r["known_gap"]:
            mark = "gap " if rank is None else "GAP CLOSED"
        else:
            mark = "ok  " if rank == 1 else ("~   " if rank else "MISS")
        print(f"{mark} #{rank or '-'}  {r['question']}  -> {r['expect_repo']}")
        if args.show:
            for hit in r["hits"]:
                print(f"       {hit['rank']}. {hit['citation']}  ({hit['score']})")

    payload = {
        "index": str(args.index),
        "documents": len(index.documents),
        "probes": len(PROBES),
        "graded_probes": len(graded),
        "known_gaps": len(gaps),
        "answered_in_top_n": answered,
        "answered_at_rank_1": first,
        "limit": args.limit,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        f"\n{first}/{len(graded)} at rank 1, {answered}/{len(graded)} in the "
        f"top {args.limit}; {len(gaps)} known gap(s) -> {args.out}"
    )


if __name__ == "__main__":
    main()
