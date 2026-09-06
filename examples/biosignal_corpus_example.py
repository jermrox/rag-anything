"""Build a searchable knowledge base of health-signal and BLE wearable code.

Fetches the curated corpus -- PPG signal quality, pulse-wave morphology, HRV
toolkits, and open BLE wearable firmware -- extracts each project's structure,
and indexes it into RAG-Anything so it can be queried alongside your own
device's recorded sessions.

Dry run (no cloning, no API keys, prints what would be indexed):

    python examples/biosignal_corpus_example.py

Fetch the repositories and show real extracted structure:

    python examples/biosignal_corpus_example.py --fetch

Fetch and index into a configured RAG-Anything instance:

    python examples/biosignal_corpus_example.py --fetch --index
"""

import argparse
import asyncio
import logging
from pathlib import Path

from raganything.biosignal import corpus


def show_corpus() -> None:
    print("=" * 78)
    print("CORPUS")
    print("=" * 78)
    for source in corpus.CORPUS:
        print(
            f"\n{source.name}  [{source.kind} / {source.language} / {source.license}]"
        )
        print(f"  {source.url}")
        print(f"  why: {source.relevance}")
        for capability in source.provides:
            print(f"    - {capability}")


def dry_run(workdir: Path) -> None:
    print()
    print("=" * 78)
    print("WHAT WOULD BE INDEXED")
    print("=" * 78)
    total = 0
    for source in corpus.CORPUS:
        checkout = workdir / source.name
        items = corpus.to_content_list(source, checkout if checkout.exists() else None)
        total += len(items)
        state = "fetched" if checkout.exists() else "not fetched"
        print(f"  {source.name:<24} {len(items):>4} items   ({state})")
    print(f"\n  TOTAL {total} content items")


def fetch_all(workdir: Path) -> None:
    print()
    print("=" * 78)
    print("FETCHING")
    print("=" * 78)
    for source in corpus.CORPUS:
        path = corpus.fetch_source(source, workdir)
        print(f"  {source.name:<24} {'ok' if path else 'unavailable'}")


def show_sample(workdir: Path, name: str = "vital_sqi") -> None:
    source = next((s for s in corpus.CORPUS if s.name == name), None)
    checkout = workdir / name
    if source is None or not checkout.exists():
        return
    items = corpus.to_content_list(source, checkout)
    extracted = [i for i in items if "Public API:" in i.get("text", "")]
    if not extracted:
        return
    print()
    print("=" * 78)
    print(f"SAMPLE EXTRACTION FROM {name}")
    print("=" * 78)
    print(extracted[0]["text"][:1200])


async def index(workdir: Path) -> None:  # pragma: no cover - needs a backend
    from raganything import RAGAnything, RAGAnythingConfig

    rag = RAGAnything(config=RAGAnythingConfig(working_dir="./rag_storage"))
    results = await corpus.ingest_corpus(rag, workdir=str(workdir), fetch=False)
    print("\nindexed:", results)

    answer = await rag.aquery(
        "Which signal quality indices are available for deciding whether a PPG "
        "segment is usable, and which project implements them?"
    )
    print("\n", answer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="clone the repositories")
    parser.add_argument("--index", action="store_true", help="index into RAG-Anything")
    parser.add_argument("--workdir", default="./corpus_checkouts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    workdir = Path(args.workdir)

    show_corpus()
    if args.fetch:
        fetch_all(workdir)
    dry_run(workdir)
    show_sample(workdir)

    if args.index:
        asyncio.run(index(workdir))
    else:
        print("\nRe-run with --fetch to clone, and --index to build the graph.")


if __name__ == "__main__":
    main()
