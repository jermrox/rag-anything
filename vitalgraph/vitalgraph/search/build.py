"""Turning a harvest into a searchable index, and querying it from a terminal.

This is the end of the keyless path: clone, ingest, index, ask. Nothing here
touches a language model or the network, so the corpus is answerable the moment
the harvest finishes.

Three kinds of document go in, and the reason they are separate rather than
merged is that they answer different questions:

**code**            what an implementation actually does, one symbol at a time
**protocol_fact**   what a UUID is and how its payload is framed
**symbol**          where a name is defined and who calls it, across repositories

Merging them would let a long source file outrank the one-line protocol fact
that answers the question exactly, because BM25 scores text length, not
authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from ..acquire.harvest import HarvestReport, harvest
from ..ingest.pipeline import IngestPipeline
from .index import Document, SearchIndex, format_hits

#: Where a built index is cached, beside the harvest checkouts.
DEFAULT_INDEX_PATH = Path.home() / ".vitalgraph" / "search-index.json"


def documents_from_harvest(
    report: HarvestReport, pipeline: IngestPipeline
) -> List[Document]:
    """Build searchable documents from a completed harvest.

    Code chunks go through ``build_content_list`` rather than being read
    directly off disk, so the licence gate has already decided what each one
    says. A copyleft repository contributes its interfaces and documented
    behaviour; its source text never enters the index in the first place,
    which is a stronger guarantee than filtering at query time.
    """
    documents: List[Document] = []

    for entry in report.entries:
        policy = entry.ingest.use_policy.value
        items = pipeline.build_content_list(entry.ingest)
        for position, item in enumerate(items):
            text = item.get("text", "")
            if not text.strip():
                continue
            citation = _citation_from(text, entry.clone.citation_ref())
            documents.append(
                Document(
                    doc_id=f"{entry.target.name}:chunk:{position}",
                    text=text,
                    citation=citation,
                    repo=entry.target.name,
                    kind="code",
                    policy=policy,
                )
            )

    for position, fact in enumerate(pipeline.registry.facts):
        documents.append(
            Document(
                doc_id=f"fact:{position}",
                text=(
                    f"Bluetooth characteristic {fact.uuid}. "
                    f"{fact.kind.replace('_', ' ')}: {fact.detail}"
                ),
                citation=fact.citation(),
                repo=fact.repo,
                kind="protocol_fact",
                policy="verbatim",
            )
        )

    return documents


def _citation_from(text: str, fallback: str) -> str:
    """Recover the ``Source:`` line the chunker wrote into each item."""
    first_line = text.split("\n", 1)[0]
    if first_line.startswith("Source: "):
        return first_line[len("Source: ") :].strip()
    return fallback


def build_index(
    cache_dir: str | Path | None = None,
    include_heavy: bool = False,
    refresh: bool = False,
) -> tuple[SearchIndex, HarvestReport]:
    """Harvest if needed, then index everything it produced."""
    kwargs = {"refresh": refresh}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    from ..acquire.targets import default_selection

    report, pipeline = harvest(default_selection(include_heavy=include_heavy), **kwargs)
    index = SearchIndex()
    index.add_all(documents_from_harvest(report, pipeline))
    return index, report


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vitalgraph.search.build",
        description="Search the harvested corpus. No API key, no network.",
    )
    parser.add_argument("query", nargs="*", help="what to search for")
    parser.add_argument(
        "--rebuild", action="store_true", help="re-harvest and rebuild the index"
    )
    parser.add_argument(
        "--index", default=str(DEFAULT_INDEX_PATH), help="where the index is cached"
    )
    parser.add_argument("--repo", help="restrict results to one repository")
    parser.add_argument(
        "--kind",
        choices=("code", "protocol_fact", "symbol"),
        help="restrict results to one kind of document",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--stats", action="store_true", help="describe the index")
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    if args.rebuild or not index_path.exists():
        print("building index from harvest (this clones on first run)...")
        index, report = build_index()
        index.save(index_path)
        print(
            f"indexed {len(index.documents)} documents from "
            f"{len(report.entries)} repositories -> {index_path}"
        )
    else:
        index = SearchIndex.load(index_path)

    if args.stats:
        for key, value in index.stats().items():
            print(f"{key}: {value}")
        return 0

    if not args.query:
        parser.print_help()
        return 0

    hits = index.search(
        " ".join(args.query), limit=args.limit, repo=args.repo, kind=args.kind
    )
    print(format_hits(hits))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
