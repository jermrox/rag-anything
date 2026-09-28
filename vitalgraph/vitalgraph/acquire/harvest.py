"""Running a harvest: clone the target list, ingest it, report what happened.

This is the layer that turns a list of repository names into a corpus. It
deliberately does not stop on failure. A harvest of a dozen upstreams will
have some that move, rename, or refuse -- recording those as failed entries
and continuing produces a usable corpus plus an honest account of what is
missing, where aborting produces neither.

The report is the point as much as the corpus is. ``HarvestReport.summary()``
names, per repository, the commit that was fetched, the license that was
detected, whether that detection matched what the target list expected, and
what the license gate consequently permits. That table is the audit trail for
the question "where did this answer come from and were we allowed to read it".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from ..ingest.pipeline import IngestPipeline, IngestResult
from .base import LicenseClass, UsePolicy
from .git_clone import CloneError, CloneResult, clone
from .targets import Target, default_selection

#: Where checkouts live by default. Kept outside the repository tree so a
#: harvest never shows up as uncommitted changes.
DEFAULT_CACHE = Path.home() / ".vitalgraph" / "harvest"

#: Per-repository ceiling. Some upstreams are enormous, and a corpus dominated
#: by one project answers every question in that project's idiom.
MAX_FILES_PER_REPO = 1500


@dataclass(frozen=True, slots=True)
class HarvestFailure:
    """A target that could not be harvested, and why."""

    name: str
    url: str
    reason: str


@dataclass
class HarvestEntry:
    """One successfully harvested repository."""

    target: Target
    clone: CloneResult
    ingest: IngestResult

    @property
    def license_matches_expectation(self) -> bool | None:
        """True/False when there is an expectation to check, None when not.

        None is not a pass. It means the target list made no claim, so nothing
        was verified -- a distinction that matters when reading the report.
        """
        expected = self.target.expected_spdx
        if expected is None:
            return None
        finding = self.ingest.license_finding
        return finding is not None and finding.spdx_id == expected

    def summary(self) -> Dict[str, Any]:
        detected = self.ingest.license_finding
        return {
            "name": self.target.name,
            "category": self.target.category.value,
            "url": self.target.url,
            "commit": self.clone.commit,
            "citation_ref": self.clone.citation_ref(),
            "reused_checkout": self.clone.reused,
            "expected_spdx": self.target.expected_spdx,
            "detected_spdx": detected.spdx_id if detected else None,
            "license_matches_expectation": self.license_matches_expectation,
            "policy": self.ingest.use_policy.value,
            "files_ingested": self.ingest.files_ingested,
            "chunks": len(self.ingest.chunks),
            "exact_chunks": sum(1 for c in self.ingest.chunks if c.is_exact),
            "protocol_facts": len(self.ingest.facts),
            "languages": sorted({c.language for c in self.ingest.chunks}),
        }


@dataclass
class HarvestReport:
    """Everything one harvest run produced, successes and failures alike."""

    entries: List[HarvestEntry] = field(default_factory=list)
    failures: List[HarvestFailure] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        return sum(len(e.ingest.chunks) for e in self.entries)

    @property
    def total_facts(self) -> int:
        return sum(len(e.ingest.facts) for e in self.entries)

    def license_mismatches(self) -> List[HarvestEntry]:
        """Entries whose detected license contradicts the target list.

        Surfaced prominently because a mismatch means either an upstream
        relicensed or our expectation was wrong, and both change what the gate
        should be permitting.
        """
        return [e for e in self.entries if e.license_matches_expectation is False]

    def policy_breakdown(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.entries:
            key = entry.ingest.use_policy.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def verbatim_repos(self) -> List[str]:
        return [
            e.target.name
            for e in self.entries
            if e.ingest.use_policy is UsePolicy.VERBATIM
        ]

    def facts_only_repos(self) -> List[str]:
        return [
            e.target.name
            for e in self.entries
            if e.ingest.use_policy is UsePolicy.FACTS_ONLY
        ]

    def summary(self) -> Dict[str, Any]:
        return {
            "repositories_harvested": len(self.entries),
            "repositories_failed": len(self.failures),
            "total_chunks": self.total_chunks,
            "total_protocol_facts": self.total_facts,
            "policy_breakdown": self.policy_breakdown(),
            "verbatim_permitted": self.verbatim_repos(),
            "facts_only": self.facts_only_repos(),
            "license_mismatches": [
                {
                    "name": e.target.name,
                    "expected": e.target.expected_spdx,
                    "detected": (
                        e.ingest.license_finding.spdx_id
                        if e.ingest.license_finding
                        else None
                    ),
                }
                for e in self.license_mismatches()
            ],
            "repositories": [e.summary() for e in self.entries],
            "failures": [
                {"name": f.name, "url": f.url, "reason": f.reason}
                for f in self.failures
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return path


def harvest(
    targets: Sequence[Target] | None = None,
    cache_dir: str | Path = DEFAULT_CACHE,
    pipeline: IngestPipeline | None = None,
    refresh: bool = False,
    max_files_per_repo: int = MAX_FILES_PER_REPO,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[HarvestReport, IngestPipeline]:
    """Clone and ingest ``targets``, returning the report and the pipeline.

    One :class:`IngestPipeline` is shared across every repository on purpose:
    its symbol graph then spans them all, which is what makes a question like
    "how does anyone decode this characteristic" answerable across upstreams
    rather than within a single file.
    """
    targets = list(targets) if targets is not None else default_selection()
    pipeline = pipeline or IngestPipeline()
    report = HarvestReport()
    cache_dir = Path(cache_dir)

    def note(message: str) -> None:
        if on_progress:
            on_progress(message)

    for target in targets:
        note(f"fetching {target.name}")
        try:
            checkout = clone(target.url, cache_dir, name=target.name, refresh=refresh)
        except CloneError as exc:
            report.failures.append(
                HarvestFailure(name=target.name, url=target.url, reason=str(exc))
            )
            note(f"  failed: {exc}")
            continue

        note(f"  at {checkout.short_commit}, ingesting")
        result = pipeline.ingest_directory(
            checkout.path,
            repo=target.name,
            ref=checkout.short_commit,
            max_files=max_files_per_repo,
        )
        entry = HarvestEntry(target=target, clone=checkout, ingest=result)
        report.entries.append(entry)

        detected = result.license_finding
        spdx = detected.spdx_id if detected else None
        note(
            f"  {result.files_ingested} files, {len(result.chunks)} chunks, "
            f"{len(result.facts)} facts, license {spdx or 'UNKNOWN'} "
            f"-> {result.use_policy.value}"
        )
        if entry.license_matches_expectation is False:
            note(
                f"  LICENSE MISMATCH: expected {target.expected_spdx}, "
                f"detected {spdx or 'UNKNOWN'}"
            )

    return report, pipeline


def format_report(report: HarvestReport) -> str:
    """Render a harvest report as a terminal table."""
    lines: List[str] = []
    summary = report.summary()
    lines.append(
        f"harvested {summary['repositories_harvested']} repositories, "
        f"{summary['repositories_failed']} failed"
    )
    lines.append(
        f"{summary['total_chunks']} code chunks, "
        f"{summary['total_protocol_facts']} protocol facts"
    )
    lines.append("")
    header = f"{'repository':<32} {'commit':<9} {'license':<18} {'policy':<11} chunks"
    lines.append(header)
    lines.append("-" * len(header))
    for entry in report.entries:
        detected = entry.ingest.license_finding
        spdx = (detected.spdx_id if detected else None) or LicenseClass.UNKNOWN.value
        flag = " !" if entry.license_matches_expectation is False else ""
        lines.append(
            f"{entry.target.name:<32} {entry.clone.short_commit:<9} "
            f"{spdx:<18} {entry.ingest.use_policy.value:<11} "
            f"{len(entry.ingest.chunks)}{flag}"
        )
    for failure in report.failures:
        lines.append(f"{failure.name:<32} {'-':<9} {'-':<18} {'FAILED':<11} 0")
    if report.license_mismatches():
        lines.append("")
        lines.append("license mismatches (expectation vs detection):")
        for entry in report.license_mismatches():
            detected = entry.ingest.license_finding
            lines.append(
                f"  {entry.target.name}: expected {entry.target.expected_spdx}, "
                f"detected {(detected.spdx_id if detected else None) or 'UNKNOWN'}"
            )
    if report.failures:
        lines.append("")
        lines.append("failures:")
        for failure in report.failures:
            lines.append(f"  {failure.name}: {failure.reason}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vitalgraph.acquire.harvest",
        description="Clone and ingest the curated public-code harvest list.",
    )
    parser.add_argument(
        "--cache", default=str(DEFAULT_CACHE), help="where checkouts are kept"
    )
    parser.add_argument(
        "--only", nargs="*", metavar="NAME", help="harvest only these targets"
    )
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="also fetch targets marked heavy",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-clone even if cached"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=MAX_FILES_PER_REPO,
        help="per-repository file ceiling",
    )
    parser.add_argument("--json", metavar="PATH", help="write the report as JSON")
    args = parser.parse_args(argv)

    from .targets import by_name

    if args.only:
        selection = [by_name(name) for name in args.only]
    else:
        selection = default_selection(include_heavy=args.include_heavy)

    report, _ = harvest(
        selection,
        cache_dir=args.cache,
        refresh=args.refresh,
        max_files_per_repo=args.max_files,
        on_progress=print,
    )
    print()
    print(format_report(report))
    if args.json:
        path = report.write_json(args.json)
        print(f"\nreport written to {path}")
    return 1 if report.failures and not report.entries else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
