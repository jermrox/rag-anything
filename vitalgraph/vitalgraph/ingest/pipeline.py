"""Ingestion orchestration -- and the single place the license gate is applied.

Everything that enters the knowledge graph from source code passes through
here. Chunking, symbol indexing and protocol-fact extraction all happen against
a repository whose license has been identified first, so no path exists that
inserts code without having decided what may be inserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..acquire.base import LicenseClass, Provenance, UsePolicy, now_utc, sha256_of
from ..acquire.licensing import LICENSE_FILENAMES, LicenseFinding, detect_license
from ..protocol.extractor import ProtocolFact, ProtocolRegistry, extract_facts
from .code_chunker import CodeChunk, chunk_source, language_for, to_content_list
from .symbols import SymbolGraph

#: Directories never worth ingesting. Skipping them is the difference between
#: indexing a project and indexing its dependencies.
SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    "target",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    "vendor",
    ".idea",
    ".vscode",
}

#: Refuse absurdly large files: a minified bundle or a generated blob is
#: mostly noise and will dominate a corpus by sheer volume.
MAX_FILE_BYTES = 512_000


@dataclass
class IngestResult:
    """What one ingestion run produced."""

    repo: str
    ref: str | None = None
    license_finding: LicenseFinding | None = None
    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    chunks: List[CodeChunk] = field(default_factory=list)
    facts: List[ProtocolFact] = field(default_factory=list)
    skipped_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def use_policy(self) -> UsePolicy:
        if self.license_finding is None:
            return UsePolicy.FACTS_ONLY
        from ..acquire.base import policy_for

        return policy_for(self.license_finding.license_class)

    def summary(self) -> Dict[str, Any]:
        return {
            "repo": self.repo,
            "ref": self.ref,
            "license": {
                "spdx_id": self.license_finding.spdx_id
                if self.license_finding
                else None,
                "class": (
                    self.license_finding.license_class.value
                    if self.license_finding
                    else LicenseClass.UNKNOWN.value
                ),
                "confidence": self.license_finding.confidence
                if self.license_finding
                else 0.0,
                "evidence": self.license_finding.evidence
                if self.license_finding
                else "",
            },
            "policy": self.use_policy.value,
            "files_seen": self.files_seen,
            "files_ingested": self.files_ingested,
            "files_skipped": self.files_skipped,
            "chunks": len(self.chunks),
            "protocol_facts": len(self.facts),
            "languages": sorted({c.language for c in self.chunks}),
            "exactly_parsed_chunks": sum(1 for c in self.chunks if c.is_exact),
            "skipped_reasons": self.skipped_reasons,
        }


def detect_repo_license(root: Path) -> LicenseFinding:
    """Identify the license governing a checkout.

    A repository's LICENSE file governs its source files, so it is read once
    and applied to everything beneath it. When no license file exists the
    result is UNKNOWN, which the gate treats as restrictively as proprietary --
    an unlicensed repository grants no permission to reproduce its code.
    """
    for name in LICENSE_FILENAMES:
        candidate = root / name
        if candidate.is_file():
            try:
                return detect_license(
                    candidate.read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                continue
    return LicenseFinding(
        None, LicenseClass.UNKNOWN, 0.0, "no license file found in repository root"
    )


class IngestPipeline:
    """Chunk, index and extract from source, honouring the license gate."""

    def __init__(
        self,
        symbol_graph: SymbolGraph | None = None,
        registry: ProtocolRegistry | None = None,
    ) -> None:
        self.symbols = symbol_graph or SymbolGraph()
        self.registry = registry or ProtocolRegistry()

    def ingest_text(
        self,
        source: str,
        path: str,
        repo: str = "local",
        ref: str | None = None,
        language: str | None = None,
    ) -> tuple[List[CodeChunk], List[ProtocolFact]]:
        """Chunk one file and index what it yields."""
        chunks = chunk_source(source, path, language)
        self.symbols.add_chunks(chunks, repo=repo, ref=ref)
        facts = extract_facts(source, path, repo=repo, ref=ref)
        self.registry.add(facts)
        return chunks, facts

    def ingest_directory(
        self,
        root: str | Path,
        repo: str | None = None,
        ref: str | None = None,
        license_override: LicenseClass | None = None,
        include_suffixes: Sequence[str] | None = None,
        max_files: int = 5000,
    ) -> IngestResult:
        """Walk a checkout, ingesting every supported source file.

        Args:
            root: directory to walk.
            repo: name recorded in citations; defaults to the directory name.
            ref: commit SHA or tag, recorded in citations.
            license_override: bypass detection. Use only when the license is
                known out of band -- it overrides the gate's input.
            include_suffixes: restrict to these file extensions.
            max_files: hard ceiling, so a mistargeted path cannot run away.
        """
        root = Path(root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"{root} is not a directory")

        repo = repo or root.name
        finding = detect_repo_license(root)
        if license_override is not None:
            finding = LicenseFinding(
                spdx_id=finding.spdx_id,
                license_class=license_override,
                confidence=1.0,
                evidence=f"caller override: {license_override.value}",
            )

        result = IngestResult(repo=repo, ref=ref, license_finding=finding)

        def skip(reason: str) -> None:
            result.files_skipped += 1
            result.skipped_reasons[reason] = result.skipped_reasons.get(reason, 0) + 1

        for path in sorted(root.rglob("*")):
            if result.files_ingested >= max_files:
                break
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue

            result.files_seen += 1
            rel = str(path.relative_to(root))

            if language_for(rel) is None:
                skip("unsupported_language")
                continue
            if include_suffixes and path.suffix.lower() not in include_suffixes:
                skip("filtered_by_suffix")
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    skip("too_large")
                    continue
                source = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                skip("unreadable")
                continue
            if not source.strip():
                skip("empty")
                continue

            chunks, facts = self.ingest_text(source, rel, repo=repo, ref=ref)
            result.chunks.extend(chunks)
            result.facts.extend(facts)
            result.files_ingested += 1

        return result

    def provenance_for(
        self, result: IngestResult, source_url: str | None = None
    ) -> Provenance:
        """Build the provenance record governing an ingestion run."""
        finding = result.license_finding
        return Provenance(
            source_url=source_url or f"repo://{result.repo}",
            retrieved_at=now_utc(),
            content_sha256=sha256_of(
                f"{result.repo}@{result.ref}:{len(result.chunks)}"
            ),
            license_class=finding.license_class if finding else LicenseClass.UNKNOWN,
            spdx_id=finding.spdx_id if finding else None,
            upstream_ref=result.ref,
        )

    def build_content_list(
        self, result: IngestResult, source_url: str | None = None
    ) -> List[Dict[str, Any]]:
        """Assemble everything from a run into insertable content.

        Code chunks pass through :func:`to_content_list`, which applies the
        gate. The symbol and protocol tables are derived facts and are always
        included -- that is the whole point of the distinction.
        """
        provenance = self.provenance_for(result, source_url)
        items = to_content_list(
            result.chunks, provenance, repo=result.repo, ref=result.ref
        )
        items.extend(self.symbols.to_content_list())
        items.extend(self.registry.to_content_list())
        return items

    async def push_to_rag(
        self, rag: Any, result: IngestResult, source_url: str | None = None
    ) -> Dict[str, Any]:
        """Insert an ingestion run into the knowledge graph.

        ``rag`` is a :class:`~vitalgraph.rag.VitalGraphRAG`-compatible object
        exposing the underlying RAGAnything instance.
        """
        content = self.build_content_list(result, source_url)
        citation = f"code://{result.repo}" + (f"@{result.ref}" if result.ref else "")
        target = getattr(rag, "_rag", rag)
        await target.insert_content_list(
            content_list=content,
            file_path=citation,
            doc_id=f"vg-code-{sha256_of(citation)[:16]}",
        )
        return {
            "citation": citation,
            "items": len(content),
            "policy": result.use_policy.value,
        }
