"""Cross-repository symbol graph.

A per-file chunk answers "what does this function do". A symbol *graph*
answers the question that actually matters for competitive work: "how does
**anyone** decode this packet" -- across every repository ingested, at once.

Extraction quality is tracked per symbol rather than assumed. Python parses
exactly; Kotlin and C are recovered by brace matching and are sometimes wrong.
A graph that hides that distinction produces confidently incorrect cross-repo
answers, so every node reports how much of its evidence was exactly parsed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from .code_chunker import EXTRACTION_AST, CodeChunk


@dataclass(frozen=True, slots=True)
class SymbolSite:
    """One place a symbol is defined."""

    qualname: str
    path: str
    language: str
    start_line: int
    end_line: int
    repo: str
    ref: str | None
    kind: str
    exact: bool
    signature: str = ""

    def citation(self) -> str:
        prefix = f"{self.repo}@{self.ref}" if self.ref else self.repo
        return f"{prefix}:{self.path}#L{self.start_line}-L{self.end_line}"


@dataclass
class SymbolNode:
    """Everything known about one symbol name across all sources."""

    name: str
    definitions: List[SymbolSite] = field(default_factory=list)
    referenced_by: List[SymbolSite] = field(default_factory=list)

    @property
    def repos(self) -> Tuple[str, ...]:
        return tuple(sorted({s.repo for s in self.definitions}))

    @property
    def languages(self) -> Tuple[str, ...]:
        return tuple(sorted({s.language for s in self.definitions}))

    @property
    def confidence(self) -> float:
        """Fraction of definitions recovered by an exact parser (0.0-1.0).

        Callers should surface this rather than presenting every cross-repo
        answer with equal authority.
        """
        if not self.definitions:
            return 0.0
        return sum(1 for s in self.definitions if s.exact) / len(self.definitions)

    @property
    def is_cross_repo(self) -> bool:
        return len(self.repos) > 1


class SymbolGraph:
    """Symbols, their definitions, and who references them."""

    def __init__(self) -> None:
        self._nodes: Dict[str, SymbolNode] = {}
        self._by_repo: Dict[str, set[str]] = defaultdict(set)

    def __len__(self) -> int:
        return len(self._nodes)

    def _node(self, name: str) -> SymbolNode:
        node = self._nodes.get(name)
        if node is None:
            node = SymbolNode(name=name)
            self._nodes[name] = node
        return node

    def add_chunks(
        self, chunks: Iterable[CodeChunk], repo: str = "local", ref: str | None = None
    ) -> int:
        """Index chunks from one repository. Returns symbols touched."""
        touched = 0
        for chunk in chunks:
            site = SymbolSite(
                qualname=chunk.qualname,
                path=chunk.path,
                language=chunk.language,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                repo=repo,
                ref=ref,
                kind=chunk.kind,
                exact=chunk.extraction == EXTRACTION_AST,
                signature=chunk.signature,
            )
            for defined in chunk.defines or (chunk.qualname,):
                self._node(defined).definitions.append(site)
                self._by_repo[repo].add(defined)
                touched += 1
            for referenced in chunk.references:
                self._node(referenced).referenced_by.append(site)
        return touched

    def get(self, name: str) -> SymbolNode | None:
        return self._nodes.get(name)

    def definitions_of(self, name: str) -> List[SymbolSite]:
        node = self._nodes.get(name)
        return list(node.definitions) if node else []

    def referrers_of(self, name: str) -> List[SymbolSite]:
        node = self._nodes.get(name)
        return list(node.referenced_by) if node else []

    def search(self, fragment: str, limit: int = 25) -> List[SymbolNode]:
        """Case-insensitive substring search over symbol names.

        Ranked by how widely a symbol appears: a decoder implemented in six
        repositories is more informative than one appearing once.
        """
        needle = fragment.lower()
        hits = [n for name, n in self._nodes.items() if needle in name.lower()]
        hits.sort(key=lambda n: (-len(n.repos), -len(n.definitions), n.name))
        return hits[:limit]

    def cross_repo_symbols(self, min_repos: int = 2) -> List[SymbolNode]:
        """Symbols defined in several repositories.

        These are the convergent implementations -- where independent vendors
        solved the same problem, which is exactly where the transferable
        knowledge lives.
        """
        hits = [n for n in self._nodes.values() if len(n.repos) >= min_repos]
        hits.sort(key=lambda n: (-len(n.repos), n.name))
        return hits

    def stats(self) -> Dict[str, Any]:
        exact = sum(1 for n in self._nodes.values() for s in n.definitions if s.exact)
        total = sum(len(n.definitions) for n in self._nodes.values())
        return {
            "symbols": len(self._nodes),
            "definitions": total,
            "exactly_parsed": exact,
            "exact_fraction": round(exact / total, 4) if total else 0.0,
            "repos": sorted(self._by_repo),
            "cross_repo_symbols": len(self.cross_repo_symbols()),
        }

    def to_content_list(
        self, min_repos: int = 2, limit: int = 200
    ) -> List[Dict[str, Any]]:
        """Render the cross-repo view as a table item for the knowledge graph.

        Only convergent symbols are emitted. Dumping every symbol would swamp
        retrieval with noise; the value is in what appears more than once.
        """
        nodes = self.cross_repo_symbols(min_repos)[:limit]
        if not nodes:
            return []

        rows = [
            "| Symbol | Repositories | Languages | Definitions | Parse confidence |",
            "| --- | --- | --- | --- | --- |",
        ]
        for n in nodes:
            rows.append(
                f"| `{n.name}` | {', '.join(n.repos)} | {', '.join(n.languages)} "
                f"| {len(n.definitions)} | {n.confidence:.0%} |"
            )

        return [
            {
                "type": "table",
                "table_body": "\n".join(rows),
                "table_caption": [
                    f"Symbols implemented across {min_repos}+ independent repositories"
                ],
                "table_footnote": [
                    "Parse confidence is the fraction of definitions recovered by an "
                    "exact parser rather than by brace matching; treat low-confidence "
                    "rows as leads to verify, not as established fact."
                ],
                "page_idx": 0,
            }
        ]
