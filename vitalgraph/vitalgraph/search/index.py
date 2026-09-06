"""Searching the harvested corpus, with no API key and no network.

The knowledge graph in ``rag_domains`` needs a language model, because building
one means extracting entities and relations from prose. Retrieval does not. A
question like "how is characteristic 0x2A37 decoded" is answered by finding the
right chunks of the right repositories and citing them, and that is lexical
work over text already on disk.

So this is the searchable half of the RAG, built to run with nothing installed
and nothing configured. BM25 over the harvested chunks, the protocol facts and
the symbol graph, with every hit carrying ``repo@commit:path#Lstart-Lend``.

**Why BM25 rather than embeddings.** Embeddings need a model, which needs a key
or a large download. They are also worse for this corpus: the highest-value
queries are exact tokens -- a UUID, a characteristic name, a struct format --
and a query for ``0x2A37`` should return the code that mentions ``0x2A37``, not
the code that is semantically adjacent to heart rate. Term frequency saturation
and length normalisation are what BM25 adds over naive counting, and both
matter when one repository's file is forty times the length of another's.

The tokenizer is code-aware, which is most of the quality. ``decode_heart_rate``
has to match a query for "heart rate"; ``CharacteristicUUID`` has to match
"characteristic"; and ``0x2A37`` must survive as one token rather than
fragmenting into ``0`` and ``x2a37``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

#: Standard BM25 parameters. k1 controls how quickly repeated terms stop
#: adding score; b controls how strongly long documents are penalised.
BM25_K1 = 1.5
BM25_B = 0.75

#: English function words only. Deliberately short, and deliberately free of
#: programming keywords: BM25's IDF already discounts anything ubiquitous, so
#: a stopword list earns its place only by removing words that are never the
#: point of a query.
#:
#: ``value`` and ``data`` were here and had to go. In a corpus of Bluetooth
#: code they are domain vocabulary -- "characteristic value", "payload data" --
#: and dropping them made ``CharacteristicUUIDValue`` unfindable by its own
#: last word. The same argument removed ``class``, ``function`` and ``return``.
STOPWORDS = frozenset(
    """
    a an the and or not is are was were be been being to of in on at for with
    from by as if then else this that these those it its
    """.split()
)

#: A hex literal, an identifier, or a bare word. Hex is matched first so that
#: 0x2A37 survives as a single token: splitting it would make the single most
#: valuable query in this corpus unanswerable.
_TOKEN = re.compile(r"0[xX][0-9a-fA-F]+|[A-Za-z_][A-Za-z0-9_]*|\d+")

#: Split camelCase and PascalCase. Handles acronym runs, so ``UUIDValue``
#: yields ``UUID`` and ``Value`` rather than ``U``, ``U``, ``I``, ``DValue``.
_CAMEL = re.compile(r".+?(?:(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")


def tokenize(text: str) -> List[str]:
    """Split text into search terms, code-aware.

    An identifier contributes both its whole form and its parts, so
    ``decode_heart_rate`` matches a query for "heart rate" and a query for the
    exact function name. Hex literals are preserved and normalised to lower
    case so ``0x2A37`` and ``0x2a37`` are the same term -- the same
    normalisation failure that once made a UUID registry lookup silently
    return nothing.
    """
    terms: List[str] = []
    for match in _TOKEN.finditer(text):
        raw = match.group(0)
        if raw.lower().startswith("0x"):
            terms.append(raw.lower())
            continue
        lowered = raw.lower()
        if lowered not in STOPWORDS and len(lowered) > 1:
            terms.append(lowered)
        parts = [p for p in raw.split("_") if p]
        for part in parts:
            for piece in _CAMEL.findall(part):
                piece_lower = piece.lower()
                if (
                    len(piece_lower) > 1
                    and piece_lower not in STOPWORDS
                    and piece_lower != lowered
                ):
                    terms.append(piece_lower)
    return terms


@dataclass(frozen=True, slots=True)
class Document:
    """One searchable unit, with the citation that makes it checkable."""

    doc_id: str
    text: str
    citation: str
    """``repo@commit:path#Lstart-Lend``, or a protocol-fact reference. A hit
    without one is unusable: the point of the corpus is that an answer can be
    traced back to a line someone can open."""

    repo: str
    kind: str
    """code | protocol_fact | symbol"""

    policy: str = "verbatim"
    """The licence gate's verdict for the source repository. Carried through
    so a caller can tell whether a hit may be quoted or only described."""

    @property
    def may_quote(self) -> bool:
        return self.policy == "verbatim"


@dataclass(frozen=True, slots=True)
class Hit:
    """A search result, with the score that produced it."""

    document: Document
    score: float
    matched_terms: Tuple[str, ...]
    """Which query terms actually hit. Shown because a result whose match is
    entirely on a common word is a different kind of answer from one that
    matched an exact UUID, and only this distinguishes them."""


@dataclass
class SearchIndex:
    """A BM25 index over harvested documents.

    Pure standard library, so it builds and queries with nothing installed.
    """

    documents: List[Document] = field(default_factory=list)
    _term_frequencies: List[Counter] = field(default_factory=list, repr=False)
    _document_frequency: Counter = field(default_factory=Counter, repr=False)
    _postings: Dict[str, List[int]] = field(
        default_factory=lambda: defaultdict(list), repr=False
    )
    _lengths: List[int] = field(default_factory=list, repr=False)
    _average_length: float = 0.0

    def add(self, document: Document) -> None:
        terms = tokenize(document.text)
        counts = Counter(terms)
        index = len(self.documents)
        self.documents.append(document)
        self._term_frequencies.append(counts)
        self._lengths.append(len(terms))
        for term in counts:
            self._document_frequency[term] += 1
            self._postings[term].append(index)
        total = sum(self._lengths)
        self._average_length = total / len(self._lengths) if self._lengths else 0.0

    def add_all(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self.add(document)

    def _idf(self, term: str) -> float:
        """Inverse document frequency, floored at zero.

        The standard BM25 formulation goes negative for a term appearing in
        more than half the corpus. Left unfloored, a document could improve its
        score by *not* matching a common query term, which is incoherent.
        """
        n = len(self.documents)
        df = self._document_frequency.get(term, 0)
        if df == 0:
            return 0.0
        return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))

    def search(
        self,
        query: str,
        limit: int = 10,
        repo: str | None = None,
        kind: str | None = None,
    ) -> List[Hit]:
        """Rank documents against ``query``, most relevant first.

        Only documents containing at least one query term are scored, via the
        postings lists. Scoring the whole corpus per query would be simpler and
        would make an interactive search over seven thousand chunks unusable.
        """
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return []

        candidates: set[int] = set()
        for term in query_terms:
            candidates.update(self._postings.get(term, ()))

        scored: List[Hit] = []
        for index in candidates:
            document = self.documents[index]
            if repo is not None and document.repo != repo:
                continue
            if kind is not None and document.kind != kind:
                continue

            counts = self._term_frequencies[index]
            length = self._lengths[index]
            score = 0.0
            matched: List[str] = []
            for term in set(query_terms):
                frequency = counts.get(term, 0)
                if frequency == 0:
                    continue
                matched.append(term)
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * length / (self._average_length or 1.0)
                )
                score += self._idf(term) * frequency * (BM25_K1 + 1.0) / denominator
            if score > 0.0:
                scored.append(
                    Hit(
                        document=document,
                        score=score,
                        matched_terms=tuple(sorted(matched)),
                    )
                )

        scored.sort(key=lambda hit: (-hit.score, hit.document.doc_id))
        return scored[:limit]

    def stats(self) -> Dict[str, object]:
        return {
            "documents": len(self.documents),
            "unique_terms": len(self._document_frequency),
            "average_length": round(self._average_length, 1),
            "repos": sorted({d.repo for d in self.documents}),
            "kinds": sorted({d.kind for d in self.documents}),
        }

    # --- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write the corpus to disk.

        Only the documents are stored, never the postings. The index rebuilds
        from them deterministically in seconds, and a serialised index would
        silently go stale the moment the tokenizer changed -- which is exactly
        the kind of change this code should stay free to make.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "text": d.text,
                    "citation": d.citation,
                    "repo": d.repo,
                    "kind": d.kind,
                    "policy": d.policy,
                }
                for d in self.documents
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SearchIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index = cls()
        index.add_all(Document(**row) for row in payload["documents"])
        return index


def format_hits(hits: Sequence[Hit], width: int = 96) -> str:
    """Render hits for a terminal, citation first.

    The citation leads because it is the part a reader acts on. A snippet
    without one is a claim; with one it is a reference.
    """
    if not hits:
        return "no matches"
    lines: List[str] = []
    for rank, hit in enumerate(hits, start=1):
        body = hit.document.text
        # Drop the leading ``Source:`` line. It is printed above as the
        # citation, and repeating it inside a truncated snippet spends the
        # whole visible width on a path the reader has already seen.
        if body.startswith("Source: "):
            _, _, body = body.partition("\n")
        snippet = " ".join(body.split())[:width]
        quotable = "" if hit.document.may_quote else "  [facts only]"
        lines.append(f"{rank:>2}. {hit.document.citation}{quotable}")
        lines.append(
            f"    score {hit.score:.2f}  matched: {', '.join(hit.matched_terms)}"
        )
        lines.append(f"    {snippet}")
    return "\n".join(lines)
