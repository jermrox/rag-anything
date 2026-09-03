"""Symbol-aware chunking of source code into knowledge-graph content.

RAG-Anything is document-centric: it parses PDFs, images and tables, and has
no notion of a function or a class. Splitting source code on blank lines or a
fixed token count destroys exactly the structure that makes code answerable, so
this module chunks by *symbol* -- one chunk per function, method or class, with
its qualified name, signature, docstring and exact line span.

Python is parsed with the standard library ``ast``, which is exact. Other
languages fall back to brace matching, which is not. Every chunk records which
method produced it in :attr:`CodeChunk.extraction`, because a symbol graph
built from silently uneven extraction quality yields confidently wrong
cross-repo answers -- knowing when the parse was approximate matters more than
pretending it never is.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..acquire.base import Provenance, UsePolicy

#: Extension -> language name. Only languages we can chunk meaningfully.
LANGUAGE_BY_SUFFIX: Dict[str, str] = {
    ".py": "python",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".js": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
}

#: Extraction methods, in descending order of trustworthiness.
EXTRACTION_AST = "ast"
"""Parsed with a real grammar. Line spans and symbol names are exact."""

EXTRACTION_BRACE = "brace"
"""Recovered by brace matching. Spans are approximate; nested or unusual
constructs may be mis-attributed."""

EXTRACTION_WHOLE_FILE = "whole_file"
"""No structure recovered; the file is one chunk."""

#: Declaration headers for brace-delimited languages. Deliberately loose --
#: precision comes from the brace matcher, not from this pattern.
_DECL = re.compile(
    r"^[ \t]*(?:(?:public|private|internal|protected|static|final|open|"
    r"override|suspend|inline|export|default|async|const|extern)\s+)*"
    r"(?P<kind>fun|func|function|class|struct|interface|enum|object|impl)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """One symbol's worth of source."""

    qualname: str
    kind: str
    """module | class | function | method | fragment"""
    path: str
    """Repository-relative path."""
    language: str
    start_line: int
    """1-indexed, inclusive."""
    end_line: int
    """1-indexed, inclusive."""
    code: str
    signature: str = ""
    docstring: str = ""
    defines: Tuple[str, ...] = field(default_factory=tuple)
    references: Tuple[str, ...] = field(default_factory=tuple)
    extraction: str = EXTRACTION_AST
    decorators: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def line_span(self) -> str:
        return f"L{self.start_line}-L{self.end_line}"

    @property
    def is_exact(self) -> bool:
        """Whether the span and symbol name can be trusted precisely."""
        return self.extraction == EXTRACTION_AST

    def citation(self, repo: str | None = None, ref: str | None = None) -> str:
        """Citation reference resolving to real upstream lines.

        Shaped ``repo@ref:path#Lstart-Lend`` so a reader can open exactly the
        code an answer was drawn from.
        """
        prefix = repo or "local"
        if ref:
            prefix = f"{prefix}@{ref}"
        return f"{prefix}:{self.path}#{self.line_span}"


# --------------------------------------------------------------------------
# Python: exact, via the standard library
# --------------------------------------------------------------------------


def _signature_of(node: ast.AST) -> str:
    """Render a def/class header without its body."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        try:
            args = ast.unparse(node.args)
        except Exception:  # pragma: no cover - unparse is reliable on 3.9+
            args = "..."
        returns = ""
        if node.returns is not None:
            try:
                returns = f" -> {ast.unparse(node.returns)}"
            except Exception:  # pragma: no cover
                returns = ""
        return f"{prefix} {node.name}({args}){returns}"
    if isinstance(node, ast.ClassDef):
        bases = []
        for b in node.bases:
            try:
                bases.append(ast.unparse(b))
            except Exception:  # pragma: no cover
                continue
        suffix = f"({', '.join(bases)})" if bases else ""
        return f"class {node.name}{suffix}"
    return ""


def _referenced_names(node: ast.AST) -> Tuple[str, ...]:
    """Names this node refers to -- the raw material of the symbol graph."""
    names: List[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
        elif isinstance(child, ast.Attribute):
            names.append(child.attr)
    # Stable order, no duplicates: chunk output must be deterministic so the
    # doc_id derived from it is stable across runs.
    return tuple(sorted(set(names)))


def _decorators_of(node: ast.AST) -> Tuple[str, ...]:
    decorators = getattr(node, "decorator_list", [])
    out: List[str] = []
    for d in decorators:
        try:
            out.append(ast.unparse(d))
        except Exception:  # pragma: no cover
            continue
    return tuple(out)


def chunk_python(source: str, path: str) -> List[CodeChunk]:
    """Chunk Python source into one chunk per class and function.

    Methods are emitted as their own chunks (qualified ``Class.method``) *and*
    the class keeps a chunk carrying its header and docstring, so a query can
    match either the type or the specific behaviour.

    Raises:
        SyntaxError: if the source does not parse. Callers should fall back to
            :func:`chunk_generic` rather than dropping the file.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: List[CodeChunk] = []

    def segment(node: ast.AST) -> Tuple[str, int, int]:
        start = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", start) or start
        return "\n".join(lines[start - 1 : end]), start, end

    def emit(node: ast.AST, qualname: str, kind: str) -> None:
        code, start, end = segment(node)
        chunks.append(
            CodeChunk(
                qualname=qualname,
                kind=kind,
                path=path,
                language="python",
                start_line=start,
                end_line=end,
                code=code,
                signature=_signature_of(node),
                docstring=ast.get_docstring(node) or "",
                defines=(qualname,),
                references=_referenced_names(node),
                extraction=EXTRACTION_AST,
                decorators=_decorators_of(node),
            )
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            emit(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            emit(node, node.name, "class")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    emit(item, f"{node.name}.{item.name}", "method")

    if not chunks:
        # A module of only imports and constants still carries meaning.
        module_doc = ast.get_docstring(tree) or ""
        chunks.append(
            CodeChunk(
                qualname=Path(path).stem,
                kind="module",
                path=path,
                language="python",
                start_line=1,
                end_line=max(len(lines), 1),
                code=source,
                docstring=module_doc,
                defines=(Path(path).stem,),
                references=_referenced_names(tree),
                extraction=EXTRACTION_AST,
            )
        )
    return chunks


# --------------------------------------------------------------------------
# Brace-delimited languages: approximate, and says so
# --------------------------------------------------------------------------


def _match_braces(lines: Sequence[str], start_idx: int) -> int:
    """Index of the line closing the block opened at or after ``start_idx``.

    Counts braces outside string literals and line comments. This is a
    heuristic: it is defeated by braces in block comments or in regex
    literals, which is precisely why chunks it produces are marked
    ``EXTRACTION_BRACE`` rather than treated as exact.
    """
    depth = 0
    seen_open = False
    for i in range(start_idx, len(lines)):
        line = lines[i]
        in_string: str | None = None
        j = 0
        while j < len(line):
            ch = line[j]
            if in_string:
                if ch == "\\":
                    j += 2
                    continue
                if ch == in_string:
                    in_string = None
            elif ch in "\"'":
                in_string = ch
            elif ch == "/" and j + 1 < len(line) and line[j + 1] == "/":
                break  # line comment
            elif ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}":
                depth -= 1
                if seen_open and depth <= 0:
                    return i
            j += 1
        # A declaration whose brace never opens within a few lines is not a block.
        if not seen_open and i - start_idx > 3:
            return start_idx
    return len(lines) - 1


def chunk_generic(source: str, path: str, language: str) -> List[CodeChunk]:
    """Chunk a brace-delimited language by declaration headers.

    Used for Kotlin, Swift, JavaScript, C and friends. Falls back to a single
    whole-file chunk when no declarations are recognised, so content is never
    dropped -- only marked as less precisely extracted.
    """
    lines = source.splitlines()
    chunks: List[CodeChunk] = []

    for match in _DECL.finditer(source):
        start_idx = source[: match.start()].count("\n")
        end_idx = _match_braces(lines, start_idx)
        if end_idx <= start_idx:
            continue
        name = match.group("name")
        kind = match.group("kind")
        body = "\n".join(lines[start_idx : end_idx + 1])
        chunks.append(
            CodeChunk(
                qualname=name,
                kind="class"
                if kind in {"class", "struct", "interface", "enum", "object"}
                else "function",
                path=path,
                language=language,
                start_line=start_idx + 1,
                end_line=end_idx + 1,
                code=body,
                signature=lines[start_idx].strip(),
                defines=(name,),
                references=tuple(
                    sorted(
                        set(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", body)) - {name}
                    )
                ),
                extraction=EXTRACTION_BRACE,
            )
        )

    if not chunks:
        chunks.append(
            CodeChunk(
                qualname=Path(path).stem,
                kind="fragment",
                path=path,
                language=language,
                start_line=1,
                end_line=max(len(lines), 1),
                code=source,
                defines=(),
                references=(),
                extraction=EXTRACTION_WHOLE_FILE,
            )
        )
    return chunks


def language_for(path: str) -> str | None:
    return LANGUAGE_BY_SUFFIX.get(Path(path).suffix.lower())


def chunk_source(
    source: str, path: str, language: str | None = None
) -> List[CodeChunk]:
    """Chunk any supported source file, choosing the best available parser.

    Python that fails to parse degrades to brace chunking rather than being
    dropped -- harvested code is often a partial file or a different dialect.
    """
    language = language or language_for(path)
    if language is None:
        return []
    if language == "python":
        try:
            return chunk_python(source, path)
        except SyntaxError:
            return chunk_generic(source, path, "python")
    return chunk_generic(source, path, language)


# --------------------------------------------------------------------------
# Content-list emission -- where the license gate actually bites
# --------------------------------------------------------------------------


def _facts_only_text(chunk: CodeChunk) -> str:
    """Describe a symbol without reproducing its body.

    This is what enters the corpus for copyleft, proprietary and unlicensed
    material: the interface and behaviour, which are interoperability facts,
    with the expression withheld.
    """
    parts = [
        f"Symbol `{chunk.qualname}` ({chunk.kind}) in {chunk.path}, "
        f"lines {chunk.start_line}-{chunk.end_line}, {chunk.language}."
    ]
    if chunk.signature:
        parts.append(f"Signature: {chunk.signature}")
    if chunk.docstring:
        parts.append(f"Documented behaviour: {chunk.docstring}")
    if chunk.references:
        parts.append(f"Calls or references: {', '.join(chunk.references[:24])}.")
    parts.append(
        "Source text withheld under the licensing policy; this entry records "
        "interface and behaviour only."
    )
    return "\n".join(parts)


def _verbatim_text(chunk: CodeChunk) -> str:
    header = (
        f"Symbol `{chunk.qualname}` ({chunk.kind}) in {chunk.path}, "
        f"lines {chunk.start_line}-{chunk.end_line}, {chunk.language}."
    )
    if chunk.docstring:
        header += f"\nDocumented behaviour: {chunk.docstring}"
    return f"{header}\n\n```{chunk.language}\n{chunk.code}\n```"


def to_content_list(
    chunks: Iterable[CodeChunk],
    provenance: Provenance,
) -> List[Dict[str, Any]]:
    """Render chunks as ``insert_content_list`` items, applying the gate.

    The policy on ``provenance`` decides whether each chunk contributes its
    source verbatim or only its interface and behaviour. This is the single
    place that decision is applied to code, so it cannot be bypassed by a
    caller assembling content some other way.
    """
    policy = provenance.use_policy
    items: List[Dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        text = (
            _verbatim_text(chunk)
            if policy is UsePolicy.VERBATIM
            else _facts_only_text(chunk)
        )
        items.append({"type": "text", "text": text, "page_idx": index})
    return items
