"""Symbol-aware chunking, and the license gate applied to code content."""

import pytest

from vitalgraph.acquire.base import LicenseClass, local_provenance
from vitalgraph.ingest.code_chunker import (
    EXTRACTION_AST,
    EXTRACTION_BRACE,
    chunk_generic,
    chunk_python,
    chunk_source,
    language_for,
    to_content_list,
)

PY = '''"""Module docstring."""

import struct


def decode(data: bytes) -> int:
    """Decode a payload."""
    return struct.unpack_from("<H", data, 0)[0]


class Decoder:
    """A decoder."""

    def run(self, raw):
        return decode(raw)
'''

KOTLIN = """
package com.vendor.ble

class HeartRateParser {
    fun parse(data: ByteArray): Int {
        val flags = data[0].toInt()
        return flags and 0x01
    }
}
"""


def test_python_chunks_by_symbol():
    chunks = chunk_python(PY, "m.py")
    names = {c.qualname for c in chunks}
    assert names == {"decode", "Decoder", "Decoder.run"}
    assert all(c.extraction == EXTRACTION_AST for c in chunks)


def test_line_spans_are_exact():
    chunk = next(c for c in chunk_python(PY, "m.py") if c.qualname == "decode")
    lines = PY.splitlines()
    assert lines[chunk.start_line - 1].startswith("def decode")
    assert "unpack_from" in "\n".join(lines[chunk.start_line - 1 : chunk.end_line])


def test_signature_and_docstring_are_captured():
    chunk = next(c for c in chunk_python(PY, "m.py") if c.qualname == "decode")
    assert chunk.signature == "def decode(data: bytes) -> int"
    assert chunk.docstring == "Decode a payload."


def test_references_are_extracted_and_deterministic():
    chunk = next(c for c in chunk_python(PY, "m.py") if c.qualname == "decode")
    assert "unpack_from" in chunk.references
    # Sorted and deduplicated, so doc_ids derived from chunks stay stable.
    assert list(chunk.references) == sorted(set(chunk.references))


def test_methods_are_qualified_by_class():
    chunks = chunk_python(PY, "m.py")
    method = next(c for c in chunks if c.kind == "method")
    assert method.qualname == "Decoder.run"


def test_module_without_symbols_still_produces_a_chunk():
    chunks = chunk_python("import os\nX = 1\n", "consts.py")
    assert len(chunks) == 1
    assert chunks[0].kind == "module"


def test_citation_points_at_real_lines():
    chunk = next(c for c in chunk_python(PY, "ble/gatt.py") if c.qualname == "decode")
    citation = chunk.citation(repo="org/repo", ref="abc1234")
    assert (
        citation
        == f"org/repo@abc1234:ble/gatt.py#L{chunk.start_line}-L{chunk.end_line}"
    )


def test_brace_languages_are_chunked_and_marked_inexact():
    chunks = chunk_generic(KOTLIN, "Parser.kt", "kotlin")
    names = {c.qualname for c in chunks}
    assert "HeartRateParser" in names and "parse" in names
    # Honesty about extraction quality is the point: these are not exact.
    assert all(c.extraction == EXTRACTION_BRACE for c in chunks)
    assert all(not c.is_exact for c in chunks)


def test_unparseable_python_degrades_instead_of_dropping():
    broken = "def f(:\n  this is not python\n"
    chunks = chunk_source(broken, "broken.py")
    assert chunks  # content is never silently lost
    assert all(not c.is_exact for c in chunks)


def test_unsupported_extension_yields_nothing():
    assert chunk_source("hello", "notes.txt") == []
    assert language_for("notes.txt") is None
    assert language_for("a.py") == "python"


# --- the gate, applied to code content -------------------------------------


def test_permissive_source_enters_verbatim():
    chunks = chunk_python(PY, "m.py")
    prov = local_provenance("/m.py", PY, LicenseClass.PERMISSIVE, "MIT")
    text = "\n".join(i["text"] for i in to_content_list(chunks, prov))
    assert "struct.unpack_from" in text


@pytest.mark.parametrize(
    "cls", [LicenseClass.COPYLEFT, LicenseClass.PROPRIETARY, LicenseClass.UNKNOWN]
)
def test_non_permissive_source_never_enters_verbatim(cls):
    """The central guarantee: restricted source bodies stay out of the corpus."""
    chunks = chunk_python(PY, "m.py")
    prov = local_provenance("/m.py", PY, cls)
    text = "\n".join(i["text"] for i in to_content_list(chunks, prov))
    assert "struct.unpack_from" not in text
    assert "return decode(raw)" not in text


def test_facts_survive_the_gate():
    """Interface and behaviour are interoperability facts and must remain."""
    chunks = chunk_python(PY, "m.py")
    prov = local_provenance("/m.py", PY, LicenseClass.COPYLEFT, "GPL-3.0-only")
    text = "\n".join(i["text"] for i in to_content_list(chunks, prov))
    assert "def decode(data: bytes) -> int" in text
    assert "Decode a payload." in text
    assert "withheld" in text


def test_content_list_matches_insert_content_list_contract():
    chunks = chunk_python(PY, "m.py")
    items = to_content_list(chunks, local_provenance("/m.py", PY))
    assert items
    for item in items:
        assert set(item) == {"type", "text", "page_idx"}
        assert item["type"] == "text"
        assert item["text"].strip()
