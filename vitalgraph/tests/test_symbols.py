"""Cross-repository symbol graph."""

from vitalgraph.ingest.code_chunker import chunk_generic, chunk_python
from vitalgraph.ingest.symbols import SymbolGraph

PY_A = '''
def decode_hr(data):
    """Decode heart rate."""
    return unpack(data)
'''

PY_B = '''
def decode_hr(payload):
    """Vendor B's decoder."""
    return parse(payload)
'''

KT = """
class HrParser {
    fun decode_hr(data: ByteArray): Int {
        return 0
    }
}
"""


def _graph():
    g = SymbolGraph()
    g.add_chunks(chunk_python(PY_A, "a.py"), repo="vendor/a", ref="aaa")
    g.add_chunks(chunk_python(PY_B, "b.py"), repo="vendor/b", ref="bbb")
    return g


def test_symbols_are_indexed_with_citations():
    g = _graph()
    sites = g.definitions_of("decode_hr")
    assert len(sites) == 2
    # PY_A opens with a newline, so the def begins on line 2 and its body
    # ends on line 4 -- the citation must name exactly those lines.
    assert sites[0].citation() == "vendor/a@aaa:a.py#L2-L4"


def test_cross_repo_symbols_are_the_convergent_ones():
    """The question worth answering: who else implements this?"""
    g = _graph()
    node = g.get("decode_hr")
    assert node.is_cross_repo
    assert node.repos == ("vendor/a", "vendor/b")
    assert [n.name for n in g.cross_repo_symbols()] == ["decode_hr"]


def test_single_repo_symbols_are_not_cross_repo():
    g = SymbolGraph()
    g.add_chunks(chunk_python(PY_A, "a.py"), repo="vendor/a")
    assert g.cross_repo_symbols() == []


def test_confidence_reflects_extraction_quality():
    """A graph mixing exact and approximate parses must say so."""
    g = _graph()
    assert g.get("decode_hr").confidence == 1.0

    g.add_chunks(chunk_generic(KT, "P.kt", "kotlin"), repo="vendor/c")
    node = g.get("decode_hr")
    assert 0.0 < node.confidence < 1.0  # two exact, one brace-matched
    assert "kotlin" in node.languages


def test_references_are_tracked_separately_from_definitions():
    g = _graph()
    assert g.definitions_of("unpack") == []
    assert len(g.referrers_of("unpack")) == 1


def test_search_ranks_widely_implemented_symbols_first():
    g = _graph()
    g.add_chunks(
        chunk_python("def decode_only_here(x):\n    return x\n", "c.py"),
        repo="vendor/a",
    )
    results = g.search("decode")
    assert results[0].name == "decode_hr"  # 2 repos beats 1


def test_stats_report_parse_quality():
    g = _graph()
    stats = g.stats()
    assert stats["exact_fraction"] == 1.0
    assert stats["cross_repo_symbols"] == 1
    assert stats["repos"] == ["vendor/a", "vendor/b"]


def test_content_list_emits_only_convergent_symbols():
    """Emitting every symbol would swamp retrieval; the value is in overlap."""
    g = _graph()
    items = g.to_content_list()
    assert len(items) == 1
    assert items[0]["type"] == "table"
    assert "decode_hr" in items[0]["table_body"]
    assert "unpack" not in items[0]["table_body"]
    assert "confidence" in items[0]["table_footnote"][0].lower()


def test_content_list_is_empty_without_convergence():
    g = SymbolGraph()
    g.add_chunks(chunk_python(PY_A, "a.py"), repo="vendor/a")
    assert g.to_content_list() == []


def test_missing_symbol_queries_are_safe():
    g = _graph()
    assert g.get("nope") is None
    assert g.definitions_of("nope") == []
    assert g.referrers_of("nope") == []
