"""Tests for the health-signal code corpus. Entirely offline -- no cloning."""

import asyncio
import textwrap


from raganything.biosignal import corpus
from raganything.biosignal.corpus import (
    CORPUS,
    Source,
    extract_python_structure,
    fetch_source,
    ingest_corpus,
    sources_for,
    to_content_list,
)


class TestCorpusIntegrity:
    def test_corpus_is_not_empty(self):
        assert len(CORPUS) >= 10

    def test_names_are_unique(self):
        names = [s.name for s in CORPUS]
        assert len(names) == len(set(names))

    def test_every_source_declares_a_licence(self):
        # Licence is a product decision, not a footnote: a source without one
        # recorded cannot be safely built on.
        for source in CORPUS:
            assert source.license, f"{source.name} has no licence recorded"

    def test_every_source_justifies_itself(self):
        for source in CORPUS:
            assert len(source.relevance) > 40, f"{source.name} lacks a rationale"
            assert source.provides, f"{source.name} lists no capabilities"

    def test_urls_look_fetchable(self):
        for source in CORPUS:
            assert source.url.startswith("https://"), source.name

    def test_covers_the_capability_classes_that_matter(self):
        kinds = {s.kind for s in CORPUS}
        assert {"library", "firmware", "model"} <= kinds
        languages = {s.language for s in CORPUS}
        assert {"python", "c"} <= languages

    def test_signal_quality_and_morphology_are_both_present(self):
        blob = " ".join(s.relevance + " ".join(s.provides) for s in CORPUS).lower()
        assert "quality" in blob
        assert "morpholog" in blob

    def test_copyleft_sources_are_labelled_as_such(self):
        # A GPL dependency changes what you may ship; it must be visible.
        gpl = [s for s in CORPUS if "GPL" in s.license]
        for source in gpl:
            assert "licence" in source.relevance.lower() or "license" in (
                source.relevance.lower()
            ), f"{source.name} is {source.license} without flagging it"

    def test_filtering(self):
        assert all(s.kind == "firmware" for s in sources_for(kind="firmware"))
        assert all(s.language == "c" for s in sources_for(language="c"))
        assert sources_for(kind="nonexistent") == []

    def test_source_serialises(self):
        payload = CORPUS[0].to_dict()
        assert payload["name"] and payload["url"] and payload["license"]


SAMPLE = textwrap.dedent(
    '''
    """Compute signal quality indices.

    Longer description that should be captured.
    """

    import numpy as np


    def skewness_sqi(signal, axis=0):
        """Skewness-based signal quality index."""
        return 0.0


    async def stream_sqi(signal):
        """Async variant."""
        return 0.0


    def _private_helper(x):
        """Should not appear."""
        return x


    class SignalQuality(Base):
        """Holds the quality assessment for one segment."""

        def score(self, window, *args, threshold=0.5, **kwargs):
            """Return the score."""
            return 1.0

        def _internal(self):
            return None
    '''
)


class TestPythonExtraction:
    def _write(self, tmp_path, text=SAMPLE, name="sqi.py"):
        path = tmp_path / name
        path.write_text(text)
        return path

    def test_module_docstring_is_captured(self, tmp_path):
        structure = extract_python_structure(self._write(tmp_path))
        assert "Compute signal quality indices" in structure["module_doc"]

    def test_public_functions_and_classes(self, tmp_path):
        structure = extract_python_structure(self._write(tmp_path))
        signatures = [e["signature"] for e in structure["api"]]
        assert "def skewness_sqi(signal, axis)" in signatures
        assert "async def stream_sqi(signal)" in signatures
        assert "class SignalQuality(Base)" in signatures

    def test_private_names_are_skipped(self, tmp_path):
        structure = extract_python_structure(self._write(tmp_path))
        blob = str(structure["api"])
        assert "_private_helper" not in blob
        assert "_internal" not in blob

    def test_class_methods_are_listed(self, tmp_path):
        structure = extract_python_structure(self._write(tmp_path))
        cls = next(e for e in structure["api"] if e["signature"].startswith("class"))
        assert "score" in cls["methods"]

    def test_varargs_and_kwargs_render(self, tmp_path):
        structure = extract_python_structure(self._write(tmp_path))
        cls = next(e for e in structure["api"] if e["signature"].startswith("class"))
        assert "*args" in cls["methods"] and "**kwargs" in cls["methods"]

    def test_unparseable_file_returns_none_rather_than_raising(self, tmp_path):
        # Someone else's syntax error is not a reason to abort ingestion.
        path = tmp_path / "broken.py"
        path.write_text("def (((:\n")
        assert extract_python_structure(path) is None

    def test_empty_module_returns_none(self, tmp_path):
        path = tmp_path / "empty.py"
        path.write_text("import os\n")
        assert extract_python_structure(path) is None


def fake_source(**kw):
    defaults = dict(
        name="fake",
        url="https://github.com/example/fake",
        kind="library",
        license="MIT",
        relevance="A stand-in project used to exercise the corpus machinery end to end.",
        provides=("thing one", "thing two"),
    )
    defaults.update(kw)
    return Source(**defaults)


class TestContentList:
    def test_overview_and_capabilities_without_a_checkout(self):
        # The corpus stays queryable even when a fetch failed.
        items = to_content_list(fake_source(), None)
        assert items[0]["type"] == "text"
        assert "A stand-in project" in items[0]["text"]
        table = items[1]
        assert table["type"] == "table"
        assert "thing one" in table["table_body"]

    def test_licence_travels_in_the_table_footnote(self):
        items = to_content_list(fake_source(license="GPL-3.0"), None)
        assert "GPL-3.0" in " ".join(items[1]["table_footnote"])

    def test_files_become_content_items(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "sqi.py").write_text(SAMPLE)
        items = to_content_list(fake_source(), tmp_path)
        texts = [i.get("text", "") for i in items]
        assert any("skewness_sqi" in t for t in texts)
        assert any("Public API:" in t for t in texts)

    def test_excluded_directories_are_skipped(self, tmp_path):
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_sqi.py").write_text(SAMPLE)
        (tmp_path / "keep.py").write_text(SAMPLE)
        items = to_content_list(fake_source(), tmp_path)
        texts = " ".join(i.get("text", "") for i in items)
        assert "keep.py" in texts
        assert "test_sqi.py" not in texts

    def test_include_paths_narrow_the_walk(self, tmp_path):
        (tmp_path / "wanted").mkdir()
        (tmp_path / "wanted" / "a.py").write_text(SAMPLE)
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "b.py").write_text(SAMPLE)
        items = to_content_list(fake_source(include=("wanted",)), tmp_path)
        texts = " ".join(i.get("text", "") for i in items)
        assert "a.py" in texts
        assert "b.py" not in texts

    def test_c_sources_use_the_header_extractor(self, tmp_path):
        (tmp_path / "hrs.c").write_text(
            "/* Heart Rate Service implementation */\n"
            "static ssize_t read_hrs(struct bt_conn *conn)\n"
            "{\n  return 0;\n}\n"
        )
        items = to_content_list(fake_source(language="c"), tmp_path)
        texts = " ".join(i.get("text", "") for i in items)
        assert "hrs.c" in texts

    def test_max_files_is_reported_not_silently_truncated(self, tmp_path):
        for i in range(5):
            (tmp_path / f"m{i}.py").write_text(SAMPLE)
        items = to_content_list(fake_source(), tmp_path, max_files=2)
        assert any("were indexed" in i.get("text", "") for i in items)

    def test_page_indices_are_sequential(self, tmp_path):
        (tmp_path / "a.py").write_text(SAMPLE)
        items = to_content_list(fake_source(), tmp_path)
        assert [i["page_idx"] for i in items] == list(range(len(items)))

    def test_empty_checkout_still_yields_the_overview(self, tmp_path):
        items = to_content_list(fake_source(), tmp_path)
        assert len(items) == 2  # overview + capabilities, no files found


class TestFetching:
    def test_non_clonable_source_returns_none(self, tmp_path):
        source = fake_source(url="https://example.com/docs")
        assert fetch_source(source, tmp_path) is None

    def test_existing_checkout_is_reused(self, tmp_path):
        source = fake_source()
        (tmp_path / source.name).mkdir(parents=True)
        assert fetch_source(source, tmp_path) == tmp_path / source.name

    def test_clone_failure_is_logged_not_raised(self, tmp_path, monkeypatch):
        import subprocess

        def boom(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "git", stderr=b"not found")

        monkeypatch.setattr(corpus.subprocess, "run", boom)
        # One unreachable repository must not abort the other ten.
        assert fetch_source(fake_source(), tmp_path) is None

    def test_clone_timeout_is_handled(self, tmp_path, monkeypatch):
        import subprocess

        def slow(*args, **kwargs):
            raise subprocess.TimeoutExpired("git", 1)

        monkeypatch.setattr(corpus.subprocess, "run", slow)
        assert fetch_source(fake_source(), tmp_path) is None


class _FakeRAG:
    def __init__(self):
        self.calls = []

    async def insert_content_list(self, content_list, file_path, doc_id, **kwargs):
        self.calls.append(
            {"n": len(content_list), "file_path": file_path, "doc_id": doc_id}
        )


class TestIngestion:
    def test_indexes_each_source_under_a_stable_doc_id(self, tmp_path):
        rag = _FakeRAG()
        (tmp_path / "fake").mkdir()
        (tmp_path / "fake" / "a.py").write_text(SAMPLE)

        results = asyncio.run(
            ingest_corpus(rag, [fake_source()], workdir=str(tmp_path), fetch=False)
        )
        assert results["fake"] > 0
        assert rag.calls[0]["doc_id"] == "corpus:fake"
        assert rag.calls[0]["file_path"] == "fake.corpus"

    def test_missing_checkout_still_indexes_the_rationale(self, tmp_path):
        rag = _FakeRAG()
        results = asyncio.run(
            ingest_corpus(rag, [fake_source()], workdir=str(tmp_path), fetch=False)
        )
        # Two items: what it is, and why we wanted it.
        assert results["fake"] == 2

    def test_reports_per_source_counts(self, tmp_path):
        rag = _FakeRAG()
        sources = [fake_source(name="a"), fake_source(name="b")]
        results = asyncio.run(
            ingest_corpus(rag, sources, workdir=str(tmp_path), fetch=False)
        )
        assert set(results) == {"a", "b"}
        assert len(rag.calls) == 2
