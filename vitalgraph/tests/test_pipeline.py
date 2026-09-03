"""End-to-end ingestion, including gate enforcement against a real tree."""

import textwrap

import pytest

from vitalgraph.acquire.base import LicenseClass, UsePolicy
from vitalgraph.ingest.pipeline import IngestPipeline, detect_repo_license

MIT = "Permission is hereby granted, free of charge, to any person obtaining a copy"
GPL = "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"

SOURCE = textwrap.dedent('''
    """Vendor BLE parser."""

    HR_CHAR = "0x2A37"


    def parse(data):
        """Parse a heart-rate frame."""
        flags = data[0]
        return flags & 0x10
''')


def _repo(tmp_path, license_text=MIT, name="repo"):
    root = tmp_path / name
    (root / "src").mkdir(parents=True)
    if license_text is not None:
        (root / "LICENSE").write_text(license_text)
    (root / "src" / "parser.py").write_text(SOURCE)
    (root / "README.md").write_text("# readme")
    (root / "src" / "__pycache__").mkdir()
    (root / "src" / "__pycache__" / "junk.py").write_text("cached = 1")
    return root


def test_detects_repository_license(tmp_path):
    assert detect_repo_license(_repo(tmp_path)).license_class is LicenseClass.PERMISSIVE


def test_missing_license_file_is_unknown_not_permissive(tmp_path):
    finding = detect_repo_license(_repo(tmp_path, license_text=None))
    assert finding.license_class is LicenseClass.UNKNOWN


def test_ingests_source_and_reports_a_summary(tmp_path):
    result = IngestPipeline().ingest_directory(
        _repo(tmp_path), repo="org/repo", ref="abc"
    )
    summary = result.summary()
    assert summary["files_ingested"] == 1
    assert summary["policy"] == "verbatim"
    assert summary["chunks"] >= 1
    assert summary["languages"] == ["python"]
    assert summary["exactly_parsed_chunks"] == summary["chunks"]


def test_skips_generated_and_unsupported_files(tmp_path):
    result = IngestPipeline().ingest_directory(_repo(tmp_path))
    # README.md and the extension-less LICENSE are both unsupported;
    # __pycache__ is skipped entirely, so it is not even counted as seen.
    assert result.skipped_reasons.get("unsupported_language") == 2
    assert not any("__pycache__" in c.path for c in result.chunks)


def test_symbol_graph_and_registry_are_populated(tmp_path):
    p = IngestPipeline()
    p.ingest_directory(_repo(tmp_path), repo="org/repo", ref="abc")
    assert p.symbols.definitions_of("parse")
    assert "0x2A37" in p.registry.uuids()


def test_gpl_repository_contributes_facts_but_not_source(tmp_path):
    """The end-to-end guarantee, exercised against a real directory."""
    p = IngestPipeline()
    result = p.ingest_directory(_repo(tmp_path, license_text=GPL), repo="gpl/repo")
    assert result.use_policy is UsePolicy.FACTS_ONLY

    text = "\n".join(
        i.get("text", "") + i.get("table_body", "")
        for i in p.build_content_list(result)
    )
    assert "flags & 0x10" not in text  # implementation withheld
    assert "def parse(data)" in text  # interface retained
    assert "Parse a heart-rate frame." in text  # documented behaviour retained


def test_mit_repository_contributes_source(tmp_path):
    p = IngestPipeline()
    result = p.ingest_directory(_repo(tmp_path), repo="mit/repo")
    text = "\n".join(i.get("text", "") for i in p.build_content_list(result))
    assert "flags & 0x10" in text


def test_unlicensed_repository_is_gated_like_proprietary(tmp_path):
    p = IngestPipeline()
    result = p.ingest_directory(_repo(tmp_path, license_text=None), repo="unknown/repo")
    assert result.use_policy is UsePolicy.FACTS_ONLY
    text = "\n".join(i.get("text", "") for i in p.build_content_list(result))
    assert "flags & 0x10" not in text


def test_license_override_is_honoured(tmp_path):
    result = IngestPipeline().ingest_directory(
        _repo(tmp_path), license_override=LicenseClass.PROPRIETARY
    )
    assert result.use_policy is UsePolicy.FACTS_ONLY


def test_max_files_bounds_a_run(tmp_path):
    root = _repo(tmp_path)
    for i in range(5):
        (root / "src" / f"extra{i}.py").write_text("def f():\n    return 1\n")
    result = IngestPipeline().ingest_directory(root, max_files=2)
    assert result.files_ingested == 2


def test_ingesting_a_non_directory_raises(tmp_path):
    with pytest.raises(NotADirectoryError):
        IngestPipeline().ingest_directory(tmp_path / "nope")


def test_provenance_carries_the_detected_license(tmp_path):
    p = IngestPipeline()
    result = p.ingest_directory(
        _repo(tmp_path, license_text=GPL), repo="gpl/repo", ref="deadbeef"
    )
    prov = p.provenance_for(result)
    assert prov.license_class is LicenseClass.COPYLEFT
    assert prov.upstream_ref == "deadbeef"
    assert not prov.allows_verbatim


@pytest.mark.asyncio
async def test_push_to_rag_uses_a_code_citation(tmp_path):
    """Inserted code must cite its repository, not an opaque chunk id."""
    captured = {}

    class FakeRag:
        async def insert_content_list(self, content_list, file_path, doc_id, **kw):
            captured.update(
                content_list=content_list, file_path=file_path, doc_id=doc_id
            )

    p = IngestPipeline()
    result = p.ingest_directory(_repo(tmp_path), repo="org/repo", ref="abc123")
    out = await p.push_to_rag(FakeRag(), result)

    assert captured["file_path"] == "code://org/repo@abc123"
    assert captured["doc_id"].startswith("vg-code-")
    assert out["policy"] == "verbatim"
    assert len(captured["content_list"]) == out["items"]


@pytest.mark.asyncio
async def test_push_to_rag_never_sends_restricted_source(tmp_path):
    """Asserted at the insert boundary -- the last place it could leak."""
    captured = {}

    class FakeRag:
        async def insert_content_list(self, content_list, file_path, doc_id, **kw):
            captured["content_list"] = content_list

    p = IngestPipeline()
    result = p.ingest_directory(_repo(tmp_path, license_text=GPL), repo="gpl/repo")
    await p.push_to_rag(FakeRag(), result)

    sent = "\n".join(
        i.get("text", "") + i.get("table_body", "") for i in captured["content_list"]
    )
    assert "flags & 0x10" not in sent
