"""Tests for clone-based acquisition and the harvest runner.

No network. Cloning is exercised against repositories created locally with
``git init``, which is a real clone through real git -- only the transport is
local. That keeps the tests honest about the code path while leaving them
runnable anywhere.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vitalgraph.acquire import git_clone, harvest, targets  # noqa: E402
from vitalgraph.acquire.base import UsePolicy  # noqa: E402
from vitalgraph.acquire.git_clone import CloneError  # noqa: E402
from vitalgraph.acquire.targets import Category, Target  # noqa: E402

MIT_TEXT = """MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.
"""

GPL_TEXT = """                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Everyone is permitted to copy and distribute verbatim copies of this
 license document, but changing it is not allowed.
"""

SAMPLE_SOURCE = '''"""A decoder, so the chunker has something to find."""


def decode_heart_rate(payload: bytes) -> int:
    """Decode characteristic 0x2A37."""
    flags = payload[0]
    if flags & 0x01:
        return int.from_bytes(payload[1:3], "little")
    return payload[1]
'''


def _make_repo(root: Path, license_text: str) -> Path:
    """Create a real local git repository with one commit."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "LICENSE").write_text(license_text, encoding="utf-8")
    (root / "decoder.py").write_text(SAMPLE_SOURCE, encoding="utf-8")
    env_args = [
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "user.name=Test",
        "-c",
        "commit.gpgsign=false",
    ]
    subprocess.run(["git", "init", "--quiet", "-b", "main", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), *env_args, "commit", "--quiet", "-m", "initial"],
        check=True,
    )
    return root


# --- URL safety ------------------------------------------------------------


def test_validate_url_accepts_https():
    url = "https://github.com/example/repo.git"
    assert git_clone.validate_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "ext::sh -c 'touch /tmp/pwned'",
        "file:///etc",
        "git://github.com/example/repo.git",
        "ssh://git@github.com/example/repo.git",
        "http://github.com/example/repo.git",
    ],
)
def test_validate_url_rejects_non_https_transports(url):
    """git supports transports that execute commands or read local disk.

    Only HTTPS may reach ``git clone``, so a bad target-list entry cannot
    become code execution or local file disclosure.
    """
    with pytest.raises(CloneError):
        git_clone.validate_url(url)


def test_validate_url_rejects_option_like_value():
    with pytest.raises(CloneError):
        git_clone.validate_url("--upload-pack=touch /tmp/pwned")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/hbldh/bleak.git", "bleak"),
        ("https://github.com/hbldh/bleak", "bleak"),
        ("https://github.com/hbldh/bleak/", "bleak"),
        ("https://example.org/a/b/c.git", "c"),
    ],
)
def test_name_for_url(url, expected):
    assert git_clone.name_for_url(url) == expected


# --- cloning ---------------------------------------------------------------


def test_clone_pins_full_commit_sha(tmp_path):
    """A clone records the 40-character SHA, never a branch name.

    ``file://`` is refused by :func:`validate_url` by design, so the clone runs
    through the internal runner. That exercises pinning without weakening the
    URL policy for the sake of a test.
    """
    origin = _make_repo(tmp_path / "origin", MIT_TEXT)
    expected = git_clone.resolve_head(origin)
    assert len(expected) == 40

    checkout = tmp_path / "cache" / "origin"
    proc = git_clone._run_git(["clone", "--quiet", "--", str(origin), str(checkout)])
    assert proc.returncode == 0, proc.stderr
    assert git_clone.resolve_head(checkout) == expected


def test_clone_result_citation_ref_uses_short_commit():
    result = git_clone.CloneResult(
        name="bleak",
        url="https://github.com/hbldh/bleak.git",
        path=Path("/tmp/bleak"),
        commit="a" * 40,
    )
    assert result.short_commit == "aaaaaaa"
    assert result.citation_ref() == "bleak@aaaaaaa"


def test_clone_retries_then_raises(tmp_path, monkeypatch):
    """A persistently failing clone raises rather than returning nothing.

    Returning an empty result would make a broken target indistinguishable
    from an empty repository, which is how a silently missing source ends up
    in a corpus nobody audits.
    """
    attempts = {"n": 0}

    def fake_run(args, cwd=None, timeout=git_clone.CLONE_TIMEOUT_S):
        attempts["n"] += 1
        return subprocess.CompletedProcess(args, 128, "", "fatal: repository not found")

    monkeypatch.setattr(git_clone, "_run_git", fake_run)
    slept: list[float] = []

    with pytest.raises(CloneError, match="repository not found"):
        git_clone.clone(
            "https://example.invalid/nope.git",
            tmp_path,
            sleep=slept.append,
        )

    assert attempts["n"] == len(git_clone.RETRY_BACKOFF_S) + 1
    assert slept == list(git_clone.RETRY_BACKOFF_S)


def test_clone_reuses_existing_checkout(tmp_path, monkeypatch):
    origin = _make_repo(tmp_path / "origin", MIT_TEXT)
    cache = tmp_path / "cache"
    cache.mkdir()
    subprocess.run(
        ["git", "clone", "--quiet", str(origin), str(cache / "origin")], check=True
    )

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("clone attempted despite a cached checkout")

    monkeypatch.setattr(git_clone, "_run_git", lambda a, **k: explode())
    monkeypatch.setattr(git_clone, "resolve_head", lambda p: "b" * 40)

    result = git_clone.clone("https://example.org/origin.git", cache, name="origin")
    assert result.reused is True
    assert result.commit == "b" * 40


# --- target catalogue ------------------------------------------------------


def test_every_target_url_is_https_and_clonable_in_principle():
    for target in targets.TARGETS:
        assert git_clone.validate_url(target.url) == target.url


def test_target_names_are_unique():
    names = [t.name for t in targets.TARGETS]
    assert len(names) == len(set(names))


def test_every_target_states_a_rationale():
    """An entry nobody can justify is an entry that should be removed."""
    for target in targets.TARGETS:
        assert target.rationale.strip(), f"{target.name} has no rationale"
        assert len(target.rationale) > 40, f"{target.name} rationale is too thin"


def test_default_selection_excludes_heavy_targets():
    default = targets.default_selection()
    assert all(not t.heavy for t in default)
    assert any(t.heavy for t in targets.default_selection(include_heavy=True))


def test_catalogue_covers_every_category():
    covered = set(targets.catalogue_summary())
    assert covered == {c.value for c in Category}


def test_by_name_raises_for_unknown():
    with pytest.raises(KeyError):
        targets.by_name("no-such-repository")


# --- the harvest runner ----------------------------------------------------


def _local_target(name: str, path: Path, expected_spdx: str | None) -> Target:
    return Target(
        name=name,
        url=f"https://example.invalid/{name}.git",
        category=Category.BLE_STACK,
        rationale="A fixture repository used to exercise the harvest runner end to end.",
        expected_spdx=expected_spdx,
    )


def _patch_clone_to_local(monkeypatch, mapping: dict[str, Path]):
    def fake_clone(url, dest_root, name=None, refresh=False, **kwargs):
        name = name or git_clone.name_for_url(url)
        if name not in mapping:
            raise CloneError(f"failed to clone {url}: repository not found")
        path = mapping[name]
        return git_clone.CloneResult(
            name=name,
            url=url,
            path=path,
            commit=git_clone.resolve_head(path),
        )

    monkeypatch.setattr(harvest, "clone", fake_clone)


def test_harvest_applies_the_license_gate_per_repository(tmp_path, monkeypatch):
    """Permissive source may be quoted; copyleft source may only be described."""
    permissive = _make_repo(tmp_path / "permissive", MIT_TEXT)
    copyleft = _make_repo(tmp_path / "copyleft", GPL_TEXT)
    _patch_clone_to_local(monkeypatch, {"permissive": permissive, "copyleft": copyleft})

    report, pipeline = harvest.harvest(
        [
            _local_target("permissive", permissive, "MIT"),
            _local_target("copyleft", copyleft, "GPL-3.0-only"),
        ],
        cache_dir=tmp_path / "cache",
    )

    assert len(report.entries) == 2
    assert report.failures == []
    policies = {e.target.name: e.ingest.use_policy for e in report.entries}
    assert policies["permissive"] is UsePolicy.VERBATIM
    assert policies["copyleft"] is UsePolicy.FACTS_ONLY
    assert report.verbatim_repos() == ["permissive"]
    assert report.facts_only_repos() == ["copyleft"]
    # One shared symbol graph across repositories is what makes cross-repo
    # questions answerable at all.
    assert pipeline.symbols is not None


def test_harvest_flags_a_license_that_contradicts_the_target_list(
    tmp_path, monkeypatch
):
    """Detection governs; a wrong expectation is reported, never obeyed."""
    repo = _make_repo(tmp_path / "surprise", GPL_TEXT)
    _patch_clone_to_local(monkeypatch, {"surprise": repo})

    report, _ = harvest.harvest(
        [_local_target("surprise", repo, expected_spdx="MIT")],
        cache_dir=tmp_path / "cache",
    )

    entry = report.entries[0]
    assert entry.license_matches_expectation is False
    assert report.license_mismatches() == [entry]
    # The mismatch must not have widened what is permitted.
    assert entry.ingest.use_policy is UsePolicy.FACTS_ONLY
    assert "surprise" in format_mismatches(report)


def format_mismatches(report: harvest.HarvestReport) -> str:
    return harvest.format_report(report)


def test_harvest_records_no_expectation_as_none_not_as_a_pass(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path / "unchecked", MIT_TEXT)
    _patch_clone_to_local(monkeypatch, {"unchecked": repo})

    report, _ = harvest.harvest(
        [_local_target("unchecked", repo, expected_spdx=None)],
        cache_dir=tmp_path / "cache",
    )
    assert report.entries[0].license_matches_expectation is None
    assert report.license_mismatches() == []


def test_harvest_continues_past_a_failed_clone(tmp_path, monkeypatch):
    """A dead upstream must not cost the rest of the run."""
    good = _make_repo(tmp_path / "good", MIT_TEXT)
    _patch_clone_to_local(monkeypatch, {"good": good})

    report, _ = harvest.harvest(
        [
            _local_target("missing", tmp_path / "missing", "MIT"),
            _local_target("good", good, "MIT"),
        ],
        cache_dir=tmp_path / "cache",
    )

    assert [e.target.name for e in report.entries] == ["good"]
    assert [f.name for f in report.failures] == ["missing"]
    assert "repository not found" in report.failures[0].reason
    assert report.summary()["repositories_failed"] == 1


def test_harvest_report_json_is_written_and_complete(tmp_path, monkeypatch):
    repo = _make_repo(tmp_path / "one", MIT_TEXT)
    _patch_clone_to_local(monkeypatch, {"one": repo})

    report, _ = harvest.harvest(
        [_local_target("one", repo, "MIT")], cache_dir=tmp_path / "cache"
    )
    path = report.write_json(tmp_path / "report.json")
    assert path.is_file()

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["repositories_harvested"] == 1
    entry = data["repositories"][0]
    assert entry["name"] == "one"
    assert len(entry["commit"]) == 40
    assert entry["citation_ref"].startswith("one@")
    assert entry["policy"] == UsePolicy.VERBATIM.value
    assert entry["chunks"] >= 1


def test_harvest_citations_carry_the_pinned_commit(tmp_path, monkeypatch):
    """A citation must name a commit, not a branch -- a branch moves."""
    repo = _make_repo(tmp_path / "pinned", MIT_TEXT)
    _patch_clone_to_local(monkeypatch, {"pinned": repo})

    report, _ = harvest.harvest(
        [_local_target("pinned", repo, "MIT")], cache_dir=tmp_path / "cache"
    )
    entry = report.entries[0]
    sha = git_clone.resolve_head(repo)
    assert entry.ingest.ref == sha[:7]

    chunk = entry.ingest.chunks[0]
    assert chunk.citation("pinned", entry.ingest.ref) == (
        f"pinned@{sha[:7]}:{chunk.path}#{chunk.line_span}"
    )

    # The citation has to survive into what is actually inserted. The host
    # framework's file_path is document-level, so a chunk that does not carry
    # its own Source line reaches the graph unattributed.
    items = pipeline_for(entry).build_content_list(entry.ingest)
    code_items = [i for i in items if "Symbol `" in i["text"]]
    assert code_items, "no code chunks were emitted"
    for item in code_items:
        assert f"pinned@{sha[:7]}:" in item["text"]


def pipeline_for(entry: harvest.HarvestEntry):
    from vitalgraph.ingest.pipeline import IngestPipeline

    return IngestPipeline()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
