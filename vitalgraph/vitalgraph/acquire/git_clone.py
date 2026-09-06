"""Acquisition by shallow git clone.

This is the harvester that actually works from here, and the reason is worth
recording. GitHub's code-search API is unavailable to this environment --
requests to ``/search/code`` are refused because the session is bound to its
configured repositories -- so a search-driven harvest cannot run. Cloning a
public repository over HTTPS is unrestricted, so acquisition is built on that
instead.

The trade is real and worth stating: clone-based acquisition cannot *discover*
repositories, only fetch named ones. So the target list is curated (see
``targets.py``) rather than produced by a query. A curated list is smaller than
a search, but it is also reproducible -- the same names on the same commits
yield the same corpus, which a ranked search never does.

Every clone is pinned to the commit SHA it resolved to, and that SHA becomes
the ``ref`` in every citation the ingest pipeline emits, so a chunk cited as
``bleak@1a2b3c4:bleak/backends/device.py#L10-L40`` names lines that can be
fetched back and checked.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

#: Only these schemes may be cloned. A URL is passed to ``git`` as an argument,
#: and git supports transports (``ext::``, ``file://``) that execute commands or
#: read the local disk. Restricting to HTTPS keeps a malformed or hostile entry
#: in a target list from becoming code execution.
ALLOWED_SCHEMES = ("https://",)

#: Seconds before a clone is abandoned. A wedged transfer must not hang a
#: harvest run indefinitely.
CLONE_TIMEOUT_S = 300

#: Network retry schedule, in seconds. Clones fail transiently often enough
#: that a single attempt makes a multi-repository run flaky.
RETRY_BACKOFF_S = (2.0, 4.0, 8.0, 16.0)


class CloneError(RuntimeError):
    """A repository could not be fetched."""


@dataclass(frozen=True, slots=True)
class CloneResult:
    """A repository on disk, pinned to the revision that was fetched."""

    name: str
    url: str
    path: Path
    commit: str
    """Full 40-character SHA. Never a branch name: a branch moves, and a
    citation that points at a moving target is not a citation."""

    reused: bool = False
    """True when an existing checkout was used instead of a fresh fetch."""

    @property
    def short_commit(self) -> str:
        return self.commit[:7]

    def citation_ref(self) -> str:
        return f"{self.name}@{self.short_commit}"


def validate_url(url: str) -> str:
    """Return ``url`` if it is safe to hand to ``git clone``, else raise.

    Rejects anything not HTTPS, and anything beginning with a dash -- git would
    read such a value as an option rather than a repository.
    """
    if url.startswith("-"):
        raise CloneError(f"refusing option-like URL: {url!r}")
    if not url.startswith(ALLOWED_SCHEMES):
        raise CloneError(
            f"refusing non-HTTPS URL: {url!r} "
            f"(allowed schemes: {', '.join(ALLOWED_SCHEMES)})"
        )
    return url


def name_for_url(url: str) -> str:
    """Derive a stable repository name from its URL.

    ``https://github.com/hbldh/bleak.git`` becomes ``bleak``. The name lands in
    every citation, so it stays short and human-recognisable rather than
    carrying the whole path.
    """
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    if tail.endswith(".git"):
        tail = tail[: -len(".git")]
    return tail or "repository"


def _run_git(
    args: Sequence[str], cwd: Path | None = None, timeout: int = CLONE_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - argument vector, never a shell string
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def resolve_head(path: Path) -> str:
    """Return the full commit SHA a checkout is sitting on."""
    proc = _run_git(["rev-parse", "HEAD"], cwd=path, timeout=30)
    if proc.returncode != 0:
        raise CloneError(f"cannot resolve HEAD in {path}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def clone(
    url: str,
    dest_root: str | Path,
    name: str | None = None,
    depth: int = 1,
    refresh: bool = False,
    sleep: Callable[[float], None] = time.sleep,
) -> CloneResult:
    """Shallow-clone ``url`` beneath ``dest_root`` and pin its commit.

    An existing checkout is reused unless ``refresh`` is set, so re-running a
    harvest costs nothing and produces identical citations. Network failures
    are retried on the standard backoff; a failure that is not transient --
    a bad URL, a repository that does not exist -- still raises after the
    retries are spent, because there is no honest way to distinguish the two
    from the outside and silently returning nothing would hide a broken target.
    """
    url = validate_url(url)
    name = name or name_for_url(url)
    dest_root = Path(dest_root)
    path = dest_root / name

    if path.exists() and not refresh:
        return CloneResult(
            name=name, url=url, path=path, commit=resolve_head(path), reused=True
        )
    if path.exists():
        shutil.rmtree(path)
    dest_root.mkdir(parents=True, exist_ok=True)

    args = ["clone", "--quiet", "--single-branch"]
    if depth > 0:
        args += ["--depth", str(depth)]
    args += ["--", url, str(path)]

    last_error = ""
    for attempt in range(len(RETRY_BACKOFF_S) + 1):
        try:
            proc = _run_git(args)
        except subprocess.TimeoutExpired:
            last_error = f"timed out after {CLONE_TIMEOUT_S}s"
        else:
            if proc.returncode == 0:
                return CloneResult(
                    name=name, url=url, path=path, commit=resolve_head(path)
                )
            last_error = proc.stderr.strip() or f"git exited {proc.returncode}"

        if attempt < len(RETRY_BACKOFF_S):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
            sleep(RETRY_BACKOFF_S[attempt])

    raise CloneError(f"failed to clone {url}: {last_error}")
