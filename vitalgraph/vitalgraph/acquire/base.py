"""Provenance and licensing primitives for harvested material.

Every artifact VitalGraph ingests -- a source file, a specification, a paper --
carries a :class:`Provenance` record. This is not bookkeeping for its own sake:
the license class recorded here decides, in ``ingest/pipeline.py``, whether the
artifact's *text* may enter the corpus verbatim or whether only the *facts*
extracted from it may.

That distinction is the whole basis for studying competitors' implementations
safely. Protocol behaviour -- that characteristic 0x2A37 carries a flags byte
followed by a heart rate, that a vendor frames its payload a certain way -- is
interoperability information, and stays usable no matter where it was learned.
Verbatim source under a copyleft or proprietary license is what must never end
up in a corpus that generates product code.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List


class LicenseClass(str, Enum):
    """How an artifact's license constrains downstream use."""

    PERMISSIVE = "permissive"
    """MIT, BSD, Apache-2.0 and similar. Verbatim use with attribution."""

    COPYLEFT = "copyleft"
    """GPL, AGPL, LGPL, MPL. Facts yes; verbatim source no."""

    PROPRIETARY = "proprietary"
    """All rights reserved, or a license forbidding redistribution."""

    UNKNOWN = "unknown"
    """No license found. Treated as strictly as proprietary -- absence of a
    license is absence of permission, not permission."""


class UsePolicy(str, Enum):
    """What the pipeline is allowed to do with an artifact."""

    VERBATIM = "verbatim"
    """Full text may enter the knowledge graph, with attribution."""

    FACTS_ONLY = "facts_only"
    """Only extracted behavioural facts and summaries may enter."""


#: The gate. Deliberately a plain mapping so the policy is auditable at a
#: glance rather than buried in branching logic.
POLICY: Dict[LicenseClass, UsePolicy] = {
    LicenseClass.PERMISSIVE: UsePolicy.VERBATIM,
    LicenseClass.COPYLEFT: UsePolicy.FACTS_ONLY,
    LicenseClass.PROPRIETARY: UsePolicy.FACTS_ONLY,
    LicenseClass.UNKNOWN: UsePolicy.FACTS_ONLY,
}


def policy_for(license_class: LicenseClass) -> UsePolicy:
    """Return the use policy for a license class.

    Defaults to the restrictive policy for anything unrecognised, so a new or
    misspelled class can never silently widen what is permitted.
    """
    return POLICY.get(license_class, UsePolicy.FACTS_ONLY)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where an artifact came from and what may be done with it."""

    source_url: str
    retrieved_at: datetime
    content_sha256: str
    license_class: LicenseClass = LicenseClass.UNKNOWN
    spdx_id: str | None = None
    upstream_ref: str | None = None
    """Commit SHA, DOI, or specification version -- whatever pins the exact
    revision this content came from."""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.retrieved_at.tzinfo is None:
            raise ValueError("Provenance.retrieved_at must be timezone-aware (UTC)")

    @property
    def use_policy(self) -> UsePolicy:
        return policy_for(self.license_class)

    @property
    def allows_verbatim(self) -> bool:
        return self.use_policy is UsePolicy.VERBATIM

    def attribution(self) -> str:
        """Human-readable credit line for the knowledge graph."""
        parts = [self.source_url]
        if self.upstream_ref:
            parts.append(f"@{self.upstream_ref}")
        if self.spdx_id:
            parts.append(f"({self.spdx_id})")
        return " ".join(parts)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def local_provenance(
    path: str,
    text: str,
    license_class: LicenseClass = LicenseClass.UNKNOWN,
    spdx_id: str | None = None,
    upstream_ref: str | None = None,
) -> Provenance:
    """Provenance for a file already on disk.

    Used when ingesting a local checkout, which is how the pipeline is
    exercised without any network access.

    ``license_class`` defaults to UNKNOWN, not PERMISSIVE. A caller that
    forgets to identify the license gets the restrictive policy; defaulting the
    other way would let an unclassified repository through the gate on a
    forgotten argument, which is the one failure mode that silently defeats it.
    """
    return Provenance(
        source_url=f"file://{path}",
        retrieved_at=now_utc(),
        content_sha256=sha256_of(text),
        license_class=license_class,
        spdx_id=spdx_id,
        upstream_ref=upstream_ref,
    )


@dataclass(frozen=True, slots=True)
class Artifact:
    """A harvested unit of content plus its provenance."""

    identifier: str
    """Stable id within its source, e.g. a repo-relative path or a DOI."""
    content: str
    provenance: Provenance
    media_type: str = "text/plain"


class Harvester(ABC):
    """Base class for all acquisition sources.

    Implementations must attach a :class:`Provenance` to everything they emit;
    the pipeline refuses artifacts without one, so an unlicensed source cannot
    slip into the corpus by omission.
    """

    name: str = "harvester"

    @abstractmethod
    def harvest(self, query: str, limit: int = 50) -> Iterable[Artifact]:
        """Yield artifacts matching ``query``."""

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "type": type(self).__name__}


def summarise_licensing(artifacts: Iterable[Artifact]) -> List[Dict[str, Any]]:
    """Group artifacts by license class -- the audit view of a harvest run."""
    buckets: Dict[LicenseClass, int] = {}
    for a in artifacts:
        cls = a.provenance.license_class
        buckets[cls] = buckets.get(cls, 0) + 1
    return [
        {
            "license_class": cls.value,
            "count": count,
            "policy": policy_for(cls).value,
        }
        for cls, count in sorted(buckets.items(), key=lambda kv: kv[0].value)
    ]
