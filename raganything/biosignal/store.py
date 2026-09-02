"""Durable persistence for analysed sessions.

The knowledge graph is the right place to *ask about* physiology and the wrong
place to *compute over* it. Retrieval finds text that resembles a question; it
cannot take a mean, fit a slope, or tell you that four of the eleven sessions in
a window were excluded. Those need the numbers themselves, addressable by date.

So every analysed session is written here as well as indexed. Two consequences
follow, both deliberate:

* **Arithmetic survives a stale graph.** LightRAG will not update a document
  under a `doc_id` it already knows, so a re-analysed session may never reach
  the graph. The store overwrites by session id, which makes it -- not the
  graph -- the numeric source of truth.
* **Withholding survives serialisation.** A record exposes values only through
  ``metrics`` and ``withheld``. The diagnostic ``hrv`` block, which still holds
  values the quality gate rejected, is deliberately not part of the read model.

Layout under the store root (normally ``<working_dir>/biosignal/``)::

    reports/<safe_session_id>.json   source of truth, one envelope per session
    index.jsonl                      flattened cache, rebuildable, last-wins
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ReportRecord",
    "ReportSchemaError",
    "ReportStore",
    "content_hash_for",
]

#: Bumped whenever the on-disk envelope changes shape. Readers migrate forward
#: in memory; a version from the future is an error, never a guess.
REPORT_SCHEMA_VERSION = 1

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class ReportSchemaError(RuntimeError):
    """An envelope this build cannot safely interpret."""


def _safe_name(session_id: str) -> str:
    """Filesystem-safe filename that cannot collide after sanitising.

    Two session ids differing only in characters this strips would otherwise
    overwrite each other, so the original id's digest is always appended.
    """
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:8]
    stem = _UNSAFE.sub("_", session_id)[:120] or "session"
    return f"{stem}-{digest}.json"


def content_hash_for(content_list: Any) -> str:
    """Stable digest of an indexed content list, for staleness detection."""
    payload = json.dumps(content_list, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# migrations
# --------------------------------------------------------------------------

#: version -> function taking an envelope at that version and returning the
#: next one. Applied as a chain on read; disk is untouched unless the caller
#: explicitly asks for a rewrite.
_MIGRATIONS: Dict[int, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}


def _migrate(envelope: Dict[str, Any]) -> Dict[str, Any]:
    version = int(envelope.get("schema_version", 0))
    if version > REPORT_SCHEMA_VERSION:
        raise ReportSchemaError(
            f"report envelope is schema version {version}, but this build "
            f"understands at most {REPORT_SCHEMA_VERSION}. Refusing to guess at "
            "a newer format."
        )
    while version < REPORT_SCHEMA_VERSION:
        migration = _MIGRATIONS.get(version)
        if migration is None:
            raise ReportSchemaError(
                f"no migration from report schema version {version} to {version + 1}"
            )
        envelope = migration(envelope)
        version = int(envelope.get("schema_version", version + 1))
    return envelope


# --------------------------------------------------------------------------
# read model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportRecord:
    """A flattened, query-ready view of one analysed session.

    This is the only shape the deterministic analytics see. It exposes values
    through :meth:`metric` -- which returns ``None`` for a withheld metric as
    surely as for an absent one -- so no caller can accidentally read past a
    withholding decision.
    """

    session_id: str
    subject_id: str
    start: float
    end: float
    duration_s: float
    labels: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    withheld: Dict[str, str] = field(default_factory=dict)
    #: source_id -> QualityReport.to_dict()
    quality: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    #: modality value -> quality score of the stream that metric derives from
    modality_quality: Dict[str, float] = field(default_factory=dict)
    #: metric name -> the quality score its ingest-time gate was evaluated
    #: against, recorded by ``analyze_session`` rather than reconstructed.
    metric_quality: Dict[str, float] = field(default_factory=dict)
    hrv_confidence: Optional[float] = None
    hrv_n_beats: Optional[int] = None
    hrv_artifact_fraction: Optional[float] = None
    warnings: List[str] = field(default_factory=list)
    modalities: List[str] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)
    doc_id: Optional[str] = None
    content_hash: Optional[str] = None
    indexed_at: Optional[float] = None
    schema_version: int = REPORT_SCHEMA_VERSION

    # -- value access ----------------------------------------------------

    def metric(self, name: str) -> Optional[float]:
        """The metric's value, or ``None`` if it is absent **or withheld**."""
        if name in self.withheld:
            return None
        value = self.metrics.get(name)
        return None if value is None else float(value)

    def is_withheld(self, name: str) -> bool:
        return name in self.withheld

    def withheld_reason(self, name: str) -> Optional[str]:
        return self.withheld.get(name)

    def quality_for(self, modality: str) -> Optional[float]:
        score = self.modality_quality.get(modality)
        return None if score is None else float(score)

    def day(self) -> _dt.date:
        """UTC calendar date the session started on."""
        return _dt.datetime.fromtimestamp(self.start, tz=_dt.timezone.utc).date()

    def iso_date(self) -> str:
        return self.day().isoformat()

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "start": self.start,
            "end": self.end,
            "duration_s": self.duration_s,
            "labels": dict(self.labels),
            "metrics": dict(self.metrics),
            "withheld": dict(self.withheld),
            "quality": {k: dict(v) for k, v in self.quality.items()},
            "modality_quality": dict(self.modality_quality),
            "metric_quality": dict(self.metric_quality),
            "hrv_confidence": self.hrv_confidence,
            "hrv_n_beats": self.hrv_n_beats,
            "hrv_artifact_fraction": self.hrv_artifact_fraction,
            "warnings": list(self.warnings),
            "modalities": list(self.modalities),
            "devices": list(self.devices),
            "doc_id": self.doc_id,
            "content_hash": self.content_hash,
            "indexed_at": self.indexed_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReportRecord":
        return cls(
            session_id=str(data["session_id"]),
            subject_id=str(data.get("subject_id", "self")),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            duration_s=float(data.get("duration_s", 0.0)),
            labels=dict(data.get("labels") or {}),
            metrics={k: float(v) for k, v in (data.get("metrics") or {}).items()},
            withheld=dict(data.get("withheld") or {}),
            quality={k: dict(v) for k, v in (data.get("quality") or {}).items()},
            modality_quality={
                k: float(v) for k, v in (data.get("modality_quality") or {}).items()
            },
            metric_quality={
                k: float(v) for k, v in (data.get("metric_quality") or {}).items()
            },
            hrv_confidence=data.get("hrv_confidence"),
            hrv_n_beats=data.get("hrv_n_beats"),
            hrv_artifact_fraction=data.get("hrv_artifact_fraction"),
            warnings=list(data.get("warnings") or []),
            modalities=list(data.get("modalities") or []),
            devices=list(data.get("devices") or []),
            doc_id=data.get("doc_id"),
            content_hash=data.get("content_hash"),
            indexed_at=data.get("indexed_at"),
            schema_version=int(data.get("schema_version", REPORT_SCHEMA_VERSION)),
        )


def _modality_quality_from(report_dict: Mapping[str, Any]) -> Dict[str, float]:
    """Score of the stream each modality's metrics were actually derived from.

    Mirrors the selection ``analyze_session`` makes: the reconciliation winner
    where several streams competed, otherwise the only stream there was. Getting
    this wrong in either direction would make the query-time gate disagree with
    the ingest-time gate, so it is computed from the same recorded facts rather
    than approximated by taking the best score available.
    """
    recorded = report_dict.get("modality_quality")
    if recorded:
        # analyze_session recorded exactly which stream it gated against.
        # Prefer that over re-deriving it: a reconstruction that disagreed
        # with the ingest-time decision would make the query-time gate and
        # the withholding decision contradict each other.
        return {k: float(v) for k, v in recorded.items()}

    quality = report_dict.get("quality") or {}
    fusion = report_dict.get("fusion") or {}
    streams = (report_dict.get("session") or {}).get("streams") or []

    by_modality: Dict[str, List[str]] = {}
    for stream in streams:
        modality = stream.get("modality")
        source_id = (stream.get("provenance") or {}).get("source_id")
        if modality and source_id:
            by_modality.setdefault(modality, []).append(source_id)

    out: Dict[str, float] = {}
    for modality, source_ids in by_modality.items():
        chosen = (fusion.get(modality) or {}).get("chosen")
        if chosen is None and len(source_ids) == 1:
            chosen = source_ids[0]
        if chosen is None:
            continue
        entry = quality.get(chosen)
        if entry and entry.get("score") is not None:
            out[modality] = float(entry["score"])
    return out


def record_from_report(
    report: Any,
    *,
    doc_id: Optional[str] = None,
    content_hash: Optional[str] = None,
    indexed_at: Optional[float] = None,
) -> ReportRecord:
    """Flatten a :class:`~.narrative.SessionReport` into a record."""
    data = report.to_dict()
    session = data.get("session") or {}
    streams = session.get("streams") or []
    hrv = data.get("hrv") or {}

    return ReportRecord(
        session_id=str(session.get("session_id", "unknown")),
        subject_id=str(session.get("subject_id", "self")),
        start=float(session.get("start", 0.0)),
        end=float(session.get("end", 0.0)),
        duration_s=float(session.get("duration_s", 0.0)),
        labels=dict(session.get("labels") or {}),
        metrics={k: float(v) for k, v in (data.get("metrics") or {}).items()},
        withheld=dict(data.get("withheld") or {}),
        quality={k: dict(v) for k, v in (data.get("quality") or {}).items()},
        modality_quality=_modality_quality_from(data),
        metric_quality={
            k: float(v) for k, v in (data.get("metric_quality") or {}).items()
        },
        hrv_confidence=hrv.get("confidence"),
        hrv_n_beats=hrv.get("n_beats_used"),
        hrv_artifact_fraction=hrv.get("artifact_fraction"),
        warnings=list(data.get("warnings") or []),
        modalities=sorted({s.get("modality") for s in streams if s.get("modality")}),
        devices=sorted(
            {
                (s.get("provenance") or {}).get("device")
                for s in streams
                if (s.get("provenance") or {}).get("device")
            }
        ),
        doc_id=doc_id,
        content_hash=content_hash,
        indexed_at=indexed_at,
    )


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


class ReportStore:
    """File-backed store of analysed sessions.

    Deliberately plain files rather than a database: the whole point is that a
    person can open one and read what their body did, and that a corrupted
    write costs one session rather than the history.
    """

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root)
        self.reports_dir = self.root / "reports"
        self.index_path = self.root / "index.jsonl"
        #: session_id -> reason, for files that could not be read.
        self.errors: Dict[str, str] = {}

    # -- construction ----------------------------------------------------

    @classmethod
    def for_rag(cls, rag: Any) -> "ReportStore":
        """Store rooted at ``<rag.working_dir>/biosignal``."""
        working_dir = getattr(rag, "working_dir", None)
        if working_dir is None:
            config = getattr(rag, "config", None)
            working_dir = getattr(config, "working_dir", None)
        if working_dir is None:
            raise ValueError(
                "cannot locate a working_dir on the supplied RAG instance; "
                "construct ReportStore(root=...) explicitly"
            )
        return cls(Path(working_dir) / "biosignal")

    @classmethod
    def maybe_for_rag(cls, rag: Any) -> Optional["ReportStore"]:
        """Like :meth:`for_rag`, but ``None`` instead of raising."""
        try:
            return cls.for_rag(rag)
        except (ValueError, TypeError):
            return None

    # -- writing ---------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        report: Any,
        *,
        doc_id: Optional[str] = None,
        content_hash: Optional[str] = None,
        indexed_at: Optional[float] = None,
    ) -> ReportRecord:
        """Write a report, overwriting any earlier analysis of that session."""
        record = record_from_report(
            report, doc_id=doc_id, content_hash=content_hash, indexed_at=indexed_at
        )
        return self.put_record(record, report=report)

    def put_record(self, record: ReportRecord, report: Any = None) -> ReportRecord:
        self._ensure_dirs()
        envelope = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "written_at": _dt.datetime.now(tz=_dt.timezone.utc).timestamp(),
            "session_id": record.session_id,
            "doc_id": record.doc_id,
            "content_hash": record.content_hash,
            "indexed_at": record.indexed_at,
            "record": record.to_dict(),
            "report": report.to_dict() if report is not None else None,
        }

        path = self.reports_dir / _safe_name(record.session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)  # atomic on POSIX and Windows

        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=str) + "\n")
        return record

    def delete(self, session_id: str) -> bool:
        path = self.reports_dir / _safe_name(session_id)
        if not path.exists():
            return False
        path.unlink()
        self.rebuild_index()
        return True

    # -- reading ---------------------------------------------------------

    def _read_envelope(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self.errors[path.name] = str(exc)
            logger.warning("skipping unreadable report %s: %s", path, exc)
            return None
        try:
            return _migrate(envelope)
        except ReportSchemaError:
            raise
        except Exception as exc:  # noqa: BLE001 - a bad file must not poison list()
            self.errors[path.name] = str(exc)
            logger.warning("skipping malformed report %s: %s", path, exc)
            return None

    def get_envelope(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self.reports_dir / _safe_name(session_id)
        if not path.exists():
            return None
        return self._read_envelope(path)

    def get(self, session_id: str) -> Optional[ReportRecord]:
        envelope = self.get_envelope(session_id)
        if envelope is None:
            return None
        return ReportRecord.from_dict(envelope["record"])

    def _scan(self) -> List[ReportRecord]:
        if not self.reports_dir.exists():
            return []
        records: List[ReportRecord] = []
        for path in sorted(self.reports_dir.glob("*.json")):
            envelope = self._read_envelope(path)
            if envelope is None:
                continue
            try:
                records.append(ReportRecord.from_dict(envelope["record"]))
            except (KeyError, TypeError, ValueError) as exc:
                self.errors[path.name] = str(exc)
                logger.warning("skipping malformed record in %s: %s", path, exc)
        return records

    def list(
        self,
        *,
        start: Optional[float] = None,
        end: Optional[float] = None,
        subject_id: Optional[str] = None,
        labels: Optional[Mapping[str, Any]] = None,
        metric: Optional[str] = None,
    ) -> List[ReportRecord]:
        """Records matching the filters, oldest first.

        ``metric`` selects sessions that *mention* the metric at all -- reported
        or withheld -- because a caller asking about RMSSD needs to know about
        the nights it was withheld just as much as the nights it was not.
        """
        out = []
        for record in self._scan():
            if start is not None and record.start < start:
                continue
            if end is not None and record.start >= end:
                continue
            if subject_id is not None and record.subject_id != subject_id:
                continue
            if labels and any(record.labels.get(k) != v for k, v in labels.items()):
                continue
            if metric is not None and not (
                metric in record.metrics or metric in record.withheld
            ):
                continue
            out.append(record)
        out.sort(key=lambda r: r.start)
        return out

    def span(self) -> Optional[Tuple[float, float]]:
        """``(earliest start, latest end)`` across the store, or ``None``."""
        records = self._scan()
        if not records:
            return None
        return (min(r.start for r in records), max(r.end for r in records))

    # -- maintenance -----------------------------------------------------

    def rebuild_index(self) -> int:
        """Rewrite ``index.jsonl`` from the report files. Always safe."""
        records = self._scan()
        self._ensure_dirs()
        tmp = self.index_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for record in sorted(records, key=lambda r: r.start):
                handle.write(json.dumps(record.to_dict(), default=str) + "\n")
        os.replace(tmp, self.index_path)
        return len(records)

    #: The index is append-only and last-wins, so compaction is a rebuild.
    compact = rebuild_index

    # -- dunder ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._scan())

    def __contains__(self, session_id: object) -> bool:
        if not isinstance(session_id, str):
            return False
        return (self.reports_dir / _safe_name(session_id)).exists()

    def __iter__(self) -> Iterator[ReportRecord]:
        return iter(self.list())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ReportStore(root={str(self.root)!r}, n={len(self)})"
