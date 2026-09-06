"""Indexing biosignal sessions into RAG-Anything's knowledge graph.

Why a graph rather than a time-series database: the questions people actually
ask about their own bodies are relational, not numeric. "Does my sleep get
worse in the week after a hard block?" joins sessions to nights to subjective
notes; "which of my devices disagrees during lifting?" joins devices to
modalities to contexts. A time-series store answers none of those without
bespoke query code, while a knowledge graph built over evidence-annotated
narratives answers them by retrieval.

The graph is not, however, where arithmetic happens. Every session indexed here
is also written to a :class:`~.store.ReportStore`, which is what
:mod:`.timeseries` computes over. That split matters because of a hard property
of the pipeline underneath: **an insert under a ``doc_id`` the graph already
knows is discarded, not applied.** A re-analysed session can therefore fail to
reach the graph while remaining perfectly correct in the store.
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional, Sequence

from .narrative import SessionReport, analyze_session, to_content_list
from .schema import Session
from .store import ReportStore, content_hash_for

logger = logging.getLogger(__name__)

__all__ = ["index_report", "index_session", "index_sessions", "reindex_session"]


async def index_session(
    rag: Any,
    session: Session,
    doc_id: Optional[str] = None,
    file_path: Optional[str] = None,
    store: Optional[ReportStore] = None,
    replace: bool = False,
    **analysis_kwargs: Any,
) -> SessionReport:
    """Analyse a session, persist the report, and index it.

    Args:
        rag: An initialised ``RAGAnything``.
        session: The session to index.
        doc_id: Stable document id. Defaults to ``biosignal:<session_id>``.
        file_path: Citation label. Defaults to the session id.
        store: Where to persist the analysed report. Defaults to
            ``<rag.working_dir>/biosignal``; pass one explicitly to override,
            or if ``rag`` exposes no working directory.
        replace: Delete any existing document under ``doc_id`` before
            inserting. See :func:`index_report` for why this is not the
            default and what happens without it.
        **analysis_kwargs: Passed to :func:`~.narrative.analyze_session`
            (``rest_hr``, ``max_hr``, ``threshold_power``, ``sex``,
            ``min_quality``).

    Returns:
        The :class:`~.narrative.SessionReport` that was produced, so the caller
        can surface the same metrics and caveats in a UI.
    """
    report = analyze_session(session, **analysis_kwargs)
    await index_report(
        rag,
        report,
        doc_id=doc_id,
        file_path=file_path,
        store=store,
        replace=replace,
    )
    return report


async def index_report(
    rag: Any,
    report: SessionReport,
    doc_id: Optional[str] = None,
    file_path: Optional[str] = None,
    store: Optional[ReportStore] = None,
    replace: bool = False,
) -> None:
    """Persist a computed report and insert it into the graph.

    **On re-indexing.** Inserting under a ``doc_id`` the graph has already seen
    does *not* update it. The pipeline underneath filters ids it knows and
    records the attempt as a separate failed duplicate; the original document's
    chunks are left exactly as they were. So:

    * ``replace=False`` (default) and the content is unchanged -- nothing to do,
      and the insert is skipped rather than leaving a failed duplicate behind.
    * ``replace=False`` and the content *has* changed -- the store is updated,
      the graph is not, and a warning says so. Numbers stay correct; retrieval
      goes stale.
    * ``replace=True`` -- the existing document is deleted first, so the insert
      genuinely applies.

    The report is written to the store in every one of those cases, including
    when the graph insert is skipped. That is deliberate: it is what keeps the
    arithmetic correct against a stale graph.
    """
    content_list = to_content_list(report)
    session_id = report.session.session_id
    resolved_doc_id = doc_id or f"biosignal:{session_id}"
    digest = content_hash_for(content_list)

    if store is None:
        store = ReportStore.maybe_for_rag(rag)
        if store is None:
            logger.info(
                "no report store available for %s (the RAG object exposes no "
                "working_dir); deterministic queries over this session will not "
                "be possible",
                session_id,
            )

    previous = store.get(session_id) if store is not None else None
    unchanged = previous is not None and previous.content_hash == digest

    if not content_list:
        logger.warning("session %s produced no content to index", session_id)
    elif unchanged and not replace:
        logger.info(
            "session %s is unchanged since it was last indexed; skipping the "
            "graph insert",
            session_id,
        )
    else:
        if replace:
            await _delete_document(rag, resolved_doc_id)
        elif previous is not None:
            logger.warning(
                "session %s has changed since it was last indexed, but the graph "
                "will not update a document it already knows. The report store is "
                "now ahead of the knowledge graph for this session; re-run with "
                "replace=True to bring the graph back into line.",
                session_id,
            )
        await rag.insert_content_list(
            content_list=content_list,
            file_path=file_path or f"{session_id}.biosignal",
            doc_id=resolved_doc_id,
        )

    if store is not None:
        store.put(
            report,
            doc_id=resolved_doc_id,
            content_hash=digest,
            indexed_at=time.time(),
        )


async def _delete_document(rag: Any, doc_id: str) -> None:
    """Remove a document so a re-insert genuinely applies."""
    lightrag = getattr(rag, "lightrag", None)
    delete = getattr(lightrag, "adelete_by_doc_id", None)
    if delete is None:
        raise NotImplementedError(
            f"cannot replace {doc_id}: the underlying LightRAG instance exposes "
            "no adelete_by_doc_id. Delete the document by hand, or index under a "
            "new doc_id."
        )
    await delete(doc_id)
    logger.info("deleted %s ahead of re-indexing", doc_id)


async def reindex_session(
    rag: Any,
    session: Session,
    store: Optional[ReportStore] = None,
    **kwargs: Any,
) -> SessionReport:
    """Re-analyse a session and genuinely replace it in the graph."""
    return await index_session(rag, session, store=store, replace=True, **kwargs)


async def index_sessions(
    rag: Any,
    sessions: Sequence[Session],
    store: Optional[ReportStore] = None,
    **analysis_kwargs: Any,
) -> List[SessionReport]:
    """Index a batch of sessions in order, returning every report.

    Order matters: inserting chronologically lets the graph build relations from
    earlier sessions to later ones as it goes, which is what makes trend
    questions answerable by retrieval instead of by recomputation.
    """
    reports: List[SessionReport] = []
    for session in sorted(sessions, key=lambda s: s.start):
        reports.append(
            await index_session(rag, session, store=store, **analysis_kwargs)
        )
    return reports
