"""Indexing biosignal sessions into RAG-Anything's knowledge graph.

Why a graph rather than a time-series database: the questions people actually
ask about their own bodies are relational, not numeric. "Does my sleep get
worse in the week after a hard block?" joins sessions to nights to subjective
notes; "which of my devices disagrees during lifting?" joins devices to
modalities to contexts. A time-series store answers none of those without
bespoke query code, while a knowledge graph built over evidence-annotated
narratives answers them by retrieval.

The sessions inserted here become first-class documents alongside whatever else
is in the graph -- training plans, lab results, research papers, coach notes --
which is the point. The physiology stops being a separate app.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from .narrative import SessionReport, analyze_session, to_content_list
from .schema import Session

logger = logging.getLogger(__name__)

__all__ = ["index_report", "index_session", "index_sessions"]


async def index_session(
    rag: Any,
    session: Session,
    doc_id: Optional[str] = None,
    file_path: Optional[str] = None,
    **analysis_kwargs: Any,
) -> SessionReport:
    """Analyse a session and insert it into a :class:`RAGAnything` instance.

    Args:
        rag: An initialised ``RAGAnything``.
        session: The session to index.
        doc_id: Stable document id. Defaults to ``biosignal:<session_id>``, so
            re-indexing a corrected session replaces it rather than duplicating.
        file_path: Citation label. Defaults to the session id.
        **analysis_kwargs: Passed through to :func:`~.narrative.analyze_session`
            (``rest_hr``, ``max_hr``, ``threshold_power``, ``sex``,
            ``min_quality``).

    Returns:
        The :class:`~.narrative.SessionReport` that was indexed, so the caller
        can surface the same metrics and caveats in a UI.
    """
    report = analyze_session(session, **analysis_kwargs)
    await index_report(rag, report, doc_id=doc_id, file_path=file_path)
    return report


async def index_report(
    rag: Any,
    report: SessionReport,
    doc_id: Optional[str] = None,
    file_path: Optional[str] = None,
) -> None:
    """Insert an already-computed report."""
    content_list = to_content_list(report)
    if not content_list:
        logger.warning(
            "session %s produced no content to index", report.session.session_id
        )
        return
    await rag.insert_content_list(
        content_list=content_list,
        file_path=file_path or f"{report.session.session_id}.biosignal",
        doc_id=doc_id or f"biosignal:{report.session.session_id}",
    )


async def index_sessions(
    rag: Any,
    sessions: Sequence[Session],
    **analysis_kwargs: Any,
) -> List[SessionReport]:
    """Index a batch of sessions in order, returning every report.

    Order matters: inserting chronologically lets the graph build relations from
    earlier sessions to later ones as it goes, which is what makes trend
    questions answerable by retrieval instead of by recomputation.
    """
    reports: List[SessionReport] = []
    for session in sorted(sessions, key=lambda s: s.start):
        reports.append(await index_session(rag, session, **analysis_kwargs))
    return reports
