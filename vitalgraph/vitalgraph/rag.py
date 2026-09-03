"""RAG wiring: connects VitalGraph content to a RAGAnything knowledge graph.

``raganything`` is imported lazily. The analytics, store, GATT and bridge layers
must stay usable -- and testable -- without LightRAG, MinerU or an LLM key
present, so nothing at import time may depend on them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

from .bridge.summarizer import PeriodSummary, to_content_list
from .llm import ClaudeConfig, ClaudeLLM, build_embedding_func

#: Framing constraint applied to every generated answer.
#:
#: VitalGraph reports wellness metrics. Presenting them as diagnosis would move
#: the product into medical-device regulatory scope (FDA / EU MDR), so the
#: constraint is enforced at the prompt layer rather than left to chance.
HEALTH_SYSTEM_PROMPT = """You are VitalGraph, a health-analytics assistant.

Rules:
- Ground every claim in the retrieved context. Cite the source of each fact.
- Personal biometric periods are cited as `biometrics://<user>/<period>`; refer
  to them by date so the user can verify against their own data.
- Report wellness and fitness metrics. Do NOT diagnose disease, do NOT
  recommend treatment, and do NOT interpret findings as medical advice.
- When data is insufficient or a personal baseline is not yet established, say
  so plainly instead of inferring a trend.
- Distinguish measured values from population norms and from literature claims.
"""


class RAGAnythingUnavailable(RuntimeError):
    """Raised when RAG features are used without ``raganything`` installed.

    :class:`~vitalgraph.llm.ClaudeUnavailable` subclasses nothing in common
    with this, so the API layer catches both -- to a caller they mean the same
    thing: the RAG surface cannot serve this request.
    """


def _require_raganything():
    try:
        from raganything import RAGAnything, RAGAnythingConfig  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RAGAnythingUnavailable(
            "raganything is not installed. Install it with:\n"
            '    pip install -e ".[all]"   (from the repository root)'
        ) from exc
    return RAGAnything, RAGAnythingConfig


def build_claude_rag(
    working_dir: str | None = None, config: ClaudeConfig | None = None
) -> Any:
    """Construct a RAGAnything instance driven by Claude.

    Claude supplies the completion half -- entity and relation extraction at
    ingest, and answering at query time. The embedding half comes from a
    separate provider because Anthropic offers no embeddings endpoint; see
    :func:`vitalgraph.llm.build_embedding_func`.
    """
    RAGAnything, RAGAnythingConfig = _require_raganything()

    llm = ClaudeLLM(config)
    embedding_func = build_embedding_func()

    rag_config = RAGAnythingConfig(
        working_dir=working_dir or os.getenv("VITALGRAPH_RAG_DIR", "./vg_rag_storage"),
    )

    return RAGAnything(
        llm_model_func=llm.as_lightrag_func(),
        embedding_func=embedding_func,
        config=rag_config,
    )


class VitalGraphRAG:
    """Thin facade over a RAGAnything instance.

    Kept deliberately small: RAGAnything already does the hard work, so this
    class only supplies VitalGraph's content and its health-specific prompt.
    """

    def __init__(self, rag: Any) -> None:
        self._rag = rag

    @classmethod
    def from_env(cls, working_dir: str | None = None) -> "VitalGraphRAG":
        return cls(build_claude_rag(working_dir))

    async def ingest_summary(self, summary: PeriodSummary) -> Dict[str, Any]:
        """Insert one period summary into the knowledge graph.

        Uses the summary's deterministic ``doc_id`` so re-ingesting a period
        replaces it rather than accumulating near-duplicate nights.
        """
        await self._rag.insert_content_list(
            content_list=to_content_list(summary),
            file_path=summary.citation_ref,
            doc_id=summary.doc_id,
        )
        return {"doc_id": summary.doc_id, "citation": summary.citation_ref}

    async def ingest_summaries(
        self, summaries: Iterable[PeriodSummary]
    ) -> List[Dict[str, Any]]:
        return [await self.ingest_summary(s) for s in summaries]

    async def query(self, question: str, mode: str = "mix") -> str:
        """Answer a question over biometrics, literature and protocol knowledge.

        ``mode="mix"`` blends graph traversal with vector retrieval, which is
        what lets one question reach a personal night, a paper, and a GATT
        specification at the same time.
        """
        return await self._rag.aquery(
            question, mode=mode, system_prompt=HEALTH_SYSTEM_PROMPT
        )
