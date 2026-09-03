"""RAG wiring: connects VitalGraph content to a RAGAnything knowledge graph.

``raganything`` is imported lazily. The analytics, store, GATT and bridge layers
must stay usable -- and testable -- without LightRAG, MinerU or an LLM key
present, so nothing at import time may depend on them.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

from .bridge.summarizer import PeriodSummary, to_content_list

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
    """Raised when RAG features are used without ``raganything`` installed."""


def _require_raganything():
    try:
        from raganything import RAGAnything, RAGAnythingConfig  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RAGAnythingUnavailable(
            "raganything is not installed. Install it with:\n"
            '    pip install -e ".[all]"   (from the repository root)'
        ) from exc
    return RAGAnything, RAGAnythingConfig


def build_openai_rag(working_dir: str | None = None) -> Any:
    """Construct a RAGAnything instance backed by OpenAI models.

    Mirrors the wiring in ``examples/raganything_example.py`` rather than
    inventing a new pattern.
    """
    RAGAnything, RAGAnythingConfig = _require_raganything()
    from functools import partial

    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RAGAnythingUnavailable(
            "Set LLM_BINDING_API_KEY (or OPENAI_API_KEY) to enable RAG queries."
        )
    base_url = os.getenv("LLM_BINDING_HOST")

    config = RAGAnythingConfig(
        working_dir=working_dir or os.getenv("VITALGRAPH_RAG_DIR", "./vg_rag_storage"),
    )

    def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return openai_complete_if_cache(
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=int(os.getenv("EMBEDDING_DIM", "3072")),
        func=partial(
            openai_embed,
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-large"),
            api_key=api_key,
            base_url=os.getenv("EMBEDDING_BINDING_HOST", base_url),
        ),
    )

    return RAGAnything(
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        config=config,
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
        return cls(build_openai_rag(working_dir))

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
