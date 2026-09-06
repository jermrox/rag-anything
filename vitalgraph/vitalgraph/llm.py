"""Claude wrapper: the language model behind VitalGraph's knowledge graphs.

RAGAnything (via LightRAG) needs two callables -- one that completes text and
one that embeds it. Claude supplies the first. **Anthropic does not offer an
embeddings API**, so the second comes from a separate provider; see
:func:`build_embedding_func`. Pretending otherwise would produce code that
fails on its first call, so the split is explicit here rather than hidden.

What Claude is doing in this system: entity and relation extraction when a
document is ingested, and answering at query time. Both are reasoning tasks
over health evidence where being wrong is expensive, which is why this
defaults to Opus with adaptive thinking rather than to a cheaper model.

Note the deliberate boundary: nightly biometric narratives in
``bridge/summarizer.py`` are still rendered deterministically from measured
numbers, not written by Claude. Nothing hallucinated enters the corpus at
ingest. Claude reasons at query time over facts it can trust.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

#: Default model. Health reasoning over evidence-graded material is exactly the
#: case where the capable model earns its cost.
DEFAULT_MODEL = "claude-opus-5"

#: Streaming is used for every call, so a generous ceiling costs nothing when
#: responses are short and avoids truncating mid-thought when they are not.
DEFAULT_MAX_TOKENS = 16000

#: Kwargs LightRAG passes into the completion callable for its own bookkeeping.
#: They are not API parameters and must never reach the request, or it 400s.
_LIGHTRAG_INTERNAL_KWARGS = frozenset(
    {
        "hashing_kv",
        "keyword_extraction",
        "mode",
        "enable_cot",
        "history_messages",
        "stream",
        "llm_model_func",
        "cache_type",
    }
)

#: API parameters worth forwarding if a caller supplies one.
_FORWARDABLE = frozenset({"temperature", "stop_sequences", "metadata"})


class ClaudeUnavailable(RuntimeError):
    """Raised when the Anthropic SDK or an API key is missing."""


@dataclass
class ClaudeConfig:
    """How VitalGraph calls Claude."""

    # Namespaced deliberately. Bare CLAUDE_* names collide with variables the
    # Claude Code harness already sets in developer environments -- CLAUDE_EFFORT
    # is set here right now, and an un-namespaced read silently inherited it.
    model: str = field(
        default_factory=lambda: os.getenv("VITALGRAPH_CLAUDE_MODEL", DEFAULT_MODEL)
    )
    max_tokens: int = field(
        default_factory=lambda: int(
            os.getenv("VITALGRAPH_CLAUDE_MAX_TOKENS", DEFAULT_MAX_TOKENS)
        )
    )
    effort: str = field(
        default_factory=lambda: os.getenv("VITALGRAPH_CLAUDE_EFFORT", "high")
    )
    """low | medium | high | xhigh | max. Graph extraction over clinical text
    rewards thoroughness, so this defaults higher than a chat app would."""

    adaptive_thinking: bool = True
    enable_fallbacks: bool = True
    """Route around a policy decline instead of failing the call. Health
    material sits near several safety boundaries, so an ingest run that dies
    part-way through a corpus is a real operational risk."""

    def resolve_api_key(self) -> str:
        key = (
            os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("VITALGRAPH_CLAUDE_API_KEY")
            or os.getenv("LLM_BINDING_API_KEY")
        )
        if not key:
            raise ClaudeUnavailable(
                "No Anthropic credential found. Set ANTHROPIC_API_KEY, or run "
                "`ant auth login` and leave it unset -- the SDK reads the stored "
                "profile automatically."
            )
        return key


def _require_anthropic():
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ClaudeUnavailable(
            "The Anthropic SDK is not installed. Install it with:\n"
            '    pip install -e "vitalgraph[rag]"'
        ) from exc
    return anthropic


class ClaudeLLM:
    """Async Claude client shaped for LightRAG's completion contract."""

    def __init__(self, config: ClaudeConfig | None = None, client: Any = None) -> None:
        self.config = config or ClaudeConfig()
        if client is not None:
            # Injectable so the wiring is testable without a key or a network.
            self._client = client
        else:
            anthropic = _require_anthropic()
            # A key is required here rather than left to the SDK's own
            # resolution so the failure names the missing variable instead of
            # surfacing as a 401 several frames later.
            self._client = anthropic.AsyncAnthropic(
                api_key=self.config.resolve_api_key()
            )

    # -- request construction -------------------------------------------

    def _build_messages(
        self, prompt: str, history: Sequence[Dict[str, str]] | None
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for turn in history or []:
            role = turn.get("role")
            if role in ("user", "assistant") and turn.get("content"):
                messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _request_kwargs(
        self, system_prompt: str | None, extra: Dict[str, Any]
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "output_config": {"effort": self.config.effort},
        }
        if self.config.adaptive_thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if system_prompt:
            kwargs["system"] = system_prompt
        if self.config.enable_fallbacks:
            # Scalar form: the server routes by refusal category, so there is
            # no fallback model list to keep current.
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"

        # LightRAG passes internal bookkeeping through the same **kwargs it
        # uses for API options. Forwarding those verbatim is a 400.
        for key, value in extra.items():
            if key in _FORWARDABLE:
                kwargs[key] = value
        return kwargs

    # -- completion ------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: Sequence[Dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Complete one prompt and return the text.

        Streams every call: graph extraction over a long document can run well
        past a non-streaming HTTP timeout, and streaming costs nothing when the
        response turns out to be short.
        """
        request = self._request_kwargs(system_prompt, kwargs)
        request["messages"] = self._build_messages(prompt, history_messages)

        # The fallbacks parameter is beta-gated, so it decides the namespace.
        endpoint = (
            self._client.beta.messages
            if self.config.enable_fallbacks
            else self._client.messages
        )

        async with endpoint.stream(**request) as stream:
            message = await stream.get_final_message()

        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise ClaudeUnavailable(
                f"Claude declined this request (category: {category}). With "
                f"fallbacks enabled this means every model in the chain "
                f"declined, so the content itself is the problem, not routing."
            )

        return "".join(
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        )

    def as_lightrag_func(self):
        """Return the callable RAGAnything expects as ``llm_model_func``.

        LightRAG calls this as ``f(prompt, system_prompt=..., history_messages=...,
        **kwargs)`` and awaits the result.
        """

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: Sequence[Dict[str, str]] | None = None,
            **kwargs: Any,
        ) -> str:
            filtered = {
                k: v for k, v in kwargs.items() if k not in _LIGHTRAG_INTERNAL_KWARGS
            }
            return await self.complete(
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                **filtered,
            )

        return llm_model_func


# -- embeddings ---------------------------------------------------------


def build_embedding_func(embedding_dim: int | None = None) -> Any:
    """Build the embedding callable RAGAnything requires.

    **Anthropic does not provide an embeddings endpoint**, so this is the one
    part of the stack Claude cannot supply. Two providers are supported, tried
    in order:

    1. **Voyage AI** (``VOYAGE_API_KEY``) -- Anthropic's recommended embedding
       partner, and the better retrieval quality of the two.
    2. **sentence-transformers**, running locally with no key and no network.
       Lower quality, but it makes the whole pipeline runnable offline, which
       matters here because health data should not leave the machine without
       an explicit decision to send it.

    Raises:
        ClaudeUnavailable: when neither provider is available, naming both
            remedies rather than failing deep inside LightRAG.
    """
    from lightrag.utils import EmbeddingFunc

    if os.getenv("VOYAGE_API_KEY"):
        try:
            import voyageai

            model = os.getenv("VOYAGE_MODEL", "voyage-3")
            dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "1024"))
            client = voyageai.AsyncClient()

            async def voyage_embed(texts: List[str]):
                import numpy as np

                result = await client.embed(texts, model=model, input_type="document")
                return np.array(result.embeddings)

            return EmbeddingFunc(embedding_dim=dim, func=voyage_embed)
        except ImportError:
            pass  # fall through to the local provider

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ClaudeUnavailable(
            "No embedding provider available, and Anthropic does not offer an "
            "embeddings API. Either set VOYAGE_API_KEY and install `voyageai`, "
            "or install `sentence-transformers` to embed locally."
        ) from exc

    name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    encoder = SentenceTransformer(name)
    dim = embedding_dim or encoder.get_sentence_embedding_dimension()

    async def local_embed(texts: List[str]):
        return encoder.encode(texts, convert_to_numpy=True)

    return EmbeddingFunc(embedding_dim=dim, func=local_embed)
