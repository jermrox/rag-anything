"""The Claude wrapper.

Every test runs without a network call or an API key: the client is injected,
so the request-construction logic -- which is where the real bugs live -- is
verified directly.
"""

import asyncio
from types import SimpleNamespace

import pytest

from vitalgraph.llm import (
    DEFAULT_MODEL,
    ClaudeConfig,
    ClaudeLLM,
    ClaudeUnavailable,
)


class FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self._message


class FakeMessages:
    """Records the request it was handed so assertions can inspect it."""

    def __init__(self, message):
        self._message = message
        self.last_request = None

    def stream(self, **kwargs):
        self.last_request = kwargs
        return FakeStream(self._message)


class FakeClient:
    def __init__(self, message):
        self.messages = FakeMessages(message)
        self.beta = SimpleNamespace(messages=FakeMessages(message))


def _message(text="hello", stop_reason="end_turn", stop_details=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=stop_details,
    )


def _llm(message=None, **cfg):
    client = FakeClient(message or _message())
    return ClaudeLLM(ClaudeConfig(**cfg), client=client), client


def test_defaults_to_opus():
    assert DEFAULT_MODEL == "claude-opus-5"
    assert ClaudeConfig().model == "claude-opus-5"


def test_env_vars_are_namespaced(monkeypatch):
    """Regression: bare CLAUDE_EFFORT is set by the Claude Code harness itself,
    so an un-namespaced read silently inherited an unrelated value."""
    monkeypatch.setenv("CLAUDE_EFFORT", "low")
    monkeypatch.delenv("VITALGRAPH_CLAUDE_EFFORT", raising=False)
    assert ClaudeConfig().effort == "high"

    monkeypatch.setenv("VITALGRAPH_CLAUDE_EFFORT", "max")
    assert ClaudeConfig().effort == "max"


def test_completion_returns_text():
    llm, _ = _llm(_message("the answer"))
    assert asyncio.run(llm.complete("q")) == "the answer"


def test_adaptive_thinking_is_on_by_default():
    llm, client = _llm()
    asyncio.run(llm.complete("q"))
    assert client.beta.messages.last_request["thinking"] == {"type": "adaptive"}


def test_effort_is_sent_inside_output_config():
    """Effort is a field of output_config, not a top-level parameter."""
    llm, client = _llm(effort="max")
    asyncio.run(llm.complete("q"))
    assert client.beta.messages.last_request["output_config"] == {"effort": "max"}


def test_fallbacks_enabled_by_default_uses_the_beta_endpoint():
    llm, client = _llm()
    asyncio.run(llm.complete("q"))
    req = client.beta.messages.last_request
    assert req["fallbacks"] == "default"
    assert req["betas"] == ["server-side-fallback-2026-07-01"]
    assert client.messages.last_request is None  # non-beta path unused


def test_fallbacks_can_be_disabled_and_then_uses_the_plain_endpoint():
    llm, client = _llm(enable_fallbacks=False)
    asyncio.run(llm.complete("q"))
    assert client.messages.last_request is not None
    assert "fallbacks" not in client.messages.last_request


def test_system_prompt_is_forwarded():
    llm, client = _llm()
    asyncio.run(llm.complete("q", system_prompt="be terse"))
    assert client.beta.messages.last_request["system"] == "be terse"


def test_history_is_replayed_before_the_prompt():
    llm, client = _llm()
    history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    asyncio.run(llm.complete("c", history_messages=history))
    msgs = client.beta.messages.last_request["messages"]
    assert [m["content"] for m in msgs] == ["a", "b", "c"]
    assert msgs[-1]["role"] == "user"


def test_malformed_history_entries_are_dropped():
    llm, client = _llm()
    bad = [{"role": "system", "content": "x"}, {"role": "user", "content": ""}]
    asyncio.run(llm.complete("q", history_messages=bad))
    assert [m["content"] for m in client.beta.messages.last_request["messages"]] == [
        "q"
    ]


def test_lightrag_internal_kwargs_never_reach_the_api():
    """LightRAG passes bookkeeping through the same **kwargs it uses for API
    options. Forwarding those verbatim is a 400."""
    llm, client = _llm()
    func = llm.as_lightrag_func()
    asyncio.run(func("q", hashing_kv={"x": 1}, keyword_extraction=True, mode="local"))
    req = client.beta.messages.last_request
    for leaked in ("hashing_kv", "keyword_extraction", "mode"):
        assert leaked not in req


def test_genuine_api_options_are_forwarded():
    llm, client = _llm()
    func = llm.as_lightrag_func()
    asyncio.run(func("q", temperature=0.2, hashing_kv={"x": 1}))
    assert client.beta.messages.last_request["temperature"] == 0.2


def test_refusal_raises_with_the_category():
    llm, _ = _llm(
        _message(
            "", stop_reason="refusal", stop_details=SimpleNamespace(category="bio")
        )
    )
    with pytest.raises(ClaudeUnavailable, match="bio"):
        asyncio.run(llm.complete("q"))


def test_missing_credential_names_both_remedies(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "VITALGRAPH_CLAUDE_API_KEY",
        "LLM_BINDING_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(ClaudeUnavailable) as exc:
        ClaudeConfig().resolve_api_key()
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "ant auth login" in str(exc.value)


def test_lightrag_func_is_awaitable():
    llm, _ = _llm(_message("ok"))
    assert asyncio.run(llm.as_lightrag_func()("q")) == "ok"
