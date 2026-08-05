"""Tests for agent assembly and the rate-limit failover middleware wiring."""

from __future__ import annotations

import pytest
from langchain.agents import create_agent
from langchain_core.tools import tool
from ratelimit_fallback import ModelPool, RateLimitFallbackMiddleware

from general_agent.agent import (
    GENERATION_NAME,
    build_agent,
    build_middleware,
    build_model_chain,
)
from general_agent.config import Settings
from tests.conftest import FakeChatModel, FakeRateLimitError, NameRecorder


def test_model_chain_passes_strings_through(settings: Settings) -> None:
    assert build_model_chain(settings) == list(settings.models)


def test_model_chain_pools_openrouter_accounts() -> None:
    settings = Settings.from_env(
        {
            "AGENT_MODELS": "openrouter:a/b,google:gemini-flash-latest",
            "AGENT_POOL_OPENROUTER_ACCOUNTS": "true",
            "OPENROUTER_API_KEY": "sk-or-v1-one,sk-or-v1-two",
        }
    )
    chain = build_model_chain(settings)
    assert isinstance(chain[0], ModelPool)
    # A different provider has a different limit, so it stays a plain entry.
    assert chain[1] == "google:gemini-flash-latest"


def test_model_chain_skips_pooling_with_a_single_credential() -> None:
    settings = Settings.from_env(
        {
            "AGENT_MODELS": "openrouter:a/b",
            "AGENT_POOL_OPENROUTER_ACCOUNTS": "true",
            "OPENROUTER_API_KEY": "sk-or-v1-only",
        }
    )
    assert build_model_chain(settings) == ["openrouter:a/b"]


def test_middleware_is_the_custom_failover_middleware(settings: Settings) -> None:
    assert isinstance(build_middleware(settings)[0], RateLimitFallbackMiddleware)


def test_middleware_uses_a_stable_generation_name(settings: Settings) -> None:
    """Observation names must not follow whichever model served the call."""
    assert build_middleware(settings)[0].generation_name == GENERATION_NAME


def test_only_the_failover_middleware_is_needed(settings: Settings) -> None:
    """The stable-name workaround is gone; the middleware handles it upstream."""
    assert len(build_middleware(settings)) == 1


@tool
def _echo(text: str) -> str:
    """Echo the text back."""
    return text


@pytest.mark.parametrize("tools", [[], [_echo]], ids=["no-tools", "with-tools"])
def test_every_model_call_reports_a_stable_name(tools: list) -> None:
    """The property the traces depend on, asserted at this layer.

    Named observations are what Langfuse filters, dashboards and evaluators
    target, so the name must not follow whichever model failover landed on.
    The fix lives in the middleware, but this app is what breaks if it
    regresses — including via a LangChain change to how tools are bound, which
    is exactly how it broke before (``bind_tools`` dropped the run name for
    tool-using agents, leaving each generation named after its model).
    """
    primary = FakeChatModel(name="primary", error=FakeRateLimitError())
    backup = FakeChatModel(name="backup", reply="done")

    middleware = RateLimitFallbackMiddleware(
        models=[primary, backup],
        try_request_model_first=False,
        generation_name=GENERATION_NAME,
    )
    agent = create_agent(model=primary, tools=tools, middleware=[middleware])

    recorder = NameRecorder()
    agent.invoke(
        {"messages": [{"role": "user", "content": "hello"}]},
        config={"callbacks": [recorder]},
    )

    assert recorder.names, "no model call was recorded"
    assert set(recorder.names) == {GENERATION_NAME}, recorder.names


def test_agent_fails_over_to_the_next_model_on_429() -> None:
    """A 429 from the primary model is retried against the fallback."""
    primary = FakeChatModel(name="primary", error=FakeRateLimitError())
    backup = FakeChatModel(name="backup", reply="17 * 23 = 391")
    events = []

    middleware = RateLimitFallbackMiddleware(
        models=[primary, backup],
        try_request_model_first=False,
        on_rate_limit=events.append,
    )

    from langchain.agents import create_agent

    agent = create_agent(model=primary, tools=[], middleware=[middleware])
    result = agent.invoke({"messages": [{"role": "user", "content": "What is 17 * 23?"}]})

    assert result["messages"][-1].content == "17 * 23 = 391"
    assert primary.calls == 1
    assert backup.calls == 1
    assert [(e.model_name, e.next_model_name) for e in events] == [
        ("FakeChatModel:primary", "FakeChatModel:backup")
    ]


def test_agent_builds_with_injected_tools_and_model(monkeypatch) -> None:
    """build_agent wires tools and middleware without touching a provider."""
    fake = FakeChatModel(name="stub")
    # Both the agent's own model and the middleware's lazily-resolved chain
    # entries go through build_chat_model, from two different modules.
    monkeypatch.setattr("ratelimit_fallback.build_chat_model", lambda spec: fake)
    monkeypatch.setattr("ratelimit_fallback.middleware.build_chat_model", lambda spec: fake)

    settings = Settings.from_env({"AGENT_MODELS": "openrouter:a/b"})
    agent = build_agent(settings, tools=[])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "hello"}]},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert result["messages"][-1].content == "ok"
