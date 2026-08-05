"""Tests for agent assembly and the rate-limit failover middleware wiring."""

from __future__ import annotations

from ratelimit_fallback import ModelPool, RateLimitFallbackMiddleware

from general_agent.agent import (
    GENERATION_NAME,
    _NamedModel,
    build_agent,
    build_middleware,
    build_model_chain,
)
from general_agent.config import Settings
from tests.conftest import FakeChatModel, FakeRateLimitError


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


def test_stable_name_middleware_is_inner(settings: Settings) -> None:
    """It has to wrap whichever model the failover middleware selected."""
    middleware = build_middleware(settings)
    assert len(middleware) == 2
    assert isinstance(middleware[0], RateLimitFallbackMiddleware)


def test_run_name_survives_tool_binding() -> None:
    """The regression this wrapper exists for.

    ``.with_config(run_name=...)`` alone is not enough: ``bind_tools`` on a
    ``RunnableBinding`` resolves through to the underlying chat model and
    returns a fresh binding with an empty config, so the name — and with it the
    stable observation name in Langfuse — is silently lost.
    """
    model = FakeChatModel(name="primary")

    plain = model.with_config(run_name=GENERATION_NAME).bind_tools([])
    assert plain.config.get("run_name") is None  # the behaviour being worked around

    named = _NamedModel(model, GENERATION_NAME).bind_tools([])
    assert named.config["run_name"] == GENERATION_NAME


def test_named_model_preserves_the_name_without_tools() -> None:
    """Agents with no tools take the ``bind`` path instead of ``bind_tools``."""
    bound = _NamedModel(FakeChatModel(name="primary"), GENERATION_NAME).bind()
    assert bound.config["run_name"] == GENERATION_NAME


def test_named_model_delegates_everything_else() -> None:
    model = FakeChatModel(name="primary")
    wrapper = _NamedModel(model, GENERATION_NAME)
    assert wrapper.model_name == "primary"
    assert wrapper._llm_type == "fake"


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
