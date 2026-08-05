"""Assembly of the agent: model chain, middleware, tools.

The interesting part is :func:`build_model_chain`. Everything else is the
standard LangChain 1.0 ``create_agent`` wiring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from general_agent.config import Settings
from general_agent.tools import build_tools

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from ratelimit_fallback import RateLimitEvent

__all__ = [
    "GENERATION_NAME",
    "build_agent",
    "build_agent_tools",
    "build_middleware",
    "build_model_chain",
]

logger = logging.getLogger(__name__)

#: Name every model call reports under, whichever model actually serves it.
GENERATION_NAME = "generate-response"


def build_model_chain(settings: Settings) -> list[Any]:
    """Turn the configured model list into failover-chain entries.

    Plain ``"provider:model"`` strings are passed through untouched. When
    ``pool_openrouter_accounts`` is set, OpenRouter entries become a
    :class:`~ratelimit_fallback.ModelPool` spanning every credential found in
    the configured environment variables — because OpenRouter meters per
    account, so once a key is exhausted, switching *model* on the same key
    walks into the same wall.
    """
    from ratelimit_fallback import ModelPool, accounts_from_env

    if not settings.pool_openrouter_accounts:
        return list(settings.models)

    accounts = accounts_from_env(*settings.openrouter_key_vars, environ=settings.environ)
    if len(accounts) < 2:
        # One credential is not a pool; the extra indirection would only
        # obscure the chain in logs and traces.
        logger.info("Account pooling requested but only %d credential found", len(accounts))
        return list(settings.models)

    logger.info("Pooling %d OpenRouter credentials per model", len(accounts))
    chain: list[Any] = []
    for entry in settings.models:
        if entry.split(":", 1)[0].lower() in {"openrouter", "open-router", "or"}:
            chain.append(ModelPool(entry, accounts=accounts))
        else:
            chain.append(entry)
    return chain


def build_middleware(
    settings: Settings,
    *,
    on_rate_limit: Callable[[RateLimitEvent], None] | None = None,
) -> list[Any]:
    """Build the agent's middleware stack.

    Currently one entry: the custom rate-limit failover middleware from
    https://github.com/bug-ss/ratelimit-fallback. It hooks ``wrap_model_call``,
    so it sees the exception a 429 raises and can re-issue the same request
    against the next model in the chain — something ``after_model`` cannot do,
    because a failed call never produces a message for it to run on.
    """
    from ratelimit_fallback import RateLimitFallbackMiddleware

    fallback = RateLimitFallbackMiddleware(
        models=build_model_chain(settings),
        # The chain is already the full priority order, starting with the
        # agent's own model — trying the request model first would just call
        # models[0] twice.
        try_request_model_first=False,
        cooldown_seconds=settings.cooldown_seconds,
        cooldown_scope="account" if settings.account_scoped_cooldown else "route",
        respect_retry_after=settings.respect_retry_after,
        on_rate_limit=on_rate_limit,
        # One stable observation name for every model call. Left to the
        # tracing integration's default, each generation would be named after
        # whichever model served it — so under failover the name changes from
        # trace to trace, breaking any filter or evaluator targeting it.
        generation_name=GENERATION_NAME,
        environ=settings.environ,
    )
    return [fallback]


def build_agent_tools(settings: Settings) -> list[BaseTool]:
    """The built-in tools plus every tool the configured MCP servers expose.

    MCP tools come last and lose any name collision. Two tools with one name is
    not a resolvable situation — the model gets one schema and calls whichever
    the framework happened to keep — so the local, known-good one wins and the
    clash is logged. ``AGENT_MCP_TOOL_PREFIX`` (on by default) makes collisions
    rare in the first place by namespacing MCP tools by server.
    """
    from general_agent.mcp import load_mcp_tools_blocking

    tools = build_tools(
        enable_web_search=settings.enable_web_search,
        max_search_results=settings.max_search_results,
    )
    taken = {tool.name for tool in tools}

    for tool in load_mcp_tools_blocking(settings):
        if tool.name in taken:
            logger.warning("MCP tool %r collides with an existing tool; skipping it", tool.name)
            continue
        taken.add(tool.name)
        tools.append(tool)

    return tools


def build_agent(
    settings: Settings | None = None,
    *,
    tools: Sequence[BaseTool] | None = None,
    on_rate_limit: Callable[[RateLimitEvent], None] | None = None,
    checkpointer: Any | None = None,
) -> Any:
    """Create the agent.

    Args:
        settings: Configuration; read from the environment when omitted.
        tools: Override the default tool set. Passing this skips MCP
            onboarding entirely — you are supplying the full set.
        on_rate_limit: Called on every 429 failover. Pass
            ``Observability.on_rate_limit`` to record failovers on the trace.
        checkpointer: Conversation persistence. Defaults to an in-process
            saver, which is enough for a CLI; swap in a database-backed
            checkpointer for a server.

    Returns:
        The compiled LangGraph agent.
    """
    settings = settings or Settings.from_env()
    from ratelimit_fallback import build_chat_model

    agent_tools = list(tools) if tools is not None else build_agent_tools(settings)

    return create_agent(
        # Built through ratelimit-fallback rather than passed as a bare
        # "openrouter:..." string: LangChain's own string resolution routes
        # that prefix to a separate langchain-openrouter package, while this
        # reaches OpenRouter through langchain-openai.
        model=build_chat_model(settings.primary_model),
        tools=agent_tools,
        system_prompt=settings.system_prompt,
        middleware=build_middleware(settings, on_rate_limit=on_rate_limit),
        checkpointer=checkpointer if checkpointer is not None else InMemorySaver(),
        name=settings.agent_name,
    )
