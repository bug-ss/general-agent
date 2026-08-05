"""Assembly of the agent: model chain, middleware, tools.

The interesting part is :func:`build_model_chain`. Everything else is the
standard LangChain 1.0 ``create_agent`` wiring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langgraph.checkpoint.memory import InMemorySaver

from general_agent.config import Settings
from general_agent.tools import build_tools

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from ratelimit_fallback import RateLimitEvent

__all__ = [
    "GENERATION_NAME",
    "build_agent",
    "build_middleware",
    "build_model_chain",
    "stable_generation_name",
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


class _NamedModel:
    """Delegating wrapper that keeps a run name attached across tool binding.

    ``RateLimitFallbackMiddleware`` applies ``.with_config(run_name=...)`` so
    every generation reports under one stable name. That wrapper does not
    survive an agent that has tools: ``create_agent`` calls
    ``request.model.bind_tools(...)``, which on a ``RunnableBinding`` resolves
    straight through to the underlying chat model and returns a fresh binding
    with an empty config — dropping the run name, and with it the stable
    observation name in Langfuse.

    Re-applying the name *after* binding is what makes it stick. Everything
    other than ``bind`` / ``bind_tools`` is delegated to the real model.
    """

    __slots__ = ("_model", "_run_name")

    def __init__(self, model: Any, run_name: str) -> None:
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_run_name", run_name)

    def bind_tools(self, *args: Any, **kwargs: Any) -> Any:
        bound = self._model.bind_tools(*args, **kwargs)
        return bound.with_config(run_name=self._run_name)

    def bind(self, **kwargs: Any) -> Any:
        return self._model.bind(**kwargs).with_config(run_name=self._run_name)

    def with_config(self, *args: Any, **kwargs: Any) -> Any:
        return self._model.with_config(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __repr__(self) -> str:
        return f"_NamedModel({self._model!r}, run_name={self._run_name!r})"


def stable_generation_name(run_name: str = GENERATION_NAME) -> Any:
    """Middleware giving every model call the same observation name.

    Langfuse's [best practices](https://langfuse.com/docs/observability/best-practices)
    warn against names that follow the model: evaluators, dashboards and saved
    filters target observations *by name*, so a model-derived name breaks the
    moment the model changes. Under 429 failover it changes run to run.

    Must sit **after** the failover middleware in the stack so that it wraps
    whichever model that one selected.
    """

    @wrap_model_call(name=f"stable-generation-name[{run_name}]")
    def middleware(request: Any, handler: Callable[[Any], Any]) -> Any:
        return handler(request.override(model=_NamedModel(request.model, run_name)))

    return middleware


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
        generation_name=GENERATION_NAME,
        environ=settings.environ,
    )
    # Inner, so it wraps whichever model the failover middleware picked. This
    # is what actually makes generation_name survive tool binding — see
    # _NamedModel.
    return [fallback, stable_generation_name(GENERATION_NAME)]


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
        tools: Override the default tool set.
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

    agent_tools = (
        list(tools)
        if tools is not None
        else build_tools(
            enable_web_search=settings.enable_web_search,
            max_search_results=settings.max_search_results,
        )
    )

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
