"""Langfuse tracing for the agent.

Built on the ``ratelimit-fallback`` package's tracing helpers, which already
wire up the Langfuse LangChain integration, secret masking and 429-failover
events. This module adds the parts specific to running an application:

* a single object that owns the client, the LangChain callback handler and the
  root observation, so callers never juggle three things;
* a **no-op implementation** used when tracing is off or unconfigured, so the
  agent code path is identical either way and a missing key can never take the
  agent down;
* user-feedback scores attached to the trace a reply came from.

Nothing here is imported at module load beyond type hints — the Langfuse client
is built inside :func:`build_observability`, which must run *after*
``load_dotenv()`` or the SDK captures an environment with no credentials in it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ratelimit_fallback import RateLimitEvent

    from general_agent.config import Settings

__all__ = ["NullObservability", "Observability", "TracedRun", "build_observability"]

logger = logging.getLogger(__name__)

#: Score name for explicit thumbs feedback. Named after the *signal* rather
#: than what we hope it measures: a thumbs-down says the user was unhappy, not
#: which part of the answer was wrong.
THUMBS_SCORE = "user-thumbs"


class TracedRun(Protocol):
    """The handle yielded by :meth:`Observability.run`."""

    trace_id: str | None

    def update(self, **kwargs: Any) -> None:
        """Set output/metadata on the root observation."""


class _NullRun:
    """Stand-in for a root observation when tracing is disabled."""

    trace_id: str | None = None

    def update(self, **kwargs: Any) -> None:
        return None


class Observability:
    """Owns the Langfuse client and exposes it as agent-shaped operations."""

    enabled = True

    def __init__(self, client: Any, handler: Any, settings: Settings) -> None:
        self._client = client
        self._handler = handler
        self._settings = settings

    @property
    def callbacks(self) -> list[Any]:
        """Callbacks to pass as ``config={"callbacks": ...}`` on every run.

        The LangChain integration is what captures model names, token usage,
        tool calls and the nesting between them — all of it automatically, and
        with far more context than hand-rolled spans would carry.
        """
        return [self._handler]

    @contextmanager
    def run(
        self,
        name: str,
        *,
        input: Any = None,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[TracedRun]:
        """Wrap one agent run in a root ``agent`` observation.

        Left to itself the LangChain handler makes the raw LangGraph state dict
        the trace's input and output — a JSON blob in the traces table that no
        evaluator can read. This sets the input to the user's actual question
        and lets the caller set the answer as the output.

        Args:
            name: Stable, verb-first observation name (``answer-question``).
                Dashboards, evaluators and saved filters target it by name, so
                it must not vary per run.
            input: The user-facing request, not the whole argument payload.
            session_id: Groups the turns of one conversation.
            user_id: Attributes cost and quality to an end user.
            tags: Business dimensions to break metrics down by.
            metadata: Extra context that fits in neither name nor input.
        """
        from ratelimit_fallback import traced_agent_run

        with traced_agent_run(
            name,
            input=input,
            session_id=session_id,
            user_id=user_id,
            tags=list(tags) or None,
            metadata=dict(metadata) if metadata else None,
            version=self._settings.release,
        ) as span:
            yield span

    def on_rate_limit(self, event: RateLimitEvent) -> None:
        """Record a 429 failover on the active trace.

        Handed to the middleware as its ``on_rate_limit`` callback. Without it
        a trace shows one successful generation and gives no hint that the
        primary model was ever rate limited.
        """
        from ratelimit_fallback import trace_rate_limit_event

        trace_rate_limit_event(event)

    def score_thumbs(
        self,
        trace_id: str | None,
        *,
        positive: bool,
        comment: str | None = None,
    ) -> None:
        """Attach thumbs-up/down feedback to a finished trace."""
        if not trace_id:
            return
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=THUMBS_SCORE,
                value=1 if positive else 0,
                data_type="BOOLEAN",
                comment=comment,
            )
        except Exception:  # pragma: no cover - feedback must never break a run
            logger.debug("Could not record feedback score", exc_info=True)

    def flush(self) -> None:
        """Send buffered spans. Required before a short-lived process exits."""
        try:
            self._client.flush()
        except Exception:  # pragma: no cover - shutdown path
            logger.debug("Langfuse flush failed", exc_info=True)

    def shutdown(self) -> None:
        """Flush and stop the background exporter."""
        try:
            self._client.shutdown()
        except Exception:  # pragma: no cover - shutdown path
            logger.debug("Langfuse shutdown failed", exc_info=True)


class NullObservability:
    """Does nothing, with the same surface as :class:`Observability`.

    Lets the agent run unchanged when Langfuse is not configured, instead of
    scattering ``if tracing_enabled`` through the call path.
    """

    enabled = False

    @property
    def callbacks(self) -> list[Any]:
        return []

    @contextmanager
    def run(self, name: str, **kwargs: Any) -> Iterator[TracedRun]:
        yield _NullRun()

    def on_rate_limit(self, event: RateLimitEvent) -> None:
        return None

    def score_thumbs(self, trace_id: str | None, **kwargs: Any) -> None:
        return None

    def flush(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def build_observability(settings: Settings) -> Observability | NullObservability:
    """Create the tracing backend for these settings.

    Returns a :class:`NullObservability` — never raises — when tracing is
    disabled, credentials are missing, or the SDK is not installed. Losing
    traces is an observability problem; taking the agent down over it would be
    a much worse one.
    """
    if not settings.tracing_enabled:
        logger.info("Langfuse tracing disabled (no credentials or AGENT_TRACING_ENABLED=false)")
        return NullObservability()

    try:
        from ratelimit_fallback import configure_langfuse, get_langfuse_handler

        client = configure_langfuse(
            environment=settings.environment,
            release=settings.release,
            sample_rate=settings.sample_rate,
        )
        handler = get_langfuse_handler()
    except Exception:
        logger.warning("Langfuse setup failed; continuing without tracing", exc_info=True)
        return NullObservability()

    # auth_check() is a cheap round trip that turns "traces silently never
    # appear" into a warning at startup, which is where you can act on it.
    try:
        if not client.auth_check():
            logger.warning("Langfuse credentials were rejected; traces may not be recorded")
    except Exception:  # pragma: no cover - network dependent
        logger.debug("Langfuse auth check could not be completed", exc_info=True)

    logger.info("Langfuse tracing enabled (environment=%s)", settings.environment)
    return Observability(client, handler, settings)
