"""Running the agent with tracing attached.

:class:`AgentRunner` is the entry point application code should use. It owns
the agent, the Langfuse client and the conversation id, and turns one user
question into one trace.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from general_agent.agent import build_agent
from general_agent.config import Settings
from general_agent.observability import NullObservability, Observability, build_observability

__all__ = ["AgentReply", "AgentRunner"]

logger = logging.getLogger(__name__)

#: Root observation name. Stable and verb-first, so filters and evaluators
#: that target it keep working across releases.
RUN_NAME = "answer-question"


@dataclass(frozen=True)
class AgentReply:
    """One answer, plus what is needed to trace or score it afterwards."""

    text: str
    #: Langfuse trace id, or ``None`` when tracing is off. Needed to attach
    #: user feedback to the run that produced this answer.
    trace_id: str | None
    #: Model that actually served the reply — which is not necessarily the
    #: first one in the chain, once a 429 has moved the request along.
    model: str | None
    session_id: str


class AgentRunner:
    """An agent bound to a conversation and a trace backend."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        session_id: str | None = None,
        observability: Observability | NullObservability | None = None,
        agent: Any | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.observability = (
            observability if observability is not None else build_observability(self.settings)
        )
        # One id for both, deliberately: the Langfuse session and the
        # checkpointer thread describe the same conversation, so a trace in
        # the Sessions view maps straight onto the memory the agent had.
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        if agent is None:
            agent = build_agent(self.settings, on_rate_limit=self.observability.on_rate_limit)
        self.agent = agent

    @property
    def tracing_enabled(self) -> bool:
        """Whether runs are being recorded to Langfuse."""
        return self.observability.enabled

    def ask(
        self,
        question: str,
        *,
        user_id: str | None = None,
        tags: Sequence[str] = (),
    ) -> AgentReply:
        """Answer one question, recording the whole run as a single trace.

        A blocking wrapper around :meth:`aask`, for scripts and the CLI.
        Callers already inside an event loop must await :meth:`aask` instead —
        there is no correct way to block on a coroutine from the loop running it.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aask(question, user_id=user_id, tags=tags))

        msg = "AgentRunner.ask() cannot be called from a running event loop; await aask() instead"
        raise RuntimeError(msg)

    async def aask(
        self,
        question: str,
        *,
        user_id: str | None = None,
        tags: Sequence[str] = (),
    ) -> AgentReply:
        """Answer one question, recording the whole run as a single trace.

        The agent is driven asynchronously because it has to be: tools loaded
        from an MCP server are built with a coroutine and no synchronous
        implementation, so a sync ``invoke`` would raise the moment the model
        called one. The failover middleware implements ``awrap_model_call``
        alongside its sync hook, so 429 handling is identical on this path.
        """
        payload = {"messages": [{"role": "user", "content": question}]}
        config: dict[str, Any] = {
            "configurable": {"thread_id": self.session_id},
            "callbacks": self.observability.callbacks,
        }

        with self.observability.run(
            RUN_NAME,
            # The user's question, not the LangGraph state dict — this is what
            # the traces table and any evaluator will read.
            input=question,
            session_id=self.session_id,
            user_id=user_id or self.settings.user_id,
            tags=self.settings.resolved_tags(tags),
            metadata={"model_chain": list(self.settings.models)},
        ) as span:
            result = await self.agent.ainvoke(payload, config=config)
            final = result["messages"][-1]
            text = _as_text(final.content)
            model = _served_by(final)

            span.update(
                output=text,
                # Which model answered belongs on the trace: under failover it
                # varies per run, and it is the first thing you want when
                # comparing quality or cost across the chain.
                metadata={"served_by_model": model},
            )
            trace_id = getattr(span, "trace_id", None)

        return AgentReply(text=text, trace_id=trace_id, model=model, session_id=self.session_id)

    def feedback(self, reply: AgentReply, *, positive: bool, comment: str | None = None) -> None:
        """Record thumbs-up/down feedback against the reply's trace."""
        self.observability.score_thumbs(reply.trace_id, positive=positive, comment=comment)

    def close(self) -> None:
        """Flush pending spans and stop the exporter.

        Skipping this in a short-lived process loses every batched span.
        """
        self.observability.shutdown()

    def __enter__(self) -> AgentRunner:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _as_text(content: Any) -> str:
    """Flatten message content to text.

    Content is a plain string for most providers but a list of typed blocks
    for others, so it cannot simply be interpolated.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _served_by(message: Any) -> str | None:
    """Best-effort model name from a response's provider metadata."""
    metadata = getattr(message, "response_metadata", None) or {}
    return metadata.get("model_name") or metadata.get("model") or None
