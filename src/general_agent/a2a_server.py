"""Serving this agent over the Agent2Agent (A2A) protocol.

Other agents discover this one by fetching its **agent card** from
``/.well-known/agent-card.json``, then send it work over JSON-RPC or REST. The
protocol is what lets an agent written against a different framework — or
running in someone else's cluster — treat this one as a callable participant
rather than an HTTP endpoint someone has to write a client for.

```bash
general-agent --serve --port 8080
curl http://127.0.0.1:8080/.well-known/agent-card.json
```

Three mappings do the real work:

* **A2A ``context_id`` → conversation.** It becomes both the LangGraph
  checkpointer thread and the Langfuse session id, so a multi-turn exchange
  with a remote agent has memory, and reads as one session in the traces.
* **A2A task lifecycle → run.** ``submitted`` → ``working`` → an artifact
  carrying the answer → ``completed``, or ``failed`` with the error. A caller
  polling the task sees where the work actually is.
* **Tools → skills.** Every tool the agent holds, including the ones onboarded
  from MCP servers, is published as a skill on the card. That is what makes
  the card worth fetching: a remote agent can see that this one can, say, query
  your internal API, without anyone updating a description by hand.

This module serves A2A. It does not make this agent an A2A *client* — calling
other agents is a separate capability, and nothing here reaches outward.

!!! warning "The card is public and the endpoint is unauthenticated"
    The default bind is loopback for that reason. Exposing it more widely means
    putting a proxy in front that terminates TLS and authenticates callers, and
    setting ``AGENT_A2A_URL`` to the address that proxy answers on.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Part,
    Task,
    TaskState,
    TaskStatus,
)

from general_agent.config import Settings

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from general_agent.runner import AgentRunner

__all__ = [
    "AGENT_CARD_PATH",
    "JSONRPC_PATH",
    "REST_PREFIX",
    "GeneralAgentExecutor",
    "build_agent_card",
    "build_app",
    "serve",
]

logger = logging.getLogger(__name__)

#: Where the A2A spec says the card lives, and where clients look first.
AGENT_CARD_PATH = "/.well-known/agent-card.json"
JSONRPC_PATH = "/a2a/jsonrpc"
REST_PREFIX = "/a2a/rest"

#: Protocol version this server speaks. The v0.3 compatibility routes are also
#: mounted, so clients built against the older wire format still work.
PROTOCOL_VERSION = "1.0"

#: Card name for the skill that is just "ask it a question" — the agent's
#: general capability, as distinct from any individual tool.
GENERAL_SKILL_ID = "answer-question"

#: The A2A artifact the answer is returned as.
RESPONSE_ARTIFACT = "response"


def _skills_from_tools(tools: list[Any]) -> list[AgentSkill]:
    """Publish each tool as a skill on the card.

    Deliberately one skill per tool rather than a hand-written summary: the
    tool set is not fixed. Onboard an MCP server and the agent gains
    capabilities that no static description would mention, and a card that does
    not mention them is a card no one can route work against.
    """
    skills = []
    for tool in tools:
        description = (getattr(tool, "description", "") or "").strip()
        skills.append(
            AgentSkill(
                id=tool.name,
                name=tool.name,
                # First line only: tool docstrings carry an Args: block that is
                # for the model calling the tool, not for an agent deciding
                # whether to send work here.
                description=description.split("\n\n")[0] or f"The {tool.name} tool.",
                tags=["tool"],
                input_modes=["text"],
                output_modes=["text"],
            )
        )
    return skills


def build_agent_card(
    settings: Settings | None = None,
    *,
    url: str | None = None,
    tools: list[Any] | None = None,
) -> AgentCard:
    """Describe this agent for the A2A discovery endpoint.

    Args:
        settings: Configuration; read from the environment when omitted.
        url: Base URL other agents reach this server on. Defaults to
            ``AGENT_A2A_URL``, then to the configured bind address.
        tools: The agent's tools, published as skills. Loaded from the
            configuration — MCP servers included — when omitted.

    Returns:
        The agent card, served at :data:`AGENT_CARD_PATH`.
    """
    settings = settings or Settings.from_env()

    if url is None:
        url = settings.a2a_url
    if url is None:
        if settings.a2a_host in {"0.0.0.0", "::", ""}:
            # The bind address is not an address anyone can call back on, and a
            # card advertising it sends every caller nowhere.
            logger.warning(
                "Binding to %s with no AGENT_A2A_URL set; the agent card will advertise "
                "127.0.0.1, which remote agents cannot reach",
                settings.a2a_host,
            )
            url = f"http://127.0.0.1:{settings.a2a_port}"
        else:
            url = f"http://{settings.a2a_host}:{settings.a2a_port}"
    url = url.rstrip("/")

    if tools is None:
        from general_agent.agent import build_agent_tools

        tools = list(build_agent_tools(settings))

    general = AgentSkill(
        id=GENERAL_SKILL_ID,
        name="Answer a question",
        description=(
            "Answer a general-purpose question, using tools where they are more "
            "reliable than recall. Falls over to a backup model when the primary "
            "one is rate limited."
        ),
        tags=["general", "question-answering"],
        examples=["What is 17 * 23?", "What time is it in Tokyo?"],
        input_modes=["text"],
        output_modes=["text"],
    )

    return AgentCard(
        name=settings.agent_name,
        description=(
            "A generic LangChain agent with rate-limit failover and Langfuse "
            "tracing. Its tool set is extensible at runtime by onboarding MCP servers."
        ),
        version=settings.release or "0.1.0",
        capabilities=AgentCapabilities(
            # Task-state updates stream over SSE — submitted, working, the
            # artifact, completed. Individual tokens do not.
            streaming=True,
            push_notifications=False,
        ),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[general, *_skills_from_tools(tools)],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION,
                url=f"{url}{JSONRPC_PATH}",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version=PROTOCOL_VERSION,
                url=f"{url}{REST_PREFIX}",
            ),
        ],
    )


class GeneralAgentExecutor(AgentExecutor):
    """Runs one A2A task by putting it through :class:`AgentRunner`.

    The compiled agent, its middleware and the Langfuse client are built once
    and shared across every caller. Only the cheap per-conversation wrapper is
    per task, which is what keeps a second concurrent caller from paying agent
    construction — and, with MCP servers configured, a round of tool discovery.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        agent: Any | None = None,
        observability: Any | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        from general_agent.agent import build_agent, build_agent_tools
        from general_agent.observability import build_observability

        self.settings = settings or Settings.from_env()
        self.observability = (
            observability if observability is not None else build_observability(self.settings)
        )
        # Discovered once, held onto: these are both what the agent can do and
        # what the card advertises, and asking every MCP server for its tool
        # list twice to answer the same question would be silly.
        self.tools = list(tools) if tools is not None else build_agent_tools(self.settings)
        self.agent = (
            agent
            if agent is not None
            else build_agent(
                self.settings,
                tools=self.tools,
                on_rate_limit=self.observability.on_rate_limit,
            )
        )
        #: Task id to the asyncio task running it, so ``cancel`` can act.
        self._running: dict[str, asyncio.Task[Any]] = {}

    def runner_for(self, context_id: str) -> AgentRunner:
        """A runner scoped to one A2A conversation.

        The context id becomes the checkpointer thread, so a caller that sends
        a second message on the same context gets an agent that remembers the
        first — and the Langfuse session id, so the exchange reads as one
        conversation rather than a scattering of unrelated traces.
        """
        from general_agent.runner import AgentRunner

        return AgentRunner(
            self.settings,
            session_id=context_id,
            observability=self.observability,
            agent=self.agent,
        )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Answer the incoming message, reporting progress as task state."""
        task_id = context.task_id or ""
        context_id = context.context_id or ""
        updater = TaskUpdater(event_queue=event_queue, task_id=task_id, context_id=context_id)

        if context.current_task is None:
            # The task has to exist before anything can update its status —
            # the server rejects a status event for a task it has never seen.
            # A follow-up message on an existing task already has one.
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    history=[context.message] if context.message else [],
                )
            )

        question = context.get_user_input()
        if not question.strip():
            # A rejected task is not a failed one: nothing went wrong here, the
            # request had nothing in it to answer.
            await updater.reject(
                updater.new_agent_message(parts=[Part(text="No text content in the message.")])
            )
            return

        await updater.start_work()

        current = asyncio.current_task()
        if current is not None and task_id:
            self._running[task_id] = current

        try:
            reply = await self.runner_for(context_id).aask(
                question,
                user_id=_user_id_from(context) or self.settings.user_id,
                tags=["a2a"],
            )
        except asyncio.CancelledError:
            # cancel() has already moved the task to canceled; re-raise so the
            # cancellation is not swallowed here.
            raise
        except Exception as exc:
            logger.exception("A2A task %s failed", task_id)
            await updater.failed(
                updater.new_agent_message(parts=[Part(text=f"Agent run failed: {exc}")])
            )
            return
        finally:
            if task_id:
                self._running.pop(task_id, None)

        await updater.add_artifact(
            parts=[Part(text=reply.text)],
            name=RESPONSE_ARTIFACT,
            # Which model served the answer, and the trace it came from: the
            # two things a caller debugging a bad answer needs and cannot see.
            metadata={"served_by_model": reply.model, "trace_id": reply.trace_id},
            last_chunk=True,
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel a running task.

        Real cancellation, not a bookkeeping change: the asyncio task running
        the agent is cancelled, so an in-flight model call stops rather than
        continuing to bill against a task nobody is waiting for.
        """
        task_id = context.task_id or ""
        running = self._running.pop(task_id, None)
        if running is not None:
            running.cancel()

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context.context_id or "",
        )
        await updater.cancel()


def _user_id_from(context: RequestContext) -> str | None:
    """Best-effort end user from the message metadata.

    A2A carries no user identity of its own — the caller is an agent, not a
    person. When one passes ``user_id`` in the message metadata, honouring it
    keeps Langfuse's per-user cost and quality attribution meaningful across
    the hop.
    """
    message = getattr(context, "message", None)
    metadata = getattr(message, "metadata", None)
    if metadata is None:
        return None
    try:
        # Membership first, deliberately: metadata is a protobuf Struct, whose
        # __getitem__ raises ValueError rather than KeyError for an absent key.
        if "user_id" not in metadata:
            return None
        value = metadata["user_id"]
    except (KeyError, TypeError, ValueError):
        return None
    return str(value) if value else None


def build_app(
    settings: Settings | None = None,
    *,
    executor: AgentExecutor | None = None,
    card: AgentCard | None = None,
) -> Starlette:
    """Build the ASGI application serving this agent over A2A.

    Args:
        settings: Configuration; read from the environment when omitted.
        executor: Override the executor. Handy for tests, which can supply one
            wrapping a stub agent instead of a real model.
        card: Override the agent card, e.g. to advertise a public URL that
            differs from anything in the configuration.

    Returns:
        A Starlette app exposing the agent card, JSON-RPC and REST — with the
        v0.3 compatibility routes mounted alongside, so clients written against
        the older protocol version keep working.
    """
    from starlette.applications import Starlette

    settings = settings or Settings.from_env()
    if executor is None:
        executor = GeneralAgentExecutor(settings)
    if card is None:
        # From the executor's own tools, so the card describes the agent that
        # is actually running rather than what a second discovery pass finds.
        card = build_agent_card(settings, tools=list(getattr(executor, "tools", [])))

    handler = DefaultRequestHandler(
        agent_executor=executor,
        # In-process, so tasks are lost on restart. Swap in a database-backed
        # store before running more than one replica: a caller that polls a
        # task id has to reach the process that holds it.
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )

    routes = [
        *create_agent_card_routes(agent_card=card, card_url=AGENT_CARD_PATH),
        *create_jsonrpc_routes(
            request_handler=handler,
            rpc_url=JSONRPC_PATH,
            enable_v0_3_compat=True,
        ),
        *create_rest_routes(
            request_handler=handler,
            path_prefix=REST_PREFIX,
            enable_v0_3_compat=True,
        ),
    ]
    return Starlette(routes=routes)


def serve(settings: Settings | None = None) -> None:
    """Run the A2A server until interrupted."""
    import uvicorn

    settings = settings or Settings.from_env()
    app = build_app(settings)

    logger.info(
        "A2A server on http://%s:%s — agent card at %s",
        settings.a2a_host,
        settings.a2a_port,
        AGENT_CARD_PATH,
    )
    uvicorn.run(app, host=settings.a2a_host, port=settings.a2a_port)
