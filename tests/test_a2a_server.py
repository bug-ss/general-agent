"""Tests for the A2A server.

Driven over real HTTP through Starlette's test client rather than by calling
the executor directly: what this feature promises is that *another agent* can
talk to this one, and only the wire proves that. The agent behind it is a stub,
so nothing reaches a model.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

pytest.importorskip("a2a", reason="needs the a2a extra")
pytest.importorskip("starlette", reason="needs the a2a extra")

from starlette.testclient import TestClient

from general_agent.a2a_server import (
    AGENT_CARD_PATH,
    GENERAL_SKILL_ID,
    JSONRPC_PATH,
    GeneralAgentExecutor,
    build_agent_card,
    build_app,
)
from general_agent.config import Settings
from general_agent.observability import NullObservability


class StubAgent:
    """Stands in for the compiled graph, recording the configs it was given."""

    def __init__(self, text: str = "391") -> None:
        self.text = text
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.configs.append(config)

        class _Message:
            content: str = self.text
            response_metadata: dict[str, Any] = {"model_name": "backup"}  # noqa: RUF012

        return {"messages": [_Message()]}


class ExplodingAgent:
    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        msg = "provider is down"
        raise RuntimeError(msg)


class SlowAgent:
    """Blocks until cancelled, and records that it noticed."""

    def __init__(self) -> None:
        import asyncio

        self.started = asyncio.Event()
        self.cancelled = False

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return {"messages": []}  # pragma: no cover - the sleep never finishes


class RecordingQueue:
    """The A2A event queue, reduced to the one method the executor uses."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def enqueue_event(self, event: Any) -> None:
        self.events.append(event)


class StubRequestContext:
    """Just the fields the executor reads off a real RequestContext."""

    def __init__(self, task_id: str, context_id: str, text: str) -> None:
        self.task_id = task_id
        self.context_id = context_id
        self.current_task = None
        self.message = None
        self._text = text

    def get_user_input(self) -> str:
        return self._text


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env({})


def _executor(settings: Settings, agent: Any) -> GeneralAgentExecutor:
    return GeneralAgentExecutor(
        settings,
        agent=agent,
        observability=NullObservability(),
        tools=[],
    )


def _send(client: TestClient, text: str, *, context_id: str | None = None) -> dict[str, Any]:
    """Send one A2A message over JSON-RPC and return the parsed result."""
    message: dict[str, Any] = {
        "messageId": uuid.uuid4().hex,
        "role": "ROLE_USER",
        "parts": [{"text": text}],
    }
    if context_id:
        message["contextId"] = context_id

    response = client.post(
        JSONRPC_PATH,
        # Without this header the SDK reads the request as protocol 0.3 —
        # see test_a_v0_3_client_is_still_understood for that path.
        headers={"A2A-Version": "1.0"},
        json={
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "SendMessage",
            "params": {"message": message},
        },
    )
    response.raise_for_status()
    body = response.json()
    assert "error" not in body, body
    return body["result"]


def _artifact_text(result: dict[str, Any]) -> str:
    """Pull the answer out of a completed task's artifacts."""
    task = result.get("task", result)
    parts = [part for artifact in task["artifacts"] for part in artifact["parts"]]
    return "".join(part.get("text", "") for part in parts)


# --- The agent card ----------------------------------------------------


def test_the_agent_card_is_served_where_the_spec_says(settings: Settings) -> None:
    with TestClient(build_app(settings, executor=_executor(settings, StubAgent()))) as client:
        card = client.get(AGENT_CARD_PATH).json()

    assert card["name"] == "general-agent"
    assert any(skill["id"] == GENERAL_SKILL_ID for skill in card["skills"])


def test_tools_are_published_as_skills(settings: Settings) -> None:
    """Including MCP-onboarded ones — that is what makes the card worth fetching."""
    from general_agent.tools import calculator

    card = build_agent_card(settings, tools=[calculator], url="https://agents.example.com")
    skills = {skill.id for skill in card.skills}

    assert skills == {GENERAL_SKILL_ID, "calculator"}


def test_the_card_advertises_the_public_url_not_the_bind_address() -> None:
    """Behind a proxy the bind address is not an address anyone can call."""
    settings = Settings.from_env(
        {"AGENT_A2A_HOST": "0.0.0.0", "AGENT_A2A_URL": "https://agents.example.com"}
    )
    card = build_agent_card(settings, tools=[])

    assert all(
        interface.url.startswith("https://agents.example.com")
        for interface in card.supported_interfaces
    )


def test_a_wildcard_bind_without_a_public_url_does_not_advertise_the_wildcard(caplog) -> None:
    settings = Settings.from_env({"AGENT_A2A_HOST": "0.0.0.0"})
    card = build_agent_card(settings, tools=[])

    assert all("0.0.0.0" not in interface.url for interface in card.supported_interfaces)
    assert "AGENT_A2A_URL" in caplog.text


# --- Answering ---------------------------------------------------------


def test_another_agent_gets_an_answer(settings: Settings) -> None:
    with TestClient(build_app(settings, executor=_executor(settings, StubAgent()))) as client:
        result = _send(client, "What is 17 * 23?")

    assert _artifact_text(result) == "391"


def test_the_task_reaches_a_completed_state(settings: Settings) -> None:
    with TestClient(build_app(settings, executor=_executor(settings, StubAgent()))) as client:
        result = _send(client, "hi")

    task = result.get("task", result)
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"


def test_the_answer_carries_the_model_and_trace_that_produced_it(settings: Settings) -> None:
    """A caller debugging a bad answer cannot see either from its own side."""
    with TestClient(build_app(settings, executor=_executor(settings, StubAgent()))) as client:
        result = _send(client, "hi")

    task = result.get("task", result)
    metadata = task["artifacts"][0]["metadata"]
    assert metadata["served_by_model"] == "backup"


def test_the_a2a_context_becomes_the_conversation_thread(settings: Settings) -> None:
    """Two messages on one context must reach the agent as one conversation."""
    agent = StubAgent()
    with TestClient(build_app(settings, executor=_executor(settings, agent))) as client:
        first = _send(client, "my name is Ada")
        context_id = first.get("task", first)["contextId"]
        _send(client, "what is my name?", context_id=context_id)

    threads = [config["configurable"]["thread_id"] for config in agent.configs]
    assert threads == [context_id, context_id]


def test_an_empty_message_is_rejected_rather_than_failed(settings: Settings) -> None:
    """Nothing went wrong; there was just nothing to answer."""
    with TestClient(build_app(settings, executor=_executor(settings, StubAgent()))) as client:
        result = _send(client, "   ")

    task = result.get("task", result)
    assert task["status"]["state"] == "TASK_STATE_REJECTED"


def test_a_failing_run_reports_a_failed_task_not_a_dead_connection(settings: Settings) -> None:
    with TestClient(build_app(settings, executor=_executor(settings, ExplodingAgent()))) as client:
        result = _send(client, "hi")

    task = result.get("task", result)
    assert task["status"]["state"] == "TASK_STATE_FAILED"
    assert "provider is down" in str(task["status"])


def test_a_v0_3_client_is_still_understood(settings: Settings) -> None:
    """Why the compatibility routes are mounted.

    A client written against A2A 0.3 sends no version header, dotted-path
    method names and ``kind``-tagged parts. Refusing those would mean every
    caller has to upgrade in step with this server.
    """
    with TestClient(build_app(settings, executor=_executor(settings, StubAgent()))) as client:
        response = client.post(
            JSONRPC_PATH,
            json={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "message/send",
                "params": {
                    "message": {
                        "messageId": uuid.uuid4().hex,
                        "role": "user",
                        "parts": [{"kind": "text", "text": "What is 17 * 23?"}],
                    }
                },
            },
        )

    body = response.json()
    assert "error" not in body, body
    assert "391" in str(body["result"])


async def test_cancelling_a_task_stops_the_run_in_flight(settings: Settings) -> None:
    """Cancellation has to reach the model call, not just the bookkeeping.

    A task nobody is waiting for should stop costing tokens the moment the
    caller says so.
    """
    import asyncio

    from a2a.types import TaskState

    agent = SlowAgent()
    executor = _executor(settings, agent)
    queue = RecordingQueue()
    context = StubRequestContext("task-1", "context-1", "hi")

    running = asyncio.create_task(executor.execute(context, queue))
    await asyncio.wait_for(agent.started.wait(), timeout=5)

    await executor.cancel(context, queue)

    with pytest.raises(asyncio.CancelledError):
        await running

    assert agent.cancelled, "the agent run kept going after the task was cancelled"
    states = [event.status.state for event in queue.events if hasattr(event, "status")]
    assert TaskState.TASK_STATE_CANCELED in states
