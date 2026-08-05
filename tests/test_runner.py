"""Tests for the runner and the tracing it attaches to each run."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

from general_agent.config import Settings
from general_agent.observability import NullObservability
from general_agent.runner import AgentRunner, _as_text, _served_by


class RecordingObservability(NullObservability):
    """A NullObservability that remembers what it was asked to record."""

    enabled = True

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []
        self.closed = False

    @property
    def callbacks(self) -> list[Any]:
        return ["handler"]

    @contextmanager
    def run(self, name: str, **kwargs: Any):
        self.runs.append({"name": name, **kwargs})
        updates = self.updates

        class _Span:
            trace_id = "trace-abc"

            def update(self, **fields: Any) -> None:
                updates.append(fields)

        yield _Span()

    def score_thumbs(self, trace_id: str | None, **kwargs: Any) -> None:
        self.scores.append({"trace_id": trace_id, **kwargs})

    def shutdown(self) -> None:
        self.closed = True


class StubAgent:
    """Stands in for the compiled graph."""

    def __init__(self, text: str = "391", model: str = "backup") -> None:
        self.text = text
        self.model = model
        self.configs: list[dict[str, Any]] = []

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        # Async because the runner is: tools onboarded from an MCP server have
        # no synchronous implementation, so the whole run path has to be async.
        self.configs.append(config)

        class _Message:
            content: str = self.text
            response_metadata: dict[str, Any] = {"model_name": self.model}  # noqa: RUF012

        return {"messages": [_Message()]}


def _runner(**kwargs: Any) -> tuple[AgentRunner, RecordingObservability, StubAgent]:
    observability = RecordingObservability()
    agent = StubAgent()
    runner = AgentRunner(
        Settings.from_env({}),
        observability=observability,
        agent=agent,
        **kwargs,
    )
    return runner, observability, agent


def test_ask_returns_answer_trace_and_serving_model() -> None:
    runner, _, _ = _runner()
    reply = runner.ask("What is 17 * 23?")

    assert reply.text == "391"
    assert reply.trace_id == "trace-abc"
    assert reply.model == "backup"
    assert reply.session_id == runner.session_id


def test_trace_input_is_the_question_not_the_state_dict() -> None:
    """The raw LangGraph state would be an unreadable blob in the UI."""
    runner, observability, _ = _runner()
    runner.ask("What is 17 * 23?")

    assert observability.runs[0]["input"] == "What is 17 * 23?"
    assert observability.runs[0]["name"] == "answer-question"


def test_trace_output_and_serving_model_are_recorded() -> None:
    runner, observability, _ = _runner()
    runner.ask("hi")

    assert observability.updates[0]["output"] == "391"
    assert observability.updates[0]["metadata"]["served_by_model"] == "backup"


def test_session_id_is_shared_by_the_trace_and_the_checkpointer() -> None:
    """One conversation: the Langfuse session and the memory thread agree."""
    runner, observability, agent = _runner(session_id="session-42")
    runner.ask("hi")

    assert observability.runs[0]["session_id"] == "session-42"
    assert agent.configs[0]["configurable"]["thread_id"] == "session-42"


def test_callbacks_are_attached_to_every_run() -> None:
    runner, _, agent = _runner()
    runner.ask("hi")
    assert agent.configs[0]["callbacks"] == ["handler"]


def test_tags_combine_configured_and_per_run_values() -> None:
    observability = RecordingObservability()
    runner = AgentRunner(
        Settings.from_env({"AGENT_TAGS": "prod"}),
        observability=observability,
        agent=StubAgent(),
    )
    runner.ask("hi", tags=["cli"])
    assert observability.runs[0]["tags"] == ["prod", "cli"]


def test_feedback_scores_the_reply_trace() -> None:
    runner, observability, _ = _runner()
    reply = runner.ask("hi")
    runner.feedback(reply, positive=False, comment="wrong answer")

    assert observability.scores == [
        {"trace_id": "trace-abc", "positive": False, "comment": "wrong answer"}
    ]


def test_context_manager_flushes_on_exit() -> None:
    observability = RecordingObservability()
    with AgentRunner(
        Settings.from_env({}), observability=observability, agent=StubAgent()
    ) as runner:
        runner.ask("hi")
    assert observability.closed


def test_runs_without_tracing_configured() -> None:
    """No Langfuse credentials must not change the agent's behaviour."""
    runner = AgentRunner(
        Settings.from_env({}), observability=NullObservability(), agent=StubAgent()
    )
    reply = runner.ask("hi")

    assert reply.text == "391"
    assert reply.trace_id is None
    assert not runner.tracing_enabled


async def test_aask_is_the_async_entry_point() -> None:
    """What the A2A server and any async caller uses."""
    runner, _, _ = _runner()
    reply = await runner.aask("hi")

    assert reply.text == "391"


async def test_ask_refuses_to_block_inside_a_running_loop() -> None:
    """Better a clear error than a deadlock or a nested-loop crash."""
    runner, _, _ = _runner()

    with pytest.raises(RuntimeError, match="aask"):
        runner.ask("hi")


def test_block_content_is_flattened_to_text() -> None:
    assert _as_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "ab"
    assert _as_text("plain") == "plain"


def test_serving_model_falls_back_across_metadata_keys() -> None:
    class _Message:
        response_metadata: dict[str, Any] = {"model": "gemini-flash-latest"}  # noqa: RUF012

    assert _served_by(_Message()) == "gemini-flash-latest"
    assert _served_by(object()) is None
