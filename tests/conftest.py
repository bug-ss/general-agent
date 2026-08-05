"""Shared fakes. The suite runs fully offline — no provider, no Langfuse."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from general_agent.config import Settings


class FakeRateLimitError(Exception):
    """Stands in for ``openai.RateLimitError`` — duck-typed on ``status_code``."""

    def __init__(self, message: str = "429 Too Many Requests", status_code: int = 429) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeChatModel(BaseChatModel):
    """Returns a canned reply, or raises a canned error, and counts calls."""

    name: str = "fake"
    error: Exception | None = None
    reply: str = "ok"
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:  # type: ignore[override]
        return self.name

    def _generate(self, messages: list[Any], stop: Any = None, **kwargs: Any) -> ChatResult:
        self.calls += 1
        if self.error is not None:
            raise self.error
        message = AIMessage(content=self.reply, response_metadata={"model_name": self.name})
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages: list[Any], stop: Any = None, **kwargs: Any) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        # Deliberately the same shape as a real provider's: BaseChatModel.bind
        # returns a binding around the *raw* model with an empty config, which
        # is exactly what drops a run name applied further out.
        return self.bind(tools=tools, **kwargs)


@pytest.fixture
def settings() -> Settings:
    """Settings with no credentials anywhere, so nothing reaches a network."""
    return Settings.from_env({})
