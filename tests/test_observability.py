"""Tests for the tracing backend selection and its failure modes."""

from __future__ import annotations

from typing import Any

import pytest

from general_agent.config import Settings
from general_agent.observability import (
    NullObservability,
    Observability,
    build_observability,
)

_CREDENTIALS = {"LANGFUSE_PUBLIC_KEY": "pk-lf-test", "LANGFUSE_SECRET_KEY": "sk-lf-test"}


def test_returns_null_backend_without_credentials() -> None:
    backend = build_observability(Settings.from_env({}))
    assert isinstance(backend, NullObservability)
    assert not backend.enabled
    assert backend.callbacks == []


def test_null_backend_run_yields_a_usable_span() -> None:
    """The agent path must be identical whether or not tracing is on."""
    with NullObservability().run("answer-question", input="hi") as span:
        span.update(output="there")
        assert span.trace_id is None


def test_setup_failure_degrades_to_null_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken tracing setup must not take the agent down."""
    import ratelimit_fallback

    def explode(**kwargs: Any) -> Any:
        raise RuntimeError("langfuse is unreachable")

    monkeypatch.setattr(ratelimit_fallback, "configure_langfuse", explode)
    backend = build_observability(Settings.from_env(_CREDENTIALS))
    assert isinstance(backend, NullObservability)


def test_builds_a_real_backend_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    import ratelimit_fallback

    captured: dict[str, Any] = {}

    class FakeClient:
        def auth_check(self) -> bool:
            return True

    def fake_configure(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(ratelimit_fallback, "configure_langfuse", fake_configure)
    monkeypatch.setattr(ratelimit_fallback, "get_langfuse_handler", lambda: "handler")

    settings = Settings.from_env({**_CREDENTIALS, "APP_ENV": "production", "APP_RELEASE": "v1"})
    backend = build_observability(settings)

    assert isinstance(backend, Observability)
    assert backend.callbacks == ["handler"]
    # Environment and release keep test traces out of production dashboards
    # and tie a regression to a deploy.
    assert captured["environment"] == "production"
    assert captured["release"] == "v1"


def test_rejected_credentials_do_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    import ratelimit_fallback

    class RejectingClient:
        def auth_check(self) -> bool:
            return False

    monkeypatch.setattr(ratelimit_fallback, "configure_langfuse", lambda **k: RejectingClient())
    monkeypatch.setattr(ratelimit_fallback, "get_langfuse_handler", lambda: "handler")

    assert isinstance(build_observability(Settings.from_env(_CREDENTIALS)), Observability)


def test_scoring_swallows_backend_errors() -> None:
    """Feedback is best-effort; it must never surface as an agent failure."""

    class BrokenClient:
        def create_score(self, **kwargs: Any) -> None:
            raise RuntimeError("network down")

    backend = Observability(BrokenClient(), "handler", Settings.from_env({}))
    backend.score_thumbs("trace-1", positive=True)  # does not raise


def test_scoring_is_skipped_without_a_trace_id() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create_score(self, **kwargs: Any) -> None:
            self.calls.append(kwargs)

    client = Client()
    backend = Observability(client, "handler", Settings.from_env({}))
    backend.score_thumbs(None, positive=True)
    assert client.calls == []

    backend.score_thumbs("trace-1", positive=True, comment="good")
    assert client.calls[0]["name"] == "user-thumbs"
    assert client.calls[0]["value"] == 1
    assert client.calls[0]["data_type"] == "BOOLEAN"
