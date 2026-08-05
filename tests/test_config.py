"""Tests for environment-driven configuration."""

from __future__ import annotations

from general_agent.config import DEFAULT_MODEL_CHAIN, Settings


def test_defaults_when_environment_is_empty() -> None:
    settings = Settings.from_env({})
    assert settings.models == DEFAULT_MODEL_CHAIN
    assert settings.primary_model == DEFAULT_MODEL_CHAIN[0]
    assert settings.environment == "development"
    assert settings.cooldown_scope == "route"


def test_model_chain_is_read_from_environment() -> None:
    settings = Settings.from_env({"AGENT_MODELS": "google:gemini-flash-latest, openrouter:a/b"})
    assert settings.models == ("google:gemini-flash-latest", "openrouter:a/b")


def test_blank_model_chain_falls_back_to_default() -> None:
    assert Settings.from_env({"AGENT_MODELS": "  ,  "}).models == DEFAULT_MODEL_CHAIN


def test_tracing_requires_both_langfuse_keys() -> None:
    """A half-configured SDK would authenticate with nothing and drop spans."""
    assert not Settings.from_env({"LANGFUSE_PUBLIC_KEY": "pk-lf-x"}).tracing_enabled
    assert not Settings.from_env({"LANGFUSE_SECRET_KEY": "sk-lf-x"}).tracing_enabled
    assert Settings.from_env(
        {"LANGFUSE_PUBLIC_KEY": "pk-lf-x", "LANGFUSE_SECRET_KEY": "sk-lf-x"}
    ).tracing_enabled


def test_tracing_can_be_disabled_with_credentials_present() -> None:
    settings = Settings.from_env(
        {
            "LANGFUSE_PUBLIC_KEY": "pk-lf-x",
            "LANGFUSE_SECRET_KEY": "sk-lf-x",
            "AGENT_TRACING_ENABLED": "false",
        }
    )
    assert not settings.tracing_enabled


def test_account_cooldown_flag_sets_scope() -> None:
    settings = Settings.from_env({"AGENT_ACCOUNT_COOLDOWN": "true"})
    assert settings.cooldown_scope == "account"
    assert settings.account_scoped_cooldown


def test_tags_are_merged_and_deduplicated() -> None:
    settings = Settings.from_env({"AGENT_TAGS": "prod,support"})
    assert settings.resolved_tags(["support", "cli"]) == ["prod", "support", "cli"]


def test_invalid_numbers_fall_back_to_defaults() -> None:
    settings = Settings.from_env({"AGENT_COOLDOWN_SECONDS": "not-a-number"})
    assert settings.cooldown_seconds == 60.0
