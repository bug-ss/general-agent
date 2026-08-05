"""Environment-driven configuration for the agent.

Everything the agent needs to start is read from the environment once, here,
so the rest of the package never touches ``os.environ`` directly and tests can
build a settings object by hand.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["DEFAULT_MODEL_CHAIN", "DEFAULT_SYSTEM_PROMPT", "Settings"]

#: Failover chain used when ``AGENT_MODELS`` is unset. Ordered best-first: the
#: middleware walks it top to bottom whenever a provider answers HTTP 429.
#: Two OpenRouter models then a Google one — crossing providers matters,
#: because a rate limit is usually metered per account, not per model.
DEFAULT_MODEL_CHAIN: tuple[str, ...] = (
    "openrouter:openai/gpt-4o-mini",
    "openrouter:meta-llama/llama-3.3-70b-instruct",
    "google:gemini-flash-latest",
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful general-purpose assistant. Use the tools available to "
    "you when they can answer a question more reliably than you can from "
    "memory — arithmetic, the current date and time, and web search. Say when "
    "you are unsure rather than guessing, and keep answers concise."
)


def _split(value: str) -> list[str]:
    """Split a comma- or newline-separated env value, dropping blanks."""
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _flag(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _number(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Resolved configuration for one agent process."""

    # --- Agent ---
    models: tuple[str, ...] = DEFAULT_MODEL_CHAIN
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    agent_name: str = "general-agent"

    # --- Rate-limit failover middleware ---
    cooldown_seconds: float = 60.0
    cooldown_scope: str = "route"
    respect_retry_after: bool = False
    #: Cool down every route sharing a credential rather than just the model
    #: that failed. Set when your limits are account-wide (OpenRouter's free
    #: tier is the classic case).
    account_scoped_cooldown: bool = False
    #: Extra env vars holding additional OpenRouter credentials to pool over.
    openrouter_key_vars: tuple[str, ...] = ("OPENROUTER_API_KEY",)
    pool_openrouter_accounts: bool = False

    # --- Observability ---
    tracing_enabled: bool = True
    environment: str = "development"
    release: str | None = None
    tags: tuple[str, ...] = ()
    user_id: str = "anonymous"
    #: Fraction of traces to send. 1.0 = everything; lower it under load.
    sample_rate: float = 1.0

    # --- Tools ---
    enable_web_search: bool = True
    max_search_results: int = 5

    # --- MCP ---
    #: Path to an ``mcpServers`` JSON file. ``None`` means "look for ./mcp.json
    #: and use it if it exists" — an absent file is not an error.
    mcp_config_path: str | None = None
    #: Inline server definitions, same shape as the file. For containers and CI,
    #: where mounting a file to add one server is more ceremony than it's worth.
    mcp_servers_json: str | None = None
    #: Fail startup when a configured MCP server cannot be reached. Off by
    #: default: one unreachable server should cost you its tools, not the agent.
    mcp_strict: bool = False
    #: Prefix MCP tool names with their server name (``github_create_issue``).
    #: On by default — two servers exposing ``search`` is entirely normal, and
    #: an unprefixed collision silently shadows one of them.
    mcp_tool_prefix: bool = True
    #: How long to wait for one server to hand over its tool list.
    mcp_startup_timeout: float = 30.0

    # --- A2A server ---
    a2a_host: str = "127.0.0.1"
    a2a_port: int = 8080
    #: Externally reachable base URL to advertise in the agent card. Set this
    #: behind a proxy or in a container, where the bind address is not the
    #: address other agents can actually reach.
    a2a_url: str | None = None

    _environ: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        """Build settings from the process environment.

        Call this *after* ``load_dotenv()`` — a ``.env`` loaded later cannot
        retroactively change what was already read.
        """
        env = os.environ if environ is None else environ

        raw_models = env.get("AGENT_MODELS", "")
        models = tuple(_split(raw_models)) or DEFAULT_MODEL_CHAIN

        # Tracing needs both keys; without them the SDK would initialise with
        # nothing and drop every span silently, which looks like a bug.
        has_credentials = bool(env.get("LANGFUSE_PUBLIC_KEY") and env.get("LANGFUSE_SECRET_KEY"))

        return cls(
            models=models,
            system_prompt=env.get("AGENT_SYSTEM_PROMPT") or DEFAULT_SYSTEM_PROMPT,
            agent_name=env.get("AGENT_NAME") or "general-agent",
            cooldown_seconds=_number(env, "AGENT_COOLDOWN_SECONDS", 60.0),
            cooldown_scope="account" if _flag(env, "AGENT_ACCOUNT_COOLDOWN", False) else "route",
            respect_retry_after=_flag(env, "AGENT_RESPECT_RETRY_AFTER", False),
            account_scoped_cooldown=_flag(env, "AGENT_ACCOUNT_COOLDOWN", False),
            openrouter_key_vars=tuple(_split(env.get("AGENT_OPENROUTER_KEY_VARS", "")))
            or ("OPENROUTER_API_KEY",),
            pool_openrouter_accounts=_flag(env, "AGENT_POOL_OPENROUTER_ACCOUNTS", False),
            tracing_enabled=has_credentials and _flag(env, "AGENT_TRACING_ENABLED", True),
            environment=env.get("APP_ENV") or "development",
            release=env.get("APP_RELEASE") or None,
            tags=tuple(_split(env.get("AGENT_TAGS", ""))),
            user_id=env.get("USER_ID") or "anonymous",
            sample_rate=_number(env, "LANGFUSE_SAMPLE_RATE", 1.0),
            enable_web_search=_flag(env, "AGENT_ENABLE_WEB_SEARCH", True),
            max_search_results=int(_number(env, "AGENT_MAX_SEARCH_RESULTS", 5)),
            mcp_config_path=env.get("AGENT_MCP_CONFIG") or None,
            mcp_servers_json=env.get("AGENT_MCP_SERVERS") or None,
            mcp_strict=_flag(env, "AGENT_MCP_STRICT", False),
            mcp_tool_prefix=_flag(env, "AGENT_MCP_TOOL_PREFIX", True),
            mcp_startup_timeout=_number(env, "AGENT_MCP_STARTUP_TIMEOUT", 30.0),
            a2a_host=env.get("AGENT_A2A_HOST") or "127.0.0.1",
            a2a_port=int(_number(env, "AGENT_A2A_PORT", 8080)),
            a2a_url=env.get("AGENT_A2A_URL") or None,
            _environ=env,
        )

    @property
    def environ(self) -> Mapping[str, str]:
        """The environment these settings were read from."""
        return self._environ or os.environ

    @property
    def primary_model(self) -> str:
        """First entry of the failover chain — the agent's own model."""
        return self.models[0]

    def resolved_tags(self, extra: Sequence[str] = ()) -> list[str]:
        """Trace tags: configured tags plus per-run additions, de-duplicated."""
        seen: dict[str, None] = {}
        for tag in (*self.tags, *extra):
            if tag:
                seen.setdefault(tag, None)
        return list(seen)
