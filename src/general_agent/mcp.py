"""Onboarding MCP servers as agent tools.

Any [MCP](https://modelcontextprotocol.io) server can be pointed at this agent
and its tools become the agent's tools, alongside the built-in ones. Nothing
in this module knows what a given server does — that is the point.

Configuration is the ``mcpServers`` JSON that Claude Desktop, VS Code and Cursor
already use, so a server someone has running elsewhere can be onboarded by
copying its stanza across:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"]
    },
    "internal-api": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": {"Authorization": "Bearer ${INTERNAL_API_TOKEN}"}
    }
  }
}
```

``${VAR}`` is expanded from the environment, so tokens live in ``.env`` and the
config file stays committable.

Two things about MCP tools are worth knowing before wiring them in:

* **They are async-only.** ``langchain-mcp-adapters`` builds them with a
  coroutine and no sync implementation, so an agent holding one must be driven
  with ``ainvoke``. :class:`~general_agent.runner.AgentRunner` does.
* **A session is opened per tool call.** That is the adapter's stateless
  default, and it is the right one here: a long-lived stdio subprocess owned by
  a CLI process that might sit idle for an hour is a worse trade than paying
  startup cost per call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

    from general_agent.config import Settings

__all__ = [
    "DEFAULT_CONFIG_FILENAME",
    "MCPConfigError",
    "load_mcp_connections",
    "load_mcp_tools",
    "load_mcp_tools_blocking",
]

logger = logging.getLogger(__name__)

#: Looked for in the working directory when no path is configured. Absent is
#: fine — it just means no MCP servers, not a broken configuration.
DEFAULT_CONFIG_FILENAME = "mcp.json"

#: ``${VAR}`` references inside string values.
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

#: Keys accepted per transport, and the transport each config key implies.
#: An allowlist rather than a passthrough: the adapter hands these straight to
#: the session factory as keyword arguments, so a stray key becomes a
#: ``TypeError`` at the first tool call rather than at startup.
_ALLOWED_KEYS: Mapping[str, frozenset[str]] = {
    "stdio": frozenset({"command", "args", "env", "cwd", "encoding", "encoding_error_handler"}),
    "streamable_http": frozenset({"url", "headers", "timeout", "sse_read_timeout"}),
    "sse": frozenset({"url", "headers", "timeout", "sse_read_timeout"}),
    "websocket": frozenset({"url"}),
}

#: What the wider MCP ecosystem writes in ``type``/``transport``, mapped onto
#: the adapter's names.
_TRANSPORT_ALIASES: Mapping[str, str] = {
    "stdio": "stdio",
    "http": "streamable_http",
    "streamable-http": "streamable_http",
    "streamable_http": "streamable_http",
    "sse": "sse",
    "ws": "websocket",
    "websocket": "websocket",
}


class MCPConfigError(ValueError):
    """A server entry could not be turned into a usable connection."""


def _expand(value: Any, environ: Mapping[str, str], *, where: str) -> Any:
    """Recursively expand ``${VAR}`` references in strings.

    Raises:
        MCPConfigError: If a referenced variable is unset. Substituting an
            empty string instead would produce an ``Authorization: Bearer``
            header and a 401 three layers away from the cause.
    """
    if isinstance(value, str):
        missing: list[str] = []

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = environ.get(name)
            if resolved is None:
                missing.append(name)
                return ""
            return resolved

        expanded = _PLACEHOLDER.sub(replace, value)
        if missing:
            names = ", ".join(sorted(set(missing)))
            msg = f"{where} references unset environment variable(s): {names}"
            raise MCPConfigError(msg)
        return expanded
    if isinstance(value, list):
        return [_expand(item, environ, where=where) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, environ, where=where) for key, item in value.items()}
    return value


def _transport_for(name: str, entry: Mapping[str, Any]) -> str:
    """Decide which transport a server entry describes.

    ``type`` is what the ecosystem's config files use; ``transport`` is what
    ``langchain-mcp-adapters`` calls the same thing. Both are accepted, and
    either can be omitted — a ``command`` means stdio, a ``url`` means HTTP.
    """
    declared = entry.get("type") or entry.get("transport")
    if declared:
        transport = _TRANSPORT_ALIASES.get(str(declared).strip().lower())
        if transport is None:
            supported = ", ".join(sorted(set(_TRANSPORT_ALIASES)))
            msg = (
                f"MCP server {name!r}: unknown transport {declared!r} (expected one of {supported})"
            )
            raise MCPConfigError(msg)
        return transport
    if entry.get("command"):
        return "stdio"
    if entry.get("url"):
        return "streamable_http"
    msg = f"MCP server {name!r}: needs a 'command' (stdio) or a 'url' (http/sse), and has neither"
    raise MCPConfigError(msg)


def _connection_for(
    name: str,
    entry: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    """Turn one config entry into an adapter connection dict."""
    transport = _transport_for(name, entry)
    allowed = _ALLOWED_KEYS[transport]

    # 'type', 'transport' and 'disabled' are ours, not the adapter's; anything
    # else unrecognised is far more likely a typo than a feature, so say so
    # rather than dropping it silently.
    ignored = set(entry) - allowed - {"type", "transport", "disabled"}
    if ignored:
        logger.warning(
            "MCP server %r: ignoring unsupported option(s) %s for %s transport",
            name,
            ", ".join(sorted(ignored)),
            transport,
        )

    connection: dict[str, Any] = {"transport": transport}
    for key in allowed & set(entry):
        connection[key] = _expand(entry[key], environ, where=f"MCP server {name!r} ({key})")

    if transport == "stdio":
        # Required by the adapter even when there are none.
        connection.setdefault("args", [])
    elif not connection.get("url"):
        msg = f"MCP server {name!r}: {transport} transport needs a 'url'"
        raise MCPConfigError(msg)

    return connection


def _entries_from(raw: Any, *, source: str) -> dict[str, Any]:
    """Pull the server mapping out of a parsed config document."""
    if not isinstance(raw, dict):
        msg = f"{source}: expected a JSON object"
        raise MCPConfigError(msg)
    # "mcpServers" is the ecosystem's key; "servers" is VS Code's; a bare
    # mapping of names to entries is what people write by hand.
    for key in ("mcpServers", "servers"):
        if key in raw:
            entries = raw[key]
            break
    else:
        entries = raw
    if not isinstance(entries, dict):
        msg = f"{source}: 'mcpServers' must be an object mapping names to server definitions"
        raise MCPConfigError(msg)
    return entries


def _raw_entries(settings: Settings) -> dict[str, Any]:
    """Collect server entries from the config file and the inline env var."""
    entries: dict[str, Any] = {}

    path = Path(settings.mcp_config_path) if settings.mcp_config_path else None
    if path is None:
        default = Path(DEFAULT_CONFIG_FILENAME)
        path = default if default.is_file() else None
    if path is not None:
        if not path.is_file():
            # An explicitly configured path that isn't there is a mistake worth
            # reporting; the implicit ./mcp.json simply not existing is not.
            msg = f"AGENT_MCP_CONFIG points at {path}, which does not exist"
            raise MCPConfigError(msg)
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            msg = f"{path}: invalid JSON ({exc})"
            raise MCPConfigError(msg) from exc
        entries.update(_entries_from(document, source=str(path)))

    if settings.mcp_servers_json:
        try:
            document = json.loads(settings.mcp_servers_json)
        except json.JSONDecodeError as exc:
            msg = f"AGENT_MCP_SERVERS: invalid JSON ({exc})"
            raise MCPConfigError(msg) from exc
        # Inline wins: it is the more specific, per-deployment setting.
        entries.update(_entries_from(document, source="AGENT_MCP_SERVERS"))

    return entries


def load_mcp_connections(settings: Settings) -> dict[str, dict[str, Any]]:
    """Resolve the configured MCP servers into adapter connection dicts.

    Pure: reads the config file and the environment, talks to no server. That
    makes the whole normalisation path — transports, env expansion, unknown
    keys — testable without a single subprocess or socket.

    Args:
        settings: Configuration, for the config path, inline servers,
            ``mcp_strict`` and the environment to expand ``${VAR}`` from.

    Returns:
        Server name to connection dict, ready for ``MultiServerMCPClient``.
        Empty when nothing is configured.

    Raises:
        MCPConfigError: On a malformed config, or on a bad server entry when
            ``mcp_strict`` is set. Otherwise a bad entry is logged and skipped.
    """
    entries = _raw_entries(settings)
    connections: dict[str, dict[str, Any]] = {}

    for name, entry in entries.items():
        if not isinstance(entry, dict):
            msg = f"MCP server {name!r}: expected an object, got {type(entry).__name__}"
            if settings.mcp_strict:
                raise MCPConfigError(msg)
            logger.warning("%s — skipping", msg)
            continue
        if entry.get("disabled"):
            logger.info("MCP server %r is disabled; skipping", name)
            continue
        try:
            connections[name] = _connection_for(name, entry, settings.environ)
        except MCPConfigError:
            if settings.mcp_strict:
                raise
            logger.warning("Skipping MCP server %r", name, exc_info=True)

    return connections


async def load_mcp_tools(settings: Settings) -> list[BaseTool]:
    """Connect to every configured MCP server and return its tools.

    Servers are contacted one at a time and failures are contained: an
    unreachable server costs you its tools and a warning, not the agent. Set
    ``AGENT_MCP_STRICT=true`` when a missing server should instead stop
    startup — which is the right setting for a deployment whose whole job
    depends on one of them.

    Args:
        settings: Configuration.

    Returns:
        The MCP tools, ready to hand to ``create_agent``. Empty when no server
        is configured, so this is always safe to call.

    Raises:
        MCPConfigError: Only under ``mcp_strict``; see :func:`load_mcp_connections`.
        Exception: Only under ``mcp_strict``, propagated from the failing server.
    """
    connections = load_mcp_connections(settings)
    if not connections:
        return []

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        # Not degraded to a warning even outside strict mode: servers are
        # configured, so this is a missing install, not a server having a bad
        # day, and quietly running without them would be the wrong answer.
        msg = (
            f"MCP servers are configured ({', '.join(connections)}) but the MCP "
            'extra is not installed. Run: pip install "general-agent[mcp]"'
        )
        raise MCPConfigError(msg) from exc

    client = MultiServerMCPClient(connections, tool_name_prefix=settings.mcp_tool_prefix)

    tools: list[BaseTool] = []
    for name in connections:
        try:
            # Per server rather than one gathered call, so one bad server can
            # be reported and skipped by name instead of failing the batch.
            # The timeout is the real defence: a stdio server that starts but
            # never answers ``initialize`` would otherwise hang startup for good.
            server_tools = await asyncio.wait_for(
                client.get_tools(server_name=name),
                timeout=settings.mcp_startup_timeout,
            )
        except Exception:
            if settings.mcp_strict:
                raise
            logger.warning(
                "MCP server %r is unavailable; continuing without it", name, exc_info=True
            )
            continue

        logger.info(
            "Onboarded MCP server %r: %d tool(s) — %s",
            name,
            len(server_tools),
            ", ".join(tool.name for tool in server_tools) or "none",
        )
        tools.extend(server_tools)

    return tools


def load_mcp_tools_blocking(settings: Settings) -> list[BaseTool]:
    """Synchronous :func:`load_mcp_tools`, usable from inside a running loop.

    The work runs on a worker thread with its own event loop, because the
    callers that need this — building an agent from synchronous code — may or
    may not already be inside one, and ``asyncio.run`` only works when they are
    not. The tools themselves are unaffected: each opens its own session in
    whichever loop later calls it.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mcp-load") as pool:
        return pool.submit(lambda: asyncio.run(load_mcp_tools(settings))).result()
