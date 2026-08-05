"""Tests for MCP server onboarding.

The normalisation half runs entirely offline. The loading half uses a real MCP
server over stdio — a tiny one written for the test — because the thing worth
proving is that an arbitrary server's tools arrive as working LangChain tools,
and a mocked client would prove only that the mock was called.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from general_agent.config import Settings
from general_agent.mcp import (
    MCPConfigError,
    load_mcp_connections,
    load_mcp_tools,
    load_mcp_tools_blocking,
)

mcp_server = pytest.importorskip("mcp.server.fastmcp", reason="needs the mcp extra")

#: A minimal MCP server, run as a subprocess over stdio by the tests below.
SERVER_SOURCE = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Demo")


@mcp.tool()
def shout(text: str) -> str:
    """Return the text in capitals."""
    return text.upper()


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


def _settings(config: dict, tmp_path: Path, **env: str) -> Settings:
    """Settings pointed at a config file written from ``config``."""
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps(config))
    return Settings.from_env({"AGENT_MCP_CONFIG": str(path), **env})


# --- Configuration -----------------------------------------------------


def test_no_configuration_means_no_servers(tmp_path: Path, monkeypatch) -> None:
    """An absent ./mcp.json is a valid state, not an error."""
    monkeypatch.chdir(tmp_path)
    assert load_mcp_connections(Settings.from_env({})) == {}


def test_stdio_server_is_inferred_from_a_command(tmp_path: Path) -> None:
    settings = _settings(
        {"mcpServers": {"fs": {"command": "npx", "args": ["-y", "server-filesystem"]}}},
        tmp_path,
    )
    assert load_mcp_connections(settings) == {
        "fs": {"transport": "stdio", "command": "npx", "args": ["-y", "server-filesystem"]}
    }


def test_http_server_is_inferred_from_a_url(tmp_path: Path) -> None:
    settings = _settings({"mcpServers": {"api": {"url": "https://x/mcp"}}}, tmp_path)
    assert load_mcp_connections(settings)["api"]["transport"] == "streamable_http"


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("http", "streamable_http"),
        ("streamable-http", "streamable_http"),
        ("sse", "sse"),
        ("stdio", "stdio"),
    ],
)
def test_ecosystem_transport_names_are_accepted(
    declared: str, expected: str, tmp_path: Path
) -> None:
    """The wider MCP ecosystem writes 'http'; the adapter says 'streamable_http'."""
    entry = {"type": declared, "url": "https://x/mcp", "command": "x"}
    settings = _settings({"mcpServers": {"s": entry}}, tmp_path)
    assert load_mcp_connections(settings)["s"]["transport"] == expected


def test_environment_variables_are_expanded_in_headers(tmp_path: Path) -> None:
    """So a config file with a token in it never has to be written down."""
    settings = _settings(
        {
            "mcpServers": {
                "api": {
                    "url": "https://x/mcp",
                    "headers": {"Authorization": "Bearer ${API_TOKEN}"},
                }
            }
        },
        tmp_path,
        API_TOKEN="secret-value",
    )
    connection = load_mcp_connections(settings)["api"]
    assert connection["headers"] == {"Authorization": "Bearer secret-value"}


def test_an_unset_variable_is_an_error_not_an_empty_string(tmp_path: Path) -> None:
    """'Bearer ' would fail as a 401 three layers from the actual cause."""
    settings = _settings(
        {"mcpServers": {"api": {"url": "https://x/mcp", "headers": {"A": "${NOPE}"}}}},
        tmp_path,
        AGENT_MCP_STRICT="true",
    )
    with pytest.raises(MCPConfigError, match="NOPE"):
        load_mcp_connections(settings)


def test_a_bad_server_is_skipped_but_the_others_load(tmp_path: Path) -> None:
    settings = _settings(
        {
            "mcpServers": {
                "broken": {"description": "no command or url"},
                "ok": {"url": "https://x"},
            }
        },
        tmp_path,
    )
    assert list(load_mcp_connections(settings)) == ["ok"]


def test_strict_mode_refuses_to_start_without_a_configured_server(tmp_path: Path) -> None:
    settings = _settings(
        {"mcpServers": {"broken": {"description": "no command or url"}}},
        tmp_path,
        AGENT_MCP_STRICT="true",
    )
    with pytest.raises(MCPConfigError):
        load_mcp_connections(settings)


def test_disabled_servers_stay_in_the_file_and_out_of_the_agent(tmp_path: Path) -> None:
    settings = _settings(
        {"mcpServers": {"off": {"url": "https://x", "disabled": True}, "on": {"url": "https://y"}}},
        tmp_path,
    )
    assert list(load_mcp_connections(settings)) == ["on"]


def test_unsupported_options_are_dropped_rather_than_passed_through(tmp_path: Path) -> None:
    """The adapter forwards these as kwargs, so a stray key is a late TypeError."""
    settings = _settings(
        {"mcpServers": {"api": {"url": "https://x/mcp", "autoApprove": ["everything"]}}},
        tmp_path,
    )
    assert "autoApprove" not in load_mcp_connections(settings)["api"]


def test_inline_servers_override_the_config_file(tmp_path: Path) -> None:
    """AGENT_MCP_SERVERS is the per-deployment setting, so it wins."""
    settings = _settings(
        {"mcpServers": {"api": {"url": "https://from-file/mcp"}}},
        tmp_path,
        AGENT_MCP_SERVERS=json.dumps({"mcpServers": {"api": {"url": "https://from-env/mcp"}}}),
    )
    assert load_mcp_connections(settings)["api"]["url"] == "https://from-env/mcp"


def test_a_configured_path_that_does_not_exist_is_an_error() -> None:
    settings = Settings.from_env({"AGENT_MCP_CONFIG": "/nonexistent/mcp.json"})
    with pytest.raises(MCPConfigError, match="does not exist"):
        load_mcp_connections(settings)


def test_malformed_json_says_so(tmp_path: Path) -> None:
    path = tmp_path / "mcp.json"
    path.write_text("{not json")
    settings = Settings.from_env({"AGENT_MCP_CONFIG": str(path)})
    with pytest.raises(MCPConfigError, match="invalid JSON"):
        load_mcp_connections(settings)


# --- Loading from a real server ----------------------------------------


@pytest.fixture
def demo_server(tmp_path: Path) -> Settings:
    """Settings onboarding a real stdio MCP server run as a subprocess."""
    server = tmp_path / "demo_server.py"
    server.write_text(SERVER_SOURCE)
    return Settings.from_env(
        {
            "AGENT_MCP_SERVERS": json.dumps(
                {"mcpServers": {"demo": {"command": sys.executable, "args": [str(server)]}}}
            ),
            "AGENT_MCP_STRICT": "true",
        }
    )


async def test_tools_are_onboarded_from_a_real_server(demo_server: Settings) -> None:
    tools = await load_mcp_tools(demo_server)

    assert [tool.name for tool in tools] == ["demo_shout"]
    assert "capitals" in tools[0].description


async def test_an_onboarded_tool_actually_runs(demo_server: Settings) -> None:
    """The point of the whole feature: the model calls it and gets an answer."""
    (shout,) = await load_mcp_tools(demo_server)

    result = await shout.ainvoke({"text": "hello"})

    # MCP returns typed content blocks rather than a bare string, and the
    # adapter passes them through as LangChain content.
    assert [block["text"] for block in result] == ["HELLO"]


async def test_onboarded_tools_are_async_only(demo_server: Settings) -> None:
    """Why the runner drives the agent with ainvoke, pinned as a fact.

    The adapter builds these with a coroutine and no sync implementation. If
    that ever changes the sync path becomes viable again — but until then, a
    sync ``invoke`` anywhere in the run path breaks every MCP tool.
    """
    (shout,) = await load_mcp_tools(demo_server)

    with pytest.raises(NotImplementedError):
        shout.invoke({"text": "hello"})


def test_unprefixed_tools_keep_their_own_names(tmp_path: Path) -> None:
    server = tmp_path / "demo_server.py"
    server.write_text(SERVER_SOURCE)
    settings = Settings.from_env(
        {
            "AGENT_MCP_SERVERS": json.dumps(
                {"mcpServers": {"demo": {"command": sys.executable, "args": [str(server)]}}}
            ),
            "AGENT_MCP_TOOL_PREFIX": "false",
        }
    )
    assert [tool.name for tool in load_mcp_tools_blocking(settings)] == ["shout"]


async def test_an_unreachable_server_costs_its_tools_not_the_agent() -> None:
    """Default behaviour: degrade, don't fail."""
    settings = Settings.from_env(
        {
            "AGENT_MCP_SERVERS": json.dumps(
                {"mcpServers": {"gone": {"command": "definitely-not-a-real-command", "args": []}}}
            ),
        }
    )
    assert await load_mcp_tools(settings) == []


async def test_strict_mode_surfaces_an_unreachable_server() -> None:
    settings = Settings.from_env(
        {
            "AGENT_MCP_SERVERS": json.dumps(
                {"mcpServers": {"gone": {"command": "definitely-not-a-real-command", "args": []}}}
            ),
            "AGENT_MCP_STRICT": "true",
        }
    )
    with pytest.raises(Exception, match=r".*"):
        await load_mcp_tools(settings)


def test_blocking_loader_works_from_inside_an_event_loop(demo_server: Settings) -> None:
    """build_agent is synchronous and may be called from either world."""
    import asyncio

    async def inside() -> list:
        return load_mcp_tools_blocking(demo_server)

    assert [tool.name for tool in asyncio.run(inside())] == ["demo_shout"]


# --- End to end --------------------------------------------------------


class ToolCallingModel(BaseChatModel):
    """Calls one named tool, then answers. Stands in for a real model."""

    tool_name: str = ""
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def _generate(self, messages: list, stop: object = None, **kwargs: object) -> ChatResult:
        self.calls += 1
        if self.calls == 1:
            message = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": {"text": "hello"}, "id": "call-1"}],
            )
        else:
            message = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages: list, stop: object = None, **kwargs: object) -> ChatResult:
        return self._generate(messages, stop, **kwargs)

    def bind_tools(self, tools: object, **kwargs: object) -> object:
        return self.bind(tools=tools, **kwargs)


async def test_the_agent_calls_a_tool_it_onboarded_from_an_mcp_server(
    demo_server: Settings,
) -> None:
    """The whole feature, end to end: a server's tool runs inside an agent turn."""
    from langchain.agents import create_agent

    tools = await load_mcp_tools(demo_server)
    agent = create_agent(model=ToolCallingModel(tool_name="demo_shout"), tools=tools)

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "shout hello"}]})

    tool_output = [message for message in result["messages"] if message.type == "tool"]
    assert tool_output, "the MCP tool was never reached"
    assert "HELLO" in str(tool_output[0].content)


def test_mcp_tools_join_the_built_in_ones(demo_server: Settings) -> None:
    from general_agent.agent import build_agent_tools

    names = [tool.name for tool in build_agent_tools(demo_server)]

    assert "calculator" in names
    assert "demo_shout" in names


def test_a_colliding_mcp_tool_does_not_shadow_a_built_in(tmp_path: Path) -> None:
    """One name, two tools is not resolvable — the known-good one keeps it."""
    server = tmp_path / "clash.py"
    server.write_text(SERVER_SOURCE.replace("def shout(", "def calculator("))
    settings = Settings.from_env(
        {
            "AGENT_MCP_SERVERS": json.dumps(
                {"mcpServers": {"x": {"command": sys.executable, "args": [str(server)]}}}
            ),
            "AGENT_MCP_TOOL_PREFIX": "false",
        }
    )

    from general_agent.agent import build_agent_tools
    from general_agent.tools import calculator

    tools = build_agent_tools(settings)
    assert [tool for tool in tools if tool.name == "calculator"] == [calculator]
