"""Command-line entry point.

Four modes:

    general-agent "What is 17 * 23?"     # one question, then exit
    general-agent                        # interactive chat
    general-agent --serve                # answer other agents over A2A
    general-agent --list-tools           # show the tool set, MCP included

Environment is loaded before anything else in :func:`main`. That ordering is
load-bearing: the Langfuse SDK reads credentials when its client is built, so a
``load_dotenv()`` that runs afterwards leaves it authenticated with nothing and
silently dropping every span.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

__all__ = ["main"]


def _load_environment() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv is optional; real env vars still work
        return
    load_dotenv()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="general-agent",
        description="A generic LangChain agent with 429 failover and Langfuse tracing.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Question to ask. Omit to start an interactive session.",
    )
    parser.add_argument("--session-id", help="Group these turns under an existing session.")
    parser.add_argument("--user-id", help="End user this run belongs to.")
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        dest="tags",
        help="Extra trace tag; repeatable.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")

    serving = parser.add_argument_group("A2A server")
    serving.add_argument(
        "--serve",
        action="store_true",
        help="Serve this agent over A2A so other agents can call it.",
    )
    serving.add_argument("--host", help="Bind address (default 127.0.0.1).")
    serving.add_argument("--port", type=int, help="Port to bind (default 8080).")
    serving.add_argument(
        "--public-url",
        help="Base URL to advertise in the agent card, when it differs from the bind address.",
    )

    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List the agent's tools, including any onboarded from MCP servers, and exit.",
    )
    return parser.parse_args(argv)


_HELP = """
Commands:
  :up [comment]     record thumbs-up feedback on the last answer
  :down [comment]   record thumbs-down feedback on the last answer
  :session          show the current session id
  :help             show this message
  :quit             exit
""".strip()


def _handle_command(runner, last, line: str) -> bool:
    """Handle a ``:command``. Returns False when the session should end."""
    command, _, argument = line[1:].partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command in {"quit", "exit", "q"}:
        return False
    if command == "help":
        print(_HELP)
    elif command == "session":
        print(f"session: {runner.session_id}")
    elif command in {"up", "down"}:
        if last is None:
            print("No answer to rate yet.")
        elif not runner.tracing_enabled:
            print("Tracing is off, so feedback has nowhere to go.")
        else:
            runner.feedback(last, positive=command == "up", comment=argument or None)
            print(f"Recorded thumbs-{command} on trace {last.trace_id}.")
    else:
        print(f"Unknown command {line!r}. Try :help")
    return True


def _interactive(runner) -> int:
    print(f"{runner.settings.agent_name} — session {runner.session_id}")
    print("Ask a question, or :help for commands. Ctrl-D to exit.\n")

    last = None
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not line:
            continue
        if line.startswith(":"):
            if not _handle_command(runner, last, line):
                return 0
            continue

        try:
            last = runner.ask(line, tags=["interactive"])
        except Exception as exc:
            # Keep the session alive: one failed turn should not end the chat,
            # and the traceback is in the logs (and the trace) either way.
            logging.getLogger(__name__).debug("Agent run failed", exc_info=True)
            print(f"error: {exc}\n")
            continue

        print(f"\n{last.text}\n")
        if last.model:
            print(f"  (served by {last.model})")
        if last.trace_id:
            print(f"  (trace {last.trace_id} — rate it with :up / :down)")
        print()


def _list_tools(settings) -> int:
    """Print the tool set, separating built-ins from onboarded MCP servers."""
    from general_agent.mcp import MCPConfigError, load_mcp_connections, load_mcp_tools_blocking
    from general_agent.tools import build_tools

    builtin = build_tools(
        enable_web_search=settings.enable_web_search,
        max_search_results=settings.max_search_results,
    )
    print("Built-in tools:")
    for tool in builtin:
        print(f"  {tool.name:<28} {(tool.description or '').splitlines()[0]}")

    try:
        connections = load_mcp_connections(settings)
    except MCPConfigError as exc:
        print(f"\nMCP configuration error: {exc}", file=sys.stderr)
        return 1

    if not connections:
        print("\nNo MCP servers configured (set AGENT_MCP_CONFIG or add ./mcp.json).")
        return 0

    print(f"\nMCP servers: {', '.join(connections)}")
    mcp_tools = load_mcp_tools_blocking(settings)
    if not mcp_tools:
        # Servers configured but nothing came back: reachable-but-empty and
        # unreachable look the same here, and the warning logs say which.
        print("  no tools loaded — see the warnings above")
        return 1
    for tool in mcp_tools:
        first_line = (tool.description or "").splitlines()
        print(f"  {tool.name:<28} {first_line[0] if first_line else ''}")
    return 0


def _serve(settings) -> int:
    """Run the A2A server."""
    try:
        from general_agent.a2a_server import AGENT_CARD_PATH, serve
    except ImportError:
        print(
            'error: serving over A2A needs the extra: pip install "general-agent[a2a]"',
            file=sys.stderr,
        )
        return 1

    print(
        f"Serving {settings.agent_name} over A2A on "
        f"http://{settings.a2a_host}:{settings.a2a_port}{AGENT_CARD_PATH}"
    )
    print("This endpoint is unauthenticated; put a proxy in front before exposing it.\n")
    try:
        serve(settings)
    except KeyboardInterrupt:
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code."""
    _load_environment()

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Imported after the environment is loaded, so settings and the Langfuse
    # client both see the .env file.
    from general_agent.config import Settings
    from general_agent.runner import AgentRunner

    settings = Settings.from_env()
    overrides = {
        key: value
        for key, value in (
            ("a2a_host", args.host),
            ("a2a_port", args.port),
            ("a2a_url", args.public_url),
        )
        if value is not None
    }
    if overrides:
        settings = replace(settings, **overrides)

    if args.list_tools:
        return _list_tools(settings)

    if not settings.tracing_enabled:
        print(
            "note: Langfuse tracing is off — set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY to record traces.\n",
            file=sys.stderr,
        )

    if args.serve:
        return _serve(settings)

    with AgentRunner(settings, session_id=args.session_id) as runner:
        if not args.question:
            return _interactive(runner)

        try:
            reply = runner.ask(
                " ".join(args.question),
                user_id=args.user_id,
                tags=[*args.tags, "cli"],
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(reply.text)
        if reply.model:
            print(f"\n(served by {reply.model})", file=sys.stderr)
        if reply.trace_id:
            print(f"(trace {reply.trace_id})", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
