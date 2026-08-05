"""Command-line entry point.

Two modes:

    general-agent "What is 17 * 23?"     # one question, then exit
    general-agent                        # interactive chat

Environment is loaded before anything else in :func:`main`. That ordering is
load-bearing: the Langfuse SDK reads credentials when its client is built, so a
``load_dotenv()`` that runs afterwards leaves it authenticated with nothing and
silently dropping every span.
"""

from __future__ import annotations

import argparse
import logging
import sys

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
    if not settings.tracing_enabled:
        print(
            "note: Langfuse tracing is off — set LANGFUSE_PUBLIC_KEY and "
            "LANGFUSE_SECRET_KEY to record traces.\n",
            file=sys.stderr,
        )

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
