"""Tools the generic agent can call.

Deliberately small and dependency-light: arithmetic and clock questions are the
two things an LLM reliably gets wrong on its own, and web search is the one
capability that turns a chat model into an agent that knows about today. Add
your own tools here and they are picked up by :func:`build_tools`.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

__all__ = ["build_tools", "calculator", "current_datetime"]

# Only these node types are evaluated. Anything else — names, attribute access,
# calls — is rejected, so the tool cannot become an arbitrary-code sink for
# whatever the model decides to emit.
_BIN_OPS: Mapping[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: Mapping[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Guard against `9**9**9` locking up the process for minutes.
_MAX_EXPONENT = 1000


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            msg = f"Unsupported constant: {node.value!r}"
            raise ValueError(msg)
        return node.value
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            msg = f"Unsupported unary operator: {type(node.op).__name__}"
            raise ValueError(msg)
        return op(_evaluate(node.operand))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            msg = f"Unsupported operator: {type(node.op).__name__}"
            raise ValueError(msg)
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            msg = f"Exponent too large (max {_MAX_EXPONENT})"
            raise ValueError(msg)
        return op(left, right)
    msg = f"Unsupported expression element: {type(node).__name__}"
    raise ValueError(msg)


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.

    Use this for any calculation rather than working it out yourself.
    Supports + - * / // % ** and parentheses, on numbers only.

    Args:
        expression: The arithmetic to evaluate, e.g. "17 * 23" or "(2+3)**4".
    """
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return f"Error: {expression!r} is not a valid expression ({exc.msg})."

    try:
        result = _evaluate(parsed)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, OverflowError) as exc:
        return f"Error: {exc}"

    # Present 6.0 as 6, but keep genuine fractions intact.
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


@tool
def current_datetime(timezone: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone.

    Use this whenever the answer depends on what "now", "today" or "this year"
    means — the model's own sense of the date is stale.

    Args:
        timezone: IANA timezone name, e.g. "UTC", "Europe/Berlin",
            "America/New_York". Defaults to UTC.
    """
    name = (timezone or "UTC").strip()
    if name.upper() == "UTC":
        zone = UTC
    else:
        try:
            zone = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            return f"Error: unknown timezone {name!r}. Use an IANA name like 'Europe/Berlin'."

    now = datetime.now(zone)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


def _web_search_tool(max_results: int) -> BaseTool | None:
    """Return a Tavily search tool, or ``None`` if it isn't available.

    Search is optional on purpose: the agent is fully functional without it,
    so a missing package or API key degrades the tool set instead of failing
    the whole process at startup.
    """
    try:
        from langchain_tavily import TavilySearch
    except ImportError:
        return None

    import os

    if not os.environ.get("TAVILY_API_KEY"):
        return None

    return TavilySearch(max_results=max_results)


def build_tools(*, enable_web_search: bool = True, max_search_results: int = 5) -> list[BaseTool]:
    """Assemble the agent's tool list.

    Args:
        enable_web_search: Include Tavily web search when the package and
            ``TAVILY_API_KEY`` are both present.
        max_search_results: Results per search call.

    Returns:
        The tools to hand to ``create_agent``.
    """
    tools: list[BaseTool] = [calculator, current_datetime]
    if enable_web_search:
        search = _web_search_tool(max_search_results)
        if search is not None:
            tools.append(search)
    return tools
