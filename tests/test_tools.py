"""Tests for the agent's tools."""

from __future__ import annotations

import pytest

from general_agent.tools import build_tools, calculator, current_datetime


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("17 * 23", "391"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) ** 4", "625"),
        ("7 / 2", "3.5"),
        ("7 // 2", "3"),
        ("-5 + 2", "-3"),
        ("10 % 3", "1"),
        ("6 / 3", "2"),  # whole result, formatted without a trailing .0
    ],
)
def test_calculator_evaluates(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected


def test_calculator_rejects_division_by_zero() -> None:
    assert "division by zero" in calculator.invoke({"expression": "1/0"})


def test_calculator_rejects_syntax_errors() -> None:
    assert calculator.invoke({"expression": "2 +"}).startswith("Error:")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "[].__class__",
        "lambda: 1",
    ],
)
def test_calculator_refuses_non_arithmetic(expression: str) -> None:
    """The tool takes model-authored strings, so it must not evaluate code."""
    result = calculator.invoke({"expression": expression})
    assert result.startswith("Error:")


def test_calculator_caps_exponent() -> None:
    """A huge power would otherwise stall the agent for minutes."""
    assert "Exponent too large" in calculator.invoke({"expression": "9 ** 999999"})


def test_current_datetime_defaults_to_utc() -> None:
    assert "UTC" in current_datetime.invoke({})


def test_current_datetime_accepts_named_zone() -> None:
    result = current_datetime.invoke({"timezone": "Europe/Berlin"})
    assert result.startswith("20")


def test_current_datetime_reports_unknown_zone() -> None:
    assert "unknown timezone" in current_datetime.invoke({"timezone": "Mars/Olympus"})


def test_build_tools_without_search() -> None:
    names = [tool.name for tool in build_tools(enable_web_search=False)]
    assert names == ["calculator", "current_datetime"]


def test_build_tools_skips_search_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Search is optional: no key means fewer tools, not a startup failure."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    names = [tool.name for tool in build_tools(enable_web_search=True)]
    assert "calculator" in names
