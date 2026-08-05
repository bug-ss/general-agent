"""Tests for the middleware pin check.

Only the pure parts — reading the pin and rewriting it. Resolving upstream
needs the network, so CI exercises that path for real instead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "check_middleware_pin.py"
_spec = importlib.util.spec_from_file_location("check_middleware_pin", _PATH)
assert _spec and _spec.loader
check_pin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_pin)

URL = "https://github.com/bug-ss/ratelimit-fallback.git"
SHA = "360a4559cc413b11779c5fcedcf1726d2aa3eca5"


def test_reads_a_pinned_dependency() -> None:
    url, rev = check_pin.read_pin(f'    "ratelimit-fallback @ git+{URL}@{SHA}",\n')
    assert url == URL
    assert rev == SHA


def test_reads_an_unpinned_dependency() -> None:
    """An unpinned URL is reported, not mistaken for being up to date."""
    url, rev = check_pin.read_pin(f'    "ratelimit-fallback @ git+{URL}",\n')
    assert url == URL
    assert rev is None


def test_rejects_a_missing_dependency() -> None:
    with pytest.raises(check_pin.PinError):
        check_pin.read_pin('dependencies = ["langchain>=1.0"]')


def test_bump_rewrites_the_pin_in_place() -> None:
    new = "a" * 40
    text = f'    "ratelimit-fallback @ git+{URL}@{SHA}",\n'
    assert check_pin.bump(text, URL, SHA, new) == f'    "ratelimit-fallback @ git+{URL}@{new}",\n'


def test_bump_pins_a_previously_unpinned_dependency() -> None:
    new = "b" * 40
    text = f'    "ratelimit-fallback @ git+{URL}",\n'
    assert check_pin.bump(text, URL, None, new) == f'    "ratelimit-fallback @ git+{URL}@{new}",\n'


def test_the_real_pyproject_is_pinned() -> None:
    """Guards against someone 'tidying' the SHA away."""
    url, rev = check_pin.read_pin(check_pin.PYPROJECT.read_text())
    assert url == URL
    assert rev is not None, "the middleware dependency must stay pinned to a commit"
    assert len(rev) == 40, "pin to a full SHA, not an abbreviation"
