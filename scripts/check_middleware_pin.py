"""Report whether the pinned middleware commit is behind upstream.

The middleware dependency in ``pyproject.toml`` is pinned to a commit so that
builds are reproducible (see the comment there). The cost of a pin is that a
fix upstream never arrives on its own — so this checks for one, and CI runs it
on a schedule.

    python scripts/check_middleware_pin.py            # report only
    python scripts/check_middleware_pin.py --bump     # rewrite the pin too

Exit codes: 0 up to date, 1 behind, 2 could not determine.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

#: Captures the repo URL and the pinned revision of the middleware dependency.
PIN_PATTERN = re.compile(
    r'"ratelimit-fallback @ git\+(?P<url>[^"@]+?)(?:@(?P<rev>[0-9a-fA-F]{7,40}))?"'
)

DEFAULT_BRANCH = "main"


class PinError(RuntimeError):
    """The pin could not be read or resolved."""


def read_pin(text: str) -> tuple[str, str | None]:
    """Return the dependency's ``(url, pinned_revision)``.

    Raises:
        PinError: If the dependency line is missing or malformed.
    """
    match = PIN_PATTERN.search(text)
    if match is None:
        msg = f"No ratelimit-fallback git dependency found in {PYPROJECT.name}"
        raise PinError(msg)
    return match.group("url"), match.group("rev")


def resolve_upstream(url: str, branch: str = DEFAULT_BRANCH) -> str:
    """Return the current commit SHA of ``branch`` on the remote.

    Uses ``git ls-remote`` rather than cloning: it is one network round trip
    and needs no working copy.
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", url, f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        msg = f"Could not reach {url}: {exc}"
        raise PinError(msg) from exc

    line = result.stdout.strip()
    if not line:
        msg = f"Remote {url} has no branch {branch!r}"
        raise PinError(msg)
    return line.split()[0]


def bump(text: str, url: str, old: str | None, new: str) -> str:
    """Return ``text`` with the pin rewritten to ``new``."""
    old_spec = f'"ratelimit-fallback @ git+{url}' + (f'@{old}"' if old else '"')
    return text.replace(old_spec, f'"ratelimit-fallback @ git+{url}@{new}"')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bump", action="store_true", help="Rewrite the pin to upstream HEAD.")
    parser.add_argument("--branch", default=DEFAULT_BRANCH, help="Upstream branch to compare to.")
    args = parser.parse_args(argv)

    text = PYPROJECT.read_text()
    try:
        url, pinned = read_pin(text)
        upstream = resolve_upstream(url, args.branch)
    except PinError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if pinned is None:
        print(f"unpinned: dependency tracks {args.branch} (currently {upstream[:12]})")
        print("Pin it so builds are reproducible.")
        return 1

    if pinned.lower() == upstream.lower():
        print(f"up to date: pinned at {pinned[:12]} == {args.branch}")
        return 0

    print(f"behind: pinned {pinned[:12]}, {args.branch} is at {upstream[:12]}")
    print(f"  compare: {url.removesuffix('.git')}/compare/{pinned}...{upstream}")

    if args.bump:
        PYPROJECT.write_text(bump(text, url, pinned, upstream))
        print(f"  bumped pin to {upstream[:12]} — reinstall and run the tests")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
