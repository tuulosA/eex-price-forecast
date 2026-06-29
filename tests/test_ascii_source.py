"""Enforce ASCII-only source.

Console output (CLI messages, logs, exception tracebacks, ``--help`` text) is encoded with the
terminal's code page, which on Windows is legacy cp1252 by default. A stray non-ASCII character (an em
dash, an arrow, ...) in any printable string crashes with ``UnicodeEncodeError`` there. Keeping the
whole package ASCII removes that entire class of failure, so we assert it.
"""

from __future__ import annotations

from pathlib import Path

import eex_forecast

PACKAGE_ROOT = Path(eex_forecast.__file__).parent


def test_source_is_ascii_only() -> None:
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            non_ascii = sorted({hex(ord(c)) for c in line if ord(c) > 127})
            if non_ascii:
                offenders.append(f"{path.relative_to(PACKAGE_ROOT)}:{lineno}: {non_ascii}")
    assert not offenders, (
        "Non-ASCII characters found in source (use ASCII for portable console output):\n"
        + "\n".join(offenders)
    )
