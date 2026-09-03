"""Locate the Hermes source tree so Zixi can reuse Hermes' own model client.

The Hermes `agent` package lives in a source tree ($HERMES_HOME/hermes-agent
or wherever the Hermes process runs from), not in site-packages. Zixi's
daemon is a separate process; for it to call auxiliary_client.call_llm()
it needs that directory on sys.path / PYTHONPATH.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_roots() -> list[Path]:
    roots: list[Path] = []
    home = os.environ.get("HERMES_HOME")
    if home:
        roots.append(Path(home).expanduser() / "hermes-agent")
    roots.append(Path("~/.hermes/hermes-agent").expanduser())
    return roots


def find_hermes_src() -> str | None:
    """Return the directory that contains the ``agent`` package, or None."""
    try:
        import agent  # type: ignore[import]

        p = Path(agent.__file__).resolve().parent.parent  # type: ignore[union-attr]
        if (p / "agent" / "auxiliary_client.py").exists():
            return str(p)
    except Exception:  # noqa: BLE001
        pass
    for entry in sys.path:
        if not entry:
            continue
        cand = Path(entry) / "agent" / "auxiliary_client.py"
        if cand.exists():
            return entry
    for root in _candidate_roots():
        if (root / "agent" / "auxiliary_client.py").exists():
            return str(root)
    return None


def ensure_hermes_path() -> str | None:
    """Put the Hermes source tree on sys.path in this process. Returns it."""
    src = find_hermes_src()
    if src and src not in sys.path:
        sys.path.insert(0, src)
    return src
