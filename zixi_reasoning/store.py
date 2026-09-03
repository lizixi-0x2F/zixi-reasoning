"""Zixi.Reasoning store — filesystem substrate plus git version control.

Layout (see spec §§12, 14, 24, 30):

    <root>/
    ├── ACTIVE.md          # fast memory snapshot (the one truth of current state)
    ├── queue/             # filesystem spool; events & consolidations
    ├── memory/            # crystallized memory nodes (slow memory)
    └── archive/           # optional; historical snapshots

Invariants:
  * Every write is atomic: write temp -> fsync -> os.replace.
  * Only one slow-memory writer exists (the daemon); this module never
    serializes access itself.
  * Derivable state is never persisted: no backlink db, no graph files.

Git is an optional observability layer, not a runtime dependency. Each
committed write produces exactly one commit.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

ACTIVE_FILENAME = "ACTIVE.md"
QUEUE_DIR = "queue"
MEMORY_DIR = "memory"
ARCHIVE_DIR = "archive"

DEFAULT_ROOT = Path("~/.hermes/zixi").expanduser()

_INIT_TEMPLATE = """# ACTIVE

<!-- Zixi.Reasoning active memory.
     Primitives: [FACT] [STATE] [REASONING] [REFLECT]
     Operations: [[WikiLink]]  ->[STATE]  =>[[Node]]
     This file is a snapshot of current cognition, not a journal. -->

"""


def default_root() -> Path:
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env).expanduser() / "zixi"
    return DEFAULT_ROOT


def ensure_layout(root: Path) -> None:
    for sub in (QUEUE_DIR, MEMORY_DIR, ARCHIVE_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)
    active = root / ACTIVE_FILENAME
    if not active.exists():
        atomic_write(active, _INIT_TEMPLATE)


def read_active(root: Path) -> str:
    p = root / ACTIVE_FILENAME
    return p.read_text(encoding="utf-8") if p.exists() else _INIT_TEMPLATE


def atomic_write(path: Path, content: str) -> Path:
    """Write temp -> fsync -> os.replace. Never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    # Best-effort directory fsync (POSIX only; harmless elsewhere)
    try:
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass
    return path


def enqueue_event(root: Path, text: str) -> Path:
    """Write an event into the spool (atomic). Returns its path."""
    (root / QUEUE_DIR).mkdir(parents=True, exist_ok=True)
    return atomic_write(root / QUEUE_DIR / f"event-{uuid.uuid4().hex}.md", text)


def enqueue_consolidation(root: Path, body: str) -> Path:
    (root / QUEUE_DIR).mkdir(parents=True, exist_ok=True)
    return atomic_write(root / QUEUE_DIR / f"consolidate-{uuid.uuid4().hex}.md", body)


def queue_paths(root: Path) -> list[Path]:
    q = root / QUEUE_DIR
    if not q.exists():
        return []
    return sorted(q.glob("*.md"), key=lambda p: p.stat().st_mtime)


def memory_path(root: Path, node: str) -> Path:
    """Resolve a WikiLink node name to its file. Nodes are files; files are nodes."""
    from .parser import slugify

    name = slugify(node)
    p = root / MEMORY_DIR / f"{name}.md"
    assert not p.is_dir(), f"node path conflicts with directory: {p}"
    return p


def node_title(path: Path) -> str:
    """Recover display name from a memory file's first heading."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def list_memory_files(root: Path) -> list[Path]:
    d = root / MEMORY_DIR
    if not d.exists():
        return []
    return sorted(d.glob("*.md"))


# ---------------------------------------------------------------------------
# Git (optional observability layer, spec §30)
# ---------------------------------------------------------------------------

def git_available(root: Path) -> bool:
    return (root / ".git").exists() and shutil.which("git") is not None


def git_commit(root: Path, message: str) -> str | None:
    """Stage the whole memory tree and commit. Returns commit hash or None."""
    if not git_available(root):
        return None
    try:
        subprocess.run(["git", "add", "-A"], cwd=root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message, "--quiet"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, check=True, capture_output=True, text=True,
        )
        return res.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def git_init_repo(root: Path, name: str = "zixi-memoryd", email: str = "zixi-memoryd@local") -> None:
    if (root / ".git").exists():
        return
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", name], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", email], cwd=root, check=True)
    (root / ".gitignore").write_text("# transient spool — processed events are deleted, not archived\nqueue/\n", encoding="utf-8")
