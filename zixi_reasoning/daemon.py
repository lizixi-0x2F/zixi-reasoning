"""Zixi.Reasoning daemon — zixi-memoryd, the single slow-memory writer.

Consumes the filesystem spool (spec §12, §28, §29):

    queue/event-*.md        -> fast worker -> atomic ACTIVE rewrite -> git commit
                             -> reflection lines with =>[[node]] trigger consolidation
    queue/consolidate-*.md  -> consolidator (explicit/manual trigger)

Success deletes the job; failure leaves it for the next cycle. Because the
spool is plain files, daemon death loses nothing: on restart we scan the queue
oldest-first and replay.

Exactly one slow-memory writer exists (enforced via a pidfile; the Hermes
provider never writes memory/*.md itself, it only enqueues).
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import consolidate, fast, parser, store

logger = logging.getLogger("zixi.memoryd")

PIDFILE = "memoryd.pid"


def _ensure_single_writer(root: Path) -> bool:
    pidfile = root / PIDFILE
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip())
            os.kill(pid, 0)  # alive?
            logger.error(f"another zixi-memoryd seems to be running (pid {pid}); aborting")
            return False
        except (ValueError, ProcessLookupError):
            pidfile.unlink(missing_ok=True)
    pidfile.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _clear_pidfile(root: Path) -> None:
    (root / PIDFILE).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Async git commits (2026-09-04): version control is an observability layer,
# it must never block digestion. Commits queue to a single worker (one
# writer order — two concurrent `git add -A` on one repo would interleave)
# and execute asynchronously; callers fire-and-forget.
# ---------------------------------------------------------------------------

_git_executor: ThreadPoolExecutor | None = None


def _git_pool() -> ThreadPoolExecutor:
    global _git_executor
    if _git_executor is None:
        _git_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="zixi-git")
    return _git_executor


def git_commit_async(root: Path, message: str) -> None:
    """Fire-and-forget commit, ordered per repo by the single worker."""
    try:
        _git_pool().submit(store.git_commit, root, message)
    except RuntimeError:  # pool shut down (shutdown race) — drop, git is optional
        pass


def git_flush(timeout: float = 30.0) -> None:
    """Block until queued commits finished (shutdown + drain)."""
    global _git_executor
    if _git_executor is not None:
        _git_executor.shutdown(wait=True, cancel_futures=False)
        _git_executor = None


def _consolidations_in(active: str, old_active: str) -> list[tuple[str, str, list[str]]]:
    """Collect (kind, candidate_text, targets) from new REFLECT/SKILL lines marked =>."""
    jobs: list[tuple[str, str, list[str]]] = []
    old_set = set(old_active.splitlines())
    for el in parser.parse_text(active):
        if el.kind not in ("REFLECT", "SKILL") or not el.consolidate_targets:
            continue
        if el.raw in old_set:
            continue
        jobs.append((el.kind, el.text, el.consolidate_targets))
    return jobs


def process_queue_batch(root: Path) -> int:
    """Digest the WHOLE queue in one batch.

    PRIMITIVES-ONLY: an event contributes to ACTIVE iff it carries explicit
    primitive lines ([ASSUME]/[LAB]/[FACT]/[STATE]/[REASONING]/[REFLECT]/
    [SKILL]); everything else is dropped — no LLM in the ingestion path, no
    synthesis from narration. Events fold deterministically (same-subject
    STATE/ASSUME/LAB replace, truth-zone append). One atomic ACTIVE rewrite
    + ONE commit at batch end (events + consolidations committed together:
    `git add -A` snapshots the whole tree, so committing twice per batch
    would fold the second commit into the first's tree state and the
    second commit would be emptied to "nothing to commit").

    Returns number of event files consumed (incl. dropped).
    """
    paths = store.queue_paths(root)
    if not paths:
        return 0
    # Queue may hold consolidate-* spills without any pending event.
    if not any(p.name.startswith("event-") for p in paths):
        for spill in [p for p in paths if p.name.startswith("consolidate-")]:
            _process_consolidation(root, spill)
        git_commit_async(root, "active: consolidation spill batch")
        return 0

    active = store.read_active(root)
    done: list[Path] = []
    n_consol = 0

    for path in paths:
        if not path.name.startswith("event-"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue  # unreadable job stays for the next cycle
        if fast.has_primitives(text):
            active = fast._rules_update(active, text)
        # else: no primitives — drop. Nothing to fold, nothing invented.
        done.append(path)

    old_was = store.read_active(root)
    if active != old_was:
        store.atomic_write(root / store.ACTIVE_FILENAME, active)

    # Crash-safety spill: crystallize => targets born in this batch.
    for kind, candidate, targets in _consolidations_in(active, old_was):
        for target in targets:
            store.enqueue_consolidation(
                root,
                f"[TARGET] {target}\n\n{parser.make_tag(kind, candidate)}",
            )

    for path in done:
        path.unlink(missing_ok=True)

    # Best effort: process spills we just created (still one writer).
    for spill in store.queue_paths(root):
        if spill.name.startswith("consolidate-"):
            if _process_consolidation(root, spill):
                n_consol += 1

    git_commit_async(root, f"active: batch {len(done)} event(s), {n_consol} consolidation(s)")
    return len(done)


def _process_consolidation(root: Path, path: Path) -> bool:
    """Consolidate one spill. Returns True when a memory node changed.

    NO git commit here: the caller (batch digest) commits once at batch end
    — `git add -A` snapshots the whole tree, so per-spill commits would
    fold each other into the first commit's tree.
    """
    text = path.read_text(encoding="utf-8")
    elements = parser.parse_text(text)
    target = None
    for el in elements:
        if el.kind is None and el.raw.startswith("[TARGET]"):
            target = el.raw[len("[TARGET]") :].strip()
            break
        if el.kind is None and el.raw.startswith("# "):
            target = el.raw[2:].strip()
            break
    if target is None:
        logger.warning("consolidation file %s has no target; leaving it", path.name)
        return False
    changed = False
    for el in elements:
        if el.kind in ("REFLECT", "SKILL"):
            action, node = consolidate.consolidate(root, target, el.text, el.links, kind=el.kind)
            if action not in ("drop", "noop"):
                changed = True
                logger.info("consolidated %s -> %s (%s)", path.name, node, action)
    path.unlink(missing_ok=True)
    return changed


def process_queue(root: Path) -> int:
    """Digest the queue in one batch: primitive-only deterministic fold
    (no LLM in the ingestion path; events without primitives are dropped).

    Returns job count handled (unreadable jobs stay for the next cycle).
    """
    return process_queue_batch(root)


def run_forever(root: Path, poll: float = 1.0) -> None:
    store.ensure_layout(root)
    store.git_init_repo(root)
    if not _ensure_single_writer(root):
        sys.exit(1)
    logger.info("zixi-memoryd serving %s (poll %.1fs)", root, poll)
    try:
        while True:
            process_queue(root)
            time.sleep(poll)
    except KeyboardInterrupt:
        pass
    finally:
        git_flush()
        _clear_pidfile(root)


def parse_args(argv: list[str] | None = None) -> dict:
    import argparse

    ap = argparse.ArgumentParser(prog="zixi-memoryd", description="Zixi.Reasoning memory daemon")
    ap.add_argument("--root", default=os.environ.get("HERMES_HOME"), help="memory root (default ~/.hermes/zixi)")
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--once", action="store_true", help="process queue once, then exit")
    ap.add_argument("--backend", choices=["rules", "llm"], default=None)
    ap.add_argument("--verbose", action="store_true")
    return vars(ap.parse_args(argv))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args["verbose"] else logging.INFO,
        format="[zixi-memoryd] %(asctime)s %(levelname)s %(message)s",
    )
    if args["backend"]:
        os.environ["ZIXI_BACKEND"] = args["backend"]
    # Make Hermes' model client importable from this standalone process so
    # the LLM backend reuses Hermes' own configuration (spec: one client).
    from .hermes_env import ensure_hermes_path

    src = ensure_hermes_path()
    if src:
        logger.info("using Hermes model client at %s", src)
    root = Path(args["root"] or "").expanduser() if args["root"] else store.default_root()
    if args["once"]:
        store.ensure_layout(root)
        store.git_init_repo(root)
        n = process_queue(root)
        git_flush()
        logger.info("once: processed %d job(s)", n)
        return
    run_forever(root, poll=args["poll"])


if __name__ == "__main__":
    main()
