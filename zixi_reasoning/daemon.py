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


def process_event_file(root: Path, path: Path) -> None:
    old_active = store.read_active(root)
    event_text = path.read_text(encoding="utf-8")
    new_active = fast.process_event(old_active, event_text)
    store.atomic_write(root / store.ACTIVE_FILENAME, new_active)
    store.git_commit(root, f"active: {path.stem}")

    # Crash-safety gap (spec §39): spill crystallization candidates into the
    # spool BEFORE running them, so a crash between the ACTIVE write and the
    # consolidation loses nothing. The spill is then processed in this same
    # loop cycle (queue_paths re-scans after we return).
    for kind, candidate, targets in _consolidations_in(new_active, old_active):
        for target in targets:
            store.enqueue_consolidation(
                root,
                f"[TARGET] {target}\n\n{parser.make_tag(kind, candidate)}",
            )
    path.unlink(missing_ok=True)
    # Best effort: process the spills we just created (still one writer).
    for spill in store.queue_paths(root):
        if spill.name.startswith("consolidate-"):
            _process_consolidation(root, spill)


def _process_consolidation(root: Path, path: Path) -> None:
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
        return
    for el in elements:
        if el.kind in ("REFLECT", "SKILL"):
            action, node = consolidate.consolidate(root, target, el.text, el.links, kind=el.kind)
            if action not in ("drop", "noop"):
                store.git_commit(root, f"consolidate: {store.memory_path(root, node).name}")
                logger.info("consolidated %s -> %s (%s)", path.name, node, action)
    path.unlink(missing_ok=True)


def process_consolidation_file(root: Path, path: Path) -> None:
    _process_consolidation(root, path)


def process_queue(root: Path) -> int:
    """Process everything currently in the queue. Returns job count handled."""
    handled = 0
    for path in store.queue_paths(root):
        try:
            if path.name.startswith("event-"):
                process_event_file(root, path)
            elif path.name.startswith("consolidate-"):
                process_consolidation_file(root, path)
            else:
                continue
            handled += 1
        except Exception as exc:  # noqa: BLE001 — job stays, daemon survives
            logger.error("job failed; will retry next cycle: %s (%s)", path.name, exc)
    return handled


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
        logger.info("once: processed %d job(s)", n)
        return
    run_forever(root, poll=args["poll"])


if __name__ == "__main__":
    main()
