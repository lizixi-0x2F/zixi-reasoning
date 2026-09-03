"""Zixi.Reasoning CLI — interact with the machine by hand.

    zixi init                     create the memory tree + git repo
    zixi active                   print ACTIVE.md
    zixi ingest "TEXT"            enqueue one event (manual observation)
    zixi recall "query"           compile the <zixi-memory> recall block
    zixi node "Name"              print one crystallized node
    zixi crystallize "REFLECT..." --to "Node"   enqueue a consolidation
    zixi drain                    run the daemon loop once (same as zixi-memoryd --once)
    zixi log                      show the memory git log
    zixi stats                    quick inventory
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from pathlib import Path

from . import consolidate, parser, recall, store


def _root(args) -> Path:
    if args.root:
        return Path(args.root).expanduser()
    return store.default_root()


def cmd_init(args) -> None:
    root = _root(args)
    store.ensure_layout(root)
    store.git_init_repo(root)
    status = store.git_commit(root, "init: zixi memory tree")
    print(f"zixi memory initialized at {root}")
    print(f"git: {'committed ' + status if status else 'not available'}")


def cmd_active(args) -> None:
    print(store.read_active(_root(args)), end="")


def cmd_ingest(args) -> None:
    root = _root(args)
    store.ensure_layout(root)
    p = store.enqueue_event(root, args.text)
    print(f"event enqueued: {p.name}")


def cmd_recall(args) -> None:
    print(recall.compile_context(_root(args), args.query))


def cmd_node(args) -> None:
    root = _root(args)
    ok, text = recall.grep_node(root, args.name)
    if not ok:
        print(f"(no node '{args.name}' yet; create one by crystallizing)")
        return
    print(text)


def cmd_crystallize(args) -> None:
    root = _root(args)
    store.ensure_layout(root)
    body = f"[TARGET] {args.to}\n\n{parser.make_tag('REFLECT', args.text)}"
    if args.links:
        body += "\n" + " ".join(f"[[{l}]]" for l in args.links)
    p = store.enqueue_consolidation(root, body)
    print(f"consolidation enqueued: {p.name} (run `zixi drain` to process)")


def cmd_drain(args) -> None:
    from .daemon import process_queue

    root = _root(args)
    store.ensure_layout(root)
    store.git_init_repo(root)
    n = process_queue(root)
    print(f"drained {n} job(s)")


def cmd_log(args) -> None:
    root = _root(args)
    if not (root / ".git").exists():
        print("no git repo at", root)
        return
    subprocess.run(["git", "-C", str(root), "log", "--oneline", "-n", str(args.n or 20)])


def cmd_stats(args) -> None:
    root = _root(args)
    active = store.read_active(root)
    mem = store.list_memory_files(root)
    print(f"root      : {root}")
    print(f"ACTIVE.md : {len(active.splitlines())} lines, {len(active)} chars")
    print(f"memory    : {len(mem)} node(s)")
    for p in mem:
        print(f"  - {p.name}")
    print(f"queue     : {len(store.queue_paths(root))} pending")
    print(f"backend   : {os.environ.get('ZIXI_BACKEND', 'rules')}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(prog="zixi", description="Zixi.Reasoning memory tool")
    ap.add_argument("--root", default=None, help="memory root (default ~/.hermes/zixi)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(fn=cmd_init)

    sub.add_parser("active").set_defaults(fn=cmd_active)

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("text")
    p_ingest.set_defaults(fn=cmd_ingest)

    p_recall = sub.add_parser("recall")
    p_recall.add_argument("query")
    p_recall.set_defaults(fn=cmd_recall)

    p_node = sub.add_parser("node")
    p_node.add_argument("name")
    p_node.set_defaults(fn=cmd_node)

    p_cr = sub.add_parser("crystallize")
    p_cr.add_argument("text")
    p_cr.add_argument("--to", dest="to", required=True)
    p_cr.add_argument("--links", nargs="*", default=[])
    p_cr.set_defaults(fn=cmd_crystallize)

    sub.add_parser("drain").set_defaults(fn=cmd_drain)

    p_log = sub.add_parser("log")
    p_log.add_argument("-n", type=int, default=20)
    p_log.set_defaults(fn=cmd_log)

    sub.add_parser("stats").set_defaults(fn=cmd_stats)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
