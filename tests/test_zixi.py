"""Tests for Zixi.Reasoning core."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from zixi_reasoning import consolidate, fast, parser, recall, store
from zixi_reasoning.daemon import process_queue


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parser_four_tags():
    text = "[FACT] facts here\n[STATE] state here\n[REASONING] reasoning here\n[REFLECT] reflect here\n"
    els = parser.parse_text(text)
    assert [e.kind for e in els] == ["FACT", "STATE", "REASONING", "REFLECT"]
    assert els[0].text == "facts here"
    assert els[1].text == "state here"


def test_parser_wikilinks():
    text = "[REFLECT] Memory is state revision. [[Hermes]] [[Reflective Memory]]"
    (el,) = parser.parse_text(text)
    assert el.links == ["Hermes", "Reflective Memory"]


def test_parser_state_arrow_and_consolidate_inline():
    text = "[REFLECT] database is not a necessary primitive =>[[Zixi.Reasoning]]"
    (el,) = parser.parse_text(text)
    assert el.kind == "REFLECT"
    assert el.text == "database is not a necessary primitive"
    assert el.consolidate_targets == ["Zixi.Reasoning"]


def test_parser_state_arrow_separate_line():
    text = "->[STATE] implementing standalone provider"
    (el,) = parser.parse_text(text)
    assert el.state_transitions == ["implementing standalone provider"]


def test_parser_plain_lines_preserved():
    text = "# ACTIVE\n\nsome prose\n\n[FACT] real fact\n"
    els = parser.parse_text(text)
    assert els[0].kind is None and els[0].raw == "# ACTIVE"
    assert [e.kind for e in els if e.kind] == ["FACT"]


def test_parser_collect_links_dedup():
    text = "[[A]] [[B]] [[A]]"
    assert parser.collect_links(text) == ["A", "B"]


def test_slugify():
    assert parser.slugify("Reflective Memory") == "reflective-memory"
    assert parser.slugify("Zixi.Reasoning") == "zixi-reasoning"
    assert parser.slugify("心智模型") == "心智模型"


# ---------------------------------------------------------------------------
# Fast worker (rules)
# ---------------------------------------------------------------------------

def test_fast_same_subject_state_replace():
    active = "# ACTIVE\n\n[STATE] building provider [[Hermes]]\n"
    event = "[STATE] deploying provider [[Hermes]]"
    new = fast.process_event(active, event)
    states = [e.text for e in parser.parse_text(new) if e.kind == "STATE"]
    assert states == ["deploying provider"]


def test_fast_unstructured_becomes_fact():
    active = "# ACTIVE\n"
    new = fast.process_event(active, "the user asked for memory framework")
    facts = [e.text for e in parser.parse_text(new) if e.kind == "FACT"]
    assert len(facts) == 1
    assert "memory framework" in facts[0]


def test_fast_explicit_reflect_kept():
    active = "# ACTIVE\n"
    event = "[REFLECT] storage is not memory"
    new = fast.process_event(active, event)
    reflects = [e.text for e in parser.parse_text(new) if e.kind == "REFLECT"]
    assert reflects == ["storage is not memory"]


# ---------------------------------------------------------------------------
# Consolidator (rules)
# ---------------------------------------------------------------------------

def test_consolidate_add_new_node(tmp_path):
    action, node = consolidate.consolidate(tmp_path, "Test-Node", "a stable reflection")
    assert action == "add"
    p = tmp_path / "memory" / "test-node.md"
    assert p.exists()
    assert "a stable reflection" in p.read_text(encoding="utf-8")


def test_consolidate_drop_duplicate(tmp_path):
    store.ensure_layout(tmp_path)
    consolidate.consolidate(tmp_path, "Test-Node", "same reflection")
    action, _ = consolidate.consolidate(tmp_path, "Test-Node", "same reflection")
    assert action == "drop"


def test_consolidate_revises_via_llm_only_placeholder():
    # rules backend never silently mutates existing content (ADD only w/ dedup)
    pass


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------

def test_recall_lexical_then_wikilink(tmp_path):
    store.ensure_layout(tmp_path)
    (tmp_path / "memory").mkdir(exist_ok=True)
    (tmp_path / "memory" / "hermes.md").write_text(
        "# Hermes\n\n[FACT] agent framework\n[[Zixi.Reasoning]]\n", encoding="utf-8"
    )
    (tmp_path / "memory" / "zixi-reasoning.md").write_text(
        "# Zixi.Reasoning\n\n[REFLECT] memory is state transition\n", encoding="utf-8"
    )
    nodes = recall.recall(tmp_path, "hermes agent")
    names = [p.name for p in nodes]
    assert "hermes.md" in names
    assert "zixi-reasoning.md" in names  # 1-hop association


def test_compile_context_block(tmp_path):
    store.ensure_layout(tmp_path)
    ctx = recall.compile_context(tmp_path, "hermes")
    assert "<zixi-memory>" in ctx
    assert "Current:" in ctx
    assert "</zixi-memory>" in ctx


# ---------------------------------------------------------------------------
# Daemon end-to-end (rules)
# ---------------------------------------------------------------------------

def test_daemon_event_to_crystallization(tmp_path):
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    store.enqueue_event(
        tmp_path,
        "[EVENT] test\n"
        "[USER] build the memory framework\n"
        "[REFLECT] filesystem markdown is the only substrate =>[[Zixi.Reasoning]]",
    )
    n = process_queue(tmp_path)
    assert n == 1
    active = store.read_active(tmp_path)
    assert "filesystem markdown is the only substrate" in active
    node = tmp_path / "memory" / "zixi-reasoning.md"
    assert node.exists()
    # git history exists
    log = subprocess.run(["git", "-C", str(tmp_path), "log", "--oneline"], capture_output=True, text=True)
    assert "active:" in log.stdout
    assert "consolidate:" in log.stdout


def test_daemon_failed_job_kept(tmp_path):
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    # break the daemon by pointing it at an unreadable event? Instead: no-op.
    store.enqueue_event(tmp_path, "[EVENT] fine")
    p = store.queue_paths(tmp_path)[0]
    # simulate failure: make the file unreadable by deleting read permission
    p.chmod(0o000)
    n = process_queue(tmp_path)
    assert n == 0
    p.chmod(0o644)
    n = process_queue(tmp_path)
    assert n == 1


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

def test_atomic_write_no_tmp_left(tmp_path):
    p = tmp_path / "x.md"
    store.atomic_write(p, "hello")
    assert p.read_text(encoding="utf-8") == "hello"
    assert list(tmp_path.glob(".*.tmp")) == []


def test_parity_queues_ignored_in_git(tmp_path):
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    store.enqueue_event(tmp_path, "[EVENT] transient")
    (tmp_path / ".gitignore").read_text(encoding="utf-8")
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "queue/" in gitignore
