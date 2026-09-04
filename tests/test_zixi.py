"""Tests for Zixi.Reasoning core."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Tests must be deterministic: never hit an LLM endpoint.
os.environ["ZIXI_BACKEND"] = "rules"

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


def test_parser_six_tags():
    text = (
        "[FACT] verified\n"
        "[STATE] current\n"
        "[REASONING] inference\n"
        "[REFLECT] lesson\n"
        "[ASSUME] the block is the player\n"
        "[LAB] pushed UP x3 -> block reached rows 35-39\n"
    )
    els = parser.parse_text(text)
    assert [e.kind for e in els] == ["FACT", "STATE", "REASONING", "REFLECT", "ASSUME", "LAB"]
    assert els[4].text == "the block is the player"
    assert els[5].text == "pushed UP x3 -> block reached rows 35-39"


def test_parser_hypothesis_helper():
    assert parser.is_hypothesis("ASSUME")
    assert parser.is_hypothesis("LAB")
    assert not parser.is_hypothesis("FACT")
    assert not parser.is_hypothesis(None)


def test_parser_assume_with_links():
    text = "[ASSUME] actions move the block [[ls20]]"
    (el,) = parser.parse_text(text)
    assert el.kind == "ASSUME"
    assert el.text == "actions move the block"
    assert el.links == ["ls20"]


def test_parser_skill_tag():
    text = "[SKILL] verify player identity by watching which entity moves [[ArcAGI]]"
    (el,) = parser.parse_text(text)
    assert el.kind == "SKILL"
    assert parser.is_hypothesis(el.kind) is False
    assert el.links == ["ArcAGI"]


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


def test_fast_unstructured_ignored():
    # User ruling ([OBS] never enters): unstructured text is NOT absorbed
    # into any primitive. The old flatten-to-FACT heuristic is gone.
    active = "# ACTIVE\n"
    new = fast.process_event(active, "the user asked for memory framework")
    assert new == active
    facts = [e.text for e in parser.parse_text(new) if e.kind == "FACT"]
    assert facts == []


def test_fast_explicit_reflect_kept():
    active = "# ACTIVE\n"
    event = "[REFLECT] storage is not memory"
    new = fast.process_event(active, event)
    reflects = [e.text for e in parser.parse_text(new) if e.kind == "REFLECT"]
    assert reflects == ["storage is not memory"]


def test_fast_assume_same_subject_folded():
    active = "# ACTIVE\n\n[ASSUME] actions move the block [[ls20]]\n"
    event = "[ASSUME] actions move the player [[ls20]]"
    new = fast.process_event(active, event)
    assumes = [e.text for e in parser.parse_text(new) if e.kind == "ASSUME"]
    assert assumes == ["actions move the player"]


def test_fast_assume_keeps_hypothesis_identity():
    # An ASSUME must never be flattened into the FACT zone by the rules worker.
    active = "# ACTIVE\n"
    event = "[ASSUME] the 5x5 o/b block is me [[ls20]]"
    new = fast.process_event(active, event)
    kinds = [e.kind for e in parser.parse_text(new) if e.kind]
    assert kinds == ["ASSUME"]
    facts = [e for e in parser.parse_text(new) if e.kind == "FACT"]
    assert facts == []


def test_fast_lab_state_like_folding():
    active = "# ACTIVE\n\n[LAB] pushed UP -> rows 35-39 [[ls20]]\n"
    event = "[LAB] pushed RIGHT -> rows 40-44 [[ls20]]"
    new = fast.process_event(active, event)
    labs = [e.text for e in parser.parse_text(new) if e.kind == "LAB"]
    assert labs == ["pushed RIGHT -> rows 40-44"]


def test_fast_skill_cumulative_not_folded():
    # SKILL is truth-zone: two different skill lines both survive (no folding).
    active = "# ACTIVE\n"
    event = (
        "[SKILL] read the map axes before acting [[ArcAGI]]\n"
        "[SKILL] confirm identity by watching which entity moves [[ArcAGI]]"
    )
    new = fast.process_event(active, event)
    skills = [e.text for e in parser.parse_text(new) if e.kind == "SKILL"]
    assert len(skills) == 2
    # and never lands in the hypothesis split
    main, hyp = recall._split_hypotheses(new)
    assert "[SKILL]" in main and "[SKILL]" not in hyp


# ---------------------------------------------------------------------------
# Primitive-only digestion (zero-LLM core, 2026-09-04 [OBS] ruling)
# ---------------------------------------------------------------------------

def test_has_primitives_gate():
    event_with = "[EVENT] t\n[SESSION] arc-ls20\n[STEP] 5 | last action=ACTION1, levels=0/7, state=NOT_FINISHED\n[ASSUME] the block is me [[ls20]]\n[LAB] push UP: does the block enter the shaft [[ls20]]\n"
    event_without = "[EVENT] t\n[SESSION] arc-ls20\n[STEP] 5 | last action=ACTION1, levels=0/7, state=NOT_FINISHED\n[OBS] The block moved up. Nothing else happened.\n"
    assert fast.has_primitives(event_with)
    assert not fast.has_primitives(event_without)


def test_narrative_never_becomes_fact():
    # User ruling: [OBS] never enters. Unstructured text must NOT turn into
    # a [FACT] — the old flatten/heuristic path is gone.
    active = "# ACTIVE\n"
    event = "[EVENT] t\n[OBS] The block moved UP to rows 5-9. Level unchanged.\n"
    new = fast._rules_update(active, event)
    assert new == active  # nothing absorbed, nothing invented
    facts = [e for e in parser.parse_text(new) if e.kind == "FACT"]
    assert facts == []


def test_narrative_with_primitives_only_absorbs_primitives():
    active = "# ACTIVE\n"
    event = "[EVENT] t\n[OBS] The block moved UP. But this is narrative.\n[ASSUME] the goal box is the top box [[ls20]]\n[LAB] push UP: cover W -> shift? [[ls20]]\n"
    new = fast._rules_update(active, event)
    kinds = [e.kind for e in parser.parse_text(new) if e.kind]
    assert kinds == ["ASSUME", "LAB"]
    # narrative line itself never appears in ACTIVE
    assert "narrative" not in new or "But this is narrative" not in new


def test_explicit_remember_phrase_ignored():
    # No heuristic: "remember this" phrases do NOT crystallize by themselves.
    active = "# ACTIVE\n"
    event = "[EVENT] t\nremember this: the block covers W at rows 30-34\n"
    new = fast._rules_update(active, event)
    assert new == active
    reflects = [e for e in parser.parse_text(new) if e.kind == "REFLECT"]
    assert reflects == []


def test_daemon_batch_primitives_only(tmp_path):
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    store.enqueue_event(tmp_path, "[EVENT] t\n[SESSION] arc-ls20\n[STEP] 12 | last action=ACTION1, levels=0/7, state=NOT_FINISHED\n[OBS] I'll observe.\n")
    store.enqueue_event(tmp_path, "[EVENT] t\n[SESSION] arc-ls20\n[STEP] 30 | last action=ACTION2, levels=1/7, state=NOT_FINISHED\n[OBS] The block entered the top box.\n[REFLECT] goal box entry triggers the level =>[[arc-ls20]]\n")
    n = process_queue(tmp_path)
    assert n == 2
    active = store.read_active(tmp_path)
    # only the primitive-carrying event contributed
    assert "goal box entry triggers the level" in active
    assert "entered the top box" not in active    # narrative stays out
    assert "I'll observe" not in active           # narrative stays out
    assert store.queue_paths(tmp_path) == []  # both consumed


def test_daemon_skill_crystallizes(tmp_path):
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    store.enqueue_event(
        tmp_path,
        "[EVENT] skill test\n"
        "[SKILL] verify identity by watching which entity moves =>[[arc-player]]",
    )
    n = process_queue(tmp_path)
    assert n == 1
    node = tmp_path / "memory" / "arc-player.md"
    assert node.exists()
    body = node.read_text(encoding="utf-8")
    assert "[SKILL]" in body
    assert "watching which entity moves" in body


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


def test_compile_context_hypotheses_split(tmp_path):
    store.ensure_layout(tmp_path)
    store.atomic_write(
        tmp_path / store.ACTIVE_FILENAME,
        "# ACTIVE\n"
        "\n"
        "[FACT] level 1 was completed [[ls20]]\n"
        "[ASSUME] actions move the block [[ls20]]\n"
        "[LAB] pushed UP -> rows 35-39 [[ls20]]\n"
        "[REFLECT] verify before trusting guesses\n",
    )
    ctx = recall.compile_context(tmp_path)
    # truth zone keeps FACT/REFLECT, hypothesis lines are split out
    assert "level 1 was completed" in ctx
    assert "verify before trusting guesses" in ctx
    assert "Hypotheses (UNVERIFIED" in ctx
    assert "actions move the block" in ctx
    assert "pushed UP -> rows 35-39" in ctx
    # the truth zone (between "Current:" and the hypotheses header) must not
    # contain hypothesis lines
    current = ctx.split("Current:")[1].split("Hypotheses (UNVERIFIED")[0]
    assert "[ASSUME]" not in current
    assert "[LAB]" not in current


def test_recall_split_hypotheses_roundtrip():
    active = "# ACTIVE\n\n[FACT] f\n[ASSUME] a [[X]]\n[LAB] l\n[REFLECT] r\n"
    main, hyp = recall._split_hypotheses(active)
    assert "[ASSUME]" in hyp and "[LAB]" in hyp
    assert "[FACT]" in main and "[REFLECT]" in main
    assert "[ASSUME]" not in main


def test_consolidate_ignores_hypothesis_lines(tmp_path):
    # Crystallization only accepts REFLECT candidates: an ASSUME line
    # submitted as a candidate still lands as a REFLECT, but a bare
    # ASSUME in ACTIVE never triggers nodes by itself. (Non-regression
    # guard for the hypothesis zone.)
    store.ensure_layout(tmp_path)
    action, _ = consolidate.consolidate(tmp_path, "Test-Node", "a stable reflection")
    assert action == "add"
    p = tmp_path / "memory" / "test-node.md"
    assert "[ASSUME]" not in p.read_text(encoding="utf-8")


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
    # git history exists (async commits: drain the pool before reading)
    from zixi_reasoning.daemon import git_flush

    git_flush()
    log = subprocess.run(["git", "-C", str(tmp_path), "log", "--oneline"], capture_output=True, text=True)
    assert "active:" in log.stdout  # batch commit covers events + consolidations


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


# ---------------------------------------------------------------------------
# Provider generalization (2026-09-04, ARC removed — generic memory provider)
# ---------------------------------------------------------------------------

def test_extract_primitive_lines_only_line_start():
    text = (
        "some prose mentioning [ASSUME] not at line start\n"
        "[ASSUME] the block is me [[ls20]]\n"
        "plain narrative\n"
        "  [LAB] push UP: enter shaft? [[ls20]]\n"
        "[FACT] level 1 completes on box entry [[ls20]]\n"
    )
    lines = parser.extract_primitive_lines(text)
    assert lines == [
        "[ASSUME] the block is me [[ls20]]",
        "[LAB] push UP: enter shaft? [[ls20]]",
        "[FACT] level 1 completes on box entry [[ls20]]",
    ]


def test_sync_turn_forwards_every_turn(tmp_path):
    from zixi_reasoning.provider import ZixiMemoryProvider

    p = ZixiMemoryProvider()
    p._root = tmp_path
    # pure conversation: STILL forwarded (background listener watches every turn)
    p.sync_turn("hey, what's the weather?", "It's sunny out.", session_id="s1")
    assert len(store.queue_paths(tmp_path)) == 1
    text = store.queue_paths(tmp_path)[0].read_text(encoding="utf-8")
    assert "[USER]" in text and "It's sunny out." in text
    # turn carrying primitive lines: forwarded with primitives inside
    p.sync_turn(
        "remember: the maps always show row axes first",
        "[REFLECT] map reading order matters [[zixi]]\n[LAB] check axes on next map [[zixi]]",
        session_id="s2",
    )
    paths = store.queue_paths(tmp_path)
    assert len(paths) == 2
    assert "[REFLECT]" in paths[1].read_text(encoding="utf-8")


def test_provider_to_daemon_primitive_path(tmp_path):
    # provider forwards a turn whose assistant message carries a primitive;
    # daemon folds it deterministically (fast lane, no LLM in tests/rules).
    from zixi_reasoning.provider import ZixiMemoryProvider

    p = ZixiMemoryProvider()
    p._root = tmp_path
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    p.sync_turn("", "[REFLECT] slower memory is state revision, not append =>[[zixi-reasoning]]", session_id="s")
    n = process_queue(tmp_path)
    assert n == 1
    active = store.read_active(tmp_path)
    assert "slower memory is state revision, not append" in active


def test_parity_queues_ignored_in_git(tmp_path):
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    store.enqueue_event(tmp_path, "[EVENT] transient")
    (tmp_path / ".gitignore").read_text(encoding="utf-8")
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "queue/" in gitignore
