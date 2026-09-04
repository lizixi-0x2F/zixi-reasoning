"""Tests for the MCP server core handlers (no transport spin-up)."""

from __future__ import annotations

import os

os.environ["ZIXI_BACKEND"] = "rules"  # deterministic, never hit LLM

import pytest

from zixi_reasoning import store
from zixi_reasoning.daemon import process_queue
from zixi_reasoning.mcp_server import handle_remember, handle_recall, handle_search


def test_remember_accepts_only_primitives(tmp_path):
    store.ensure_layout(tmp_path)
    res = handle_remember(
        tmp_path,
        [
            "[REFLECT] MCP clients are just ledger customers [[zixi]]",
            "this prose line must be rejected",
        ],
    )
    assert res["accepted"] == 1
    assert res["rejected"] == 1
    # the primitive made it into the queue; the prose didn't
    paths = store.queue_paths(tmp_path)
    assert len(paths) == 1
    text = paths[0].read_text(encoding="utf-8")
    assert "[REFLECT]" in text
    assert "prose line" not in text


def test_remember_no_lines_no_event(tmp_path):
    store.ensure_layout(tmp_path)
    res = handle_remember(tmp_path, [])
    assert (res["accepted"], res["rejected"]) == (0, 0)
    assert store.queue_paths(tmp_path) == []


def test_recall_truth_and_hypothesis_split(tmp_path):
    store.ensure_layout(tmp_path)
    store.enqueue_event(
        tmp_path,
        "[EVENT] t\n[SESSION] s\n"
        "[FACT] the level advances on box entry [[ls20]]\n"
        "[ASSUME] the block is me [[ls20]]\n",
    )
    process_queue(tmp_path)
    ctx = handle_recall(tmp_path)
    assert "[FACT]" in ctx and "box entry" in ctx
    assert "Hypotheses (UNVERIFIED" in ctx
    assert "the block is me" in ctx
    # strip hypotheses: guesses vanish, facts stay
    truth = handle_recall(tmp_path, include_hypotheses=False)
    assert "box entry" in truth
    assert "the block is me" not in truth


def test_search_matches_after_crystallization(tmp_path):
    store.ensure_layout(tmp_path)
    store.git_init_repo(tmp_path)
    store.enqueue_event(
        tmp_path,
        "[EVENT] t\n[SESSION] s\n"
        "[REFLECT] memory revision should be async, never blocking =>[[zixi-reasoning]]\n",
    )
    process_queue(tmp_path)
    node = tmp_path / "memory" / "zixi-reasoning.md"
    assert node.exists()
    hits = handle_search(tmp_path, "async")
    assert "async" in hits
