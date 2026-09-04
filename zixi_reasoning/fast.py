# ---------------------------------------------------------------------------
# Fast worker — turns events into ACTIVE.md (deterministic fold only).
# ---------------------------------------------------------------------------
# ACTIVE.md is a snapshot of current cognition, not a journal (spec §10).
# The worker *replaces* the file every time; it is never appended to.
#
# PRIMITIVES-ONLY (2026-09-04 user rulings): ingestion is a pure
# deterministic fold — no LLM backend, no narrative-to-FACT flattening, no
# "remember this"-phrase heuristics. Only the agent's explicitly
# self-reported primitive lines ([FACT]/[STATE]/[REASONING]/[REFLECT]/
# [ASSUME]/[LAB]/[SKILL]) are absorbed; everything else is dropped.

from __future__ import annotations

import logging

from . import parser
from .parser import Element

logger = logging.getLogger(__name__)

_STATE_SUBJECT_PREFIX = 16
_STATE_LIKE = ("STATE", "ASSUME", "LAB")   # snapshot semantics (state-like folding)


def _subject_of_state(el: Element) -> str:
    if el.links:
        return el.links[0]
    return el.text[:_STATE_SUBJECT_PREFIX]


def fold_state_like(lines: list[str]) -> tuple[list[str], dict[tuple[str, str], str], list[tuple[str, str]]]:
    """Fold a line list into (non_state_lines, state_blocks, state_order).

    Shared by the rules worker and the patch applier: state-like lines
    (STATE/ASSUME/LAB) collapse to one line per (kind, subject); everything
    else is kept verbatim.
    """
    non_state: list[str] = []
    state_blocks: dict[tuple[str, str], str] = {}
    state_order: list[tuple[str, str]] = []
    for ln in lines:
        els = parser.parse_text(ln)
        state_els = [e for e in els if e.kind in _STATE_LIKE]
        if state_els:
            for e in state_els:
                assert e.kind is not None  # narrowed by the filter above
                key = (e.kind, _subject_of_state(e))
                if key not in state_blocks:
                    state_order.append(key)
                state_blocks[key] = parser.make_tag(e.kind, e.text, e.links)
            continue
        non_state.append(ln)
    return non_state, state_blocks, state_order


# ---------------------------------------------------------------------------
# Primitive-only digestion (2026-09-04, user ruling: [OBS] never enters)
# ---------------------------------------------------------------------------
# The ledger accepts ONLY the agent's explicitly self-reported primitive
# lines ([ASSUME]/[LAB]/[FACT]/[STATE]/[REASONING]/[REFLECT]/[SKILL]).
# Observation narrative never enters: neither directly nor as material an
# LLM may mine. Digestion is therefore a pure deterministic fold — no LLM
# in the ingestion path at all, no heuristic synthesis anywhere.

_PRIMITIVE_IN_TEXT = ("[ASSUME]", "[LAB]", "[SKILL]", "[REFLECT]", "[FACT]", "[REASONING]", "[STATE]")


def has_primitives(event_text: str) -> bool:
    """True when the event contains at least one line that STARTS with a tag.

    Strict line-start check (not substring): mentioning "[ASSUME]" in
    prose does not open the gate — only an actual primitive line does.
    """
    return bool(parser.extract_primitive_lines(event_text))


def _rules_update(active: str, event_text: str) -> str:
    """Deterministic transform: absorb explicit primitives, replace same-subject STATE.

    Simple and honest: it cannot summarize, abstract, or detect contradiction
    (that is the LLM backend); it keeps the active state machine sound.
    """
    lines = active.splitlines()

    # Pass 1: fold existing file — keep non-state lines as-is; collapse
    # state-like lines (STATE/ASSUME/LAB) so each (tag, subject) pair holds
    # only its latest declaration.
    non_state: list[str] = []
    state_blocks: dict[tuple[str, str], str] = {}
    state_order: list[tuple[str, str]] = []
    for ln in lines:
        els = parser.parse_text(ln)
        state_els = [e for e in els if e.kind in _STATE_LIKE]
        if state_els:
            for e in state_els:
                assert e.kind is not None  # narrowed by the filter above
                key = (e.kind, _subject_of_state(e))
                if key not in state_blocks:
                    state_order.append(key)
                state_blocks[key] = parser.make_tag(e.kind, e.text, e.links)
            continue
        non_state.append(ln)

    # Pass 2: absorb the event's explicit primitives. Nothing else is
    # absorbed: no narrative-to-FACT synthesis (user ruling: [OBS] never
    # enters), no guessing from prose.
    event_els = parser.parse_text(event_text)
    for el in event_els:
        if el.kind in _STATE_LIKE:
            assert el.kind is not None  # narrowed by the branch above
            key = (el.kind, _subject_of_state(el))
            state_blocks[key] = parser.make_tag(el.kind, el.text, el.links)
            if key not in state_order:
                state_order.append(key)
        elif el.kind in ("FACT", "REASONING", "REFLECT", "SKILL"):
            non_state.append(
                parser.make_tag(el.kind, el.text, el.links, el.consolidate_targets)
            )
        elif el.state_transitions:
            for st in el.state_transitions:
                s = st[:_STATE_SUBJECT_PREFIX]
                state_blocks[("STATE", s)] = parser.make_tag("STATE", st)
                if ("STATE", s) not in state_order:
                    state_order.append(("STATE", s))

    # Absorb done. No "remember this"-phrase heuristics, no narrative guesses:
    # memory enters ONLY through the agent's own primitive lines.

    # Assemble: non-state lines first, STATEs last (current position sits
    # at the bottom where readers land).
    out = non_state + [""] + [state_blocks[s] for s in state_order]
    return "\n".join(out).rstrip() + "\n"


def process_event(active: str, event_text: str) -> str:
    """Return the NEW complete ACTIVE.md for this event.

    Deterministic primitive-only fold (no LLM backend since 2026-09-04:
    ingestion is zero-LLM by design).
    """
    return _rules_update(active, event_text)
