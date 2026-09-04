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

from . import backends, parser
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

    # Absorb done. No "remember this"-phrase heuristics; no narrative
    # guessing. Events WITHOUT primitives are not folded here (the caller
    # decides: deterministic drop, or LLM distill — see process_event).

    # Assemble: non-state lines first, STATEs last (current position sits
    # at the bottom where readers land).
    out = non_state + [""] + [state_blocks[s] for s in state_order]
    return "\n".join(out).rstrip() + "\n"


_FAST_WORKER_PROMPT = """You maintain ACTIVE.md, a snapshot of current cognition.

INPUT: the current ACTIVE.md snapshot (may already contain primitive
lines) and a conversation turn as prose.

RULES:
- Preserve every existing primitive line in ACTIVE.md. You do NOT
  rewrite them; they were folded deterministically before this call.
- Distill the conversation turn into AT MOST 2-4 ADDITIONAL primitives
  ([FACT]/[STATE]/[REASONING]/[REFLECT]/[ASSUME]/[LAB]/[SKILL]) — only
  durable knowledge for future turns. No trivia, no finished tasks, no
  chit-chat. When the turn has nothing durable, return ACTIVE unchanged.
- Never store prose in the output. Only primitives.
- [ASSUME] for unverified beliefs; [REFLECT] for lessons changing future
  behavior; [SKILL] for verified reusable how-to; [STATE] for current
  situation; [FACT] for verified fact.
- Return the COMPLETE new ACTIVE.md. Markdown only. No fences."""


def _llm_update(active: str, event_text: str) -> str:
    """LLM distillation for conversation events (no primitives inside).

    Slow path — the daemon only takes it when the event carries no
    explicit primitives; primitive-rich events never pay this (fast path).
    """
    from . import backends as bb

    user = (
        "## Current ACTIVE.md\n\n"
        f"{active}\n\n"
        "## New event\n\n"
        f"{event_text}\n\n"
        "Return the complete new ACTIVE.md. Markdown only."
    )
    new_active = bb.complete(_FAST_WORKER_PROMPT, user, max_tokens=8192)
    text = new_active.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    if not text:
        raise RuntimeError("fast worker returned empty ACTIVE.md")
    return text + ("\n" if not text.endswith("\n") else "")


def process_event(active: str, event_text: str, *, llm: bool = False) -> str:
    """Return the NEW complete ACTIVE.md for this event.

    Two-lane design (2026-09-04 user correction: background listener):
      * llm=False: deterministic fold — absorbs line-start primitives,
        drops everything else. Zero LLM.
      * llm=True: the daemon's listener path. Primitives fold
        deterministically, then the LLM distills the turn's prose into
        additional primitives (only when it carries durable knowledge;
        trivia yields no change). Explicit primitives ALWAYS win the
        fast lane — the LLM never reinterprets them, it only distills
        the untagged remainder.
    """
    if not llm:
        return _rules_update(active, event_text)
    if not backends.llm_ready():
        logger.warning("llm requested but Hermes client unavailable; fast fold only")
        return _rules_update(active, event_text)
    try:
        # 1. fold explicit primitives deterministically — the agent's own
        #    tagged lines are the floor, never re-interpreted
        active = _rules_update(active, event_text)
        # 2. distill the untagged prose of the turn with the LLM; it gets
        #    the already-folded snapshot and returns the complete new one
        prose = _strip_fences(event_text)
        if not prose.strip():
            return active
        return _llm_update(active, prose)
    except Exception as exc:  # noqa: BLE001 — never kill digestion
        logger.warning("llm distill failed (%s); fast fold kept", exc)
        return _rules_update(active, event_text)


def _strip_fences(event_text: str) -> str:
    """Remove the [USER]/[ASSISTANT] fence lines for LLM distilling.

    The primitives are ALREADY folded before this call; the fences exist
    only to mark roles for the daemon. The LLM gets the prose body.
    """
    out: list[str] = []
    for ln in event_text.splitlines():
        if ln.strip() in ("[USER]", "[ASSISTANT]", "[EVENT]", "") or ln.startswith("[SESSION]"):
            continue
        if ln.startswith("[USER]"):
            out.append(ln[len("[USER]"):].strip())
        elif ln.startswith("[ASSISTANT]"):
            out.append(ln[len("[ASSISTANT]"):].strip())
        else:
            out.append(ln)
    return "\n".join(out)
