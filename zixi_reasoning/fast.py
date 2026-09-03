"""Zixi.Reasoning fast worker — turns events into ACTIVE.md.

ACTIVE.md is a snapshot of current cognition, not a journal (spec §10).
The worker therefore *replaces* the file every time; it is never appended to.

Two backends:
  * rules — deterministic. Absorbs explicit primitives from the event, keeps
    one STATE per subject (same first [[wiki-link]], or same short text prefix),
    and turns unstructured observation into a short [FACT]. No summarization:
    that is the LLM's job.
  * llm   — the full worker prompt (spec §26): model returns the COMPLETE new
    ACTIVE.md, we atomic-replace. This is where stale states die and REFLECTs
    are minted.
"""

from __future__ import annotations

import logging

from . import backends, parser, store
from .parser import Element

logger = logging.getLogger(__name__)

_FACT_CUTOFF = 220
_STATE_SUBJECT_PREFIX = 16

_FAST_WORKER_PROMPT = """You maintain ACTIVE.md.

Represent only the agent's current cognitive state.

Allowed primitives:
[FACT]
[STATE]
[REASONING]
[REFLECT]
[[link]]
->[STATE]
=>[[memory]]

Do not summarize the conversation.
Do not preserve trivia.
Remove stale states.
Preserve unresolved goals and constraints.
Create REFLECT only when the event changes future behavior.
Use => only when information deserves crystallized memory.
If the user explicitly asks you to remember something, crystallize it with =>.
Return the complete new ACTIVE.md."""


def _flatten(text: str, limit: int = _FACT_CUTOFF) -> str:
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _subject_of_state(el: Element) -> str:
    if el.links:
        return el.links[0]
    return el.text[:_STATE_SUBJECT_PREFIX]


def _rules_update(active: str, event_text: str) -> str:
    """Deterministic transform: absorb explicit primitives, replace same-subject STATE.

    Simple and honest: it cannot summarize, abstract, or detect contradiction
    (that is the LLM backend); it keeps the active state machine sound.
    """
    lines = active.splitlines()

    # Pass 1: fold existing file — keep non-state lines as-is; collapse
    # STATEs so each subject holds only its latest declaration.
    non_state: list[str] = []
    state_blocks: dict[str, str] = {}
    state_order: list[str] = []
    for ln in lines:
        els = parser.parse_text(ln)
        state_els = [e for e in els if e.kind == "STATE"]
        if state_els:
            for e in state_els:
                s = _subject_of_state(e)
                if s not in state_blocks:
                    state_order.append(s)
                state_blocks[s] = parser.make_tag("STATE", e.text, e.links)
            continue
        non_state.append(ln)

    # Pass 2: absorb the event's explicit primitives (unstructured text
    # becomes one short [FACT]).
    event_els = parser.parse_text(event_text)
    had_explicit = False
    for el in event_els:
        if el.kind == "STATE":
            had_explicit = True
            s = _subject_of_state(el)
            state_blocks[s] = parser.make_tag("STATE", el.text, el.links)
            if s not in state_order:
                state_order.append(s)
        elif el.kind in ("FACT", "REASONING", "REFLECT"):
            had_explicit = True
            non_state.append(
                parser.make_tag(el.kind, el.text, el.links, el.consolidate_targets)
            )
        elif el.state_transitions:
            for st in el.state_transitions:
                had_explicit = True
                s = st[:_STATE_SUBJECT_PREFIX]
                state_blocks[s] = parser.make_tag("STATE", st)
                if s not in state_order:
                    state_order.append(s)

    if not had_explicit:
        non_state.append(parser.make_tag("FACT", _flatten(event_text)))
    else:
        # Spec §17 class 5: an explicit "remember this" is a direct
        # crystallization request — no heuristic gate.
        for phrase in ("记住这个", "请记住", "记住以下", "记一下", "remember this"):
            if phrase in event_text:
                idx = event_text.find(phrase)
                content = event_text[idx + len(phrase):].strip()
                if content:
                    non_state.append(
                        parser.make_tag(
                            "REFLECT",
                            _flatten(content, 500),
                            consolidate_targets=["Memories"],
                        )
                    )
                break

    # Assemble: non-state lines first, STATEs last (current position sits
    # at the bottom where readers land).
    out = non_state + [""] + [state_blocks[s] for s in state_order]
    return "\n".join(out).rstrip() + "\n"


def _llm_update(active: str, event_text: str) -> str:
    from . import backends as bb

    user = (
        "## Current ACTIVE.md\n\n"
        f"{active}\n\n"
        "## New event\n\n"
        f"{event_text}\n\n"
        "Return the complete new ACTIVE.md. Markdown only."
    )
    new_active = bb.complete(_FAST_WORKER_PROMPT, user)
    # Defensive: if the model wrapped a code fence, unwrap it.
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


def process_event(active: str, event_text: str) -> str:
    """Return the NEW complete ACTIVE.md for this event."""
    if backends.backend_mode() == "llm":
        if not backends.llm_ready():
            logger.warning("ZIXI_BACKEND=llm but Hermes model client unavailable; falling back to rules")
            return _rules_update(active, event_text)
        return _llm_update(active, event_text)
    return _rules_update(active, event_text)
