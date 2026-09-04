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
import re

from . import backends, parser, store
from .parser import Element

logger = logging.getLogger(__name__)

_FACT_CUTOFF = 220
_STATE_SUBJECT_PREFIX = 16
_STATE_LIKE = ("STATE", "ASSUME", "LAB")   # snapshot semantics (state-like folding)

_FAST_WORKER_PROMPT = """You maintain ACTIVE.md.

Represent only the agent's current cognitive state.

Allowed primitives:
[FACT]
[STATE]
[REASONING]
[REFLECT]
[ASSUME]
[LAB]
[SKILL]
[[link]]
->[STATE]
=>[[memory]]

[ASSUME] = an UNVERIFIED working belief. Keep it only while it is being
tested. Once verified, promote the verified conclusion to [FACT] or
[REFLECT] and remove the ASSUME; once refuted or stale, remove it.
Never rewrite an [ASSUME] as [FACT] unless the event shows it was verified.
[LAB] = a probe experiment ("tested X, observed Y"). Snapshot semantics,
replace same-subject LABs with the latest.
[SKILL] = procedural memory: a reusable verified how-to (method, protocol,
procedure). Truth zone, cumulative — keep it while the procedure stays
valid; fold duplicates into one line. [SKILL] lines may carry => to
crystallize the procedure.

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
# Thin/rich classification + patch applier (2026-09-04 digestion re-arch)
# ---------------------------------------------------------------------------
# The old worker paid one full LLM rewrite (30s) per event. The new pipeline:
#   thin events   -> apply_rules_event (zero LLM, milliseconds)
#   rich events   -> parallel LLM patch workers (only +/- keyed lines, ~1KB)
#                    then apply_patch serially on the single writer thread.
# apply_patch is deterministic, idempotent and order-insensitive per
# (kind, subject): worker threads never touch ACTIVE.md directly.

_THIN_OBS_CUTOFF = 80
_PATCH_PREFIX_RE = re.compile(r"^\s*([+-])\s*(\[(?:FACT|STATE|REASONING|REFLECT|ASSUME|LAB|SKILL)\].*)$")
_PRIMITIVE_IN_TEXT = ("[ASSUME]", "[LAB]", "[SKILL]", "[REFLECT]", "[FACT]", "[REASONING]")


def is_thin_event(event_text: str) -> bool:
    """True when an event carries nothing an LLM could turn into memory.

    ARC heartbeats whose [OBS] is a placeholder ("I'll observe the result
    of ACTION3.") and that carry no primitive lines are information-less:
    their only increment is the [STEP] status line. Zero-LLM fold.
    """
    if any(t in event_text for t in _PRIMITIVE_IN_TEXT):
        return False
    obs = event_text.split("[OBS]", 1)[-1] if "[OBS]" in event_text else event_text
    return len(obs.strip()) < _THIN_OBS_CUTOFF


def apply_rules_event(active: str, event_text: str) -> str:
    """Deterministic fold for events carrying explicit primitives.

    No heuristic synthesis: if the event has no primitive lines, this call
    is not made (the caller drops the event — thin events are information-
    less, nothing is invented for them).
    """
    assert any(t in event_text for t in _PRIMITIVE_IN_TEXT)
    return _rules_update(active, event_text)


_PATCH_WORKER_PROMPT = """You maintain ACTIVE.md as a snapshot of current cognition.

You are shown the CURRENT ACTIVE.md plus ONE new event. Produce the EXACT
line-level changes this event implies. Never rewrite the file; never echo
unchanged lines.

Patch syntax — one line per change:
  + [TAG] text      upsert (add or replace by subject)
  - [TAG] text      remove (by subject, or identical text)
TAG is one of FACT|STATE|REASONING|REFLECT|ASSUME|LAB|SKILL.
[[WikiLink]] may follow the text; =>[[Node]] may follow for crystallizable
lessons (REFLECT/SKILL only).

Semantics:
- STATE/ASSUME/LAB upsert by (tag, subject): subject = the first [[link]],
  else the first 16 chars of the text. A new same-subject line replaces the
  old one.
- FACT/REASONING/REFLECT/SKILL append unless an identical line exists.
- Do NOT paraphrase facts the event does not support. State unknowns as
  [ASSUME]. Keep it minimal: one line per change, nothing else.
- If nothing needs to change, output exactly: (no change)
Output ONLY patch lines. No fences, no prose, no commentary."""


def parse_patch(text: str) -> list[tuple[str, Element]]:
    """Parse a patch response into [(op, element)] preserving order.

    STRICT: every non-blank line must be a well-formed patch line. If any
    line is not (prose, fences, commentary, a partial echo of ACTIVE), the
    whole response is rejected with [] — never a heuristic partial apply.
    """
    ops: list[tuple[str, Element]] = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        m = _PATCH_PREFIX_RE.match(ln)
        if not m:
            return []
        els = parser.parse_text(m.group(2))
        if not els or not els[0].kind:
            return []
        ops.append((m.group(1), els[0]))
    return ops


def _same_kind_text(line: str, el: Element) -> bool:
    for e in parser.parse_text(line):
        if e.kind == el.kind and e.text == el.text:
            return True
    return False


def _normalize_block(lines: list[str]) -> list[str]:
    """Collapse blank-line runs to one and strip leading/trailing blanks.

    Makes the rebuilt ACTIVE canonical: applying the same patch twice must
    be a fixpoint (blank lines must not accumulate across folds).
    """
    out: list[str] = []
    prev_blank = True
    for ln in lines:
        if not ln.strip():
            if prev_blank:
                continue
            prev_blank = True
            out.append("")
        else:
            prev_blank = False
            out.append(ln)
    while out and not out[-1].strip():
        out.pop()
    return out


def apply_patch(active: str, patch_text: str) -> str:
    """Apply keyed patch lines to ACTIVE. Deterministic, idempotent.

    '-' then '+' order per key cannot clobber: removals are computed on the
    snapshot, upserts applied after. Non-cognitive lines (headings, prose)
    are untouched — the patch language has no affordance for them.
    """
    ops = parse_patch(patch_text)
    if not ops:
        return active
    non_state, blocks, order = fold_state_like(active.splitlines())

    for op, el in ops:
        if op == "-":
            if el.kind in _STATE_LIKE:
                key = (el.kind, _subject_of_state(el))
                blocks.pop(key, None)
            else:
                non_state = [ln for ln in non_state if not _same_kind_text(ln, el)]
        else:  # '+'
            assert el.kind is not None  # narrowed by parse_patch
            if el.kind in _STATE_LIKE:
                key = (el.kind, _subject_of_state(el))
                if key not in blocks:
                    order.append(key)
                blocks[key] = parser.make_tag(el.kind, el.text, el.links)
            else:
                if not any(_same_kind_text(ln, el) for ln in non_state):
                    non_state.append(parser.make_tag(el.kind, el.text, el.links, el.consolidate_targets))
    body = _normalize_block(non_state)
    out = body + ([""] if body else []) + [blocks[s] for s in order if s in blocks]
    return "\n".join(_normalize_block(out)).rstrip() + "\n"


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

    # Pass 2: absorb the event's explicit primitives (unstructured text
    # becomes one short [FACT]).
    event_els = parser.parse_text(event_text)
    had_explicit = False
    for el in event_els:
        if el.kind in _STATE_LIKE:
            had_explicit = True
            assert el.kind is not None  # narrowed by the branch above
            key = (el.kind, _subject_of_state(el))
            state_blocks[key] = parser.make_tag(el.kind, el.text, el.links)
            if key not in state_order:
                state_order.append(key)
        elif el.kind in ("FACT", "REASONING", "REFLECT", "SKILL"):
            had_explicit = True
            non_state.append(
                parser.make_tag(el.kind, el.text, el.links, el.consolidate_targets)
            )
        elif el.state_transitions:
            for st in el.state_transitions:
                had_explicit = True
                s = st[:_STATE_SUBJECT_PREFIX]
                state_blocks[("STATE", s)] = parser.make_tag("STATE", st)
                if ("STATE", s) not in state_order:
                    state_order.append(("STATE", s))

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
