"""Zixi.Reasoning parser — the entire syntax.

Recognizes exactly:

    [FACT] ... | [STATE] ... | [REASONING] ... | [REFLECT] ...
    [ASSUME] ... | [LAB] ...
    [SKILL] ...
    [[WikiLink]]
    ->[STATE] ...
    =>[[Node]]

Everything else is plain markdown (headings, blank lines, prose) and is
preserved. The parser never throws; unknown lines are returned as-is.

[ASSUME]/[LAB] are the hypothesis layer: unverified beliefs and the
experiments that probe them. They are intentionally kept OUT of the truth
zone — recall groups them separately (see recall.compile_context) so a
guessed world-model can never be injected as an established fact.
[SKILL] is procedural memory (reusable how-to knowledge): truth zone,
cumulative, same injection path as FACT/REFLECT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

TAG_RE = re.compile(r"^(\s*)\[(FACT|STATE|REASONING|REFLECT|ASSUME|LAB|SKILL)\]\s*(.*)$")
STATE_ARROW_RE = re.compile(r"->\[STATE\]\s*([^\n]*)")
CONSOLIDATE_RE = re.compile(r"=>\[\[([^\n\]]+)\]\]")
WIKILINK_RE = re.compile(r"\[\[([^\]\n]+?)\]\]")

TAGS = ("FACT", "STATE", "REASONING", "REFLECT", "ASSUME", "LAB", "SKILL")
# Hypothesis layer: current guesses + the experiments probing them.
# They are state-like (snapshot semantics) but NOT truth-zone tokens.
HYPOTHESES = ("ASSUME", "LAB")
# SKILL is truth-zone procedural memory (reusable how-to knowledge):
# cumulative like FACT/REFLECT, never folded, shares the injection path.


@dataclass
class Element:
    """One parsed line carrying cognitive content."""

    kind: str | None            # FACT|STATE|REASONING|REFLECT or None (plain line)
    text: str                   # tag body, or '' for plain lines
    links: list[str] = field(default_factory=list)
    state_transitions: list[str] = field(default_factory=list)
    consolidate_targets: list[str] = field(default_factory=list)
    raw: str = ""
    line_no: int = 0

    def is_cognitive(self) -> bool:
        return self.kind in TAGS


def _strip_tail(text: str) -> str:
    """Clean a tag body: remove trailing arrows/consolidations and any
    trailing wikilink cluster (links are carried as structured fields)."""
    out = text
    # Remove =>[[...]] anywhere
    out = CONSOLIDATE_RE.sub("", out)
    # Remove ->[STATE] ... runs (up to end of line)
    out = STATE_ARROW_RE.sub("", out)
    # Remove a trailing "  [[L1]] [[L2]]" cluster (links are structured)
    out = re.sub(r"\s+\[\[[^\]\n]+\]\](\s+\[\[[^\]\n]+\]\]\s*)*$", "", out)
    return out.strip()


def parse_text(text: str) -> list[Element]:
    """Parse a markdown text into a list of Element. Line-based, lossless."""
    elements: list[Element] = []
    for no, raw in enumerate(text.splitlines(), start=1):
        m = TAG_RE.match(raw)
        if not m:
            # Separate-line arrows still carry cognition (spec §5).
            transitions = [s.strip() for s in STATE_ARROW_RE.findall(raw) if s.strip()]
            consolidate_targets = [t.strip() for t in CONSOLIDATE_RE.findall(raw)]
            elements.append(
                Element(
                    kind=None,
                    text="",
                    links=[l.strip() for l in WIKILINK_RE.findall(raw)],
                    state_transitions=transitions,
                    consolidate_targets=consolidate_targets,
                    raw=raw,
                    line_no=no,
                )
            )
            continue
        kind, body = m.group(2), m.group(3)
        links = WIKILINK_RE.findall(raw)
        state_transitions = [s.strip() for s in STATE_ARROW_RE.findall(body) if s.strip()]
        consolidate_targets = [t.strip() for t in CONSOLIDATE_RE.findall(raw)]
        elements.append(
            Element(
                kind=kind,
                text=_strip_tail(body),
                links=[l.strip() for l in links],
                state_transitions=state_transitions,
                consolidate_targets=consolidate_targets,
                raw=raw,
                line_no=no,
            )
        )
    return elements


def extract_content(text: str) -> dict[str, list[Element]]:
    """Split a parsed text into {facts, states, reasonings, reflects, assumes, labs}."""
    buckets: dict[str, list[Element]] = {t: [] for t in TAGS}
    for el in parse_text(text):
        if el.kind in buckets:
            buckets[el.kind].append(el)
    return buckets


TAG_PREFIX_RE = re.compile(r"^\s*\[(FACT|STATE|REASONING|REFLECT|ASSUME|LAB|SKILL)\]\s*(.*)$")


def extract_primitive_lines(text: str) -> list[str]:
    """Pull lines that START with a primitive tag.

    The sole ingestion gate (2026-09-04): memory enters only through
    explicitly tagged lines; narrative/prose at any other position is
    never captured, never read as memory material, never synthesized.
    Shared by the provider (sync_turn), the ARC players, and the daemon.
    """
    out: list[str] = []
    for ln in text.splitlines():
        if TAG_PREFIX_RE.match(ln):
            out.append(ln.strip())
    return out


def is_hypothesis(kind: str | None) -> bool:
    """True for [ASSUME]/[LAB] — the hypothesis layer, never truth-zone tokens."""
    return kind in HYPOTHESES


def collect_links(text: str) -> list[str]:
    """All [[WikiLink]] targets in order of appearance, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for m in WIKILINK_RE.finditer(text):
        t = m.group(1).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def collect_outgoing_consolidations(text: str) -> list[str]:
    """All =>[[Target]] targets in order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for m in CONSOLIDATE_RE.finditer(text):
        t = m.group(1).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def make_tag(
    kind: str,
    text: str,
    links: list[str] | None = None,
    consolidate_targets: list[str] | None = None,
) -> str:
    """Render one canonical primitive line."""
    assert kind in TAGS, f"unknown tag {kind}"
    line = f"[{kind}] {text}".rstrip()
    if links:
        line += "  " + " ".join(f"[[{l}]]" for l in links)
    if consolidate_targets:
        line += "  " + " ".join(f"=>[[{t}]]" for t in consolidate_targets)
    return line


# ---------------------------------------------------------------------------
# Slugs: [[Wiki Link]] -> Wiki-Link.md  (see spec §6)
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "-", s).strip("-")
    return s or "untitled"
