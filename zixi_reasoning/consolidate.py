"""Zixi.Reasoning consolidator — active -> crystallized memory.

    C(A, M) -> M'

Crystallization is NOT copying (spec §15): transient STATE becomes a stable
REFLECT or is dropped. Slow memory is revised, never append-only (spec §18):

    M_{t+1} = Revise(M_t, E_t)   not   M_{t+1} = M_t ∪ E_t

Two backends:
  * rules — deterministic: ADD (new reflection), DROP (exact duplicate).
  * llm   — the consolidator prompt (spec §27): the model reads the candidate
    plus the target file (and its 1-hop linked files), chooses between
    ADD / MERGE / REVISE / LINK / DROP, and returns the COMPLETE revised file.
"""

from __future__ import annotations

import logging

from . import backends, parser, store

logger = logging.getLogger(__name__)

_CONSOLIDATOR_PROMPT = """Integrate the candidate into long-term crystallized memory.

Do not append blindly.

Choose conceptually between:
add
merge
revise
link
discard

Remove obsolete claims.
Prefer stable abstractions over event narration.
Do not preserve temporary STATE unless it represents a durable project state.

Return the complete revised target Markdown file."""


def default_title(node: str) -> str:
    return "# " + node.strip()


def _duplicated(candidate: str, existing_text: str, kind: str = "REFLECT") -> bool:
    """Deterministic duplicate detector: candidate body (same kind) already present."""
    if not candidate.strip():
        return True
    # Compare on the normalized body of a single line (allow link suffix drift).
    for el in parser.parse_text(existing_text):
        if el.kind == kind and el.text.strip() == candidate.strip():
            return True
    return False


def _rules_consolidate(
    target: str, target_path, candidate_text: str, links: list[str], kind: str = "REFLECT"
) -> tuple[str, str]:
    """Returns (new_file_text, action)."""
    if not candidate_text.strip():
        return ("", "noop")
    if not target_path.exists():
        body = parser.make_tag(kind, candidate_text, links)
        return (default_title(target) + "\n\n" + body + "\n", "add")
    existing = target_path.read_text(encoding="utf-8")
    if _duplicated(candidate_text, existing, kind):
        return (existing, "drop")
    # ADD: keep the file as-is, append the new line + its links.
    new = existing.rstrip() + "\n\n" + parser.make_tag(kind, candidate_text, links) + "\n"
    return (new, "add")


def _llm_consolidate(
    target_path, candidate_text: str, links: list[str], linked_text: str, kind: str = "REFLECT"
) -> tuple[str, str]:
    user = (
        f"## Target file: {target_path.name}\n\n"
        f"{target_path.read_text(encoding='utf-8') if target_path.exists() else '(new file)'}\n\n"
        f"## 1-hop linked files\n\n{linked_text or '(none)'}\n\n"
        f"## Candidate ({kind})\n\n"
        f"{parser.make_tag(kind, candidate_text, links)}\n\n"
        "Return the complete revised target Markdown file. Markdown only."
    )
    revised = backends.complete(_CONSOLIDATOR_PROMPT, user)
    text = revised.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    if not text:
        raise RuntimeError("consolidator returned empty file")
    return (text + ("\n" if not text.endswith("\n") else ""), "revise")


def consolidate(
    root,
    target: str,
    candidate_text: str,
    links: list[str] | None = None,
    kind: str = "REFLECT",
) -> tuple[str, str]:
    """Consolidate one candidate of ``kind`` (REFLECT or SKILL) into node ``target``.

    Returns (message, action) where action in {add, drop, revise, noop}.
    """
    links = links or []
    target_path = store.memory_path(root, target)

    if backends.backend_mode() == "llm" and backends.llm_ready():
        linked_text = _one_hop_context(root, target, links)
        new_text, action = _llm_consolidate(target_path, candidate_text, links, linked_text, kind)
    else:
        new_text, action = _rules_consolidate(target, target_path, candidate_text, links, kind)

    if action not in ("drop", "noop") and new_text:
        store.atomic_write(target_path, new_text)
    return action, target


def _one_hop_context(root, target: str, links: list[str]) -> str:
    """Concatenate 1-hop linked node contents (bounded) for the LLM backends."""
    chunks: list[str] = []
    budget = 6000
    total = 0
    for name in (links or []):
        p = store.memory_path(root, name)
        if p.exists():
            text = p.read_text(encoding="utf-8")
            chunks.append(f"### {store.node_title(p)}\n\n{text}")
            total += len(text)
            if total >= budget:
                break
    return "\n\n".join(chunks)
