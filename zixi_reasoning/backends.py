"""Zixi.Reasoning LLM backend — REUSES Hermes' own model client.

Zixi never holds a separate API key and never talks to an LLM endpoint on
its own at default. It calls Hermes' auxiliary_client.call_llm(), which
resolves provider/model/base_url/api_key from the Hermes configuration
(main model settings + $HERMES_HOME/.env) -- the exact same client, model,
and wallet the Hermes agent itself uses.

    ZIXI_BACKEND=rules|llm   default: llm (falls back to rules when the
                             Hermes client is unavailable)
    ZIXI_LLM_*               accepted for explicit overrides (provider,
                             model, base_url, api_key) -- meaning "as if
                             you had configured Hermes with those". Empty
                             by default: full reuse.

This keeps one key, one client, one billing stream. Memory work is just
another auxiliary task on Hermes' own model configuration.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 8192


def backend_mode() -> str:
    mode = os.environ.get("ZIXI_BACKEND", "llm").strip().lower()
    return mode if mode in ("rules", "llm") else "llm"


def _hermes_client():
    """Import Hermes' auxiliary LLM client (call_llm + extractor)."""
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    return call_llm, extract_content_or_reasoning


def _overrides() -> dict[str, str | None]:
    """Explicit ZIXI_LLM_* overrides; all None by default (full reuse)."""
    out: dict[str, str | None] = {}
    for key, name in (
        ("ZIXI_LLM_PROVIDER", "provider"),
        ("ZIXI_LLM_MODEL", "model"),
        ("ZIXI_LLM_BASE_URL", "base_url"),
        ("ZIXI_LLM_API_KEY", "api_key"),
    ):
        val = os.environ.get(key)
        if val:
            out[name] = val
    return out


def llm_ready() -> bool:
    """True when Hermes' model client can be imported (i.e. we run inside
    the Hermes environment or with Hermes' source tree on PYTHONPATH)."""
    try:
        _hermes_client()
        return True
    except Exception:  # noqa: BLE001 — import is the check
        return False


def complete(
    system: str,
    user: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    retries: int = 1,
) -> str:
    """One completion through Hermes' own model client.

    provider/model/base_url/api_key come from Hermes config unless the user
    explicitly set ZIXI_LLM_* overrides.
    """
    call_llm, extract_content_or_reasoning = _hermes_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, object] = _overrides()
            kwargs.update(
                task="zixi_memory",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            resp = call_llm(**kwargs)  # type: ignore[arg-type]
            text = extract_content_or_reasoning(resp).strip()
            if text:
                return text
            raise RuntimeError("Hermes LLM returned empty content")
        except Exception as exc:  # noqa: BLE001 — backend must never crash the daemon
            last_exc = exc
            logger.warning("zixi llm call failed: %s", exc)
    if retries <= 1:
        raise RuntimeError(f"zixi llm call failed: {last_exc}")
    raise RuntimeError(f"zixi llm call failed after {retries} attempts: {last_exc}")
