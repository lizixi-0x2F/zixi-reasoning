"""Zixi.Reasoning LLM backend — OpenAI-compatible chat completions via httpx.

Why not the openai SDK: some reasoning models (DeepSeek v4 line) return an
empty `content` with the actual answer in `reasoning_content`; the SDK's
`content=None` path then crashes. We stay on raw JSON and fall back.

Config (env vars only — bring your own model, no config files):
    ZIXI_BACKEND=rules|llm           default: llm (falls back to rules if no key)
    ZIXI_LLM_BASE_URL               default: api.deepseek.com/v1
    ZIXI_LLM_API_KEY                default: $DEEPSEEK_API_KEY (Hermes' own key —
                                    see llm_config: we also read $HERMES_HOME/.env)
    ZIXI_LLM_MODEL                  default: deepseek-v4-pro
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_TOKENS = 8192

_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def _read_hermes_env() -> dict[str, str]:
    """Best-effort read of $HERMES_HOME/.env so the daemon can share the
    Hermes key in standalone runs (systemd, tmux, no .bashrc)."""
    home = os.environ.get("HERMES_HOME")
    if not home:
        home = str(Path("~/.hermes").expanduser())
    env_path = Path(home) / ".env"
    out: dict[str, str] = {}
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            m = _ENV_LINE_RE.match(line.strip())
            if not m:
                continue
            value = m.group(2).strip().strip('"').strip("'")
            out[m.group(1)] = value
    except OSError:
        pass
    return out


def backend_mode() -> str:
    mode = os.environ.get("ZIXI_BACKEND", "llm").strip().lower()
    return mode if mode in ("rules", "llm") else "llm"


def llm_config() -> tuple[str, str, str]:
    """Return (base_url, api_key, model) honoring env + Hermes' own key."""
    base = os.environ.get("ZIXI_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    key = os.environ.get("ZIXI_LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        key = _read_hermes_env().get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("ZIXI_LLM_MODEL", DEFAULT_MODEL)
    return base, key, model


def llm_ready() -> bool:
    base, key, model = llm_config()
    return bool(key) and bool(base) and bool(model)


def complete(
    system: str,
    user: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    """One chat completion. Returns the model's final text (never None)."""
    base, key, mdl = llm_config()
    base = (base_url or base).rstrip("/")
    key = api_key or key
    mdl = model or mdl
    if not key:
        raise RuntimeError("Zixi LLM backend requested but no API key configured")

    payload = {
        "model": mdl,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(
                    f"{base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"LLM returned no choices: {str(data)[:300]}")
            msg = choices[0].get("message") or {}
            text = (msg.get("content") or "").strip()
            if text:
                return text
            # Fallback: reasoning-style responses keep the answer here
            rc = (msg.get("reasoning_content") or "")
            if rc.strip():
                logger.warning("LLM returned empty content; using reasoning_content")
                return rc.strip()
            raise RuntimeError(f"LLM returned empty message: {str(data)[:300]}")
        except Exception as exc:  # noqa: BLE001 — backend must never crash the daemon
            last_err = exc
            delay = min(2 ** attempt, 6)
            time.sleep(delay)
    raise RuntimeError(f"Zixi LLM call failed after {retries} attempts: {last_err}")
