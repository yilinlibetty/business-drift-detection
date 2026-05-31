"""Unified LLM client wrapper.

Auto-detects Anthropic vs OpenAI based on env vars, then exposes an
OpenAI-SDK-compatible ``chat.completions.create`` interface so the rest of
the codebase (`evaluation.py`, `llm_analyst_official.py`,
`run_llm_evaluation.py`) does not need to branch on provider.

Resolution order:
    1. ANTHROPIC_API_KEY set   -> anthropic SDK (wrapped to look like OpenAI)
    2. else                    -> openai SDK (with optional OPENAI_BASE_URL)

Model defaults per tier (override with env vars):
    high   -> high-quality model used for analyst + judge
    low    -> cheaper / faster model used for the extract step
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Defaults — verified against Shopee compass gateway and standard endpoints
# ---------------------------------------------------------------------------


ANTHROPIC_DEFAULTS = {
    "high": os.getenv("ANTHROPIC_MODEL_HIGH", "claude-sonnet-4-5@20250929"),
    "low":  os.getenv("ANTHROPIC_MODEL_LOW",  "claude-3-haiku@20240307"),
}

OPENAI_DEFAULTS = {
    "high": os.getenv("OPENAI_MODEL_HIGH", os.getenv("OPENAI_MODEL", "gpt-4o")),
    "low":  os.getenv("OPENAI_MODEL_LOW",  os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
}


def detect_provider() -> str:
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def default_model(tier: str = "high") -> str:
    if tier not in {"high", "low"}:
        raise ValueError(f"tier must be 'high' or 'low', got {tier!r}")
    provider = detect_provider()
    if provider == "anthropic":
        return ANTHROPIC_DEFAULTS[tier]
    return OPENAI_DEFAULTS[tier]


# ---------------------------------------------------------------------------
# OpenAI-shaped response objects (so callers can do .choices[0].message.content)
# ---------------------------------------------------------------------------


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list = field(default_factory=list)
    usage: Any = None


# ---------------------------------------------------------------------------
# Anthropic wearing an OpenAI mask
# ---------------------------------------------------------------------------


class _AnthropicOpenAICompat:
    """Adapter: Anthropic Messages API -> OpenAI ChatCompletion shape.

    Supports only ``client.chat.completions.create(model, messages, temperature, max_tokens, ...)``.
    Other OpenAI SDK surface (streaming, tools, etc.) is not implemented; add
    here when needed.
    """

    def __init__(self):
        import anthropic
        self._inner = anthropic.Anthropic()
        # OpenAI SDK access pattern: client.chat.completions.create(...)
        self.chat = self
        self.completions = self

    def create(self, *, model, messages, temperature=0.5, max_tokens=4096, **_) -> _Response:
        system_parts = []
        anth_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                anth_msgs.append({"role": role, "content": content})
            else:
                raise ValueError(f"unsupported role {role!r} (only system/user/assistant)")
        system_prompt = "\n\n".join(s for s in system_parts if s).strip()
        msg = self._inner.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt if system_prompt else "",
            messages=anth_msgs,
        )
        text = "".join(getattr(block, "text", "") for block in msg.content)
        return _Response(
            choices=[_Choice(message=_Message(content=text))],
            usage=msg.usage,
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def get_client(api_key: str | None = None, base_url: str | None = None):
    """Return a chat client that exposes ``.chat.completions.create``.

    Provider is decided by ``detect_provider()``. ``api_key`` / ``base_url``
    are honored only for the OpenAI path; Anthropic uses its own SDK env
    vars (``ANTHROPIC_API_KEY``, optional ``ANTHROPIC_BASE_URL``).
    """
    provider = detect_provider()
    if provider == "anthropic":
        return _AnthropicOpenAICompat()
    # OpenAI / OpenAI-compatible
    from openai import OpenAI
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
    )
