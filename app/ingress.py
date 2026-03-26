"""
Ingress Normalizer
------------------
Handles auth, schema parse, model detection, and splits the payload into:
  - system_prompt  (primary compression target)
  - history        (rolling window, subject to full pipeline)
  - user_message   (NEVER modified)
"""
from __future__ import annotations
import re
from typing import Optional
from fastapi import Header, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Supported model families and their tokenizer identifiers
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    # key: lowercase fragment found in model name
    "gpt-4o":     {"family": "openai",    "encoding": "o200k_base",  "ctx_window": 128_000},
    "gpt-4":      {"family": "openai",    "encoding": "cl100k_base", "ctx_window": 128_000},
    "gpt-3.5":    {"family": "openai",    "encoding": "cl100k_base", "ctx_window": 16_385},
    "gpt-5":      {"family": "openai",    "encoding": "o200k_base",  "ctx_window": 200_000},
    "o1":         {"family": "openai",    "encoding": "o200k_base",  "ctx_window": 128_000},
    "o3":         {"family": "openai",    "encoding": "o200k_base",  "ctx_window": 200_000},
    "claude-3-5": {"family": "anthropic", "encoding": "anthropic",   "ctx_window": 200_000},
    "claude-3":   {"family": "anthropic", "encoding": "anthropic",   "ctx_window": 200_000},
    "claude-2":   {"family": "anthropic", "encoding": "anthropic",   "ctx_window": 100_000},
    "claude":     {"family": "anthropic", "encoding": "anthropic",   "ctx_window": 200_000},
}


def detect_model(model_name: str) -> dict:
    """Return the registry entry for the best-matching model, or a safe default."""
    m = model_name.lower()
    # Longest-matching key wins (more specific first)
    for key in sorted(MODEL_REGISTRY.keys(), key=len, reverse=True):
        if key in m:
            return {**MODEL_REGISTRY[key], "model": model_name}
    return {"family": "openai", "encoding": "cl100k_base", "ctx_window": 128_000, "model": model_name}


# ---------------------------------------------------------------------------
# Payload splitting
# ---------------------------------------------------------------------------

class SplitPayload(BaseModel):
    model_config = {"protected_namespaces": ()}

    system_prompt: str = ""
    history: list[dict] = []      # list of {role, content}
    user_message: str = ""
    model_info: dict = {}


def _extract_text(content: object) -> str:
    """
    Normalise message content to a plain string.

    OpenAI simple format:   "hello"
    Multimodal / Cline:     [{"type": "text", "text": "hello"}, ...]
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    # Any other block type — grab any "text" key if present
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def split_messages(messages: list[dict], model: str = "gpt-4o") -> SplitPayload:
    """
    Split an OpenAI-style messages list into the three parts:
      system_prompt – content of the first system message (may be "")
      history       – all non-system messages except the final user message
      user_message  – the last user turn (untouched)

    Handles both plain-string content and multimodal content blocks
    (the format Cline and vision models use).
    """
    if not messages:
        return SplitPayload(model_info=detect_model(model))

    system_prompt = ""
    remaining: list[dict] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_text(msg.get("content"))
        if role == "system" and not system_prompt:
            system_prompt = content
        else:
            remaining.append({"role": role, "content": content})

    # The very last user message is sacrosanct — never compressed
    user_message = ""
    history: list[dict] = []

    if remaining:
        last = remaining[-1]
        if last.get("role") == "user":
            user_message = last["content"]
            history = remaining[:-1]
        else:
            # last message is assistant — preserve it all as history
            history = remaining

    return SplitPayload(
        system_prompt=system_prompt,
        history=history,
        user_message=user_message,
        model_info=detect_model(model),
    )


# ---------------------------------------------------------------------------
# Auth helper (bearer token or x-api-key, with env-based passthrough)
# ---------------------------------------------------------------------------

import os

_PROXY_API_KEY: str | None = os.environ.get("PROXY_API_KEY")


def verify_auth(authorization: Optional[str] = None, x_api_key: Optional[str] = None) -> None:
    """
    If PROXY_API_KEY env var is set, enforce it.
    Accepts `Authorization: Bearer <key>` or `X-Api-Key: <key>`.
    """
    if not _PROXY_API_KEY:
        return  # open mode – no auth enforced

    provided: str | None = None
    if authorization:
        match = re.match(r"^Bearer\s+(.+)$", authorization.strip(), re.IGNORECASE)
        if match:
            provided = match.group(1)
    if not provided and x_api_key:
        provided = x_api_key.strip()

    if provided != _PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
