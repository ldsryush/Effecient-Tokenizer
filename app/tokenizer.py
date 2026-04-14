"""
Tokenizer-aware token counter.
Supports OpenAI (tiktoken) and Anthropic (approximation) model families.
"""
from __future__ import annotations
import re
from typing import Any, Optional

# Mapping of model name substrings → tiktoken encoding
_TIKTOKEN_ENCODINGS: dict[str, str] = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "gpt-5": "o200k_base",
    "o1": "o200k_base",
    "o3": "o200k_base",
}

_ANTHROPIC_PREFIXES = ("claude",)

# ---------------------------------------------------------------------------
# Tiktoken encoder cache — instantiated once per encoding name, not per call.
# tiktoken.get_encoding() is cheap after the first call (it returns a cached
# object internally), but the Python-level dict lookup here avoids even that
# overhead and makes the hot path a pure dict read.
# ---------------------------------------------------------------------------
_ENCODER_CACHE: dict[str, Any] = {}


def _get_encoder(name: str) -> Any:
    """Return a cached tiktoken encoder, creating it on first use."""
    if name not in _ENCODER_CACHE:
        import tiktoken  # type: ignore
        _ENCODER_CACHE[name] = tiktoken.get_encoding(name)
    return _ENCODER_CACHE[name]


def _pick_tiktoken_encoding(model: str) -> str:
    m = model.lower()
    for fragment, enc in _TIKTOKEN_ENCODINGS.items():
        if fragment in m:
            return enc
    return "cl100k_base"  # safe fallback


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Return exact (or best-estimate) token count for *text* given *model*.

    - OpenAI / GPT models  → tiktoken (exact)
    - Anthropic / Claude   → character-based heuristic (~4 chars / token)
    - Unknown              → whitespace split (rough)
    """
    if not text:
        return 0

    m = model.lower()

    # Anthropic path – no public Python tokenizer; use heuristic
    if any(m.startswith(p) for p in _ANTHROPIC_PREFIXES):
        return _anthropic_estimate(text)

    # tiktoken path — encoder is cached; only the encode() call runs each time
    try:
        enc = _get_encoder(_pick_tiktoken_encoding(model))
        return len(enc.encode(text))
    except Exception:
        return _fallback_count(text)


def count_tokens_messages(messages: list[dict], model: str = "gpt-4o") -> int:
    """
    Count tokens across a list of {'role': ..., 'content': ...} messages.
    Adds OpenAI message-framing overhead (3 tokens per message + 3 reply primer).
    """
    total = 3  # reply primer
    for msg in messages:
        total += 4  # framing per message
        total += count_tokens(msg.get("content") or "", model)
        total += count_tokens(msg.get("role") or "", model)
    return total


def _anthropic_estimate(text: str) -> int:
    # Anthropic tokens ≈ n_chars / 3.5 — rounds up
    return max(1, int(len(text) / 3.5 + 0.5))


def _fallback_count(text: str) -> int:
    return len(re.findall(r"\S+", text))
