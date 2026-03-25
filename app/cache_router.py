"""
Cache Router (Stage between Pipeline and Dispatcher)
------------------------------------------------------
Segments each payload into:
  - static_prefix  : system prompt / instructions (high cache-hit probability)
  - variable_part  : user content + current turn (never cached)

Checks the store for an existing cache key for the static prefix.
If found, marks the request so the dispatcher can send a prompt-cache hint
to providers that support it (Anthropic cache_control, OpenAI prompt caching).

This module does NOT call the LLM — it only manages cache metadata.
"""
from __future__ import annotations
import hashlib
import time
from typing import Optional

from .store import store


# TTL for static-prefix cache entries (seconds)
_STATIC_PREFIX_TTL = 3_600   # 1 hour — system prompts rarely change


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def static_prefix_key(system_prompt: str, model: str) -> str:
    return f"cache:prefix:{model}:{_sha(system_prompt)}"


def full_turn_key(system_prompt: str, history_texts: list[str], user_message: str, model: str) -> str:
    combined = "||".join([system_prompt] + history_texts + [user_message])
    return f"cache:turn:{model}:{_sha(combined)}"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class CacheRouteResult:
    __slots__ = (
        "static_cache_hit", "static_cache_key",
        "full_cache_hit", "full_cache_key",
        "cached_response",
    )

    def __init__(
        self,
        static_cache_hit: bool,
        static_cache_key: str,
        full_cache_hit: bool,
        full_cache_key: str,
        cached_response: Optional[dict],
    ) -> None:
        self.static_cache_hit   = static_cache_hit
        self.static_cache_key   = static_cache_key
        self.full_cache_hit     = full_cache_hit
        self.full_cache_key     = full_cache_key
        self.cached_response    = cached_response


def route(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    model: str,
    summary: str = "",
) -> CacheRouteResult:
    """
    Check the store for a full-turn cache hit, then check for a static-prefix hit.

    A *full-turn* hit means we've seen the exact same (system + history + query) before
    and can return the stored optimised payload immediately (no LLM call needed).

    A *static-prefix* hit means the system prompt is already cached server-side,
    so we tag the outbound request so the dispatcher can send a prompt-cache hint.
    """
    skey = static_prefix_key(system_prompt, model)
    hist_texts = [m.get("content") or "" for m in history]
    tkey = full_turn_key(system_prompt, hist_texts, user_message, model)

    # Full-turn cache (compressed payload)
    cached = store.get(tkey)
    if cached:
        return CacheRouteResult(
            static_cache_hit=True,
            static_cache_key=skey,
            full_cache_hit=True,
            full_cache_key=tkey,
            cached_response=cached,
        )

    # Static-prefix cache
    static_hit = store.get(skey) is not None
    if not static_hit:
        # Warm the static prefix cache
        store.set(skey, {"system_prompt": system_prompt, "model": model, "ts": time.time()},
                  ttl_s=_STATIC_PREFIX_TTL)

    return CacheRouteResult(
        static_cache_hit=static_hit,
        static_cache_key=skey,
        full_cache_hit=False,
        full_cache_key=tkey,
        cached_response=None,
    )


def store_turn(key: str, payload: dict, ttl_s: int = 300) -> None:
    """Persist a completed turn payload for future cache hits."""
    store.set(key, payload, ttl_s=ttl_s)


def invalidate_session(session_id: str) -> None:
    """
    Best-effort invalidation of all keys related to a session.
    Works properly with InMemoryStore; with Redis would need a key-scan.
    """
    if hasattr(store, "_data"):
        # InMemoryStore path
        import threading
        prefix = f"cache:turn:"
        with store._lock:  # type: ignore[attr-defined]
            to_del = [k for k in store._data if session_id in k]  # type: ignore[attr-defined]
        for k in to_del:
            store.delete(k)
