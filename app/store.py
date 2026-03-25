"""
State Store
------------
Provides a unified KV store interface used by:
  - Entity graphs (conversation memory)
  - Prompt cache keys
  - Rolling summaries

Two backends:
  1. InMemoryStore  — default, single-process, no dependencies
  2. RedisStore     — multi-node, requires REDIS_URL env var

The active backend is chosen at import time via the STORE_BACKEND env var
(default: "memory").  Switch to "redis" for horizontally-scaled deployments.

Usage:
    from app.store import store

    store.set("key", value, ttl_s=300)
    value = store.get("key")
    store.delete("key")
"""
from __future__ import annotations
import json
import os
import threading
import time
from typing import Any, Optional


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class InMemoryStore:
    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ttl_s: int = 300) -> None:
        expires = time.time() + ttl_s
        with self._lock:
            self._data[key] = (value, expires)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
        if not item:
            return None
        value, expires = item
        if time.time() > expires:
            with self._lock:
                self._data.pop(key, None)
            return None
        return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def sweep(self) -> int:
        now = time.time()
        with self._lock:
            expired = [k for k, (_, exp) in self._data.items() if exp < now]
            for k in expired:
                del self._data[k]
        return len(expired)

    def keys(self) -> list[str]:
        now = time.time()
        with self._lock:
            return [k for k, (_, exp) in self._data.items() if exp >= now]

    def size(self) -> int:
        return len(self.keys())

    def ping(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class RedisStore:
    """
    Redis-backed store.  Requires the `redis` package and REDIS_URL env var.
    Values are JSON-serialised before storage.
    """

    def __init__(self, url: str) -> None:
        import redis  # type: ignore
        self._client = redis.from_url(url, decode_responses=True)

    def set(self, key: str, value: Any, ttl_s: int = 300) -> None:
        self._client.set(key, json.dumps(value), ex=ttl_s)

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def sweep(self) -> int:
        # Redis handles TTL expiry natively; nothing to do
        return 0

    def keys(self) -> list[str]:
        return self._client.keys("*")

    def size(self) -> int:
        return self._client.dbsize()

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _build_store():
    backend = os.environ.get("STORE_BACKEND", "memory").lower()
    if backend == "redis":
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        try:
            s = RedisStore(url)
            if s.ping():
                return s
        except Exception as exc:
            print(f"[store] Redis unavailable ({exc}), falling back to in-memory")
    return InMemoryStore()


# Singleton instance — import this everywhere
store: InMemoryStore | RedisStore = _build_store()
