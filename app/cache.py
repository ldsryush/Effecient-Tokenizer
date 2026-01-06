from typing import Any, Dict, Optional, List
import time
import hashlib
import threading

# Default TTL for cache entries in secs
_TTL_DEFAULT_S = 120

# Cache Store: { key: { "value": Any, "expires_at": int(ms) } }
_store: Dict[str, Dict[str, Any]] = {}

# Basic metric to track performance
_hits: int = 0
_misses: int = 0

# Global lock for thread-safe access within a single process
_lock = threading.Lock()

def _now_ms() -> int:
    return int(time.time() * 1000) # Return time in millisec

def make_key(*parts: str) -> str:
    data = "||".join(parts)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def set(key: str, value: Any, ttl_s: int = _TTL_DEFAULT_S) -> None:
    # Insert/Update cache entry with a time to live
    expires_at = _now_ms() + int(ttl_s * 1000)
    with _lock:
        _store[key] = {"value": value, "expires_at": expires_at}

def get(key: str) -> Optional[Any]:
    # Retrieve cache value if it exist and not expired and update hit/miss counter

    global _hits, _misses
    now = _now_ms()
    with _lock:
        item = _store.get(key)
        if not item:
            _misses += 1
            return None
        if item["expires_at"] < now:
            # Expired, delete and count as miss
            del _store[key]
            _misses += 1
            return None
        _hits += 1
        return item["value"]
    
def sweep() -> int:
    # Remove all expired entries and return number of removed items

    now = _now_ms()
    removed = 0
    with _lock:
        expired_keys = {k for k, v in _store.items() if v["expires_at"] < now}
        for k in expired_keys:
            del _store[k]
            removed += 1
    return removed

def clear() -> None:
    # Clear cache and reset hit/miss metrics
    global _hits, _misses
    with _lock:
        _store.clear()
        _hits = 0
        _misses = 0

def stats() -> Dict[str, Any]:
    # Return current cache metric and size
    with _lock:
        return {"hits": _hits, "misses": _misses, "size": len(_store)}
    
# Key builder (help standardize how you derive keys per endpoint)

def key_compose(normalized_prompt: str, instruction: Optional[str], model: str) -> str:
    # Compose keys consider normalized input + instruction + model. Ensures identical optimized prompts map to a single cache entry.
    return make_key(normalized_prompt, instruction or "", model)

def key_profile(prompt_text: str, reponse_text: str, model: str) -> str:
    # Profile keys consider input prompt + output response + model. Useful when re-profiling the same pair repeatedly.
    return make_key(prompt_text, reponse_text, model)

def key_chat(
    summary: str,
    recent_messages: List[Dict[str, str]],
    instruction: Optional[str],
    model: str,
    user_message: str,
) -> str:
    # Chat keys consider session summary + recent messages + instruction + model + latest user message.
    # Helps cache chat optimizations for recurring similar sessions.
    recent_text = "|".join(m.get("content", "") for m in recent_messages)
    return make_key(summary, recent_text, instruction or "", model, user_message)
