"""
Shared model singletons — loaded once per process, shared across all modules.

Centralising the SentenceTransformer singleton here means deduplicator.py and
relevance.py no longer each maintain their own copy, eliminating the duplicate
5-8 second cold-start cost that occurred when both modules loaded the model
independently.
"""
from __future__ import annotations

_ST_MODEL = None
_ST_MODEL_LOADED = False


def get_st_model():
    """
    Return the cached SentenceTransformer model, loading it once if needed.

    Thread-safety note: CPython's GIL makes the double-checked pattern safe
    for simple assignments.  If you run with a free-threaded build, wrap the
    load block in a threading.Lock.
    """
    global _ST_MODEL, _ST_MODEL_LOADED
    if _ST_MODEL_LOADED:
        return _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _ST_MODEL = None
    _ST_MODEL_LOADED = True
    return _ST_MODEL
