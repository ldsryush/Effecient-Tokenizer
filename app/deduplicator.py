"""
Semantic Deduplicator (Stage 3 of Pipeline)
--------------------------------------------
Embeds each conversation turn and collapses turns that are near-duplicates
(cosine similarity above threshold).

Embedding strategy (no heavy ML dependency required):
  - Primary:  sentence-transformers if installed  (float32 vectors)
  - Fallback: TF-IDF style bag-of-words from the turn corpus itself

Returns the deduplicated history list + a confidence score per collapsed turn.
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Sentence-transformer singleton — loaded ONCE on first use, not per request
# ---------------------------------------------------------------------------

_ST_MODEL = None
_ST_MODEL_LOADED = False

def _get_st_model():
    """Return the cached SentenceTransformer model, loading it once if needed."""
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


# ---------------------------------------------------------------------------
# Lightweight vector utilities
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _tfidf_vector(text: str, corpus_df: Counter, n_docs: int) -> dict[str, float]:
    """Return a TF-IDF weighted sparse vector for *text*."""
    tokens = _tokenize(text)
    if not tokens:
        return {}
    tf = Counter(tokens)
    n = len(tokens)
    vec: dict[str, float] = {}
    for word, count in tf.items():
        idf = math.log((n_docs + 1) / (corpus_df.get(word, 0) + 1)) + 1
        vec[word] = (count / n) * idf
    return vec


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a.get(w, 0.0) * v for w, v in b.items())
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _try_sentence_transformers(texts: list[str]) -> Optional[list[list[float]]]:
    model = _get_st_model()
    if model is None:
        return None
    try:
        embs = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embs]
    except Exception:
        return None


def _st_cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deduplicate(
    history: list[dict],
    threshold: float = 0.92,
    mode: str = "lossless",
) -> dict:
    """
    Collapse near-duplicate turns from *history*.

    Args:
        history:   list of {role, content} dicts
        threshold: cosine similarity above which turns are considered duplicates
        mode:      "lossless" preserves one copy; "lossy" drops both

    Returns:
        {
          "history":      <deduplicated list>,
          "removed":      <list of removed turn indices>,
          "confidence":   <float, min confidence across removed turns>,
          "stage":        "deduplication",
        }
    """
    if len(history) <= 1:
        return {"history": history, "removed": [], "confidence": 1.0, "stage": "deduplication"}

    texts = [m.get("content") or "" for m in history]

    # Try sentence-transformers first; fall back to TF-IDF
    st_embs = _try_sentence_transformers(texts)

    if st_embs is not None:
        similarities = _compute_similarities_st(st_embs)
        confidence_per_pair = {(i, j): s for (i, j), s in similarities.items()}
    else:
        similarities = _compute_similarities_tfidf(texts)
        confidence_per_pair = {(i, j): s for (i, j), s in similarities.items()}

    # Mark duplicates: greedy — keep first, remove later near-duplicates
    keep = [True] * len(history)
    removed_indices: list[int] = []
    confidences: list[float] = []

    for i in range(len(history)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(history)):
            if not keep[j]:
                continue
            sim = similarities.get((i, j), 0.0)
            if sim >= threshold:
                keep[j] = False
                removed_indices.append(j)
                confidences.append(sim)

    deduped = [m for i, m in enumerate(history) if keep[i]]
    min_conf = min(confidences, default=1.0)

    return {
        "history":    deduped,
        "removed":    removed_indices,
        "confidence": round(min_conf, 4),
        "stage":      "deduplication",
    }


def _compute_similarities_st(embs: list[list[float]]) -> dict[tuple, float]:
    sims: dict[tuple, float] = {}
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sims[(i, j)] = _st_cosine(embs[i], embs[j])
    return sims


def _compute_similarities_tfidf(texts: list[str]) -> dict[tuple, float]:
    # Build corpus DF
    df: Counter = Counter()
    for t in texts:
        for w in set(_tokenize(t)):
            df[w] += 1
    n = len(texts)
    vecs = [_tfidf_vector(t, df, n) for t in texts]
    sims: dict[tuple, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            sims[(i, j)] = _cosine(vecs[i], vecs[j])
    return sims
