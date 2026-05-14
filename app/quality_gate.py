"""
Semantic Drift Quality Gate
---------------------------
Compares original and compressed contexts to detect excessive semantic drift.
Uses sentence-transformers embeddings when available, falling back to token
overlap similarity when the model is unavailable.
"""
from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityGateResult:
    drift_score: float
    passed: bool
    fallback_used: bool
    recommendation: str  # proceed | use_lossless | abort_compression


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def _cosine_distance(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    denom = norm_a * norm_b
    if denom == 0:
        return 1.0
    cos = max(-1.0, min(1.0, dot / denom))
    return 1.0 - cos


def _token_overlap_distance(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 0.0
    sim = len(ta & tb) / max(1, len(ta | tb))
    return 1.0 - sim


def _recommendation(drift_score: float, threshold: float) -> str:
    if drift_score <= threshold:
        return "proceed"
    if drift_score <= threshold * 2:
        return "use_lossless"
    return "abort_compression"


def evaluate_quality_gate(
    original_context: str,
    compressed_context: str,
    threshold: float = 0.15,
) -> QualityGateResult:
    """
    Compute semantic drift between original and compressed contexts.
    Returns a QualityGateResult with a recommendation.
    """
    drift_score: float
    fallback_used = False

    try:
        from .shared_models import get_st_model
        model = get_st_model()
        if model is None:
            raise RuntimeError("sentence-transformers unavailable")
        emb = model.encode([original_context, compressed_context])
        drift_score = _cosine_distance(emb[0].tolist(), emb[1].tolist())
    except Exception:
        fallback_used = True
        drift_score = _token_overlap_distance(original_context, compressed_context)

    rec = _recommendation(drift_score, threshold)
    passed = drift_score <= threshold
    return QualityGateResult(
        drift_score=round(float(drift_score), 4),
        passed=passed,
        fallback_used=fallback_used,
        recommendation=rec,
    )
