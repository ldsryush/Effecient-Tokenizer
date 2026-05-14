"""
RAG Context Compressor
----------------------
Compresses retrieved document chunks before they enter the prompt.
Uses sentence-transformers when available and falls back to keyword overlap.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from .tokenizer import count_tokens
from .relevance import _keyword_similarity
from .shared_models import get_st_model


@dataclass
class RagCompressResult:
    chunks: list[str]
    scores: list[float]
    tokens_before: int
    tokens_after: int


def _score_chunks(query: str, chunks: list[str]) -> list[float]:
    model = get_st_model()
    if model is not None:
        try:
            from sentence_transformers import util  # type: ignore
            q_emb = model.encode(query, convert_to_tensor=True)
            c_embs = model.encode(chunks, convert_to_tensor=True)
            scores = util.cos_sim(q_emb, c_embs)[0].tolist()
            return [float(s) for s in scores]
        except Exception:
            pass
    return [_keyword_similarity(query, c) for c in chunks]


def compress_rag_chunks(
    *,
    chunks: list[str],
    query: str,
    model: str = "gpt-4o",
    max_tokens: int = 4_000,
) -> RagCompressResult:
    if not chunks:
        return RagCompressResult(chunks=[], scores=[], tokens_before=0, tokens_after=0)

    scores = _score_chunks(query, chunks)
    scored = list(enumerate(chunks))
    scored.sort(key=lambda x: scores[x[0]], reverse=True)

    tokens_before = sum(count_tokens(c, model) for c in chunks)

    kept: list[str] = []
    kept_scores: list[float] = []
    total = 0
    for idx, chunk in scored:
        c_tokens = count_tokens(chunk, model)
        if total + c_tokens > max_tokens:
            continue
        kept.append(chunk)
        kept_scores.append(scores[idx])
        total += c_tokens

    tokens_after = sum(count_tokens(c, model) for c in kept)
    return RagCompressResult(
        chunks=kept,
        scores=[round(s, 4) for s in kept_scores],
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )
