"""
Compression Pipeline Orchestrator
-----------------------------------
Runs the five stages in sequence and returns a PipelineResult that carries:
  - optimised system_prompt and history
  - per-stage token savings for the observability layer
  - overall confidence score (minimum across all stages)
  - compression mode used (lossless | lossy)

Stage order:
  1. Tokenizer-aware counter   (baseline measurement)
  2. Structural compressor     (lossless — JSON minify, key shorten, whitespace)
  3. Semantic deduplicator     (lossless/lossy — collapses near-duplicate turns)
  4. Relevance scorer          (lossless/lossy — prunes low-signal turns)
  5. Rolling summarizer        (lossy — compresses old turns into summary node)

Performance notes
-----------------
- Embeddings are computed ONCE per request (for the post-structural history)
  and passed into both Stage 3 (dedup) and Stage 4 (relevance) so the
  sentence-transformer model is called only once per pipeline run.
- Token counting is batched: we measure before/after each stage boundary
  rather than re-encoding inside each stage helper.  Mid-stage measurements
  are skipped in production; set DEBUG_PIPELINE=1 to restore verbose logging.
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .tokenizer import count_tokens, count_tokens_messages
from .compressor import compress as structural_compress
from .deduplicator import (
    deduplicate,
    _try_sentence_transformers,
    _compute_similarities_st,
    _compute_similarities_tfidf,
)
from .relevance import prune_history, score_turns
from .summarizer import rolling_summarize
from .entity_graph import ConversationGraph, extract_entities
from .quality_gate import QualityGateResult, evaluate_quality_gate
from . import observability as obs

_DEBUG = os.environ.get("DEBUG_PIPELINE", "0") == "1"
_QUALITY_GATE_THRESHOLD = float(os.environ.get("QUALITY_GATE_THRESHOLD", "0.15"))


@dataclass
class PipelineConfig:
    mode: str = "lossy"                    # "lossless" | "lossy"
    dedup_threshold: float = 0.92          # cosine sim to collapse duplicates
    relevance_threshold: float = 0.15      # min score to keep a turn (lossy)
    max_history_tokens: int = 4_000        # hard cap before summarization fires
    max_summary_len: int = 800
    summarize_keep_recent: int = 6         # keep last N turns verbatim


@dataclass
class PipelineResult:
    # Optimised payloads
    system_prompt: str
    history: list[dict]
    summary: str

    # Token counts
    raw_tokens: int
    post_tokens: int

    # Per-stage savings (tokens saved at each step)
    savings_by_stage: dict[str, int] = field(default_factory=dict)

    # Quality metadata
    confidence_score: float = 1.0
    compression_mode: str = "lossless"

    # Entity snapshot (always preserved)
    entity_snapshot: dict[str, list[str]] = field(default_factory=dict)
    quality_gate: Optional[QualityGateResult] = None
    compression_details: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared-embedding helpers
# ---------------------------------------------------------------------------

def _compute_shared_embeddings(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Encode *texts* with the shared ST model exactly once.
    Returns None if sentence-transformers is unavailable.
    """
    return _try_sentence_transformers(texts)


def _deduplicate_with_embeddings(
    history: list[dict],
    st_embs: Optional[list[list[float]]],
    threshold: float,
    mode: str,
) -> dict:
    """
    Run deduplication using pre-computed embeddings when available,
    falling back to TF-IDF otherwise.  Mirrors deduplicator.deduplicate()
    but skips the redundant model.encode() call.
    """
    if len(history) <= 1:
        return {"history": history, "removed": [], "confidence": 1.0, "stage": "deduplication"}

    texts = [m.get("content") or "" for m in history]

    if st_embs is not None:
        similarities = _compute_similarities_st(st_embs)
    else:
        similarities = _compute_similarities_tfidf(texts)

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


def _prune_with_embeddings(
    history: list[dict],
    query: str,
    st_embs: Optional[list[list[float]]],
    threshold: float,
    mode: str,
    graph: Optional[ConversationGraph],
    load_bearing_ids: set[int],
) -> dict:
    """
    Run relevance pruning using pre-computed turn embeddings when available.
    The query still needs its own embedding (it wasn't in the original batch),
    but the turn embeddings are reused from the shared batch.
    """
    import math
    from .relevance import _keyword_similarity, _tokenize
    from .entity_graph import extract_entities

    if not history:
        return {"history": history, "scores": [], "removed": [], "confidence": 1.0, "stage": "relevance"}

    texts = [m.get("content") or "" for m in history]
    query_entities = extract_entities(query)

    # Compute ST scores for the query against the pre-encoded turns
    st_scores: Optional[list[float]] = None
    if st_embs is not None:
        try:
            from .shared_models import get_st_model
            from sentence_transformers import util  # type: ignore
            model = get_st_model()
            if model is not None:
                import torch
                # Encode only the query; turn embeddings come from the shared batch
                q_emb = model.encode(query, convert_to_tensor=True)
                import numpy as np
                t_tensor = torch.tensor(st_embs)
                scores_raw = util.cos_sim(q_emb, t_tensor)[0].tolist()
                st_scores = scores_raw
        except Exception:
            st_scores = None

    scores: list[float] = []
    for i, text in enumerate(texts):
        if st_scores is not None:
            base = float(st_scores[i])
        else:
            base = _keyword_similarity(query, text)

        if graph and query_entities:
            turn_entities = extract_entities(text)
            q_names = {n for names in query_entities.values() for n in names}
            t_names = {n for names in turn_entities.values() for n in names}
            overlap = len(q_names & t_names) / max(1, len(q_names))
            base = min(1.0, base + overlap * 0.3)

        scores.append(round(base, 4))

    if mode == "lossless":
        return {
            "history":    history,
            "scores":     scores,
            "removed":    [],
            "confidence": 1.0,
            "stage":      "relevance",
        }

    keep_indices: list[int] = []
    removed_indices: list[int] = []
    removed_scores: list[float] = []

    for i, (msg, score) in enumerate(zip(history, scores)):
        if i in load_bearing_ids or score >= threshold:
            keep_indices.append(i)
        else:
            removed_indices.append(i)
            removed_scores.append(score)

    pruned = [history[i] for i in keep_indices]
    max_removed = max(removed_scores, default=0.0)
    return {
        "history":    pruned,
        "scores":     scores,
        "removed":    removed_indices,
        "confidence": round(1.0 - max_removed, 4),
        "stage":      "relevance",
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    model: str = "gpt-4o",
    config: Optional[PipelineConfig] = None,
    graph: Optional[ConversationGraph] = None,
) -> PipelineResult:
    """
    Execute the full compression pipeline.

    Args:
        system_prompt: raw system prompt text
        history:       list of {role, content} prior turns
        user_message:  the current user turn (NEVER modified)
        model:         target model name (for tokenizer selection)
        config:        pipeline configuration knobs
        graph:         live ConversationGraph for this session

    Returns:
        PipelineResult with optimised content + telemetry
    """
    cfg = config or PipelineConfig()
    original_context = _context_text(system_prompt, history)

    result = _run_core(
        system_prompt=system_prompt,
        history=history,
        user_message=user_message,
        model=model,
        config=cfg,
        graph=graph,
    )

    # Quality gate: only applies to lossy runs
    if cfg.mode != "lossless":
        compressed_context = _context_text(
            result.system_prompt,
            result.history,
            result.summary,
        )
        qres = evaluate_quality_gate(original_context, compressed_context, _QUALITY_GATE_THRESHOLD)
        result.quality_gate = qres

        if not qres.passed:
            obs.log_quality_gate({
                "ts": time.time(),
                "session_id": graph.session_id if graph else None,
                "drift_score": qres.drift_score,
                "threshold": _QUALITY_GATE_THRESHOLD,
                "passed": qres.passed,
                "fallback_used": qres.fallback_used,
                "recommendation": qres.recommendation,
                "mode_before": cfg.mode,
                "mode_after": "lossless",
            })

            lossless_cfg = PipelineConfig(
                mode="lossless",
                dedup_threshold=cfg.dedup_threshold,
                relevance_threshold=cfg.relevance_threshold,
                max_history_tokens=cfg.max_history_tokens,
                max_summary_len=cfg.max_summary_len,
                summarize_keep_recent=cfg.summarize_keep_recent,
            )
            result = _run_core(
                system_prompt=system_prompt,
                history=history,
                user_message=user_message,
                model=model,
                config=lossless_cfg,
                graph=graph,
            )
            result.quality_gate = qres

    return result


def _context_text(system_prompt: str, history: list[dict], summary: str = "") -> str:
    parts: list[str] = []
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}")
    if summary:
        parts.append(f"[Summary]\n{summary}")
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        parts.append(f"[{role}] {content}")
    return "\n".join(p for p in parts if p).strip()


def _entities_from_texts(texts: list[str]) -> list[str]:
    names: set[str] = set()
    for text in texts:
        for ents in extract_entities(text).values():
            for name in ents:
                names.add(name)
    return sorted(names)


def _run_core(
    system_prompt: str,
    history: list[dict],
    user_message: str,
    model: str,
    config: PipelineConfig,
    graph: Optional[ConversationGraph],
) -> PipelineResult:
    cfg = config
    savings: dict[str, int] = {}
    confidences: list[float] = [1.0]
    compression_details: list[dict] = []

    # ------------------------------------------------------------------
    # Stage 1 – Tokenizer-aware baseline count
    # ------------------------------------------------------------------
    raw_sys_tokens = count_tokens(system_prompt, model)
    raw_hist_tokens = count_tokens_messages(history, model)
    raw_user_tokens = count_tokens(user_message, model)
    # Include user_message so the baseline matches what the dispatcher
    # actually sends to the LLM (user_message is always forwarded unchanged).
    raw_tokens = raw_sys_tokens + raw_hist_tokens + raw_user_tokens

    # ------------------------------------------------------------------
    # Stage 2 – Structural compression of system prompt + each turn
    # ------------------------------------------------------------------
    sys_result = structural_compress(system_prompt, mode=cfg.mode)
    opt_system = sys_result["text"]

    opt_history: list[dict] = []
    stage2_hist_saved = 0
    for msg in history:
        original_content = msg.get("content") or ""
        r = structural_compress(original_content, mode=cfg.mode)
        opt_history.append({"role": msg.get("role", "user"), "content": r["text"]})
        if _DEBUG:
            stage2_hist_saved += max(
                0,
                count_tokens(original_content, model) - count_tokens(r["text"], model),
            )

    if _DEBUG:
        stage2_sys_saved = count_tokens(system_prompt, model) - count_tokens(opt_system, model)
        savings["structural"] = max(0, stage2_sys_saved + stage2_hist_saved)
    else:
        # Fast path: measure structural savings as a single before/after delta
        # on the full history rather than per-message, avoiding N extra encode calls.
        tokens_before_structural = raw_hist_tokens
        tokens_after_structural = count_tokens_messages(opt_history, model)
        stage2_sys_saved = raw_sys_tokens - count_tokens(opt_system, model)
        savings["structural"] = max(
            0, stage2_sys_saved + (tokens_before_structural - tokens_after_structural)
        )

    # ------------------------------------------------------------------
    # Shared embedding pass — encode all post-structural turn texts ONCE.
    # Both Stage 3 (dedup) and Stage 4 (relevance) consume these embeddings
    # so the sentence-transformer model is called only once per request.
    # ------------------------------------------------------------------
    hist_texts = [m.get("content") or "" for m in opt_history]
    shared_embs: Optional[list[list[float]]] = (
        _compute_shared_embeddings(hist_texts) if hist_texts else None
    )

    # ------------------------------------------------------------------
    # Stage 3 – Semantic deduplication (uses shared embeddings)
    # ------------------------------------------------------------------
    tokens_before_dedup = count_tokens_messages(opt_history, model)
    pre_dedup_history = opt_history
    dedup_result = _deduplicate_with_embeddings(
        opt_history,
        st_embs=shared_embs,
        threshold=cfg.dedup_threshold,
        mode=cfg.mode,
    )
    opt_history = dedup_result["history"]
    if dedup_result["removed"]:
        removed_texts = [pre_dedup_history[i]["content"] for i in dedup_result["removed"]]
        compression_details.append({
            "stage": "deduplication",
            "removed_indices": dedup_result["removed"],
            "confidence": dedup_result["confidence"],
            "entities": _entities_from_texts(removed_texts),
            "removed_texts": removed_texts,
        })

    # After dedup some turns may have been removed; rebuild the embedding
    # slice that corresponds to the surviving turns so Stage 4 can reuse them.
    if shared_embs is not None and dedup_result["removed"]:
        removed_set = set(dedup_result["removed"])
        shared_embs_pruned: Optional[list[list[float]]] = [
            emb for idx, emb in enumerate(shared_embs) if idx not in removed_set
        ]
    else:
        shared_embs_pruned = shared_embs

    tokens_after_dedup = count_tokens_messages(opt_history, model)
    savings["deduplication"] = max(0, tokens_before_dedup - tokens_after_dedup)
    confidences.append(dedup_result["confidence"])

    # ------------------------------------------------------------------
    # Stage 4 – Relevance scoring + pruning (reuses pruned embeddings)
    # ------------------------------------------------------------------
    lb_ids: set[int] = set()
    if graph:
        query_ents = extract_entities(user_message)
        lb_ids = graph.load_bearing_turns(query_ents)

    tokens_before_rel = tokens_after_dedup  # already computed above — free reuse
    pre_rel_history = opt_history
    rel_result = _prune_with_embeddings(
        opt_history,
        query=user_message,
        st_embs=shared_embs_pruned,
        threshold=cfg.relevance_threshold,
        mode=cfg.mode,
        graph=graph,
        load_bearing_ids=lb_ids,
    )
    opt_history = rel_result["history"]
    if rel_result["removed"]:
        removed_texts = [pre_rel_history[i]["content"] for i in rel_result["removed"]]
        compression_details.append({
            "stage": "relevance",
            "removed_indices": rel_result["removed"],
            "confidence": rel_result["confidence"],
            "entities": _entities_from_texts(removed_texts),
            "removed_texts": removed_texts,
        })
    tokens_after_rel = count_tokens_messages(opt_history, model)
    savings["relevance"] = max(0, tokens_before_rel - tokens_after_rel)
    confidences.append(rel_result["confidence"])

    # ------------------------------------------------------------------
    # Stage 5 – Rolling summarizer (fires when history is still over budget)
    # ------------------------------------------------------------------
    hist_tokens_now = tokens_after_rel  # reuse — no extra encode call
    summary = graph.summary if graph else ""
    history_turn_ids: list[int] = []
    if graph:
        if user_message:
            history_turn_ids = [t.turn_id for t in graph.turns[-(len(history) + 1):-1]]
        else:
            history_turn_ids = [t.turn_id for t in graph.turns[-len(history):]]

    if hist_tokens_now > cfg.max_history_tokens:
        # Keep recent N turns verbatim; compress the rest
        keep_n = cfg.summarize_keep_recent
        to_compress = opt_history[:-keep_n] if keep_n > 0 else opt_history
        recent = opt_history[-keep_n:] if keep_n > 0 else []
        to_compress_ids = history_turn_ids[:len(to_compress)] if history_turn_ids else None

        sum_result = rolling_summarize(
            turns_to_compress=to_compress,
            existing_summary=summary,
            graph=graph,
            turn_ids=to_compress_ids,
            max_summary_len=cfg.max_summary_len,
        )
        summary = sum_result["summary"]
        if graph:
            graph.summary = summary

        tokens_before_sum = hist_tokens_now
        opt_history = recent
        tokens_after_sum = count_tokens_messages(opt_history, model)
        savings["summarization"] = max(0, tokens_before_sum - tokens_after_sum)
        confidences.append(sum_result["confidence"])
        if to_compress:
            removed_indices = list(range(0, len(to_compress)))
            removed_texts = [m.get("content") or "" for m in to_compress]
            compression_details.append({
                "stage": "summarization",
                "removed_indices": removed_indices,
                "confidence": sum_result["confidence"],
                "entities": _entities_from_texts(removed_texts),
                "removed_texts": removed_texts,
            })
    else:
        savings["summarization"] = 0

    # ------------------------------------------------------------------
    # Final token count
    # ------------------------------------------------------------------
    post_sys_tokens = count_tokens(opt_system, model)
    post_hist_tokens = count_tokens_messages(opt_history, model)
    # user_message is never modified by the pipeline, so add it to both
    # raw_tokens (Stage 1) and post_tokens so the comparison is symmetric.
    post_tokens = post_sys_tokens + post_hist_tokens + raw_user_tokens

    overall_confidence = min(confidences)
    compression_mode = "lossless" if cfg.mode == "lossless" else (
        "lossy" if any(v > 0 for k, v in savings.items() if k != "structural") else "lossless"
    )

    entity_snapshot = graph.all_entities_snapshot() if graph else {}

    return PipelineResult(
        system_prompt=opt_system,
        history=opt_history,
        summary=summary,
        raw_tokens=raw_tokens,
        post_tokens=post_tokens,
        savings_by_stage=savings,
        confidence_score=overall_confidence,
        compression_mode=compression_mode,
        entity_snapshot=entity_snapshot,
        compression_details=compression_details,
    )
