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
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .tokenizer import count_tokens, count_tokens_messages
from .compressor import compress as structural_compress
from .deduplicator import deduplicate
from .relevance import prune_history
from .summarizer import rolling_summarize
from .entity_graph import ConversationGraph, extract_entities


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
    savings: dict[str, int] = {}
    confidences: list[float] = [1.0]

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
    stage2_sys_saved = count_tokens(system_prompt, model) - count_tokens(opt_system, model)

    opt_history: list[dict] = []
    stage2_hist_saved = 0
    for msg in history:
        original_content = msg.get("content") or ""
        r = structural_compress(original_content, mode=cfg.mode)
        opt_history.append({"role": msg.get("role", "user"), "content": r["text"]})
        # Use the real tokenizer instead of the rough chars/4 approximation
        # to get accurate per-message structural savings.
        stage2_hist_saved += max(0, count_tokens(original_content, model) - count_tokens(r["text"], model))

    savings["structural"] = max(0, stage2_sys_saved + stage2_hist_saved)

    # ------------------------------------------------------------------
    # Stage 3 – Semantic deduplication
    # ------------------------------------------------------------------
    dedup_result = deduplicate(opt_history, threshold=cfg.dedup_threshold, mode=cfg.mode)
    tokens_before_dedup = count_tokens_messages(opt_history, model)
    opt_history = dedup_result["history"]
    tokens_after_dedup = count_tokens_messages(opt_history, model)
    savings["deduplication"] = max(0, tokens_before_dedup - tokens_after_dedup)
    confidences.append(dedup_result["confidence"])

    # ------------------------------------------------------------------
    # Stage 4 – Relevance scoring + pruning
    # ------------------------------------------------------------------
    lb_ids: set[int] = set()
    if graph:
        query_ents = extract_entities(user_message)
        lb_ids = graph.load_bearing_turns(query_ents)

    rel_result = prune_history(
        opt_history,
        query=user_message,
        threshold=cfg.relevance_threshold,
        mode=cfg.mode,
        graph=graph,
        load_bearing_ids=lb_ids,
    )
    tokens_before_rel = count_tokens_messages(opt_history, model)
    opt_history = rel_result["history"]
    tokens_after_rel = count_tokens_messages(opt_history, model)
    savings["relevance"] = max(0, tokens_before_rel - tokens_after_rel)
    confidences.append(rel_result["confidence"])

    # ------------------------------------------------------------------
    # Stage 5 – Rolling summarizer (fires when history is still over budget)
    # ------------------------------------------------------------------
    hist_tokens_now = count_tokens_messages(opt_history, model)
    summary = graph.summary if graph else ""

    if hist_tokens_now > cfg.max_history_tokens:
        # Keep recent N turns verbatim; compress the rest
        keep_n = cfg.summarize_keep_recent
        to_compress = opt_history[:-keep_n] if keep_n > 0 else opt_history
        recent = opt_history[-keep_n:] if keep_n > 0 else []

        sum_result = rolling_summarize(
            turns_to_compress=to_compress,
            existing_summary=summary,
            graph=graph,
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
    else:
        savings["summarization"] = 0

    # ------------------------------------------------------------------
    # Final token count
    # ------------------------------------------------------------------
    post_sys_tokens = count_tokens(opt_system, model)
    post_hist_tokens = count_tokens_messages(opt_history, model)
    # user_message is never modified by the pipeline, so add it to both
    # raw_tokens (Stage 1) and post_tokens so the comparison is symmetric.
    # Without this, user_message tokens inflate raw_tokens and appear as
    # phantom "savings" that weren't actually achieved by compression.
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
    )
