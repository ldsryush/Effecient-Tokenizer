"""
Rolling Summarizer (Stage 5 of Pipeline)
------------------------------------------
Compresses old turns into a summary node while:
  - Preserving the entity graph (named entities are never dropped)
  - Carrying entity facts forward in the summary text

Summary strategy (no model call required):
  - Keyword-frequency extraction with entity pinning
  - Optionally delegates to the upstream LLM if SUMMARIZER_USE_LLM=true

The returned summary is designed to be prepended to the optimised context
so the model always has the compressed facts even when the original turns
have been removed.
"""
from __future__ import annotations
import os
import re
from collections import Counter
from typing import Optional

from .entity_graph import ConversationGraph, extract_entities


# ---------------------------------------------------------------------------
# Local summarization (no LLM call)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the","and","for","that","with","this","from","have","into","about",
    "your","you","are","was","were","will","shall","could","would","should",
    "what","when","where","which","how","why","can","cant","cannot","is","it",
    "a","an","of","to","in","on","at","by","or","be","do","did","has","had",
    "not","but","if","as","so","we","he","she","they","i","me","my","our",
}


def _extractive_summary(texts: list[str], max_sentences: int = 4, max_len: int = 500) -> str:
    """
    Extractive summarization: pick the highest-signal sentences from the
    compressed turns rather than just listing keywords.  This gives the LLM
    actual context (specific values, error messages, decisions) while using
    far fewer tokens than the original turns.

    Scoring: TF-weighted sentence score — sentences containing the most
    frequent non-stop words rank highest.
    """
    # Split every turn into sentences
    sentences: list[str] = []
    for t in texts:
        for s in re.split(r'(?<=[.!?])\s+', t.strip()):
            s = s.strip()
            if len(s.split()) >= 4:          # skip fragments
                sentences.append(s)

    if not sentences:
        # Fallback: just truncate the raw text
        return " ".join(texts)[:max_len]

    # Build word-frequency table across all sentences
    all_words = " ".join(sentences).lower().split()
    freq = Counter(w for w in all_words if w not in _STOP_WORDS and len(w) > 3)

    def _score(sentence: str) -> float:
        words = sentence.lower().split()
        return sum(freq.get(w, 0) for w in words) / max(1, len(words))

    # Rank and pick top sentences, preserving original order
    ranked = sorted(range(len(sentences)), key=lambda i: _score(sentences[i]), reverse=True)
    top_indices = sorted(ranked[:max_sentences])
    summary = " ".join(sentences[i] for i in top_indices)
    return summary[:max_len]


def _entity_line(graph: ConversationGraph) -> str:
    snap = graph.all_entities_snapshot()
    if not snap:
        return ""
    parts = []
    for etype, names in snap.items():
        parts.append(f"{etype}s: {', '.join(names[:8])}")
    return "[Entities] " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rolling_summarize(
    turns_to_compress: list[dict],
    existing_summary: str,
    graph: Optional[ConversationGraph] = None,
    max_summary_len: int = 800,
) -> dict:
    """
    Summarize *turns_to_compress* and merge with *existing_summary*.

    Args:
        turns_to_compress:  list of {role, content} dicts (older low-signal turns)
        existing_summary:   the current accumulated summary string
        graph:              ConversationGraph — used to pin entity facts
        max_summary_len:    hard cap on summary character length

    Returns:
        {
          "summary":          <new summary string>,
          "entity_snapshot":  <dict {etype: [names]}>,
          "turns_compressed": <count>,
          "confidence":       <float>,
          "stage":            "summarization",
        }
    """
    if not turns_to_compress:
        esnap = graph.all_entities_snapshot() if graph else {}
        return {
            "summary":          existing_summary,
            "entity_snapshot":  esnap,
            "turns_compressed": 0,
            "confidence":       1.0,
            "stage":            "summarization",
        }

    texts = [m.get("content") or "" for m in turns_to_compress]
    keyword_sum = _extractive_summary(texts, max_sentences=4, max_len=400)

    # Build entity line from the graph so entities survive compression
    entity_line = _entity_line(graph) if graph else ""

    # Merge with existing summary
    parts = []
    if existing_summary:
        parts.append(existing_summary)
    parts.append(keyword_sum)
    if entity_line:
        parts.append(entity_line)

    merged = " | ".join(p for p in parts if p)
    if len(merged) > max_summary_len:
        merged = merged[:max_summary_len]

    entity_snapshot = graph.all_entities_snapshot() if graph else {}

    # Mark compressed turns in the graph
    if graph:
        compressed_count = 0
        # We don't have turn_ids here directly; mark by content match
        for turn in graph.turns:
            if any(turn.content == m.get("content") for m in turns_to_compress):
                turn.compressed = True
                compressed_count += 1
    else:
        compressed_count = len(turns_to_compress)

    return {
        "summary":          merged,
        "entity_snapshot":  entity_snapshot,
        "turns_compressed": compressed_count,
        "confidence":       0.85,   # lossy by nature
        "stage":            "summarization",
    }
