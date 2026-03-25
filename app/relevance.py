"""
Relevance Scorer (Stage 4 of Pipeline)
----------------------------------------
Scores each history turn against the current user query and prunes
low-scoring turns.

Scoring strategy:
  - Exact keyword overlap (always available, O(n))
  - Cosine similarity via sentence-transformers if installed
  - Entity-graph overlap from entity_graph.py (bonus score)

Mode behaviour:
  - lossless: no turns pruned regardless of score (scores computed for telemetry)
  - lossy:    turns below *threshold* are removed
"""
from __future__ import annotations
import math
import re
from collections import Counter
from typing import Optional

from .entity_graph import extract_entities, ConversationGraph


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _keyword_similarity(a: str, b: str) -> float:
    ta, tb = set(_tokenize(a)), set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / math.sqrt(len(ta) * len(tb))  # Jaccard-like


def _try_sentence_transformer_scores(query: str, turns: list[str]) -> Optional[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode(query, convert_to_tensor=True)
        t_embs = model.encode(turns, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, t_embs)[0].tolist()
        return scores
    except Exception:
        return None


def score_turns(
    history: list[dict],
    query: str,
    graph: Optional[ConversationGraph] = None,
) -> list[float]:
    """
    Return a relevance score in [0, 1] for each turn in *history* against *query*.
    Higher = more relevant.
    """
    if not history:
        return []

    texts = [m.get("content") or "" for m in history]
    query_entities = extract_entities(query)

    # Try sentence-transformers
    st_scores = _try_sentence_transformer_scores(query, texts)

    scores: list[float] = []
    for i, text in enumerate(texts):
        # Base score
        if st_scores is not None:
            base = float(st_scores[i])
        else:
            base = _keyword_similarity(query, text)

        # Entity graph bonus: turns that share entities with query get a boost
        if graph and query_entities:
            turn_entities = extract_entities(text)
            q_names = {n for names in query_entities.values() for n in names}
            t_names = {n for names in turn_entities.values() for n in names}
            overlap = len(q_names & t_names) / max(1, len(q_names))
            base = min(1.0, base + overlap * 0.3)

        scores.append(round(base, 4))

    return scores


def prune_history(
    history: list[dict],
    query: str,
    threshold: float = 0.15,
    mode: str = "lossy",
    graph: Optional[ConversationGraph] = None,
    load_bearing_ids: Optional[set[int]] = None,
) -> dict:
    """
    Score and optionally prune history turns.

    Args:
        history:           list of {role, content}
        query:             current user message
        threshold:         minimum relevance score to keep (lossy mode only)
        mode:              "lossless" | "lossy"
        graph:             ConversationGraph for entity bonus
        load_bearing_ids:  turn IDs that must survive (entity-graph protected)

    Returns:
        {
          "history":      <pruned list>,
          "scores":       <list of float scores, same length as input history>,
          "removed":      <list of removed indices>,
          "confidence":   <min confidence of removed turns>,
          "stage":        "relevance",
        }
    """
    scores = score_turns(history, query, graph)
    lb_ids = load_bearing_ids or set()

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
        turn_id = i  # index serves as turn_id when graph is not provided
        if turn_id in lb_ids or score >= threshold:
            keep_indices.append(i)
        else:
            removed_indices.append(i)
            removed_scores.append(score)

    pruned = [history[i] for i in keep_indices]
    min_conf = min(removed_scores, default=1.0)

    return {
        "history":    pruned,
        "scores":     scores,
        "removed":    removed_indices,
        "confidence": round(1.0 - (1.0 - min_conf), 4),  # how confident we are the drop is safe
        "stage":      "relevance",
    }
