"""
Observability Layer
--------------------
Every request flows through here and is tagged + logged with:
  - raw_tokens          : token count before any compression
  - post_tokens         : token count after compression
  - savings_by_stage    : dict breakdown per pipeline stage
  - compression_mode    : lossless | lossy
  - confidence_score    : min confidence across all lossy ops
  - cost_usd_saved      : estimated dollar saving

Data surfaces:
  1. TELEMETRY_BUS  – in-process deque (last 1 000 events)
  2. /admin/metrics – aggregate stats endpoint
  3. /admin/attribution – per-request attribution detail
  4. /admin/confidence-log – auditable log of lossy drops
"""
from __future__ import annotations
import time
import statistics
import threading
from collections import deque
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Internal storage
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()

# Circular buffers
TELEMETRY_BUS: deque[dict] = deque(maxlen=1_000)
CONFIDENCE_LOG: deque[dict] = deque(maxlen=1_000)
ATTRIBUTION_LOG: deque[dict] = deque(maxlen=500)

# Pricing table (USD per 1M tokens, input side — as quoted by providers)
# e.g. GPT-4o = $5.00 per 1 million input tokens
_COST_PER_1M: dict[str, float] = {
    "gpt-4o":            5.00,
    "gpt-4":            10.00,
    "gpt-3.5":           0.50,
    "gpt-5":             6.00,
    "o1":               15.00,
    "o3":               10.00,
    "claude-3-5-sonnet": 3.00,
    "claude-3-opus":    15.00,
    "claude-3-haiku":    0.25,
    "claude":            3.00,
}

def _price_per_1m(model: str) -> float:
    m = model.lower()
    for key, price in _COST_PER_1M.items():
        if key in m:
            return price
    return 5.00  # default


# ---------------------------------------------------------------------------
# Public: record an event
# ---------------------------------------------------------------------------

def record(
    *,
    request_id: str,
    endpoint: str,
    model: str,
    raw_tokens: int,
    post_tokens: int,
    savings_by_stage: dict[str, int],
    compression_mode: str,
    confidence_score: float,
    overhead_ms: float,
    session_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    """
    Build a telemetry event, push to all buffers, and return it.
    """
    ts = time.time()
    token_savings = max(0, raw_tokens - post_tokens)
    pct_saved = round(100.0 * token_savings / max(1, raw_tokens), 2)
    price = _price_per_1m(model)
    cost_saved = round(token_savings / 1_000_000 * price, 6)

    event: dict[str, Any] = {
        "request_id":      request_id,
        "ts":              ts,
        "endpoint":        endpoint,
        "model":           model,
        "session_id":      session_id,
        "raw_tokens":      raw_tokens,
        "post_tokens":     post_tokens,
        "token_savings":   token_savings,
        "pct_saved":       pct_saved,
        "cost_usd_saved":  cost_saved,
        "savings_by_stage": savings_by_stage,
        "compression_mode": compression_mode,
        "confidence_score": round(confidence_score, 4),
        "overhead_ms":     round(overhead_ms, 3),
        **(extra or {}),
    }

    with _LOCK:
        TELEMETRY_BUS.append(event)

    # Write to confidence log if any lossy ops occurred
    if compression_mode == "lossy" or confidence_score < 1.0:
        _write_confidence_log(event, savings_by_stage)

    # Write to attribution log
    _write_attribution(event)

    return event


def _write_confidence_log(event: dict, savings_by_stage: dict[str, int]) -> None:
    entry = {
        "ts":               event["ts"],
        "request_id":       event["request_id"],
        "model":            event["model"],
        "session_id":       event.get("session_id"),
        "compression_mode": event["compression_mode"],
        "confidence_score": event["confidence_score"],
        "stages_applied":   list(savings_by_stage.keys()),
        "savings_by_stage": savings_by_stage,
        "tokens_dropped":   event["token_savings"],
    }
    with _LOCK:
        CONFIDENCE_LOG.append(entry)


def _write_attribution(event: dict) -> None:
    entry = {
        "ts":            event["ts"],
        "request_id":    event["request_id"],
        "endpoint":      event["endpoint"],
        "model":         event["model"],
        "raw_tokens":    event["raw_tokens"],
        "post_tokens":   event["post_tokens"],
        "pct_saved":     event["pct_saved"],
        "cost_usd_saved": event["cost_usd_saved"],
        "savings_by_stage": event["savings_by_stage"],
    }
    with _LOCK:
        ATTRIBUTION_LOG.append(entry)


# ---------------------------------------------------------------------------
# Public: aggregate stats (usage dashboard data)
# ---------------------------------------------------------------------------

def aggregate_stats(limit: int = 1_000) -> dict:
    with _LOCK:
        events = list(TELEMETRY_BUS)[-limit:]

    if not events:
        return _empty_stats()

    total = len(events)
    savings = [e["token_savings"] for e in events]
    pcts    = [e["pct_saved"] for e in events]
    costs   = [e["cost_usd_saved"] for e in events]
    lats    = [e["overhead_ms"] for e in events]
    confs   = [e["confidence_score"] for e in events]
    llm_lats = [e.get("llm_latency_ms", 0.0) for e in events if e.get("llm_latency_ms")]

    by_endpoint: dict[str, dict] = {}
    for e in events:
        ep = e.get("endpoint", "unknown")
        by_endpoint.setdefault(ep, {"count": 0, "token_savings": 0, "cost_usd_saved": 0.0})
        by_endpoint[ep]["count"] += 1
        by_endpoint[ep]["token_savings"] += e["token_savings"]
        by_endpoint[ep]["cost_usd_saved"] += e["cost_usd_saved"]

    return {
        "events_total":         total,
        "total_token_savings":  sum(savings),
        "total_cost_usd_saved": round(sum(costs), 6),
        "avg_pct_saved":        round(statistics.mean(pcts), 2) if pcts else 0.0,
        "avg_overhead_ms":      round(statistics.mean(lats), 3) if lats else 0.0,
        "avg_llm_latency_ms":   round(statistics.mean(llm_lats), 3) if llm_lats else 0.0,
        "avg_confidence":       round(statistics.mean(confs), 4) if confs else 1.0,
        "min_confidence":       round(min(confs), 4) if confs else 1.0,
        "by_endpoint":          by_endpoint,
    }


def _empty_stats() -> dict:
    return {
        "events_total": 0,
        "total_token_savings": 0,
        "total_cost_usd_saved": 0.0,
        "avg_pct_saved": 0.0,
        "avg_overhead_ms": 0.0,
        "avg_llm_latency_ms": 0.0,
        "avg_confidence": 1.0,
        "min_confidence": 1.0,
        "by_endpoint": {},
    }


def recent_events(limit: int = 50) -> list[dict]:
    with _LOCK:
        return list(TELEMETRY_BUS)[-limit:]


def recent_confidence_log(limit: int = 50) -> list[dict]:
    with _LOCK:
        return list(CONFIDENCE_LOG)[-limit:]


def recent_attribution(limit: int = 50) -> list[dict]:
    with _LOCK:
        return list(ATTRIBUTION_LOG)[-limit:]
