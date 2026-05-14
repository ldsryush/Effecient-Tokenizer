"""
Efficient Tokenizer Middleware — Main Application
==================================================
Exposes two interface groups:

  PROXY INTERFACE  (OpenAI-compatible, zero code changes required)
  ──────────────────────────────────────────────────────────────────
  POST /v1/chat/completions   — drop-in replacement for OpenAI endpoint
                                runs full pipeline + dispatcher + observability

  LEGACY ENDPOINTS  (preserved for backwards compatibility)
  ──────────────────────────────────────────────────────────────────
  POST /compose               — single-turn optimised prompt builder
  POST /chat                  — multi-turn session manager
  POST /chat/reset            — clear a session
  GET  /cache/stats           — legacy cache stats
  POST /cache/sweep           — evict expired entries
  POST /cache/clear           — clear everything
  GET  /analytics/recent      — recent telemetry events

  ADMIN API
  ──────────────────────────────────────────────────────────────────
  GET  /admin/metrics         — aggregate usage dashboard data
  GET  /admin/attribution     — per-request attribution log
  GET  /admin/confidence-log  — auditable lossy-drop log
  GET  /admin/sessions        — active session list
  DELETE /admin/sessions/{id} — delete a session + its graph
  GET  /admin/store/stats     — backing store health
  GET  /health                — liveness check
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import Counter, deque
from typing import Any, Dict, List, Optional

import os as _os
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Internal modules
from .normalize import normalize_text
from . import cache                         # legacy in-process cache
from .tokenizer import count_tokens, count_tokens_messages
from .ingress import split_messages, verify_auth, detect_model
from .pipeline import run as run_pipeline, PipelineConfig
from .cache_router import route as cache_route, store_turn, invalidate_session
from .dispatcher import dispatch
from . import observability as obs
from .entity_graph import get_graph, delete_graph, all_session_ids, load_or_create
from .rag_compressor import compress_rag_chunks
from .store import store


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Efficient Tokenizer Middleware",
    description="Drop-in OpenAI-compatible proxy with multi-stage token compression.",
    version="2.0.0",
)


# ---------------------------------------------------------------------------
# Startup warmup
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _prewarm() -> None:
    try:
        count_tokens("Warmup text", "gpt-4o")
    except Exception:
        pass
    # Prewarm the shared sentence-transformer singleton so the first real
    # request doesn't pay the ~5-8s model-load cost.  Both deduplicator.py
    # and relevance.py now delegate to shared_models.get_st_model(), so a
    # single call here is sufficient for the whole process.
    try:
        from .shared_models import get_st_model
        get_st_model()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers (cost table kept for legacy endpoints)
# ---------------------------------------------------------------------------

COST_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o":  {"input": 5.0,  "output": 15.0},
    "gpt-4":   {"input": 30.0, "output": 60.0},
    "gpt-3.5": {"input": 0.5,  "output": 1.5},
    "gpt-5":   {"input": 6.0,  "output": 18.0},
    "claude":  {"input": 3.0,  "output": 15.0},
}


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    m = model.lower()
    pricing = None
    for key in COST_PER_1K:
        if key in m:
            pricing = COST_PER_1K[key]
            break
    if not pricing:
        pricing = COST_PER_1K["gpt-4o"]
    return (in_tok / 1000) * pricing["input"] + (out_tok / 1000) * pricing["output"]


def _output_instructions(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    m = mode.lower()
    if m == "bullets":
        return "Respond with concise bullet points. Keep it under 6!"
    if m == "short":
        return "Respond in short sentences. Be concise!"
    if m == "code":
        return "Return code only in a single block, no prose."
    return "Be Concise!"


# Legacy session store (used by /chat and /compose endpoints)
SESSIONS: Dict[str, Dict[str, Any]] = {}


# ===========================================================================
# ── PROXY ENDPOINT  POST /v1/chat/completions ──────────────────────────────
# ===========================================================================

class ChatCompletionsRequest(BaseModel):
    model: str = "gpt-4o"
    messages: List[Dict[str, Any]]
    # Pipeline tuning (optional, ignored by upstream LLM)
    compression_mode: str = "lossy"           # "lossless" | "lossy"
    max_history_tokens: Optional[int] = None
    relevance_threshold: float = 0.15
    dedup_threshold: float = 0.92
    # Session for entity graph continuity
    session_id: Optional[str] = None
    # Anything else forwarded to the LLM
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = None


class RagCompressRequest(BaseModel):
    model: str = "gpt-4o"
    query: str
    chunks: List[str]
    max_tokens: int = 4_000


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionsRequest,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    verify_auth(authorization, x_api_key)
    t0 = time.perf_counter()
    request_id = f"req_{uuid.uuid4().hex[:12]}"

    # ── 1. Ingress: split payload ──────────────────────────────────────────
    split = split_messages(req.messages, req.model)
    session_id = req.session_id or f"anon_{uuid.uuid4().hex[:8]}"
    graph = load_or_create(session_id, "anonymous", store)

    # Register all history turns in the entity graph
    for msg in split.history:
        graph.add_turn(msg["role"], msg["content"])
    if split.user_message:
        signal = obs.detect_context_loss(split.user_message)
        user_turn = graph.add_turn("user", split.user_message)
        if signal.detected:
            obs.log_context_loss(session_id, user_turn.turn_id, signal)

    # ── 2. Compression pipeline ────────────────────────────────────────────
    if req.max_history_tokens is None:
        ctx_window = split.model_info.get("ctx_window", 128_000)
        derived_max = int(ctx_window * 0.6)
        max_hist = max(512, derived_max)
    else:
        max_hist = req.max_history_tokens

    cfg = PipelineConfig(
        mode=req.compression_mode,
        dedup_threshold=req.dedup_threshold,
        relevance_threshold=req.relevance_threshold,
        max_history_tokens=max_hist,
    )
    result = run_pipeline(
        system_prompt=split.system_prompt,
        history=split.history,
        user_message=split.user_message,
        model=req.model,
        config=cfg,
        graph=graph,
    )
    obs.update_last_compression(
        session_id=session_id,
        compression_details=result.compression_details,
        entity_snapshot=result.entity_snapshot,
        confidence_score=result.confidence_score,
    )

    # ── 3. Cache router ────────────────────────────────────────────────────
    cache_result = cache_route(
        system_prompt=result.system_prompt,
        history=result.history,
        user_message=split.user_message,
        model=req.model,
        summary=result.summary,
    )

    if cache_result.full_cache_hit and cache_result.cached_response:
        overhead_ms = (time.perf_counter() - t0) * 1000
        obs.record(
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model=req.model,
            raw_tokens=result.raw_tokens,
            post_tokens=result.post_tokens,
            savings_by_stage=result.savings_by_stage,
            compression_mode=result.compression_mode,
            confidence_score=result.confidence_score,
            overhead_ms=overhead_ms,
            session_id=session_id,
            extra={
                "cache_hit": True,
                "compression_details": result.compression_details,
            },
        )
        return cache_result.cached_response

    # ── 4. LLM dispatcher ─────────────────────────────────────────────────
    # Stop the middleware timer BEFORE the LLM call so overhead_ms only
    # reflects the pipeline processing cost, not the LLM's response time.
    pipeline_ms = (time.perf_counter() - t0) * 1000

    extra_params: Dict[str, Any] = {}
    if req.temperature is not None:
        extra_params["temperature"] = req.temperature
    if req.max_tokens is not None:
        extra_params["max_tokens"] = req.max_tokens

    # Extract the calling app's API key from the Authorization header so the
    # proxy can forward it transparently — no env var required on the proxy.
    import re as _re
    forwarded_key: str | None = None
    if authorization:
        _m = _re.match(r"^Bearer\s+(.+)$", authorization.strip(), _re.IGNORECASE)
        if _m:
            forwarded_key = _m.group(1).strip()
    if not forwarded_key and x_api_key:
        forwarded_key = x_api_key.strip() or None

    t_llm = time.perf_counter()
    llm_response = await dispatch(
        system_prompt=result.system_prompt,
        history=result.history,
        user_message=split.user_message,
        model=req.model,
        static_cache_hit=cache_result.static_cache_hit,
        summary=result.summary,
        extra_params=extra_params,
        forwarded_api_key=forwarded_key,
    )
    llm_latency_ms = (time.perf_counter() - t_llm) * 1000

    # Register assistant reply in entity graph
    if llm_response.get("content"):
        graph.add_turn("assistant", llm_response["content"])

    # ── 5. Observability ──────────────────────────────────────────────────
    overhead_ms = pipeline_ms   # middleware-only cost
    usage = llm_response.get("usage", {})
    # Use the pipeline's own post_tokens (measured the same way as raw_tokens)
    # instead of usage.prompt_tokens from the LLM.  The LLM's prompt_tokens
    # includes the user_message and summary injected by the dispatcher, which
    # are NOT counted in raw_tokens — causing post > raw and 0 % savings.
    post_tokens_actual = result.post_tokens

    telemetry = obs.record(
        request_id=request_id,
        endpoint="/v1/chat/completions",
        model=req.model,
        raw_tokens=result.raw_tokens,
        post_tokens=post_tokens_actual,
        savings_by_stage=result.savings_by_stage,
        compression_mode=result.compression_mode,
        confidence_score=result.confidence_score,
        overhead_ms=overhead_ms,
        session_id=session_id,
        extra={
            "cache_hit":      cache_result.full_cache_hit,
            "fallback_used":  llm_response.get("fallback_used", False),
            "llm_latency_ms": round(llm_latency_ms, 3),
            "compression_details": result.compression_details,
        },
    )

    # Build OpenAI-compatible response envelope
    response_payload: Dict[str, Any] = {
        "id":      llm_response.get("id", request_id),
        "object":  "chat.completion",
        "model":   llm_response.get("model", req.model),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": llm_response.get("content", "")},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        # Middleware-specific metadata (non-standard fields)
        "_middleware": {
            "request_id":      request_id,
            "raw_tokens":      result.raw_tokens,
            "post_tokens":     post_tokens_actual,
            "token_savings":   telemetry["token_savings"],
            "pct_saved":       telemetry["pct_saved"],
            "cost_usd_saved":  telemetry["cost_usd_saved"],
            "savings_by_stage": result.savings_by_stage,
            "compression_mode": result.compression_mode,
            "confidence_score": result.confidence_score,
            "overhead_ms":     round(overhead_ms, 3),
            "cache_hit":       cache_result.full_cache_hit,
            "static_prefix_cached": cache_result.static_cache_hit,
            "entity_snapshot": result.entity_snapshot,
        },
    }

    # Store for future cache hits
    store_turn(cache_result.full_cache_key, response_payload, ttl_s=300)

    # ── Streaming response (SSE) ────────────────────────────────────────────
    # Cline and many other clients send stream=true.  We run the full
    # (non-streaming) pipeline and then emit the result as two SSE chunks
    # so the client's stream parser is satisfied.
    if req.stream:
        completion_id = response_payload.get("id", request_id)
        model_name    = response_payload.get("model", req.model)
        content_text  = llm_response.get("content", "")

        def _sse_generator():
            # chunk 1 — role delta
            chunk1 = {
                "id": completion_id, "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                "_middleware": response_payload.get("_middleware"),
            }
            yield f"data: {json.dumps(chunk1)}\n\n"

            # chunk 2 — full content in one shot
            chunk2 = {
                "id": completion_id, "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": content_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk2)}\n\n"

            # chunk 3 — finish
            chunk3 = {
                "id": completion_id, "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": response_payload.get("usage", {}),
            }
            yield f"data: {json.dumps(chunk3)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    return response_payload


# ===========================================================================
# ── MODELS ENDPOINT  GET /v1/models ─────────────────────────────────────────
# ===========================================================================

@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    """
    Stub models list — required by the OpenAI SDK and Cline on startup.
    Returns the models this proxy knows how to route.
    """
    _now = int(time.time())
    models = [
        {"id": "gpt-4o",                      "object": "model", "created": _now, "owned_by": "openai"},
        {"id": "gpt-4o-mini",                 "object": "model", "created": _now, "owned_by": "openai"},
        {"id": "gpt-4-turbo",                 "object": "model", "created": _now, "owned_by": "openai"},
        {"id": "gpt-4",                       "object": "model", "created": _now, "owned_by": "openai"},
        {"id": "gpt-3.5-turbo",               "object": "model", "created": _now, "owned_by": "openai"},
        {"id": "claude-3-5-sonnet-20241022",   "object": "model", "created": _now, "owned_by": "anthropic"},
        {"id": "claude-3-5-haiku-20241022",    "object": "model", "created": _now, "owned_by": "anthropic"},
        {"id": "claude-3-opus-20240229",       "object": "model", "created": _now, "owned_by": "anthropic"},
    ]
    return {"object": "list", "data": models}


# ===========================================================================
# ── ADMIN API ───────────────────────────────────────────────────────────────
# ===========================================================================

@app.get("/admin/metrics")
def admin_metrics(limit: int = 1_000) -> Dict[str, Any]:
    return obs.aggregate_stats(limit=limit)


@app.get("/admin/attribution")
def admin_attribution(limit: int = 50) -> Dict[str, Any]:
    return {"attribution": obs.recent_attribution(limit=limit)}


@app.get("/admin/confidence-log")
def admin_confidence_log(limit: int = 50) -> Dict[str, Any]:
    return {"confidence_log": obs.recent_confidence_log(limit=limit)}


@app.get("/admin/context-loss-log")
def admin_context_loss_log(limit: int = 50) -> Dict[str, Any]:
    return {"context_loss_log": obs.recent_context_loss_log(limit=limit)}


@app.get("/admin/events")
def admin_events(limit: int = 50) -> Dict[str, Any]:
    return {"events": obs.recent_events(limit=limit)}


@app.get("/admin/sessions")
def admin_sessions() -> Dict[str, Any]:
    ids = all_session_ids()
    return {"sessions": ids, "count": len(ids)}


@app.delete("/admin/sessions/{session_id}")
def admin_delete_session(session_id: str) -> Dict[str, Any]:
    delete_graph(session_id, "anonymous", store)
    invalidate_session(session_id)
    SESSIONS.pop(session_id, None)
    return {"deleted": session_id}


@app.get("/admin/store/stats")
def admin_store_stats() -> Dict[str, Any]:
    return {
        "backend": type(store).__name__,
        "alive":   store.ping(),
        "size":    store.size(),
    }


@app.post("/rag/compress")
def rag_compress(req: RagCompressRequest) -> Dict[str, Any]:
    result = compress_rag_chunks(
        chunks=req.chunks,
        query=req.query,
        model=req.model,
        max_tokens=req.max_tokens,
    )
    return {
        "model": req.model,
        "tokens_before": result.tokens_before,
        "tokens_after": result.tokens_after,
        "chunks": result.chunks,
        "scores": result.scores,
    }


# ===========================================================================
# ── HEALTH ──────────────────────────────────────────────────────────────────
# ===========================================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "store":  store.ping(),
        "version": "2.0.0",
    }


# ===========================================================================
# ── DASHBOARD ───────────────────────────────────────────────────────────────
# ===========================================================================

_DASHBOARD_PATH = _os.path.join(_os.path.dirname(__file__), "dashboard.html")


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root URL to the dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    """Serve the built-in browser dashboard."""
    try:
        with open(_DASHBOARD_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found</h1>", status_code=404)


# ===========================================================================
# ── LEGACY ENDPOINTS (preserved) ────────────────────────────────────────────
# ===========================================================================

class ProfileRequest(BaseModel):
    normalize: bool = False
    model: str = "gpt-4o"
    prompt: str
    response: str = ""
    output_mode: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    model: str = "gpt-4o"
    output_mode: Optional[str] = "short"
    max_context_tokens: int = 800
    recent_messages: int = 4


class ComposeRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"
    output_mode: Optional[str] = "short"


class ResetRequest(BaseModel):
    session_id: str


# ---------- legacy helpers ----------

def _summarize_messages(messages: list[dict], max_len: int = 500) -> str:
    import re
    text = " ".join(m["content"] for m in messages if m.get("content"))
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stop = {
        "the","and","for","that","with","this","from","have","into","about",
        "your","you","are","was","were","will","shall","could","would","should",
        "what","when","where","which","how","why","can","cant","cannot","is","it",
    }
    freq = Counter(w for w in words if w not in stop)
    top = [w for w, _ in freq.most_common(12)]
    if not top:
        return text[:max_len]
    return f"Key points: {', '.join(top)}"[:max_len]


def _format_context(summary: str, messages: list[dict], recent_n: int) -> str:
    recent = messages[-recent_n:] if recent_n > 0 else []
    parts = []
    if summary:
        parts.append(f"Summary: {summary}")
    if recent:
        parts.append("Recent:")
        for m in recent:
            role = m.get("role", "user").capitalize()
            parts.append(f"{role}: {m.get('content', '')}")
    return "\n".join(parts) if parts else ""


# ---------- legacy cache helpers ----------

_REQUEST_COUNT = 0

def _maybe_sweep() -> None:
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    if _REQUEST_COUNT % 200 == 0:
        cache.sweep()


# ── /compose ─────────────────────────────────────────────────────────────────

@app.post("/compose")
def compose(req: ComposeRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    normalized = normalize_text(req.prompt)
    instruction = _output_instructions(req.output_mode)
    key = cache.key_compose(normalized, instruction, req.model)
    cached = cache.get(key)
    if cached:
        _maybe_sweep()
        return cached

    optimized_prompt = f"{instruction}\n\n{normalized}" if instruction else normalized
    original_input_tokens = count_tokens(req.prompt, req.model)
    optimized_input_tokens = count_tokens(optimized_prompt, req.model)
    token_savings = max(0, original_input_tokens - optimized_input_tokens)
    token_savings_pct = round(100.0 * token_savings / max(1, original_input_tokens), 2)
    overhead_ms = (time.perf_counter() - t0) * 1000

    payload = {
        "model":                    req.model,
        "optimized_prompt":         optimized_prompt,
        "normalized_prompt":        normalized,
        "instruction":              instruction,
        "input_tokens":             optimized_input_tokens,
        "estimated_cost_usd":       round(_estimate_cost(req.model, optimized_input_tokens, 0), 6),
        "overhead_ms":              round(overhead_ms, 3),
        "original_input_tokens":    original_input_tokens,
        "optimized_input_tokens":   optimized_input_tokens,
        "token_savings":            token_savings,
        "token_savings_pct":        token_savings_pct,
        "estimated_cost_usd_baseline": round(_estimate_cost(req.model, original_input_tokens, 0), 6),
    }
    cache.set(key, payload, ttl_s=180)
    _maybe_sweep()
    return payload


# ── /chat ─────────────────────────────────────────────────────────────────────

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    session = SESSIONS.setdefault(req.session_id, {"summary": "", "messages": []})
    user_norm = normalize_text(req.user_message)
    session["messages"].append({"role": "user", "content": user_norm})
    instruction = _output_instructions(req.output_mode)
    prior_messages = session["messages"][:-1]
    context_text = _format_context(session["summary"], prior_messages, req.recent_messages)
    context_tokens = count_tokens(context_text, req.model)

    if context_tokens > req.max_context_tokens:
        older = prior_messages[:-req.recent_messages] if req.recent_messages > 0 else prior_messages
        new_summary = _summarize_messages(older, max_len=600)
        session["summary"] = (session["summary"] + " " + new_summary).strip() if session["summary"] else new_summary
        kept_prior = prior_messages[-req.recent_messages:] if req.recent_messages > 0 else []
        latest = session["messages"][-1]
        session["messages"] = kept_prior + [latest]
        context_text = _format_context(session["summary"], kept_prior, req.recent_messages)
        context_tokens = count_tokens(context_text, req.model)

    prior_for_key = session["messages"][:-1]
    recent_prior = prior_for_key[-req.recent_messages:] if req.recent_messages > 0 else []
    key = cache.key_chat(session["summary"], recent_prior, instruction, req.model, user_norm)
    cached = cache.get(key)
    if cached:
        _maybe_sweep()
        return cached

    if context_tokens > req.max_context_tokens:
        shrink_n = max(0, req.recent_messages - 1)
        prior_messages_local = session["messages"][:-1]
        while shrink_n > 0 and context_tokens > req.max_context_tokens:
            kept = prior_messages_local[-shrink_n:] if shrink_n > 0 else []
            context_text = _format_context(session["summary"], kept, shrink_n)
            context_tokens = count_tokens(context_text, req.model)
            shrink_n -= 1

    optimized_prompt = (
        f"{instruction}\n\n{context_text}\nUser: {user_norm}\nAssistant:"
        if instruction else f"{context_text}\nUser: {user_norm}\nAssistant:"
    )
    input_tokens = count_tokens(optimized_prompt, req.model)
    overhead_ms = (time.perf_counter() - t0) * 1000

    payload = {
        "session_id":          req.session_id,
        "model":               req.model,
        "instruction":         instruction,
        "optimized_prompt":    optimized_prompt,
        "context_tokens":      context_tokens,
        "input_tokens":        input_tokens,
        "estimated_cost_usd":  round(_estimate_cost(req.model, input_tokens, 0), 6),
        "summary":             session["summary"],
        "recent_message_count": len(session["messages"]),
        "overhead_ms":         round(overhead_ms, 3),
    }
    cache.set(key, payload, ttl_s=120)
    _maybe_sweep()
    return payload


# ── /chat/reset ───────────────────────────────────────────────────────────────

@app.post("/chat/reset")
def chat_reset(req: ResetRequest) -> Dict[str, Any]:
    SESSIONS.pop(req.session_id, None)
    delete_graph(req.session_id, "anonymous", store)
    invalidate_session(req.session_id)
    return {"reset": True, "session_id": req.session_id}


# ── legacy cache endpoints ────────────────────────────────────────────────────

@app.get("/cache/stats")
def cache_stats() -> Dict[str, Any]:
    return cache.stats()


@app.post("/cache/sweep")
def cache_sweep() -> Dict[str, Any]:
    removed = cache.sweep()
    return {"removed": removed, **cache.stats()}


@app.post("/cache/clear")
def cache_clear() -> Dict[str, Any]:
    cache.clear()
    return {"cleared": True, **cache.stats()}


# ── legacy analytics ─────────────────────────────────────────────────────────

@app.get("/analytics/recent")
def analytics_recent(limit: int = 50) -> Dict[str, Any]:
    events = obs.recent_events(limit=limit)
    stats = obs.aggregate_stats(limit=limit)
    return {"events": events, "stats": stats}
