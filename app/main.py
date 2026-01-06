from fastapi import FastAPI
from pydantic import BaseModel
from collections import Counter, deque
from .normalize import normalize_text
from . import cache 
from typing import Dict, Any
import time
import re
import statistics

SESSIONS: Dict[str, Dict[str, Any]] = {} # {session_id: {"summary": str, "messages": [{"role": "user|assistant", "content": str}]}}
ANALYTICS_EVENTS = deque(maxlen=250)  # recent events buffer
REQUEST_COUNT = 0  # simple counter for periodic cache sweep

# Pricing table (USD per 1K tokens)
COST_PER_1K = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "haiku": {"input": 0.5, "output": 1.5},
    "gpt-5": {"input": 6.0, "output": 18.0},  # placeholder
}


def count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        return len(text.split())

def output_instructions(mode: str | None) -> str | None:
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

    

def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    pricing = COST_PER_1K.get(model, COST_PER_1K["gpt-4o"])
    return (in_tokens / 1000) * pricing["input"] + (out_tokens / 1000) * pricing["output"]

def log_event(event: Dict[str, Any]) -> None:
    # Minimal validation and append to buffer
    try:
        event["ts"] = event.get("ts", time.time())
        ANALYTICS_EVENTS.append(event)
    except Exception:
        # Avoid raising from analytics
        pass

def summarize_messages(messages: list[dict], max_len: int = 500) -> str:
    # Simple keyword based summary that doesnt call model
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
    summary = f"Key points: {', '.join(top)}"
    return summary[:max_len]
    
def format_context(summary: str, messages: list[dict], recent_n: int) -> str:
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
    
    

class ProfileRequest(BaseModel):
    normalize: bool = False
    model: str = "gpt-4o"
    prompt: str
    response: str = ""
    output_mode: str | None = None 

class ChatRequest(BaseModel):
    session_id: str
    user_message: str
    model: str = "gpt-4o"
    output_mode: str | None = "short"  # default concise answer
    max_context_tokens: int = 800  # budget for the context for the llm
    recent_messages: int = 4  # keep last N turns verbatim

class ComposeRequest(BaseModel):
    prompt: str
    model: str = "gpt-4o"
    output_mode: str | None = "short" #default which is short answers

class ResetRequest(BaseModel):
    session_id: str


app = FastAPI()

@app.get("/cache/stats")
def cache_stats():
    return cache.stats()

@app.post("/cache/sweep")
def cache_sweep():
    removed = cache.sweep()
    return {"removed": removed, **cache.stats()}

@app.post("/cache/clear")
def cache_clear():
    cache.clear()
    return {"cleared": True, **cache.stats()}

@app.on_event("startup")
def _prewarm() -> None:
    # Prime tokenizer and regex to reduce first-call latency
    try:
        _ = count_tokens_tiktoken("Warmup text", "cl100k_base")
    except Exception:
        pass
    _ = re.compile(r"\b[a-zA-Z]{3,}\b")

@app.get("/analytics/recent")
def analytics_recent(limit: int = 50) -> Dict[str, Any]:
    # Return last N events and lightweight aggregates
    events = list(ANALYTICS_EVENTS)[-limit:]
    count = len(ANALYTICS_EVENTS)
    cache_hits = sum(1 for e in ANALYTICS_EVENTS if e.get("cache_hit"))
    latencies = [e.get("overhead_ms", 0.0) for e in ANALYTICS_EVENTS if isinstance(e.get("overhead_ms"), (int, float))]
    savings = [e.get("token_savings", 0) for e in ANALYTICS_EVENTS if isinstance(e.get("token_savings"), (int, float))]
    stats = {
        "events_total": count,
        "cache_hit_rate": round(cache_hits / count, 4) if count else 0.0,
        "avg_overhead_ms": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "avg_token_savings": round(statistics.mean(savings), 3) if savings else 0.0,
    }
    return {"events": events, "stats": stats}

@app.post("/compose")
def compose(req: ComposeRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    global REQUEST_COUNT

    # 1) Normalize user input early to remove token waste
    normalized = normalize_text(req.prompt)

    # 2) Build output instruction
    instruction = output_instructions(req.output_mode)

    # 3) Cache key + early return on hit 
    key = cache.key_compose(normalized, instruction, req.model)
    cached = cache.get(key)
    if cached:
        # Log analytics for cache hit
        overhead_ms = (time.perf_counter() - t0) * 1000
        log_event({
            "endpoint": "compose",
            "model": req.model,
            "cache_hit": True,
            "overhead_ms": round(overhead_ms, 3),
            "input_tokens": cached.get("input_tokens"),
            "token_savings": cached.get("token_savings"),
        })
        # Periodic cache sweep
        REQUEST_COUNT += 1
        if REQUEST_COUNT % 200 == 0:
            cache.sweep()
        return cached

    # 4) Build optimized_prompt from normalized input + instruction
    optimized_prompt = f"{instruction}\n\n{normalized}" if instruction else normalized

    # 5) Compute tokens/costs and savings BEFORE creating payload
    original_input_tokens = count_tokens_tiktoken(req.prompt)
    optimized_input_tokens = count_tokens_tiktoken(optimized_prompt)

    token_savings = max(0, original_input_tokens - optimized_input_tokens)
    token_savings_pct = round(100.0 * token_savings / max(1, original_input_tokens), 2)

    baseline_cost = estimate_cost(req.model, original_input_tokens, 0)  # cost on raw input
    optimized_cost = estimate_cost(req.model, optimized_input_tokens, 0)

    overhead_ms = (time.perf_counter() - t0) * 1000

    # 6) Create payload 
    payload = {
        "model": req.model,
        "optimized_prompt": optimized_prompt,
        "normalized_prompt": normalized,
        "instruction": instruction,
        "input_tokens": optimized_input_tokens,
        "estimated_cost_usd": round(optimized_cost, 6),
        "overhead_ms": round(overhead_ms, 3),

        # Savings metrics
        "original_input_tokens": original_input_tokens,
        "optimized_input_tokens": optimized_input_tokens,
        "token_savings": token_savings,
        "token_savings_pct": token_savings_pct,
        "estimated_cost_usd_baseline": round(baseline_cost, 6),
    }

    # 7) Cache the complete payload so cache hits include savings fields
    cache.set(key, payload, ttl_s=180)
    # Log analytics for normal path
    log_event({
        "endpoint": "compose",
        "model": req.model,
        "cache_hit": False,
        "overhead_ms": round(overhead_ms, 3),
        "input_tokens": optimized_input_tokens,
        "token_savings": token_savings,
    })
    # Periodic cache sweep
    REQUEST_COUNT += 1
    if REQUEST_COUNT % 200 == 0:
        cache.sweep()
    return payload

@app.post("/chat")
def chat(req: ChatRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
    global REQUEST_COUNT

    # Prepare session state
    session = SESSIONS.setdefault(req.session_id, {"summary": "", "messages": []})

    # Normalize new user message
    user_norm = normalize_text(req.user_message)

    # Append to messages
    session["messages"].append({"role": "user", "content": user_norm})

    # Concise output instructions
    instruction = output_instructions(req.output_mode)

    # Build context and enforce the budget via rolling summary
    # Exclude the latest user message from context to avoid duplication
    prior_messages = session["messages"][:-1]
    context_text = format_context(session["summary"], prior_messages, req.recent_messages)
    context_tokens = count_tokens_tiktoken(context_text)

    if context_tokens > req.max_context_tokens:
        # Compress older messages into a summary, keep only recent_n (excluding latest)
        older = prior_messages[:-req.recent_messages] if req.recent_messages > 0 else prior_messages
        new_summary = summarize_messages(older, max_len=600)

        # Merge summaries
        session["summary"] = (session["summary"] + " " + new_summary).strip() if session["summary"] else new_summary

        # Keep only recent prior messages and reattach latest
        kept_prior = prior_messages[-req.recent_messages:] if req.recent_messages > 0 else []
        latest = session["messages"][-1]
        session["messages"] = kept_prior + [latest]

        # Recompute context based on kept prior messages
        context_text = format_context(session["summary"], kept_prior, req.recent_messages)
        context_tokens = count_tokens_tiktoken(context_text)

    # Cache key for chat turn (based on compact context and latest user message)
    prior_for_key = session["messages"][:-1]
    recent_prior = prior_for_key[-req.recent_messages:] if req.recent_messages > 0 else []
    key = cache.key_chat(session["summary"], recent_prior, instruction, req.model, user_norm)
    cached = cache.get(key)
    if cached:
        overhead_ms = (time.perf_counter() - t0) * 1000
        log_event({
            "endpoint": "chat",
            "model": req.model,
            "cache_hit": True,
            "overhead_ms": round(overhead_ms, 3),
            "context_tokens": cached.get("context_tokens"),
            "input_tokens": cached.get("input_tokens"),
        })
        REQUEST_COUNT += 1
        if REQUEST_COUNT % 200 == 0:
            cache.sweep()
        return cached

    # If still over budget after summarization, shrink recent window stepwise
    if context_tokens > req.max_context_tokens:
        shrink_n = max(0, req.recent_messages - 1)
        while shrink_n > 0 and context_tokens > req.max_context_tokens:
            kept_prior2 = prior_messages[-shrink_n:] if shrink_n > 0 else []
            context_text = format_context(session["summary"], kept_prior2, shrink_n)
            context_tokens = count_tokens_tiktoken(context_text)
            shrink_n -= 1

    # Make the optimized prompt for LLM
    # Keep context compact and enforce concise output
    optimized_prompt = (
        f"{instruction}\n\n{context_text}\nUser: {user_norm}\nAssistant:"
        if instruction else f"{context_text}\nUser: {user_norm}\nAssistant:"
    )

    input_tokens = count_tokens_tiktoken(optimized_prompt)
    overhead_ms = (time.perf_counter() - t0) * 1000

    payload = {
        "session_id": req.session_id,
        "model": req.model,
        "instruction": instruction,
        "optimized_prompt": optimized_prompt,
        "context_tokens": context_tokens,
        "input_tokens": input_tokens,
        "estimated_cost_usd": round(estimate_cost(req.model, input_tokens, 0), 6),
        "summary": session["summary"],
        "recent_message_count": len(session["messages"]),
        "overhead_ms": round(overhead_ms, 3),
    }
    cache.set(key, payload, ttl_s=120)
    log_event({
        "endpoint": "chat",
        "model": req.model,
        "cache_hit": False,
        "overhead_ms": round(overhead_ms, 3),
        "context_tokens": context_tokens,
        "input_tokens": input_tokens,
    })
    REQUEST_COUNT += 1
    if REQUEST_COUNT % 200 == 0:
        cache.sweep()
    return payload

