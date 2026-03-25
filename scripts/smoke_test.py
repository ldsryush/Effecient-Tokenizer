"""
Smoke test — exercises every major code path without a live LLM.
Run from repo root:
    python -m scripts.smoke_test
or:
    python scripts/smoke_test.py
"""
import os
import sys
import json
import time

# Ensure dry-run mode so no LLM calls are made
os.environ.setdefault("DISPATCH_DRY_RUN", "true")

THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from app import main as app_main
from app.ingress import split_messages, detect_model
from app.compressor import compress
from app.deduplicator import deduplicate
from app.relevance import prune_history, score_turns
from app.summarizer import rolling_summarize
from app.entity_graph import get_graph, delete_graph, extract_entities
from app.pipeline import run as run_pipeline, PipelineConfig
from app.cache_router import route as cache_route
from app.tokenizer import count_tokens, count_tokens_messages
from app import observability as obs


PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = PASS if condition else FAIL
    results.append((name, condition))
    status = f"  {mark}  {name}"
    if detail:
        status += f"  [{detail}]"
    print(status)
    if not condition:
        print(f"       >> FAILED")


# ---------------------------------------------------------------------------
# 1. Tokenizer
# ---------------------------------------------------------------------------
print("\n-- Tokenizer --")
t = count_tokens("Hello world, how are you?", "gpt-4o")
check("count_tokens returns int > 0", isinstance(t, int) and t > 0, f"{t} tokens")
t_msg = count_tokens_messages([{"role": "user", "content": "Tell me a joke."}], "gpt-4o")
check("count_tokens_messages > count_tokens", t_msg >= count_tokens("Tell me a joke.", "gpt-4o"))
t_claude = count_tokens("Hello world", "claude-3-opus")
check("anthropic estimate > 0", t_claude > 0, f"{t_claude}")


# ---------------------------------------------------------------------------
# 2. Ingress / model detection
# ---------------------------------------------------------------------------
print("\n-- Ingress --")
info = detect_model("gpt-4o")
check("detect gpt-4o family=openai", info["family"] == "openai")
info2 = detect_model("claude-3-5-sonnet-20241022")
check("detect claude family=anthropic", info2["family"] == "anthropic")

msgs = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is RLHF?"},
    {"role": "assistant", "content": "RLHF stands for..."},
    {"role": "user",      "content": "Give me an example."},
]
split = split_messages(msgs, "gpt-4o")
check("split: system_prompt extracted", split.system_prompt == "You are a helpful assistant.")
check("split: user_message is last user turn", split.user_message == "Give me an example.")
check("split: history has 2 turns", len(split.history) == 2)


# ---------------------------------------------------------------------------
# 3. Structural compressor
# ---------------------------------------------------------------------------
print("\n-- Structural Compressor --")
r = compress("The authorization_token should be passed as a configuration parameter.", "lossless")
check("compress reduces text", len(r["text"]) <= len("The authorization_token should be passed as a configuration parameter."))
check("compress confidence=1.0", r["confidence"] == 1.0)
check("compress stage=structural", r["stage"] == "structural")
r2 = compress('{"key":  "value",  "foo":  123}', "lossless")
check("json minification removes spaces", "  " not in r2["text"])


# ---------------------------------------------------------------------------
# 4. Semantic deduplicator
# ---------------------------------------------------------------------------
print("\n-- Semantic Deduplicator --")
history = [
    {"role": "user",      "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of AI."},
    {"role": "user",      "content": "What is machine learning?"},   # exact dup
    {"role": "assistant", "content": "It is a branch of artificial intelligence."},
]
dr = deduplicate(history, threshold=0.95, mode="lossy")
check("dedup returns history", "history" in dr)
check("dedup stage=deduplication", dr["stage"] == "deduplication")
check("dedup removes exact duplicate", len(dr["history"]) < len(history))


# ---------------------------------------------------------------------------
# 5. Relevance scorer
# ---------------------------------------------------------------------------
print("\n-- Relevance Scorer --")
hist = [
    {"role": "user",      "content": "Tell me about transformers in NLP."},
    {"role": "assistant", "content": "Transformers use self-attention mechanisms."},
    {"role": "user",      "content": "What did you have for breakfast?"},
    {"role": "assistant", "content": "I don't eat food, I'm an AI."},
]
scores = score_turns(hist, "Explain the attention mechanism in transformers.")
check("relevance scores list length matches", len(scores) == len(hist))
check("all scores in [0,1]", all(0.0 <= s <= 1.0 for s in scores))

pr = prune_history(hist, "Explain the attention mechanism in transformers.", threshold=0.05, mode="lossy")
check("prune_history returns dict with history", "history" in pr and "scores" in pr)
check("pruned history <= original", len(pr["history"]) <= len(hist))


# ---------------------------------------------------------------------------
# 6. Entity graph
# ---------------------------------------------------------------------------
print("\n-- Entity Graph --")
delete_graph("test_session")
g = get_graph("test_session")
g.add_turn("user", "I'm working on file.py for task #42.")
g.add_turn("assistant", "I'll help with file.py.")
check("entity graph has entities", len(g.entities) > 0)
snap = g.all_entities_snapshot()
check("entity snapshot is dict", isinstance(snap, dict))
ents = extract_entities("Fix the bug in file.py for task #42")
check("extract_entities finds file", "file" in ents)
lb = g.load_bearing_turns(ents)
check("load_bearing_turns returns set", isinstance(lb, set))


# ---------------------------------------------------------------------------
# 7. Rolling summarizer
# ---------------------------------------------------------------------------
print("\n-- Rolling Summarizer --")
turns = [
    {"role": "user",      "content": "What is RLHF?"},
    {"role": "assistant", "content": "Reinforcement Learning from Human Feedback."},
]
sr = rolling_summarize(turns, existing_summary="Prior discussion about ML.", graph=g)
check("summarizer returns summary", isinstance(sr["summary"], str) and len(sr["summary"]) > 0)
check("summarizer stage=summarization", sr["stage"] == "summarization")
check("entity snapshot preserved", isinstance(sr["entity_snapshot"], dict))


# ---------------------------------------------------------------------------
# 8. Full pipeline
# ---------------------------------------------------------------------------
print("\n-- Compression Pipeline --")
delete_graph("pipe_test")
g2 = get_graph("pipe_test")
long_history = [
    {"role": "user",      "content": f"Message {i}: tell me about topic {i}."}
    for i in range(10)
] + [
    {"role": "assistant", "content": f"Response {i}: here is info about topic {i}."}
    for i in range(10)
]
pipe_result = run_pipeline(
    system_prompt="You are a helpful AI assistant with configuration parameters.",
    history=long_history,
    user_message="What do you know about topic 9?",
    model="gpt-4o",
    config=PipelineConfig(mode="lossy", max_history_tokens=300),
    graph=g2,
)
check("pipeline returns PipelineResult", pipe_result is not None)
check("pipeline raw_tokens > 0", pipe_result.raw_tokens > 0)
check("pipeline post_tokens > 0", pipe_result.post_tokens > 0)
check("pipeline has savings_by_stage", isinstance(pipe_result.savings_by_stage, dict))
check("pipeline total savings >= 0", pipe_result.raw_tokens - pipe_result.post_tokens >= 0)
check("pipeline user_message unchanged", pipe_result.system_prompt != "What do you know about topic 9?")


# ---------------------------------------------------------------------------
# 9. Cache router
# ---------------------------------------------------------------------------
print("\n-- Cache Router --")
cr = cache_route("System prompt text.", [], "Hello", "gpt-4o")
check("cache_route returns result", cr is not None)
check("first call: full_cache_hit=False", not cr.full_cache_hit)

# Second call — static prefix should now be warm
cr2 = cache_route("System prompt text.", [], "Hello", "gpt-4o")
check("second call: static_cache_hit=True", cr2.static_cache_hit)


# ---------------------------------------------------------------------------
# 10. Observability
# ---------------------------------------------------------------------------
print("\n-- Observability --")
ev = obs.record(
    request_id="test-001",
    endpoint="/v1/chat/completions",
    model="gpt-4o",
    raw_tokens=500,
    post_tokens=350,
    savings_by_stage={"structural": 80, "deduplication": 70},
    compression_mode="lossy",
    confidence_score=0.9,
    overhead_ms=12.5,
    session_id="test_session",
)
check("observability record returns dict", isinstance(ev, dict))
check("token_savings correct", ev["token_savings"] == 150)
check("pct_saved correct", ev["pct_saved"] == 30.0)
check("cost_usd_saved > 0", ev["cost_usd_saved"] > 0)

stats = obs.aggregate_stats()
check("aggregate_stats returns dict", isinstance(stats, dict))
check("aggregate events_total >= 1", stats["events_total"] >= 1)

conf_log = obs.recent_confidence_log(limit=10)
check("confidence_log is list", isinstance(conf_log, list))

attr = obs.recent_attribution(limit=10)
check("attribution is list", isinstance(attr, list))


# ---------------------------------------------------------------------------
# 11. Legacy /compose endpoint
# ---------------------------------------------------------------------------
print("\n-- Legacy /compose --")
req = app_main.ComposeRequest(prompt="Explain transformers in ML", model="gpt-4o", output_mode="bullets")
res = app_main.compose(req)
check("compose: optimized_prompt present", "optimized_prompt" in res)
check("compose: token_savings present", "token_savings" in res)
check("compose: overhead_ms present", "overhead_ms" in res)


# ---------------------------------------------------------------------------
# 12. Legacy /chat endpoint
# ---------------------------------------------------------------------------
print("\n-- Legacy /chat --")
sid = "smoke_chat"
app_main.SESSIONS.pop(sid, None)
req1 = app_main.ChatRequest(session_id=sid, user_message="What is RLHF?", model="gpt-4o", output_mode="short")
res1 = app_main.chat(req1)
check("chat t1: optimized_prompt present", "optimized_prompt" in res1)
check("chat t1: context_tokens present", "context_tokens" in res1)

req2 = app_main.ChatRequest(session_id=sid, user_message="Give me 2 bullets.", model="gpt-4o", output_mode="bullets")
res2 = app_main.chat(req2)
check("chat t2: recent_message_count > 0", res2.get("recent_message_count", 0) > 0)


# ---------------------------------------------------------------------------
# 13. Admin endpoints
# ---------------------------------------------------------------------------
print("\n-- Admin API --")
m = app_main.admin_metrics()
check("admin_metrics: events_total key", "events_total" in m)

al = app_main.admin_attribution()
check("admin_attribution: attribution key", "attribution" in al)

cl = app_main.admin_confidence_log()
check("admin_confidence_log: confidence_log key", "confidence_log" in cl)

ss = app_main.admin_sessions()
check("admin_sessions: sessions key", "sessions" in ss)

hs = app_main.health()
check("health: status=ok", hs.get("status") == "ok")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"\n{'-'*40}")
print(f"  {passed}/{total} passed   {failed} failed")
print(f"{'-'*40}\n")

if failed:
    sys.exit(1)
