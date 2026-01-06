import os
import sys
import json
import time

# Ensure we can import the app module by adding its folder to sys.path
THIS_DIR = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from app import main as app_main  # import app/main.py via package


def run_compose():
    req = app_main.ComposeRequest(prompt="Explain transformers in ML", model="gpt-4o", output_mode="bullets")
    res = app_main.compose(req)
    assert "optimized_prompt" in res and "token_savings" in res, "compose payload missing fields"
    print("compose ok:", json.dumps({
        "input_tokens": res.get("input_tokens"),
        "token_savings": res.get("token_savings"),
        "overhead_ms": res.get("overhead_ms"),
    }))


def run_chat():
    # Reset session state if exists
    sid = "smoke"
    # Use direct session dict manipulation since reset endpoint is part of app_main
    app_main.SESSIONS.pop(sid, None)

    req1 = app_main.ChatRequest(session_id=sid, user_message="What is RLHF?", model="gpt-4o", output_mode="short")
    res1 = app_main.chat(req1)
    assert "optimized_prompt" in res1 and "context_tokens" in res1, "chat payload missing fields (first)"

    # Second turn to exercise cache key and analytics
    req2 = app_main.ChatRequest(session_id=sid, user_message="Give me 2 bullets.", model="gpt-4o", output_mode="bullets")
    res2 = app_main.chat(req2)
    assert "optimized_prompt" in res2 and "context_tokens" in res2, "chat payload missing fields (second)"
    print("chat ok:", json.dumps({
        "t1_tokens": res1.get("input_tokens"),
        "t2_tokens": res2.get("input_tokens"),
        "summary_len": len(res2.get("summary", ""))
    }))


def run_cache_stats():
    stats = app_main.cache_stats()
    assert "size" in stats, "cache stats missing 'size'"
    print("cache stats:", json.dumps(stats))


def run_analytics():
    res = app_main.analytics_recent(limit=20)
    assert "events" in res and "stats" in res, "analytics payload missing fields"
    print("analytics ok:", json.dumps({
        "events": len(res["events"]),
        "avg_overhead_ms": res["stats"].get("avg_overhead_ms", 0),
        "cache_hit_rate": res["stats"].get("cache_hit_rate", 0),
    }))


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_compose()
    run_chat()
    run_cache_stats()
    run_analytics()
    print("smoke done in ms:", round((time.perf_counter() - t0) * 1000, 2))
