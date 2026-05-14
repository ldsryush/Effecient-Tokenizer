"""
Token Savings Benchmark
========================
Measures token savings and estimated quality preservation across realistic
conversation scenarios. Supports JSONL input for real logs.

Run from repo root:
    python -m scripts.savings_benchmark
    python -m scripts.savings_benchmark --input /path/to/logs.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

os.environ.setdefault("DISPATCH_DRY_RUN", "true")

from app.tokenizer import count_tokens, count_tokens_messages
from app.pipeline import run as run_pipeline, PipelineConfig
from app.quality_gate import evaluate_quality_gate
from app.entity_graph import get_graph, delete_graph

MODEL = "gpt-4o"
COST_PER_1M = 5.00   # GPT-4o input, USD

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in software engineering. "
    "You provide clear, accurate, and detailed explanations with code examples. "
    "Always structure your responses with examples where appropriate. "
    "Be concise but thorough. Avoid unnecessary repetition."
)


def _context_text(system_prompt: str, history: list[dict], summary: str, user_message: str) -> str:
    parts = []
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}")
    if summary:
        parts.append(f"[Summary]\n{summary}")
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        parts.append(f"[{role}] {content}")
    if user_message:
        parts.append(f"[user] {user_message}")
    return "\n".join(p for p in parts if p).strip()


def _pct(raw: int, post: int) -> float:
    return round(100.0 * max(0, raw - post) / max(1, raw), 1)


def _load_jsonl(path: str) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _extract_messages(obj: dict) -> list[dict]:
    if "messages" in obj and isinstance(obj["messages"], list):
        return obj["messages"]
    if "request" in obj and isinstance(obj["request"], dict):
        msgs = obj["request"].get("messages")
        if isinstance(msgs, list):
            return msgs
    return []


def _split_openai_messages(messages: list[dict]) -> tuple[str, list[dict], str]:
    system_prompt = ""
    history: list[dict] = []
    user_message = ""

    for msg in messages:
        if msg.get("role") == "system" and not system_prompt:
            system_prompt = msg.get("content") or ""
        else:
            history.append({"role": msg.get("role", "user"), "content": msg.get("content") or ""})

    # Pick the last user message as the query, keep the rest as history
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            user_message = history[i].get("content") or ""
            history = history[:i] + history[i + 1:]
            break

    return system_prompt or SYSTEM_PROMPT, history, user_message


def _synthetic_history(turns: int) -> list[dict]:
    topics = [
        ("Authentication", "We need to rotate the auth tokens and update AuthService."),
        ("Billing", "The invoice PDF is missing line items after the latest release."),
        ("Deployment", "The canary rollout failed in us-west due to a bad config."),
        ("Monitoring", "The error budget alert fired for checkout latency."),
        ("Data Pipeline", "The nightly ETL job for analytics is skipping rows."),
        ("Frontend", "The settings page fails to load in Safari 17."),
        ("Security", "We need to validate CSP headers for the app."),
        ("Integrations", "The Slack webhook returns 401 after token refresh."),
        ("Search", "Query results are stale due to index lag."),
        ("Docs", "The API guide needs an updated pagination example."),
        ("QA", "Regression suite shows flaky tests in auth_spec."),
        ("Performance", "CPU spikes on the summarizer worker during peak."),
    ]

    turns_out: list[dict] = []
    for i, (topic, detail) in enumerate(topics):
        turns_out.append({"role": "user", "content": f"Topic {i+1}: {topic}. {detail}"})
        turns_out.append({"role": "assistant", "content": f"Noted. For {topic}, I will outline next steps and mitigations."})
        if len(turns_out) >= turns:
            break

    return turns_out[:turns]


def _build_scenarios(input_path: str | None) -> dict[str, tuple[str, list[dict], str]]:
    if input_path:
        scenarios: dict[str, tuple[str, list[dict], str]] = {}
        items = _load_jsonl(input_path)
        for idx, item in enumerate(items):
            msgs = _extract_messages(item)
            if not msgs:
                continue
            system_prompt, history, user_msg = _split_openai_messages(msgs)
            name = f"JSONL Scenario {idx + 1}"
            scenarios[name] = (system_prompt, history, user_msg)
        return scenarios

    history_16 = _synthetic_history(16)
    history_24 = _synthetic_history(24)

    return {
        "Synthetic Multi-Topic (16 turns)": (SYSTEM_PROMPT, history_16, "Summarize the key risks and next actions."),
        "Synthetic Multi-Topic (24 turns)": (SYSTEM_PROMPT, history_24, "What are the top three issues to fix first?"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Efficient Tokenizer savings benchmark")
    parser.add_argument("--input", help="Path to JSONL conversation logs", default=None)
    args = parser.parse_args()

    print("\n" + "=" * 72)
    print("  Efficient Tokenizer Middleware — Token Savings Benchmark")
    print("=" * 72)
    print(f"  Model: {MODEL}  |  Cost: ${COST_PER_1M}/1M input tokens")

    scenarios = _build_scenarios(args.input)
    results: dict[str, Any] = {}
    lossless_pcts = []
    lossy_pcts = []
    quality_scores = []

    for scenario_name, (system_prompt, history, user_msg) in scenarios.items():
        raw_tokens = (
            count_tokens(system_prompt, MODEL)
            + count_tokens_messages(history, MODEL)
            + count_tokens(user_msg, MODEL)
        )

        # Run lossless (baseline)
        delete_graph(f"s_{scenario_name}_ll")
        g_ll = get_graph(f"s_{scenario_name}_ll")
        for m in history:
            g_ll.add_turn(m["role"], m["content"])
        r_ll = run_pipeline(
            system_prompt=system_prompt,
            history=history,
            user_message=user_msg,
            model=MODEL,
            config=PipelineConfig(mode="lossless", max_history_tokens=20_000),
            graph=g_ll,
        )

        # Run lossy
        delete_graph(f"s_{scenario_name}_ly")
        g_ly = get_graph(f"s_{scenario_name}_ly")
        for m in history:
            g_ly.add_turn(m["role"], m["content"])
        r_ly = run_pipeline(
            system_prompt=system_prompt,
            history=history,
            user_message=user_msg,
            model=MODEL,
            config=PipelineConfig(mode="lossy", max_history_tokens=600),
            graph=g_ly,
        )

        ll_pct = _pct(r_ll.raw_tokens, r_ll.post_tokens)
        ly_pct = _pct(r_ly.raw_tokens, r_ly.post_tokens)
        lossless_pcts.append(ll_pct)
        lossy_pcts.append(ly_pct)

        orig_ctx = _context_text(r_ll.system_prompt, r_ll.history, r_ll.summary, user_msg)
        comp_ctx = _context_text(r_ly.system_prompt, r_ly.history, r_ly.summary, user_msg)
        qres = evaluate_quality_gate(orig_ctx, comp_ctx, threshold=0.15)
        quality_score = round(1.0 - qres.drift_score, 4)
        quality_scores.append(quality_score)

        results[scenario_name] = {
            "raw_tokens": raw_tokens,
            "ll_post": r_ll.post_tokens,
            "ly_post": r_ly.post_tokens,
            "ll_pct": ll_pct,
            "ly_pct": ly_pct,
            "quality_preservation": quality_score,
            "quality_fallback_used": qres.fallback_used,
            "ll_savings_by_stage": r_ll.savings_by_stage,
            "ly_savings_by_stage": r_ly.savings_by_stage,
            "ll_confidence": r_ll.confidence_score,
            "ly_confidence": r_ly.confidence_score,
        }

        print(f"\n  ── {scenario_name}")
        print(f"     Raw tokens:      {raw_tokens:>6}")
        print(f"     Lossless output: {r_ll.post_tokens:>6}  ({ll_pct}% saved)  confidence={r_ll.confidence_score}")
        print(f"     Lossy output:    {r_ly.post_tokens:>6}  ({ly_pct}% saved)  confidence={r_ly.confidence_score}")
        print(f"     Quality preserve: {quality_score:.3f}  fallback_used={qres.fallback_used}")
        print(f"     Lossless savings by stage: {r_ll.savings_by_stage}")
        print(f"     Lossy    savings by stage: {r_ly.savings_by_stage}")

    avg_ll = round(statistics.mean(lossless_pcts), 1) if lossless_pcts else 0.0
    avg_ly = round(statistics.mean(lossy_pcts), 1) if lossy_pcts else 0.0
    avg_quality = round(statistics.mean(quality_scores), 4) if quality_scores else 0.0

    print("\n" + "=" * 72)
    print("  AGGREGATE SAVINGS SUMMARY")
    print("=" * 72)
    if lossless_pcts:
        print(f"  Lossless mode — avg savings: {avg_ll}%  range: {min(lossless_pcts)}%–{max(lossless_pcts)}%")
    if lossy_pcts:
        print(f"  Lossy mode    — avg savings: {avg_ly}%  range: {min(lossy_pcts)}%–{max(lossy_pcts)}%")
    print(f"  Avg quality preservation: {avg_quality}")

    print("\n  COST IMPACT AT SCALE (GPT-4o, $5.00/1M input tokens)")
    print(f"  {'Volume':<30} {'Baseline/mo':>14} {'After lossless':>16} {'After lossy':>14}")
    print(f"  {'-'*30} {'-'*14} {'-'*16} {'-'*14}")
    avg_raw = statistics.mean([v["raw_tokens"] for v in results.values()]) if results else 0
    avg_ll_post = statistics.mean([v["ll_post"] for v in results.values()]) if results else 0
    avg_ly_post = statistics.mean([v["ly_post"] for v in results.values()]) if results else 0
    for label, reqs_per_day in [("1K req/day", 1_000), ("10K req/day", 10_000), ("100K req/day", 100_000)]:
        monthly = reqs_per_day * 30
        base_cost = monthly * avg_raw / 1_000_000 * COST_PER_1M
        ll_cost = monthly * avg_ll_post / 1_000_000 * COST_PER_1M
        ly_cost = monthly * avg_ly_post / 1_000_000 * COST_PER_1M
        print(f"  {label:<30} ${base_cost:>13,.2f} ${ll_cost:>15,.2f} ${ly_cost:>13,.2f}")

    output = {
        "model": MODEL,
        "cost_per_1m": COST_PER_1M,
        "avg_lossless_pct": avg_ll,
        "avg_lossy_pct": avg_ly,
        "avg_quality_preservation": avg_quality,
        "scenarios": results,
    }

    out_path = os.path.join(PKG_ROOT, "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
