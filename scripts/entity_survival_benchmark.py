"""
Entity Survival Benchmark
==========================
Compares entity preservation across:
  - Efficient Tokenizer pipeline (entity graph)
  - Sliding-window baseline

Run from repo root:
    python -m scripts.entity_survival_benchmark
    python -m scripts.entity_survival_benchmark --input /path/to/logs.jsonl
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from app.entity_graph import extract_entities, get_graph, delete_graph
from app.pipeline import run as run_pipeline, PipelineConfig
from app.tokenizer import count_tokens, count_tokens_messages

MODEL = "gpt-4o"
SYSTEM_PROMPT = "You are a helpful coding assistant."


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

    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user":
            user_message = history[i].get("content") or ""
            history = history[:i] + history[i + 1:]
            break

    return system_prompt or SYSTEM_PROMPT, history, user_message


def _synthetic_history(turns: int) -> list[dict]:
    pairs = [
        ("user", "In auth.py, the AuthService.login() function fails on ticket #4201."),
        ("assistant", "Got it. auth.py and AuthService.login() are noted for #4201."),
        ("user", "Also check config.yaml for timeout settings on task #4201."),
        ("assistant", "Noted. config.yaml and timeout settings are in scope."),
        ("user", "Update validate_token() in auth.py to handle null tokens."),
        ("assistant", "Acknowledged. validate_token() update tracked."),
        ("user", "Add logging to AuthService.login() with request_id."),
        ("assistant", "Will do. request_id logging added to AuthService.login()."),
    ]
    out = []
    for i in range(turns):
        role, content = pairs[i % len(pairs)]
        out.append({"role": role, "content": content})
    return out


def _entity_recall(reference: set[str], context_text: str) -> float:
    if not reference:
        return 1.0
    ents = extract_entities(context_text)
    context_set = {n for names in ents.values() for n in names}
    return len(reference & context_set) / max(1, len(reference))


def _context_text(system_prompt: str, history: list[dict], summary: str = "") -> str:
    parts = []
    if system_prompt:
        parts.append(f"[System]\n{system_prompt}")
    if summary:
        parts.append(f"[Summary]\n{summary}")
    for msg in history:
        parts.append(f"[{msg.get('role', 'user')}] {msg.get('content') or ''}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entity survival benchmark")
    parser.add_argument("--input", help="Path to JSONL conversation logs", default=None)
    parser.add_argument("--keep", type=int, default=8, help="Sliding window size")
    args = parser.parse_args()

    scenarios: list[tuple[str, str, list[dict], str]] = []
    if args.input:
        items = _load_jsonl(args.input)
        for idx, item in enumerate(items):
            msgs = _extract_messages(item)
            if not msgs:
                continue
            system_prompt, history, user_msg = _split_openai_messages(msgs)
            scenarios.append((f"JSONL Scenario {idx + 1}", system_prompt, history, user_msg))
    else:
        history = _synthetic_history(20)
        scenarios.append(("Synthetic Coding Session", SYSTEM_PROMPT, history, "What should we fix first?"))

    results: dict[str, Any] = {}
    for name, system_prompt, history, user_msg in scenarios:
        ref_entities = extract_entities(history[0]["content"] if history else "")
        ref_set = {n for names in ref_entities.values() for n in names}

        delete_graph(f"es_{name}")
        g = get_graph(f"es_{name}")
        for m in history:
            g.add_turn(m["role"], m["content"])
        r = run_pipeline(
            system_prompt=system_prompt,
            history=history,
            user_message=user_msg,
            model=MODEL,
            config=PipelineConfig(mode="lossy", max_history_tokens=400),
            graph=g,
        )
        et_context = _context_text(r.system_prompt, r.history, r.summary)
        et_recall = _entity_recall(ref_set, et_context)

        sliding = history[-args.keep:]
        sw_context = _context_text(system_prompt, sliding, "")
        sw_recall = _entity_recall(ref_set, sw_context)

        results[name] = {
            "entity_recall_et": round(et_recall, 4),
            "entity_recall_sliding": round(sw_recall, 4),
            "ref_entities": sorted(ref_set),
        }

        print(f"\n{name}")
        print(f"  Entity recall (ET): {et_recall:.3f}")
        print(f"  Entity recall (SW): {sw_recall:.3f}")

    out_path = os.path.join(PKG_ROOT, "entity_survival_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
