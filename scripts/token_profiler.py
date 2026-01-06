import argparse
import time
from typing import Dict, Any

# Simple, editable pricing table (USD per 1K tokens)
COST_PER_1K = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "haiku": {"input": 0.5, "output": 1.5},
    "gpt-5": {"input": 6.0, "output": 18.0},  # placeholders
}


def count_tokens_tiktoken(text: str, encoding_name: str = "cl100k_base") -> int:
    """Fast token counting using tiktoken; fallback to whitespace tokens."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    except Exception:
        # Fallback: simple whitespace tokenization
        return len(text.split())


def estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float:
    pricing = COST_PER_1K.get(model, COST_PER_1K["gpt-4o"])
    return (in_tokens / 1000) * pricing["input"] + (out_tokens / 1000) * pricing["output"]


def profile(prompt: str, response: str, model: str = "gpt-4o") -> Dict[str, Any]:
    t0 = time.perf_counter()
    in_tokens = count_tokens_tiktoken(prompt)
    out_tokens = count_tokens_tiktoken(response)
    cost = estimate_cost(model, in_tokens, out_tokens)
    overhead_ms = (time.perf_counter() - t0) * 1000
    return {
        "model": model,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "total_tokens": in_tokens + out_tokens,
        "estimated_cost_usd": round(cost, 6),
        "profiler_overhead_ms": round(overhead_ms, 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline token profiler")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--response", default="")
    args = parser.parse_args()
    result = profile(args.prompt, args.response, args.model)
    print(result)
