"""
Comprehensive Pipeline Benchmark
=================================
Measures wall-clock processing time for every stage of the token-reducing
middleware under three realistic workload sizes:
  - small  : 4 turns  (~200 tokens)
  - medium : 20 turns (~1 000 tokens)
  - large  : 60 turns (~4 000 tokens)

Each scenario is run N_RUNS times; we report min / mean / median / p95 / max.

Run from repo root:
    python -m scripts.benchmark_pipeline
or:
    python scripts/benchmark_pipeline.py
"""
from __future__ import annotations

import os
import sys
import time
import statistics
import json

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

# Dry-run so no LLM calls are made
os.environ.setdefault("DISPATCH_DRY_RUN", "true")

# ── imports ───────────────────────────────────────────────────────────────────
from app.tokenizer import count_tokens, count_tokens_messages
from app.compressor import compress as structural_compress
from app.deduplicator import deduplicate, _try_sentence_transformers, _compute_similarities_st, _compute_similarities_tfidf
from app.relevance import prune_history, score_turns
from app.summarizer import rolling_summarize
from app.entity_graph import get_graph, delete_graph, extract_entities
from app.pipeline import run as run_pipeline, PipelineConfig
from app.cache_router import route as cache_route
from app.shared_models import get_st_model

# ── constants ─────────────────────────────────────────────────────────────────
N_RUNS = 10          # repetitions per scenario
MODEL  = "gpt-4o"

# ── helpers ───────────────────────────────────────────────────────────────────

def _ms(seconds: float) -> float:
    return round(seconds * 1000, 3)

def _stats(samples: list[float]) -> dict:
    s = sorted(samples)
    n = len(s)
    p95_idx = max(0, int(n * 0.95) - 1)
    return {
        "min_ms":    round(min(s), 3),
        "mean_ms":   round(statistics.mean(s), 3),
        "median_ms": round(statistics.median(s), 3),
        "p95_ms":    round(s[p95_idx], 3),
        "max_ms":    round(max(s), 3),
        "stdev_ms":  round(statistics.stdev(s), 3) if n > 1 else 0.0,
    }

def _make_history(n_turns: int) -> list[dict]:
    turns = []
    topics = [
        "machine learning", "neural networks", "transformers", "attention mechanisms",
        "RLHF", "fine-tuning", "embeddings", "vector databases", "RAG pipelines",
        "token efficiency", "context windows", "prompt engineering", "LLM inference",
        "quantization", "LoRA adapters", "PEFT methods", "chain-of-thought",
        "few-shot learning", "zero-shot prompting", "instruction tuning",
    ]
    for i in range(n_turns):
        topic = topics[i % len(topics)]
        turns.append({
            "role": "user",
            "content": f"Can you explain {topic} in detail? I want to understand how it works and what the key concepts are. Message index {i}."
        })
        turns.append({
            "role": "assistant",
            "content": f"Sure! {topic.capitalize()} is a fundamental concept in modern AI. "
                       f"It involves several key ideas: first, the core mechanism; second, the training procedure; "
                       f"third, practical applications. Let me walk you through each. [Turn {i}]"
        })
    return turns[:n_turns]

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in machine learning and natural language processing. "
    "You provide clear, accurate, and detailed explanations. "
    "Always structure your responses with examples where appropriate."
)

USER_MESSAGE = "Explain the attention mechanism in transformers and how it relates to RLHF fine-tuning."

SCENARIOS = {
    "small  ( 4 turns)":  _make_history(4),
    "medium (20 turns)":  _make_history(20),
    "large  (60 turns)":  _make_history(60),
}

# ── warm up the ST model ONCE before benchmarking ────────────────────────────
print("\n" + "="*70)
print("  Efficient Tokenizer Middleware — Processing Time Benchmark")
print("="*70)
print(f"\n  Warming up sentence-transformer model (all-MiniLM-L6-v2)...")
t_warm = time.perf_counter()
st_model = get_st_model()
warm_ms = _ms(time.perf_counter() - t_warm)
if st_model:
    print(f"  ✓ Model loaded in {warm_ms:.1f} ms  (one-time startup cost, not counted in benchmarks)")
else:
    print(f"  ⚠  sentence-transformers not available — will use TF-IDF fallback")

# Also warm tiktoken
print(f"  Warming up tiktoken encoder...")
t_tok = time.perf_counter()
count_tokens("warmup", MODEL)
tok_warm_ms = _ms(time.perf_counter() - t_tok)
print(f"  ✓ tiktoken ready in {tok_warm_ms:.1f} ms")

# ── per-stage micro-benchmarks ────────────────────────────────────────────────

print("\n" + "─"*70)
print("  MICRO-BENCHMARKS: Individual Stage Timing")
print("─"*70)

# ── Stage 1: Tokenizer ────────────────────────────────────────────────────────
print("\n  [Stage 1] Tokenizer (count_tokens / count_tokens_messages)")
sample_text = "The attention mechanism in transformers uses query, key, and value matrices to compute weighted sums."
sample_msgs = _make_history(10)

tok_single_samples = []
for _ in range(100):
    t0 = time.perf_counter()
    count_tokens(sample_text, MODEL)
    tok_single_samples.append(_ms(time.perf_counter() - t0))

tok_msgs_samples = []
for _ in range(50):
    t0 = time.perf_counter()
    count_tokens_messages(sample_msgs, MODEL)
    tok_msgs_samples.append(_ms(time.perf_counter() - t0))

tok_single_stats = _stats(tok_single_samples)
tok_msgs_stats   = _stats(tok_msgs_samples)
print(f"    count_tokens (single string, 100 runs):")
print(f"      min={tok_single_stats['min_ms']}ms  mean={tok_single_stats['mean_ms']}ms  "
      f"median={tok_single_stats['median_ms']}ms  p95={tok_single_stats['p95_ms']}ms  "
      f"max={tok_single_stats['max_ms']}ms")
print(f"    count_tokens_messages (10 msgs, 50 runs):")
print(f"      min={tok_msgs_stats['min_ms']}ms  mean={tok_msgs_stats['mean_ms']}ms  "
      f"median={tok_msgs_stats['median_ms']}ms  p95={tok_msgs_stats['p95_ms']}ms  "
      f"max={tok_msgs_stats['max_ms']}ms")

# ── Stage 2: Structural Compressor ───────────────────────────────────────────
print("\n  [Stage 2] Structural Compressor")
compress_samples_short = []
compress_samples_long  = []
long_text = " ".join([
    'The authorization_token configuration parameter should be passed as an environment variable. '
    'The response message contains additional information about the initialization process. '
    '{"key":  "value",  "foo":  123,  "description":  "test configuration"}  '
    'Please ensure the authentication parameters are properly initialized!!!'
] * 5)

for _ in range(200):
    t0 = time.perf_counter()
    structural_compress(sample_text, "lossy")
    compress_samples_short.append(_ms(time.perf_counter() - t0))

for _ in range(100):
    t0 = time.perf_counter()
    structural_compress(long_text, "lossy")
    compress_samples_long.append(_ms(time.perf_counter() - t0))

cs_stats = _stats(compress_samples_short)
cl_stats = _stats(compress_samples_long)
print(f"    Short text (~20 tokens, 200 runs):")
print(f"      min={cs_stats['min_ms']}ms  mean={cs_stats['mean_ms']}ms  "
      f"median={cs_stats['median_ms']}ms  p95={cs_stats['p95_ms']}ms  "
      f"max={cs_stats['max_ms']}ms")
print(f"    Long text (~{len(long_text.split())} words, 100 runs):")
print(f"      min={cl_stats['min_ms']}ms  mean={cl_stats['mean_ms']}ms  "
      f"median={cl_stats['median_ms']}ms  p95={cl_stats['p95_ms']}ms  "
      f"max={cl_stats['max_ms']}ms")

# ── Stage 3: Semantic Deduplicator ───────────────────────────────────────────
print("\n  [Stage 3] Semantic Deduplicator")
dedup_histories = {
    "4 turns":  _make_history(4),
    "20 turns": _make_history(20),
    "40 turns": _make_history(40),
}
for label, hist in dedup_histories.items():
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        deduplicate(hist, threshold=0.92, mode="lossy")
        samples.append(_ms(time.perf_counter() - t0))
    st = _stats(samples)
    print(f"    {label} ({N_RUNS} runs): "
          f"min={st['min_ms']}ms  mean={st['mean_ms']}ms  "
          f"median={st['median_ms']}ms  p95={st['p95_ms']}ms  max={st['max_ms']}ms")

# ── Stage 4: Relevance Scorer ────────────────────────────────────────────────
print("\n  [Stage 4] Relevance Scorer (prune_history)")
rel_histories = {
    "4 turns":  _make_history(4),
    "20 turns": _make_history(20),
    "40 turns": _make_history(40),
}
for label, hist in rel_histories.items():
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        prune_history(hist, USER_MESSAGE, threshold=0.15, mode="lossy")
        samples.append(_ms(time.perf_counter() - t0))
    st = _stats(samples)
    print(f"    {label} ({N_RUNS} runs): "
          f"min={st['min_ms']}ms  mean={st['mean_ms']}ms  "
          f"median={st['median_ms']}ms  p95={st['p95_ms']}ms  max={st['max_ms']}ms")

# ── Stage 5: Rolling Summarizer ──────────────────────────────────────────────
print("\n  [Stage 5] Rolling Summarizer")
delete_graph("bench_sum")
g_sum = get_graph("bench_sum")
for msg in _make_history(20):
    g_sum.add_turn(msg["role"], msg["content"])

sum_histories = {
    "4 turns":  _make_history(4),
    "20 turns": _make_history(20),
}
for label, hist in sum_histories.items():
    samples = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        rolling_summarize(hist, existing_summary="Prior ML discussion.", graph=g_sum)
        samples.append(_ms(time.perf_counter() - t0))
    st = _stats(samples)
    print(f"    {label} ({N_RUNS} runs): "
          f"min={st['min_ms']}ms  mean={st['mean_ms']}ms  "
          f"median={st['median_ms']}ms  p95={st['p95_ms']}ms  max={st['max_ms']}ms")

# ── Cache Router ─────────────────────────────────────────────────────────────
print("\n  [Cache Router] cache_route()")
cache_samples_miss = []
cache_samples_hit  = []
# Prime the cache
cache_route(SYSTEM_PROMPT, _make_history(4), USER_MESSAGE, MODEL)

for _ in range(50):
    t0 = time.perf_counter()
    cache_route(SYSTEM_PROMPT, _make_history(4), f"unique_{time.time_ns()}", MODEL)
    cache_samples_miss.append(_ms(time.perf_counter() - t0))

for _ in range(50):
    t0 = time.perf_counter()
    cache_route(SYSTEM_PROMPT, _make_history(4), USER_MESSAGE, MODEL)
    cache_samples_hit.append(_ms(time.perf_counter() - t0))

cm_stats = _stats(cache_samples_miss)
ch_stats = _stats(cache_samples_hit)
print(f"    Cache MISS (50 runs): "
      f"min={cm_stats['min_ms']}ms  mean={cm_stats['mean_ms']}ms  "
      f"median={cm_stats['median_ms']}ms  p95={cm_stats['p95_ms']}ms  max={cm_stats['max_ms']}ms")
print(f"    Cache HIT  (50 runs): "
      f"min={ch_stats['min_ms']}ms  mean={ch_stats['mean_ms']}ms  "
      f"median={ch_stats['median_ms']}ms  p95={ch_stats['p95_ms']}ms  max={ch_stats['max_ms']}ms")

# ── Full Pipeline End-to-End ──────────────────────────────────────────────────
print("\n" + "─"*70)
print("  END-TO-END PIPELINE: run_pipeline() — all 5 stages combined")
print("─"*70)

pipeline_results = {}

for scenario_label, history in SCENARIOS.items():
    delete_graph(f"bench_{scenario_label}")
    g = get_graph(f"bench_{scenario_label}")
    for msg in history:
        g.add_turn(msg["role"], msg["content"])

    raw_tok = count_tokens(SYSTEM_PROMPT, MODEL) + count_tokens_messages(history, MODEL) + count_tokens(USER_MESSAGE, MODEL)

    samples_lossless = []
    samples_lossy    = []

    for _ in range(N_RUNS):
        delete_graph(f"bench_{scenario_label}_l")
        g2 = get_graph(f"bench_{scenario_label}_l")
        for msg in history:
            g2.add_turn(msg["role"], msg["content"])
        t0 = time.perf_counter()
        r = run_pipeline(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            user_message=USER_MESSAGE,
            model=MODEL,
            config=PipelineConfig(mode="lossless"),
            graph=g2,
        )
        samples_lossless.append(_ms(time.perf_counter() - t0))

    for _ in range(N_RUNS):
        delete_graph(f"bench_{scenario_label}_ly")
        g3 = get_graph(f"bench_{scenario_label}_ly")
        for msg in history:
            g3.add_turn(msg["role"], msg["content"])
        t0 = time.perf_counter()
        r = run_pipeline(
            system_prompt=SYSTEM_PROMPT,
            history=history,
            user_message=USER_MESSAGE,
            model=MODEL,
            config=PipelineConfig(mode="lossy", max_history_tokens=800),
            graph=g3,
        )
        samples_lossy.append(_ms(time.perf_counter() - t0))

    ll_stats = _stats(samples_lossless)
    ly_stats = _stats(samples_lossy)
    pipeline_results[scenario_label] = {
        "raw_tokens": raw_tok,
        "n_turns": len(history),
        "lossless": ll_stats,
        "lossy": ly_stats,
    }

    print(f"\n  Scenario: {scenario_label}  |  raw_tokens={raw_tok}  |  {N_RUNS} runs each")
    print(f"    LOSSLESS mode:")
    print(f"      min={ll_stats['min_ms']}ms  mean={ll_stats['mean_ms']}ms  "
          f"median={ll_stats['median_ms']}ms  p95={ll_stats['p95_ms']}ms  "
          f"max={ll_stats['max_ms']}ms  stdev={ll_stats['stdev_ms']}ms")
    print(f"    LOSSY mode:")
    print(f"      min={ly_stats['min_ms']}ms  mean={ly_stats['mean_ms']}ms  "
          f"median={ly_stats['median_ms']}ms  p95={ly_stats['p95_ms']}ms  "
          f"max={ly_stats['max_ms']}ms  stdev={ly_stats['stdev_ms']}ms")

# ── Embedding-only timing (the dominant cost) ─────────────────────────────────
print("\n" + "─"*70)
print("  EMBEDDING PASS: sentence-transformer encode() timing")
print("─"*70)

if st_model:
    for n_texts, label in [(4, "4 texts"), (20, "20 texts"), (60, "60 texts")]:
        texts = [f"This is turn number {i} about machine learning and transformers." for i in range(n_texts)]
        samples = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            st_model.encode(texts, convert_to_numpy=True)
            samples.append(_ms(time.perf_counter() - t0))
        st = _stats(samples)
        print(f"    encode({label}, {N_RUNS} runs): "
              f"min={st['min_ms']}ms  mean={st['mean_ms']}ms  "
              f"median={st['median_ms']}ms  p95={st['p95_ms']}ms  max={st['max_ms']}ms")
else:
    print("    (sentence-transformers not installed — TF-IDF fallback active)")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  SUMMARY TABLE  (median latency, ms)")
print("="*70)
print(f"  {'Scenario':<28} {'Mode':<12} {'min':>8} {'mean':>8} {'median':>8} {'p95':>8} {'max':>8}")
print(f"  {'-'*28} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for label, data in pipeline_results.items():
    for mode_key, mode_label in [("lossless", "lossless"), ("lossy", "lossy")]:
        s = data[mode_key]
        print(f"  {label:<28} {mode_label:<12} "
              f"{s['min_ms']:>8} {s['mean_ms']:>8} {s['median_ms']:>8} "
              f"{s['p95_ms']:>8} {s['max_ms']:>8}")

print("\n" + "─"*70)
print("  STAGE MICRO-BENCHMARK SUMMARY (mean latency, ms)")
print("─"*70)
print(f"  {'Stage':<40} {'mean_ms':>10}")
print(f"  {'-'*40} {'-'*10}")
print(f"  {'Tokenizer: count_tokens (single)':<40} {tok_single_stats['mean_ms']:>10}")
print(f"  {'Tokenizer: count_tokens_messages (10 msgs)':<40} {tok_msgs_stats['mean_ms']:>10}")
print(f"  {'Structural compressor (short text)':<40} {cs_stats['mean_ms']:>10}")
print(f"  {'Structural compressor (long text)':<40} {cl_stats['mean_ms']:>10}")
print(f"  {'Cache router (miss)':<40} {cm_stats['mean_ms']:>10}")
print(f"  {'Cache router (hit)':<40} {ch_stats['mean_ms']:>10}")

print("\n  Notes:")
print(f"  • Model: {MODEL}  |  Runs per scenario: {N_RUNS}")
print(f"  • ST model cold-start (one-time): {warm_ms:.1f} ms  (not counted in pipeline timings)")
print(f"  • tiktoken cold-start (one-time): {tok_warm_ms:.1f} ms  (not counted in pipeline timings)")
print(f"  • All timings are wall-clock (time.perf_counter), Python process only")
print(f"  • No LLM calls made (DISPATCH_DRY_RUN=true)")
print("="*70 + "\n")
