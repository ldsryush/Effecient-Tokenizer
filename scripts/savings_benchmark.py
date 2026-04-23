"""
Token Savings Benchmark
========================
Measures actual token reduction across realistic conversation scenarios.
Produces the publishable numbers: % saved per stage, per scenario, per mode.

Run from repo root:
    python -m scripts.savings_benchmark
"""
from __future__ import annotations
import os, sys, statistics

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

os.environ.setdefault("DISPATCH_DRY_RUN", "true")

from app.tokenizer import count_tokens, count_tokens_messages
from app.pipeline import run as run_pipeline, PipelineConfig
from app.entity_graph import get_graph, delete_graph

MODEL = "gpt-4o"
COST_PER_1M = 5.00   # GPT-4o input, USD

SYSTEM_PROMPT = (
    "You are a helpful AI assistant specializing in software engineering. "
    "You provide clear, accurate, and detailed explanations with code examples. "
    "Always structure your responses with examples where appropriate. "
    "Be concise but thorough. Avoid unnecessary repetition."
)

# ── Scenario builders ─────────────────────────────────────────────────────────

def _chat_support_history(n: int) -> list[dict]:
    """Customer support chat — lots of repetition and boilerplate."""
    pairs = [
        ("user",      "Hi, I'm having trouble with my account. I can't log in."),
        ("assistant", "I'm sorry to hear you're having trouble logging in. Can you tell me your username?"),
        ("user",      "My username is john_doe. I keep getting an error message saying invalid password."),
        ("assistant", "Thank you, john_doe. I can see your account. Let me reset your password. Please check your email."),
        ("user",      "I checked my email but I didn't receive any reset email. Can you resend it?"),
        ("assistant", "Of course! I've resent the password reset email to the address on file. Please check your spam folder too."),
        ("user",      "I still can't find the email. My email is john@example.com. Is that the one you have?"),
        ("assistant", "Yes, john@example.com is the email we have on file. The reset email should arrive within 5 minutes."),
        ("user",      "I'm having trouble with my account. I can't log in still. The password reset didn't work."),
        ("assistant", "I apologize for the inconvenience. Let me escalate this to our technical team. Ticket #4521 has been created."),
    ]
    turns = []
    for i in range(n):
        role, content = pairs[i % len(pairs)]
        turns.append({"role": role, "content": content})
    return turns

def _coding_session_history(n: int) -> list[dict]:
    """Coding assistant session — entity-rich, file references, task IDs."""
    pairs = [
        ("user",      "I'm working on file.py for task #42. The function process_data() is throwing a KeyError."),
        ("assistant", "I can help with file.py. The KeyError in process_data() is likely because the key doesn't exist in the dict. Use .get() instead."),
        ("user",      "Here's the code in file.py: def process_data(data): return data['result']. How do I fix it?"),
        ("assistant", "Change it to: def process_data(data): return data.get('result', None). This handles missing keys safely."),
        ("user",      "Thanks! Now I have another issue in file.py. The function validate_input() is not working correctly for task #42."),
        ("assistant", "For validate_input() in file.py, can you share the current implementation? I'll help debug it for task #42."),
        ("user",      "def validate_input(x): return x > 0. It should also check if x is an integer for task #42."),
        ("assistant", "Update validate_input() to: def validate_input(x): return isinstance(x, int) and x > 0. This handles both checks."),
        ("user",      "Perfect. Now for task #42, I need to add logging to file.py. What's the best approach?"),
        ("assistant", "For file.py and task #42, use Python's logging module: import logging; logger = logging.getLogger(__name__)."),
    ]
    turns = []
    for i in range(n):
        role, content = pairs[i % len(pairs)]
        turns.append({"role": role, "content": content})
    return turns

def _research_history(n: int) -> list[dict]:
    """Research assistant — long verbose turns, good compression target."""
    pairs = [
        ("user",      "Can you explain the transformer architecture in detail? I want to understand the attention mechanism, positional encoding, and how the encoder-decoder structure works together."),
        ("assistant", "The transformer architecture consists of several key components. The attention mechanism uses query, key, and value matrices to compute weighted sums of values. Positional encoding adds position information since transformers have no inherent sequence order. The encoder processes input and the decoder generates output, with cross-attention connecting them."),
        ("user",      "That's helpful. Can you explain more about the attention mechanism? Specifically, how does multi-head attention work and why is it better than single-head attention?"),
        ("assistant", "Multi-head attention runs multiple attention operations in parallel, each with different learned projections. This allows the model to attend to information from different representation subspaces simultaneously. Single-head attention can only focus on one type of relationship at a time, while multi-head attention captures diverse relationships."),
        ("user",      "I see. Now can you explain RLHF? I want to understand how reinforcement learning from human feedback works and why it's important for aligning language models."),
        ("assistant", "RLHF stands for Reinforcement Learning from Human Feedback. It involves three steps: supervised fine-tuning on demonstrations, training a reward model from human preferences, and using RL to optimize the policy against the reward model. This aligns model outputs with human values and preferences."),
        ("user",      "Can you explain the transformer architecture again? I want to make sure I understand the attention mechanism correctly before moving on."),
        ("assistant", "Of course. The transformer uses self-attention where each token attends to all other tokens. The attention score is computed as softmax(QK^T/sqrt(d_k))V. This allows parallel processing unlike RNNs and captures long-range dependencies effectively."),
        ("user",      "What are the main differences between GPT and BERT architectures? Both use transformers but seem to work differently."),
        ("assistant", "GPT uses a decoder-only architecture with causal (left-to-right) attention, making it ideal for generation. BERT uses an encoder-only architecture with bidirectional attention, making it better for understanding tasks. GPT predicts the next token; BERT predicts masked tokens."),
    ]
    turns = []
    for i in range(n):
        role, content = pairs[i % len(pairs)]
        turns.append({"role": role, "content": content})
    return turns

SCENARIOS = {
    "Customer Support Chat (10 turns)":  (_chat_support_history(10),    "What is the status of my password reset?"),
    "Customer Support Chat (20 turns)":  (_chat_support_history(20),    "What is the status of my password reset?"),
    "Coding Session (10 turns)":         (_coding_session_history(10),  "How do I add error handling to file.py for task #42?"),
    "Coding Session (20 turns)":         (_coding_session_history(20),  "How do I add error handling to file.py for task #42?"),
    "Research Assistant (10 turns)":     (_research_history(10),        "Explain the attention mechanism in transformers."),
    "Research Assistant (20 turns)":     (_research_history(20),        "Explain the attention mechanism in transformers."),
}

print("\n" + "="*72)
print("  Efficient Tokenizer Middleware — Token Savings Benchmark")
print("="*72)
print(f"  Model: {MODEL}  |  Cost: ${COST_PER_1M}/1M input tokens")

all_lossless_pcts = []
all_lossy_pcts    = []

results = {}

for scenario_name, (history, user_msg) in SCENARIOS.items():
    raw_tokens = (
        count_tokens(SYSTEM_PROMPT, MODEL)
        + count_tokens_messages(history, MODEL)
        + count_tokens(user_msg, MODEL)
    )

    # Run lossless
    delete_graph(f"s_{scenario_name}_ll")
    g_ll = get_graph(f"s_{scenario_name}_ll")
    for m in history: g_ll.add_turn(m["role"], m["content"])
    r_ll = run_pipeline(
        system_prompt=SYSTEM_PROMPT, history=history,
        user_message=user_msg, model=MODEL,
        config=PipelineConfig(mode="lossless"),
        graph=g_ll,
    )

    # Run lossy
    delete_graph(f"s_{scenario_name}_ly")
    g_ly = get_graph(f"s_{scenario_name}_ly")
    for m in history: g_ly.add_turn(m["role"], m["content"])
    r_ly = run_pipeline(
        system_prompt=SYSTEM_PROMPT, history=history,
        user_message=user_msg, model=MODEL,
        config=PipelineConfig(mode="lossy", max_history_tokens=600),
        graph=g_ly,
    )

    def _pct(raw, post): return round(100.0 * max(0, raw - post) / max(1, raw), 1)
    def _cost_saved(raw, post): return round((raw - post) / 1_000_000 * COST_PER_1M, 6)
    def _cost_per_1k(tokens): return round(tokens / 1_000_000 * COST_PER_1M * 1000, 4)

    ll_pct = _pct(r_ll.raw_tokens, r_ll.post_tokens)
    ly_pct = _pct(r_ly.raw_tokens, r_ly.post_tokens)
    all_lossless_pcts.append(ll_pct)
    all_lossy_pcts.append(ly_pct)

    results[scenario_name] = {
        "raw_tokens":    raw_tokens,
        "ll_post":       r_ll.post_tokens,
        "ly_post":       r_ly.post_tokens,
        "ll_pct":        ll_pct,
        "ly_pct":        ly_pct,
        "ll_savings_by_stage": r_ll.savings_by_stage,
        "ly_savings_by_stage": r_ly.savings_by_stage,
        "ll_confidence": r_ll.confidence_score,
        "ly_confidence": r_ly.confidence_score,
    }

    print(f"\n  ── {scenario_name}")
    print(f"     Raw tokens:      {raw_tokens:>6}")
    print(f"     Lossless output: {r_ll.post_tokens:>6}  ({ll_pct}% saved)  confidence={r_ll.confidence_score}")
    print(f"     Lossy output:    {r_ly.post_tokens:>6}  ({ly_pct}% saved)  confidence={r_ly.confidence_score}")
    print(f"     Lossless savings by stage: {r_ll.savings_by_stage}")
    print(f"     Lossy    savings by stage: {r_ly.savings_by_stage}")

print("\n" + "="*72)
print("  AGGREGATE SAVINGS SUMMARY")
print("="*72)
print(f"  Lossless mode — avg savings: {round(statistics.mean(all_lossless_pcts),1)}%  "
      f"range: {min(all_lossless_pcts)}%–{max(all_lossless_pcts)}%")
print(f"  Lossy mode    — avg savings: {round(statistics.mean(all_lossy_pcts),1)}%  "
      f"range: {min(all_lossy_pcts)}%–{max(all_lossy_pcts)}%")

# Cost impact at scale
print("\n  COST IMPACT AT SCALE (GPT-4o, $5.00/1M input tokens)")
print(f"  {'Volume':<30} {'Baseline/mo':>14} {'After lossless':>16} {'After lossy':>14}")
print(f"  {'-'*30} {'-'*14} {'-'*16} {'-'*14}")
avg_raw = statistics.mean([v["raw_tokens"] for v in results.values()])
avg_ll_post = statistics.mean([v["ll_post"] for v in results.values()])
avg_ly_post = statistics.mean([v["ly_post"] for v in results.values()])
for label, reqs_per_day in [("1K req/day", 1_000), ("10K req/day", 10_000), ("100K req/day", 100_000)]:
    monthly = reqs_per_day * 30
    base_cost  = monthly * avg_raw / 1_000_000 * COST_PER_1M
    ll_cost    = monthly * avg_ll_post / 1_000_000 * COST_PER_1M
    ly_cost    = monthly * avg_ly_post / 1_000_000 * COST_PER_1M
    print(f"  {label:<30} ${base_cost:>13,.2f} ${ll_cost:>15,.2f} ${ly_cost:>13,.2f}")

print("="*72 + "\n")
