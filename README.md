# Effecient-Tokenizer

Middleware to minimize tokens when using LLMs.

## Features

- Normalization: trims, preserves newlines, collapses intra-line whitespace, dedupes lines.
- Output control: concise modes (short, bullets, code-only).
- Auto-optimized prompts: instruction + normalized input with token/cost savings.
- Chat compression: rolling summaries and strict token budget enforcement.
- Cache: in-memory TTL with hit/miss stats, sweep and clear endpoints.
- Analytics: recent events with cache-hit rate and overhead averages.
- Startup pre-warm: tokenizer and regex primed to reduce first-call latency.
- Tests: lightweight smoke script to validate flows without server.

## Quick Start

### 1) Create a virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r Effecient-Tokenizer\requirements.txt
```

### 2) Run baseline profiler (CLI)

```powershell
python .\Effecient-Tokenizer\scripts\token_profiler.py --model gpt-4o --prompt "Summarize the following document." --response "A brief summary."
```

### 3) Start FastAPI server

```powershell
uvicorn Effecient-Tokenizer.app.main:app --reload
```

Test:

```powershell
curl -X POST "http://127.0.0.1:8000/profile" -H "Content-Type: application/json" -d "{\"model\":\"gpt-4o\",\"prompt\":\"Write a haiku about the moon\",\"response\":\"Silver orb whispers\"}"
```

You should see JSON with input/output tokens, total tokens, estimated cost, and profiler overhead.

### 4) Analytics and Cache

Recent analytics events (latency, savings, cache hits):

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/analytics/recent | ConvertTo-Json -Depth 4
```

Cache stats and hygiene:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/cache/stats | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/cache/sweep | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/cache/clear | ConvertTo-Json -Depth 4
```

Compose optimized prompt and savings:

```powershell
$body = @{ model = "gpt-4o"; prompt = "Explain transformers"; output_mode = "bullets" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/compose -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

Chat with rolling summary:

```powershell
$body = @{ session_id = "demo"; user_message = "What is RLHF?"; model = "gpt-4o"; output_mode = "short"; recent_messages = 4; max_context_tokens = 800 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 6
```

## Demo

Run the lightweight smoke test to verify core flows without starting the server.

```powershell
& "C:\Users\ldsry\Desktop\token minimizer\.venv311\Scripts\python.exe" "Effecient-Tokenizer\scripts\smoke_test.py"
```

Example output:

```
compose ok: {"input_tokens": 17, "token_savings": 0, "overhead_ms": 349.161}
chat ok: {"t1_tokens": 17, "t2_tokens": 31, "summary_len": 0}
cache stats: {"hits": 0, "misses": 3, "size": 3}
analytics ok: {"events": 3, "avg_overhead_ms": 116.531, "cache_hit_rate": 0.0}
smoke done in ms: 351.06
```

Files:
- [Effecient-Tokenizer/scripts/smoke_test.py](Effecient-Tokenizer/scripts/smoke_test.py)
- [Effecient-Tokenizer/app/main.py](Effecient-Tokenizer/app/main.py)

## Sample Output

### Compose

```json
{
	"model": "gpt-4o",
	"optimized_prompt": "Respond with concise bullet points. Keep it under 6!\n\nExplain transformers in ML",
	"normalized_prompt": "Explain transformers in ML",
	"instruction": "Respond with concise bullet points. Keep it under 6!",
	"input_tokens": 17,
	"estimated_cost_usd": 0.085,
	"overhead_ms": 120.5,
	"original_input_tokens": 17,
	"optimized_input_tokens": 17,
	"token_savings": 0,
	"token_savings_pct": 0.0,
	"estimated_cost_usd_baseline": 0.085
}
```

### Chat

## Architecture

```
User Request
	│
	├─> Normalize Text (preserve newlines, collapse intra-line whitespace, dedupe)
	│
	├─> Output Instructions (short | bullets | code)
	│
	├─> Cache GET (TTL, hashed key per endpoint)
	│      └─ hit → return payload, log analytics
	│
	├─> Compose Path (optimized prompt + savings)
	│      ├─ Build optimized_prompt = instruction + normalized
	│      ├─ Compute tokens + cost + savings
	│      ├─ Cache SET (store enriched payload)
	│      └─ Log Analytics (overhead, savings, cache_hit=false)
	│
	└─> Chat Path (rolling summary + token budget)
			 ├─ Append user → prior messages exclude latest
			 ├─ Build context: summary + recent N
			 ├─ If over budget: summarize older → keep recent → recompute
			 ├─ If still over: shrink recent window stepwise
			 ├─ Cache GET (summary + recent + instruction + latest)
			 │      └─ hit → return payload, log analytics
			 ├─ Build optimized_prompt (context + user)
			 ├─ Cache SET (compact payload)
			 └─ Log Analytics (overhead, tokens, cache_hit=false)

Observability
	├─ /cache/stats, /cache/sweep, /cache/clear
	└─ /analytics/recent (events + aggregates)
```

```json
{
	"session_id": "demo",
	"model": "gpt-4o",
	"instruction": "Respond in short sentences. Be concise!",
	"optimized_prompt": "Respond in short sentences. Be concise!\n\nSummary: Key points: rlhf, reward, human, model, policy\nRecent:\nUser: What is RLHF?\nAssistant:",
	"context_tokens": 64,
	"input_tokens": 81,
	"estimated_cost_usd": 0.405,
	"summary": "Key points: rlhf, reward, human, model, policy",
	"recent_message_count": 1,
	"overhead_ms": 98.7
}
```

### Analytics

```json
{
	"events": [
		{"endpoint": "compose", "cache_hit": false, "overhead_ms": 120.5, "token_savings": 0},
		{"endpoint": "chat", "cache_hit": false, "overhead_ms": 98.7, "input_tokens": 81}
	],
	"stats": {
		"events_total": 2,
		"cache_hit_rate": 0.0,
		"avg_overhead_ms": 109.6,
		"avg_token_savings": 0.0
	}
}
```
