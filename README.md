# Efficient Tokenizer Middleware

A drop-in proxy that sits between your app and any LLM API (OpenAI, Anthropic, etc.).
Every request passes through a five-stage compression pipeline before hitting the model.
Your calling app sees a standard OpenAI-compatible interface and notices nothing — but
spends meaningfully fewer tokens on every call.

---

## Table of Contents
1. [How to Run the App](#1-how-to-run-the-app)
2. [Testing It Yourself — curl Commands](#2-testing-it-yourself--curl-commands)
3. [Using It With the OpenAI SDK (Python / JS)](#3-using-it-with-the-openai-sdk)
4. [Architecture Deep-Dive](#4-architecture-deep-dive)
5. [All Endpoints Reference](#5-all-endpoints-reference)
6. [Configuration Reference](#6-configuration-reference)
7. [VS Code Extension — Is It Possible?](#7-vs-code-extension--is-it-possible)

---

## 1. How to Run the App

### Option A — Local Python (fastest to start)

**Step 1: Install dependencies**
```bash
pip install -r requirements.txt
```
This installs FastAPI (the web framework), uvicorn (the ASGI server), tiktoken (OpenAI's
tokenizer), httpx (HTTP client for calling LLM APIs), and redis (optional, for multi-node).

**Step 2: Start the server**
```bash
uvicorn app.main:app --reload --port 8000
```
- `--reload` means the server restarts automatically when you edit a file (dev mode)
- `--port 8000` sets the port (change if 8000 is taken)

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

**Step 3: Verify it's alive**
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{"status": "ok", "store": true, "version": "2.0.0"}
```

---

### Option B — Docker Compose (recommended for real use, includes Redis)

**Step 1: Make sure Docker Desktop is running**

**Step 2: Set your API key (optional — skip if you want dry-run mode)**
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."

# Or create a .env file in the repo root:
echo OPENAI_API_KEY=sk-... > .env
```

**Step 3: Build and start everything**
```bash
docker compose up --build
```
This spins up two containers:
- `proxy` — the middleware on port 8000
- `redis` — Redis on port 6379 (used for the entity graph + prompt cache store)

To run in the background:
```bash
docker compose up --build -d
```

To stop:
```bash
docker compose down
```

---

### Option C — Dry-run mode (no API key needed, no LLM calls)

Set the env var to skip actual LLM API calls. The middleware still runs the full
compression pipeline and returns a placeholder response — perfect for testing:

```bash
# Windows PowerShell
$env:DISPATCH_DRY_RUN = "true"
uvicorn app.main:app --reload --port 8000
```

```bash
# Windows cmd
set DISPATCH_DRY_RUN=true
uvicorn app.main:app --reload --port 8000
```

---

### Run the smoke tests (no API key required)
```bash
python scripts/smoke_test.py
```
This runs 54 checks covering every module end-to-end. Expected output:
```
  54/54 passed   0 failed
```

---

## 2. Testing It Yourself — curl Commands

All commands below assume the server is running on `http://localhost:8000`.
Set `DISPATCH_DRY_RUN=true` to skip real LLM calls.

---

### Health check
```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","store":true,"version":"2.0.0"}
```

---

### POST /v1/chat/completions — The main proxy endpoint

This is the drop-in replacement for the OpenAI endpoint. Send it any standard
OpenAI messages array and it runs the full compression pipeline before dispatching.

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. Your job is to answer questions clearly and concisely. Always provide accurate information. Never make things up. Be professional and courteous at all times."},
      {"role": "user", "content": "What is RLHF?"}
    ],
    "compression_mode": "lossy",
    "session_id": "my-test-session"
  }'
```

**What happens inside this one request:**
1. **Ingress normalizer** — Parses the messages, detects the model is `gpt-4o`, splits the payload into: system prompt (`"You are a helpful assistant..."`) + history (empty) + user message (`"What is RLHF?"`)
2. **Stage 2 — Structural compressor** — Minifies any JSON blobs in the system prompt, shortens verbose key names, collapses whitespace
3. **Stage 3 — Semantic deduplicator** — Computes cosine similarity between history turns (nothing to deduplicate here, only one turn)
4. **Stage 4 — Relevance scorer** — Scores each history turn against "What is RLHF?" — again nothing to prune on the first message
5. **Stage 5 — Rolling summarizer** — Checks if history is over the token budget. Not yet.
6. **Cache router** — Hashes the system prompt, checks for a cache hit. First call = miss, but it warms the static prefix cache so future calls with the same system prompt get a cache hint
7. **LLM dispatcher** — Sends the optimised payload to OpenAI (or returns dry-run placeholder)
8. **Observability** — Logs the event with token counts, savings, confidence score, cost saved

**What comes back:**
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "gpt-4o",
  "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 42, "completion_tokens": 85, "total_tokens": 127},
  "_middleware": {
    "raw_tokens": 58,
    "post_tokens": 42,
    "token_savings": 16,
    "pct_saved": 27.6,
    "cost_usd_saved": 0.00008,
    "savings_by_stage": {"structural": 16, "deduplication": 0, "relevance": 0, "summarization": 0},
    "compression_mode": "lossy",
    "confidence_score": 1.0,
    "overhead_ms": 4.2,
    "cache_hit": false,
    "static_prefix_cached": false
  }
}
```

The `_middleware` block is non-standard — your calling app can ignore it. It tells you
exactly where every token saving came from.

---

### Multi-turn conversation (entity graph in action)

Send multiple turns with the same `session_id`. The entity graph builds up across turns
and protects load-bearing facts from being compressed away:

```bash
# Turn 1 — introduce an entity (file.py)
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "I have a bug in file.py on line 42. The function process_data() crashes."}
    ],
    "session_id": "debug-session-1"
  }'

# Turn 2 — reference the same entity
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "system", "content": "You are a coding assistant."},
      {"role": "user", "content": "I already told you about file.py. Can you help fix it?"},
      {"role": "assistant", "content": "What error does process_data() throw?"},
      {"role": "user", "content": "It throws a KeyError. What should I check first?"}
    ],
    "session_id": "debug-session-1"
  }'
```

The entity graph now knows `file.py` and `process_data(` are load-bearing entities for
this session. Even if the conversation grows long and older turns get summarized away,
the entity facts are pinned into the summary so the model always has them.

---

### Test compression in lossless vs lossy mode

```bash
# Lossless — only structural compression (whitespace, JSON minify, key shortening)
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Summarize transformers."}],
    "compression_mode": "lossless"
  }' | python -m json.tool

# Lossy — full pipeline (relevance pruning + summarization active)
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Summarize transformers."}],
    "compression_mode": "lossy",
    "relevance_threshold": 0.2
  }' | python -m json.tool
```

---

### View the usage dashboard
```bash
curl http://localhost:8000/admin/metrics | python -m json.tool
```
Returns aggregate stats across all requests processed so far — total token savings,
cost saved in USD, average compression percentage, overhead latency, per-endpoint breakdown.

---

### View the attribution log (which stage saved what)
```bash
curl http://localhost:8000/admin/attribution | python -m json.tool
```
Every request is logged with a per-stage breakdown. This is the "debugger" surface —
you can see exactly which part of your prompt costs the most tokens.

---

### View the confidence log (what was dropped and why)
```bash
curl http://localhost:8000/admin/confidence-log | python -m json.tool
```
Every lossy operation is logged here with: what stage performed it, the confidence score
(how sure the compressor was that the drop was safe), and how many tokens were dropped.
Use this to tune aggressiveness — if you see low confidence scores on dropped content,
raise the relevance threshold.

---

### View active sessions (entity graphs in memory)
```bash
curl http://localhost:8000/admin/sessions
```

### Delete a session (clear entity graph + cache)
```bash
curl -X DELETE http://localhost:8000/admin/sessions/my-test-session
```

### View store health (in-memory vs Redis)
```bash
curl http://localhost:8000/admin/store/stats
```

---

### Legacy single-turn compose endpoint
```bash
curl -s -X POST http://localhost:8000/compose \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain the authorization_token configuration parameter in detail with examples and edge cases",
    "model": "gpt-4o",
    "output_mode": "bullets"
  }' | python -m json.tool
```
Returns the optimised prompt with token savings metrics. Does not call the LLM.

---

### Legacy multi-turn chat endpoint
```bash
# Turn 1
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "user_message": "What is gradient descent?", "model": "gpt-4o"}'

# Turn 2
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "s1", "user_message": "How does learning rate affect it?", "model": "gpt-4o"}'
```

---

## 3. Using It With the OpenAI SDK

### Python

```python
from openai import OpenAI

# Change ONLY the base_url — everything else stays exactly the same
client = OpenAI(
    api_key="your-openai-key",       # your real OpenAI key
    base_url="http://localhost:8000/v1",  # point at the middleware
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain RLHF in 3 sentences."},
    ],
)

print(response.choices[0].message.content)

# Access middleware telemetry (non-standard field)
if hasattr(response, '_middleware'):
    print(f"Token savings: {response._middleware['token_savings']}")
    print(f"Cost saved: ${response._middleware['cost_usd_saved']}")
```

### JavaScript / Node.js

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "your-openai-key",
  baseURL: "http://localhost:8000/v1",  // only change
});

const response = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user",   content: "Explain RLHF." },
  ],
});

console.log(response.choices[0].message.content);
console.log(response._middleware); // compression telemetry
```

### With Claude / Anthropic (coming soon)

The dispatcher already supports Anthropic — just pass `model: "claude-3-5-sonnet-20241022"`.
Set `ANTHROPIC_API_KEY` in the environment. The middleware automatically injects
`cache_control` on the system prompt when a static-prefix cache hit is detected,
unlocking Anthropic's native prompt caching.

---

## 4. Architecture Deep-Dive

### How the request flows

```
Your app
  │
  │  POST /v1/chat/completions
  │  {model, messages[], compression_mode, session_id, ...}
  ▼
┌─────────────────────────────────────────────────────────────────┐
│                  TOKENIZER MIDDLEWARE PROXY                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 1. INGRESS NORMALIZER                                   │   │
│  │    • Verify auth (PROXY_API_KEY env, if set)            │   │
│  │    • Detect model → load correct tokenizer vocab        │   │
│  │    • Split messages into 3 parts:                       │   │
│  │        system_prompt  ← primary compression target      │   │
│  │        history        ← subject to full pipeline        │   │
│  │        user_message   ← NEVER modified, always intact   │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ 2. COMPRESSION PIPELINE (5 stages)                      │   │
│  │                                                         │   │
│  │  Stage 1 — Token counter                                │   │
│  │    Counts tokens using tiktoken (OpenAI) or             │   │
│  │    char-heuristic (Anthropic). Establishes the          │   │
│  │    "before" baseline that drives the savings metric.    │   │
│  │                                                         │   │
│  │  Stage 2 — Structural compressor  [always lossless]     │   │
│  │    • JSON minification ({"key":  "v"} → {"key":"v"})    │   │
│  │    • Key shortening (authorization_token → auth_tok,    │   │
│  │      configuration → cfg, description → desc, etc.)     │   │
│  │    • Whitespace normalisation (3+ blank lines → 2)      │   │
│  │    • Redundant punctuation removal (!!!→!, ???→?)        │   │
│  │    Typical saving: 20–40% on structured prompts.        │   │
│  │    Confidence: always 1.0 (zero semantic loss).         │   │
│  │                                                         │   │
│  │  Stage 3 — Semantic deduplicator  [lossless or lossy]   │   │
│  │    • Embeds each history turn as a vector               │   │
│  │    • Computes cosine similarity between every pair      │   │
│  │    • Turns with similarity ≥ dedup_threshold are        │   │
│  │      collapsed — one copy kept, duplicates removed      │   │
│  │    • Uses sentence-transformers if installed,           │   │
│  │      falls back to TF-IDF (no extra deps needed)        │   │
│  │    Confidence: the similarity score of the dropped turn.│   │
│  │                                                         │   │
│  │  Stage 4 — Relevance scorer       [lossless or lossy]   │   │
│  │    • Scores each history turn against the current query │   │
│  │    • Turns below relevance_threshold are pruned         │   │
│  │    • Entity-graph protected turns are NEVER pruned      │   │
│  │      regardless of score (see Entity Graph below)       │   │
│  │    Confidence: the relevance score of the pruned turn.  │   │
│  │                                                         │   │
│  │  Stage 5 — Rolling summarizer     [lossy]               │   │
│  │    • Fires only when history is still over              │   │
│  │      max_history_tokens after stages 2–4                │   │
│  │    • Keeps the last N turns verbatim                    │   │
│  │    • Compresses older turns into a keyword summary      │   │
│  │    • Entity graph facts are pinned into the summary     │   │
│  │      so the model never loses load-bearing context      │   │
│  │    Confidence: 0.85 (lossy by definition).              │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ 3. ENTITY GRAPH                                         │   │
│  │    Runs in parallel with the pipeline. Models the       │   │
│  │    conversation as a graph, not a flat list:            │   │
│  │                                                         │   │
│  │    Nodes: turns + named entities                        │   │
│  │    Edges: references from a turn to its entities        │   │
│  │                                                         │   │
│  │    Entity types detected (regex-based, no ML needed):   │   │
│  │      file      → file.py, config.yaml, main.ts          │   │
│  │      function  → process_data(, handle_request(         │   │
│  │      task      → task #42, issue #7, PR #100            │   │
│  │      user      → @username, "my name is Alice"          │   │
│  │      variable  → OPENAI_API_KEY, MAX_TOKENS             │   │
│  │      url       → https://api.openai.com/v1              │   │
│  │                                                         │   │
│  │    Load-bearing turns: turns whose entity set has       │   │
│  │    high overlap with the current query's entities.      │   │
│  │    These are immune to relevance pruning.               │   │
│  │                                                         │   │
│  │    Entity summary line is always appended to the        │   │
│  │    rolling summary so no entity is ever lost.           │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ 4. CACHE ROUTER                                         │   │
│  │    Two levels of caching:                               │   │
│  │                                                         │   │
│  │    Full-turn cache                                      │   │
│  │      Hash of (system + history + user_message).         │   │
│  │      If hit: return stored response immediately.        │   │
│  │      Zero LLM calls, zero latency beyond hash lookup.   │   │
│  │                                                         │   │
│  │    Static-prefix cache                                  │   │
│  │      Hash of (system_prompt + model) only.              │   │
│  │      If hit: tag the outbound request so the dispatcher │   │
│  │      injects cache_control (Anthropic) or relies on     │   │
│  │      OpenAI's automatic prompt caching. Reduces the     │   │
│  │      cost of repeated system prompts significantly.     │   │
│  │                                                         │   │
│  │    Store backend: in-memory (default) or Redis          │   │
│  │    (set STORE_BACKEND=redis for multi-node deployments) │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ 5. LLM DISPATCHER                                       │   │
│  │    • Routes to correct endpoint:                        │   │
│  │        openai → /v1/chat/completions                    │   │
│  │        anthropic → /v1/messages                         │   │
│  │    • Injects Anthropic cache_control on system prompt   │   │
│  │      when static prefix is already cached               │   │
│  │    • Retry with exponential backoff on 429/5xx          │   │
│  │      (0.5s, 1s, 2s, 4s... up to LLM_MAX_RETRIES)       │   │
│  │    • Fallback to LLM_FALLBACK_MODEL on final failure    │   │
│  │    • httpx if installed, urllib fallback (no extra dep) │   │
│  └──────────────────────────┬──────────────────────────────┘   │
│                             │                                   │
│  ┌──────────────────────────▼──────────────────────────────┐   │
│  │ 6. OBSERVABILITY LAYER                                  │   │
│  │    Every request is tagged and written to 3 buffers:    │   │
│  │                                                         │   │
│  │    Telemetry bus (last 1000 events)                     │   │
│  │      raw_tokens, post_tokens, pct_saved, cost_saved,    │   │
│  │      overhead_ms, compression_mode, confidence_score,   │   │
│  │      savings_by_stage {structural, dedup, relevance,    │   │
│  │      summarization}                                     │   │
│  │                                                         │   │
│  │    Attribution log (last 500 events)                    │   │
│  │      Per-stage savings breakdown. Use this to debug     │   │
│  │      which part of your prompt costs the most.         │   │
│  │                                                         │   │
│  │    Confidence log (last 1000 events)                    │   │
│  │      Auditable record of every lossy operation:         │   │
│  │      what was dropped, which stage did it, confidence.  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
  │
  │  Standard OpenAI chat.completion response
  │  + _middleware telemetry block
  ▼
Your app
```

---

### The Entity Graph in detail

Without an entity graph, compression is purely text-based — it can accidentally drop a
turn that introduced a critical fact (a filename, a task ID, a user's name) because the
turn scored low on recency or keyword overlap.

The entity graph prevents this by modelling *what the conversation is about*, not just
*what the conversation says*:

```
Turn 1: "I'm working on file.py for task #42."
  Entities: {file: ["file.py"], task: ["#42"]}

Turn 2: "The process_data() function crashes."
  Entities: {function: ["process_data("]}

Turn 7 (current query): "How do I fix file.py?"
  Entities: {file: ["file.py"]}
```

The graph sees that `file.py` is referenced in Turn 1 AND the current query.
Turn 1 is therefore **load-bearing** — it gets a free pass through the relevance pruner
even if it scored low by pure keyword similarity. The turn that introduced `file.py`
is also protected regardless of how old it is.

Entity nodes are always carried forward in the rolling summary:
```
[Summary] Key topics: crash, function, task | [Entities] files: file.py; functions: process_data(; tasks: #42
```

The model always has the facts it needs even when the original turns have been compressed.

---

### Confidence scores explained

Every compression operation produces a confidence score between 0 and 1:

| Stage | Confidence meaning |
|---|---|
| Structural | Always 1.0 — lossless, no information is removed |
| Deduplication | The cosine similarity of the dropped duplicate. 0.98 = very safe drop. |
| Relevance pruning | The relevance score of the pruned turn. Low score = low risk drop. |
| Summarization | Fixed at 0.85 — lossy by nature, you should always review |

The **overall confidence** for a request is the **minimum** across all stages. This is
what appears in `_middleware.confidence_score` and the confidence log.

A request with `confidence_score: 1.0` means only lossless operations were applied.
A request with `confidence_score: 0.72` means something was dropped with 72% confidence
it was safe — check the confidence log to see exactly what.

---

## 5. All Endpoints Reference

### Proxy
| Method | Path | What it does |
|---|---|---|
| POST | `/v1/chat/completions` | Full pipeline → LLM dispatch → OpenAI-compatible response |

**Request body:**
```json
{
  "model": "gpt-4o",
  "messages": [...],
  "compression_mode": "lossy",
  "session_id": "optional-string",
  "max_history_tokens": 4000,
  "relevance_threshold": 0.15,
  "dedup_threshold": 0.92,
  "temperature": 0.7,
  "max_tokens": 1024
}
```

### Admin
| Method | Path | What it does |
|---|---|---|
| GET | `/admin/metrics` | Aggregate usage dashboard — savings, costs, latency |
| GET | `/admin/attribution` | Per-request token attribution by stage |
| GET | `/admin/confidence-log` | Auditable log of every lossy drop |
| GET | `/admin/events` | Raw telemetry event stream |
| GET | `/admin/sessions` | List of active session IDs (entity graphs in memory) |
| DELETE | `/admin/sessions/{id}` | Delete a session and its entity graph |
| GET | `/admin/store/stats` | KV store health and size |
| GET | `/health` | Liveness check |

### Legacy (backwards compatible)
| Method | Path | What it does |
|---|---|---|
| POST | `/compose` | Single-turn prompt optimisation, returns metrics without calling LLM |
| POST | `/chat` | Multi-turn session-aware prompt builder |
| POST | `/chat/reset` | Clear a session |
| GET | `/cache/stats` | In-process cache stats |
| POST | `/cache/sweep` | Evict expired cache entries |
| POST | `/cache/clear` | Clear the entire cache |
| GET | `/analytics/recent` | Recent telemetry events (legacy format) |

---

## 6. Configuration Reference

Set these as environment variables before starting the server.

| Variable | Default | What it does |
|---|---|---|
| `OPENAI_API_KEY` | *(none)* | Your OpenAI key — forwarded to OpenAI as `Bearer` token |
| `ANTHROPIC_API_KEY` | *(none)* | Your Anthropic key — forwarded as `x-api-key` |
| `PROXY_API_KEY` | *(none)* | If set, the proxy enforces this key on all incoming requests (`Authorization: Bearer <key>` or `X-Api-Key: <key>`). Leave empty to disable auth. |
| `STORE_BACKEND` | `memory` | `memory` for single-node in-process store. `redis` for multi-node (requires `REDIS_URL`). |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL. Only used when `STORE_BACKEND=redis`. |
| `DISPATCH_DRY_RUN` | `false` | Set `true` to skip all LLM API calls. The pipeline still runs — returns a placeholder response. Useful for testing and benchmarking compression. |
| `LLM_MAX_RETRIES` | `3` | How many times to retry a failed LLM call before switching to fallback model. |
| `LLM_TIMEOUT_S` | `60` | Per-request timeout in seconds for LLM calls. |
| `LLM_FALLBACK_MODEL` | `gpt-4o-mini` | Model to use if the primary model fails all retries. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Override to point at a different OpenAI-compatible API (e.g., Azure OpenAI, local Ollama). |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com/v1` | Override Anthropic base URL. |

**Setting env vars — Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY = "sk-..."
$env:DISPATCH_DRY_RUN = "true"
$env:STORE_BACKEND = "redis"
$env:REDIS_URL = "redis://localhost:6379/0"
uvicorn app.main:app --reload --port 8000
```

**Setting env vars — .env file (Docker Compose reads this automatically):**
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
PROXY_API_KEY=my-secret-proxy-key
STORE_BACKEND=redis
LLM_FALLBACK_MODEL=gpt-4o-mini
```

---

## 7. How to Share This With Others

There are three ways to distribute this, from simplest to most polished.

---

### Option 1 — Share the GitHub repo (simplest, right now)

Your repo is already at `https://github.com/ldsryush/Effecient-Tokenizer`.

Anyone who wants to use it does this in 3 commands:

```bash
git clone https://github.com/ldsryush/Effecient-Tokenizer.git
cd Effecient-Tokenizer
pip install -r requirements.txt
```

Then start it:
```bash
# No API key — dry run mode (no real LLM calls, pipeline still runs):
set DISPATCH_DRY_RUN=true
uvicorn app.main:app --port 8000

# With an OpenAI key:
set OPENAI_API_KEY=sk-...
uvicorn app.main:app --port 8000
```

Open the dashboard in a browser: **http://localhost:8000/dashboard**

That's it. Point any LLM app at `http://localhost:8000/v1` and it routes through compression.

---

### Option 2 — Publish to PyPI (pip install)

This lets anyone install it with a single command: `pip install efficient-tokenizer`

**Step 1: Create a PyPI account**
Go to https://pypi.org/account/register/

**Step 2: Build the package**
```bash
pip install build twine
python -m build
```
This creates `dist/efficient_tokenizer-2.0.0.tar.gz` and a `.whl` file.

**Step 3: Upload to PyPI**
```bash
twine upload dist/*
```
Enter your PyPI username and password (or API token).

**That's it.** Anyone can now install and run it with:
```bash
pip install efficient-tokenizer
efficient-tokenizer serve                  # starts on :8000
efficient-tokenizer serve --port 9000      # custom port
efficient-tokenizer serve --dry-run        # no real LLM calls
efficient-tokenizer test                   # run smoke tests
```

The CLI also prints:
```
[efficient-tokenizer] Starting proxy on http://0.0.0.0:8000
[efficient-tokenizer] Admin dashboard: http://localhost:8000/dashboard
[efficient-tokenizer] Health check:    http://localhost:8000/health
[efficient-tokenizer] Docs:            http://localhost:8000/docs
```

**Optional extras people can install:**
```bash
pip install "efficient-tokenizer[redis]"   # Redis-backed store for multi-node
pip install "efficient-tokenizer[embed]"   # sentence-transformers for better dedup
pip install "efficient-tokenizer[all]"     # everything
```

---

### Option 3 — VS Code Marketplace extension (most polished)

The extension lives in `vscode-extension/`. It:
- Auto-starts the proxy server when VS Code opens
- Shows a **live token savings counter** in the status bar: `⚡ ET: 4,231 tokens saved (28.4% avg)`
- Click the status bar item → opens the full dashboard in a VS Code panel
- Has a sidebar icon in the activity bar
- Settings panel for port, compression mode, API key, dry-run

**File layout:**
```
vscode-extension/
  package.json          ← extension manifest
  tsconfig.json         ← TypeScript config
  .vscodeignore         ← files excluded from the .vsix package
  src/
    extension.ts        ← all extension logic (compiled to out/extension.js)
  media/
    sidebar-icon.svg    ← activity bar icon
    icon.png            ← marketplace listing icon (128x128, add yours)
  out/
    extension.js        ← compiled output (generated by npm run compile)
```

**How to run the extension locally (test before publishing):**
```bash
cd vscode-extension
npm install
npm run compile
```
Then in VS Code: Press **F5** → "Run Extension" → opens a new VS Code window with the
extension active. You'll see the status bar item and the sidebar icon immediately.

**How to package it as a `.vsix` file (share without the Marketplace):**
```bash
cd vscode-extension
npm install -g @vscode/vsce
vsce package
# produces: efficient-tokenizer-2.0.0.vsix
```
Send the `.vsix` file to anyone. They install it with:
```
Extensions panel → ... → Install from VSIX
```
Or from the command line:
```bash
code --install-extension efficient-tokenizer-2.0.0.vsix
```

**How to publish to the VS Code Marketplace:**

1. Go to https://marketplace.visualstudio.com/manage and create a publisher ID
2. Get a Personal Access Token from Azure DevOps
3. Update `package.json` → set `"publisher": "your-publisher-id"`
4. Add a 128x128 PNG as `vscode-extension/media/icon.png`
5. Run:
```bash
cd vscode-extension
vsce login your-publisher-id
vsce publish
```
Done — it will appear on the Marketplace within minutes. Anyone can find it by searching
"Efficient Tokenizer" and install it with one click.

**After the extension is installed, using it with Cline:**
The extension starts the proxy automatically. To wire Cline through it, open VS Code settings
(`Ctrl+,`) and search for Cline's base URL setting:
```json
{
  "cline.openaiBaseUrl": "http://localhost:8000/v1"
}
```
Every Cline request now flows through the compression pipeline. The status bar shows live savings.

---

### Option 4 — Docker Hub (for server/team deployments)

Build and push to Docker Hub so anyone can run it with a single `docker run`:

```bash
# Build
docker build -t yourusername/efficient-tokenizer:latest .

# Push to Docker Hub (create account at hub.docker.com)
docker login
docker push yourusername/efficient-tokenizer:latest
```

Anyone can then run it with:
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  yourusername/efficient-tokenizer:latest
```

Or with Docker Compose (Redis included):
```bash
# Their docker-compose.yml just needs:
services:
  proxy:
    image: yourusername/efficient-tokenizer:latest
    ports: ["8000:8000"]
    environment:
      OPENAI_API_KEY: "sk-..."
      STORE_BACKEND: redis
      REDIS_URL: redis://redis:6379/0
  redis:
    image: redis:7-alpine
```

---

### Quick comparison

| Method | Who it's for | Time to ship | User friction |
|---|---|---|---|
| GitHub repo | Developers | Already done | Clone + pip install |
| PyPI package | Python developers | ~30 min | `pip install efficient-tokenizer` |
| VS Code `.vsix` | VS Code users | ~15 min to build | Install from VSIX |
| VS Code Marketplace | Everyone | ~1 hr + review | One-click install |
| Docker Hub | Teams / servers | ~20 min | `docker run` |

---

## 8. VS Code Extension — Is It Possible?

**Short answer: Yes, absolutely — and it's actually a great fit.**

### What a VS Code extension would look like

A VS Code extension wrapping this middleware could:
- Add a sidebar panel (like Cline/GitHub Copilot Chat) where you chat with the LLM
- Show a live token savings counter in the status bar
- Show a "Compression Report" panel with attribution data after each message
- Let you configure compression aggressiveness (lossless vs lossy, threshold sliders)
- Intercept any LLM call made from within VS Code and route it through the middleware

### How to build it (high level)

VS Code extensions are written in TypeScript. The extension would:

1. **Start the middleware** — either spawn the Python process in the background (`uvicorn app.main:app`) or connect to one the user has already started
2. **Create a WebviewPanel** — VS Code's sidebar chat UI (exactly what Cline uses) built with React or plain HTML/JS
3. **Send requests to the proxy** — the webview POSTs to `http://localhost:8000/v1/chat/completions` instead of directly to OpenAI
4. **Display telemetry** — read `response._middleware` and show token savings, confidence score, savings by stage in the UI

### Skeleton extension structure
```
vscode-efficient-tokenizer/
  package.json          ← extension manifest (activation events, commands)
  src/
    extension.ts        ← activation, commands, status bar
    panel.ts            ← WebviewPanel for the chat sidebar
    proxy.ts            ← starts/connects to the Python proxy process
    telemetry.ts        ← reads _middleware fields, formats for display
  media/
    main.js             ← webview JS (sends messages to extension host)
    style.css
```

**`package.json` key fields:**
```json
{
  "contributes": {
    "viewsContainers": {
      "activitybar": [{"id": "efficientTokenizer", "title": "Efficient Tokenizer"}]
    },
    "views": {
      "efficientTokenizer": [{"id": "chatView", "type": "webview", "name": "Chat"}]
    },
    "commands": [
      {"command": "efficientTokenizer.start", "title": "Start Tokenizer Proxy"},
      {"command": "efficientTokenizer.showMetrics", "title": "Show Token Metrics"}
    ]
  }
}
```

### Integration with Cline specifically

Cline (and other extensions that use the OpenAI SDK) already support custom `baseURL`.
If you set the Cline config to point `base_url` at `http://localhost:8000/v1`, Cline
will automatically route through this middleware with **zero changes to Cline's code**.

In VS Code settings (`settings.json`):
```json
{
  "cline.openaiBaseUrl": "http://localhost:8000/v1",
  "cline.openaiApiKey": "your-real-openai-key"
}
```

Every Cline request would then flow through the full compression pipeline. You'd see
token savings on every coding assistant interaction.

### Steps to make this happen

1. Run the proxy locally: `uvicorn app.main:app --port 8000`
2. In VS Code settings, point Cline (or any other LLM extension) at `http://localhost:8000/v1`
3. Use the extension as normal — the middleware runs invisibly in the background
4. Check `http://localhost:8000/admin/metrics` to see cumulative savings

The VS Code extension would just be a nicer wrapper around this workflow — adding a
sidebar UI, a status bar indicator, and a settings panel, rather than requiring manual
server management.
