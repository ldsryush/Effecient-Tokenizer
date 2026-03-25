# Efficient Tokenizer Middleware

A drop-in proxy between your application and any LLM API. Requests pass through a multi-stage compression pipeline, hit the model, and responses return through a standard OpenAI-compatible interface — your calling app notices nothing different, but spends fewer tokens on every call.

---

## Architecture

```
Client app  ──POST /v1/chat/completions──►  Tokenizer Middleware Proxy
                                                      │
                                          ┌───────────▼────────────┐
                                          │   Ingress Normalizer   │
                                          │ auth · split · detect  │
                                          └───────────┬────────────┘
                                                      │
                                          ┌───────────▼────────────┐
                                          │  Compression Pipeline  │
                                          │  1. token counter      │
                                          │  2. structural         │
                                          │  3. semantic dedup     │
                                          │  4. relevance score    │
                                          │  5. rolling summary    │
                                          └───────────┬────────────┘
                                                      │
                                          ┌───────────▼────────────┐
                                          │     Cache Router       │
                                          │  KV cache · prefix hit │
                                          └───────────┬────────────┘
                                                      │
                                          ┌───────────▼────────────┐
                                          │    LLM Dispatcher      │
                                          │  route · retry · fbk   │
                                          └───────────┬────────────┘
                                                      │
                                          LLM API  (Claude / GPT / etc.)
                                                      │
                                          ┌───────────▼────────────┐
                                          │  Observability Layer   │
                                          │  telemetry · attribution│
                                          └────────────────────────┘
```

### Modules

| File | Responsibility |
|---|---|
| `app/ingress.py` | Auth, schema parse, model detection, payload splitting |
| `app/tokenizer.py` | Model-accurate token counts (tiktoken / Anthropic heuristic) |
| `app/compressor.py` | Lossless structural compression (JSON minify, key shorten, whitespace) |
| `app/deduplicator.py` | Cosine-similarity deduplication via TF-IDF (or sentence-transformers) |
| `app/relevance.py` | Relevance scoring + pruning of low-signal turns |
| `app/summarizer.py` | Rolling summarizer with entity-graph preservation |
| `app/entity_graph.py` | Entity graph: models conversation as nodes + edges |
| `app/pipeline.py` | Orchestrates all 5 stages, returns `PipelineResult` |
| `app/cache_router.py` | Static-prefix cache detection + full-turn cache store |
| `app/dispatcher.py` | HTTP dispatch to OpenAI / Anthropic with retry + fallback |
| `app/observability.py` | Telemetry bus, attribution log, confidence log, metrics |
| `app/store.py` | KV store — in-memory (default) or Redis (set `STORE_BACKEND=redis`) |
| `app/main.py` | FastAPI app — all endpoints |

---

## Quick Start

### Local (no Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Docker Compose (with Redis)

```bash
OPENAI_API_KEY=sk-... docker compose up --build
```

---

## Usage

### Drop-in proxy (zero code changes)

Change **one line** in your existing OpenAI SDK setup:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-openai-key",
    base_url="http://localhost:8000/v1",   # ← only change
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Explain RLHF."},
    ],
)
```

The response is a standard `chat.completion` object with one extra field `_middleware` that carries compression telemetry.

### Middleware-specific options

Pass these in your request body (ignored by the upstream LLM):

| Field | Default | Description |
|---|---|---|
| `compression_mode` | `"lossy"` | `"lossless"` or `"lossy"` |
| `session_id` | auto | Ties turns to an entity graph for multi-turn memory |
| `max_history_tokens` | `4000` | Budget before summarization fires |
| `relevance_threshold` | `0.15` | Min relevance score to keep a turn |
| `dedup_threshold` | `0.92` | Cosine similarity threshold for deduplication |

---

## Compression Pipeline

### Stage 1 — Tokenizer-aware counter
Counts tokens using the **exact vocabulary for the target model** (tiktoken for OpenAI, char-heuristic for Anthropic). Establishes the before/after baseline.

### Stage 2 — Structural compressor (always lossless)
- JSON minification
- Verbose key shortening (`authorization_token` → `auth_tok`, `configuration` → `cfg`, etc.)
- Whitespace normalization
- Redundant punctuation removal

Typically saves **20–40%** on structured payloads with zero semantic loss.

### Stage 3 — Semantic deduplicator
Embeds each turn and computes cosine similarity. Near-duplicate turns (similarity ≥ `dedup_threshold`) are collapsed — one copy is kept. Uses `sentence-transformers` if installed, falls back to TF-IDF.

### Stage 4 — Relevance scorer
Scores each history turn against the current query. Turns below `relevance_threshold` are pruned in lossy mode. Entity-graph load-bearing turns are **never pruned** regardless of score.

### Stage 5 — Rolling summarizer
When the history is still over `max_history_tokens` after stages 2–4, old turns are compressed into a summary node. The entity graph's named entities (files, tasks, users, code variables) are pinned into the summary so the model never loses load-bearing facts.

---

## Entity Graph

The entity graph models the conversation as a graph rather than a flat list:
- **Nodes**: turns + named entities (files, tasks, users, URLs, constants)
- **Edges**: references from a turn to entities it mentions

When compression runs, it asks: *which turns are load-bearing for the current query?* Turns with entity overlap with the current query are retained. Entity nodes are **always preserved in the summary**.

---

## Observability

Every request is tagged and logged. Three surfaces:

### `/admin/metrics` — Usage dashboard
```json
{
  "events_total": 142,
  "total_token_savings": 28400,
  "total_cost_usd_saved": 0.142,
  "avg_pct_saved": 31.4,
  "avg_overhead_ms": 8.2,
  "avg_confidence": 0.94,
  "by_endpoint": { "/v1/chat/completions": { "count": 142, ... } }
}
```

### `/admin/attribution` — Per-request breakdown
Shows exactly which pipeline stage saved how many tokens — makes the middleware a **debugger**, not just a compressor.

### `/admin/confidence-log` — Auditable drop log
Every lossy operation is recorded with: what was dropped, why, and with what confidence score. Developers can audit and tune aggressiveness per stage.

---

## API Reference

### Proxy
| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible, runs full pipeline |

### Admin
| Method | Path | Description |
|---|---|---|
| GET | `/admin/metrics` | Aggregate usage dashboard |
| GET | `/admin/attribution` | Per-request token attribution |
| GET | `/admin/confidence-log` | Auditable lossy-drop log |
| GET | `/admin/events` | Raw telemetry event stream |
| GET | `/admin/sessions` | Active session list |
| DELETE | `/admin/sessions/{id}` | Delete session + entity graph |
| GET | `/admin/store/stats` | Backing store health |
| GET | `/health` | Liveness check |

### Legacy (backwards compatible)
| Method | Path | Description |
|---|---|---|
| POST | `/compose` | Single-turn optimised prompt builder |
| POST | `/chat` | Multi-turn session manager |
| POST | `/chat/reset` | Clear a session |
| GET | `/cache/stats` | In-process cache stats |
| GET | `/analytics/recent` | Recent telemetry events |

---

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `PROXY_API_KEY` | — | If set, enforces auth on all proxy requests |
| `STORE_BACKEND` | `memory` | `memory` or `redis` |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `DISPATCH_DRY_RUN` | `false` | Skip actual LLM calls (for testing) |
| `LLM_MAX_RETRIES` | `3` | Retry attempts before fallback |
| `LLM_TIMEOUT_S` | `60` | Per-request timeout (seconds) |
| `LLM_FALLBACK_MODEL` | `gpt-4o-mini` | Model used if primary is unavailable |

---

## Running Tests

```bash
python scripts/smoke_test.py
```

No LLM key required — uses `DISPATCH_DRY_RUN=true` automatically.
