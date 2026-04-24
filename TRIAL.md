# Efficient Tokenizer — Trial Setup Guide

> **What this does:** A drop-in proxy that sits between your app and OpenAI/Anthropic.
> Every LLM request passes through a compression pipeline before hitting the model.
> Your app sees a standard OpenAI-compatible interface and notices nothing —
> but spends meaningfully fewer tokens on every call.
>
> **Average savings: 43% on input tokens. Zero code changes beyond one URL.**

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- Your existing OpenAI or Anthropic API key

That's it. No Python, no Node, no configuration files.

---

## Step 1 — Start the proxy (one command)

Open a terminal and run:

```bash
docker run -d \
  --name efficient-tokenizer \
  -p 8000:8000 \
  -e OPENAI_API_KEY="sk-your-key-here" \
  ldsryush/efficient-tokenizer:latest
```

> **Using Anthropic instead of OpenAI?** Replace the env var:
> ```bash
> -e ANTHROPIC_API_KEY="sk-ant-your-key-here"
> ```
>
> **Using both?** Add both `-e` flags.

The proxy starts in about 10 seconds. Verify it's running:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "ok", "store": true, "version": "2.0.0"}
```

---

## Step 2 — Point your app at the proxy

Change **one line** in your application. Everything else stays exactly the same.

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-your-key-here",
    base_url="http://localhost:8000/v1",   # ← only change this line
)

# Everything else is identical to your existing code
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### JavaScript / Node.js

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  apiKey: "sk-your-key-here",
  baseURL: "http://localhost:8000/v1",   // ← only change this line
});
```

### Direct HTTP / curl

```bash
# Replace https://api.openai.com/v1 with http://localhost:8000/v1
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}'
```

> **Your API key is never stored.** The proxy reads it from the Authorization header
> and forwards it directly to OpenAI/Anthropic. It is not logged or persisted anywhere.

---

## Step 3 — Watch your savings in real time

Open the dashboard in your browser:

**http://localhost:8000/dashboard**

You'll see:

| Metric | What it means |
|---|---|
| **Total tokens saved** | Cumulative tokens not sent to the LLM |
| **Avg % saved per request** | Your average compression rate |
| **Total cost saved (USD)** | Dollar value of tokens saved (at your model's price) |
| **Middleware pipeline (ms)** | How long the compression takes — typically 2–7ms |
| **Savings by stage** | Which compression stage saved the most |

The dashboard auto-refreshes every 5 seconds.

---

## What the proxy adds to each response

Every response includes a `_middleware` block with the exact savings for that request:

```json
"_middleware": {
  "raw_tokens": 780,
  "post_tokens": 254,
  "token_savings": 526,
  "pct_saved": 67.4,
  "cost_usd_saved": 0.00263,
  "savings_by_stage": {
    "structural": 0,
    "deduplication": 368,
    "relevance": 158,
    "summarization": 0
  },
  "overhead_ms": 6.9,
  "confidence_score": 0.92
}
```

Every number is auditable. You can verify the savings on any individual request.

---

## Compression modes

Pass `compression_mode` in your request body to control aggressiveness:

| Mode | What it does | Typical savings |
|---|---|---|
| `"lossless"` | Structural only — JSON minify, whitespace, key shortening. Zero semantic loss. | 0–47% |
| `"lossy"` | Full pipeline — adds relevance pruning and deduplication. | 15–67% |

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    extra_body={"compression_mode": "lossy"},   # or "lossless"
)
```

Default is `"lossy"`. Switch to `"lossless"` if you need guaranteed zero information loss.

---

## Admin endpoints (for your engineering team)

| URL | What it shows |
|---|---|
| `http://localhost:8000/dashboard` | Live visual dashboard |
| `http://localhost:8000/admin/metrics` | Aggregate stats (JSON) |
| `http://localhost:8000/admin/attribution` | Per-request savings breakdown |
| `http://localhost:8000/admin/confidence-log` | Log of every lossy operation |
| `http://localhost:8000/health` | Liveness check |

---

## Stopping the proxy

```bash
docker stop efficient-tokenizer
docker rm efficient-tokenizer
```

---

## Frequently asked questions

**Does the proxy store our conversation content?**
No. Each request is processed in memory and discarded immediately. Only aggregate
metrics (token counts, savings percentages) are kept in the in-process buffer.
Message content is never written to disk or any external service.

**Does the proxy store our API key?**
No. The key is read from the Authorization header on each request and forwarded
directly to OpenAI/Anthropic. It is not logged or persisted.

**What if it doesn't save any tokens on a request?**
Nothing changes — the request goes through normally. The proxy never blocks or
modifies a request in a way that would cause it to fail.

**Can we run this inside our own infrastructure?**
Yes. The Docker image runs anywhere Docker runs — your own servers, AWS, GCP, Azure,
or on-premise. Nothing phones home.

**What models are supported?**
OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`
Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`, `claude-3-opus-20240229`

**Can we tune the compression aggressiveness?**
Yes, per request. Pass `relevance_threshold` (default 0.15) and `dedup_threshold`
(default 0.92) in the request body to tune how aggressively the pipeline prunes.

---

## Measured savings by workload type

All numbers measured from live code. Reproducible with the included benchmark scripts.

| Workload | Session length | Lossless savings | Lossy savings |
|---|---|---|---|
| Customer support chat | 10 turns | 0% | **23%** |
| Customer support chat | 20 turns | **45%** | **58%** |
| Coding assistant | 10 turns | 0% | **15%** |
| Coding assistant | 20 turns | **45%** | **53%** |
| Research / Q&A | 10 turns | 0% | **44%** |
| Research / Q&A | 20 turns | **47%** | **70%** |
| **Average** | — | **23%** | **44%** |

---

## Questions or issues?

GitHub: https://github.com/ldsryush/Effecient-Tokenizer

Open an issue or reach out directly. During the trial period, responses within 24 hours.
