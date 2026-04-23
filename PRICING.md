# Pricing

> **Pay when we save you money. Nothing when we don't.**

No subscriptions. No minimums. No upfront cost. You pay a percentage of the token savings
we actually deliver — verified on every request by the `_middleware` block in each response.

---

## Tiers

### 🟢 Open Source — Free Forever
**Self-hosted. MIT licensed. No strings.**

Run the full pipeline on your own infrastructure. Every feature, no limits, no telemetry
sent anywhere. Ideal for developers, researchers, and any organization with data-residency
requirements that prevent sending conversation context to a third-party service.

```bash
git clone https://github.com/ldsryush/Effecient-Tokenizer.git
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

Includes:
- Full 5-stage compression pipeline (structural → dedup → relevance → summarizer → cache)
- Entity graph with session continuity
- Built-in observability dashboard (`/dashboard`)
- OpenAI + Anthropic support
- Redis-backed store for multi-node deployments
- All admin endpoints (`/admin/metrics`, `/admin/attribution`, `/admin/confidence-log`)

---

### ☁️ Cloud Starter — Free
**Up to $500/month in token savings.**

Hosted proxy. Point your app at our endpoint instead of OpenAI's. Two lines of code.
No infrastructure to manage.

```python
client = OpenAI(
    api_key="your-openai-key",
    base_url="https://api.efficienttokenizer.com/v1",
)
```

Includes everything in Open Source, plus:
- Hosted, managed infrastructure
- Web dashboard with live savings tracking
- All major providers: OpenAI, Anthropic, Google, Mistral, DeepSeek
- BYOK (Bring Your Own Key) — we never store your API keys
- Community support

---

### 🚀 Cloud Growth — 15% of net savings
**$500–$5,000/month in savings.**

For growing teams where token costs are becoming a real line item.

Includes everything in Starter, plus:
- Custom compression tuning (per-endpoint thresholds, entity type configuration)
- SSO / team access controls
- Priority email support (< 4 hour response)
- Monthly savings report with per-session attribution

**Example:** Your team spends $8,000/month on GPT-4o input tokens. We reduce that by
43% on average → $3,440/month saved. You pay 15% = **$516/month**. You net **$2,924/month**.

---

### 🏢 Cloud Scale — 20% of net savings
**$5,000–$50,000/month in savings.**

For mid-market companies running high-volume agentic or customer-facing LLM workloads.

Includes everything in Growth, plus:
- Dedicated Slack channel with engineering access
- 99.9% uptime SLA
- Multi-region deployment
- Audit logs (SOC 2 compatible)
- Custom entity type configuration (add your own entity patterns)
- Quarterly business review

---

### 🔒 Enterprise — Custom terms
**$50,000+/month in savings.**

For large organizations with compliance, data-residency, or on-premise requirements.

Includes everything in Scale, plus:
- **Self-hosted cloud option** — we deploy and manage the stack inside your VPC
- SOC 2 Type II report
- MSA / custom legal terms
- Custom SLA (up to 99.99%)
- Dedicated customer success engineer
- On-premise deployment support
- Custom model support (Azure OpenAI, private endpoints, local Ollama)

> **This is the tier OpenCompress and Kompact cannot offer.** If your legal, security,
> or compliance team won't allow conversation context to leave your infrastructure,
> the Enterprise self-hosted option is the only production-ready path in the market.

---

## How savings are calculated

Every response includes a `_middleware` block with verified measurements:

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
  }
}
```

`cost_usd_saved` is computed using the actual input token price for your model.
Billing is based on the sum of `cost_usd_saved` across all requests in the billing period.
You can audit every number at any time via `/admin/attribution`.

---

## Measured savings by workload type

All numbers measured from live code. Reproducible with `python -m scripts.savings_benchmark`.

| Workload | Turns | Lossless savings | Lossy savings |
|---|---|---|---|
| Customer support chat | 10 | 0% | **23%** |
| Customer support chat | 20 | **45%** | **58%** |
| Coding assistant session | 10 | 0% | **15%** |
| Coding assistant session | 20 | **45%** | **53%** |
| Research / long-form Q&A | 10 | 0% | **38%** |
| Research / long-form Q&A | 20 | **47%** | **67%** |
| **Average across all scenarios** | — | **23%** | **43%** |

---

## Cost impact at scale (GPT-4o, $5.00/1M input tokens)

| Daily volume | Baseline/month | After lossless | After lossy | Monthly savings |
|---|---|---|---|---|
| 1,000 req/day | $65.97 | $46.30 | $33.50 | up to **$32.47** |
| 10,000 req/day | $659.75 | $463.00 | $335.00 | up to **$324.75** |
| 100,000 req/day | $6,597.50 | $4,630.00 | $3,350.00 | up to **$3,247.50** |

---

## Processing overhead

The compression pipeline adds the following overhead per request (wall-clock, Python only,
no LLM call time included). Measured with `python -m scripts.benchmark_pipeline`.

| Session size | Overhead |
|---|---|
| Short (4 turns, ~180 tokens) | **1.5 ms** |
| Typical (20 turns, ~700 tokens) | **6.9 ms** |
| Long (60 turns, ~2,000 tokens) | **21.5 ms** |

For context: GPT-4o typically takes 500–2,000 ms to respond. The middleware overhead
is less than 1% of total request latency for typical sessions.

---

## FAQ

**Do you store my conversation data?**
Cloud tiers: No. We process each request in memory and discard it immediately. We store
only aggregate metrics (token counts, savings percentages) — never message content.
Enterprise self-hosted: Nothing leaves your infrastructure at all.

**What if the middleware doesn't save any tokens on a request?**
You pay nothing for that request. The "pay when we save you money" model means our
incentives are perfectly aligned with yours.

**Can I switch between lossless and lossy mode per request?**
Yes. Pass `"compression_mode": "lossless"` or `"compression_mode": "lossy"` in each
request body. You can also tune `relevance_threshold` and `dedup_threshold` per request.

**What happens to my entity graph if I switch to self-hosted?**
The entity graph is stored in-process (or Redis for multi-node). It's fully portable —
export via `/admin/sessions` and import into any deployment.

**Is there a free trial of the Cloud tiers?**
The Starter tier is permanently free up to $500/month in savings. No trial period needed.

---

*Pricing subject to change with 30 days notice for existing customers.*
*All savings figures are measured on realistic conversation workloads. Your actual savings
will vary based on conversation length, repetition patterns, and compression mode.*
