# Quorum

> Multi-LLM consensus engine. 8+ models in parallel. Semantic consensus via embeddings. Self-evolves with use. Patent-protected by HSP (PCT/US26/11908).

```bash
pip install quorum-ai
```

```python
from quorum import consensus

result = await consensus("What is the chemical structure of L-Cysteine?")

# result.answer            → final consensus answer
# result.confidence        → semantic agreement score 0..1
# result.models            → [{name, response, weight, latency_ms, cost_usd}]
# result.disagreements     → list of points where models disagreed
# result.evolution_signals → which loops fired this query
```

## What makes Quorum different

- **8+ models in parallel by default**: Claude, GPT, Gemini, Grok, Llama (local), Llama 3.3, Mistral, DeepSeek, Qwen, Phi
- **Semantic consensus, not lexical**: cosine similarity on embeddings, not Jaccard noise
- **Adversarial revision**: round 2 where models see each other's answers and can change their mind
- **13 self-evolution loops**: RLHF, Hebbian, distillation, router, memory, meta-learning, model-vs-model, A/B testing, synthetic data, federated, self-prompting, adversarial, architecture search
- **HSP gate** on every high-stakes decision (patent pending)
- **Auto-certification** for EU AI Act 2026-08 — every query generates an audit-ready PDF
- **Hosted API + BYOK**: run locally with your own keys OR call our managed FastAPI with metered billing
- **Switzerland of LLMs**: the incumbents structurally can't sell this — they'd commoditize themselves

## Why we exist

Anthropic, OpenAI, Google, Microsoft **structurally cannot** sell a multi-vendor consensus engine — it would commoditize their own models. Quorum occupies that gap. Neutral. Auditable. Open source. Patent-protected.

## The 13 self-evolution loops

Each loop closes a feedback gap that single-model deployments leak silently. They run async, behind the consensus call, and write back into router weights, memory, and prompts.

| # | Loop | What it does | File |
|---|------|--------------|------|
| 1 | **RLHF** | Learns from explicit user thumbs / corrections | `evolution/rlhf.py` |
| 2 | **Hebbian** | "Neurons that fire together wire together" — co-correct models get correlated weights | `evolution/hebbian.py` |
| 3 | **Distillation** | Cheap models learn from expensive consensus | `evolution/distillation.py` |
| 4 | **Router** | Per-domain weighting — Claude for code, Gemini for vision, etc. | `evolution/router.py` |
| 5 | **Memory** | Vector recall of past consensus on similar prompts | `evolution/memory_loop.py` + `core/memory.py` |
| 6 | **Meta-learning** | Loops learn which loops are working — second-order updates | `evolution/meta.py` |
| 7 | **Competition** (model-vs-model) | Pairwise duels; ELO-style ranking | `evolution/competition.py` |
| 8 | **A/B testing** | Two prompt variants per call; track which wins | `evolution/ab_testing.py` |
| 9 | **Synthetic data** | High-confidence consensus becomes training data | `evolution/synthetic_data.py` |
| 10 | **Federated** | Cross-tenant signal aggregation without raw data leak | `evolution/federated.py` |
| 11 | **Self-prompting** | Quorum rewrites ambiguous prompts before fan-out | `evolution/self_prompt.py` |
| 12 | **Adversarial** | Red-team prompts; models that fall for them lose weight | `evolution/adversarial.py` |
| 13 | **Architecture search** | Tries new model combos / topologies; promotes winners | `evolution/architecture_search.py` |

## Billing tiers (hosted)

The OSS package is free forever. The hosted API at `api.quorum-ai.com` is metered. BYOK customers pay only platform fees; pass-through customers pay LLM costs at retail + margin.

| Tier | Price / mo | Included | Overage | Notes |
|------|------------|----------|---------|-------|
| **Free** | £0 | 100 queries, 3 models max, no evolution loops | — | Sandbox / dev |
| **Pro** | £49 | 5,000 queries, 8 models, all 13 loops, BYOK | £0.012 / query | For solo devs |
| **Team** | £199 | 25,000 queries, federated loop on, audit log retention 90d | £0.008 / query | Sharing across users |
| **Enterprise** | £1,499 | Unlimited, EU AI Act cert PDFs, SLA 99.9%, dedicated HSP gate | Custom | SSO, on-prem, training data licence |
| **Compliance add-on** | +£500 | Per-query EU AI Act PDF certificate, signed, hash-chained | — | Required for high-risk AI uses 2026-08+ |

Stripe-backed. Webhook handler with in-memory fallback so tests run without keys. See `billing/stripe_billing.py`.

## Hosted API endpoints

FastAPI server in `server/main.py`. Run locally with `quorum-server` or deploy to any container host. All endpoints rate-limited via slowapi; auth via Bearer JWT or BYOK header.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/consensus` | Run a consensus query. Returns `ConsensusResult`. |
| `POST` | `/v1/consensus/stream` | Same, server-sent events as each model returns |
| `GET`  | `/v1/models` | List enabled providers + per-tenant overrides |
| `POST` | `/v1/feedback` | Thumbs up/down on a past query — feeds RLHF loop |
| `GET`  | `/v1/cert/{query_id}` | Download EU AI Act PDF certificate |
| `POST` | `/v1/billing/checkout` | Create Stripe checkout session for tier upgrade |
| `POST` | `/v1/billing/webhook` | Stripe webhook (signature-verified) |
| `POST` | `/v1/hsp/webhook` | HSP gate decision callback (patent PCT/US26/11908) |
| `GET`  | `/v1/usage` | Current period usage + remaining quota |
| `GET`  | `/healthz` | Liveness / readiness |
| `GET`  | `/metrics` | Prometheus scrape endpoint |

## EU AI Act certification (2026-08 deadline)

The EU AI Act enforcement window starts 2026-08-02 for general-purpose AI systems and 2027-08-02 for high-risk uses. Quorum auto-generates a per-query audit certificate that satisfies Art. 13 (transparency) and Art. 12 (record-keeping) obligations:

- Every model that ran, its weight, its raw response
- Consensus method (semantic / lexical), threshold used
- HSP gate decision + signing key
- Cost paid, latency, tokens — for energy-use disclosure
- SHA-256 chain link to previous certificate in tenant (tamper-evident)

PDF generated via reportlab. Stored in tenant bucket; downloadable via `/v1/cert/{query_id}`. Code in `hsp/ai_act_cert.py`.

## Architecture

```
                       ┌─────────────────────────────┐
   user prompt ───────▶│       FastAPI server        │  uvicorn + slowapi rate limit
                       │  /v1/consensus[/stream]     │  Bearer JWT or BYOK header
                       └──────────────┬──────────────┘
                                      │
                                      ▼
                  ┌───────────────────────────────────────┐
                  │  core/consensus.py — orchestrator     │
                  │  async fan-out, max_concurrency=8     │
                  └─────┬───────────────────────────┬─────┘
                        │                           │
                        ▼                           ▼
     ┌─────────────────────────────┐    ┌────────────────────────┐
     │ providers/  (BYOK)          │    │ core/embeddings.py     │
     │  anthropic, gemini, openai, │    │ core/memory.py         │
     │  ollama, replicate, ...     │    │  vector recall + sqlite│
     └─────────────────────────────┘    └────────────────────────┘
                        │                           │
                        └───────────┬───────────────┘
                                    │
                                    ▼
                       ┌────────────────────────────┐
                       │ HSP gate (patent PCT/      │
                       │  US26/11908) — decides if  │
                       │  consensus is binding for  │
                       │  a high-stakes domain      │
                       │  hsp/gate.py               │
                       └─────────────┬──────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
  ┌─────────────────┐   ┌──────────────────────┐  ┌────────────────────┐
  │ 13 evolution    │   │ EU AI Act cert PDF   │  │ Stripe billing     │
  │ loops (async    │   │ hsp/ai_act_cert.py   │  │ billing/           │
  │ writebacks):    │   │ — hash-chained,      │  │  stripe_billing.py │
  │  1 RLHF         │   │   reportlab signed   │  │  webhook handler   │
  │  2 Hebbian      │   └──────────────────────┘  │  in-memory fallback│
  │  3 Distillation │                             └────────────────────┘
  │  4 Router       │
  │  5 Memory       │
  │  6 Meta         │
  │  7 Competition  │
  │  8 A/B          │
  │  9 Synthetic    │
  │ 10 Federated    │
  │ 11 Self-prompt  │
  │ 12 Adversarial  │
  │ 13 Arch search  │
  └─────────────────┘
```

All evolution loops are async writebacks — they never block the response path. Loop outputs feed back into `router` weights, `memory` vectors, and `self_prompt` templates on the next query. Meta-learning audits loop effectiveness and can disable loops that regress.

## Roadmap

| Version | Status | Date |
|---|---|---|
| v0.0.1 | 🟢 5 providers, semantic consensus, CLI | 2026-06-16 |
| v0.1.0 | 🟢 13 evolution loops, vector memory, HSP gate, FastAPI server, Stripe billing, EU AI Act cert | 2026-06-16 |
| v1.0.0 | 🟡 Hosted SaaS public launch, multi-tenant, federated loop GA | Q4 2026 |

## License

Apache 2.0 (core) + HSP Commercial Restrictions on evolution / compliance modules. See [LICENSE-HSP](../LICENSE-HSP).

## Founder

Jaqueline Martins — Sovereign Chain Ltd, UK. Patent holder of [HSP Protocol (PCT/US26/11908)](https://github.com/jaquelinejaque/hsp-protocol).
