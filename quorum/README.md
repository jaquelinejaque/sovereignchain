# Quorum

[![License](https://img.shields.io/badge/License-FSL--1.1-blue.svg)](LICENSE)

> Multi-LLM consensus engine. 14+ models in parallel. Semantic consensus via embeddings. Self-evolves with use. Patent Pending: HSP (PCT/US26/11908).

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

> **Pro tier: £149/mo** — paid commercial product, BYOK, all evolution loops, 5,000 queries/mo, HSP audit chain.
> [Buy on Stripe](https://buy.stripe.com/aFadR9d6E5rf8JGeINdwc0j) · Source-available under FSL-1.1 (Apache-2.0 in 2028).

## Recommended usage pattern — context profiles

Before asking Quorum anything substantial, create a context profile for the project or domain you're working in. The profile is auto-injected into every consensus query so all the LLMs start from the same ground truth instead of falling back to training-data priors.

```bash
quorum context add my-project --file README.md   # one-time setup
quorum ask --all "what should I prioritize next?"  # context auto-injects, web is live by default
```

Full guide: [docs/CONTEXT_PROFILES.md](docs/CONTEXT_PROFILES.md) (also documents the failure mode this prevents — same idea as Claude Projects / Cursor `.cursorrules` / ChatGPT Custom Instructions, applied to multi-LLM consensus).

## What makes Quorum different

- **8+ models in parallel by default**: Claude, GPT, Gemini, Grok, Llama (local), Llama 3.3, Mistral, DeepSeek, Qwen, Phi
- **Semantic consensus, not lexical**: cosine similarity on embeddings, not Jaccard noise
- **Adversarial revision**: round 2 where models see each other's answers and can change their mind
- **13 self-evolution loops**: RLHF, Hebbian, distillation, router, memory, meta-learning, model-vs-model, A/B testing, synthetic data, federated, self-prompting, adversarial, architecture search
- **HSP gate** on every high-stakes decision (patent pending)
- **EU AI Act readiness toolkit** for the 2026-08 enforcement window — every query generates a tamper-evident PDF record as internal evidence material (advisory; not a conformity assessment)
- **Hosted API + BYOK**: run locally with your own keys OR call our managed FastAPI with metered billing
- **Switzerland of LLMs**: the incumbents structurally can't sell this — they'd commoditize themselves

## Why we exist

Anthropic, OpenAI, Google, Microsoft **structurally cannot** sell a multi-vendor consensus engine — it would commoditize their own models. Quorum occupies that gap. Neutral. Auditable. Open source. Patent Pending.

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

The OSS package is free forever. The hosted API at `api.quorum-ai.com` is metered. **BYOK only — Quorum never proxies your provider keys.** You pay the platform fee; your LLM spend stays on your own Anthropic / OpenAI / Gemini / Grok bills.

### Pro — £49/mo (start here)

| Tier | Price / mo | Included | Overage |
|------|------------|----------|---------|
| **Pro** | **£49** | 5,000 queries, 8 models in parallel, all 13 evolution loops, BYOK | £0.012 / query |

**Why Pro is the right tier for you.** If you're a solo backend dev, indie hacker, or an agency engineer shipping LLM features under your own name, Pro is built for your workflow. You get the full consensus engine — 8 models, semantic agreement, every self-evolution loop — at a price that fits a single-developer P&L, and you keep your own provider keys so there's nothing to migrate when you scale. No seat minimums, no procurement call, no "contact sales" wall between you and shipping.

Sign up at https://quorum-ai.dev — 30 seconds, Stripe-backed, cancel any time.

### Free sandbox

| Tier | Price / mo | Included |
|------|------------|----------|
| **Free** | £0 | 100 queries, 3 models max, no evolution loops — for dev/test only |

### Higher tiers (talk to us: jaqueline@hsp-protocol.com)

For multi-user accounts, regulated workloads, or the EU AI Act PDF certification path, the following exist but are deliberately out of the self-serve flow. Email if you need them.

| Tier | Price / mo | Included | Overage |
|------|------------|----------|---------|
| Team | £199 | 25,000 queries, federated loop on, traceability log retention 90d | £0.008 / query |
| Enterprise | £1,499 | Unlimited, SLA 99.9%, SSO, on-prem, training data licence | Custom |
| Readiness add-on | +£500 | Per-query EU AI Act PDF evidence record, signed, hash-chained (advisory toolkit) | — |

Stripe-backed. Webhook handler with in-memory fallback so tests run without keys. See `billing/stripe_billing.py`.

## Hosted API endpoints

FastAPI server in `server/main.py`. Run locally with `quorum-server` or deploy to any container host. All endpoints rate-limited via slowapi; auth via Bearer JWT or BYOK header.

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/v1/consensus` | Run a consensus query. Returns `ConsensusResult`. |
| `POST` | `/v1/consensus/stream` | Same, server-sent events as each model returns |
| `GET`  | `/v1/models` | List enabled providers + per-tenant overrides |
| `POST` | `/v1/feedback` | Thumbs up/down on a past query — feeds RLHF loop |
| `GET`  | `/v1/cert/{query_id}` | Download EU AI Act PDF evidence record (advisory) |
| `POST` | `/v1/billing/checkout` | Create Stripe checkout session for tier upgrade |
| `POST` | `/v1/billing/webhook` | Stripe webhook (signature-verified) |
| `POST` | `/v1/hsp/webhook` | HSP gate decision callback (patent PCT/US26/11908) |
| `GET`  | `/v1/usage` | Current period usage + remaining quota |
| `GET`  | `/healthz` | Liveness / readiness |
| `GET`  | `/metrics` | Prometheus scrape endpoint |

## EU AI Act readiness toolkit (2026-08 deadline)

The EU AI Act enforcement window starts 2026-08-02 for general-purpose AI systems and 2027-08-02 for high-risk uses. Quorum generates a per-query PDF **evidence record** that helps providers prepare internal documentation referenced by Art. 13 (transparency) and Art. 12 (record-keeping) obligations:

- Every model that ran, its weight, its raw response
- Consensus method (semantic / lexical), threshold used
- HSP gate decision + signing key
- Cost paid, latency, tokens — for energy-use disclosure
- SHA-256 chain link to previous record in tenant (tamper-evident)

PDF generated via reportlab. Stored in tenant bucket; downloadable via `/v1/cert/{query_id}`. Code in `hsp/ai_act_cert.py`.

> **Legal notice.** Quorum is an **advisory technical toolkit**. It does **not** perform a conformity assessment under Regulation (EU) 2024/1689 and Sovereign Chain Ltd is **not a Notified Body** under Article 31 of that Regulation. Final conformity assessment remains the responsibility of the AI system provider (internal, Annex VI) or a designated Notified Body (external, Annex VII).

## Tamper-evident traceability log — HSP Black Box

Every consensus() call appends to a tamper-evident SHA-256 hash chain at
~/.quorum/audit_chain.db. Operators and their auditors can verify integrity offline:

  quorum-audit verify-chain     # exit 0 = intact, 2 = broken
  quorum-audit status           # row count + first/last timestamps
  quorum-audit export --since 2026-01-01T00:00:00Z --out /tmp/audit.jsonl

(The `quorum-audit` binary name is preserved for backward compatibility; the underlying capability is a tamper-evident traceability log, not a regulatory audit.) Helpful for evidence collection referenced by EU AI Act Article 14 (human oversight) and SOC2 CC7.2 controls. See
docs/HSP_BLACK_BOX.md for details.

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
| v0.0.1 | 🟢 5 providers, semantic consensus, CLI | 2026-06-15 |
| v0.1.0 | 🟢 13 evolution loops, vector memory, HSP gate, FastAPI server, Stripe billing, EU AI Act cert | 2026-06-16 |
| v0.1.5 | 🟢 BYOK, Firestore persistence, free signup, Hermes 3 (Nous), `/v1/consensus` provider filter | 2026-06-18 |
| v1.0.0 | 🟡 Hosted SaaS public launch, multi-tenant, federated loop GA | Q4 2026 |

## License

Apache 2.0 (core) + HSP Commercial Restrictions on evolution / compliance modules. See [LICENSE-HSP](../LICENSE-HSP).

## Founder

Jaqueline Martins — Sovereign Chain Ltd, UK. Patent holder of [HSP Protocol (PCT/US26/11908)](https://github.com/jaquelinejaque/hsp-protocol).
