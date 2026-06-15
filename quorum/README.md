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

## Why we exist

Anthropic, OpenAI, Google, Microsoft **structurally cannot** sell a multi-vendor consensus engine — it would commoditize their own models. Quorum occupies that gap. Neutral. Auditable. Open source. Patent-protected.

This is the "Switzerland of LLMs."

## Roadmap

| Version | Status | Date |
|---|---|---|
| v0.0.1 | 🟢 5 providers, semantic consensus, CLI | 2026-06-16 |
| v0.1.0 | 🟡 13 evolution loops, vector memory, HSP gate | Q3 2026 |
| v1.0.0 | 🔴 Hosted SaaS, EU AI Act cert, multi-tenant | Q4 2026 |

## License

Apache 2.0 (core) + HSP Commercial Restrictions on evolution / compliance modules. See [LICENSE-HSP](../LICENSE-HSP).

## Founder

Jaqueline Martins — Sovereign Chain Ltd, UK. Patent holder of [HSP Protocol (PCT/US26/11908)](https://github.com/jaquelinejaque/hsp-protocol).
