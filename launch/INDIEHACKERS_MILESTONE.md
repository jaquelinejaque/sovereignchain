## Shipped Quorum v0.1.3 — used my own product to find 8 bugs unit tests missed

I built Quorum because I got tired of asking the same question to Claude, GPT, and Gemini one at a time and squinting at the differences. It's open-source, BYOK, and went from idea to v0.1.3 + VS Code Marketplace in under 48 hours.

**Live**:
- Landing: https://quorum-ai.dev
- API + docs: https://api.quorum-ai.dev/docs
- GitHub: https://github.com/jaquelinejaque/sovereignchain
- VS Code extension: https://marketplace.visualstudio.com/items?itemName=sovereignchain.quorum-vscode

### What it actually does, in one example

```python
from quorum import consensus

result = await consensus("Is float comparison with == safe in Python 3.12?")

# result.answer            → top-weighted answer
# result.confidence        → 0.78
# result.disagreements     → ['gpt-5 said "depends on the platform"']
# result.total_cost_usd    → 0.000041
```

It fans your prompt to 8 LLMs in parallel (Claude, GPT, Gemini, Llama 3.3 70B, DeepSeek V4, Llama-4 Maverick, Dracarys 70B, local Ollama), scores semantic agreement via cosine on embeddings, and tells you which models disagreed and on what. **Divergence is the signal.**

### The dogfood moment that sold me on shipping it publicly

I used Quorum on its own source. The first query caught two bugs my unit tests hadn't — Google had silently deprecated the Gemini embedding endpoint (about to start 404-ing), and my `__version__` string was stale. Fixed both in 30 seconds because the divergence report named them.

Then I kept going. The v0.1.2 self-audit (`docs/AUDIT_FINDINGS_v0.1.2.md`, in the repo) caught five more high-severity bugs, including a Stripe webhook with hand-rolled HMAC verification I had forgotten about. Patched and shipped them in the same release.

### Honest scope (the part that gets me banned from Show HN if I overstate)

The README originally implied all 13 self-evolution loops were live. They weren't. Today's honest scorecard, committed in the repo:

- **5 functional**: memory recall, MoE router, RLHF feedback, A/B testing, synthetic data
- **2 partial**: Hebbian co-activation, meta-learner
- **6 skeleton**: distillation, adversarial probing, architecture search, competition, federated, self-prompting

I update this every release. If a loop is scaffold-only, the README says so.

### The structural wedge (why I think this can survive Anthropic/OpenAI)

Anthropic, OpenAI, and Google **cannot** ship a multi-vendor consensus engine without commoditizing their own answer to one vote among several. A solo team can. That's the entire reason this exists.

### Day 1 numbers, no inflation

- **8 of 10 providers** actually returned answers in production today ($0.000001 per consensus query — NVIDIA AI Foundation free tier carries 6 OSS models; the 2 that failed were a Replicate rate-limit and a DeepSeek-paid account out of credit, both my fault, not the engine's)
- **16 Marketplace installs** in 13 hours (most are crawler bots — I checked the User-Agents; maybe 2-3 humans)
- **9 unique IPs** organic traffic before I'd told anyone (Twitter + Facebook preview bots fired, so someone shared a link, still figuring out where)
- **0 paying users** yet. You'd literally be #1.

### Pricing (deliberately small)

- **Free**: Apache 2.0 self-host unlimited; hosted sandbox 100 queries/mo
- **Pro £49/mo**: 5,000 hosted queries, BYOK any backend, all currently-functional loops
- **Team / Enterprise**: on request

### Ask

Two real questions because I'd rather hear yours than build the wrong thing:

1. **For solo devs**: what's the smallest unit of consensus that's actually useful to you? A single query when a decision matters? A daily digest of disagreements across what your agents already ran? An IDE inline check that fires on save?

2. **For founders**: have you ever shipped a feature based on what one LLM said, then regretted it because a different LLM would have flagged the bug? I have. That's the wedge.

Open source: Apache 2.0 for the engine. The HSP consensus-scoring protocol is patent-pending (PCT/US26/11908) with a commercial-use carve-out in `LICENSE-HSP` — self-host + fork + audit + ship internal tools all permitted; reselling the protocol as a service needs a chat with me first.

Made in UK by Sovereign Chain Ltd. Roast me in the comments — I'd rather hear it now than after 100 users.
