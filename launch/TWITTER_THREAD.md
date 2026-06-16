# Twitter/X thread — Quorum launch (8 tweets)

Post as a reply chain. Attach `~/Desktop/quorum-23-models-live.png` to tweet 1.

---

**1/8** [attach screenshot]

I shipped an open-source multi-LLM consensus engine yesterday.

Then I asked it about its own source code.

In 30 seconds it found 2 bugs my unit tests had missed.

23 models in parallel. $0.011 a query. Live at https://quorum-ai.dev

🧵👇

---

**2/8**

The two bugs it caught:

- Google had silently deprecated the Gemini embedding endpoint (about to 404 in prod)
- My __version__ string was stale

Fixed both in 30s because the divergence report named them.

Most teams ship with one LLM and pretend it's right.

---

**3/8**

What's in the consensus pool today, ALL responding live:

• Anthropic — Claude Sonnet 4.6, Opus 4.8, Haiku 4.5
• OpenAI — GPT-4.1, GPT-4o-mini
• xAI — Grok-4
• Google — Gemini Flash
• 6 NVIDIA-hosted OSS (Llama 3.3, Llama-4 Maverick, DeepSeek V4, Dracarys...)

---

**4/8**

• Mistral — Large, Codestral, Small
• Cohere — Command R+, R, A
• DeepSeek-direct — Chat, Reasoner
• Local Llama via Ollama

23 of 25 models OK. Cost per consensus call ~$0.011. 87% semantic agreement on "should I use SQLite or Postgres for 100 paying users" → Postgres won.

---

**5/8**

How: cosine similarity on embeddings (NOT Jaccard — paraphrases count). Top-weighted answer + audit trail of every model's response + disagreements explicitly listed.

GitHub: https://github.com/jaquelinejaque/sovereignchain

Apache 2.0 + HSP patent PCT/US26/11908.

---

**6/8**

Honest scorecard: 10 of 13 self-evolution loops are functional today (memory, MoE router, RLHF, A/B testing, synthetic data, Hebbian, meta-learner, ELO competition, self-prompting, adversarial probing).

3 still scaffold (down from 9 at launch — Quorum is building itself).

Published in repo. Updated every release. I'd rather lose your trust now than after 100 users.

---

**7/8**

The wedge is structural:

Anthropic, OpenAI, Google CANNOT ship multi-vendor consensus without commoditizing their own answer to one vote among several.

A solo team can. That's the bet.

VS Code extension live: https://marketplace.visualstudio.com/items?itemName=sovereignchain.quorum-vscode

---

**8/8**

Pro tier: £49/mo. 5,000 queries, BYOK any backend.

Free OSS forever, self-host unlimited.

If you ship AI features and want a second (or eighth) opinion before something hits prod, DM me — looking for design partners.

Roast me. I'd rather hear it now.

#AI #OpenSource #LLM
