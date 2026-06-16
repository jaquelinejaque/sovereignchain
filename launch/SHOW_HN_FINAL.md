Show HN: Quorum — 8 LLMs in parallel, semantic consensus, $0.000001/query

URL: https://github.com/jaquelinejaque/sovereignchain

----

I got tired of asking the same question to Claude, GPT, and Gemini one at a time and squinting at the differences. Quorum fans your prompt out to multiple LLMs in parallel, scores semantic agreement (cosine on embeddings, not Jaccard), and surfaces the divergence points — the specific places where the models disagree. Divergence is the signal.

Live, hosted, BYOK:

    https://quorum-ai.dev             Landing Page
    https://api.quorum-ai.dev/docs    Swagger UI
    https://api.quorum-ai.dev/v1/healthz  -> {"status":"ok","version":"0.1.3"}

Default pool today: Gemini, Claude, GPT, local Llama via Ollama, plus Llama 3.3 70B / 3.2 3B / 3.1 8B / Llama-4 Maverick / DeepSeek V4 Flash / Dracarys 70B — last six all free via NVIDIA AI Foundation. Mistral, Qwen, Phi-4 via Replicate if you set the key.

A real consensus run right now: 8 models replied, $0.000001 total cost, 11.7 s wall time (the 70B cold-start is the long pole). All eight agreed on "4" for 2+2.

How I built it: I used Quorum on its own source. First dogfood query caught two bugs unit tests had not — a deprecated Gemini embedding endpoint and a stale __version__ string. Both fixed in 30 s. The v0.1.2 self-audit (docs/AUDIT_FINDINGS_v0.1.2.md) caught five more high-severity issues including a Stripe webhook with hand-rolled HMAC verification I didn't realize I'd written. Patched and shipped in the same release.

Honest scope. README originally implied all 13 self-evolution loops were live. They are not. Honest count: 3 functional (memory, router, RLHF), 2 partial (Hebbian, meta-learner), 9 still scaffold. README corrected in v0.1.2.

The wedge is structural: Anthropic, OpenAI, Google cannot ship a multi-vendor consensus engine without commoditizing their own answer to one vote. A solo team can.

Pricing: Free 100 q/mo. Pro £49/mo, 5k queries, BYOK. Team/Enterprise on request.

VS Code extension v0.1.0 live on Marketplace: https://marketplace.visualstudio.com/items?itemName=sovereignchain.quorum-vscode — install + right-click selection -> "Quorum: Review code (bugs/security)". API key in SecretStorage, not settings.json.

License: Apache 2.0 for the engine. HSP consensus-scoring protocol under PCT/US26/11908 with commercial-use carve-out in LICENSE-HSP.

ASK HN: solo devs — what is the smallest unit of consensus actually useful to you? A query at the moment a decision matters? A daily digest of disagreements across what your agents already ran? An IDE inline check on save? I have opinions but they are mine — would rather hear yours before picking the next thing to build.
