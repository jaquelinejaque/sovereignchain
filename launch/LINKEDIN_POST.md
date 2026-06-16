I shipped an open-source multi-LLM consensus engine yesterday. Then I used it on its own source code.

In 30 seconds it found 2 bugs my unit tests had missed: a Gemini embedding endpoint that Google had silently deprecated (about to start 404-ing in production) and a stale `__version__` string.

I kept going. The next self-audit found 5 more high-severity issues, including a Stripe webhook with hand-rolled HMAC verification I had forgotten about. All patched and shipped in the same release.

That's the entire reason Quorum exists: when models disagree about your code, your contract, your dosage calculation — that disagreement is the signal worth knowing. Most teams ship with one LLM and pretend it's right.

What's live today at https://quorum-ai.dev (Apache 2.0):
↳ 8 LLMs in parallel (Claude, GPT, Gemini, Llama 3.3 70B, DeepSeek V4, Llama-4 Maverick, Dracarys 70B, local Ollama)
↳ Semantic agreement via cosine on embeddings — paraphrases count, exact-match doesn't matter
↳ $0.000001 per consensus query on the NVIDIA AI Foundation free tier
↳ EU AI Act Article 12/13 evidence helper: per-query SHA-256 hash-chained PDF audit log
↳ HSP Gate (patent-pending PCT/US26/11908) for high-stakes async approval — optional, off by default
↳ VS Code extension live on Marketplace (`sovereignchain.quorum-vscode`)

Honest scope: 8 of 13 evolution loops are functional today (memory, MoE router, RLHF, A/B, synthetic data, Hebbian co-activation, meta-learner, model-vs-model ELO). 5 are scaffold. The README and release notes say so. I update the scorecard every release.

Bonus dogfood: the 3 loops promoted from skeleton/partial in v0.1.4 had their design decisions made by Quorum itself — the consensus across 23 LLMs picked the Hebbian decay rate, the meta-learner update policy, and the ELO K-factor. Self-evolution isn't a roadmap claim. It's a commit history.

The wedge is structural: Anthropic, OpenAI, and Google cannot ship multi-vendor consensus without commoditizing their own answer to one vote. A solo team can. That's the bet.

Pro tier is £49/mo. Free OSS forever. Looking for design partners — DM me if you ship AI features and want a second (or eighth) opinion before something hits prod.

#AIAct #MachineLearning #OpenSource #LLM #SovereignAI
