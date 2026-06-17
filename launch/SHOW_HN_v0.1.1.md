> **DEPRECATED (2026-06-17).** This is the v0.1.1 draft. Use
> [`SHOW_HN_FINAL.md`](./SHOW_HN_FINAL.md) for the current copy — it has
> the right Cloud Run region (europe-west1, not -west2), the corrected
> version string, and the correct API host (`api.quorum-ai.dev`, not the
> root). Kept on disk for audit/history only.

Show HN: Quorum – multi-LLM consensus engine, Pro £49/mo for solo devs

Two days ago I ran Quorum against its own source. The thing I built to catch what unit tests miss caught two things my unit tests missed: a deprecated Gemini embedding endpoint that was about to start 404-ing, and a stale `__version__` string still pinned to 0.0.1 while the package shipped as 0.1.0. Both fixed in under thirty seconds once the divergence report surfaced them. That run is the reason this post exists.

Quorum fans a prompt out to multiple LLM backends (Claude, Gemini, GPT, local Llama via Ollama, optionally Grok) and returns a consensus score plus the divergence points — the specific places where the models disagree. Divergence is the signal. Agreement is the cheap part.

Live, hosted, no install required:

    https://quorum-ai.dev

DNS is still propagating in some regions; if that URL is cold for you, the Cloud Run origin is reachable directly at:

    https://quorum-api-86770458722.europe-west2.run.app

A `GET /v1/healthz` against the live origin right now returns:

    {"status":"ok","version":"0.0.1","time":"2026-06-16T00:44:11.430200Z"}

(Yes — the version string is one of the two bugs the self-audit caught. The fix is in v0.1.1, redeploy is queued. Leaving the stale response in this post on purpose so you can see what dogfooding actually looks like.)

Honest scope, because I would rather lose your trust now than later: the README originally implied all 13 evolution loops were live. They are not. Thirteen loops are scaffolded in v0.1.x; roughly three of them — memory, router, and an RLHF feedback loop — have functional baselines in production today. The other ten are roadmap for v1.0. The README has been corrected to reflect this. I self-audited the gap using Quorum itself; findings are in `docs/AUDIT_FINDINGS.md`.

The moat, if there is one, is structural rather than technical. Anthropic, OpenAI, and Google cannot ship a multi-vendor consensus engine without commoditizing their own answer as one vote among several. A solo team can. That is the entire wedge.

Pricing, deliberately small:

- Free: 100 queries/mo, sandbox only, no BYOK.
- Pro: £49/mo. 5,000 queries, bring your own keys for any backend, all currently functional loops included. Aimed at solo devs and small teams who already pay for Claude, Gemini, and GPT and want a second opinion on the important calls.
- Team £199/mo and Enterprise £1,499/mo exist but are on-request — talk to me first.

License: Apache 2.0 for the engine. The HSP transport layer (the consensus-scoring protocol itself) is under PCT/US26/11908 with a commercial-use carve-out documented in `LICENSE-HSP`. Self-host, fork, audit, ship internal tools — all permitted. Resell the consensus protocol as a service — talk to me.

ASK HN: solo devs — what is the smallest unit of consensus that is actually useful to you? A single query at the moment a decision matters? A daily digest of disagreements across the prompts your agents already ran? An IDE inline check that fires on save? I have opinions but they are mine, and I would rather hear yours before I pick.
