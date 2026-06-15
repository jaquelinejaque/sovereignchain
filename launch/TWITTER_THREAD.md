1/8
GPT, Claude, and Gemini disagree on ~30% of non-trivial questions.

Nobody ships the layer that tells you which one to trust.

So I did.

Quorum v0.1.0 is live.

2/8
Query N frontier LLMs in parallel. Get a confidence-scored answer with disagreement surfaced.

When models split, the question was hard.
When they converge, you have signal — not a vibe.

Single-model answers are bets. Consensus answers are measurements.

3/8
Why hasn't OpenAI shipped this?

They won't route to Claude. Anthropic won't route to Gemini. Google won't route to GPT.

A neutral consensus layer has to be independent. Switzerland of LLMs.

Incumbents structurally can't sell it. That's the whole moat.

4/8
13 self-evolution loops run in the background:

- track per-model accuracy by domain
- detect drift
- route around degraded providers
- learn who to trust on what

Routing gets better while you sleep. You don't tune prompts. The system tunes itself.

5/8
If your prod stack still trusts one LLM for high-stakes calls in 2026, you're flying blind and calling it "AI strategy."

Reply with your eval suite. I'll wait.

6/8
Free tier: 100 queries/mo, no card.

Pro: £49/mo, unlimited, priority routing, full audit log of every model vote.

Self-host: fork it tonight, point it at your own keys, no telemetry phones home.

7/8
Apache 2.0 on the engine.

HSP consensus protocol filed under PCT/US26/11908 — patent protects the protocol from capture, not the user.

Built solo in the UK. No VC. No moat games. No waitlist theater.

Code that ships.

8/8
🚀 Quorum v0.1.0 is live.

Star it. Fork it. Break it. Tell me what's broken.

https://github.com/jaquelinejaque/sovereignchain
