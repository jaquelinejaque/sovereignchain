# Quorum — Multi-LLM Consensus for VS Code (Pro)

Quorum runs a single prompt across **14+ frontier LLMs** — Claude, GPT, Gemini, Llama, DeepSeek, Mistral, Qwen, Cohere, NVIDIA — directly inside VS Code, scores the answers for semantic agreement, and surfaces exactly where the models disagree. Disagreement between strong models is the most valuable signal you can get before shipping code, taking a security call, or making a regulatory commitment.

Backed by the hosted [Quorum](https://quorum-ai.dev) consensus engine with a **tamper-evident HSP traceability log** — every consensus event sealed with a SHA-256 hash chain, useful as advisory evidence material for internal review and EU AI Act readiness documentation.

> **Paid commercial product — £15/mo.** Quorum is source-available under FSL-1.1 and requires a paid Pro license for use in VS Code. No free tier, no trial queries. [Buy Pro](https://buy.stripe.com/aFadR9d6E5rf8JGeINdwc0j).

## Why Quorum

Single-model answers are a single point of failure. When Claude, GPT, and Gemini all return the same answer, you have evidence. When they disagree, you have a flag — and you can decide whether to escalate, re-prompt, or stop. Quorum makes that signal first-class: weighted agreement scoring, per-model breakdown, total cost, and a tamper-evident traceability log for everything you submit.

Built for:

- **Regulated dev teams** (finance, healthcare, legal) — every consensus result is sealed in an HSP traceability log you can present as advisory evidence material during internal review or EU AI Act Annex VI self-assessment preparation.
- **Security engineers** — adversarial review where one model's miss is another model's catch.
- **Researchers and consultants** — defensible answers with a paper trail.

> ⚖️ **Legal notice.** Quorum is an advisory technical toolkit. It is not a conformity assessment under Regulation (EU) 2024/1689 and Sovereign Chain Ltd is not a Notified Body under Article 31. Final conformity assessment remains the responsibility of the AI system provider (internal, Annex VI) or a designated Notified Body (external, Annex VII).

## Get started in 3 steps

1. **Buy a Pro license — £15/mo** via [Stripe Checkout](https://buy.stripe.com/aFadR9d6E5rf8JGeINdwc0j), or run `Quorum: Get Pro License` from the Command Palette after install.
2. **Open Settings** (`Cmd+,` / `Ctrl+,`), search `quorum`, paste your license key into `quorum.apiKey`.
3. **Register your provider keys** so Quorum has models to orchestrate. Bring your own keys — the more you register, the richer the consensus.

   ```bash
   curl -X POST https://api.quorum-ai.dev/v1/customer/keys \
     -H "X-Quorum-API-Key: YOUR_PRO_LICENSE" \
     -H "Content-Type: application/json" \
     -d '{"anthropic":"sk-ant-...","openai":"sk-...","gemini":"..."}'
   ```

   Supported: anthropic, openai, gemini, nvidia, mistral, cohere, grok, dashscope, replicate, deepseek, zhipu, moonshot. Keys are encrypted server-side (Fernet KEK). You pay your providers directly; Quorum charges only for orchestration.

## Commands

| Command | What it does |
|---|---|
| `Quorum: Ask` | Free-form prompt, returns consensus answer in a side panel |
| `Quorum: Ask about selection` | Highlighted code + your follow-up question |
| `Quorum: Explain selected code` | Multi-model explanation of the highlighted code |
| `Quorum: Review selected code (bugs/security)` | Adversarial review across all configured models |
| `Quorum: Compare two implementations` | Pick two snippets, get a structured A/B verdict |
| `Quorum: Get Pro License` | Opens the pricing page |
| `Quorum: Open settings` | Jumps to the Quorum settings page |

The **Explain** and **Review** commands also appear in the editor right-click menu when you have a selection.

## Settings

| Setting | Default | Description |
|---|---|---|
| `quorum.endpoint` | `https://api.quorum-ai.dev` | Quorum API endpoint |
| `quorum.apiKey` | `""` | **Paid Pro license key** from quorum-ai.dev/pricing. Required. |
| `quorum.providers` | `["gemini-flash","llama-3.3-70b","deepseek-v3","claude-sonnet-4-6"]` | Models to query in parallel |
| `quorum.maxLatencyMs` | `30000` | Per-query timeout in milliseconds |
| `quorum.showCostInline` | `true` | Show per-query cost inline in the result panel |

## Sample workflow

1. Highlight a function in any editor (TypeScript, Python, Rust, anything).
2. Right-click → **Quorum: Review selected code (bugs/security)**.
3. Quorum fans the snippet out to all configured models, scores agreement, and renders a panel with the consensus verdict, dissenting opinions, and per-model latency/cost breakdown.

## Pricing

| Plan | Price | What you get |
|---|---|---|
| **Pro** | **£15 / mo** | 10,000 consensus queries/mo, BYOK across all providers, tamper-evident HSP traceability log, all 14+ models |
| **Enterprise** | Contact sales | SSO, dedicated traceability-log retention, custom SLA, on-prem option |

[Buy Pro on Stripe](https://buy.stripe.com/aFadR9d6E5rf8JGeINdwc0j) · [Enterprise inquiry](mailto:facecomercce1@gmail.com?subject=Quorum%20Enterprise%20License%20inquiry).

## License

Source-available under **FSL-1.1 (Functional Source License)** — read the [LICENSE](LICENSE) file. Commercial use in VS Code requires a paid Pro license validated against `api.quorum-ai.dev/v1/license/validate`. Source converts to Apache-2.0 two years after each release.

## Links

- Website: <https://quorum-ai.dev>
- Buy Pro (£15/mo): <https://buy.stripe.com/aFadR9d6E5rf8JGeINdwc0j>
- Repository: <https://github.com/jaquelinejaque/sovereignchain>
- Issues / feedback: <https://github.com/jaquelinejaque/sovereignchain/issues>

Copyright 2026 Sovereign Chain Ltd · UK.
