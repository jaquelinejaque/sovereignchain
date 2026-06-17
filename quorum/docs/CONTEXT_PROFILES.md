# Context Profiles — recommended usage pattern

> **TL;DR**: Before asking Quorum anything substantial, create a context profile for the project or domain you're working in. The profile gets injected into every query automatically, so all 14+ LLMs in the consensus pool start from the same ground truth instead of falling back to their training-data priors.

---

## Why this exists

On 2026-06-17 we ran Quorum on itself to evaluate a B2B iOS app for hair professionals. 14 of the 16 LLMs in the pool generated copy that drifted into **B2C consumer** language and one even hallucinated a non-existent "community forum" feature. The reason was simple: the per-query prompt was the only context, and consumer beauty apps are dramatically more common in LLM training data than B2B professional tools. The models collectively defaulted to the wrong prior.

A persistent context profile — same idea as Claude Projects, ChatGPT Custom Instructions, Cursor `.cursorrules`, Aider conventions — fixes that failure mode for multi-LLM consensus too.

## The 3-step pattern

### 1. Create a profile once

```bash
# from a markdown file (recommended for rich context)
quorum context add keratin-app --file /Users/me/keratin-app/APP_STORE_METADATA.md

# from inline text
quorum context add quorum-product --text "Open-source multi-LLM consensus engine. Apache 2.0 + LICENSE-HSP. Brand = honest disclosure. Target = solo devs + small teams who already pay for Claude+GPT+Gemini and want a second opinion before high-stakes decisions."

# or via stdin
cat README.md | quorum context add my-project
```

By default, the new profile is made **active** immediately.

### 2. Ask away — context auto-injects + web is live

```bash
quorum ask --all "What is the single highest-leverage next move?"
```

You'll see a `📎 context profile: 'keratin-app' (4321 chars injected)` line in the output confirming the injection. Every model in the pool now sees the project context BEFORE the question, and treats it as authoritative ground truth.

Live web context is also ON by default since v0.1.x (`--no-web` to disable).

### 3. Switch profiles as you switch hats

```bash
quorum context list                    # see all profiles
quorum context current                 # which is active right now
quorum context use quorum-product      # switch to a different one
quorum context clear                   # work without any profile
```

## Profile content — what works well

A good context profile is **3 things**:

1. **Identity & positioning** — who is the product/company/project for, what does it do, what is its tone of voice
2. **Concrete facts** — features, pricing, tech stack, brand assets, naming conventions, dates that matter
3. **Negative space** — what the project is NOT, what to never claim, common mistakes to avoid

### Sample profile structure

```markdown
# Keratin Pro Mastery (iOS app)

## What this is
B2B iOS app for hair professionals — stylists, salons, beauty experts.
Educational course (11 modules, 35 lessons, 60+ quizzes) + AI diagnostic tools.

## What this is NOT
- NOT a consumer beauty app
- NOT a "treat yourself at home" product
- NOT a chemistry textbook (we're operational, not academic)

## Target user
Working stylist who wants to add keratin services, or salon owner training a team.

## Pricing model
- £24.99 one-time IAP unlocks full course (lifetime access, no learn-paywall)
- Optional monthly AI tools subscription via StoreKit (GOD IA Vision, Style Simulator)
- ALL payments through Apple In-App Purchase — never Stripe inside the app
- Physical product orders go through keratintreatment.co.uk (outside app, Stripe OK there)

## Real features (do not invent others)
- 11 modules of professional content
- 60+ quizzes with instant feedback
- AI: GOD IA Vision (porosity diagnostic), Style Simulator (post-treatment preview), KM Assistant, AI Diagnostic Scan
- Treatment Timer with thermal zone indicators
- ROI Calculator (12-month projection)
- Certificate of completion (shareable)
- Brand-by-brand guidance: Brazilian Blowout, GKhair, Cezanne, Lasio, S-Aqua
- Offline support, Apple Sign-In, English UI

## Brand voice
Luxe technical. Pro-to-pro. No consumer fluff. UK English.

## What I am working on right now
Apple App Store submission. Need to maximize approval probability.
```

## Failure modes this prevents (verified)

- ✅ Wrong target audience drift (B2C / B2B)
- ✅ Hallucinated features that don't exist
- ✅ Wrong pricing assumptions
- ✅ Brand voice inconsistency between models
- ✅ Outdated training-data priors on company stack/tooling

## Failure modes this does NOT prevent

- ❌ Models making up facts about *third-party* things not in the profile (e.g. competitor pricing) — use `--web` for that
- ❌ Models drifting if the profile itself is wrong (garbage in → garbage out)
- ❌ True deep specialization — for life-safety or legal-binding answers, no consensus replaces a human expert

## Storage & privacy

Profiles live at `~/.quorum/contexts/<name>.md` as plain Markdown — human-editable, no DB, no telemetry. The active profile name is in `~/.quorum/active_context`. To wipe everything: `rm -rf ~/.quorum/contexts`.

For the hosted API (api.quorum-ai.dev): a server-side equivalent will land in v0.2 — profiles tied to your API key, encrypted at rest. Until then, the CLI-only pattern is the recommended way.

## Why this matters for the brand

Quorum's brand promise is **honest disclosure**. Without context, models default to mass priors and hallucinate. With context, the multi-LLM consensus is grounded in **your** ground truth, not the average of the internet. The audit trail (`docs/SELF_EVAL_*.json` outputs include the injected context body) lets anyone reading later verify what the models were given before they answered.

That's the difference between "ask 14 LLMs and pray" and "ask 14 LLMs about *your specific context* and let them disagree productively". The second is a product. The first is a toy.
