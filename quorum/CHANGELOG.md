# Changelog

All notable changes to Quorum are documented here. Format loosely follows Keep-a-Changelog; versioning follows SemVer.

> **Legal terminology notice (added 2026-06-23, applies retroactively to every entry below).**
> Earlier entries in this changelog describe HSP features as an "audit chain", "compliance primitive", "compliance audit", "certificate" or "auto-certification". From version 0.2.4 onwards the canonical terminology is **tamper-evident traceability log**, **readiness toolkit**, **advisory evidence record**, and **EU AI Act Article 14 readiness assessment**.
>
> The renaming is **strictly vocabulary** — no feature changes — and corrects a misrepresentation risk under Fraud Act 2006 s.2 and the Consumer Protection from Unfair Trading Regulations 2008 (UK). Quorum and Sovereign Chain Ltd are NOT a Notified Body under Article 31 of Regulation (EU) 2024/1689; the toolkit is advisory only. Final conformity assessment remains the responsibility of the AI system provider (internal, Annex VI) or a designated Notified Body (external, Annex VII).
>
> Function names, CLI binaries (`quorum-audit`) and SQLite column names that contain "audit" / "cert" are preserved for backward compatibility; they are documented in the relevant module docstrings.

## 0.2.4 — Legal terminology clean-up (2026-06-23)

### Vocabulary clean-up (no behavioural change)
- README, landing pages, VS Code extension, `hsp/black_box.py` and `hsp/ai_act_cert.py` docstrings updated to remove every client-facing use of "audit", "certificate", "certification", "compliance audit", "audit-grade compliance", and "auto-certification".
- PDF evidence record (formerly "EU AI Act Compliance Certificate") now titled **"EU AI Act Readiness — Evidence Record"** with a mandatory header banner and footer disclaimer citing Article 31 and Annex VI/VII.
- New legal disclaimer footer on quorum-ai.dev landing page.
- New `quorum.proactive` module exposing analyst pipeline (signal ingest / enrich / analyse / draft / notify).

### Why this matters
- Sovereign Chain Ltd is not a Notified Body. Selling the Quorum toolkit as a "compliance audit" would expose the company to UK Fraud Act 2006 s.2 (false representation) and CPUTR 2008 (unfair commercial practices).
- The corrected wording lets the product remain valuable (advisory toolkit for EU AI Act Annex VI self-assessment) while keeping the company legally protected.

### Companion artefacts
- New contract template `04-quorum-readiness-services-agreement.md` (with §1.3, §1.4, §3, §9, §10.2 and §15 glossary as the mandatory protection clauses).

## 0.2.3 — HSP Black Box tamper-evident traceability log (2026-06-20)

### Compliance primitive (EU AI Act Article 14 / SOC2 CC7.2)
- New module: hsp/black_box.py — append-only SHA-256 hash chain over consensus calls
- New CLI: quorum-audit (verify-chain, status, export, append) — separate binary, does not touch cli.py
- Auto-hook in consensus.py: every consensus() call appends query_hash + metadata (NOT raw query text — privacy)
- Tampering detection: any altered/deleted/inserted row breaks chain at that point
- Export to JSONL with 0o444 (read-only WORM-lite) for external auditors

### Tests
- tests/hsp/test_black_box.py: 6 adversarial tests (tamper, delete, fail-safe append, export)

### Backward compatible
- Audit append is best-effort try/except — never breaks consensus response

## 0.2.2 — Transparency + behaviour parity (2026-06-20)

### Transparency (CRITICAL)
- ConsensusResult exposes scoring_method ("embedding"|"jaccard"); Jaccard fallback logs at ERROR. Callers can detect degraded scoring.

### Behaviour parity (marketing matches code)
- SelfPromptOptimizer wired into hot path as system_prompt per provider (was dead code). "Self-prompt evolution" now genuinely runs.
- Meta-loop enforcement: loops with priority < 0.1 are SKIPPED for that query (was logged-only). "Loops learn which loops work" now genuinely acts.

### Backwards compatible
- scoring_method defaults "embedding"; existing callers unchanged.
- Provider.complete() new system_prompt param defaults None.

## 0.2.1 — Security patch (2026-06-20)

### Critical
- Killed fail-open in offline license validation
- Added HMAC signature to ~/.quorum/license_cache.json
- Restricted QUORUM_LICENSE_VALIDATE_URL to allowlisted hosts

### High
- QUORUM_HOSTED=1 now requires GCP metadata attestation
- Replaced implicit PYTEST_CURRENT_TEST bypass with explicit "import quorum.testing"

### Server
- /v1/license/validate returns valid=false on lookup exception

All 0.2.0 installs MUST upgrade. Old cache files will regenerate.

## [0.1.0] — 2026-06-16

The "real product" release: 13 self-evolution loops, vector memory, HSP gate, FastAPI server, Stripe billing, and EU AI Act per-query PDF certificate. All modules ship with in-memory fallbacks so tests pass without external keys.

### Added — core

- `src/quorum/core/embeddings.py` — async embedding client (OpenAI / Cohere / local) with cosine + euclidean similarity helpers; replaces the v0.0.1 Jaccard placeholder for semantic consensus
- `src/quorum/core/memory.py` — vector memory store backed by aiosqlite (with pure-Python in-memory fallback); recall-by-prompt for the memory loop

### Added — 13 self-evolution loops (`src/quorum/evolution/`)

- `rlhf.py` — explicit thumbs/correction-driven weight updates
- `hebbian.py` — co-correctness correlation between model pairs
- `distillation.py` — cheap models learn from consensus answers
- `router.py` — per-domain model weighting (code, vision, biomed, …)
- `memory_loop.py` — retrieval-augmented consensus on similar past prompts
- `meta.py` — meta-learning audit of which loops are improving accuracy
- `competition.py` — pairwise model duels, ELO-style ranking
- `ab_testing.py` — two prompt variants per call, win-rate tracking
- `synthetic_data.py` — high-confidence consensus → training data export
- `federated.py` — cross-tenant signal aggregation without raw data leak
- `self_prompt.py` — Quorum rewrites ambiguous prompts before fan-out
- `adversarial.py` — red-team prompts; models that fall for them lose weight
- `architecture_search.py` — trial new model combos / topologies; promote winners

### Added — HSP gate & compliance (`src/quorum/hsp/`)

- `ai_act_cert.py` — per-query PDF certificate (reportlab) satisfying EU AI Act Art. 12 + Art. 13; SHA-256 hash-chain across tenant; required for the 2026-08-02 enforcement window
- `webhook.py` — signed HSP gate decision callback handler (patent PCT/US26/11908)

### Added — billing (`src/quorum/billing/`)

- `stripe_billing.py` — Stripe-backed metered billing for Free / Pro / Team / Enterprise tiers + Compliance add-on; signature-verified webhook handler; in-memory fallback when `STRIPE_API_KEY` is unset so CI passes without secrets

### Added — server (`src/quorum/server/`)

- `main.py` — FastAPI app with slowapi rate limiting, Bearer/BYOK auth, endpoints: `/v1/consensus`, `/v1/consensus/stream` (SSE), `/v1/models`, `/v1/feedback`, `/v1/cert/{id}`, `/v1/billing/checkout`, `/v1/billing/webhook`, `/v1/hsp/webhook`, `/v1/usage`, `/healthz`, `/metrics`; entry point `quorum-server`

### Changed

- `pyproject.toml` — version bump 0.0.1 → 0.1.0; status Alpha → Beta; new deps: reportlab, stripe, fastapi, uvicorn[standard], slowapi; aiosqlite as optional `storage` extra; mypy added to `dev`; new script `quorum-server`
- `README.md` — full rewrite covering the 13 loops, billing tiers, hosted API endpoints, EU AI Act certification, and an ASCII architecture diagram
- `src/quorum/core/consensus.py` — replaced Jaccard placeholder with semantic-similarity path through `core/embeddings.py`; surfaces `evolution_signals` from the 13 loops

### Notes

- Every external service (Stripe, Supabase, HSP webhook signing) has a graceful in-memory or stub fallback when env vars are missing — `pytest` runs green on a clean machine
- All evolution loops are async write-backs; they never block the request path
- HSP-gated modules carry the dual Apache 2.0 + HSP commercial-restriction header

## [0.0.1] — 2026-06-16

Initial public cut. Minimal but real: five providers, async fan-out, lexical placeholder for consensus scoring, CLI.

### Added

- `README.md`, `LICENSE` (Apache 2.0), `LICENSE-HSP` (PCT/US26/11908 commercial restrictions)
- `quorum/pyproject.toml` — package config, Python 3.10+, BYOK extras
- `quorum/src/quorum/__init__.py` — public `consensus` re-export
- `quorum/src/quorum/cli.py` — typer-based `quorum` CLI
- `quorum/src/quorum/core/consensus.py` — async orchestrator, Jaccard placeholder scoring
- `quorum/src/quorum/providers/base.py` — provider ABC
- `quorum/src/quorum/providers/registry.py` — provider discovery + ordering
- `quorum/src/quorum/providers/anthropic.py` — Claude client (BYOK)
- `quorum/src/quorum/providers/gemini.py` — Gemini client (BYOK)
- `quorum/src/quorum/providers/openai.py` — GPT client (BYOK)
- `quorum/src/quorum/providers/ollama.py` — local Ollama client
- `quorum/src/quorum/providers/replicate.py` — Replicate-hosted Llama/Mistral
- `quorum/src/quorum/hsp/__init__.py`, `hsp/gate.py` — decorator stub for HSP-gated calls
- `quorum/tests/test_consensus.py` — async smoke test, no external keys required
- `quorum/.env.example`, `quorum/.gitignore`

[0.1.0]: https://github.com/jaquelinejaque/sovereignchain/releases/tag/v0.1.0
[0.0.1]: https://github.com/jaquelinejaque/sovereignchain/releases/tag/v0.0.1
