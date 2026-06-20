# Changelog

All notable changes to Quorum are documented here. Format loosely follows Keep-a-Changelog; versioning follows SemVer.

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
