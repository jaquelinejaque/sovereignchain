# Self-Audit Findings v0.1.2 — by Quorum on Quorum

> We used the Quorum CLI to audit our own 14k-line codebase. This is the honest scorecard.

Generated on a clean checkout of `main` at the v0.1.2 cut. The audit fanned out
across providers, billing, the consensus core, and the 13 self-evolution loops.
Where the audit produced a patch we could apply within the v0.1.2 window, the
fix is described inline. Where it could not, the gap is logged in
*Open blockers for v1.0.0* below — no hand-waving.

This document is intentionally uncomfortable. If we are wrong about the wedge
("a multi-vendor consensus engine the incumbents structurally cannot ship"),
the reason will live in one of these gaps long before it shows up in a
customer ticket.

---

## Critical findings

No findings at severity **critical** were produced by this round. The closest
candidates (Gemini key in URL query string, missing webhook signature
delegation, unbounded prompt fan-out) were classified **high** because each
required attacker control over a specific surface — a TLS-terminating proxy,
a forged Stripe payload, an oversized prompt — rather than yielding silent
remote compromise on a default deployment.

If you find something that should be critical and isn't, open an issue at
`https://github.com/sovereignchain/quorum/issues` and reference this file by
its SHA at HEAD. We will re-classify in public.

---

## High findings

### H-1 — Gemini API key transported as URL query parameter

- **Module:** `providers/gemini`
- **File:** `quorum/src/quorum/providers/gemini.py` (lines 32–35 pre-fix)
- **Issue:** the key was appended to the request URL as `?key={api_key}`.
  URLs surface in reverse-proxy access logs, corporate TLS-terminating
  middleboxes, OS-level network traces, and any error message that echoes
  the URL — including `httpx.HTTPError.request.url`. No `try/except` wrapped
  the post call, so an upstream failure would propagate the URL (and the
  key) to whatever logger caught the exception.
- **Fix in v0.1.2:** key now sent via `x-goog-api-key` header (the documented
  safe transport), URL stripped of the `?key=` query, and the entire
  `client.post` wrapped in `try/except httpx.HTTPError` returning
  `error="network_error"`.
- **Remaining:** none for this finding. A broader pass to add the same
  defensive wrap to every provider's HTTP path is tracked in *Open blockers*.

### H-2 — Anthropic provider echoed unsanitised upstream body into logs

- **Module:** `providers/anthropic`
- **File:** `quorum/src/quorum/providers/anthropic.py`
- **Issue:** on non-200 responses the error string included `r.text[:120]`,
  which can carry CR/LF and partial multibyte sequences. The first enables
  log-injection (a crafted prompt echoed back by the upstream could synthesize
  fake log lines); the second can corrupt downstream log parsers that expect
  valid UTF-8.
- **Fix in v0.1.2:** decode `r.content[:200]` with `errors="replace"` so no
  mid-codepoint splits leak, then strip `\n` and `\r` before interpolating
  into the `error` field. Length budget loosened from 120→200 chars because
  the replacement codepoint costs more bytes than ASCII.
- **Remaining:** apply the same sanitiser to `openai.py` and `gemini.py`
  error paths. Tracked in *Open blockers*.

### H-3 — OpenAI provider lacked structured error handling

- **Module:** `providers/openai`
- **File:** `quorum/src/quorum/providers/openai.py`
- **Issue:** the body of `complete()` assumed `r.json()` would succeed and
  `data["choices"][0]["message"]["content"]` would index. A malformed
  upstream payload (or a 200 that returned an error envelope) would raise
  `KeyError`/`IndexError`/`ValueError`/`json.JSONDecodeError` straight out of
  the provider — instead of returning a structured `ModelResponse` with an
  `error` field, which is what every caller assumes.
- **Fix in v0.1.2:** outer `try/except Exception` returns
  `error="internal_error"`; inner narrow except around `r.json()` + indexing
  returns `error="parse_error"`. Provider can no longer throw past its own
  boundary.
- **Remaining:** same pattern needs to be applied to `gemini.py` and
  `replicate.py` (Ollama already returns structured errors).

### H-4 — Unbounded prompt/response amplification in consensus core

- **Module:** `core/consensus`
- **File:** `quorum/src/quorum/core/consensus.py`
- **Issue:** `consensus(prompt, ...)` had no size cap. A single oversized
  prompt fanned out into N provider bills + 2N embedding calls + 1 permanent
  vector-memory write per call. An attacker on a Pro tier could burn a month's
  budget in one request, and a runaway client could bloat the per-user
  `VectorMemory` table indefinitely.
- **Fix in v0.1.2:** introduced `MAX_PROMPT_BYTES = 32_000` and
  `MAX_RESPONSE_BYTES = 16_000`. `consensus()` raises `ValueError` for
  oversize prompts at the entry point; provider responses are truncated
  before embedding *and* before memory ingest. Documented inline in the
  function docstring under *Size limits (anti-abuse)*.
- **Remaining:** the caps are global; per-tier caps (Pro vs Team vs
  Enterprise) would let us sell larger windows on higher tiers. Tracked in
  *Open blockers*.

### H-5 — Stripe webhook used hand-rolled HMAC verification

- **Module:** `billing/stripe_billing`
- **File:** `quorum/src/quorum/billing/stripe_billing.py`
- **Issue:** `_parse_and_verify` implemented Stripe's v1 signing scheme by
  hand (`hmac.compare_digest`, manual 5-minute replay window). The algorithm
  was correct, but rolling your own crypto verification against a moving
  third-party spec is exactly the kind of code that silently breaks the day
  Stripe ships a v2 scheme — and we would not notice until a real webhook
  was rejected.
- **Fix in v0.1.2:** when the `stripe` SDK is importable, delegate to
  `stripe.Webhook.construct_event`; only fall back to the hand-rolled v1
  path in the dev/test branch where the SDK is absent.
  `SignatureVerificationError` and `ValueError` from the SDK are wrapped
  into our stable `BillingError` so downstream callers don't need to know
  which path verified the request.
- **Remaining:** none. The `stripe` SDK is now a hard runtime dep for
  hosted; self-host without it is supported but explicitly second-class.

---

## Medium / Low findings

| ID | Sev | Module | File | Finding | Status |
|----|-----|--------|------|---------|--------|
| M-1 | medium | billing | `stripe_billing.py` | `_infer_tier_from_price` used magic literals `2_900`/`19_900` instead of referencing `TIERS[*].amount_pence` — silently drifts if Pro is repriced | **Fixed v0.1.2** — thresholds re-anchored to `TIERS["pro"].amount_pence` and `TIERS["team"].amount_pence` |
| M-2 | medium | billing | `stripe_billing.py` | `create_subscription` hard-coded `enterprise` as the only contact-sales tier — new contact-sales tiers (Team, Compliance) would silently get a self-serve checkout flow they aren't priced for | **Fixed v0.1.2** — added `contact_sales: bool` on `TierConfig`; `create_subscription` routes any `contact_sales=True` tier to the `contact-sales` sentinel |
| M-3 | medium | core | `consensus.py` | Public size caps were not documented in the function docstring, so SDK users couldn't tell whether a 50 kB prompt would be rejected, silently truncated, or quietly billed in full | **Fixed v0.1.2** — `Size limits (anti-abuse)` section added to docstring; `ValueError` is the documented contract |
| M-4 | medium | providers | `replicate.py` | Same JSON-parse fragility as OpenAI pre-H-3 — assumes upstream envelope shape | **Open** — replicate is currently optional/experimental, deferred to v0.1.3 |
| M-5 | low | providers | `ollama.py` | Local Ollama endpoint defaults to `http://localhost:11434` with no TLS or auth — fine for the documented use case (developer laptop) but worth a docs callout for users who proxy it onto a LAN | **Open** — docs-only fix planned for v0.1.3 README |
| L-1 | low | billing | `stripe_billing.py` | `__all__` did not previously export `DEFAULT_TIER`, `get_default_tier`, `list_tiers` despite those being part of the new public API | **Fixed v0.1.2** — exported |
| L-2 | low | landing | `landing/index.html` | Two pre-existing inline-style attributes on the patent notice and Apache link `<p>` tags flagged by IDE diagnostics | **Won't fix** — pre-existing pattern shared with the hero subhead; refactoring violates "minimum mudança necessária" |

---

## Evolution loop classification

The README originally implied all 13 self-evolution loops were live in v0.1.0.
They are not. Below is the honest scorecard at v0.1.2. *Functional* means it
runs end-to-end on real consensus traffic in production. *Partial* means the
code path is wired in but the algorithm is a stub, the storage layer is
ephemeral, or it only fires under a feature flag. *Skeleton* means the file
exists, the public API is sketched, and there is little or no behavior behind
it yet.

### Functional (3)

- **Loop 1 — Memory (`evolution/memory_loop.py`, 585 LOC).** Per-user
  `VectorMemory` ingest fires on every `consensus()` call. Truncation caps
  from H-4 land here too. Used in production today; gates the personalization
  surface for Pro tier.
- **Loop 4 — Router / MoE (`evolution/router.py`, 894 LOC).** Selects the
  per-query provider panel from a budget cap. v0.1.2 adds a module-level
  `get_class_boosts` wrapper in `hebbian.py` so the router can read pairwise
  alignment without owning the matrix singleton. Functional end-to-end;
  upgrade path to per-class shards is documented.
- **Loop 6 — RLHF feedback (`evolution/rlhf.py`, 1021 LOC).** Records
  thumbs-up/down on consensus answers, persists to sqlite, surfaces weights
  back into the router. v0.1.2 adds the `record_feedback` convenience wrapper
  the FastAPI handler was already calling. Functional; the reward model
  itself is a logistic baseline, not yet a learned scorer.

### Partial (2)

- **Loop 7 — Hebbian co-activation (`evolution/hebbian.py`, 611 LOC).** The
  pairwise alignment matrix is real, the decay job runs, the sqlite store is
  durable. Missing: per-query-class shards (currently global) and a
  documented bootstrap for new model pairs.
- **Loop 10 — Meta-learner (`evolution/meta.py`, 545 LOC).** Logs every
  consensus round into a structured event store the meta-learner can train
  on. Missing: the actual training loop. The events are useful as a dataset
  even without it.

### Skeleton (9)

- **Loop 2 — A/B testing (`evolution/ab_testing.py`).** Public API sketched,
  no live experiments.
- **Loop 3 — Adversarial probing (`evolution/adversarial.py`).** Red-team
  prompt harness exists but is not wired into CI.
- **Loop 5 — Architecture search (`evolution/architecture_search.py`).**
  Search-space defined, no controller.
- **Loop 8 — Competition (`evolution/competition.py`).** Tournament bracket
  logic, no scheduler.
- **Loop 9 — Distillation (`evolution/distillation.py`).** Teacher/student
  interfaces, no training pipeline.
- **Loop 11 — Federated (`evolution/federated.py`).** Round protocol
  sketched, no aggregator deployed.
- **Loop 12 — Self-prompt (`evolution/self_prompt.py`).** Prompt-mutation
  primitives, no optimization loop.
- **Loop 13 — Synthetic data (`evolution/synthetic_data.py`).** Generator
  interfaces, no curation pipeline.
- **Loop 0 — Bootstrap / Genesis.** Counted in the original "13 loops"
  marketing claim but never had its own module; the bootstrap behavior lives
  in `core/consensus.py` and `providers/registry.py`.

**Honest summary:** of the originally-advertised 13 loops, **3 are
functional**, **2 are partial**, and **9 (counting the never-modularized
bootstrap) are skeleton**. The Show HN post for v0.1.1 already disclosed
this in the body; this doc is the artifact that disclosure pointed at.

---

## Open blockers for v1.0.0

The minimum work required before we drop the "self-evolution" framing
from the README and call this v1.0:

1. **Get loops 2, 3, 5, 8, 9, 10, 11, 12, 13 to *partial* or better.**
   Specifically: the meta-learner (Loop 10) is the highest-leverage next
   step because every other skeleton loop produces training data for it.
2. **Per-tier size caps.** `MAX_PROMPT_BYTES` should scale with tier
   (Pro 32k, Team 128k, Enterprise unbounded with explicit budget gate).
3. **Apply the H-2 unicode/CRLF sanitiser to `openai.py` and `gemini.py`**
   error paths. H-3's structured-error wrap belongs on `replicate.py` and
   any future provider.
4. **`replicate.py` JSON parsing hardening** (M-4).
5. **Per-class shards for the Hebbian matrix** so router boosts can be
   conditioned on `query_class` instead of averaged globally.
6. **Reward model upgrade for RLHF** — replace the logistic baseline with
   a learned scorer once we have ≥10k feedback events.
7. **Quorum-doctor coverage of the audit findings file itself** — the
   doctor command should fail-loud if `docs/AUDIT_FINDINGS_v*.md` is
   older than the package version it ships under.
8. **A regression test that asserts `__version__` matches the package
   metadata** — the v0.1.0 deploy shipped with a stale `0.0.1` string that
   only surfaced when a customer hit `/v1/healthz`. We never want that
   class of bug again.
9. **Quorum-on-Quorum CI job** — run this same audit against every PR
   and fail the build if a critical or high finding is introduced.
10. **EU AI Act conformity refresh** — the Compliance/Sovereign tier
    paperwork was drafted against the August 2026 high-risk deadline; it
    should be re-verified against the final regulatory text once the
    delegated acts land.

---

Generated 2026-06-16. Quorum eats its own dog food.
