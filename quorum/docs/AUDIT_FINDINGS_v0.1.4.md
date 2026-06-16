# Self-Audit Findings v0.1.4 — Loop Scorecard Update

> Continuation of v0.1.2 audit. Phase 1 promotion of three evolution loops
> from skeleton/partial to functional, each design-driven by the Quorum
> engine itself (dogfooding).

See `AUDIT_FINDINGS_v0.1.2.md` for the baseline scorecard and original
classification methodology.

---

## Loop scorecard delta

| Status         | v0.1.3 | v0.1.4 |
|----------------|--------|--------|
| **Functional** | 5      | **8**  |
| **Partial**    | 2      | **0**  |
| **Skeleton**   | 6      | **5**  |

Net movement: +3 functional, −2 partial, −1 skeleton.

---

## Loops promoted in v0.1.4

### Loop 2 — Hebbian co-activation (skeleton → functional)

- **Module:** `quorum/src/quorum/evolution/hebbian.py`
- **Tests:** `quorum/tests/evolution/test_hebbian.py` (6 passed in 0.11s)
- **One-line note:** Dense SQLite matrix with EMA rate 0.1 and per-`(model_pair,
  query_class)` keys; reads below `SAMPLE_THRESHOLD` return neutral so cold
  starts do not poison the consensus weight.
- **Quorum design verdict:** dense > sparse (O(n²) lookup is trivial at the
  scale we run); EMA 0.1 catches model-version drift within ~10 e-folding
  observations; per-class isolation is the entire point of the loop.

### Loop 5 (meta-learner) — partial → functional

- **Module:** `quorum/src/quorum/evolution/meta.py`
- **Tests:** `quorum/tests/evolution/test_meta.py` (6 passed in 0.11s)
- **One-line note:** Hybrid online/batch update; `observe()` is O(1) against
  SQLite, `recommend_loops()` drops a loop only when `samples >= 5` AND mean
  `confidence_delta <= 0`, otherwise keeps it for cold-start exploration.
- **Quorum design verdict:** Quorum `--all`, 92% confidence, 5/7 backends
  agreeing (gemini-flash and llama-3.3-70b returned http_429). Hybrid update;
  primary signal = downstream confidence_delta; cold start = enable all loops
  with uniform priors to avoid self-fulfilling-prophecy bias.

### Loop 7 — Model-vs-Model competition / ELO (skeleton → functional)

- **Module:** `quorum/src/quorum/evolution/competition.py`
- **Tests:** `quorum/tests/evolution/test_competition.py`
- **One-line note:** Pairwise ELO per query class; winner = similarity to
  current top-weighted answer (always-available, no extra LLM cost); K-factor
  16 to damp per-query swings; ratings stored per-`(model, query_class)`
  rather than globally averaged.
- **Quorum design verdict:** Quorum `--all`. K=32 would let many correlated
  battles per query swing ratings ~96pts and overshoot; K=16 halves that to
  ~48pts. Per-class is mandatory — global average mushes per-domain skill and
  starves the router.

---

## Test evidence

- `pytest tests/evolution/` — 39 passed in 0.47s
- 33 pre-existing tests (ab_testing, synthetic_data, etc.) — zero regressions
- 6 new tests across the three promoted loops

## Remaining loop work (5 skeletons)

Tracked in `docs/AUDIT_FINDINGS_v0.1.2.md` *Open blockers for v1.0.0*. Phase 2
will target the next three skeletons in priority order; expected v0.1.5.
