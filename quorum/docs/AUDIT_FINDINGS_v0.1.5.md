# Self-Audit Findings v0.1.5 — Loop Scorecard Update

> Phase 2 continuation of the v0.1.4 audit. Two further evolution loops
> promoted from skeleton to functional, each design-driven by the Quorum
> engine itself (dogfooding).

See `AUDIT_FINDINGS_v0.1.4.md` for the v0.1.4 scorecard and
`AUDIT_FINDINGS_v0.1.2.md` for the original classification methodology.

---

## Loop scorecard delta

| Status         | v0.1.4 | v0.1.5 |
|----------------|--------|--------|
| **Functional** | 8      | **10** |
| **Partial**    | 0      | **0**  |
| **Skeleton**   | 5      | **3**  |

Net movement: +2 functional, −2 skeleton. Partial count remains zero.

---

## Loops promoted in v0.1.5

### Loop 11 — Self-prompting (skeleton → functional)

- **Module:** `quorum/src/quorum/evolution/self_prompt.py`
- **Wired:** `quorum/src/quorum/core/consensus.py` (lazy import,
  evolution_signals['self_prompt'] flag)
- **Tests:** `quorum/tests/evolution/test_self_prompt.py` (10 passed in 0.05s)
- **One-line note:** Query-time confidence-triggered rewrite. When
  first-pass confidence < `DEFAULT_REWRITE_CONFIDENCE_THRESHOLD=0.6`,
  `PromptRewriter.rewrite()` composes `ORIGINAL_QUERY` +
  `CLARIFIED_QUERY` and selected providers re-run on the rewritten
  prompt. Retry only adopted when `new_confidence > original`;
  delta logged unconditionally (positive and negative) so the
  meta-learner sees both wins and losses.
- **Quorum design verdict:** Quorum `--all` consensus on 3 design
  questions: (1) clarification + decomposition combined into a single
  rewrite template (cheaper than two calls; shared root cause = ambiguous
  query); (2) `ORIGINAL_QUERY` / `CLARIFIED_QUERY` markers preserve
  intent verbatim for downstream models; (3) `DEFAULT_REWRITE_MAX_ATTEMPTS=2`
  — first rewrite is the cheap win, second helps when the first rewrite
  is itself ambiguous, third is dominated by genuinely hard queries.
  Degrades to `None` on rewriter outage or when both Anthropic and OpenAI
  keys are missing.

### Loop 12 — Adversarial probing (skeleton → functional)

- **Module:** `quorum/src/quorum/evolution/adversarial.py`
- **Wired:** `quorum/src/quorum/core/consensus.py` (per-provider
  `penalty_multiplier` applied before consensus synthesis)
- **Tests:** `quorum/tests/evolution/test_adversarial.py` (5 passed in 0.13s)
- **One-line note:** 15 probes across 3 categories
  (injection / jailbreak / hallucination) as v0 floor, expandable via the
  public `DEFAULT_PROBES` list. Pattern-detection scoring (regex or
  callable on output) — explicitly avoids the recursive-vulnerability
  problem where an LLM-judge can itself be jailbroken. Vulnerable
  providers stay in the pool with weight multiplied by a coefficient
  in `[0.5, 1.0]` before synthesis (penalty-not-gating); catastrophic
  all-category failure is left to the orchestrator.
- **Quorum design verdict:** Quorum `--all` consensus on 3 design
  questions: (1) MVP probe count 20-30 across 5-7 categories — we
  shipped the spec minimum of 15 across 3 for the v0 floor;
  (2) pattern detection over LLM-judge — Quorum specifically endorsed
  canary-string detection for injection; (3) penalty-not-gating with
  multiplier floor 0.5 to keep diversity while reflecting risk.

---

## Updated scorecard summary

10 functional / 0 partial / 3 skeleton.

The remaining 3 skeleton loops are tracked in
`AUDIT_FINDINGS_v0.1.4.md` and remain unchanged in v0.1.5.

---

## Known pre-existing blockers (not introduced by v0.1.5)

- `tests/test_consensus.py` imports `_extract_disagreements` and
  `_score_agreement` which were removed in the v0.1 consensus refactor.
  Broken before v0.1.5; unrelated to either Loop 11 or Loop 12.

---

*Released: 2026-06-17. Tag: v0.1.5. Phase 2 of self-evolution audit complete.*
