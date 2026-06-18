# Quorum Research Mode

**Local Mac only. Black-box behavior, autonomous evolution. Never ships to hosted.**

This directory holds the experimental "research mode" tooling that gives Quorum
the kind of emergent, partially-opaque behavior that big-tech LLMs have — while
keeping the hosted commercial product (`api.quorum-ai.dev`) fail-closed,
auditable, and EU AI Act compliant.

## Opt-in

All scripts here refuse to run unless BOTH:

1. No hosted environment markers present (`K_SERVICE`, `KUBERNETES_SERVICE_HOST`, etc.)
2. `QUORUM_LOCAL_RESEARCH_MODE=1` set explicitly in your shell.

So: `export QUORUM_LOCAL_RESEARCH_MODE=1` once per session.

## Four levels of "black-box-ness"

### Level 1 — Self-modify auto-apply on strong consensus
Lives in `../quorum-self-modify/self_modify.py`. When research mode is on, a
proposal that ≥80% of the 10-LLM panel approves is **applied automatically**
with a git tag snapshot for rollback. Daily cap: 10 auto-applies (env
`QUORUM_RESEARCH_DAILY_CAP`).

### Level 2 — HSP gate bypass
`src/quorum/hsp/gate.py` lets the 8 gated evolution paths (`apply_to_rlhf`,
`update_policy`, `weekly_evolve`, etc.) run without webhook approval — but
every bypass writes a line to `~/.quorum/research-mode.log`. Hosted env always
ignores this bypass.

### Level 3 — LoRA fine-tune of Hermes 8B
`lora_finetune_setup.py` reads your RLHF feedback from `~/.quorum/rlhf.db`,
emits an MLX-LM training dataset + config tuned for M4. Training itself (6-12h)
runs separately with `mlx_lm.lora --config training_config.yaml`. The resulting
adapter holds learned weights that even you can't fully explain — that's the
"black box" you wanted, in a place where only you bear the risk.

### Level 4 — Genetic prompt evolution
`genetic_prompts.py` evolves a seed system prompt over N generations using
Gemini for mutation and Hermes-local for response generation. After 5+
generations the winner is often unrecognizable vs the seed — emergent,
not designed.

## Audit trail

Every action logs to `~/.quorum/research-mode.log`:

```
2026-06-18 2026-06-18T18:53:45Z HSP_BYPASS action='router_update_policy' risk='high' fn='update_policy'
2026-06-18 2026-06-18T19:02:11Z AUTO_APPLY pid='20260618-185930-...' snapshot='research-snapshot-...' approves=9 total=10
2026-06-19 2026-06-19T03:14:00Z FINETUNE_SETUP rows=247 out='/Users/facec/.quorum/finetune-hermes-v1'
```

You can grep this any time. If something gets weird, every snapshot tag in
git lets you roll back to before that auto-apply.

## What this is NOT

- ❌ Not for hosted production. Hosted always overrides these flags.
- ❌ Not a replacement for human review when you're tired. Even with
  autonomous mode on, the git log is your sanity check.
- ❌ Not a license to deploy untested mutations to paying B2B customers
  — the HSP gate guarantee for them stays intact.

## Quick start

```bash
export QUORUM_LOCAL_RESEARCH_MODE=1

# Level 1+2 demo — propose a small change, watch it auto-apply if panel agrees
. /tmp/.quorum-keys.env  # the 9 cloud keys
SELF_MODIFY_REVIEWERS=claude,hermes,llama,grok,gpt,mistral,cohere,deepseek \
python3 ../quorum-self-modify/self_modify.py propose \
    --repo /Users/facec/sovereignchain/quorum \
    --file src/quorum/_local_mode.py \
    --goal "improve clarity of the module docstring"

# Level 4 demo
echo "What's 2+2?" > /tmp/evals.txt
echo "Explain quantum entanglement briefly." >> /tmp/evals.txt
GEMINI_API_KEY=$(cat /tmp/.gem) python3 genetic_prompts.py \
    --seed "Answer concisely." \
    --task-file /tmp/evals.txt --generations 3 --population 4

# Level 3 setup (training itself happens separately)
python3 lora_finetune_setup.py --min-samples 50
```
