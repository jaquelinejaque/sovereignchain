#!/usr/bin/env python3
"""Genetic Prompt Evolution — Quorum research mode.

Evolves a system prompt over N generations:
  1. Start with a seed prompt + fitness eval question.
  2. Generate K children via Gemini (paraphrase, expand, contract, simplify).
  3. Score each child by running it through the multi-LLM panel on
     a fixed eval set, taking median semantic agreement as fitness.
  4. Keep top-M children, retire losers, mutate the winners next gen.
  5. After N generations, surface the winner — often unrecognizable
     versus the seed (this is the "black-box-ish" behavior you wanted).

LOCAL ONLY. Refuses to run if hosted env markers are present.

Usage:
  QUORUM_LOCAL_RESEARCH_MODE=1 GEMINI_API_KEY=... \
      python3 genetic_prompts.py \
          --seed "Answer concisely. Cite uncertainty." \
          --task-file evals.txt \
          --generations 5 --population 4

evals.txt is a newline-delimited list of test questions. The fitness
score is the average self-modify-panel APPROVE rate across all eval
questions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Reuse adapters from self_modify
sys.path.insert(0, str(Path(__file__).parent.parent / "quorum-self-modify"))
from self_modify import (  # type: ignore
    _query_gemini, _query_ollama, _query_claude_via_cli,
    HOSTED_MARKERS,
)

GEN_DIR = Path.home() / ".quorum" / "prompt-evolution"


def _refuse_if_hosted() -> None:
    hosted = [m for m in HOSTED_MARKERS if os.environ.get(m)]
    if hosted:
        sys.exit(
            f"REFUSED: hosted markers present ({hosted}). "
            "Genetic evolution is LOCAL ONLY."
        )


def _check_research_mode() -> None:
    if os.environ.get("QUORUM_LOCAL_RESEARCH_MODE") != "1":
        sys.exit(
            "REFUSED: QUORUM_LOCAL_RESEARCH_MODE=1 required. "
            "This evolves prompts autonomously; opt in explicitly."
        )


MUTATE_PROMPT = """Take the system prompt below and create ONE mutated variant. The variant
should still aim at the same general goal (helpful, accurate answers) but
explore a DIFFERENT phrasing, structure, or emphasis. Try one of: paraphrase,
add a constraint, remove a constraint, change voice, restructure as bullet
list, restructure as flowing prose, add metacognitive instruction, simplify.

Pick the strategy that the original seems to be lacking. Output ONLY the
mutated prompt, no preamble, no quotes around it.

Original system prompt:
{prompt}"""


SCORE_PROMPT = """Rate the answer to the question on a scale of 0.0-1.0 where:
- 1.0 = correct, complete, well-calibrated
- 0.5 = partially correct or vague
- 0.0 = wrong, hallucinated, or refused without reason

Output ONLY a single number between 0.0 and 1.0.

Question: {question}

Answer: {answer}

Score:"""


def mutate(prompt: str) -> str:
    return _query_gemini(MUTATE_PROMPT.format(prompt=prompt)).strip().strip('"')


def answer_with(prompt: str, question: str, model_name: str = "hermes3:8b") -> str:
    combined = f"{prompt}\n\nQuestion: {question}"
    return _query_ollama(model_name, combined)


def score_answer(question: str, answer: str) -> float:
    raw = _query_gemini(SCORE_PROMPT.format(question=question, answer=answer))
    for tok in raw.replace("\n", " ").split():
        try:
            v = float(tok.strip(".,"))
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return 0.0


def evaluate_population(prompts: list[str], evals: list[str]) -> list[tuple[str, float]]:
    """Return [(prompt, fitness)] sorted descending by fitness."""
    scored: list[tuple[str, float]] = []
    for i, p in enumerate(prompts):
        print(f"  candidate {i+1}/{len(prompts)} — evaluating against {len(evals)} questions", file=sys.stderr)
        scores: list[float] = []
        for q in evals:
            ans = answer_with(p, q)
            scores.append(score_answer(q, ans))
        fit = sum(scores) / len(scores) if scores else 0.0
        scored.append((p, fit))
        print(f"    fitness = {fit:.3f}", file=sys.stderr)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", required=True, help="Starting prompt")
    ap.add_argument("--task-file", required=True, help="Newline-delimited eval questions")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--population", type=int, default=4, help="Children per generation")
    ap.add_argument("--keep-top", type=int, default=2, help="Top survivors carried forward")
    args = ap.parse_args()

    _refuse_if_hosted()
    _check_research_mode()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY required for mutation + scoring")

    evals = [ln.strip() for ln in Path(args.task_file).read_text().splitlines() if ln.strip()]
    if not evals:
        sys.exit("task-file is empty")

    GEN_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rundir = GEN_DIR / run_id
    rundir.mkdir()
    log = rundir / "evolution.log"

    def logl(msg: str) -> None:
        with log.open("a") as f:
            f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n")
        print(msg, file=sys.stderr)

    logl(f"START seed={args.seed!r} pop={args.population} gens={args.generations} evals={len(evals)}")

    population = [args.seed]
    for gen in range(args.generations):
        logl(f"GENERATION {gen+1}/{args.generations} pop_size={len(population)}")
        # Mutate every survivor up to target population
        while len(population) < args.population:
            parent = population[len(population) % max(1, args.keep_top)]
            try:
                child = mutate(parent)
                if child and child != parent:
                    population.append(child)
                else:
                    population.append(parent)  # fallback, won't survive selection
            except Exception as e:  # noqa: BLE001
                logl(f"  mutate failed: {e}")
                break

        ranked = evaluate_population(population[:args.population], evals)
        for i, (p, f) in enumerate(ranked):
            logl(f"  rank {i+1}: fitness={f:.3f}  prompt={p[:80]!r}")
        # Persist generation snapshot
        (rundir / f"gen_{gen+1:02d}.json").write_text(json.dumps(
            [{"rank": i+1, "fitness": f, "prompt": p} for i, (p, f) in enumerate(ranked)],
            indent=2,
        ))
        # Keep top-K survivors as seed for next gen
        population = [p for p, _ in ranked[:args.keep_top]]

    winner = population[0]
    logl(f"WINNER fitness-leader after {args.generations} generations:")
    logl(f"  {winner!r}")
    (rundir / "winner.txt").write_text(winner + "\n")
    print(f"\n✓ Evolution complete. Winner saved to {rundir}/winner.txt")
    print(f"  Run log: {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
