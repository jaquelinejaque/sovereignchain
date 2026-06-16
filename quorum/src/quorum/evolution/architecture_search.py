"""Loop 13 — Architecture Search for Consensus Topologies.

Copyright 2026 Sovereign Chain / Jaqueline Martins.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

HSP-Gated Module — Patent PCT/US26/11908.
Commercial adoption of search-discovered topologies requires an HSP license.
See LICENSE-HSP at the repository root for terms.

WHY THIS LOOP EXISTS
====================
The default `consensus()` engine runs every configured provider in parallel and
synthesizes a single answer. That is a *good* default — it minimises latency by
fanning out — but it is rarely *optimal* per query class:

    * For factual lookup ("What's the capital of Mali?"), one cheap model is
      already at ceiling; the other 7 parallel calls are pure waste.

    * For multi-step reasoning ("Plan a Keratin batch under £2k COGS with
      these constraints"), a hierarchical decomposition into specialist
      sub-questions (pricing model → fluid-mechanics model → regulatory model
      → synthesizer) beats a parallel vote because the specialists each see
      a smaller surface area.

    * For arbitration-style queries where one wrong model poisons the average
      (medical dosing, security postures), a tournament that knocks out
      low-confidence answers in early rounds dominates parallel averaging.

    * For cost-sensitive routing in production ("answer this customer ticket"),
      a *cascade* — try the cheap model first, escalate only on low confidence
      — beats both parallel and tournament on £/query at equal quality.

Rather than hand-tune which topology to use when, Loop 13 *learns* the answer.
It runs head-to-head experiments per query class, records (quality, cost,
latency) per topology, and recommends the Pareto winner. Adopting a new
recommendation is HSP-gated because a bad topology change can silently
degrade quality across an entire query class for thousands of downstream
users.

Trigger: weekly (cron). The orchestrator calls `run_experiment(...)` once
per query class with a fresh sample of recent queries, then optionally calls
`adopt(...)` for any class where the winner significantly beat the incumbent.

Persistence: SQLite at ~/.quorum/arch.db. All disk I/O is wrapped in
`asyncio.to_thread` so the event loop never blocks. Schema migrations are
forward-compatible (additive columns only).
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Final, Literal, Sequence

from quorum.core.consensus import ConsensusResult, _score_agreement, consensus
from quorum.hsp.gate import requires_hsp_approval
from quorum.providers.base import ModelResponse, Provider

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

Topology = Literal["parallel", "cascade", "tournament", "hierarchical"]
TOPOLOGIES: Final[tuple[Topology, ...]] = (
    "parallel",
    "cascade",
    "tournament",
    "hierarchical",
)

DATA_DIR: Final[Path] = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
DEFAULT_DB_PATH: Final[Path] = DATA_DIR / "arch.db"

# Confidence below which a cascade escalates to the next tier.
CASCADE_ESCALATE_THRESHOLD: Final[float] = 0.55

# How tightly to weight cost vs latency vs quality when picking a winner.
# quality_score - cost_penalty * cost_usd - latency_penalty * latency_s.
QUALITY_WEIGHT: Final[float] = 1.0
COST_PENALTY: Final[float] = 2.0  # per USD
LATENCY_PENALTY: Final[float] = 0.05  # per second

# Minimum samples before a recommendation is considered stable.
MIN_SAMPLES_FOR_RECOMMEND: Final[int] = 10


# --- Data types --------------------------------------------------------------


@dataclass(frozen=True)
class TopologyResult:
    """Outcome of one topology run on one query.

    `quality` is the consensus confidence on a 0..1 scale; `cost_usd` and
    `latency_ms` are summed across whatever sub-calls the topology made.
    """

    topology: Topology
    quality: float
    cost_usd: float
    latency_ms: float
    n_calls: int  # how many model invocations the topology actually made

    @property
    def score(self) -> float:
        """Single-scalar Pareto proxy. Higher is better.

        We bake quality, cost, and latency into one number so the search loop
        can pick a winner without a human in the loop for the *experiment*
        stage. The HSP gate still guards *adoption*.
        """
        return (
            QUALITY_WEIGHT * self.quality
            - COST_PENALTY * self.cost_usd
            - LATENCY_PENALTY * (self.latency_ms / 1000.0)
        )


@dataclass
class ArchPolicyRow:
    """One row of the arch_policy table — a learned recommendation."""

    query_class: str
    topology: Topology
    avg_quality: float
    avg_cost: float
    avg_latency: float
    samples: int


# --- Topology executors ------------------------------------------------------


async def _run_parallel(
    prompt: str,
    providers: Sequence[Provider],
    *,
    timeout_s: float,
) -> TopologyResult:
    """All providers in parallel — the current default consensus mode.

    Best for: simple factual queries where redundancy beats specialization,
    and latency matters more than cost.
    """
    t0 = time.perf_counter()
    res = await consensus(
        prompt,
        providers=list(providers),
        timeout_s=timeout_s,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    return TopologyResult(
        topology="parallel",
        quality=res.confidence,
        cost_usd=res.total_cost_usd,
        latency_ms=latency_ms,
        n_calls=len(providers),
    )


async def _run_cascade(
    prompt: str,
    providers: Sequence[Provider],
    *,
    timeout_s: float,
    escalate_below: float = CASCADE_ESCALATE_THRESHOLD,
) -> TopologyResult:
    """Cheap-first cascade: ask one model, escalate only if low confidence.

    We approximate "cheapness" via the provider order — caller passes them
    sorted ascending. On a confident first answer we stop, saving £ and ms.
    On a shaky answer we add the next tier and re-score.

    Best for: high-volume, cost-sensitive workloads (support tickets, RAG
    retrieval validation) where most queries are easy.
    """
    if not providers:
        return TopologyResult("cascade", 0.0, 0.0, 0.0, 0)

    t0 = time.perf_counter()
    chosen: list[Provider] = []
    last_res: ConsensusResult | None = None
    n_calls = 0

    for p in providers:
        chosen.append(p)
        last_res = await consensus(
            prompt,
            providers=list(chosen),
            timeout_s=timeout_s,
        )
        n_calls = len(chosen)
        # Stop escalating once confidence clears the bar — or after we've
        # exhausted the pool, which is its own answer.
        if last_res.confidence >= escalate_below or len(chosen) == len(providers):
            break

    latency_ms = (time.perf_counter() - t0) * 1000
    assert last_res is not None
    return TopologyResult(
        topology="cascade",
        quality=last_res.confidence,
        cost_usd=last_res.total_cost_usd,
        latency_ms=latency_ms,
        n_calls=n_calls,
    )


async def _run_tournament(
    prompt: str,
    providers: Sequence[Provider],
    *,
    timeout_s: float,
) -> TopologyResult:
    """Knockout rounds — bottom-half by per-model weight is eliminated each round.

    Round 1: all providers run in parallel; we score pairwise agreement and
    drop the bottom 50% by weight (the dissenters or the off-topic). Round 2:
    surviving providers run again on the same prompt with a "challenge"
    prefix to force re-evaluation. Repeat until one survivor.

    Best for: arbitration queries where one bad answer poisons an average
    (dosing, security, legal) — the knockout structure isolates outliers
    before they can dilute the final answer.
    """
    if not providers:
        return TopologyResult("tournament", 0.0, 0.0, 0.0, 0)

    t0 = time.perf_counter()
    pool: list[Provider] = list(providers)
    total_cost = 0.0
    total_calls = 0
    last_res: ConsensusResult | None = None
    round_no = 0

    while len(pool) > 1:
        round_no += 1
        round_prompt = (
            prompt
            if round_no == 1
            else f"Re-evaluate carefully: {prompt}"
        )
        res = await consensus(
            round_prompt,
            providers=pool,
            timeout_s=timeout_s,
        )
        total_cost += res.total_cost_usd
        total_calls += len(pool)
        last_res = res

        # Sort survivors by weight assigned by consensus; keep top half.
        ranked = sorted(
            res.models,
            key=lambda m: m.weight,
            reverse=True,
        )
        keep_n = max(1, len(pool) // 2)
        winners = {m.name for m in ranked[:keep_n] if not m.error}
        pool = [p for p in pool if p.name in winners] or pool[:1]

    if last_res is None:
        # Single provider — degenerate case; just run it once.
        last_res = await consensus(
            prompt, providers=pool, timeout_s=timeout_s,
        )
        total_cost += last_res.total_cost_usd
        total_calls += len(pool)

    latency_ms = (time.perf_counter() - t0) * 1000
    return TopologyResult(
        topology="tournament",
        quality=last_res.confidence,
        cost_usd=total_cost,
        latency_ms=latency_ms,
        n_calls=total_calls,
    )


async def _run_hierarchical(
    prompt: str,
    providers: Sequence[Provider],
    *,
    timeout_s: float,
) -> TopologyResult:
    """Specialists per sub-question, then a synthesizer call.

    We treat the first provider as the *planner* (decomposes the prompt into
    sub-questions), the middle providers as *specialists* (one per
    sub-question), and the last provider as the *synthesizer* (assembles the
    final answer). When the pool is too small to fill all three roles, roles
    are doubled up on the strongest available provider.

    Best for: multi-step reasoning where decomposition reduces each model's
    surface area — planning, multi-constraint optimization, multi-domain
    questions ("regulatory + chemistry + cost").
    """
    if not providers:
        return TopologyResult("hierarchical", 0.0, 0.0, 0.0, 0)

    t0 = time.perf_counter()
    pool = list(providers)
    planner = pool[0]
    synthesizer = pool[-1] if len(pool) >= 2 else pool[0]
    specialists = pool[1:-1] if len(pool) >= 3 else pool

    total_cost = 0.0
    n_calls = 0

    # Plan.
    plan_prompt = (
        "Decompose the following request into 2-4 independent sub-questions, "
        "one per line, prefixed 'Q:'. Do not answer them. Request: " + prompt
    )
    plan_resp = await planner.complete(plan_prompt)
    total_cost += plan_resp.cost_usd
    n_calls += 1
    sub_qs = [
        line.split("Q:", 1)[1].strip()
        for line in plan_resp.response.splitlines()
        if line.strip().startswith("Q:")
    ]
    if not sub_qs:
        # Planner refused or produced nothing parseable — fall back to one
        # sub-question = the original prompt. Hierarchical degrades to single
        # specialist + synth, which is still a valid topology.
        sub_qs = [prompt]

    # Specialists — one per sub-question, round-robin across the pool.
    spec_pool = specialists or pool
    spec_calls = await asyncio.gather(
        *[
            asyncio.wait_for(
                spec_pool[i % len(spec_pool)].complete(q),
                timeout=timeout_s,
            )
            for i, q in enumerate(sub_qs)
        ],
        return_exceptions=True,
    )
    spec_responses: list[ModelResponse] = []
    for sc in spec_calls:
        if isinstance(sc, ModelResponse):
            spec_responses.append(sc)
            total_cost += sc.cost_usd
            n_calls += 1

    if not spec_responses:
        # No specialist returned — emit a degenerate but valid result.
        latency_ms = (time.perf_counter() - t0) * 1000
        return TopologyResult(
            topology="hierarchical",
            quality=0.0,
            cost_usd=total_cost,
            latency_ms=latency_ms,
            n_calls=n_calls,
        )

    # Synthesize.
    synth_prompt = (
        "Given these sub-answers, synthesize a single coherent answer to the "
        "original request.\n\nOriginal: " + prompt + "\n\nSub-answers:\n"
        + "\n---\n".join(
            f"Q: {q}\nA: {r.response}"
            for q, r in zip(sub_qs, spec_responses)
        )
    )
    synth_resp = await synthesizer.complete(synth_prompt)
    total_cost += synth_resp.cost_usd
    n_calls += 1

    # Quality proxy: agreement among specialist sub-answers (we don't have a
    # second synthesizer to triangulate against, so this is our best signal).
    if len(spec_responses) >= 2:
        quality, _ = _score_agreement(spec_responses)
    else:
        # Single specialist — no agreement signal. Default to a neutral
        # mid-band so it doesn't dominate or get crushed by the cascade.
        quality = 0.5

    latency_ms = (time.perf_counter() - t0) * 1000
    return TopologyResult(
        topology="hierarchical",
        quality=quality,
        cost_usd=total_cost,
        latency_ms=latency_ms,
        n_calls=n_calls,
    )


_RUNNERS: dict[Topology, Callable[..., Awaitable[TopologyResult]]] = {
    "parallel": _run_parallel,
    "cascade": _run_cascade,
    "tournament": _run_tournament,
    "hierarchical": _run_hierarchical,
}


# --- ArchitectureSearch class -----------------------------------------------


class ArchitectureSearch:
    """Persistent search loop over consensus topologies per query class.

    Backed by SQLite at ~/.quorum/arch.db (overridable via QUORUM_ARCH_DB env
    var or constructor argument). Recommendations are stored as one row per
    (query_class, topology) so the table also functions as a leaderboard.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialise persistence.

        Args:
            db_path: Optional override for the SQLite file. If None, uses
                QUORUM_ARCH_DB env var, else ~/.quorum/arch.db. Parent
                directory is created if missing — we want zero-config out
                of the box for `pip install quorum && python -c "..."`.
        """
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            env_path = os.getenv("QUORUM_ARCH_DB")
            self.db_path = Path(env_path) if env_path else DEFAULT_DB_PATH

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # -- Schema ---------------------------------------------------------------

    def _init_schema(self) -> None:
        """Create the arch_policy table if it doesn't exist.

        Schema is forward-compatible: future columns must be added with
        ALTER TABLE ... ADD COLUMN and a default, never renamed in place.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS arch_policy (
                    query_class    TEXT NOT NULL,
                    topology       TEXT NOT NULL,
                    avg_quality    REAL NOT NULL DEFAULT 0.0,
                    avg_cost       REAL NOT NULL DEFAULT 0.0,
                    avg_latency    REAL NOT NULL DEFAULT 0.0,
                    samples        INTEGER NOT NULL DEFAULT 0,
                    updated_at     REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (query_class, topology)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS arch_adoption_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_class  TEXT NOT NULL,
                    topology     TEXT NOT NULL,
                    adopted_at   REAL NOT NULL,
                    reason       TEXT
                )
                """
            )
            conn.commit()

    # -- Public API -----------------------------------------------------------

    async def recommend_topology(self, query_class: str) -> Topology:
        """Return the best topology learned so far for a query class.

        Falls back to 'parallel' (the current default) when:
            * No samples exist for this query class.
            * No topology has reached MIN_SAMPLES_FOR_RECOMMEND.

        This conservative fallback matters: we don't want a single lucky
        experiment to flip production behaviour. Trust must be earned.
        """
        rows = await asyncio.to_thread(self._fetch_rows, query_class)
        if not rows:
            logger.info(
                "arch_search: no policy for query_class=%s, defaulting to parallel",
                query_class,
            )
            return "parallel"

        eligible = [r for r in rows if r.samples >= MIN_SAMPLES_FOR_RECOMMEND]
        if not eligible:
            logger.info(
                "arch_search: query_class=%s has data but no topology has "
                "%d+ samples yet; defaulting to parallel",
                query_class,
                MIN_SAMPLES_FOR_RECOMMEND,
            )
            return "parallel"

        # Pareto proxy: same scalar used in TopologyResult.score.
        def _score(r: ArchPolicyRow) -> float:
            return (
                QUALITY_WEIGHT * r.avg_quality
                - COST_PENALTY * r.avg_cost
                - LATENCY_PENALTY * (r.avg_latency / 1000.0)
            )

        winner = max(eligible, key=_score)
        return winner.topology

    async def run_experiment(
        self,
        query_class: str,
        n_queries: int = 50,
        *,
        sample_provider_pool: Sequence[Provider],
        sample_prompts: Sequence[str] | None = None,
        prompt_sampler: Callable[[], str] | None = None,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        """Run all 4 topologies on `n_queries` sample prompts and pick a winner.

        Either `sample_prompts` (an explicit list to draw from with
        replacement) or `prompt_sampler` (a zero-arg callable returning a
        fresh prompt) must be provided. In practice the orchestrator passes
        a sampler that pulls from the recent query log filtered by class —
        this keeps the experiment representative of production traffic.

        Returns a dict with per-topology aggregates and the winner. The
        winner is *not* automatically adopted — `adopt()` must be called,
        and `adopt()` is HSP-gated.
        """
        if sample_prompts is None and prompt_sampler is None:
            raise ValueError(
                "run_experiment requires either sample_prompts or prompt_sampler"
            )
        if n_queries <= 0:
            raise ValueError("n_queries must be positive")
        if not sample_provider_pool:
            raise ValueError("sample_provider_pool must contain at least one provider")

        def _next_prompt(i: int) -> str:
            if prompt_sampler is not None:
                return prompt_sampler()
            assert sample_prompts is not None
            return sample_prompts[i % len(sample_prompts)]

        per_topology: dict[Topology, list[TopologyResult]] = {
            t: [] for t in TOPOLOGIES
        }

        for i in range(n_queries):
            prompt = _next_prompt(i)
            # Shuffle a copy of the pool so cascade ordering doesn't
            # systematically favour one provider during the experiment.
            shuffled = list(sample_provider_pool)
            random.shuffle(shuffled)

            # Run all 4 topologies sequentially per query so we don't
            # multiply API spend by 4x in parallel during the experiment.
            # Topologies internally still parallelize their calls.
            for topo in TOPOLOGIES:
                runner = _RUNNERS[topo]
                try:
                    result = await runner(
                        prompt,
                        shuffled,
                        timeout_s=timeout_s,
                    )
                    per_topology[topo].append(result)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "arch_search: topology=%s failed on query %d: %s",
                        topo,
                        i,
                        e,
                    )

        # Aggregate.
        aggregates: dict[Topology, dict[str, float]] = {}
        for topo, results in per_topology.items():
            if not results:
                aggregates[topo] = {
                    "avg_quality": 0.0,
                    "avg_cost": 0.0,
                    "avg_latency": 0.0,
                    "avg_score": float("-inf"),
                    "samples": 0,
                }
                continue
            avg_q = sum(r.quality for r in results) / len(results)
            avg_c = sum(r.cost_usd for r in results) / len(results)
            avg_l = sum(r.latency_ms for r in results) / len(results)
            avg_s = sum(r.score for r in results) / len(results)
            aggregates[topo] = {
                "avg_quality": avg_q,
                "avg_cost": avg_c,
                "avg_latency": avg_l,
                "avg_score": avg_s,
                "samples": len(results),
            }

        # Persist aggregates.
        await asyncio.to_thread(
            self._upsert_aggregates, query_class, aggregates
        )

        # Pick winner by avg_score.
        scored = {
            t: a["avg_score"]
            for t, a in aggregates.items()
            if a["samples"] > 0
        }
        winner: Topology | None = (
            max(scored, key=lambda k: scored[k])  # type: ignore[arg-type]
            if scored
            else None
        )

        logger.info(
            "arch_search: experiment complete query_class=%s winner=%s "
            "aggregates=%s",
            query_class,
            winner,
            aggregates,
        )

        return {
            "query_class": query_class,
            "n_queries": n_queries,
            "aggregates": aggregates,
            "winner": winner,
        }

    @requires_hsp_approval(
        action="adopt_new_topology",
        risk_level="medium",
    )
    async def adopt(self, query_class: str, topology: Topology) -> dict[str, Any]:
        """Atomically adopt a topology as the recommendation for a query class.

        This is HSP-gated: a bad topology adoption can silently degrade
        quality for every downstream user of the class. The gate forces a
        human (or automated HSP webhook policy) to sign off.

        Adoption is implemented as a write to arch_adoption_log plus a
        rebalancing of arch_policy: the chosen row gets its `samples`
        floored at MIN_SAMPLES_FOR_RECOMMEND so future `recommend_topology`
        calls return it immediately even on a small experiment.
        """
        if topology not in TOPOLOGIES:
            raise ValueError(
                f"Unknown topology={topology!r}; must be one of {TOPOLOGIES}"
            )

        def _persist() -> None:
            now = time.time()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO arch_adoption_log
                        (query_class, topology, adopted_at, reason)
                    VALUES (?, ?, ?, ?)
                    """,
                    (query_class, topology, now, "experiment winner"),
                )
                # Ensure the adopted topology has a recommendable row.
                conn.execute(
                    """
                    INSERT INTO arch_policy
                        (query_class, topology, avg_quality, avg_cost,
                         avg_latency, samples, updated_at)
                    VALUES (?, ?, 0.0, 0.0, 0.0, ?, ?)
                    ON CONFLICT(query_class, topology) DO UPDATE SET
                        samples = MAX(samples, excluded.samples),
                        updated_at = excluded.updated_at
                    """,
                    (query_class, topology, MIN_SAMPLES_FOR_RECOMMEND, now),
                )
                conn.commit()

        await asyncio.to_thread(_persist)
        logger.warning(
            "arch_search: ADOPTED topology=%s for query_class=%s",
            topology,
            query_class,
        )
        return {
            "query_class": query_class,
            "topology": topology,
            "adopted_at": time.time(),
        }

    # -- Internal SQLite helpers ---------------------------------------------

    def _fetch_rows(self, query_class: str) -> list[ArchPolicyRow]:
        """Synchronous fetch — must be called via asyncio.to_thread."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """
                SELECT query_class, topology, avg_quality, avg_cost,
                       avg_latency, samples
                FROM arch_policy
                WHERE query_class = ?
                """,
                (query_class,),
            )
            return [
                ArchPolicyRow(
                    query_class=row[0],
                    topology=row[1],
                    avg_quality=row[2],
                    avg_cost=row[3],
                    avg_latency=row[4],
                    samples=row[5],
                )
                for row in cur.fetchall()
            ]

    def _upsert_aggregates(
        self,
        query_class: str,
        aggregates: dict[Topology, dict[str, float]],
    ) -> None:
        """Merge new experiment aggregates into arch_policy using a running mean.

        Running mean instead of overwrite: each experiment contributes its
        own evidence weighted by sample count. This keeps adoption stable
        across small weekly experiments while still letting the signal
        accumulate.
        """
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            for topo, agg in aggregates.items():
                if agg["samples"] == 0:
                    continue
                cur = conn.execute(
                    """
                    SELECT avg_quality, avg_cost, avg_latency, samples
                    FROM arch_policy
                    WHERE query_class = ? AND topology = ?
                    """,
                    (query_class, topo),
                )
                row = cur.fetchone()
                if row is None:
                    conn.execute(
                        """
                        INSERT INTO arch_policy
                            (query_class, topology, avg_quality, avg_cost,
                             avg_latency, samples, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            query_class,
                            topo,
                            agg["avg_quality"],
                            agg["avg_cost"],
                            agg["avg_latency"],
                            int(agg["samples"]),
                            now,
                        ),
                    )
                else:
                    prev_q, prev_c, prev_l, prev_n = row
                    new_n = prev_n + int(agg["samples"])
                    if new_n == 0:
                        continue
                    # Weighted running mean.
                    merged_q = (
                        prev_q * prev_n + agg["avg_quality"] * agg["samples"]
                    ) / new_n
                    merged_c = (
                        prev_c * prev_n + agg["avg_cost"] * agg["samples"]
                    ) / new_n
                    merged_l = (
                        prev_l * prev_n + agg["avg_latency"] * agg["samples"]
                    ) / new_n
                    conn.execute(
                        """
                        UPDATE arch_policy
                        SET avg_quality = ?, avg_cost = ?, avg_latency = ?,
                            samples = ?, updated_at = ?
                        WHERE query_class = ? AND topology = ?
                        """,
                        (
                            merged_q,
                            merged_c,
                            merged_l,
                            new_n,
                            now,
                            query_class,
                            topo,
                        ),
                    )
            conn.commit()


# --- Smoke tests -------------------------------------------------------------


class _FakeProvider(Provider):
    """Deterministic fake for tests — no network, controllable cost & quality.

    `quality_phrase` is what the model "says"; the consensus engine's
    Jaccard scorer turns overlapping phrases into high agreement.
    """

    def __init__(
        self,
        name: str,
        quality_phrase: str,
        *,
        cost: float = 0.001,
        latency_s: float = 0.01,
    ) -> None:
        self.name = name
        self._phrase = quality_phrase
        self._cost = cost
        self._latency_s = latency_s

    async def complete(
        self, prompt: str, *, max_tokens: int = 800
    ) -> ModelResponse:
        await asyncio.sleep(self._latency_s)
        return ModelResponse(
            name=self.name,
            response=self._phrase,
            latency_ms=self._latency_s * 1000,
            cost_usd=self._cost,
            tokens_in=len(prompt.split()),
            tokens_out=len(self._phrase.split()),
        )


async def _smoke_test_recommend_default(tmp_db: Path) -> None:
    """A fresh search instance with no data must recommend 'parallel'.

    This guards the safety invariant: out-of-the-box behaviour matches
    the current default, no surprises.
    """
    search = ArchitectureSearch(db_path=tmp_db)
    rec = await search.recommend_topology("any_class")
    assert rec == "parallel", f"expected parallel default, got {rec}"


async def _smoke_test_experiment_and_adopt(tmp_db: Path) -> None:
    """End-to-end: run a small experiment, verify aggregates land in SQLite,
    then exercise the HSP-gated adopt() path.

    Without HSP_GATE_WEBHOOK set the gate passes through, so adopt() should
    update arch_policy and arch_adoption_log without network calls.
    """
    providers = [
        _FakeProvider("cheap", "answer is forty two", cost=0.0001),
        _FakeProvider("mid", "answer is forty two precise", cost=0.001),
        _FakeProvider("expensive", "answer is forty two precise quantum", cost=0.01),
    ]
    search = ArchitectureSearch(db_path=tmp_db)
    result = await search.run_experiment(
        "smoke_class",
        n_queries=3,
        sample_provider_pool=providers,
        sample_prompts=["What is the answer?"],
    )
    assert result["winner"] in TOPOLOGIES, f"unexpected winner={result['winner']}"
    assert all(
        result["aggregates"][t]["samples"] > 0 for t in TOPOLOGIES
    ), f"missing topology samples: {result['aggregates']}"

    adopted = await search.adopt("smoke_class", result["winner"])
    assert adopted["topology"] == result["winner"]

    # Recommendation should now return the adopted topology.
    rec = await search.recommend_topology("smoke_class")
    assert rec == result["winner"], (
        f"expected adopted={result['winner']} to be recommended, got {rec}"
    )


def _run_smoke_tests() -> None:
    """CLI entry: `python -m quorum.evolution.architecture_search`."""
    import tempfile

    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "arch.db"
        asyncio.run(_smoke_test_recommend_default(db))
        asyncio.run(_smoke_test_experiment_and_adopt(db))
    logger.info("arch_search smoke tests passed")


__all__ = [
    "ArchitectureSearch",
    "ArchPolicyRow",
    "Topology",
    "TOPOLOGIES",
    "TopologyResult",
]


if __name__ == "__main__":
    _run_smoke_tests()
