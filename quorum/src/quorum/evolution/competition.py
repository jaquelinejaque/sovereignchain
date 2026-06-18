# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sovereign Chain / Quorum contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Loop 7 — Model-vs-Model Competition.
# No HSP gate: the safety property is a 5%% sampling rate that bounds
# downstream RLHF-weight drift to a calibration-grade signal, not a
# steering vector. If sampling exceeds the safe envelope (>15%%) the
# orchestrator should escalate to the HSP gate.
"""Loop 7 — Model-vs-Model Competition.

WHY THIS EXISTS
---------------
Static RLHF weights rot. A model that was the strongest writer six months
ago may have been overtaken; a cheap local model may have surpassed an
expensive frontier model on a narrow query class. Without continuous
calibration, the consensus engine slowly mis-weights its inputs and
its confidence numbers become a lie.

The tournament fixes that. On a small fraction of live queries
(default 5%%) we run a blind pairwise duel: two random models answer
the same prompt, a third (judge) model picks the better answer
without seeing which is which. Winner +1, loser -1. Many tiny duels
over many queries converge to a per-(user, query_class) ranking that
tracks reality.

WHY 5%% AND NOT 100%%
--------------------
Two reasons:
  1. Cost: a tournament costs N*2 + N model calls on top of the normal
     consensus call. At 100%% sampling that triples the bill.
  2. Safety: aggressive RLHF updates from a noisy pairwise judge can
     stampede the weights. 5%% sampling + small learning rate
     (configurable, default 0.02) keeps the dynamics gentle. A sudden
     ranking shift then has to survive across many independent samples
     before it dominates.

WHY NO HSP GATE
---------------
The HSP gate exists to guard high-stakes, single-shot decisions. This
loop is the opposite: low-stakes, high-volume, statistical. The
sampling rate IS the safety property. Bumping the rate above the safe
envelope is the only escalation that should re-introduce the gate, and
the orchestrator (not this module) owns that decision.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

from quorum.hsp.gate import requires_hsp_approval
from quorum.providers.base import ModelResponse, Provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

DEFAULT_SAMPLING_RATE: float = 0.05
"""Fraction of live queries that trigger a tournament. 5%% is the default
calibration envelope; the orchestrator may override per environment."""

DEFAULT_LEARNING_RATE: float = 0.02
"""SGD step size applied to RLHF weights per duel outcome.
Kept small so a single noisy judgement can't dominate."""

DEFAULT_JUDGE_TIMEOUT_S: float = 30.0
"""Per-judge call timeout. Tournaments run in the background so we don't
want them to hang forever, but we also don't want to truncate a slow
judge unnecessarily."""

DEFAULT_DUEL_TIMEOUT_S: float = 30.0
"""Per-duelist call timeout."""

MIN_WEIGHT: float = 0.05
MAX_WEIGHT: float = 5.0
"""Hard clamps on the RLHF weight. Without these the SGD can drift to
zero (model effectively muted) or to infinity (model dominates). The
clamps keep every model in play even when it's losing."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Duel:
    """One pairwise comparison inside a tournament.

    Frozen because once the duel is decided we don't want callers
    mutating the record — auditability matters when RLHF weights move
    based on these outcomes.
    """

    round_index: int
    model_a: str
    model_b: str
    response_a: str
    response_b: str
    judge: str
    winner: str  # "a", "b", or "draw"
    judge_rationale: str
    latency_ms: float
    cost_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "judge": self.judge,
            "winner": self.winner,
            "judge_rationale": self.judge_rationale,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class CompetitionResult:
    """Aggregate outcome of a full tournament.

    The summary fields are the inputs the RLHF tracker actually consumes;
    `duels` is kept for audit/debug only.
    """

    query: str
    n_rounds: int
    duels: list[Duel] = field(default_factory=list)
    wins: dict[str, int] = field(default_factory=dict)
    losses: dict[str, int] = field(default_factory=dict)
    draws: dict[str, int] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    sampled: bool = True  # False if the sampling roll skipped this query

    @property
    def summary(self) -> dict[str, dict[str, int]]:
        """Per-model W/L/D, used by RLHF tracker and dashboards."""
        models = set(self.wins) | set(self.losses) | set(self.draws)
        return {
            m: {
                "wins": self.wins.get(m, 0),
                "losses": self.losses.get(m, 0),
                "draws": self.draws.get(m, 0),
            }
            for m in sorted(models)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "n_rounds": self.n_rounds,
            "summary": self.summary,
            "duels": [d.to_dict() for d in self.duels],
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "sampled": self.sampled,
        }


# ---------------------------------------------------------------------------
# RLHF tracker protocol + in-memory fallback
# ---------------------------------------------------------------------------


class RLHFTracker(Protocol):
    """Minimal interface the competition loop needs from an RLHF store.

    A real implementation hits Supabase / Postgres. The in-memory
    fallback below is enough for tests and offline development.
    """

    async def get_weight(self, user_id: str, query_class: str, model: str) -> float:
        ...

    async def set_weight(
        self, user_id: str, query_class: str, model: str, weight: float
    ) -> None:
        ...


class InMemoryRLHFTracker:
    """Process-local RLHF tracker.

    Used when no Supabase / Postgres URL is configured (tests, local dev,
    air-gapped CI). The on-disk variant below persists to SQLite so a
    crashed worker doesn't lose its calibration; this one is purely RAM.
    """

    def __init__(self) -> None:
        self._weights: dict[tuple[str, str, str], float] = {}
        self._lock = asyncio.Lock()

    async def get_weight(self, user_id: str, query_class: str, model: str) -> float:
        async with self._lock:
            return self._weights.get((user_id, query_class, model), 1.0)

    async def set_weight(
        self, user_id: str, query_class: str, model: str, weight: float
    ) -> None:
        async with self._lock:
            self._weights[(user_id, query_class, model)] = weight

    async def snapshot(self) -> dict[tuple[str, str, str], float]:
        """For debugging — dump the entire weight table."""
        async with self._lock:
            return dict(self._weights)


class SQLiteRLHFTracker:
    """SQLite-backed RLHF tracker.

    Used when QUORUM_RLHF_DB is set but no remote DB URL is configured.
    Writes go through asyncio.to_thread so the event loop never blocks
    on disk I/O. This is the recommended fallback for single-node
    deployments.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._init_lock = asyncio.Lock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because asyncio.to_thread may schedule
        # us on different worker threads across calls.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema_sync(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rlhf_weights (
                    user_id TEXT NOT NULL,
                    query_class TEXT NOT NULL,
                    model TEXT NOT NULL,
                    weight REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (user_id, query_class, model)
                )
                """
            )
            conn.commit()

    async def _ensure_schema(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._ensure_schema_sync)
            self._initialized = True

    def _get_sync(self, user_id: str, query_class: str, model: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT weight FROM rlhf_weights "
                "WHERE user_id=? AND query_class=? AND model=?",
                (user_id, query_class, model),
            ).fetchone()
        return float(row[0]) if row else 1.0

    async def get_weight(self, user_id: str, query_class: str, model: str) -> float:
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_sync, user_id, query_class, model)

    def _set_sync(
        self, user_id: str, query_class: str, model: str, weight: float
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rlhf_weights
                    (user_id, query_class, model, weight, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, query_class, model) DO UPDATE SET
                    weight=excluded.weight, updated_at=excluded.updated_at
                """,
                (user_id, query_class, model, weight, time.time()),
            )
            conn.commit()

    async def set_weight(
        self, user_id: str, query_class: str, model: str, weight: float
    ) -> None:
        await self._ensure_schema()
        await asyncio.to_thread(
            self._set_sync, user_id, query_class, model, weight
        )


def build_default_tracker() -> RLHFTracker:
    """Pick the best available tracker without keys.

    Resolution order:
      1. QUORUM_RLHF_DB (sqlite path) -> SQLite
      2. otherwise -> in-memory

    Callers in production are expected to inject their own Supabase /
    Postgres-backed tracker explicitly; this helper exists so tests
    and the CLI work zero-config.
    """
    db_path = os.environ.get("QUORUM_RLHF_DB")
    if db_path:
        logger.info("Using SQLite RLHF tracker at %s", db_path)
        return SQLiteRLHFTracker(db_path)
    logger.info("Using in-memory RLHF tracker (no QUORUM_RLHF_DB set)")
    return InMemoryRLHFTracker()


# ---------------------------------------------------------------------------
# Tournament engine
# ---------------------------------------------------------------------------


JUDGE_PROMPT = """\
You are a blind judge in a model competition. Two anonymous AI systems
(System A and System B) answered the same query. Pick the better answer
on correctness, completeness, and clarity. You do NOT know which model
produced which answer.

QUERY:
{query}

SYSTEM A:
{answer_a}

SYSTEM B:
{answer_b}

Respond with exactly one JSON object of the form:
  {{"winner": "a"}}  or  {{"winner": "b"}}  or  {{"winner": "draw"}}
followed by a one-sentence rationale on a new line.
"""


def _parse_judge_verdict(raw: str) -> tuple[str, str]:
    """Pull the winner + rationale out of the judge's free-form reply.

    We accept three failure modes gracefully:
      * Malformed JSON -> scan for the literal tokens "winner": "a/b/draw".
      * Capitalisation drift -> lowercase before matching.
      * No verdict at all -> return ("draw", raw).

    Returning "draw" on parse failure is intentional: a confused judge
    should not move RLHF weights.
    """
    text = raw.strip()
    rationale = ""
    try:
        first_brace = text.index("{")
        last_brace = text.index("}", first_brace) + 1
        obj = json.loads(text[first_brace:last_brace])
        winner = str(obj.get("winner", "draw")).strip().lower()
        rationale = text[last_brace:].strip()
    except (ValueError, json.JSONDecodeError):
        lowered = text.lower()
        if '"winner": "a"' in lowered or "winner: a" in lowered:
            winner = "a"
        elif '"winner": "b"' in lowered or "winner: b" in lowered:
            winner = "b"
        else:
            winner = "draw"
        rationale = text

    if winner not in {"a", "b", "draw"}:
        winner = "draw"
    return winner, rationale[:500]


class ModelCompetition:
    """Loop 7 — Model-vs-Model Competition.

    Stateless across queries. The RLHF state lives in the tracker; this
    class only orchestrates duels and computes deltas.
    """

    def __init__(
        self,
        *,
        sampling_rate: float = DEFAULT_SAMPLING_RATE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        duel_timeout_s: float = DEFAULT_DUEL_TIMEOUT_S,
        judge_timeout_s: float = DEFAULT_JUDGE_TIMEOUT_S,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in [0, 1]")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        self.sampling_rate = sampling_rate
        self.learning_rate = learning_rate
        self.duel_timeout_s = duel_timeout_s
        self.judge_timeout_s = judge_timeout_s
        # Injectable RNG -> deterministic tests.
        self._rng = rng or random.Random()

    # -- sampling -----------------------------------------------------------

    def should_sample(self) -> bool:
        """Decide whether this query triggers a tournament.

        Pulled out so callers (and tests) can stub the sampling
        behaviour without monkeypatching random.
        """
        return self._rng.random() < self.sampling_rate

    # -- one duel -----------------------------------------------------------

    async def _ask(
        self, provider: Provider, prompt: str, *, timeout_s: float
    ) -> ModelResponse:
        """Call one provider with bounded latency + total error containment.

        Mirrors core.consensus._call: never raise, always return a
        ModelResponse with .error set on failure. The competition loop
        treats any errored response as a forfeit (the other side wins).
        """
        t0 = time.perf_counter()
        try:
            resp = await asyncio.wait_for(provider.complete(prompt), timeout=timeout_s)
        except asyncio.TimeoutError:
            return ModelResponse(
                name=provider.name,
                response="",
                error="timeout",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:  # noqa: BLE001 — provider may raise anything
            logger.warning("Provider %s raised in duel: %s", provider.name, e)
            return ModelResponse(
                name=provider.name,
                response="",
                error=str(e)[:200],
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        resp.latency_ms = (time.perf_counter() - t0) * 1000
        return resp

    async def _run_duel(
        self,
        round_index: int,
        query: str,
        a: Provider,
        b: Provider,
        judge: Provider,
    ) -> Duel:
        """Run a single A-vs-B duel adjudicated by `judge`.

        Forfeit rules (kept explicit because they directly drive RLHF):
          * Both forfeit -> draw.
          * One forfeits -> the other wins by default; no judge call.
          * Judge forfeits -> draw (we refuse to invent a verdict).
        """
        ra, rb = await asyncio.gather(
            self._ask(a, query, timeout_s=self.duel_timeout_s),
            self._ask(b, query, timeout_s=self.duel_timeout_s),
        )

        a_failed = bool(ra.error) or not ra.response.strip()
        b_failed = bool(rb.error) or not rb.response.strip()

        # Forfeit fast paths — no point spending judge tokens.
        if a_failed and b_failed:
            return Duel(
                round_index=round_index,
                model_a=a.name,
                model_b=b.name,
                response_a=ra.response,
                response_b=rb.response,
                judge=judge.name,
                winner="draw",
                judge_rationale="both forfeited",
                latency_ms=ra.latency_ms + rb.latency_ms,
                cost_usd=ra.cost_usd + rb.cost_usd,
            )
        if a_failed:
            return Duel(
                round_index=round_index,
                model_a=a.name,
                model_b=b.name,
                response_a=ra.response,
                response_b=rb.response,
                judge=judge.name,
                winner="b",
                judge_rationale=f"a forfeited: {ra.error or 'empty'}",
                latency_ms=ra.latency_ms + rb.latency_ms,
                cost_usd=ra.cost_usd + rb.cost_usd,
            )
        if b_failed:
            return Duel(
                round_index=round_index,
                model_a=a.name,
                model_b=b.name,
                response_a=ra.response,
                response_b=rb.response,
                judge=judge.name,
                winner="a",
                judge_rationale=f"b forfeited: {rb.error or 'empty'}",
                latency_ms=ra.latency_ms + rb.latency_ms,
                cost_usd=ra.cost_usd + rb.cost_usd,
            )

        # Real judge call.
        judge_prompt = JUDGE_PROMPT.format(
            query=query, answer_a=ra.response, answer_b=rb.response
        )
        rj = await self._ask(judge, judge_prompt, timeout_s=self.judge_timeout_s)
        if rj.error or not rj.response.strip():
            winner, rationale = "draw", f"judge forfeited: {rj.error or 'empty'}"
        else:
            winner, rationale = _parse_judge_verdict(rj.response)

        return Duel(
            round_index=round_index,
            model_a=a.name,
            model_b=b.name,
            response_a=ra.response,
            response_b=rb.response,
            judge=judge.name,
            winner=winner,
            judge_rationale=rationale,
            latency_ms=ra.latency_ms + rb.latency_ms + rj.latency_ms,
            cost_usd=ra.cost_usd + rb.cost_usd + rj.cost_usd,
        )

    # -- public API ---------------------------------------------------------

    async def run_tournament(
        self,
        query: str,
        providers: Sequence[Provider],
        judge_provider: Provider,
        n_rounds: int = 3,
    ) -> CompetitionResult:
        """Run `n_rounds` duels and aggregate the W/L/D summary.

        Args:
            query: the user query the contestants answer.
            providers: pool of contestants. Must contain at least 2.
                The judge MAY be in this pool; we filter it out per duel
                so a model never judges its own answer.
            judge_provider: the adjudicator. Should generally be a
                stronger / more expensive model than the contestants.
            n_rounds: number of duels.

        Returns:
            CompetitionResult with per-model summary and full duel log.

        Why no early-stopping: with 5%% sampling we already throw away
        most queries; the ones we keep we want to extract maximum signal
        from. A small fixed `n_rounds` is simpler than dynamic stopping
        and easier to reason about in budget planning.
        """
        if n_rounds <= 0:
            raise ValueError("n_rounds must be >= 1")
        if len(providers) < 2:
            raise ValueError("Need at least 2 providers to run a tournament")

        result = CompetitionResult(query=query, n_rounds=n_rounds)
        t_start = time.perf_counter()

        # Filter out the judge from the contestant pool per query, not
        # per duel — a model that judged a query should never duel in
        # the same query either (avoids meta-bias).
        contestants = [p for p in providers if p.name != judge_provider.name]
        if len(contestants) < 2:
            raise ValueError(
                "Need at least 2 contestants distinct from the judge"
            )

        for round_index in range(n_rounds):
            a, b = self._rng.sample(contestants, 2)
            duel = await self._run_duel(round_index, query, a, b, judge_provider)
            result.duels.append(duel)
            result.total_cost_usd += duel.cost_usd

            if duel.winner == "a":
                result.wins[a.name] = result.wins.get(a.name, 0) + 1
                result.losses[b.name] = result.losses.get(b.name, 0) + 1
            elif duel.winner == "b":
                result.wins[b.name] = result.wins.get(b.name, 0) + 1
                result.losses[a.name] = result.losses.get(a.name, 0) + 1
            else:
                result.draws[a.name] = result.draws.get(a.name, 0) + 1
                result.draws[b.name] = result.draws.get(b.name, 0) + 1

        result.total_latency_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Tournament finished: %d rounds, summary=%s, cost=$%.4f, %.0fms",
            n_rounds,
            result.summary,
            result.total_cost_usd,
            result.total_latency_ms,
        )
        return result

    # -- RLHF application ---------------------------------------------------

    @requires_hsp_approval(action="competition_apply_to_rlhf", risk_level="high")
    async def apply_to_rlhf(
        self,
        rlhf_tracker: RLHFTracker,
        user_id: str,
        query_class: str,
        results: CompetitionResult,
    ) -> dict[str, float]:
        """Apply a small SGD step to each contestant's RLHF weight.

        Update rule per model:
            net = wins - losses
            delta = learning_rate * net
            w_new = clamp(w_old + delta, MIN_WEIGHT, MAX_WEIGHT)

        Draws don't move weights — they're either ties on the merits or
        judge confusion, neither of which is a signal worth acting on.

        Returns the new weight per model so the caller can log or
        broadcast the deltas.

        The reason we hit the tracker per model (instead of batching)
        is that the underlying store may not support transactions and
        the per-model cost is cheap — these calls happen at 5%% of
        traffic, not on the hot path.
        """
        if not results.sampled:
            return {}

        new_weights: dict[str, float] = {}
        for model, record in results.summary.items():
            net = record["wins"] - record["losses"]
            if net == 0:
                # Skip the round-trip when there's nothing to apply.
                continue
            old = await rlhf_tracker.get_weight(user_id, query_class, model)
            delta = self.learning_rate * net
            new = max(MIN_WEIGHT, min(MAX_WEIGHT, old + delta))
            await rlhf_tracker.set_weight(user_id, query_class, model, new)
            new_weights[model] = new
            logger.info(
                "RLHF update user=%s class=%s model=%s: %.4f -> %.4f (net=%+d)",
                user_id,
                query_class,
                model,
                old,
                new,
                net,
            )
        return new_weights


# ---------------------------------------------------------------------------
# Convenience: one-shot per-query trigger
# ---------------------------------------------------------------------------


async def maybe_run_competition(
    query: str,
    providers: Sequence[Provider],
    judge_provider: Provider,
    *,
    user_id: str,
    query_class: str,
    rlhf_tracker: RLHFTracker | None = None,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    n_rounds: int = 3,
) -> CompetitionResult:
    """Thin shim for the consensus engine's per-query hook.

    The engine calls this once per query. If the sampling roll comes up
    short we return an empty result with `sampled=False` and skip the
    tournament entirely — the caller can branch on that flag for logs
    or metrics without having to know about the sampling rate itself.

    A skipped result intentionally has zero duels and zero RLHF impact;
    the engine should NOT pass it to apply_to_rlhf (apply_to_rlhf also
    short-circuits on sampled=False as a belt-and-braces guard).
    """
    comp = ModelCompetition(sampling_rate=sampling_rate)
    if not comp.should_sample():
        return CompetitionResult(query=query, n_rounds=0, sampled=False)

    result = await comp.run_tournament(
        query=query,
        providers=providers,
        judge_provider=judge_provider,
        n_rounds=n_rounds,
    )
    if rlhf_tracker is None:
        rlhf_tracker = build_default_tracker()
    await comp.apply_to_rlhf(rlhf_tracker, user_id, query_class, result)
    return result


# ---------------------------------------------------------------------------
# CompetitionStore — ELO-style pairwise ranking from natural query traffic.
# ---------------------------------------------------------------------------
#
# WHY a second mechanism alongside ModelCompetition
# -------------------------------------------------
# ModelCompetition runs *explicit* tournaments at 5%% sampling: an extra
# judge call per duel, deliberately paid for. The store below is the
# *implicit* sibling: every regular consensus call already produces N
# answers for the same prompt — that's a free pairwise battlefield. We
# decide winners by "who agrees most with the consensus answer" (cheap
# lexical similarity; the consensus engine already paid for the embedding
# cost upstream, but we don't carry it across the call boundary), and
# fold the outcome into per-(model, query_class) ELO ratings.
#
# WHO WINS without ground truth
# -----------------------------
# Quorum design consensus (Gemini + Llama + Claude + Grok, 2026-06-16):
# pick (a) — similarity to the current top-weighted (consensus) answer —
# as the *base* signal. RLHF feedback (b) can layer on later via a
# blended observation, but we don't gate the loop on its availability.
# Rationale: similarity-to-canonical is always available, always cheap,
# and converges to the same ranking as RLHF in the limit because the
# top-weighted answer is itself shaped by RLHF priors.
#
# WHY K=16 not 32
# ---------------
# Standard chess uses K=32 with one battle per pairing per event. We get
# (N choose 2) battles per query, dozens of queries per minute. K=32
# would let a single noisy query swing a rating by ~96 points; K=16
# halves that to ~48, still letting ratings move meaningfully but
# preventing one bad day from steamrollering a model's reputation.
# Quorum panel confirmed: "lower K is correct when you have many
# correlated observations per event."
#
# WHY per-query-class
# -------------------
# A model that's excellent at code may be mediocre at creative writing.
# A global ELO would average those signals into mush and the router
# couldn't pick "the best fighter for this query." We key on the
# router's existing classify_query() taxonomy so the rankings drop
# straight into MoERouter without translation.
#
# WHY similarity = word-overlap (Jaccard)
# ---------------------------------------
# Two reasons. (1) The store is called inside the consensus hot path; we
# can't justify a second embedding round-trip on every query. (2) For
# the *pairwise* comparison we only need a monotone proxy for "closer to
# canonical." Jaccard on lowercased token sets is monotone with
# semantic similarity at the granularity that matters here (was your
# answer roughly in the same lexical neighbourhood as the winner?) and
# never crashes when the embedding backend is unreachable.


_ELO_DEFAULT_RATING: float = 1500.0
"""Standard ELO start rating. Every (model, query_class) row materialises
here on first observation so nobody is privileged on day one."""

_ELO_K_FACTOR: float = 16.0
"""See module note above for why this is 16, not 32."""

_ELO_DB_FILENAME: str = "competition.db"
"""Lives under ``${QUORUM_DATA_DIR}/competition.db`` (default ``~/.quorum``)."""


def _default_competition_db_path() -> Path:
    """Resolve the ELO SQLite path honouring ``QUORUM_DATA_DIR``.

    Resolved at call time (not import time) so tests can monkeypatch
    ``QUORUM_DATA_DIR`` per case without having to reload the module.
    """
    raw = os.environ.get("QUORUM_DATA_DIR") or str(Path.home() / ".quorum")
    return Path(raw).expanduser() / _ELO_DB_FILENAME


def _jaccard_similarity(a: str, b: str) -> float:
    """Cheap symmetric similarity in [0, 1] used to decide pairwise winners.

    Lowercased word-set overlap. Empty-vs-empty returns 0.0 (degenerate;
    we never want a battle between two no-answers to count as a clean
    win for either side). One-side-empty returns 0.0 so the non-empty
    side strictly wins.
    """
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Classic ELO expected-score formula.

    Returns the probability that A beats B given their current ratings.
    Used for both the prediction step (so a high-rated model needs less
    confirmation to climb further) and the update step (delta is
    proportional to surprise = actual - expected).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _derive_pairwise_battles(
    responses: Sequence[ModelResponse],
    consensus_answer: str,
) -> list[tuple[str, str]]:
    """Turn a single query into (winner_name, loser_name) battle tuples.

    Per Quorum design: for each unordered pair (i, j) of valid responses,
    whichever response has higher Jaccard similarity to the consensus
    answer wins that battle. Ties are silently dropped — no signal, no
    update — because feeding ties into ELO at K=16 is just noise.

    The consensus answer is one of the responses by construction
    (canonical = top-weighted response). That response will tie itself
    against itself with similarity 1.0 (which we exclude via the i<j
    iteration anyway) and will tend to win every pair it's in. This is
    intentional: the top-weighted model is the round's de facto winner.
    """
    valid = [r for r in responses if r.response and not r.error]
    out: list[tuple[str, str]] = []
    sims = [_jaccard_similarity(r.response, consensus_answer) for r in valid]
    n = len(valid)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = sims[i], sims[j]
            if si == sj:
                # Either both identical to the consensus or both equally
                # far from it. No information; skip.
                continue
            if si > sj:
                out.append((valid[i].name, valid[j].name))
            else:
                out.append((valid[j].name, valid[i].name))
    return out


@dataclass(slots=True, frozen=True)
class EloRow:
    """Read-only snapshot of one (model, query_class) row.

    Frozen because callers (dashboards, the router) treat ratings as a
    quoted price — mutation would let a downstream bug silently rewrite
    the leaderboard.
    """

    model_name: str
    query_class: str
    rating: float
    games: int
    updated_at: float


class CompetitionStore:
    """SQLite-backed ELO leaderboard keyed by ``(model_name, query_class)``.

    All disk I/O is wrapped in ``asyncio.to_thread`` so the event loop is
    never blocked — the consensus hot path calls ``record_query`` once
    per query and must not stall on a write. ``WAL`` is enabled for the
    same reason (concurrent readers from CLI / dashboards never block
    the writer).
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path else _default_competition_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_lock = asyncio.Lock()
        self._initialized = False

    # -- schema -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        # check_same_thread=False because asyncio.to_thread may move us
        # between worker threads. WAL journalling keeps reads concurrent
        # with writes — the router will scan rankings frequently.
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema_sync(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS elo (
                    model_name  TEXT NOT NULL,
                    query_class TEXT NOT NULL,
                    rating      REAL NOT NULL,
                    games       INTEGER NOT NULL DEFAULT 0,
                    updated_at  REAL NOT NULL,
                    PRIMARY KEY (model_name, query_class)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_elo_class_rating "
                "ON elo(query_class, rating DESC)"
            )
            conn.commit()

    async def _ensure_schema(self) -> None:
        async with self._init_lock:
            if self._initialized:
                return
            await asyncio.to_thread(self._ensure_schema_sync)
            self._initialized = True

    # -- core read helpers ------------------------------------------------

    def _get_rating_sync(self, model_name: str, query_class: str) -> tuple[float, int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rating, games FROM elo "
                "WHERE model_name=? AND query_class=?",
                (model_name, query_class),
            ).fetchone()
        if row is None:
            return _ELO_DEFAULT_RATING, 0
        return float(row[0]), int(row[1])

    async def get_rating(self, model_name: str, query_class: str) -> tuple[float, int]:
        """Return ``(rating, games_played)``; default rating if absent.

        Used by tests and by the router when it needs to bias provider
        selection. New (model, class) pairs return 1500.0 / 0 — the
        canonical ELO unseen-player default.
        """
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_rating_sync, model_name, query_class)

    # -- core write -------------------------------------------------------

    def _observe_battle_sync(
        self, query_class: str, winner: str, loser: str
    ) -> tuple[float, float]:
        """One battle = two row writes inside a single transaction.

        Returning the new ratings is a convenience for logs and tests;
        the persisted values are the source of truth.
        """
        now = time.time()
        with self._connect() as conn:
            # Read both rows first so the rating delta is symmetric.
            win_row = conn.execute(
                "SELECT rating, games FROM elo "
                "WHERE model_name=? AND query_class=?",
                (winner, query_class),
            ).fetchone()
            los_row = conn.execute(
                "SELECT rating, games FROM elo "
                "WHERE model_name=? AND query_class=?",
                (loser, query_class),
            ).fetchone()
            r_w = float(win_row[0]) if win_row else _ELO_DEFAULT_RATING
            g_w = int(win_row[1]) if win_row else 0
            r_l = float(los_row[0]) if los_row else _ELO_DEFAULT_RATING
            g_l = int(los_row[1]) if los_row else 0

            # Classic ELO update. Winner's actual score = 1, expected
            # depends on the gap. A higher-rated winner gains less; a
            # lower-rated upset gains more. Symmetric for the loser.
            exp_w = _expected_score(r_w, r_l)
            new_r_w = r_w + _ELO_K_FACTOR * (1.0 - exp_w)
            new_r_l = r_l + _ELO_K_FACTOR * (0.0 - (1.0 - exp_w))

            conn.execute(
                """
                INSERT INTO elo (model_name, query_class, rating, games, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_name, query_class) DO UPDATE SET
                    rating=excluded.rating,
                    games=excluded.games,
                    updated_at=excluded.updated_at
                """,
                (winner, query_class, new_r_w, g_w + 1, now),
            )
            conn.execute(
                """
                INSERT INTO elo (model_name, query_class, rating, games, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(model_name, query_class) DO UPDATE SET
                    rating=excluded.rating,
                    games=excluded.games,
                    updated_at=excluded.updated_at
                """,
                (loser, query_class, new_r_l, g_l + 1, now),
            )
            conn.commit()
        return new_r_w, new_r_l

    async def observe_battle(
        self, query_class: str, winner: str, loser: str
    ) -> tuple[float, float]:
        """Apply one ELO update. Returns ``(winner_rating, loser_rating)`` post-update.

        Self-battles (winner == loser) are no-ops: feeding them in would
        be a bug, and silently logging them as draws would hide it.
        """
        if winner == loser:
            logger.debug("observe_battle skipped: self-battle %s", winner)
            cur, _ = await self.get_rating(winner, query_class)
            return cur, cur
        await self._ensure_schema()
        return await asyncio.to_thread(
            self._observe_battle_sync, query_class, winner, loser
        )

    async def record_query(
        self,
        query_class: str,
        responses: Sequence[ModelResponse],
        consensus_answer: str,
    ) -> int:
        """Derive pairwise battles from one query, persist all ELO updates.

        Returns the number of battles actually applied (zero if every
        pair tied or fewer than two valid responses came back). Failures
        in one battle don't abort the rest — each pair is its own write
        transaction, so a transient SQLite hiccup degrades gracefully.
        """
        if not consensus_answer:
            return 0
        battles = _derive_pairwise_battles(responses, consensus_answer)
        if not battles:
            return 0
        applied = 0
        for winner, loser in battles:
            try:
                await self.observe_battle(query_class, winner, loser)
                applied += 1
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "ELO observe_battle failed (winner=%s loser=%s class=%s): %s",
                    winner,
                    loser,
                    query_class,
                    e,
                )
        return applied

    # -- read for router / dashboards -------------------------------------

    def _get_rankings_sync(
        self, query_class: str, top_n: int
    ) -> list[tuple[str, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_name, rating FROM elo "
                "WHERE query_class=? "
                "ORDER BY rating DESC LIMIT ?",
                (query_class, top_n),
            ).fetchall()
        return [(str(r[0]), float(r[1])) for r in rows]

    async def get_rankings(
        self, query_class: str, top_n: int = 8
    ) -> list[tuple[str, float]]:
        """Top-N models for one class, sorted by rating descending.

        ``top_n=8`` matches the consensus engine's default fan-out cap —
        the router will typically request exactly the number of fighters
        it intends to enter into the next round of consensus.
        """
        if top_n <= 0:
            return []
        await self._ensure_schema()
        return await asyncio.to_thread(self._get_rankings_sync, query_class, top_n)


# ---------------------------------------------------------------------------
# Smoke tests — runnable via `python -m quorum.evolution.competition`
# ---------------------------------------------------------------------------


class _FakeProvider(Provider):
    """Deterministic provider for unit tests.

    `quality` is a float; higher quality means the judge stub will pick
    this provider. Returns a fixed string per call so the verdict logic
    can be exercised without touching a real LLM.
    """

    def __init__(self, name: str, quality: float, *, fail: bool = False) -> None:
        self.name = name
        self.quality = quality
        self.fail = fail

    async def complete(
        self, prompt: str, *, max_tokens: int = 800
    ) -> ModelResponse:
        if self.fail:
            return ModelResponse(name=self.name, response="", error="injected")
        # Embed the quality so a deterministic judge can score it.
        return ModelResponse(
            name=self.name,
            response=f"[{self.name} q={self.quality}] {prompt[:32]}",
        )


class _DeterministicJudge(Provider):
    """Judge stub: picks whichever response embeds the higher q= value.

    Lets us test the W/L/D bookkeeping without hitting a real LLM.
    """

    name = "deterministic-judge"

    async def complete(
        self, prompt: str, *, max_tokens: int = 800
    ) -> ModelResponse:
        import re

        scores = [float(x) for x in re.findall(r"q=([0-9.]+)", prompt)]
        if len(scores) < 2:
            return ModelResponse(name=self.name, response='{"winner": "draw"}')
        if scores[0] > scores[1]:
            return ModelResponse(
                name=self.name,
                response='{"winner": "a"}\nA scored higher.',
            )
        if scores[1] > scores[0]:
            return ModelResponse(
                name=self.name,
                response='{"winner": "b"}\nB scored higher.',
            )
        return ModelResponse(name=self.name, response='{"winner": "draw"}\nTie.')


async def _smoke_test_tournament() -> None:
    """End-to-end: stronger model should win more duels."""
    strong = _FakeProvider("strong", quality=0.9)
    weak = _FakeProvider("weak", quality=0.1)
    judge = _DeterministicJudge()

    comp = ModelCompetition(rng=random.Random(42))
    result = await comp.run_tournament(
        query="What is 2+2?",
        providers=[strong, weak],
        judge_provider=judge,
        n_rounds=5,
    )

    assert result.n_rounds == 5
    assert len(result.duels) == 5
    summary = result.summary
    assert summary["strong"]["wins"] > summary["weak"]["wins"], summary
    logger.info("smoke_test_tournament OK: %s", summary)


async def _smoke_test_rlhf_apply() -> None:
    """apply_to_rlhf moves the winner up and the loser down, within clamps."""
    tracker = InMemoryRLHFTracker()
    await tracker.set_weight("u1", "general", "strong", 1.0)
    await tracker.set_weight("u1", "general", "weak", 1.0)

    result = CompetitionResult(query="q", n_rounds=2)
    result.wins["strong"] = 2
    result.losses["weak"] = 2

    comp = ModelCompetition(learning_rate=0.1)
    new = await comp.apply_to_rlhf(tracker, "u1", "general", result)

    assert new["strong"] > 1.0, new
    assert new["weak"] < 1.0, new
    # Clamp check.
    assert MIN_WEIGHT <= new["weak"] <= MAX_WEIGHT
    assert MIN_WEIGHT <= new["strong"] <= MAX_WEIGHT
    logger.info("smoke_test_rlhf_apply OK: %s", new)


async def _smoke_test_forfeit() -> None:
    """A failing provider forfeits without spending judge tokens."""
    strong = _FakeProvider("strong", quality=0.9)
    broken = _FakeProvider("broken", quality=0.1, fail=True)
    judge = _DeterministicJudge()

    comp = ModelCompetition(rng=random.Random(0))
    result = await comp.run_tournament(
        query="ping", providers=[strong, broken], judge_provider=judge, n_rounds=3
    )
    summary = result.summary
    assert summary["broken"]["wins"] == 0
    assert summary["strong"]["wins"] == 3, summary
    logger.info("smoke_test_forfeit OK: %s", summary)


async def _smoke_test_sampling_skip() -> None:
    """sampling_rate=0 returns a sampled=False result and no duels."""
    strong = _FakeProvider("strong", quality=0.9)
    weak = _FakeProvider("weak", quality=0.1)
    judge = _DeterministicJudge()

    res = await maybe_run_competition(
        "q",
        providers=[strong, weak],
        judge_provider=judge,
        user_id="u1",
        query_class="general",
        rlhf_tracker=InMemoryRLHFTracker(),
        sampling_rate=0.0,
    )
    assert res.sampled is False
    assert res.duels == []
    logger.info("smoke_test_sampling_skip OK")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    async def _main() -> None:
        await _smoke_test_tournament()
        await _smoke_test_rlhf_apply()
        await _smoke_test_forfeit()
        await _smoke_test_sampling_skip()

    asyncio.run(_main())
