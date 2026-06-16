"""Quorum Evolution Loop 6 — Meta-learning.

Copyright 2026 Sovereign Chain Ltd.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

This module is NOT HSP-gated. The meta-learner is observational: it watches
which other evolution loops actually improve downstream quality (consensus
confidence + RLHF reward) and reallocates orchestrator compute toward the
winners. It does not, by itself, mutate any model weights or policies — it only
publishes priorities. Because of that, no PCT/US26/11908 (HSP) gate is needed.

WHY a separate meta-loop?
    Each evolution loop (RLHF, prompt-mutation, provider-weight tuning, prior-
    elicitation, etc.) declares its own optimum. Without a meta-observer, they
    compete for the same compute budget and we end up funding whichever loop
    *runs most often* rather than whichever loop *helps most*. Loop 6 closes
    that gap: it scores each sibling loop on the delta it produced to a north-
    star quality metric, then publishes a normalized priority vector that the
    orchestrator uses to allocate the next week's compute.

WHY weekly?
    Evolution-loop impact is high-variance over short windows (a single bad
    sample can swamp the signal). A weekly cadence gives every loop time to
    accumulate enough before/after measurements to be statistically meaningful,
    while still being responsive enough that a regressing loop is defunded
    within ~7 days.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Weight given to consensus-confidence delta vs RLHF-reward delta when scoring
# a loop's impact. Confidence is cheap and ubiquitous; reward is sparse but
# closer to the true objective. 40/60 favors the rarer-but-truer signal.
_CONFIDENCE_WEIGHT: float = 0.4
_REWARD_WEIGHT: float = 0.6

# Minimum priority floor so a momentarily-bad loop is not starved to zero
# (which would prevent it from ever earning back its share). 2% per loop.
_PRIORITY_FLOOR: float = 0.02

# How many recent impact measurements to average when computing priorities.
# 8 ≈ ~2 months of weekly data, enough to smooth noise without going stale.
_PRIORITY_WINDOW: int = 8

# Smoothing factor for exponential-moving-average priority update. New
# evaluations contribute 30% weight; previous priority keeps 70%. This dampens
# whiplash from a single outlier week.
_EMA_ALPHA: float = 0.3

# Default DB path under user home so multiple Quorum installs share state.
DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
_DEFAULT_DB_PATH = DATA_DIR / "meta.db"

# ---------------------------------------------------------------------------
# Online per-(loop, query_class) learning constants
# ---------------------------------------------------------------------------
# Quorum consensus design (92% confidence, 5/7 backends agreeing): use a
# *hybrid* update strategy — confidence-delta as primary signal, loop-fired
# binary as the gating filter, and cold-start = "enable all" so the learner
# can discover emergent strengths instead of being biased by initial priors.
#
# Minimum samples a (loop, class) pair must accumulate before its measured
# mean delta is trusted enough to *exclude* a loop. Below this floor, the
# loop is always recommended (exploration > exploitation while sample-poor).
_MIN_SAMPLES_FOR_EXCLUSION: int = 5

# A loop is excluded for a class only if its mean confidence_delta is at or
# below this threshold AND it has enough samples. Set slightly negative so a
# loop must *consistently hurt* (not just be neutral) to get dropped.
_EXCLUSION_DELTA_THRESHOLD: float = -1e-6


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopImpact:
    """One measured before/after impact event for a sibling evolution loop.

    Kept as an immutable dataclass (rather than pydantic) because rows are
    written once and never mutated, and frozen dataclasses are ~3x faster to
    construct on the hot path of measurement ingestion.
    """

    loop_name: str
    timestamp: float
    confidence_delta: float
    reward_delta: float
    weight_shift: float
    score: float

    def as_row(self) -> tuple[str, float, float, float, float, float]:
        """Flatten for SQLite insertion in column order matching the schema."""
        return (
            self.loop_name,
            self.timestamp,
            self.confidence_delta,
            self.reward_delta,
            self.weight_shift,
            self.score,
        )


# ---------------------------------------------------------------------------
# MetaLearner
# ---------------------------------------------------------------------------


class MetaLearner:
    """Tracks per-loop impact and publishes orchestrator priorities.

    WHY a class rather than module-level functions?
        Tests need to point at an isolated temp DB, and the orchestrator wants
        to inject a custom path in production. Encapsulating the DB handle on
        an instance makes both ergonomic without globals.

    WHY SQLite (not Redis / a JSON file)?
        - The orchestrator is single-host: no cross-process contention.
        - We need cheap range scans ("last N impacts for loop X") which JSON
          doesn't give us.
        - SQLite ships with Python; zero ops cost. If we ever need multi-host
          we can swap the storage backend without touching the public API.
    """

    def __init__(self, db_path: str | os.PathLike[str] | None = None) -> None:
        """Open (or create) the meta-learning DB.

        WHY eager initialization?
            Schema mistakes should fail fast, at construction, not on first
            weekly run a month later in production.
        """
        self._db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("MetaLearner initialized at %s", self._db_path)

    # -- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        """Create tables idempotently.

        WHY two tables (not one)?
            `loop_impacts` is append-only event log used for analysis and
            audit. `loop_priorities` is the *current* priority vector consumed
            by the orchestrator on every request — separating them lets the
            hot read path hit a single tiny row per loop instead of scanning
            the event log.
        """
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS loop_impacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    loop_name TEXT NOT NULL,
                    ts REAL NOT NULL,
                    confidence_delta REAL NOT NULL,
                    reward_delta REAL NOT NULL,
                    weight_shift REAL NOT NULL,
                    score REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_loop_impacts_name_ts
                    ON loop_impacts (loop_name, ts DESC);

                CREATE TABLE IF NOT EXISTS loop_priorities (
                    loop_name TEXT PRIMARY KEY,
                    priority REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    loops_evaluated INTEGER NOT NULL,
                    notes TEXT
                );

                -- Online per-(loop, query_class) effect tracker.
                -- Composite PK is the natural lookup key for `observe` and
                -- `recommend_loops`; storing aggregates here (sum + count)
                -- lets us update online with one UPSERT instead of scanning
                -- the event log on every consensus call.
                CREATE TABLE IF NOT EXISTS loop_effect (
                    loop_name TEXT NOT NULL,
                    query_class TEXT NOT NULL,
                    fired_count INTEGER NOT NULL DEFAULT 0,
                    confidence_delta_sum REAL NOT NULL DEFAULT 0.0,
                    samples INTEGER NOT NULL DEFAULT 0,
                    last_confidence REAL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (loop_name, query_class)
                );

                -- Per-query_class baseline (rolling mean of final confidence).
                -- Used to compute the *delta* signal even though caller only
                -- passes the absolute final_confidence to observe().
                CREATE TABLE IF NOT EXISTS class_baseline (
                    query_class TEXT PRIMARY KEY,
                    confidence_sum REAL NOT NULL DEFAULT 0.0,
                    samples INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection.

        WHY not keep a persistent connection?
            We jump between threads via asyncio.to_thread; sqlite3 connections
            are not thread-safe by default. A new connection per call is
            cheap (<1ms) and removes a whole class of "object created in
            thread A used in thread B" bugs.
        """
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- measurement ------------------------------------------------------

    def measure_loop_impact(
        self,
        loop_name: str,
        before_metrics: Mapping[str, Any],
        after_metrics: Mapping[str, Any],
    ) -> float:
        """Score a single loop's impact and persist it.

        Inputs are intentionally loose mappings (rather than typed dataclasses)
        so each sibling loop can pass whatever instrumentation it already
        emits. We only require the keys we actually read.

        Expected keys (all optional, default 0.0):
          - "avg_confidence": float in [0, 1], post-loop consensus confidence
          - "avg_reward":     float in [-1, 1], RLHF preference signal
          - "weight_shift":   float >= 0, L2 norm of the weight/policy delta

        Returns
        -------
        score : float
            Composite impact in roughly [-1, 1]. Positive means the loop helped.
        """
        confidence_delta = float(after_metrics.get("avg_confidence", 0.0)) - float(
            before_metrics.get("avg_confidence", 0.0)
        )
        reward_delta = float(after_metrics.get("avg_reward", 0.0)) - float(
            before_metrics.get("avg_reward", 0.0)
        )
        weight_shift = float(after_metrics.get("weight_shift", 0.0))

        # Composite score. The weight_shift term is a tiny tie-breaker:
        # of two loops with the same quality delta, the one that moved the
        # policy *less* is preferred (Occam's razor — same gain for less
        # disruption). We cap it so it can never dominate the quality signal.
        quality = _CONFIDENCE_WEIGHT * confidence_delta + _REWARD_WEIGHT * reward_delta
        parsimony_bonus = -0.05 * math.tanh(weight_shift)
        score = quality + parsimony_bonus

        impact = LoopImpact(
            loop_name=loop_name,
            timestamp=time.time(),
            confidence_delta=confidence_delta,
            reward_delta=reward_delta,
            weight_shift=weight_shift,
            score=score,
        )
        self._persist_impact(impact)
        logger.debug(
            "Measured impact for loop=%s score=%.4f (conf_d=%.4f rew_d=%.4f shift=%.4f)",
            loop_name,
            score,
            confidence_delta,
            reward_delta,
            weight_shift,
        )
        return score

    async def measure_loop_impact_async(
        self,
        loop_name: str,
        before_metrics: Mapping[str, Any],
        after_metrics: Mapping[str, Any],
    ) -> float:
        """Async wrapper around :meth:`measure_loop_impact`.

        WHY a wrapper instead of native async?
            sqlite3 is synchronous and we want to keep the event loop
            non-blocking. ``asyncio.to_thread`` is the cheapest path.
        """
        return await asyncio.to_thread(
            self.measure_loop_impact, loop_name, before_metrics, after_metrics
        )

    def _persist_impact(self, impact: LoopImpact) -> None:
        """Append one impact event. Synchronous; called from to_thread."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO loop_impacts
                    (loop_name, ts, confidence_delta, reward_delta, weight_shift, score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                impact.as_row(),
            )
            conn.commit()

    # -- priorities -------------------------------------------------------

    def get_loop_priorities(self) -> dict[str, float]:
        """Return current normalized priorities.

        WHY normalized (sum=1.0)?
            The orchestrator divides a fixed compute budget across loops.
            Probabilities compose cleanly with whatever budget unit is in use
            (tokens, GPU-seconds, wall-clock minutes).

        WHY a dict not a list?
            New loops can be added without re-numbering; consumers look up
            by name.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT loop_name, priority FROM loop_priorities"
            ).fetchall()

        if not rows:
            logger.debug("No priorities stored yet; returning empty dict")
            return {}

        priorities = {row["loop_name"]: float(row["priority"]) for row in rows}
        return _normalize(priorities)

    async def get_loop_priorities_async(self) -> dict[str, float]:
        """Async wrapper around :meth:`get_loop_priorities`."""
        return await asyncio.to_thread(self.get_loop_priorities)

    # -- weekly evaluation -----------------------------------------------

    def weekly_evaluate(self) -> dict[str, float]:
        """Re-compute and persist priorities from recent impact history.

        WHY EMA over a hard window cutoff?
            A pure last-N average produces step changes when the oldest sample
            drops out of the window. EMA gives smoothly-decaying influence,
            which produces stabler compute allocation week-over-week.

        Returns the freshly-published priority vector.
        """
        with self._connect() as conn:
            loop_names = [
                row["loop_name"]
                for row in conn.execute(
                    "SELECT DISTINCT loop_name FROM loop_impacts"
                ).fetchall()
            ]
            if not loop_names:
                logger.info("weekly_evaluate: no impact data yet; nothing to update")
                conn.execute(
                    "INSERT INTO meta_runs (ts, loops_evaluated, notes) VALUES (?, ?, ?)",
                    (time.time(), 0, "no-data"),
                )
                conn.commit()
                return {}

            raw_scores: dict[str, float] = {}
            for name in loop_names:
                recent = conn.execute(
                    """
                    SELECT score FROM loop_impacts
                    WHERE loop_name = ?
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    (name, _PRIORITY_WINDOW),
                ).fetchall()
                scores = [float(r["score"]) for r in recent]
                # Center each score at 0.5 so a "neutral" loop still gets some
                # mass; otherwise loops with score=0 collapse to floor only.
                raw_scores[name] = max(0.0, 0.5 + _mean(scores))

            previous = {
                row["loop_name"]: float(row["priority"])
                for row in conn.execute(
                    "SELECT loop_name, priority FROM loop_priorities"
                ).fetchall()
            }

            # EMA: new = alpha * raw + (1-alpha) * prev. New loops have prev=0
            # so they start from their raw score, which is what we want.
            blended = {
                name: _EMA_ALPHA * raw_scores[name]
                + (1.0 - _EMA_ALPHA) * previous.get(name, raw_scores[name])
                for name in raw_scores
            }
            normalized = _normalize_with_floor(blended, floor=_PRIORITY_FLOOR)

            now = time.time()
            conn.executemany(
                """
                INSERT INTO loop_priorities (loop_name, priority, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(loop_name) DO UPDATE SET
                    priority = excluded.priority,
                    updated_at = excluded.updated_at
                """,
                [(name, prio, now) for name, prio in normalized.items()],
            )
            conn.execute(
                "INSERT INTO meta_runs (ts, loops_evaluated, notes) VALUES (?, ?, ?)",
                (now, len(normalized), json.dumps(normalized, sort_keys=True)),
            )
            conn.commit()

        logger.info(
            "weekly_evaluate: published priorities for %d loops: %s",
            len(normalized),
            normalized,
        )
        return normalized

    async def weekly_evaluate_async(self) -> dict[str, float]:
        """Async wrapper around :meth:`weekly_evaluate`."""
        return await asyncio.to_thread(self.weekly_evaluate)

    # -- online per-(loop, query_class) learning --------------------------
    #
    # Design (validated by Quorum consensus, 92% confidence):
    #   * ONLINE updates (not nightly batch) so a meta-learner reading the
    #     table mid-day sees the freshest evidence. We aggregate into sums +
    #     counts on each call, so cost is O(1) per loop per query.
    #   * Signal = downstream confidence_delta. Loop-fired binary alone is
    #     too coarse (misses subtle degradation); a static "suggested-weight"
    #     is brittle. The delta against the per-class rolling baseline
    #     captures whether enabling this loop actually moved the needle.
    #   * Cold start = enable ALL loops. With zero history we don't know
    #     which loops help; excluding any would create a self-fulfilling
    #     prophecy ("loop X never fires → never learns it helps → never
    #     fires"). Exploration > exploitation until samples accumulate.

    def observe(
        self,
        query_class: str,
        loops_fired: Mapping[str, bool],
        final_confidence: float,
    ) -> None:
        """Record one consensus round's loop firings and final confidence.

        Computes confidence_delta = final_confidence - prior_class_baseline,
        attributes it to every loop that fired this round (positive credit
        when the loop helped beat baseline, negative when it didn't), then
        folds final_confidence into the class baseline for the *next* call.

        Sync version; the async wrapper schedules this on a worker thread.
        """
        if not isinstance(loops_fired, Mapping):
            raise TypeError("loops_fired must be a mapping of loop_name -> bool")
        try:
            final_confidence = float(final_confidence)
        except (TypeError, ValueError) as e:
            raise TypeError(f"final_confidence must be float-like: {e}") from e

        now = time.time()
        with self._connect() as conn:
            # 1) Read current baseline so the delta is against the pre-update
            #    mean (otherwise the loop "explains" its own contribution).
            row = conn.execute(
                "SELECT confidence_sum, samples FROM class_baseline "
                "WHERE query_class = ?",
                (query_class,),
            ).fetchone()
            if row and row["samples"] > 0:
                baseline = float(row["confidence_sum"]) / int(row["samples"])
            else:
                # No history yet → delta is 0 so we only accumulate the
                # "fired" count and let the baseline catch up over time.
                baseline = final_confidence
            delta = final_confidence - baseline

            # 2) Upsert per-(loop, class) effect rows. Only loops that *fired*
            #    get credit/blame; unfired loops would muddy the signal
            #    (we can't know what they would have done).
            for loop_name, fired in loops_fired.items():
                if not fired:
                    continue
                conn.execute(
                    """
                    INSERT INTO loop_effect (
                        loop_name, query_class, fired_count,
                        confidence_delta_sum, samples,
                        last_confidence, updated_at
                    )
                    VALUES (?, ?, 1, ?, 1, ?, ?)
                    ON CONFLICT(loop_name, query_class) DO UPDATE SET
                        fired_count = fired_count + 1,
                        confidence_delta_sum = confidence_delta_sum + excluded.confidence_delta_sum,
                        samples = samples + 1,
                        last_confidence = excluded.last_confidence,
                        updated_at = excluded.updated_at
                    """,
                    (loop_name, query_class, delta, final_confidence, now),
                )

            # 3) Update the class baseline AFTER we've used the prior value.
            conn.execute(
                """
                INSERT INTO class_baseline (
                    query_class, confidence_sum, samples, updated_at
                ) VALUES (?, ?, 1, ?)
                ON CONFLICT(query_class) DO UPDATE SET
                    confidence_sum = confidence_sum + excluded.confidence_sum,
                    samples = samples + 1,
                    updated_at = excluded.updated_at
                """,
                (query_class, final_confidence, now),
            )
            conn.commit()

        logger.debug(
            "meta.observe class=%s delta=%.4f loops=%s",
            query_class,
            delta,
            [n for n, f in loops_fired.items() if f],
        )

    async def observe_async(
        self,
        query_class: str,
        loops_fired: Mapping[str, bool],
        final_confidence: float,
    ) -> None:
        """Async wrapper; keeps SQLite off the event loop."""
        await asyncio.to_thread(
            self.observe, query_class, loops_fired, final_confidence
        )

    def recommend_loops(
        self,
        query_class: str,
        candidate_loops: Iterable[str] | None = None,
    ) -> list[str]:
        """Return the set of loops worth enabling for ``query_class``.

        Policy (per Quorum consensus design):
          * Cold start: if no history at all for the class, enable every
            candidate loop (or return [] when no candidates were passed,
            signalling "no opinion — caller should default to all").
          * For each known loop:
              - keep it if mean confidence_delta > threshold;
              - keep it if samples < ``_MIN_SAMPLES_FOR_EXCLUSION``
                (exploration phase — not enough evidence to drop);
              - otherwise drop it.
          * Unknown loops in ``candidate_loops`` are always kept (zero
            evidence ⇒ exploration), so newly-added loops bootstrap into
            the rotation automatically.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT loop_name, samples, confidence_delta_sum
                FROM loop_effect
                WHERE query_class = ?
                """,
                (query_class,),
            ).fetchall()

        learned: dict[str, tuple[int, float]] = {
            r["loop_name"]: (
                int(r["samples"]),
                float(r["confidence_delta_sum"]),
            )
            for r in rows
        }

        # No evidence for this class yet → cold start.
        if not learned and candidate_loops is None:
            return []
        if not learned and candidate_loops is not None:
            return list(dict.fromkeys(candidate_loops))  # dedup, preserve order

        keepers: list[str] = []
        # Iterate over union of learned + candidates so order is stable and
        # newly-introduced loops still appear even with zero rows.
        if candidate_loops is None:
            universe: list[str] = list(learned.keys())
        else:
            seen: set[str] = set()
            universe = []
            for name in list(candidate_loops) + list(learned.keys()):
                if name not in seen:
                    seen.add(name)
                    universe.append(name)

        for name in universe:
            if name not in learned:
                # Candidate with no evidence yet — explore.
                keepers.append(name)
                continue
            samples, delta_sum = learned[name]
            if samples < _MIN_SAMPLES_FOR_EXCLUSION:
                keepers.append(name)
                continue
            mean_delta = delta_sum / samples if samples > 0 else 0.0
            if mean_delta > _EXCLUSION_DELTA_THRESHOLD:
                keepers.append(name)
            # else: enough samples AND mean delta is non-positive → drop.

        return keepers

    async def recommend_loops_async(
        self,
        query_class: str,
        candidate_loops: Iterable[str] | None = None,
    ) -> list[str]:
        """Async wrapper around :meth:`recommend_loops`."""
        return await asyncio.to_thread(
            self.recommend_loops, query_class, candidate_loops
        )

    # -- introspection (handy for dashboards / tests) ---------------------

    def recent_impacts(self, loop_name: str, limit: int = 10) -> list[LoopImpact]:
        """Return the most recent impact events for ``loop_name``.

        Exposed for the orchestrator dashboard and for assertions in tests.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT loop_name, ts, confidence_delta, reward_delta,
                       weight_shift, score
                FROM loop_impacts
                WHERE loop_name = ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (loop_name, limit),
            ).fetchall()
        return [
            LoopImpact(
                loop_name=r["loop_name"],
                timestamp=float(r["ts"]),
                confidence_delta=float(r["confidence_delta"]),
                reward_delta=float(r["reward_delta"]),
                weight_shift=float(r["weight_shift"]),
                score=float(r["score"]),
            )
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(xs: Iterable[float]) -> float:
    """Mean with safe default for empty sequences.

    Statistics.mean raises on empty input; in our context empty means
    "no signal yet" which should map to 0.0, not an exception.
    """
    xs = list(xs)
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def _normalize(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize so values sum to 1.0; uniform fallback if all zero."""
    total = sum(weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights} if n else {}
    return {k: v / total for k, v in weights.items()}


def _normalize_with_floor(
    weights: Mapping[str, float], *, floor: float
) -> dict[str, float]:
    """Normalize, then enforce a per-key floor and re-normalize.

    WHY a floor?
        See class docstring — keeps a recently-bad loop from being permanently
        starved. A bounded minimum (~2%) still leaves ~85% of budget for the
        winners when there are 8 loops.
    """
    if not weights:
        return {}
    n = len(weights)
    if floor * n >= 1.0:
        # Floor too aggressive; degenerate to uniform.
        return {k: 1.0 / n for k in weights}

    normalized = _normalize(weights)
    floored = {k: max(v, floor) for k, v in normalized.items()}
    return _normalize(floored)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


async def _smoke_test_basic(tmp_db: Path) -> None:
    """Round-trip: ingest impacts, evaluate, read priorities back.

    Verifies the happy path: a clearly-winning loop gets a clearly-higher
    priority than a clearly-losing one, while both stay above the floor.
    """
    learner = MetaLearner(db_path=tmp_db)

    winner_before = {"avg_confidence": 0.50, "avg_reward": 0.10, "weight_shift": 0.1}
    winner_after = {"avg_confidence": 0.80, "avg_reward": 0.40, "weight_shift": 0.1}
    loser_before = {"avg_confidence": 0.60, "avg_reward": 0.20, "weight_shift": 0.1}
    loser_after = {"avg_confidence": 0.55, "avg_reward": 0.10, "weight_shift": 0.5}

    for _ in range(5):
        await learner.measure_loop_impact_async("rlhf", winner_before, winner_after)
        await learner.measure_loop_impact_async(
            "prompt_mutation", loser_before, loser_after
        )

    priorities = await learner.weekly_evaluate_async()
    assert set(priorities) == {"rlhf", "prompt_mutation"}, priorities
    assert abs(sum(priorities.values()) - 1.0) < 1e-6, priorities
    assert priorities["rlhf"] > priorities["prompt_mutation"], priorities
    assert priorities["prompt_mutation"] >= _PRIORITY_FLOOR, priorities

    read_back = await learner.get_loop_priorities_async()
    assert read_back == priorities, (read_back, priorities)
    logger.info("smoke_test_basic OK: %s", priorities)


async def _smoke_test_empty(tmp_db: Path) -> None:
    """A fresh DB returns empty priorities and weekly_evaluate is a no-op."""
    learner = MetaLearner(db_path=tmp_db)
    assert await learner.get_loop_priorities_async() == {}
    assert await learner.weekly_evaluate_async() == {}
    logger.info("smoke_test_empty OK")


async def _run_smoke_tests() -> None:
    """Driver invoked by the __main__ guard."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        await _smoke_test_empty(Path(td) / "empty.db")
        await _smoke_test_basic(Path(td) / "basic.db")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_run_smoke_tests())
