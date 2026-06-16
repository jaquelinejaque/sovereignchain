"""Hebbian co-activation matrix for Quorum consensus weighting.

Loop 2 of the Quorum evolution system.

Core idea (paraphrased Donald Hebb, 1949): "neurons that fire together wire
together." Translated to a multi-LLM consensus engine: models that consistently
produce similar answers on rewarded queries are more likely to be trustworthy
*together* than apart. We track that pair-trust in a persistent matrix and
expose it as a multiplier (1.0 -> 1.5) that the consensus engine can apply when
weighting individual votes.

Why a persistent SQLite store (not in-memory)?
    - Quorum runs as ephemeral async workers; we need correlations to survive
      process restarts.
    - The matrix is sparse and small (O(n_models^2)), so a single-file SQLite
      DB at ~/.quorum/hebbian.db is plenty and avoids a service dependency.
    - SQLite calls are synchronous; we wrap each one in ``asyncio.to_thread``
      so the event loop is never blocked.

Why a decay step?
    - Without decay, two collusive models that agreed a year ago would keep
      inflating each other's weight forever. ``decay(half_life_days=30)``
      should be called nightly (Loop 5/6 cron) to half stale similarity sums.

Trigger: real-time after each consensus round. No HSP gate (this loop is
purely statistical bookkeeping, not patentable optimization).

License
-------
Copyright 2026 Sovereign Chain Ltd.
Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quorum.providers.base import ModelResponse

logger = logging.getLogger(__name__)

# -- Tunables ---------------------------------------------------------------

#: Similarity threshold above which two models are considered "co-firing"
#: in a given round. 0.75 matches the disagreement threshold the consensus
#: engine itself uses elsewhere; keep them in sync.
SIMILARITY_THRESHOLD: float = 0.75

#: Base learning rate. Each rewarded co-firing increases pair similarity_sum
#: by ``LEARNING_RATE * reward``. Kept small so a few unlucky alignments
#: don't dominate the matrix.
LEARNING_RATE: float = 0.01

#: Boost multiplier ceiling. Pairs with very high average similarity get
#: this much extra weight in consensus. 1.5 is deliberately gentle: we want
#: to *nudge* the engine, not let two friendly models swamp a dissenter.
MAX_BOOST: float = 1.5
MIN_BOOST: float = 1.0


# -- Helpers ----------------------------------------------------------------


DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()


def _default_db_path() -> Path:
    """Resolve the default SQLite path.

    We honour ``QUORUM_HEBBIAN_DB`` for tests/CI so we never accidentally
    pollute the user's real matrix during a unit test.
    """
    override = os.environ.get("QUORUM_HEBBIAN_DB")
    if override:
        return Path(override)
    return DATA_DIR / "hebbian.db"


def _canonical_pair(model_a: str, model_b: str) -> tuple[str, str]:
    """Order a pair lexicographically.

    The matrix is symmetric (similarity(a, b) == similarity(b, a)), so we
    store each pair exactly once and look it up under a deterministic key.
    Saves storage and removes a class of "why is the boost different both
    ways" bugs.
    """
    return (model_a, model_b) if model_a <= model_b else (model_b, model_a)


# -- Data class -------------------------------------------------------------


@dataclass(frozen=True)
class PairStat:
    """A single row from the coactivation table, convenient for analytics."""

    model_a: str
    model_b: str
    similarity_sum: float
    count: int
    last_updated: float

    @property
    def mean_similarity(self) -> float:
        """Average similarity across all rewarded co-firings.

        Used by ``get_pair_boost`` and the dashboard. Returns 0.0 when
        ``count == 0`` so callers don't have to special-case division by
        zero.
        """
        return self.similarity_sum / self.count if self.count else 0.0


# -- Main class -------------------------------------------------------------


class HebbianMatrix:
    """Persistent symmetric pair-trust matrix between LLM providers.

    Why a class (not a module of free functions)?
        - We want to allow swapping DB paths for tests, batch runs, and
          per-tenant matrices without monkey-patching globals.
        - Connection lifecycle and decay state are easier to reason about
          when scoped to an instance.

    Thread safety
        SQLite has its own locking; we additionally serialize *writes*
        through a single asyncio.Lock so two concurrent consensus rounds
        can't race-update the same pair counter.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path: Path = Path(db_path) if db_path else _default_db_path()
        self._write_lock = asyncio.Lock()
        self._ensure_schema()

    # ---- schema --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived SQLite connection.

        We open per call (and close in the caller) so each ``to_thread``
        invocation owns its own connection. Avoids "SQLite objects created
        in a thread can only be used in that same thread" issues entirely.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self) -> None:
        """Create the table on first use.

        Idempotent — safe to call on every process boot.
        """
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS coactivation (
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    similarity_sum REAL NOT NULL DEFAULT 0.0,
                    count INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL NOT NULL DEFAULT 0.0,
                    PRIMARY KEY (model_a, model_b)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_coactivation_strength
                ON coactivation (similarity_sum DESC)
                """
            )
            conn.commit()
        logger.debug("Hebbian schema ready at %s", self.db_path)

    # ---- public API ----------------------------------------------------

    async def record_round(
        self,
        responses: list[ModelResponse],
        pairwise_similarities: dict[tuple[str, str], float],
        reward: float,
    ) -> int:
        """Update the matrix from one consensus round.

        Why ``reward`` is a parameter (not always 1.0): downstream feedback
        loops (Loop 3/Bayesian, Loop 8/cost-aware) compute a scalar reward
        for the *whole round* — e.g., human thumbs-up, downstream click,
        bug-bounty triage hit. We multiply the learning delta by it so
        unrewarded agreement doesn't accumulate.

        We only learn when:
            * ``reward > 0`` (no negative reinforcement — that's Loop 3's job)
            * pair similarity strictly exceeds ``SIMILARITY_THRESHOLD``
            * neither model errored on this round (errored responses have no
              meaningful "vote" to align on)

        Returns:
            Number of pairs whose stats were updated. Useful for tests and
            for surfacing "learning happened" in CLI/dashboards.
        """
        if reward <= 0.0:
            logger.debug("record_round skipped: reward=%.3f <= 0", reward)
            return 0

        # Build a set of valid (non-errored) model names so we can ignore
        # similarity entries that involve a failed call.
        valid_names = {r.name for r in responses if not r.error}
        delta_base = LEARNING_RATE * reward
        now = time.time()

        updates: list[tuple[str, str, float, float]] = []
        for (a_raw, b_raw), sim in pairwise_similarities.items():
            if a_raw == b_raw:
                continue
            if sim <= SIMILARITY_THRESHOLD:
                continue
            if a_raw not in valid_names or b_raw not in valid_names:
                continue
            a, b = _canonical_pair(a_raw, b_raw)
            updates.append((a, b, sim, delta_base))

        if not updates:
            return 0

        async with self._write_lock:
            updated = await asyncio.to_thread(self._apply_updates, updates, now)
        logger.info(
            "Hebbian: updated %d pair(s), reward=%.3f, sample=%s",
            updated,
            reward,
            updates[0][:2] if updates else None,
        )
        return updated

    def _apply_updates(
        self,
        updates: list[tuple[str, str, float, float]],
        now: float,
    ) -> int:
        """Synchronous DB upsert. Called via to_thread from record_round."""
        with self._connect() as conn:
            for a, b, sim, delta in updates:
                # UPSERT: insert with sim as sum+1 count, else add delta and
                # bump count. We weight by ``sim`` so a 0.95 alignment moves
                # the average more than a borderline 0.76.
                conn.execute(
                    """
                    INSERT INTO coactivation
                        (model_a, model_b, similarity_sum, count, last_updated)
                    VALUES (?, ?, ?, 1, ?)
                    ON CONFLICT(model_a, model_b) DO UPDATE SET
                        similarity_sum = similarity_sum + excluded.similarity_sum * ?,
                        count          = count + 1,
                        last_updated   = excluded.last_updated
                    """,
                    (a, b, sim, now, delta / LEARNING_RATE if LEARNING_RATE else 0.0),
                )
            conn.commit()
        return len(updates)

    async def get_pair_boost(self, model_a: str, model_b: str) -> float:
        """Return a [1.0, 1.5] multiplier for the given model pair.

        Why this shape, not raw mean similarity?
            The consensus engine multiplies model weights by this value.
            Anything < 1.0 would *demote* a model just because it lacks
            history with a partner, which would penalize new providers and
            create a lock-in effect. We therefore clamp to >= 1.0 and let
            the **upper** end reward consistent alignment.

        Boost formula:
            boost = 1 + (max - 1) * tanh(mean_sim * sqrt(count) / 4)

        - ``mean_sim``: average similarity across rewarded rounds (0..1).
        - ``sqrt(count)``: confidence — a pair with 100 rewarded rounds
          gets weighted more than one with 2.
        - ``tanh`` saturates so we never exceed ``MAX_BOOST``.
        """
        if model_a == model_b:
            return MIN_BOOST
        a, b = _canonical_pair(model_a, model_b)
        row = await asyncio.to_thread(self._fetch_pair, a, b)
        if row is None or row.count == 0:
            return MIN_BOOST
        score = row.mean_similarity * math.sqrt(row.count) / 4.0
        boost = MIN_BOOST + (MAX_BOOST - MIN_BOOST) * math.tanh(score)
        return max(MIN_BOOST, min(MAX_BOOST, boost))

    def _fetch_pair(self, a: str, b: str) -> PairStat | None:
        """Read one pair row. Synchronous; called via to_thread."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT model_a, model_b, similarity_sum, count, last_updated "
                "FROM coactivation WHERE model_a=? AND model_b=?",
                (a, b),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return PairStat(
            model_a=row[0],
            model_b=row[1],
            similarity_sum=row[2],
            count=row[3],
            last_updated=row[4],
        )

    async def get_strongest_pairs(self, top_k: int = 5) -> list[tuple[str, str, float]]:
        """Return the ``top_k`` pairs by *mean* similarity.

        Why mean and not raw similarity_sum?
            similarity_sum grows monotonically with count, so the leaderboard
            would just rank the most-used pairs. Mean lets the dashboard
            surface genuinely well-aligned pairs even when one is new.

        Used by the analytics dashboard and the nightly ops digest.
        """
        rows = await asyncio.to_thread(self._fetch_all_pairs)
        ranked = sorted(
            ((p.model_a, p.model_b, p.mean_similarity) for p in rows if p.count > 0),
            key=lambda t: t[2],
            reverse=True,
        )
        return ranked[: max(0, top_k)]

    def _fetch_all_pairs(self) -> list[PairStat]:
        """Read every row. Cheap because the matrix is O(n_models^2)."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT model_a, model_b, similarity_sum, count, last_updated "
                "FROM coactivation"
            )
            rows = cur.fetchall()
        return [
            PairStat(
                model_a=r[0],
                model_b=r[1],
                similarity_sum=r[2],
                count=r[3],
                last_updated=r[4],
            )
            for r in rows
        ]

    async def decay(self, half_life_days: float = 30.0) -> int:
        """Exponentially decay stale entries.

        Why decay and not eviction?
            Eviction loses the "we used to agree" signal entirely. Decay
            keeps long-lived collaborators visible while letting one-off
            alignments fade naturally.

        Math:
            For each pair, scale similarity_sum by:
                factor = 0.5 ** (days_since_update / half_life_days)
            Count is scaled similarly so mean_similarity is preserved while
            confidence (count) decays — exactly the behaviour we want for
            the boost formula above.

        Trigger: nightly (Loop 5/6 cron). Returns the number of rows
        touched, for logging.
        """
        if half_life_days <= 0:
            raise ValueError("half_life_days must be > 0")
        now = time.time()
        touched = await asyncio.to_thread(self._apply_decay, now, half_life_days)
        logger.info("Hebbian decay: %d rows scaled (half_life=%.1f days)", touched, half_life_days)
        return touched

    def _apply_decay(self, now: float, half_life_days: float) -> int:
        """Synchronous decay step.

        We iterate in Python rather than in SQL because computing
        ``0.5 ** (delta/H)`` per row is awkward in standard SQLite and the
        table is small enough that it doesn't matter.

        Rows whose count would round to zero after decay are deleted; they
        carry no information and would otherwise clutter the leaderboard
        forever.
        """
        seconds_per_day = 86400.0
        touched = 0
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT model_a, model_b, similarity_sum, count, last_updated "
                "FROM coactivation"
            ).fetchall()
            for model_a, model_b, sim_sum, count, last_updated in rows:
                age_days = max(0.0, (now - last_updated) / seconds_per_day)
                if age_days == 0.0:
                    continue
                factor = 0.5 ** (age_days / half_life_days)
                new_sum = sim_sum * factor
                # Keep count as an int >= 0; if a pair decays below 1 we
                # consider it forgotten and delete it.
                new_count = int(round(count * factor))
                if new_count <= 0:
                    conn.execute(
                        "DELETE FROM coactivation WHERE model_a=? AND model_b=?",
                        (model_a, model_b),
                    )
                else:
                    conn.execute(
                        "UPDATE coactivation SET similarity_sum=?, count=?, last_updated=? "
                        "WHERE model_a=? AND model_b=?",
                        (new_sum, new_count, now, model_a, model_b),
                    )
                touched += 1
            conn.commit()
        return touched

    async def reset(self) -> None:
        """Wipe the matrix. Intended for tests and disaster recovery only."""
        async with self._write_lock:
            await asyncio.to_thread(self._reset_sync)
        logger.warning("Hebbian matrix at %s was reset", self.db_path)

    def _reset_sync(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM coactivation")
            conn.commit()


# -- Smoke tests ------------------------------------------------------------
#
# These are intentionally NOT pytest fixtures — keeping them as plain
# coroutines means CI can `python -m quorum.evolution.hebbian` and get a
# fast, dependency-free sanity check. Wire them into the project's pytest
# suite by importing and awaiting them, e.g.:
#
#     async def test_hebbian_record_round():
#         await _smoke_record_round()


async def _smoke_record_round() -> None:
    """Round-trip: record, then boost should rise above 1.0."""
    from dataclasses import dataclass as _dc

    @_dc
    class _MR:
        name: str
        response: str = ""
        error: str = ""

    tmp = Path("/tmp/hebbian_smoke_record.db")
    if tmp.exists():
        tmp.unlink()
    matrix = HebbianMatrix(db_path=tmp)

    responses = [_MR("claude"), _MR("gemini"), _MR("gpt4")]
    sims = {
        ("claude", "gemini"): 0.9,
        ("claude", "gpt4"): 0.5,  # below threshold, ignored
        ("gemini", "gpt4"): 0.8,
    }
    updated = await matrix.record_round(responses, sims, reward=1.0)  # type: ignore[arg-type]
    assert updated == 2, f"expected 2 pair updates, got {updated}"

    boost_cg = await matrix.get_pair_boost("claude", "gemini")
    boost_cgpt = await matrix.get_pair_boost("claude", "gpt4")
    assert MIN_BOOST < boost_cg <= MAX_BOOST, f"boost out of range: {boost_cg}"
    assert boost_cgpt == MIN_BOOST, f"unsimilar pair should have base boost, got {boost_cgpt}"

    # Pair order shouldn't matter.
    boost_gc = await matrix.get_pair_boost("gemini", "claude")
    assert abs(boost_cg - boost_gc) < 1e-9, "pair lookup must be symmetric"

    top = await matrix.get_strongest_pairs(top_k=5)
    assert len(top) == 2, f"expected 2 top pairs, got {len(top)}"
    assert top[0][2] >= top[1][2], "leaderboard must be sorted descending"

    tmp.unlink(missing_ok=True)


async def _smoke_decay() -> None:
    """Decay should reduce similarity_sum monotonically toward zero."""
    tmp = Path("/tmp/hebbian_smoke_decay.db")
    if tmp.exists():
        tmp.unlink()
    matrix = HebbianMatrix(db_path=tmp)

    # Seed a row "from 60 days ago" so decay actually fires.
    sixty_days_ago = time.time() - 60 * 86400
    with matrix._connect() as conn:
        conn.execute(
            "INSERT INTO coactivation VALUES (?, ?, ?, ?, ?)",
            ("a", "b", 10.0, 20, sixty_days_ago),
        )
        conn.commit()

    before = await matrix.get_pair_boost("a", "b")
    touched = await matrix.decay(half_life_days=30.0)
    assert touched == 1, f"expected 1 row touched, got {touched}"

    after = await matrix.get_pair_boost("a", "b")
    # After two half-lives, count should be ~5 and sim_sum ~2.5 -> still
    # gives some boost, but strictly less than before.
    assert after < before, f"decay should reduce boost: before={before}, after={after}"

    tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Module-level convenience wrapper for the router (Loop 4).
# ---------------------------------------------------------------------------
#
# The MoE router (quorum.evolution.router) imports `get_class_boosts` from
# this module. It wants a per-model multiplier given a query_class and a
# candidate list. We translate the pairwise stored matrix into a per-model
# score by averaging each candidate's pairwise boost against every other
# candidate in the panel: a model that historically aligns well with the
# rest of the panel gets a higher multiplier than one that doesn't.
#
# query_class is currently unused at this layer — the underlying matrix is
# global, not partitioned by class — but we accept it in the signature so
# the router doesn't have to change later when we add per-class shards.

_default_matrix: HebbianMatrix | None = None
_default_matrix_lock = asyncio.Lock()


async def _get_default_matrix() -> HebbianMatrix:
    """Return a process-wide singleton matrix.

    Why a singleton: the router calls this on every consensus(), and the
    matrix constructor opens a sqlite connection to ensure schema. Once is
    enough.
    """
    global _default_matrix
    if _default_matrix is not None:
        return _default_matrix
    async with _default_matrix_lock:
        if _default_matrix is None:
            _default_matrix = HebbianMatrix()
    return _default_matrix


async def get_class_boosts(
    query_class: str, models: "list[str]"
) -> dict[str, float]:
    """Return a {model_name: boost} mapping for the router.

    For each model in ``models`` we compute the mean pairwise boost against
    every other model in the same panel. Models that consistently co-fire
    productively with the panel get a multiplier above 1.0; lone wolves stay
    at 1.0. Never raises — falls back to neutral 1.0 on any internal error
    so the router's hot path can't be broken by a Hebbian outage.

    ``query_class`` is currently advisory (the matrix is global) but kept in
    the signature for forward compatibility with a future per-class shard.
    """
    try:
        if not models or len(models) < 2:
            return {m: MIN_BOOST for m in models}
        matrix = await _get_default_matrix()
        out: dict[str, float] = {}
        for m in models:
            others = [o for o in models if o != m]
            if not others:
                out[m] = MIN_BOOST
                continue
            pair_boosts = await asyncio.gather(
                *(matrix.get_pair_boost(m, o) for o in others)
            )
            # Average boost across the panel: a model that aligns with most
            # of the panel gets a higher score; a lone dissenter stays at 1.0.
            out[m] = sum(pair_boosts) / len(pair_boosts)
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("get_class_boosts failed (%s); returning neutral", e)
        return {m: MIN_BOOST for m in models}


__all__ = [
    "HebbianMatrix",
    "PairStat",
    "SIMILARITY_THRESHOLD",
    "LEARNING_RATE",
    "MAX_BOOST",
    "MIN_BOOST",
    "get_class_boosts",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s :: %(message)s")
    asyncio.run(_smoke_record_round())
    asyncio.run(_smoke_decay())
    logger.info("Hebbian smoke tests passed.")
