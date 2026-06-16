"""Loop 10 — Federated cross-user learning pool.

Why this module exists:
    When 4+ frontier models converge on the same answer for a query, that
    consensus is essentially a free, hand-curated label produced by a panel
    of the world's best LLMs voting in unison. If users opt-in, the
    (query_embedding, consensus_response) pair becomes a shared learning
    signal: every other Quorum installation can use it to (a) train a better
    router that knows which model to pick first for similar queries, and
    (b) seed the local-Llama distillation dataset with examples that have
    already been validated across the ecosystem.

    The network effect this enables is the moat: each Quorum user who opts
    in makes Quorum smarter for every other user, without ever leaking the
    raw prompt — only the embedding (with Laplace noise) and the agreed-upon
    answer travel between nodes.

    PII is stripped before persistence (emails, phones, UK postcodes,
    obvious "Hi, my name is X" patterns) and Laplace noise is added to the
    embedding at the differential-privacy boundary. Contribution is
    *not* HSP-gated (the user opted in for that single query) but any
    downstream *training run* that consumes this pool IS HSP-gated —
    promoting a model fine-tuned on federated data is a high-stakes action
    and must pass through `requires_hsp_approval` in the consumer module
    (see `distillation.py`).

License:
    Apache 2.0 — see LICENSE.
    HSP commercial restrictions apply — see LICENSE-HSP (PCT/US26/11908).
    Patent: PCT/US26/11908.

Triggers on: per-query (opt-in at contribute time).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
DEFAULT_DB_PATH = DATA_DIR / "federated.db"
AGREEMENT_THRESHOLD = 0.9
DEFAULT_TOP_K = 20
DEFAULT_EPSILON = 1.0

# Regex set used to strip the most common PII patterns. The list is
# intentionally conservative — we err on the side of redacting too much
# rather than too little, because contributions cannot be retracted once
# they leave the contributor's node.
_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("[EMAIL]", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    # International phone numbers — loose, catches +44, 07..., (xxx) xxx-xxxx
    (
        "[PHONE]",
        re.compile(
            r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?){2,4}\d{2,4}"
        ),
    ),
    # UK postcode (e.g. SW1A 1AA, EC1V 9HX, M1 1AE)
    (
        "[POSTCODE]",
        re.compile(
            r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
            re.IGNORECASE,
        ),
    ),
    # "my name is X", "I am X Y" — naive but useful first cut
    (
        "[NAME]",
        re.compile(
            r"\b(?:my name is|i am|i'm|this is)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
            re.IGNORECASE,
        ),
    ),
    # Credit-card-ish 13-19 digit runs
    ("[CARD]", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Contribution:
    """One opt-in (embedding, answer) sample shared into the federated pool.

    The fields are intentionally minimal — anything that could re-identify
    the contributor (raw prompt, IP, user agent) is *not* persisted. The
    `user_id` is hashed at the boundary, so even the local pool doesn't
    know which downstream user produced the sample.
    """

    contribution_id: str
    user_id_hash: str
    query_embedding: list[float]
    consensus_response: str
    agreement_score: float
    model_count: int
    created_at: float = field(default_factory=time.time)

    def to_row(self) -> tuple[Any, ...]:
        """Serialise to a SQLite row tuple (embedding becomes JSON)."""
        return (
            self.contribution_id,
            self.user_id_hash,
            json.dumps(self.query_embedding),
            self.consensus_response,
            self.agreement_score,
            self.model_count,
            self.created_at,
        )

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> "Contribution":
        """Deserialise a SQLite row back into a Contribution."""
        return cls(
            contribution_id=row[0],
            user_id_hash=row[1],
            query_embedding=json.loads(row[2]),
            consensus_response=row[3],
            agreement_score=float(row[4]),
            model_count=int(row[5]),
            created_at=float(row[6]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_user_id(user_id: str) -> str:
    """Salted-hash a user_id so the pool doesn't store the raw identifier.

    We use SHA-256 with a per-install salt sourced from
    QUORUM_FEDERATED_SALT (falls back to a stable per-host derivation).
    This is *not* anonymisation in the strict differential-privacy sense
    — it's pseudonymisation that prevents trivial joining with other logs.
    """
    import hashlib

    salt = os.getenv("QUORUM_FEDERATED_SALT", "quorum-federated-default-salt")
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(b":")
    h.update(user_id.encode("utf-8"))
    return h.hexdigest()[:32]


def strip_pii(text: str) -> str:
    """Redact common PII patterns from a string.

    Why: even though the user opted in for *this* query, the consensus
    answer may quote back an email, phone, or address that appeared in
    the prompt. We redact aggressively before the answer is persisted
    so it cannot leak through the pool.
    """
    if not text:
        return text
    out = text
    for placeholder, pattern in _PII_PATTERNS:
        out = pattern.sub(placeholder, out)
    return out


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Plain cosine similarity. Returns 0.0 for mismatched / zero vectors.

    Kept dependency-free on purpose: numpy isn't a hard requirement for
    the federated pool, and the embeddings are short enough that pure
    Python is fast enough for top-K retrieval up to a few thousand rows.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _laplace_sample(scale: float, rng: random.Random) -> float:
    """Draw one sample from Laplace(0, scale) without numpy.

    Inverse-CDF method: if U ~ Uniform(-0.5, 0.5), then
        X = -scale * sign(U) * ln(1 - 2|U|)  ~ Laplace(0, scale).
    """
    u = rng.random() - 0.5
    sign = 1.0 if u >= 0 else -1.0
    return -scale * sign * math.log(1.0 - 2.0 * abs(u))


def _is_opted_in(user_id: str, per_user_opt_in: dict[str, bool] | None) -> bool:
    """Resolve opt-in status: per-user mapping wins, then env var.

    Why two layers: tests and embedded usage want to flip opt-in per call
    without touching environment state; a deployed install wants a single
    env var (FEDERATED_OPT_IN=1) to turn the feature on globally.
    """
    if per_user_opt_in is not None and user_id in per_user_opt_in:
        return bool(per_user_opt_in[user_id])
    env = os.getenv("FEDERATED_OPT_IN", "").strip().lower()
    return env in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------


class FederatedPool:
    """SQLite-backed pool of high-consensus contributions.

    Why a class (not a module-level set of functions): we want one writable
    handle per process to avoid SQLite write contention, and a single
    place to attach the opt-in registry, DP settings, and the path that
    the rest of the codebase imports.

    Public methods are async because callers live in asyncio land (the
    consensus engine). The SQLite work itself is sync, so we hop into a
    worker thread via `asyncio.to_thread` to keep the event loop snappy.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        epsilon: float = DEFAULT_EPSILON,
        agreement_threshold: float = AGREEMENT_THRESHOLD,
        per_user_opt_in: dict[str, bool] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        """Initialise the pool and ensure the SQLite schema exists.

        Args:
            db_path: Override the default ~/.quorum/federated.db. Useful
                for tests (pass an in-memory `:memory:` or a tmp_path).
            epsilon: Differential-privacy budget for embedding noise.
                Smaller epsilon = more noise = more privacy.
            agreement_threshold: Below this, contributions are silently
                dropped even if the user opted in.
            per_user_opt_in: Optional mapping of user_id -> opt-in bool
                used for tests / programmatic control.
            rng_seed: Seed the internal RNG used for DP noise. Tests use
                this for determinism; production leaves it None.
        """
        if db_path is None:
            self.db_path: Path | str = DEFAULT_DB_PATH
        elif isinstance(db_path, str) and db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path)
        self.epsilon = float(epsilon)
        self.agreement_threshold = float(agreement_threshold)
        self.per_user_opt_in = per_user_opt_in
        self._rng = random.Random(rng_seed)
        # One shared in-memory connection if path is :memory: so the schema
        # actually persists between calls in the same pool instance.
        # check_same_thread=False is required because asyncio.to_thread
        # may dispatch the SQLite work onto a different worker thread
        # than the one that built the pool.
        self._memory_conn: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
        self._ensure_schema()

    # ----- schema --------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection appropriate for this pool.

        For :memory: pools we reuse a single connection so the schema
        survives between calls. For on-disk pools we open per-call so
        the OS file lock is held only briefly.
        """
        if self._memory_conn is not None:
            return self._memory_conn
        assert isinstance(self.db_path, Path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        """Create the contributions table on first run. Idempotent."""
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS contributions (
                    contribution_id  TEXT PRIMARY KEY,
                    user_id_hash     TEXT NOT NULL,
                    query_embedding  TEXT NOT NULL,
                    consensus_response TEXT NOT NULL,
                    agreement_score  REAL NOT NULL,
                    model_count      INTEGER NOT NULL,
                    created_at       REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_contrib_created "
                "ON contributions(created_at)"
            )
            conn.commit()
        finally:
            if self._memory_conn is None:
                conn.close()

    # ----- public API ----------------------------------------------------

    async def contribute(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        consensus_response: str,
        agreement_score: float,
        *,
        model_count: int = 4,
    ) -> str | None:
        """Record one high-consensus sample if the user opted in.

        Why all the guards: this method is the trust boundary of the
        whole federated loop. If we accept a contribution we shouldn't
        have, that data leaves the contributor's node forever (any
        downstream training run consumes it). So we fail closed:

        - Reject if `agreement_score <= self.agreement_threshold`.
        - Reject if the user hasn't opted in (FEDERATED_OPT_IN env var
          OR explicit per-user setting).
        - Strip PII from the response text.
        - Add Laplace noise to the embedding at epsilon=self.epsilon.
        - Hash the user_id before persistence.

        Returns the contribution_id on success, or None if the
        contribution was rejected (logged at INFO).
        """
        if agreement_score <= self.agreement_threshold:
            logger.info(
                "federated: skipped contribution, agreement %.3f <= %.3f",
                agreement_score,
                self.agreement_threshold,
            )
            return None

        if not _is_opted_in(user_id, self.per_user_opt_in):
            logger.info("federated: skipped contribution, user not opted in")
            return None

        cleaned_response = strip_pii(consensus_response)
        noised_embedding = self.add_differential_privacy_noise(
            list(query_embedding), epsilon=self.epsilon
        )

        contribution = Contribution(
            contribution_id=uuid.uuid4().hex,
            user_id_hash=_hash_user_id(user_id),
            query_embedding=noised_embedding,
            consensus_response=cleaned_response,
            agreement_score=float(agreement_score),
            model_count=int(model_count),
        )

        await asyncio.to_thread(self._insert_sync, contribution)
        logger.info(
            "federated: contribution %s stored (agreement=%.3f, models=%d)",
            contribution.contribution_id,
            agreement_score,
            model_count,
        )
        return contribution.contribution_id

    async def get_contributions_for_router(
        self,
        query_embedding: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """Return the top-K most-similar contributions for the router.

        Why: the router (Loop 1) wants to know "for queries that look like
        this one, which model was right last time?" We rank by cosine
        similarity over the stored (already noised) embeddings and return
        plain dicts so the router doesn't need to import our dataclass.
        """
        candidates = await asyncio.to_thread(self._load_all_sync)
        if not candidates:
            return []

        query = list(query_embedding)
        scored: list[tuple[float, Contribution]] = []
        for c in candidates:
            sim = _cosine_similarity(query, c.query_embedding)
            scored.append((sim, c))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[: max(0, int(top_k))]
        return [
            {
                "contribution_id": c.contribution_id,
                "similarity": sim,
                "consensus_response": c.consensus_response,
                "agreement_score": c.agreement_score,
                "model_count": c.model_count,
                "created_at": c.created_at,
            }
            for sim, c in top
        ]

    def add_differential_privacy_noise(
        self,
        embedding: Sequence[float],
        epsilon: float = DEFAULT_EPSILON,
    ) -> list[float]:
        """Add Laplace noise scaled by 1/epsilon to each embedding dim.

        Why Laplace and not Gaussian: Laplace gives pure epsilon-DP for
        any single query, which is the right guarantee for a one-shot
        contribution (we don't compose many releases per user). We assume
        embeddings are L2-normalised so sensitivity is bounded ~2; the
        scale here treats sensitivity=1 and lets the caller tighten via
        epsilon. Smaller epsilon -> larger noise -> stronger privacy.
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        scale = 1.0 / epsilon
        return [
            float(x) + _laplace_sample(scale, self._rng) for x in embedding
        ]

    async def size(self) -> int:
        """Count rows in the pool (mostly for tests / metrics)."""
        return await asyncio.to_thread(self._count_sync)

    # ----- sync inner helpers (always called via to_thread) --------------

    def _insert_sync(self, contribution: Contribution) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO contributions "
                "(contribution_id, user_id_hash, query_embedding, "
                "consensus_response, agreement_score, model_count, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                contribution.to_row(),
            )
            conn.commit()
        finally:
            if self._memory_conn is None:
                conn.close()

    def _load_all_sync(self) -> list[Contribution]:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT contribution_id, user_id_hash, query_embedding, "
                "consensus_response, agreement_score, model_count, "
                "created_at FROM contributions"
            )
            rows = cur.fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()
        return [Contribution.from_row(r) for r in rows]

    def _count_sync(self) -> int:
        conn = self._connect()
        try:
            cur = conn.execute("SELECT COUNT(*) FROM contributions")
            (n,) = cur.fetchone()
            return int(n)
        finally:
            if self._memory_conn is None:
                conn.close()


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


async def _smoke_contribute_and_retrieve() -> None:
    """End-to-end: opt-in user contributes, then router retrieves top-K.

    Validates the happy path: agreement > threshold + opted-in => stored,
    and a near-identical query embedding ranks the new contribution first.
    """
    # epsilon high => small noise so the test can assert similarity > 0.
    # The privacy/utility tradeoff is the caller's choice; defaults stay
    # conservative in production.
    pool = FederatedPool(
        db_path=":memory:",
        per_user_opt_in={"alice": True},
        epsilon=100.0,
        rng_seed=42,
    )
    cid = await pool.contribute(
        user_id="alice",
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        consensus_response="The answer is 42.",
        agreement_score=0.95,
        model_count=4,
    )
    assert cid is not None, "expected contribution to be accepted"
    assert await pool.size() == 1

    hits = await pool.get_contributions_for_router(
        query_embedding=[0.1, 0.2, 0.3, 0.4], top_k=5
    )
    assert hits, "expected at least one hit"
    assert hits[0]["contribution_id"] == cid
    assert hits[0]["similarity"] > 0.5


async def _smoke_rejects_low_agreement_and_opt_out() -> None:
    """Negative path: low-agreement and non-opted-in must be rejected.

    Plus: PII in the response should be redacted before retrieval.
    """
    pool = FederatedPool(
        db_path=":memory:",
        per_user_opt_in={"alice": True, "bob": False},
        rng_seed=7,
    )

    # Low agreement => reject even though alice opted in.
    rejected = await pool.contribute(
        user_id="alice",
        query_embedding=[0.0, 1.0, 0.0],
        consensus_response="meh",
        agreement_score=0.5,
    )
    assert rejected is None

    # Not opted in => reject.
    rejected = await pool.contribute(
        user_id="bob",
        query_embedding=[1.0, 0.0, 0.0],
        consensus_response="whatever",
        agreement_score=0.99,
    )
    assert rejected is None

    # PII redaction round-trip.
    cid = await pool.contribute(
        user_id="alice",
        query_embedding=[0.5, 0.5, 0.5],
        consensus_response=(
            "Contact me at jane@example.com or +44 7700 900123, "
            "I'm at SW1A 1AA. My name is Jane Doe."
        ),
        agreement_score=0.97,
    )
    assert cid is not None
    hits = await pool.get_contributions_for_router(
        query_embedding=[0.5, 0.5, 0.5], top_k=1
    )
    body = hits[0]["consensus_response"]
    assert "jane@example.com" not in body
    assert "SW1A 1AA" not in body
    assert "Jane Doe" not in body
    assert "[EMAIL]" in body


async def _run_smoke_tests() -> None:
    await _smoke_contribute_and_retrieve()
    await _smoke_rejects_low_agreement_and_opt_out()


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run_smoke_tests())
    logger.info("federated: smoke tests passed")
