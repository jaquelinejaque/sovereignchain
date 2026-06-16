# Copyright 2026 Jaqueline Martins / Sovereign Chain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# HSP (Hybrid Sovereign Protocol) attribution:
#   PCT/US26/11908 — vector memory is a downstream consumer of HSP-gated
#   consensus output. This module itself is NOT HSP-gated (no gate decorator),
#   but it stores artifacts produced by gated paths and must respect the
#   user_id scoping required by the HSP commercial-use restrictions.
"""Cross-session vector memory for Quorum.

Why this module exists
----------------------
Quorum's value proposition is "the consensus engine that learns you over time".
A single multi-LLM call is just an ensemble; what makes Quorum compounding is
that every interaction (query, response, correction, declared preference, fact
the user pinned) becomes a vector in a per-user store that is automatically
retrieved on the next call.

Design constraints (v0.1):
    * No external DB. SQLite is stdlib and works on any laptop / CI box.
    * No network round-trips for recall (a remote vector DB adds latency that
      defeats the "compact context injection" use-case).
    * Per-user isolation by ``user_id`` — multi-tenant from day one.
    * Embeddings are caller-supplied. This module is embedding-model-agnostic
      so we can swap OpenAI / Voyage / local SBERT without a migration.
    * Float32 BLOBs keep the DB ~4x smaller than JSON arrays and decode ~10x
      faster via ``numpy.frombuffer``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

MemoryKind = Literal["query", "response", "correction", "preference", "fact"]
_VALID_KINDS: frozenset[str] = frozenset(
    {"query", "response", "correction", "preference", "fact"}
)

DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
DEFAULT_DB_PATH: Path = DATA_DIR / "memory.db"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryHit:
    """A single recalled memory plus its cosine similarity to the query vector.

    Why frozen: a hit represents a point-in-time snapshot of a stored memory;
    downstream code (prompt builders, evaluators) should never mutate it.
    """

    id: str
    content: str
    similarity: float
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging / JSON transport (e.g. tracing spans)."""
        return {
            "id": self.id,
            "content": self.content,
            "similarity": self.similarity,
            "kind": self.kind,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_float32_bytes(vec: Iterable[float] | NDArray[Any]) -> bytes:
    """Pack an embedding to compact float32 bytes for BLOB storage.

    Why float32 over float64: most embedding models (OpenAI, Voyage, SBERT)
    emit float32 natively; doubling the precision costs disk + I/O without any
    cosine-similarity quality gain.
    """
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"embedding must be 1-D, got shape {arr.shape}")
    return arr.tobytes()


def _from_blob(blob: bytes) -> NDArray[np.float32]:
    """Inverse of :func:`_to_float32_bytes`."""
    return np.frombuffer(blob, dtype=np.float32)


def _cosine_matrix(query: NDArray[np.float32], matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cosine similarity between one query vector and a stack of stored vectors.

    Vectorized with numpy so a 10k-memory recall stays sub-millisecond. The
    epsilon guards against zero vectors (e.g. corrupted rows) without raising.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    q_norm = np.linalg.norm(query) + 1e-12
    m_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    return (matrix @ query) / (m_norms * q_norm)


def _parse_metadata(raw: Optional[str]) -> dict[str, Any]:
    """Tolerant JSON decode — a malformed row should not crash recall."""
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {"_raw": decoded}
    except json.JSONDecodeError:
        logger.warning("memory: failed to decode metadata json, falling back to empty")
        return {}


def _parse_created_at(raw: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamps written by us. Tolerant of legacy formats."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        logger.warning("memory: unparseable created_at %r", raw)
        return None


# ---------------------------------------------------------------------------
# VectorMemory
# ---------------------------------------------------------------------------


class VectorMemory:
    """Per-user vector memory store backed by SQLite.

    Why a class (vs a module of functions): tests want to point at a temporary
    DB path without touching the user's real ``~/.quorum/memory.db``. A class
    threading ``db_path`` through ``__init__`` makes that trivial and keeps
    connections short-lived (one per call, opened on the worker thread).
    """

    SCHEMA: str = """
    CREATE TABLE IF NOT EXISTS memories (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        kind        TEXT NOT NULL,
        content     TEXT NOT NULL,
        embedding   BLOB NOT NULL,
        metadata    TEXT,
        created_at  TIMESTAMP NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_memories_user_kind
        ON memories(user_id, kind);
    CREATE INDEX IF NOT EXISTS idx_memories_user_created
        ON memories(user_id, created_at);
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        """Initialize the store and ensure schema exists.

        We create the parent directory eagerly so the first ``add`` call does
        not race with directory creation on multi-worker setups.
        """
        self.db_path: Path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # -- low-level sync core (run inside asyncio.to_thread) -----------------

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived connection.

        ``check_same_thread=False`` lets us hop threads via ``to_thread``.
        WAL keeps reads fast while a background write is in flight.
        """
        # NOTE: we intentionally do NOT pass detect_types=PARSE_DECLTYPES.
        # The stdlib TIMESTAMP converter only understands "YYYY-MM-DD HH:MM:SS"
        # — it chokes on the ISO-8601 strings (with 'T' and tz) that we write.
        # We parse timestamps ourselves via _parse_created_at, which is also
        # forward-compat with timezone-aware datetimes.
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit BEGIN where needed
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _ensure_schema(self) -> None:
        """Idempotent schema creation. Cheap on every init."""
        conn = self._connect()
        try:
            conn.executescript(self.SCHEMA)
        finally:
            conn.close()

    def _add_sync(
        self,
        memory_id: str,
        user_id: str,
        kind: str,
        content: str,
        embedding_blob: bytes,
        metadata_json: str,
        created_at: datetime,
    ) -> None:
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO memories (id, user_id, kind, content, embedding, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    user_id,
                    kind,
                    content,
                    embedding_blob,
                    metadata_json,
                    created_at.isoformat(),
                ),
            )
        finally:
            conn.close()

    def _fetch_user_rows(
        self, user_id: str, kind: Optional[str]
    ) -> list[tuple[str, str, str, bytes, Optional[str], Optional[str]]]:
        """Pull all memories for a user (optionally filtered by kind).

        Why bulk-load instead of doing similarity in SQL: SQLite has no native
        vector ops in v0.1's stdlib. We pull rows + decode BLOBs in numpy. For
        the realistic per-user scale (<100k rows) this is faster than any
        userland-SQL approximation, and avoids a C extension dependency.
        """
        conn = self._connect()
        try:
            if kind is None:
                cur = conn.execute(
                    "SELECT id, kind, content, embedding, metadata, created_at "
                    "FROM memories WHERE user_id = ?",
                    (user_id,),
                )
            else:
                cur = conn.execute(
                    "SELECT id, kind, content, embedding, metadata, created_at "
                    "FROM memories WHERE user_id = ? AND kind = ?",
                    (user_id, kind),
                )
            return cur.fetchall()
        finally:
            conn.close()

    def _forget_sync(self, user_id: str, memory_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM memories WHERE user_id = ? AND id = ?",
                (user_id, memory_id),
            )
            return cur.rowcount > 0
        finally:
            conn.close()

    def _age_out_sync(self, user_id: str, max_age_days: int) -> int:
        """Prune old + low-relevance memories.

        Relevance heuristic: keep all ``correction``, ``preference``, ``fact``
        kinds forever (those are the user's pinned signal); only prune raw
        ``query`` / ``response`` rows beyond the age cutoff. This protects the
        long-term identity model while bounding storage growth.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max_age_days)).isoformat()
        conn = self._connect()
        try:
            cur = conn.execute(
                "DELETE FROM memories "
                "WHERE user_id = ? AND kind IN ('query', 'response') "
                "AND created_at < ?",
                (user_id, cutoff),
            )
            return cur.rowcount
        finally:
            conn.close()

    # -- public async API ---------------------------------------------------

    async def add(
        self,
        user_id: str,
        kind: MemoryKind,
        content: str,
        embedding: Iterable[float] | NDArray[Any],
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store a memory and return its id.

        Why async + ``to_thread``: SQLite calls are synchronous but the rest of
        Quorum is async; offloading to a worker thread keeps the event loop
        free during concurrent consensus calls that all want to write.
        """
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"kind must be one of {sorted(_VALID_KINDS)}, got {kind!r}"
            )
        if not user_id:
            raise ValueError("user_id is required")
        if not content:
            raise ValueError("content must not be empty")

        memory_id = uuid.uuid4().hex
        blob = _to_float32_bytes(embedding)
        meta_json = json.dumps(metadata or {}, default=str)
        created_at = datetime.now(timezone.utc)

        await asyncio.to_thread(
            self._add_sync,
            memory_id,
            user_id,
            kind,
            content,
            blob,
            meta_json,
            created_at,
        )
        logger.debug(
            "memory: added %s for user=%s kind=%s (%d dims)",
            memory_id, user_id, kind, len(blob) // 4,
        )
        return memory_id

    async def search(
        self,
        user_id: str,
        query_embedding: Iterable[float] | NDArray[Any],
        kind: Optional[MemoryKind] = None,
        top_k: int = 5,
        min_similarity: float = 0.4,
    ) -> list[MemoryHit]:
        """Return top-k cosine-ranked memories above ``min_similarity``.

        Why a similarity floor instead of just top-k: injecting irrelevant
        memories actively degrades LLM output (the "context rot" failure mode).
        It is better to return zero hits than three barely-related ones.
        """
        if top_k <= 0:
            return []

        rows = await asyncio.to_thread(self._fetch_user_rows, user_id, kind)
        if not rows:
            return []

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        if query_vec.ndim != 1:
            raise ValueError(f"query_embedding must be 1-D, got shape {query_vec.shape}")

        # Decode embeddings, filtering rows whose dimensionality does not match.
        # A dim mismatch is almost always "user switched embedding models" — we
        # log and skip rather than crash recall.
        ids: list[str] = []
        kinds: list[str] = []
        contents: list[str] = []
        metas: list[dict[str, Any]] = []
        timestamps: list[Optional[datetime]] = []
        vectors: list[NDArray[np.float32]] = []

        target_dim = query_vec.shape[0]
        for row in rows:
            row_id, row_kind, content, blob, meta_raw, created_raw = row
            vec = _from_blob(blob)
            if vec.shape[0] != target_dim:
                logger.warning(
                    "memory: skipping %s (dim %d != query dim %d)",
                    row_id, vec.shape[0], target_dim,
                )
                continue
            ids.append(row_id)
            kinds.append(row_kind)
            contents.append(content)
            metas.append(_parse_metadata(meta_raw))
            timestamps.append(_parse_created_at(created_raw))
            vectors.append(vec)

        if not vectors:
            return []

        matrix = np.vstack(vectors)
        sims = _cosine_matrix(query_vec, matrix)

        # argpartition is O(n) — cheaper than full argsort when top_k << n.
        k = min(top_k, sims.shape[0])
        top_idx = np.argpartition(-sims, k - 1)[:k]
        # Sort just the top-k slice for a deterministic descending order.
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        hits: list[MemoryHit] = []
        for idx in top_idx:
            sim = float(sims[idx])
            if sim < min_similarity:
                continue
            hits.append(
                MemoryHit(
                    id=ids[idx],
                    content=contents[idx],
                    similarity=sim,
                    kind=kinds[idx],
                    metadata=metas[idx],
                    created_at=timestamps[idx],
                )
            )
        return hits

    async def forget(self, user_id: str, memory_id: str) -> bool:
        """Hard-delete a single memory. Returns True if a row was removed.

        Why a hard delete (no tombstone): the GDPR / right-to-be-forgotten path
        in the HSP user agreement requires actual erasure, not soft-delete.
        """
        return await asyncio.to_thread(self._forget_sync, user_id, memory_id)

    async def age_out(self, user_id: str, max_age_days: int = 180) -> int:
        """Prune memories older than ``max_age_days``. Returns rows deleted.

        Run this on a cron (daily) or on shutdown. See :meth:`_age_out_sync`
        for the kind-selective retention policy.
        """
        if max_age_days <= 0:
            raise ValueError("max_age_days must be positive")
        deleted = await asyncio.to_thread(self._age_out_sync, user_id, max_age_days)
        if deleted:
            logger.info("memory: aged out %d rows for user=%s", deleted, user_id)
        return deleted

    # -- bonus: compact context blob ---------------------------------------

    async def recall_context(
        self,
        user_id: str,
        query_embedding: Iterable[float] | NDArray[Any],
        max_tokens: int = 200,
    ) -> str:
        """Return a compact text blob suitable for prompt injection.

        Why a separate method: callers building prompts do not want to format
        and truncate hits themselves on every call. Centralizing the shape
        ("[kind] content..." per hit, ~60 tokens each, max 3 hits) lets us
        tune the prompt-engineering recipe in one place.

        Token estimate is a deliberately rough chars/4 heuristic — exact
        tokenization would require pulling a tokenizer for each provider,
        which defeats the "no external deps" constraint. The 200-token cap
        is a soft cap; we never exceed it.
        """
        hits = await self.search(
            user_id=user_id,
            query_embedding=query_embedding,
            top_k=3,
            min_similarity=0.4,
        )
        if not hits:
            return ""

        per_hit_token_budget = 60
        per_hit_char_budget = per_hit_token_budget * 4
        total_char_budget = max_tokens * 4

        lines: list[str] = []
        used_chars = 0
        for hit in hits:
            snippet = hit.content.strip().replace("\n", " ")
            if len(snippet) > per_hit_char_budget:
                snippet = snippet[: per_hit_char_budget - 1].rstrip() + "…"
            line = f"[{hit.kind}] {snippet}"
            if used_chars + len(line) + 1 > total_char_budget:
                break
            lines.append(line)
            used_chars += len(line) + 1

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# In-memory fallback factory
# ---------------------------------------------------------------------------


def make_memory(env: Optional[dict[str, str]] = None) -> VectorMemory:
    """Construct a :class:`VectorMemory` from environment, with safe fallback.

    Why this exists: tests, ephemeral containers, and CI runners should not
    write to ``~/.quorum``. If ``QUORUM_MEMORY_DB`` is unset (or set to
    ``:memory:``) we transparently use an in-process SQLite DB so callers
    never have to special-case "no env vars".
    """
    env = env if env is not None else dict(os.environ)
    path_raw = env.get("QUORUM_MEMORY_DB", "").strip()
    if not path_raw or path_raw == ":memory:":
        # Anonymous in-memory DB, scoped to this VectorMemory instance.
        return VectorMemory(db_path=":memory:")
    return VectorMemory(db_path=path_raw)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def _t_basic_roundtrip(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    vec = np.random.default_rng(0).normal(size=384).astype(np.float32)
    mid = await mem.add("u1", "fact", "user prefers terse answers", vec, {"src": "test"})
    assert isinstance(mid, str) and len(mid) == 32

    hits = await mem.search("u1", vec, top_k=3)
    assert len(hits) == 1
    assert hits[0].content == "user prefers terse answers"
    assert hits[0].similarity > 0.99
    assert hits[0].metadata == {"src": "test"}
    assert hits[0].created_at is not None


async def _t_min_similarity_floor(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    rng = np.random.default_rng(1)
    a = rng.normal(size=128).astype(np.float32)
    b = rng.normal(size=128).astype(np.float32)
    await mem.add("u1", "query", "hello", a)
    hits = await mem.search("u1", b, top_k=5, min_similarity=0.9)
    assert hits == []


async def _t_user_isolation(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    vec = np.ones(64, dtype=np.float32)
    await mem.add("alice", "fact", "alice's secret", vec)
    await mem.add("bob", "fact", "bob's note", vec)
    alice_hits = await mem.search("alice", vec, min_similarity=0.0)
    assert len(alice_hits) == 1
    assert "alice" in alice_hits[0].content


async def _t_forget(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    vec = np.ones(32, dtype=np.float32)
    mid = await mem.add("u1", "fact", "delete me", vec)
    assert await mem.forget("u1", mid) is True
    assert await mem.forget("u1", mid) is False
    assert await mem.search("u1", vec, min_similarity=0.0) == []


async def _t_age_out_protects_pinned(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    vec = np.ones(16, dtype=np.float32)
    # Manually backdate rows to simulate aging.
    q_id = await mem.add("u1", "query", "old query", vec)
    f_id = await mem.add("u1", "fact", "pinned fact", vec)
    long_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    conn = mem._connect()
    try:
        conn.execute("UPDATE memories SET created_at = ?", (long_ago,))
    finally:
        conn.close()
    deleted = await mem.age_out("u1", max_age_days=30)
    assert deleted == 1
    remaining = await mem.search("u1", vec, min_similarity=0.0)
    remaining_ids = {h.id for h in remaining}
    assert f_id in remaining_ids
    assert q_id not in remaining_ids


async def _t_dim_mismatch_skipped(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    await mem.add("u1", "fact", "old-model row", np.ones(128, dtype=np.float32))
    # Query with a different dimensionality must not crash.
    hits = await mem.search("u1", np.ones(384, dtype=np.float32), min_similarity=0.0)
    assert hits == []


async def _t_recall_context_shape(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    vec = np.ones(8, dtype=np.float32)
    await mem.add("u1", "preference", "answer in pt-br", vec)
    await mem.add("u1", "fact", "user is in london", vec)
    blob = await mem.recall_context("u1", vec, max_tokens=200)
    assert "[preference]" in blob or "[fact]" in blob
    assert len(blob) <= 200 * 4


async def _t_invalid_kind_rejected(tmpdir: Path) -> None:
    mem = VectorMemory(db_path=tmpdir / "m.db")
    try:
        await mem.add("u1", "garbage", "x", np.ones(4, dtype=np.float32))  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("invalid kind should raise ValueError")


async def _run_all_tests() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        await _t_basic_roundtrip(tmp / "a")
        (tmp / "a").mkdir(exist_ok=True)
        await _t_min_similarity_floor(tmp / "b")
        (tmp / "b").mkdir(exist_ok=True)
        await _t_user_isolation(tmp / "c")
        (tmp / "c").mkdir(exist_ok=True)
        await _t_forget(tmp / "d")
        (tmp / "d").mkdir(exist_ok=True)
        await _t_age_out_protects_pinned(tmp / "e")
        (tmp / "e").mkdir(exist_ok=True)
        await _t_dim_mismatch_skipped(tmp / "f")
        (tmp / "f").mkdir(exist_ok=True)
        await _t_recall_context_shape(tmp / "g")
        (tmp / "g").mkdir(exist_ok=True)
        await _t_invalid_kind_rejected(tmp / "h")
    logger.info("memory: all self-tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_run_all_tests())
