# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Raw response logging — opt-in per-consensus persistence of model outputs.

Why this exists
---------------
The aggregate stores (``hebbian.db``, ``competition.db``) keep
similarity sums and ELO ratings but throw away the original model
responses. That makes the historical dataset useless for re-analysis
in a different embedding space, for cross-vendor diffing, or for
re-scoring under a new metric — you have to re-run the (paid) LLM
calls to recover anything.

This module persists each ``(query_hash, model, response_text)``
tuple in a separate SQLite table the moment the consensus call
finishes, so the raw signal survives for later analysis.

Privacy posture
---------------
OFF BY DEFAULT. Activated only when the caller sets
``QUORUM_LOG_RESPONSES=1`` in the environment. We log:

* a SHA-256 ``query_hash`` (never the prompt plaintext)
* the model name
* the response text (truncated to ``MAX_RESPONSE_BYTES``)
* latency / cost / timestamp
* optional ``embedding_provider`` + ``embedding_dim`` if the caller
  passes a vector at write time (so the dataset is self-describing)

Design constraints (the things that must not change)
----------------------------------------------------
1. **Never block the response path.** Every write is wrapped in
   ``try/except`` and runs in ``asyncio.to_thread`` so a corrupt
   DB or full disk cannot delay the consensus answer.
2. **Opt-in only.** A fresh clone with no env var sees zero
   behaviour change — this module is dead code until activated.
3. **Schema-stable.** Adding columns later is OK (ALTER TABLE);
   renames are forbidden (existing dumps break).
4. **Append-only.** No update/delete in the hot path; an explicit
   ``vacuum_older_than()`` is the only retention knob.

Sister stores: ``hebbian.py``, ``competition.py``, ``rlhf.py``.
Follow the same idiomatic shape (lazy ``__init__``, async public
methods over ``asyncio.to_thread``, sqlite + WAL mode).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

# Same hard cap as ``core.consensus.MAX_RESPONSE_BYTES`` so we never
# store more than what the embedder already saw. If the upstream cap
# changes this constant should follow — but it is duplicated rather
# than imported to keep this module a leaf node (no cycles with core).
MAX_RESPONSE_BYTES = 16_000

_ENV_FLAG = "QUORUM_LOG_RESPONSES"
_ENV_DB_PATH = "QUORUM_RESPONSE_LOG_DB"
_DEFAULT_DB = Path.home() / ".quorum" / "responses.db"


def is_enabled() -> bool:
    """True iff the operator has opted in via env var.

    Cheap to call on the hot path — does not touch disk.
    """
    return os.getenv(_ENV_FLAG, "").strip() in {"1", "true", "yes", "on"}


def _db_path() -> Path:
    """Resolve the configured DB path (env override → default)."""
    override = os.getenv(_ENV_DB_PATH, "").strip()
    if override:
        return Path(override).expanduser()
    return _DEFAULT_DB


_SCHEMA = """
CREATE TABLE IF NOT EXISTS response_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    query_hash      TEXT    NOT NULL,
    query_class     TEXT    NOT NULL DEFAULT 'general',
    model           TEXT    NOT NULL,
    response_text   TEXT    NOT NULL,
    latency_ms      REAL    NOT NULL DEFAULT 0.0,
    cost_usd        REAL    NOT NULL DEFAULT 0.0,
    weight          REAL    NOT NULL DEFAULT 0.0,
    was_canonical   INTEGER NOT NULL DEFAULT 0,
    embedding_provider TEXT,
    embedding_dim   INTEGER,
    embedding_json  TEXT,
    created_at      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_response_log_query_hash
    ON response_log(query_hash);
CREATE INDEX IF NOT EXISTS ix_response_log_model
    ON response_log(model);
CREATE INDEX IF NOT EXISTS ix_response_log_created
    ON response_log(created_at);
"""


@contextmanager
def _connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the DB with WAL + sensible defaults.

    Yields a connection that auto-commits on context exit. The caller
    is responsible for using ``with`` so cleanup runs on every path.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        yield conn
    finally:
        conn.close()


def _hash_query(prompt: str) -> str:
    """Stable SHA-256 of the query — never store plaintext."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _record_sync(
    db_path: Path,
    rows: list[dict[str, Any]],
) -> int:
    """Insert ``rows`` in one transaction. Returns count inserted.

    Synchronous body — the async wrapper offloads to ``to_thread`` so
    the hot path never sees a disk wait.
    """
    if not rows:
        return 0
    with _connect(db_path) as conn:
        conn.execute("BEGIN")
        try:
            conn.executemany(
                """INSERT INTO response_log
                   (query_hash, query_class, model, response_text,
                    latency_ms, cost_usd, weight, was_canonical,
                    embedding_provider, embedding_dim, embedding_json,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        r["query_hash"],
                        r.get("query_class", "general"),
                        r["model"],
                        r["response_text"],
                        r.get("latency_ms", 0.0),
                        r.get("cost_usd", 0.0),
                        r.get("weight", 0.0),
                        1 if r.get("was_canonical") else 0,
                        r.get("embedding_provider"),
                        r.get("embedding_dim"),
                        r.get("embedding_json"),
                        r.get("created_at", time.time()),
                    )
                    for r in rows
                ],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return len(rows)


async def record_consensus_round(
    *,
    prompt: str,
    query_class: str,
    model_responses: Iterable[dict[str, Any]],
    canonical_model: str | None = None,
) -> int:
    """Persist one consensus round. No-op when disabled.

    Args:
        prompt: The original user query (hashed before storage).
        query_class: Router/meta classification (general/code/…).
        model_responses: Iterable of dicts, one per provider, with
            keys: ``model``, ``response_text``, ``latency_ms``,
            ``cost_usd``, ``weight``, and optionally
            ``embedding_provider`` / ``embedding_dim`` /
            ``embedding`` (list[float]).
        canonical_model: Name of the model whose response was picked
            as the synthesized answer. ``None`` if no winner.

    Returns:
        Number of rows inserted; ``0`` if logging is disabled or the
        write failed silently. Never raises — callers can fire-and-
        forget without a try/except.
    """
    if not is_enabled():
        return 0

    query_hash = _hash_query(prompt)
    now = time.time()
    db_path = _db_path()

    rows: list[dict[str, Any]] = []
    for mr in model_responses:
        text = (mr.get("response_text") or "")[:MAX_RESPONSE_BYTES]
        if not text:
            continue  # skip failed providers — no useful signal
        emb = mr.get("embedding")
        row = {
            "query_hash": query_hash,
            "query_class": query_class,
            "model": mr["model"],
            "response_text": text,
            "latency_ms": float(mr.get("latency_ms", 0.0)),
            "cost_usd": float(mr.get("cost_usd", 0.0)),
            "weight": float(mr.get("weight", 0.0)),
            "was_canonical": (mr["model"] == canonical_model),
            "embedding_provider": mr.get("embedding_provider"),
            "embedding_dim": (len(emb) if isinstance(emb, (list, tuple)) else None),
            # JSON-encode the embedding only if explicitly provided. This
            # is heavy (~30 KB for 3072-dim float32 stringified) so the
            # caller has to opt in by passing it. When omitted we still
            # capture the model output, just without the vector.
            "embedding_json": (json.dumps(emb) if isinstance(emb, (list, tuple)) else None),
            "created_at": now,
        }
        rows.append(row)

    try:
        return await asyncio.to_thread(_record_sync, db_path, rows)
    except Exception as e:  # noqa: BLE001
        logger.debug("response_log write skipped (%s)", e)
        return 0


# --------------------------------------------------------------------------- #
# Export / inspection helpers — used by ``quorum responses export`` CLI
# --------------------------------------------------------------------------- #


def export_jsonl(
    *,
    since: float | None = None,
    until: float | None = None,
    model: str | None = None,
    db_path: Path | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield rows as dicts. Synchronous — designed for CLI streaming.

    Args:
        since/until: Unix timestamps for date filtering (None = open).
        model: Restrict to a single model name (None = all).
        db_path: Override DB path; default resolves via env.
    """
    path = db_path or _db_path()
    if not path.exists():
        return
    sql = ["SELECT * FROM response_log WHERE 1=1"]
    args: list[Any] = []
    if since is not None:
        sql.append("AND created_at >= ?")
        args.append(since)
    if until is not None:
        sql.append("AND created_at < ?")
        args.append(until)
    if model:
        sql.append("AND model = ?")
        args.append(model)
    sql.append("ORDER BY created_at ASC")
    query = " ".join(sql)
    with _connect(path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(query, args):
            d = dict(row)
            # Decode embedding back to list[float] for downstream use.
            if d.get("embedding_json"):
                try:
                    d["embedding"] = json.loads(d["embedding_json"])
                except Exception:  # noqa: BLE001
                    d["embedding"] = None
                d.pop("embedding_json", None)
            yield d


def stats(db_path: Path | None = None) -> dict[str, Any]:
    """Cheap overview — row count, distinct models, date range."""
    path = db_path or _db_path()
    if not path.exists():
        return {"enabled": is_enabled(), "db_exists": False}
    with _connect(path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0]
        n_models = conn.execute(
            "SELECT COUNT(DISTINCT model) FROM response_log"
        ).fetchone()[0]
        n_queries = conn.execute(
            "SELECT COUNT(DISTINCT query_hash) FROM response_log"
        ).fetchone()[0]
        first_last = conn.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM response_log"
        ).fetchone()
        with_emb = conn.execute(
            "SELECT COUNT(*) FROM response_log WHERE embedding_json IS NOT NULL"
        ).fetchone()[0]
    return {
        "enabled": is_enabled(),
        "db_path": str(path),
        "rows": n,
        "distinct_models": n_models,
        "distinct_queries": n_queries,
        "rows_with_embedding": with_emb,
        "first_at": first_last[0],
        "last_at": first_last[1],
    }


def vacuum_older_than(seconds: float, db_path: Path | None = None) -> int:
    """Delete rows older than ``seconds`` and ``VACUUM``. Returns rows deleted.

    Operator-controlled retention — not called automatically. Run from
    a cron / launchd job, never from the consensus hot path.
    """
    path = db_path or _db_path()
    if not path.exists():
        return 0
    cutoff = time.time() - seconds
    with _connect(path) as conn:
        cur = conn.execute(
            "DELETE FROM response_log WHERE created_at < ?", (cutoff,)
        )
        deleted = cur.rowcount
        conn.execute("VACUUM")
    return deleted


__all__ = [
    "is_enabled",
    "record_consensus_round",
    "export_jsonl",
    "stats",
    "vacuum_older_than",
    "MAX_RESPONSE_BYTES",
]
