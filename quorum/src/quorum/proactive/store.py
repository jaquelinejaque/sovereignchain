# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed store for Signal/Draft/Action with HSP-style hash chain.

Why SQLite:
* Zero infra to start (file on disk; on Cloud Run mount /tmp)
* WAL mode handles the ingest-worker concurrency
* Easy to inspect manually with ``sqlite3`` CLI when debugging
* Migration to Postgres later is a one-file swap (same SQL)

Schema is intentionally permissive (TEXT columns + JSON blobs) so we can
evolve the dataclasses without ALTER TABLE dances. Indexes cover the
two hot queries: dedupe lookup and "what's pending review".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from quorum.proactive.signal import Action, Draft, Signal

logger = logging.getLogger(__name__)


_DB_ENV = "QUORUM_PROACTIVE_DB"


def _db_path() -> Path:
    raw = os.getenv(_DB_ENV) or str(Path.home() / ".quorum" / "proactive.db")
    p = Path(raw).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id           TEXT PRIMARY KEY,
    prev_hash    TEXT NOT NULL DEFAULT '',
    dedupe_key   TEXT NOT NULL,
    source       TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    author       TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL DEFAULT '',
    body         TEXT NOT NULL DEFAULT '',
    url          TEXT NOT NULL DEFAULT '',
    fetched_at   TEXT NOT NULL,
    extra_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_dedupe ON signals(dedupe_key);
CREATE INDEX IF NOT EXISTS idx_signals_fetched ON signals(fetched_at DESC);

CREATE TABLE IF NOT EXISTS drafts (
    id                TEXT PRIMARY KEY,
    prev_hash         TEXT NOT NULL DEFAULT '',
    signal_id         TEXT NOT NULL,
    kind              TEXT NOT NULL,
    target            TEXT NOT NULL DEFAULT '',
    subject           TEXT NOT NULL DEFAULT '',
    body              TEXT NOT NULL DEFAULT '',
    intent_score      REAL NOT NULL DEFAULT 0.0,
    draft_score       REAL NOT NULL DEFAULT 0.0,
    rationale         TEXT NOT NULL DEFAULT '',
    consensus_models  TEXT NOT NULL DEFAULT '[]',
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drafts_signal ON drafts(signal_id);
CREATE INDEX IF NOT EXISTS idx_drafts_created ON drafts(created_at DESC);

CREATE TABLE IF NOT EXISTS actions (
    id                  TEXT PRIMARY KEY,
    prev_hash           TEXT NOT NULL DEFAULT '',
    draft_id            TEXT NOT NULL,
    decision            TEXT NOT NULL DEFAULT 'pending',
    edited_body         TEXT NOT NULL DEFAULT '',
    decided_at          TEXT NOT NULL DEFAULT '',
    executed_at         TEXT NOT NULL DEFAULT '',
    execution_result    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_actions_decision ON actions(decision);
CREATE INDEX IF NOT EXISTS idx_actions_draft ON actions(draft_id);
"""


@contextmanager
def _connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    p = path or _db_path()
    conn = sqlite3.connect(p, timeout=10.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def _last_chain_hash(conn: sqlite3.Connection, table: str) -> str:
    """Return id of the most-recent row in ``table`` for chain linkage."""
    row = conn.execute(
        f"SELECT id FROM {table} ORDER BY rowid DESC LIMIT 1"
    ).fetchone()
    return row["id"] if row else ""


# --------------------------------------------------------------------------- #
# Public API — async wrappers around sync SQLite                              #
# --------------------------------------------------------------------------- #


async def insert_signal(s: Signal) -> bool:
    """Insert one Signal. Returns False if dedupe_key already exists.

    Computes ``s.id`` from prev chain hash + sealed payload before write.
    The whole sync body runs in ``asyncio.to_thread`` so the event loop
    is never blocked.
    """
    def _do() -> bool:
        with _connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM signals WHERE dedupe_key = ? LIMIT 1",
                (s.dedupe_key,),
            ).fetchone()
            if existing:
                return False
            prev = _last_chain_hash(conn, "signals")
            s.seal(prev)
            conn.execute(
                """INSERT INTO signals
                   (id, prev_hash, dedupe_key, source, external_id, author,
                    title, body, url, fetched_at, extra_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    s.id, s.prev_hash, s.dedupe_key, s.source, s.external_id,
                    s.author, s.title, s.body, s.url, s.fetched_at,
                    json.dumps(s.extra, ensure_ascii=False, default=str),
                ),
            )
            return True
    return await asyncio.to_thread(_do)


async def insert_draft(d: Draft) -> str:
    """Persist one Draft. Returns the sealed id."""
    def _do() -> str:
        with _connect() as conn:
            prev = _last_chain_hash(conn, "drafts")
            d.seal(prev)
            conn.execute(
                """INSERT INTO drafts
                   (id, prev_hash, signal_id, kind, target, subject, body,
                    intent_score, draft_score, rationale, consensus_models,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    d.id, d.prev_hash, d.signal_id, d.kind, d.target,
                    d.subject, d.body, d.intent_score, d.draft_score,
                    d.rationale,
                    json.dumps(d.consensus_models, ensure_ascii=False),
                    d.created_at,
                ),
            )
            return d.id
    return await asyncio.to_thread(_do)


async def insert_action(a: Action) -> str:
    def _do() -> str:
        with _connect() as conn:
            prev = _last_chain_hash(conn, "actions")
            a.seal(prev)
            conn.execute(
                """INSERT INTO actions
                   (id, prev_hash, draft_id, decision, edited_body,
                    decided_at, executed_at, execution_result)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    a.id, a.prev_hash, a.draft_id, a.decision, a.edited_body,
                    a.decided_at, a.executed_at,
                    json.dumps(a.execution_result, ensure_ascii=False,
                               default=str),
                ),
            )
            return a.id
    return await asyncio.to_thread(_do)


async def update_action(action_id: str, **fields: Any) -> None:
    """Update a pending action with new decision/edited_body/etc."""
    if not fields:
        return
    def _do() -> None:
        with _connect() as conn:
            sets = []
            args: list[Any] = []
            for k, v in fields.items():
                sets.append(f"{k} = ?")
                args.append(json.dumps(v, default=str) if k == "execution_result" else v)
            args.append(action_id)
            conn.execute(
                f"UPDATE actions SET {', '.join(sets)} WHERE id = ?", args,
            )
    await asyncio.to_thread(_do)


async def pending_drafts(limit: int = 50) -> list[dict[str, Any]]:
    """Return drafts that have NO action yet OR whose action is pending.

    Used by the email digest builder.
    """
    def _do() -> list[dict[str, Any]]:
        with _connect() as conn:
            rows = conn.execute(
                """SELECT d.*, s.author AS signal_author, s.title AS signal_title,
                          s.url AS signal_url, s.source AS signal_source
                   FROM drafts d
                   JOIN signals s ON s.id = d.signal_id
                   LEFT JOIN actions a ON a.draft_id = d.id
                   WHERE a.id IS NULL OR a.decision = 'pending'
                   ORDER BY d.intent_score DESC, d.created_at DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
    return await asyncio.to_thread(_do)


async def get_draft(draft_id: str) -> dict[str, Any] | None:
    def _do() -> dict[str, Any] | None:
        with _connect() as conn:
            row = conn.execute(
                "SELECT * FROM drafts WHERE id = ?", (draft_id,),
            ).fetchone()
            return dict(row) if row else None
    return await asyncio.to_thread(_do)


async def stats() -> dict[str, int]:
    def _do() -> dict[str, int]:
        with _connect() as conn:
            n_signals = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
            n_drafts = conn.execute("SELECT COUNT(*) FROM drafts").fetchone()[0]
            n_pending = conn.execute(
                """SELECT COUNT(*) FROM drafts d
                   LEFT JOIN actions a ON a.draft_id = d.id
                   WHERE a.id IS NULL OR a.decision = 'pending'"""
            ).fetchone()[0]
            n_executed = conn.execute(
                "SELECT COUNT(*) FROM actions WHERE executed_at != ''"
            ).fetchone()[0]
            return {
                "signals": n_signals,
                "drafts": n_drafts,
                "pending_review": n_pending,
                "executed": n_executed,
            }
    return await asyncio.to_thread(_do)


__all__ = [
    "insert_signal", "insert_draft", "insert_action", "update_action",
    "pending_drafts", "get_draft", "stats",
]
