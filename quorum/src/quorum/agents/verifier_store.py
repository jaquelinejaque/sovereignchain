"""Verifier persistence layer — SQLite-backed audit trail for drafts.

Why this exists
---------------
The verifier (`draft_verifier.py`) was a pure stateless regex linter: every
call evaluated a draft in isolation, returned conflicts, and forgot
everything. That made the verifier a "facade by design" — impossible to
answer questions like:

- Was this draft already verified before?
- Did the confidence increase between attempts?
- What was the lineage of revisions that led to the final published copy?

This module is the minimum-viable persistence layer. It is intentionally
narrow: one table, two public functions (`save_draft`, `get_history`),
plus a `connect()` helper. No ORM, no migrations framework, no asyncio.

Design choices
--------------
- SQLite with `check_same_thread=False` + WAL mode so it can be called
  from async contexts without ceremony.
- Default location `~/.quorum/drafts.db`, overridable via env
  `QUORUM_DRAFTS_DB` or the `db_path` kwarg (tests use tmp paths).
- `save_draft()` MUST NOT raise on disk failure — drafts are expensive
  (consensus call) and losing persistence is acceptable, but losing the
  draft itself because the disk filled up is not. Failures are swallowed
  and a sentinel empty string is returned.
- `get_history()` walks the `parent_draft_id` chain in Python (max 3-5
  hops in practice — trivial). Returns oldest→newest.
- Schema includes `parent_draft_id`, `version`, `confidence_delta`, and
  `fact_sheet_hash` even though the minimal entrypoint may not populate
  them yet — the columns are there so a future patch on `_ask_quorum_all`
  can fill in the lineage without a migration.

Backward-compat
---------------
Existing call sites of `find_conflicts()` / `annotate_draft()` continue to
work unchanged. `save_draft()` is an additive side-effect inside
`draft_verifier`, gated by a feature flag (`QUORUM_VERIFIER_PERSIST`,
default ON) so tests that don't want disk writes can disable it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    draft_id         TEXT PRIMARY KEY,
    parent_draft_id  TEXT,
    version          INTEGER NOT NULL DEFAULT 1,
    kind             TEXT NOT NULL DEFAULT 'unknown',
    content          TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    conflicts_json   TEXT,
    conflicts_count  INTEGER NOT NULL DEFAULT 0,
    confidence       REAL NOT NULL DEFAULT 0.0,
    confidence_delta REAL,
    fact_sheet_hash  TEXT,
    created_at       TEXT NOT NULL,
    FOREIGN KEY (parent_draft_id) REFERENCES drafts(draft_id)
);
CREATE INDEX IF NOT EXISTS idx_drafts_parent ON drafts(parent_draft_id);
CREATE INDEX IF NOT EXISTS idx_drafts_kind_created ON drafts(kind, created_at);
"""


# ---------------------------------------------------------------------------
# Path / connection helpers
# ---------------------------------------------------------------------------


def default_db_path() -> Path:
    """Where the drafts DB lives by default. Honors QUORUM_DRAFTS_DB."""
    env = os.environ.get("QUORUM_DRAFTS_DB")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".quorum" / "drafts.db"


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and lazily create) the drafts DB. Enables WAL + foreign keys."""
    path = db_path or default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
    except sqlite3.DatabaseError:
        # PRAGMA failures are non-fatal; we can still read/write.
        pass
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hash_fact_sheet(fact_sheet: dict | None) -> str | None:
    if not fact_sheet:
        return None
    try:
        encoded = json.dumps(fact_sheet, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _derive_verdict(conflicts: list[dict] | None) -> str:
    if not conflicts:
        return "clean"
    return "conflicts"


def _lookup_parent(
    conn: sqlite3.Connection, parent_draft_id: str | None
) -> sqlite3.Row | None:
    if not parent_draft_id:
        return None
    cur = conn.execute(
        "SELECT version, confidence FROM drafts WHERE draft_id = ?",
        (parent_draft_id,),
    )
    return cur.fetchone()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_draft(
    content: str,
    verdict: str | None = None,
    confidence: float = 0.0,
    *,
    conflicts: list[dict] | None = None,
    parent_draft_id: str | None = None,
    kind: str = "unknown",
    fact_sheet: dict | None = None,
    db_path: Path | None = None,
) -> str:
    """Persist a single verified draft attempt. Returns its draft_id.

    NEVER raises on disk failure. On error, returns the empty string so
    callers can keep going (the draft itself is more valuable than its
    audit row).

    Parameters
    ----------
    content : str
        The draft text after verification (annotated or not).
    verdict : str | None
        If omitted, derived from `conflicts` ('clean' or 'conflicts').
        Callers that annotate may pass 'annotated' explicitly.
    confidence : float
        Consensus confidence for this attempt (0.0-1.0).
    conflicts : list[dict] | None
        Full conflict objects as returned by `find_conflicts`. Stored as
        JSON for later forensic analysis.
    parent_draft_id : str | None
        Previous attempt in the same lineage. NULL on first attempt.
    kind : str
        Draft kind ('linkedin', 'twitter', 'show_hn', etc.). Free-form.
    fact_sheet : dict | None
        The fact_sheet used during verification. Hashed (sha256) and
        stored; the dict itself is NOT persisted.
    db_path : Path | None
        Override DB location (tests).
    """
    try:
        conn = connect(db_path)
    except (sqlite3.DatabaseError, OSError):
        return ""

    try:
        parent_row = _lookup_parent(conn, parent_draft_id)
        if parent_row is not None:
            version = int(parent_row["version"]) + 1
            try:
                confidence_delta: float | None = float(confidence) - float(
                    parent_row["confidence"]
                )
            except (TypeError, ValueError):
                confidence_delta = None
        else:
            version = 1
            confidence_delta = None

        if verdict is None:
            verdict = _derive_verdict(conflicts)

        try:
            conflicts_json = json.dumps(conflicts or [], default=str)
        except (TypeError, ValueError):
            conflicts_json = "[]"

        draft_id = uuid.uuid4().hex
        row = (
            draft_id,
            parent_draft_id,
            version,
            kind,
            content,
            verdict,
            conflicts_json,
            len(conflicts or []),
            float(confidence),
            confidence_delta,
            _hash_fact_sheet(fact_sheet),
            datetime.now(timezone.utc).isoformat(),
        )

        try:
            conn.execute(
                """
                INSERT INTO drafts (
                    draft_id, parent_draft_id, version, kind, content,
                    verdict, conflicts_json, conflicts_count, confidence,
                    confidence_delta, fact_sheet_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            conn.commit()
        except sqlite3.DatabaseError:
            return ""

        return draft_id
    finally:
        try:
            conn.close()
        except sqlite3.DatabaseError:
            pass


def get_history(
    draft_id: str, *, db_path: Path | None = None
) -> list[dict[str, Any]]:
    """Walk the parent chain. Returns versions oldest→newest.

    Each entry is a dict mirroring the schema columns. Empty list if the
    draft_id isn't found or the DB can't be opened.
    """
    if not draft_id:
        return []

    try:
        conn = connect(db_path)
    except (sqlite3.DatabaseError, OSError):
        return []

    try:
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        cursor_id: str | None = draft_id
        # Cap at 64 hops as a paranoia guard against cycles introduced by
        # manual DB edits. Real lineages are 1-3 deep.
        for _ in range(64):
            if not cursor_id or cursor_id in seen:
                break
            seen.add(cursor_id)
            cur = conn.execute(
                "SELECT * FROM drafts WHERE draft_id = ?", (cursor_id,)
            )
            row = cur.fetchone()
            if row is None:
                break
            entry = dict(row)
            # Re-hydrate conflicts_json for callers that want structured data.
            raw = entry.get("conflicts_json")
            if raw:
                try:
                    entry["conflicts"] = json.loads(raw)
                except (TypeError, ValueError):
                    entry["conflicts"] = []
            else:
                entry["conflicts"] = []
            chain.append(entry)
            cursor_id = entry.get("parent_draft_id")

        # Walked newest→oldest; flip so callers get chronological order.
        chain.reverse()
        return chain
    finally:
        try:
            conn.close()
        except sqlite3.DatabaseError:
            pass
