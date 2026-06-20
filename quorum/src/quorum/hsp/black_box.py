"""HSP Black Box — tamper-evident audit log with hash chain.

EU AI Act Article 14 ("Logs traceable in chronological order") +
SOC2 CC7.2 (system monitoring) compliance primitive.

Architecture:
  - SQLite append-only table audit_log
  - Each row: (id, prev_hash, payload_json, this_hash, created_at)
  - this_hash = SHA256(prev_hash + payload_json + created_at_iso)
  - Chain is verifiable: walk from id=1 forward, re-hash each, compare
  - Any tampered/deleted/inserted row breaks the chain at that point
  - Best-effort write: NEVER raises (audit failure shouldn't break consensus)
  - 0o600 file permissions (single-user threat model — Forensic+ would need WORM FS)

Why hash chain not signature:
  HMAC requires key management. Hash chain proves ordering + integrity
  without secret material — auditor verifies offline by recomputing.
  Combine with weekly archive + checksum-on-archive for stronger guarantee.

Usage:
  from quorum.hsp.black_box import append, verify_chain, export_jsonl
  append({"event": "consensus", "query_hash": "abc...", "confidence": 0.85})
  ok, broken_at = verify_chain()
  export_jsonl(since_iso="2026-01-01T00:00:00Z", out_path="/tmp/audit.jsonl")
"""

from __future__ import annotations
import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path.home() / ".quorum" / "audit_chain.db"
_DB_PATH = Path(os.getenv("QUORUM_AUDIT_DB", str(_DEFAULT_PATH)))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    prev_hash    TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    this_hash    TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_created ON audit_log(created_at);
"""

_GENESIS_HASH = "0" * 64  # SHA-256 sized zero hash for first record


def _open() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    try:
        os.chmod(_DB_PATH, 0o600)
    except Exception:
        pass
    return conn


def _hash(prev_hash: str, payload_json: str, created_at: str) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(b"|")
    h.update(payload_json.encode("utf-8"))
    h.update(b"|")
    h.update(created_at.encode("utf-8"))
    return h.hexdigest()


def append(payload: dict[str, Any]) -> str | None:
    """Append payload to the audit chain. Returns this_hash or None on failure.

    Never raises — audit failure must NOT poison the consensus response path.
    """
    try:
        conn = _open()
        try:
            cur = conn.execute("SELECT this_hash FROM audit_log ORDER BY id DESC LIMIT 1")
            row = cur.fetchone()
            prev_hash = row[0] if row else _GENESIS_HASH
            created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            # Canonical JSON: sort keys, no whitespace — so hash is deterministic
            payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            this_hash = _hash(prev_hash, payload_json, created_at)
            conn.execute(
                "INSERT INTO audit_log(prev_hash, payload_json, this_hash, created_at) VALUES(?,?,?,?)",
                (prev_hash, payload_json, this_hash, created_at),
            )
            conn.commit()
            return this_hash
        finally:
            conn.close()
    except Exception as e:
        logger.debug("audit_log append failed (%s); chain not extended", e)
        return None


def verify_chain() -> tuple[bool, int | None]:
    """Walk the chain forward, recomputing each hash. Returns (ok, broken_at_id).

    broken_at_id is the id where mismatch was found, or None if chain is intact.
    """
    try:
        conn = _open()
        try:
            prev_hash = _GENESIS_HASH
            for row in conn.execute(
                "SELECT id, prev_hash, payload_json, this_hash, created_at FROM audit_log ORDER BY id ASC"
            ):
                row_id, row_prev, payload_json, row_this, created_at = row
                if row_prev != prev_hash:
                    return False, row_id
                expected = _hash(prev_hash, payload_json, created_at)
                if expected != row_this:
                    return False, row_id
                prev_hash = row_this
            return True, None
        finally:
            conn.close()
    except Exception as e:
        logger.warning("audit_log verify failed (%s)", e)
        return False, -1


def export_jsonl(since_iso: str | None = None, out_path: str | None = None) -> int:
    """Export audit rows to JSONL. Returns count exported."""
    try:
        conn = _open()
        try:
            if since_iso:
                cur = conn.execute(
                    "SELECT id, prev_hash, payload_json, this_hash, created_at FROM audit_log WHERE created_at >= ? ORDER BY id ASC",
                    (since_iso,),
                )
            else:
                cur = conn.execute(
                    "SELECT id, prev_hash, payload_json, this_hash, created_at FROM audit_log ORDER BY id ASC"
                )
            out = out_path or str(Path.home() / ".quorum" / f"audit_export_{int(time.time())}.jsonl")
            n = 0
            with open(out, "w", encoding="utf-8") as f:
                for row in cur:
                    record = {
                        "id": row[0],
                        "prev_hash": row[1],
                        "payload": json.loads(row[2]),
                        "this_hash": row[3],
                        "created_at": row[4],
                    }
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
                    n += 1
            try:
                os.chmod(out, 0o444)  # primitive WORM — make export read-only
            except Exception:
                pass
            logger.info("audit_log exported %d rows to %s", n, out)
            return n
        finally:
            conn.close()
    except Exception as e:
        logger.warning("audit_log export failed (%s)", e)
        return 0


def stats() -> dict[str, Any]:
    """Quick summary for /quorum-audit status."""
    try:
        conn = _open()
        try:
            row = conn.execute(
                "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM audit_log"
            ).fetchone()
            return {
                "count": row[0],
                "first_at": row[1],
                "last_at": row[2],
                "db_path": str(_DB_PATH),
                "db_size_bytes": _DB_PATH.stat().st_size if _DB_PATH.exists() else 0,
            }
        finally:
            conn.close()
    except Exception as e:
        return {"error": str(e)}


__all__ = ["append", "verify_chain", "export_jsonl", "stats"]
