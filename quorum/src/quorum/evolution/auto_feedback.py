"""Auto-feedback staging for implicit RLHF signals.

Stages per-call payloads keyed by a deterministic call_id; if no explicit
feedback arrives within the TTL window, callers treat the staged entry as an
implicit thumbs-up and forward it to ``RLHFTracker.record_feedback``.

Do NOT import from ``quorum.evolution.rlhf`` here — the caller injects the
tracker to avoid a circular import.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_DEFAULT_PATH = Path.home() / ".quorum" / "pending_feedback.json"


def emit_call_id(prompt: str, user_id: str, ts_ms: int | None = None) -> str:
    """Return a 16-char sha256-derived id for a (user, prompt, time) tuple."""
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    raw = f"{user_id}|{ts_ms}|{prompt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


class PendingStore:
    """File-backed pending-feedback store, safe across threads and processes."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else _DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load_locked(self, fh) -> dict[str, Any]:
        fh.seek(0)
        data = fh.read()
        if not data:
            return {"v": _SCHEMA_VERSION, "items": {}}
        try:
            doc = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("pending_feedback corrupted; resetting")
            return {"v": _SCHEMA_VERSION, "items": {}}
        doc.setdefault("v", _SCHEMA_VERSION)
        doc.setdefault("items", {})
        return doc

    def _atomic_write(self, doc: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as out:
            json.dump(doc, out, ensure_ascii=False, indent=2)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, self.path)

    def _with_lock(self, mutate):
        # Use a separate sidecar lock file so concurrent os.replace() on the
        # data file doesn't invalidate the held file descriptor.
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.touch(exist_ok=True)
        with open(lock_path, "r+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                # Re-open the data file inside the lock so each writer sees
                # the latest inode (it may have been replaced by another holder).
                if self.path.exists():
                    with open(self.path, "r", encoding="utf-8") as data_fh:
                        doc = self._load_locked(data_fh)
                else:
                    doc = {"v": _SCHEMA_VERSION, "items": {}}
                result = mutate(doc)
                self._atomic_write(doc)
                return result
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    def add(self, call_id: str, payload: dict) -> None:
        def _mut(doc):
            entry = dict(payload)
            # Preserve caller-provided staged_at (useful for tests/replay);
            # default to now() if not supplied.
            entry.setdefault("staged_at", time.time())
            doc["items"][call_id] = entry
            logger.info("staged pending feedback call_id=%s", call_id)
        self._with_lock(_mut)

    def pop(self, call_id: str) -> dict | None:
        return self._with_lock(lambda doc: doc["items"].pop(call_id, None))

    def get(self, call_id: str) -> dict | None:
        return self._with_lock(lambda doc: doc["items"].get(call_id))

    def all(self) -> dict[str, dict]:
        return self._with_lock(lambda doc: dict(doc["items"]))

    def expire_and_thumbs_up(self, max_age_sec: int = 900) -> list[tuple[str, dict]]:
        """Atomically harvest+remove entries older than TTL.

        Returns list of (call_id, payload) tuples for entries whose
        ``staged_at`` is at least ``max_age_sec`` seconds in the past.
        The entries are removed from the store as part of the same locked
        write — callers can treat each tuple as an implicit thumbs-up signal
        without re-locking.
        """
        def _mut(doc):
            now = time.time()
            harvested: list[tuple[str, dict]] = []
            for cid, p in list(doc["items"].items()):
                if now - float(p.get("staged_at", now)) >= max_age_sec:
                    harvested.append((cid, dict(p)))
                    del doc["items"][cid]
            if harvested:
                logger.info("expired %d pending feedback entries (implicit +1)", len(harvested))
            return harvested
        return self._with_lock(_mut)
