# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for ``quorum.evolution.auto_feedback`` — the implicit-feedback
staging layer that turns ephemeral consensus calls into stable, poppable
records keyed by a deterministic ``call_id``.

Design under test:
  * ``emit_call_id(prompt, user_id, ts_ms)`` is a pure hash — same inputs
    always yield the same id, so a delayed thumbs-up can find its call.
  * ``PendingStore`` is a tiny file-backed mailbox. ``add()`` stages a
    payload; ``pop()`` removes-and-returns it (or ``None`` if absent or
    already popped). Concurrent ``add()`` from multiple threads MUST NOT
    drop writes — atomic-rename or equivalent is required.
  * Each payload carries a ``staged_at`` epoch-second; consumers ask
    ``expire_and_thumbs_up(max_age_sec)`` to harvest entries older than
    the cutoff (auto-positive feedback after the user moved on without
    complaining).
  * The on-disk format is the explicit envelope ``{"v": 1, "items": {...}}``
    — versioned so future migrations don't silently corrupt old stores.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from quorum.evolution.auto_feedback import (
    PendingStore,
    emit_call_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    """Isolated pending-store path per test (no ``~/.quorum`` bleed)."""
    return tmp_path / "pending.json"


# ---------------------------------------------------------------------------
# 1. Deterministic call_id
# ---------------------------------------------------------------------------


def test_emit_call_id_stable() -> None:
    """Same (prompt, user_id, ts_ms) → identical call_id, every time.

    This is the contract that makes implicit feedback possible at all:
    the thumbs-up arrives seconds later with no shared in-memory state,
    so the only way to reconcile is a pure deterministic hash.
    """
    prompt = "what is the capital of france?"
    user_id = "jaqueline"
    ts_ms = 1_750_000_000_000

    a = emit_call_id(prompt, user_id, ts_ms)
    b = emit_call_id(prompt, user_id, ts_ms)
    assert a == b
    assert isinstance(a, str) and a  # non-empty string

    # Any field change → different id (no accidental collisions on the
    # three dimensions we actually key on).
    assert emit_call_id(prompt + " ", user_id, ts_ms) != a
    assert emit_call_id(prompt, user_id + "x", ts_ms) != a
    assert emit_call_id(prompt, user_id, ts_ms + 1) != a


# ---------------------------------------------------------------------------
# 2. add / pop round-trip
# ---------------------------------------------------------------------------


def test_pending_store_add_pop_roundtrip(store_path: Path) -> None:
    """``add`` then ``pop`` returns the payload; a second ``pop`` returns
    ``None``. Pop is destructive by design — implicit feedback must not
    fire twice for the same call."""
    store = PendingStore(store_path)
    call_id = "call-abc"
    payload = {"prompt": "hi", "user_id": "jaq", "staged_at": 1_000.0}

    store.add(call_id, payload)

    popped = store.pop(call_id)
    assert popped is not None
    # Round-trip preserves the staged fields.
    assert popped["prompt"] == "hi"
    assert popped["user_id"] == "jaq"
    assert popped["staged_at"] == 1_000.0

    # Second pop is a no-op.
    assert store.pop(call_id) is None

    # Unknown id also returns None (not a KeyError).
    assert store.pop("never-added") is None


# ---------------------------------------------------------------------------
# 3. Concurrent add — no lost writes
# ---------------------------------------------------------------------------


def test_pending_store_atomic_concurrent(store_path: Path) -> None:
    """Two threads adding *different* call_ids must both persist.

    The naive read-modify-write would lose one entry under a race; the
    spec requires file-locking or atomic rename so this scenario is safe.
    """
    store = PendingStore(store_path)

    n_per_thread = 25
    threads_count = 4
    barrier = threading.Barrier(threads_count)
    errors: list[BaseException] = []

    def _worker(tag: str) -> None:
        try:
            barrier.wait(timeout=5.0)  # maximise overlap
            for i in range(n_per_thread):
                store.add(
                    f"{tag}-{i}",
                    {"prompt": f"p{i}", "user_id": tag, "staged_at": float(i)},
                )
        except BaseException as exc:  # noqa: BLE001 — capture for re-raise
            errors.append(exc)

    threads = [
        threading.Thread(target=_worker, args=(f"t{n}",))
        for n in range(threads_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    assert not errors, f"worker raised: {errors!r}"

    # All distinct ids must be present and individually poppable.
    fresh = PendingStore(store_path)  # re-open to bypass any in-process cache
    for n in range(threads_count):
        for i in range(n_per_thread):
            cid = f"t{n}-{i}"
            assert fresh.pop(cid) is not None, f"lost write: {cid}"


# ---------------------------------------------------------------------------
# 4. Expiry → auto thumbs-up
# ---------------------------------------------------------------------------


def test_expire_and_thumbs_up(store_path: Path) -> None:
    """Payloads older than ``max_age_sec`` are returned and removed;
    younger ones stay put for a future sweep."""
    store = PendingStore(store_path)
    now = time.time()

    # Old enough to expire.
    store.add(
        "old-call",
        {"prompt": "old", "user_id": "jaq", "staged_at": now - 600.0},
    )
    # Just-staged — must NOT expire on this sweep.
    store.add(
        "fresh-call",
        {"prompt": "fresh", "user_id": "jaq", "staged_at": now - 1.0},
    )

    expired = store.expire_and_thumbs_up(max_age_sec=300)

    # expire_and_thumbs_up returns list[tuple[call_id, payload]]
    expired_ids = {call_id for call_id, _ in expired}
    assert "old-call" in expired_ids
    assert "fresh-call" not in expired_ids

    # The fresh one is still in the store (re-pop succeeds).
    assert store.pop("fresh-call") is not None
    # The old one is gone (already harvested).
    assert store.pop("old-call") is None


# ---------------------------------------------------------------------------
# 5. On-disk format is the explicit v1 envelope
# ---------------------------------------------------------------------------


def test_pending_file_format(store_path: Path) -> None:
    """After any ``add``, the file is JSON shaped
    ``{"v": 1, "items": {<call_id>: <payload>}}``.

    Pinning the envelope keeps room for a future ``v: 2`` migration
    (e.g., switching ``items`` from a dict to a list) without ambiguity.
    """
    store = PendingStore(store_path)
    store.add(
        "call-xyz",
        {"prompt": "hello", "user_id": "jaq", "staged_at": 42.0},
    )

    raw = json.loads(store_path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    assert raw.get("v") == 1
    assert isinstance(raw.get("items"), dict)
    assert "call-xyz" in raw["items"]

    entry = raw["items"]["call-xyz"]
    assert entry["prompt"] == "hello"
    assert entry["user_id"] == "jaq"
    assert entry["staged_at"] == 42.0
