# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for ``response_log`` — opt-in persistence of raw model outputs.

The contract we verify:

1. **Opt-out is the default.** With ``QUORUM_LOG_RESPONSES`` unset
   no DB is created and the public function is a no-op.
2. **Opt-in writes rows.** With the flag set the rows land in
   ``responses.db`` and survive a reconnect.
3. **PII safety.** The prompt is never persisted in cleartext —
   only its SHA-256.
4. **Failed providers are dropped.** Empty-text responses do not
   pollute the dataset.
5. **Embeddings round-trip.** When the caller passes an ``embedding``
   list, ``export_jsonl`` returns it as ``list[float]`` (not the
   JSON string we use for storage).
6. **Truncation matches consensus.** Responses longer than
   ``MAX_RESPONSE_BYTES`` are clipped, matching the bound the
   embedder already enforces — so the on-disk text is never larger
   than what the rest of Quorum saw.
7. **Retention works.** ``vacuum_older_than`` deletes only rows
   older than the cutoff and leaves recent rows intact.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

import pytest

from quorum.evolution import response_log


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Point every test at a fresh DB so they cannot leak into one another."""
    db = tmp_path / "responses.db"
    monkeypatch.setenv(response_log._ENV_DB_PATH, str(db))
    # Ensure opt-out default for tests that haven't asked otherwise.
    monkeypatch.delenv(response_log._ENV_FLAG, raising=False)
    yield db


def _enable(monkeypatch):
    monkeypatch.setenv(response_log._ENV_FLAG, "1")


def test_default_is_disabled(_isolated_db):
    """With no env flag, ``is_enabled`` is False and writes are no-ops."""
    assert response_log.is_enabled() is False

    async def _run():
        inserted = await response_log.record_consensus_round(
            prompt="never written anywhere",
            query_class="general",
            model_responses=[{"model": "claude", "response_text": "hi"}],
        )
        return inserted

    assert asyncio.run(_run()) == 0
    assert not _isolated_db.exists(), "no DB should be created when disabled"


def test_opt_in_writes_rows(_isolated_db, monkeypatch):
    """With the flag set, rows persist and survive reconnect."""
    _enable(monkeypatch)
    assert response_log.is_enabled() is True

    async def _run():
        return await response_log.record_consensus_round(
            prompt="What is 2+2?",
            query_class="factual",
            model_responses=[
                {
                    "model": "claude-haiku",
                    "response_text": "Four.",
                    "latency_ms": 123.4,
                    "cost_usd": 0.0001,
                    "weight": 0.6,
                },
                {
                    "model": "gpt-4o-mini",
                    "response_text": "2+2 = 4.",
                    "latency_ms": 234.5,
                    "cost_usd": 0.0002,
                    "weight": 0.4,
                },
            ],
            canonical_model="claude-haiku",
        )

    n = asyncio.run(_run())
    assert n == 2

    rows = list(response_log.export_jsonl())
    assert len(rows) == 2
    by_model = {r["model"]: r for r in rows}
    assert by_model["claude-haiku"]["was_canonical"] == 1
    assert by_model["gpt-4o-mini"]["was_canonical"] == 0
    assert by_model["claude-haiku"]["response_text"] == "Four."
    assert by_model["claude-haiku"]["query_class"] == "factual"


def test_prompt_never_stored_plaintext(_isolated_db, monkeypatch):
    """Only the SHA-256 of the prompt is persisted; plaintext is not."""
    _enable(monkeypatch)
    sensitive = "internal/secret/path/to/some/document.pdf"

    asyncio.run(
        response_log.record_consensus_round(
            prompt=sensitive,
            query_class="general",
            model_responses=[{"model": "any", "response_text": "ok"}],
        )
    )

    expected_hash = hashlib.sha256(sensitive.encode("utf-8")).hexdigest()
    rows = list(response_log.export_jsonl())
    assert all(r["query_hash"] == expected_hash for r in rows)
    # And no row anywhere should contain the sensitive substring.
    db_bytes = _isolated_db.read_bytes()
    assert sensitive.encode("utf-8") not in db_bytes


def test_failed_providers_are_skipped(_isolated_db, monkeypatch):
    """Empty ``response_text`` means provider failed — drop it."""
    _enable(monkeypatch)

    asyncio.run(
        response_log.record_consensus_round(
            prompt="anything",
            query_class="general",
            model_responses=[
                {"model": "ok-model", "response_text": "got a reply"},
                {"model": "broken-model", "response_text": ""},  # 429/timeout
                {"model": "also-broken", "response_text": None},  # NoneType
            ],
        )
    )

    rows = list(response_log.export_jsonl())
    models = {r["model"] for r in rows}
    assert models == {"ok-model"}


def test_embeddings_roundtrip(_isolated_db, monkeypatch):
    """When caller passes ``embedding`` we store + return it intact."""
    _enable(monkeypatch)
    vec = [0.1, -0.2, 0.3, 0.4, 0.5]

    asyncio.run(
        response_log.record_consensus_round(
            prompt="embed me",
            query_class="general",
            model_responses=[
                {
                    "model": "claude",
                    "response_text": "answer",
                    "embedding": vec,
                    "embedding_provider": "gemini-embedding-001",
                }
            ],
        )
    )

    rows = list(response_log.export_jsonl())
    assert len(rows) == 1
    r = rows[0]
    assert r["embedding"] == vec
    assert r["embedding_dim"] == 5
    assert r["embedding_provider"] == "gemini-embedding-001"
    assert "embedding_json" not in r, "raw JSON column should not leak to caller"


def test_truncation(_isolated_db, monkeypatch):
    """Responses longer than the cap are clipped on the way in."""
    _enable(monkeypatch)
    huge = "x" * (response_log.MAX_RESPONSE_BYTES + 5_000)

    asyncio.run(
        response_log.record_consensus_round(
            prompt="big",
            query_class="general",
            model_responses=[{"model": "verbose", "response_text": huge}],
        )
    )

    rows = list(response_log.export_jsonl())
    assert len(rows) == 1
    assert len(rows[0]["response_text"]) == response_log.MAX_RESPONSE_BYTES


def test_export_filters_by_model_and_time(_isolated_db, monkeypatch):
    """``export_jsonl`` honours ``model`` and ``since`` filters."""
    _enable(monkeypatch)
    # Two writes with a tiny gap so timestamps differ.
    asyncio.run(
        response_log.record_consensus_round(
            prompt="q1",
            query_class="general",
            model_responses=[{"model": "alpha", "response_text": "a1"}],
        )
    )
    time.sleep(0.01)
    cutoff = time.time()
    time.sleep(0.01)
    asyncio.run(
        response_log.record_consensus_round(
            prompt="q2",
            query_class="general",
            model_responses=[
                {"model": "alpha", "response_text": "a2"},
                {"model": "beta", "response_text": "b1"},
            ],
        )
    )

    after_cutoff = list(response_log.export_jsonl(since=cutoff))
    assert len(after_cutoff) == 2
    assert {r["response_text"] for r in after_cutoff} == {"a2", "b1"}

    alphas = list(response_log.export_jsonl(model="alpha"))
    assert {r["response_text"] for r in alphas} == {"a1", "a2"}


def test_stats_overview(_isolated_db, monkeypatch):
    """Cheap stats summary returns the right counts."""
    _enable(monkeypatch)
    asyncio.run(
        response_log.record_consensus_round(
            prompt="q",
            query_class="general",
            model_responses=[
                {"model": "m1", "response_text": "a"},
                {"model": "m2", "response_text": "b"},
            ],
        )
    )
    s = response_log.stats()
    assert s["rows"] == 2
    assert s["distinct_models"] == 2
    assert s["distinct_queries"] == 1
    assert s["rows_with_embedding"] == 0


def test_vacuum_only_deletes_old(_isolated_db, monkeypatch):
    """``vacuum_older_than`` deletes the old row, leaves the new one."""
    _enable(monkeypatch)
    # Write an "old" row by manually setting created_at far in the past.
    import sqlite3
    with response_log._connect(_isolated_db) as conn:
        conn.execute(
            """INSERT INTO response_log
               (query_hash, query_class, model, response_text,
                latency_ms, cost_usd, weight, was_canonical,
                created_at)
               VALUES (?, 'general', 'old-model', 'old-text',
                       0, 0, 0, 0, ?)""",
            (hashlib.sha256(b"old").hexdigest(), time.time() - 365 * 24 * 3600),
        )

    asyncio.run(
        response_log.record_consensus_round(
            prompt="fresh",
            query_class="general",
            model_responses=[{"model": "new-model", "response_text": "fresh-text"}],
        )
    )

    deleted = response_log.vacuum_older_than(seconds=30 * 24 * 3600)
    assert deleted == 1

    remaining = list(response_log.export_jsonl())
    assert len(remaining) == 1
    assert remaining[0]["model"] == "new-model"
