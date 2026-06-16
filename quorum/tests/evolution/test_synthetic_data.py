"""Tests for :class:`SyntheticDatasetStore`.

Uses plain ``asyncio.run`` (no pytest-asyncio dependency) so the suite runs
under a vanilla ``pip install pytest``. Each test gets a fresh tmp_path so
the corpus file is hermetic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from quorum.core.consensus import ConsensusResult
from quorum.evolution.synthetic_data import SyntheticDatasetStore
from quorum.providers.base import ModelResponse


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_result(
    answer: str = "The answer is 42.",
    confidence: float = 0.92,
    models: list[ModelResponse] | None = None,
) -> ConsensusResult:
    if models is None:
        models = [
            ModelResponse(name="anthropic", response=answer, weight=0.6),
            ModelResponse(name="openai", response=answer, weight=0.4),
        ]
    return ConsensusResult(
        answer=answer,
        confidence=confidence,
        models=models,
        embedding_confidence=confidence,
    )


def _store(tmp_path: Path) -> SyntheticDatasetStore:
    return SyntheticDatasetStore(path=tmp_path / "synth.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------- #
# maybe_ingest gating
# --------------------------------------------------------------------------- #


def test_ingest_high_confidence_writes_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _make_result(confidence=0.97)

    ingested = asyncio.run(
        store.maybe_ingest(
            "What is the meaning of life?",
            result,
            user_id="u-1",
            opt_in=True,
            min_confidence=0.85,
        )
    )
    assert ingested is True

    rows = _read_jsonl(store._path)
    assert len(rows) == 1
    row = rows[0]
    assert row["prompt"] == "What is the meaning of life?"
    assert row["answer"] == "The answer is 42."
    assert row["confidence"] == pytest.approx(0.97)
    assert row["user_id"] == "u-1"
    assert row["prompt_hash"] and len(row["prompt_hash"]) == 16
    assert "ts" in row and isinstance(row["ts"], str)
    # models list carries name + weight per provider with no error
    assert {m["name"] for m in row["models"]} == {"anthropic", "openai"}


def test_ingest_low_confidence_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _make_result(confidence=0.5)

    ingested = asyncio.run(
        store.maybe_ingest(
            "How do I sort a list in Python?",
            result,
            user_id="u-1",
            opt_in=True,
            min_confidence=0.85,
        )
    )
    assert ingested is False
    assert _read_jsonl(store._path) == []


def test_ingest_requires_opt_in(tmp_path: Path) -> None:
    """Default-deny: opt_in=False must never persist, even at high confidence."""
    store = _store(tmp_path)
    result = _make_result(confidence=0.99)

    ingested = asyncio.run(
        store.maybe_ingest(
            "Will this be saved?",
            result,
            user_id="u-1",
            opt_in=False,
        )
    )
    assert ingested is False
    assert _read_jsonl(store._path) == []


def test_ingest_dedup_by_prompt_hash(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _make_result(confidence=0.95)

    async def run() -> tuple[bool, bool, bool]:
        a = await store.maybe_ingest(
            "Duplicate prompt", result, user_id="u-1", opt_in=True
        )
        b = await store.maybe_ingest(
            "Duplicate prompt", result, user_id="u-1", opt_in=True
        )
        c = await store.maybe_ingest(
            "Different prompt", result, user_id="u-1", opt_in=True
        )
        return a, b, c

    a, b, c = asyncio.run(run())
    assert a is True
    assert b is False  # dedup hit
    assert c is True

    rows = _read_jsonl(store._path)
    assert len(rows) == 2
    assert {r["prompt"] for r in rows} == {"Duplicate prompt", "Different prompt"}


def test_ingest_empty_answer_skipped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _make_result(answer="   ", confidence=0.99)
    ingested = asyncio.run(
        store.maybe_ingest("anything", result, user_id="u-1", opt_in=True)
    )
    assert ingested is False


def test_ingest_oversized_prompt_skipped(tmp_path: Path) -> None:
    from quorum.core.consensus import MAX_PROMPT_BYTES

    store = _store(tmp_path)
    result = _make_result(confidence=0.99)
    huge = "x" * (MAX_PROMPT_BYTES + 1)
    ingested = asyncio.run(
        store.maybe_ingest(huge, result, user_id="u-1", opt_in=True)
    )
    assert ingested is False


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def test_stats_empty_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stats = asyncio.run(store.stats())
    assert stats["total_examples"] == 0
    assert stats["by_user"] == {}
    assert stats["date_range"] == {"min": None, "max": None}


def test_stats_counts_by_user(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _make_result(confidence=0.95)

    async def run() -> None:
        await store.maybe_ingest("q1", result, user_id="alice", opt_in=True)
        await store.maybe_ingest("q2", result, user_id="alice", opt_in=True)
        await store.maybe_ingest("q3", result, user_id="bob", opt_in=True)
        await store.maybe_ingest("q4", result, user_id=None, opt_in=True)

    asyncio.run(run())
    stats = asyncio.run(store.stats())
    assert stats["total_examples"] == 4
    assert stats["by_user"]["alice"] == 2
    assert stats["by_user"]["bob"] == 1
    assert stats["by_user"]["_anon_"] == 1
    assert stats["date_range"]["min"] is not None
    assert stats["date_range"]["max"] is not None
    assert stats["date_range"]["min"] <= stats["date_range"]["max"]


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def test_export_anonymized_strips_user_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _make_result(confidence=0.95)

    async def run() -> int:
        await store.maybe_ingest("q1", result, user_id="alice", opt_in=True)
        await store.maybe_ingest("q2", result, user_id="bob", opt_in=True)
        return await store.export_jsonl(tmp_path / "exported.jsonl", anonymize=True)

    n = asyncio.run(run())
    assert n == 2

    rows = _read_jsonl(tmp_path / "exported.jsonl")
    assert len(rows) == 2
    for row in rows:
        assert "user_id" not in row
        # prompt + answer still present
        assert row["prompt"] in {"q1", "q2"}
        assert row["answer"]


def test_export_non_anonymized_keeps_user_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = _make_result(confidence=0.95)

    async def run() -> int:
        await store.maybe_ingest("q1", result, user_id="alice", opt_in=True)
        return await store.export_jsonl(tmp_path / "raw.jsonl", anonymize=False)

    n = asyncio.run(run())
    assert n == 1
    rows = _read_jsonl(tmp_path / "raw.jsonl")
    assert rows[0]["user_id"] == "alice"
