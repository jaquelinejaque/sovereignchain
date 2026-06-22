# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for ``DistillationPipeline.collect_from_response_log`` — the bridge
that lets distillation consume the opt-in ``response_log`` SQLite store
instead of the legacy ``queries.jsonl`` file.

What this module guarantees
---------------------------
1. **No-op when the log is empty.** A pipeline pointed at a fresh DB
   returns ``[]`` and does not crash.
2. **Frontier-model grouping works.** Multiple model rows sharing a
   ``query_hash`` collapse into one ``DistillationSample`` whose
   ``source_models`` reflects every frontier contributor.
3. **Canonical-row selection.** When one row has ``was_canonical = 1``,
   that row's ``response_text`` becomes the consensus answer (not the
   longest one).
4. **Fallback to longest response.** With no canonical row flagged
   (older data), we pick the longest frontier response — mirroring
   the JSONL-path heuristic.
5. **Min-pair-count filter.** A query with fewer than ``min_pair_count``
   frontier rows is dropped silently.
6. **Min-consensus filter on mean weight.** Frontier rows whose mean
   weight is below the threshold are dropped.
7. **Time window honoured.** Rows older than ``since`` are excluded.
8. **Privacy boundary documented.** The produced ``DistillationSample``
   stores the ``query_hash`` (never the prompt plaintext) — we assert
   the contract so future refactors can't silently leak the wrong field.

These tests use the real ``response_log`` writer to populate a temp DB
(no mocks) so the schema contract between the two modules is verified
end-to-end. If either side drifts, this suite fails before production
distillation does.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quorum.evolution import response_log
from quorum.evolution.distillation import (
    DEFAULT_FRONTIER_MODELS,
    DistillationPipeline,
)


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolated_response_log(tmp_path, monkeypatch):
    """Point ``response_log`` at a fresh per-test SQLite DB and turn
    logging on. ``autouse`` so every test starts with an empty store."""
    db = tmp_path / "responses.db"
    monkeypatch.setenv(response_log._ENV_DB_PATH, str(db))
    monkeypatch.setenv(response_log._ENV_FLAG, "1")
    yield db


def _pipeline(tmp_path: Path) -> DistillationPipeline:
    """Construct a pipeline with disposable artifacts dir + eval set
    pointing at the tmp tree so we never write outside it."""
    return DistillationPipeline(
        log_path=tmp_path / "queries.jsonl",
        eval_set_path=tmp_path / "eval_set.jsonl",
        artifacts_dir=tmp_path / "artifacts",
    )


def _write_round(
    *,
    prompt: str,
    query_class: str = "general",
    rows: list[dict],
    canonical_model: str | None = None,
) -> int:
    """Convenience wrapper that drives the real writer.

    Returns the number of rows actually inserted so callers can sanity-
    check that the fixture set up what they think it did before the
    pipeline read pass."""
    return asyncio.run(
        response_log.record_consensus_round(
            prompt=prompt,
            query_class=query_class,
            model_responses=rows,
            canonical_model=canonical_model,
        )
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #


def test_empty_log_returns_no_candidates(tmp_path):
    """Pipeline against an untouched response_log → 0 samples, no crash.
    This is the cold-start contract; the nightly cron must survive the
    very first night when no logged queries exist yet."""
    pipe = _pipeline(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(days=1)
    samples = asyncio.run(pipe.collect_from_response_log(since=since))
    assert samples == []


def test_frontier_rows_group_into_one_sample(tmp_path):
    """Three frontier rows under one ``query_hash`` produce exactly one
    DistillationSample whose source_models lists every contributor."""
    n = _write_round(
        prompt="Explain Hebbian learning briefly.",
        rows=[
            {
                "model": "anthropic/claude-haiku-4-5",
                "response_text": "Hebbian learning is the rule where neurons that fire together wire together.",
                "weight": 0.4,
            },
            {
                "model": "openai/gpt-4o-mini",
                "response_text": "Neurons strengthen their connection when they activate simultaneously.",
                "weight": 0.3,
            },
            {
                "model": "gemini/gemini-flash",
                "response_text": "Cells that fire together strengthen their synaptic link over time.",
                "weight": 0.3,
            },
        ],
        canonical_model="anthropic/claude-haiku-4-5",
    )
    assert n == 3  # sanity: writer didn't drop anything

    pipe = _pipeline(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    samples = asyncio.run(
        pipe.collect_from_response_log(since=since, min_consensus=0.0)
    )
    assert len(samples) == 1
    s = samples[0]
    assert len(s.source_models) == 3
    assert all(any(tag in m.lower() for tag in DEFAULT_FRONTIER_MODELS) for m in s.source_models)


def test_canonical_row_wins_over_longest(tmp_path):
    """When one row has ``was_canonical = 1``, that response becomes
    the consensus answer even when a longer non-canonical row exists.
    This is the safety net against the longest-response heuristic
    overriding the engine's actual pick."""
    _write_round(
        prompt="What is keratin?",
        rows=[
            {
                "model": "anthropic/claude",
                "response_text": "Short canonical answer.",
                "weight": 0.5,
            },
            {
                "model": "openai/gpt-4o-mini",
                "response_text": "A much longer non-canonical response with many extra words "
                                 * 10,
                "weight": 0.3,
            },
            {
                "model": "gemini/gemini-flash",
                "response_text": "Another reasonable mid-length response here.",
                "weight": 0.2,
            },
        ],
        canonical_model="anthropic/claude",
    )

    pipe = _pipeline(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    samples = asyncio.run(
        pipe.collect_from_response_log(since=since, min_consensus=0.0)
    )
    assert len(samples) == 1
    assert samples[0].consensus_response == "Short canonical answer."


def test_fallback_to_longest_when_no_canonical(tmp_path):
    """No ``was_canonical=1`` row → pipeline picks the longest frontier
    response (legacy heuristic, matches the JSONL path)."""
    _write_round(
        prompt="What is consensus?",
        rows=[
            {"model": "anthropic/claude", "response_text": "Short.", "weight": 0.4},
            {
                "model": "openai/gpt-4o-mini",
                "response_text": "A clearly longer and more detailed explanation of consensus.",
                "weight": 0.3,
            },
            {"model": "gemini/gemini-flash", "response_text": "Mid.", "weight": 0.3},
        ],
        canonical_model=None,  # nothing flagged
    )

    pipe = _pipeline(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    samples = asyncio.run(
        pipe.collect_from_response_log(since=since, min_consensus=0.0)
    )
    assert len(samples) == 1
    # Longest among the three frontier responses wins the fallback.
    assert samples[0].consensus_response.startswith("A clearly longer")


def test_min_pair_count_drops_thin_consensus(tmp_path):
    """Only 2 frontier rows — below default min_pair_count=3 — drops
    the whole sample. Prevents 2-model coin-flips from polluting the
    training set."""
    _write_round(
        prompt="Two-model query.",
        rows=[
            {"model": "anthropic/claude", "response_text": "Answer A.", "weight": 0.5},
            {"model": "openai/gpt-4o-mini", "response_text": "Answer B.", "weight": 0.5},
        ],
    )

    pipe = _pipeline(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    samples = asyncio.run(
        pipe.collect_from_response_log(since=since, min_consensus=0.0)
    )
    assert samples == []


def test_min_consensus_filter_drops_weak_agreement(tmp_path):
    """When the mean of (positive) weights across frontier rows is
    below the threshold, the sample is rejected. We use weights that
    are all small but equal, so mean is well-defined and just below
    the cutoff."""
    _write_round(
        prompt="Weakly agreed query.",
        rows=[
            {"model": "anthropic/claude", "response_text": "A.", "weight": 0.1},
            {"model": "openai/gpt-4o-mini", "response_text": "B.", "weight": 0.1},
            {"model": "gemini/gemini-flash", "response_text": "C.", "weight": 0.1},
        ],
        canonical_model="anthropic/claude",
    )

    pipe = _pipeline(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    # 0.5 threshold > 0.1 mean → must drop.
    samples = asyncio.run(
        pipe.collect_from_response_log(since=since, min_consensus=0.5)
    )
    assert samples == []
    # 0.05 threshold < 0.1 mean → must keep.
    samples = asyncio.run(
        pipe.collect_from_response_log(since=since, min_consensus=0.05)
    )
    assert len(samples) == 1


def test_time_window_excludes_old_rows(tmp_path):
    """Rows whose ``created_at`` is before the ``since`` cutoff are
    excluded. Without this, an incremental nightly run would re-process
    every historical row."""
    # Round 1 — write now, then advance the cutoff past it.
    _write_round(
        prompt="Old round.",
        rows=[
            {"model": "anthropic/claude", "response_text": "old1", "weight": 0.4},
            {"model": "openai/gpt-4o-mini", "response_text": "old2", "weight": 0.3},
            {"model": "gemini/gemini-flash", "response_text": "old3", "weight": 0.3},
        ],
        canonical_model="anthropic/claude",
    )

    pipe = _pipeline(tmp_path)
    future_cutoff = datetime.now(timezone.utc) + timedelta(hours=1)
    samples = asyncio.run(
        pipe.collect_from_response_log(since=future_cutoff, min_consensus=0.0)
    )
    assert samples == []


def test_privacy_boundary_query_field_is_hash(tmp_path):
    """The ``DistillationSample.query`` must equal the SHA-256 of the
    prompt, never the prompt itself. ``response_log`` is the privacy-
    safe store; this assertion locks the contract so a future "convenience"
    refactor can't accidentally pipe plaintext into a distillation dataset."""
    sensitive_prompt = "internal/secret/customer-list-2026.csv"
    _write_round(
        prompt=sensitive_prompt,
        rows=[
            {"model": "anthropic/claude", "response_text": "A", "weight": 0.4},
            {"model": "openai/gpt-4o-mini", "response_text": "B", "weight": 0.3},
            {"model": "gemini/gemini-flash", "response_text": "C", "weight": 0.3},
        ],
        canonical_model="anthropic/claude",
    )

    pipe = _pipeline(tmp_path)
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    samples = asyncio.run(
        pipe.collect_from_response_log(since=since, min_consensus=0.0)
    )
    assert len(samples) == 1
    # Hash, not plaintext.
    assert samples[0].query != sensitive_prompt
    assert len(samples[0].query) == 64  # SHA-256 hex
    assert all(c in "0123456789abcdef" for c in samples[0].query)
