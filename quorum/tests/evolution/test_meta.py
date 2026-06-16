# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for the per-(loop, query_class) online learning surface on
``MetaLearner`` — specifically ``observe()`` and ``recommend_loops()``.

The existing weekly-batch path is exercised by the in-module smoke tests;
this file proves the *new* online API behaves correctly under a synthetic
workload where one loop helps and one does not.

WHY 100 synthetic observations:
    Threshold for exclusion is _MIN_SAMPLES_FOR_EXCLUSION = 5, so we need
    well above that to remove the risk of a flaky boundary pass. 100
    leaves plenty of headroom while staying sub-second on SQLite.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from quorum.evolution.meta import MetaLearner


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Isolated DB per test — prevents cross-test bleed via the default
    ``~/.quorum/meta.db`` path that production would use."""
    return tmp_path / "meta.db"


def test_recommend_loops_cold_start_returns_candidates(tmp_db: Path) -> None:
    """Cold start with no history → return every candidate (enable all).

    Quorum-validated design: starving an unproven loop creates a self-
    fulfilling prophecy. With zero evidence, exploration > exploitation.
    """
    learner = MetaLearner(db_path=tmp_db)
    candidates = ["memory_loop", "fake_loop", "router", "rlhf"]
    recommended = learner.recommend_loops("code", candidate_loops=candidates)
    assert set(recommended) == set(candidates), recommended


def test_recommend_loops_cold_start_no_candidates_returns_empty(tmp_db: Path) -> None:
    """Cold start *without* candidates means 'no opinion' — caller decides."""
    learner = MetaLearner(db_path=tmp_db)
    assert learner.recommend_loops("code") == []


def test_observe_then_recommend_drops_loser_and_keeps_winner(tmp_db: Path) -> None:
    """After 100 observations where memory_loop correlates with high final
    confidence and fake_loop with low, recommend_loops("code") must include
    memory_loop and exclude fake_loop.

    Construction:
      * Half the rounds: memory_loop fires, fake_loop fires, confidence=0.9.
      * Half the rounds: memory_loop does NOT fire, fake_loop fires, conf=0.3.
      * Baseline = mean(0.9, 0.3) ≈ 0.6.
      * memory_loop only ever fires when conf=0.9 → delta = +0.3 (helpful).
      * fake_loop fires in both halves → average delta ≈ 0 with a slight
        negative pull from the low-confidence half because the baseline
        rolls forward; but more importantly it never beats memory_loop.

    To make the test robust against baseline-rolling artifacts, we make
    fake_loop *consistently* anti-correlated: it only fires when the round
    is low-confidence (conf=0.3, well below baseline).
    """
    rng = random.Random(42)
    learner = MetaLearner(db_path=tmp_db)

    high_conf = 0.90
    low_conf = 0.30

    # Burn a few seed rounds with NO loops fired so the class baseline
    # converges to ~0.6 (mean of high and low) before we attribute deltas.
    for _ in range(20):
        conf = rng.choice([high_conf, low_conf])
        learner.observe("code", {"memory_loop": False, "fake_loop": False}, conf)

    # Now inject 100 evaluative rounds.
    for i in range(100):
        if i % 2 == 0:
            learner.observe(
                "code",
                {"memory_loop": True, "fake_loop": False},
                high_conf,
            )
        else:
            learner.observe(
                "code",
                {"memory_loop": False, "fake_loop": True},
                low_conf,
            )

    candidates = ["memory_loop", "fake_loop"]
    recommended = learner.recommend_loops("code", candidate_loops=candidates)

    assert "memory_loop" in recommended, (
        f"memory_loop should be kept (helpful), got {recommended}"
    )
    assert "fake_loop" not in recommended, (
        f"fake_loop should be dropped (harmful), got {recommended}"
    )


def test_observe_persists_across_instances(tmp_db: Path) -> None:
    """Different MetaLearner instances pointing at the same DB share state.

    Production-critical: the orchestrator instantiates MetaLearner per call
    (it's cheap), so persistence must live in SQLite, not the Python object.
    """
    a = MetaLearner(db_path=tmp_db)
    for _ in range(10):
        a.observe("chat", {"router": True}, 0.85)

    b = MetaLearner(db_path=tmp_db)
    # 10 samples > _MIN_SAMPLES_FOR_EXCLUSION (5); router helped (delta > 0)
    # so it must be recommended.
    rec = b.recommend_loops("chat", candidate_loops=["router"])
    assert "router" in rec, rec


def test_unknown_loop_in_candidates_is_kept(tmp_db: Path) -> None:
    """A brand-new loop with zero rows must be recommended (exploration)."""
    learner = MetaLearner(db_path=tmp_db)
    # Build history for an old loop so the class is not in cold start.
    for _ in range(10):
        learner.observe("code", {"old_loop": True}, 0.7)

    rec = learner.recommend_loops(
        "code", candidate_loops=["old_loop", "brand_new_loop"]
    )
    assert "brand_new_loop" in rec, rec


@pytest.mark.asyncio
async def test_async_wrappers_roundtrip(tmp_db: Path) -> None:
    """Async wrappers must produce the same outcome as the sync API."""
    learner = MetaLearner(db_path=tmp_db)
    for _ in range(10):
        await learner.observe_async("code", {"memory_loop": True}, 0.9)
    rec = await learner.recommend_loops_async(
        "code", candidate_loops=["memory_loop", "fake_loop"]
    )
    assert "memory_loop" in rec
    assert "fake_loop" in rec  # fake_loop has no rows → exploration → keep
