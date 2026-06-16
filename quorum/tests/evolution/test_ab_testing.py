# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for the prompt-template A/B store (``ABTestStore``).

These tests exercise the *new* surface added on top of the existing
Loop-8 ``ABTestRunner``: ``record_experiment``, ``report_winner``,
``get_active_winner`` (with Wilson lower bound), and ``stats``.

WHY a separate test file rather than co-locating with ABTestRunner's smoke
tests: the runner does Welch/chi-square testing of evolution policies; the
store does Wilson-bounded prompt-template ranking. Different invariants,
different failure modes — keeping them apart makes a regression here
attributable to the new code without cross-talk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from quorum.evolution.ab_testing import (
    ABTestStore,
    _wilson_lower_bound,
)


@dataclass
class _FakeResult:
    """Minimal duck-typed ConsensusResult — just enough surface for record_experiment.

    WHY a fake instead of the real ConsensusResult: keeps the test hermetic
    (no provider auto-discovery, no embedding backend) and exercises the
    store's defensive ``getattr`` path that the production code relies on.
    """

    embedding_confidence: float = 0.0
    confidence: float = 0.0
    answer: str = ""


@pytest.fixture
def store(tmp_path: Path) -> ABTestStore:
    """Fresh on-disk store per test under tmp_path/ab_tests.db."""
    return ABTestStore(db_path=tmp_path / "ab_tests.db")


# --------------------------------------------------------------------------- #
# Wilson lower bound — pure math sanity
# --------------------------------------------------------------------------- #


def test_wilson_lower_bound_known_anchors() -> None:
    """3/3 should rank BELOW 38/50 — the canonical motivating example."""
    lb_streak = _wilson_lower_bound(3, 3)
    lb_record = _wilson_lower_bound(38, 50)
    assert lb_streak < lb_record, (lb_streak, lb_record)
    # Sanity: zero-trials → zero bound (no evidence).
    assert _wilson_lower_bound(0, 0) == 0.0
    # Sanity: 0/n stays at 0 (never won).
    assert _wilson_lower_bound(0, 10) == 0.0
    # Sanity: a perfect 100/100 still has a lower bound strictly below 1.0
    # (the whole point of "lower bound").
    assert _wilson_lower_bound(100, 100) < 1.0


# --------------------------------------------------------------------------- #
# record_experiment
# --------------------------------------------------------------------------- #


async def test_record_experiment_persists_row_and_returns_id(
    store: ABTestStore,
) -> None:
    """A recorded experiment shows up in stats with no decision attached."""
    result_a = _FakeResult(embedding_confidence=0.8)
    result_b = _FakeResult(embedding_confidence=0.6)
    exp_id = await store.record_experiment(
        prompt_a="explain X simply",
        prompt_b="explain X technically",
        result_a=result_a,
        result_b=result_b,
        prompt_template_id="tpl_explain",
    )
    assert isinstance(exp_id, str) and len(exp_id) == 32

    s = await store.stats()
    assert s["total_experiments"] == 1
    assert s["decided_experiments"] == 0


async def test_record_experiment_rejects_empty_prompts(store: ABTestStore) -> None:
    """Empty prompts are a programmer error, not a runtime input — raise."""
    with pytest.raises(ValueError):
        await store.record_experiment(
            prompt_a="", prompt_b="b", result_a=_FakeResult(), result_b=_FakeResult()
        )
    with pytest.raises(ValueError):
        await store.record_experiment(
            prompt_a="a", prompt_b="", result_a=_FakeResult(), result_b=_FakeResult()
        )


async def test_record_experiment_falls_back_to_legacy_confidence(
    store: ABTestStore,
) -> None:
    """When ``embedding_confidence`` is missing, ``confidence`` is used.

    The store accepts duck-typed results, but the v0.0.1 ConsensusResult
    only had ``.confidence`` — the fallback path must still record cleanly.
    """

    class _LegacyResult:
        confidence = 0.42  # noqa: RUF012

    exp_id = await store.record_experiment(
        prompt_a="a",
        prompt_b="b",
        result_a=_LegacyResult(),
        result_b=_LegacyResult(),
        prompt_template_id="tpl_legacy",
    )
    assert isinstance(exp_id, str)
    s = await store.stats()
    assert s["total_experiments"] == 1


# --------------------------------------------------------------------------- #
# report_winner
# --------------------------------------------------------------------------- #


async def test_report_winner_updates_row(store: ABTestStore) -> None:
    """A reported winner flips a row from undecided to decided."""
    exp_id = await store.record_experiment(
        prompt_a="a",
        prompt_b="b",
        result_a=_FakeResult(),
        result_b=_FakeResult(),
        prompt_template_id="tpl",
    )
    await store.report_winner(exp_id, winner="a", source="user")
    s = await store.stats()
    assert s["decided_experiments"] == 1
    assert s["win_rates_per_arm"]["a"]["wins"] == 1


async def test_report_winner_rejects_invalid_arm(store: ABTestStore) -> None:
    exp_id = await store.record_experiment(
        prompt_a="a",
        prompt_b="b",
        result_a=_FakeResult(),
        result_b=_FakeResult(),
    )
    with pytest.raises(ValueError):
        await store.report_winner(exp_id, winner="c", source="user")  # type: ignore[arg-type]


async def test_report_winner_rejects_invalid_source(store: ABTestStore) -> None:
    exp_id = await store.record_experiment(
        prompt_a="a",
        prompt_b="b",
        result_a=_FakeResult(),
        result_b=_FakeResult(),
    )
    with pytest.raises(ValueError):
        await store.report_winner(exp_id, winner="a", source="bot")  # type: ignore[arg-type]


async def test_report_winner_unknown_id_raises(store: ABTestStore) -> None:
    """Reporting on an id that doesn't exist must be loud, not silent.

    Silent acceptance would hide a class of bug where the dashboard sends
    verdicts referencing a recreated database — the votes would just vanish.
    """
    with pytest.raises(KeyError):
        await store.report_winner("does_not_exist", winner="a", source="user")


# --------------------------------------------------------------------------- #
# get_active_winner — the Wilson-bounded ranking
# --------------------------------------------------------------------------- #


async def _seed_decided(
    store: ABTestStore, template_id: str, a_wins: int, b_wins: int, ties: int = 0
) -> None:
    """Helper: create + decide ``a_wins`` 'a' / ``b_wins`` 'b' / ``ties`` 'tie'."""
    for _ in range(a_wins):
        eid = await store.record_experiment(
            prompt_a="a",
            prompt_b="b",
            result_a=_FakeResult(),
            result_b=_FakeResult(),
            prompt_template_id=template_id,
        )
        await store.report_winner(eid, winner="a", source="user")
    for _ in range(b_wins):
        eid = await store.record_experiment(
            prompt_a="a",
            prompt_b="b",
            result_a=_FakeResult(),
            result_b=_FakeResult(),
            prompt_template_id=template_id,
        )
        await store.report_winner(eid, winner="b", source="user")
    for _ in range(ties):
        eid = await store.record_experiment(
            prompt_a="a",
            prompt_b="b",
            result_a=_FakeResult(),
            result_b=_FakeResult(),
            prompt_template_id=template_id,
        )
        await store.report_winner(eid, winner="tie", source="user")


async def test_get_active_winner_returns_none_below_min_n(
    store: ABTestStore,
) -> None:
    """Fewer than min_n decided experiments → no winner (insufficient evidence)."""
    await _seed_decided(store, "tpl_small", a_wins=2, b_wins=0)
    assert await store.get_active_winner("tpl_small") is None


async def test_get_active_winner_picks_clear_winner_with_enough_n(
    store: ABTestStore,
) -> None:
    """Strong imbalance over the minimum sample → 'a' wins."""
    await _seed_decided(store, "tpl_clear", a_wins=20, b_wins=2)
    assert await store.get_active_winner("tpl_clear") == "a"


async def test_get_active_winner_wilson_beats_streak(store: ABTestStore) -> None:
    """The motivating Wilson invariant: an established record beats a hot streak.

    Template T1 has 3 wins out of 3 for B (raw 1.0 but only 3 trials).
    Template T2 has 38 wins out of 50 for A (raw 0.76 but 50 trials).
    Both templates are checked independently; we assert each one's verdict
    follows the Wilson lower bound, not the naive rate.
    """
    await _seed_decided(store, "t1_streak", a_wins=0, b_wins=3)
    await _seed_decided(store, "t2_record", a_wins=38, b_wins=12)

    # T1 has only 3 decided, below the min_n=5 default → None.
    assert await store.get_active_winner("t1_streak") is None
    # T2 has 50 decided, A's Wilson lower bound clearly above B's → "a".
    assert await store.get_active_winner("t2_record") == "a"

    # Same templates with a relaxed min_n=3: T1's "raw 1.0" should still
    # lose to T2's "raw 0.76" when compared head-to-head on Wilson bound.
    # We can't compare across templates directly via the API (correctly —
    # templates are isolated by design), but we can verify T1's bound is
    # numerically below T2's via the pure-math helper.
    lb_t1 = _wilson_lower_bound(3, 3)
    lb_t2 = _wilson_lower_bound(38, 50)
    assert lb_t1 < lb_t2


async def test_get_active_winner_returns_none_on_tie_bound(
    store: ABTestStore,
) -> None:
    """When A and B have identical decided counts, the bounds tie → None."""
    await _seed_decided(store, "tpl_tie", a_wins=10, b_wins=10)
    assert await store.get_active_winner("tpl_tie") is None


async def test_get_active_winner_rejects_empty_template(store: ABTestStore) -> None:
    with pytest.raises(ValueError):
        await store.get_active_winner("")


async def test_get_active_winner_ties_count_toward_n_not_wins(
    store: ABTestStore,
) -> None:
    """A 'tie' vote increments neither A nor B's wins, but does count in n.

    This is the conservatism that makes "tied A/B" not falsely promote
    either arm: the Wilson bound for both shrinks as n grows without
    either arm racking up wins, so an all-ties stream returns None.
    """
    await _seed_decided(store, "tpl_allties", a_wins=0, b_wins=0, ties=10)
    assert await store.get_active_winner("tpl_allties") is None


# --------------------------------------------------------------------------- #
# stats — aggregate shape
# --------------------------------------------------------------------------- #


async def test_stats_shape_and_counts(store: ABTestStore) -> None:
    """Stats has the documented shape and counts add up across arms."""
    await _seed_decided(store, "tpl_x", a_wins=6, b_wins=4, ties=2)
    # Also record one undecided experiment to verify total vs decided diverge.
    await store.record_experiment(
        prompt_a="a",
        prompt_b="b",
        result_a=_FakeResult(),
        result_b=_FakeResult(),
        prompt_template_id="tpl_x",
    )

    s = await store.stats()
    assert s["total_experiments"] == 13
    assert s["decided_experiments"] == 12
    arms = s["win_rates_per_arm"]
    assert arms["a"]["wins"] == 6
    assert arms["b"]["wins"] == 4
    assert arms["tie"]["wins"] == 2
    # Wilson lower bound for A on 6/12 should be a positive number < 0.5.
    assert 0.0 < arms["a"]["wilson_lb"] < 0.5
    # Per-template winner map includes our template.
    assert "tpl_x" in s["current_winner_per_template"]


async def test_stats_empty_store(store: ABTestStore) -> None:
    """An empty store reports zero everywhere without crashing on div-by-zero."""
    s = await store.stats()
    assert s["total_experiments"] == 0
    assert s["decided_experiments"] == 0
    assert s["win_rates_per_arm"]["a"]["rate"] == 0.0
    assert s["win_rates_per_arm"]["a"]["wilson_lb"] == 0.0
    assert s["current_winner_per_template"] == {}


# --------------------------------------------------------------------------- #
# In-memory mode — verifies the ":memory:" branch of _connect()
# --------------------------------------------------------------------------- #


async def test_in_memory_store_persists_within_instance() -> None:
    """A ``:memory:`` store keeps state across calls within one instance.

    Mainly exercising the shared-connection branch of _connect() since the
    other tests all use on-disk DBs.
    """
    mem_store = ABTestStore(db_path=":memory:")
    exp_id = await mem_store.record_experiment(
        prompt_a="a",
        prompt_b="b",
        result_a=_FakeResult(),
        result_b=_FakeResult(),
        prompt_template_id="tpl_mem",
    )
    await mem_store.report_winner(exp_id, winner="b", source="auto")
    s = await mem_store.stats()
    assert s["total_experiments"] == 1
    assert s["decided_experiments"] == 1
    assert s["win_rates_per_arm"]["b"]["wins"] == 1
