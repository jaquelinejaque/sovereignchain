# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sovereign Chain / Quorum contributors
"""Tests for ``CompetitionStore`` — ELO-style pairwise ranking.

WHAT THIS FILE PROVES
=====================
The loop converges. After many observations where model A's response is
consistently closer to the canonical (consensus) answer than model C's,
A's ELO rating should pull strictly above C's by a margin that's far
too large to be coincidence — proving the pairwise battles + ELO
update actually do something useful.

We avoid hitting any real provider; the test uses a hand-built list of
``ModelResponse`` objects per round so the similarity ordering is fully
deterministic and the assertion below is a real claim about the math,
not about an LLM's behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quorum.evolution.competition import (
    _ELO_DEFAULT_RATING,
    CompetitionStore,
    _derive_pairwise_battles,
    _jaccard_similarity,
)
from quorum.providers.base import ModelResponse


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def store(tmp_path: Path) -> CompetitionStore:
    """Fresh on-disk store under tmp_path/competition.db.

    Using tmp_path keeps every test hermetic — no leaking ELO rows
    between cases via ``~/.quorum/competition.db``.
    """
    return CompetitionStore(db_path=tmp_path / "competition.db")


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_jaccard_similarity_basics() -> None:
    """Identical strings -> 1.0; disjoint -> 0.0; empty -> 0.0."""
    assert _jaccard_similarity("alpha beta gamma", "alpha beta gamma") == 1.0
    assert _jaccard_similarity("alpha beta", "delta epsilon") == 0.0
    assert _jaccard_similarity("", "anything") == 0.0
    # Partial overlap = |intersection| / |union|.
    sim = _jaccard_similarity("a b c", "b c d")
    assert 0.0 < sim < 1.0
    assert sim == pytest.approx(2 / 4)


def test_derive_pairwise_battles_orders_by_similarity_to_consensus() -> None:
    """The response closer to the consensus wins the pair against the farther one."""
    consensus = "the answer is forty two"
    responses = [
        ModelResponse(name="A", response="the answer is forty two"),
        ModelResponse(name="B", response="the answer is forty"),
        ModelResponse(name="C", response="potatoes are nice"),
    ]
    battles = _derive_pairwise_battles(responses, consensus)
    # A is identical (best), B is closer than C; expected winners:
    # (A,B), (A,C), (B,C).
    assert ("A", "B") in battles
    assert ("A", "C") in battles
    assert ("B", "C") in battles
    assert len(battles) == 3


def test_derive_pairwise_battles_skips_ties_and_errored_responses() -> None:
    """Errored responses are excluded; equal-similarity pairs are dropped."""
    consensus = "alpha beta gamma"
    responses = [
        ModelResponse(name="A", response="alpha beta gamma"),
        ModelResponse(name="B", response="alpha beta gamma"),  # tie with A
        ModelResponse(name="C", response="", error="timeout"),  # excluded
    ]
    battles = _derive_pairwise_battles(responses, consensus)
    # A vs B is a tie (both perfect similarity) -> skipped; C excluded
    # -> no battles at all.
    assert battles == []


# --------------------------------------------------------------------------- #
# Store ops
# --------------------------------------------------------------------------- #


async def test_default_rating_when_absent(store: CompetitionStore) -> None:
    """Brand-new (model, class) returns the ELO unseen-player default."""
    rating, games = await store.get_rating("never-seen", "general")
    assert rating == _ELO_DEFAULT_RATING
    assert games == 0


async def test_single_battle_moves_ratings_symmetrically(
    store: CompetitionStore,
) -> None:
    """Winner gains; loser loses the SAME amount; magnitude bounded by K."""
    new_w, new_l = await store.observe_battle("general", winner="A", loser="C")
    # At equal priors the expected score is 0.5, so K*(1-0.5) = K/2 = 8.0
    # The store uses K=16; this is a load-bearing claim about the math.
    assert new_w == pytest.approx(_ELO_DEFAULT_RATING + 8.0)
    assert new_l == pytest.approx(_ELO_DEFAULT_RATING - 8.0)
    # Symmetry: total rating conserved.
    assert (new_w - _ELO_DEFAULT_RATING) == pytest.approx(
        _ELO_DEFAULT_RATING - new_l
    )


async def test_self_battle_is_noop(store: CompetitionStore) -> None:
    """A model can't beat itself; the row stays at default."""
    pre, pre_games = await store.get_rating("solo", "general")
    a, b = await store.observe_battle("general", "solo", "solo")
    assert a == b == pre
    # And persisted row is untouched.
    post, post_games = await store.get_rating("solo", "general")
    assert post == pre
    assert post_games == pre_games


async def test_record_query_skips_when_under_two_valid(
    store: CompetitionStore,
) -> None:
    """One response = no pairwise battles possible -> applied=0."""
    responses = [ModelResponse(name="A", response="alone")]
    applied = await store.record_query("general", responses, "alone")
    assert applied == 0


async def test_record_query_with_no_consensus_answer_skipped(
    store: CompetitionStore,
) -> None:
    """An empty canonical answer means we have nothing to score against."""
    responses = [
        ModelResponse(name="A", response="hello world"),
        ModelResponse(name="B", response="goodbye"),
    ]
    applied = await store.record_query("general", responses, "")
    assert applied == 0


# --------------------------------------------------------------------------- #
# Rankings
# --------------------------------------------------------------------------- #


async def test_get_rankings_sorted_descending(store: CompetitionStore) -> None:
    """After a clear winner pattern, get_rankings reflects the order."""
    for _ in range(5):
        await store.observe_battle("code", "winner", "loser")
    ranked = await store.get_rankings("code", top_n=10)
    names = [n for n, _ in ranked]
    assert names[0] == "winner"
    assert names[1] == "loser"
    # And the ratings are strictly descending.
    ratings = [r for _, r in ranked]
    assert ratings == sorted(ratings, reverse=True)


async def test_get_rankings_top_n_zero_returns_empty(
    store: CompetitionStore,
) -> None:
    """Defensive: top_n<=0 short-circuits before touching the DB."""
    await store.observe_battle("general", "A", "B")
    assert await store.get_rankings("general", top_n=0) == []


async def test_get_rankings_scoped_per_query_class(
    store: CompetitionStore,
) -> None:
    """ELO is per-class; code wins don't leak into creative rankings."""
    for _ in range(3):
        await store.observe_battle("code", "code-champ", "everyone-else")
    creative = await store.get_rankings("creative", top_n=5)
    # code-champ has no creative-class history -> never appears.
    assert all(name != "code-champ" for name, _ in creative)


# --------------------------------------------------------------------------- #
# CONVERGENCE — the load-bearing claim
# --------------------------------------------------------------------------- #


async def test_convergence_consistent_winner_pulls_ahead(
    store: CompetitionStore,
) -> None:
    """The loop must converge.

    SCENARIO
    --------
    Three models — A, B, C — answer 100 queries in the ``general`` class.
    On every query:
      * A's answer is *identical* to the consensus answer.
      * B's answer is somewhat close.
      * C's answer is consistently far from the consensus.

    After 100 rounds, A should have a clearly higher rating than C —
    proof that the pairwise-battle pipeline actually moves the ELO
    needle in the right direction. Margin > 50 is the spec threshold.
    """
    canonical = "the official consensus answer about widgets"
    a_response = "the official consensus answer about widgets"   # perfect
    b_response = "the consensus answer about widgets"            # close
    c_response = "completely unrelated thoughts on bicycles"     # far

    for _ in range(100):
        responses = [
            ModelResponse(name="A", response=a_response),
            ModelResponse(name="B", response=b_response),
            ModelResponse(name="C", response=c_response),
        ]
        applied = await store.record_query("general", responses, canonical)
        # Three models -> three pairwise battles per query.
        assert applied == 3

    a_rating, a_games = await store.get_rating("A", "general")
    c_rating, c_games = await store.get_rating("C", "general")

    # 100 queries * 2 battles each (A vs B, A vs C; symmetric for C) -> 200 games.
    assert a_games == 200
    assert c_games == 200

    margin = a_rating - c_rating
    assert margin > 50, (
        f"Expected A's rating to lead C's by >50 after 100 rounds; "
        f"got A={a_rating:.1f}, C={c_rating:.1f}, margin={margin:.1f}"
    )

    # And A should be above the unseen-player default, C should be below
    # — a sharper convergence check that catches "both drifted together".
    assert a_rating > _ELO_DEFAULT_RATING
    assert c_rating < _ELO_DEFAULT_RATING

    # Rankings reflect it.
    rankings = await store.get_rankings("general", top_n=3)
    names_in_order = [n for n, _ in rankings]
    assert names_in_order.index("A") < names_in_order.index("C")
