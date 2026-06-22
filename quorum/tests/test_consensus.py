"""Smoke tests for the consensus engine (no real API calls).

These tests exercise the Jaccard lexical fallback in
``quorum.core.consensus._jaccard_fallback`` — the path the engine
takes when no embedding backend is reachable. We deliberately stay
on the lexical path so the suite runs in <100 ms without any HTTP.

Historical note: an earlier revision of consensus.py exposed a
``_score_agreement`` and ``_extract_disagreements`` pair. The
v0.1.x refactor merged them into ``_jaccard_fallback``, which
returns a 4-tuple ``(confidence, weights, disagreement_pairs,
scoring_method)``. These tests are rewritten against the current
shape so the regression suite isn't blocked by dead imports.
"""

from __future__ import annotations

import pytest

from quorum.core.consensus import _jaccard_fallback
from quorum.providers.base import ModelResponse


def _resp(name: str, text: str) -> ModelResponse:
    return ModelResponse(name=name, response=text)


def test_full_agreement_yields_high_confidence():
    """Three identical responses must produce confidence ≈ 1 and
    uniform weights — the equilibrium case the scorer was designed
    around. We compare against >= 0.99 (not == 1.0) because Jaccard
    on equal bags is exactly 1.0; using >= leaves room for the
    pair-average to differ by a float epsilon if the implementation
    is later changed without changing semantics."""
    responses = [
        _resp("a", "the answer is 42"),
        _resp("b", "the answer is 42"),
        _resp("c", "the answer is 42"),
    ]
    confidence, weights, pairs, method = _jaccard_fallback(responses)
    assert confidence >= 0.99
    assert method == "jaccard"
    assert len(weights) == 3
    assert all(abs(w - 1 / 3) < 0.01 for w in weights)
    # Pair list contains disagreements — none here, all sims == 1.
    assert pairs == []


def test_total_disagreement_yields_low_confidence():
    """Three responses with pairwise-disjoint vocabularies have zero
    Jaccard overlap → confidence ~= 0 and every pair appears in the
    disagreement list (sims < 0.5)."""
    responses = [
        _resp("a", "apple banana cherry"),
        _resp("b", "dog elephant fish"),
        _resp("c", "guitar harp piano"),
    ]
    confidence, weights, pairs, method = _jaccard_fallback(responses)
    assert confidence < 0.05
    assert method == "jaccard"
    # All C(3,2) = 3 pairs are flagged as disagreements (sim < 0.5).
    assert len(pairs) == 3


def test_one_dissenter_identified():
    """When two responses cluster and a third diverges, the dissenter
    must show up in the disagreement pair list. We resolve names via
    the indices returned by the scorer so we never depend on order."""
    responses = [
        _resp("agree_a", "the keratin treatment is formaldehyde free"),
        _resp("agree_b", "the keratin treatment is formaldehyde free indeed"),
        _resp("dissent", "completely unrelated nonsense about cats"),
    ]
    confidence, weights, pairs, method = _jaccard_fallback(responses)
    assert method == "jaccard"
    # Resolve indices → names; dissenter participates in every pair
    # where similarity is < 0.5.
    dissenters: set[str] = set()
    for i, j, sim in pairs:
        if sim < 0.5:
            dissenters.add(responses[i].name)
            dissenters.add(responses[j].name)
    assert "dissent" in dissenters


def test_single_response_handled():
    """One-response edge case: the scorer must return confidence 1.0
    and a single-element weight vector summing to 1, with no pairs
    to compare. Without this guard the implementation could crash on
    a 1-of-N fan-out where N-1 providers errored."""
    responses = [_resp("only", "single model output")]
    confidence, weights, pairs, method = _jaccard_fallback(responses)
    assert confidence == 1.0
    assert weights == [1.0]
    assert pairs == []
    assert method == "jaccard"
