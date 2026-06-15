"""Smoke tests for the consensus engine (no real API calls)."""

from __future__ import annotations

import pytest

from quorum.core.consensus import _extract_disagreements, _score_agreement
from quorum.providers.base import ModelResponse


def _resp(name: str, text: str) -> ModelResponse:
    return ModelResponse(name=name, response=text)


def test_full_agreement_yields_high_confidence():
    responses = [
        _resp("a", "the answer is 42"),
        _resp("b", "the answer is 42"),
        _resp("c", "the answer is 42"),
    ]
    confidence, weights = _score_agreement(responses)
    assert confidence > 0.99
    assert all(abs(w - 1 / 3) < 0.01 for w in weights)


def test_total_disagreement_yields_low_confidence():
    responses = [
        _resp("a", "apple banana cherry"),
        _resp("b", "dog elephant fish"),
        _resp("c", "guitar harp piano"),
    ]
    confidence, weights = _score_agreement(responses)
    assert confidence < 0.05


def test_one_dissenter_identified():
    responses = [
        _resp("agree_a", "the keratin treatment is formaldehyde free"),
        _resp("agree_b", "the keratin treatment is formaldehyde free indeed"),
        _resp("dissent", "completely unrelated nonsense about cats"),
    ]
    _, weights = _score_agreement(responses)
    dissenters = _extract_disagreements(responses, weights)
    assert "dissent" in dissenters


def test_single_response_handled():
    responses = [_resp("only", "single model output")]
    confidence, weights = _score_agreement(responses)
    assert confidence == 1.0
    assert weights == [1.0]
