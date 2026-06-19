"""Tests for the verify-and-regenerate loop in ``quorum.agents.drafts``.

Context — why this file exists
------------------------------
``regenerate_loop`` is the retry harness that sits between the draft
generators and ``consensus()``. When ``find_conflicts`` flags hallucinated
numbers or invented names in a generated draft, the loop must:

  1. Re-prompt ``consensus`` with a corrections block that names every
     specific conflict (so the second attempt has anchored facts, not
     just a generic "try again").
  2. Stop as soon as a clean draft is produced (zero conflicts).
  3. Cap retries at ``max_attempts`` (default 3) and, when exhausted,
     annotate the final draft with a ``[VERIFICATION]`` footer so the
     human reviewer sees something is unresolved before publishing.

Every test in this file replaces ``quorum.core.consensus.consensus``
(and the symbol re-exported into ``quorum.agents.drafts``) with a
small ``_FakeConsensus`` instance via ``monkeypatch``. Real provider
calls would cost money, require network, and be flaky — the loop's
control flow is what we're pinning here, not provider behaviour.

The shape of the function under test:

    async def regenerate_loop(
        prompt: str,
        fact_sheet: dict,
        *,
        max_attempts: int = 3,
    ) -> dict

returning a dict with at least the keys:

    {
        "text": str,              # final answer (possibly annotated)
        "attempts": int,          # how many consensus calls were made
        "unresolved": int,        # conflicts remaining at the end
    }
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeConsensusResult:
    """Mimics ``quorum.core.consensus.ConsensusResult`` for the loop's needs.

    The loop only reads ``.answer`` (to scan for conflicts) and the
    handful of bookkeeping fields below (passed through to the caller
    so cost/latency accounting still works). Everything else is filler
    that exists purely to satisfy attribute access if the loop happens
    to touch it.
    """

    answer: str
    confidence: float = 1.0
    models: list[Any] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    disagreements: list[str] = field(default_factory=list)
    evolution_signals: dict[str, bool] = field(default_factory=dict)


class _FakeConsensus:
    """Records every call and serves canned answers in order.

    ``answers`` is a list of strings — call 1 returns the first, call 2
    the second, and so on. If the loop calls past the end, the last
    entry is repeated (so a single-element list models a "consensus
    always returns this" provider without needing a generator).
    """

    def __init__(self, answers: list[str]):
        assert answers, "must supply at least one canned answer"
        self.answers = answers
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, prompt: str, **kwargs: Any) -> _FakeConsensusResult:
        # Capture the full call so tests can later assert on what the
        # loop sent on attempt N (notably: did the retry prompt name
        # the specific bad number?).
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        idx = min(len(self.calls) - 1, len(self.answers) - 1)
        return _FakeConsensusResult(answer=self.answers[idx])


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


# Fact sheet matches the structure ``find_conflicts`` reads:
#   - ``provider_count`` / ``provider_class_count`` for numeric checks
#   - ``competitor_names`` for invented-name checks
# A draft saying "14 providers" matches; "99 LLMs" or "7 models" does not.
_FACT_SHEET = {
    "provider_count": 14,
    "provider_class_count": 14,
    "competitor_names": ["OpenRouter Fusion", "OrcaRouter"],
}

_PROMPT = "Write a one-paragraph blurb describing Quorum for a dev audience."


def _install_fake_consensus(monkeypatch: pytest.MonkeyPatch, fake: _FakeConsensus) -> None:
    """Patch every binding the loop might import ``consensus`` through.

    Module-level ``from quorum.core.consensus import consensus`` binds
    the name into ``quorum.agents.drafts`` at import time, so patching
    only the source module would miss that re-bind. We patch both
    locations to make the test robust to either import style.
    """
    import quorum.core.consensus as core_consensus
    import quorum.agents.drafts as drafts

    monkeypatch.setattr(core_consensus, "consensus", fake, raising=True)
    monkeypatch.setattr(drafts, "consensus", fake, raising=False)


def _run(coro):
    """Run an async coroutine in a fresh event loop (no pytest-asyncio dep)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. Clean first draft → no retry
# ---------------------------------------------------------------------------


def test_no_retry_when_zero_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean first draft must short-circuit the loop after one call.

    "Clean" here means ``find_conflicts`` returns ``[]`` against the
    fact sheet. The loop must NOT call ``consensus`` a second time
    "just to be sure" — that would double cost and latency for no
    information gain.
    """
    fake = _FakeConsensus(
        answers=["Quorum has 14 providers wired together for stronger answers."]
    )
    _install_fake_consensus(monkeypatch, fake)

    from quorum.agents.drafts import regenerate_loop

    result = _run(regenerate_loop(_PROMPT, _FACT_SHEET, max_attempts=3))

    assert result["attempts"] == 1, (
        f"clean draft must stop after the first call; got {result['attempts']}"
    )
    assert result["unresolved"] == 0, (
        f"clean draft must report zero unresolved conflicts; got {result['unresolved']}"
    )
    assert len(fake.calls) == 1, (
        f"consensus must be invoked exactly once; got {len(fake.calls)} calls"
    )
    # The final text must be the (clean) draft itself, NOT annotated —
    # there is nothing to annotate.
    assert "[VERIFICATION]" not in result["text"], (
        "no [VERIFICATION] footer is allowed when conflicts == 0"
    )


# ---------------------------------------------------------------------------
# 2. Retry recovers after one bad attempt
# ---------------------------------------------------------------------------


def test_retry_with_corrections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad draft on call 1, clean on call 2 → ``attempts==2, unresolved==0``.

    This is the happy-path retry: the loop sees "99 LLMs" on attempt 1,
    re-prompts with corrections, and accepts the corrected "14
    providers wired" on attempt 2 without burning the third attempt.
    """
    fake = _FakeConsensus(
        answers=[
            "Quorum has 99 LLMs running in parallel.",            # conflict
            "Quorum has 14 providers wired in parallel.",         # clean
        ]
    )
    _install_fake_consensus(monkeypatch, fake)

    from quorum.agents.drafts import regenerate_loop

    result = _run(regenerate_loop(_PROMPT, _FACT_SHEET, max_attempts=3))

    assert result["attempts"] == 2, (
        f"loop must stop on first clean draft (attempt 2); got {result['attempts']}"
    )
    assert result["unresolved"] == 0, (
        f"second draft was clean; unresolved must be 0; got {result['unresolved']}"
    )
    assert len(fake.calls) == 2, (
        f"consensus must be invoked exactly twice; got {len(fake.calls)} calls"
    )
    # On success, no annotation footer should leak into the final text.
    assert "[VERIFICATION]" not in result["text"], (
        "no [VERIFICATION] footer is allowed when the retry succeeded"
    )


# ---------------------------------------------------------------------------
# 3. Max attempts exhausted → annotate + report unresolved
# ---------------------------------------------------------------------------


def test_max_attempts_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """If consensus never converges, the loop must stop at max_attempts.

    The contract on exhaustion:
      - ``attempts == max_attempts`` (we did not stop early, but we
        also did not loop forever).
      - ``unresolved >= 1`` (at least the conflict that triggered
        every retry is still present).
      - The returned text contains a ``[VERIFICATION]`` footer so the
        human reviewer SEES that the draft is suspect before posting.
    """
    fake = _FakeConsensus(
        answers=["Quorum has 99 LLMs running in parallel."]  # always bad
    )
    _install_fake_consensus(monkeypatch, fake)

    from quorum.agents.drafts import regenerate_loop

    result = _run(regenerate_loop(_PROMPT, _FACT_SHEET, max_attempts=3))

    assert result["attempts"] == 3, (
        f"loop must use full budget when never clean; got {result['attempts']}"
    )
    assert result["unresolved"] >= 1, (
        f"unresolved must be >= 1 after exhausted retries; got {result['unresolved']}"
    )
    assert len(fake.calls) == 3, (
        f"consensus must be invoked exactly max_attempts times; got {len(fake.calls)}"
    )
    assert "[VERIFICATION]" in result["text"], (
        "final text must carry a [VERIFICATION] footer when retries are exhausted "
        "with unresolved conflicts — otherwise a hallucinated draft slips through"
    )


# ---------------------------------------------------------------------------
# 4. Retry prompt must name the specific bad number
# ---------------------------------------------------------------------------


def test_retry_instruction_mentions_specific_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry prompt on attempt 2 must literally contain the bad number.

    Why this matters: a generic "please try again, there were errors"
    re-prompt almost never works — the model has no idea what to
    change, so it rewrites the same hallucination in different words.
    The fix that does work is to embed the offending claim verbatim
    in the corrections block ("you wrote '99 LLMs' but the fact sheet
    says 14 providers"). This test pins that behaviour.

    We deliberately keep the assertion narrow: just check that the
    digits ``99`` appear somewhere in the prompt sent on attempt 2.
    We do NOT pin a specific wording, because the loop's correction
    template is free to evolve.
    """
    fake = _FakeConsensus(
        answers=[
            "Quorum has 99 LLMs running in parallel.",         # attempt 1: bad
            "Quorum has 14 providers wired in parallel.",      # attempt 2: clean
        ]
    )
    _install_fake_consensus(monkeypatch, fake)

    from quorum.agents.drafts import regenerate_loop

    _run(regenerate_loop(_PROMPT, _FACT_SHEET, max_attempts=3))

    assert len(fake.calls) >= 2, (
        f"precondition: loop must have retried at least once; got {len(fake.calls)} calls"
    )

    attempt_2_prompt = fake.calls[1]["prompt"]
    assert "99" in attempt_2_prompt, (
        "retry prompt on attempt 2 must literally contain the bad number '99' so "
        "the model can correct the specific hallucination; "
        f"got prompt:\n{attempt_2_prompt!r}"
    )
