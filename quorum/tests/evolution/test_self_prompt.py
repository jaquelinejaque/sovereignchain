# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for the query-time self-prompting loop (PromptRewriter).

Distinct from ``test_self_prompt_optimizer.py`` (if/when added) — those
cover the long-horizon system-prompt bandit. THIS file covers the
single-call rewrite-and-retry path.

Design under test (Quorum-validated, 2026-06-17):
  * rewrite(prompt, confidence, query_class) returns None when confidence
    is already above threshold; otherwise calls the rewriter provider
    and returns the composed prompt with the original preserved.
  * log_rewrite persists (original, rewritten, before_conf, after_conf,
    delta, query_class) to SQLite so the meta-learner can pick up the
    signal. ``delta = after - before`` is precomputed.
  * The plumbing into ``consensus()`` uses these primitives so the
    smoke-test scenario (first pass 0.45, rewrite, second pass 0.82)
    flows end-to-end with no real provider keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from quorum.evolution.self_prompt import (
    DEFAULT_REWRITE_CONFIDENCE_THRESHOLD,
    PromptRewriter,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubResponse:
    """Mimics the shape PromptRewriter reads from a Provider.complete()."""

    response: str
    error: str = ""


class _StubRewriter:
    """Provider stub that prepends ``[CLARIFIED] `` to the user prompt.

    The PromptRewriter wraps the user's *rewrite-instruction* prompt around
    the original query; we slice off the instruction preamble by returning
    a fixed deterministic transform of the marker. Tests assert on the
    final composed prompt, not on the rewriter's intermediate text.
    """

    name = "stub-rewriter"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> Any:
        self.calls.append(prompt)
        # The rewriter is asked to produce ONLY the rewritten query (no
        # preamble). We mimic that contract here.
        return _StubResponse(response="[CLARIFIED] What is the answer?")


class _BrokenRewriter:
    """Stub that simulates an outage — PromptRewriter must swallow."""

    name = "broken"

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> Any:
        raise RuntimeError("simulated rewriter outage")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Path:
    """Isolated SQLite path per test — keeps ``~/.quorum/self_prompt.db``
    untouched and prevents cross-test bleed."""
    return tmp_path / "self_prompt.db"


# ---------------------------------------------------------------------------
# rewrite() — confidence gating
# ---------------------------------------------------------------------------


async def test_rewrite_returns_none_above_threshold(tmp_db: Path) -> None:
    """If confidence is already above the threshold, we must NOT rewrite.

    Why this matters: a rewrite is a paid LLM call. Triggering it when the
    first pass already won wastes money AND can degrade an already-good
    answer by feeding the second pass a noisier prompt.
    """
    stub = _StubRewriter()
    rw = PromptRewriter(
        db_path=tmp_db,
        confidence_threshold=0.6,
        rewriter_provider=stub,
    )
    out = await rw.rewrite("Original query.", current_confidence=0.85,
                           query_class="general")
    assert out is None
    assert stub.calls == []  # no provider call attempted


async def test_rewrite_returns_composed_prompt_below_threshold(
    tmp_db: Path,
) -> None:
    """Low-confidence pass MUST trigger the rewriter and compose the
    final prompt as ORIGINAL + CLARIFIED markers."""
    stub = _StubRewriter()
    rw = PromptRewriter(
        db_path=tmp_db,
        confidence_threshold=0.6,
        rewriter_provider=stub,
    )
    out = await rw.rewrite(
        "What is X?", current_confidence=0.45, query_class="technical",
    )
    assert out is not None
    assert "ORIGINAL_QUERY" in out
    assert "What is X?" in out
    assert "CLARIFIED_QUERY" in out
    assert "[CLARIFIED] What is the answer?" in out
    assert len(stub.calls) == 1
    # Rewriter instruction must carry the confidence + class context so
    # the prompt actually targets THIS query's failure mode.
    assert "0.45" in stub.calls[0]
    assert "technical" in stub.calls[0]


async def test_rewrite_returns_none_on_provider_outage(tmp_db: Path) -> None:
    """An outage during rewriting must degrade gracefully (None) so the
    consensus engine can keep its first-pass result instead of crashing."""
    rw = PromptRewriter(
        db_path=tmp_db,
        confidence_threshold=0.6,
        rewriter_provider=_BrokenRewriter(),
    )
    out = await rw.rewrite("Q?", current_confidence=0.2)
    assert out is None


async def test_rewrite_returns_none_when_no_provider(
    tmp_db: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No Anthropic key, no OpenAI key, no explicit rewriter → None.

    The consensus engine relies on this contract to detect "no rewriter
    available" without inspecting environment variables itself.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rw = PromptRewriter(db_path=tmp_db, confidence_threshold=0.6)
    out = await rw.rewrite("Q?", current_confidence=0.2)
    assert out is None


# ---------------------------------------------------------------------------
# log_rewrite() — persistence + delta
# ---------------------------------------------------------------------------


async def test_log_rewrite_persists_delta(tmp_db: Path) -> None:
    """log_rewrite MUST record (before, after, delta) so the meta-learner
    can compute per-query-class ROI without scanning two columns."""
    rw = PromptRewriter(db_path=tmp_db, confidence_threshold=0.6)
    rid = await rw.log_rewrite(
        original="Original query.",
        rewritten="ORIGINAL_QUERY:\nOriginal query.\n\nCLARIFIED_QUERY:\n[CLARIFIED] ...",
        original_confidence=0.45,
        new_confidence=0.82,
        query_class="technical",
        rewriter_name="stub-rewriter",
    )
    assert rid

    row = await rw.get_rewrite(rid)
    assert row is not None
    assert row["original"] == "Original query."
    assert row["original_confidence"] == pytest.approx(0.45)
    assert row["new_confidence"] == pytest.approx(0.82)
    assert row["delta"] == pytest.approx(0.37, abs=1e-9)
    assert row["query_class"] == "technical"
    assert row["rewriter_name"] == "stub-rewriter"


async def test_log_rewrite_records_negative_delta(tmp_db: Path) -> None:
    """Negative delta (rewrite hurt confidence) must be persisted too —
    that signal is exactly what tells the meta-learner to skip rewriting
    for some query classes."""
    rw = PromptRewriter(db_path=tmp_db, confidence_threshold=0.6)
    rid = await rw.log_rewrite(
        original="Q",
        rewritten="Q rewritten",
        original_confidence=0.55,
        new_confidence=0.40,
        query_class="creative",
    )
    row = await rw.get_rewrite(rid)
    assert row is not None
    assert row["delta"] == pytest.approx(-0.15, abs=1e-9)


# ---------------------------------------------------------------------------
# End-to-end scenario from the task spec
# ---------------------------------------------------------------------------


async def test_first_pass_low_then_rewrite_then_second_pass_high(
    tmp_db: Path,
) -> None:
    """Spec scenario: first pass 0.45 → rewriter prepends ``[CLARIFIED] ``
    → second pass 0.82. Assert log_rewrite records the delta correctly.

    This is the integration shape the consensus engine uses: rewrite()
    THEN log_rewrite() with the post-pass confidence. We mirror that
    sequence here without spinning up the full consensus stack.
    """
    stub = _StubRewriter()
    rw = PromptRewriter(
        db_path=tmp_db,
        confidence_threshold=0.6,
        rewriter_provider=stub,
    )

    first_pass_confidence = 0.45
    original_prompt = "What is the answer?"

    rewritten = await rw.rewrite(
        original_prompt,
        current_confidence=first_pass_confidence,
        query_class="general",
    )
    assert rewritten is not None
    assert "[CLARIFIED] What is the answer?" in rewritten

    # Simulate the second pass coming back at 0.82.
    second_pass_confidence = 0.82

    rid = await rw.log_rewrite(
        original=original_prompt,
        rewritten=rewritten,
        original_confidence=first_pass_confidence,
        new_confidence=second_pass_confidence,
        query_class="general",
        rewriter_name=stub.name,
    )

    row = await rw.get_rewrite(rid)
    assert row is not None
    assert row["original_confidence"] == pytest.approx(0.45)
    assert row["new_confidence"] == pytest.approx(0.82)
    assert row["delta"] == pytest.approx(0.37, abs=1e-9)
    assert row["query_class"] == "general"
    assert row["rewriter_name"] == "stub-rewriter"


# ---------------------------------------------------------------------------
# Defensive constructor checks
# ---------------------------------------------------------------------------


def test_constructor_rejects_bad_threshold(tmp_db: Path) -> None:
    with pytest.raises(ValueError):
        PromptRewriter(db_path=tmp_db, confidence_threshold=1.5)
    with pytest.raises(ValueError):
        PromptRewriter(db_path=tmp_db, confidence_threshold=-0.1)


def test_constructor_rejects_bad_max_attempts(tmp_db: Path) -> None:
    with pytest.raises(ValueError):
        PromptRewriter(db_path=tmp_db, max_attempts=0)


def test_default_threshold_matches_design_consensus() -> None:
    """Quorum design called for ~0.6 default — guard against silent drift."""
    assert DEFAULT_REWRITE_CONFIDENCE_THRESHOLD == 0.6
