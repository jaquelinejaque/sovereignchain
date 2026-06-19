"""Tests for the agent fact-sheet + draft-verifier loop.

Context — why this file exists
------------------------------
Quorum's marketing-draft agents (LinkedIn, Twitter, Show HN, email,
marketplace listing) have a recurring failure mode: the LLM happily
invents numbers. "Quorum unifies 99 LLMs", "Quorum unifies 7 LLMs",
"Quorum unifies dozens of providers" — none of which match what the
provider registry actually wires today.

The fix is a two-step guard:

1. ``build_fact_sheet()`` introspects the live provider registry and
   returns the ground-truth counts the draft is allowed to claim.
2. ``find_conflicts()`` scans the generated draft for numeric claims
   about provider count and flags any that contradict the fact sheet,
   so ``annotate()`` can append a human-readable footer before the
   draft is shown for approval.

These tests pin that contract WITHOUT making any real API calls — the
provider registry is exercised in operator/env-var mode but with no
keys set, then ``build_fact_sheet`` is asked to return the *known*
catalogue size rather than the runtime-active size. See the
implementation for the exact split.
"""

from __future__ import annotations

import pytest

from quorum.agents.fact_sheet import build_fact_sheet, format_as_prompt_block
from quorum.agents.verifier import annotate, find_conflicts


# ---------------------------------------------------------------------------
# build_fact_sheet
# ---------------------------------------------------------------------------


def test_build_fact_sheet_returns_real_provider_count():
    """The fact sheet must surface a concrete int provider count >= 10.

    The number itself is intentionally not pinned to an exact value here
    because the catalogue grows; ``>= 10`` is the floor implied by the
    current registry (Anthropic 3 + OpenAI 2 + Gemini 1 + Replicate 4 +
    NVIDIA 6 + DeepSeek 2 + Mistral 3 + Cohere 3 + Grok 2 + Zhipu 3 +
    Moonshot 2 + Qwen 4 = 35 wired models across 12 vendor families).
    What we guard against is the regression where the sheet silently
    falls back to ``None`` because key discovery failed.
    """
    sheet = build_fact_sheet()

    assert sheet.provider_count is not None, "fact sheet must populate provider_count"
    assert isinstance(sheet.provider_count, int), (
        f"provider_count must be int, got {type(sheet.provider_count).__name__}"
    )
    assert sheet.provider_count >= 10, (
        f"expected >= 10 wired providers, got {sheet.provider_count}"
    )


# ---------------------------------------------------------------------------
# format_as_prompt_block
# ---------------------------------------------------------------------------


def test_format_block_contains_count():
    """The prompt block injected into the LLM must literally name the count.

    LLMs ignore facts that aren't surfaced verbatim. The block must
    contain both the integer and an unambiguous label ("providers
    wired") so the model is anchored on the right number.
    """
    sheet = build_fact_sheet()
    block = format_as_prompt_block(sheet)

    assert "providers wired" in block, (
        "prompt block must contain literal phrase 'providers wired' so the "
        "draft LLM has an unambiguous anchor for the count"
    )
    assert str(sheet.provider_count) in block, (
        f"prompt block must contain the integer {sheet.provider_count}"
    )


# ---------------------------------------------------------------------------
# find_conflicts
# ---------------------------------------------------------------------------


class _Sheet:
    """Minimal stand-in for the real FactSheet used in conflict tests.

    Using a stub keeps the conflict-detection tests independent from
    whatever the live provider catalogue happens to be on any given
    day. The contract is: ``find_conflicts`` reads
    ``sheet.provider_count`` and nothing else.
    """

    def __init__(self, provider_count: int):
        self.provider_count = provider_count


def test_find_conflicts_catches_inflated_number():
    """A draft claiming a wildly different count must yield a conflict."""
    sheet = _Sheet(provider_count=13)
    draft = "Quorum has 99 LLMs working together for stronger answers."

    conflicts = find_conflicts(draft, sheet)

    assert len(conflicts) == 1, (
        f"expected exactly 1 conflict for '99 vs 13', got {len(conflicts)}: {conflicts}"
    )


def test_find_conflicts_passes_correct_number():
    """A draft with the correct count must produce zero conflicts."""
    sheet = _Sheet(provider_count=13)
    draft = "Quorum has 13 LLMs working together for stronger answers."

    conflicts = find_conflicts(draft, sheet)

    assert conflicts == [], (
        f"expected no conflicts when draft matches fact sheet, got: {conflicts}"
    )


# ---------------------------------------------------------------------------
# annotate
# ---------------------------------------------------------------------------


def test_annotate_appends_footer(tmp_path):
    """When conflicts exist, ``annotate`` must append a footer.

    ``tmp_path`` is accepted so the implementation is free to evolve
    into writing the annotated draft to disk for review; today the
    contract is purely string-in / string-out, and we only assert the
    footer is present.
    """
    sheet = _Sheet(provider_count=13)
    draft = "Quorum has 99 LLMs working together."

    conflicts = find_conflicts(draft, sheet)
    assert conflicts, "precondition: there must be conflicts to annotate"

    annotated = annotate(draft, conflicts)

    assert annotated.startswith(draft), (
        "annotate must preserve the original draft verbatim at the top"
    )
    assert len(annotated) > len(draft), (
        "annotate must append a footer when conflicts are present"
    )
    # Optional smoke: dump the annotated draft to tmp_path so a human
    # reviewing the test run can eyeball the output without rerunning.
    (tmp_path / "annotated.txt").write_text(annotated)
