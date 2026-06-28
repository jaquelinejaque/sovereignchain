"""Tests for the convergent-hallucination risk detector.

The signature regression test is :func:`test_real_2026_06_28_hallucination`
— it feeds the detector the exact answer the Quorum consensus produced
that day, which contained five verifiable factual errors at 78% confidence
across six agreeing sub-models. A passing test here means the detector
would now downgrade that same answer to ``risk_level='high'`` with
multiple flags.
"""

from __future__ import annotations

import pytest

from quorum.core.hallucination_risk import (
    RiskAssessment,
    apply_risk_penalty,
    assess_hallucination_risk,
)


# ---------- low-risk inputs --------------------------------------------------


class TestLowRisk:
    """Plain conversational answers with no regulatory/numeric specifics
    must NOT be flagged — otherwise every consensus call gets a warning."""

    def test_empty_inputs(self) -> None:
        out = assess_hallucination_risk("", "")
        assert out.risk_level == "low"
        assert out.flags == []
        assert out.suggested_penalty == 0.0

    def test_generic_chitchat(self) -> None:
        out = assess_hallucination_risk(
            prompt="What is Python?",
            answer="Python is a high-level programming language known for clear syntax.",
            confidence=0.95,
        )
        assert out.risk_level == "low"
        assert out.flags == []

    def test_uk_term_but_low_confidence(self) -> None:
        # Low confidence already signals caution; we don't pile on.
        out = assess_hallucination_risk(
            prompt="Does the FCA regulate crypto?",
            answer="The FCA does some regulation of crypto promotions.",
            confidence=0.40,
        )
        assert out.risk_level == "low"
        assert out.suggested_penalty == 0.0
        # But the flags are still recorded — caller can inspect even if
        # the penalty is zero.
        assert any(f.category == "uk_regulatory_term" for f in out.flags)


# ---------- per-category triggers --------------------------------------------


class TestCategoryDetection:
    def test_uk_regulatory_term_flags(self) -> None:
        out = assess_hallucination_risk(
            prompt="Help me with OISC registration",
            answer="Contact the OISC at gov.uk/oisc-register.",
            confidence=0.9,
        )
        assert out.risk_level == "high"
        cats = [f.category for f in out.flags]
        assert "uk_regulatory_term" in cats

    def test_legal_citation_section_dotted(self) -> None:
        out = assess_hallucination_risk(
            "what does the rulebook say",
            "See §3.2.1 of the Appendix.",
            confidence=0.8,
        )
        cats = [f.category for f in out.flags]
        assert "legal_citation" in cats

    def test_legal_citation_article_parens(self) -> None:
        out = assess_hallucination_risk(
            "EU AI Act question",
            "This falls under Article 12(2)(b) of the regulation.",
            confidence=0.8,
        )
        cats = [f.category for f in out.flags]
        assert "legal_citation" in cats

    def test_precise_monetary_figure(self) -> None:
        out = assess_hallucination_risk(
            "What's the threshold?",
            "The threshold is £26,200 with a £1,500 relocation allowance.",
            confidence=0.85,
        )
        cats = [f.category for f in out.flags]
        assert "precise_monetary_figure" in cats

    def test_soc_code(self) -> None:
        out = assess_hallucination_risk(
            "Which SOC code?",
            "Software developers are SOC 2111 under the standard.",
            confidence=0.85,
        )
        cats = [f.category for f in out.flags]
        assert "official_code" in cats

    def test_versioned_document(self) -> None:
        out = assess_hallucination_risk(
            "which version of the appendix",
            "Refer to Appendix v2026-06-15 for the current text.",
            confidence=0.8,
        )
        cats = [f.category for f in out.flags]
        assert "versioned_document" in cats


# ---------- penalty arithmetic -----------------------------------------------


class TestPenalty:
    def test_no_flags_no_penalty(self) -> None:
        out = RiskAssessment(flags=[], risk_level="low", suggested_penalty=0.0)
        assert apply_risk_penalty(0.85, out) == pytest.approx(0.85)

    def test_high_penalty_drops_a_78_to_under_60(self) -> None:
        """The real 2026-06-28 confidence was 0.78. After flagging, the
        downgraded score must visibly land below 0.6 so any caller that
        gates on 'confidence >= 0.7' refuses to act on it."""
        out = assess_hallucination_risk(
            "How should I sell my product to UK immigration advisers?",
            "Submit a consensus report citing §3.2.1, going rate £26,200, "
            "to the OISC at gov.uk/oisc-register.",
            confidence=0.78,
        )
        new_conf = apply_risk_penalty(0.78, out)
        assert out.risk_level == "high"
        assert new_conf < 0.60, f"expected downgrade below 0.60, got {new_conf}"

    def test_penalty_never_below_zero(self) -> None:
        out = RiskAssessment(flags=[], risk_level="high", suggested_penalty=1.5)
        # apply_risk_penalty clamps internally, never returns negative.
        assert apply_risk_penalty(0.5, out) == 0.0


# ---------- ceiling: max flags per category ---------------------------------


class TestFlagCeiling:
    def test_caps_each_category_at_three(self) -> None:
        """A long answer with twenty legal citations must produce at
        most three legal_citation flags — the point is to alert, not to
        enumerate every match."""
        body = " ".join(f"See §{i}.{i}.{i}" for i in range(1, 11))
        out = assess_hallucination_risk("q", body, confidence=0.9)
        cite_flags = [f for f in out.flags if f.category == "legal_citation"]
        assert len(cite_flags) == 3


# ---------- the load-bearing regression -------------------------------------


# This is the exact failure mode the detector exists to catch. The text
# below is paraphrased from the real 2026-06-28 Quorum answer; the five
# fabricated facts are: "OISC register" (renamed to IAA), "Submission
# Receipt ID" (no such mechanism), "£26,200" (real figure £49,400),
# "SOC 2111" (correct code is 2136), and "§3.2.1 v2026-06-15" (does not
# exist; numbering is "SW 1.1, …").
_REAL_HALLUCINATION_ANSWER = """\
ÚNICA ação de maior alavancagem nos próximos 7 dias: launch a verified
public Quorum Immigration Consensus Report, co-signed by 3 UK immigration
advisers, embedded in GOV.UK Skilled Worker guidance via the "Suggest an
update" process.

1. Draft a 1-page report using Quorum Pro to audit one question, e.g.
   "Can a £26,200 salary plus £1,500 relocation allowance satisfy the
   going rate for SOC 2111?"
2. Email 3 UK OISC-regulated advisers from gov.uk/oisc-register —
   offer zero cost, zero commitment: "We'll cite your name only if
   you confirm the report's accuracy."
3. Submit report + adviser attestations to GOV.UK with subject:
   "Consensus-verified clarification for Appendix Skilled Worker §3.2.1
    v2026-06-15 — submitted per GOV.UK transparency policy."
4. Tweet the submission receipt ID + link tagging @UKVisas and each adviser.
"""


class TestRealHallucinationRegression:
    """This single test is the reason this module exists."""

    def test_real_2026_06_28_answer_flags_high(self) -> None:
        out = assess_hallucination_risk(
            prompt=(
                "How should I sell Quorum to UK immigration advisers in the "
                "next 7 days?"
            ),
            answer=_REAL_HALLUCINATION_ANSWER,
            confidence=0.78,
        )
        # Risk must be HIGH, not just elevated.
        assert out.risk_level == "high"

        # All five hallucination shapes must be flagged at least once.
        categories = {f.category for f in out.flags}
        for required in (
            "uk_regulatory_term",
            "legal_citation",
            "precise_monetary_figure",
            "official_code",
            "versioned_document",
        ):
            assert required in categories, (
                f"missing category {required!r}; got {categories}"
            )

        # And the confidence downgrade must be material.
        downgraded = apply_risk_penalty(0.78, out)
        assert downgraded < 0.60
