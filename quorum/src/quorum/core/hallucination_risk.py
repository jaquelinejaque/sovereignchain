"""Convergent-hallucination risk detector.

Why this module exists
----------------------
Observed 2026-06-28: a real consensus query asking the Quorum engine how to
sell itself produced a 78%-confidence answer in which **5 out of 5
verifiable factual claims were wrong**:

  1. Cited a non-existent "GOV.UK Suggest an update receipt ID" mechanism.
  2. Used the old "OISC" name and URL — the regulator renamed to IAA in
     January 2025 and the register lives at a different host.
  3. Quoted a UK Skilled Worker going rate of £26,200 + £1,500 relocation
     for SOC 2111. The real figure is £49,400 under SOC 2136 and there is
     no "relocation allowance" line item at all.
  4. Encouraged advisers to co-sign "zero cost" reports — that is exactly
     the touting pattern banned by IAA Code of Standards 6.3.
  5. Cited "Appendix Skilled Worker §3.2.1 v2026-06-15". The numbering
     scheme is "SW 1.1, SW 2.1, …", §3.2.1 does not exist, and the
     latest revision date is 29 April 2026.

Six of the eleven sub-models agreed with each other inside this fictional
world. The agreement score did exactly what it was designed to do — and
that is the bug, not the symptom. **Semantic agreement between LLMs
trained on overlapping data is not evidence of factual truth, especially
in long-tail regulated domains where the training corpus is thin.**

What this detector does
-----------------------
At the end of the consensus call we run :func:`assess_hallucination_risk`
over the prompt + the answer. If risk flags fire AND ``confidence`` is
above a threshold, the consumer can:

  * downgrade ``confidence`` via :func:`apply_risk_penalty` so callers
    that gate on the score behave more cautiously, AND/OR
  * surface ``hallucination_risk`` to the user so they know to fact-check.

We do NOT block the response. False positives are cheap (the user fact-
checks one extra answer); false negatives are expensive (the user trusts
a confidently-wrong answer about a regulated domain). The detector is
intentionally biased toward *flagging*.

The detector is heuristic, not perfect. It runs offline (no extra LLM
call) and is deterministic, so it is safe to wire into the hot path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger("quorum.core.hallucination_risk")


# ---------------------------------------------------------------------------
# Risk categories — each is a known long-tail / specialised topic where
# all major LLMs share the same training-data gaps and tend to agree on
# the same plausible-but-fabricated answer.
# ---------------------------------------------------------------------------

# UK regulatory bodies that have been renamed, restructured, or had their
# rules change post-2023 (which is around most models' useful cut-off for
# regulatory detail). Hits here are NOT "this answer is wrong" — they are
# "any specific factual claim about this body needs verification".
UK_REGULATORY_TERMS: tuple[str, ...] = (
    "OISC", "IAA", "Immigration Advice Authority",
    "FCA", "Financial Conduct Authority",
    "PRA", "Prudential Regulation Authority",
    "ICO", "Information Commissioner",
    "Ofcom", "Ofgem", "Ofwat",
    "UKVI", "UK Visas and Immigration", "Home Office",
    "Skilled Worker", "going rate", "shortage occupation",
    "Companies House", "HMRC", "Charity Commission",
    "GMC", "NMC", "GDC",  # medical regs
    "SRA", "BSB",  # legal regs
    "MHRA",
)

# Specific patterns that historically produce confident fabrications.
_LEGAL_CITATION_PATTERNS = (
    # "Appendix Foo §3.2.1", "Article 12(2)(b)", "Section 4.5",
    re.compile(r"§\s*\d+(?:\.\d+){1,}"),
    re.compile(r"\bArticle\s+\d+(?:\(\d+\)){1,}", re.IGNORECASE),
    re.compile(r"\bAppendix\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+§", re.IGNORECASE),
    re.compile(r"\bAnnex\s+[A-Z]\s+v\d{4}-\d{2}-\d{2}", re.IGNORECASE),
    # "Code 6.3" "Section 12.1.4"
    re.compile(r"\bCode\s+\d+\.\d+\b"),
    re.compile(r"\bSection\s+\d+(?:\.\d+){1,}\b", re.IGNORECASE),
)

# Specific GBP / USD figures with no rounding — these are exactly the
# shape ("£26,200", "$1,500") that LLMs fabricate to sound authoritative.
_MONETARY_FIGURE_PATTERN = re.compile(r"[£$€]\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?")

# SOC codes, NACE codes, ICD-10 codes — short alphanumeric identifiers
# that LLMs frequently typo (SOC 2111 vs 2136).
_OFFICIAL_CODE_PATTERN = re.compile(r"\bSOC\s+\d{4}\b|\bNACE\s+[A-Z]\d{2,4}\b|\bICD-?10\s+[A-Z]\d{2}", re.IGNORECASE)

# Versioned legal docs with specific dates — "v2026-06-15", "as of 2025-03-01".
_VERSIONED_DOC_PATTERN = re.compile(r"\bv?\d{4}-\d{2}-\d{2}\b")


@dataclass
class RiskFlag:
    """One reason the answer is at elevated risk of convergent hallucination."""

    category: str
    """Short identifier, e.g. ``"uk_regulatory_term"`` or ``"legal_citation"``."""

    evidence: str
    """The substring from the prompt or answer that triggered the flag."""

    detail: str
    """Human-readable explanation surfacable to the end-user."""


@dataclass
class RiskAssessment:
    """Outcome of :func:`assess_hallucination_risk`."""

    flags: list[RiskFlag] = field(default_factory=list)
    """One entry per match. Empty list = no risk detected."""

    risk_level: str = "low"
    """``"low"`` | ``"elevated"`` | ``"high"`` — see :func:`assess_hallucination_risk`."""

    suggested_penalty: float = 0.0
    """Multiplier in ``[0, 1]``. ``apply_risk_penalty`` multiplies
    confidence by ``1 - suggested_penalty``. ``0.0`` = no change."""

    def to_dict(self) -> dict:
        return {
            "risk_level": self.risk_level,
            "suggested_penalty": round(self.suggested_penalty, 3),
            "flags": [
                {"category": f.category, "evidence": f.evidence, "detail": f.detail}
                for f in self.flags
            ],
        }


# Hard cap on how many flags of each category we record. The point is to
# tell the caller "this is risky", not to enumerate every legal citation
# in a 5-page answer.
_MAX_FLAGS_PER_CATEGORY = 3


def _find_terms(text: str, terms: Iterable[str]) -> list[str]:
    """Return the subset of ``terms`` that appear in ``text`` (case-insensitive)
    as whole words. We use word-boundary matching so "OISC" doesn't trigger
    on "OISCONNECT" or similar."""
    lower = text.lower()
    hits: list[str] = []
    for term in terms:
        pat = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pat, lower):
            hits.append(term)
    return hits


def assess_hallucination_risk(
    prompt: str,
    answer: str,
    *,
    confidence: float = 1.0,
) -> RiskAssessment:
    """Score how likely the answer is to be a convergent hallucination.

    Returns a :class:`RiskAssessment`. The detector is intentionally noisy
    on the side of flagging — false positives are cheap (user fact-checks
    one extra answer), false negatives are expensive (user trusts a
    confidently-wrong answer about a regulated domain).

    The ``confidence`` argument is the consensus confidence the caller is
    about to report. We use it to decide between ``"elevated"`` and
    ``"high"`` — a high agreement score on a risky topic is the SPECIFIC
    failure mode we are guarding against (six models agreeing in a
    shared fictional world).

    Risk levels:

    * ``"low"``     — no flags fired, OR flags fired but confidence is
                      below 0.5 so the caller is already cautious.
    * ``"elevated"`` — at least one flag fired AND confidence in [0.5, 0.7).
    * ``"high"``    — at least one flag fired AND confidence >= 0.7. This
                      is the exact pattern that produced the 2026-06-28
                      five-out-of-five hallucination.
    """
    flags: list[RiskFlag] = []

    combined = f"{prompt}\n{answer}"

    # UK regulatory terms
    uk_hits = _find_terms(combined, UK_REGULATORY_TERMS)[:_MAX_FLAGS_PER_CATEGORY]
    for term in uk_hits:
        flags.append(RiskFlag(
            category="uk_regulatory_term",
            evidence=term,
            detail=(
                f"References UK regulator/regime '{term}'. UK regulators "
                "rename, restructure, and revise rules frequently; specific "
                "claims (URLs, body names, rule numbers, fees) should be "
                "verified against the official source before relying on them."
            ),
        ))

    # Legal citations of the form "§3.2.1", "Article 12(2)(b)", etc.
    citation_count = 0
    for pat in _LEGAL_CITATION_PATTERNS:
        for m in pat.finditer(combined):
            if citation_count >= _MAX_FLAGS_PER_CATEGORY:
                break
            flags.append(RiskFlag(
                category="legal_citation",
                evidence=m.group(0),
                detail=(
                    f"Cites '{m.group(0)}'. LLMs frequently fabricate "
                    "section numbers in legal/regulatory text — the "
                    "section may not exist or may have been renumbered."
                ),
            ))
            citation_count += 1

    # Exact monetary figures
    money_seen: set[str] = set()
    for m in _MONETARY_FIGURE_PATTERN.finditer(combined):
        value = m.group(0)
        if value in money_seen or len(money_seen) >= _MAX_FLAGS_PER_CATEGORY:
            continue
        money_seen.add(value)
        flags.append(RiskFlag(
            category="precise_monetary_figure",
            evidence=value,
            detail=(
                f"States exact figure '{value}'. Salaries, thresholds, and "
                "fees change between budget cycles; LLM training cut-offs "
                "make precise current figures one of the most-fabricated "
                "answer shapes. Verify against the official source."
            ),
        ))

    # Official codes (SOC, NACE, ICD)
    for m in _OFFICIAL_CODE_PATTERN.finditer(combined):
        if sum(1 for f in flags if f.category == "official_code") >= _MAX_FLAGS_PER_CATEGORY:
            break
        flags.append(RiskFlag(
            category="official_code",
            evidence=m.group(0),
            detail=(
                f"Names code '{m.group(0)}'. Codes such as SOC, NACE, and "
                "ICD-10 are short, similar-looking strings that LLMs "
                "frequently typo or misattribute (e.g. SOC 2111 vs 2136). "
                "Verify the exact code against the official catalogue."
            ),
        ))

    # Versioned doc references
    for m in _VERSIONED_DOC_PATTERN.finditer(combined):
        if sum(1 for f in flags if f.category == "versioned_document") >= _MAX_FLAGS_PER_CATEGORY:
            break
        flags.append(RiskFlag(
            category="versioned_document",
            evidence=m.group(0),
            detail=(
                f"Cites a document version '{m.group(0)}'. Specific dated "
                "versions are a common fabrication pattern — the document "
                "may exist but a different revision is current."
            ),
        ))

    # Risk level + penalty.
    if not flags:
        return RiskAssessment(flags=[], risk_level="low", suggested_penalty=0.0)

    if confidence < 0.5:
        # Caller is already cautious; no extra penalty.
        return RiskAssessment(
            flags=flags,
            risk_level="low",
            suggested_penalty=0.0,
        )

    if confidence < 0.7:
        return RiskAssessment(
            flags=flags,
            risk_level="elevated",
            suggested_penalty=0.10,
        )

    # confidence >= 0.7 with flags — the 2026-06-28 failure pattern.
    return RiskAssessment(
        flags=flags,
        risk_level="high",
        suggested_penalty=0.25,
    )


def apply_risk_penalty(confidence: float, assessment: RiskAssessment) -> float:
    """Return a penalised confidence in ``[0, 1]``.

    Penalty is multiplicative: ``confidence * (1 - suggested_penalty)``.
    A 25% penalty on a 0.78 score lands at 0.585 — visibly downgraded
    but not pretending the answer is hostile garbage.
    """
    penalty = max(0.0, min(1.0, assessment.suggested_penalty))
    return max(0.0, min(1.0, confidence * (1.0 - penalty)))


__all__ = [
    "RiskFlag",
    "RiskAssessment",
    "assess_hallucination_risk",
    "apply_risk_penalty",
    "UK_REGULATORY_TERMS",
]
