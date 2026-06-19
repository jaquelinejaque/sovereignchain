"""Draft verifier — fact-checks Quorum drafts against an authoritative fact_sheet.

Catches the two most common hallucination patterns we've seen in autopilot
drafts:

1. Numeric inflation ("23 LLMs" when we actually support 13).
2. Invented competitor names ("FusionRouter AI") that don't exist.

The verifier is intentionally conservative: it only flags numeric claims that
match a known fact_sheet key, and it only flags Capitalized phrases ending in
specific suspicious suffixes. False negatives are preferred over false positives
— a verifier that cries wolf gets ignored.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Numeric claim: a number (int or decimal) followed by a known unit noun.
# Captures the whole match so we can render it back to the user verbatim.
_NUMERIC_CLAIM_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:LLMs?|models?|loops?|providers?|installs?|users?|%|stars?|customers?|months?|years?)\b",
    re.IGNORECASE,
)

# Just the leading number, used to pull the numeric value out of a claim.
_LEADING_NUMBER_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")

# Capitalized phrase ending in a suspicious suffix (Router / Fusion / AI).
# Requires at least one Capitalized token before the suffix to avoid matching
# a bare "AI" or "Router" sitting on its own.
_COMPETITOR_NAME_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9]+\s+)*[A-Z][a-zA-Z0-9]+\s+(?:Router|Fusion|AI)\b"
    r"|"
    r"\b[A-Z][a-zA-Z0-9]*(?:Router|Fusion|AI)\b"
)

# Maps the unit noun (lowercased, singular) to the fact_sheet key we expect
# to find the authoritative value under. Add entries here as fact_sheet
# vocabulary grows.
_UNIT_TO_FACT_KEY: dict[str, str] = {
    "llm": "provider_count",
    "llms": "provider_count",
    "model": "provider_count",
    "models": "provider_count",
    "provider": "provider_count",
    "providers": "provider_count",
    "loop": "loop_count",
    "loops": "loop_count",
    "install": "install_count",
    "installs": "install_count",
    "user": "user_count",
    "users": "user_count",
    "customer": "customer_count",
    "customers": "customer_count",
    "star": "star_count",
    "stars": "star_count",
    "month": "month_count",
    "months": "month_count",
    "year": "year_count",
    "years": "year_count",
    "%": "percent",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_unit(claim_text: str) -> str | None:
    """Return the lowercased unit noun from a numeric claim, or None."""
    # Strip the leading number + whitespace, what's left is the unit.
    stripped = _LEADING_NUMBER_RE.sub("", claim_text, count=1).strip()
    if not stripped:
        return None
    return stripped.lower()


def _extract_number(claim_text: str) -> float | None:
    m = _LEADING_NUMBER_RE.search(claim_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _numbers_match(claimed: float, expected: Any) -> bool:
    """Treat numeric equality leniently — allow int/float comparison."""
    try:
        return float(claimed) == float(expected)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_conflicts(draft_text: str, fact_sheet: dict) -> list[dict]:
    """Return a list of conflict dicts found in the draft.

    Each conflict dict has the shape:
        {
            "claim_text": str,       # what the draft said
            "expected_value": Any,   # what the fact_sheet says (or None)
            "type": str,             # 'numeric_inflated' | 'name_invented' | 'unverifiable'
        }
    """
    conflicts: list[dict] = []
    seen: set[tuple[str, str]] = set()  # dedupe by (claim_text, type)

    # ---- 1. Numeric claims ------------------------------------------------
    for match in _NUMERIC_CLAIM_RE.finditer(draft_text):
        claim_text = match.group(0)
        unit = _extract_unit(claim_text)
        claimed = _extract_number(claim_text)
        if unit is None or claimed is None:
            continue

        fact_key = _UNIT_TO_FACT_KEY.get(unit)
        if fact_key is None:
            # Unit we don't know how to verify — flag as unverifiable so the
            # author at least eyeballs it.
            key = (claim_text, "unverifiable")
            if key not in seen:
                seen.add(key)
                conflicts.append(
                    {
                        "claim_text": claim_text,
                        "expected_value": None,
                        "type": "unverifiable",
                    }
                )
            continue

        expected = fact_sheet.get(fact_key)
        if expected is None:
            # Fact_sheet doesn't have an entry for this unit — can't verify.
            key = (claim_text, "unverifiable")
            if key not in seen:
                seen.add(key)
                conflicts.append(
                    {
                        "claim_text": claim_text,
                        "expected_value": None,
                        "type": "unverifiable",
                    }
                )
            continue

        if not _numbers_match(claimed, expected):
            key = (claim_text, "numeric_inflated")
            if key not in seen:
                seen.add(key)
                conflicts.append(
                    {
                        "claim_text": claim_text,
                        "expected_value": expected,
                        "type": "numeric_inflated",
                    }
                )

    # ---- 2. Invented competitor names -------------------------------------
    known_competitors = {
        name.strip().lower()
        for name in fact_sheet.get("competitor_names", []) or []
        if isinstance(name, str)
    }

    for match in _COMPETITOR_NAME_RE.finditer(draft_text):
        name = match.group(0).strip()
        # Skip the bare suffix-as-standalone-word case (defensive — regex
        # already guards against it, but cheap to double-check).
        if name.lower() in {"router", "fusion", "ai"}:
            continue
        if name.lower() in known_competitors:
            continue
        key = (name, "name_invented")
        if key not in seen:
            seen.add(key)
            conflicts.append(
                {
                    "claim_text": name,
                    "expected_value": None,
                    "type": "name_invented",
                }
            )

    return conflicts


def annotate_draft(draft_text: str, conflicts: list[dict]) -> str:
    """Return draft_text with inline markers + a verification footer."""
    annotated = draft_text

    # Inline markers — replace first occurrence only per conflict to avoid
    # double-marking when the same claim appears multiple times. Process in
    # descending order of claim length so longer matches win against shorter
    # substrings (e.g. "FusionRouter AI" before "AI").
    for conflict in sorted(
        conflicts, key=lambda c: len(c.get("claim_text", "")), reverse=True
    ):
        claim = conflict.get("claim_text", "")
        if not claim:
            continue
        ctype = conflict.get("type", "unverifiable")
        expected = conflict.get("expected_value")
        if ctype == "numeric_inflated" and expected is not None:
            marker = f"{claim} [CONFLICT: expected {expected}]"
        elif ctype == "name_invented":
            marker = f"{claim} [CONFLICT: name not in fact_sheet]"
        else:
            marker = f"{claim} [CONFLICT: unverifiable]"

        # Only replace the first occurrence — repeated claims share one marker.
        if claim in annotated and marker not in annotated:
            annotated = annotated.replace(claim, marker, 1)

    footer = f"\n\n[VERIFICATION] {len(conflicts)} conflicts found; review before posting"
    return annotated + footer
