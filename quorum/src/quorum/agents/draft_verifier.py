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

import os
import re
from typing import Any

from quorum.agents.verifier_store import save_draft as _persist_draft

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


def _persistence_enabled() -> bool:
    """Feature flag — set QUORUM_VERIFIER_PERSIST=0 to disable disk writes.

    Default ON. Tests that don't want disk writes opt out explicitly.
    """
    return os.environ.get("QUORUM_VERIFIER_PERSIST", "1") not in {"0", "false", "False"}


def find_conflicts(
    draft_text: str,
    fact_sheet: dict,
    *,
    persist: bool | None = None,
    kind: str = "unknown",
    confidence: float = 0.0,
    parent_draft_id: str | None = None,
) -> list[dict]:
    """Return a list of conflict dicts found in the draft.

    Each conflict dict has the shape:
        {
            "claim_text": str,       # what the draft said
            "expected_value": Any,   # what the fact_sheet says (or None)
            "type": str,             # 'numeric_inflated' | 'name_invented' | 'unverifiable'
        }

    Side-effect (additive, backward-compat)
    ---------------------------------------
    When `persist` is True (default: env-flag controlled), the draft text
    plus the conflict list are written to the SQLite audit store via
    `verifier_store.save_draft`. Disk failures are swallowed — see
    `verifier_store.save_draft` for the rationale.

    Callers that need the draft_id back should use `verify_and_persist()`
    instead — `find_conflicts` keeps its original return signature for
    backward compatibility.
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

    # ---- 3. Side-effect: audit persistence -------------------------------
    # Additive, backward-compat: only fires when the flag is on AND the
    # caller did not explicitly opt out via persist=False. Failure is
    # swallowed inside save_draft itself.
    should_persist = persist if persist is not None else _persistence_enabled()
    if should_persist:
        _persist_draft(
            draft_text,
            conflicts=conflicts,
            confidence=confidence,
            parent_draft_id=parent_draft_id,
            kind=kind,
            fact_sheet=fact_sheet,
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


# ---------------------------------------------------------------------------
# Persistence-aware helpers (additive — public surface for callers that want
# the draft_id back, or that want to drive a verify→rewrite loop)
# ---------------------------------------------------------------------------


def verify_and_persist(
    draft_text: str,
    fact_sheet: dict,
    *,
    kind: str = "unknown",
    confidence: float = 0.0,
    parent_draft_id: str | None = None,
    annotate: bool = False,
) -> tuple[list[dict], str, str]:
    """Run verification, persist, and return (conflicts, draft_id, content).

    Unlike `find_conflicts`, this returns the persisted `draft_id` so
    callers can chain attempts via `parent_draft_id`. If `annotate=True`,
    the persisted content is the annotated form and verdict is
    'annotated'; otherwise the raw draft is stored with verdict 'clean'
    or 'conflicts' derived from the conflict list.
    """
    # Bypass the side-effect inside find_conflicts — we want to control
    # the persistence call ourselves (so we get the draft_id back and can
    # set verdict='annotated' when applicable).
    conflicts = find_conflicts(draft_text, fact_sheet, persist=False)

    content = annotate_draft(draft_text, conflicts) if (annotate and conflicts) else draft_text
    verdict: str | None = "annotated" if (annotate and conflicts) else None

    draft_id = _persist_draft(
        content,
        verdict=verdict,
        conflicts=conflicts,
        confidence=confidence,
        parent_draft_id=parent_draft_id,
        kind=kind,
        fact_sheet=fact_sheet,
    )
    return conflicts, draft_id, content


def iterate_draft(
    content: str,
    fact_sheet: dict,
    *,
    rewriter,
    max_iter: int = 3,
    kind: str = "unknown",
    initial_confidence: float = 0.0,
) -> dict:
    """Verify → rewrite → re-verify loop with lineage persistence.

    `rewriter` is a sync callable `(prev_content, conflicts) -> (new_content,
    new_confidence)`. The function is provided by the caller (it's where
    the LLM call lives) so this module stays free of LLM dependencies and
    remains trivially unit-testable with a stub rewriter.

    Stops when the verifier returns zero conflicts or `max_iter` is hit.
    Each attempt is persisted with `parent_draft_id` set, so the full
    lineage is queryable via `verifier_store.get_history(draft_id)`.

    Returns a dict with the final content, the final draft_id, the number
    of iterations actually run, and the unresolved conflict count.
    """
    if max_iter < 1:
        max_iter = 1

    current_content = content
    current_confidence = float(initial_confidence)
    parent_id: str | None = None
    iterations = 0
    final_conflicts: list[dict] = []
    final_draft_id = ""

    for iterations in range(1, max_iter + 1):
        conflicts, draft_id, persisted_content = verify_and_persist(
            current_content,
            fact_sheet,
            kind=kind,
            confidence=current_confidence,
            parent_draft_id=parent_id,
            annotate=False,
        )
        final_conflicts = conflicts
        final_draft_id = draft_id or final_draft_id
        parent_id = draft_id or parent_id

        if not conflicts:
            current_content = persisted_content
            break

        if iterations >= max_iter:
            # Annotate + persist final, parent linked to last attempt.
            annotated = annotate_draft(persisted_content, conflicts)
            annotated_id = _persist_draft(
                annotated,
                verdict="annotated",
                conflicts=conflicts,
                confidence=current_confidence,
                parent_draft_id=parent_id,
                kind=kind,
                fact_sheet=fact_sheet,
            )
            final_draft_id = annotated_id or final_draft_id
            current_content = annotated
            break

        new_content, new_confidence = rewriter(persisted_content, conflicts)
        current_content = new_content
        current_confidence = float(new_confidence)

    return {
        "content": current_content,
        "draft_id": final_draft_id,
        "iterations": iterations,
        "unresolved": len(final_conflicts),
    }
