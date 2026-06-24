"""Refusal detection — exclude models that refused to answer from consensus.

WHY this exists
---------------
Different LLMs have different safety/refusal training. A query that one model
refuses ("I can't help with that") may be answered fine by another model with
weaker safety overlay (open-weight Llama, Hermes local). In a consensus engine,
a refusal is **not data** — it's a non-answer that distorts the agreement
score: if 3 of 5 models refuse and 2 answer, the "agreement" is artificial
(the 3 refusals will look identical to each other) and the canonical pick is
biased toward whichever model happens to have weights set higher.

WHAT this does
--------------
Marks a ``ModelResponse`` as a refusal when its text matches one of the known
refusal patterns. The consensus layer can then **exclude refusals from the
agreement scoring** so that the remaining (genuine) answers dominate.

WHAT this is NOT
----------------
- This is NOT a jailbreak — it does not try to make a refusing model respond.
- This is NOT an instruction injection — it does not modify the prompt.
- This only OBSERVES the model's own output and labels it.

The legal posture is: every model retains its own safety floor. Quorum simply
routes around refusers to whichever model already chose to answer of its own
accord. No model is asked to bypass its training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Refusal patterns observed in real LLM outputs (2025-2026). Compiled once.
# Patterns are tested case-insensitive against the first 400 chars only —
# refusals are almost always front-loaded in the response.
_REFUSAL_PATTERNS = [
    # English
    r"\bI (can't|cannot|can not|won't|will not) (help|assist|provide|answer|comply|engage|do that|create|write|generate)",
    r"\bI'm (not able|unable|not comfortable|sorry, but I cannot)",
    r"\bAs an AI(,| language model| assistant)",  # often follows by refusal/disclaimer
    r"\bI (must|have to|need to) (decline|refuse)",
    r"\bI'm (designed|programmed|trained) (not |to refuse|to decline)",
    r"\bSorry, (I|but I) (can't|cannot|won't|will not)",
    r"\bThis (request|query|content) (violates|goes against|is against|cannot be)",
    r"\bI (don't|do not) (have the ability|feel comfortable)",
    # Portuguese
    r"\bNão posso (ajudar|te ajudar|fornecer|responder|criar)",
    r"\bDesculpe, mas (não posso|eu não posso)",
    r"\bComo (uma|um) (IA|assistente|modelo)",
    r"\bMinhas diretrizes (não permitem|me impedem)",
    r"\bSinto muito, (mas não|eu não)",
]
_COMPILED = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS]


@dataclass
class RefusalVerdict:
    is_refusal: bool
    pattern_matched: str | None = None


def looks_like_refusal(text: str | None) -> RefusalVerdict:
    """Return whether the given response text reads as a refusal.

    Short, fast, no I/O. Only looks at the first 400 chars (refusals are
    almost always front-loaded; deeper text may include the actual answer
    even after a hedging preamble — we don't want to mis-flag those).
    """
    if not text:
        return RefusalVerdict(is_refusal=False)
    head = text.strip()[:400]
    if not head:
        return RefusalVerdict(is_refusal=False)
    for compiled in _COMPILED:
        m = compiled.search(head)
        if m:
            return RefusalVerdict(is_refusal=True, pattern_matched=m.group(0))
    return RefusalVerdict(is_refusal=False)


def partition_refusals(responses):
    """Split a list of ModelResponse-like objects into (answered, refused).

    ``responses`` items must have ``.response`` and ``.name`` attributes.
    Items with empty/None response stay in ``answered`` (they're handled
    elsewhere as "empty provider", a different failure mode).
    """
    answered = []
    refused = []
    for r in responses:
        text = getattr(r, "response", None)
        if not text:
            answered.append(r)
            continue
        verdict = looks_like_refusal(text)
        if verdict.is_refusal:
            refused.append((r, verdict))
        else:
            answered.append(r)
    return answered, refused


__all__ = ["RefusalVerdict", "looks_like_refusal", "partition_refusals"]
