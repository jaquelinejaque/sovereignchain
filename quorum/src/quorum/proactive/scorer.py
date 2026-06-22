# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""4-LLM consensus scorer for signals + draft generator.

Two responsibilities:

1. ``score_signal(signal, profile)`` → returns ``(intent_score 0..1,
   rationale)``. Runs Quorum's 4-LLM consensus on a tight scoring
   prompt that asks each model "is this signal worth a draft? rate
   0..1 and explain in one sentence."

2. ``draft_action(signal, profile, action_kind)`` → returns ``Draft``.
   Runs consensus on a generation prompt that asks each model to
   write the proposed action (DM, email reply, post). The synthesis
   step picks the best draft + reports a ``draft_score``.

The owner's profile is the single most important context: who she
serves, what she's selling, her tone, what she absolutely won't say.
Stored in ``~/.quorum/proactive_profile.yaml`` (a plain dict here for
the MVP; YAML loader is one line later).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from quorum.core.consensus import consensus
from quorum.providers.base import Provider
from quorum.proactive.signal import Draft, Signal

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Owner profile                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class OwnerProfile:
    """What Quorum needs to know to score signals + write drafts in voice.

    Default profile is intentionally minimal — operator fills it once.
    """

    name: str = "Jaqueline Martins"
    role: str = "Founder, Sovereign Chain Ltd UK (Quorum + Keratin)"
    bio: str = ""
    interests: list[str] = field(default_factory=list)
    selling: list[str] = field(default_factory=list)
    tone_examples: list[str] = field(default_factory=list)  # 1-2 sentences in her voice
    never_say: list[str] = field(default_factory=list)  # forbidden topics/phrases
    accept_signal_types: list[str] = field(default_factory=lambda: [
        "potential customer with explicit pain",
        "collaboration opportunity (paid or skill-trade)",
        "media/journalist asking about AI / consensus / Keratin",
        "academic / research collaboration",
        "regulatory or industry-body open call",
    ])
    reject_signal_types: list[str] = field(default_factory=lambda: [
        "spam / mass cold pitch from vendor",
        "low-effort newsletter / promo",
        "internal team chatter unrelated to her work",
        "anything that requires medical / legal advice",
    ])


DEFAULT_PROFILE = OwnerProfile(
    bio="Solo UK founder. Built Quorum (multi-LLM consensus + EU AI Act audit). "
        "Also runs Keratin Pro Mastery (V6.1 Diamond Duo). Patent pending PCT/US26/11908.",
    interests=["multi-LLM consensus", "AI compliance", "bug bounty",
               "keratin chemistry", "academic AI safety"],
    selling=["Quorum Pro tier", "Quorum-as-a-Service audits",
             "Keratin V6.1 retail + B2B salon supply"],
)


# --------------------------------------------------------------------------- #
# Score signal                                                                 #
# --------------------------------------------------------------------------- #


_SCORE_PROMPT = """You are scoring an inbound signal for the owner of Quorum.

OWNER PROFILE:
{profile_block}

SIGNAL:
- Source: {source}
- Author: {author}
- Title: {title}
- URL: {url}
- Body (truncated to 1500 chars): {body}

Score this signal 0.0–1.0 on "is this worth the owner's time to respond / act on?"

Rate based on:
- Does this match an ACCEPT_SIGNAL_TYPE in the profile? (+)
- Does this match a REJECT_SIGNAL_TYPE? (–)
- Is the author specific and contactable (not anonymous spam)? (+)
- Is there an explicit ask, pain, or opportunity? (+)
- Is it timely (this week vs. months old)? (+)

Reply with ONE JSON object, no markdown:
{{
  "intent_score": <float 0.0–1.0>,
  "rationale": "<one sentence, max 200 chars>",
  "suggested_action": "dm" | "email_reply" | "post" | "note" | "ignore"
}}
"""


def _profile_block(p: OwnerProfile) -> str:
    return (
        f"- Name: {p.name}\n"
        f"- Role: {p.role}\n"
        f"- Bio: {p.bio}\n"
        f"- Interests: {', '.join(p.interests)}\n"
        f"- Selling: {', '.join(p.selling)}\n"
        f"- ACCEPT signal types: {'; '.join(p.accept_signal_types)}\n"
        f"- REJECT signal types: {'; '.join(p.reject_signal_types)}\n"
        f"- Never say: {'; '.join(p.never_say) if p.never_say else '(no constraints set)'}\n"
    )


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of free-form LLM output."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.rsplit("```", 1)[0]
    m = _JSON_BLOCK.search(s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def score_signal(
    signal: Signal,
    *,
    providers: Sequence[Provider],
    profile: OwnerProfile = DEFAULT_PROFILE,
    budget_usd: float = 0.10,
) -> tuple[float, str, str]:
    """Return ``(intent_score, rationale, suggested_action)``.

    Falls back to ``(0.0, "no_consensus", "ignore")`` if every model
    fails or returns unparseable output — never raises.
    """
    body = (signal.body or "")[:1500]
    prompt = _SCORE_PROMPT.format(
        profile_block=_profile_block(profile),
        source=signal.source,
        author=signal.author,
        title=signal.title,
        url=signal.url,
        body=body,
    )

    try:
        result = await consensus(
            prompt, providers=list(providers),
            budget_usd=budget_usd,
            enable_self_prompt=False, route=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("scorer.consensus_failed err=%s", e)
        return 0.0, f"consensus_error: {e}", "ignore"

    # Take the highest-weighted model that returned parseable JSON.
    scores: list[float] = []
    rationales: list[str] = []
    actions: list[str] = []
    for m in sorted(result.models, key=lambda mm: -mm.weight):
        data = _extract_json(m.response)
        if not data:
            continue
        try:
            scores.append(max(0.0, min(1.0, float(data.get("intent_score", 0.0)))))
            rationales.append(str(data.get("rationale", ""))[:300])
            actions.append(str(data.get("suggested_action", "ignore")))
        except (TypeError, ValueError):
            continue

    if not scores:
        return 0.0, "no_parseable_score", "ignore"

    avg = sum(scores) / len(scores)
    # Majority action; tie → first
    action = max(set(actions), key=actions.count) if actions else "ignore"
    rationale = rationales[0]  # top-weighted model's wording
    return avg, rationale, action


# --------------------------------------------------------------------------- #
# Draft action                                                                 #
# --------------------------------------------------------------------------- #


_DRAFT_PROMPT = """You are writing a {kind} draft on behalf of the owner of Quorum.

OWNER PROFILE:
{profile_block}

ORIGINAL SIGNAL:
- Source: {source}
- Author: {author}
- Title: {title}
- URL: {url}
- Body: {body}

Write a draft {kind} that:
- Sounds like the owner's voice (look at tone_examples in profile)
- Is direct, no fluff, no marketing speak
- Has a clear single ask or value offer
- Is appropriate length for the medium ({kind})
- NEVER mentions topics in `never_say`

Reply with ONE JSON object only:
{{
  "subject": "<short subject if email/post, else empty string>",
  "body": "<the actual draft text>",
  "draft_score": <float 0..1 — your honest confidence this draft is good>,
  "rationale": "<one sentence: why this draft, what it asks for>"
}}
"""


async def draft_action(
    signal: Signal,
    suggested_kind: str,
    target: str,
    *,
    providers: Sequence[Provider],
    profile: OwnerProfile = DEFAULT_PROFILE,
    budget_usd: float = 0.15,
) -> Draft | None:
    """Generate a Draft via 4-LLM consensus. Returns None on total failure."""
    prompt = _DRAFT_PROMPT.format(
        kind=suggested_kind,
        profile_block=_profile_block(profile),
        source=signal.source,
        author=signal.author,
        title=signal.title,
        url=signal.url,
        body=(signal.body or "")[:1500],
    )
    try:
        result = await consensus(
            prompt, providers=list(providers),
            budget_usd=budget_usd,
            enable_self_prompt=False, route=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("scorer.draft_consensus_failed err=%s", e)
        return None

    candidates: list[tuple[float, str, str, str, str]] = []  # (score, subject, body, rationale, model_name)
    for m in result.models:
        data = _extract_json(m.response)
        if not data or not data.get("body"):
            continue
        try:
            score = max(0.0, min(1.0, float(data.get("draft_score", 0.5))))
        except (TypeError, ValueError):
            score = 0.5
        candidates.append((
            score,
            str(data.get("subject", "")),
            str(data["body"]),
            str(data.get("rationale", "")),
            m.name,
        ))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    best = candidates[0]

    return Draft(
        signal_id=signal.id,
        kind=suggested_kind,
        target=target,
        subject=best[1],
        body=best[2],
        intent_score=0.0,  # filled by caller from score_signal result
        draft_score=best[0],
        rationale=best[3],
        consensus_models=[c[4] for c in candidates],
    )


__all__ = [
    "OwnerProfile", "DEFAULT_PROFILE",
    "score_signal", "draft_action",
]
