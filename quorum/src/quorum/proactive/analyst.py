# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Analyst layer — Quorum reads everything ingested and gives a strategic verdict.

This is the missing piece between `scorer.py` (per-signal score) and
`notifier.py` (digest send). The owner does not want a flat list of
"here are 50 signals ranked"; she wants:

  "I analysed all 50 signals + your profile + recent context.
   THE 3 THINGS WORTH YOUR TIME RIGHT NOW are:
     1. <signal X> → do <action Y> within <window Z>, here is the draft
     2. ...
     3. ...
   You can safely IGNORE these 47 because <one-line reason per cluster>.
   Across the batch I noticed <pattern> — that might be the real lead."

That requires a second consensus call where Quorum sees the WHOLE
batch and reasons across it, not just per-item. The output is a
single `BatchVerdict` that the notifier renders at the top of the
digest, above the raw drafts.

Why this matters: owner is exhausted; she does not want to triage
50 items. She wants Quorum to triage and only surface the 1-3 that
move the needle this week.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from quorum.core.consensus import consensus
from quorum.providers.base import Provider
from quorum.proactive.signal import Signal
from quorum.proactive.scorer import (
    DEFAULT_PROFILE, OwnerProfile, _extract_json, _profile_block,
)

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """One specific action Quorum recommends from a batch."""

    signal_external_id: str  # which signal this came from
    rank: int  # 1 = highest priority
    headline: str  # one-line "WHAT to do"
    why_now: str  # why this is the right thing this week
    suggested_kind: str  # dm | email_reply | post | call | research | wait
    suggested_target: str
    deadline_hint: str = ""  # "today", "this week", "before Friday", ""
    risk_note: str = ""  # any caveat (privacy, rep, $$)
    confidence: float = 0.0  # consensus confidence 0..1


@dataclass
class BatchVerdict:
    """Quorum's strategic read on a whole batch of ingested signals."""

    n_signals_seen: int
    n_signals_worth_acting: int
    headline: str  # one-paragraph executive summary
    recommendations: list[Recommendation] = field(default_factory=list)
    ignored_clusters: list[str] = field(default_factory=list)  # 1-line reasons
    cross_batch_pattern: str = ""  # "I noticed N tweets about X this week"
    overall_confidence: float = 0.0
    consensus_models: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Prompt                                                                       #
# --------------------------------------------------------------------------- #


_ANALYST_PROMPT = """You are Quorum acting as the OWNER'S CHIEF OF STAFF.

She is exhausted. She does NOT want a triage list. She wants you to read
EVERYTHING and tell her the 1-3 things that actually matter this week,
why, and what to do — with everything else explained away in one line.

OWNER PROFILE:
{profile_block}

THE BATCH ({n} signals from the last ingest window):
{signals_block}

YOUR JOB — produce a strategic verdict in ONE JSON object only (no markdown):

{{
  "headline": "<one short paragraph: what you saw in this batch and the bottom line>",
  "n_signals_worth_acting": <int>,
  "recommendations": [
    {{
      "signal_external_id": "<external_id from the batch>",
      "rank": <int starting at 1>,
      "headline": "<one line: what to DO>",
      "why_now": "<why this is the move THIS WEEK, not later>",
      "suggested_kind": "dm" | "email_reply" | "post" | "call" | "research" | "wait",
      "suggested_target": "<handle / email / channel>",
      "deadline_hint": "today" | "this week" | "before [date]" | "",
      "risk_note": "<single caveat if any>",
      "confidence": <float 0..1>
    }}
    // ... up to 3 max. quality over volume.
  ],
  "ignored_clusters": [
    "<one-line reason for ignoring a cluster of similar signals, e.g. '14 generic AI news posts — no actionable hook'>"
  ],
  "cross_batch_pattern": "<if you noticed a pattern across multiple signals worth flagging, say so in one sentence. else empty string.>",
  "overall_confidence": <float 0..1 — your honest confidence in the batch read>
}}

RULES:
- Max 3 recommendations. If fewer than 3 signals are worth action, return fewer.
- Recommendations MUST cite the signal's `external_id` exactly.
- If NOTHING is worth acting on, return `recommendations: []` and headline must say so plainly. Honest empty hands beat fake leads.
- Never recommend an action that requires the owner to claim expertise or authority she doesn't have (no fake credentials).
- Never recommend posting publicly in her name about something risky to her brand.
- Prefer recommendations where Quorum can write the draft afterwards (dm, email_reply, post). Only use "research" / "call" / "wait" when no direct action fits.
"""


def _signal_block(signals: Sequence[Signal]) -> str:
    """Render signals compactly so the prompt stays inside context window.

    When a signal carries ``extra['enrichment']`` (added by enricher.py),
    we inline up to 3 web-search snippets per signal so the analyst can
    cross-reference the claim against the broader internet before
    recommending action.
    """
    lines = []
    for i, s in enumerate(signals, 1):
        snippet = (s.body or "")[:300].replace("\n", " ").strip()
        block = (
            f"[{i}] id={s.external_id} src={s.source} by={s.author}\n"
            f"    title: {s.title[:160]}\n"
            f"    body: {snippet}\n"
            f"    url: {s.url}"
        )
        enrich = s.extra.get("enrichment") if isinstance(s.extra, dict) else None
        if enrich and isinstance(enrich, dict):
            ctx_lines = []
            for label, results in (enrich.get("results") or {}).items():
                for r in (results or [])[:2]:
                    snip = (r.get("snippet") or r.get("title", ""))[:140]
                    if snip:
                        ctx_lines.append(f"      · [{label}/{r.get('source','?')}] {snip}")
            if ctx_lines:
                block += "\n    web_context:\n" + "\n".join(ctx_lines[:5])
        lines.append(block)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public entry                                                                 #
# --------------------------------------------------------------------------- #


async def analyse_batch(
    signals: Sequence[Signal],
    *,
    providers: Sequence[Provider],
    profile: OwnerProfile = DEFAULT_PROFILE,
    budget_usd: float = 0.30,
    max_signals_in_prompt: int = 80,
) -> BatchVerdict:
    """Read the whole batch through 4-LLM consensus → return one verdict.

    If ``signals`` exceeds ``max_signals_in_prompt`` we keep the latest
    that many — the rest are flagged in ``ignored_clusters`` so the
    owner knows there's more to look at if she wants.
    """
    if not signals:
        return BatchVerdict(
            n_signals_seen=0, n_signals_worth_acting=0,
            headline="No signals in this batch. Nothing to report.",
            recommendations=[], ignored_clusters=[], cross_batch_pattern="",
            overall_confidence=1.0, consensus_models=[],
        )

    seen = list(signals)
    used = seen[-max_signals_in_prompt:]
    truncated = len(seen) - len(used)

    prompt = _ANALYST_PROMPT.format(
        profile_block=_profile_block(profile),
        n=len(used),
        signals_block=_signal_block(used),
    )

    try:
        result = await consensus(
            prompt, providers=list(providers),
            budget_usd=budget_usd,
            enable_self_prompt=False, route=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("analyst.consensus_failed err=%s", e)
        return BatchVerdict(
            n_signals_seen=len(seen), n_signals_worth_acting=0,
            headline=f"Consensus engine failed: {e}",
            overall_confidence=0.0,
        )

    # Pick the best parseable JSON from the highest-weighted model
    verdict: BatchVerdict | None = None
    parsed_models: list[str] = []
    for m in sorted(result.models, key=lambda mm: -mm.weight):
        data = _extract_json(m.response)
        if not data:
            continue
        parsed_models.append(m.name)
        if verdict is not None:
            continue
        try:
            recs_raw = data.get("recommendations") or []
            recs: list[Recommendation] = []
            for r in recs_raw[:3]:
                if not isinstance(r, dict):
                    continue
                try:
                    recs.append(Recommendation(
                        signal_external_id=str(r.get("signal_external_id", "")),
                        rank=int(r.get("rank", len(recs) + 1)),
                        headline=str(r.get("headline", ""))[:300],
                        why_now=str(r.get("why_now", ""))[:500],
                        suggested_kind=str(r.get("suggested_kind", "note")),
                        suggested_target=str(r.get("suggested_target", "")),
                        deadline_hint=str(r.get("deadline_hint", "")),
                        risk_note=str(r.get("risk_note", "")),
                        confidence=max(0.0, min(1.0, float(r.get("confidence", 0.5)))),
                    ))
                except (TypeError, ValueError):
                    continue
            ignored = data.get("ignored_clusters") or []
            verdict = BatchVerdict(
                n_signals_seen=len(seen),
                n_signals_worth_acting=int(data.get("n_signals_worth_acting", len(recs))),
                headline=str(data.get("headline", ""))[:1000],
                recommendations=recs,
                ignored_clusters=[str(x)[:300] for x in ignored if x][:10],
                cross_batch_pattern=str(data.get("cross_batch_pattern", ""))[:500],
                overall_confidence=max(0.0, min(1.0,
                    float(data.get("overall_confidence", 0.5)))),
                consensus_models=[],  # filled below across all parseable models
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("analyst.parse_failed model=%s err=%s", m.name, e)
            continue

    if verdict is None:
        return BatchVerdict(
            n_signals_seen=len(seen), n_signals_worth_acting=0,
            headline="Consensus produced no parseable verdict.",
            overall_confidence=0.0, consensus_models=[m.name for m in result.models],
        )

    verdict.consensus_models = parsed_models
    if truncated > 0:
        verdict.ignored_clusters.insert(
            0, f"{truncated} older signals not shown to keep prompt size bounded"
        )
    return verdict


# --------------------------------------------------------------------------- #
# Render verdict as HTML block (for the email digest)                          #
# --------------------------------------------------------------------------- #


def render_verdict_html(v: BatchVerdict) -> str:
    """Render the BatchVerdict as the executive-summary section of the digest."""
    if v.n_signals_seen == 0:
        return '<div class="verdict empty"><h2>Quorum analysed nothing this round.</h2></div>'

    recs_html = ""
    if v.recommendations:
        recs_rows = []
        for r in sorted(v.recommendations, key=lambda x: x.rank):
            deadline = (f' · <span class="deadline">{r.deadline_hint}</span>'
                        if r.deadline_hint else "")
            risk = (f'<div class="risk">⚠ {r.risk_note}</div>'
                    if r.risk_note else "")
            recs_rows.append(f"""<div class="rec">
  <div class="rank">#{r.rank}</div>
  <div class="rec-body">
    <div class="rec-head">{r.headline}</div>
    <div class="rec-meta">→ {r.suggested_kind} · {r.suggested_target}{deadline}
        · confidence {r.confidence:.0%}</div>
    <div class="rec-why"><b>Why now:</b> {r.why_now}</div>
    {risk}
  </div>
</div>""")
        recs_html = '<h3>Top recommendations</h3>' + "".join(recs_rows)
    else:
        recs_html = '<p><b>Nothing worth action this round.</b> Honest empty hands.</p>'

    ignored = ""
    if v.ignored_clusters:
        items = "".join(f'<li>{c}</li>' for c in v.ignored_clusters)
        ignored = f'<details><summary>Ignored ({len(v.ignored_clusters)})</summary><ul>{items}</ul></details>'

    pattern = ""
    if v.cross_batch_pattern:
        pattern = f'<div class="pattern"><b>Pattern across batch:</b> {v.cross_batch_pattern}</div>'

    models = ", ".join(v.consensus_models) if v.consensus_models else "—"

    return f"""<div class="verdict">
<style>
.verdict {{ background: #f0f6ff; border-left: 4px solid #2962ff; padding: 16px 18px;
            border-radius: 4px; margin: 0 0 24px 0; }}
.verdict h2 {{ margin: 0 0 8px; font-size: 18px; }}
.verdict h3 {{ margin: 16px 0 6px; font-size: 14px; color: #555; }}
.headline {{ font-size: 15px; line-height: 1.5; margin: 0 0 12px; }}
.rec {{ display: flex; gap: 12px; background: white; padding: 12px;
        border-radius: 4px; margin: 8px 0; }}
.rank {{ font-size: 22px; font-weight: 700; color: #2962ff; min-width: 32px; }}
.rec-head {{ font-weight: 600; font-size: 14px; }}
.rec-meta {{ font-size: 12px; color: #666; margin: 4px 0 8px; }}
.rec-why {{ font-size: 13px; }}
.risk {{ color: #c62828; font-size: 12px; margin-top: 4px; }}
.pattern {{ background: #fff8e1; padding: 8px 12px; border-radius: 4px;
            margin: 12px 0; font-size: 13px; }}
.deadline {{ color: #b06a00; font-weight: 600; }}
details summary {{ cursor: pointer; color: #666; font-size: 13px; }}
.verdict-meta {{ color: #999; font-size: 11px; margin-top: 12px; }}
</style>
<h2>Quorum's read on {v.n_signals_seen} signals</h2>
<p class="headline">{v.headline}</p>
{pattern}
{recs_html}
{ignored}
<div class="verdict-meta">Consensus from: {models} · overall confidence {v.overall_confidence:.0%}</div>
</div>"""


__all__ = [
    "Recommendation", "BatchVerdict",
    "analyse_batch", "render_verdict_html",
]
