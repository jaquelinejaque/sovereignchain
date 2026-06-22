# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Proactive runner — one full ingest → analyse → notify cycle.

Intended invocation:
  * cron: every N minutes / hours
  * Cloud Run: scheduled task hitting an HTTP endpoint that calls run_once()
  * CLI: ``python -m quorum.proactive.runner --once``

This file does NOT execute any action in the owner's name. The runner
ends at the email digest. The owner clicks approve/edit/reject which
flows into a separate executor (TODO, post-MVP).

Architecture:
  ingest_all() → list[Signal] (parallel across sources)
   → dedupe via store.insert_signal (returns False on duplicate)
   → analyse_batch(fresh_signals) → BatchVerdict
   → for each recommendation: optional draft_action() via scorer
   → render digest (verdict + drafts) → send email
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Sequence

from quorum.proactive import analyst, enricher, ingest, notifier, scorer, store
from quorum.proactive.signal import Signal
from quorum.providers.base import Provider

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config (env-driven for now; YAML loader later)                              #
# --------------------------------------------------------------------------- #


def _list_env(name: str, default: list[str] | None = None) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default or []
    return [x.strip() for x in raw.split(",") if x.strip()]


DEFAULT_RSS_FEEDS = [
    "https://hnrss.org/newest?points=50",
    "https://www.theregister.com/headlines.atom",
    "https://feeds.feedburner.com/TechCrunch/artificial-intelligence",
]

DEFAULT_SITES = [
    "https://quorum-ai.dev",
    "https://api.quorum-ai.dev/health",
    "https://keratintreatment.co.uk",
]

DEFAULT_TWITTER_QUERIES = [
    '"multi-LLM consensus" -is:retweet lang:en',
    '"EU AI Act" "compliance" -is:retweet lang:en',
    '"AI audit" startup -is:retweet lang:en',
]


def _load_providers() -> list[Provider]:
    """Build the provider pool from whatever API keys are present."""
    providers: list[Provider] = []
    # Lazy imports so missing optional deps don't crash the runner.
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from quorum.providers.anthropic import AnthropicProvider
            providers.append(AnthropicProvider(model="claude-haiku-4-5-20251001"))
        except Exception as e:  # noqa: BLE001
            logger.warning("provider.anthropic.skip err=%s", e)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from quorum.providers.openai import OpenAIProvider
            providers.append(OpenAIProvider(model="gpt-5-mini"))
        except Exception as e:  # noqa: BLE001
            logger.warning("provider.openai.skip err=%s", e)
    if os.getenv("GOOGLE_AI_STUDIO_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            from quorum.providers.gemini import GeminiProvider
            providers.append(GeminiProvider())
        except Exception as e:  # noqa: BLE001
            logger.warning("provider.gemini.skip err=%s", e)
    if os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY"):
        try:
            from quorum.providers.qwen import qwen_max
            providers.append(qwen_max())
        except Exception as e:  # noqa: BLE001
            logger.warning("provider.qwen.skip err=%s", e)
    if os.getenv("XAI_API_KEY"):
        try:
            from quorum.providers.grok import grok_4
            providers.append(grok_4())
        except Exception as e:  # noqa: BLE001
            logger.warning("provider.grok.skip err=%s", e)
    if os.getenv("MISTRAL_API_KEY"):
        try:
            from quorum.providers.mistral import MistralProvider
            providers.append(MistralProvider())
        except Exception as e:  # noqa: BLE001
            logger.warning("provider.mistral.skip err=%s", e)
    return providers


# --------------------------------------------------------------------------- #
# Ingest                                                                       #
# --------------------------------------------------------------------------- #


async def ingest_all(
    *,
    rss_feeds: Sequence[str] | None = None,
    sites: Sequence[str] | None = None,
    twitter_queries: Sequence[str] | None = None,
) -> list[Signal]:
    """Fan out across all configured sources in parallel. Resilient."""
    rss = rss_feeds or _list_env("PROACTIVE_RSS_FEEDS") or DEFAULT_RSS_FEEDS
    sites_ = sites or _list_env("PROACTIVE_SITES") or DEFAULT_SITES
    twq = twitter_queries or _list_env("PROACTIVE_TWITTER_QUERIES") or DEFAULT_TWITTER_QUERIES

    tasks = []
    if rss:
        tasks.append(("rss", ingest.ingest_rss(rss)))
    if sites_:
        tasks.append(("sites", ingest.ingest_sites(sites_)))
    if twq and os.getenv("TWITTER_BEARER_TOKEN"):
        tasks.append(("twitter", ingest.ingest_twitter_search(twq)))

    if not tasks:
        return []

    results = await asyncio.gather(*[t for _, t in tasks], return_exceptions=True)
    all_signals: list[Signal] = []
    for (name, _), res in zip(tasks, results):
        if isinstance(res, Exception):
            logger.warning("ingest.%s.failed err=%s", name, res)
            continue
        logger.info("ingest.%s got=%d", name, len(res))
        all_signals.extend(res)
    return all_signals


async def persist_new(signals: Sequence[Signal]) -> list[Signal]:
    """Insert each signal; return only those that were NEW (not deduped)."""
    fresh: list[Signal] = []
    for s in signals:
        inserted = await store.insert_signal(s)
        if inserted:
            fresh.append(s)
    return fresh


# --------------------------------------------------------------------------- #
# Cycle                                                                        #
# --------------------------------------------------------------------------- #


async def run_once(*, send_email: bool = True) -> dict:
    """One full cycle: ingest → persist new → analyse → notify.

    Returns a summary dict the cron wrapper can log.
    """
    providers = _load_providers()
    if not providers:
        logger.error("runner.no_providers — set at least one *_API_KEY env var")
        return {"error": "no_providers"}
    logger.info("runner.start providers=%s", [p.name for p in providers])

    raw = await ingest_all()
    logger.info("runner.ingested_raw=%d", len(raw))

    fresh = await persist_new(raw)
    logger.info("runner.fresh_after_dedupe=%d", len(fresh))

    # Cross-internet enrichment on the top-K most promising signals so
    # the analyst can reason against broader web context, not just the
    # raw signal body. Soft-fails per signal — never blocks the cycle.
    enrich_k = int(os.getenv("PROACTIVE_ENRICH_TOP_K", "10"))
    if enrich_k > 0 and fresh:
        fresh = await enricher.enrich_top_k(fresh, k=enrich_k)
        logger.info("runner.enriched top_k=%d", min(enrich_k, len(fresh)))

    verdict = await analyst.analyse_batch(fresh, providers=providers)
    logger.info("runner.verdict recs=%d worth=%d conf=%.2f",
                len(verdict.recommendations),
                verdict.n_signals_worth_acting,
                verdict.overall_confidence)

    # For each recommendation, generate a concrete draft + persist it.
    drafts_payload: list[dict] = []
    for r in verdict.recommendations:
        # find the signal by external_id
        match = next((s for s in fresh if s.external_id == r.signal_external_id),
                     None)
        if match is None:
            continue
        kind = r.suggested_kind if r.suggested_kind in ("dm", "email_reply", "post") \
               else "note"
        d = await scorer.draft_action(
            match, suggested_kind=kind,
            target=r.suggested_target or match.author,
            providers=providers,
        )
        if d is None:
            continue
        d.intent_score = r.confidence
        d.rationale = r.why_now or r.headline
        await store.insert_draft(d)
        drafts_payload.append({
            "id": d.id,
            "kind": d.kind,
            "target": d.target,
            "subject": d.subject,
            "body": d.body,
            "intent_score": d.intent_score,
            "draft_score": d.draft_score,
            "rationale": d.rationale,
            "signal_author": match.author,
            "signal_title": match.title,
            "signal_url": match.url,
            "signal_source": match.source,
        })

    # Compose digest = verdict header + drafts
    verdict_html = analyst.render_verdict_html(verdict)
    drafts_html = notifier.render_digest_html(drafts_payload)
    html = verdict_html + drafts_html

    sent = False
    if send_email:
        sent = notifier.send_digest(
            html,
            subject=f"Quorum Proactive — {len(verdict.recommendations)} actions, "
                    f"{len(fresh)} signals",
        )

    return {
        "raw_signals": len(raw),
        "fresh_signals": len(fresh),
        "recommendations": len(verdict.recommendations),
        "drafts_created": len(drafts_payload),
        "email_sent": sent,
        "verdict_headline": verdict.headline[:200],
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Quorum Proactive runner")
    p.add_argument("--once", action="store_true", help="Run one cycle then exit")
    p.add_argument("--no-email", action="store_true", help="Skip email (dry run)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    summary = asyncio.run(run_once(send_email=not args.no_email))
    import json
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["ingest_all", "persist_new", "run_once"]
