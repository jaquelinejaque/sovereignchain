# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Ingest workers for proactive monitoring.

MVP supports:
  * RSS / Atom feeds (no API key)
  * Owner sites (uptime + content delta detection)
  * Twitter/X search (uses tools/twitter.py + bearer token)
  * Gmail (uses google-api-python-client, opt-in, label-filtered)

Each ingest function returns a list of `Signal` objects. The
orchestrator (`runner.py`) handles dedupe + persistence.

Design: ingest is pure I/O — fetch, parse, normalise. NO scoring,
NO drafting. That keeps each worker simple + independently testable.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any, Sequence
from xml.etree import ElementTree as ET

import httpx

from quorum.proactive.signal import Signal

logger = logging.getLogger(__name__)


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
}


# --------------------------------------------------------------------------- #
# RSS / Atom ingest                                                            #
# --------------------------------------------------------------------------- #


async def ingest_rss(
    feed_urls: Sequence[str],
    *,
    timeout_s: float = 20.0,
    max_per_feed: int = 30,
) -> list[Signal]:
    """Fetch each feed in parallel, parse, return Signals.

    Resilient: failure on one feed never blocks the others.
    """
    async def _one(url: str) -> list[Signal]:
        try:
            async with httpx.AsyncClient(timeout=timeout_s,
                                          headers=_BROWSER_HEADERS) as cli:
                r = await cli.get(url, follow_redirects=True)
        except httpx.HTTPError as e:
            logger.warning("ingest_rss.fetch_failed url=%s err=%s", url, e)
            return []
        if r.status_code != 200:
            logger.warning("ingest_rss.http_%d url=%s", r.status_code, url)
            return []
        try:
            return _parse_feed(r.text, url)[:max_per_feed]
        except Exception as e:  # noqa: BLE001
            logger.warning("ingest_rss.parse_failed url=%s err=%s", url, e)
            return []

    results = await asyncio.gather(*[_one(u) for u in feed_urls])
    return [s for batch in results for s in batch]


def _parse_feed(xml: str, feed_url: str) -> list[Signal]:
    """Parse RSS 2.0 or Atom 1.0 → list[Signal]. Best-effort."""
    out: list[Signal] = []
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return out

    # Strip namespaces for predictable lookups
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    # RSS 2.0: channel/item
    items = root.findall(".//item")
    if items:
        for it in items:
            title = (it.findtext("title") or "").strip()
            link = (it.findtext("link") or "").strip()
            desc = (it.findtext("description") or "").strip()
            author = (it.findtext("author")
                      or it.findtext("creator") or "").strip() or feed_url
            guid = (it.findtext("guid") or link or title).strip()
            if not (title or desc):
                continue
            out.append(Signal(
                source="rss",
                external_id=guid,
                author=author,
                title=title[:300],
                body=_strip_html(desc)[:4000],
                url=link,
                extra={"feed_url": feed_url},
            ))
        return out

    # Atom 1.0: feed/entry
    entries = root.findall(".//entry")
    for e in entries:
        title = (e.findtext("title") or "").strip()
        link_el = e.find("link")
        link = link_el.get("href", "") if link_el is not None else ""
        summary = (e.findtext("summary") or e.findtext("content") or "").strip()
        author = (e.findtext("author/name") or "").strip() or feed_url
        ext_id = (e.findtext("id") or link or title).strip()
        if not (title or summary):
            continue
        out.append(Signal(
            source="rss",
            external_id=ext_id,
            author=author,
            title=title[:300],
            body=_strip_html(summary)[:4000],
            url=link,
            extra={"feed_url": feed_url},
        ))
    return out


_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _strip_html(s: str) -> str:
    return _WS.sub(" ", _HTML_TAG.sub(" ", s or "")).strip()


# --------------------------------------------------------------------------- #
# Owner-sites ingest (uptime + content-delta)                                  #
# --------------------------------------------------------------------------- #


async def ingest_sites(
    urls: Sequence[str],
    *,
    timeout_s: float = 15.0,
) -> list[Signal]:
    """Probe each URL. Emit a Signal when status is non-2xx OR content
    hash changes since last fetch (delta state lives in extra; the
    orchestrator can compare against the previous signal of same
    external_id).
    """
    async def _one(url: str) -> Signal | None:
        try:
            async with httpx.AsyncClient(timeout=timeout_s,
                                          headers=_BROWSER_HEADERS) as cli:
                r = await cli.get(url, follow_redirects=True)
        except httpx.HTTPError as e:
            return Signal(
                source="site", external_id=f"down::{url}",
                author=url, title=f"DOWN: {url}",
                body=f"HTTP error: {e}", url=url,
                extra={"status": "down", "err": str(e)},
            )
        body_snippet = _strip_html(r.text)[:2000]
        content_hash = hashlib.sha256(r.text.encode("utf-8")).hexdigest()
        # external_id pins to the hourly bucket so we re-emit at most once
        # per hour per (url, content_hash) — orchestrator dedupes.
        hour = datetime.now(timezone.utc).strftime("%Y%m%d-%H")
        return Signal(
            source="site", external_id=f"{url}::{hour}::{content_hash[:12]}",
            author=url,
            title=f"{r.status_code} {url}",
            body=body_snippet,
            url=url,
            extra={
                "status_code": r.status_code,
                "content_hash": content_hash,
                "headers": dict(r.headers.items())
                if hasattr(r.headers, "items") else {},
            },
        )

    results = await asyncio.gather(*[_one(u) for u in urls],
                                    return_exceptions=True)
    return [r for r in results if isinstance(r, Signal)]


# --------------------------------------------------------------------------- #
# Twitter ingest                                                               #
# --------------------------------------------------------------------------- #


async def ingest_twitter_search(
    queries: Sequence[str],
    *,
    max_per_query: int = 20,
    bearer_token: str | None = None,
) -> list[Signal]:
    """Run Twitter v2 recent-search per query. Requires
    ``TWITTER_BEARER_TOKEN`` (Basic tier minimum).

    Tweepy is lazy-imported so the module loads even if tweepy is absent.
    """
    import os
    token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")
    if not token:
        logger.warning("ingest_twitter.skipped reason=no_bearer_token")
        return []
    try:
        import tweepy  # noqa: F401
    except ImportError:
        logger.warning("ingest_twitter.skipped reason=tweepy_not_installed")
        return []

    import tweepy as tw
    client = tw.Client(bearer_token=token, wait_on_rate_limit=False)

    out: list[Signal] = []
    for q in queries:
        try:
            resp = await asyncio.to_thread(
                client.search_recent_tweets,
                query=q, max_results=min(max_per_query, 100),
                tweet_fields=["author_id", "created_at", "lang"],
                expansions=["author_id"],
                user_fields=["username", "name"],
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("ingest_twitter.query_failed q=%s err=%s", q, e)
            continue
        if not resp or not getattr(resp, "data", None):
            continue
        users = {u.id: u for u in (resp.includes.get("users", []) if resp.includes else [])}
        for t in resp.data:
            user = users.get(t.author_id)
            handle = f"@{user.username}" if user else "@unknown"
            out.append(Signal(
                source="twitter",
                external_id=str(t.id),
                author=handle,
                title=(t.text or "")[:200].replace("\n", " "),
                body=t.text or "",
                url=f"https://x.com/{user.username if user else 'i'}/status/{t.id}",
                extra={"query": q, "lang": getattr(t, "lang", "")},
            ))
    return out


__all__ = [
    "ingest_rss",
    "ingest_sites",
    "ingest_twitter_search",
]
