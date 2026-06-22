# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Web enrichment for proactive signals.

The analyst sees each Signal's body, but a one-line tweet or RSS title
rarely tells the whole story. Before the analyst recommends an action,
we enrich the top-K candidate signals with cross-internet context:

  * who is the author (recent posts / bio / company)
  * what is the topic (latest news / consensus)
  * is there a competing or duplicate effort already
  * is there a deadline / event tied to this

We use the existing ``web/multi_source.py`` (8 endpoints, no API keys)
so this costs zero $$ beyond LLM calls.

Output: each Signal gets an ``extra["enrichment"]`` dict the analyst
can read in its prompt.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

from quorum.proactive.signal import Signal

logger = logging.getLogger(__name__)


async def _safe_search(query: str, *, per_source: int = 3,
                       timeout_s: float = 12.0) -> list[dict[str, Any]]:
    """Wrap multi_source.search with a hard timeout + swallow errors."""
    try:
        from quorum.web.multi_source import search_multi
    except Exception as e:  # noqa: BLE001
        logger.warning("enricher.import_multi_source_failed err=%s", e)
        return []
    try:
        # search_multi signature varies; try the common forms
        try:
            results = await asyncio.wait_for(
                search_multi(query, per_source=per_source),
                timeout=timeout_s,
            )
        except TypeError:
            results = await asyncio.wait_for(
                search_multi(query), timeout=timeout_s,
            )
    except asyncio.TimeoutError:
        logger.warning("enricher.search_timeout q=%s", query[:80])
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("enricher.search_failed q=%s err=%s", query[:80], e)
        return []
    # Normalise to list of dicts — multi_source returns SearchResult dataclasses
    if not isinstance(results, list):
        return []
    normalised: list[dict[str, Any]] = []
    for r in results[: 8 * per_source]:
        if isinstance(r, dict):
            normalised.append(r)
        elif hasattr(r, "to_dict"):
            normalised.append(r.to_dict())
        else:
            normalised.append({
                "title": getattr(r, "title", ""),
                "url": getattr(r, "url", ""),
                "snippet": getattr(r, "snippet", ""),
                "source": getattr(r, "source", ""),
            })
    return normalised


def _short(s: str, n: int = 200) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


async def enrich_signal(signal: Signal, *, max_queries: int = 3) -> Signal:
    """Add ``signal.extra['enrichment']`` with cross-internet context.

    Mutates the signal in place AND returns it (convenient for gather).
    """
    # Build up to ``max_queries`` targeted searches.
    queries: list[tuple[str, str]] = []  # (label, query)

    # 1) author context (if it looks like a real handle, not a URL)
    author = (signal.author or "").strip()
    if author and not author.startswith("http") and len(author) <= 60:
        author_clean = author.lstrip("@").split("/")[-1]
        queries.append(("author", f'"{author_clean}" recent 2026'))

    # 2) topic / title
    title = (signal.title or "").strip()
    if title:
        queries.append(("topic", title[:120]))

    # 3) body keywords — pick the longest noun-ish phrase as cheap heuristic
    body = (signal.body or "")[:600]
    if body and len(queries) < max_queries:
        # naive: first sentence
        first = body.split(".")[0][:120].strip()
        if first and first.lower() not in title.lower():
            queries.append(("context", first))

    queries = queries[:max_queries]
    if not queries:
        signal.extra["enrichment"] = {"queries": [], "results": {}}
        return signal

    tasks = [_safe_search(q, per_source=2) for _, q in queries]
    results = await asyncio.gather(*tasks)

    enrichment: dict[str, Any] = {
        "queries": [{"label": l, "query": q} for l, q in queries],
        "results": {},
    }
    for (label, q), batch in zip(queries, results):
        enrichment["results"][label] = [
            {
                "title": _short(r.get("title", ""), 160),
                "url": r.get("url", ""),
                "snippet": _short(r.get("snippet", ""), 240),
                "source": r.get("source", ""),
            }
            for r in batch[:5]
        ]
    signal.extra["enrichment"] = enrichment
    return signal


async def enrich_top_k(signals: Sequence[Signal], *, k: int = 10) -> list[Signal]:
    """Enrich the most promising K signals in parallel.

    "Most promising" here = ones with the most body content (proxy for
    real news vs. status update). Sites with status_code != 200 are
    bumped to the front (down/changed = always interesting).
    """
    if not signals:
        return list(signals)

    def _priority(s: Signal) -> int:
        # error sites first, then by body length
        status = s.extra.get("status_code")
        if isinstance(status, int) and status >= 400:
            return -10**9
        return -len(s.body or "")

    ordered = sorted(signals, key=_priority)
    head, tail = ordered[:k], ordered[k:]
    enriched = await asyncio.gather(*(enrich_signal(s) for s in head))
    return list(enriched) + tail


__all__ = ["enrich_signal", "enrich_top_k"]
