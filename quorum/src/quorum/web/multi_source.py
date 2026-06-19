"""Multi-source web search — rotates across 4 endpoints to dodge throttles.

The overnight learner originally hit only DuckDuckGo, which throttles after
~50 sequential queries from the same IP. That capped the knowledge-base
growth at whatever DDG was willing to serve in one session.

This module fans the same query out to:
  * DuckDuckGo Lite (existing scraper — still used opportunistically)
  * Wikipedia REST API (no key, no rate limit in practice)
  * HackerNews Algolia API (no key, no rate limit — strong tech signal)
  * arXiv API (no key, 1 request per 3 seconds courtesy rule)

Each source returns differently-shaped data, but we normalise to:
    [{"title": str, "url": str, "snippet": str, "source": str}, ...]

The overnight loop can then chunk + embed snippets the same way it does
DDG results today.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # "ddg" | "wikipedia" | "hn" | "arxiv"

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "source": self.source}


_USER_AGENT = "QuorumLearner/0.1 (+https://quorum-ai.dev)"


async def _search_ddg(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """DuckDuckGo HTML scrape — same logic as web/search.py but tolerant."""
    try:
        r = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=15.0,
            follow_redirects=True,
        )
        if r.status_code != 200:
            return []
        html = r.text
        # Quick-and-dirty: pull <a class="result__a" href="..."> + snippets
        results: list[SearchResult] = []
        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
            r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        ):
            url, title, snippet = m.group(1), m.group(2), m.group(3)
            # strip tags
            title = re.sub(r"<[^>]+>", "", title).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet).strip()
            # decode DDG redirect
            if "/l/?uddg=" in url:
                ud = re.search(r"uddg=([^&]+)", url)
                if ud:
                    url = urllib.parse.unquote(ud.group(1))
            results.append(SearchResult(title, url, snippet, "ddg"))
            if len(results) >= n:
                break
        return results
    except Exception:  # noqa: BLE001
        return []


async def _search_wikipedia(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """Wikipedia REST search — returns title + extract."""
    try:
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "list": "search",
                "srsearch": query, "srlimit": str(n),
            },
            headers={"User-Agent": _USER_AGENT}, timeout=15.0,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out: list[SearchResult] = []
        for item in data.get("query", {}).get("search", [])[:n]:
            title = item.get("title", "")
            snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
            url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            out.append(SearchResult(title, url, snippet, "wikipedia"))
        return out
    except Exception:  # noqa: BLE001
        return []


async def _search_hackernews(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """HackerNews via Algolia — strong tech-news signal."""
    try:
        r = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": query, "hitsPerPage": str(n), "tags": "story"},
            headers={"User-Agent": _USER_AGENT}, timeout=15.0,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out: list[SearchResult] = []
        for hit in data.get("hits", [])[:n]:
            title = hit.get("title") or hit.get("story_title") or ""
            url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID','')}"
            snippet = (hit.get("story_text") or hit.get("comment_text") or "")
            snippet = re.sub(r"<[^>]+>", "", snippet)[:280]
            if title:
                out.append(SearchResult(title, url, snippet or title, "hn"))
        return out
    except Exception:  # noqa: BLE001
        return []


async def _search_arxiv(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """arXiv — scientific papers. Courtesy rate is 1 req per 3s; one call here is fine."""
    try:
        r = await client.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": str(n)},
            headers={"User-Agent": _USER_AGENT}, timeout=15.0,
        )
        if r.status_code != 200:
            return []
        text = r.text
        out: list[SearchResult] = []
        # Parse Atom feed — minimal, no xml lib
        for entry in re.finditer(r"<entry>(.*?)</entry>", text, re.DOTALL):
            body = entry.group(1)
            t = re.search(r"<title>(.*?)</title>", body, re.DOTALL)
            u = re.search(r"<id>(.*?)</id>", body, re.DOTALL)
            s = re.search(r"<summary>(.*?)</summary>", body, re.DOTALL)
            if t and u:
                title = re.sub(r"\s+", " ", t.group(1)).strip()
                url = u.group(1).strip()
                snippet = re.sub(r"\s+", " ", (s.group(1) if s else "")).strip()[:280]
                out.append(SearchResult(title, url, snippet, "arxiv"))
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


async def search_multi(query: str, per_source: int = 4) -> list[SearchResult]:
    """Fan-out search across all 4 sources in parallel. Returns interleaved
    results so downstream chunking gets diversity even if it cuts the list short.
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _search_ddg(client, query, per_source),
            _search_wikipedia(client, query, per_source),
            _search_hackernews(client, query, per_source),
            _search_arxiv(client, query, per_source),
            return_exceptions=False,
        )
    # Interleave: take 1st from each source, then 2nd from each, etc.
    interleaved: list[SearchResult] = []
    max_len = max(len(r) for r in results) if results else 0
    for i in range(max_len):
        for src_results in results:
            if i < len(src_results):
                interleaved.append(src_results[i])
    return interleaved


def search_multi_sync(query: str, per_source: int = 4) -> list[dict[str, str]]:
    """Sync wrapper. Returns dicts so it works from non-async callers.

    Safe to call from inside an already-running event loop: detects that
    case and runs the coroutine in a worker thread. Without this, calling
    asyncio.run() from inside an active loop raises
    'asyncio.run() cannot be called from a running event loop' and leaves
    the coroutine in the 'never awaited' state — exactly the bug that
    caused draft show-hn to produce empty output in autopilot.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop active in this thread — safe to use asyncio.run directly.
        results = asyncio.run(search_multi(query, per_source))
        return [r.to_dict() for r in results]

    # An event loop is already running here — run the coroutine in a
    # detached thread that owns its own loop, then block on the result.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            lambda: asyncio.run(search_multi(query, per_source))
        )
        results = future.result()
    return [r.to_dict() for r in results]
