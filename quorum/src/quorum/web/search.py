"""Zero-API-key web search via DuckDuckGo HTML."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import List
from urllib.parse import unquote

import httpx


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str

    def to_dict(self):
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_DDG_REDIR = re.compile(r"//duckduckgo\.com/l/\?uddg=([^&]+)")


def _clean(s: str) -> str:
    return html.unescape(_TAG_RE.sub("", s)).strip()


def _resolve(url: str) -> str:
    m = _DDG_REDIR.search(url)
    if m:
        return unquote(m.group(1))
    if url.startswith("//"):
        return "https:" + url
    return url


def web_search(query: str, n: int = 5, timeout: float = 12.0) -> List[WebResult]:
    """Return top-n web results, fanning out across 4 sources to dodge DDG throttle.

    Tries multi_source.search_multi (DDG + Wikipedia + HackerNews + arXiv).
    Falls back to DDG-only on any failure so the call still works in tests
    where multi_source can't be imported.
    """
    import os
    # Allow env-var opt-out for tests/CI
    if os.getenv("QUORUM_SINGLE_SOURCE") == "1":
        return _web_search_ddg_only(query, n, timeout)
    try:
        from quorum.web.multi_source import search_multi_sync
        rows = search_multi_sync(query, per_source=max(2, n // 2))
        if rows:
            return [WebResult(title=r["title"], url=r["url"], snippet=r["snippet"]) for r in rows[:n]]
    except Exception:
        pass  # fall back to DDG-only
    return _web_search_ddg_only(query, n, timeout)


def _web_search_ddg_only(query: str, n: int, timeout: float) -> List[WebResult]:
    """Original DDG-only path, kept as fallback when multi_source unavailable."""
    try:
        resp = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36"
                ),
            },
            timeout=timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception:
        return []

    results: list[WebResult] = []
    for m in _RESULT_RE.finditer(resp.text):
        url = _resolve(m.group(1))
        title = _clean(m.group(2))
        snippet = _clean(m.group(3))
        if not url or not title:
            continue
        results.append(WebResult(title=title, url=url, snippet=snippet))
        if len(results) >= n:
            break
    return results


def search_to_context(query: str, n: int = 5) -> str:
    """Format search results as a short context block for prompts."""
    results = web_search(query, n=n)
    if not results:
        return f"[no web results for: {query}]"
    lines = [f"Web context (DuckDuckGo, top {len(results)} for '{query}'):"]
    for i, r in enumerate(results, 1):
        snippet = (r.snippet or "")[:240]
        lines.append(f"{i}. {r.title}\n   {r.url}\n   {snippet}")
    return "\n".join(lines)
