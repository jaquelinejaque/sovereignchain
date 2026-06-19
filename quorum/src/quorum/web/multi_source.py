"""Multi-source web search — fans out across 8 endpoints for breadth + redundancy.

The overnight learner originally hit only DuckDuckGo, which APPEARED to throttle
after ~50 sequential queries — actually it was returning empty pages because the
``QuorumLearner/0.1`` User-Agent was an instant anti-bot flag. With a real
Mozilla Chrome UA + standard browser headers, DDG behaves normally.

This module fans the same query out to:
  * DuckDuckGo Lite — full browser headers (Mozilla UA + Accept-* + DNT)
  * Wikipedia REST API (no key, no rate limit in practice)
  * HackerNews Algolia API (no key, no rate limit — strong tech signal)
  * arXiv API (no key, 1 request per 3 seconds courtesy rule)
  * Bing News RSS (no key) — covers business/marketing/trend topics that the
    other four miss (e.g. "hair salon e-commerce conversion" returned 0 from
    the original four but multiple fresh hits from Bing News).
  * Google News RSS (no key) — HIGHEST-yield single source: ~100 items per
    query across 50,000+ publishers. Should typically saturate per_source.
  * GitHub Search API (no auth, 10 req/min) — strong tech repo signal.
  * Stack Exchange API (no key, 300/day) — Q&A on technical topics.

Future addition (requires user signup): eBay Browse API (5000/day free) for
product/commerce topics — gated on an App ID the user creates manually.

Each source returns differently-shaped data, but we normalise to:
    [{"title": str, "url": str, "snippet": str, "source": str}, ...]

The overnight loop can then chunk + embed snippets the same way it does
DDG results today.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
from dataclasses import dataclass

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str  # "ddg" | "wikipedia" | "hn" | "arxiv" | "bing_news" | "google_news" | "github" | "stackoverflow"

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "source": self.source}


_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)
_BROWSER_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "DNT": "1",
    # NOTE: deliberately omit Accept-Encoding — httpx handles gzip/br transparently
    # and advertising it explicitly trips some anti-bot heuristics.
}

# Wikipedia's User-Agent policy is the OPPOSITE of DDG's — they actively block
# generic browser UAs and require an identifying name + contact URL/email per
# https://meta.wikimedia.org/wiki/User-Agent_policy. A Mozilla UA returns 403
# with "Please respect our robot policy"; a self-identifying UA returns 200.
_WIKI_HEADERS = {
    "User-Agent": "Quorum/0.1 (+https://quorum-ai.dev; ops@quorum-ai.dev) httpx",
    "Accept": "application/json",
}


async def _search_ddg(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """DuckDuckGo HTML scrape — same logic as web/search.py but tolerant.

    Uses full browser-style headers (Mozilla UA + Accept-* + DNT) because the
    bare ``QuorumLearner/0.1`` UA used previously was an instant anti-bot flag —
    DDG was returning empty pages, not actually rate-limiting.
    """
    try:
        r = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=_BROWSER_HEADERS,
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
    """Wikipedia REST search — returns title + extract.

    Wikipedia REQUIRES a self-identifying User-Agent per the Wikimedia bot policy
    (https://meta.wikimedia.org/wiki/User-Agent_policy). Generic browser UAs are
    actively blocked with 403 + "Please respect our robot policy".
    """
    try:
        r = await client.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "format": "json", "list": "search",
                "srsearch": query, "srlimit": str(n),
            },
            headers=_WIKI_HEADERS, timeout=15.0,
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
            headers=_BROWSER_HEADERS, timeout=15.0,
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
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": str(n)},
            headers=_BROWSER_HEADERS, timeout=15.0,
            follow_redirects=True,
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


async def _search_github(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """GitHub repository search — strong signal for any technical topic.

    Unauthenticated cap is 10 requests/minute per IP, which is fine for the
    overnight loop's 1-topic-per-2min pace. README excerpts populate snippets.
    """
    try:
        r = await client.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "per_page": str(n), "sort": "stars"},
            headers={**_BROWSER_HEADERS, "Accept": "application/vnd.github+json"},
            timeout=15.0,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out: list[SearchResult] = []
        for repo in data.get("items", [])[:n]:
            title = repo.get("full_name") or repo.get("name") or ""
            url = repo.get("html_url", "")
            stars = repo.get("stargazers_count", 0)
            desc = (repo.get("description") or "").strip()
            snippet = f"⭐{stars} · {desc}"[:280] if desc else f"⭐{stars}"
            if title and url:
                out.append(SearchResult(title, url, snippet, "github"))
        return out
    except Exception:  # noqa: BLE001
        return []


async def _search_stackoverflow(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """Stack Exchange API — Q&A signal for technical topics.

    No key required for 300 req/day per IP. Returns top-voted matching questions
    from the StackOverflow site. Excerpt from question body populates snippet.
    """
    try:
        r = await client.get(
            "https://api.stackexchange.com/2.3/search/advanced",
            params={
                "order": "desc", "sort": "votes",
                "q": query, "site": "stackoverflow",
                "pagesize": str(n), "filter": "withbody",
            },
            headers={**_BROWSER_HEADERS, "Accept": "application/json"},
            timeout=15.0,
        )
        if r.status_code != 200:
            return []
        data = r.json()
        out: list[SearchResult] = []
        for q in data.get("items", [])[:n]:
            title = q.get("title", "")
            url = q.get("link", "")
            score = q.get("score", 0)
            body = re.sub(r"<[^>]+>", " ", q.get("body", "") or "")
            snippet = f"+{score} · {body}"[:280]
            if title and url:
                out.append(SearchResult(title, url, snippet, "stackoverflow"))
        return out
    except Exception:  # noqa: BLE001
        return []


async def _search_google_news(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """Google News RSS — no key, no signup, ~100 items per query.

    By far the highest-yield source — Google News indexes 50,000+ publishers
    and a single query returns ~100 RSS items vs Bing News's 1-4. The redirect
    URLs are intentionally opaque (news.google.com/rss/articles/...) but the
    title + description give enough signal for the embedding pass downstream.
    """
    try:
        r = await client.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers=_BROWSER_HEADERS, timeout=15.0,
        )
        if r.status_code != 200:
            return []
        text = r.text
        out: list[SearchResult] = []
        for item in re.finditer(r"<item>(.*?)</item>", text, re.DOTALL):
            body = item.group(1)
            t = re.search(r"<title>(.*?)</title>", body, re.DOTALL)
            l = re.search(r"<link>(.*?)</link>", body, re.DOTALL)
            d = re.search(r"<description>(.*?)</description>", body, re.DOTALL)
            if t and l:
                title = re.sub(r"<[^>]+>", "", t.group(1)).strip()
                url = l.group(1).strip()
                snippet = re.sub(r"<[^>]+>", "", (d.group(1) if d else "")).strip()[:280]
                if title:
                    out.append(SearchResult(title, url, snippet or title, "google_news"))
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


async def _search_bing_news(client: httpx.AsyncClient, query: str, n: int = 5) -> list[SearchResult]:
    """Bing News RSS — no API key, strong coverage for current events / blog posts.

    Closes a gap left by DDG/Wikipedia/HN/arXiv: business-strategy, marketing,
    industry-trend topics (e.g. "hair salon e-commerce conversion") return
    nothing from the existing four sources but show up well in Bing News.
    """
    try:
        r = await client.get(
            "https://www.bing.com/news/search",
            params={"q": query, "format": "rss"},
            headers=_BROWSER_HEADERS, timeout=15.0,
        )
        if r.status_code != 200:
            return []
        text = r.text
        out: list[SearchResult] = []
        # Parse RSS items minimally (no xml lib dep)
        for item in re.finditer(r"<item>(.*?)</item>", text, re.DOTALL):
            body = item.group(1)
            t = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", body, re.DOTALL)
            l = re.search(r"<link>(.*?)</link>", body, re.DOTALL)
            d = re.search(r"<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>", body, re.DOTALL)
            if t and l:
                title = re.sub(r"<[^>]+>", "", t.group(1)).strip()
                url = l.group(1).strip()
                snippet = re.sub(r"<[^>]+>", "", (d.group(1) if d else "")).strip()[:280]
                if title:
                    out.append(SearchResult(title, url, snippet or title, "bing_news"))
            if len(out) >= n:
                break
        return out
    except Exception:  # noqa: BLE001
        return []


async def search_multi(query: str, per_source: int = 4) -> list[SearchResult]:
    """Fan-out search across all 8 sources in parallel. Returns interleaved
    results so downstream chunking gets diversity even if it cuts the list short.
    """
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _search_ddg(client, query, per_source),
            _search_wikipedia(client, query, per_source),
            _search_hackernews(client, query, per_source),
            _search_arxiv(client, query, per_source),
            _search_bing_news(client, query, per_source),
            _search_google_news(client, query, per_source),
            _search_github(client, query, per_source),
            _search_stackoverflow(client, query, per_source),
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
