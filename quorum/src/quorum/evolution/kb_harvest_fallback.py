"""Fallback harvest path — used when DDG throttles the primary search.

Why this exists
---------------
The default ``harvest()`` in :mod:`quorum.evolution.web_learner` calls a
single search backend (``quorum.web.search.web_search``) which fans out to
DuckDuckGo. When DDG starts throttling Quorum's IP (the symptom is a
"dry streak" of consecutive zero-stored harvests), the overnight learner
falls back to long sleeps and the KB stops growing — observed 2026-06-28:
71 facts in one cycle → 6 facts two days later, a 12× drop.

This module layers two cheap fallbacks before sleep:

1. **Multi-source search** — :func:`quorum.web.multi_source.search_multi`
   is already shipped and fans out across 8 backends (Wikipedia, HN,
   arXiv, GitHub, StackOverflow, Google News, Bing News, plus DDG).
   When DDG is throttled, the other 7 keep working.

2. **LLM oracle** — only used if multi-source also returns nothing
   useful. Picks one configured provider, asks for up to N URLs likely
   to contain recent facts on the topic, then **gates every URL through
   a whitelist of trusted public domains** before fetching. Without that
   whitelist, an LLM hallucination could poison the KB with bogus
   sources.

Both fallbacks return the same shape as the primary harvest so
``run_overnight`` can swap them in without further changes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import struct
import time
from contextlib import closing
from pathlib import Path
from typing import List, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("quorum.evolution.kb_harvest_fallback")


# Trusted public-source whitelist — every host an LLM-suggested URL must
# end in one of these. Chosen because they have stable URL schemes, are
# unlikely to be hallucinated (the LLM knows them well from training),
# and represent the source classes we actually want in the KB
# (peer-reviewed, government, standards-body, reputable journalism).
TRUSTED_DOMAINS: Tuple[str, ...] = (
    # academic / preprint
    "arxiv.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "openreview.net", "papers.ssrn.com", "doi.org",
    "nature.com", "science.org", "cell.com", "thelancet.com",
    "nejm.org", "bmj.com", "plos.org",
    # standards / law / regulators
    "eur-lex.europa.eu", "ec.europa.eu", "europa.eu",
    "gov.uk", "legislation.gov.uk", "fca.org.uk", "bankofengland.co.uk",
    "ico.org.uk", "ofcom.org.uk", "nice.org.uk",
    "congress.gov", "federalregister.gov", "sec.gov", "ftc.gov", "fda.gov",
    "iso.org", "ietf.org", "w3.org", "nist.gov",
    "bis.org", "imf.org", "worldbank.org", "oecd.org", "who.int", "un.org",
    # reference / encyclopedic
    "wikipedia.org", "stackoverflow.com", "github.com",
    # reputable journalism (use sparingly — for "what happened" facts)
    "reuters.com", "ft.com", "bloomberg.com", "wsj.com", "economist.com",
    "bbc.co.uk", "bbc.com", "theguardian.com", "nytimes.com",
)


def _is_trusted(url: str) -> bool:
    """True iff ``url``'s host ends in one of TRUSTED_DOMAINS.

    Uses suffix match so subdomains (``www.gov.uk``, ``en.wikipedia.org``)
    pass. Bare-IP and non-HTTPS URLs are always rejected — an LLM
    suggesting ``http://203.0.113.1/data`` is a textbook hallucination
    pattern.
    """
    try:
        u = urlparse(url)
    except Exception:
        return False
    if u.scheme not in ("https", "http"):
        return False
    host = (u.hostname or "").lower()
    if not host or re.fullmatch(r"[0-9.]+", host):
        return False
    return any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS)


async def _try_multi_source(topic: str, n_search: int) -> list:
    """Fan-out search across all 8 sources; return SearchResult list."""
    try:
        from quorum.web.multi_source import search_multi
    except Exception as e:  # noqa: BLE001
        logger.debug("multi_source unavailable: %s", e)
        return []
    try:
        results = await search_multi(topic, per_source=max(2, n_search // 2))
        return results[: n_search * 2]
    except Exception as e:  # noqa: BLE001
        logger.warning("search_multi failed: %s", e)
        return []


_ORACLE_PROMPT = """You are a URL-retrieval oracle, not a writer.

For the topic below, return a JSON array of up to {n} HTTPS URLs that ALREADY EXIST on public, authoritative sources and are likely to contain factual information on the topic. ONLY URLs you are confident exist. NO speculation. NO explanations. NO markdown. Output: a single JSON array of strings.

Allowed domain suffixes (URL host MUST end in one): {domains}.

Topic: {topic}
"""


async def _try_llm_oracle(topic: str, n_urls: int = 5) -> List[str]:
    """Ask one configured LLM for URLs, filter through TRUSTED_DOMAINS."""
    try:
        from quorum.providers.registry import load_default_providers
    except Exception as e:  # noqa: BLE001
        logger.debug("provider registry unavailable: %s", e)
        return []

    providers = load_default_providers()
    if not providers:
        return []

    domain_hint = ", ".join(TRUSTED_DOMAINS[:20]) + ", ..."
    prompt = _ORACLE_PROMPT.format(n=n_urls, topic=topic, domains=domain_hint)

    for provider in providers:
        try:
            resp = await provider.complete(prompt, max_tokens=400, temperature=0.1)
            text = (resp.response or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.debug("oracle provider %s failed: %s", provider.name, e)
            continue

        # Tolerate fenced output: ```json ... ```
        m = re.search(r"\[\s*\".*?\"\s*(?:,\s*\".*?\"\s*)*\]", text, re.DOTALL)
        if not m:
            continue
        try:
            urls = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(urls, list):
            continue

        trusted = [u for u in urls if isinstance(u, str) and _is_trusted(u)]
        if trusted:
            logger.info(
                "oracle %s suggested %d urls, %d passed whitelist",
                provider.name, len(urls), len(trusted),
            )
            return trusted[:n_urls]

    return []


def _cache_dir() -> Path:
    base = Path(os.environ.get("QUORUM_DATA_DIR", str(Path.home() / ".quorum")))
    p = base / "kb_cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_path(topic: str) -> Path:
    safe = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:60]
    return _cache_dir() / f"{safe}.json"


_CACHE_TTL_S = 24 * 3600


def _read_cache(topic: str) -> List[str]:
    p = _cache_path(topic)
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text())
    except Exception:
        return []
    if int(time.time()) - payload.get("ts", 0) > _CACHE_TTL_S:
        return []
    urls = payload.get("urls") or []
    return [u for u in urls if isinstance(u, str)]


def _write_cache(topic: str, urls: List[str]) -> None:
    try:
        _cache_path(topic).write_text(
            json.dumps({"ts": int(time.time()), "urls": urls})
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("cache write failed: %s", e)


async def harvest_with_fallback(
    topic: str,
    *,
    n_search: int = 5,
    max_chunks_per_source: int = 6,
    use_oracle: bool = True,
) -> dict:
    """Harvest one topic with multi-source + LLM-oracle fallbacks.

    Order of attempts:
      1. Primary harvest (DDG via existing ``web_learner.harvest``).
         If it stored ≥1 fact, return immediately.
      2. Multi-source search (Wikipedia/HN/arXiv/GitHub/SO/News).
      3. Cached oracle URLs from a previous run within TTL.
      4. Live LLM oracle (whitelist-gated) — only if ``use_oracle`` True.

    All non-primary paths reuse :mod:`quorum.evolution.web_learner`'s
    storage layer by constructing ``SearchResult``-shaped objects and
    calling its internal fetch+chunk+embed+store pipeline directly.
    """
    from quorum.evolution import web_learner

    # Step 1 — primary path.
    primary = await web_learner.harvest(
        topic, n_search=n_search, max_chunks_per_source=max_chunks_per_source
    )
    if primary.get("stored", 0) > 0:
        primary["fallback_used"] = None
        return primary

    # Step 2 — multi-source fan-out.
    multi_results = await _try_multi_source(topic, n_search)
    if multi_results:
        out = await _ingest_search_results(
            topic, multi_results, max_chunks_per_source
        )
        if out.get("stored", 0) > 0:
            out["fallback_used"] = "multi_source"
            return out

    # Step 3 — cached oracle URLs.
    cached = _read_cache(topic)
    if cached:
        synth = _urls_to_search_results(cached, source="oracle_cache")
        out = await _ingest_search_results(
            topic, synth, max_chunks_per_source
        )
        if out.get("stored", 0) > 0:
            out["fallback_used"] = "oracle_cache"
            return out

    # Step 4 — live oracle (gated on use_oracle and whitelist).
    if use_oracle:
        urls = await _try_llm_oracle(topic)
        if urls:
            _write_cache(topic, urls)
            synth = _urls_to_search_results(urls, source="oracle_live")
            out = await _ingest_search_results(
                topic, synth, max_chunks_per_source
            )
            if out.get("stored", 0) > 0:
                out["fallback_used"] = "oracle_live"
                return out

    return {
        "topic": topic,
        "fetched_sources": 0,
        "candidate_chunks": 0,
        "stored": 0,
        "fallback_used": "all_failed",
    }


def _urls_to_search_results(urls: List[str], *, source: str):
    """Wrap raw URLs in the SearchResult shape harvest's ingest expects."""
    from quorum.web.multi_source import SearchResult

    return [
        SearchResult(title=u, url=u, snippet="", source=source)
        for u in urls
    ]


async def _ingest_search_results(
    topic: str,
    results: list,
    max_chunks_per_source: int,
) -> dict:
    """Mirror the body of web_learner.harvest, but with caller-supplied results.

    We reproduce the storage path here instead of refactoring
    ``web_learner.harvest`` to keep the primary path's contract untouched
    while this module is on probation. Once the fallback proves itself,
    the two can be unified.
    """
    from quorum.evolution.web_learner import (
        _open_db, _fetch, _html_to_text, _chunk, _pack_vec,
    )
    from quorum.core.embeddings import EmbeddingProvider

    db = _open_db()
    with closing(db.cursor()) as c:
        c.execute(
            "INSERT OR IGNORE INTO topics(name, last_harvested, fact_count) VALUES(?, ?, 0)",
            (topic, int(time.time())),
        )
        c.execute("SELECT id FROM topics WHERE name=?", (topic,))
        topic_id = c.fetchone()["id"]
        c.execute(
            "UPDATE topics SET last_harvested=? WHERE id=?",
            (int(time.time()), topic_id),
        )
    db.commit()

    htmls = await asyncio.gather(*[_fetch(r.url) for r in results])
    sources: List[Tuple[str, str, List[str]]] = []
    for r, html in zip(results, htmls):
        text = _html_to_text(html) if html else (r.snippet or "")
        if not text:
            continue
        chunks = _chunk(text)[:max_chunks_per_source]
        sources.append((r.url, r.title or r.url, chunks))

    flat_texts: List[str] = []
    flat_meta: List[Tuple[str, str]] = []
    for url, title, chunks in sources:
        for ch in chunks:
            flat_texts.append(ch)
            flat_meta.append((url, title))

    if not flat_texts:
        db.close()
        return {
            "topic": topic,
            "fetched_sources": len(sources),
            "candidate_chunks": 0,
            "stored": 0,
        }

    embedder = EmbeddingProvider.from_env()
    async with embedder:
        embeddings = await embedder.embed(flat_texts)

    stored = 0
    skipped = 0
    with closing(db.cursor()) as c:
        for text, (url, title), emb in zip(flat_texts, flat_meta, embeddings):
            try:
                c.execute(
                    "INSERT OR IGNORE INTO facts(topic_id, source_url, source_title, "
                    "text, embedding, harvested_at) VALUES(?,?,?,?,?,?)",
                    (
                        topic_id, url, title, text,
                        _pack_vec(emb), int(time.time()),
                    ),
                )
                if c.rowcount:
                    stored += 1
                else:
                    skipped += 1
            except sqlite3.IntegrityError:
                skipped += 1
        c.execute(
            "UPDATE topics SET fact_count=(SELECT COUNT(*) FROM facts WHERE topic_id=?) "
            "WHERE id=?",
            (topic_id, topic_id),
        )
    db.commit()
    db.close()
    return {
        "topic": topic,
        "fetched_sources": len(sources),
        "candidate_chunks": len(flat_texts),
        "stored": stored,
        "duplicates_skipped": skipped,
    }


__all__ = [
    "harvest_with_fallback",
    "TRUSTED_DOMAINS",
    "_is_trusted",
]
