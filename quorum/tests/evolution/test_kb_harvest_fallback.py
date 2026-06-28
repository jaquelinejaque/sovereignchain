"""Tests for kb_harvest_fallback — security-critical: an LLM-suggested URL
that bypasses the whitelist would let a hallucinated source land in the KB
and corrupt every future ``recall()`` answer.

Coverage:
- Whitelist accept-list: each canonical trusted host is accepted.
- Whitelist reject patterns: typosquats, IP literals, non-HTTPS schemes,
  and look-alikes that *contain* a trusted name as substring.
- Cache TTL: a stale cache entry must be ignored.
- harvest_with_fallback returns the no-op shape when everything fails.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from quorum.evolution import kb_harvest_fallback as fb


# ---------- whitelist --------------------------------------------------------


class TestIsTrusted:
    """The single most important function in this module: it is the gate
    between hallucinated URLs and the persistent KB."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://arxiv.org/abs/2401.12345",
            "https://www.gov.uk/government/publications/foo",
            "https://en.wikipedia.org/wiki/Multi-LLM_consensus",
            "https://eur-lex.europa.eu/legal-content/EN/TXT/",
            "https://pubmed.ncbi.nlm.nih.gov/12345",
            "https://www.fca.org.uk/news/example",
            "https://nature.com/articles/foo",
            "https://github.com/anthropics/claude-code",
        ],
    )
    def test_accepts_trusted_hosts(self, url: str) -> None:
        assert fb._is_trusted(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            # IP literal — classic hallucination shape.
            "http://203.0.113.1/data",
            "https://203.0.113.1/data",
            # Typosquat with a trusted name in the middle/start.
            "https://evil-arxiv.org/abs/123",
            "https://arxiv.org.evil.com/abs/123",
            "https://gov-uk.attacker.io/policy",
            # Wrong scheme.
            "ftp://gov.uk/foo",
            # Empty / malformed.
            "not a url at all",
            "",
            # Substack / Medium / random blogs are NOT trusted — bar must
            # stay high; popular ≠ canonical.
            "https://random-blog.substack.com/p/post",
            "https://medium.com/@author/post",
            "https://news.ycombinator.com/item?id=1",  # HN is fine via _search_hackernews, NOT for oracle output
        ],
    )
    def test_rejects_untrusted_or_malformed(self, url: str) -> None:
        assert fb._is_trusted(url) is False

    def test_subdomain_of_trusted_passes(self) -> None:
        # Real-world: en.wikipedia.org, www.gov.uk, www.ncbi.nlm.nih.gov.
        for url in [
            "https://en.wikipedia.org/wiki/Foo",
            "https://pt.wikipedia.org/wiki/Foo",
            "https://www.bbc.co.uk/news/foo",
        ]:
            assert fb._is_trusted(url) is True, url


# ---------- cache TTL --------------------------------------------------------


class TestCache:
    """A fresh cache entry is reused; an expired one is silently dropped so
    the loop re-issues the oracle query."""

    def test_round_trip(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("QUORUM_DATA_DIR", str(tmp_path))
        topic = "EU AI Act Article 12 audit requirements"
        urls = ["https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R0001"]
        fb._write_cache(topic, urls)
        assert fb._read_cache(topic) == urls

    def test_expired_cache_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("QUORUM_DATA_DIR", str(tmp_path))
        topic = "stale topic"
        # Write a payload as if it were 25 hours old.
        p = fb._cache_path(topic)
        p.write_text(json.dumps({
            "ts": int(time.time()) - 25 * 3600,
            "urls": ["https://arxiv.org/abs/old"],
        }))
        assert fb._read_cache(topic) == []

    def test_corrupt_cache_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("QUORUM_DATA_DIR", str(tmp_path))
        topic = "broken"
        fb._cache_path(topic).write_text("{not valid json")
        assert fb._read_cache(topic) == []


# ---------- harvest_with_fallback end-to-end ---------------------------------


class TestHarvestWithFallback:
    """The orchestrator's contract: it must NEVER raise, even when every
    layer fails — overnight depends on a structured return."""

    def test_returns_failure_shape_when_everything_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("QUORUM_DATA_DIR", str(tmp_path))

        async def _empty_primary(*a, **kw):
            return {"topic": a[0] if a else "x", "stored": 0,
                    "fetched_sources": 0, "candidate_chunks": 0}

        async def _empty_multi(*a, **kw):
            return []

        async def _empty_oracle(*a, **kw):
            return []

        with patch("quorum.evolution.web_learner.harvest", side_effect=_empty_primary):
            with patch(
                "quorum.evolution.kb_harvest_fallback._try_multi_source",
                side_effect=_empty_multi,
            ):
                with patch(
                    "quorum.evolution.kb_harvest_fallback._try_llm_oracle",
                    side_effect=_empty_oracle,
                ):
                    out = asyncio.run(fb.harvest_with_fallback("xx"))

        assert out["stored"] == 0
        assert out["fallback_used"] == "all_failed"

    def test_short_circuits_on_primary_success(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """When DDG works, the fallback path must NOT be reached — paying
        for the oracle on a healthy harvest would be a regression."""
        monkeypatch.setenv("QUORUM_DATA_DIR", str(tmp_path))

        async def _primary_works(*a, **kw):
            return {"topic": a[0], "stored": 5, "fetched_sources": 4,
                    "candidate_chunks": 12}

        oracle_calls = []

        async def _oracle_spy(*a, **kw):
            oracle_calls.append(a)
            return ["https://arxiv.org/x"]

        with patch("quorum.evolution.web_learner.harvest", side_effect=_primary_works):
            with patch(
                "quorum.evolution.kb_harvest_fallback._try_llm_oracle",
                side_effect=_oracle_spy,
            ):
                out = asyncio.run(fb.harvest_with_fallback("x"))

        assert out["stored"] == 5
        assert out.get("fallback_used") is None
        assert oracle_calls == []  # never reached oracle
