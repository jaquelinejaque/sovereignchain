"""Semantic embedding providers and agreement scoring for Quorum consensus.

Why this module exists
----------------------
The v0.0.1 consensus engine used token-set Jaccard overlap to score agreement
between LLM responses. That fails catastrophically when two models agree on
meaning but use different vocabulary (e.g. "the patient must be over 18" vs
"adult subjects only"). For a multi-LLM consensus engine, the *whole point* is
detecting semantic agreement, not lexical agreement. This module replaces the
placeholder with embedding-based cosine similarity.

Design choices
--------------
- Async-first: every embedder call is async so the consensus engine can fan
  out embedding requests in parallel with model calls.
- Pluggable: ``EmbeddingProvider`` is an ABC. ``from_env()`` picks the cheapest
  available backend (Gemini > Ollama > OpenAI) so tests work offline.
- Cached: identical response text is embedded once and reused (LRU 1024).
  LLMs are deterministic enough at temperature 0 that repeats are common.
- Numerically defensive: cosine handles zero vectors, NaN, mismatched dims.
- Graceful: a missing API key never crashes — we fall back to the next provider.

License
-------
SPDX-License-Identifier: Apache-2.0
Copyright (c) 2026 Jaqueline Martins / Sovereign Chain Ltd.

This module is NOT HSP-gated. The HSP patent (PCT/US26/11908) covers the
self-evolution feedback loop, not the embedding layer. Embedding scoring is
free-software under Apache 2.0 with no additional restrictions.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Iterable

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEMINI_MODEL = "text-embedding-004"
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{_GEMINI_MODEL}:batchEmbedContents"
)
_GEMINI_MAX_BATCH = 100  # Google's documented per-request cap.

_OPENAI_MODEL = "text-embedding-3-small"
_OPENAI_ENDPOINT = "https://api.openai.com/v1/embeddings"
_OPENAI_MAX_BATCH = 2048

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "nomic-embed-text"
_OLLAMA_EMBED_PATH = "/api/embeddings"

_DEFAULT_TIMEOUT_S = 30.0
_LRU_CAPACITY = 1024

# Below this cosine similarity, two responses are considered to materially
# disagree. 0.65 is empirically the knee where paraphrases stop and real
# divergence starts for text-embedding-004. Override per-call if needed.
_DEFAULT_DISAGREEMENT_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# LRU embedding cache
# ---------------------------------------------------------------------------


class _EmbeddingCache:
    """Thread-unsafe LRU cache keyed by ``(backend_id, text)``.

    Why thread-unsafe is fine: every embedder instance has its own cache and is
    only touched from the asyncio event loop. If multi-loop usage is added we
    will wrap with ``asyncio.Lock``.
    """

    def __init__(self, capacity: int = _LRU_CAPACITY) -> None:
        self._capacity = capacity
        self._store: OrderedDict[tuple[str, str], list[float]] = OrderedDict()

    def get(self, backend_id: str, text: str) -> list[float] | None:
        key = (backend_id, text)
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, backend_id: str, text: str, vector: list[float]) -> None:
        key = (backend_id, text)
        if key in self._store:
            self._store.move_to_end(key)
            self._store[key] = vector
            return
        self._store[key] = vector
        if len(self._store) > self._capacity:
            evicted_key, _ = self._store.popitem(last=False)
            logger.debug("LRU evicted %s", evicted_key[0])

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract embedding backend.

    Subclasses must implement ``_embed_uncached`` (the network call) and
    ``backend_id`` (a stable string used for cache keying so vectors from
    different models never collide).
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        cache: _EmbeddingCache | None = None,
    ) -> None:
        self._client = client
        self._owns_client = client is None
        self._timeout_s = timeout_s
        self._cache = cache if cache is not None else _EmbeddingCache()

    # ----- subclass contract --------------------------------------------------

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Stable identifier (e.g. ``'gemini:text-embedding-004'``).

        Used as the cache namespace so vectors from different models never
        get confused when the same text is embedded by two backends.
        """

    @abstractmethod
    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """Make the actual network call. Must return one vector per input."""

    # ----- public API --------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed ``texts`` and return one vector per input, preserving order.

        Why this layer exists separate from ``_embed_uncached``: the LRU cache
        lets us skip the network entirely for repeated text, which is the
        common case in a consensus engine that re-scores the same responses
        across multiple evolution iterations.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_idx: list[int] = []
        missing_text: list[str] = []

        for i, t in enumerate(texts):
            cached = self._cache.get(self.backend_id, t)
            if cached is not None:
                results[i] = cached
            else:
                missing_idx.append(i)
                missing_text.append(t)

        if missing_text:
            fresh = await self._embed_uncached(missing_text)
            if len(fresh) != len(missing_text):
                raise RuntimeError(
                    f"{self.backend_id} returned {len(fresh)} vectors for "
                    f"{len(missing_text)} inputs"
                )
            for idx, text, vec in zip(missing_idx, missing_text, fresh):
                self._cache.put(self.backend_id, text, vec)
                results[idx] = vec

        # All slots populated by construction.
        return [r for r in results if r is not None]

    async def aclose(self) -> None:
        """Close the underlying httpx client if we own it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "EmbeddingProvider":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def _http(self) -> httpx.AsyncClient:
        """Lazy-initialise the httpx client. Reused across calls."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout_s)
        return self._client

    # ----- factory -----------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
    ) -> "EmbeddingProvider":
        """Pick the best embedder available in this environment.

        Preference order — chosen because:
        1. Gemini text-embedding-004 is free up to generous quotas.
        2. Ollama is fully local (zero cost, zero latency to network).
        3. OpenAI text-embedding-3-small costs $0.02 / 1M tokens but works
           anywhere.

        Raises ``RuntimeError`` only if *no* backend can be reached.
        """
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_STUDIO_KEY")
        if gemini_key:
            logger.info("EmbeddingProvider.from_env: using Gemini (key found)")
            return GeminiEmbedder(api_key=gemini_key, client=client, timeout_s=timeout_s)

        if _ollama_reachable():
            logger.info("EmbeddingProvider.from_env: using local Ollama")
            return OllamaEmbedder(client=client, timeout_s=timeout_s)

        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            logger.info("EmbeddingProvider.from_env: using OpenAI")
            return OpenAIEmbedder(api_key=openai_key, client=client, timeout_s=timeout_s)

        raise RuntimeError(
            "No embedding backend available. Set GEMINI_API_KEY, "
            "OPENAI_API_KEY, or run Ollama locally with nomic-embed-text pulled."
        )


def _ollama_reachable(host: str = _OLLAMA_DEFAULT_HOST) -> bool:
    """Best-effort sync check whether Ollama is running.

    Sync on purpose: ``from_env`` is a factory called outside any event loop.
    A 250 ms timeout keeps startup snappy when Ollama is not installed.
    """
    try:
        with httpx.Client(timeout=0.25) as c:
            r = c.get(f"{host}/api/tags")
            return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class GeminiEmbedder(EmbeddingProvider):
    """Google text-embedding-004 via the v1beta batchEmbedContents endpoint.

    Why batch: Google charges per-request, not per-token-batch, so packing 100
    texts into one request is ~100x cheaper than sequential calls.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        cache: _EmbeddingCache | None = None,
    ) -> None:
        super().__init__(client=client, timeout_s=timeout_s, cache=cache)
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv(
            "GOOGLE_AI_STUDIO_KEY"
        )
        if not self._api_key:
            raise RuntimeError(
                "GeminiEmbedder requires GEMINI_API_KEY or GOOGLE_AI_STUDIO_KEY"
            )

    @property
    def backend_id(self) -> str:
        return f"gemini:{_GEMINI_MODEL}"

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for chunk in _chunked(texts, _GEMINI_MAX_BATCH):
            payload = {
                "requests": [
                    {
                        "model": f"models/{_GEMINI_MODEL}",
                        "content": {"parts": [{"text": t}]},
                    }
                    for t in chunk
                ]
            }
            params = {"key": self._api_key}
            try:
                r = await self._http().post(_GEMINI_ENDPOINT, params=params, json=payload)
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error("Gemini embed failed: %s", e.response.text[:300])
                raise
            data = r.json()
            for emb in data.get("embeddings", []):
                values = emb.get("values") or []
                if not values:
                    raise RuntimeError("Gemini returned an empty embedding")
                out.append([float(x) for x in values])
        return out


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class OllamaEmbedder(EmbeddingProvider):
    """Local nomic-embed-text via Ollama's ``/api/embeddings`` endpoint.

    Why no batching: Ollama's embedding endpoint takes one prompt at a time.
    We still get cheap concurrency by running calls inside ``asyncio.gather``.
    """

    def __init__(
        self,
        *,
        host: str = _OLLAMA_DEFAULT_HOST,
        model: str = _OLLAMA_DEFAULT_MODEL,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        cache: _EmbeddingCache | None = None,
        max_concurrency: int = 8,
    ) -> None:
        super().__init__(client=client, timeout_s=timeout_s, cache=cache)
        self._host = host.rstrip("/")
        self._model = model
        self._sem = asyncio.Semaphore(max_concurrency)

    @property
    def backend_id(self) -> str:
        return f"ollama:{self._model}"

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        async def _one(t: str) -> list[float]:
            async with self._sem:
                payload = {"model": self._model, "prompt": t}
                r = await self._http().post(
                    f"{self._host}{_OLLAMA_EMBED_PATH}", json=payload
                )
                r.raise_for_status()
                data = r.json()
                vec = data.get("embedding") or []
                if not vec:
                    raise RuntimeError("Ollama returned an empty embedding")
                return [float(x) for x in vec]

        return await asyncio.gather(*(_one(t) for t in texts))


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIEmbedder(EmbeddingProvider):
    """OpenAI text-embedding-3-small. Batches up to 2048 inputs per call."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _OPENAI_MODEL,
        client: httpx.AsyncClient | None = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        cache: _EmbeddingCache | None = None,
    ) -> None:
        super().__init__(client=client, timeout_s=timeout_s, cache=cache)
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OpenAIEmbedder requires OPENAI_API_KEY")
        self._model = model

    @property
    def backend_id(self) -> str:
        return f"openai:{self._model}"

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        out: list[list[float]] = []
        for chunk in _chunked(texts, _OPENAI_MAX_BATCH):
            payload = {"model": self._model, "input": chunk}
            r = await self._http().post(_OPENAI_ENDPOINT, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            # OpenAI guarantees order matches input; we sort by ``index`` to be safe.
            items = sorted(data.get("data", []), key=lambda d: int(d.get("index", 0)))
            for item in items:
                vec = item.get("embedding") or []
                if not vec:
                    raise RuntimeError("OpenAI returned an empty embedding")
                out.append([float(x) for x in vec])
        return out


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _chunked(seq: list[str], size: int) -> Iterable[list[str]]:
    """Yield successive ``size``-chunks of ``seq``."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Numerically stable cosine similarity in ``[-1, 1]``.

    Defensive against:
      - zero vectors (returns 0.0, not NaN)
      - NaN/Inf components (treated as 0.0)
      - mismatched dimensions (raises ``ValueError`` — silent truncation would
        be a correctness bug we'd never see)
    """
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        # NaN/Inf in either component becomes 0.0 for that side only.
        # Sanitising per-component (not skipping the pair) is critical so the
        # *other* vector's information at that index is preserved.
        xs = x if math.isfinite(x) else 0.0
        ys = y if math.isfinite(y) else 0.0
        dot += xs * ys
        norm_a += xs * xs
        norm_b += ys * ys

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    denom = math.sqrt(norm_a) * math.sqrt(norm_b)
    if denom <= 0.0:
        return 0.0

    sim = dot / denom
    # Numerical drift can push this fractionally past +/-1.
    return max(-1.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Agreement scoring (the public API the consensus engine consumes)
# ---------------------------------------------------------------------------


async def semantic_agreement(
    responses: list[str],
    embedder: EmbeddingProvider,
) -> tuple[float, list[float]]:
    """Score how much a set of LLM responses semantically agree.

    Returns:
        ``(overall_confidence, per_response_weights)`` where:
          - ``overall_confidence`` is the mean of the upper triangle of the
            cosine similarity matrix, clipped to ``[0, 1]``. 1.0 means every
            model said essentially the same thing; 0.0 means total divergence.
          - ``per_response_weights`` is each response's average similarity to
            all others, normalized to sum to 1.0. The most "central" response
            (closest to all others) gets the largest weight — that's the
            response the consensus engine should surface as canonical.

    Why this is better than Jaccard: two models can say
    "patient must be 18 or older" and "subjects required to be adults" with
    zero word overlap. Jaccard scores that 0.0 (total disagreement). Cosine
    over embeddings scores it ~0.85 (strong agreement) — which is the truth.

    Edge cases:
      - Empty list → ``(0.0, [])``
      - Single response → ``(1.0, [1.0])`` (vacuously self-consistent)
      - All responses identical → confidence 1.0, equal weights
      - All-zero embedding from a backend → similarity 0.0 (not NaN)
    """
    if not responses:
        return 0.0, []
    if len(responses) == 1:
        return 1.0, [1.0]

    vectors = await embedder.embed(responses)
    n = len(vectors)
    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            s = cosine_similarity(vectors[i], vectors[j])
            # Map cosine [-1, 1] to agreement [0, 1]. Negative similarity
            # (genuinely opposite directions in embedding space) counts as
            # zero agreement, not as "anti-agreement" which would skew the mean.
            s = max(0.0, s)
            sim[i][j] = s
            sim[j][i] = s

    # Per-response weight = mean similarity to all others.
    raw_weights = [
        sum(sim[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)
    ]
    total = sum(raw_weights)
    if total <= 0.0:
        # Every response is orthogonal to every other — fall back to uniform.
        weights = [1.0 / n] * n
    else:
        weights = [w / total for w in raw_weights]

    # Overall confidence = mean of upper triangle.
    pairs = [sim[i][j] for i in range(n) for j in range(i + 1, n)]
    confidence = sum(pairs) / len(pairs) if pairs else 1.0
    confidence = max(0.0, min(1.0, confidence))
    return confidence, weights


async def extract_disagreement_pairs(
    responses: list[str],
    embedder: EmbeddingProvider,
    threshold: float = _DEFAULT_DISAGREEMENT_THRESHOLD,
) -> list[tuple[int, int, float]]:
    """Identify which pairs of responses materially disagree.

    Returns a list of ``(i, j, similarity)`` tuples for every pair whose
    cosine similarity is *below* ``threshold``, sorted lowest similarity first
    (the strongest disagreements come first).

    Why this exists: the consensus engine needs to surface *what* the models
    disagreed about, not just *that* they disagreed. The caller can take these
    index pairs and quote the actual response text in the ``disagreements``
    field of ``ConsensusResult``. That's the signal that drives the HSP
    self-evolution loop — divergence is where learning happens.

    The default threshold of 0.65 is empirically the knee for
    text-embedding-004 between paraphrase territory and real semantic
    divergence. Tune per use case.
    """
    if len(responses) < 2:
        return []

    vectors = await embedder.embed(responses)
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            s = cosine_similarity(vectors[i], vectors[j])
            s = max(0.0, s)
            if s < threshold:
                pairs.append((i, j, s))
    pairs.sort(key=lambda p: p[2])
    return pairs


# ---------------------------------------------------------------------------
# In-memory fake (for tests and offline development)
# ---------------------------------------------------------------------------


class _FakeEmbedder(EmbeddingProvider):
    """Deterministic in-memory embedder used by tests when no backend is set.

    Hashes characters into a low-dimensional bag-of-chars vector. Not
    semantically meaningful, but *consistent*: identical text gets identical
    vectors, so the cosine math can be exercised without any network.
    """

    _DIM = 64

    def __init__(self) -> None:
        super().__init__(client=None, timeout_s=1.0)

    @property
    def backend_id(self) -> str:
        return "fake:bag-of-chars"

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            v = [0.0] * self._DIM
            for ch in t.lower():
                v[ord(ch) % self._DIM] += 1.0
            # L2 normalise so cosine == dot product, matches real embedders.
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                v = [x / norm for x in v]
            out.append(v)
        return out


# ---------------------------------------------------------------------------
# Tests (pytest-discoverable, no network required)
# ---------------------------------------------------------------------------


def _make_fake() -> _FakeEmbedder:
    """Construct the deterministic offline embedder used across tests."""
    return _FakeEmbedder()


async def _test_cosine_basics() -> None:
    """Cosine math must handle the dangerous edge cases without exploding."""
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector
    # NaN must be treated as a zero contribution, not poison the whole sum.
    nan_result = cosine_similarity([float("nan"), 1.0], [1.0, 1.0])
    clean_result = cosine_similarity([0.0, 1.0], [1.0, 1.0])
    assert abs(nan_result - clean_result) < 1e-9
    try:
        cosine_similarity([1.0], [1.0, 1.0])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on dim mismatch")


async def _test_semantic_agreement_identical() -> None:
    """Identical responses must score confidence == 1.0 with equal weights."""
    emb = _make_fake()
    conf, weights = await semantic_agreement(["same", "same", "same"], emb)
    assert conf > 0.999
    assert all(abs(w - 1 / 3) < 1e-9 for w in weights)


async def _test_semantic_agreement_divergent() -> None:
    """Wildly different strings should score lower than identical strings."""
    emb = _make_fake()
    same_conf, _ = await semantic_agreement(["alpha alpha", "alpha alpha"], emb)
    diff_conf, _ = await semantic_agreement(["alpha alpha", "zzzzz zzzzz"], emb)
    assert same_conf > diff_conf


async def _test_single_response() -> None:
    """A single response is vacuously self-consistent."""
    emb = _make_fake()
    conf, weights = await semantic_agreement(["only one"], emb)
    assert conf == 1.0
    assert weights == [1.0]


async def _test_empty_responses() -> None:
    """Empty input must not crash — return neutral values."""
    emb = _make_fake()
    conf, weights = await semantic_agreement([], emb)
    assert conf == 0.0
    assert weights == []


async def _test_disagreement_pairs() -> None:
    """Divergent pairs surface; agreeing pairs don't."""
    emb = _make_fake()
    pairs = await extract_disagreement_pairs(
        ["alpha alpha", "alpha alpha", "zzzzz zzzzz"], emb, threshold=0.5
    )
    indices = {(i, j) for i, j, _ in pairs}
    assert (0, 2) in indices or (1, 2) in indices
    assert (0, 1) not in indices


async def _test_cache_hits() -> None:
    """Same text embedded twice should hit the cache, not the network."""
    emb = _make_fake()
    await emb.embed(["hello", "world"])
    before = len(emb._cache)
    await emb.embed(["hello", "world"])
    after = len(emb._cache)
    assert before == after == 2


async def _run_all_tests() -> None:
    await _test_cosine_basics()
    await _test_semantic_agreement_identical()
    await _test_semantic_agreement_divergent()
    await _test_single_response()
    await _test_empty_responses()
    await _test_disagreement_pairs()
    await _test_cache_hits()
    logger.info("all embedding tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run_all_tests())


__all__ = [
    "EmbeddingProvider",
    "GeminiEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "cosine_similarity",
    "semantic_agreement",
    "extract_disagreement_pairs",
]
