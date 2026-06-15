# Copyright 2026 Jaqueline Martins / Sovereign Chain
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# HSP (Hybrid Sovereign Protocol) attribution:
#   This loop is NOT HSP-gated. It stores artifacts produced by gated
#   consensus paths and must respect the user_id scoping that the
#   commercial-use restrictions in LICENSE-HSP (PCT/US26/11908) impose
#   on its upstream callers.
"""Loop 5 — Memory evolution.

Why this module exists
----------------------
A multi-LLM consensus call is a one-shot ensemble. What turns Quorum into a
*personal* consensus engine is the feedback loop: every query the user issues
enriches a per-user vector memory, and the most relevant past memories are
auto-injected into the next prompt. This module wires that loop end-to-end.

Two cadences are exposed:

1. **Real-time on ingest** — after each consensus call the caller invokes
   :meth:`MemoryEvolution.ingest` with the prompt and the synthesized answer.
   Both get embedded and stored as separate rows (``query`` and ``response``
   kinds) so future recall can match on either side of a Q/A pair.

2. **Weekly batch on auto_extract** — once a week a cron runs
   :meth:`MemoryEvolution.auto_extract_preferences`. It feeds the user's
   recent query history into a single Gemini call that distills durable
   long-term preferences (language, style, recurring topics) and promotes
   them as ``preference`` rows. Cheap (one extra Flash call per user per
   week) and high-signal (preferences survive ``age_out`` pruning).

Design notes
~~~~~~~~~~~~
* The embedder is caller-supplied to keep this module agnostic of which
  embedding backend (Gemini, OpenAI, Ollama) the user actually paid for.
* Gemini access for the weekly extractor is *optional*: if no API key is
  configured we fall back to a deterministic in-memory heuristic so tests
  and offline CI runs still exercise the full code path.
* SQLite I/O is delegated entirely to :class:`VectorMemory`, which already
  wraps sync calls in ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import httpx

from quorum.core.memory import MemoryHit, VectorMemory, make_memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Minimal contract for any embedding backend.

    Why a Protocol instead of the concrete ``EmbeddingProvider`` ABC: this
    module must work in tests that pass a tiny lambda-style stub. Protocol +
    ``runtime_checkable`` keeps the type hint precise without forcing the
    caller to subclass anything.
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in order."""
        ...


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestResult:
    """Bookkeeping for a single :meth:`MemoryEvolution.ingest` call.

    Returning ids (rather than just True/False) lets the caller log a
    trace span tying the consensus run to the memory rows it created — useful
    when debugging "why did the next call get this weird context?".
    """

    query_id: str
    response_id: str
    bytes_embedded: int


@dataclass(frozen=True)
class PreferenceExtraction:
    """Output of the weekly preference extractor."""

    statements: list[str]
    promoted_ids: list[str]
    source: str  # "gemini" | "heuristic"
    skipped_duplicates: int = 0


@dataclass
class _GeminiClient:
    """Tiny inline Gemini client.

    Why not reuse ``quorum.providers.gemini.GeminiProvider``: that provider
    returns a ``ModelResponse`` shaped for consensus voting. The extractor
    just needs raw text, no token accounting, no cost rollup. Inlining the
    httpx call (≈20 LOC) avoids the structural mismatch.
    """

    model: str = "gemini-2.5-flash"
    api_key: str = ""
    timeout_s: float = 20.0
    endpoint: str = "https://generativelanguage.googleapis.com/v1beta/models"

    @classmethod
    def from_env(cls) -> "_GeminiClient":
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_STUDIO_KEY") or ""
        return cls(api_key=key)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def complete(self, prompt: str, *, max_tokens: int = 400) -> str:
        if not self.api_key:
            return ""
        url = f"{self.endpoint}/{self.model}:generateContent?key={self.api_key}"
        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                r = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            logger.warning("memory_loop: gemini transport error: %s", exc)
            return ""
        if r.status_code != 200:
            logger.warning(
                "memory_loop: gemini http %d: %s", r.status_code, r.text[:120]
            )
            return ""
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)


# ---------------------------------------------------------------------------
# Heuristic fallback for preference extraction
# ---------------------------------------------------------------------------


_PREFERENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:i (?:prefer|like|love|want|need|hate|dislike))\b[^.?!\n]{3,140}", re.IGNORECASE),
    re.compile(r"\b(?:always|never) (?:answer|reply|respond|use|avoid)\b[^.?!\n]{3,140}", re.IGNORECASE),
    re.compile(r"\bremember(?: that)?\b[^.?!\n]{3,140}", re.IGNORECASE),
    # pt-br
    re.compile(r"\b(?:eu (?:prefiro|gosto|odeio|quero|preciso))\b[^.?!\n]{3,140}", re.IGNORECASE),
    re.compile(r"\b(?:sempre|nunca) (?:responda|use|evite|escreva)\b[^.?!\n]{3,140}", re.IGNORECASE),
    re.compile(r"\blembre(?:-se)?(?: que)?\b[^.?!\n]{3,140}", re.IGNORECASE),
)


def _heuristic_extract(recent_queries: Sequence[str], cap: int = 5) -> list[str]:
    """Regex-based preference distillation for the no-API-key path.

    Why ship this at all: the weekly job must remain useful in offline tests
    and air-gapped deployments. The regex set is intentionally conservative
    — it accepts only first-person preference statements with imperative or
    declarative shape, in English and pt-br (the two languages our v0.1
    users actually write in).
    """
    found: list[str] = []
    seen: set[str] = set()
    for q in recent_queries:
        for pat in _PREFERENCE_PATTERNS:
            for match in pat.findall(q):
                snippet = match.strip().strip(",;:")
                key = snippet.lower()
                if key in seen or len(snippet) < 8:
                    continue
                seen.add(key)
                found.append(snippet)
                if len(found) >= cap:
                    return found
    return found


def _parse_gemini_json_array(text: str) -> list[str]:
    """Pull a JSON array of strings out of a Gemini response.

    Gemini Flash sometimes wraps JSON in ```json fences or trailing prose.
    We try the strict parse first, then fall back to the first balanced
    ``[...]`` substring. Anything we cannot parse becomes an empty list
    rather than an exception — preference extraction is best-effort.
    """
    text = text.strip()
    if not text:
        return []
    # Strip common markdown fence
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    # Greedy balanced-array fallback
    m = re.search(r"\[[^\[\]]*\]", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            return []
    return []


# ---------------------------------------------------------------------------
# MemoryEvolution
# ---------------------------------------------------------------------------


_EXTRACTOR_PROMPT_TEMPLATE = """\
You are distilling DURABLE user preferences from recent queries to a
multi-LLM consensus assistant. A "preference" is a stable, long-lived fact
about how the user wants to be served: language, tone, depth, technical
domain, recurring constraints. Ignore one-off task content.

Return a JSON array of short imperative statements (max 12 words each,
max {cap} items), in the same language the user writes in. Return ONLY
the JSON array — no prose, no fences.

Recent queries (newest first):
{queries}

JSON array:"""


@dataclass
class MemoryEvolution:
    """High-level façade orchestrating real-time + weekly memory updates.

    Why a dataclass: the wired-up object is essentially `(store, gemini)` +
    a couple of tunables. ``@dataclass`` gives us a clean constructor and a
    repr without boilerplate, and `field(default_factory=...)` lets the
    Gemini client lazy-bind to env vars when the caller does not pass one.
    """

    store: VectorMemory = field(default_factory=make_memory)
    gemini: _GeminiClient = field(default_factory=_GeminiClient.from_env)
    max_extract_statements: int = 5
    response_excerpt_chars: int = 1200

    # ---- real-time path --------------------------------------------------

    async def ingest(
        self,
        user_id: str,
        query: str,
        consensus_response: str,
        embedder: Embedder,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> IngestResult:
        """Persist a Q/A pair so it can shape future prompts.

        Embeds query and response in a single batched call (most embedding
        APIs charge per-token but bill the network round-trip flat — batching
        halves wall-clock latency). The response is truncated to
        ``response_excerpt_chars`` to keep BLOBs bounded; the full text lives
        in whatever upstream logging the caller maintains.
        """
        if not user_id:
            raise ValueError("user_id is required")
        if not query or not consensus_response:
            raise ValueError("query and consensus_response must be non-empty")

        response_excerpt = consensus_response.strip()
        if len(response_excerpt) > self.response_excerpt_chars:
            response_excerpt = (
                response_excerpt[: self.response_excerpt_chars - 1].rstrip() + "…"
            )

        vectors = await embedder.embed([query, response_excerpt])
        if len(vectors) != 2:
            raise RuntimeError(
                f"embedder returned {len(vectors)} vectors, expected 2"
            )
        q_vec, r_vec = vectors
        meta = dict(metadata or {})

        # Add both rows concurrently; SQLite WAL serializes the writes safely.
        q_task = self.store.add(user_id, "query", query, q_vec, {**meta, "role": "query"})
        r_task = self.store.add(
            user_id, "response", response_excerpt, r_vec, {**meta, "role": "response"}
        )
        q_id, r_id = await asyncio.gather(q_task, r_task)

        bytes_embedded = (len(q_vec) + len(r_vec)) * 4
        logger.debug(
            "memory_loop: ingest user=%s q_id=%s r_id=%s (%d B)",
            user_id, q_id, r_id, bytes_embedded,
        )
        return IngestResult(query_id=q_id, response_id=r_id, bytes_embedded=bytes_embedded)

    async def retrieve_context(
        self,
        user_id: str,
        query: str,
        embedder: Embedder,
        max_tokens: int = 200,
    ) -> str:
        """Return prompt-injectable context for ``query``.

        Why delegate to :meth:`VectorMemory.recall_context`: it already
        encodes the formatting + truncation policy we want. This method
        adds the one missing piece — embedding the query — so callers in
        the consensus path can call a single coroutine instead of two.
        """
        if not query.strip():
            return ""
        try:
            vectors = await embedder.embed([query])
        except Exception as exc:  # noqa: BLE001 — best-effort retrieval
            logger.warning("memory_loop: embed failed during retrieve: %s", exc)
            return ""
        if not vectors:
            return ""
        return await self.store.recall_context(
            user_id=user_id,
            query_embedding=vectors[0],
            max_tokens=max_tokens,
        )

    # ---- explicit preference path ---------------------------------------

    async def promote_preference(
        self,
        user_id: str,
        statement: str,
        embedder: Embedder,
        *,
        source: str = "explicit",
    ) -> str:
        """Pin a user-declared preference (e.g. "remember that I prefer X").

        Stored under the ``preference`` kind so the existing ``age_out``
        retention policy keeps it forever. Returns the new memory id so the
        caller can echo "saved as <id>" for the user's confidence.
        """
        statement = statement.strip()
        if not statement:
            raise ValueError("statement must be non-empty")
        vectors = await embedder.embed([statement])
        if not vectors:
            raise RuntimeError("embedder returned no vectors")
        return await self.store.add(
            user_id,
            "preference",
            statement,
            vectors[0],
            {"source": source},
        )

    # ---- weekly batch path ----------------------------------------------

    async def auto_extract_preferences(
        self,
        user_id: str,
        recent_queries: Sequence[str],
        embedder: Embedder,
        *,
        dedupe_threshold: float = 0.85,
    ) -> PreferenceExtraction:
        """Distill long-term preferences from recent query history.

        Runs *once per user per week*. Sends a single Gemini Flash call (≈
        $0.0001 per user) to extract durable statements; falls back to a
        regex heuristic when no API key is present so the loop still works
        in tests and air-gapped CI.

        Each extracted statement is promoted only if no existing
        ``preference`` row scores above ``dedupe_threshold`` cosine
        similarity — this stops the weekly job from polluting the store
        with paraphrases of the same fact.
        """
        if not recent_queries:
            return PreferenceExtraction(
                statements=[], promoted_ids=[], source="heuristic"
            )

        # Trim and de-duplicate input to keep the prompt cheap.
        cleaned = [q.strip() for q in recent_queries if q and q.strip()]
        cleaned = list(dict.fromkeys(cleaned))[:50]

        statements: list[str] = []
        source = "heuristic"
        if self.gemini.available:
            prompt = _EXTRACTOR_PROMPT_TEMPLATE.format(
                cap=self.max_extract_statements,
                queries="\n".join(f"- {q}" for q in cleaned),
            )
            raw = await self.gemini.complete(prompt, max_tokens=400)
            statements = _parse_gemini_json_array(raw)
            if statements:
                source = "gemini"
        if not statements:
            statements = _heuristic_extract(cleaned, cap=self.max_extract_statements)

        if not statements:
            return PreferenceExtraction(
                statements=[], promoted_ids=[], source=source
            )

        # Embed all candidate statements in one batch, then dedupe against
        # the existing preference set before writing.
        cand_vecs = await embedder.embed(statements)
        promoted_ids: list[str] = []
        skipped = 0

        for stmt, vec in zip(statements, cand_vecs):
            existing: list[MemoryHit] = await self.store.search(
                user_id=user_id,
                query_embedding=vec,
                kind="preference",
                top_k=1,
                min_similarity=dedupe_threshold,
            )
            if existing:
                skipped += 1
                logger.debug(
                    "memory_loop: skipped dup preference (sim=%.3f): %s",
                    existing[0].similarity, stmt[:60],
                )
                continue
            new_id = await self.store.add(
                user_id,
                "preference",
                stmt,
                vec,
                {"source": f"auto_extract:{source}"},
            )
            promoted_ids.append(new_id)

        logger.info(
            "memory_loop: auto_extract user=%s source=%s promoted=%d skipped=%d",
            user_id, source, len(promoted_ids), skipped,
        )
        return PreferenceExtraction(
            statements=statements,
            promoted_ids=promoted_ids,
            source=source,
            skipped_duplicates=skipped,
        )


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class _DeterministicEmbedder:
    """Tiny embedder for smoke tests.

    Maps text -> a small fixed-dim vector via a hash. Two identical strings
    embed to identical vectors (so dedupe paths fire); different strings get
    near-orthogonal vectors (so search returns sensible top-k).
    """

    dim = 32

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.strip().lower().encode("utf-8")).digest()
            # Stretch 32 bytes -> 32 floats in [-1, 1]
            vec = [((b / 127.5) - 1.0) for b in h[: self.dim]]
            out.append(vec)
        return out


async def _t_ingest_roundtrip() -> None:
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = VectorMemory(db_path=Path(td) / "m.db")
        evo = MemoryEvolution(store=store, gemini=_GeminiClient(api_key=""))
        emb = _DeterministicEmbedder()
        res = await evo.ingest("u1", "How does HSP work?", "HSP is a protocol that…", emb)
        assert res.query_id and res.response_id
        ctx = await evo.retrieve_context("u1", "How does HSP work?", emb, max_tokens=200)
        assert "[query]" in ctx or "[response]" in ctx, f"expected hit, got: {ctx!r}"


async def _t_explicit_preference_then_recall() -> None:
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = VectorMemory(db_path=Path(td) / "m.db")
        evo = MemoryEvolution(store=store, gemini=_GeminiClient(api_key=""))
        emb = _DeterministicEmbedder()
        pid = await evo.promote_preference("u1", "Sempre responda em pt-br.", emb)
        assert pid
        # Recall using the same statement should surface the preference.
        ctx = await evo.retrieve_context("u1", "Sempre responda em pt-br.", emb)
        assert "preference" in ctx


async def _t_auto_extract_heuristic_offline() -> None:
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = VectorMemory(db_path=Path(td) / "m.db")
        evo = MemoryEvolution(store=store, gemini=_GeminiClient(api_key=""))
        emb = _DeterministicEmbedder()
        recent = [
            "Could you explain quantum tunneling? I prefer terse answers without filler.",
            "Eu prefiro respostas em pt-br, por favor.",
            "Always answer with citations when discussing medical topics.",
            "What time is it in London?",
        ]
        extraction = await evo.auto_extract_preferences("u1", recent, emb)
        assert extraction.source == "heuristic"
        assert len(extraction.promoted_ids) >= 1
        # Re-running with the same input should dedupe everything.
        again = await evo.auto_extract_preferences("u1", recent, emb)
        assert again.skipped_duplicates >= len(again.statements) - 0
        assert len(again.promoted_ids) == 0


async def _t_empty_recent_queries_noop() -> None:
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        store = VectorMemory(db_path=Path(td) / "m.db")
        evo = MemoryEvolution(store=store, gemini=_GeminiClient(api_key=""))
        emb = _DeterministicEmbedder()
        out = await evo.auto_extract_preferences("u1", [], emb)
        assert out.statements == []
        assert out.promoted_ids == []


async def _run_all_tests() -> None:
    await _t_ingest_roundtrip()
    await _t_explicit_preference_then_recall()
    await _t_auto_extract_heuristic_offline()
    await _t_empty_recent_queries_noop()
    logger.info("memory_loop: all self-tests passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(_run_all_tests())
