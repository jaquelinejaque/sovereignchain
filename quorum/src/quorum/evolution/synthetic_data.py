"""Loop 9 — Synthetic Q&A data generation for local-model fine-tuning.

Copyright 2026 Sovereign Chain / Jaqueline Martins.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

HSP-Gated Module — Patent PCT/US26/11908.
Commercial use of the synthetic-data evolution loop requires an HSP license.
See LICENSE-HSP at the repository root for terms.

WHY THIS LOOP EXISTS
====================
The Quorum router prefers the cheapest model that still meets quality. The
cheapest model in our stack is a locally-hosted Llama variant — but Llama is
weaker than the frontier models in certain query classes (long-form code,
multi-step math, niche scientific reasoning, low-resource languages). Loop 9
identifies *which* query classes Llama under-performs in (signal sourced from
the RLHF reward tracker — Loop 7) and pays a premium model exactly once per
domain to manufacture synthetic Q&A pairs that the next distillation pass
(Loop 5) can fine-tune Llama on. Net effect: Llama gets specifically better
where it is specifically weak, the router can lean on it more, and total spend
drops.

WHY THE HSP GATE
================
Synthetic data has a well-documented failure mode: training a model on its own
or another model's outputs collapses the diversity of the distribution
("model collapse" — Shumailov et al., 2024). If we generated unlimited
synthetic samples without oversight, every distillation pass would walk Llama
deeper into the frontier model's specific biases. The HSP gate forces a human
(or HSP-certified policy webhook) to approve each generation run, capping
drift toward synthetic-only training. The gate is annotated risk_level="high"
so the default operator pager rings.

PERSISTENCE
===========
Generated samples are emitted as JSON Lines in the Unsloth conversational
format so they can be fed directly into the Loop 5 LoRA distillation job. No
SQLite — samples are write-once artefacts; the only durable state we keep is
"which domains have we already generated for this user this cycle?" and that
lives in the rlhf_tracker, not here.

TRIGGER
=======
Nightly cron, scheduled *after* Loop 5 (distillation) finishes so the next
distillation pass picks up the new corpus. The orchestrator is expected to:

    1. tracker = RLHFTracker(...)
    2. gen = SyntheticDataGenerator()
    3. weak = gen.identify_weak_domains(tracker, user_id="*")
    4. for domain in weak:
           samples = await gen.generate_for_domain(domain, n_samples=200,
                                                   generator_provider=anthropic)
           gen.export_to_distillation_dataset(samples, out_path)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from quorum.hsp.gate import requires_hsp_approval
from quorum.providers.base import ModelResponse, Provider

if TYPE_CHECKING:
    from quorum.core.consensus import ConsensusResult

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

#: Below this win-rate against the consensus answer, Llama is considered weak.
_WEAK_THRESHOLD: Final[float] = 0.55

#: Minimum number of graded interactions a domain must have before we trust
#: its win-rate signal. Avoids flagging a domain off a single bad query.
_MIN_DOMAIN_SAMPLES: Final[int] = 12

#: Llama's canonical provider name as registered in the RLHF tracker.
#: Override at construction if the local model is registered under a different
#: name (e.g. "ollama:llama3:8b", "local-mistral", etc.).
_DEFAULT_LOCAL_MODEL: Final[str] = "ollama"

#: Default cap on synthetic samples per generate_for_domain call. Bounds spend
#: and limits model-collapse risk per HSP-gated run.
_DEFAULT_N_SAMPLES: Final[int] = 50

#: Hard ceiling no matter what the caller asks for. Even with HSP approval we
#: refuse to generate more than this in one call — operator must split runs.
_MAX_N_SAMPLES: Final[int] = 500

#: Concurrent generator calls. Premium APIs throttle hard; keep this modest.
_MAX_CONCURRENCY: Final[int] = 4

#: Per-sample generation timeout. Premium models can occasionally hang; we
#: drop the sample rather than block the nightly job.
_PER_SAMPLE_TIMEOUT_S: Final[float] = 60.0


# --------------------------------------------------------------------------- #
# Tracker contract — kept as a Protocol so we don't hard-depend on Loop 7's
# concrete class; any object that exposes the two methods below works (and
# tests can pass a hand-rolled stub).
# --------------------------------------------------------------------------- #


@runtime_checkable
class RLHFTrackerLike(Protocol):
    """Subset of the Loop 7 RLHFTracker interface we depend on.

    We intentionally keep this loose so the same module works against an
    in-memory tracker stub during tests and against the real persistent
    tracker in production.
    """

    def per_domain_winrate(
        self, *, user_id: str, model_name: str
    ) -> dict[str, tuple[float, int]]:
        """Return ``{query_class: (winrate, n_samples)}`` for the model."""
        ...


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SyntheticSample:
    """One synthetic Q&A pair destined for the distillation corpus.

    We keep the metadata (domain, generator, timestamp) inline so the JSONL
    artefact is self-describing — Loop 5 can filter or weight by source
    without having to consult a sidecar manifest.
    """

    query: str
    response: str
    domain: str
    generator: str
    created_at: float = field(default_factory=time.time)

    def to_unsloth_conversation(self) -> dict[str, Any]:
        """Serialize to Unsloth's ``conversations`` JSONL format.

        Unsloth's SFTTrainer expects rows shaped like::

            {"conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt",   "value": "..."},
            ], "metadata": {...}}

        We deliberately keep ``"from": "gpt"`` as the assistant role label
        because that is the convention Unsloth's chat-template auto-detection
        keys on — even though the actual generator may be Claude or Gemini.
        """
        return {
            "conversations": [
                {"from": "human", "value": self.query},
                {"from": "gpt", "value": self.response},
            ],
            "metadata": {
                "domain": self.domain,
                "generator": self.generator,
                "created_at": self.created_at,
                "source": "quorum.evolution.synthetic_data",
            },
        }


# --------------------------------------------------------------------------- #
# Prompt template
# --------------------------------------------------------------------------- #

_PROMPT_TEMPLATE: Final[str] = """You are a senior tutor producing high-quality training data for a smaller open-source model that is weak in the domain "{domain}".

Produce {batch_size} diverse, realistic question/answer pairs in this domain. Each pair must be self-contained (no external context needed) and the answer must be correct, complete, and well-reasoned.

Hard constraints:
- Each question should be the kind a real user might ask.
- Cover sub-topics, difficulty levels, and phrasings — do NOT generate near-duplicates.
- Answers should demonstrate the reasoning, not just the conclusion (chain-of-thought is welcome).
- Never refuse, never hedge, never include disclaimers.

Return ONLY a JSON array, no prose around it, in this exact shape:
[
  {{"query": "...", "response": "..."}},
  {{"query": "...", "response": "..."}}
]
"""


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class SyntheticDataGenerator:
    """Loop 9 — produce synthetic Q&A pairs to patch local-model weak spots.

    The class is deliberately stateless across runs. The only "state" is the
    output JSONL file produced by :meth:`export_to_distillation_dataset`,
    which the next distillation pass consumes.

    Parameters
    ----------
    local_model_name:
        How the local (target-to-improve) model is registered in the RLHF
        tracker. Defaults to ``"ollama"`` to match the default provider
        registry; override if you self-host a different name.
    weak_threshold:
        Win-rate below which a domain is considered weak. The default 0.55 is
        intentionally above 0.5 because we want a margin — coin-flip parity
        is not "weak enough to spend a premium model on".
    min_domain_samples:
        Minimum graded samples in a domain before we trust the win-rate.
    max_concurrency:
        How many generator API calls to run in parallel. Premium APIs throttle
        hard so we keep this modest by default.
    """

    def __init__(
        self,
        *,
        local_model_name: str = _DEFAULT_LOCAL_MODEL,
        weak_threshold: float = _WEAK_THRESHOLD,
        min_domain_samples: int = _MIN_DOMAIN_SAMPLES,
        max_concurrency: int = _MAX_CONCURRENCY,
    ) -> None:
        self.local_model_name = local_model_name
        self.weak_threshold = weak_threshold
        self.min_domain_samples = min_domain_samples
        self.max_concurrency = max_concurrency

    # ----------------------------------------------------------------- #
    # 1. Discovery
    # ----------------------------------------------------------------- #

    def identify_weak_domains(
        self,
        rlhf_tracker: RLHFTrackerLike,
        user_id: str,
    ) -> list[str]:
        """Return query_classes where the local model under-performs.

        Why a synchronous method on an async-first module?
            The tracker call is a single bookkeeping read; making it ``async``
            adds nothing useful and forces every caller into a coroutine
            context. Loop 7's concrete tracker also exposes this synchronously.

        Selection rule:
            domain is weak  iff  winrate < weak_threshold
                            AND  n_samples >= min_domain_samples

        Ordered by ascending winrate so the orchestrator can budget premium
        spend on the worst gaps first.
        """
        try:
            stats = rlhf_tracker.per_domain_winrate(
                user_id=user_id, model_name=self.local_model_name
            )
        except Exception as e:  # noqa: BLE001
            # Tracker outage must not crash the nightly loop — just no-op.
            logger.warning(
                "rlhf_tracker.per_domain_winrate failed user=%s model=%s err=%s",
                user_id,
                self.local_model_name,
                e,
            )
            return []

        weak: list[tuple[str, float]] = []
        for domain, (winrate, n) in stats.items():
            if n < self.min_domain_samples:
                continue
            if winrate >= self.weak_threshold:
                continue
            weak.append((domain, winrate))

        weak.sort(key=lambda kv: kv[1])
        logger.info(
            "identify_weak_domains user=%s model=%s -> %d weak domains",
            user_id,
            self.local_model_name,
            len(weak),
        )
        return [d for d, _ in weak]

    # ----------------------------------------------------------------- #
    # 2. Generation (HSP-gated)
    # ----------------------------------------------------------------- #

    @requires_hsp_approval(
        action="generate_synthetic_training_data",
        risk_level="high",
    )
    async def generate_for_domain(
        self,
        domain: str,
        n_samples: int,
        generator_provider: Provider,
    ) -> list[dict[str, Any]]:
        """Produce ``n_samples`` synthetic {query, response} pairs for ``domain``.

        Why is this the HSP-gated entrypoint?
            Spend, drift, and model-collapse risk all enter the system here.
            Loop 5 (distillation) will consume whatever we emit; the only
            place to intercept that risk is at generation time.

        Implementation:
            We ask the premium model to emit batches as a JSON array, then
            validate each item, drop malformed ones, and keep going until we
            either reach ``n_samples`` or exhaust a small retry budget. We
            run several batches in parallel bounded by ``max_concurrency``.

        Returns:
            A list of plain dicts ``{"query": ..., "response": ...,
            "domain": ..., "generator": ..., "created_at": ...}``. We return
            dicts (not :class:`SyntheticSample`) so the result is trivially
            JSON-serializable — useful for in-memory tests and easy logging.
        """
        if n_samples <= 0:
            return []
        if n_samples > _MAX_N_SAMPLES:
            logger.warning(
                "generate_for_domain capped n_samples %d -> %d (hard ceiling)",
                n_samples,
                _MAX_N_SAMPLES,
            )
            n_samples = _MAX_N_SAMPLES

        # Batch sizing: ask for ~10 per call so a single bad JSON parse only
        # costs us ~10 samples rather than the whole run.
        batch_size = 10
        n_batches = (n_samples + batch_size - 1) // batch_size

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one_batch(batch_idx: int) -> list[SyntheticSample]:
            async with sem:
                return await self._generate_batch(
                    domain=domain,
                    batch_size=batch_size,
                    generator_provider=generator_provider,
                    batch_idx=batch_idx,
                )

        tasks = [_one_batch(i) for i in range(n_batches)]
        batches = await asyncio.gather(*tasks, return_exceptions=True)

        out: list[SyntheticSample] = []
        for b in batches:
            if isinstance(b, BaseException):
                logger.warning("synthetic batch failed: %s", b)
                continue
            out.extend(b)
            if len(out) >= n_samples:
                break

        out = out[:n_samples]
        logger.info(
            "generate_for_domain domain=%s requested=%d produced=%d generator=%s",
            domain,
            n_samples,
            len(out),
            getattr(generator_provider, "name", "<unknown>"),
        )
        return [
            {
                "query": s.query,
                "response": s.response,
                "domain": s.domain,
                "generator": s.generator,
                "created_at": s.created_at,
            }
            for s in out
        ]

    async def _generate_batch(
        self,
        *,
        domain: str,
        batch_size: int,
        generator_provider: Provider,
        batch_idx: int,
    ) -> list[SyntheticSample]:
        """Run one provider call and parse its JSON array response.

        Kept private because callers should always go through
        :meth:`generate_for_domain` (which carries the HSP gate). Wrapped in
        :func:`asyncio.wait_for` so a hung premium-model call cannot block
        the nightly job past ``_PER_SAMPLE_TIMEOUT_S``.
        """
        prompt = _PROMPT_TEMPLATE.format(domain=domain, batch_size=batch_size)
        try:
            resp: ModelResponse = await asyncio.wait_for(
                generator_provider.complete(prompt, max_tokens=4000),
                timeout=_PER_SAMPLE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "synthetic batch %d for domain=%s timed out", batch_idx, domain
            )
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "synthetic batch %d for domain=%s provider error: %s",
                batch_idx,
                domain,
                e,
            )
            return []

        if resp.error:
            logger.warning(
                "synthetic batch %d for domain=%s provider returned error=%s",
                batch_idx,
                domain,
                resp.error,
            )
            return []

        pairs = _parse_pairs(resp.response)
        gen_name = getattr(generator_provider, "name", resp.name or "unknown")
        return [
            SyntheticSample(
                query=q,
                response=a,
                domain=domain,
                generator=gen_name,
            )
            for q, a in pairs
        ]

    # ----------------------------------------------------------------- #
    # 3. Export
    # ----------------------------------------------------------------- #

    def export_to_distillation_dataset(
        self,
        samples: list[dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Write samples to JSONL in Unsloth conversational format.

        Why append-mode is *not* the default:
            Each nightly run produces a versioned file so distillation can
            reproduce exactly which corpus was used. The orchestrator should
            pass a timestamped path like
            ``~/.quorum/synthetic/2026-06-16-math.jsonl``.

        Why we still wrap the write in to_thread:
            On the nightly job this is small (~50–500 lines) so it barely
            matters, but the module advertises "async-first" — keeping the
            event loop unblocked is cheap and consistent.

        Returns:
            The :class:`Path` actually written, for the caller's logs.
        """
        path = Path(output_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for s in samples:
            sample = SyntheticSample(
                query=s["query"],
                response=s["response"],
                domain=s.get("domain", "unknown"),
                generator=s.get("generator", "unknown"),
                created_at=float(s.get("created_at", time.time())),
            )
            lines.append(json.dumps(sample.to_unsloth_conversation(), ensure_ascii=False))

        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.info(
            "export_to_distillation_dataset wrote %d samples to %s",
            len(samples),
            path,
        )
        return path


# --------------------------------------------------------------------------- #
# JSON-array extraction helper
# --------------------------------------------------------------------------- #


_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE
)


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Best-effort extraction of ``[{query, response}, ...]`` from model text.

    Premium models tend to comply with "ONLY a JSON array" but occasionally
    wrap the array in a ``code fence`` or prepend a "Here you go:" preamble.
    This helper:

      1. strips one code fence if present,
      2. locates the outermost ``[ ... ]`` substring,
      3. parses it,
      4. drops any item that isn't a dict with non-empty query+response.

    We deliberately never raise — a malformed batch simply yields zero
    samples and the run moves on. Loop 9 is a best-effort top-up, not a
    hard dependency.
    """
    if not raw:
        return []

    text = raw.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    blob = text[start : end + 1]

    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    out: list[tuple[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("query") or item.get("question") or item.get("prompt")
        a = item.get("response") or item.get("answer") or item.get("completion")
        if not isinstance(q, str) or not isinstance(a, str):
            continue
        q, a = q.strip(), a.strip()
        if not q or not a:
            continue
        out.append((q, a))
    return out


# --------------------------------------------------------------------------- #
# Test helpers / smoke tests
# --------------------------------------------------------------------------- #


class _StubTracker:
    """In-memory tracker used by smoke tests so we don't depend on Loop 7."""

    def __init__(self, data: dict[str, tuple[float, int]]) -> None:
        self._data = data

    def per_domain_winrate(
        self, *, user_id: str, model_name: str
    ) -> dict[str, tuple[float, int]]:
        del user_id, model_name
        return dict(self._data)


class _StubProvider(Provider):
    """In-memory provider that returns a canned JSON batch.

    Lets us exercise ``generate_for_domain`` without an API key. The HSP gate
    is dev-mode (no HSP_GATE_WEBHOOK env var) so the call passes straight
    through, which is exactly the path tests run under.
    """

    name = "stub"

    def __init__(self, batch_size: int = 10) -> None:
        self._batch_size = batch_size

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> ModelResponse:
        del prompt, max_tokens
        items = [
            {
                "query": f"What is the integral of x^{i}?",
                "response": f"x^{i + 1}/{i + 1} + C",
            }
            for i in range(1, self._batch_size + 1)
        ]
        return ModelResponse(
            name=self.name,
            response=json.dumps(items),
            latency_ms=1.0,
            cost_usd=0.0,
            tokens_in=10,
            tokens_out=200,
        )


async def _smoke_identify_and_generate() -> None:
    """End-to-end smoke: weak domain -> synthetic batch -> JSONL export.

    Exposed at module level so the project's test runner can wire it up
    cheaply. Asserts are lightweight on purpose — full provider integration
    tests live in tests/test_synthetic_data.py.
    """
    tracker = _StubTracker(
        {
            "math.integration": (0.30, 50),  # weak, plenty of samples -> flagged
            "code.python": (0.80, 50),       # strong -> ignored
            "math.exotic": (0.10, 3),        # too few samples -> ignored
        }
    )
    gen = SyntheticDataGenerator()

    weak = gen.identify_weak_domains(tracker, user_id="u1")
    assert weak == ["math.integration"], weak

    samples = await gen.generate_for_domain(
        domain="math.integration",
        n_samples=5,
        generator_provider=_StubProvider(batch_size=10),
    )
    assert len(samples) == 5, len(samples)
    assert all(s["domain"] == "math.integration" for s in samples)
    assert all(s["query"] and s["response"] for s in samples)

    out_path = Path(os.environ.get("QUORUM_SYNTH_TMP", "/tmp")) / "quorum_synth_smoke.jsonl"
    written = gen.export_to_distillation_dataset(samples, out_path)
    assert written.exists()
    lines = written.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5, len(lines)
    parsed = json.loads(lines[0])
    assert parsed["conversations"][0]["from"] == "human"
    assert parsed["conversations"][1]["from"] == "gpt"
    assert parsed["metadata"]["domain"] == "math.integration"


def _smoke_parse_pairs() -> None:
    """Make sure the JSON-array extractor handles real-world model quirks."""
    fenced = """Here you go:
```json
[
  {"query": "Q1?", "response": "A1."},
  {"query": "Q2?", "answer": "A2."}
]
```
Hope that helps!"""
    pairs = _parse_pairs(fenced)
    assert pairs == [("Q1?", "A1."), ("Q2?", "A2.")], pairs

    assert _parse_pairs("") == []
    assert _parse_pairs("no json here") == []
    assert _parse_pairs('[{"query": "Q", "response": ""}]') == []
    assert _parse_pairs('{"not": "a list"}') == []


# --------------------------------------------------------------------------- #
# SyntheticDatasetStore — consensus-derived training corpus
# --------------------------------------------------------------------------- #
#
# WHY THIS LIVES NEXT TO THE GENERATOR
# ====================================
# Loop 9 has two complementary ingestion paths:
#
#   1. SyntheticDataGenerator  — actively pays a premium model to manufacture
#      Q&A pairs in domains where Llama is weak (cold-start, HSP-gated).
#   2. SyntheticDatasetStore   — passively skims high-confidence consensus
#      results during normal traffic and persists them as training examples
#      (warm-loop, opt-in per user).
#
# Both feed the same Loop 5 distillation pass, but path (2) is essentially
# free: the consensus already ran for a paying customer. Keeping them in one
# module makes it obvious they share the same downstream consumer.
#
# PRIVACY / OPT-IN
# ================
# Customer prompts are sensitive by default. The store NEVER persists unless
# the caller passes opt_in=True. Even then, the JSONL row carries an opaque
# user_id, and export_jsonl(anonymize=True) strips it before sharing.
#
# DEDUP
# =====
# Real traffic has heavy near-duplicate prompts ("write hello world",
# "what is 2+2", system-prompt boilerplate). We hash-dedup on a sliding
# in-memory window so we don't bloat the corpus or over-weight common
# prompts during distillation. The window is intentionally bounded — full
# dedup across the entire file would require either loading every line on
# startup (slow) or a sidecar index (complexity Loop 5 doesn't need).

#: Path under QUORUM_DATA_DIR where the JSONL corpus lives.
_SYNTH_DATASET_FILENAME: Final[str] = "synthetic_dataset.jsonl"

#: How many recent prompt hashes to remember for dedup. ~1000 covers a
#: typical interactive session; beyond that the marginal cost of writing a
#: rare duplicate is dominated by the cost of keeping a huge set in memory.
_DEDUP_WINDOW: Final[int] = 1000

#: Default minimum confidence to ingest. 0.85 picks the strongest answers
#: (semantic agreement well above coin flip) without being so strict that
#: only trivial prompts qualify.
_DEFAULT_MIN_CONFIDENCE: Final[float] = 0.85


def _data_dir() -> Path:
    """Resolve the JSONL corpus directory honouring ``QUORUM_DATA_DIR``.

    WHY a function rather than a module-level constant: tests want to point
    a fresh env var at a tmpdir per case, and a module constant would
    snapshot the value at import time.
    """
    raw = os.environ.get("QUORUM_DATA_DIR") or str(Path.home() / ".quorum")
    return Path(raw).expanduser()


def _prompt_hash(prompt: str) -> str:
    """SHA-256 of the prompt, truncated to 16 hex chars (~64 bits).

    WHY only 16 chars: dedup is a best-effort cache scan, not a security
    boundary. 64 bits gives ~1 collision per 4 billion prompts — fine for a
    sliding-window dedup, and keeps the JSONL row small.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


class SyntheticDatasetStore:
    """Append-only JSONL store of high-confidence consensus answers.

    The store is process-local: each instance keeps its own dedup window and
    its own write lock. Multiple instances pointed at the same file path
    will still write safely (each append is one ``write`` syscall ending in
    ``\\n``), but the dedup window is per-instance, so two instances may
    each write the same prompt once. That's acceptable for a best-effort
    training corpus.

    Parameters
    ----------
    path:
        Override the default ``${QUORUM_DATA_DIR}/synthetic_dataset.jsonl``.
        Mostly used by tests.
    dedup_window:
        How many recent ``prompt_hash`` values to remember.
    """

    def __init__(
        self,
        *,
        path: Path | str | None = None,
        dedup_window: int = _DEDUP_WINDOW,
    ) -> None:
        if path is None:
            self._path = _data_dir() / _SYNTH_DATASET_FILENAME
        else:
            self._path = Path(path).expanduser()
        self._dedup_window = max(1, int(dedup_window))
        # Sliding-window set of recent prompt hashes; OrderedDict gives us
        # O(1) "drop oldest" without an external library.
        self._recent_hashes: OrderedDict[str, None] = OrderedDict()
        self._lock = asyncio.Lock()

    # ----------------------------------------------------------------- #
    # Ingest
    # ----------------------------------------------------------------- #

    async def maybe_ingest(
        self,
        prompt: str,
        result: "ConsensusResult",
        *,
        user_id: str | None = None,
        opt_in: bool = False,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> bool:
        """Append (prompt, winning answer) to the corpus if all gates pass.

        Returns ``True`` if the row was actually written. Returns ``False``
        (silently — this is a side path, never user-visible) when:

        * ``opt_in`` is False (default-deny for privacy),
        * the consensus confidence is below ``min_confidence``,
        * the prompt exceeds ``MAX_PROMPT_BYTES`` (re-used from consensus.py
          so we never persist anything the engine itself would have rejected),
        * the prompt has already been ingested within the dedup window,
        * the answer is empty.
        """
        if not opt_in:
            return False

        # Import lazily so this module stays importable without consensus.
        from quorum.core.consensus import MAX_PROMPT_BYTES

        if not prompt or len(prompt) > MAX_PROMPT_BYTES:
            return False

        answer = (result.answer or "").strip()
        if not answer:
            return False

        confidence = float(result.confidence or 0.0)
        if confidence < float(min_confidence):
            return False

        phash = _prompt_hash(prompt)

        async with self._lock:
            if phash in self._recent_hashes:
                # Refresh recency so a repeatedly-asked prompt stays in the
                # window rather than expiring and getting re-ingested.
                self._recent_hashes.move_to_end(phash)
                return False

            row = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "prompt_hash": phash,
                "prompt": prompt,
                "answer": answer,
                "confidence": round(confidence, 6),
                "models": [
                    {
                        "name": m.name,
                        "weight": round(float(getattr(m, "weight", 0.0) or 0.0), 6),
                    }
                    for m in (result.models or [])
                    if not getattr(m, "error", "")
                ],
                "user_id": user_id,
            }

            await asyncio.to_thread(self._append_line, row)

            self._recent_hashes[phash] = None
            if len(self._recent_hashes) > self._dedup_window:
                self._recent_hashes.popitem(last=False)

        return True

    def _append_line(self, row: dict[str, Any]) -> None:
        """Synchronous append of one JSONL row. Called via ``to_thread``.

        WHY a separate sync method: ``open(..., "a")`` + ``write`` is a
        blocking syscall; running it on the event-loop thread would stall
        every other consensus call sharing the loop.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ----------------------------------------------------------------- #
    # Stats
    # ----------------------------------------------------------------- #

    async def stats(self) -> dict[str, Any]:
        """Return aggregate counts over the persisted corpus.

        The shape is intentionally minimal — total count, per-user counts
        (anonymous: keys are the user_id strings as-stored, no extra
        joining), and the timestamp range. No prompts or answers are
        returned, so this is safe to expose over an authenticated HTTP
        endpoint without further PII review.
        """
        return await asyncio.to_thread(self._stats_sync)

    def _stats_sync(self) -> dict[str, Any]:
        total = 0
        by_user: dict[str, int] = {}
        ts_min: str | None = None
        ts_max: str | None = None

        if not self._path.exists():
            return {
                "total_examples": 0,
                "by_user": {},
                "date_range": {"min": None, "max": None},
            }

        with self._path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    # Corrupt line — skip rather than crash the stats call.
                    continue
                total += 1
                uid = row.get("user_id")
                key = uid if uid is not None else "_anon_"
                by_user[key] = by_user.get(key, 0) + 1
                ts = row.get("ts")
                if isinstance(ts, str):
                    if ts_min is None or ts < ts_min:
                        ts_min = ts
                    if ts_max is None or ts > ts_max:
                        ts_max = ts

        return {
            "total_examples": total,
            "by_user": by_user,
            "date_range": {"min": ts_min, "max": ts_max},
        }

    # ----------------------------------------------------------------- #
    # Export
    # ----------------------------------------------------------------- #

    async def export_jsonl(
        self,
        out_path: Path | str,
        *,
        anonymize: bool = True,
    ) -> int:
        """Copy the corpus to ``out_path``, optionally stripping ``user_id``.

        Returns the number of rows written. The output is fresh JSONL
        (one valid JSON object per line), even if the source file has
        legacy corrupt rows — those are silently dropped, same policy as
        :meth:`stats`.
        """
        out = Path(out_path).expanduser()
        return await asyncio.to_thread(self._export_sync, out, anonymize)

    def _export_sync(self, out_path: Path, anonymize: bool) -> int:
        if not self._path.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text("", encoding="utf-8")
            return 0

        out_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with (
            self._path.open("r", encoding="utf-8") as src,
            out_path.open("w", encoding="utf-8") as dst,
        ):
            for raw in src:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if anonymize:
                    row.pop("user_id", None)
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        return written


__all__ = [
    "SyntheticDataGenerator",
    "SyntheticDatasetStore",
    "SyntheticSample",
    "RLHFTrackerLike",
]


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _smoke_parse_pairs()
    asyncio.run(_smoke_identify_and_generate())
    logger.info("synthetic_data smoke tests passed")
