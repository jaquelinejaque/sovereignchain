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
# HSP attribution:
#   This loop is NOT HSP-gated. Prompt templates evolved here are stored
#   per-model and are downstream artifacts; the gated bits (consensus
#   weighting, HSP-specific synthesis) live in other modules.
"""Loop 11 — Self-prompting evolution.

Why this module exists
----------------------
Every model in a Quorum ensemble responds better to a slightly different
system-prompt phrasing — Claude likes structured XML-ish framings, Gemini
likes bulleted instructions, GPT-class models prefer concise role primers,
local Llama variants do better with explicit "step by step" cues. Hand-tuning
each one is fragile. This loop *learns* the best system prompt per model by
generating variants, A/B-testing them via RLHF reward signals, and rolling a
Bayesian average per variant.

Cadence
~~~~~~~
* **Per call** — :meth:`SelfPromptOptimizer.get_current_prompt` returns the
  champion template (highest posterior mean, ties broken by sample count).
* **On reward** — :meth:`SelfPromptOptimizer.record_outcome` updates the
  posterior mean of a variant using an online Bayesian rolling-mean update
  (no need to keep individual samples).
* **Weekly** — :meth:`SelfPromptOptimizer.weekly_evolve` proposes one new
  candidate via a "premium" generator provider, then keeps the top-3
  variants per model and retires the rest. Top-3 leaves enough room for the
  bandit to keep exploring.

Storage
~~~~~~~
SQLite at ``~/.quorum/prompts.db`` (override with ``QUORUM_PROMPTS_DB``).
All blocking I/O is wrapped in :func:`asyncio.to_thread` so the loop never
blocks the event loop running consensus calls. The schema is intentionally
narrow — one row per (model, template) — which keeps weekly evolution a
single ``DELETE … WHERE id NOT IN (top_3)``.

Mutation strategies
~~~~~~~~~~~~~~~~~~~
``paraphrase`` keeps semantics, varies wording. ``reorder`` shuffles
existing instruction blocks (good when prompts have multiple constraints).
``add_instruction`` appends a new constraint sampled from a small library
of useful steering nudges. Each strategy maps to a different generator
prompt sent to the premium provider; if no provider is supplied we fall
back to a deterministic offline mutator so tests work without API keys.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from quorum.hsp.gate import requires_hsp_approval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


MutationStrategy = Literal["paraphrase", "reorder", "add_instruction"]
VariantStatus = Literal["active", "retired"]


@runtime_checkable
class GeneratorProvider(Protocol):
    """Minimal contract for the premium provider used to propose variants.

    Why a Protocol instead of importing :class:`Provider`: this module must
    work in offline tests where the caller passes ``None`` or a stub. We
    only need ``complete(prompt, *, max_tokens) -> object_with_.response``.
    Any :class:`quorum.providers.base.Provider` satisfies it for free, but
    we never type-couple to it.
    """

    name: str

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> Any:
        """Return any object exposing a ``.response: str`` attribute."""
        ...


@dataclass(frozen=True)
class PromptVariant:
    """One row of the ``prompt_variants`` table.

    Why frozen: variants are read-only snapshots — updates go through
    SQL, not through Python mutation. Freezing prevents accidental drift
    between the in-memory copy and the persisted row.
    """

    id: str
    model_name: str
    prompt_template: str
    avg_reward: float
    samples: int
    status: VariantStatus
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "prompt_template": self.prompt_template,
            "avg_reward": round(self.avg_reward, 6),
            "samples": self.samples,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class EvolutionReport:
    """Bookkeeping returned by :meth:`SelfPromptOptimizer.weekly_evolve`."""

    model_name: str
    proposed_id: Optional[str]
    proposal_strategy: Optional[MutationStrategy]
    kept_ids: list[str]
    retired_ids: list[str]
    source: str  # "generator" | "offline"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


_DEFAULT_TEMPLATE = (
    "You are a careful, concise assistant. "
    "Answer directly. Cite uncertainty when relevant. "
    "Prefer precision over fluff."
)


_OFFLINE_ADDONS: tuple[str, ...] = (
    "Be explicit about your level of confidence.",
    "If you do not know, say so plainly instead of guessing.",
    "Quote sources verbatim when they materially affect the answer.",
    "Prefer worked examples over abstract claims.",
    "Reject hidden assumptions in the user's question before answering.",
)


DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()


def _default_db_path() -> Path:
    """Return the on-disk location of ``prompts.db``.

    Honors ``QUORUM_PROMPTS_DB`` so tests can redirect to a tmpdir without
    monkeypatching. Falls back to ``~/.quorum/prompts.db``.
    """
    override = os.getenv("QUORUM_PROMPTS_DB")
    if override:
        return Path(override).expanduser()
    return DATA_DIR / "prompts.db"


# ---------------------------------------------------------------------------
# Generator prompts (one per mutation strategy)
# ---------------------------------------------------------------------------


_PARAPHRASE_PROMPT = """\
Rewrite the following SYSTEM PROMPT so that its meaning is preserved but
the wording is different. Keep it concise (under 80 words). Return ONLY
the rewritten prompt, no preamble, no quotation marks, no markdown fences.

SYSTEM PROMPT:
{template}
"""

_REORDER_PROMPT = """\
Reorder the instructions in the following SYSTEM PROMPT to put the most
load-bearing constraint first, while preserving every original instruction.
Keep it under 80 words. Return ONLY the reordered prompt, no preamble.

SYSTEM PROMPT:
{template}
"""

_ADD_INSTRUCTION_PROMPT = """\
Extend the following SYSTEM PROMPT by adding ONE new instruction that
materially improves answer quality (e.g. an explicit confidence cue, a
formatting rule, a sourcing requirement). Do not remove any existing
instruction. Keep total length under 100 words. Return ONLY the extended
prompt, no preamble.

SYSTEM PROMPT:
{template}
"""


_STRATEGY_PROMPTS: dict[MutationStrategy, str] = {
    "paraphrase": _PARAPHRASE_PROMPT,
    "reorder": _REORDER_PROMPT,
    "add_instruction": _ADD_INSTRUCTION_PROMPT,
}


# ---------------------------------------------------------------------------
# SelfPromptOptimizer
# ---------------------------------------------------------------------------


@dataclass
class SelfPromptOptimizer:
    """Backs prompt evolution with a SQLite-resident bandit.

    Why a dataclass rather than a class with ``__init__``: every field has
    a sensible default, and dataclasses give us a free ``repr`` for
    debugging the wired-up loop. The ``__post_init__`` hook handles DB
    bootstrap so callers can construct the optimizer at import time.
    """

    db_path: Path = field(default_factory=_default_db_path)
    seed_template: str = _DEFAULT_TEMPLATE
    keep_top_k: int = 3
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(self.rng_seed)
        self._init_schema()

    # ---- schema ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a short-lived SQLite connection with WAL enabled.

        Why a fresh connection per call: SQLite connections are not safe
        to share across threads, and we routinely run from
        :func:`asyncio.to_thread`. WAL gives us concurrent readers + one
        writer, which matches the optimizer's access pattern (frequent
        reads on ``get_current_prompt``, occasional writes on rewards).
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS prompt_variants (
                    id              TEXT PRIMARY KEY,
                    model_name      TEXT NOT NULL,
                    prompt_template TEXT NOT NULL,
                    avg_reward      REAL NOT NULL DEFAULT 0.0,
                    samples         INTEGER NOT NULL DEFAULT 0,
                    status          TEXT NOT NULL DEFAULT 'active',
                    created_at      REAL NOT NULL,
                    updated_at      REAL NOT NULL,
                    UNIQUE(model_name, prompt_template)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_variants_model_status "
                "ON prompt_variants(model_name, status)"
            )
            conn.commit()

    # ---- public API ------------------------------------------------------

    async def get_current_prompt(self, model_name: str) -> str:
        """Return the best-performing active template for ``model_name``.

        Selection rule: highest ``avg_reward`` among ``status='active'``
        rows; ties broken by ``samples`` (more samples = higher posterior
        confidence). If the model has no rows yet, seed one with the
        default template so the caller always gets something usable.
        """
        if not model_name:
            raise ValueError("model_name is required")
        return await asyncio.to_thread(self._get_current_prompt_sync, model_name)

    def _get_current_prompt_sync(self, model_name: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT prompt_template
                FROM prompt_variants
                WHERE model_name = ? AND status = 'active'
                ORDER BY avg_reward DESC, samples DESC, created_at ASC
                LIMIT 1
                """,
                (model_name,),
            ).fetchone()
            if row is not None:
                return str(row["prompt_template"])
            # Seed on first read so subsequent reads stay deterministic.
            self._insert_variant_sync(
                conn, model_name=model_name, template=self.seed_template
            )
            conn.commit()
            return self.seed_template

    async def propose_variant(
        self,
        model_name: str,
        mutation_strategy: MutationStrategy = "paraphrase",
        generator_provider: Optional[GeneratorProvider] = None,
        *,
        base_template: Optional[str] = None,
    ) -> str:
        """Generate and persist a new candidate variant.

        The premium provider (``generator_provider``) is asked to mutate
        the current champion (or ``base_template`` if given) according to
        ``mutation_strategy``. If no provider is supplied, or the call
        fails, we fall back to an offline mutator so the loop never gets
        stuck waiting on an external API. Returns the new variant id.
        """
        if mutation_strategy not in _STRATEGY_PROMPTS:
            raise ValueError(
                f"unknown mutation_strategy: {mutation_strategy!r}; "
                f"expected one of {sorted(_STRATEGY_PROMPTS)}"
            )

        base = base_template or await self.get_current_prompt(model_name)
        candidate = await self._mutate(base, mutation_strategy, generator_provider)
        candidate = _clean_template(candidate) or _offline_mutate(
            base, mutation_strategy, self._rng
        )

        variant_id = await asyncio.to_thread(
            self._insert_variant_safe_sync, model_name, candidate
        )
        logger.info(
            "self_prompt: proposed variant model=%s strategy=%s id=%s",
            model_name, mutation_strategy, variant_id,
        )
        return variant_id

    async def record_outcome(
        self,
        variant_id: str,
        reward: float,
        sample_count: int = 1,
    ) -> PromptVariant:
        """Update the Bayesian rolling mean for a variant.

        We treat ``avg_reward`` as the posterior mean of a Beta-like
        process with ``samples`` as the pseudo-count. The update is the
        standard incremental mean::

            new_mean = old_mean + (reward - old_mean) * w / (samples + w)

        where ``w = sample_count``. This is numerically stable (no
        catastrophic cancellation for small reward deltas) and lets the
        caller batch multiple observations into one DB write.
        """
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        return await asyncio.to_thread(
            self._record_outcome_sync, variant_id, float(reward), int(sample_count)
        )

    def _record_outcome_sync(
        self, variant_id: str, reward: float, sample_count: int
    ) -> PromptVariant:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM prompt_variants WHERE id = ?", (variant_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown variant_id: {variant_id}")
            old_mean = float(row["avg_reward"])
            old_n = int(row["samples"])
            new_n = old_n + sample_count
            # Incremental Bayesian-flavored mean update; equivalent to a
            # weighted average when sample_count > 1.
            new_mean = old_mean + (reward - old_mean) * (sample_count / new_n)
            now = time.time()
            conn.execute(
                """
                UPDATE prompt_variants
                SET avg_reward = ?, samples = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_mean, new_n, now, variant_id),
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM prompt_variants WHERE id = ?", (variant_id,)
            ).fetchone()
            return _row_to_variant(updated)

    @requires_hsp_approval(action="self_prompt_weekly_evolve", risk_level="high")
    async def weekly_evolve(
        self,
        model_name: str,
        generator_provider: Optional[GeneratorProvider] = None,
        *,
        mutation_strategy: MutationStrategy = "paraphrase",
        min_samples_to_retire: int = 5,
    ) -> EvolutionReport:
        """Propose a new variant, retire losers, keep top-``keep_top_k``.

        Why ``min_samples_to_retire``: a freshly-proposed variant has zero
        samples and would otherwise be retired immediately by an
        avg-reward sort. The threshold gives new variants a grace period
        in which they are kept regardless of rank, so the bandit can
        actually explore.
        """
        if not model_name:
            raise ValueError("model_name is required")
        source = "offline"
        proposed_id: Optional[str] = None
        try:
            proposed_id = await self.propose_variant(
                model_name,
                mutation_strategy=mutation_strategy,
                generator_provider=generator_provider,
            )
            source = "generator" if generator_provider is not None else "offline"
        except Exception as exc:  # noqa: BLE001 — best-effort proposal
            logger.warning(
                "self_prompt: propose_variant failed in weekly_evolve: %s", exc
            )

        kept, retired = await asyncio.to_thread(
            self._retire_losers_sync, model_name, min_samples_to_retire
        )
        report = EvolutionReport(
            model_name=model_name,
            proposed_id=proposed_id,
            proposal_strategy=mutation_strategy if proposed_id else None,
            kept_ids=kept,
            retired_ids=retired,
            source=source,
        )
        logger.info(
            "self_prompt: weekly_evolve model=%s kept=%d retired=%d source=%s",
            model_name, len(kept), len(retired), source,
        )
        return report

    # ---- introspection helpers (sync wrappers) --------------------------

    async def list_variants(
        self, model_name: str, *, include_retired: bool = False
    ) -> list[PromptVariant]:
        """Return all variants for a model, newest-best first.

        Useful for dashboards and the weekly job's report. Sorted by
        ``avg_reward DESC, samples DESC`` so the champion is always
        ``[0]``.
        """
        return await asyncio.to_thread(
            self._list_variants_sync, model_name, include_retired
        )

    def _list_variants_sync(
        self, model_name: str, include_retired: bool
    ) -> list[PromptVariant]:
        clause = "" if include_retired else " AND status = 'active'"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM prompt_variants
                WHERE model_name = ?{clause}
                ORDER BY avg_reward DESC, samples DESC, created_at ASC
                """,
                (model_name,),
            ).fetchall()
            return [_row_to_variant(r) for r in rows]

    # ---- internal: DB mutation --------------------------------------

    def _insert_variant_sync(
        self, conn: sqlite3.Connection, *, model_name: str, template: str
    ) -> str:
        now = time.time()
        vid = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO prompt_variants
                (id, model_name, prompt_template, avg_reward, samples,
                 status, created_at, updated_at)
            VALUES (?, ?, ?, 0.0, 0, 'active', ?, ?)
            """,
            (vid, model_name, template, now, now),
        )
        return vid

    def _insert_variant_safe_sync(
        self, model_name: str, template: str
    ) -> str:
        """Insert a new variant or return the existing id on UNIQUE conflict.

        Why surface the existing id instead of raising: identical mutation
        outputs are common (e.g. paraphrase of an already-terse prompt).
        Treating it as a no-op write keeps weekly_evolve idempotent.
        """
        with self._connect() as conn:
            try:
                vid = self._insert_variant_sync(
                    conn, model_name=model_name, template=template
                )
                conn.commit()
                return vid
            except sqlite3.IntegrityError:
                conn.rollback()
                row = conn.execute(
                    """
                    SELECT id FROM prompt_variants
                    WHERE model_name = ? AND prompt_template = ?
                    """,
                    (model_name, template),
                ).fetchone()
                if row is None:
                    raise
                return str(row["id"])

    def _retire_losers_sync(
        self, model_name: str, min_samples_to_retire: int
    ) -> tuple[list[str], list[str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, samples, avg_reward, created_at
                FROM prompt_variants
                WHERE model_name = ? AND status = 'active'
                ORDER BY avg_reward DESC, samples DESC, created_at ASC
                """,
                (model_name,),
            ).fetchall()
            kept_ids: list[str] = []
            retired_ids: list[str] = []
            now = time.time()
            # Always keep the top-k unconditionally.
            top_k = rows[: self.keep_top_k]
            kept_ids = [str(r["id"]) for r in top_k]
            # The rest are candidates for retirement, but only if they
            # have enough samples to be statistically informative.
            for row in rows[self.keep_top_k :]:
                if int(row["samples"]) < min_samples_to_retire:
                    kept_ids.append(str(row["id"]))
                    continue
                conn.execute(
                    "UPDATE prompt_variants SET status='retired', updated_at=? "
                    "WHERE id = ?",
                    (now, row["id"]),
                )
                retired_ids.append(str(row["id"]))
            conn.commit()
            return kept_ids, retired_ids

    # ---- internal: mutation ---------------------------------------------

    async def _mutate(
        self,
        base: str,
        strategy: MutationStrategy,
        provider: Optional[GeneratorProvider],
    ) -> str:
        """Run one mutation, falling back to offline mode on failure."""
        if provider is None:
            return _offline_mutate(base, strategy, self._rng)
        prompt = _STRATEGY_PROMPTS[strategy].format(template=base)
        try:
            resp = await provider.complete(prompt, max_tokens=300)
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning(
                "self_prompt: generator provider %s failed: %s",
                getattr(provider, "name", "<unknown>"), exc,
            )
            return _offline_mutate(base, strategy, self._rng)
        text = getattr(resp, "response", "") or ""
        err = getattr(resp, "error", "") or ""
        if err or not text.strip():
            logger.warning(
                "self_prompt: generator returned empty/error (err=%r) — fallback",
                err,
            )
            return _offline_mutate(base, strategy, self._rng)
        return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_to_variant(row: sqlite3.Row) -> PromptVariant:
    return PromptVariant(
        id=str(row["id"]),
        model_name=str(row["model_name"]),
        prompt_template=str(row["prompt_template"]),
        avg_reward=float(row["avg_reward"]),
        samples=int(row["samples"]),
        status=str(row["status"]),  # type: ignore[arg-type]
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _clean_template(text: str) -> str:
    """Strip markdown fences, surrounding quotes, and preamble lines.

    LLMs frequently wrap the requested rewrite in ```...```, leading
    quotation marks, or a polite "Here is the rewrite:" preamble. We
    scrub those so the persisted template is usable verbatim.
    """
    if not text:
        return ""
    t = text.strip()
    # Drop fences
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    # Drop leading "Here is..." preambles
    t = re.sub(
        r"^(here(?:'s| is)|sure[,!]|rewritten prompt:?|system prompt:?)\s*",
        "",
        t,
        flags=re.IGNORECASE,
    )
    # Drop wrapping quotes
    if len(t) >= 2 and t[0] in {'"', "'"} and t[-1] == t[0]:
        t = t[1:-1].strip()
    return t.strip()


def _offline_mutate(
    base: str, strategy: MutationStrategy, rng: random.Random
) -> str:
    """Deterministic, no-API mutator used as a fallback.

    Why ship this: weekly_evolve must keep working when the user has not
    configured a premium provider (e.g. in CI, in air-gapped deploys, or
    during the first run after install). The offline mutator is dumb but
    legal: it never produces an empty or syntactically broken prompt.
    """
    base = base.strip()
    if strategy == "paraphrase":
        replacements = {
            "concise": "brief",
            "careful": "thoughtful",
            "directly": "without preamble",
            "precision": "accuracy",
            "fluff": "filler",
        }
        out = base
        for src, dst in replacements.items():
            out = re.sub(rf"\b{re.escape(src)}\b", dst, out)
        if out == base:
            out = base + " Be terse."
        return out
    if strategy == "reorder":
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", base) if s.strip()]
        if len(sentences) <= 1:
            return base + " Prioritize the user's stated constraints first."
        rng.shuffle(sentences)
        return " ".join(sentences)
    # add_instruction
    pool = [s for s in _OFFLINE_ADDONS if s not in base]
    if not pool:
        return base
    addon = rng.choice(pool)
    sep = "" if base.endswith((".", "!", "?")) else "."
    return f"{base}{sep} {addon}"


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class _StubGenerator:
    """Synchronous-ish stub that mimics a Provider for tests.

    Returns a deterministic mutation so we can assert on persisted output
    without depending on an API key. ``error`` is exposed to match the
    real :class:`ModelResponse` shape.
    """

    name = "stub"

    def __init__(self, transform: Any = None) -> None:
        self._transform = transform or (lambda p: "Be terse and cite sources.")

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> Any:
        @dataclass
        class _R:
            response: str
            error: str = ""

        return _R(response=self._transform(prompt))


async def _t_get_current_prompt_seeds_default() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        opt = SelfPromptOptimizer(db_path=Path(td) / "p.db", rng_seed=1)
        got = await opt.get_current_prompt("claude-opus")
        assert got == opt.seed_template
        # Second read should return the same row (no duplicate seeding).
        again = await opt.get_current_prompt("claude-opus")
        assert again == got
        variants = await opt.list_variants("claude-opus")
        assert len(variants) == 1


async def _t_record_outcome_updates_mean() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        opt = SelfPromptOptimizer(db_path=Path(td) / "p.db", rng_seed=1)
        await opt.get_current_prompt("g1")  # seed
        variants = await opt.list_variants("g1")
        vid = variants[0].id
        v1 = await opt.record_outcome(vid, 1.0)
        assert v1.samples == 1
        assert abs(v1.avg_reward - 1.0) < 1e-9
        v2 = await opt.record_outcome(vid, 0.0)
        assert v2.samples == 2
        assert abs(v2.avg_reward - 0.5) < 1e-9
        v3 = await opt.record_outcome(vid, 1.0, sample_count=2)
        # New mean = 0.5 + (1.0 - 0.5) * (2/4) = 0.75
        assert v3.samples == 4
        assert abs(v3.avg_reward - 0.75) < 1e-9


async def _t_propose_variant_with_stub_and_offline() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        opt = SelfPromptOptimizer(db_path=Path(td) / "p.db", rng_seed=1)
        # Online path via stub generator
        vid = await opt.propose_variant(
            "g1",
            mutation_strategy="paraphrase",
            generator_provider=_StubGenerator(),
        )
        assert vid
        variants = await opt.list_variants("g1")
        assert any(v.id == vid for v in variants)
        # Offline path (no provider) — must not raise and must persist a row
        before = len(variants)
        vid2 = await opt.propose_variant(
            "g1", mutation_strategy="add_instruction", generator_provider=None
        )
        assert vid2
        after = await opt.list_variants("g1")
        assert len(after) >= before  # may dedupe but never shrink


async def _t_weekly_evolve_keeps_top_k() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        opt = SelfPromptOptimizer(
            db_path=Path(td) / "p.db", keep_top_k=3, rng_seed=1
        )
        await opt.get_current_prompt("m")  # seed v0

        # Add 4 more variants directly via the offline mutator path so we
        # have enough rows for retirement to fire.
        for strat in ("paraphrase", "reorder", "add_instruction", "paraphrase"):
            await opt.propose_variant(
                "m",
                mutation_strategy=strat,  # type: ignore[arg-type]
                generator_provider=_StubGenerator(
                    transform=lambda p, s=strat: f"variant-{s}-{uuid.uuid4().hex[:6]}"
                ),
            )

        variants = await opt.list_variants("m")
        # Give each enough samples to be retire-eligible, then assign
        # distinct rewards so a stable top-3 emerges.
        for i, v in enumerate(variants):
            await opt.record_outcome(v.id, reward=float(i) / max(1, len(variants) - 1), sample_count=5)

        report = await opt.weekly_evolve(
            "m",
            generator_provider=_StubGenerator(
                transform=lambda p: "variant-weekly-" + uuid.uuid4().hex[:6]
            ),
        )
        # New proposal lands; top-3 active variants remain (plus the new
        # one, which is grace-period protected because samples=0).
        active = await opt.list_variants("m")
        assert len(active) >= 3
        assert report.proposed_id is not None
        assert report.source == "generator"


async def _t_offline_fallback_when_generator_errors() -> None:
    import tempfile

    class _Broken:
        name = "broken"

        async def complete(self, prompt: str, *, max_tokens: int = 800) -> Any:
            raise RuntimeError("simulated outage")

    with tempfile.TemporaryDirectory() as td:
        opt = SelfPromptOptimizer(db_path=Path(td) / "p.db", rng_seed=1)
        vid = await opt.propose_variant(
            "m",
            mutation_strategy="add_instruction",
            generator_provider=_Broken(),
        )
        assert vid  # offline fallback persisted a usable template


async def _run_all_tests() -> None:
    await _t_get_current_prompt_seeds_default()
    await _t_record_outcome_updates_mean()
    await _t_propose_variant_with_stub_and_offline()
    await _t_weekly_evolve_keeps_top_k()
    await _t_offline_fallback_when_generator_errors()
    logger.info("self_prompt: all self-tests passed")


# ---------------------------------------------------------------------------
# PromptRewriter — query-time self-prompting loop
# ---------------------------------------------------------------------------
#
# This sits ABOVE the SelfPromptOptimizer (which is a long-horizon, per-model
# system-prompt bandit). The rewriter operates on a single consensus call:
# when the first pass's consensus confidence is below threshold, it asks the
# strongest available model to *rewrite the user query itself* — clarifying
# ambiguous terms and decomposing complex sub-questions — and the consensus
# engine re-runs with the rewritten prompt.
#
# Design (Quorum-validated, 2026-06-17):
#   * Clarification + decomposition combined in one rewrite (single model
#     call cheaper than two; both failure modes share root cause = vague
#     query that no single model could disambiguate).
#   * Append-as-context rather than full replace: the original is preserved
#     verbatim under an ORIGINAL_QUERY marker so the downstream consensus
#     pass cannot silently lose user intent. The strongest model's expansion
#     follows under a CLARIFIED_QUERY marker.
#   * Default max_attempts=2 (one rewrite is the cheap win; a second is
#     occasionally helpful when the first rewrite was itself ambiguous;
#     three or more is dominated by the original failing on a hard query).
#
# Storage: SQLite at ``$QUORUM_DATA_DIR/self_prompt.db`` — a separate file
# from prompts.db so the meta-learner can scan it without joining against
# the variant bandit. The schema records (original, rewritten, before/after
# confidence, query_class, timestamp) so a downstream model can fit a
# classifier on "which classes benefit from rewriting".


DEFAULT_REWRITE_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_REWRITE_MAX_ATTEMPTS = 2


def _default_rewrite_db_path() -> Path:
    """Location of ``self_prompt.db`` — distinct from ``prompts.db``.

    Honors ``QUORUM_SELF_PROMPT_DB`` for tests that need isolation; falls
    back to ``$QUORUM_DATA_DIR/self_prompt.db`` (default ``~/.quorum/``).
    """
    override = os.getenv("QUORUM_SELF_PROMPT_DB")
    if override:
        return Path(override).expanduser()
    return DATA_DIR / "self_prompt.db"


_REWRITE_INSTRUCTION_TEMPLATE = """\
You are a query-clarification expert. A multi-LLM consensus engine ran the
following USER_QUERY and the models disagreed (consensus confidence
{confidence:.2f}, below the {threshold:.2f} threshold). Query class: {query_class}.

Rewrite the query so it is unambiguous and actionable by an ensemble of LLMs:
  (a) Clarify any ambiguous terms, implicit assumptions, or under-specified
      goals.
  (b) If the query has multiple sub-questions, decompose it into an explicit
      enumerated list of sub-questions.
  (c) Preserve the user's intent — do not invent constraints they did not
      state. When in doubt, prefer narrowing over expanding scope.

Return ONLY the rewritten query as plain text. No preamble, no markdown
fences, no quotation marks, no commentary about what you changed.

USER_QUERY:
{prompt}
"""


def _format_rewritten_prompt(original: str, rewritten: str) -> str:
    """Compose the final prompt sent to the second consensus pass.

    Append-as-context (not replace) so the downstream models still see the
    user's exact words alongside the rewriter's expansion. We delimit with
    explicit MARKER lines instead of just blank lines so providers that
    do prompt-injection sanitisation can spot the boundary.
    """
    return (
        "ORIGINAL_QUERY:\n"
        f"{original.strip()}\n"
        "\n"
        "CLARIFIED_QUERY (auto-expanded after a low-confidence consensus pass):\n"
        f"{rewritten.strip()}"
    )


@dataclass
class PromptRewriter:
    """Rewrites a low-consensus prompt using the strongest available model.

    Wires into :func:`quorum.core.consensus.consensus`: after the first pass
    the engine inspects the confidence; if below ``confidence_threshold``
    the rewriter is invoked and the consensus runs again on the rewritten
    prompt. The original confidence and the post-rewrite confidence are
    logged to SQLite so the meta-learner can later decide whether to skip
    the rewrite entirely for some query classes (negative ROI).

    Why a dataclass: every field has a defensible default (the DB path, the
    threshold, the attempt cap). Tests can override individual fields
    without monkeypatching globals.
    """

    db_path: Path = field(default_factory=_default_rewrite_db_path)
    confidence_threshold: float = DEFAULT_REWRITE_CONFIDENCE_THRESHOLD
    max_attempts: int = DEFAULT_REWRITE_MAX_ATTEMPTS
    rewriter_provider: Optional[GeneratorProvider] = None

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0, 1]; got "
                f"{self.confidence_threshold}"
            )
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1; got {self.max_attempts}"
            )
        self._init_schema()

    # ---- schema ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Short-lived SQLite connection with WAL — same pattern as the
        bandit. Fresh per call so we are safe under :func:`asyncio.to_thread`.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rewrites (
                    id                   TEXT PRIMARY KEY,
                    original             TEXT NOT NULL,
                    rewritten            TEXT NOT NULL,
                    original_confidence  REAL NOT NULL,
                    new_confidence       REAL NOT NULL,
                    delta                REAL NOT NULL,
                    query_class          TEXT NOT NULL DEFAULT 'general',
                    rewriter_name        TEXT NOT NULL DEFAULT '',
                    created_at           REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rewrites_class "
                "ON rewrites(query_class, created_at)"
            )
            conn.commit()

    # ---- public API -----------------------------------------------------

    async def rewrite(
        self,
        prompt: str,
        current_confidence: float,
        query_class: str = "general",
    ) -> Optional[str]:
        """Return a rewritten prompt or ``None`` if rewriting is unnecessary.

        Returns ``None`` (the caller should keep the original answer) when:
          * ``current_confidence`` is already >= ``confidence_threshold``;
          * no rewriter provider is available (no Anthropic key, no OpenAI
            key, and no override was injected) — silent no-op rather than
            raising, so the consensus call degrades gracefully.

        Otherwise returns the composed prompt (original preserved as
        context, plus the rewriter's clarified expansion).
        """
        if not prompt:
            raise ValueError("prompt must be non-empty")

        # Confidence above threshold → no rewrite needed. The consensus
        # engine can keep its first-pass result.
        if current_confidence >= self.confidence_threshold:
            return None

        provider = self.rewriter_provider or _resolve_rewriter_provider()
        if provider is None:
            logger.debug(
                "self_prompt.rewrite: no rewriter provider available; "
                "returning None (consensus will keep first-pass result)"
            )
            return None

        instruction = _REWRITE_INSTRUCTION_TEMPLATE.format(
            prompt=prompt[:MAX_REWRITE_INPUT_BYTES],
            confidence=current_confidence,
            threshold=self.confidence_threshold,
            query_class=query_class or "general",
        )

        try:
            resp = await provider.complete(instruction, max_tokens=600)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "self_prompt.rewrite: rewriter %s raised: %s",
                getattr(provider, "name", "<unknown>"), exc,
            )
            return None

        text = getattr(resp, "response", "") or ""
        err = getattr(resp, "error", "") or ""
        cleaned = _clean_template(text)
        if err or not cleaned:
            logger.warning(
                "self_prompt.rewrite: empty/error from rewriter "
                "(err=%r) — returning None", err,
            )
            return None

        return _format_rewritten_prompt(prompt, cleaned)

    async def log_rewrite(
        self,
        original: str,
        rewritten: str,
        original_confidence: float,
        new_confidence: float,
        *,
        query_class: str = "general",
        rewriter_name: str = "",
    ) -> str:
        """Persist a rewrite outcome so the meta-learner can learn from it.

        ``delta = new_confidence - original_confidence`` is precomputed so
        downstream SQL can ``AVG(delta) GROUP BY query_class`` without
        recomputation. Returns the row id (uuid hex) so tests can fetch it.
        """
        rid = uuid.uuid4().hex
        delta = float(new_confidence) - float(original_confidence)
        now = time.time()
        await asyncio.to_thread(
            self._log_rewrite_sync,
            rid, original, rewritten, float(original_confidence),
            float(new_confidence), delta, query_class, rewriter_name, now,
        )
        logger.info(
            "self_prompt.log_rewrite: class=%s before=%.3f after=%.3f "
            "delta=%+.3f rewriter=%s",
            query_class, original_confidence, new_confidence, delta,
            rewriter_name or "<unset>",
        )
        return rid

    def _log_rewrite_sync(
        self,
        rid: str,
        original: str,
        rewritten: str,
        original_confidence: float,
        new_confidence: float,
        delta: float,
        query_class: str,
        rewriter_name: str,
        now: float,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rewrites
                    (id, original, rewritten, original_confidence,
                     new_confidence, delta, query_class, rewriter_name,
                     created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rid, original, rewritten, original_confidence,
                 new_confidence, delta, query_class, rewriter_name, now),
            )
            conn.commit()

    async def get_rewrite(self, rid: str) -> Optional[dict[str, Any]]:
        """Fetch a single rewrite row by id (used by tests + dashboards)."""
        return await asyncio.to_thread(self._get_rewrite_sync, rid)

    def _get_rewrite_sync(self, rid: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rewrites WHERE id = ?", (rid,)
            ).fetchone()
            return dict(row) if row else None


# Hard cap on the size of the prompt we hand to the rewriter — bounds the
# token bill if a user accidentally pastes an entire transcript.
MAX_REWRITE_INPUT_BYTES = 8_000


def _resolve_rewriter_provider() -> Optional[GeneratorProvider]:
    """Pick the strongest available rewriter, preferring Claude Sonnet.

    Lazy-import so this module imports cleanly with no provider keys set
    (tests, air-gapped CI). Order:
      1. AnthropicProvider(claude-sonnet-4-6) if ANTHROPIC_API_KEY set —
         Sonnet is the cost-quality sweet spot for query rewriting; Opus
         is overkill, Haiku occasionally produces lazy rewrites.
      2. OpenAIProvider(gpt-4.1) if OPENAI_API_KEY set — fallback when no
         Anthropic key.
      3. None — signals to ``rewrite()`` that it should no-op silently.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from quorum.providers.anthropic import claude_sonnet
            return claude_sonnet()  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            logger.debug("self_prompt: anthropic import failed (%s)", exc)
    if os.getenv("OPENAI_API_KEY"):
        try:
            from quorum.providers.openai import OpenAIProvider
            return OpenAIProvider(model="gpt-4.1")  # type: ignore[return-value]
        except Exception as exc:  # noqa: BLE001
            logger.debug("self_prompt: openai import failed (%s)", exc)
    return None


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    asyncio.run(_run_all_tests())
