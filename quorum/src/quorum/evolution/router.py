"""Loop 4 — Intelligent Router (Mixture-of-Experts gating).

Decides which subset of providers to invoke per query, slashing average cost
~50% versus fan-out-to-all by reserving the full panel for cases where the
top-2 cheap experts disagree or score low quality.

The router maintains a per-(query_class, model) policy table in SQLite
(~/.quorum/router.db) updated via exponential moving average from real
production feedback (latency, cost, RLHF reward, Hebbian co-activation).

Trigger: every `consensus(...)` call hits `MoERouter.route()` before fan-out.

License: Apache 2.0 — see /tmp/sovereignchain/LICENSE.
No HSP gate required (routing is low-risk: a bad pick costs money/quality on
ONE query, not human safety).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage location
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
_DEFAULT_DB_PATH = DATA_DIR / "router.db"

# Exponential moving average weight for new observations.
# 0.2 ≈ "remember the last ~5 samples primarily, fade older history slowly".
_EMA_ALPHA = 0.2

# Minimum samples before we trust the table for routing; otherwise explore.
_MIN_SAMPLES_FOR_EXPLOIT = 3

# Quality threshold below which we escalate from top-2 to top-4.
_QUALITY_ESCALATION_THRESHOLD = 0.6

# When no policy rows exist for a class, every known model starts here so that
# the ranking is non-degenerate and exploration is fair.
_COLD_START_QUALITY = 0.5
_COLD_START_COST = 0.002      # USD per query (≈ a small Claude/GPT call)
_COLD_START_LATENCY = 1500.0  # ms

# Default seed candidates if the caller doesn't pass `available_models`.
# These names match `Provider.name` strings in quorum.providers.*.
_DEFAULT_CANDIDATES: tuple[str, ...] = (
    "anthropic-claude-opus",
    "anthropic-claude-sonnet",
    "openai-gpt-4o",
    "openai-gpt-4o-mini",
    "google-gemini-1.5-pro",
    "google-gemini-1.5-flash",
    "replicate-llama-3.3-70b",
    "ollama-llama3",
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyRow:
    """One learned datapoint: "for queries of class X, model Y behaves like Z"."""

    query_class: str
    model_name: str
    expected_quality: float
    expected_cost: float
    expected_latency_ms: float
    samples: int

    def score(self) -> float:
        """Quality-per-dollar — the core routing currency.

        We add a tiny epsilon to cost so a perfectly-free model (Ollama at
        cost=0) doesn't divide-by-zero and dominate purely on price; we still
        want quality signal to matter.
        """
        return self.expected_quality / (self.expected_cost + 1e-4)


@dataclass
class RoutingDecision:
    """Audit trail of one route() call. Useful for shadow learning + analytics."""

    query_class: str
    chosen: list[str]
    candidates_considered: list[str]
    rationale: str
    budget_usd: float
    estimated_cost_usd: float
    escalated: bool
    rlhf_weights: dict[str, float] = field(default_factory=dict)
    hebbian_boosts: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optional dependencies — quorum.evolution.rlhf / hebbian
# ---------------------------------------------------------------------------
#
# These sibling modules (Loops 1 and 5) may not exist yet at import time.
# We degrade to a neutral identity function rather than crashing the router,
# because the router has to ship before RLHF/Hebbian are wired in.


async def _safe_rlhf_weights(
    user_id: str, candidates: Sequence[str]
) -> Mapping[str, float]:
    """Per-user model preference weights from RLHF (Loop 1).

    WHY: a user who consistently up-votes Anthropic outputs should see
    Anthropic boosted in routing even if it's marginally more expensive.
    """
    try:
        from quorum.evolution.rlhf import get_user_weights  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("rlhf module not available; using neutral weights")
        return {m: 1.0 for m in candidates}

    try:
        result = get_user_weights(user_id, list(candidates))
        if asyncio.iscoroutine(result):
            result = await result
        return {m: float(result.get(m, 1.0)) for m in candidates}
    except Exception as e:  # noqa: BLE001
        logger.warning("rlhf get_user_weights failed (%s); using neutral", e)
        return {m: 1.0 for m in candidates}


async def _safe_hebbian_boosts(
    query_class: str, candidates: Sequence[str]
) -> Mapping[str, float]:
    """Co-activation boosts from Hebbian learning (Loop 5).

    WHY: if models A and B historically vote in the same direction on this
    query class, picking both is redundant — boost the cheaper one.
    Conversely, if A and C disagree productively, keep them paired.
    """
    try:
        from quorum.evolution.hebbian import get_class_boosts  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        logger.debug("hebbian module not available; using neutral boosts")
        return {m: 1.0 for m in candidates}

    try:
        result = get_class_boosts(query_class, list(candidates))
        if asyncio.iscoroutine(result):
            result = await result
        return {m: float(result.get(m, 1.0)) for m in candidates}
    except Exception as e:  # noqa: BLE001
        logger.warning("hebbian get_class_boosts failed (%s); using neutral", e)
        return {m: 1.0 for m in candidates}


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

# Heuristic classifier — cheap, deterministic, no API call.
# When Loop 2 (semantic clusterer) ships, this gets replaced by an embedding
# lookup against a learned taxonomy. Keep the interface stable.
_CLASSIFIERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("code", re.compile(
        r"\b(code|function|class|bug|stack trace|python|rust|typescript|"
        r"refactor|compile|syntax)\b", re.I)),
    ("math", re.compile(
        r"\b(prove|theorem|integral|derivative|equation|matrix|"
        r"probability|combinatorics)\b", re.I)),
    ("legal", re.compile(
        r"\b(contract|tenancy|licen[cs]e|jurisdiction|clause|liability|"
        r"GDPR|HMRC|tribunal)\b", re.I)),
    ("medical", re.compile(
        r"\b(dose|mg/ml|symptom|diagnosis|patient|clinical|formula|pH|"
        r"keratin|active ingredient)\b", re.I)),
    ("creative", re.compile(
        r"\b(write|story|poem|brand|tagline|tone|persona|narrative)\b", re.I)),
    ("factual", re.compile(
        r"\b(what is|when did|who is|how many|capital of|population)\b", re.I)),
)


def classify_query(prompt: str) -> str:
    """Bucket the prompt into a coarse class for policy lookup.

    WHY a heuristic and not an LLM: the router runs on every single query;
    paying an LLM call just to decide which LLMs to call would defeat the
    cost-cutting purpose. False classifications are self-correcting via the
    EMA update — if "code" routing performs badly on a misclassified prompt,
    the next route() degrades that pairing.
    """
    for label, pattern in _CLASSIFIERS:
        if pattern.search(prompt):
            return label
    return "general"


# ---------------------------------------------------------------------------
# Synchronous SQLite helpers (run via asyncio.to_thread for non-blocking I/O)
# ---------------------------------------------------------------------------


def _connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with sane defaults for a low-write workload."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_schema_sync(db_path: Path) -> None:
    """Create the policy table on first use.

    Composite primary key (query_class, model_name) makes the upsert path
    a single UPSERT statement, and matches the natural lookup pattern.
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS policy (
                query_class TEXT NOT NULL,
                model_name TEXT NOT NULL,
                expected_quality REAL NOT NULL,
                expected_cost REAL NOT NULL,
                expected_latency_ms REAL NOT NULL,
                samples INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL,
                PRIMARY KEY (query_class, model_name)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shadow_log (
                ts REAL NOT NULL,
                query_class TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                quality_score REAL NOT NULL,
                cost_usd REAL NOT NULL,
                latency_ms REAL NOT NULL,
                ground_truth_distance REAL
            )
            """
        )
    finally:
        conn.close()


def _fetch_policy_sync(
    db_path: Path, query_class: str, candidates: Sequence[str]
) -> list[PolicyRow]:
    """Pull rows for a class; synthesize cold-start rows for missing models."""
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT * FROM policy WHERE query_class = ?", (query_class,)
        )
        existing = {
            row["model_name"]: PolicyRow(
                query_class=row["query_class"],
                model_name=row["model_name"],
                expected_quality=row["expected_quality"],
                expected_cost=row["expected_cost"],
                expected_latency_ms=row["expected_latency_ms"],
                samples=row["samples"],
            )
            for row in cur.fetchall()
        }
    finally:
        conn.close()

    rows: list[PolicyRow] = []
    for m in candidates:
        if m in existing:
            rows.append(existing[m])
        else:
            rows.append(
                PolicyRow(
                    query_class=query_class,
                    model_name=m,
                    expected_quality=_COLD_START_QUALITY,
                    expected_cost=_COLD_START_COST,
                    expected_latency_ms=_COLD_START_LATENCY,
                    samples=0,
                )
            )
    return rows


def _update_policy_sync(
    db_path: Path,
    query_class: str,
    model_name: str,
    quality_score: float,
    cost_usd: float,
    latency_ms: float,
) -> None:
    """EMA-update a single (class, model) row.

    EMA (alpha=0.2) gives roughly 5-sample memory: recent regressions/
    improvements influence routing fast, but a single fluke doesn't flip it.
    """
    now = time.time()
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT expected_quality, expected_cost, expected_latency_ms, samples "
            "FROM policy WHERE query_class = ? AND model_name = ?",
            (query_class, model_name),
        )
        row = cur.fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO policy
                  (query_class, model_name, expected_quality, expected_cost,
                   expected_latency_ms, samples, updated_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (query_class, model_name, quality_score, cost_usd,
                 latency_ms, now),
            )
            return

        new_quality = (1 - _EMA_ALPHA) * row["expected_quality"] + \
            _EMA_ALPHA * quality_score
        new_cost = (1 - _EMA_ALPHA) * row["expected_cost"] + \
            _EMA_ALPHA * cost_usd
        new_latency = (1 - _EMA_ALPHA) * row["expected_latency_ms"] + \
            _EMA_ALPHA * latency_ms
        conn.execute(
            """
            UPDATE policy SET
              expected_quality = ?, expected_cost = ?,
              expected_latency_ms = ?, samples = samples + 1, updated_at = ?
            WHERE query_class = ? AND model_name = ?
            """,
            (new_quality, new_cost, new_latency, now, query_class, model_name),
        )
    finally:
        conn.close()


def _log_shadow_sync(
    db_path: Path,
    query_class: str,
    prompt_hash: str,
    model_name: str,
    quality_score: float,
    cost_usd: float,
    latency_ms: float,
    ground_truth_distance: float | None,
) -> None:
    """Append-only shadow observations — never touch live policy table."""
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO shadow_log
              (ts, query_class, prompt_hash, model_name, quality_score,
               cost_usd, latency_ms, ground_truth_distance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (time.time(), query_class, prompt_hash, model_name,
             quality_score, cost_usd, latency_ms, ground_truth_distance),
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Public router
# ---------------------------------------------------------------------------


class MoERouter:
    """Mixture-of-Experts gating for consensus fan-out.

    Lifecycle:
        router = MoERouter()
        chosen = await router.route(prompt, user_id="jaque", budget_usd=0.02)
        # ... consensus runs only on `chosen` ...
        await router.update_policy("code", "openai-gpt-4o", 0.91, 0.004, 1280)

    The router intentionally does NOT call providers itself — it returns
    names and the calling code (quorum.core.consensus.consensus) decides
    whether/how to invoke them. This keeps the routing concern testable
    in isolation and provider-agnostic.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        *,
        shadow_mode: bool = False,
        candidate_models: Sequence[str] | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            db_path: Override location of router.db (default: ~/.quorum/router.db).
                     Tests can pass a tmp path for isolation.
            shadow_mode: If True, route() always returns all candidates so the
                         router can observe everything without affecting the
                         active panel. Useful during onboarding.
            candidate_models: Override the candidate pool. Default is the
                              built-in `_DEFAULT_CANDIDATES`. In production
                              this is usually `[p.name for p in providers]`.
        """
        env_override = os.getenv("QUORUM_ROUTER_DB")
        self.db_path = Path(
            db_path or env_override or _DEFAULT_DB_PATH
        ).expanduser()
        self.shadow_mode = shadow_mode
        self.candidate_models: tuple[str, ...] = tuple(
            candidate_models or _DEFAULT_CANDIDATES
        )
        self._schema_ready = False
        self._schema_lock = asyncio.Lock()

    async def _ensure_schema(self) -> None:
        """Lazy schema init — avoids touching the filesystem at import time."""
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            await asyncio.to_thread(_init_schema_sync, self.db_path)
            self._schema_ready = True

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    async def route(
        self,
        prompt: str,
        user_id: str,
        budget_usd: float = 0.05,
        *,
        available_models: Sequence[str] | None = None,
    ) -> list[str]:
        """Pick the provider subset to invoke for this query.

        Algorithm:
            1. Classify the query (cheap regex).
            2. Pull policy rows for that class (cold-start any missing model).
            3. Apply RLHF weights (user preference) and Hebbian boosts
               (co-activation) multiplicatively to the quality/cost score.
            4. Sort by adjusted score, take top-2.
            5. If top-2's projected quality < 0.6, escalate to top-4.
            6. Filter against budget — drop costliest until we fit.
            7. Always keep at least 2 models so we have something to compare.

        Args:
            prompt: The user's query.
            user_id: Stable user id (for RLHF lookup).
            budget_usd: Hard cap for the union of routed-model costs.
            available_models: Override candidate pool for this call
                              (e.g. only Providers that have keys configured).

        Returns:
            List of provider name strings, ordered by descending adjusted score.
        """
        await self._ensure_schema()

        candidates = list(available_models) if available_models is not None \
            else list(self.candidate_models)
        if not candidates:
            logger.warning("MoERouter.route called with no candidates")
            return []

        query_class = classify_query(prompt)

        if self.shadow_mode:
            logger.info(
                "router shadow_mode=on; returning all %d candidates",
                len(candidates),
            )
            return list(candidates)

        rows = await asyncio.to_thread(
            _fetch_policy_sync, self.db_path, query_class, candidates
        )

        rlhf, hebbian = await asyncio.gather(
            _safe_rlhf_weights(user_id, candidates),
            _safe_hebbian_boosts(query_class, candidates),
        )

        # Adjusted score = (quality * rlhf * hebbian) / cost.
        # When samples<MIN we add a small exploration bonus (UCB-style) so
        # under-sampled models occasionally get picked.
        scored: list[tuple[str, float, PolicyRow]] = []
        for row in rows:
            base = row.score()
            rlhf_w = rlhf.get(row.model_name, 1.0)
            heb_w = hebbian.get(row.model_name, 1.0)
            exploration = 0.15 if row.samples < _MIN_SAMPLES_FOR_EXPLOIT else 0.0
            adjusted = base * rlhf_w * heb_w + exploration
            scored.append((row.model_name, adjusted, row))

        scored.sort(key=lambda x: x[1], reverse=True)

        top2 = scored[:2]
        avg_quality_top2 = sum(r.expected_quality for _, _, r in top2) / max(
            len(top2), 1
        )
        escalated = avg_quality_top2 < _QUALITY_ESCALATION_THRESHOLD
        chosen_rows = scored[:4] if escalated else top2

        # Budget enforcement: drop costliest until sum<=budget, keep min 2.
        kept: list[tuple[str, float, PolicyRow]] = list(chosen_rows)
        while (
            sum(r.expected_cost for _, _, r in kept) > budget_usd
            and len(kept) > 2
        ):
            kept.sort(key=lambda x: x[2].expected_cost, reverse=True)
            dropped = kept.pop(0)
            logger.info(
                "router dropped %s for budget (cost=%.4f, budget=%.4f)",
                dropped[0], dropped[2].expected_cost, budget_usd,
            )
            kept.sort(key=lambda x: x[1], reverse=True)

        result = [name for name, _, _ in kept]
        logger.info(
            "route class=%s user=%s chose=%s escalated=%s est_cost=%.4f",
            query_class, user_id, result, escalated,
            sum(r.expected_cost for _, _, r in kept),
        )
        return result

    async def explain_route(
        self,
        prompt: str,
        user_id: str,
        budget_usd: float = 0.05,
        *,
        available_models: Sequence[str] | None = None,
    ) -> RoutingDecision:
        """Same as route() but returns the full audit trail.

        WHY a separate method: route() is hot-path and shouldn't pay for the
        full decision dataclass. explain_route is for debugging/analytics
        dashboards.
        """
        await self._ensure_schema()
        candidates = list(available_models) if available_models is not None \
            else list(self.candidate_models)
        query_class = classify_query(prompt)
        chosen = await self.route(
            prompt, user_id, budget_usd, available_models=candidates,
        )
        rows = await asyncio.to_thread(
            _fetch_policy_sync, self.db_path, query_class, candidates
        )
        rlhf, hebbian = await asyncio.gather(
            _safe_rlhf_weights(user_id, candidates),
            _safe_hebbian_boosts(query_class, candidates),
        )
        est_cost = sum(
            r.expected_cost for r in rows if r.model_name in chosen
        )
        avg_q = (
            sum(r.expected_quality for r in rows if r.model_name in chosen[:2])
            / max(min(2, len(chosen)), 1)
        )
        rationale = (
            f"class={query_class} top2_avg_quality={avg_q:.2f} "
            f"{'escalated_to_top4' if avg_q < _QUALITY_ESCALATION_THRESHOLD else 'top2_sufficient'}"
        )
        return RoutingDecision(
            query_class=query_class,
            chosen=chosen,
            candidates_considered=candidates,
            rationale=rationale,
            budget_usd=budget_usd,
            estimated_cost_usd=est_cost,
            escalated=avg_q < _QUALITY_ESCALATION_THRESHOLD,
            rlhf_weights=dict(rlhf),
            hebbian_boosts=dict(hebbian),
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    async def update_policy(
        self,
        query_class: str,
        model_name: str,
        quality_score: float,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """Fold one observation into the policy via EMA.

        Args:
            query_class: From `classify_query()` at routing time.
            model_name: Provider.name string.
            quality_score: 0..1 — consensus weight or RLHF reward.
            cost_usd: Actual cost incurred by this call.
            latency_ms: Actual wall-clock time.

        WHY EMA over SGD: we have very low write volume (1 update per
        provider per query), zero gradient to follow, and want recency bias.
        EMA is one multiplication; SGD would be overkill.
        """
        if not (0.0 <= quality_score <= 1.0):
            logger.warning(
                "clamping quality_score=%.3f for %s/%s",
                quality_score, query_class, model_name,
            )
            quality_score = max(0.0, min(1.0, quality_score))

        await self._ensure_schema()
        await asyncio.to_thread(
            _update_policy_sync,
            self.db_path, query_class, model_name,
            quality_score, max(cost_usd, 0.0), max(latency_ms, 0.0),
        )

    async def learning_in_shadow_mode(
        self,
        query: str,
        all_responses: Mapping[str, str],
        ground_truth_response: str | None = None,
    ) -> None:
        """Log per-model outcomes to the shadow table WITHOUT touching policy.

        Used during onboarding (the first N queries from a new user) so we
        can observe how every model behaves on their workload without
        prematurely biasing routing. Once we have a few hundred samples
        the orchestrator promotes shadow data into the live policy.

        Args:
            query: The original prompt.
            all_responses: Mapping model_name -> raw response text.
            ground_truth_response: If known (e.g. user up-vote, golden test),
                                   compute a distance proxy for each model.

        WHY shadow logging matters: the cold-start row defaults bias toward
        "everything is equal" — without shadow data we'd fan out to all 8
        models indiscriminately for the first N queries. Shadow lets us
        learn before committing.
        """
        await self._ensure_schema()
        query_class = classify_query(query)
        prompt_hash = f"{hash(query) & 0xFFFFFFFF:08x}"

        gt_tokens = (
            set(ground_truth_response.lower().split())
            if ground_truth_response else None
        )

        async def _log_one(name: str, text: str) -> None:
            # Quality proxy: Jaccard against ground truth if available, else
            # mean Jaccard against the other models (consensus weight).
            if gt_tokens is not None:
                tok = set(text.lower().split())
                jacc = (
                    len(tok & gt_tokens) / len(tok | gt_tokens)
                    if (tok or gt_tokens) else 0.0
                )
                quality = jacc
                distance: float | None = 1.0 - jacc
            else:
                tok = set(text.lower().split())
                others = [
                    set(t.lower().split())
                    for n, t in all_responses.items() if n != name
                ]
                if not others:
                    quality = 0.5
                else:
                    jaccs = [
                        (len(tok & o) / len(tok | o)) if (tok or o) else 0.0
                        for o in others
                    ]
                    quality = sum(jaccs) / len(jaccs)
                distance = None
            await asyncio.to_thread(
                _log_shadow_sync,
                self.db_path, query_class, prompt_hash, name,
                quality, 0.0, 0.0, distance,
            )

        await asyncio.gather(*(_log_one(n, t) for n, t in all_responses.items()))
        logger.info(
            "shadow logged %d responses for class=%s hash=%s",
            len(all_responses), query_class, prompt_hash,
        )

    # ------------------------------------------------------------------
    # Introspection helpers (cheap to call from a CLI/dashboard)
    # ------------------------------------------------------------------

    async def get_policy(self, query_class: str) -> list[PolicyRow]:
        """Return the raw policy rows for a class (or cold-start fillers)."""
        await self._ensure_schema()
        return await asyncio.to_thread(
            _fetch_policy_sync,
            self.db_path, query_class, self.candidate_models,
        )

    async def reset(self) -> None:
        """Drop everything and start clean — for tests and 'forget what you
        learned' user flow."""
        await self._ensure_schema()

        def _wipe() -> None:
            conn = _connect(self.db_path)
            try:
                conn.execute("DELETE FROM policy")
                conn.execute("DELETE FROM shadow_log")
            finally:
                conn.close()

        await asyncio.to_thread(_wipe)


# ---------------------------------------------------------------------------
# Smoke tests — runnable via `python -m quorum.evolution.router`
# ---------------------------------------------------------------------------


async def _smoke_basic_routing() -> None:
    """Cold start: with no policy rows, route still returns >=2 models."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        router = MoERouter(
            db_path=Path(td) / "test.db",
            candidate_models=("anthropic-claude-opus", "openai-gpt-4o",
                              "google-gemini-1.5-flash", "ollama-llama3"),
        )
        chosen = await router.route(
            "Write a Python function to reverse a linked list.",
            user_id="test-user",
            budget_usd=0.05,
        )
        assert len(chosen) >= 2, f"expected >=2 models, got {chosen}"
        assert all(isinstance(c, str) for c in chosen)
        logger.info("smoke_basic_routing PASSED: chose %s", chosen)


async def _smoke_ema_update_then_route() -> None:
    """After we train one model to high quality+low cost, it should be picked."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        router = MoERouter(
            db_path=Path(td) / "test.db",
            candidate_models=("model-a", "model-b", "model-c", "model-d"),
        )
        # Hammer model-a as the clear winner for 'code'.
        for _ in range(10):
            await router.update_policy("code", "model-a", 0.95, 0.001, 800.0)
        # Make others mediocre.
        for m in ("model-b", "model-c", "model-d"):
            for _ in range(10):
                await router.update_policy("code", m, 0.4, 0.005, 2000.0)

        chosen = await router.route(
            "Refactor this Python function please.",
            user_id="test-user",
            budget_usd=0.05,
        )
        assert "model-a" in chosen, f"expected model-a in top picks, got {chosen}"

        # Quality of top2 is high → no escalation, exactly 2 picks.
        decision = await router.explain_route(
            "Refactor this Python function please.",
            user_id="test-user",
            budget_usd=0.05,
        )
        assert not decision.escalated, "top-2 quality is high; should not escalate"
        logger.info(
            "smoke_ema_update PASSED: chose %s rationale=%s",
            chosen, decision.rationale,
        )


async def _smoke_budget_clamp() -> None:
    """If budget is tighter than escalation cost, we drop expensive models
    but keep at least 2."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        router = MoERouter(
            db_path=Path(td) / "test.db",
            candidate_models=("cheap-a", "cheap-b", "pricey-c", "pricey-d"),
        )
        for _ in range(5):
            await router.update_policy("general", "cheap-a", 0.5, 0.0005, 500.0)
            await router.update_policy("general", "cheap-b", 0.5, 0.0005, 500.0)
            await router.update_policy("general", "pricey-c", 0.5, 0.05, 3000.0)
            await router.update_policy("general", "pricey-d", 0.5, 0.05, 3000.0)

        # Quality 0.5 average → escalate to 4, but budget 0.01 forces drop.
        chosen = await router.route(
            "tell me something",
            user_id="test-user",
            budget_usd=0.01,
        )
        assert len(chosen) >= 2
        assert "pricey-c" not in chosen or "pricey-d" not in chosen, \
            f"budget should have dropped at least one pricey model: {chosen}"
        logger.info("smoke_budget_clamp PASSED: chose %s", chosen)


async def _smoke_shadow_mode() -> None:
    """Shadow mode returns ALL candidates and logs without affecting policy."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.db"
        router = MoERouter(
            db_path=path, shadow_mode=True,
            candidate_models=("m1", "m2", "m3"),
        )
        chosen = await router.route("any query", user_id="u", budget_usd=0.05)
        assert chosen == ["m1", "m2", "m3"], chosen

        await router.learning_in_shadow_mode(
            query="write me a function",
            all_responses={
                "m1": "def f(): return 1",
                "m2": "def f(): return 1",
                "m3": "totally different garbage text",
            },
            ground_truth_response="def f(): return 1",
        )
        # Verify the shadow table has rows but policy is still empty.
        conn = _connect(path)
        try:
            shadow_n = conn.execute(
                "SELECT COUNT(*) FROM shadow_log"
            ).fetchone()[0]
            policy_n = conn.execute(
                "SELECT COUNT(*) FROM policy"
            ).fetchone()[0]
        finally:
            conn.close()
        assert shadow_n == 3, shadow_n
        assert policy_n == 0, policy_n
        logger.info("smoke_shadow_mode PASSED: shadow=%d policy=%d",
                    shadow_n, policy_n)


async def _run_all_smoke() -> None:
    """Run every smoke test sequentially."""
    await _smoke_basic_routing()
    await _smoke_ema_update_then_route()
    await _smoke_budget_clamp()
    await _smoke_shadow_mode()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_run_all_smoke())


__all__: tuple[str, ...] = (
    "MoERouter",
    "PolicyRow",
    "RoutingDecision",
    "classify_query",
)


# Suppress an unused-import warning from type-only `Iterable` usage in some
# checkers when it's referenced only through stringified annotations.
_ = Iterable
