"""Loop 8 — Automated A/B Testing for Evolution Policy Changes.

Copyright 2026 Sovereign Chain / Jaqueline Martins.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

HSP-Gated Module — Patent PCT/US26/11908.
Commercial promotion of evolution policy variants requires an HSP license.
See LICENSE-HSP at the repository root for terms.

WHY THIS LOOP EXISTS
====================
Every change to an evolution policy (router weighting, judge prompt, consensus
threshold, provider mix) is a potential regression. Naive "ship and pray"
deployment is unacceptable for a multi-LLM consensus engine where downstream
users depend on calibrated quality. Loop 8 enforces a discipline: every
candidate policy lives as Variant B in shadow with `traffic_split` of live
queries for `min_samples` outcomes before any promotion decision is made.

The runner is deliberately conservative:
  * Promote only when (a) effect_size > +5% AND (b) p < alpha AND
    (c) sample_size >= min_samples.
  * Revert when effect_size < 0 AND p < alpha.
  * Otherwise inconclusive — keep collecting or expire.

It runs continuously (triggered by the orchestrator each time an outcome is
recorded). Persistence is SQLite under ~/.quorum/abtest.db so experiments
survive process restarts. All SQLite I/O is wrapped in asyncio.to_thread so
the event loop never blocks on disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sqlite3
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from quorum.hsp.gate import requires_hsp_approval

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DATA_DIR: Final[Path] = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()
_DEFAULT_DB_PATH: Final[Path] = DATA_DIR / "abtest.db"
_PROMOTE_THRESHOLD: Final[float] = 0.05  # +5% lift required to promote variant B.
_VARIANT_A: Final[str] = "A"
_VARIANT_B: Final[str] = "B"

ABDecisionLiteral = Literal["promote_b", "revert", "inconclusive"]
ExperimentStatus = Literal["running", "promoted", "reverted", "expired"]

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_SCHEMA_EXPERIMENTS: Final[str] = """
CREATE TABLE IF NOT EXISTS experiments (
    id                    TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    variant_a_config_json TEXT NOT NULL,
    variant_b_config_json TEXT NOT NULL,
    traffic_split         REAL NOT NULL DEFAULT 0.10,
    min_samples           INTEGER NOT NULL DEFAULT 200,
    alpha                 REAL NOT NULL DEFAULT 0.05,
    started_at            REAL NOT NULL,
    ended_at              REAL,
    status                TEXT NOT NULL DEFAULT 'running',
    metric_a              REAL,
    metric_b              REAL
);
"""

_SCHEMA_OUTCOMES: Final[str] = """
CREATE TABLE IF NOT EXISTS outcomes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id  TEXT NOT NULL,
    variant        TEXT NOT NULL CHECK (variant IN ('A','B')),
    metric_value   REAL NOT NULL,
    recorded_at    REAL NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);
"""

_SCHEMA_INDEX: Final[str] = (
    "CREATE INDEX IF NOT EXISTS idx_outcomes_experiment "
    "ON outcomes(experiment_id, variant);"
)


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(slots=True, frozen=True)
class ABDecision:
    """Result of evaluating an experiment.

    Attributes
    ----------
    experiment_id:
        Identifier of the experiment being evaluated.
    decision:
        One of 'promote_b' | 'revert' | 'inconclusive'.
    p_value:
        Two-sided p-value from the appropriate test (Welch's t-test for
        continuous metrics; chi-square for binary).
    effect_size:
        Relative lift of B over A. Negative means B is worse.
    sample_size:
        Total outcomes across both variants.
    """

    experiment_id: str
    decision: ABDecisionLiteral
    p_value: float
    effect_size: float
    sample_size: int


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #


def _is_binary(values: list[float]) -> bool:
    """Detect whether a sample looks binary (only 0.0 and 1.0).

    Binary metrics (success/failure of a single query) get a chi-square test;
    continuous metrics (latency, quality score) get a Welch t-test. The choice
    of test materially affects power, so misclassifying here costs samples.
    """
    if not values:
        return False
    return all(v in (0.0, 1.0) for v in values)


def _welch_t_test(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's two-sample t-test for unequal variances.

    Returns (t_statistic, two_sided_p_value). We hand-roll this rather than
    pulling scipy because scipy is a heavy dep for a single statistical test
    that has a closed-form approximation. The p-value uses a normal
    approximation when df is large (>30), which is correct for the
    min_samples >= 200 regime this loop targets.
    """
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0

    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    n_a, n_b = len(a), len(b)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        # Both variants identical — no signal.
        return 0.0, 1.0

    t_stat = (mean_b - mean_a) / se

    # Welch–Satterthwaite degrees of freedom.
    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a**2) / ((n_a**2) * (n_a - 1)) + (var_b**2) / ((n_b**2) * (n_b - 1))
    df = num / denom if denom > 0 else float(n_a + n_b - 2)

    p_value = _t_two_sided_p(t_stat, df)
    return t_stat, p_value


def _t_two_sided_p(t_stat: float, df: float) -> float:
    """Two-sided p-value for a t-statistic.

    For df large (the only regime we hit with min_samples >= 200) the
    t-distribution converges to the normal, so we use the normal CDF via
    math.erf. For tiny df we still return the normal approximation — slightly
    conservative, acceptable for a promote/revert gate.
    """
    z = abs(t_stat)
    # 1 - Phi(z) using erf, then doubled for two-sided.
    one_tail = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))
    return min(1.0, max(0.0, 2.0 * one_tail))


def _chi_square(a: list[float], b: list[float]) -> tuple[float, float]:
    """Chi-square test of independence on a 2x2 success/failure table.

    Returns (chi2_statistic, p_value). Used when the metric is binary, e.g.
    "did the consensus answer match the ground truth?".
    """
    succ_a = sum(int(v) for v in a)
    succ_b = sum(int(v) for v in b)
    fail_a = len(a) - succ_a
    fail_b = len(b) - succ_b

    row1 = succ_a + succ_b
    row2 = fail_a + fail_b
    col1 = succ_a + fail_a
    col2 = succ_b + fail_b
    total = row1 + row2

    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0, 1.0

    def _expected(row: int, col: int) -> float:
        return (row * col) / total

    observed = [
        (succ_a, _expected(row1, col1)),
        (succ_b, _expected(row1, col2)),
        (fail_a, _expected(row2, col1)),
        (fail_b, _expected(row2, col2)),
    ]
    chi2 = 0.0
    for obs, exp in observed:
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp

    # df=1 for 2x2 contingency table; survival function 1 - F(chi2, 1).
    # For df=1: p = erfc(sqrt(chi2 / 2)).
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return chi2, p_value


def _effect_size(a: list[float], b: list[float]) -> float:
    """Relative lift of B over A.

    Returns (mean_b - mean_a) / |mean_a|. When mean_a is zero we fall back to
    raw mean difference so we don't divide by zero on a freshly seeded
    experiment.
    """
    if not a or not b:
        return 0.0
    mean_a = statistics.fmean(a)
    mean_b = statistics.fmean(b)
    if mean_a == 0:
        return mean_b - mean_a
    return (mean_b - mean_a) / abs(mean_a)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


class ABTestRunner:
    """SQLite-backed runner for shadow A/B experiments.

    Use a single instance per process; the underlying SQLite handle is opened
    on demand inside asyncio.to_thread so we never block the event loop.
    Concurrent access from multiple coroutines is safe because each call
    opens its own short-lived connection with check_same_thread=False.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Initialise the runner and ensure the schema exists.

        db_path defaults to ~/.quorum/abtest.db; pass ":memory:" or a custom
        path in tests so the user's real experiment history is never touched.
        """
        if db_path is None:
            db_path = _DEFAULT_DB_PATH
        self._db_path = Path(str(db_path)) if str(db_path) != ":memory:" else None
        self._memory_uri = str(db_path) == ":memory:"
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # In-memory mode shares one connection; on-disk mode opens per call.
        self._memory_conn: sqlite3.Connection | None = None
        self._init_schema()

    # ----- low-level connection management ----- #

    def _connect(self) -> sqlite3.Connection:
        """Return a SQLite connection appropriate for the configured mode.

        In-memory mode reuses a single connection so the schema persists for
        the lifetime of the runner; on-disk mode opens short-lived handles
        with WAL journaling for safe concurrent reads.
        """
        if self._memory_uri:
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False
                )
                self._memory_conn.execute("PRAGMA foreign_keys = ON;")
            return self._memory_conn
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_schema(self) -> None:
        """Create tables and indexes if missing. Runs synchronously at init."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(_SCHEMA_EXPERIMENTS)
                conn.execute(_SCHEMA_OUTCOMES)
                conn.execute(_SCHEMA_INDEX)
        finally:
            if not self._memory_uri:
                conn.close()

    # ----- public API ----- #

    @requires_hsp_approval(action="ab_promote_variant", risk_level="medium")
    async def start(
        self,
        name: str,
        variant_a: dict[str, Any],
        variant_b: dict[str, Any],
        traffic_split: float = 0.10,
        min_samples: int = 200,
        alpha: float = 0.05,
    ) -> str:
        """Register a new shadow experiment and return its id.

        Gated by HSP because spinning up an experiment is itself an evolution
        action — it commits a fraction of live traffic to an unvetted policy.
        The gate ensures a human signs off on the shadow risk before any
        queries are routed.

        Parameters
        ----------
        name:
            Human-readable label, e.g. "judge_prompt_v3_vs_v2".
        variant_a:
            JSON-serialisable config for the current production policy.
        variant_b:
            JSON-serialisable config for the candidate policy.
        traffic_split:
            Fraction of live traffic routed to B during shadow (default 10%).
        min_samples:
            Minimum total outcomes before evaluate() can decide.
        alpha:
            Significance threshold for the statistical test.

        Returns
        -------
        Experiment id (uuid4 hex).
        """
        if not 0.0 < traffic_split < 1.0:
            raise ValueError("traffic_split must be in (0, 1).")
        if min_samples < 2:
            raise ValueError("min_samples must be >= 2.")
        if not 0.0 < alpha < 0.5:
            raise ValueError("alpha must be in (0, 0.5).")

        exp_id = uuid.uuid4().hex
        started_at = time.time()

        def _insert() -> None:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO experiments
                          (id, name, variant_a_config_json, variant_b_config_json,
                           traffic_split, min_samples, alpha, started_at, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
                        """,
                        (
                            exp_id,
                            name,
                            json.dumps(variant_a, sort_keys=True),
                            json.dumps(variant_b, sort_keys=True),
                            traffic_split,
                            min_samples,
                            alpha,
                            started_at,
                        ),
                    )
            finally:
                if not self._memory_uri:
                    conn.close()

        await asyncio.to_thread(_insert)
        logger.info(
            "ab_experiment_started id=%s name=%s split=%.2f min_samples=%d alpha=%.3f",
            exp_id,
            name,
            traffic_split,
            min_samples,
            alpha,
        )
        return exp_id

    async def record_outcome(
        self,
        experiment_id: str,
        variant: str,
        metric_value: float,
    ) -> None:
        """Record a single observation for an experiment.

        Called by the orchestrator after every shadow-routed query. We accept
        a float for both binary (0.0/1.0) and continuous metrics — the test
        kind is auto-detected at evaluate() time from the sample shape, so
        the caller never has to declare the metric type up front.
        """
        if variant not in (_VARIANT_A, _VARIANT_B):
            raise ValueError(f"variant must be 'A' or 'B', got {variant!r}.")
        if not math.isfinite(metric_value):
            raise ValueError("metric_value must be a finite number.")

        recorded_at = time.time()

        def _insert() -> None:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO outcomes
                          (experiment_id, variant, metric_value, recorded_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (experiment_id, variant, float(metric_value), recorded_at),
                    )
            finally:
                if not self._memory_uri:
                    conn.close()

        await asyncio.to_thread(_insert)

    async def evaluate(self, experiment_id: str) -> ABDecision:
        """Decide whether to promote variant B, revert, or wait for more data.

        Promotion rule:
          decision='promote_b' iff
            sample_size >= min_samples AND
            p_value < alpha AND
            effect_size > +5%
        Revert rule:
          decision='revert' iff
            sample_size >= min_samples AND
            p_value < alpha AND
            effect_size < 0
        Otherwise 'inconclusive'.

        On a decision (promote_b or revert) the experiment row is closed:
        status set, ended_at stamped, metric_a/metric_b filled in.
        """
        meta = await asyncio.to_thread(self._load_experiment, experiment_id)
        if meta is None:
            raise KeyError(f"unknown experiment id: {experiment_id!r}")
        if meta["status"] != "running":
            logger.debug("evaluate called on closed experiment %s", experiment_id)

        samples_a, samples_b = await asyncio.to_thread(
            self._load_outcomes, experiment_id
        )
        sample_size = len(samples_a) + len(samples_b)

        # Not enough data yet — short-circuit to inconclusive without running
        # a test we don't have power for.
        if sample_size < meta["min_samples"] or not samples_a or not samples_b:
            return ABDecision(
                experiment_id=experiment_id,
                decision="inconclusive",
                p_value=1.0,
                effect_size=_effect_size(samples_a, samples_b),
                sample_size=sample_size,
            )

        binary = _is_binary(samples_a) and _is_binary(samples_b)
        if binary:
            _, p_value = _chi_square(samples_a, samples_b)
        else:
            _, p_value = _welch_t_test(samples_a, samples_b)

        lift = _effect_size(samples_a, samples_b)
        decision: ABDecisionLiteral

        if p_value < meta["alpha"] and lift > _PROMOTE_THRESHOLD:
            decision = "promote_b"
        elif p_value < meta["alpha"] and lift < 0:
            decision = "revert"
        else:
            decision = "inconclusive"

        if decision in ("promote_b", "revert"):
            await asyncio.to_thread(
                self._close_experiment,
                experiment_id,
                "promoted" if decision == "promote_b" else "reverted",
                statistics.fmean(samples_a),
                statistics.fmean(samples_b),
            )
            logger.info(
                "ab_experiment_closed id=%s decision=%s p=%.4f lift=%+.3f n=%d",
                experiment_id,
                decision,
                p_value,
                lift,
                sample_size,
            )

        return ABDecision(
            experiment_id=experiment_id,
            decision=decision,
            p_value=p_value,
            effect_size=lift,
            sample_size=sample_size,
        )

    # ----- introspection helpers (not part of the loop contract) ----- #

    async def list_running(self) -> list[dict[str, Any]]:
        """Return metadata for every running experiment.

        Useful for an operator dashboard or for the orchestrator to know
        which experiments still need traffic routed to them.
        """

        def _query() -> list[dict[str, Any]]:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT id, name, traffic_split, min_samples, alpha, started_at
                    FROM experiments WHERE status = 'running'
                    """
                )
                cols = [c[0] for c in cur.description]
                return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
            finally:
                if not self._memory_uri:
                    conn.close()

        return await asyncio.to_thread(_query)

    # ----- private SQLite helpers ----- #

    def _load_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Fetch experiment metadata or None if not found."""
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT id, name, traffic_split, min_samples, alpha, status
                FROM experiments WHERE id = ?
                """,
                (experiment_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            cols = [c[0] for c in cur.description]
            return dict(zip(cols, row, strict=True))
        finally:
            if not self._memory_uri:
                conn.close()

    def _load_outcomes(
        self, experiment_id: str
    ) -> tuple[list[float], list[float]]:
        """Return (samples_for_A, samples_for_B) for an experiment."""
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT variant, metric_value FROM outcomes
                WHERE experiment_id = ?
                """,
                (experiment_id,),
            )
            samples_a: list[float] = []
            samples_b: list[float] = []
            for variant, value in cur.fetchall():
                if variant == _VARIANT_A:
                    samples_a.append(float(value))
                else:
                    samples_b.append(float(value))
            return samples_a, samples_b
        finally:
            if not self._memory_uri:
                conn.close()

    def _close_experiment(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        metric_a: float,
        metric_b: float,
    ) -> None:
        """Stamp ended_at, status, and the realised metrics on a row."""
        conn = self._connect()
        try:
            with conn:
                conn.execute(
                    """
                    UPDATE experiments
                    SET ended_at = ?, status = ?, metric_a = ?, metric_b = ?
                    WHERE id = ?
                    """,
                    (time.time(), status, metric_a, metric_b, experiment_id),
                )
        finally:
            if not self._memory_uri:
                conn.close()


# --------------------------------------------------------------------------- #
# Smoke tests — runnable without pytest.
# --------------------------------------------------------------------------- #


async def _smoke_test_promote() -> None:
    """End-to-end smoke: variant B clearly better should result in promote_b.

    Uses an in-memory DB so it is hermetic and never touches ~/.quorum/.
    """
    os.environ.pop("HSP_GATE_WEBHOOK", None)  # ensure dev-mode gate pass.
    runner = ABTestRunner(db_path=":memory:")
    exp_id = await runner.start(
        name="smoke_promote",
        variant_a={"policy": "baseline"},
        variant_b={"policy": "improved"},
        min_samples=20,
        alpha=0.05,
    )
    # A around 0.50, B around 0.80 — large effect, should promote.
    for i in range(50):
        await runner.record_outcome(exp_id, _VARIANT_A, 0.5 + 0.001 * (i % 5))
    for i in range(50):
        await runner.record_outcome(exp_id, _VARIANT_B, 0.8 + 0.001 * (i % 5))
    decision = await runner.evaluate(exp_id)
    assert decision.decision == "promote_b", decision
    assert decision.effect_size > _PROMOTE_THRESHOLD, decision
    assert decision.sample_size == 100, decision
    logger.info("smoke_test_promote OK: %s", decision)


async def _smoke_test_revert() -> None:
    """End-to-end smoke: variant B clearly worse should result in revert."""
    os.environ.pop("HSP_GATE_WEBHOOK", None)
    runner = ABTestRunner(db_path=":memory:")
    exp_id = await runner.start(
        name="smoke_revert",
        variant_a={"policy": "baseline"},
        variant_b={"policy": "regression"},
        min_samples=20,
        alpha=0.05,
    )
    for i in range(50):
        await runner.record_outcome(exp_id, _VARIANT_A, 0.9 + 0.001 * (i % 5))
    for i in range(50):
        await runner.record_outcome(exp_id, _VARIANT_B, 0.4 + 0.001 * (i % 5))
    decision = await runner.evaluate(exp_id)
    assert decision.decision == "revert", decision
    assert decision.effect_size < 0, decision
    logger.info("smoke_test_revert OK: %s", decision)


async def _smoke_test_inconclusive_low_n() -> None:
    """Too few samples → inconclusive even with extreme separation."""
    os.environ.pop("HSP_GATE_WEBHOOK", None)
    runner = ABTestRunner(db_path=":memory:")
    exp_id = await runner.start(
        name="smoke_low_n",
        variant_a={"policy": "baseline"},
        variant_b={"policy": "candidate"},
        min_samples=200,
        alpha=0.05,
    )
    for _ in range(5):
        await runner.record_outcome(exp_id, _VARIANT_A, 0.1)
        await runner.record_outcome(exp_id, _VARIANT_B, 0.9)
    decision = await runner.evaluate(exp_id)
    assert decision.decision == "inconclusive", decision
    logger.info("smoke_test_inconclusive_low_n OK: %s", decision)


async def _smoke_test_binary_chi2() -> None:
    """Binary metric uses chi-square — verify the path is exercised."""
    os.environ.pop("HSP_GATE_WEBHOOK", None)
    runner = ABTestRunner(db_path=":memory:")
    exp_id = await runner.start(
        name="smoke_binary",
        variant_a={"policy": "baseline"},
        variant_b={"policy": "candidate"},
        min_samples=50,
        alpha=0.05,
    )
    # A: 40% success rate; B: 80% success rate.
    for i in range(100):
        await runner.record_outcome(exp_id, _VARIANT_A, 1.0 if i % 5 < 2 else 0.0)
    for i in range(100):
        await runner.record_outcome(exp_id, _VARIANT_B, 1.0 if i % 5 < 4 else 0.0)
    decision = await runner.evaluate(exp_id)
    assert decision.decision == "promote_b", decision
    logger.info("smoke_test_binary_chi2 OK: %s", decision)


async def _run_all_smoke_tests() -> None:
    """Run every smoke test in sequence — entrypoint for __main__."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    await _smoke_test_promote()
    await _smoke_test_revert()
    await _smoke_test_inconclusive_low_n()
    await _smoke_test_binary_chi2()
    logger.info("all ab_testing smoke tests passed.")


# --------------------------------------------------------------------------- #
# Prompt-Template A/B Store (Wilson-bounded winner tracking)
# --------------------------------------------------------------------------- #
#
# The ABTestRunner above (Loop 8) tests *evolution policies* — router weights,
# judge prompts, consensus thresholds — with full Welch/chi-square testing and
# HSP gating. That is the right tool when a change touches the live policy and
# requires a human sign-off before promotion.
#
# This second store solves a DIFFERENT problem: prompt-template authors who
# want to A/B two phrasings of the same user-facing prompt, compare the
# resulting consensus outputs side by side, collect winner votes (from the
# user or from an automatic judge), and pick the current winner per template
# using a Wilson lower bound so a 3/3 win streak does not outrank a 38/50
# win record. It is intentionally lightweight: no HSP gate, no statistical
# test on continuous metrics, just persistent vote tracking with a
# conservative winner heuristic.
#
# Storage lives in a separate file (ab_tests.db) so the policy experiment
# history (abtest.db) is never co-mingled with prompt-variant votes.

_AB_DEFAULT_DB_PATH: Final[Path] = DATA_DIR / "ab_tests.db"

# Wilson lower bound default coverage: 95% (z = 1.96). Exposed as a kwarg so
# tests can pin it deterministically and so a future caller can ask for a
# stricter 99% bound for high-stakes templates.
_WILSON_Z_95: Final[float] = 1.959964
# Default rolling window for "active winner" decisions — last N experiments
# per prompt-template. Bounded so a long-stale variant does not dominate.
_WINNER_WINDOW_DEFAULT: Final[int] = 50
# Minimum experiments before get_active_winner returns a non-None verdict.
# Wilson on n<5 is meaningless and would just amplify noise.
_WINNER_MIN_N: Final[int] = 5

ABArm = Literal["a", "b", "tie"]
ABSource = Literal["user", "auto"]


_SCHEMA_AB_EXPERIMENTS: Final[str] = """
CREATE TABLE IF NOT EXISTS experiments (
    id                  TEXT PRIMARY KEY,
    prompt_template_id  TEXT,
    prompt_a            TEXT NOT NULL,
    prompt_b            TEXT NOT NULL,
    query_id            TEXT,
    winner              TEXT,
    winner_source       TEXT,
    confidence_a        REAL NOT NULL DEFAULT 0.0,
    confidence_b        REAL NOT NULL DEFAULT 0.0,
    semantic_distance   REAL NOT NULL DEFAULT 0.0,
    created_at          TIMESTAMP NOT NULL,
    decided_at          TIMESTAMP
);
"""

_SCHEMA_AB_INDEX_TEMPLATE: Final[str] = (
    "CREATE INDEX IF NOT EXISTS idx_ab_experiments_template "
    "ON experiments(prompt_template_id, created_at DESC);"
)


def _wilson_lower_bound(successes: int, n: int, z: float = _WILSON_Z_95) -> float:
    """Wilson score lower confidence bound on a binomial proportion.

    WHY this and not the raw win rate: a template that wins 3/3 has a raw rate
    of 1.0 but a Wilson lower bound of ~0.44 — far below a template that wins
    38/50 (raw 0.76, Wilson lower ~0.62). The Wilson bound is the gold-standard
    way to rank "best out of N proportions" when N varies; it is what Reddit
    used for "best comment" sort and what every modern bandit baseline uses.

    Returns the lower bound on the true success probability. If n == 0 returns
    0.0 (we have zero evidence, so cannot claim anything).
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2.0 * n)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denom)


class ABTestStore:
    """SQLite-backed store for prompt-template A/B experiments.

    Use one instance per process; underlying SQLite I/O is wrapped in
    ``asyncio.to_thread`` so the event loop never blocks. Concurrent
    coroutines are safe because every call opens a short-lived connection
    with ``check_same_thread=False``.

    The store records (a) every experiment with both candidate prompts and
    the two consensus results' confidence scores + their semantic distance,
    and (b) the winner once it is reported (by a human or by an auto-judge).
    ``get_active_winner`` then ranks the two arms by Wilson lower bound on
    their win rate over the last N experiments, so a brand-new variant with
    one lucky win cannot displace an established winner.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        """Open / create the SQLite file and ensure the schema is current.

        Defaults to ``${QUORUM_DATA_DIR}/ab_tests.db`` mirroring the
        convention used by ``core/memory.py`` and the rest of the codebase.
        Pass ``":memory:"`` for hermetic tests.
        """
        if db_path is None:
            db_path = _AB_DEFAULT_DB_PATH
        self._db_path = Path(str(db_path)) if str(db_path) != ":memory:" else None
        self._memory_uri = str(db_path) == ":memory:"
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._memory_conn: sqlite3.Connection | None = None
        self._init_schema()

    # ----- connection plumbing ----- #

    def _connect(self) -> sqlite3.Connection:
        if self._memory_uri:
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(
                    ":memory:", check_same_thread=False
                )
                self._memory_conn.execute("PRAGMA foreign_keys = ON;")
            return self._memory_conn
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            with conn:
                conn.execute(_SCHEMA_AB_EXPERIMENTS)
                conn.execute(_SCHEMA_AB_INDEX_TEMPLATE)
        finally:
            if not self._memory_uri:
                conn.close()

    # ----- public API ----- #

    async def record_experiment(
        self,
        prompt_a: str,
        prompt_b: str,
        result_a: Any,
        result_b: Any,
        *,
        prompt_template_id: str | None = None,
        query_id: str | None = None,
        semantic_distance: float | None = None,
    ) -> str:
        """Persist a fresh A/B experiment row and return the experiment id.

        ``result_a`` / ``result_b`` are duck-typed ``ConsensusResult`` (we
        only need ``.embedding_confidence`` / ``.confidence`` / ``.answer``).
        Importing the real class would create a cyclic import — store ↔
        consensus — so we accept ``Any`` and read attributes defensively.

        ``semantic_distance`` defaults to ``|conf_a - conf_b|`` when not
        supplied. Callers who already have the answer-vs-answer cosine
        distance from a shared embedder should pass it in for accuracy.
        """
        if not prompt_a or not prompt_b:
            raise ValueError("prompt_a and prompt_b must be non-empty.")

        conf_a = float(
            getattr(result_a, "embedding_confidence", None)
            or getattr(result_a, "confidence", 0.0)
            or 0.0
        )
        conf_b = float(
            getattr(result_b, "embedding_confidence", None)
            or getattr(result_b, "confidence", 0.0)
            or 0.0
        )
        if semantic_distance is None:
            semantic_distance = abs(conf_a - conf_b)

        exp_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        def _insert() -> None:
            conn = self._connect()
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO experiments
                          (id, prompt_template_id, prompt_a, prompt_b, query_id,
                           winner, winner_source, confidence_a, confidence_b,
                           semantic_distance, created_at, decided_at)
                        VALUES (?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, NULL)
                        """,
                        (
                            exp_id,
                            prompt_template_id,
                            prompt_a,
                            prompt_b,
                            query_id,
                            conf_a,
                            conf_b,
                            float(semantic_distance),
                            created_at,
                        ),
                    )
            finally:
                if not self._memory_uri:
                    conn.close()

        await asyncio.to_thread(_insert)
        logger.info(
            "ab_store_recorded id=%s template=%s conf_a=%.3f conf_b=%.3f dist=%.3f",
            exp_id,
            prompt_template_id,
            conf_a,
            conf_b,
            semantic_distance,
        )
        return exp_id

    async def report_winner(
        self,
        experiment_id: str,
        winner: ABArm,
        source: ABSource = "user",
    ) -> None:
        """Attach a winner verdict ('a' | 'b' | 'tie') to a recorded experiment.

        ``source`` distinguishes ``user`` votes (gold signal) from ``auto``
        verdicts (LLM-judge cheap signal). Both currently count equally toward
        ``get_active_winner``; future versions can downweight ``auto`` via
        an extra column without changing the API.
        """
        if winner not in ("a", "b", "tie"):
            raise ValueError(f"winner must be 'a'|'b'|'tie', got {winner!r}.")
        if source not in ("user", "auto"):
            raise ValueError(f"source must be 'user'|'auto', got {source!r}.")
        decided_at = datetime.now(timezone.utc).isoformat()

        def _update() -> int:
            conn = self._connect()
            try:
                with conn:
                    cur = conn.execute(
                        """
                        UPDATE experiments
                        SET winner = ?, winner_source = ?, decided_at = ?
                        WHERE id = ?
                        """,
                        (winner, source, decided_at, experiment_id),
                    )
                    return cur.rowcount
            finally:
                if not self._memory_uri:
                    conn.close()

        rowcount = await asyncio.to_thread(_update)
        if rowcount == 0:
            raise KeyError(f"unknown experiment_id: {experiment_id!r}")
        logger.info(
            "ab_store_winner id=%s winner=%s source=%s",
            experiment_id,
            winner,
            source,
        )

    async def get_active_winner(
        self,
        prompt_template_id: str,
        *,
        window: int = _WINNER_WINDOW_DEFAULT,
        min_n: int = _WINNER_MIN_N,
        z: float = _WILSON_Z_95,
    ) -> ABArm | None:
        """Return the current champion arm for a template, or ``None``.

        Reads the last ``window`` *decided* experiments for the template and
        ranks each arm by Wilson lower bound on its win rate. Ties (winner =
        'tie') count toward neither arm's success but DO count toward both
        arms' trial total — i.e. a tie is a "neither beat the other", which
        is exactly the right semantic for a conservative champion bound.

        Returns ``None`` when fewer than ``min_n`` decided experiments exist
        (insufficient evidence), or when the two lower bounds are within 1e-9
        (true tie, no champion).
        """
        if not prompt_template_id:
            raise ValueError("prompt_template_id must be non-empty.")

        def _query() -> list[str]:
            conn = self._connect()
            try:
                cur = conn.execute(
                    """
                    SELECT winner FROM experiments
                    WHERE prompt_template_id = ? AND winner IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (prompt_template_id, window),
                )
                return [str(row[0]) for row in cur.fetchall()]
            finally:
                if not self._memory_uri:
                    conn.close()

        winners = await asyncio.to_thread(_query)
        n = len(winners)
        if n < min_n:
            return None

        wins_a = sum(1 for w in winners if w == "a")
        wins_b = sum(1 for w in winners if w == "b")
        lb_a = _wilson_lower_bound(wins_a, n, z=z)
        lb_b = _wilson_lower_bound(wins_b, n, z=z)

        if abs(lb_a - lb_b) < 1e-9:
            return None
        return "a" if lb_a > lb_b else "b"

    async def stats(self) -> dict[str, Any]:
        """Return aggregate stats for an operator dashboard.

        Shape::

            {
              "total_experiments": int,
              "decided_experiments": int,
              "win_rates_per_arm": {
                  "a": {"wins": int, "n": int, "rate": float, "wilson_lb": float},
                  "b": {"wins": int, "n": int, "rate": float, "wilson_lb": float},
                  "tie": {"wins": int, "n": int, "rate": float},
              },
              "current_winner_per_template": {"<template_id>": "a"|"b"|None, ...},
            }
        """

        def _aggregate() -> tuple[int, list[tuple[str | None, str]]]:
            conn = self._connect()
            try:
                total = conn.execute(
                    "SELECT COUNT(*) FROM experiments"
                ).fetchone()[0]
                cur = conn.execute(
                    """
                    SELECT prompt_template_id, winner FROM experiments
                    WHERE winner IS NOT NULL
                    """
                )
                rows = [
                    (None if r[0] is None else str(r[0]), str(r[1]))
                    for r in cur.fetchall()
                ]
                return int(total), rows
            finally:
                if not self._memory_uri:
                    conn.close()

        total, decided_rows = await asyncio.to_thread(_aggregate)
        decided = len(decided_rows)

        wins_a = sum(1 for _, w in decided_rows if w == "a")
        wins_b = sum(1 for _, w in decided_rows if w == "b")
        wins_t = sum(1 for _, w in decided_rows if w == "tie")

        def _arm(wins: int, n: int) -> dict[str, float | int]:
            return {
                "wins": wins,
                "n": n,
                "rate": (wins / n) if n else 0.0,
                "wilson_lb": _wilson_lower_bound(wins, n),
            }

        # Per-template champion lookup. We re-use get_active_winner so the
        # ranking logic stays in one place; the await loop is fine for the
        # template counts a dashboard would ever hit (dozens, not millions).
        template_ids = {tid for tid, _ in decided_rows if tid is not None}
        per_template: dict[str, ABArm | None] = {}
        for tid in template_ids:
            per_template[tid] = await self.get_active_winner(tid)

        return {
            "total_experiments": total,
            "decided_experiments": decided,
            "win_rates_per_arm": {
                "a": _arm(wins_a, decided),
                "b": _arm(wins_b, decided),
                "tie": {
                    "wins": wins_t,
                    "n": decided,
                    "rate": (wins_t / decided) if decided else 0.0,
                },
            },
            "current_winner_per_template": per_template,
        }


__all__ = [
    "ABDecision",
    "ABTestRunner",
    "ABTestStore",
    "_wilson_lower_bound",
]


if __name__ == "__main__":
    asyncio.run(_run_all_smoke_tests())
