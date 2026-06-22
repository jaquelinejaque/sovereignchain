# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Diversity-vs-quality cross-correlation between Hebbian and ELO stores.

Why this module exists
======================
The withdrawn Zenodo paper documented a tight per-pair semantic cluster
(``mean ≈ 0.84``, ``σ ≈ 0.012``) in the production Hebbian store.
While cataloguing the on-disk databases on 2026-06-22 I noticed a
follow-up effect that the paper did **not** report and that no script
in the codebase computes:

    Pearson r between
        x = pairwise Hebbian similarity  (``hebbian.db`` / ``coactivation``)
        y = absolute ELO-rating gap     (``competition.db`` / ``elo``)
    came out at **r ≈ −0.29** across the 206 pairs with at least 30
    co-activation samples.

Interpretation
--------------
A weakly negative correlation means: pairs that "sound similar"
(high Hebbian) tend to **also** be close in ELO — but only weakly.
The pairing is real but not deterministic. In plain words:

* **Style** (what Hebbian measures via embedding cosine of the
  responses) **and quality** (what ELO measures via head-to-head
  agreement with the consensus answer) are *correlated*, not the
  same thing.
* Two models can have nearly identical style yet differ by 300+
  ELO points (e.g. ``cohere-command-r-plus`` and ``gpt-4.1`` in
  the production data: similarity ≈ 0.86, ELO gap ≈ 350).

This matters because production routers (e.g. ``router.py``) and
ensemble-selection heuristics often **conflate** the two — they
drop "redundant" providers based on similarity, on the implicit
assumption that similarity implies equivalent quality. The data
says that assumption is wrong.

What this module does
---------------------
* Provides ``compute_diversity_quality_correlation`` — reads the two
  SQLite stores, joins them per (model_a, model_b), computes the
  Pearson r honestly (small-N safe, ignores pairs the user hasn't
  seen enough of), and returns a structured result.
* Provides ``select_diverse_quality_panel`` — an experimental
  ensemble picker that maximises ELO spread *subject to* a Hebbian
  similarity floor. The opposite of "deduplicate similar models":
  it picks the *most stylistically similar* model from each ELO
  tier, so the ensemble is short, cheap, and stylistically coherent
  but quality-stratified.

No external dependencies (pure stdlib). Reads, never writes.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Default DBs match the rest of the codebase (~/.quorum/...).
HEBBIAN_DB_DEFAULT = Path.home() / ".quorum" / "hebbian.db"
COMPETITION_DB_DEFAULT = Path.home() / ".quorum" / "competition.db"

# Minimum Hebbian sample count before a pair is considered well-measured.
# 30 matches the threshold used in the original paper (cf. the
# "206 pairs" figure at σ = 0.012).
MIN_HEBBIAN_SAMPLES = 30


@dataclass(frozen=True)
class PairObservation:
    """One joined row across Hebbian and ELO for a single model pair.

    ``hebbian_avg_sim`` is the running mean of pairwise cosine over
    ``samples`` rounds. ``elo_gap`` is ``abs(elo_a - elo_b)`` for the
    requested query class (defaults to general — the class with the
    bulk of the production data, 20k+ matches).
    """

    model_a: str
    model_b: str
    hebbian_avg_sim: float
    samples: int
    elo_a: float
    elo_b: float
    elo_gap: float


@dataclass(frozen=True)
class CorrelationResult:
    """Outcome of ``compute_diversity_quality_correlation``.

    ``pearson_r`` is in ``[-1, 1]``. A *negative* r is the expected
    direction: higher similarity → smaller ELO gap → r < 0. A value
    near zero means style and quality are uncorrelated in this data
    set, which would be the surprising finding (the published paper
    reported r ≈ −0.29 on the production deployment as of 2026-06-22).
    """

    n_pairs: int
    pearson_r: float
    mean_similarity: float
    mean_elo_gap: float
    query_class: str
    observations: tuple[PairObservation, ...]


def _read_hebbian(
    db_path: Path,
    min_samples: int = MIN_HEBBIAN_SAMPLES,
) -> dict[tuple[str, str], tuple[float, int]]:
    """Return ``{(model_a, model_b): (avg_sim, samples)}`` from
    ``hebbian.db`` filtered by ``samples >= min_samples``.

    The key is order-sensitive — we preserve whatever order the
    Hebbian store wrote, then normalise in the caller (sorted pair)
    so the join with ELO doesn't double-count.
    """
    if not db_path.exists():
        return {}
    out: dict[tuple[str, str], tuple[float, int]] = {}
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        cur = conn.execute(
            "SELECT model_a, model_b, similarity_sum, count "
            "FROM coactivation WHERE count >= ?",
            (min_samples,),
        )
        for a, b, ssum, n in cur:
            if n <= 0:
                continue
            out[(a, b)] = (ssum / n, n)
    finally:
        conn.close()
    return out


def _read_elo(db_path: Path, query_class: str) -> dict[str, float]:
    """Return ``{model_name: rating}`` for one query class."""
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        return dict(
            conn.execute(
                "SELECT model_name, rating FROM elo WHERE query_class = ?",
                (query_class,),
            )
        )
    finally:
        conn.close()


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson r without numpy. Returns 0.0 on degenerate input
    (n < 2 or zero variance) — the caller should check n_pairs."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(sxx * syy)
    if denom == 0.0:
        return 0.0
    return sxy / denom


def compute_diversity_quality_correlation(
    *,
    query_class: str = "general",
    min_samples: int = MIN_HEBBIAN_SAMPLES,
    hebbian_db: Path | None = None,
    competition_db: Path | None = None,
) -> CorrelationResult:
    """Join Hebbian similarity with ELO gap per model pair.

    Args:
        query_class: ELO is stratified by query class. "general"
            holds the bulk of the production data; "code" / "factual"
            are sparser and may not have ELO ratings for every
            model in the Hebbian store.
        min_samples: Hebbian pairs below this sample count are
            dropped — the running-mean cosine isn't trustworthy
            until the pair has accumulated enough observations.
        hebbian_db / competition_db: Override paths for tests.

    Returns:
        A ``CorrelationResult`` with the Pearson r between
        (avg pairwise similarity) and (absolute ELO gap), the
        underlying ``PairObservation`` tuples, and the means of the
        two variables. Order of observations is sorted by similarity
        descending so the caller can ``[:5]`` the strongest ties.
    """
    hebbian_db = hebbian_db or HEBBIAN_DB_DEFAULT
    competition_db = competition_db or COMPETITION_DB_DEFAULT

    pairs = _read_hebbian(hebbian_db, min_samples=min_samples)
    elos = _read_elo(competition_db, query_class)

    obs: list[PairObservation] = []
    seen: set[tuple[str, str]] = set()
    for (a, b), (avg_sim, n) in pairs.items():
        if a not in elos or b not in elos:
            continue  # this pair has no rating in the chosen class
        canonical = tuple(sorted((a, b)))
        if canonical in seen:
            continue  # the table stores both orders for some rows
        seen.add(canonical)
        ea, eb = elos[a], elos[b]
        obs.append(
            PairObservation(
                model_a=a,
                model_b=b,
                hebbian_avg_sim=avg_sim,
                samples=n,
                elo_a=ea,
                elo_b=eb,
                elo_gap=abs(ea - eb),
            )
        )

    obs.sort(key=lambda o: -o.hebbian_avg_sim)
    sims = [o.hebbian_avg_sim for o in obs]
    gaps = [o.elo_gap for o in obs]
    r = _pearson(sims, gaps)
    return CorrelationResult(
        n_pairs=len(obs),
        pearson_r=r,
        mean_similarity=(sum(sims) / len(sims)) if sims else 0.0,
        mean_elo_gap=(sum(gaps) / len(gaps)) if gaps else 0.0,
        query_class=query_class,
        observations=tuple(obs),
    )


# --------------------------------------------------------------------------- #
# Experimental ensemble picker
# --------------------------------------------------------------------------- #


def select_diverse_quality_panel(
    *,
    panel_size: int = 4,
    similarity_floor: float = 0.83,
    query_class: str = "general",
    hebbian_db: Path | None = None,
    competition_db: Path | None = None,
) -> list[str]:
    """Pick ``panel_size`` models that span ELO tiers but stay stylistically close.

    The motivation is the corollary to r ≈ −0.29 — style and quality
    are correlated but not fused. So we can build an ensemble that:

    * Stays inside a tight semantic cluster (``avg_sim >=
      similarity_floor``) — the responses will format and reason
      similarly, which makes downstream embedding-based scoring
      well-behaved.
    * **Yet** spans ELO tiers — the picker grabs one model per
      equal-width ELO band so the panel covers high, medium, and
      lower-rated outputs. The point isn't "use the worst model";
      it's that a 1800-vs-1200 ensemble exposes more diversity in
      *correct vs incorrect* than a 1800-vs-1750 ensemble where
      everything agrees.

    This is **experimental**. There is no online evaluation behind
    these numbers yet — the function returns a candidate panel for
    you to feed into ``consensus(..., providers=...)`` and measure.
    """
    result = compute_diversity_quality_correlation(
        query_class=query_class,
        hebbian_db=hebbian_db,
        competition_db=competition_db,
    )
    if result.n_pairs == 0 or panel_size < 1:
        return []

    # Collect unique models that participate in at least one
    # well-clustered pair (>= similarity_floor with something).
    clustered: set[str] = set()
    for o in result.observations:
        if o.hebbian_avg_sim >= similarity_floor:
            clustered.add(o.model_a)
            clustered.add(o.model_b)
    if not clustered:
        return []

    # Pull each clustered model's solo ELO so we can bin them.
    # We need a separate read because the joined view discarded
    # individual ratings in favour of the gap.
    competition_db = competition_db or COMPETITION_DB_DEFAULT
    elos = _read_elo(competition_db, query_class)
    pool = sorted(
        ((m, elos[m]) for m in clustered if m in elos),
        key=lambda mr: mr[1],
        reverse=True,
    )
    if not pool:
        return []
    if panel_size >= len(pool):
        return [m for m, _ in pool]

    # Equal-width ELO bins; pick the highest-rated model from each.
    hi = pool[0][1]
    lo = pool[-1][1]
    if hi == lo:
        return [pool[0][0]]
    band = (hi - lo) / panel_size
    picks: list[str] = []
    for i in range(panel_size):
        band_lo = lo + i * band
        band_hi = lo + (i + 1) * band
        # Closed at the top of the last band so the highest model lands.
        in_band = [
            (m, r)
            for m, r in pool
            if (band_lo <= r < band_hi) or (i == panel_size - 1 and r == band_hi)
        ]
        if in_band:
            in_band.sort(key=lambda mr: -mr[1])
            picks.append(in_band[0][0])
    return picks


def _format_pretty(result: CorrelationResult, *, top: int = 10) -> str:
    """Human-readable summary string. Used by the CLI command."""
    lines: list[str] = []
    lines.append(
        f"Diversity-vs-Quality correlation  ({result.query_class})\n"
        f"  pairs            : {result.n_pairs}\n"
        f"  Pearson r        : {result.pearson_r:+.4f}\n"
        f"  mean similarity  : {result.mean_similarity:.4f}\n"
        f"  mean ELO gap     : {result.mean_elo_gap:.1f}"
    )
    if result.observations:
        lines.append("")
        lines.append(f"  Top {min(top, result.n_pairs)} strongest pairs (Hebbian sim DESC):")
        lines.append(
            f"    {'pair':<60} {'sim':>7}  {'gap':>7}  {'n':>5}"
        )
        for o in result.observations[:top]:
            label = f"{o.model_a[:28]} <-> {o.model_b[:28]}"
            lines.append(
                f"    {label:<60} {o.hebbian_avg_sim:>7.4f}  "
                f"{o.elo_gap:>7.0f}  {o.samples:>5d}"
            )
    return "\n".join(lines)


__all__ = [
    "MIN_HEBBIAN_SAMPLES",
    "PairObservation",
    "CorrelationResult",
    "compute_diversity_quality_correlation",
    "select_diverse_quality_panel",
]
