# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for ``evolution.diversity``.

The math being tested is Pearson r between Hebbian similarity and ELO
gap. To keep these unit tests deterministic (and decoupled from the
real ``~/.quorum`` databases), we materialise two tiny SQLite stores
in ``tmp_path`` and drive ``compute_diversity_quality_correlation``
against them.

Test scenarios:

1. **Empty stores.** Function must return ``n_pairs=0`` and not crash.
2. **Perfect anti-correlation.** Hand-crafted pairs where higher
   similarity strictly implies smaller ELO gap → r should be very
   close to −1.
3. **Sample threshold honoured.** A pair below ``min_samples`` must
   be excluded; raising the threshold prunes it from the result.
4. **Missing ELO entries.** Pairs where one model has no rating in
   the queried class are silently dropped, not crashed.
5. **Ensemble picker.** Given a clustered group spanning a wide ELO
   range, ``select_diverse_quality_panel`` returns one model per
   ELO tier and respects the ``similarity_floor``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from quorum.evolution.diversity import (
    MIN_HEBBIAN_SAMPLES,
    compute_diversity_quality_correlation,
    select_diverse_quality_panel,
)


def _make_hebbian(db_path: Path, rows: list[tuple[str, str, float, int]]) -> None:
    """Materialise a ``coactivation`` table that matches the real schema.

    Each row is ``(model_a, model_b, avg_sim, samples)``. We store
    ``avg_sim * samples`` as ``similarity_sum`` so the production
    division gives the intended ``avg_sim``.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE coactivation (
                model_a TEXT, model_b TEXT,
                similarity_sum REAL, count INTEGER,
                last_updated REAL DEFAULT 0,
                PRIMARY KEY (model_a, model_b)
            )"""
        )
        conn.executemany(
            "INSERT INTO coactivation VALUES (?, ?, ?, ?, 0)",
            [(a, b, sim * n, n) for (a, b, sim, n) in rows],
        )
        conn.commit()
    finally:
        conn.close()


def _make_competition(db_path: Path, ratings: dict[str, float], cls: str = "general") -> None:
    """Materialise the ``elo`` table that matches the production schema."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """CREATE TABLE elo (
                model_name TEXT, query_class TEXT, rating REAL,
                games INTEGER DEFAULT 0, updated_at REAL DEFAULT 0,
                PRIMARY KEY (model_name, query_class)
            )"""
        )
        conn.executemany(
            "INSERT INTO elo VALUES (?, ?, ?, 100, 0)",
            [(m, cls, r) for m, r in ratings.items()],
        )
        conn.commit()
    finally:
        conn.close()


def test_empty_stores_return_zero_pairs(tmp_path):
    """Both DBs missing → graceful 0-pair result, no exception."""
    h = tmp_path / "h.db"
    c = tmp_path / "c.db"
    result = compute_diversity_quality_correlation(
        hebbian_db=h, competition_db=c
    )
    assert result.n_pairs == 0
    assert result.pearson_r == 0.0
    assert result.observations == ()


def test_strong_anti_correlation(tmp_path):
    """Hand-crafted strictly monotone data → r close to -1.

    Six pairs across four models, designed so that sim_rank ==
    inverse(gap_rank). Pearson r on perfect monotone data is
    -1 (Pearson is invariant under linear scaling and these are
    linearly related by construction).
    """
    h = tmp_path / "h.db"
    c = tmp_path / "c.db"
    # 4 models, 6 pairs; rate them so gap = 0, 100, 200, ... 500.
    _make_competition(
        c,
        {"m1": 1500.0, "m2": 1500.0, "m3": 1600.0, "m4": 2000.0},
    )
    _make_hebbian(
        h,
        [
            # (a, b, sim, n)
            ("m1", "m2", 0.95, 50),  # gap = 0   → highest sim
            ("m1", "m3", 0.90, 50),  # gap = 100
            ("m2", "m3", 0.90, 50),  # gap = 100
            ("m1", "m4", 0.80, 50),  # gap = 500 → lowest sim
            ("m2", "m4", 0.80, 50),  # gap = 500
            ("m3", "m4", 0.85, 50),  # gap = 400
        ],
    )
    result = compute_diversity_quality_correlation(
        hebbian_db=h, competition_db=c
    )
    assert result.n_pairs == 6
    # Strong negative — higher similarity → smaller gap.
    assert result.pearson_r < -0.9, f"expected r < -0.9, got {result.pearson_r}"
    # Sanity on bookkeeping.
    assert result.observations[0].hebbian_avg_sim == pytest.approx(0.95)
    assert result.observations[0].elo_gap == pytest.approx(0.0)


def test_sample_threshold_filters_low_n(tmp_path):
    """A pair with samples below MIN_HEBBIAN_SAMPLES is excluded.

    We make one pair with samples=5 (well below default 30) and
    one with samples=50. Default threshold drops the small one;
    lowering the threshold restores it.
    """
    h = tmp_path / "h.db"
    c = tmp_path / "c.db"
    _make_competition(c, {"a": 1500.0, "b": 1700.0, "x": 1500.0, "y": 1900.0})
    _make_hebbian(
        h,
        [
            ("a", "b", 0.90, 50),   # well-measured
            ("x", "y", 0.85, 5),    # noisy
        ],
    )
    default = compute_diversity_quality_correlation(
        hebbian_db=h, competition_db=c
    )
    assert default.n_pairs == 1  # noisy pair excluded

    relaxed = compute_diversity_quality_correlation(
        hebbian_db=h, competition_db=c, min_samples=1
    )
    assert relaxed.n_pairs == 2


def test_pair_without_elo_is_dropped(tmp_path):
    """A model in Hebbian but missing in ELO must not crash —
    the joined view just skips that pair."""
    h = tmp_path / "h.db"
    c = tmp_path / "c.db"
    _make_competition(c, {"known_a": 1500.0, "known_b": 1700.0})
    _make_hebbian(
        h,
        [
            ("known_a", "known_b", 0.90, 50),
            ("known_a", "unknown_model", 0.85, 50),  # unknown has no ELO
        ],
    )
    result = compute_diversity_quality_correlation(
        hebbian_db=h, competition_db=c
    )
    assert result.n_pairs == 1
    pair = result.observations[0]
    assert {pair.model_a, pair.model_b} == {"known_a", "known_b"}


def test_panel_picker_returns_one_per_band(tmp_path):
    """With 4 models in a tight cluster spanning 1400-1900 ELO and
    panel_size=4, the picker should return all 4 (one per equal-width
    ELO band, top of each band)."""
    h = tmp_path / "h.db"
    c = tmp_path / "c.db"
    _make_competition(
        c,
        {"low": 1400.0, "mid_lo": 1550.0, "mid_hi": 1700.0, "top": 1900.0},
    )
    _make_hebbian(
        h,
        [
            # All in the same cluster (sim >= 0.83) — picker should
            # be allowed to span ELO tiers freely.
            ("low", "mid_lo", 0.85, 50),
            ("low", "mid_hi", 0.84, 50),
            ("low", "top", 0.83, 50),
            ("mid_lo", "mid_hi", 0.86, 50),
            ("mid_lo", "top", 0.84, 50),
            ("mid_hi", "top", 0.85, 50),
        ],
    )
    panel = select_diverse_quality_panel(
        panel_size=4,
        similarity_floor=0.83,
        hebbian_db=h,
        competition_db=c,
    )
    assert set(panel) == {"low", "mid_lo", "mid_hi", "top"}


def test_panel_picker_respects_similarity_floor(tmp_path):
    """A floor above every pair's similarity yields an empty panel."""
    h = tmp_path / "h.db"
    c = tmp_path / "c.db"
    _make_competition(c, {"a": 1500.0, "b": 1700.0})
    _make_hebbian(h, [("a", "b", 0.80, 50)])
    panel = select_diverse_quality_panel(
        panel_size=2,
        similarity_floor=0.95,  # nothing passes
        hebbian_db=h,
        competition_db=c,
    )
    assert panel == []
