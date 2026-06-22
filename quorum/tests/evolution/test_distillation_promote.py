# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for ``DistillationPipeline._run_benchmark`` and
``promote_checkpoint``.

These are the last two steps in the distillation pipeline and the
ones whose misuse breaks the system: the benchmark turns a checkpoint
into a numeric score, and promote_checkpoint is the function that
swaps the model production queries hit.

Contracts under test:

``_run_benchmark``
    * No eval set → return a blocking (all-zero) BenchmarkResult so the
      "samples_evaluated == 0" guard in promote_checkpoint trips.
    * Eval set present + sidecar JSON present → parse the sidecar.
    * Eval set present + sidecar absent → return a blocking result,
      logging that the evaluator must run.

``promote_checkpoint``
    * samples_evaluated == 0 → reject (defence against an empty benchmark
      passing the gate).
    * No baseline (first-ever promotion) → accept if HSP gate allows,
      regardless of the absolute score (the HSP gate is the sole
      safeguard at bootstrap).
    * Baseline accuracy higher → reject with regression message.
    * Baseline safety lower than candidate → accept (safety improved).
    * Baseline safety higher → reject (safety regressed).
    * Successful promotion writes ``current.json`` with the expected
      fields.

The HSP gate is bypassed via ``HSP_GATE_DEV_MODE=1`` for these tests —
the gate's own coverage lives in tests/hsp/.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quorum.evolution import distillation
from quorum.evolution.distillation import (
    BenchmarkResult,
    DistillationPipeline,
)


@pytest.fixture(autouse=True)
def _hsp_dev_mode(monkeypatch):
    """Unlock the HSP gate for these tests; we cover the gate elsewhere."""
    monkeypatch.setenv("HSP_GATE_DEV_MODE", "1")
    yield


def _pipeline(tmp_path: Path, *, min_improvement: float = 0.0) -> DistillationPipeline:
    return DistillationPipeline(
        log_path=tmp_path / "queries.jsonl",
        eval_set_path=tmp_path / "eval_set.jsonl",
        artifacts_dir=tmp_path / "artifacts",
        min_improvement=min_improvement,
    )


# --------------------------------------------------------------------------- #
# _run_benchmark                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_run_benchmark_no_eval_set_returns_blocking_zero(tmp_path):
    """Missing eval set → all-zero result so promote_checkpoint trips its
    samples_evaluated==0 guard.  The HSP gate is *not* the only line of
    defence: an empty benchmark must never look like a successful one."""
    pipe = _pipeline(tmp_path)
    assert not pipe.eval_set_path.exists()
    result = await pipe._run_benchmark("v1", ckpt_dir=None)
    assert result.samples_evaluated == 0
    assert result.accuracy == 0.0
    assert result.safety_score == 0.0


@pytest.mark.asyncio
async def test_run_benchmark_reads_existing_sidecar(tmp_path):
    """When the eval_runner already wrote ``bench-<version>.json``,
    _run_benchmark must parse it instead of treating it as missing."""
    pipe = _pipeline(tmp_path)
    pipe.eval_set_path.write_text("{}\n", encoding="utf-8")

    sidecar = pipe.artifacts_dir / "bench-v9.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({
        "accuracy": 0.82,
        "safety_score": 0.95,
        "avg_latency_ms": 180.5,
        "samples_evaluated": 50,
        "extra": {"class_factual_mean": 0.91, "ignored_string": "x"},
    }), encoding="utf-8")

    result = await pipe._run_benchmark("v9", ckpt_dir=None)
    assert result.samples_evaluated == 50
    assert result.accuracy == pytest.approx(0.82)
    assert result.safety_score == pytest.approx(0.95)
    assert result.avg_latency_ms == pytest.approx(180.5)
    # extras: numeric kept, non-numeric dropped (the int/float guard).
    assert result.extra == {"class_factual_mean": 0.91}


@pytest.mark.asyncio
async def test_run_benchmark_missing_sidecar_returns_blocking(tmp_path):
    """Eval set exists but no sidecar yet → blocking result."""
    pipe = _pipeline(tmp_path)
    pipe.eval_set_path.write_text("{}\n", encoding="utf-8")

    result = await pipe._run_benchmark("v-no-sidecar", ckpt_dir=tmp_path / "ckpt")
    assert result.samples_evaluated == 0


# --------------------------------------------------------------------------- #
# promote_checkpoint                                                          #
# --------------------------------------------------------------------------- #


def _bench(acc: float, safety: float, samples: int = 50) -> BenchmarkResult:
    return BenchmarkResult(
        version="v-bench",
        accuracy=acc,
        safety_score=safety,
        avg_latency_ms=100.0,
        samples_evaluated=samples,
    )


async def _seed_current(pipe: DistillationPipeline, acc: float, safety: float) -> None:
    """Write a fake ``current.json`` so _load_baseline_benchmark returns it."""
    pipe.artifacts_dir.mkdir(parents=True, exist_ok=True)
    pointer = pipe.artifacts_dir / "current.json"
    pointer.write_text(json.dumps({
        "version": "incumbent",
        "dataset_path": "/dev/null",
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "version": "incumbent",
            "accuracy": acc,
            "safety_score": safety,
            "avg_latency_ms": 100.0,
            "samples_evaluated": 50,
        },
    }), encoding="utf-8")


@pytest.mark.asyncio
async def test_promote_rejects_empty_benchmark(tmp_path):
    """samples_evaluated==0 must be rejected even with no baseline."""
    pipe = _pipeline(tmp_path)
    empty = BenchmarkResult(version="v", accuracy=0.9, safety_score=0.9,
                            avg_latency_ms=0.0, samples_evaluated=0)
    ok = await pipe.promote_checkpoint("v", tmp_path / "ds.jsonl", empty)
    assert ok is False
    assert not (pipe.artifacts_dir / "current.json").exists()


@pytest.mark.asyncio
async def test_promote_first_ever_accepts_any_real_benchmark(tmp_path):
    """No current.json → first promotion. HSP gate is the sole safeguard."""
    pipe = _pipeline(tmp_path)
    ds = tmp_path / "ds.jsonl"
    ds.write_text("{}\n", encoding="utf-8")
    ok = await pipe.promote_checkpoint("v-first", ds, _bench(0.7, 0.8))
    assert ok is True
    pointer = pipe.artifacts_dir / "current.json"
    assert pointer.exists()


@pytest.mark.asyncio
async def test_promote_rejects_accuracy_regression(tmp_path):
    """Lower accuracy than the incumbent → reject and leave current.json alone."""
    pipe = _pipeline(tmp_path, min_improvement=0.0)
    await _seed_current(pipe, acc=0.80, safety=0.90)

    ok = await pipe.promote_checkpoint(
        "v-regress", tmp_path / "ds.jsonl", _bench(0.79, 0.90),
    )
    assert ok is False
    # current.json still points at the incumbent
    pointer_data = json.loads((pipe.artifacts_dir / "current.json").read_text("utf-8"))
    assert pointer_data["version"] == "incumbent"


@pytest.mark.asyncio
async def test_promote_rejects_safety_regression(tmp_path):
    """Even with accuracy improvement, dropping safety must reject.
    The "no safety regression" rule is hard, not weighted-against-acc."""
    pipe = _pipeline(tmp_path)
    await _seed_current(pipe, acc=0.80, safety=0.90)
    ok = await pipe.promote_checkpoint(
        "v-unsafe", tmp_path / "ds.jsonl", _bench(0.99, 0.85),
    )
    assert ok is False


@pytest.mark.asyncio
async def test_promote_accepts_safety_improvement(tmp_path):
    """Safety up + accuracy ≥ baseline → accept."""
    pipe = _pipeline(tmp_path)
    await _seed_current(pipe, acc=0.80, safety=0.85)
    ok = await pipe.promote_checkpoint(
        "v-safer", tmp_path / "ds.jsonl", _bench(0.80, 0.92),
    )
    assert ok is True
    pointer = json.loads((pipe.artifacts_dir / "current.json").read_text("utf-8"))
    assert pointer["version"] == "v-safer"


@pytest.mark.asyncio
async def test_promote_respects_min_improvement(tmp_path):
    """A min_improvement threshold of 0.05 rejects a 0.01 accuracy gain."""
    pipe = _pipeline(tmp_path, min_improvement=0.05)
    await _seed_current(pipe, acc=0.80, safety=0.90)

    # +0.01 → below threshold, rejected.
    ok = await pipe.promote_checkpoint(
        "v-too-small", tmp_path / "ds.jsonl", _bench(0.81, 0.90),
    )
    assert ok is False
    # +0.06 → above threshold, accepted.
    ok = await pipe.promote_checkpoint(
        "v-big-enough", tmp_path / "ds.jsonl", _bench(0.86, 0.90),
    )
    assert ok is True


@pytest.mark.asyncio
async def test_promote_writes_pointer_with_expected_fields(tmp_path):
    """current.json must include every field the rollback / serving
    layer reads: version, dataset_path, promoted_at, full benchmark."""
    pipe = _pipeline(tmp_path)
    ds = tmp_path / "training-set.jsonl"
    ds.write_text("{}\n", encoding="utf-8")
    bench = _bench(0.88, 0.95)
    ok = await pipe.promote_checkpoint("v-shape", ds, bench)
    assert ok is True

    data = json.loads((pipe.artifacts_dir / "current.json").read_text("utf-8"))
    assert data["version"] == "v-shape"
    # dataset_path is normalised to an absolute path
    assert Path(data["dataset_path"]).is_absolute()
    assert str(ds.resolve()) == data["dataset_path"]
    # promoted_at is ISO 8601 parseable
    assert datetime.fromisoformat(data["promoted_at"].replace("Z", "+00:00"))
    # full benchmark embedded
    assert data["benchmark"]["accuracy"] == pytest.approx(0.88)
    assert data["benchmark"]["samples_evaluated"] == 50


# --------------------------------------------------------------------------- #
# _load_baseline_benchmark                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_load_baseline_returns_none_when_no_pointer(tmp_path):
    pipe = _pipeline(tmp_path)
    assert (await pipe._load_baseline_benchmark()) is None


@pytest.mark.asyncio
async def test_load_baseline_returns_none_on_corrupt_pointer(tmp_path):
    """A broken current.json must not raise — promotion can still proceed
    as if no baseline existed; the HSP gate stays as final guard."""
    pipe = _pipeline(tmp_path)
    pipe.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (pipe.artifacts_dir / "current.json").write_text("{not json", "utf-8")
    assert (await pipe._load_baseline_benchmark()) is None


@pytest.mark.asyncio
async def test_load_baseline_parses_valid_pointer(tmp_path):
    pipe = _pipeline(tmp_path)
    await _seed_current(pipe, acc=0.77, safety=0.88)
    b = await pipe._load_baseline_benchmark()
    assert b is not None
    assert b.version == "incumbent"
    assert b.accuracy == pytest.approx(0.77)
    assert b.safety_score == pytest.approx(0.88)
