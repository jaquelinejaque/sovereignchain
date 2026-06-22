# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Cover ``DistillationPipeline.run_nightly`` — the cron orchestrator.

run_nightly stitches the four-step pipeline together. It is the entry
point the cron job calls, and the function whose silent failure-mode
the original "AUTONOMOUS_SESSION" doc called out: if any step crashes,
the scheduler must NOT crash — every path returns a structured summary
the wrapper can log.

Paths covered:

1. **No candidates collected** → noop, summary.candidates == 0,
   error is None.
2. **Fine-tune skipped** (no Unsloth) → summary.checkpoint is None,
   benchmark and promoted untouched, error is None.
3. **Promote denied by HSP gate** → summary.error == "hsp_denied: ...",
   summary.promoted is False.
4. **Successful promotion** → summary.promoted is True, current.json
   is written.
5. **Unexpected exception** (e.g. SQLite I/O fails) → summary.error
   is the str of the exception; the scheduler does NOT see it raised.

Every dependency is monkeypatched so the test runs in milliseconds.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quorum.evolution import distillation
from quorum.evolution.distillation import (
    BenchmarkResult,
    DistillationPipeline,
    DistillationSample,
)
from quorum.hsp.gate import HSPGateDenied


@pytest.fixture(autouse=True)
def _hsp_dev_mode(monkeypatch):
    monkeypatch.setenv("HSP_GATE_DEV_MODE", "1")
    yield


def _pipeline(tmp_path: Path) -> DistillationPipeline:
    return DistillationPipeline(
        log_path=tmp_path / "queries.jsonl",
        eval_set_path=tmp_path / "eval_set.jsonl",
        artifacts_dir=tmp_path / "artifacts",
    )


def _sample() -> DistillationSample:
    return DistillationSample(
        query="q",
        consensus_response="a",
        source_models=["anthropic", "openai", "gemini"],
        agreement_score=0.92,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------- #
# Path 1: no candidates → clean noop                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nightly_noop_when_no_candidates(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path)

    async def _empty(*_a, **_kw):
        return []

    monkeypatch.setattr(pipe, "collect_distillation_candidates", _empty)

    summary = await pipe.run_nightly(since=datetime.now(timezone.utc) - timedelta(days=1))

    assert summary["candidates"] == 0
    assert summary["dataset_path"] is None
    assert summary["checkpoint"] is None
    assert summary["promoted"] is False
    assert summary["error"] is None


# --------------------------------------------------------------------------- #
# Path 2: fine-tune skipped (no Unsloth) → noop after dataset build           #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nightly_noop_when_finetune_skipped(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path)

    async def _one(*_a, **_kw):
        return [_sample()]

    async def _build(samples, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n", encoding="utf-8")
        return out_path

    async def _no_ckpt(*_a, **_kw):
        return None

    monkeypatch.setattr(pipe, "collect_distillation_candidates", _one)
    monkeypatch.setattr(pipe, "build_dataset", _build)
    monkeypatch.setattr(pipe, "_run_finetune", _no_ckpt)

    summary = await pipe.run_nightly(since=datetime.now(timezone.utc) - timedelta(days=1))

    assert summary["candidates"] == 1
    assert summary["dataset_path"] is not None
    assert summary["checkpoint"] is None
    assert summary["benchmark"] is None
    assert summary["promoted"] is False
    assert summary["error"] is None


# --------------------------------------------------------------------------- #
# Path 3: HSP gate denial → captured as error, not raised                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nightly_captures_hsp_denial(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path)

    ckpt = tmp_path / "artifacts" / "checkpoint-x"

    async def _one(*_a, **_kw):
        return [_sample()]

    async def _build(samples, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}", "utf-8")
        return out_path

    async def _ckpt(*_a, **_kw):
        ckpt.mkdir(parents=True, exist_ok=True)
        return ckpt

    async def _bench(version, ckpt_dir):
        return BenchmarkResult(version=version, accuracy=0.9, safety_score=0.95,
                               avg_latency_ms=10.0, samples_evaluated=50)

    async def _denied(*_a, **_kw):
        raise HSPGateDenied("approver said no")

    monkeypatch.setattr(pipe, "collect_distillation_candidates", _one)
    monkeypatch.setattr(pipe, "build_dataset", _build)
    monkeypatch.setattr(pipe, "_run_finetune", _ckpt)
    monkeypatch.setattr(pipe, "_run_benchmark", _bench)
    monkeypatch.setattr(pipe, "promote_checkpoint", _denied)

    summary = await pipe.run_nightly(since=datetime.now(timezone.utc) - timedelta(days=1))

    assert summary["promoted"] is False
    assert summary["error"] is not None
    assert summary["error"].startswith("hsp_denied:")
    assert summary["benchmark"] is not None


# --------------------------------------------------------------------------- #
# Path 4: successful promotion                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nightly_success_writes_pointer(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path)

    async def _one(*_a, **_kw):
        return [_sample()]

    async def _build(samples, out_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}", "utf-8")
        return out_path

    async def _ckpt(*_a, version="v", **_kw):
        d = pipe.artifacts_dir / f"checkpoint-{version}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def _bench(version, ckpt_dir):
        return BenchmarkResult(version=version, accuracy=0.9, safety_score=0.95,
                               avg_latency_ms=10.0, samples_evaluated=50)

    monkeypatch.setattr(pipe, "collect_distillation_candidates", _one)
    monkeypatch.setattr(pipe, "build_dataset", _build)
    monkeypatch.setattr(pipe, "_run_finetune", _ckpt)
    monkeypatch.setattr(pipe, "_run_benchmark", _bench)
    # Don't stub promote_checkpoint — let the real (HSP-bypassed) path
    # exercise its happy branch and write current.json.

    summary = await pipe.run_nightly(since=datetime.now(timezone.utc) - timedelta(days=1))

    assert summary["promoted"] is True
    assert summary["error"] is None
    pointer = pipe.artifacts_dir / "current.json"
    assert pointer.exists()
    data = json.loads(pointer.read_text("utf-8"))
    # version is a UTC timestamp like 20260622-...
    assert data["version"] == summary["version"]
    assert data["benchmark"]["samples_evaluated"] == 50


# --------------------------------------------------------------------------- #
# Path 5: unexpected exception in collect → captured, not raised              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nightly_captures_unexpected_exception(tmp_path, monkeypatch):
    """A SQLite I/O error during collect must surface in summary.error
    instead of crashing the scheduler."""
    pipe = _pipeline(tmp_path)

    async def _boom(*_a, **_kw):
        raise OSError("disk full")

    monkeypatch.setattr(pipe, "collect_distillation_candidates", _boom)

    summary = await pipe.run_nightly(since=datetime.now(timezone.utc) - timedelta(days=1))

    assert summary["promoted"] is False
    assert summary["error"] == "disk full"
    # Cron wrapper can still parse the structured summary.
    assert "version" in summary and summary["candidates"] == 0


# --------------------------------------------------------------------------- #
# Bonus: summary version is timestamp-shaped and stable inside one call       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_nightly_version_is_utc_timestamp(tmp_path, monkeypatch):
    pipe = _pipeline(tmp_path)

    async def _empty(*_a, **_kw):
        return []

    monkeypatch.setattr(pipe, "collect_distillation_candidates", _empty)
    summary = await pipe.run_nightly(since=datetime.now(timezone.utc) - timedelta(days=1))

    # Shape: YYYYmmdd-HHMMSS
    v = summary["version"]
    assert len(v) == 15
    assert v[8] == "-"
    # Parseable round-trip
    datetime.strptime(v, "%Y%m%d-%H%M%S")
