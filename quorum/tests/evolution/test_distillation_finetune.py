# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for ``DistillationPipeline._run_finetune``.

The fine-tune step shells out to the Unsloth CLI. CI hosts have no
GPU and no Unsloth binary, so the production path can never run there
— but the three contract branches still need coverage so a refactor
doesn't quietly break them:

1. **Unsloth not installed** → return ``None`` (the upstream
   ``promote_checkpoint`` treats None as "no new candidate this round").
2. **Unsloth present, exits 0** → return the checkpoint directory.
3. **Unsloth present, exits nonzero** → return ``None`` and log the
   stderr tail.

These tests stub :func:`asyncio.create_subprocess_exec` and
:func:`shutil.which` so they run anywhere, in well under a second.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quorum.evolution import distillation
from quorum.evolution.distillation import DistillationPipeline


# --------------------------------------------------------------------------- #
# Subprocess test double                                                       #
# --------------------------------------------------------------------------- #


def _fake_process(returncode: int, stdout: bytes = b"", stderr: bytes = b""):
    """Build an object that quacks like ``asyncio.subprocess.Process``.

    Only ``.communicate()`` and ``.returncode`` are used by the SUT.
    """
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    return proc


def _pipeline(tmp_path: Path) -> DistillationPipeline:
    """Pipeline with all paths under tmp_path — no global state leakage."""
    return DistillationPipeline(
        log_path=tmp_path / "queries.jsonl",
        eval_set_path=tmp_path / "eval_set.jsonl",
        artifacts_dir=tmp_path / "artifacts",
    )


# --------------------------------------------------------------------------- #
# Path 1: Unsloth absent                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_finetune_returns_none_when_unsloth_missing(tmp_path, monkeypatch):
    """No ``unsloth`` binary and no ``UNSLOTH_BIN`` env → return None."""
    monkeypatch.setattr(distillation.shutil, "which", lambda _: None)
    monkeypatch.delenv("UNSLOTH_BIN", raising=False)

    pipe = _pipeline(tmp_path)
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")

    out = await pipe._run_finetune(dataset, "v-missing")

    assert out is None
    # And no checkpoint dir was created — the function returned before
    # touching the filesystem.
    assert not (pipe.artifacts_dir / "checkpoint-v-missing").exists()


@pytest.mark.asyncio
async def test_finetune_uses_env_var_when_path_lookup_fails(tmp_path, monkeypatch):
    """``shutil.which`` may not find unsloth in a custom-install layout;
    the ``UNSLOTH_BIN`` env var must take over so operators can point at
    a manually-placed binary without rebuilding ``$PATH``."""
    monkeypatch.setattr(distillation.shutil, "which", lambda _: None)
    fake_bin = tmp_path / "fake-unsloth"
    monkeypatch.setenv("UNSLOTH_BIN", str(fake_bin))

    create = AsyncMock(return_value=_fake_process(0, stdout=b"trained"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    pipe = _pipeline(tmp_path)
    dataset = tmp_path / "ds.jsonl"

    out = await pipe._run_finetune(dataset, "v1")

    assert out == pipe.artifacts_dir / "checkpoint-v1"
    # Subprocess was called with the env-var binary, not "unsloth".
    args, _ = create.call_args
    assert args[0] == str(fake_bin)
    assert args[1] == "train"


# --------------------------------------------------------------------------- #
# Path 2: Unsloth present, exits 0                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_finetune_happy_path_returns_checkpoint_dir(tmp_path, monkeypatch):
    """rc=0 → return the freshly-created checkpoint directory."""
    monkeypatch.setattr(distillation.shutil, "which", lambda _: "/usr/local/bin/unsloth")

    proc = _fake_process(0, stdout=b"epoch 1/1 done\n", stderr=b"")
    create = AsyncMock(return_value=proc)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    pipe = _pipeline(tmp_path)
    dataset = tmp_path / "ds.jsonl"

    out = await pipe._run_finetune(dataset, "v2")

    assert out is not None
    assert out == pipe.artifacts_dir / "checkpoint-v2"
    assert out.exists() and out.is_dir()

    # Command shape: <bin> train --dataset <path> --output <ckpt_dir>
    args, _ = create.call_args
    assert args[1] == "train"
    assert "--dataset" in args and "--output" in args
    assert str(dataset) in args
    assert str(out) in args


@pytest.mark.asyncio
async def test_finetune_pipes_stdout_and_stderr(tmp_path, monkeypatch):
    """Subprocess must be created with PIPE for stdout AND stderr so the
    happy-path log line can include a stdout tail, and the failure-path
    log line can include the stderr tail."""
    monkeypatch.setattr(distillation.shutil, "which", lambda _: "/u/bin")

    create = AsyncMock(return_value=_fake_process(0))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    pipe = _pipeline(tmp_path)
    await pipe._run_finetune(tmp_path / "ds.jsonl", "v3")

    _, kwargs = create.call_args
    assert kwargs["stdout"] == asyncio.subprocess.PIPE
    assert kwargs["stderr"] == asyncio.subprocess.PIPE


# --------------------------------------------------------------------------- #
# Path 3: Unsloth present, exits nonzero                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_finetune_returns_none_on_nonzero_exit(tmp_path, monkeypatch):
    """rc≠0 → None, even though the checkpoint dir was created.

    The caller treats None as "no new candidate", which is exactly the
    semantic we want when fine-tuning crashed."""
    monkeypatch.setattr(distillation.shutil, "which", lambda _: "/u/bin")

    proc = _fake_process(2, stderr=b"CUDA out of memory")
    create = AsyncMock(return_value=proc)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    pipe = _pipeline(tmp_path)
    out = await pipe._run_finetune(tmp_path / "ds.jsonl", "v-crash")

    assert out is None


@pytest.mark.asyncio
async def test_finetune_creates_artifacts_dir_under_artifacts_dir(tmp_path, monkeypatch):
    """The checkpoint dir must be a child of ``self.artifacts_dir``,
    not a sibling — this is how ``promote_checkpoint`` finds it later."""
    monkeypatch.setattr(distillation.shutil, "which", lambda _: "/u/bin")
    monkeypatch.setattr(asyncio, "create_subprocess_exec",
                        AsyncMock(return_value=_fake_process(0)))

    pipe = _pipeline(tmp_path)
    out = await pipe._run_finetune(tmp_path / "ds.jsonl", "v4")

    assert out is not None
    assert out.parent == pipe.artifacts_dir
    assert out.name == "checkpoint-v4"
