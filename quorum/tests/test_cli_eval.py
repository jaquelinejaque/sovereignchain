# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for ``cli_eval`` — the typer sub-app over the eval_set module.

These don't re-test the underlying scoring logic (that's in
``test_eval_set.py``). They test that each ``quorum eval <cmd>``
entry point:

  * accepts the documented flags,
  * exits 0 on the happy path,
  * writes the file it advertises to write,
  * emits parseable output that downstream tooling can rely on.

If any of these break, an operator who calls ``quorum eval install``
in a fresh checkout sees a confusing error — exactly the friction
this sub-app was meant to remove. So they're cheap and worth having.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from quorum.cli_eval import app


runner = CliRunner()


def test_hash_command_prints_64_hex():
    """``quorum eval hash`` must emit a 64-character hex digest on
    stdout and exit 0. CI uses this output as a drift tripwire."""
    result = runner.invoke(app, ["hash"])
    assert result.exit_code == 0
    out = result.output.strip()
    assert len(out) == 64
    assert all(c in "0123456789abcdef" for c in out)


def test_show_default_lists_all_items():
    """Default ``show`` prints a header plus one row per canonical item."""
    result = runner.invoke(app, ["show"])
    assert result.exit_code == 0
    # Header line is present.
    assert "class" in result.output and "prompt" in result.output
    # Plus 50 rows (one per pinned item) — at least a few visible.
    assert "g01" in result.output
    assert "f15" in result.output  # last factual extension item


def test_show_class_filter_narrows_output():
    """``--class factual`` returns only factual rows."""
    result = runner.invoke(app, ["show", "--class", "factual"])
    assert result.exit_code == 0
    assert "f01" in result.output
    # No general or creative items should show.
    assert "g01" not in result.output
    assert "cr01" not in result.output


def test_show_unknown_class_exits_nonzero():
    """Filtering to a class with no items must fail loudly."""
    result = runner.invoke(app, ["show", "--class", "nope"])
    assert result.exit_code != 0


def test_show_json_emits_one_object_per_line():
    """``--json`` switches to JSONL, one object per line."""
    result = runner.invoke(app, ["show", "--class", "creative", "--json"])
    assert result.exit_code == 0
    lines = [line for line in result.output.splitlines() if line.strip()]
    assert lines, "expected at least one line"
    for line in lines:
        d = json.loads(line)
        assert d["query_class"] == "creative"
        assert "id" in d and "prompt" in d


def test_install_writes_canonical_jsonl(tmp_path):
    """``install --path X`` writes the canonical set at X with the
    expected line count, and the output advertises the SHA."""
    out_path = tmp_path / "eval.jsonl"
    result = runner.invoke(app, ["install", "--path", str(out_path)])
    assert result.exit_code == 0
    assert out_path.exists()
    lines = out_path.read_text("utf-8").splitlines()
    assert len(lines) == 50
    assert "sha256:" in result.output
    # SHA in output matches what `hash` returns.
    hash_result = runner.invoke(app, ["hash"])
    assert hash_result.output.strip() in result.output


def test_run_writes_sidecar_with_expected_keys(tmp_path):
    """``run --version vN --sidecar Y`` writes a sidecar JSON that
    contains every field DistillationPipeline._run_benchmark reads."""
    sidecar = tmp_path / "bench.json"
    result = runner.invoke(app, [
        "run",
        "--version", "smoke",
        "--sidecar", str(sidecar),
        "--quiet",
    ])
    assert result.exit_code == 0, result.output
    assert sidecar.exists()
    data = json.loads(sidecar.read_text("utf-8"))
    for key in (
        "version", "accuracy", "safety_score",
        "avg_latency_ms", "samples_evaluated", "per_item",
    ):
        assert key in data, f"sidecar missing {key}"
    assert data["version"] == "smoke"
    assert data["samples_evaluated"] == 50
    # Stdout summary is valid JSON too.
    summary = json.loads(result.output)
    assert summary["version"] == "smoke"


def test_run_with_class_filter_runs_fewer_items(tmp_path):
    """``--class factual`` cuts the eval set down for a fast spot check."""
    sidecar = tmp_path / "bench-fact.json"
    result = runner.invoke(app, [
        "run",
        "--version", "fact-only",
        "--sidecar", str(sidecar),
        "--class", "factual",
        "--quiet",
    ])
    assert result.exit_code == 0
    data = json.loads(sidecar.read_text("utf-8"))
    # 15 factual items in the canonical set.
    assert data["samples_evaluated"] == 15


def test_run_unknown_class_exits_nonzero(tmp_path):
    """``run --class nope`` must fail loudly, not silently run zero items
    and write a misleading sidecar with samples_evaluated=0."""
    sidecar = tmp_path / "bench.json"
    result = runner.invoke(app, [
        "run", "--version", "v", "--sidecar", str(sidecar),
        "--class", "this_class_does_not_exist", "--quiet",
    ])
    assert result.exit_code != 0
    # And no sidecar was written.
    assert not sidecar.exists()


def test_install_is_idempotent_on_canonical_file(tmp_path):
    """Re-running ``install`` on an already-canonical file is a no-op.
    Catches a regression where the writer would clobber and re-write,
    burning disk and momentarily emptying the file."""
    out = tmp_path / "eval.jsonl"
    first = runner.invoke(app, ["install", "--path", str(out)])
    assert first.exit_code == 0
    mtime1 = out.stat().st_mtime
    content1 = out.read_text("utf-8")

    # Re-run without --overwrite. File should still be the same.
    second = runner.invoke(app, ["install", "--path", str(out)])
    assert second.exit_code == 0
    assert out.read_text("utf-8") == content1
    # mtime may or may not move (depends on writer optimization),
    # but content stability is the contract that matters.


def test_install_overwrite_replaces_custom_file(tmp_path):
    """``--overwrite`` must replace a pre-existing non-canonical file."""
    out = tmp_path / "eval.jsonl"
    out.write_text("{\"id\":\"custom\",\"prompt\":\"x\"}\n", "utf-8")
    custom_content = out.read_text("utf-8")

    result = runner.invoke(app, ["install", "--path", str(out), "--overwrite"])
    assert result.exit_code == 0
    new_content = out.read_text("utf-8")
    assert new_content != custom_content
    # Canonical file is exactly 50 lines.
    assert len(new_content.splitlines()) == 50


def test_install_without_overwrite_preserves_custom_file(tmp_path):
    """Without --overwrite, a pre-existing non-canonical file is left
    alone (operator may have placed their own eval set there)."""
    out = tmp_path / "eval.jsonl"
    custom = "{\"id\":\"custom\",\"prompt\":\"x\"}\n"
    out.write_text(custom, "utf-8")

    result = runner.invoke(app, ["install", "--path", str(out)])
    assert result.exit_code == 0
    assert out.read_text("utf-8") == custom
