# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for ``cli_responses`` — typer sub-app over response_log.

Sibling of ``test_cli_eval``. These tests don't re-validate the
underlying response_log storage logic (that's in
``test_response_log.py``); they verify that each CLI surface:

  * accepts the documented flags,
  * emits parseable output,
  * writes the file it advertises to write,
  * respects ``QUORUM_LOG_RESPONSES`` (no-op vs active).

The fixture isolates the DB path per test so commands cannot bleed
across runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quorum.evolution import response_log
from quorum.cli_responses import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Fresh per-test DB and a clean opt-in flag."""
    db = tmp_path / "responses.db"
    monkeypatch.setenv(response_log._ENV_DB_PATH, str(db))
    monkeypatch.delenv(response_log._ENV_FLAG, raising=False)
    yield db


def _enable(monkeypatch):
    monkeypatch.setenv(response_log._ENV_FLAG, "1")


def _seed_rows(prompt: str = "demo prompt"):
    """Insert one consensus round so `stats` and `export` have data."""
    asyncio.run(
        response_log.record_consensus_round(
            prompt=prompt,
            query_class="general",
            model_responses=[
                {"model": "claude", "response_text": "answer-from-claude", "weight": 0.4},
                {"model": "gpt", "response_text": "answer-from-gpt", "weight": 0.3},
            ],
            canonical_model="claude",
        )
    )


def test_stats_no_db(_isolated_db):
    """Without writes, `stats` must still exit 0 and report db_exists=false.
    Avoids spurious red CI runs on fresh checkouts."""
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["enabled"] is False
    assert data["db_exists"] is False


def test_stats_after_writes(_isolated_db, monkeypatch):
    """After at least one consensus round is logged, `stats` reports
    nonzero counts and a coherent date range."""
    _enable(monkeypatch)
    _seed_rows()
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["rows"] == 2
    assert data["distinct_models"] == 2
    assert data["distinct_queries"] == 1


def test_export_to_stdout_yields_jsonl(_isolated_db, monkeypatch):
    """`export` without `--out` writes JSONL to stdout — one object
    per line, ready for `jq`."""
    _enable(monkeypatch)
    _seed_rows()
    result = runner.invoke(app, ["export"])
    assert result.exit_code == 0
    lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert {r["model"] for r in rows} == {"claude", "gpt"}


def test_export_to_file_writes_count_message(_isolated_db, monkeypatch, tmp_path):
    """`export --out X` writes to file and reports the row count on
    stderr (so stdout stays parseable when piped)."""
    _enable(monkeypatch)
    _seed_rows()
    out = tmp_path / "dump.jsonl"
    result = runner.invoke(app, ["export", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert len(out.read_text("utf-8").splitlines()) == 2
    # `wrote N rows` message lands on stderr (typer.echo err=True).
    # CliRunner merges streams by default; check the merged output.
    assert "wrote 2 rows" in result.output


def test_export_model_filter(_isolated_db, monkeypatch):
    """`--model` filters to a single model."""
    _enable(monkeypatch)
    _seed_rows()
    result = runner.invoke(app, ["export", "--model", "gpt"])
    assert result.exit_code == 0
    lines = [l for l in result.output.splitlines() if l.strip()]
    assert len(lines) == 1
    assert json.loads(lines[0])["model"] == "gpt"


def test_vacuum_rejects_zero_days(_isolated_db, monkeypatch):
    """Passing `--older-than-days 0` must fail loudly — a silent
    accept would delete every row, the opposite of the operator's intent."""
    _enable(monkeypatch)
    _seed_rows()
    result = runner.invoke(app, ["vacuum", "--older-than-days", "0", "--yes"])
    assert result.exit_code != 0


def test_vacuum_with_yes_deletes_old_rows(_isolated_db, monkeypatch):
    """`vacuum --yes --older-than-days 0.something` would delete
    everything; use a long horizon to verify nothing is deleted and
    the command still exits 0. (Old-row deletion behaviour is tested
    in test_response_log.)"""
    _enable(monkeypatch)
    _seed_rows()
    # 365 days is well beyond the rows we just wrote.
    result = runner.invoke(app, ["vacuum", "--older-than-days", "365", "--yes"])
    assert result.exit_code == 0
    assert "deleted 0 rows" in result.output
