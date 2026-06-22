# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Coverage for the JSONL legacy path: ``collect_distillation_candidates``,
``_record_to_sample``, and ``build_dataset``.

The bridge path (``collect_from_response_log``) is exercised in
``test_distillation_bridge.py``. This file pins the *original* code
path the docstrings reference — the JSONL log that self-hosted
operators may still ship — so a refactor of either path doesn't quietly
break the other.

Behaviours under contract:

* missing log file → empty list, no exception
* malformed JSON lines are skipped (single broken line doesn't abort)
* time filter: records before ``since`` are dropped, records at or
  after are kept; both naive and ISO-Z timestamps are accepted
* consensus filter: ``confidence`` below threshold drops a record
* frontier count: fewer than ``min_pair_count`` frontier responders
  → drop
* canonical answer fallback: when no ``answer`` field is present,
  pick the longest frontier response
* build_dataset emits one JSONL line per sample, each with the
  Unsloth ``messages`` shape, and creates parent dirs as needed
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quorum.evolution.distillation import (
    DEFAULT_FRONTIER_MODELS,
    DistillationPipeline,
    DistillationSample,
)


def _pipeline(tmp_path: Path) -> DistillationPipeline:
    return DistillationPipeline(
        log_path=tmp_path / "queries.jsonl",
        eval_set_path=tmp_path / "eval_set.jsonl",
        artifacts_dir=tmp_path / "artifacts",
    )


def _write_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _record(
    *,
    prompt: str = "q",
    answer: str = "a",
    confidence: float = 0.92,
    when: datetime | None = None,
    models: list[dict] | None = None,
) -> dict:
    """Build a JSONL record with all the fields the JSONL parser reads."""
    if when is None:
        when = datetime.now(timezone.utc)
    if models is None:
        models = [
            {"name": "anthropic", "response": answer},
            {"name": "openai", "response": answer},
            {"name": "gemini", "response": answer},
        ]
    return {
        "timestamp": when.isoformat(),
        "prompt": prompt,
        "answer": answer,
        "confidence": confidence,
        "models": models,
    }


# --------------------------------------------------------------------------- #
# collect_distillation_candidates                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_collect_returns_empty_when_log_missing(tmp_path):
    pipe = _pipeline(tmp_path)
    assert not pipe.log_path.exists()
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert out == []


@pytest.mark.asyncio
async def test_collect_skips_malformed_lines_and_keeps_valid(tmp_path):
    """A broken line in the middle of the log must not abort the
    rest of the read — the cron job runs unattended."""
    pipe = _pipeline(tmp_path)
    pipe.log_path.parent.mkdir(parents=True, exist_ok=True)
    with pipe.log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_record(prompt="ok-1")) + "\n")
        fh.write("{ this is not json\n")
        fh.write("\n")  # blank line
        fh.write(json.dumps(_record(prompt="ok-2")) + "\n")

    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
    )
    queries = {s.query for s in out}
    assert queries == {"ok-1", "ok-2"}


@pytest.mark.asyncio
async def test_collect_filters_by_since(tmp_path):
    pipe = _pipeline(tmp_path)
    now = datetime.now(timezone.utc)
    _write_log(pipe.log_path, [
        _record(prompt="old", when=now - timedelta(days=5)),
        _record(prompt="fresh", when=now - timedelta(hours=1)),
    ])
    out = await pipe.collect_distillation_candidates(
        since=now - timedelta(days=1),
    )
    assert [s.query for s in out] == ["fresh"]


@pytest.mark.asyncio
async def test_collect_drops_low_confidence(tmp_path):
    pipe = _pipeline(tmp_path)
    _write_log(pipe.log_path, [
        _record(prompt="weak", confidence=0.50),
        _record(prompt="strong", confidence=0.95),
    ])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
        min_consensus=0.85,
    )
    assert [s.query for s in out] == ["strong"]


@pytest.mark.asyncio
async def test_collect_requires_min_pair_count_frontier_models(tmp_path):
    """Only two frontier responses → drop (default threshold is 3)."""
    pipe = _pipeline(tmp_path)
    _write_log(pipe.log_path, [
        _record(prompt="thin", models=[
            {"name": "anthropic", "response": "a"},
            {"name": "openai", "response": "a"},
            # No third frontier — the next one is a local model.
            {"name": "ollama-llama", "response": "a"},
        ]),
    ])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
        min_pair_count=3,
    )
    assert out == []


@pytest.mark.asyncio
async def test_collect_falls_back_to_longest_response_when_no_answer(tmp_path):
    """When the record has no `answer` field, the longest frontier
    response wins as the canonical answer."""
    pipe = _pipeline(tmp_path)
    rec = _record(models=[
        {"name": "anthropic", "response": "short"},
        {"name": "openai", "response": "this is the longest response by far"},
        {"name": "gemini", "response": "medium length"},
    ])
    rec.pop("answer")
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert len(out) == 1
    assert out[0].consensus_response == "this is the longest response by far"


@pytest.mark.asyncio
async def test_collect_accepts_z_suffix_timestamp(tmp_path):
    """`2026-06-22T10:00:00Z` (trailing Z) must parse — production
    logs from some providers use this form instead of `+00:00`."""
    pipe = _pipeline(tmp_path)
    now = datetime.now(timezone.utc)
    rec = _record(when=now)
    # Replace the +00:00 suffix with Z to exercise the .replace("Z", "+00:00") path.
    rec["timestamp"] = rec["timestamp"].replace("+00:00", "Z")
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=now - timedelta(days=1),
    )
    assert len(out) == 1


@pytest.mark.asyncio
async def test_collect_drops_records_without_timestamp(tmp_path):
    pipe = _pipeline(tmp_path)
    rec = _record()
    rec.pop("timestamp")
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert out == []


@pytest.mark.asyncio
async def test_collect_drops_records_with_unparseable_timestamp(tmp_path):
    """A record whose ``timestamp`` field can't be parsed as ISO 8601
    must drop quietly, not abort the read. Real production logs have
    accumulated junk over time."""
    pipe = _pipeline(tmp_path)
    rec = _record()
    rec["timestamp"] = "yesterday-ish"
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert out == []


@pytest.mark.asyncio
async def test_collect_accepts_naive_iso_timestamp_as_utc(tmp_path):
    """A timestamp without timezone info ("2026-06-22T10:00:00") must
    be interpreted as UTC, matching the docstring contract. Otherwise
    naive logs would unpredictably drift in/out of the `since` window."""
    pipe = _pipeline(tmp_path)
    now = datetime.now(timezone.utc)
    rec = _record(when=now)
    # Strip tz info: "2026-06-22T10:00:00+00:00" → "2026-06-22T10:00:00"
    rec["timestamp"] = rec["timestamp"].replace("+00:00", "")
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=now - timedelta(days=1),
    )
    assert len(out) == 1


@pytest.mark.asyncio
async def test_collect_uses_alt_field_names_query_and_agreement_score(tmp_path):
    """``_record_to_sample`` reads two key shapes for prompt and confidence
    (``prompt`` OR ``query``; ``confidence`` OR ``agreement_score``).
    Coverage prevents a rename from quietly dropping the alt path."""
    pipe = _pipeline(tmp_path)
    rec = _record()
    # Move prompt -> query, confidence -> agreement_score
    rec["query"] = rec.pop("prompt")
    rec["agreement_score"] = rec.pop("confidence")
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert len(out) == 1
    assert out[0].query == "q"


@pytest.mark.asyncio
async def test_collect_drops_record_with_empty_prompt(tmp_path):
    """An empty prompt produces a useless sample; drop it before it
    hits the trainer."""
    pipe = _pipeline(tmp_path)
    rec = _record(prompt="")
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
    )
    assert out == []


@pytest.mark.asyncio
async def test_collect_skips_models_with_error_or_empty_response(tmp_path):
    """A model that errored or returned an empty string is NOT a
    frontier responder for the min_pair_count test."""
    pipe = _pipeline(tmp_path)
    rec = _record(models=[
        {"name": "anthropic", "response": "good"},
        {"name": "openai", "response": "", "error": "rate_limited"},
        {"name": "gemini", "response": ""},
        # Two would have qualified; only one valid.
    ])
    _write_log(pipe.log_path, [rec])
    out = await pipe.collect_distillation_candidates(
        since=datetime.now(timezone.utc) - timedelta(days=1),
        min_pair_count=2,
    )
    assert out == []


# --------------------------------------------------------------------------- #
# build_dataset                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_build_dataset_writes_one_unsloth_line_per_sample(tmp_path):
    pipe = _pipeline(tmp_path)
    samples = [
        DistillationSample(
            query="q1", consensus_response="a1",
            source_models=["claude"], agreement_score=0.9,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
        DistillationSample(
            query="q2", consensus_response="a2",
            source_models=["openai", "gemini"], agreement_score=0.95,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ),
    ]
    out_path = tmp_path / "nested" / "more" / "dataset.jsonl"
    out = await pipe.build_dataset(samples, out_path)

    assert out == out_path.resolve()
    lines = out.read_text("utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["messages"][0] == {"role": "user", "content": "q1"}
    assert first["messages"][1] == {"role": "assistant", "content": "a1"}
    assert first["_meta"]["agreement_score"] == 0.9


@pytest.mark.asyncio
async def test_build_dataset_empty_input_writes_empty_file(tmp_path):
    """Zero samples → empty file (not no file). The cron wrapper
    reads it next; an absent file would be a confusing fail signal."""
    pipe = _pipeline(tmp_path)
    out_path = tmp_path / "empty.jsonl"
    out = await pipe.build_dataset([], out_path)
    assert out.exists()
    assert out.read_text("utf-8") == ""


# --------------------------------------------------------------------------- #
# Sanity: frontier set defaults                                                #
# --------------------------------------------------------------------------- #


def test_default_frontier_models_contains_expected_tags():
    """If this assertion ever fails, the frontier-model filter is
    quietly accepting / rejecting a model nobody intended to move."""
    for tag in ("anthropic", "openai", "gemini", "claude", "gpt"):
        assert tag in DEFAULT_FRONTIER_MODELS
