"""Loop 3 — Knowledge Distillation Pipeline (the crown jewel).

Why this module exists:
    Every night we mine the query log for cases where the frontier models
    (Claude + GPT + Gemini) agreed strongly on the same answer. Those high-
    consensus (query, answer) pairs are gold: they are essentially free
    supervised data, hand-curated by 3 frontier models voting in unison.
    We package them as a fine-tuning dataset for the local Llama (Unsloth
    JSONL format) so the local model progressively closes the quality gap
    with the frontier — without us ever writing a label by hand.

    This is the loop that makes Quorum compound: every consensus query that
    happens during the day is a potential training example tomorrow.

License:
    Apache 2.0 — see LICENSE.
    HSP commercial restrictions apply — see LICENSE-HSP (PCT/US26/11908).
    This module is HSP-GATED: promoting a fine-tuned checkpoint to production
    requires human (or HSP-certified webhook) approval. Bypassing the gate to
    auto-promote checkpoints in a commercial deployment is a license violation.

Triggers on: nightly cron.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from quorum.hsp.gate import HSPGateDenied, requires_hsp_approval

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class DistillationSample:
    """One (query, consensus_response) training pair distilled from the log.

    Why a dataclass and not a pydantic model:
        These objects are produced by the thousands in a nightly batch and
        then serialised to JSONL. We want zero pydantic validation overhead
        on the hot path; the upstream consensus engine already validated
        the shape of every field.
    """

    query: str
    consensus_response: str
    source_models: list[str]
    agreement_score: float
    timestamp: str  # ISO 8601 — string form keeps JSONL round-tripping trivial.

    def to_unsloth_messages(self) -> dict[str, Any]:
        """Convert to Unsloth chat-format dict.

        Why: Unsloth's `SFTTrainer` accepts `{"messages": [{"role": ..., "content": ...}]}`
        directly when `dataset_text_field="messages"` and a chat template is set.
        Keeping this conversion in the dataclass means callers don't have to
        remember the wire format.
        """
        return {
            "messages": [
                {"role": "user", "content": self.query},
                {"role": "assistant", "content": self.consensus_response},
            ],
            "_meta": {
                "source_models": self.source_models,
                "agreement_score": self.agreement_score,
                "timestamp": self.timestamp,
            },
        }


@dataclass
class BenchmarkResult:
    """Eval-set benchmark used to gate promotion.

    Why we need this:
        Fine-tuning can silently degrade a model (catastrophic forgetting,
        overfit on the consensus style, hallucinations on out-of-distribution
        prompts). We refuse to promote a checkpoint unless the new version
        beats the incumbent on a fixed held-out eval set by at least
        `min_improvement` AND does not regress on any pinned safety metric.
    """

    version: str
    accuracy: float
    safety_score: float
    avg_latency_ms: float
    samples_evaluated: int
    extra: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


DATA_DIR = Path(os.getenv("QUORUM_DATA_DIR", str(Path.home() / ".quorum"))).expanduser()

# Frontier models we trust enough to consider their consensus a label.
# Other models in the log (local Llama, smaller open models) are NOT used
# as labellers — they're the students, not the teachers.
DEFAULT_FRONTIER_MODELS: frozenset[str] = frozenset(
    {"anthropic", "openai", "gemini", "claude", "gpt", "gpt-4", "gpt-5"}
)


class DistillationPipeline:
    """Nightly distillation pipeline.

    Lifecycle:
        run_nightly()
            → collect_distillation_candidates()   # scan query log
            → build_dataset()                     # JSONL for Unsloth
            → _run_finetune()                     # subprocess to unsloth (best-effort)
            → _run_benchmark()                    # eval set comparison
            → promote_checkpoint()                # HSP-gated; fails closed

    Why a class and not free functions:
        We carry a small amount of config (log path, eval-set path, frontier
        set, minimum improvement threshold) that would otherwise be passed
        through every function call. A class keeps the wire clean.
    """

    def __init__(
        self,
        *,
        log_path: Path | str | None = None,
        eval_set_path: Path | str | None = None,
        artifacts_dir: Path | str | None = None,
        frontier_models: Iterable[str] | None = None,
        min_improvement: float = 0.0,
    ) -> None:
        self.log_path = Path(log_path or DATA_DIR / "queries.jsonl")
        self.eval_set_path = Path(
            eval_set_path or DATA_DIR / "eval_set.jsonl"
        )
        self.artifacts_dir = Path(
            artifacts_dir or DATA_DIR / "distillation"
        )
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.frontier_models = (
            frozenset(m.lower() for m in frontier_models)
            if frontier_models is not None
            else DEFAULT_FRONTIER_MODELS
        )
        self.min_improvement = min_improvement

    # ------------------------------------------------------------------
    # Step 1 — mine the query log
    # ------------------------------------------------------------------

    async def collect_distillation_candidates(
        self,
        since: datetime,
        min_consensus: float = 0.85,
        min_pair_count: int = 3,
    ) -> list[DistillationSample]:
        """Read the JSONL query log and return strong-consensus samples.

        Args:
            since: Only consider queries logged at-or-after this UTC instant.
                   Nightly cron passes "now - 24h" to incrementally distil.
            min_consensus: Minimum agreement score to keep a sample. The 0.85
                   default is empirical: below that, frontier models tend to
                   disagree on substance, not just phrasing, and the label is
                   noisy.
            min_pair_count: A sample only counts as "frontier consensus" if at
                   least this many frontier models contributed an answer. With
                   only 2 frontier models it's hard to distinguish agreement
                   from a coin flip.

        Why we do file IO in `asyncio.to_thread`:
            The log can be large (100MB+ after a busy day). Blocking the event
            loop while we stream-parse it would freeze any concurrent web/CLI
            work. asyncio.to_thread keeps the loop responsive.
        """
        if not self.log_path.exists():
            logger.warning(
                "distillation.log_missing path=%s — returning 0 candidates",
                self.log_path,
            )
            return []

        # Normalise `since` to UTC for comparisons.
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)

        def _read_and_filter() -> list[DistillationSample]:
            samples: list[DistillationSample] = []
            with self.log_path.open("r", encoding="utf-8") as fh:
                for line_no, raw in enumerate(fh, 1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        record = json.loads(raw)
                    except json.JSONDecodeError:
                        logger.debug("distillation.bad_json line=%d", line_no)
                        continue

                    sample = self._record_to_sample(
                        record, min_consensus, min_pair_count, since
                    )
                    if sample is not None:
                        samples.append(sample)
            return samples

        samples = await asyncio.to_thread(_read_and_filter)
        logger.info(
            "distillation.collected count=%d since=%s min_consensus=%.2f",
            len(samples),
            since.isoformat(),
            min_consensus,
        )
        return samples

    def _record_to_sample(
        self,
        record: dict[str, Any],
        min_consensus: float,
        min_pair_count: int,
        since: datetime,
    ) -> DistillationSample | None:
        """Convert one raw log record to a sample, or None if it doesn't qualify.

        Why a separate method:
            Keeping the filter logic out of the IO loop makes both halves
            unit-testable in isolation.
        """
        # Time filter.
        ts_raw = record.get("timestamp") or record.get("ts")
        if not ts_raw:
            return None
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < since:
            return None

        # Consensus threshold.
        score = float(record.get("confidence", record.get("agreement_score", 0.0)))
        if score < min_consensus:
            return None

        # Pull per-model answers.
        models = record.get("models") or []
        frontier_hits: list[tuple[str, str]] = []
        for m in models:
            name = str(m.get("name", "")).lower()
            response = str(m.get("response", "")).strip()
            error = m.get("error")
            if error or not response:
                continue
            if any(tag in name for tag in self.frontier_models):
                frontier_hits.append((name, response))

        if len(frontier_hits) < min_pair_count:
            return None

        # Canonical answer: prefer the explicit `answer` field; fall back to
        # the longest frontier response (proxy for most-complete).
        consensus_response = str(record.get("answer", "")).strip()
        if not consensus_response:
            consensus_response = max(frontier_hits, key=lambda x: len(x[1]))[1]

        query = str(record.get("prompt") or record.get("query") or "").strip()
        if not query or not consensus_response:
            return None

        return DistillationSample(
            query=query,
            consensus_response=consensus_response,
            source_models=[name for name, _ in frontier_hits],
            agreement_score=score,
            timestamp=ts.isoformat(),
        )

    # ------------------------------------------------------------------
    # Step 2 — write the Unsloth-compatible dataset
    # ------------------------------------------------------------------

    async def build_dataset(
        self,
        candidates: list[DistillationSample],
        output_path: Path | str,
    ) -> Path:
        """Write candidates to a JSONL file in Unsloth chat format.

        Why JSONL and not parquet:
            Unsloth's loader handles JSONL natively and it streams (we don't
            have to load the whole dataset into memory). Parquet would be
            denser on disk but adds a pyarrow dep on the trainer host.

        Returns:
            The absolute path to the written file (callers may need it to
            hand off to the fine-tune subprocess).
        """
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        def _write() -> None:
            with out.open("w", encoding="utf-8") as fh:
                for sample in candidates:
                    fh.write(json.dumps(sample.to_unsloth_messages(), ensure_ascii=False))
                    fh.write("\n")

        await asyncio.to_thread(_write)
        logger.info(
            "distillation.dataset_written path=%s samples=%d",
            out,
            len(candidates),
        )
        return out

    # ------------------------------------------------------------------
    # Step 3 — fine-tune (best-effort subprocess to Unsloth)
    # ------------------------------------------------------------------

    async def _run_finetune(
        self,
        dataset_path: Path,
        version: str,
    ) -> Path | None:
        """Spawn the Unsloth CLI in a subprocess; skip cleanly if absent.

        Why subprocess and not a Python import:
            Unsloth pulls in torch + CUDA toolchain. Importing it into the
            Quorum process would balloon memory and tie our test environment
            to GPU-class machines. A subprocess keeps Quorum lightweight.

        Returns:
            The path to the produced checkpoint directory, or None if Unsloth
            isn't installed (caller should treat that as "no new candidate").
        """
        unsloth_bin = shutil.which("unsloth") or os.getenv("UNSLOTH_BIN")
        if not unsloth_bin:
            logger.warning(
                "distillation.finetune_skipped reason=unsloth_not_installed "
                "dataset=%s version=%s",
                dataset_path,
                version,
            )
            return None

        ckpt_dir = self.artifacts_dir / f"checkpoint-{version}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            unsloth_bin,
            "train",
            "--dataset",
            str(dataset_path),
            "--output",
            str(ckpt_dir),
        ]
        logger.info("distillation.finetune_start cmd=%s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(
                "distillation.finetune_failed rc=%d stderr=%s",
                proc.returncode,
                stderr.decode("utf-8", errors="replace")[:500],
            )
            return None

        logger.info(
            "distillation.finetune_done rc=0 ckpt=%s stdout_tail=%s",
            ckpt_dir,
            stdout.decode("utf-8", errors="replace")[-200:],
        )
        return ckpt_dir

    # ------------------------------------------------------------------
    # Step 4 — benchmark on a held-out eval set
    # ------------------------------------------------------------------

    async def _run_benchmark(
        self,
        version: str,
        ckpt_dir: Path | None,
    ) -> BenchmarkResult:
        """Score a candidate checkpoint against a fixed eval set.

        Why "fixed" eval set:
            If the eval set drifts with the training data we lose any signal
            about regression. The file at `self.eval_set_path` is treated as
            an immutable contract; CI should fail any PR that modifies it
            without an explicit rotation note.

        Fallback:
            When no eval set is present (dev machine, first install), we
            return a neutral score that will block promotion. The HSP gate
            still has final say — but the default must be "do not promote".
        """
        if not self.eval_set_path.exists():
            logger.warning(
                "distillation.benchmark_no_evalset path=%s — returning blocking score",
                self.eval_set_path,
            )
            return BenchmarkResult(
                version=version,
                accuracy=0.0,
                safety_score=0.0,
                avg_latency_ms=0.0,
                samples_evaluated=0,
            )

        # Real eval is delegated to a separate evaluator binary (out of scope
        # here). We emit the path so an external job can run; for now we read
        # any sidecar JSON the evaluator may have produced previously.
        sidecar = self.artifacts_dir / f"bench-{version}.json"
        if sidecar.exists():
            data = json.loads(await asyncio.to_thread(sidecar.read_text, "utf-8"))
            return BenchmarkResult(
                version=version,
                accuracy=float(data.get("accuracy", 0.0)),
                safety_score=float(data.get("safety_score", 0.0)),
                avg_latency_ms=float(data.get("avg_latency_ms", 0.0)),
                samples_evaluated=int(data.get("samples_evaluated", 0)),
                extra={
                    k: float(v)
                    for k, v in data.get("extra", {}).items()
                    if isinstance(v, (int, float))
                },
            )

        logger.info(
            "distillation.benchmark_pending version=%s ckpt=%s — evaluator must run",
            version,
            ckpt_dir,
        )
        return BenchmarkResult(
            version=version,
            accuracy=0.0,
            safety_score=0.0,
            avg_latency_ms=0.0,
            samples_evaluated=0,
        )

    # ------------------------------------------------------------------
    # Step 5 — promote (HSP-gated)
    # ------------------------------------------------------------------

    @requires_hsp_approval(action="promote_llama_checkpoint", risk_level="high")
    async def promote_checkpoint(
        self,
        version: str,
        dataset_path: Path | str,
        benchmark_results: BenchmarkResult,
    ) -> bool:
        """Promote a candidate to the production Llama slot.

        This is THE function whose misuse breaks the whole system: it swaps
        the model that millions of downstream queries will hit. Two safeguards:

            1. HSP gate (decorator): a human or certified webhook must say yes.
            2. Hard regression check below: even if HSP approves, we still
               refuse to promote a checkpoint that lost ground on the eval set.

        The gate is fail-closed: any error in the webhook call aborts promotion.

        Returns:
            True if promoted, False if rejected for regression. Raises
            HSPGateDenied (from the decorator) if the gate refuses.
        """
        baseline = await self._load_baseline_benchmark()

        if benchmark_results.samples_evaluated == 0:
            logger.error(
                "distillation.promote_rejected reason=no_benchmark version=%s",
                version,
            )
            return False

        if baseline is not None:
            acc_delta = benchmark_results.accuracy - baseline.accuracy
            safety_delta = benchmark_results.safety_score - baseline.safety_score
            if acc_delta < self.min_improvement or safety_delta < 0:
                logger.error(
                    "distillation.promote_rejected reason=regression "
                    "version=%s acc_delta=%.4f safety_delta=%.4f",
                    version,
                    acc_delta,
                    safety_delta,
                )
                return False

        # Atomic-ish swap: write a "current" pointer file. The serving layer
        # reads this on next request. We never overwrite the old checkpoint
        # so a rollback is a one-line edit.
        pointer = self.artifacts_dir / "current.json"
        payload = {
            "version": version,
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": asdict(benchmark_results),
        }
        await asyncio.to_thread(
            pointer.write_text, json.dumps(payload, indent=2), "utf-8"
        )
        logger.info(
            "distillation.promote_success version=%s acc=%.4f safety=%.4f",
            version,
            benchmark_results.accuracy,
            benchmark_results.safety_score,
        )
        return True

    async def _load_baseline_benchmark(self) -> BenchmarkResult | None:
        """Return the currently-promoted checkpoint's benchmark, if any.

        Why this matters:
            "No regression vs current" is the contract. If there is no current
            checkpoint (first promotion ever), any non-zero benchmark passes,
            and we let the HSP gate be the sole safeguard.
        """
        pointer = self.artifacts_dir / "current.json"
        if not pointer.exists():
            return None
        try:
            data = json.loads(await asyncio.to_thread(pointer.read_text, "utf-8"))
            b = data.get("benchmark", {})
            return BenchmarkResult(
                version=str(data.get("version", "unknown")),
                accuracy=float(b.get("accuracy", 0.0)),
                safety_score=float(b.get("safety_score", 0.0)),
                avg_latency_ms=float(b.get("avg_latency_ms", 0.0)),
                samples_evaluated=int(b.get("samples_evaluated", 0)),
            )
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning("distillation.baseline_unreadable err=%s", e)
            return None

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    async def run_nightly(
        self,
        *,
        since: datetime,
        min_consensus: float = 0.85,
        min_pair_count: int = 3,
    ) -> dict[str, Any]:
        """End-to-end nightly run. Designed to be cron-safe (idempotent-ish).

        We return a structured summary instead of raising so that the cron
        wrapper can log a single line per night and surface failures to
        observability without crashing the scheduler.
        """
        version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        summary: dict[str, Any] = {
            "version": version,
            "candidates": 0,
            "dataset_path": None,
            "checkpoint": None,
            "benchmark": None,
            "promoted": False,
            "error": None,
        }
        try:
            candidates = await self.collect_distillation_candidates(
                since=since,
                min_consensus=min_consensus,
                min_pair_count=min_pair_count,
            )
            summary["candidates"] = len(candidates)
            if not candidates:
                logger.info("distillation.nightly_noop reason=no_candidates")
                return summary

            dataset_path = await self.build_dataset(
                candidates,
                self.artifacts_dir / f"dataset-{version}.jsonl",
            )
            summary["dataset_path"] = str(dataset_path)

            ckpt = await self._run_finetune(dataset_path, version)
            summary["checkpoint"] = str(ckpt) if ckpt else None
            if ckpt is None:
                # Without a fresh checkpoint there is nothing to promote.
                # That's a clean no-op, not an error.
                return summary

            bench = await self._run_benchmark(version, ckpt)
            summary["benchmark"] = asdict(bench)

            try:
                promoted = await self.promote_checkpoint(version, dataset_path, bench)
                summary["promoted"] = bool(promoted)
            except HSPGateDenied as e:
                logger.warning("distillation.promote_denied err=%s", e)
                summary["error"] = f"hsp_denied: {e}"
        except Exception as e:  # noqa: BLE001
            logger.exception("distillation.nightly_failed")
            summary["error"] = str(e)
        return summary


# ---------------------------------------------------------------------------
# Smoke tests — easy to wire into pytest, runnable as a script.
# ---------------------------------------------------------------------------


async def _smoke_test_collect(tmp_dir: Path) -> None:
    """Smoke test: write a synthetic log, ensure the pipeline picks the right rows.

    We use deliberately mixed inputs — some too old, some below threshold, one
    valid — to exercise the filter logic end to end.
    """
    log = tmp_dir / "queries.jsonl"
    now = datetime.now(timezone.utc)
    records = [
        # Valid: high agreement, recent, 3 frontier models.
        {
            "timestamp": now.isoformat(),
            "prompt": "What is the capital of France?",
            "answer": "Paris.",
            "confidence": 0.95,
            "models": [
                {"name": "anthropic", "response": "Paris."},
                {"name": "openai", "response": "Paris."},
                {"name": "gemini", "response": "Paris."},
            ],
        },
        # Rejected: agreement below threshold.
        {
            "timestamp": now.isoformat(),
            "prompt": "Best programming language?",
            "answer": "Python",
            "confidence": 0.40,
            "models": [
                {"name": "anthropic", "response": "Python"},
                {"name": "openai", "response": "Rust"},
                {"name": "gemini", "response": "Go"},
            ],
        },
        # Rejected: only 2 frontier hits (one is local Llama).
        {
            "timestamp": now.isoformat(),
            "prompt": "2+2?",
            "answer": "4",
            "confidence": 0.99,
            "models": [
                {"name": "anthropic", "response": "4"},
                {"name": "openai", "response": "4"},
                {"name": "ollama-llama", "response": "4"},
            ],
        },
    ]
    log.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    pipe = DistillationPipeline(
        log_path=log,
        eval_set_path=tmp_dir / "eval.jsonl",
        artifacts_dir=tmp_dir / "artifacts",
    )
    samples = await pipe.collect_distillation_candidates(
        since=now.replace(hour=0, minute=0, second=0, microsecond=0),
        min_consensus=0.85,
        min_pair_count=3,
    )
    assert len(samples) == 1, f"expected 1 sample, got {len(samples)}"
    assert samples[0].query.startswith("What is the capital"), samples[0].query
    logger.info("smoke_test_collect: ok")


async def _smoke_test_build_dataset(tmp_dir: Path) -> None:
    """Smoke test: round-trip a sample through the JSONL writer."""
    sample = DistillationSample(
        query="ping",
        consensus_response="pong",
        source_models=["anthropic", "openai", "gemini"],
        agreement_score=0.92,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    pipe = DistillationPipeline(
        log_path=tmp_dir / "queries.jsonl",
        eval_set_path=tmp_dir / "eval.jsonl",
        artifacts_dir=tmp_dir / "artifacts",
    )
    out = await pipe.build_dataset([sample], tmp_dir / "out.jsonl")
    text = out.read_text(encoding="utf-8").strip()
    parsed = json.loads(text)
    assert parsed["messages"][0]["content"] == "ping"
    assert parsed["messages"][1]["content"] == "pong"
    logger.info("smoke_test_build_dataset: ok")


async def _smoke_test_promote_blocks_without_benchmark(tmp_dir: Path) -> None:
    """Smoke test: promotion must refuse when benchmark has zero samples.

    This guards against the "first run on a fresh machine silently promotes a
    half-trained model" foot-gun.
    """
    pipe = DistillationPipeline(
        log_path=tmp_dir / "queries.jsonl",
        eval_set_path=tmp_dir / "eval.jsonl",
        artifacts_dir=tmp_dir / "artifacts",
    )
    empty_bench = BenchmarkResult(
        version="v0",
        accuracy=0.0,
        safety_score=0.0,
        avg_latency_ms=0.0,
        samples_evaluated=0,
    )
    # HSP_GATE_WEBHOOK is unset in tests → decorator passes through, so the
    # inner regression check is what we're exercising here.
    promoted = await pipe.promote_checkpoint("v0", tmp_dir / "ds.jsonl", empty_bench)
    assert promoted is False
    logger.info("smoke_test_promote_blocks_without_benchmark: ok")


async def _run_all_smoke_tests() -> None:
    import tempfile

    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        await _smoke_test_collect(tmp)
        await _smoke_test_build_dataset(tmp)
        await _smoke_test_promote_blocks_without_benchmark(tmp)


if __name__ == "__main__":
    asyncio.run(_run_all_smoke_tests())


__all__ = [
    "DistillationPipeline",
    "DistillationSample",
    "BenchmarkResult",
]
