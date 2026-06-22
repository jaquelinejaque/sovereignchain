"""CLI commands for the canonical eval set + evaluator.

Sibling of ``cli_responses`` — lives in its own module so the main
``cli.py`` (which has unrelated WIP) is not touched. Wire-in pattern
is the same: a typer sub-app the operator can hang under ``quorum eval``
when they want it.

Usage examples (with sub-app mounted under ``quorum``)::

    quorum eval install                       # write the canonical
                                              # eval_set.jsonl into
                                              # ~/.quorum/

    quorum eval show --class factual          # print pinned items

    quorum eval run --version smoke-1         # run the canonical set
                                              # against the echo
                                              # responder; write a
                                              # bench-smoke-1.json
                                              # sidecar so distillation
                                              # can read it

    quorum eval hash                          # print sha256 of the
                                              # pinned set (drift
                                              # tripwire for CI)

All commands are read-only on user data except ``run`` (writes the
sidecar) and ``install`` (writes the eval set file). Neither touches
production or anything that requires explicit OK.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

from quorum.evolution import eval_set

app = typer.Typer(
    help=(
        "Manage the canonical distillation eval set and run benchmarks.\n\n"
        "The eval set is a pinned 50-item, 6-class JSONL the distillation "
        "pipeline reads at promote time. These commands install it, show "
        "what's in it, hash it for CI drift detection, and run a benchmark "
        "(with optional sidecar emission) without needing the production "
        "Provider stack."
    ),
    no_args_is_help=True,
)


@app.command("install")
def cmd_install(
    path: Path = typer.Option(
        Path.home() / ".quorum" / "eval_set.jsonl",
        "--path", "-p",
        help="Where to write the JSONL file.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="Replace any existing non-canonical file at the path.",
    ),
) -> None:
    """Write the canonical eval set to disk.

    Idempotent — re-running on an already-canonical file is a no-op.
    A custom file present at the path is left alone unless
    ``--overwrite`` is passed (so an operator who placed their own
    eval set there doesn't lose it).
    """
    out = eval_set.write_default_eval_set(path, overwrite=overwrite)
    typer.echo(f"eval_set installed: {out}")
    typer.echo(f"items: {len(eval_set.CANONICAL_EVAL_SET)}")
    typer.echo(f"sha256: {eval_set.canonical_eval_set_sha256()}")


@app.command("show")
def cmd_show(
    query_class: str | None = typer.Option(
        None, "--class", "-c",
        help="Filter to one query class (general / code / factual / "
             "legal / security / creative).",
    ),
    json_out: bool = typer.Option(
        False, "--json",
        help="Emit JSONL instead of pretty text.",
    ),
) -> None:
    """Print the canonical eval items, optionally filtered by class."""
    items = [
        it for it in eval_set.CANONICAL_EVAL_SET
        if query_class is None or it.query_class == query_class
    ]
    if not items:
        typer.echo(f"no items match class={query_class!r}", err=True)
        raise typer.Exit(code=1)
    if json_out:
        for it in items:
            typer.echo(json.dumps({
                "id": it.id,
                "query_class": it.query_class,
                "prompt": it.prompt,
                "expected_keywords": list(it.expected_keywords),
                "must_refuse": it.must_refuse,
            }, ensure_ascii=False))
        return
    typer.echo(f"{'id':>5}  {'class':<10}  prompt")
    typer.echo("-" * 80)
    for it in items:
        mark = "  ⛔" if it.must_refuse else ""
        typer.echo(f"{it.id:>5}  {it.query_class:<10}  {it.prompt[:60]}{mark}")


@app.command("hash")
def cmd_hash() -> None:
    """Print the SHA-256 of the canonical eval set.

    Use this in CI as a drift tripwire — any code change that mutates
    ``CANONICAL_EVAL_SET`` will change this digest, forcing a
    deliberate update of the pinned value.
    """
    typer.echo(eval_set.canonical_eval_set_sha256())


@app.command("run")
def cmd_run(
    version: str = typer.Option(
        ..., "--version", "-v",
        help="Identifier embedded in the sidecar name (bench-<version>.json).",
    ),
    sidecar: Path | None = typer.Option(
        None, "--sidecar", "-o",
        help="Where to write the sidecar JSON. Defaults to "
             "~/.quorum/distillation/bench-<version>.json — the path "
             "DistillationPipeline._run_benchmark already reads.",
    ),
    query_class: str | None = typer.Option(
        None, "--class", "-c",
        help="Restrict the run to a single query class.",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q",
        help="Suppress per-item progress output.",
    ),
) -> None:
    """Run the eval set against the deterministic echo responder.

    Useful as a smoke test (does the evaluator pipeline produce a
    valid sidecar?) without needing the production Provider keys.
    For a real model run, callers should import ``evaluate_checkpoint``
    directly and pass a Provider-backed responder — wiring a Provider
    here would force a hard dependency on the entire provider stack
    just to inspect the eval set.
    """
    if sidecar is None:
        sidecar = (
            Path.home() / ".quorum" / "distillation" / f"bench-{version}.json"
        )

    items = list(eval_set.CANONICAL_EVAL_SET)
    if query_class is not None:
        items = [it for it in items if it.query_class == query_class]
        if not items:
            typer.echo(f"no items in class={query_class!r}", err=True)
            raise typer.Exit(code=1)

    if not quiet:
        typer.echo(f"running {len(items)} items against echo responder ...", err=True)

    report = asyncio.run(
        eval_set.evaluate_checkpoint(
            version=version,
            eval_set=items,
            sidecar_path=sidecar,
        )
    )

    typer.echo(json.dumps({
        "version": report.version,
        "samples_evaluated": report.samples_evaluated,
        "accuracy": round(report.accuracy, 4),
        "safety_score": round(report.safety_score, 4),
        "avg_latency_ms": round(report.avg_latency_ms, 2),
        "sidecar": str(sidecar),
        "eval_set_sha": report.eval_set_sha,
    }, indent=2))


__all__ = ["app"]
