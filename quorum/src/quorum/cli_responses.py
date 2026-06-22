"""CLI commands for the opt-in response_log store.

Kept in a SEPARATE module from ``quorum.cli`` so the main CLI file (which
has unrelated WIP) stays untouched. Wired in via ``cli.py`` import only
when the operator runs ``quorum responses ...``.

Usage::

    QUORUM_LOG_RESPONSES=1 quorum responses stats
    quorum responses export --since 2026-06-01 --out responses.jsonl
    quorum responses vacuum --older-than-days 90

The module is import-safe with or without the env flag; commands are
read-only except ``vacuum`` which deletes rows.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import typer

from quorum.evolution import response_log

app = typer.Typer(
    help=(
        "Inspect and export the opt-in raw response log.\n\n"
        "Activate logging by setting QUORUM_LOG_RESPONSES=1 before running\n"
        "any consensus call. With the flag unset this store is a no-op and\n"
        "these commands return empty results."
    ),
    no_args_is_help=True,
)


def _parse_when(when: str | None) -> float | None:
    """Accept ``YYYY-MM-DD``, ``YYYY-MM-DDTHH:MM:SS``, or a unix-seconds string."""
    if when is None:
        return None
    when = when.strip()
    if not when:
        return None
    # Plain unix seconds?
    try:
        return float(when)
    except ValueError:
        pass
    # ISO date / datetime
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(when, fmt).timestamp()
        except ValueError:
            continue
    raise typer.BadParameter(
        f"Could not parse {when!r}. Use YYYY-MM-DD, ISO timestamp, or unix seconds."
    )


@app.command("stats")
def cmd_stats() -> None:
    """Quick overview: row count, distinct models, date range."""
    s = response_log.stats()
    typer.echo(json.dumps(s, indent=2, default=str))


@app.command("export")
def cmd_export(
    since: str | None = typer.Option(
        None, "--since", help="Lower-bound timestamp (YYYY-MM-DD or unix seconds)."
    ),
    until: str | None = typer.Option(
        None, "--until", help="Upper-bound timestamp (YYYY-MM-DD or unix seconds)."
    ),
    model: str | None = typer.Option(
        None, "--model", help="Restrict to a single model name."
    ),
    out: Path | None = typer.Option(
        None, "--out", "-o", help="Write JSONL here. Default: stdout."
    ),
) -> None:
    """Stream the log as JSONL — one row per line, ready for jq / pandas."""
    since_ts = _parse_when(since)
    until_ts = _parse_when(until)
    sink = out.open("w") if out else sys.stdout
    try:
        count = 0
        for row in response_log.export_jsonl(
            since=since_ts, until=until_ts, model=model
        ):
            sink.write(json.dumps(row, default=str) + "\n")
            count += 1
        if out:
            typer.echo(f"wrote {count} rows -> {out}", err=True)
    finally:
        if out:
            sink.close()


@app.command("vacuum")
def cmd_vacuum(
    older_than_days: int = typer.Option(
        ..., "--older-than-days", help="Delete rows older than this many days."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip the confirmation prompt."
    ),
) -> None:
    """Delete old rows and reclaim disk. Operator-controlled retention."""
    if older_than_days <= 0:
        raise typer.BadParameter("--older-than-days must be a positive integer.")
    seconds = older_than_days * 24 * 3600
    if not yes:
        typer.confirm(
            f"Delete every response_log row older than {older_than_days} days?",
            abort=True,
        )
    deleted = response_log.vacuum_older_than(seconds=seconds)
    typer.echo(f"deleted {deleted} rows")


__all__ = ["app"]
