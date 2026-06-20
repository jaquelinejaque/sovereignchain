"""quorum-audit CLI — verify and export the HSP Black Box chain.

Separate binary (entry_point in pyproject.toml) to avoid touching cli.py
which carries unrelated WIP.
"""
from __future__ import annotations
import json
import sys
import typer
from quorum.hsp import black_box

app = typer.Typer(help="Quorum HSP Black Box — tamper-evident audit log")


@app.command("status")
def status_cmd() -> None:
    """Show audit chain summary."""
    s = black_box.stats()
    typer.echo(json.dumps(s, indent=2))


@app.command("verify-chain")
def verify_cmd() -> None:
    """Walk the chain, recompute hashes, prove integrity. Exit 0 if ok, 2 if broken."""
    ok, broken_at = black_box.verify_chain()
    if ok:
        s = black_box.stats()
        typer.echo(f"OK — {s.get('count', 0)} records, chain intact")
        raise typer.Exit(code=0)
    typer.echo(f"BROKEN at id={broken_at} — possible tampering or schema corruption", err=True)
    raise typer.Exit(code=2)


@app.command("export")
def export_cmd(
    since: str = typer.Option("", "--since", help="ISO timestamp (e.g. 2026-01-01T00:00:00Z)"),
    out: str = typer.Option("", "--out", help="Output path (default ~/.quorum/audit_export_<ts>.jsonl)"),
) -> None:
    """Export chain to JSONL for an external auditor."""
    n = black_box.export_jsonl(since_iso=since or None, out_path=out or None)
    typer.echo(f"Exported {n} rows")


@app.command("append")
def append_cmd(
    payload: str = typer.Argument(..., help="JSON payload string"),
) -> None:
    """Manually append a record. Useful for testing."""
    try:
        obj = json.loads(payload)
    except Exception as e:
        typer.echo(f"Invalid JSON: {e}", err=True)
        raise typer.Exit(code=1)
    h = black_box.append(obj)
    if h:
        typer.echo(h)
    else:
        typer.echo("APPEND FAILED — check logs", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
