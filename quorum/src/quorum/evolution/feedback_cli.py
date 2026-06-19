"""CLI for staging Quorum consensus calls and confirming RLHF feedback."""
from __future__ import annotations

import asyncio
import json
import sys
import time

import typer

from quorum.evolution.auto_feedback import PendingStore, emit_call_id

app = typer.Typer(help="Stage and confirm Quorum consensus feedback events.")


@app.command("stage")
def stage(
    prompt: str = typer.Option(..., "--prompt", help="Original prompt string."),
    user_id: str = typer.Option("jaqueline", "--user-id"),
) -> None:
    """Read ConsensusResult.to_dict() JSON from stdin, stage it, emit call_id."""
    raw = sys.stdin.read()
    result = json.loads(raw)
    call_id = emit_call_id(prompt, user_id)
    models = [
        {
            "name": m.get("name"),
            "weight": m.get("weight"),
            "latency": m.get("latency"),
            "cost": m.get("cost"),
        }
        for m in result.get("models", [])
    ]
    answer = result.get("answer", "") or ""
    payload = {
        "staged_at": time.time(),
        "user_id": user_id,
        "prompt": prompt,
        "chosen_model": result.get("chosen_model"),
        "models": models,
        "answer_excerpt": answer[:500],
    }
    PendingStore().add(call_id, payload)
    typer.echo(call_id)


def _record(payload: dict, rating: int, reason: str | None) -> None:
    from quorum.evolution.rlhf import RLHFTracker  # lazy import

    tracker = RLHFTracker()
    coro = tracker.record_feedback(
        payload["user_id"],
        payload["prompt"],
        payload["chosen_model"],
        payload["models"],
        rating,
    )
    asyncio.run(coro)


@app.command("confirm")
def confirm(
    call_id: str = typer.Argument(...),
    rating: int = typer.Option(..., "--rating"),
    reason: str | None = typer.Option(None, "--reason"),
) -> None:
    if rating not in (1, -1, 0):
        typer.echo("rating must be one of 1, -1, 0", err=True)
        raise typer.Exit(code=1)
    payload = PendingStore().pop(call_id)
    if payload is None:
        typer.echo("call_id not found", err=True)
        raise typer.Exit(code=1)
    _record(payload, rating, reason)
    typer.echo(f"ok {call_id}")


@app.command("sweep")
def sweep(
    max_age_sec: int = typer.Option(900, "--max-age-sec"),
    default_rating: int = typer.Option(1, "--default-rating"),
) -> None:
    store = PendingStore()
    now = time.time()
    swept = 0
    for call_id, payload in list(store.all().items()):
        if now - payload.get("staged_at", now) > max_age_sec:
            _record(payload, default_rating, None)
            store.pop(call_id)
            swept += 1
    typer.echo(str(swept))


@app.command("list")
def list_pending() -> None:
    now = time.time()
    for call_id, payload in PendingStore().all().items():
        age = int(now - payload.get("staged_at", now))
        typer.echo(f"{call_id} {age}")


if __name__ == "__main__":
    app()
