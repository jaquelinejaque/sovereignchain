"""Quorum CLI."""

from __future__ import annotations

import asyncio
import json

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from quorum.core.consensus import consensus

app = typer.Typer(no_args_is_help=True, help="Quorum — multi-LLM consensus engine")
console = Console()


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Your question"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Run a query across all configured LLMs and print the consensus."""
    result = asyncio.run(consensus(prompt))

    if json_output:
        console.print_json(json.dumps(result.to_dict()))
        return

    # Pretty output
    console.print(
        Panel(
            result.answer,
            title=f"[bold green]Consensus[/] · confidence {result.confidence:.0%}",
            border_style="green",
            box=box.ROUNDED,
        )
    )

    tbl = Table(title="Models", box=box.SIMPLE_HEAD)
    tbl.add_column("Model", style="cyan")
    tbl.add_column("Weight", justify="right")
    tbl.add_column("Latency", justify="right")
    tbl.add_column("Tokens", justify="right")
    tbl.add_column("Cost (USD)", justify="right")
    tbl.add_column("Status")
    for m in result.models:
        status = "✅" if not m.error else f"❌ {m.error[:30]}"
        tbl.add_row(
            m.name,
            f"{m.weight:.2f}" if m.weight else "—",
            f"{m.latency_ms:.0f}ms",
            f"{m.tokens_in}/{m.tokens_out}",
            f"${m.cost_usd:.6f}",
            status,
        )
    console.print(tbl)

    if result.disagreements:
        console.print(
            f"[yellow]⚠ Dissenters: {', '.join(result.disagreements)}[/]"
        )

    console.print(
        f"[dim]Total: ${result.total_cost_usd:.6f} · "
        f"{result.total_latency_ms:.0f}ms · "
        f"{len([m for m in result.models if not m.error])}/{len(result.models)} models OK[/]"
    )


@app.command()
def version():
    """Print version info."""
    from quorum import __version__
    console.print(f"Quorum v{__version__}")
    console.print("[dim]Patent: PCT/US26/11908 (HSP Protocol)[/]")
    console.print("[dim]https://github.com/jaquelinejaque/sovereignchain[/]")


if __name__ == "__main__":
    app()
