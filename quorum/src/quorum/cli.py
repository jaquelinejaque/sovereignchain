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
from quorum.doctor import run_doctor

app = typer.Typer(no_args_is_help=True, help="Quorum — multi-LLM consensus engine")
console = Console()


@app.command()
def doctor():
    """Detect cross-config conflicts (pyproject ↔ Dockerfile ↔ env) before deploy."""
    raise typer.Exit(code=run_doctor())


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="Your question"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
    all_providers: bool = typer.Option(
        False, "--all", "--no-router",
        help="Skip MoE router; fan out to every configured provider (use for demos)",
    ),
    budget: float = typer.Option(
        0.05, "--budget", help="Hard cap on total spend per query (USD)",
    ),
    web: bool = typer.Option(
        True, "--web/--no-web",
        help="Fetch live web context (DuckDuckGo, no key) and inject into every model's prompt. ON by default since 2026-06-17 — bypasses training cutoffs.",
    ),
    web_n: int = typer.Option(
        5, "--web-n", help="Number of web results to inject when --web is set",
    ),
    recall: bool = typer.Option(
        False, "--recall",
        help="Inject top-k learned facts from the local KB (built with `quorum learn`).",
    ),
    recall_k: int = typer.Option(
        5, "--recall-k", help="Number of facts to recall when --recall is set",
    ),
    reframe_ctx: str = typer.Option(
        None, "--reframe",
        help="Wrap prompt in authorized-research context. Values: research, bug-bounty, ctf, red-team. Records every wrap in $QUORUM_DATA_DIR/reframe_audit/.",
    ),
    no_context: bool = typer.Option(
        False, "--no-context",
        help="Skip injection of the active context profile (see `quorum context use`). Default: injects whatever is set as active.",
    ),
):
    """Run a query across all configured LLMs and print the consensus."""
    # Active context profile (set via `quorum context use NAME`) — injected first
    # so models treat it as ground truth before any per-query reframe / web / recall.
    if not no_context:
        from quorum.context import current_name, load_active_context, wrap_prompt_with_context
        active = current_name()
        if active:
            ctx_body = load_active_context()
            if ctx_body:
                prompt = wrap_prompt_with_context(prompt, ctx_body)
                if not json_output:
                    console.print(
                        f"[dim]📎 context profile: '{active}' ({len(ctx_body)} chars injected)[/]"
                    )

    # Reframe pass (applied to the original prompt before any --web/--recall)
    if reframe_ctx:
        from quorum.agents.reframe import reframe
        rec = reframe(prompt, reframe_ctx, audit=True)
        if not json_output:
            console.print(
                f"[dim]🎓 reframe={reframe_ctx} (+{len(rec.framing_text)} chars context).  "
                f"audit={rec.audit_path}[/]"
            )
        prompt = rec.wrapped_prompt
    extra_context_parts = []
    if recall:
        from quorum.evolution.web_learner import recall_as_context
        kb_ctx = asyncio.run(recall_as_context(prompt, top_k=recall_k))
        if kb_ctx:
            extra_context_parts.append(kb_ctx)
            if not json_output:
                console.print(
                    f"[dim]🧠 KB recall: {len(kb_ctx)} chars from {recall_k} learned facts[/]"
                )
        else:
            if not json_output:
                console.print("[dim yellow]🧠 KB recall: nothing learned yet — use `quorum learn --topic '...'` first[/]")
    if web:
        from quorum.web import search_to_context
        ctx = search_to_context(prompt, n=web_n)
        extra_context_parts.append(ctx)
        if not json_output:
            console.print(
                f"[dim]🌐 web context: {len(ctx)} chars injected (DuckDuckGo, no key)[/]"
            )
    if extra_context_parts:
        prompt_with_web = (
            "\n\n".join(extra_context_parts)
            + f"\n\nUSER QUESTION (answer using the context above when relevant):\n{prompt}"
        )
    else:
        prompt_with_web = prompt

    if all_providers:
        from quorum.providers.registry import load_default_providers
        providers = load_default_providers()
        result = asyncio.run(consensus(
            prompt_with_web, providers=providers, budget_usd=10.0, route=False,
        ))
    else:
        result = asyncio.run(consensus(prompt_with_web, budget_usd=budget))

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


# --------- Camada 1: agent draft commands ---------

draft_app = typer.Typer(no_args_is_help=True, help="Quorum-generated drafts (human-approval gated, NEVER published).")
app.add_typer(draft_app, name="draft")


def _print_draft_result(kind: str, d):
    console.print(
        Panel(
            d.content,
            title=f"[bold green]Draft · {kind}[/] · "
                  f"confidence {d.confidence:.0%} · {d.models_ok}/{d.models_total} OK · "
                  f"${d.cost_usd:.4f} · {d.latency_ms:.0f}ms",
            border_style="green",
            box=box.ROUNDED,
        )
    )


@draft_app.command("linkedin")
def draft_linkedin(no_web: bool = typer.Option(False, "--no-web")):
    """Draft a LinkedIn post for Quorum."""
    from quorum.agents.drafts import draft_linkedin_post
    d = asyncio.run(draft_linkedin_post(use_web=not no_web))
    _print_draft_result("linkedin", d)


@draft_app.command("twitter")
def draft_twitter(no_web: bool = typer.Option(False, "--no-web")):
    """Draft an 8-tweet Twitter/X thread for Quorum."""
    from quorum.agents.drafts import draft_twitter_thread
    d = asyncio.run(draft_twitter_thread(use_web=not no_web))
    _print_draft_result("twitter", d)


@draft_app.command("show-hn")
def draft_show_hn_cmd(no_web: bool = typer.Option(False, "--no-web")):
    """Draft a Show HN submission for Quorum."""
    from quorum.agents.drafts import draft_show_hn
    d = asyncio.run(draft_show_hn(use_web=not no_web))
    _print_draft_result("show_hn", d)


@draft_app.command("email")
def draft_email(no_web: bool = typer.Option(False, "--no-web")):
    """Draft a cold outreach email."""
    from quorum.agents.drafts import draft_email_outreach
    d = asyncio.run(draft_email_outreach(use_web=not no_web))
    _print_draft_result("email", d)


@draft_app.command("vscode-listing")
def draft_vscode(no_web: bool = typer.Option(False, "--no-web")):
    """Draft an upgraded VS Code Marketplace listing."""
    from quorum.agents.drafts import draft_vscode_listing
    d = asyncio.run(draft_vscode_listing(use_web=not no_web))
    _print_draft_result("vscode_listing", d)


@app.command("sell-quorum")
def sell_quorum_cmd(
    output_dir: str = typer.Option(
        None, "--out",
        help="Where to write the draft bundle (default ~/Desktop/quorum-drafts)",
    ),
    only: str = typer.Option(
        None, "--only",
        help="Comma-separated subset: linkedin,twitter,show_hn,email,vscode_listing",
    ),
    no_web: bool = typer.Option(False, "--no-web", help="Skip live web context"),
):
    """One-shot: generate ALL 5 drafts (or a subset) in parallel and write a bundle to disk."""
    from quorum.agents.drafts import sell_quorum
    only_list = [s.strip() for s in only.split(",")] if only else None
    bundle = asyncio.run(sell_quorum(output_dir=output_dir, only=only_list, use_web=not no_web))
    tbl = Table(title=f"Quorum draft bundle → {bundle.output_dir}", box=box.SIMPLE_HEAD)
    tbl.add_column("Draft", style="cyan")
    tbl.add_column("Conf", justify="right")
    tbl.add_column("Models OK", justify="right")
    tbl.add_column("Cost (USD)", justify="right")
    tbl.add_column("Latency", justify="right")
    for d in bundle.drafts:
        tbl.add_row(
            d.kind,
            f"{d.confidence:.0%}",
            f"{d.models_ok}/{d.models_total}",
            f"${d.cost_usd:.4f}",
            f"{d.latency_ms:.0f}ms",
        )
    console.print(tbl)
    console.print(
        f"[dim]Total cost (parallel): ${bundle.total_cost_usd:.4f}  · "
        f"Wall-clock: {bundle.total_latency_ms:.0f}ms[/]"
    )
    console.print(
        f"[green]✓[/] Review each draft in [bold]{bundle.output_dir}[/] before publishing. "
        f"Nothing has been published."
    )


# --------- Camada 2: publishing, metrics, autopilot, tools ---------

publish_app = typer.Typer(no_args_is_help=True, help="Publish drafts via APIs / browser. --dry-run default.")
app.add_typer(publish_app, name="publish")


@publish_app.command("twitter")
def publish_twitter(
    file: str = typer.Argument(..., help="Path to a markdown file containing the thread (numbered 1/8 ... 8/8)."),
    really_post: bool = typer.Option(False, "--really-post", help="Actually publish (default: dry-run preview only)."),
):
    """Post a Twitter/X thread from a markdown file. DEFAULT IS DRY-RUN."""
    import re
    from quorum.agents.tools import default_registry, load_builtins

    load_builtins()
    raw = open(file, "r", encoding="utf-8").read()
    # Split by numbered tweet markers like "1/8", "1/", or markdown-bold "**1/8**"
    tweets = re.split(r"\n(?=\*{0,2}\d+\s*/\s*\d+)", raw)
    tweets = [t.strip() for t in tweets if t.strip()]
    # Strip trailing "---" separators that sit between tweets in the source file
    tweets = [re.sub(r"\n*---\s*$", "", t).strip() for t in tweets]
    tweets = [t for t in tweets if t and not t.startswith("# ") and not t.startswith("_")]
    # Strip the leading "**N/M**" or "N/M" tracker — useful for the writer but
    # noise in the actual post (Twitter renders `**1/8**` as literal asterisks).
    tweets = [re.sub(r"^\*{0,2}\d+\s*/\s*\d+\*{0,2}\s*\n+", "", t).strip() for t in tweets]
    if not tweets:
        console.print("[red]No tweets found in file.[/]")
        raise typer.Exit(1)

    console.print(f"[dim]Parsed {len(tweets)} tweets from {file}[/]")
    for i, t in enumerate(tweets, 1):
        console.print(f"  [{i}] {t[:120]}{'…' if len(t) > 120 else ''}")

    reg = default_registry()
    result = asyncio.run(reg.call("twitter.post_thread", dry_run=not really_post, tweets=tweets))
    console.print_json(json.dumps(result))

    # If we actually posted, record provenance
    if really_post and result.get("posted"):
        root_id = result["root_id"]
        asyncio.run(reg.call("metrics.record_post", dry_run=False,
                             channel="twitter", post_id=root_id,
                             draft_kind="twitter_thread",
                             body_preview=tweets[0]))
        console.print(f"[green]✓ Recorded for autopilot tracking.[/] root_id={root_id}")


@publish_app.command("linkedin")
def publish_linkedin(
    file: str = typer.Argument(..., help="Path to a markdown file with the LinkedIn body."),
    use_mdp: bool = typer.Option(False, "--mdp", help="Try MDP API (requires LINKEDIN_ACCESS_TOKEN+URN)."),
    really_post: bool = typer.Option(False, "--really-post", help="Actually publish (only with --mdp)."),
):
    """Post to LinkedIn. By default opens a real browser tab with the draft pasted; you click Post."""
    from quorum.agents.tools import default_registry, load_builtins
    load_builtins()
    body = open(file, "r", encoding="utf-8").read()
    # Strip our metadata header
    if "---" in body:
        body = body.split("---", 1)[-1].strip()

    reg = default_registry()
    if use_mdp:
        result = asyncio.run(reg.call("linkedin.post_via_mdp", dry_run=not really_post, body=body))
    else:
        result = asyncio.run(reg.call("linkedin.open_composer", dry_run=False, body=body))
    console.print_json(json.dumps(result))


@app.command()
def metrics(
    channel: str = typer.Option("twitter", "--channel"),
    post_id: str = typer.Option(None, "--post-id"),
):
    """Show recorded metrics. With --post-id: latest snapshot for that post."""
    from quorum.agents.tools import default_registry, load_builtins
    load_builtins()
    reg = default_registry()
    if post_id:
        out = asyncio.run(reg.call("metrics.latest", dry_run=False, channel=channel, post_id=post_id))
    else:
        out = asyncio.run(reg.call("metrics.list_posts", dry_run=False, channel=channel, limit=20))
    console.print_json(json.dumps(out, default=str))


@app.command()
def autopilot(
    channel: str = typer.Option("twitter", "--channel"),
    threshold_likes: int = typer.Option(5, "--likes"),
    threshold_impressions: int = typer.Option(200, "--impr"),
    out_dir: str = typer.Option(None, "--out"),
):
    """Observe metrics for tracked posts. If a post is flopping after 24h, regenerate drafts. Never publishes."""
    from quorum.agents.autopilot import observe_and_decide
    result = asyncio.run(observe_and_decide(
        channel=channel,
        threshold_likes=threshold_likes,
        threshold_impressions=threshold_impressions,
        out_dir=out_dir,
    ))
    console.print(Panel(
        f"Observed: {result['observed']}  · Rewrites: {len(result.get('rewrites', []))}\nReport: {result.get('report', '?')}",
        title="[bold green]Autopilot[/]",
        border_style="green",
    ))


tools_app = typer.Typer(no_args_is_help=True, help="Tool registry — list / probe / load MCPs.")
app.add_typer(tools_app, name="tools")


# ---------- Context profiles (inject domain context before every `ask`) ----------

context_app = typer.Typer(
    no_args_is_help=True,
    help="Context profiles — inject domain/project context into every consensus query.",
)
app.add_typer(context_app, name="context")


@context_app.command("add")
def context_add(
    name: str = typer.Argument(..., help="Profile name (e.g. 'keratin-app', 'quorum-product')."),
    file: str = typer.Option(None, "--file", "-f", help="Read context body from a file instead of stdin."),
    text: str = typer.Option(None, "--text", "-t", help="Pass context body inline as a string."),
    use: bool = typer.Option(True, "--use/--no-use", help="Set this profile as active after creating."),
):
    """Create or overwrite a context profile.

    Examples:
      quorum context add keratin-app --file ./APP_STORE_METADATA.md
      quorum context add quorum-product --text 'Open-source multi-LLM consensus engine, B2B, Apache 2.0'
      cat README.md | quorum context add my-project   # stdin fallback
    """
    from quorum.context import save_context, set_active
    if file:
        body = open(file, "r", encoding="utf-8").read()
    elif text:
        body = text
    else:
        import sys as _sys
        if _sys.stdin.isatty():
            console.print("[red]Provide --file, --text, or pipe via stdin.[/]")
            raise typer.Exit(1)
        body = _sys.stdin.read()
    p = save_context(name, body)
    console.print(f"[green]✓[/] saved context [bold]{name}[/] ({len(body)} chars) → {p}")
    if use:
        set_active(name)
        console.print(f"[green]✓[/] active context now: [bold]{name}[/]")


@context_app.command("list")
def context_list():
    """List all context profiles."""
    from quorum.context import list_contexts, current_name, CONTEXT_DIR
    items = list_contexts()
    if not items:
        console.print(f"[dim]no context profiles yet. Create one: `quorum context add <name> --file <path>`[/]")
        return
    active = current_name()
    tbl = Table(title=f"Context profiles ({CONTEXT_DIR})", box=box.SIMPLE_HEAD)
    tbl.add_column("Name", style="cyan")
    tbl.add_column("Active", justify="center")
    tbl.add_column("Size")
    for name in items:
        from quorum.context import get_context
        body = get_context(name) or ""
        marker = "★" if name == active else ""
        tbl.add_row(name, marker, f"{len(body):,} chars")
    console.print(tbl)


@context_app.command("use")
def context_use(name: str = typer.Argument(..., help="Profile name to make active.")):
    """Make a context profile active — it will be auto-injected into every `quorum ask`."""
    from quorum.context import set_active
    try:
        set_active(name)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/] active context now: [bold]{name}[/]")


@context_app.command("show")
def context_show(name: str = typer.Argument(None, help="Profile name (defaults to active).")):
    """Print the body of a context profile."""
    from quorum.context import get_context, current_name
    target = name or current_name()
    if not target:
        console.print("[red]no profile name given and no active profile set.[/]")
        raise typer.Exit(1)
    body = get_context(target)
    if body is None:
        console.print(f"[red]context '{target}' not found.[/]")
        raise typer.Exit(1)
    console.print(Panel(body, title=f"context: {target}", border_style="cyan"))


@context_app.command("current")
def context_current():
    """Print the name of the active context (or 'none')."""
    from quorum.context import current_name
    name = current_name()
    console.print(name or "[dim]none[/]")


@context_app.command("clear")
def context_clear():
    """Unset the active context (does NOT delete the profile)."""
    from quorum.context import clear_active
    clear_active()
    console.print("[green]✓[/] active context cleared (profiles preserved).")


@context_app.command("rm")
def context_rm(name: str = typer.Argument(..., help="Profile name to delete.")):
    """Delete a context profile permanently."""
    from quorum.context import remove_context
    ok = remove_context(name)
    if not ok:
        console.print(f"[red]context '{name}' not found.[/]")
        raise typer.Exit(1)
    console.print(f"[green]✓[/] removed context [bold]{name}[/].")


@tools_app.command("list")
def tools_list(load_mcps: bool = typer.Option(False, "--mcps", help="Probe and register QUORUM_MCPS servers first.")):
    """List all registered tools (built-in + plugins + MCP servers)."""
    from quorum.agents.tools import default_registry, load_builtins, load_plugins
    load_builtins()
    load_plugins()
    if load_mcps:
        from quorum.agents.tools.mcp import register_mcp_servers
        try:
            registered = asyncio.run(register_mcp_servers())
            console.print(f"[dim]Registered {len(registered)} MCP tool(s).[/]")
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]MCP load failed: {e}[/]")
    reg = default_registry()
    tbl = Table(title="Quorum tool registry", box=box.SIMPLE_HEAD)
    tbl.add_column("Name", style="cyan")
    tbl.add_column("Destructive")
    tbl.add_column("Env ready")
    tbl.add_column("Description")
    for t in sorted(reg.list(), key=lambda x: x.name):
        tbl.add_row(
            t.name,
            "💥" if t.is_destructive else "",
            "✅" if t.env_ready() else f"❌ ({', '.join(t.missing_env())})",
            t.description[:60],
        )
    console.print(tbl)


@tools_app.command("call")
def tools_call(
    name: str = typer.Argument(..., help="Tool name (e.g. metrics.list_posts)."),
    args_json: str = typer.Option("{}", "--args", help="JSON dict of args."),
    really_do: bool = typer.Option(False, "--really-do", help="Bypass dry-run for destructive tools."),
):
    """Invoke a tool directly. Use --args to pass JSON kwargs."""
    from quorum.agents.tools import default_registry, load_builtins, load_plugins
    load_builtins()
    load_plugins()
    kwargs = json.loads(args_json)
    reg = default_registry()
    out = asyncio.run(reg.call(name, dry_run=not really_do, **kwargs))
    console.print_json(json.dumps(out, default=str))


# --------- Loop 14: Web Knowledge Harvester ---------

@app.command()
def learn(
    topic: list[str] = typer.Option(
        ..., "--topic", help="One or more topics to harvest (repeat the flag).",
    ),
    n_search: int = typer.Option(5, "--n-search", help="Web results per topic."),
    max_chunks: int = typer.Option(6, "--max-chunks", help="Cap chunks per source page."),
):
    """Harvest one or more topics from the web into Quorum's local knowledge base.

    Each topic is searched on DuckDuckGo, top pages are fetched, text chunked
    and embedded, then stored in $QUORUM_DATA_DIR/web_kb.db. Run again later to
    refresh. Future `quorum ask` calls can use these facts via --recall.
    """
    from quorum.evolution.web_learner import harvest, stats
    tbl = Table(title="Quorum learn — harvest report", box=box.SIMPLE_HEAD)
    tbl.add_column("Topic", style="cyan")
    tbl.add_column("Sources", justify="right")
    tbl.add_column("Chunks", justify="right")
    tbl.add_column("Stored (new)", justify="right")
    tbl.add_column("Dup skipped", justify="right")
    for t in topic:
        r = asyncio.run(harvest(t, n_search=n_search, max_chunks_per_source=max_chunks))
        tbl.add_row(
            t,
            str(r.get("fetched_sources", 0)),
            str(r.get("candidate_chunks", 0)),
            str(r.get("stored", 0)),
            str(r.get("duplicates_skipped", 0)),
        )
    console.print(tbl)
    s = stats()
    console.print(
        f"[dim]KB now: {s['topics']} topics, {s['facts']} facts, {s['db_size_kb']} KB at {s['db_path']}[/]"
    )


@app.command()
def recall(
    query: str = typer.Argument(..., help="What you want Quorum to recall."),
    top_k: int = typer.Option(5, "--top-k"),
):
    """Show the top-k learned facts most semantically relevant to your query."""
    from quorum.evolution.web_learner import recall as do_recall
    hits = asyncio.run(do_recall(query, top_k=top_k))
    if not hits:
        console.print(
            "[yellow]No facts in the KB yet. Run `quorum learn --topic '...'` first.[/]"
        )
        return
    for i, h in enumerate(hits, 1):
        console.print(Panel(
            f"[bold]{h.source_title or h.source_url}[/]\n"
            f"[dim]{h.source_url}[/]\n"
            f"[green]score={h.score:.3f}[/] · topic=[cyan]{h.topic}[/]\n\n"
            f"{h.text[:600]}{'…' if len(h.text) > 600 else ''}",
            title=f"[{i}] recall",
            border_style="green" if h.score > 0.5 else "yellow" if h.score > 0.3 else "red",
            box=box.ROUNDED,
        ))


@app.command(name="framings")
def framings_cmd():
    """List the available --reframe contexts and what each one declares to the models."""
    from quorum.agents.reframe import FRAMINGS
    for key, text in FRAMINGS.items():
        console.print(Panel(
            text,
            title=f"[bold green]--reframe {key}[/]",
            border_style="green",
            box=box.ROUNDED,
        ))
    console.print(
        "[dim]All wraps are logged to $QUORUM_DATA_DIR/reframe_audit/. "
        "Use these only when the framing is TRUE for your situation.[/]"
    )


@app.command()
def overnight(
    topics_file: str = typer.Option(
        "quorum_overnight_topics.txt", "--topics-file",
        help="Path to newline-delimited topic list (# = comment).",
    ),
    max_hours: float = typer.Option(8.0, "--max-hours"),
    base_delay: float = typer.Option(120.0, "--delay"),
    jitter: float = typer.Option(60.0, "--jitter"),
    log_path: str = typer.Option(None, "--log"),
    stop_file: str = typer.Option(None, "--stop-file"),
    n_search: int = typer.Option(4, "--n-search"),
    max_chunks: int = typer.Option(4, "--max-chunks"),
):
    """Run the autonomous overnight learning loop. Designed to run in the background for hours."""
    from quorum.evolution.overnight import run_overnight
    summary = asyncio.run(run_overnight(
        topics_file=topics_file,
        max_hours=max_hours,
        base_delay_s=base_delay,
        jitter_s=jitter,
        n_search=n_search,
        max_chunks=max_chunks,
        log_path=log_path,
        stop_file=stop_file,
    ))
    console.print_json(json.dumps(summary, default=str))


@app.command()
def facts(
    list_topics_flag: bool = typer.Option(False, "--topics", help="List all harvested topics."),
):
    """Knowledge-base summary or topic list."""
    from quorum.evolution.web_learner import list_topics, stats
    s = stats()
    console.print(Panel(
        f"DB path: {s['db_path']}\n"
        f"Topics: {s['topics']}\n"
        f"Facts: {s['facts']}\n"
        f"Size: {s['db_size_kb']} KB",
        title="[bold green]Quorum KB[/]",
        border_style="green",
    ))
    if list_topics_flag:
        topics = list_topics()
        if not topics:
            console.print("[yellow]No topics yet.[/]")
            return
        import datetime
        tbl = Table(title="Topics", box=box.SIMPLE_HEAD)
        tbl.add_column("Topic", style="cyan")
        tbl.add_column("Facts", justify="right")
        tbl.add_column("Last harvested")
        for t in topics:
            ts = datetime.datetime.fromtimestamp(t["last_harvested"]).strftime("%Y-%m-%d %H:%M") if t["last_harvested"] else "—"
            tbl.add_row(t["name"], str(t["fact_count"]), ts)
        console.print(tbl)


if __name__ == "__main__":
    app()
