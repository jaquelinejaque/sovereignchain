# Quorum MCP Server

Stdio MCP (Model Context Protocol) server exposing the Quorum consensus engine to any MCP-aware AI client (Claude Desktop, Cursor, Gemini Antigravity, Cline, etc.).

## Status

- **Transport**: stdio only (no auth, no rate limiting in this first cut)
- **Tools**: 6 (consensus, multi, verdict, disagreement matrix, readiness check, evidence record)
- **Pricing**: free locally; remote/paid tier in roadmap
- **Disclaimer**: every response payload includes a mandatory legal disclaimer that Sovereign Chain Ltd is NOT a Notified Body under Article 31 of Regulation (EU) 2024/1689 and outputs are advisory only

## Install

```bash
cd ~/sovereignchain/quorum
.venv/bin/python -m pip install -e ".[mcp]"
```

## Run standalone

```bash
QUORUM_DEV_MODE=1 .venv/bin/python -m quorum.mcp.server
# Server now reads JSON-RPC on stdin, writes on stdout. Press Ctrl-D to stop.
```

Or via the console script (after `pip install -e .`):

```bash
QUORUM_DEV_MODE=1 quorum-mcp
```

## Wire to Gemini Antigravity

Already wired — entry added to `~/.gemini/config/mcp_config.json` under `mcpServers.quorum`:

```json
{
  "mcpServers": {
    "quorum": {
      "command": "/Users/facec/sovereignchain/quorum/.venv/bin/python",
      "args": ["-m", "quorum.mcp.server"],
      "env": {
        "QUORUM_DEV_MODE": "1",
        "QUORUM_MCP_LOG": "WARNING"
      }
    }
  }
}
```

After Gemini restarts, the 6 tools become callable as `mcp(quorum/quorum_consensus)`, `mcp(quorum/quorum_verdict)`, etc.

## Wire to Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "quorum": {
      "command": "/Users/facec/sovereignchain/quorum/.venv/bin/python",
      "args": ["-m", "quorum.mcp.server"],
      "env": {
        "QUORUM_DEV_MODE": "1"
      }
    }
  }
}
```

Restart Claude Desktop. The Quorum tools appear in the tool picker.

## Wire to Cursor

In Cursor settings, add to `mcpServers`:

```json
{
  "quorum": {
    "command": "/Users/facec/sovereignchain/quorum/.venv/bin/python",
    "args": ["-m", "quorum.mcp.server"]
  }
}
```

## Tools exposed

### `quorum_consensus(prompt, budget_usd?, timeout_s?, user_id?)`

Multi-LLM consensus query. Fans the prompt across every configured provider, scores semantic agreement, returns the synthesized winner with per-model breakdown.

Use for: high-stakes decisions where single-model bias is unacceptable.

### `quorum_multi(prompt)`

Raw per-model answers without picking a winner. Use when you want to see every model's full response side-by-side (content authoring, compliance review, manual decision).

### `quorum_verdict(claim)`

Yes/no verdict on a factual or evaluative claim. Each model votes TRUE/FALSE/UNCLEAR; consensus tally returned with per-model breakdown. Use for fact-checking and binary decisions.

### `quorum_disagreement_matrix(prompt)`

Pairwise lexical agreement matrix across every model. Useful as EU AI Act Article 14 automation-bias evidence material — shows where the model panel converged vs diverged.

### `quorum_readiness_check(system_description)`

Annex VI readiness gap-analysis on a free-text description of an AI system. Returns advisory report: likely Annex III risk class, relevant Articles 9/12/13/14/16 obligations, internal vs external assessment route, top documentation artefacts to prepare. **Advisory — not a conformity assessment.**

### `quorum_evidence_record(prompt)`

Generates a per-query tamper-evident PDF (or Markdown fallback) evidence record written to `~/.quorum/evidence_records/`. Designed as internal material for EU AI Act Articles 12 + 13. **Advisory — not a conformity assessment.**

## Environment variables

| Var | Purpose | Default |
|---|---|---|
| `QUORUM_DEV_MODE` | Bypass paid-license check (set to `1` on dev machines) | unset |
| `QUORUM_MCP_LOG` | Log level (WARNING, INFO, DEBUG) | `WARNING` |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `XAI_API_KEY`, `DASHSCOPE_API_KEY` | BYOK provider keys consumed by `consensus()` | autodiscover from env |

The server does **not** proxy API keys — every provider call is made with the client's own key.

## Smoke test

```bash
QUORUM_DEV_MODE=1 .venv/bin/python /tmp/mcp_smoke.py
```

Or manually drive the JSON-RPC handshake:

```bash
QUORUM_DEV_MODE=1 .venv/bin/python -m quorum.mcp.server <<'EOF'
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"manual","version":"0.1"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
EOF
```

You should see the initialize response, followed by 6 tools.

## Legal disclaimer (mandatory)

Every tool response includes a `_disclaimer` field. The full text:

> "Advisory only — not a conformity assessment under Regulation (EU) 2024/1689. Sovereign Chain Ltd is not a Notified Body under Article 31. Outputs are technical evidence material to support, not replace, the documentation and risk-management obligations of the AI system provider (internal assessment under Annex VI) or a designated Notified Body (external assessment under Annex VII)."

Removing this disclaimer or shipping a fork that hides it violates the FSL-1.1 commercial restrictions and may expose the operator to Fraud Act 2006 s.2 (false representation) and CPUTR 2008 Reg 6 (misleading commercial practice) under UK law.

## Roadmap

- HTTP/SSE transport for remote multi-tenant access (paid)
- Stripe metering for per-call billing
- Authentication via Bearer token (BYO key on top of `customer_keys.py`)
- Rate limiting (60 calls/minute default)
- Optional `_disclaimer_compact` mode for low-bandwidth callers
- Tool: `quorum_compare_models` (head-to-head with ELO update)
- Tool: `quorum_explain_disagreement` (chain-of-thought on why models diverged)
