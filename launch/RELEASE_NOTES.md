# Quorum v0.1.0

First tagged release. Sets the public API surface and billing path. v0.0.1 was an internal preview and is not upgrade-safe.

## Highlights
- 13 evolution loops as composable primitives — each runs standalone, returns a typed result, and chains without glue code.
- Single-process server: HTTP + SSE streaming, no broker, no queue required.
- Stripe-metered billing with idempotent webhooks and per-key usage caps.
- HSP reference implementation (patent-pending) backing the synthesis and consensus loops.
- `pip install quorum` ships the SDK and a local-mode runner. First loop call in under a minute.

## Breaking changes from v0.0.1
- `quorum.run()` now returns `LoopResult`, not a bare dict. Use `.value` for the old payload.
- Config moved to `~/.config/quorum/config.toml`. The YAML loader is removed.
- `/v0/invoke` is gone. Use `/v1/loops/{name}:invoke`. v0 routes return 410.
- Loop names are lowercase-with-hyphens. Camel-case names from v0.0.1 will not resolve.

## What's in this release

**Foundation**
Typed loop registry, deterministic run IDs, structured error envelopes, OpenTelemetry traces on every invocation, deterministic seeds for replay.

**Evolution Loops (13)**
Thirteen independently versioned loops. Run solo or chain via `quorum.chain([...])`. Each loop has a typed input/output schema and is registered by lowercase-hyphenated name.

**Server**
FastAPI app, SSE for streaming results, graceful shutdown, request-scoped tracing. Endpoints: `/v1/loops`, `/v1/loops/{name}:invoke`, `/v1/runs/{id}`, `/healthz`.

**Billing**
Stripe metered billing per loop invocation. Idempotent webhook ingestion (replay-safe), per-key monthly caps, rolling usage export. Local dev bypasses Stripe; set `QUORUM_BILLING=stripe` to enable.

**HSP**
Reference implementation of the Hyper-Synthesis Protocol used by the synthesis and consensus loops. Wire format frozen for v0.1; semantics may shift before v0.2 — see roadmap.

## How to try

```bash
pip install quorum
```

```bash
curl -X POST https://api.quorum.dev/v1/loops/consensus:invoke \
  -H "Authorization: Bearer $QUORUM_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":"your prompt here"}'
```

## Roadmap to v0.2
- Persistent run history with deterministic replay (`quorum runs replay <id>`). Today: in-memory only.
- Multi-tenant key scoping with org-level rate limits. Today: single-tenant.
- HSP wire format v1.0 with a published conformance suite — stable across implementations, not just this one.

## Patent
HSP is covered by pending application **PCT/US26/11908**. Filing: <https://patentscope.wipo.int/search/en/detail.jsf?docId=PCT/US26/11908>
