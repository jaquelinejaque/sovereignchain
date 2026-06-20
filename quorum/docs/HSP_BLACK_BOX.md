# HSP Black Box — tamper-evident audit log

> Every `consensus()` call appends a row to a SHA-256 hash chain stored at
> `~/.quorum/audit_chain.db` (override with `QUORUM_AUDIT_DB`). Auditors can
> walk the chain offline and prove no row was deleted, reordered, or modified.
>
> Compliance primitive for **EU AI Act Article 14** ("Logs traceable in
> chronological order") and **SOC2 CC7.2** (system monitoring / integrity).
> Patent context: HSP gate (PCT/US26/11908).

---

## TL;DR

```bash
# 1. Use Quorum normally — every consensus() call auto-appends to the chain.
quorum ask --all "What is the chemical structure of L-Cysteine?"

# 2. Prove the chain is intact (exit 0 = OK, exit 2 = tampering detected).
quorum-audit verify-chain

# 3. Inspect.
quorum-audit status

# 4. Hand a slice to an external auditor.
quorum-audit export --since 2026-01-01T00:00:00Z --out /tmp/audit.jsonl
```

---

## What gets logged

Each `consensus()` invocation writes one row containing the canonical-JSON
payload:

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | autoincrement, gap-free walk |
| `prev_hash` | TEXT(64) | `this_hash` of the previous row, or 64 zeros for genesis |
| `payload_json` | TEXT | `json.dumps(payload, sort_keys=True, separators=(",",":"))` |
| `this_hash` | TEXT(64) | `SHA256(prev_hash + "|" + payload_json + "|" + created_at)` |
| `created_at` | TEXT | UTC ISO-8601 with microseconds |

Typical payload fields (the engine writes these; you can add your own via
`black_box.append(...)`):

```json
{
  "event": "consensus",
  "query_hash": "9f4c…",
  "models": ["claude", "gpt", "gemini", "grok", "llama-local"],
  "confidence": 0.86,
  "method": "semantic",
  "hsp_decision": "binding",
  "tokens_in": 412,
  "tokens_out": 1180,
  "cost_usd": 0.0143,
  "latency_ms": 2710
}
```

Storage is **best-effort and never raises** — an audit-DB failure must not
poison the consensus response path. If a write fails, the chain simply does
not extend on that call; the next successful call links to the most recent
intact `this_hash`.

The DB file is created with mode `0o600`. Exports are written `0o444`
(primitive WORM — file becomes read-only at the OS layer once exported).

---

## Verification — how an auditor proves integrity offline

```bash
quorum-audit verify-chain
# OK — 1284 records, chain intact          → exit 0
# BROKEN at id=842 — possible tampering    → exit 2
```

Under the hood (`black_box.verify_chain`):

1. Start with `prev_hash = "0"*64` (genesis).
2. Walk rows in `id ASC` order.
3. For each row, check `row.prev_hash == prev_hash` (ordering).
4. Recompute `SHA256(prev_hash + "|" + payload_json + "|" + created_at)` and
   compare against the stored `this_hash` (integrity).
5. Set `prev_hash = row.this_hash` and continue.

Any deleted, reordered, or modified row breaks the chain at that point and
the function returns `(False, broken_at_id)`. The CLI exits with code `2`
so CI / cron jobs can alert.

No secret key is required — the auditor reproduces the hash from public
inputs, so verification can happen on a sealed forensic copy of the DB
without trusting the operator.

---

## Export for an external auditor

```bash
quorum-audit export --since 2026-01-01T00:00:00Z --out /tmp/audit.jsonl
# Exported 1284 rows
```

Each line of the JSONL is one row, with `payload_json` already parsed:

```json
{"id":1,"prev_hash":"0000…","payload":{"event":"consensus",…},"this_hash":"9f4c…","created_at":"2026-06-19T10:14:22.481731+00:00"}
```

The auditor can re-run the same verification logic against the JSONL alone
(no DB software required) — `prev_hash` of row `N` must equal `this_hash` of
row `N-1`, and each `this_hash` must equal the recomputed SHA-256.

---

## Inspecting current state

```bash
quorum-audit status
# {
#   "count": 1284,
#   "first_at": "2026-06-15T08:02:11.118044+00:00",
#   "last_at":  "2026-06-19T14:51:03.992017+00:00",
#   "db_path":  "/Users/you/.quorum/audit_chain.db",
#   "db_size_bytes": 412160
# }
```

---

## Programmatic use

```python
from quorum.hsp import black_box

# Append a custom event (e.g. a manual override decision)
h = black_box.append({
    "event": "manual_override",
    "operator": "jaqueline@sovereign-chain.com",
    "reason": "regulatory request 2026-06-19",
    "original_query_hash": "9f4c…",
})
print(h)  # this_hash, or None if write failed

# Verify
ok, broken_at = black_box.verify_chain()
assert ok, f"chain broken at row {broken_at}"

# Export
n = black_box.export_jsonl(since_iso="2026-01-01T00:00:00Z",
                           out_path="/tmp/audit.jsonl")
print(f"exported {n} rows")

# Stats
print(black_box.stats())
```

API surface lives in `src/quorum/hsp/black_box.py` and is intentionally
small: `append`, `verify_chain`, `export_jsonl`, `stats`.

---

## Design notes & honest limitations

- **Hash chain, not HMAC.** No secret key to manage. Integrity + ordering are
  provable from public material. Trade-off: an attacker with full write
  access to the DB *and* enough time can rewrite the entire chain forward
  from a tampered row. The mitigation is to periodically archive
  `audit_chain.db` to append-only storage (S3 Object Lock, immutable
  Cloud Storage bucket, on-prem WORM filer) and pin a checksum of the
  archive — that pin is what the auditor trusts.
- **Single-machine threat model.** `0o600` perms on the DB and `0o444` on
  exports stop casual tampering by other local users; they don't defend
  against a root-level attacker. For Forensic+ guarantees you need
  filesystem-level WORM.
- **Never raises.** A broken audit chain must not break consensus. Writes
  swallow exceptions and log at `DEBUG`. Operators who care should run
  `quorum-audit verify-chain` from cron and alert on non-zero exit.
- **Canonical JSON.** Payloads are serialized with `sort_keys=True,
  separators=(",",":")` so two clients hashing the same logical payload
  produce identical bytes.

---

## Where this fits

| Compliance | What this gives you | What it does NOT give you |
|---|---|---|
| EU AI Act Art. 14 | Chronological, tamper-evident log per consensus call | Article 13 transparency PDF (separate — `hsp/ai_act_cert.py`) |
| SOC2 CC7.2 | System-monitoring integrity primitive | Continuous-monitoring SIEM, alerting, retention policy |
| ISO 27001 A.12.4 | Event log + protection of log info | Centralized log management |

For per-query signed PDF certificates (Article 13), see
`hsp/ai_act_cert.py` and the `/v1/cert/{query_id}` endpoint.
