# quorum/scripts

Local automation for the quorum-api service. Run BEFORE any cloud deploy.

## smoke_test_docker.sh

End-to-end Docker smoke test. Builds the image, boots the container, hits
`/healthz`, `/v1/models`, and `/v1/consensus`, then tears the container down.

### Run it

```bash
export GEMINI_API_KEY=...        # required
export REPLICATE_API_TOKEN=...   # required
./scripts/smoke_test_docker.sh
```

That's it. Exit code `0` = safe to deploy. Exit code `1` = do NOT deploy; the
failure reason is printed to stderr and the last 50 container log lines are
shown for post-mortem.

### What it verifies

1. `docker build` succeeds against `/tmp/sovereignchain/quorum`.
2. Container starts and binds to host port `8081`.
3. `GET /healthz` returns `200` within 30s (polled every 1s).
4. `GET /v1/models` returns JSON with at least one provider.
5. `POST /v1/consensus` returns JSON with `answer` + non-empty `models` array.

### Cleanup

A `trap EXIT` removes the container even on failure or Ctrl-C. No manual
`docker rm` needed.

### Requirements

`docker`, `curl`, `python3` on the host. No `jq` required.
