# Postmortem: Quorum v0.1.0 Deploy Session

**Status:** Resolved
**Severity:** High (4 failed deploys, ~3h engineer time)
**Date:** 2026-06-16

---

## 1. Timeline — Four Deploy Attempts

| # | Failure point | Time lost | Symptom |
|---|---|---|---|
| 1 | `docker build` exit 1 at `COPY uv.loc[k]` | ~25 min | BuildKit read brackets as literal characters |
| 2 | `docker build` exit 1 inside hatchling metadata | ~40 min | `Readme file does not exist: README.md` |
| 3 | `docker build` exit 1, same line, different reason | ~35 min | README.md silently excluded by `.dockerignore` |
| 4 | Container booted on Cloud Run, crashed on first request | ~55 min | `Read-only file system: '/app/.quorum'` |

Total wall clock: ~2h 55m. Most of it was lost to slow `gcloud builds submit` round-trips (~6 min each).

---

## 2. The Four Bugs

### Bug 1 — `COPY uv.loc[k]` glob, BuildKit literal interpretation

Original line (carried from a pip-based template):

```dockerfile
COPY pyproject.toml uv.loc[k] ./
```

Intent: optionally copy `uv.lock` if present. Classic builder treated `uv.loc[k]` as a glob; BuildKit (Cloud Build default since 2024) treats it as a literal filename:

```
ERROR: failed to calculate checksum of ref ... "/uv.loc[k]": not found
```

Fix: copy `uv.lock` unconditionally; generate and commit it.

### Bug 2 — hatchling needs README.md but Dockerfile didn't COPY it

`pyproject.toml` declared `readme = "README.md"`. The Dockerfile copied source but not the README. During the metadata step, hatchling exploded:

```
ValueError: Readme file does not exist: README.md
```

Fix: explicit `COPY README.md ./` before `pip install`.

### Bug 3 — `.dockerignore` excluded README.md while pyproject required it

After Bug 2 was "fixed," the build still failed at the same step with the same error. The `.dockerignore` from our scaffold listed:

```
*.md
!CHANGELOG.md
```

`COPY README.md` silently produced no file because the build context filtered it out before COPY could see it. There is no warning — BuildKit treats a `COPY` of a `.dockerignore`d path as a no-op. Three files (`pyproject.toml`, `Dockerfile`, `.dockerignore`) each individually valid, jointly broken.

Fix: add `!README.md` to `.dockerignore`.

### Bug 4 — APIKeyStore writes to `/app/.quorum` on a read-only filesystem

Container built and pushed cleanly. First request to `/v1/consensus` crashed:

```
File "quorum/storage/api_key_store.py", line 47, in _ensure_dir
    os.makedirs(self.path, exist_ok=True)
PermissionError: [Errno 30] Read-only file system: '/app/.quorum'
```

Cloud Run runs containers as non-root with the image filesystem read-only — only `/tmp` is writable. `APIKeyStore` hardcoded `Path.home() / ".quorum"`, resolving to `/app/.quorum` because `HOME=/app` for our user. The code worked perfectly on a dev laptop and inside the GH Actions runner.

Fix: `Path(os.getenv("QUORUM_DATA_DIR", "/tmp/quorum"))`.

---

## 3. Root Cause Analysis

**Why didn't Workflow #1 (the 20-agent build) catch these?** The fan-out produced four artifacts in parallel — Dockerfile, pyproject, `.dockerignore`, storage layer — each by a different agent with its own context. No agent ever saw the full graph. There was no integration step where someone ran `docker build .` against the merged tree before we shipped to Cloud Build. Agents wrote code in silos; we trusted the silos.

**Cross-config conflicts are a class, not an incident.** Bugs 2 and 3 share a shape: file A declares a requirement, file B is supposed to satisfy it, file C silently denies it. Static review of any single file finds nothing. We need a validator that asks: "given pyproject's declared inputs, will the docker build context actually contain them?"

**Cloud Run filesystem quirks are missing from most Python templates.** Every `cookiecutter-fastapi` we checked still writes to `~/.appname`. The "read-only except `/tmp`" rule is easy to forget when the same image runs fine under `docker run` locally.

---

## 4. Lessons & Prevention for v0.2.0

1. **`integration-sanity` agent on every shipping Workflow.** Its sole job: take the merged tree and run the build/test/smoke pipeline end-to-end before declaring done. Owns no source files; owns the verdict.
2. **Mandatory `docker build .` smoke test before any `gcloud run deploy`.** Pre-push hook and required CI check.
3. **Storage paths env-configurable, default `/tmp`.** Any write path goes through `get_data_dir()` reading `QUORUM_DATA_DIR`, defaulting to `/tmp/quorum`. Lint rule to enforce.
4. **Pre-deploy checklist as a CLI: `quorum doctor`.** Checks `uv.lock` present, README.md not ignored, no hardcoded `Path.home()`, all `pyproject` readme/license files resolvable in build context, Dockerfile `USER` matches HOME assumptions. Non-zero blocks deploy.

---

## 5. Meta-Irony

We are shipping a multi-LLM consensus product. The pitch is "ask three models, surface disagreement, catch what one model missed." We had a Dockerfile, a pyproject, and a `.dockerignore` that pairwise contradicted each other — exactly the kind of low-context disagreement our product is built to surface. We never ran the configs through Quorum. For v0.2.0, `quorum doctor` will call the consensus API on the deploy configs as one of its checks. If we won't point our product at our own repo, why would anyone else?
