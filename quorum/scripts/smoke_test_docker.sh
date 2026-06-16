#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# smoke_test_docker.sh
#
# Local Docker smoke test for the quorum-api service.
# Runs end-to-end BEFORE any cloud deploy to catch broken images, broken
# endpoints, or missing env wiring without burning Fly/Render/Railway minutes.
#
# Exit codes:
#   0  - all checks passed
#   1  - any failure (build, healthz, models, consensus, JSON parse, etc.)
#
# Usage:
#   export GEMINI_API_KEY=...
#   export REPLICATE_API_TOKEN=...
#   ./scripts/smoke_test_docker.sh
# ------------------------------------------------------------------------------

set -euo pipefail

# ---- Configuration -----------------------------------------------------------
IMAGE_TAG="quorum-api:smoke"
CONTAINER_NAME="quorum-smoke-$$"        # PID-suffixed to avoid collisions
HOST_PORT=8081
CONTAINER_PORT=8080                      # internal port the API listens on
HEALTH_TIMEOUT=30                        # seconds to wait for /healthz
BUILD_CONTEXT="/tmp/sovereignchain/quorum"
BASE_URL="http://127.0.0.1:${HOST_PORT}"

# ---- Pretty printers ---------------------------------------------------------
ok()   { printf "\033[0;32m[OK]\033[0m    %s\n" "$*"; }
info() { printf "\033[0;36m[INFO]\033[0m  %s\n" "$*"; }
fail() { printf "\033[0;31m[FAIL]\033[0m  %s\n" "$*" >&2; }

# ---- Teardown trap -----------------------------------------------------------
# Ensure the container is always removed, even on failure / Ctrl-C.
cleanup() {
  local exit_code=$?
  info "Tearing down container ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  if [[ ${exit_code} -eq 0 ]]; then
    ok "Smoke test PASSED"
  else
    fail "Smoke test FAILED (exit ${exit_code})"
  fi
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

# ---- Pre-flight: required tools ---------------------------------------------
command -v docker >/dev/null || { fail "docker not installed"; exit 1; }
command -v curl   >/dev/null || { fail "curl not installed";   exit 1; }
command -v python3 >/dev/null || { fail "python3 not installed (needed for JSON parsing)"; exit 1; }

# ---- Pre-flight: required env vars ------------------------------------------
# These must exist in the host shell. We do NOT echo their values.
: "${GEMINI_API_KEY:?GEMINI_API_KEY must be exported before running this script}"
: "${REPLICATE_API_TOKEN:?REPLICATE_API_TOKEN must be exported before running this script}"

# ---- Step 1: Build the Docker image -----------------------------------------
info "Building image ${IMAGE_TAG} from ${BUILD_CONTEXT}..."
if ! docker build -t "${IMAGE_TAG}" "${BUILD_CONTEXT}"; then
  fail "docker build failed — fix Dockerfile or build context before retrying"
  exit 1
fi
ok "Image built: ${IMAGE_TAG}"

# ---- Step 2: Run the container (detached) -----------------------------------
# Map host:container port. Inject API keys WITHOUT logging them.
# --rm is intentionally NOT used because we want the trap to control teardown
# and preserve logs for post-mortem if needed.
info "Starting container ${CONTAINER_NAME} on host port ${HOST_PORT}..."
docker run -d \
  --name "${CONTAINER_NAME}" \
  -p "${HOST_PORT}:${CONTAINER_PORT}" \
  -e GEMINI_API_KEY \
  -e REPLICATE_API_TOKEN \
  "${IMAGE_TAG}" >/dev/null

ok "Container started"

# ---- Step 3: Wait for /healthz (max 30s, poll every 1s) ---------------------
info "Waiting up to ${HEALTH_TIMEOUT}s for ${BASE_URL}/healthz to return 200..."
healthy=0
for ((i=1; i<=HEALTH_TIMEOUT; i++)); do
  # -f returns non-zero on HTTP >= 400; -s silent; -o discard body
  if curl -fsS -o /dev/null "${BASE_URL}/healthz" 2>/dev/null; then
    healthy=1
    ok "/healthz returned 200 after ${i}s"
    break
  fi
  sleep 1
done

if [[ ${healthy} -ne 1 ]]; then
  fail "/healthz did not respond 200 within ${HEALTH_TIMEOUT}s"
  info "Last container logs:"
  docker logs --tail 50 "${CONTAINER_NAME}" >&2 || true
  exit 1
fi

# ---- Step 4: GET /v1/models — at least 1 provider ---------------------------
info "Checking ${BASE_URL}/v1/models..."
models_body="$(curl -fsS "${BASE_URL}/v1/models")" || {
  fail "/v1/models request failed"
  exit 1
}

# Parse JSON in python3 (more reliable than jq, which may not be installed)
provider_count="$(printf '%s' "${models_body}" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception as e:
    sys.stderr.write(f"invalid JSON: {e}\n")
    sys.exit(2)
# Accept either {"providers":[...]} or {"models":[...]} or a bare list
candidates = payload.get("providers") if isinstance(payload, dict) else None
if candidates is None and isinstance(payload, dict):
    candidates = payload.get("models")
if candidates is None and isinstance(payload, list):
    candidates = payload
if not isinstance(candidates, list):
    sys.stderr.write("response did not contain a providers/models array\n")
    sys.exit(3)
print(len(candidates))
')" || { fail "/v1/models JSON shape invalid"; exit 1; }

if [[ "${provider_count}" -lt 1 ]]; then
  fail "/v1/models returned 0 providers — at least 1 required"
  exit 1
fi
ok "/v1/models returned ${provider_count} provider(s)"

# ---- Step 5: POST /v1/consensus with a 1-sentence test prompt --------------
info "Hitting ${BASE_URL}/v1/consensus with a 1-sentence test prompt..."
consensus_body="$(curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Reply with the single word: pong."}' \
  "${BASE_URL}/v1/consensus")" || {
  fail "/v1/consensus request failed"
  exit 1
}

# Verify response shape: must have `answer` AND `models` array.
printf '%s' "${consensus_body}" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
except Exception as e:
    sys.stderr.write(f"invalid JSON: {e}\n")
    sys.exit(2)
if not isinstance(payload, dict):
    sys.stderr.write("top-level response must be an object\n")
    sys.exit(3)
if "answer" not in payload:
    sys.stderr.write("response missing required key: answer\n")
    sys.exit(4)
models = payload.get("models")
if not isinstance(models, list):
    sys.stderr.write("response key `models` must be an array\n")
    sys.exit(5)
if len(models) < 1:
    sys.stderr.write("response key `models` is empty\n")
    sys.exit(6)
print("consensus_ok")
' >/dev/null || { fail "/v1/consensus response shape invalid"; exit 1; }

ok "/v1/consensus returned valid answer + models array"

# ---- Step 6: Success summary ------------------------------------------------
echo
echo "============================================================"
echo " QUORUM-API SMOKE TEST — SUMMARY"
echo "============================================================"
echo " Image:           ${IMAGE_TAG}"
echo " Container:       ${CONTAINER_NAME}"
echo " Base URL:        ${BASE_URL}"
echo " /healthz:        OK"
echo " /v1/models:      OK (${provider_count} provider(s))"
echo " /v1/consensus:   OK (answer + models)"
echo "============================================================"
echo " Safe to deploy to cloud."
echo "============================================================"

# trap will run cleanup() and exit 0
exit 0
