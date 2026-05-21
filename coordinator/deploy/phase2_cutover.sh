#!/usr/bin/env bash
# Phase-2 claim cutover: drain fleet, flip hub+workers, verify health.
# Run from Mac with coordinator.env configured (see OPERATOR_GUIDE.md).
set -euo pipefail

BASE="${REE_WORKING:-$HOME/REE_Working}"
ENV_FILE="$BASE/REE_assembly/coordinator.env"
SERVE="${SERVE_URL:-http://127.0.0.1:8000}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: missing $ENV_FILE" >&2
  exit 1
fi
env_val() {
  grep -E "^${1}=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2-
}
COORDINATOR_URL="$(env_val COORDINATOR_URL)"
COORDINATOR_LOCAL_TOKEN="$(env_val COORDINATOR_LOCAL_TOKEN)"
COORDINATOR_SSH_USER="$(env_val COORDINATOR_SSH_USER)"
SSH_USER="${COORDINATOR_SSH_USER:-ree}"

ssh_host() {
  local v
  v="$(env_val "SHADOW_SSH_HOST_$1")"
  echo "${v:-$1}"
}

ssh_run() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new \
    "${SSH_USER}@${host}" "$@"
}

echo "=== Phase 2 cutover: drain fleet ==="
curl -sf -X POST "$SERVE/api/runner/v3/stop" >/dev/null || true
sleep 3
if curl -sf "$SERVE/api/runner/status" | grep -q '"running": true'; then
  echo "Mac runner still running after stop; force_stop"
  curl -sf -X POST "$SERVE/api/runner/v3/force_stop" >/dev/null || true
  sleep 2
fi

for c in ree-cloud-1 ree-cloud-2 ree-cloud-3 ree-cloud-4; do
  h="$(ssh_host "$c")"
  echo "stop ree-runner on $c ($h)"
  ssh_run "$h" "sudo systemctl kill -s SIGTERM ree-runner 2>/dev/null; sleep 2; systemctl is-active ree-runner 2>/dev/null || echo stopped" || echo "WARN: stop failed on $c"
done
sleep 2

echo "=== Hub: coordinator + sync modes ==="
HUB="$(ssh_host ree-cloud-1)"
ssh_run "$HUB" "sudo sed -i 's/^COORDINATOR_MODE=.*/COORDINATOR_MODE=coordinator/' /etc/ree-coordinator.env && \
  sudo sed -i 's/^SYNC_MODE=.*/SYNC_MODE=coordinator/' /etc/ree-coordinator.env && \
  sudo systemctl restart ree-coordinator ree-sync-daemon && sleep 2 && \
  curl -sf http://10.8.0.1:8787/health"

echo "=== Workers: COORDINATION_MODE=coordinator ==="
FLIP="sudo sed -i 's/COORDINATION_MODE=shadow/COORDINATION_MODE=coordinator/' \
  /etc/systemd/system/ree-runner.service.d/shadow.conf && \
  sudo systemctl daemon-reload"
for c in ree-cloud-1 ree-cloud-2 ree-cloud-3 ree-cloud-4; do
  h="$(ssh_host "$c")"
  echo "flip $c ($h)"
  ssh_run "$h" "$FLIP"
done

echo "=== Resume: POST /api/coordinator/start (serve.py must include endpoint) ==="
curl -sf -X POST "$SERVE/api/coordinator/start" | python3 -m json.tool

echo "=== Verify ==="
curl -sf "${COORDINATOR_URL%/}/health" | python3 -m json.tool
/opt/local/bin/python3 "$BASE/ree-v3/coordinator/check_shadow.py" \
  --url "$COORDINATOR_URL" --token "$COORDINATOR_LOCAL_TOKEN" || true
