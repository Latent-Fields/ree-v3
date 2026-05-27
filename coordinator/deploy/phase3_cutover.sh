#!/usr/bin/env bash
# Phase-3 result cutover: drain fleet, run preflight, flip hub (when ready).
# REFUSES to continue unless phase3_preflight.py exits 0.
#
# Phase 3 git writer is NOT implemented yet -- this script stops after
# preflight today. Do not pass --force.
set -euo pipefail

BASE="${REE_WORKING:-$HOME/REE_Working}"
PYTHON="${PYTHON:-/opt/local/bin/python3}"
PREFLIGHT="$BASE/ree-v3/coordinator/phase3_preflight.py"
VERIFY="$BASE/ree-v3/coordinator/phase3_verify.py"
ENV_FILE="$BASE/REE_assembly/coordinator.env"
SERVE="${SERVE_URL:-http://127.0.0.1:8000}"

FORCE=0
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    --dry-run) DRY_RUN=1 ;;
  esac
done

if [[ ! -f "$PREFLIGHT" ]]; then
  echo "ERROR: missing $PREFLIGHT" >&2
  exit 1
fi

echo "=== Phase 3 cutover: preflight (required, cutover-window strict) ==="
# --cutover-window requires fleet_lifecycle=live for every expected peer.
# phase3_wake_fleet.sh must have run successfully before this.
PREF_ARGS=(--cutover-window)
[[ "$DRY_RUN" -eq 1 ]] && PREF_ARGS+=(--dry-run)
if ! "$PYTHON" "$PREFLIGHT" "${PREF_ARGS[@]}"; then
  echo "ERROR: phase3_preflight failed -- aborting cutover" >&2
  exit 1
fi

# Import gate: writer must be implemented before hub flip.
READY=$("$PYTHON" -c "
import sys
sys.path.insert(0, '$BASE/ree-v3/coordinator')
import sync_daemon
print('1' if sync_daemon.PHASE3_GIT_WRITER_READY else '0')
")
if [[ "$READY" != "1" && "$FORCE" -ne 1 ]]; then
  echo "STOP: PHASE3_GIT_WRITER_READY is not True in sync_daemon.py" >&2
  echo "  Phase 3 substrate only -- implement phase3_git_writer before hub flip." >&2
  echo "  See ree-v3/coordinator/PHASE3_CUTOVER.md" >&2
  exit 1
fi

if [[ "$FORCE" -eq 1 ]]; then
  echo "WARN: --force passed; proceeding past writer-ready gate" >&2
fi

echo "=== Phase 3 cutover: drain fleet ==="
curl -sf -X POST "$SERVE/api/runner/v3/stop" >/dev/null || true
sleep 3
if curl -sf "$SERVE/api/runner/status" 2>/dev/null | grep -q '"running": true'; then
  echo "Mac runner still running; force_stop"
  curl -sf -X POST "$SERVE/api/runner/v3/force_stop" >/dev/null || true
  sleep 2
fi

env_val() {
  grep -E "^${1}=" "$ENV_FILE" 2>/dev/null | head -1 | cut -d= -f2-
}
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

for c in ree-cloud-1 ree-cloud-2 ree-cloud-3 ree-cloud-4; do
  h="$(ssh_host "$c")"
  echo "stop ree-runner on $c ($h)"
  ssh_run "$h" "sudo systemctl kill -s SIGTERM ree-runner 2>/dev/null; sleep 2; systemctl is-active ree-runner 2>/dev/null || echo stopped" || echo "WARN: stop failed on $c"
done
sleep 2

echo "=== Hub: SYNC_MODE=authoritative (maintenance) ==="
HUB="$(ssh_host ree-cloud-1)"
ssh_run "$HUB" "sudo sed -i 's/^SYNC_MODE=.*/SYNC_MODE=authoritative/' /etc/ree-coordinator.env && \
  grep SYNC_MODE /etc/ree-coordinator.env && \
  sudo systemctl restart ree-sync-daemon && sleep 2 && \
  systemctl is-active ree-sync-daemon"

echo "=== Post-cutover verify ==="
if [[ -f "$VERIFY" ]]; then
  "$PYTHON" "$VERIFY" --expect-cutover || true
fi

echo "=== Phase 3 cutover script finished (manual runner resume required) ==="
echo "  Re-enable workers only after phase3_verify.py is all PASS."
