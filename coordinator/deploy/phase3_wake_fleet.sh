#!/usr/bin/env bash
# Phase-3 wake-fleet: prepare the fleet for cutover by ensuring every
# expected peer is up and lifecycle_state=live on /shadow/status.
#
# Pairs with phase3_release_fleet.sh -- run release AFTER phase3_verify.py
# passes, NOT after this script.
#
# Steps:
#   1. Disable the cloud-scaler workflow (gh CLI) so it won't shut down
#      newly-woken workers during the cutover window.
#   2. For each expected peer (ree-worker-1..4 by default), check its
#      Hetzner status; poweron if off.
#   3. Poll /shadow/status until every expected machine_affinity shows
#      lifecycle_state=live (or timeout).
#
# Exits 0 only when the fleet is fully live (or all expected peers either
# live or gracefully_offline within timeout -- gracefully_offline accepted
# only with --accept-graceful which the cutover path does NOT use).
#
# Requires: gh (authenticated), hcloud CLI + HCLOUD_TOKEN env, the Mac's
# REE_assembly/coordinator.env (for COORDINATOR_URL +
# COORDINATOR_LOCAL_TOKEN), and Python 3.
#
# All output is ASCII-only.

set -euo pipefail

BASE="${REE_WORKING:-$HOME/REE_Working}"
PYTHON="${PYTHON:-/opt/local/bin/python3}"
ENV_FILE="$BASE/REE_assembly/coordinator.env"
REPO="${REPO:-Latent-Fields/ree-v3}"
WORKFLOW="${WORKFLOW:-cloud-scaler.yml}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

# Hetzner server -> machine_affinity pairs. Matches WORKERS in
# .github/workflows/cloud-scaler.yml.
WORKERS=(
  "ree-worker-1:ree-cloud-1"
  "ree-worker-2:ree-cloud-2"
  "ree-worker-3:ree-cloud-3"
  "ree-worker-4:ree-cloud-4"
)

DRY_RUN=0
SKIP_DISABLE=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --skip-disable-scaler) SKIP_DISABLE=1 ;;
    --help|-h)
      echo "Usage: $0 [--dry-run] [--skip-disable-scaler]"
      echo "  --dry-run            print intended actions, don't execute"
      echo "  --skip-disable-scaler  poweron + poll without touching the scaler workflow"
      exit 0
      ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: $*"
  else
    "$@"
  fi
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: missing $ENV_FILE (need COORDINATOR_URL + COORDINATOR_LOCAL_TOKEN)" >&2
  exit 1
fi
# Source only the keys we need; skip SHADOW_SSH_HOST_* lines which contain
# shell-unfriendly characters in some setups.
COORDINATOR_URL=$(grep -E '^COORDINATOR_URL=' "$ENV_FILE" | head -1 | cut -d= -f2-)
COORDINATOR_LOCAL_TOKEN=$(grep -E '^COORDINATOR_LOCAL_TOKEN=' "$ENV_FILE" | head -1 | cut -d= -f2-)
if [[ -z "$COORDINATOR_URL" || -z "$COORDINATOR_LOCAL_TOKEN" ]]; then
  echo "ERROR: COORDINATOR_URL or COORDINATOR_LOCAL_TOKEN missing from $ENV_FILE" >&2
  exit 1
fi

command -v gh >/dev/null || { echo "ERROR: gh CLI not installed"; exit 1; }
command -v hcloud >/dev/null || { echo "ERROR: hcloud CLI not installed"; exit 1; }
[[ -n "${HCLOUD_TOKEN:-}" ]] || { echo "ERROR: HCLOUD_TOKEN env var required" >&2; exit 1; }

echo "=== Step 1: disable cloud-scaler workflow ==="
if [[ "$SKIP_DISABLE" -eq 1 ]]; then
  echo "  --skip-disable-scaler: leaving cloud-scaler.yml as-is"
else
  STATE=$(gh api "repos/$REPO/actions/workflows/$WORKFLOW" -q .state 2>/dev/null || echo unknown)
  echo "  current state: $STATE"
  if [[ "$STATE" == "active" ]]; then
    run gh workflow disable "$WORKFLOW" --repo "$REPO"
    echo "  scaler disabled"
  else
    echo "  scaler already $STATE; no change"
  fi
fi

echo
echo "=== Step 2: poweron offline cloud workers ==="
for entry in "${WORKERS[@]}"; do
  IFS=: read -r SERVER AFFINITY <<< "$entry"
  STATUS=$(hcloud server describe "$SERVER" -o format='{{.Status}}' 2>/dev/null || echo unknown)
  echo "  $SERVER ($AFFINITY): $STATUS"
  if [[ "$STATUS" == "off" ]]; then
    run hcloud server poweron "$SERVER"
  elif [[ "$STATUS" == "unknown" ]]; then
    echo "    WARN: hcloud could not describe $SERVER; skipping"
  fi
done

echo
echo "=== Step 3: poll /shadow/status for lifecycle=live ==="
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY-RUN: would poll $COORDINATOR_URL/shadow/status every ${POLL_INTERVAL}s"
  echo "         until ree-cloud-1..4 + DLAPTOP-4.local all show lifecycle_state=live"
  echo "         (timeout ${TIMEOUT_SECONDS}s); skipping in dry-run"
  exit 0
fi
deadline=$(($(date +%s) + TIMEOUT_SECONDS))
FLEET_LIVE=0
EVER_PARSED_RESPONSE=0
LAST_PENDING=""
while true; do
  SNAPSHOT=$(curl -fsS -H "Authorization: Bearer $COORDINATOR_LOCAL_TOKEN" \
                 "$COORDINATOR_URL/shadow/status" 2>/dev/null) || {
    echo "  WARN: /shadow/status unreachable; retrying"
    if [[ $(date +%s) -ge $deadline ]]; then
      break
    fi
    sleep "$POLL_INTERVAL"
    continue
  }
  # Build a "needed but not-live" list. Mac (DLAPTOP-4.local) is also
  # required and IS expected to be live on the operator's machine
  # because they're running this script there.
  # NOTE: parse via `if ! VAR=$(...)` so an unparseable JSON response does
  # NOT silently produce an empty PENDING -- the empty-pending branch below
  # treats empty as "all live", which combined with a python KeyError /
  # JSONDecodeError would be a second false-positive path.
  if ! PENDING=$(echo "$SNAPSHOT" | "$PYTHON" -c "
import json, sys
d = json.load(sys.stdin)
needed = {'ree-cloud-1','ree-cloud-2','ree-cloud-3','ree-cloud-4','DLAPTOP-4.local'}
seen = {m['machine']: m.get('lifecycle_state','?') for m in d.get('machines', [])}
pending = []
for n in sorted(needed):
    state = seen.get(n, 'missing')
    if state != 'live':
        pending.append(f'{n}={state}')
print('|'.join(pending))
" 2>/dev/null); then
    echo "  WARN: /shadow/status returned unparseable response; retrying"
    if [[ $(date +%s) -ge $deadline ]]; then
      break
    fi
    sleep "$POLL_INTERVAL"
    continue
  fi
  EVER_PARSED_RESPONSE=1
  LAST_PENDING="$PENDING"
  if [[ -z "$PENDING" ]]; then
    echo "  all expected peers live; fleet ready"
    FLEET_LIVE=1
    break
  fi
  echo "  pending: ${PENDING//|/, }   (deadline in $((deadline - $(date +%s)))s)"
  if [[ $(date +%s) -ge $deadline ]]; then
    break
  fi
  sleep "$POLL_INTERVAL"
done

echo
echo "=== Wake complete ==="
if [[ "$FLEET_LIVE" -eq 1 ]]; then
  echo "  RESULT: fleet truly live -- all expected peers confirmed lifecycle_state=live"
  echo "  Run phase3_preflight.py next."
  echo "  After phase3_verify.py PASSes, run phase3_release_fleet.sh to"
  echo "  re-enable the cloud-scaler workflow."
  exit 0
elif [[ "$EVER_PARSED_RESPONSE" -eq 0 ]]; then
  echo "  RESULT: exiting on polling failure -- coordinator never responded with a parseable status" >&2
  echo "  $COORDINATOR_URL/shadow/status was unreachable (or unparseable) for the entire ${TIMEOUT_SECONDS}s window." >&2
  echo "  Likely causes: coordinator process down, hub WireGuard tunnel broken, or wrong COORDINATOR_URL." >&2
  echo "  DO NOT proceed to phase3_preflight.py until the coordinator is reachable." >&2
  exit 3
else
  if [[ -n "$LAST_PENDING" ]]; then
    N_PENDING=$(echo -n "$LAST_PENDING" | awk -F'|' '{print NF}')
  else
    N_PENDING="unknown"
  fi
  echo "  RESULT: exiting on timeout -- ${N_PENDING} peer(s) never confirmed live within ${TIMEOUT_SECONDS}s" >&2
  echo "  pending at deadline: ${LAST_PENDING//|/, }" >&2
  echo "  DO NOT proceed to phase3_preflight.py with peers not live." >&2
  exit 2
fi
