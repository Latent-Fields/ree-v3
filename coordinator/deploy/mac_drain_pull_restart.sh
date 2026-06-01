#!/usr/bin/env bash
# mac_drain_pull_restart.sh
#
# Unattended cutover for DLAPTOP-4.local (the user-launched runner that
# has no systemd supervisor). Drains the local V3 runner gracefully,
# waits for it to finish its in-flight experiment and exit, then pulls
# the latest ree-v3 main and re-launches it via the explorer's
# /api/runner/v3/start endpoint.
#
# Use this when a code change to experiment_runner.py needs to land on
# the Mac without interrupting the current experiment. Cloud-1/cloud-2
# use systemctl restart -- this is the Mac-side equivalent.
#
# Foreground watcher: run inside a long-lived terminal (or under nohup
# if you want to walk away). Exits 0 on success, non-zero on any step
# that needed manual intervention.
#
# Env overrides:
#   SERVE_URL       default http://localhost:8000
#   REPO_ROOT       default /Users/dgolden/REE_Working/ree-v3
#   POLL_INTERVAL   default 10 (seconds between PID checks)
#   MAX_DRAIN_SECS  default 14400 (4h ceiling; longer than any V3 exp)

set -u

SERVE_URL="${SERVE_URL:-http://localhost:8000}"
REPO_ROOT="${REPO_ROOT:-/Users/dgolden/REE_Working/ree-v3}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
MAX_DRAIN_SECS="${MAX_DRAIN_SECS:-14400}"

PID_FILE="$REPO_ROOT/runner.pid"

log() { printf '[mac-drain] %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

# ----- 1. Confirm the runner is up and grab its PID --------------------------
if [ ! -f "$PID_FILE" ]; then
  log "no PID file at $PID_FILE -- runner not running; nothing to drain."
  log "running git pull then start anyway."
  PID=""
else
  PID="$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')"
  if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
    log "PID file present but process $PID not running -- stale; treating as down."
    PID=""
  else
    log "current runner PID=$PID"
  fi
fi

# ----- 2. Request graceful drain --------------------------------------------
if [ -n "$PID" ]; then
  log "POST $SERVE_URL/api/runner/v3/stop (graceful drain)"
  if ! curl -sf -X POST "$SERVE_URL/api/runner/v3/stop" >/dev/null; then
    log "WARN: stop endpoint did not return 2xx; falling back to SIGTERM on $PID."
    kill -TERM "$PID" 2>/dev/null || true
  fi

  # ----- 3. Wait for the process to exit ------------------------------------
  log "waiting up to ${MAX_DRAIN_SECS}s for PID $PID to exit (polling every ${POLL_INTERVAL}s)"
  start_ts=$(date +%s)
  while kill -0 "$PID" 2>/dev/null; do
    now=$(date +%s)
    elapsed=$(( now - start_ts ))
    if [ "$elapsed" -ge "$MAX_DRAIN_SECS" ]; then
      log "ERROR: drain ceiling ${MAX_DRAIN_SECS}s exceeded; PID $PID still alive."
      log "manual intervention required: investigate why the experiment did not finish."
      log "to force-kill: curl -X POST $SERVE_URL/api/runner/v3/force_stop"
      exit 2
    fi
    sleep "$POLL_INTERVAL"
  done
  log "PID $PID exited after ${elapsed}s."
fi

# ----- 4. git pull ree-v3 main ----------------------------------------------
log "git -C $REPO_ROOT pull origin main"
if ! git -C "$REPO_ROOT" pull --ff-only origin main; then
  log "ERROR: git pull failed; not restarting runner."
  log "investigate the conflict, then POST $SERVE_URL/api/runner/v3/start manually."
  exit 3
fi

# ----- 5. Relaunch runner via explorer ---------------------------------------
log "POST $SERVE_URL/api/runner/v3/start"
if ! curl -sf -X POST "$SERVE_URL/api/runner/v3/start" >/dev/null; then
  log "ERROR: start endpoint did not return 2xx."
  log "check serve.py is running; if so, retry: curl -X POST $SERVE_URL/api/runner/v3/start"
  exit 4
fi

# Brief sanity check: the PID file should reappear within a few seconds.
sleep 3
if [ -f "$PID_FILE" ]; then
  NEW_PID="$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')"
  log "runner restarted: PID=$NEW_PID"
else
  log "WARN: no PID file after start request; check serve.py logs."
fi

log "done."
exit 0
