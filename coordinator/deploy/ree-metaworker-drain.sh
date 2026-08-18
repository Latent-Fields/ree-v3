#!/bin/bash
# Drain in-flight metaworker chip workers before this box powers off.
#
# Reference copy -- install as /usr/local/bin/ree_metaworker_drain.sh, root-owned,
# 755. Invoked as the ExecStop of ree-metaworker-drain.service, which is ordered
# Before=shutdown.target, so it runs on ANY stop path: `hcloud server shutdown`
# (ACPI -> systemd shutdown), `systemctl poweroff`, or an operator reboot.
#
# WHY THIS EXISTS. The cloud-scaler's shutdown is graceful for EXPERIMENTS -- the
# runner has a SIGTERM handler that drains the current run -- but nothing drained
# chip workers. A dispatched worker is a `claude -p` process in its own worktree
# with no signal handling; a box shutdown simply kills it mid-edit, and its chip
# stays claimed until CLAIM_STALE_HOURS (3h) before anyone can pick it up again.
# The work is not corrupted (the chip ledger is a compare-and-swap and the claim
# is honoured), but it is LOST and re-done from scratch.
#
# Confirmed live 2026-08-18: ree-worker-4 was shut down mid-cycle with a
# dispatched worker on it, because the veto keeping the box alive aged out under
# a long cycle. The veto bug is fixed separately; this is the second half --
# even a CORRECT shutdown decision must not kill live workers.
#
# WHAT IT DOES NOT DO: it never kills a worker and never resolves a chip. It
# stops NEW dispatch, then waits. If the wait times out, the worker dies exactly
# as it would have without this script -- the timeout is a bound on how long a
# shutdown can be delayed, not a promise that draining always completes.

set -u

export PATH="/opt/local/bin:/usr/local/bin:/home/ree/.local/bin:/usr/bin:/bin"
REPO="${REE_REPO:-/Users/dgolden/REE_Working}"
WORKTREE_DIR="$REPO/.claude/worktrees"
LOG="${REE_DRAIN_LOG:-/home/ree/ree_metaworker_dispatch.log}"

# Bound on the wait. A dispatch cycle's own hard cap is REE_DISPATCH_MAX_SEC
# (1500s / 25min), so a worker cannot outlive that by much. 1200s sits below the
# ~26min ACPI window Hetzner allows a draining box before a hard power-off --
# past that the host pulls the plug anyway and a longer bound buys nothing while
# making an operator reboot feel wedged.
MAX_WAIT="${REE_DRAIN_MAX_SEC:-1200}"
POLL="${REE_DRAIN_POLL_SEC:-10}"

ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] drain: $*" >> "$LOG" 2>/dev/null || true; echo "drain: $*"; }

# `kill -0` succeeds on a ZOMBIE -- the pid still exists until its parent reaps
# it -- so a kill -0 check alone can wait out the full bound on a process that
# has already finished. In production the dispatcher exits first and its workers
# are reparented to init, which reaps them promptly, so this is rare rather than
# routine; it was found by a test whose child Python had not reaped, and it is
# guarded here because "rare" and "never" are different and the cost of being
# wrong is a shutdown stalled for the full 20 minutes. /proc is Linux-only and
# this script only ever runs on the cloud boxes, but the check degrades to the
# old kill -0 behaviour anywhere /proc is absent rather than breaking.
_pid_is_live() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 1
  if [ -r "/proc/$pid/stat" ]; then
    # field 3 is the state character; Z = zombie, already finished
    [ "$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)" != "Z" ] || return 1
  fi
  return 0
}

count_alive_dispatches() {
  local n=0 pid
  for wt in "$WORKTREE_DIR"/metaworker-*/; do
    [ -d "$wt" ] || continue
    pid=$(cat "$wt/.dispatch_pid" 2>/dev/null)
    if [ -n "$pid" ] && _pid_is_live "$pid"; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

# 1. Stop NEW work first. Without this the timer keeps firing during the drain
#    and the box never reaches zero in-flight. --no-block so we do not deadlock
#    against systemd's own shutdown transaction.
log "stopping ree-metaworker.timer so no new cycles start"
systemctl stop --no-block ree-metaworker.timer 2>/dev/null || true

ALIVE=$(count_alive_dispatches)
if [ "$ALIVE" -eq 0 ]; then
  log "no in-flight chip workers; nothing to drain"
  exit 0
fi

log "waiting for $ALIVE in-flight chip worker(s), bound ${MAX_WAIT}s"
WAITED=0
while [ "$WAITED" -lt "$MAX_WAIT" ]; do
  ALIVE=$(count_alive_dispatches)
  [ "$ALIVE" -eq 0 ] && { log "drained cleanly after ${WAITED}s"; exit 0; }
  sleep "$POLL"
  WAITED=$((WAITED + POLL))
  [ $((WAITED % 60)) -eq 0 ] && log "still draining: $ALIVE worker(s), ${WAITED}s elapsed"
done

# Timed out. Say so LOUDLY and name the survivors -- their chips stay claimed
# until CLAIM_STALE_HOURS and will be redone, which is worth knowing about.
ALIVE=$(count_alive_dispatches)
log "TIMEOUT after ${MAX_WAIT}s with $ALIVE worker(s) still running; they will be"
log "  killed by the shutdown. Their chips stay claimed until CLAIM_STALE_HOURS"
log "  (3h) and are then re-dispatched from scratch. Survivors:"
for wt in "$WORKTREE_DIR"/metaworker-*/; do
  [ -d "$wt" ] || continue
  pid=$(cat "$wt/.dispatch_pid" 2>/dev/null)
  if [ -n "$pid" ] && _pid_is_live "$pid"; then
    log "    $(basename "$wt") pid=$pid"
  fi
done
exit 0   # never block the shutdown itself
