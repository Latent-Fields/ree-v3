#!/bin/bash
# REE role arbiter -- dual-role arbitration (ree-cloud-4), on its OWN fast,
# free cadence. Reference copy -- the INSTALLED copy lives at
# /usr/local/bin/ree_role_arbiter.sh; this tracked one is canonical, same
# convention as ree-metaworker-dispatch.sh / ree-metaworker-healer.sh (see
# check_metaworker_wrapper_deploy.py).
#
# WHY THIS IS ITS OWN SCRIPT/TIMER, NOT PART OF THE HEALER.
#
# This exact logic used to live inside ree-metaworker-healer.sh's "Dual-role
# arbitration" block (moved there 2026-08-26, chip-20260825-cloud4-
# arbitration-into-healer, from the Dispatcher wrapper before that). That
# placement inherited the Healer's HOURLY cadence
# (ree-metaworker-healer.timer, OnUnitActiveSec=1h) -- a cadence chosen for a
# different cost reason entirely: the Healer pays a `claude` invocation (a
# ~117k-token context floor) per cycle when there is repair work, and hourly
# was the right interval to bound THAT cost.
#
# Dual-role arbitration has NO such cost -- it is plain `git fetch` + python +
# `systemctl`, no `claude` invocation, ever -- so tying it to the Healer's
# cost-driven hourly cadence was an unforced coupling, and it was never free:
# measured live on ree-cloud-4, 2026-08-26 through 2026-08-29 (every "role:"
# line in ~/ree_metaworker_healer.log across that whole window), the Healer's
# hourly arbitration check NEVER ONCE actually issued a `systemctl start` or
# `systemctl stop` for ree-runner -- 0 actions in ~70 hourly ticks. Every real
# role transition in that window (and the ones before the Healer existed) was
# instead caught by `ree-runner-failsafe.timer`'s 15-MINUTE poll -- the
# NEGATIVE-space watchdog scripts/ree_runner_failsafe.py, which is supposed to
# be this arbitration's rare backstop for when it fails to run at all, not its
# only-ever-working implementation. A queue surge-to-drain-complete transition
# routinely completes well inside an hour, so the tighter 15-minute cadence
# wins the race essentially every time -- which inverted the intended
# primary/backstop relationship without anyone deciding that on purpose.
#
# THE FIX: pull this block out into its own script on its own timer, paced by
# what it actually costs (git fetch + python + systemctl -- cheap enough to
# run every couple of minutes indefinitely) rather than by what an unrelated
# sibling process costs. This is what actually restores the failsafe to being
# a rare backstop instead of the load-bearing mechanism. See
# scripts/ree_runner_failsafe.py's own module docstring for the failsafe side
# of this story, and chip-20260829-cloud4-runner-failsafe for the incident
# that found the 0-actions-in-70-ticks measurement above.
#
# Logs: journalctl -u ree-role-arbiter -f  |  ~/ree_role_arbiter.log
set -uo pipefail

export PATH="/opt/local/bin:/usr/local/bin:/home/ree/.local/bin:/usr/bin:/bin"

REPO="${REE_REPO:-/home/ree/REE_Working}"
LOG="${REE_ARBITER_LOG:-/home/ree/ree_role_arbiter.log}"
LOCK="${REE_ARBITER_LOCK:-/tmp/ree_role_arbiter.lock}"
MACHINE="${REE_MACHINE:-$(hostname)}"
PY="$(command -v python3 || echo /usr/bin/python3)"
# Own state dir -- separate from the Healer's ($HOME/.ree_metaworker_healer)
# and the Dispatcher's ($HOME/.ree_metaworker), same reasoning as those two
# keeping independent scratch dirs: the queue snapshot cache below is
# per-wrapper scratch, not shared coordination state.
STATE_DIR="${REE_ARBITER_STATE_DIR:-$HOME/.ree_role_arbiter}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Alive dispatched chip-WORKER count on this box -- exact copy of
# ree-metaworker-dispatch.sh's/ree-metaworker-healer.sh's own
# count_alive_dispatches (worktree names and the .dispatch_pid convention are
# a box-wide fact, not owned by any one wrapper). Kept as a literal duplicate
# rather than a shared helper file -- same reasoning as every other
# vendored/duplicated helper in this codebase (CLAUDE.md, Session Startup
# step 7a): these are independently-deployed /usr/local/bin scripts outside
# any repo's import path.
count_alive_dispatches() {
  local n=0
  for wt in "$REPO"/.claude/worktrees/metaworker-*/; do
    [ -d "$wt" ] || continue
    local pid
    pid=$(cat "$wt/.dispatch_pid" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

# Non-reentrancy: a slow git fetch (network hiccup) must not pile a second
# cycle on top of one still in flight. Own lock, separate from the Healer's
# and the Dispatcher's -- this timer's cadence is independent of both.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(ts)] arbiter: another cycle holds $LOCK; skipping" >> "$LOG"
  exit 0
fi

cd "$REPO" || { echo "[$(ts)] arbiter: no repo at $REPO" >> "$LOG"; exit 1; }

# OPT-IN, same gate as the Healer's block used to carry: a box that is not a
# dual-role box must never touch ree-runner or metaworker_role_verdict.py.
# Set REE_DUAL_ROLE=1 in the systemd drop-in on dual-role boxes only
# (ree-cloud-4 today).
VERDICT_SCRIPT="$REPO/scripts/metaworker_role_verdict.py"
if [ "${REE_DUAL_ROLE:-0}" != "1" ] || [ ! -f "$VERDICT_SCRIPT" ]; then
  echo "[$(ts)] arbiter: not a dual-role box (REE_DUAL_ROLE unset) or verdict script missing; exiting" >> "$LOG"
  exit 0
fi

mkdir -p "$STATE_DIR"

# Machine affinity for the verdict's --machine arg (e.g. "ree-cloud-4"),
# resolved through machine_identity.canonical_machine_name -- NOT a raw
# hostname compare (CLAUDE.md machine-identity note). Falls back to the short
# hostname if the resolver is unavailable, and strips a "-metaworker" suffix
# defensively in case REE_LEASE_AFFINITY/REE_MACHINE is ever set to the
# heartbeat-naming form by mistake.
ROLE_AFFINITY="${REE_LEASE_AFFINITY:-}"
if [ -z "$ROLE_AFFINITY" ]; then
  ROLE_AFFINITY=$("$PY" -c '
import socket, sys
sys.path.insert(0, sys.argv[1])
raw = socket.gethostname()
try:
    from machine_identity import canonical_machine_name
    print(canonical_machine_name(raw) or raw.split(".")[0])
except Exception:
    print(raw.split(".")[0])
' "$REPO/ree-v3" 2>/dev/null)
fi
[ -n "$ROLE_AFFINITY" ] || ROLE_AFFINITY=$(hostname -s 2>/dev/null || echo unknown-metaworker)
case "$ROLE_AFFINITY" in *-metaworker) ROLE_AFFINITY="${ROLE_AFFINITY%-metaworker}" ;; esac

_runner_active=0
systemctl is-active --quiet ree-runner 2>/dev/null && _runner_active=1

# READ THE QUEUE FROM origin/main, NOT the working tree -- same reasoning as
# ree-metaworker-dispatch.sh's own comment: stopping ree-runner removes the
# only thing pulling ree-v3 on this box, so the working-tree queue file
# freezes the moment the box goes into metaworker mode, and the verdict would
# never see a surge again. Fetch + `git show` is read-only and immune to
# whatever divergence this checkout already has.
_qfile="$STATE_DIR/queue_snapshot.json"
if git -C "$REPO/ree-v3" fetch -q origin main 2>>"$LOG" \
   && git -C "$REPO/ree-v3" show origin/main:experiment_queue.json > "$_qfile.tmp" 2>>"$LOG" \
   && [ -s "$_qfile.tmp" ]; then
  mv -f "$_qfile.tmp" "$_qfile"
else
  rm -f "$_qfile.tmp"
  echo "[$(ts)] arbiter: WARN could not read queue from origin/main; falling back to the working-tree copy (may be stale)" >> "$LOG"
  _qfile="$REPO/ree-v3/experiment_queue.json"
fi

_v=$("$PY" "$VERDICT_SCRIPT" \
      --queue "$_qfile" \
      --machine "$ROLE_AFFINITY" \
      --in-flight "$(count_alive_dispatches)" \
      --runner-active "$_runner_active" 2>>"$LOG")
if [ -n "$_v" ]; then
  _role_state=$(printf '%s' "$_v" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null || echo dispatching)
  _reason=$(printf '%s' "$_v" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["reason"])' 2>/dev/null || echo "")
  _start=$(printf '%s' "$_v" | "$PY" -c 'import json,sys; print(int(json.load(sys.stdin)["start_runner"]))' 2>/dev/null || echo 0)
  _stop=$(printf '%s' "$_v" | "$PY" -c 'import json,sys; print(int(json.load(sys.stdin)["stop_runner"]))' 2>/dev/null || echo 0)
  echo "[$(ts)] role: $_role_state -- $_reason" >> "$LOG"
  if [ "$_start" = "1" ]; then
    sudo -n systemctl start ree-runner >>"$LOG" 2>&1 \
      && echo "[$(ts)] role: started ree-runner (drain complete)" >> "$LOG"
  fi
  if [ "$_stop" = "1" ]; then
    # Safe by construction: the verdict only requests this when the runner
    # holds NO claim, so nothing is interrupted.
    sudo -n systemctl stop ree-runner >>"$LOG" 2>&1 \
      && echo "[$(ts)] role: stopped ree-runner (holds no claim)" >> "$LOG"
  fi
else
  echo "[$(ts)] arbiter: verdict script produced no output" >> "$LOG"
fi
