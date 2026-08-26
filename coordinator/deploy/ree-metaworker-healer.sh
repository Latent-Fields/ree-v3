#!/bin/bash
# REE metaworker HEALER -- one repair-only cycle of /metaworker-repair.
#
# Reference copy. The INSTALLED copy lives at
# /usr/local/bin/ree_metaworker_healer.sh; this tracked one is canonical, same
# convention as ree-metaworker-dispatch.sh (see check_metaworker_wrapper_deploy.py).
#
# WHAT MAKES THIS DIFFERENT FROM THE DISPATCH WRAPPER, and why it can run
# resident while the Dispatcher cannot:
#
#   1. IT NEVER DISPATCHES `kind: work`. /metaworker-repair has no Step 4
#      dispatch lane at all. That is what makes it safe to leave running
#      unattended -- it cannot spend the token budget on new work, so it needs
#      no run lease from the Orchestrator. The plumbing has to stay healthy
#      whether or not anyone is dispatching, which is the whole reason the
#      roles were split on 2026-08-25.
#
#   2. IT GATES ON WORK EXISTING, BEFORE LOADING ANY CONTEXT. This is the
#      direct lesson of the 2026-08-23 pause: ~513 dispatcher ticks in one day
#      across two boxes, each paying a ~117k-token context floor (CLAUDE.md +
#      SKILL.md) BEFORE deciding it had nothing to do -- roughly 59M tokens a
#      day of mostly-idle overhead. So the cheap check runs first, in plain
#      python, and `claude` is never invoked when there is nothing to repair.
#      Do not "simplify" by moving this gate inside the skill: inside the
#      skill, the floor has already been paid.
#
# DUAL-ROLE ARBITRATION (ree-cloud-4) LIVES HERE, NOT IN THE DISPATCHER,
# since 2026-08-26 (chip-20260825-cloud4-arbitration-into-healer). Before the
# 2026-08-25 Dispatcher/Healer split, `metaworker_role_verdict.py` +
# `sudo systemctl start/stop ree-runner` ran inside ree-metaworker-dispatch.sh
# on that wrapper's own resident timer, so the box's role was re-evaluated on
# a steady cadence regardless of anything else. After the split the Dispatcher
# runs ONLY while an Orchestrator session holds a run lease
# (dispatcher_control.json) -- nothing grants ree-cloud-4 a standing one -- so
# arbitration would go inert the moment nobody happens to lease it, unable to
# even reverse a stale "runner owns the box" verdict when a real surge later
# arrives (chip-20260825-cloud4-role-arbitration-lease-gap). Moving it here
# fits the Healer's own charter above: role arbitration never spends the
# token budget on new work (plain systemctl/python, no `claude` invocation),
# so it costs nothing to keep resident and unleased, and "does this box's
# role match reality right now" is inherently a resident-cadence question,
# not a lease-gated-dispatch one. The Dispatcher keeps a narrow, read-only
# safety net of its own -- see its "Runner-ownership guard" comment -- so a
# lease granted to cloud-4 for an unrelated reason still never dispatches
# chip work onto a box the runner already owns; it is not a second decision
# maker, since it never calls the verdict script or touches systemctl beyond
# a read-only `is-active` check.
#
# Logs: journalctl -u ree-metaworker-healer -f  |  ~/ree_metaworker_healer.log
set -uo pipefail

# systemd gives a Type=oneshot unit a MINIMAL PATH, and this script is
# #!/bin/bash rather than a login shell, so ~/.profile never runs and
# `command -v claude` finds nothing -- claude lives in ~/.local/bin. Measured
# on ree-cloud-5 2026-08-25: the first `systemctl start` of this unit exited
# 127 with "no claude binary on PATH", caught by smoke-testing before
# enabling the timer. Same line, same order, as ree-metaworker-dispatch.sh.
export PATH="/opt/local/bin:/usr/local/bin:/home/ree/.local/bin:/usr/bin:/bin"

REPO="${REE_REPO:-/home/ree/REE_Working}"
LOG="${REE_HEALER_LOG:-/home/ree/ree_metaworker_healer.log}"
LOCK="${REE_HEALER_LOCK:-/tmp/ree_metaworker_healer.lock}"
MAX_SEC="${REE_HEALER_MAX_SEC:-900}"
MACHINE="${REE_MACHINE:-$(hostname)}"
PY="$(command -v python3 || echo /usr/bin/python3)"
# Own state dir, separate from the Dispatcher's ($HOME/.ree_metaworker) even
# on a box that runs both wrappers -- the queue snapshot cache below is
# per-wrapper scratch, not shared coordination state, and giving each wrapper
# its own directory means neither can race the other's cache file.
STATE_DIR="${REE_HEALER_STATE_DIR:-$HOME/.ree_metaworker_healer}"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Alive dispatched chip-WORKER count on this box -- exact copy of
# ree-metaworker-dispatch.sh's own count_alive_dispatches (worktree names and
# the .dispatch_pid convention are a box-wide fact, not owned by either
# wrapper: only the Dispatcher's Step 4 ever creates one, but both wrappers
# may need to read the count). Kept as a literal duplicate rather than a
# shared helper file -- these are two independently-deployed
# /usr/local/bin scripts outside any repo's import path, same reasoning as
# every other vendored/duplicated helper in this codebase (CLAUDE.md,
# Session Startup step 7a).
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

# Non-reentrancy: one healer per box, ever. A cycle that outlives its timer
# interval must not be joined by a second one.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(ts)] healer: another cycle holds $LOCK; skipping" >> "$LOG"
  exit 0
fi

cd "$REPO" || { echo "[$(ts)] healer: no repo at $REPO" >> "$LOG"; exit 1; }

# --- Dual-role arbitration (ree-cloud-4) ------------------------------------
# See the header comment for why this lives here rather than in the
# Dispatcher. Runs UNCONDITIONALLY -- before GATE 1/2 below and regardless of
# their verdicts -- mirroring the Dispatcher's own original placement: a
# coordination-plane pause withholds new WORK, not the physical fact of which
# process owns the box's CPU, and this block spends no token budget either
# way (no `claude` invocation), so there is no cost to skip by gating it.
#
# Neither direction ever kills work: a hand-over to the runner waits for
# in-flight chip workers to DRAIN, and the runner is only ever stopped when
# it holds no claim, so a running experiment always finishes.
VERDICT_SCRIPT="$REPO/scripts/metaworker_role_verdict.py"
if [ "${REE_DUAL_ROLE:-0}" = "1" ] && [ -f "$VERDICT_SCRIPT" ]; then
  mkdir -p "$STATE_DIR"
  # Machine affinity for the verdict's --machine arg (e.g. "ree-cloud-4"),
  # resolved through machine_identity.canonical_machine_name -- NOT a raw
  # hostname compare (CLAUDE.md machine-identity note: a raw compare is what
  # silently disarmed a different fleet feature for three weeks). Falls back
  # to the short hostname if the resolver is unavailable, and strips a
  # "-metaworker" suffix defensively in case REE_LEASE_AFFINITY or REE_MACHINE
  # is ever set to the heartbeat-naming form by mistake (see
  # ree-metaworker-dispatch.sh's own LEASE_AFFINITY for the same guard).
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

  # READ THE QUEUE FROM origin/main, NOT the working tree -- same reasoning
  # as ree-metaworker-dispatch.sh's own comment: stopping ree-runner removes
  # the only thing pulling ree-v3 on this box, so the working-tree queue file
  # freezes the moment the box goes into metaworker mode, and the verdict
  # would never see a surge again. Fetch + `git show` is read-only and immune
  # to whatever divergence this checkout already has.
  _qfile="$STATE_DIR/queue_snapshot.json"
  if git -C "$REPO/ree-v3" fetch -q origin main 2>>"$LOG" \
     && git -C "$REPO/ree-v3" show origin/main:experiment_queue.json > "$_qfile.tmp" 2>>"$LOG" \
     && [ -s "$_qfile.tmp" ]; then
    mv -f "$_qfile.tmp" "$_qfile"
  else
    rm -f "$_qfile.tmp"
    echo "[$(ts)] role: WARN could not read queue from origin/main; falling back to the working-tree copy (may be stale)" >> "$LOG"
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
  fi
fi

# ---- GATE 1: is there anything to repair? (cheap, no claude) --------------
FINDINGS=$("$PY" - <<'PYGATE' 2>/dev/null || echo -1
import json, sys
try:
    doc = json.load(open("/home/ree/REE_Working/TASK_CHIPS.json"))
except Exception:
    print(-1); sys.exit(0)
chips = doc.get("chips") if isinstance(doc, dict) else doc
n = sum(1 for c in (chips or [])
        if isinstance(c, dict)
        and c.get("status") == "open"
        and c.get("origin") == "hygiene_tick")
print(n)
PYGATE
)

if [ "$FINDINGS" = "-1" ]; then
  # Fail OPEN: an unreadable ledger is itself something a healer should look
  # at, and this gate must never become the reason repairs stop happening.
  echo "[$(ts)] healer: ledger unreadable -- proceeding (fail-open gate)" >> "$LOG"
elif [ "$FINDINGS" -eq 0 ]; then
  echo "[$(ts)] healer: no open hygiene_tick findings; skipping (no context loaded)" >> "$LOG"
  exit 0
else
  echo "[$(ts)] healer: $FINDINGS open hygiene_tick finding(s); running a cycle" >> "$LOG"
fi

# ---- GATE 2: coordination-plane pause -------------------------------------
if ! "$PY" -c "
import sys; sys.path.insert(0, '$REPO/scripts')
import coordination_plane as cp
paused, holder = cp.is_coordination_plane_paused()
sys.exit(1 if paused else 0)" 2>/dev/null; then
  echo "[$(ts)] healer: coordination plane paused; skipping" >> "$LOG"
  exit 0
fi

claude_bin=$(command -v claude)
if [ -z "$claude_bin" ]; then
  echo "[$(ts)] healer: no claude binary on PATH" >> "$LOG"
  exit 127
fi

echo "[$(ts)] healer: cycle start on $MACHINE" >> "$LOG"
timeout --signal=TERM --kill-after=60 "$MAX_SEC" \
  "$claude_bin" -p "Run exactly ONE cycle of the metaworker-repair skill (see $REPO/.claude/skills/metaworker-repair/SKILL.md) in its resident HEALER form, then exit. This is the healer on machine $MACHINE. You must NEVER dispatch a kind:work chip -- repair only; anything outside the skill's Step 4 fix-now bounds becomes a kind:decision chip for the Orchestrator. Never touch a worker classified PARKED or PARKED-STALE by dispatcher_control.classify_worker. Do not call ScheduleWakeup -- an external systemd timer re-invokes this script." \
  --permission-mode auto >> "$LOG" 2>&1
rc=$?
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "[$(ts)] healer: TIMED OUT after ${MAX_SEC}s (SIGTERM+SIGKILL)" >> "$LOG"
else
  echo "[$(ts)] healer: cycle end rc=$rc" >> "$LOG"
fi
exit 0
