#!/bin/bash
# REE metaworker-dispatch wrapper (systemd timer, every 5 minutes).
#
# Reference copy, synced by hand from the live file on ree-worker-5
# (/usr/local/bin/ree_metaworker_dispatch.sh) -- install as that path (note the
# underscore, not hyphen) per ree-metaworker.service's header. Same
# manual-sync convention as ree-runner.service in this repo.
#
# Runs ONE cycle of .claude/skills/metaworker-dispatch/SKILL.md via a headless
# `claude -p` invocation, then writes an orchestrator-role heartbeat so this
# box shows up on REE_assembly's /machines dashboard.
#
# Deliberately a stateless relaunch, NOT a persistent `/loop` session: whether
# Claude Code's ScheduleWakeup dynamic self-pacing keeps a headless `claude -p`
# process alive/resumable across a sleep interval on a bare unattended VM is
# an open, unverified question (see the Phase H runbook). This pattern mirrors
# IGW's own real, working hourly tick (~/.local/bin/ree_igw_routine.sh on the
# Mac, launchd StartInterval) instead of assuming the untested one.
#
# Manual:  /usr/local/bin/ree_metaworker_dispatch.sh
# Log:     ~/ree_metaworker_dispatch.log
# State:   ~/.ree_metaworker/cycle_count

set -u

export PATH="/opt/local/bin:/usr/local/bin:/home/ree/.local/bin:/usr/bin:/bin"

MACHINE="ree-cloud-5"
REPO="/Users/dgolden/REE_Working"
STATE_DIR="$HOME/.ree_metaworker"
LOG="$HOME/ree_metaworker_dispatch.log"
mkdir -p "$STATE_DIR"

ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

# --- Per-box non-reentrancy lock (chip-20260809-metaworker-cycles-overlap) ---
# The systemd unit is Type=oneshot + KillMode=process + TimeoutStartSec. A cycle
# running longer than the start timeout has ITS BASH WRAPPER SIGTERMed while the
# `claude -p` child SURVIVES -- KillMode=process signals only the unit's main pid,
# not descendants (that is deliberate: it stops systemd killing live dispatched
# workers, see the .service header). The timer then starts a fresh instance, so the
# surviving orphan of cycle N overlaps cycle N+1. That breaks the skill's per-box
# "cap 2 in-flight" invariant: each overlapping dispatcher computes Step 4a's
# available_slots independently against the same 2-core box, so N dispatchers can
# each launch up to 2 workers. Confirmed live 2026-08-09T11:32Z: THREE dispatcher
# sessions alive at once (cycles 1830/1831/1832) on this box.
#
# Fix: hold an flock across the claude launch so a second cycle refuses to dispatch.
# CRITICAL -- the lock is held by the `flock` PROCESS wrapping the launch below
# (flock -> timeout -> claude), NOT by this script. A script-held lock (e.g.
# `exec 200>lock; flock 200`) would RELEASE the instant systemd SIGTERMs the
# wrapper at TimeoutStartSec, while the claude child lives on -- re-opening the
# exact overlap window. flock as a child of the wrapper survives that SIGTERM and
# holds the lock for claude's whole lifetime; `timeout` bounds a hung/orphaned
# claude so a wedged cycle can never hold the lock forever (TimeoutStartSec's
# SIGTERM never reaches claude under KillMode=process). Both properties verified
# empirically on THIS box 2026-08-09 (PROBE2 lock-held-across-parent-SIGTERM,
# PROBE3 timeout-self-release). See chip resolution note for the full test.
LOCKFILE="$STATE_DIR/dispatch.lock"
DISPATCH_MAX_SEC="${REE_DISPATCH_MAX_SEC:-1500}"

# Auto-sync: pull the umbrella repo before each cycle, so a landed script or
# SKILL.md fix reaches this box within one 5-minute tick instead of needing a
# human to SSH in and pull by hand. Confirmed gap 2026-08-03: this box ran 47
# cycles (~4h) on a since-fixed heartbeat-commit bug before anyone noticed,
# because there was no mechanism to pick the fix up short of a manual pull --
# and every cycle in between silently ran stale code with no signal that it
# was stale. --ff-only refuses rather than silently diverging if this box has
# unpushed local commits of its own (should not normally happen -- every
# ree_commit.py call below already passes --push); a refusal is logged and
# the cycle proceeds anyway on whatever code is already on disk, since a
# dispatcher running stale code is still better than one blocked entirely by
# a sync hiccup.
if git -C "$REPO" pull --ff-only origin master >> "$LOG" 2>&1; then
  echo "[$(ts)] autosync: ok ($(git -C "$REPO" rev-parse --short HEAD))" >> "$LOG"
else
  echo "[$(ts)] autosync: FAILED (git pull --ff-only) -- continuing this cycle on existing code" >> "$LOG"
fi

# Coordination-plane dirt check -- report only, never touches anything.
# Confirmed live 2026-08-03: THIS box's own REE_assembly checkout sat with
# 1220 uncommitted files (a stale build_experiment_indexes.py regen, never
# committed) for an unknown multi-hour window with zero signal anywhere --
# the exact box this dispatcher runs on. See
# scripts/audit_coordination_plane_dirt.py's own docstring for the full
# incident and the documented ~1050-1190-file governance-regen shape this
# detects (REE_assembly/evidence/planning/
# ree_assembly_orphaned_autostash_triage.md). A flag here does NOT block or
# fix anything -- same "detect and report, never auto-discard blindly"
# stance as everywhere else in this codebase; a human/session verifies
# against origin before touching anything.
if [ -f "$REPO/scripts/audit_coordination_plane_dirt.py" ]; then
  DIRT_OUT="$(/opt/local/bin/python3 "$REPO/scripts/audit_coordination_plane_dirt.py" --exit-nonzero 2>&1)"
  DIRT_RC=$?
  echo "[$(ts)] coordination-plane dirt check:" >> "$LOG"
  echo "$DIRT_OUT" >> "$LOG"
  if [ "$DIRT_RC" -ne 0 ]; then
    echo "[$(ts)] WARNING: coordination-plane dirt flagged -- see log above" >> "$LOG"
  fi
fi

CYCLE_FILE="$STATE_DIR/cycle_count"
[ -f "$CYCLE_FILE" ] || echo 0 > "$CYCLE_FILE"
CYCLE=$(( $(cat "$CYCLE_FILE") + 1 ))
echo "$CYCLE" > "$CYCLE_FILE"

DISPATCHED_FILE="$STATE_DIR/chips_dispatched_total"
[ -f "$DISPATCHED_FILE" ] || echo 0 > "$DISPATCHED_FILE"
PREV_DISPATCHED_TOTAL=$(cat "$DISPATCHED_FILE")

# Ground-truth dispatch signal: a NEW metaworker-* worktree appearing during
# this cycle means Step 4 actually dispatched (mirrors the skill's own Step 4a
# in-flight tracking exactly, rather than trusting the agent's self-report).
# NOTE: this counts worktree DIRECTORIES, which persist after their dispatched
# session finishes (cleanup is a separate, deliberately-manual GC step) -- it
# is the right signal for "how many dispatches have ever happened" (below) but
# the WRONG one for "how many are running right now" (see count_alive_dispatches).
WORKTREE_DIR="$REPO/.claude/worktrees"
count_metaworker_worktrees() {
  find "$WORKTREE_DIR" -maxdepth 1 -type d -name 'metaworker-*' 2>/dev/null | wc -l | tr -d ' '
}
WT_BEFORE=$(count_metaworker_worktrees)

# Actually-running signal for the heartbeat's in_flight_dispatches field --
# reads each worktree's .dispatch_pid and checks liveness, exactly mirroring
# the skill's own Step 4a in-flight cap check. Fixed 2026-08-03: the heartbeat
# was previously using count_metaworker_worktrees (directory existence) for
# this field, which overcounts once a dispatched session finishes but its
# worktree hasn't been GC'd yet.
count_alive_dispatches() {
  local n=0
  for wt in "$WORKTREE_DIR"/metaworker-*/; do
    [ -d "$wt" ] || continue
    local pid
    pid=$(cat "$wt/.dispatch_pid" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      n=$((n + 1))
    fi
  done
  echo "$n"
}

# Concurrency guard -- machine-level backstop on how many claude sessions run here.
# This box is 2 vCPU / 3.7 GB with NO swap, and a mature session is ~500 MB RSS.
# Dispatcher + 2 in-flight workers + 1 interactive session is ~2 GB. Past that the
# OOM killer fires with no swap to absorb it and picks its own victim -- which can
# be a worker mid-git-write, wedging a .git/index. Refuse to start another session
# rather than let the kernel choose which one dies.
#
# Worker COUNT is capped separately by the skill's Step 4 in-flight cap (2). This
# guard is independent of that: it holds even if the skill's cap is not honoured,
# and it also counts interactive sessions the skill knows nothing about.
MAX_CLAUDE_SESSIONS="${REE_MAX_CLAUDE_SESSIONS:-4}"
MIN_AVAIL_MB="${REE_MIN_AVAIL_MB:-700}"
LIVE_CLAUDE=$(pgrep -x claude 2>/dev/null | wc -l | tr -d ' ')
AVAIL_MB=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo 9999)

echo "[$(ts)] cycle $CYCLE start" >> "$LOG"

# Coordination-plane pause check -- mirrors the skill's own Step 1, done here
# too so a paused cycle short-circuits before spending a claude invocation.
PAUSED="false"
if /opt/local/bin/python3 -c "
import sys; sys.path.insert(0, '$REPO/scripts')
from coordination_plane import is_coordination_plane_paused
paused, who = is_coordination_plane_paused()
sys.exit(0 if paused else 1)
" 2>>"$LOG"; then
  PAUSED="true"
fi

if [ "$PAUSED" = "true" ]; then
  echo "[$(ts)] cycle $CYCLE: coordination plane paused, skipping dispatch" >> "$LOG"
  STATE="paused"
elif [ "$LIVE_CLAUDE" -ge "$MAX_CLAUDE_SESSIONS" ] || [ "$AVAIL_MB" -lt "$MIN_AVAIL_MB" ]; then
  echo "[$(ts)] cycle $CYCLE: THROTTLED -- claude sessions=$LIVE_CLAUDE/$MAX_CLAUDE_SESSIONS, MemAvailable=${AVAIL_MB}MB (floor ${MIN_AVAIL_MB}MB); skipping dispatch" >> "$LOG"
  STATE="throttled"
else
  STATE="dispatching"
  cd "$REPO"
  # --permission-mode auto, not a full permission bypass -- changed 2026-08-03,
  # REE_Working commit d39de60 (see metaworker-dispatch/SKILL.md Step 4's own
  # note). This box is a bare systemd-timer target with no live, classifier-
  # gated Claude Code session wrapping this launch, so the hard-deny that
  # motivated the original fix (a sandboxed session's own Bash/Edit tool call
  # getting blocked purely for containing the flag's text) does not apply to
  # THIS invocation -- it's applied anyway because it is strictly safer on its
  # own merits: this outer cycle-runner session stays classifier-checked on
  # its own actions (including the nested `nohup ... &` launches it performs
  # for Step 4 sub-dispatches) rather than unconditionally trusted, which
  # matters most precisely because this box runs fully unattended. Verified
  # empirically on this box (ree-cloud-5, 2026-08-03) before this change: a
  # `claude -p ... --permission-mode auto` session with NO
  # `.claude/settings.json` present at all (confirmed absent on this box --
  # it is gitignored and not distributed cross-machine) still completed a
  # nested `nohup /bin/bash -lc "..." &` launch cleanly, matching exactly
  # what Step 4 sub-dispatch needs. See REE_Working WORKSPACE_STATE.md
  # 2026-08-03 for the full cross-repo audit this was part of.
  # Hold the per-box lock across the ENTIRE claude lifetime via flock -> timeout ->
  # claude (see the LOCKFILE note near the top of this file for why the lock must be
  # process-held, not script-held). `-E 99` makes an flock lock-conflict return a
  # code distinct from any claude/timeout exit, so a lost race is unambiguous.
  # `timeout` bounds a hung or orphaned claude (SIGTERM at DISPATCH_MAX_SEC, then
  # SIGKILL 60s later) so a wedged cycle releases the lock instead of pinning the
  # box forever. Under normal operation the lock is uncontended and this is a
  # transparent wrapper around the same claude invocation as before.
  flock -n -E 99 "$LOCKFILE" \
    timeout --signal=TERM --kill-after=60 "$DISPATCH_MAX_SEC" \
    claude -p "Run exactly ONE cycle of the metaworker-dispatch skill (see $REPO/.claude/skills/metaworker-dispatch/SKILL.md), then exit. This is cycle $CYCLE on machine $MACHINE. Do not call ScheduleWakeup or otherwise self-pace via /loop -- an external systemd timer re-invokes this script every 5 minutes, so pacing is handled outside this session." \
      --permission-mode auto >> "$LOG" 2>&1
  DISPATCH_RC=$?
  if [ "$DISPATCH_RC" -eq 99 ]; then
    echo "[$(ts)] cycle $CYCLE: LOCKED OUT -- a sibling dispatch cycle already holds $LOCKFILE; not dispatching (reentrancy guard working as intended)" >> "$LOG"
    STATE="locked-out"
  elif [ "$DISPATCH_RC" -eq 124 ]; then
    echo "[$(ts)] cycle $CYCLE: claude -p TIMED OUT after ${DISPATCH_MAX_SEC}s (SIGTERM+SIGKILL); lock released" >> "$LOG"
    STATE="timed-out"
  else
    echo "[$(ts)] cycle $CYCLE: claude -p exited $DISPATCH_RC" >> "$LOG"
  fi
fi

# Best-effort telemetry: recomputed deterministically from the ledger/worktree
# state rather than trusted from the agent's own self-report, so a heartbeat
# can't drift from reality even if the cycle's summary text is wrong or
# truncated.
OPEN_WORK=$(/opt/local/bin/python3 "$REPO/scripts/chip_ledger.py" list --status open --kind work 2>/dev/null | grep -c '^task_\|^chip_' || true)
OPEN_DECISION=$(/opt/local/bin/python3 "$REPO/scripts/chip_ledger.py" list --status open --kind decision 2>/dev/null | grep -c '^task_\|^chip_' || true)

WT_AFTER=$(count_metaworker_worktrees)
NEW_DISPATCHES=$(( WT_AFTER > WT_BEFORE ? WT_AFTER - WT_BEFORE : 0 ))
DISPATCHED_TOTAL=$(( PREV_DISPATCHED_TOTAL + NEW_DISPATCHES ))
echo "$DISPATCHED_TOTAL" > "$DISPATCHED_FILE"

ALIVE_COUNT=$(count_alive_dispatches)

/opt/local/bin/python3 "$REPO/scripts/ree_metaworker_heartbeat.py" \
  --machine "$MACHINE" \
  --state "$STATE" \
  --cycles-completed "$CYCLE" \
  --chips-dispatched-total "$DISPATCHED_TOTAL" \
  --chips-open-work "${OPEN_WORK:-0}" \
  --chips-open-decision "${OPEN_DECISION:-0}" \
  --in-flight-dispatches "$ALIVE_COUNT" \
  $( [ "$PAUSED" = "true" ] && echo "--paused" ) \
  --note "cycle $CYCLE ($STATE)" \
  --push >> "$LOG" 2>&1

echo "[$(ts)] cycle $CYCLE end" >> "$LOG"
