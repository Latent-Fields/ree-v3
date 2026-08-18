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

REPO="/Users/dgolden/REE_Working"

# Machine identity. Was hardcoded "ree-cloud-5" until 2026-08-18; parameterised
# so this one wrapper serves any metaworker box (cloud-4 dual-role trial).
# Resolved through machine_identity.py, NOT a raw hostname compare -- that
# resolver is an ALLOWLIST, and for ree-cloud-1..5 the trailing digit IS the
# identity and must never be collapsed (CLAUDE.md machine-identity note).
# Falls back to the short hostname if the resolver is unavailable, because a
# heartbeat under a slightly-wrong name beats no heartbeat at all.
MACHINE="${REE_METAWORKER_MACHINE:-}"
if [ -z "$MACHINE" ]; then
  MACHINE=$(/opt/local/bin/python3 -c '
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
[ -n "$MACHINE" ] || MACHINE=$(hostname -s 2>/dev/null || echo unknown-metaworker)
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

# --- Executable-surface freshness gate --------------------------------------
# The autosync above CANNOT be assumed to have worked, and on this box it
# mostly has not: measured to 2026-08-16, 1299 FAILED against 853 ok. Once the
# checkout carries any local commit origin does not have -- the normal residue
# of every ree_commit.py cherry-pick push retry -- `--ff-only` refuses forever,
# and the only trace is the log line above, in a file nobody reads.
#
# What that costs is not hypothetical. 2026-08-14: the orphaned-close push
# guard was on origin all day; this box's scripts/task_claim.py was 399 lines
# behind and had none of it; at 19:53Z the exact corruption it was written to
# prevent happened anyway, marking a LIVE sibling session's ACTIVE claim done
# (origin 7241ae05). A guard that ships to origin but not to the box that
# needs it is not deployed. Full write-up: REE_assembly
# evidence/planning/cloud5_stale_scripts_wedge_staged_20260814.md.
#
# WHY THE VERDICT GOES INTO THE PROMPT rather than a log line or a SKILL.md
# step: this wrapper lives at /usr/local/bin, OUTSIDE the repo. It is the only
# delivery channel that still reaches the dispatcher when the checkout cannot
# adopt origin -- a SKILL.md step, a hook, or a python guard is inside the very
# tree that is frozen, so landing one on origin does not deploy it. Same shape
# as `.claude/settings.json` carrying the worktree-skills audit (CLAUDE.md
# Session Startup Protocol step 7b) for exactly the same reason.
#
# It does NOT abort the cycle, deliberately. The repair for a wedged checkout
# is itself carried out by a headless chip session dispatched by this loop, and
# so are the ref_convergence_wedge escalation chips that name it -- halting on
# stale scripts would deadlock its own fix.
FRESHNESS_NOTE=""
if [ -f "$REPO/scripts/check_dispatch_scripts_freshness.py" ]; then
  FRESH_OUT="$(/opt/local/bin/python3 "$REPO/scripts/check_dispatch_scripts_freshness.py" 2>&1)"
  FRESH_RC=$?
  echo "[$(ts)] scripts-freshness check:" >> "$LOG"
  echo "$FRESH_OUT" >> "$LOG"
  if [ "$FRESH_RC" -ne 0 ]; then
    FRESHNESS_NOTE="

STALE EXECUTABLE SURFACE ON THIS BOX -- read this before trusting any guard. This checkout's scripts/ does not match origin/master, so the scripts every dispatched session invokes by absolute path are FROZEN at an older revision, and any guard landed on origin since the freeze is NOT in force here. Do NOT stop dispatching -- clearing this is itself work for a dispatched session -- but treat it as this cycle's priority: look for an open ref-convergence / wedge / stale-scripts chip in TASK_CHIPS.json and dispatch it ahead of other work (if none exists, that is the chip to raise), and do not assume any recently-landed fix is deployed on this box. Verbatim check output follows.
$FRESH_OUT"
  fi
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

# Chip-backlog counts. Read the ledger JSON directly rather than screen-scraping
# `chip_ledger.py list`. The previous `grep -c '^task_\|^chip_'` undercounted by
# ~4x -- measured 2026-08-18 on this box: reported 26, actual 97 -- because a
# headless-origin chip has no task_id and `list` prints it as the literal
# `None`, which neither anchor matches. The obvious repair (anchor on
# `\[open\] chip_ref=`) was ALSO wrong: `list` inserts a
# `CLAIMED(by=..., at=...)` token between those two when a chip is claimed, so
# it still missed 2 of 97. Both misses are DISPLAY-FORMAT drift in a field this
# wrapper never had any business parsing; the JSON is the stable surface.
count_open_chips() {
  /opt/local/bin/python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(0); raise SystemExit(0)
items = d.get("chips") or d.get("items") or []
print(sum(1 for c in items
          if c.get("status") == "open" and c.get("kind") == sys.argv[2]))
' "$REPO/TASK_CHIPS.json" "$1" 2>/dev/null || echo 0
}

# --- Scaler work lease (primary keepalive) --------------------------------
# The scaler must not power this box off while it has chip work. There are two
# transports for that signal and this is the PRIMARY one:
#
#   lease      hub-resident file, written over ssh. No lag, 30-min clamp.
#   heartbeat  git-transported orchestrator heartbeat. Fallback: it depends on
#              a COMMIT landing and the hub's checkout pulling it, so it needs a
#              50-minute window to cover the writer's 30-min liveness floor plus
#              observed git lag.
#
# Same shape as the fleet telemetry generally -- coordinator-primary with the
# git mirror as the unreachable-coordinator fallback (CLAUDE.md, Coordinator).
# The lease became available to this box on 2026-08-18 when it was given an ssh
# key to the hub; before that the git heartbeat was the only channel it had.
#
# AFFINITY, NOT the metaworker identity: the scaler keys leases by the worker's
# machine_affinity (ree-cloud-4), while the orchestrator heartbeat is written
# under ree-cloud-4-metaworker to avoid colliding with the runner's heartbeat.
LEASE_AFFINITY="${REE_LEASE_AFFINITY:-}"
if [ -z "$LEASE_AFFINITY" ]; then
  case "$MACHINE" in
    *-metaworker) LEASE_AFFINITY="${MACHINE%-metaworker}" ;;
    *)            LEASE_AFFINITY="$MACHINE" ;;
  esac
fi
HUB_SSH="${REE_HUB_SSH:-ree@10.8.0.1}"
LEASE_PURPOSE="metaworker_dispatch"
LEASE_MIN="${REE_LEASE_MIN:-30}"   # the scaler CLAMPS to 30; asking for more is ignored

take_lease() {
  _exp=$(/opt/local/bin/python3 -c "
import datetime
print((datetime.datetime.now(datetime.timezone.utc)
       + datetime.timedelta(minutes=$LEASE_MIN)).strftime('%Y-%m-%dT%H:%M:%SZ'))" 2>/dev/null) || return 1
  printf '{"expires_at": "%s", "owner": "%s/%s", "purpose": "%s"}\n' \
      "$_exp" "$MACHINE" "$$" "$LEASE_PURPOSE" \
    | ssh -o BatchMode=yes -o ConnectTimeout=10 "$HUB_SSH" \
        "mkdir -p /home/ree/pytest_leases && cat > /home/ree/pytest_leases/$LEASE_AFFINITY.lease" \
        2>>"$LOG" \
    && echo "[$(ts)] lease: held for $LEASE_AFFINITY until $_exp" >> "$LOG" \
    || echo "[$(ts)] lease: WARN could not reach hub; falling back to the git heartbeat veto" >> "$LOG"
}

# Release ONLY a lease this dispatcher owns. remote_pytest.sh uses the SAME
# per-affinity path for its own runs, so an unconditional rm would yank the
# lease out from under a live test suite on this box and let the scaler kill it
# mid-run -- the exact incident the lease was built for in the first place.
release_lease() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$HUB_SSH" \
    "grep -q '\"purpose\": \"$LEASE_PURPOSE\"' /home/ree/pytest_leases/$LEASE_AFFINITY.lease 2>/dev/null \
       && rm -f /home/ree/pytest_leases/$LEASE_AFFINITY.lease && echo released || echo 'not ours -- left alone'" \
    >>"$LOG" 2>&1 || true
}

# Emit a heartbeat. Called TWICE per cycle -- once before the claude launch and
# once after -- because the scaler's orchestrator veto (cloud-scaler.py
# read_orchestrator) keeps this box alive only while this heartbeat is FRESH,
# and a dispatch cycle can legitimately run to REE_DISPATCH_MAX_SEC (25 min).
#
# End-of-cycle only was the bug. Confirmed live 2026-08-18: ree-worker-4's
# heartbeat was written at 19:16 (end of cycle 2), cycle 3 started 19:35:30 and
# was still running at 19:40, by which point the veto had aged past its window
# and the scaler shut the box down MID-CYCLE, killing a live dispatched worker.
# Scaler log: orch=active age=9min -> 14min -> 19min -> orch=none -> shutdown.
# This is the same shape as remote_pytest.sh's mid-run lease renewal, and for
# the same reason: work that outlives its own keepalive loses the box.
emit_heartbeat() {
  _hb_state="$1"; _hb_note="$2"
  /opt/local/bin/python3 "$REPO/scripts/ree_metaworker_heartbeat.py" \
    --machine "$MACHINE" \
    --state "$_hb_state" \
    --cycles-completed "$CYCLE" \
    --chips-dispatched-total "$(cat "$DISPATCHED_FILE" 2>/dev/null || echo 0)" \
    --chips-open-work "$(count_open_chips work)" \
    --chips-open-decision "$(count_open_chips decision)" \
    --in-flight-dispatches "$(count_alive_dispatches)" \
    $( [ "${PAUSED:-false}" = "true" ] && echo "--paused" ) \
    --note "$_hb_note" \
    --push >> "$LOG" 2>&1
}

CYCLE_FILE="$STATE_DIR/cycle_count"
[ -f "$CYCLE_FILE" ] || echo 0 > "$CYCLE_FILE"
CYCLE=$(( $(cat "$CYCLE_FILE") + 1 ))
echo "$CYCLE" > "$CYCLE_FILE"

DISPATCHED_FILE="$STATE_DIR/chips_dispatched_total"
[ -f "$DISPATCHED_FILE" ] || echo 0 > "$DISPATCHED_FILE"
PREV_DISPATCHED_TOTAL=$(cat "$DISPATCHED_FILE")

WORKTREE_DIR="$REPO/.claude/worktrees"

# Ground-truth dispatch signal: a NEW metaworker-* worktree NAME appearing
# during this cycle means Step 4 actually dispatched (mirrors the skill's own
# Step 4a in-flight tracking, rather than trusting the agent's self-report).
# This is the right signal for "how many dispatches have ever happened"; for
# "how many are running right now" see count_alive_dispatches below.
#
# This lists NAMES and takes a set difference, rather than comparing two
# COUNTS. The count form (`WT_AFTER > WT_BEFORE ? WT_AFTER - WT_BEFORE : 0`,
# used until 2026-08-18) silently lost every dispatch that happened in a cycle
# where GC removed at least as many worktrees as were created -- the delta went
# zero-or-negative and clamped to 0. Since metaworker_worktree_gc runs against
# the same directory, that is the common case, not a corner: measured
# 2026-08-18 on ree-cloud-5, the lifetime counter read 292 while the box had
# created 65 worktrees in the preceding 24h alone. A name set difference is
# immune to concurrent removal because it never reasons about totals.
list_metaworker_worktrees() {
  find "$WORKTREE_DIR" -maxdepth 1 -type d -name 'metaworker-*' 2>/dev/null \
    | sed 's#.*/##' | LC_ALL=C sort
}
WT_BEFORE_FILE="$STATE_DIR/.wt_names_before"
WT_AFTER_FILE="$STATE_DIR/.wt_names_after"
list_metaworker_worktrees > "$WT_BEFORE_FILE"

# Actually-running signal for the heartbeat's in_flight_dispatches field --
# reads each worktree's .dispatch_pid and checks liveness, exactly mirroring
# the skill's own Step 4a in-flight cap check. Fixed 2026-08-03: the heartbeat
# was previously using a worktree-directory COUNT (existence) for
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
# Per-box in-flight WORKER cap, consumed by the dispatch skill's Step 4a
# (available_slots = REE_DISPATCH_MAX_INFLIGHT - in-flight). Exported so the
# `claude -p` child below inherits it. Default stays 2 -- an interactive
# dispatcher on the Mac shares CPU/RAM with the user's own foreground session
# and must NOT inherit a resident box's higher cap (SKILL.md's "one real,
# non-hypothetical tradeoff"). Raise it per-box via the systemd unit only.
DISPATCH_MAX_INFLIGHT="${REE_DISPATCH_MAX_INFLIGHT:-2}"
export REE_DISPATCH_MAX_INFLIGHT="$DISPATCH_MAX_INFLIGHT"

# Machine-level backstop on ALL claude processes here (dispatcher + workers +
# any interactive session). Scales with the worker cap: N workers + this
# dispatcher + 1 interactive-session headroom. Was a bare 4 (correct only for
# N=2) until 2026-08-18; a hardcoded 4 against a cap of 3 would have throttled
# the box at exactly the moment the raised cap was meant to take effect.
MAX_CLAUDE_SESSIONS="${REE_MAX_CLAUDE_SESSIONS:-$(( DISPATCH_MAX_INFLIGHT + 2 ))}"
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

# --- Dual-role arbitration (ree-cloud-4) ----------------------------------
# This box holds BOTH roles: ree-runner and metaworker-dispatch. Until
# 2026-08-18 they simply ran at the same time on 2 vCPU with no coordination in
# either direction. metaworker_role_verdict.py decides which role owns the box,
# recomputed from the queue file every tick and remembering nothing, so a
# wrapper that dies mid-transition cannot strand the box in a phantom state.
#
# Neither direction ever kills work: a hand-over to the runner waits for
# in-flight chips to DRAIN, and the runner is only ever stopped when it holds
# no claim, so a running experiment always finishes.
ROLE_STATE="dispatching"
ROLE_SLOTS=""
# In the UMBRELLA scripts/, not ree-v3, deliberately: this wrapper autosyncs
# the umbrella every cycle, whereas nothing pulls ree-v3 on this box once the
# runner is stopped -- which is precisely the state this script puts it in.
# Same delivery-channel reasoning as the freshness gate above.
VERDICT_SCRIPT="$REPO/scripts/metaworker_role_verdict.py"
if [ "${REE_DUAL_ROLE:-0}" = "1" ] && [ -f "$VERDICT_SCRIPT" ]; then
  _runner_active=0
  systemctl is-active --quiet ree-runner 2>/dev/null && _runner_active=1
  _v=$(/opt/local/bin/python3 "$VERDICT_SCRIPT" \
        --queue "$REPO/ree-v3/experiment_queue.json" \
        --machine "$LEASE_AFFINITY" \
        --in-flight "$(count_alive_dispatches)" \
        --runner-active "$_runner_active" 2>>"$LOG")
  if [ -n "$_v" ]; then
    ROLE_STATE=$(printf '%s' "$_v" | /opt/local/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["state"])' 2>/dev/null || echo dispatching)
    _reason=$(printf '%s' "$_v" | /opt/local/bin/python3 -c 'import json,sys; print(json.load(sys.stdin)["reason"])' 2>/dev/null || echo "")
    _start=$(printf '%s' "$_v" | /opt/local/bin/python3 -c 'import json,sys; print(int(json.load(sys.stdin)["start_runner"]))' 2>/dev/null || echo 0)
    _stop=$(printf '%s' "$_v" | /opt/local/bin/python3 -c 'import json,sys; print(int(json.load(sys.stdin)["stop_runner"]))' 2>/dev/null || echo 0)
    echo "[$(ts)] role: $ROLE_STATE -- $_reason" >> "$LOG"
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

# Refresh the scaler veto BEFORE the (potentially 25-minute) claude launch.
emit_heartbeat "$ROLE_STATE" "cycle $CYCLE ($ROLE_STATE)"

# Lease follows DEMAND, not mere liveness -- the same test the scaler applies to
# the heartbeat (read_orchestrator): live workers, or backlog this box would pick
# up. With neither, release, so an idle box with an empty ledger can actually be
# powered down instead of being pinned up by the fact that its timer is ticking.
# OPT-IN per box. A lease is only meaningful for a box the scaler actually
# manages -- i.e. one that appears in cloud-scaler.py's WORKERS list. ree-cloud-5
# is deliberately absent from that list and is never power-cycled, so a lease
# there would be an ssh round-trip every 5 minutes that nothing ever reads.
# Set REE_LEASE_ENABLED=1 in the systemd drop-in on scaler-managed boxes only.
if [ "${REE_LEASE_ENABLED:-0}" != "1" ]; then
  :   # not scaler-managed; the orchestrator heartbeat is the only signal needed
elif _inflight=$(count_alive_dispatches); _backlog=$(count_open_chips work); \
     [ "${_inflight:-0}" -gt 0 ] || [ "${_backlog:-0}" -gt 0 ]; then
  take_lease
else
  echo "[$(ts)] lease: no in-flight workers and empty work ledger -- releasing" >> "$LOG"
  release_lease
fi

if [ "$PAUSED" = "true" ]; then
  echo "[$(ts)] cycle $CYCLE: coordination plane paused, skipping dispatch" >> "$LOG"
  STATE="paused"
elif [ "$ROLE_STATE" != "dispatching" ]; then
  # The runner owns the box (or is about to). Skip the claude launch entirely
  # rather than starting a cycle that would compute available_slots = 0 anyway
  # -- that saves a whole model invocation per tick for the whole time the box
  # is running experiments, which on a deep queue is hours.
  echo "[$(ts)] cycle $CYCLE: role=$ROLE_STATE, not dispatching this cycle" >> "$LOG"
  STATE="$ROLE_STATE"
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
    claude -p "Run exactly ONE cycle of the metaworker-dispatch skill (see $REPO/.claude/skills/metaworker-dispatch/SKILL.md), then exit. This is cycle $CYCLE on machine $MACHINE. Your Step 4a in-flight worker cap on this box is $DISPATCH_MAX_INFLIGHT (available_slots = $DISPATCH_MAX_INFLIGHT - in-flight), not the default 2. AUTH: this box IS authenticated -- your own session is proof, since a claude -p cycle cannot run unauthenticated. Do NOT test for ~/.claude/.credentials.json; it is legitimately absent under CLAUDE_CODE_OAUTH_TOKEN auth, and treating its absence as an auth failure idled ree-cloud-4 for seven cycles on 2026-08-18. If a dispatched worker reports 'Not logged in', that is an environment-inheritance defect to report, NOT an account problem and NOT a reason to stop dispatching. Do not call ScheduleWakeup or otherwise self-pace via /loop -- an external systemd timer re-invokes this script every 5 minutes, so pacing is handled outside this session.$FRESHNESS_NOTE" \
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
OPEN_WORK=$(count_open_chips work)
OPEN_DECISION=$(count_open_chips decision)

list_metaworker_worktrees > "$WT_AFTER_FILE"
NEW_DISPATCHES=$(LC_ALL=C comm -13 "$WT_BEFORE_FILE" "$WT_AFTER_FILE" | wc -l | tr -d ' ')
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
