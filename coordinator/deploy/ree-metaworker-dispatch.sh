#!/bin/bash
# REE metaworker-dispatch wrapper (systemd timer, every 5 minutes).
#
# Reference copy -- THIS TRACKED FILE IS CANONICAL, same convention as
# ree-metaworker-healer.sh (see check_metaworker_wrapper_deploy.py). The
# INSTALLED copy lives at /usr/local/bin/ree_metaworker_dispatch.sh (note the
# underscore, not hyphen, per ree-metaworker.service's header) and is a
# manually-synced DEPLOY TARGET, not the source of truth -- deploy by copying
# this file's content to that path. Do not treat the installed copy as
# authoritative or hand-edit it expecting the edit to persist; a hand patch
# made there needs to be folded back into this tracked file, the same stance
# check_metaworker_wrapper_deploy.py's docstring takes.
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

# Open work chips that are UNCLAIMED -- the heartbeat's `eligible_work`, and the
# only thing that makes health "idle" distinguishable from health "stalled" (see
# ree_metaworker_heartbeat.derive_health). A box with nothing it is allowed to
# take is FINE; a box with a backlog and no dispatch is not, and `chips_open_work`
# alone cannot tell those apart because it counts chips another dispatcher is
# already running.
#
# Deliberately CONSERVATIVE w.r.t. CLAIM state, i.e. it can only ever
# UNDERSTATE a stall on that axis. It does not re-run the skill's Step 3
# STOP-CHECKs (this wrapper has no business reimplementing them, and a wrong
# guess in the other direction would fire the stall signal on ordinary states
# -- which is how a guard gets switched off). So a chip claimed by a dead
# worker still reads as ineligible until its claim is reaped: a missed
# stall, never a false one on THAT axis.
#
# KNOWN GAP, on a DIFFERENT axis (2026-08-22, chip-20260822-metaworkerrepair-
# cloud5-stalled-allwithheld-cycle3528): this count does NOT exclude a chip
# dispatch_candidate_order.py's already_dispatched() marks [ALREADY-
# DISPATCHED] (a .claude/worktrees/metaworker-<chip_ref>/ already exists on
# this box). Step 4c skips such a candidate entirely -- no box-constraint
# condition is evaluated on it, so it can never produce a WITHHELD line --
# so counting it here CAN overstate eligible_work and starve rule 6b's
# WITHHELD-coverage check permanently while any such chip sits open. Do NOT
# "fix" this function to match: ree_metaworker_heartbeat.recompute_
# eligible_work() already corrects for it downstream, in the tracked python
# file every box re-pulls on its own next `git pull` -- fixing it HERE too
# would require a manual /usr/local/bin redeploy on every box for no benefit
# (see recompute_eligible_work()'s own docstring for why the correction was
# deliberately placed there instead of here).
count_eligible_work_chips() {
  /opt/local/bin/python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print(0); raise SystemExit(0)
items = d.get("chips") or d.get("items") or []
print(sum(1 for c in items
          if c.get("status") == "open" and c.get("kind") == "work"
          and not c.get("claimed_by")))
' "$REPO/TASK_CHIPS.json" 2>/dev/null || echo 0
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
LEASE_MIN="${REE_LEASE_MIN:-40}"
# 30 -> 40 (2026-08-24), paired with cloud-scaler.py DEFAULTS["PYTEST_LEASE_MAX_MIN"]
# 30 -> 40 and ree-metaworker.timer's OnUnitActiveSec 5min -> 30min. At the old
# 5-min cadence a 30-min lease was renewed 6x within its own lifetime -- huge
# slack. At 30-min cadence, requesting exactly 30 would make the lease expire
# right as the next cycle renews it, with near-zero margin against scheduling
# jitter or a slow renewal push. The scaler CLAMPS to whatever
# PYTEST_LEASE_MAX_MIN is (currently 40, see that constant's own comment for
# why); asking for more than the clamp is ignored, so this and that must move
# together -- raising one alone does nothing.

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
    --health-carry \
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

# Account-wide usage-limit cooldown check -- dispatch_usage_cooldown.py's
# Step-4-pre gate, run HERE too (one more early-exit branch of the same kind
# as PAUSED/THROTTLED below) so a withheld cycle short-circuits before
# spending a claude invocation. This is the WRAPPER-side half of the
# 2026-08-19 fix: the gate's other call sites (Step 4-pre in SKILL.md, and
# check_deferral_exit.py on a dispatched CHILD's death) can only ever run if
# the dispatcher's OWN session survives to reach them, and an account-wide
# usage-limit death is exactly what kills that session first -- measured on
# ree-cloud-5, 1578 of 2706 completed cycles carried the banner in the
# dispatcher's own stdout, against 2 real writes to the cooldown state file
# in its whole git history (REE_assembly evidence/planning/
# metaworker_dispatch_cooldown_wrapper_mis_siting_staged_20260819.md).
COOLDOWN_WITHHELD="false"
if [ -f "$REPO/scripts/dispatch_usage_cooldown.py" ]; then
  COOLDOWN_OUT="$(/opt/local/bin/python3 "$REPO/scripts/dispatch_usage_cooldown.py" check 2>&1)"
  COOLDOWN_RC=$?
  echo "[$(ts)] cooldown check: $COOLDOWN_OUT" >> "$LOG"
  [ "$COOLDOWN_RC" -eq 3 ] && COOLDOWN_WITHHELD="true"
fi

# --- No-open-work pre-exit (dispatch_preexit_check.py) ---------------------
# chip-20260823-preflight-batch-triage-of-candidate-chips item (a): a
# genuinely idle cycle (zero open chips of either kind) never needs to load
# the metaworker-dispatch SKILL.md at all -- skip the claude launch entirely
# rather than pay for a session that will find nothing in Step 3/4/5.
#
# Deliberately NOT folded into the PAUSED check above, even though
# dispatch_preexit_check.py's own preexit_verdict() covers both conditions.
# Its "plane paused" branch returns HAS-WORK on purpose -- that is the
# INTERACTIVE /loop convention (Step 1 needs a live session to log the skip
# and the recheck-delay correctly; see the script's own module docstring).
# This wrapper's PAUSED check above already short-circuits BEFORE reaching
# this point, more cheaply (no claude launch at all, not even this check), so
# by the time this runs the plane is known clear -- only the exit-3 IDLE
# verdict (zero open chips) is ever meaningful here. Any other outcome --
# exit 0 HAS-WORK, or an unexpected exit code on a crash/missing-file --
# falls through to the ordinary throttle/dispatch checks below: fail toward
# launching, same direction as every other gate in this file.
IDLE_NO_WORK="false"
if [ -f "$REPO/scripts/dispatch_preexit_check.py" ]; then
  PREEXIT_OUT="$(/opt/local/bin/python3 "$REPO/scripts/dispatch_preexit_check.py" check 2>&1)"
  PREEXIT_RC=$?
  echo "[$(ts)] preexit check: $PREEXIT_OUT" >> "$LOG"
  [ "$PREEXIT_RC" -eq 3 ] && IDLE_NO_WORK="true"
fi

# --- Runner-ownership guard (ree-cloud-4) -----------------------------------
# The FULL dual-role arbitration (metaworker_role_verdict.py's decision, plus
# the actual `sudo systemctl start/stop ree-runner` calls) moved to the
# resident, unleased Healer on 2026-08-26
# (chip-20260825-cloud4-arbitration-into-healer) -- see
# ree-metaworker-healer.sh's own "Dual-role arbitration" block, which is now
# this box's ONLY decision-maker for that transition. It had to move: since
# the 2026-08-25 Dispatcher/Healer split this wrapper runs only while an
# Orchestrator session holds a run lease, so a verdict computed only here
# would go inert (and unable to reverse itself) the moment nobody happens to
# lease this box (chip-20260825-cloud4-role-arbitration-lease-gap) -- whereas
# the Healer's hourly resident cadence re-evaluates it regardless.
#
# What stays here is a narrower, READ-ONLY safety net: a lease can be granted
# to this box for a reason unrelated to role state, and if that happens while
# the runner already owns it, this cycle must not dispatch new chip work onto
# a box actively running an experiment -- the exact CPU contention this
# arbitration exists to prevent. This check never calls the verdict script
# and never touches systemctl beyond the read-only `is-active` query below,
# so it is not a second decision-maker racing the Healer's `sudo systemctl
# start/stop` -- only the Healer ever issues those.
ROLE_STATE="dispatching"
if [ "${REE_DUAL_ROLE:-0}" = "1" ] && systemctl is-active --quiet ree-runner 2>/dev/null; then
  ROLE_STATE="runner"
  echo "[$(ts)] role: ree-runner is active on this dual-role box; not dispatching this cycle (arbitration itself now lives in the Healer)" >> "$LOG"
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
# there would be an ssh round-trip every cycle that nothing ever reads.
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

# --- This cycle's own claude session, for the heartbeat's health verdict -----
# Only set in the branch that actually launches claude. Left EMPTY in every
# branch that does not, so ree_metaworker_heartbeat.py is never handed an
# "outcome" for a session that was never started -- an empty log slice plus a
# non-zero status is indistinguishable from a session that died on arrival.
SESSION_LOG_FROM=""
SESSION_RC=""

if [ "$PAUSED" = "true" ]; then
  echo "[$(ts)] cycle $CYCLE: coordination plane paused, skipping dispatch" >> "$LOG"
  STATE="paused"
elif [ "$COOLDOWN_WITHHELD" = "true" ]; then
  echo "[$(ts)] cycle $CYCLE: WITHHELD -- account-wide usage-limit cooldown active, skipping dispatch" >> "$LOG"
  STATE="withheld"
elif [ "$ROLE_STATE" != "dispatching" ]; then
  # The runner owns the box (or is about to). Skip the claude launch entirely
  # rather than starting a cycle that would compute available_slots = 0 anyway
  # -- that saves a whole model invocation per tick for the whole time the box
  # is running experiments, which on a deep queue is hours.
  echo "[$(ts)] cycle $CYCLE: role=$ROLE_STATE, not dispatching this cycle" >> "$LOG"
  STATE="$ROLE_STATE"
elif [ "$IDLE_NO_WORK" = "true" ]; then
  echo "[$(ts)] cycle $CYCLE: IDLE -- no open chips (work or decision), skipping dispatch" >> "$LOG"
  STATE="idle"
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
  # Byte offset of the log RIGHT BEFORE the launch. Everything appended past
  # this point until DISPATCH_RC is read is this cycle's own claude stdout, and
  # nothing else writes to $LOG in between -- which is what lets the heartbeat
  # classify a usage-limit death the session itself could never report, because
  # it never got a turn. Measured on this box: of 1578 cycles whose output
  # carried the limit banner, 1430 exited 0, so the TEXT is the signal and the
  # exit status is only corroboration.
  SESSION_LOG_FROM=$(wc -c < "$LOG" 2>/dev/null | tr -d ' ')
  [ -n "$SESSION_LOG_FROM" ] || SESSION_LOG_FROM=0
  flock -n -E 99 "$LOCKFILE" \
    timeout --signal=TERM --kill-after=60 "$DISPATCH_MAX_SEC" \
    claude -p "Run exactly ONE cycle of the metaworker-dispatch skill (see $REPO/.claude/skills/metaworker-dispatch/SKILL.md), then exit. This is cycle $CYCLE on machine $MACHINE. Your Step 4a in-flight worker cap on this box is $DISPATCH_MAX_INFLIGHT (available_slots = $DISPATCH_MAX_INFLIGHT - in-flight), not the default 2. AUTH: this box IS authenticated -- your own session is proof, since a claude -p cycle cannot run unauthenticated. Do NOT test for ~/.claude/.credentials.json; it is legitimately absent under CLAUDE_CODE_OAUTH_TOKEN auth, and treating its absence as an auth failure idled ree-cloud-4 for seven cycles on 2026-08-18. If a dispatched worker reports 'Not logged in', that is an environment-inheritance defect to report, NOT an account problem and NOT a reason to stop dispatching. Do not call ScheduleWakeup or otherwise self-pace via /loop -- an external systemd timer re-invokes this script every 30 minutes, so pacing is handled outside this session.$FRESHNESS_NOTE" \
      --permission-mode auto >> "$LOG" 2>&1
  DISPATCH_RC=$?
  SESSION_RC="$DISPATCH_RC"
  if [ "$DISPATCH_RC" -eq 99 ]; then
    # flock refused; claude never started. Withdraw the observation rather than
    # report an empty slice with a non-zero status, which reads as a death.
    # derive_health also guards this on state=locked-out -- belt and braces,
    # because the two live in different repos and drift independently.
    SESSION_LOG_FROM=""
    SESSION_RC=""
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
NEW_WT_NAMES_FILE="$STATE_DIR/.wt_new_names"
LC_ALL=C comm -13 "$WT_BEFORE_FILE" "$WT_AFTER_FILE" > "$NEW_WT_NAMES_FILE"
NEW_DISPATCHES=$(wc -l < "$NEW_WT_NAMES_FILE" | tr -d ' ')
DISPATCHED_TOTAL=$(( PREV_DISPATCHED_TOTAL + NEW_DISPATCHES ))
echo "$DISPATCHED_TOTAL" > "$DISPATCHED_FILE"

ALIVE_COUNT=$(count_alive_dispatches)

# Chip_ref of the most recently dispatched worktree this cycle, for the
# heartbeat's last_dispatch_chip_ref field. Worktree names are
# metaworker-<chip_ref> (Step 4's own convention); when more than one was
# created this cycle, take the lexicographically LAST -- chip_ref sorts
# chip-YYYYMMDD-..., so this is a same-cycle approximation of "most
# recent", not a hazard: ree_metaworker_heartbeat.py's build_heartbeat()
# carries the field forward across every OTHER call (the START-of-cycle
# keepalive, and any cycle that dispatches nothing), so a same-cycle
# ordering miss self-corrects on the very next real dispatch and is never
# silently permanent the way a plain args-only field would be.
LAST_DISPATCH_CHIP_REF=""
if [ -s "$NEW_WT_NAMES_FILE" ]; then
  LAST_DISPATCH_CHIP_REF=$(tail -1 "$NEW_WT_NAMES_FILE" | sed 's/^metaworker-//')
fi

/opt/local/bin/python3 "$REPO/scripts/ree_metaworker_heartbeat.py" \
  --machine "$MACHINE" \
  --state "$STATE" \
  --cycles-completed "$CYCLE" \
  --chips-dispatched-total "$DISPATCHED_TOTAL" \
  --chips-open-work "${OPEN_WORK:-0}" \
  --chips-open-decision "${OPEN_DECISION:-0}" \
  --in-flight-dispatches "$ALIVE_COUNT" \
  --dispatched-this-cycle "${NEW_DISPATCHES:-0}" \
  --eligible-work "$(count_eligible_work_chips)" \
  $( [ -n "$SESSION_LOG_FROM" ] && echo "--session-log $LOG --session-log-from-byte $SESSION_LOG_FROM" ) \
  $( [ -n "$SESSION_RC" ] && echo "--session-rc $SESSION_RC" ) \
  $( [ "$PAUSED" = "true" ] && echo "--paused" ) \
  $( [ -n "$LAST_DISPATCH_CHIP_REF" ] && printf -- '--last-dispatch-chip-ref %s' "$LAST_DISPATCH_CHIP_REF" ) \
  --note "cycle $CYCLE ($STATE)" \
  --push >> "$LOG" 2>&1

echo "[$(ts)] cycle $CYCLE end" >> "$LOG"
