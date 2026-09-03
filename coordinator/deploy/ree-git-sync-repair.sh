#!/bin/bash
# REE periodic git-sync repair -- CLOUD WORKER variant (systemd timer).
#
# WHY: defense-in-depth backstop. The runner's heartbeat/commands push and
# its REE_assembly sync pull were made rebase-free (2026-05-18), so the repo
# should no longer wedge. This timer catches any residual wedge (manual git,
# unforeseen race, older runner build) within the timer interval instead of
# leaving a cloud worker dark for hours -- the failure mode that left
# cloud-2/3 stale ~10h on 2026-05-18.
#
# ---------------------------------------------------------------------------
# 2026-09-03 -- THE FALSE GREEN (same bug class as the Mac variant's)
# (original fix: chip-20260819-gitsyncrepair-reports-ok-while-not-syncing,
#  landed in the umbrella's scripts/ree_git_sync_repair.sh 2026-08-22)
#
# NOTE ON THE TWO VARIANTS: scripts/ree_git_sync_repair.sh (launchd, Mac) and
# this file (systemd, cloud worker) are DELIBERATELY different scripts, not
# copies -- ~416 diff lines, both correct as written; see
# check_metaworker_wrapper_deploy.py's block comment, which warns that pairing
# them by basename would report a false DRIFTED. So this is NOT a re-sync of a
# stale fork. It is the same DEFECT, which happened to live in both variants
# and was fixed in only one. The reasoning below is adapted from the Mac
# variant's; the exempted path set differs (telemetry here, derived registry
# artifacts there) because the recurring dirt differs per box role.
#
# The healthy (non-wedged) branch used to gate its fast-forward on an
# ENTIRELY clean `git status --porcelain --untracked-files=no`, and fell
# through to a single catch-all `log "$name OK (ahead=$ahead behind=$behind)"`
# for everything else. So a repo that was behind AND dirty logged **OK** and
# synced nothing: "OK" meant "I looked", not "you are in sync".
#
# That disarmed this job permanently on exactly the boxes it matters most on.
# A worker running the runner writes its own heartbeat into
# evidence/experiments/runner_heartbeats/<box>.json, so REE_assembly is
# PERMANENTLY dirty there and the pre-fix gate declined forever. Measured on
# ree-cloud-4, from this log, 2026-09-03:
#     2026-09-03T04:50:56Z REE_assembly OK (ahead=0 behind=58)
#     2026-09-03T05:54:37Z REE_assembly OK (ahead=0 behind=8)
# 44 such lines in the last 200; the box had drifted 61 commits / ~7.5h.
#
# Two fixes, mirroring the Mac copy:
#
# (1) DISTINCT VERDICTS. Behind-but-not-synced is now BEHIND_NOT_SYNCED with
#     the blocking paths named; unpushed local commits are AHEAD_UNPUSHED.
#     Only genuinely in-sync logs OK. The refusal to fast-forward over
#     uncommitted work was never the bug -- calling that refusal "OK" was.
#
# (2) TELEMETRY DIRT NO LONGER BLOCKS THE ATTEMPT. TELEMETRY_RE is now
#     consulted on the fast-forward path, so heartbeat/status dirt stops
#     being a reason not to TRY. This is NON-DESTRUCTIVE and that is the
#     whole design: it does not `checkout --` anything and does not widen the
#     clean-tree rule -- it just lets `git merge --ff-only` adjudicate. If the
#     incoming commits do not touch those paths the ff succeeds and the
#     working-tree copies are preserved untouched; if they DO touch them git
#     refuses on its own and we log BEHIND_NOT_SYNCED. Nothing is ever
#     discarded to make room.
#
#     SCOPED TO THE FAST-FORWARD PATH ONLY. The wedged safety gate below is
#     deliberately left exactly as it was -- it is the one that runs
#     `reset --hard`, and its own telemetry handling there is a separate,
#     intentional design. Do not merge the two gates into one predicate:
#     they are asymmetric on purpose.
# ---------------------------------------------------------------------------
# SAFETY ENVELOPE (identical to the Mac launchd repair):
#   A repo is auto-repaired iff ALL of:
#     (1) it is wedged: a .git/rebase-(merge|apply) dir exists, OR HEAD is
#         detached, OR it is simultaneously ahead AND behind origin;
#     (2) EVERY commit ahead of origin/<branch> touches ONLY paths under
#         evidence/experiments/runner_heartbeats|status|commands/
#         (pure regenerable telemetry -- nothing precious to lose);
#     (3) there are NO uncommitted *tracked* changes outside those dirs.
#   Anything else -> log NEEDS_HUMAN, make NO destructive change.
#   Healthy-but-behind + clean -> fast-forward only (non-destructive).
#   Healthy -> no-op.
#
# Pure git ops only: never touches the ree-runner service or the running
# experiment subprocess. Never edits TASK_CLAIMS.json.
#
# Manual run:  /usr/local/bin/ree_git_sync_repair.sh
# Log:         /home/ree/ree_git_sync_repair.log

set -u
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"
GIT=/usr/bin/git
LOG=/home/ree/ree_git_sync_repair.log
LOCKDIR=/tmp/ree_git_sync_repair.lock
TELEMETRY_RE='^evidence/experiments/runner_(heartbeats|status|commands)/'

# repo|branch
REPOS=(
  "/home/ree/REE_Working/REE_assembly|master"
  "/home/ree/REE_Working/ree-v3|main"
)

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "$(ts) $*" >> "$LOG"; }

# Single-instance lock (mkdir is atomic). Stale lock >30min is reclaimed.
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  if [ -d "$LOCKDIR" ] && [ "$(find "$LOCKDIR" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then
    rmdir "$LOCKDIR" 2>/dev/null; mkdir "$LOCKDIR" 2>/dev/null || { log "LOCK busy; exit"; exit 0; }
  else
    log "LOCK busy; exit"; exit 0
  fi
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT

log "=== run start ($(hostname)) ==="

for entry in "${REPOS[@]}"; do
  repo="${entry%%|*}"; branch="${entry##*|}"
  name="$(basename "$repo")"
  [ -d "$repo/.git" ] || { log "$name SKIP (no .git)"; continue; }
  (
    cd "$repo" || exit 0
    "$GIT" fetch --quiet origin "$branch" 2>/dev/null

    has_rebase=0
    { [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; } && has_rebase=1
    detached=0
    "$GIT" symbolic-ref -q HEAD >/dev/null 2>&1 || detached=1
    ahead=$("$GIT" rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo 0)
    behind=$("$GIT" rev-list --count "HEAD..origin/$branch" 2>/dev/null || echo 0)
    ahead=${ahead:-0}; behind=${behind:-0}

    wedged=0
    if [ "$has_rebase" = 1 ] || [ "$detached" = 1 ] || { [ "$ahead" -gt 0 ] && [ "$behind" -gt 0 ]; }; then
      wedged=1
    fi

    if [ "$wedged" = 0 ]; then
      # Tracked dirt, and the subset of it that actually BLOCKS a
      # fast-forward (i.e. excluding regenerable telemetry). See the
      # "THE FALSE GREEN" block comment at the top of this file.
      dirty="$("$GIT" status --porcelain --untracked-files=no | sed 's/^...//')"
      blocking="$(printf '%s\n' "$dirty" | grep -v '^$' | grep -Ev "$TELEMETRY_RE")"
      blocking_list="$(printf '%s\n' "$blocking" | grep -v '^$' | head -5 | tr '\n' ' ')"
      derived_list="$(printf '%s\n' "$dirty" | grep -v '^$' | grep -E "$TELEMETRY_RE" | head -5 | tr '\n' ' ')"

      if [ "$behind" = 0 ]; then
        if [ "$ahead" -gt 0 ]; then
          # Not a failure of THIS tool -- it only ever syncs down, never
          # pushes. Still not "OK": unpushed local commits are the strand
          # hazard CLAUDE.md documents at length (see ref_convergence.py).
          log "$name AHEAD_UNPUSHED (ahead=$ahead behind=0) -- $ahead local commit(s) not on origin/$branch; this job never pushes"
        else
          log "$name OK (ahead=0 behind=0)"
        fi
        exit 0
      fi

      # behind > 0. Not wedged, so ahead is necessarily 0 here.
      if [ -n "$blocking_list" ]; then
        log "$name BEHIND_NOT_SYNCED (ahead=$ahead behind=$behind) -- NOT synced; ff blocked by uncommitted tracked change(s): $blocking_list"
        exit 0
      fi

      if "$GIT" merge --ff-only --quiet "origin/$branch" 2>/dev/null; then
        if [ -n "$derived_list" ]; then
          log "$name SYNCED (ff-only, was behind $behind; telemetry dirt kept untouched: $derived_list)"
        else
          log "$name SYNCED (ff-only, was behind $behind)"
        fi
      else
        if [ -n "$derived_list" ]; then
          log "$name BEHIND_NOT_SYNCED (ahead=$ahead behind=$behind) -- ff-only refused; incoming commits would overwrite telemetry dirt: $derived_list"
        else
          log "$name NEEDS_HUMAN (behind $behind, ff-only refused with a clean tree)"
        fi
      fi
      exit 0
    fi

    # --- wedged: safety gate before any destructive op ---
    reason=""

    precious="$("$GIT" status --porcelain --untracked-files=no \
      | sed 's/^...//' \
      | grep -Ev "$TELEMETRY_RE" | head -1)"
    [ -n "$precious" ] && reason="uncommitted non-telemetry change: $precious"

    if [ -z "$reason" ] && [ "$ahead" -gt 0 ]; then
      for sha in $("$GIT" rev-list "origin/$branch..HEAD" 2>/dev/null); do
        nonteleme="$("$GIT" diff-tree --no-commit-id --name-only -r "$sha" 2>/dev/null \
          | grep -Ev "$TELEMETRY_RE" | head -1)"
        if [ -n "$nonteleme" ]; then
          reason="ahead commit ${sha:0:9} touches non-telemetry: $nonteleme"
          break
        fi
      done
    fi

    if [ -n "$reason" ]; then
      log "$name NEEDS_HUMAN (wedged: rebase=$has_rebase detached=$detached ahead=$ahead behind=$behind) -- $reason"
      exit 0
    fi

    # --- SAFE signature: non-destructive-to-work repair ---
    log "$name REPAIRING (rebase=$has_rebase detached=$detached ahead=$ahead behind=$behind; ahead all telemetry, no precious uncommitted)"
    if [ "$has_rebase" = 1 ]; then
      "$GIT" rebase --abort 2>/dev/null || "$GIT" rebase --quit 2>/dev/null
    fi
    "$GIT" checkout -- evidence/experiments/runner_heartbeats evidence/experiments/runner_status evidence/experiments/runner_commands 2>/dev/null
    "$GIT" checkout -f -B "$branch" "origin/$branch" --quiet 2>/dev/null

    still_rebase=0
    { [ -d .git/rebase-merge ] || [ -d .git/rebase-apply ]; } && still_rebase=1
    cur="$("$GIT" symbolic-ref -q --short HEAD 2>/dev/null)"
    head_sha="$("$GIT" rev-parse HEAD 2>/dev/null)"
    orig_sha="$("$GIT" rev-parse "origin/$branch" 2>/dev/null)"
    if [ "$still_rebase" = 0 ] && [ "$cur" = "$branch" ] && [ "$head_sha" = "$orig_sha" ]; then
      log "$name REPAIRED -> $branch @ ${head_sha:0:9} (== origin/$branch)"
    else
      log "$name NEEDS_HUMAN (repair did not converge: rebase=$still_rebase branch=$cur head=${head_sha:0:9} origin=${orig_sha:0:9})"
    fi
  )
done

log "=== run end ==="
exit 0
