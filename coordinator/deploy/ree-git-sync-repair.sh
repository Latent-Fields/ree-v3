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
      if [ "$ahead" = 0 ] && [ "$behind" -gt 0 ] && [ -z "$("$GIT" status --porcelain --untracked-files=no)" ]; then
        if "$GIT" merge --ff-only --quiet "origin/$branch" 2>/dev/null; then
          log "$name SYNCED (ff-only, was behind $behind)"
        else
          log "$name NEEDS_HUMAN (behind $behind, ff-only refused)"
        fi
      else
        log "$name OK (ahead=$ahead behind=$behind)"
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
