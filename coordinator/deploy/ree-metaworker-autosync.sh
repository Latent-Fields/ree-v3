#!/bin/bash
# REE metaworker umbrella autosync (systemd timer, independent of dispatch
# cadence and the scaler lease).
#
# Reference copy -- THIS TRACKED FILE IS CANONICAL, same convention as
# ree-metaworker-dispatch.sh / ree-metaworker-healer.sh (see
# check_metaworker_wrapper_deploy.py). The INSTALLED copy lives at
# /usr/local/bin/ree_metaworker_autosync.sh (underscore) and is a
# manually-synced DEPLOY TARGET, not the source of truth -- deploy by copying
# this file's content to that path.
#
# WHY THIS EXISTS (chip-20260901-fleet-autosync-repair)
# --------------------------------------------------------------------------
# ree-metaworker-dispatch.sh's own autosync only runs as part of a dispatch
# CYCLE -- and a cycle only fires when an Orchestrator session is actively
# leasing/dispatching this box. Since the resident 5-minute
# ree-metaworker.timer was retired in favour of the current 30-minute
# cadence (2026-08-24) and the Dispatcher/Healer split (2026-08-25), a box
# that is idle, coordination-plane-paused, cooldown-withheld, or simply
# between cycles does not sync AT ALL in the meantime -- there is no lower
# bound on how far it can drift while quiet.
#
# Confirmed live 2026-09-01: ree-cloud-4's REE_Working sat 305 commits
# behind origin/master while its dispatcher had been silently refusing to
# dispatch for 2+ days, reading a stale WORKING-TREE copy of TASK_CLAIMS.json
# that still held a pause lock origin had long since cleared -- a case where
# the very mechanism that would have corrected the staleness (a dispatch
# cycle noticing new work) could not run because of the staleness itself.
#
# ree-git-sync-repair.timer (30-min cadence, already installed fleet-wide)
# does NOT cover this gap: its cloud-worker REPOS list (see this directory's
# ree-git-sync-repair.sh) names only REE_assembly and ree-v3 -- never the
# umbrella REE_Working checkout that this wrapper, TASK_CLAIMS.json,
# TASK_CHIPS.json, scripts/, and every dispatched chip session's SKILL.md
# actually live in. Its unit Description ("wedge backstop, telemetry-only
# safe signature") is honest about what IT does; the umbrella just was never
# in its scope. This unit is deliberately NOT bolted onto that one -- a
# separate, honestly-named unit for a separate repo and a separate purpose,
# so the two never need to agree on a safety envelope for each other's repo.
#
# MECHANISM: safe_adopt_ref.py, not a bare `git pull` -- same reasoning as
# ree-metaworker-dispatch.sh's own autosync (see that script's header for the
# full account of why `--ff-only` refuses permanently against the normal
# residue of a ree_commit.py cherry-pick push retry). No --allow-discard or
# --force is ever passed here: an unattended periodic job must never
# auto-acknowledge a discarded local commit -- a refusal (exit 3) is logged
# and left for a human/session to audit per-commit, exactly as the dispatch
# wrapper's own autosync does.
#
# SCOPE: this unit touches ONLY the umbrella REE_Working checkout. It never
# touches ree-v3 or REE_assembly (that stays ree-git-sync-repair.timer's
# job), and it never touches TASK_CLAIMS.json / TASK_CHIPS.json content --
# those are coordinator-authoritative and rendered by the hub materializer,
# not written by this box at all.
#
# No hook of any kind is installed by this script or its unit files --
# CLAUDE.md is explicit that the reference-transaction ref-move GUARD HOOK
# must not be installed on cloud workers (their writers legitimately do
# non-fast-forward moves). Only safe_adopt_ref.py, a plain script, runs here.
#
# Manual:  /usr/local/bin/ree_metaworker_autosync.sh
# Log:     ~/ree_metaworker_autosync.log

set -u
export PATH="/opt/local/bin:/usr/local/bin:/home/ree/.local/bin:/usr/bin:/bin"

REPO="/Users/dgolden/REE_Working"
LOG="$HOME/ree_metaworker_autosync.log"
LOCKDIR="/tmp/ree_metaworker_autosync.lock"

ts() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

# Single-instance lock (mkdir is atomic) -- same convention as
# ree-git-sync-repair.sh. This job should complete in a few seconds; a stale
# lock older than 15 minutes is reclaimed rather than wedging the timer
# forever on a killed prior run.
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  if [ -d "$LOCKDIR" ] && [ "$(find "$LOCKDIR" -maxdepth 0 -mmin +15 2>/dev/null)" ]; then
    rmdir "$LOCKDIR" 2>/dev/null
    mkdir "$LOCKDIR" 2>/dev/null || { echo "[$(ts)] LOCK busy; exit" >> "$LOG"; exit 0; }
  else
    echo "[$(ts)] LOCK busy; exit" >> "$LOG"
    exit 0
  fi
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null' EXIT

if [ ! -d "$REPO/.git" ]; then
  echo "[$(ts)] SKIP (no .git at $REPO)" >> "$LOG"
  exit 0
fi

SYNC_OUT="$(/opt/local/bin/python3 "$REPO/scripts/safe_adopt_ref.py" --repo "$REPO" --branch master 2>&1)"
SYNC_RC=$?
echo "[$(ts)] $SYNC_OUT" >> "$LOG"
case "$SYNC_RC" in
  0)
    echo "[$(ts)] autosync: ok ($(git -C "$REPO" rev-parse --short HEAD 2>/dev/null))" >> "$LOG"
    ;;
  1)
    echo "[$(ts)] autosync: ADOPTED but the post-move skew repair needs a human -- see safe_adopt_ref.py output above" >> "$LOG"
    ;;
  3)
    echo "[$(ts)] autosync: REFUSED -- this box holds local commit(s) not on origin/master (see safe_adopt_ref.py output above for the shas); needs a per-commit audit, not a retry" >> "$LOG"
    ;;
  *)
    echo "[$(ts)] autosync: FAILED (safe_adopt_ref.py exit $SYNC_RC)" >> "$LOG"
    ;;
esac
exit 0
