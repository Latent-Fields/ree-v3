#!/usr/bin/env bash
# clear_worker_stashes.sh
#
# Drop accumulated dormant `autostash` entries on a fleet worker's two
# git checkouts (ree-v3 and REE_assembly). Cloud-3 accumulated 191
# stashes by 2026-05-31, all from historical failure modes where
# `git pull --rebase --autostash` left the autostash in conflict and
# nobody dropped it. The runner now drops its own autostash entries
# automatically when ephemeral-conflict recovery fires, but workers
# that ran the pre-fix runner for a long time have a backlog to clear.
#
# Run against a single worker (this machine):
#   bash coordinator/deploy/clear_worker_stashes.sh
#
# Run against a remote worker via SSH:
#   ssh ree@<worker-ip> 'bash -s' < coordinator/deploy/clear_worker_stashes.sh
#
# Or wrap in a fleet loop:
#   for host in ree-cloud-1 ree-cloud-2 ree-cloud-3 ree-cloud-4; do
#     echo "=== $host ==="
#     ssh "ree@$host" 'bash -s' \
#       < coordinator/deploy/clear_worker_stashes.sh
#   done
#
# Env overrides:
#   REE_V3        default $HOME/REE_Working/ree-v3
#   REE_ASSEMBLY  default $HOME/REE_Working/REE_assembly
#   DRY_RUN       set to 1 to list-only without dropping (default 0)
#
# Exit 0 on success (whether or not any stashes were dropped).
# Exit non-zero only if both checkouts were unreadable.

set -u

REE_V3="${REE_V3:-$HOME/REE_Working/ree-v3}"
REE_ASSEMBLY="${REE_ASSEMBLY:-$HOME/REE_Working/REE_assembly}"
DRY_RUN="${DRY_RUN:-0}"

log() { printf '[clear-stashes] %s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

clear_repo() {
  local repo="$1"
  local label="$2"
  if [ ! -d "$repo/.git" ]; then
    log "$label: $repo missing or not a git repo; skipping."
    return 1
  fi
  local count
  count="$(git -C "$repo" stash list 2>/dev/null | wc -l | tr -d ' ')"
  log "$label: $count stash entries pre-cleanup."
  if [ "$count" -eq 0 ]; then
    return 0
  fi
  if [ "$DRY_RUN" = "1" ]; then
    log "$label: DRY_RUN=1, listing only:"
    git -C "$repo" stash list 2>&1 | head -20 | sed 's/^/    /'
    if [ "$count" -gt 20 ]; then
      log "    ... ($((count - 20)) more)"
    fi
    return 0
  fi
  # `git stash clear` drops all entries in one shot.
  if git -C "$repo" stash clear 2>/dev/null; then
    log "$label: cleared $count stash entries."
  else
    log "$label: WARN: git stash clear failed."
    return 1
  fi
  return 0
}

ok=0
clear_repo "$REE_V3" "ree-v3" && ok=1
clear_repo "$REE_ASSEMBLY" "REE_assembly" && ok=1

if [ "$ok" -eq 0 ]; then
  log "neither checkout could be cleared; exiting non-zero."
  exit 1
fi
log "done."
exit 0
