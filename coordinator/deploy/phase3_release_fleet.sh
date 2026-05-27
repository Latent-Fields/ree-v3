#!/usr/bin/env bash
# Phase-3 release-fleet: re-enable the cloud-scaler workflow after a
# successful cutover. Pairs with phase3_wake_fleet.sh.
#
# Run this AFTER phase3_verify.py PASSes, NOT before. If verify fails,
# you may want the scaler paused while you roll back -- don't re-enable
# it prematurely.
#
# Does NOT power off any workers. The scaler will make its own decisions
# at its next scheduled tick (15 min) about whether each worker should
# stay up based on queue contents + IDLE_GRACE_MIN.
#
# All output is ASCII-only.

set -euo pipefail

REPO="${REPO:-Latent-Fields/ree-v3}"
WORKFLOW="${WORKFLOW:-cloud-scaler.yml}"

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --help|-h)
      echo "Usage: $0 [--dry-run]"; exit 0 ;;
  esac
done

run() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY-RUN: $*"
  else
    "$@"
  fi
}

command -v gh >/dev/null || { echo "ERROR: gh CLI not installed"; exit 1; }

STATE=$(gh api "repos/$REPO/actions/workflows/$WORKFLOW" -q .state 2>/dev/null || echo unknown)
echo "cloud-scaler.yml state: $STATE"
if [[ "$STATE" == "active" ]]; then
  echo "  scaler already active; no change"
elif [[ "$STATE" == "disabled_manually" || "$STATE" == "disabled" ]]; then
  run gh workflow enable "$WORKFLOW" --repo "$REPO"
  echo "  scaler re-enabled"
else
  echo "ERROR: unexpected workflow state '$STATE'; investigate" >&2
  exit 1
fi
