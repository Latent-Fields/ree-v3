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
# Logs: journalctl -u ree-metaworker-healer -f  |  ~/ree_metaworker_healer.log
set -uo pipefail

REPO="${REE_REPO:-/home/ree/REE_Working}"
LOG="${REE_HEALER_LOG:-/home/ree/ree_metaworker_healer.log}"
LOCK="${REE_HEALER_LOCK:-/tmp/ree_metaworker_healer.lock}"
MAX_SEC="${REE_HEALER_MAX_SEC:-900}"
MACHINE="${REE_MACHINE:-$(hostname)}"
PY="$(command -v python3 || echo /usr/bin/python3)"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Non-reentrancy: one healer per box, ever. A cycle that outlives its timer
# interval must not be joined by a second one.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(ts)] healer: another cycle holds $LOCK; skipping" >> "$LOG"
  exit 0
fi

cd "$REPO" || { echo "[$(ts)] healer: no repo at $REPO" >> "$LOG"; exit 1; }

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
