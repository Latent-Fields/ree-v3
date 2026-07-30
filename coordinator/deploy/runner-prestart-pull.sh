#!/bin/sh
# Refresh the ree-v3 checkout BEFORE ree-runner.service loads experiment_runner.py.
#
# WHY THIS EXISTS
# --------------
# The cloud-scaler powers workers on and off continuously. A worker that powers
# on loads whatever experiment_runner.py its disk held when it was last powered
# OFF, and Python does not hot-reload -- so the running process keeps executing
# launch-time code for its ENTIRE session even though its own loop pulls newer
# code to disk within ~60s. _refresh_runner_version() surfaces that skew as
# "r4530 9077eb7 2026-07-30 (running r4515 e68d52b)", but nothing REFRESHED the
# code before the process loaded it. So every runner bugfix had a silent
# one-boot lag.
#
# Measured 2026-07-30 (the incident this closes):
#   - `systemctl cat ree-runner | grep -c ExecStartPre` was 0 on every worker.
#   - The only boot-time git refresh was ree-git-sync-repair.timer, which is
#     OnBootSec=10min -- it fires roughly TEN MINUTES AFTER the runner has
#     already started and claimed work, so it cannot protect the code load.
#   - ree-cloud-3 kept running r4515 e68d52b for 10h after the prepull-stash fix
#     (9077eb7073) landed on origin/main at 06:51:52Z, because its process
#     started 2026-07-29T21:27Z.
#   - ree-cloud-2 then cold-booted at 07:35:34Z straight onto r4528 ba79ebb --
#     which is OLDER than the fix. That is this failure, live, on a fresh boot.
#
# SCOPE: ree-v3 ONLY. Deliberately not REE_assembly.
# The defect is code LOADED INTO THE PYTHON PROCESS. All runner code, ree_core,
# the experiment drivers and validate_queue.py's input live in ree-v3;
# REE_assembly is a data checkout that nothing imports at launch and that the
# runner's own loop pulls anyway. Pulling it here would widen the blast radius
# onto the hub's writer-adjacent trees for no benefit to this failure.
#
# WHICH CHECKOUT: whatever the unit's WorkingDirectory is -- we never name a path.
# systemd applies WorkingDirectory to ALL Exec* lines, and the hub's drop-in sets
# it to /home/ree/REE_Working_runner/ree-v3 (an ISOLATED checkout, separate from
# the ~/REE_Working tree that sync_daemon owns -- dirtying that one wedges every
# phase3 writer). Operating on cwd is therefore correct on the hub AND on the
# workers with zero per-box special-casing. Do NOT "clarify" this by hardcoding
# an absolute path: that reintroduces the hub co-tenant wedge.
#
# FAILS OPEN, ALWAYS. A network blip at boot must never leave a worker with a
# dead runner -- a runner on stale code is strictly better than no runner. Three
# independent layers guarantee that:
#   1. the unit uses `ExecStartPre=-` (leading dash: systemd ignores the exit
#      status entirely),
#   2. this script ends with an unconditional `exit 0`,
#   3. every git call is wrapped in `timeout`, because the unit's
#      TimeoutStartUSec is 1min30s and it COVERS ExecStartPre -- a hung git
#      would otherwise burn the start timeout and fail the unit, turning a
#      hygiene step into an outage.
#
# The `main() { ... }; main "$@"` shape is not stylistic: this script lives in
# the very tree it pulls, so `sh` must finish parsing the body before the pull
# can possibly rewrite the file underneath its file descriptor.
#
# Output is ASCII-only (journal is read on mixed terminals).

main() {
    GIT_TIMEOUT=30

    if ! timeout 5 git rev-parse --git-dir >/dev/null 2>&1; then
        echo "[prestart] $(pwd) is not a git checkout -- skipping code refresh"
        return 0
    fi

    before=$(timeout 5 git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
    echo "[prestart] refreshing ree-v3 before runner launch (at $before, cwd $(pwd))"

    # --rebase --autostash matches experiment_runner.git_pull's own idiom, and
    # for the same reason: a worker tree legitimately carries ephemeral dirt
    # (runner_heartbeats/<host>.json, runner_status/<host>.json, the queue
    # file's claim flag). Plain --ff-only refuses on those even when a
    # fast-forward would not conflict at content level -- it would fail open on
    # exactly the boxes this is meant to protect, which is no fix at all.
    if timeout "$GIT_TIMEOUT" git pull --rebase --autostash 2>&1; then
        after=$(timeout 5 git rev-parse --short=7 HEAD 2>/dev/null || echo "unknown")
        if [ "$before" = "$after" ]; then
            echo "[prestart] ree-v3 already current at $after"
        else
            echo "[prestart] ree-v3 advanced $before -> $after before runner launch"
        fi
        return 0
    fi

    # A failed --rebase pull can stop mid-rebase and leave .git/rebase-merge
    # behind, which wedges EVERY later git op on this repo ("there is already a
    # rebase-merge directory") -- including the runner's own first in-loop pull,
    # 60 seconds from now. Clear it so we degrade to "stale code" (recoverable
    # in one tick) rather than "wedged checkout" (needs a human). Losing work
    # here is not possible: the abort restores the autostash, and a failed pull
    # changed nothing to begin with.
    echo "[prestart] WARN ree-v3 pull failed; starting runner on the code" \
         "already on disk ($before) and clearing any partial rebase"
    timeout 10 git rebase --abort >/dev/null 2>&1
    timeout 10 git rebase --quit  >/dev/null 2>&1
    return 0
}

main "$@"

# Unconditional: see FAILS OPEN above. Never propagate a non-zero status.
exit 0
