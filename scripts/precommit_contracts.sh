#!/usr/bin/env bash
# precommit_contracts.sh -- run ree-v3/tests/contracts when staged changes
# touch ree_core/** or experiments/_lib/**, AND run validate_experiments.py
# --strict on staged experiments/v3_exq_*.py paths. Both checks self-gate: if
# no relevant paths are staged, the corresponding block is a no-op.
#
# The contract SUITE (Block 2) is ROUTED to a free machine (2026-07-29): a
# free-memory-gated choice between the local Mac (only when it has a real
# margin) and the cloud fleet via remote_pytest.sh. Coverage is unchanged --
# the full suite runs either way; only WHERE it runs moves, to keep the 8GB
# multi-session Mac off the OOM edge. See the Block 2 header below.
#
# Called from the PreToolUse hook in REE_Working/.claude/settings.json on
# any `git commit` bash invocation. Self-gates: if no ree_core/** or
# experiments/_lib/** paths are staged in the ree-v3 repo, this script exits 0
# with no output so commits to REE_assembly / other repos aren't penalised.
#
# Exit codes:
#   0 -- nothing to check, or contracts passed
#   2 -- contracts failed (blocks the commit; same code as validate_queue)
#   3 -- internal error (e.g. ree-v3 repo missing, pytest unavailable)
#
# Usage:
#   bash ree-v3/scripts/precommit_contracts.sh [--no-block]
#
# --no-block: report failure but return 0 (for CI/advisory use).

set -u

NO_BLOCK=0
if [ "${1:-}" = "--no-block" ]; then
    NO_BLOCK=1
fi

# Resolve ree-v3 repo root, worktree-aware. Priority order, each tier
# accepted only if it actually looks like a ree-v3 checkout (has ree_core/
# and tests/contracts/) -- see is_ree_v3_repo below:
#
#  1. `git rev-parse --show-toplevel`, called BEFORE anything else touches
#     cwd. Git invokes hooks with the working tree already correct for the
#     commit in question -- including a `git worktree add` checkout of
#     ree-v3 other than the primary shared one (confirmed empirically: cwd
#     is the worktree root, and GIT_DIR/GIT_INDEX_FILE are set to that
#     worktree's own gitdir/index, both inherited by this script when
#     invoked via pre-commit.local). This is what makes a worktree's commit
#     validate ITS OWN staged tree by default -- the fix for the
#     dazzling-taussig-f58f4c worktree-blindness bug (2026-07-24): before
#     this, a script that existed only in the worktree made Block 1 read it
#     off the wrong (shared-checkout) disk path and fail with
#     FileNotFoundError, and Block 2 ran contracts against the shared
#     checkout's unrelated -- possibly multi-session-dirty -- tree instead
#     of what was actually staged.
#  2. CLAUDE_PROJECT_DIR/ree-v3 -- set by the Claude Code harness. Correct
#     when cwd is NOT inside any ree-v3 tree at all, e.g. a Claude Code
#     umbrella-repo worktree session (`.claude/worktrees/<slug>`, which has
#     no ree-v3 of its own) invoking this via the settings.json PreToolUse
#     hook rather than the git-level one.
#  3. this script's own on-disk location -- last resort for manual
#     invocation with neither of the above.
#
# A resolved candidate that ISN'T a real ree-v3 checkout is rejected and the
# next tier tried, rather than exiting 0 outright -- so an umbrella-worktree
# session (tier 1 resolves to the umbrella repo, not ree-v3) still falls
# through to tier 2/3 instead of silently skipping the gate.
is_ree_v3_repo() {
    [ -n "$1" ] && [ -d "$1/ree_core" ] && [ -d "$1/tests/contracts" ]
}

REPO="$(git rev-parse --show-toplevel 2>/dev/null)"
if ! is_ree_v3_repo "$REPO"; then
    REPO=""
    if [ -n "${CLAUDE_PROJECT_DIR:-}" ] && is_ree_v3_repo "$CLAUDE_PROJECT_DIR/ree-v3"; then
        REPO="$CLAUDE_PROJECT_DIR/ree-v3"
    fi
fi
if ! is_ree_v3_repo "$REPO"; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if ! is_ree_v3_repo "$REPO"; then
    # Defensive: if the ree-v3 layout isn't what we expect, don't block
    # arbitrary commits.
    exit 0
fi

# Pick a python with torch. /opt/local/bin/python3 is the project default;
# fall back to PATH.
PY="/opt/local/bin/python3"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || true)"
fi
if [ -z "$PY" ]; then
    echo "[precommit_contracts] python3 not found; skipping contracts" >&2
    exit 3
fi

STAGED=$(git -C "$REPO" diff --cached --name-only 2>/dev/null || true)

# Block 1: experiments/v3_exq_*.py conformance (ratchet -- only enforces on
# new/modified scripts). Catches scripts that don't import experiment_protocol
# or don't call emit_outcome in __main__. Backward-compatible: legacy
# untouched scripts continue to work via the runner's stdout fallback. New
# and edited scripts must conform on commit.
STAGED_EXPERIMENTS=$(echo "$STAGED" | grep -E '^experiments/v3_exq_.*\.py$' || true)
if [ -n "$STAGED_EXPERIMENTS" ] && [ -f "$REPO/validate_experiments.py" ]; then
    echo "[precommit_contracts] staged experiment script(s) -- running validate_experiments.py --strict" >&2
    # shellcheck disable=SC2086
    if ! (cd "$REPO" && "$PY" validate_experiments.py --strict --quiet --paths $STAGED_EXPERIMENTS) >&2; then
        echo "[precommit_contracts] non-conforming experiment script(s) -- blocking commit" >&2
        echo "[precommit_contracts] each staged experiment must import experiment_protocol and call emit_outcome(...) in __main__" >&2
        echo "[precommit_contracts] retrofit with: /opt/local/bin/python3 scripts/retrofit_experiments.py --apply --paths <script>" >&2
        if [ "$NO_BLOCK" = "1" ]; then
            :
        else
            exit 2
        fi
    fi
fi

# Block 1b: manifest-writer chokepoint gate (pack_writer_single_writer_migration_plan
# sec 7 item 3; experimental_recording_standard sec 4). A NEW/modified experiment
# script that hand-rolls a raw json.dump flat-manifest tail bypasses the always-record
# core (substrate_hash / machine / elapsed_seconds / config / seeds via
# stamp_recording_core) instead of routing through pack_writer.write_flat_manifest --
# the recording-debt the migration closes. This is the last uncovered path: a script
# committed OUTSIDE the /queue-experiment skill (direct edit, another agent, plain CLI)
# could reintroduce a raw json.dump. Scope is ALL staged experiments/v3_*.py (broader
# than Block 1's v3_exq_ glob -- the debt is a regression in ANY v3 script) via
# --diff-filter=ACM (added/copied/modified only; a deleted path is never read).
# `--checks manifest_writer` keeps this SURGICAL: it runs ONLY the manifest-writer lint,
# so it does NOT expand the emit_outcome/degeneracy/arm-fingerprint contracts onto the
# non-v3_exq_ scripts it also scopes. The lint is HARD under --paths and respects the
# MANIFEST_WRITER_EXEMPT opt-out; the full-glob advisory backlog stays advisory. No-op
# when no v3 script is staged (a docs/queue-only commit is unaffected).
STAGED_V3=$(git -C "$REPO" diff --cached --name-only --diff-filter=ACM -- 'experiments/v3_*.py' 2>/dev/null || true)
if [ -n "$STAGED_V3" ] && [ -f "$REPO/validate_experiments.py" ]; then
    echo "[precommit_contracts] staged v3 experiment script(s) -- manifest-writer chokepoint gate" >&2
    # shellcheck disable=SC2086
    if ! (cd "$REPO" && "$PY" validate_experiments.py --strict --quiet --checks manifest_writer --paths $STAGED_V3) >&2; then
        echo "[precommit_contracts] staged experiment hand-rolls a flat-manifest json.dump -- blocking commit" >&2
        echo "[precommit_contracts] route the write through experiments/pack_writer.write_flat_manifest(...)" >&2
        echo "[precommit_contracts] or (if deliberately outside the standard) add MANIFEST_WRITER_EXEMPT = \"<reason>\"" >&2
        if [ "$NO_BLOCK" = "1" ]; then
            :
        else
            exit 2
        fi
    fi
fi

# Block 1c: staged experiments/*.py outside _lib/ -> corpus-lint subset
# (REE_assembly/evidence/planning/experiment_verification_harness_plan.md, Gap 1).
#
# Block 2 below runs the full ~1873-test contracts suite (all 47
# tests/contracts/test_*_lint.py corpus lints included) but ONLY when
# ree_core/ or experiments/_lib/ is staged. A brand-new experiments/v3_exq_*.py
# -- the single most common artifact /queue-experiment produces -- touches
# neither, so it got only Block 1 (conformance) and Block 1b (manifest-writer
# only): it could introduce a fresh instance of an already-known bad pattern
# (a new test_dead_z_goal_stream_lint-shaped bug, a new degenerate contract)
# undetected until some UNRELATED later commit happened to touch ree_core/ or
# _lib/ and a corpus-count pin broke elsewhere, misattributed to whatever was
# staged then. Same root cause, same shape, as the two incidents Block 2's own
# header documents for ITS trigger scope (mech457_retention_trajectory_probe,
# the coordinator/-not-collected pytest-default-args incident) -- this closes
# the analogous gap for experiment scripts specifically.
#
# Deliberately NOT the full suite: just the test_*_lint.py files (still using
# the shared tests/contracts/conftest.py::corpus_scan fixture, so coverage of
# the corpus lints is identical to what Block 2 would run) -- dominated by the
# ~100s corpus-scan setup rather than the full suite's ~13min, since it
# excludes the slow non-lint contracts (test_sd081_dualsystem_arbitration,
# test_graceful_timeout_lockfile, etc). Cheap enough to always run locally;
# unlike Block 2 there is no OOM-routing decision at this size. Skipped
# entirely when Block 2 will already run (that already covers every lint), so
# a commit touching both ree_core/ and an experiment script never double-runs.
#
# See tests/contracts/test_precommit_contracts_experiment_lint_scope.py.
STAGED_EXPERIMENT_PY=$(echo "$STAGED" | grep -E '^experiments/.*\.py$' | grep -v '^experiments/_lib/' || true)
if [ -n "$STAGED_EXPERIMENT_PY" ] && ! echo "$STAGED" | grep -qE '^(ree_core/|experiments/_lib/)'; then
    LINT_FILES=$(cd "$REPO" && ls tests/contracts/test_*_lint.py 2>/dev/null || true)
    if [ -n "$LINT_FILES" ]; then
        echo "[precommit_contracts] staged experiment script(s) outside _lib/ -- running corpus-lint subset" >&2
        # shellcheck disable=SC2086
        if ! (cd "$REPO" && "$PY" -m pytest -q --tb=line $LINT_FILES) >&2; then
            echo "[precommit_contracts] corpus-lint subset failed -- blocking commit" >&2
            echo "[precommit_contracts] fix the failing lint(s) or run with --no-verify to bypass" >&2
            if [ "$NO_BLOCK" = "1" ]; then
                :
            else
                exit 2
            fi
        fi
    fi
fi

# Block 2: ree_core/** OR experiments/_lib/** -> contracts test suite.
#
# experiments/_lib/ was added to this trigger 2026-07-19. It holds the SHARED
# training substrate consumed by every mech457-family experiment and bound into
# substrate_hash -- mech457_explorer_classes.py (train_a2c),
# mech457_bootstrap_explorer.py, mech457_fanout.py, capability_eval.py,
# arm_fingerprint.py and others. Before this it matched no block: Block 1 globs
# experiments/v3_exq_*.py, Block 1b globs experiments/v3_*.py, and Block 2 keyed
# on ^ree_core/ only. So a change to the actual A2C training loop committed with
# NO contract run and no warning, while a one-line ree_core/ change ran the full
# suite -- a fail-open guard, the dangerous direction. Concrete instance: the
# mech457_retention_trajectory_probe build (ree-v3 7e4f6e932b) added a hook inside
# train_a2c and modified BootstrapExplorerConfig; the gate never fired, and it was
# caught only because that session happened to run the suite by hand.
#
# FULL suite, not a targeted subset -- deliberately the same treatment ree_core/
# already gets. 21 of 171 contract files import experiments/_lib directly, but
# _lib is reached transitively from far more (train_a2c sits under the mech457
# drivers, the arm-fingerprint/reuse lints and the recording-standard checks), so
# any file-glob selection would itself be a fail-open guard with an unprincipled
# boundary. Matching the existing ree_core/ policy keeps one rule, not two.
if ! echo "$STAGED" | grep -qE '^(ree_core/|experiments/_lib/)'; then
    exit 0
fi

# ---------------------------------------------------------------------------
# Block 2 ROUTING (2026-07-29): the full contract suite must not run on the
# 8GB Mac. Several Claude sessions share this laptop; the suite peaks
# hundreds of MB and the box regularly sits at tens of MB free, so a local
# `pytest tests/contracts` here risks an OUT-OF-MEMORY (OOM) kill -- and an
# OOM kill auto-resumes every open session at once, recreating the overload.
# The repo rule is already "run the suite on the fleet, not the Mac"
# (REE_Working/CLAUDE.md "Running the test suite"); this gate now obeys it.
#
# WHERE the suite runs (this is routing only -- COVERAGE is unchanged; the
# full tests/contracts suite runs either way, so there is no false-pass risk,
# unlike a test-SUBSET optimisation):
#   * remote (default when the Mac is loaded): delegate to remote_pytest.sh,
#     which does its own fleet routing (hub first, then running workers, then
#     wakes an off worker), ships the WORKING TREE incl. uncommitted edits
#     (correct for a pre-commit gate), holds a cross-session lock, and blocks
#     on a red run. All the "route to a free box" logic lives there already;
#     this gate only decides local-vs-delegate.
#   * local (fast path, no rsync): the Mac IS a valid free machine when it
#     genuinely has a memory margin -- e.g. a solo/light session -- so skip
#     the ~74s worker wake + rsync round-trip and run here.
#
# The decision is memory-gated and errs toward remote when uncertain. Env:
#   REE_PRECOMMIT_CONTRACTS_TARGET       auto (default) | local | remote
#   REE_PRECOMMIT_CONTRACTS_LOCAL_FLOOR_MB   local only if avail >= this (default 3000)
#   REE_PRECOMMIT_CONTRACTS_FREE_MB      override the measured available MB (tests)
#   REE_PRECOMMIT_REMOTE_PYTEST          path to the fleet router (default resolved)
#   REE_PRECOMMIT_CONTRACTS_DECIDE_ONLY  1 -> print the resolved target and exit 0
#                                        WITHOUT running (tests only)
# FAIL-SAFE: if remote is chosen but the router is missing/not executable, fall
# back to LOCAL rather than skip the gate -- a skipped gate is the dangerous
# direction, and local is exactly today's behaviour.
#
# STAGGERED LOCAL-RACE FALLBACK (2026-08-01): when TARGET=remote is chosen,
# remote_pytest's OWN "expect roughly 2x" hub-contention advisory is a single
# 2026-07-20 measurement against a suite roughly half today's size and does
# not bound slowdown when the hub is nice'd behind a CPU-heavy experiment --
# confirmed 2026-08-01, a nice'd run took 2h+ against a ~13min clean
# baseline (~9-10x, not ~2x) while a same-tree local run finished in 13m53s.
# Rather than block indefinitely OR flip the default to local (which
# reintroduces the multi-session OOM incident above), the gate now starts
# remote immediately as before and, if it is still running after
# REE_PRECOMMIT_CONTRACTS_REMOTE_RACE_AFTER_MIN minutes (default 20),
# RE-CHECKS the same memory floor at that later point and -- only if it
# still clears, and no other session already holds the local-race lock --
# ALSO starts a local run, taking whichever finishes first. See the Block 2
# execution section below for the full env var list. Coverage is unchanged
# either way; this is still routing, not a test-subset shortcut.
# ---------------------------------------------------------------------------
TARGET="${REE_PRECOMMIT_CONTRACTS_TARGET:-auto}"
FLOOR_MB="${REE_PRECOMMIT_CONTRACTS_LOCAL_FLOOR_MB:-3000}"
REMOTE_PYTEST="${REE_PRECOMMIT_REMOTE_PYTEST:-$REPO/../scripts/remote_pytest.sh}"

# Available (reclaimable) memory in MB on macOS: free + speculative + inactive
# pages x page size. Returns -1 on any non-macOS / unparseable state, which the
# numeric compare below treats as "below floor" -> route remote.
mac_available_mb() {
    if [ -n "${REE_PRECOMMIT_CONTRACTS_FREE_MB:-}" ]; then
        echo "$REE_PRECOMMIT_CONTRACTS_FREE_MB"; return
    fi
    command -v vm_stat >/dev/null 2>&1 || { echo -1; return; }
    local psz vs f s i
    psz=$(sysctl -n hw.pagesize 2>/dev/null || echo 4096)
    vs=$(vm_stat 2>/dev/null) || { echo -1; return; }
    f=$(echo "$vs" | awk -F'[:.]' '/Pages free/{gsub(/ /,"",$2);print $2}')
    s=$(echo "$vs" | awk -F'[:.]' '/Pages speculative/{gsub(/ /,"",$2);print $2}')
    i=$(echo "$vs" | awk -F'[:.]' '/Pages inactive/{gsub(/ /,"",$2);print $2}')
    [ -z "$f" ] && { echo -1; return; }
    echo $(( ( (f + ${s:-0} + ${i:-0}) * psz ) / 1048576 ))
}

if [ "$TARGET" = "auto" ]; then
    AVAIL=$(mac_available_mb)
    if [ "${AVAIL:-0}" -ge "$FLOOR_MB" ] 2>/dev/null; then
        TARGET="local"
    else
        TARGET="remote"
    fi
    echo "[precommit_contracts] auto target=$TARGET (mac_available=${AVAIL}MB, local_floor=${FLOOR_MB}MB)" >&2
fi

if [ "$TARGET" = "remote" ] && [ ! -x "$REMOTE_PYTEST" ]; then
    echo "[precommit_contracts] remote_pytest not executable ($REMOTE_PYTEST) -- FALLING BACK to local (gate never skipped)" >&2
    TARGET="local"
fi

if [ "${REE_PRECOMMIT_CONTRACTS_DECIDE_ONLY:-0}" = "1" ]; then
    echo "[precommit_contracts] DECIDE_ONLY: resolved target=$TARGET" >&2
    exit 0
fi

echo "[precommit_contracts] ree_core/ or experiments/_lib/ change staged -- running contracts ($TARGET)" >&2

run_local_pytest() {
    # Testable indirection, same shape as REMOTE_PYTEST above.
    if [ -n "${REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST:-}" ]; then
        "$REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST"
    else
        (cd "$REPO" && "$PY" -m pytest -q --tb=line tests/contracts)
    fi
}

if [ "$TARGET" = "remote" ]; then
    # ---------------------------------------------------------------------
    # STAGGERED LOCAL-RACE FALLBACK (2026-08-01).
    #
    # remote_pytest's own hub-contention advisory ("expect roughly 2x") is a
    # single 2026-07-20 measurement against a suite roughly half today's
    # size, and does not bound slowdown when the hub is nice'd behind a
    # CPU-heavy experiment: confirmed 2026-08-01, a nice'd run took 2h+
    # against a ~13min clean baseline (~9-10x, not ~2x), while a same-tree
    # local run on this Mac finished in 13m53s clean (3217 passed).
    #
    # Neither extreme is right: blocking indefinitely on a stuck remote run
    # wastes real time; flipping the DEFAULT to local reintroduces the
    # documented multi-session OOM incident this gate exists to prevent
    # (CLAUDE.md "several sessions... load 25-30"). So: start remote
    # immediately as before (unchanged fast path when it's healthy), and if
    # it has not finished after RACE_AFTER minutes, ALSO start a local run
    # -- re-checking the SAME memory floor at that later point (Mac load can
    # only be judged when you're about to use it, not 20 minutes earlier),
    # and only if no other session already holds the local-race lock (the
    # guard against every contended session piling a local run onto the Mac
    # at once, which is exactly the incident this whole gate exists to
    # avoid). Whichever finishes first decides the commit; the other is
    # killed. Coverage is identical either way -- this is routing, not a
    # test-subset shortcut.
    #
    # Env:
    #   REE_PRECOMMIT_CONTRACTS_REMOTE_RACE_AFTER_MIN  minutes before trying
    #                                                   a local race (default 20)
    #   REE_PRECOMMIT_CONTRACTS_REMOTE_RACE_AFTER_SEC  test-only override, raw
    #                                                   seconds, takes precedence
    #   REE_PRECOMMIT_CONTRACTS_RACE_POLL_SEC          poll interval (default 15)
    #   REE_PRECOMMIT_CONTRACTS_DISABLE_RACE           1 -> old behaviour, block
    #                                                   on remote only, no race
    #   REE_PRECOMMIT_CONTRACTS_RACE_LOCK_DIR          cross-session lock dir
    #                                                   (default /tmp/ree_precommit_local_race.lock)
    #   REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST           test-only: run this
    #                                                   instead of `$PY -m pytest`
    # ---------------------------------------------------------------------
    RACE_AFTER_SEC="${REE_PRECOMMIT_CONTRACTS_REMOTE_RACE_AFTER_SEC:-}"
    if [ -z "$RACE_AFTER_SEC" ]; then
        RACE_AFTER_MIN="${REE_PRECOMMIT_CONTRACTS_REMOTE_RACE_AFTER_MIN:-20}"
        RACE_AFTER_SEC=$(( RACE_AFTER_MIN * 60 ))
    fi
    POLL_SEC="${REE_PRECOMMIT_CONTRACTS_RACE_POLL_SEC:-15}"
    RACE_LOCK_DIR="${REE_PRECOMMIT_CONTRACTS_RACE_LOCK_DIR:-/tmp/ree_precommit_local_race.lock}"
    RACE_LOCK_MAX_MIN="${REE_PRECOMMIT_CONTRACTS_RACE_LOCK_MAX_MIN:-45}"

    # Stale-lock recovery, mirroring remote_pytest.sh's own
    # REMOTE_PYTEST_LOCK_MAX_MIN pattern: a session that dies without
    # running its EXIT trap (SIGKILL, an OOM kill -- exactly the failure
    # mode this whole gate exists to avoid) would otherwise leave the lock
    # dir behind forever, silently disabling every future session's race.
    steal_stale_race_lock() {
        [ -d "$RACE_LOCK_DIR" ] || return 0
        local mtime now age_min
        # BSD stat (macOS, where this gate normally runs) first; GNU stat
        # (Linux, e.g. this function's own contract tests when routed
        # through the cloud fleet) as a fallback -- portable, not a
        # production requirement, since the gate itself always runs on the
        # Mac in real use.
        mtime=$(stat -f %m "$RACE_LOCK_DIR" 2>/dev/null) || mtime=$(stat -c %Y "$RACE_LOCK_DIR" 2>/dev/null)
        [ -n "$mtime" ] || return 0
        now=$(date +%s)
        age_min=$(( (now - mtime) / 60 ))
        if [ "$age_min" -ge "$RACE_LOCK_MAX_MIN" ]; then
            echo "[precommit_contracts] stealing stale race lock -- ${age_min}min old (>= ${RACE_LOCK_MAX_MIN}min)" >&2
            rmdir "$RACE_LOCK_DIR" >/dev/null 2>&1
        fi
    }

    WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ree_precommit_race.XXXXXX")"
    REMOTE_LOG="$WORKDIR/remote.log"; REMOTE_RC="$WORKDIR/remote.rc"
    LOCAL_LOG="$WORKDIR/local.log"; LOCAL_RC="$WORKDIR/local.rc"
    REMOTE_PID=""; LOCAL_PID=""; LOCAL_LOCK_HELD=""

    cleanup_race() {
        [ -n "$REMOTE_PID" ] && kill "$REMOTE_PID" >/dev/null 2>&1
        [ -n "$LOCAL_PID" ] && kill "$LOCAL_PID" >/dev/null 2>&1
        [ -n "$LOCAL_LOCK_HELD" ] && rmdir "$RACE_LOCK_DIR" >/dev/null 2>&1
        rm -rf "$WORKDIR" >/dev/null 2>&1
    }
    trap cleanup_race EXIT

    # remote_pytest ships the working tree to a free fleet box and runs
    # there. Unset the private index (ree_commit sets GIT_INDEX_FILE/GIT_DIR
    # when it re-runs this hook) so the router's own git calls see the
    # normal repo; it tests on-disk content, which is exactly what
    # ree_commit commits.
    ( cd "$REPO" && env -u GIT_INDEX_FILE -u GIT_DIR "$REMOTE_PYTEST" tests/contracts -q --tb=line >"$REMOTE_LOG" 2>&1
      echo $? >"$REMOTE_RC" ) &
    REMOTE_PID=$!

    if [ "${REE_PRECOMMIT_CONTRACTS_DISABLE_RACE:-0}" = "1" ]; then
        DEADLINE=0   # never race -- old blocking-on-remote-only behaviour
    else
        DEADLINE=$(( $(date +%s) + RACE_AFTER_SEC ))
    fi
    RACE_SKIP_LOGGED=""

    while :; do
        [ -f "$REMOTE_RC" ] && break
        if [ "$DEADLINE" != "0" ] && [ "$(date +%s)" -ge "$DEADLINE" ] && [ -z "$LOCAL_PID" ]; then
            NOW_AVAIL=$(mac_available_mb)
            steal_stale_race_lock
            if [ "${NOW_AVAIL:-0}" -ge "$FLOOR_MB" ] 2>/dev/null && mkdir "$RACE_LOCK_DIR" >/dev/null 2>&1; then
                LOCAL_LOCK_HELD=1
                echo "[precommit_contracts] remote still running after ${RACE_AFTER_SEC}s -- starting a local race (mac_available=${NOW_AVAIL}MB >= ${FLOOR_MB}MB)" >&2
                ( run_local_pytest >"$LOCAL_LOG" 2>&1; echo $? >"$LOCAL_RC" ) &
                LOCAL_PID=$!
            elif [ -z "$RACE_SKIP_LOGGED" ]; then
                RACE_SKIP_LOGGED=1
                if [ "${NOW_AVAIL:-0}" -lt "$FLOOR_MB" ] 2>/dev/null; then
                    echo "[precommit_contracts] remote still running after ${RACE_AFTER_SEC}s -- NOT racing locally (mac_available=${NOW_AVAIL}MB < ${FLOOR_MB}MB)" >&2
                else
                    echo "[precommit_contracts] remote still running after ${RACE_AFTER_SEC}s -- NOT racing locally (another session already racing: $RACE_LOCK_DIR held)" >&2
                fi
            fi
        fi
        [ -n "$LOCAL_PID" ] && [ -f "$LOCAL_RC" ] && break
        sleep "$POLL_SEC"
    done

    if [ -f "$LOCAL_RC" ]; then
        WINNER="local"; WINNER_RC="$(cat "$LOCAL_RC")"; cat "$LOCAL_LOG" >&2
    else
        WINNER="remote"; WINNER_RC="$(cat "$REMOTE_RC")"; cat "$REMOTE_LOG" >&2
    fi
    echo "[precommit_contracts] race winner: $WINNER (rc=$WINNER_RC)" >&2
    if [ "$WINNER_RC" = "0" ]; then
        exit 0
    fi
else
    if run_local_pytest >&2; then
        exit 0
    fi
fi

echo "[precommit_contracts] contract tests failed ($TARGET) -- blocking commit" >&2
echo "[precommit_contracts] fix the failing tests or run with --no-verify to bypass" >&2
if [ "$NO_BLOCK" = "1" ]; then
    exit 0
fi
exit 2
