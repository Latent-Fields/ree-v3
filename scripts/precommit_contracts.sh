#!/usr/bin/env bash
# precommit_contracts.sh -- run ree-v3/tests/contracts when staged changes
# touch ree_core/** or experiments/_lib/**, AND run validate_experiments.py
# --strict on staged experiments/v3_exq_*.py paths. Both checks self-gate: if
# no relevant paths are staged, the corresponding block is a no-op.
#
# Narrow author-side blocks sit between those two (1b manifest-writer, 1c
# corpus-lint subset, 1d flag registry, 1e substrate-docs index integrity),
# each keyed on the one path class that can break the check it runs. 1e is the
# only block keyed on docs: staged docs/substrate/*.md or CLAUDE.md.
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

# MAIN_REPO: the MAIN ree-v3 checkout, un-worktreed, distinct from REPO above.
# REPO is deliberately worktree-aware (tier 1 above) so Block 1/2 validate
# whatever tree is actually staged -- correct. But some paths are derived
# relative to REPO assuming it sits at REE_Working/ree-v3 with
# REE_Working/scripts/ as an adjacent sibling (chip-20260907-precommit-
# remote-pytest-worktree-resolution): from a ree-v3 WORKTREE (the pattern
# CLAUDE.md mandates for rebases and for landing against a busy shared
# checkout -- "Rebase via throwaway worktree"), REPO is the worktree
# directory, so "$REPO/../scripts/..." lands one level above the worktree,
# not at REE_Working/scripts/, and a fail-safe silently routes a gate that
# had just decided "remote" back onto the Mac instead (observed live
# 2026-09-07, session nifty-chebyshev-227274: 1% of tests/contracts in
# ~12min before the session killed it, on a box the gate had itself just
# measured as too memory-constrained to run locally).
#
# `git rev-parse --path-format=absolute --git-common-dir` un-worktrees
# correctly -- the SAME idiom CLAUDE.md already documents for the harness
# hook ("Worktree / Chipped Sessions" point 4: "The PreToolUse hook ...
# un-worktrees $CLAUDE_PROJECT_DIR before locating ... precommit_contracts.sh
# ... Do not 'simplify' that resolution back to a bare $CLAUDE_PROJECT_DIR").
# For a non-worktree checkout it returns the same .git REPO already sits
# next to, so MAIN_REPO == REPO there and nothing changes.
MAIN_REPO="$REPO"
GIT_COMMON_DIR="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"
if [ -n "$GIT_COMMON_DIR" ]; then
    MAIN_REPO_CANDIDATE="$(cd "$(dirname "$GIT_COMMON_DIR")" && pwd)"
    if is_ree_v3_repo "$MAIN_REPO_CANDIDATE"; then
        MAIN_REPO="$MAIN_REPO_CANDIDATE"
    fi
fi

# Pick a python with torch. /opt/local/bin/python3 is the project default;
# fall back to PATH.
#
# REE_PRECOMMIT_CONTRACTS_PYTHON overrides the first choice. This is not only a
# test knob: the fallback picks whatever `python3` is first on PATH, which is not
# guaranteed to have pytest -- on the hub it resolves to /usr/bin/python3 and
# `-m pytest` dies with "No module named pytest" while the real interpreter is a
# venv elsewhere. The blocks below then report "corpus-lint subset failed" and
# BLOCK, which is the safe direction but names the wrong cause. Point this at the
# interpreter that actually carries pytest when the default is absent.
PY="${REE_PRECOMMIT_CONTRACTS_PYTHON:-/opt/local/bin/python3}"
if [ ! -x "$PY" ]; then
    PY="$(command -v python3 || true)"
fi
if [ -z "$PY" ]; then
    echo "[precommit_contracts] python3 not found; skipping contracts" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# COMMIT-CONTENT STAGING (2026-09-18, chip-20260917-precommit-gate-shared-tree)
#
# THE GATE MUST TEST WHAT IT CERTIFIES, AND UNTIL NOW IT DID NOT.
#
# Every block below used to run against $REPO -- the ambient WORKING TREE. In
# the shared checkout at /Users/dgolden/REE_Working/ree-v3 that tree is not this
# commit: it is HEAD, plus this session's staged paths, plus this session's
# UNSTAGED edits, plus every concurrent session's uncommitted edits, plus every
# untracked file anyone has dropped in it. ree_commit.py meanwhile commits
# exactly the paths it was given, from a private index seeded `read-tree HEAD`.
# So the tree that was tested and the tree that gets committed are different
# trees, and the gap is everyone else's work in progress.
#
# That single fact produced BOTH failure directions, and the second is the one
# that matters:
#
#  * FALSE FAIL (measured 2026-09-17, three sessions in one afternoon). A
#    contract test file written for a build whose implementation is deliberately
#    not applied yet fails, and takes down whatever OTHER session happens to be
#    gating. Session substrate-build-20260917-triad: 16 failed / 5081 passed,
#    all 16 in ONE foreign file belonging to a different session's unlanded
#    build, zero in its own paths. Session substrate-build-20260917-allon: four
#    attempts over ~100 minutes, runs 2 and 3 blocked by 26 then 16 failures
#    located ENTIRELY in two foreign untracked test files, zero in its own
#    paths. Its verdict: "with four sessions in one checkout this gate is close
#    to structurally unsatisfiable from the shared tree."
#
#  * FALSE PASS -- the dangerous direction, and not hypothetical. A foreign (or
#    your own unstaged) file can SATISFY a dependency your commit introduces,
#    so the gate goes green on a combination that will never exist on trunk.
#    Demonstrated live on 2026-09-18 against the checkout as it then stood:
#    ree_core/hippocampal/ghost_goal_bank.py (tracked, modified) carries a
#    module-scope `from ree_core.hippocampal.possibility_topology import
#    PossibilityTopology`, while possibility_topology.py was UNTRACKED -- not in
#    HEAD. Committing ghost_goal_bank.py alone:
#        ambient working tree  -> import OK      -> gate GREEN -> trunk broken
#        isolated commit tree  -> ModuleNotFoundError -> gate RED (correct)
#    This is exactly CLAUDE.md's "(a2) coupled set" hazard -- an implementation
#    and its pinned contract, each individually complete -- and the gate was on
#    the wrong side of it.
#
# THE FIX IS THE KNOWN-GOOD WORKAROUND, MOVED FROM THE OPERATOR INTO THE GATE.
# Two sessions went green on 2026-09-17 by hand-rolling a throwaway worktree at
# origin/main containing only their own files; remote_pytest.sh's own header
# already recommends a throwaway worktree for any run that must not see the
# shared tree. This does the same thing automatically, and exactly rather than
# approximately: `git write-tree` against the index the hook was handed IS the
# tree this commit will carry (ree_commit.py's private index is HEAD + the
# declared paths; a plain `git commit` stages the same way), so the worktree is
# materialised from that tree via a throwaway commit-tree object. No guessing
# about which paths to copy.
#
# DOES THIS WEAKEN THE GATE? Argued explicitly, because it is the one question
# that matters and "it's faster" is not an answer.
#
#  * Coverage of the COMMITTED content is unchanged -- same suite, same tests,
#    same trigger scope. Only the tree moves, and it moves ONTO the thing being
#    certified. Both failure directions above are removed by the same change.
#  * What is genuinely LOST: another session's UNCOMMITTED work is no longer
#    co-tested with yours. So two mutually-incompatible changes that are both
#    uncommitted AT THE SAME MOMENT can now both pass in isolation.
#  * Why that loss is small and bounded. The shared checkout means a landed
#    commit moves HEAD for everyone, and the isolated tree is HEAD + your paths
#    -- so whoever commits SECOND still tests the combination. The residual is
#    only the window where two gates overlap in flight.
#  * Why the loss was largely notional anyway. The old "coverage" was a
#    coincidence of timing, not a guarantee (it caught the pair only if the
#    other session happened to have its edit on disk right then), and it was
#    MISATTRIBUTED when it did fire -- the failing session correctly concluded
#    "not my change" and retried, so the signal converted into hours of lost
#    wall clock and pressure toward --no-verify, not into a fix.
#  * And there IS a net underneath. .github/workflows/contract-tests.yml runs
#    `pytest tests/ -q` on a FRESH CLONE for every push to main touching the
#    code plane -- i.e. against the real merged trunk content, which is strictly
#    better evidence about a combination than the shared tree's accident. (That
#    workflow is currently red on one environment-sensitive file and is brushing
#    its 30-minute timeout; that is worth fixing on its own account, and does
#    not change which tree THIS gate should test.)
#
# FAIL-SAFE, in the direction CLAUDE.md requires: every failure to build the
# stage falls back to the OLD ambient-tree behaviour, loudly. A gate that runs
# on a contaminated tree is bad; a gate that does not run is worse. Isolation is
# never a reason to skip.
#
# Env:
#   REE_PRECOMMIT_CONTRACTS_ISOLATE  1 (default) | 0 -> test the ambient tree
#   REE_PRECOMMIT_CONTRACTS_STAGE_DIR  test-only: where to build the stage
# ---------------------------------------------------------------------------
RUN_ROOT="$REPO"      # the tree the gate actually tests -- $REPO until staged
STAGE_WT=""
STAGE_PARENT=""
STAGE_PARENT_OWNED=""   # 1 only when WE mktemp'd it, i.e. only then may we rm -rf it

cleanup_stage() {
    [ -n "$STAGE_WT" ] || return 0
    env -u GIT_INDEX_FILE -u GIT_DIR git -C "$REPO" worktree remove --force "$STAGE_WT" >/dev/null 2>&1
    env -u GIT_INDEX_FILE -u GIT_DIR git -C "$REPO" worktree prune >/dev/null 2>&1
    # rm -rf ONLY a directory this script created with mktemp. An operator- or
    # test-supplied REE_PRECOMMIT_CONTRACTS_STAGE_DIR is never deleted: it may be
    # a real directory with other contents, and a recursive delete of a path we
    # did not make is not ours to do.
    if [ "$STAGE_PARENT_OWNED" = "1" ] && [ -n "$STAGE_PARENT" ]; then
        rm -rf "$STAGE_PARENT" >/dev/null 2>&1
    fi
    STAGE_WT=""
    STAGE_PARENT=""
    STAGE_PARENT_OWNED=""
    return 0
}

# stage_commit_tree: materialise THIS COMMIT's tree into a throwaway worktree
# and point RUN_ROOT at it. Idempotent; returns 1 (and leaves RUN_ROOT at $REPO)
# on any failure, which is the documented fall-back-to-ambient path.
#
# NOTE ON `local`: the assignments below deliberately do NOT use it. `local x=$(cmd)`
# makes `$?` the status of `local`, not of `cmd`, so every `|| { fall back; }`
# here would become unreachable and a failed write-tree/commit-tree would sail
# on with an empty variable. Do not "tidy" these into locals.
#
# LEAK NOTE: a session killed with SIGKILL (or OOM-killed, the very case Block 2
# routes to avoid) never runs its EXIT trap, so its stage survives in TMPDIR and
# in `git worktree list`. No reaper is built for this on purpose -- one would
# have to distinguish a dead session's stage from a live concurrent session's,
# and removing the wrong one kills a running gate. The stage is named
# `ree_precommit_stage.*/ree-v3` so it is recognisable; `dev-doctor.sh` reports
# dead worktrees, and macOS reaps /var/folders on its own schedule.
stage_commit_tree() {
    [ -z "$STAGE_WT" ] || return 0
    if [ "${REE_PRECOMMIT_CONTRACTS_ISOLATE:-1}" != "1" ]; then
        echo "[precommit_contracts] isolation OFF (REE_PRECOMMIT_CONTRACTS_ISOLATE=0) -- testing the ambient working tree" >&2
        return 1
    fi

    # The index the hook was handed. Under ree_commit.py that is the PRIVATE
    # index (GIT_INDEX_FILE), which is exactly HEAD + the declared paths; under
    # a plain `git commit` it is the repo index. Either way write-tree yields
    # the tree the commit will carry. Unmerged paths make write-tree fail --
    # correctly, and we fall back rather than guessing.
    _stage_tree=$(git -C "$REPO" write-tree 2>/dev/null) || {
        echo "[precommit_contracts] write-tree failed (unmerged paths?) -- FALLING BACK to the ambient working tree" >&2
        return 1
    }
    # A throwaway commit so `worktree add` has something to check out. It is
    # never referenced by any ref, so it is unreachable and gc-able; it is NOT a
    # branch move, so the reference-transaction ref-move guard (which gates on
    # refs/heads/*) does not and must not see it.
    _stage_commit=$(env -u GIT_INDEX_FILE git -C "$REPO" commit-tree "$_stage_tree" -p HEAD \
                        -m "precommit gate staging (throwaway, unreferenced)" 2>/dev/null) || {
        echo "[precommit_contracts] commit-tree failed -- FALLING BACK to the ambient working tree" >&2
        return 1
    }
    _stage_parent="${REE_PRECOMMIT_CONTRACTS_STAGE_DIR:-}"
    _stage_owned=""
    if [ -z "$_stage_parent" ]; then
        _stage_parent=$(mktemp -d "${TMPDIR:-/tmp}/ree_precommit_stage.XXXXXX") || {
            echo "[precommit_contracts] mktemp failed -- FALLING BACK to the ambient working tree" >&2
            return 1
        }
        _stage_owned=1
    fi
    # Named ree-v3 and one level deep on purpose: tests/contracts/test_arm_reuse.py
    # resolves REE_Working via `git rev-parse --git-common-dir`, which from a
    # worktree points back at the MAIN checkout's .git and therefore still finds
    # the real REE_assembly sibling. remote_pytest.sh reads REE_assembly from a
    # fixed absolute path, so the remote route is unaffected either way.
    _stage_wt="$_stage_parent/ree-v3"
    if ! env -u GIT_INDEX_FILE -u GIT_DIR git -C "$REPO" worktree add --detach \
             "$_stage_wt" "$_stage_commit" >/dev/null 2>&1; then
        echo "[precommit_contracts] worktree add failed -- FALLING BACK to the ambient working tree" >&2
        [ "$_stage_owned" = "1" ] && rm -rf "$_stage_parent" >/dev/null 2>&1
        return 1
    fi
    STAGE_PARENT="$_stage_parent"
    STAGE_PARENT_OWNED="$_stage_owned"
    STAGE_WT="$_stage_wt"
    RUN_ROOT="$_stage_wt"
    trap cleanup_stage EXIT
    echo "[precommit_contracts] testing the COMMIT's content, isolated from the shared checkout" >&2
    echo "[precommit_contracts]   stage: $_stage_wt (tree $_stage_tree)" >&2
    return 0
}

STAGED=$(git -C "$REPO" diff --cached --name-only 2>/dev/null || true)

# Block 1: experiments/v3_exq_*.py conformance (ratchet -- only enforces on
# new/modified scripts). Catches scripts that don't import experiment_protocol
# or don't call emit_outcome in __main__. Backward-compatible: legacy
# untouched scripts continue to work via the runner's stdout fallback. New
# and edited scripts must conform on commit.
STAGED_EXPERIMENTS=$(echo "$STAGED" | grep -E '^experiments/v3_exq_.*\.py$' || true)
if [ -n "$STAGED_EXPERIMENTS" ] && [ -f "$REPO/validate_experiments.py" ]; then
    echo "[precommit_contracts] staged experiment script(s) -- running validate_experiments.py --strict" >&2
    stage_commit_tree || :
    # shellcheck disable=SC2086
    if ! (cd "$RUN_ROOT" && "$PY" validate_experiments.py --strict --quiet --paths $STAGED_EXPERIMENTS) >&2; then
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
    stage_commit_tree || :
    # shellcheck disable=SC2086
    if ! (cd "$RUN_ROOT" && "$PY" validate_experiments.py --strict --quiet --checks manifest_writer --paths $STAGED_V3) >&2; then
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
    stage_commit_tree || :
    LINT_FILES=$(cd "$RUN_ROOT" && ls tests/contracts/test_*_lint.py 2>/dev/null || true)
    if [ -n "$LINT_FILES" ]; then
        echo "[precommit_contracts] staged experiment script(s) outside _lib/ -- running corpus-lint subset" >&2
        # shellcheck disable=SC2086
        if ! (cd "$RUN_ROOT" && "$PY" -m pytest -q --tb=line $LINT_FILES) >&2; then
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

# Block 1d: staged ree_core/utils/config.py -> flag-registry currency check
# (chip-20260907-flag-registry-commit-time-gate).
#
# A new/renamed `use_*`/`*_enabled` config flag that is not categorized into
# PROBED / KNOWN_INERT / KNOWN_UNPROBED / KNOWN_UNPROBED_NESTED leaves
# tests/test_flag_inertness.py::test_flag_registry_is_current red on trunk
# until some LATER, unrelated session stumbles on it and has to re-diagnose it
# as pre-existing -- confirmed three times: GFLAG-0051/MECH-151 (ree-v3
# 84e211a), SD-e1 ITEM 2 (ree-v3 6447b45), SD-105 (ree-v3 ba95c43, red for
# three days before ree-v3 59936e9446 registered it).
#
# Block 2 below does NOT already cover this gap: it runs `pytest
# tests/contracts` only, and test_flag_inertness.py lives directly under
# tests/, not tests/contracts/ -- so a config.py change that also touches
# ree_core/ still would not exercise this test via Block 2.
#
# Deliberately a SINGLE fast introspection test (a dataclass-field scan of
# ree_core/utils/config.py; no agent/model construction) run LOCALLY, the same
# treatment as Block 1c -- not routed through Block 2's OOM-avoidance
# machinery, which exists for the ~13min full suite, not a sub-second check.
# All config dataclasses (including every nested one the flag scan walks) are
# defined directly in this one file, so gating on it alone is complete.
#
# See tests/contracts/test_precommit_contracts_flag_registry_scope.py.
STAGED_FLAG_CONFIG=$(echo "$STAGED" | grep -E '^ree_core/utils/config\.py$' || true)
if [ -n "$STAGED_FLAG_CONFIG" ]; then
    echo "[precommit_contracts] staged ree_core/utils/config.py -- checking flag registry currency" >&2
    stage_commit_tree || :
    if ! (cd "$RUN_ROOT" && "$PY" -m pytest -q --tb=short tests/test_flag_inertness.py::test_flag_registry_is_current) >&2; then
        echo "[precommit_contracts] flag registry is stale -- blocking commit" >&2
        echo "[precommit_contracts] add a behavioural probe to PROBED, or record the new/renamed flag in KNOWN_UNPROBED / KNOWN_UNPROBED_NESTED with a reason (tests/test_flag_inertness.py)" >&2
        echo "[precommit_contracts] or run with --no-verify to bypass" >&2
        if [ "$NO_BLOCK" = "1" ]; then
            :
        else
            exit 2
        fi
    fi
fi

# Block 1e: staged docs/substrate/*.md or CLAUDE.md -> substrate-index integrity
# (chip-20260919-wi1-index-contract-prose-false-positive).
#
# tests/docs_integrity/test_wi1_substrate_split_index_integrity.py is a pure
# text lint over docs/substrate/*.md and CLAUDE.md's "Substrate feature index".
# It used to live in tests/contracts/, i.e. under Block 2 -- which keys on
# ree_core/** and experiments/_lib/**, never on docs. So a docs commit that
# broke it was checked by NOTHING, and the red then blocked every UNRELATED
# ree_core commit fleet-wide until someone tripped over it (2026-09-19: ree-v3
# 65c1f72 landed one prose bullet opening with a foreign id; ~6.5h wedge, two
# ~28min remote gate runs burned by a bystander session).
#
# This block puts the check on the commit that can actually break it, and the
# move out of tests/contracts/ takes it off the commits that cannot. A DOCS
# lint must not be able to block a CODE commit.
#
# Run LOCALLY, same treatment as Blocks 1c/1d: no torch, no agent, no fleet --
# a glob and a few regexes, sub-second. CLAUDE.md is keyed as well as the
# substrate files because the index <-> file bijection reads both sides.
#
# Boxes with no commit guards (the cloud workers, where 65c1f72 was authored)
# are covered post-push by .github/workflows/docs-integrity.yml instead.
#
# See tests/contracts/test_precommit_contracts_docs_integrity_scope.py.
STAGED_SUBSTRATE_DOCS=$(echo "$STAGED" | grep -E '^(docs/substrate/[^/]+\.md|CLAUDE\.md)$' || true)
if [ -n "$STAGED_SUBSTRATE_DOCS" ]; then
    echo "[precommit_contracts] staged substrate docs -- checking substrate index integrity" >&2
    stage_commit_tree || :
    if ! (cd "$RUN_ROOT" && "$PY" -m pytest -q --tb=short tests/docs_integrity/test_wi1_substrate_split_index_integrity.py) >&2; then
        echo "[precommit_contracts] substrate index integrity failed -- blocking commit" >&2
        echo "[precommit_contracts] a bullet opening with an id its file's heading does not own: lead with a word (e.g. 'Note: ') if it is prose, or give the record its own docs/substrate file + CLAUDE.md index entry" >&2
        echo "[precommit_contracts] an unlinked/broken/duplicated index entry: fix CLAUDE.md's 'Substrate feature index' section" >&2
        echo "[precommit_contracts] or run with --no-verify to bypass" >&2
        if [ "$NO_BLOCK" = "1" ]; then
            :
        else
            exit 2
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

# Stage before the cache block: the cache key must describe the tree this run
# will actually exercise (see --hash-root below), not the ambient one.
stage_commit_tree || :

# ---------------------------------------------------------------------------
# Block 2 VALIDATION CACHE (2026-08-10): a HIT means this exact ree_core/ +
# experiments/_lib/ content was already validated on this machine class /
# toolchain within a bounded TTL -- skip re-paying the ~13min suite. See
# REE_assembly/docs/architecture/landing_integration_worker_investigation.md
# sec 6 for the design and scripts/validation_cache.py for the implementation.
#
# COVERAGE IS THE INVARIANT HERE, NOT SPEED: this is validation REUSE, never
# BYPASS -- a HIT only ever stands in for a full green run that provably
# already happened against equivalent content. FAIL-SAFE in every direction:
# a missing/unexecutable validation_cache.py, a corrupt or missing cache
# file, or ANY unexpected error inside it, is unconditionally a MISS -- the
# gate falls straight through to routing/running the suite exactly as if
# this whole block did not exist. A MISS never blocks; only a HIT ever skips
# the suite, and every HIT is loudly logged.
#
# Env:
#   REE_PRECOMMIT_VALIDATION_CACHE_DISABLE  1 -> skip this block entirely
#   REE_PRECOMMIT_VALIDATION_CACHE_PY       override path (tests only)
#   REE_PRECOMMIT_VALIDATION_CACHE_PATH     override cache file (tests only)
#   REE_PRECOMMIT_VALIDATION_CACHE_TTL_MIN  TTL in minutes (default 45)
# ---------------------------------------------------------------------------
VALIDATION_CACHE_TIER="ree-v3-contracts-full-suite"
VALIDATION_CACHE_PY="${REE_PRECOMMIT_VALIDATION_CACHE_PY:-$REPO/scripts/validation_cache.py}"
VALIDATION_CACHE_TTL_MIN="${REE_PRECOMMIT_VALIDATION_CACHE_TTL_MIN:-45}"

# Shared --repo-root/--tier/[--cache-path]/--ttl-minutes, an indexed array
# (bash 3.2+, no bash-4 associative-array/mapfile builtins -- CLAUDE.md Shell
# Portability) so the check and record call sites can never drift apart on
# the key, and paths with spaces survive intact (unlike a printf+word-split).
#
# --repo-root and --hash-root are DELIBERATELY DIFFERENT when the run is
# isolated. --repo-root is the MAIN checkout: it is where the cache FILE lives
# and where validation_cache.py commits it, and pointing it at the throwaway
# stage would try to commit into a detached worktree that is about to be
# deleted. --hash-root is what gets CONTENT-HASHED, and that must be the tree
# the suite actually runs against -- otherwise a PASS would be recorded under
# the ambient tree's key and a later commit with different staged content could
# HIT on it and skip the suite entirely. They collapse to the same path when
# staging fell back, which is exactly the pre-2026-09-18 behaviour.
#
# Side benefit worth naming, since it was raised as a blocker: keyed on the
# COMMIT's content, the key no longer moves every time another session edits
# ree_core/ in the shared checkout, so a 13-minute suite can now certify a key
# that still exists when a retry runs.
VALIDATION_CACHE_ARGS=(--repo-root "$REPO" --hash-root "$RUN_ROOT"
                       --tier "$VALIDATION_CACHE_TIER"
                       --ttl-minutes "$VALIDATION_CACHE_TTL_MIN")
if [ -n "${REE_PRECOMMIT_VALIDATION_CACHE_PATH:-}" ]; then
    VALIDATION_CACHE_ARGS+=(--cache-path "$REE_PRECOMMIT_VALIDATION_CACHE_PATH")
fi

record_validation_cache_result() {
    # Called unconditionally after the suite (or a cache hit's stand-in for
    # it) resolves -- validation_cache.py record itself no-ops (logged, not
    # silent) on --result fail, so the caller never has to branch on outcome.
    #
    # THIS RUNS INSIDE A PRE-COMMIT HOOK, so the commit it makes must never
    # move refs/heads/<branch>: ree_commit.py reads old_head, runs this hook,
    # then CAS's `update-ref <ref> <new> <old_head>`, and a ref move in here
    # makes that CAS fail -- the gate rejecting the very commit it is gating,
    # after paying the full ~13min suite. Deterministic on every cache
    # MISS+PASS (a HIT exits above, before this is reached), confirmed on five
    # historical instances (2026-08-27, chip-20260827-precommit-cache-self-
    # collision). validation_cache.py `record` defaults to ree_commit.py
    # --to-remote-tip for exactly this reason and needs no flag from here --
    # but --push is what makes that mode available, so do NOT drop it below.
    #
    # env -u (2026-09-25, chip-20260925-precommit-cache-index-leak-fix): git
    # exports the OUTER commit's GIT_INDEX_FILE -- and, from a linked worktree,
    # GIT_DIR -- to this hook, and record's nested ree_commit.py inherited them.
    # Its throwaway-worktree git calls then acted on the lander's own index and
    # gitdir: reproduced in a sandbox, the lander's staged set was reset, its
    # HEAD moved onto the cache commit, its commit failed, and the structural
    # re-apply fired this hook again -- the nested full gate seen on the hub
    # 2026-09-25 12:08-12:41 (REE_Working docs/reference/
    # commit_latency_diagnosis_20260925.md R4/P2). Same scrub the router call
    # below already applies; record needs none of them (every path it uses is
    # passed explicitly). validation_cache.py scrubs again, and ree_commit.py
    # refuses loudly if a caller ever leaks them -- do not drop this line on
    # the strength of those backstops.
    [ "${REE_PRECOMMIT_VALIDATION_CACHE_DISABLE:-0}" = "1" ] && return 0
    [ -f "$VALIDATION_CACHE_PY" ] || return 0
    env -u GIT_INDEX_FILE -u GIT_DIR -u GIT_WORK_TREE \
        "$PY" "$VALIDATION_CACHE_PY" record "${VALIDATION_CACHE_ARGS[@]}" --result "$1" --push >&2 || true
}

if [ "${REE_PRECOMMIT_VALIDATION_CACHE_DISABLE:-0}" != "1" ] && [ -f "$VALIDATION_CACHE_PY" ]; then
    if "$PY" "$VALIDATION_CACHE_PY" check "${VALIDATION_CACHE_ARGS[@]}" >&2; then
        echo "[precommit_contracts] validation cache HIT -- reusing a prior green run (see above), skipping the full suite" >&2
        exit 0
    fi
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
# FAIL-SAFE, revised 2026-09-08 (user decision, chip-20260907-precommit-remote-
# pytest-worktree-resolution's open design question): if remote is chosen but
# the router is missing/not executable, the response now DEPENDS on whether the
# Mac clears FLOOR_MB at that moment --
#   * at/above the floor: unchanged -- fall back to LOCAL (the Mac has a real
#     margin, so running here is safe; a skipped gate is the dangerous direction).
#   * BELOW the floor: fail LOUDLY and BLOCK the commit (exit 2) instead of
#     falling back to local. The old fall-back-to-local-always behaviour tied up
#     the laptop for hours twice on 2026-09-08 (a081ff3616's worktree-resolution
#     fix made the missing-router case rare but not impossible, e.g. a genuinely
#     unreachable REE_Working/scripts sibling) -- running the full suite on an
#     already memory-constrained Mac is worse than blocking the commit and
#     telling the operator to route explicitly. --no-block still downgrades this
#     to advisory (falls back to local anyway, loudly logged) for CI/advisory use.
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
# Resolved against MAIN_REPO (un-worktreed above), not REPO -- see the
# MAIN_REPO comment near the top of this file. REE_Working/scripts/ only
# sits next to the MAIN ree-v3 checkout, never next to a worktree.
REMOTE_PYTEST="${REE_PRECOMMIT_REMOTE_PYTEST:-$MAIN_REPO/../scripts/remote_pytest.sh}"

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
    ROUTER_MISSING_AVAIL=$(mac_available_mb)
    if [ "${ROUTER_MISSING_AVAIL:-0}" -lt "$FLOOR_MB" ] 2>/dev/null; then
        echo "[precommit_contracts] BLOCKING COMMIT: below the memory floor with no cloud router available (user decision 2026-09-08)" >&2
        echo "[precommit_contracts]   mac_available=${ROUTER_MISSING_AVAIL}MB < local_floor=${FLOOR_MB}MB" >&2
        echo "[precommit_contracts]   looked for the router at: $REMOTE_PYTEST (missing or not executable)" >&2
        echo "[precommit_contracts]   remedy 1: run the contracts on the fleet directly, then retry the commit --" >&2
        echo "[precommit_contracts]     scripts/remote_pytest.sh tests/contracts -q   (from the main checkout)" >&2
        echo "[precommit_contracts]   remedy 2: commit from the main checkout (/Users/dgolden/REE_Working), where the router resolves" >&2
        if [ "$NO_BLOCK" = "1" ]; then
            echo "[precommit_contracts] --no-block set -- falling back to local anyway (advisory mode)" >&2
        else
            exit 2
        fi
    else
        echo "[precommit_contracts] remote_pytest not executable ($REMOTE_PYTEST) -- FALLING BACK to local (mac_available=${ROUTER_MISSING_AVAIL}MB >= ${FLOOR_MB}MB, gate never skipped)" >&2
    fi
    TARGET="local"
fi

if [ "${REE_PRECOMMIT_CONTRACTS_DECIDE_ONLY:-0}" = "1" ]; then
    echo "[precommit_contracts] DECIDE_ONLY: resolved target=$TARGET" >&2
    exit 0
fi

echo "[precommit_contracts] ree_core/ or experiments/_lib/ change staged -- running contracts ($TARGET)" >&2

run_local_pytest() {
    # Testable indirection, same shape as REMOTE_PYTEST above.
    #
    # BOTH branches cd to RUN_ROOT. The stub branch used to run in the gate's own
    # cwd, which made the indirection UNFAITHFUL in exactly the dimension the
    # 2026-09-18 isolation change turns on: a stub could not observe which tree
    # the real pytest would have run against, so a test asserting "the suite runs
    # against the commit's tree" passed vacuously against the shared checkout.
    if [ -n "${REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST:-}" ]; then
        (cd "$RUN_ROOT" && "$REE_PRECOMMIT_CONTRACTS_LOCAL_PYTEST")
    else
        (cd "$RUN_ROOT" && "$PY" -m pytest -q --tb=line tests/contracts)
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
        # TERM the ROUTER itself, then its subshell. The router is the
        # subshell's CHILD, so killing only the subshell orphaned it and its
        # detached suite ran on unread -- after a local-race win, or a gate
        # killed mid-run (REE_Working commit_latency_diagnosis_20260925.md R5/P5;
        # the hub's orphaned 12:08-12:41 run). remote_pytest.sh (REE_Working
        # 2568d2b4b+) stops its remote run by run dir on TERM; an older one just
        # exits, as before. A router that already finished has no child here.
        [ -n "$REMOTE_PID" ] && pkill -TERM -P "$REMOTE_PID" >/dev/null 2>&1
        [ -n "$REMOTE_PID" ] && kill "$REMOTE_PID" >/dev/null 2>&1
        [ -n "$LOCAL_PID" ] && kill "$LOCAL_PID" >/dev/null 2>&1
        [ -n "$LOCAL_LOCK_HELD" ] && rmdir "$RACE_LOCK_DIR" >/dev/null 2>&1
        rm -rf "$WORKDIR" >/dev/null 2>&1
        # This trap REPLACES the one stage_commit_tree installed, so it must do
        # that job too or the staging worktree leaks into `git worktree list`.
        cleanup_stage
    }
    trap cleanup_race EXIT

    # remote_pytest ships the working tree to a free fleet box and runs
    # there. Unset the private index (ree_commit sets GIT_INDEX_FILE/GIT_DIR
    # when it re-runs this hook) so the router's own git calls see the
    # normal repo; it tests on-disk content, which is exactly what
    # ree_commit commits.
    #
    # REMOTE_PYTEST_CALLER_PID=$$ registers THIS gate with the router (P5
    # second half): if the gate dies without signalling it -- SIGKILL, an
    # OOM kill -- the router notices within seconds and stops the remote run
    # instead of finishing a suite nobody will read. ($$ is the gate's pid,
    # also inside this subshell.)
    # REMOTE_PYTEST_CACHE_CREDIT=0: the router's own P1b cache credit is for
    # ad-hoc runs on a clean branch tree; this gate records its own result
    # below (record_validation_cache_result), keyed on the staged tree. The
    # stage is a detached worktree, which the router already refuses to
    # credit -- this says it explicitly rather than relying on that.
    ( cd "$RUN_ROOT" && env -u GIT_INDEX_FILE -u GIT_DIR REMOTE_PYTEST_CALLER_PID=$$ REMOTE_PYTEST_CACHE_CREDIT=0 "$REMOTE_PYTEST" tests/contracts -q --tb=line >"$REMOTE_LOG" 2>&1
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

    # -----------------------------------------------------------------------
    # A ROUTING CONDITION IS NOT A RED SUITE (2026-09-18,
    # chip-20260917-precommit-gate-shared-tree defect 2).
    #
    # remote_pytest.sh's exit codes are documented in two OVERLAPPING bands:
    # "0-5 pytest's own result" and "2-8 this wrapper's PRE-RUN failures".
    # pytest really does return 2/3/4/5 (interrupted / internal error / usage
    # error / no tests collected), so rc=4 alone cannot distinguish "pytest
    # usage error" from "no box could be acquired". This gate used to treat any
    # non-zero as a contract failure and block the commit. Measured 2026-09-17:
    # session substrate-build-20260917-sd097 lost two rejected commit attempts
    # and then a 42-minute starved LOCAL run (aborted at 73%, zero failures, on
    # a Mac carrying ~17 other sessions' pytest processes) to that misreading.
    #
    # The distinction is now authoritative rather than inferred: remote_pytest.sh
    # prints "remote-pytest: NO RESULT (infra exit=<rc>)" on every path where no
    # suite verdict exists -- pre-run refusals and post-run infrastructure
    # failures alike -- and on no other path. We match that, never the number.
    #
    # FAIL CLOSED, deliberately: a non-zero rc WITHOUT the sentinel stays a red
    # suite and blocks. An old remote_pytest.sh, a truncated log, a lost capture
    # -- all of them land in the blocking branch. A gate that silently passes
    # when it did not run is far worse than one that blocks.
    #
    # WHAT WE DO WITH A NO-RESULT is the policy the user already set on
    # 2026-09-08 for the structurally identical "remote chosen but the router is
    # missing" case, applied to "remote chosen but the router could not acquire
    # a box": re-check the memory floor, and
    #   * at/above it -> run the suite LOCALLY. Coverage is unchanged; this is
    #     routing, not a subset, and the gate still runs in full.
    #   * below it    -> BLOCK, saying plainly that the gate DID NOT RUN and
    #     giving the operator the two commands that fix it. Blocking is still
    #     the safe direction; what changes is that we no longer claim the tests
    #     failed, and we no longer burn ~40 minutes discovering it.
    # -----------------------------------------------------------------------
    if [ "$WINNER" = "remote" ] && [ "$WINNER_RC" != "0" ] \
       && grep -q 'remote-pytest: NO RESULT (infra exit=' "$REMOTE_LOG" 2>/dev/null; then
        echo "[precommit_contracts] the ROUTER could not run the suite (rc=$WINNER_RC)." >&2
        echo "[precommit_contracts] THIS IS NOT A TEST FAILURE -- no suite verdict exists yet." >&2
        if [ -n "$LOCAL_PID" ]; then
            echo "[precommit_contracts] a local race is already in flight -- waiting for it rather than starting another" >&2
            wait "$LOCAL_PID" >/dev/null 2>&1
            if [ -f "$LOCAL_RC" ]; then
                WINNER="local"; WINNER_RC="$(cat "$LOCAL_RC")"; cat "$LOCAL_LOG" >&2
                echo "[precommit_contracts] local race resolved it: rc=$WINNER_RC" >&2
            fi
        else
            NO_RESULT_AVAIL=$(mac_available_mb)
            if [ "${NO_RESULT_AVAIL:-0}" -ge "$FLOOR_MB" ] 2>/dev/null; then
                echo "[precommit_contracts] re-routing to a LOCAL run (mac_available=${NO_RESULT_AVAIL}MB >= ${FLOOR_MB}MB) -- coverage unchanged" >&2
                if run_local_pytest >&2; then
                    WINNER="local"; WINNER_RC=0
                else
                    WINNER="local"; WINNER_RC=1
                fi
            else
                echo "[precommit_contracts] BLOCKING COMMIT: the contract gate DID NOT RUN." >&2
                echo "[precommit_contracts]   The fleet had no box available, and this Mac is below the" >&2
                echo "[precommit_contracts]   memory floor to run it here (mac_available=${NO_RESULT_AVAIL}MB < ${FLOOR_MB}MB)." >&2
                echo "[precommit_contracts]   Nothing is known to be broken -- the tests were never run." >&2
                echo "[precommit_contracts]   remedy 1: wait for a box, then retry the commit --" >&2
                echo "[precommit_contracts]     scripts/remote_pytest.sh tests/contracts -q   (from the main checkout)" >&2
                echo "[precommit_contracts]   remedy 2: free memory on this Mac (close idle sessions) and retry." >&2
                echo "[precommit_contracts]   Do NOT reach for --no-verify: the gate has not cleared this commit." >&2
                if [ "$NO_BLOCK" = "1" ]; then
                    exit 0
                fi
                exit 2
            fi
        fi
    fi

    if [ "$WINNER_RC" = "0" ]; then
        record_validation_cache_result pass
        exit 0
    fi
    record_validation_cache_result fail
else
    if run_local_pytest >&2; then
        record_validation_cache_result pass
        exit 0
    fi
    record_validation_cache_result fail
fi

echo "[precommit_contracts] contract tests failed ($TARGET) -- blocking commit" >&2
echo "[precommit_contracts] fix the failing tests or run with --no-verify to bypass" >&2
if [ "$NO_BLOCK" = "1" ]; then
    exit 0
fi
exit 2
