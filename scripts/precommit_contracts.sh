#!/usr/bin/env bash
# precommit_contracts.sh -- run ree-v3/tests/contracts when staged changes
# touch ree_core/**, AND run validate_experiments.py --strict on staged
# experiments/v3_exq_*.py paths. Both checks self-gate: if no relevant
# paths are staged, the corresponding block is a no-op.
#
# Called from the PreToolUse hook in REE_Working/.claude/settings.json on
# any `git commit` bash invocation. Self-gates: if no ree_core/** paths are
# staged in the ree-v3 repo, this script exits 0 with no output so commits
# to REE_assembly / other repos aren't penalised.
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

# Resolve ree-v3 repo root. Prefer CLAUDE_PROJECT_DIR (set by the harness
# when the hook runs) but fall back to this script's own location for
# manual invocation.
if [ -n "${CLAUDE_PROJECT_DIR:-}" ] && [ -d "$CLAUDE_PROJECT_DIR/ree-v3" ]; then
    REPO="$CLAUDE_PROJECT_DIR/ree-v3"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

if [ ! -d "$REPO/ree_core" ] || [ ! -d "$REPO/tests/contracts" ]; then
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

# Block 2: ree_core/** -> contracts test suite.
if ! echo "$STAGED" | grep -q '^ree_core/'; then
    exit 0
fi

echo "[precommit_contracts] ree_core/ change staged -- running contracts" >&2
if (cd "$REPO" && "$PY" -m pytest -q --tb=line tests/contracts) >&2; then
    exit 0
fi

echo "[precommit_contracts] contract tests failed -- blocking commit" >&2
echo "[precommit_contracts] fix the failing tests or run with --no-verify to bypass" >&2
if [ "$NO_BLOCK" = "1" ]; then
    exit 0
fi
exit 2
