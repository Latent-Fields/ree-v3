#!/usr/bin/env bash
# precommit_contracts.sh -- run ree-v3/tests/contracts when staged changes
# touch ree_core/**.
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

# Check the staged diff in ree-v3 for any ree_core/** path.
STAGED=$(git -C "$REPO" diff --cached --name-only 2>/dev/null || true)
if ! echo "$STAGED" | grep -q '^ree_core/'; then
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
