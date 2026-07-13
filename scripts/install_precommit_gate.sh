#!/bin/sh
# Install the ree-v3 pre-commit contract gate into ree-v3/.git/hooks so that
# plain-CLI / non-Claude-Code / other-agent commits are gated the same way the Claude
# Code settings.json PreToolUse hook gates Claude Code commits. Closes the pack_writer
# sec 7 item 3 gap: a script committed OUTSIDE /queue-experiment (direct `git commit`,
# another agent) could reintroduce a raw json.dump flat-manifest tail undetected.
#
# The gate logic lives in scripts/git-hooks/pre-commit.local (tracked). This installer
# copies it to .git/hooks/pre-commit.local (the CHAIN SLOT the clinical-hours guard
# already execs) and, if no top-level pre-commit exists, installs a minimal chainer so
# the gate still fires without the clinical guard.
#
# Idempotent: re-running refreshes the hook. Mac / developer machines only -- do NOT run
# on the cloud workers / hub (their phase3 writers must push 24/7 and never author
# experiment scripts).
set -e
ROOT="${REE_WORKING_ROOT:-/Users/dgolden/REE_Working}"
REPO="$ROOT/ree-v3"
HOOKS="$REPO/.git/hooks"
SRC="$REPO/scripts/git-hooks/pre-commit.local"

[ -d "$REPO/.git" ] || { echo "no ree-v3/.git at $REPO -- run from a machine with the ree-v3 checkout" >&2; exit 1; }
[ -f "$SRC" ] || { echo "missing source hook $SRC" >&2; exit 1; }
mkdir -p "$HOOKS"

cp "$SRC" "$HOOKS/pre-commit.local"
chmod +x "$HOOKS/pre-commit.local"
echo "installed $HOOKS/pre-commit.local"

DST="$HOOKS/pre-commit"
if [ ! -e "$DST" ]; then
    # No top-level hook: install a minimal chainer so pre-commit.local runs.
    cat > "$DST" <<'EOF'
#!/bin/sh
# Minimal chainer -> ree-v3 pre-commit contract gate (scripts/install_precommit_gate.sh).
[ -x "$0.local" ] && exec "$0.local" "$@"
exit 0
EOF
    chmod +x "$DST"
    echo "installed minimal top-level pre-commit chainer -> pre-commit.local"
elif grep -q "REE clinical-hours" "$DST" 2>/dev/null; then
    echo "clinical-hours guard is the top-level pre-commit -- it execs pre-commit.local (chained)"
elif grep -q 'exec "\$0.local"' "$DST" 2>/dev/null || grep -q "exec \"\$0.local\"" "$DST" 2>/dev/null; then
    echo "existing top-level pre-commit chains to pre-commit.local -- gate active"
else
    echo "WARNING: an existing top-level pre-commit is present and does NOT chain to \$0.local;" >&2
    echo "         the contract gate at pre-commit.local will not run until it does." >&2
    echo "         reconcile by having that hook end with:  [ -x \"\$0.local\" ] && exec \"\$0.local\" \"\$@\"" >&2
fi
echo "done -- ree-v3 pre-commit contract gate active on this machine"
