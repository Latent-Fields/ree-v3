#!/bin/bash
# coordinator_announce_shutdown.sh -- POST /shutdown_notify on the local
# coordinator. Intended to be invoked by the cloud-scaler GitHub Actions
# workflow over SSH (ree-v3/.github/workflows/cloud-scaler.yml).
#
# Install on the coordinator hub (ree-cloud-1):
#   sudo cp coordinator_announce_shutdown.sh /usr/local/bin/
#   sudo chmod 755 /usr/local/bin/coordinator_announce_shutdown.sh
#
# Lock the SSH key down (recommended): in ~ree/.ssh/authorized_keys for
# the scaler key, prepend
#   command="/usr/local/bin/coordinator_announce_shutdown.sh",no-pty,no-agent-forwarding,no-X11-forwarding,no-port-forwarding
# so a compromised GH secret can ONLY trigger announces, not arbitrary
# shell access.
#
# Inputs:
#   $1                           -- machine affinity label (required;
#                                   falls back to $AFF env var for local
#                                   testing without going through SSH).
#   $REASON                      -- optional, default "scaler_idle_after_grace"
#   $EXPECTED_WAKE_CONDITION     -- optional, default "claimable>0"
#
# AFF as $1: under SSH forced-command the wrapper script is responsible
# for extracting safe args from $SSH_ORIGINAL_COMMAND; without forced-
# command, the GH workflow passes $AFFINITY positionally. Either way the
# helper sees the machine label as $1.
#
# Reads coordinator URL + token from /etc/ree-coordinator.env (the same
# file the systemd unit consumes). Token NEVER appears in workflow logs.
#
# Best-effort: any failure prints to stderr and exits non-zero, but the
# caller (cloud-scaler.yml) ignores the exit code so the underlying
# hcloud shutdown still fires.
#
# All output is ASCII-only.

set -eu

CONF=${COORDINATOR_ENV:-/etc/ree-coordinator.env}
if [ ! -r "$CONF" ]; then
    echo "announce_shutdown: $CONF not readable" >&2
    exit 2
fi

# shellcheck disable=SC1090
. "$CONF"

if [ -z "${COORDINATOR_URL:-}" ]; then
    echo "announce_shutdown: COORDINATOR_URL not set in $CONF" >&2
    exit 2
fi

# Token name: COORDINATOR_SCALER_TOKEN preferred (dedicated scaler token,
# revocable independently). Falls back to the hub's local token so a
# minimal install (no extra token provisioned) still works.
TOKEN="${COORDINATOR_SCALER_TOKEN:-${COORDINATOR_LOCAL_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
    echo "announce_shutdown: no token in $CONF (need COORDINATOR_SCALER_TOKEN or COORDINATOR_LOCAL_TOKEN)" >&2
    exit 2
fi

# Accept AFF as the first positional arg; fall back to the legacy AFF
# env var (still used by local smoke tests that source the helper via
# bash without going through SSH). Validate strictly -- this string
# ends up in a JSON payload AND in an SSH-forced-command audit log, so
# a stray quote/backslash would be ugly.
AFF=${1:-${AFF:-}}
if [ -z "$AFF" ]; then
    echo "announce_shutdown: machine affinity required (argv[1] or \$AFF)" >&2
    exit 2
fi
case "$AFF" in
    *[!A-Za-z0-9._-]*)
        echo "announce_shutdown: invalid affinity '$AFF' (alphanumeric + . _ - only)" >&2
        exit 2
        ;;
esac

REASON_VALUE=${REASON:-scaler_idle_after_grace}
WAKE_VALUE=${EXPECTED_WAKE_CONDITION:-claimable>0}

# JSON-encode the three free-text fields defensively. None of them
# currently carry user input -- AFF comes from the workflow's WORKERS
# array, REASON/EXPECTED_WAKE_CONDITION are hard-coded defaults -- but
# defending against quotes/backslashes here costs nothing and stops a
# future caller from injecting via a stray apostrophe.
json_escape() {
    python3 -c '
import json, sys
sys.stdout.write(json.dumps(sys.argv[1]))
' "$1"
}

AFF_J=$(json_escape "$AFF")
REASON_J=$(json_escape "$REASON_VALUE")
WAKE_J=$(json_escape "$WAKE_VALUE")

PAYLOAD="{\"machine\":${AFF_J},\"reason\":${REASON_J},\"expected_wake_condition\":${WAKE_J}}"

# 5-second connect + 5-second total timeout: this is a localhost call,
# anything slower means the coordinator is wedged and we should not
# block the scaler tick.
curl -fsS \
     --connect-timeout 5 --max-time 5 \
     -X POST "${COORDINATOR_URL}/shutdown_notify" \
     -H "Authorization: Bearer ${TOKEN}" \
     -H "Content-Type: application/json" \
     -d "$PAYLOAD" >/dev/null
echo "announce_shutdown: posted machine=$AFF reason=$REASON_VALUE"
