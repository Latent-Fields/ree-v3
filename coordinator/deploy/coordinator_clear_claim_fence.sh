#!/bin/bash
# coordinator_clear_claim_fence.sh -- POST /claim_fence/clear on the local
# coordinator. The symmetric disarm for coordinator_announce_shutdown.sh:
# the announce ARMS the drain-window claim fence, this CLEARS it.
#
# Called by the scaler immediately after `hcloud server poweron` (see
# coordinator/deploy/cloud-scaler.py). Without it a worker woken shortly
# after its own shutdown would sit out the remainder of that shutdown's
# fence window (COORDINATOR_CLAIM_FENCE_SECONDS, default 1800) unable to
# claim -- the fence would cost real capacity in exactly the flapping case
# where the queue has work again.
#
# The fence also expires on its own, so a machine woken by a MANUAL
# `hcloud server poweron` (documented in CLAUDE.md) recovers without this
# script. That expiry is the backstop; this is the fast path.
#
# Install on the coordinator hub (ree-cloud-1):
#   sudo cp coordinator_clear_claim_fence.sh /usr/local/bin/
#   sudo chmod 755 /usr/local/bin/coordinator_clear_claim_fence.sh
#
# Inputs:
#   $1  -- machine affinity label (required; falls back to $AFF env var
#          for local testing).
#
# Reads coordinator URL + token from /etc/ree-coordinator.env, same as the
# announce helper. Token NEVER appears in logs.
#
# Best-effort: any failure prints to stderr and exits non-zero. The caller
# ignores the exit code -- a failed clear only means the fence expires on
# its own timer instead of immediately, which is degraded, not broken.
#
# All output is ASCII-only.

set -eu

CONF=${COORDINATOR_ENV:-/etc/ree-coordinator.env}
if [ ! -r "$CONF" ]; then
    echo "clear_claim_fence: $CONF not readable" >&2
    exit 2
fi

# shellcheck disable=SC1090
. "$CONF"

if [ -z "${COORDINATOR_URL:-}" ]; then
    echo "clear_claim_fence: COORDINATOR_URL not set in $CONF" >&2
    exit 2
fi

TOKEN="${COORDINATOR_SCALER_TOKEN:-${COORDINATOR_LOCAL_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
    echo "clear_claim_fence: no token in $CONF (need COORDINATOR_SCALER_TOKEN or COORDINATOR_LOCAL_TOKEN)" >&2
    exit 2
fi

AFF=${1:-${AFF:-}}
if [ -z "$AFF" ]; then
    echo "clear_claim_fence: machine affinity required (argv[1] or \$AFF)" >&2
    exit 2
fi
case "$AFF" in
    *[!A-Za-z0-9._-]*)
        echo "clear_claim_fence: invalid affinity '$AFF' (alphanumeric + . _ - only)" >&2
        exit 2
        ;;
esac

json_escape() {
    python3 -c '
import json, sys
sys.stdout.write(json.dumps(sys.argv[1]))
' "$1"
}

AFF_J=$(json_escape "$AFF")
PAYLOAD="{\"machine\":${AFF_J}}"

curl -fsS \
     --connect-timeout 5 --max-time 5 \
     -X POST "${COORDINATOR_URL}/claim_fence/clear" \
     -H "Authorization: Bearer ${TOKEN}" \
     -H "Content-Type: application/json" \
     -d "$PAYLOAD" >/dev/null
echo "clear_claim_fence: posted machine=$AFF"
