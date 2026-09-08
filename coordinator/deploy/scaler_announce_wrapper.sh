#!/bin/bash
# Forced-command wrapper. Accepts only:
#   /usr/local/bin/coordinator_announce_shutdown.sh <affinity>
# Anything else exits non-zero.
set -eu
cmd=${SSH_ORIGINAL_COMMAND:-}
case "$cmd" in
  "/usr/local/bin/coordinator_announce_shutdown.sh '"*"'")
    aff=${cmd#"/usr/local/bin/coordinator_announce_shutdown.sh '"}
    aff=${aff%"'"}
    case "$aff" in
      *[!A-Za-z0-9._-]*) echo "bad affinity" >&2; exit 2 ;;
    esac
    exec /usr/local/bin/coordinator_announce_shutdown.sh "$aff" ;;
  *) echo "unauthorized command: $cmd" >&2; exit 1 ;;
esac
