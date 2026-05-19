"""Mint a per-worker bearer token and merge it into tokens.json.

One token per machine so a single worker can be revoked (delete its line
from tokens.json + restart the coordinator) without rotating everyone.

Usage:
  python3 gen_token.py <machine-label> [--tokens PATH]

ASCII-only output.
"""

import argparse
import json
import os
import secrets
import sys

DEFAULT_TOKENS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tokens.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("machine", help="e.g. DLAPTOP-4.local, Daniel-PC")
    ap.add_argument("--tokens", default=DEFAULT_TOKENS)
    args = ap.parse_args()

    tokens = {}
    if os.path.exists(args.tokens):
        with open(args.tokens, "r", encoding="utf-8") as fh:
            tokens = json.load(fh)

    # Drop any existing token for this machine (rotation).
    tokens = {t: m for t, m in tokens.items() if m != args.machine}

    new_tok = secrets.token_urlsafe(32)
    tokens[new_tok] = args.machine

    tmp = args.tokens + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(tokens, fh, indent=2)
    os.replace(tmp, args.tokens)
    try:
        os.chmod(args.tokens, 0o600)
    except OSError:
        pass

    sys.stdout.write(
        "\nToken minted for %s (tokens.json now has %d worker(s)).\n"
        "Restart ree-coordinator for it to take effect.\n\n"
        "Put these in that worker's runner environment:\n"
        "  COORDINATION_MODE=shadow\n"
        "  COORDINATOR_URL=http://10.8.0.1:8787\n"
        "  COORDINATOR_TOKEN=%s\n\n"
        % (args.machine, len(tokens), new_tok))
    return 0


if __name__ == "__main__":
    sys.exit(main())
