#!/usr/bin/env python3
"""Generate a dispatch bearer token and write dispatch_tokens.json.

Usage:
  python3 deploy/gen_token.py [label]      # default label: "phone"
Prints the token to stdout (copy it into the phone page + the executor plist).
Append more labels by re-running; existing tokens are preserved.
"""
import json
import os
import secrets
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENS_FILE = os.path.join(HERE, "dispatch_tokens.json")


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "phone"
    tokens = {}
    if os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, "r", encoding="utf-8") as fh:
            tokens = json.load(fh)
    tok = secrets.token_urlsafe(32)
    tokens[tok] = label
    with open(TOKENS_FILE, "w", encoding="utf-8") as fh:
        json.dump(tokens, fh, indent=2)
    os.chmod(TOKENS_FILE, 0o600)
    sys.stdout.write("token for label '%s':\n%s\n" % (label, tok))
    sys.stdout.write("written to %s (chmod 600)\n" % TOKENS_FILE)


if __name__ == "__main__":
    main()
