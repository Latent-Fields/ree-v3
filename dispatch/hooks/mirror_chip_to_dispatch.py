#!/usr/bin/env python3
"""PostToolUse hook: mirror a spawn_task chip into the phone-dispatch queue.

Wired in .claude/settings.json as a PostToolUse hook matching
`mcp__ccd_session__spawn_task`. When Claude spawns a chip, this reads the chip's
self-contained prompt from the hook's stdin (tool_input) and POSTs it to the
dispatch service as a `staged` job, so it shows up on the phone page and can be
Launched from the iPhone.

FAIL-OPEN BY DESIGN: this ALWAYS exits 0 and never raises. If the dispatch
service is undeployed/unreachable, or the client config is missing, it is a
silent no-op -- a spawned chip is never disrupted (PostToolUse runs after the
tool has already executed).

Config (first match wins):
  env DISPATCH_URL + DISPATCH_TOKEN
  else  <this-dir>/../.dispatch_client.json  ({"url": ..., "token": ...})
Both are git-ignored; the token never enters settings.json or git.
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_CONFIG = os.path.join(HERE, "..", ".dispatch_client.json")


def _config():
    url = os.environ.get("DISPATCH_URL", "").strip()
    token = os.environ.get("DISPATCH_TOKEN", "").strip()
    if url and token:
        return url, token
    try:
        with open(CLIENT_CONFIG, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        return (cfg.get("url", "").strip(), cfg.get("token", "").strip())
    except Exception:  # noqa: BLE001
        return "", ""


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:  # noqa: BLE001
        return  # nothing parseable -> no-op
    if not isinstance(payload, dict):
        return
    ti = payload.get("tool_input") or {}
    prompt = (ti.get("prompt") or "").strip()
    if not prompt:
        return
    url, token = _config()
    if not url or not token:
        return  # not configured / service not deployed -> silent no-op

    body = json.dumps({
        "title": (ti.get("title") or "").strip(),
        "cwd": (ti.get("cwd") or "").strip(),
        "prompt": prompt,
        "source": "chip",
        "status": "staged",   # awaiting a tap on the phone
    }).encode("utf-8")
    req = urllib.request.Request(url.rstrip("/") + "/api/enqueue",
                                 data=body, method="POST")
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req, timeout=4).read()
    except Exception as exc:  # noqa: BLE001  fire-and-forget
        sys.stderr.write("[mirror_chip] enqueue failed (non-fatal): %s\n" % exc)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.exit(0)  # ALWAYS fail-open
