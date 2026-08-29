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
  else  $DISPATCH_CLIENT_CONFIG, else <this-dir>/../.dispatch_client.json
        ({"url": ..., "token": ...})
Both are git-ignored; the token never enters settings.json or git.

DISPATCH_CLIENT_CONFIG exists so tests can point the lookup at a path that does
not exist. The on-disk config is found relative to THIS FILE, so a test that
merely clears DISPATCH_URL/TOKEN still finds it and posts to the real queue
(that leak put two junk jobs in the live DB before it was fixed, 2026-07-18).
"""
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_CONFIG = (os.environ.get("DISPATCH_CLIENT_CONFIG", "").strip()
                 or os.path.join(HERE, "..", ".dispatch_client.json"))
SPOOL_PATH = (os.environ.get("DISPATCH_SPOOL", "").strip()
              or os.path.join(HERE, "..", "dispatch_spool.jsonl"))


def _spool(record):
    """Persist a chip we could not deliver, for the service to drain on boot.

    Fail-open is right for a PostToolUse hook -- it must never disrupt a chip
    that has already spawned -- but fail-open plus no trace means a wedged
    service silently swallows every chip, which is what happened for four hours
    on 2026-08-29 (181 dropped POSTs, noticed only by reading the error log).
    Spooling keeps fail-open while making the loss recoverable.
    """
    try:
        with open(SPOOL_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[mirror_chip] spool failed: %s\n" % exc)


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

    record = {
        "title": (ti.get("title") or "").strip(),
        "cwd": (ti.get("cwd") or "").strip(),
        "prompt": prompt,
        "source": "chip",
        "status": "staged",   # awaiting a tap on the phone
    }
    req = urllib.request.Request(url.rstrip("/") + "/api/enqueue",
                                 data=json.dumps(record).encode("utf-8"),
                                 method="POST")
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req, timeout=4).read()
    except Exception as exc:  # noqa: BLE001  fire-and-forget
        sys.stderr.write("[mirror_chip] enqueue failed (spooled): %s\n" % exc)
        _spool(record)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.exit(0)  # ALWAYS fail-open
