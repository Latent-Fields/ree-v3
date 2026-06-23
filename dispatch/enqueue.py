#!/usr/bin/env python3
"""Enqueue a dispatch job into the hub queue.

Primary use: MIRROR a spawn_task chip into the durable phone-reachable queue so
it survives app restarts and can be launched from the phone. A chip's prompt is
already self-contained, which is exactly what a dispatched `claude -p` needs.

Default status is `staged` (a suggestion awaiting your tap on the phone), matching
chip semantics. Use --launch to enqueue straight to `pending` (auto-runnable).

Usage:
  DISPATCH_URL=http://10.8.0.1:8799 DISPATCH_TOKEN=... \\
    python3 enqueue.py --title "Queue sleep GAP-3b run" \\
      --cwd /Users/dgolden/REE_Working/REE_assembly \\
      --prompt "In REE_assembly, via /queue-experiment ..."

  # prompt from stdin:
  cat chip.txt | python3 enqueue.py --title "..." --stdin
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser(description="Enqueue a REE dispatch job.")
    ap.add_argument("--title", default="")
    ap.add_argument("--cwd", default="")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--stdin", action="store_true",
                    help="read the prompt from stdin")
    ap.add_argument("--launch", action="store_true",
                    help="enqueue as pending (auto-runnable) not staged")
    ap.add_argument("--source", default="chip")
    ap.add_argument("--url", default=os.environ.get("DISPATCH_URL", ""))
    ap.add_argument("--token", default=os.environ.get("DISPATCH_TOKEN", ""))
    args = ap.parse_args()

    prompt = sys.stdin.read() if args.stdin else args.prompt
    prompt = (prompt or "").strip()
    if not prompt:
        sys.stderr.write("error: prompt required (--prompt or --stdin)\n")
        sys.exit(2)
    if not args.url or not args.token:
        sys.stderr.write("error: --url/--token (or DISPATCH_URL/DISPATCH_TOKEN) required\n")
        sys.exit(2)

    body = {
        "title": args.title.strip(),
        "cwd": args.cwd.strip(),
        "prompt": prompt,
        "source": args.source,
        "status": "pending" if args.launch else "staged",
    }
    req = urllib.request.Request(
        args.url.rstrip("/") + "/api/enqueue",
        data=json.dumps(body).encode("utf-8"), method="POST")
    req.add_header("Authorization", "Bearer " + args.token)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        sys.stderr.write("error: HTTP %s %s\n" % (exc.code, exc.read().decode("utf-8", "replace")))
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("error: %s\n" % exc)
        sys.exit(1)
    sys.stdout.write("enqueued id=%s status=%s\n" % (out.get("id"), out.get("status")))


if __name__ == "__main__":
    main()
