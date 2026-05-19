"""Daily go/no-go instrument for the Phase-1 shadow soak.

Run this from any machine on the WireGuard mesh (e.g. your Mac) during
the shadow period. It hits /health + /shadow/status and prints a single
verdict so you do not have to eyeball raw JSON.

  python3 check_shadow.py \
      --url http://10.8.0.1:8787 --token <any-worker-token>

  (or set COORDINATOR_URL / COORDINATOR_TOKEN in the environment)

Exit codes (scriptable):
  0  HEALTHY -- soak running, zero divergence. Keep soaking; this is the
     state that, sustained over days of real load, clears Phase 2.
  1  DIVERGENCE -- the coordinator's claim verdict disagreed with git at
     least once. DO NOT advance. Investigate every row first.
  2  NO SIGNAL -- coordinator reachable but no claim traffic yet and/or
     all heartbeats stale. The soak is not actually exercising anything
     (runners drained or not flipped to COORDINATION_MODE=shadow).
  3  UNREACHABLE -- coordinator down / WireGuard / token problem.

All output is ASCII-only.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _get(url, token, path):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        headers={"Authorization": "Bearer " + token} if token else {},
        method="GET")
    with urllib.request.urlopen(req, timeout=8) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _age_secs(iso):
    if not iso:
        return None
    try:
        ts = iso.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("COORDINATOR_URL", ""))
    ap.add_argument("--token",
                    default=os.environ.get("COORDINATOR_TOKEN", ""))
    ap.add_argument("--stale-mins", type=float, default=10.0,
                    help="a heartbeat older than this is STALE")
    args = ap.parse_args()

    if not args.url:
        sys.stderr.write("need --url or COORDINATOR_URL\n")
        return 3

    try:
        _, health = _get(args.url, None, "/health")
    except (urllib.error.URLError, OSError) as exc:
        print("UNREACHABLE: %s (%r)" % (args.url, exc))
        print("  -> check WireGuard is up and ree-cloud-1 coordinator "
              "is running")
        return 3

    try:
        _, st = _get(args.url, args.token, "/shadow/status")
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            print("UNREACHABLE: 401 -- bad/!missing bearer token")
            return 3
        print("UNREACHABLE: HTTP %s on /shadow/status" % exc.code)
        return 3
    except (urllib.error.URLError, OSError) as exc:
        print("UNREACHABLE: %r" % exc)
        return 3

    mode = st.get("mode")
    total = st.get("total_claims", 0)
    ndiv = st.get("divergences", 0)
    nexp = st.get("experiments_in_mirror", 0)
    machines = st.get("machines", [])
    stale_cut = args.stale_mins * 60.0

    print("coordinator %s  mode=%s  queue-mirror=%d items"
          % (args.url, mode, nexp))
    print("claims observed: %d   divergences: %d" % (total, ndiv))
    print("machines (heartbeat freshness):")
    fresh = 0
    if not machines:
        print("  (none reporting)")
    for m in machines:
        age = _age_secs(m.get("last_seen"))
        if age is None:
            tag, a = "NO-TS", "?"
        elif age <= stale_cut:
            tag, a = "FRESH", "%ds" % int(age)
            fresh += 1
        else:
            tag, a = "STALE", "%dm" % int(age / 60)
        print("  [%s] %-18s last=%s state=%s exq=%s"
              % (tag, m.get("machine"), a, m.get("state"),
                 m.get("current_exq")))

    if ndiv > 0:
        print("")
        print("VERDICT: DIVERGENCE DETECTED (%d) -- DO NOT ADVANCE TO "
              "PHASE 2" % ndiv)
        for r in st.get("recent_divergences", [])[:20]:
            print("  %s machine=%s git=%s coord=%s @ %s"
                  % (r.get("queue_id"), r.get("machine"),
                     r.get("git_verdict"), r.get("coord_verdict"),
                     r.get("logged_at")))
        print("  Investigate each: a real logic mismatch blocks cutover; "
              "an explainable stale-pull race should be rare and "
              "self-consistent.")
        return 1

    if total == 0 or fresh == 0:
        print("")
        print("VERDICT: NO SIGNAL -- coordinator healthy but the soak is "
              "not exercising it")
        print("  claims_seen=%d fresh_heartbeats=%d. Bring runners back "
              "up with COORDINATION_MODE=shadow so real claim traffic "
              "reaches the coordinator." % (total, fresh))
        return 2

    print("")
    print("VERDICT: HEALTHY -- %d claims, 0 divergence, %d machine(s) "
          "live. Keep soaking; sustained over several days of real "
          "multi-machine load this clears Phase 2." % (total, fresh))
    return 0


if __name__ == "__main__":
    sys.exit(main())
