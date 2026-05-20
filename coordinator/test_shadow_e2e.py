"""Local end-to-end self-test for the shadow coordinator.

No WireGuard, no network config, no third-party deps. Spins the real
coordinator entrypoint as a subprocess against a throwaway DB + tokens
file, seeds a tiny queue via seed_from_queue.py, and asserts:

  1. /health is unauthenticated and 200
  2. a bad bearer token is rejected 401
  3. a consistent git/coord claim sequence logs ZERO divergence
  4. an inconsistent (injected) claim logs exactly ONE divergence
  5. /result is idempotent on run_id (replay = no-op)
  6. db.try_claim is a real atomic mutex under concurrent threads
     (exactly one 'ok' when N threads race the same queue_id)
  7. sync_daemon --once reconciles the mirror from the queue file

Exit code 0 = PASS. ASCII-only output.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import db  # noqa: E402
import experiment_runner  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"
_failures = []


def check(name, cond):
    mark = PASS if cond else FAIL
    print("  [%s] %s" % (mark, name))
    if not cond:
        _failures.append(name)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http(method, url, token=None, body=None):
    headers = {}
    data = None
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, None


def main():
    tmp = tempfile.mkdtemp(prefix="coord_e2e_")
    db_path = os.path.join(tmp, "c.db")
    tokens_path = os.path.join(tmp, "tokens.json")
    queue_path = os.path.join(tmp, "experiment_queue.json")
    port = free_port()

    with open(tokens_path, "w", encoding="utf-8") as fh:
        json.dump({"good-token": "DLAPTOP-4.local"}, fh)

    with open(queue_path, "w", encoding="utf-8") as fh:
        json.dump({
            "schema_version": "v1",
            "calibration": {},
            "items": [
                {"queue_id": "V3-EXQ-901", "script": "experiments/a.py",
                 "priority": 5, "machine_affinity": "any",
                 "status": "pending", "estimated_minutes": 10},
                {"queue_id": "V3-EXQ-902", "script": "experiments/b.py",
                 "priority": 4, "machine_affinity": "any",
                 "status": "pending", "estimated_minutes": 10},
            ]}, fh)

    env = dict(os.environ)
    env.update({
        "COORDINATOR_DB": db_path,
        "COORDINATOR_TOKENS_FILE": tokens_path,
        "COORDINATOR_BIND_HOST": "127.0.0.1",
        "COORDINATOR_BIND_PORT": str(port),
        "COORDINATOR_MODE": "shadow",
    })

    # seed
    seed = subprocess.run(
        [sys.executable, os.path.join(HERE, "seed_from_queue.py"),
         "--queue", queue_path, "--db", db_path],
        capture_output=True, text=True)
    check("seed_from_queue ran", seed.returncode == 0)

    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "app.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    base = "http://127.0.0.1:%d" % port
    try:
        # wait for liveness
        up = False
        for _ in range(50):
            try:
                st, _ = http("GET", base + "/health")
                if st == 200:
                    up = True
                    break
            except urllib.error.URLError:
                time.sleep(0.1)
        check("server came up", up)

        st, jb = http("GET", base + "/health")
        check("/health 200 unauthenticated", st == 200 and jb["ok"])

        st, _ = http("GET", base + "/shadow/divergence",
                     token="WRONG-token")
        check("bad token -> 401", st == 401)

        # consistent sequence: A claims 901 (git ok), B attempts 901
        # (git already_claimed). Coordinator must agree both times.
        st, jb = http("POST", base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-901",
                            "machine": "DLAPTOP-4.local",
                            "git_verdict": "ok"})
        check("A claim 901 -> coord ok, no divergence",
              st == 200 and jb["verdict"] == "ok"
              and jb["diverged"] is False)

        # E2 harness: same machine re-reports git ok after mirror already
        # holds the claim (sync_daemon upsert or prior /claim).
        st, jb = http("POST", base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-901",
                            "machine": "DLAPTOP-4.local",
                            "git_verdict": "ok"})
        check("E2 idempotent same-machine re-report -> no divergence",
              st == 200 and jb["verdict"] == "already_claimed"
              and jb["diverged"] is False)

        st, jb = http("POST", base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-901", "machine": "Daniel-PC",
                            "git_verdict": "already_claimed"})
        check("B claim 901 -> coord already_claimed, no divergence",
              st == 200 and jb["verdict"] == "already_claimed"
              and jb["diverged"] is False)

        # injected divergence: 902 is pending in the mirror so the
        # coordinator would say 'ok', but we tell it git said
        # 'already_claimed'. That MUST be flagged.
        st, jb = http("POST", base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-902", "machine": "Daniel-PC",
                            "git_verdict": "already_claimed"})
        check("injected mismatch flagged diverged",
              st == 200 and jb["diverged"] is True)

        st, jb = http("GET", base + "/shadow/divergence",
                      token="good-token")
        check("/shadow/divergence reports exactly 1",
              st == 200 and jb["divergences"] == 1)

        # result idempotency
        man = json.dumps({"run_id": "r_901_v3", "queue_id": "V3-EXQ-901",
                          "outcome": "PASS"}).encode("utf-8")
        import gzip
        req = urllib.request.Request(
            base + "/result", data=gzip.compress(man),
            headers={"Authorization": "Bearer good-token",
                     "Content-Type": "application/json",
                     "Content-Encoding": "gzip"}, method="POST")
        with urllib.request.urlopen(req, timeout=5) as r:
            jb1 = json.loads(r.read().decode("utf-8"))
        with urllib.request.urlopen(req, timeout=5) as r:
            jb2 = json.loads(r.read().decode("utf-8"))
        check("first /result recorded",
              jb1["ok"] and jb1["idempotent_noop"] is False)
        check("replayed /result is idempotent no-op",
              jb2["ok"] and jb2["idempotent_noop"] is True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ---- coordinator-mode HTTP claim behaviour (Phase 2) --------------
    coord_db = os.path.join(tmp, "coord.db")
    coord_port = free_port()
    seed = subprocess.run(
        [sys.executable, os.path.join(HERE, "seed_from_queue.py"),
         "--queue", queue_path, "--db", coord_db],
        capture_output=True, text=True)
    check("coordinator seed_from_queue ran", seed.returncode == 0)

    coord_env = dict(env)
    coord_env.update({
        "COORDINATOR_DB": coord_db,
        "COORDINATOR_BIND_PORT": str(coord_port),
        "COORDINATOR_MODE": "coordinator",
    })
    proc = subprocess.Popen(
        [sys.executable, os.path.join(HERE, "app.py")],
        env=coord_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    coord_base = "http://127.0.0.1:%d" % coord_port
    try:
        up = False
        for _ in range(50):
            try:
                st, _ = http("GET", coord_base + "/health")
                if st == 200:
                    up = True
                    break
            except urllib.error.URLError:
                time.sleep(0.1)
        check("coordinator-mode server came up", up)

        st, jb = http("POST", coord_base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-901",
                            "machine": "DLAPTOP-4.local"})
        check("coordinator claim 901 -> ok authoritative",
              st == 200 and jb["verdict"] == "ok"
              and jb["authoritative"] is True)

        st, jb = http("POST", coord_base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-901",
                            "machine": "Daniel-PC"})
        check("coordinator second claim 901 -> already_claimed",
              st == 200 and jb["verdict"] == "already_claimed")

        st, jb = http("POST", coord_base + "/claim/release",
                      token="good-token",
                      body={"queue_id": "V3-EXQ-901",
                            "machine": "DLAPTOP-4.local"})
        check("coordinator release by owner -> ok",
              st == 200 and jb["ok"] is True and jb["applied"] is True)

        st, jb = http("POST", coord_base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-901",
                            "machine": "Daniel-PC"})
        check("coordinator claim after release -> ok",
              st == 200 and jb["verdict"] == "ok")

        st, jb = http("POST", coord_base + "/queue/remove",
                      token="good-token",
                      body={"queue_id": "V3-EXQ-901", "reason": "PASS"})
        check("coordinator queue/remove applies",
              st == 200 and jb["ok"] is True and jb["applied"] is True)

        st, jb = http("POST", coord_base + "/claim", token="good-token",
                      body={"queue_id": "V3-EXQ-901",
                            "machine": "DLAPTOP-4.local"})
        check("completed queue item is not claimable",
              st == 200 and jb["verdict"] == "already_claimed")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # ---- atomic-mutex property at the db layer (Phase 2 correctness) ---
    race_db = os.path.join(tmp, "race.db")
    db.init_db(race_db)
    seed_conn = db.connect(race_db)
    db.upsert_experiment(seed_conn, {
        "queue_id": "V3-EXQ-RACE", "script": "x", "priority": 1,
        "machine_affinity": "any", "status": "pending",
        "estimated_minutes": 1})
    seed_conn.close()

    results = []
    rlock = threading.Lock()

    def racer(idx):
        c = db.connect(race_db)
        try:
            v = db.try_claim(c, "V3-EXQ-RACE", "m%d" % idx)
        finally:
            c.close()
        with rlock:
            results.append(v)

    threads = [threading.Thread(target=racer, args=(i,))
               for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    oks = results.count("ok")
    check("concurrent try_claim: exactly one winner (got %d)" % oks,
          oks == 1)

    # ---- stale-claim lease is extended by a fresh owner heartbeat -----
    lease_db = os.path.join(tmp, "lease.db")
    db.init_db(lease_db)
    lc = db.connect(lease_db)
    old_claim = "2000-01-01T00:00:00Z"
    try:
        db.upsert_experiment(lc, {
            "queue_id": "V3-EXQ-LEASE-NO-HB", "script": "x",
            "priority": 1, "machine_affinity": "any",
            "status": "claimed", "estimated_minutes": 1,
            "claimed_by": {"machine": "Mac", "claimed_at": old_claim}})
        check("old claim without heartbeat is recoverable",
              db.try_claim(lc, "V3-EXQ-LEASE-NO-HB", "Daniel-PC") == "ok")

        db.upsert_experiment(lc, {
            "queue_id": "V3-EXQ-LEASE-HB", "script": "x",
            "priority": 1, "machine_affinity": "any",
            "status": "claimed", "estimated_minutes": 1,
            "claimed_by": {"machine": "Mac", "claimed_at": old_claim}})
        db.upsert_heartbeat(lc, "Mac", "running", "V3-EXQ-LEASE-HB",
                            {"overall_pct": 50.0}, {"available": False})
        check("fresh owner heartbeat blocks stale recovery",
              db.try_claim(lc, "V3-EXQ-LEASE-HB", "Daniel-PC")
              == "already_claimed")
        check("evaluate_claim also honors fresh owner heartbeat",
              db.evaluate_claim(lc, "V3-EXQ-LEASE-HB", "Daniel-PC")
              == "already_claimed")
    finally:
        lc.close()

    assembly = os.path.join(tmp, "REE_assembly")
    hb_dir = os.path.join(
        assembly, "evidence", "experiments", "runner_heartbeats")
    os.makedirs(hb_dir, exist_ok=True)
    hb_path = os.path.join(hb_dir, "ree-cloud-1.json")
    with open(hb_path, "w", encoding="utf-8") as fh:
        json.dump({"machine": "ree-cloud-1",
                   "last_tick_utc": experiment_runner.now_utc(),
                   "state": "running",
                   "current_exq": "V3-EXQ-LONG"}, fh)
    runner_claim = {"machine": "ree-cloud-1", "claimed_at": old_claim}
    check("runner stale check honors fresh owner heartbeat",
          experiment_runner._is_stale_claim(
              runner_claim, "V3-EXQ-LONG", Path(assembly)) is False)
    check("runner stale check rejects heartbeat for a different queue id",
          experiment_runner._is_stale_claim(
              runner_claim, "V3-EXQ-OTHER", Path(assembly)) is True)

    # ---- sync_daemon --once reconciles mirror from the queue file -----
    sd = subprocess.run(
        [sys.executable, os.path.join(HERE, "sync_daemon.py"),
         "--queue", queue_path, "--db", db_path, "--once"],
        capture_output=True, text=True)
    check("sync_daemon --once exited 0", sd.returncode == 0)
    check("sync_daemon reconciled output present",
          "reconciled" in sd.stdout)

    # In Phase 2, git remains a worklist but DB claims are authoritative.
    # A sync against a git/local queue that still says "pending" must not
    # overwrite an existing coordinator claim back to pending.
    preserve_db = os.path.join(tmp, "preserve.db")
    db.init_db(preserve_db)
    pc = db.connect(preserve_db)
    try:
        db.upsert_experiment(pc, {
            "queue_id": "V3-EXQ-PRESERVE", "script": "x",
            "priority": 1, "machine_affinity": "any",
            "status": "pending", "estimated_minutes": 1})
        check("preserve setup claim -> ok",
              db.try_claim(pc, "V3-EXQ-PRESERVE", "Mac") == "ok")
    finally:
        pc.close()
    preserve_queue = os.path.join(tmp, "preserve_queue.json")
    with open(preserve_queue, "w", encoding="utf-8") as fh:
        json.dump({"items": [
            {"queue_id": "V3-EXQ-PRESERVE", "script": "x",
             "priority": 1, "machine_affinity": "any",
             "status": "pending", "estimated_minutes": 1}
        ]}, fh)
    sd_env = dict(os.environ)
    sd_env["SYNC_MODE"] = "coordinator"
    sd = subprocess.run(
        [sys.executable, os.path.join(HERE, "sync_daemon.py"),
         "--queue", preserve_queue, "--db", preserve_db, "--once"],
        env=sd_env, capture_output=True, text=True)
    pc = db.connect(preserve_db)
    try:
        row = pc.execute(
            "SELECT status, claimed_by_machine FROM experiments "
            "WHERE queue_id='V3-EXQ-PRESERVE'").fetchone()
    finally:
        pc.close()
    check("sync_daemon coordinator mode preserves DB claim",
          sd.returncode == 0 and row["status"] == "claimed"
          and row["claimed_by_machine"] == "Mac")

    print("")
    if _failures:
        print("RESULT: FAIL (%d): %s" % (len(_failures),
                                         ", ".join(_failures)))
        return 1
    print("RESULT: PASS (all checks green)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
