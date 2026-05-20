"""Experiment coordinator -- pure stdlib (http.server + sqlite3).

Zero third-party dependencies by design: the deploy target is a minimal
CX22 and a venv/dependency is itself a fragility source. Low-QPS internal
service (a handful of workers polling ~every 30s) so ThreadingHTTPServer is
more than adequate.

Auth: every endpoint except /health requires `Authorization: Bearer <tok>`.
Tokens live in a JSON file mapping token -> machine label (per-worker, so a
single worker can be revoked without rotating everyone). The socket should
bind to the WireGuard interface IP only (set COORDINATOR_BIND_HOST); the
token is defense-in-depth, not the only control.

All printed text is ASCII-only (Windows cp1252 safety).

Config (env):
  COORDINATOR_DB           sqlite path (default ./coordinator.db)
  COORDINATOR_BIND_HOST    default 127.0.0.1 (set to the WG IP in prod)
  COORDINATOR_BIND_PORT    default 8787
  COORDINATOR_TOKENS_FILE  JSON {token: machine} (default ./tokens.json)
  COORDINATOR_STALE_HOURS  stale-claim cutoff, default 6
  COORDINATOR_HEARTBEAT_FRESH_SECONDS live-owner grace window, default 900
  COORDINATOR_MODE         shadow (default) | coordinator
"""

import gzip
import hashlib
import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import db

MAX_BODY = 32 * 1024 * 1024  # 32MB; observed max manifest ~7MB

DB_PATH = os.environ.get("COORDINATOR_DB", os.path.join(
    os.path.dirname(__file__), "coordinator.db"))
BIND_HOST = os.environ.get("COORDINATOR_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("COORDINATOR_BIND_PORT", "8787"))
TOKENS_FILE = os.environ.get("COORDINATOR_TOKENS_FILE", os.path.join(
    os.path.dirname(__file__), "tokens.json"))
STALE_HOURS = float(os.environ.get("COORDINATOR_STALE_HOURS", "6"))
HEARTBEAT_FRESH_SECONDS = int(
    os.environ.get("COORDINATOR_HEARTBEAT_FRESH_SECONDS", "900")
)
MODE = os.environ.get("COORDINATOR_MODE", "shadow")

_tokens_lock = threading.Lock()
_tokens = {}


def load_tokens():
    global _tokens
    with _tokens_lock:
        if os.path.exists(TOKENS_FILE):
            with open(TOKENS_FILE, "r", encoding="utf-8") as fh:
                _tokens = json.load(fh)
        else:
            _tokens = {}


def auth_machine(header_value):
    """Return the machine label for a valid bearer token, else None.
    Constant-time compare against each known token."""
    if not header_value or not header_value.startswith("Bearer "):
        return None
    presented = header_value[len("Bearer "):].strip()
    with _tokens_lock:
        for tok, machine in _tokens.items():
            if hmac.compare_digest(presented, tok):
                return machine
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "REECoordinator/1.0"

    # ---- helpers -------------------------------------------------------
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        if length > MAX_BODY:
            return None
        raw = self.rfile.read(length)
        if (self.headers.get("Content-Encoding") or "").lower() == "gzip":
            try:
                raw = gzip.decompress(raw)
            except OSError:
                return None
        return raw

    def _json_body(self):
        raw = self._read_body()
        if raw is None:
            return None
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _authed(self):
        m = auth_machine(self.headers.get("Authorization"))
        if m is None:
            self._send(401, {"error": "unauthorized"})
            return None
        return m

    def log_message(self, fmt, *args):
        # ASCII-only, single line, to stderr.
        sys.stderr.write("[coordinator] %s - %s\n" % (
            self.address_string(), (fmt % args)))

    # ---- routing -------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True, "mode": MODE})
            return
        if self._authed() is None:
            return
        if path == "/commands":
            qs = parse_qs(urlparse(self.path).query)
            machine = (qs.get("machine") or [""])[0]
            conn = db.connect(DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT id, kind, args, issued_by, issued_at FROM "
                    "commands WHERE machine=? AND acked_at IS NULL "
                    "ORDER BY id", (machine,)).fetchall()
                cmds = [dict(r) for r in rows]
            finally:
                conn.close()
            self._send(200, {"machine": machine, "commands": cmds})
            return
        if path == "/shadow/divergence":
            conn = db.connect(DB_PATH)
            try:
                rows = conn.execute(
                    "SELECT queue_id, machine, git_verdict, coord_verdict, "
                    "detail, logged_at FROM claim_log WHERE diverged=1 "
                    "ORDER BY id DESC LIMIT 200").fetchall()
                total = conn.execute(
                    "SELECT COUNT(*) c FROM claim_log").fetchone()["c"]
                ndiv = conn.execute("SELECT COUNT(*) c FROM claim_log "
                                    "WHERE diverged=1").fetchone()["c"]
                dstat = db.divergence_stats(conn)
            finally:
                conn.close()
            self._send(200, {"total_claims": total, "divergences": ndiv,
                             "divergences_explained": dstat["explained"],
                             "divergences_blocking": dstat["blocking"],
                             "rows": [dict(r) for r in rows]})
            return
        if path == "/shadow/status":
            # One-call operator soak snapshot: traffic seen, divergence
            # count, per-machine heartbeat freshness. Backs check_shadow.py.
            conn = db.connect(DB_PATH)
            try:
                total = conn.execute(
                    "SELECT COUNT(*) c FROM claim_log").fetchone()["c"]
                ndiv = conn.execute("SELECT COUNT(*) c FROM claim_log "
                                    "WHERE diverged=1").fetchone()["c"]
                dstat = db.divergence_stats(conn)
                nexp = conn.execute(
                    "SELECT COUNT(*) c FROM experiments").fetchone()["c"]
                machines = []
                for r in conn.execute(
                    "SELECT machine, last_seen, state, current_exq, "
                    "progress_json FROM heartbeats ORDER BY machine"
                ).fetchall():
                    row = dict(r)
                    raw_pj = row.pop("progress_json", None)
                    try:
                        row["progress"] = (
                            json.loads(raw_pj)
                            if isinstance(raw_pj, str) and raw_pj else {})
                    except (TypeError, ValueError):
                        row["progress"] = {}
                    machines.append(row)
                recent = [dict(r) for r in conn.execute(
                    "SELECT queue_id, machine, git_verdict, coord_verdict, "
                    "logged_at FROM claim_log WHERE diverged=1 "
                    "ORDER BY id DESC LIMIT 20").fetchall()]
            finally:
                conn.close()
            self._send(200, {"mode": MODE, "total_claims": total,
                             "divergences": ndiv,
                             "divergences_explained": dstat["explained"],
                             "divergences_blocking": dstat["blocking"],
                             "adjusted_divergences": dstat["blocking"],
                             "experiments_in_mirror": nexp,
                             "machines": machines,
                             "recent_divergences": recent})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        machine_tok = self._authed()
        if machine_tok is None:
            return
        body = self._json_body() if path != "/result" else None

        if path == "/claim":
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            qid = body.get("queue_id")
            machine = body.get("machine") or machine_tok
            git_verdict = body.get("git_verdict")  # may be None
            if not qid:
                self._send(400, {"error": "queue_id required"})
                return
            conn = db.connect(DB_PATH)
            try:
                coord_verdict = db.evaluate_claim(
                    conn, qid, machine, STALE_HOURS,
                    HEARTBEAT_FRESH_SECONDS)
                if MODE == "shadow":
                    diverged = db.log_claim(
                        conn, qid, machine, git_verdict, coord_verdict)
                    db.apply_git_outcome(conn, qid, machine, git_verdict)
                    self._send(200, {"verdict": coord_verdict,
                                     "diverged": bool(diverged),
                                     "authoritative": False})
                else:
                    real = db.try_claim(
                        conn, qid, machine, STALE_HOURS,
                        HEARTBEAT_FRESH_SECONDS)
                    self._send(200, {"verdict": real,
                                     "authoritative": True})
            finally:
                conn.close()
            return

        if path == "/claim/release":
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            qid = body.get("queue_id")
            machine = body.get("machine") or machine_tok
            if not qid:
                self._send(400, {"error": "queue_id required"})
                return
            if MODE == "shadow":
                self._send(200, {"ok": True, "applied": False,
                                 "note": "shadow mode"})
                return
            conn = db.connect(DB_PATH)
            try:
                ok, note = db.release_claim(conn, qid, machine)
            finally:
                conn.close()
            self._send(200 if ok else 409,
                       {"ok": ok, "applied": ok, "note": note})
            return

        if path == "/heartbeat":
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            conn = db.connect(DB_PATH)
            try:
                db.upsert_heartbeat(
                    conn, body.get("machine") or machine_tok,
                    body.get("state"), body.get("current_exq"),
                    body.get("progress"), body.get("gpu"))
            finally:
                conn.close()
            self._send(200, {"ok": True})
            return

        if path == "/status":
            # Status blob is regenerable telemetry; record receipt only.
            if body is None:
                self._send(400, {"error": "bad body"})
                return
            self._send(200, {"ok": True})
            return

        if path == "/result":
            raw = self._read_body()
            if raw is None:
                self._send(413, {"error": "body too large or bad gzip"})
                return
            try:
                manifest = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                self._send(400, {"error": "result not valid json"})
                return
            run_id = manifest.get("run_id")
            if not run_id:
                self._send(400, {"error": "run_id required"})
                return
            sha = hashlib.sha256(raw).hexdigest()
            conn = db.connect(DB_PATH)
            try:
                fresh = db.record_result(
                    conn, run_id, manifest.get("queue_id"),
                    manifest.get("machine") or machine_tok,
                    manifest.get("outcome"), sha, len(raw))
            finally:
                conn.close()
            self._send(200, {"ok": True, "run_id": run_id,
                             "idempotent_noop": (not fresh)})
            return

        if path == "/queue/remove":
            if body is None or not body.get("queue_id"):
                self._send(400, {"error": "queue_id required"})
                return
            # Shadow: do not mutate the mirror destructively; sync_daemon
            # reconciles removals from experiment_queue.json (git is
            # authoritative in Phase 1). Coordinator mode marks the item
            # completed immediately so no other coordinator-mode worker can
            # reclaim it while the git queue removal propagates.
            applied = False
            if MODE != "shadow":
                conn = db.connect(DB_PATH)
                try:
                    applied = db.mark_queue_removed(
                        conn, body.get("queue_id"), body.get("reason"))
                finally:
                    conn.close()
            self._send(200, {"ok": True, "applied": applied})
            return

        self._send(404, {"error": "not found"})


def main():
    db.init_db(DB_PATH)
    load_tokens()
    srv = ThreadingHTTPServer((BIND_HOST, BIND_PORT), Handler)
    sys.stderr.write(
        "[coordinator] mode=%s listening on %s:%d db=%s tokens=%d\n" % (
            MODE, BIND_HOST, BIND_PORT, DB_PATH, len(_tokens)))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
