"""Coordinator client shim -- imported by the experiment runner.

CONTRACT: this module must never block or raise into the runner. Every
public function returns quickly, swallows all exceptions, and is a hard
no-op unless COORDINATION_MODE is explicitly set. The default mode is
"git", under which every function returns immediately doing nothing, so a
runner with this module present behaves byte-identically to one without it.

Modes:
  git         (default) -- no-op. Live path unchanged.
  shadow      -- best-effort reports alongside the existing git ops.
  coordinator -- Phase 2+. Reserved; treated like shadow for reporting.

Env:
  COORDINATION_MODE     git | shadow | coordinator
  COORDINATOR_URL       e.g. http://10.8.0.1:8787
  COORDINATOR_TOKEN     per-worker bearer token
  COORDINATOR_TIMEOUT   seconds, default 3
  COORDINATOR_LOG       optional path for a local best-effort audit log

All printed/logged text is ASCII-only.
"""

import gzip
import json
import os
import time
import urllib.parse
import urllib.request

MODE = os.environ.get("COORDINATION_MODE", "git")
URL = os.environ.get("COORDINATOR_URL", "").rstrip("/")
TOKEN = os.environ.get("COORDINATOR_TOKEN", "")
TIMEOUT = float(os.environ.get("COORDINATOR_TIMEOUT", "3"))
LOG_PATH = os.environ.get("COORDINATOR_LOG", "")

_ENABLED = MODE in ("shadow", "coordinator") and bool(URL)


def enabled():
    return _ENABLED


def _log(msg):
    if not LOG_PATH:
        return
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write("%s %s\n" % (
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg))
    except OSError:
        pass


def _post(path, payload, gzip_body=False):
    """Best-effort POST. Returns parsed JSON dict or None. Never raises."""
    if not _ENABLED:
        return None
    try:
        data = json.dumps(payload).encode("utf-8") if not gzip_body \
            else payload  # payload is already bytes when gzip_body
        headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer " + TOKEN}
        if gzip_body:
            headers["Content-Encoding"] = "gzip"
        req = urllib.request.Request(
            URL + path, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- must never escape
        _log("POST %s failed: %r" % (path, exc))
        return None


def _get(path):
    if not _ENABLED:
        return None
    try:
        req = urllib.request.Request(
            URL + path, headers={"Authorization": "Bearer " + TOKEN},
            method="GET")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log("GET %s failed: %r" % (path, exc))
        return None


# ---- public API (called from the runner) ------------------------------

def report_claim(queue_id, machine, git_verdict):
    """Report a git-claim attempt + its verdict for shadow comparison."""
    if not _ENABLED:
        return None
    r = _post("/claim", {"queue_id": queue_id, "machine": machine,
                         "git_verdict": git_verdict})
    if r and r.get("diverged"):
        _log("DIVERGENCE queue_id=%s machine=%s git=%s coord=%s" % (
            queue_id, machine, git_verdict, r.get("verdict")))
    return r


def report_heartbeat(machine, state, current_exq, progress, gpu):
    if not _ENABLED:
        return None
    return _post("/heartbeat", {"machine": machine, "state": state,
                                "current_exq": current_exq,
                                "progress": progress, "gpu": gpu})


def report_status(machine, status_obj):
    if not _ENABLED:
        return None
    return _post("/status", {"machine": machine, "status": status_obj})


def report_result(queue_id, run_id, manifest_path, outcome, machine):
    """Ship the manifest bytes to the coordinator. Idempotent on run_id
    server-side, so a replay after a partition is a harmless no-op."""
    if not _ENABLED:
        return None
    try:
        with open(manifest_path, "rb") as fh:
            raw = fh.read()
        # Ensure run_id/queue_id/outcome are present for the server even if
        # the manifest itself omits them (it should not, but be safe).
        try:
            doc = json.loads(raw.decode("utf-8"))
            doc.setdefault("run_id", run_id)
            doc.setdefault("queue_id", queue_id)
            doc.setdefault("outcome", outcome)
            doc.setdefault("machine", machine)
            raw = json.dumps(doc).encode("utf-8")
        except (ValueError, UnicodeDecodeError):
            pass
        body = gzip.compress(raw)
    except OSError as exc:
        _log("report_result read failed %s: %r" % (manifest_path, exc))
        return None
    return _post("/result", body, gzip_body=True)


def report_queue_remove(queue_id, reason):
    if not _ENABLED:
        return None
    return _post("/queue/remove", {"queue_id": queue_id, "reason": reason})


def fetch_commands(machine):
    if not _ENABLED:
        return None
    return _get("/commands?machine=" + urllib.parse.quote(machine))
