"""Coordinator client shim -- imported by the experiment runner.

CONTRACT: this module must never block or raise into the runner. Every
public function returns quickly, swallows all exceptions, and is a hard
no-op unless COORDINATION_MODE is explicitly set. The default mode is
"git", under which every function returns immediately doing nothing, so a
runner with this module present behaves byte-identically to one without it.

Modes:
  git         (default) -- no-op. Live path unchanged.
  shadow      -- best-effort reports alongside the existing git ops.
  coordinator -- Phase 2 claim cutover. /claim is authoritative; status,
                 result, and queue-remove reports remain best-effort side
                 channels while git still carries committed evidence.

Env:
  COORDINATION_MODE     git | shadow | coordinator
  COORDINATOR_URL       e.g. http://10.8.0.1:8787
  COORDINATOR_TOKEN     per-worker bearer token
  COORDINATOR_TIMEOUT   seconds, default 10 (raised from 3 on 2026-05-30
                        after fleet-wide stale heartbeats traced to
                        consistent /heartbeat POST timeouts under the
                        3s default with the rich per-tick payload)
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
TIMEOUT = float(os.environ.get("COORDINATOR_TIMEOUT", "10"))
LOG_PATH = os.environ.get("COORDINATOR_LOG", "")

_ENABLED = MODE in ("shadow", "coordinator") and bool(URL)


def enabled():
    return _ENABLED


def claims_authoritative():
    return MODE == "coordinator"


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


def claim(queue_id, machine):
    """Authoritative Phase-2 claim. Returns ok/already_claimed/error.

    Coordinator mode must not fall through to running unclaimed work. Any
    HTTP/auth/timeout failure returns 'error' so the runner can skip and
    retry on the next loop tick.
    """
    if MODE != "coordinator":
        return "error"
    r = _post("/claim", {"queue_id": queue_id, "machine": machine})
    if not r or not r.get("authoritative"):
        return "error"
    verdict = r.get("verdict")
    if verdict in ("ok", "already_claimed", "error"):
        return verdict
    return "error"


def release_claim(queue_id, machine):
    if MODE != "coordinator":
        return None
    return _post("/claim/release", {"queue_id": queue_id,
                                    "machine": machine})


def report_heartbeat(machine, state, current_exq, progress, gpu, *,
                     seconds_elapsed=None, seconds_remaining=None,
                     payload=None):
    """Report a heartbeat tick to the coordinator.

    `payload` (optional) is the full runner-side dict written to
    `runner_heartbeats/<machine>.json`. Send it under PLAN.md step 6
    so sync_daemon can materialise the rich file from the coordinator
    DB and the runner can stop git-pushing it directly. None is the
    legacy path: the structured fields still flow, only the rich
    payload is missing (lifecycle_state remains derivable from the
    structured columns).
    """
    if not _ENABLED:
        return None
    body = {"machine": machine, "state": state,
            "current_exq": current_exq, "progress": progress, "gpu": gpu}
    if seconds_elapsed is not None:
        body["seconds_elapsed"] = seconds_elapsed
    if seconds_remaining is not None:
        body["seconds_remaining"] = seconds_remaining
    if payload is not None:
        body["payload"] = payload
    return _post("/heartbeat", body)


def report_status(machine, status_obj):
    if not _ENABLED:
        return None
    return _post("/status", {"machine": machine, "status": status_obj})


def report_shutdown(machine, reason=None, expected_wake_condition=None):
    """Announce an intentional shutdown so the coordinator can return
    lifecycle_state=gracefully_offline for this machine on /shadow/status
    until heartbeats resume.

    Best-effort: failures are logged but never raise. The runner is in
    the middle of exiting; an HTTP retry loop would just hold up the
    shutdown for a coordinator that's already unreachable.

    No-op when COORDINATION_MODE is unset/git (workers not in the
    coordinator state graph have nothing to announce).
    """
    if not _ENABLED:
        return None
    body = {"machine": machine}
    if reason is not None:
        body["reason"] = reason
    if expected_wake_condition is not None:
        body["expected_wake_condition"] = expected_wake_condition
    return _post("/shutdown_notify", body)


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


def issue_command(machine, kind, args=None, issued_by="unknown"):
    """Issue a remote-control command for `machine` via the coordinator.

    Parity helper for in-process ree-v3 callers and tests; serve.py issues
    via its own urllib path (REE_assembly cannot import this module). Returns
    the parsed response dict ({ok, command}) or None when disabled / on any
    transport failure."""
    if not _ENABLED:
        return None
    return _post("/commands/issue", {
        "machine": machine, "kind": kind,
        "args": args or {}, "issued_by": issued_by})


def ack_command(command_id, machine, result_status="done", result_note=None):
    """Ack a command this runner has executed. Stamps the terminal result on
    the coordinator so GET /commands stops returning it (this is what lets a
    worker drop the git command-file without restart-looping a stop command).

    Best-effort: returns the response dict or None. A None return (coordinator
    unreachable) leaves the command pending; it will be re-fetched and the
    idempotent command kinds re-executed harmlessly next tick."""
    if not _ENABLED:
        return None
    body = {"id": command_id, "machine": machine,
            "result_status": result_status}
    if result_note is not None:
        body["result_note"] = result_note
    return _post("/commands/ack", body)
