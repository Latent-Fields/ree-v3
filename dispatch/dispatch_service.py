#!/usr/bin/env python3
"""REE phone-dispatch service (standalone, hub-hosted).

A small, ISOLATED job queue so the user can start chip-spawned Claude sessions
from a phone. It is DELIBERATELY separate from the experiment coordinator
(ree-v3/coordinator/app.py): its own port, its own sqlite db, its own token.
If it crashes it cannot affect the coordinator (the sole git writer) or the
experiment pipeline.

Topology (Design B, Mac executor):
  phone  --(WireGuard + browser)-->  THIS service on the hub (always-on)
                                         |  durable sqlite queue
  Mac    --(WireGuard + poll)------->  GET /api/pending  ->  runs `claude -p`
                                         |
  phone  <--(ntfy push)-------------  notify on done/failed

Job lifecycle:
  staged    -- a mirrored chip suggestion, awaiting the user's tap
  pending   -- the user tapped Launch (or enqueued manually); executor may pick it
  claimed   -- an executor claimed it (atomic pending->claimed)
  running   -- executor is running `claude -p`
  done      -- finished, exit 0
  failed    -- finished, non-zero / error
  cancelled -- user cancelled (from staged/pending)

Auth: every /api/* and the executor endpoints require Authorization: Bearer <tok>.
GET / (the mobile page shell) and GET /health are unauthenticated -- the page is
a static shell; its JS prompts for the token once (localStorage) and sends it on
every data call. Bind to the WireGuard interface IP only in prod
(DISPATCH_BIND_HOST=10.8.0.1).

Env:
  DISPATCH_BIND_HOST    default 127.0.0.1 (set to the WG hub IP in prod)
  DISPATCH_BIND_PORT    default 8799
  DISPATCH_DB           default ./dispatch.db
  DISPATCH_TOKENS_FILE  JSON {token: label} (default ./dispatch_tokens.json)
  DISPATCH_TOKEN        single token (alternative to the tokens file; label "default")
  DISPATCH_NTFY_TOPIC   ntfy topic for push notifications (optional)
  DISPATCH_NTFY_SERVER  default https://ntfy.sh
  DISPATCH_MAX_BODY     default 262144
"""
import hmac
import json
import os
import sqlite3
import sys
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

BIND_HOST = os.environ.get("DISPATCH_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("DISPATCH_BIND_PORT", "8799"))
DB_PATH = os.environ.get(
    "DISPATCH_DB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispatch.db"))
TOKENS_FILE = os.environ.get(
    "DISPATCH_TOKENS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dispatch_tokens.json"))
SINGLE_TOKEN = os.environ.get("DISPATCH_TOKEN", "")
NTFY_TOPIC = os.environ.get("DISPATCH_NTFY_TOPIC", "")
NTFY_SERVER = os.environ.get("DISPATCH_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
MAX_BODY = int(os.environ.get("DISPATCH_MAX_BODY", str(256 * 1024)))

# Terminal + live status sets.
OPEN_STATUSES = ("staged", "pending", "claimed", "running")
TERMINAL_STATUSES = ("done", "failed", "cancelled")
ALL_STATUSES = OPEN_STATUSES + TERMINAL_STATUSES

_tokens = {}
_tokens_lock = threading.Lock()
_db_lock = threading.Lock()


# --------------------------------------------------------------------------
# time / tokens
# --------------------------------------------------------------------------
def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_tokens():
    global _tokens
    toks = {}
    if SINGLE_TOKEN:
        toks[SINGLE_TOKEN] = "default"
    if os.path.exists(TOKENS_FILE):
        try:
            with open(TOKENS_FILE, "r", encoding="utf-8") as fh:
                disk = json.load(fh)
            if isinstance(disk, dict):
                toks.update(disk)
        except (ValueError, OSError):
            pass
    with _tokens_lock:
        _tokens = toks


def auth_label(header_value):
    """Return the token label for a valid bearer token, else None."""
    if not header_value or not header_value.startswith("Bearer "):
        return None
    presented = header_value[len("Bearer "):].strip()
    with _tokens_lock:
        for tok, label in _tokens.items():
            if hmac.compare_digest(presented, tok):
                return label
    return None


# --------------------------------------------------------------------------
# db
# --------------------------------------------------------------------------
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT '',
    prompt        TEXT NOT NULL,
    cwd           TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'pending',
    source        TEXT NOT NULL DEFAULT 'manual',   -- manual | chip | api
    created_by    TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    claimed_by    TEXT,
    claimed_at    TEXT,
    started_at    TEXT,
    finished_at   TEXT,
    exit_code     INTEGER,
    summary       TEXT,
    log_tail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    with _db_lock, db_connect() as conn:
        conn.executescript(SCHEMA)


def _row_to_dict(row):
    return {k: row[k] for k in row.keys()}


def create_job(title, prompt, cwd, status, source, created_by):
    job_id = uuid.uuid4().hex[:12]
    ts = now_iso()
    with _db_lock, db_connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id,title,prompt,cwd,status,source,created_by,"
            "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (job_id, title, prompt, cwd, status, source, created_by, ts, ts))
    return job_id


def list_jobs(limit=200):
    # Open jobs first (staged/pending/claimed/running) oldest-first, then
    # terminal jobs newest-first.
    with _db_lock, db_connect() as conn:
        rows = conn.execute("SELECT * FROM jobs").fetchall()
    jobs = [_row_to_dict(r) for r in rows]
    order = {s: i for i, s in enumerate(ALL_STATUSES)}

    def sort_key(j):
        open_ = j["status"] in OPEN_STATUSES
        return (order.get(j["status"], 99),
                j["created_at"] if open_ else "")
    open_jobs = sorted([j for j in jobs if j["status"] in OPEN_STATUSES],
                       key=lambda j: (order[j["status"]], j["created_at"]))
    term_jobs = sorted([j for j in jobs if j["status"] in TERMINAL_STATUSES],
                       key=lambda j: j["updated_at"], reverse=True)
    return (open_jobs + term_jobs)[:limit]


def get_job(job_id):
    with _db_lock, db_connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return _row_to_dict(row) if row else None


def pending_jobs(limit=20):
    with _db_lock, db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs WHERE status='pending' ORDER BY created_at ASC "
            "LIMIT ?", (limit,)).fetchall()
    return [_row_to_dict(r) for r in rows]


def set_status_guarded(job_id, new_status, from_statuses, **fields):
    """Atomically move job_id to new_status only if currently in from_statuses.
    Returns the updated row dict, or None if the guard failed."""
    cols = ["status=?", "updated_at=?"]
    vals = [new_status, now_iso()]
    for k, v in fields.items():
        cols.append("%s=?" % k)
        vals.append(v)
    placeholders = ",".join("?" for _ in from_statuses)
    sql = ("UPDATE jobs SET " + ",".join(cols) +
           " WHERE id=? AND status IN (" + placeholders + ")")
    vals2 = vals + [job_id] + list(from_statuses)
    with _db_lock, db_connect() as conn:
        cur = conn.execute(sql, vals2)
        changed = cur.rowcount
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if changed == 0:
        return None
    return _row_to_dict(row) if row else None


# --------------------------------------------------------------------------
# notifications (ntfy push to the phone) -- best-effort, never raises
# --------------------------------------------------------------------------
def notify(title, message, priority="default", tags=""):
    if not NTFY_TOPIC:
        return
    url = "%s/%s" % (NTFY_SERVER, NTFY_TOPIC)
    try:
        req = urllib.request.Request(url, data=message.encode("utf-8"),
                                     method="POST")
        req.add_header("Title", title)
        if priority:
            req.add_header("Priority", priority)
        if tags:
            req.add_header("Tags", tags)
        urllib.request.urlopen(req, timeout=5).read()
    except Exception as exc:  # noqa: BLE001  best-effort
        sys.stderr.write("[dispatch] notify failed: %s\n" % exc)


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "REEDispatch/1.0"

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, code, html):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None

    def _authed(self):
        label = auth_label(self.headers.get("Authorization"))
        if label is None:
            self._send(401, {"error": "unauthorized"})
            return None
        return label

    def log_message(self, fmt, *args):
        sys.stderr.write("[dispatch] %s - %s\n" % (
            self.address_string(), (fmt % args)))

    # ---- GET ----------------------------------------------------------
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            self._send(200, {"ok": True, "service": "ree-dispatch",
                             "now": now_iso()})
            return
        if path == "/" or path == "/index.html":
            self._send_html(200, PAGE_HTML)
            return
        if path == "/api/jobs":
            if self._authed() is None:
                return
            self._send(200, {"jobs": list_jobs()})
            return
        if path == "/api/pending":
            if self._authed() is None:
                return
            self._send(200, {"jobs": pending_jobs()})
            return
        self._send(404, {"error": "not found", "path": path})

    # ---- POST ---------------------------------------------------------
    def do_POST(self):
        path = urlparse(self.path).path
        label = self._authed()
        if label is None:
            return
        body = self._json_body()
        if body is None:
            self._send(400, {"error": "bad body"})
            return

        if path == "/api/enqueue":
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                self._send(400, {"error": "prompt required"})
                return
            status = body.get("status") or "pending"
            if status not in ("staged", "pending"):
                status = "pending"
            source = body.get("source") or "manual"
            job_id = create_job(
                title=(body.get("title") or "").strip(),
                prompt=prompt,
                cwd=(body.get("cwd") or "").strip(),
                status=status, source=source, created_by=label)
            self._send(200, {"ok": True, "id": job_id, "status": status})
            return

        if path == "/api/launch":
            job_id = body.get("id")
            row = set_status_guarded(job_id, "pending", ("staged",))
            if row is None:
                self._send(409, {"error": "not in staged state"})
                return
            self._send(200, {"ok": True, "job": row})
            return

        if path == "/api/cancel":
            job_id = body.get("id")
            row = set_status_guarded(job_id, "cancelled", ("staged", "pending"))
            if row is None:
                self._send(409, {"error": "not cancellable (already running/done?)"})
                return
            self._send(200, {"ok": True, "job": row})
            return

        if path == "/api/claim":
            job_id = body.get("id")
            machine = body.get("machine") or label
            row = set_status_guarded(job_id, "claimed", ("pending",),
                                     claimed_by=machine, claimed_at=now_iso())
            if row is None:
                self._send(409, {"error": "already claimed or not pending"})
                return
            self._send(200, {"ok": True, "job": row})
            return

        if path == "/api/update":
            job_id = body.get("id")
            new_status = body.get("status")
            if new_status not in ("running", "done", "failed"):
                self._send(400, {"error": "status must be running|done|failed"})
                return
            fields = {}
            if "exit_code" in body:
                fields["exit_code"] = body.get("exit_code")
            if "summary" in body:
                fields["summary"] = body.get("summary")
            if "log_tail" in body:
                fields["log_tail"] = body.get("log_tail")
            if new_status == "running":
                fields["started_at"] = now_iso()
                row = set_status_guarded(job_id, "running",
                                         ("claimed", "running"), **fields)
            else:
                fields["finished_at"] = now_iso()
                row = set_status_guarded(job_id, new_status,
                                         ("claimed", "running"), **fields)
            if row is None:
                self._send(409, {"error": "not in claimed/running state"})
                return
            if new_status in ("done", "failed"):
                title = row.get("title") or row.get("id")
                ok = new_status == "done"
                notify(
                    "Dispatch %s: %s" % ("done" if ok else "FAILED", title),
                    (row.get("summary") or "")[:280],
                    priority="default" if ok else "high",
                    tags="white_check_mark" if ok else "x")
            self._send(200, {"ok": True, "job": row})
            return

        self._send(404, {"error": "not found", "path": path})


# --------------------------------------------------------------------------
# mobile page (static shell; data via authed fetch with a localStorage token)
# --------------------------------------------------------------------------
PAGE_HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>REE Dispatch</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0;
         background: #0f1115; color: #e6e6e6; }
  header { padding: 14px 16px; background: #161a22; position: sticky; top: 0;
           border-bottom: 1px solid #262c38; display: flex; align-items: center;
           justify-content: space-between; }
  h1 { font-size: 17px; margin: 0; }
  button { font-size: 15px; border: 0; border-radius: 8px; padding: 9px 12px;
           background: #2d6cdf; color: #fff; }
  button.sec { background: #2a3140; }
  button.warn { background: #7a2230; }
  button:active { opacity: .7; }
  main { padding: 12px 14px 80px; }
  .job { background: #161a22; border: 1px solid #262c38; border-radius: 10px;
         padding: 12px; margin-bottom: 10px; }
  .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .title { font-weight: 600; font-size: 15px; flex: 1; word-break: break-word; }
  .pill { font-size: 11px; padding: 2px 8px; border-radius: 999px;
          text-transform: uppercase; letter-spacing: .04em; }
  .s-staged{background:#3a3320;color:#e8c766}.s-pending{background:#1f3350;color:#7db1ff}
  .s-claimed,.s-running{background:#1f4a3a;color:#6fe0a8}
  .s-done{background:#1d3a24;color:#7fe39a}.s-failed{background:#4a1f26;color:#ff9aa8}
  .s-cancelled{background:#2a3140;color:#9aa3b2}
  .prompt { font-size: 12.5px; color: #9aa3b2; margin-top: 7px;
            white-space: pre-wrap; word-break: break-word; max-height: 90px;
            overflow: auto; }
  .meta { font-size: 11px; color: #6b7382; margin-top: 6px; }
  .actions { margin-top: 10px; display: flex; gap: 8px; }
  textarea, input { width: 100%; background: #0f1115; color: #e6e6e6;
                    border: 1px solid #2a3140; border-radius: 8px; padding: 9px;
                    font-size: 14px; font-family: inherit; }
  .new { background:#161a22; border:1px solid #262c38; border-radius:10px;
         padding:12px; margin-bottom:14px; }
  label { font-size: 12px; color: #9aa3b2; display:block; margin: 8px 0 4px; }
  .muted { color:#6b7382; font-size:12px; }
</style></head><body>
<header><h1>REE Dispatch</h1>
  <div class="row"><button class="sec" onclick="setToken()">Token</button>
  <button class="sec" onclick="load()">Refresh</button></div>
</header>
<main>
  <div class="new">
    <label>New job prompt (self-contained)</label>
    <textarea id="np" rows="4" placeholder="e.g. In REE_assembly, ..."></textarea>
    <label>Title (optional)</label><input id="nt" placeholder="short label">
    <label>cwd (repo path on the Mac, optional)</label>
    <input id="nc" placeholder="/Users/dgolden/REE_Working/REE_assembly">
    <div class="actions"><button onclick="enqueue()">Enqueue + Launch</button>
      <button class="sec" onclick="enqueue('staged')">Stage only</button></div>
  </div>
  <div id="list"></div>
  <p class="muted" id="status"></p>
</main>
<script>
const TK="ree_dispatch_token";
function tok(){return localStorage.getItem(TK)||"";}
function setToken(){const t=prompt("Dispatch bearer token:",tok());if(t!==null)localStorage.setItem(TK,t.trim());load();}
function hdr(){return {"Authorization":"Bearer "+tok(),"Content-Type":"application/json"};}
async function api(path,method,body){
  const o={method:method||"GET",headers:hdr()};
  if(body)o.body=JSON.stringify(body);
  const r=await fetch(path,o);
  if(r.status===401){st("Unauthorized -- tap Token.");throw new Error("401");}
  return r.json();
}
function st(m){document.getElementById("status").textContent=m;}
function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function jobHtml(j){
  let a="";
  if(j.status==="staged")a+=`<button onclick="act('launch','${j.id}')">Launch</button>`;
  if(j.status==="staged"||j.status==="pending")a+=`<button class="warn" onclick="act('cancel','${j.id}')">Cancel</button>`;
  const meta=[j.source,j.created_at,j.cwd].filter(Boolean).join(" &middot; ");
  const tail=j.summary?`<div class="meta">${esc(j.summary)}</div>`:"";
  return `<div class="job"><div class="row"><span class="title">${esc(j.title||j.id)}</span>
    <span class="pill s-${j.status}">${j.status}</span></div>
    <div class="prompt">${esc(j.prompt)}</div>
    <div class="meta">${esc(meta)}</div>${tail}
    <div class="actions">${a}</div></div>`;
}
async function load(){
  try{const d=await api("/api/jobs");
    document.getElementById("list").innerHTML=(d.jobs||[]).map(jobHtml).join("")||'<p class="muted">No jobs.</p>';
    st("Updated "+new Date().toLocaleTimeString());
  }catch(e){}
}
async function act(kind,id){try{await api("/api/"+kind,"POST",{id});load();}catch(e){}}
async function enqueue(status){
  const prompt=document.getElementById("np").value.trim();
  if(!prompt){st("Prompt required.");return;}
  const body={prompt,title:document.getElementById("nt").value.trim(),
              cwd:document.getElementById("nc").value.trim()};
  if(status)body.status=status;
  try{await api("/api/enqueue","POST",body);
    document.getElementById("np").value="";document.getElementById("nt").value="";
    load();}catch(e){}
}
if(!tok())setToken();else load();
setInterval(load,15000);
</script></body></html>
"""


def main():
    db_init()
    load_tokens()
    with _tokens_lock:
        n_tokens = len(_tokens)
    if n_tokens == 0:
        sys.stderr.write(
            "[dispatch] WARNING: no tokens loaded -- all /api calls will 401. "
            "Set DISPATCH_TOKEN or populate %s\n" % TOKENS_FILE)
    srv = ThreadingHTTPServer((BIND_HOST, BIND_PORT), Handler)
    sys.stdout.write(
        "[dispatch] listening on %s:%d db=%s tokens=%d ntfy=%s\n" % (
            BIND_HOST, BIND_PORT, DB_PATH, n_tokens,
            NTFY_TOPIC or "(off)"))
    sys.stdout.flush()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
