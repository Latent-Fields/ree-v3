#!/usr/bin/env python3
"""
REE-V3 Experiment Runner

Runs pending experiments from experiment_queue.json sequentially,
writing live progress to runner_status.json for the claims explorer
dashboard.

Usage:
    # Run in foreground (Ctrl+C to stop):
    /opt/local/bin/python3 experiment_runner.py

    # Run detached (survives closing this terminal):
    nohup /opt/local/bin/python3 experiment_runner.py > runner.log 2>&1 &
    echo $! > runner.pid

    # Stop a detached run:
    kill $(cat runner.pid)

V3 changes from V2 runner:
  - ree_version: "v3"
  - Completion detection patterns updated for V3 experiment stdout format
  - REPO_ROOT is ree-v3 dir
  - Status file still goes to REE_assembly runner_status.json (shared explorer)
  - Preserves existing completed runs in runner_status.json when starting
"""

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


# Ensure UTF-8 output on Windows (default cp1252 breaks -> and other Unicode in experiment scripts)
os.environ['PYTHONIOENCODING'] = 'utf-8'

REPO_ROOT = Path(__file__).resolve().parent
QUEUE_FILE = REPO_ROOT / "experiment_queue.json"
PID_FILE = REPO_ROOT / "runner.pid"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "experiments"
SCRIPT_TIMING_FILE = REPO_ROOT / "script_timing.json"

# Auto-detect REE_assembly runner_status directory (per-machine files)
_REE_ASSEMBLY_STATUS_DIRS = [
    REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / "runner_status",
    Path.home() / "Documents" / "GitHub" / "REE_Working" / "REE_assembly" / "evidence" / "experiments" / "runner_status",
]

STATUS_WRITE_INTERVAL = 5

# V3 experiment completion detection patterns (stdout signals)
RE_SEED_CONDITION = re.compile(r'Seed\s+(\d+)\s+Condition\s+(\w+)')
RE_EP_PROGRESS = re.compile(r'ep\s+(\d+)/(\d+)')
RE_TRAIN_PROGRESS = re.compile(r'\[train\]\s+ep\s+(\d+)/(\d+)')
RE_RUN_DONE_PATTERNS = [
    re.compile(r'verdict:\s+(PASS|FAIL)'),
    re.compile(r'V3-EXQ-\d+\s+verdict:\s+(PASS|FAIL)'),
    re.compile(r'SD-\d+\s+/\s+V3-EXQ-\d+\s+verdict:\s+(PASS|FAIL)'),
]
RE_STATUS_LINE = re.compile(r'^Status:\s+(PASS|FAIL)')
RE_EXQ_VERDICT = re.compile(r'\[EXQ-[\w-]+\]\s+(PASS|FAIL)')
RE_DONE_OUTCOME = re.compile(r'Done\.\s+Outcome:\s+(PASS|FAIL)')
RE_SAVED_TO = re.compile(r'Result (?:pack )?written to:?\s+(.+)')


def find_ree_assembly_path() -> Path | None:
    """Locate the REE_assembly repo (for git auto-sync pushes)."""
    candidates = [
        REPO_ROOT.parent / "REE_assembly",
        Path.home() / "Documents" / "GitHub" / "REE_Working" / "REE_assembly",
    ]
    for c in candidates:
        if c.is_dir() and (c / "evidence" / "experiments").is_dir():
            return c
    return None


def git_pull(repo_path: Path, label: str) -> None:
    """Pull latest changes. Retries on transient lock errors. Never raises."""
    import time
    _LOCK_HINTS = ("cannot lock ref", "unable to resolve reference",
                   "lock file", "index.lock")
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(repo_path), capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                msg = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok"
                print(f"[runner] git pull {label}: {msg}", flush=True)
                return
            stderr = r.stderr.strip()
            if any(h in stderr.lower() for h in _LOCK_HINTS) and attempt < 2:
                print(f"[runner] git pull {label}: transient lock, retrying "
                      f"({attempt + 1}/2)...", flush=True)
                time.sleep(2)
                continue
            print(f"[runner] git pull {label} warn: {stderr}", flush=True)
            return
        except Exception as e:
            print(f"[runner] git pull {label} error: {e}", flush=True)
            return


def _check_active_claim_on_file(relative_path: str) -> bool:
    """Return True if TASK_CLAIMS.json has an active claim covering relative_path.

    Used to log a warning before overwriting local edits during git reset --hard.
    Never raises -- returns False on any error.
    """
    try:
        claims_path = REPO_ROOT.parent / "TASK_CLAIMS.json"
        if not claims_path.exists():
            return False
        data = json.loads(claims_path.read_text(encoding="utf-8"))
        for entry in data.get("claims", []):
            if entry.get("status") != "active":
                continue
            for res in entry.get("resources", []):
                if relative_path in res or res.endswith(relative_path):
                    return True
        return False
    except Exception:
        return False


def _merge_queue_json(remote_content: str, saved_content: str) -> str:
    """JSON-level merge of two experiment_queue.json strings.

    Strategy: take remote as the base (preserving completed-item removals pushed
    by the remote runner), then append any items from saved_content whose queue_id
    does not appear in the remote version.

    After merging, validates the result with validate_queue logic.  If validation
    fails, returns remote_content unchanged to avoid committing a broken queue.

    Returns the merged JSON string, or remote_content unchanged if merging fails.
    """
    try:
        remote = json.loads(remote_content)
        saved = json.loads(saved_content)
        remote_ids = {item["queue_id"] for item in remote.get("items", [])}
        new_items = [
            item for item in saved.get("items", [])
            if item["queue_id"] not in remote_ids
        ]
        if not new_items:
            return remote_content
        remote["items"] = remote.get("items", []) + new_items
        merged_str = json.dumps(remote, indent=2)

        # Validate merged result before accepting it
        try:
            from validate_queue import validate
            # Write to a temp file for validation
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                tf.write(merged_str)
                tf_path = tf.name
            errors = validate(Path(tf_path))
            Path(tf_path).unlink(missing_ok=True)
            if errors:
                print(f"[runner] queue merge validation FAILED ({len(errors)} errors) "
                      f"-- keeping remote version", flush=True)
                for e in errors[:3]:
                    print(f"[runner]   {e}", flush=True)
                return remote_content
        except ImportError:
            pass  # validator not available -- accept merge without validation

        labels = [item["queue_id"] for item in new_items]
        print(f"[runner] queue merge: restored {len(new_items)} item(s): {labels}", flush=True)
        return merged_str
    except Exception as e:
        print(f"[runner] queue merge failed: {e} -- keeping remote version", flush=True)
        return remote_content


def _git_push_with_retry(cwd: str, branch: str, label: str,
                         result_files: list[str] | None = None,
                         max_retries: int = 3) -> bool:
    """Push to origin, retrying with pull --rebase on rejection. Returns True on success.

    Uses git stash to preserve uncommitted work (e.g. from concurrent Claude sessions)
    instead of git reset --hard, which would destroy uncommitted edits.

    If pull --rebase fails (e.g. conflict), aborts the rebase and skips pushing
    rather than force-resetting. Lost pushes are recoverable on the next sync;
    lost uncommitted edits are not.
    """
    for attempt in range(max_retries):
        r = subprocess.run(
            ["git", "push", "origin", f"HEAD:{branch}"],
            cwd=cwd, capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print(f"[runner] auto-sync: pushed {label}", flush=True)
            return True
        if "fetch first" not in r.stderr and "non-fast-forward" not in r.stderr:
            # Some other push error
            print(f"[runner] auto-sync push warn ({label}): {r.stderr.strip()}", flush=True)
            return False

        # Remote has new commits -- pull rebase and retry
        pull = subprocess.run(
            ["git", "pull", "--rebase", "origin", branch],
            cwd=cwd, capture_output=True, text=True, timeout=30,
        )
        if pull.returncode == 0:
            print(f"[runner] auto-sync: pull-rebase {label} (retry {attempt+1})", flush=True)
            continue

        # Rebase failed (conflict) -- abort rebase, then use stash-based recovery
        print(f"[runner] auto-sync: rebase conflict ({label}), resolving safely...", flush=True)
        subprocess.run(["git", "rebase", "--abort"], cwd=cwd, capture_output=True, timeout=10)

        # Check for active claims on REE_assembly files before any destructive action
        if _check_active_claim_on_file("evidence/experiments/"):
            print(f"[runner] auto-sync: active TASK_CLAIMS on evidence/experiments/ "
                  f"-- skipping push to avoid data loss ({label})", flush=True)
            return False

        # Stash uncommitted work (preserves concurrent Claude session edits)
        stash_result = subprocess.run(
            ["git", "stash", "--include-untracked", "-m", f"runner-auto-sync-{now_utc()[:10]}"],
            cwd=cwd, capture_output=True, text=True, timeout=10,
        )
        stashed = "No local changes" not in stash_result.stdout

        # Fetch + reset to remote
        subprocess.run(["git", "fetch", "origin"], cwd=cwd, capture_output=True, timeout=30)
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"],
                       cwd=cwd, capture_output=True, timeout=10)

        # Pop stash to restore uncommitted work
        if stashed:
            pop = subprocess.run(
                ["git", "stash", "pop"],
                cwd=cwd, capture_output=True, text=True, timeout=10,
            )
            if pop.returncode != 0:
                # Stash pop conflict -- restore stash and skip pushing
                print(f"[runner] auto-sync: stash pop conflict -- skipping push ({label}). "
                      f"Stash preserved for manual recovery.", flush=True)
                return False

        # Re-stage only the specific result files the runner wrote (selective)
        if result_files:
            for f in result_files:
                try:
                    rel = str(Path(f).relative_to(Path(cwd)))
                except ValueError:
                    rel = f
                subprocess.run(["git", "add", rel], cwd=cwd, capture_output=True, timeout=10)
        else:
            # Fallback: broad staging (only if no specific files known)
            subprocess.run(["git", "add", "evidence/experiments/"],
                           cwd=cwd, capture_output=True, timeout=10)

        # Re-stage queue if present
        queue_path = Path(cwd) / "experiment_queue.json"
        if queue_path.exists():
            subprocess.run(["git", "add", "experiment_queue.json"],
                           cwd=cwd, capture_output=True, timeout=10)

        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd, timeout=5)
        if diff.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", f"auto-sync: re-apply results after conflict {now_utc()[:10]}"],
                cwd=cwd, capture_output=True, text=True, timeout=15,
            )
            print(f"[runner] auto-sync: re-applied local results ({label})", flush=True)
            continue  # retry push
        else:
            print(f"[runner] auto-sync: no local changes after reset ({label})", flush=True)
            return True  # remote already has everything

    print(f"[runner] auto-sync: push failed after {max_retries} retries ({label})", flush=True)
    return False


def git_push_queue() -> None:
    """Stage, commit, and push experiment_queue.json to ree-v3. Warns on failure."""
    try:
        subprocess.run(
            ["git", "add", "experiment_queue.json"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=10,
        )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(REPO_ROOT), timeout=5,
        )
        if diff.returncode == 0:
            return  # nothing changed
        subprocess.run(
            ["git", "commit", "-m", f"queue: remove completed/failed items {now_utc()[:10]}"],
            cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=15,
        )
        _git_push_with_retry(str(REPO_ROOT), "main", "queue update -> ree-v3")
    except Exception as e:
        print(f"[runner] auto-sync queue push error: {e}", flush=True)


def git_push_results(ree_assembly_path: Path, result_files: list[str] | None = None) -> None:
    """Stage, commit, and push experiment results in REE_assembly.

    If result_files is provided, only those specific files are staged (selective
    commit).  Otherwise falls back to staging the entire evidence/experiments/
    directory -- but this broad mode is discouraged because it can sweep up
    unrelated files from concurrent Claude sessions.

    Warns on failure; never raises.
    """
    try:
        if result_files:
            # Selective staging: only the files the runner actually wrote
            for f in result_files:
                # Convert absolute paths to repo-relative
                try:
                    rel = str(Path(f).relative_to(ree_assembly_path))
                except ValueError:
                    rel = f  # already relative or external -- stage as-is
                subprocess.run(
                    ["git", "add", rel],
                    cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
                )
        else:
            # Fallback: broad staging (legacy behaviour)
            subprocess.run(
                ["git", "add", "evidence/experiments/"],
                cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
            )
        # Nothing staged -> skip
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(ree_assembly_path), timeout=5,
        )
        if diff.returncode == 0:
            print("[runner] auto-sync: nothing new to push", flush=True)
            return
        msg = f"auto-sync: v3 results {now_utc()[:10]}"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=15,
        )
        _git_push_with_retry(str(ree_assembly_path), "master", "results -> REE_assembly")
    except Exception as e:
        print(f"[runner] auto-sync push error: {e}", flush=True)


# ── Multi-machine coordination ────────────────────────────────────────────────

CLAIM_TTL_HOURS = 6  # claims older than this are treated as stale/abandoned


def _get_machine_name(override: str | None = None) -> str:
    return override or socket.gethostname()


def _affinity_matches(item: dict, machine: str) -> bool:
    """Return True if this machine is allowed to run the experiment."""
    affinity = item.get("machine_affinity", "any")
    return affinity in ("any", None, "") or affinity == machine


def _is_stale_claim(claimed_by: dict) -> bool:
    """Return True if a claim is older than CLAIM_TTL_HOURS."""
    try:
        claimed_at = datetime.fromisoformat(claimed_by["claimed_at"])
        age = datetime.now(timezone.utc) - claimed_at
        return age.total_seconds() > CLAIM_TTL_HOURS * 3600
    except Exception:
        return True  # malformed -> treat as stale


def _git_undo_last_commit(repo: Path) -> None:
    """Undo the most recent local commit (pre-push rollback)."""
    subprocess.run(["git", "reset", "--soft", "HEAD~1"],
                   cwd=str(repo), capture_output=True)
    subprocess.run(["git", "reset", "HEAD", "experiment_queue.json"],
                   cwd=str(repo), capture_output=True)
    subprocess.run(["git", "checkout", "--", "experiment_queue.json"],
                   cwd=str(repo), capture_output=True)


def attempt_claim(queue_file: Path, queue_id: str, machine: str
                  ) -> str:  # "ok" | "already_claimed" | "error"
    """
    Atomically claim an experiment using git push as a mutex.

    Flow:
      1. git pull (get latest state)
      2. Check item is unclaimed + affinity matches
      3. Write claim, commit, push
      4. If push rejected (non-fast-forward) -> undo commit, return "already_claimed"
      5. On unrelated error -> return "error" (runner proceeds anyway)
    """
    repo = queue_file.parent
    try:
        # 1. Pull latest
        r = subprocess.run(["git", "pull", "--ff-only"],
                           cwd=str(repo), capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"[runner] claim pull warn ({queue_id}): {r.stderr.strip()}", flush=True)

        # 2. Load fresh queue
        data = json.loads(queue_file.read_text())
        item = next((i for i in data.get("items", []) if i["queue_id"] == queue_id), None)
        if item is None:
            return "error"

        existing = item.get("claimed_by")
        if existing and existing.get("machine") != machine and not _is_stale_claim(existing):
            return "already_claimed"

        if not _affinity_matches(item, machine):
            return "already_claimed"

        # 3. Write claim
        item["claimed_by"] = {"machine": machine, "claimed_at": now_utc()}
        item["status"] = "claimed"
        queue_file.write_text(json.dumps(data, indent=2))

        # 4. Commit + push
        subprocess.run(["git", "add", queue_file.name],
                       cwd=str(repo), capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", f"claim: {queue_id} -> {machine}"],
                       cwd=str(repo), capture_output=True, check=True)

        push = subprocess.run(["git", "push", "origin", "HEAD:main"],
                               cwd=str(repo), capture_output=True, text=True, timeout=30)

        if push.returncode == 0:
            return "ok"

        # Push rejected -- another machine got there first
        _git_undo_last_commit(repo)
        stderr = push.stderr.lower()
        if "non-fast-forward" in stderr or "rejected" in stderr:
            return "already_claimed"
        # Network or auth error -- don't block the experiment
        print(f"[runner] claim push error ({queue_id}): {push.stderr.strip()}", flush=True)
        return "error"

    except Exception as e:
        print(f"[runner] claim exception ({queue_id}): {e}", flush=True)
        try:
            _git_undo_last_commit(repo)
        except Exception:
            pass
        return "error"


def release_claim(queue_file: Path, queue_id: str, machine: str) -> None:
    """
    Release a claim on shutdown so another machine can pick up the experiment.
    Best-effort -- warns on failure but never raises.
    """
    repo = queue_file.parent
    try:
        subprocess.run(["git", "pull", "--ff-only"],
                       cwd=str(repo), capture_output=True, timeout=30)
        data = json.loads(queue_file.read_text())
        changed = False
        for item in data.get("items", []):
            if item["queue_id"] == queue_id:
                cb = item.get("claimed_by")
                if cb and cb.get("machine") == machine:
                    item["claimed_by"] = None
                    item["status"] = "pending"
                    changed = True
                break
        if not changed:
            return
        queue_file.write_text(json.dumps(data, indent=2))
        subprocess.run(["git", "add", queue_file.name],
                       cwd=str(repo), capture_output=True)
        subprocess.run(["git", "commit", "-m",
                        f"release claim: {queue_id} <- {machine} (shutdown)"],
                       cwd=str(repo), capture_output=True)
        subprocess.run(["git", "push", "origin", "HEAD:main"],
                       cwd=str(repo), capture_output=True, timeout=30)
        print(f"[runner] Released claim on {queue_id}", flush=True)
    except Exception as e:
        print(f"[runner] Release claim error ({queue_id}): {e}", flush=True)


# STUB: smart_assign ──────────────────────────────────────────────────────────
def smart_assign(items: list[dict], available_machines: list[str]) -> None:
    """
    TODO: Assign machine_affinity to queue items based on capabilities.

    Ideas for implementation:
    - Read a machines.json config mapping hostnames to GPU memory, estimated
      throughput (ms/episode), etc.
    - Assign GPU-heavy experiments to the machine with most VRAM.
    - Assign short-run experiments to whichever machine is currently idle.
    - Respect existing manual machine_affinity assignments.

    This would be called by a separate dispatch script, not the runner itself.
    For now, machine_affinity defaults to "any" and both machines run whatever
    they can claim first.
    """
    pass


# STUB: recover_stale_claims ──────────────────────────────────────────────────
def recover_stale_claims(queue_file: Path, machine: str) -> int:
    """
    TODO: Scan queue for stale claims from offline machines and reset them.

    Current behaviour: just logs any stale claims it finds.
    Full implementation would:
    - Check claimed_by.claimed_at against CLAIM_TTL_HOURS
    - Verify the claiming machine is unreachable (e.g. ping or last-seen file)
    - Reset item to pending + commit + push
    - Needs care to avoid two machines both recovering the same claim simultaneously
      (use attempt_claim pattern: try to push, back off if rejected).
    """
    try:
        data = json.loads(queue_file.read_text())
        stale = []
        for item in data.get("items", []):
            cb = item.get("claimed_by")
            if cb and cb.get("machine") != machine and _is_stale_claim(cb):
                stale.append((item["queue_id"], cb["machine"], cb["claimed_at"]))
        if stale:
            print(f"[runner] Stale claims detected (not yet auto-recovering): "
                  f"{[q for q, _, _ in stale]}", flush=True)
            print(f"[runner] To recover manually: set claimed_by=null in experiment_queue.json "
                  f"and push.", flush=True)
        return len(stale)
    except Exception as e:
        print(f"[runner] recover_stale_claims error: {e}", flush=True)
        return 0


# ─────────────────────────────────────────────────────────────────────────────

def find_default_status_path(machine: str = "unknown") -> Path:
    """Return per-machine status file path: runner_status/<machine>.json"""
    for candidate_dir in _REE_ASSEMBLY_STATUS_DIRS:
        if candidate_dir.parent.exists():
            candidate_dir.mkdir(parents=True, exist_ok=True)
            return candidate_dir / f"{machine}.json"
    # Fallback: local file in repo
    status_dir = REPO_ROOT / "runner_status"
    status_dir.mkdir(parents=True, exist_ok=True)
    return status_dir / f"{machine}.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


_write_status_lock = threading.Lock()


def write_status(status: dict, path: Path) -> None:
    with _write_status_lock:
        tmp = path.with_suffix(".tmp")
        status["last_updated"] = now_utc()
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(path)  # replace() is atomic on Unix and works on Windows (unlike rename)


def merge_peer_status(status_path: Path) -> set:
    """Merge all per-machine runner_status files into the monolithic runner_status.json.

    Reads every *.json file in the runner_status/ directory (one per machine),
    deduplicates by queue_id (preferring non-ERROR over ERROR for the same ID),
    and writes the combined completed list to the monolithic runner_status.json.

    Returns the set of all queue_ids present across all machines, so the caller
    can absorb them into completed_ids and prevent re-running peer experiments.

    Never raises -- logs warnings and returns empty set on any error.
    """
    status_dir = status_path.parent          # .../runner_status/
    monolithic = status_dir.parent / "runner_status.json"  # .../evidence/experiments/runner_status.json

    if not status_dir.is_dir():
        return set()

    all_completed: list = []
    seen_ids: set = set()

    for f in sorted(status_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"[runner] status sync: could not read {f.name}: {e}", flush=True)
            continue
        for entry in data.get("completed", []):
            qid = entry.get("queue_id", "")
            if not qid:
                continue
            if qid not in seen_ids:
                seen_ids.add(qid)
                all_completed.append(entry)
            elif entry.get("result") != "ERROR":
                # Prefer non-ERROR over ERROR for the same experiment
                for i, x in enumerate(all_completed):
                    if x.get("queue_id") == qid and x.get("result") == "ERROR":
                        all_completed[i] = entry
                        break

    if not all_completed:
        return seen_ids

    try:
        existing = json.loads(monolithic.read_text()) if monolithic.exists() else {}
    except Exception:
        existing = {}

    old_count = len(existing.get("completed", []))
    existing["schema_version"] = "v1"
    existing["completed"] = all_completed
    with _write_status_lock:
        tmp = monolithic.with_suffix(".tmp")
        existing["last_updated"] = now_utc()
        tmp.write_text(json.dumps(existing, indent=2))
        tmp.replace(monolithic)

    new_count = len(all_completed)
    if new_count != old_count:
        n_files = len(list(status_dir.glob("*.json")))
        print(f"[runner] status sync: {new_count} completed entries merged from "
              f"{n_files} machine file(s) -> runner_status.json "
              f"({new_count - old_count:+d})", flush=True)

    return seen_ids


def load_queue() -> dict:
    # Validate schema before loading -- raises SystemExit on errors so the
    # runner never silently skips malformed entries.
    try:
        from validate_queue import validate
        errors = validate(QUEUE_FILE)
        if errors:
            print(f"[runner] Queue validation FAILED -- {len(errors)} error(s):",
                  flush=True)
            for e in errors:
                print(f"[runner]   ERROR: {e}", flush=True)
            sys.exit(1)
    except ImportError:
        pass  # validator not present -- skip (graceful degradation)
    with open(QUEUE_FILE) as f:
        return json.load(f)


def load_script_timing() -> dict:
    if SCRIPT_TIMING_FILE.exists():
        try:
            return json.loads(SCRIPT_TIMING_FILE.read_text())
        except Exception:
            pass
    return {}


def save_script_timing(script: str, actual_secs: float, seeds: int, conditions: int, episodes: int) -> None:
    total_ep_cond = seeds * conditions * episodes
    if total_ep_cond <= 0:
        return
    actual_ms_per = round((actual_secs * 1000) / total_ep_cond, 1)
    timing = load_script_timing()
    timing[script] = actual_ms_per
    SCRIPT_TIMING_FILE.write_text(json.dumps(timing, indent=2))
    print(f"[runner] Calibration updated: {script} -> {actual_ms_per:.0f} ms/ep-cond", flush=True)


def estimate_minutes(item: dict, calibration: dict, script_timing: dict | None = None) -> float:
    seeds = item.get("seeds", 1)
    conditions = item.get("conditions", 1)
    episodes = item.get("episodes_per_run", 130)
    script = item.get("script", "")
    if script_timing and script in script_timing:
        ms_per = script_timing[script]
    else:
        ms_per = calibration.get("ms_per_episode_condition", 8000)
    return (seeds * conditions * episodes * ms_per) / 60_000


def build_initial_status(queue_data: dict, script_timing: dict | None = None) -> dict:
    calibration = queue_data.get("calibration", {})
    queue_items = []
    for item in queue_data.get("items", []):
        queue_items.append({
            "queue_id": item["queue_id"],
            "backlog_id": item.get("backlog_id", ""),
            "claim_id": item.get("claim_id", ""),
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "estimated_minutes": round(estimate_minutes(item, calibration, script_timing), 1),
            "status": item.get("status", "pending"),
            "status_reason": item.get("status_reason", ""),
            "ree_version": "v3",
        })
    return {
        "schema_version": "v1",
        "runner_pid": os.getpid(),
        "runner_started_at": now_utc(),
        "last_updated": now_utc(),
        "idle": False,
        "current": None,
        "queue": queue_items,
        "completed": [],
    }


def run_experiment(item: dict, status: dict, status_path: Path, calibration: dict,
                   script_timing: dict | None = None,
                   proc_ref: list | None = None) -> dict:
    """Run a single experiment script as a subprocess.

    proc_ref: if provided, the active Popen object is stored as proc_ref[0] immediately
    after launch and cleared on completion.  The signal handler uses this to kill the
    subprocess on a forced stop.
    """
    script = REPO_ROOT / item["script"]
    raw_args = item.get("args", [])
    if isinstance(raw_args, str):
        raw_args = shlex.split(raw_args)
    args = [sys.executable, "-u", str(script)] + raw_args

    seeds = item.get("seeds", 1)
    conditions = item.get("conditions", 1)
    total_runs = max(1, seeds * conditions)
    episodes_per_run = item.get("episodes_per_run", 130)

    runs_done = 0
    current_run_label = "starting..."
    episodes_in_run = 0
    recent_lines: list[str] = []
    run_end_times: list[float] = []

    started_at = time.monotonic()
    started_at_utc = now_utc()

    def overall_pct() -> float:
        run_frac = (runs_done + episodes_in_run / max(episodes_per_run, 1)) / max(total_runs, 1)
        return round(run_frac * 100, 1)

    def seconds_remaining() -> float:
        elapsed = time.monotonic() - started_at
        pct = overall_pct()
        static_secs = estimate_minutes(item, calibration, script_timing) * 60
        if pct <= 0:
            return static_secs
        if run_end_times:
            avg_secs_per_run = run_end_times[-1] / runs_done
            remaining = total_runs - runs_done - episodes_in_run / max(episodes_per_run, 1)
            return max(0, avg_secs_per_run * remaining)
        live_total = elapsed / (pct / 100)
        blend = min(pct / 20.0, 1.0)
        total_estimated = (1 - blend) * static_secs + blend * live_total
        return max(0, total_estimated - elapsed)

    def update_status_current():
        status["current"] = {
            "queue_id": item["queue_id"],
            "backlog_id": item.get("backlog_id", ""),
            "claim_id": item.get("claim_id", ""),
            "title": item.get("title", ""),
            "description": item.get("description", ""),
            "script": item["script"],
            "started_at": started_at_utc,
            "progress": {
                "run_label": current_run_label,
                "runs_done": runs_done,
                "runs_total": total_runs,
                "episodes_done": episodes_in_run,
                "episodes_total": episodes_per_run,
                "overall_pct": overall_pct(),
            },
            "seconds_elapsed": round(time.monotonic() - started_at),
            "seconds_remaining": round(seconds_remaining()),
            "recent_lines": recent_lines[-5:],
            "ree_version": "v3",
        }
        for qi in status["queue"]:
            if qi["queue_id"] == item["queue_id"]:
                qi["status"] = "running"
        write_status(status, status_path)

    est = item.get('estimated_minutes')
    est_str = f" -- est. {est} min" if est else ""
    print(f"[runner] Starting: {item.get('title', item['queue_id'])} ({item['queue_id']}){est_str}", flush=True)
    print(f"[runner] Command: {' '.join(str(a) for a in args)}", flush=True)

    last_write = time.monotonic()
    last_bar_pct = 0
    update_status_current()

    def print_progress_bar():
        pct = overall_pct()
        remaining = seconds_remaining()
        width = 30
        filled = int(width * pct / 100)
        bar = '█' * filled + '░' * (width - filled)
        if remaining > 90:
            time_str = f"~{int(remaining / 60)} min remaining"
        elif remaining > 0:
            time_str = f"~{int(remaining)} sec remaining"
        else:
            time_str = "finishing..."
        print(f"[runner] {bar} {pct:.0f}% | {time_str}", flush=True)

    result_info = {
        "result": "UNKNOWN",
        "result_summary": "",
        "started_at": started_at_utc,
        "completed_at": "",
        "output_file": "",
        "actual_secs": 0.0,
    }

    try:
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc_ref is not None:
            proc_ref.clear()
            proc_ref.append(proc)

        _hb_stop = threading.Event()
        def _heartbeat():
            while not _hb_stop.wait(timeout=STATUS_WRITE_INTERVAL):
                update_status_current()
        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        for line in proc.stdout:
            line = line.rstrip()
            print(line, flush=True)

            # Progress: Condition label from V3 experiments
            m = RE_SEED_CONDITION.search(line)
            if m:
                current_run_label = f"Seed {m.group(1)} / {m.group(2)}"
                episodes_in_run = 0

            # Training progress
            m = RE_TRAIN_PROGRESS.search(line)
            if m:
                episodes_in_run = int(m.group(1))

            m = RE_EP_PROGRESS.search(line)
            if m:
                episodes_in_run = int(m.group(1))
                episodes_per_run = int(m.group(2))  # use denominator for accurate pct

            # Progress bar -- print every 20% of progress
            if episodes_in_run > 0:
                pct_milestone = (int(overall_pct()) // 20) * 20
                if pct_milestone > last_bar_pct:
                    print_progress_bar()
                    last_bar_pct = pct_milestone

            # Run completion: V3 verdict patterns
            for pat in RE_RUN_DONE_PATTERNS:
                if pat.search(line):
                    run_end_times.append(time.monotonic() - started_at)
                    runs_done += 1
                    episodes_in_run = episodes_per_run
                    print_progress_bar()
                    last_bar_pct = 100
                    break

            m = RE_STATUS_LINE.match(line)
            if m:
                run_end_times.append(time.monotonic() - started_at)
                runs_done = max(runs_done, 1)

            m = RE_SAVED_TO.search(line)
            if m:
                result_info["output_file"] = m.group(1).strip()

            # Capture verdict
            _exq_m = RE_EXQ_VERDICT.search(line)
            _done_m = RE_DONE_OUTCOME.search(line)
            if "verdict: PASS" in line or (RE_STATUS_LINE.match(line) and "PASS" in line) \
                    or (_exq_m and _exq_m.group(1) == "PASS") \
                    or (_done_m and _done_m.group(1) == "PASS"):
                result_info["result"] = "PASS"
            elif "verdict: FAIL" in line or (RE_STATUS_LINE.match(line) and "FAIL" in line) \
                    or (_exq_m and _exq_m.group(1) == "FAIL") \
                    or (_done_m and _done_m.group(1) == "FAIL"):
                result_info["result"] = "FAIL"

            stripped = line.strip()
            if stripped:
                recent_lines.append(stripped)
                if len(recent_lines) > 20:
                    recent_lines.pop(0)

            if time.monotonic() - last_write >= STATUS_WRITE_INTERVAL:
                update_status_current()
                last_write = time.monotonic()

        _hb_stop.set()
        proc.wait()
        exit_code = proc.returncode

        if exit_code != 0 and result_info["result"] == "UNKNOWN":
            result_info["result"] = "ERROR"
            result_info["result_summary"] = f"Non-zero exit code {exit_code}"

    except Exception as exc:
        result_info["result"] = "ERROR"
        result_info["result_summary"] = str(exc)
        print(f"[runner] ERROR running {item['queue_id']}: {exc}", flush=True)

    finally:
        # Always clear proc_ref so the signal handler doesn't try to kill a finished process.
        if proc_ref is not None:
            proc_ref.clear()

    result_info["completed_at"] = now_utc()
    result_info["actual_secs"] = round(time.monotonic() - started_at, 1)

    if not result_info["result_summary"]:
        summary_lines = [l for l in recent_lines if any(
            kw in l for kw in [
                "calibration_gap", "selectivity_margin", "verdict", "PASS", "FAIL",
                "corr", "Criteria met", "harm", "causal_sig",
            ]
        )]
        result_info["result_summary"] = " | ".join(summary_lines[-3:])

    return result_info


def main():
    parser = argparse.ArgumentParser(description="REE-V3 Experiment Runner")
    parser.add_argument("--status-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Poll experiment_queue.json every --loop-interval seconds after queue exhaustion.",
    )
    parser.add_argument("--loop-interval", type=int, default=60, metavar="SECONDS")
    parser.add_argument(
        "--auto-sync",
        action="store_true",
        help="Git-pull queue repo before each batch; git-push results to REE_assembly after. "
             "Also enables git-based experiment claiming (multi-machine coordination). "
             "Useful for remote PC setups.",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=None,
        help="Machine identity for experiment claiming (default: hostname). "
             "Use 'any' to disable affinity filtering.",
    )
    args = parser.parse_args()

    machine = _get_machine_name(args.machine)
    status_path = args.status_file or find_default_status_path(machine)
    ree_assembly_path = find_ree_assembly_path()
    print(f"[runner] Status file: {status_path}", flush=True)
    print(f"[runner] Queue file:  {QUEUE_FILE}", flush=True)
    print(f"[runner] Machine identity: {machine}", flush=True)
    if args.auto_sync:
        if ree_assembly_path:
            print(f"[runner] Auto-sync: ON (REE_assembly: {ree_assembly_path})", flush=True)
            git_pull(REPO_ROOT, "ree-v3")
            git_pull(ree_assembly_path, "REE_assembly")
        else:
            print("[runner] Auto-sync: ON but REE_assembly not found -- sync disabled", flush=True)
        recover_stale_claims(QUEUE_FILE, machine)

    # Merge all per-machine status files into monolithic runner_status.json (always,
    # not just in auto-sync mode) so the explorer has an up-to-date combined view.
    _peer_ids = merge_peer_status(status_path)

    PID_FILE.write_text(str(os.getpid()))

    # Track active claim so signal handler can release it
    _current_claim: list[str] = []  # 0 or 1 elements (mutable container for closure)

    # Graceful-drain state: set by first stop signal; second signal force-kills.
    # _current_proc holds the active experiment subprocess so the second signal can kill it.
    _drain_flag: list[bool] = []           # non-empty -> drain requested
    _current_proc: list[subprocess.Popen] = []  # 0 or 1 elements

    def _do_immediate_exit() -> None:
        """Final cleanup steps shared by both force-exit and post-drain exit."""
        if args.auto_sync and _current_claim:
            release_claim(QUEUE_FILE, _current_claim[0], machine)
        if status_path.exists():
            try:
                s = json.loads(status_path.read_text())
                s["idle"] = True
                s["draining"] = False
                s["current"] = None
                s["runner_pid"] = None
                for qi in s.get("queue", []):
                    if qi.get("status") == "running":
                        qi["status"] = "pending"
                write_status(s, status_path)
            except Exception:
                pass
        if PID_FILE.exists():
            PID_FILE.unlink()

    def handle_signal(sig, frame):
        if _drain_flag:
            # Second signal while already draining -- force exit immediately.
            print(f"\n[runner] Second signal -- force-stopping now.", flush=True)
            if _current_proc:
                try:
                    _current_proc[0].kill()
                except Exception:
                    pass
            _do_immediate_exit()
            sys.exit(0)

        # First stop signal: request graceful drain.
        print(f"\n[runner] Caught signal {sig} -- will stop after current experiment finishes.",
              flush=True)
        print("[runner] Send signal again to force-stop immediately.", flush=True)
        _drain_flag.append(True)

        # Write draining indicator so the Explorer can show the correct state.
        try:
            if status_path.exists():
                s = json.loads(status_path.read_text())
                s["draining"] = True
                write_status(s, status_path)
        except Exception:
            pass

        # If no experiment is currently running, exit immediately.
        if not _current_claim:
            _do_immediate_exit()
            sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    if sys.platform != "win32":  # SIGTERM not available on Windows
        signal.signal(signal.SIGTERM, handle_signal)

    queue_data = load_queue()
    calibration = queue_data.get("calibration", {})
    items = queue_data.get("items", [])
    items.sort(key=lambda x: x.get("priority", 0), reverse=True)
    script_timing = load_script_timing()

    # Preserve existing completed runs from per-machine file
    existing_completed = []
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
            existing_completed = existing.get("completed", [])
        except Exception:
            pass

    # Migration: seed from old monolithic runner_status.json if per-machine file is empty
    if not existing_completed:
        old_monolithic = status_path.parent.parent / "runner_status.json"
        if old_monolithic.exists():
            try:
                old_data = json.loads(old_monolithic.read_text())
                old_completed = old_data.get("completed", [])
                # Take entries completed by this machine (or unattributed ones)
                existing_completed = [
                    c for c in old_completed
                    if c.get("completed_by", machine) == machine
                       or not c.get("completed_by")
                ]
                if existing_completed:
                    print(f"[runner] Migrated {len(existing_completed)} completed entries "
                          f"from old runner_status.json", flush=True)
            except Exception:
                pass

    status = build_initial_status(queue_data, script_timing)
    status["completed"] = existing_completed
    write_status(status, status_path)

    if args.dry_run:
        print(f"[runner] Dry run -- V3 queue (machine: {machine}):")
        for item in items:
            script = REPO_ROOT / item["script"]
            runnable = script.exists()
            mins = estimate_minutes(item, calibration, script_timing)
            affinity = item.get("machine_affinity", "any")
            claim = item.get("claimed_by")
            claim_str = f" [claimed:{claim['machine']}]" if claim else ""
            mine = "✓" if _affinity_matches(item, machine) else f"✗({affinity})"
            print(f"  {mine} {item['queue_id']} {item.get('claim_id', ''):12s} ~{mins:.0f}min  "
                  f"{'READY' if runnable else 'NEEDS_SCRIPT'}: {item.get('title', item['queue_id'])}{claim_str}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return

    print(f"[runner] PID {os.getpid()} -- {len(items)} experiments queued", flush=True)
    if args.loop:
        print(f"[runner] Loop mode: polling every {args.loop_interval}s", flush=True)

    # Include peer-machine completed IDs so we never re-run an experiment
    # another machine already finished (extra safety net beyond queue removal).
    completed_ids = {c["queue_id"] for c in existing_completed} | _peer_ids

    # Prune already-completed items from queue display
    status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] not in completed_ids]
    write_status(status, status_path)

    # Collect output files written during this pass so git_push_results can
    # stage them selectively instead of sweeping evidence/experiments/.
    _result_files_this_pass: list[str] = []

    while True:
        ran_any = False

        for item in items:
            queue_id = item["queue_id"]

            if queue_id in completed_ids:
                continue

            # Skip experiments that previously failed (scientific FAIL -- not retried automatically).
            # On first encounter: log clearly, move to completed list, and remove from queue file
            # so the explorer queue shows only actionable (pending) items.
            if item.get("status") == "failed":
                if queue_id not in completed_ids:
                    failure_reason = item.get("failure_reason", "")
                    reason_short = (failure_reason[:80] + "...") if len(failure_reason) > 80 else failure_reason
                    print(f"[runner] Skipping {queue_id} -- previously failed"
                          f"{': ' + reason_short if reason_short else ''}", flush=True)
                    completed_entry = {
                        "queue_id": queue_id,
                        "backlog_id": item.get("backlog_id", ""),
                        "claim_id": item.get("claim_id", ""),
                        "title": item.get("title", ""),
                        "description": item.get("description", ""),
                        "result": "FAIL",
                        "result_summary": failure_reason,
                        "started_at": "",
                        "completed_at": item.get("failed_at", ""),
                        "output_file": "",
                        "completed_by": machine,
                    }
                    status["completed"].append(completed_entry)
                    status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] != queue_id]
                    write_status(status, status_path)
                    try:
                        qdata = json.loads(QUEUE_FILE.read_text())
                        qdata["items"] = [qi for qi in qdata.get("items", [])
                                          if qi.get("queue_id") != queue_id]
                        QUEUE_FILE.write_text(json.dumps(qdata, indent=2) + "\n")
                        if args.auto_sync:
                            git_push_queue()
                    except Exception as _qe:
                        print(f"[runner] warn: could not remove {queue_id} from queue file: {_qe}",
                              flush=True)
                completed_ids.add(queue_id)
                continue

            # Skip experiments assigned to a different machine
            if not _affinity_matches(item, machine):
                print(f"[runner] Skipping {queue_id} -- affinity={item.get('machine_affinity')} "
                      f"(this machine: {machine})", flush=True)
                continue

            # Skip experiments already claimed by another active machine
            existing_claim = item.get("claimed_by")
            if (existing_claim
                    and existing_claim.get("machine") != machine
                    and not _is_stale_claim(existing_claim)):
                print(f"[runner] Skipping {queue_id} -- claimed by "
                      f"{existing_claim['machine']}", flush=True)
                continue

            script = REPO_ROOT / item["script"]
            if not script.exists():
                print(f"[runner] Skipping {queue_id} -- script not found: {item['script']}", flush=True)
                for qi in status["queue"]:
                    if qi["queue_id"] == queue_id:
                        qi["status"] = "needs_script"
                write_status(status, status_path)
                continue

            # In auto-sync mode, use git claim as mutex before running
            if args.auto_sync:
                claim_result = attempt_claim(QUEUE_FILE, queue_id, machine)
                if claim_result == "already_claimed":
                    print(f"[runner] {queue_id} -- claim lost to another machine, skipping",
                          flush=True)
                    continue
                if claim_result == "error":
                    print(f"[runner] {queue_id} -- claim push failed (network?), "
                          f"running anyway", flush=True)
                # "ok" or "error" -> proceed; track for signal handler
                _current_claim.clear()
                _current_claim.append(queue_id)

            try:
                result = run_experiment(item, status, status_path, calibration, script_timing,
                                        proc_ref=_current_proc)
            except Exception as _run_exc:
                # Unexpected exception escaping run_experiment -- treat as ERROR and continue
                print(f"[runner] UNEXPECTED ERROR in {queue_id}: {_run_exc}", flush=True)
                completed_ids.add(queue_id)
                status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] != queue_id]
                status["current"] = None
                write_status(status, status_path)
                ran_any = True
                _current_claim.clear()
                continue
            ran_any = True
            _current_claim.clear()  # no longer running this experiment

            # Collect output file for selective git staging
            if result.get("output_file"):
                _result_files_this_pass.append(result["output_file"])

            if result["result"] not in ("ERROR", "UNKNOWN") and result.get("actual_secs"):
                save_script_timing(
                    item["script"],
                    result["actual_secs"],
                    item.get("seeds", 1),
                    item.get("conditions", 1),
                    item.get("episodes_per_run", 130),
                )
                script_timing = load_script_timing()

            if result["result"] == "ERROR":
                # Script crashed -- move to completed (so it appears in the explorer
                # completed list) and remove from queue, just like a finished experiment.
                # The ERROR result label distinguishes it from PASS/FAIL.
                completed_entry = {
                    "queue_id": queue_id,
                    "backlog_id": item.get("backlog_id", ""),
                    "claim_id": item.get("claim_id", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "result": "ERROR",
                    "result_summary": result.get("result_summary", ""),
                    "started_at": result.get("started_at", ""),
                    "completed_at": result["completed_at"],
                    "output_file": result.get("output_file", ""),
                    "completed_by": machine,
                    "actual_secs": result.get("actual_secs", 0),
                }
                status["completed"].append(completed_entry)
                completed_ids.add(queue_id)
                status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] != queue_id]
                status["current"] = None
                write_status(status, status_path)
                try:
                    qdata = json.loads(QUEUE_FILE.read_text())
                    qdata["items"] = [qi for qi in qdata.get("items", [])
                                      if qi.get("queue_id") != queue_id]
                    QUEUE_FILE.write_text(json.dumps(qdata, indent=2) + "\n")
                    if args.auto_sync:
                        git_push_queue()
                except Exception as _qe:
                    print(f"[runner] warn: could not remove {queue_id} from queue file: {_qe}",
                          flush=True)
                print(f"[runner] ERROR: {queue_id} -- moved to completed, continuing", flush=True)
                continue

            if result["result"] == "FAIL":
                # Scientific FAIL -- move to completed (with FAIL label) and remove from queue,
                # same treatment as ERROR. Failed experiments should not accumulate as dead weight
                # in the queue; they appear in the explorer completed list for review/redesign.
                failure_reason = result.get("result_summary", "")
                completed_entry = {
                    "queue_id": queue_id,
                    "backlog_id": item.get("backlog_id", ""),
                    "claim_id": item.get("claim_id", ""),
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "result": "FAIL",
                    "result_summary": failure_reason,
                    "started_at": result.get("started_at", ""),
                    "completed_at": result["completed_at"],
                    "output_file": result.get("output_file", ""),
                    "completed_by": machine,
                    "actual_secs": result.get("actual_secs", 0),
                }
                status["completed"].append(completed_entry)
                completed_ids.add(queue_id)
                status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] != queue_id]
                status["current"] = None
                write_status(status, status_path)
                try:
                    qdata = json.loads(QUEUE_FILE.read_text())
                    qdata["items"] = [qi for qi in qdata.get("items", [])
                                      if qi.get("queue_id") != queue_id]
                    QUEUE_FILE.write_text(json.dumps(qdata, indent=2) + "\n")
                    if args.auto_sync:
                        git_push_queue()
                except Exception as _qe:
                    print(f"[runner] warn: could not remove {queue_id} from queue file: {_qe}",
                          flush=True)
                print(f"[runner] FAIL: {queue_id} -- moved to completed, continuing to next",
                      flush=True)
                continue

            completed_entry = {
                "queue_id": queue_id,
                "backlog_id": item.get("backlog_id", ""),
                "claim_id": item.get("claim_id", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "result": result["result"],
                "result_summary": result["result_summary"],
                "started_at": result.get("started_at", ""),
                "completed_at": result["completed_at"],
                "output_file": result.get("output_file", ""),
                "completed_by": machine,
                "actual_secs": result.get("actual_secs", 0),
            }
            status["completed"].append(completed_entry)
            completed_ids.add(queue_id)

            status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] != queue_id]
            status["current"] = None

            write_status(status, status_path)

            # Remove completed item from queue file -- runner_status.json is the
            # authoritative record of what has run; queue file should only contain
            # pending work to avoid silent accumulation of stale entries.
            try:
                qdata = json.loads(QUEUE_FILE.read_text())
                qdata["items"] = [qi for qi in qdata.get("items", [])
                                   if qi.get("queue_id") != queue_id]
                QUEUE_FILE.write_text(json.dumps(qdata, indent=2) + "\n")
                if args.auto_sync:
                    git_push_queue()
            except Exception as _qe:
                print(f"[runner] warn: could not update queue file for {queue_id}: {_qe}", flush=True)

            print(f"[runner] Done: {queue_id} -- {result['result']}", flush=True)

            # Graceful drain: if a stop was requested, exit after this experiment.
            if _drain_flag:
                print("[runner] Drain complete -- stopping as requested.", flush=True)
                break

        if _drain_flag:
            break  # drain requested while no experiment was running (between items)

        if not args.loop:
            break

        if args.auto_sync and ran_any and ree_assembly_path:
            git_push_results(ree_assembly_path, _result_files_this_pass or None)
            _result_files_this_pass.clear()

        status["idle"] = True
        status["current"] = None
        write_status(status, status_path)
        if ran_any:
            print(f"[runner] Pass complete. Waiting {args.loop_interval}s ...", flush=True)
        else:
            print(f"[runner] No new items. Waiting {args.loop_interval}s ...", flush=True)

        time.sleep(args.loop_interval)

        if args.auto_sync and ree_assembly_path:
            git_pull(REPO_ROOT, "ree-v3")

        # Re-merge peer status after pull so monolithic file stays current and
        # completed_ids absorbs anything another machine finished since last pass.
        completed_ids |= merge_peer_status(status_path)

        queue_data = load_queue()
        calibration = queue_data.get("calibration", {})
        items = queue_data.get("items", [])
        items.sort(key=lambda x: x.get("priority", 0), reverse=True)

        new_pending = [i for i in items if i["queue_id"] not in completed_ids]
        if new_pending:
            print(f"[runner] Found {len(new_pending)} new item(s): "
                  f"{[i['queue_id'] for i in new_pending]}", flush=True)
            new_queue_display = []
            for i in new_pending:
                new_queue_display.append({
                    "queue_id": i["queue_id"],
                    "backlog_id": i.get("backlog_id", ""),
                    "claim_id": i.get("claim_id", ""),
                    "title": i.get("title", ""),
                    "description": i.get("description", ""),
                    "estimated_minutes": round(estimate_minutes(i, calibration, script_timing), 1),
                    "status": i.get("status", "pending"),
                    "status_reason": i.get("status_reason", ""),
                    "ree_version": "v3",
                })
            status["queue"] = new_queue_display
            status["idle"] = False
            write_status(status, status_path)

    status["idle"] = True
    status["draining"] = False
    status["current"] = None
    status["runner_pid"] = None
    write_status(status, status_path)
    if _drain_flag:
        print("[runner] Graceful drain complete. Exiting.", flush=True)
    else:
        print("[runner] Queue exhausted. Runner idle.", flush=True)

    if args.auto_sync and ree_assembly_path:
        git_push_results(ree_assembly_path, _result_files_this_pass or None)

    if PID_FILE.exists():
        PID_FILE.unlink()


if __name__ == "__main__":
    main()
