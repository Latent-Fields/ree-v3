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
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
QUEUE_FILE = REPO_ROOT / "experiment_queue.json"
PID_FILE = REPO_ROOT / "runner.pid"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "experiments"
SCRIPT_TIMING_FILE = REPO_ROOT / "script_timing.json"

# Auto-detect REE_assembly runner_status.json (shared with V2 explorer)
_REE_ASSEMBLY_CANDIDATES = [
    REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / "runner_status.json",
    Path.home() / "Documents" / "GitHub" / "REE_Working" / "REE_assembly" / "evidence" / "experiments" / "runner_status.json",
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
RE_SAVED_TO = re.compile(r'Result written to:\s+(.+)')


def find_default_status_path() -> Path:
    for candidate in _REE_ASSEMBLY_CANDIDATES:
        if candidate.parent.exists():
            return candidate
    return REPO_ROOT / "runner_status.json"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


_write_status_lock = threading.Lock()


def write_status(status: dict, path: Path) -> None:
    with _write_status_lock:
        tmp = path.with_suffix(".tmp")
        status["last_updated"] = now_utc()
        tmp.write_text(json.dumps(status, indent=2))
        tmp.rename(path)


def load_queue() -> dict:
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
    print(f"[runner] Calibration updated: {script} → {actual_ms_per:.0f} ms/ep-cond", flush=True)


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


def run_experiment(item: dict, status: dict, status_path: Path, calibration: dict, script_timing: dict | None = None) -> dict:
    script = REPO_ROOT / item["script"]
    args = [sys.executable, "-u", str(script)] + item.get("args", [])

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

    print(f"[runner] Starting: {item['title']} ({item['queue_id']})", flush=True)
    print(f"[runner] Command: {' '.join(str(a) for a in args)}", flush=True)

    last_write = time.monotonic()
    update_status_current()

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

            # Run completion: V3 verdict patterns
            for pat in RE_RUN_DONE_PATTERNS:
                if pat.search(line):
                    run_end_times.append(time.monotonic() - started_at)
                    runs_done += 1
                    episodes_in_run = episodes_per_run
                    break

            m = RE_STATUS_LINE.match(line)
            if m:
                run_end_times.append(time.monotonic() - started_at)
                runs_done = max(runs_done, 1)

            m = RE_SAVED_TO.search(line)
            if m:
                result_info["output_file"] = m.group(1).strip()

            # Capture verdict
            if "verdict: PASS" in line or (RE_STATUS_LINE.match(line) and "PASS" in line):
                result_info["result"] = "PASS"
            elif "verdict: FAIL" in line or (RE_STATUS_LINE.match(line) and "FAIL" in line):
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
    args = parser.parse_args()

    status_path = args.status_file or find_default_status_path()
    print(f"[runner] Status file: {status_path}", flush=True)
    print(f"[runner] Queue file:  {QUEUE_FILE}", flush=True)

    PID_FILE.write_text(str(os.getpid()))

    def handle_signal(sig, frame):
        print(f"\n[runner] Caught signal {sig}, shutting down.", flush=True)
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text())
                status["idle"] = True
                status["current"] = None
                status["runner_pid"] = None
                for qi in status.get("queue", []):
                    if qi.get("status") == "running":
                        qi["status"] = "pending"
                write_status(status, status_path)
            except Exception:
                pass
        if PID_FILE.exists():
            PID_FILE.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    queue_data = load_queue()
    calibration = queue_data.get("calibration", {})
    items = queue_data.get("items", [])
    script_timing = load_script_timing()

    # Preserve existing completed runs (V2 + previous V3 runs)
    existing_completed = []
    if status_path.exists():
        try:
            existing = json.loads(status_path.read_text())
            existing_completed = existing.get("completed", [])
        except Exception:
            pass

    status = build_initial_status(queue_data, script_timing)
    status["completed"] = existing_completed
    write_status(status, status_path)

    if args.dry_run:
        print("[runner] Dry run — V3 queue:")
        for item in items:
            script = REPO_ROOT / item["script"]
            runnable = script.exists()
            mins = estimate_minutes(item, calibration, script_timing)
            print(f"  {item['queue_id']} {item['claim_id']:12s} ~{mins:.0f}min  "
                  f"{'READY' if runnable else 'NEEDS_SCRIPT'}: {item['title']}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        return

    print(f"[runner] PID {os.getpid()} — {len(items)} experiments queued", flush=True)
    if args.loop:
        print(f"[runner] Loop mode: polling every {args.loop_interval}s", flush=True)

    completed_ids = {c["queue_id"] for c in existing_completed}

    # Prune already-completed items from queue display
    status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] not in completed_ids]
    write_status(status, status_path)

    while True:
        ran_any = False

        for item in items:
            queue_id = item["queue_id"]

            if queue_id in completed_ids:
                continue

            script = REPO_ROOT / item["script"]
            if not script.exists():
                print(f"[runner] Skipping {queue_id} — script not found: {item['script']}", flush=True)
                for qi in status["queue"]:
                    if qi["queue_id"] == queue_id:
                        qi["status"] = "needs_script"
                write_status(status, status_path)
                continue

            result = run_experiment(item, status, status_path, calibration, script_timing)
            ran_any = True

            if result["result"] not in ("ERROR", "UNKNOWN") and result.get("actual_secs"):
                save_script_timing(
                    item["script"],
                    result["actual_secs"],
                    item.get("seeds", 1),
                    item.get("conditions", 1),
                    item.get("episodes_per_run", 130),
                )
                script_timing = load_script_timing()

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
            }
            status["completed"].append(completed_entry)
            completed_ids.add(queue_id)

            status["queue"] = [qi for qi in status["queue"] if qi["queue_id"] != queue_id]
            status["current"] = None

            write_status(status, status_path)
            print(f"[runner] Done: {queue_id} — {result['result']}", flush=True)

        if not args.loop:
            break

        status["idle"] = True
        status["current"] = None
        write_status(status, status_path)
        if ran_any:
            print(f"[runner] Pass complete. Waiting {args.loop_interval}s …", flush=True)
        else:
            print(f"[runner] No new items. Waiting {args.loop_interval}s …", flush=True)

        time.sleep(args.loop_interval)

        queue_data = load_queue()
        calibration = queue_data.get("calibration", {})
        items = queue_data.get("items", [])

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
                    "status": "pending",
                    "status_reason": i.get("status_reason", ""),
                    "ree_version": "v3",
                })
            status["queue"] = new_queue_display
            status["idle"] = False
            write_status(status, status_path)

    status["idle"] = True
    status["current"] = None
    status["runner_pid"] = None
    write_status(status, status_path)
    print("[runner] Queue exhausted. Runner idle.", flush=True)

    if PID_FILE.exists():
        PID_FILE.unlink()


if __name__ == "__main__":
    main()
