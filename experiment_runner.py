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

# Coordinator shadow shim. Env-gated: a hard no-op unless
# COORDINATION_MODE=shadow. Import is guarded so a missing or broken shim
# can never stop the runner (the live git path must stay byte-identical).
try:
    import coordinator_client
except Exception:  # pragma: no cover -- shim must never break the runner
    class _NoCoordClient:
        def __getattr__(self, _name):
            return lambda *a, **k: None
    coordinator_client = _NoCoordClient()


# Ensure UTF-8 output on Windows (default cp1252 breaks -> and other Unicode in experiment scripts)
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Hoisted to module top-level so the _push_remote_heartbeat closure inside
# run_experiment() resolves _rrc via module globals. Previously this import
# lived only inside main()'s local scope, which caused a NameError every
# heartbeat tick once run_experiment started (the closure runs on its own
# thread and never sees main()'s locals).
try:
    import runner_remote_control as _rrc
    _rrc_import_error = None
except Exception as _rrc_exc:
    _rrc = None
    _rrc_import_error = _rrc_exc

try:
    import runner_checkpoint as _rckpt
except Exception as _rckpt_exc:
    _rckpt = None
    _rckpt_import_error = _rckpt_exc
else:
    _rckpt_import_error = None

REPO_ROOT = Path(__file__).resolve().parent
QUEUE_FILE = REPO_ROOT / "experiment_queue.json"
PID_FILE = REPO_ROOT / "runner.pid"
EVIDENCE_DIR = REPO_ROOT / "evidence" / "experiments"
SCRIPT_TIMING_FILE = REPO_ROOT / "script_timing.json"

# Runner-conformance sentinel directory. Each experiment subprocess writes
# <SIGNAL_DIR>/<queue_id>.json via experiment_protocol.emit_outcome(); the
# runner reads it after subprocess exit. See ree-v3/experiment_protocol.py.
RUNNER_SIGNAL_SUBPATH = Path("evidence") / "experiments" / "_runner_signals"

# Auto-detect REE_assembly runner_status directory (per-machine files)
_REE_ASSEMBLY_STATUS_DIRS = [
    REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / "runner_status",
    Path.home() / "REE_Working" / "REE_assembly" / "evidence" / "experiments" / "runner_status",
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
RE_BARE_OUTCOME = re.compile(r'(?im)^outcome:\s+(PASS|FAIL)\b')
RE_EXQ_BANNER = re.compile(r'===\s+(?:V3-)?EXQ-[\w-]+\s+(PASS|FAIL)\s*===')
RE_EXQ_DASHED_OUTCOME = re.compile(
    r'(?:V3-)?EXQ-[\w-]+\s+\([^)]+\)\s+--\s+(PASS|FAIL)\s+in'
)
RE_SAVED_TO = re.compile(r'Result (?:pack )?written to:?\s+(.+)')


def _result_manifest_exists(result: dict) -> bool:
    """Return True only when a PASS/FAIL result names an existing manifest."""
    manifest = result.get("output_file")
    if not isinstance(manifest, str) or not manifest.strip():
        return False
    return Path(manifest).is_file()


def _classify_no_sentinel_result(stdout_result: str, exit_code: int) -> tuple[str, str | None]:
    """Classify a completed run that produced NO runner sentinel.

    Per the experiment_protocol contract a missing sentinel means the script
    never reached emit_outcome (crash / kill / un-retrofitted legacy script).

    - Non-zero exit -> ERROR, ALWAYS. A process that dies mid-run may have
      already printed partial `verdict:`/`outcome:` lines on stdout (e.g.
      V3-EXQ-624 crashed in ARM_2 after ARM_0/ARM_1 printed `verdict: PASS`);
      that stdout-derived verdict must never be trusted, or the crashed item
      is recorded as PASS and left in the queue as a fleet crash-magnet that
      any worker re-claims and re-crashes (incident 2026-06-02).
    - Clean exit (0) + stdout PASS/FAIL -> trust stdout (legacy un-retrofitted
      script). Returns (stdout_result, None); the caller prints the retrofit
      NOTE and keeps its own summary.
    - Clean exit (0) + no stdout verdict -> ERROR.

    Returns (result, summary). summary is None ONLY for the trusted legacy
    path; otherwise it is a populated ERROR summary string.
    """
    if exit_code != 0:
        return "ERROR", (
            f"Non-zero exit code {exit_code}; no runner sentinel "
            f"(stdout-derived {stdout_result!r} not trusted on crash)"
        )
    if stdout_result in ("PASS", "FAIL"):
        return stdout_result, None
    return "ERROR", "No runner sentinel emitted and no PASS/FAIL on stdout"


def find_ree_assembly_path() -> Path | None:
    """Locate the REE_assembly repo (for git auto-sync pushes)."""
    candidates = [
        REPO_ROOT.parent / "REE_assembly",
        Path.home() / "REE_Working" / "REE_assembly",
    ]
    for c in candidates:
        if c.is_dir() and (c / "evidence" / "experiments").is_dir():
            return c
    return None


# Paths that the worker writes locally but where origin is the canonical
# source of truth under Phase 3 (the hub's sync_daemon writers publish the
# authoritative version). When `git pull --rebase --autostash` leaves UU
# markers on one of these because the worker's pre-pull mutation conflicts
# with a concurrent hub-writer update, auto-resolve by taking origin's
# version. The worker's local mutation is either already pushed (queue
# claim flag) or about to be re-emitted on the next tick (heartbeat /
# status / commands), so no work is lost.
#
# Background: 2026-05-31 cloud-3 wedge. cloud-3 claimed V3-EXQ-618, ran it
# to PASS, hub writer dropped the completed entry on origin. Next worker
# loop's `git pull --rebase --autostash` rebased cleanly but the autostash
# pop produced UU markers in experiment_queue.json (autostash held the
# claimed entry; origin had the entry absent). Preflight kept failing on
# the conflict markers, every subsequent pull failed with "Pulling is not
# possible because you have unmerged files", and systemd marked the unit
# failed after 5 retries. Worker sat dead for 10 minutes during which
# V3-EXQ-621 went unclaimed. See WORKSPACE_STATE.md 2026-05-31T17:59Z.
_EPHEMERAL_WORKER_PATH_PREFIXES = (
    "experiment_queue.json",                          # ree-v3
    "evidence/experiments/runner_heartbeats/",        # REE_assembly
    "evidence/experiments/runner_status/",            # REE_assembly
    "evidence/experiments/runner_commands/",          # REE_assembly
)

# Untracked paths safe to stash before REE_assembly pull (Phase 3 hub writer
# is canonical once phase3: lands). Run-pack dirs under v3_exq_* are NOT
# matched -- only flat manifests and per-EXQ runner signals.
_UNTRACKED_FLAT_MANIFEST_RE = re.compile(
    r"^evidence/experiments/v3_[A-Za-z0-9_.-]+\.json$"
)
_UNTRACKED_RUNNER_SIGNAL_RE = re.compile(
    r"^evidence/experiments/_runner_signals/V3-EXQ-[A-Za-z0-9_.-]+\.json$"
)
_PREPULL_STASH_MESSAGE = "runner-prepull-untracked"


def _path_is_ephemeral_worker_owned(rel_path: str) -> bool:
    rel_path = rel_path.strip().strip('"')
    if not rel_path:
        return False
    for p in _EPHEMERAL_WORKER_PATH_PREFIXES:
        if rel_path == p.rstrip("/") or rel_path.startswith(p):
            return True
    return False


def _list_unmerged_paths(repo_path: Path) -> tuple[list[str], list[str]] | None:
    """Return (ephemeral_paths, non_ephemeral_paths) split of UU entries.

    Returns None if `git status --porcelain` failed.
    """
    st = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(repo_path), capture_output=True, text=True, timeout=10,
    )
    if st.returncode != 0:
        return None
    ephemeral: list[str] = []
    other: list[str] = []
    for line in st.stdout.splitlines():
        if len(line) < 4:
            continue
        flag = line[:2]
        # UU / AA / DD / AU / UA / UD / DU all denote unmerged paths.
        # "U" in either column is the canonical unmerged signal; AA / DD
        # are both-added / both-deleted which also need resolution.
        if "U" not in flag and flag not in ("AA", "DD"):
            continue
        path = line[3:].strip()
        if _path_is_ephemeral_worker_owned(path):
            ephemeral.append(path)
        else:
            other.append(path)
    return ephemeral, other


def _recover_ephemeral_pull_conflict(repo_path: Path, label: str) -> bool:
    """Auto-resolve UU markers on ephemeral worker-owned paths.

    Takes origin's version for any UU path matching the ephemeral set,
    finishes any in-progress rebase, and drops the autostash entry that
    pull --rebase --autostash created. Best-effort; never raises.

    Returns True iff the working tree is left free of UU markers (caller
    can proceed). Returns False if non-ephemeral conflicts are present
    (manual repair required) or if a step failed unexpectedly.
    """
    split = _list_unmerged_paths(repo_path)
    if split is None:
        return False
    ephemeral, other = split
    if not ephemeral and not other:
        return True
    if other:
        print(
            f"[runner] git pull {label}: UU on non-ephemeral path(s) "
            f"{other}, skipping auto-recovery (manual repair required)",
            flush=True,
        )
        return False
    print(
        f"[runner] git pull {label}: auto-resolving ephemeral UU on "
        f"{ephemeral} (origin / hub writer is authoritative)",
        flush=True,
    )
    # Always take the upstream-tracking ref's version. We deliberately
    # do NOT use `git checkout --theirs/--ours` because the orientation
    # changes between merge / rebase / stash-pop conflicts (a stash-pop
    # conflict's --theirs is the STASHED, i.e. worker-mutated, side --
    # the opposite of what we want here). `@{u}` resolves to whichever
    # origin/<branch> this clone tracks and gives us the canonical bytes
    # regardless of in-progress operation state.
    ub = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref",
         "--symbolic-full-name", "@{u}"],
        cwd=str(repo_path), capture_output=True, text=True, timeout=10,
    )
    upstream = ub.stdout.strip() if ub.returncode == 0 else ""
    if not upstream:
        print(
            f"[runner] git pull {label}: no upstream-tracking ref "
            f"resolvable, cannot recover: {ub.stderr.strip()}",
            flush=True,
        )
        return False
    co = subprocess.run(
        ["git", "checkout", upstream, "--"] + ephemeral,
        cwd=str(repo_path), capture_output=True, text=True, timeout=10,
    )
    if co.returncode != 0:
        print(
            f"[runner] git pull {label}: checkout {upstream} failed: "
            f"{co.stderr.strip()}", flush=True,
        )
        return False
    subprocess.run(
        ["git", "add", "--"] + ephemeral,
        cwd=str(repo_path), capture_output=True, timeout=10,
    )
    # If a rebase is mid-flight (rare: failure occurred mid-rebase rather
    # than mid-stash-pop), finish it. GIT_EDITOR=true skips the commit
    # message editor on the auto-continue.
    if (repo_path / ".git" / "rebase-merge").exists() or \
       (repo_path / ".git" / "rebase-apply").exists():
        cont = subprocess.run(
            ["git", "rebase", "--continue"],
            cwd=str(repo_path), capture_output=True, text=True,
            env={**os.environ, "GIT_EDITOR": "true"}, timeout=15,
        )
        if cont.returncode != 0:
            subprocess.run(
                ["git", "rebase", "--abort"], cwd=str(repo_path),
                capture_output=True, timeout=10,
            )
    # Drop the autostash entry that pull --rebase --autostash created and
    # left behind because its pop conflicted. Matching on "autostash"
    # avoids touching unrelated stashes the operator may have left.
    sl = subprocess.run(
        ["git", "stash", "list"], cwd=str(repo_path),
        capture_output=True, text=True, timeout=10,
    )
    if sl.returncode == 0:
        for line in sl.stdout.splitlines():
            if "autostash" not in line:
                continue
            ref = line.split(":", 1)[0]
            drop = subprocess.run(
                ["git", "stash", "drop", ref], cwd=str(repo_path),
                capture_output=True, text=True, timeout=10,
            )
            if drop.returncode == 0:
                print(f"[runner] git pull {label}: dropped {ref}",
                      flush=True)
            # Drop one per pass; indices rotate after a drop and the
            # next pull tick handles any remaining entry. The stash-bloat
            # warning surfaces it if accumulation continues.
            break
    # Final verification: working tree must be clear of UU markers.
    split2 = _list_unmerged_paths(repo_path)
    if split2 is None:
        return False
    ephemeral2, other2 = split2
    return not ephemeral2 and not other2


def _warn_on_stash_bloat(
    repo_path: Path, label: str, threshold: int = 20,
) -> None:
    """One-line WARN when the stash stack exceeds `threshold` entries.

    Dormant autostash entries accumulate when historical failure modes
    left stashes that nobody dropped. cloud-3 had 191 entries when the
    2026-05-31 wedge was diagnosed -- a scarred-timeline indicator that
    no longer matters operationally but signals the worker would benefit
    from a one-shot cleanup (coordinator/deploy/clear_worker_stashes.sh).
    Best-effort, never raises.
    """
    try:
        r = subprocess.run(
            ["git", "stash", "list"], cwd=str(repo_path),
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            return
        n = sum(1 for ln in r.stdout.splitlines() if ln.strip())
        if n > threshold:
            print(
                f"[runner] git pull {label}: stash list has {n} entries "
                f"(> {threshold}); consider running "
                f"coordinator/deploy/clear_worker_stashes.sh", flush=True,
            )
    except Exception:
        pass


def _untracked_paths_for_prepull_stash(repo_path: Path) -> list[str]:
    """Return repo-relative untracked paths safe to stash before REE_assembly pull."""
    try:
        st = subprocess.run(
            ["git", "status", "--porcelain", "-u", "--ignored=no"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=15,
        )
        if st.returncode != 0:
            return []
        out: list[str] = []
        for line in st.stdout.splitlines():
            if not line.startswith("?? "):
                continue
            rel = line[3:].strip().strip('"')
            if _UNTRACKED_FLAT_MANIFEST_RE.match(rel) or _UNTRACKED_RUNNER_SIGNAL_RE.match(rel):
                out.append(rel)
        return out
    except Exception:
        return []


def _prepull_stash_blocking_untracked(repo_path: Path, label: str) -> bool:
    """Stash untracked flat manifests/signals that block alignment. Never raises."""
    if label != "REE_assembly":
        return False
    paths = _untracked_paths_for_prepull_stash(repo_path)
    if not paths:
        return False
    try:
        r = subprocess.run(
            ["git", "stash", "push", "--include-untracked", "-m", _PREPULL_STASH_MESSAGE,
             "--", *paths],
            cwd=str(repo_path), capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            print(f"[runner] git pull {label}: stashed {len(paths)} untracked "
                  f"runner-owned path(s) before pull", flush=True)
            return True
        print(f"[runner] git pull {label}: prepull stash warn: "
              f"{r.stderr.strip()}", flush=True)
    except Exception as exc:
        print(f"[runner] git pull {label}: prepull stash error: {exc}", flush=True)
    return False


def _postpull_restore_prepull_stash(repo_path: Path, label: str) -> None:
    """Pop or drop the prepull stash if the hub writer now owns those paths."""
    if label != "REE_assembly":
        return
    try:
        top = subprocess.run(
            ["git", "stash", "list", "-1", "--format=%s"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=10,
        )
        if top.returncode != 0 or _PREPULL_STASH_MESSAGE not in (top.stdout or ""):
            return
        pop = subprocess.run(
            ["git", "stash", "pop"],
            cwd=str(repo_path), capture_output=True, text=True, timeout=30,
        )
        if pop.returncode != 0:
            subprocess.run(
                ["git", "stash", "drop"],
                cwd=str(repo_path), capture_output=True, timeout=10,
            )
            print(f"[runner] git pull {label}: dropped prepull stash "
                  f"(paths likely on origin now)", flush=True)
    except Exception:
        pass


def git_pull(repo_path: Path, label: str) -> None:
    """Pull latest changes. Retries on transient lock errors. Never raises.

    Uses --rebase --autostash so that local edits to heartbeat / status JSONs
    don't block the pull with "Your local changes would be overwritten by merge"
    (the cloud-1 stall we hit 2026-05-10, where the runner couldn't pull
    REE_assembly while the heartbeat thread was concurrently writing
    runner_heartbeats/<host>.json and runner_status/<host>.json). Plain
    --ff-only refuses on a dirty tree even when the dirty files are exactly
    the ones this machine writes (so a fast-forward would not actually
    conflict at content level).

    Autostash-pop conflicts on ephemeral worker-owned paths (the queue
    file's claim flag, runner_heartbeats/<host>.json, runner_status/
    <host>.json, runner_commands/<host>.json) are auto-resolved by taking
    origin's version: origin (the hub writer) is the canonical source for
    these paths under Phase 3. Non-ephemeral conflicts are left in place
    and a warning is emitted -- those need manual repair.
    """
    import time
    _LOCK_HINTS = ("cannot lock ref", "unable to resolve reference",
                   "lock file", "index.lock")
    # Heal any UU state left over from a previous tick BEFORE attempting
    # the new pull -- otherwise pull bails with "Pulling is not possible
    # because you have unmerged files" and the wedge persists indefinitely.
    pre = _list_unmerged_paths(repo_path)
    if pre is not None and (pre[0] or pre[1]):
        _recover_ephemeral_pull_conflict(repo_path, label)
    _prepull_stash_blocking_untracked(repo_path, label)
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["git", "pull", "--rebase", "--autostash"],
                cwd=str(repo_path), capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                # Git returns 0 even when the rebase fast-forwarded
                # cleanly but the autostash pop produced UU markers --
                # the wedge surface that bit cloud-3. Check stdout for
                # the telltale "Applying autostash resulted in conflicts"
                # line OR scan porcelain status; either way, trigger
                # the ephemeral-conflict recovery before returning so
                # the next tick doesn't bail with "Pulling is not possible
                # because you have unmerged files."
                msg = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "ok"
                combined = (r.stdout or "") + "\n" + (r.stderr or "")
                if "autostash resulted in conflicts" in combined:
                    print(f"[runner] git pull {label}: rebase ok but "
                          f"autostash pop conflicted, resolving...",
                          flush=True)
                    _recover_ephemeral_pull_conflict(repo_path, label)
                else:
                    print(f"[runner] git pull {label}: {msg}", flush=True)
                _postpull_restore_prepull_stash(repo_path, label)
                _warn_on_stash_bloat(repo_path, label)
                return
            stderr = r.stderr.strip()
            if any(h in stderr.lower() for h in _LOCK_HINTS) and attempt < 2:
                print(f"[runner] git pull {label}: transient lock, retrying "
                      f"({attempt + 1}/2)...", flush=True)
                time.sleep(2)
                continue
            # A failed --rebase pull may stop mid-rebase and leave
            # .git/rebase-merge behind, which wedges EVERY later git op on
            # this repo ("there is already a rebase-merge directory") until
            # a manual / launchd repair. Abort it so the next tick starts
            # clean. Lost work is impossible here: autostash is restored by
            # the abort, and a failed pull changed nothing to begin with.
            subprocess.run(["git", "rebase", "--abort"], cwd=str(repo_path),
                            capture_output=True, timeout=10)
            subprocess.run(["git", "rebase", "--quit"], cwd=str(repo_path),
                            capture_output=True, timeout=10)
            # If the failure left UU markers on ephemeral worker-owned
            # paths (the 2026-05-31 cloud-3 wedge), auto-resolve by taking
            # origin's version and retry the pull once.
            if _recover_ephemeral_pull_conflict(repo_path, label):
                r2 = subprocess.run(
                    ["git", "pull", "--rebase", "--autostash"],
                    cwd=str(repo_path), capture_output=True, text=True,
                    timeout=30,
                )
                if r2.returncode == 0:
                    msg = (r2.stdout.strip().splitlines()[-1]
                           if r2.stdout.strip() else "ok")
                    print(f"[runner] git pull {label}: {msg} "
                          f"(post-recovery)", flush=True)
                else:
                    print(f"[runner] git pull {label} post-recovery warn: "
                          f"{r2.stderr.strip()}", flush=True)
                _postpull_restore_prepull_stash(repo_path, label)
                _warn_on_stash_bloat(repo_path, label)
                return
            print(f"[runner] git pull {label} warn: {stderr}", flush=True)
            _warn_on_stash_bloat(repo_path, label)
            return
        except Exception as e:
            subprocess.run(["git", "rebase", "--abort"], cwd=str(repo_path),
                            capture_output=True, timeout=10)
            subprocess.run(["git", "rebase", "--quit"], cwd=str(repo_path),
                            capture_output=True, timeout=10)
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


def align_ree_assembly_checkout(ree_assembly_path: Path | None) -> None:
    """Best-effort align ree-v3 + REE_assembly with origin. Never raises.

    Public entry for serve.py and scripts. Same guards as the runner
    background pull (skips REE_assembly when an active TASK_CLAIMS entry
    covers evidence/).
    """
    _sync_pull_tick(ree_assembly_path)


def align_after_coordinator_result(
    ree_assembly_path: Path | None,
    manifest_path: str | None = None,
) -> None:
    """Pull soon after POST /result so phase3: commits appear locally.

    manifest_path is accepted for logging/future use; alignment is a full
    repo pull, not a single-file fetch. Schedules two delayed pulls (45s,
    120s) to cover hub writer tick latency without blocking the runner loop.
    """
    if not ree_assembly_path:
        return
    _sync_pull_tick(ree_assembly_path)
    if manifest_path:
        print(f"[runner] post-result align: scheduled pulls for "
              f"{manifest_path}", flush=True)

    def _delayed_pulls() -> None:
        for delay in (45, 120):
            time.sleep(delay)
            _sync_pull_tick(ree_assembly_path)

    threading.Thread(
        target=_delayed_pulls, daemon=True, name="post-result-align",
    ).start()


def _report_result_and_align(
    ree_assembly_path: Path | None,
    queue_id: str,
    run_id: str | None,
    manifest_path: str,
    outcome: str,
    machine: str,
) -> None:
    """POST /result then align checkout with origin. Never raises."""
    try:
        coordinator_client.report_result(
            queue_id, run_id, manifest_path, outcome, machine)
    except Exception as exc:
        print(f"[runner] report_result warn for {queue_id}: {exc}", flush=True)
    align_after_coordinator_result(ree_assembly_path, manifest_path)


def _sync_pull_tick(ree_assembly_path: Path | None) -> None:
    """One iteration of the --auto-sync background pull. Never raises.

    Pulls ree-v3 unconditionally, then pulls REE_assembly UNLESS a Claude
    session holds an active TASK_CLAIMS claim covering any evidence/ path.
    The REE_assembly skip mirrors runner_remote_control.push_heartbeat /
    push_commands: `git pull --rebase --autostash` can stack failing
    autostash stashes and silently revert a concurrent session's
    uncommitted evidence/claims edits (the EXQ-232 / substrate_queue.json
    revert incident class). The ree-v3 pull is intentionally unguarded --
    only REE_assembly carries the high-contention evidence files.

    Default path (no active evidence claim, or runner_remote_control
    unimportable) is bit-identical to the pre-guard behaviour: both pulls
    run, each best-effort.
    """
    try:
        git_pull(REPO_ROOT, "ree-v3")
    except Exception:
        pass
    if not ree_assembly_path:
        return
    if _rrc is not None and _rrc._active_claim_on_evidence_dir(ree_assembly_path):
        return
    try:
        git_pull(ree_assembly_path, "REE_assembly")
    except Exception:
        pass


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

        # Capture pre-reset HEAD SHA so we can restore committed result files
        # after `git reset --hard` destroys the local commit.  `git stash
        # --include-untracked` only captures the working tree; the manifest
        # that was just committed by git_push_results is in HEAD, not the
        # working tree, and would otherwise be lost.  See the V3-EXQ-541
        # leak incident (2026-05-08) and the fix prompt at
        # /tmp/cloud_manifest_leak_diagnosis_prompt.md.
        pre_reset_sha_r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
            text=True, timeout=5,
        )
        pre_reset_sha = pre_reset_sha_r.stdout.strip() if pre_reset_sha_r.returncode == 0 else ""

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

        # Restore committed result files from the destroyed pre-reset commit
        # BEFORE attempting stash pop.  This guarantees the manifest is on
        # disk regardless of whether the pop succeeds or conflicts.
        # Build the list of paths to restore.  When result_files is supplied
        # we restore only those (selective recovery, matches selective stage).
        # When result_files is None (broad-fallback path), derive the list
        # from the destroyed commit's diff against the new remote HEAD --
        # otherwise the broad `git add evidence/experiments/` recovery
        # below stages nothing because everything was nuked by reset --hard.
        paths_to_restore: list[str] = []
        if result_files:
            for f in result_files:
                try:
                    paths_to_restore.append(str(Path(f).relative_to(Path(cwd))))
                except ValueError:
                    paths_to_restore.append(f)
        elif pre_reset_sha:
            diff_r = subprocess.run(
                ["git", "diff", "--name-only", f"origin/{branch}", pre_reset_sha],
                cwd=cwd, capture_output=True, text=True, timeout=10,
            )
            if diff_r.returncode == 0:
                paths_to_restore = [
                    p for p in diff_r.stdout.splitlines() if p.strip()
                ]

        restored_paths: list[str] = []
        if pre_reset_sha and paths_to_restore:
            for rel in paths_to_restore:
                co = subprocess.run(
                    ["git", "checkout", pre_reset_sha, "--", rel],
                    cwd=cwd, capture_output=True, text=True, timeout=10,
                )
                if co.returncode == 0:
                    restored_paths.append(rel)
                else:
                    print(f"[runner] WARN: could not restore {rel} from "
                          f"{pre_reset_sha[:10]} ({label}): "
                          f"{co.stderr.strip()}", flush=True)

        # Pop stash to restore uncommitted work
        pop_succeeded = True
        if stashed:
            pop = subprocess.run(
                ["git", "stash", "pop"],
                cwd=cwd, capture_output=True, text=True, timeout=10,
            )
            if pop.returncode != 0:
                # Stash pop conflict -- log but continue.  The stash is
                # preserved in `git stash list` for manual recovery; the
                # restored manifest is on disk + staged and we still want
                # to commit + push it so the scientific result reaches
                # REE_assembly master.  Resolve every unmerged path by
                # taking the remote version (--ours, since after reset
                # HEAD == origin/branch); this clears the conflict
                # markers without overwriting the manifest restore.
                print(f"[runner] auto-sync: stash pop conflict ({label}). "
                      f"Stash preserved for manual recovery; continuing "
                      f"with manifest-only commit.", flush=True)
                pop_succeeded = False
                unmerged_r = subprocess.run(
                    ["git", "diff", "--name-only", "--diff-filter=U"],
                    cwd=cwd, capture_output=True, text=True, timeout=5,
                )
                unmerged = (
                    unmerged_r.stdout.splitlines()
                    if unmerged_r.returncode == 0 else []
                )
                for path in unmerged:
                    subprocess.run(
                        ["git", "checkout", "--ours", "--", path],
                        cwd=cwd, capture_output=True, timeout=10,
                    )
                    subprocess.run(["git", "reset", "HEAD", "--", path],
                                   cwd=cwd, capture_output=True, timeout=10)

        # Re-stage only the specific result files the runner wrote (selective)
        if result_files:
            for f in result_files:
                try:
                    rel = str(Path(f).relative_to(Path(cwd)))
                except ValueError:
                    rel = f
                subprocess.run(["git", "add", rel], cwd=cwd, capture_output=True, timeout=10)
            # Diagnostic: warn if the selective add staged none of the
            # expected files.  The historical silent-no-op (file removed
            # by reset --hard, `git add` matches nothing, stderr swallowed)
            # is what masked this bug for weeks.
            staged_r = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=cwd, capture_output=True, text=True, timeout=5,
            )
            staged = set(staged_r.stdout.splitlines()) if staged_r.returncode == 0 else set()
            expected = set()
            for f in result_files:
                try:
                    expected.add(str(Path(f).relative_to(Path(cwd))))
                except ValueError:
                    expected.add(f)
            missing = expected - staged
            if missing:
                print(f"[runner] WARN: post-recovery selective add staged "
                      f"none of {sorted(missing)} ({label}). "
                      f"pop_succeeded={pop_succeeded} "
                      f"restored_via_checkout={sorted(restored_paths)}",
                      flush=True)
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


# --- Phase 3 runner-push gates --------------------------------------------
# When PHASE3_GIT_WRITER_READY=True on the hub, sync_daemon becomes the sole
# writer to REE_assembly. The runner's existing per-result / per-queue /
# per-heartbeat git pushes do `git pull --rebase --autostash` and would
# fight the writer for the index -- exactly the autostash mechanism Phase 3
# exists to retire. These three env flags gate the runner-side pushes;
# default OFF means current Phase 2 behaviour (push as today). Each flag
# should be flipped to "1" only after its sync_daemon counterpart is wired:
#   PHASE3_DISABLE_RUNNER_RESULT_PUSH    -> sync_daemon.phase3_git_writer
#                                          (LANDED 2026-05-27; writer-ready
#                                          flag still False pending review)
#   PHASE3_DISABLE_RUNNER_QUEUE_PUSH     -> PLAN.md step 5 queue snapshot
#                                          writeback (NOT YET WIRED)
#   PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH -> PLAN.md step 6 derived
#                                          heartbeats + runner_status
#                                          writeback (NOT YET WIRED)
#   PHASE3_DISABLE_RUNNER_CLAIM_PUSH     -> coordinator /claim endpoint owns
#                                          claim arbitration (Phase 3 live
#                                          2026-05-29). Suppresses the legacy
#                                          git-mutex `claim:` / `release claim:`
#                                          commits in attempt_claim /
#                                          release_claim. SAFE to set whenever
#                                          COORDINATION_MODE=coordinator -- the
#                                          coordinator DB (db.try_claim,
#                                          BEGIN IMMEDIATE) is the authoritative
#                                          mutex, so the git push is pure noise.
#                                          Unlike the other three flags this does
#                                          NOT strand an artifact: the claimed_by
#                                          write still lands in the LOCAL queue
#                                          file (the runner reads it on the next
#                                          tick); only the commit/push is skipped.
# Setting a flag whose sync_daemon counterpart isn't implemented yet means
# that artifact stops reaching origin entirely (does NOT apply to the
# claim-push flag, which has the coordinator /claim DB as its counterpart).

def _phase3_gate(env_name: str) -> bool:
    """Truthy when the env var is '1', 'true', or 'yes' (case-insensitive)."""
    return os.environ.get(env_name, "").strip().lower() in ("1", "true", "yes")


def _claim_push_gated() -> bool:
    """True when the legacy git-based claim mutex push should be suppressed.

    The `claim:` / `release claim:` commits in attempt_claim / release_claim
    are the last git-as-IPC coordination path. Under Phase 3 the coordinator
    /claim endpoint (db.try_claim, atomic BEGIN IMMEDIATE) is the authoritative
    arbiter, so these commits are pure noise on ree-v3/main.

    Fires under EITHER:
      - PHASE3_DISABLE_RUNNER_CLAIM_PUSH (the dedicated claim-signaling gate;
        set in the coordinator-mode fleet configs), OR
      - PHASE3_DISABLE_RUNNER_QUEUE_PUSH (legacy entanglement: the claim write
        lands in experiment_queue.json, so the pre-dedicated-gate fleet configs
        that set QUEUE_PUSH=1 already suppressed the claim push via this same
        code path. Kept in the OR so introducing the dedicated flag is NOT a
        behaviour change for any worker already running QUEUE_PUSH=1).

    Default OFF (neither set) -> the legacy git-mutex pull/commit/push runs,
    bit-identical to any pre-Phase-3 / git-mode runner. In git/shadow mode the
    push IS the mutex, so leave both flags unset there.
    """
    return (_phase3_gate("PHASE3_DISABLE_RUNNER_CLAIM_PUSH")
            or _phase3_gate("PHASE3_DISABLE_RUNNER_QUEUE_PUSH"))


_PHASE3_HUB_FILE_WRITE_GATE_LOGGED = False


def _phase3_hub_local_ree_assembly_writes_gated() -> bool:
    """Skip local REE_assembly runner_status file writes that dirty sync_daemon.

    Fires under EITHER gate:
      - PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE (hub-only co-tenancy; ree-cloud-1
        shares its checkout with sync_daemon), OR
      - PHASE3_RUNNER_TELEMETRY_OFF_GIT (worker-safe; stops the per-tick
        runner_status/<host>.json write that conflicts with the hub-materialised
        version on `git pull --rebase --autostash` and accumulates dormant
        autostashes). Does NOT gate the command channel -- no restart-loop risk.
    In-memory status + coordinator POST /status still run; only the on-disk
    runner_status/<machine>.json write is skipped. The hub sync_daemon
    materialises the canonical file from the coordinator DB.
    """
    global _PHASE3_HUB_FILE_WRITE_GATE_LOGGED
    if _rrc is None:
        return False
    gated = (_rrc._phase3_heartbeat_write_gated()
             or _rrc._phase3_telemetry_file_write_gated())
    if gated and not _PHASE3_HUB_FILE_WRITE_GATE_LOGGED:
        print("[runner] phase3 gate: skipping local runner_status file "
              "writes (coordinator POST is transport; sync_daemon "
              "materialises git from the coordinator DB)",
              flush=True)
        _PHASE3_HUB_FILE_WRITE_GATE_LOGGED = True
    return gated


def git_push_queue() -> None:
    """Stage, commit, and push experiment_queue.json to ree-v3. Warns on failure."""
    if _phase3_gate("PHASE3_DISABLE_RUNNER_QUEUE_PUSH"):
        print("[runner] phase3 gate: skipping git_push_queue "
              "(sync_daemon owns queue snapshot writeback)", flush=True)
        return
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

    Phase 3: when PHASE3_DISABLE_RUNNER_RESULT_PUSH=1, becomes a no-op.
    sync_daemon's phase3_git_writer takes over publishing manifests via
    its own commit + push (no autostash). The runner still spools the
    manifest into the coordinator (POST /result writes spool bytes when
    COORDINATOR_SPOOL_DIR is set on the hub), so the bytes still reach
    origin -- just via the writer instead of the runner.
    """
    if _phase3_gate("PHASE3_DISABLE_RUNNER_RESULT_PUSH"):
        print("[runner] phase3 gate: skipping git_push_results "
              "(sync_daemon writer owns result commits)", flush=True)
        return
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
        _git_push_with_retry(str(ree_assembly_path), "master",
                             "results -> REE_assembly",
                             result_files=result_files)
    except Exception as e:
        print(f"[runner] auto-sync push error: {e}", flush=True)


def git_push_status(ree_assembly_path: Path, status_path: Path, queue_id: str) -> None:
    """Stage, commit, and push the per-machine runner_status file only.

    Called per-completion to maintain the invariant that GitHub's status is at
    least as fresh as the queue. Without this, the queue-removal commit lands
    (via git_push_queue) while the corresponding status entry stays local --
    other machines see "queue shrunk" with no record of what ran.

    Warns on failure; never raises.

    Phase 3: gated by PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH (status writes
    are derived from the same table sync_daemon's step-6 writeback owns).
    Setting the gate before step 6 is wired stops status from reaching
    origin entirely.
    """
    if _phase3_gate("PHASE3_DISABLE_RUNNER_HEARTBEAT_PUSH"):
        # No log here -- this is called per-completion and would be
        # noisy. The push_heartbeat gate prints once per tick already.
        return
    try:
        rel = str(status_path.relative_to(ree_assembly_path))
        subprocess.run(
            ["git", "add", rel],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=10,
        )
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(ree_assembly_path), timeout=5,
        )
        if diff.returncode == 0:
            return  # nothing to push
        subprocess.run(
            ["git", "commit", "-m", f"runner_status: {queue_id} -> {status_path.stem}"],
            cwd=str(ree_assembly_path), capture_output=True, text=True, timeout=15,
        )
        _git_push_with_retry(str(ree_assembly_path), "master", f"status {queue_id} -> REE_assembly")
    except Exception as e:
        print(f"[runner] auto-sync status push error ({queue_id}): {e}", flush=True)


# ── Multi-machine coordination ────────────────────────────────────────────────

CLAIM_TTL_HOURS = float(os.environ.get("REE_CLAIM_TTL_HOURS", "6"))
CLAIM_HEARTBEAT_FRESH_SECONDS = int(
    os.environ.get("REE_CLAIM_HEARTBEAT_FRESH_SECONDS", "900")
)


def _get_machine_name(override: str | None = None) -> str:
    return override or socket.gethostname()


def _affinity_matches(item: dict, machine: str) -> bool:
    """Return True if this machine is allowed to run the experiment."""
    affinity = item.get("machine_affinity", "any")
    return affinity in ("any", None, "") or affinity == machine


# Cloud workers the laptop yields to when --laptop-yield-to-cloud is on.
# Hardcoded list matches the cloud-scaler.yml WORKERS pairing (cloud-1 is the
# coordinator hub but may run experiments when PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1
# is set in hub shadow.conf; scaler never powers off the hub VM).
LAPTOP_YIELD_CLOUD_HOSTS = (
    "ree-cloud-1",
    "ree-cloud-2",
    "ree-cloud-3",
    "ree-cloud-4",
)

# Hostname that auto-arms --laptop-yield-to-cloud when not explicitly set.
LAPTOP_AUTO_YIELD_HOSTNAME = "DLAPTOP-4.local"


def _cloud_worker_is_fresh(
    cloud_host: str,
    freshness_minutes: int,
    ree_assembly_path: Path | None,
) -> bool:
    """True iff runner_heartbeats/<cloud_host>.json has last_tick_utc within
    freshness_minutes of now. Missing/unparseable heartbeats are treated as
    not fresh (the cloud worker is not alive enough to be preferred over the
    laptop)."""
    base = ree_assembly_path or find_ree_assembly_path()
    if base is None:
        return False
    hb_path = (
        base / "evidence" / "experiments" / "runner_heartbeats" /
        f"{_safe_heartbeat_filename(cloud_host)}.json"
    )
    try:
        hb = json.loads(hb_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    last_tick = _parse_utc_timestamp(hb.get("last_tick_utc"))
    if last_tick is None:
        return False
    age = datetime.now(timezone.utc) - last_tick
    return age.total_seconds() <= freshness_minutes * 60


# Heartbeat `state` values in which a cloud worker is available to claim a NEW
# item soon. "idle" is the only one: "running" is busy (often for hours on a
# behavioural run), "paused" is deliberately held, "draining" is shutting down,
# "starting" is mid-boot and will report "idle" on its next tick anyway.
_AVAILABLE_CLOUD_STATES = frozenset({"idle"})


def _cloud_worker_is_available(
    cloud_host: str,
    freshness_minutes: int,
    ree_assembly_path: Path | None,
) -> bool:
    """True iff runner_heartbeats/<cloud_host>.json is fresh within
    freshness_minutes AND the worker is idle (state in _AVAILABLE_CLOUD_STATES
    with no current_exq). A fresh-but-busy cloud worker is NOT available: it is
    alive but will not claim a new item until its current run finishes, so the
    laptop must not yield to it (doing so starves the queue while the whole
    cloud fleet is saturated). Missing/unparseable heartbeats are treated as
    not available."""
    base = ree_assembly_path or find_ree_assembly_path()
    if base is None:
        return False
    hb_path = (
        base / "evidence" / "experiments" / "runner_heartbeats" /
        f"{_safe_heartbeat_filename(cloud_host)}.json"
    )
    try:
        hb = json.loads(hb_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    last_tick = _parse_utc_timestamp(hb.get("last_tick_utc"))
    if last_tick is None:
        return False
    age = datetime.now(timezone.utc) - last_tick
    if age.total_seconds() > freshness_minutes * 60:
        return False
    if hb.get("state") not in _AVAILABLE_CLOUD_STATES:
        return False
    if hb.get("current_exq"):
        return False
    return True


def _should_yield_to_cloud(
    item: dict,
    freshness_minutes: int,
    ree_assembly_path: Path | None,
) -> tuple[bool, str | None]:
    """Decide whether the laptop should skip this queue item in favour of a
    cloud worker. Returns (should_yield, available_cloud_host).

    Yields only when the item's affinity is "any" (None/"" treated as "any")
    AND at least one cloud worker in LAPTOP_YIELD_CLOUD_HOSTS is both fresh
    within freshness_minutes AND idle (available to claim it). A cloud worker
    that is alive but busy on its own long run does NOT trigger a yield -- when
    every cloud worker is saturated the laptop runs the item itself instead of
    leaving it to starve. Items pinned to a specific hostname are never yielded
    -- if the pin is the laptop the laptop must run it; if the pin is a cloud
    host the existing affinity check already filters it out.
    """
    affinity = item.get("machine_affinity", "any")
    if affinity not in ("any", None, ""):
        return False, None
    for host in LAPTOP_YIELD_CLOUD_HOSTS:
        if _cloud_worker_is_available(host, freshness_minutes, ree_assembly_path):
            return True, host
    return False, None


def _parse_utc_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_heartbeat_filename(machine: str) -> str:
    keep = "-_."
    return "".join(c if (c.isalnum() or c in keep) else "_" for c in machine)


def _claim_owner_has_fresh_heartbeat(
    claimed_by: dict,
    queue_id: str,
    ree_assembly_path: Path | None = None,
) -> bool:
    """True when the claim owner is freshly reporting this same queue item.

    This is the lease extension for multi-day experiments. The original
    six-hour claimed_at TTL is still useful for dead workers, but a live
    owner must not be reclaimed just because the run is long.
    """
    owner = claimed_by.get("machine")
    if not owner or not queue_id:
        return False

    base = ree_assembly_path or find_ree_assembly_path()
    if base is None:
        return False
    hb_path = (
        base / "evidence" / "experiments" / "runner_heartbeats" /
        f"{_safe_heartbeat_filename(owner)}.json"
    )
    try:
        hb = json.loads(hb_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    if hb.get("current_exq") != queue_id:
        return False
    last_tick = _parse_utc_timestamp(hb.get("last_tick_utc"))
    if last_tick is None:
        return False
    age = datetime.now(timezone.utc) - last_tick
    return age.total_seconds() <= CLAIM_HEARTBEAT_FRESH_SECONDS


def _is_stale_claim(
    claimed_by: dict,
    queue_id: str | None = None,
    ree_assembly_path: Path | None = None,
) -> bool:
    """Return True if a claim is old and the owner is not freshly alive."""
    try:
        claimed_at = _parse_utc_timestamp(claimed_by["claimed_at"])
        if claimed_at is None:
            raise ValueError("bad claimed_at")
        age = datetime.now(timezone.utc) - claimed_at
        timestamp_stale = age.total_seconds() > CLAIM_TTL_HOURS * 3600
    except Exception:
        timestamp_stale = True  # malformed -> treat as stale
    if not timestamp_stale:
        return False
    if queue_id and _claim_owner_has_fresh_heartbeat(
        claimed_by, queue_id, ree_assembly_path
    ):
        return False
    return True


def _git_undo_last_commit(repo: Path) -> None:
    """Undo the most recent local commit (pre-push rollback)."""
    subprocess.run(["git", "reset", "--soft", "HEAD~1"],
                   cwd=str(repo), capture_output=True)
    subprocess.run(["git", "reset", "HEAD", "experiment_queue.json"],
                   cwd=str(repo), capture_output=True)
    subprocess.run(["git", "checkout", "--", "experiment_queue.json"],
                   cwd=str(repo), capture_output=True)


def _atomic_write_queue(path: Path, data: dict, trailing_newline: bool = True) -> None:
    """Atomically write JSON to a queue file via tmp+os.replace.

    Avoids the partial-read race that produced the 2026-05-28 cloud-2
    crashloop: write_text() opens+truncates+writes, so a concurrent
    `git pull` reader can see an empty / partial file -- and worse, if
    git pull writes conflict markers into the file, the runner's next
    json.loads fails (Expecting ',' delimiter: line 23 column 1).

    tmp.replace(path) is atomic on POSIX and works on Windows (unlike
    rename), matching the write_status() pattern already in this module.
    """
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps(data, indent=2)
    if trailing_newline:
        payload += "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


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

    Phase 3: when PHASE3_DISABLE_RUNNER_CLAIM_PUSH=1 (or the legacy
    PHASE3_DISABLE_RUNNER_QUEUE_PUSH=1 -- see _claim_push_gated), the
    coordinator's /claim endpoint owns claim arbitration (via acquire_claim
    -> coordinator_client.claim); the git-push-as-mutex path here is legacy
    infrastructure that races sync_daemon for the ree-v3 main index. Under
    the gate, skip the git pull / commit / push entirely. The local queue
    file is still updated (atomically) because the runner reads from it on
    subsequent ticks, but no `claim:` commit reaches origin.
    """
    repo = queue_file.parent
    phase3_gated = _claim_push_gated()
    try:
        # 1. Pull latest (skipped under Phase 3 -- sync_daemon owns refresh)
        if not phase3_gated:
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
        if (existing and existing.get("machine") != machine
                and not _is_stale_claim(existing, queue_id)):
            return "already_claimed"

        if not _affinity_matches(item, machine):
            return "already_claimed"

        # 3. Write claim atomically (replaces non-atomic write_text)
        item["claimed_by"] = {"machine": machine, "claimed_at": now_utc()}
        item["status"] = "claimed"
        _atomic_write_queue(queue_file, data)

        if phase3_gated:
            # Local file written; coordinator owns the race resolution
            # (acquire_claim called coordinator_client.claim before us, or
            # will via the shadow-report path). No commit reaches origin.
            return "ok"

        # 4. Commit + push (legacy git-as-mutex path)
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
        if not phase3_gated:
            try:
                _git_undo_last_commit(repo)
            except Exception:
                pass
        return "error"


def release_claim(queue_file: Path, queue_id: str, machine: str) -> None:
    """
    Release a claim on shutdown so another machine can pick up the experiment.
    Best-effort -- warns on failure but never raises.

    Phase 3: when PHASE3_DISABLE_RUNNER_CLAIM_PUSH=1 (or the legacy
    PHASE3_DISABLE_RUNNER_QUEUE_PUSH=1 -- see _claim_push_gated), skip the
    git pull / commit / push. The local queue file is updated atomically so
    the runner doesn't act on a stale local claim; coordinator-side release
    is owned by release_active_claim -> coordinator_client.
    """
    repo = queue_file.parent
    phase3_gated = _claim_push_gated()
    try:
        if not phase3_gated:
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
        _atomic_write_queue(queue_file, data)
        if phase3_gated:
            print(f"[runner] Released local claim on {queue_id} "
                  f"(phase3: no commit/push)", flush=True)
            return
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


def coordinator_claims_authoritative() -> bool:
    """True when Phase-2 coordinator claiming is explicitly enabled."""
    try:
        return bool(coordinator_client.claims_authoritative())
    except Exception:
        return False


def acquire_claim(queue_file: Path, queue_id: str, machine: str) -> str:
    """Acquire a claim via the selected coordination authority."""
    if coordinator_claims_authoritative():
        return coordinator_client.claim(queue_id, machine)

    claim_result = attempt_claim(queue_file, queue_id, machine)
    # SHADOW: report the git verdict so the coordinator can compare its
    # own atomic-claim logic against git's. Under Phase 3 the local
    # heartbeat files that drive _is_stale_claim are materialised by the
    # hub's sync_daemon and can lag the DB by minutes; the legacy git
    # path can then take a "stale" claim that the writer-authoritative
    # DB still considers active. When the shadow report comes back with
    # coord_verdict="already_claimed" against our local "ok", believe
    # the coordinator and release the local claim before the runner
    # spawns a duplicate experiment process.
    report = coordinator_client.report_claim(queue_id, machine, claim_result)
    if (claim_result == "ok"
            and report is not None
            and report.get("verdict") == "already_claimed"):
        print(f"[runner] coordinator says {queue_id} already_claimed; "
              f"releasing local claim and yielding", flush=True)
        try:
            release_claim(queue_file, queue_id, machine)
        except Exception as exc:  # noqa: BLE001 -- best-effort rollback
            print(f"[runner] local release after divergence "
                  f"failed ({queue_id}): {exc}", flush=True)
        return "already_claimed"
    return claim_result


def release_active_claim(queue_file: Path, queue_id: str, machine: str) -> None:
    """Release a live claim through the selected coordination authority."""
    if coordinator_claims_authoritative():
        r = coordinator_client.release_claim(queue_id, machine)
        if not r or not r.get("ok"):
            note = r.get("note") if isinstance(r, dict) else "no response"
            print(f"[runner] Coordinator release warn ({queue_id}): {note}",
                  flush=True)
        return
    release_claim(queue_file, queue_id, machine)


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
            if cb and cb.get("machine") != machine and _is_stale_claim(cb, item["queue_id"]):
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
    if _phase3_hub_local_ree_assembly_writes_gated():
        status["last_updated"] = now_utc()
        return
    with _write_status_lock:
        tmp = path.with_suffix(".tmp")
        status["last_updated"] = now_utc()
        # Pin encoding so cross-platform readers (sync_daemon reads on
        # Linux; runners write on Mac/Windows) interpret the bytes the
        # same way the writer produced them.
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.replace(path)  # replace() is atomic on Unix and works on Windows (unlike rename)


def item_has_force_rerun(item: dict) -> bool:
    """True when queue item requests an intentional re-run under the same queue_id."""
    return item.get("force_rerun") is True


def should_skip_as_completed(item: dict, completed_ids: set) -> bool:
    """True if this queue item should be skipped due to a prior completion record.

    force_rerun items stay eligible even when queue_id appears in completed_ids
    (local runner_status or peer machines). Mirrors validate_queue.py guard.
    """
    queue_id = item.get("queue_id")
    if not queue_id:
        return True
    if item_has_force_rerun(item):
        return False
    return queue_id in completed_ids


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

    if _phase3_hub_local_ree_assembly_writes_gated():
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


def _run_axis_count(value, field_name: str) -> int:
    """Return the number of runs implied by an int or explicit list field."""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an int or list, got bool")
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return len(value)
    raise TypeError(f"{field_name} must be an int or list, got {type(value).__name__}")


def save_script_timing(script: str, actual_secs: float, seeds, conditions, episodes: int) -> None:
    try:
        seed_count = _run_axis_count(seeds, "seeds")
        condition_count = _run_axis_count(conditions, "conditions")
    except TypeError as exc:
        print(f"[runner] Calibration skip: {exc}", flush=True)
        return
    total_ep_cond = seed_count * condition_count * episodes
    if total_ep_cond <= 0:
        return
    actual_ms_per = round((actual_secs * 1000) / total_ep_cond, 1)
    timing = load_script_timing()
    timing[script] = actual_ms_per
    SCRIPT_TIMING_FILE.write_text(json.dumps(timing, indent=2))
    print(f"[runner] Calibration updated: {script} -> {actual_ms_per:.0f} ms/ep-cond", flush=True)


def estimate_minutes(item: dict, calibration: dict, script_timing: dict | None = None) -> float:
    seed_count = _run_axis_count(item.get("seeds", 1), "seeds")
    condition_count = _run_axis_count(item.get("conditions", 1), "conditions")
    episodes = item.get("episodes_per_run", 130)
    script = item.get("script", "")
    if script_timing and script in script_timing:
        ms_per = script_timing[script]
    else:
        ms_per = calibration.get("ms_per_episode_condition", 8000)
    return (seed_count * condition_count * episodes * ms_per) / 60_000


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


def _resolve_signal_dir(ree_assembly_path: Path | None) -> Path | None:
    """Return REE_assembly/evidence/experiments/_runner_signals/, or None."""
    if ree_assembly_path is None:
        return None
    d = ree_assembly_path / RUNNER_SIGNAL_SUBPATH
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def _build_subprocess_env(queue_id: str, signal_dir: Path | None) -> dict:
    """Build the env dict for an experiment subprocess.

    Both REE_QUEUE_ID and REE_RUNNER_SIGNAL_DIR are ALWAYS written -- with
    empty-string fallbacks -- so the child never inherits a stale value
    from the runner's own shell env. A stale REE_QUEUE_ID would route the
    sentinel emit_outcome() writes to the wrong file under the wrong
    signal dir, masking real-run sentinels with phantom ones.
    """
    env = os.environ.copy()
    env["REE_QUEUE_ID"] = queue_id or ""
    env["REE_RUNNER_SIGNAL_DIR"] = str(signal_dir) if signal_dir is not None else ""
    return env


def _read_sentinel(signal_dir: Path | None, queue_id: str) -> dict | None:
    """Read <signal_dir>/<queue_id>.json. Return parsed dict or None."""
    if signal_dir is None or not queue_id:
        return None
    sig_path = signal_dir / f"{queue_id}.json"
    if not sig_path.is_file():
        return None
    try:
        return json.loads(sig_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[runner] WARN: sentinel {sig_path} unreadable: {exc}", flush=True)
        return None


def run_experiment(item: dict, status: dict, status_path: Path, calibration: dict,
                   script_timing: dict | None = None,
                   proc_ref: list | None = None,
                   suspend_flag: list | None = None,
                   auto_sync: bool = False,
                   ree_assembly_path: Path | None = None,
                   remote_control: bool = False,
                   machine: str | None = None) -> dict:
    """Run a single experiment script as a subprocess.

    proc_ref: if provided, the active Popen object is stored as proc_ref[0] immediately
    after launch and cleared on completion.  The signal handler uses this to kill the
    subprocess on a forced stop.
    """
    script = REPO_ROOT / item["script"]
    raw_args = item.get("args", [])
    if isinstance(raw_args, str):
        raw_args = shlex.split(raw_args)
    args = [sys.executable, "-u", str(script)] + list(raw_args)
    if _rckpt is not None and "--no-resume" not in args:
        for extra in _rckpt.resume_args_for_item(item, ree_assembly_path):
            if extra == "--resume":
                if "--resume" not in args:
                    args.append(extra)
            elif extra.startswith("--checkpoint-path="):
                if not any(a.startswith("--checkpoint-path=") for a in args):
                    args.append(extra)

    signal_dir = _resolve_signal_dir(ree_assembly_path)
    queue_id = item.get("queue_id", "")
    # Pre-clean any stale sentinel from a previous attempt of the same queue_id
    # (e.g. requeued after manual edit). Avoids spurious "PASS" reads.
    if signal_dir is not None and queue_id:
        stale = signal_dir / f"{queue_id}.json"
        if stale.exists():
            try:
                stale.unlink()
            except OSError as _exc:
                print(f"[runner] WARN: could not unlink stale sentinel {stale}: {_exc}",
                      flush=True)

    seed_count = _run_axis_count(item.get("seeds", 1), "seeds")
    condition_count = _run_axis_count(item.get("conditions", 1), "conditions")
    total_runs = max(1, seed_count * condition_count)
    episodes_per_run = item.get("episodes_per_run", 130)
    # Floor prevents warmup/eval phase denominators from shrinking below the
    # queue-specified per-run episode count (fix for multi-phase experiments).
    episodes_per_run_floor = item.get("episodes_per_run", 0)
    # True when the experiment manages seeds/conditions internally (no queue metadata).
    _unstructured_est = (
        bool(item.get("estimated_minutes"))
        and not item.get("seeds")
        and not item.get("conditions")
        and total_runs == 1
    )

    runs_done = 0
    current_run_label = "starting..."
    episodes_in_run = 0
    recent_lines: list[str] = []
    run_end_times: list[float] = []

    started_at = time.monotonic()
    started_at_utc = now_utc()

    def overall_pct() -> float:
        # For experiments with estimated_minutes but no seed/condition metadata,
        # use elapsed-time fraction so multi-phase internal loops don't cause
        # the timer to show 100% after the first phase completes.
        if _unstructured_est:
            elapsed = time.monotonic() - started_at
            est_secs = float(item["estimated_minutes"]) * 60
            return min(round(elapsed / est_secs * 100, 1), 99.0)
        run_frac = (runs_done + episodes_in_run / max(episodes_per_run, 1)) / max(total_runs, 1)
        return round(run_frac * 100, 1)

    def seconds_remaining() -> float:
        elapsed = time.monotonic() - started_at
        pct = overall_pct()
        manual_est = item.get("estimated_minutes")
        if manual_est:
            static_secs = float(manual_est) * 60
        else:
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
        "exit_code": None,
        "has_sentinel": False,
    }
    exit_code = None
    has_sentinel = False

    try:
        env = _build_subprocess_env(queue_id, signal_dir)
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if proc_ref is not None:
            proc_ref.clear()
            proc_ref.append(proc)

        _hb_stop = threading.Event()
        # Track last remote-control heartbeat push so we can write the local
        # heartbeat file every STATUS_WRITE_INTERVAL but push to git only
        # every ~60s (matches the idle-loop heartbeat cadence; avoids
        # 12 commits/min/machine during a run).
        _last_hb_push = [0.0]
        HB_PUSH_INTERVAL = 60.0

        def _build_progress_payload() -> dict:
            return {
                "run_label": current_run_label,
                "runs_done": runs_done,
                "runs_total": total_runs,
                "episodes_done": episodes_in_run,
                "episodes_total": episodes_per_run,
                "overall_pct": overall_pct(),
            }

        def _push_remote_heartbeat():
            """Write + (throttled) push the per-machine remote-control
            heartbeat with full progress payload. Best-effort; never raises.
            """
            if not (remote_control and _rrc is not None and ree_assembly_path
                    and machine):
                return
            try:
                hb_path = _rrc.write_heartbeat(
                    ree_assembly_path, machine, state="running",
                    current_exq=item["queue_id"],
                    current_exq_started_utc=started_at_utc,
                    current_title=item.get("title", ""),
                    current_claim_id=item.get("claim_id", ""),
                    current_description=item.get("description", ""),
                    progress=_build_progress_payload(),
                    seconds_elapsed=round(time.monotonic() - started_at),
                    seconds_remaining=round(seconds_remaining()),
                    recent_lines=list(recent_lines[-5:]),
                    runner_pid=os.getpid(),
                )
                if auto_sync and hb_path is not None:
                    now_mono = time.monotonic()
                    if now_mono - _last_hb_push[0] >= HB_PUSH_INTERVAL:
                        _rrc.push_heartbeat(ree_assembly_path, hb_path)
                        _last_hb_push[0] = now_mono
            except Exception as _exc:
                print(f"[runner] remote heartbeat warn: {_exc}", flush=True)

        def _heartbeat():
            while not _hb_stop.wait(timeout=STATUS_WRITE_INTERVAL):
                update_status_current()
                _push_remote_heartbeat()
        _hb_thread = threading.Thread(target=_heartbeat, daemon=True)
        _hb_thread.start()

        if auto_sync:
            def _background_sync():
                while not _hb_stop.wait(timeout=60):
                    _sync_pull_tick(ree_assembly_path)
            _sync_thread = threading.Thread(target=_background_sync, daemon=True)
            _sync_thread.start()

        for line in proc.stdout:
            if suspend_flag:
                print("[runner] suspend requested -- terminating experiment",
                      flush=True)
                try:
                    proc.terminate()
                except Exception:
                    pass
                break
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
                new_denom = int(m.group(2))
                # Only update if the new denominator is at least as large as the
                # queue-specified floor, so warmup/eval phase denominators don't
                # shrink episodes_per_run and cause overall_pct to hit 100% early.
                if episodes_per_run_floor == 0 or new_denom >= episodes_per_run_floor:
                    episodes_per_run = new_denom

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
            _banner_m = RE_EXQ_BANNER.search(line)
            _dashed_m = RE_EXQ_DASHED_OUTCOME.search(line)
            _bare_m = RE_BARE_OUTCOME.match(line)
            if "verdict: PASS" in line or (RE_STATUS_LINE.match(line) and "PASS" in line) \
                    or (_exq_m and _exq_m.group(1) == "PASS") \
                    or (_done_m and _done_m.group(1) == "PASS") \
                    or (_banner_m and _banner_m.group(1) == "PASS") \
                    or (_dashed_m and _dashed_m.group(1) == "PASS") \
                    or (_bare_m and _bare_m.group(1) == "PASS"):
                result_info["result"] = "PASS"
            elif "verdict: FAIL" in line or (RE_STATUS_LINE.match(line) and "FAIL" in line) \
                    or (_exq_m and _exq_m.group(1) == "FAIL") \
                    or (_done_m and _done_m.group(1) == "FAIL") \
                    or (_banner_m and _banner_m.group(1) == "FAIL") \
                    or (_dashed_m and _dashed_m.group(1) == "FAIL") \
                    or (_bare_m and _bare_m.group(1) == "FAIL"):
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
        result_info["exit_code"] = exit_code

        if suspend_flag and _rckpt is not None and ree_assembly_path:
            suspend_flag.clear()
            exp_type = _rckpt.experiment_type_for_item(item)
            ckpt_path = _rckpt.partial_checkpoint_path(
                ree_assembly_path, queue_id, exp_type)
            payload = _rckpt.load_partial_checkpoint(ckpt_path, queue_id)
            if _rckpt.is_resumable_partial(payload):
                result_info["result"] = "SUSPENDED"
                result_info["result_summary"] = (
                    f"Suspended with partial checkpoint at {ckpt_path}")
                result_info["checkpoint_path"] = str(ckpt_path)
            else:
                result_info["result"] = "ERROR"
                result_info["result_summary"] = (
                    "Suspend requested but no resumable partial checkpoint "
                    f"(looked for {ckpt_path})")
            result_info["completed_at"] = now_utc()
            result_info["actual_secs"] = round(time.monotonic() - started_at, 1)
            return result_info

        # Sentinel file authoritatively determines outcome (replaces fragile
        # stdout-regex scraping that caused 2026-05-08 silent drops). The
        # stdout-derived result_info["result"] is kept as a diagnostic
        # cross-check but the sentinel wins when present.
        sentinel = _read_sentinel(signal_dir, queue_id)
        has_sentinel = sentinel is not None
        result_info["has_sentinel"] = has_sentinel
        if sentinel is not None:
            sent_outcome = sentinel.get("outcome")
            sent_manifest = sentinel.get("manifest_path")
            if sent_outcome in ("PASS", "FAIL"):
                if (result_info["result"] in ("PASS", "FAIL")
                        and result_info["result"] != sent_outcome):
                    print(f"[runner] WARN: stdout said {result_info['result']} but "
                          f"sentinel says {sent_outcome}; trusting sentinel.", flush=True)
                result_info["result"] = sent_outcome
                if sent_manifest:
                    result_info["output_file"] = sent_manifest
                exit_reason = sentinel.get("exit_reason", "ok")
                if not result_info["result_summary"] and exit_reason and exit_reason != "ok":
                    result_info["result_summary"] = f"sentinel exit_reason={exit_reason}"
            else:
                print(f"[runner] WARN: sentinel for {queue_id} has invalid "
                      f"outcome={sent_outcome!r}; classifying ERROR", flush=True)
                result_info["result"] = "ERROR"
                result_info["result_summary"] = f"sentinel invalid outcome: {sent_outcome!r}"
        else:
            # Sentinel missing -- script did not call emit_outcome, OR was
            # killed before reaching the call, OR ran on a stale binary
            # without the protocol module. Classify ERROR (NOT UNKNOWN);
            # downstream branches keep the queue item in place if the
            # script is not yet retrofitted (legacy stdout result still
            # surfaces in result_summary).
            _ns_result, _ns_summary = _classify_no_sentinel_result(
                result_info["result"], exit_code)
            if _ns_summary is None:
                # Trusted legacy stdout path (clean exit + stdout PASS/FAIL):
                # flag the missing sentinel so the user can retrofit the script.
                print(f"[runner] NOTE: no sentinel for {queue_id}; using "
                      f"legacy stdout-derived result {result_info['result']} "
                      f"(retrofit experiment_protocol.emit_outcome to silence)",
                      flush=True)
            else:
                # Crash (non-zero exit) or clean-exit-with-no-verdict -> ERROR.
                # A partial stdout verdict from a crashed run is NOT trusted
                # (V3-EXQ-624 incident 2026-06-02: crashed item was recorded
                # PASS off stdout and left in queue as a fleet crash-magnet).
                if exit_code == 0:
                    _ns_summary = (
                        f"No runner sentinel emitted and no PASS/FAIL on stdout "
                        f"(exit={exit_code}; secs={round(time.monotonic() - started_at, 1)})"
                    )
                result_info["result"] = _ns_result
                result_info["result_summary"] = _ns_summary

        # Belt-and-braces: a true non-zero exit with no positive verdict from
        # either source is always ERROR.
        if exit_code != 0 and result_info["result"] == "UNKNOWN":
            result_info["result"] = "ERROR"
            result_info["result_summary"] = (
                result_info["result_summary"] or f"Non-zero exit code {exit_code}"
            )

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
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the regression-suite preflight layer. Also honoured via "
             "REE_SKIP_PREFLIGHT=1 (useful when a preflight test itself is broken).",
    )
    parser.add_argument(
        "--remote-control",
        action="store_true",
        help="Write per-machine heartbeat to REE_assembly each loop tick "
             "(REE_assembly/evidence/experiments/runner_heartbeats/<hostname>.json). "
             "Phase 1 of the multi-machine dashboard; off by default.",
    )
    parser.add_argument(
        "--laptop-yield-to-cloud",
        dest="laptop_yield_to_cloud",
        action="store_true",
        default=None,
        help="On the laptop, skip 'any'-affinity items while at least one cloud "
             "worker (ree-cloud-1..4) has a fresh runner_heartbeat. Auto-armed "
             f"when hostname == {LAPTOP_AUTO_YIELD_HOSTNAME!s}; off elsewhere. "
             "Use --no-laptop-yield-to-cloud to force off on the laptop.",
    )
    parser.add_argument(
        "--no-laptop-yield-to-cloud",
        dest="laptop_yield_to_cloud",
        action="store_false",
        help="Disable laptop-yield-to-cloud even on the laptop (force the "
             "laptop to claim 'any'-affinity items alongside cloud workers).",
    )
    parser.add_argument(
        "--laptop-yield-freshness-min",
        type=int,
        default=35,
        metavar="MINUTES",
        help="Freshness threshold (minutes) for the cloud-worker heartbeat "
             "check used by --laptop-yield-to-cloud. Default: 35 minutes. "
             "This must accommodate the phase3 sync_daemon heartbeat writer's "
             "30-min liveness floor (an idle-but-alive cloud worker may not "
             "have its heartbeat file refreshed on REE_assembly for up to "
             "~30 min). A tighter threshold (e.g. 3 min) will make the laptop "
             "fail to yield to alive-but-idle cloud workers.",
    )
    args = parser.parse_args()

    if args.remote_control and _rrc is None:
        print(f"[runner] --remote-control requested but module import failed: "
              f"{_rrc_import_error}. Heartbeats disabled.", flush=True)

    # Pull ree-v3 before preflight so a stale local queue doesn't block startup.
    # (The full auto-sync pull of REE_assembly happens after preflight as before.)
    if args.auto_sync:
        git_pull(REPO_ROOT, "ree-v3")

    if not args.skip_preflight and os.environ.get("REE_SKIP_PREFLIGHT") != "1":
        preflight_dir = REPO_ROOT / "tests" / "preflight"
        if preflight_dir.exists():
            print(f"[runner] Preflight: running {preflight_dir}", flush=True)
            rc = subprocess.call(
                [sys.executable, "-m", "pytest", "-q", "--tb=line", str(preflight_dir)],
                cwd=str(REPO_ROOT),
            )
            if rc != 0:
                print(
                    "[runner] Preflight FAILED (exit {}). Fix failing tests or "
                    "re-run with --skip-preflight to bypass.".format(rc),
                    flush=True,
                )
                sys.exit(rc)
            print("[runner] Preflight: OK", flush=True)
        else:
            print("[runner] Preflight: tests/preflight not found -- skipping.", flush=True)

    machine = _get_machine_name(args.machine)
    status_path = args.status_file or find_default_status_path(machine)
    ree_assembly_path = find_ree_assembly_path()
    print(f"[runner] Status file: {status_path}", flush=True)
    print(f"[runner] Queue file:  {QUEUE_FILE}", flush=True)
    print(f"[runner] Machine identity: {machine}", flush=True)

    # Resolve --laptop-yield-to-cloud auto-arming. None (default) -> on iff
    # we're on the laptop. Explicit True/False from the CLI overrides.
    if args.laptop_yield_to_cloud is None:
        args.laptop_yield_to_cloud = (
            socket.gethostname() == LAPTOP_AUTO_YIELD_HOSTNAME
        )
    if args.laptop_yield_to_cloud:
        print(
            f"[runner] Laptop-yield-to-cloud: ON "
            f"(skip 'any' items while any of {LAPTOP_YIELD_CLOUD_HOSTS} "
            f"has a heartbeat within {args.laptop_yield_freshness_min} min)",
            flush=True,
        )
    if args.remote_control and _rrc is not None and ree_assembly_path:
        print(f"[runner] Remote-control: ON (heartbeats -> "
              f"{ree_assembly_path}/evidence/experiments/runner_heartbeats/)", flush=True)
        _rrc.write_heartbeat(
            ree_assembly_path, machine, state="starting",
            runner_pid=os.getpid(),
        )
    elif args.remote_control and not ree_assembly_path:
        print("[runner] --remote-control requested but REE_assembly not found; "
              "heartbeats disabled.", flush=True)
    if args.auto_sync:
        if ree_assembly_path:
            print(f"[runner] Auto-sync: ON (REE_assembly: {ree_assembly_path})", flush=True)
            if coordinator_claims_authoritative():
                print("[runner] Coordination: coordinator claims authoritative; "
                      "git remains status/result/queue transport", flush=True)
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

    # Graceful-drain state: SIGTERM / remote stop only request drain. Immediate
    # termination is reserved for remote force_stop or a second interactive SIGINT.
    _drain_flag: list[bool] = []           # non-empty -> drain requested
    _current_proc: list[subprocess.Popen] = []  # 0 or 1 elements
    _pause_flag: list[bool] = []           # non-empty -> remote pause requested
    _suspend_flag: list[bool] = []        # non-empty -> terminate run, save partial
    _resume_run_target: list[str] = []    # optional queue_id for resume_run cmd
    _force_stop_flag: list[bool] = []      # non-empty -> remote force_stop requested
    _sigint_force_armed: list[bool] = []   # non-empty -> next SIGINT force-stops

    def _announce_intentional_shutdown(reason: str) -> None:
        """Best-effort POST /shutdown_notify. coordinator_client swallows
        every error in _post, so this never raises. No-op when the runner
        is in git-mode (worker not in the coordinator state graph)."""
        try:
            coordinator_client.report_shutdown(
                machine=machine, reason=reason)
        except Exception:
            # Defense in depth -- coordinator_client already swallows
            # exceptions; this catch is for an import-time failure or
            # similar that wouldn't go through _post.
            pass

    def _do_immediate_exit() -> None:
        """Final cleanup steps shared by both force-exit and post-drain exit."""
        _announce_intentional_shutdown("runner_signal_exit")
        if args.auto_sync and _current_claim:
            release_active_claim(QUEUE_FILE, _current_claim[0], machine)
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
        is_sigint = sig == signal.SIGINT
        if is_sigint and _sigint_force_armed:
            print(f"\n[runner] Second SIGINT -- force-stopping now.", flush=True)
            if _current_proc:
                try:
                    _current_proc[0].kill()
                except Exception:
                    pass
            _do_immediate_exit()
            sys.exit(0)

        if _drain_flag:
            print(f"\n[runner] Signal {sig} received while drain is already pending.",
                  flush=True)
            if is_sigint and not _sigint_force_armed:
                _sigint_force_armed.append(True)
                print("[runner] Send SIGINT again to force-stop immediately.",
                      flush=True)
            else:
                print("[runner] SIGTERM remains graceful-only; use remote "
                      "force_stop for an immediate kill.", flush=True)
            return

        # First stop signal: request graceful drain.
        print(f"\n[runner] Caught signal {sig} -- will stop after current experiment finishes.",
              flush=True)
        if is_sigint:
            _sigint_force_armed.append(True)
            print("[runner] Send SIGINT again to force-stop immediately.", flush=True)
        else:
            print("[runner] SIGTERM is graceful-only; use remote force_stop for "
                  "an immediate kill.", flush=True)
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
    # force_rerun items bypass this skip (see should_skip_as_completed).
    completed_ids = {c["queue_id"] for c in existing_completed} | _peer_ids
    force_rerun_by_id = {
        i["queue_id"]: i.get("force_rerun")
        for i in items
        if i.get("queue_id")
    }

    # Prune already-completed items from queue display (keep force_rerun items visible)
    status["queue"] = [
        qi for qi in status["queue"]
        if not should_skip_as_completed(
            {**qi, "force_rerun": force_rerun_by_id.get(qi.get("queue_id"))},
            completed_ids,
        )
    ]
    write_status(status, status_path)

    # Collect output files written during this pass so git_push_results can
    # stage them selectively instead of sweeping evidence/experiments/.
    _result_files_this_pass: list[str] = []

    # Reset every outer pass. Three downstream paths (INFRA-CRASH,
    # UNKNOWN-result, manifest-missing) want to skip a queue item for the
    # rest of the current pass while leaving it in the queue for later. Use
    # this per-pass set instead of completed_ids -- adding to completed_ids
    # poisons the runner's in-memory view of the queue for its entire
    # lifetime, so a worker that hits one of those paths silently stops
    # considering the item even after the underlying issue is fixed or the
    # entry is re-queued. Reproduced on ree-cloud-2 / 2026-05-29 when 569a
    # and 612c both hit manifest-missing and the runner then sat idle
    # despite 612c carrying its affinity.
    _pass_skip: set[str] = set()

    while True:
        ran_any = False
        _pass_skip.clear()

        # Drain any pending remote-control commands at the top of each pass
        # (before claiming the next experiment). Commands like 'stop', 'pause',
        # 'kick', 'release_claim' need to take effect before the inner loop
        # picks up its next item.
        if args.remote_control and _rrc is not None and ree_assembly_path:
            _rrc.process_pending_commands(
                ree_assembly_path, machine, QUEUE_FILE,
                drain_flag=_drain_flag,
                pause_flag=_pause_flag,
                force_stop_flag=_force_stop_flag,
                suspend_flag=_suspend_flag,
                resume_run_target=_resume_run_target,
                current_proc=_current_proc,
                auto_sync=args.auto_sync,
                # `reclassify` needs to mutate status["completed"] and
                # rewrite runner_status.<machine>.json atomically. The
                # next heartbeat POST carries the corrected payload to
                # the coordinator.
                status_ref=status,
                status_path=status_path,
                write_status_fn=write_status,
            )
            if _force_stop_flag:
                print("[runner] Remote force_stop received -- exiting.", flush=True)
                break
            if _drain_flag:
                print("[runner] Remote stop received -- exiting before next claim.",
                      flush=True)
                break
            if _pause_flag:
                # Skip the inner experiment loop entirely while paused.
                # Heartbeat update at the bottom will record state=paused.
                pass

        for item in items:
            if _pause_flag:
                break  # paused: don't pick up new experiments this pass
            queue_id = item["queue_id"]

            if queue_id in _pass_skip:
                # Earlier this pass we hit a "leave-in-queue" outcome on this
                # item (manifest missing, infra crash with no sentinel, or
                # UNKNOWN result). Skip it for the rest of this pass and let
                # the next loop tick re-evaluate.
                continue

            if _resume_run_target and queue_id != _resume_run_target[0]:
                continue

            if item.get("status") == "suspended":
                if not _resume_run_target or queue_id != _resume_run_target[0]:
                    continue
                try:
                    qdata = json.loads(QUEUE_FILE.read_text())
                    for qi in qdata.get("items", []):
                        if qi.get("queue_id") == queue_id:
                            qi["status"] = "pending"
                            break
                    _atomic_write_queue(QUEUE_FILE, qdata)
                    item = next(i for i in qdata["items"] if i["queue_id"] == queue_id)
                except Exception as _re:
                    print(f"[runner] warn: could not reopen suspended {queue_id}: {_re}",
                          flush=True)

            if should_skip_as_completed(item, completed_ids):
                continue
            if item_has_force_rerun(item) and queue_id in completed_ids:
                print(f"[runner] force_rerun: {queue_id} -- running again despite "
                      f"prior completion record", flush=True)

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
                    if args.auto_sync and ree_assembly_path:
                        git_push_status(ree_assembly_path, status_path, queue_id)
                    try:
                        qdata = json.loads(QUEUE_FILE.read_text())
                        qdata["items"] = [qi for qi in qdata.get("items", [])
                                          if qi.get("queue_id") != queue_id]
                        _atomic_write_queue(QUEUE_FILE, qdata)
                        if args.auto_sync:
                            git_push_queue()
                    except Exception as _qe:
                        print(f"[runner] warn: could not remove {queue_id} from queue file: {_qe}",
                              flush=True)
                    if args.auto_sync:
                        coordinator_client.report_queue_remove(queue_id, "FAIL")
                completed_ids.add(queue_id)
                continue

            # Skip experiments assigned to a different machine
            if not _affinity_matches(item, machine):
                print(f"[runner] Skipping {queue_id} -- affinity={item.get('machine_affinity')} "
                      f"(this machine: {machine})", flush=True)
                continue

            # Lever 1: laptop yields to cloud on 'any'-affinity items while at
            # least one cloud worker is fresh AND idle (available to claim). If
            # every cloud worker is alive-but-busy the laptop runs the item
            # itself rather than letting it starve. Items pinned to this host
            # are unaffected (the affinity check above let them through).
            if args.laptop_yield_to_cloud:
                yield_, available_host = _should_yield_to_cloud(
                    item, args.laptop_yield_freshness_min, ree_assembly_path,
                )
                if yield_:
                    print(
                        f"[runner] Yielding {queue_id} (affinity=any) to "
                        f"{available_host} -- cloud worker idle and fresh "
                        f"within {args.laptop_yield_freshness_min}min",
                        flush=True,
                    )
                    continue
                else:
                    print(
                        f"[runner] Not yielding {queue_id} (affinity=any) -- "
                        f"no idle cloud worker available (fleet busy or stale); "
                        f"running locally",
                        flush=True,
                    )

            # Skip experiments already claimed by another active machine
            if not coordinator_claims_authoritative():
                existing_claim = item.get("claimed_by")
                if (existing_claim
                        and existing_claim.get("machine") != machine
                        and not _is_stale_claim(existing_claim, queue_id)):
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

            # In auto-sync mode, acquire the selected claim mutex before
            # running. Default/shadow use git; coordinator mode uses the
            # SQLite coordinator and never runs on a claim error.
            if args.auto_sync:
                claim_result = acquire_claim(QUEUE_FILE, queue_id, machine)
                if claim_result == "already_claimed":
                    print(f"[runner] {queue_id} -- claim lost to another machine, skipping",
                          flush=True)
                    continue
                if claim_result == "error":
                    if coordinator_claims_authoritative():
                        print(f"[runner] {queue_id} -- coordinator claim unavailable; "
                              "will retry later", flush=True)
                        continue
                    print(f"[runner] {queue_id} -- claim push failed (network?), "
                          f"running anyway", flush=True)
                # "ok" or "error" -> proceed; track for signal handler
                _current_claim.clear()
                _current_claim.append(queue_id)

            try:
                result = run_experiment(item, status, status_path, calibration, script_timing,
                                        proc_ref=_current_proc,
                                        suspend_flag=_suspend_flag,
                                        auto_sync=args.auto_sync,
                                        ree_assembly_path=ree_assembly_path,
                                        remote_control=args.remote_control,
                                        machine=machine)
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
            if _resume_run_target:
                _resume_run_target.clear()

            if result.get("result") == "SUSPENDED":
                print(f"[runner] {queue_id} suspended -- checkpoint kept, "
                      "claim retained; draining for runner restart", flush=True)
                try:
                    qdata = json.loads(QUEUE_FILE.read_text())
                    for qi in qdata.get("items", []):
                        if qi.get("queue_id") == queue_id:
                            qi["status"] = "suspended"
                            if result.get("checkpoint_path"):
                                qi["checkpoint_path"] = result["checkpoint_path"]
                            qi["suspended_at"] = now_utc()
                            break
                    _atomic_write_queue(QUEUE_FILE, qdata)
                    if args.auto_sync:
                        git_push_queue()
                except Exception as _se:
                    print(f"[runner] warn: could not mark {queue_id} suspended: {_se}",
                          flush=True)
                status["current"] = None
                write_status(status, status_path)
                if not _drain_flag:
                    _drain_flag.append(True)
                break

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

            # Detect transient infrastructure crashes (OOM, SIGKILL, SIGTERM) vs
            # genuine script errors. Exit code 137 = SIGKILL on Linux (OOM-killer).
            # Negative codes = killed by signal directly. 143 = SIGTERM as reported
            # by shell wrappers (128 + 15). These must NOT permanently remove the
            # queue item -- release the claim so another machine (or the next pass)
            # can retry.
            # Root cause: Hetzner CX22 2-shared-vCPU OOM kills emit code 137; the
            # script dies before emit_outcome() runs, so no sentinel is written, which
            # caused the belt-and-braces block to convert UNKNOWN -> ERROR, and the
            # ERROR path below to permanently drop the experiment. This block intercepts
            # that case before it reaches the permanent-removal path.
            # SIGTERM addendum (2026-05-30 fleet incident): cloud-scaler killed
            # ree-cloud-2 and ree-cloud-3 mid-experiment via systemd SIGTERM. The
            # subprocesses returned exit_code=-15 (or 143), which previously fell
            # through to the ERROR branch and wrote a phantom completion row with
            # no manifest, then tripped preflight test_queue_integrity.py on the
            # next boot and blocked startup. Treating -15/143 as infra-crash (no
            # completion written, claim released) is the SIGTERM-mid-experiment fix.
            _transient_exit_codes = {137, -9, -11, -15, 143}  # SIGKILL (OOM), SIGKILL direct, SIGSEGV, SIGTERM (neg), SIGTERM (shell)
            _run_exit_code = result.get("exit_code")
            _run_has_sentinel = result.get("has_sentinel", False)
            _is_infra_crash = (
                result["result"] == "ERROR"
                and _run_exit_code is not None
                and _run_exit_code in _transient_exit_codes
                and not _run_has_sentinel  # only when script never called emit_outcome
            )
            if _is_infra_crash:
                _crash_kind = "SIGTERM" if _run_exit_code in (-15, 143) else "OOM/SIGKILL"
                print(
                    f"[runner] INFRA-CRASH: {queue_id} exit={_run_exit_code} "
                    f"(likely {_crash_kind}); leaving in queue, releasing claim, "
                    f"no completion written. actual_secs={result.get('actual_secs')}",
                    flush=True,
                )
                if args.auto_sync:
                    release_active_claim(QUEUE_FILE, queue_id, machine)
                _pass_skip.add(queue_id)  # don't re-pick this pass
                status["current"] = None
                write_status(status, status_path)
                continue

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
                # V3-EXQ-592b autopsy fix (2026-05-29): when an ERROR result
                # claims a manifest (output_file non-empty), enforce the same
                # manifest-existence contract the PASS branch uses (line ~2331).
                # An ERROR with empty output_file is a normal script crash and
                # the existing flow proceeds to queue removal. An ERROR that
                # NAMES a manifest but the manifest is missing on disk is the
                # FAIL/ERROR-class counterpart of the line-1394 UNKNOWN bug;
                # leave the queue entry in place for operator investigation.
                _err_manifest_str = result.get("output_file") or ""
                if _err_manifest_str and not _result_manifest_exists(result):
                    print(f"[runner] WARN: {queue_id} reports ERROR but manifest "
                          f"{_err_manifest_str!r} is missing on disk or empty. "
                          f"Leaving in queue; investigate before requeueing.",
                          flush=True)
                    if args.auto_sync:
                        release_active_claim(QUEUE_FILE, queue_id, machine)
                    _pass_skip.add(queue_id)
                    continue
                # Ship manifest BEFORE queue removal (mirrors PASS branch at
                # line ~2397-2412). When output_file is non-empty and the
                # manifest exists, push it to REE_assembly and report to the
                # coordinator so the manifest reaches origin/master before the
                # queue removal commits the experiment as "done".
                if args.auto_sync and ree_assembly_path and _err_manifest_str:
                    try:
                        git_push_results(ree_assembly_path, [result["output_file"]])
                    except Exception as _re:
                        print(f"[runner] warn: per-experiment ERROR results push "
                              f"failed for {queue_id}: {_re}", flush=True)
                    _report_result_and_align(
                        ree_assembly_path, queue_id, result.get("run_id"),
                        result["output_file"], result["result"], machine)
                try:
                    qdata = json.loads(QUEUE_FILE.read_text())
                    qdata["items"] = [qi for qi in qdata.get("items", [])
                                      if qi.get("queue_id") != queue_id]
                    _atomic_write_queue(QUEUE_FILE, qdata)
                    if args.auto_sync:
                        git_push_queue()
                except Exception as _qe:
                    print(f"[runner] warn: could not remove {queue_id} from queue file: {_qe}",
                          flush=True)
                if args.auto_sync:
                    coordinator_client.report_queue_remove(queue_id, "ERROR")
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
                # V3-EXQ-592b autopsy fix (2026-05-29): the FAIL branch
                # previously skipped the manifest-existence check, the
                # git_push_results call, and coordinator_client.report_result
                # that the PASS branch runs (line ~2326-2412). The 592b
                # silent-drop on DLAPTOP-4 2026-05-29T08:32:39Z went straight
                # from completed list append to report_queue_remove with no
                # manifest reaching origin/master and an empty coordinator
                # results row. Enforce the same manifest contract as PASS:
                # missing manifest leaves the queue entry in place for
                # operator investigation; present manifest is shipped before
                # the queue removal commits the experiment as "done".
                _fail_manifest_str = result.get("output_file") or ""
                if not _result_manifest_exists(result):
                    print(f"[runner] WARN: {queue_id} reports FAIL but manifest "
                          f"{_fail_manifest_str!r} is missing on disk or empty. "
                          f"Leaving in queue; investigate before requeueing.",
                          flush=True)
                    if args.auto_sync:
                        release_active_claim(QUEUE_FILE, queue_id, machine)
                    _pass_skip.add(queue_id)
                    continue
                if args.auto_sync and ree_assembly_path:
                    try:
                        git_push_results(ree_assembly_path, [result["output_file"]])
                    except Exception as _re:
                        print(f"[runner] warn: per-experiment FAIL results push "
                              f"failed for {queue_id}: {_re}", flush=True)
                    _report_result_and_align(
                        ree_assembly_path, queue_id, result.get("run_id"),
                        result["output_file"], result["result"], machine)
                try:
                    qdata = json.loads(QUEUE_FILE.read_text())
                    qdata["items"] = [qi for qi in qdata.get("items", [])
                                      if qi.get("queue_id") != queue_id]
                    _atomic_write_queue(QUEUE_FILE, qdata)
                    if args.auto_sync:
                        git_push_queue()
                except Exception as _qe:
                    print(f"[runner] warn: could not remove {queue_id} from queue file: {_qe}",
                          flush=True)
                if args.auto_sync:
                    coordinator_client.report_queue_remove(queue_id, "FAIL")
                print(f"[runner] FAIL: {queue_id} -- moved to completed, continuing to next",
                      flush=True)
                continue

            # UNKNOWN result MUST NOT reach the success branch. Without a
            # PASS/FAIL/ERROR classification we cannot safely remove the
            # queue item. Release the claim, leave the entry in the queue,
            # and surface loudly. Pre-2026-05-08 the fall-through on
            # line 1394 let UNKNOWN reach the queue-removal block here,
            # silently dropping V3-EXQ-433f / 537 / 538 on cloud-1.
            if result["result"] == "UNKNOWN":
                print(f"[runner] UNKNOWN result for {queue_id} (no sentinel and "
                      f"no stdout verdict); leaving in queue, releasing claim. "
                      f"actual_secs={result.get('actual_secs')}, "
                      f"output_file={result.get('output_file')!r}",
                      flush=True)
                if args.auto_sync:
                    release_active_claim(QUEUE_FILE, queue_id, machine)
                _pass_skip.add(queue_id)  # don't re-pick this pass
                status["current"] = None
                write_status(status, status_path)
                continue

            # Verify the manifest exists before declaring done. A PASS/FAIL
            # outcome with a missing manifest is a contract violation; do
            # not remove the queue entry. (sentinel.manifest_path was
            # already validated for str-shape; check existence here.)
            manifest_str = result.get("output_file") or ""
            if not _result_manifest_exists(result):
                print(f"[runner] WARN: {queue_id} reports {result['result']} "
                      f"but manifest {manifest_str!r} is missing on disk or empty. "
                      f"Leaving in queue; investigate before requeueing.",
                      flush=True)
                if args.auto_sync:
                    release_active_claim(QUEUE_FILE, queue_id, machine)
                _pass_skip.add(queue_id)  # don't re-pick this pass
                status["current"] = None
                write_status(status, status_path)
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

            # Push status to REE_assembly BEFORE queue push. Invariant: GitHub's
            # status is always at least as fresh as the queue. Otherwise a
            # crash between queue-push and end-of-pass results-push loses the
            # status entry permanently (406a/429a/430a on 2026-04-18).
            if args.auto_sync and ree_assembly_path:
                git_push_status(ree_assembly_path, status_path, queue_id)
                # SHADOW (no-op unless COORDINATION_MODE=shadow).
                # PLAN.md step 6: send the FULL status file content so
                # sync_daemon can materialise runner_status/<machine>.json
                # from the coordinator DB. Falls back to the legacy stub
                # if the local file is unreadable for any reason -- the
                # coordinator's record_status_payload tolerates either
                # shape (both are dicts; the writer prefers the rich
                # form when present).
                full_status = None
                if _phase3_hub_local_ree_assembly_writes_gated():
                    full_status = status
                else:
                    try:
                        if status_path.exists():
                            full_status = json.loads(
                                status_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, UnicodeDecodeError):
                        # UnicodeDecodeError catches the Windows-runner case
                        # where the platform default text encoding doesn't
                        # match the UTF-8 write_status produced. Stub fallback
                        # below preserves the legacy report shape.
                        full_status = None
                coordinator_client.report_status(
                    machine,
                    full_status if isinstance(full_status, dict) else {
                        "last_completed": queue_id,
                        "result": result["result"],
                    })

            # Push results BEFORE queue removal. Otherwise a Hetzner-style
            # mid-pass shutdown between queue-push and end-of-pass results-push
            # strands the manifest on the dying VM (the V3-EXQ-483b SIGTERM
            # signature on 2026-05-08). Same invariant as status above.
            if args.auto_sync and ree_assembly_path and result.get("output_file"):
                try:
                    git_push_results(ree_assembly_path, [result["output_file"]])
                except Exception as _re:
                    print(f"[runner] warn: per-experiment results push "
                          f"failed for {queue_id}: {_re}", flush=True)
                # SHADOW (no-op unless COORDINATION_MODE=shadow): ship the
                # manifest bytes to the coordinator. Idempotent on run_id
                # server-side; the shim swallows any path/IO error.
                _report_result_and_align(
                    ree_assembly_path, queue_id, result.get("run_id"),
                    result["output_file"], result["result"], machine)

            # Remove completed item from queue file -- runner_status.json is the
            # authoritative record of what has run; queue file should only contain
            # pending work to avoid silent accumulation of stale entries.
            try:
                qdata = json.loads(QUEUE_FILE.read_text())
                qdata["items"] = [qi for qi in qdata.get("items", [])
                                   if qi.get("queue_id") != queue_id]
                _atomic_write_queue(QUEUE_FILE, qdata)
                if args.auto_sync:
                    git_push_queue()
            except Exception as _qe:
                print(f"[runner] warn: could not update queue file for {queue_id}: {_qe}", flush=True)

            # SHADOW (no-op unless COORDINATION_MODE=shadow)
            coordinator_client.report_queue_remove(queue_id, result["result"])

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

        if args.remote_control and _rrc is not None and ree_assembly_path:
            queue_pending = [
                qi for qi in status.get("queue", [])
                if not should_skip_as_completed(
                    {**qi, "force_rerun": force_rerun_by_id.get(qi.get("queue_id"))},
                    completed_ids,
                )
            ]
            head_id = queue_pending[0]["queue_id"] if queue_pending else None
            recent = status.get("completed", [])[-5:]
            hb_state = "paused" if _pause_flag else (
                "draining" if _drain_flag else "idle"
            )
            hb_path = _rrc.write_heartbeat(
                ree_assembly_path, machine, state=hb_state,
                queue_depth=len(queue_pending),
                queue_id_at_head=head_id,
                recent_completed=[
                    {
                        "queue_id": c.get("queue_id"),
                        "result": c.get("result"),
                        "completed_at": c.get("completed_at"),
                    }
                    for c in recent
                ],
                runner_pid=os.getpid(),
            )
            if args.auto_sync and hb_path is not None:
                _rrc.push_heartbeat(ree_assembly_path, hb_path)

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
        force_rerun_by_id = {
            i["queue_id"]: i.get("force_rerun")
            for i in items
            if i.get("queue_id")
        }

        new_pending = [i for i in items if not should_skip_as_completed(i, completed_ids)]
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
        # Drain was signal-induced -- announce intentional shutdown so the
        # coordinator returns lifecycle_state=gracefully_offline (the
        # scaler workflow's pre-shutdown announce is the primary signal;
        # this fires for paths the scaler doesn't see: manual systemctl
        # stop, SSH-issued shutdown, remote /api/machines/.../command
        # stop). NOT announced for natural queue exhaustion (--once mode)
        # because the machine is staying up, only the process is exiting.
        _announce_intentional_shutdown("runner_drain_complete")
    else:
        print("[runner] Queue exhausted. Runner idle.", flush=True)

    if args.auto_sync and ree_assembly_path:
        git_push_results(ree_assembly_path, _result_files_this_pass or None)

    if PID_FILE.exists():
        PID_FILE.unlink()


if __name__ == "__main__":
    main()
