#!/usr/bin/env python3
"""Pause-the-puller substrate mutex -- option A2 of the structural-isolation
design in `REE_assembly/evidence/planning/substrate_stability_and_drift_detection_plan.md`
section 3.

THE PROBLEM
-----------
The runner executes each experiment IN PLACE, inside the same shared `ree-v3`
checkout that a co-resident loop periodically `git pull --rebase --autostash`es
against `origin/main` -- including while an experiment subprocess is mid-flight,
importing source files live from that same tree. So a `ree_core` commit landing
mid-run silently becomes part of what that run measured, and the result can no
longer be attributed to one substrate revision. Confirmed on V3-EXQ-875
(MECH-471): a ~20.5h run recorded `substrate_stable_across_run: false` because
six `ree_core`-touching commits landed inside its window. Benign that time
(all six were default-off flags the run never enabled); nothing STRUCTURAL
prevented it.

WHAT THIS BUILDS (A2, the plan's recommended default)
-----------------------------------------------------
A worker-scoped, purely LOCAL mutex. A running experiment holds a freeze; the
puller checks the freeze and defers a pull that would move substrate files
underneath it. No new checkouts, no disk copies -- the plan's "cheapest" option,
matching the fact that drift is usually benign.

This is deliberately NOT option A1 (a pinned `git worktree` per run), which the
plan reserves for experiments an author explicitly flags as expensive or
claim-decisive, and deliberately NOT option A3 (rsync snapshot), which the plan
REJECTED: it re-creates the `.git`-file-vs-directory trap `scripts/remote_pytest.sh`
had to fix for zero isolation benefit over A1. Do not re-litigate that.

DEFAULT OFF
-----------
`freeze_enabled()` is False unless `REE_SUBSTRATE_FREEZE` is set truthy, so an
unflagged fleet never acquires a freeze and behaves exactly as before. The CHECK
side (`pull_deferred_reason`) is deliberately NOT gated on the flag -- see its
docstring for why that is safe and strictly more useful than gating it.

FAIL-OPEN, UNIFORMLY -- AND THAT IS THE OPPOSITE OF THE SIBLING GUARD
---------------------------------------------------------------------
`experiment_runner._ree_v3_pull_blocked` fails CLOSED when it knows a session is
at risk and merely cannot confirm the tree state: a swept autostash is silent
and unrecoverable, so a deferred pull is the cheaper error there.

This guard takes the opposite direction on purpose. Every uncertain state here
-- unreadable lock dir, corrupt lock file, git unavailable, fetch failure --
resolves to "do not defer, let the pull run". Two reasons, both specific rather
than a general preference for leniency:

  1. The harm this guard prevents is ALREADY DETECTED after the fact.
     `arm_fingerprint.substrate_stability_report()` re-checks the substrate hash
     within the process and self-reports `substrate_stable_across_run: false` --
     that is exactly how V3-EXQ-875 was caught with none of this built. So a
     missed defer degrades to the status quo ante, loudly, in the manifest.
  2. The harm a fail-closed bug would cause is NOT detected and is fleet-wide.
     This guard sits in front of the only path that pulls `ree-v3`, and
     `experiment_queue.json` lives in `ree-v3`. A guard stuck in the blocking
     direction stops a worker seeing new work at all -- the "stale code and,
     worse, a stale queue" wedge `_ree_v3_pull_blocked` names as the reason it
     rejected a claim-only gate.

A guard that fails open degrades to no guard. A guard that fails closed takes
out the fleet's code and work sync at once. Do not "harden" this by inverting a
failure direction without re-reading both reasons above.

STALE FREEZES SELF-HEAL
-----------------------
A crashed experiment must not wedge the puller. A held freeze is honoured only
while its holder is provably live: same-host holders are liveness-checked by pid
(`os.kill(pid, 0)`), and EVERY holder is bounded by a TTL regardless. A freeze
failing either test is reaped on sight by the next reader. This mirrors the
stale-lock reclaim `ree_commit.py` already uses for its rebase lock rather than
inventing a new convention.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Repo-root-relative path prefixes whose movement mid-run is what this mutex
# exists to prevent. Kept in lockstep with
# `experiments/_lib/arm_fingerprint._SUBSTRATE_GLOBS` -- that tuple is what
# actually gets hashed into a run's substrate identity, so a path it hashes but
# this does not guard would drift undetected by the freeze while still flipping
# `substrate_stable_across_run`. Expressed as PREFIXES rather than globs because
# the consumer is `git diff --name-only`, which emits plain paths.
# Pinned by tests/contracts/test_substrate_freeze.py::test_prefixes_cover_arm_fingerprint_globs.
SUBSTRATE_PREFIXES: tuple[str, ...] = (
    "ree_core/",
    "experiments/_harness.py",
    "experiments/_metrics.py",
    "experiments/_lib/",
)

_FREEZE_DIRNAME = "ree_substrate_freeze"

# Default TTL backstop. Must comfortably exceed the longest plausible run --
# V3-EXQ-875 itself ran 20.5h -- because expiring a freeze under a still-running
# experiment silently reintroduces the exact drift this module prevents. The pid
# check is what actually reaps a crashed holder promptly; this is only the
# backstop for the cases pid cannot cover (pid reuse, a foreign-host holder).
DEFAULT_TTL_SECONDS = 26 * 3600

_TRUTHY = ("1", "true", "yes", "on")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def freeze_enabled() -> bool:
    """True when this runner should ACQUIRE a freeze around each experiment.

    Default False: with the flag unset nothing is ever acquired, so an unflagged
    fleet is behaviourally identical to the pre-A2 runner.
    """
    return _env_flag("REE_SUBSTRATE_FREEZE")


def freeze_ignored() -> bool:
    """Operator escape hatch: treat every held freeze as absent.

    For the case where a freeze is wedging a worker and the operator wants sync
    back NOW without hunting down the lock directory. Same spirit as
    `REE_SKIP_PUSH_CHECK` / `REE_ALLOW_REF_DISCARD` elsewhere in this repo.
    """
    return _env_flag("REE_SUBSTRATE_FREEZE_IGNORE")


def ttl_seconds() -> int:
    raw = os.environ.get("REE_SUBSTRATE_FREEZE_TTL_SECONDS", "").strip()
    if raw:
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
    return DEFAULT_TTL_SECONDS


def _git_dir(repo_root: Path) -> Path | None:
    """Resolve the real git directory for `repo_root`, handling BOTH shapes.

    In an ordinary clone `<root>/.git` is a DIRECTORY. In a `git worktree` it is
    a 72-byte FILE reading `gitdir: <abs path>`. That distinction is not
    academic here: it is the documented 2026-07-31 `remote_pytest.sh` defect
    (CLAUDE.md, "Running the test suite"), where a `.git` FILE slipped past a
    directory-shaped check and produced a phantom contract failure. A freeze
    stored under a mis-resolved git dir would be invisible to the puller reading
    the correct one -- i.e. a silently inert mutex, the worst failure available.

    Returns None when there is no usable git dir at all (e.g. the rsync-staged
    tree `remote_pytest.sh` builds, which deliberately excludes `.git`).
    """
    dot_git = repo_root / ".git"
    try:
        if dot_git.is_dir():
            return dot_git
        if dot_git.is_file():
            text = dot_git.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                target = text.split(":", 1)[1].strip()
                if target:
                    p = Path(target)
                    if not p.is_absolute():
                        p = (repo_root / p).resolve()
                    return p
    except OSError:
        return None
    return None


def freeze_dir(repo_root: Path) -> Path:
    """Directory holding this checkout's freeze files.

    Inside the git dir, never inside the working tree: a lock file in the tree
    would show up as untracked dirt, and this repo's runner has a whole family
    of untracked-path stash/clear recovery paths that would then have to reason
    about it. Being per-git-dir also gives exactly the right SCOPE -- a worktree
    has its own checkout of the substrate files and its own puller, so it gets
    its own freeze, independently of the main checkout.

    When there is no git dir at all, falls back to a temp-dir location keyed by
    a hash of the repo path. Still per-checkout, still never dirties the tree.
    """
    gd = _git_dir(repo_root)
    if gd is not None:
        return gd / _FREEZE_DIRNAME
    key = hashlib.sha256(str(repo_root.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"{_FREEZE_DIRNAME}_{key}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by another user. Alive for our purposes.
        return True
    except (OSError, OverflowError, ValueError):
        # Cannot tell -> do not claim it is dead. The TTL still bounds it.
        return True


def _holder_is_live(rec: dict, now: float) -> bool:
    """True if this freeze record should still be honoured.

    Two independent tests, both of which must pass:
      * TTL -- every holder is bounded, whatever host it came from.
      * pid -- applied ONLY when the record's host matches ours. A pid from
        another host means nothing locally, and checking it would reap a
        legitimate freeze the moment a foreign pid happened not to exist here.
        (Not reachable through `acquire()`, which is always local; guarded
        anyway so a hand-placed or copied lock file cannot be mis-adjudicated.)
    """
    try:
        acquired = float(rec.get("acquired_monotonic_wall", 0.0))
        ttl = float(rec.get("ttl_seconds", DEFAULT_TTL_SECONDS))
    except (TypeError, ValueError):
        return False
    if ttl <= 0 or now - acquired > ttl:
        return False
    if rec.get("host") == socket.gethostname():
        try:
            pid = int(rec.get("pid", -1))
        except (TypeError, ValueError):
            return False
        if pid <= 0 or not _pid_alive(pid):
            return False
    return True


def held(repo_root: Path) -> list[dict]:
    """Live freeze records for this checkout, reaping dead/stale/corrupt ones.

    Never raises. Returns [] on any unreadable state -- see the module docstring
    on why every uncertain path here resolves to "not frozen".
    """
    if freeze_ignored():
        return []
    live: list[dict] = []
    try:
        d = freeze_dir(repo_root)
        if not d.is_dir():
            return []
        now = time.time()
        for entry in sorted(d.glob("*.json")):
            try:
                rec = json.loads(entry.read_text(encoding="utf-8"))
                if not isinstance(rec, dict):
                    raise ValueError("not an object")
            except (OSError, ValueError):
                # A corrupt record can be neither aged nor pid-checked, so
                # honouring it would wedge the puller forever. Reap it.
                _unlink_quiet(entry)
                continue
            if _holder_is_live(rec, now):
                rec["_path"] = str(entry)
                live.append(rec)
            else:
                _unlink_quiet(entry)
    except OSError:
        return []
    return live


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def acquire(repo_root: Path, run_label: str, ttl: int | None = None) -> Path | None:
    """Take a freeze for `run_label`. Returns the token path, or None on failure.

    Never raises: a freeze that cannot be taken must degrade to "run without
    isolation", exactly as today, not to "refuse to run the experiment".
    """
    try:
        d = freeze_dir(repo_root)
        d.mkdir(parents=True, exist_ok=True)
        rec = {
            "run_label": str(run_label),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": _utc_now_iso(),
            "acquired_monotonic_wall": time.time(),
            "ttl_seconds": int(ttl if ttl is not None else ttl_seconds()),
            "repo_root": str(repo_root),
        }
        token = d / f"{os.getpid()}-{uuid.uuid4().hex[:8]}.json"
        tmp = token.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, sort_keys=True), encoding="utf-8")
        # Atomic publish: a reader must never see a half-written record.
        os.replace(tmp, token)
        return token
    except (OSError, ValueError):
        return None


def release(token: Path | None) -> None:
    """Drop a freeze. Never raises; safe on None and on an already-gone token."""
    if token is None:
        return
    _unlink_quiet(Path(token))


@contextlib.contextmanager
def hold(repo_root: Path, run_label: str, enabled: bool | None = None):
    """Hold a freeze for the duration of the block.

    `enabled=None` consults `freeze_enabled()`; when disabled this is a pure
    no-op, which is what keeps the unflagged runner path unchanged. The release
    is in a `finally`, so an experiment that raises cannot leak a freeze -- the
    same single-exit-point reasoning `git_pull`'s prepull-stash `finally` is
    built on.
    """
    if enabled is None:
        enabled = freeze_enabled()
    token = acquire(repo_root, run_label) if enabled else None
    try:
        yield token
    finally:
        release(token)


def _run_git(args: list[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def substrate_paths_incoming(repo_root: Path, timeout: int = 30) -> bool | None:
    """True if fetching origin would bring in changes under SUBSTRATE_PREFIXES.

    Returns None when it cannot be determined (no upstream, git failure,
    timeout) -- the caller must treat that as "do not defer".

    This is what keeps A2 from being unusable in practice. Deferring EVERY pull
    for the life of a freeze would also stall `experiment_queue.json`, which
    lives in this same repo -- a 20h experiment would leave its worker blind to
    new work for 20h. Fetching first and inspecting what is actually incoming
    means the common case (a queue snapshot, a docs commit, a new experiment
    driver) pulls normally and only genuine substrate movement defers.

    `git fetch` is safe under a freeze: it writes only to the object store and
    the remote-tracking refs, never to the working tree the running experiment
    is importing from.
    """
    fetched = _run_git(["git", "fetch", "-q"], repo_root, timeout)
    if fetched is None or fetched.returncode != 0:
        return None
    up = _run_git(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
        repo_root, 15,
    )
    if up is None or up.returncode != 0:
        return None
    upstream = (up.stdout or "").strip()
    if not upstream:
        return None
    diff = _run_git(["git", "diff", "--name-only", "HEAD", upstream], repo_root, 30)
    if diff is None or diff.returncode != 0:
        return None
    for line in (diff.stdout or "").splitlines():
        path = line.strip().strip('"')
        if path.startswith(SUBSTRATE_PREFIXES):
            return True
    return False


def pull_deferred_reason(repo_root: Path) -> str | None:
    """The puller's question: should a `ree-v3` pull be deferred right now?

    Returns a one-line ASCII reason to log, or None to pull normally.

    NOT gated on `freeze_enabled()`, deliberately. The flag says whether THIS
    process should acquire a freeze; it should not decide whether an EXISTING
    freeze is honoured. With the flag off nothing ever acquires, so this is a
    single `scandir` of a directory that does not exist -- no behaviour change
    -- while a freeze taken by any other holder (a flagged sibling runner, a
    future A1 implementation, an operator) is still respected. Gating it would
    make the mutex silently one-sided in exactly the mixed-configuration case
    where it matters most.

    Never raises. Every uncertain state returns None; see the module docstring.
    """
    try:
        holders = held(repo_root)
        if not holders:
            return None
        incoming = substrate_paths_incoming(repo_root)
        if incoming is not True:
            # False (nothing substrate-y inbound) and None (undeterminable)
            # both mean "do not defer" -- see the module docstring.
            return None
        labels = ", ".join(str(h.get("run_label", "?")) for h in holders[:3])
        if len(holders) > 3:
            labels += f", +{len(holders) - 3} more"
        return (
            f"substrate freeze held by {len(holders)} run(s) [{labels}] and "
            f"incoming commits touch substrate paths"
        )
    except Exception:
        return None
