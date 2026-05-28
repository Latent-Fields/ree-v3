"""Sync daemon.

PHASE 1 (shadow) behaviour -- the default behaviour:
  * Periodically read experiment_queue.json (git is authoritative in
    shadow) and reconcile the coordinator DB mirror to match it, so the
    coordinator's claim logic always evaluates against fresh state.
  * Claim-level shadow checks happen via runner POST /claim (git_verdict
    vs coordinator evaluate_claim). State-level pre-upsert reconcile was
    removed 2026-05-20 (false positives; see SOAK_LOG.md E1/E2).
  * It does NOT write git. Read-only on the queue file. No autostash, no
    rebase -- this daemon is structurally incapable of the failure class
    the whole project exists to remove.

PHASE 3 (authoritative) behaviour is present but guarded OFF: becoming the
sole git writer (commit result manifests, push, snapshot queue) only
activates when SYNC_MODE=authoritative AND --i-understand-phase3 is passed.
Stubbed deliberately; do not enable until Phase 1 has proven out.

PHASE 2 (claim cutover) behaviour is selected by SYNC_MODE=coordinator:
git remains the queue worklist and result/status transport, but the DB is
the claim authority. Reconciliation refreshes metadata and removals from
git without overwriting coordinator claim state.

All printed text is ASCII-only.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import db
import manifest_spool

DEFAULT_QUEUE = os.path.join(
    os.path.dirname(__file__), "..", "experiment_queue.json")


def _load_queue_json(queue_path):
    """Return the parsed queue dict from the AUTHORITATIVE git ref
    (SYNC_QUEUE_REF, default origin/main), fetched read-only.

    Why not just read queue_path: the local working-tree copy is only as
    fresh as this box's last `git pull`. When this box's runner is drained
    nothing pulls, so the file goes stale and every other machine's
    git-claim looks like a state-divergence (mirror=claimed vs stale
    file=pending) -- a false positive, not a coordinator-logic fault.
    `git fetch` + `git show <ref>:file` never touches the working tree
    (no autostash risk, consistent with sync_daemon being git-read-only).
    Degrades to the local file if git is unavailable, logging that it is
    running on a possibly-stale source."""
    repo = os.path.dirname(os.path.abspath(queue_path))
    rel = os.path.basename(queue_path)
    ref = os.environ.get("SYNC_QUEUE_REF", "origin/main")
    try:
        subprocess.run(["git", "-C", repo, "fetch", "--quiet", "origin"],
                        check=True, capture_output=True, timeout=30)
        out = subprocess.run(
            ["git", "-C", repo, "show", "%s:%s" % (ref, rel)],
            check=True, capture_output=True, timeout=15)
        return json.loads(out.stdout.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- degrade, never crash
        if not os.path.exists(queue_path):
            sys.stderr.write(
                "[sync] no git queue and no local file: %r\n" % exc)
            return None
        sys.stderr.write(
            "[sync] WARN authoritative git queue unavailable (%r); "
            "falling back to STALE local file\n" % exc)
        with open(queue_path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def reconcile_once(conn, queue_path, *, claim_authority="git",
                   upsert_only=False):
    """Make the mirror match the AUTHORITATIVE git queue (not this box's
    possibly-stale local file). Returns (n_items, n_state_divergences).

    claim_authority='git' is Phase 1 shadow: git claims are truth and
    state-level mismatches are logged. claim_authority='coordinator' is
    Phase 2: git is only the worklist, so existing DB claim state is
    preserved and git-vs-DB claim mismatches are not divergence rows.

    upsert_only=False (default): items missing from the git queue but
    present in the DB are DELETEd (Phase 1/2 semantics: git is the
    worklist; what's gone from the file has been completed/removed by
    the authoritative path).

    upsert_only=True is Phase 3 authoritative: the DB owns the queue,
    `experiment_queue.json` is a DERIVED view written back by
    `phase3_queue_writer`. Operator hand-edits to the file MUST be
    additions only (use `POST /queue/remove` to drop items via the
    coordinator); items missing from the file are NOT deleted from the
    DB, because the DB row's status='completed' is the authoritative
    "this was done" record and must survive the writeback round-trip.
    """
    qdata = _load_queue_json(queue_path)
    if qdata is None:
        return (0, 0)
    items = {it["queue_id"]: it for it in qdata.get("items", [])
             if it.get("queue_id")}

    divergences = 0
    conn.execute("BEGIN IMMEDIATE")
    try:
        mirror = {r["queue_id"]: r for r in conn.execute(
            "SELECT queue_id, status, claimed_by_machine FROM experiments"
        ).fetchall()}

        for qid, item in items.items():
            # Phase 1 (git authority): upsert mirror from authoritative git
            # queue each tick. Pre-upsert state-reconcile logged false
            # divergences when the mirror was briefly ahead of origin/main
            # (harness E1 in SOAK_LOG.md) -- claim-level shadow /claim
            # compares git_verdict vs evaluate_claim instead.
            db.upsert_experiment(
                conn, item, preserve_claim=(claim_authority == "coordinator"))

        stale = set(mirror) - set(items)
        if not upsert_only:
            # Items no longer in the queue file have been completed/removed
            # by the authoritative path; drop them from the mirror so the
            # coordinator does not hand them out.
            for qid in stale:
                conn.execute(
                    "DELETE FROM experiments WHERE queue_id=?", (qid,))
        elif stale:
            # Phase 3 authoritative path: file is a DERIVED view, so items
            # missing from the file must NOT be deleted from the DB (the
            # row may be claimed/running/completed and the file just hasn't
            # caught up yet). But surface the silent revert: an operator
            # hand-edit removing a pending item will reappear in the file
            # on the next phase3_queue_writer tick with no signal. WARN
            # cheaply lists what was preserved so the operator can spot
            # an unintended revert in the log.
            non_terminal = [
                qid for qid in stale
                if mirror[qid]["status"] not in ("completed", "failed")
            ]
            if non_terminal:
                sys.stderr.write(
                    "[sync] upsert_only: %d non-terminal item(s) missing "
                    "from queue file are PRESERVED in DB (use "
                    "POST /queue/remove to drop): %s\n" % (
                        len(non_terminal), non_terminal[:5]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return (len(items), divergences)


# Set True only after phase3_git_writer steps 1-6 are implemented and
# phase3_preflight.py + phase3_verify.py pass on the live fleet.
PHASE3_GIT_WRITER_READY = True

# Hub paths (override via env when deploying). PHASE3_REE_ASSEMBLY and
# PHASE3_REE_V3 are validated below, AFTER the _validate_* helpers are
# defined (top-to-bottom Python module execution).
PHASE3_QUEUE_FILE = os.environ.get(
    "COORDINATOR_QUEUE_FILE", DEFAULT_QUEUE)


# Max manifests to write+commit per writer tick. Bounds tick latency and
# limits worst-case rollback if a push fails (we ROLLBACK committed_at for
# the unpushed batch, then retry). Must be >0; an operator setting it to
# 0 or negative via env override would produce silently misleading
# refusals (writer slices an empty batch out of a non-empty spool, hits
# "no manifests staged" and refuses for the wrong reason). Validate at
# module load and fall back to the default with a loud log.
#
# LOW-B note: env validation runs at MODULE IMPORT. Setting PHASE3_*
# env vars (this one, and the branch / relpath / interval validators
# below) after `import sync_daemon` is silently ignored. The daemon's
# main() reads the validated module-level constants. Operators must
# set the env in the systemd unit / launcher, before the Python
# process starts.
def _validate_batch_size(raw, default=32):
    try:
        v = int(raw)
    except (TypeError, ValueError):
        sys.stderr.write(
            "[phase3] PHASE3_BATCH_SIZE=%r is not an integer; "
            "using default %d\n" % (raw, default))
        return default
    if v <= 0:
        sys.stderr.write(
            "[phase3] PHASE3_BATCH_SIZE=%d is not > 0; using default %d\n"
            % (v, default))
        return default
    return v


PHASE3_BATCH_SIZE = _validate_batch_size(
    os.environ.get("PHASE3_BATCH_SIZE", "32"))


# LOW-C: same fail-loudly-then-default pattern for the other env knobs
# that can silently misdirect the writer when malformed. Empty string,
# whitespace-only, or values containing whitespace / path separators get
# rejected back to the documented default with a stderr warning.
def _validate_branch_name(raw, env_name, default):
    """Branch names: non-empty, no whitespace, no path separators."""
    if not isinstance(raw, str) or not raw.strip():
        sys.stderr.write(
            "[phase3] %s=%r is empty/blank; using default %r\n"
            % (env_name, raw, default))
        return default
    s = raw.strip()
    if any(c.isspace() for c in s) or "/" in s or "\\" in s:
        sys.stderr.write(
            "[phase3] %s=%r contains whitespace or path separator; "
            "using default %r\n" % (env_name, raw, default))
        return default
    return s


def _validate_abs_repo_path(raw, env_name, default):
    """Repo checkout absolute paths: non-empty, no embedded whitespace,
    absolute (POSIX) or default. The writer runs `git` with cwd=this; an
    empty or whitespace-only env override would silently target the
    daemon's cwd and write to whatever happened to be checked out there.
    Falls back to the documented default with a stderr warning.

    Existence is NOT checked here -- the writer's `_hub_working_tree_clean`
    + `git fetch` give clearer per-tick failures with the actual git error
    output, and a path that doesn't exist YET (e.g. fresh deploy before
    the checkout is created) shouldn't refuse module import."""
    if not isinstance(raw, str) or not raw.strip():
        sys.stderr.write(
            "[phase3] %s=%r is empty/blank; using default %r\n"
            % (env_name, raw, default))
        return default
    s = raw.strip()
    if any(c.isspace() for c in s):
        sys.stderr.write(
            "[phase3] %s=%r contains whitespace; using default %r\n"
            % (env_name, raw, default))
        return default
    if not os.path.isabs(s):
        sys.stderr.write(
            "[phase3] %s=%r is not absolute; using default %r\n"
            % (env_name, raw, default))
        return default
    return s


def _validate_repo_relpath(raw, env_name, default):
    """Repo-internal relative paths: non-empty, no leading slash, no
    `..` segment escape. The writer writes files to repo/<relpath>;
    accepting absolute paths or parent-dir escapes would let an env
    override write outside the managed checkout."""
    if not isinstance(raw, str) or not raw.strip():
        sys.stderr.write(
            "[phase3] %s=%r is empty/blank; using default %r\n"
            % (env_name, raw, default))
        return default
    s = raw.strip()
    # `os.path.isabs` covers both POSIX "/foo" and Windows drive paths.
    if os.path.isabs(s) or s.startswith("/") or s.startswith("\\"):
        sys.stderr.write(
            "[phase3] %s=%r is absolute; using default %r\n"
            % (env_name, raw, default))
        return default
    parts = s.replace("\\", "/").split("/")
    if any(p == ".." for p in parts):
        sys.stderr.write(
            "[phase3] %s=%r contains '..' segment; using default %r\n"
            % (env_name, raw, default))
        return default
    return s


def _validate_float(raw, env_name, default):
    """Float env knobs (SYNC_INTERVAL): parse with default fallback.
    Same shape as _validate_batch_size but for floats. Negative and
    zero are rejected -- a zero interval would spin the daemon."""
    try:
        v = float(raw)
    except (TypeError, ValueError):
        sys.stderr.write(
            "[phase3] %s=%r is not a number; using default %s\n"
            % (env_name, raw, default))
        return default
    if v <= 0:
        sys.stderr.write(
            "[phase3] %s=%s is not > 0; using default %s\n"
            % (env_name, v, default))
        return default
    return v


# PLAN.md step 5: queue snapshot writeback. Materialises the canonical
# experiment_queue.json from the coordinator DB and pushes ree-v3. Gated
# by its OWN flag (independent of the result writer) so result-cutover
# and queue-cutover can be staged separately. Default False until the
# implementation is reviewed AND the runner-side
# PHASE3_DISABLE_RUNNER_QUEUE_PUSH flag is set on every worker.
PHASE3_QUEUE_WRITER_READY = True

# Hub paths for the queue writer. ree-v3 checkout is separate from the
# REE_assembly checkout used by the result writer.
PHASE3_REE_V3 = _validate_abs_repo_path(
    os.environ.get("PHASE3_REE_V3", "/home/ree/REE_Working/ree-v3"),
    "PHASE3_REE_V3", "/home/ree/REE_Working/ree-v3")
PHASE3_REE_V3_BRANCH = _validate_branch_name(
    os.environ.get("PHASE3_REE_V3_BRANCH", "main"),
    "PHASE3_REE_V3_BRANCH", "main")
# Relative path inside the ree-v3 checkout for the canonical queue file.
PHASE3_QUEUE_RELPATH = _validate_repo_relpath(
    os.environ.get("PHASE3_QUEUE_RELPATH", "experiment_queue.json"),
    "PHASE3_QUEUE_RELPATH", "experiment_queue.json")

# REE_assembly checkout path (result writer + heartbeat writer share it).
# Same validation as PHASE3_REE_V3; deferred to here so the helper exists.
PHASE3_REE_ASSEMBLY = _validate_abs_repo_path(
    os.environ.get("PHASE3_REE_ASSEMBLY",
                   "/home/ree/REE_Working/REE_assembly"),
    "PHASE3_REE_ASSEMBLY", "/home/ree/REE_Working/REE_assembly")


# PLAN.md step 6: derived heartbeats + runner_status writeback. The writer
# materialises evidence/experiments/runner_heartbeats/<machine>.json AND
# runner_status/<machine>.json from rows in the heartbeats table, replacing
# the per-runner runner_remote_control.push_heartbeat git push (the original
# autostash-war bug source). Gated by its OWN flag so the heartbeat cutover
# can stage separately from result + queue cutovers.
PHASE3_HEARTBEAT_WRITER_READY = True

# Subdirectories (relative to REE_assembly) where the writer materialises
# the per-machine files. Match the existing legacy layout that explorer
# + scaler workflow + governance scripts already read.
PHASE3_HEARTBEATS_RELDIR = "evidence/experiments/runner_heartbeats"
PHASE3_STATUS_RELDIR = "evidence/experiments/runner_status"

# Default branch on the hub's REE_assembly checkout. Override via env if the
# hub is ever moved to a non-master deploy layout.
PHASE3_ASSEMBLY_BRANCH = _validate_branch_name(
    os.environ.get("PHASE3_ASSEMBLY_BRANCH", "master"),
    "PHASE3_ASSEMBLY_BRANCH", "master")


def _fsync_dir(path):
    """MED-B: after `os.replace(tmp, target)` the rename is not crash-
    durable until the containing directory's metadata journal entry is
    flushed. Open the dir read-only and fsync its descriptor. Linux:
    standard pattern; macOS: the open() succeeds, the fsync is a no-op
    but harmless. Windows: O_DIRECTORY is unsupported, so we swallow
    the EINVAL/ENOTDIR (the hub deploy target is Linux; this guard is
    for the smoke harness which runs on macOS / dev machines).
    Best-effort: any failure is swallowed -- the rename itself already
    succeeded, the worst case is a non-durable directory entry across a
    power loss, and the writer's spool retains the source-of-truth bytes
    until committed_at is set."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _revert_target_to_head(repo, relpath, target):
    """Best-effort rollback after a `git add` failure.

    The atomic write already replaced the working-tree file with the new
    content. If `git add` then fails (out of disk, EAGAIN, permission
    glitch on the .git/index lockfile, ...) the working tree is dirty and
    blocks the next tick's clean-tree check, stalling the writer until
    operator intervention.

    First try `git checkout HEAD -- <relpath>` to restore the pre-write
    content (the common case: we were updating an existing tracked file).
    If that fails (e.g. brand-new file with no HEAD blob to restore), fall
    back to `os.unlink(target)` to leave the working tree clean.

    Both steps are best-effort: if BOTH fail, the leak is the same as
    pre-fix behaviour. Never raises."""
    try:
        result = _git(
            repo, "checkout", "HEAD", "--", relpath,
            check=False, timeout=10)
        if result.returncode == 0:
            return
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    try:
        os.unlink(target)
    except OSError:
        pass


def _git(repo, *args, timeout=30, check=True):
    """Run git in repo. Returns CompletedProcess. capture_output=True so
    nothing leaks to stdout/stderr unless we choose to log it."""
    return subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True, text=True, timeout=timeout, check=check,
    )


def _hub_working_tree_clean(repo):
    """Phase 3 explicitly retires autostash, so the writer refuses to
    operate on a dirty tree -- any uncommitted edit on the hub checkout
    must be resolved by a human, not silently stashed. Returns (clean,
    reason). reason is a one-line string when clean=False."""
    try:
        out = _git(repo, "status", "--porcelain", check=True).stdout
    except subprocess.CalledProcessError as exc:
        return (False, "git status failed: %r" % exc)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return (False, "git status error: %r" % exc)
    if out.strip():
        first = out.strip().splitlines()[0]
        return (False, "dirty working tree: %s" % first[:120])
    return (True, "")


# Every writer-authored commit's message starts with one of these prefixes;
# see commit_msg construction inside phase3_git_writer (results),
# phase3_queue_writer (queue snapshot), and phase3_heartbeat_writer
# (telemetry). The foreign-commit check uses prefix membership to gate
# pushes -- the writer refuses to publish any commit no phase3 writer
# authored. Each writer has its own prefix for log readability + audit
# (`git log --grep=phase3-heartbeats:` answers "what did the heartbeat
# writer do today?").
_PHASE3_COMMIT_PREFIX = "phase3: "
_PHASE3_QUEUE_COMMIT_PREFIX = "phase3-queue: "
_PHASE3_HEARTBEAT_COMMIT_PREFIX = "phase3-heartbeats: "

# The foreign-commit check accepts any of these prefixes -- the result
# writer and heartbeat writer share REE_assembly, so each must tolerate
# the other's unpushed commit (which can happen after a transient push
# failure). Treating sibling-writer commits as "foreign" caused a permanent
# deadlock: each writer rejected the other's leftover commit on every
# subsequent tick. ADDING A NEW PHASE3 WRITER? Add its prefix here, or the
# deadlock comes back.
_PHASE3_WRITER_PREFIXES = (
    _PHASE3_COMMIT_PREFIX,
    _PHASE3_QUEUE_COMMIT_PREFIX,
    _PHASE3_HEARTBEAT_COMMIT_PREFIX,
)


def _check_ahead_writer_authored(repo, branch,
                                 prefixes=_PHASE3_WRITER_PREFIXES):
    """Inspect every commit reachable from HEAD but not origin/<branch>.

    Returns (ok, detail). ok=True iff every such commit's subject starts
    with one of `prefixes` (i.e. was authored by SOME phase3 writer).
    detail is a list of up to three foreign subject lines when ok=False,
    or a single-element list with the git-log failure message when the
    log itself errored.

    Caller is expected to have refreshed origin/<branch> via `git fetch`
    immediately before this call -- a stale ref would over-count ahead
    commits (false positives) for the foreign check.

    `prefixes` defaults to the full writer-authored set so that two
    writers sharing a repo (result + heartbeat both push to REE_assembly)
    do not treat each other's unpushed commits as foreign. Pass a
    narrower tuple only when the caller deliberately wants to scope
    "authored by THIS writer" -- e.g. an audit tool.
    """
    log = _git(
        repo, "log", "--format=%s",
        "origin/" + branch + "..HEAD",
        check=False, timeout=10)
    if log.returncode != 0:
        return (False, ["<git-log-failed: %s>"
                        % log.stderr.strip()[:120]])
    foreign = [
        line for line in log.stdout.splitlines()
        if line and not any(line.startswith(p) for p in prefixes)
    ]
    return (not foreign, foreign[:3])


def phase3_git_writer(
    conn,
    queue_path,
    *,
    ree_assembly_path=None,
    dry_run=False,
):
    """Sole git writer tick (Phase 3).

    Reads pending manifests from the filesystem spool, writes them under
    REE_assembly/evidence/experiments/, commits, and pushes. Marks
    `results.committed_at` only after a successful push so a crash midway
    leaves the manifest available for the next tick (idempotent retry).

    SAFETY:
      - PHASE3_GIT_WRITER_READY is checked at every entry. While False the
        writer logs intent and returns False; main()'s authoritative-mode
        loop then refuses to advance, so no git writes can happen even if
        the operator flips SYNC_MODE prematurely.
      - Refuses to operate on a dirty REE_assembly working tree (the whole
        point of Phase 3 is to retire the autostash war; a human must
        clean up unexpected dirt).
      - Never calls `git pull --rebase --autostash`. A non-fast-forward
        push fails the tick loudly and leaves the spool intact for retry.
      - Before marking results.committed_at on a "no new diff" tick, runs
        `git fetch --quiet origin <branch>` then checks
        `git rev-list --count origin/<branch>..HEAD`. The diff-cached
        short-circuit only fires when ahead==0 (bytes truly on origin);
        ahead>0 forces a push of the unpushed local commit first, or
        refuses the tick if that push is still rejected. The pre-fetch
        is load-bearing: rev-list reads the local remote-tracking ref,
        and a stale ref would let ahead==0 lie (writer would drain the
        spool against a stale view of origin while bytes never reach
        the remote). Fetch failure refuses the tick rather than
        proceeding with a possibly-stale ref. Without this guard a
        rejected-push tick followed by a no-operator-action tick would
        silently drain the spool without origin ever receiving the
        bytes.
      - Before EVERY push, verifies that every commit in
        origin/<branch>..HEAD was writer-authored (subject starts with
        the `phase3: ` prefix). If any foreign commit is found (operator
        hand-edit on the hub, accidental tooling commit, force-push
        residue), the writer refuses to push and retains the spool. The
        post-cutover invariant "all REE_assembly commits attributable to
        sync_daemon" depends on this check; without it, an unpushed
        operator commit between origin/<branch> and HEAD would be
        published silently by the writer.
      - Batched to PHASE3_BATCH_SIZE manifests per tick; the rest land in
        subsequent ticks.

    Returns True only when a full tick completed (or dry_run simulated).
    Returns False when the writer stub guard is active, the tree is dirty,
    or the spool is empty (nothing to do -- the daemon is idle).

    Out-of-scope for this sketch (deferred TODO):
      - Step 5: snapshot completed queue items from `experiments` table
        into the ree-v3 checkout's experiment_queue.json and push.
      - Step 6: write derived runner_heartbeats/*.json + runner_status/
        from the heartbeats table (replaces the per-runner git heartbeat
        push that runner_remote_control.push_heartbeat does today).
      Both extensions live in this same function once the results path is
      validated under a test fleet.
    """
    asm = ree_assembly_path or PHASE3_REE_ASSEMBLY
    if not PHASE3_GIT_WRITER_READY:
        sys.stderr.write(
            "[phase3] git writer stub (PHASE3_GIT_WRITER_READY=False); "
            "no git writes performed\n")
        return False

    # Spool is the prerequisite. Without it /result has no bytes to
    # commit; refusing is louder than producing empty ticks forever.
    if manifest_spool.spool_root() is None:
        sys.stderr.write(
            "[phase3] COORDINATOR_SPOOL_DIR unset; refusing -- /result "
            "is not persisting manifest bytes, so nothing to commit\n")
        return False

    pending_ids = list(manifest_spool.list_pending_run_ids())
    if not pending_ids:
        return True  # idle tick is a successful no-op

    batch = pending_ids[:PHASE3_BATCH_SIZE]

    if dry_run:
        sys.stdout.write(
            "[phase3] dry_run tick: %d pending, would commit %d\n" % (
                len(pending_ids), len(batch)))
        return True

    clean, reason = _hub_working_tree_clean(asm)
    if not clean:
        sys.stderr.write(
            "[phase3] refusing tick: REE_assembly at %s is %s. Phase 3 "
            "does NOT autostash -- resolve the dirt by hand, then the "
            "next tick will retry.\n" % (asm, reason))
        return False

    # Stage 1: write manifests onto the working tree and stage them.
    staged = []  # list of (run_id, relpath) successfully written
    for run_id in batch:
        raw = manifest_spool.read_manifest(run_id)
        meta = manifest_spool.read_meta(run_id) or {}
        if raw is None:
            sys.stderr.write(
                "[phase3] WARN missing manifest bytes for %s; skipping\n"
                % run_id)
            continue
        try:
            manifest_doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            manifest_doc = {}
        # Prefer the meta sidecar's hint (runner-supplied at /result time);
        # fall back to the manifest body; finally to the run_id default.
        hint = meta.get("manifest_relpath") or manifest_doc.get(
            "manifest_relpath")
        try:
            relpath = manifest_spool.derive_evidence_relpath(
                run_id, {"manifest_relpath": hint} if hint else manifest_doc)
        except ValueError as exc:
            sys.stderr.write(
                "[phase3] WARN derive_evidence_relpath rejected %s: %s\n"
                % (run_id, exc))
            continue
        target = os.path.join(asm, relpath)
        target_dir = os.path.dirname(target)
        # Atomic write: tmp file + os.replace. A crash mid-`fh.write`
        # otherwise leaves a truncated file that the immediately-following
        # `git add` would happily stage. Tmp+rename mirrors the spool
        # writer's atomic semantics.
        tmp_target = target + ".phase3.tmp"
        target_replaced = False
        try:
            os.makedirs(target_dir, exist_ok=True)
            with open(tmp_target, "wb") as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_target, target)
            target_replaced = True
            _fsync_dir(target_dir)
        except OSError as exc:
            sys.stderr.write(
                "[phase3] WARN atomic write failed for %s -> %s: %r\n" % (
                    run_id, relpath, exc))
            # Best-effort cleanup of the tmp file. Target was NOT replaced
            # (os.replace ran last in the try block); nothing to revert.
            try:
                os.unlink(tmp_target)
            except OSError:
                pass
            continue
        try:
            _git(asm, "add", relpath, timeout=15, check=True)
            staged.append((run_id, relpath))
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            sys.stderr.write(
                "[phase3] WARN git add failed for %s -> %s: %r. "
                "Reverting working-tree write to keep the next tick's "
                "clean-tree check passing.\n" % (run_id, relpath, exc))
            if target_replaced:
                _revert_target_to_head(asm, relpath, target)

    if not staged:
        sys.stderr.write(
            "[phase3] no manifests staged this tick; nothing to commit\n")
        return False

    # Stage 2: single commit + single push for the whole batch.
    today = db.utcnow()[:10]
    # MED-A: build the subject from the same constant the foreign-commit
    # check reads. Drifting one without the other (e.g. dropping the
    # trailing space, or rewording the prefix locally) would make the
    # writer reject its own commits as foreign.
    commit_msg = "%s%d v3 result manifest(s) %s" % (
        _PHASE3_COMMIT_PREFIX, len(staged), today)
    try:
        # Refresh origin/<branch> once at the top of the push-decision
        # block. Both the ahead-of-origin guard (HIGH-1) and the
        # writer-authored-only push guard (HIGH-2) need an accurate
        # remote-tracking ref:
        #   - HIGH-1: a stale ref would let `ahead==0` lie in case (a),
        #     draining the spool against a stale view of origin.
        #   - HIGH-2: a stale ref over-counts ahead commits, false-
        #     positiving the foreign-commit check (refusing legitimate
        #     work). Fetching first keeps the check on the correct
        #     reference set.
        # Fetch failure refuses the tick rather than proceeding with a
        # possibly-stale ref.
        fetched = _git(
            asm, "fetch", "--quiet", "origin", PHASE3_ASSEMBLY_BRANCH,
            check=False, timeout=30)
        if fetched.returncode != 0:
            sys.stderr.write(
                "[phase3] refusing tick: fetch origin %s failed (%s). "
                "Spool retained for next tick.\n" % (
                    PHASE3_ASSEMBLY_BRANCH,
                    fetched.stderr.strip()[:240]))
            return False

        diff = _git(asm, "diff", "--cached", "--quiet", check=False,
                    timeout=10)
        if diff.returncode == 0:
            # `git add` produced no diff. Two cases are indistinguishable
            # from `git diff --cached` alone:
            #   (a) bytes already on origin (true idempotent re-spool of
            #       a previously-committed-and-pushed run), OR
            #   (b) bytes live in an UNPUSHED local commit -- the tick
            #       after a rejected push (Phase 3 explicitly retires
            #       autostash, so a rejected push leaves the local
            #       commit in HEAD with no operator intervention).
            # Marking committed_at in case (b) without a push is unsafe:
            # the DB says "done" but origin never received the bytes.
            # `git rev-list --count origin/<branch>..HEAD` distinguishes
            # the two (fresh ref guaranteed by the fetch above).
            ahead = _git(
                asm, "rev-list", "--count",
                "origin/" + PHASE3_ASSEMBLY_BRANCH + "..HEAD",
                check=False, timeout=10)
            if ahead.returncode != 0:
                sys.stderr.write(
                    "[phase3] refusing to mark committed: rev-list "
                    "ahead-count failed (%s). Spool retained.\n" % (
                        ahead.stderr.strip()[:240]))
                return False
            ahead_count = ahead.stdout.strip()
            if ahead_count and ahead_count != "0":
                # Case (b): push the existing unpushed commit -- BUT
                # only if every ahead commit is writer-authored. A
                # foreign commit (operator hand-edit, accidental tooling
                # commit) must not be published under sync_daemon's
                # authority; refuse and let the operator investigate.
                ok, foreign = _check_ahead_writer_authored(
                    asm, PHASE3_ASSEMBLY_BRANCH)
                if not ok:
                    sys.stderr.write(
                        "[phase3] refusing tick: %d foreign commit(s) "
                        "in origin/%s..HEAD that the writer did not "
                        "author: %s. NOT marking committed_at; spool "
                        "retained. Operator must investigate (do not "
                        "let the writer publish unrelated commits under "
                        "sync_daemon's authority).\n" % (
                            len(foreign), PHASE3_ASSEMBLY_BRANCH,
                            foreign))
                    return False
                push = _git(
                    asm, "push", "origin",
                    "HEAD:" + PHASE3_ASSEMBLY_BRANCH,
                    timeout=60, check=False)
                if push.returncode != 0:
                    sys.stderr.write(
                        "[phase3] push REJECTED for unpushed local "
                        "commit: %s. NOT marking committed_at; spool "
                        "retained. Operator must investigate (non-"
                        "fast-forward = hub is behind origin; resolve "
                        "by hand).\n" % (push.stderr.strip()[:240]))
                    return False
                sys.stdout.write(
                    "[phase3] pushed unpushed local commit (HEAD was %s "
                    "ahead of origin/%s); %d row(s) committed\n" % (
                        ahead_count, PHASE3_ASSEMBLY_BRANCH, len(staged)))
            else:
                # Case (a): true idempotent re-spool. ahead == 0 means
                # the bytes really are on origin already.
                sys.stdout.write(
                    "[phase3] batch already on tree and on origin "
                    "(ahead==0); marking %d row(s) committed without "
                    "a push\n" % len(staged))
        else:
            _git(asm, "commit", "-m", commit_msg, timeout=20, check=True)
            # After the writer's own commit lands, origin/<branch>..HEAD
            # includes the writer's commit (matches phase3: prefix) plus
            # any operator commits that were already ahead of origin.
            # Refuse the push if any foreign commit would be carried
            # along. The writer's commit remains in local HEAD; the next
            # tick will re-enter via case (b) and refuse again until the
            # operator resolves the foreign commit.
            ok, foreign = _check_ahead_writer_authored(
                asm, PHASE3_ASSEMBLY_BRANCH)
            if not ok:
                sys.stderr.write(
                    "[phase3] refusing tick: writer's commit landed but "
                    "%d foreign commit(s) are ahead of origin/%s and "
                    "would be carried by the push: %s. NOT marking "
                    "committed_at; spool retained. Operator must "
                    "resolve the foreign commit(s) before next tick.\n"
                    % (len(foreign), PHASE3_ASSEMBLY_BRANCH, foreign))
                return False
            push = _git(
                asm, "push", "origin", "HEAD:" + PHASE3_ASSEMBLY_BRANCH,
                timeout=60, check=False)
            if push.returncode != 0:
                sys.stderr.write(
                    "[phase3] push REJECTED: %s. NOT marking committed_at; "
                    "spool retained for retry on the next tick. Operator "
                    "must investigate (non-fast-forward = hub is behind "
                    "origin; resolve by hand).\n" % (push.stderr.strip()[:240]))
                return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write("[phase3] git commit/push error: %r\n" % exc)
        return False

    # Stage 3: mark DB committed + delete spool entries. Order matters:
    # update DB first (cheap, atomic) before deleting bytes from disk so a
    # crash between the two leaves the spool entries that the next tick
    # will re-process as if they were uncommitted, which `git add` will
    # detect as no-diff and short-circuit (the idempotent-already-on-tree
    # branch above).
    #
    # MED-1 from the 2026-05-27 review: `UPDATE ... WHERE committed_at
    # IS NULL` returns rowcount 0 both when the row exists but is
    # already marked AND when the row is missing entirely. The second
    # case is an invariant violation -- bytes reached origin via the
    # writer's push but the DB has no record of the run. Surface it
    # loudly with a per-run WARN, but still proceed to drop the spool:
    # bytes ARE on origin, retaining the spool would replay forever
    # against the same missing-row condition.
    now = db.utcnow()
    pre_existing = {
        row["run_id"]
        for row in conn.execute(
            "SELECT run_id FROM results WHERE run_id IN (%s)" % (
                ",".join("?" * len(staged))),
            [run_id for run_id, _ in staged]).fetchall()
    }
    missing = [run_id for run_id, _ in staged
               if run_id not in pre_existing]
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            "UPDATE results SET committed_at=? WHERE run_id=? "
            "AND committed_at IS NULL",
            [(now, run_id) for run_id, _ in staged],
        )
        conn.execute("COMMIT")
    except Exception as exc:  # noqa: BLE001 -- daemon must not die
        sys.stderr.write(
            "[phase3] WARN committed_at update failed: %r. Spool retained; "
            "next tick will replay idempotently.\n" % exc)
        try:
            conn.execute("ROLLBACK")
        except Exception:  # noqa: BLE001
            pass
        return False
    if missing:
        sys.stderr.write(
            "[phase3] WARN invariant violation: %d manifest(s) reached "
            "origin via this commit but have no `results` row in the "
            "coordinator DB: %s. Spool will be drained (bytes ARE on "
            "origin); the DB/origin mismatch needs operator audit. "
            "Likely causes: POST /result wrote spool bytes before "
            "`db.record_result` recorded the row, or the row was "
            "deleted out-of-band.\n" % (
                len(missing), missing[:5]))

    for run_id, _ in staged:
        manifest_spool.delete_manifest(run_id)

    sys.stdout.write(
        "[phase3] committed %d manifest(s) (%d remaining in spool)\n" % (
            len(staged), max(0, len(pending_ids) - len(staged))))
    return True


# Canonical key order for items emitted in experiment_queue.json. Stored
# `item_json` blobs are written with sort_keys=True (db.upsert_experiment),
# so the loaded dict has alphabetical keys; writing the materialised file
# without re-ordering would land an "every item reordered" diff on the
# first Phase 3 cutover commit and lose operator-meaningful field grouping
# (identifier -> executable -> scheduling -> claim -> state). This list is
# the operator-visible shape contract; do NOT reorder casually -- any
# change here produces another full-rewrite commit on every queue file
# already on origin. Unknown keys (forward-compat custom fields) land
# alphabetically AFTER this canonical block.
_QUEUE_ITEM_KEY_ORDER = (
    # 1. Identity
    "queue_id",
    "title",
    "description",
    # 2. Executable
    "script",
    # 3. Scheduling
    "priority",
    "machine_affinity",
    "estimated_minutes",
    # 4. Provenance / lineage
    "supersedes",
    "claim_id",
    "backlog_id",
    # 5. Operator flags
    "force_rerun",
    "note",
    # 6. Coordinator-managed state (overlaid below from live DB columns)
    "status",
    "claimed_by",
)


def _canonicalise_queue_item(item):
    """Return `item` with keys in _QUEUE_ITEM_KEY_ORDER first, then any
    unknown keys alphabetically. Keys absent from `item` are skipped (not
    inserted as None) so the materialised file is shape-faithful to what
    the operator originally wrote."""
    canonical = {}
    for k in _QUEUE_ITEM_KEY_ORDER:
        if k in item:
            canonical[k] = item[k]
    for k in sorted(item.keys()):
        if k not in canonical:
            canonical[k] = item[k]
    return canonical


def _materialise_queue_from_db(conn, current_calibration):
    """Build the canonical experiment_queue.json content from DB rows.

    Returns the queue dict (schema_version + calibration + items). Only
    items with status NOT IN ('completed', 'failed') are emitted -- those
    are terminal states and don't belong in the worklist.

    Each item is reconstructed from the stored `item_json` blob with the
    live DB columns (status, claimed_by_machine, claimed_at) overlaid:
    item_json preserves the operator-supplied fields (script, priority,
    machine_affinity, etc.) verbatim, while the DB columns hold the
    coordinator-managed state. Keys are then reordered into the
    _QUEUE_ITEM_KEY_ORDER canonical shape so per-tick claim/release
    transitions don't produce noisy "claimed_by appeared at end of dict"
    diffs.

    `current_calibration` is preserved verbatim from the existing file --
    the DB doesn't store calibration data so we must round-trip it.
    """
    rows = conn.execute(
        "SELECT queue_id, status, claimed_by_machine, claimed_at, "
        "item_json, priority FROM experiments "
        "WHERE status NOT IN ('completed', 'failed') "
        "ORDER BY priority DESC, queue_id"
    ).fetchall()
    items = []
    for r in rows:
        try:
            item = json.loads(r["item_json"])
        except (ValueError, TypeError):
            # Corrupt blob -- skip, log later. Should not happen in practice.
            sys.stderr.write(
                "[phase3-queue] WARN skipping %s: item_json unparseable\n"
                % r["queue_id"])
            continue
        if not isinstance(item, dict):
            sys.stderr.write(
                "[phase3-queue] WARN skipping %s: item_json not a dict\n"
                % r["queue_id"])
            continue
        # Overlay coordinator-managed state. claim_authority='coordinator'
        # in upsert_experiment preserves claim fields IN THE DB across
        # operator file edits; here we surface that DB state into the
        # written-back file so operators see the canonical view.
        item["status"] = r["status"]
        if r["claimed_by_machine"]:
            item["claimed_by"] = {
                "machine": r["claimed_by_machine"],
                "claimed_at": r["claimed_at"],
            }
        else:
            item.pop("claimed_by", None)
        items.append(_canonicalise_queue_item(item))
    return {
        "schema_version": "v1",
        "calibration": current_calibration or {},
        "items": items,
    }


def phase3_queue_writer(
    conn,
    *,
    ree_v3_path=None,
    queue_relpath=None,
    branch=None,
):
    """PLAN.md step 5: snapshot the canonical queue from the DB into
    `experiment_queue.json` on the hub's ree-v3 checkout, commit, push.

    Pairs with `phase3_git_writer` -- same safety contract:
      - PHASE3_QUEUE_WRITER_READY gates execution (False -> log stub
        message and return False).
      - Refuses on dirty working tree on the ree-v3 checkout.
      - Refuses on fetch failure (a stale `origin/main` ref would fool
        the ahead-of-origin guard).
      - Foreign-commit check before push: only commits whose subject
        starts with `_PHASE3_QUEUE_COMMIT_PREFIX` may be carried by the
        writer's push.
      - Atomic write (tmp + os.replace) before `git add`.
      - Returns True on success (or idle no-op when the materialised
        view matches the current file).
      - Returns False on any refusal; never raises.
    """
    repo = ree_v3_path or PHASE3_REE_V3
    rel = queue_relpath or PHASE3_QUEUE_RELPATH
    br = branch or PHASE3_REE_V3_BRANCH

    if not PHASE3_QUEUE_WRITER_READY:
        sys.stderr.write(
            "[phase3-queue] queue writer stub "
            "(PHASE3_QUEUE_WRITER_READY=False); no git writes performed\n")
        return False

    target = os.path.join(repo, rel)

    # Read current file to preserve calibration block and to compare for
    # idempotent no-op when DB-materialised view matches.
    current_text = None
    current_calibration = {}
    if os.path.exists(target):
        try:
            with open(target, "r", encoding="utf-8") as fh:
                current_text = fh.read()
            try:
                current_doc = json.loads(current_text)
                if isinstance(current_doc, dict):
                    current_calibration = current_doc.get(
                        "calibration") or {}
            except ValueError:
                # Existing file is unparseable -- the writer should still
                # overwrite it with a fresh DB-materialised view, but we
                # have no calibration to preserve.
                sys.stderr.write(
                    "[phase3-queue] WARN current %s is not valid JSON; "
                    "calibration block will be empty in the rewrite\n"
                    % rel)
        except OSError as exc:
            sys.stderr.write(
                "[phase3-queue] WARN could not read current %s: %r\n"
                % (rel, exc))

    # Materialise the DB view.
    try:
        new_doc = _materialise_queue_from_db(conn, current_calibration)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            "[phase3-queue] DB query failed: %r. Skipping tick.\n" % exc)
        return False

    new_text = json.dumps(new_doc, indent=2, sort_keys=False) + "\n"

    # Idempotent no-op when content matches.
    if current_text is not None and current_text == new_text:
        return True

    clean, reason = _hub_working_tree_clean(repo)
    if not clean:
        sys.stderr.write(
            "[phase3-queue] refusing tick: %s at %s is %s. Phase 3 "
            "does NOT autostash -- resolve the dirt by hand, then the "
            "next tick will retry.\n" % (rel, repo, reason))
        return False

    # Atomic write to the working tree.
    tmp_target = target + ".phase3.tmp"
    target_dir = os.path.dirname(target)
    try:
        os.makedirs(target_dir, exist_ok=True)
        with open(tmp_target, "w", encoding="utf-8") as fh:
            fh.write(new_text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_target, target)
        _fsync_dir(target_dir)
    except OSError as exc:
        sys.stderr.write(
            "[phase3-queue] WARN write failed: %r. Skipping tick.\n" % exc)
        try:
            os.unlink(tmp_target)
        except OSError:
            pass
        return False

    today = db.utcnow()[:10]
    # MED-A: same constant-sharing rationale as the result writer above.
    commit_msg = "%ssnapshot %s" % (_PHASE3_QUEUE_COMMIT_PREFIX, today)

    try:
        # Same fetch + foreign-check + commit/push sequence as the
        # result writer, applied to the ree-v3 repo.
        fetched = _git(
            repo, "fetch", "--quiet", "origin", br,
            check=False, timeout=30)
        if fetched.returncode != 0:
            sys.stderr.write(
                "[phase3-queue] refusing tick: fetch origin %s failed "
                "(%s). Working tree edit retained; next tick will retry.\n"
                % (br, fetched.stderr.strip()[:240]))
            return False

        # Stage the file. If `git add` produces no diff (the working-tree
        # write was byte-identical to HEAD's blob), fall through to the
        # ahead-of-origin check just like the result writer does.
        _git(repo, "add", rel, timeout=15, check=True)

        diff = _git(repo, "diff", "--cached", "--quiet",
                    check=False, timeout=10)
        if diff.returncode == 0:
            # No-diff path. ahead-of-origin guard:
            ahead = _git(
                repo, "rev-list", "--count",
                "origin/" + br + "..HEAD",
                check=False, timeout=10)
            if ahead.returncode != 0:
                sys.stderr.write(
                    "[phase3-queue] refusing tick: rev-list ahead-count "
                    "failed (%s).\n" % ahead.stderr.strip()[:240])
                return False
            ahead_count = ahead.stdout.strip()
            if ahead_count and ahead_count != "0":
                ok, foreign = _check_ahead_writer_authored(repo, br)
                if not ok:
                    sys.stderr.write(
                        "[phase3-queue] refusing tick: %d foreign "
                        "commit(s) in origin/%s..HEAD that no phase3 "
                        "writer authored: %s. Operator must "
                        "investigate.\n" % (
                            len(foreign), br, foreign))
                    return False
                push = _git(
                    repo, "push", "origin", "HEAD:" + br,
                    timeout=60, check=False)
                if push.returncode != 0:
                    sys.stderr.write(
                        "[phase3-queue] push REJECTED for unpushed local "
                        "commit: %s. Working tree edit retained.\n"
                        % push.stderr.strip()[:240])
                    return False
                sys.stdout.write(
                    "[phase3-queue] pushed unpushed local commit "
                    "(HEAD was %s ahead of origin/%s)\n" % (
                        ahead_count, br))
                return True
            # ahead == 0 -- view is already on origin. Treat as no-op.
            return True

        # diff.returncode != 0: there's a real change to commit.
        _git(repo, "commit", "-m", commit_msg, timeout=20, check=True)
        ok, foreign = _check_ahead_writer_authored(repo, br)
        if not ok:
            sys.stderr.write(
                "[phase3-queue] refusing tick: writer's commit landed "
                "but %d foreign commit(s) are ahead of origin/%s and "
                "would be carried by the push: %s. Operator must "
                "resolve the foreign commit(s) before next tick.\n" % (
                    len(foreign), br, foreign))
            return False
        push = _git(
            repo, "push", "origin", "HEAD:" + br,
            timeout=60, check=False)
        if push.returncode != 0:
            sys.stderr.write(
                "[phase3-queue] push REJECTED: %s. Working tree commit "
                "retained for retry.\n" % push.stderr.strip()[:240])
            return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(
            "[phase3-queue] git commit/push error: %r\n" % exc)
        return False

    sys.stdout.write(
        "[phase3-queue] snapshot pushed (%d active item(s))\n"
        % len(new_doc["items"]))
    return True


def _atomic_write_text(target_path, text):
    """Atomic working-tree write (tmp + fsync + os.replace). Returns
    True on success, False on OSError. Matches the LOW-2 pattern in
    phase3_git_writer."""
    tmp = target_path + ".phase3.tmp"
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target_path)
        return True
    except OSError as exc:
        sys.stderr.write(
            "[phase3-heartbeats] WARN write failed for %s: %r\n" % (
                target_path, exc))
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return False


def _safe_machine_filename(machine):
    """Defensive normalisation for the per-machine output filename.

    Accept alphanumeric + dot + dash + underscore (matches the existing
    experiment_runner convention for status filenames). Reject anything
    that would let a hostile heartbeat write outside the target
    subdirectory:
      - empty string
      - the bare current/parent indicators "." or ".."
      - ANY string containing the path-traversal sequence ".." (catches
        cases like "../escape" where character-stripping otherwise
        leaves "..escape" still containing a traversal token)
      - leading dot (hidden-file convention; not a real machine name and
        confuses ls/glob)
    """
    if not machine or not isinstance(machine, str):
        return None
    cleaned = "".join(
        c for c in machine if c.isalnum() or c in ("-", "_", "."))
    if not cleaned:
        return None
    if cleaned in (".", ".."):
        return None
    if ".." in cleaned:
        return None
    if cleaned.startswith("."):
        return None
    return cleaned


def phase3_heartbeat_writer(
    conn,
    *,
    ree_assembly_path=None,
    branch=None,
):
    """PLAN.md step 6: materialise runner_heartbeats/<machine>.json and
    runner_status/<machine>.json from the coordinator's heartbeats
    table into the hub's REE_assembly checkout, commit + push.

    Pairs with phase3_git_writer + phase3_queue_writer to give
    sync_daemon sole-writer authority across both coordination repos.

    Same safety contract as the queue writer:
      - PHASE3_HEARTBEAT_WRITER_READY gates execution (False -> stub).
      - Dirty-tree refusal.
      - Fetch-before-push + ahead-of-origin guard + foreign-commit
        check (`phase3-heartbeats: ` prefix).
      - Atomic per-file write (tmp + fsync + os.replace).
      - Idempotent no-op when every file matches what's already in HEAD.
      - Never raises.

    Per-machine output is byte-for-byte the JSON payload the runner
    POSTed: the coordinator stored heartbeat_payload_json and
    status_payload_json verbatim, so consumers (explorer, scaler
    workflow, governance) see exactly the shape they did under the
    runner's old git push -- only the transport changed.

    Rows whose payload columns are NULL (legacy clients that haven't
    been updated to send the rich payload yet) are SKIPPED for the
    file write. Their structured columns still drive lifecycle_state
    via /shadow/status -- they just don't generate a per-machine file
    until the runner is updated.
    """
    asm = ree_assembly_path or PHASE3_REE_ASSEMBLY
    br = branch or PHASE3_ASSEMBLY_BRANCH

    if not PHASE3_HEARTBEAT_WRITER_READY:
        sys.stderr.write(
            "[phase3-heartbeats] writer stub "
            "(PHASE3_HEARTBEAT_WRITER_READY=False); no git writes performed\n")
        return False

    # Collect what the DB has. Each row may have either or both payload
    # columns populated; if NEITHER, skip (nothing to materialise).
    rows = conn.execute(
        "SELECT machine, heartbeat_payload_json, status_payload_json "
        "FROM heartbeats "
        "WHERE heartbeat_payload_json IS NOT NULL "
        "   OR status_payload_json IS NOT NULL"
    ).fetchall()
    if not rows:
        # No machine has sent a payload yet (e.g. mid-transition); nothing
        # to write. Return True (idle no-op) so the main loop doesn't
        # treat it as a failure.
        return True

    # Stage each per-machine file. Skip + warn on unsafe machine names
    # (defence-in-depth; coordinator auth already rejects bad tokens).
    pending_writes = []   # list of (relpath, text)
    for r in rows:
        safe = _safe_machine_filename(r["machine"])
        if not safe:
            sys.stderr.write(
                "[phase3-heartbeats] WARN skipping unsafe machine name %r\n"
                % (r["machine"],))
            continue
        for kind, reldir, column in (
            ("heartbeat", PHASE3_HEARTBEATS_RELDIR,
             "heartbeat_payload_json"),
            ("status", PHASE3_STATUS_RELDIR, "status_payload_json"),
        ):
            raw = r[column]
            if not raw:
                continue
            # Pretty-print so the file diff is meaningful + matches the
            # runner's existing on-disk format (json.dumps(payload, indent=2)
            # + "\n").
            try:
                doc = json.loads(raw)
            except (ValueError, TypeError):
                sys.stderr.write(
                    "[phase3-heartbeats] WARN skipping %s/%s: stored "
                    "%s is not valid JSON\n" % (
                        reldir, safe, column))
                continue
            text = json.dumps(doc, indent=2) + "\n"
            rel = "%s/%s.json" % (reldir, safe)
            pending_writes.append((rel, text))

    if not pending_writes:
        return True

    # Compare each against working tree -- skip writes when content
    # matches (idempotent no-op preserves the per-tick rhythm without
    # creating empty commits).
    actually_changed = []
    for rel, text in pending_writes:
        target = os.path.join(asm, rel)
        if os.path.exists(target):
            try:
                with open(target, "r", encoding="utf-8") as fh:
                    if fh.read() == text:
                        continue
            except OSError:
                pass
        actually_changed.append((rel, text))

    if not actually_changed:
        return True

    clean, reason = _hub_working_tree_clean(asm)
    if not clean:
        sys.stderr.write(
            "[phase3-heartbeats] refusing tick: REE_assembly at %s is "
            "%s. Phase 3 does NOT autostash -- resolve the dirt by "
            "hand.\n" % (asm, reason))
        return False

    # Atomic writes + git add. Track what successfully staged so a
    # partial failure doesn't poison the commit. On git-add failure the
    # working-tree write has already landed, so revert to HEAD content to
    # keep the next tick's clean-tree check passing.
    staged = []
    for rel, text in actually_changed:
        target = os.path.join(asm, rel)
        if not _atomic_write_text(target, text):
            continue
        try:
            _git(asm, "add", rel, timeout=15, check=True)
            staged.append(rel)
        except (subprocess.CalledProcessError,
                subprocess.TimeoutExpired) as exc:
            sys.stderr.write(
                "[phase3-heartbeats] WARN git add failed for %s: %r. "
                "Reverting working-tree write to keep the next tick's "
                "clean-tree check passing.\n" % (rel, exc))
            _revert_target_to_head(asm, rel, target)

    if not staged:
        sys.stderr.write(
            "[phase3-heartbeats] no files staged this tick; skipping\n")
        return False

    today = db.utcnow()[:10]
    commit_msg = "%s%d telemetry file(s) %s" % (
        _PHASE3_HEARTBEAT_COMMIT_PREFIX, len(staged), today)

    try:
        fetched = _git(
            asm, "fetch", "--quiet", "origin", br,
            check=False, timeout=30)
        if fetched.returncode != 0:
            sys.stderr.write(
                "[phase3-heartbeats] refusing tick: fetch origin %s "
                "failed (%s). Working tree retained for next tick.\n" % (
                    br, fetched.stderr.strip()[:240]))
            return False

        diff = _git(asm, "diff", "--cached", "--quiet",
                    check=False, timeout=10)
        if diff.returncode == 0:
            # No-diff path (same as result/queue writers): check ahead-of-
            # origin to distinguish "bytes already on origin" from
            # "unpushed local commit".
            ahead = _git(
                asm, "rev-list", "--count",
                "origin/" + br + "..HEAD",
                check=False, timeout=10)
            if ahead.returncode != 0:
                sys.stderr.write(
                    "[phase3-heartbeats] refusing tick: rev-list "
                    "failed (%s).\n" % ahead.stderr.strip()[:240])
                return False
            ahead_count = ahead.stdout.strip()
            if ahead_count and ahead_count != "0":
                ok, foreign = _check_ahead_writer_authored(asm, br)
                if not ok:
                    sys.stderr.write(
                        "[phase3-heartbeats] refusing tick: %d foreign "
                        "commit(s) in origin/%s..HEAD that no phase3 "
                        "writer authored: %s.\n" % (
                            len(foreign), br, foreign))
                    return False
                push = _git(
                    asm, "push", "origin", "HEAD:" + br,
                    timeout=60, check=False)
                if push.returncode != 0:
                    sys.stderr.write(
                        "[phase3-heartbeats] push REJECTED for unpushed "
                        "local commit: %s.\n" %
                        push.stderr.strip()[:240])
                    return False
                sys.stdout.write(
                    "[phase3-heartbeats] pushed unpushed local commit "
                    "(HEAD was %s ahead of origin/%s)\n" % (ahead_count, br))
                return True
            # ahead == 0: idempotent re-spool, already on origin.
            return True

        _git(asm, "commit", "-m", commit_msg, timeout=20, check=True)
        ok, foreign = _check_ahead_writer_authored(asm, br)
        if not ok:
            sys.stderr.write(
                "[phase3-heartbeats] refusing tick: writer's commit landed "
                "but %d foreign commit(s) are ahead of origin/%s and no "
                "phase3 writer authored them: %s.\n" % (
                    len(foreign), br, foreign))
            return False
        push = _git(
            asm, "push", "origin", "HEAD:" + br,
            timeout=60, check=False)
        if push.returncode != 0:
            sys.stderr.write(
                "[phase3-heartbeats] push REJECTED: %s. Working tree "
                "commit retained for retry.\n" %
                push.stderr.strip()[:240])
            return False
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        sys.stderr.write(
            "[phase3-heartbeats] git commit/push error: %r\n" % exc)
        return False

    sys.stdout.write(
        "[phase3-heartbeats] pushed %d telemetry file(s)\n" % len(staged))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.environ.get(
        "COORDINATOR_QUEUE_FILE", DEFAULT_QUEUE))
    ap.add_argument("--db", default=os.environ.get(
        "COORDINATOR_DB", os.path.join(
            os.path.dirname(__file__), "coordinator.db")))
    ap.add_argument("--interval", type=float,
                    default=_validate_float(
                        os.environ.get("SYNC_INTERVAL", "60"),
                        "SYNC_INTERVAL", 60.0))
    ap.add_argument("--once", action="store_true",
                    help="reconcile once and exit (used by tests)")
    ap.add_argument("--i-understand-phase3", action="store_true")
    args = ap.parse_args()

    sync_mode = os.environ.get("SYNC_MODE", "shadow")
    if sync_mode == "authoritative":
        if not args.i_understand_phase3:
            sys.stderr.write(
                "refusing: SYNC_MODE=authoritative needs "
                "--i-understand-phase3 (Phase 3 not built)\n")
            return 2
        db.init_db(args.db)
        while True:
            conn = db.connect(args.db)
            try:
                # Authoritative tick has three jobs:
                # (1) pick up operator-added queue items from the git file
                #     into the DB (upsert_only=True; the DB never deletes
                #     terminal-state rows just because the file lost them,
                #     because the file is now a DERIVED view).
                # (2) write committed result manifests through to
                #     REE_assembly (PLAN.md step 4 = phase3_git_writer).
                # (3) materialise the canonical queue file back from the
                #     DB to ree-v3 (PLAN.md step 5 = phase3_queue_writer).
                try:
                    reconcile_once(conn, args.queue,
                                   claim_authority="coordinator",
                                   upsert_only=True)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] authoritative reconcile error "
                        "(non-fatal): %r\n" % exc)
                # phase3_git_writer claims to never raise, but defensive
                # wrap matches the queue + heartbeat writer pattern. On
                # an unexpected raise default ok=False so the `if not ok`
                # below trips and the daemon exits with code 2 (the
                # result writer is the cutover gate -- if it goes off
                # the rails we want loud failure, not silent skip).
                try:
                    ok = phase3_git_writer(conn, args.queue)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] phase3_git_writer raised (unexpected): %r. "
                        "Treating as not-ready; daemon will exit.\n" % exc)
                    ok = False
                # phase3_queue_writer is independent of the result writer;
                # gated by its own PHASE3_QUEUE_WRITER_READY flag. When
                # the flag is False the function logs the stub message
                # and returns False -- treat as a non-fatal no-op in the
                # main loop (the result writer's readiness is the cutover
                # gate; the queue writer is a follow-on step that can be
                # enabled later without disrupting the result path).
                try:
                    phase3_queue_writer(conn)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] phase3_queue_writer error "
                        "(non-fatal): %r\n" % exc)
                # PLAN.md step 6: materialise runner_heartbeats/*.json
                # and runner_status/*.json from heartbeat_payload_json /
                # status_payload_json columns. Gated by its own
                # PHASE3_HEARTBEAT_WRITER_READY flag; non-fatal.
                try:
                    phase3_heartbeat_writer(conn)
                except Exception as exc:  # noqa: BLE001
                    sys.stderr.write(
                        "[sync] phase3_heartbeat_writer error "
                        "(non-fatal): %r\n" % exc)
            finally:
                conn.close()
            if not ok:
                sys.stderr.write(
                    "refusing: phase3 git writer not ready (see "
                    "PHASE3_GIT_WRITER_READY and phase3_preflight.py)\n")
                return 2
            if args.once:
                return 0
            time.sleep(args.interval)
    if sync_mode not in ("shadow", "coordinator"):
        sys.stderr.write(
            "refusing: SYNC_MODE must be shadow, coordinator, or "
            "authoritative (got %r)\n" % sync_mode)
        return 2
    claim_authority = "coordinator" if sync_mode == "coordinator" else "git"

    db.init_db(args.db)
    while True:
        conn = db.connect(args.db)
        try:
            n, div = reconcile_once(conn, args.queue,
                                    claim_authority=claim_authority)
            sys.stdout.write(
                "[sync] reconciled %d items, %d state-divergence(s)\n" % (
                    n, div))
            sys.stdout.flush()
        except Exception as exc:  # noqa: BLE001 -- daemon must not die
            sys.stderr.write("[sync] reconcile error: %r\n" % exc)
        finally:
            conn.close()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
