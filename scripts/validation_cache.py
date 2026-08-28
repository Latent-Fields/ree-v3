#!/usr/bin/env python3
"""validation_cache.py -- tree-content-hash-keyed validation-result cache.

Design record: REE_assembly/docs/architecture/landing_integration_worker_investigation.md
section 6 ("What is the minimum implementation?") and section 7 ("Safety invariants
that must not regress"). This module implements exactly that spec -- read the design
doc before changing the shape of the cache key or the storage format.

WHAT THIS IS FOR
----------------
precommit_contracts.sh's Block 2 pays the full ~13-minute `tests/contracts` suite
every time staged changes touch ree_core/ or experiments/_lib/, even when an
equivalent tree was already validated minutes ago by a concurrent or very recent
session. This cache lets Block 2 ask "has this content already been validated on
this machine class, recently?" before paying that cost again.

CACHE KEY -- three components, all required to match:
  1. relevant_hash  : content hash of the CURRENT on-disk state of ree_core/ and
                       experiments/_lib/ (the exact trees Block 2's own trigger
                       reads -- see compute_relevant_hash). NOT a commit/tree SHA:
                       an irrelevant commit elsewhere (docs, evidence, queue
                       entries) must never invalidate a hit, and the working tree
                       -- not HEAD -- is what actually gets executed (both the
                       local pytest run and remote_pytest.sh's rsync ship
                       uncommitted edits).
  2. machine_class  : platform + arch + python + TORCH VERSION, via
                       experiments/_lib/arm_fingerprint.machine_class() --
                       imported, not recomputed, so this cache cannot silently
                       reintroduce the omits-torch-version gap that function was
                       already hard-cut to close (2026-07-19, ree-v3 f1d517c).
                       A hit recorded on darwin-arm64/torch2.10.0 must never
                       certify linux-x86_64/torch2.11.0+cpu -- this is exactly
                       the axis with a CONFIRMED cross-machine-class divergence
                       (torch.multinomial; see
                       [[reference-cross-machine-class-contract-divergence]]).
  3. tier           : the validation tier name (currently one value,
                       "ree-v3-contracts-full-suite" -- Block 2's full suite only;
                       Block 1/1b/1c are not covered by this cache).

TTL. A hit older than --ttl-minutes is treated as a miss. This bounds exposure to
any residual test nondeterminism beyond the already-fixed multinomial case, and is
a conservative default (30-60 min) that can be loosened later with evidence -- see
the design doc section 8 for how to judge that.

STORAGE. A single small JSON record file (default ree-v3/.contract_validation_cache.json),
committed via scripts/ree_commit.py with an explicit path list -- the same pattern
as every other shared JSON file in this codebase (CLAUDE.md "High-Contention
Files"). No new ledger abstraction: this reuses task_claim.py's already-hardened
atomic-write / CAS-retry / commit-landed-locally primitives (imported, not
restated -- a private copy would drift, exactly the vendored-copy hazard CLAUDE.md
warns about) rather than reinventing them.

...BUT THAT COMMIT MUST NOT MOVE THE LOCAL BRANCH REF, unlike every other caller
of that pattern. This module is unique among them in running INSIDE a pre-commit
hook -- i.e. inside another ree_commit.py invocation, between that invocation's
`old_head` read and its compare-and-swap -- so an ordinary local commit here
invalidates the outer CAS and the gate rejects the very commit it was gating.
`record` therefore defaults to ree_commit.py --to-remote-tip. Full mechanism,
the five confirmed historical instances, and the fallback trade: see
_write_and_commit_pass's docstring. Do not "simplify" that flag away.

FAIL-OPEN, ALWAYS. A missing, corrupt, or unparseable cache file -- or any
unexpected exception anywhere in the lookup path -- resolves to a MISS, never an
error propagated to the caller. `record` never raises either: recording is
best-effort and must never be able to block the commit it is a side effect of.
Coverage (whether the full suite runs) never depends on this module being healthy.

NEVER A SILENT SKIP. Every cache HIT is logged with the record, the hash, the
recorded-by commit/session, and its age -- this is validation REUSE, not
validation BYPASS: the content genuinely was already tested; the record just
avoids re-paying for a test that would necessarily reproduce the same result.

NOT REACHABLE FROM THE EXPERIMENT-RUNNER / PHASE3 WRITER PATH. This module is
only ever invoked from precommit_contracts.sh (a Mac-only git hook); nothing in
sync_daemon, the coordinator, or the runner imports or calls it, and it must stay
that way (design doc section 7).

CLI
---
    validation_cache.py check --repo-root PATH --tier NAME
        [--cache-path PATH] [--ttl-minutes N]
    Exit 0 = HIT (also prints the loud audit lines). Exit 1 = MISS.

    validation_cache.py record --repo-root PATH --tier NAME --result {pass,fail}
        [--cache-path PATH] [--ttl-minutes N] [--push] [--session-id ID]
        [--no-bot] [--ree-commit-path PATH]
    Always exits 0 -- best-effort. `--result fail` is a deliberate, logged no-op
    (a failing run must never write a cache entry); the flag exists so the shell
    caller can invoke `record` unconditionally after either outcome instead of
    branching on pass/fail itself.

ASCII-only output (repo rule).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# ree-v3/scripts/validation_cache.py -> parents[0]=ree-v3/scripts, [1]=ree-v3,
# [2]=REE_Working. Fixed regardless of the --repo-root being hashed (which may
# be a throwaway test fixture) -- the machine-class/commit-tooling imports
# below always come from the REAL, adjacent checkouts, never from repo-root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_REE_V3_ROOT = _SCRIPT_DIR.parent
_UMBRELLA_ROOT = _REE_V3_ROOT.parent

# Shared atomic-replace / CAS-retry / commit-landed-locally primitives,
# IMPORTED rather than restated -- see chip_ledger.py for the identical
# reasoning and the 2026-08-09 TASK_CLAIMS.json incident that motivated it.
# Cross-repo (ree-v3/scripts -> REE_Working/scripts) by path, matching the
# existing precedent in precommit_contracts.sh itself
# ($REPO/../scripts/remote_pytest.sh) -- this module only ever runs on the
# Mac (a git hook), where REE_Working/scripts is always present at this fixed
# relative location; it is never shipped to a cloud worker or the hub.
sys.path.insert(0, str(_UMBRELLA_ROOT / "scripts"))
from task_claim import (  # noqa: E402
    CommitLandedLocally,
    atomic_write_text,
    default_session_id,
    head_sha,
    ree_commit_once,
)

SCHEMA_VERSION = "contract_validation_cache/v1"
DEFAULT_CACHE_REL = ".contract_validation_cache.json"
DEFAULT_TTL_MINUTES = 45
# Bounds file growth by dropping records this stale on every write, regardless
# of the TTL any particular caller asked for -- generous relative to the 30-60
# min TTL window so it never interferes with a legitimately long TTL override,
# it only stops the file accumulating dead entries forever.
PRUNE_AFTER_MINUTES = 24 * 60
MAX_COMMIT_ATTEMPTS = 3
# The trees Block 2's own trigger reads (precommit_contracts.sh: STAGED grep
# '^(ree_core/|experiments/_lib/)'). Kept as a single source of truth here;
# if that trigger scope ever changes, this must change with it.
RELEVANT_TREES = ("ree_core", "experiments/_lib")

REE_COMMIT_DEFAULT = _UMBRELLA_ROOT / "scripts" / "ree_commit.py"


def utc_now() -> str:
    """Current UTC from the system clock. Never estimated or derived from context."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print("[validation_cache] %s" % msg)


# ---------------------------------------------------------------------------
# Cache key components
# ---------------------------------------------------------------------------

def compute_relevant_hash(repo_root) -> dict:
    """Content hash over the WORKING-TREE state of ree_core/ + experiments/_lib/.

    Every file under those two trees (not just *.py -- the git trigger this
    mirrors is extension-agnostic), read from disk rather than git objects:
    both the local pytest run and remote_pytest.sh's rsync validate on-disk
    content including uncommitted edits, so that -- not HEAD -- is what the
    suite actually exercises.

    Deliberately does NOT reuse experiments/_lib/arm_fingerprint.py's
    compute_substrate_hash(): that function additionally folds in
    experiments/_harness.py and experiments/_metrics.py (outside Block 2's own
    trigger scope) and exists for a different purpose -- scientific arm-reuse
    fingerprinting. Coupling this engineering cache to it would blur the
    "engineering landing and scientific closure remain fully separate"
    invariant (design doc section 7). The hashing STYLE (sorted rel paths,
    rel+NUL+sha256hex+newline) intentionally matches it for readability, but
    the scope is independent and purpose-built.

    Over-inclusion (hashing a file that turns out not to affect the suite) only
    ever causes a cheap false miss; under-inclusion would risk a false hit,
    which is why the whole tree is walked rather than a narrower glob.
    """
    root = Path(repo_root).resolve()
    paths = {}
    for tree in RELEVANT_TREES:
        base = root / tree
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if "__pycache__" in p.parts:
                continue
            if p.name.startswith("."):
                continue
            paths[str(p.relative_to(root))] = p

    h = hashlib.sha256()
    missing = []
    for rel in sorted(paths):
        p = paths[rel]
        try:
            content = p.read_bytes()
        except OSError:
            missing.append(rel)
            continue
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(content).hexdigest().encode("ascii"))
        h.update(b"\n")
    for rel in missing:
        h.update(b"MISSING\0")
        h.update(rel.encode("utf-8"))
        h.update(b"\n")
    return {
        "relevant_hash": h.hexdigest(),
        "n_files": len(paths) - len(missing),
        "missing": missing,
        "trees": list(RELEVANT_TREES),
    }


def compute_machine_class() -> str:
    """Machine-class + toolchain identity, torch-version-inclusive.

    Delegates to experiments/_lib/arm_fingerprint.machine_class() -- the
    canonical, already torch-fixed implementation (2026-07-19 hard cut) --
    instead of recomputing it, so this cache cannot silently reintroduce the
    omits-torch-version gap that function itself exists to close. See the
    module docstring's "CACHE KEY" section for why this axis matters here.
    """
    try:
        from experiments._lib.arm_fingerprint import machine_class as _machine_class
    except Exception:
        sys.path.insert(0, str(_REE_V3_ROOT / "experiments"))
        from _lib.arm_fingerprint import machine_class as _machine_class  # type: ignore
    return _machine_class()


def cache_key_for(tier: str, relevant_hash: str, machine_class: str) -> str:
    raw = "validation_cache/v1|%s|%s|%s" % (tier, relevant_hash, machine_class)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _age_minutes(iso_ts):
    """Minutes between `iso_ts` (UTC, "%Y-%m-%dT%H:%M:%SZ") and now. None if
    `iso_ts` is missing or unparseable -- callers treat that as a miss."""
    if not iso_ts:
        return None
    try:
        dt = datetime.strptime(iso_ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _empty_cache() -> dict:
    return {"schema_version": SCHEMA_VERSION, "records": {}}


def load_cache(cache_path: Path) -> dict:
    """Fail-open: missing/corrupt/unexpected-shape -> an empty cache, never an
    exception. This is the load-bearing half of invariant 5 (fail-open)."""
    if not cache_path.exists():
        return _empty_cache()
    try:
        raw = cache_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return _empty_cache()
    if not isinstance(data, dict) or not isinstance(data.get("records"), dict):
        return _empty_cache()
    return data


def _prune_expired(data: dict, floor_minutes: float = PRUNE_AFTER_MINUTES) -> int:
    """Drop records older than `floor_minutes`, in place. Returns count dropped.

    Bounds file growth; independent of any particular caller's --ttl-minutes so
    it can never make a legitimate longer-TTL hit disappear."""
    records = data.get("records", {})
    stale = [k for k, v in records.items()
            if (_age_minutes(v.get("recorded_at")) or 0) > floor_minutes]
    for k in stale:
        del records[k]
    return len(stale)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def lookup(cache_path: Path, repo_root, tier: str, ttl_minutes: float) -> tuple:
    """Returns (hit: bool, info: dict). NEVER raises -- any internal error
    resolves to a miss with the exception recorded in info['error']. This is
    the fail-open invariant applied to the read path."""
    info = {}
    try:
        rel = compute_relevant_hash(repo_root)
        mc = compute_machine_class()
        key = cache_key_for(tier, rel["relevant_hash"], mc)
        info.update(tier=tier, relevant_hash=rel["relevant_hash"], n_files=rel["n_files"],
                    machine_class=mc, cache_key=key, cache_path=str(cache_path),
                    ttl_minutes=ttl_minutes)
        data = load_cache(cache_path)
        record = data.get("records", {}).get(key)
        if record is None:
            info["reason"] = "no record for this key"
            return False, info
        info["recorded_at"] = record.get("recorded_at")
        info["recorded_by_session_id"] = record.get("recorded_by_session_id")
        info["recorded_by_commit"] = record.get("recorded_by_commit")
        age_min = _age_minutes(record.get("recorded_at"))
        info["age_minutes"] = age_min
        if age_min is None:
            info["reason"] = "record has a missing/unparseable recorded_at timestamp"
            return False, info
        if age_min > ttl_minutes:
            info["reason"] = "record is %.1f min old, TTL is %s min" % (age_min, ttl_minutes)
            return False, info
        info["reason"] = "hit"
        return True, info
    except Exception as exc:  # fail-open: cache machinery must never block/alter behavior
        info["error"] = "%s: %s" % (type(exc).__name__, exc)
        info["reason"] = "cache lookup raised -- treating as miss (fail-open)"
        return False, info


def cmd_check(args) -> int:
    cache_path = Path(args.cache_path) if args.cache_path else \
        Path(args.repo_root).resolve() / DEFAULT_CACHE_REL
    hit, info = lookup(cache_path, args.repo_root, args.tier, args.ttl_minutes)
    if hit:
        log("HIT -- key=%s tier=%s machine_class=%s" % (info["cache_key"][:16], info["tier"], info["machine_class"]))
        log("  relevant_hash=%s (%d files)" % (info["relevant_hash"][:16], info["n_files"]))
        log("  recorded_at=%s (%.1f min ago) by session=%s commit=%s"
            % (info.get("recorded_at"), info.get("age_minutes") or 0.0,
               info.get("recorded_by_session_id"), info.get("recorded_by_commit")))
        log("  reusing this result instead of re-running the suite -- see cache_path=%s" % info["cache_path"])
        return 0
    log("MISS -- %s (key=%s tier=%s machine_class=%s relevant_hash=%s)"
        % (info.get("reason", "unknown"), info.get("cache_key", "?")[:16], args.tier,
           info.get("machine_class", "?"), info.get("relevant_hash", "?")[:16]))
    if "error" in info:
        log("  (%s)" % info["error"])
    return 1


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------

def _write_and_commit_pass(cache_path: Path, repo_root, tier: str, session_id, push: bool,
                           bot: bool, ree_commit_path: Path,
                           remote_tip: bool = True) -> str:
    """Upsert one cache_key entry after a PASSING run, then commit via
    ree_commit.py. Narrow structural update (load, upsert one key, write) --
    not a whole-file rewrite -- and re-read happens immediately before write on
    every attempt (Concurrency Rules: re-read right before editing a shared
    file). CAS-retry loop mirrors chip_ledger.py's mutate_and_commit.

    Best-effort: every failure mode returns a status string rather than
    raising -- recording must never be able to block the caller's own commit.

    THE COMMIT MUST NOT MOVE refs/heads/<branch> -- THIS IS A CORRECTNESS
    REQUIREMENT, NOT A PREFERENCE (2026-08-27, chip-20260827-precommit-cache-
    self-collision). This function only ever runs from INSIDE a pre-commit
    hook (precommit_contracts.sh Block 2's record_validation_cache_result),
    and ree_commit.py's own flow is:

        old_head = rev-parse HEAD
        build private index
        run_pre_commit_hook(...)          <-- THIS FUNCTION RUNS IN HERE
        commit-tree <tree> -p old_head
        update-ref <ref> <new> <old_head>  <-- compare-and-swap

    so an ordinary local commit made here advances the branch and makes the
    OUTER commit's `old_head` stale, and the CAS then fails with "branch moved
    under us while building the commit ... Nothing was committed." The gate
    rejects the very commit it was invoked to gate -- AFTER paying the ~13min
    suite. It is deterministic, not a race: it fires on every cache MISS whose
    suite PASSes (a HIT exits before recording, which is why the cold path was
    the only one affected and the defect survived from 2026-08-10). Confirmed
    on five historical instances, all with the identical signature -- the cache
    commit lands and the gated commit lands as its CHILD one to eight minutes
    later, i.e. after the failure and a manual retry that then hit the cache:
    55dbe77 -> 031377d (2026-08-27), 0481c1c -> 88287f1, 547f053 -> 492e51f,
    12d6839 -> 1a4b6be, 6702d5a -> 775eb55.

    `remote_tip` (default True) therefore routes the commit through
    ree_commit.py --to-remote-tip, which lands it on origin/<branch> via the
    already-hardened throwaway-worktree cherry-pick path and NEVER moves the
    local ref, so the outer CAS's `old_head` stays valid. Nothing about the
    record itself changes: same content, same push, same audit trail. On a
    genuine conflict or lock contention --to-remote-tip falls back to exactly
    today's local CAS landing, which re-exposes the collision -- accepted,
    because that path is rare and the fallback is the pre-existing "the commit
    is safe locally" safety net, not a new risk.

    `--to-remote-tip` requires `--push`, so with push=False this degrades to an
    ordinary local commit and the collision is back. That combination is
    test-only: precommit_contracts.sh's record_validation_cache_result always
    passes --push. Do NOT call `record` without --push from inside a hook.

    Note also which invariant this restores. The design doc's safety-invariant
    list (REE_assembly/docs/architecture/landing_integration_worker_investigation.md
    sec 7) asserts "Path-scoped/intended-file commits, CAS ... untouched" --
    the cache write was in fact defeating the CAS, so this is a regression of a
    documented invariant, not an optimisation.
    """
    repo_root = Path(repo_root).resolve()
    try:
        rel_cache_path = str(cache_path.resolve().relative_to(repo_root))
    except ValueError:
        return "record: FAILED -- cache_path %s is not under repo_root %s" % (cache_path, repo_root)

    last_error = None
    for attempt in range(1, MAX_COMMIT_ATTEMPTS + 1):
        try:
            rel = compute_relevant_hash(repo_root)
            mc = compute_machine_class()
            key = cache_key_for(tier, rel["relevant_hash"], mc)
            data = load_cache(cache_path)  # re-read immediately before write
            n_pruned = _prune_expired(data)
            # recorded_by_commit is the base HEAD the validated WORKING TREE sits
            # on top of, captured before this call does anything -- not the sha
            # of the cache-commit itself, which cannot be known before it exists
            # (chicken-and-egg). This is what a later consumer can actually use:
            # `git log <sha>` shows what the validated content was layered on.
            data["records"][key] = {
                "tier": tier,
                "relevant_hash": rel["relevant_hash"],
                "n_files": rel["n_files"],
                "machine_class": mc,
                "recorded_at": utc_now(),
                "recorded_by_session_id": session_id,
                "recorded_by_commit": head_sha(repo_root),
                "result": "pass",
            }
            text = json.dumps(data, indent=2, sort_keys=True) + "\n"
            json.loads(text)  # sanity check before touching disk
            atomic_write_text(cache_path, text)
            try:
                sha = ree_commit_once(
                    repo_root,
                    rel_cache_path,
                    "contract validation cache: record pass (tier=%s, key=%s)"
                    % (tier, key[:12]),
                    push,
                    bot,
                    ree_commit_path=ree_commit_path,
                    # Never move refs/heads/<branch> -- see this function's
                    # docstring. ree_commit_once() makes this a no-op when
                    # push is False (--to-remote-tip requires --push).
                    to_remote_tip=remote_tip,
                )
            except CommitLandedLocally as landed:
                return ("record: committed %s locally but a later step failed (exit %d) -- "
                        "content IS saved, see: git -C %s push origin HEAD"
                        % (landed.sha[:10], landed.code, repo_root))
            return "record: cache entry written and committed (key=%s..., sha=%s, pruned=%d)" \
                % (key[:12], (sha or "?")[:10], n_pruned)
        except SystemExit as exc:
            last_error = "ree_commit.py exited %s" % exc.code
            continue
        except Exception as exc:
            last_error = "%s: %s" % (type(exc).__name__, exc)
            continue
    return "record: FAILED after %d attempts (%s) -- cache not updated, suite result unaffected" \
        % (MAX_COMMIT_ATTEMPTS, last_error)


def cmd_record(args) -> int:
    cache_path = Path(args.cache_path) if args.cache_path else \
        Path(args.repo_root).resolve() / DEFAULT_CACHE_REL
    if args.result == "fail":
        log("not recording -- run was a FAIL (a failing run must never certify a cache hit)")
        return 0
    session_id = args.session_id or default_session_id()
    try:
        status = _write_and_commit_pass(
            cache_path, args.repo_root, args.tier, session_id, args.push,
            not args.no_bot, Path(args.ree_commit_path) if args.ree_commit_path else REE_COMMIT_DEFAULT,
            remote_tip=not args.no_remote_tip,
        )
    except Exception as exc:  # belt-and-braces: record must NEVER raise out to the caller
        status = "record: FAILED (%s: %s) -- cache not updated, suite result unaffected" \
            % (type(exc).__name__, exc)
    log(status)
    return 0  # always 0: recording is best-effort and must never block the commit


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Tree-content-hash-keyed validation-result cache for "
                    "precommit_contracts.sh Block 2.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", required=True, help="ree-v3 checkout root to hash")
    common.add_argument("--tier", required=True, help="validation tier name")
    common.add_argument("--cache-path", default=None,
                        help="override the cache file path (default: <repo-root>/%s)"
                        % DEFAULT_CACHE_REL)
    common.add_argument("--ttl-minutes", type=float, default=DEFAULT_TTL_MINUTES,
                        help="a hit older than this is treated as a miss (default %d)"
                        % DEFAULT_TTL_MINUTES)

    c = sub.add_parser("check", parents=[common], help="look up a cache hit; exit 0=hit, 1=miss")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("record", parents=[common], help="record a validation result (best-effort)")
    r.add_argument("--result", required=True, choices=("pass", "fail"),
                   help="the suite's actual outcome; 'fail' is a logged no-op")
    r.add_argument("--push", action="store_true", help="push the commit (default: local only)")
    r.add_argument("--session-id", default=None,
                   help="defaults to the current worktree slug, if any")
    r.add_argument("--no-bot", action="store_true",
                   help="author as the real user instead of the automation bot identity")
    r.add_argument("--no-remote-tip", action="store_true",
                   help="land the commit on the LOCAL branch instead of straight "
                        "to origin's tip. Escape hatch only: the default exists "
                        "because this command runs inside a pre-commit hook and a "
                        "local ref move defeats the outer ree_commit.py CAS -- see "
                        "_write_and_commit_pass's docstring.")
    r.add_argument("--ree-commit-path", default=None,
                   help="override the ree_commit.py path (tests only)")
    r.set_defaults(func=cmd_record)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
