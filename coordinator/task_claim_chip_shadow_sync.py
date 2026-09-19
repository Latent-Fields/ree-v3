"""TASK_CLAIMS.json / TASK_CHIPS.json shadow mirror sync.

PHASE 1 (shadow) of the migration described in REE_assembly/evidence/planning/
task_claim_chip_coordinator_migration_plan.md. Behaviour, by design:

  * Read TASK_CLAIMS.json and TASK_CHIPS.json from the AUTHORITATIVE git ref
    of the REE_Working umbrella repo (default origin/master; NOT this repo --
    see --repo-path) and reconcile the coordinator DB mirror (task_claims,
    task_claim_resources, chip_ledger) to match, so a human can watch the
    mirror converge before anything is asked to depend on it.
  * It does NOT write git. No autostash, no rebase, no commit, no push --
    this module is structurally incapable of the failure class the whole
    migration exists to remove. There is no code path in this file that can
    mutate a git working tree or ref; every git invocation below is `fetch`,
    `rev-parse`, or `show`, all read-only.
  * It does NOT expose a write path back through task_claim.py/chip_ledger.py
    either. Nothing in this codebase calls into this module from those CLIs.
    Phase 2 (the coordinator becoming the claim/chip AUTHORITY) is a
    separate, not-yet-built, not-yet-user-ratified step -- see the plan
    doc's HARD STOP note and PHASE-2 frontmatter node.

Modeled directly on sync_daemon.py's own PHASE 1 (shadow) design and its
`_load_queue_json` git-read pattern, generalised to a different repo (the
umbrella REE_Working, not ree-v3) and two files instead of one.

Deploy shape: a systemd `.timer` firing a `Type=oneshot` `.service` that runs
this module's main() once and exits (see deploy/ree-task-claim-chip-shadow-
sync.service + .timer), the same "committed-template" convention as
deploy/ree-git-wedge-watchdog.{service,timer}. Not a persistent loop -- the
timer owns pacing, this process owns one reconciliation tick.

All printed text is ASCII-only.
"""

import argparse
import json
import os
import subprocess
import sys

import db

DEFAULT_REPO_PATH = os.environ.get(
    "TASK_CLAIM_CHIP_REPO_PATH",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
DEFAULT_REF = os.environ.get("TASK_CLAIM_CHIP_SYNC_REF", "origin/master")
CLAIMS_REL_PATH = "TASK_CLAIMS.json"
CHIPS_REL_PATH = "TASK_CHIPS.json"


def _resolve_and_fetch(repo_path, ref):
    """`git fetch origin` then resolve `ref` to a commit sha, all read-only
    (never touches the working tree, mirrors sync_daemon._load_queue_json).
    Returns the sha, or None if git is unavailable / repo_path is not a git
    checkout / the fetch failed or timed out -- the caller then SKIPS the
    tick. It does not degrade to a local-file read (see
    load_source_documents for why that fallback was removed)."""
    try:
        subprocess.run(
            ["git", "-C", repo_path, "fetch", "--quiet", "origin"],
            check=True, capture_output=True, timeout=30)
        out = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", ref],
            check=True, capture_output=True, timeout=15)
        return out.stdout.decode("utf-8").strip()
    except Exception as exc:  # noqa: BLE001 -- skip the tick, never crash
        sys.stderr.write(
            "[task-claim-chip-sync] WARN git fetch/resolve failed for "
            "repo_path=%r ref=%r: %r\n" % (repo_path, ref, exc))
        return None


def _show_json(repo_path, rev, rel_path):
    """`git show <rev>:<rel_path>`, parsed as JSON. Read-only, never touches
    the working tree. Returns None on any failure (missing path, bad rev,
    invalid JSON) -- logged, never raised, so one bad file cannot wedge the
    whole reconciliation tick."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_path, "show",
             "%s:%s" % (rev, rel_path)],
            check=True, capture_output=True, timeout=15)
        return json.loads(out.stdout.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(
            "[task-claim-chip-sync] WARN could not read %s at %s: %r\n"
            % (rel_path, rev, exc))
        return None


def load_source_documents(repo_path, ref):
    """Read TASK_CLAIMS.json and TASK_CHIPS.json at `ref`. Returns
    (claims_doc, chips_doc, source_ref) where source_ref is the resolved
    commit sha, or (None, None, None) when the ref could not be fetched and
    resolved -- reconcile_once then skips the tick.

    NO WORKING-TREE FALLBACK (removed 2026-09-19, statusregress 70f6849fab).
    This function used to degrade to reading repo_path's working-tree copy
    and label the source "<ref>-stale-local". That was harmless while the
    mirror was a shadow nothing depended on. Since the 2026-08-28 cutover
    these tables are the AUTHORITY, and this module only ever fetches and
    shows -- it never checks anything out -- so the mirror clone's working
    tree is frozen at the commit it was provisioned from. On
    2026-09-18T12:52:35Z a single 30 s `git fetch` timeout made one tick
    read the hub mirror's 2026-08-26T20:23Z working-tree files (1707 chips,
    6923 commits behind origin) and upsert them through upsert_chip's "only
    git moved -> adopt git" branch: 22 terminal chips reopened, 1178 lost
    their `archived` marker, and open chips regained three-week-old claims,
    prompts and claim notes and lost handoff_pending trackers. The terminal
    and archived halves are now refused in upsert_chip (7f51dff); the rest
    of the damage has no row-level signature a guard could test, so the
    stale source itself must never be read. A skipped tick costs nothing:
    the registry writer ingests origin every 2 minutes on its own fetch.
    """
    sha = _resolve_and_fetch(repo_path, ref)
    if sha is None:
        return None, None, None
    claims_doc = _show_json(repo_path, sha, CLAIMS_REL_PATH)
    chips_doc = _show_json(repo_path, sha, CHIPS_REL_PATH)
    return claims_doc, chips_doc, sha


def reconcile_once(conn, repo_path, ref=None):
    """One shadow-mirror reconciliation tick: read both source files at the
    same resolved ref, upsert both tables inside one transaction, log one
    drift-log row. Returns (claim_stats, chip_stats, source_ref, diverged),
    or None if neither source file could be read at all (nothing to do --
    caller should log and exit non-zero rather than log a misleading
    all-zeros drift row).
    """
    ref = ref or DEFAULT_REF
    claims_doc, chips_doc, source_ref = load_source_documents(repo_path, ref)
    if claims_doc is None and chips_doc is None:
        return None
    claims = (claims_doc or {}).get("claims", []) if claims_doc else []
    chips = (chips_doc or {}).get("chips", []) if chips_doc else []

    conn.execute("BEGIN IMMEDIATE")
    try:
        claim_stats = db.reconcile_task_claims(conn, claims)
        chip_stats = db.reconcile_chips(conn, chips)
        diverged = db.log_task_claim_chip_drift(
            conn, source_ref, claim_stats, chip_stats)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return claim_stats, chip_stats, source_ref, diverged


def main():
    ap = argparse.ArgumentParser(
        description="Read-only shadow mirror of TASK_CLAIMS.json/"
                     "TASK_CHIPS.json into the coordinator DB. One tick "
                     "per invocation -- pace with a systemd timer, not an "
                     "internal loop (see deploy/ "
                     "ree-task-claim-chip-shadow-sync.{service,timer}).")
    ap.add_argument("--repo-path", default=DEFAULT_REPO_PATH,
                     help="Path to a git checkout of the REE_Working "
                          "umbrella repo (NOT ree-v3). Default assumes the "
                          "usual dev layout (ree-v3 is a sibling of "
                          "REE_Working's other repos): %r" % DEFAULT_REPO_PATH)
    ap.add_argument("--ref", default=DEFAULT_REF,
                     help="git ref to read (default %r)" % DEFAULT_REF)
    ap.add_argument("--db", default=os.environ.get(
        "COORDINATOR_DB", os.path.join(
            os.path.dirname(__file__), "coordinator.db")))
    args = ap.parse_args()

    db.init_db(args.db)
    conn = db.connect(args.db)
    try:
        result = reconcile_once(conn, args.repo_path, args.ref)
    finally:
        conn.close()

    if result is None:
        sys.stderr.write(
            "[task-claim-chip-sync] neither %s nor %s could be read from "
            "%s at %s; nothing reconciled this tick\n"
            % (CLAIMS_REL_PATH, CHIPS_REL_PATH, args.repo_path, args.ref))
        return 1

    claim_stats, chip_stats, source_ref, diverged = result
    sys.stdout.write(
        "[task-claim-chip-sync] ref=%s claims: git=%d db=%d new=%d "
        "updated=%d orphan=%d | chips: git=%d db=%d new=%d updated=%d "
        "orphan=%d | diverged=%d\n" % (
            source_ref,
            claim_stats["n_git"], claim_stats["n_db"], claim_stats["n_new"],
            claim_stats["n_updated"], len(claim_stats["orphans"]),
            chip_stats["n_git"], chip_stats["n_db"], chip_stats["n_new"],
            chip_stats["n_updated"], len(chip_stats["orphans"]),
            diverged,
        ))
    # Conditional suffix: entries neither reconciler could key. NOT part of
    # `diverged` (that means "the DB holds what git lost"; this is the other
    # direction, and folding it in would make the soak's zero-diverged exit
    # criterion unmeetable the same way `retired` did). The registry writer's
    # render preserves them; this line is what makes them visible.
    n_keyless = (claim_stats.get("n_keyless", 0)
                 + chip_stats.get("n_keyless", 0))
    if n_keyless:
        sys.stdout.write(
            "[task-claim-chip-sync] WARN keyless: claims=%d chips=%d -- "
            "source entries with no usable key, not mirrored\n"
            % (claim_stats.get("n_keyless", 0),
               chip_stats.get("n_keyless", 0)))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
