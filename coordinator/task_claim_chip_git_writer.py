"""TASK_CLAIMS.json / TASK_CHIPS.json DB->git materializer (PHASE-2b).

PHASE-2b of the migration described in REE_assembly/evidence/planning/
task_claim_chip_coordinator_migration_plan.md: the analogue of sync_daemon's
phase3_*_writer family for the two umbrella-repo coordination registries.
Once clients stop writing git themselves (the cutover this module gates),
this is the ONE git writer for TASK_CLAIMS.json and TASK_CHIPS.json.

One tick per invocation (pace with a systemd timer, same convention as
task_claim_chip_shadow_sync.py). Each tick:

  1. INGEST  -- fetch origin, read both files at origin/<branch>, and
     reconcile them into the DB via db.reconcile_task_claims /
     db.reconcile_chips (upsert-only; the reconciler never deletes).
     This is NOT redundant with the shadow-sync timer: it closes the
     fallback-commit race. A session whose coordinator is unreachable
     falls back to today's git-mutate-and-commit path (plan doc 5.3), and
     a render taken from a DB that has not yet ingested that commit would
     DROP the fallback entry -- a silent revert of live work. Ingesting in
     the same tick, immediately before rendering, makes the render a
     superset of origin's content by construction.
     Deliberately does NOT call db.log_task_claim_chip_drift: the drift
     log's total_ticks is the PHASE-1 soak denominator and belongs to the
     shadow-sync timer's cadence alone.

  2. RENDER  -- serialise both registries from the DB's lossless
     entry_json blobs (plan doc 5.2.1), preserving:
       * per-entry key order (entry_json is stored VERBATIM since
         2026-08-28 -- file order for reconciled rows, the client's
         canonical construction order for coordinator-mutated rows);
       * row order (rowid = insertion order = the files' append order);
       * each file's own top-level key order and metadata
         ("claims, schema_version, stale_after_hours" vs
         "schema_version, chips" -- D15), with metadata carried over from
         the source document rather than re-invented;
       * the client serialisation format exactly:
         json.dumps(doc, indent=2) + "\\n".

     RETENTION (D14): the mirror is a SUPERSET of the claims file --
     prune_task_claims_done.py removes done entries older than 24h, the
     reconciler never deletes. A naive dump would resurrect every claim
     ever pruned (measured 228 DB rows vs 63 in git, 2026-08-28). The
     rule is applied statelessly at render time: emit status != 'done',
     plus 'done' whose closed_at (falling back to claimed_at) is within
     RETAIN_HOURS (24, matching the pruner). Consequence: pruning stops
     being an event a session performs and becomes a property of what
     gets rendered. Chips are NEVER dropped (D5): every chip row renders.

  3. COMPARE -- byte-compare each render against origin's file.
     mode=check (the default): report, write nothing. This is the
     deployment soak: run in check mode until renders match origin (or
     mismatch only in the expected ways: retention drops, coordinator-
     mutated rows), THEN flip the clients and switch to write mode.
     mode=write: if either file differs, reset the dedicated writer
     checkout hard to origin/<branch>, write both files, commit, push.
     Commit-on-state-change only -- an unchanged render produces no
     commit (root CLAUDE.md forbids reintroducing liveness-tick commits).

  4. PUSH-RETRY (write mode) -- a rejected push means a concurrent
     client/fallback commit landed. The whole tick re-runs (ingest ->
     render -> commit) up to --max-push-retries times; because ingest
     folds origin's new content into the DB first, the retry is lossless
     by construction. This checkout is DEDICATED to the writer: nothing
     else edits it, so `git reset --hard origin/<branch>` here is the
     same derived-materialisation idiom the phase3 queue writer uses (and
     the reason the ref-move guard is not installed on the hub).

Deploy shape: deploy/ree-task-claim-chip-git-writer.{service,timer} --
Type=oneshot + timer, a WRITABLE clone of the umbrella repo (e.g.
/home/ree/REE_Working_registry_writer), env REGISTRY_WRITER_MODE=check
until the cutover decision. All printed text is ASCII-only.
"""

import argparse
import json
import os
import subprocess
import sys

import db

DEFAULT_REPO_PATH = os.environ.get("REGISTRY_WRITER_REPO_PATH", "")
DEFAULT_BRANCH = os.environ.get("REGISTRY_WRITER_BRANCH", "master")
DEFAULT_MODE = os.environ.get("REGISTRY_WRITER_MODE", "check")
CLAIMS_REL_PATH = "TASK_CLAIMS.json"
CHIPS_REL_PATH = "TASK_CHIPS.json"
RETAIN_HOURS = float(os.environ.get("COORDINATOR_TASK_CLAIM_RETAIN_HOURS", "24"))
COMMIT_PREFIX = "phase2b-registry:"


def _git(repo_path, *args, check=True, timeout=60):
    return subprocess.run(
        ["git", "-C", repo_path] + list(args),
        check=check, capture_output=True, timeout=timeout)


def _fetch_and_resolve(repo_path, branch):
    """fetch origin, resolve origin/<branch> to a sha. Returns None on
    failure (callers abort the tick -- a materializer must never render
    against unknown origin state)."""
    try:
        _git(repo_path, "fetch", "--quiet", "origin", branch)
        out = _git(repo_path, "rev-parse", "origin/%s" % branch, timeout=15)
        return out.stdout.decode("utf-8").strip()
    except Exception as exc:  # noqa: BLE001 -- degrade, never crash
        sys.stderr.write("[registry-writer] WARN fetch/resolve failed for "
                         "%r origin/%s: %r\n" % (repo_path, branch, exc))
        return None


def _show_text(repo_path, rev, rel_path):
    try:
        out = _git(repo_path, "show", "%s:%s" % (rev, rel_path), timeout=30)
        return out.stdout.decode("utf-8")
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write("[registry-writer] WARN could not read %s at %s: "
                         "%r\n" % (rel_path, rev, exc))
        return None


def _parse_iso(stamp):
    """ISO-8601 Z stamp -> epoch seconds, or None. Mirrors the pruner's
    tolerance: an unparseable stamp is treated as un-ageable (kept)."""
    if not stamp:
        return None
    try:
        import datetime
        return datetime.datetime.strptime(
            stamp, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _now_epoch(now_iso=None):
    if now_iso is not None:
        parsed = _parse_iso(now_iso)
        if parsed is not None:
            return parsed
    parsed = _parse_iso(db.utcnow())
    return parsed if parsed is not None else 0.0


def _source_order(source_doc, list_key, key_fn):
    """entry key -> position in the source file, for order-faithful
    rendering. The FILE's row order is authoritative for rows it carries:
    a client-side rewrite (chip_ledger's merge_origin_into_local) can
    reorder the file relative to DB insertion order, and a render that
    ignored that would emit a content-identical but byte-different file
    every tick. DB-only rows (coordinator-opened, not yet materialized)
    append after, in rowid order -- the same append semantics the clients
    have."""
    order = {}
    for i, entry in enumerate((source_doc or {}).get(list_key, [])):
        key = key_fn(entry)
        if key is not None and key not in order:
            order[key] = i
    return order


def render_task_claims(conn, source_doc=None, now_iso=None):
    """Render TASK_CLAIMS.json's full text from the DB.

    D14 retention at render time: keep every non-'done' row, plus 'done'
    rows whose closed_at (fallback claimed_at) is within RETAIN_HOURS of
    `now`. Strictly a subset of what a pruner-less git file carries --
    the rule never resurrects anything git dropped (measured 2026-08-28:
    0 rule-only entries against the live file).

    `now_iso` is injected for testability (matching db.py's explicit-now
    convention). Top-level metadata is carried over from `source_doc`
    (the file as read from origin this tick) so the materializer never
    invents values; defaults cover a missing/first-run source.
    """
    now_epoch = _now_epoch(now_iso)
    cutoff = now_epoch - RETAIN_HOURS * 3600.0
    order = _source_order(
        source_doc, "claims",
        lambda e: (e.get("session_id"), e.get("claimed_at")))
    kept = []
    n_dropped = 0
    rows = conn.execute(
        "SELECT rowid, entry_json, session_id, claimed_at, status, closed_at "
        "FROM task_claims ORDER BY rowid").fetchall()
    for row in rows:
        if row["status"] == "done":
            stamp = _parse_iso(row["closed_at"]) or _parse_iso(row["claimed_at"])
            if stamp is not None and stamp < cutoff:
                n_dropped += 1
                continue
        try:
            entry = json.loads(row["entry_json"])
        except (TypeError, ValueError):
            sys.stderr.write("[registry-writer] WARN unparseable entry_json "
                             "for claim row; skipping\n")
            continue
        key = (row["session_id"], row["claimed_at"])
        kept.append((order.get(key, len(order) + row["rowid"]), entry))
    kept.sort(key=lambda pair: pair[0])
    entries = [entry for _, entry in kept]
    source_doc = source_doc or {}
    doc = {
        "claims": entries,
        "schema_version": source_doc.get("schema_version", "task_claims/v1"),
        "stale_after_hours": source_doc.get("stale_after_hours", 6),
    }
    return json.dumps(doc, indent=2) + "\n", {
        "n_rendered": len(entries), "n_retention_dropped": n_dropped}


def render_chips(conn, source_doc=None):
    """Render TASK_CHIPS.json's full text from the DB. Every row renders
    -- chips are never deleted (D5); archiving strips fields in place and
    the stripped state is already reflected in entry_json."""
    order = _source_order(source_doc, "chips", lambda e: e.get("chip_ref"))
    kept = []
    rows = conn.execute(
        "SELECT rowid, chip_ref, entry_json FROM chip_ledger "
        "ORDER BY rowid").fetchall()
    for row in rows:
        try:
            entry = json.loads(row["entry_json"])
        except (TypeError, ValueError):
            sys.stderr.write("[registry-writer] WARN unparseable entry_json "
                             "for chip row; skipping\n")
            continue
        kept.append((order.get(row["chip_ref"], len(order) + row["rowid"]),
                     entry))
    kept.sort(key=lambda pair: pair[0])
    entries = [entry for _, entry in kept]
    source_doc = source_doc or {}
    doc = {
        "schema_version": source_doc.get("schema_version", "task_chips/v1"),
        "chips": entries,
    }
    return json.dumps(doc, indent=2) + "\n", {"n_rendered": len(entries)}


def ingest(conn, claims_doc, chips_doc):
    """Reconcile origin's current content into the DB, one transaction.
    Upsert-only (the reconcilers never delete), no drift-log row -- see
    the module docstring for both whys."""
    claims = (claims_doc or {}).get("claims", []) if claims_doc else []
    chips = (chips_doc or {}).get("chips", []) if chips_doc else []
    conn.execute("BEGIN IMMEDIATE")
    try:
        claim_stats = db.reconcile_task_claims(conn, claims)
        chip_stats = db.reconcile_chips(conn, chips)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return claim_stats, chip_stats


def _texts_match(source_text, rendered_text):
    """Byte-equality, tolerating exactly one historical quirk: a source
    file missing its trailing newline (the live TASK_CHIPS.json on
    origin/master ends `}` with no final newline -- some past writer
    dropped it). The render always emits the canonical client format
    (`json.dumps(doc, indent=2) + "\\n"`); a source differing ONLY by
    that final newline is the same content, and rewriting a 10 MB file
    to add one byte is exactly the no-op churn commit-on-state-change
    exists to prevent."""
    if source_text == rendered_text:
        return True
    return (source_text is not None
            and not source_text.endswith("\n")
            and rendered_text == source_text + "\n")


def _semantic_delta(source_doc, rendered_text, key_fields):
    """Which entry keys the render adds/drops vs the source file. Cheap
    orientation for a check-mode mismatch report; byte-equality is the
    real gate."""
    def keys_of(doc, list_key):
        out = set()
        for e in (doc or {}).get(list_key, []):
            out.add(tuple(str(e.get(f)) for f in key_fields))
        return out
    list_key = "claims" if "claims" in (source_doc or {}) else "chips"
    src = keys_of(source_doc, list_key)
    try:
        rnd = keys_of(json.loads(rendered_text), list_key)
    except ValueError:
        return {"added": -1, "dropped": -1}
    return {"added": len(rnd - src), "dropped": len(src - rnd)}


def materialize_once(conn, repo_path, branch=None, mode=None, now_iso=None,
                     max_push_retries=3):
    """One materializer tick. Returns a result dict (ASCII-safe values)
    or None when origin state could not be established."""
    branch = branch or DEFAULT_BRANCH
    mode = mode or DEFAULT_MODE
    for attempt in range(max_push_retries + 1):
        sha = _fetch_and_resolve(repo_path, branch)
        if sha is None:
            return None
        claims_text = _show_text(repo_path, sha, CLAIMS_REL_PATH)
        chips_text = _show_text(repo_path, sha, CHIPS_REL_PATH)
        if claims_text is None and chips_text is None:
            return None
        try:
            claims_doc = json.loads(claims_text) if claims_text else None
        except ValueError:
            claims_doc = None
        try:
            chips_doc = json.loads(chips_text) if chips_text else None
        except ValueError:
            chips_doc = None

        ingest(conn, claims_doc, chips_doc)
        claims_render, claims_stats = render_task_claims(
            conn, source_doc=claims_doc, now_iso=now_iso)
        chips_render, chips_stats = render_chips(conn, source_doc=chips_doc)

        claims_match = _texts_match(claims_text, claims_render)
        chips_match = _texts_match(chips_text, chips_render)
        result = {
            "source_ref": sha,
            "mode": mode,
            "claims_match": claims_match,
            "chips_match": chips_match,
            "claims": claims_stats,
            "chips": chips_stats,
            "committed": False,
        }
        if not claims_match:
            result["claims_delta"] = _semantic_delta(
                claims_doc, claims_render, ("session_id", "claimed_at"))
        if not chips_match:
            result["chips_delta"] = _semantic_delta(
                chips_doc, chips_render, ("chip_ref",))

        if mode != "write" or (claims_match and chips_match):
            return result

        # -- write mode, something differs: materialize onto origin tip.
        try:
            _git(repo_path, "checkout", "--quiet", branch)
            _git(repo_path, "reset", "--hard", "--quiet", sha)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("[registry-writer] ERROR could not reset writer "
                             "checkout: %r\n" % (exc,))
            return result
        wrote = []
        if not claims_match:
            with open(os.path.join(repo_path, CLAIMS_REL_PATH), "w",
                      encoding="utf-8") as fh:
                fh.write(claims_render)
            wrote.append(CLAIMS_REL_PATH)
        if not chips_match:
            with open(os.path.join(repo_path, CHIPS_REL_PATH), "w",
                      encoding="utf-8") as fh:
                fh.write(chips_render)
            wrote.append(CHIPS_REL_PATH)
        msg = ("%s materialize %s (claims %d kept/%d aged-out, chips %d)"
               % (COMMIT_PREFIX, "+".join(wrote),
                  claims_stats["n_rendered"],
                  claims_stats["n_retention_dropped"],
                  chips_stats["n_rendered"]))
        try:
            _git(repo_path, "add", "--", *wrote)
            _git(repo_path, "commit", "--quiet", "-m", msg)
            push = _git(repo_path, "push", "--quiet", "origin",
                        "HEAD:%s" % branch, check=False, timeout=120)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("[registry-writer] ERROR commit failed: %r\n"
                             % (exc,))
            return result
        if push.returncode == 0:
            result["committed"] = True
            return result
        sys.stderr.write("[registry-writer] push rejected (attempt %d/%d); "
                         "re-running tick\n"
                         % (attempt + 1, max_push_retries + 1))
    sys.stderr.write("[registry-writer] ERROR push retries exhausted\n")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="DB->git materializer for TASK_CLAIMS.json/"
                    "TASK_CHIPS.json (PHASE-2b). One tick per invocation; "
                    "pace with the paired systemd timer.")
    ap.add_argument("--repo-path", default=DEFAULT_REPO_PATH,
                    help="WRITABLE clone of the REE_Working umbrella repo "
                         "dedicated to this writer.")
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--mode", choices=("check", "write"), default=DEFAULT_MODE,
                    help="check: render + compare + report, never write "
                         "(the deployment soak). write: commit + push on "
                         "a state change.")
    ap.add_argument("--db", default=os.environ.get(
        "COORDINATOR_DB", os.path.join(
            os.path.dirname(__file__), "coordinator.db")))
    ap.add_argument("--max-push-retries", type=int, default=3)
    args = ap.parse_args()

    if not args.repo_path:
        sys.stderr.write("[registry-writer] ERROR --repo-path (or "
                         "REGISTRY_WRITER_REPO_PATH) is required\n")
        return 2
    db.init_db(args.db)
    conn = db.connect(args.db)
    try:
        result = materialize_once(conn, args.repo_path, branch=args.branch,
                                  mode=args.mode,
                                  max_push_retries=args.max_push_retries)
    finally:
        conn.close()
    if result is None:
        sys.stderr.write("[registry-writer] nothing materialized this tick "
                         "(origin unreadable)\n")
        return 1
    sys.stdout.write(
        "[registry-writer] ref=%s mode=%s claims: match=%s rendered=%d "
        "aged_out=%d | chips: match=%s rendered=%d | committed=%s\n" % (
            result["source_ref"][:12], result["mode"],
            result["claims_match"], result["claims"]["n_rendered"],
            result["claims"]["n_retention_dropped"],
            result["chips_match"], result["chips"]["n_rendered"],
            result["committed"]))
    for side in ("claims_delta", "chips_delta"):
        if side in result:
            sys.stdout.write("[registry-writer]   %s: +%d -%d vs source\n"
                             % (side, result[side]["added"],
                                result[side]["dropped"]))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
