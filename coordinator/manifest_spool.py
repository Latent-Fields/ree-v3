"""Manifest filesystem spool for Phase 3 sole-git-writer.

POST /result currently records manifest metadata (run_id, sha, byte count)
into the DB but discards the actual manifest bytes. Phase 3 needs the bytes
to commit them into REE_assembly later, so /result now optionally spools
them to disk keyed on run_id. The Phase 3 writer in sync_daemon.py reads
pending manifests from the spool, commits them, and deletes the spool entry.

Both the spool write at /result time AND the writer that consumes it are
behind feature flags:
  COORDINATOR_SPOOL_DIR=<path>   enables /result spooling
  PHASE3_GIT_WRITER_READY=True   enables sync_daemon writer body

Default deployment has neither set, so behaviour is bit-identical to today.

Layout:
  <spool>/pending/<run_id>.json        manifest bytes (lossless)
  <spool>/pending/<run_id>.meta.json   metadata sidecar (received_at, sha,
                                       optional manifest_relpath hint)
After a successful commit the writer deletes both files. A partial-write
crash leaves only the .tmp file (atomic rename); the next /result for the
same run_id is idempotent (record_result is PK-on-run_id).

All printed text is ASCII-only (Windows cp1252 safety).
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

# run_id sanity: filesystem-safe ASCII tokens. Reject anything weird so we
# cannot be tricked into writing outside the spool root via a crafted POST.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")

# Default evidence-path convention when the runner does not supply one.
# Matches today's flat-manifest layout (evidence/experiments/<basename>_v3.json).
# Phase 3 PRs that touch coordinator_client.report_result should start sending
# manifest_relpath explicitly so the writer doesn't have to guess.
DEFAULT_EVIDENCE_PREFIX = "evidence/experiments"


def spool_root() -> Optional[Path]:
    """Return the configured spool root, or None when spooling is disabled."""
    raw = os.environ.get("COORDINATOR_SPOOL_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _pending_dir() -> Optional[Path]:
    root = spool_root()
    if root is None:
        return None
    return root / "pending"


def _safe_run_id(run_id: str) -> bool:
    if not run_id or run_id in (".", ".."):
        return False
    return _RUN_ID_RE.match(run_id) is not None


def write_manifest(
    run_id: str,
    raw: bytes,
    *,
    manifest_relpath: Optional[str] = None,
    received_at: Optional[str] = None,
    sha256_hex: Optional[str] = None,
) -> Optional[Path]:
    """Atomically spool one manifest. Returns the spooled path, or None when
    spooling is disabled / run_id is unsafe / the write fails.

    Atomic semantics: write to <name>.tmp then os.replace -> <name>. A crash
    between record_result and write_manifest leaves no spool entry; the
    runner will not retry (record_result is idempotent on run_id), so that
    manifest is lost to Phase 3 -- the runner still has it in its own
    evidence/experiments/ checkout under Phase 2 semantics.
    """
    pending = _pending_dir()
    if pending is None:
        return None
    if not _safe_run_id(run_id):
        sys.stderr.write(
            "[spool] WARN refusing unsafe run_id %r\n" % (run_id,))
        return None
    try:
        pending.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        sys.stderr.write("[spool] WARN cannot mkdir %s: %r\n" % (pending, exc))
        return None

    manifest_path = pending / ("%s.json" % run_id)
    meta_path = pending / ("%s.meta.json" % run_id)
    tmp_manifest = pending / ("%s.json.tmp" % run_id)
    tmp_meta = pending / ("%s.meta.json.tmp" % run_id)

    meta = {
        "run_id": run_id,
        "received_at": received_at,
        "sha256": sha256_hex,
        "manifest_relpath": manifest_relpath,
        "manifest_bytes": len(raw),
    }
    try:
        with open(tmp_manifest, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        # Rename meta first so a partial state never claims "manifest ready"
        # without metadata. Reader checks both files exist.
        os.replace(tmp_meta, meta_path)
        os.replace(tmp_manifest, manifest_path)
        return manifest_path
    except OSError as exc:
        sys.stderr.write(
            "[spool] WARN write failed for %s: %r\n" % (run_id, exc))
        for p in (tmp_manifest, tmp_meta):
            try:
                p.unlink()
            except OSError:
                pass
        return None


def count_pending() -> Optional[int]:
    """Return the number of manifests waiting in the spool.

    Returns None when COORDINATOR_SPOOL_DIR is unset (spooling disabled).
    """
    if _pending_dir() is None:
        return None
    return sum(1 for _ in list_pending_run_ids())


def list_pending_run_ids() -> Iterator[str]:
    """Yield run_ids that have both manifest+meta files in the spool.

    Skips orphaned .tmp files and partial states where only one of the two
    files is present (atomicity-preserving). Returns nothing when spooling
    is disabled.
    """
    pending = _pending_dir()
    if pending is None or not pending.is_dir():
        return
    for entry in sorted(pending.iterdir()):
        name = entry.name
        if not name.endswith(".meta.json"):
            continue
        run_id = name[: -len(".meta.json")]
        if not _safe_run_id(run_id):
            continue
        if not (pending / ("%s.json" % run_id)).is_file():
            continue
        yield run_id


def read_manifest(run_id: str) -> Optional[bytes]:
    pending = _pending_dir()
    if pending is None or not _safe_run_id(run_id):
        return None
    path = pending / ("%s.json" % run_id)
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except OSError:
        return None


def read_meta(run_id: str) -> Optional[dict]:
    pending = _pending_dir()
    if pending is None or not _safe_run_id(run_id):
        return None
    path = pending / ("%s.meta.json" % run_id)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def delete_manifest(run_id: str) -> bool:
    """Remove both manifest and meta files for run_id. Returns True iff
    both deletions succeeded (or the files were already gone)."""
    pending = _pending_dir()
    if pending is None or not _safe_run_id(run_id):
        return False
    ok = True
    for suffix in (".json", ".meta.json"):
        path = pending / ("%s%s" % (run_id, suffix))
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            sys.stderr.write(
                "[spool] WARN delete failed for %s: %r\n" % (path, exc))
            ok = False
    return ok


def derive_evidence_relpath(run_id: str, manifest: dict) -> str:
    """Return the REE_assembly-relative path to commit a manifest at.

    Preference order:
      1. Explicit manifest_relpath in the meta sidecar (set by the runner).
         The writer passes meta["manifest_relpath"] in as `manifest` here
         when available -- callers should consult meta first.
      2. manifest["manifest_relpath"] field embedded in the manifest body
         (forward-compat: a runner can include the hint inside the body
         without changing the POST schema).
      3. DEFAULT_EVIDENCE_PREFIX / <run_id>.json -- coarse but correct;
         the run_id usually already encodes experiment_type + timestamp.

    Always returns a path INSIDE evidence/experiments/ to make a hostile
    manifest unable to write to e.g. .git/ or scripts/.
    """
    candidate = manifest.get("manifest_relpath") if isinstance(
        manifest, dict) else None
    if isinstance(candidate, str) and candidate:
        candidate = candidate.lstrip("/").replace("\\", "/")
        # Refuse traversal. The spool writer commits relative to
        # REE_assembly root, so anything outside evidence/experiments/ is a
        # boundary violation.
        if (".." in candidate.split("/")
                or not candidate.startswith(DEFAULT_EVIDENCE_PREFIX + "/")):
            sys.stderr.write(
                "[spool] WARN ignoring suspicious manifest_relpath %r; "
                "falling back to default\n" % candidate)
        else:
            return candidate
    if not _safe_run_id(run_id):
        # Should never happen at this layer -- list_pending already filtered
        # -- but defence-in-depth: refuse rather than synthesise a path.
        raise ValueError("unsafe run_id %r" % run_id)
    return "%s/%s.json" % (DEFAULT_EVIDENCE_PREFIX, run_id)
