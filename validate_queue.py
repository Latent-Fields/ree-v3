#!/usr/bin/env python3
"""
validate_queue.py -- Validate experiment_queue.json against the expected schema.

Called automatically by experiment_runner.py at startup.
Also run manually after editing the queue:
    /opt/local/bin/python3 validate_queue.py

Exit codes:
    0  -- queue is valid
    1  -- one or more validation errors (details printed to stderr)
    2  -- queue file missing or unparseable JSON
"""

import json
import re
import subprocess
import sys
from pathlib import Path

QUEUE_FILE = Path(__file__).resolve().parent / "experiment_queue.json"

# ------------------------------------------------------------------
# Valid values for enum fields
# ------------------------------------------------------------------
VALID_STATUSES = {"pending", "claimed", "failed", "suspended"}
VALID_AFFINITIES = {"any", "DLAPTOP-4.local", "Daniel-PC", "EWIN-PC", "ree-cloud-1", "ree-cloud-2", "ree-cloud-3", "ree-cloud-4"}

# queue_id must match: V3-EXQ-<digits>[optional letter][optional -<letter>]
# OR onboarding smoke tests: V3-ONBOARD-smoke-<machine-name>
# Examples: V3-EXQ-047, V3-EXQ-047j, V3-EXQ-001-a, V3-ONBOARD-smoke-EWIN-PC
RE_QUEUE_ID = re.compile(r"^V3-EXQ-\d+[a-z]?(-[a-z])?$|^V3-ONBOARD-smoke-.+$")

# Contract with experiment_runner.py RE_SAVED_TO (line 73).
# If a script writes a manifest under evidence/experiments, it MUST print
# 'Result written to: <path>' on stdout so the runner captures output_file.
# Without this, runner_status.output_file stays empty and generate_pending_review
# cannot derive the on-disk dir_name -- the manifest exists but appears undiscussed.
# Historical incident: V3-EXQ-325b/325c (2026-04-18/19) used 'Results -> {path}'.
RE_SAVED_TO_IN_SCRIPT = re.compile(r"Result (?:pack )?written to")

# Regression guard against the 2026-05-29 emit_outcome copy-paste bug.
# emit_outcome() in experiment_protocol.py accepts only: outcome, manifest_path,
# run_id, queue_id, exit_reason, extra, signal_dir. Six scripts copy-pasted
# extra kwargs (experiment_type, claim_ids, evidence_direction, results,
# metrics, experiment_purpose, architecture_epoch) that the function has
# never accepted -- the call crashes with TypeError AFTER the manifest is
# written, so the runner classifies ERROR and the Phase 3 writer never
# ingests the result. Canonical incident: V3-EXQ-610a 2026-05-29T22:44Z
# on ree-cloud-3 (manifest preserved as .bak.20260530, rescued via the
# 2026-05-30 sweep). Pattern is permissive of valid kwargs but flags the
# disallowed ones.
RE_EMIT_OUTCOME_CALL = re.compile(r"emit_outcome\s*\([^)]*\)", re.DOTALL)
EMIT_OUTCOME_DISALLOWED_KWARGS = (
    "experiment_type",
    "claim_ids",
    "evidence_direction",
    "results",
    "metrics",
    "experiment_purpose",
    "architecture_epoch",
)

# ------------------------------------------------------------------
# Field specs: (field_name, required, expected_type_or_None_for_any)
# ------------------------------------------------------------------
TOP_LEVEL_REQUIRED = [
    ("schema_version", True, str),
    ("calibration", True, dict),
    ("items", True, list),
]

ITEM_REQUIRED = [
    ("queue_id", True, str),
    ("script", True, str),
    ("priority", True, int),
    ("machine_affinity", True, str),
    ("status", True, str),
    ("estimated_minutes", True, (int, float)),
]

ITEM_OPTIONAL = [
    ("note", False, str),
    ("title", False, str),
    ("backlog_id", False, str),
    ("claim_id", False, str),
    ("claim_ids", False, list),
    ("supersedes", False, str),
    ("claimed_by", False, (dict, type(None))),
    ("machine_affinity_note", False, str),
    ("force_rerun", False, bool),
    ("experiment_type", False, str),
    ("checkpoint_resumable", False, bool),
    ("checkpoint_experiment_type", False, str),
    ("checkpoint_path", False, str),
    ("suspended_at", False, str),
]


# ------------------------------------------------------------------
# Per-machine runner_status scan (silent re-queue guard)
# ------------------------------------------------------------------
# Historical incidents (canonical: EXQ-126 on 2026-04-20/21) showed that a
# previously-run queue_id can be re-added to the queue and silently re-executed
# when its original completion record is not present in the local per-machine
# status file -- e.g. the completion was recorded under a prior hostname
# (Mac -> DLAPTOP-4.local), or on a different machine whose status file is
# offline. The runner only checks the local per-machine file + any peer files
# it can see at startup. If none of those contain the queue_id, dedup silently
# passes and the experiment runs again.
#
# This guard scans every per-machine runner_status file in REE_assembly and
# raises a validation error on any queue_id that already has a completion
# record, unless the queue item carries force_rerun: true. New letter/number
# suffix IDs (EXQ-126a, EXQ-127) are the normal path; force_rerun is the
# explicit escape hatch for the rare case where re-using the same ID is
# intentional (e.g. the prior record is from a superseded contamination epoch).

_REE_ASSEMBLY_STATUS_DIR_CANDIDATES = [
    QUEUE_FILE.parent.parent / "REE_assembly" / "evidence" / "experiments" / "runner_status",
    Path.home() / "REE_Working" / "REE_assembly" / "evidence" / "experiments" / "runner_status",
]


def _find_status_dir() -> Path | None:
    for cand in _REE_ASSEMBLY_STATUS_DIR_CANDIDATES:
        if cand.is_dir():
            return cand
    return None


def _scan_completed_queue_ids() -> dict[str, list[tuple[str, str, str]]]:
    """Scan per-machine runner_status files for completed queue_ids.

    Returns a dict mapping queue_id -> list of (machine_file, result, completed_at).
    Returns an empty dict (fail-soft) if the status dir is missing or unreadable.
    """
    status_dir = _find_status_dir()
    if status_dir is None:
        return {}
    out: dict[str, list[tuple[str, str, str]]] = {}
    for f in sorted(status_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        for entry in data.get("completed", []) or []:
            qid = entry.get("queue_id", "")
            if not qid:
                continue
            out.setdefault(qid, []).append(
                (f.name, entry.get("result", "?"), entry.get("completed_at", ""))
            )
    return out


def _type_name(t) -> str:
    if isinstance(t, tuple):
        return "/".join(x.__name__ for x in t if x is not type(None))
    return t.__name__


def _is_tracked(repo_root: Path, rel_path: str) -> bool:
    # Returns True iff `rel_path` is tracked in the git index at repo_root.
    # If git is unavailable (no .git, no git binary), we fail-open: return
    # True so non-git checkouts (e.g. fresh tarball extractions in CI sanity
    # checks) do not spuriously fail validation. The real production path
    # always has git.
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", rel_path],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True  # fail-open when git is unavailable
    return result.returncode == 0


def _validate_run_axis(prefix: str, field_name: str, value, elem_type) -> list[str]:
    """Validate seeds/conditions fields which may be counts or explicit lists."""
    errors: list[str] = []
    if isinstance(value, bool):
        return [f"{prefix}: '{field_name}' must be int or list, got bool"]
    if isinstance(value, int):
        if value <= 0:
            errors.append(f"{prefix}: '{field_name}' must be > 0, got {value}")
        return errors
    if isinstance(value, list):
        if not value:
            errors.append(f"{prefix}: '{field_name}' list must not be empty")
            return errors
        for sub_idx, sub_val in enumerate(value):
            if isinstance(sub_val, bool) or not isinstance(sub_val, elem_type):
                errors.append(
                    f"{prefix}: '{field_name}[{sub_idx}]' must be {elem_type.__name__}, "
                    f"got {type(sub_val).__name__}"
                )
                continue
            if elem_type is int and sub_val <= 0:
                errors.append(
                    f"{prefix}: '{field_name}[{sub_idx}]' must be > 0, got {sub_val}"
                )
            if elem_type is str and not sub_val.strip():
                errors.append(
                    f"{prefix}: '{field_name}[{sub_idx}]' must not be empty"
                )
        return errors
    return [f"{prefix}: '{field_name}' must be int or list, got {type(value).__name__}"]


# Non-blocking advisories collected by the most recent validate() call.
# Kept separate from the returned error list so callers (the runner, the
# preflight test) keep their exit-code contract -- warnings never block a
# commit or a runner start. main() prints them; validate() resets + fills it.
_LAST_WARNINGS: list[str] = []


def validate(queue_path: Path = QUEUE_FILE) -> list[str]:
    """
    Validate the queue file.  Returns a list of error strings.
    Empty list means the queue is valid.

    Non-fatal advisories (e.g. an item carrying no claim tag) are appended to
    the module-level ``_LAST_WARNINGS`` list rather than the returned errors,
    so they surface to a human without failing the commit hook.
    """
    errors: list[str] = []
    _LAST_WARNINGS.clear()

    # --- 1. Parse JSON ---
    try:
        raw = queue_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        errors.append(f"Queue file not found: {queue_path}")
        return errors

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"JSON parse error: {exc}")
        return errors

    if not isinstance(data, dict):
        errors.append("Top-level value must be a JSON object.")
        return errors

    # --- 2. Top-level fields ---
    for fname, required, ftype in TOP_LEVEL_REQUIRED:
        val = data.get(fname)
        if val is None:
            if required:
                errors.append(f"Missing required top-level field: '{fname}'")
        elif not isinstance(val, ftype):
            errors.append(
                f"Top-level '{fname}' must be {_type_name(ftype)}, "
                f"got {type(val).__name__}"
            )

    if data.get("schema_version") not in (None, "v1"):
        errors.append(
            f"Unknown schema_version '{data['schema_version']}' -- expected 'v1'"
        )

    items = data.get("items")
    if not isinstance(items, list):
        # Already reported above; bail to avoid cascading noise
        return errors

    # --- 3. Per-item validation ---
    seen_ids: dict[str, int] = {}
    completed_scan = _scan_completed_queue_ids()

    for idx, item in enumerate(items):
        prefix = f"items[{idx}]"

        if not isinstance(item, dict):
            errors.append(f"{prefix}: each item must be a JSON object")
            continue

        queue_id = item.get("queue_id", f"<unknown at index {idx}>")
        prefix = f"items[{idx}] ({queue_id})"

        # Required fields
        for fname, required, ftype in ITEM_REQUIRED:
            val = item.get(fname)
            if val is None:
                if required:
                    errors.append(f"{prefix}: missing required field '{fname}'")
            elif not isinstance(val, ftype):
                errors.append(
                    f"{prefix}: '{fname}' must be {_type_name(ftype)}, "
                    f"got {type(val).__name__}"
                )

        # Optional fields -- type-check if present
        for fname, _required, ftype in ITEM_OPTIONAL:
            val = item.get(fname)
            if val is not None and not isinstance(val, ftype):
                errors.append(
                    f"{prefix}: '{fname}' must be {_type_name(ftype)}, "
                    f"got {type(val).__name__}"
                )

        # queue_id format
        if isinstance(queue_id, str):
            if not RE_QUEUE_ID.match(queue_id):
                errors.append(
                    f"{prefix}: queue_id '{queue_id}' does not match expected pattern "
                    f"V3-EXQ-<digits>[letter][-letter] "
                    f"(e.g. V3-EXQ-047, V3-EXQ-047j, V3-EXQ-001-a)"
                )
            # Duplicate check
            if queue_id in seen_ids:
                errors.append(
                    f"{prefix}: duplicate queue_id '{queue_id}' "
                    f"(first seen at index {seen_ids[queue_id]})"
                )
            else:
                seen_ids[queue_id] = idx

        # machine_affinity enum
        affinity = item.get("machine_affinity")
        if isinstance(affinity, str) and affinity not in VALID_AFFINITIES:
            errors.append(
                f"{prefix}: machine_affinity '{affinity}' is not a recognised value "
                f"({', '.join(sorted(VALID_AFFINITIES))})"
            )

        # status enum
        status = item.get("status")
        if isinstance(status, str) and status not in VALID_STATUSES:
            errors.append(
                f"{prefix}: status '{status}' is not a recognised value "
                f"({', '.join(sorted(VALID_STATUSES))})"
            )

        # estimated_minutes must be > 0
        est = item.get("estimated_minutes")
        if isinstance(est, (int, float)) and est <= 0:
            errors.append(
                f"{prefix}: estimated_minutes must be > 0, got {est}"
            )

        # claim-tag presence (WARN, non-blocking). Every queue item should declare
        # either claim_id / a non-empty claim_ids (the claim(s) it tests) OR an
        # explicit claim_ids: [] marking an intentional claimless diagnostic.
        # Items declaring NEITHER evade per-claim governance trend tracking -- the
        # 39-untagged-ERROR class flagged in insights_report.md 2026-06-06. An
        # explicit claim_ids: [] silences this (use it for substrate-readiness /
        # ablation runs that legitimately map to no claims.yaml id).
        _cid = item.get("claim_id")
        _has_claim_id = isinstance(_cid, str) and _cid.strip()
        _cids = item.get("claim_ids")
        _has_claim_ids = isinstance(_cids, list) and len(_cids) > 0
        _declares_claimless = "claim_ids" in item and isinstance(_cids, list)
        if not _has_claim_id and not _has_claim_ids and not _declares_claimless:
            _LAST_WARNINGS.append(
                f"{prefix}: no claim tag -- set claim_id / claim_ids to the "
                f"claim(s) under test, or claim_ids: [] to mark an intentional "
                f"claimless diagnostic."
            )

        if "seeds" in item:
            errors.extend(_validate_run_axis(prefix, "seeds", item["seeds"], int))
        if "conditions" in item:
            errors.extend(_validate_run_axis(prefix, "conditions", item["conditions"], str))

        # script field -- require the file to be both on disk AND tracked in git.
        # An untracked-but-on-disk script passes Path.exists() in the producer's
        # checkout but fails on every consumer that pulls the queue (the 2026-05-27
        # ree-cloud-1/2/3 fleet wedge: V3-EXQ-610 queue entry committed without its
        # script via a parallel-session in-file edit; producer's validate passed
        # because the untracked script existed on disk locally; every cloud worker
        # crashed at startup). git ls-files is the authoritative check.
        script_val = item.get("script")
        if isinstance(script_val, str):
            script_path = queue_path.parent / script_val
            if not script_path.exists():
                errors.append(
                    f"{prefix}: script file not found on disk: {script_val}"
                )
            elif not _is_tracked(queue_path.parent, script_val):
                errors.append(
                    f"{prefix}: script file exists on disk but is not tracked in "
                    f"git (untracked or ignored): {script_val}. Run `git add "
                    f"{script_val}` in the same commit as the queue entry to "
                    f"prevent pulling consumers from crashing at validate startup."
                )
            else:
                # Manifest-writing scripts must print the runner-matching save line.
                try:
                    source = script_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    source = ""
                writes_manifest = (
                    "json.dump(" in source and "evidence/experiments" in source
                )
                if writes_manifest and not RE_SAVED_TO_IN_SCRIPT.search(source):
                    errors.append(
                        f"{prefix}: script {script_val} writes a JSON manifest "
                        f"under evidence/experiments but does not print "
                        f"'Result written to: <path>' -- experiment_runner.py "
                        f"RE_SAVED_TO will not capture output_file. "
                        f"Add: print(f\"Result written to: {{out_path}}\", flush=True)"
                    )

                # Regression guard: emit_outcome() must not be called with
                # disallowed kwargs (see RE_EMIT_OUTCOME_CALL above).
                for call_match in RE_EMIT_OUTCOME_CALL.finditer(source):
                    call_text = call_match.group(0)
                    for bad_kwarg in EMIT_OUTCOME_DISALLOWED_KWARGS:
                        if re.search(rf"\b{bad_kwarg}\s*=", call_text):
                            errors.append(
                                f"{prefix}: script {script_val} calls "
                                f"emit_outcome({bad_kwarg}=...) but the "
                                f"function never accepted this kwarg. "
                                f"emit_outcome's signature is "
                                f"(outcome, manifest_path, *, run_id, "
                                f"queue_id, exit_reason, extra, signal_dir). "
                                f"Drop the disallowed kwarg(s); the call "
                                f"will crash AFTER the manifest is written "
                                f"and the runner will mis-classify ERROR "
                                f"(canonical incident: V3-EXQ-610a 2026-05-29)."
                            )
                            break  # one error per call site is enough

        # Silent re-queue guard: queue_id must not already have a completion
        # record in any per-machine runner_status file, unless force_rerun=true.
        if isinstance(queue_id, str) and queue_id in completed_scan:
            if item.get("force_rerun") is not True:
                records = completed_scan[queue_id]
                rec_strs = "; ".join(
                    f"{mfile} ({result} at {cat})" for mfile, result, cat in records
                )
                errors.append(
                    f"{prefix}: queue_id already has a completion record in "
                    f"{rec_strs}. The runner WILL silently skip or re-run under a "
                    f"lost-completion edge case. Use a new letter/number suffix "
                    f"(EXQ-126a, EXQ-127, ...), or set 'force_rerun': true to "
                    f"intentionally re-run under the same ID."
                )

        # claimed_by structure check
        claimed_by = item.get("claimed_by")
        if isinstance(claimed_by, dict):
            for sub in ("machine", "claimed_at"):
                if sub not in claimed_by:
                    errors.append(
                        f"{prefix}: claimed_by missing sub-field '{sub}'"
                    )

    return errors


def main() -> int:
    errors = validate()
    if _LAST_WARNINGS:
        print(f"Queue advisories -- {len(_LAST_WARNINGS)} warning(s):", file=sys.stderr)
        for w in _LAST_WARNINGS:
            print(f"  WARN: {w}", file=sys.stderr)
    if errors:
        print(f"Queue validation FAILED -- {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Queue OK -- {QUEUE_FILE.name} is valid.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
