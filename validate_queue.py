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
import sys
from pathlib import Path

QUEUE_FILE = Path(__file__).resolve().parent / "experiment_queue.json"

# ------------------------------------------------------------------
# Valid values for enum fields
# ------------------------------------------------------------------
VALID_STATUSES = {"pending", "claimed", "failed"}
VALID_AFFINITIES = {"any", "DLAPTOP-4.local", "Daniel-PC", "EWIN-PC"}

# queue_id must match: V3-EXQ-<digits>[optional letter][optional -<letter>]
# OR onboarding smoke tests: V3-ONBOARD-smoke-<machine-name>
# Examples: V3-EXQ-047, V3-EXQ-047j, V3-EXQ-001-a, V3-ONBOARD-smoke-EWIN-PC
RE_QUEUE_ID = re.compile(r"^V3-EXQ-\d+[a-z]?(-[a-z])?$|^V3-ONBOARD-smoke-.+$")

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
    ("supersedes", False, str),
    ("claimed_by", False, (dict, type(None))),
    ("machine_affinity_note", False, str),
]


def _type_name(t) -> str:
    if isinstance(t, tuple):
        return "/".join(x.__name__ for x in t if x is not type(None))
    return t.__name__


def validate(queue_path: Path = QUEUE_FILE) -> list[str]:
    """
    Validate the queue file.  Returns a list of error strings.
    Empty list means the queue is valid.
    """
    errors: list[str] = []

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

        # script field -- warn (not error) if the file does not exist
        script_val = item.get("script")
        if isinstance(script_val, str):
            script_path = queue_path.parent / script_val
            if not script_path.exists():
                errors.append(
                    f"{prefix}: script file not found: {script_val}"
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
    if errors:
        print(f"Queue validation FAILED -- {len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Queue OK -- {QUEUE_FILE.name} is valid.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
