"""P2: queue integrity preflight.

Thin wrapper around validate_queue.validate(). The runner already calls the
same validator at startup; this test exists so `pytest tests/preflight` is a
complete preflight gate and so the suite wrapper can run the same check
without shelling out.

Also checks `supersedes` references look well-formed (the validator only
type-checks them).
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from validate_queue import validate, RE_QUEUE_ID  # noqa: E402

import json  # noqa: E402


def test_queue_schema_valid():
    errors = validate()
    assert errors == [], "queue schema errors:\n  " + "\n  ".join(errors)


def test_supersedes_targets_well_formed():
    """Any `supersedes` field should be a queue_id-shaped string."""
    queue_path = REPO_ROOT / "experiment_queue.json"
    data = json.loads(queue_path.read_text())
    bad = []
    for item in data.get("items", []):
        sup = item.get("supersedes")
        if sup is None:
            continue
        if not RE_QUEUE_ID.match(sup):
            bad.append(f"{item.get('queue_id')}: supersedes='{sup}'")
    assert not bad, "malformed supersedes targets:\n  " + "\n  ".join(bad)
