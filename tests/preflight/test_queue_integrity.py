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
    """Any `supersedes` field should be a queue_id-shaped string.

    A QUEUE entry supersedes a QUEUE_ID ("V3-EXQ-826"); a run MANIFEST
    supersedes a RUN_ID ("v3_exq_826_..._v3"). Same field name, two
    artifacts -- see CLAUDE.md "Supersession and evidence validity". Writing
    the run_id here is the recurring slip.

    This asserts over the LIVE queue, which is DB-authoritative and mutates
    continuously, so a failure is transient data, not a code regression: it
    clears by itself once the offending item leaves the queue, which makes it
    easy to mistake for flake. Confirmed 2026-07-27: V3-EXQ-826a (611e8a9)
    carried a run_id and reddened trunk until the item completed out.
    `validate_queue.py` now WARNs on the same shape at commit time, which is
    where it can actually be fixed -- if this test fires, that warning was
    ignored or bypassed.
    """
    queue_path = REPO_ROOT / "experiment_queue.json"
    data = json.loads(queue_path.read_text())
    bad = []
    for item in data.get("items", []):
        sup = item.get("supersedes")
        if sup is None:
            continue
        if not RE_QUEUE_ID.match(sup):
            hint = ""
            m = re.match(r"^v(\d+)_exq_(\d+[a-z]?)_", sup)
            if m:
                hint = (f"  -> looks like a run_id; use "
                        f"'V{m.group(1)}-EXQ-{m.group(2).upper()}' here and "
                        f"put the run_id in the manifest instead")
            bad.append(f"{item.get('queue_id')}: supersedes='{sup}'{hint}")
    assert not bad, (
        "malformed supersedes targets (queue entries take a queue_id; the "
        "run_id form belongs in the manifest):\n  " + "\n  ".join(bad))
