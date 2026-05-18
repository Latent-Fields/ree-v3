"""P1: runner preflight.

Checks that the experiment runner can import cleanly and that every script
referenced from the queue actually exists. The script-existence check is
also performed by validate_queue.py as an error, but this test gives the
failure a dedicated name and lets the regression suite wrapper run the
check without re-parsing the queue.
"""

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_experiment_runner_imports():
    """experiment_runner.py must import without side effects."""
    mod = importlib.import_module("experiment_runner")
    assert hasattr(mod, "main"), "experiment_runner.main missing"
    assert hasattr(mod, "load_queue"), "experiment_runner.load_queue missing"


def test_validate_queue_imports():
    mod = importlib.import_module("validate_queue")
    assert callable(getattr(mod, "validate", None))


def test_every_queued_script_exists():
    queue_path = REPO_ROOT / "experiment_queue.json"
    data = json.loads(queue_path.read_text())
    missing = []
    for item in data.get("items", []):
        script = item.get("script")
        if not isinstance(script, str):
            continue
        if not (REPO_ROOT / script).exists():
            missing.append(f"{item.get('queue_id')}: {script}")
    assert not missing, "queued scripts missing on disk:\n  " + "\n  ".join(missing)
