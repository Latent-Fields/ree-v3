"""Contract: no basename collides between ree-v3/scripts/ and
REE_assembly/evidence/experiments/scripts/ (2026-09-17,
chip-20260917-cross-repo-scripts-dir-collision-guard).

Mechanism this guards against: `tests/contracts/test_burned_queue_entry_detector.py`
does an UNGUARDED `sys.path.insert(0, str(REPO_ROOT / "scripts"))` (ree-v3
scripts/) at import time, to reach scripts/audit_burned_queue_entries.py.
`coordinator/test_phase3_runpack_materialize.py` does the same at position 0
for REE_assembly's evidence/experiments/scripts/ (to reach sync_v3_results.py).
Both inserts write sys.path[0] for the WHOLE pytest process, so whichever
module is imported LAST during collection wins -- the outcome is
COLLECTION-ORDER DEPENDENT for any basename both directories share.

Measured 2026-09-17 (before the sync_v3_results.py fossil in ree-v3/scripts
was deleted, ree-v3 commit 55ff977c50):

    pytest tests/contracts/test_burned_queue_entry_detector.py \
           coordinator/test_phase3_runpack_materialize.py
      -> live REE_assembly module wins  -> 1 failed (pre-existing, unrelated), 26 passed
    pytest coordinator/test_phase3_runpack_materialize.py \
           tests/contracts/test_burned_queue_entry_detector.py
      -> ree-v3/scripts module wins     -> 13 failed, 14 passed

remote_pytest.sh's DEFAULT_PYTEST_ARGS lists tests/ before coordinator/, so
the full suite always lands on the safe order -- but any TARGETED invocation
(a single file, a --changed subset, an IDE test-runner) can flip it, and in
the losing order the coordinator's runpack materialisation ALSO degrades
silently by design ("AttributeError: module 'sync_v3_results' has no
attribute 'runpack_for_flat'; committing flat manifest(s) only") -- i.e. a
shadowing collision here can look like a partial evidence outage rather than
an import bug.

The 2026-09-17 deletion removed the only basename BOTH directories shared, so
today the overlap set is empty and the unguarded inserts are harmless. This
test pins that invariant so the NEXT .py file added to ree-v3/scripts/ whose
basename already exists in REE_assembly/evidence/experiments/scripts/ fails
loudly here, instead of silently re-arming the collection-order hazard above.

REE_assembly resolution deliberately duplicates (does not import)
coordinator/test_phase3_runpack_materialize.py's `_resolve_ree_assembly`
logic -- same behaviour (REE_ASSEMBLY_ROOT env var wins, else the
REE_Working sibling), kept local so this test does not itself pull
coordinator/ onto sys.path (which would be a second instance of exactly the
hazard this file exists to catch). Skips cleanly when no REE_assembly
checkout is found, matching that module's pattern -- this must also behave
on the hub and cloud workers (checkout at /home/ree/REE_Working there, not
/Users/dgolden).

ASCII-only. Run: pytest tests/contracts/test_cross_repo_scripts_dir_collision.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
REE_V3_SCRIPTS = REPO_ROOT / "scripts"


def _resolve_ree_assembly() -> Path:
    """REE_ASSEMBLY_ROOT env var wins; else the REE_Working sibling.

    Mirrors coordinator/test_phase3_runpack_materialize.py's
    `_resolve_ree_assembly` exactly (env var override, else
    <ree-v3>/../REE_assembly) -- duplicated rather than imported, see the
    module docstring.
    """
    env = os.environ.get("REE_ASSEMBLY_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return REPO_ROOT.parent / "REE_assembly"


REE_ASSEMBLY = _resolve_ree_assembly()
ASSEMBLY_SCRIPTS = REE_ASSEMBLY / "evidence" / "experiments" / "scripts"

_NEEDS_ASSEMBLY = (
    "no REE_assembly checkout with evidence/experiments/scripts/ at %s -- "
    "run from the REE_Working layout, or set REE_ASSEMBLY_ROOT" % REE_ASSEMBLY)


def _py_basenames(directory: Path) -> set:
    return {p.name for p in directory.glob("*.py")}


@pytest.mark.skipif(not ASSEMBLY_SCRIPTS.is_dir(), reason=_NEEDS_ASSEMBLY)
def test_no_basename_collision_between_scripts_dirs():
    """The two script directories that different contract tests each insert
    at sys.path[0] must never share a basename -- a shared basename makes
    which module wins COLLECTION-ORDER DEPENDENT (see module docstring for
    the measured before/after counts). This is the mechanical guard so the
    next file added to either directory that re-arms the collision fails
    here instead of surfacing later as an order-dependent test failure or a
    silently-degraded coordinator runpack materialisation."""
    assert REE_V3_SCRIPTS.is_dir(), (
        "expected ree-v3/scripts/ to exist at %s" % REE_V3_SCRIPTS)

    v3_names = _py_basenames(REE_V3_SCRIPTS)
    assembly_names = _py_basenames(ASSEMBLY_SCRIPTS)
    collisions = sorted(v3_names & assembly_names)

    assert not collisions, (
        "basename collision between ree-v3/scripts/ and "
        "REE_assembly/evidence/experiments/scripts/: %s -- two different "
        "contract tests each do an unguarded sys.path.insert(0, ...) for "
        "these directories (test_burned_queue_entry_detector.py for "
        "ree-v3/scripts/, test_phase3_runpack_materialize.py for the "
        "REE_assembly one), so whichever module is imported LAST during "
        "pytest collection silently shadows the other -- collection-order "
        "dependent, no error. Rename one of the colliding files, or route "
        "the newer one through importlib.util.spec_from_file_location "
        "instead of adding it to either sys.path insert." % collisions)
