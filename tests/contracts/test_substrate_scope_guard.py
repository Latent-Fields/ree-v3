"""Contracts for the shared, scope-generic conservatism guards (plan sec 11).

The guard machinery was promoted out of maturation_curriculum.py into
experiments/_lib/substrate_scope_guard.py so ANY experiment can prove a declared
substrate scope is a safe over-approximation before passing it to
compute_arm_fingerprint(substrate_scope=...). This suite tests the promoted module
standalone (stdlib-only, no cell execution):

  - expand_scope resolves exact paths + wildcard globs to concrete files.
  - static_data_closure is a fixpoint on a known-good (data-closed) scope.
  - verify_scope_static PASSES a data-closed scope and RAISES loudly on one that
    value-imports a module-level constant from an UNSCOPED module (guard 2 -- the
    false-HIT tripwire).
  - traced_execution_files + verify_scope_conservatism guard 1: an under-approximating
    scope (a file executes that is not declared) RAISES; a covering scope passes.

Uses the maturation world/harm scopes as known-good declared closures.

Design plan: REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md sec 11.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
for _p in (str(REPO_ROOT), str(REPO_ROOT / "experiments"), str(REPO_ROOT / "experiments" / "_lib")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import substrate_scope_guard as G  # noqa: E402
from _lib.baselines import maturation_curriculum as MC  # noqa: E402
from _lib.arm_fingerprint import machine_class  # noqa: E402

WORLD = MC._WORLD_SUBSTRATE_SCOPE
HARM = MC._HARM_SUBSTRATE_SCOPE


def test_expand_scope_exact_and_glob():
    """Exact paths map to themselves; a wildcard glob expands to its matches."""
    exact = G.expand_scope(["ree_core/latent/stack.py"])
    assert exact == ["ree_core/latent/stack.py"]
    glob = G.expand_scope(["ree_core/latent/*.py"])
    assert "ree_core/latent/stack.py" in glob
    assert len(glob) > 1  # more than one file under ree_core/latent


def test_static_data_closure_fixpoint():
    """The known-good leg scopes are data-closed fixpoints (closure adds nothing)."""
    for scope in (WORLD, HARM):
        files = set(G.expand_scope(scope))
        assert G.static_data_closure(files) == files


def test_verify_scope_static_passes_known_good():
    """Guard 2 passes the world + harm scopes and returns the expanded file set."""
    fw = G.verify_scope_static(WORLD, label="world")
    fh = G.verify_scope_static(HARM, label="harm")
    assert len(fw) == 24
    assert len(fh) == 19
    assert fh <= fw


def test_verify_scope_static_catches_data_closure_escape():
    """Guard 2 tripwire: dropping the regulators SITE_* leaf (a value-import channel a
    scope file reads a constant from) makes the scope non-data-closed -> loud raise."""
    bad = tuple(x for x in WORLD if "regulators" not in x)
    raised = False
    try:
        G.verify_scope_static(bad, label="world-minus-regulators")
    except AssertionError as exc:
        raised = True
        assert "data-closed" in str(exc)
    assert raised


def test_verify_scope_static_catches_missing_file():
    """An exact declared file that does not exist (refactor drift) raises loudly."""
    raised = False
    try:
        G.verify_scope_static(["ree_core/does_not_exist_xyz.py"], label="drift")
    except AssertionError as exc:
        raised = True
        assert "do not exist" in str(exc)
    assert raised


def test_traced_execution_files_captures_repo_frames():
    """Guard 1 primitive: traced_execution_files records the repo files whose code ran."""
    executed = G.traced_execution_files(machine_class)
    assert "experiments/_lib/arm_fingerprint.py" in executed


def test_trace_guard_catches_under_approximation():
    """Guard 1: a (data-closed) scope that omits a file which actually EXECUTES raises.
    WORLD passes guard 2; run_once executes substrate_scope_guard.py (NOT in WORLD via
    expand_scope) so guard 1 fires the under-approximation tripwire."""
    def run_once():
        G.expand_scope(["ree_core/latent/stack.py"])  # runs substrate_scope_guard.py

    assert "experiments/_lib/substrate_scope_guard.py" not in set(WORLD)
    raised = False
    try:
        G.verify_scope_conservatism(WORLD, run_once, label="world-missing-guard")
    except AssertionError as exc:
        raised = True
        assert "EXECUTED" in str(exc)
    assert raised


def test_trace_guard_passes_when_scope_covers_execution():
    """Guard 1 passes when the scope covers every executed repo file. The trace captures
    ANY repo file whose code runs -- including this test module itself, where run_once is
    defined (in real use the cell driver lives in the experiment script) -- so the covering
    scope must name it, the guard module, and WORLD. All stay data-closed (each added
    file's constant-imports already resolve inside the scope)."""
    here = str(Path(__file__).resolve().relative_to(REPO_ROOT))

    def run_once():
        machine_class()                                # runs arm_fingerprint.py (in WORLD)
        G.expand_scope(["ree_core/latent/stack.py"])   # runs substrate_scope_guard.py

    scope = list(WORLD) + ["experiments/_lib/substrate_scope_guard.py", here]
    rep = G.verify_scope_conservatism(scope, run_once, label="covering")
    assert rep["static_guard"] == "ok"
    assert rep["trace_guard"] == "ok"
    assert rep["n_executed"] >= 2


def test_conservatism_static_only_report():
    """With run_once=None only guard 2 runs; report carries the static result."""
    rep = G.verify_scope_conservatism(WORLD, label="world")
    assert rep["static_guard"] == "ok"
    assert "trace_guard" not in rep
    assert rep["n_declared_files"] == 24
