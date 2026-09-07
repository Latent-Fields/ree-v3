"""Contract for `use_before_def` -- a local name read above the line that binds it.

THE INCIDENT. V3-EXQ-591g ran 5h46m on ree-cloud-2 (2026-09-02 14:31Z-20:17Z),
completed all 20 cells, then died in its aggregation block with
`UnboundLocalError: local variable 'dropout_bypassed' referenced before assignment`
at line 590, where the guard is first bound at line 597. The crash preceded the
manifest write, so the whole run was lost. The fix (V3-EXQ-591h, ree-v3 32004c8)
was a pure statement reordering.

WHY THIS NEEDED A NEW GATE. Every existing gate passes 591g clean, and the
mandatory --dry-run smoke test does so STRUCTURALLY, not by luck: the crashing
expression is `gate_green and not dropout_bypassed and ...`, and `gate_green` is
the leading conjunct, always False under --dry-run. Python short-circuits and
never evaluates the undefined name. Any guard consumed inside a boolean
expression behind a precondition conjunct is invisible to dry-run smoke testing.

TWO SEVERITIES, and the split is the load-bearing design decision. A finding is
PROVABLE when no loop encloses both the read and the binding (the 591g shape);
it is LOOP-CARRIED when one does, because a previous iteration may legitimately
bind the name. Only the provable bucket hardens, and only under `--paths`. A
corpus scan at authoring time (2026-09-02 incident, scan 2026-09-07 over 1437
drivers) returned exactly ONE provable finding -- 591g itself -- and five
loop-carried ones, so hardening blocks nothing historical.

THE VACUITY TRAP, pinned by `test_module_exclusion_does_not_swallow_function_locals`.
The check excludes module-level names. Collecting them with `ast.walk(tree)`
descends into top-level function BODIES and adds every function LOCAL to the
exclusion set -- after which the check reports every file in the corpus clean,
including 591g, while still "passing". That is a silent, total loss of the
check's value, and it is what the first prototype of this lint actually did.
"""
import ast
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# The incident and its fix. Both are real committed files in the corpus, asserted
# by NAME rather than by a count, and with the exact lines pinned.
POSITIVE = "v3_exq_591g_isef005_phase01_gate_live_v3.py"
POSITIVE_NAME = "dropout_bypassed"
POSITIVE_READ_LINE = 590
POSITIVE_BIND_LINE = 597
NEGATIVE = "v3_exq_591h_isef005_phase01_gate_live_v3.py"


def _lint(src: str, tmp_path: Path, name: str = "drv.py"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(src), encoding="utf-8")
    return V.use_before_def_lint(p)


# --------------------------------------------------------------------------------
# POSITIVE / NEGATIVE controls on the real committed drivers
# --------------------------------------------------------------------------------

def test_flags_the_591g_incident_at_the_exact_lines():
    """The motivating incident must be reported, as PROVABLE, at lines 590 -> 597."""
    path = EXPERIMENTS_DIR / POSITIVE
    assert path.exists(), f"{POSITIVE} missing -- the positive control must stay in the corpus"
    out = V.use_before_def_lint(path)
    assert out is not None, "591g must be flagged -- it is the incident this lint exists for"
    hard = "; ".join(out["hard"])
    assert POSITIVE_NAME in hard, f"expected '{POSITIVE_NAME}' in the PROVABLE bucket, got {out}"
    assert f"line {POSITIVE_READ_LINE}" in hard
    assert f"line {POSITIVE_BIND_LINE}" in hard
    assert "run_experiment()" in hard


def test_591h_the_shipped_fix_is_clean():
    """The fix was a pure reordering, so the successor must report clean."""
    path = EXPERIMENTS_DIR / NEGATIVE
    assert path.exists(), f"{NEGATIVE} missing -- the negative control must stay in the corpus"
    assert V.use_before_def_lint(path) is None


def test_591g_fails_strict_paths_and_591h_passes():
    """End to end through the CLI: this is the shape `precommit_contracts.sh` runs."""
    def run(script):
        return subprocess.run(
            [sys.executable, "validate_experiments.py", "--strict", "--quiet",
             "--checks", "use_before_def", "--paths", f"experiments/{script}"],
            cwd=REPO_ROOT, capture_output=True, text=True)

    bad = run(POSITIVE)
    assert bad.returncode == 1, f"591g must block under --strict --paths; got {bad.returncode}"
    assert POSITIVE_NAME in bad.stdout

    good = run(NEGATIVE)
    assert good.returncode == 0, f"591h must pass; got {good.returncode}\n{good.stdout}"


# --------------------------------------------------------------------------------
# THE VACUITY TRAP -- the check that keeps this lint from silently doing nothing
# --------------------------------------------------------------------------------

def test_module_exclusion_does_not_swallow_function_locals():
    """`_ubd_module_level_names` must return module bindings ONLY.

    If it descends into a top-level def's BODY, every function local joins the
    exclusion set and the lint goes vacuous while still reporting success. This
    asserts the set's CONTENTS, not merely that 591g still fires, so the trap is
    caught at its source rather than through one downstream symptom.
    """
    tree = ast.parse(textwrap.dedent("""
        import os
        MODULE_CONST = 1

        def top_level_fn(param):
            local_binding = 2
            another_local = 3
            return local_binding + another_local + param

        class TopLevelClass:
            class_attr = 4

            def method(self):
                method_local = 5
                return method_local
    """))
    names = V._ubd_module_level_names(tree)
    assert names == {"os", "MODULE_CONST", "top_level_fn", "TopLevelClass"}, names
    for leaked in ("local_binding", "another_local", "method_local", "class_attr",
                   "param", "method"):
        assert leaked not in names, (
            f"'{leaked}' leaked into the module-level exclusion set -- this is the "
            "vacuity trap: the lint would now report every file clean")


def test_lint_is_not_vacuous_on_a_minimal_synthetic_case(tmp_path):
    """A floor under the whole check: the simplest possible instance must fire."""
    out = _lint("""
        def f():
            y = x
            x = 1
            return y
    """, tmp_path)
    assert out is not None and any("'x'" in m for m in out["hard"]), out


# --------------------------------------------------------------------------------
# THE SEVERITY SPLIT
# --------------------------------------------------------------------------------

def test_loop_carried_is_advisory_not_provable(tmp_path):
    """A read and a binding inside one loop -- a previous iteration may bind it."""
    out = _lint("""
        def f(items):
            for item in items:
                if item == prev:
                    pass
                prev = item
    """, tmp_path)
    assert out is not None
    assert out["hard"] == [], f"loop-carried must never enter the blocking bucket: {out}"
    assert any("'prev'" in m for m in out["carried"]), out


def test_the_320_guarded_accumulator_is_loop_carried_only():
    """v3_exq_320's `a_prev` shape: real fragility, not a defect. Must not block.

    It is safe only because a CORRELATED variable (`z_prev is not None`) gates the
    read, which no static scan can prove -- so it is reported, never hardened.
    """
    path = EXPERIMENTS_DIR / "v3_exq_320_sd013_interventional_training.py"
    if not path.exists():
        pytest.skip("v3_exq_320 not present")
    out = V.use_before_def_lint(path)
    assert out is not None
    assert out["hard"] == [], f"320 must not enter the blocking bucket: {out['hard']}"
    assert any("a_prev" in m for m in out["carried"])


def test_provable_shape_outside_a_loop_is_hard(tmp_path):
    """The 591g shape itself: a straight-line read above the binding."""
    out = _lint("""
        def f(seeds):
            for s in seeds:
                pass
            ok = all(seeds)
            verdict = ok and not guard
            guard = compute()
            return verdict
    """, tmp_path)
    assert out is not None
    assert any("'guard'" in m for m in out["hard"]), out


# --------------------------------------------------------------------------------
# FALSE-POSITIVE REGRESSIONS -- each of these was measured, not imagined.
# The first prototype produced ~180 findings of the first shape and ~17 of the
# second across the corpus; both are pinned so a refactor cannot reintroduce them.
# --------------------------------------------------------------------------------

def test_nested_function_parameters_do_not_leak_into_the_parent_scope(tmp_path):
    """~180 spurious findings: a nested helper's params read as parent unbound reads."""
    out = _lint("""
        def outer(env):
            def _probe(ax, ay):
                env.agent_x = ax
                env.agent_y = ay
                return ax + ay
            results = []
            for hx, hy in env.hazards:
                ax, ay = hx, hy
                results.append(_probe(ax, ay))
            return results
    """, tmp_path)
    assert out is None, f"nested-def params must not be judged in the parent scope: {out}"


def test_comprehension_target_nested_in_an_expression_is_not_a_late_binding(tmp_path):
    """~17 spurious findings: a comprehension inside a dict/call read its own target."""
    out = _lint("""
        def write_manifest(result):
            return {
                "per_seed": {
                    k: [{kk: vv for kk, vv in s.items() if kk != "drop"} for s in v]
                    for k, v in result.items()
                },
            }
    """, tmp_path)
    assert out is None, f"comprehension targets bind in their own scope: {out}"


def test_generator_inside_a_comprehension_condition_is_clean(tmp_path):
    """The v3_exq_047j `ph` shape -- a genexp nested in a listcomp's `if`."""
    out = _lint("""
        def f(params, agent):
            standard = [
                p for p in params
                if not any(p is ph for ph in agent.head.parameters())
            ]
            return standard
    """, tmp_path)
    assert out is None, out


def test_closure_over_a_later_binding_is_legal(tmp_path):
    """A nested def may reference a name bound later in the enclosing scope."""
    out = _lint("""
        def outer():
            def inner():
                return cache
            cache = {}
            return inner
    """, tmp_path)
    assert out is None, out


def test_global_and_nonlocal_declarations_are_respected(tmp_path):
    out = _lint("""
        COUNTER = 0

        def bump():
            global COUNTER
            print(COUNTER)
            COUNTER = COUNTER + 1
    """, tmp_path)
    assert out is None, out


def test_module_level_name_shadowed_locally_is_excluded(tmp_path):
    out = _lint("""
        SEEDS = [1, 2]

        def f():
            print(SEEDS)
            SEEDS = [3]
            return SEEDS
    """, tmp_path)
    assert out is None, out


def test_exception_binding_and_walrus_count_as_bindings(tmp_path):
    out = _lint("""
        def f(data):
            try:
                pass
            except ValueError as err:
                print(err)
            if (n := len(data)) > 0:
                return n
            return 0
    """, tmp_path)
    assert out is None, out


def test_try_except_import_binding_is_seen(tmp_path):
    out = _lint("""
        def f():
            import json
            return json.dumps({})
    """, tmp_path)
    assert out is None, out


# --------------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------------

def test_exempt_marker_silences_the_check(tmp_path):
    src = """
        USE_BEFORE_DEF_EXEMPT = "deliberate"

        def f():
            y = x
            x = 1
            return y
    """
    assert _lint(src, tmp_path) is None


def test_unparseable_file_returns_none_rather_than_raising(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("def f(:\n", encoding="utf-8")
    assert V.use_before_def_lint(p) is None


def test_check_is_registered_and_selectable():
    assert "use_before_def" in V.CHECK_NAMES
    proc = subprocess.run(
        [sys.executable, "validate_experiments.py", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert "use_before_def" in proc.stdout


def test_lint_output_is_ascii():
    """CLAUDE.md: anything reaching stdout must be ASCII (cp1252 terminals)."""
    out = V.use_before_def_lint(EXPERIMENTS_DIR / POSITIVE)
    for msg in out["hard"] + out["carried"]:
        msg.encode("ascii")


def test_corpus_provable_backlog_stays_at_the_known_carrier():
    """The blocking bucket must stay empty apart from 591g itself.

    This is what licenses hardening under --paths. If a new provable carrier
    lands, this test is the place that says so -- fix the driver, do not relax
    the bound.
    """
    provable = []
    for path in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py")):
        out = V.use_before_def_lint(path)
        if out and out["hard"]:
            provable.append(path.name)
    assert provable == [POSITIVE], f"unexpected provable use-before-def carriers: {provable}"
