"""Contracts for the config_slice under-declaration lint.

Surfaces under test:
  (1) validate_experiments.config_slice_under_declaration_lint -- flags a driver that
      mints a CROSS-DRIVER-reusable arm cell (`include_driver_script_in_hash=False`)
      whose `config_slice` omits a module-level numeric constant the cell's own call
      graph reads.
  (2) validate_experiments.py --checks config_slice_declaration -- the selector, and the
      invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under --paths,
      never affects the exit code even under --strict).
  (3) The FINGERPRINT FACT the gate rests on, asserted against
      experiments/_lib/arm_fingerprint.py rather than trusted: it is
      `include_driver_script_in_hash` that decides whether the driver's own content is
      folded into `substrate_hash`.

WHY THIS GATE EXISTS. `arm_reuse_fingerprint_plan.md` section 7b: a `config_slice` that
UNDER-approximates -- omits a parameter the cell's RECORDED READOUTS depend on -- is
exactly a false-cache-HIT bug. The governing asymmetry is that a false MISS only wastes
compute while a false HIT corrupts a scientific conclusion, which is why the fingerprint
is meant to be OVER-inclusive.

THE INTERACTION THAT MAKES IT SHARP. `include_driver_script_in_hash=False` is MANDATORY
for a cross-driver-reusable mint (CLAUDE.md, "Saving a baseline for reuse"): with the
default `True` the driver's content is in the hash, so a consumer's distinct driver can
never match and the cell is not reusable cross-driver at all. But it is exactly that
flag which removes the driver -- and therefore every module-level constant defined in it
-- from the hash. So the more reusable a mint is made, the more load the `config_slice`
has to carry, and the flag-False set is precisely the exposed set.

CONFIRMED INSTANCE (V3-EXQ-798, landed manifest 20260723T081627Z). Its cells record
`learn_pe_by_ssl_bin` / `learn_ssl_bin_counts` / `learn_decay_frac`, all computed against
the module-level `SSL_BIN_EDGES = (2, 5, 12)`. That constant is absent from the slice AND
the script emits with the flag False, so the binning scheme is invisible to the
fingerprint entirely -- a consumer using different edges HITs those cells and silently
reads bin means computed under a different scheme. The successor 798a (ree-v3
`7eeb06449d`) declares it (`"binning"` / `"n_bins"` / `"ssl_bin_edges"`) and is the
reference for the correct shape. `test_confirmed_798_instance_is_the_differential` below
is that pair asserted directly: the lint must name SSL_BIN_EDGES on 798 and must NOT
name it on 798a.

CALL-GRAPH SCOPING IS THE DISCRIMINATOR. Scoping the constant scan to the cell's own
call graph (the cell body plus every module-level function transitively called from
it) is what separates a readout-affecting parameter from an adjudication threshold
applied downstream in `evaluate(rows)`. Measured while this gate was written: without
the scoping the check fires on 67 of the 83 cross-driver drivers, dominated by
`THRESH_*` / `FLOOR_*` criterion constants that no cell ever reads; with it, 48. A
whole-module constant scan over-counts by ~40%, which is why this is a call-graph gate
and not a name scan.

BOTH CALL FORMS ARE SCOPED (extended 2026-07-28). The gate originally located the cell
body only via `with arm_cell(...) as cell:`, so the 15 cross-driver scripts calling
`compute_arm_fingerprint(...)` directly returned early and were scanned not at all --
unexamined, not cleared. The direct-call form has no lexical cell body, so its scope is
the NEAREST ENCLOSING LOOP of the fingerprint call, which is where the sibling statement
that actually runs the arm lives. `test_direct_call_form_is_scoped` and
`test_direct_call_scope_is_the_loop_not_the_enclosing_function` are that pair.

WHY NOT THE ENCLOSING FUNCTION -- the measurement that decided it, and the reason not to
"simplify" the loop walk away. In the direct-call corpus the enclosing function is
`run_experiment` in 13 of 15 cases, and `run_experiment` calls `evaluate(rows)` too, so
enclosing-function scope collapses into very nearly the whole-module scan this gate was
built to avoid. Calibrated against the with-form scripts, where the `with` body IS ground
truth: enclosing-function scope reproduces the correct missing set in only 28 of 60, adds
271 spurious constants, and is byte-identical to a whole-module scan in 18 of 60; loop
scope reproduces it in 40 of 60, adds 79, degenerates in 7, and changes the result of
ZERO with-form scripts.

TWO DEFECTS, NOT ONE. Scoping was only half the gap. All 15 direct-call scripts build
their slice through a HELPER rather than a literal dict, and the resolver did not follow
one: 10 via a local in-file `_arm_config_slice()`, 4 via a helper imported from
`experiments/_lib/baselines/`. Without following the local helper the slice reads as
empty and the new fires would have been dominated by constants that ARE declared one call
away -- so `_absorb` now absorbs a local function's `return` values. That also removes
PRE-EXISTING false positives from 13 with-form scripts (V3-EXQ-793 41 names -> 15, 794
11 -> 2, 751 3 -> 0), which is why the pinned count moved DOWN for the with-form half
while the direct-call half was added.

THE CROSS-MODULE BAND IS A DOCUMENTED SKIP. For the 4 whose helper is imported from
`_lib/baselines/`, the declaration is in another file and a single-file scan cannot see
one key of it. Firing there would be a near-total false positive landing on precisely the
scripts that FOLLOW the canonical-baseline pattern CLAUDE.md mandates -- measured on
V3-EXQ-700c, 21 of the 27 constants a fire would name are declared in that module's
`MATCHED_ENVELOPE`. So the gate says nothing for them. `test_cross_module_slice_is_a_skip`
pins that, and it is the one band the gate knowingly cannot assess.

SCOPE. This gates NEW scripts, and the gate stays WARN-only: a landed carrier's run is
complete and its pre-registered emission is not rewritten to chase a lint. A false HIT
also needs a CONSUMER to exist before it can corrupt anything, so the backlog is a
standing risk register rather than a live fire.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402


def _write(tmp_path: Path, body: str, name: str = "v3_exq_999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _named(msg: str) -> set:
    """The constants the message actually NAMES, i.e. only the `reads: ...` list.

    Not a plain substring test on the whole message: the remediation boilerplate cites
    the confirmed instance by name ("...V3-EXQ-798 (SSL_BIN_EDGES absent...)"), so
    `"SSL_BIN_EDGES" in msg` is TRUE for every fire in the corpus and would make the
    798/798a differential below vacuously pass. Caught by that test failing on 798a.
    """
    body = msg.split("reads: ", 1)[1].split(". Under-approximating", 1)[0]
    return {tok.strip() for tok in body.split(",") if not tok.strip().startswith("+")}


# --------------------------------------------------------------------------------------
# (1) the fingerprint fact the gate rests on -- asserted, not trusted
# --------------------------------------------------------------------------------------

def test_include_driver_flag_actually_gates_the_substrate_hash():
    """`include_driver_script_in_hash=False` is what drops the driver from the hash.

    The whole premise of this lint is that with the flag False, a module-level constant
    in the driver is invisible to the fingerprint. If arm_fingerprint.py ever folded the
    driver in regardless, the gate would be flagging a non-problem.
    """
    src = (EXPERIMENTS_DIR / "_lib" / "arm_fingerprint.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "compute_arm_fingerprint")
    kwonly = [a.arg for a in fn.args.kwonlyargs] + [a.arg for a in fn.args.args]
    assert "include_driver_script_in_hash" in kwonly, (
        "compute_arm_fingerprint no longer takes include_driver_script_in_hash -- the "
        "config_slice lint's premise has changed; re-derive the gate.")
    assert "config_slice" in kwonly

    # The flag must actually AND with the script path to decide folding.
    assert "fold_script = bool(script_path) and include_driver_script_in_hash" in src, (
        "the driver-folding decision in compute_arm_fingerprint changed shape -- "
        "re-verify that the flag still removes the driver's content from substrate_hash.")

    # arm_cell must forward it, or every `with arm_cell(...)` the lint scans is moot.
    cell = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "arm_cell")
    cell_kw = [a.arg for a in cell.args.kwonlyargs] + [a.arg for a in cell.args.args]
    assert "include_driver_script_in_hash" in cell_kw
    assert "config_slice" in cell_kw


# --------------------------------------------------------------------------------------
# (2) positive / negative shapes
# --------------------------------------------------------------------------------------

_UNDECLARED = """
    from pathlib import Path
    from _lib.arm_fingerprint import arm_cell

    SSL_BIN_EDGES = (2, 5, 12)
    LR = 0.001

    def _bin(x):
        lo, mid, hi = SSL_BIN_EDGES
        return 0 if x <= lo else (1 if x <= mid else (2 if x <= hi else 3))

    def run_cell(seed):
        return {"by_bin": _bin(seed), "lr": LR}

    def main():
        for seed in (1, 2):
            cfg = {"lr": LR, "arm_id": "A"}
            with arm_cell(seed, config_slice=cfg, script_path=Path(__file__),
                          include_driver_script_in_hash=False) as cell:
                cell.stamp(run_cell(seed))
"""


def test_fires_on_undeclared_readout_constant(tmp_path):
    """The canonical 798 shape: a binning constant read by the cell, absent from slice."""
    msg = V.config_slice_under_declaration_lint(_write(tmp_path, _UNDECLARED))
    assert msg is not None
    assert _named(msg) == {"SSL_BIN_EDGES"}, "LR is declared in the slice -- must not fire"


def test_quiet_when_the_constant_is_declared(tmp_path):
    """798a's fix: naming the constant in the slice clears it."""
    fixed = _UNDECLARED.replace(
        'cfg = {"lr": LR, "arm_id": "A"}',
        'cfg = {"lr": LR, "arm_id": "A", "ssl_bin_edges": list(SSL_BIN_EDGES)}')
    assert V.config_slice_under_declaration_lint(_write(tmp_path, fixed)) is None


def test_quiet_when_driver_is_in_the_hash(tmp_path):
    """With the default flag the driver's content IS hashed, so constants are bound.

    This is the gate's scoping rule, and it is the reason the lint is not simply a
    whole-corpus config_slice audit: only the cross-driver-reusable set is exposed.
    """
    default_flag = _UNDECLARED.replace(
        ",\n                          include_driver_script_in_hash=False", "")
    assert "include_driver_script_in_hash" not in default_flag
    assert V.config_slice_under_declaration_lint(_write(tmp_path, default_flag)) is None


def test_quiet_on_adjudication_threshold_outside_the_cell(tmp_path):
    """A threshold read only by `evaluate(rows)` is NOT readout-affecting.

    This is the call-graph scoping that took the corpus fire count from 67 to 48.
    """
    # NB same indentation as _UNDECLARED -- _write() dedents the CONCATENATION, so an
    # already-flush block here would leave mixed indentation and an unparseable fixture
    # (the lint then returns None and the test passes for the wrong reason).
    src = _UNDECLARED + """
    THRESH_C1 = 0.42

    def evaluate(rows):
        return all(r["by_bin"] >= THRESH_C1 for r in rows)
"""
    path = _write(tmp_path, src)
    ast.parse(path.read_text(encoding="utf-8"))  # fixture must be valid Python
    msg = V.config_slice_under_declaration_lint(path)
    assert msg is not None, "the SSL_BIN_EDGES fire should be unaffected"
    assert _named(msg) == {"SSL_BIN_EDGES"}, "THRESH_C1 is adjudication-only -- must not fire"


def test_exempt_marker_suppresses(tmp_path):
    src = 'CONFIG_SLICE_DECLARATION_EXEMPT = "bound by declared edges"\n' + _UNDECLARED
    assert V.config_slice_under_declaration_lint(_write(tmp_path, src)) is None


def test_slice_update_idiom_is_followed(tmp_path):
    """`slice_cfg = dict(base); slice_cfg.update({...})` is the dominant per-arm idiom."""
    src = _UNDECLARED.replace(
        'cfg = {"lr": LR, "arm_id": "A"}',
        'cfg = dict({"lr": LR})\n            '
        'cfg.update({"arm_id": "A", "ssl_bin_edges": SSL_BIN_EDGES})')
    assert V.config_slice_under_declaration_lint(_write(tmp_path, src)) is None


def test_unparseable_file_is_not_an_error(tmp_path):
    p = tmp_path / "v3_exq_999_broken.py"
    p.write_text("def (: oops\n", encoding="utf-8")
    assert V.config_slice_under_declaration_lint(p) is None


# --------------------------------------------------------------------------------------
# (2b) the DIRECT `compute_arm_fingerprint(...)` form -- no lexical cell body
# --------------------------------------------------------------------------------------
# The shape of all 15 direct-call drivers: the fingerprint call computes a hash and the
# arm is run by a SIBLING statement in the same loop body. `run_experiment` also calls
# `evaluate(rows)`, which is what makes enclosing-function scope degenerate -- so these
# fixtures deliberately include that adjudication call at the same level the real drivers
# do, and the negative assertion below is the whole point of using the loop.
_DIRECT_CALL = """
    from pathlib import Path
    from _lib.arm_fingerprint import compute_arm_fingerprint

    SSL_BIN_EDGES = (2, 5, 12)
    LR = 0.001
    THRESH_C1 = 0.42

    def _bin(x):
        lo, mid, hi = SSL_BIN_EDGES
        return 0 if x <= lo else (1 if x <= mid else (2 if x <= hi else 3))

    def _run_seed_arm(arm, seed):
        return {"by_bin": _bin(seed), "lr": LR}

    def evaluate(rows):
        return all(r["by_bin"] >= THRESH_C1 for r in rows)

    def run_experiment():
        rows = []
        for seed in (1, 2):
            row = _run_seed_arm("A", seed)
            row["arm_fingerprint"] = compute_arm_fingerprint(
                config_slice={"lr": LR, "arm_id": "A"},
                seed=seed,
                script_path=Path(__file__),
                include_driver_script_in_hash=False,
            )
            rows.append(row)
        return evaluate(rows)
"""


def test_direct_call_form_is_scoped(tmp_path):
    """The 15-script gap: before 2026-07-28 this returned None and scanned nothing.

    Same defect as the confirmed 798 instance, expressed in the form that has no `with`.
    """
    msg = V.config_slice_under_declaration_lint(_write(tmp_path, _DIRECT_CALL))
    assert msg is not None, (
        "a direct compute_arm_fingerprint(include_driver_script_in_hash=False) call is "
        "cross-driver-reusable exactly like the `with arm_cell` form -- it must be scoped")
    assert "SSL_BIN_EDGES" in _named(msg)


def test_direct_call_scope_is_the_loop_not_the_enclosing_function(tmp_path):
    """THRESH_C1 is read only by `evaluate(rows)`, called from the SAME function.

    This is the measurement that rejected enclosing-function scope: `run_experiment` is
    the enclosing function in 13 of the 15 real scripts and it calls `evaluate` too, so
    scoping to it would pull the entire adjudication block in and degenerate into the
    whole-module scan (28/60 fidelity, +271 constants, vs loop scope's 40/60 and +79).
    If this assertion ever fails, the scope has been widened back to the function.
    """
    msg = V.config_slice_under_declaration_lint(_write(tmp_path, _DIRECT_CALL))
    assert msg is not None
    assert "THRESH_C1" not in _named(msg), (
        "THRESH_C1 is an adjudication threshold read only downstream in evaluate(rows) -- "
        "the direct-call scope must be the LOOP holding the fingerprint call, not its "
        "enclosing function")
    assert _named(msg) == {"SSL_BIN_EDGES"}


def test_local_slice_helper_return_is_resolved(tmp_path):
    """`config_slice=_arm_config_slice(...)` -- 10 of the 15 declare their slice this way.

    Without following the local helper's `return`, the slice resolves as EMPTY and every
    constant the cell reads is reported missing -- including the ones declared one call
    away. That over-firing is what this resolution removes (and it cleared pre-existing
    false positives on 13 with-form scripts too).
    """
    src = _DIRECT_CALL.replace(
        'config_slice={"lr": LR, "arm_id": "A"},',
        'config_slice=_arm_config_slice("A"),'
    ).replace(
        "    def _bin(x):",
        '    def _arm_config_slice(arm):\n'
        '        return {"lr": LR, "arm_id": arm, "ssl_bin_edges": list(SSL_BIN_EDGES)}\n'
        "\n"
        "    def _bin(x):",
    )
    path = _write(tmp_path, src)
    ast.parse(path.read_text(encoding="utf-8"))  # fixture must be valid Python
    assert V.config_slice_under_declaration_lint(path) is None, (
        "the helper's returned dict declares ssl_bin_edges -- the resolver must follow "
        "a local function's return value")


def test_cross_module_slice_is_a_skip(tmp_path):
    """A slice built by an `_lib/baselines/` helper is declared in ANOTHER file.

    The gate cannot see one key of it, so firing would be a near-total false positive --
    and it would land on the scripts following the canonical-baseline pattern CLAUDE.md
    mandates for a reusable mint. Measured on V3-EXQ-700c: 21 of the 27 constants a fire
    would name are declared in that module's MATCHED_ENVELOPE. Silence is the honest
    answer here; closing the band needs cross-module parsing.
    """
    src = _DIRECT_CALL.replace(
        "    from _lib.arm_fingerprint import compute_arm_fingerprint",
        "    from _lib.arm_fingerprint import compute_arm_fingerprint\n"
        "    from experiments._lib.baselines.exq700_arc108_settling_baseline import (\n"
        "        arm_config_slice,\n"
        "    )",
    ).replace(
        'config_slice={"lr": LR, "arm_id": "A"},',
        'config_slice=arm_config_slice("A", 1, 1, 1, 1),')
    path = _write(tmp_path, src)
    ast.parse(path.read_text(encoding="utf-8"))
    assert V.config_slice_under_declaration_lint(path) is None, (
        "a config_slice built by a cross-module _lib/baselines helper is unassessable "
        "by a single-file scan and must be skipped, not fired on")


def test_cross_module_skip_needs_the_slice_to_come_from_that_helper(tmp_path):
    """Merely IMPORTING from `_lib/baselines/` must not buy a blanket exemption.

    The skip is keyed on the config_slice argument itself being a call to the imported
    helper. A script that imports (say) REUSABLE_ARM_IDS from a baseline module but still
    builds its slice inline is fully assessable, and silencing it would be a hole big
    enough to drive the confirmed 798 defect through.
    """
    src = _DIRECT_CALL.replace(
        "    from _lib.arm_fingerprint import compute_arm_fingerprint",
        "    from _lib.arm_fingerprint import compute_arm_fingerprint\n"
        "    from experiments._lib.baselines.exq700_arc108_settling_baseline import (\n"
        "        REUSABLE_ARM_IDS,\n"
        "    )")
    path = _write(tmp_path, src)
    ast.parse(path.read_text(encoding="utf-8"))
    msg = V.config_slice_under_declaration_lint(path)
    assert msg is not None and "SSL_BIN_EDGES" in _named(msg), (
        "the slice is still an inline dict -- importing something unrelated from a "
        "baselines module must not suppress the check")


# --------------------------------------------------------------------------------------
# (3) the confirmed instance, as a differential
# --------------------------------------------------------------------------------------

def test_confirmed_798_instance_is_the_differential():
    """798 must name SSL_BIN_EDGES; its successor 798a must not.

    This is the gate's reason for existing, pinned against the two real drivers rather
    than a synthetic fixture. 798a is the reference for the correct shape -- it declares
    `"binning"` / `"n_bins"` / `"ssl_bin_edges"` in its cell_config.

    NOTE 798a still fires on OTHER constants (training hyperparameters its slice also
    omits). That is deliberate and honest: 798a fixed the BINNING under-declaration it
    was written to fix, not every one in the file. The assertion here is specifically
    about SSL_BIN_EDGES, which is the confirmed hazard.
    """
    bad = EXPERIMENTS_DIR / "v3_exq_798_sdmelproducer_graded_nonconverging_world.py"
    good = EXPERIMENTS_DIR / "v3_exq_798a_sdmelproducer_graded_nonconverging_world_c4readable.py"
    if not bad.exists() or not good.exists():
        pytest.skip("V3-EXQ-798 pair not present in this checkout")

    bad_msg = V.config_slice_under_declaration_lint(bad)
    assert bad_msg is not None and "SSL_BIN_EDGES" in _named(bad_msg), (
        "V3-EXQ-798's undeclared SSL_BIN_EDGES is the confirmed instance this gate was "
        "built from -- if it no longer fires, the gate has regressed.")

    good_msg = V.config_slice_under_declaration_lint(good)
    if good_msg is not None:
        assert "SSL_BIN_EDGES" not in _named(good_msg), (
            "798a DECLARES ssl_bin_edges in its cell_config -- the lint must not "
            "re-flag it, or the declaration path is broken.")


# --------------------------------------------------------------------------------------
# (4) selector + WARN-only invariant
# --------------------------------------------------------------------------------------

def test_check_is_registered():
    assert "config_slice_declaration" in V.CHECK_NAMES


def test_gate_is_warn_only_even_under_strict(tmp_path):
    """Never hardens: not under --strict, not under --paths.

    A landed carrier's run is complete and its pre-registered emission is not rewritten,
    and the scan is best-effort in both directions (see the lint docstring's limits), so
    a fire is a triage signal and never a commit blocker.
    """
    script = _write(tmp_path, _UNDECLARED)
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"),
         "--strict", "--checks", "config_slice_declaration", "--paths", str(script)],
        capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, (
        "config_slice_declaration must be WARN-only in BOTH modes; it changed the exit "
        f"code.\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    assert "CONFIG_SLICE-DECLARATION WARNINGS" in proc.stdout
    assert "SSL_BIN_EDGES" in proc.stdout


# --------------------------------------------------------------------------------------
# (5) pinned corpus fire count
# --------------------------------------------------------------------------------------
# 56 of the 85 cross-driver drivers, re-pinned 2026-07-28 (from 48) when the direct-call
# form was brought into scope. The move is the NET of three effects, all measured, and it
# is worth keeping them separate because two of them pull the count DOWN:
#
#   48 -> 46  with-form, resolver: two scripts' fires were PRE-EXISTING FALSE POSITIVES
#             and went to zero once a local slice-helper's return was followed (735, 751).
#             Nine more shrank without clearing (793/793a 41 -> 15, 794/794a 11 -> 2,
#             766 8 -> 1, 767/767a/768/768a, 795, 832).
#   46 -> 45  with-form, cross-module skip: 833 builds its slice from an `_lib/baselines/`
#             helper, so it moved into the unassessable band.
#   45 -> 56  +11 direct-call scripts, newly scoped at all rather than silently skipped.
#
# Current split: 45 with-form + 11 direct-call. The remaining 4 direct-call scripts are
# 685 (nothing undeclared), 700c_mint (no module-level numeric constants at all), and
# 700c/700d (cross-module skip). Nothing in the direct-call set is silently unexamined
# any more -- each is either fired on, clean, or skipped for a stated reason.
#
# The backlog is NOT to be retro-fixed: these runs are complete and a completed run's
# pre-registered emission is not rewritten. It is a risk register -- the entry that
# matters is the one whose baseline a FUTURE consumer tries to reuse.
#
# The 11 direct-call carriers are ONE DRIVER FAMILY (704/704b, 707/707a/707b/707c,
# 708/708a/708b, 710, 714 -- the ARC-110 / MECH-440 / MECH-451 lineage on a shared
# template), so their 38-45 names each are the same defect replicated, not 11 independent
# problems: their `_arm_config_slice` declares the envelope's BOOLEANS
# (`use_f_eligibility_demotion`) but not its VALUES (`f_eligibility_envelope_floor`,
# passed straight to the agent build). Spot-verified on 704.
#
# Severity is not uniform across the 56, and the written finding
# (REE_assembly/evidence/planning/) records the classification. The band that matters
# most is a SCHEME constant -- a binning/quantisation choice like V3-EXQ-798's
# SSL_BIN_EDGES -- because a consumer that changes it gets numbers that are not merely
# differently-parameterised but differently-MEANING. Training hyperparameters
# (MAX_GRAD_NORM, CONTRASTIVE_BATCH_K) are the common band and are lower severity: a
# consumer changing them is usually running a different experiment anyway.
_PINNED_CORPUS_FIRE_COUNT = 56


def test_config_slice_corpus_fire_rate_is_pinned(corpus_scan):
    """Consumes the SHARED corpus walk (`tests/contracts/conftest.py`).

    Per that module's standing pattern, a new corpus-wide lint goes in `path_lints` and
    its corpus test takes `corpus_scan` rather than enumerating `experiments/` itself.
    The file set is exactly `sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py"))`, same as the
    other five pinned lints.
    """
    fired = corpus_scan["config_slice_under_declaration_lint"]
    assert len(fired) == _PINNED_CORPUS_FIRE_COUNT, (
        f"config_slice-declaration fire count moved: {len(fired)} vs pinned "
        f"{_PINNED_CORPUS_FIRE_COUNT}. If a NEW script is in this list, fix the script "
        f"(declare the constant in its config_slice, or add "
        f"CONFIG_SLICE_DECLARATION_EXEMPT) rather than re-pinning. If the count DROPPED "
        f"because a backlog carrier was fixed, re-pin.\nfired:\n  "
        + "\n  ".join(p.name for p in fired))


def test_confirmed_carrier_is_in_the_pinned_set(corpus_scan):
    """V3-EXQ-798 must be in the backlog -- it is the confirmed instance."""
    names = {p.name for p in corpus_scan["config_slice_under_declaration_lint"]}
    assert "v3_exq_798_sdmelproducer_graded_nonconverging_world.py" in names


# The direct-call carriers, named. A regression to `with`-only scoping takes the total
# count from 56 to 45 and would read as "11 backlog carriers were fixed" -- which is
# indistinguishable from real remediation if only the total is pinned. Naming them makes
# a silent loss of coverage fail as a loss of coverage.
_DIRECT_CALL_CARRIERS = frozenset({
    "v3_exq_704_mech451_finer_channel_granularity_falsifier.py",
    "v3_exq_704b_mech451_finer_channel_granularity_falsifier.py",
    "v3_exq_707_arc110_loop_segregation_validation.py",
    "v3_exq_707a_arc110_loop_segregation_validation.py",
    "v3_exq_707b_arc110_loop_segregation_c2_release.py",
    "v3_exq_707c_arc110_loop_segregation_c2_release_repair.py",
    "v3_exq_708_mech440_noisy_selection_head_propagation_falsifier.py",
    "v3_exq_708a_mech440_noisy_selection_head_propagation_falsifier.py",
    "v3_exq_708b_mech440_precommit_distribution_shape_falsifier.py",
    "v3_exq_710_disinhibitory_soft_competitive_settling_validation.py",
    "v3_exq_714_fullstack_selection_valuation_conversion_falsifier.py",
})


def test_direct_call_carriers_stay_covered(corpus_scan):
    """These 11 have NO `with arm_cell(...)` -- they are covered only by the loop scope."""
    fired = {p.name for p in corpus_scan["config_slice_under_declaration_lint"]}
    present = {n for n in _DIRECT_CALL_CARRIERS if (EXPERIMENTS_DIR / n).exists()}
    missing = sorted(present - fired)
    assert not missing, (
        "direct-call carriers stopped firing -- if their config_slice was genuinely fixed, "
        "drop them from _DIRECT_CALL_CARRIERS and re-pin; if not, the non-`with` scoping "
        f"has regressed and these are unexamined again:\n  " + "\n  ".join(missing))


def test_no_cross_driver_script_is_silently_unscanned():
    """The invariant the 2026-07-28 extension exists to establish.

    Every script passing `include_driver_script_in_hash=False` must be in exactly one of
    three states -- fires, verifiably clean, or skipped for a STATED reason (exempt
    marker, no module-level numeric constants, cross-module slice). The old failure mode
    was a fourth, invisible state: `return None` because the scan could not find a `with`,
    which looks identical to "clean" from the outside. This test reconstructs the reason
    for every silent script and fails if one has no accounted-for reason.
    """
    unexplained = []
    for path in sorted(EXPERIMENTS_DIR.glob("v3_exq_*.py")):
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        cross = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and V._call_name(n) in V._FP_CALL_NAMES
                 and any(k.arg == "include_driver_script_in_hash"
                         and isinstance(k.value, ast.Constant) and k.value.value is False
                         for k in n.keywords)]
        if not cross:
            continue
        if V.config_slice_under_declaration_lint(path) is not None:
            continue                                    # fires
        if V._CONFIG_SLICE_EXEMPT_MARKER in src:
            continue                                    # stated: exempt
        if V._BASELINE_HELPER_MODULE_MARKER in src:
            continue                                    # stated: cross-module slice
        has_consts = any(
            isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and len(n.targets[0].id) > 2
            and n.targets[0].id.isupper() and V._is_numeric_literal(n.value)
            for n in tree.body)
        if not has_consts:
            continue                                    # stated: nothing to check
        # Silent with constants present and no stated reason: it must be that every
        # constant the cell reads is declared. Assert the scope actually found a cell.
        if not any(isinstance(n, (ast.With, ast.AsyncWith)) for n in ast.walk(tree)):
            unexplained.append(path.name)
    assert not unexplained, (
        "cross-driver scripts silently unscanned -- no `with` cell, module-level numeric "
        "constants present, and no stated skip reason. This is the exact gap the 15 "
        f"direct-call scripts were in before 2026-07-28:\n  " + "\n  ".join(unexplained))
