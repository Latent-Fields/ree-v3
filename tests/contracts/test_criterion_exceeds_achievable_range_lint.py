"""Contracts for the DV-headroom class: the static lint AND the runtime precondition kind.

Substrate entry: `dv-dynamic-range-precondition-class` (priority 1, severity DEGRADING),
created by governance-20260903T2013 from the confirmed cluster autopsy
REE_assembly/evidence/planning/failure_autopsy_ext-claim-probe-cluster_2026-09-03.md
(target V3-EXQ-993).

Surfaces under test:
  (1) validate_experiments.criterion_exceeds_achievable_range_lint -- flags a driver that
      adjudicates a LOAD-BEARING criterion (or gates on a readiness precondition) whose
      threshold's FEASIBILITY nothing establishes.
  (2) validate_experiments.py --checks criterion_exceeds_achievable_range -- the selector,
      and the invariant that this gate is WARN-ONLY IN BOTH MODES (never hardens under
      --paths, never affects the exit code even under --strict).
  (3) experiments/_metrics.dv_achievable / dv_headroom_check / p0_readiness_gate -- the
      runtime half: measuring what the DV can actually reach, and self-routing to
      substrate_not_ready_requeue when the registered threshold is out of reach.
  (4) The corpus fire count, pinned, with a non-vacuity guard naming both canonical
      specimens rather than trusting the number alone.

THE DEFECT. Every readiness gate in this corpus certifies the INTERVENTION -- was the
channel perturbed, did the head train, were there enough samples -- and NONE certifies that
the DEPENDENT VARIABLE had room to move. Across the seven 2026-09-03 pending-review runs,
SIX passed all their preconditions and still could not discriminate, because the registered
pass threshold lay outside the range the configuration could produce. The compute was
spent; the load-bearing comparison was never adjudicated:

    V3-EXQ-981  C1 threshold 1.154 on a DV bounded in [0,1]        (unsatisfiable)
    V3-EXQ-981  precision-margin elevation 0.000195 vs floor 0.01        (51x)
    V3-EXQ-951c gate_caused with zero reachable ticks                (no support)
    V3-EXQ-983  decline_gap realised range 0.0468 vs C1 0.15            (3.2x)
    V3-EXQ-993  max |calibration_gap| 0.00152 vs floor 0.02            (13.1x)
    V3-EXQ-994  retention spread 0.00078 vs 0.02                       (25.6x)
    V3-EXQ-978  arm-mean difference one third of the DV's 0.05 quantum

THE LINT AND THE GATE ARE ONE FEATURE, and the tests are in one file for that reason: the
lint's stated remedy IS the runtime precondition, and a driver mentioning `dv_headroom` at
all silences the lint. Splitting them would let one drift from the other.

TWO SUB-CASES, and the second one is NOT criteria-only. Sub-case (a) is a multiplicative
threshold on a unit-interval DV (981's C1: `mean_hv_rate >= 2 * mean_base_rate`, needing
1.154 from a DV that cannot exceed 1.0). Sub-case (b) is an absolute floor on a
derived-range statistic -- and 981's OWN instance of it is a PRECONDITION, not a criterion
(`precision_margin_norm_elevated_under_hv`, a 0.01 floor on an elevation whose arithmetic
ceiling was 0.000195). A criteria-only scan would miss the very case the entry was written
from, which is why `test_cear_fires_on_a_precondition_floor` exists.

THE LOAD-BEARING NARROWING IS THE NOISE CONTROL, and it is the part to preserve if this is
ever touched. Scanning every criterion-shaped assignment fires on 210 of 1448 drivers
(14.5%); restricting to the corpus's own explicit `load_bearing: True` tag fires on 112
(7.7%) while KEEPING both known carriers. That is not cherry-picking a smaller number: the
autopsy's finding is specifically that "the LOAD-BEARING comparison was never adjudicated".
`test_cear_a_non_load_bearing_criterion_is_silent` holds that line.

WARN-ONLY BY CONSTRUCTION, not by caution. The lint cannot prove the criterion IS
unreachable -- the baseline is a runtime quantity it has no access to. It reports that
nothing establishes the threshold's feasibility. A warning resting on an unprovable premise
must never block a commit, and the 112 landed carriers' runs are complete. The right remedy
for a landed carrier is to adjudicate the affected RESULT.

SCOPE. This gates NEW scripts, like every sibling in this family. Do NOT retro-edit a
landed driver whose run is complete to silence it.
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import pytest  # noqa: E402

import validate_experiments as V  # noqa: E402
import _metrics as M  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Both carriers are named, not just counted: a pinned integer alone goes vacuously green if
# the gate stops firing entirely.
SPECIMEN_MULTIPLICATIVE = "v3_exq_981_mech027_control_plane_pathological_modes.py"
SPECIMEN_DERIVED_RANGE = "v3_exq_983_ext002_residue_error_persistence.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate_experiments.py"), *args],
        capture_output=True, text=True, cwd=str(REPO_ROOT))


def _lint_src(src: str):
    """Lint a synthetic script written into experiments/ (so relative scoping holds)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(src)
        name = f.name
    try:
        return V.criterion_exceeds_achievable_range_lint(Path(name))
    finally:
        os.unlink(name)


# The 981 shape, reduced to its skeleton: a load-bearing C1 demanding a MULTIPLE of a
# baseline on a DV that cannot exceed 1.0.
_MULTIPLICATIVE = '''
"""A driver whose load-bearing C1 cannot be satisfied by any policy."""
FALSE_ALARM_ELEVATION_MULTIPLIER = 2.0


def adjudicate(mean_hv_rate, mean_base_rate):
    c1_pass = bool(mean_hv_rate >= FALSE_ALARM_ELEVATION_MULTIPLIER * mean_base_rate)
    return {
        "criteria": [
            {"name": "C1_false_alarm_elevation", "load_bearing": True, "passed": c1_pass},
        ],
    }
'''

# The 983 shape: an absolute floor on a paired DIFFERENCE statistic.
_DERIVED_RANGE = '''
"""A driver whose load-bearing C1 floors a gap statistic at an absolute value."""
THRESH_C1_DECLINE_GAP = 0.15


def adjudicate(decline_a0, decline_a1):
    decline_gap = decline_a0 - decline_a1
    c1 = bool(decline_gap >= THRESH_C1_DECLINE_GAP)
    return {
        "criteria": [
            {"name": "C1_decline_gap", "load_bearing": True, "passed": c1},
        ],
    }
'''

# 981's OWN sub-case (b): the floor lives on a READINESS PRECONDITION, not a criterion.
_PRECONDITION_FLOOR = '''
"""A driver flooring an elevation precondition without bounding the baseline."""
PRECISION_MARGIN_HV_ELEVATION_FLOOR = 0.01


def gate(precision_margin_hv_elevation):
    return [
        {
            "name": "precision_margin_norm_elevated_under_hv",
            "measured": precision_margin_hv_elevation,
            "threshold": PRECISION_MARGIN_HV_ELEVATION_FLOOR,
            "direction": "lower",
        },
    ]
'''


# --------------------------------------------------------------------------- #
# (1) the lint fires on the shapes the autopsy names
# --------------------------------------------------------------------------- #

def test_cear_fires_on_the_multiplicative_shape():
    w = _lint_src(_MULTIPLICATIVE)
    assert w is not None
    assert "2x a baseline" in w and "unit-interval" in w


def test_cear_names_the_baseline_bound_it_would_need():
    """The message must be actionable: 1/K is the value the baseline may not exceed."""
    w = _lint_src(_MULTIPLICATIVE)
    assert "exceeds 0.5" in w, w


def test_cear_fires_on_the_derived_range_shape():
    w = _lint_src(_DERIVED_RANGE)
    assert w is not None
    assert "absolute floor of 0.15" in w and "decline_gap" in w


def test_cear_fires_on_a_precondition_floor():
    """981's own sub-case (b) is a PRECONDITION. A criteria-only scan misses it."""
    w = _lint_src(_PRECONDITION_FLOOR)
    assert w is not None
    assert "precondition" in w and "0.01" in w


def test_cear_fires_on_a_mirrored_comparison():
    src = _DERIVED_RANGE.replace("decline_gap >= THRESH_C1_DECLINE_GAP",
                                 "THRESH_C1_DECLINE_GAP <= decline_gap")
    assert _lint_src(src) is not None


def test_cear_fires_on_a_bare_literal_threshold():
    """Not every driver routes its threshold through a module constant."""
    src = _DERIVED_RANGE.replace("decline_gap >= THRESH_C1_DECLINE_GAP",
                                 "decline_gap >= 0.15")
    assert _lint_src(src) is not None


def test_cear_names_the_line_and_the_criterion():
    w = _lint_src(_DERIVED_RANGE)
    assert "line " in w and "`c1`" in w


def test_cear_message_is_ascii_only():
    """CLAUDE.md: anything reaching a terminal must be cp1252-safe."""
    for src in (_MULTIPLICATIVE, _DERIVED_RANGE, _PRECONDITION_FLOOR):
        w = _lint_src(src)
        assert w is not None
        w.encode("ascii")


def test_cear_message_points_at_the_runtime_remedy():
    """The lint and the gate are one feature; the message must say so."""
    w = _lint_src(_DERIVED_RANGE)
    assert "dv_headroom" in w and "p0_readiness_gate" in w


# --------------------------------------------------------------------------- #
# (2) the lint stays silent where it should -- the non-vacuity half
# --------------------------------------------------------------------------- #

def test_cear_a_non_load_bearing_criterion_is_silent():
    """THE noise control. Removing the tag must silence it -- see the docstring."""
    src = _DERIVED_RANGE.replace('"load_bearing": True, ', "")
    assert _lint_src(src) is None


def test_cear_declaring_dv_headroom_silences_it():
    """The stated remedy must actually work, or the lint is unfixable noise."""
    src = _DERIVED_RANGE.replace(
        "    c1 = bool(",
        "    _ = dv_headroom_check('dg', dv_name='decline_gap',\n"
        "                          criterion_threshold=THRESH_C1_DECLINE_GAP,\n"
        "                          control_values=[0.0, 0.5])\n"
        "    c1 = bool(")
    assert _lint_src(src) is None


def test_cear_explicit_opt_out_is_honoured():
    src = 'CRITERION_ACHIEVABLE_RANGE_EXEMPT = "range guaranteed by construction"\n' \
        + _DERIVED_RANGE
    assert _lint_src(src) is None


def test_cear_a_sub_unit_multiplier_is_silent():
    """K <= 1 cannot push a unit-interval DV out of range."""
    src = _MULTIPLICATIVE.replace("= 2.0", "= 0.5")
    assert _lint_src(src) is None


def test_cear_a_multiplicative_threshold_on_an_unbounded_dv_is_silent():
    """The claim is about the [0,1] ceiling; an unbounded DV has no such ceiling."""
    src = _MULTIPLICATIVE.replace("mean_hv_rate", "mean_hv_latency")
    assert _lint_src(src) is None


def test_cear_a_plain_statistic_floor_is_silent():
    """Only DERIVED-range statistics carry the claim; a level does not."""
    src = _DERIVED_RANGE.replace("decline_gap", "decline_level")
    assert _lint_src(src) is None


def test_cear_a_ceiling_criterion_is_silent():
    """An upper bound is a different shape; this lint makes no claim about it."""
    src = _DERIVED_RANGE.replace("decline_gap >= THRESH_C1_DECLINE_GAP",
                                 "decline_gap <= THRESH_C1_DECLINE_GAP")
    assert _lint_src(src) is None


def test_cear_an_upper_bound_precondition_is_silent():
    src = _PRECONDITION_FLOOR.replace('"direction": "lower"', '"direction": "upper"')
    assert _lint_src(src) is None


def test_cear_a_zero_floor_is_silent():
    """A floor of 0 asserts a sign, not a magnitude -- always reachable."""
    src = _DERIVED_RANGE.replace("= 0.15", "= 0.0")
    assert _lint_src(src) is None


def test_cear_an_unresolvable_threshold_is_silent():
    """A runtime-assembled threshold is invisible; the gate must not guess."""
    src = _DERIVED_RANGE.replace("THRESH_C1_DECLINE_GAP = 0.15",
                                 "THRESH_C1_DECLINE_GAP = compute_threshold()")
    assert _lint_src(src) is None


def test_cear_syntax_error_is_silent_not_fatal():
    assert _lint_src("def broken(:\n    pass\n") is None


# --------------------------------------------------------------------------- #
# (3) selector + WARN-only invariants
# --------------------------------------------------------------------------- #

def test_cear_is_a_registered_check_name():
    assert "criterion_exceeds_achievable_range" in V.CHECK_NAMES


def test_cear_is_warn_only_under_strict_and_paths():
    """The whole family's invariant: never hardens, never changes the exit code."""
    for spec in (SPECIMEN_MULTIPLICATIVE, SPECIMEN_DERIVED_RANGE):
        p = EXPERIMENTS_DIR / spec
        if not p.exists():
            pytest.skip(f"{spec} not present")
        r = _run("--checks", "criterion_exceeds_achievable_range", "--quiet", "--strict",
                 "--paths", f"experiments/{spec}")
        assert r.returncode == 0, r.stdout[-2000:]


def test_cear_is_selectable_and_does_not_drag_in_other_checks():
    p = EXPERIMENTS_DIR / SPECIMEN_DERIVED_RANGE
    if not p.exists():
        pytest.skip("specimen not present")
    r = _run("--checks", "criterion_exceeds_achievable_range", "--quiet",
             "--paths", f"experiments/{SPECIMEN_DERIVED_RANGE}")
    assert r.returncode == 0
    assert "criterion-exceeds-achievable-range-warning(s)" in r.stdout
    assert "0 readiness-warning(s)" in r.stdout


# --------------------------------------------------------------------------- #
# (4) corpus pin, with the non-vacuity guard
# --------------------------------------------------------------------------- #

def test_cear_corpus_fire_count_is_pinned_and_names_its_specimens():
    """A bare count goes vacuously green if the gate stops firing. Name the carriers.

    The count is a DRIFT ALARM, not a target: a large move means the gate's shape
    changed, and the band is wide enough that ordinary corpus growth does not trip it.
    """
    files = sorted(EXPERIMENTS_DIR.glob("*.py"))
    if len(files) < 100:
        pytest.skip("corpus not present")
    fired = [p.name for p in files
             if V.criterion_exceeds_achievable_range_lint(p) is not None]
    assert SPECIMEN_MULTIPLICATIVE in fired, "981 (sub-case a + precondition b) stopped firing"
    assert SPECIMEN_DERIVED_RANGE in fired, "983 (sub-case b) stopped firing"
    frac = len(fired) / len(files)
    assert 0.02 <= frac <= 0.15, (
        f"{len(fired)}/{len(files)} = {frac:.1%} fired; measured 7.7% at build time "
        "(2026-09-04). A large move means the gate's shape changed -- investigate "
        "before re-pinning.")


# --------------------------------------------------------------------------- #
# (5) the runtime half -- dv_achievable / dv_headroom_check / p0_readiness_gate
# --------------------------------------------------------------------------- #

def test_dv_achievable_range_is_max_minus_min():
    assert M.dv_achievable([0.1, 0.5, 0.3], "range") == pytest.approx(0.4)


def test_dv_achievable_max_abs_takes_magnitude():
    """993's shape: a signed gap against an absolute floor."""
    assert M.dv_achievable([-0.00065, 0.00152, -0.00022], "max_abs") == pytest.approx(0.00152)


def test_dv_achievable_ceiling_headroom_is_the_room_above_a_saturated_baseline():
    """981's sub-case (b): 0.000195 available against a 0.01 floor."""
    got = M.dv_achievable([0.999805], "ceiling_headroom", dv_bounds=(0.0, 1.0))
    assert got == pytest.approx(0.000195, abs=1e-9)


def test_dv_achievable_floor_headroom_mirrors_the_ceiling_case():
    assert M.dv_achievable([0.2, 0.4], "floor_headroom",
                           dv_bounds=(0.0, 1.0)) == pytest.approx(0.2)


def test_dv_achievable_non_finite_yields_nan_not_an_ordering_accident():
    """max() over NaN is order-dependent; the gate must get a deterministic UNMET."""
    assert math.isnan(M.dv_achievable([0.1, float("nan"), 0.3], "range"))
    assert math.isnan(M.dv_achievable([0.1, float("inf")], "range"))


def test_dv_achievable_refuses_an_empty_control_arm():
    """The vacuity this class exists to catch, arriving one level up."""
    with pytest.raises(ValueError, match="empty"):
        M.dv_achievable([], "range")


def test_dv_achievable_refuses_headroom_without_bounds():
    with pytest.raises(ValueError, match="dv_bounds"):
        M.dv_achievable([0.5], "ceiling_headroom")


def test_dv_achievable_refuses_an_unknown_statistic():
    with pytest.raises(ValueError, match="statistic"):
        M.dv_achievable([0.5], "vibes")


def test_dv_headroom_check_reproduces_the_983_shortfall():
    c = M.dv_headroom_check("dg", dv_name="decline_gap", criterion_threshold=0.15,
                            control_values=[0.0, 0.0468])
    assert c["kind"] == "dv_headroom"
    assert c["direction"] == "lower"
    assert 1.0 / c["headroom_ratio"] == pytest.approx(3.2, abs=0.05)


def test_dv_headroom_check_reproduces_the_981_precision_margin_shortfall():
    c = M.dv_headroom_check("pm", dv_name="precision_margin_norm",
                            criterion_threshold=0.01, control_values=[0.999805],
                            statistic="ceiling_headroom", dv_bounds=(0.0, 1.0))
    assert 1.0 / c["headroom_ratio"] == pytest.approx(51.3, abs=0.5)


def test_dv_headroom_check_reproduces_the_981_c1_unsatisfiability():
    """achievable 1.0 < required 1.154: no policy could have passed it."""
    c = M.dv_headroom_check("c1", dv_name="mean_hv_rate",
                            criterion_threshold=2 * 0.5771, achievable=1.0)
    assert c["measured"] < c["threshold"]
    with pytest.raises(M.P0NotReady):
        M.p0_readiness_gate([c])


def test_dv_headroom_margin_scales_the_requirement():
    c = M.dv_headroom_check("x", dv_name="d", criterion_threshold=0.1,
                            control_values=[0.0, 0.5], margin=2.0)
    assert c["threshold"] == pytest.approx(0.2)
    assert c["headroom_margin"] == 2.0


def test_dv_headroom_check_refuses_a_margin_below_one():
    """A margin < 1 inverts the gate's meaning rather than loosening it."""
    with pytest.raises(ValueError, match="margin"):
        M.dv_headroom_check("x", dv_name="d", criterion_threshold=0.1,
                            control_values=[0.0, 0.5], margin=0.5)


def test_dv_headroom_check_refuses_both_or_neither_source():
    for kw in ({}, {"control_values": [0.1, 0.2], "achievable": 0.5}):
        with pytest.raises(ValueError, match="exactly one"):
            M.dv_headroom_check("x", dv_name="d", criterion_threshold=0.1, **kw)


def test_dv_headroom_unmet_entry_raises_p0_not_ready_with_the_payload():
    """The routing the entry asks for: substrate_not_ready_requeue, not a false FAIL."""
    c = M.dv_headroom_check("cg", dv_name="calibration_gap", criterion_threshold=0.02,
                            control_values=[-0.00065, 0.00152, -0.00022],
                            statistic="max_abs")
    with pytest.raises(M.P0NotReady) as ei:
        M.p0_readiness_gate([c])
    entry = ei.value.preconditions[0]
    assert entry["met"] is False
    assert entry["kind"] == "dv_headroom"
    assert entry["dv_name"] == "calibration_gap"


def test_dv_headroom_met_entry_passes_through_the_gate():
    c = M.dv_headroom_check("ok", dv_name="d", criterion_threshold=0.01,
                            control_values=[0.0, 0.5])
    out = M.p0_readiness_gate([c])
    assert out[0]["met"] is True


def test_dv_headroom_entry_is_recomputable_by_the_indexer_contract():
    """The indexer recomputes met from (measured, threshold, direction) and is
    kind-agnostic -- which is WHY no indexer change was needed. Pin that shape."""
    c = M.dv_headroom_check("x", dv_name="d", criterion_threshold=0.02,
                            control_values=[0.0, 0.001])
    with pytest.raises(M.P0NotReady) as ei:
        M.p0_readiness_gate([c])
    e = ei.value.preconditions[0]
    # floor semantics: unmet iff measured < threshold
    assert (e["measured"] < e["threshold"]) is (e["met"] is False)
    assert e["direction"] == "lower"
    assert e["achievable_statistic"] in M.DV_HEADROOM_STATISTICS


def test_dv_headroom_refuses_an_upper_bound():
    """An upper bound would pass a PINNED DV and fail a live one -- the inversion."""
    c = M.dv_headroom_check("x", dv_name="d", criterion_threshold=0.1,
                            control_values=[0.0, 0.5])
    c["direction"] = "upper"
    with pytest.raises(ValueError, match="UPPER"):
        M.p0_readiness_gate([c])


def test_dv_headroom_refuses_a_missing_dv_name():
    c = M.dv_headroom_check("x", dv_name="d", criterion_threshold=0.1,
                            control_values=[0.0, 0.5])
    c["dv_name"] = "  "
    with pytest.raises(ValueError, match="dv_name"):
        M.p0_readiness_gate([c])


def test_dv_headroom_refuses_an_unknown_statistic_label():
    c = M.dv_headroom_check("x", dv_name="d", criterion_threshold=0.1,
                            control_values=[0.0, 0.5])
    c["achievable_statistic"] = "eyeballed"
    with pytest.raises(ValueError, match="achievable_statistic"):
        M.p0_readiness_gate([c])


# --------------------------------------------------------------------------- #
# (6) the default-off guarantee -- the governance boundary
# --------------------------------------------------------------------------- #

def test_p0_readiness_gate_is_byte_identical_for_drivers_that_do_not_opt_in():
    """governance-20260903T2013 scoped this build so it CANNOT perturb the 1,201
    drivers that import the substrate. An ordinary readiness check must come out
    exactly as it did before this class existed."""
    checks = [
        {"name": "policy_trained", "measured": 0.42, "threshold": 0.1},
        {"name": "bounded", "measured": 0.19, "threshold": 1e6, "direction": "upper"},
        {"name": "strict", "measured": 5.0, "threshold": 4.0, "comparator": ">"},
    ]
    out = M.p0_readiness_gate([dict(c) for c in checks])
    assert [e["met"] for e in out] == [True, True, True]
    assert [e["kind"] for e in out] == ["readiness"] * 3
    for e in out:
        assert "dv_name" not in e and "achievable_statistic" not in e


def test_dv_headroom_validation_never_fires_on_a_foreign_kind():
    """The validation is gated on kind == 'dv_headroom'. A bespoke kind carrying an
    upper bound (the corpus has several) must remain untouched."""
    out = M.p0_readiness_gate([
        {"name": "x", "measured": 0.19, "threshold": 1e6, "direction": "upper",
         "kind": "capability"},
    ])
    assert out[0]["met"] is True and out[0]["kind"] == "capability"


# --------------------------------------------------------------------------- #
# (6) the adopter registry -- gated on COMMITTED content
# --------------------------------------------------------------------------- #
#
# The first adopter of the dv_headroom kind. The assertion below was `hits == []`
# at build time (governance-20260903T2013), with its own docstring saying: "If
# this ever fails it is because a driver adopted the kind -- expected, and the
# assertion should move." It has moved twice: first onto the explicit allowlist
# below (which still catches an UNREVIEWED adoption while recording the reviewed
# ones), and then -- 2026-09-07, chip-20260904-dvheadroom-corpuslint-disposition
# -- off the WORKING TREE and onto COMMITTED content.
#
# WHY THE SOURCE OF THE FILE LIST IS LOAD-BEARING. The rule, and the thing to keep
# if this is ever touched: HEAD IS THE ONLY THING THAT CAN FAIL YOU; THE WORKING
# TREE CAN ONLY EXCUSE YOU.
#
# These two tests run inside a COMMIT GATE -- scripts/precommit_contracts.sh Block
# 1c fires them whenever a staged experiments/*.py outside _lib/ is committed --
# and ree-v3 is a SHARED CHECKOUT that several sessions edit at once. A glob over
# experiments/ therefore reads other sessions' untracked and staged scratch files:
# work that belongs to no commit, and cannot be any commit's business. Measured
# consequences while the glob stood:
#   2026-09-04  a campaign-C2 session's untracked 993a driver blocked an unrelated
#               session from committing v3_exq_1004_*, a driver that does not
#               mention the kind at all (1 failed, 688 passed in 311.98s).
#   2026-09-07  three further hits in one day. One of them -- session
#               hopeful-solomon-01a60c, committing a V3-EXQ-1005 refusal archive
#               that touches nothing related -- was resolved with --no-verify
#               (ree-v3 8132312). That is the gate being ROUTED AROUND rather than
#               satisfied, which is strictly worse than the check not existing.
# The repo's own doctrine already settles this shape of question: CLAUDE.md Session
# Startup Protocol step 7a, on the vendored-copy audit -- "The gate is on COMMITTED
# content ... A worktree difference is a NOTE, not a finding."
#
# WHAT THIS COSTS, stated rather than papered over: the failing direction is now
# POST-HOC. An unreviewed adopter is caught on the commit AFTER it lands, not at
# the moment it lands. That is the right trade because the two failures are not
# equivalent. A red against HEAD is legible and shared -- every session sees the
# same failure, and one commit adding one allowlist line clears it for everyone. A
# red against the working tree is invisible, per-session, and names a file the
# blocked session must not touch. Only the second kind produces a --no-verify.
#
# The PERMISSIVE direction deliberately still consults the working tree (see
# test_the_adopter_allowlist_has_no_stale_entries): the commit that adopts the kind
# and the commit that registers the adopter are the SAME commit, so at gate time
# the new driver is in neither HEAD nor the index (ree_commit.py stages into a
# PRIVATE index). Excusing an allowlist entry on working-tree evidence is safe
# precisely because a foreign session's file can only ever ADD an excuse there --
# never a failure.
KNOWN_DV_HEADROOM_ADOPTERS = {
    # V3-EXQ-993a: the ARC-021/MECH-069 redesign the dv_headroom class was
    # minted for. Its predecessor V3-EXQ-993 burned a 12-cell grid before
    # discovering its control arm produced no signal; H1/H2 are what refuse
    # that run before the compute (see the driver's docstring).
    "v3_exq_993a_arc021_merged_channel_action_conditioned_harm.py",
    # V3-EXQ-970a (2026-09-07): the ContextMemory content-half instrument
    # redesign. Its predecessor V3-EXQ-970's Regime A never produced its DV
    # (a fixed held-out N unreachable in 12/12 cells) and its Regime B
    # readout was pinned by a near-binary set-Jaccard; the redesign's
    # substrate-entry spec REQUIRES the DV's achievable range to be measured
    # at probe scale before the bar is pre-registered, and the dv_headroom
    # entry (ceiling_headroom above the seed-matched UNTRAINED control) is how
    # that requirement is enforced at run time, per regime, through the
    # regime-conditioned gate rather than a whole-run P0NotReady.
    "v3_exq_970a_contextmemory_write_content_h1_mi_instrument.py",
    # V3-EXQ-972a (2026-09-07): the SD-070 write-stream held-out linear probe.
    # Two entries, both reviewed at the Step 4.5 red-team pass: one per arm
    # certifying the DV has room above that arm's own label-shuffle null, and
    # one (dv_headroom_T3_above_lineage_accuracy) certifying the paired
    # routing contrast has room above the LINEAGE arm's REALISED accuracy --
    # the second exists because the DV of a difference is bounded by
    # 1 - acc(baseline), which a null-referenced headroom gate cannot see.
    "v3_exq_972a_sd070_write_stream_heldout_linear_probe.py",
}

_DV_HEADROOM_LITERAL = '"kind": "dv_headroom"'


def _require_git():
    """Skip when this tree has no git, rather than failing closed.

    remote_pytest.sh rsyncs the tree WITHOUT `.git` (see its RSYNC_EXCLUDES and
    CLAUDE.md "Running the test suite"), so anything shelling out to git there
    sees "not a git repository". Failing closed on that is the documented
    `validate_queue._is_tracked` trap -- a phantom contract failure that looks
    exactly like a real one.

    Skipping costs nothing that matters: this pair gates a COMMIT, and the commit
    gate (precommit_contracts.sh Block 1c) runs pytest locally in the checkout,
    where git is present. The fleet suite adds no adopters of its own.
    """
    if not (REPO_ROOT / ".git").exists():
        pytest.skip("no .git in this tree (remote_pytest.sh stages without it); "
                    "this pair gates commits and runs for real in "
                    "precommit_contracts.sh Block 1c, which executes locally")


def _git(*args):
    """git in the ree-v3 checkout, or None if it could not be run at all."""
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None


def _committed_experiment_scripts():
    """`experiments/*.py` filenames as COMMITTED at HEAD.

    Top level only -- the non-recursive shape the working-tree glob had, so
    experiments/_lib/** stays out of scope.
    """
    _require_git()
    r = _git("ls-tree", "-r", "--name-only", "HEAD", "--", "experiments/")
    if r is None or r.returncode != 0:
        pytest.skip("git ls-tree unavailable in this tree")
    return {line.rsplit("/", 1)[-1] for line in r.stdout.splitlines()
            if line.endswith(".py") and line.count("/") == 1}


def _committed_dv_headroom_declarers():
    """Filenames of top-level experiment scripts declaring the kind AT HEAD."""
    _require_git()
    r = _git("grep", "-l", "--fixed-strings", _DV_HEADROOM_LITERAL,
             "HEAD", "--", ":(glob)experiments/*.py")
    if r is None or r.returncode not in (0, 1):  # 1 == no match, not an error
        pytest.skip("git grep unavailable in this tree")
    names = set()
    for line in r.stdout.splitlines():
        _, _, path = line.partition(":")          # "HEAD:experiments/foo.py"
        if path:
            names.add(path.rsplit("/", 1)[-1])
    return names


def _worktree_dv_headroom_declarers():
    """The same set as it stands ON DISK. Only ever used to EXCUSE, never to fail."""
    return {p.name for p in EXPERIMENTS_DIR.glob("*.py")
            if _DV_HEADROOM_LITERAL in p.read_text(encoding="utf-8", errors="ignore")}


def test_only_reviewed_drivers_declare_the_new_kind():
    """Opt-in means opt-in: a driver may adopt this kind only deliberately.

    Adoption is not forbidden -- it is the point of the class -- but it must be
    a reviewed change rather than a copy-paste side effect, because a
    dv_headroom entry GATES the run (an unmet one raises P0NotReady and
    self-routes to substrate_not_ready_requeue). Add the filename to
    KNOWN_DV_HEADROOM_ADOPTERS in the same commit that adopts the kind.

    Reads HEAD, not the working tree -- see the block comment above the allowlist.
    """
    unreviewed = sorted(_committed_dv_headroom_declarers() - KNOWN_DV_HEADROOM_ADOPTERS)
    assert unreviewed == [], (
        f"committed drivers declare kind=dv_headroom without being listed in "
        f"KNOWN_DV_HEADROOM_ADOPTERS: {unreviewed}")


def test_the_adopter_allowlist_has_no_stale_entries():
    """A listed adopter that no longer declares the kind (renamed, reverted,
    deleted) must be removed, or the allowlist silently grows into a rubber
    stamp that permits any future file of that name.

    An entry is stale only if it declares the kind in NEITHER HEAD nor the
    working tree. The working-tree half is what lets the adopting commit and the
    registering commit be one commit; it can only excuse an entry, so a foreign
    session's file can never make this test fail.
    """
    declaring = _committed_dv_headroom_declarers() | _worktree_dv_headroom_declarers()
    stale = sorted(n for n in KNOWN_DV_HEADROOM_ADOPTERS if n not in declaring)
    assert stale == [], (
        f"allowlist entries declare the kind in neither HEAD nor the working "
        f"tree: {stale}")


def test_a_foreign_uncommitted_declarer_cannot_fail_either_test():
    """The regression pin for the defect this pair was rebuilt to close.

    Stands in for another session's in-flight driver sitting untracked in the
    shared checkout. Before 2026-09-07 such a file failed
    test_only_reviewed_drivers_declare_the_new_kind and blocked that session's
    unrelated commit; four measured occurrences, one forcing --no-verify.
    """
    _require_git()
    probe = EXPERIMENTS_DIR / f"v3_zz_probe_foreign_dv_headroom_{os.getpid()}.py"
    probe.write_text(
        '"""Stand-in for a FOREIGN session\'s untracked in-flight driver."""\n'
        'PRECONDITIONS = [{"name": "dv_headroom_x", ' + _DV_HEADROOM_LITERAL + '}]\n',
        encoding="utf-8")
    try:
        # non-vacuity: the probe really is on disk and really does declare the kind
        assert probe.name in _worktree_dv_headroom_declarers()
        # ... and is invisible to the failing direction, so nobody is blocked by it
        assert probe.name not in _committed_dv_headroom_declarers()
        assert sorted(_committed_dv_headroom_declarers()
                      - KNOWN_DV_HEADROOM_ADOPTERS) == []
    finally:
        probe.unlink(missing_ok=True)


def test_the_committed_corpus_enumeration_is_not_empty():
    """`_committed_experiment_scripts` going empty would make any future caller
    vacuously green. Named separately so that failure cannot hide inside one."""
    names = _committed_experiment_scripts()
    assert len(names) > 100, f"only {len(names)} committed experiments/*.py at HEAD"
    assert KNOWN_DV_HEADROOM_ADOPTERS <= names, sorted(KNOWN_DV_HEADROOM_ADOPTERS - names)
