"""Contracts for the dv_headroom STATISTIC-MISMATCH check: the static lint AND the
runtime observation falsifier.

Source: the confirmed four-diagnostic cluster autopsy of 2026-09-07 (REE_assembly
cb4a71fbd9,
evidence/planning/failure_autopsy_dv-headroom-diagnostics-cluster_2026-09-07.md).

SIBLING FILE, and read it first if this one is being changed:
tests/contracts/test_criterion_exceeds_achievable_range_lint.py covers the ORIGINAL
dv_headroom class -- whether a driver certifies its DV's range AT ALL. That gate is
silenced by ANY mention of `dv_headroom` in the file, deliberately, because its purpose
is to make the author answer the question rather than to police how. This file covers
what happens NEXT: having answered it, was the entry measured on the quantity the
criterion actually gates?

THE DEFECT. The dv_headroom precondition class is deployed and valuable, and TWO of its
three firings on 2026-09-07 were computed on a quantity other than the one the criterion
gates. Both directions cost real compute:

  V3-EXQ-972a  FALSE POSITIVE, the expensive direction. Criterion T3 tests the MEAN
               paired difference against a 0.15 margin. The entry used
               `ceiling_headroom` = 1 - MAX(per-seed control accuracy) = 0.0806. The
               MEAN-matched ceiling is 1 - mean(acc) = 0.155563, ratio 1.037, ADEQUATE.
               A genuine, adequately ranged null was presented as an instrument failure.
               The decisive tell was in the run's own data: 2 of 8 observed paired diffs
               (0.16135, 0.09297) EXCEEDED the asserted 0.0806 ceiling, one clearing the
               0.15 threshold outright.
  V3-EXQ-1009  WRONG STATISTIC. `custom_information.dv_headroom.headroom_ratio_by_cell`
               is computed on `delta_dbar` while criterion C1 gates
               `projected_lineage_increment = sqrt(B^2+d^2)-B`. The cell the headroom
               table makes look closest (GROUNDED/floor0.2, 0.9973) is 8.69x short on
               C1's own statistic; the true marginal cell (FROZEN/floor0.0, 2.31x) is
               never named.

TWO SURFACES, ONE FEATURE, tested in one file for the same reason the sibling gives:
the lint's stated remedy names the runtime check, and splitting them lets one drift.

  (1) validate_experiments.dv_headroom_statistic_mismatch_lint -- static, WARN-only.
  (2) experiments/_metrics.dv_headroom_observation_check -- runtime, and STRONGER. A
      headroom entry asserts a CEILING, which is a universal claim, so one observation
      above it REFUTES it: no distributional assumption, no judgement about which
      statistic was right, no access to the source. It catches what the static scan
      cannot see at all.

THE UNDER-FIRING BIAS IS THE PART TO PRESERVE, and it is measured rather than asserted.
The naive form of this lint fired on 10 of the 15 driver adopters while MISSING 972a
entirely. Three tightenings got it to 5 of 15 with BOTH carriers kept:

  - named-metric-keys on BOTH sides for the statistic-mismatch shape. Keying on bare
    Names fires on the documented-correct pairing, where `control_values=` is an ARM
    name (`control`, `base_rates`) and the criterion names the derived STATISTIC
    (`decline_gap`).
  - the adjudicating-comparison narrowing: only compares inside a `c1`/`..._pass`
    assignment or a criterion dict's `passed`. This is what excludes 993a, whose
    `max_abs` choice is documented against the autopsy's own table and whose comparison
    is a degeneracy guard rather than the adjudication.
  - `statistic="range"` excluded entirely: it is `max - min`, the documented pairing for
    a spread/difference criterion, so pairing it with a mean is not evidence.

test_dhsm_corpus_fire_count_is_pinned holds that line, and names both carriers rather
than trusting the integer -- a pinned count alone goes vacuously green if the gate stops
firing at all.

WARN-ONLY BY CONSTRUCTION. The lint cannot decide whether a mean-vs-max pairing is
actually wrong for a given DV; that depends on runtime distributions it has no access
to. It reports that the entry certifies one statistic while the criterion reads another,
and asks. A warning resting on a question rather than a proof must never block a commit,
and both carriers are landed drivers whose runs are complete and already adjudicated.

SCOPE. This gates NEW scripts. Do NOT retro-edit a landed driver whose run is complete
to silence it.
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

# Both carriers named, not merely counted.
SPECIMEN_ORDER_STATISTIC = "v3_exq_972a_sd070_write_stream_heldout_linear_probe.py"
SPECIMEN_WRONG_STATISTIC = "v3_exq_1009_mech267_elite_channel_ceiling_spike.py"
# Measured 2026-09-07 over the 1465-driver corpus. See the module docstring for the
# tightenings that produced it and the naive baseline (10 of 15 adopters, 972a missed).
# RE-PINNED 2026-09-08 (V3-EXQ-1015 landing): the 5 was measured on a WORKING TREE carrying
# uncommitted drivers. The COMMITTED corpus fires 3 -- at fcb3f16 (the commit that pinned
# 5) itself and at origin/main 976db83 alike: 1009 (wrong statistic), 642c, 972a (order
# statistic). Every full-suite run since the pin therefore failed this test on a tree with
# no lint-relevant change (first caught by the V3-EXQ-1015 pre-commit run, hub
# DLAPTOP-4-26576-20260908T184831Z: 1 failed / 4790 passed, this test). A pin must be
# measured on the committed tree, never on a checkout that also holds drafts.
EXPECTED_CORPUS_FIRES = 3


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
        return V.dv_headroom_statistic_mismatch_lint(Path(name))
    finally:
        os.unlink(name)


# 972a's shape, reduced to its skeleton. The mean lives one line above the comparison,
# in `m3 = _mean(d_vals)`, which is exactly why the lint must resolve a bare Name
# through its binding: without that the comparison is a Name against a constant and
# nothing is visible.
_ORDER_MISMATCH = '''
"""A driver certifying a MEAN criterion with a MAX-based ceiling."""
from experiments._metrics import dv_headroom_check

PROBE_MARGIN = 0.15
DV_BOUNDS = (0.0, 1.0)


def adjudicate(control_accuracies, paired_diffs):
    entry = dv_headroom_check(
        "dv_headroom_T3_above_lineage_accuracy",
        dv_name="paired excess diff",
        criterion_threshold=PROBE_MARGIN,
        control_values=control_accuracies,
        statistic="ceiling_headroom",
        dv_bounds=DV_BOUNDS,
    )
    m3 = _mean(paired_diffs)
    return {"criteria": [{"name": "T3", "load_bearing": True,
                          "passed": bool(m3 >= PROBE_MARGIN)}],
            "preconditions": [entry]}
'''

# 1009's shape: a hand-rolled custom_information dv_headroom dict -- no
# dv_headroom_check() call anywhere in the file -- reading one metric key while the
# criterion gates on another out of the same per-cell record.
_WRONG_STATISTIC = '''
"""A driver whose headroom table ranks cells on a metric its criterion does not read."""
CONTENT_FLOOR_ABS = 0.02


def adjudicate(by_cell):
    c1_cells = [lab for lab, c in by_cell.items()
                if c["projected_lineage_increment_mean"] >= CONTENT_FLOOR_ABS]
    c1_pass = len(c1_cells) > 0
    return {
        "criteria": [{"name": "C1", "load_bearing": True, "passed": c1_pass}],
        "custom_information": {
            "dv_headroom": {
                "name": "oracle_elite_centroid_ceiling",
                "floor": CONTENT_FLOOR_ABS,
                "achievable_by_cell": {
                    lab: c["delta_dbar_mean"] for lab, c in by_cell.items()},
            },
        },
    }
'''

# The MEAN-matched repair of _ORDER_MISMATCH: an analytic ceiling supplied via
# achievable=, which has no sample to take an order statistic of.
_MEAN_MATCHED = '''
"""The repair: an analytic, mean-matched ceiling rather than an order statistic."""
from experiments._metrics import dv_headroom_check

PROBE_MARGIN = 0.15


def adjudicate(mean_control_accuracy, paired_diffs):
    entry = dv_headroom_check(
        "dv_headroom_T3_above_lineage_accuracy",
        dv_name="paired excess diff",
        criterion_threshold=PROBE_MARGIN,
        achievable=1.0 - mean_control_accuracy,
    )
    m3 = _mean(paired_diffs)
    return {"criteria": [{"name": "T3", "load_bearing": True,
                          "passed": bool(m3 >= PROBE_MARGIN)}],
            "preconditions": [entry]}
'''


# --------------------------------------------------------------------------- #
# (1) the lint fires on the two shapes the autopsy names
# --------------------------------------------------------------------------- #

def test_dhsm_fires_on_the_order_statistic_shape():
    w = _lint_src(_ORDER_MISMATCH)
    assert w is not None
    assert "ceiling_headroom" in w and "MEAN" in w


def test_dhsm_names_the_entry_and_the_criterion_line():
    """Actionable or it is noise: the message must say WHICH entry and WHICH compare."""
    w = _lint_src(_ORDER_MISMATCH)
    assert "dv_headroom_T3_above_lineage_accuracy" in w, w
    assert "PROBE_MARGIN" in w, w


def test_dhsm_resolves_the_mean_through_a_bare_name_binding():
    """The load-bearing tightening. 972a's criterion reads `m3 >= PROBE_MARGIN`; the
    mean is in `m3 = _mean(d_vals)` one line above. A scan that inspects only the
    comparison expression sees a Name against a constant and MISSES the confirmed
    case entirely -- which is what the naive form of this lint did."""
    w = _lint_src(_ORDER_MISMATCH)
    assert w is not None and "_mean" in w, w


def test_dhsm_fires_on_the_wrong_statistic_shape():
    w = _lint_src(_WRONG_STATISTIC)
    assert w is not None
    assert "delta_dbar" in w and "projected_lineage_increment" in w


def test_dhsm_reads_a_hand_rolled_custom_information_dict():
    """1009 has NO dv_headroom_check() call anywhere. A call-only scan sees an adopter
    with no entries and stays silent on one of the two confirmed carriers."""
    assert "dv_headroom_check" not in _WRONG_STATISTIC
    assert _lint_src(_WRONG_STATISTIC) is not None


# --------------------------------------------------------------------------- #
# (2) the under-firing bias -- each tightening, held individually
# --------------------------------------------------------------------------- #

def test_dhsm_silent_on_a_mean_matched_analytic_ceiling():
    """The repair must actually silence it, or the lint teaches nothing."""
    assert _lint_src(_MEAN_MATCHED) is None


def test_dhsm_silent_on_a_non_adopter():
    """Scoped to drivers that mention the class. A file that never does cannot fire."""
    assert _lint_src('''
"""No dv_headroom anywhere."""
THRESH = 0.15


def adjudicate(vals):
    c1_pass = bool(_mean(vals) >= THRESH)
    return {"criteria": [{"name": "C1", "load_bearing": True, "passed": c1_pass}]}
''') is None


def test_dhsm_range_statistic_is_not_evidence():
    """`range` is max-min, the documented pairing for a spread/difference criterion
    (983's decline_gap). Pairing it with a mean is not by itself a mismatch, and
    treating it as one is a large part of what took the naive form to 10 of 15."""
    assert _lint_src(_ORDER_MISMATCH.replace('"ceiling_headroom"', '"range"')) is None


def test_dhsm_silent_when_the_comparison_is_not_the_adjudication():
    """993a's shape: `non_degenerate = mean_sep >= FLOOR` is a degeneracy guard, not the
    criterion, and its max_abs choice is documented against the autopsy's own table.
    Only compares inside a `c1`/`..._pass` assignment or a criterion dict's `passed`
    count."""
    src = _ORDER_MISMATCH.replace(
        '"passed": bool(m3 >= PROBE_MARGIN)', '"note": "descriptive"').replace(
        "m3 = _mean(paired_diffs)",
        "m3 = _mean(paired_diffs)\n    non_degenerate = bool(m3 >= PROBE_MARGIN)")
    assert _lint_src(src) is None


def test_dhsm_silent_on_a_bare_name_control_values():
    """The statistic-mismatch shape needs NAMED metric keys on BOTH sides. A plain
    `control_values=control` names an ARM, and an arm name differing from the
    criterion's derived-statistic name is the documented-CORRECT pairing (983a:
    `control` vs `decline_gap`), not a defect."""
    assert _lint_src('''
"""A correct entry whose control_values is a plain arm-named local."""
from experiments._metrics import dv_headroom_check

THRESH_C1_DECLINE_GAP = 0.15


def adjudicate(control, decline_gap):
    entry = dv_headroom_check(
        "dv_headroom_decline_gap", dv_name="decline_gap",
        criterion_threshold=THRESH_C1_DECLINE_GAP,
        control_values=control, statistic="range")
    c1 = bool(decline_gap >= THRESH_C1_DECLINE_GAP)
    return {"criteria": [{"name": "C1", "load_bearing": True, "passed": c1}],
            "preconditions": [entry]}
''') is None


def test_dhsm_silent_on_a_literal_threshold_it_cannot_join():
    """The join key is the criterion's own module constant. A re-typed literal joins to
    nothing, and the lint says nothing rather than guessing which criterion an entry
    was meant to certify -- a stated blind spot, held here so it stays deliberate."""
    src = _ORDER_MISMATCH.replace("criterion_threshold=PROBE_MARGIN",
                                  "criterion_threshold=0.15")
    assert _lint_src(src) is None


def test_dhsm_exemption_silences_it():
    src = _ORDER_MISMATCH.replace(
        "PROBE_MARGIN = 0.15",
        'DV_HEADROOM_STATISTIC_EXEMPT = "the ceiling is deliberately conservative"\n'
        "PROBE_MARGIN = 0.15")
    assert _lint_src(src) is None


def test_dhsm_one_finding_per_logical_entry():
    """A dv_headroom_check() call plus its NaN-fallback dict on the else branch is ONE
    entry written twice (972a lines 1067/1073). Reporting it twice trains the reader
    to skim."""
    src = _ORDER_MISMATCH.replace(
        "    m3 = _mean(paired_diffs)",
        '''    if not control_accuracies:
        entry = {"name": "dv_headroom_T3_above_lineage_accuracy",
                 "kind": "dv_headroom", "measured": float("nan"),
                 "threshold": PROBE_MARGIN, "direction": "lower",
                 "criterion_threshold": PROBE_MARGIN,
                 "achievable_statistic": "ceiling_headroom",
                 "achievable_by_seed": {}}
    m3 = _mean(paired_diffs)''')
    w = _lint_src(src)
    assert w is not None
    assert w.count("dv_headroom_T3_above_lineage_accuracy") == 1, w


# --------------------------------------------------------------------------- #
# (3) WARN-only in both modes, and the selector
# --------------------------------------------------------------------------- #

def test_dhsm_is_a_selectable_check():
    assert "dv_headroom_statistic_mismatch" in V.CHECK_NAMES


def test_dhsm_never_affects_the_exit_code():
    """WARN-only in BOTH modes -- it cannot prove the pairing is wrong (that depends on
    runtime distributions), and both carriers are landed drivers whose runs are
    complete."""
    r = _run("--checks", "dv_headroom_statistic_mismatch", "--strict",
             "--paths", f"experiments/{SPECIMEN_ORDER_STATISTIC}")
    assert r.returncode == 0, (r.returncode, r.stdout[-3000:], r.stderr[-2000:])


def test_dhsm_reports_the_carrier_under_the_selector():
    r = _run("--checks", "dv_headroom_statistic_mismatch",
             "--paths", f"experiments/{SPECIMEN_ORDER_STATISTIC}")
    out = r.stdout + r.stderr
    assert "DV_HEADROOM-STATISTIC-MISMATCH WARNINGS" in out, out[-3000:]


# --------------------------------------------------------------------------- #
# (4) the corpus fire count, with a non-vacuity guard
# --------------------------------------------------------------------------- #

def test_dhsm_corpus_fire_count_is_pinned():
    """A pinned integer alone goes vacuously green if the gate stops firing, so both
    carriers are named. If this drifts UP, a tightening was loosened -- re-measure
    before re-pinning; the whole point of the number is the under-firing bias.

    COMMITTED-ONLY (2026-09-09): filtered through `V.committed_driver_names()` so
    another session's uncommitted draft driver sitting in this shared checkout
    cannot move the pin -- see that function's docstring and the RE-PINNED
    2026-09-08 note on EXPECTED_CORPUS_FIRES above, which is exactly this failure
    mode caught once already and re-pinned rather than fixed at the root. When
    git is unavailable, `committed_driver_names()` returns None and this falls
    back to the unfiltered working-tree glob, unchanged from before."""
    tracked = V.committed_driver_names()
    candidates = sorted(EXPERIMENTS_DIR.glob("*.py"))
    if tracked is not None:
        candidates = [p for p in candidates if p.name in tracked]
    fired = [p.name for p in candidates if V.dv_headroom_statistic_mismatch_lint(p)]
    assert SPECIMEN_ORDER_STATISTIC in fired, fired
    assert SPECIMEN_WRONG_STATISTIC in fired, fired
    assert len(fired) == EXPECTED_CORPUS_FIRES, fired


def test_dhsm_corpus_pin_ignores_an_untracked_specimen():
    """Regression for the failure mode above: an untracked file dropped into
    `experiments/` -- exactly what another session's in-progress draft looks
    like from here -- must not move the pinned count, whether or not it would
    itself fire the lint. Uses the ORDER_MISMATCH shape (a genuine firer) as the
    specimen precisely because that is the case that would otherwise move the
    number; a specimen that never fires would not exercise the filter at all."""
    if V.committed_driver_names() is None:
        pytest.skip("no .git in this tree -- filter has nothing to prove here "
                    "(see committed_driver_names()'s docstring)")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir=str(EXPERIMENTS_DIR)) as f:
        f.write(_ORDER_MISMATCH)
        specimen = Path(f.name)
    try:
        assert specimen.name not in V.committed_driver_names(), (
            "specimen leaked into git's index -- fix the test, not the filter")
        tracked = V.committed_driver_names()
        candidates = [p for p in sorted(EXPERIMENTS_DIR.glob("*.py")) if p.name in tracked]
        assert specimen.name not in {p.name for p in candidates}
        fired = [p.name for p in candidates if V.dv_headroom_statistic_mismatch_lint(p)]
        assert len(fired) == EXPECTED_CORPUS_FIRES, fired
    finally:
        specimen.unlink()


def test_dhsm_fires_on_a_minority_of_adopters():
    """The bias, stated as a relation rather than a number: most drivers that adopt the
    class are measuring it correctly, and a check that flags most of them is broken."""
    adopters = [p for p in EXPERIMENTS_DIR.glob("*.py")
                if "dv_headroom" in p.read_text(encoding="utf-8", errors="ignore")]
    fired = [p for p in adopters if V.dv_headroom_statistic_mismatch_lint(p)]
    assert len(adopters) >= 10, len(adopters)
    assert len(fired) * 2 < len(adopters), (len(fired), len(adopters))


# --------------------------------------------------------------------------- #
# (5) the runtime falsifier -- stronger than the lint, and the generalisable half
# --------------------------------------------------------------------------- #

def test_observation_check_catches_972a_from_its_own_data():
    """The confirmed case, with its real numbers. The entry asserted a 0.0806 ceiling;
    the run observed 0.16135 and 0.09297. A ceiling is a universal claim, so this
    refutes it outright -- no distributional assumption, no source access."""
    entry = M.dv_headroom_check(
        "dv_headroom_T3", dv_name="paired excess diff", criterion_threshold=0.15,
        control_values=[0.91935, 0.844437, 0.80], statistic="ceiling_headroom",
        dv_bounds=(0.0, 1.0))
    assert entry["measured"] == pytest.approx(1 - 0.91935)
    M.dv_headroom_observation_check(
        entry, [0.16135, 0.09297, 0.01, -0.02, 0.04, 0.03, 0.02, 0.0])
    flag = entry[M.DV_HEADROOM_OBSERVATION_FLAG]
    assert flag["exceeded"] is True
    assert flag["n_exceeding"] == 2
    assert flag["exceeding_values"] == [pytest.approx(0.16135), pytest.approx(0.09297)]


def test_observation_check_names_the_values_that_cleared_the_criterion():
    """The sharpest tell, and the one an autopsy needs: an observed value above the
    ceiling that ALSO clears the criterion's own threshold means the run produced a
    passing value of a statistic the gate called out of reach."""
    entry = M.dv_headroom_check(
        "e", dv_name="d", criterion_threshold=0.15,
        control_values=[0.91935], statistic="ceiling_headroom", dv_bounds=(0.0, 1.0))
    M.dv_headroom_observation_check(entry, [0.16135, 0.09297])
    assert "clear the criterion's own threshold" in \
        entry[M.DV_HEADROOM_OBSERVATION_FLAG]["reason"]


def test_observation_check_clears_the_mean_matched_ceiling():
    """The repair must survive its own data, or the check is just noise."""
    entry = M.dv_headroom_check("e", dv_name="d", criterion_threshold=0.15,
                                achievable=1 - 0.844437)
    M.dv_headroom_observation_check(entry, [0.09297, 0.01, 0.04, 0.02])
    assert entry[M.DV_HEADROOM_OBSERVATION_FLAG]["exceeded"] is False


def test_observation_check_never_raises_at_emit_time():
    """By emit time the compute is spent; an exception here would cost the manifest to
    report a problem WITH the manifest. Every degenerate input records and returns."""
    for entry, observed in [
            ({}, [1.0]),
            ({"measured": None}, [1.0]),
            ({"measured": "not a float"}, [1.0]),
            ({"measured": 0.5}, None),
            ({"measured": 0.5}, []),
            ({"measured": float("nan")}, [1.0]),
            ({"measured": 0.5}, [None, "x", float("inf")]),
    ]:
        out = M.dv_headroom_observation_check(dict(entry), observed)
        assert out[M.DV_HEADROOM_OBSERVATION_FLAG]["exceeded"] is False


def test_observation_check_nan_ceiling_is_indeterminate_not_met():
    """Same asymmetry dv_headroom_check applies, for the same reason: a measurement
    that failed must not certify anything, and must not be reported as falsified
    either."""
    entry = {"measured": float("nan"), "dv_name": "d"}
    M.dv_headroom_observation_check(entry, [99.0])
    flag = entry[M.DV_HEADROOM_OBSERVATION_FLAG]
    assert flag["checked"] is False and flag["exceeded"] is False
    assert "INDETERMINATE" in flag["reason"]


def test_observation_check_drops_and_counts_non_finite_observations():
    entry = {"measured": 0.5, "dv_name": "d"}
    M.dv_headroom_observation_check(entry, [float("nan"), 0.4, float("inf"), 0.3])
    flag = entry[M.DV_HEADROOM_OBSERVATION_FLAG]
    assert flag["n_observed"] == 2 and flag["n_observed_nonfinite_dropped"] == 2
    assert flag["exceeded"] is False


def test_observation_check_equality_is_not_an_exceedance():
    """A DV that touched its own bound has not falsified the bound."""
    entry = {"measured": 0.5, "dv_name": "d"}
    M.dv_headroom_observation_check(entry, [0.5])
    assert entry[M.DV_HEADROOM_OBSERVATION_FLAG]["exceeded"] is False


def test_observation_check_returns_the_same_entry_object():
    """Annotates in place and returns, so it can wrap an existing preconditions[] entry
    without restructuring the emit."""
    entry = {"measured": 0.5, "dv_name": "d"}
    assert M.dv_headroom_observation_check(entry, [0.4]) is entry


def test_observation_check_reason_is_ascii():
    """CLAUDE.md: this reaches stdout and lands in manifests."""
    entry = M.dv_headroom_check("e", dv_name="d", criterion_threshold=0.15,
                                control_values=[0.9], statistic="ceiling_headroom",
                                dv_bounds=(0.0, 1.0))
    M.dv_headroom_observation_check(entry, [0.5])
    entry[M.DV_HEADROOM_OBSERVATION_FLAG]["reason"].encode("ascii")


def test_lint_message_points_at_the_runtime_check():
    """The two surfaces are one feature. A reader who hits the static WARN must be told
    about the falsifier that can actually settle it."""
    w = _lint_src(_ORDER_MISMATCH)
    assert "dv_headroom_observation_check" in w, w
