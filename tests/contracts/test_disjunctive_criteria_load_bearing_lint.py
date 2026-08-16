"""Contracts for `validate_experiments.disjunctive_criteria_load_bearing_lint`.

THE DEFECT THIS GUARDS (authoring-side mistag, adjudicated 2026-08-16).

`build_experiment_indexes.py` rule (3b) flags `vacuous_pass` on any overall-PASS
manifest carrying a criterion tagged `load_bearing: true` with `passed: false`.
Its own comment says it is "gated on the explicit load_bearing tag so it never
over-fires on a legitimate M-of-N pass" -- and that gating is defeated by the
DRIVER's tagging, not by the indexer. A driver that declares a DISJUNCTIVE
acceptance rule (">=1 arm clears") and then tags its per-arm criteria
load-bearing has encoded a conjunction the rule does not declare, so the arm
whose failure is the INFORMATIVE half of the localisation trips the flag.

MEASURED. Over all 1777 indexed runs, exactly THREE PASS manifests trip rule
(3b) -- v3_exq_921, v3_exq_927, v3_exq_928 -- and all three are disjunctive: a
100% false-positive rate on that rule's output, each fire costing a governance
adjudication. The 927/928 pair was cleared as a tagging artifact in the confirmed
autopsy `REE_assembly/evidence/planning/
failure_autopsy_927-928-mech267-cluster_2026-08-16.md` section 2, which recorded
the fix direction used here: tag the AGGREGATE criterion `load_bearing: true` and
the per-member ones `load_bearing: false`, exactly as the 928 driver already
correctly does for its OFF control arm.

THE FIX DIRECTION IS THE ARTIFACT, NOT THE COUNTER. Widening indexer rule (3b) to
skip disjunctive manifests would stop it catching genuine aggregation vacuity,
which is what it exists for. Hence a lint on the authoring side.

WHAT IS DELIBERATELY NOT CAUGHT. v3_exq_921 -- one of the three carriers -- does
NOT fire, and that is accepted rather than fixed: its rule is an `A AND B` /
else-cascade with no disjunctive phrasing, and its failing load-bearing criterion
(`spawn_match_rate_met_precondition`) is a flat precondition, not a per-member
family. Chasing it would mean either inferring disjunction from prose that does
not say so, or flagging flat criteria -- and under a disjunction SOME flat
criterion genuinely is load-bearing (the aggregate). Both directions produce false
positives on ordinary authoring work, which is what gets a lint ignored. Sections
(2) and (3) below are the negative controls that pin that choice.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import validate_experiments as V  # noqa: E402

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

CARRIER = "v3_exq_928_mech267_cem_selection_fix_validation.py"


def _write(tmp_path: Path, body: str, name: str = "v3_exq_999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ---- (1) the firing forms ---------------------------------------------------------------

def test_fires_on_the_real_carrier_shape(tmp_path):
    """The exact v3_exq_928 shape: a `>=1` rule with the per-arm family tagged by a
    membership test that is true for 3 of 4 arms."""
    p = _write(tmp_path, '''
        _FIX_ARMS = ("H2", "H3", "BOTH")

        def build(arms, per_arm_clears):
            criteria = []
            for arm in arms:
                criteria.append({
                    "name": f"C_{arm}_gap_clears_floor",
                    "load_bearing": arm in _FIX_ARMS,
                    "passed": bool(per_arm_clears.get(arm, False)),
                })
            return {"interpretation": {
                "criteria": criteria,
                "combination_rule": (
                    "PASS iff >=1 fix arm's across-seed mean gap clears "
                    "FLOOR_PRODUCTION while the OFF control stays washed out."
                ),
            }}
    ''')
    issue = V.disjunctive_criteria_load_bearing_lint(p)
    assert issue is not None
    assert "arm in _FIX_ARMS" in issue, f"must name the offending tag: {issue}"
    assert "'>=1'" in issue, f"must name the matched disjunctive phrase: {issue}"
    assert "line 7" in issue, f"must name the criterion line: {issue}"


def test_fires_on_literal_true_in_a_loop(tmp_path):
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True, "passed": False})
            return {"combination_rule": "PASS iff at least one arm clears the floor."}
    ''')
    issue = V.disjunctive_criteria_load_bearing_lint(p)
    assert issue is not None
    assert "load_bearing: True" in issue, issue


def test_fires_on_a_comprehension_built_family(tmp_path):
    """`*[{...} for arm in arms]` is the same one-source-many-members shape as a
    `for` loop and must not escape by syntax."""
    p = _write(tmp_path, '''
        def build(arms):
            return {
                "criteria": [*[{"name": f"C_{a}", "load_bearing": True} for a in arms]],
                "combination_rule": "PASS iff any of the fix arms clears the floor.",
            }
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is not None


def test_fires_on_an_inequality_tag_true_for_many_members(tmp_path):
    """`arm != "OFF"` holds for every arm but one -- a conjunction over the fix arms."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": arm != "OFF"})
            return {"combination_rule": "PASS iff one or more fix arms clears the floor."}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is not None


def test_rule_bound_to_a_local_name_is_resolved(tmp_path):
    """10 of the 44 drivers declaring a combination_rule build it as a local string
    first (`combination_rule = (...)`) -- dropping that shape would lose a quarter
    of the lint's reach."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            combination_rule = (
                "PASS iff at least one arm clears the floor while OFF stays washed out."
            )
            return {"interpretation": {"criteria": criteria,
                                       "combination_rule": combination_rule}}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is not None


def test_reports_every_offending_family_not_just_the_first(tmp_path):
    p = _write(tmp_path, '''
        def build(arms, modes):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            for mode in modes:
                criteria.append({"name": f"M_{mode}", "load_bearing": True})
            return {"combination_rule": "PASS iff at least one arm clears."}
    ''')
    issue = V.disjunctive_criteria_load_bearing_lint(p)
    assert issue is not None
    assert "line 5" in issue and "line 7" in issue, issue


# ---- (2) negative controls: the rule is NOT disjunctive --------------------------------

def test_conjunctive_rule_does_not_fire(tmp_path):
    """The whole point: an `A AND B` gate legitimately has load-bearing members."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"combination_rule": "PASS iff EVERY arm clears the floor (C1 AND C2)."}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_decimal_threshold_is_not_read_as_a_disjunction(tmp_path):
    """The v3_exq_861c false positive the first draft produced: a pure conjunction
    whose text contains `>= 1.0 + K_CALIB_MARGIN`. `>=1` must not match a decimal."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"combination_rule": (
                "PASS iff readiness_ok AND c2_pass, where c2_pass = pinned_ok AND "
                "(ARM_3 mean_duration_factor >= 1.0 + K_CALIB_MARGIN * rel_sd) on "
                ">= 2/3 seeds."
            )}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_neither_does_not_match_either(tmp_path):
    """Real rule text says "informative neither way" -- a substring match on
    "either" would fire on it."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"combination_rule": (
                "PASS iff C1 AND C2. A null is informative neither way and routes "
                "to non_contributory."
            )}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_no_combination_rule_at_all_does_not_fire(tmp_path):
    """~97% of the corpus declares no combination_rule (44 of 1437 files mention it).
    The lint must be silent on them rather than guessing the acceptance shape -- and
    this is also the case the cheap substring pre-filter short-circuits before any
    tree work, so this pins the pre-filter's verdict as well as the semantics."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"criteria": criteria}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_prose_mention_of_a_disjunction_does_not_fire(tmp_path):
    """Docstrings and comments are invisible to the AST -- deliberately. Only the
    DECLARED combination_rule counts.

    The prose deliberately contains the literal string `combination_rule` so this
    clears the cheap substring pre-filter and actually reaches the AST scan --
    otherwise it would pass for the wrong reason and never test what it names.
    """
    p = _write(tmp_path, '''
        """Design note: the combination_rule here is PASS iff at least one arm clears."""
        # combination_rule (not declared in the manifest): >=1 fix arm clears

        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"criteria": criteria}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


# ---- (3) negative controls: the rule IS disjunctive but the tagging is correct ---------

def test_correctly_tagged_disjunctive_driver_does_not_fire(tmp_path):
    """The prescribed fix, verbatim: aggregate load_bearing True, members False.
    If this fires the lint is telling authors to do something it then flags."""
    p = _write(tmp_path, '''
        def build(arms, per_arm_clears, any_fix_clears, off_clears):
            criteria = []
            for arm in arms:
                criteria.append({
                    "name": f"C_{arm}_gap_clears_floor",
                    "load_bearing": False,
                    "passed": bool(per_arm_clears.get(arm, False)),
                })
            criteria.append({
                "name": "C_at_least_one_fix_arm_clears_with_off_washed_out",
                "load_bearing": True,
                "passed": bool(any_fix_clears and not off_clears),
            })
            return {"interpretation": {
                "criteria": criteria,
                "combination_rule": "PASS iff >=1 fix arm clears FLOOR_PRODUCTION.",
            }}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_flat_load_bearing_criterion_under_a_disjunction_does_not_fire(tmp_path):
    """A hand-written criterion is never a per-member family, and under a
    disjunction the aggregate gate is SUPPOSED to be load-bearing."""
    p = _write(tmp_path, '''
        def build(manipulation_ok, any_clears):
            criteria = [
                {"name": "C_manipulation_engagement_check",
                 "load_bearing": True, "passed": bool(manipulation_ok)},
                {"name": "C_at_least_one_arm_clears",
                 "load_bearing": True, "passed": bool(any_clears)},
            ]
            return {"criteria": criteria,
                    "combination_rule": "PASS iff at least one arm clears the floor."}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_single_member_equality_tag_does_not_fire(tmp_path):
    """`arm == "H2"` selects at most ONE member -- an aggregate criterion wearing a
    loop's clothes, not the all-members-load-bearing defect."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": arm == "H2"})
            return {"combination_rule": "PASS iff at least one arm clears the floor."}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_criterion_without_a_load_bearing_key_does_not_fire(tmp_path):
    """A per-member dict that is not a criterion at all (no load_bearing tag) is
    not this lint's business."""
    p = _write(tmp_path, '''
        def build(arms):
            rows = []
            for arm in arms:
                rows.append({"name": arm, "mean_gap": 0.0})
            return {"rows": rows,
                    "combination_rule": "PASS iff at least one arm clears the floor."}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


# ---- (4) robustness ---------------------------------------------------------------------

def test_exempt_marker_suppresses(tmp_path):
    p = _write(tmp_path, '''
        DISJUNCTIVE_CRITERIA_LOAD_BEARING_EXEMPT = "every arm must hold; rule text is loose"

        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"combination_rule": "PASS iff at least one arm clears the floor."}
    ''')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_syntax_error_returns_none_not_raise(tmp_path):
    """Must carry the literal `combination_rule` so it gets PAST the substring
    pre-filter and actually reaches `ast.parse` -- a bare `def broken(:` would
    short-circuit earlier and leave the SyntaxError branch untested."""
    p = _write(tmp_path, 'x = {"combination_rule": "PASS iff at least one arm"}\ndef broken(:\n')
    assert V.disjunctive_criteria_load_bearing_lint(p) is None


def test_missing_file_returns_none_not_raise(tmp_path):
    assert V.disjunctive_criteria_load_bearing_lint(tmp_path / "nope.py") is None


def test_message_is_ascii_only(tmp_path):
    """CLAUDE.md: anything reaching stdout must be ASCII (cp1252 mojibake on
    Windows terminals)."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"combination_rule": "PASS iff at least one arm clears the floor."}
    ''')
    issue = V.disjunctive_criteria_load_bearing_lint(p)
    assert issue is not None
    issue.encode("ascii")


# ---- (5) corpus calibration --------------------------------------------------------------

def test_carrier_set_is_exactly_the_known_carrier(corpus_scan):
    """Pinned at ONE fire across the whole corpus, by name.

    A NEW driver appearing here means the mistag was copied forward into a fresh
    script -- fix that one per the autopsy (aggregate load_bearing True, members
    False). The carrier LEAVING this set means either the landed 928 driver was
    retro-edited (it must not be -- its run is complete and already adjudicated)
    or the predicate silently stopped working.
    """
    expected = {CARRIER}
    actual = {p.name for p in corpus_scan["disjunctive_criteria_load_bearing_lint"]}
    assert actual == expected, (
        f"carrier set changed: newly firing {sorted(actual - expected)}, "
        f"no longer firing {sorted(expected - actual)}")


def test_the_carrier_driver_exists(corpus_scan):
    """Guards the pin above against going vacuously green on a renamed file."""
    assert (EXPERIMENTS_DIR / CARRIER).exists(), f"missing {CARRIER}"


def test_v3_exq_921_is_an_accepted_miss(corpus_scan):
    """921 trips indexer rule (3b) but is NOT this lint's shape -- see the module
    docstring. Pinned so that if a future widening DOES reach it, that is a
    deliberate, reviewed change rather than a silent one."""
    fired = {p.name for p in corpus_scan["disjunctive_criteria_load_bearing_lint"]}
    assert "v3_exq_921_mech490_sleep_commitgate_spawn_matched.py" not in fired


def test_conjunctive_corpus_drivers_are_clean(corpus_scan):
    """Real drivers that declare a combination_rule and are genuinely conjunctive.
    v3_exq_861c is the one that broke the first draft of the phrase list (its
    ">= 1.0 + K_CALIB_MARGIN" matched a naive `>=1`), so it is pinned by name."""
    conforming = {
        "v3_exq_861b_inv050_mech180_independent_seed_replication.py",
        "v3_exq_861c_inv050_mech180_calibration_fixed_replication.py",
        "v3_exq_886_mech481_coalition_4arm_falsifier.py",
    }
    for name in conforming:
        assert (EXPERIMENTS_DIR / name).exists(), f"missing {name}"
    fired = {p.name for p in corpus_scan["disjunctive_criteria_load_bearing_lint"]}
    assert not (conforming & fired), f"regressed: {sorted(conforming & fired)}"


# ---- (6) wiring ---------------------------------------------------------------------------

def test_lint_is_registered_in_check_names():
    """An unregistered lint is dead code: `main()` gates every lint on `selected`."""
    assert "disjunctive_criteria_load_bearing" in V.CHECK_NAMES


def test_lint_is_reachable_through_main(tmp_path, capsys):
    """Executes the real CLI over a synthetic carrier. Registration in CHECK_NAMES is
    necessary but not sufficient -- the main-loop branch and the report block are
    separate edits and any one of them can be missed."""
    p = _write(tmp_path, '''
        def build(arms):
            criteria = []
            for arm in arms:
                criteria.append({"name": f"C_{arm}", "load_bearing": True})
            return {"combination_rule": "PASS iff at least one arm clears the floor."}
    ''')
    argv = sys.argv
    sys.argv = ["validate_experiments.py", "--checks", "disjunctive_criteria_load_bearing",
                "--paths", str(p)]
    try:
        rc = V.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "DISJUNCTIVE-CRITERIA-LOAD_BEARING WARNINGS" in out, out
    assert rc == 0, "WARN-only: must never harden, even under --paths"
