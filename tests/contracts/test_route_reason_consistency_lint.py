"""Contracts for `validate_experiments.route_reason_consistency_lint`.

THE DEFECT THIS GUARDS (authoring-side, confirmed 2026-08-17).

`interpretation.route_reason` is a driver's own machine-readable account of WHY it
reached its verdict, and governance and failure-autopsy sessions read it as such.
When the acceptance gate is a CONJUNCTION of N criteria, the fall-through `else`
is reached whenever ANY ONE of them fails -- so a single hardcoded literal there
names one of N possible causes and is factually wrong on the other N-1 branches.

MEASURED. v3_exq_935 gates on
`rule_supported = bool(c1_passed and c3_passed and (c2_passed or c2_scoped_out))`
and hardcodes `route_reason =
"no_common_normalised_rule_outperformed_the_best_absolute_cap"` on the `else`.
The run failed on C1 alone; the manifest's own
`acceptance.c3_beats_best_absolute_cap` was `true`. Adjudicated in
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-935_2026-08-18.md`.
Nothing audited it -- a human autopsy caught it by re-reading the manifest.

WHY THE DISCRIMINATOR IS NOT "hardcoded literal on a fall-through". 34 of the
1344 drivers assign a string literal to a reason variable in a terminal `else`,
and 20 of those do it under a >=2-conjunct guard. Firing on all 20 would be
wrong: a fall-through reason naming the DISJUNCTION of causes ("criteria_unmet",
"rule_not_supported") is correct by construction and is what most of them do.
The defect is that the literal SINGLES OUT one criterion. So the test is lexical
against the criteria the guard actually names -- >=2 shared content tokens with
one declared criterion name, strictly more than every other criterion in the same
conjunction. That step takes 20 fires to 1. Section (5) pins the funnel; sections
(2) and (3) are the negative controls that stop it being widened back.

WHAT IS DELIBERATELY NOT CAUGHT, and it is one of the two confirmed instances.
v3_exq_467e / v3_exq_464e emit `route_reason = "external_task_mode_not_occupied"`
while the mode was 100% occupied -- but from an `elif not
occupancy_non_vacuity_met:` branch, i.e. the guard IS the criterion the string
names and the driver is self-consistent at the boolean level. That reason was
false because the underlying STATISTIC (a `min()` across arms) cannot distinguish
"occupancy 0.0 everywhere" from "occupancy {0.0, 1.0}", and the label was written
for the first. No static scan over the AST can see a criterion/statistic semantic
mismatch. Both drivers are pinned in section (5) as accepted misses so that a
later widening reaching them is a deliberate, reviewed change rather than a
silent one -- and so that this lint is never cited as covering that instance.
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

CARRIER = "v3_exq_935_mech266_margin_normalised_cap_rule.py"
ACCEPTED_MISSES = (
    "v3_exq_467e_mech266_mode_stickiness_behavioural.py",
    "v3_exq_464e_mech266_competing_goals_behavioural.py",
)


def _write(tmp_path: Path, body: str, name: str = "v3_exq_999_probe.py") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


# ---- (1) the firing forms ---------------------------------------------------------------

def test_fires_on_the_real_carrier_shape(tmp_path):
    """The exact v3_exq_935 shape: a readiness cascade, then `elif <conjunction>`, then
    a bare `else` hardcoding a reason that names C3 while C1 can equally be the cause."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c2_passed, c2_scoped_out, c3_passed, ready):
            rule_supported = bool(c1_passed and c3_passed and (c2_passed or c2_scoped_out))
            if not ready:
                outcome = "FAIL"
                route_reason = "contact_guard_unmet"
            elif rule_supported:
                outcome = "PASS"
                route_reason = "single_pre_registered_r_grades_across_seeds"
            else:
                outcome = "FAIL"
                route_reason = "no_common_normalised_rule_outperformed_the_best_absolute_cap"
            return {"outcome": outcome, "interpretation": {
                "route_reason": route_reason,
                "criteria": [
                    {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                    {"name": "C2_rule_generalises_out_of_sample", "passed": c2_passed},
                    {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
                ],
            }}
    ''')
    issue = V.route_reason_consistency_lint(p)
    assert issue is not None
    assert "c3_passed" in issue, f"must name the criterion singled out: {issue}"
    assert "c1_passed" in issue, f"must name the co-causes: {issue}"
    assert "line 12" in issue, f"must name the assignment line: {issue}"


def test_fires_on_an_inline_conjunction_guard(tmp_path):
    """The guard need not be bound to a name first -- `elif a and b:` is the same shape."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed, ready):
            if not ready:
                route_reason = "contact_guard_unmet"
            elif c1_passed and c3_passed:
                route_reason = "both_criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is not None


def test_fires_on_a_bare_if_else_with_no_elif(tmp_path):
    """`if <conjunction>: PASS else: <hardcoded>` has the identical arity defect and must
    not escape merely by having no readiness cascade in front of it."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is not None


def test_criterion_names_resolve_through_an_acceptance_map(tmp_path):
    """Form (B): drivers also bind criterion identity as `{"<name>": <var>}` in the
    acceptance block rather than a `{"name": ..., "passed": ...}` dict. Dropping that
    form would lose the link for any driver that only writes the map."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"acceptance": {
                "c1_rule_grades_at_r_star": c1_passed,
                "c3_beats_best_absolute_cap": c3_passed,
            }, "route_reason": route_reason}
    ''')
    assert V.route_reason_consistency_lint(p) is not None


def test_readiness_route_is_scanned_too(tmp_path):
    """`readiness_route` carries the same self-report weight as `route_reason`."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                readiness_route = "rule_admits_common_mixed_regime"
            else:
                readiness_route = "rule_did_not_beat_the_best_absolute_cap"
            return {"readiness_route": readiness_route, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is not None


def test_each_offending_else_is_reported_once_not_once_per_elif(tmp_path):
    """An `elif` is itself an `ast.If`, so a naive `ast.walk` reports the same terminal
    `else` once per branch of the chain. The carrier produced FIVE copies of one finding
    before the outermost-If filter went in."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed, a, b, c):
            if not a:
                route_reason = "guard_a_unmet"
            elif not b:
                route_reason = "guard_b_unmet"
            elif not c:
                route_reason = "guard_c_unmet"
            elif c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    issue = V.route_reason_consistency_lint(p)
    assert issue is not None
    # Assert the dedup PROPERTY, not a line number: the chain above has four branches, so
    # a regression here reports the same terminal `else` four or five times.
    assert issue.count("singles out") == 1, f"finding duplicated per elif: {issue}"
    assert issue.count("line 12") == 1, f"finding duplicated per elif: {issue}"


# ---- (2) negative controls: the reason is NOT a single-criterion claim -------------------

def test_generic_fall_through_reason_does_not_fire(tmp_path):
    """19 of the 20 corpus drivers with a >=2-conjunct guard and a hardcoded fall-through
    reason are THIS shape -- the reason names the disjunction, which is correct. If this
    fires, the lint fires on ordinary correct authoring and gets turned off."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "criteria_unmet_genuine_weakens"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_reason_derived_from_the_failing_criterion_does_not_fire(tmp_path):
    """The prescribed fix, verbatim: the reason is COMPUTED from which conjunct came out
    false. A conditional expression is not a hardcoded claim. If this fires the lint is
    flagging the very shape it tells authors to adopt."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = ("c1_rule_grades_at_r_star_unmet" if not c1_passed
                                else "c3_beats_best_absolute_cap_unmet")
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_reason_naming_every_criterion_equally_does_not_fire(tmp_path):
    """A reason that overlaps the conjuncts EQUALLY is not singling one out, so the
    strict-max test must spare it -- this is what makes the discriminator 'distinctive'
    rather than merely 'has some overlap'.

    The overlaps here are a genuine tie at 2 each (`alpha`+`requirements` vs
    `beta`+`requirements`). An earlier draft of this fixture scored 3 vs 2 and therefore
    fired -- which was the lint being right and the test being wrong. If this ever needs
    changing, recount the token overlaps rather than adjusting the assertion.
    """
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "alpha_and_beta_requirements_both_unresolved"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_alpha_requirements", "passed": c1_passed},
                {"name": "C3_beta_requirements", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


# ---- (3) negative controls: the gate is NOT a multi-criterion conjunction ---------------

def test_single_criterion_gate_does_not_fire(tmp_path):
    """One criterion means the fall-through has exactly ONE cause, so naming it is
    correct. Explicitly required by the chip that commissioned this lint."""
    p = _write(tmp_path, '''
        def adjudicate(c3_passed):
            if c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_negated_single_criterion_guard_does_not_fire(tmp_path):
    """The v3_exq_467e shape: `elif not <criterion>:` -- the guard IS the criterion the
    reason names, so the driver is self-consistent at the boolean level. This is the
    confirmed instance the lint deliberately does not catch (its reason was false at the
    STATISTIC level); pinned so a widening cannot claim it silently."""
    p = _write(tmp_path, '''
        def adjudicate(contact_met, occupancy_met, overall_pass):
            if not contact_met:
                route_reason = "contact_guard_unmet"
            elif not occupancy_met:
                route_reason = "external_task_mode_not_occupied"
            else:
                route_reason = "c1_c2_c3_majority_met" if overall_pass else "criteria_unmet"
            return {"route_reason": route_reason, "criteria": [
                {"name": "contact_non_vacuity", "passed": contact_met},
                {"name": "external_task_mode_not_occupied", "passed": occupancy_met},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_disjunctive_guard_does_not_fire(tmp_path):
    """`a or b` is not a conjunction -- the fall-through means BOTH failed, so naming
    either one is not a false single-cause claim in the same way."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed or c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_guard_conjuncts_not_bound_to_declared_criteria_does_not_fire(tmp_path):
    """Without the manifest's own criterion dicts there is no structural link between a
    guard variable and a criterion NAME, and the lint refuses to guess one from the
    variable's spelling. An accepted, deliberate blind spot."""
    p = _write(tmp_path, '''
        def adjudicate(a, b):
            if a and b:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_chain_with_no_bare_else_does_not_fire(tmp_path):
    """No fall-through, no unconditional claim."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            route_reason = "unset"
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            elif not c1_passed:
                route_reason = "c1_rule_grades_at_r_star_unmet"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_no_reason_variable_at_all_does_not_fire(tmp_path):
    """~94% of the corpus never mentions a reason variable. This is also the case the
    cheap substring pre-filter short-circuits before any tree work, so this pins the
    pre-filter's verdict as well as the semantics."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                label = "criteria_met"
            else:
                label = "rule_did_not_beat_the_best_absolute_cap"
            return {"label": label, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


# ---- (4) robustness ---------------------------------------------------------------------

def test_exempt_marker_suppresses(tmp_path):
    p = _write(tmp_path, '''
        ROUTE_REASON_CONSISTENCY_EXEMPT = "c1 is a precondition routed away above"

        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    assert V.route_reason_consistency_lint(p) is None


def test_syntax_error_returns_none_not_raise(tmp_path):
    """Must carry a reason variable name so it gets PAST the substring pre-filter and
    actually reaches `ast.parse` -- otherwise it would pass for the wrong reason and
    leave the SyntaxError branch untested."""
    p = _write(tmp_path, 'route_reason = "x"\ndef broken(:\n')
    assert V.route_reason_consistency_lint(p) is None


def test_missing_file_returns_none_not_raise(tmp_path):
    assert V.route_reason_consistency_lint(tmp_path / "nope.py") is None


def test_message_is_ascii_only(tmp_path):
    """CLAUDE.md: anything reaching stdout must be ASCII (cp1252 mojibake on Windows)."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    issue = V.route_reason_consistency_lint(p)
    assert issue is not None
    issue.encode("ascii")


# ---- (5) corpus calibration --------------------------------------------------------------

def test_carrier_set_is_exactly_the_known_carrier(corpus_scan):
    """Pinned at ONE fire across the whole corpus, by name.

    A NEW driver appearing here means the fall-through mistake was copied forward into a
    fresh script -- fix that one (derive the reason from the failing conjuncts). The
    carrier LEAVING this set means either the landed 935 driver was retro-edited (it must
    not be -- its run is complete and already adjudicated) or the predicate silently
    stopped working.
    """
    expected = {CARRIER}
    actual = {p.name for p in corpus_scan["route_reason_consistency_lint"]}
    assert actual == expected, (
        f"carrier set changed: newly firing {sorted(actual - expected)}, "
        f"no longer firing {sorted(expected - actual)}")


def test_the_carrier_driver_exists(corpus_scan):
    """Guards the pin above against going vacuously green on a renamed file."""
    assert (EXPERIMENTS_DIR / CARRIER).exists(), f"missing {CARRIER}"


def test_the_467e_464e_instance_is_an_accepted_miss(corpus_scan):
    """The FIRST of the two confirmed instances is NOT this lint's shape.

    467e/464e emit a factually-false `external_task_mode_not_occupied` from an `elif not
    occupancy_non_vacuity_met:` branch -- self-consistent at the boolean level, false
    because the `min()` statistic behind the criterion cannot distinguish "0.0 everywhere"
    from "{0.0, 1.0}". Static analysis cannot see that. Pinned so that a future widening
    reaching these is a deliberate, reviewed change, and so this lint is never cited as
    having covered that instance.
    """
    fired = {p.name for p in corpus_scan["route_reason_consistency_lint"]}
    for name in ACCEPTED_MISSES:
        assert (EXPERIMENTS_DIR / name).exists(), f"missing {name}"
        assert name not in fired, f"{name} now fires -- update the docstring deliberately"


def test_generic_fall_through_corpus_drivers_are_clean(corpus_scan):
    """The load-bearing negative control: real drivers that reach the LAST funnel stage
    and are correctly spared there.

    Each of these hardcodes a fall-through reason under a guard that resolves to a
    >=2-conjunct AND naming >=2 declared criteria -- i.e. every precondition of the lint
    holds except the final distinctiveness test. They are spared only because their reason
    names the DISJUNCTION of causes rather than one criterion, which is correct authoring:
    `c3_c4_c5_not_all_met_genuine_weakens` (629b/629c) says exactly that, and
    `readiness_met_but_child_gate(s)_failed_or_disjoint` (715/715a) is generic by
    construction. 20 drivers reach this stage corpus-wide and exactly one fires.

    This is what stops the discriminator being widened back to "hardcoded literal on a
    fall-through", which would fire on all 20 and get the lint turned off.
    """
    conforming = {
        "v3_exq_629b_mech342_ecological_maintenance_release_evidence.py",
        "v3_exq_629c_mech342_ecological_maintenance_release_evidence.py",
        "v3_exq_715_sd034_decommit_science_closure_commit_entry_falsifier.py",
        "v3_exq_715a_sd034_decommit_science_selection_face_ceiling_lift_falsifier.py",
        "v3_exq_460n_closure_commit_entry_trajectory_readiness.py",
        "v3_exq_625d_sd037_axis_b_phase1b_joint_composite.py",
    }
    for name in conforming:
        assert (EXPERIMENTS_DIR / name).exists(), f"missing {name}"
    fired = {p.name for p in corpus_scan["route_reason_consistency_lint"]}
    assert not (conforming & fired), f"regressed: {sorted(conforming & fired)}"


def test_the_carriers_own_readiness_route_is_spared(corpus_scan):
    """Sharpest available control: the SAME driver, the SAME `else` block, one line apart.

    v3_exq_935:1135 sets `readiness_route = "cap_recalibration_is_seed_idiosyncratic"` and
    :1136 sets the offending `route_reason`. Both are hardcoded literals on the same
    fall-through under the same guard, so everything except the distinctiveness test is
    identical between them -- the readiness_route names the ROUTE generically and shares
    no criterion vocabulary. A regression that fired on both would report two findings for
    this file, so the exact-count assertion below is what carries the control.
    """
    fired = [p for p in corpus_scan["route_reason_consistency_lint"] if p.name == CARRIER]
    assert len(fired) == 1, f"expected the carrier once, got {fired}"
    issue = V.route_reason_consistency_lint(fired[0])
    assert issue is not None
    assert issue.count("singles out") == 1, f"readiness_route must be spared: {issue}"
    assert "cap_recalibration_is_seed_idiosyncratic" not in issue, issue


# ---- (6) wiring ---------------------------------------------------------------------------

def test_lint_is_registered_in_check_names():
    """An unregistered lint is dead code: `main()` gates every lint on `selected`."""
    assert "route_reason_consistency" in V.CHECK_NAMES


def test_lint_is_reachable_through_main(tmp_path, capsys):
    """Executes the real CLI over a synthetic carrier. Registration in CHECK_NAMES is
    necessary but not sufficient -- the main-loop branch and the report block are separate
    edits and any one of them can be missed."""
    p = _write(tmp_path, '''
        def adjudicate(c1_passed, c3_passed):
            if c1_passed and c3_passed:
                route_reason = "criteria_met"
            else:
                route_reason = "rule_did_not_beat_the_best_absolute_cap"
            return {"route_reason": route_reason, "criteria": [
                {"name": "C1_rule_grades_at_r_star", "passed": c1_passed},
                {"name": "C3_beats_best_absolute_cap", "passed": c3_passed},
            ]}
    ''')
    argv = sys.argv
    sys.argv = ["validate_experiments.py", "--checks", "route_reason_consistency",
                "--paths", str(p)]
    try:
        rc = V.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "ROUTE_REASON-CONSISTENCY WARNINGS" in out, out
    assert rc == 0, "WARN-only: must never harden, even under --paths"
