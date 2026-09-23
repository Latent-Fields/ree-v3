"""Contracts for the cannot-determine category in precondition_gate.py.

THE DEFECT THESE LOCK DOWN
--------------------------
`assert_no_structurally_unsatisfiable_gate` detects an unsatisfiable precondition
ONLY via the optional `structural_max` / `structural_min` lambdas. Until
2026-09-22 a spec declaring neither was audited as `"satisfiable"` -- identical
to a spec that was checked and found fine. The guard therefore ran, passed, and
proved nothing, silently.

Measured 2026-09-23 at cbfd7c1e41, by AST (counting PreconditionSpec keyword
arguments, not by grepping for the token): 108 drivers call the guard, 41 declare
a structural bound, 67 (62%) declare neither; at spec level 110 of 526 (20.9%)
carry a bound. A grep for the token reports 43/65 -- it over-counts by 3, because
a file whose prose says it CANNOT be bounded still contains the string. V3-EXQ-1062 is the
confirmed instance -- all eight of its PreconditionSpecs omit the bounds, so the
guard could not fire even though `fresh_select_sample_floor` (200) was
structurally unreachable under that run's own --dry-run P2 budget of 60 steps.
See `REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1062_2026-09-22.md`
section 6, and CLAUDE.md General Rules, "Negative instruments".

THE BLIND-SPOT MEASUREMENT (CLAUDE.md: "a guard that supplies the thing it
asserts is not a guard")
------------------------------------------------------------------------------
Every test below was run against the PRE-CHANGE module before landing.
`test_unbounded_spec_is_not_evaluated_not_satisfiable` and the three that follow
it FAIL there (old code returns `"satisfiable"`, and neither
`summarize_structural_audit` nor `structural_vacuity_verdict` exists), while the
whole of `test_precondition_gate_regime_conditioning.py` PASSES there -- i.e.
the old guard was blind to exactly this, and the new tests are not restating
something already covered.
"""

import pytest

from experiments._lib.precondition_gate import (
    STRUCTURAL_STATUSES,
    PreconditionSpec,
    StructurallyUnsatisfiableGate,
    assert_no_structurally_unsatisfiable_gate,
    detect_structural_vacuity,
    format_structural_audit_report,
    structural_vacuity_verdict,
    summarize_structural_audit,
)

# --- fixtures ---------------------------------------------------------------- #

ARM = {"id": "p2", "n_steps": 60}
OTHER_ARM = {"id": "p3", "n_steps": 4000}


def _bare_spec(name="fresh_select_sample_floor", threshold=200.0, direction="lower"):
    """The V3-EXQ-1062 shape: a real floor, and NO structural bound declared."""
    return PreconditionSpec(
        name=name,
        description="fresh selection samples drawn in this phase",
        control="a phase long enough to draw them",
        threshold=threshold,
        direction=direction,
    )


def _bounded_spec(bound, threshold=200.0):
    spec = _bare_spec(threshold=threshold)
    spec.structural_max = lambda ctx, _b=bound: _b
    return spec


def _status_map(audited):
    return {(a["arm"], a["precondition"]): a["status"] for a in audited}


# --- (1) the category exists and is not collapsed ---------------------------- #

def test_unbounded_spec_is_not_evaluated_not_satisfiable():
    """THE regression. A spec with no bound was NOT checked -- say so."""
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec()], [ARM], report=False)
    assert _status_map(audited)[("p2", "fresh_select_sample_floor")] == "not_evaluated"


def test_not_evaluated_names_the_missing_bound():
    """The reason must be actionable, not a bare flag."""
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec()], [ARM], report=False)
    reason = audited[0]["reason"]
    assert "structural_max" in reason
    assert "NOT checked" in reason


def test_ceiling_precondition_names_structural_min_instead():
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec(direction="upper")], [ARM], report=False)
    assert audited[0]["status"] == "not_evaluated"
    assert "structural_min" in audited[0]["reason"]


def test_bound_returning_none_for_this_arm_is_also_not_evaluated():
    """A declared lambda that yields no bound HERE proved nothing HERE either.

    This is the subtler half: the spec looks checked at the call site, but for
    this particular arm there was no bound to reason from.
    """
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bounded_spec(None)], [ARM], report=False)
    assert audited[0]["status"] == "not_evaluated"
    assert "no structural bound is derivable" in audited[0]["reason"]


def test_every_status_is_a_declared_category():
    spec_out = _bare_spec(name="scoped")
    spec_out.applies_to = lambda ctx: ctx["id"] != "p2"
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec(), _bounded_spec(500.0), spec_out], [ARM, OTHER_ARM],
        report=False)
    assert {a["status"] for a in audited} <= set(STRUCTURAL_STATUSES)
    assert all("reason" in a for a in audited)


# --- (2) the existing verdicts are unchanged --------------------------------- #

def test_declared_and_clearing_bound_is_still_satisfiable():
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bounded_spec(500.0)], [ARM], report=False)
    assert audited[0]["status"] == "satisfiable"


def test_declared_and_failing_bound_is_still_refused():
    """The 1062 arithmetic, made explicit: 60 steps cannot reach a 200 floor."""
    with pytest.raises(StructurallyUnsatisfiableGate) as exc:
        assert_no_structurally_unsatisfiable_gate(
            [_bounded_spec(60.0)], [ARM, OTHER_ARM], report=False)
    assert "fresh_select_sample_floor" in str(exc.value)
    assert "p2" in str(exc.value)


def test_scoped_out_is_still_scoped_out():
    spec = _bare_spec()
    spec.applies_to = lambda ctx: False
    audited = assert_no_structurally_unsatisfiable_gate([spec], [ARM], report=False)
    assert audited[0]["status"] == "scoped_out"


def test_unbounded_spec_does_not_block_the_run():
    """Report, do not block -- 60% of call sites would fail at once (CLAUDE.md)."""
    assert_no_structurally_unsatisfiable_gate(
        [_bare_spec(), _bare_spec(name="other")], [ARM, OTHER_ARM], report=False)


# --- (3) the printed denominator --------------------------------------------- #

def test_summary_reports_zero_coverage_for_a_bare_spec_set():
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec(), _bare_spec(name="other")], [ARM, OTHER_ARM], report=False)
    summary = summarize_structural_audit(audited)
    assert summary["n_applicable"] == 4
    assert summary["n_evaluated"] == 0
    assert summary["n_not_evaluated"] == 4
    assert summary["coverage"] == 0.0
    assert summary["proved_nothing"] is True


def test_summary_reports_partial_coverage():
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec(), _bounded_spec(500.0, threshold=1.0)], [ARM], report=False)
    summary = summarize_structural_audit(audited)
    assert (summary["n_evaluated"], summary["n_applicable"]) == (1, 2)
    assert summary["coverage"] == pytest.approx(0.5)
    assert summary["proved_nothing"] is False


def test_scoped_out_pairs_are_outside_the_denominator():
    """A precondition that is N/A here was not skipped for want of a bound."""
    spec = _bare_spec(name="scoped")
    spec.applies_to = lambda ctx: False
    audited = assert_no_structurally_unsatisfiable_gate(
        [spec, _bounded_spec(500.0, threshold=1.0)], [ARM], report=False)
    summary = summarize_structural_audit(audited)
    assert summary["n_scoped_out"] == 1
    assert summary["n_applicable"] == 1
    assert summary["coverage"] == 1.0


def test_zero_coverage_run_says_it_proved_nothing(capsys):
    assert_no_structurally_unsatisfiable_gate([_bare_spec()], [ARM])
    out = capsys.readouterr().out
    assert "NOT EVALUATED" in out
    assert "PROVED NOTHING" in out


def test_fully_covered_run_prints_no_coverage_warning(capsys):
    assert_no_structurally_unsatisfiable_gate(
        [_bounded_spec(500.0, threshold=1.0)], [ARM])
    out = capsys.readouterr().out
    assert "PROVED NOTHING" not in out
    assert "COVERAGE" not in out


def test_report_can_be_silenced(capsys):
    assert_no_structurally_unsatisfiable_gate([_bare_spec()], [ARM], report=False)
    assert capsys.readouterr().out == ""


def test_report_lines_are_ascii_only():
    """Windows cp1252 terminals -- CLAUDE.md 'ASCII-Only in Python Output'."""
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec(), _bounded_spec(500.0, threshold=1.0)], [ARM], report=False)
    for line in format_structural_audit_report(summarize_structural_audit(audited)):
        line.encode("ascii")


# --- (4) the same split on the per-arm vacuity path -------------------------- #

def test_vacuity_verdict_separates_not_evaluated_from_clear():
    """`detect_structural_vacuity` returns None for BOTH; the verdict does not."""
    assert detect_structural_vacuity([_bare_spec()], ARM) is None
    assert detect_structural_vacuity([_bounded_spec(500.0)], ARM) is None

    assert structural_vacuity_verdict([_bare_spec()], ARM)["verdict"] == "not_evaluated"
    assert structural_vacuity_verdict([_bounded_spec(500.0)], ARM)["verdict"] == "clear"


def test_vacuity_verdict_still_reports_vacuous():
    verdict = structural_vacuity_verdict([_bounded_spec(60.0)], ARM)
    assert verdict["verdict"] == "vacuous"
    assert "fresh_select_sample_floor" in verdict["vacuity_reason"]


def test_vacuity_verdict_carries_its_own_denominator():
    scoped = _bare_spec(name="scoped")
    scoped.applies_to = lambda ctx: False
    verdict = structural_vacuity_verdict(
        [scoped, _bare_spec(), _bounded_spec(500.0, threshold=1.0)], ARM)
    assert verdict["n_specs"] == 3
    assert verdict["n_applicable"] == 2
    assert verdict["n_evaluated"] == 1
    assert verdict["n_not_evaluated"] == 1
    assert verdict["verdict"] == "clear"  # one real check cleared, one unchecked


def test_a_mix_of_unchecked_and_unsatisfiable_is_vacuous_not_not_evaluated():
    """An actual proof outranks the absence of one."""
    verdict = structural_vacuity_verdict([_bare_spec(), _bounded_spec(60.0)], ARM)
    assert verdict["verdict"] == "vacuous"


# --- (5) the audit survives into a manifest ---------------------------------- #

def test_audit_rows_are_json_serialisable():
    """787 and 970a dump this list straight into the manifest."""
    import json
    audited = assert_no_structurally_unsatisfiable_gate(
        [_bare_spec(), _bounded_spec(500.0, threshold=1.0)], [ARM], report=False)
    payload = json.loads(json.dumps({
        "structural_gate_audit": audited,
        "structural_gate_coverage": summarize_structural_audit(audited),
    }))
    statuses = [r["status"] for r in payload["structural_gate_audit"]]
    assert "not_evaluated" in statuses
    assert payload["structural_gate_coverage"]["coverage"] == pytest.approx(0.5)
