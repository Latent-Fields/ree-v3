"""Contracts for experiments/_lib/precondition_gate.py.

The governing requirement, from the V3-EXQ-785 autopsy (sections 2a and 8):

    A two-regime run where ONE arm is structurally degenerate and the other is
    clean must NOT vacate the clean arm.

The confirmed regression this locks down: 785's `harm_incumbent` arm passed all six
preconditions on 3959 committed ticks (rho -0.8303 vs a pre-registered +0.6, ~40 SE)
while its `entropy_incumbent` arm missed exactly one precondition
(n_components_with_nontrivial_share, 1.0 vs a 1.5 floor). A whole-run AND recorded
the run non_contributory / "substrate not ready" and buried the clean finding.
"""

import pytest

from experiments._lib.precondition_gate import (
    PreconditionSpec,
    StructurallyUnsatisfiableGate,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)

# --- the 785 fixture, with its real pre-registered numbers ------------------- #

HARM_ARM = {
    "id": "harm_incumbent",
    "expected_incumbent": "harm_weighted",
    "expected_incumbent_share": 0.606,
}
ENTROPY_ARM = {
    "id": "entropy_incumbent",
    "expected_incumbent": "CH:mech341",
    # >= 1.0: shares sum to 1.0 by construction, so every OTHER component sums
    # to -0.043 and none can clear the |0.01| floor. This number was committed in
    # the manifest config alongside the gate it made unsatisfiable.
    "expected_incumbent_share": 1.043,
}

AUTHORITY_FLOOR = 0.05
N_COMPONENTS_FLOOR = 1.5


# A third regime that is clean under every spec -- used where a test needs an
# all-green run without dragging in 785's deliberately vacuous entropy arm.
CLEAN_CHANNEL_ARM = {
    "id": "clean_channel",
    "expected_incumbent": "CH:other",
    "expected_incumbent_share": 0.44,
}


def _specs():
    """The CORRECT 785 gate: P1 gets disposition (a), P7 gets disposition (b).

    P1 is NOT MEANINGFUL in a primary-component regime -- the modulatory channels
    legitimately contribute ~0 there -- so it is scoped OUT via applies_to and the
    arm stays scorable.

    P7 IS meaningful everywhere: a single-component decomposition is arithmetically
    forced wherever it occurs. So it gets NO applies_to. Its structural_max instead
    marks the ARM vacuous, which keeps 785's forced rho +0.5879 out of scoring
    rather than letting it pass a gate and become citable.
    """
    return [
        PreconditionSpec(
            name="modulatory_authority_active_frac",
            description="fraction of selection ticks where the authority gate fired",
            control="candidates that genuinely differ",
            threshold=AUTHORITY_FLOOR,
            applies_to=lambda ctx: ctx["expected_incumbent"].startswith("CH:"),
            applies_note="channel-incumbent regimes only -- N/A when the incumbent "
                         "is a primary score component",
        ),
        PreconditionSpec(
            name="n_components_with_nontrivial_share",
            description="number of components holding |share| > 0.01",
            control="multi-component primary score",
            threshold=N_COMPONENTS_FLOOR,
            # no applies_to: this precondition is meaningful in EVERY regime
            structural_max=lambda ctx: (
                1.0 if ctx["expected_incumbent_share"] > 1.0 else None),
        ),
        PreconditionSpec(
            name="committed_tick_count",
            description="committed ticks available to score",
            control="driver drives the running-variance EMA per 396a",
            threshold=100.0,
        ),
    ]


def _unconditioned_specs():
    """The BUGGY shipped shape: P1 asserted everywhere, P7 with no structural proof."""
    specs = _specs()
    specs[0].applies_to = None
    specs[1].structural_max = None
    return specs


def _unconditioned_specs_with_structural_bound():
    """The buggy shape, but retaining P7's design-time proof."""
    specs = _specs()
    specs[0].applies_to = None
    return specs


def _gate(arm_ctx, specs, **measured):
    return evaluate_arm_gate(arm_ctx["id"], arm_ctx, specs, measured=measured)


# --- THE headline contract --------------------------------------------------- #

def test_structurally_degenerate_arm_does_not_vacate_the_clean_arm():
    """A red arm must not vacate a green one. This is the 785 regression."""
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0,
                 committed_tick_count=3959.0)
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.01,  # RED
                    n_components_with_nontrivial_share=1.0,
                    committed_tick_count=241.0)

    assert harm["gate_green"] is True
    assert entropy["gate_green"] is False

    agg = aggregate_arm_gates([harm, entropy])

    # The whole run is NOT vacuous -- one arm is clean and well powered.
    assert agg["non_degenerate"] is True
    assert agg["all_green"] is False
    assert agg["green_arms"] == ["harm_incumbent"]
    assert agg["red_arms"] == ["entropy_incumbent"]


def test_whole_run_and_would_have_vacated_it(  # documents the defect
):
    """Same inputs under the OLD whole-run AND: the clean arm is vacated.

    Kept as an executable statement of what regressing would mean.
    """
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0,
                 committed_tick_count=3959.0)
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.01,
                    n_components_with_nontrivial_share=1.0,
                    committed_tick_count=241.0)
    legacy_non_degenerate = all(g["gate_green"] for g in (harm, entropy))
    assert legacy_non_degenerate is False          # the bug
    assert aggregate_arm_gates([harm, entropy])["non_degenerate"] is True  # the fix


# --- regime conditioning ----------------------------------------------------- #

def test_precondition_is_scoped_out_of_a_regime_it_cannot_apply_to():
    """Disposition (a): P1 is scoped OUT of a primary-component regime, not failed.

    With it scoped out the arm can actually pass -- which is the point: a
    not-meaningful precondition must not make a regime structurally un-passable.
    """
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0,
                 committed_tick_count=3959.0)
    scoped = {s["precondition"] for s in harm["scoped_out"]}
    assert "modulatory_authority_active_frac" in scoped
    applied = {p["precondition"] for p in harm["preconditions"]}
    assert "modulatory_authority_active_frac" not in applied
    assert harm["gate_green"] is True
    assert harm["structurally_vacuous"] is False


def test_p1_stays_scoped_out_of_a_primary_component_regime():
    """The conditioning the 785 script already had must survive the refactor."""
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0,
                 committed_tick_count=3959.0)
    scoped = {s["precondition"] for s in harm["scoped_out"]}
    assert "modulatory_authority_active_frac" in scoped
    assert harm["gate_green"] is True


def test_unconditioned_gate_reproduces_the_785_failure():
    """Without conditioning, the entropy arm fails P7 -- structurally."""
    specs = _unconditioned_specs()
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.9,
                    n_components_with_nontrivial_share=1.0,  # forced by share 1.043
                    committed_tick_count=241.0)
    assert entropy["gate_green"] is False
    assert entropy["failed_preconditions"] == ["n_components_with_nontrivial_share"]


# --- design-time structural satisfiability (autopsy learning 2) -------------- #

def test_unsatisfiable_gate_is_refused_at_design_time():
    """`expected_incumbent_share = 1.043` vs a >= 2-components gate is a proof."""
    with pytest.raises(StructurallyUnsatisfiableGate) as exc:
        assert_no_structurally_unsatisfiable_gate(
            _unconditioned_specs_with_structural_bound(), [HARM_ARM, ENTROPY_ARM])
    msg = str(exc.value)
    assert "entropy_incumbent" in msg
    assert "n_components_with_nontrivial_share" in msg
    assert "applies_to" in msg  # points at the correct fix, not a lower threshold


def _unconditioned_specs_with_structural_bound():
    specs = _specs()
    specs[1].applies_to = None          # the bug: asserted everywhere ...
    return specs                        # ... but structural_max still proves it


def test_scoping_out_resolves_the_unsatisfiability():
    """Disposition (a) clears the audit: a scoped-out precondition cannot fail an arm.

    P1 carries a structural bound that no primary-component regime could ever meet.
    Left applicable it would refuse the run; scoped out via applies_to it is inert.
    """
    p1_with_bound = PreconditionSpec(
        name="modulatory_authority_active_frac",
        description="fraction of selection ticks where the authority gate fired",
        control="candidates that genuinely differ",
        threshold=AUTHORITY_FLOOR,
        # a primary-component regime can never fire the authority gate
        structural_max=lambda ctx: (
            0.0 if not ctx["expected_incumbent"].startswith("CH:") else None),
    )
    with pytest.raises(StructurallyUnsatisfiableGate):
        assert_no_structurally_unsatisfiable_gate([p1_with_bound], [HARM_ARM])

    p1_with_bound.applies_to = lambda ctx: ctx["expected_incumbent"].startswith("CH:")
    audited = assert_no_structurally_unsatisfiable_gate(
        [p1_with_bound], [HARM_ARM, CLEAN_CHANNEL_ARM])
    statuses = {(a["arm"], a["precondition"]): a["status"] for a in audited}
    assert statuses[
        ("harm_incumbent", "modulatory_authority_active_frac")] == "scoped_out"
    assert statuses[
        ("clean_channel", "modulatory_authority_active_frac")] == "satisfiable"


def test_correct_785_gate_still_flags_the_entropy_arm_as_vacuous():
    """Under the CORRECT gate the entropy arm must be acknowledged, not silently green."""
    with pytest.raises(StructurallyUnsatisfiableGate, match="entropy_incumbent"):
        assert_no_structurally_unsatisfiable_gate(_specs(), [HARM_ARM, ENTROPY_ARM])
    # acknowledged -> the run proceeds, with that arm excluded from scoring
    assert_no_structurally_unsatisfiable_gate(
        _specs(), [HARM_ARM, ENTROPY_ARM],
        acknowledged_vacuous_arms=["entropy_incumbent"])


def test_acknowledging_a_vacuous_arm_also_resolves_the_audit():
    """Disposition (b): the arm cannot be scored, and the author says so."""
    audited = assert_no_structurally_unsatisfiable_gate(
        _unconditioned_specs_with_structural_bound(), [HARM_ARM, ENTROPY_ARM],
        acknowledged_vacuous_arms=["entropy_incumbent"])
    statuses = {(a["arm"], a["precondition"]): a["status"] for a in audited}
    assert statuses[
        ("entropy_incumbent", "n_components_with_nontrivial_share")] == "unsatisfiable"


def test_run_with_every_arm_vacuous_is_refused_even_if_acknowledged():
    """A run that can produce no scorable result must not consume compute."""
    with pytest.raises(StructurallyUnsatisfiableGate, match="EVERY arm"):
        assert_no_structurally_unsatisfiable_gate(
            _unconditioned_specs_with_structural_bound(),
            [ENTROPY_ARM, dict(ENTROPY_ARM, id="entropy_two")],
            acknowledged_vacuous_arms=["entropy_incumbent", "entropy_two"])


# --- disposition (b): a vacuous arm must not become citable ------------------ #

def test_structurally_vacuous_arm_is_never_green():
    """P7 in the entropy regime is MEANINGFUL -- it detected a forced decomposition.

    Scoping the PRECONDITION out (disposition (a)) would let that arm pass its gate
    and make its arithmetically forced rho +0.5879 citable as support for MECH-463.
    That is a worse failure than burying the clean arm. The arm is scoped out
    instead.
    """
    specs = _unconditioned_specs_with_structural_bound()
    entropy = evaluate_arm_gate(
        "entropy_incumbent", ENTROPY_ARM, specs,
        measured={"modulatory_authority_active_frac": 0.9,
                  "n_components_with_nontrivial_share": 1.0,
                  "committed_tick_count": 3959.0})
    assert entropy["structurally_vacuous"] is True
    assert entropy["gate_green"] is False
    assert "artifact" in entropy["vacuity_reason"]


def test_vacuous_arm_is_detected_even_when_its_measurements_would_pass():
    """Auto-detection does not depend on the arm happening to fail numerically."""
    specs = _unconditioned_specs_with_structural_bound()
    entropy = evaluate_arm_gate(
        "entropy_incumbent", ENTROPY_ARM, specs,
        measured={"modulatory_authority_active_frac": 0.9,
                  "n_components_with_nontrivial_share": 99.0,  # would pass
                  "committed_tick_count": 3959.0})
    assert entropy["structurally_vacuous"] is True
    assert entropy["gate_green"] is False


def test_vacuous_arm_still_does_not_vacate_the_clean_arm():
    """Both halves of the rule hold at once: (b) excluded AND (a) preserved."""
    specs = _unconditioned_specs_with_structural_bound()
    harm = evaluate_arm_gate(
        "harm_incumbent", HARM_ARM, specs,
        measured={"n_components_with_nontrivial_share": 3.0,
                  "committed_tick_count": 3959.0})
    entropy = evaluate_arm_gate(
        "entropy_incumbent", ENTROPY_ARM, specs,
        measured={"modulatory_authority_active_frac": 0.9,
                  "n_components_with_nontrivial_share": 1.0,
                  "committed_tick_count": 3959.0})
    agg = aggregate_arm_gates([harm, entropy])

    assert agg["non_degenerate"] is True                  # clean arm survives
    assert agg["green_arms"] == ["harm_incumbent"]
    assert agg["vacuous_arms"] == ["entropy_incumbent"]
    # the vacuous arm is named as an artifact, not as a refutation
    assert "STRUCTURALLY VACUOUS" in agg["degeneracy_reason"]
    assert "do not cite" in agg["degeneracy_reason"].lower()
    # and it is kept out of the flat adjudication list
    assert {p["arm"] for p in agg["adjudication_preconditions"]} == {"harm_incumbent"}


def test_vacuous_arm_criteria_are_marked_degenerate():
    """The artifact must not reach claim scoring through the criterion channel."""
    specs = _unconditioned_specs_with_structural_bound()
    harm = evaluate_arm_gate(
        "harm_incumbent", HARM_ARM, specs,
        measured={"n_components_with_nontrivial_share": 3.0,
                  "committed_tick_count": 3959.0})
    entropy = evaluate_arm_gate(
        "entropy_incumbent", ENTROPY_ARM, specs,
        measured={"modulatory_authority_active_frac": 0.9,
                  "n_components_with_nontrivial_share": 1.0,
                  "committed_tick_count": 3959.0})
    agg = aggregate_arm_gates([harm, entropy])
    cnd = arm_criteria_non_degenerate(
        {"harm_incumbent": ["C_harm"], "entropy_incumbent": ["C_entropy"]}, agg)
    assert cnd["C_harm"] is True
    assert cnd["C_entropy"] is False


# --- attribution + manifest surfacing (task 3) ------------------------------- #

def test_failed_gate_names_which_arm_failed():
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0, committed_tick_count=3959.0)
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.01,
                    n_components_with_nontrivial_share=1.0,
                    committed_tick_count=241.0)
    agg = aggregate_arm_gates([harm, entropy])

    assert "entropy_incumbent" in agg["degeneracy_reason"]
    assert "modulatory_authority_active_frac" in agg["degeneracy_reason"]
    assert agg["per_arm_gate"]["failed_preconditions_by_arm"] == {
        "entropy_incumbent": ["modulatory_authority_active_frac",
                              "n_components_with_nontrivial_share"]}
    # the green arm is named as scored, so a reader cannot miss it
    assert "harm_incumbent" in agg["degeneracy_reason"]


def test_precondition_names_are_arm_namespaced():
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0, committed_tick_count=3959.0)
    for p in harm["preconditions"]:
        assert p["name"].startswith("harm_incumbent::")
        assert p["arm"] == "harm_incumbent"


def test_per_arm_gate_block_carries_red_arm_detail_at_top_level():
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0, committed_tick_count=3959.0)
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.01,
                    n_components_with_nontrivial_share=1.0,
                    committed_tick_count=241.0)
    block = aggregate_arm_gates([harm, entropy])["per_arm_gate"]

    red = {a["arm"]: a for a in block["red"]}
    assert "entropy_incumbent" in red
    # full precondition detail is preserved, not summarised away
    assert any(p["precondition"] == "modulatory_authority_active_frac"
               and p["met"] is False
               for p in red["entropy_incumbent"]["preconditions"])
    assert block["preconditions_scope_note"]


# --- the indexer contract ---------------------------------------------------- #

def test_partial_run_keeps_red_arm_out_of_flat_adjudication_list():
    """The indexer reads interpretation.preconditions flat and arm-blind.

    A red arm's unmet entry in that list re-vacates the green arm at adjudication
    time, reproducing 785 downstream of the script fix.
    """
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0, committed_tick_count=3959.0)
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.01,
                    n_components_with_nontrivial_share=1.0,
                    committed_tick_count=241.0)
    agg = aggregate_arm_gates([harm, entropy])

    flat = agg["adjudication_preconditions"]
    assert all(p["met"] for p in flat), "an unmet entry here vacates the whole run"
    assert {p["arm"] for p in flat} == {"harm_incumbent"}
    # but the red arm is NOT hidden -- it is at top level, and the note says so
    assert "entropy_incumbent" in agg["per_arm_gate"]["preconditions_scope_note"]


def test_all_red_run_carries_every_precondition_and_is_vacuous():
    """When no arm is clean the run really is vacuous -- the gate must still fire."""
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=1.0,   # RED
                 committed_tick_count=3959.0)
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.01,  # RED
                    n_components_with_nontrivial_share=1.0,
                    committed_tick_count=241.0)
    agg = aggregate_arm_gates([harm, entropy])

    assert agg["non_degenerate"] is False
    assert agg["green_arms"] == []
    flat = agg["adjudication_preconditions"]
    assert any(p["met"] is False for p in flat), "vacuity must reach the indexer"
    assert {p["arm"] for p in flat} == {"harm_incumbent", "entropy_incumbent"}


def test_all_green_run_is_non_degenerate_with_no_reason():
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0, committed_tick_count=3959.0)
    entropy = _gate(CLEAN_CHANNEL_ARM, specs,
                    modulatory_authority_active_frac=0.9,
                    n_components_with_nontrivial_share=3.0,
                    committed_tick_count=241.0)
    agg = aggregate_arm_gates([harm, entropy])
    assert agg["all_green"] is True
    assert agg["non_degenerate"] is True
    assert agg["degeneracy_reason"] == ""
    assert {p["arm"] for p in agg["adjudication_preconditions"]} == {
        "harm_incumbent", "clean_channel"}


# --- per-criterion separability ---------------------------------------------- #

def test_red_arm_criteria_are_marked_degenerate_and_green_arm_criteria_are_not():
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0, committed_tick_count=3959.0)
    entropy = _gate(ENTROPY_ARM, specs,
                    modulatory_authority_active_frac=0.01,
                    n_components_with_nontrivial_share=1.0,
                    committed_tick_count=241.0)
    agg = aggregate_arm_gates([harm, entropy])

    cnd = arm_criteria_non_degenerate(
        {"harm_incumbent": ["C_harm_incumbent_incumbent_share_rises_with_urgency"],
         "entropy_incumbent": ["C_entropy_incumbent_incumbent_share_rises_with_urgency"]},
        agg)
    assert cnd["C_harm_incumbent_incumbent_share_rises_with_urgency"] is True
    assert cnd["C_entropy_incumbent_incumbent_share_rises_with_urgency"] is False


def test_power_shortfall_can_degrade_a_green_arms_criterion():
    """A green gate does not force non-degeneracy if the criterion lacks power."""
    specs = _specs()
    harm = _gate(HARM_ARM, specs,
                 n_components_with_nontrivial_share=3.0, committed_tick_count=3959.0)
    agg = aggregate_arm_gates([harm])
    cnd = arm_criteria_non_degenerate(
        {"harm_incumbent": ["C_x"]}, agg, extra={"C_x": False})
    assert cnd["C_x"] is False


# --- guard rails ------------------------------------------------------------- #

def test_applying_precondition_without_a_measurement_is_an_error():
    """A silently-absent measurement is exactly the vacuity these gates catch."""
    with pytest.raises(KeyError):
        _gate(HARM_ARM, _specs(), committed_tick_count=3959.0)  # P7 missing


def test_ceiling_direction_is_not_read_as_a_floor():
    """The 648a/649 directionality bug must not reappear through this module."""
    spec = PreconditionSpec(
        name="zworld_bounded", description="stayed below the explosion ceiling",
        control="rollout clamp", threshold=1e6, direction="upper")
    gate = evaluate_arm_gate("a", {"id": "a"}, [spec], measured={"zworld_bounded": 0.19})
    assert gate["gate_green"] is True
    assert gate["preconditions"][0]["direction"] == "upper"


def test_met_override_is_honoured_for_non_threshold_pass_conditions():
    spec = PreconditionSpec(name="identity", description="incumbent is as pre-registered",
                            control="measured 2026-07-18", threshold=0.05)
    gate = evaluate_arm_gate("a", {"id": "a"}, [spec],
                             measured={"identity": 0.9},
                             met_overrides={"identity": False})
    assert gate["gate_green"] is False
    assert gate["preconditions"][0]["measured"] == 0.9  # numeric leg still emitted
