"""Contracts for experiments/_lib/capability_contract.py (GOV-CAPCONTRACT-1).

The governing requirement, from the claim as registered (REE_assembly
docs/claims/claims.yaml, `cbac6ceea6`):

    A run is not admissible as negative evidence for a claim unless the organism
    instantiated for that run was demonstrably able to express -- and, where the
    claim is learning-dependent, to acquire -- the faculty under test. A run
    failing any declared precondition self-routes to a capability-precondition
    diagnostic status and MUST NOT be recorded as evidence against the claim.
    "It did not learn" and "it could not have learned" are different scientific
    outcomes.

FOUR PROPERTIES THIS FILE EXISTS TO PIN, each of which is a way the machinery
could silently invert into the failure it was built to prevent:

  1. NEGATIVE CONTROL -- a properly-instantiated, plastic, authoritative organism
     must come back `interpretable` with NOTHING undetermined. Without this the
     gate is a rubber stamp that routes every run to a diagnostic status and
     quietly empties the evidence record.
  2. An unmet contract must NEVER be recordable as `outcome: "FAIL"`.
  3. UNDETERMINED must never read as satisfied and must never block a run.
  4. The whole-run verdict must not be reachable from ARM-scoped facts -- the
     V3-EXQ-785 shape that precondition_gate.py exists to prevent.

Time-independent: no wall clock, no RNG, no torch, no live organism.
"""

import pytest

from experiments._lib.capability_contract import (
    MANIFEST_KEY,
    PLASTICITY_MODES,
    ROUTE_AUTHORITY_FLOOR_UNMET,
    ROUTE_CAPABILITY_PRECONDITION_UNMET,
    ROUTE_INTERPRETABLE,
    ROUTE_MECHANISM_UNREACHED,
    ROUTE_NONPLASTIC_MISFIRE,
    ROUTE_PRECEDENCE,
    ROUTED_OUTCOME,
    STATUS_SATISFIED,
    STATUS_UNDETERMINED,
    STATUS_UNMET,
    CapabilityContract,
    CapabilityContractDeclarationError,
    CapabilityRequirement,
    MechanismRequirement,
    PlasticityRequirement,
    as_adjudication_preconditions,
    assert_not_recorded_as_fail,
    capture_parameter_witness,
    compare_parameter_witness,
    format_report,
    learning_mode_snapshot,
    optimizer_membership,
    route_for_manifest,
    verify_capability_contract,
)


# --- fixtures: a duck-typed organism, optimizer and parameters --------------- #


class _Latent:
    def __init__(self, e3_enabled=True):
        self.e3_enabled = e3_enabled


class _Organism:
    """Minimal stand-in for an instantiated agent. No torch, no ree_core."""

    def __init__(self, e3_enabled=True):
        self.latent = _Latent(e3_enabled=e3_enabled)


class _FakeOptimizer:
    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]


class _Param(list):
    """A parameter that behaves like a flat float sequence (duck-typed tensor)."""


def _params(*rows):
    return [_Param(r) for r in rows]


HEALTHY_ENGAGEMENT = {
    "e3_selection_ticks_frac": 0.21,
    "e3_committed_share": 0.08,
}


def _mech(**overrides):
    kwargs = dict(
        name="e3_selector",
        description="E3 candidate selection must exist, run, and compete",
        config_flag="latent.e3_enabled",
        reached_metric="e3_selection_ticks_frac",
        reached_floor=0.05,
        authority_metric="e3_committed_share",
        authority_floor=0.02,
        claim_ids=("MECH-463",),
    )
    kwargs.update(overrides)
    return MechanismRequirement(**kwargs)


def _contract(**overrides):
    kwargs = dict(
        experiment_id="V3-EXQ-TEST",
        canonical_profile="ree_v3_baseline@v0",
        canonical_profile_hash="deadbeef",
        requires_mechanisms=[_mech()],
        requires_plasticity=[
            PlasticityRequirement(mode="parameters",
                                  description="policy head must be trainable"),
        ],
        claim_ids=("MECH-463",),
    )
    kwargs.update(overrides)
    return CapabilityContract(**kwargs)


def _plastic_run(**overrides):
    """A fully healthy verification: capable, authoritative, and genuinely plastic."""
    before_params = _params([0.0, 1.0, 2.0, 3.0], [4.0, 5.0])
    after_params = _params([0.1, 1.1, 2.1, 3.1], [4.1, 5.1])
    kwargs = dict(
        agent=_Organism(e3_enabled=True),
        engagement=dict(HEALTHY_ENGAGEMENT),
        optimizers=[_FakeOptimizer(before_params)],
        trainable_params=before_params,
        parameter_witness_before=capture_parameter_witness(before_params),
        parameter_witness_after=capture_parameter_witness(after_params),
        learning_mode=learning_mode_snapshot(grad_enabled=True),
        observed_profile="ree_v3_baseline@v0",
        observed_profile_hash="deadbeef",
    )
    kwargs.update(overrides)
    return kwargs


def _check_named(result, name):
    for check in result["checks"]:
        if check["name"] == name:
            return check
    raise AssertionError(f"no check named {name!r} in "
                         f"{[c['name'] for c in result['checks']]}")


# --- 1. THE NEGATIVE CONTROL ------------------------------------------------- #


def test_properly_instantiated_plastic_run_is_interpretable():
    """The load-bearing negative control: a healthy organism must pass cleanly.

    Without this, a gate that routed EVERY run to a diagnostic status would
    satisfy every other test in this file while silently emptying the evidence
    record -- the exact inverse of the failure GOV-CAPCONTRACT-1 addresses.
    """
    result = verify_capability_contract(_contract(), **_plastic_run())

    assert result["satisfied"] is True
    assert result["interpretation_route"] == ROUTE_INTERPRETABLE
    assert result["routes_triggered"] == []
    assert result["unmet_checks"] == []


def test_healthy_run_has_nothing_undetermined():
    """Stronger form of the control: every declared check was actually VERIFIED.

    `satisfied` alone is not enough -- a contract whose probes all returned
    UNDETERMINED is also `satisfied`, and would be a rubber stamp of a different
    kind. This pins that the healthy fixture determines every check.
    """
    result = verify_capability_contract(_contract(), **_plastic_run())

    assert result["has_undetermined"] is False, result["undetermined_checks"]
    assert result["undetermined_checks"] == []
    assert all(c["status"] == STATUS_SATISFIED for c in result["checks"])


def test_healthy_run_routes_to_no_manifest_changes():
    """A met contract changes nothing about how the run is recorded."""
    result = verify_capability_contract(_contract(), **_plastic_run())
    assert route_for_manifest(result) == {}


def test_healthy_run_may_still_be_recorded_as_fail():
    """A GENUINE negative result must remain recordable -- outcome (a).

    The contract only forbids a FAIL from an INCAPABLE organism. If it forbade
    every FAIL it would make the evidence record unfalsifiable, which is a worse
    epistemic failure than the one it fixes.
    """
    result = verify_capability_contract(_contract(), **_plastic_run())
    assert_not_recorded_as_fail({"outcome": "FAIL"}, result)  # must not raise


# --- 2. the four unmet routes ------------------------------------------------ #


def test_mechanism_disabled_routes_capability_precondition_unmet():
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    assert result["satisfied"] is False
    assert result["interpretation_route"] == ROUTE_CAPABILITY_PRECONDITION_UNMET
    assert _check_named(result, "e3_selector::enabled")["status"] == STATUS_UNMET


def test_mechanism_enabled_but_never_reached_routes_mechanism_unreached():
    """Constructed and enabled, but the decisive readout never engaged."""
    run = _plastic_run(engagement={"e3_selection_ticks_frac": 0.0,
                                   "e3_committed_share": 0.08})
    result = verify_capability_contract(_contract(), **run)

    assert result["interpretation_route"] == ROUTE_MECHANISM_UNREACHED
    assert _check_named(result, "e3_selector::enabled")["status"] == STATUS_SATISFIED


def test_reached_but_below_authority_floor_routes_authority_floor_unmet():
    """ARC-130: non-zero influence is not necessarily competitive influence."""
    run = _plastic_run(engagement={"e3_selection_ticks_frac": 0.21,
                                   "e3_committed_share": 0.001})
    result = verify_capability_contract(_contract(), **run)

    assert result["interpretation_route"] == ROUTE_AUTHORITY_FLOOR_UNMET
    assert _check_named(result, "e3_selector::reached")["status"] == STATUS_SATISFIED


def test_gradients_disabled_routes_nonplastic_misfire():
    """The fifth GOV-FAILLOC-1 bucket: capable organism, frozen parameters."""
    run = _plastic_run(learning_mode=learning_mode_snapshot(grad_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    assert result["satisfied"] is False
    assert result["interpretation_route"] == ROUTE_NONPLASTIC_MISFIRE
    check = _check_named(result, "plasticity::parameters::gradients_enabled")
    assert check["status"] == STATUS_UNMET
    assert "could not have learned" in check["detail"]


def test_parameter_delta_witness_showing_no_change_routes_nonplastic_misfire():
    """Gradients on, optimizer correct, and still no actual update path."""
    frozen = _params([0.0, 1.0, 2.0, 3.0], [4.0, 5.0])
    run = _plastic_run(
        parameter_witness_before=capture_parameter_witness(frozen),
        parameter_witness_after=capture_parameter_witness(frozen),
    )
    result = verify_capability_contract(_contract(), **run)

    assert result["interpretation_route"] == ROUTE_NONPLASTIC_MISFIRE
    check = _check_named(result,
                         "plasticity::parameters::parameter_delta_witness")
    assert check["status"] == STATUS_UNMET


def test_intended_parameters_absent_from_optimizer_routes_nonplastic_misfire():
    other = _params([9.0, 9.0])
    intended = _params([0.0, 1.0])
    run = _plastic_run(optimizers=[_FakeOptimizer(other)],
                       trainable_params=intended)
    result = verify_capability_contract(_contract(), **run)

    assert result["interpretation_route"] == ROUTE_NONPLASTIC_MISFIRE
    membership = _check_named(result,
                              "plasticity::parameters::optimizer_membership")
    assert membership["status"] == STATUS_UNMET


# --- 3. route precedence summarises; it never hides -------------------------- #


def test_most_upstream_route_wins_but_every_route_is_listed():
    """A mis-instantiated AND frozen organism names the mechanism first.

    Precedence exists so an autopsy is not sent looking for an optimizer bug in a
    run that had no mechanism to train. It must summarise, not conceal: both
    routes stay visible in `routes_triggered`.
    """
    run = _plastic_run(agent=_Organism(e3_enabled=False),
                       learning_mode=learning_mode_snapshot(grad_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    assert result["interpretation_route"] == ROUTE_CAPABILITY_PRECONDITION_UNMET
    assert ROUTE_NONPLASTIC_MISFIRE in result["routes_triggered"]
    assert result["routes_triggered"] == [
        r for r in ROUTE_PRECEDENCE if r in result["routes_triggered"]]


def test_route_precedence_is_the_arc130_ladder_order():
    assert ROUTE_PRECEDENCE == (
        ROUTE_CAPABILITY_PRECONDITION_UNMET,
        ROUTE_MECHANISM_UNREACHED,
        ROUTE_AUTHORITY_FLOOR_UNMET,
        ROUTE_NONPLASTIC_MISFIRE,
    )


def test_non_load_bearing_plasticity_cannot_trigger_nonplastic_misfire():
    """Declaring what a run ALLOWED to change is not declaring what it DEPENDED on."""
    contract = _contract(requires_plasticity=[
        PlasticityRequirement(mode="residue_ema_state", load_bearing=False,
                              requires_gradients=True)])
    run = _plastic_run(learning_mode=learning_mode_snapshot(grad_enabled=False))
    result = verify_capability_contract(contract, **run)

    assert ROUTE_NONPLASTIC_MISFIRE not in result["routes_triggered"]
    assert result["interpretation_route"] == ROUTE_INTERPRETABLE
    check = _check_named(result,
                         "plasticity::residue_ema_state::gradients_enabled")
    assert check["status"] == STATUS_UNMET, "still reported, just not fatal"


# --- 4. AN UNMET CONTRACT IS NOT A FAIL -------------------------------------- #


def test_route_for_manifest_never_yields_a_fail_outcome():
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)
    fields = route_for_manifest(result)

    assert fields["outcome"] != "FAIL"
    assert fields["outcome"] == "DIAGNOSTIC"


def test_route_for_manifest_sets_both_scoring_exclusion_channels():
    """The two channels build_experiment_indexes.py already honours.

    experiment_purpose="diagnostic"       -> scoring_excluded="diagnostic_probe"
    evidence_direction="non_contributory" -> scoring_excluded="non_contributory"

    Both, because they exclude at different granularities (run-wide vs per tagged
    claim) and an incapable organism must be excluded at both.
    """
    run = _plastic_run(learning_mode=learning_mode_snapshot(grad_enabled=False))
    result = verify_capability_contract(_contract(), **run)
    fields = route_for_manifest(result)

    assert fields["experiment_purpose"] == "diagnostic"
    assert fields["evidence_direction"] == "non_contributory"
    assert fields["non_degenerate"] is False
    assert "admissible as negative evidence" in fields["degeneracy_reason"]
    assert "could not have learned" in fields["degeneracy_reason"]


def test_assert_not_recorded_as_fail_refuses_an_unmet_run():
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    with pytest.raises(CapabilityContractDeclarationError) as exc:
        assert_not_recorded_as_fail({"outcome": "FAIL"}, result)
    assert "not negative evidence" in str(exc.value)


def test_assert_not_recorded_as_fail_accepts_the_routed_manifest():
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)
    manifest = {"outcome": "PASS"}
    manifest.update(route_for_manifest(result))

    assert_not_recorded_as_fail(manifest, result)  # must not raise


def test_the_second_door_a_non_pass_outcome_with_no_direction_is_refused():
    """The quiet prohibited recording: `weakens` INFERRED, no FAIL anywhere.

    build_experiment_indexes.py:
        if inferred_direction == "unknown" and not direction_explicitly_set:
            inferred_direction = "supports" if final_status == "PASS" else "weakens"

    So a driver that takes the `outcome` override and drops `evidence_direction`
    lands the run against the claim as `weakens`. That is the exact recording
    GOV-CAPCONTRACT-1 forbids, reached without the word FAIL, and it is the reason
    route_for_manifest() returns the fields as one bundle.
    """
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    half_applied = {"outcome": ROUTED_OUTCOME}  # direction dropped
    with pytest.raises(CapabilityContractDeclarationError) as exc:
        assert_not_recorded_as_fail(half_applied, result)
    assert "weakens" in str(exc.value)


def test_the_second_door_also_catches_an_explicitly_weakening_direction():
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    with pytest.raises(CapabilityContractDeclarationError):
        assert_not_recorded_as_fail(
            {"outcome": "PASS", "evidence_direction": "weakens"}, result)


def test_inconclusive_is_an_accepted_excluded_direction():
    """Both directions the indexer maps to scoring_excluded are honoured."""
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    assert_not_recorded_as_fail(
        {"outcome": ROUTED_OUTCOME, "evidence_direction": "inconclusive"}, result)


def test_superseded_is_not_an_accepted_excluded_direction():
    """It excludes, but it misattributes WHY -- a different fact from incapacity."""
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    result = verify_capability_contract(_contract(), **run)

    with pytest.raises(CapabilityContractDeclarationError):
        assert_not_recorded_as_fail(
            {"outcome": ROUTED_OUTCOME, "evidence_direction": "superseded"}, result)


def test_routed_outcome_is_neither_pass_nor_fail():
    """Both are already read as scientific verdicts; this run has none.

    A distinct token cannot be silently misread as either, and the indexer
    branches on `outcome` only for "ERROR", so it is inert there.
    """
    assert ROUTED_OUTCOME not in ("PASS", "FAIL", "ERROR")


def test_verify_never_raises_on_a_wholly_unmet_contract():
    """Raising would invite a caller-side except that records FAIL."""
    contract = _contract(canonical_profile="other@v1")
    run = _plastic_run(agent=_Organism(e3_enabled=False),
                       engagement={},
                       optimizers=None,
                       learning_mode=learning_mode_snapshot(grad_enabled=False))
    result = verify_capability_contract(contract, **run)
    assert result["satisfied"] is False


# --- 5. FAIL OPEN, LOUDLY: undetermined is neither satisfied nor blocking ----- #


def test_missing_config_flag_is_undetermined_not_unmet():
    """A renamed config field must not read as 'the mechanism was off'.

    That would manufacture a mis-instantiation verdict out of a refactor, and
    route a perfectly good run out of the evidence record.
    """
    class _NoLatent:
        pass

    run = _plastic_run(agent=_NoLatent())
    result = verify_capability_contract(_contract(), **run)

    check = _check_named(result, "e3_selector::enabled")
    assert check["status"] == STATUS_UNDETERMINED
    assert result["satisfied"] is True
    assert result["interpretation_route"] == ROUTE_INTERPRETABLE
    assert result["has_undetermined"] is True


def test_unmeasured_metric_is_undetermined_not_unmet():
    run = _plastic_run(engagement={})
    result = verify_capability_contract(_contract(), **run)

    assert _check_named(result, "e3_selector::reached")["status"] == STATUS_UNDETERMINED
    assert result["satisfied"] is True
    assert "e3_selector::reached" in result["undetermined_checks"]


def test_a_probe_that_raises_is_undetermined_not_unmet():
    def _explodes(_agent):
        raise RuntimeError("probe is broken")

    contract = _contract(requires_mechanisms=[
        _mech(constructed=_explodes, config_flag=None,
              reached_metric=None, authority_metric=None,
              authority_floor=None)])
    result = verify_capability_contract(contract, **_plastic_run())

    check = _check_named(result, "e3_selector::constructed")
    assert check["status"] == STATUS_UNDETERMINED
    assert "RuntimeError" in check["detail"]
    assert result["satisfied"] is True, "a broken probe must not block a run"


def test_a_probe_returning_none_is_undetermined():
    contract = _contract(requires_mechanisms=[
        _mech(constructed=lambda _a: None, config_flag=None,
              reached_metric=None, authority_metric=None, authority_floor=None)])
    result = verify_capability_contract(contract, **_plastic_run())
    assert _check_named(result,
                        "e3_selector::constructed")["status"] == STATUS_UNDETERMINED


def test_nan_metric_is_undetermined_not_unmet():
    run = _plastic_run(engagement={"e3_selection_ticks_frac": float("nan"),
                                   "e3_committed_share": 0.08})
    result = verify_capability_contract(_contract(), **run)
    assert _check_named(result, "e3_selector::reached")["status"] == STATUS_UNDETERMINED


def test_undetermined_checks_are_named_in_the_manifest_block():
    """"Said so in the manifest" is the whole point of failing open LOUDLY."""
    run = _plastic_run(engagement={})
    result = verify_capability_contract(_contract(), **run)
    block = result["manifest_block"]

    assert block["has_undetermined"] is True
    assert block["undetermined_checks"] == result["undetermined_checks"]
    assert block["undetermined_checks"], "the gap must be visible, not merely counted"


# --- 6. the vacuous-True inversions ------------------------------------------ #


def test_no_optimizer_is_undetermined_not_a_pass():
    assert optimizer_membership(None, _params([1.0]))["all_present"] is None
    assert optimizer_membership([], _params([1.0]))["all_present"] is None


def test_no_intended_parameters_is_undetermined_not_a_vacuous_pass():
    """A vacuous True here would certify a frozen organism as plastic."""
    result = optimizer_membership([_FakeOptimizer(_params([1.0]))], [])
    assert result["all_present"] is None
    assert "no intended trainable parameters" in result["reason"]


def test_missing_witness_endpoint_is_undetermined_not_no_change():
    """'no witness' and 'witnessed no change' are exactly what must not collapse."""
    witness = capture_parameter_witness(_params([1.0, 2.0]))
    assert compare_parameter_witness(None, witness)["changed"] is None
    assert compare_parameter_witness(witness, None)["changed"] is None
    assert compare_parameter_witness(None, None)["changed"] is None


def test_witness_shape_change_is_undetermined_not_a_change():
    before = capture_parameter_witness(_params([1.0, 2.0]))
    after = capture_parameter_witness(_params([1.0, 2.0], [3.0]))
    comparison = compare_parameter_witness(before, after)
    assert comparison["changed"] is None
    assert "not comparable" in comparison["reason"]


def test_unreadable_parameters_yield_no_readable_sample_not_a_false_pass():
    class _Hostile:
        def __iter__(self):
            raise RuntimeError("nope")

    witness = capture_parameter_witness([_Hostile()])
    assert witness["n_tensors_sampled"] == 0
    assert witness["n_unreadable"] == 1
    other = capture_parameter_witness([_Hostile()])
    assert compare_parameter_witness(witness, other)["changed"] is None


def test_learning_mode_without_torch_reports_undetermined_not_true():
    snapshot = learning_mode_snapshot(grad_enabled=None)
    assert snapshot["grad_enabled"] in (True, False, None)
    if snapshot["grad_enabled"] is None:
        assert "unavailable" in snapshot["grad_enabled_source"]


# --- 7. the parameter-delta witness is CHEAP by construction ----------------- #


def test_witness_cost_is_bounded_regardless_of_parameter_count():
    """Cheap enough to leave on by default -- a bound, not a hope.

    A full-tensor snapshot would scale with model size and get switched off, which
    is how the audit found parameter-delta witnesses existing in exactly 4 scripts.
    """
    many = _params(*[[float(i)] * 50 for i in range(500)])
    witness = capture_parameter_witness(many, max_tensors=8, coords_per_tensor=4)

    assert witness["n_tensors_total"] == 500
    assert witness["n_tensors_sampled"] == 8
    for entry in witness["sampled"]:
        assert len(entry["values"]) <= 4


def test_witness_is_deterministic_across_calls():
    """No RNG draw -- a witness must not perturb a seeded run's stream."""
    params = _params([0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0])
    assert (capture_parameter_witness(params)["signature"]
            == capture_parameter_witness(params)["signature"])


def test_witness_detects_a_real_update():
    before = capture_parameter_witness(_params([0.0, 1.0, 2.0, 3.0]))
    after = capture_parameter_witness(_params([0.0, 1.0, 2.0, 3.5]))
    comparison = compare_parameter_witness(before, after)
    assert comparison["changed"] is True
    assert comparison["max_abs_delta"] > 0.0


# --- 8. NOT the V3-EXQ-785 shape --------------------------------------------- #


def test_module_writes_its_own_block_and_not_interpretation_preconditions():
    """precondition_gate.py OWNS interpretation.preconditions[]; two writers is the bug.

    The indexer reads that list FLAT and ARM-BLIND and returns a whole-run
    `precondition_unmet` on the first unmet entry -- the mechanism by which 785's
    clean arm was re-vacated at adjudication time.
    """
    result = verify_capability_contract(_contract(), **_plastic_run())

    assert MANIFEST_KEY == "capability_contract"
    assert "preconditions" not in result["manifest_block"]
    assert "per_arm_gate" not in result["manifest_block"]
    assert "interpretation" not in result["manifest_block"]


def test_adjudication_preconditions_are_opt_in_not_emitted_by_verify():
    result = verify_capability_contract(_contract(), **_plastic_run())
    assert "adjudication_preconditions" not in result
    assert as_adjudication_preconditions(result), "still available on request"


def test_adjudication_preconditions_omit_undetermined_checks():
    """The indexer recomputes `met` and cannot represent 'not verified'.

    Emitting an undetermined check there would force it to read as a pass or a
    fail, both of which are lies.
    """
    run = _plastic_run(engagement={})
    result = verify_capability_contract(_contract(), **run)
    entries = as_adjudication_preconditions(result)
    names = [e["name"] for e in entries]

    assert not any("::reached" in n for n in names)
    assert all(isinstance(e["met"], bool) for e in entries)


def test_the_scope_note_records_why_this_is_not_the_785_shape():
    block = verify_capability_contract(_contract(),
                                       **_plastic_run())["manifest_block"]
    assert "785" in block["scope_note"]
    assert "WHOLE-ORGANISM" in block["scope_note"]


def test_contract_exposes_no_per_arm_concept():
    """Structural guarantee, not a convention: there is no arm to promote.

    Every check is a property of the single instantiated organism, so an
    arm-scoped fact cannot enter this module's verdict in the first place.
    """
    fields = set(CapabilityContract.__dataclass_fields__)
    assert not any("arm" in f for f in fields), fields
    result = verify_capability_contract(_contract(), **_plastic_run())
    assert not any("arm" in str(k) for k in result["manifest_block"])


# --- 9. the plasticity vocabulary is a STUB, and advisory -------------------- #


def test_plasticity_modes_is_the_registered_claim_enumeration():
    """Transcribed from the claim title, NOT invented here.

    TODO(chip-20260827-plasticity-inventory): replace with the real within-life
    plasticity inventory once REE_assembly/evidence/planning/
    within_life_plasticity_inventory_*.md exists.
    """
    assert PLASTICITY_MODES == (
        "parameters",
        "policy_value",
        "e1_e2_representations",
        "memory_state",
        "residue_ema_state",
        "offline_updates",
    )


def test_an_unrecognised_plasticity_mode_is_surfaced_not_refused():
    """A stub must never be able to refuse the declarations the inventory adds."""
    contract = _contract(requires_plasticity=[
        PlasticityRequirement(mode="hippocampal_buffers",
                              requires_gradients=False,
                              requires_optimizer=False,
                              requires_delta_witness=False)])
    result = verify_capability_contract(contract, **_plastic_run())

    assert result["unrecognised_plasticity_modes"] == ["hippocampal_buffers"]
    assert result["satisfied"] is True
    assert result["interpretation_route"] == ROUTE_INTERPRETABLE


def test_manifest_block_declares_the_vocabulary_as_provisional():
    block = verify_capability_contract(_contract(),
                                       **_plastic_run())["manifest_block"]
    status = block["plasticity_vocabulary_status"]
    assert "PROVISIONAL" in status
    assert "chip-20260827-plasticity-inventory" in status


def test_non_gradient_plasticity_mode_is_not_faulted_under_no_grad():
    """A memory-state driver correctly running under no_grad is not a misfire."""
    contract = _contract(requires_plasticity=[
        PlasticityRequirement(mode="memory_state", requires_gradients=False,
                              requires_optimizer=False,
                              requires_delta_witness=False)])
    run = _plastic_run(learning_mode=learning_mode_snapshot(grad_enabled=False))
    result = verify_capability_contract(contract, **run)

    assert result["satisfied"] is True
    assert result["interpretation_route"] == ROUTE_INTERPRETABLE


# --- 10. organism identity --------------------------------------------------- #


def test_undeclared_profile_on_either_side_is_undetermined():
    """Honest for a hand-assembled config and for the unpopulated v0 placeholder."""
    contract = _contract(canonical_profile=None, canonical_profile_hash=None)
    run = _plastic_run(observed_profile=None, observed_profile_hash=None)
    result = verify_capability_contract(contract, **run)

    assert _check_named(result,
                        "canonical_profile_identity")["status"] == STATUS_UNDETERMINED
    assert result["satisfied"] is True


def test_profile_mismatch_is_a_capability_precondition_failure():
    run = _plastic_run(observed_profile="something_else@v2")
    result = verify_capability_contract(_contract(), **run)

    assert result["interpretation_route"] == ROUTE_CAPABILITY_PRECONDITION_UNMET
    assert _check_named(result,
                        "canonical_profile_identity")["status"] == STATUS_UNMET


def test_profile_hash_drift_under_the_same_name_is_caught():
    run = _plastic_run(observed_profile_hash="0000")
    result = verify_capability_contract(_contract(), **run)

    check = _check_named(result, "canonical_profile_identity")
    assert check["status"] == STATUS_UNMET
    assert "hash differs" in check["detail"]


def test_an_undeclared_deviation_is_a_capability_precondition_failure():
    """'every explicit deviation from it' -- an undeclared one is not explicit."""
    run = _plastic_run(observed_deviations={"latent.e3_temperature": 0.9})
    result = verify_capability_contract(_contract(), **run)

    check = _check_named(result, "canonical_profile_identity")
    assert check["status"] == STATUS_UNMET
    assert check["undeclared_deviations"] == {"latent.e3_temperature": 0.9}


def test_a_declared_deviation_is_accepted():
    contract = _contract(declared_deviations={"latent.e3_temperature": 0.9})
    run = _plastic_run(observed_deviations={"latent.e3_temperature": 0.9})
    result = verify_capability_contract(contract, **run)

    assert _check_named(result,
                        "canonical_profile_identity")["status"] == STATUS_SATISFIED


# --- 11. capability requirements --------------------------------------------- #


def test_capability_floor_below_threshold_is_a_precondition_failure():
    contract = _contract(requires_capabilities=[
        CapabilityRequirement(name="foraging_competence",
                              metric="resources_per_episode", floor=1.0)])
    run = _plastic_run(capabilities_measured={"resources_per_episode": 0.065})
    result = verify_capability_contract(contract, **run)

    assert result["interpretation_route"] == ROUTE_CAPABILITY_PRECONDITION_UNMET
    assert _check_named(result,
                        "foraging_competence::floor")["status"] == STATUS_UNMET


def test_capability_floor_above_threshold_passes():
    contract = _contract(requires_capabilities=[
        CapabilityRequirement(name="foraging_competence",
                              metric="resources_per_episode", floor=1.0)])
    run = _plastic_run(capabilities_measured={"resources_per_episode": 2.4})
    result = verify_capability_contract(contract, **run)

    assert result["satisfied"] is True


def test_capability_upper_direction_is_honoured():
    contract = _contract(requires_capabilities=[
        CapabilityRequirement(name="death_rate", metric="death_rate", floor=0.5,
                              direction="upper")])
    run = _plastic_run(capabilities_measured={"death_rate": 0.2})
    result = verify_capability_contract(contract, **run)
    assert _check_named(result, "death_rate::floor")["status"] == STATUS_SATISFIED


# --- 12. declaration errors are programmer errors, and the ONLY raise -------- #


def test_authority_floor_without_a_metric_is_refused():
    with pytest.raises(CapabilityContractDeclarationError):
        MechanismRequirement(name="m", authority_metric=None, authority_floor=0.1)


def test_capability_metric_without_a_floor_is_refused():
    with pytest.raises(CapabilityContractDeclarationError):
        CapabilityRequirement(name="c", metric="x")


def test_capability_floor_without_a_metric_is_refused():
    with pytest.raises(CapabilityContractDeclarationError):
        CapabilityRequirement(name="c", floor=1.0)


def test_duplicate_requirement_names_are_refused():
    with pytest.raises(CapabilityContractDeclarationError) as exc:
        CapabilityContract(experiment_id="X",
                           requires_mechanisms=[_mech(), _mech()])
    assert "duplicate" in str(exc.value)


def test_empty_names_are_refused():
    with pytest.raises(CapabilityContractDeclarationError):
        MechanismRequirement(name="  ")
    with pytest.raises(CapabilityContractDeclarationError):
        PlasticityRequirement(mode="")
    with pytest.raises(CapabilityContractDeclarationError):
        CapabilityContract(experiment_id="")


# --- 13. reporting ----------------------------------------------------------- #


def test_report_is_ascii_only():
    """CLAUDE.md repo rule -- non-ASCII becomes mojibake on cp1252 terminals."""
    run = _plastic_run(agent=_Organism(e3_enabled=False),
                       engagement={})
    text = format_report(verify_capability_contract(_contract(), **run))
    text.encode("ascii")  # raises UnicodeEncodeError on a violation


def test_report_names_the_route_and_the_prohibition():
    run = _plastic_run(agent=_Organism(e3_enabled=False))
    text = format_report(verify_capability_contract(_contract(), **run))

    assert ROUTE_CAPABILITY_PRECONDITION_UNMET in text
    assert "do NOT record it as FAIL" in text


def test_manifest_block_is_json_serialisable():
    import json

    block = verify_capability_contract(_contract(),
                                       **_plastic_run())["manifest_block"]
    json.loads(json.dumps(block))


def test_manifest_block_names_the_governing_claim():
    block = verify_capability_contract(_contract(),
                                       **_plastic_run())["manifest_block"]
    assert block["claim"] == "GOV-CAPCONTRACT-1"
