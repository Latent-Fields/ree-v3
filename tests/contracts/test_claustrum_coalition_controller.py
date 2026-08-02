"""Contract tests for the SD-091/MECH-481 coalition-control MVP (2026-08-02).

Why this file exists
---------------------
sd_091_coalition_topology_control.md (REE_assembly/docs/architecture/) is a
registration-before-build design doc: SD-091 (a graph-valued control output
G_t, alongside the existing mode M_t and parameter theta_t outputs) and
MECH-481 (typed control demands -> typed coalition templates) were both
substrate_conditional -- "no code exists yet" -- until this landing. This
file covers the new ree_core/claustrum/ package: ControlDemandType
(control_demand.py), COALITION_TEMPLATES (coalition_templates.py), and
CoalitionController/CoalitionControllerConfig/CoalitionState
(coalition_controller.py).

Scope: this lands only steps 1-3 of the doc's "Minimum-viable V3
implementation path" (the standalone primitive + the 2 MVP templates).
Steps 4-5 (wiring real consumer read sites and REEAgent.select_action) are
a deliberate follow-up -- nothing in ree_core imports ree_core.claustrum
yet, so there is no live-agent-loop surface to contract-test here. What IS
tested is everything reachable standalone: the doc's own step-6 smoke-test
requirement (SENSORY_RESAMPLE and PROVENANCE_CHECK must produce measurably
different write_gate() outputs) and the Section-4 guardrails restated as
structural properties in coalition_controller.py's module docstring.

Contracts:
  C1  Enum surface -- all 10 MECH-481 ControlDemandType members declared;
      MVP_TEMPLATED_TYPES and COALITION_TEMPLATES agree on exactly
      {SENSORY_RESAMPLE, PROVENANCE_CHECK}.
  C2  Bit-identical OFF -- CoalitionControllerConfig(enabled=False)
      (the default): request_coalition() is a no-op (returns None, no
      state recorded), tick() is a no-op, write_gate()/channel_gain()
      always return the pass-through identity (1.0), recruited_targets()/
      suppressed_targets() are always empty.
  C3  Flag-on smoke test (doc step 6) -- with the controller enabled,
      request_coalition(SENSORY_RESAMPLE) and request_coalition
      (PROVENANCE_CHECK) on separate controller instances produce
      measurably DIFFERENT write_gate() outputs at every consumer-site
      target named in coalition_templates.py's docstring table, and
      disjoint recruited_targets()/suppressed_targets() sets.
  C4  Unregistered-type no-op -- requesting one of the 8 untemplated
      ControlDemandType members increments unregistered_request_count and
      returns None; never raises.
  C5  Gamma_t timeout dissolution -- a coalition with no
      completion_condition dissolves exactly when
      (tick - opened_tick) >= max_duration_ticks, not before.
  C6  Gamma_t completion-condition dissolution -- a coalition dissolves
      early when completion_condition(agent_state) returns True, before
      its timeout would otherwise fire.
  C7  BetaGate-adjacent monotone-non-increasing guardrail -- for
      e3_commitment_monitor / motor_commitment (the PROVENANCE_CHECK
      targets doc Section 4 names as the concrete BetaGate failure mode),
      write_gate() never exceeds 1.0, for the shipped templates AND for an
      adversarially-constructed CoalitionState with out-of-range weights
      (clamping is what makes this a property of the arithmetic, not a
      convention).
  C8  No-global-broadcast default -- participating/suppressed are sparse
      per-type dicts (never an all-target wildcard) for both shipped
      templates; recruited_targets()/suppressed_targets() are empty when
      disabled (never an implicit "everything recruited").
  C9  Weight clamping -- CoalitionState.__post_init__ clamps
      participating/suppressed to [0, 1] and channel_gain to >= 0.0
      regardless of constructor input.
  C10 Multi-coalition multiplicative composition -- two simultaneously
      active coalitions naming the same target compose write_gate()
      multiplicatively (product of both coalition_gate() factors).
  C11 Mode-classification isolation -- coalition_controller.py does not
      import ree_core.cingulate at all (the one-directional-dependency
      guardrail: coalition reads mode via a future consumer-site
      composition, but never reaches back into SalienceCoordinator).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.claustrum.coalition_controller import (
    CoalitionController,
    CoalitionControllerConfig,
    CoalitionState,
)
from ree_core.claustrum.coalition_templates import COALITION_TEMPLATES
from ree_core.claustrum.control_demand import (
    MVP_TEMPLATED_TYPES,
    ControlDemandType,
)


# ----------------------------------------------------------------------
# C1 -- enum surface
# ----------------------------------------------------------------------
def test_c1_enum_surface_and_mvp_subset():
    expected = {
        "SENSORY_RESAMPLE",
        "PROVENANCE_CHECK",
        "COUNTERFACTUAL_EXPANSION",
        "CROSS_HORIZON_RECONCILIATION",
        "SOCIAL_MODEL_CHECK",
        "INVARIANT_CONFLICT_REVIEW",
        "ACTION_OUTCOME_RECALIBRATION",
        "LANGUAGE_EXPLICITATION",
        "COMMITMENT_REOPEN",
        "SAFE_DEFER",
    }
    actual = {member.name for member in ControlDemandType}
    assert actual == expected, f"taxonomy drift: {actual.symmetric_difference(expected)}"

    mvp = {ControlDemandType.SENSORY_RESAMPLE, ControlDemandType.PROVENANCE_CHECK}
    assert MVP_TEMPLATED_TYPES == mvp
    assert set(COALITION_TEMPLATES.keys()) == mvp


# ----------------------------------------------------------------------
# C2 -- bit-identical OFF
# ----------------------------------------------------------------------
def test_c2_disabled_is_full_noop():
    controller = CoalitionController(CoalitionControllerConfig())  # enabled=False default
    assert controller.config.enabled is False

    result = controller.request_coalition(ControlDemandType.SENSORY_RESAMPLE, tick=0)
    assert result is None
    assert controller.active_coalitions == []
    assert controller.unregistered_request_count == 0  # disabled short-circuits before counting

    controller.tick(current_tick=5)  # must not raise, must not create state
    assert controller.active_coalitions == []

    for target in ("e1_sensory_encoder", "hippocampal_anchor_set", "anything_unregistered"):
        assert controller.write_gate(target) == 1.0
        assert controller.channel_gain(target) == 1.0

    assert controller.recruited_targets() == frozenset()
    assert controller.suppressed_targets() == frozenset()


# ----------------------------------------------------------------------
# C3 -- flag-on smoke test: the two MVP types must be distinguishable
# ----------------------------------------------------------------------
def _enabled_controller() -> CoalitionController:
    return CoalitionController(CoalitionControllerConfig(enabled=True))


def test_c3_sensory_resample_and_provenance_check_are_distinguishable():
    all_targets = set()
    for template in COALITION_TEMPLATES.values():
        all_targets.update(template.participating.keys())
        all_targets.update(template.suppressed.keys())
    assert len(all_targets) > 0

    sensory = _enabled_controller()
    state = sensory.request_coalition(ControlDemandType.SENSORY_RESAMPLE, tick=0)
    assert state is not None
    assert state.demand_type is ControlDemandType.SENSORY_RESAMPLE

    provenance = _enabled_controller()
    state2 = provenance.request_coalition(ControlDemandType.PROVENANCE_CHECK, tick=0)
    assert state2 is not None
    assert state2.demand_type is ControlDemandType.PROVENANCE_CHECK

    differing = {
        target
        for target in all_targets
        if sensory.write_gate(target) != provenance.write_gate(target)
    }
    # Every named consumer-site target must differ between the two MVP
    # templates -- this is the doc's own step-6 smoke-test requirement,
    # and the concrete falsifier for "typing does no work" (Arm 3 vs Arm 4
    # collapsing to the same behaviour).
    assert differing == all_targets, f"not distinguishable at: {all_targets - differing}"

    # Disjoint subsystem sets (doc Section 2's stated non-vacuity argument).
    assert sensory.recruited_targets().isdisjoint(provenance.recruited_targets())
    assert sensory.recruited_targets() != provenance.recruited_targets()


# ----------------------------------------------------------------------
# C4 -- unregistered type is a no-op with a diagnostic counter
# ----------------------------------------------------------------------
def test_c4_unregistered_type_noop_with_counter():
    controller = _enabled_controller()
    untemplated = set(ControlDemandType) - MVP_TEMPLATED_TYPES
    assert len(untemplated) == 8

    for demand_type in untemplated:
        before = controller.unregistered_request_count
        result = controller.request_coalition(demand_type, tick=0)
        assert result is None
        assert controller.unregistered_request_count == before + 1

    assert controller.active_coalitions == []


# ----------------------------------------------------------------------
# C5 -- Gamma_t timeout dissolution
# ----------------------------------------------------------------------
def test_c5_timeout_dissolution():
    controller = _enabled_controller()
    controller.request_coalition(
        ControlDemandType.SENSORY_RESAMPLE, tick=10, max_duration_ticks=5
    )
    assert len(controller.active_coalitions) == 1

    controller.tick(current_tick=14)  # 14 - 10 = 4 < 5: still active
    assert len(controller.active_coalitions) == 1

    controller.tick(current_tick=15)  # 15 - 10 = 5 >= 5: dissolves
    assert controller.active_coalitions == []
    assert controller.write_gate("e1_sensory_encoder") == 1.0


# ----------------------------------------------------------------------
# C6 -- Gamma_t completion-condition dissolution (fires before timeout)
# ----------------------------------------------------------------------
def test_c6_completion_condition_dissolution():
    controller = _enabled_controller()
    controller.request_coalition(
        ControlDemandType.PROVENANCE_CHECK,
        tick=0,
        max_duration_ticks=1000,  # would not time out for a long while
        completion_condition=lambda state: state.get("resolved", False),
    )
    assert len(controller.active_coalitions) == 1

    controller.tick(current_tick=1, agent_state={"resolved": False})
    assert len(controller.active_coalitions) == 1  # not resolved yet, well under timeout

    controller.tick(current_tick=2, agent_state={"resolved": True})
    assert controller.active_coalitions == []  # dissolved by completion, not timeout


def test_c6b_misbehaving_completion_condition_falls_back_to_timeout():
    controller = _enabled_controller()

    def _raises(_state):
        raise RuntimeError("boom")

    controller.request_coalition(
        ControlDemandType.SENSORY_RESAMPLE,
        tick=0,
        max_duration_ticks=3,
        completion_condition=_raises,
    )
    controller.tick(current_tick=2, agent_state={})  # must not raise
    assert len(controller.active_coalitions) == 1
    controller.tick(current_tick=3, agent_state={})  # timeout floor still fires
    assert controller.active_coalitions == []


# ----------------------------------------------------------------------
# C7 -- BetaGate-adjacent monotone-non-increasing guardrail
# ----------------------------------------------------------------------
def test_c7_betagate_adjacent_targets_never_exceed_baseline():
    controller = _enabled_controller()
    controller.request_coalition(ControlDemandType.PROVENANCE_CHECK, tick=0)

    for target in ("e3_commitment_monitor", "motor_commitment", "hippocampal_write_consolidation"):
        gate = controller.write_gate(target)
        assert 0.0 <= gate <= 1.0
        # PROVENANCE_CHECK's own template only ever attenuates or leaves
        # these unchanged -- never a gate above the 1.0 no-op baseline.
        assert gate <= 1.0

    # Adversarial construction: out-of-range weights must still clamp to
    # <= 1.0 -- this is what makes the guardrail a property of the
    # arithmetic (CoalitionState.__post_init__), not caller discipline.
    hostile = CoalitionState(
        demand_type=ControlDemandType.PROVENANCE_CHECK,
        participating={"e3_commitment_monitor": 5.0},  # attempt to force-open
        suppressed={"motor_commitment": -3.0},  # attempt to invert suppression into a boost
        opened_tick=0,
        max_duration_ticks=10,
    )
    assert hostile.coalition_gate("e3_commitment_monitor") <= 1.0
    assert hostile.coalition_gate("motor_commitment") <= 1.0
    assert hostile.participating["e3_commitment_monitor"] == 1.0  # clamped down from 5.0
    assert hostile.suppressed["motor_commitment"] == 0.0  # clamped up from -3.0


# ----------------------------------------------------------------------
# C8 -- no-global-broadcast default
# ----------------------------------------------------------------------
def test_c8_templates_are_sparse_not_global_broadcast():
    # A template naming "every" subsystem would be indistinguishable from
    # the untyped-coalition failure mode (MECH-481 Arm 3). Cheap structural
    # proxy: each template's participating+suppressed set is small and the
    # two MVP templates' sets are disjoint (already asserted in C3) --
    # here we additionally assert neither template is suspiciously large
    # (a stand-in for "not literally every registered target").
    for demand_type, template in COALITION_TEMPLATES.items():
        touched = set(template.participating) | set(template.suppressed)
        assert 0 < len(touched) <= 5, f"{demand_type} touches {len(touched)} targets"


def test_c8b_disabled_recruited_targets_never_implicit_all():
    controller = CoalitionController(CoalitionControllerConfig(enabled=False))
    assert controller.recruited_targets() == frozenset()
    assert controller.suppressed_targets() == frozenset()


# ----------------------------------------------------------------------
# C9 -- weight clamping
# ----------------------------------------------------------------------
def test_c9_coalition_state_clamps_out_of_range_weights():
    state = CoalitionState(
        demand_type=ControlDemandType.SENSORY_RESAMPLE,
        participating={"a": 1.5, "b": -0.2},
        suppressed={"c": 1.7, "d": -0.4},
        channel_gain={"e": -2.0, "f": 3.0},
        opened_tick=0,
        max_duration_ticks=0,  # must also clamp up to >= 1
    )
    assert state.participating == {"a": 1.0, "b": 0.0}
    assert state.suppressed == {"c": 1.0, "d": 0.0}
    assert state.channel_gain == {"e": 0.0, "f": 3.0}  # gain has no upper clamp
    assert state.max_duration_ticks >= 1


# ----------------------------------------------------------------------
# C10 -- multi-coalition multiplicative composition
# ----------------------------------------------------------------------
def test_c10_multiple_active_coalitions_compose_multiplicatively():
    controller = _enabled_controller()
    controller.request_coalition(ControlDemandType.SENSORY_RESAMPLE, tick=0)
    controller.request_coalition(ControlDemandType.PROVENANCE_CHECK, tick=0)
    assert len(controller.active_coalitions) == 2

    # e1_sensory_encoder: only SENSORY_RESAMPLE touches it (0.9); expect
    # exactly that factor (PROVENANCE_CHECK contributes its 1.0 default).
    assert controller.write_gate("e1_sensory_encoder") == 0.9

    # motor_commitment: only PROVENANCE_CHECK touches it (1 - 0.6 = 0.4).
    assert abs(controller.write_gate("motor_commitment") - 0.4) < 1e-9

    # A target neither template names: pure 1.0 identity.
    assert controller.write_gate("never_named_anywhere") == 1.0


# ----------------------------------------------------------------------
# C11 -- mode-classification isolation (structural, not just behavioural)
# ----------------------------------------------------------------------
def test_c11_coalition_controller_does_not_import_cingulate():
    import ast

    path = REPO_ROOT / "ree_core" / "claustrum" / "coalition_controller.py"
    tree = ast.parse(path.read_text())
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    assert not any("cingulate" in name for name in imported_modules), (
        f"coalition_controller.py must never import ree_core.cingulate -- the "
        f"one-directional dependency (coalition reads mode, mode never reads "
        f"coalition) is enforced by this module having no import path back "
        f"into SalienceCoordinator at all. Found: {imported_modules}"
    )
