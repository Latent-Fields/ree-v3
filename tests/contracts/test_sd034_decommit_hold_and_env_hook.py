"""SD-034 commitment-closure-control-plane contracts (2026-06-12).

Two-leg substrate amend giving the SD-034 ClosureOperator behavioural authority:

  Leg A -- explicit env-completion hook seam: REEAgent.notify_env_completion ->
           ClosureOperator.emit_closure (closes V3-EXQ-460c n_closures=0).
  Leg B -- BetaGate de-commitment hold / refractory: a closure-driven release
           installs a refractory window so the latch cannot immediately
           re-commit (closes V3-EXQ-468c committed_frac cap-pin).

All new config defaults are no-op -> bit-identical OFF.
"""

import torch

from ree_core.heartbeat.beta_gate import BetaGate
from ree_core.governance.closure_operator import (
    ClosureOperator,
    ClosureOperatorConfig,
    ClosureEvent,
)
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent


def _build_agent(**kw):
    cfg = REEConfig.from_dims(
        world_obs_dim=250,
        body_obs_dim=12,
        action_dim=4,
        use_lateral_pfc_analog=True,
        use_dacc=True,
        **kw,
    )
    return REEAgent(cfg)


def test_c1_betagate_refractory_default_off_bit_identical():
    """Default BetaGate never installs a refractory -> elevate is unaffected."""
    g = BetaGate()
    assert g.refractory_remaining == 0
    g.elevate()
    assert g.is_elevated
    st = g.get_state()
    assert st["sd034_refractory_remaining"] == 0
    assert st["sd034_n_elevation_refractory_blocked"] == 0


def test_c2_betagate_refractory_blocks_then_expires():
    """apply_refractory blocks elevate for N propagate ticks, then releases."""
    g = BetaGate()
    g.elevate()
    g.release()
    g.apply_refractory(3)
    g.elevate()
    assert not g.is_elevated, "elevate must be blocked during the hold"
    assert g.get_state()["sd034_n_elevation_refractory_blocked"] == 1
    ps = torch.zeros(1, 4)
    for _ in range(3):
        g.propagate(ps)  # decrement 3 -> 0
    assert g.refractory_remaining == 0
    g.elevate()
    assert g.is_elevated, "elevate must succeed once the hold expires"
    # max-not-truncate: overlapping holds extend
    g.release()
    g.apply_refractory(2)
    g.apply_refractory(5)
    assert g.refractory_remaining == 5
    # n_ticks <= 0 is a no-op
    g.reset()
    g.apply_refractory(0)
    assert g.refractory_remaining == 0


def test_c3_closure_operator_installs_refractory_on_fire():
    """decommit_hold_ticks > 0 -> _fire installs the BetaGate refractory."""
    g = BetaGate()
    g.elevate()
    op = ClosureOperator(
        config=ClosureOperatorConfig(
            use_closure_operator=True, decommit_hold_ticks=4
        ),
        beta_gate=g,
    )
    ev = op.emit_closure(action_class=1, z_world=torch.zeros(32),
                         bypass_mode_conditioning=True)
    assert isinstance(ev, ClosureEvent) and ev.fired
    assert ev.decommit_refractory_applied == 4
    assert g.refractory_remaining == 4
    # default 0 -> no refractory installed (bit-identical)
    g2 = BetaGate()
    g2.elevate()
    op2 = ClosureOperator(
        config=ClosureOperatorConfig(use_closure_operator=True),
        beta_gate=g2,
    )
    ev2 = op2.emit_closure(action_class=1, z_world=torch.zeros(32),
                          bypass_mode_conditioning=True)
    assert ev2.decommit_refractory_applied == 0
    assert g2.refractory_remaining == 0


def test_c4_agent_env_hook_off_is_noop():
    """notify_env_completion returns None / no-op when the hook flag is off."""
    a = _build_agent(use_closure_operator=True)  # hook flag default False
    a.act(torch.zeros(262))
    n0 = int(a.closure_operator._n_closures)
    assert a.notify_env_completion(action_class=2) is None
    assert int(a.closure_operator._n_closures) == n0


def test_c5_agent_env_hook_on_fires_closure_and_hold():
    """Hook ON routes the env completion into emit_closure + installs the hold."""
    a = _build_agent(
        use_closure_operator=True,
        use_closure_env_completion_hook=True,
        closure_decommit_hold_ticks=5,
    )
    a.act(torch.zeros(262))
    n0 = int(a.closure_operator._n_closures)
    ev = a.notify_env_completion(action_class=2, bypass_mode_conditioning=True)
    assert ev is not None and ev.fired
    assert int(a.closure_operator._n_closures) == n0 + 1
    assert ev.decommit_refractory_applied == 5
    assert a.beta_gate.refractory_remaining == 5
    a.beta_gate.elevate()
    assert not a.beta_gate.is_elevated, "de-commit hold must block re-commit"


def test_c6_agent_env_hook_simulation_gate():
    """MECH-094: a simulation/replay completion does not emit a waking closure."""
    a = _build_agent(
        use_closure_operator=True, use_closure_env_completion_hook=True
    )
    a.act(torch.zeros(262))
    assert a.notify_env_completion(action_class=1, simulation_mode=True) is None
