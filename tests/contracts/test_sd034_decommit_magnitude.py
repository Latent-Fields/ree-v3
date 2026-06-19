"""SD-034 commitment-closure-control-plane DE-COMMIT-AUTHORITY MAGNITUDE amend
contracts (2026-06-19).

Routed by failure_autopsy_V3-EXQ-460f_2026-06-18: the fixed Leg-B refractory
(~5 ticks) suppresses only ~20-35 tick-blocks of latch occupancy, swamped by
~530-560 natural-commit elevated steps -> the de-commit occupancy-drop DV is
underpowered. This amend scales the refractory installed at a closure fire by
the committed-run length captured from the BetaGate BEFORE the closure's own
release(): a long committed run triggers a proportionally long post-closure
hold.

All new config defaults are no-op (scale 0.0) -> bit-identical OFF.
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


def test_c1_committed_run_length_counter():
    """committed_run_length tracks the per-run elevated tick count."""
    g = BetaGate()
    ps = torch.zeros(1, 4)
    assert g.committed_run_length == 0
    assert g.get_state()["sd034_committed_run_length"] == 0

    # Not elevated -> propagate does not advance the run length.
    g.propagate(ps)
    assert g.committed_run_length == 0

    # Elevate, then each propagate tick while elevated advances the counter.
    g.elevate()
    for i in range(1, 6):
        g.propagate(ps)
        assert g.committed_run_length == i
    assert g.get_state()["sd034_committed_run_length"] == 5

    # release() ends the run -> counter resets to 0.
    g.release()
    assert g.committed_run_length == 0

    # A fresh elevate (not-elevated -> elevated) starts a new run; a re-elevate
    # while already elevated leaves the run length unchanged.
    g.elevate()
    g.propagate(ps)
    g.propagate(ps)
    assert g.committed_run_length == 2
    g.elevate()  # already elevated -> no reset
    assert g.committed_run_length == 2
    g.propagate(ps)
    assert g.committed_run_length == 3

    # reset() clears the counter.
    g.reset()
    assert g.committed_run_length == 0


def test_c2_scale_off_bit_identical_to_fixed_hold():
    """scale 0.0 (default) -> the refractory equals decommit_hold_ticks exactly,
    independent of the committed-run length (bit-identical to the fixed hold)."""
    g = BetaGate()
    g.elevate()
    ps = torch.zeros(1, 4)
    for _ in range(40):
        g.propagate(ps)  # run_length = 40
    assert g.committed_run_length == 40
    op = ClosureOperator(
        config=ClosureOperatorConfig(
            use_closure_operator=True,
            decommit_hold_ticks=5,
            # decommit_hold_scale_with_run defaults to 0.0
        ),
        beta_gate=g,
    )
    # default config fields are no-op
    assert op.config.decommit_hold_scale_with_run == 0.0
    assert op.config.decommit_hold_max_ticks == 0
    ev = op.emit_closure(
        action_class=1, z_world=torch.zeros(32), bypass_mode_conditioning=True
    )
    assert isinstance(ev, ClosureEvent) and ev.fired
    # 40-tick run does NOT scale the hold when scale is 0.0.
    assert ev.decommit_refractory_applied == 5
    assert g.refractory_remaining == 5

    # decommit_hold_ticks 0 + scale 0.0 -> no refractory at all.
    g2 = BetaGate()
    g2.elevate()
    for _ in range(40):
        g2.propagate(ps)
    op2 = ClosureOperator(
        config=ClosureOperatorConfig(use_closure_operator=True), beta_gate=g2
    )
    ev2 = op2.emit_closure(
        action_class=1, z_world=torch.zeros(32), bypass_mode_conditioning=True
    )
    assert ev2.decommit_refractory_applied == 0
    assert g2.refractory_remaining == 0


def test_c3_refractory_scales_with_committed_run_length():
    """scale > 0 -> n = decommit_hold_ticks + round(scale * run_length), where
    run_length is captured BEFORE the closure's own release()."""
    g = BetaGate()
    g.elevate()
    ps = torch.zeros(1, 4)
    for _ in range(50):
        g.propagate(ps)  # run_length = 50
    assert g.committed_run_length == 50
    op = ClosureOperator(
        config=ClosureOperatorConfig(
            use_closure_operator=True,
            decommit_hold_ticks=2,
            decommit_hold_scale_with_run=0.1,
        ),
        beta_gate=g,
    )
    ev = op.emit_closure(
        action_class=0, z_world=torch.zeros(32), bypass_mode_conditioning=True
    )
    # n = 2 + round(0.1 * 50) = 2 + 5 = 7
    assert ev.decommit_refractory_applied == 7
    assert g.refractory_remaining == 7
    # the closure's release() ended the run -> counter is back to 0
    assert g.committed_run_length == 0
    assert not g.is_elevated

    # A longer run yields a proportionally longer hold (the magnitude property).
    g2 = BetaGate()
    g2.elevate()
    for _ in range(200):
        g2.propagate(ps)
    op2 = ClosureOperator(
        config=ClosureOperatorConfig(
            use_closure_operator=True,
            decommit_hold_ticks=2,
            decommit_hold_scale_with_run=0.1,
        ),
        beta_gate=g2,
    )
    ev2 = op2.emit_closure(
        action_class=0, z_world=torch.zeros(32), bypass_mode_conditioning=True
    )
    assert ev2.decommit_refractory_applied == 22  # 2 + round(0.1 * 200)
    assert ev2.decommit_refractory_applied > ev.decommit_refractory_applied


def test_c4_max_ticks_clamp():
    """decommit_hold_max_ticks > 0 caps the scaled refractory."""
    g = BetaGate()
    g.elevate()
    ps = torch.zeros(1, 4)
    for _ in range(100):
        g.propagate(ps)  # run_length = 100 -> 2 + 10 = 12 uncapped
    op = ClosureOperator(
        config=ClosureOperatorConfig(
            use_closure_operator=True,
            decommit_hold_ticks=2,
            decommit_hold_scale_with_run=0.1,
            decommit_hold_max_ticks=4,
        ),
        beta_gate=g,
    )
    ev = op.emit_closure(
        action_class=0, z_world=torch.zeros(32), bypass_mode_conditioning=True
    )
    assert ev.decommit_refractory_applied == 4  # min(12, 4)
    assert g.refractory_remaining == 4


def test_c5_from_dims_and_agent_wiring():
    """from_dims surfaces the magnitude flags onto config + the agent's
    ClosureOperatorConfig build site forwards them."""
    cfg = REEConfig.from_dims(
        world_obs_dim=250,
        body_obs_dim=12,
        action_dim=4,
        use_lateral_pfc_analog=True,
        use_dacc=True,
        use_closure_operator=True,
        closure_decommit_hold_ticks=3,
        closure_decommit_hold_scale_with_run=0.2,
        closure_decommit_hold_max_ticks=30,
    )
    assert cfg.closure_decommit_hold_scale_with_run == 0.2
    assert cfg.closure_decommit_hold_max_ticks == 30
    a = REEAgent(cfg)
    co_cfg = a.closure_operator.config
    assert co_cfg.decommit_hold_ticks == 3
    assert co_cfg.decommit_hold_scale_with_run == 0.2
    assert co_cfg.decommit_hold_max_ticks == 30
    # defaults (flags absent) -> no-op on the agent build
    a_def = _build_agent(use_closure_operator=True)
    assert a_def.closure_operator.config.decommit_hold_scale_with_run == 0.0
    assert a_def.closure_operator.config.decommit_hold_max_ticks == 0


def test_c6_agent_action_stream_bit_identical_scale_off():
    """Default agent == explicit scale=0.0 agent: bit-identical action stream
    (the magnitude lever changes nothing when scale is 0.0)."""
    obs = torch.zeros(262)

    def _run(**kw):
        torch.manual_seed(0)
        a = _build_agent(
            use_closure_operator=True,
            use_closure_env_completion_hook=True,
            **kw,
        )
        return [a.act(obs).clone() for _ in range(8)]

    default_stream = _run()
    explicit_stream = _run(
        closure_decommit_hold_scale_with_run=0.0,
        closure_decommit_hold_max_ticks=0,
    )
    for ad, ae in zip(default_stream, explicit_stream):
        assert torch.equal(ad, ae), "scale=0.0 must not change the action stream"
