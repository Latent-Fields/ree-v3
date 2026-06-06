"""Contract tests for MECH-353 blocked-agency / control-failure (z_block).

Interface-level guarantees that should hold regardless of tuning:
  C1  default OFF -> agent.blocked_agency is None, LatentState.z_block None,
      bit-identical action stream (no RNG perturbation).
  C2  BlockedAgencyConfig validation (loud on bad values).
  C3  external-attribution gate: motor_agency below floor -> no accumulation.
  C4  capacity gate: capacity collapse -> assert share decays, withdraw_handoff
      rises (opposite controllability pole to suffering).
  C5  predicted-effect / goal gates: no goal OR sub-floor mismatch -> no rise.
  C6  ASSERT score-bias signs: negative on action, positive on no-op, extra
      positive on the just-blocked action class (alternative-action search).
  C7  DECOMMIT signal after sustained asserting block; MECH-094 sim no-op.
  C8  env action-block knob: OFF bit-identical; ON cancels the move (agent
      stays put), emits info tags, inflicts no harm.
  C9  LatentState.z_block populated + survives detach() when ON.
"""

import torch

from ree_core.affect.blocked_agency import (
    BlockedAgency,
    BlockedAgencyConfig,
)
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2


def _build(env, **kw):
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        **kw,
    )


def _seed_goal(ag):
    ag.goal_state._z_goal = torch.ones(1, ag.goal_state.config.goal_dim) * 0.5


def _action_stream(use_ba, n=20):
    torch.manual_seed(321)
    env = CausalGridWorldV2(size=8, seed=11)
    ag = REEAgent(_build(env, **({"use_blocked_agency": True} if use_ba else {})))
    _, od = env.reset()
    acts = []
    for _ in range(n):
        a = ag.act_with_split_obs(od["body_state"], od["world_state"])
        acts.append(int(a.argmax()))
        _, h, d, inf, od = env.step(a)
        if d:
            _, od = env.reset()
            ag.reset()
    return acts


# ---------------------------------------------------------------- C1
def test_c1_default_off_no_op():
    env = CausalGridWorldV2(size=8, seed=0)
    ag = REEAgent(_build(env))
    _, od = env.reset()
    lat = ag.sense(od["body_state"], od["world_state"])
    assert ag.blocked_agency is None
    assert lat.z_block is None
    # use_blocked_agency=True with no env block must NOT perturb the RNG stream:
    # the regulator uses no torch random ops and adds zero bias when z_block~0.
    assert _action_stream(False) == _action_stream(True)


# ---------------------------------------------------------------- C2
def test_c2_config_validation():
    for bad in (
        {"accumulation_rate": 1.5},
        {"leak_rate": -0.1},
        {"z_block_cap": 0.0},
        {"assert_bias_scale": 0.0},
        {"decommit_consecutive_ticks": 0},
    ):
        try:
            BlockedAgency(BlockedAgencyConfig(use_blocked_agency=True, **bad))
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad}")


# ---------------------------------------------------------------- C3
def test_c3_external_attribution_gate():
    ba = BlockedAgency(BlockedAgencyConfig(
        use_blocked_agency=True, attribution_motor_floor=0.5,
        outcome_mismatch_floor=0.1))
    # High mismatch but LOW motor_agency (own motor error) -> no accumulation.
    for _ in range(10):
        out = ba.update(outcome_mismatch=0.9, motor_agency=0.2,
                        goal_active=True, capacity_belief=1.0)
    assert out.external_block_this_tick is False
    assert ba.get_z_block() == 0.0
    # High mismatch AND high motor_agency (external) -> accumulates.
    ba.reset()
    for _ in range(10):
        out = ba.update(outcome_mismatch=0.9, motor_agency=0.9,
                        goal_active=True, capacity_belief=1.0)
    assert out.external_block_this_tick is True
    assert ba.get_z_block() > 0.0


# ---------------------------------------------------------------- C4
def test_c4_capacity_gate_assert_vs_withdraw():
    ba = BlockedAgency(BlockedAgencyConfig(use_blocked_agency=True))
    # Capacity retained -> assert dominates, handoff ~0.
    for _ in range(8):
        out_hi = ba.update(0.9, 0.9, True, capacity_belief=1.0)
    assert out_hi.z_block_assert > out_hi.withdraw_handoff
    # Capacity collapsed -> withdraw handoff dominates, assert ~0.
    ba.reset()
    for _ in range(8):
        out_lo = ba.update(0.9, 0.9, True, capacity_belief=0.0)
    assert out_lo.z_block_assert == 0.0
    assert out_lo.withdraw_handoff > 0.0


# ---------------------------------------------------------------- C5
def test_c5_goal_and_mismatch_gates():
    ba = BlockedAgency(BlockedAgencyConfig(
        use_blocked_agency=True, require_goal_active=True,
        outcome_mismatch_floor=0.1))
    # No live goal -> no accumulation even under strong external mismatch.
    for _ in range(10):
        ba.update(0.9, 0.9, goal_active=False, capacity_belief=1.0)
    assert ba.get_z_block() == 0.0
    # Sub-floor mismatch (action succeeded) -> no accumulation, leaks.
    ba.reset()
    for _ in range(10):
        ba.update(0.01, 0.9, goal_active=True, capacity_belief=1.0)
    assert ba.get_z_block() == 0.0


# ---------------------------------------------------------------- C6
def test_c6_assert_score_bias_signs():
    ba = BlockedAgency(BlockedAgencyConfig(use_blocked_agency=True, noop_class=0))
    # Drive an asserting block on action class 2.
    for _ in range(8):
        ba.update(0.9, 0.9, True, 1.0, blocked_action_class=2)
    bias = ba.compute_assert_score_bias(
        [0, 1, 2, 3], device=torch.device("cpu"), dtype=torch.float32)
    assert bias[0].item() > 0.0          # no-op penalised (positive)
    assert bias[1].item() < 0.0          # other action favoured (negative)
    assert bias[3].item() < 0.0          # other action favoured (negative)
    # blocked class 2 gets a less-favourable bias than an unblocked action.
    assert bias[2].item() > bias[1].item()
    # zero when no asserting block this tick.
    ba.reset()
    z = ba.compute_assert_score_bias([0, 1, 2, 3],
                                     device=torch.device("cpu"), dtype=torch.float32)
    assert torch.allclose(z, torch.zeros(4))


# ---------------------------------------------------------------- C7
def test_c7_decommit_and_mech094():
    ba = BlockedAgency(BlockedAgencyConfig(
        use_blocked_agency=True, decommit_bound=0.5,
        decommit_consecutive_ticks=3, accumulation_rate=1.0))
    fired = False
    for _ in range(12):
        out = ba.update(0.9, 0.9, True, 1.0)
        fired = fired or out.decommit_signal
    assert fired is True
    # MECH-094: simulation_mode never fires a decommit and does not advance state.
    z_before = ba.get_z_block()
    out_sim = ba.update(0.9, 0.9, True, 1.0, simulation_mode=True)
    assert out_sim.decommit_signal is False
    assert ba.get_z_block() == z_before


# ---------------------------------------------------------------- C8
def test_c8_env_action_block():
    # OFF: bit-identical (no info tags toggled; agent moves normally).
    env_off = CausalGridWorldV2(size=8, seed=5)
    _, _ = env_off.reset()
    _, h, d, inf, _ = env_off.step(torch.eye(4)[1])
    assert inf["scheduled_action_block_enabled"] is False
    assert inf["action_blocked_this_step"] is False

    # ON: every step (after step 0) blocked. On a BLOCKED step the agent stays
    # put, the block adds no harm, and the transition is tagged "action_blocked".
    # (Step 0 is never blocked -- the steps>0 guard mirrors the scheduled-hazard
    # precedent -- so the agent may move / incur ambient harm there.)
    env = CausalGridWorldV2(size=8, seed=5,
                            scheduled_action_block_enabled=True,
                            scheduled_action_block_interval=1,
                            scheduled_action_block_prob=1.0)
    _, _ = env.reset()
    blocked = 0
    for _ in range(10):
        before = (env.agent_x, env.agent_y)
        _, h, d, inf, _ = env.step(torch.eye(4)[1])  # action 1 (a move)
        if inf["action_blocked_this_step"]:
            blocked += 1
            assert (env.agent_x, env.agent_y) == before  # agent did not move
            assert inf.get("transition_type") == "action_blocked"
            assert h == 0.0  # the block itself inflicts no harm
    assert blocked >= 1
    assert inf["action_block_event_count"] >= 1


# ---------------------------------------------------------------- C9
def test_c9_latent_z_block_field():
    env = CausalGridWorldV2(size=8, seed=5,
                            scheduled_action_block_enabled=True,
                            scheduled_action_block_interval=1,
                            scheduled_action_block_prob=1.0)
    ag = REEAgent(_build(env, use_blocked_agency=True, z_goal_enabled=True,
                         drive_weight=2.0))
    _, od = env.reset()
    _seed_goal(ag)
    lat = ag.sense(od["body_state"], od["world_state"])
    assert lat.z_block is not None
    assert tuple(lat.z_block.shape) == (1, 1)
    # detach() preserves the field.
    det = lat.detach()
    assert det.z_block is not None
    assert torch.allclose(det.z_block, lat.z_block)
