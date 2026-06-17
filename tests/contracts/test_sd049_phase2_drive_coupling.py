"""SD-049-PHASE-2 drive-coupling amend contract (failure_autopsy_V3-EXQ-514r, MECH-436).

Two coupled no-op-default levers, kappa load-bearing:

  (a) GoalConfig.incentive_drive_kappa_scale -- scales the effective drive->score
      coupling kappa in IncentiveTokenBank.wanting(). Default 1.0 = bit-identical;
      >1 lets a realistic per-axis drive spread compete with real object base_value
      gaps (the 514r overshoot showed real gaps on seeds 45/46/47 exceed 0.5).
  (b) CausalGridWorldV2.per_axis_restoration_fraction -- fraction of the curve
      deficit restored on contact. Default 1.0 = bit-identical full-restore-to-0;
      <1 leaves STANDING per-axis drive so the spread survives to the WL scoring
      moment (around consumption) instead of being equalised to ~0.006.

C1  kappa_scale default 1.0 -> wanting() byte-identical to the unscaled formula.
C2  kappa_scale > 1 amplifies the drive term and can flip a most_wanted argmax that
    the unscaled kappa cannot (the load-bearing behaviour).
C3  restoration_fraction default 1.0 -> contact fully restores the axis (legacy).
C4  restoration_fraction < 1.0 -> contact leaves standing drive on the axis.
C5  from_dims surfaces incentive_drive_kappa_scale onto cfg.goal; default 1.0.
C6  restoration_fraction clamps to [0, 1]; the kappa_scale getattr fallback is 1.0.
"""
import torch

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.goal import GoalConfig, IncentiveTokenBank
from ree_core.utils.config import REEConfig


def _bank(**kw):
    cfg = GoalConfig(goal_dim=4, use_incentive_token_bank=True,
                     incentive_decay=0.0, incentive_value_alpha=1.0, **kw)
    return IncentiveTokenBank(cfg, torch.device("cpu"))


# --- Lever (a): kappa scale ---------------------------------------------------

def test_c1_kappa_scale_default_bit_identical():
    cfg = GoalConfig(goal_dim=4, use_incentive_token_bank=True)
    assert cfg.incentive_drive_kappa_scale == 1.0
    bank = _bank(incentive_drive_kappa_weight=2.0)  # scale defaults 1.0
    z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    bank.update(1, 0.5, z)
    w = bank.wanting(per_axis_drive=torch.tensor([0.9, 0.0, 0.0]))
    # base_value 0.5 * (1 + 2.0*0.9) = 1.4 -- unchanged by the default scale.
    # 1e-6 tolerance: drive_axis is read from a float32 tensor element.
    assert abs(w[1] - 0.5 * (1 + 2.0 * 0.9)) < 1e-6


def test_c2_kappa_scale_amplifies_and_flips_argmax():
    # A high-base low-drive object vs a low-base high-drive object. At the
    # baseline kappa the high-base object wins; scaling kappa flips most_wanted
    # to the high-drive object (the realistic-spread-vs-real-base_value-gap case).
    food = torch.tensor([[1.0, 0.0, 0.0, 0.0]])   # base_value 1.0, axis 0 (low drive)
    water = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # base_value 0.6, axis 1 (high drive)
    drive = torch.tensor([0.05, 0.30, 0.0])

    base = _bank(incentive_drive_kappa_weight=2.0, incentive_drive_kappa_scale=1.0)
    base.update(1, 1.0, food)
    base.update(2, 0.6, water)
    k_base, _, _ = base.most_wanted(per_axis_drive=drive)
    # 1.0*(1+2*0.05)=1.10  vs  0.6*(1+2*0.30)=0.96 -> food wins at baseline kappa.
    assert k_base == 1

    scaled = _bank(incentive_drive_kappa_weight=2.0, incentive_drive_kappa_scale=12.0)
    scaled.update(1, 1.0, food)
    scaled.update(2, 0.6, water)
    k_scaled, z_scaled, _ = scaled.most_wanted(per_axis_drive=drive)
    # 1.0*(1+24*0.05)=2.20  vs  0.6*(1+24*0.30)=4.92 -> water wins once kappa scales.
    assert k_scaled == 2 and torch.allclose(z_scaled, water)


def test_c5_from_dims_surfaces_kappa_scale():
    cfg = REEConfig.from_dims(world_obs_dim=32, body_obs_dim=12, action_dim=4)
    assert cfg.goal.incentive_drive_kappa_scale == 1.0  # default
    cfg2 = REEConfig.from_dims(world_obs_dim=32, body_obs_dim=12, action_dim=4,
                               incentive_drive_kappa_scale=8.0)
    assert cfg2.goal.incentive_drive_kappa_scale == 8.0


def test_c6_kappa_scale_getattr_fallback_is_one():
    # A GoalConfig built without the field (simulated via a bare object) must not
    # raise -- wanting() uses getattr(..., 1.0). Real GoalConfig always has it.
    bank = _bank(incentive_drive_kappa_weight=2.0)
    delattr(bank.config, "incentive_drive_kappa_scale")
    bank.update(1, 0.5, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    w = bank.wanting(per_axis_drive=torch.tensor([0.9, 0.0, 0.0]))
    assert abs(w[1] - 0.5 * (1 + 2.0 * 0.9)) < 1e-6


# --- Lever (b): partial restoration ------------------------------------------

def _consume_typed_resource(restoration_fraction, cur_drive=0.5):
    """Build a multi-resource env, force a typed (sigmoidal, axis 0) resource into
    the agent's east neighbour, set the axis drive, and step east to consume it.
    Returns the axis-0 drive immediately after the consumption step."""
    env = CausalGridWorldV2(
        size=8, num_hazards=0, num_resources=1, seed=7,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=3,
        per_axis_drive_enabled=True,
        per_axis_restoration_fraction=restoration_fraction,
    )
    env.reset()
    env.agent_x, env.agent_y = 2, 2
    tx, ty = 3, 2  # action 1 = (1, 0)
    env.grid[tx, ty] = env.ENTITY_TYPES["resource"]
    env.resources = [[tx, ty]]
    env._resource_type_grid[:, :] = 0
    env._resource_type_grid[tx, ty] = 1  # type 0 (sigmoidal_saturating)
    env._resources_by_type = [[[tx, ty]], [], []]
    env._per_axis_drive[:] = 0.0
    env._per_axis_drive[0] = cur_drive
    env.step(1)  # move east -> consume
    return float(env._per_axis_drive[0])


def test_c3_full_restoration_default_bit_identical():
    assert CausalGridWorldV2(size=8).per_axis_restoration_fraction == 1.0
    # fraction 1.0: sigmoidal fully restores 0.5 -> 0.0, then one step of depletion
    # (decay[0]=0.001) -> axis ~= 0.001. Essentially fully restored.
    axis = _consume_typed_resource(1.0, cur_drive=0.5)
    assert axis < 0.02


def test_c4_partial_restoration_leaves_standing_drive():
    # fraction 0.5: restore 0.25 of the 0.5 deficit -> standing 0.25 (+ ~0.001
    # depletion). The argmax-relevant standing spread the 514r autopsy needs.
    axis = _consume_typed_resource(0.5, cur_drive=0.5)
    assert 0.24 < axis < 0.28


def test_c6_restoration_fraction_clamps():
    assert CausalGridWorldV2(size=8, per_axis_restoration_fraction=-1.0).per_axis_restoration_fraction == 0.0
    assert CausalGridWorldV2(size=8, per_axis_restoration_fraction=5.0).per_axis_restoration_fraction == 1.0
