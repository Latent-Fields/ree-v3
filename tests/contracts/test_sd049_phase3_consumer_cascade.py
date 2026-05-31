"""SD-049 Phase 3 consumer cascade contracts.

Validates the migration of the 7 SD-032 consumer modules (AIC, PCC, pACC,
dACC, SalienceCoordinator, BroadcastOverrideRegulator, MECH-295 liking-
bridge) from reading goal_state._last_drive_level (collapsed scalar) to
reading obs_dict["per_axis_drive"] directly when SD-049 per-axis is on.

All consumer changes preserve bit-identical OFF: when per_axis_drive=None
(legacy / SD-049 OFF / master flag OFF), every consumer takes its legacy
scalar drive_level path unchanged.

Contracts:
    C1: shared collapse_per_axis_drive helper -- None passthrough; modes
        {max, mean, sum}; bounds; ValueError on unknown mode.
    C2: select_axis helper -- None passthrough; out-of-range returns None;
        bounds.
    C3: REEConfig has the 8 SD-049 Phase 3 fields (master flag + 7 per-
        consumer combiners) with the documented defaults.
    C4: REEConfig.from_dims surfaces all 8 fields.
    C5: agent.config.use_sd049_per_axis_consumer_cascade defaults False;
        _per_axis_drive_for_consumers() returns None.
    C6: each of the 7 consumer modules accepts per_axis_drive kwarg and
        produces bit-identical output to the scalar path when per_axis_drive
        is None.
    C7: each of the 6 whole-organism consumers (AIC/PCC/pACC/dACC/salience/
        override) collapses per_axis_drive via its combiner and produces
        output identical to passing the combined scalar as drive_level
        (the SD-049 "axis-aware via combiner" semantic).
    C8: MECH-295 axis-matched routing -- goal_axis_idx selects the deficit
        channel; per-axis vector with high deficit on goal axis + low on
        others produces stronger bias than max-collapse fallback (when
        goal axis is not the max axis).
    C9: agent end-to-end -- master flag ON + non-None obs_per_axis_drive
        in sense() -> the cached vector flows through and is observable
        via the helper; OFF -> None.
    C10: bit-identical regression -- master flag OFF, agent.sense() with
        the new kwarg produces identical LatentState as without the kwarg.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from ree_core.utils.per_axis_drive import (
    collapse_per_axis_drive,
    select_axis,
    validate_combiner,
)
from ree_core.utils.config import REEConfig


# ----- C1: shared collapse helper -----


def test_c1_collapse_none_passthrough():
    assert collapse_per_axis_drive(None) is None
    assert collapse_per_axis_drive([]) is None


def test_c1_collapse_modes_basic():
    v = [0.1, 0.5, 0.9]
    assert collapse_per_axis_drive(v, "max") == pytest.approx(0.9)
    assert collapse_per_axis_drive(v, "mean") == pytest.approx(0.5)
    assert collapse_per_axis_drive(v, "sum") == pytest.approx(1.0)  # clipped


def test_c1_collapse_clips_to_unit_interval():
    # sum is the only combiner that can exceed [0, 1]; verify clip
    assert collapse_per_axis_drive([0.6, 0.6, 0.6], "sum") == pytest.approx(1.0)
    assert collapse_per_axis_drive([-0.1, 0.0, 0.0], "max") == pytest.approx(0.0)


def test_c1_collapse_accepts_numpy_torch():
    np_v = np.array([0.2, 0.4, 0.6], dtype=np.float32)
    t_v = torch.tensor([0.2, 0.4, 0.6])
    assert collapse_per_axis_drive(np_v, "max") == pytest.approx(0.6)
    assert collapse_per_axis_drive(t_v, "max") == pytest.approx(0.6)


def test_c1_collapse_rejects_unknown_mode():
    with pytest.raises(ValueError):
        collapse_per_axis_drive([0.1, 0.2], "argmax")


def test_c1_validate_combiner():
    for mode in ("max", "mean", "sum"):
        assert validate_combiner(mode) == mode
    with pytest.raises(ValueError):
        validate_combiner("median")


# ----- C2: select_axis helper -----


def test_c2_select_axis_none_passthrough():
    assert select_axis(None, 0) is None
    assert select_axis([0.1, 0.2], None) is None
    assert select_axis([], 0) is None


def test_c2_select_axis_out_of_range():
    assert select_axis([0.1, 0.2], -1) is None
    assert select_axis([0.1, 0.2], 2) is None


def test_c2_select_axis_returns_clipped():
    assert select_axis([0.1, 1.5, -0.2], 0) == pytest.approx(0.1)
    assert select_axis([0.1, 1.5, -0.2], 1) == pytest.approx(1.0)
    assert select_axis([0.1, 1.5, -0.2], 2) == pytest.approx(0.0)


# ----- C3: REEConfig defaults -----


def test_c3_reeconfig_has_phase3_fields_with_documented_defaults():
    cfg = REEConfig()
    assert cfg.use_sd049_per_axis_consumer_cascade is False
    assert cfg.sd049_aic_per_axis_combiner == "max"
    assert cfg.sd049_pcc_per_axis_combiner == "mean"
    assert cfg.sd049_pacc_per_axis_combiner == "sum"
    assert cfg.sd049_dacc_per_axis_combiner == "max"
    assert cfg.sd049_salience_per_axis_combiner == "max"
    assert cfg.sd049_override_per_axis_combiner == "max"
    assert cfg.sd049_mech295_per_axis_combiner == "max"


# ----- C4: from_dims surfaces all fields -----


def test_c4_from_dims_surfaces_phase3_fields():
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sd049_per_axis_consumer_cascade=True,
        sd049_aic_per_axis_combiner="mean",
        sd049_mech295_per_axis_combiner="sum",
    )
    assert cfg.use_sd049_per_axis_consumer_cascade is True
    assert cfg.sd049_aic_per_axis_combiner == "mean"
    assert cfg.sd049_mech295_per_axis_combiner == "sum"
    # untouched fields keep defaults
    assert cfg.sd049_pcc_per_axis_combiner == "mean"


# ----- C6: each consumer accepts per_axis_drive kwarg, None -> bit-identical -----


def test_c6_aic_bit_identical_when_per_axis_drive_none():
    from ree_core.cingulate.aic_analog import AICAnalog, AICConfig

    a = AICAnalog(AICConfig())
    out1 = a.tick(z_harm_a_norm=0.5, drive_level=0.4, beta_gate_elevated=False)
    b = AICAnalog(AICConfig())
    out2 = b.tick(
        z_harm_a_norm=0.5, drive_level=0.4, beta_gate_elevated=False,
        per_axis_drive=None,
    )
    assert out1 == out2


def test_c6_pcc_bit_identical_when_per_axis_drive_none():
    from ree_core.cingulate.pcc_analog import PCCAnalog, PCCConfig

    p1 = PCCAnalog(PCCConfig())
    p2 = PCCAnalog(PCCConfig())
    out1 = p1.tick(drive_level=0.6)
    out2 = p2.tick(drive_level=0.6, per_axis_drive=None)
    assert out1 == out2


def test_c6_pacc_bit_identical_when_per_axis_drive_none():
    from ree_core.cingulate.pacc_analog import PACCAnalog, PACCConfig

    p1 = PACCAnalog(PACCConfig())
    p2 = PACCAnalog(PACCConfig())
    out1 = p1.tick(z_harm_a_norm=0.3, write_gate=1.0)
    out2 = p2.tick(z_harm_a_norm=0.3, write_gate=1.0, per_axis_drive=None)
    assert out1 == out2
    # effective_drive_from_per_axis(None) -> None falls back to scalar path.
    assert p1.effective_drive_from_per_axis(None) is None


def test_c6_dacc_bit_identical_when_per_axis_drive_none():
    from ree_core.cingulate.dacc import DACCAdaptiveControl, DACCConfig

    cfg = DACCConfig(dacc_drive_coupling=0.5)
    d1 = DACCAdaptiveControl(cfg)
    d2 = DACCAdaptiveControl(cfg)
    z = torch.tensor([0.1, 0.2, 0.3])
    payoffs = torch.tensor([0.1, 0.2, 0.3])
    effort = torch.tensor([1.0, 1.0, 1.0])
    classes = [0, 1, 2]
    b1 = d1(z, None, payoffs, effort, classes, precision=1.0, drive_level=0.4)
    b2 = d2(
        z, None, payoffs, effort, classes,
        precision=1.0, drive_level=0.4, per_axis_drive=None,
    )
    assert b1["drive_gain"] == pytest.approx(b2["drive_gain"])
    assert torch.allclose(b1["mode_ev"], b2["mode_ev"])


def test_c6_salience_bit_identical_when_per_axis_drive_none():
    from ree_core.cingulate.salience_coordinator import (
        SalienceCoordinator,
        SalienceCoordinatorConfig,
    )

    s1 = SalienceCoordinator(SalienceCoordinatorConfig())
    s2 = SalienceCoordinator(SalienceCoordinatorConfig())
    o1 = s1.tick(drive_level=0.5)
    o2 = s2.tick(drive_level=0.5, per_axis_drive=None)
    assert o1["operating_mode"] == o2["operating_mode"]
    assert o1["current_mode"] == o2["current_mode"]


def test_c6_override_bit_identical_when_per_axis_drive_none():
    from ree_core.regulators.broadcast_override import (
        BroadcastOverrideConfig,
        BroadcastOverrideRegulator,
    )

    b1 = BroadcastOverrideRegulator(BroadcastOverrideConfig())
    b2 = BroadcastOverrideRegulator(BroadcastOverrideConfig())
    v1 = b1.tick(drive_level=0.7, z_harm_norm=0.3)
    v2 = b2.tick(drive_level=0.7, z_harm_norm=0.3, per_axis_drive=None)
    assert v1 == pytest.approx(v2)


def test_c6_mech295_bit_identical_when_per_axis_drive_none():
    from ree_core.regulators.mech295_liking_bridge import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )

    cfg = MECH295LikingBridgeConfig()
    m1 = MECH295LikingBridge(cfg)
    m2 = MECH295LikingBridge(cfg)
    w1 = m1.compute_anticipatory_liking_write(drive_level=0.5, z_goal_norm=0.5)
    w2 = m2.compute_anticipatory_liking_write(
        drive_level=0.5, z_goal_norm=0.5, per_axis_drive=None,
    )
    assert w1 == pytest.approx(w2)
    prox = torch.tensor([0.5, 0.3, 0.8])
    b1 = m1.compute_approach_cue_score_bias(0.5, prox)
    b2 = m2.compute_approach_cue_score_bias(0.5, prox, per_axis_drive=None)
    assert torch.allclose(b1, b2)


# ----- C7: combiner correctness -----


def test_c7_aic_uses_max_combiner_default():
    from ree_core.cingulate.aic_analog import AICAnalog, AICConfig

    a = AICAnalog(AICConfig())
    # per_axis_drive = [0.1, 0.2, 0.9] -> max=0.9 should override drive_level=0.0
    out = a.tick(
        z_harm_a_norm=0.5, drive_level=0.0, beta_gate_elevated=False,
        per_axis_drive=[0.1, 0.2, 0.9],
    )
    # urgency_scaled = ratio * (1 + drive_coupling * 0.9)
    # vs scalar 0.0 path which would not lift urgency.
    a_ref = AICAnalog(AICConfig())
    out_ref = a_ref.tick(
        z_harm_a_norm=0.5, drive_level=0.9, beta_gate_elevated=False,
    )
    # Both AICs should produce the same aic_salience (max-collapsed vs
    # explicit scalar 0.9 are equivalent under default combiner "max").
    assert out["aic_salience"] == pytest.approx(out_ref["aic_salience"])


def test_c7_pcc_uses_mean_combiner_default():
    from ree_core.cingulate.pcc_analog import PCCAnalog, PCCConfig

    p1 = PCCAnalog(PCCConfig())
    p2 = PCCAnalog(PCCConfig())
    # mean([0.0, 0.5, 1.0]) = 0.5
    out1 = p1.tick(drive_level=0.0, per_axis_drive=[0.0, 0.5, 1.0])
    out2 = p2.tick(drive_level=0.5)
    assert out1["pcc_stability"] == pytest.approx(out2["pcc_stability"])


def test_c7_override_uses_max_combiner_default():
    from ree_core.regulators.broadcast_override import (
        BroadcastOverrideConfig,
        BroadcastOverrideRegulator,
    )

    cfg = BroadcastOverrideConfig()
    b1 = BroadcastOverrideRegulator(cfg)
    b2 = BroadcastOverrideRegulator(cfg)
    v1 = b1.tick(drive_level=0.0, z_harm_norm=0.0, per_axis_drive=[0.0, 0.0, 0.95])
    v2 = b2.tick(drive_level=0.95, z_harm_norm=0.0)
    assert v1 == pytest.approx(v2)


def test_c7_pacc_effective_drive_from_per_axis_sum_default():
    from ree_core.cingulate.pacc_analog import PACCAnalog, PACCConfig

    p = PACCAnalog(PACCConfig())
    # No drive_bias accumulated; effective_drive_from_per_axis should match
    # effective_drive(sum_collapsed) under the default "sum" combiner.
    eff = p.effective_drive_from_per_axis([0.2, 0.3, 0.4])  # sum->0.9
    assert eff == pytest.approx(0.9, abs=1e-6)


def test_c7_dacc_drive_gain_uses_max_combiner_default():
    from ree_core.cingulate.dacc import DACCAdaptiveControl, DACCConfig

    cfg = DACCConfig(dacc_drive_coupling=1.0)
    d1 = DACCAdaptiveControl(cfg)
    d2 = DACCAdaptiveControl(cfg)
    z = torch.tensor([0.1, 0.1, 0.1])
    payoffs = torch.zeros(3)
    effort = torch.ones(3)
    classes = [0, 1, 2]
    b1 = d1(
        z, None, payoffs, effort, classes,
        precision=1.0, drive_level=0.0,
        per_axis_drive=[0.0, 0.0, 0.8],
    )
    b2 = d2(
        z, None, payoffs, effort, classes,
        precision=1.0, drive_level=0.8,
    )
    assert b1["drive_gain"] == pytest.approx(b2["drive_gain"])


# ----- C8: MECH-295 axis-matched routing -----


def test_c8_mech295_axis_matched_overrides_combiner():
    from ree_core.regulators.mech295_liking_bridge import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )

    cfg = MECH295LikingBridgeConfig(min_drive_to_fire=0.01, min_z_goal_norm_to_fire=0.01)
    bridge = MECH295LikingBridge(cfg)
    # per_axis_drive: hunger=0.9 (max), thirst=0.1, curiosity=0.2.
    # goal axis = thirst (idx=1): low deficit -> small write.
    write_thirst_goal = bridge.compute_anticipatory_liking_write(
        drive_level=0.5,
        z_goal_norm=0.5,
        per_axis_drive=[0.9, 0.1, 0.2],
        goal_axis_idx=1,
    )
    # goal axis = hunger (idx=0): high deficit -> large write.
    bridge2 = MECH295LikingBridge(cfg)
    write_hunger_goal = bridge2.compute_anticipatory_liking_write(
        drive_level=0.5,
        z_goal_norm=0.5,
        per_axis_drive=[0.9, 0.1, 0.2],
        goal_axis_idx=0,
    )
    # Axis-matched writes diverge: hunger goal >> thirst goal.
    assert write_hunger_goal > write_thirst_goal
    # Fallback (None goal_axis_idx) uses max combiner -> equivalent to hunger.
    bridge3 = MECH295LikingBridge(cfg)
    write_fallback = bridge3.compute_anticipatory_liking_write(
        drive_level=0.5,
        z_goal_norm=0.5,
        per_axis_drive=[0.9, 0.1, 0.2],
        goal_axis_idx=None,
    )
    assert write_fallback == pytest.approx(write_hunger_goal)


def test_c8_mech295_score_bias_axis_matched():
    from ree_core.regulators.mech295_liking_bridge import (
        MECH295LikingBridge,
        MECH295LikingBridgeConfig,
    )

    cfg = MECH295LikingBridgeConfig(min_drive_to_fire=0.01)
    bridge = MECH295LikingBridge(cfg)
    prox = torch.tensor([0.5, 0.5, 0.5])
    bias_low = bridge.compute_approach_cue_score_bias(
        drive_level=0.5, candidate_proximities=prox,
        per_axis_drive=[0.9, 0.1, 0.2], goal_axis_idx=1,  # thirst=0.1
    )
    bridge2 = MECH295LikingBridge(cfg)
    bias_high = bridge2.compute_approach_cue_score_bias(
        drive_level=0.5, candidate_proximities=prox,
        per_axis_drive=[0.9, 0.1, 0.2], goal_axis_idx=0,  # hunger=0.9
    )
    # bias is negative; high-deficit axis produces a more-negative bias.
    assert bias_high.mean().item() < bias_low.mean().item()


# ----- C9 / C10: agent end-to-end -----


def _build_test_agent(use_cascade: bool = False):
    from ree_core.agent import REEAgent

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_aic_analog=True,
        use_pacc_analog=True,
        use_pcc_analog=True,
        use_sd049_per_axis_consumer_cascade=use_cascade,
    )
    return REEAgent(cfg), cfg


def test_c9_agent_phase3_helper_returns_none_when_master_off():
    agent, cfg = _build_test_agent(use_cascade=False)
    assert agent._per_axis_drive_for_consumers() is None
    # sense() with obs_per_axis_drive: stores cache, but helper still returns
    # None because master flag is off.
    obs_body = torch.zeros(1, cfg.latent.body_obs_dim)
    obs_world = torch.zeros(1, cfg.latent.world_obs_dim)
    agent.sense(obs_body, obs_world, obs_per_axis_drive=torch.tensor([0.1, 0.2, 0.3]))
    assert agent._per_axis_drive is not None  # cache populated
    assert agent._per_axis_drive_for_consumers() is None  # gate closed


def test_c9_agent_phase3_helper_returns_vector_when_master_on():
    agent, cfg = _build_test_agent(use_cascade=True)
    obs_body = torch.zeros(1, cfg.latent.body_obs_dim)
    obs_world = torch.zeros(1, cfg.latent.world_obs_dim)
    agent.sense(obs_body, obs_world, obs_per_axis_drive=torch.tensor([0.1, 0.2, 0.3]))
    got = agent._per_axis_drive_for_consumers()
    assert got is not None
    assert torch.allclose(got.cpu(), torch.tensor([0.1, 0.2, 0.3]))


def test_c10_agent_sense_bit_identical_with_master_off():
    """Master flag OFF: passing obs_per_axis_drive must not alter sense().

    Uses one agent so weight init is identical between calls. Tick A omits
    obs_per_axis_drive; tick B passes a vector; both should yield identical
    LatentState because the master flag gates downstream consumption.
    """
    agent, cfg = _build_test_agent(use_cascade=False)
    obs_body = torch.randn(1, cfg.latent.body_obs_dim)
    obs_world = torch.randn(1, cfg.latent.world_obs_dim)
    # First tick: no obs_per_axis_drive.
    ls_a = agent.sense(obs_body, obs_world)
    z_world_a = ls_a.z_world.detach().clone()
    z_self_a = ls_a.z_self.detach().clone()
    # Reset and re-sense with the same inputs + obs_per_axis_drive.
    agent.reset()
    ls_b = agent.sense(
        obs_body, obs_world,
        obs_per_axis_drive=torch.tensor([0.4, 0.5, 0.6]),
    )
    # Bit-identical under master OFF: passing the vector caches it but
    # downstream consumers are gated by the master flag.
    assert torch.allclose(z_world_a, ls_b.z_world, atol=1e-7)
    assert torch.allclose(z_self_a, ls_b.z_self, atol=1e-7)
