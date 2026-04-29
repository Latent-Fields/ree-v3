"""Contract tests for MECH-269b + MECH-284 staleness wiring (2026-04-29).

Verifies the substrate change that wires staleness_accumulator into
VsRolloutGate.gate() before the threshold comparison so MECH-269b's
stale-stream-discrimination claim becomes testable without smoke
threshold overrides (Q-040b strong reading).

Guarantees enforced here:
  C1. Default VsRolloutGateConfig has use_staleness_lookup=False. Default
      REEConfig surface has use_vs_gate_staleness_lookup=False. Backward
      compatible.
  C2. With use_staleness_lookup=False, supplying a per_stream_staleness
      dict to gate() / gate_stream() has no effect (legacy raw-V_s path).
  C3. With use_staleness_lookup=True and no per_stream_staleness supplied,
      gate() falls back to the raw-V_s path -- equivalent to staleness=0
      for all streams.
  C4. With use_staleness_lookup=True and a per_stream_staleness dict that
      pushes effective_vs below threshold, gate() substitutes the held
      snapshot even when raw V_s is at or above threshold.
  C5. Per-stream isolation: staleness on one stream does not change the
      gate decision for a different stream.
  C6. HippocampalModule.compute_per_stream_staleness aggregates region
      staleness via max-over-active-anchors-whose-stream_mixture-includes-stream.
  C7. Diagnostics expose staleness lookup activity (call counter +
      per-stream max staleness) and reset clears them.
  C8. Agent precondition: use_vs_gate_staleness_lookup=True without
      use_staleness_accumulator=True OR without use_anchor_sets=True
      raises ValueError at agent build time.

ASCII-only output.
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _make_gate(use_staleness_lookup: bool = False,
               e1_threshold: float = 0.4,
               e2_threshold: float = 0.4):
    from ree_core.regulators.vs_rollout_gate import (
        VsRolloutGate, VsRolloutGateConfig,
    )
    cfg = VsRolloutGateConfig(
        streams=("z_world", "z_self", "z_harm_s", "z_harm_a", "z_goal", "z_beta"),
        snapshot_refresh_threshold=0.5,
        e1_threshold=e1_threshold,
        e2_threshold=e2_threshold,
        use_staleness_lookup=use_staleness_lookup,
    )
    return VsRolloutGate(cfg)


def _make_latent(z_world=None, z_self=None, z_harm=None, z_harm_a=None, z_beta=None):
    from ree_core.latent.stack import LatentState
    return LatentState(
        z_self=z_self if z_self is not None else torch.zeros(1, 8),
        z_world=z_world if z_world is not None else torch.zeros(1, 8),
        z_beta=z_beta if z_beta is not None else torch.zeros(1, 8),
        z_theta=torch.zeros(1, 8),
        z_delta=torch.zeros(1, 8),
        precision={},
        z_harm=z_harm if z_harm is not None else torch.zeros(1, 8),
        z_harm_a=z_harm_a if z_harm_a is not None else torch.zeros(1, 8),
    )


# ---------------------------------------------------------------------- #
# C1 -- defaults are backward compatible
# ---------------------------------------------------------------------- #
def test_c1_default_config_backward_compat():
    from ree_core.regulators.vs_rollout_gate import VsRolloutGateConfig
    from ree_core.utils.config import HippocampalConfig

    assert VsRolloutGateConfig().use_staleness_lookup is False
    assert getattr(
        HippocampalConfig(world_dim=8, action_dim=4, action_object_dim=4),
        "use_vs_gate_staleness_lookup",
        False,
    ) is False


# ---------------------------------------------------------------------- #
# C2 -- staleness ignored when flag is off
# ---------------------------------------------------------------------- #
def test_c2_staleness_ignored_when_flag_off():
    gate = _make_gate(use_staleness_lookup=False, e1_threshold=0.4)
    z_world = torch.ones(1, 8)
    latent = _make_latent(z_world=z_world)
    per_stream_vs = {"z_world": 0.9}  # well above threshold
    # Refresh snapshot so a substitution would be possible.
    gate.update_snapshots(latent, per_stream_vs)
    # Even with a huge staleness, flag-OFF must not subtract.
    out = gate.gate(
        latent, per_stream_vs, side="e1",
        per_stream_staleness={"z_world": 0.99},
    )
    # No held substitution: returned latent is the input.
    diag = gate.get_diagnostics()
    assert diag["vs_gate_held_e1_z_world"] == 0
    assert diag["vs_gate_staleness_lookup_calls"] == 0
    # And the output z_world is the original (no replace).
    assert torch.allclose(out.z_world, z_world)


# ---------------------------------------------------------------------- #
# C3 -- staleness lookup ON without dict falls back to raw V_s path
# ---------------------------------------------------------------------- #
def test_c3_lookup_on_without_dict_falls_back_to_raw_vs():
    gate = _make_gate(use_staleness_lookup=True, e1_threshold=0.4)
    latent = _make_latent(z_world=torch.ones(1, 8))
    per_stream_vs = {"z_world": 0.9}
    gate.update_snapshots(latent, per_stream_vs)
    # No staleness dict passed -> raw V_s 0.9 >= 0.4 -> no hold.
    gate.gate(latent, per_stream_vs, side="e1", per_stream_staleness=None)
    diag = gate.get_diagnostics()
    assert diag["vs_gate_held_e1_z_world"] == 0
    assert diag["vs_gate_staleness_lookup_calls"] == 0


# ---------------------------------------------------------------------- #
# C4 -- staleness pushes effective V_s below threshold -> hold fires
# ---------------------------------------------------------------------- #
def test_c4_staleness_triggers_hold_at_realistic_vs():
    gate = _make_gate(use_staleness_lookup=True, e1_threshold=0.4)
    z_world = torch.ones(1, 8)  # current value 1s
    latent = _make_latent(z_world=z_world)
    per_stream_vs = {"z_world": 0.9}
    # Seed the snapshot at the original value, then perturb the live latent.
    gate.update_snapshots(latent, per_stream_vs)
    perturbed = _make_latent(z_world=torch.full((1, 8), 5.0))
    # 0.9 - 0.7 = 0.2 < 0.4 -> hold should fire and substitute the
    # original (1s) snapshot for the perturbed (5s) current value.
    out = gate.gate(
        perturbed, per_stream_vs, side="e1",
        per_stream_staleness={"z_world": 0.7},
    )
    diag = gate.get_diagnostics()
    assert diag["vs_gate_held_e1_z_world"] == 1
    assert diag["vs_gate_staleness_lookup_calls"] >= 1
    assert diag["vs_gate_max_staleness_z_world"] == pytest.approx(0.7)
    # Snapshot substituted: out.z_world equals the original 1s, not the 5s.
    assert torch.allclose(out.z_world, z_world)


# ---------------------------------------------------------------------- #
# C5 -- per-stream isolation
# ---------------------------------------------------------------------- #
def test_c5_per_stream_isolation():
    gate = _make_gate(use_staleness_lookup=True, e1_threshold=0.4)
    z_world = torch.ones(1, 8)
    z_self = torch.full((1, 8), 2.0)
    latent = _make_latent(z_world=z_world, z_self=z_self)
    per_stream_vs = {"z_world": 0.9, "z_self": 0.9}
    gate.update_snapshots(latent, per_stream_vs)
    # Stale only z_world. z_self.staleness defaults to 0 via .get().
    perturbed = _make_latent(
        z_world=torch.full((1, 8), 5.0),
        z_self=torch.full((1, 8), 9.0),
    )
    out = gate.gate(
        perturbed, per_stream_vs, side="e1",
        per_stream_staleness={"z_world": 0.7},
    )
    diag = gate.get_diagnostics()
    assert diag["vs_gate_held_e1_z_world"] == 1
    assert diag["vs_gate_held_e1_z_self"] == 0
    # z_self is the perturbed live value; z_world is the held snapshot.
    assert torch.allclose(out.z_self, torch.full((1, 8), 9.0))
    assert torch.allclose(out.z_world, z_world)


# ---------------------------------------------------------------------- #
# C6 -- HippocampalModule.compute_per_stream_staleness aggregator
# ---------------------------------------------------------------------- #
def test_c6_compute_per_stream_staleness_max_over_anchors():
    from ree_core.utils.config import (
        HippocampalConfig, E2Config, ResidueConfig,
    )
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.residue.field import ResidueField
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.hippocampal.anchor_set import AnchorSet, AnchorSetConfig

    hcfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=4,
        hidden_dim=16, horizon=3, num_candidates=4,
        num_cem_iterations=1, elite_fraction=0.5,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_staleness_accumulator=True,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=8, world_dim=8, action_dim=4, action_object_dim=4,
    ))
    residue = ResidueField(ResidueConfig(world_dim=8, num_basis_functions=4))
    mod = HippocampalModule(hcfg, e2, residue)
    # Seed per_stream_vs so the aggregator has streams to aggregate over.
    mod.per_stream_vs = {"z_world": 0.9, "z_self": 0.9, "z_harm_s": 0.9}
    # Two active anchors on different (scale, segment_id) keys with
    # distinct stream_mixtures. anchor_a covers z_world; anchor_b covers
    # both z_world and z_harm_s.
    z = torch.zeros(1, 8)
    a = mod.anchor_set.write_anchor(
        scale="fast", segment_id="0.1",
        stream_mixture=("z_world",), z_world=z,
    )
    b = mod.anchor_set.write_anchor(
        scale="fast", segment_id="0.2",
        stream_mixture=("z_world", "z_harm_s"),
        z_world=z,
    )
    # Inject per-region staleness directly into the accumulator.
    mod.staleness_accumulator._staleness[(a.key[0], a.key[1])] = 0.3
    mod.staleness_accumulator._staleness[(b.key[0], b.key[1])] = 0.6

    out = mod.compute_per_stream_staleness()
    # z_world participates in BOTH anchors -> max(0.3, 0.6) = 0.6.
    assert out["z_world"] == pytest.approx(0.6)
    # z_harm_s only in anchor b -> 0.6.
    assert out["z_harm_s"] == pytest.approx(0.6)
    # z_self in NO anchor -> 0.0.
    assert out["z_self"] == 0.0


# ---------------------------------------------------------------------- #
# C7 -- diagnostics + reset
# ---------------------------------------------------------------------- #
def test_c7_diagnostics_and_reset():
    gate = _make_gate(use_staleness_lookup=True, e1_threshold=0.4)
    z_world = torch.ones(1, 8)
    latent = _make_latent(z_world=z_world)
    per_stream_vs = {"z_world": 0.9}
    gate.update_snapshots(latent, per_stream_vs)
    gate.gate(
        latent, per_stream_vs, side="e1",
        per_stream_staleness={"z_world": 0.55},
    )
    d = gate.get_diagnostics()
    assert d["vs_gate_use_staleness_lookup"] is True
    assert d["vs_gate_staleness_lookup_calls"] >= 1
    assert d["vs_gate_max_staleness_z_world"] == pytest.approx(0.55)
    gate.reset()
    d2 = gate.get_diagnostics()
    assert d2["vs_gate_staleness_lookup_calls"] == 0
    assert d2["vs_gate_max_staleness_z_world"] == 0.0


# ---------------------------------------------------------------------- #
# C8 -- agent precondition raises on missing dependencies
# ---------------------------------------------------------------------- #
def test_c8_agent_precondition_requires_staleness_and_anchor_substrates():
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent

    # Missing use_staleness_accumulator.
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=64, harm_obs_dim=8,
        action_dim=4, world_dim=16, self_dim=8, harm_dim=8,
        use_per_stream_vs=True,
        use_anchor_sets=True,
        use_staleness_accumulator=False,
        use_vs_rollout_gating=True,
        use_vs_gate_staleness_lookup=True,
    )
    with pytest.raises(ValueError, match="use_staleness_accumulator"):
        REEAgent(cfg)

    # Missing use_anchor_sets.
    cfg2 = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=64, harm_obs_dim=8,
        action_dim=4, world_dim=16, self_dim=8, harm_dim=8,
        use_per_stream_vs=True,
        use_anchor_sets=False,
        use_staleness_accumulator=True,
        use_vs_rollout_gating=True,
        use_vs_gate_staleness_lookup=True,
    )
    with pytest.raises(ValueError, match="use_anchor_sets"):
        REEAgent(cfg2)
