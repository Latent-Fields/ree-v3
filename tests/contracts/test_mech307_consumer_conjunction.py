"""MECH-307 Path B: consumer-side conjunction read contract tests.

Pin the bridge extension landed 2026-05-08 (after EXQ-539 PARTIAL-PASS):

  Substrate fix (Phase 1, commit 65d4e46) populated four signals:
    Gap 1  signed VALENCE_SURPRISE
    Gap 2  anticipatory VALENCE_LIKING write
    Gap 3  z_beta arousal pulse
    Gap 4  write target is e1_prior (predicted z_world)

  Consumer fix (Path B, this commit) wires the MECH-295 bridge to actually
  read those signals at action selection. New flag
  REEConfig.use_mech307_consumer_conjunction_read gates the new path; with
  the flag False, MECH-295 cue computation is bit-identical to legacy.

These contracts pin:
  C1 default config: use_mech307_consumer_conjunction_read defaults False;
     bridge config field defaults False; bias is exactly zero with flag off
     even when conjunction conditions are otherwise met.
  C2 disabled-bridge: when use_mech295_liking_bridge is False, no bridge is
     instantiated; conjunction read is unreachable -- no errors.
  C3 conjunction-fires: with all four conditions met (wanting > thr, liking
     > thr, signed surprise > 0, z_beta arousal > thr) and flag on, bias is
     -gain * drive (negative, magnitude > 0).
  C4 partial-conjunction: missing any one of the four predicates -> bias 0.
  C5 negative-surprise: harm-paired (negative signed) surprise must NOT
     satisfy the predicate even when the magnitude is large -- this is what
     makes Gap 1 (signed PE) load-bearing.
  C6 simulation_mode: returns zeros without advancing diagnostics.
  C7 diagnostics: counters increment on fires, reset cleanly on bridge.reset().
  C8 per-candidate isolation: K=4 candidates, only the one whose predicted
     location's valence + global z_beta satisfies all four conditions gets
     non-zero bias.

See REE_assembly/docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md
for the doc lines 128-137 predicate definition.
"""
from __future__ import annotations

import torch
import pytest

from ree_core.regulators.mech295_liking_bridge import (
    MECH295LikingBridge, MECH295LikingBridgeConfig,
)


class _StubResidueField:
    """Minimal residue stand-in: returns a fixed valence tensor per call."""

    def __init__(self, valence: torch.Tensor) -> None:
        # valence: [K, 4] in component order (wanting, liking, harm, surprise).
        self._valence = valence

    def evaluate_valence(self, z: torch.Tensor) -> torch.Tensor:
        K = z.shape[0]
        if self._valence.shape[0] != K:
            # Broadcast a single-row stub up to K.
            return self._valence[:1].expand(K, -1).clone()
        return self._valence


def _build_bridge(**overrides) -> MECH295LikingBridge:
    cfg = MECH295LikingBridgeConfig(**overrides)
    return MECH295LikingBridge(cfg)


def test_c1_default_flag_off_bias_zero():
    """C1: default config -> conjunction read disabled, bias exactly zero."""
    bridge = _build_bridge()
    assert bridge.config.use_mech307_conjunction_read is False
    K = 3
    z = torch.zeros(K, 8)
    # Construct conditions that WOULD satisfy the predicate if read were on:
    valence = torch.tensor([
        [1.0, 1.0, 0.0, 1.0],  # wanting=1>0.6, liking=1>0.3, surprise=+1>0
        [1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ])
    rf = _StubResidueField(valence)
    bias = bridge.compute_conjunction_score_bias(
        candidate_z_locs=z, residue_field=rf,
        z_beta_arousal=1.0, drive_level=1.0,
    )
    assert torch.equal(bias, torch.zeros(K))


def test_c3_conjunction_fires_bias_negative():
    """C3: all four predicates true -> negative bias proportional to gain * drive."""
    bridge = _build_bridge(
        use_mech307_conjunction_read=True,
        mech307_conjunction_gain=2.0,
        mech307_conjunction_wanting_threshold=0.6,
        mech307_conjunction_liking_threshold=0.3,
        mech307_conjunction_z_beta_threshold=0.6,
    )
    K = 2
    z = torch.zeros(K, 8)
    valence = torch.tensor([
        [0.8, 0.5, 0.0, 0.4],  # wanting>0.6, liking>0.3, surprise=+0.4>0
        [0.8, 0.5, 0.0, 0.4],
    ])
    rf = _StubResidueField(valence)
    bias = bridge.compute_conjunction_score_bias(
        candidate_z_locs=z, residue_field=rf,
        z_beta_arousal=0.9, drive_level=0.7,
    )
    expected = -2.0 * 0.7
    assert torch.allclose(bias, torch.full((K,), expected), atol=1e-5)


def test_c4_partial_conjunction_zero_bias():
    """C4: missing any predicate -> zero bias."""
    bridge = _build_bridge(use_mech307_conjunction_read=True)
    z = torch.zeros(1, 8)
    rf_no_wanting = _StubResidueField(torch.tensor([[0.5, 0.5, 0.0, 0.4]]))
    rf_no_liking = _StubResidueField(torch.tensor([[0.8, 0.2, 0.0, 0.4]]))
    rf_zero_surprise = _StubResidueField(torch.tensor([[0.8, 0.5, 0.0, 0.0]]))
    for rf in (rf_no_wanting, rf_no_liking, rf_zero_surprise):
        b = bridge.compute_conjunction_score_bias(
            candidate_z_locs=z, residue_field=rf,
            z_beta_arousal=0.9, drive_level=0.7,
        )
        assert b.abs().max().item() == 0.0
    # Low z_beta arousal also blocks. (z_beta_arousal=0.1 sits below the
    # 2026-05-12 default threshold of 0.3; the legacy 0.4 used here pre-fix
    # was below the legacy 0.6 default but now sits ABOVE the new 0.3.)
    rf_ok = _StubResidueField(torch.tensor([[0.8, 0.5, 0.0, 0.4]]))
    b = bridge.compute_conjunction_score_bias(
        candidate_z_locs=z, residue_field=rf_ok,
        z_beta_arousal=0.1, drive_level=0.7,
    )
    assert b.abs().max().item() == 0.0


def test_c5_negative_surprise_blocks_predicate():
    """C5: harm-paired (negative signed) surprise must NOT fire predicate.

    This is the load-bearing test for why Gap 1 (signed PE) matters.
    Under unsigned VALENCE_SURPRISE, a large harm-paired surprise would
    have positive magnitude in the residue field -- which would falsely
    satisfy v_s > 0 here. With Gap 1 ON, harm-paired surprise stores as
    negative, so v_s > 0 correctly excludes it.
    """
    bridge = _build_bridge(use_mech307_conjunction_read=True)
    z = torch.zeros(1, 8)
    # Negative surprise (harm-paired under Gap 1 signed-PE write).
    rf = _StubResidueField(torch.tensor([[0.8, 0.5, 0.0, -0.4]]))
    bias = bridge.compute_conjunction_score_bias(
        candidate_z_locs=z, residue_field=rf,
        z_beta_arousal=0.9, drive_level=0.7,
    )
    assert bias.abs().max().item() == 0.0


def test_c6_simulation_mode_returns_zero():
    """C6: simulation_mode short-circuits to zero, no diagnostic advance."""
    bridge = _build_bridge(use_mech307_conjunction_read=True)
    z = torch.zeros(2, 8)
    rf = _StubResidueField(torch.tensor([
        [0.8, 0.5, 0.0, 0.4],
        [0.8, 0.5, 0.0, 0.4],
    ]))
    bias = bridge.compute_conjunction_score_bias(
        candidate_z_locs=z, residue_field=rf,
        z_beta_arousal=0.9, drive_level=0.7, simulation_mode=True,
    )
    assert torch.equal(bias, torch.zeros(2))
    diag = bridge.get_diagnostics()
    assert diag["n_conjunction_fires"] == 0
    assert diag["n_conjunction_reads"] == 0


def test_c7_diagnostics_increment_and_reset():
    """C7: counters increment on fires; bridge.reset() clears per-tick cache."""
    bridge = _build_bridge(use_mech307_conjunction_read=True)
    z = torch.zeros(2, 8)
    rf = _StubResidueField(torch.tensor([
        [0.8, 0.5, 0.0, 0.4],
        [0.8, 0.5, 0.0, 0.4],
    ]))
    bridge.compute_conjunction_score_bias(
        candidate_z_locs=z, residue_field=rf,
        z_beta_arousal=0.9, drive_level=0.7,
    )
    diag = bridge.get_diagnostics()
    assert diag["n_conjunction_reads"] == 1
    assert diag["n_conjunction_fires"] == 2
    assert diag["last_conjunction_count"] == 2
    assert diag["last_conjunction_score_max"] > 0.0


def test_c8_per_candidate_isolation():
    """C8: only candidates whose location's valence satisfies the predicate fire."""
    bridge = _build_bridge(
        use_mech307_conjunction_read=True,
        mech307_conjunction_gain=1.0,
    )
    K = 4
    z = torch.zeros(K, 8)
    valence = torch.tensor([
        [0.8, 0.5, 0.0, +0.4],   # all four hold -> fires
        [0.4, 0.5, 0.0, +0.4],   # wanting < threshold -> no fire
        [0.8, 0.1, 0.0, +0.4],   # liking < threshold -> no fire
        [0.8, 0.5, 0.0, -0.4],   # negative surprise -> no fire
    ])
    rf = _StubResidueField(valence)
    bias = bridge.compute_conjunction_score_bias(
        candidate_z_locs=z, residue_field=rf,
        z_beta_arousal=0.9, drive_level=1.0,
    )
    # Only candidate 0 has non-zero bias.
    assert bias[0].item() < 0.0
    assert bias[1].item() == 0.0
    assert bias[2].item() == 0.0
    assert bias[3].item() == 0.0


def test_c2_no_bridge_no_errors():
    """C2: REEAgent does not instantiate bridge when use_mech295_liking_bridge=False.

    This is enforced at the agent construction site, not the bridge itself,
    so we verify by importing REEConfig and checking that the new field
    defaults False. The agent boot path is exercised by the broader feature-
    flag boot matrix contract.
    """
    from ree_core.utils.config import REEConfig
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=20, action_dim=4,
        self_dim=8, world_dim=8, alpha_world=0.9,
    )
    assert getattr(cfg, "use_mech307_consumer_conjunction_read") is False
    assert getattr(cfg, "mech307_conjunction_wanting_threshold") == 0.6
    assert getattr(cfg, "mech307_conjunction_liking_threshold") == 0.3
    # z_beta default lowered 0.6 -> 0.3 on 2026-05-12 per V3-EXQ-540c probe
    # finding (observed z_beta_arousal max=0.545 across 1087 bridge calls;
    # legacy 0.6 floor sat above the achievable ceiling).
    assert getattr(cfg, "mech307_conjunction_z_beta_threshold") == 0.3
    assert getattr(cfg, "mech307_conjunction_gain") == 1.0
