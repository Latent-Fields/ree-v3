"""Contract tests for the MECH-203 harm-salience write path (2026-08-02).

Why this file exists
---------------------
The SR-2 design doc (REE_assembly/docs/architecture/sleep/
serotonergic_cross_state_substrate.md) describes benefit_salience tagging
as "complementary to the existing harm salience from the residue field" --
but no calibrated, tonic-5HT-modulated harm-salience WRITE into
VALENCE_HARM_DISCRIMINATIVE actually existed:

  - update_residue() -> ResidueField.accumulate() only ever wrote the legacy
    scalar `weights` (residue density), never valence_vecs.
  - The separate SD-014 `valence_harm_enabled` path (agent.py sense()) writes
    raw post-attenuation z_harm.norm() every step -- ungated by tonic_5ht,
    not symmetric with benefit_salience's calibration.

A naive direct fix (writing raw |harm_signal| into VALENCE_HARM_DISCRIMINATIVE
via the same update_valence() call benefit_salience uses) was tried and
rejected: because ResidueField.update_valence() always targets the NEAREST
ACTIVE center in the shared rbf_field (the same centers benefit_salience's
WANTING write lands on), and raw |harm_signal| (~0.05-0.13) is a ~13-20x
larger scale than benefit_salience's own tonic-modulated output
(~0.0037-0.0066), the two channels do not coexist -- harm swamps benefit to
a 100%/0% degenerate split.

SerotoninModule.harm_salience() (serotonin.py) fixes this by mirroring
benefit_salience's structure -- harm_salience = (1 - tonic_5ht) * harm_exposure
-- and, critically, requiring the SAME EMA'd-exposure convention
benefit_exposure already uses (CausalGridWorldV2.harm_exposure / body_obs[10],
symmetric to benefit_exposure / body_obs[11]), not raw harm_signal.
agent.update_harm_salience() writes it into VALENCE_HARM_DISCRIMINATIVE via
the same ResidueField.update_valence() path update_benefit_salience() uses.

Contracts:
  C1  Surface -- SerotoninModule.harm_salience() and
      REEAgent.update_harm_salience() exist; harm_salience() is 0.0 when
      disabled and follows (1 - tonic_5ht) * harm_exposure when enabled.
  C2  Live-path write -- through the real REEAgent API (agent.sense() +
      update_residue() to allocate an active center, then
      update_harm_salience()), VALENCE_HARM_DISCRIMINATIVE at the visited
      z_world becomes non-trivial (> 0).
  C3  SWAMPING REGRESSION -- reproduces the rejected naive fix (writing raw
      |harm_signal| via update_valence directly) and shows it degenerates
      the shared field to ~100% harm / ~0% benefit; then shows
      update_harm_salience() + update_benefit_salience() together, at
      realistic EMA-scale exposures, leave BOTH VALENCE_WANTING and
      VALENCE_HARM_DISCRIMINATIVE non-trivial (neither swamped to ~0).
  C4  Bit-identical OFF -- tonic_5ht_enabled=False (default): harm_salience()
      returns 0.0 regardless of input, and update_harm_salience() writes
      nothing (valence vector at the visited location is unchanged / stays
      zero on that component) -- matching update_benefit_salience()'s
      existing no-op-default-OFF behaviour exactly.
  C5  No new config flag -- update_harm_salience() is gated by the SAME
      tonic_5ht_enabled master switch update_benefit_salience() uses (no new
      REEConfig field introduced).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.neuromodulation.serotonin import SerotoninConfig, SerotoninModule
from ree_core.residue.field import (
    VALENCE_HARM_DISCRIMINATIVE,
    VALENCE_WANTING,
    ResidueField,
)
from ree_core.utils.config import REEConfig, ResidueConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent(seed: int = 7, tonic_5ht_enabled: bool = True):
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        tonic_5ht_enabled=tonic_5ht_enabled,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _sense(agent, seed: int = 7):
    torch.manual_seed(seed)
    body = torch.randn(1, 12)
    world = torch.randn(1, 250)
    agent.sense(body, world)
    return agent._current_latent.z_world


# ----------------------------------------------------------------------
# C1 -- surface + arithmetic
# ----------------------------------------------------------------------
def test_c1_module_surface_and_arithmetic():
    sero_off = SerotoninModule(SerotoninConfig(tonic_5ht_enabled=False))
    assert sero_off.harm_salience(0.5) == 0.0

    sero = SerotoninModule(SerotoninConfig(tonic_5ht_enabled=True, tonic_5ht_baseline=0.4))
    assert hasattr(sero, "harm_salience")
    # tonic_5ht starts at baseline 0.4 -> harm_salience = (1 - 0.4) * exposure
    val = sero.harm_salience(0.1)
    assert abs(val - (0.6 * 0.1)) < 1e-9

    # Negative exposure clamps to 0 contribution (mirrors benefit_salience).
    assert sero.harm_salience(-1.0) == 0.0

    assert hasattr(REEAgent, "update_harm_salience")


# ----------------------------------------------------------------------
# C2 -- live-path write through the real agent API
# ----------------------------------------------------------------------
def test_c2_live_path_populates_harm_discriminative():
    agent = _build_agent()
    z = _sense(agent)

    # Allocate an active RBF center (mirrors how the real step loop drives
    # update_residue() on harm contact) so update_valence() has a center to
    # find. Small magnitude -- this is the density write, not the valence tag.
    agent.update_residue(-0.05)

    before = agent.residue_field.evaluate_valence(z)[0, VALENCE_HARM_DISCRIMINATIVE].item()
    assert before == 0.0  # accumulate() alone never touches valence_vecs

    agent.update_harm_salience(harm_exposure=0.05)  # EMA-scale convention

    after = agent.residue_field.evaluate_valence(z)[0, VALENCE_HARM_DISCRIMINATIVE].item()
    assert after > 0.0, "update_harm_salience() must write a non-trivial VALENCE_HARM_DISCRIMINATIVE"


# ----------------------------------------------------------------------
# C3 -- THE SWAMPING REGRESSION
# ----------------------------------------------------------------------
def _fresh_field(num_centers=32):
    return ResidueField(ResidueConfig(world_dim=8, num_basis_functions=num_centers))


def test_c3_naive_raw_harm_signal_swamps_benefit():
    """Reproduce the REJECTED naive fix: writing raw |harm_signal| straight
    into VALENCE_HARM_DISCRIMINATIVE via update_valence(), at the same shared
    RBF capacity benefit_salience's WANTING write uses. This is the failure
    mode harm_salience()'s EMA-scale calibration exists to avoid.
    """
    field = _fresh_field()
    torch.manual_seed(0)

    RAW_HARM_SIGNAL = 0.10   # env-scale, per the investigation (~0.05-0.13)
    BENEFIT_SALIENCE = 0.005  # realistic tonic_5ht * benefit_exposure scale

    # Interleave many benefit and harm writes at nearby locations sharing the
    # same 32-center capacity -- exactly the shared-field mechanism in play.
    for i in range(40):
        loc = torch.randn(8) * 0.1
        field.accumulate(loc.unsqueeze(0), harm_magnitude=0.01)  # allocates/refreshes a center
        field.update_valence(loc, VALENCE_WANTING, BENEFIT_SALIENCE)
        field.update_valence(loc, VALENCE_HARM_DISCRIMINATIVE, RAW_HARM_SIGNAL)

    total_wanting = field.rbf_field.valence_vecs[:, VALENCE_WANTING].abs().sum().item()
    total_harm = field.rbf_field.valence_vecs[:, VALENCE_HARM_DISCRIMINATIVE].abs().sum().item()
    harm_fraction = total_harm / (total_harm + total_wanting)

    # This IS the degenerate split the naive fix produced -- pinned here as a
    # regression fixture, not a desirable property.
    assert harm_fraction > 0.9, (
        f"expected the naive raw-harm-signal write to swamp benefit "
        f"(harm_fraction={harm_fraction:.4f}); if this now fails, the shared-"
        f"capacity swamping mechanism itself has changed and the calibrated "
        f"fix's rationale should be re-examined"
    )


def test_c3_calibrated_harm_salience_does_not_swamp_benefit():
    """The FIX: both harm_salience() and benefit_salience(), fed their
    respective EMA-scale exposure conventions, coexist in the shared field --
    neither channel is swamped to ~0.
    """
    agent = _build_agent()
    torch.manual_seed(1)

    # EMA-scale exposures (both clipped to [0,1], same nociception_ema_alpha
    # convention in CausalGridWorldV2) -- NOT raw harm_signal.
    BENEFIT_EXPOSURE = 0.006
    HARM_EXPOSURE = 0.006

    for i in range(40):
        body = torch.randn(1, 12)
        world = torch.randn(1, 250)
        agent.sense(body, world)
        agent.update_residue(-0.01)  # allocate/refresh an active center
        agent.update_benefit_salience(BENEFIT_EXPOSURE)
        agent.update_harm_salience(HARM_EXPOSURE)

    rf = agent.residue_field
    total_wanting = rf.rbf_field.valence_vecs[:, VALENCE_WANTING].abs().sum().item()
    total_harm = rf.rbf_field.valence_vecs[:, VALENCE_HARM_DISCRIMINATIVE].abs().sum().item()

    assert total_wanting > 0.0, "benefit channel must remain non-trivial"
    assert total_harm > 0.0, "harm channel must remain non-trivial"

    harm_fraction = total_harm / (total_harm + total_wanting)
    assert 0.1 < harm_fraction < 0.9, (
        f"harm_fraction={harm_fraction:.4f} -- calibrated write should not "
        f"degenerate to either extreme the way the naive raw-signal write did"
    )


# ----------------------------------------------------------------------
# C4 -- bit-identical OFF
# ----------------------------------------------------------------------
def test_c4_default_off_is_bit_identical():
    agent = _build_agent(tonic_5ht_enabled=False)
    z = _sense(agent)
    agent.update_residue(-0.05)

    before = agent.residue_field.evaluate_valence(z)[0, VALENCE_HARM_DISCRIMINATIVE].item()
    agent.update_harm_salience(harm_exposure=0.05)
    after = agent.residue_field.evaluate_valence(z)[0, VALENCE_HARM_DISCRIMINATIVE].item()

    assert before == 0.0
    assert after == 0.0
    assert after == before

    # Default REEConfig() is tonic_5ht_enabled=False.
    assert REEConfig().serotonin.tonic_5ht_enabled is False


# ----------------------------------------------------------------------
# C5 -- reuses the existing master switch, no new config flag
# ----------------------------------------------------------------------
def test_c5_reuses_tonic_5ht_enabled_no_new_flag():
    cfg = REEConfig()
    fields_before = set(vars(cfg.serotonin).keys())
    # harm_salience must be governed by the identical flag benefit_salience
    # uses -- SerotoninConfig gains no new master-switch field.
    assert "tonic_5ht_enabled" in fields_before
    assert not any(
        "harm_salience" in f or "harm_5ht" in f for f in fields_before
    ), "harm_salience should reuse tonic_5ht_enabled, not introduce a new flag"
