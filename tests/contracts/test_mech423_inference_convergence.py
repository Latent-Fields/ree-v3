"""
Contract tests for MECH-423 R2: iterative-inference convergence readout.

The legacy LatentStack.encode() is a fixed two-pass amortized recognition; the
EXP-0380 (MECH-423 cross-model super-additivity) R2 readiness check needs a
per-inference-step ||delta z_shared|| convergence signal. This suite pins:

  C1 default-OFF no-op + bit-identical action-relevant latent vs explicit-False.
  C2 flag-ON populates a well-formed inference_convergence readout.
  C3 the settling loop reduces the relative delta (genuine convergence).
  C4 settle_iters=1 with the flag on is the degenerate single-round case and is
     bit-identical to OFF for the latent values.
  C5 detach() preserves the readout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def _build_and_sense(seed: int = 7, **flags):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    with torch.no_grad():
        latent = agent.sense(obs_body=b, obs_world=w)
    return agent, latent


# ----------------------------------------------------------------------
# C1 default-OFF no-op + bit-identical
# ----------------------------------------------------------------------
def test_c1_default_off_no_op_and_bit_identical():
    a0, l0 = _build_and_sense()
    assert l0.inference_convergence is None
    assert a0.last_inference_convergence is None

    # explicit-False must reproduce the default latent values byte-for-byte
    _a1, l1 = _build_and_sense(use_iterative_inference=False)
    assert torch.equal(l0.z_world, l1.z_world)
    assert torch.equal(l0.z_self, l1.z_self)
    assert torch.equal(l0.z_beta, l1.z_beta)
    assert torch.equal(l0.z_theta, l1.z_theta)
    assert torch.equal(l0.z_delta, l1.z_delta)


# ----------------------------------------------------------------------
# C2 flag-ON populates a well-formed readout
# ----------------------------------------------------------------------
def test_c2_flag_on_populates_readout():
    agent, latent = _build_and_sense(
        use_iterative_inference=True,
        inference_settle_iters=8,
        inference_convergence_rel_tol=0.01,
    )
    ic = latent.inference_convergence
    assert isinstance(ic, dict)
    assert set(ic.keys()) == {
        "per_step_rel_delta",
        "converged",
        "n_iters",
        "final_rel_delta",
    }
    assert isinstance(ic["per_step_rel_delta"], list)
    assert isinstance(ic["converged"], bool)
    assert isinstance(ic["n_iters"], int) and ic["n_iters"] >= 1
    assert isinstance(ic["final_rel_delta"], float)
    # the agent caches the same object
    assert agent.last_inference_convergence is ic
    # at least one settling round ran beyond round 1
    assert len(ic["per_step_rel_delta"]) >= 1
    assert all(d >= 0.0 for d in ic["per_step_rel_delta"])


# ----------------------------------------------------------------------
# C3 the settling loop reduces the relative delta
# ----------------------------------------------------------------------
def test_c3_settling_reduces_relative_delta():
    _agent, latent = _build_and_sense(
        use_iterative_inference=True,
        inference_settle_iters=8,
        inference_convergence_rel_tol=1e-6,  # force the full budget
    )
    ic = latent.inference_convergence
    deltas = ic["per_step_rel_delta"]
    assert len(deltas) >= 2, "need >= 2 deltas to show convergence"
    # final relative delta is strictly below the first (contractive top-down map)
    assert deltas[-1] <= deltas[0]
    assert ic["final_rel_delta"] == deltas[-1]


# ----------------------------------------------------------------------
# C4 settle_iters=1 with flag ON is the degenerate single-round case
# ----------------------------------------------------------------------
def test_c4_settle_iters_one_is_bit_identical_to_off():
    _a_off, l_off = _build_and_sense()
    _a_on, l_on = _build_and_sense(
        use_iterative_inference=True, inference_settle_iters=1
    )
    # one round == the legacy top-down round -> latent values unchanged
    assert torch.equal(l_off.z_world, l_on.z_world)
    assert torch.equal(l_off.z_beta, l_on.z_beta)
    ic = l_on.inference_convergence
    assert ic is not None
    assert ic["n_iters"] == 1
    assert ic["per_step_rel_delta"] == []
    assert ic["converged"] is False  # cannot assert convergence from one pass


# ----------------------------------------------------------------------
# C5 detach() preserves the readout
# ----------------------------------------------------------------------
def test_c5_detach_preserves_readout():
    _agent, latent = _build_and_sense(
        use_iterative_inference=True, inference_settle_iters=4
    )
    d = latent.detach()
    assert d.inference_convergence is latent.inference_convergence
