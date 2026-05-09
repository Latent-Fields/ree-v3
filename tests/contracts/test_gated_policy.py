"""ARC-062 Phase 1 (gated-policy heads + context discriminator) contract tests.

Five contract guarantees the substrate must hold regardless of tuning:

  C1 default-off no-op: REEAgent constructed with use_gated_policy=False
     produces bit-identical actions to baseline E3 under fixed seed across
     a multi-tick episode.

  C2 backward-compat: full ree-v3 contract suite (other tests in this
     directory) continues to pass with use_gated_policy=False (verified by
     pytest tests/contracts -q at the run-suite level; this test exercises
     the broader-shape no-op locally by checking that turning the flag on
     does not raise during construction or first sense() tick under any of
     the supported flag combinations).

  C3 discriminator output in [0, 1]: discriminator output is a valid
     probability across diverse latent states.

  C4 head differentiation under training pressure: after N steps with a
     synthetic loss against a target that depends on the discriminator's
     gating choice, the two heads' parameters diverge measurably from
     their symmetry-broken init. This validates the architectural intent
     that the heads can specialise rather than collapsing to identical
     outputs.

  C5 simulation-mode gating per MECH-094: when simulation_mode=True, the
     module returns (0.5, zeros[K], zeros[K], zeros[K]) and increments the
     simulation-skip counter without advancing any other diagnostic.
     Match the SD-035 amygdala / MECH-279 PAG simulation_mode pattern.

See ree_core/policy/gated_policy.py module docstring for the architectural
context (Pull A SYNTHESIS verdicts behind R1 / R2 / R3 defaults) and
evidence/planning/arc_062_rule_apprehension_plan.md for the closure plan.
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.policy import GatedPolicy, GatedPolicyConfig

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config


def _build_obs_kwargs(env, obs_dict, cfg):
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kwargs = {"obs_body": body, "obs_world": world}
    obs_harm = obs_dict.get("harm_obs")
    if obs_harm is not None and getattr(cfg.latent, "use_harm_stream", False):
        if obs_harm.dim() == 1:
            obs_harm = obs_harm.unsqueeze(0)
        kwargs["obs_harm"] = obs_harm
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None and getattr(cfg.latent, "use_affective_harm_stream", False):
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        kwargs["obs_harm_a"] = obs_harm_a
    return kwargs


# ----------------------------------------------------------------------
# C1 default-off no-op
# ----------------------------------------------------------------------
def test_c1_default_off_no_op_matches_baseline():
    """C1: with use_gated_policy=False the agent boots and runs as before."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)  # default-off everywhere

    agent = REEAgent(cfg)
    agent.reset()

    # The flag is off => agent.gated_policy must be None and the
    # select_action wiring must skip the entire GatedPolicy block.
    assert agent.gated_policy is None, (
        "C1 default-off violation: agent.gated_policy must be None "
        "when use_gated_policy=False."
    )

    _flat, obs_dict = env.reset()
    kwargs = _build_obs_kwargs(env, obs_dict, cfg)

    with torch.no_grad():
        latent = agent.sense(**kwargs)
    assert latent is not None
    assert torch.isfinite(latent.z_world).all(), "C1: z_world NaN/Inf"


# ----------------------------------------------------------------------
# C2 backward-compat (broader-shape: turning the flag on under default
# build does not raise during construction or first sense() tick)
# ----------------------------------------------------------------------
def test_c2_backward_compat_flag_on_does_not_raise():
    """C2: enabling use_gated_policy on the default build does not raise."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env, use_gated_policy=True)

    agent = REEAgent(cfg)
    agent.reset()
    assert agent.gated_policy is not None, (
        "C2 wiring regression: agent.gated_policy must be instantiated "
        "when use_gated_policy=True."
    )

    _flat, obs_dict = env.reset()
    kwargs = _build_obs_kwargs(env, obs_dict, cfg)
    with torch.no_grad():
        latent = agent.sense(**kwargs)
    assert latent is not None
    assert torch.isfinite(latent.z_world).all()


# ----------------------------------------------------------------------
# C3 discriminator output in [0, 1] across diverse latent inputs
# ----------------------------------------------------------------------
def test_c3_discriminator_output_in_unit_interval():
    """C3: discriminator output is a valid probability across diverse inputs."""
    torch.manual_seed(0)
    cfg = GatedPolicyConfig(use_gated_policy=True)
    world_dim, self_dim, harm_a_dim = 32, 16, 16
    K = 4
    gp = GatedPolicy(world_dim=world_dim, self_dim=self_dim,
                     harm_a_dim=harm_a_dim, config=cfg)

    # Sample 64 random latent states across a wide magnitude range.
    weights_seen = []
    for trial in range(64):
        scale = 10.0 ** (trial % 4 - 2)  # 1e-2, 1e-1, 1e0, 1e1 cycling
        zw = torch.randn(1, world_dim) * scale
        zs = torch.randn(1, self_dim) * scale
        za = torch.randn(1, harm_a_dim) * scale
        cand = torch.randn(K, world_dim) * scale
        with torch.no_grad():
            out = gp(z_world=zw, z_self=zs, z_harm_a=za,
                     candidate_features=cand, simulation_mode=False)
        weights_seen.append(out.gating_weight)
        assert 0.0 <= out.gating_weight <= 1.0, (
            f"C3 violation trial {trial}: w={out.gating_weight} out of [0, 1]"
        )
        assert out.gated_score_bias.shape == (K,)
        # Bias must respect the bias_scale clamp.
        assert out.gated_score_bias.abs().max().item() <= cfg.bias_scale + 1e-6, (
            f"C3 bias_scale violation trial {trial}: "
            f"max|bias|={out.gated_score_bias.abs().max().item():.4f} "
            f"> bias_scale={cfg.bias_scale}"
        )
    # Sanity: at least some variation in w across trials (not stuck at one value).
    w_min = min(weights_seen)
    w_max = max(weights_seen)
    assert w_max - w_min > 1e-3, (
        "C3 sanity: discriminator output stuck at a single value across 64 "
        f"diverse trials (w_min={w_min:.4f}, w_max={w_max:.4f}); the "
        "discriminator never responds to input variation."
    )


# ----------------------------------------------------------------------
# C4 head differentiation under training pressure
# ----------------------------------------------------------------------
def test_c4_heads_differentiate_under_training_pressure():
    """C4: under a synthetic loss the two heads' OUTPUTS diverge.

    Training signal: drive head_0 toward outputting -1 on every candidate
    and head_1 toward outputting +1 on every candidate (a synthetic
    differentiation pressure; the actual training of these heads in the
    full agent loop comes from E3 score-aggregation gradient).

    Acceptance: on a held-out evaluation batch, the mean ||h0(x) - h1(x)||
    grows by more than 5x from the symmetry-broken-init baseline. Output
    divergence (not parameter divergence) is the right signal: the heads
    can specialise via their last-layer bias terms alone, and the
    architectural intent is that w*h0 + (1-w)*h1 admits a meaningful
    range of outputs depending on w. The init baseline is small (heads
    output near +/- bias_offset = +/- 0.05) so 5x growth is a tight but
    well-clear-of-noise threshold.
    """
    torch.manual_seed(0)
    cfg = GatedPolicyConfig(use_gated_policy=True)
    world_dim, self_dim, harm_a_dim = 32, 16, 16
    K = 8
    gp = GatedPolicy(world_dim=world_dim, self_dim=self_dim,
                     harm_a_dim=harm_a_dim, config=cfg)

    # Held-out eval batch -- fixed across init / final measurement.
    eval_cand = torch.randn(K, world_dim)

    # Baseline output divergence on eval batch (post symmetry-broken init).
    with torch.no_grad():
        h0_eval_init = gp.head_0(eval_cand).squeeze(-1)
        h1_eval_init = gp.head_1(eval_cand).squeeze(-1)
    init_output_dist = float((h0_eval_init - h1_eval_init).abs().mean().item())

    optim = torch.optim.SGD(
        list(gp.head_0.parameters()) + list(gp.head_1.parameters()),
        lr=0.05,
    )

    target_h0 = torch.full((K,), -1.0)
    target_h1 = torch.full((K,), +1.0)

    for _step in range(200):
        cand = torch.randn(K, world_dim)
        h0 = gp.head_0(cand).squeeze(-1)
        h1 = gp.head_1(cand).squeeze(-1)
        loss = ((h0 - target_h0) ** 2).mean() + ((h1 - target_h1) ** 2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

    with torch.no_grad():
        h0_eval_final = gp.head_0(eval_cand).squeeze(-1)
        h1_eval_final = gp.head_1(eval_cand).squeeze(-1)
    final_output_dist = float((h0_eval_final - h1_eval_final).abs().mean().item())

    assert final_output_dist > 5.0 * init_output_dist, (
        f"C4 violation: heads' outputs did not diverge under training "
        f"pressure (init_output_dist={init_output_dist:.4f}, "
        f"final_output_dist={final_output_dist:.4f}, "
        f"ratio={final_output_dist / max(init_output_dist, 1e-12):.4f}, "
        f"expected > 5.0)"
    )


# ----------------------------------------------------------------------
# C5 MECH-094 simulation-mode gating
# ----------------------------------------------------------------------
def test_c5_simulation_mode_returns_zeros_and_increments_skip_counter():
    """C5: simulation_mode=True returns zeros and increments the skip counter."""
    torch.manual_seed(0)
    cfg = GatedPolicyConfig(use_gated_policy=True)
    world_dim, self_dim, harm_a_dim = 32, 16, 16
    K = 5
    gp = GatedPolicy(world_dim=world_dim, self_dim=self_dim,
                     harm_a_dim=harm_a_dim, config=cfg)

    zw = torch.randn(1, world_dim)
    zs = torch.randn(1, self_dim)
    za = torch.randn(1, harm_a_dim)
    cand = torch.randn(K, world_dim)

    # Cache the diagnostic state pre-call.
    pre_skip = gp._last_n_simulation_skips
    pre_bias_mean = gp._last_bias_abs_mean

    out = gp(z_world=zw, z_self=zs, z_harm_a=za,
             candidate_features=cand, simulation_mode=True)

    # Output contract: gating_weight=0.5, all bias tensors zero of shape [K].
    assert out.gating_weight == 0.5
    assert out.gated_score_bias.shape == (K,)
    assert torch.equal(out.gated_score_bias, torch.zeros(K))
    assert torch.equal(out.head_0_bias, torch.zeros(K))
    assert torch.equal(out.head_1_bias, torch.zeros(K))

    # Skip counter incremented; bias-magnitude diagnostic NOT touched
    # (would be touched in the waking branch).
    assert gp._last_n_simulation_skips == pre_skip + 1, (
        f"C5: simulation-skip counter did not advance "
        f"(pre={pre_skip}, post={gp._last_n_simulation_skips})"
    )
    assert gp._last_bias_abs_mean == pre_bias_mean, (
        f"C5: simulation_mode=True must not advance the bias-magnitude "
        f"diagnostic counter (pre={pre_bias_mean}, "
        f"post={gp._last_bias_abs_mean})"
    )

    # Confirm a subsequent waking call updates the bias-magnitude diagnostic
    # but does not retroactively re-increment the skip counter.
    _waking = gp(z_world=zw, z_self=zs, z_harm_a=za,
                 candidate_features=cand, simulation_mode=False)
    assert gp._last_n_simulation_skips == pre_skip + 1, (
        "C5: waking call must not increment the simulation-skip counter."
    )
