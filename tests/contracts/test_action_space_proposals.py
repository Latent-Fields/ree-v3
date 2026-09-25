"""Contracts for W1-alt ASP: action-space proposals (coupled-loop-repair campaign).

Design of record: REE_assembly evidence/planning/action_space_proposals_design_
20260925.md (412882b845), sec 2.1 (ASP-E), 2.4 (ASP-R / ASP-0), 3.2 (knobs),
3.3 (mutual exclusions), 4 (member gate G-ASP). Build lands on
integration/coupled-loop-repair, default-OFF.

What is pinned here:
  * the four knobs exist, default OFF/inert, and survive REEConfig.from_dims
    (the three-site rule: a knob missing from one site is silently swallowed);
  * OFF bit-identity: master OFF with every sub-knob at a NON-default value
    gives byte-identical pools, world states and RNG state to the default;
  * ON changes the pool (the flag-inertness probe for use_action_space_proposals);
  * G-ASP (a): no ASP parameter group exists; terrain_prior and
    action_object_decoder are never run with grad enabled on the ON path and
    receive NO gradient from anything propose returns;
  * G-ASP (b): every candidate action is an exact one-hot at every step
    (no zero vector, no fractional entry), decoder calls 0, max norm 1.0;
  * G-ASP (c): the in-run rollout-norm readout is computed correctly. The gate
    VERDICT needs the W3-trained world head (design sec 4 "read with the W3
    head"); on an untrained E2 both the ASP and the codec pools grow ~x1.15/step,
    so the growth is the head's, not the proposer's. Not asserted here;
  * G-ASP (d): stratified counts are exactly floor/ceil(K/A) with every class
    present in every pool; refit categoricals never drop below the floor. The
    refit-mode COVERAGE bar (>= 0.95 of pools hold every class) FAILS at the
    DRAFT floor 0.02 -- pinned as a strict xfail so a fix is forced to update it;
  * incompatible O-space options and invalid sub-knob values raise at construction.

Test half: every test here FAILS on the pre-build tree (the knobs do not exist:
HippocampalConfig raises TypeError on the unknown field).
"""

from __future__ import annotations

import math

import pytest
import torch

A = 5    # A1 env action_dim
K = 32   # deployed num_candidates
H = 10   # deployed horizon
I = 3    # deployed num_cem_iterations
WD = 32  # deployed world_dim


def _make_module(seed: int = 0, action_dim: int = A, num_candidates: int = K, **kw):
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2Config, E2FastPredictor
    from ree_core.residue.field import ResidueConfig, ResidueField
    from ree_core.utils.config import HippocampalConfig

    torch.manual_seed(seed)
    cfg = HippocampalConfig(
        world_dim=WD,
        action_dim=action_dim,
        action_object_dim=16,
        hidden_dim=32,
        horizon=H,
        num_candidates=num_candidates,
        num_cem_iterations=I,
        **kw,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=16,
        world_dim=WD,
        action_dim=action_dim,
        action_object_dim=16,
        hidden_dim=32,
    ))
    residue = ResidueField(ResidueConfig(
        world_dim=WD, hidden_dim=32, num_basis_functions=16,
    ))
    return HippocampalModule(cfg, e2=e2, residue_field=residue)


def _states(n: int, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    return [
        (torch.randn(1, WD, generator=g), torch.randn(1, 16, generator=g))
        for _ in range(n)
    ]


def _pool_bytes(pool):
    out = []
    for t in pool:
        out.append(t.actions.detach().numpy().tobytes())
        out.append(t.get_world_state_sequence().detach().numpy().tobytes())
    return out


def _stratified_counts(k: int, a: int):
    base, extra = divmod(k, a)
    return {c: base + (1 if c < extra else 0) for c in range(a)}


# --------------------------------------------------------------------------- #
# Knobs: defaults and the three config sites                                   #
# --------------------------------------------------------------------------- #


def test_knobs_default_off_and_survive_from_dims():
    from ree_core.utils.config import HippocampalConfig, REEConfig

    cfg = HippocampalConfig()
    assert cfg.use_action_space_proposals is False
    assert cfg.action_space_first_action_mode == "stratified"
    assert cfg.action_space_prob_floor == 0.02
    assert cfg.action_space_cem_score_horizon is None

    default = REEConfig.from_dims(
        body_obs_dim=4, world_obs_dim=8, action_dim=A, self_dim=8, world_dim=8,
        alpha_world=0.3,
    )
    assert default.hippocampal.use_action_space_proposals is False
    assert default.hippocampal.action_space_first_action_mode == "stratified"
    assert default.hippocampal.action_space_prob_floor == 0.02
    assert default.hippocampal.action_space_cem_score_horizon is None

    master = REEConfig.from_dims(
        body_obs_dim=4, world_obs_dim=8, action_dim=A, self_dim=8, world_dim=8,
        alpha_world=0.3,
        use_action_space_proposals=True,
        action_space_first_action_mode="refit",
        action_space_prob_floor=0.05,
        action_space_cem_score_horizon=3,
    )
    assert master.hippocampal.use_action_space_proposals is True
    assert master.hippocampal.action_space_first_action_mode == "refit"
    assert master.hippocampal.action_space_prob_floor == 0.05
    assert master.hippocampal.action_space_cem_score_horizon == 3


# --------------------------------------------------------------------------- #
# OFF bit-identity and ON non-inertness                                        #
# --------------------------------------------------------------------------- #


def _run_pools(module, states, seed=7):
    torch.manual_seed(seed)
    pools = [module.propose_trajectories(zw, z_self=zs) for zw, zs in states]
    return pools, torch.get_rng_state().clone()


def test_off_is_bit_identical_with_every_sub_knob_non_default():
    """Master OFF must make the three sub-knobs unreadable: same pools, same
    world states, same RNG stream as the untouched default -- and the codec
    path must still run (the counters see decoder / terrain_prior calls, which
    is what gives the ON-path zeros below their teeth)."""
    states = _states(4)
    ref = _make_module(seed=0)
    alt = _make_module(
        seed=0,
        use_action_space_proposals=False,
        action_space_first_action_mode="not-a-mode",   # invalid, but unread
        action_space_prob_floor=0.9,                     # invalid, but unread
        action_space_cem_score_horizon=2,
    )
    ref_pools, ref_rng = _run_pools(ref, states)
    alt_pools, alt_rng = _run_pools(alt, states)
    for a, b in zip(ref_pools, alt_pools):
        assert _pool_bytes(a) == _pool_bytes(b)
    assert torch.equal(ref_rng, alt_rng)
    assert not any(
        k.startswith("action_space_") for k in alt._last_propose_diagnostics
    )
    assert alt._decode_action_objects_calls == len(states) * K * I
    assert alt._terrain_prior_calls == len(states)


def test_off_rng_stream_is_the_pre_build_codec_stream():
    """Pins OFF against the PRE-BUILD code, not against itself: the comparison
    above runs the same (possibly regressed) OFF path on both sides, so it
    cannot see an OFF-path change common to both (measured: an injected
    torch.rand(1) on the OFF path passed it). The pre-build codec CEM consumes
    exactly K*I draws of randn([1, H, ao_dim]) per propose call and nothing
    else (verified against cc20be5 / 5f965cf's ree_core, 2026-09-25)."""
    module = _make_module(seed=0)
    zw, zs = _states(1)[0]
    torch.manual_seed(7)
    module.propose_trajectories(zw, z_self=zs)
    after_propose = torch.get_rng_state().clone()
    torch.manual_seed(7)
    for _ in range(K * I):
        torch.randn(1, H, 16)
    assert torch.equal(after_propose, torch.get_rng_state())


def test_on_changes_the_pool():
    zw, zs = _states(1)[0]
    off = _make_module(seed=0)
    on = _make_module(seed=0, use_action_space_proposals=True)
    torch.manual_seed(3)
    off_pool = off.propose_trajectories(zw, z_self=zs)
    torch.manual_seed(3)
    on_pool = on.propose_trajectories(zw, z_self=zs)
    assert _pool_bytes(off_pool) != _pool_bytes(on_pool)
    assert all(
        (t.metadata or {}).get("source") == "action_space_cem" for t in on_pool
    )
    assert on._last_propose_diagnostics["use_action_space_proposals"] is True


# --------------------------------------------------------------------------- #
# G-ASP (a): no group; decoder / terrain_prior off the act path               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["stratified", "refit", "stratified_uniform"])
def test_gate_a_decoder_and_terrain_prior_off_the_proposal_path(mode):
    off = _make_module(seed=0)
    on = _make_module(
        seed=0, use_action_space_proposals=True, action_space_first_action_mode=mode,
    )
    # No ASP parameter group: ON adds no parameter at all.
    assert [n for n, _ in on.named_parameters()] == [
        n for n, _ in off.named_parameters()
    ]

    grad_forwards = {"terrain_prior": 0, "action_object_decoder": 0}
    any_forwards = {"terrain_prior": 0, "action_object_decoder": 0}

    def _hook(name):
        def _h(_mod, _inp, _out):
            any_forwards[name] += 1
            if torch.is_grad_enabled():
                grad_forwards[name] += 1
        return _h

    on.terrain_prior.register_forward_hook(_hook("terrain_prior"))
    on.action_object_decoder.register_forward_hook(_hook("action_object_decoder"))

    zw, zs = _states(1)[0]
    zw.requires_grad_(True)
    torch.manual_seed(5)
    pool = on.propose_trajectories(zw, z_self=zs)
    diag = on._last_propose_diagnostics
    assert diag["action_space_decoder_calls"] == 0
    assert diag["action_space_terrain_prior_calls"] == 0
    assert any_forwards["terrain_prior"] == 0
    # The only decoder forward allowed is the no-grad round-trip DIAGNOSTIC
    # (action_object_roundtrip_recovery), never a proposal-path call.
    assert grad_forwards["action_object_decoder"] == 0
    assert any_forwards["action_object_decoder"] <= 1

    # G5 LEAK: nothing propose returns carries gradient into either module.
    loss = zw.sum() * 0.0
    for t in pool:
        loss = loss + t.get_world_state_sequence().sum() + t.actions.sum()
        ao = t.get_action_object_sequence()
        if ao is not None:
            loss = loss + ao.sum()
    loss.backward()
    for name in ("terrain_prior", "action_object_decoder"):
        for p in getattr(on, name).parameters():
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, name
    # ... while the rollout itself does carry gradient (the check is not vacuous).
    assert zw.grad is not None and float(zw.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# G-ASP (b): action validity                                                   #
# --------------------------------------------------------------------------- #


def _is_exact_one_hot(a):
    return bool(((a == 0) | (a == 1)).all()) and bool((a.sum(dim=-1) == 1).all())


@pytest.mark.parametrize("mode", ["stratified", "refit", "stratified_uniform"])
def test_gate_b_every_candidate_action_is_an_exact_one_hot(mode):
    """Stratified modes: the FINAL pool E3 receives. Refit: the candidates ASP
    generated (source action_space_cem) -- see the xfail below for the final
    refit pool."""
    module = _make_module(
        seed=0, use_action_space_proposals=True, action_space_first_action_mode=mode,
    )
    torch.manual_seed(9)
    for zw, zs in _states(25):
        pool = module.propose_trajectories(zw, z_self=zs)
        assert len(pool) == K
        if mode == "refit":
            pool = [
                t for t in pool
                if (t.metadata or {}).get("source") == "action_space_cem"
            ]
        for t in pool:
            a = t.actions.detach()
            assert a.shape[-2:] == (H, A)
            assert bool(((a == 0) | (a == 1)).all())      # no fractional entry
            assert bool((a.sum(dim=-1) == 1).all())         # no zero vector
        diag = module._last_propose_diagnostics
        assert diag["action_space_decoder_calls"] == 0
        assert diag["action_space_max_action_norm"] == 1.0
        assert diag["action_space_all_actions_one_hot"] is True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "G-ASP (b) on the FINAL refit pool FAILS: when ASP-R re-concentrates on "
        "one first-action class, the default-ON SP-CEM injector adds a "
        "_build_action_class_scaffold_candidates token whose continuation is "
        "ZERO vectors (design P2, module.py support-preserving injection). "
        "Measured: tick 3 of 25 at K=32, A=5, floor 0.02. Stratified modes are "
        "unaffected (every class present -> injector no-ops). The zero-"
        "continuation fix is an owner decision (MECH-131 / ARC-065); a fix must "
        "delete this marker."
    ),
)
def test_gate_b_final_refit_pool_is_all_one_hot():
    module = _make_module(
        seed=0, use_action_space_proposals=True,
        action_space_first_action_mode="refit",
    )
    torch.manual_seed(9)
    for zw, zs in _states(25):
        pool = module.propose_trajectories(zw, z_self=zs)
        assert all(_is_exact_one_hot(t.actions.detach()) for t in pool)


# --------------------------------------------------------------------------- #
# G-ASP (c): rollout-norm readout (verdict needs the W3 head)                 #
# --------------------------------------------------------------------------- #


def test_gate_c_rollout_norm_readout_matches_independent_recomputation():
    module = _make_module(seed=0, use_action_space_proposals=True)
    zw, zs = _states(1)[0]
    torch.manual_seed(2)
    pool = module.propose_trajectories(zw, z_self=zs)
    diag = module._last_propose_diagnostics
    ratios, growth = [], []
    for t in pool:
        norms = t.get_world_state_sequence().detach().norm(dim=-1).mean(dim=0)
        assert norms.shape[0] == H + 1
        ratios.append(float(norms[-1] / norms[0]))
        growth.append(float((norms[1:] / norms[:-1]).max()))
    med = lambda xs: float(torch.tensor(xs, dtype=torch.float64).median())  # noqa: E731
    assert diag["action_space_rollout_norm_ratio_median"] == pytest.approx(med(ratios), rel=1e-6)
    assert diag["action_space_rollout_max_step_growth_median"] == pytest.approx(
        med(growth), rel=1e-6
    )
    assert diag["action_space_rollout_max_step_growth_max"] == pytest.approx(
        max(growth), rel=1e-6
    )


# --------------------------------------------------------------------------- #
# G-ASP (d): coverage and support                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["stratified", "stratified_uniform"])
@pytest.mark.parametrize("k,a", [(32, 5), (8, 5), (32, 4)])
def test_gate_d_stratified_counts_exact_and_every_class_present(mode, k, a):
    module = _make_module(
        seed=0, action_dim=a, num_candidates=k,
        use_action_space_proposals=True, action_space_first_action_mode=mode,
    )
    expected = _stratified_counts(k, a)
    if (k, a) == (32, 5):
        assert list(expected.values()) == [7, 7, 6, 6, 6]
    torch.manual_seed(4)
    for zw, zs in _states(20):
        pool = module.propose_trajectories(zw, z_self=zs)
        counts = {c: 0 for c in range(a)}
        for t in pool:
            counts[int(t.actions[0, 0].argmax())] += 1
        assert counts == expected                    # exact n_c, every class present
        diag = module._last_propose_diagnostics
        assert diag["action_space_stratified_counts"] == expected
        assert diag["action_space_step0_counts"] == expected
        assert diag["support_preserving_injected_candidates"] == 0


def test_gate_d_refit_categoricals_never_drop_below_the_floor():
    floor = 0.02
    module = _make_module(
        seed=0, use_action_space_proposals=True,
        action_space_first_action_mode="refit", action_space_prob_floor=floor,
    )
    torch.manual_seed(4)
    for zw, zs in _states(10):
        module.propose_trajectories(zw, z_self=zs)
        for it in module._last_propose_diagnostics["cem_iteration_diagnostics"]:
            assert it["action_space_min_categorical_prob"] >= floor
    # And the refit rule itself: rows sum to 1, every entry >= floor.
    from ree_core.hippocampal.module import HippocampalModule
    elites = torch.tensor([[0, 0, 1], [0, 0, 1], [0, 2, 1]])
    p = HippocampalModule._asp_refit(elites, A, floor, torch.float64)
    assert p.shape == (3, A)
    assert torch.allclose(p.sum(dim=-1), torch.ones(3, dtype=torch.float64))
    assert float(p.min()) >= floor
    assert float(p[0, 0]) == pytest.approx((1 - floor * A) * 1.0 + floor)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "G-ASP (d) refit-mode coverage FAILS at the DRAFT floor 0.02: the joint "
        "categorical re-concentrates within 3 iterations (measured 4/40 pools "
        "with every class at K=32, A=5; floor 0.1 -> 37/40, 0.19 -> 40/40). "
        "ASP-R is the non-recommended variant; a fix (e.g. a higher floor) must "
        "delete this marker."
    ),
)
def test_gate_d_refit_coverage_every_class_in_95pct_of_pools():
    module = _make_module(
        seed=0, use_action_space_proposals=True,
        action_space_first_action_mode="refit",
    )
    torch.manual_seed(4)
    states = _states(40)
    full = 0
    for zw, zs in states:
        module.propose_trajectories(zw, z_self=zs)
        counts = module._last_propose_diagnostics["action_space_step0_counts"]
        full += int(all(v >= 1 for v in counts.values()))
    assert full / len(states) >= 0.95


# --------------------------------------------------------------------------- #
# Mechanism: refit moves the continuation; ASP-0 does not; window knob reaches  #
# the scorer; sampler never draws a zero-probability class                     #
# --------------------------------------------------------------------------- #


def test_stratified_refit_moves_the_continuation_and_asp0_does_not():
    zw, zs = _states(1)[0]
    asp_e = _make_module(seed=0, use_action_space_proposals=True)
    asp_0 = _make_module(
        seed=0, use_action_space_proposals=True,
        action_space_first_action_mode="stratified_uniform",
    )
    torch.manual_seed(1)
    asp_e.propose_trajectories(zw, z_self=zs)
    torch.manual_seed(1)
    asp_0.propose_trajectories(zw, z_self=zs)
    e_iters = asp_e._last_propose_diagnostics["cem_iteration_diagnostics"]
    z_iters = asp_0._last_propose_diagnostics["cem_iteration_diagnostics"]
    assert len(e_iters) == I and len(z_iters) == 1
    log_a = math.log(A)
    assert all(v == pytest.approx(log_a) for v in z_iters[-1]["action_space_continuation_entropy"])
    assert z_iters[-1]["action_space_refit_applied"] is False
    assert min(e_iters[-1]["action_space_continuation_entropy"]) < log_a - 1e-3


@pytest.mark.parametrize("window", [None, 3])
def test_score_horizon_knob_reaches_the_elite_scorer(window, monkeypatch):
    module = _make_module(
        seed=0, use_action_space_proposals=True,
        action_space_cem_score_horizon=window,
    )
    seen = []
    orig = module._score_trajectory

    def _spy(traj, max_horizon=None, **kw):
        seen.append(max_horizon)
        return orig(traj, max_horizon=max_horizon, **kw)

    monkeypatch.setattr(module, "_score_trajectory", _spy)
    zw, zs = _states(1)[0]
    module.propose_trajectories(zw, z_self=zs)
    assert len(seen) >= K * I
    assert seen[: K * I] == [window] * (K * I)


def test_sampler_never_draws_a_zero_probability_class():
    from ree_core.hippocampal.module import HippocampalModule

    probs = torch.tensor(
        [[0.0, 0.5, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64
    )
    torch.manual_seed(0)
    draws = HippocampalModule._asp_sample_categorical(probs, 2000)
    assert draws.shape == (2000, 2)
    assert set(draws[:, 0].tolist()) == {1, 3}
    assert set(draws[:, 1].tolist()) == {0}
    frac = float((draws[:, 0] == 1).double().mean())
    assert 0.45 < frac < 0.55


# --------------------------------------------------------------------------- #
# Construction refuses incompatible / invalid configurations                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "flag",
    [
        "use_differentiable_cem",
        "use_orthogonal_cem_seeding",
        "mode_conditioning_enabled",
        "use_mech293_ghost_probes",
        "use_cem_modulatory_authority",
    ],
)
def test_incompatible_o_space_option_raises(flag):
    with pytest.raises(ValueError, match="mutually exclusive"):
        _make_module(seed=0, use_action_space_proposals=True, **{flag: True})
    # ... and the same option is accepted with ASP off (the refusal is ASP's).
    # MECH-293 is skipped here: it has its own prerequisite chain (MECH-292
    # bank -> MECH-269 anchor sets -> SD-039 payload) checked later in
    # construction; the ASP refusal above fires before any of it.
    if flag != "use_mech293_ghost_probes":
        _make_module(seed=0, **{flag: True})


@pytest.mark.parametrize(
    "kw",
    [
        {"action_space_first_action_mode": "joint"},
        {"action_space_prob_floor": 0.2},     # floor * A == 1
        {"action_space_prob_floor": -0.01},
        {"action_space_cem_score_horizon": 0},
        {"action_space_cem_score_horizon": H + 1},
        {"action_space_cem_score_horizon": True},
    ],
)
def test_invalid_sub_knob_raises_when_on(kw):
    with pytest.raises(ValueError):
        _make_module(seed=0, use_action_space_proposals=True, **kw)
