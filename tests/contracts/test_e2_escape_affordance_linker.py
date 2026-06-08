"""Contracts for the post-603i E2 escape-affordance linker / readout.

The linker is a READOUT over the existing E2 (cerebellar-analog) action-
consequence forward model -- not a duplicate predictor. V3-EXQ-603i did NOT
weaken SD-059 / MECH-358; it surfaced a prerequisite-representation gap. These
contracts pin the scaffold's guarantees:

  1.  Disabled path is a no-op (module + agent default-off bit-identical).
  2.  Inputs (E2 + compact latents) are detached -- no backprop into upstream.
  3.  Positive escape transitions increase the readout for the successful action.
  4.  Failed escape transitions decrease the readout (extinction).
  5.  No-op/freeze receives no credit by default.
  6.  Simulation / hypothesis mode blocks learning.
  7.  Bias is exactly zero when safe.
  8.  Bias is clamped under threat (never exceeds bias_scale).
  9.  Learned weights persist across episode reset; traces are cleared.
  10. Linker features can be consumed by TrainableEscapeAffordanceLearner
      without changing its default behaviour.
  + agent default-off bit-identical and E2-reuse (world_forward feature flows).
  + forced-choice readiness microdiagnostic (2/3-seed gates).
"""

import math

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.pfc.e2_escape_affordance_linker import (
    E2EscapeAffordanceLinker,
    E2EscapeAffordanceLinkerConfig,
)
from ree_core.pfc.trainable_escape_affordance_learner import (
    TrainableEscapeAffordanceLearner,
    TrainableEscapeAffordanceLearnerConfig,
)
from ree_core.utils.config import REEConfig


# Distinct per-action E2-consequence features so the shared trunk can discriminate.
_E2FEAT = {
    0: torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    1: torch.tensor([0.9, -0.8, 0.1, 0.2], dtype=torch.float32),
    2: torch.tensor([-0.7, 0.6, -0.5, 0.4], dtype=torch.float32),
    3: torch.tensor([0.1, 0.1, 0.1, -0.1], dtype=torch.float32),
}


def _linker(**kw):
    torch.manual_seed(7)
    params = dict(
        enabled=True,
        n_action_classes=5,
        hidden_dim=32,
        action_embedding_dim=6,
        learn_rate=0.3,
        optimizer_lr=0.02,
        leak_rate=0.0,
        bias_scale=0.2,
        threat_floor=0.1,
        threat_ref=0.5,
        noop_class=0,
        relief_reward_floor=0.05,
        harm_delta_scale=0.6,
        prediction_floor=0.02,
    )
    params.update(kw)
    return E2EscapeAffordanceLinker(E2EscapeAffordanceLinkerConfig(**params))


def _e2(action):
    return _E2FEAT.get(int(action), _E2FEAT[2])


def _trial(linker, action, threat=0.6, outcome=0.05, **kw):
    """One clean (under-threat -> outcome) escape trial; resets trace after."""
    linker.update(threat, last_action_class=action, e2_features=_e2(action), **kw)
    out = linker.update(outcome, last_action_class=action, e2_features=_e2(action), **kw)
    linker.reset()
    return out


def _train_escape(linker, action=2, n=40):
    for _ in range(n):
        _trial(linker, action, threat=0.6, outcome=0.04)


# --- 1. disabled path is a no-op ---------------------------------------------

def test_disabled_path_is_noop_module_and_agent_default_off():
    lk = _linker(enabled=False)
    _trial(lk, 2)
    bias = lk.compute_approach_bias(0.6, [0, 1, 2, 3, 4])
    st = lk.get_state()
    assert lk.model is None and lk.optimizer is None
    assert float(bias.abs().max()) == 0.0
    assert st["e2_escape_linker_n_optimizer_steps"] == 0

    def run(seed, **kw):
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = REEConfig.from_dims(
            world_obs_dim=250, body_obs_dim=12, harm_obs_dim=50,
            harm_obs_a_dim=50, action_dim=5, **kw
        )
        ag = REEAgent(cfg)
        if not kw.get("use_e2_escape_affordance_linker", False):
            assert ag.e2_escape_affordance_linker is None
        return [
            int(ag.act_with_split_obs(torch.zeros(1, 12), torch.zeros(1, 250))
                .argmax(dim=-1).flatten()[0].item())
            for _ in range(8)
        ]

    assert run(0) == run(0, use_e2_escape_affordance_linker=False)


# --- 2. inputs detached ------------------------------------------------------

def test_inputs_detached_from_upstream_latents():
    lk = _linker()
    e2 = torch.tensor([-0.7, 0.6, -0.5, 0.4], dtype=torch.float32, requires_grad=True)
    zw = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float32, requires_grad=True)
    zha = torch.tensor([0.6], dtype=torch.float32, requires_grad=True)
    lk.update(0.6, last_action_class=2, e2_features=e2, z_world=zw, z_harm_a=zha)
    lk.update(0.05, last_action_class=2, e2_features=e2, z_world=zw, z_harm_a=zha)
    assert lk.model is not None
    assert e2.grad is None and zw.grad is None and zha.grad is None


# --- 3. positive escape increases readout ------------------------------------

def test_positive_escape_increases_prediction():
    lk = _linker()
    before = lk.escape_salience(2, lk.build_state_vector(e2_features=_e2(2), z_harm_a_norm=0.6, threat_scale=1.0))
    _train_escape(lk, action=2, n=40)
    after = lk.escape_salience(2, lk.build_state_vector(e2_features=_e2(2), z_harm_a_norm=0.6, threat_scale=1.0))
    st = lk.get_state()
    assert after > before + 0.1
    assert st["e2_escape_linker_n_positive"] > 0
    assert st["e2_escape_linker_n_optimizer_steps"] > 0
    assert lk.predict_head("harm_delta", 2,
                           lk.build_state_vector(e2_features=_e2(2), z_harm_a_norm=0.6, threat_scale=1.0)) > 0.0


# --- 4. failed escape decreases readout --------------------------------------

def test_failed_escape_decreases_prediction():
    lk = _linker()
    # Moderate pre-training so the readout is non-trivial but not saturated.
    _train_escape(lk, action=2, n=12)
    probe = lk.build_state_vector(e2_features=_e2(2), z_harm_a_norm=0.6, threat_scale=1.0)
    before = lk.escape_salience(2, probe)
    # directed action 2 under threat that does NOT reduce harm or end threat ->
    # extinction of both the harm-delta and threat-termination readouts.
    for _ in range(80):
        _trial(lk, 2, threat=0.6, outcome=0.6)
    after = lk.escape_salience(2, probe)
    assert after < before - 0.05
    assert lk.get_state()["e2_escape_linker_n_negative"] > 0


# --- 5. no-op/freeze gets no credit ------------------------------------------

def test_noop_freeze_gets_no_credit_by_default():
    lk = _linker()
    for _ in range(20):
        _trial(lk, 0, threat=0.6, outcome=0.04)   # no-op class
    st = lk.get_state()
    bias = lk.compute_approach_bias(0.6, [0, 1, 2])
    assert st["e2_escape_linker_n_optimizer_steps"] == 0
    assert st["e2_escape_linker_n_noop_skipped"] > 0
    assert lk.model is None
    assert float(bias[0]) == 0.0
    assert float(bias.abs().max()) == 0.0


# --- 6. simulation / hypothesis blocks learning ------------------------------

def test_simulation_and_hypothesis_block_learning():
    sim = _linker()
    sim.update(0.6, last_action_class=2, e2_features=_e2(2))
    sim.update(0.05, last_action_class=2, e2_features=_e2(2), simulation_mode=True)
    assert sim.model is None
    assert sim.get_state()["e2_escape_linker_n_optimizer_steps"] == 0

    hyp = _linker()
    hyp.update(0.6, last_action_class=2, e2_features=_e2(2))
    hyp.update(0.05, last_action_class=2, e2_features=_e2(2), hypothesis_tag=True)
    assert hyp.model is None
    assert hyp.get_state()["e2_escape_linker_n_optimizer_steps"] == 0


# --- 7 + 8. bias zero when safe; clamped under threat ------------------------

def test_bias_zero_when_safe_and_clamped_under_threat():
    lk = _linker(bias_scale=0.05)
    _train_escape(lk, action=2, n=40)
    safe = lk.compute_approach_bias(0.0, [0, 1, 2, 3, 4])
    assert float(safe.abs().max()) == 0.0
    threat = lk.compute_approach_bias(0.6, [0, 1, 2, 3, 4])
    assert threat[2] < 0.0
    assert float(threat[0]) == 0.0
    assert float(threat.abs().max()) <= lk.config.bias_scale + 1e-9
    assert lk.get_state()["e2_escape_linker_n_bias_fires"] > 0


# --- 9. weights persist across reset -----------------------------------------

def test_learned_weights_persist_across_reset():
    lk = _linker()
    _train_escape(lk, action=2, n=30)
    probe = lk.build_state_vector(e2_features=_e2(2), z_harm_a_norm=0.6, threat_scale=1.0)
    model_before = lk.model
    pred_before = lk.predict_head("harm_delta", 2, probe)
    lk.reset()
    pred_after = lk.predict_head("harm_delta", 2, probe)
    assert lk.model is model_before
    assert math.isclose(pred_after, pred_before, rel_tol=0.0, abs_tol=1e-7)
    assert lk._prev_z_harm_a_norm is None and lk._prev_state_vector is None


# --- 10. learner consumes linker features without changing default ----------

def test_learner_consumes_linker_features_without_changing_default():
    lk = _linker()
    _train_escape(lk, action=2, n=30)
    feat = lk.escape_affordance_features(2)
    assert feat is not None and feat.numel() > 0

    learner = TrainableEscapeAffordanceLearner(
        TrainableEscapeAffordanceLearnerConfig(enabled=True, n_action_classes=5)
    )
    base = learner.build_state_vector(
        z_world=torch.tensor([0.2, -0.1, 0.4]),
        z_self=torch.tensor([0.3, -0.2]),
        z_harm_a=torch.tensor([0.6]),
        z_harm_a_norm=0.6,
        threat_scale=1.0,
    )
    with_feat = learner.build_state_vector(
        z_world=torch.tensor([0.2, -0.1, 0.4]),
        z_self=torch.tensor([0.3, -0.2]),
        z_harm_a=torch.tensor([0.6]),
        z_harm_a_norm=0.6,
        threat_scale=1.0,
        extra_features=feat,
    )
    # Default (no extra_features) is unchanged; the linker feature is appended.
    assert with_feat.numel() == base.numel() + feat.numel()
    # The base build is bit-identical to passing extra_features=None.
    base_none = learner.build_state_vector(
        z_world=torch.tensor([0.2, -0.1, 0.4]),
        z_self=torch.tensor([0.3, -0.2]),
        z_harm_a=torch.tensor([0.6]),
        z_harm_a_norm=0.6,
        threat_scale=1.0,
        extra_features=None,
    )
    assert torch.equal(base, base_none)


# --- agent E2-reuse: world_forward feature flows into the linker -------------

def test_agent_reuses_e2_world_forward_feature():
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = REEConfig.from_dims(
        world_obs_dim=250, body_obs_dim=12, harm_obs_dim=50, harm_obs_a_dim=50,
        action_dim=5, use_e2_escape_affordance_linker=True,
        use_e2_escape_linker_e3_bias=True,
    )
    ag = REEAgent(cfg)
    assert ag.e2_escape_affordance_linker is not None
    for _ in range(6):
        ag.act_with_split_obs(torch.zeros(1, 12), torch.zeros(1, 250))
    st = ag.e2_escape_affordance_linker.get_state()
    # The detached E2 forward feature was read on >=1 tick (reuse path fires).
    assert st["e2_escape_linker_n_e2_feature_ticks"] > 0


# --- forced-choice readiness microdiagnostic --------------------------------

def _microdiagnostic_seed(seed):
    """Tiny forced-choice readiness probe -- NOT an ecological survival claim.

    Actions: 0=no-op/freeze, 1=harm-worsening, 2=escape-producing, 3=neutral.
    Each non-noop action gets a distinct E2 consequence feature and its true
    outcome. Returns the readiness booleans for this seed.
    """
    torch.manual_seed(seed)
    lk = _linker(n_action_classes=4, noop_class=0)

    probe = lambda a: lk.build_state_vector(e2_features=_e2(a), z_harm_a_norm=0.6, threat_scale=1.0)
    esc_before = lk.escape_salience(2, probe(2))

    for _ in range(24):
        _trial(lk, 2, threat=0.6, outcome=0.04)   # escape: harm drops, threat ends
        _trial(lk, 1, threat=0.6, outcome=0.95)   # harm-worsening: harm rises
        _trial(lk, 3, threat=0.6, outcome=0.60)   # neutral: no change, still threat
        _trial(lk, 0, threat=0.6, outcome=0.04)   # no-op: skipped (no credit)

    esc_after = lk.escape_salience(2, probe(2))
    bias = lk.compute_approach_bias(0.6, [0, 1, 2, 3])
    safe = lk.compute_approach_bias(0.0, [0, 1, 2, 3])
    nonnoop = {c: float(bias[c]) for c in (1, 2, 3)}
    bias_points_to_escape = min(nonnoop, key=nonnoop.get) == 2  # most negative = favoured
    noop_uncredited = (
        float(bias[0]) == 0.0
        and lk.get_state()["e2_escape_linker_n_noop_skipped"] > 0
    )
    return {
        "learns_escape": esc_after > esc_before + 0.05,
        "bias_points_to_escape": bias_points_to_escape,
        "noop_uncredited": noop_uncredited,
        "bias_zero_when_safe": float(safe.abs().max()) == 0.0,
    }


def test_forced_choice_readiness_microdiagnostic():
    seeds = [0, 1, 2]
    results = [_microdiagnostic_seed(s) for s in seeds]

    def passed(gate):
        return sum(1 for r in results if r[gate])

    # 2/3-seed readiness gates (substrate readiness only -- no survival claim).
    assert passed("learns_escape") >= 2
    assert passed("bias_points_to_escape") >= 2
    assert passed("noop_uncredited") >= 2
    # Threat-gated bias is zero when safe in every seed.
    assert passed("bias_zero_when_safe") == 3

    # Learning blocked under simulation/hypothesis (all seeds, deterministic).
    sim = _linker(n_action_classes=4, noop_class=0)
    sim.update(0.6, last_action_class=2, e2_features=_e2(2))
    sim.update(0.04, last_action_class=2, e2_features=_e2(2), simulation_mode=True)
    assert sim.model is None
