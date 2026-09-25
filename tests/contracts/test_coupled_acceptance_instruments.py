"""Contracts for experiments/_lib/coupled_acceptance.py (coupled campaign plan node I1).

Every instrument there is a NEGATIVE instrument (REE_Working CLAUDE.md "Negative
instruments"): these tests pin (1) each instrument's CANARY, a known-baseline finding that
must keep reproducing, and (2) the explicit CANNOT_DETERMINE category for empty/degenerate
derivations. They follow the test-half rule: each check is a function of the instrument,
and each is ALSO run against a deliberately broken instrument (monkeypatched in-test) and
must FAIL there -- so a guard that stopped being able to notice breakage fails here too.

Kept fast: toy predictors, 6x6 grid, a handful of agent steps; no training.
"""
from __future__ import annotations

import copy
import functools
import json
import types

import numpy as np
import pytest
import torch

import experiments._lib.coupled_acceptance as C
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


def _fails_when_broken(monkeypatch, check, breaks):
    """Run ``check`` with every (obj, name, value) in ``breaks`` patched: it must raise
    AssertionError. The test-half rule, applied inline."""
    with monkeypatch.context() as m:
        for obj, name, value in breaks:
            m.setattr(obj, name, value)
        with pytest.raises(AssertionError):
            check()


def _no_cd(reason, **extra):
    """A broken _cd: the CANNOT_DETERMINE category silently removed."""
    out = {"verdict": C.MEASURED}
    out.update(extra)
    return out


@pytest.fixture(scope="module")
def small_agent():
    torch.manual_seed(3)
    np.random.seed(3)
    env = CausalGridWorldV2(size=6, num_hazards=2, num_resources=3, max_episode_steps=200, seed=3)
    cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                              action_dim=env.action_dim, alpha_world=0.3)
    agent = REEAgent(cfg)
    agent.eval()
    return env, agent


def _latents(env, agent, n=4, seed=0):
    g = np.random.default_rng(seed)
    _f, obs = env.reset()
    agent.reset()
    out = []
    for _ in range(n):
        lat = C.sense_latent(agent, obs)
        out.append((lat.z_world.detach().clone(), lat.z_self.detach().clone()))
        _f, _h, done, _i, obs = env.step(int(g.integers(0, 4)))
        if done:
            _f, obs = env.reset()
            agent.reset()
    return out


# ------------------------------------------------------------------ 1. action discrimination
def _check_disc_canary():
    r = C.canary_action_discrimination(0)
    assert r["reproduced"], r
    assert r["real"]["disc4_h1"] >= 0.95 and r["real"]["k"] >= 3
    assert abs(r["blind"]["disc4_h1"] - 0.25) < 1e-9, "action-blind head must score exactly chance"


def test_action_discrimination_canary_reproduces(monkeypatch):
    _check_disc_canary()
    # broken: credit every start (tie-fair credit replaced by 'always hit')
    _fails_when_broken(monkeypatch, _check_disc_canary, [(C, "_tie_fair_hit", lambda errs, t: (1.0, False))])
    # broken: credit "argmin is class 0" instead of "argmin is the executed class"
    _fails_when_broken(monkeypatch, _check_disc_canary,
                       [(C, "_tie_fair_hit", lambda errs, t: (1.0 if int(np.argmin(errs)) == 0 else 0.0, False))])


def _check_disc_cannot_determine():
    eps, _ = C._toy_world(0, world_seed=0)
    p = C._delta_predictor(C._fit_delta_head(eps, 5))
    short = [{"z": e["z"][:5], "a": e["a"][:4]} for e in eps]
    assert C.action_discrimination(p, short, action_dim=5)["verdict"] == C.CANNOT_DETERMINE
    static = [{"z": torch.zeros_like(e["z"]), "a": e["a"]} for e in eps]
    assert C.action_discrimination(p, static, action_dim=5)["verdict"] == C.CANNOT_DETERMINE
    no3 = [{"z": e["z"], "a": torch.where(e["a"] == 3, torch.zeros_like(e["a"]), e["a"])} for e in eps]
    r = C.action_discrimination(p, no3, action_dim=5)
    assert r["verdict"] == C.CANNOT_DETERMINE and "never executed" in r["reason"], r
    ok = C.action_discrimination(p, eps, action_dim=5, bar_disc4_h1=0.9)
    assert ok["verdict"] == C.PASS
    assert C.action_discrimination(p, eps, action_dim=5, bar_disc4_h1=1.01)["verdict"] == C.FAIL


def test_action_discrimination_cannot_determine(monkeypatch):
    _check_disc_cannot_determine()
    _fails_when_broken(monkeypatch, _check_disc_cannot_determine, [(C, "_cd", _no_cd)])


def test_action_discrimination_runs_on_native_e2(small_agent):
    env, agent = small_agent
    eps = C.collect_uniform_random_episodes(agent, env, 60, seed=11)
    r = C.action_discrimination(C.e2_world_predictor(agent.e2), eps, action_dim=env.action_dim,
                                horizons=(1,), fidelity_horizon=3, max_starts=20)
    assert r["verdict"] in C.VERDICTS
    assert r["n_starts"] > 0 and set(r["executed_class_counts"]) == {str(c) for c in range(env.action_dim)}
    assert r["executed_class_counts"]["4"] == 0, "held-out test set is uniform over classes 0..3 only"


# ------------------------------------------------------------------ 2. cloned-env probe states
def _check_probe_validation(env, agent):
    states = C.collect_probe_states(agent, env, 10, 4, seed=5, n_cont=1, cont_len=1)
    v = C.probe_state_validation(states)
    assert v["verdict"] == C.PASS and v["max_abs_diff"] == 0.0, v
    assert all(len(s["z_true"]) == env.action_dim and s["q"].shape == (env.action_dim,) for s in states)


def test_probe_states_reproduce_sense_exactly(monkeypatch, small_agent):
    env, agent = small_agent
    _check_probe_validation(env, agent)
    orig = C.encode_next_side_effect_free
    _fails_when_broken(monkeypatch, lambda: _check_probe_validation(env, agent),
                       [(C, "encode_next_side_effect_free", lambda *a, **k: orig(*a, **k) + 1e-3)])


def _check_env_q_does_not_mutate(env):
    env.reset()
    before = copy.deepcopy(env.__dict__.get("agent_x")), copy.deepcopy(env.__dict__.get("agent_y"))
    q = C.env_q_values(env, env.action_dim, np.random.default_rng(0), n_cont=2, cont_len=3)
    assert q.shape == (env.action_dim,)
    assert (env.agent_x, env.agent_y) == before, "env_q_values must act on deep copies only"


def test_env_q_uses_clones(monkeypatch, small_agent):
    env, _agent = small_agent
    env2 = CausalGridWorldV2(size=6, num_hazards=2, num_resources=3, max_episode_steps=200, seed=4)
    _check_env_q_does_not_mutate(env2)
    _fails_when_broken(monkeypatch, lambda: _check_env_q_does_not_mutate(env2),
                       [(C, "copy", types.SimpleNamespace(deepcopy=lambda x: x, copy=copy.copy))])


def _check_probe_validation_cd():
    assert C.probe_state_validation([])["verdict"] == C.CANNOT_DETERMINE
    assert C.probe_state_validation([{"validation_exact_maxabs": 0.5}])["verdict"] == C.FAIL


def test_probe_validation_cannot_determine(monkeypatch):
    _check_probe_validation_cd()
    _fails_when_broken(monkeypatch, _check_probe_validation_cd, [(C, "_cd", _no_cd)])


# ------------------------------------------------------------------ 3. E3 decomposition
def _check_e3_canary():
    r = C.canary_e3_structure(0)
    assert r["reproduced"], r


def test_e3_depth_structure_canary(monkeypatch):
    _check_e3_canary()
    orig = C.e3_depth_structure
    blind = lambda score, pools, **kw: orig(lambda t, n=None: score(t, None), pools, **kw)  # never truncates
    _fails_when_broken(monkeypatch, _check_e3_canary, [(C, "e3_depth_structure", blind)])


def _check_choice_quality():
    truth = [np.array([0.0, 1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0, 0.0])]
    good = [t.copy() for t in truth]
    bad = [-t for t in truth]
    r = C.e3_choice_quality({"FULL": good, "DEPTH1": bad}, truth)
    assert r["arms"]["FULL"]["pick_in_best"] == 1.0 and r["arms"]["FULL"]["spearman_mean"] == pytest.approx(1.0)
    assert r["arms"]["DEPTH1"]["pick_in_best"] == 0.0 and r["arms"]["DEPTH1"]["spearman_mean"] == pytest.approx(-1.0)
    assert r["chance"] == pytest.approx(0.25)
    q = C.e3_choice_quality({"FULL": good}, [-t for t in truth], truth_lower_is_better=False)
    assert q["arms"]["FULL"]["pick_in_best"] == 1.0, "env-Q truth is higher-is-better"
    assert C.e3_choice_quality({"FULL": good}, truth, bar_arm="FULL", bar_margin_over_chance=0.5)["verdict"] == C.PASS
    assert C.e3_choice_quality({"FULL": bad}, truth, bar_arm="FULL", bar_margin_over_chance=0.1)["verdict"] == C.FAIL
    flat = [np.zeros(4), np.ones(4)]
    assert C.e3_choice_quality({"FULL": good}, flat)["verdict"] == C.CANNOT_DETERMINE
    assert C.e3_depth_structure(lambda t, n=None: np.zeros(len(t)), [[object()]])["verdict"] == C.CANNOT_DETERMINE
    hs = C.head_swap_flip_rate([np.array([0.0, 1.0]), np.array([1.0, 0.0])], [np.array([0.0, 1.0]), np.array([0.0, 1.0])])
    assert hs["flip_rate"] == 0.5
    assert C.head_swap_flip_rate([np.array([1.0])], [np.array([1.0])])["verdict"] == C.CANNOT_DETERMINE


def test_e3_choice_quality_and_head_swap(monkeypatch):
    _check_choice_quality()
    _fails_when_broken(monkeypatch, _check_choice_quality, [(C, "_cd", _no_cd)])
    _fails_when_broken(monkeypatch, _check_choice_quality, [(C, "spearman", lambda a, b: 1.0)])


# ------------------------------------------------------------------ 4. codec trace
def test_codec_trace_canary_untrained_contraction_and_gain_divergence(monkeypatch, small_agent):
    env, agent = small_agent
    lat = _latents(env, agent, n=4)

    def check():
        r = C.canary_codec_trace(agent, lat, seed=1)
        assert r["reproduced"], r

    before = {k: v.clone() for k, v in agent.hippocampal.action_object_decoder.state_dict().items()}
    check()
    after = agent.hippocampal.action_object_decoder.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "canary must restore the decoder"
    assert agent.hippocampal._decode_action_objects.__func__ is type(agent.hippocampal)._decode_action_objects, \
        "cem_codec_trace must restore the wrapped method"
    # broken: image-norm scale bug (reads 10x too large) hides the iteration-0 mismatch
    orig = C.encoder_image_norm
    _fails_when_broken(monkeypatch, check, [(C, "encoder_image_norm", lambda *a, **k: 10.0 * orig(*a, **k))])


def _check_codec_cd():
    assert C.codec_ranges({"verdict": C.CANNOT_DETERMINE, "reason": "x"}, 1.0)["verdict"] == C.CANNOT_DETERMINE
    tr = {"verdict": C.MEASURED, "per_iteration": [{"ao_input_norm_median": 1.0, "decoded_norm_median": 1.0}],
          "decoded_growth_per_iteration": []}
    assert C.codec_ranges(tr, 0.0)["verdict"] == C.CANNOT_DETERMINE
    assert C.codec_ranges(tr, 1.0)["verdict"] == C.PASS
    assert C.pool_qbest_coverage([[0, 1]], [np.zeros(5)])["verdict"] == C.CANNOT_DETERMINE
    cov = C.pool_qbest_coverage([[0, 1], [2, 2]], [np.array([1.0, 0, 0, 0, 0]), np.array([1.0, 0, 0, 0, 0])])
    assert cov["coverage"] == 0.5
    assert C.proposal_m4([[0, 0, 1]])["verdict"] == C.CANNOT_DETERMINE
    m4 = C.proposal_m4([[0, 0, 1], [0, 2, 2], [3, 3, 3]])
    assert m4["frac_states_with_modal_majority"] == pytest.approx(1.0 / 3.0)


def test_codec_readouts_cannot_determine(monkeypatch):
    _check_codec_cd()
    _fails_when_broken(monkeypatch, _check_codec_cd, [(C, "_cd", _no_cd)])


# ------------------------------------------------------------------ 5. stratum classifier
def _check_stratum_canary():
    r = C.canary_stratum()
    assert r["reproduced"], r


def test_stratum_canary(monkeypatch):
    _check_stratum_canary()
    _fails_when_broken(monkeypatch, _check_stratum_canary,
                       [(C, "classify_stratum", functools.partial(C.classify_stratum, min_early=9))])
    _fails_when_broken(monkeypatch, _check_stratum_canary,
                       [(C, "classify_stratum", functools.partial(C.classify_stratum, window=599))])


def _check_sidecar_order(tmp_path, writer, require):
    p = str(tmp_path / "s301" / "stratum.json")
    res = C.classify_stratum([False] * 600)
    with pytest.raises(C.StratumOrderError):
        require(p, 301)                                     # nothing classified yet
    with pytest.raises(C.StratumOrderError):
        writer(p, seed=301, result=res, arm="INTEGRATED")   # not the NATIVE arm
    with pytest.raises(C.StratumOrderError):
        writer(p, seed=301, result=res, arms_already_read=("NATIVE", "INTEGRATED"))
    with pytest.raises(C.StratumOrderError):
        writer(p, seed=301, result=C.classify_stratum([False] * 10))  # CANNOT_DETERMINE
    rec = writer(p, seed=301, result=res)
    assert rec["stratum"] == C.BENIGN and require(p, 301) == C.BENIGN
    assert writer(p, seed=301, result=res)["stratum"] == C.BENIGN   # identical rewrite: no-op
    trapped = C.classify_stratum(([False] * 39 + [True]) * 12 + [False] * 200)
    with pytest.raises(C.StratumOrderError):
        writer(p, seed=301, result=trapped)                 # write-once
    assert json.load(open(p))["source_arm"] == "NATIVE"


def test_stratum_sidecar_written_before_other_arms(monkeypatch, tmp_path):
    _check_sidecar_order(tmp_path / "ok", C.write_stratum_sidecar, C.require_stratum_sidecar)

    def naive_writer(path, *, seed, result, arm="NATIVE", arms_already_read=()):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rec = {"seed": seed, "stratum": result.get("stratum"), "source_arm": arm}
        json.dump(rec, open(path, "w"))
        return rec

    with pytest.raises(AssertionError):
        try:
            _check_sidecar_order(tmp_path / "broken", naive_writer, C.require_stratum_sidecar)
        except pytest.fail.Exception as e:  # pytest.raises(...) not raising -> Failed
            raise AssertionError(str(e))


def test_admit_seeds_under_admission_is_cannot_determine():
    seq = [(301 + i, C.BENIGN if i % 3 else C.TRAPPED) for i in range(30)]
    r = C.admit_seeds(seq)
    assert r["verdict"] == C.MEASURED and len(r["admitted"][C.BENIGN]) == 5 and len(r["admitted"][C.TRAPPED]) == 5
    few = C.admit_seeds([(301 + i, C.BENIGN) for i in range(90)])
    assert few["verdict"] == C.CANNOT_DETERMINE and few["stratum_verdict"][C.TRAPPED] == C.CANNOT_DETERMINE
    assert few["n_screened"] == 80


# ------------------------------------------------------------------ 6. outcome decomposition
def _check_outcome_canary():
    r = C.canary_outcome_decomposition()
    assert r["reproduced"], r


def test_outcome_decomposition_canary(monkeypatch):
    _check_outcome_canary()
    merged = (("true_contacts", C.TRUE_CONTACT_TYPES + C.PROXIMITY_TYPES), ("proximity_steps", ()),
              ("consumptions", C.CONSUMPTION_TYPES), ("approach_steps", C.APPROACH_TYPES))
    _fails_when_broken(monkeypatch, _check_outcome_canary, [(C, "_CATEGORIES", merged)])
    _fails_when_broken(monkeypatch, _check_outcome_canary, [(C, "_cd", _no_cd)])


def _check_real_env_coverage():
    env = CausalGridWorldV2(size=6, num_hazards=3, num_resources=3, max_episode_steps=200, seed=1)
    g = np.random.default_rng(0)
    env.reset()
    tts, rw = [], []
    for _ in range(400):
        _f, h, d, info, _o = env.step(int(g.integers(0, env.action_dim)))
        tts.append(C.step_outcome(info))
        rw.append(float(h))
        if d:
            env.reset()
    r = C.outcome_decomposition(tts, rw)
    assert r["verdict"] == C.MEASURED, r
    assert r["true_contacts"] > 0 and r["proximity_steps"] > 0, "canary regime must exercise both"
    harmful = C.TRUE_CONTACT_TYPES + C.PROXIMITY_TYPES
    assert all(t in harmful for t, h in zip(tts, rw) if h < 0), "a harmful step escaped both harm categories"


def test_outcome_decomposition_covers_real_env_types(monkeypatch):
    _check_real_env_coverage()
    no_env_hazard = (("true_contacts", ("agent_caused_hazard", "env_caused_multisource")),
                     ("proximity_steps", C.PROXIMITY_TYPES), ("consumptions", C.CONSUMPTION_TYPES),
                     ("approach_steps", C.APPROACH_TYPES))
    _fails_when_broken(monkeypatch, _check_real_env_coverage, [(C, "_CATEGORIES", no_env_hazard)])
    _fails_when_broken(monkeypatch, _check_real_env_coverage, [(C, "step_outcome", lambda info: None)])
