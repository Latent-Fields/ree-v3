"""Contracts for the structured babbling source (campaign W2a; default OFF).

Plan of record: REE_assembly evidence/planning/coupled_loop_repair_campaign_plan.md,
section 3 W2 (W2a). Form: the L2 dose of
evidence/planning/babbling_e2_action_coverage_probe_20260925.md (0ac69c87446), widened
from classes {0..3} to ALL env action classes (stay = 4 included; that probe's premise 1
found the native Phase-0 generator never emits class 4). Build:
ree_core/developmental/structured_babbling.py ``StructuredBabbler``; REEAgent builds it
only when ``structured_babbling_enabled``; nothing in ree_core calls it.

  B1  OFF: the flag defaults False; the agent holds ``structured_babbler = None``; the
      class is never constructed; in a fresh interpreter a default agent never imports
      the module.
  B2  OFF / ON byte-identity: a default rollout equals the same rollout with the flag ON
      (construction draws nothing from the global RNG and nothing calls the source), so
      OFF equals the pre-change path; B2b the comparison DETECTS a constructor that
      draws one global torch number (not blind).
  B3  ON: the agent holds a StructuredBabbler over the env's full action count (5,
      incl. stay) with runs 1..4, from the config.
  B4  RNG neutrality: constructing and drawing 5,000 actions leaves the global torch /
      numpy / python RNG states identical.
  B5  Class balance: every class incl. stay (4) at 1/n +- 0.02 over 20,000 steps; the
      balance check FAILS a 591c-style stream that never emits stay (not blind).
  B6  Run lengths uniform on {1..4} (each 0.25 +- 0.02, none outside), and the
      observable persistence P(a_t == a_{t-1}) matches the L2 form's 0.68 +- 0.02; a
      memoryless uniform stream (0.20) fails that check (not blind).
  B7  Seeded: same seed -> same stream; different seed -> different stream.
  B8  Knobs reach the agent through ``REEConfig.from_dims``; n_classes 0 -> the
      from_dims action_dim; one-hot output shape [1, n_classes].
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments._harness import StepHarness
from ree_core.agent import REEAgent
from ree_core.developmental import structured_babbling as sb_mod
from ree_core.developmental.structured_babbling import StructuredBabbler
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

SEED = 7
TICKS = 30
EP_LEN = 12
DIM = 32
N_DRAW = 20000
REPO = Path(__file__).resolve().parents[2]


def _env(seed: int = SEED) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=3, num_resources=2, hazard_harm=0.5,
                             seed=seed)


def _cfg(**flags) -> REEConfig:
    env = _env()
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=DIM, world_dim=DIM, alpha_world=0.3, **flags)


def _rng_states():
    return (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())


def _rng_equal(a, b):
    return (torch.equal(a[0], b[0]) and a[1][0] == b[1][0]
            and np.array_equal(a[1][1], b[1][1]) and tuple(a[1][2:]) == tuple(b[1][2:])
            and a[2] == b[2])


def _rollout(cfg: REEConfig, ticks: int = TICKS, seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = REEAgent(cfg)
    harness = StepHarness(agent, _env(seed), train_mode=False, seed=seed)
    actions, zworld = [], []
    while len(actions) < ticks:
        for r in harness.run_episode(max_steps=min(EP_LEN, ticks - len(actions))):
            actions.append(int(r.action.argmax().item()))
            zworld.append(r.latent.z_world.detach().clone())
    return {"agent": agent, "actions": actions, "zworld": zworld,
            "state": {k: v.detach().clone() for k, v in agent.state_dict().items()},
            "rng": _rng_states()}


def _differences(a, b):
    diffs = []
    if a["actions"] != b["actions"]:
        diffs.append("actions")
    if any(not torch.equal(x, y) for x, y in zip(a["zworld"], b["zworld"])):
        diffs.append("z_world")
    if set(a["state"]) != set(b["state"]) or any(
            not torch.equal(a["state"][k], b["state"][k]) for k in a["state"]):
        diffs.append("state_dict")
    if not _rng_equal(a["rng"], b["rng"]):
        diffs.append("rng")
    return diffs


def _stream(b: StructuredBabbler, n: int):
    classes, runs = [], []
    for _ in range(n):
        before = b.n_runs
        classes.append(b.next_class())
        if b.n_runs != before:
            runs.append(b.last_run_length)
    return classes, runs


def _balance_ok(classes, n_classes, tol=0.02):
    freq = np.bincount(np.asarray(classes), minlength=n_classes)[:n_classes] / len(classes)
    return bool(np.all(np.abs(freq - 1.0 / n_classes) <= tol)), freq


def _persistence(classes):
    c = np.asarray(classes)
    return float(np.mean(c[1:] == c[:-1]))


# --- B1 / B2: default OFF ----------------------------------------------------------------

def test_b1_off_builds_nothing(monkeypatch):
    assert REEConfig().structured_babbling_enabled is False

    def _boom(*a, **k):
        raise AssertionError("StructuredBabbler constructed at default config")

    monkeypatch.setattr(StructuredBabbler, "__init__", _boom)
    agent = REEAgent(_cfg())
    assert agent.structured_babbler is None


def test_b1b_off_never_imports_the_module_fresh_interpreter():
    code = (
        "import sys; from ree_core.agent import REEAgent; "
        "from ree_core.utils.config import REEConfig; "
        "c = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=5, "
        "self_dim=32, world_dim=32, alpha_world=0.3); a = REEAgent(c); "
        "print(int('ree_core.developmental.structured_babbling' in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO), capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "0"


def test_b2_off_and_on_rollouts_byte_identical():
    off = _rollout(_cfg())
    on = _rollout(_cfg(structured_babbling_enabled=True))
    assert off["agent"].structured_babbler is None
    assert isinstance(on["agent"].structured_babbler, StructuredBabbler)
    assert on["agent"].structured_babbler.n_emitted == 0      # nothing called it
    assert _differences(off, on) == []


def test_b2b_identity_check_detects_a_global_rng_draw(monkeypatch):
    ref = _rollout(_cfg(structured_babbling_enabled=True))
    orig = StructuredBabbler.__init__

    def _leaky(self, *a, **k):
        torch.rand(1)
        orig(self, *a, **k)

    monkeypatch.setattr(StructuredBabbler, "__init__", _leaky)
    leaky = _rollout(_cfg(structured_babbling_enabled=True))
    assert "rng" in _differences(ref, leaky)


# --- B3 / B4 ------------------------------------------------------------------------------

def test_b3_on_builds_full_action_count_source():
    env = _env()
    b = REEAgent(_cfg(structured_babbling_enabled=True)).structured_babbler
    assert env.action_dim == 5
    assert (b.n_classes, b.max_run, b.seed) == (env.action_dim, 4, 0)


def test_b4_construction_and_draws_are_global_rng_neutral():
    torch.manual_seed(3)
    np.random.seed(3)
    random.seed(3)
    before = _rng_states()
    b = StructuredBabbler(n_classes=5, max_run=4, seed=11)
    for _ in range(5000):
        b.next_action()
    assert _rng_equal(before, _rng_states())


# --- B5 / B6: distribution ------------------------------------------------------------

def test_b5_class_balance_includes_stay_and_is_not_blind():
    classes, _ = _stream(StructuredBabbler(n_classes=5, max_run=4, seed=0), N_DRAW)
    ok, freq = _balance_ok(classes, 5)
    assert ok, freq
    assert freq[4] > 0.15                                     # stay really emitted
    # Not blind: a 591c-style stream (argmax % 4: class 4 never emitted) fails.
    no_stay, _ = _stream(StructuredBabbler(n_classes=4, max_run=4, seed=0), N_DRAW)
    assert not _balance_ok(no_stay, 5)[0]


def test_b6_run_lengths_uniform_and_persistence_matches_l2_form():
    classes, runs = _stream(StructuredBabbler(n_classes=5, max_run=4, seed=0), N_DRAW)
    runs = np.asarray(runs)
    assert runs.min() >= 1 and runs.max() <= 4
    frac = np.bincount(runs, minlength=5)[1:5] / len(runs)
    assert np.all(np.abs(frac - 0.25) <= 0.02), frac
    # E[L] = 2.5; P(same) = (E[L]-1)/E[L] + (1/E[L]) * (1/5) = 0.6 + 0.08 = 0.68
    assert abs(_persistence(classes) - 0.68) <= 0.02, _persistence(classes)
    # Not blind: a memoryless uniform stream (max_run 1) has P(same) = 0.20.
    flat, _ = _stream(StructuredBabbler(n_classes=5, max_run=1, seed=0), N_DRAW)
    assert abs(_persistence(flat) - 0.68) > 0.02


def test_b7_seeded_stream():
    a, _ = _stream(StructuredBabbler(5, 4, seed=5), 500)
    b, _ = _stream(StructuredBabbler(5, 4, seed=5), 500)
    c, _ = _stream(StructuredBabbler(5, 4, seed=6), 500)
    assert a == b and a != c


# --- B8: knobs ------------------------------------------------------------------------------

def test_b8_knobs_plumb_through_from_dims_and_output_shape():
    cfg = _cfg(structured_babbling_enabled=True, structured_babbling_n_classes=3,
               structured_babbling_max_run=2, structured_babbling_seed=9)
    assert (cfg.structured_babbling_enabled, cfg.structured_babbling_n_classes,
            cfg.structured_babbling_max_run, cfg.structured_babbling_seed) == (True, 3, 2, 9)
    b = REEAgent(cfg).structured_babbler
    assert (b.n_classes, b.max_run, b.seed) == (3, 2, 9)
    a = b.next_action()
    assert a.shape == (1, 3) and float(a.sum()) == 1.0
    with pytest.raises(ValueError):
        StructuredBabbler(n_classes=0)
    assert sb_mod.StructuredBabbler is StructuredBabbler
