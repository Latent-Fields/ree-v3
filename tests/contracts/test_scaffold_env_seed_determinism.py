"""Contracts for `ScaffoldedSD054OnboardingConfig.scaffold_env_seed`.

WHAT THIS PINS. Every `_build_env(...)` in the scaffold scheduler used to build a
CausalGridWorldV2 without passing `seed`. CausalGridWorldV2 is a FACTORY FUNCTION
that forwards to `CausalGridWorld.__init__`, whose only RNG is
`self._rng = np.random.default_rng(seed)`; `default_rng(None)` seeds from OS
entropy AT CONSTRUCTION and does NOT consume the numpy global RNG. So a driver's
`np.random.seed(seed)` / `torch.manual_seed(seed)` never reached the env, and the
grid layout / resource placement / agent spawn differed on every construction --
not merely per process. That is the residual entropy source that made the scaffold
curriculum non-reproducible across processes, so a no-op / bit-identical-by-design
change to scaffold-driving code could not be verified by re-running and diffing.

Triage:
REE_assembly/evidence/planning/scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md

Two directions are pinned here and BOTH matter:
  * knob UNSET -> `seed=None` is still passed through and the build counter never
    advances, i.e. bit-identical to every landed scaffold run;
  * knob SET   -> a deterministic, distinct, reproducible seed per construction,
    and a same-seed env reproduces its trajectory bitwise.

Deliberately env-level rather than whole-curriculum: the causal chain is
seed -> env layout -> observations, and a full dry-run curriculum costs minutes.
The end-to-end result (two byte-identical full 460c dry-run curricula once the env
seed is pinned) is recorded in the triage doc, not re-run here.
"""
import os
import sys

import numpy as np
import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _derive_env_seed,
)


def _cfg(**kw):
    return ScaffoldedSD054OnboardingConfig(**kw)


# --------------------------------------------------------------------------- #
# Default path: bit-identical to every landed scaffold run
# --------------------------------------------------------------------------- #

def test_default_env_seed_is_unset():
    """The knob must default OFF -- flipping it changes results for 90 importers."""
    assert _cfg().scaffold_env_seed is None


def test_default_path_returns_none_and_never_advances_the_counter():
    """Bit-identity guarantee: unset knob -> seed=None, counter frozen at 0."""
    sched = ScaffoldedSD054OnboardingScheduler(_cfg())
    assert sched._env_build_count == 0
    for _ in range(5):
        assert sched._next_env_seed() is None
    assert sched._env_build_count == 0, (
        "the build counter advanced on the default path -- the knob is no longer "
        "a no-op when unset"
    )


def test_derive_env_seed_passes_none_through():
    assert _derive_env_seed(None, stream=0, idx=0) is None
    assert _derive_env_seed(None, stream=1, idx=7) is None


def test_default_build_env_hands_seed_none_to_the_factory(monkeypatch):
    """Pin the WIRING, not just the helper: the default must still reach the env
    constructor as seed=None."""
    from ree_core.environment import causal_grid_world as cgw

    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cgw, "CausalGridWorldV2", _capture)
    _build_env(_cfg(), phase="stage0")
    assert "seed" in seen, "_build_env stopped forwarding a seed kwarg entirely"
    assert seen["seed"] is None


# --------------------------------------------------------------------------- #
# Seeded path: deterministic, distinct, reproducible
# --------------------------------------------------------------------------- #

def test_seeded_scheduler_sequence_is_deterministic_and_distinct():
    a = ScaffoldedSD054OnboardingScheduler(_cfg(scaffold_env_seed=123))
    b = ScaffoldedSD054OnboardingScheduler(_cfg(scaffold_env_seed=123))
    seq_a = [a._next_env_seed() for _ in range(12)]
    seq_b = [b._next_env_seed() for _ in range(12)]
    assert seq_a == seq_b, "same base seed gave different sequences across schedulers"
    assert all(s is not None for s in seq_a)
    assert len(set(seq_a)) == len(seq_a), (
        "constructions collided on one seed -- every stage would share a layout"
    )


def test_distinct_bases_do_not_collide():
    a = [_derive_env_seed(1, stream=0, idx=i) for i in range(50)]
    b = [_derive_env_seed(2, stream=0, idx=i) for i in range(50)]
    assert not (set(a) & set(b))


def test_probe_stream_cannot_collide_with_curriculum_stream():
    """Stream 1 (read-only harm probe) must stay clear of stream 0 (curriculum)."""
    curriculum = {_derive_env_seed(9, stream=0, idx=i) for i in range(1000)}
    probe = _derive_env_seed(9, stream=1, idx=0)
    assert probe not in curriculum


def test_build_env_forwards_the_seed(monkeypatch):
    from ree_core.environment import causal_grid_world as cgw

    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(cgw, "CausalGridWorldV2", _capture)
    _build_env(_cfg(), phase="p0", seed=4242)
    assert seen["seed"] == 4242


# --------------------------------------------------------------------------- #
# The property that actually matters: same seed -> same env trajectory
# --------------------------------------------------------------------------- #

def _rollout_fingerprint(seed, n_steps=12):
    """Deterministic scripted rollout -- no policy, so ONLY the env is under test."""
    cfg = _cfg()
    env = _build_env(cfg, phase="stage0", seed=seed)
    obs, info = env.reset()
    arr = obs.detach().cpu().numpy() if torch.is_tensor(obs) else np.asarray(obs)
    parts = [arr.copy(), np.array([env.agent_x, env.agent_y], dtype=np.float64)]
    for t in range(n_steps):
        nxt, _reward, done, _info, _obs_dict = env.step(t % env.action_dim)
        parts.append(
            nxt.detach().cpu().numpy() if torch.is_tensor(nxt) else np.asarray(nxt)
        )
        parts.append(np.array([env.agent_x, env.agent_y], dtype=np.float64))
        if done:
            # Length is itself part of the fingerprint -- a seed that terminates
            # earlier produces a shorter vector, which array_equal already catches.
            break
    return np.concatenate([p.ravel().astype(np.float64) for p in parts])


def test_same_seed_gives_bit_identical_env_trajectory():
    """The causal link the knob buys: seed -> layout -> observations."""
    a = _rollout_fingerprint(20260728)
    b = _rollout_fingerprint(20260728)
    assert np.array_equal(a, b), (
        "two envs built with the SAME explicit seed diverged -- the env carries "
        "an entropy source beyond its own _rng and scaffold_env_seed cannot make "
        "the curriculum reproducible"
    )


def test_global_reseeding_does_not_rescue_an_unseeded_env():
    """Pins the DEFECT itself, deterministically.

    `np.random.default_rng(None)` ignores the numpy global RNG, so reseeding the
    globals identically before each construction cannot make unseeded envs agree.
    Asserted as 'not all identical' over several builds rather than 'the first two
    differ', so it cannot flake on a chance layout collision.
    """
    fps = []
    for _ in range(6):
        np.random.seed(42)
        torch.manual_seed(42)
        fps.append(_rollout_fingerprint(None, n_steps=4))
    all_same = all(np.array_equal(fps[0], f) for f in fps[1:])
    assert not all_same, (
        "unseeded envs became reproducible under global reseeding -- if the env "
        "RNG was genuinely made seed-following, retire this test and the knob's "
        "SCOPE note rather than weakening the assertion"
    )


def test_seeded_and_unseeded_paths_coexist():
    """A seeded build must not perturb a later unseeded one into determinism, and
    vice versa -- they are independent."""
    seeded_1 = _rollout_fingerprint(777, n_steps=4)
    _ = _rollout_fingerprint(None, n_steps=4)
    seeded_2 = _rollout_fingerprint(777, n_steps=4)
    assert np.array_equal(seeded_1, seeded_2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
