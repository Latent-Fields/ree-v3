"""Contract tests for ``experiments/_harness.py`` and the underlying agent
API call shapes the harness assumes.

Background: 11 ree-v3 experiment scripts (EXQ-471/475/483/483a/483b/490/490b/
490c/490e/490f/524) all called

    agent.update_z_goal(latent, benefit_exposure=..., drive_level=...)

which raised TypeError on every tick. A bare ``except Exception: pass`` wrapper
silently swallowed the error, leaving GoalState.is_active() always False. The
StepHarness was created to make this class of bug structurally impossible.

Guarantees enforced here:
  H1. ``REEAgent.update_z_goal`` accepts only ``(benefit_exposure, drive_level=1.0)``
      kwargs. Pinning the signature catches any future refactor that would
      re-introduce a positional that scripts could collide with.
  H2. ``REEAgent.update_residue`` accepts ``(harm_signal, world_delta=None,
      hypothesis_tag=False, owned=True)`` -- the harness call shape.
  H3. ``StepHarness.step()`` runs end-to-end on a default tiny config without
      raising, advances the per-tick clock once, and produces a finite latent.
  H4. ``StepHarness.step()`` calls ``agent.sense`` exactly once and
      ``agent.update_residue`` exactly once per env step. (The cohort bug was
      double-sense; this is the structural fix.)
  H5. ``StepHarness.step()`` calls ``agent.update_z_goal`` exactly once per
      env step, kwargs only -- no positional that could collide.
  H6. ``StepHarness.step()`` leaves schema wanting untouched when the
      MECH-216 switch is off.
  H7. ``StepHarness.step()`` calls ``agent.update_schema_wanting`` exactly
      once before action selection when MECH-216 is enabled.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

# ree-v3/experiments is not on sys.path by default in pytest; add it.
EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from _harness import StepHarness, StepHooks, StepResult  # noqa: E402

from ree_core.agent import REEAgent

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config


# -- H1: update_z_goal signature pin -----------------------------------------

def test_update_z_goal_signature_is_kwargs_only_friendly():
    """Pin the exact parameter list. If this changes, every experiment script
    using StepHarness might silently break. Update the harness call site in
    lockstep with any change to this signature."""
    sig = inspect.signature(REEAgent.update_z_goal)
    params = list(sig.parameters.values())
    # First param is self.
    assert params[0].name == "self"
    # The rest must be exactly (benefit_exposure, drive_level, resource_type).
    # resource_type added 2026-06-04 for SD-057 (GAP-7 L2 object-identity
    # binding); it has a None default so positional 2-arg callers are
    # unaffected, and the StepHarness call site was updated in lockstep to
    # forward obs_dict["resource_type_at_agent"].
    real = [p.name for p in params[1:]]
    assert real == ["benefit_exposure", "drive_level", "resource_type"], (
        f"update_z_goal signature drift: got {real}; harness assumes "
        f"['benefit_exposure', 'drive_level', 'resource_type']. If you "
        f"intentionally added a parameter, update _harness.py and this test "
        f"together."
    )
    # drive_level should still have a default.
    drive_param = sig.parameters["drive_level"]
    assert drive_param.default is not inspect.Parameter.empty, (
        "drive_level lost its default; scripts that omit it will TypeError."
    )
    # resource_type must default to None (bit-identical legacy path when omitted).
    rtype_param = sig.parameters["resource_type"]
    assert rtype_param.default is None, (
        "resource_type must default to None so 2-arg callers stay bit-identical."
    )


# -- H2: update_residue signature pin ----------------------------------------

def test_update_residue_signature_matches_harness_call_shape():
    sig = inspect.signature(REEAgent.update_residue)
    params = [p.name for p in sig.parameters.values() if p.name != "self"]
    # Harness passes (harm_signal=, world_delta=, hypothesis_tag=, owned=) as kwargs.
    # The names must be in the signature; ordering is enforced separately.
    for required in ("harm_signal", "world_delta", "hypothesis_tag", "owned"):
        assert required in params, (
            f"update_residue lost the '{required}' parameter; the harness "
            f"calls it by keyword. Update _harness.py if intentional."
        )


# -- H3-H5: harness end-to-end smoke -----------------------------------------

class _CallCounter:
    """Wraps a method and counts invocations without changing semantics."""

    def __init__(self, owner, attr):
        self._owner = owner
        self._attr = attr
        self._original = getattr(owner, attr)
        self.calls = 0

        def wrapper(*args, **kwargs):
            self.calls += 1
            return self._original(*args, **kwargs)

        setattr(owner, attr, wrapper)

    def restore(self):
        setattr(self._owner, self._attr, self._original)


def test_step_harness_runs_one_step_end_to_end():
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)

    harness = StepHarness(agent, env, train_mode=False, seed=0)

    flat_obs, obs_dict = env.reset()
    agent.reset()
    harness.reset()

    result = harness.step(obs_dict)

    assert isinstance(result, StepResult)
    assert result.action.shape[-1] == cfg.e2.action_dim
    # Latent finite.
    assert torch.isfinite(result.latent.z_world).all()
    assert torch.isfinite(result.latent.z_self).all()
    # update_residue ran (returned a dict including harm_signal).
    assert "harm_signal" in result.residue_metrics


def test_step_harness_calls_sense_and_update_residue_exactly_once_per_step():
    """H4: structural fix for the double-sense cohort bug."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)

    sense_counter = _CallCounter(agent, "sense")
    residue_counter = _CallCounter(agent, "update_residue")
    z_goal_counter = _CallCounter(agent, "update_z_goal")

    try:
        harness = StepHarness(agent, env, train_mode=False, seed=0)
        flat_obs, obs_dict = env.reset()
        agent.reset()
        harness.reset()
        for _ in range(5):
            r = harness.step(obs_dict)
            obs_dict = r.next_obs_dict
            if r.done:
                break
    finally:
        sense_counter.restore()
        residue_counter.restore()
        z_goal_counter.restore()

    assert sense_counter.calls == 5, (
        f"sense() called {sense_counter.calls} times across 5 env steps; "
        f"expected exactly 5. The double-sense cohort bug is back."
    )
    assert residue_counter.calls == 5, (
        f"update_residue() called {residue_counter.calls} times across 5 "
        f"env steps; expected exactly 5. Without it, e3.post_action_update "
        f"never fires and the residue field stays empty."
    )
    assert z_goal_counter.calls == 5, (
        f"update_z_goal() called {z_goal_counter.calls} times across 5 env "
        f"steps; expected exactly 5."
    )


def test_step_harness_update_z_goal_call_uses_kwargs_only():
    """H5: the harness must call update_z_goal with ``benefit_exposure=`` and
    ``drive_level=`` kwargs and nothing else. This is the structural fix for
    the EXQ-471 cohort bug where scripts passed ``latent`` as a positional and
    silently TypeError'd."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)

    seen_calls = []
    real = agent.update_z_goal

    def spy(*args, **kwargs):
        seen_calls.append((args, dict(kwargs)))
        return real(*args, **kwargs)

    agent.update_z_goal = spy

    harness = StepHarness(agent, env, train_mode=False, seed=0)
    flat_obs, obs_dict = env.reset()
    obs_dict["body_state"] = obs_dict["body_state"].clone()
    obs_dict["body_state"][11] = 0.25
    agent.reset()
    harness.reset()
    harness.step(obs_dict)

    assert len(seen_calls) == 1
    args, kwargs = seen_calls[0]
    # No positional args beyond the bound self (which Python strips).
    assert args == (), (
        f"harness passed positional args to update_z_goal: {args}. The "
        f"cohort bug was passing `latent` positionally; the harness must "
        f"pass benefit_exposure and drive_level as kwargs only."
    )
    # SD-057 (2026-06-04): the harness also forwards resource_type (the SD-049
    # per-type identity tag) as a kwarg; None here because make_tiny_env does
    # not enable SD-049 multi_resource_heterogeneity (key absent -> None ->
    # bit-identical legacy path).
    assert set(kwargs.keys()) == {
        "benefit_exposure", "drive_level", "resource_type"
    }, (
        f"harness update_z_goal kwargs drift: {set(kwargs.keys())}"
    )
    assert kwargs["resource_type"] is None, (
        "resource_type must be None when SD-049 is off (no "
        "resource_type_at_agent key) -> bit-identical legacy path."
    )
    assert kwargs["benefit_exposure"] == pytest.approx(0.25), (
        "StepHarness must source benefit_exposure from body_state[11] when "
        "the env does not expose a top-level obs_dict['benefit_exposure'] key."
    )


def test_step_harness_default_off_does_not_call_schema_wanting():
    """H6: default-off runs should preserve the historical no-op path."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)

    calls = []
    real = agent.update_schema_wanting

    def spy(*args, **kwargs):
        calls.append((args, dict(kwargs)))
        return real(*args, **kwargs)

    agent.update_schema_wanting = spy

    harness = StepHarness(agent, env, train_mode=False, seed=0)
    flat_obs, obs_dict = env.reset()
    agent.reset()
    harness.reset()
    harness.step(obs_dict)

    assert calls == []


def test_step_harness_schema_wanting_runs_before_select_action_when_enabled():
    """H7: canonical MECH-216 write path must feed action-side consumers."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    cfg.e1.schema_wanting_enabled = True
    cfg.schema_wanting_threshold = 1.1
    agent = REEAgent(cfg)

    order = []
    real_schema = agent.update_schema_wanting
    real_select = agent.select_action

    def schema_spy(*args, **kwargs):
        order.append(("schema", args, dict(kwargs)))
        return real_schema(*args, **kwargs)

    def select_spy(*args, **kwargs):
        order.append(("select", args, dict(kwargs)))
        return real_select(*args, **kwargs)

    agent.update_schema_wanting = schema_spy
    agent.select_action = select_spy

    harness = StepHarness(agent, env, train_mode=False, seed=0)
    flat_obs, obs_dict = env.reset()
    agent.reset()
    harness.reset()
    harness.step(obs_dict)

    assert len(order) >= 2
    assert order[0][0] == "schema"
    assert order[1][0] == "select"
    assert order[0][1] == ()
    assert set(order[0][2].keys()) == {"drive_level"}
