"""Guard C: no-op-default contract for higher-version (V4/V5) flags.

Version-Layering Doctrine invariant: NO higher-version (V4/V5) code may change
V3 *default* execution behaviour. Concretely:

  C1. Every generation-tagged master flag in ree_core/version_layering.py
      defaults OFF on a default-built config.
  C2. The registry paths exist on a real config (drift guard -- a renamed flag
      must be caught, not silently skipped).
  C3. The default agent's forward/select path runs without error and is
      bit-identical to an explicit-all-OFF build (the no-op guarantee).

Sibling enforcement lives in tests/preflight/test_v3_parity_smoke.py (the runner
preflight) and ree_core/agent.py (the conditional DR-12 call-site, Guard A).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from tests.fixtures.seed_utils import set_all_seeds  # noqa: E402
from tests.fixtures.tiny_env import make_tiny_env  # noqa: E402
from tests.fixtures.tiny_configs import make_tiny_config  # noqa: E402
from tests.fixtures.tiny_loop import run_episode  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.version_layering import (  # noqa: E402
    GENERATION_FLAGS,
    _read_path,
    find_flags_on,
    assert_all_off,
)


def _default_cfg():
    env = make_tiny_env(seed=0)
    return make_tiny_config(env), env


def test_c1_every_generation_flag_defaults_off():
    """C1: a default config leaves every registered V4/V5 flag at its no-op value."""
    cfg, _env = _default_cfg()
    on = find_flags_on(cfg)
    assert on == [], f"V4/V5 flags not at no-op default on a default config: {on}"
    # assert_all_off is the helper the runner/preflight use -- exercise it too.
    assert_all_off(cfg)


def test_c2_registry_paths_exist_on_real_config():
    """C2: every registered path resolves on a real config (catches renamed flags)."""
    cfg, _env = _default_cfg()
    assert GENERATION_FLAGS, "registry is empty -- did the first V4 flag get dropped?"
    for flag in GENERATION_FLAGS:
        # Raises AttributeError if a segment is missing -> a renamed flag is a
        # hard failure, not a silent skip.
        value = _read_path(cfg, flag.config_path)
        assert value == flag.default_off, (
            f"{flag.config_path} default {value!r} != registered no-op "
            f"{flag.default_off!r}"
        )


def test_c3_default_path_bit_identical_to_explicit_all_off():
    """C3: default build == explicit-all-V4/V5-OFF build, and both run cleanly.

    The default agent already has every V4/V5 flag off; building a second agent
    with those same flags explicitly set to their no-op value must produce an
    identical action stream over the default select_action path.
    """
    set_all_seeds(0)
    env_a = make_tiny_env(seed=0)
    cfg_a = make_tiny_config(env_a)
    agent_a = REEAgent(cfg_a)
    actions_default = run_episode(agent_a, env_a, steps=6)

    # Explicit no-op overrides for every registered flag (here: DR-12).
    set_all_seeds(0)
    env_b = make_tiny_env(seed=0)
    overrides = {}
    for flag in GENERATION_FLAGS:
        # Only the master boolean flags are passed via from_dims kwargs; the
        # config attribute leaf name is the last path segment.
        leaf = flag.config_path.split(".")[-1]
        overrides[leaf] = flag.default_off
    cfg_b = make_tiny_config(env_b, **overrides)
    agent_b = REEAgent(cfg_b)
    actions_explicit = run_episode(agent_b, env_b, steps=6)

    assert actions_default == actions_explicit, (
        "explicit-all-OFF action stream diverged from default -- a V4/V5 flag's "
        "no-op default is not actually a no-op"
    )


def test_c4_default_select_action_runs_without_error():
    """C4: the default forward/select path runs to completion (no NaN, no crash)."""
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)
    agent = REEAgent(cfg)
    actions = run_episode(agent, env, steps=5)
    assert len(actions) == 5
    assert all(isinstance(a, int) for a in actions)
