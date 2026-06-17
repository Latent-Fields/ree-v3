"""V3-parity smoke: prove the default V3 execution path is intact.

This is the shared core of Guard B (runner preflight) and is also exercised by
the startup preflight test (tests/preflight/test_v3_parity_smoke.py).

It builds a DEFAULT tiny agent (every higher-version V4/V5 flag at its no-op
default per ree_core/version_layering.py) and runs the default-path
agent.select_action loop a few steps. If construction, the no-op-default
invariant, or select_action raises, the smoke raises -- which the runner treats
as "refuse to claim" (a transient cross-checkout skew like the 2026-06-17
V3-EXQ-654e DR-12 incident self-heals on the next pull rather than crash-burning
a claimed experiment).

Kept dependency-light and fast (one tiny agent, a handful of steps) so it is
cheap enough to run before claiming.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_v3_parity_smoke(steps: int = 3) -> None:
    """Run the default V3 select_action path; raise on any failure.

    Raises whatever the underlying build/step raises (TypeError on a skewed
    call-site, AssertionError on a version-layering violation, etc.). Callers
    that want a boolean should wrap this in try/except.
    """
    # Lazy imports: keep import-time cost out of the runner's module load.
    from tests.fixtures.seed_utils import set_all_seeds
    from tests.fixtures.tiny_env import make_tiny_env
    from tests.fixtures.tiny_configs import make_tiny_config
    from tests.fixtures.tiny_loop import run_episode
    from ree_core.agent import REEAgent
    from ree_core.version_layering import assert_all_off

    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env)

    # Invariant: a default config must leave every registered V4/V5 flag OFF.
    # A flipped default would mean a higher-version change leaked into V3-default
    # behaviour -- exactly what the doctrine forbids.
    assert_all_off(cfg)

    agent = REEAgent(cfg)

    # Default-path select_action loop. run_episode resets env+agent and drives
    # the canonical sense -> generate_trajectories -> select_action -> env.step
    # sequence. A skewed/older shared module (e.g. an e3_selector.select missing
    # a V4 param while agent.py passes it unconditionally) raises here.
    run_episode(agent, env, steps=steps)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    run_v3_parity_smoke()
    print("[v3_parity_smoke] OK")
