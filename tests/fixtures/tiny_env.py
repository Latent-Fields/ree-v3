"""Tiny deterministic CausalGridWorldV2 fixture for regression tests.

Small grid, few entities, fixed seed. Intentionally minimal so tests run on
CPU in well under a second.
"""

from ree_core.environment.causal_grid_world import CausalGridWorldV2


def make_tiny_env(seed: int = 0) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=5,
        num_hazards=1,
        num_resources=1,
        use_proxy_fields=True,
    )
