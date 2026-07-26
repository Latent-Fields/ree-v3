"""Canonical OFF-arm baseline for the DOSE-ESCALATED HARSHENED-ENV
policy_decomposition discrimination sub-lineage (V3-EXQ-816d and successors).

WHY A SEPARATE (v2) HARSHENED MODULE (do NOT edit mech321_policy_decomposition_harshened)
    V3-EXQ-816b (failure_autopsy_V3-EXQ-816b_2026-07-26.json) ran the
    `mech321_policy_decomposition_harshened` dose (env_drift_interval=3,
    world_rule_shift_interval=24, world_rule_shift_depth=1) and moved
    off_pe_mean_worst from the 816/820 baseline (~0.005) to 0.0086 -- 86% of
    the PE_ELEVATED_FLOOR=0.01 discrimination floor, but still short of it, so
    zero low-V_s steps resulted (same failure shape as 816/820). The autopsy's
    routing_note (user-confirmed 2026-07-26) explicitly recommends "a next
    letter (env axis, stronger dose -- e.g. world_rule_shift_depth 1->2
    and/or env_drift_interval->1)" continuing the SAME hypothesis
    (H-env-underdrives-uncertainty) BEFORE running the P-B measurement-
    comparator probe (V3-EXQ-816c, which presupposes PE is already elevated).

    This escalated dose is a DIFFERENT OFF closure from 816b's (different env
    non-stationarity magnitude), so per the arm-fingerprint reuse model it
    needs its OWN canonical baseline module -- editing
    `mech321_policy_decomposition_harshened.py` in place would silently
    invalidate V3-EXQ-816c's already-run reuse contract with 816b's mint.

DOSE ESCALATION (all three autopsy-named levers, each moved one step further)
    | knob                        | 816b (short of floor) | 816d (this dose) |
    |------------------------------|------------------------|-------------------|
    | env_drift_interval           | 3                      | 1 (every step)    |
    | world_rule_shift_interval    | 24                     | 12 (halved)       |
    | world_rule_shift_depth       | 1                      | 2 (doubled)       |

    env_drift_interval=1 is the fastest possible hazard-drift setting (drift
    fires every step past step 0; see causal_grid_world.py `self.steps %
    self.env_drift_interval == 0`). world_rule_shift_interval=12 doubles the
    shift frequency (twice per episode-worth of steps instead of once).
    world_rule_shift_depth=2 applies two random action-pair transpositions per
    shift instead of one, per `_maybe_shift_world_rule()` (depth is a simple
    loop over `self.world_rule_shift_depth` random `_rng.choice` swaps -- a
    legal, monotonic escalation, not a new lever). Everything else (grid
    shape, hazards, resources, proxy fields, substrate stack, schedule,
    alpha_world, seeded chunk) is inherited unchanged from the harshened (v1)
    module, so the ONLY difference from 816b is dose magnitude on the same
    three axes -- a clean escalation, not a redesign.

MINT / REUSE
    V3-EXQ-816d is the FIRST experiment of this dose-escalated sub-lineage:
    its ARM_0 (OFF) cells are emitted reuse-ELIGIBLE
    (include_driver_script_in_hash=False) and constructed from THIS module, so
    any future same-dose successor matches the fingerprint BY CONSTRUCTION.
"""
from __future__ import annotations

from typing import Any, Dict, List

from experiments._lib.baselines import mech321_policy_decomposition_harshened as base_h

# --- Env shape / schedule / representation / chunk: inherited unchanged from
# the (v1) harshened module, which itself inherits from the default lineage
# module -- single source of truth all the way down. ---
ENV_SIZE = base_h.ENV_SIZE
ENV_NUM_HAZARDS = base_h.ENV_NUM_HAZARDS
ENV_NUM_RESOURCES = base_h.ENV_NUM_RESOURCES
ENV_USE_PROXY_FIELDS = base_h.ENV_USE_PROXY_FIELDS
WARMUP_EPISODES = base_h.WARMUP_EPISODES
MEASURE_EPISODES = base_h.MEASURE_EPISODES
STEPS_PER_EPISODE = base_h.STEPS_PER_EPISODE
ALPHA_WORLD = base_h.ALPHA_WORLD
SEEDED_CHUNK_SEQUENCE = base_h.SEEDED_CHUNK_SEQUENCE

# --- Escalated harshening knobs (the ONLY difference from the v1 harshened
# module -- see the dose table in the module docstring above). ---
HARSH_ENV_DRIFT_INTERVAL = 1
HARSH_WORLD_RULE_SHIFT_ENABLED = True
HARSH_WORLD_RULE_SHIFT_INTERVAL = 12
HARSH_WORLD_RULE_SHIFT_DEPTH = 2


def total_episodes() -> int:
    """Denominator M for the runner progress contract (warmup + measure)."""
    return WARMUP_EPISODES + MEASURE_EPISODES


def env_kwargs(seed: int) -> Dict[str, Any]:
    """Dose-escalated harshened CausalGridWorldV2 kwargs."""
    return {
        "size": ENV_SIZE,
        "num_hazards": ENV_NUM_HAZARDS,
        "num_resources": ENV_NUM_RESOURCES,
        "use_proxy_fields": ENV_USE_PROXY_FIELDS,
        "env_drift_interval": HARSH_ENV_DRIFT_INTERVAL,
        "world_rule_shift_enabled": HARSH_WORLD_RULE_SHIFT_ENABLED,
        "world_rule_shift_interval": HARSH_WORLD_RULE_SHIFT_INTERVAL,
        "world_rule_shift_depth": HARSH_WORLD_RULE_SHIFT_DEPTH,
        "seed": seed,
    }


def substrate_stack_flags() -> Dict[str, Any]:
    """The MECH-321 required substrate stack -- IDENTICAL to the default lineage."""
    return base_h.substrate_stack_flags()


def off_arm_flags() -> Dict[str, Any]:
    """ARM_0: substrate stack ON, decomposition OFF -- identical to default lineage."""
    return base_h.off_arm_flags()


def off_path_config_slice() -> Dict[str, Any]:
    """The canonical dose-escalated harshened OFF-arm fingerprint config slice.

    MUST be identical byte-for-byte between V3-EXQ-816d's ARM_0 emit and any
    successor's try_reuse_cell(...) call, or reuse is (correctly) refused.
    Declares only what the OFF computation reads -- including the escalated
    harshening knobs (they change the env the OFF arm runs in), and EXCLUDING
    ARM_1-only decomposition knobs (inert when use_policy_decomposition is
    False).
    """
    slice_: Dict[str, Any] = {
        "env": {
            "size": ENV_SIZE,
            "num_hazards": ENV_NUM_HAZARDS,
            "num_resources": ENV_NUM_RESOURCES,
            "use_proxy_fields": ENV_USE_PROXY_FIELDS,
            "env_drift_interval": HARSH_ENV_DRIFT_INTERVAL,
            "world_rule_shift_enabled": HARSH_WORLD_RULE_SHIFT_ENABLED,
            "world_rule_shift_interval": HARSH_WORLD_RULE_SHIFT_INTERVAL,
            "world_rule_shift_depth": HARSH_WORLD_RULE_SHIFT_DEPTH,
        },
        "schedule": {
            "warmup_episodes": WARMUP_EPISODES,
            "measure_episodes": MEASURE_EPISODES,
            "steps": STEPS_PER_EPISODE,
        },
        "alpha_world": ALPHA_WORLD,
        "seeded_chunk_sequence": list(SEEDED_CHUNK_SEQUENCE),
    }
    slice_.update(off_arm_flags())
    return slice_


def off_needed_keys() -> List[str]:
    """OFF-arm metrics a consumer reads back from a reused escalated-dose cell."""
    return [
        "region_vs_mean",
        "region_vs_min",
        "region_vs_max",
        "low_vs_step_frac",
        "fwd_pe_all_mean",
        "fwd_pe_all_var",
        "n_measure_steps",
    ]
