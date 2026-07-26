"""Canonical OFF-arm baseline for the HARSHENED-ENV policy_decomposition
discrimination sub-lineage (V3-EXQ-816b / V3-EXQ-816c -- the GOV-FANOUT-1
portfolio spun out of the V3-EXQ-816 + V3-EXQ-820 cluster autopsy).

WHY A SEPARATE HARSHENED MODULE (do NOT edit the default one)
    The default lineage module `mech321_policy_decomposition` bakes the *easy*
    env (8x8, 3 hazards, env_drift_interval=5, 40 warmup) into its
    off_path_config_slice. V3-EXQ-816 + V3-EXQ-820 both FAILed on that env with
    ZERO low-V_s steps in every arm/seed: the trained forward model reached
    near-zero PE (~0.005) by the measure window, so region-V_s never dropped
    below 0.5 and the R1 V_s-drop trigger's condition was never satisfied
    (failure_autopsy_816-820-policy-decomposition-cluster_2026-07-26). The
    autopsy routed a GOV-FANOUT-1 discrimination portfolio whose legs run on a
    HARSHENED env. That harshened env is a DIFFERENT OFF closure, so it needs its
    own canonical baseline module -- and the default module MUST stay untouched
    so V3-EXQ-816 / V3-EXQ-820 (still runnable) keep matching their own mints.

WHAT "HARSHENED" MEANS HERE -- and why not env_drift_interval ALONE
    The autopsy (and the /queue-experiment task) name `env_drift_interval -> 3`,
    citing V3-EXQ-677. But the env's OWN source note (causal_grid_world.py, the
    world_rule_shift kwargs block) records that env_drift_interval merely MOVES
    hazards on a random walk, and "the optimal prediction of a random walk is its
    mean, so the world-forward model learns that fast and PE floors at the
    irreducible noise level -- which is why V3-EXQ-677's env_drift_interval
    999 -> 3 produced a high-vs-low mean-PE DIFFERENCE of 8.8e-07 against a 0.01
    threshold". In other words env_drift_interval alone is a KNOWN-WEAK
    prediction-error lever. If H-env-underdrives-uncertainty is to be tested
    FAIRLY -- so that a null ("still no low-V_s") is a real refutation of H-env
    and not just "you picked a weak knob" -- the harshening must also invalidate
    LEARNED STRUCTURE, which is exactly what world_rule_shift_* was built for: it
    periodically re-permutes the action -> displacement map so
    E2.world_forward(z_world, a) becomes systematically wrong and stays wrong
    until re-learned (learnable between shifts, invalidated at each shift =
    genuine, graded re-learning load). So this module applies BOTH levers:

      * env_drift_interval = 3          (the task-named lever; spatial PE via
                                         faster hazard movement)
      * world_rule_shift_enabled = True (the purpose-built lever; sustained,
        world_rule_shift_interval = 24   temporal PE via structure invalidation
        world_rule_shift_depth = 1       -- ~one action-pair transposition per
                                         episode-worth of steps)

    Everything else (grid size, hazards, resources, proxy fields, the substrate
    stack, the schedule, alpha_world, the seeded crystallised chunk) is IDENTICAL
    to the default module, so the ONLY difference between the easy and harshened
    lineages is the env non-stationarity -- a clean single-axis manipulation.

MINT / REUSE
    V3-EXQ-816b is the FIRST experiment of this harshened sub-lineage: its ARM_0
    (OFF) cells are emitted reuse-ELIGIBLE (include_driver_script_in_hash=False)
    and constructed from THIS module, so V3-EXQ-816c (P-B) -- whose observation
    arm is the same harshened OFF path -- matches the fingerprint BY CONSTRUCTION
    and can reuse it (or self-mint if the parallel schedule runs P-B first).
"""
from __future__ import annotations

from typing import Any, Dict, List

from experiments._lib.baselines import mech321_policy_decomposition as base

# --- Env shape: inherited from the default module (single source of truth). ---
ENV_SIZE = base.ENV_SIZE
ENV_NUM_HAZARDS = base.ENV_NUM_HAZARDS
ENV_NUM_RESOURCES = base.ENV_NUM_RESOURCES
ENV_USE_PROXY_FIELDS = base.ENV_USE_PROXY_FIELDS

# --- Harshening knobs (the ONLY difference from the default lineage). ---
# env_drift_interval: faster hazard movement (task-named; spatial PE heterogeneity).
HARSH_ENV_DRIFT_INTERVAL = 3
# world_rule_shift_*: the purpose-built non-converging-world lever. Re-permutes the
# action->displacement map every HARSH_WORLD_RULE_SHIFT_INTERVAL world-steps, so the
# forward model faces sustained re-learning load (elevated forward-PE that DECAYS
# within each stationary window -> the low-V_s regime the R1 trigger needs).
HARSH_WORLD_RULE_SHIFT_ENABLED = True
HARSH_WORLD_RULE_SHIFT_INTERVAL = 24   # ~one shift per episode-worth of steps
HARSH_WORLD_RULE_SHIFT_DEPTH = 1       # one action-pair transposition per shift

# --- Schedule / representation / chunk: inherited unchanged. ---
WARMUP_EPISODES = base.WARMUP_EPISODES
MEASURE_EPISODES = base.MEASURE_EPISODES
STEPS_PER_EPISODE = base.STEPS_PER_EPISODE
ALPHA_WORLD = base.ALPHA_WORLD
SEEDED_CHUNK_SEQUENCE = base.SEEDED_CHUNK_SEQUENCE


def total_episodes() -> int:
    """Denominator M for the runner progress contract (warmup + measure)."""
    return WARMUP_EPISODES + MEASURE_EPISODES


def env_kwargs(seed: int) -> Dict[str, Any]:
    """Harshened CausalGridWorldV2 kwargs: base shape + the two harshening levers."""
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
    return base.substrate_stack_flags()


def off_arm_flags() -> Dict[str, Any]:
    """ARM_0: substrate stack ON, decomposition OFF -- identical to default lineage."""
    return base.off_arm_flags()


def off_path_config_slice() -> Dict[str, Any]:
    """The canonical harshened OFF-arm fingerprint config slice (producer + consumer).

    MUST be identical byte-for-byte between V3-EXQ-816b's ARM_0 emit and any
    successor's try_reuse_cell(...) call, or reuse is (correctly) refused.
    Declares only what the OFF computation reads -- including the harshening knobs
    (they change the env the OFF arm runs in), and EXCLUDING ARM_1-only
    decomposition knobs (inert when use_policy_decomposition is False).
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
    """OFF-arm metrics a consumer (P-B) reads back from a reused harshened cell."""
    return [
        "region_vs_mean",
        "region_vs_min",
        "region_vs_max",
        "low_vs_step_frac",
        "fwd_pe_all_mean",
        "fwd_pe_all_var",
        "n_measure_steps",
    ]
