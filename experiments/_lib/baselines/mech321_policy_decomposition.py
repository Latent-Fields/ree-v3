"""Canonical OFF-arm baseline for the ARC-070 / MECH-321 policy_decomposition
discriminative lineage (V3-EXQ-816 and successors).

WHY THIS MODULE EXISTS
    V3-EXQ-816 is the FIRST behavioural experiment of the policy_decomposition
    lineage: ARM_0 (use_policy_decomposition=False) vs ARM_1 (V_s-drop trigger
    ON). A LATER experiment (the R5 bottleneck-state trigger arm, ARM_2 -- see
    the spawned /implement-substrate chip 2026-07-25) will compare the SAME
    ARM_0 baseline against a bottleneck-trigger arm. Rather than have that
    successor re-train ARM_0's cells from scratch, 816 emits its ARM_0 cells
    reuse-ELIGIBLE (include_driver_script_in_hash=False) and both drivers
    construct the OFF arm from THIS module, so the arm fingerprint matches BY
    CONSTRUCTION (the under-declared-slice false-hit trap the arm_reuse plan
    section 7b warns about is avoided because the config slice has a single
    source of truth here).

    Mint run_id for the lineage: recorded in WORKSPACE_STATE.md / the 816 queue
    note once 816 lands. A consumer cites it via reuse_baseline_from and calls
    experiments._lib.arm_reuse.try_reuse_cell(config_slice=off_path_config_slice(),
    include_driver_script_in_hash=False, cite_run_id=<816 run_id>, ...).

WHAT THE OFF ARM IS
    The full waking behavioural loop on CausalGridWorldV2 with the required
    MECH-321 substrate stack ACTIVE (event_segmenter + policy_chunking + chunk
    injection + per_stream_vs) but use_policy_decomposition=False -- i.e. chunks
    form and are injected into the CEM pool and can be committed and executed
    blindly, paying prediction-failure cost at execution, with NO decomposition
    ever withholding/replacing them. This is exactly the ARM_0 arm of MECH-321's
    functional_restatement discriminative-pair design.

The config slice declares ONLY what the OFF computation reads: env params +
schedule + the substrate-operating flags every arm runs + use_policy_decomposition
=False. It deliberately EXCLUDES ARM_1/ARM_2-only knobs (decomposition_vs_threshold,
decomposition_depth_cap, trigger mode) because they are inert when
use_policy_decomposition is False -- including them would make the OFF fingerprint
spuriously depend on a treatment-arm parameter and refuse a legitimate reuse.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Environment shape (predictability heterogeneity: hazards + native env drift so
# some regions show fast tick-to-tick latent change -> low region V_s, the R1
# trigger condition; see V3-EXQ-816 module docstring).
ENV_SIZE = 8
ENV_NUM_HAZARDS = 3
ENV_NUM_RESOURCES = 6
ENV_USE_PROXY_FIELDS = True

# Schedule (one seed x arm cell): warmup lets world_forward become
# action-conditional (execution-time forward PE meaningful) and lets per_stream_vs
# populate; the measurement window is where DVs are collected.
WARMUP_EPISODES = 40
MEASURE_EPISODES = 20
STEPS_PER_EPISODE = 24

ALPHA_WORLD = 0.9  # SD-008: z_world fidelity (chunks must ground on a faithful z_world)

# Seeded crystallised chunk (identical in both arms) -- insurance that a chunk
# candidate is always in the CEM pool for the mechanism to act on, independent of
# whether ARC-071 online formation fires within the horizon (formation readiness
# is the separate, already-answered V3-EXQ-810 question). depth 1 = the common
# ARC-071 formation shape.
SEEDED_CHUNK_SEQUENCE = (0, 1, 2)


def total_episodes() -> int:
    """Denominator M for the runner progress contract (warmup + measure)."""
    return WARMUP_EPISODES + MEASURE_EPISODES


def env_kwargs(seed: int) -> Dict[str, Any]:
    return {
        "size": ENV_SIZE,
        "num_hazards": ENV_NUM_HAZARDS,
        "num_resources": ENV_NUM_RESOURCES,
        "use_proxy_fields": ENV_USE_PROXY_FIELDS,
        "seed": seed,
    }


def substrate_stack_flags() -> Dict[str, Any]:
    """The MECH-321 required substrate stack -- constant across ALL arms.

    Without these there is nothing in the CEM pool for MECH-321 to act on, and
    without use_per_stream_vs the R1 V_s-drop half of the OR trigger is inert
    (region V_s falls back to a constant 1.0).
    """
    return {
        "use_event_segmenter": True,
        "use_policy_chunking": True,
        "use_chunk_proposal_injection": True,
        "use_per_stream_vs": True,
    }


def off_arm_flags() -> Dict[str, Any]:
    """ARM_0: substrate stack ON, decomposition OFF. Nothing else."""
    flags = substrate_stack_flags()
    flags["use_policy_decomposition"] = False
    return flags


def off_path_config_slice() -> Dict[str, Any]:
    """The canonical OFF-arm fingerprint config slice (producer AND consumer).

    MUST be identical byte-for-byte between V3-EXQ-816's ARM_0 emit and any
    successor's try_reuse_cell(...) call, or reuse is (correctly) refused.
    Declares only what the OFF computation reads.
    """
    slice_: Dict[str, Any] = {
        "env": {
            "size": ENV_SIZE,
            "num_hazards": ENV_NUM_HAZARDS,
            "num_resources": ENV_NUM_RESOURCES,
            "use_proxy_fields": ENV_USE_PROXY_FIELDS,
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
    """The OFF-arm metrics a consumer (ARM_2 run) reads back from a reused cell.

    try_reuse_cell refuses unless set(needed) is a subset of the cached cell's
    keys, so keep this in sync with the row a consumer compares against.
    """
    return [
        "fwd_pe_lowvs_mean",
        "fwd_pe_all_mean",
        "delib_cost_lowvs_mean",
        "delib_cost_all_mean",
        "net_harm_per_step",
        "low_vs_step_frac",
        "n_measure_steps",
    ]
