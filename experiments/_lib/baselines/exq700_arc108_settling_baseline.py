"""Canonical reusable-arm config slice for the V3-EXQ-700 ARC-108 sec-7 settling lineage.

Arm-reuse Phase 0 (instrument-only). Design plan:
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (sections 2, 7b, 9).

WHAT THIS MODULE IS
-------------------
The single source of truth for the per-arm config slice of the four REUSABLE arms
of the V3-EXQ-700 ARC-108 sec-7 learned-gating SETTLING lineage
(V3-EXQ-700b -> 700c -> ...). Those four arms are:
  A0_ENVELOPE_ONLY     -- envelope-only control (no learning, no noise)
  A2_SETTLING_SIGNED   -- learned W_lat settling, signed RPE (the converting lever)
  A3_BOTH_SIGNED       -- learned w_chan + learned W_lat settling, signed RPE
  C3_SETTLING_UNSIGNED -- learned W_lat settling, UNSIGNED RPE (B5 ablation)
The fifth arm (ARM_NOISE) is the CHANGED arm in each lettered iteration (700b used a
MECH-313 policy-temperature null; 700c uses a same-layer settling-field null), so it is
NEVER reusable and is NOT covered here.

This module is matched by the arm-fingerprint substrate glob
`experiments/_lib/**/*.py`, so any change to it correctly flips the substrate hash and
*refuses* a stale reuse (a false miss is free; a false hit corrupts science).

THE CONTRACT (the part that must be exactly right)
--------------------------------------------------
`arm_config_slice(arm, p0, p1, p2, steps)` returns the config_slice dict that BOTH the
700c consumer AND the V3-EXQ-700c-mint baseline mint pass to compute_arm_fingerprint for
a given reusable arm. Because BOTH sides import this one function, their fingerprint
config_slices match by construction. The slice declares everything the OFF/settling
computation reads:
  * the arm's own (lcg_on, settle_on, noise_on, rpe_mode) -- noise_on is always False on
    the reusable arms;
  * the full matched arithmetic-envelope + diversity-stack constants A0/A2/A3/C3 all
    share (the landed envelope: MECH-448 demotion + adaptive floor, MECH-449 go/no-go,
    modulatory authority + routing + top_k shortlist, MECH-341 stratified, SD-056 levers,
    matured CRF pool, the trained lateral_pfc bias head, use_dacc, the learned-gating /
    settling knobs);
  * env_kwargs, sd056_weight, lr_lpfc_bias;
  * the P0/P1/P2/steps schedule.
It MUST NOT include any ON-only / ARM_NOISE-only field (FIELD_NOISE_*; the
field-noise frozen random W_lat is a property of the changed arm, never these four).
It mirrors the keys 700b's compute_arm_fingerprint passed (700b lines ~1136-1167) MINUS
the noise_floor_alpha key (irrelevant to the noise-off settling arms -- noise_floor is
inert on every reusable arm), so a 700b->700c OFF/settling arm matches.

REUSE MACHINE-CLASS
-------------------
A ree-cloud worker class (the 700b run + the mint run on ree-cloud-4). Since 2026-07-19 the
tag also carries the TORCH BUILD -- currently `linux-x86_64-py3.10-torch2.5.1+cu121`. The
authority is `machine_class()` in experiments/_lib/arm_fingerprint.py; treat any tag written
here as indicative only. A Mac-run iteration cannot match a cloud-minted baseline
(machine_class enters the fingerprint), so reuse is intrinsically cloud-scoped. A fleet
torch upgrade now retires a banked baseline as an OS or python change always did, and any
baseline minted BEFORE that hard cut -- including the pre-cut 700b arms -- is DEAD: it
cannot be migrated and must be re-minted under the new class (plan section 12).

MINT RUN_ID
-----------
Recorded by the parent session once the V3-EXQ-700c-mint completes (then cited via
REUSE_BASELINE_FROM in the 700c consumer). Until then REUSE_BASELINE_FROM=None and every
arm runs fresh.

ASCII-only output (repo rule). Stdlib + numbers only (importable without ree_core).
"""

from __future__ import annotations

from typing import Any, Dict

LINEAGE = "v3_exq_700_arc108_sec7_learned_gating_settling"

# The four arms a future mint may reuse. ARM_NOISE is excluded (it is the changed arm).
REUSABLE_ARM_IDS = (
    "A0_ENVELOPE_ONLY",
    "A2_SETTLING_SIGNED",
    "A3_BOTH_SIGNED",
    "C3_SETTLING_UNSIGNED",
)

# --- Matched arithmetic-envelope + diversity-stack constants (identical on ALL arms;
# the landed envelope -- mirror of the 700b _make_agent / fingerprint slice values). ---
ENV_KWARGS: Dict[str, Any] = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)

# Matched-stack lever constants (identical on ALL arms; the landed envelope). These
# mirror the 700b module constants exactly so the slice and the actual agent build cannot
# drift apart -- the 700c consumer + the mint both build their agents from the same 700b
# constants and declare them here.
MATCHED_ENVELOPE: Dict[str, Any] = dict(
    use_modulatory_selection_authority=True,
    modulatory_authority_gain=2.0,
    modulatory_authority_normalize_basis="std",
    use_modulatory_channel_routing=True,
    modulatory_channel_route_source="cand_world_summary",
    modulatory_channel_route_weight=1.0,
    modulatory_route_min_range_floor=1e-6,
    use_modulatory_shortlist_then_modulate=True,
    modulatory_shortlist_mode="top_k",
    modulatory_shortlist_k=3,
    use_f_eligibility_demotion=True,
    f_eligibility_envelope_floor=0.30,
    f_eligibility_dn_sigma=0.0,
    use_f_eligibility_adaptive_floor=True,
    f_eligibility_adaptive_mean_factor=1.0,
    use_go_nogo_constitution=True,
    use_dacc=True,
    gng_perseveration_floor=0.5,
    gng_safety_floor=0.5,
    gng_protect_min_eligible=1,
    mech341_entropy_bias_scale=2.0,
    vs_snapshot_refresh_threshold=0.5,
    vs_e1_threshold=0.4,
    use_candidate_rule_field=True,
    # ARC-108 JOB-1 step-1 learned-gating knobs (matched when armed).
    lcg_eta=0.01,
    lcg_elig_decay=0.9,
    lcg_value_baseline_beta=0.05,
    lcg_asym_potentiation=1.0,
    lcg_asym_depression=0.5,
    # MECH-450 settling knobs (matched when armed).
    settling_rounds=3,
    settling_temperature=1.0,
    settling_eta=0.01,
    settling_elig_decay=0.9,
    settling_n_action_classes=8,
)

# SD-056 online e2 training (mirror 700b).
SD056_WEIGHT = 0.05
# P1 bias-head REINFORCE training (mirror 700b).
LR_LPFC_BIAS = 5e-4


def arm_config_slice(
    arm: Dict[str, Any],
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """The declared config slice for ONE reusable arm's fingerprint.

    Returns the SAME dict the 700c consumer and the V3-EXQ-700c-mint both pass to
    compute_arm_fingerprint for ``arm``. Declares the per-arm (lcg_on/settle_on/noise_on/
    rpe_mode) plus the full matched envelope + env + sd056/lpfc + schedule -- everything
    the OFF/settling computation reads. EXCLUDES any ARM_NOISE-only / ON-only field
    (FIELD_NOISE_*; noise_floor_alpha), so a 700b->700c OFF/settling arm matches.

    The four reusable arms always have noise_on=False (the changed arm is ARM_NOISE).
    """
    return {
        "lineage": LINEAGE,
        # The per-arm swept variables.
        "arm_id": str(arm["arm_id"]),
        "lcg_on": bool(arm["lcg_on"]),
        "settle_on": bool(arm["settle_on"]),
        "noise_on": bool(arm["noise_on"]),
        "rpe_mode": str(arm.get("rpe_mode", "signed")),
        # The matched arithmetic envelope + diversity stack (identical on all arms).
        "matched_envelope": dict(MATCHED_ENVELOPE),
        # Env + online-training params.
        "env_kwargs": dict(ENV_KWARGS),
        "sd056_weight": float(SD056_WEIGHT),
        "lr_lpfc_bias": float(LR_LPFC_BIAS),
        # Schedule.
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "p2_episodes": int(p2_episodes),
        "steps_per_episode": int(steps_per_episode),
    }


def off_path_config_slice(
    p0_episodes: int = 100,
    p1_episodes: int = 50,
    p2_episodes: int = 100,
    steps_per_episode: int = 200,
) -> Dict[str, Any]:
    """Convenience: the A0_ENVELOPE_ONLY arm's slice (skill convention).

    The A0 arm is the canonical envelope-only OFF arm of the lineage. Mirrors the
    700b A0 arm dict (lcg_on=False, settle_on=False, noise_on=False, rpe_mode=signed).
    """
    a0 = {
        "arm_id": "A0_ENVELOPE_ONLY",
        "lcg_on": False,
        "settle_on": False,
        "noise_on": False,
        "rpe_mode": "signed",
    }
    return arm_config_slice(a0, p0_episodes, p1_episodes, p2_episodes, steps_per_episode)


__all__ = [
    "LINEAGE",
    "REUSABLE_ARM_IDS",
    "ENV_KWARGS",
    "MATCHED_ENVELOPE",
    "SD056_WEIGHT",
    "LR_LPFC_BIAS",
    "arm_config_slice",
    "off_path_config_slice",
]
