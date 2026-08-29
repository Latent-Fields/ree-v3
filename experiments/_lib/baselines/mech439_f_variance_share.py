"""Canonical OFF/baseline arm for the MECH-439 F-variance-share lineage (V3-EXQ-936 ->).

Arm-reuse Phase 0 (instrument-only). Design plan:
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md (sections 2, 7b).

WHAT THIS MODULE IS
-------------------
The single source of truth for the OFF / baseline arm of the MECH-439
"does a converting selection-face lever REDUCE F's committed-selection variance
share?" lineage. The OFF arm is the selection-face constitution fully disabled
(use_f_eligibility_demotion=False, use_go_nogo_constitution=False) on the
GAP-A-ready conversion stack that V3-EXQ-689d/689i established.

It is factored here -- rather than declared inline in the V3-EXQ-936 driver --
so that its identity is **content-hashed**: this file is matched by the
arm-fingerprint substrate glob `experiments/_lib/**/*.py`, so any edit to it
correctly flips the substrate hash and REFUSES a stale reuse.

THE CONTRACT (the part that must be exactly right)
--------------------------------------------------
A future 936a / 936b that wants to *reuse* a baseline minted from this module
MUST construct its OFF arm by importing `off_path_config_slice()` / `make_env()`
/ `make_agent_kwargs()` from here -- NOT by re-deriving the OFF path inline.
The OFF arm's computation is the conjunction of:
  * ENV_KWARGS                    -- identical to V3-EXQ-689d / 689i (GAP-A reef
                                     bipartite foraging substrate)
  * the training schedule         -- P0_WARMUP_EPISODES, STEPS_PER_EPISODE
  * the substrate-operating config -- CONFIG_FLAGS below, which fire on EVERY arm
  * the OFF arm's own flags       -- use_f_eligibility_demotion=False,
                                     use_go_nogo_constitution=False
It does **NOT** depend on the ON arms' settings, the acceptance thresholds, or
the arm labels -- so none of those appear in `off_path_config_slice()`. That
narrowed slice (not whole-config) is what makes cross-iteration reuse HIT
(plan 7b constraint 2).

WHY THE SLICE OMITS `e3_score_decomp_enabled`
---------------------------------------------
The V3-EXQ-936 driver sets `agent.e3.e3_score_decomp_enabled = True` to collect
the per-candidate score decomposition. That flag is DIAGNOSTICS-ONLY and gated:
`e3_selector.py` populates `last_score_decomp` / `last_channel_terms` inside an
`if self.e3_score_decomp_enabled:` block whose own comment records that the
selection path is "bit-identical when OFF". It therefore does not enter the OFF
arm's *computation* and is deliberately not part of the fingerprint slice --
including it would refuse reuse against any sibling that happened not to
instrument, for no correctness gain.

SEED 44 IS DELIBERATELY ABSENT
------------------------------
`reef_enabled=True` puts this on the reef config family, which has a confirmed
per-seed instability on seed 44 (early episode death ~step 40; EXQ-539-540 and
V3-EXQ-538a autopsies). V3-EXQ-689i carried seed 44; this lineage substitutes 46
rather than debug the same truncation a third time.

ASCII-only output (repo rule).
"""

from __future__ import annotations

from typing import Any, Dict

from ree_core.environment.causal_grid_world import CausalGridWorldV2

__all__ = [
    "ENV_KWARGS",
    "CONFIG_FLAGS",
    "SEEDS",
    "P0_WARMUP_EPISODES",
    "STEPS_PER_EPISODE",
    "SD056_WEIGHT",
    "E2_CONTRASTIVE_LR",
    "CONTRASTIVE_BATCH_K",
    "TRANSITION_BUFFER_MAX",
    "E2_TRAIN_EVERY_K_TICKS",
    "MAX_GRAD_NORM",
    "sd056_training_slice",
    "make_env",
    "make_agent_kwargs",
    "off_path_config_slice",
]

# --- Environment: GAP-A reef-bipartite foraging substrate (689d / 689i) -------
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

# Seed 44 excluded on purpose -- see the module docstring.
SEEDS = [42, 43, 45, 46]

P0_WARMUP_EPISODES = 60
STEPS_PER_EPISODE = 200

# --- SD-056 online contrastive training (fires on EVERY arm, OFF included) ---
# These live HERE rather than in the driver because they are part of the OFF
# arm's own COMPUTATION: the OFF cell trains e2 with exactly this scheme, so a
# consumer that reused an OFF cell minted under different values would be
# reading readouts computed under a different training regime. Declaring them
# in `off_path_config_slice()` is what makes that a cache MISS (wasted compute)
# rather than a false cache HIT (a corrupted conclusion) -- the asymmetry in
# arm_reuse_fingerprint_plan.md section 7b.
SD056_WEIGHT = 0.5
E2_CONTRASTIVE_LR = 1e-4
CONTRASTIVE_BATCH_K = 8
TRANSITION_BUFFER_MAX = 2000
E2_TRAIN_EVERY_K_TICKS = 4
MAX_GRAD_NORM = 1.0

# --- Substrate-operating config that fires on EVERY arm ----------------------
# Mirrors V3-EXQ-689i's _make_agent minus the arm-specific MECH-448/449 flags
# and minus the MECH-439 near-tie parametric levers (Factor A / Factor B), which
# the confirmed 689a autopsy retired as an exhausted family and which this
# lineage therefore holds OFF on every arm.
CONFIG_FLAGS: Dict[str, Any] = dict(
    self_dim=32,
    world_dim=32,
    alpha_world=0.9,
    alpha_self=0.3,
    use_harm_stream=True,
    z_harm_dim=32,
    use_affective_harm_stream=True,
    z_harm_a_dim=16,
    harm_history_len=10,
    z_goal_enabled=True,
    goal_weight=0.5,
    drive_weight=2.0,
    e1_goal_conditioned=True,
    use_resource_proximity_head=True,
    resource_proximity_weight=0.5,
    benefit_eval_enabled=True,
    benefit_weight=1.0,
    # ARC-065 SP-CEM (Layer A) -- the GAP-A divergent eligible set
    use_support_preserving_cem=True,
    support_preserving_stratified_elites=True,
    support_preserving_ao_std_floor=0.2,
    support_preserving_min_first_action_classes=2,
    candidate_summary_source="e2_world_forward",
    # Shared E3-side modulatory bias channels (-> _modulatory_accum non-None).
    # These are what F must be measured AGAINST: if they carry no variance the
    # F-share question is starved rather than answered, which is exactly what
    # the P3 readiness precondition in the driver guards.
    use_lateral_pfc_analog=True,
    use_mech295_liking_bridge=True,
    use_modulatory_channel_routing=True,
    modulatory_channel_route_source="cand_world_summary",
    modulatory_channel_route_weight=1.0,
    use_modulatory_selection_authority=True,
    use_modulatory_shortlist_then_modulate=True,
    # MECH-439 near-tie parametric family (Factor A / Factor B) OFF on every arm.
    modulatory_shortlist_conflict_graded=False,
    use_gap_scaled_commit_temperature=False,
    # Other policy-layer regulators + CRF stack OFF (689i parity).
    use_structured_curiosity=False,
    use_e3_score_diversity=False,
    use_noise_floor=False,
    use_tonic_vigor=False,
    use_dacc=False,
    use_ofc_analog=False,
    use_gated_policy=False,
    use_candidate_rule_field=False,
    # BUG FIX (V3-EXQ-936a, 2026-08-29): SD-056 online contrastive training
    # (above) fires on every arm but this lineage never enabled the E2
    # rollout-norm clamp that V3-EXQ-689i used alongside it ("689i parity" was
    # claimed in the module docstring but not actually declared here). Without
    # the clamp, world_forward_contrastive_loss's rollout magnitudes overflow
    # to 1e16-1e18 (confirmed V3-EXQ-569e, V3-EXQ-936; failure_autopsy
    # V3-EXQ-936_2026-08-18, severity corrupting), saturating f_variance_share
    # to 1.0 and destroying the load-bearing C2 criterion. The clamp mechanism
    # itself is BUILT and VALIDATED (substrate_queue.json SD-056,
    # status=implemented) -- this only turns it on for this lineage, mirroring
    # 689i's own per-experiment enable. Does NOT touch the REEConfig class
    # default (still False globally; the autopsy explicitly vetoed a default
    # flip -- see its recommended_substrate_queue_entry).
    e2_rollout_output_norm_clamp_enabled=True,
    e2_rollout_output_norm_clamp_ratio=2.0,
)

# --- The OFF arm's own flags -------------------------------------------------
OFF_ARM_FLAGS: Dict[str, Any] = dict(
    use_f_eligibility_demotion=False,
    use_go_nogo_constitution=False,
)


def make_env(seed: int) -> CausalGridWorldV2:
    """Construct the lineage's canonical environment at `seed`."""
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def make_agent_kwargs(env: CausalGridWorldV2, arm_flags: Dict[str, Any]) -> Dict[str, Any]:
    """REEConfig.from_dims(**kwargs) for this lineage, with `arm_flags` applied last.

    Every arm -- OFF and ON alike -- is built through this one function, so the
    only thing that can differ between arms is `arm_flags`.
    """
    kwargs: Dict[str, Any] = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    kwargs.update(CONFIG_FLAGS)
    kwargs.update(arm_flags)
    return kwargs


def sd056_training_slice() -> Dict[str, Any]:
    """The SD-056 online-training constants, as a fingerprint sub-slice.

    Shared by the OFF slice and by every treatment arm's slice, so all cells
    declare the training scheme identically.
    """
    return {
        "sd056_weight": SD056_WEIGHT,
        "e2_contrastive_lr": E2_CONTRASTIVE_LR,
        "contrastive_batch_k": CONTRASTIVE_BATCH_K,
        "transition_buffer_max": TRANSITION_BUFFER_MAX,
        "e2_train_every_k_ticks": E2_TRAIN_EVERY_K_TICKS,
        "max_grad_norm": MAX_GRAD_NORM,
    }


def off_path_config_slice() -> Dict[str, Any]:
    """The fingerprint slice for the OFF arm.

    Declares ONLY what the OFF computation reads: env kwargs, the schedule, the
    SD-056 training constants, the substrate-operating config every arm runs,
    and the OFF arm's own flags. Never the ON-arm settings, acceptance
    thresholds, or arm labels.
    """
    return {
        "env_kwargs": dict(ENV_KWARGS),
        "schedule": {
            "p0_warmup_episodes": P0_WARMUP_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
        },
        "sd056_training": sd056_training_slice(),
        "config_flags": dict(CONFIG_FLAGS),
        "off_arm_flags": dict(OFF_ARM_FLAGS),
    }
