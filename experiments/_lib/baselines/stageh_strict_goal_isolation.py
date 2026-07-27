"""
Canonical baseline module for the `stageh_strict_goal_isolation` lineage
(V3-EXQ-833 and successors).

WHAT THIS LINEAGE ASKS. `_set_goal_pipeline_frozen(agent, frozen=True)` silences
the goal WRITE paths (MECH-295 liking bridge, MECH-307 conjunction) and nothing
else; the two goal READ paths -- the E3 `goal_weight * goal_proximity`
subtraction and E1 goal-conditioning -- are gated independently and stay live
through Stage-0b / P0 / Stage-H. The 2026-07-27 triage
(REE_assembly/evidence/planning/scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md)
measured the consequence on the 460c dry-run config: in Stage-H the goal term's
candidate spread is 0.65x the harm term's, and removing it counterfactually
moves the cost-argmin candidate on 38% of ticks. That is a SELECTION
counterfactual. Whether it changes what avoidance the agent LEARNS is the
lineage's question.

THE OFF / BASELINE ARM (`ARM_LEGACY`) is the stock scaffold curriculum with
`scaffold_strict_goal_isolation=False` -- i.e. bit-identical to every landed
scaffold run. The ON arm flips exactly that one knob. Everything else in this
module is shared by both arms BY CONSTRUCTION, which is what makes the
single-variable claim checkable rather than asserted.

WHY THIS CONFIG. It is V3-EXQ-603q's `ARM_BASE_IA_ONLY` cell, truncated after
Stage-H (P1 / P2 are dropped -- they run after the frozen stages and cannot
inform a Stage-H DV, and their 65 episodes/cell buy seeds instead). 603q is the
validated anchor: on this exact substrate it recorded Stage-0 z_goal peaks of
0.497 / 0.430 / 0.369, a discriminative harm landscape (harm_eval_range 0.398 /
0.129 / 0.185 against the 603i flat-landscape reference of ~0.002), and a base
Stage-H mean survival of 37.7 steps out of 200 -- unsaturated in BOTH
directions, which a survival DV needs.

The four settings the lineage's question depends on, and why each is load-bearing
rather than incidental:

  * `scaffold_stage0_enabled=True`   -- z_goal must actually FORM. With Stage-0
      off, `goal_state.is_active()` is False, the E3 goal term is skipped
      outright, and the ON/OFF comparison is vacuous by construction. These are
      the triage's five inert exceptions (603d/603e/621/621a/625c).
  * `scaffold_hazard_stage_enabled=True` -- Stage-H is the DV.
  * `scaffold_feed_harm_stream=True` -- populates z_harm / z_harm_a.
  * `scaffold_train_harm_pathway=True` (+ the 603q stabilization pair) -- without
      it the E3 harm cost is a random constant (V3-EXQ-603i measured the range at
      [0.522, 0.524]) and survival is unreachable at any budget. Comparing two
      arms against a noise harm landscape would answer nothing.

REUSE. `off_path_config_slice()` is the content address of the OFF path. Mint
and consume it with `include_driver_script_in_hash=False` so a successor's
different driver can match this lineage's banked ARM_LEGACY cells (see
REE_assembly/evidence/planning/arm_reuse_fingerprint_plan.md sections 7b/9.7).
"""

from __future__ import annotations

from typing import Any, Dict

from scaffolded_sd054_onboarding import ScaffoldedSD054OnboardingConfig
from ree_core.utils.config import REEConfig

# --- Lineage identity -------------------------------------------------------
LINEAGE = "stageh_strict_goal_isolation"

# --- Encoder / latent dims (mirror 603q exactly) ----------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0
ALPHA_WORLD = 0.9  # z_world fidelity; the 0.3 default is a known SD-008 root cause

# --- Curriculum budgets -----------------------------------------------------
# Truncated after Stage-H: 603q's P1 (50) + P2 (15) are dropped. They run AFTER
# the frozen stages, so they cannot inform a Stage-H DV; their 65 episodes/cell
# are spent on seeds instead, which is where this lineage's variance lives.
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
TRAIN_STEPS = 200
P0_NUM_HAZARDS = 1
EPISODES_PER_CELL = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET  # 170

# Dry-run (smoke) budgets.
DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_STEPS = 2, 2, 5, 5, 30
DRY_EPISODES_PER_CELL = DRY_STAGE0 + DRY_STAGE0B + DRY_P0 + DRY_HAZARD  # 14

# --- Stage-H regime (603q's amend-validated anchor) -------------------------
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 603q) ------
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal ---------------------------
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# --- Harm-pathway training (603k) + stabilization (603q amend) --------------
HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4   # decoupled (lower) encoder LR
HARM_PATHWAY_WARMUP_STEPS = 250  # linear LR warmup over the first N steps


def build_scaffold_cfg(dry_run: bool,
                       strict_goal_isolation: bool) -> ScaffoldedSD054OnboardingConfig:
    """The scheduler config shared by BOTH arms.

    `strict_goal_isolation` is the ONLY parameter that differs between arms. It
    threads to `_set_goal_pipeline_frozen(..., strict=...)` at every freeze site,
    so with it True the three frozen stages (Stage-0b, P0, Stage-H) additionally
    zero `agent.e3.config.goal_weight` and clear `goal.e1_goal_conditioned`,
    restoring the saved priors on unfreeze. Default False is bit-identical to
    every landed scaffold run.
    """
    if dry_run:
        stage0, stage0b, p0, hazard, steps = (
            DRY_STAGE0, DRY_STAGE0B, DRY_P0, DRY_HAZARD, DRY_STEPS)
    else:
        stage0, stage0b, p0, hazard, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET, TRAIN_STEPS)

    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        # ===== THE SINGLE MANIPULATED VARIABLE =====
        scaffold_strict_goal_isolation=bool(strict_goal_isolation),
        # ===== everything below is IDENTICAL across arms =====
        # Stage-0 forced-feed nursery: z_goal MUST form or the goal term is inert.
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # Stage-H -- the DV.
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver.
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # Harm stream + harm-pathway training: without these the E3 harm cost is
        # a random constant (603i) and survival is unreachable at any budget.
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
    )
    if steps < 75:
        # Smoke only: the 75-step survival gate is unreachable in a 30-step episode.
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def build_agent_config(env) -> REEConfig:
    """The REEConfig shared by BOTH arms -- 603q's ARM_BASE_IA_ONLY exactly.

    The escape-affordance bridge (SD-059 / MECH-358) is OFF, as it is on 603q's
    base arm: this lineage isolates the goal term, not the bridge.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA_WORLD,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # Escape-affordance bridge OFF (603q ARM_BASE_IA_ONLY).
        use_escape_affordance_bridge=False,
        use_escape_relief_credit=False,
        use_escape_safety_credit=False,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def off_path_config_slice(dry_run: bool = False) -> Dict[str, Any]:
    """Content address of the OFF (`ARM_LEGACY`) path.

    Declares ONLY what the OFF computation reads: env/stage regime, schedule, the
    substrate-operating config both arms run, and the OFF arm's own value of the
    manipulated knob. It must NOT carry acceptance thresholds or ON-arm gains --
    those do not change the computation, and folding them in would refuse a
    legitimate reuse on every threshold tweak.
    """
    return {
        "lineage": LINEAGE,
        "scaffold_strict_goal_isolation": False,   # the OFF arm's own value
        "world_dim": WORLD_DIM,
        "alpha_world": ALPHA_WORLD,
        "drive_weight": DRIVE_WEIGHT,
        "harm_dims": [HARM_A_DIM, HARM_OBS_A_DIM, HARM_HISTORY_LEN],
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET,
                    HAZARD_STAGE_BUDGET, TRAIN_STEPS, P0_NUM_HAZARDS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS,
                         HAZARD_STAGE_STABILITY_WINDOW],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR,
                    N_RESOURCE_TYPES, CUE_RECALL_GAIN],
        "avoidance": [AVOIDANCE_SCAFFOLD_FLOOR_START, AVOIDANCE_SCAFFOLD_FLOOR_END,
                      AVOIDANCE_THREAT_REF, PAG_THETA_FREEZE,
                      PAG_DURATION_INPUT_THRESHOLD],
        "harm_pathway": [HARM_PATHWAY_LR, HARM_PATHWAY_ENCODER_LR,
                         HARM_PATHWAY_WARMUP_STEPS],
        "escape_bridge": False,
        "truncated_after_stage_h": True,
        "dry_run": bool(dry_run),
    }


def arm_config_slice(strict_goal_isolation: bool, dry_run: bool = False) -> Dict[str, Any]:
    """Content address for EITHER arm: the OFF slice with the knob overridden."""
    slice_ = off_path_config_slice(dry_run=dry_run)
    slice_["scaffold_strict_goal_isolation"] = bool(strict_goal_isolation)
    return slice_
