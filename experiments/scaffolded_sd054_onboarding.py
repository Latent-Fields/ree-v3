"""
scaffolded_sd054_onboarding.py -- substrate-level goal-pipeline onboarding harness.

Lands the scaffolded_sd054_onboarding substrate (substrate_queue.json entry,
status pending_implementation; IGW-20260531-029). Plan-of-record:
REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
(2026-05-29).

Why this exists
---------------
The 591 substrate-uniform z_goal-zero family (V3-EXQ-540 series / 590a / 591 /
603 series) shows z_goal collapsing to ~1e-7 across all arms under random-policy
training in the target reef+bipartite+hazard_food_attraction=0.7 env. Code trace
(failure_autopsy_V3-EXQ-591_2026-05-27.md + z_goal_collapse_triage_2026-05-31.md):
GoalState.update() pulls z_goal toward z_world only when

    benefit_exposure * z_goal_seeding_gain * (1 + drive_weight*drive_trace) > benefit_threshold

At the V3 default config (drive_floor=0.0, benefit_threshold=0.1) the gate is
almost never cleared in a random-init agent that dies in the target env before
drive accumulates. The 490 cohort sidesteps this by running with gap4
(drive_floor=0.9) ON, but that does not match the default-config question
prereq (2) of behavioral_diversity_isolation:GAP-C asks.

What this harness does
----------------------
Three-phase scheduler matching the substrate-design memo:

    P0  scaffolded SD-054 env (refuge-band spawn, hazard_food_attraction=0.0,
        proximity_harm_scale=0.05) with goal-pipeline writes FROZEN. Encoder +
        E2 + E3 warm up on the SD-054 spatial structure without the goal
        pipeline gating its own training data.

    P1  spawn admissibility narrows back to the midline band. Linear anneal
        across the P1 window:
          hazard_food_attraction 0.0 -> 0.7
          proximity_harm_scale   0.05 -> 0.1
          mech295_min_drive_to_fire        1.0 -> 0.01
          mech307_conjunction_z_beta_threshold 0.6 -> 0.3
        Goal-pipeline flags hard-on. End-of-P1 survival gate: median episode
        length over the last P1_STABILITY_WINDOW (10) episodes must clear
        scaffold_p1_survival_gate_steps (75) or the cell routes to
        non_contributory (Fix D from V3-EXQ-603c, retained).

    P2  full target env config, policy frozen. N_p2 episodes for measurement
        (z_goal_norm_peak, approach_commit_rate, bridge_cue_fires,
        dacc_bias_nonzero_steps).

NOT a ree_core substrate scheduler. Lives in experiments/ alongside
infant_curriculum.py and committed_mode_curriculum.py as a pure training-loop
helper that experiment scripts import. ree_core/ is untouched; the ONE env-side
change is the new reef_bipartite_agent_spawn_in_reef_half kwarg on
CausalGridWorldV2 (added 2026-05-31).

Master switch: use_scaffolded_sd054_onboarding_scheduler (default False on the
ScaffoldedSD054OnboardingConfig dataclass). Bit-identical OFF: no behaviour
change is reachable unless an experiment script explicitly constructs a
ScaffoldedSD054OnboardingScheduler with master=True.
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ScaffoldedSD054OnboardingConfig:
    """
    Configuration for the scaffolded SD-054 onboarding scheduler.

    All knobs default to no-op-friendly values that match the substrate-design
    memo (sd_054_scaffolded_onboarding_substrate_design.md, 2026-05-29).
    Master switch defaults False; scheduler is inert unless explicitly
    constructed with master=True.
    """

    # Master switch. Default False; bit-identical OFF.
    use_scaffolded_sd054_onboarding_scheduler: bool = False

    # Phase budgets. Memo suggests 30/30/30 as starting points; calibration
    # is owned by the V3-EXQ-620 substrate-readiness validation.
    scaffold_p0_episode_budget: int = 30
    scaffold_p1_episode_budget: int = 30
    scaffold_p2_episode_budget: int = 30
    scaffold_steps_per_episode: int = 200

    # P0 env config (relaxed; agent spawns inside reef refuge band, goal
    # pipeline FROZEN). Sub-target proximity_harm_scale; reduced hazard /
    # resource density.
    scaffold_p0_proximity_harm_scale: float = 0.05
    scaffold_p0_num_hazards: int = 2
    scaffold_p0_num_resources: int = 3

    # P1 anneal endpoints. hazard_food_attraction + proximity_harm_scale ramp
    # up from P0 values to target-env values across the P1 window.
    scaffold_p1_anneal_hazard_food_attraction_min: float = 0.0
    scaffold_p1_anneal_hazard_food_attraction_max: float = 0.7
    scaffold_p1_anneal_proximity_harm_scale_min: float = 0.05
    scaffold_p1_anneal_proximity_harm_scale_max: float = 0.1

    # P1 anneal of the goal-pipeline gates. mech295_min_drive_to_fire starts
    # at 1.0 (bridge silent at all realistic drives) and ramps down to the
    # 2026-05-12 default 0.01 (bridge fires routinely). z_beta_threshold
    # starts at the legacy pre-recalibration value (0.6) and ramps down to
    # the 2026-05-12 default (0.3).
    scaffold_p1_anneal_mech295_min_drive_to_fire_max: float = 1.0
    scaffold_p1_anneal_mech295_min_drive_to_fire_min: float = 0.01
    scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_max: float = 0.6
    scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_min: float = 0.3

    # P1 survival gate (Fix D from V3-EXQ-603c). Median episode length over
    # the last P1_STABILITY_WINDOW episodes must clear this floor for the
    # cell to be eligible for P2 measurement.
    scaffold_p1_survival_gate_steps: int = 75
    scaffold_p1_stability_window: int = 10

    # P2 target env config. Pinned to the V3-EXQ-603b target env (matches
    # the eventual GAP-4 Tier-1 measurement env).
    scaffold_p2_hazard_food_attraction: float = 0.7
    scaffold_p2_proximity_harm_scale: float = 0.1
    scaffold_p2_num_hazards: int = 4
    scaffold_p2_num_resources: int = 5

    # Common env config (size, reef_bipartite kwargs).
    scaffold_env_size: int = 12
    scaffold_reef_bipartite_axis: str = "horizontal"
    scaffold_reef_bipartite_agent_band_radius: int = 1

    # -------- Stage-0 nursery / forced-benefit feeding (2026-06-03 amend) ------
    # Infant-REE "nursery": a protected forced-feeding warmup that DECOUPLES
    # z_goal formation from survival/foraging competence. The agent is fed a
    # supra-threshold benefit every step regardless of whether it actually
    # contacts a resource, so GoalState.update fires and z_goal forms. This is
    # both a positive control ("the goal stream lights when fed") and a
    # developmental phase that prevents interpreting starvation as
    # representational absence. Routed by failure_autopsy_V3-EXQ-603e-626a-622
    # (2026-06-03): 603e showed z_goal=0 ecologically because 2/3 seeds never
    # reach foraging competence and the hard P2 env starves benefit_exposure.
    # All Stage-0 knobs are additive and default to no-op (scaffold_stage0_enabled
    # False); the V3-EXQ-603f re-issue opts in.
    scaffold_stage0_enabled: bool = False
    scaffold_stage0_episode_budget: int = 20
    # Forced benefit fed to update_z_goal every Stage-0 step. 1.0 is robustly
    # supra-threshold (GoalState gate: benefit * z_goal_seeding_gain *
    # (1 + drive_weight*drive_trace) > benefit_threshold[=0.1]).
    scaffold_stage0_forced_benefit: float = 1.0
    # Forced drive (depleted-infant analog; keeps the SD-012 multiplier high so
    # the gate clears even if z_goal_seeding_gain is small).
    scaffold_stage0_forced_drive: float = 0.9
    # Dense, hazard-free nursery env so the encoder/E2 warm on safe structure.
    scaffold_stage0_num_resources: int = 6
    scaffold_stage0_num_hazards: int = 0
    scaffold_stage0_proximity_harm_scale: float = 0.0
    # Stage-0 acceptance: z_goal_norm_peak must exceed this (per-seed) for the
    # forced-feed positive control to pass.
    scaffold_stage0_z_goal_peak_gate: float = 0.4

    # -------- P2 measurement guard (2026-06-03 amend) -------------------------
    # The hard P2 env (hazard_food_attraction=0.7) can suppress contact and make
    # z_goal=0 uninterpretable. When >= 0, this guard value OVERRIDES
    # scaffold_p2_hazard_food_attraction in the P2 env so the measurement window
    # admits foraging contact. Default -1.0 = no guard (P2 hfa stays 0.7,
    # contract-stable); the V3-EXQ-603f re-issue sets it lower (e.g. 0.3).
    scaffold_p2_hazard_food_attraction_guard: float = -1.0
    # A P2 step counts as a foraging-contact step when the post-step benefit
    # exposure exceeds this floor. Used for the contact-rate guard so a
    # z_goal=0 read is distinguishable from "infant REE was never fed".
    scaffold_p2_contact_benefit_threshold: float = 1e-6

    # -------- Gentler P1 schedule lever (2026-06-03 amend) --------------------
    # Fraction of the P1 window held at anneal_t=0 (full nursery relaxation)
    # before the linear ramp begins -- a staged-withdrawal-of-assistance lever
    # so >=2/3 seeds reach foraging competence before hazards ramp in. Default
    # 0.0 = current pure-linear anneal (contract-stable).
    scaffold_p1_anneal_hold_fraction: float = 0.0

    # -------- Developmental-window / consolidation amend (2026-06-03b) --------
    # Routed by the V3-EXQ-634 design-error review: GoalState.update() ALWAYS
    # decays the persistent z_goal attractor (goal.py:173) and only refreshes it
    # when benefit crosses threshold. P1/P2 call update_z_goal every step incl.
    # unfed steps (decay-only), so the fragile Stage-0 trace is washed out
    # before ecological contact occurs -- 634 then tests "stay goal-active while
    # fed-then-starved under decay-only updates", not "form -> consolidate ->
    # learn contact". This amend protects the Stage-0 trace. All knobs are
    # additive and default to NO-OP (master False); V3-EXQ-634b opts in. With
    # the master OFF the scheduler is bit-identical to the pre-amend 634 path.
    scaffold_developmental_window_enabled: bool = False
    # Stage-0b consolidation: a short PROTECTED window between Stage-0 and P0
    # where update_z_goal is NOT called, so the just-formed z_goal cannot decay
    # (encoder/E2 keep training). Records start/end norm + retention_ratio.
    scaffold_stage0b_enabled: bool = False
    scaffold_stage0b_episode_budget: int = 10
    # Acceptance: retained z_goal must stay >= this fraction of the Stage-0
    # baseline norm across Stage-0b (pre-registered; do not retune to force pass).
    scaffold_stage0b_retention_gate: float = 0.75
    # The KEY fix: when True, P1/P2 only call update_z_goal on a VALIDATED
    # contact step (benefit > scaffold_p2_contact_benefit_threshold); unfed
    # steps are skipped (no decay-only washout). Stage-0 forced-feed is
    # unaffected (forced benefit is always supra-threshold). decay_only is thus
    # reserved for mature/autonomous tests, NOT the nursery gate.
    scaffold_contact_gated_goal_updates: bool = False

    # -------- Seeding-calibration amend (V3-EXQ-634b autopsy, 2026-06-03c) -----
    # The 634b autopsy isolated a benefit-magnitude / threshold mismatch:
    # contact-gating skips only benefit <= scaffold_p2_contact_benefit_threshold
    # (1e-6), but GoalState.update (goal.py:209-224) seeds z_goal only when
    #   effective_benefit = benefit * z_goal_seeding_gain(1.0)
    #                       * (1 + drive_weight(2.0) * drive_trace) > benefit_threshold(0.1).
    # Natural wild benefit (obs_body[11] ~0.03) stays sub-threshold, so the band
    # (1e-6, ~0.1-effective) is NOT skipped yet does NOT seed -- it only applies
    # the unconditional 0.5%/step decay (goal.py:173), DECAYING the consolidated
    # trace during real foraging. Two coupled, no-op-default fixes:
    #
    # (1) DECOUPLED CONTACT-GATING THRESHOLD. The skip/update decision now keys
    #     off a SEPARATE gating threshold so sub-seeding whiffs in the band
    #     (readout_floor, seeding_floor) are PROTECTED (skipped, not decayed)
    #     while the contact-RATE readout (g2 "was the infant fed at all") keeps
    #     using scaffold_p2_contact_benefit_threshold. Sentinel < 0 (default)
    #     falls back to scaffold_p2_contact_benefit_threshold -> bit-identical to
    #     the pre-amend 634b path. The 634c re-validation sets this to the
    #     effective seeding floor (matched to benefit_threshold / gain / drive_floor).
    scaffold_contact_gating_benefit_threshold: float = -1.0
    #
    # (2) GOAL-SEEDING MAGNITUDE PROPAGATION. When set (not None), these scaffold
    #     knobs are written onto the agent's GoalConfig (agent.goal_state.config)
    #     at the top of each run_* stage so genuine wild contact can clear the
    #     GoalState firing threshold. None (default) leaves the agent's existing
    #     GoalConfig untouched -> bit-identical. The 634c sweep picks the
    #     combination (autopsy: "one or a combination, pick via a small sweep").
    #       scaffold_z_goal_seeding_gain -> GoalConfig.z_goal_seeding_gain (raise > 1.0)
    #       scaffold_benefit_threshold   -> GoalConfig.benefit_threshold (lower < 0.1)
    #       scaffold_drive_floor         -> GoalConfig.drive_floor (raise ~0.9; the
    #         EXQ-582a first-PASS arm, effective_benefit >= benefit * 2.8).
    scaffold_z_goal_seeding_gain: Optional[float] = None
    scaffold_benefit_threshold: Optional[float] = None
    scaffold_drive_floor: Optional[float] = None

    # Training rates (mirrors committed_mode_curriculum.py defaults).
    scaffold_lr_e1: float = 1e-4
    scaffold_lr_e2_wf: float = 1e-3
    scaffold_batch_size: int = 32
    scaffold_wf_buf_max: int = 2000

    # SD-057 cue-recall bridge (2026-06-04 amend; GAP-2 foraging-contact lever).
    # Hypothesis: the nursery already builds z_goal by forced-feed but has no path
    # from a nursery-built goal to APPROACHING a resource the agent can SEE but
    # has not yet contacted. SD-057 L6 cue-recall is that path (Pavlovian-
    # instrumental transfer / sign-tracking): forced-feed builds per-object tokens
    # in the SD-057 incentive bank; in P1/P2 a PERCEIVED resource cue retrieves
    # its token and pulls z_goal toward it -> MECH-295 approach bias -> first
    # contact. Targets the CONTACT axis of the foraging ceiling (NOT survival).
    #
    # Master switch -- default False, bit-identical OFF. When True it (a) enables
    # SD-049 (per-type identity tags + per-type proximity views + per-axis drive)
    # in this scheduler's envs, (b) passes resource_type into update_z_goal so the
    # bank binds per-object tokens, and (c) fires agent.cue_recall_wanting each
    # P1/P2 step on the strongest-perceived resource type. REQUIRES the CALLER to
    # build the agent with use_incentive_token_bank=True + use_cue_recall=True +
    # use_resource_encoder=True (the SD-057 substrate flags); without them the
    # wiring is harmless no-op (bank None -> resource_type ignored; cue_recall
    # returns 0).
    scaffold_cue_recall_bridge_enabled: bool = False
    # SD-049 resource-type count for the scaffold envs when the bridge is on.
    scaffold_cue_n_resource_types: int = 3
    # Minimum perceived-cue proximity for the auto cue-recall to fire each step.
    scaffold_cue_recall_min_proximity: float = 0.0
    # SD-057 cue-recall FORMATION fix (V3-EXQ-638 cue-silent autopsy, 2026-06-04).
    # When True (and the bridge is enabled), Stage-0 forced feeding binds the
    # incentive token to the STRONGEST-PERCEIVED resource type each step instead
    # of the (almost-always-None) ACTUALLY-CONTACTED type. Forced feeding is
    # decoupled from standing on a typed cell, so _contacted_resource_type returns
    # None on nearly every Stage-0 step -> bank.update (gated resource_type>0) is
    # never reached -> the IncentiveTokenBank stays EMPTY entering P1/P2 ->
    # cue_recall_wanting returns 0 at `k not in bank._base_value` -> cue_fires=0
    # (the 638 C1 failure). Binding to the perceived type ("the infant is fed;
    # bind the token to whatever food it perceives") populates the bank so the
    # cue has something to recall in the wild. Default False -> bit-identical
    # (rt = _contacted_resource_type, exactly as the pre-fix Stage-0 path).
    scaffold_stage0_bind_incentive_token: bool = False
    # V3-EXQ-640 post-cue MEASUREMENT-ONLY instrumentation (routed by
    # failure_autopsy_V3-EXQ-638a). When True, _eval_episode populates a
    # post_cue_diag accumulator with windowed per-cue-fire measurements
    # (z_goal-norm/pull delta around each cue fire, cue->action approach rate,
    # manhattan-distance-to-resource delta, hazard-salience-interrupt count,
    # first-gradient-improving-move latency, oscillation rate) -- the
    # discriminator the 638a autopsy needs to tell cue-to-action AUTHORITY /
    # displacement / gradient-following / interrupt apart, none of which 638a
    # could measure (no post-cue action trace was logged). PURELY READ-ONLY:
    # the agent still senses / selects / steps identically; this only reads env
    # + goal_state state. Default False -> bit-identical (no accumulator built,
    # the per-cue tracking block is skipped entirely).
    scaffold_post_cue_instrumentation: bool = False
    # Look-ahead window (steps) over which each cue fire's downstream movement is
    # attributed (gradient-improving move / hazard interrupt / oscillation /
    # first-improving-move latency). Only consulted when the flag above is set.
    scaffold_post_cue_window_steps: int = 4

    # -------- Foraging-competence residual (GAP-2 reach-contact, 2026-06-05) ----
    # The V3-EXQ-634c review (substrate_queue scaffolded_sd054_onboarding) confirmed
    # the z_goal SEEDING half is validated (seeded arms g3_zgoal ~0.44) but the
    # wean-to-wild foraging-competence / reach-contact half is the residual GAP-2
    # ceiling (seed-42 zero-contact, P1 survival 1/3). Two no-op-default levers
    # close it; both bit-identical OFF.
    #
    # (1) RECONCILE the contact-GATING decision with the GoalState SEEDING firing
    #     threshold. The 634c amend decoupled the gating floor
    #     (scaffold_contact_gating_benefit_threshold) from the contact-RATE readout,
    #     but it still had to be HAND-MATCHED to the GoalState seeding floor as a
    #     magic number -- a mismatch is exactly the 634b anti-correlation (the
    #     scaffold counts a step as "seeded" while GoalState only decay-updated it).
    #     When this flag is True, the gating floor is DERIVED from the agent's live
    #     GoalConfig magnitudes each stage (see _reconciled_gating_threshold), so the
    #     scaffold's "seeds" boolean tracks GoalState.update's actual firing decision
    #     (effective_benefit = benefit * gain * (1 + drive_weight * drive_trace) >
    #     benefit_threshold; the raw-benefit seeding floor is
    #     benefit_threshold / (gain * (1 + drive_weight * drive_floor))). Genuine
    #     wild contact that clears the GoalState floor seeds; sub-seeding whiffs are
    #     protected -- without the experiment having to keep the two knobs in sync.
    #     Default False -> _effective_gating_threshold falls back to the static
    #     _gating_threshold(), bit-identical to the 634c path.
    scaffold_auto_reconcile_gating_to_seeding: bool = False
    #
    # (2) GRADED SPAWN WEANING into early P1. P0 spawns the agent inside the reef
    #     refuge band (safe); the legacy P1 abruptly moves spawn to the midline for
    #     EVERY P1 episode, so a not-yet-foraging-competent agent is thrown to the
    #     hazard band before it has made a single benefit contact in the wild
    #     (603e: P1 survival 1/3). This lever keeps the reef-refuge spawn for the
    #     first `fraction` of P1 episodes (then switches to midline), extending the
    #     developmental safety window so the agent can reach food from safety before
    #     facing the midline. Complements scaffold_p1_anneal_hold_fraction (which
    #     holds the hazard/food-attraction anneal low) -- this holds the SPAWN safe.
    #     Default 0.0 = spawn at midline for all of P1 (bit-identical to legacy).
    scaffold_p1_reef_spawn_hold_fraction: float = 0.0

    # -------- Curriculum decomposition: isolated hazard-avoidance stage --------
    # (V3-EXQ-603f autopsy, 2026-06-07). 603f proved the goal-formation +
    # ecological-seeding chain is SOUND (seed 44 foraged contact_rate 0.393 AND
    # seeded z_goal 0.450) but the single remaining GAP-2 blocker is the P1
    # SURVIVAL / hazard-avoidance leg (G1 0/3; even the foraging seed died at
    # median 28.5 vs gate 75). Root cause: P1 couples TWO competencies at once
    # (goal-pipeline unfreeze + wean into the hazard band) and the agent cannot
    # acquire both simultaneously; P0 trains only in the safe reef refuge, so the
    # agent never learns hazard navigation before P1 throws it at hazards.
    #
    # THE FIX (user-directed): a SEPARATE hazard-avoidance training stage between
    # P0 (safe, goal-frozen warm-up) and P1 (combined wean). Hazards present,
    # foraging pressure minimal, hazard_food_attraction=0 (so foraging does NOT
    # raise hazard exposure -- clean avoidance signal), goal pipeline FROZEN (the
    # isolation: no goal-unfreeze competing). Midline spawn so the agent must
    # actually navigate the hazard band (the reef refuge stays available as the
    # flee-to-safety attractor). P1 is then entered by an already-survival-AND-
    # goal-competent policy. All knobs additive + no-op default
    # (scaffold_hazard_stage_enabled False); the V3-EXQ-603g re-issue opts in.
    # Bit-identical OFF: the scheduler never runs the stage unless an experiment
    # both sets the flag AND calls run_hazard_avoidance.
    scaffold_hazard_stage_enabled: bool = False
    scaffold_hazard_stage_episode_budget: int = 40
    # Hazards present at target density; resources minimal so avoidance (not
    # foraging) is the dominant learning signal.
    scaffold_hazard_stage_num_hazards: int = 4
    scaffold_hazard_stage_num_resources: int = 2
    # hazard_food_attraction=0.0 -> hazards drift randomly (NOT toward food), so
    # foraging does not increase hazard exposure; avoidance is learnable in
    # isolation. proximity_harm_scale at the target level (0.1) so avoidance is
    # genuinely incentivised (matches P2).
    scaffold_hazard_stage_hazard_food_attraction: float = 0.0
    scaffold_hazard_stage_proximity_harm_scale: float = 0.1
    # Midline spawn (False) -> the agent must navigate the hazard band, the way
    # P1 does, but with the goal frozen. True keeps the safe reef-refuge spawn
    # (a gentler isolated stage; the experiment chooses).
    scaffold_hazard_stage_spawn_in_reef_half: bool = False
    # Survival readout: median episode length over the last stability window must
    # clear this floor for the isolated stage to count as survival-competent.
    # Diagnostic only (does NOT abort the curriculum) -- the canonical readiness
    # gate stays G0/G1/G2/G3; G_H is reported so the 603g manifest can confirm the
    # isolated stage achieved avoidance before P1.
    scaffold_hazard_stage_survival_gate_steps: int = 75
    scaffold_hazard_stage_stability_window: int = 10


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Stage0NurseryResult:
    """
    Outcome of ScaffoldedSD054OnboardingScheduler.run_stage0_nursery().

    The forced-benefit feeding positive control. z_goal_formed is the per-seed
    Stage-0 acceptance: did z_goal light when the infant was fed?
    """

    n_episodes: int
    mean_forced_benefit: float
    z_goal_norm_peak: float
    z_goal_formed: bool
    aborted: bool
    abort_reason: str = ""
    # SD-057 cue-recall FORMATION readout (2026-06-04): number of distinct
    # incentive-token types in the bank after Stage-0 forced feeding. 0 == the
    # bank is empty (the V3-EXQ-638 cue-silent root cause). Non-zero only when
    # scaffold_stage0_bind_incentive_token is on; the headline formation gate.
    token_bank_size_end: int = 0


@dataclass
class Stage0bConsolidationResult:
    """Outcome of ScaffoldedSD054OnboardingScheduler.run_stage0b_consolidation().

    The protected consolidation window after Stage-0: update_z_goal is NOT
    called, so the just-formed z_goal cannot be washed out by decay-only
    updating while the encoder/E2 keep training on the safe nursery. Retention
    is the developmental-window acceptance metric.
    """

    n_episodes: int
    z_goal_norm_start: float
    z_goal_norm_end: float
    retention_ratio: float
    retention_gate_passed: bool
    aborted: bool
    abort_reason: str = ""


@dataclass
class P0OnboardingResult:
    """Outcome of ScaffoldedSD054OnboardingScheduler.run_p0()."""

    n_episodes: int
    mean_episode_length: float
    final_running_variance: float
    aborted: bool
    abort_reason: str = ""


@dataclass
class HazardAvoidanceResult:
    """Outcome of ScaffoldedSD054OnboardingScheduler.run_hazard_avoidance().

    The isolated hazard-avoidance stage (2026-06-07 curriculum-decomposition
    amend). Goal pipeline frozen; hazards present; foraging pressure minimal.
    survival_gate_passed (median episode length over the last stability window
    >= scaffold_hazard_stage_survival_gate_steps) is the G_H readout: did the
    policy acquire hazard avoidance in isolation before P1 combines the
    competencies? Diagnostic -- it does NOT abort the curriculum or change the
    canonical G0/G1/G2/G3 readiness gate.
    """

    n_episodes: int
    mean_episode_length: float
    median_last_window_episode_length: float
    survival_gate_passed: bool
    final_running_variance: float
    aborted: bool
    abort_reason: str = ""
    episode_lengths: List[int] = field(default_factory=list)


@dataclass
class P1OnboardingResult:
    """Outcome of ScaffoldedSD054OnboardingScheduler.run_p1()."""

    n_episodes: int
    median_last_window_episode_length: float
    survival_gate_passed: bool
    final_hazard_food_attraction: float
    final_mech295_min_drive_to_fire: float
    final_mech307_conjunction_z_beta_threshold: float
    aborted: bool
    abort_reason: str = ""
    episode_lengths: List[int] = field(default_factory=list)
    # Developmental-window diagnostics (2026-06-03b amend; default 0 when the
    # contact-gating flag is off so these are pure additive readouts).
    n_contact_refresh_updates: int = 0
    n_decay_only_updates: int = 0
    n_skipped_protected_updates: int = 0
    contact_gated: bool = False
    # Foraging-competence residual (2026-06-05): number of P1 episodes the agent
    # spawned in the reef refuge half under graded spawn weaning
    # (scaffold_p1_reef_spawn_hold_fraction). 0 == legacy all-midline P1 (bit-
    # identical). The reconciled gating floor actually used this stage (derived
    # from the live GoalConfig when scaffold_auto_reconcile_gating_to_seeding is
    # on; -1.0 sentinel == static-fallback path).
    n_reef_spawn_episodes: int = 0
    reconciled_gating_threshold: float = -1.0
    # SD-057 cue-recall FORMATION readout (2026-06-04b aggregation fix): total
    # L6 cue-recall fires across all P1 episodes, surfaced from the
    # goal_write_diag accumulator. Contract: equals cue_diag["n_cue_recall_fires"]
    # (both increment by 1 on each fire of the same shared accumulators); 0 when
    # the bridge is off -> bit-identical. Closes the V3-EXQ-638 measurement gap
    # where a consumer doing getattr(p1, "n_cue_recall_fires", 0) silently read 0
    # EVEN WHEN THE CUE FIRED (no aggregated field existed).
    n_cue_recall_fires: int = 0
    # SD-057 cue-recall diagnostics (2026-06-04): see _new_cue_diag(). Empty dict
    # when the bridge is off -> bit-identical readout. Turns cue_fires=0 into an
    # attributed nonfire reason (no_token / resource_field_absent / ...).
    cue_diag: Dict[str, Any] = field(default_factory=dict)


@dataclass
class P2OnboardingMetrics:
    """P2 measurement outcomes per the memo Acceptance section."""

    n_episodes: int
    z_goal_norm_peak_per_episode: List[float]
    z_goal_norm_peak_max: float
    approach_commit_steps: int
    approach_commit_rate: float
    bridge_cue_fires: int
    dacc_bias_nonzero_steps: int
    mean_episode_length: float
    per_episode: List[Dict[str, Any]]
    # P2 measurement-guard additions (2026-06-03 amend): foraging-contact-rate
    # readout so a z_goal=0 read is distinguishable from benefit starvation.
    contact_steps: int = 0
    contact_rate: float = 0.0
    hazard_food_attraction_used: float = 0.0
    # Foraging-competence residual (2026-06-05): the reconciled gating floor used
    # for the P2 skip/seed decision (derived from the live GoalConfig when
    # scaffold_auto_reconcile_gating_to_seeding is on; -1.0 sentinel == static
    # _gating_threshold fallback). Recorded so a manifest can confirm the P2 seed
    # decision tracked the GoalState firing floor.
    reconciled_gating_threshold: float = -1.0
    # Developmental-window diagnostics (2026-06-03b amend; default 0 when the
    # contact-gating flag is off).
    n_contact_refresh_updates: int = 0
    n_decay_only_updates: int = 0
    n_skipped_protected_updates: int = 0
    contact_gated: bool = False
    # Consumption-event-gated z_goal readout (V3-EXQ-634b autopsy, 2026-06-03c):
    # the G3 acceptance readout must read z_goal AT genuine consumption events
    # (632-style), NOT the forced-feed-calibrated frozen peak (z_goal_norm_peak_max),
    # which seed 42 "passed" by carrying the untouched nursery trace through a
    # zero-contact P2. z_goal_norm_at_contact_peak is the max goal-norm read only
    # on a validated-contact step (post-seeding); 0.0 when the agent never made
    # ecological contact -> a z_goal=0-at-contact read is now interpretable as
    # "goal not maintained by foraging" rather than masked by the frozen trace.
    z_goal_norm_at_contact_peak: float = 0.0
    num_contact_events: int = 0
    # SD-057 cue-recall FORMATION readout (2026-06-04b aggregation fix): total
    # L6 cue-recall fires across all P2 episodes, summed from the per-episode
    # ep_metrics["n_cue_recall_fires"]. Contract: equals cue_diag["n_cue_recall_fires"]
    # (the per-episode returns and the shared cue_diag both count the same fires);
    # 0 when the bridge is off -> bit-identical. Closes the V3-EXQ-638 measurement
    # gap where a consumer doing getattr(p2, "n_cue_recall_fires", 0) silently read
    # 0 EVEN WHEN THE CUE FIRED (the 638 C1 gate read this; 638a worked around it
    # by sourcing cue_diag["n_cue_recall_fires"]).
    n_cue_recall_fires: int = 0
    # SD-057 cue-recall diagnostics (2026-06-04): see _new_cue_diag(). Empty dict
    # when the bridge is off -> bit-identical readout.
    cue_diag: Dict[str, Any] = field(default_factory=dict)
    # V3-EXQ-640 post-cue action/gradient diagnostics (2026-06-05): see
    # _new_post_cue_diag(). Empty dict when scaffold_post_cue_instrumentation is
    # off -> bit-identical readout.
    post_cue_diag: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Goal-write developmental windowing modes (2026-06-03b amend)
# ---------------------------------------------------------------------------
#
# The persistent GoalState attractor must be written under an explicit
# developmental mode rather than blindly decay-updated every step. These are
# legibility + diagnostics constants; the actual mechanism is simply "whether
# update_z_goal is called this step, and whether the step is a validated
# contact." See run_stage0_nursery / run_stage0b_consolidation / _train_episode
# / _eval_episode.
GOAL_WRITE_FORCED_FEED_OPEN = "forced_feed_open"          # Stage-0: forced benefit seeds z_goal
GOAL_WRITE_CONSOLIDATE_PROTECTED = "consolidate_protected"  # Stage-0b: no decay-only washout
GOAL_WRITE_ECOLOGICAL_CONTACT_OPEN = "ecological_contact_open"  # P1/P2 gated: update only on real contact
GOAL_WRITE_DECAY_ONLY_ALLOWED = "decay_only_allowed"      # mature tests only; NOT the nursery gate
GOAL_WRITE_MEASUREMENT_READONLY = "measurement_readonly"  # read z_goal without modifying
GOAL_WRITE_MODES = (
    GOAL_WRITE_FORCED_FEED_OPEN,
    GOAL_WRITE_CONSOLIDATE_PROTECTED,
    GOAL_WRITE_ECOLOGICAL_CONTACT_OPEN,
    GOAL_WRITE_DECAY_ONLY_ALLOWED,
    GOAL_WRITE_MEASUREMENT_READONLY,
)


def _new_goal_write_diag() -> Dict[str, int]:
    """Fresh per-phase goal-write diagnostics accumulator. Counts how z_goal
    was written so a manifest can distinguish goal loss due to no-contact /
    decay-only washout / failed-formation-despite-contact."""
    return {
        "n_contact_refresh": 0,      # update called on a validated-contact step (pull)
        "n_decay_only": 0,           # update called on an unfed step (decay-only)
        "n_skipped_protected": 0,    # update skipped (contact-gated protection)
        "n_contact_steps": 0,        # steps with benefit > contact threshold
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation; t clamped to [0, 1]."""
    t = max(0.0, min(1.0, float(t)))
    return float(start + (end - start) * t)


def _benefit_and_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    """
    Extract (benefit_exposure, drive_level) from a body-state observation,
    mirroring experiments/goal_stream_stages_sd054.py:_benefit_and_drive (the
    reference goal-stream runner the V3-EXQ-622 autopsy confirmed feeds z_goal).

    benefit_exposure = obs_body[11] (resource-contact benefit proxy).
    drive_level      = clip(1 - energy, 0, 1) where energy = obs_body[3] (SD-012).

    Robust to both [body_dim] and [1, body_dim] shapes (CausalGridWorldV2 emits
    1-D body_state of length 17 when limb_damage is enabled).
    """
    b = obs_body.reshape(-1)
    benefit = float(b[11].item()) if b.shape[0] > 11 else 0.0
    energy = float(b[3].item()) if b.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _sd049_kwargs(cfg: ScaffoldedSD054OnboardingConfig) -> Dict[str, Any]:
    """SD-057 cue-recall bridge: SD-049 enablement kwargs for the scaffold envs.

    Returns {} when the bridge is off (bit-identical legacy envs). When on, the
    envs emit per-type identity tags (resource_type_at_agent /
    sd049_consumed_type_tag_this_tick), per-type proximity field views
    (resource_field_view_<name>), and per_axis_drive -- the inputs the SD-057
    bank-token-binding (L2) and cue-recall (L6) need.
    """
    if not cfg.scaffold_cue_recall_bridge_enabled:
        return {}
    return {
        "multi_resource_heterogeneity_enabled": True,
        "n_resource_types": int(cfg.scaffold_cue_n_resource_types),
        "per_axis_drive_enabled": True,
    }


def _contacted_resource_type(obs_dict: Dict[str, Any]) -> Optional[int]:
    """SD-057 L2: the SD-049 identity tag for bank-token binding. Prefers the
    consumed-this-tick tag, falls back to the at-agent tag. None when absent
    (SD-049 off) or no resource (tag 0)."""
    for key in ("sd049_consumed_type_tag_this_tick", "resource_type_at_agent"):
        raw = obs_dict.get(key, None)
        if raw is None:
            continue
        try:
            tag = int(raw[0] if hasattr(raw, "__len__") else raw)
        except (TypeError, ValueError):
            continue
        if tag > 0:
            return tag
    return None


def _strongest_perceived_type(env, obs_dict: Dict[str, Any]) -> Tuple[int, float]:
    """SD-057 perceptual primitive shared by cue-recall (L6) and Stage-0 token
    FORMATION binding. Returns (best_tag, best_prox) for the strongest-perceived
    SD-049 resource type from the per-type proximity field views.

    best_tag is the 1-based SD-049 type tag (0 = no typed field perceived);
    best_prox is its peak field-view value (-1.0 when no field views present).
    Identical logic at formation and recall time guarantees the token the
    nursery lays down is keyed to the same perceptual channel the wild cue reads.
    """
    type_names = getattr(env, "resource_type_names", ()) or ()
    best_tag, best_prox = 0, -1.0
    for i, name in enumerate(type_names):
        fv = obs_dict.get(f"resource_field_view_{name}", None)
        if fv is None:
            continue
        v = float(fv.max()) if hasattr(fv, "max") else float(max(fv))
        if v > best_prox:
            best_prox, best_tag = v, i + 1
    return best_tag, best_prox


def _new_cue_diag() -> Dict[str, Any]:
    """Fresh per-phase SD-057 cue-recall diagnostics accumulator. Turns
    cue_fires=0 from an undiagnosable mystery (the pre-2026-06-04 silent
    `except: pass`) into an attributed reason -- the load-bearing diagnostic for
    the V3-EXQ-638 / 638a cue-silent autopsy."""
    return {
        "n_external_cues_seen": 0,        # steps where a typed resource field was perceived
        "n_cue_recall_attempts": 0,       # steps where a matching token was looked up
        "n_cue_recall_fires": 0,          # steps where cue_recall_wanting moved z_goal
        "n_token_matches": 0,             # perceived-type had a bound token
        # Reserved for the NEXT layer (interoceptive need-gating); stays 0 here.
        "n_interoceptive_need_cues": 0,
        "n_joint_cues": 0,
        "best_prox_peak": 0.0,
        "drive_peak": 0.0,
        "token_bank_size": 0,             # len(bank._base_value) -- 0 == empty bank
        "matched_token_strength_peak": 0.0,
        "cue_nonfire_reason_counts": {},  # reason -> count (the 'why a zero' map)
    }


def _bump_reason(diag: Optional[Dict[str, Any]], reason: str) -> None:
    if diag is None:
        return
    rc = diag["cue_nonfire_reason_counts"]
    rc[reason] = int(rc.get(reason, 0)) + 1


def _nearest_resource_manhattan(env) -> Optional[int]:
    """V3-EXQ-640: Manhattan distance from the agent to the nearest resource cell
    (any type). None when the env has no resources left this episode. The scaffold
    envs are non-toroidal (CausalGridWorldV2 default toroidal=False), so plain
    |dx| + |dy| is the grid metric. Read-only -- the post-cue gradient diagnostic
    uses this to ask whether the agent's moves after a cue fire reduce distance to
    food."""
    res = getattr(env, "resources", None)
    if not res:
        return None
    ax, ay = int(getattr(env, "agent_x", 0)), int(getattr(env, "agent_y", 0))
    best: Optional[int] = None
    for r in res:
        try:
            d = abs(int(r[0]) - ax) + abs(int(r[1]) - ay)
        except (TypeError, ValueError, IndexError):
            continue
        if best is None or d < best:
            best = d
    return best


def _new_post_cue_diag(window_steps: int) -> Dict[str, Any]:
    """V3-EXQ-640 per-cue-fire MEASUREMENT accumulator. The 638a autopsy settled
    only "cue fires vs contact lifts" and could not discriminate cue-to-action
    AUTHORITY / displacement / gradient-following / hazard-interrupt because no
    post-cue action trace was logged. This accumulator captures, per cue fire and
    over a short look-ahead window, the signals that route the next move (see the
    638a-autopsy discriminator grid). Purely read-only; bit-identical when the
    instrumentation flag is off (this dict is simply never built).

    Rate/mean derivations are left to the experiment script (it owns the
    interpretation), so this stores raw sums + counts only.
    """
    return {
        "post_cue_window_steps": int(window_steps),
        "n_steps_total": 0,                  # all P2 steps (denominator for background rates)
        "n_cue_fire_steps": 0,               # steps where a cue actually fired
        # --- z_goal around each cue fire (the DISPLACEMENT test) ---
        # delta = ||z_goal||_after_cue - ||z_goal||_before_cue. mean < 0 => the
        # cue pulls z_goal toward a WEAKER token (displacement), not authority.
        "sum_post_cue_zgoal_norm_delta": 0.0,
        "n_post_cue_zgoal_norm_delta": 0,
        # ||z_goal_after - z_goal_before|| -- the cue's actual pull MAGNITUDE
        # (the SD-057 analog of cue_action_bias; nonzero => the cue moves z_goal).
        "sum_cue_zgoal_pull_norm": 0.0,
        # absolute ||z_goal|| read right after the cue fired -- compared ON-vs-OFF
        # arm, a LOWER post-cue norm than the OFF wild-seeded attractor is the
        # displacement signature.
        "sum_zgoal_norm_at_cue_fire": 0.0,
        "min_zgoal_norm_at_cue_fire": float("inf"),
        "max_zgoal_norm_at_cue_fire": 0.0,
        # SD-016 cue_action_proj bias norm (agent._cue_action_bias); usually 0 in
        # the 638a config (SD-016 off) -- captured for completeness so a future
        # SD-016-on arm is comparable. The SD-057 cue's authority lives in the
        # z_goal pull above, not here.
        "sum_cue_action_bias_norm": 0.0,
        "n_cue_action_bias_present": 0,
        # --- windowed post-cue MOVEMENT (the gradient-following / interrupt test) ---
        "n_cue_windows": 0,                  # one per cue fire that opened a window
        "n_windows_first_move_approach": 0,  # first post-cue move reduced distance (immediate authority)
        "n_windows_with_approach_move": 0,   # >=1 gradient-improving move within the window
        "n_windows_improved": 0,             # windows that produced any improving move (latency denom)
        "sum_first_improving_latency": 0,    # 1-based step index of first improving move, over improved windows
        "n_windows_with_hazard_interrupt": 0,  # a harm spike occurred within the window
        "sum_window_oscillations": 0,        # action-direction reversals within the window
        # --- background (cue-independent) movement rates, so post-cue rates are
        #     interpretable against the agent's baseline foraging competence ---
        "n_move_eval_steps": 0,              # steps where distance was computable both ends
        "sum_move_improved_all_steps": 0,    # background approach-move count
        "n_postcue_eval_steps": 0,           # move-eval steps that fell inside an active cue window
        "sum_move_improved_postcue_steps": 0,  # approach moves on post-cue steps
        "sum_zgoal_norm_all_steps": 0.0,     # mean ||z_goal|| over all steps (norm-context)
        "n_zgoal_norm_all_steps": 0,
    }


def _opposite_action(env, a_idx: int, b_idx: int) -> bool:
    """True when env action a_idx is the spatial inverse of b_idx (dx,dy negated)
    -- a movement reversal, used for the post-cue oscillation count. Robust to a
    missing / malformed ACTIONS table (returns False)."""
    actions = getattr(env, "ACTIONS", None)
    if actions is None:
        return False
    try:
        ax, ay = actions[a_idx]
        bx, by = actions[b_idx]
    except (TypeError, ValueError, IndexError, KeyError):
        return False
    return int(ax) == -int(bx) and int(ay) == -int(by)


def _read_zgoal(goal_state) -> Tuple[float, Optional[torch.Tensor]]:
    """V3-EXQ-640: read (||z_goal||, z_goal vector clone) for the post-cue
    displacement diagnostic. Returns (0.0, None) when no goal_state / z_goal."""
    if goal_state is None:
        return 0.0, None
    try:
        n = float(goal_state.goal_norm())
    except TypeError:
        try:
            n = float(goal_state.goal_norm)
        except Exception:
            n = 0.0
    except Exception:
        n = 0.0
    zg = getattr(goal_state, "z_goal", None)
    vec = None
    if zg is not None:
        try:
            vec = zg.detach().clone()
        except Exception:
            vec = None
    return n, vec


def _finalize_post_cue_window(diag: Dict[str, Any], w: Dict[str, Any]) -> None:
    """V3-EXQ-640: fold a completed (or episode-end-truncated) per-cue-fire window
    into the post_cue_diag accumulator."""
    diag["n_cue_windows"] += 1
    if w["first_move_approach"]:
        diag["n_windows_first_move_approach"] += 1
    if w["improved"]:
        diag["n_windows_with_approach_move"] += 1
        diag["n_windows_improved"] += 1
        diag["sum_first_improving_latency"] += int(w["first_latency"])
    if w["hazard"]:
        diag["n_windows_with_hazard_interrupt"] += 1
    diag["sum_window_oscillations"] += int(w["osc"])


def _maybe_cue_recall(agent, env, obs_dict: Dict[str, Any], drive: float,
                      cfg: ScaffoldedSD054OnboardingConfig,
                      diag: Optional[Dict[str, Any]] = None) -> int:
    """SD-057 L6 cue-recall, ecological auto-perception. Derives the strongest-
    perceived resource type from the SD-049 per-type proximity field views and
    fires agent.cue_recall_wanting on it (raising wanting/approach toward a
    perceived-but-uncontacted resource). Best-effort; returns 1 if a cue fired.

    No-op (returns 0) when the bridge is off, the agent lacks the SD-057
    bank/cue-recall (caller didn't set the flags), or the env emits no per-type
    views. Bit-identical when off.

    When `diag` (a _new_cue_diag() dict) is supplied, EVERY non-fire is attributed
    to a reason and the substrate quantities (best_prox, drive, token_bank_size,
    matched strength) are recorded. The prior `except: pass` is replaced with an
    `exception:<type>` reason so a thrown error is visible, not swallowed -- while
    still never breaking the episode loop (cue-recall is best-effort).
    """
    if not cfg.scaffold_cue_recall_bridge_enabled:
        return 0
    if diag is not None:
        diag["drive_peak"] = max(float(diag["drive_peak"]), float(drive))
    try:
        # Set per-axis drive so the bank's drive-specific wanting is identity-
        # matched; scalar-drive fallback inside cue_recall_wanting otherwise.
        pad = obs_dict.get("per_axis_drive", None)
        if pad is not None:
            agent._per_axis_drive = pad.reshape(-1) if hasattr(pad, "reshape") else pad
        best_tag, best_prox = _strongest_perceived_type(env, obs_dict)
        if diag is not None and best_prox > float(diag["best_prox_peak"]):
            diag["best_prox_peak"] = float(best_prox)
        # Token-bank introspection (drives the nonfire-reason attribution).
        gs = getattr(agent, "goal_state", None)
        bank = getattr(gs, "incentive_bank", None) if gs is not None else None
        bank_size = len(getattr(bank, "_base_value", {})) if bank is not None else 0
        if diag is not None:
            diag["token_bank_size"] = int(bank_size)
        if best_tag <= 0:
            _bump_reason(diag, "resource_field_absent")
            return 0
        if diag is not None:
            diag["n_external_cues_seen"] += 1
        if best_prox < float(cfg.scaffold_cue_recall_min_proximity):
            _bump_reason(diag, "proximity_below_threshold")
            return 0
        if bank is None:
            _bump_reason(diag, "bank_none")
            return 0
        if best_tag not in getattr(bank, "_base_value", {}):
            _bump_reason(diag, "no_token")  # the V3-EXQ-638 cue-silent root cause
            return 0
        if diag is not None:
            diag["n_token_matches"] += 1
            diag["n_cue_recall_attempts"] += 1
        s = agent.cue_recall_wanting(cue_type=best_tag, drive_level=float(drive))
        if s is not None and s > 0.0:
            if diag is not None:
                diag["n_cue_recall_fires"] += 1
                diag["matched_token_strength_peak"] = max(
                    float(diag["matched_token_strength_peak"]), float(s)
                )
            return 1
        _bump_reason(diag, "amp_zero_or_zobject_none")
        return 0
    except Exception as e:  # never break the loop, but make the failure VISIBLE
        _bump_reason(diag, f"exception:{type(e).__name__}")
        return 0


def _build_env(cfg: ScaffoldedSD054OnboardingConfig, phase: str, anneal_t: float = 0.0,
               p1_spawn_in_reef_half: bool = False):
    """
    Build a CausalGridWorldV2 instance for the named phase.

    phase in {"stage0", "p0", "p1", "p2"}. anneal_t in [0, 1] used only for p1.

    p1_spawn_in_reef_half (foraging-competence residual, 2026-06-05): when True,
    the P1 env spawns the agent inside the reef refuge half instead of the
    midline band -- the graded spawn-weaning lever
    (scaffold_p1_reef_spawn_hold_fraction) extending P0's developmental safety
    into early P1. Default False = legacy midline P1 spawn (bit-identical).
    Ignored for non-p1 phases (stage0/p0 always reef-half; p2 always midline).
    """
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    if phase == "stage0":
        # Infant-REE nursery: dense resources, no hazards, agent spawns inside
        # the reef refuge band, hazard_food_attraction=0. A safe, food-rich
        # crib so the forced-benefit feed and encoder warm-up are not gated by
        # survival pressure.
        return CausalGridWorldV2(
            size=cfg.scaffold_env_size,
            num_hazards=cfg.scaffold_stage0_num_hazards,
            num_resources=cfg.scaffold_stage0_num_resources,
            **_sd049_kwargs(cfg),  # SD-057 cue-recall bridge (no-op when off)
            hazard_food_attraction=0.0,
            proximity_harm_scale=cfg.scaffold_stage0_proximity_harm_scale,
            limb_damage_enabled=True,
            reef_enabled=True,
            reef_bipartite_layout=True,
            reef_bipartite_axis=cfg.scaffold_reef_bipartite_axis,
            reef_bipartite_agent_band_radius=cfg.scaffold_reef_bipartite_agent_band_radius,
            reef_bipartite_agent_spawn_in_reef_half=True,
        )
    if phase == "p0":
        return CausalGridWorldV2(
            size=cfg.scaffold_env_size,
            num_hazards=cfg.scaffold_p0_num_hazards,
            num_resources=cfg.scaffold_p0_num_resources,
            **_sd049_kwargs(cfg),  # SD-057 cue-recall bridge (no-op when off)
            hazard_food_attraction=0.0,
            proximity_harm_scale=cfg.scaffold_p0_proximity_harm_scale,
            limb_damage_enabled=True,
            reef_enabled=True,
            reef_bipartite_layout=True,
            reef_bipartite_axis=cfg.scaffold_reef_bipartite_axis,
            reef_bipartite_agent_band_radius=cfg.scaffold_reef_bipartite_agent_band_radius,
            reef_bipartite_agent_spawn_in_reef_half=True,
        )
    if phase == "hazard":
        # Isolated hazard-avoidance stage (2026-06-07): hazards present, foraging
        # pressure minimal, hazard_food_attraction=0 so foraging does not raise
        # hazard exposure. Spawn at the midline (default) so the agent must
        # navigate the hazard band; the reef refuge stays available as the
        # flee-to-safety attractor. Same structural kwargs (reef + bipartite +
        # SD-049 + limb_damage) as every other phase so world_obs_dim matches the
        # single shared agent.
        return CausalGridWorldV2(
            size=cfg.scaffold_env_size,
            num_hazards=cfg.scaffold_hazard_stage_num_hazards,
            num_resources=cfg.scaffold_hazard_stage_num_resources,
            **_sd049_kwargs(cfg),  # SD-057 cue-recall bridge (no-op when off)
            hazard_food_attraction=cfg.scaffold_hazard_stage_hazard_food_attraction,
            proximity_harm_scale=cfg.scaffold_hazard_stage_proximity_harm_scale,
            limb_damage_enabled=True,
            reef_enabled=True,
            reef_bipartite_layout=True,
            reef_bipartite_axis=cfg.scaffold_reef_bipartite_axis,
            reef_bipartite_agent_band_radius=cfg.scaffold_reef_bipartite_agent_band_radius,
            reef_bipartite_agent_spawn_in_reef_half=bool(
                cfg.scaffold_hazard_stage_spawn_in_reef_half
            ),
        )
    if phase == "p1":
        hfa = _lerp(
            cfg.scaffold_p1_anneal_hazard_food_attraction_min,
            cfg.scaffold_p1_anneal_hazard_food_attraction_max,
            anneal_t,
        )
        phs = _lerp(
            cfg.scaffold_p1_anneal_proximity_harm_scale_min,
            cfg.scaffold_p1_anneal_proximity_harm_scale_max,
            anneal_t,
        )
        return CausalGridWorldV2(
            size=cfg.scaffold_env_size,
            num_hazards=cfg.scaffold_p2_num_hazards,
            num_resources=cfg.scaffold_p2_num_resources,
            hazard_food_attraction=hfa,
            **_sd049_kwargs(cfg),  # SD-057 cue-recall bridge (no-op when off)
            proximity_harm_scale=phs,
            limb_damage_enabled=True,
            reef_enabled=True,
            reef_bipartite_layout=True,
            reef_bipartite_axis=cfg.scaffold_reef_bipartite_axis,
            reef_bipartite_agent_band_radius=cfg.scaffold_reef_bipartite_agent_band_radius,
            reef_bipartite_agent_spawn_in_reef_half=bool(p1_spawn_in_reef_half),
        )
    if phase == "p2":
        # P2 measurement guard: when scaffold_p2_hazard_food_attraction_guard
        # is >= 0 it overrides the (hard) default so the measurement window
        # admits foraging contact; default -1.0 keeps the legacy 0.7.
        p2_hfa = (
            cfg.scaffold_p2_hazard_food_attraction_guard
            if cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
            else cfg.scaffold_p2_hazard_food_attraction
        )
        return CausalGridWorldV2(
            size=cfg.scaffold_env_size,
            num_hazards=cfg.scaffold_p2_num_hazards,
            num_resources=cfg.scaffold_p2_num_resources,
            hazard_food_attraction=p2_hfa,
            **_sd049_kwargs(cfg),  # SD-057 cue-recall bridge (no-op when off)
            proximity_harm_scale=cfg.scaffold_p2_proximity_harm_scale,
            limb_damage_enabled=True,
            reef_enabled=True,
            reef_bipartite_layout=True,
            reef_bipartite_axis=cfg.scaffold_reef_bipartite_axis,
            reef_bipartite_agent_band_radius=cfg.scaffold_reef_bipartite_agent_band_radius,
            reef_bipartite_agent_spawn_in_reef_half=False,
        )
    raise ValueError(f"unknown phase: {phase!r}")


def _set_goal_pipeline_frozen(agent, frozen: bool) -> None:
    """
    Freeze or unfreeze the goal-pipeline write paths on an agent.

    "Frozen" means the MECH-295 liking bridge and MECH-307 conjunction
    write/read sites short-circuit even if their config flags would
    otherwise admit a write. Implemented by mutating the runtime flags
    on agent.config so the existing per-tick guards take the short-circuit
    branch. No ree_core changes required.
    """
    if frozen:
        agent.config.use_mech295_liking_bridge = False
        agent.config.use_mech307_conjunction = False
    else:
        agent.config.use_mech295_liking_bridge = True
        agent.config.use_mech307_conjunction = True


def _set_p1_anneal_state(agent, cfg: ScaffoldedSD054OnboardingConfig, anneal_t: float) -> None:
    """
    Apply the P1 anneal step to agent's goal-pipeline gate config.

    Mutates the live bridge config dataclass so the next tick reads the
    updated thresholds. The bridge reads its own config.min_drive_to_fire
    and config.mech307_conjunction_z_beta_threshold per-call, so the
    mutation takes effect immediately without rebuild.
    """
    drive_floor = _lerp(
        cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_max,
        cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_min,
        anneal_t,
    )
    z_beta = _lerp(
        cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_max,
        cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_min,
        anneal_t,
    )
    bridge = getattr(agent, "mech295_bridge", None)
    if bridge is not None:
        bridge.config.min_drive_to_fire = float(drive_floor)
        bridge.config.mech307_conjunction_z_beta_threshold = float(z_beta)
    agent.config.mech295_min_drive_to_fire = float(drive_floor)
    agent.config.mech307_conjunction_z_beta_threshold = float(z_beta)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class ScaffoldedSD054OnboardingScheduler:
    """
    Three-phase scheduler for the scaffolded_sd054_onboarding substrate.

    Holds the config + per-phase telemetry. Caller owns the agent and the
    REEConfig; the scheduler mutates agent.config flags across phase
    boundaries but does NOT touch encoder weights, predictor weights, or
    optimizer state.

    Usage outline:

        scheduler = ScaffoldedSD054OnboardingScheduler(cfg)
        p0 = scheduler.run_p0(agent, device)
        if p0.aborted:
            return {"outcome": "scaffold_p0_aborted", "p0": p0}
        p1 = scheduler.run_p1(agent, device)
        if not p1.survival_gate_passed:
            return {"outcome": "scaffold_p1_survival_failed", "p0": p0, "p1": p1}
        metrics = scheduler.run_p2(agent, device)
        return {"p0": p0, "p1": p1, "p2": metrics}
    """

    def __init__(self, cfg: ScaffoldedSD054OnboardingConfig):
        self.cfg = cfg
        self._stage0_result: Optional[Stage0NurseryResult] = None
        self._p0_result: Optional[P0OnboardingResult] = None
        self._hazard_result: Optional[HazardAvoidanceResult] = None
        self._p1_result: Optional[P1OnboardingResult] = None
        self._p2_metrics: Optional[P2OnboardingMetrics] = None

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.use_scaffolded_sd054_onboarding_scheduler)

    def _gating_threshold(self) -> float:
        """Benefit threshold for the contact-GATING (skip/update) decision.

        V3-EXQ-634b autopsy seeding-calibration amend: decoupled from the
        contact-RATE readout threshold (scaffold_p2_contact_benefit_threshold).
        Sentinel < 0 (default) -> fall back to the readout threshold, so the
        skip decision is bit-identical to the pre-amend 634b path. When >= 0,
        the band (readout_floor, gating_threshold) is PROTECTED (skipped) rather
        than decay-only updated -- the 634c re-validation matches this to the
        GoalState seeding firing floor so sub-seeding whiffs do not erode the
        consolidated trace.
        """
        g = float(self.cfg.scaffold_contact_gating_benefit_threshold)
        if g < 0.0:
            return float(self.cfg.scaffold_p2_contact_benefit_threshold)
        return g

    def _reconciled_gating_threshold(self, agent) -> Optional[float]:
        """Raw-benefit gating floor DERIVED from the agent's live GoalConfig so
        the scaffold's seed/skip decision tracks GoalState.update's actual firing
        (foraging-competence residual, 2026-06-05).

        GoalState seeds when
            effective_benefit = benefit * z_goal_seeding_gain
                                * (1 + drive_weight * drive_trace) > benefit_threshold
        (goal.py:383-388). In steady state drive_trace >= drive_floor (the SD-012
        insatiability floor, goal.py:369), so the raw-benefit floor at which a step
        clears the GoalState seeding gate is

            benefit_seed_floor = benefit_threshold
                                 / (z_goal_seeding_gain * (1 + drive_weight * drive_floor)).

        Returns None when auto-reconcile is off, the agent has no GoalState, or
        the denominator is degenerate -> caller falls back to the static
        _gating_threshold(). This is the RECONCILIATION half of the residual: it
        removes the need to hand-match scaffold_contact_gating_benefit_threshold to
        the seeding magnitudes (a mismatch is the 634b anti-correlation).

        Call AFTER _apply_goal_seeding_calibration so gc reflects the scaffold
        calibration.
        """
        if not self.cfg.scaffold_auto_reconcile_gating_to_seeding:
            return None
        gs = getattr(agent, "goal_state", None)
        gc = getattr(gs, "config", None) if gs is not None else None
        if gc is None:
            return None
        gain = float(getattr(gc, "z_goal_seeding_gain", 1.0))
        thr = float(getattr(gc, "benefit_threshold", 0.1))
        dw = float(getattr(gc, "drive_weight", 0.0))
        df = float(getattr(gc, "drive_floor", 0.0))
        denom = gain * (1.0 + dw * df)
        if denom <= 1e-12:
            return None
        return thr / denom

    def _effective_gating_threshold(self, agent) -> float:
        """The gating floor actually used this stage: the reconciled (GoalConfig-
        derived) floor when scaffold_auto_reconcile_gating_to_seeding is on,
        otherwise the static _gating_threshold(). Bit-identical to the 634c path
        when auto-reconcile is off."""
        reconciled = self._reconciled_gating_threshold(agent)
        if reconciled is not None:
            return float(reconciled)
        return self._gating_threshold()

    def _apply_goal_seeding_calibration(self, agent) -> None:
        """Propagate the scaffold seeding-magnitude knobs onto the agent's live
        GoalConfig (V3-EXQ-634b autopsy amend). No-op when every knob is None
        (default) or the agent has no goal_state -> bit-identical to pre-amend.

        Idempotent: called at the top of each seeding-capable run_* stage so the
        calibration holds regardless of which stages an experiment runs. Sets
        GoalState.config in place; the GoalState reads these per-call in update(),
        so the change takes effect on the next update_z_goal without rebuild.
        """
        gs = getattr(agent, "goal_state", None)
        if gs is None:
            return
        gc = getattr(gs, "config", None)
        if gc is None:
            return
        if self.cfg.scaffold_z_goal_seeding_gain is not None:
            gc.z_goal_seeding_gain = float(self.cfg.scaffold_z_goal_seeding_gain)
        if self.cfg.scaffold_benefit_threshold is not None:
            gc.benefit_threshold = float(self.cfg.scaffold_benefit_threshold)
        if self.cfg.scaffold_drive_floor is not None:
            gc.drive_floor = float(self.cfg.scaffold_drive_floor)

    # ---------------- Stage-0 nursery (forced-benefit feeding) ---------------- #

    def run_stage0_nursery(self, agent, device: torch.device) -> Stage0NurseryResult:
        """
        Stage 0: infant-REE nursery / forced-benefit feeding.

        Runs scaffold_stage0_episode_budget episodes in a dense, hazard-free
        nursery env (agent spawns in the reef refuge band). The goal pipeline
        is UNFROZEN and, every step, update_z_goal is fed a FORCED supra-
        threshold benefit (scaffold_stage0_forced_benefit) + forced drive --
        the agent is "fed" regardless of whether it actually contacts a
        resource. This decouples z_goal FORMATION from survival/foraging skill
        and proves the goal stream lights when fed (the positive control the
        603e/626a autopsy says is necessary before mature autonomous goal
        formation can be fairly tested). E1+E2 also warm up on the safe nursery
        structure.

        Requires the agent to be built with z_goal_enabled=True (else
        agent.goal_state is None and update_z_goal early-returns) -- the same
        two-part-fix precondition the V3-EXQ-603e config carries.
        """
        if not self.enabled:
            self._stage0_result = Stage0NurseryResult(
                n_episodes=0,
                mean_forced_benefit=0.0,
                z_goal_norm_peak=0.0,
                z_goal_formed=False,
                aborted=True,
                abort_reason="master_switch_off",
            )
            return self._stage0_result

        if not self.cfg.scaffold_stage0_enabled:
            self._stage0_result = Stage0NurseryResult(
                n_episodes=0,
                mean_forced_benefit=0.0,
                z_goal_norm_peak=0.0,
                z_goal_formed=False,
                aborted=True,
                abort_reason="stage0_disabled",
            )
            return self._stage0_result

        # Decoupling-from-survival positive control needs a live goal_state.
        if getattr(agent, "goal_state", None) is None:
            self._stage0_result = Stage0NurseryResult(
                n_episodes=0,
                mean_forced_benefit=0.0,
                z_goal_norm_peak=0.0,
                z_goal_formed=False,
                aborted=True,
                abort_reason="goal_state_none_set_z_goal_enabled_true",
            )
            return self._stage0_result

        self._apply_goal_seeding_calibration(agent)
        _set_goal_pipeline_frozen(agent, frozen=False)
        env = _build_env(self.cfg, phase="stage0")
        agent.train()

        world_dim = agent.config.latent.world_dim
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=self.cfg.scaffold_lr_e1)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=self.cfg.scaffold_lr_e2_wf,
        )
        wf_buf: Deque = deque(maxlen=self.cfg.scaffold_wf_buf_max)

        goal_peaks: List[float] = []
        n_eps = max(1, self.cfg.scaffold_stage0_episode_budget)
        for _ep in range(n_eps):
            self._train_episode(
                agent, env, device, e1_opt, wf_opt, wf_buf, world_dim,
                seed_goal=True,
                forced_benefit=self.cfg.scaffold_stage0_forced_benefit,
                forced_drive=self.cfg.scaffold_stage0_forced_drive,
                goal_peak_sink=goal_peaks,
            )

        peak = float(max(goal_peaks)) if goal_peaks else 0.0
        # SD-057 formation readout: how many incentive-token types the nursery
        # laid down. 0 == empty bank (the 638 cue-silent root cause); non-zero
        # only when scaffold_stage0_bind_incentive_token bound to perceived types.
        _gs = getattr(agent, "goal_state", None)
        _bank = getattr(_gs, "incentive_bank", None) if _gs is not None else None
        token_bank_size_end = len(getattr(_bank, "_base_value", {})) if _bank is not None else 0
        self._stage0_result = Stage0NurseryResult(
            n_episodes=n_eps,
            mean_forced_benefit=float(self.cfg.scaffold_stage0_forced_benefit),
            z_goal_norm_peak=peak,
            z_goal_formed=peak > float(self.cfg.scaffold_stage0_z_goal_peak_gate),
            aborted=False,
            token_bank_size_end=int(token_bank_size_end),
        )
        return self._stage0_result

    # ---------------- Stage 0b: protected consolidation ---------------- #

    def run_stage0b_consolidation(
        self, agent, device: torch.device, stage0_baseline_norm: Optional[float] = None
    ) -> Stage0bConsolidationResult:
        """
        Stage 0b: PROTECTED consolidation of the just-formed Stage-0 z_goal.

        Developmental-window amend (2026-06-03b). Routed by the V3-EXQ-634
        design-error review: after Stage-0 feeds the infant and lights z_goal,
        the prior scaffold exposed that fragile trace straight to P1's
        every-step decay-only update_z_goal calls (washout before ecological
        contact). This short window runs in the safe nursery env with E1/E2
        training still open but update_z_goal NOT called, so the persistent
        attractor cannot be washed out by decay-only updating. The retention
        ratio is the developmental acceptance metric: feeding must be followed
        by consolidation, not immediate exposure to decay-only updating.

        Gated by scaffold_developmental_window_enabled AND scaffold_stage0b_enabled
        (both default False -> this phase is never run unless an experiment opts
        in; bit-identical to the pre-amend scheduler otherwise).

        stage0_baseline_norm: optional Stage-0 peak to measure retention
        against (per the ">=0.75 of Stage-0 peak" acceptance). Defaults to the
        z_goal norm read at Stage-0b entry.
        """
        gate = float(self.cfg.scaffold_stage0b_retention_gate)
        if not self.enabled:
            self._stage0b_result = Stage0bConsolidationResult(
                n_episodes=0, z_goal_norm_start=0.0, z_goal_norm_end=0.0,
                retention_ratio=0.0, retention_gate_passed=False,
                aborted=True, abort_reason="master_switch_off",
            )
            return self._stage0b_result
        if not (self.cfg.scaffold_developmental_window_enabled
                and self.cfg.scaffold_stage0b_enabled):
            self._stage0b_result = Stage0bConsolidationResult(
                n_episodes=0, z_goal_norm_start=0.0, z_goal_norm_end=0.0,
                retention_ratio=0.0, retention_gate_passed=False,
                aborted=True, abort_reason="stage0b_disabled",
            )
            return self._stage0b_result
        if getattr(agent, "goal_state", None) is None:
            self._stage0b_result = Stage0bConsolidationResult(
                n_episodes=0, z_goal_norm_start=0.0, z_goal_norm_end=0.0,
                retention_ratio=0.0, retention_gate_passed=False,
                aborted=True, abort_reason="goal_state_none_set_z_goal_enabled_true",
            )
            return self._stage0b_result

        def _norm() -> float:
            gs = agent.goal_state
            try:
                return float(gs.goal_norm())
            except TypeError:
                return float(gs.goal_norm)

        start_norm = _norm()
        baseline = float(stage0_baseline_norm) if stage0_baseline_norm is not None else start_norm

        # Protected window: goal pipeline bridge frozen + update_z_goal NOT
        # called (seed_goal=False) -> no decay-only washout. Encoder/E2 keep
        # training on the safe nursery so consolidation is not idle.
        _set_goal_pipeline_frozen(agent, frozen=True)
        env = _build_env(self.cfg, phase="stage0")
        agent.train()
        world_dim = agent.config.latent.world_dim
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=self.cfg.scaffold_lr_e1)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=self.cfg.scaffold_lr_e2_wf,
        )
        wf_buf: Deque = deque(maxlen=self.cfg.scaffold_wf_buf_max)

        n_eps = max(1, self.cfg.scaffold_stage0b_episode_budget)
        for _ep in range(n_eps):
            # seed_goal=False -> update_z_goal is never called -> z_goal is
            # protected from decay-only washout (the whole point of Stage-0b).
            self._train_episode(
                agent, env, device, e1_opt, wf_opt, wf_buf, world_dim,
                seed_goal=False,
            )

        end_norm = _norm()
        retention = (end_norm / baseline) if baseline > 1e-9 else 0.0
        self._stage0b_result = Stage0bConsolidationResult(
            n_episodes=n_eps,
            z_goal_norm_start=start_norm,
            z_goal_norm_end=end_norm,
            retention_ratio=retention,
            retention_gate_passed=(retention >= gate),
            aborted=False,
        )
        return self._stage0b_result

    # ---------------- P0 ---------------- #

    def run_p0(self, agent, device: torch.device) -> P0OnboardingResult:
        """
        Phase 0: train E1+E2 on the scaffolded SD-054 env with goal
        pipeline frozen. Encoder + E2 + E3 warm up on the reef refuge
        substrate while the agent spawns inside the reef band.
        """
        if not self.enabled:
            self._p0_result = P0OnboardingResult(
                n_episodes=0,
                mean_episode_length=0.0,
                final_running_variance=float(getattr(agent.e3, "_running_variance", 0.0)),
                aborted=True,
                abort_reason="master_switch_off",
            )
            return self._p0_result

        _set_goal_pipeline_frozen(agent, frozen=True)
        env = _build_env(self.cfg, phase="p0")
        agent.train()

        world_dim = agent.config.latent.world_dim
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=self.cfg.scaffold_lr_e1)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=self.cfg.scaffold_lr_e2_wf,
        )
        wf_buf: Deque = deque(maxlen=self.cfg.scaffold_wf_buf_max)

        ep_lengths: List[int] = []
        rv_final = float(getattr(agent.e3, "_running_variance", 0.0))
        for ep in range(self.cfg.scaffold_p0_episode_budget):
            ep_len = self._train_episode(
                agent, env, device, e1_opt, wf_opt, wf_buf, world_dim
            )
            ep_lengths.append(ep_len)
            rv_final = float(getattr(agent.e3, "_running_variance", rv_final))

        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        self._p0_result = P0OnboardingResult(
            n_episodes=len(ep_lengths),
            mean_episode_length=mean_len,
            final_running_variance=rv_final,
            aborted=False,
        )
        return self._p0_result

    # ---------------- Stage-H: isolated hazard avoidance ---------------- #

    def run_hazard_avoidance(self, agent, device: torch.device) -> HazardAvoidanceResult:
        """
        Stage-H: isolated hazard-avoidance training (2026-06-07 curriculum-
        decomposition amend; V3-EXQ-603f autopsy).

        Inserted between P0 (safe goal-frozen warm-up) and P1 (combined wean).
        The goal pipeline is FROZEN (the isolation: no goal-unfreeze competing),
        hazards are present at the target density / proximity_harm, foraging
        pressure is minimal, and hazard_food_attraction=0 so foraging does not
        raise hazard exposure -- the policy learns avoidance on its own. E1+E2
        (+E3 running-variance) train exactly as in run_p0; the agent's E3 harm
        evaluation drives survival without the goal pipeline. P1 is then entered
        by an already-survival-competent policy.

        Survival readout: median episode length over the last
        scaffold_hazard_stage_stability_window episodes vs
        scaffold_hazard_stage_survival_gate_steps (G_H). DIAGNOSTIC only -- it
        does NOT abort the curriculum and does NOT change the canonical
        G0/G1/G2/G3 readiness gate; it lets the manifest confirm the isolated
        stage achieved avoidance before P1.

        Gated by scaffold_hazard_stage_enabled (default False -> aborts
        disabled). Bit-identical OFF: an experiment that does not set the flag
        AND call this method sees the pre-amend curriculum.
        """
        if not self.enabled:
            self._hazard_result = HazardAvoidanceResult(
                n_episodes=0, mean_episode_length=0.0,
                median_last_window_episode_length=0.0, survival_gate_passed=False,
                final_running_variance=float(getattr(agent.e3, "_running_variance", 0.0)),
                aborted=True, abort_reason="master_switch_off",
            )
            return self._hazard_result
        if not self.cfg.scaffold_hazard_stage_enabled:
            self._hazard_result = HazardAvoidanceResult(
                n_episodes=0, mean_episode_length=0.0,
                median_last_window_episode_length=0.0, survival_gate_passed=False,
                final_running_variance=float(getattr(agent.e3, "_running_variance", 0.0)),
                aborted=True, abort_reason="hazard_stage_disabled",
            )
            return self._hazard_result

        # Goal pipeline FROZEN (isolation): mech295/mech307 short-circuit + no
        # update_z_goal call (seed_goal=False) -> z_goal is untouched here.
        _set_goal_pipeline_frozen(agent, frozen=True)
        env = _build_env(self.cfg, phase="hazard")
        agent.train()

        world_dim = agent.config.latent.world_dim
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=self.cfg.scaffold_lr_e1)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=self.cfg.scaffold_lr_e2_wf,
        )
        wf_buf: Deque = deque(maxlen=self.cfg.scaffold_wf_buf_max)

        n_eps = max(1, self.cfg.scaffold_hazard_stage_episode_budget)
        ep_lengths: List[int] = []
        recent_lengths: Deque[int] = deque(
            maxlen=self.cfg.scaffold_hazard_stage_stability_window
        )
        rv_final = float(getattr(agent.e3, "_running_variance", 0.0))
        for _ep in range(n_eps):
            ep_len = self._train_episode(
                agent, env, device, e1_opt, wf_opt, wf_buf, world_dim,
                seed_goal=False,  # goal frozen -> survival learned in isolation
            )
            ep_lengths.append(ep_len)
            recent_lengths.append(ep_len)
            rv_final = float(getattr(agent.e3, "_running_variance", rv_final))

        mean_len = float(np.mean(ep_lengths)) if ep_lengths else 0.0
        median_last_window = (
            float(np.median(list(recent_lengths))) if recent_lengths else 0.0
        )
        survival_passed = median_last_window >= float(
            self.cfg.scaffold_hazard_stage_survival_gate_steps
        )
        self._hazard_result = HazardAvoidanceResult(
            n_episodes=len(ep_lengths),
            mean_episode_length=mean_len,
            median_last_window_episode_length=median_last_window,
            survival_gate_passed=survival_passed,
            final_running_variance=rv_final,
            aborted=False,
            episode_lengths=ep_lengths,
        )
        return self._hazard_result

    # ---------------- P1 ---------------- #

    def run_p1(self, agent, device: torch.device) -> P1OnboardingResult:
        """
        Phase 1: anneal env hazard parameters + goal-pipeline gates from P0
        floor to P2 ceiling across the P1 window. Spawn admissibility
        narrows back to the SD-054 default midline band. End-of-P1 survival
        gate (median episode length over the last stability_window) gates
        admission to P2.
        """
        if not self.enabled:
            self._p1_result = P1OnboardingResult(
                n_episodes=0,
                median_last_window_episode_length=0.0,
                survival_gate_passed=False,
                final_hazard_food_attraction=0.0,
                final_mech295_min_drive_to_fire=0.0,
                final_mech307_conjunction_z_beta_threshold=0.0,
                aborted=True,
                abort_reason="master_switch_off",
            )
            return self._p1_result

        self._apply_goal_seeding_calibration(agent)
        _set_goal_pipeline_frozen(agent, frozen=False)
        agent.train()

        world_dim = agent.config.latent.world_dim
        e1_opt = optim.Adam(list(agent.e1.parameters()), lr=self.cfg.scaffold_lr_e1)
        wf_opt = optim.Adam(
            list(agent.e2.world_transition.parameters())
            + list(agent.e2.world_action_encoder.parameters()),
            lr=self.cfg.scaffold_lr_e2_wf,
        )
        wf_buf: Deque = deque(maxlen=self.cfg.scaffold_wf_buf_max)

        n_eps = max(1, self.cfg.scaffold_p1_episode_budget)
        recent_lengths: Deque[int] = deque(maxlen=self.cfg.scaffold_p1_stability_window)
        all_episode_lengths: List[int] = []
        last_anneal_t = 0.0
        hold = max(0.0, min(0.95, float(self.cfg.scaffold_p1_anneal_hold_fraction)))
        # Developmental-window contact-gating: active only when both the master
        # switch and the lever are set (default OFF -> legacy every-step decay).
        contact_gated = bool(
            self.cfg.scaffold_developmental_window_enabled
            and self.cfg.scaffold_contact_gated_goal_updates
        )
        contact_threshold = float(self.cfg.scaffold_p2_contact_benefit_threshold)
        # Foraging-competence residual: derive the gating floor from the live
        # GoalConfig when auto-reconcile is on (so the seed/skip decision tracks
        # GoalState's actual firing); else the static 634c path.
        gating_threshold = self._effective_gating_threshold(agent)
        # Graded spawn weaning: keep the reef-refuge spawn for the first
        # `reef_hold` fraction of P1 (then midline). 0.0 == legacy all-midline.
        reef_hold = max(0.0, min(0.95, float(self.cfg.scaffold_p1_reef_spawn_hold_fraction)))
        n_reef_spawn = 0
        goal_write_diag = _new_goal_write_diag()
        cue_diag = _new_cue_diag()
        for ep in range(n_eps):
            raw_t = ep / max(1, n_eps - 1) if n_eps > 1 else 1.0
            # Staged withdrawal: hold full nursery relaxation (anneal_t=0) for
            # the first `hold` fraction of P1, then ramp linearly to 1.0.
            if hold > 0.0:
                anneal_t = 0.0 if raw_t <= hold else (raw_t - hold) / (1.0 - hold)
            else:
                anneal_t = raw_t
            _set_p1_anneal_state(agent, self.cfg, anneal_t)
            # Graded spawn weaning: reef-refuge spawn while raw_t is inside the
            # held fraction, then switch to the midline band (legacy P1).
            spawn_in_reef = reef_hold > 0.0 and raw_t <= reef_hold
            if spawn_in_reef:
                n_reef_spawn += 1
            env = _build_env(self.cfg, phase="p1", anneal_t=anneal_t,
                             p1_spawn_in_reef_half=spawn_in_reef)
            ep_len = self._train_episode(
                agent, env, device, e1_opt, wf_opt, wf_buf, world_dim,
                seed_goal=True,
                contact_gated=contact_gated,
                contact_threshold=contact_threshold,
                gating_threshold=gating_threshold,
                goal_write_diag=goal_write_diag,
                cue_diag=cue_diag,
            )
            all_episode_lengths.append(ep_len)
            recent_lengths.append(ep_len)
            last_anneal_t = anneal_t

        median_last_window = (
            float(np.median(list(recent_lengths))) if recent_lengths else 0.0
        )
        survival_passed = median_last_window >= float(
            self.cfg.scaffold_p1_survival_gate_steps
        )

        final_hfa = _lerp(
            self.cfg.scaffold_p1_anneal_hazard_food_attraction_min,
            self.cfg.scaffold_p1_anneal_hazard_food_attraction_max,
            last_anneal_t,
        )
        final_min_drive = _lerp(
            self.cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_max,
            self.cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_min,
            last_anneal_t,
        )
        final_z_beta = _lerp(
            self.cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_max,
            self.cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_min,
            last_anneal_t,
        )

        self._p1_result = P1OnboardingResult(
            n_episodes=n_eps,
            median_last_window_episode_length=median_last_window,
            survival_gate_passed=survival_passed,
            final_hazard_food_attraction=final_hfa,
            final_mech295_min_drive_to_fire=final_min_drive,
            final_mech307_conjunction_z_beta_threshold=final_z_beta,
            aborted=False,
            abort_reason="" if survival_passed else "p1_survival_gate_failed",
            episode_lengths=all_episode_lengths,
            n_contact_refresh_updates=goal_write_diag["n_contact_refresh"],
            n_decay_only_updates=goal_write_diag["n_decay_only"],
            n_skipped_protected_updates=goal_write_diag["n_skipped_protected"],
            contact_gated=contact_gated,
            n_reef_spawn_episodes=n_reef_spawn,
            reconciled_gating_threshold=float(gating_threshold),
            n_cue_recall_fires=int(goal_write_diag.get("n_cue_recall_fires", 0)),
            cue_diag=dict(cue_diag),
        )
        return self._p1_result

    # ---------------- P2 ---------------- #

    def run_p2(self, agent, device: torch.device) -> P2OnboardingMetrics:
        """
        Phase 2: frozen-policy measurement on the full target env.

        Records z_goal_norm_peak, approach_commit_rate, bridge_cue_fires,
        and dacc_bias_nonzero_steps per the substrate-design memo
        Acceptance section.
        """
        if not self.enabled:
            self._p2_metrics = P2OnboardingMetrics(
                n_episodes=0,
                z_goal_norm_peak_per_episode=[],
                z_goal_norm_peak_max=0.0,
                approach_commit_steps=0,
                approach_commit_rate=0.0,
                bridge_cue_fires=0,
                dacc_bias_nonzero_steps=0,
                mean_episode_length=0.0,
                per_episode=[],
                contact_steps=0,
                contact_rate=0.0,
                hazard_food_attraction_used=0.0,
            )
            return self._p2_metrics

        self._apply_goal_seeding_calibration(agent)
        p2_hfa_used = (
            self.cfg.scaffold_p2_hazard_food_attraction_guard
            if self.cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
            else self.cfg.scaffold_p2_hazard_food_attraction
        )
        env = _build_env(self.cfg, phase="p2")
        agent.eval()

        # Developmental-window contact-gating for the measurement window: active
        # only when both the master switch and the lever are set (default OFF ->
        # legacy every-step decay-only path; peak still captured via the
        # pre-update read in _eval_episode).
        contact_gated = bool(
            self.cfg.scaffold_developmental_window_enabled
            and self.cfg.scaffold_contact_gated_goal_updates
        )
        contact_threshold = float(self.cfg.scaffold_p2_contact_benefit_threshold)
        # Foraging-competence residual: reconciled (GoalConfig-derived) gating
        # floor when auto-reconcile is on, else the static 634c fallback.
        gating_threshold = self._effective_gating_threshold(agent)

        per_episode: List[Dict[str, Any]] = []
        peak_per_ep: List[float] = []
        total_approach_commit = 0
        total_bridge_cue = 0
        total_dacc_bias_nonzero = 0
        total_steps = 0
        total_contact = 0
        total_contact_refresh = 0
        total_decay_only = 0
        total_skipped_protected = 0
        contact_peak_max = 0.0
        total_contact_events = 0
        total_cue_recall_fires = 0
        cue_diag = _new_cue_diag()
        # V3-EXQ-640 post-cue instrumentation accumulator (None -> bit-identical).
        post_cue_diag = (
            _new_post_cue_diag(self.cfg.scaffold_post_cue_window_steps)
            if self.cfg.scaffold_post_cue_instrumentation
            else None
        )
        for ep in range(self.cfg.scaffold_p2_episode_budget):
            ep_metrics = self._eval_episode(
                agent, env, device,
                contact_gated=contact_gated,
                contact_threshold=contact_threshold,
                gating_threshold=gating_threshold,
                cue_diag=cue_diag,
                post_cue_diag=post_cue_diag,
            )
            per_episode.append(ep_metrics)
            peak_per_ep.append(ep_metrics["z_goal_norm_peak"])
            total_approach_commit += int(ep_metrics["approach_commit_steps"])
            total_bridge_cue += int(ep_metrics["bridge_cue_fires"])
            total_dacc_bias_nonzero += int(ep_metrics["dacc_bias_nonzero_steps"])
            total_steps += int(ep_metrics["episode_length"])
            total_contact += int(ep_metrics["contact_steps"])
            total_contact_refresh += int(ep_metrics.get("n_contact_refresh_updates", 0))
            total_decay_only += int(ep_metrics.get("n_decay_only_updates", 0))
            total_skipped_protected += int(ep_metrics.get("n_skipped_protected_updates", 0))
            contact_peak_max = max(
                contact_peak_max, float(ep_metrics.get("z_goal_norm_at_contact_peak", 0.0))
            )
            total_contact_events += int(ep_metrics.get("num_contact_events", 0))
            total_cue_recall_fires += int(ep_metrics.get("n_cue_recall_fires", 0))

        n_eps = max(1, len(per_episode))
        peak_max = float(max(peak_per_ep)) if peak_per_ep else 0.0
        approach_rate = (
            float(total_approach_commit) / float(total_steps) if total_steps else 0.0
        )
        mean_len = float(total_steps) / float(n_eps)

        self._p2_metrics = P2OnboardingMetrics(
            n_episodes=len(per_episode),
            z_goal_norm_peak_per_episode=peak_per_ep,
            z_goal_norm_peak_max=peak_max,
            approach_commit_steps=total_approach_commit,
            approach_commit_rate=approach_rate,
            bridge_cue_fires=total_bridge_cue,
            dacc_bias_nonzero_steps=total_dacc_bias_nonzero,
            mean_episode_length=mean_len,
            per_episode=per_episode,
            contact_steps=total_contact,
            contact_rate=(float(total_contact) / float(total_steps) if total_steps else 0.0),
            hazard_food_attraction_used=float(p2_hfa_used),
            reconciled_gating_threshold=float(gating_threshold),
            n_contact_refresh_updates=total_contact_refresh,
            n_decay_only_updates=total_decay_only,
            n_skipped_protected_updates=total_skipped_protected,
            contact_gated=contact_gated,
            z_goal_norm_at_contact_peak=contact_peak_max,
            num_contact_events=total_contact_events,
            n_cue_recall_fires=total_cue_recall_fires,
            cue_diag=dict(cue_diag),
            post_cue_diag=(dict(post_cue_diag) if post_cue_diag is not None else {}),
        )
        return self._p2_metrics

    # ---------------- Episode loops ---------------- #

    def _train_episode(
        self,
        agent,
        env,
        device: torch.device,
        e1_opt,
        wf_opt,
        wf_buf: Deque,
        world_dim: int,
        seed_goal: bool = False,
        forced_benefit: Optional[float] = None,
        forced_drive: Optional[float] = None,
        goal_peak_sink: Optional[List[float]] = None,
        contact_gated: bool = False,
        contact_threshold: float = 0.0,
        gating_threshold: Optional[float] = None,
        goal_write_diag: Optional[Dict[str, int]] = None,
        cue_diag: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        One training episode. Returns realised episode length in steps.

        Follows the committed_mode_curriculum._one_episode_train pattern:
        env.reset() returns (_, obs_dict); obs_dict carries body_state +
        world_state torch tensors; agent.sense(body, world) -> LatentState;
        ticks -> generate_trajectories -> select_action -> env.step.

        seed_goal: when True, call agent.update_z_goal(benefit, drive) after
        each env.step using the post-step body-state, mirroring the reference
        goal-stream runner (goal_stream_stages_sd054.py:537). Set True only in
        P1 (goal pipeline UNFROZEN); left False in P0 so the encoder/E2/E3
        warm-up is not gated by goal-pipeline writes (the documented P0
        design). Wiring this call is the V3-EXQ-603d / 625b harness-fix: the
        scheduler previously never reached GoalState.update, so z_goal stayed
        zero-init across every step of every arm.

        forced_benefit / forced_drive (2026-06-03 Stage-0 nursery amend): when
        seed_goal is True AND forced_benefit is not None, feed these FORCED
        values to update_z_goal instead of reading the post-step body-state.
        This is the infant-REE forced-feeding path -- it decouples z_goal
        formation from foraging/survival competence (the agent need not contact
        a resource to be "fed"). goal_peak_sink, if provided, accumulates the
        per-step goal_norm so the caller can report a Stage-0 z_goal peak.
        """
        _, obs_dict = env.reset()
        agent.reset()

        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for step in range(self.cfg.scaffold_steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)
            z_world_curr = latent.z_world.detach()

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append(
                    (z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu())
                )

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            # E1 prediction loss.
            e1_opt.zero_grad()
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.e1.parameters()), 1.0)
                e1_opt.step()

            # E2 world-forward loss drives running_variance toward convergence.
            if len(wf_buf) >= self.cfg.scaffold_batch_size:
                k = min(self.cfg.scaffold_batch_size, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_opt.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_opt.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            z_world_prev = z_world_curr
            action_prev = action.detach()
            _, _harm_signal, done, _, obs_dict = env.step(action_idx)

            # Goal-pipeline seeding (P1 only): drive z_goal from the post-step
            # body-state, mirroring goal_stream_stages_sd054.py:537. Without this
            # call GoalState.update is never reached and z_goal stays zero-init
            # (the V3-EXQ-603d / 625b harness-fix root cause). Gated to P1 via
            # seed_goal so P0 warm-up stays goal-pipeline-frozen by design.
            if seed_goal:
                is_forced = forced_benefit is not None
                if is_forced:
                    # Forced-feed (Stage-0 nursery): supra-threshold benefit
                    # regardless of actual resource contact -> decouples z_goal
                    # formation from foraging/survival.
                    benefit = float(forced_benefit)
                    drive = (
                        float(forced_drive)
                        if forced_drive is not None
                        else _benefit_and_drive(obs_dict["body_state"].to(device))[1]
                    )
                else:
                    benefit, drive = _benefit_and_drive(
                        obs_dict["body_state"].to(device)
                    )
                # Contact-RATE readout (g2 "was the infant fed at all").
                is_contact = benefit > contact_threshold
                if goal_write_diag is not None and is_contact:
                    goal_write_diag["n_contact_steps"] += 1
                # Contact-GATING (skip/update) decision keyed off a SEPARATE
                # threshold (V3-EXQ-634b seeding-calibration amend). When the
                # gating threshold > readout threshold, sub-seeding whiffs in the
                # band (readout_floor, gating_floor) are PROTECTED (skipped) so
                # they do not decay-only erode the consolidated trace; only
                # benefit that actually clears the GoalState seeding floor writes.
                # Default (gating_threshold None) falls back to contact_threshold
                # -> bit-identical to the pre-amend 634b path.
                gate = contact_threshold if gating_threshold is None else gating_threshold
                seeds = benefit > gate
                # Developmental-window contact-gating (2026-06-03b): when
                # contact-gated, skip update_z_goal on a sub-seeding step so the
                # persistent attractor is NOT washed out by a decay-only call.
                # Forced-feed (Stage-0) always writes. Default (contact_gated
                # False) keeps the legacy every-step decay-only path.
                # SD-057 L2: bind benefit to object identity (no-op when bridge
                # off -> rt None -> bank None or ignored). In the forced-feed
                # nursery this lays down per-object tokens cue-recall later reads.
                # FORMATION FIX (V3-EXQ-638, 2026-06-04): forced feeding is
                # decoupled from standing on a typed cell, so _contacted_resource_type
                # is almost always None during Stage-0 -> the bank.update bind
                # (gated resource_type>0) is never reached -> empty bank -> cue
                # silent in the wild. When scaffold_stage0_bind_incentive_token is
                # on, bind the forced-feed token to the STRONGEST-PERCEIVED type
                # instead (the infant is fed; bind to whatever food it perceives),
                # using the SAME perceptual primitive the wild cue reads. Default
                # off -> rt = _contacted_resource_type, bit-identical to pre-fix.
                if not self.cfg.scaffold_cue_recall_bridge_enabled:
                    rt = None
                elif is_forced and self.cfg.scaffold_stage0_bind_incentive_token:
                    _bt, _bp = _strongest_perceived_type(env, obs_dict)
                    rt = _bt if _bt > 0 else None
                else:
                    rt = _contacted_resource_type(obs_dict)
                if contact_gated and not is_forced and not seeds:
                    if goal_write_diag is not None:
                        goal_write_diag["n_skipped_protected"] += 1
                else:
                    agent.update_z_goal(
                        benefit_exposure=benefit, drive_level=drive, resource_type=rt
                    )
                    if goal_write_diag is not None:
                        if seeds or is_forced:
                            goal_write_diag["n_contact_refresh"] += 1
                        else:
                            goal_write_diag["n_decay_only"] += 1
                    if goal_peak_sink is not None:
                        gs = getattr(agent, "goal_state", None)
                        if gs is not None and hasattr(gs, "goal_norm"):
                            try:
                                gn = float(gs.goal_norm())
                            except TypeError:
                                gn = float(gs.goal_norm)
                            goal_peak_sink.append(gn)
                # SD-057 L6: cue-recall on ecological (non-forced) wild steps --
                # a perceived resource cue retrieves its token and pulls z_goal
                # toward it before contact (the wean-to-wild approach bridge).
                if not is_forced:
                    n_cue = _maybe_cue_recall(
                        agent, env, obs_dict, drive, self.cfg, diag=cue_diag
                    )
                    if goal_write_diag is not None and n_cue:
                        goal_write_diag["n_cue_recall_fires"] = (
                            goal_write_diag.get("n_cue_recall_fires", 0) + n_cue
                        )

            if done:
                return step + 1
        return self.cfg.scaffold_steps_per_episode

    def _eval_episode(
        self,
        agent,
        env,
        device: torch.device,
        contact_gated: bool = False,
        contact_threshold: float = 1e-6,
        gating_threshold: Optional[float] = None,
        cue_diag: Optional[Dict[str, Any]] = None,
        post_cue_diag: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        One eval episode: policy frozen (no optimizer steps), env at target
        config. Measures the P2 acceptance metrics per the substrate-design
        memo Acceptance section.

        post_cue_diag (V3-EXQ-640): when a _new_post_cue_diag() accumulator is
        supplied, per-cue-fire windowed measurements are recorded (z_goal-norm/
        pull delta around each cue, cue->action approach rate, manhattan-distance
        gradient, hazard-interrupt count, first-improving-move latency, oscillation
        rate). PURELY READ-ONLY -- it never alters sensing / selection / stepping.
        None (default) -> the instrumentation block is skipped, bit-identical.

        contact_gated (2026-06-03b): when True, update_z_goal is only called on a
        seeding step; sub-seeding steps are skipped so the measured z_goal
        reflects the RETAINED + ecologically-refreshed attractor, not decay-only
        erosion. Default False = legacy every-step path (peak still captured via
        the pre-update read).

        gating_threshold (V3-EXQ-634b seeding-calibration amend): the SEPARATE
        benefit floor used for the skip/update decision (a step seeds when
        benefit > gating_threshold). Decoupled from contact_threshold (the
        contact-RATE readout). None -> falls back to the readout threshold
        (bit-identical). z_goal_norm_at_contact_peak reads z_goal AT a genuine
        seeding event (632-style), NOT the frozen forced-feed-calibrated peak.
        """
        _, obs_dict = env.reset()
        agent.reset()

        world_dim = agent.config.latent.world_dim
        z_goal_norm_peak = 0.0
        z_goal_norm_at_contact_peak = 0.0
        num_contact_events = 0
        approach_commit_steps = 0
        contact_steps = 0
        n_contact_refresh = 0
        n_decay_only = 0
        n_skipped_protected = 0
        n_cue_recall_fires = 0  # SD-057 L6 cue-recall fires this episode
        bridge_cue_fires_baseline = 0
        bridge_cue_fires_final = 0
        dacc_bias_nonzero_steps_baseline = 0
        dacc_bias_nonzero_steps_final = 0
        ep_len = 0

        bridge = getattr(agent, "mech295_bridge", None)
        if bridge is not None:
            bridge_cue_fires_baseline = int(getattr(bridge, "_n_cue_fires", 0))
        dacc = getattr(agent, "dacc", None)
        # dACC bias is tracked per-step by integration sites (no internal counter
        # on the module itself); fall back to per-tick checking via _last_bundle.
        dacc_bias_nonzero_local = 0

        # V3-EXQ-640 post-cue instrumentation state (read-only). active_windows
        # holds one entry per recent cue fire; each is aged by the per-step move
        # and folded into post_cue_diag at the window horizon.
        instr = post_cue_diag is not None
        pc_window = max(1, int(self.cfg.scaffold_post_cue_window_steps))
        active_windows: List[Dict[str, Any]] = []
        prev_move_idx: Optional[int] = None

        for step in range(self.cfg.scaffold_steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            # z_goal peak (probably the most informative acceptance metric).
            goal_state = getattr(agent, "goal_state", None)
            if goal_state is not None and hasattr(goal_state, "goal_norm"):
                try:
                    cur = float(goal_state.goal_norm())
                except TypeError:
                    cur = float(goal_state.goal_norm)
                if cur > z_goal_norm_peak:
                    z_goal_norm_peak = cur

            beta_gate = getattr(agent, "beta_gate", None)
            if beta_gate is not None and getattr(beta_gate, "is_elevated", False):
                approach_commit_steps += 1

            # dACC bias nonzero step-wise: bundle is populated each select_action tick.
            if dacc is not None:
                bundle = getattr(dacc, "_last_bundle", None)
                if bundle is not None:
                    sb = bundle.get("mode_ev") or bundle.get("harm_interaction")
                    if sb is not None:
                        try:
                            if float(torch.as_tensor(sb).norm().item()) > 1e-6:
                                dacc_bias_nonzero_local += 1
                        except Exception:
                            pass

            action_idx = int(action.argmax(dim=-1).item())
            # V3-EXQ-640: nearest-resource distance BEFORE the move (read-only).
            dist_before = _nearest_resource_manhattan(env) if instr else None
            _, _harm_signal, done, _, obs_dict = env.step(action_idx)
            ep_len = step + 1

            # V3-EXQ-640 post-cue movement attribution: age every open cue window
            # with THIS step's move (whether it reduced distance to food, whether a
            # harm spike landed, whether the move reversed direction), then retire
            # windows that reached the look-ahead horizon. Cue windows opened LATER
            # this step (in the cue block below) are first aged next step, so the
            # firing step's own move is never miscounted as a post-cue move.
            if instr:
                dist_after = _nearest_resource_manhattan(env)
                move_eval = dist_before is not None and dist_after is not None
                move_improved = move_eval and dist_after < dist_before
                hazard_spike = abs(float(_harm_signal)) > 1e-6
                reversal = (
                    prev_move_idx is not None
                    and _opposite_action(env, action_idx, prev_move_idx)
                )
                post_cue_diag["n_steps_total"] += 1
                if move_eval:
                    post_cue_diag["n_move_eval_steps"] += 1
                    if move_improved:
                        post_cue_diag["sum_move_improved_all_steps"] += 1
                if active_windows:
                    if move_eval:
                        post_cue_diag["n_postcue_eval_steps"] += 1
                        if move_improved:
                            post_cue_diag["sum_move_improved_postcue_steps"] += 1
                    for w in active_windows:
                        w["age"] += 1
                        if move_improved and not w["improved"]:
                            w["improved"] = True
                            w["first_latency"] = w["age"]
                            if w["age"] == 1:
                                w["first_move_approach"] = True
                        if hazard_spike:
                            w["hazard"] = True
                        if reversal:
                            w["osc"] += 1
                    survivors: List[Dict[str, Any]] = []
                    for w in active_windows:
                        if w["age"] >= pc_window:
                            _finalize_post_cue_window(post_cue_diag, w)
                        else:
                            survivors.append(w)
                    active_windows = survivors
                prev_move_idx = action_idx

            # P2 measurement on the trained goal pipeline: seed z_goal from the
            # post-step body-state and re-read the peak (mirrors the reference
            # runner goal_stream_stages_sd054.py:590). The frozen-policy eval
            # does not optimise, but z_goal MUST be driven for the C4
            # z_goal_norm_peak acceptance metric to be non-zero -- the
            # V3-EXQ-603d harness-fix.
            benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(device))
            # Foraging-contact-rate guard: a step where the agent actually
            # received benefit (resource contact) -- distinguishes a z_goal=0
            # read caused by "no contact / infant not fed" from one caused by a
            # genuine goal-formation failure despite contact.
            is_contact = benefit > float(self.cfg.scaffold_p2_contact_benefit_threshold)
            if is_contact:
                contact_steps += 1
            # Contact-GATING (skip/update) decision keyed off the SEPARATE
            # seeding floor (V3-EXQ-634b amend), decoupled from the contact-RATE
            # readout above. A step SEEDS when benefit clears the gating floor;
            # sub-seeding steps are protected (skipped) under contact-gating so
            # the P2 z_goal read reflects the RETAINED + genuinely-refreshed
            # attractor, not decay-only erosion. None -> readout threshold
            # (bit-identical). The pre-update read above already captured the
            # episode-entry retained norm into z_goal_norm_peak either way.
            gate = (
                float(self.cfg.scaffold_p2_contact_benefit_threshold)
                if gating_threshold is None
                else float(gating_threshold)
            )
            seeds = benefit > gate
            # SD-057 L2 bind (no-op when bridge off).
            rt = (
                _contacted_resource_type(obs_dict)
                if self.cfg.scaffold_cue_recall_bridge_enabled
                else None
            )
            if contact_gated and not seeds:
                n_skipped_protected += 1
            else:
                agent.update_z_goal(
                    benefit_exposure=benefit, drive_level=drive, resource_type=rt
                )
                if seeds:
                    n_contact_refresh += 1
                else:
                    n_decay_only += 1
                if goal_state is not None and hasattr(goal_state, "goal_norm"):
                    try:
                        cur = float(goal_state.goal_norm())
                    except TypeError:
                        cur = float(goal_state.goal_norm)
                    if cur > z_goal_norm_peak:
                        z_goal_norm_peak = cur
                    # Consumption-event-gated readout: z_goal AT a genuine
                    # seeding event (632-style num_contact_events), the fair G3
                    # input. Stays 0.0 when wild contact never clears the seeding
                    # floor -> a z_goal=0-at-contact read is interpretable rather
                    # than masked by the carried forced-feed nursery trace.
                    if seeds:
                        num_contact_events += 1
                        if cur > z_goal_norm_at_contact_peak:
                            z_goal_norm_at_contact_peak = cur

            # SD-057 L6: cue-recall on the perceived resource each P2 step.
            # V3-EXQ-640: read z_goal immediately before/after the cue so the cue's
            # OWN pull on z_goal is isolated (the displacement test).
            gn_before, zg_before = _read_zgoal(goal_state) if instr else (0.0, None)
            n_cue = _maybe_cue_recall(
                agent, env, obs_dict, drive, self.cfg, diag=cue_diag
            )
            n_cue_recall_fires += n_cue
            if instr:
                gn_after, zg_after = _read_zgoal(goal_state)
                post_cue_diag["sum_zgoal_norm_all_steps"] += gn_after
                post_cue_diag["n_zgoal_norm_all_steps"] += 1
                if n_cue:
                    post_cue_diag["n_cue_fire_steps"] += 1
                    post_cue_diag["sum_post_cue_zgoal_norm_delta"] += (
                        gn_after - gn_before
                    )
                    post_cue_diag["n_post_cue_zgoal_norm_delta"] += 1
                    post_cue_diag["sum_zgoal_norm_at_cue_fire"] += gn_after
                    post_cue_diag["min_zgoal_norm_at_cue_fire"] = min(
                        post_cue_diag["min_zgoal_norm_at_cue_fire"], gn_after
                    )
                    post_cue_diag["max_zgoal_norm_at_cue_fire"] = max(
                        post_cue_diag["max_zgoal_norm_at_cue_fire"], gn_after
                    )
                    if zg_before is not None and zg_after is not None:
                        try:
                            pull = float((zg_after - zg_before).norm().item())
                        except Exception:
                            pull = 0.0
                        post_cue_diag["sum_cue_zgoal_pull_norm"] += pull
                    cab = getattr(agent, "_cue_action_bias", None)
                    if cab is not None:
                        try:
                            post_cue_diag["sum_cue_action_bias_norm"] += float(
                                torch.as_tensor(cab).norm().item()
                            )
                            post_cue_diag["n_cue_action_bias_present"] += 1
                        except Exception:
                            pass
                    # Open a fresh look-ahead window for this cue fire (aged from
                    # next step's move onward).
                    active_windows.append({
                        "age": 0,
                        "improved": False,
                        "first_latency": 0,
                        "first_move_approach": False,
                        "hazard": False,
                        "osc": 0,
                    })

            if done:
                break

        if bridge is not None:
            bridge_cue_fires_final = int(getattr(bridge, "_n_cue_fires", 0))

        # V3-EXQ-640: fold any windows still open at episode end (truncated).
        if instr:
            for w in active_windows:
                _finalize_post_cue_window(post_cue_diag, w)

        return {
            "episode_length": ep_len,
            "z_goal_norm_peak": z_goal_norm_peak,
            "approach_commit_steps": approach_commit_steps,
            "contact_steps": contact_steps,
            "contact_rate": (float(contact_steps) / float(ep_len)) if ep_len else 0.0,
            "bridge_cue_fires": bridge_cue_fires_final - bridge_cue_fires_baseline,
            "dacc_bias_nonzero_steps": dacc_bias_nonzero_local,
            "n_contact_refresh_updates": n_contact_refresh,
            "n_decay_only_updates": n_decay_only,
            "n_skipped_protected_updates": n_skipped_protected,
            "z_goal_norm_at_contact_peak": z_goal_norm_at_contact_peak,
            "num_contact_events": num_contact_events,
            "n_cue_recall_fires": n_cue_recall_fires,  # SD-057 L6
        }


# ---------------------------------------------------------------------------
# Cloner (parallel to committed_mode_curriculum.clone_trained_agent)
# ---------------------------------------------------------------------------


def clone_trained_agent(trained_agent, device: torch.device):
    """
    Clone trained_agent for the V3-EXQ-620 SCAFFOLD_AND_ANNEAL_CONTROL_FROM_SCRATCH
    arm.

    Uses load_state_dict (deepcopy fails on autograd non-leaf tensors).
    Matches the committed_mode_curriculum.clone_trained_agent precedent.
    """
    from ree_core.agent import REEAgent

    cfg_clone = copy.deepcopy(trained_agent.config)
    agent_clone = REEAgent(cfg_clone).to(device)

    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_clone.load_state_dict(state)
    except RuntimeError:
        agent_clone.load_state_dict(state, strict=False)

    if hasattr(trained_agent, "e3") and hasattr(trained_agent.e3, "_running_variance"):
        agent_clone.e3._running_variance = float(trained_agent.e3._running_variance)

    return agent_clone


# ---------------------------------------------------------------------------
# Conceptual feeding-stage plan (explicit per the 2026-06-03 amend)
# ---------------------------------------------------------------------------

# Infant REE must be fed before mature agency is judged. The conceptual
# developmental sequence (recorded in the 603f manifest as `stage_plan`):
#   Stage 0  nursery: actively fed / forced-benefit warmup (run_stage0_nursery)
#   Stage 1  guided to food under low-conflict nursery conditions (run_p0)
#   Stage 2  finds food in easy conditions (early P1, anneal_t held low)
#   Stage 3  finds food under guarded hazard / conflict (late P1 anneal)
#   Stage 4  mature goal-to-action behaviour tested (run_p2, guarded hfa)
STAGE_PLAN: List[Dict[str, str]] = [
    {"stage": "0", "name": "nursery_forced_feed",
     "method": "run_stage0_nursery", "desc": "forced supra-threshold benefit; z_goal formation decoupled from survival"},
    {"stage": "1", "name": "guided_low_conflict",
     "method": "run_p0", "desc": "reef-refuge spawn, hfa=0, goal pipeline frozen; encoder/E2/E3 warm-up"},
    {"stage": "2", "name": "easy_foraging",
     "method": "run_p1 (early anneal)", "desc": "goal pipeline unfrozen, low hazard; ecological contact begins"},
    {"stage": "3", "name": "guarded_hazard",
     "method": "run_p1 (late anneal)", "desc": "hazard_food_attraction + proximity_harm ramp toward target"},
    {"stage": "4", "name": "mature_test",
     "method": "run_p2", "desc": "frozen-policy measurement under guarded hfa; substrate gate evaluated"},
]


def stage_plan() -> List[Dict[str, str]]:
    """Return the conceptual feeding-stage plan for inclusion in the manifest."""
    return [dict(s) for s in STAGE_PLAN]


def _fraction_passing(values: List[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v)) / float(len(values))


def evaluate_substrate_gate(
    stage0_z_goal_peaks_per_seed: List[float],
    p1_survival_pass_per_seed: List[bool],
    p2_z_goal_peaks_per_seed: List[float],
    p2_contact_rates_per_seed: List[float],
    *,
    z_goal_gate: float = 0.4,
    contact_gate: float = 0.0,
    min_fraction: float = 2.0 / 3.0,
) -> Dict[str, Any]:
    """
    Evaluate the four substrate-readiness gates that MUST pass before
    Q-045 / MECH-313 / MECH-260 discrimination is interpreted in V3-EXQ-603f.

    All gates require >= min_fraction (default 2/3) of seeds to pass:
      stage0_positive_control : Stage-0 forced-feed produced z_goal > z_goal_gate
                                (proves the goal stream lights when fed -- if this
                                fails the substrate itself is broken, NOT the claim).
      g1_survival             : P1 survival/foraging gate passed.
      g2_contact              : P2 foraging-contact-rate > contact_gate (infant was
                                actually fed in the measurement window).
      g3_zgoal                : P2 z_goal_norm_peak > z_goal_gate (goal formed
                                ecologically).

    substrate_gate_passed is the conjunction. A same-substrate retest with the
    nursery disabled supplies empty/zero Stage-0 peaks -> stage0_positive_control
    False -> gate cannot pass, so the old path cannot masquerade as 603f.
    """
    stage0_pc = _fraction_passing(
        [p > z_goal_gate for p in stage0_z_goal_peaks_per_seed]
    ) >= min_fraction
    g1 = _fraction_passing(list(p1_survival_pass_per_seed)) >= min_fraction
    g2 = _fraction_passing(
        [r > contact_gate for r in p2_contact_rates_per_seed]
    ) >= min_fraction
    g3 = _fraction_passing(
        [p > z_goal_gate for p in p2_z_goal_peaks_per_seed]
    ) >= min_fraction
    return {
        "substrate_gate_passed": bool(stage0_pc and g1 and g2 and g3),
        "stage0_positive_control": bool(stage0_pc),
        "g1_survival": bool(g1),
        "g2_contact": bool(g2),
        "g3_zgoal": bool(g3),
        "z_goal_gate": float(z_goal_gate),
        "contact_gate": float(contact_gate),
        "min_fraction": float(min_fraction),
    }


def substrate_readiness_from_results(
    stage0_results: List[Stage0NurseryResult],
    p1_results: List[P1OnboardingResult],
    p2_metrics: List[P2OnboardingMetrics],
    *,
    z_goal_gate: float = 0.4,
    contact_gate: float = 0.0,
    min_fraction: float = 2.0 / 3.0,
    use_consumption_gated_g3: bool = True,
) -> Dict[str, Any]:
    """Readiness check from per-seed scheduler results (foraging-competence
    residual, 2026-06-05). The canonical readiness path: it feeds the
    CONSUMPTION-EVENT-GATED z_goal readout (P2OnboardingMetrics.z_goal_norm_at_contact_peak,
    632-style -- z_goal read AT a genuine seeding event) as the G3 input rather
    than the frozen carried-trace peak (z_goal_norm_peak_max). This is the
    "redefine the mature-test z_goal readout to consumption-event-gated" half of
    the residual: a seed that carries an untouched Stage-0 nursery trace through a
    zero-contact P2 (the seed-42 artifact) reads g3=0 here, so G3 cannot be passed
    by a non-foraging seed.

    Set use_consumption_gated_g3=False to fall back to the legacy frozen peak (for
    side-by-side comparison only). Returns the evaluate_substrate_gate dict plus a
    `g3_source` field naming which readout fed G3.
    """
    g3_source = (
        "z_goal_norm_at_contact_peak" if use_consumption_gated_g3
        else "z_goal_norm_peak_max"
    )
    p2_z_goal = [
        float(getattr(m, g3_source)) for m in p2_metrics
    ]
    gate = evaluate_substrate_gate(
        [float(s.z_goal_norm_peak) for s in stage0_results],
        [bool(p.survival_gate_passed) for p in p1_results],
        p2_z_goal,
        [float(m.contact_rate) for m in p2_metrics],
        z_goal_gate=z_goal_gate,
        contact_gate=contact_gate,
        min_fraction=min_fraction,
    )
    gate["g3_source"] = g3_source
    return gate


def classify_interpretation_branch(
    gate: Dict[str, Any],
    *,
    diversity_resolved: Optional[bool] = None,
    behaviour_harmful: Optional[bool] = None,
) -> str:
    """
    Pre-registered five-way interpretation grid for V3-EXQ-603f. `gate` is the
    dict from evaluate_substrate_gate; diversity_resolved / behaviour_harmful
    are supplied by the experiment after it computes the Q-045 arm deltas and
    harm/churn readouts (None until known).

    Branches:
      1 substrate_not_engaged            -- feeding/contact/survival gates fail
                                            -> non_contributory, return to substrate work.
      2 fed_but_no_goal                  -- contact occurs but z_goal does not form
                                            -> goal-formation issue.
      3 goal_formed_diversity_inert      -- goal forms, diversity/action mechanisms inert
                                            -> selection-authority / Ethics-Engine-3
                                               arbitration issue (modulatory-bias-selection-authority).
      4 goal_formed_mechanisms_load_bearing -- goal forms, MECH-313/MECH-260 resolve
                                            -> supports Q-045 / MECH-313 / MECH-260.
      5 goal_formed_behaviour_random_harmful -- goal forms but behaviour random/harmful
                                            -> arbitration failure, NOT a simple success.
    """
    if not (gate.get("stage0_positive_control") and gate.get("g1_survival") and gate.get("g2_contact")):
        return "substrate_not_engaged"
    if gate.get("g2_contact") and not gate.get("g3_zgoal"):
        return "fed_but_no_goal"
    # gate fully passed -> goal formed ecologically
    if behaviour_harmful:
        return "goal_formed_behaviour_random_harmful"
    if diversity_resolved is True:
        return "goal_formed_mechanisms_load_bearing"
    if diversity_resolved is False:
        return "goal_formed_diversity_inert"
    return "goal_formed_diversity_undetermined"


__all__ = [
    "ScaffoldedSD054OnboardingConfig",
    "ScaffoldedSD054OnboardingScheduler",
    "Stage0NurseryResult",
    "Stage0bConsolidationResult",
    "P0OnboardingResult",
    "HazardAvoidanceResult",
    "P1OnboardingResult",
    "P2OnboardingMetrics",
    "clone_trained_agent",
    "stage_plan",
    "STAGE_PLAN",
    "evaluate_substrate_gate",
    "substrate_readiness_from_results",
    "classify_interpretation_branch",
    "GOAL_WRITE_FORCED_FEED_OPEN",
    "GOAL_WRITE_CONSOLIDATE_PROTECTED",
    "GOAL_WRITE_ECOLOGICAL_CONTACT_OPEN",
    "GOAL_WRITE_DECAY_ONLY_ALLOWED",
    "GOAL_WRITE_MEASUREMENT_READONLY",
    "GOAL_WRITE_MODES",
]
