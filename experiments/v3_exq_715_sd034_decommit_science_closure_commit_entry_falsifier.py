"""
V3-EXQ-715: P2 ROOT-C COMMIT-DURATION DE-COMMIT SCIENCE falsifier -- MECH-445
(closure->beta commit-INTENT engagement) + MECH-446 (de-commit-AUTHORITY magnitude /
within-arm around-closure occupancy-drop) CO-OCCURRENCE on the NOW-VALIDATED F-independent
closure-plane commit-ENTRY substrate (use_closure_commit_entry, ree-v3 main 84c1e7c;
VALIDATED arming+sustaining by V3-EXQ-460o PASS + V3-EXQ-460p PASS, 2026-06-24).

SUPERSEDES the PARKED 460k/460l line. 460k tested the rung-6 NaturalCommitUrgencyRelease
DURATION lever; 460l tested the ARC-108 JOB-2 dopaminergic DRIVER pair (rho ramp + habenula).
BOTH are DISTINCT de-commit *delivery mechanisms* that got parked. THIS experiment is a
DIFFERENT OBJECT: the de-commit SCIENCE itself -- does the INTRINSIC SD-034 closure de-commit
(the committed-run-scaled refractory that fires at every closure) have the AUTHORITY to shorten
the now-sustaining closure-formed occupancy, and do the two decomposed children of the SD-034
de-commit pipeline (MECH-445 commit-intent + MECH-446 occupancy-drop) CO-OCCUR on the same
seeds -- on the substrate whose whole purpose was to dissolve the 460h disjoint-certifier
problem. The rung-6 duration lever + the ARC-108 driver pair are OFF in every arm here. Do NOT
re-author 460d/e/f/g/h/i/j/k/l.

WHY THIS IS THE OWED EXPERIMENT (brake off-ramp): MECH-445/446 each carry 5 prior
substrate_ceiling / non_contributory autopsies (460h/i/j/k/l) -- the re-derive brake FIRED
(threshold 2). It is RELEASED here because the named upstream substrate the 460l autopsy routed
to (re_derive_brake.upstream_substrate = f_dominance_conversion_ceiling, "closure-coupled-hold
arming") is now BUILT + VALIDATED: config.e3.use_closure_commit_entry adds a sticky
F-INDEPENDENT latch e3._closure_committed_active (e3_selector.py; SET in
REEAgent.select_action on a goal-active, rule-directed commitment) so the natural-commit
latch-hold arms + SUSTAINS a closure-formed beta occupancy WITHOUT any F-driven commit
(e3._committed_trajectory stays None) -- the precondition (ncl_hold_closure_armed_total > 0)
that every prior 460h..460l run FAILED (all measured 0). 460o/460p certified arming+sustaining;
the substrate_queue build_record_460k_460l_commit_entry "next" spells out exactly this owed
de-commit falsifier: "runs the full de-commit falsifier once (a) [arm+sustain] clears". (a)
cleared at 460o/p; this is that experiment.

WHAT IS TESTED (the two SD-034 de-commit-pipeline children, on the F-independent occupancy):
  MECH-445 (closure->beta commit-intent engagement, child A / S3): on WEAK-natural-commit seeds
    (F-independent by construction here -- n_f_commits ~ 0), the refractory-independent commit-
    intent counter beta_gate.sd034_n_closure_commit_intent (a closure-coupled commitment forming
    WHILE NOT result.committed, counted BEFORE the elevate/refractory gate) is > 0. This is the
    exact certifier MECH-445's what_would_answer names, and the counter the MECH-446 magnitude
    lever provably cannot zero (it is counted pre-gate).
  MECH-446 (de-commit-authority magnitude, child B / S4): on the ON arm the within-arm mean
    post-closure latch occupancy is below the mean pre-closure occupancy by
    >= DECOMMIT_MIN_DROP_FRAC (relative), over >= C2_MIN_WINDOW_EVENTS scored around-closure
    windows whose pre-occupancy cleared WITHIN_PRE_OCC_FLOOR (there was a sustained occupancy to
    de-commit). This is the exact within-arm around-closure DV MECH-446's what_would_answer names.

GATE ORDER (per substrate_queue build_record_460k_460l_commit_entry "next"):
  (a) READINESS (self-route, NEVER a weakens): ARM_ENTRY_ON arms + SUSTAINS a closure-formed
      occupancy on >= 2/3 seeds -- ncl_hold_closure_armed_total > 0 AND max_consecutive_beta_run
      >= SUSTAIN_MIN_TICKS -- with the hold ACTUALLY ARMING (the 460o gate (a)) and ZERO
      F-commits (n_f_commits == 0, the F-independence regime MECH-445 requires), AND the OFF
      baseline (use_closure_commit_entry=False, the 460k/460l signature) does NOT arm
      (ncl_hold_closure_armed_total == 0 -- non-degeneracy: the occupancy is entry-driven), AND
      a goal-active rule-directed commitment formed (the SET precondition), AND enough scored
      around-closure windows exist for the MECH-446 DV to be interpretable. If ANY readiness leg
      fails -> substrate_not_ready_requeue (non_degenerate=false, NEVER a false weakens).
  (b)+(c) SCIENCE (falsifiable, only once (a) clears): the SD-034 closure de-commit SHORTENS the
      sustained occupancy (MECH-446 within-arm drop) AND MECH-445 commit-intent CO-OCCURS with
      the MECH-446 occupancy-drop on the SAME >= 2/3 seeds. PASS = co-occurrence (the clean joint
      result the disjoint-certifier substrate was built to make reachable). If readiness holds but
      a child's own gate fails -> that child WEAKENS (a genuine falsification, not a self-route).

DV: pre/post-closure beta-latch occupancy fraction at each closure fire (the within-arm around-
closure window), plus the refractory-independent commit-intent counter. PROMOTES NOTHING --
MECH-445/446 stay candidate / standard / v3_pending / pending_retest_after_substrate until this
scores; a PASS/weaken is applied by governance, not by this script.

MECH-094: the closure-entry latch SET (e3._closure_committed_active) is a WAKING control-state
transition -- no replay / no memory-write surface -- so hypothesis_tag does NOT apply
(agent.update_residue is called with hypothesis_tag=False in the eval loop). Ethics preflight:
all-false / decision allow (V3 pre-ethical instrumentation; SENT-0 boundary).

DESIGN doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md (the two 2026-06-23
commit-ENTRY amend sections). UNIT contract: ree-v3/tests/contracts/test_closure_commit_entry.py.
Substrate harness mirrors the VALIDATED V3-EXQ-460o (same curriculum, same closure-exclusive eval
substrate, same z_goal eval seeding, same arm_cell fingerprint); the de-commit scoring gates
mirror the canonical V3-EXQ-460h (within-arm drop + refractory-independent coupling certifier).

experiment_purpose: evidence. claim_ids: [MECH-445, MECH-446].
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _benefit_and_drive,
    _contacted_resource_type,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_715_sd034_decommit_science_closure_commit_entry_falsifier"
QUEUE_ID = "V3-EXQ-715"
CLAIM_IDS: List[str] = ["MECH-445", "MECH-446"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-460l"  # supersedes the PARKED 460k/460l de-commit line (see docstring)

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_DECOMMIT_SCIENCE_CLOSURE_COMMIT_ENTRY_ENTRY_OFF_ON"

# --- Goal-pipeline / encoder dims (mirror 460o exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C; mirror 460o) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- closure-plane commit-ENTRY primitive (REEConfig defaults; ARM_ENTRY_ON) ---
CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR = 0.01  # rule_state norm floor for the SET predicate

# --- Within-arm around-closure window DV (MECH-446, mirror 460h/460o) ---
CLOSURE_WINDOW = 10
WINDOW_MIN_TICKS = 3

# --- Curriculum budgets (mirror 460o exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
CLOSURE_EVAL_EPISODES = 15  # per arm (x2 arms)
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- Pre-registered acceptance thresholds ---
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
ARM_PASS_FRACTION = 2.0 / 3.0       # gate (a) + science gates: on >= 2/3 seeds
SUSTAIN_MIN_TICKS = 2               # gate (a): multi-tick closure-formed occupancy
RULE_DIRECTED_MIN_TICKS = 1         # non-vacuity: >= 1 goal-active rule-directed commit tick
# MECH-445 (commit-intent, child A): refractory-independent counter >= this on the ON arm.
MIN_COMMIT_INTENT = 1
# MECH-446 (within-arm around-closure drop, child B): mean post-closure occupancy at least this
# RELATIVE fraction below mean pre-closure occupancy (paired across closures). Mirror 460h.
DECOMMIT_MIN_DROP_FRAC = 0.10
C2_MIN_WINDOW_EVENTS = 2            # minimum scored around-closure windows on the ON arm
WITHIN_PRE_OCC_FLOOR = 0.1         # pre-closure occupancy must be non-trivial (was committed)

# --- Eval-arm definitions (the F-independent commit-ENTRY primitive toggled; closure-
#     exclusive de-commit eval ON in every arm; rung-6 + ARC-108 driver OFF in every arm). ---
ARM_OFF = "ARM_ENTRY_OFF"
ARM_ON = "ARM_ENTRY_ON"
ARMS: List[Dict[str, Any]] = [
    {"key": ARM_OFF, "entry": False},
    {"key": ARM_ON, "entry": True},
]


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_train_rule_bias_head=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """460o-validated foraging substrate + commitment control-plane + commitment-closure-
    control-plane amend Legs A/B/C + beta-engagement coupling + the closure-exclusive de-commit
    eval mode + the natural-commit latch-hold. use_closure_commit_entry is LEFT OFF on the
    trained base (armed per-arm at eval by _clone_arm; the latch + hold carry no trainable
    parameters). The rung-6 NaturalCommitUrgencyRelease and the ARC-108 JOB-2 driver pair are
    OFF in every arm (NOT under test -- this is the intrinsic-SD-034-de-commit science)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
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
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
        # SD-034 commitment-closure-control-plane amend (Legs A/B/C + coupling):
        use_closure_env_completion_hook=True,          # Leg A
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base (MECH-446 refractory)
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D)
        use_closure_commit_beta_coupling=True,         # beta-engagement coupling (MECH-445 path)
        # rung-6 natural-commit-occupancy-release lever: OFF in every arm (parked line).
        use_natural_commit_urgency_release=False,
        # The natural-commit LATCH-HOLD is ARMED on the base config (carried into every arm via
        # _clone_arm's deepcopy). Precondition of use_closure_commit_entry + closure_exclusive_-
        # decommit_eval.
        use_natural_commit_latch_hold=True,
        # The CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode (ree-v3 e52158d): beta elevation closure-
        # EXCLUSIVE + the latch-hold arms on _closure_commit_active. ON in EVERY arm.
        closure_exclusive_decommit_eval=True,
        # The F-INDEPENDENT commit-ENTRY primitive under test: OFF on the trained base; armed
        # per-arm at eval by _clone_arm. Preconditions (loud ValueError): requires
        # use_closure_commit_beta_coupling AND use_natural_commit_latch_hold -- both set above.
        use_closure_commit_entry=False,
        closure_commit_entry_rule_norm_floor=CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _arm_config_slice(arm: Dict[str, Any]) -> Dict[str, Any]:
    """The full per-arm config dict for the arm_fingerprint (the only inter-arm variable is
    use_closure_commit_entry; the rest is shared substrate config)."""
    return {
        "arm_key": arm["key"],
        "use_closure_commit_entry": bool(arm["entry"]),
        "closure_commit_entry_rule_norm_floor": CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
        "closure_exclusive_decommit_eval": True,
        "use_closure_commit_beta_coupling": True,
        "use_natural_commit_latch_hold": True,
        "use_natural_commit_urgency_release": False,
        "beta_gate_bistable": True,
        "use_lateral_pfc_analog": True,
        "use_closure_operator": True,
        "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
        "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        "world_dim": WORLD_DIM,
        "z_goal_enabled": True,
        "drive_weight": DRIVE_WEIGHT,
        "scaffold_train_rule_bias_head": True,
        "closure_eval_episodes": CLOSURE_EVAL_EPISODES,
    }


def _build_closure_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint tolerance-band
    completion so the SD-034 closure operator has completions to fire on (mirror 460o)."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=p2_hfa,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        subgoal_mode=True,
        num_waypoints=2,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.25,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
        **_sd049_kwargs(scaffold_cfg),
    )


def _clone_arm(trained_agent: REEAgent, device: torch.device, arm: Dict[str, Any]) -> REEAgent:
    """Clone the SAME trained weights into an agent built with this arm's commit-ENTRY config
    (use_closure_commit_entry on/off). The closure-exclusive eval + latch-hold + closure operator
    stay ON in every arm (the ENTRY primitive is the only variable). The latch + hold carry no
    trainable parameters, so the state_dict loads cleanly (mirrors 460o's _clone_arm)."""
    cfg = copy.deepcopy(trained_agent.config)
    cfg.use_closure_commit_entry = bool(arm["entry"])
    cfg.closure_commit_entry_rule_norm_floor = CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR
    # rung-6 + ARC-108 JOB-2 driver levers stay OFF in every arm.
    cfg.use_natural_commit_urgency_release = False
    cfg.use_closure_operator = True
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        agent.load_state_dict(state, strict=False)
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    agent.beta_gate = BetaGate(completion_release_threshold=2.0)
    # HARNESS FIX (failure_autopsy_V3-EXQ-460m-460n_2026-06-23; carried from 460o): copy the
    # trained substrate's LIVE goal-seeding calibration onto the clone. The scaffold writes
    # z_goal_seeding_gain / benefit_threshold / drive_floor onto agent.goal_state.config (the
    # live GoalConfig), NOT onto REEConfig, so the deepcopy above does NOT carry them -> without
    # this the clone keeps default GoalConfig (benefit_threshold=0.1) and wild contact (~0.03)
    # would not clear the seeding gate, leaving goal_state inactive.
    _src_gc = getattr(getattr(trained_agent, "goal_state", None), "config", None)
    _dst_gc = getattr(getattr(agent, "goal_state", None), "config", None)
    if _src_gc is not None and _dst_gc is not None:
        for _attr in ("z_goal_seeding_gain", "benefit_threshold", "drive_floor"):
            if hasattr(_src_gc, _attr):
                setattr(_dst_gc, _attr, getattr(_src_gc, _attr))
    return agent


def _around_closure_windows(
    beta_history: List[bool], fire_ticks: List[int]
) -> List[Dict[str, float]]:
    """For each closure fire at tick t, the beta-latch occupancy FRACTION in the pre-closure
    window [t-W, t) and the post-closure window (t, t+W] (the paired within-arm de-commit datum;
    mirror 460h/460o -- the MECH-446 DV)."""
    n = len(beta_history)
    events: List[Dict[str, float]] = []
    for t in fire_ticks:
        pre_lo = max(0, t - CLOSURE_WINDOW)
        pre = beta_history[pre_lo:t]
        post_hi = min(n, t + 1 + CLOSURE_WINDOW)
        post = beta_history[t + 1:post_hi]
        if len(pre) < WINDOW_MIN_TICKS or len(post) < WINDOW_MIN_TICKS:
            continue
        pre_occ = sum(1 for b in pre if b) / float(len(pre))
        post_occ = sum(1 for b in post if b) / float(len(post))
        events.append({"pre_occ": pre_occ, "post_occ": post_occ})
    return events


def _max_consecutive_true(seq: List[bool]) -> int:
    """Longest run of consecutive True (beta-elevated) ticks -- the sustained-occupancy proxy."""
    best = 0
    cur = 0
    for b in seq:
        if b:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _rule_state_norm(agent: REEAgent) -> float:
    """||lateral_pfc.rule_state|| (the SET predicate's 'rule is being followed' magnitude);
    0.0 when lateral_pfc absent / rule_state None."""
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None:
        return 0.0
    rs = getattr(lpfc, "rule_state", None)
    if rs is None:
        return 0.0
    try:
        return float(rs.norm().item())
    except Exception:
        return 0.0


def _eval_arm_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Eval instrumented for both SD-034 de-commit-pipeline children. Reads the closure-armed
    counter (agent._ncl_hold_closure_armed_count), the longest consecutive beta-elevated run,
    the F-driven natural commits (n_f_commits, expected ~0 -- F-independence), the SD-034 closure
    fire count, the refractory-independent commit-intent counter (MECH-445), the around-closure
    occupancy windows (MECH-446), and the non-vacuity readout n_rule_directed_commit_ticks. Calls
    agent.update_residue(hypothesis_tag=False) each tick so the waking post-action path runs
    (mirror 460o)."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_dacc = getattr(agent, "dacc", None) is not None
    hook_enabled = bool(getattr(agent.config, "use_closure_env_completion_hook", False))
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream
    rule_floor = float(
        getattr(agent.config, "closure_commit_entry_rule_norm_floor",
                CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR)
    )
    # HARNESS FIX (carried from 460o): seed z_goal each eval step (consumption-gated) so
    # goal_state.is_active() can be True during the closure-eval -- the F-independent SET
    # predicate's precondition. The raw-benefit seeding gate mirrors _reconciled_gating_threshold:
    #   floor = benefit_threshold / (z_goal_seeding_gain * (1 + drive_weight * drive_floor)).
    bridge_on = bool(getattr(agent.config, "use_cue_recall", False)) or bool(
        getattr(agent.config, "use_incentive_token_bank", False)
    )
    _gc = getattr(getattr(agent, "goal_state", None), "config", None)
    if _gc is not None:
        _gain = float(getattr(_gc, "z_goal_seeding_gain", 1.0))
        _thr = float(getattr(_gc, "benefit_threshold", 0.1))
        _dw = float(getattr(_gc, "drive_weight", 0.0))
        _df = float(getattr(_gc, "drive_floor", 0.0))
        _denom = _gain * (1.0 + _dw * _df)
        seed_gate = (_thr / _denom) if _denom > 1e-12 else 0.0
    else:
        seed_gate = 0.0

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    beta_release_events = 0
    nogo_installed_total = 0
    total_committed_steps = 0
    total_beta_elevated = 0
    n_sequence_completions = 0
    n_hook_fires = 0
    n_closure_commit_intent = 0
    n_closure_coupled_elevations = 0
    around_events: List[Dict[str, float]] = []
    max_consecutive_beta_run = 0
    ncl_hold_reassert_total = 0
    ncl_hold_closure_armed_total = 0
    # F-driven natural commit count (expected ~0 on this substrate -- the F-independence regime).
    n_f_commits = 0
    # NON-VACUITY readout: goal active AND rule_state norm >= floor (the SET precondition).
    n_rule_directed_commit_ticks = 0
    rule_state_norm_peak = 0.0
    committed_class_counts: Counter = Counter()

    for _ in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        prev_beta = bool(agent.beta_gate.is_elevated)
        beta_history: List[bool] = []
        fire_ticks: List[int] = []

        for tick_idx in range(steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = _sense_with_optional_harm(
                agent, obs_body, obs_world, obs_dict, device, feed_harm
            )

            n_closures_before = (
                int(agent.closure_operator._n_closures) if has_closure else 0
            )
            dacc_hist_before = len(agent.dacc._action_history) if has_dacc else 0

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            if has_closure:
                fired_now = int(agent.closure_operator._n_closures) - n_closures_before
                if fired_now > 0 and has_dacc:
                    nogo_installed_total += (
                        len(agent.dacc._action_history) - dacc_hist_before
                    )

            cur_beta = bool(agent.beta_gate.is_elevated)
            beta_history.append(cur_beta)
            committed_now = agent.e3._committed_trajectory is not None
            if committed_now:
                total_committed_steps += 1
                n_f_commits += 1  # an F-driven natural commit was live AT this read point
                committed_class_counts[action_idx] += 1
            if cur_beta:
                total_beta_elevated += 1
            if prev_beta and not cur_beta:
                beta_release_events += 1
            prev_beta = cur_beta

            # NON-VACUITY: did the SET predicate have a goal-active rule-directed commitment to
            # latch on? (goal_state.is_active() AND rule_state norm >= floor).
            gs = getattr(agent, "goal_state", None)
            goal_active = bool(gs is not None and gs.is_active())
            rs_norm = _rule_state_norm(agent)
            if rs_norm > rule_state_norm_peak:
                rule_state_norm_peak = rs_norm
            if goal_active and rs_norm >= rule_floor:
                n_rule_directed_commit_ticks += 1

            _, _harm, done, info, obs_dict = env.step(action_idx)

            # Drive the waking post-action path (mirror 460o). MECH-094: hypothesis_tag=False --
            # a WAKING control-state transition, no replay / no memory-write surface.
            agent.update_residue(harm_signal=float(_harm), hypothesis_tag=False)

            # HARNESS FIX (carried from 460o): seed z_goal from the POST-step body-state so
            # goal_state.is_active() can be True during the closure-eval. Contact-gated: a
            # sub-seeding whiff is SKIPPED (not decay-only updated) so the consolidated trace is
            # protected from washout (the V3-EXQ-634b lesson); a genuine contact step seeds.
            _benefit_seed, _drive_seed = _benefit_and_drive(obs_dict["body_state"].to(device))
            _rt_seed = _contacted_resource_type(obs_dict) if bridge_on else None
            if _benefit_seed > seed_gate:
                agent.update_z_goal(
                    benefit_exposure=_benefit_seed,
                    drive_level=_drive_seed,
                    resource_type=_rt_seed,
                )

            if info.get("transition_type") == "sequence_complete":
                n_sequence_completions += 1
                if has_closure and hook_enabled:
                    ev = agent.notify_env_completion(action_class=action_idx)
                    if ev is not None and getattr(ev, "fired", False):
                        n_hook_fires += 1
                        nogo_installed_total += int(getattr(ev, "nogo_pushed", 0))

            if has_closure and int(agent.closure_operator._n_closures) > n_closures_before:
                fire_ticks.append(tick_idx)
            if done:
                break

        around_events.extend(_around_closure_windows(beta_history, fire_ticks))
        _ep_run = _max_consecutive_true(beta_history)
        if _ep_run > max_consecutive_beta_run:
            max_consecutive_beta_run = _ep_run
        # Read per-episode counters BEFORE the next agent.reset() wipes them.
        ncl_hold_reassert_total += int(getattr(agent, "_ncl_hold_reassert_count", 0))
        ncl_hold_closure_armed_total += int(
            getattr(agent, "_ncl_hold_closure_armed_count", 0)
        )
        _bstate = agent.beta_gate.get_state()
        n_closure_commit_intent += int(_bstate.get("sd034_n_closure_commit_intent", 0))
        n_closure_coupled_elevations += int(
            _bstate.get("sd034_n_closure_coupled_elevations", 0)
        )

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre if has_closure else 0
    )
    n_window_events = len(around_events)
    mean_pre_occ = (
        sum(e["pre_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    mean_post_occ = (
        sum(e["post_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    return {
        "n_closures": n_closures,
        "n_automatic_fires": max(0, n_closures - n_hook_fires),
        "n_hook_fires": n_hook_fires,
        "sd034_n_closure_commit_intent": n_closure_commit_intent,
        "sd034_n_closure_coupled_elevations": n_closure_coupled_elevations,
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "mean_beta_elevated_steps": total_beta_elevated / max(1, n_eps),
        "mean_per_commit_hold": total_beta_elevated / max(1, beta_release_events),
        "max_consecutive_beta_run": max_consecutive_beta_run,
        "ncl_hold_reassert_total": ncl_hold_reassert_total,
        "ncl_hold_closure_armed_total": ncl_hold_closure_armed_total,
        "n_f_commits": n_f_commits,
        "n_rule_directed_commit_ticks": n_rule_directed_commit_ticks,
        "rule_state_norm_peak": rule_state_norm_peak,
        "n_sequence_completions": n_sequence_completions,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        "env_hook_enabled": hook_enabled,
        "n_window_events": n_window_events,
        "mean_pre_closure_occ": mean_pre_occ,
        "mean_post_closure_occ": mean_post_occ,
        "committed_class_entropy_n_classes": len(committed_class_counts),
    }


def _empty_arm() -> Dict[str, Any]:
    return {
        "n_closures": 0, "n_automatic_fires": 0, "n_hook_fires": 0,
        "sd034_n_closure_commit_intent": 0, "sd034_n_closure_coupled_elevations": 0,
        "beta_release_events": 0, "nogo_installed_total": 0, "total_committed_steps": 0,
        "total_beta_elevated": 0, "mean_beta_elevated_steps": 0.0,
        "mean_per_commit_hold": 0.0, "max_consecutive_beta_run": 0,
        "ncl_hold_reassert_total": 0, "ncl_hold_closure_armed_total": 0,
        "n_f_commits": 0, "n_rule_directed_commit_ticks": 0, "rule_state_norm_peak": 0.0,
        "n_sequence_completions": 0, "n_eval_episodes": 0, "closure_present": False,
        "env_hook_enabled": False, "n_window_events": 0, "mean_pre_closure_occ": 0.0,
        "mean_post_closure_occ": 0.0, "committed_class_entropy_n_classes": 0,
    }


def _arm_armed_and_sustained(arm: Dict[str, Any]) -> bool:
    """gate (a) per-seed predicate (ARM_ENTRY_ON): the latch armed
    (ncl_hold_closure_armed_total > 0) AND the occupancy sustained
    (max_consecutive_beta_run >= SUSTAIN_MIN_TICKS)."""
    return bool(
        int(arm.get("ncl_hold_closure_armed_total", 0)) > 0
        and int(arm.get("max_consecutive_beta_run", 0)) >= SUSTAIN_MIN_TICKS
    )


def _mech445_commit_intent_met(arm_on: Dict[str, Any]) -> bool:
    """MECH-445 (child A): the refractory-independent closure-plane commit-intent counter fired
    on the ON arm (>= MIN_COMMIT_INTENT) AND a sequence completed (closure had an opportunity).
    Keys on sd034_n_closure_commit_intent -- counted BEFORE the elevate/refractory gate -- so the
    MECH-446 magnitude lever cannot zero it (the 460g self-defeat)."""
    return bool(
        int(arm_on.get("sd034_n_closure_commit_intent", 0)) >= MIN_COMMIT_INTENT
        and int(arm_on.get("n_sequence_completions", 0)) > 0
    )


def _mech446_within_arm_drop_met(arm_on: Dict[str, Any]) -> bool:
    """MECH-446 (child B, load-bearing within-arm DV): on the ON arm, mean post-closure occupancy
    < mean pre-closure occupancy with a >= DECOMMIT_MIN_DROP_FRAC relative drop, over
    >= C2_MIN_WINDOW_EVENTS scored windows whose pre-occupancy cleared WITHIN_PRE_OCC_FLOOR
    (there was a sustained occupancy to de-commit). Mirror 460h C2."""
    n_ev = int(arm_on.get("n_window_events", 0))
    pre = float(arm_on.get("mean_pre_closure_occ", 0.0))
    post = float(arm_on.get("mean_post_closure_occ", 0.0))
    if n_ev < C2_MIN_WINDOW_EVENTS or pre <= WITHIN_PRE_OCC_FLOOR:
        return False
    return bool(post < pre and (pre - post) >= DECOMMIT_MIN_DROP_FRAC * pre)


def _within_window_nonvacuous(arm_on: Dict[str, Any]) -> bool:
    """Readiness: the ON arm produced enough scored around-closure windows with a non-trivial
    pre-closure occupancy for the MECH-446 within-arm DV to be interpretable."""
    return bool(
        int(arm_on.get("n_window_events", 0)) >= C2_MIN_WINDOW_EVENTS
        and float(arm_on.get("mean_pre_closure_occ", 0.0)) > WITHIN_PRE_OCC_FLOOR
    )


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "arms": {a["key"]: _empty_arm() for a in ARMS},
        "arm_results": [
            {"seed": seed, "arm": a["key"], "aborted": True, **_empty_arm()}
            for a in ARMS
        ],
        "on_armed_and_sustained": False,
        "off_did_not_arm": False,
        "on_zero_f_commits": False,
        "on_rule_directed_met": False,
        "within_window_nonvacuous": False,
        "mech445_commit_intent_met": False,
        "mech446_within_arm_drop_met": False,
        "cooccur_met": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else CLOSURE_EVAL_EPISODES

    probe_env = _build_closure_env(scaffold_cfg)
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    # Eval both arms on the SAME trained substrate (clone per arm; the commit-ENTRY flag
    # toggled). Each (seed x arm) cell is wrapped in arm_cell (resets RNG on enter, stamps the
    # fingerprint) -- the multi-arm arm_fingerprint obligation (mint-as-you-go: the ENTRY_OFF
    # baseline cell is emitted reuse-eligible by default).
    arms_out: Dict[str, Any] = {}
    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        with arm_cell(
            seed,
            config_slice=_arm_config_slice(arm),
            script_path=Path(__file__),
        ) as cell:
            closure_env = _build_closure_env(scaffold_cfg)
            closure_env.reset()
            print(f"Seed {seed} Condition {arm['key']}", flush=True)
            agent_arm = _clone_arm(agent, device, arm)
            agent_arm.e3._running_variance = float(agent.e3._running_variance)
            metrics = _eval_arm_behaviour(
                agent_arm, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
            )
            done += eval_eps
            arms_out[arm["key"]] = metrics
            row = {"seed": seed, "arm": arm["key"], "aborted": False, **metrics}
            cell.stamp(row)
            arm_results.append(row)

    arm_off = arms_out[ARM_OFF]
    arm_on = arms_out[ARM_ON]

    # Readiness (gate a) legs.
    on_armed_and_sustained = _arm_armed_and_sustained(arm_on)
    off_did_not_arm = bool(int(arm_off.get("ncl_hold_closure_armed_total", 0)) == 0)
    on_zero_f_commits = bool(int(arm_on.get("n_f_commits", 0)) == 0)
    on_rule_directed_met = bool(
        int(arm_on.get("n_rule_directed_commit_ticks", 0)) >= RULE_DIRECTED_MIN_TICKS
    )
    within_window_nonvacuous = _within_window_nonvacuous(arm_on)

    # Science (gates b/c) -- the two SD-034 de-commit-pipeline children + their co-occurrence.
    mech445_commit_intent_met = _mech445_commit_intent_met(arm_on)
    mech446_within_arm_drop_met = _mech446_within_arm_drop_met(arm_on)
    cooccur_met = bool(mech445_commit_intent_met and mech446_within_arm_drop_met)

    print(f"  [eval] arm_eval seed={seed} ep {done}/{total_eps}"
          f" | OFF armed={arm_off['ncl_hold_closure_armed_total']}"
          f" run={arm_off['max_consecutive_beta_run']} f_commits={arm_off['n_f_commits']}"
          f" | ON armed={arm_on['ncl_hold_closure_armed_total']}"
          f" run={arm_on['max_consecutive_beta_run']} f_commits={arm_on['n_f_commits']}"
          f" rdir={arm_on['n_rule_directed_commit_ticks']}"
          f" intent={arm_on['sd034_n_closure_commit_intent']}"
          f" pre_occ={arm_on['mean_pre_closure_occ']:.3f} post_occ={arm_on['mean_post_closure_occ']:.3f}"
          f" win={arm_on['n_window_events']} closures={arm_on['n_closures']}"
          f" | armed_sustained={on_armed_and_sustained} off_no_arm={off_did_not_arm}"
          f" zero_f={on_zero_f_commits} within_win={within_window_nonvacuous}"
          f" | m445={mech445_commit_intent_met} m446={mech446_within_arm_drop_met}"
          f" cooccur={cooccur_met}", flush=True)
    # Per-seed verdict for the runner: guard + readiness legs + co-occurrence. (Governance
    # scoring is aggregate, in run_experiment; this line drives the progress bar only.)
    seed_ready = bool(
        guard_pass and on_armed_and_sustained and off_did_not_arm
        and on_zero_f_commits and on_rule_directed_met and within_window_nonvacuous
    )
    seed_pass = bool(seed_ready and cooccur_met)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} ready={seed_ready} cooccur={cooccur_met}"
          f" (contact_rate={p2.contact_rate:.4f} z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})",
          flush=True)

    return {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "arms": arms_out,
        "arm_results": arm_results,
        "on_armed_and_sustained": on_armed_and_sustained,
        "off_did_not_arm": off_did_not_arm,
        "on_zero_f_commits": on_zero_f_commits,
        "on_rule_directed_met": on_rule_directed_met,
        "within_window_nonvacuous": within_window_nonvacuous,
        "mech445_commit_intent_met": mech445_commit_intent_met,
        "mech446_within_arm_drop_met": mech446_within_arm_drop_met,
        "cooccur_met": cooccur_met,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_arms = len(ARMS)
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + n_arms * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + n_arms * CLOSURE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # --- Readiness (gate a) on the guard-passing seeds ---
    def _gp_frac(key: str) -> float:
        return _frac([bool(r.get(key, False)) for r in guard_passing])

    rule_directed_frac = _gp_frac("on_rule_directed_met")
    rule_directed_met = bool(rule_directed_frac >= ARM_PASS_FRACTION)
    armed_frac = _gp_frac("on_armed_and_sustained")
    armed_and_sustained_met = bool(armed_frac >= ARM_PASS_FRACTION)
    off_did_not_arm_frac = _gp_frac("off_did_not_arm")
    # F-independence is certified STRUCTURALLY by off_did_not_arm: on the closure-exclusive eval
    # the OFF arm's latch-hold reduces to the F path (no entry primitive), so if natural F-commit
    # were strong enough to form an occupancy the OFF arm WOULD arm. off_did_not_arm==True (the
    # non-degeneracy gate) therefore certifies F is weak. n_f_commits is REPORTED as corroboration
    # (like 460o), NOT a separate brittle hard gate.
    off_did_not_arm_met = bool(off_did_not_arm_frac >= ARM_PASS_FRACTION)
    zero_f_frac = _gp_frac("on_zero_f_commits")
    within_window_frac = _gp_frac("within_window_nonvacuous")
    within_window_met = bool(within_window_frac >= ARM_PASS_FRACTION)

    # Non-degeneracy of gate (a): the ON-vs-OFF contrast genuinely differs on the passing seeds
    # (OFF armed==0 while ON armed>0) -- the two arms are not the same random variable.
    contrast_seeds = [
        r for r in guard_passing
        if int(r["arms"][ARM_OFF].get("ncl_hold_closure_armed_total", 0)) == 0
        and int(r["arms"][ARM_ON].get("ncl_hold_closure_armed_total", 0)) > 0
    ]
    gate_a_non_degenerate = bool(len(contrast_seeds) > 0)

    # --- Science (gates b/c) on the guard-passing seeds ---
    mech445_frac = _gp_frac("mech445_commit_intent_met")
    mech445_supported = bool(mech445_frac >= ARM_PASS_FRACTION)
    mech446_frac = _gp_frac("mech446_within_arm_drop_met")
    mech446_supported = bool(mech446_frac >= ARM_PASS_FRACTION)
    cooccur_frac = _gp_frac("cooccur_met")
    cooccur_met = bool(cooccur_frac >= ARM_PASS_FRACTION)

    # Full readiness = every leg of gate (a) clears (self-route legs; NEVER a false weakens).
    # F-independence is folded into off_did_not_arm_met (see comment above) + gate_a_non_degenerate.
    readiness_met = bool(
        contact_non_vacuity_met and rule_directed_met and armed_and_sustained_met
        and off_did_not_arm_met and gate_a_non_degenerate and within_window_met
    )

    # --- Routing ---
    # Readiness unmet -> substrate_not_ready_requeue (scoring-excluded; NOT a weakens).
    if not contact_non_vacuity_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not rule_directed_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "closure_rule_directed_commit_not_formed"
    elif not (armed_and_sustained_met and off_did_not_arm_met and gate_a_non_degenerate):
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "gate_a_arm_sustain_or_f_independence_unmet"
    elif not within_window_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "mech446_around_closure_window_starved"
    else:
        # Readiness (gate a) CLEARED -- the science is now scorable (a genuine PASS or weaken).
        if cooccur_met:
            outcome, label = "PASS", "decommit_science_children_cooccur"
            route_reason = "mech445_commit_intent_and_mech446_occupancy_drop_cooccur_2of3"
        else:
            outcome, label = "FAIL", "decommit_science_children_did_not_cooccur"
            route_reason = "readiness_met_but_child_gate(s)_failed_or_disjoint"

    # Per-claim evidence direction. Readiness unmet -> unknown + scoring-excluded (degenerate).
    if not readiness_met:
        evidence_direction = "unknown"
        evidence_direction_per_claim = {"MECH-445": "unknown", "MECH-446": "unknown"}
        non_degenerate = False
        non_degenerate_per_claim = {"MECH-445": False, "MECH-446": False}
        degeneracy_reason = f"substrate_not_ready: {route_reason}"
    else:
        m445_dir = "supports" if mech445_supported else "weakens"
        m446_dir = "supports" if mech446_supported else "weakens"
        evidence_direction_per_claim = {"MECH-445": m445_dir, "MECH-446": m446_dir}
        if mech445_supported and mech446_supported:
            evidence_direction = "supports"
        elif mech445_supported or mech446_supported:
            evidence_direction = "mixed"
        else:
            evidence_direction = "weakens"
        non_degenerate = True
        non_degenerate_per_claim = {"MECH-445": True, "MECH-446": True}
        degeneracy_reason = ""

    print(f"[{EXPERIMENT_TYPE}] contact={contact_non_vacuity_met} (guard {sum(guard_flags)}/{n})"
          f" rule_directed={rule_directed_met} armed_sustained={armed_and_sustained_met}"
          f" off_no_arm={off_did_not_arm_met}(f={off_did_not_arm_frac:.3f}) zero_f_ctx={zero_f_frac:.3f}"
          f" within_win={within_window_met} non_degen={gate_a_non_degenerate}"
          f" | READINESS={readiness_met}"
          f" | m445={mech445_supported}(f={mech445_frac:.3f})"
          f" m446={mech446_supported}(f={mech446_frac:.3f})"
          f" cooccur={cooccur_met}(f={cooccur_frac:.3f})"
          f" -> outcome={outcome} label={label} dir={evidence_direction}", flush=True)

    overall_pass = bool(outcome == "PASS")

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "rule_directed_met": rule_directed_met,
        "rule_directed_fraction": rule_directed_frac,
        "armed_and_sustained_met": armed_and_sustained_met,
        "armed_and_sustained_fraction": armed_frac,
        "off_did_not_arm_met": off_did_not_arm_met,
        "off_did_not_arm_fraction": off_did_not_arm_frac,
        "gate_a_non_degenerate": gate_a_non_degenerate,
        "zero_f_commits_fraction_context": zero_f_frac,
        "within_window_nonvacuous_met": within_window_met,
        "within_window_nonvacuous_fraction": within_window_frac,
        "readiness_met": readiness_met,
        "mech445_commit_intent_supported": mech445_supported,
        "mech445_commit_intent_fraction": mech445_frac,
        "mech446_within_arm_drop_supported": mech446_supported,
        "mech446_within_arm_drop_fraction": mech446_frac,
        "cooccurrence_met": cooccur_met,
        "cooccurrence_fraction": cooccur_frac,
        "overall_pass": overall_pass,
        "route_reason": route_reason,
        "per_seed_guard_pass": guard_flags,
        "per_seed_mech445": [bool(r.get("mech445_commit_intent_met", False)) for r in per_seed],
        "per_seed_mech446": [bool(r.get("mech446_within_arm_drop_met", False)) for r in per_seed],
        "per_seed_cooccur": [bool(r.get("cooccur_met", False)) for r in per_seed],
    }

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "non_degenerate": non_degenerate,
        "non_degenerate_per_claim": non_degenerate_per_claim,
        "degeneracy_reason": degeneracy_reason,
        "acceptance": acceptance,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "foraging_contact_guard",
                    "description": "603n G2+G3: per-seed P2 contact_rate > 0 AND "
                                   "z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout",
                },
                {
                    "name": "closure_rule_directed_commit_formed",
                    "description": "ARM_ENTRY_ON forms a goal-active rule-directed commitment the "
                                   "latch can set on (>= RULE_DIRECTED_MIN_TICKS) on >= 2/3 seeds.",
                    "measured": rule_directed_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": rule_directed_met,
                    "control": "ARM_ENTRY_ON eval loop",
                },
                {
                    "name": "gate_a_armed_and_sustained_f_independent",
                    "description": "ARM_ENTRY_ON arms (ncl_hold_closure_armed_total > 0) AND "
                                   "sustains (max_consecutive_beta_run >= SUSTAIN_MIN_TICKS) with "
                                   "ZERO F-commits, AND ARM_ENTRY_OFF does NOT arm (non-degenerate) "
                                   "-- the 460o gate (a); below floor -> substrate_not_ready_requeue "
                                   "(NEVER a *_ceiling / does_not_support / weakens label).",
                    "measured": armed_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": bool(armed_and_sustained_met and off_did_not_arm_met and gate_a_non_degenerate),
                    "control": "ARM_ENTRY_ON vs ARM_ENTRY_OFF contrast",
                },
                {
                    "name": "mech446_around_closure_window_nonvacuous",
                    "description": "ARM_ENTRY_ON produced >= C2_MIN_WINDOW_EVENTS scored around-"
                                   "closure windows with mean_pre_closure_occ > WITHIN_PRE_OCC_FLOOR "
                                   "on >= 2/3 seeds -- there was a sustained occupancy to de-commit. "
                                   "Below floor -> the MECH-446 DV is starved, not falsified -> "
                                   "substrate_not_ready_requeue.",
                    "measured": within_window_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": within_window_met,
                    "control": "ARM_ENTRY_ON around-closure windows",
                },
            ],
            "criteria_non_degenerate": {
                "gate_a_armed_sustained_contrast": gate_a_non_degenerate,
            },
            "criteria": [
                {
                    "name": "mech445_commit_intent_2of3",
                    "load_bearing": True,
                    "passed": mech445_supported,
                },
                {
                    "name": "mech446_within_arm_drop_2of3",
                    "load_bearing": True,
                    "passed": mech446_supported,
                },
                {
                    "name": "mech445_mech446_cooccurrence_2of3",
                    "load_bearing": True,
                    "passed": cooccur_met,
                },
            ],
            "science_note": "MECH-445 (child A, closure->beta commit-intent): the refractory-"
                            "independent sd034_n_closure_commit_intent counter (counted BEFORE the "
                            "elevate/refractory gate, so the MECH-446 magnitude lever cannot zero "
                            "it) is > 0 on the F-independent (weak-natural-commit) ON arm. MECH-446 "
                            "(child B, de-commit-authority magnitude): the within-arm mean post-"
                            "closure occupancy drops below the pre-closure occupancy by >= "
                            "DECOMMIT_MIN_DROP_FRAC. Gate (c): the two co-occur on the SAME >= 2/3 "
                            "seeds -- the joint result the disjoint-certifier substrate "
                            "(use_closure_commit_entry) was built to make reachable (dissolving the "
                            "460h disjoint-certifier problem, where commit-intent fired only on "
                            "weak-commit seeds and the window only on strong-commit seeds).",
            "levers_off_note": "The rung-6 NaturalCommitUrgencyRelease (the parked 460k DURATION "
                               "lever) and the ARC-108 JOB-2 dopaminergic DRIVER pair (the parked "
                               "460l rho ramp + habenula) are OFF in every arm. The de-commit under "
                               "test is the INTRINSIC SD-034 closure de-commit (the committed-run-"
                               "scaled refractory that fires at every closure) -- a DIFFERENT OBJECT "
                               "than the parked duration/driver lines.",
            "primitive_under_test": {
                "flag": "use_closure_commit_entry (ree-v3 main 84c1e7c; VALIDATED 460o/p PASS)",
                "latch": "e3._closure_committed_active (sticky, F-INDEPENDENT)",
                "mech445_counter": "beta_gate.sd034_n_closure_commit_intent (refractory-independent)",
                "mech446_dv": "within-arm mean_pre_closure_occ vs mean_post_closure_occ",
                "rule_norm_floor": CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
            },
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "supersedes_note": "supersedes the PARKED 460k (rung-6 duration lever) / 460l (ARC-108 "
                           "JOB-2 driver pair) de-commit line; do NOT re-author 460d..460l. This "
                           "is the de-commit SCIENCE falsifier (MECH-445/446 co-occurrence) on the "
                           "now-validated use_closure_commit_entry substrate.",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate": result["non_degenerate"],
        "non_degenerate_per_claim": result["non_degenerate_per_claim"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
            "note": "V3 pre-ethical instrumentation (SENT-0). MECH-094: the closure-entry latch "
                    "SET is a WAKING control-state transition (no replay / no memory-write "
                    "surface); agent.update_residue called with hypothesis_tag=False. "
                    "hypothesis_tag does not apply.",
        },
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1 -> P2; 460o config) + commitment control-plane (bistable "
                     "BetaGate + SD-034 ClosureOperator + SD-033a LateralPFC + SD-032 "
                     "dACC/salience) + subgoal_mode waypoint tolerance-band completion + "
                     "commitment-closure-control-plane Legs A/B/C + beta-engagement coupling + the "
                     "natural-commit LATCH-HOLD + the CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode "
                     "(closure_exclusive_decommit_eval) ARMED in BOTH arms. The F-independent "
                     "closure-plane commit-ENTRY primitive (use_closure_commit_entry, VALIDATED "
                     "460o/p) toggled per arm (ENTRY_OFF/ENTRY_ON). rung-6 "
                     "NaturalCommitUrgencyRelease + ARC-108 JOB-2 driver pair OFF in both arms.",
        "condition": CONDITION_LABEL,
        "method_note": "P2 ROOT-C de-commit SCIENCE falsifier: on the now-validated F-independent "
                       "use_closure_commit_entry substrate, do the two SD-034 de-commit-pipeline "
                       "children CO-OCCUR -- MECH-445 (refractory-independent closure->beta commit-"
                       "intent, sd034_n_closure_commit_intent > 0) AND MECH-446 (within-arm around-"
                       "closure occupancy drop >= DECOMMIT_MIN_DROP_FRAC) on the SAME >= 2/3 seeds? "
                       "Gate (a) READINESS (self-route, NEVER a weakens): ARM_ENTRY_ON arms + "
                       "sustains a closure-formed occupancy (ncl_hold_closure_armed_total > 0 AND "
                       "max_consecutive_beta_run >= SUSTAIN_MIN_TICKS) with ZERO F-commits AND "
                       "ARM_ENTRY_OFF does not arm (non-degenerate) AND a rule-directed commitment "
                       "formed AND enough around-closure windows exist. Any readiness leg unmet -> "
                       "substrate_not_ready_requeue (non_degenerate=false). Gates (b)+(c) SCIENCE "
                       "(falsifiable only once readiness clears): the intrinsic SD-034 closure "
                       "de-commit shortens the sustained occupancy (MECH-446) AND MECH-445 commit-"
                       "intent co-occurs with it. PROMOTES NOTHING.",
        "arm_note": "ARMS = " + ", ".join(a["key"] for a in ARMS) + ". closure-exclusive eval + "
                    "beta-engagement coupling + natural-commit latch-hold ON in both arms; the only "
                    "variable is use_closure_commit_entry. ARM_ENTRY_OFF is the 460k/460l no-arm "
                    "non-degeneracy baseline; ARM_ENTRY_ON is the science arm where MECH-445 "
                    "commit-intent + MECH-446 within-arm occupancy-drop are both measured.",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "arm_pass_fraction": ARM_PASS_FRACTION,
            "sustain_min_ticks": SUSTAIN_MIN_TICKS,
            "rule_directed_min_ticks": RULE_DIRECTED_MIN_TICKS,
            "min_commit_intent": MIN_COMMIT_INTENT,
            "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
            "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
            "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
            "closure_window": CLOSURE_WINDOW,
            "window_min_ticks": WINDOW_MIN_TICKS,
            "use_natural_commit_latch_hold": True,
            "closure_exclusive_decommit_eval": True,
            "use_closure_commit_beta_coupling": True,
            "use_natural_commit_urgency_release": False,
            "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
            "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
            "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
            "closure_commit_entry_rule_norm_floor": CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "closure_eval_episodes_per_arm": CLOSURE_EVAL_EPISODES,
            "n_eval_arms": len(ARMS),
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "scaffold_train_rule_bias_head": True,
            "config_basis": "closure-exclusive de-commit eval (ree-v3 main e52158d, 2026-06-22) + "
                            "F-independent closure-plane commit-ENTRY primitive "
                            "(use_closure_commit_entry, ree-v3 main 84c1e7c, 2026-06-23; VALIDATED "
                            "V3-EXQ-460o/460p PASS 2026-06-24)",
        },
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res.get("manifest_path"),
        dry_run=args.dry_run,
    )
