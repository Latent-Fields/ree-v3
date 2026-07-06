"""
V3-EXQ-717: MECH-445-ONLY commit-INTENT existence falsifier, REGIME-SCOPED (move M2 of
REE_assembly/evidence/planning/claim_synthesis_MECH-445-446_2026-07-06.md).

WHAT THIS IS: the "decouple + sprint MECH-445 alone" move. It DECOUPLES the closure->beta
commit-INTENT existence claim (MECH-445) from the de-commit-MAGNITUDE claim (MECH-446) and gives
MECH-445 a test with adequate POWER for a REGIME-SCOPED claim. It does NOT test MECH-446 at all --
the MECH-446 co-occurrence / within-arm around-closure occupancy-drop gate is DROPPED ENTIRELY.
claim_ids = [MECH-445].

WHY (the 715 diagnosis this fixes): V3-EXQ-715 (MECH-445 + MECH-446 co-occurrence) self-routed
substrate_not_ready on a 3-seed [42,43,44] / 2-of-3 gate. Its gate(a) armed on only 1/3 seeds
(seed 44: sd034_commit_intent = 398, n_f_commits = 7 -- a clean F-INDEPENDENT existence proof;
seeds 42/43 were strong-natural-commit, the F-driven commit monopolised selection so the closure
latch could not get purchase). MECH-445 is scoped to the WEAK-natural-commit regime, but only ~1
of 3 seeds falls in that regime, so a 3-seed/2-of-3 gate is STRUCTURALLY UNABLE to certify it --
this is a TEST-POWER deficit, not a substrate deficit. The fix (per MECH-445's what_would_answer):
run MANY seeds, classify each into weak/strong-natural-commit by its OFF-arm committed level, and
score the refractory-independent commit-intent counter ONLY on the weak-natural-commit
(in-regime) subset. Strong-natural-commit seeds are OUT OF SCOPE for MECH-445 by construction.

WHAT IS TESTED (the single SD-034 de-commit-pipeline child A, F-independent existence):
  MECH-445 (closure->beta commit-INTENT engagement, refractory-independent): on the WEAK-natural-
  commit (IN-REGIME) seeds, the refractory-independent commit-intent counter
  beta_gate.sd034_n_closure_commit_intent (a closure-coupled commitment forming WHILE NOT
  result.committed, counted BEFORE the elevate/refractory gate, so the de-commit magnitude lever
  provably cannot zero it -- the 460g self-defeat) is > 0 on >= 2/3 of THOSE in-regime seeds.
  This is exactly the certifier MECH-445's what_would_answer names.

REGIME CLASSIFIER (pre-registered, OFF-arm-anchored -- NOT circular with the scored quantity):
  Each seed's regime is read from the ARM_ENTRY_OFF (use_closure_commit_entry=False) baseline arm,
  which reduces to the pure F-driven natural-commit path (no entry latch). off_committed_frac =
  OFF.total_committed_steps / OFF.total_ticks is the FRACTION of eval ticks the OFF agent spent in
  an F-driven committed trajectory -- the intrinsic natural-commit strength. A seed is
  WEAK-natural-commit (IN-REGIME) iff off_committed_frac <= WEAK_COMMIT_FRAC_MAX. The classifier
  is anchored on a DIFFERENT arm + a DIFFERENT quantity than the scored ON-arm commit-intent
  counter, so it is not circular. A conservative (low) floor errs toward EXCLUDING borderline
  seeds -> underpopulated regime -> substrate_not_ready (a self-route), never toward contaminating
  the in-regime subset with strong-commit seeds (which would risk a false PASS).

GATE ORDER:
  (a) READINESS (self-route, NEVER a weakens): foraging contact guard passes on >= 2/3 seeds AND
      the WEAK-natural-commit subset is POPULATED (>= MIN_IN_REGIME_SEEDS guard-passing in-regime
      seeds -- else a regime-scoped claim cannot be certified: underpowered, not falsified) AND
      the commit-intent counter had an OPPORTUNITY on >= 2/3 of the in-regime seeds (a goal-active
      rule-directed commitment formed AND a sequence completed, so a closure could fire). Any
      readiness leg unmet -> substrate_not_ready_requeue (non_degenerate=false; NEVER a false
      weakens).
  (b) SCIENCE (falsifiable, only once (a) clears): the refractory-independent commit-intent
      counter is > 0 (>= MIN_COMMIT_INTENT) on >= 2/3 of the guard-passing IN-REGIME seeds. PASS =
      supported (MECH-445 F-independent commit-intent existence certified with adequate power).
      readiness-met-but-counter-0-on->1/3 -> genuine WEAKENS (a fair falsification, not a
      self-route).

DV: the refractory-independent commit-intent counter (beta_gate.sd034_n_closure_commit_intent) on
the ON arm, scored over the weak-natural-commit subset. PROMOTES NOTHING -- MECH-445 stays
candidate / v3_pending / pending_retest_after_substrate until this scores; governance applies a
PASS/weaken, not this script.

RELATION TO 715 (does NOT supersede it): 715 remains the MECH-445+MECH-446 co-occurrence record
(routed to /implement-substrate on f_dominance_conversion_ceiling for the MECH-446 face). 717 is a
NARROWER, HIGHER-POWER MECH-445-only test on the same validated substrate. The two answer
different questions.

RE-DERIVE BRAKE: MECH-445 carries 6 prior substrate_ceiling autopsies (460h/i/j/k/l, 715), so the
brake FIRED (threshold 2). It is RELEASED here on two grounds: (1) the named upstream substrate
the 460l/715 autopsies routed to (use_closure_commit_entry / F-independent closure-coupled-hold
arming) is BUILT + VALIDATED (V3-EXQ-460o/460p PASS 2026-06-24; present in ree_core/agent.py);
(2) this is a TEST-POWER redesign, NOT a same-selector de-commit-MAGNITUDE re-derive -- the
MECH-446 magnitude face the brake protects is DROPPED entirely. Certified brake-safe by
claim_synthesis_MECH-445-446_2026-07-06.md Sec.6.

MECH-094: the closure-entry latch SET is a WAKING control-state transition -- no replay / no
memory-write surface -- so hypothesis_tag does NOT apply (agent.update_residue called with
hypothesis_tag=False). Ethics preflight: all-false / decision allow (V3 pre-ethical
instrumentation; SENT-0 boundary).

Substrate harness mirrors the VALIDATED V3-EXQ-460o / V3-EXQ-715 (same curriculum, same closure-
exclusive eval substrate, same z_goal eval seeding, same arm_cell fingerprint). experiment_purpose:
evidence. claim_ids: [MECH-445].
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

EXPERIMENT_TYPE = "v3_exq_717_mech445_commit_intent_regime_scoped_falsifier"
QUEUE_ID = "V3-EXQ-717"
CLAIM_IDS: List[str] = ["MECH-445"]
EXPERIMENT_PURPOSE = "evidence"

# 12 seeds (M2: 9-12) -- the whole point is enough seeds to POPULATE the weak-natural-commit
# regime that ~1/3 of seeds fall in, so a regime-scoped claim can be certified at 2/3.
SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
CONDITION_LABEL = "REGIME_SCOPED_MECH445_COMMIT_INTENT_ENTRY_OFF_ON"

# --- Goal-pipeline / encoder dims (mirror 460o/715 exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C; mirror 460o/715) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- closure-plane commit-ENTRY primitive (REEConfig defaults; ARM_ENTRY_ON) ---
CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR = 0.01  # rule_state norm floor for the SET predicate

# --- Curriculum budgets (mirror 460o/715 exactly) ---
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
MIN_FRACTION = 2.0 / 3.0            # foraging contact guard on >= 2/3 seeds
ARM_PASS_FRACTION = 2.0 / 3.0       # science + opportunity: on >= 2/3 of the IN-REGIME seeds
RULE_DIRECTED_MIN_TICKS = 1         # non-vacuity: >= 1 goal-active rule-directed commit tick

# MECH-445 (commit-intent, child A): refractory-independent counter >= this on the ON arm.
MIN_COMMIT_INTENT = 1

# --- REGIME CLASSIFIER (pre-registered, OFF-arm-anchored) ---
# WEAK-natural-commit (IN-REGIME) iff the OFF baseline arm spent <= this FRACTION of eval ticks in
# an F-driven committed trajectory. Conservative (low) floor: lands in the bimodal valley between
# the F-collapsed (weak; 715 seed 44) and F-dominant (strong; 715 seeds 42/43) seeds, and errs
# toward excluding borderline seeds (-> substrate_not_ready) rather than contaminating the scored
# subset with strong-commit seeds (-> false PASS risk).
WEAK_COMMIT_FRAC_MAX = 0.15
# Minimum guard-passing IN-REGIME seeds required to certify the regime-scoped claim at 2/3. Below
# this the weak regime is underpopulated -> substrate_not_ready_requeue (the 715 test-power hole:
# 3 seeds / ~1 in-regime cannot certify a 2/3 gate; 12 seeds / ~4 in-regime can).
MIN_IN_REGIME_SEEDS = 3

# --- Eval-arm definitions (the F-independent commit-ENTRY primitive toggled; closure-exclusive
#     de-commit eval ON in every arm; rung-6 + ARC-108 driver OFF in every arm). ---
ARM_OFF = "ARM_ENTRY_OFF"   # regime CLASSIFIER arm (pure F path; use_closure_commit_entry=False)
ARM_ON = "ARM_ENTRY_ON"     # SCIENCE arm (use_closure_commit_entry=True; commit-intent measured)
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
    """460o/715-validated foraging substrate + commitment control-plane + commitment-closure-
    control-plane amend Legs A/B/C + beta-engagement coupling + the closure-exclusive de-commit
    eval mode + the natural-commit latch-hold. use_closure_commit_entry is LEFT OFF on the trained
    base (armed per-arm at eval by _clone_arm; the latch + hold carry no trainable parameters). The
    rung-6 NaturalCommitUrgencyRelease and the ARC-108 JOB-2 driver pair are OFF in every arm."""
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
        # The CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode: beta elevation closure-EXCLUSIVE + the latch-
        # hold arms on _closure_commit_active. ON in EVERY arm (matched substrate).
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
    completion so the SD-034 closure operator has completions to fire on (mirror 460o/715)."""
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
    trainable parameters, so the state_dict loads cleanly (mirrors 460o/715 _clone_arm)."""
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
    # HARNESS FIX (failure_autopsy_V3-EXQ-460m-460n_2026-06-23; carried from 460o/715): copy the
    # trained substrate's LIVE goal-seeding calibration onto the clone. The scaffold writes
    # z_goal_seeding_gain / benefit_threshold / drive_floor onto agent.goal_state.config (the live
    # GoalConfig), NOT onto REEConfig, so the deepcopy above does NOT carry them -> without this
    # the clone keeps default GoalConfig (benefit_threshold=0.1) and wild contact (~0.03) would
    # not clear the seeding gate, leaving goal_state inactive.
    _src_gc = getattr(getattr(trained_agent, "goal_state", None), "config", None)
    _dst_gc = getattr(getattr(agent, "goal_state", None), "config", None)
    if _src_gc is not None and _dst_gc is not None:
        for _attr in ("z_goal_seeding_gain", "benefit_threshold", "drive_floor"):
            if hasattr(_src_gc, _attr):
                setattr(_dst_gc, _attr, getattr(_src_gc, _attr))
    return agent


def _max_consecutive_true(seq: List[bool]) -> int:
    """Longest run of consecutive True (beta-elevated) ticks -- the sustained-occupancy proxy
    (reported as context only; NOT a MECH-445 gate)."""
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
    """Eval instrumented for MECH-445 (child A) + the OFF-arm regime classifier. Reads the
    refractory-independent commit-intent counter (beta_gate.sd034_n_closure_commit_intent), the
    F-driven natural commit count (total_committed_steps / n_f_commits -- the OFF-arm regime
    signal), the total eval ticks (for the committed FRACTION), the SD-034 closure fire count, the
    sequence completions (the counter's opportunity), and n_rule_directed_commit_ticks (the SET
    precondition). The MECH-446 within-arm around-closure occupancy-drop DV is DROPPED (this is a
    MECH-445-only test). Calls agent.update_residue(hypothesis_tag=False) each tick so the waking
    post-action path runs (mirror 460o/715)."""
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
    # HARNESS FIX (carried from 460o/715): seed z_goal each eval step (consumption-gated) so
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
    total_ticks = 0
    n_sequence_completions = 0
    n_hook_fires = 0
    n_closure_commit_intent = 0
    n_closure_coupled_elevations = 0
    max_consecutive_beta_run = 0
    ncl_hold_reassert_total = 0
    ncl_hold_closure_armed_total = 0
    # F-driven natural commit count (the OFF-arm regime classifier signal; on the ON arm expected
    # ~0 for in-regime seeds -- the F-independence regime).
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

            total_ticks += 1
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

            # Drive the waking post-action path (mirror 460o/715). MECH-094: hypothesis_tag=False
            # -- a WAKING control-state transition, no replay / no memory-write surface.
            agent.update_residue(harm_signal=float(_harm), hypothesis_tag=False)

            # HARNESS FIX (carried from 460o/715): seed z_goal from the POST-step body-state so
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

            if done:
                break

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
    off_committed_frac = total_committed_steps / float(max(1, total_ticks))
    return {
        "n_closures": n_closures,
        "n_automatic_fires": max(0, n_closures - n_hook_fires),
        "n_hook_fires": n_hook_fires,
        "sd034_n_closure_commit_intent": n_closure_commit_intent,
        "sd034_n_closure_coupled_elevations": n_closure_coupled_elevations,
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_ticks": total_ticks,
        "committed_frac": off_committed_frac,
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
        "committed_class_entropy_n_classes": len(committed_class_counts),
    }


def _empty_arm() -> Dict[str, Any]:
    return {
        "n_closures": 0, "n_automatic_fires": 0, "n_hook_fires": 0,
        "sd034_n_closure_commit_intent": 0, "sd034_n_closure_coupled_elevations": 0,
        "beta_release_events": 0, "nogo_installed_total": 0, "total_committed_steps": 0,
        "total_ticks": 0, "committed_frac": 0.0,
        "total_beta_elevated": 0, "mean_beta_elevated_steps": 0.0,
        "mean_per_commit_hold": 0.0, "max_consecutive_beta_run": 0,
        "ncl_hold_reassert_total": 0, "ncl_hold_closure_armed_total": 0,
        "n_f_commits": 0, "n_rule_directed_commit_ticks": 0, "rule_state_norm_peak": 0.0,
        "n_sequence_completions": 0, "n_eval_episodes": 0, "closure_present": False,
        "env_hook_enabled": False, "committed_class_entropy_n_classes": 0,
    }


def _is_weak_natural_commit(arm_off: Dict[str, Any]) -> bool:
    """REGIME classifier (pre-registered, OFF-arm-anchored): the seed is WEAK-natural-commit
    (IN-REGIME for MECH-445) iff the ARM_ENTRY_OFF baseline spent <= WEAK_COMMIT_FRAC_MAX of eval
    ticks in an F-driven committed trajectory. Requires a non-zero tick count (a run happened)."""
    return bool(
        int(arm_off.get("total_ticks", 0)) > 0
        and float(arm_off.get("committed_frac", 1.0)) <= WEAK_COMMIT_FRAC_MAX
    )


def _mech445_commit_intent_met(arm_on: Dict[str, Any]) -> bool:
    """MECH-445 (child A): the refractory-independent closure-plane commit-intent counter fired on
    the ON arm (>= MIN_COMMIT_INTENT) AND a sequence completed (closure had an opportunity). Keys
    on sd034_n_closure_commit_intent -- counted BEFORE the elevate/refractory gate -- so the (now
    dropped) MECH-446 magnitude lever cannot zero it (the 460g self-defeat)."""
    return bool(
        int(arm_on.get("sd034_n_closure_commit_intent", 0)) >= MIN_COMMIT_INTENT
        and int(arm_on.get("n_sequence_completions", 0)) > 0
    )


def _mech445_opportunity_met(arm_on: Dict[str, Any]) -> bool:
    """Readiness (per in-regime seed): the ON arm had an OPPORTUNITY for the commit-intent counter
    to fire -- a goal-active rule-directed commitment formed (the SET precondition) AND a sequence
    completed (a closure could fire). If broadly absent across the in-regime subset the counter is
    STARVED (substrate_not_ready), not falsified."""
    return bool(
        int(arm_on.get("n_rule_directed_commit_ticks", 0)) >= RULE_DIRECTED_MIN_TICKS
        and int(arm_on.get("n_sequence_completions", 0)) > 0
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
        "off_committed_frac": 0.0,
        "is_weak_natural_commit": False,
        "on_commit_intent": 0,
        "mech445_opportunity_met": False,
        "mech445_commit_intent_met": False,
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

    # Eval both arms on the SAME trained substrate (clone per arm; the commit-ENTRY flag toggled).
    # ARM_ENTRY_OFF = the regime CLASSIFIER (pure F path); ARM_ENTRY_ON = the SCIENCE arm (commit-
    # intent measured). Each (seed x arm) cell is wrapped in arm_cell (resets RNG on enter, stamps
    # the fingerprint) -- the multi-arm arm_fingerprint obligation (mint-as-you-go: the ENTRY_OFF
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

    # Regime classification (OFF-arm-anchored) + MECH-445 per-seed metrics (ON arm).
    off_committed_frac = float(arm_off.get("committed_frac", 0.0))
    is_weak_natural_commit = _is_weak_natural_commit(arm_off)
    on_commit_intent = int(arm_on.get("sd034_n_closure_commit_intent", 0))
    mech445_opportunity_met = _mech445_opportunity_met(arm_on)
    mech445_commit_intent_met = _mech445_commit_intent_met(arm_on)

    print(f"  [eval] arm_eval seed={seed} ep {done}/{total_eps}"
          f" | OFF committed_frac={off_committed_frac:.3f} committed={arm_off['total_committed_steps']}"
          f" ticks={arm_off['total_ticks']} f_commits={arm_off['n_f_commits']}"
          f" -> weak_regime={is_weak_natural_commit}"
          f" | ON intent={on_commit_intent} f_commits={arm_on['n_f_commits']}"
          f" rdir={arm_on['n_rule_directed_commit_ticks']} seq={arm_on['n_sequence_completions']}"
          f" closures={arm_on['n_closures']}"
          f" | opportunity={mech445_opportunity_met} m445={mech445_commit_intent_met}", flush=True)
    # Per-seed verdict for the runner: guard + in-regime + commit-intent met. (Governance scoring
    # is aggregate, in run_experiment; this line drives the progress bar only.)
    seed_pass = bool(guard_pass and is_weak_natural_commit and mech445_commit_intent_met)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} weak_regime={is_weak_natural_commit}"
          f" m445={mech445_commit_intent_met}"
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
        "off_committed_frac": off_committed_frac,
        "is_weak_natural_commit": is_weak_natural_commit,
        "on_commit_intent": on_commit_intent,
        "mech445_opportunity_met": mech445_opportunity_met,
        "mech445_commit_intent_met": mech445_commit_intent_met,
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

    # --- Regime partition on the guard-passing seeds (OFF-arm-anchored classifier) ---
    in_regime = [r for r in guard_passing if bool(r.get("is_weak_natural_commit", False))]
    n_in_regime = len(in_regime)
    n_strong = len(guard_passing) - n_in_regime  # context only
    regime_populated = bool(n_in_regime >= MIN_IN_REGIME_SEEDS)

    # --- Readiness (gate a): opportunity for the commit-intent counter on the in-regime subset ---
    opportunity_frac = _frac([bool(r.get("mech445_opportunity_met", False)) for r in in_regime])
    opportunity_met = bool(opportunity_frac >= ARM_PASS_FRACTION)

    # --- Science (gate b): the refractory-independent commit-intent counter on the in-regime subset ---
    mech445_frac = _frac([bool(r.get("mech445_commit_intent_met", False)) for r in in_regime])
    mech445_supported = bool(mech445_frac >= ARM_PASS_FRACTION)

    # Full readiness = every leg of gate (a) clears (self-route legs; NEVER a false weakens).
    readiness_met = bool(contact_non_vacuity_met and regime_populated and opportunity_met)

    # --- Routing ---
    if not contact_non_vacuity_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not regime_populated:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = (
            f"weak_natural_commit_regime_underpopulated"
            f"(n_in_regime={n_in_regime}<{MIN_IN_REGIME_SEEDS})"
        )
    elif not opportunity_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "commit_intent_opportunity_starved_on_in_regime_subset"
    else:
        # Readiness (gate a) CLEARED -- the science is now scorable (a genuine PASS or weaken).
        if mech445_supported:
            outcome, label = "PASS", "mech445_commit_intent_f_independent_existence_supported"
            route_reason = "commit_intent_gt0_on_ge_2of3_in_regime_seeds"
        else:
            outcome, label = "FAIL", "mech445_commit_intent_did_not_meet_regime_scoped_gate"
            route_reason = "readiness_met_but_commit_intent_below_2of3_on_in_regime_subset"

    # Evidence direction (single claim). Readiness unmet -> unknown + scoring-excluded (degenerate).
    if not readiness_met:
        evidence_direction = "unknown"
        evidence_direction_per_claim = {"MECH-445": "unknown"}
        non_degenerate = False
        non_degenerate_per_claim = {"MECH-445": False}
        degeneracy_reason = f"substrate_not_ready: {route_reason}"
    else:
        evidence_direction = "supports" if mech445_supported else "weakens"
        evidence_direction_per_claim = {"MECH-445": evidence_direction}
        non_degenerate = True
        non_degenerate_per_claim = {"MECH-445": True}
        degeneracy_reason = ""

    print(f"[{EXPERIMENT_TYPE}] contact={contact_non_vacuity_met} (guard {sum(guard_flags)}/{n})"
          f" | regime: in={n_in_regime} strong={n_strong} populated={regime_populated}"
          f" (min={MIN_IN_REGIME_SEEDS})"
          f" | opportunity={opportunity_met}(f={opportunity_frac:.3f})"
          f" | READINESS={readiness_met}"
          f" | m445={mech445_supported}(f={mech445_frac:.3f})"
          f" -> outcome={outcome} label={label} dir={evidence_direction}", flush=True)

    overall_pass = bool(outcome == "PASS")

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "n_in_regime_seeds": n_in_regime,
        "n_strong_natural_commit_seeds": n_strong,
        "weak_natural_commit_regime_populated": regime_populated,
        "min_in_regime_seeds": MIN_IN_REGIME_SEEDS,
        "opportunity_met": opportunity_met,
        "opportunity_fraction": opportunity_frac,
        "readiness_met": readiness_met,
        "mech445_commit_intent_supported": mech445_supported,
        "mech445_commit_intent_fraction": mech445_frac,
        "overall_pass": overall_pass,
        "route_reason": route_reason,
        "per_seed_guard_pass": guard_flags,
        "per_seed_off_committed_frac": [float(r.get("off_committed_frac", 0.0)) for r in per_seed],
        "per_seed_is_weak_regime": [bool(r.get("is_weak_natural_commit", False)) for r in per_seed],
        "per_seed_on_commit_intent": [int(r.get("on_commit_intent", 0)) for r in per_seed],
        "per_seed_mech445": [bool(r.get("mech445_commit_intent_met", False)) for r in per_seed],
        "in_regime_seeds": [int(r["seed"]) for r in in_regime],
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
                    "name": "weak_natural_commit_regime_populated",
                    "description": "The guard-passing WEAK-natural-commit subset (OFF-arm "
                                   "committed_frac <= WEAK_COMMIT_FRAC_MAX) has >= "
                                   "MIN_IN_REGIME_SEEDS members -- enough IN-REGIME seeds to certify "
                                   "the regime-scoped claim at 2/3. Below floor -> the 715 test-power "
                                   "hole (too few in-regime seeds) -> substrate_not_ready_requeue, "
                                   "NOT a weakens.",
                    "measured": float(n_in_regime),
                    "threshold": float(MIN_IN_REGIME_SEEDS),
                    "met": regime_populated,
                    "control": "ARM_ENTRY_OFF committed_frac classifier over guard-passing seeds",
                },
                {
                    "name": "commit_intent_opportunity_on_in_regime",
                    "description": "On >= 2/3 of the in-regime seeds the ON arm formed a goal-active "
                                   "rule-directed commitment (SET precondition) AND a sequence "
                                   "completed (a closure could fire) -- the commit-intent counter "
                                   "had an OPPORTUNITY. Below floor -> counter starved -> "
                                   "substrate_not_ready_requeue, NOT a weakens.",
                    "measured": opportunity_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": opportunity_met,
                    "control": "ARM_ENTRY_ON eval loop over the in-regime subset",
                },
            ],
            "criteria_non_degenerate": {
                "weak_regime_populated": regime_populated,
                "commit_intent_opportunity": opportunity_met,
            },
            "criteria": [
                {
                    "name": "mech445_commit_intent_2of3_in_regime",
                    "load_bearing": True,
                    "passed": mech445_supported,
                },
            ],
            "science_note": "MECH-445 (child A, closure->beta commit-intent): the refractory-"
                            "independent sd034_n_closure_commit_intent counter (counted BEFORE the "
                            "elevate/refractory gate, so the -- now dropped -- MECH-446 magnitude "
                            "lever cannot zero it) is > 0 on >= 2/3 of the guard-passing WEAK-"
                            "natural-commit (IN-REGIME) seeds. Regime is classified from the OFF "
                            "arm's F-driven committed fraction (a DIFFERENT arm + quantity than the "
                            "scored ON-arm counter -- not circular). This is the M2 test-power fix "
                            "for the 715 hole: a regime-scoped claim needs a POPULATED in-regime "
                            "subset, which a 3-seed/2-of-3 gate (~1 in-regime) cannot provide but "
                            "12 seeds (~4 in-regime) can.",
            "mech446_dropped_note": "The MECH-446 co-occurrence / within-arm around-closure "
                                    "occupancy-drop DV is DROPPED ENTIRELY -- this is a MECH-445-"
                                    "only test (claim_ids=[MECH-445]). MECH-446 remains routed to "
                                    "/implement-substrate on f_dominance_conversion_ceiling (per "
                                    "the 715 autopsy + claim_synthesis_MECH-445-446_2026-07-06.md). "
                                    "715 is NOT superseded -- it remains the MECH-446/co-occurrence "
                                    "record.",
            "levers_off_note": "The rung-6 NaturalCommitUrgencyRelease and the ARC-108 JOB-2 "
                               "dopaminergic DRIVER pair are OFF in every arm. The commit-intent "
                               "under test is the INTRINSIC SD-034 closure-plane commit-ENTRY "
                               "engagement.",
            "primitive_under_test": {
                "flag": "use_closure_commit_entry (ree-v3 main 84c1e7c; VALIDATED 460o/p PASS)",
                "latch": "e3._closure_committed_active (sticky, F-INDEPENDENT)",
                "mech445_counter": "beta_gate.sd034_n_closure_commit_intent (refractory-independent)",
                "regime_classifier": "ARM_ENTRY_OFF committed_frac <= WEAK_COMMIT_FRAC_MAX",
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
        "supersedes_note": "Does NOT supersede V3-EXQ-715. 715 remains the MECH-445+MECH-446 "
                           "co-occurrence record (routed to /implement-substrate for the MECH-446 "
                           "face). 717 is a narrower, higher-power MECH-445-only test (M2 of "
                           "claim_synthesis_MECH-445-446_2026-07-06.md).",
        "re_derive_brake_note": "MECH-445 carries 6 prior substrate_ceiling autopsies "
                                "(460h/i/j/k/l, 715); brake FIRED (threshold 2). RELEASED here: "
                                "(1) the named upstream substrate (use_closure_commit_entry / "
                                "F-independent closure-coupled-hold arming) is BUILT + VALIDATED "
                                "(V3-EXQ-460o/460p PASS 2026-06-24); (2) test-power redesign, NOT a "
                                "same-selector de-commit-MAGNITUDE re-derive -- the MECH-446 "
                                "magnitude face the brake protects is DROPPED. Brake-safe per "
                                "claim_synthesis_MECH-445-446_2026-07-06.md Sec.6.",
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
        "method_note": "MECH-445-ONLY commit-INTENT existence falsifier, REGIME-SCOPED (M2). Run 12 "
                       "seeds; classify each into weak/strong-natural-commit from the ARM_ENTRY_OFF "
                       "baseline's F-driven committed FRACTION (weak iff committed_frac <= "
                       "WEAK_COMMIT_FRAC_MAX); score the refractory-independent commit-intent "
                       "counter (sd034_n_closure_commit_intent) ONLY on the guard-passing WEAK "
                       "(IN-REGIME) subset. PASS = counter > 0 on >= 2/3 of THOSE. Gate (a) "
                       "READINESS (self-route, NEVER a weakens): contact guard on >= 2/3 seeds AND "
                       "the weak subset populated (>= MIN_IN_REGIME_SEEDS) AND the counter had an "
                       "opportunity (rule-directed commit + sequence completion) on >= 2/3 in-"
                       "regime seeds. Any readiness leg unmet -> substrate_not_ready_requeue "
                       "(non_degenerate=false). Gate (b) SCIENCE (falsifiable only once readiness "
                       "clears): commit-intent counter > 0 on >= 2/3 in-regime seeds; readiness-met-"
                       "but-counter-short -> genuine WEAKENS. The MECH-446 co-occurrence / within-"
                       "arm around-closure occupancy-drop DV is DROPPED ENTIRELY. PROMOTES NOTHING.",
        "arm_note": "ARMS = " + ", ".join(a["key"] for a in ARMS) + ". closure-exclusive eval + "
                    "beta-engagement coupling + natural-commit latch-hold ON in both arms; the only "
                    "variable is use_closure_commit_entry. ARM_ENTRY_OFF is the pure-F REGIME "
                    "CLASSIFIER arm (committed_frac -> weak/strong); ARM_ENTRY_ON is the SCIENCE "
                    "arm where the refractory-independent MECH-445 commit-intent counter is "
                    "measured.",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "arm_pass_fraction": ARM_PASS_FRACTION,
            "rule_directed_min_ticks": RULE_DIRECTED_MIN_TICKS,
            "min_commit_intent": MIN_COMMIT_INTENT,
            "weak_commit_frac_max": WEAK_COMMIT_FRAC_MAX,
            "min_in_regime_seeds": MIN_IN_REGIME_SEEDS,
            "n_seeds": len(SEEDS),
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
            "n_seeds": len(SEEDS),
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
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
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
