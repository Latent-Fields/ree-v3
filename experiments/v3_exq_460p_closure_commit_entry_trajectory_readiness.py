"""
V3-EXQ-460n: CLAIM-FREE substrate-readiness diagnostic for the F-INDEPENDENT
closure-plane commit-ENTRY *TRAJECTORY* primitive (use_closure_commit_entry_trajectory,
ree-v3 main 96ee30c) + a self-contained BOOL-vs-TRAJECTORY comparison. SIBLING of
V3-EXQ-460m (NOT a supersede): 460m certifies the bool latch (use_closure_commit_entry,
84c1e7c) 2-arm; 460n adds the TRAJECTORY arm and re-includes the bool arm so the
bool-vs-trajectory distinction is measured in ONE run on ONE trained substrate per seed.

WHY: the bool latch e3._closure_committed_active arms + SUSTAINS the closure-formed beta
occupancy (C-KEY), but a bare BOOL cannot be STEPPED -- the between-E3-tick path reads
e3._committed_trajectory to advance a committed PROGRAM, so a closure-armed hold with only a
bool falls through to repeating _last_action (no closure-formed program executes -- the
C-STEP gap). The trajectory extension installs a PARALLEL sticky
e3._closure_committed_trajectory the between-tick stepping consults, so the closure-formed
occupancy ADVANCES an actual committed program. This diagnostic measures whether the
trajectory arm buys that stepped program the bool arm cannot -- the evidence that informs
which latch the 460-lineage de-commit successor uses.

DESIGN doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  (closure-plane commit-ENTRY TRAJECTORY / C-STEP extension section).
UNIT contracts: ree-v3/tests/contracts/test_closure_commit_entry.py (bool C-KEY) +
  test_closure_commit_entry_trajectory.py (trajectory C-KEY/C-STEP/C-YIELD/C-OFF).

ARMS (one curriculum build per seed; the entry/trajectory flags toggled at eval via clone --
the latch + hold + trajectory carry no trainable parameters so the clone is exact). All arms
keep closure_exclusive_decommit_eval + use_closure_commit_beta_coupling +
use_natural_commit_latch_hold ON; the variables are the two commit-ENTRY flags:
  ARM_ENTRY_OFF  -- use_closure_commit_entry=False (trajectory off). The 460k/460l baseline;
                    the latch-hold arms ONLY via the fragile F-commit -> must NOT arm.
  ARM_BOOL       -- use_closure_commit_entry=True, trajectory=False. A's bool latch: arms +
                    sustains a beta occupancy F-independently, but cannot step a program.
  ARM_TRAJECTORY -- use_closure_commit_entry=True + use_closure_commit_entry_trajectory=True.
                    The corrected latch: arms + sustains AND the between-tick path STEPS the
                    closure-formed committed program.

The eval loop reads the closure-armed counter (ncl_hold_closure_armed_total), the longest
consecutive beta-elevated run (max_consecutive_beta_run), F-driven natural commits
(n_f_commits, expected ~0), and -- NEW -- closure_program_steps_total: between-E3-tick steps
where beta was elevated AND e3._closure_committed_trajectory was non-None AND the committed
step counter advanced (the closure-formed program executing). ARM_TRAJECTORY should show
closure_program_steps_total > 0; ARM_BOOL and ARM_ENTRY_OFF must show 0 (no trajectory to
step). The NON-VACUITY readout n_rule_directed_commit_ticks (goal active AND rule_state norm
>= floor) is the SET predicate's precondition -- if never met the test is starved, not
falsified -> substrate_not_ready_requeue.

PRE-REGISTERED GATES (claim-free):
  (a) PRIMARY occupancy (shared by both latch arms): armed_and_sustained
      (ncl_hold_closure_armed_total > 0 AND max_consecutive_beta_run >= SUSTAIN_MIN_TICKS) on
      ARM_BOOL AND ARM_TRAJECTORY on >= 2/3 guard seeds; ARM_ENTRY_OFF reproduces the
      460k/460l baseline (armed == 0).
  (b) C-STEP distinction (the bool-vs-trajectory evidence): ARM_TRAJECTORY
      closure_program_steps_total > 0 on >= 2/3 guard seeds; ARM_BOOL == 0. Tells us whether
      the trajectory latch executes a committed program the bool latch cannot.

HARNESS-FIX RE-ISSUE (supersedes V3-EXQ-460n; failure_autopsy_V3-EXQ-460m-460n_2026-06-23,
confirmed, user-routed): the predecessor's _eval_arm_behaviour loop NEVER called
agent.update_z_goal, so goal_state.is_active() was False at every eval tick and the
F-independent SET predicate's precondition (a goal-active rule-directed commitment)
was starved -> n_rule_directed_commit_ticks=0 on every seed -> the run self-routed
substrate_not_ready_requeue WITHOUT ever scoring the occupancy gates. The substrate
primitive (the F-independent commit-ENTRY writer at agent.py:6519-6532) is BUILT and
correct -- z_goal seeding demonstrably works on this exact trained agent (the
foraging_contact_guard PASSED in run_p2). This re-issue seeds z_goal each eval step
(consumption-gated, mirroring scaffolded_sd054_onboarding run_p2 / _eval_episode) so
the SET precondition has a goal-active tick to latch on. EVERYTHING ELSE IS IDENTICAL
(same arms, same gates, same self-route guards, same closure_exclusive_decommit_eval
substrate). NOT a substrate change; NOT the 460k structural-unreachability root.

experiment_purpose: diagnostic. claim_ids: [] (claim-free). NO governance weight.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
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

EXPERIMENT_TYPE = "v3_exq_460p_closure_commit_entry_trajectory_readiness"
QUEUE_ID = "V3-EXQ-460p"
CLAIM_IDS: List[str] = []  # claim-free substrate-readiness diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_CLOSURE_COMMIT_ENTRY_TRAJECTORY_READINESS_OFF_BOOL_TRAJ"

# --- Goal-pipeline / encoder dims (mirror 603n / 460l exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C; mirror 460l) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- closure-plane commit-ENTRY primitive (REEConfig defaults; ARM_ENTRY_ON) ---
CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR = 0.01  # rule_state norm floor for the SET predicate

# --- Within-arm around-closure window DV (secondary readout; mirror 460l) ---
CLOSURE_WINDOW = 10
WINDOW_MIN_TICKS = 3

# --- Curriculum budgets (mirror 603n / 460l exactly) ---
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
ARM_PASS_FRACTION = 2.0 / 3.0       # PRIMARY gate (a): armed_and_sustained on >= 2/3 seeds
SUSTAIN_MIN_TICKS = 2               # multi-tick closure-formed occupancy (load-bearing)
RULE_DIRECTED_MIN_TICKS = 1         # non-vacuity floor: >= 1 rule-directed commit tick

# --- Eval-arm definitions (the F-independent commit-ENTRY primitive toggled; closure-
#     exclusive de-commit eval ON in every arm). ---
ARM_OFF = "ARM_ENTRY_OFF"
ARM_BOOL = "ARM_BOOL"
ARM_TRAJ = "ARM_TRAJECTORY"
# entry = use_closure_commit_entry (bool latch); traj = use_closure_commit_entry_trajectory.
ARMS: List[Dict[str, Any]] = [
    {"key": ARM_OFF, "entry": False, "traj": False},   # 460k/460l baseline: must NOT arm
    {"key": ARM_BOOL, "entry": True, "traj": False},   # A's bool latch: arms + sustains
    {"key": ARM_TRAJ, "entry": True, "traj": True},    # corrected: arms + sustains + STEPS
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
    """603n-validated foraging substrate (mirror 460l) + the commitment control-plane +
    commitment-closure-control-plane amend Legs A/B/C + beta-engagement coupling +
    the closure-exclusive de-commit eval mode + the natural-commit latch-hold. The
    F-independent commit-ENTRY primitive (use_closure_commit_entry) is LEFT OFF here (the
    trained-base config); it is armed per-arm at eval by _clone_arm so both arms share one
    trained substrate (the latch + hold carry no trainable parameters). The JOB-2 DRIVER
    pair (rho ramp + habenula) and the rung-6 NaturalCommitUrgencyRelease are OFF in every
    arm (not under test here)."""
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
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D)
        use_closure_commit_beta_coupling=True,         # beta-engagement coupling
        # rung-6 natural-commit-occupancy-release lever: OFF in every arm.
        use_natural_commit_urgency_release=False,
        # The natural-commit LATCH-HOLD is ARMED on the base config (carried into every arm
        # via _clone_arm's deepcopy). The closure-exclusive eval + this hold are the regime
        # in which the F-independent commit-ENTRY primitive arms + sustains the occupancy.
        use_natural_commit_latch_hold=True,
        # The CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode (BUILT ree-v3 e52158d 2026-06-22): beta
        # elevation closure-EXCLUSIVE + the latch-hold arms on _closure_commit_active. ON in
        # EVERY arm. Preconditions (loud ValueError at REEAgent.__init__):
        # use_closure_commit_beta_coupling AND use_natural_commit_latch_hold -- both set.
        closure_exclusive_decommit_eval=True,
        # The F-INDEPENDENT commit-ENTRY primitive under test: OFF on the trained base;
        # armed per-arm at eval by _clone_arm. Preconditions (loud ValueError):
        # use_closure_commit_entry=True requires use_closure_commit_beta_coupling AND
        # use_natural_commit_latch_hold -- both set above.
        use_closure_commit_entry=False,
        closure_commit_entry_rule_norm_floor=CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
        # The C-STEP trajectory extension under test: OFF on the trained base; armed per-arm
        # at eval by _clone_arm. Precondition (loud ValueError): requires
        # use_closure_commit_entry (set per-arm) which itself requires the coupling + hold.
        use_closure_commit_entry_trajectory=False,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _arm_config_slice(arm: Dict[str, Any]) -> Dict[str, Any]:
    """The full per-arm config dict for the arm_fingerprint (the only inter-arm
    variable is use_closure_commit_entry; the rest is shared substrate config)."""
    return {
        "arm_key": arm["key"],
        "use_closure_commit_entry": bool(arm["entry"]),
        "use_closure_commit_entry_trajectory": bool(arm["traj"]),
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
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint
    tolerance-band completion so the SD-034 closure operator has completions to fire on
    (mirror 460l)."""
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
    """Clone the SAME trained weights into an agent built with this arm's commit-ENTRY
    config (use_closure_commit_entry on/off). The closure-exclusive eval + latch-hold +
    closure operator stay ON in every arm (the ENTRY primitive -- not the machinery -- is
    the variable). The latch + hold carry no trainable parameters, so the state_dict loads
    cleanly (mirrors 460l's _clone_arm)."""
    cfg = copy.deepcopy(trained_agent.config)
    cfg.use_closure_commit_entry = bool(arm["entry"])
    cfg.use_closure_commit_entry_trajectory = bool(arm["traj"])
    cfg.closure_commit_entry_rule_norm_floor = CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR
    # JOB-2 / rung-6 levers stay OFF in every arm.
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
    # HARNESS FIX (failure_autopsy_V3-EXQ-460m-460n_2026-06-23): copy the trained
    # substrate's LIVE goal-seeding calibration onto the clone. The scaffold writes
    # z_goal_seeding_gain / benefit_threshold / drive_floor onto agent.goal_state.config
    # (the live GoalConfig), NOT onto REEConfig, so the deepcopy above does NOT carry
    # them -> without this the clone keeps default GoalConfig (benefit_threshold=0.1) and
    # wild contact (~0.03) would not clear the seeding gate, leaving goal_state inactive.
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
    """For each closure fire at tick t, the beta-latch occupancy FRACTION in the
    pre-closure window [t-W, t) and the post-closure window (t, t+W] (the paired
    within-arm de-commit datum; mirror 460l, secondary readout)."""
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
    """Longest run of consecutive True (beta-elevated) ticks -- the sustained-occupancy
    proxy. ARM_ENTRY_ON should reach >= SUSTAIN_MIN_TICKS; ARM_ENTRY_OFF should not arm."""
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
    """Eval instrumented for the F-independent closure-plane commit-ENTRY primitive.
    Ticks the agent, reads the closure-armed counter (sum of per-episode
    agent._ncl_hold_closure_armed_count), the longest consecutive beta-elevated run, the
    F-driven natural commits (e3._committed_trajectory is not None AFTER select -- expected
    ~0), the SD-034 closure fire count, and the non-vacuity readout
    n_rule_directed_commit_ticks (goal active AND rule_state norm >= floor). Calls
    agent.update_residue() each tick so the waking post-action path runs (mirrors 460l)."""
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
    # HARNESS FIX (failure_autopsy_V3-EXQ-460m-460n_2026-06-23): the prior 460m/460n eval
    # loop NEVER called agent.update_z_goal, so goal_state.is_active() was False every tick
    # -> the F-independent SET predicate's precondition (goal-active rule-directed
    # commitment) was starved -> n_rule_directed_commit_ticks=0 -> substrate_not_ready_
    # requeue on every seed. Seed z_goal each step (below, after env.step), gated on contact
    # exactly like scaffolded_sd054_onboarding run_p2 / _eval_episode. The raw-benefit
    # seeding gate mirrors _reconciled_gating_threshold:
    #   floor = benefit_threshold / (z_goal_seeding_gain * (1 + drive_weight * drive_floor)).
    # The clone carries the trained calibration (see _clone_arm), so this matches the regime
    # the foraging_contact_guard certified.
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
    # F-driven natural commit count (expected ~0 on this substrate -- the point).
    n_f_commits = 0
    # C-STEP readout: between-E3-tick steps where a closure-FORMED committed program
    # advanced (beta elevated AND e3._closure_committed_trajectory non-None AND the committed
    # step counter advanced -- the 4900 stepping union consumed the closure trajectory). > 0
    # only on ARM_TRAJECTORY; exactly 0 on ARM_BOOL / ARM_ENTRY_OFF (no trajectory to step).
    closure_program_steps_total = 0
    # NON-VACUITY readout: goal active AND rule_state norm >= floor (the SET precondition).
    n_rule_directed_commit_ticks = 0
    rule_state_norm_peak = 0.0
    # Committed first-action class histogram (diversity context).
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
            # C-STEP readout: capture the committed step counter + whether this is a
            # between-E3-tick (the path that steps a committed trajectory) BEFORE selection.
            _is_between_tick = not bool(ticks.get("e3_tick"))
            _step_idx_before = int(getattr(agent, "_committed_step_idx", 0))
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
            # C-STEP: a closure-FORMED committed program stepped this tick (between-tick +
            # beta elevated + closure trajectory present + step counter advanced). A bool /
            # no latch -> _closure_committed_trajectory is None -> never counted.
            if (
                _is_between_tick
                and cur_beta
                and agent.e3._closure_committed_trajectory is not None
                and int(getattr(agent, "_committed_step_idx", 0)) > _step_idx_before
            ):
                closure_program_steps_total += 1
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

            # NON-VACUITY: did the SET predicate have a goal-active rule-directed commitment
            # to latch on? (goal_state.is_active() AND rule_state norm >= floor).
            gs = getattr(agent, "goal_state", None)
            goal_active = bool(gs is not None and gs.is_active())
            rs_norm = _rule_state_norm(agent)
            if rs_norm > rule_state_norm_peak:
                rule_state_norm_peak = rs_norm
            if goal_active and rs_norm >= rule_floor:
                n_rule_directed_commit_ticks += 1

            _, _harm, done, info, obs_dict = env.step(action_idx)

            # Drive the waking post-action path (mirrors 460l; identical residue dynamics).
            agent.update_residue(harm_signal=float(_harm), hypothesis_tag=False)

            # HARNESS FIX: seed z_goal from the POST-step body-state so goal_state.is_active()
            # can be True during the closure-eval -- the F-independent SET predicate's missing
            # precondition. Mirrors run_p2 / _eval_episode consumption-gated seeding
            # (benefit=obs_body[11], drive=clip(1-energy,0,1) via energy=obs_body[3]).
            # Contact-gated: a sub-seeding whiff is SKIPPED (not decay-only updated) so the
            # consolidated trace is protected from washout (the V3-EXQ-634b lesson); a genuine
            # contact step seeds. update_z_goal also binds the SD-057 per-type token (rt).
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
        "closure_program_steps_total": closure_program_steps_total,
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
        "n_f_commits": 0, "closure_program_steps_total": 0,
        "n_rule_directed_commit_ticks": 0, "rule_state_norm_peak": 0.0,
        "n_sequence_completions": 0, "n_eval_episodes": 0, "closure_present": False,
        "env_hook_enabled": False, "n_window_events": 0, "mean_pre_closure_occ": 0.0,
        "mean_post_closure_occ": 0.0, "committed_class_entropy_n_classes": 0,
    }


def _arm_armed_and_sustained(arm: Dict[str, Any]) -> bool:
    """PRIMARY gate (a) per-seed predicate (ARM_ENTRY_ON): the latch armed
    (ncl_hold_closure_armed_total > 0) AND the occupancy sustained
    (max_consecutive_beta_run >= SUSTAIN_MIN_TICKS) -- a multi-tick closure-formed hold."""
    return bool(
        int(arm.get("ncl_hold_closure_armed_total", 0)) > 0
        and int(arm.get("max_consecutive_beta_run", 0)) >= SUSTAIN_MIN_TICKS
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
        "bool_armed_and_sustained": False,
        "traj_armed_and_sustained": False,
        "off_did_not_arm": False,
        "traj_steps_program": False,
        "bool_no_steps": False,
        "on_rule_directed_met": False,
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
    # toggled). Each (seed x arm) cell is wrapped in arm_cell (resets RNG on enter, stamps
    # the fingerprint) -- the multi-arm arm_fingerprint obligation.
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
    arm_bool = arms_out[ARM_BOOL]
    arm_traj = arms_out[ARM_TRAJ]

    # PRIMARY gate (a): both latch arms arm + sustain a closure-formed occupancy.
    bool_armed_and_sustained = _arm_armed_and_sustained(arm_bool)
    traj_armed_and_sustained = _arm_armed_and_sustained(arm_traj)
    off_did_not_arm = bool(int(arm_off.get("ncl_hold_closure_armed_total", 0)) == 0)
    # (b) C-STEP distinction: only the trajectory arm steps a committed program.
    traj_steps_program = bool(int(arm_traj.get("closure_program_steps_total", 0)) > 0)
    bool_no_steps = bool(int(arm_bool.get("closure_program_steps_total", 0)) == 0)
    # NON-VACUITY: the SET predicate fired (goal-active rule-directed commitment). Identical
    # predicate in both latch arms; measured on the trajectory arm.
    on_rule_directed_met = bool(
        int(arm_traj.get("n_rule_directed_commit_ticks", 0)) >= RULE_DIRECTED_MIN_TICKS
    )

    print(f"  [eval] arm_eval seed={seed} ep {done}/{total_eps}"
          f" | OFF armed={arm_off['ncl_hold_closure_armed_total']}"
          f" run={arm_off['max_consecutive_beta_run']} steps={arm_off['closure_program_steps_total']}"
          f" | BOOL armed={arm_bool['ncl_hold_closure_armed_total']}"
          f" run={arm_bool['max_consecutive_beta_run']} steps={arm_bool['closure_program_steps_total']}"
          f" f_commits={arm_bool['n_f_commits']}"
          f" | TRAJ armed={arm_traj['ncl_hold_closure_armed_total']}"
          f" run={arm_traj['max_consecutive_beta_run']} steps={arm_traj['closure_program_steps_total']}"
          f" f_commits={arm_traj['n_f_commits']} rdir={arm_traj['n_rule_directed_commit_ticks']}"
          f" rs_peak={arm_traj['rule_state_norm_peak']:.3f}"
          f" | bool_armed={bool_armed_and_sustained} traj_armed={traj_armed_and_sustained}"
          f" off_no_arm={off_did_not_arm} traj_steps={traj_steps_program}"
          f" bool_no_steps={bool_no_steps} rdir_met={on_rule_directed_met}", flush=True)
    seed_pass = bool(
        guard_pass and on_rule_directed_met
        and bool_armed_and_sustained and traj_armed_and_sustained
        and traj_steps_program and bool_no_steps
    )
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} rdir_met={on_rule_directed_met}"
          f" bool_armed={bool_armed_and_sustained} traj_armed={traj_armed_and_sustained}"
          f" off_did_not_arm={off_did_not_arm} traj_steps={traj_steps_program}"
          f" bool_no_steps={bool_no_steps}"
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
        "bool_armed_and_sustained": bool_armed_and_sustained,
        "traj_armed_and_sustained": traj_armed_and_sustained,
        "off_did_not_arm": off_did_not_arm,
        "traj_steps_program": traj_steps_program,
        "bool_no_steps": bool_no_steps,
        "on_rule_directed_met": on_rule_directed_met,
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

    # Non-vacuity precondition: a goal-active rule-directed commitment the latch could SET on
    # (>= RULE_DIRECTED_MIN_TICKS, measured on ARM_TRAJECTORY) on >= 2/3 guard seeds.
    rdir_flags = [bool(r.get("on_rule_directed_met", False)) for r in guard_passing]
    rdir_frac = _frac(rdir_flags)
    rule_directed_met = bool(rdir_frac >= ARM_PASS_FRACTION)
    rdir_measured = (
        min(int(r["arms"][ARM_TRAJ].get("n_rule_directed_commit_ticks", 0))
            for r in guard_passing)
        if guard_passing else 0
    )

    # PRIMARY gate (a): BOTH latch arms (ARM_BOOL + ARM_TRAJECTORY) arm + sustain on >= 2/3.
    bool_armed_flags = [bool(r.get("bool_armed_and_sustained", False)) for r in guard_passing]
    traj_armed_flags = [bool(r.get("traj_armed_and_sustained", False)) for r in guard_passing]
    bool_armed_frac = _frac(bool_armed_flags)
    traj_armed_frac = _frac(traj_armed_flags)
    both_armed_met = bool(
        bool_armed_frac >= ARM_PASS_FRACTION and traj_armed_frac >= ARM_PASS_FRACTION
    )

    # Secondary contrast: ARM_ENTRY_OFF reproduces the 460k/460l baseline (armed == 0).
    off_flags = [bool(r.get("off_did_not_arm", False)) for r in guard_passing]
    off_frac = _frac(off_flags)

    # (b) C-STEP distinction: ARM_TRAJECTORY steps a committed program (> 0) AND ARM_BOOL
    # does not (== 0) on >= 2/3 guard seeds -- the bool-vs-trajectory evidence.
    traj_steps_flags = [bool(r.get("traj_steps_program", False)) for r in guard_passing]
    bool_no_steps_flags = [bool(r.get("bool_no_steps", False)) for r in guard_passing]
    traj_steps_frac = _frac(traj_steps_flags)
    bool_no_steps_frac = _frac(bool_no_steps_flags)
    cstep_met = bool(
        traj_steps_frac >= ARM_PASS_FRACTION and bool_no_steps_frac >= ARM_PASS_FRACTION
    )

    # Non-degeneracy: occupancy contrast (OFF armed==0 while BOTH latch arms armed>0) AND a
    # C-STEP contrast (TRAJ steps>0 while BOOL steps==0) on >= 1 guard seed each.
    occ_contrast_seeds = [
        r for r in guard_passing
        if int(r["arms"][ARM_OFF].get("ncl_hold_closure_armed_total", 0)) == 0
        and int(r["arms"][ARM_BOOL].get("ncl_hold_closure_armed_total", 0)) > 0
        and int(r["arms"][ARM_TRAJ].get("ncl_hold_closure_armed_total", 0)) > 0
    ]
    cstep_contrast_seeds = [
        r for r in guard_passing
        if int(r["arms"][ARM_TRAJ].get("closure_program_steps_total", 0)) > 0
        and int(r["arms"][ARM_BOOL].get("closure_program_steps_total", 0)) == 0
    ]
    occ_non_degenerate = bool(len(occ_contrast_seeds) > 0)
    cstep_non_degenerate = bool(len(cstep_contrast_seeds) > 0)

    # Routing. Claim-free diagnostic: PASS = substrate ready (BOTH latch arms arm + sustain a
    # closure-formed occupancy -- the de-commit successor's precondition). A non-vacuity
    # failure self-routes substrate_not_ready_requeue (NEVER a *_ceiling / does_not_support
    # label -- below-floor is starved, not falsified). The C-STEP distinction (b) does NOT
    # gate the occupancy-readiness PASS; it is the bool-vs-trajectory EVIDENCE that labels the
    # interpretation (which latch the de-commit successor uses).
    if not contact_non_vacuity_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not rule_directed_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route_reason = "closure_rule_directed_commit_not_formed"
    elif both_armed_met and cstep_met:
        outcome = "PASS"
        label = "trajectory_arms_sustains_and_steps_bool_only_sustains"
        route_reason = "both_latch_arms_armed_2of3_and_cstep_distinction_holds"
    elif both_armed_met:
        # Occupancy ready on BOTH latch arms, but the C-STEP distinction did not hold
        # (trajectory did not step on >= 2/3, or bool stepped). Still occupancy-ready (the
        # de-commit successor's precondition is met -- bool may suffice); the trajectory
        # primitive's added value is unconfirmed -- diagnose the SET/stepping before routing
        # the successor to the trajectory latch.
        outcome = "PASS"
        label = "occupancy_ready_cstep_distinction_unmet"
        route_reason = "both_latch_arms_armed_2of3_but_cstep_distinction_unmet"
    else:
        # Preconditions met but at least one latch arm did not arm/sustain on >= 2/3 -- a
        # genuine readiness FAIL, NOT a starved test.
        outcome = "FAIL"
        label = "closure_commit_entry_did_not_arm"
        route_reason = "rule_directed_commit_formed_but_a_latch_arm_did_not_arm_or_sustain"

    print(f"[{EXPERIMENT_TYPE}] contact={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) rule_directed={rule_directed_met} (frac={rdir_frac:.3f})"
          f" | gate_a both_armed={both_armed_met} (bool={bool_armed_frac:.3f} traj={traj_armed_frac:.3f})"
          f" off_did_not_arm_frac={off_frac:.3f} occ_non_degenerate={occ_non_degenerate}"
          f" | gate_b cstep={cstep_met} (traj_steps={traj_steps_frac:.3f} bool_no_steps={bool_no_steps_frac:.3f})"
          f" cstep_non_degenerate={cstep_non_degenerate}"
          f" -> outcome={outcome} label={label}", flush=True)

    overall_pass = bool(outcome == "PASS")

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "rule_directed_met": rule_directed_met,
        "rule_directed_fraction": rdir_frac,
        "rule_directed_min_across_traj_seeds": rdir_measured,
        # gate (a) occupancy (both latch arms):
        "both_armed_and_sustained_met": both_armed_met,
        "bool_armed_and_sustained_fraction": bool_armed_frac,
        "traj_armed_and_sustained_fraction": traj_armed_frac,
        "off_did_not_arm_fraction": off_frac,
        "occ_non_degenerate": occ_non_degenerate,
        # gate (b) C-STEP distinction:
        "cstep_met": cstep_met,
        "traj_steps_program_fraction": traj_steps_frac,
        "bool_no_steps_fraction": bool_no_steps_frac,
        "cstep_non_degenerate": cstep_non_degenerate,
        "overall_pass": overall_pass,
        "per_seed_guard_pass": guard_flags,
        "per_seed_bool_armed_and_sustained": [
            bool(r.get("bool_armed_and_sustained", False)) for r in per_seed
        ],
        "per_seed_traj_armed_and_sustained": [
            bool(r.get("traj_armed_and_sustained", False)) for r in per_seed
        ],
        "per_seed_off_did_not_arm": [
            bool(r.get("off_did_not_arm", False)) for r in per_seed
        ],
        "per_seed_traj_steps_program": [
            bool(r.get("traj_steps_program", False)) for r in per_seed
        ],
        "per_seed_bool_no_steps": [
            bool(r.get("bool_no_steps", False)) for r in per_seed
        ],
        "route_reason": route_reason,
    }

    return {
        "outcome": outcome,
        "acceptance": acceptance,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "closure_rule_directed_commit_formed",
                    "description": "ARM_TRAJECTORY forms a goal-active rule-directed "
                                   "commitment the latch can set on (goal_state.is_active() "
                                   "AND lateral_pfc.rule_state norm >= "
                                   "closure_commit_entry_rule_norm_floor on >= 2/3 guard "
                                   "seeds). Below floor -> the SET predicate is starved, not "
                                   "falsified -> substrate_not_ready_requeue (NEVER a "
                                   "*_ceiling / does_not_support label).",
                    "measured": rdir_measured,
                    "threshold": RULE_DIRECTED_MIN_TICKS,
                    "met": rule_directed_met,
                    "control": "ARM_TRAJECTORY eval loop",
                },
                {
                    "name": "foraging_contact_guard",
                    "description": "603n G2+G3: per-seed P2 contact_rate > 0 AND "
                                   "z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds (the "
                                   "trained substrate foraged + seeded ecologically).",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout",
                },
            ],
            "criteria_non_degenerate": {
                # (a) occupancy contrast non-degenerate iff ARM_ENTRY_OFF armed == 0 while
                # BOTH latch arms armed > 0 on >= 1 guard seed (the arms genuinely differ).
                "gate_a_occupancy": occ_non_degenerate,
                # (b) C-STEP contrast non-degenerate iff ARM_TRAJECTORY steps > 0 while
                # ARM_BOOL steps == 0 on >= 1 guard seed (the trajectory is not a bool).
                "gate_b_cstep": cstep_non_degenerate,
            },
            "criteria": [
                {
                    "name": "gate_a_both_latch_arms_armed_and_sustained_2of3",
                    "load_bearing": True,
                    "passed": both_armed_met,
                },
                {
                    "name": "gate_b_cstep_traj_steps_bool_does_not_2of3",
                    "load_bearing": False,  # the bool-vs-trajectory EVIDENCE, not a readiness gate
                    "passed": cstep_met,
                },
            ],
            "secondary_contrast": {
                "off_did_not_arm_fraction": off_frac,
                "traj_steps_program_fraction": traj_steps_frac,
                "bool_no_steps_fraction": bool_no_steps_frac,
                "note": "ARM_ENTRY_OFF should reproduce the 460k/460l baseline (armed == 0). "
                        "The load-bearing PASS is gate (a) occupancy on BOTH latch arms; gate "
                        "(b) C-STEP is the bool-vs-trajectory EVIDENCE (which latch the "
                        "de-commit successor uses), not the occupancy-readiness gate.",
            },
            "bool_vs_trajectory_note": "ARM_BOOL (use_closure_commit_entry) and ARM_TRAJECTORY "
                                       "(+ use_closure_commit_entry_trajectory) both sustain a "
                                       "closure-formed beta occupancy (gate a). The C-STEP "
                                       "readout closure_program_steps_total distinguishes them: "
                                       "> 0 on ARM_TRAJECTORY (the between-tick path advances the "
                                       "closure-formed committed program via "
                                       "e3._closure_committed_trajectory) and == 0 on ARM_BOOL "
                                       "(a bare bool cannot be stepped -> falls through to repeat "
                                       "_last_action). If the de-commit DV (MECH-446 "
                                       "occupancy-drop) needs only occupancy, the cheaper bool "
                                       "latch suffices; if it needs a stepped program, route the "
                                       "successor to the trajectory latch.",
            "f_independence_note": "n_f_commits per arm/seed expected ~0 (the occupancy is "
                                   "F-INDEPENDENT -- e3._committed_trajectory stays None while "
                                   "the closure-plane commit-ENTRY latch arms the hold).",
            "primitive_under_test": {
                "bool_flag": "use_closure_commit_entry (ree-v3 main 84c1e7c)",
                "trajectory_flag": "use_closure_commit_entry_trajectory (ree-v3 main 96ee30c)",
                "bool_latch": "e3._closure_committed_active (sticky, F-INDEPENDENT)",
                "trajectory_latch": "e3._closure_committed_trajectory (sticky, F-INDEPENDENT)",
                "occupancy_counter": "agent._ncl_hold_closure_armed_count",
                "cstep_counter": "closure_program_steps_total (between-tick committed-program steps)",
                "rule_norm_floor": CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
                "note": "closure_exclusive_decommit_eval + use_closure_commit_beta_coupling "
                        "+ use_natural_commit_latch_hold ON in ALL arms; the variables are the "
                        "two commit-ENTRY flags. The JOB-2 rho ramp / habenula + the rung-6 "
                        "NaturalCommitUrgencyRelease are OFF in all arms.",
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
        "supersedes": "V3-EXQ-460n",  # harness-fix re-issue (z_goal seeded in eval loop)
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> "
                     "P0 -> Stage-H -> P1 -> P2; 603n config) + commitment control-plane "
                     "(bistable BetaGate + SD-034 ClosureOperator + SD-033a LateralPFC + "
                     "SD-032 dACC/salience) + subgoal_mode waypoint tolerance-band completion "
                     "+ commitment-closure-control-plane Legs A/B/C + beta-engagement coupling "
                     "+ the natural-commit LATCH-HOLD + the CLOSURE-EXCLUSIVE DE-COMMIT EVAL "
                     "mode (closure_exclusive_decommit_eval, ree-v3 e52158d) ARMED in ALL "
                     "arms. The two F-independent closure-plane commit-ENTRY flags "
                     "(use_closure_commit_entry, ree-v3 84c1e7c; use_closure_commit_entry_"
                     "trajectory, ree-v3 96ee30c) toggled per arm "
                     "(ENTRY_OFF/BOOL/TRAJECTORY). rung-6 NaturalCommitUrgencyRelease + JOB-2 "
                     "DRIVER pair OFF in all arms.",
        "condition": CONDITION_LABEL,
        "method_note": "CLAIM-FREE substrate-readiness diagnostic for the F-independent "
                       "closure-plane commit-ENTRY TRAJECTORY primitive "
                       "(use_closure_commit_entry_trajectory) + a self-contained "
                       "bool-vs-trajectory comparison. SIBLING of V3-EXQ-460m (NOT a "
                       "supersede). Three eval arms on one trained substrate per seed (the "
                       "entry/trajectory flags toggled at eval via clone): ARM_ENTRY_OFF (the "
                       "460k/460l baseline; must NOT arm) / ARM_BOOL "
                       "(use_closure_commit_entry; arms + sustains an occupancy but cannot "
                       "step a program) / ARM_TRAJECTORY (+ use_closure_commit_entry_"
                       "trajectory; arms + sustains AND the between-tick path STEPS the "
                       "closure-formed committed program). PRIMARY load-bearing gate (a): "
                       "armed_and_sustained (ncl_hold_closure_armed_total > 0 AND "
                       "max_consecutive_beta_run >= SUSTAIN_MIN_TICKS) on BOTH latch arms on "
                       ">= 2/3 guard seeds. Gate (b) C-STEP (bool-vs-trajectory evidence, NOT "
                       "the readiness gate): ARM_TRAJECTORY closure_program_steps_total > 0 "
                       "while ARM_BOOL == 0 on >= 2/3. Non-vacuity precondition: a goal-active "
                       "rule-directed commitment formed (else substrate_not_ready_requeue, "
                       "NEVER a falsification). claim_ids=[] -- certifies the substrate is "
                       "ready (and which latch the de-commit successor uses) before any "
                       "de-commit falsifier (a 460-lineage successor) is scored.",
        "arm_note": "ARMS = " + ", ".join(a["key"] for a in ARMS) + ". closure-exclusive eval "
                    "+ beta-engagement coupling + natural-commit latch-hold ON in all arms; "
                    "the variables are the two commit-ENTRY flags. ARM_ENTRY_OFF is the "
                    "460k/460l no-arm baseline; ARM_BOOL adds use_closure_commit_entry (bool "
                    "latch); ARM_TRAJECTORY adds use_closure_commit_entry_trajectory (the "
                    "C-STEP extension that steps a closure-formed committed program).",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "arm_pass_fraction": ARM_PASS_FRACTION,
            "sustain_min_ticks": SUSTAIN_MIN_TICKS,
            "rule_directed_min_ticks": RULE_DIRECTED_MIN_TICKS,
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
            "cstep_readout": "closure_program_steps_total (ARM_TRAJECTORY > 0; ARM_BOOL == 0)",
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
            "config_basis": "closure-exclusive de-commit eval (ree-v3 main e52158d, "
                            "2026-06-22) + F-independent closure-plane commit-ENTRY bool latch "
                            "(use_closure_commit_entry, ree-v3 main 84c1e7c, 2026-06-23) + the "
                            "C-STEP TRAJECTORY extension (use_closure_commit_entry_trajectory, "
                            "ree-v3 main 96ee30c, 2026-06-23)",
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
