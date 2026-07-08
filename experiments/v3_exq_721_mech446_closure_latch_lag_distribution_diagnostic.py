"""
V3-EXQ-721: MECH-446 CLOSURE-FIRE -> LATCH-ARMED-OCCUPANCY LAG-DISTRIBUTION DIAGNOSTIC
(Move M3 of claim_synthesis_MECH-445-446_2026-07-06.md; PROMOTES NOTHING).

experiment_purpose: DIAGNOSTIC. claim_ids: [] (non_contributory; measures, does not test).

WHY THIS EXISTS (the M3 reroute, precondition now met):
  The MECH-446 de-commit-authority DV (within-arm post-closure occupancy DROP) has come up
  VACUOUS on every iteration of the 460h/460i/715/715a/717 lineage: the SD-034 closure FIRE
  (the de-commit trigger) and the latch-armed sustained beta OCCUPANCY (formed by the
  F-independent closure-plane commit-ENTRY, use_closure_commit_entry) are TEMPORALLY DISJOINT
  within-episode, so the pre/post window has no occupancy to measure a drop on (n_window_events
  ~ 0, mean_pre_closure_occ ~ 0). V3-EXQ-715a (Move M1, selection-face lift ON) did NOT
  co-locate them (moderate_f_delta -63; arming stayed 1/3) -- which is exactly M3's stated
  precondition ("the fallback if M1 does not co-locate"). The re-derive brake is FIRED for the
  same-selector de-commit-MAGNITUDE face (7 substrate-ceiling readings), so re-running the
  falsifier is REFUSED. This experiment does something different and brake-EXEMPT: it does not
  re-test the de-commit magnitude at all. It RE-ANCHORS the DV from a pass/fail drop to a GRADED
  MEASUREMENT -- the distribution of the temporal LAG between each genuine closure fire and the
  nearest sustained latch-armed occupancy. "Vacuous 0/3" becomes a number that tells the
  /implement-substrate de-commit-release build EXACTLY how much co-registration its amend must
  buy (e.g. "closures land a median N ticks from the nearest armed occupancy; the amend must
  pull them within CLOSURE_WINDOW").

WHAT IS MEASURED (the M3 DV, guarded for closure-CAUSED attribution):
  On the F-independent closure-plane commit-ENTRY substrate (use_closure_commit_entry, ree-v3
  main 84c1e7c; VALIDATED arming+sustaining by V3-EXQ-460o/460p PASS 2026-06-24), on the ON arm,
  for EACH genuine SD-034 closure fire (closure_operator._n_closures increment -- the de-commit
  trigger, NOT a generic co-occurrence) compute the signed + absolute tick LAG to the nearest
  tick belonging to a SUSTAINED latch-armed beta occupancy (a beta-elevated run of length >=
  SUSTAIN_MIN_TICKS in the SAME episode). Aggregate across fires x seeds:
    - abs-lag distribution: min / p25 / median / p75 / mean / max
    - co-registration rate at horizon W = fraction of fires with an occupancy within W ticks
      (this is the fraction the falsifier's window could ever have scored)
    - fires-with-any-occupancy fraction (fires whose episode had a sustained occupancy at all)
  ATTRIBUTION GUARD (M3 requirement): the eval is closure_exclusive_decommit_eval=True, so beta
  elevation is CLOSURE-EXCLUSIVE (_commit_for_beta driven ONLY by _closure_commit_active; the
  F-driven result.committed path is suppressed) -- every measured occupancy is closure-formed by
  construction, NOT a generic beta co-occurrence. The ARM_ENTRY_OFF arm (no entry primitive)
  forms no occupancy (ncl_hold_closure_armed_total == 0), so its lag is censored -- the negative
  control confirming the occupancy is entry-driven.

WHAT IS NOT DONE: no de-commit MAGNITUDE gate, no MECH-445/446 co-occurrence PASS/FAIL, no claim
  direction. The within-arm occupancy-drop metrics (mean_pre/post_closure_occ, n_window_events)
  are still REPORTED as context (to show the drop DV is starved) but are NOT load-bearing.

READINESS (self-route, NEVER a verdict): the lag distribution is only defined on a seed that is
  curriculum-competent (guard) AND produced BOTH a sustained latch-armed occupancy AND >= 1
  closure fire in the same episode (so a lag exists to measure). If ZERO seeds are measurable ->
  substrate_not_ready_requeue (non_degenerate=false; NEVER substrate_ceiling / does_not_support /
  *_nondiscriminative). Unlike the 715 falsifier this needs only >= 1 measurable seed (a
  distribution can start from the rare arming seed the falsifier found, e.g. 715 seed 44 =
  398 commit-intents), not 2/3 arming -- extracting graded data from the SAME regime where the
  drop DV was vacuous is the whole point.

MECH-094: the closure-entry latch SET is a WAKING control-state transition (no replay / no
  memory-write surface) -- agent.update_residue(hypothesis_tag=False). Ethics preflight:
  all-false / decision allow (V3 pre-ethical instrumentation; SENT-0 boundary).

DESIGN doc: REE_assembly/evidence/planning/claim_synthesis_MECH-445-446_2026-07-06.md (Move M3);
route context failure_autopsy_MECH-445-cluster-715a-717_2026-07-07. Harness MIRRORS the VALIDATED
V3-EXQ-715 (same curriculum, same closure-exclusive eval substrate, same arm_cell fingerprint);
only the DV read off (beta_history, fire_ticks) is swapped (drop -> lag distribution).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

EXPERIMENT_TYPE = "v3_exq_721_mech446_closure_latch_lag_distribution_diagnostic"
QUEUE_ID = "V3-EXQ-721"
CLAIM_IDS: List[str] = []  # DIAGNOSTIC -- measures, tags no claim; PROMOTES NOTHING.
EXPERIMENT_PURPOSE = "diagnostic"

# 6 seeds: includes 715's arming seed 44 + 717's armed seed 46, to robustly capture >= 1-2
# measurable (arming) seeds for a real lag distribution while staying a cheap diagnostic.
SEEDS = [42, 43, 44, 45, 46, 47]
CONDITION_LABEL = "CLOSURE_LATCH_LAG_DISTRIBUTION_ENTRY_OFF_ON"

# --- Goal-pipeline / encoder dims (mirror 715/460o exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C; mirror 715) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- closure-plane commit-ENTRY primitive (REEConfig defaults; ARM_ENTRY_ON) ---
CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR = 0.01

# --- Around-closure window (reported CONTEXT DV -- the vacuous drop; mirror 715) ---
CLOSURE_WINDOW = 10           # also the co-registration horizon W for the lag DV
WINDOW_MIN_TICKS = 3

# --- Curriculum budgets (mirror 715/460o exactly) ---
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

# --- Pre-registered thresholds ---
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
SUSTAIN_MIN_TICKS = 2               # a "sustained latch-armed occupancy" run length
LAG_HORIZON = CLOSURE_WINDOW        # co-registration horizon W (ticks)
MIN_MEASURABLE_SEEDS = 1            # readiness: >= 1 seed with a defined lag
MIN_LAG_EVENTS = 2                  # non-degeneracy: >= this many fire-with-occupancy events

# --- Eval-arm definitions (only use_closure_commit_entry toggled; mirror 715) ---
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
    """715/460o-validated foraging substrate + commitment-closure-control-plane amend Legs A/B/C
    + beta-engagement coupling + the closure-exclusive de-commit eval + the natural-commit
    latch-hold. use_closure_commit_entry is LEFT OFF on the trained base (armed per-arm at eval by
    _clone_arm; latch + hold carry no trainable parameters). rung-6 + ARC-108 driver OFF."""
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
        use_closure_env_completion_hook=True,          # Leg A
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base (MECH-446 refractory)
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D)
        use_closure_commit_beta_coupling=True,         # beta-engagement coupling (MECH-445 path)
        use_natural_commit_urgency_release=False,      # rung-6 lever OFF (parked line)
        use_natural_commit_latch_hold=True,
        closure_exclusive_decommit_eval=True,          # ON in EVERY arm (attribution guard)
        use_closure_commit_entry=False,                # toggled per-arm at eval by _clone_arm
        closure_commit_entry_rule_norm_floor=CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _arm_config_slice(arm: Dict[str, Any]) -> Dict[str, Any]:
    """Full per-arm config dict for the arm_fingerprint (the only inter-arm variable is
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
    """P2-config foraging env WITH subgoal_mode + waypoint tolerance-band completion so the SD-034
    closure operator has completions to fire on (mirror 715)."""
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
    """Clone the SAME trained weights into an agent built with this arm's commit-ENTRY config.
    Closure-exclusive eval + latch-hold + closure operator stay ON in every arm (the ENTRY
    primitive is the only variable). Mirror 715/460o."""
    cfg = copy.deepcopy(trained_agent.config)
    cfg.use_closure_commit_entry = bool(arm["entry"])
    cfg.closure_commit_entry_rule_norm_floor = CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR
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
    # HARNESS FIX (carried from 715/460o): copy the trained substrate's LIVE goal-seeding
    # calibration onto the clone (the scaffold writes it onto goal_state.config, not REEConfig).
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
    """CONTEXT DV (the vacuous drop; mirror 715). Pre/post beta-occupancy fraction at each fire."""
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


def _sustained_occupancy_ticks(beta_history: List[bool], min_run: int) -> List[int]:
    """Tick indices belonging to a beta-elevated run of length >= min_run (the sustained
    latch-armed occupancy; in the closure-exclusive eval this occupancy is closure-formed by
    construction)."""
    occ: List[int] = []
    n = len(beta_history)
    i = 0
    while i < n:
        if beta_history[i]:
            j = i
            while j < n and beta_history[j]:
                j += 1
            if (j - i) >= min_run:
                occ.extend(range(i, j))
            i = j
        else:
            i += 1
    return occ


def _closure_latch_lags(
    beta_history: List[bool], fire_ticks: List[int]
) -> List[Dict[str, Any]]:
    """The M3 DV. For each genuine closure fire, the signed + absolute tick lag to the nearest
    SUSTAINED latch-armed occupancy tick in the SAME episode. has_occupancy=False (censored) if
    the episode formed no sustained occupancy."""
    occ = _sustained_occupancy_ticks(beta_history, SUSTAIN_MIN_TICKS)
    events: List[Dict[str, Any]] = []
    for f in fire_ticks:
        if not occ:
            events.append({
                "fire_tick": int(f), "has_occupancy": False,
                "signed_lag": None, "abs_lag": None, "within_horizon": False,
            })
            continue
        signed = min((o - f for o in occ), key=lambda d: abs(d))
        abs_lag = abs(int(signed))
        events.append({
            "fire_tick": int(f), "has_occupancy": True,
            "signed_lag": int(signed), "abs_lag": abs_lag,
            "within_horizon": bool(abs_lag <= LAG_HORIZON),
        })
    return events


def _max_consecutive_true(seq: List[bool]) -> int:
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


def _lag_stats(abs_lags: List[int]) -> Dict[str, Optional[float]]:
    if not abs_lags:
        return {
            "n": 0, "min": None, "p25": None, "median": None,
            "p75": None, "mean": None, "max": None,
        }
    arr = np.asarray(abs_lags, dtype=float)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def _eval_arm_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Eval instrumented for the M3 lag DV. Reads the closure-armed counter, the longest
    consecutive beta run, F-driven natural commits (expected ~0), SD-034 closure fire count, the
    refractory-independent commit-intent counter (MECH-445 context), the around-closure drop
    windows (CONTEXT), and -- the new load-bearing read -- the per-fire lag to the nearest
    sustained latch-armed occupancy. Calls agent.update_residue(hypothesis_tag=False) each tick
    (mirror 715)."""
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
    lag_events: List[Dict[str, Any]] = []
    max_consecutive_beta_run = 0
    ncl_hold_reassert_total = 0
    ncl_hold_closure_armed_total = 0
    n_f_commits = 0
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
                n_f_commits += 1
                committed_class_counts[action_idx] += 1
            if cur_beta:
                total_beta_elevated += 1
            if prev_beta and not cur_beta:
                beta_release_events += 1
            prev_beta = cur_beta

            gs = getattr(agent, "goal_state", None)
            goal_active = bool(gs is not None and gs.is_active())
            rs_norm = _rule_state_norm(agent)
            if rs_norm > rule_state_norm_peak:
                rule_state_norm_peak = rs_norm
            if goal_active and rs_norm >= rule_floor:
                n_rule_directed_commit_ticks += 1

            _, _harm, done, info, obs_dict = env.step(action_idx)

            agent.update_residue(harm_signal=float(_harm), hypothesis_tag=False)

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
        lag_events.extend(_closure_latch_lags(beta_history, fire_ticks))
        _ep_run = _max_consecutive_true(beta_history)
        if _ep_run > max_consecutive_beta_run:
            max_consecutive_beta_run = _ep_run
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
    # --- Context drop DV aggregates (mirror 715) ---
    n_window_events = len(around_events)
    mean_pre_occ = (
        sum(e["pre_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    mean_post_occ = (
        sum(e["post_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    # --- M3 lag DV aggregates ---
    n_fires_total = len(lag_events)
    fires_with_occ = [e for e in lag_events if e["has_occupancy"]]
    n_fires_with_occ = len(fires_with_occ)
    abs_lags = [int(e["abs_lag"]) for e in fires_with_occ]
    signed_lags = [int(e["signed_lag"]) for e in fires_with_occ]
    n_within_horizon = sum(1 for e in fires_with_occ if e["within_horizon"])
    lag_abs_stats = _lag_stats(abs_lags)
    coreg_within_horizon_frac = (
        n_within_horizon / n_fires_total if n_fires_total else 0.0
    )
    fires_with_occ_frac = (
        n_fires_with_occ / n_fires_total if n_fires_total else 0.0
    )
    mean_signed_lag = float(np.mean(signed_lags)) if signed_lags else None

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
        # context drop DV (reported, NOT load-bearing):
        "n_window_events": n_window_events,
        "mean_pre_closure_occ": mean_pre_occ,
        "mean_post_closure_occ": mean_post_occ,
        # M3 lag DV (load-bearing measurement):
        "n_fires_total": n_fires_total,
        "n_fires_with_occupancy": n_fires_with_occ,
        "fires_with_occupancy_frac": fires_with_occ_frac,
        "coreg_within_horizon_frac": coreg_within_horizon_frac,
        "lag_abs_stats": lag_abs_stats,
        "lag_abs_median": lag_abs_stats["median"],
        "mean_signed_lag": mean_signed_lag,
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
        "mean_post_closure_occ": 0.0, "n_fires_total": 0, "n_fires_with_occupancy": 0,
        "fires_with_occupancy_frac": 0.0, "coreg_within_horizon_frac": 0.0,
        "lag_abs_stats": _lag_stats([]), "lag_abs_median": None, "mean_signed_lag": None,
        "committed_class_entropy_n_classes": 0,
    }


def _seed_measurable(guard_pass: bool, arm_on: Dict[str, Any]) -> bool:
    """A seed contributes to the lag distribution iff it is curriculum-competent (guard) AND the
    ON arm produced >= 1 closure fire that fell in an episode with a sustained latch-armed
    occupancy (so a lag exists to measure)."""
    return bool(guard_pass and int(arm_on.get("n_fires_with_occupancy", 0)) >= 1)


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
        "measurable": False,
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

    on_armed_and_sustained = bool(
        int(arm_on.get("ncl_hold_closure_armed_total", 0)) > 0
        and int(arm_on.get("max_consecutive_beta_run", 0)) >= SUSTAIN_MIN_TICKS
    )
    off_did_not_arm = bool(int(arm_off.get("ncl_hold_closure_armed_total", 0)) == 0)
    on_zero_f_commits = bool(int(arm_on.get("n_f_commits", 0)) == 0)
    measurable = _seed_measurable(guard_pass, arm_on)

    _med = arm_on.get("lag_abs_median")
    _med_str = f"{_med:.1f}" if _med is not None else "NA"
    print(f"  [eval] arm_eval seed={seed} ep {done}/{total_eps}"
          f" | OFF armed={arm_off['ncl_hold_closure_armed_total']}"
          f" fires_occ={arm_off['n_fires_with_occupancy']}"
          f" | ON armed={arm_on['ncl_hold_closure_armed_total']}"
          f" run={arm_on['max_consecutive_beta_run']} f_commits={arm_on['n_f_commits']}"
          f" intent={arm_on['sd034_n_closure_commit_intent']}"
          f" fires={arm_on['n_fires_total']} fires_occ={arm_on['n_fires_with_occupancy']}"
          f" lag_med={_med_str} coreg_W={arm_on['coreg_within_horizon_frac']:.3f}"
          f" | armed_sustained={on_armed_and_sustained} off_no_arm={off_did_not_arm}"
          f" measurable={measurable}", flush=True)
    # Per-seed verdict for the runner progress bar (PASS = this seed contributed a lag datum).
    print(f"verdict: {'PASS' if measurable else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} measurable={measurable}"
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
        "measurable": measurable,
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
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # Measurable seeds = curriculum-competent AND produced >= 1 fire-with-occupancy on ON arm.
    measurable_seeds = [r for r in per_seed if bool(r.get("measurable", False))]
    n_measurable = len(measurable_seeds)
    readiness_met = bool(n_measurable >= MIN_MEASURABLE_SEEDS)

    # --- Pool the M3 lag DV over the measurable seeds' ON arms (the graded output) ---
    pooled_abs_lags: List[int] = []
    pooled_signed_lags: List[int] = []
    n_fires_total = 0
    n_fires_with_occ = 0
    n_within_horizon = 0
    for r in measurable_seeds:
        arm_on = r["arms"][ARM_ON]
        st = arm_on.get("lag_abs_stats", {})
        # Re-derive pooled events from per-arm rollup counts + medians is lossy; instead pool the
        # summary counts and recompute the distribution from the per-seed medians we DO have.
        n_fires_total += int(arm_on.get("n_fires_total", 0))
        n_fires_with_occ += int(arm_on.get("n_fires_with_occupancy", 0))
        n_within_horizon += int(round(
            float(arm_on.get("coreg_within_horizon_frac", 0.0))
            * float(arm_on.get("n_fires_total", 0))
        ))
        # per-seed median contributes one representative datum per measurable seed
        _m = arm_on.get("lag_abs_median")
        if _m is not None:
            pooled_abs_lags.append(int(round(_m)))
        _ms = arm_on.get("mean_signed_lag")
        if _ms is not None:
            pooled_signed_lags.append(int(round(_ms)))

    seed_median_lag_stats = _lag_stats(pooled_abs_lags)
    pooled_coreg_within_horizon_frac = (
        n_within_horizon / n_fires_total if n_fires_total else 0.0
    )
    pooled_fires_with_occ_frac = (
        n_fires_with_occ / n_fires_total if n_fires_total else 0.0
    )
    distribution_non_degenerate = bool(n_fires_with_occ >= MIN_LAG_EVENTS)

    # --- Context drop DV (reported): how starved the falsifier window was on measurable seeds ---
    ctx_window_events = sum(
        int(r["arms"][ARM_ON].get("n_window_events", 0)) for r in measurable_seeds
    )
    ctx_mean_pre_occ = (
        float(np.mean([
            float(r["arms"][ARM_ON].get("mean_pre_closure_occ", 0.0)) for r in measurable_seeds
        ])) if measurable_seeds else 0.0
    )

    # --- Routing (diagnostic: only substrate_not_ready_requeue self-routes; never a verdict) ---
    # PASS requires BOTH readiness (a measurable seed) AND a non-degenerate distribution (enough
    # fire-with-occupancy events to characterise the lag) -- so an under-powered measurement
    # re-queues at higher power rather than reading as a false PASS (avoids the vacuous_pass flag).
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route_reason = (
            "no_measurable_seed_guard_or_no_fire_with_sustained_occupancy"
            if contact_non_vacuity_met else "contact_guard_unmet"
        )
    elif not distribution_non_degenerate:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        route_reason = (
            f"under_powered_distribution_n_fires_with_occupancy={n_fires_with_occ}"
            f"_below_min_lag_events={MIN_LAG_EVENTS}_requeue_more_seeds"
        )
    else:
        outcome = "PASS"
        label = "closure_latch_lag_distribution_measured"
        route_reason = (
            f"lag_distribution_obtained_on_{n_measurable}_measurable_seed(s)"
            f"_median_of_seed_medians={seed_median_lag_stats['median']}"
            f"_coreg_within_{LAG_HORIZON}={pooled_coreg_within_horizon_frac:.3f}"
        )

    # Diagnostic: no claim direction. non_degenerate reflects the distribution quality (and now
    # equals outcome==PASS -- an under-powered distribution self-routes to substrate_not_ready).
    evidence_direction = "unknown"
    non_degenerate = bool(readiness_met and distribution_non_degenerate)
    degeneracy_reason = (
        "" if non_degenerate
        else (f"substrate_not_ready: {route_reason}" if not readiness_met
              else f"under_powered_distribution: n_fires_with_occupancy={n_fires_with_occ} < {MIN_LAG_EVENTS}")
    )

    print(f"[{EXPERIMENT_TYPE}] guard={contact_non_vacuity_met}({sum(guard_flags)}/{n})"
          f" measurable_seeds={n_measurable}/{n} readiness={readiness_met}"
          f" | fires={n_fires_total} fires_occ={n_fires_with_occ}"
          f" seed_median_lag={seed_median_lag_stats['median']}"
          f" coreg_within_{LAG_HORIZON}={pooled_coreg_within_horizon_frac:.3f}"
          f" | outcome={outcome} label={label} non_degen={non_degenerate}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_measurable_seeds": n_measurable,
        "readiness_met": readiness_met,
        "n_fires_total": n_fires_total,
        "n_fires_with_occupancy": n_fires_with_occ,
        "pooled_fires_with_occupancy_frac": pooled_fires_with_occ_frac,
        "pooled_coreg_within_horizon_frac": pooled_coreg_within_horizon_frac,
        "lag_horizon_ticks": LAG_HORIZON,
        "seed_median_lag_stats": seed_median_lag_stats,
        "distribution_non_degenerate": distribution_non_degenerate,
        "context_window_events_on_measurable": ctx_window_events,
        "context_mean_pre_closure_occ_on_measurable": ctx_mean_pre_occ,
        "route_reason": route_reason,
        "per_seed_guard_pass": guard_flags,
        "per_seed_measurable": [bool(r.get("measurable", False)) for r in per_seed],
        "per_seed_on_lag_median": [
            r["arms"][ARM_ON].get("lag_abs_median") for r in per_seed
        ],
        "per_seed_on_coreg_frac": [
            r["arms"][ARM_ON].get("coreg_within_horizon_frac") for r in per_seed
        ],
        "per_seed_on_fires_total": [
            r["arms"][ARM_ON].get("n_fires_total") for r in per_seed
        ],
        "per_seed_on_fires_with_occ": [
            r["arms"][ARM_ON].get("n_fires_with_occupancy") for r in per_seed
        ],
    }

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "acceptance": acceptance,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "foraging_contact_guard",
                    "description": "per-seed P2 contact_rate > 0 AND z_goal_norm_at_contact_peak "
                                   "> 0.4 on >= 2/3 seeds (curriculum-competent).",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout",
                },
                {
                    "name": "lag_distribution_measurable",
                    "description": "READINESS (self-route). >= MIN_MEASURABLE_SEEDS seeds are "
                                   "curriculum-competent AND produced >= 1 closure fire in an "
                                   "episode with a sustained latch-armed occupancy, so a lag is "
                                   "DEFINED. This is the SAME statistic the load-bearing lag DV "
                                   "reads (fire-with-occupancy count on the ON arm). Below floor "
                                   "-> substrate_not_ready_requeue (NEVER a substrate_ceiling / "
                                   "does_not_support / *_nondiscriminative verdict); nothing to "
                                   "measure, not a negative result.",
                    "measured": float(n_measurable),
                    "threshold": float(MIN_MEASURABLE_SEEDS),
                    "met": readiness_met,
                    "control": "ARM_ENTRY_ON n_fires_with_occupancy on guard-competent seeds",
                },
            ],
            "criteria_non_degenerate": {
                # The 'measurement' is non-degenerate iff enough fire-with-occupancy events exist
                # to form a distribution (not a single point / no events).
                "lag_distribution_has_events": distribution_non_degenerate,
            },
            "criteria": [
                {
                    # No falsification criterion -- this is a MEASUREMENT. The load-bearing output
                    # IS the lag distribution; 'passed' means it was obtained non-degenerately.
                    "name": "lag_distribution_obtained_non_degenerate",
                    "load_bearing": True,
                    "passed": non_degenerate,
                },
            ],
            "measurement_note": "M3 (claim_synthesis_MECH-445-446_2026-07-06). The MECH-446 "
                                "de-commit DV is RE-ANCHORED from the vacuous within-arm occupancy "
                                "DROP to the DISTRIBUTION of the tick LAG between each genuine "
                                "SD-034 closure fire (closure_operator._n_closures increment, the "
                                "de-commit trigger) and the nearest sustained latch-armed beta "
                                "occupancy (a run >= SUSTAIN_MIN_TICKS). Graded output: "
                                "seed_median_lag_stats (abs-lag distribution over per-seed "
                                "medians) + pooled_coreg_within_horizon_frac (fraction of fires "
                                "with an occupancy within LAG_HORIZON ticks -- the fraction the "
                                "715 falsifier window could ever have scored). A large median lag "
                                "/ low coreg fraction quantifies exactly how much co-registration "
                                "the f_dominance de-commit-release amend must buy. PROMOTES "
                                "NOTHING; MECH-445/446 unchanged.",
            "attribution_guard": "closure_exclusive_decommit_eval=True -> beta elevation is "
                                 "CLOSURE-EXCLUSIVE, so every measured occupancy is closure-formed "
                                 "by construction (not a generic beta co-occurrence). ARM_ENTRY_OFF "
                                 "(no entry primitive) forms no occupancy -> its lag is censored: "
                                 "the negative control confirming the occupancy is entry-driven.",
            "levers_off_note": "rung-6 NaturalCommitUrgencyRelease + ARC-108 JOB-2 driver pair OFF "
                               "in every arm. The measured object is the INTRINSIC SD-034 closure "
                               "de-commit fire vs the closure-entry occupancy.",
            "context_drop_dv": "n_window_events / mean_pre_closure_occ are REPORTED (not "
                               "load-bearing) to show the 715 falsifier's drop window remained "
                               "starved on the same measurable seeds -- the lag DV is what "
                               "extracts signal the drop DV could not.",
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
        "non_degenerate": result["non_degenerate"],
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
                    "surface); agent.update_residue called with hypothesis_tag=False.",
        },
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1 -> P2; 715/460o config) + commitment control-plane + subgoal_mode "
                     "waypoint tolerance-band completion + commitment-closure-control-plane Legs "
                     "A/B/C + beta-engagement coupling + natural-commit LATCH-HOLD + the "
                     "CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode (closure_exclusive_decommit_eval) ARMED "
                     "in BOTH arms. use_closure_commit_entry (VALIDATED 460o/p) toggled per arm.",
        "condition": CONDITION_LABEL,
        "method_note": "M3 lag-distribution diagnostic (claim_synthesis_MECH-445-446_2026-07-06). "
                       "Re-anchors the MECH-446 DV from the vacuous within-arm occupancy-DROP to "
                       "the DISTRIBUTION of the tick LAG between each genuine SD-034 closure fire "
                       "and the nearest sustained latch-armed occupancy, on the F-independent "
                       "use_closure_commit_entry substrate. Turns the recurring 'vacuous 0/3' into "
                       "graded data (seed_median_lag_stats + coreg_within_horizon_frac) telling "
                       "/implement-substrate how much co-registration the de-commit-release amend "
                       "must buy. Readiness (self-route, never a verdict): >= 1 measurable seed "
                       "(guard-competent + a fire-with-sustained-occupancy). DIAGNOSTIC; claim_ids "
                       "empty; brake-EXEMPT (new DV measurement, not a same-selector de-commit-"
                       "magnitude re-derive -- that face's brake is FIRED). PROMOTES NOTHING.",
        "arm_note": "ARMS = " + ", ".join(a["key"] for a in ARMS) + ". closure-exclusive eval + "
                    "beta-engagement coupling + natural-commit latch-hold ON in both arms; the only "
                    "variable is use_closure_commit_entry. ARM_ENTRY_OFF = the no-arm attribution "
                    "control (occupancy censored); ARM_ENTRY_ON = the arm where the lag DV is "
                    "measured.",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "sustain_min_ticks": SUSTAIN_MIN_TICKS,
            "lag_horizon_ticks": LAG_HORIZON,
            "min_measurable_seeds": MIN_MEASURABLE_SEEDS,
            "min_lag_events": MIN_LAG_EVENTS,
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
            "config_basis": "closure-exclusive de-commit eval (ree-v3 main e52158d, 2026-06-22) + "
                            "F-independent closure-plane commit-ENTRY primitive "
                            "(use_closure_commit_entry, ree-v3 main 84c1e7c; VALIDATED "
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
