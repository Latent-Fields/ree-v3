"""
V3-EXQ-460e (supersedes V3-EXQ-460d): SD-034 closure-control-plane de-commit
authority, re-run on the foraging-competent scaffolded_sd054_onboarding 603n agent
with the rule_bias_head NOW TRAINED (commitment_closure:GAP-4 Leg C).

WHY THIS SUPERSEDES 460d: the confirmed failure_autopsy_SD-034-closure-control-plane-d_2026-06-13
found 460d's residual was the LITERAL "Leg C not built". Both v3_exq_460d_*.py and
v3_exq_468d_*.py set lateral_pfc_train_rule_bias_head=True (un-zeroing the GAP-D
head's last Linear) but NEVER added the head to any optimizer -- grep for
optim|Adam|.backward(|bias_head_parameters returned ZERO matches. So the head stayed
at random init: the rule_state handed to the ClosureOperator carried no task-shaped
magnitude, the closure-coupled de-commit had no net authority over the MECH-090 latch
(460d C2_beta_release/C4 FAIL: ON latch occupancy >= OFF on seeds 43/44), and the
automatic rule-stability detector stayed inert. NOT a falsification of SD-034/MECH-261.

WHAT CHANGED (commitment_closure:GAP-4 Leg C, landed 2026-06-16): the scaffold now has
a scaffold_train_rule_bias_head leg that TRAINS agent.lateral_pfc.bias_head_parameters()
during P1 (goal-unfrozen, ecological-contact, commitment forms) via the V3-EXQ-598b
outcome-coupled E3-gradient REINFORCE pattern (mirrors scaffold_train_harm_pathway;
no-op default, bit-identical OFF; ree_core untouched). 460e enables it
(scaffold_train_rule_bias_head=True) + reads P1OnboardingResult.rule_bias_diag to GATE
non-vacuity (the head must actually have trained).

WHAT THIS MEASURES (the de-commit behavioural authority 460d could not express): with a
TRAINED magnitude-bearing rule_state, does the closure-coupled de-commit LOWER the
MECH-090 beta-latch occupancy vs the no-closure control? ARM_CLOSURE_ON commits then
closure releases -> lower latch occupancy; ARM_CLOSURE_OFF (same trained weights, closure
off) commits then perseverates -> higher occupancy.

ARMS (one curriculum build per seed, two frozen-policy evals):
  ARM_CLOSURE_ON   -- full closure + env-completion hook + de-commit hold + bistable,
                      built on the TRAINED rule_bias_head.
  ARM_CLOSURE_OFF  -- clone of the SAME trained weights with closure OFF (hook is a
                      no-op when closure_operator is None). Beta perseverates.

READINESS / NON-VACUITY (all four must clear before the de-commit DV is scored; any unmet
self-routes substrate_not_ready_requeue -- NEVER a false weakens):
  (1) 603n foraging contact guard: per-seed P2 contact_rate > 0 AND
      z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.
  (2) beta-engagement BOTH arms (the 468d commit-without-beta guard, extended so the
      occupancy-drop comparison is valid): ON total_beta_elevated > 0 AND OFF
      total_beta_elevated > 0 AND ON n_sequence_completions > 0 on >= 2/3 guard seeds.
  (3) closure-trigger available: ON-arm n_closures > 0 on >= 2/3 guard seeds.
  (4) rule_bias_head trained (the DIRECT anti-460d-bug gate): P1 rule_bias_pathway_enabled
      AND mean per-candidate |bias| (rule_bias_diag) > RULE_BIAS_MEAN_FLOOR on >= 2/3
      seeds. If the head did not train (mean |bias| ~ 0, the 460d signature), the
      de-commit cannot have authority -> substrate_not_ready_requeue, never a weakens.

PRE-REGISTERED ACCEPTANCE (constants; per-seed PASS = C1 AND C2 AND C3; overall PASS =
majority 2/3 guard seeds; scored only once all four readiness gates clear):
  C1  ARM_CLOSURE_ON  n_closures >= 1                  (closure fires in a live loop)
  C2  NON-CAP-PINNED DE-COMMIT OCCUPANCY DROP (load-bearing): ARM_CLOSURE_ON
      mean_beta_elevated_steps < ARM_CLOSURE_OFF mean_beta_elevated_steps with a
      >= DECOMMIT_MIN_DROP_FRAC relative drop (OFF must have committed). Replaces 460d's
      count-based C2_beta_release + conjunctive cap-pinned C4 with a single continuous,
      non-cap-pinned latch-occupancy statistic.
  C3  ARM_CLOSURE_ON  nogo_installed_total >= 1        (targeted MECH-260 No-Go)

claim_ids: SD-034, MECH-260, MECH-261.
experiment_purpose: evidence (the substrate is built + readiness-gated; the de-commit
  authority is now a real claim hypothesis. The readiness gates self-route
  non_contributory when unmet; a genuine criteria-unmet result is a real weakens).
supersedes: V3-EXQ-460d.

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_460e_sd034_closure_control_plane_behavioural"
QUEUE_ID = "V3-EXQ-460e"
CLAIM_IDS: List[str] = ["SD-034", "MECH-260", "MECH-261"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-460d"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_TRAINED_RULE_BIAS_DECOMMIT_RETEST"

# --- Goal-pipeline / encoder dims (mirror 603n / 460d exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5  # Leg B: post-closure latch refractory window

# --- Curriculum budgets (mirror 603n / 460d exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
CLOSURE_EVAL_EPISODES = 15  # per arm (ON + OFF)
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
C1_MIN_CLOSURES = 1
C3_MIN_NOGO = 1
# C2 non-cap-pinned de-commit occupancy-drop DV: ARM_CLOSURE_ON mean beta-latch
# occupancy must be at least this RELATIVE fraction below ARM_CLOSURE_OFF (the
# closure-coupled de-commit lowers occupancy vs the no-closure control). Continuous
# statistic on the [0, steps] occupancy mean -- NOT a 0.85-capped post/pre ratio.
DECOMMIT_MIN_DROP_FRAC = 0.10
# OFF arm must actually have committed (mean occupancy above this floor) for the drop
# comparison to be meaningful; below it the beta-engagement readiness gate fires.
MIN_OFF_OCC = 0.5
# Leg C readiness: the rule_bias_head must have TRAINED -- mean per-candidate |bias|
# above this floor. The untrained 460d head produced ~0; the Leg-C smoke produced 0.039.
RULE_BIAS_MEAN_FLOOR = 0.005


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
        # commitment_closure:GAP-4 Leg C (2026-06-16): TRAIN the rule_bias_head in P1
        # via the 598b REINFORCE pattern. The 460d gap was this flag absent (head
        # un-zeroed but never optimized). lateral_pfc_train_rule_bias_head=True on the
        # agent config (un-zero) + this flag (train) are BOTH required.
        scaffold_train_rule_bias_head=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """603n-validated foraging substrate (mirror 460d) + the commitment control-plane +
    the 2026-06-12 commitment-closure-control-plane amend (Legs A/B/C). Leg C here is
    the GAP-D un-zero flag (lateral_pfc_train_rule_bias_head=True); the scaffold leg
    (scaffold_train_rule_bias_head, set on the scheduler cfg) supplies the training."""
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
        # SD-034 commitment-closure-control-plane amend (2026-06-12):
        use_closure_env_completion_hook=True,          # Leg A
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D); trained by scaffold leg
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_closure_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint
    tolerance-band completion so the SD-034 closure operator has completions to fire on."""
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


def _clone_closure_off(trained_agent: REEAgent, device: torch.device) -> REEAgent:
    """Clone the SAME trained weights into a closure-OFF agent (closure carries no
    trainable parameters, so the state_dict loads cleanly)."""
    cfg_off = copy.deepcopy(trained_agent.config)
    cfg_off.use_closure_operator = False
    cfg_off.heartbeat.beta_gate_bistable = True
    agent_off = REEAgent(cfg_off).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_off.load_state_dict(state)
    except RuntimeError:
        agent_off.load_state_dict(state, strict=False)
    agent_off.e3._running_variance = float(trained_agent.e3._running_variance)
    agent_off.beta_gate = BetaGate(completion_release_threshold=2.0)
    return agent_off


def _eval_closure_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for SD-034 closure behaviour (identical
    instrumentation to 460d; the difference between 460d and 460e is upstream -- the
    rule_bias_head is now trained, so the closure-coupled de-commit has authority).
    n_hook_fires counts Leg-A env-completion-hook closures; n_closures - n_hook_fires
    is the automatic-detector contribution (whether MECH-261 mode-conditioning was
    load-bearing or the hook bypassed it)."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_dacc = getattr(agent, "dacc", None) is not None
    hook_enabled = bool(getattr(agent.config, "use_closure_env_completion_hook", False))
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    beta_release_events = 0
    nogo_installed_total = 0
    total_committed_steps = 0
    total_beta_elevated = 0
    n_sequence_completions = 0
    n_hook_fires = 0

    with torch.no_grad():
        for _ in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            prev_beta = bool(agent.beta_gate.is_elevated)

            for _ in range(steps_per_episode):
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

                if agent.e3._committed_trajectory is not None:
                    total_committed_steps += 1
                cur_beta = bool(agent.beta_gate.is_elevated)
                if cur_beta:
                    total_beta_elevated += 1
                if prev_beta and not cur_beta:
                    beta_release_events += 1
                prev_beta = cur_beta

                _, _harm, done, info, obs_dict = env.step(action_idx)
                if info.get("transition_type") == "sequence_complete":
                    n_sequence_completions += 1
                    if has_closure and hook_enabled:
                        ev = agent.notify_env_completion(action_class=action_idx)
                        if ev is not None and getattr(ev, "fired", False):
                            n_hook_fires += 1
                            nogo_installed_total += int(getattr(ev, "nogo_pushed", 0))
                if done:
                    break

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre if has_closure else 0
    )
    return {
        "n_closures": n_closures,
        "n_hook_fires": n_hook_fires,
        "n_automatic_fires": max(0, n_closures - n_hook_fires),
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "mean_beta_elevated_steps": total_beta_elevated / max(1, n_eps),
        "n_sequence_completions": n_sequence_completions,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        "env_hook_enabled": hook_enabled,
    }


def _decommit_occupancy_drop(arm_on: Dict[str, Any], arm_off: Dict[str, Any]) -> bool:
    """C2 non-cap-pinned de-commit DV: ON mean beta-latch occupancy is at least a
    DECOMMIT_MIN_DROP_FRAC relative drop below OFF (OFF must have committed). Continuous
    occupancy statistic, NOT a capped post/pre ratio."""
    on_occ = float(arm_on.get("mean_beta_elevated_steps", 0.0))
    off_occ = float(arm_off.get("mean_beta_elevated_steps", 0.0))
    if off_occ <= MIN_OFF_OCC:
        return False
    return bool(on_occ < off_occ and (off_occ - on_occ) >= DECOMMIT_MIN_DROP_FRAC * off_occ)


def _rule_bias_mean(p1) -> float:
    diag = getattr(p1, "rule_bias_diag", None) or {}
    n = int(diag.get("n_bias_samples", 0))
    s = float(diag.get("sum_bias_abs_mean", 0.0))
    return s / n if n > 0 else 0.0


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    empty = {
        "n_closures": 0, "n_hook_fires": 0, "n_automatic_fires": 0, "beta_release_events": 0,
        "nogo_installed_total": 0, "total_committed_steps": 0, "total_beta_elevated": 0,
        "mean_beta_elevated_steps": 0.0, "n_sequence_completions": 0,
        "n_eval_episodes": 0, "closure_present": False, "env_hook_enabled": False,
    }
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "rule_bias_pathway_enabled": False,
        "rule_bias_mean_abs": 0.0,
        "rule_bias_n_train_steps": 0,
        "rule_bias_trained": False,
        "ARM_CLOSURE_ON": dict(empty), "ARM_CLOSURE_OFF": dict(empty),
        "criteria": {"C1": False, "C2": False, "C3": False},
        "beta_engagement_both_arms": False,
        "closure_trigger_available": False,
        "pass": False,
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
    rule_bias_enabled = bool(getattr(p1, "rule_bias_pathway_enabled", False))
    rule_bias_mean = _rule_bias_mean(p1)
    rule_bias_steps = int((getattr(p1, "rule_bias_diag", None) or {}).get("n_train_steps", 0))
    rule_bias_trained = bool(rule_bias_enabled and rule_bias_mean > RULE_BIAS_MEAN_FLOOR)
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
          f" rule_bias_enabled={rule_bias_enabled} rule_bias_mean={rule_bias_mean:.4f}"
          f" rule_bias_steps={rule_bias_steps} rule_bias_trained={rule_bias_trained}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    closure_env = _build_closure_env(scaffold_cfg)
    closure_env.reset()

    print(f"Seed {seed} Condition ARM_CLOSURE_ON", flush=True)
    arm_on = _eval_closure_behaviour(
        agent, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    print(f"Seed {seed} Condition ARM_CLOSURE_OFF", flush=True)
    agent_off = _clone_closure_off(agent, device)
    agent_off.e3._running_variance = float(agent.e3._running_variance)
    arm_off = _eval_closure_behaviour(
        agent_off, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
    )
    done += eval_eps

    c1 = arm_on["n_closures"] >= C1_MIN_CLOSURES
    c2 = _decommit_occupancy_drop(arm_on, arm_off)   # non-cap-pinned de-commit DV
    c3 = arm_on["nogo_installed_total"] >= C3_MIN_NOGO
    # Beta engagement on BOTH arms (the 468d commit-without-beta guard, extended so the
    # occupancy-drop comparison is valid) + ON completed a sequence (closure had an
    # opportunity to fire).
    beta_engagement_both_arms = bool(
        arm_on["total_beta_elevated"] > 0
        and arm_off["total_beta_elevated"] > 0
        and arm_on["n_sequence_completions"] > 0
    )
    closure_trigger_available = bool(arm_on["n_closures"] > 0)
    seed_pass = bool(c1 and c2 and c3)

    print(f"  [train] closure_eval seed={seed} ep {done}/{total_eps}"
          f" c1={c1} c2_decommit={c2} c3={c3}"
          f" on_occ={arm_on['mean_beta_elevated_steps']:.2f} off_occ={arm_off['mean_beta_elevated_steps']:.2f}"
          f" n_closures={arm_on['n_closures']} auto={arm_on['n_automatic_fires']} hook={arm_on['n_hook_fires']}"
          f" nogo={arm_on['nogo_installed_total']}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} beta_engaged_both={beta_engagement_both_arms}"
          f" closure_trigger={closure_trigger_available} rule_bias_trained={rule_bias_trained}"
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
        "rule_bias_pathway_enabled": rule_bias_enabled,
        "rule_bias_mean_abs": float(rule_bias_mean),
        "rule_bias_n_train_steps": rule_bias_steps,
        "rule_bias_trained": rule_bias_trained,
        "ARM_CLOSURE_ON": arm_on,
        "ARM_CLOSURE_OFF": arm_off,
        "criteria": {"C1": c1, "C2": c2, "C3": c3},
        "beta_engagement_both_arms": beta_engagement_both_arms,
        "closure_trigger_available": closure_trigger_available,
        "pass": seed_pass,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + 2 * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + 2 * CLOSURE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # Readiness gate (2): beta engagement on both arms among guard-passing seeds.
    be_flags = [bool(r.get("beta_engagement_both_arms", False)) for r in guard_passing]
    be_frac = _frac(be_flags)
    beta_engagement_met = bool(be_frac >= MIN_FRACTION)

    # Readiness gate (3): closure-trigger available among guard-passing seeds.
    ct_flags = [bool(r.get("closure_trigger_available", False)) for r in guard_passing]
    ct_frac = _frac(ct_flags)
    closure_trigger_available_met = bool(ct_frac >= MIN_FRACTION)

    # Readiness gate (4): rule_bias_head actually trained (the anti-460d-bug gate).
    rb_flags = [bool(r.get("rule_bias_trained", False)) for r in guard_passing]
    rb_frac = _frac(rb_flags)
    rule_bias_trained_met = bool(rb_frac >= MIN_FRACTION)

    seed_pass_flags = [bool(r.get("pass", False)) for r in guard_passing]
    n_pass = sum(1 for f in seed_pass_flags if f)
    pass_frac = _frac(seed_pass_flags)
    overall_criteria_pass = bool(pass_frac >= MIN_FRACTION)

    def _all_guard(crit_key: str) -> bool:
        return bool(guard_passing) and all(
            r.get("criteria", {}).get(crit_key) for r in guard_passing
        )

    c2_all = _all_guard("C2")
    c3_all = _all_guard("C3")

    if not contact_non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not rule_bias_trained_met:
        # The Leg-C scaffold leg did not produce a trained magnitude-bearing head
        # (mean |bias| ~ 0 -- the 460d signature). The de-commit cannot have authority
        # without it -> substrate not ready, NEVER a false weakens.
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "rule_bias_head_untrained"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not beta_engagement_met:
        # The agent did not commit-with-beta on both arms (the 468d signature) -> the
        # occupancy-drop comparison is not valid = substrate not ready.
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "beta_engagement_not_met_both_arms"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not closure_trigger_available_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        route_reason = "closure_trigger_unavailable"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        # All four readiness gates clear -> the de-commit DV is interpretable.
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("sd034_decommit_authority_confirmed"
                           if overall_criteria_pass else "residual_decommit_authority_open")
        route_reason = ("c1_c2_c3_majority_met" if overall_criteria_pass
                        else "decommit_dv_unmet_genuine_weakens")
        direction_map = {
            "SD-034": "supports" if overall_criteria_pass else "weakens",
            "MECH-260": "supports" if c3_all else "weakens",
            "MECH-261": "supports" if overall_criteria_pass else "weakens",
        }
        overall_direction = "supports" if overall_criteria_pass else "weakens"

    print(f"[{EXPERIMENT_TYPE}] contact_non_vacuity={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) rule_bias_trained={rule_bias_trained_met}"
          f" (frac={rb_frac:.3f}) beta_engaged_both={beta_engagement_met} (frac={be_frac:.3f})"
          f" closure_trigger={closure_trigger_available_met} (frac={ct_frac:.3f})"
          f" criteria_pass={overall_criteria_pass} ({n_pass}/{len(guard_passing)})"
          f" -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    readiness_all_met = bool(
        contact_non_vacuity_met and rule_bias_trained_met
        and beta_engagement_met and closure_trigger_available_met
    )

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "rule_bias_trained_met": rule_bias_trained_met,
        "rule_bias_trained_fraction": rb_frac,
        "beta_engagement_met": beta_engagement_met,
        "beta_engagement_fraction": be_frac,
        "closure_trigger_available_met": closure_trigger_available_met,
        "closure_trigger_fraction": ct_frac,
        "C2_decommit_all_guard_passing": c2_all,
        "C3_all_guard_passing": c3_all,
        "criteria_pass_fraction": pass_frac,
        "n_seeds_pass": n_pass,
        "overall_pass": bool(readiness_all_met and overall_criteria_pass),
        "per_seed_guard_pass": guard_flags,
        "per_seed_criteria_pass": [bool(r.get("pass", False)) for r in per_seed],
        "route_reason": route_reason,
    }

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": [
                {
                    "name": "foraging_contact_guard",
                    "description": "603n G2+G3: per-seed P2 contact_rate > 0 AND "
                                   "z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.",
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                },
                {
                    "name": "rule_bias_head_trained",
                    "description": "Leg C readiness (the DIRECT anti-460d-bug gate): P1 "
                                   "rule_bias_pathway_enabled AND mean per-candidate |bias| "
                                   "(rule_bias_diag) > floor on >= 2/3 seeds. The untrained "
                                   "460d head produced ~0; below floor -> substrate_not_ready_"
                                   "requeue (the head did not train), NEVER a false weakens.",
                    "control": "P1OnboardingResult.rule_bias_diag mean |bias| with the "
                               "scaffold_train_rule_bias_head leg enabled.",
                    "measured": rb_frac,
                    "threshold": MIN_FRACTION,
                    "met": rule_bias_trained_met,
                },
                {
                    "name": "beta_engagement_both_arms",
                    "description": "the 468d commit-without-beta guard, extended so the "
                                   "occupancy-drop comparison is valid: ON total_beta_elevated "
                                   "> 0 AND OFF total_beta_elevated > 0 AND ON "
                                   "n_sequence_completions > 0 on >= 2/3 guard seeds. Below "
                                   "floor -> substrate_not_ready_requeue.",
                    "control": "fraction of guard-passing seeds with both arms committing + "
                               "ON completing a sequence.",
                    "measured": be_frac,
                    "threshold": MIN_FRACTION,
                    "met": beta_engagement_met,
                },
                {
                    "name": "closure_trigger_available_count",
                    "description": "ON-arm n_closures > 0 reachable on >= 2/3 guard seeds "
                                   "(closure HAD an opportunity to act). Below floor -> "
                                   "substrate_not_ready_requeue.",
                    "control": "ARM_CLOSURE_ON n_closures > 0 (Leg-A hook + trained head).",
                    "measured": ct_frac,
                    "threshold": MIN_FRACTION,
                    "met": closure_trigger_available_met,
                },
            ],
            "criteria": [
                {"name": "C1_n_closures", "load_bearing": False, "passed": _all_guard("C1")},
                {"name": "C2_decommit_occupancy_drop", "load_bearing": True, "passed": c2_all},
                {"name": "C3_nogo_installed", "load_bearing": False, "passed": c3_all},
            ],
            "criteria_non_degenerate": {
                # The de-commit DV is non-degenerate iff all four readiness gates cleared
                # (both arms committed with non-zero occupancy, trained head, closure fired)
                # -- otherwise the occupancy comparison is structurally uninterpretable and
                # the run self-routes substrate_not_ready_requeue above.
                "C1": readiness_all_met,
                "C2": readiness_all_met,
                "C3": readiness_all_met,
            },
            "decommit_dv": {
                "definition": "C2 = ARM_CLOSURE_ON mean_beta_elevated_steps < "
                              "ARM_CLOSURE_OFF mean_beta_elevated_steps with a >= "
                              "DECOMMIT_MIN_DROP_FRAC relative drop (OFF must have "
                              "committed above MIN_OFF_OCC). Non-cap-pinned continuous "
                              "occupancy statistic; replaces 460d's count-based "
                              "C2_beta_release + conjunctive cap-pinned C4.",
                "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
                "min_off_occ": MIN_OFF_OCC,
            },
            "amend_legs_under_test": {
                "leg_a_env_completion_hook": "REEAgent.notify_env_completion -> emit_closure.",
                "leg_b_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
                "leg_c_trained_rule_bias_head": "scaffold_train_rule_bias_head (598b REINFORCE "
                                                "in P1) -- the gap 460d left unbuilt.",
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
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> "
                     "P0 -> Stage-H -> P1 -> P2; 603n config) + commitment control-plane "
                     "(bistable BetaGate + SD-034 ClosureOperator + SD-033a LateralPFC + "
                     "SD-032 dACC/salience) + subgoal_mode waypoint tolerance-band completion "
                     "+ the commitment-closure-control-plane amend Legs A/B (env-completion "
                     "hook + de-commit hold) + Leg C (scaffold_train_rule_bias_head: the "
                     "rule_bias_head TRAINED in P1 via 598b REINFORCE, landed 2026-06-16).",
        "condition": CONDITION_LABEL,
        "method_note": "Supersedes 460d. 460d set lateral_pfc_train_rule_bias_head=True but "
                       "never added the head to any optimizer (failure_autopsy_SD-034-closure-"
                       "control-plane-d_2026-06-13), so the rule_state carried no magnitude and "
                       "the de-commit had no MECH-090 latch authority (C2/C4 FAIL). 460e enables "
                       "the Leg-C scaffold_train_rule_bias_head leg (598b REINFORCE in P1), gates "
                       "non-vacuity on the trained-head magnitude (rule_bias_diag), and reads "
                       "de-commit on a NON-CAP-PINNED ON<OFF beta-latch-occupancy drop. Four "
                       "readiness gates (contact / trained-head / beta-engagement-both-arms / "
                       "closure-trigger) self-route substrate_not_ready_requeue when unmet -- "
                       "never a false weakens.",
        "arm_note": "ARM_CLOSURE_ON (full closure + env hook + de-commit hold + bistable, on the "
                    "TRAINED rule_bias_head) vs ARM_CLOSURE_OFF (same trained weights, closure off).",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "c1_min_closures": C1_MIN_CLOSURES,
            "c3_min_nogo": C3_MIN_NOGO,
            "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
            "min_off_occ": MIN_OFF_OCC,
            "rule_bias_mean_floor": RULE_BIAS_MEAN_FLOOR,
            "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "closure_eval_episodes_per_arm": CLOSURE_EVAL_EPISODES,
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "scaffold_train_rule_bias_head": True,
            "config_basis": "V3-EXQ-603n (substrate-readiness run that flipped "
                            "scaffolded_sd054_onboarding ready=true) + Leg C (2026-06-16)",
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
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
        )
