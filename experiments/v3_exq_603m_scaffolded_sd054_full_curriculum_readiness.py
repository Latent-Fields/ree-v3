"""
V3-EXQ-603m -- scaffolded_sd054_onboarding FULL three-leg readiness run with
scaffold_train_harm_pathway ON (the residual-leg gate after V3-EXQ-603k).

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
V3-EXQ-603k (2026-06-09 PASS) cleared ONLY the SURVIVAL leg of the
substrate_queue scaffolded_sd054_onboarding readiness gate: it was a narrow
Stage-H harm-pathway-survival probe (ARM_HARM_ON_NAV G_H 2/3, P1 survival 3/3)
that never exercised the P2 benefit-contact / P2 z_goal legs, and Stage-0
z_goal>0.4 held on only 1/3 seeds. substrate_queue ready STAYS false. The
remaining gate (substrate_queue ready_blocked_by) is the FULL three-leg run:
  Stage-0 z_goal>0.4 AND P1 survival AND P2 benefit-contact AND P2 z_goal>0.4,
  each >= 2/3 seeds, with scaffold_train_harm_pathway ON.
603m runs that full curriculum (Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1 -> P2)
SINGLE-ARM, all-levers-ON, with the 2026-06-09 harm-pathway training ON (so the
agent survives Stage-H/P1 and REACHES P2 alive) AND the 2026-06-05 foraging-
competence amend ON (auto-reconcile gating, graded P1 reef-spawn weaning,
consumption-event-gated G3 readout), and measures all four gate legs in ONE run.

WHY this is now interpretable (not vacuous): V3-EXQ-603f proved on seed 44 that
the goal-formation + ecological-seeding + P2 foraging chain is SOUND when the
agent survives (P2 contact_rate 0.393 / 85 events AND z_goal_norm_at_contact_peak
0.450 > 0.4); the sole blocker was the P1 survival/hazard-avoidance leg, which
603k's harm-pathway training fixed. So with harm-pathway ON the agent survives
P1, reaches P2 alive, and the P2 benefit-contact / P2 z_goal legs become
measurable across seeds.

PRE-REGISTERED FULL GATE (each >= 2/3 seeds; do NOT retune):
  G0 stage0_positive_control : Stage-0 forced-feed z_goal_norm_peak > 0.4 (the
                               goal stream lights when fed -- the goal-FORMATION
                               positive control, decoupled from foraging).
  G1 p1_survival             : P1 survival gate passed (median episode length over
                               the last 10 P1 episodes >= 75).
  G2 p2_contact              : P2 foraging contact_rate > 0 (benefit-contact leg).
  G3 p2_zgoal_consumption    : P2 consumption-event-gated z_goal_norm_at_contact_peak
                               > 0.4 (the ecological wanting leg; the 634c
                               consumption-gated readout, NOT the carried
                               forced-feed nursery trace).
EXPERIMENT PASS = G0 AND G1 AND G2 AND G3 (each >= 2/3 seeds). PASS => the full
scaffolded substrate is ready; a follow-on /governance + /queue-experiment action
(NOT automatic, NOT in this script) flips substrate_queue
scaffolded_sd054_onboarding ready=true and unblocks the SD-049 GAP-2 / MECH-229 /
MECH-230 / MECH-260 behavioural retests.

NON-VACUITY PRECONDITIONS (else self-route substrate_not_ready_requeue, NEVER a
foraging verdict). The P2-contact / P2-z_goal legs are interpretable as foraging
competence ONLY if the agent actually reached P2 ALIVE on a TRAINED harm
landscape -- otherwise a P2-contact-zero read is "died before contact" not
"foraging incompetence":
  (a) harm_pathway_discriminative : the harm pathway trained (n_train_steps >= 1)
      AND harm_eval(z_world) range lifts above the flat ~0.002 603i baseline
      (>= HARM_EVAL_RANGE_FLOOR). If below -> harm landscape is noise (603i
      regression) -> substrate_not_ready_requeue.
  (b) reached_p2_alive : P1 survival >= 2/3 (the agent reaches P2 alive). If
      below -> the P2 legs are starved (died before contact) -> the survival leg
      is still de-risking -> substrate_not_ready_requeue, NOT a P2 verdict.
With BOTH preconditions met, a sub-2/3 G0/G2/G3 is a GENUINE residual-leg verdict
(foraging/benefit-contact or goal-formation still open), routed to /failure-autopsy.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; gates the GAP-2 behavioural cohort, weights no claim)
supersedes: V3-EXQ-603g (same full-curriculum readiness gate; 603m adds the
  2026-06-09 harm-pathway training so the survival leg clears and the P2 legs are
  measurable, and promotes consumption-gated G3 into the PASS rule).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    classify_interpretation_branch,
    evaluate_substrate_gate,
    stage_plan,
    substrate_readiness_from_results,
)

EXPERIMENT_TYPE = "v3_exq_603m_scaffolded_sd054_full_curriculum_readiness"
QUEUE_ID = "V3-EXQ-603m"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_603g_scaffolded_sd054_substrate_readiness"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "ALL_LEVERS_ON_PLUS_HARM_PATHWAY_TRAINED"

# Goal-pipeline / encoder dims (mirror 603g / 603k exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0  # SD-012 amplification (two-part-fix precondition)

# Restored full budget (matches 603g / 603k).
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

# Isolated hazard-avoidance Stage-H (603g curriculum-decomposition amend).
HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
# NAV-competence: spawn IN the reef refuge (nav-to-safety handed) so the full
# curriculum exercises the survival leg under the same nav-handed condition that
# 603k validated, then re-unfreezes the goal pipeline in P1 + measures P2.
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603g / 603k).
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# SD-058 / MECH-357 protective-scaffold anneal (mirror 603k).
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# Harm-pathway lr (the 2026-06-09 amend under test, ON for the single arm).
HARM_PATHWAY_LR = 1e-3

# Pre-registered gate constants (NOT derived from the run's own statistics).
STAGE0_ZGOAL_GATE = 0.4
STAGE0B_RETENTION_GATE = 0.75
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0

# Non-vacuity readiness floors (mirror 603k).
# OFF flat baseline harm_eval range was [0.522,0.524] (~0.002); a trained ON head
# clears this floor. >= 1 optimizer step confirms the pathway actually trained.
HARM_EVAL_RANGE_FLOOR = 0.005
HARM_TRAIN_STEPS_FLOOR = 1.0


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
        # developmental-window / consolidation amend (2026-06-03b)
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        # 634c seeding calibration (2026-06-03c)
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        # foraging-competence residual amend (2026-06-05)
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        # SD-057 cue-recall bridge (wean-to-wild contact lever)
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # curriculum-decomposition amend (2026-06-07): isolated Stage-H
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (mirror 603k)
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE: feed the env harm stream so z_harm / z_harm_a populate
        scaffold_feed_harm_stream=True,
        # ===== THE 2026-06-09 AMEND UNDER TEST: harm-pathway training ON =====
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        # sub-flags default True -> all four harm terms engage given the agent
        # carries the sensory z_harm stream + e2_harm_s (set in _make_config).
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        # Sensory z_harm stream (SD-010) + affective stream (SD-011) so all four
        # harm-pathway training terms engage.
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # E2_harm_s forward model (ARC-033) so harm-pathway term 4 engages.
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
        # SD-056 e2 contrastive warmup (mirror 603i / 603k).
        e2_action_contrastive_enabled=True,
        # MECH-279 PAG freeze-gate.
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        # SD-058 / MECH-357 instrumental-avoidance gate.
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
    )
    cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    return cfg


def _aborted_seed_record(seed: int, stage: str, reason: str,
                         s0_peak: float = 0.0, s0b_pass: bool = False) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "stage0b_retention_gate_passed": bool(s0b_pass),
        "p0_mean_episode_length": 0.0,
        "hazard_stage_survival_pass": False,
        "hazard_stage_median_last_window": 0.0,
        "harm_pathway_n_train_steps": 0,
        "harm_eval_range": 0.0,
        "p1_survival_pass": False,
        "p1_median_last_window_episode_length": 0.0,
        "p2_contact_rate": 0.0,
        "p2_contact_steps": 0,
        "p2_num_contact_events": 0,
        "p2_z_goal_norm_peak": 0.0,
        "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_n_cue_recall_fires": 0,
        "g0_stage0_zgoal": False,
        "g1_p1_survival": False,
        "g2_p2_contact": False,
        "g3_p2_zgoal_consumption": False,
        "reached_hazard_stage": False,
        "reached_p1": False,
        "reached_p2": False,
        "seed_pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int):
    """Returns (per_seed_dict, stage0_result, p1_result, p2_metrics) so the
    caller can feed the scheduler's own substrate_readiness_from_results()."""
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)

    # Stage 0 -- forced-benefit nursery (the z_goal-formation positive control).
    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    token_bank = int(getattr(s0, "token_bank_size_end", 0))
    print(
        f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
        f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}"
        f" token_bank_size_end={token_bank}",
        flush=True,
    )
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return (_aborted_seed_record(seed, "stage0", s0.abort_reason,
                                     s0_peak=s0.z_goal_norm_peak), None, None, None)

    # Stage 0b -- PROTECTED consolidation.
    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(
        f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
        f" retention={s0b.retention_ratio:.3f}"
        f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}",
        flush=True,
    )
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return (_aborted_seed_record(seed, "stage0b", s0b.abort_reason,
                                     s0_peak=s0.z_goal_norm_peak,
                                     s0b_pass=s0b.retention_gate_passed), None, None, None)

    # Stage 1 -- guided low-conflict warm-up (run_p0, goal frozen, reef refuge).
    # Harm pathway co-trains in P0 (scaffold_harm_pathway_in_p0=True).
    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(
        f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
        f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
        flush=True,
    )
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        rec = _aborted_seed_record(seed, "p0", p0.abort_reason,
                                   s0_peak=s0.z_goal_norm_peak,
                                   s0b_pass=s0b.retention_gate_passed)
        rec["p0_mean_episode_length"] = float(p0.mean_episode_length)
        return (rec, s0, None, None)

    # Stage-H -- ISOLATED HAZARD-AVOIDANCE with harm-pathway training ON. The
    # harm_discriminativeness probe runs at the end of this stage (only when the
    # harm pathway trained) -- the readiness signal for the non-vacuity gate.
    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    harm_diag = dict(getattr(hz, "harm_discriminativeness", {}) or {})
    harm_pathway_diag = dict(getattr(hz, "harm_pathway_diag", {}) or {})
    harm_eval_range = float(harm_diag.get("harm_eval_range", 0.0))
    harm_train_steps = int(harm_pathway_diag.get("n_train_steps", 0))
    print(
        f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
        f" mean_len={hz.mean_episode_length:.1f}"
        f" median_last={hz.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
        f" harm_eval_range={harm_eval_range:.4f} harm_train_steps={harm_train_steps}",
        flush=True,
    )
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        rec = _aborted_seed_record(seed, "hazard", hz.abort_reason,
                                   s0_peak=s0.z_goal_norm_peak,
                                   s0b_pass=s0b.retention_gate_passed)
        rec["p0_mean_episode_length"] = float(p0.mean_episode_length)
        rec["harm_pathway_n_train_steps"] = harm_train_steps
        rec["harm_eval_range"] = harm_eval_range
        return (rec, s0, None, None)

    # Stage 2+3 -- easy->guarded foraging (run_p1; goal re-unfrozen). P1 is now
    # entered by an already-survival-AND-goal-competent policy.
    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(
        f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
        f" median_last={p1.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
        f" reef_spawn_eps={getattr(p1, 'n_reef_spawn_episodes', 0)}"
        f" refresh={p1.n_contact_refresh_updates}"
        f" decay_only={p1.n_decay_only_updates}",
        flush=True,
    )

    # Stage 4 -- frozen-policy guarded measurement (run_p2).
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    p2_cue_diag = dict(getattr(p2, "cue_diag", {}) or {})
    cue_fires = int(p2_cue_diag.get("n_cue_recall_fires", getattr(p2, "n_cue_recall_fires", 0)))
    print(
        f"  [train] p2_measure seed={seed} ep {done}/{total_eps}"
        f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
        f" z_goal_frozen={p2.z_goal_norm_peak_max:.4f}"
        f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}"
        f" cue_fires={cue_fires} hfa_used={p2.hazard_food_attraction_used:.2f}",
        flush=True,
    )

    # Per-seed FULL four-leg gate (G0/G1/G2/G3). G3 = consumption-event-gated P2
    # z_goal (z_goal_norm_at_contact_peak) -- the ecological wanting leg.
    g0 = bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE)
    g1 = bool(p1.survival_gate_passed)
    g2 = bool(p2.contact_rate > CONTACT_GATE)
    g3 = bool(p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE)
    seed_pass = bool(g0 and g1 and g2 and g3)
    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed}"
        f" g0={g0} g1={g1} g2={g2} g3={g3} g_h={bool(hz.survival_gate_passed)}",
        flush=True,
    )

    rec = {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "stage0_z_goal_formed": bool(s0.z_goal_formed),
        "stage0_token_bank_size_end": token_bank,
        "stage0b_retention_ratio": float(s0b.retention_ratio),
        "stage0b_retention_gate_passed": bool(s0b.retention_gate_passed),
        "p0_mean_episode_length": float(p0.mean_episode_length),
        # Stage-H survival diagnostic (G_H) + harm-pathway non-vacuity readouts.
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
        "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
        "hazard_stage_n_episodes": int(hz.n_episodes),
        "harm_pathway_n_train_steps": harm_train_steps,
        "harm_eval_range": harm_eval_range,
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
        "p1_n_reef_spawn_episodes": int(getattr(p1, "n_reef_spawn_episodes", 0)),
        "p1_n_contact_refresh_updates": int(p1.n_contact_refresh_updates),
        "p1_n_decay_only_updates": int(p1.n_decay_only_updates),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_contact_steps": int(p2.contact_steps),
        "p2_num_contact_events": int(p2.num_contact_events),
        "p2_z_goal_norm_peak": float(p2.z_goal_norm_peak_max),                  # frozen (diagnostic)
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),  # consumption-gated (G3)
        "p2_n_cue_recall_fires": cue_fires,
        "p2_hazard_food_attraction_used": float(p2.hazard_food_attraction_used),
        "g0_stage0_zgoal": g0,
        "g1_p1_survival": g1,
        "g2_p2_contact": g2,
        "g3_p2_zgoal_consumption": g3,
        "reached_hazard_stage": True,
        "reached_p1": True,
        "reached_p2": True,
        "seed_pass": seed_pass,
    }
    return (rec, s0, p1, p2)


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS

    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET
        )

    per_seed: List[Dict[str, Any]] = []
    stage0_results = []
    p1_results = []
    p2_metrics = []
    for s in seeds:
        rec, s0, p1, p2 = _run_seed(s, dry_run, total_eps)
        per_seed.append(rec)
        if s0 is not None:
            stage0_results.append(s0)
        if p1 is not None:
            p1_results.append(p1)
        if p2 is not None:
            p2_metrics.append(p2)

    n = len(per_seed)

    # --- The full four-leg gate (each >= 2/3 seeds) ---
    g0_frac = _frac([r["g0_stage0_zgoal"] for r in per_seed])
    g1_frac = _frac([r["g1_p1_survival"] for r in per_seed])
    g2_frac = _frac([r["g2_p2_contact"] for r in per_seed])
    g3_frac = _frac([r["g3_p2_zgoal_consumption"] for r in per_seed])
    g0_pass = g0_frac >= MIN_FRACTION
    g1_pass = g1_frac >= MIN_FRACTION
    g2_pass = g2_frac >= MIN_FRACTION
    g3_pass = g3_frac >= MIN_FRACTION
    overall_pass = bool(g0_pass and g1_pass and g2_pass and g3_pass)

    # --- Non-vacuity readiness preconditions (the harm-pathway + reached-P2-alive
    # guards). A below-floor reading self-routes substrate_not_ready_requeue. ---
    harm_train_steps_max = max((r.get("harm_pathway_n_train_steps", 0) for r in per_seed), default=0)
    harm_eval_range_max = max((r.get("harm_eval_range", 0.0) for r in per_seed), default=0.0)
    harm_pathway_trained = bool(harm_train_steps_max >= HARM_TRAIN_STEPS_FLOOR)
    harm_pathway_discriminative = bool(harm_pathway_trained and harm_eval_range_max >= HARM_EVAL_RANGE_FLOOR)
    reached_p2_alive = bool(g1_pass)  # P1 survival >=2/3 -> the agent reaches P2 alive

    preconditions_met = bool(harm_pathway_discriminative and reached_p2_alive)

    # Stage-H survival diagnostic (G_H) -- reported, NOT in the PASS rule.
    g_h_frac = _frac([r.get("hazard_stage_survival_pass", False) for r in per_seed])

    # --- Canonical readiness (consumption-event-gated G3); the scheduler's own
    # four-gate helper, cross-checked against the per-seed gate above. ---
    if stage0_results and p1_results and p2_metrics:
        canonical = substrate_readiness_from_results(
            stage0_results, p1_results, p2_metrics,
            z_goal_gate=STAGE0_ZGOAL_GATE,
            contact_gate=CONTACT_GATE,
            min_fraction=MIN_FRACTION,
            use_consumption_gated_g3=True,
        )
    else:
        canonical = evaluate_substrate_gate(
            [r["stage0_z_goal_norm_peak"] for r in per_seed],
            [r["p1_survival_pass"] for r in per_seed],
            [r["p2_z_goal_norm_at_contact_peak"] for r in per_seed],
            [r["p2_contact_rate"] for r in per_seed],
            z_goal_gate=STAGE0_ZGOAL_GATE, contact_gate=CONTACT_GATE, min_fraction=MIN_FRACTION,
        )
        canonical["g3_source"] = "z_goal_norm_at_contact_peak"
    branch = classify_interpretation_branch(canonical)

    # --- Routing (diagnostic adjudication gate). Preconditions decide whether the
    # gate outcome is a real substrate verdict at all. ---
    if not preconditions_met:
        # Harm landscape is noise (603i regression) OR the agent died before P2:
        # the P2 legs are starved -> re-run, never a foraging/substrate verdict.
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
    elif overall_pass:
        outcome = "PASS"
        readiness_route = "substrate_ready_flip_scaffolded_sd054_onboarding"
    else:
        outcome = "FAIL"
        # Name the residual leg(s) that are genuinely open (preconditions met).
        open_legs = []
        if not g0_pass:
            open_legs.append("stage0_zgoal")
        if not g2_pass:
            open_legs.append("p2_contact")
        if not g3_pass:
            open_legs.append("p2_zgoal_consumption")
        readiness_route = "residual_leg_open:" + ("+".join(open_legs) if open_legs else "unknown")

    # --- Diagnostic adjudication structures (preconditions + criteria). ---
    # The readiness precondition statistic the gate routes on is the harm_eval
    # range (harm-pathway discriminativeness) + the P1-survival fraction
    # (reached-P2-alive). The indexer recomputes met from measured>=threshold.
    preconditions = [
        {
            "name": "harm_pathway_discriminative",
            "kind": "readiness",
            "description": "The harm pathway must have trained (>=1 optimizer step) AND "
                           "harm_eval(z_world) range must lift above the flat ~0.002 603i "
                           "baseline. Positive control that the survival/P2 legs run on a "
                           "TRAINED harm landscape, not noise. Below-floor => harm-landscape-"
                           "noise (603i regression) => substrate_not_ready_requeue, NOT a "
                           "foraging verdict.",
            "control": "scaffold_train_harm_pathway=True co-trains harm_eval/z_harm/z_harm_a/"
                       "E2_harm_s in P0+Stage-H; harm_discriminativeness probe measures the "
                       "post-training harm_eval range over states.",
            "measured": float(harm_eval_range_max),
            "threshold": float(HARM_EVAL_RANGE_FLOOR),
            "met": bool(harm_pathway_discriminative),
        },
        {
            "name": "reached_p2_alive",
            "kind": "readiness",
            "description": "P1 survival must clear >=2/3 seeds so the agent reaches P2 ALIVE; "
                           "otherwise a P2-contact-zero / P2-z_goal-zero read is 'died before "
                           "contact' not 'foraging incompetence'. Below-floor => the survival "
                           "leg is still de-risking => substrate_not_ready_requeue, NOT a P2 "
                           "verdict. (Same statistic as the load-bearing G1 gate leg.)",
            "control": "P1 survival gate = median episode length over the last 10 P1 episodes "
                       ">= 75, entered by a Stage-H-survival-competent policy.",
            "measured": float(g1_frac),
            "threshold": float(MIN_FRACTION),
            "met": bool(reached_p2_alive),
        },
    ]
    frac_reached_p1 = _frac([r.get("reached_p1", False) for r in per_seed])
    frac_reached_p2 = _frac([r.get("reached_p2", False) for r in per_seed])
    frac_reached_hazard = _frac([r.get("reached_hazard_stage", False) for r in per_seed])
    criteria_non_degenerate = {
        "G0_stage0": bool(n >= 2),
        "G1_survival": bool(frac_reached_hazard >= MIN_FRACTION),
        "G2_contact": bool(frac_reached_p2 >= MIN_FRACTION),
        "G3_zgoal": bool(frac_reached_p2 >= MIN_FRACTION),
        "G_H_hazard_stage": bool(frac_reached_hazard >= MIN_FRACTION),
        "p1_reached": bool(frac_reached_p1 >= MIN_FRACTION),
    }
    criteria = [
        {"name": "G0_stage0_positive_control", "load_bearing": True, "passed": bool(g0_pass)},
        {"name": "G1_p1_survival", "load_bearing": True, "passed": bool(g1_pass)},
        {"name": "G2_p2_contact", "load_bearing": True, "passed": bool(g2_pass)},
        {"name": "G3_p2_zgoal_consumption", "load_bearing": True, "passed": bool(g3_pass)},
    ]

    task_gate = {
        "g0_stage0_positive_control": bool(g0_pass),
        "g1_p1_survival": bool(g1_pass),
        "g2_p2_contact": bool(g2_pass),
        "g3_p2_zgoal_consumption": bool(g3_pass),
        "overall_pass": overall_pass,
        "min_fraction": MIN_FRACTION,
        "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
        "p2_z_goal_gate": P2_ZGOAL_GATE,
        "contact_gate": CONTACT_GATE,
        "g0_fraction": g0_frac,
        "g1_fraction": g1_frac,
        "g2_fraction": g2_frac,
        "g3_fraction": g3_frac,
        "per_seed_g0": [r["g0_stage0_zgoal"] for r in per_seed],
        "per_seed_g1": [r["g1_p1_survival"] for r in per_seed],
        "per_seed_g2": [r["g2_p2_contact"] for r in per_seed],
        "per_seed_g3": [r["g3_p2_zgoal_consumption"] for r in per_seed],
    }

    hazard_stage_gate = {
        "g_h_hazard_stage_survival": bool(g_h_frac >= MIN_FRACTION),
        "g_h_fraction": float(g_h_frac),
        "per_seed_g_h": [r.get("hazard_stage_survival_pass", False) for r in per_seed],
        "harm_eval_range_max": float(harm_eval_range_max),
        "harm_train_steps_max": int(harm_train_steps_max),
        "note": "diagnostic only -- NOT in the PASS rule (PASS = G0 AND G1 AND G2 AND G3). "
                "G_H + harm_eval_range confirm the isolated Stage-H trained the harm pathway.",
    }

    print(
        f"[{EXPERIMENT_TYPE}] full gate G0={g0_pass} G1={g1_pass} G2={g2_pass} G3={g3_pass}"
        f" (G_H_diag={hazard_stage_gate['g_h_hazard_stage_survival']}) -> outcome={outcome}",
        flush=True,
    )
    print(
        f"[{EXPERIMENT_TYPE}] preconditions harm_discriminative={harm_pathway_discriminative}"
        f" (range_max={harm_eval_range_max:.4f}) reached_p2_alive={reached_p2_alive}"
        f" -> readiness_route={readiness_route}",
        flush=True,
    )

    return {
        "outcome": outcome,
        "substrate_gate_passed_full_four": overall_pass,
        "preconditions_met": preconditions_met,
        "task_gate": task_gate,
        "hazard_stage_gate": hazard_stage_gate,
        "canonical_readiness_gate": canonical,
        "interpretation": {
            "label": branch,
            "readiness_route": readiness_route,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
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
        "evidence_direction": "non_contributory",  # diagnostic; tags no claim
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> P0 "
                     "-> Stage-H -> P1 -> P2; harm-pathway training ON, 2026-06-09 amend)",
        "condition": CONDITION_LABEL,
        "amend_note": "FULL three-leg readiness run with scaffold_train_harm_pathway=ON after "
                      "V3-EXQ-603k cleared ONLY the survival leg (P1 survival 3/3) of the "
                      "scaffolded_sd054_onboarding gate. 603k was a survival-only probe (never "
                      "exercised P2 benefit-contact / P2 z_goal; Stage-0 z_goal>0.4 on 1/3). "
                      "603m runs the whole curriculum with harm-pathway ON so the agent reaches "
                      "P2 alive and all four gate legs are measurable in one run. PASS = G0 AND "
                      "G1 AND G2 AND G3 (each >=2/3). Non-vacuity: harm-pathway-discriminative "
                      "AND reached-P2-alive, else substrate_not_ready_requeue.",
        "predecessor": "V3-EXQ-603k (survival-only harm-pathway probe; G_H 2/3, P1 survival 3/3, "
                       "Stage-0 z_goal>0.4 on 1/3; P2 legs unexercised).",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "g0_stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "g1_p1_survival": "median episode length over last 10 P1 episodes >= 75",
            "g2_contact_gate": CONTACT_GATE,
            "g3_p2_zgoal_consumption_gated": "z_goal_norm_at_contact_peak > 0.4 (the ecological "
                                             "wanting leg; consumption-event-gated, NOT the "
                                             "carried forced-feed nursery trace)",
            "g_h_hazard_stage_survival": "median episode length over last 10 Stage-H episodes "
                                         ">= 75 (DIAGNOSTIC; NOT in the PASS rule)",
            "min_fraction": MIN_FRACTION,
            "pass_rule": "PASS = G0 AND G1 AND G2 AND G3 (each >= 2/3 seeds)",
            "non_vacuity_harm_eval_range_floor": HARM_EVAL_RANGE_FLOOR,
            "non_vacuity_harm_train_steps_floor": HARM_TRAIN_STEPS_FLOOR,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "train_steps": TRAIN_STEPS, "p1_anneal_hold_fraction": P1_HOLD_FRACTION,
            "p0_num_hazards": P0_NUM_HAZARDS, "p2_hfa_guard": P2_HFA_GUARD,
            "p1_reef_spawn_hold_fraction": P1_REEF_SPAWN_HOLD_FRACTION,
            "hazard_stage_num_hazards": HAZARD_STAGE_NUM_HAZARDS,
            "hazard_stage_num_resources": HAZARD_STAGE_NUM_RESOURCES,
            "hazard_stage_hazard_food_attraction": HAZARD_STAGE_HFA,
            "hazard_stage_proximity_harm_scale": HAZARD_STAGE_PROXIMITY_HARM,
            "hazard_stage_spawn_in_reef_half": HAZARD_STAGE_SPAWN_IN_REEF,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "scaffold_train_harm_pathway": True,
            "scaffold_harm_pathway_lr": HARM_PATHWAY_LR,
            "scaffold_harm_pathway_in_p0": True,
            "scaffold_feed_harm_stream": True,
            "scaffold_avoidance_driver_enabled": True,
            "use_harm_stream": True,
            "use_e2_harm_s_forward": True,
            "use_pag_freeze_gate": True,
            "use_instrumental_avoidance": True,
            "auto_reconcile_gating_to_seeding": True,
            "seeding_gain": SEED_GAIN, "seeding_benefit_threshold": SEED_BENEFIT_THRESHOLD,
            "seeding_drive_floor": SEED_DRIVE_FLOOR,
            "cue_recall_bridge_enabled": True, "cue_n_resource_types": N_RESOURCE_TYPES,
            "stage0_bind_incentive_token": True, "cue_recall_gain": CUE_RECALL_GAIN,
            "developmental_window_enabled": True, "stage0b_enabled": True,
            "contact_gated_goal_updates": True,
            "z_goal_enabled": True, "drive_weight": DRIVE_WEIGHT,
        },
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
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
        )
