"""
V3-EXQ-687 -- Q-045 / MECH-313 / MECH-260 4-arm tonic-noise / exploration-floor
ablation on the now-SURVIVAL-COMPETENT scaffolded_sd054_onboarding substrate.

THE QUESTION (Q-045, registered falsifier):
  "Is MECH-313 (LC-NE tonic noise floor) a separate parameter from MECH-260 (dACC
  anti-recency penalty), or do they collapse into one anti-monostrategy substrate?
  Falsifiable with the four-arm ablation: both-OFF / 313-only / 260-only / both-ON,
  on SD-054 reef substrate with ARC-062 gated-policy enabled."

This script IS that registered four-arm ablation. The 2x2 grid varies ONLY the two
mechanisms under test; ARC-062 use_gated_policy is enabled on all arms (per the Q-045
spec); every other substrate is held CONSTANT.

WHY A NEW NUMBER (not a 603-letter):
  The prior Q-045 four-arm chain (V3-EXQ-603 / 603a / 603b / 603c / 603d / 603e) ALL
  bottomed at substrate-readiness, NEVER at the science: effective N=1 (seeds 42/44
  died on the SD-054 enrichment before a measurement window), z_goal=0 ecologically,
  and the MECH-260 FIFO non-operative on a call-path bypass. Q-045 was routed to
  epistemic_category=substrate_ceiling (claims.yaml 2026-05-27) precisely because its
  test bed was not survivable. The 603-series number has since drifted entirely to the
  SD-059/MECH-358 escape-affordance-bridge + harm-pathway lineage (603i..603q). This is
  the SAME scientific question on a SUBSTANTIALLY new (now survivable) substrate, so it
  takes a fresh number.

WHAT CHANGED -- the substrate is now survival-competent:
  The 603 lineage built and validated a survival-competence stack ON scaffolded_sd054:
  Stage-H isolated hazard-avoidance + harm-pathway training (603k) + its 2026-06-16
  stabilization amend (decoupled encoder LR + warmup) + the SD-058/MECH-357 instrumental
  -avoidance gate + the SD-059/MECH-358 escape-affordance bridge. V3-EXQ-603q PASSed
  2026-06-17T04:28:30Z on exactly this stack. This experiment ports that survival config
  VERBATIM and holds it CONSTANT across all four arms -- it is scaffolding that makes the
  agent reach a measurement window, NOT what is under test. Only use_noise_floor (MECH-313)
  and use_dacc/dacc_suppression (MECH-260) differ between arms.

THE FOUR ARMS (the Q-045 2x2; ARC-062 gated-policy on all):
  ARM_0_both_off       -- MECH-313 OFF, MECH-260 OFF (monomodal-collapse control).
  ARM_1_mech313_only   -- use_noise_floor=True (alpha=0.5), MECH-260 OFF.
  ARM_2_mech260_only   -- use_dacc=True (suppression_weight=0.5, memory=8), MECH-313 OFF.
  ARM_3_both_on        -- MECH-313 + MECH-260 jointly active.
  (No FP-2 matched-noise 5th arm: that is MECH-313's separate structured-vs-noise
  falsifier, not the Q-045 collapse question. This run is the clean 4-arm the claim
  registers.)

CURRICULUM (held CONSTANT across all 4 arms; ported from V3-EXQ-603q):
  Stage-0 forced-feed nursery -> Stage-0b protected consolidation -> P0 encoder warm-up
  (goal frozen) -> Stage-H isolated hazard-avoidance (harm-pathway trained + stabilized)
  -> P1 combined wean. Then a bespoke FROZEN-POLICY P2 diversity measurement.

P2 DIVERSITY MEASUREMENT (frozen policy; the Q-045 dependent variable):
  Per arm/seed, the trained agent is run frozen and we measure, past a FIFO warmup:
  selected-action-class entropy (primary diversity metric), position entropy,
  reef-occupancy fraction, z_goal_norm_peak (substrate-engagement), and the MECH-260
  dACC FIFO diagnostics (forward_calls / history_len / max_suppression -> mech260_operative).
  MECH-313's noise floor is applied INSIDE select_action via config (no manual temperature
  passing -- the current canonical path), so an arm's diversity reflects the mechanism, not
  a hand-set temperature.

PRE-REGISTERED NON-VACUITY PRECONDITIONS (the load-bearing discipline; constants below).
  Each guards a documented 603-lineage failure mode and SELF-ROUTES
  substrate_not_ready_requeue (evidence_direction non_contributory) when unmet --
  NEVER a false `weakens`. A genuine `weakens` (FAIL_NO_DIVERSITY) is reachable ONLY when
  every precondition holds (i.e. the claim was actually fairly tested on a working substrate):
    PRE_REACH   -- each arm has >=2/3 seeds that reach P2 with measured_steps >= floor
                   (the effective-N=1 killer that sank 603a/b/c).
    PRE_ZGOAL   -- z_goal_norm_peak > Z_GOAL_FLOOR on >=2/3 of P2-reaching cells
                   (the z_goal=0 substrate-engagement killer; Berridge developmental prereq).
    PRE_MECH260 -- on the dACC arms (260-only, both-on), mech260_operative on >=2/3 seeds
                   (guards the 603 call-path-bypass bug: FIFO empty -> suppression zero ->
                   MECH-260 structurally absent -> cannot be tested).
    PRE_NONDEGEN-- the selected-action-entropy metric is non-degenerate across cells
                   (not structurally pinned; computed via _metrics.check_degeneracy).
  MECH-313's noise floor is config-deterministic; noise_floor_active is recorded but does
  not gate (an enabled NoiseFloor always lifts the effective temperature).

EVIDENCE GRID (only read when ALL preconditions hold; otherwise non_contributory):
  FAIL_NO_DIVERSITY (both-on NOT > both-off + margin) -> Q-045/MECH-313/MECH-260 all
    `weakens` (genuine falsification on a working substrate: even both mechanisms produce
    no behavioural diversity).
  MUTUALLY_LOAD_BEARING (both-on > max(singletons)+margin AND each singleton > off+margin)
    -> all three `supports` (both promote; the registered "mutually load-bearing" outcome).
  MECH_313_DOMINATES (313-only ~= both-on; 260-only below) -> Q-045/MECH-313 `supports`,
    MECH-260 `mixed` (contributory-not-necessary -- the registered 313-dominant outcome).
  MECH_260_DOMINATES (260-only ~= both-on; 313-only below) -> Q-045/MECH-260 `supports`,
    MECH-313 `mixed`.
  DIRECTIONALLY_COUPLED (both-on departs from the linear sum of the singleton lifts by
    > margin -- super- or sub-additive; the Q-045 R4 lit-pull "DIRECTIONALLY COUPLED" 4th
    category, Tervo 2014 LC->ACC asymmetry) -> Q-045 `mixed`; each MECH `supports` if its
    singleton beats off. Flags the 8-cell (4-arm x 2-LC-amplitude) follow-on the R4
    synthesis recommends.
  PARTIAL_LIFT (both-on beats off but no clean resolution) -> Q-045 `mixed`; each MECH
    `supports`/`mixed` by whether its singleton beats off.

experiment_purpose: evidence   (scored governance evidence for Q-045 / MECH-313 / MECH-260)
claim_ids: [Q-045, MECH-313, MECH-260]   (the four-arm grid tests all three directly)
SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking goal-pipeline
  onboarding scheduler).

Smoke:
    /opt/local/bin/python3 experiments/v3_exq_687_q045_mech313_mech260_4arm_tonic_noise_ablation.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _benefit_and_drive,
    _build_env,
    _sense_with_optional_harm,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_687_q045_mech313_mech260_4arm_tonic_noise_ablation"
QUEUE_ID = "V3-EXQ-687"
CLAIM_IDS: List[str] = ["Q-045", "MECH-313", "MECH-260"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]

# ----- Goal-pipeline / encoder dims (mirror 603q exactly) -------------------------
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# ----- Survival-competence curriculum budgets (mirror 603q) -----------------------
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

# Stage-H isolated hazard-avoidance regime (mirror 603q's amend-validated regime).
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603q).
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# SD-058/MECH-357 protective-scaffold anneal + PAG (mirror 603q).
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# SD-059/MECH-358 escape-affordance bridge knobs (mirror 603q; ON all arms).
ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

# Harm-pathway training + stabilization (603k + 603q amend).
HARM_PATHWAY_LR = 1e-3
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250

# ----- MECH-313 / MECH-260 levers (the ONLY cross-arm variables; mirror 603e) -----
NOISE_FLOOR_ALPHA = 0.5
DACC_SUPPRESSION_WEIGHT = 0.5
DACC_SUPPRESSION_MEMORY = 8

# ----- Frozen-policy P2 diversity-measurement budget ------------------------------
EVAL_EPISODES = 30
P2_STEPS_PER_EPISODE = 300
FIFO_WARMUP_STEPS = 30  # >= 2 * DACC_SUPPRESSION_MEMORY so the FIFO fills before measuring

# ----- Pre-registered acceptance / precondition constants -------------------------
ENTROPY_MARGIN = 0.05          # diversity-lift / coupling margin on selected-action entropy
Z_GOAL_FLOOR = 0.4             # PRE_ZGOAL substrate-engagement threshold (591 7-criterion C1)
MEASURED_STEPS_FLOOR = 100     # PRE_REACH minimum measured steps for a cell to count
MIN_FRACTION = 2.0 / 3.0       # >=2/3 seeds for every precondition

# Total expected verdict lines (seeds x arms); set in run_experiment.
TOTAL_RUNS = 0


ARMS: List[Dict[str, Any]] = [
    {"label": "ARM_0_both_off", "use_noise_floor": False, "use_dacc": False},
    {"label": "ARM_1_mech313_only", "use_noise_floor": True, "use_dacc": False},
    {"label": "ARM_2_mech260_only", "use_noise_floor": False, "use_dacc": True},
    {"label": "ARM_3_both_on", "use_noise_floor": True, "use_dacc": True},
]
DACC_ARM_LABELS = {"ARM_2_mech260_only", "ARM_3_both_on"}
NOISE_ARM_LABELS = {"ARM_1_mech313_only", "ARM_3_both_on"}


# ---------------------------------------------------------------------------
# Config builders (survival stack constant; only the 313/260 levers vary)
# ---------------------------------------------------------------------------

def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, steps = 2, 2, 5, 5, 5, 30
    else:
        stage0, stage0b, p0, hazard, p1, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=1,  # scheduler P2 unused; this script owns P2 measurement
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # Stage-H isolated hazard-avoidance (survival leg).
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=False,  # standard (not the nav-control)
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058/MECH-357 avoidance-learning driver (ALL arms).
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # Feed the env harm stream so z_harm / z_harm_a populate (ALL arms).
        scaffold_feed_harm_stream=True,
        # Stage-H harm-pathway training + stabilization (ALL arms).
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, arm: Dict[str, Any]) -> REEConfig:
    """Survival stack held CONSTANT; ONLY use_noise_floor (MECH-313) and the dACC
    suppression knobs (MECH-260) vary per arm. ARC-062 use_gated_policy ON all arms."""
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
        # ARC-062 gated-policy ENABLED on all arms (the Q-045 falsifier spec).
        use_gated_policy=True,
        gated_policy_use_first_action_onehot=True,
        # Survival stack (constant across arms): PAG + IA gate + escape bridge.
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_escape_affordance_bridge=True,
        use_escape_relief_credit=True,
        use_escape_safety_credit=True,
        escape_threat_floor=ESCAPE_THREAT_FLOOR,
        escape_threat_ref=ESCAPE_THREAT_REF,
        escape_approach_gain=ESCAPE_APPROACH_GAIN,
        escape_bias_scale=ESCAPE_BIAS_SCALE,
        escape_use_trained_safety_signal=True,
        escape_safety_signal_threshold=ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
        # ===== MECH-313 lever (per arm) =====
        use_noise_floor=bool(arm["use_noise_floor"]),
        noise_floor_alpha=(NOISE_FLOOR_ALPHA if arm["use_noise_floor"] else 0.1),
        # ===== MECH-260 lever (per arm) =====
        use_dacc=bool(arm["use_dacc"]),
        dacc_weight=(1.0 if arm["use_dacc"] else 0.0),
        dacc_suppression_weight=(DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0),
        dacc_suppression_memory=DACC_SUPPRESSION_MEMORY,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _config_slice(arm: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """Content-addressed config slice for the per-cell arm fingerprint. The two
    cross-arm levers + the constant survival stack identity."""
    return {
        "arm": arm["label"],
        # The two mechanisms under test (the cross-arm variables):
        "use_noise_floor": bool(arm["use_noise_floor"]),
        "noise_floor_alpha": (NOISE_FLOOR_ALPHA if arm["use_noise_floor"] else 0.1),
        "use_dacc": bool(arm["use_dacc"]),
        "dacc_suppression_weight": (DACC_SUPPRESSION_WEIGHT if arm["use_dacc"] else 0.0),
        "dacc_suppression_memory": DACC_SUPPRESSION_MEMORY,
        # Constant survival-stack identity:
        "use_gated_policy": True,
        "use_pag_freeze_gate": True,
        "use_instrumental_avoidance": True,
        "use_escape_affordance_bridge": True,
        "scaffold_train_harm_pathway": True,
        "harm_pathway_lr": HARM_PATHWAY_LR,
        "harm_pathway_encoder_lr": HARM_PATHWAY_ENCODER_LR,
        "harm_pathway_warmup_steps": HARM_PATHWAY_WARMUP_STEPS,
        "feed_harm_stream": True,
        "world_dim": WORLD_DIM,
        "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, TRAIN_STEPS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "p2": [EVAL_EPISODES, P2_STEPS_PER_EPISODE, FIFO_WARMUP_STEPS],
        "dry_run": bool(dry_run),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _z_goal_norm(agent: REEAgent) -> float:
    gs = getattr(agent, "goal_state", None)
    if gs is None:
        return 0.0
    if hasattr(gs, "goal_norm"):
        try:
            return float(gs.goal_norm())
        except TypeError:
            return float(gs.goal_norm)
    return 0.0


def _dacc_diag(agent: REEAgent) -> Dict[str, Any]:
    dacc = getattr(agent, "dacc", None)
    if dacc is None:
        return {"dacc_forward_calls": 0, "dacc_history_len": 0, "dacc_max_suppression": 0.0}
    hist_len = len(getattr(dacc, "_action_history", []))
    max_sup = 0.0
    bundle = getattr(dacc, "_last_bundle", None)
    if bundle is not None and isinstance(bundle, dict) and "suppression" in bundle:
        sup = bundle["suppression"]
        if isinstance(sup, torch.Tensor) and sup.numel() > 0:
            max_sup = float(sup.max().item())
    return {
        "dacc_forward_calls": int(getattr(dacc, "_n_forward_calls", 0)),
        "dacc_history_len": int(hist_len),
        "dacc_max_suppression": round(max_sup, 6),
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _mean(vals: List[float]) -> float:
    return float(sum(vals)) / float(len(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Frozen-policy P2 diversity measurement (current canonical stepping path)
# ---------------------------------------------------------------------------

def _run_p2_measurement(agent: REEAgent, scaffold_cfg: ScaffoldedSD054OnboardingConfig,
                        arm: Dict[str, Any], seed: int, device: torch.device,
                        episodes: int, steps_per_episode: int,
                        fifo_warmup_steps: int) -> Dict[str, Any]:
    """Frozen-policy measurement of behavioural diversity on the trained agent.
    Mirrors the scaffold's canonical _eval_episode stepping idiom (sense ->
    clock.advance -> _e1_tick -> generate_trajectories -> select_action -> env.step).
    MECH-313's noise floor is applied INSIDE select_action via config (no manual
    temperature). z_goal is driven post-step (same as _eval_episode) so the
    engagement metric is non-zero. Frozen policy = torch.no_grad() + no optimizer
    step (mirrors the scaffold's canonical _eval_episode; no agent.eval() call)."""
    env = _build_env(scaffold_cfg, "p2")
    world_dim = agent.config.latent.world_dim

    action_counts: Counter = Counter()
    position_counts: Counter = Counter()
    reef_steps = 0
    total_steps = 0
    measured_steps = 0
    z_goal_norm_peak = 0.0
    max_dacc_forward = 0
    max_dacc_history = 0
    max_dacc_suppression = 0.0
    reef_cells: set = set()

    with torch.no_grad():
        for ep in range(episodes):
            _, obs_dict = env.reset()
            agent.reset()
            if ep == 0:
                reef_cells = set(getattr(env, "_reef_cells", set()))
            for step in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    scaffold_cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent)
                    if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

                zg = _z_goal_norm(agent)
                if zg > z_goal_norm_peak:
                    z_goal_norm_peak = zg

                if arm["use_dacc"]:
                    d = _dacc_diag(agent)
                    max_dacc_forward = max(max_dacc_forward, d["dacc_forward_calls"])
                    max_dacc_history = max(max_dacc_history, d["dacc_history_len"])
                    max_dacc_suppression = max(max_dacc_suppression, d["dacc_max_suppression"])

                action_idx = int(action.argmax(dim=-1).item())
                pos = (int(env.agent_x), int(env.agent_y))
                if step >= fifo_warmup_steps:
                    action_counts[action_idx] += 1
                    position_counts[pos] += 1
                    measured_steps += 1
                if pos in reef_cells:
                    reef_steps += 1

                _, _harm, done, _, obs_dict = env.step(action_idx)
                total_steps += 1

                # Drive z_goal post-step so z_goal_norm_peak reflects a live goal
                # pipeline (mirrors _eval_episode; the 603d harness-fix).
                benefit, drive = _benefit_and_drive(obs_dict["body_state"].to(device))
                agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
                zg_post = _z_goal_norm(agent)
                if zg_post > z_goal_norm_peak:
                    z_goal_norm_peak = zg_post

                if done:
                    break

            if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
                # NOTE: deliberately NOT the 'ep N/M' token (that is reserved for the
                # training-loop progress denominator the runner reads).
                print(f"  [P2 eval] arm={arm['label']} seed={seed} eval_ep {ep + 1} of"
                      f" {episodes} measured={measured_steps}", flush=True)

    selection_entropy = _entropy(action_counts)
    position_entropy = _entropy(position_counts)
    reef_fraction = reef_steps / max(total_steps, 1)
    mech260_operative = bool(
        arm["use_dacc"] and max_dacc_forward > 0 and max_dacc_history > 0
        and max_dacc_suppression > 0.0
    )
    row: Dict[str, Any] = {
        "arm": arm["label"],
        "seed": seed,
        "reached_p2": True,
        "selected_action_entropy": round(selection_entropy, 6),
        "position_entropy": round(position_entropy, 6),
        "reef_fraction": round(reef_fraction, 6),
        "z_goal_norm_peak": round(z_goal_norm_peak, 6),
        "unique_actions": len(action_counts),
        "total_steps": int(total_steps),
        "measured_steps": int(measured_steps),
        "fifo_warmup_steps": int(fifo_warmup_steps),
        "use_noise_floor": bool(arm["use_noise_floor"]),
        "noise_floor_active": bool(arm["use_noise_floor"]),
        "use_dacc": bool(arm["use_dacc"]),
    }
    if arm["use_dacc"]:
        row.update({
            "dacc_forward_calls_max": int(max_dacc_forward),
            "dacc_history_len_max": int(max_dacc_history),
            "dacc_max_suppression": round(max_dacc_suppression, 6),
            "mech260_operative": mech260_operative,
        })
    return row


def _empty_row(arm: Dict[str, Any], seed: int, stage: str, reason: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "arm": arm["label"], "seed": seed, "reached_p2": False,
        "aborted_at": stage, "abort_reason": reason,
        "selected_action_entropy": 0.0, "position_entropy": 0.0,
        "reef_fraction": 0.0, "z_goal_norm_peak": 0.0, "unique_actions": 0,
        "total_steps": 0, "measured_steps": 0, "fifo_warmup_steps": int(FIFO_WARMUP_STEPS),
        "use_noise_floor": bool(arm["use_noise_floor"]), "noise_floor_active": False,
        "use_dacc": bool(arm["use_dacc"]),
    }
    if arm["use_dacc"]:
        row.update({"dacc_forward_calls_max": 0, "dacc_history_len_max": 0,
                    "dacc_max_suppression": 0.0, "mech260_operative": False})
    return row


# ---------------------------------------------------------------------------
# Per-cell pipeline: survival curriculum -> frozen-policy P2 diversity measurement
# ---------------------------------------------------------------------------

def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool,
                  total_eps: int) -> Dict[str, Any]:
    with arm_cell(seed, config_slice=_config_slice(arm, dry_run),
                  script_path=Path(__file__)) as cell:
        scaffold_cfg = _make_scaffold_cfg(dry_run)
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env, arm)).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)
        print(f"Seed {seed} Condition {arm['label']}", flush=True)

        done = 0
        # Stage-0 forced-feed nursery.
        s0 = scheduler.run_stage0_nursery(agent, device)
        done += s0.n_episodes
        print(f"  [train] stage0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0", flush=True)
            row = _empty_row(arm, seed, "stage0", s0.abort_reason)
            cell.stamp(row)
            return row

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0b", flush=True)
            row = _empty_row(arm, seed, "stage0b", s0b.abort_reason)
            cell.stamp(row)
            return row

        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
              flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=p0", flush=True)
            row = _empty_row(arm, seed, "p0", p0.abort_reason)
            cell.stamp(row)
            return row

        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        print(f"  [train] hazard {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=hazard", flush=True)
            row = _empty_row(arm, seed, "hazard", hz.abort_reason)
            cell.stamp(row)
            return row

        p1 = scheduler.run_p1(agent, device)
        done += p1.n_episodes
        print(f"  [train] p1 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={p1.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

        # Bespoke frozen-policy P2 diversity measurement (the Q-045 dependent variable).
        p2_eps = 1 if dry_run else EVAL_EPISODES
        p2_steps = 30 if dry_run else P2_STEPS_PER_EPISODE
        fifo = min(FIFO_WARMUP_STEPS, max(0, p2_steps - 1))
        row = _run_p2_measurement(agent, scaffold_cfg, arm, seed, device,
                                  episodes=p2_eps, steps_per_episode=p2_steps,
                                  fifo_warmup_steps=fifo)
        print(f"verdict: PASS seed={seed} arm={arm['label']}"
              f" entropy={row['selected_action_entropy']:.4f}"
              f" z_goal={row['z_goal_norm_peak']:.4f}"
              f" measured={row['measured_steps']}"
              f" mech260_op={row.get('mech260_operative', 'n/a')}", flush=True)
        cell.stamp(row)
        return row


# ---------------------------------------------------------------------------
# Preconditions + Q-045 evidence grid
# ---------------------------------------------------------------------------

def _evaluate_preconditions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)

    # PRE_REACH: each arm has >=2/3 seeds reaching P2 with measured_steps >= floor.
    per_arm_reach: Dict[str, float] = {}
    for label, cells in by_arm.items():
        flags = [bool(c.get("reached_p2")) and c.get("measured_steps", 0) >= MEASURED_STEPS_FLOOR
                 for c in cells]
        per_arm_reach[label] = _frac(flags)
    pre_reach = bool(per_arm_reach) and all(f >= MIN_FRACTION for f in per_arm_reach.values())

    # PRE_ZGOAL: z_goal engaged on >=2/3 of P2-reaching cells (substrate engaged).
    p2_cells = [r for r in rows if r.get("reached_p2")]
    zgoal_flags = [c.get("z_goal_norm_peak", 0.0) > Z_GOAL_FLOOR for c in p2_cells]
    pre_zgoal = bool(p2_cells) and _frac(zgoal_flags) >= MIN_FRACTION

    # PRE_MECH260: dACC FIFO operative on >=2/3 seeds of EACH dACC arm.
    per_dacc_arm_op: Dict[str, float] = {}
    for label in DACC_ARM_LABELS:
        cells = [c for c in by_arm.get(label, []) if c.get("reached_p2")]
        per_dacc_arm_op[label] = _frac([bool(c.get("mech260_operative")) for c in cells]) if cells else 0.0
    pre_mech260 = all(per_dacc_arm_op.get(lbl, 0.0) >= MIN_FRACTION for lbl in DACC_ARM_LABELS)

    # PRE_NONDEGEN: selected-action-entropy metric is non-degenerate across P2 cells.
    entropy_vals = [c.get("selected_action_entropy", 0.0) for c in p2_cells]
    degen = check_degeneracy({"selected_action_entropy": entropy_vals}) if entropy_vals else {
        "non_degenerate": False, "degeneracy_reason": "no_p2_cells"}
    pre_nondegen = bool(degen.get("non_degenerate", False))

    all_met = bool(pre_reach and pre_zgoal and pre_mech260 and pre_nondegen)
    return {
        "preconditions_met": all_met,
        "pre_reach": pre_reach, "per_arm_reach_frac": {k: round(v, 4) for k, v in per_arm_reach.items()},
        "pre_zgoal": pre_zgoal, "zgoal_frac": round(_frac(zgoal_flags), 4),
        "pre_mech260": pre_mech260, "per_dacc_arm_operative_frac": {k: round(v, 4) for k, v in per_dacc_arm_op.items()},
        "pre_nondegen": pre_nondegen, "entropy_degeneracy": degen,
        "measured_steps_floor": MEASURED_STEPS_FLOOR,
        "z_goal_floor": Z_GOAL_FLOOR,
        "min_fraction": round(MIN_FRACTION, 4),
    }


def _mean_arm_entropy(rows: List[Dict[str, Any]], label: str) -> float:
    vals = [r["selected_action_entropy"] for r in rows
            if r["arm"] == label and r.get("reached_p2")]
    return _mean(vals)


def _evaluate_q045(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    e0 = _mean_arm_entropy(rows, "ARM_0_both_off")
    e1 = _mean_arm_entropy(rows, "ARM_1_mech313_only")
    e2 = _mean_arm_entropy(rows, "ARM_2_mech260_only")
    e3 = _mean_arm_entropy(rows, "ARM_3_both_on")

    both_beats_off = e3 > e0 + ENTROPY_MARGIN
    mutually_lb = (e3 > max(e1, e2) + ENTROPY_MARGIN) and (e1 > e0 + ENTROPY_MARGIN) and (e2 > e0 + ENTROPY_MARGIN)
    e1_beats_off = e1 > e0 + ENTROPY_MARGIN
    e2_beats_off = e2 > e0 + ENTROPY_MARGIN
    m313_dominates = (abs(e1 - e3) <= ENTROPY_MARGIN) and (e2 < e3 - ENTROPY_MARGIN) and e1_beats_off
    m260_dominates = (abs(e2 - e3) <= ENTROPY_MARGIN) and (e1 < e3 - ENTROPY_MARGIN) and e2_beats_off
    linear_sum = (e1 - e0) + (e2 - e0)
    both_delta = e3 - e0
    coupled = abs(both_delta - linear_sum) > ENTROPY_MARGIN  # super/sub-additive (R4 Tervo)

    return {
        "entropy_ARM_0_both_off": round(e0, 6),
        "entropy_ARM_1_mech313_only": round(e1, 6),
        "entropy_ARM_2_mech260_only": round(e2, 6),
        "entropy_ARM_3_both_on": round(e3, 6),
        "entropy_margin": ENTROPY_MARGIN,
        "both_beats_off": bool(both_beats_off),
        "mutually_load_bearing": bool(mutually_lb),
        "mech313_dominates": bool(m313_dominates),
        "mech260_dominates": bool(m260_dominates),
        "directionally_coupled": bool(coupled),
        "linear_sum_of_singleton_lifts": round(linear_sum, 6),
        "both_on_lift": round(both_delta, 6),
        "e1_beats_off": bool(e1_beats_off),
        "e2_beats_off": bool(e2_beats_off),
    }


def _interpret(pre: Dict[str, Any], grid: Dict[str, Any]) -> Dict[str, Any]:
    """Map (preconditions, grid) -> label + per-claim directions. Preconditions are the
    SAFETY NET: when unmet, self-route substrate_not_ready_requeue / non_contributory --
    NEVER a false weakens. A genuine weakens is reachable ONLY when all preconditions hold."""
    if not pre["preconditions_met"]:
        return {
            "label": "substrate_not_ready_requeue",
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "edpc": {c: "non_contributory" for c in CLAIM_IDS},
        }

    e1_off = grid["e1_beats_off"]
    e2_off = grid["e2_beats_off"]

    if not grid["both_beats_off"]:
        # Genuine falsification on a working substrate: both mechanisms produce no
        # behavioural diversity over baseline.
        return {
            "label": "fail_no_diversity",
            "outcome": "FAIL",
            "evidence_direction": "weakens",
            "edpc": {"Q-045": "weakens", "MECH-313": "weakens", "MECH-260": "weakens"},
        }

    if grid["mutually_load_bearing"]:
        return {
            "label": "mutually_load_bearing",
            "outcome": "PASS",
            "evidence_direction": "supports",
            "edpc": {"Q-045": "supports", "MECH-313": "supports", "MECH-260": "supports"},
        }

    if grid["mech313_dominates"]:
        return {
            "label": "mech313_dominates",
            "outcome": "PASS",
            "evidence_direction": "supports",
            "edpc": {"Q-045": "supports", "MECH-313": "supports", "MECH-260": "mixed"},
        }

    if grid["mech260_dominates"]:
        return {
            "label": "mech260_dominates",
            "outcome": "PASS",
            "evidence_direction": "supports",
            "edpc": {"Q-045": "supports", "MECH-313": "mixed", "MECH-260": "supports"},
        }

    if grid["directionally_coupled"]:
        # R4 lit-pull "DIRECTIONALLY COUPLED" 4th category (Tervo 2014); flags the
        # 8-cell follow-on. Neither pure-independent nor pure-collapse.
        return {
            "label": "directionally_coupled",
            "outcome": "PASS",
            "evidence_direction": "mixed",
            "edpc": {
                "Q-045": "mixed",
                "MECH-313": "supports" if e1_off else "mixed",
                "MECH-260": "supports" if e2_off else "mixed",
            },
        }

    # both-on beats off but no clean resolution.
    return {
        "label": "partial_lift",
        "outcome": "PASS",
        "evidence_direction": "mixed",
        "edpc": {
            "Q-045": "mixed",
            "MECH-313": "supports" if e1_off else "mixed",
            "MECH-260": "supports" if e2_off else "mixed",
        },
    }


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    global TOTAL_RUNS
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    TOTAL_RUNS = len(ARMS) * len(seeds)
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5
    else:
        total_eps = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET + P1_BUDGET

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            rows.append(_run_seed_arm(arm, seed, dry_run, total_eps))

    pre = _evaluate_preconditions(rows)
    grid = _evaluate_q045(rows)
    interp = _interpret(pre, grid)

    outcome = interp["outcome"]
    edpc = interp["edpc"]
    evidence_direction = interp["evidence_direction"]

    # Non-degeneracy net (applies to evidence runs): mark the run scoring-excluded if
    # the load-bearing diversity metric is structurally pinned.
    degen = pre["entropy_degeneracy"]
    non_degenerate = bool(degen.get("non_degenerate", False))

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": edpc,
        "interpretation_label": interp["label"],
        "dry_run": bool(dry_run),
        "sleep_driver_pattern": "N/A",
        "non_degenerate": non_degenerate,
        "preconditions": pre,
        "q045_grid": grid,
        "arm_results": rows,
        "eval_episodes": int(1 if dry_run else EVAL_EPISODES),
        "p2_steps_per_episode": int(30 if dry_run else P2_STEPS_PER_EPISODE),
        "fifo_warmup_steps": int(FIFO_WARMUP_STEPS),
        "noise_floor_alpha": NOISE_FLOOR_ALPHA,
        "dacc_suppression_weight": DACC_SUPPRESSION_WEIGHT,
        "dacc_suppression_memory": DACC_SUPPRESSION_MEMORY,
    }
    if not non_degenerate:
        manifest["degeneracy_reason"] = degen.get("degeneracy_reason", "entropy_metric_pinned")

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    if dry_run:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)
    else:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)

    print(f"Outcome: {outcome} interpretation_label={interp['label']}"
          f" preconditions_met={pre['preconditions_met']}", flush=True)
    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_experiment(dry_run=args.dry_run)
    if args.dry_run:
        sys.exit(0)
    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(result.get("manifest_path", "/dev/null")),
    )
