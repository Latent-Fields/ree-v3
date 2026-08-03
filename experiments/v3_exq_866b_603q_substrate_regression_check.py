"""
V3-EXQ-866b -- SUBSTRATE-REGRESSION DIAGNOSTIC: fresh, exact re-run of
V3-EXQ-603q's ARM_BASE_IA_ONLY cell on the CURRENT substrate, so the
47-day-old survival reference is finally COMMIT-VERIFIABLE.

experiment_purpose: diagnostic  (NOT scored evidence; excluded from confidence)
claim_ids: [SD-059, MECH-358]   (603q's own claims -- the config re-exercised here;
            tagged for discoverability, direction non_contributory throughout)
regression_check_of: V3-EXQ-603q  (does NOT supersede it)
motivated_by:        V3-EXQ-866a  (confirmed failure_autopsy_V3-EXQ-866a_2026-08-03)

WHY THIS RUNS (the routing output of failure_autopsy_V3-EXQ-866a_2026-08-03):
V3-EXQ-866a (INV-034/Q-021 goal-maintenance test) FAILED its G0 non-degeneracy
gate on the scaffolded_sd054_onboarding curriculum that 866's own autopsy had
prescribed. Its configuration was verified LINE-BY-LINE identical to the cited
reference V3-EXQ-603q's ARM_BASE_IA_ONLY (same seeds 42/43/44, same curriculum
order/budgets, same agent flags) -- yet its Stage-H hazard-avoidance survival
(median 4-5.5 / 200 steps) came in ~7x BELOW 603q's cited
base_mean_survival = 37.725. CRITICALLY, 603q's own manifest PREDATES the
Experimental Recording Standard (no top-level substrate_hash, no substrate_commit),
so the reference is COMMIT-UNVERIFIABLE across the 47-day gap between the two runs.
Substrate drift is PLAUSIBLE but NOT confirmed.

WHAT THIS DIAGNOSTIC DOES:
Re-runs 603q's EXACT ARM_BASE_IA_ONLY config (byte-identical _make_scaffold_cfg /
_make_config for the base arm -- bridge OFF, both 603k harm-pathway-training and
603j trained-safety-signal substrate fixes ON, same HARDER hazard regime, same
seeds) FRESH on the CURRENT substrate, WITH the recording standard's
substrate_hash + substrate_commit present this time (write_flat_manifest ->
stamp_recording_core). The comparison is finally commit-verifiable.

  PRIMARY (load-bearing) DV -- Stage-H survival regression:
    base_mean_survival on the current substrate vs the pinned 603q reference 37.725.
      >= SURVIVAL_REPRODUCE_FLOOR (0.5 * 37.725 ~= 18.86)
         -> substrate REPRODUCES the 603q reference. 866a's ~7x shortfall is then
            HARNESS-SPECIFIC (something in 866a's goal-maintenance harness differs),
            NOT substrate regression -- 603q's PASS stands.
      <  floor (esp. near 866a's ~5)
         -> substrate REGRESSION vs 603q CONFIRMED. The 47-day-old base survival
            does not replicate on the current substrate; 603q's PASS is thrown into
            question and 866a's shortfall is explained by drift, not harness.

  SECONDARY DV -- z_goal decay trajectory (localization, non-gating):
    866a observed z_goal decay from a healthy Stage-0 peak (~0.5) to a P2 mean of
    only 0.12 (C6 floor 0.4). This diagnostic snapshots ||z_goal|| at EVERY stage
    boundary (after stage0 / stage0b / p0 / hazard / p1 / p2) via _read_zgoal, so a
    reader can localise WHICH curriculum stage degrades goal maintenance. Recorded
    per seed + mean-across-seeds. Purely instrumental -- it does not gate the verdict.

  DIAGNOSTIC (localization, non-gating): per-seed harm_eval_range. 603q's base ran
    ~0.133; the 603i flat defect was ~0.002. A flat harm landscape on the current
    substrate would localise the regression to harm-pathway TRAINING; a healthy
    landscape with low survival localises it downstream (selection / E3 / avoidance).
    UNLIKE 603q, a flat harm landscape here is NOT a substrate_not_ready_requeue --
    it is part of the regression FINDING, so it is recorded, never used to requeue.

READINESS PRECONDITIONS (measurability guard, NOT a performance floor):
A low survival is the FINDING, so it must never be mislabelled substrate_not_ready.
The only conditions that make the measurement UNTAKEABLE (and therefore requeue) are:
  (1) the curriculum reached Stage-H (no abort before it) on >=2/3 seeds; and
  (2) z_goal FORMS at Stage-0 (peak > 0.4) on >=2/3 seeds -- the goal-formation
      positive control (866a confirmed Stage-0 z_goal forms healthy; if it does not
      form here, the goal machinery itself is broken and nothing downstream can be
      read). Below either -> FAIL substrate_not_ready_requeue, non_contributory.
The load-bearing survival criterion is genuinely EVALUATED (not starved) whenever
seeds reached Stage-H and produced episode lengths -- so a below-reference reading
is a real regression verdict, not a starved criterion.

MACHINE-CLASS NOTE: 603q ran on machine_class linux-x86_64-py3.10 (cloud). Stage-H
survival depends on multinomial action sampling, which diverges across machine
classes (see reference-cross-machine-class-contract-divergence). This diagnostic
MUST run on a cloud worker (linux-x86_64-py3.10) for the survival comparison to be
valid; a Mac (darwin-arm64) run is NOT comparable. machine_class is recorded so a
reader can discard a mis-routed run.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).
"""

from __future__ import annotations

import argparse
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
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _read_zgoal,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_866b_603q_substrate_regression_check"
QUEUE_ID = "V3-EXQ-866b"
CLAIM_IDS: List[str] = ["SD-059", "MECH-358"]  # 603q's claims -- config re-exercised
EXPERIMENT_PURPOSE = "diagnostic"

# The readiness preconditions here are structural completion / formation checks --
# curriculum reached Stage-H, Stage-0 z_goal formed (>0.4), Stage-H produced episode
# lengths -- each a fraction-of-seeds boolean that any HEALTHY run satisfies trivially
# (603q's base run satisfied all three: reached Stage-H 3/3, g0_frac 0.667, episode
# lengths on every seed). They are reachable BY CONSTRUCTION, not signature-reproduction
# anchors narrower than the state they gate (the 778d unmeetable-predicate hazard), so no
# assert_anchor_reachable guard is warranted -- the predicate IS the reachable definition.
ANCHOR_REACHABILITY_EXEMPT = (
    "readiness preconditions are structural completion/formation checks (reached Stage-H; "
    "Stage-0 z_goal formed; Stage-H produced episode lengths), reachable by construction on "
    "any healthy 603q-config run -- not signature-reproduction anchors narrower than their "
    "gated state; a below-floor reading here means the curriculum could not be measured, "
    "which is exactly the substrate_not_ready_requeue route."
)

SEEDS = [42, 43, 44]

# ============================================================================
# 603q REFERENCE (pinned constants from
# v3_exq_603q_sd059_mech358_escape_affordance_bridge_evidence_20260617T042830Z_v3.json,
# arm ARM_BASE_IA_ONLY). NOT derived from this run's own statistics.
# ============================================================================
REFERENCE_RUN_ID = (
    "v3_exq_603q_sd059_mech358_escape_affordance_bridge_evidence_20260617T042830Z_v3"
)
REFERENCE_BASE_MEAN_SURVIVAL = 37.725
REFERENCE_PER_SEED_MEAN_SURVIVAL = {42: 67.625, 43: 9.125, 44: 36.425}
# 603q's ARM_BASE_IA_ONLY arm_fingerprint substrate_hash (seeds 42/43; note: it
# FOLDS THE DRIVER into the hash, so it is NOT a pure ree_core fingerprint and the
# 47-day-later git commit it corresponds to is unknown -- there is no substrate_commit
# field on that manifest at all. This is exactly the un-verifiability this run fixes.)
REFERENCE_SUBSTRATE_HASH = "0e67292a9350dd2c0650eadb7ea3190217ba89e7a404926626e11cd6c3516a7f"
REFERENCE_MACHINE_CLASS = "linux-x86_64-py3.10"
# Pre-registered regression decision boundary (constant): reproduced if fresh base
# mean survival clears half the reference. Generous band -- 603q's per-seed survival
# itself ranged 9.1 to 67.6, so single-run variance is high; the two hypotheses
# (reproduce ~37.7 vs 866a's ~5) are cleanly separated by this midpoint.
SURVIVAL_REPRODUCE_FRAC = 0.5
SURVIVAL_REPRODUCE_FLOOR = REFERENCE_BASE_MEAN_SURVIVAL * SURVIVAL_REPRODUCE_FRAC

# ============================================================================
# 603q CONFIG CONSTANTS (copied byte-for-byte from 603q -- ARM_BASE_IA_ONLY path).
# ============================================================================
# Goal-pipeline / encoder dims (mirror 603i/603k/603l/603q exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Budgets (mirror 603q full budget).
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

# Isolated hazard-avoidance Stage-H (HARDER regime, exactly as 603q).
HAZARD_STAGE_BUDGET = 40
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

# SD-058 / MECH-357 protective-scaffold anneal (mirror 603q).
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# SD-059 / MECH-358 escape-affordance bridge knobs (mirror 603q; bridge OFF on base
# but these knobs are still set identically so _make_config is byte-identical).
ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1

# SUBSTRATE FIX 1 (603k): Stage-H harm-pathway training.
HARM_PATHWAY_LR = 1e-3
# SUBSTRATE FIX 3 (603q amend): harm-pathway training STABILIZATION.
HARM_PATHWAY_ENCODER_LR = 3e-4
HARM_PATHWAY_WARMUP_STEPS = 250
# SUBSTRATE FIX 2 (603j): trained safety-half threat-absence signal.
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

# Pre-registered gates (constants).
STAGE0_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
HARM_DISC_RANGE_FLOOR = 0.02  # 603i flat ~0.002; 603k trained ~0.133 (localization only)

# Single arm: 603q's ARM_BASE_IA_ONLY, exactly.
ARM = {"label": "ARM_BASE_IA_ONLY", "bridge": False, "relief": False,
       "safety": False, "nav_control": False}


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    """Byte-identical to 603q's _make_scaffold_cfg for the base arm (nav_control=False)."""
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS
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
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        # base arm: spawn NOT in reef (nav_control=False).
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
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


def _make_config(env) -> REEConfig:
    """Byte-identical to 603q's _make_config for the base arm (bridge OFF)."""
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
        # bridge OFF on the base arm.
        use_escape_affordance_bridge=False,
        use_escape_relief_credit=False,
        use_escape_safety_credit=False,
        escape_threat_floor=ESCAPE_THREAT_FLOOR,
        escape_threat_ref=ESCAPE_THREAT_REF,
        escape_approach_gain=ESCAPE_APPROACH_GAIN,
        escape_bias_scale=ESCAPE_BIAS_SCALE,
        escape_use_trained_safety_signal=True,
        escape_safety_signal_threshold=ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _config_slice(dry_run: bool) -> Dict[str, Any]:
    """Content-addressed config slice for the per-cell arm fingerprint.
    Mirrors 603q's ARM_BASE_IA_ONLY slice so the substrate portion is comparable."""
    return {
        "arm": ARM["label"],
        "use_escape_affordance_bridge": False,
        "use_escape_relief_credit": False,
        "use_escape_safety_credit": False,
        "nav_control_spawn_in_reef": False,
        "use_instrumental_avoidance": True,
        "scaffold_avoidance_driver_enabled": True,
        "use_pag_freeze_gate": True,
        "pag_theta_freeze": PAG_THETA_FREEZE,
        "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
        "avoidance_threat_ref": AVOIDANCE_THREAT_REF,
        "escape_threat_floor": ESCAPE_THREAT_FLOOR,
        "escape_threat_ref": ESCAPE_THREAT_REF,
        "escape_approach_gain": ESCAPE_APPROACH_GAIN,
        "escape_bias_scale": ESCAPE_BIAS_SCALE,
        "feed_harm_stream": True,
        "e2_action_contrastive_enabled": True,
        "scaffold_train_harm_pathway": True,
        "harm_pathway_lr": HARM_PATHWAY_LR,
        "use_harm_stream": True,
        "use_e2_harm_s_forward": True,
        "escape_use_trained_safety_signal": True,
        "escape_safety_signal_threshold": ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        "use_contextual_safety_terrain": True,
        "use_conditioned_safety_store": True,
        "use_suffering_derivative_comparator": True,
        "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, P2_BUDGET, TRAIN_STEPS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "dry_run": bool(dry_run),
    }


def _auc_survival(ep_lengths: List[int]) -> float:
    if not ep_lengths:
        return 0.0
    return float(sum(ep_lengths)) / float(len(ep_lengths) * TRAIN_STEPS)


def _time_to_first_death(ep_lengths: List[int]) -> int:
    for i, ln in enumerate(ep_lengths):
        if ln < TRAIN_STEPS:
            return i + 1
    return len(ep_lengths) + 1


def _zg_norm(agent) -> float:
    """Live ||z_goal|| snapshot at a stage boundary (0.0 if no goal_state)."""
    return float(_read_zgoal(getattr(agent, "goal_state", None))[0])


def _aborted_record(seed: int, stage: str, reason: str,
                    s0_peak: float, traj: Dict[str, float]) -> Dict[str, Any]:
    return {
        "arm": ARM["label"], "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "hazard_stage_mean_episode_length": 0.0,
        "hazard_stage_median_last_window": 0.0,
        "hazard_stage_auc_survival": 0.0,
        "hazard_stage_time_to_first_death": 0,
        "hazard_stage_episode_lengths": [],
        "hazard_stage_survival_pass": False,
        "hazard_eval_range": 0.0,
        "pag_n_commits": 0,
        "pag_n_releases": 0,
        "p2_contact_rate": 0.0,
        "p2_z_goal_norm_at_contact_peak": 0.0,
        "g0_stage0_zgoal": bool(s0_peak > STAGE0_ZGOAL_GATE),
        "g_h_hazard_survival": False,
        "z_goal_trajectory": traj,
        "reached_hazard_stage": stage not in ("stage0", "stage0b", "p0"),
        "reached_p2": False,
        "seed_pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int,
              zg_acc: ZGoalStreamAccumulator) -> Dict[str, Any]:
    """Full 603q curriculum for the ARM_BASE_IA_ONLY cell of one seed. arm_cell resets
    all RNG on enter (order-independent) and stamps the fingerprint on the row."""
    with arm_cell(
        seed,
        config_slice=_config_slice(dry_run),
        script_path=Path(__file__),
        config_slice_declared=True,
    ) as cell:
        scaffold_cfg = _make_scaffold_cfg(dry_run)
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env)).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

        print(f"Seed {seed} Condition {ARM['label']}", flush=True)

        def _pag_commits() -> int:
            p = getattr(agent, "pag_freeze_gate", None)
            return int(dict(p.diagnostics).get("n_commits", 0)) if p is not None else 0

        def _pag_releases() -> int:
            p = getattr(agent, "pag_freeze_gate", None)
            return int(dict(p.diagnostics).get("n_releases", 0)) if p is not None else 0

        # z_goal decay trajectory (secondary DV): ||z_goal|| at each stage boundary.
        traj: Dict[str, float] = {}

        # Stage 0 -- forced-benefit nursery (goal-formation positive control).
        s0 = scheduler.run_stage0_nursery(agent, device)
        done = s0.n_episodes
        traj["after_stage0"] = _zg_norm(agent)
        traj["stage0_peak"] = float(s0.z_goal_norm_peak)
        print(f"  [train] stage0 {ARM['label']} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f} zg_now={traj['after_stage0']:.4f}",
              flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={ARM['label']} aborted_at=stage0", flush=True)
            rec = _aborted_record(seed, "stage0", s0.abort_reason, s0.z_goal_norm_peak, traj)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        traj["after_stage0b"] = _zg_norm(agent)
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={ARM['label']} aborted_at=stage0b", flush=True)
            rec = _aborted_record(seed, "stage0b", s0b.abort_reason, s0.z_goal_norm_peak, traj)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec

        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        traj["after_p0"] = _zg_norm(agent)
        print(f"  [train] p0 {ARM['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}"
              f" zg_now={traj['after_p0']:.4f}", flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={ARM['label']} aborted_at=p0", flush=True)
            rec = _aborted_record(seed, "p0", p0.abort_reason, s0.z_goal_norm_peak, traj)
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec

        # Stage-H -- ISOLATED HAZARD-AVOIDANCE (the survival readout under scrutiny).
        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        traj["after_hazard"] = _zg_norm(agent)
        pag_commits = _pag_commits()
        pag_releases = _pag_releases()
        harm_disc = dict(hz.harm_discriminativeness or {})
        harm_eval_range = float(harm_disc.get("harm_eval_range", 0.0))
        ep_lengths = list(hz.episode_lengths or [])
        auc_surv = _auc_survival(ep_lengths)
        ttfd = _time_to_first_death(ep_lengths)
        print(f"  [train] hazard_avoidance {ARM['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f}"
              f" median_last={hz.median_last_window_episode_length:.1f}"
              f" auc={auc_surv:.3f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
              f" harm_range={harm_eval_range:.4f} pag_commits={pag_commits}"
              f" zg_now={traj['after_hazard']:.4f}", flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={ARM['label']} aborted_at=hazard", flush=True)
            rec = _aborted_record(seed, "hazard", hz.abort_reason, s0.z_goal_norm_peak, traj)
            rec["hazard_eval_range"] = harm_eval_range
            rec["harm_discriminativeness"] = harm_disc
            rec["pag_n_commits"] = pag_commits
            rec["pag_n_releases"] = pag_releases
            zg_acc.observe(agent)
            cell.stamp(rec)
            return rec

        # P1 -- combined wean.
        p1 = scheduler.run_p1(agent, device)
        done += p1.n_episodes
        traj["after_p1"] = _zg_norm(agent)
        print(f"  [train] p1 {ARM['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={p1.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
              f" zg_now={traj['after_p1']:.4f}", flush=True)

        # P2 -- frozen-policy guarded measurement.
        p2 = scheduler.run_p2(agent, device)
        done += p2.n_episodes
        traj["after_p2"] = _zg_norm(agent)
        traj["p2_contact_peak"] = float(p2.z_goal_norm_at_contact_peak)
        print(f"  [train] p2 {ARM['label']} seed={seed} ep {done}/{total_eps}"
              f" contact_rate={p2.contact_rate:.4f}"
              f" zg_contact_peak={p2.z_goal_norm_at_contact_peak:.4f}"
              f" zg_now={traj['after_p2']:.4f}", flush=True)

        g0 = bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE)
        g_h = bool(hz.survival_gate_passed)
        seed_pass = bool(g_h)
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={ARM['label']}"
              f" mean_surv={hz.mean_episode_length:.1f} g_h={g_h} g0={g0}"
              f" harm_range={harm_eval_range:.4f}"
              f" ref_base_mean={REFERENCE_BASE_MEAN_SURVIVAL:.1f}", flush=True)

        rec = {
            "arm": ARM["label"],
            "seed": seed,
            "aborted_at": None,
            "abort_reason": "",
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "p0_mean_episode_length": float(p0.mean_episode_length),
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            "hazard_stage_auc_survival": float(auc_surv),
            "hazard_stage_time_to_first_death": int(ttfd),
            "hazard_stage_episode_lengths": ep_lengths,
            "hazard_stage_survival_pass": g_h,
            "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
            "hazard_stage_n_episodes": int(hz.n_episodes),
            "hazard_eval_range": harm_eval_range,
            "harm_discriminativeness": harm_disc,
            "pag_n_commits": int(pag_commits),
            "pag_n_releases": int(pag_releases),
            "p1_survival_pass": bool(p1.survival_gate_passed),
            "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
            "p2_contact_rate": float(p2.contact_rate),
            "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
            "g0_stage0_zgoal": g0,
            "g_h_hazard_survival": g_h,
            "z_goal_trajectory": traj,
            "reached_hazard_stage": True,
            "reached_p2": True,
            "seed_pass": seed_pass,
        }
        zg_acc.observe(agent)
        cell.stamp(rec)
        return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _mean(vals: List[float]) -> float:
    return float(sum(vals)) / float(len(vals)) if vals else 0.0


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

    zg_acc = ZGoalStreamAccumulator()
    rows = [_run_seed(s, dry_run, total_eps, zg_acc) for s in seeds]

    per_seed_mean_surv = [float(r.get("hazard_stage_mean_episode_length", 0.0)) for r in rows]
    per_seed_auc = [float(r.get("hazard_stage_auc_survival", 0.0)) for r in rows]
    per_seed_ttfd = [int(r.get("hazard_stage_time_to_first_death", 0)) for r in rows]
    per_seed_harm_range = [float(r.get("hazard_eval_range", 0.0)) for r in rows]
    per_seed_stage0_peak = [float(r.get("stage0_z_goal_norm_peak", 0.0)) for r in rows]
    per_seed_pag_commits = [int(r.get("pag_n_commits", 0)) for r in rows]

    base_mean_survival = _mean(per_seed_mean_surv)

    # --- readiness (measurability guard, NOT a performance floor) ---
    reached_hazard_flags = [bool(r.get("reached_hazard_stage", False)) for r in rows]
    reached_hazard_frac = _frac(reached_hazard_flags)
    g0_flags = [bool(r.get("g0_stage0_zgoal", False)) for r in rows]
    g0_frac = _frac(g0_flags)
    # the survival number is genuinely evaluated (not starved) iff seeds reached
    # Stage-H AND produced episode lengths.
    survival_measured_flags = [
        bool(r.get("reached_hazard_stage", False)) and len(r.get("hazard_stage_episode_lengths", [])) > 0
        for r in rows
    ]
    survival_measured_frac = _frac(survival_measured_flags)

    curriculum_reached_hazard = bool(reached_hazard_frac >= MIN_FRACTION)
    stage0_zgoal_forms = bool(g0_frac >= MIN_FRACTION)
    readiness_met = bool(curriculum_reached_hazard and stage0_zgoal_forms)

    # --- primary (load-bearing) regression decision ---
    reproduces = bool(base_mean_survival >= SURVIVAL_REPRODUCE_FLOOR)
    survival_ratio_vs_ref = (
        base_mean_survival / REFERENCE_BASE_MEAN_SURVIVAL
        if REFERENCE_BASE_MEAN_SURVIVAL > 1e-9 else 0.0
    )

    # --- localization diagnostics (non-gating) ---
    harm_disc_flags = [hr >= HARM_DISC_RANGE_FLOOR for hr in per_seed_harm_range]
    harm_landscape_discriminative_frac = _frac(harm_disc_flags)
    # z_goal decay trajectory: mean-across-seeds ||z_goal|| at each stage boundary.
    traj_keys = ["stage0_peak", "after_stage0", "after_stage0b", "after_p0",
                 "after_hazard", "after_p1", "after_p2", "p2_contact_peak"]
    z_goal_trajectory_per_seed = [
        {"seed": r["seed"], **{k: float((r.get("z_goal_trajectory") or {}).get(k, 0.0))
                               for k in traj_keys}}
        for r in rows
    ]
    z_goal_trajectory_mean = {
        k: _mean([float((r.get("z_goal_trajectory") or {}).get(k, 0.0)) for r in rows])
        for k in traj_keys
    }

    # --- route ---
    if not readiness_met:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
    elif reproduces:
        outcome = "PASS"
        label = "substrate_reproduces_603q_reference"
    else:
        outcome = "FAIL"
        label = "substrate_regression_confirmed_vs_603q"

    run_direction = "non_contributory"  # diagnostic -- never scored
    evidence_direction_per_claim = {cid: run_direction for cid in CLAIM_IDS}

    preconditions = [
        {
            "name": "curriculum_reached_hazard_stage",
            "kind": "readiness",
            "description": "The curriculum must reach Stage-H (no abort before it) on "
                           ">=2/3 seeds, else the survival number cannot be measured at all "
                           "-> substrate_not_ready_requeue (a measurability failure, NOT a "
                           "regression finding). A low-but-measured survival is a genuine "
                           "regression verdict and does NOT trip this.",
            "control": "603q ARM_BASE_IA_ONLY full curriculum run to Stage-H.",
            "measured": float(reached_hazard_frac),
            "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(curriculum_reached_hazard),
        },
        {
            "name": "stage0_forced_feed_lights_zgoal_on_base",
            "kind": "readiness",
            "description": "Stage-0 forced supra-threshold benefit lights z_goal (>0.4) on "
                           ">=2/3 seeds -- the goal-FORMATION positive control. 866a confirmed "
                           "healthy Stage-0 z_goal formation; if z_goal does not even form here, "
                           "the goal machinery itself is broken and nothing downstream (survival, "
                           "the z_goal decay trajectory) can be read -> substrate_not_ready_requeue.",
            "control": "run_stage0_nursery forced-feed.",
            "measured": float(g0_frac),
            "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(stage0_zgoal_forms),
        },
        {
            "name": "survival_measured_non_starved",
            "kind": "readiness",
            "description": "The load-bearing survival criterion is genuinely EVALUATED (not "
                           "starved) iff seeds reached Stage-H AND produced episode lengths. When "
                           "this holds, a below-reference base_mean_survival is a real regression "
                           "verdict, not a starved criterion -- this is the same-statistic "
                           "starvation guard for the survival-routed load-bearing criterion.",
            "control": "Stage-H produced >=1 episode length on >=2/3 seeds.",
            "measured": float(survival_measured_frac),
            "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(survival_measured_frac >= MIN_FRACTION),
        },
    ]
    criteria_non_degenerate = {
        "arms_reached_hazard_stage": bool(reached_hazard_frac >= MIN_FRACTION),
        "survival_measured_non_degenerate": bool(survival_measured_frac >= MIN_FRACTION),
        "stage0_zgoal_formed": bool(g0_frac >= MIN_FRACTION),
        "harm_landscape_recorded": bool(any(hr > 0.0 for hr in per_seed_harm_range)),
    }
    criteria = [
        {"name": "base_survival_reproduces_603q_reference", "load_bearing": True,
         "passed": bool(reproduces)},
    ]

    print(
        f"[{EXPERIMENT_TYPE}] base_mean_survival={base_mean_survival:.2f}"
        f" per_seed={per_seed_mean_surv}"
        f" | ref={REFERENCE_BASE_MEAN_SURVIVAL:.2f} floor={SURVIVAL_REPRODUCE_FLOOR:.2f}"
        f" ratio={survival_ratio_vs_ref:.2f}x"
        f" | reproduces={reproduces} readiness_met={readiness_met}"
        f" -> outcome={outcome} label={label}"
        f" | z_goal stage0_peak={z_goal_trajectory_mean['stage0_peak']:.3f}"
        f" -> p2={z_goal_trajectory_mean['after_p2']:.3f}"
        f" contact_peak={z_goal_trajectory_mean['p2_contact_peak']:.3f}"
        f" | harm_disc_frac={harm_landscape_discriminative_frac:.2f}",
        flush=True,
    )

    arm_results = [{
        "arm": ARM["label"],
        "use_escape_affordance_bridge": False,
        "use_escape_relief_credit": False,
        "use_escape_safety_credit": False,
        "nav_control": False,
        "mean_survival": base_mean_survival,
        "auc_survival": _mean(per_seed_auc),
        "per_seed_mean_survival": per_seed_mean_surv,
        "per_seed_auc_survival": per_seed_auc,
        "per_seed_time_to_first_death": per_seed_ttfd,
        "per_seed_stage0_z_goal_norm_peak": per_seed_stage0_peak,
        "per_seed_harm_eval_range": per_seed_harm_range,
        "per_seed_pag_n_commits": per_seed_pag_commits,
        "per_seed_hazard_median_last_window": [
            float(r.get("hazard_stage_median_last_window", 0.0)) for r in rows
        ],
        "g_h_frac": _frac([bool(r.get("g_h_hazard_survival", False)) for r in rows]),
        "g0_frac": g0_frac,
        "harm_disc_frac": harm_landscape_discriminative_frac,
        "arm_fingerprint": [r.get("arm_fingerprint") for r in rows],
    }]

    return {
        "outcome": outcome,
        "evidence_direction": run_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        # --- primary regression readouts ---
        "base_mean_survival": base_mean_survival,
        "per_seed_mean_survival": per_seed_mean_surv,
        "reference_base_mean_survival": REFERENCE_BASE_MEAN_SURVIVAL,
        "survival_ratio_vs_reference": survival_ratio_vs_ref,
        "survival_reproduce_floor": SURVIVAL_REPRODUCE_FLOOR,
        "substrate_reproduces_reference": reproduces,
        "readiness_met": readiness_met,
        # --- secondary DV: z_goal decay trajectory (localization) ---
        "z_goal_trajectory_per_seed": z_goal_trajectory_per_seed,
        "z_goal_trajectory_mean": z_goal_trajectory_mean,
        # --- localization diagnostics ---
        "per_seed_harm_eval_range": per_seed_harm_range,
        "harm_landscape_discriminative_frac": harm_landscape_discriminative_frac,
        "per_seed_stage0_z_goal_norm_peak": per_seed_stage0_peak,
        "per_seed_pag_n_commits": per_seed_pag_commits,
        "reference_603q": {
            "source_run_id": REFERENCE_RUN_ID,
            "base_mean_survival": REFERENCE_BASE_MEAN_SURVIVAL,
            "per_seed_mean_survival": REFERENCE_PER_SEED_MEAN_SURVIVAL,
            "substrate_hash_driver_inclusive": REFERENCE_SUBSTRATE_HASH,
            "substrate_commit": None,
            "machine_class": REFERENCE_MACHINE_CLASS,
            "note": "603q predates the Experimental Recording Standard -- no top-level "
                    "substrate_hash and no substrate_commit. Its arm_fingerprint "
                    "substrate_hash folds the driver script, so it is not a pure ree_core "
                    "fingerprint and its git commit is unknown. This diagnostic's run "
                    "carries substrate_commit so IT is commit-verifiable going forward.",
        },
        "acceptance": {
            "pass_rule": "readiness_met (curriculum reaches Stage-H AND Stage-0 z_goal "
                         "forms, both >=2/3 seeds) AND base_mean_survival >= "
                         "REFERENCE_BASE_MEAN_SURVIVAL * SURVIVAL_REPRODUCE_FRAC (0.5). "
                         "PASS = reproduces 603q reference (866a shortfall is harness-specific). "
                         "FAIL+substrate_regression_confirmed_vs_603q = survival regressed. "
                         "FAIL+substrate_not_ready_requeue = curriculum could not be measured.",
            "reference_base_mean_survival": REFERENCE_BASE_MEAN_SURVIVAL,
            "survival_reproduce_frac": SURVIVAL_REPRODUCE_FRAC,
            "survival_reproduce_floor": SURVIVAL_REPRODUCE_FLOOR,
            "min_fraction": MIN_FRACTION,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "train_steps_ceiling": TRAIN_STEPS,
        },
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "grid": {
                "reproduces": "base survival >= 0.5*ref -> substrate stable; 866a's ~7x "
                              "shortfall is HARNESS-specific, not substrate regression; "
                              "603q's PASS stands.",
                "regression_confirmed": "base survival < 0.5*ref (near 866a's ~5) -> "
                                        "substrate REGRESSED over the 47-day gap; 603q's PASS "
                                        "is in question; 866a's shortfall is explained by drift. "
                                        "Localise via z_goal_trajectory_mean (which stage decays "
                                        "goal maintenance) + per_seed_harm_eval_range (flat -> "
                                        "harm-pathway training regressed; healthy -> downstream).",
                "not_ready": "curriculum aborted before Stage-H OR Stage-0 z_goal never forms "
                             "-> the measurement could not be taken -> requeue, NOT a verdict.",
            },
        },
        "per_seed": rows,
        "_zg_stream_stats": zg_acc.stats(),
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

    full_config = {
        "budgets": {
            "stage0": STAGE0_BUDGET, "stage0b": STAGE0B_BUDGET, "p0": P0_BUDGET,
            "hazard": HAZARD_STAGE_BUDGET, "p1": P1_BUDGET, "p2": P2_BUDGET,
            "steps_per_episode": TRAIN_STEPS,
        },
        "hazard_stage": {
            "num_hazards": HAZARD_STAGE_NUM_HAZARDS,
            "num_resources": HAZARD_STAGE_NUM_RESOURCES,
            "hazard_food_attraction": HAZARD_STAGE_HFA,
            "proximity_harm_scale": HAZARD_STAGE_PROXIMITY_HARM,
            "survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "stability_window": HAZARD_STAGE_STABILITY_WINDOW,
        },
        "substrate_fixes": {
            "harm_pathway_lr": HARM_PATHWAY_LR,
            "harm_pathway_encoder_lr": HARM_PATHWAY_ENCODER_LR,
            "harm_pathway_warmup_steps": HARM_PATHWAY_WARMUP_STEPS,
            "escape_safety_signal_threshold": ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        },
        "encoder_dims": {
            "world_dim": WORLD_DIM, "harm_a_dim": HARM_A_DIM,
            "harm_obs_a_dim": HARM_OBS_A_DIM, "harm_history_len": HARM_HISTORY_LEN,
            "drive_weight": DRIVE_WEIGHT, "alpha_world": 0.9,
        },
        "seeding": {
            "seed_gain": SEED_GAIN, "benefit_threshold": SEED_BENEFIT_THRESHOLD,
            "drive_floor": SEED_DRIVE_FLOOR, "n_resource_types": N_RESOURCE_TYPES,
        },
        "regression_decision": {
            "reference_base_mean_survival": REFERENCE_BASE_MEAN_SURVIVAL,
            "survival_reproduce_frac": SURVIVAL_REPRODUCE_FRAC,
            "survival_reproduce_floor": SURVIVAL_REPRODUCE_FLOOR,
        },
        "arm": ARM["label"],
    }

    zg_stream_stats = result.pop("_zg_stream_stats", None)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",
        "regression_check_of": "V3-EXQ-603q",
        "motivated_by": "V3-EXQ-866a",
        "depends_on": ["V3-EXQ-603q", "V3-EXQ-866a"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "SD-059 / MECH-358 scaffolded_sd054_onboarding ARM_BASE_IA_ONLY "
                     "(603k harm-pathway-trained + 603j trained-safety-signal, HARDER hazard "
                     "regime 6 hazards / proximity_harm 0.10) -- re-run fresh for a "
                     "substrate-regression check vs V3-EXQ-603q.",
        "design_note": "SUBSTRATE-REGRESSION DIAGNOSTIC per failure_autopsy_V3-EXQ-866a_2026-08-03. "
                       "Re-runs 603q's ARM_BASE_IA_ONLY config byte-identical (_make_scaffold_cfg / "
                       "_make_config copied verbatim for the base arm; same seeds 42/43/44) FRESH on "
                       "the current substrate WITH substrate_hash + substrate_commit present, so the "
                       "47-day-old base_mean_survival=37.725 reference is finally commit-verifiable. "
                       "PRIMARY DV: base survival vs 37.725 (>=0.5*ref -> reproduces / harness-specific "
                       "866a shortfall; <floor -> regression). SECONDARY DV: per-stage z_goal decay "
                       "trajectory (localize the 866a Stage-0 ~0.5 -> P2 ~0.12 decay). Single arm, "
                       "diagnostic purpose, non_contributory (does NOT supersede 603q).",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_load_bearing": "base_mean_survival >= 37.725 * 0.5 (reproduces 603q)",
            "reference_base_mean_survival": REFERENCE_BASE_MEAN_SURVIVAL,
            "survival_reproduce_frac": SURVIVAL_REPRODUCE_FRAC,
            "secondary_dv": "per-stage z_goal decay trajectory (localization, non-gating)",
            "localization_diagnostic": "per_seed_harm_eval_range (flat -> harm-pathway "
                                       "training regressed; healthy -> downstream)",
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "min_fraction": MIN_FRACTION,
        },
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        z_goal_stream_stats=zg_stream_stats,
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']} label: {manifest['interpretation']['label']}",
          flush=True)
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
            dry_run=bool(args.dry_run),
        )
