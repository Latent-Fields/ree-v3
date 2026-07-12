"""
V3-EXQ-634c -- scaffolded_sd054_onboarding NURSERY + SEEDING-CALIBRATION
substrate-readiness diagnostic (4-arm seeding-lever sweep, 2026-06-03c).

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
Validate the 2026-06-03c seeding-calibration amend to scaffolded_sd054_onboarding.
The predecessor V3-EXQ-634b VALIDATED the consolidation half (G0 Stage-0 forced-feed
PASS 3/3, G0b Stage-0b retention PASS 3/3, n_decay_only_updates=0 everywhere) but
isolated a benefit-magnitude / threshold mismatch: natural wild benefit
(obs_body[11] ~0.03) never clears GoalState.update's firing threshold
  effective_benefit = benefit * z_goal_seeding_gain(1.0)
                      * (1 + drive_weight(2.0) * drive_trace) > benefit_threshold(0.1),
so the contact-gating band (1e-6, ~0.1-effective) was NOT skipped yet did NOT seed --
it only applied the 0.5%/step decay (goal.py:173), DECAYING the consolidated trace
during real foraging. 634b seed 43 (475 P2 contact-refresh calls, contact_rate 0.348)
collapsed z_goal to ~4.5e-05 while non-foraging seed 42 "passed" G3 by carrying the
untouched forced-feed nursery trace (0.4398, byte-identical to Stage-0b-end). G3-at-
frozen-peak was thus anti-correlated with foraging.

This re-validation sweeps the GoalConfig seeding levers (autopsy: "one or a
combination, pick via a small sweep") so genuine wild contact clears the firing
threshold, and reads G3 at GENUINE CONSUMPTION EVENTS (632-style) rather than the
forced-feed-calibrated frozen peak. For every seeding-ON arm the contact-gating
threshold is matched to the arm's seeding floor so sub-seeding whiffs are PROTECTED
(skipped) rather than decay-only eroded. P0/P1 budgets are strengthened vs 634b to
address the still-open foraging-competence half (G1 1/3 in 634b).

ARMS (3 seeds 42/43/44 each; same env/dims/flags as 634b so it is representative of
the eventual 603f re-issue). Seeding floor b* = benefit_threshold / (gain * (1 +
drive_weight * drive_floor)); contact-gating threshold is set to b* on seeding arms.
  ARM_0_baseline           -- amend OFF (gain/benefit_threshold/drive_floor default,
                              gating sentinel -1 -> readout fallback). 634b-equivalent
                              CONTROL / regression guard: must reproduce
                              substrate_not_engaged (wild 0.03 < 0.1 -> no seeding).
  ARM_1_drivefloor         -- drive_floor=0.9 only. effective_benefit >= benefit*2.8;
                              b*=0.1/2.8=0.0357 -> wild 0.03 JUST-misses 0.0357
                              (informative threshold-boundary arm; not expected to pass).
  ARM_2_drivefloor_gain    -- drive_floor=0.9 + z_goal_seeding_gain=1.5. b*=0.0238 ->
                              wild 0.03 > b* -> SEEDS.
  ARM_3_drivefloor_lowthr  -- drive_floor=0.9 + benefit_threshold=0.02. b*=0.00714 ->
                              wild 0.03 > b* -> SEEDS.

PRE-REGISTERED SUBSTRATE GATES (each requires >= 2/3 of seeds; do NOT retune):
  G0  stage0_positive_control : Stage-0 z_goal_norm_peak > 0.4 (forced-feed lights z_goal)
  G0b stage0b_retention       : Stage-0b retention_ratio >= 0.75 (consolidation protects trace)
  G1  p1_survival             : P1 survival/foraging gate passed
  G2  p2_contact              : P2 contact_rate > 0 (infant actually fed at measurement)
  G3  p2_zgoal_at_contact     : P2 z_goal_norm_AT_CONTACT_peak > 0.4 -- the FAIR readout
                                (z_goal read at a genuine seeding event, num_contact_events>0;
                                NOT the frozen z_goal_norm_peak_max that seed 42 passed via
                                the carried nursery trace).
ARM PASS = G0 AND G0b AND G1 AND G2 AND G3 (each >= 2/3 seeds) AND num_contact_events>0.
EXPERIMENT PASS = ANY seeding-ON arm passes.

INTERPRETATION ON OUTCOME:
  PASS (a seeding arm clears all gates) -> substrate ready: (a follow-on /governance +
        /queue-experiment action, NOT automatic, NOT in this script) flip
        substrate_queue scaffolded_sd054_onboarding ready=true + queue V3-EXQ-603f.
  ALL seeding arms FAIL G3-at-contact DESPITE G2 contact -> "fed_but_no_goal_beyond_magnitude"
        (goal-formation issue not explained by seeding magnitude) -> /failure-autopsy.
  G1 survival < 2/3 across all arms -> "foraging_competence_open" -> strengthen P0/P1.
  ARM_0 NOT reproducing substrate_not_engaged -> regression in the baseline control.
  In every FAIL case this is diagnostic, weights no claim, and 603f STAYS BLOCKED.

DIAGNOSTICS: with the matched gating floor, P1/P2 n_decay_only_updates should be ~0 on
seeding arms; n_skipped_protected counts the sub-seeding whiffs protected; n_contact_refresh
counts genuine seeding refreshes. z_goal_norm_peak_max (frozen) is ALSO reported per seed so
the frozen-vs-consumption-gated divergence is visible in the manifest.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking goal-pipeline
onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; not governance evidence)
predecessor (NOT supersedes): V3-EXQ-634b (the consolidation-validated nursery run;
  634b lacks the seeding-calibration levers -- this run sweeps them).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    classify_interpretation_branch,
    evaluate_substrate_gate,
    stage_plan,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_634c_scaffolded_nursery_seeding_calibration_readiness"
QUEUE_ID = "V3-EXQ-634c"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror V3-EXQ-634b _make_config so this run is
# representative of the env+agent the 603f re-issue will use).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0  # SD-012 amplification (mirrors _make_config); used for seeding-floor math

# Strengthened scaffold curriculum (vs 634b P0/P1=100/50): lengthen P0/P1 so >=2/3
# seeds reach self-sustaining consumption (the still-open G1 foraging-competence half).
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 120                # strengthened (634b: 100)
P1_BUDGET = 70                 # strengthened (634b: 50)
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3         # staged-withdrawal P1 schedule (as 634b)
P0_NUM_HAZARDS = 1             # reduced early hazard pressure (as 634b)
P2_HFA_GUARD = 0.3            # P2 measurement guard (as 634b)

# Pre-registered gates (constants; NOT derived from the run's own statistics).
STAGE0_ZGOAL_GATE = 0.4
STAGE0B_RETENTION_GATE = 0.75
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0

# Seeding-lever sweep. gain/benefit_threshold/drive_floor are GoalConfig knobs the
# scaffold propagates; None = leave the GoalConfig default (gain 1.0 / threshold 0.1 /
# floor 0.0). seeding=False is the bit-identical 634b baseline (gating sentinel -1).
ARMS: List[Dict[str, Any]] = [
    {"name": "ARM_0_baseline", "seeding": False,
     "gain": None, "benefit_threshold": None, "drive_floor": None},
    {"name": "ARM_1_drivefloor", "seeding": True,
     "gain": None, "benefit_threshold": None, "drive_floor": 0.9},
    {"name": "ARM_2_drivefloor_gain", "seeding": True,
     "gain": 1.5, "benefit_threshold": None, "drive_floor": 0.9},
    {"name": "ARM_3_drivefloor_lowthr", "seeding": True,
     "gain": None, "benefit_threshold": 0.02, "drive_floor": 0.9},
]


def _seeding_floor(gain: Optional[float], benefit_threshold: Optional[float],
                   drive_floor: Optional[float]) -> float:
    """Benefit b* at which effective_benefit == benefit_threshold under the arm's
    GoalConfig at the drive floor: b* = benefit_threshold / (gain * (1 + drive_weight
    * drive_floor)). Anything below b* cannot seed (protected); anything at/above
    seeds. Matched gating floor for the contact-gating skip decision."""
    g = float(gain) if gain is not None else 1.0
    bt = float(benefit_threshold) if benefit_threshold is not None else 0.1
    df = float(drive_floor) if drive_floor is not None else 0.0
    mult = 1.0 + DRIVE_WEIGHT * df
    return bt / (g * mult)


def _make_config(env: CausalGridWorldV2) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # Two-part-fix precondition (603e): z_goal_enabled creates GoalState;
        # drive_weight=2.0 is the SD-012 amplification the reference V3-EXQ-622 uses.
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
    )


def _make_scaffold_cfg(dry_run: bool, arm: Dict[str, Any]) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, p1, p2, steps = 2, 2, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, P1_BUDGET, P2_BUDGET, TRAIN_STEPS
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
        # --- developmental-window / consolidation amend (ON; preserve 634b mechanism) ---
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
    )
    # --- 2026-06-03c seeding-calibration amend (per-arm) ---
    if arm["seeding"]:
        if arm["gain"] is not None:
            cfg.scaffold_z_goal_seeding_gain = float(arm["gain"])
        if arm["benefit_threshold"] is not None:
            cfg.scaffold_benefit_threshold = float(arm["benefit_threshold"])
        if arm["drive_floor"] is not None:
            cfg.scaffold_drive_floor = float(arm["drive_floor"])
        # Match the contact-gating floor to the arm's seeding floor so sub-seeding
        # whiffs are protected (skipped), not decay-only eroded.
        cfg.scaffold_contact_gating_benefit_threshold = _seeding_floor(
            arm["gain"], arm["benefit_threshold"], arm["drive_floor"]
        )
    # ARM_0 (not seeding): gating sentinel stays -1.0 -> readout fallback (bit-identical 634b).
    # Dry-run: scale the P1 survival gate so short episodes can clear it.
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
    return cfg


def _aborted_seed_record(seed: int, arm_name: str, stage: str, reason: str,
                         s0_peak: float = 0.0, s0_benefit: float = 0.0,
                         s0b_retention: float = 0.0, s0b_pass: bool = False) -> Dict[str, Any]:
    return {
        "seed": seed, "arm": arm_name, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "stage0_benefit_exposure": float(s0_benefit),
        "stage0b_retention_ratio": float(s0b_retention),
        "stage0b_retention_gate_passed": bool(s0b_pass),
        "p1_survival_pass": False,
        "p2_contact_rate": 0.0,
        "p2_z_goal_norm_peak": 0.0,
        "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "seed_pass": False,
    }


def _run_seed(seed: int, dry_run: bool, arm: Dict[str, Any], total_eps: int) -> Dict[str, Any]:
    arm_name = arm["name"]
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, arm)

    target_env = CausalGridWorldV2(
        seed=seed,
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=P2_HFA_GUARD,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
    )
    cfg = _make_config(target_env)
    agent = REEAgent(cfg)
    device = torch.device("cpu")
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {arm_name}", flush=True)

    # Stage 0 -- forced-benefit nursery.
    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    print(
        f"  [train] stage0_nursery seed={seed} arm={arm_name} ep {done}/{total_eps}"
        f" forced_benefit={s0.mean_forced_benefit:.2f}"
        f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}",
        flush=True,
    )
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, arm_name, "stage0", s0.abort_reason,
                                    s0_peak=s0.z_goal_norm_peak, s0_benefit=s0.mean_forced_benefit)

    # Stage 0b -- PROTECTED consolidation.
    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(
        f"  [train] stage0b_consolidate seed={seed} arm={arm_name} ep {done}/{total_eps}"
        f" start={s0b.z_goal_norm_start:.4f} end={s0b.z_goal_norm_end:.4f}"
        f" retention={s0b.retention_ratio:.3f}"
        f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}",
        flush=True,
    )
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, arm_name, "stage0b", s0b.abort_reason,
                                    s0_peak=s0.z_goal_norm_peak, s0_benefit=s0.mean_forced_benefit,
                                    s0b_retention=s0b.retention_ratio, s0b_pass=s0b.retention_gate_passed)

    # Stage 1 -- guided low-conflict warm-up (run_p0).
    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(
        f"  [train] p0_guided seed={seed} arm={arm_name} ep {done}/{total_eps}"
        f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
        flush=True,
    )
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, arm_name, "p0", p0.abort_reason,
                                    s0_peak=s0.z_goal_norm_peak, s0_benefit=s0.mean_forced_benefit,
                                    s0b_retention=s0b.retention_ratio, s0b_pass=s0b.retention_gate_passed)

    # Stage 2+3 -- easy->guarded foraging (run_p1, CONTACT-GATED, seeding-calibrated).
    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(
        f"  [train] p1_foraging seed={seed} arm={arm_name} ep {done}/{total_eps}"
        f" median_last={p1.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
        f" final_hfa={p1.final_hazard_food_attraction:.2f}"
        f" gated={p1.contact_gated} decay_only={p1.n_decay_only_updates}"
        f" skipped={p1.n_skipped_protected_updates}"
        f" refresh={p1.n_contact_refresh_updates}",
        flush=True,
    )

    # Stage 4 -- frozen-policy guarded measurement (run_p2, CONTACT-GATED, seeding-calibrated).
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(
        f"  [train] p2_measure seed={seed} arm={arm_name} ep {done}/{total_eps}"
        f" z_goal_frozen={p2.z_goal_norm_peak_max:.4f}"
        f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}"
        f" contact_events={p2.num_contact_events}"
        f" contact_rate={p2.contact_rate:.4f}"
        f" hfa_used={p2.hazard_food_attraction_used:.2f}"
        f" gated={p2.contact_gated} decay_only={p2.n_decay_only_updates}"
        f" skipped={p2.n_skipped_protected_updates}"
        f" refresh={p2.n_contact_refresh_updates}",
        flush=True,
    )

    # G3 uses the CONSUMPTION-EVENT-GATED peak (the fair readout), NOT the frozen peak.
    seed_pass = (
        s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE
        and s0b.retention_gate_passed
        and p1.survival_gate_passed
        and p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
        and p2.num_contact_events > 0
    )
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm_name}", flush=True)

    return {
        "seed": seed,
        "arm": arm_name,
        "aborted_at": None,
        "abort_reason": "",
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "stage0_benefit_exposure": float(s0.mean_forced_benefit),
        "stage0_z_goal_formed": bool(s0.z_goal_formed),
        "stage0b_z_goal_norm_start": float(s0b.z_goal_norm_start),
        "stage0b_z_goal_norm_end": float(s0b.z_goal_norm_end),
        "stage0b_retention_ratio": float(s0b.retention_ratio),
        "stage0b_retention_gate_passed": bool(s0b.retention_gate_passed),
        "p0_mean_episode_length": float(p0.mean_episode_length),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
        "p1_contact_gated": bool(p1.contact_gated),
        "p1_n_decay_only_updates": int(p1.n_decay_only_updates),
        "p1_n_contact_refresh_updates": int(p1.n_contact_refresh_updates),
        "p1_n_skipped_protected_updates": int(p1.n_skipped_protected_updates),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_contact_steps": int(p2.contact_steps),
        "p2_z_goal_norm_peak": float(p2.z_goal_norm_peak_max),                 # frozen (diagnostic)
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),  # consumption-gated (G3)
        "p2_num_contact_events": int(p2.num_contact_events),
        "p2_hazard_food_attraction_used": float(p2.hazard_food_attraction_used),
        "p2_contact_gated": bool(p2.contact_gated),
        "p2_n_decay_only_updates": int(p2.n_decay_only_updates),
        "p2_n_contact_refresh_updates": int(p2.n_contact_refresh_updates),
        "p2_n_skipped_protected_updates": int(p2.n_skipped_protected_updates),
        "seed_pass": bool(seed_pass),
    }


def _fraction_passing(values: List[bool]) -> float:
    if not values:
        return 0.0
    return float(sum(1 for v in values if v)) / float(len(values))


def _evaluate_arm_gate(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Arm gate using the CONSUMPTION-EVENT-GATED peak for G3 (not the frozen peak)."""
    gate = evaluate_substrate_gate(
        stage0_z_goal_peaks_per_seed=[r["stage0_z_goal_norm_peak"] for r in per_seed],
        p1_survival_pass_per_seed=[r["p1_survival_pass"] for r in per_seed],
        p2_z_goal_peaks_per_seed=[r["p2_z_goal_norm_at_contact_peak"] for r in per_seed],
        p2_contact_rates_per_seed=[r["p2_contact_rate"] for r in per_seed],
        z_goal_gate=STAGE0_ZGOAL_GATE,
        contact_gate=CONTACT_GATE,
        min_fraction=MIN_FRACTION,
    )
    g0b = _fraction_passing(
        [bool(r["stage0b_retention_gate_passed"]) for r in per_seed]
    ) >= MIN_FRACTION
    gate["g0b_stage0b_retention"] = bool(g0b)
    gate["stage0b_retention_gate"] = float(STAGE0B_RETENTION_GATE)
    # Genuine-consumption requirement: >= 2/3 seeds actually had a seeding event.
    g_events = _fraction_passing(
        [int(r.get("p2_num_contact_events", 0)) > 0 for r in per_seed]
    ) >= MIN_FRACTION
    gate["g_consumption_events"] = bool(g_events)
    arm_pass = bool(gate["substrate_gate_passed"] and g0b and g_events)
    gate["substrate_gate_passed"] = arm_pass
    gate["g3_uses_consumption_gated_peak"] = True
    return gate


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS

    # total training episodes per seed (denominator for [train] ep N/M prints).
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 2
    else:
        total_eps = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + P1_BUDGET + P2_BUDGET

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        per_seed = [_run_seed(s, dry_run, arm, total_eps) for s in seeds]
        gate = _evaluate_arm_gate(per_seed)
        branch = classify_interpretation_branch(gate)
        seeding_floor = (
            _seeding_floor(arm["gain"], arm["benefit_threshold"], arm["drive_floor"])
            if arm["seeding"] else None
        )
        print(
            f"[{EXPERIMENT_TYPE}] arm={arm['name']} seeding={arm['seeding']}"
            f" seeding_floor={seeding_floor}"
            f" gate_passed={gate['substrate_gate_passed']} branch={branch}",
            flush=True,
        )
        arm_results.append({
            "arm": arm["name"],
            "seeding": bool(arm["seeding"]),
            "gain": arm["gain"],
            "benefit_threshold": arm["benefit_threshold"],
            "drive_floor": arm["drive_floor"],
            "seeding_floor": seeding_floor,
            "gate": gate,
            "interpretation_branch": branch,
            "arm_pass": bool(gate["substrate_gate_passed"]),
            "per_seed": per_seed,
        })

    seeding_arms = [a for a in arm_results if a["seeding"]]
    baseline = next((a for a in arm_results if not a["seeding"]), None)
    any_seeding_pass = any(a["arm_pass"] for a in seeding_arms)
    passing_arm = next((a["arm"] for a in seeding_arms if a["arm_pass"]), None)

    # Overall interpretation grid.
    baseline_engaged = bool(baseline and baseline["gate"].get("g2_contact") and baseline["gate"].get("g3_zgoal"))
    if any_seeding_pass:
        overall_branch = "substrate_ready_seeding_calibrated"
    elif all(a["gate"].get("g2_contact") and not a["gate"].get("g3_zgoal") for a in seeding_arms) and seeding_arms:
        overall_branch = "fed_but_no_goal_beyond_magnitude"
    elif all(not a["gate"].get("g1_survival") for a in arm_results):
        overall_branch = "foraging_competence_open"
    else:
        overall_branch = "substrate_not_engaged"

    # Regression guard: ARM_0 baseline should NOT pass (wild benefit cannot seed at default config).
    baseline_regression_ok = bool(baseline is None or not baseline["arm_pass"])

    outcome = "PASS" if any_seeding_pass else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] any_seeding_pass={any_seeding_pass} passing_arm={passing_arm}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] overall_branch={overall_branch} baseline_engaged={baseline_engaged}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] baseline_regression_ok={baseline_regression_ok}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome}", flush=True)

    return {
        "outcome": outcome,
        "substrate_gate_passed": bool(any_seeding_pass),
        "passing_arm": passing_arm,
        "interpretation_branch": overall_branch,
        "baseline_engaged": baseline_engaged,
        "baseline_regression_ok": baseline_regression_ok,
        "arms": arm_results,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    """Run + write manifest. Returns {outcome, manifest_path} for the __main__
    block to relay to emit_outcome (the runner-conformance sentinel)."""
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
        "evidence_direction": "non_contributory",  # diagnostic; tags no claim
        "substrate": "scaffolded_sd054_onboarding (seeding-calibration amend, 2026-06-03c)",
        "predecessor": "V3-EXQ-634b (consolidation-validated nursery; NOT superseded -- adds seeding-calibration)",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "stage0b_retention_gate": STAGE0B_RETENTION_GATE,
            "p2_z_goal_gate": P2_ZGOAL_GATE,
            "p2_z_goal_readout": "z_goal_norm_at_contact_peak (consumption-event-gated; NOT frozen peak)",
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "train_steps": TRAIN_STEPS, "p1_hold_fraction": P1_HOLD_FRACTION,
            "p0_num_hazards": P0_NUM_HAZARDS, "p2_hfa_guard": P2_HFA_GUARD,
            "developmental_window_enabled": True,
            "stage0b_enabled": True,
            "contact_gated_goal_updates": True,
        },
        "arms_spec": [
            {"name": a["name"], "seeding": a["seeding"], "gain": a["gain"],
             "benefit_threshold": a["benefit_threshold"], "drive_floor": a["drive_floor"]}
            for a in ARMS
        ],
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
