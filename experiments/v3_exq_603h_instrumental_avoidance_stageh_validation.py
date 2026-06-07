"""
V3-EXQ-603h -- SD-058 / MECH-357 instrumental-avoidance acquisition: Stage-H
survival-leg validation (gate ON vs bit-identical OFF baseline).

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
The validation gate for the SD-058 / MECH-357 instrumental-avoidance acquisition
substrate landed 2026-06-07. V3-EXQ-603g (the curriculum-decomposition amend,
isolated hazard-avoidance Stage-H) PROVED goal-FORMATION works (G0 3/3) but the
P1 survival / hazard-avoidance LEARNING leg does NOT train even when isolated as
a dedicated Stage-H (G1 0/3, G_H isolated-hazard 0/3 at budget). The user
adjudicated this as a DEEPER survival/aversion-learning substrate gap, NOT a
budget tweak; the targeted_review_hazard_avoidance_learning lit-pull verdict
(SD-035 x3 + MECH-279 + SD-054) is that the fix is STRUCTURAL: REE has the
Pavlovian/defensive REACTION side (SD-035 amygdala salience + MECH-279 PAG
freeze) but lacked the instrumental-ACQUISITION side. Moscarello & LeDoux 2013:
active avoidance learning is the resolution of a Pavlovian-instrumental conflict
-- learning to avoid REQUIRES the infralimbic PFC to SUPPRESS CeA-driven
freezing. A freeze-only substrate freezes instead of learning to avoid -- the
603g G_H 0/3 prediction.

THE SUBSTRATE THIS VALIDATES: SD-058 / MECH-357 InstrumentalAvoidanceGate
(ree_core/pfc/infralimbic_avoidance_gate.py) -- an ilPFC-analog (a) instrumental-
avoidance action-bias that penalises the no-op/freeze class under threat, (b)
freeze-suppression over the MECH-279 no-op, and (c) an eligibility-trace
avoidance-efficacy learner credited when a directed action under threat drops
z_harm_a. Driven in Stage-H by a PROTECTIVE-SCAFFOLD floor anneal (0.8 -> 0.0;
maternal-buffering / Turchetta 2020 reset-curriculum analogue). The gate is
active across ALL stages in ARM_ON (so its Stage-H-acquired competence carries
into P1 -- the GAP-2 transfer test).

THE LOAD-BEARING PREREQUISITE (found 2026-06-07): the legacy scaffold calls
sense(body, world) with NO harm args, so z_harm_a is None across the WHOLE
curriculum -- which leaves MECH-279 (PAG freeze), SD-035 (amygdala), AND the
SD-058/MECH-357 gate all INERT (they key on z_harm_a). This run sets
scaffold_feed_harm_stream=True so the env harm stream (harm_obs + harm_obs_a) is
fed and z_harm_a is populated (~0.34 in Stage-H) -- the agent finally has a threat
signal to freeze about / learn avoidance from.

DESIGN: literal Moscarello & LeDoux lesion-vs-intact, 2-arm, 3 seeds [42, 43, 44].
BOTH arms have MECH-279 PAG (the Pavlovian freeze reaction, tuned to z_harm_a~0.34
so it actually fires) AND the fed harm stream. The ONLY difference is the ilPFC gate:
  ARM_LESION (use_instrumental_avoidance=False) = PAG on, ilPFC gate OFF -> freezes
    under threat, cannot acquire avoidance (the ilPFC-lesion animal).
  ARM_INTACT (use_instrumental_avoidance=True + scaffold_avoidance_driver_enabled=True,
    protective-scaffold floor 0.8 -> 0.0) = PAG on + ilPFC gate -> SUPPRESSES the
    freeze + acquires instrumental avoidance (intact ilPFC). The gate is active
    across ALL stages, so Stage-H-acquired avoidance carries into P1 (GAP-2 transfer).

DISTINCT from the reflexive escape levers (note 2): SD-037 orexin raises the PAG
exit threshold and MECH-281 lowers the MECH-091 urgency-interrupt -- both reflexive
threat/arousal-driven escape. The ilPFC suppression here is gated by LEARNED
avoidance-efficacy (eligibility trace), bootstrapped by the protective-scaffold
floor; it is the acquisition mechanism, not a reflex.

PRE-REGISTERED ACCEPTANCE (constants; NOT derived from the run's own statistics):
  PRIMARY (does the survival/hazard-avoidance leg now TRAIN with the gate ON?):
    G_H_INTACT >= 2/3 INTACT seeds clear the Stage-H survival gate (median
            last-window episode length over Stage-H >= 75), AND
    G_H_INTACT_frac > G_H_LESION_frac (the ilPFC gate lifts isolated-stage survival
            over the freeze-only lesion).
  PASS = readiness_met AND both PRIMARY conditions hold.
  Reported (NOT the PASS driver): G1 (P1 survival -- the GAP-2 transfer prize),
  G0 (Stage-0 positive control), G2 (P2 contact), and the per-seed
  avoidance_gate_state + pag_n_commits so a manifest can confirm avoidance was
  ACQUIRED (efficacy rose / freeze suppressed) rather than survival-by-chance.

READINESS PRECONDITIONS (non-vacuity; the literal design must instantiate):
  (1) PAG freezes on the LESION arm (pag_n_commits>0 on >=2/3 seeds) -- the
      Pavlovian reaction the ilPFC suppresses is present.
  (2) The gate ENGAGES + SUPPRESSES on the INTACT arm (n_credit+n_decay>0 AND
      n_freeze_suppressed>0 on >=2/3 seeds) -- the SAME mechanism the PRIMARY G_H
      lift routes on.
  If either is unmet (no freeze to suppress, or the gate never engaged), a G_H
  comparison is UNINFORMATIVE -> self-route substrate_not_ready_requeue (retune
  PAG / Stage-H threat), NEVER a substrate verdict. G0 (Stage-0 forced-feed
  z_goal > 0.4) is the goal-formation positive control (shared from 603g).

INTERPRETATION ON OUTCOME (this run weights no claim; diagnostic in every case):
  Readiness unmet (gate never engaged under threat) -> substrate_not_ready_requeue.
  PASS (G_H_ON >= 2/3 AND ON > OFF) -> the instrumental-avoidance mechanism trains
    the survival leg: a follow-on /governance + /queue-experiment runs the FULL
    603g-style readiness gate with the gate ON to confirm G0/G1/G2/G3 and flip
    substrate_queue scaffolded_sd054_onboarding ready. (NOT automatic here.)
  G_H_ON < 2/3 OR ON <= OFF (gate engaged but did not lift survival) ->
    avoidance_gate_insufficient: the mechanism is wired + engaged but does not
    deliver Stage-H survival at this budget/config -> /failure-autopsy (tune the
    gate, or a deeper survival-substrate gap remains).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; validates SD-058/MECH-357 wiring, weights no claim)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_603h_instrumental_avoidance_stageh_validation"
QUEUE_ID = "V3-EXQ-603h"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror 603g exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Budgets (mirror 603g full budget so ARM_OFF reproduces the 603g signature).
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

# Isolated hazard-avoidance Stage-H (the 603g curriculum-decomposition amend).
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = False
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603g).
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (the avoidance-learning driver) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
# z_harm_a_norm in this Stage-H env measures ~0.34 (constant, untrained encoder);
# calibrate the gate threat envelope + PAG freeze thresholds to that magnitude so
# the gate engages AND PAG actually freezes (the literal Pavlovian reaction the
# ilPFC suppresses). threat_floor 0.1 < 0.34 (gate engages); threat_ref 0.35
# (threat_scale ~1 at observed magnitude). PAG duration_input_threshold 0.2 < 0.34
# (duration accumulates) + theta_freeze 0.8 (commits after ~2-3 sustained steps).
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# Pre-registered gates (constants).
STAGE0_ZGOAL_GATE = 0.4
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0

# Literal Moscarello & LeDoux lesion-vs-intact design. BOTH arms have PAG (the
# freeze reaction) + the fed harm stream; the ONLY difference is the ilPFC gate.
#   ARM_LESION  = PAG on, ilPFC gate OFF -> freezes under threat, cannot acquire
#                 avoidance (the ilPFC-lesion animal).
#   ARM_INTACT  = PAG on, ilPFC gate ON + protective-scaffold driver -> suppresses
#                 the freeze + acquires instrumental avoidance (intact ilPFC).
ARMS = [
    {"label": "ARM_LESION_pag_no_gate", "use_ia": False, "driver": False},
    {"label": "ARM_INTACT_pag_gate", "use_ia": True, "driver": True},
]


def _make_scaffold_cfg(dry_run: bool, driver: bool) -> ScaffoldedSD054OnboardingConfig:
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
        # The isolated Stage-H (603g amend).
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (INTACT arm only).
        scaffold_avoidance_driver_enabled=bool(driver),
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE (both arms): feed the env harm stream into sense() so
        # z_harm_a is populated -- without it MECH-279 PAG, SD-035 amygdala AND
        # the SD-058/MECH-357 gate are all inert (the legacy scaffold never fed
        # harm). This is the load-bearing wiring that gives the agent a threat
        # signal to freeze about / learn avoidance from.
        scaffold_feed_harm_stream=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, use_ia: bool) -> REEConfig:
    cfg = REEConfig.from_dims(
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
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        # MECH-279 PAG freeze-gate (BOTH arms) -- the literal Pavlovian freeze
        # reaction the ilPFC suppresses. Tuned to z_harm_a_norm ~0.34 so freezing
        # actually fires (duration_input_threshold < 0.34; moderate theta_freeze).
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        # SD-058 / MECH-357 instrumental-avoidance gate (INTACT arm only).
        use_instrumental_avoidance=bool(use_ia),
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _config_slice(arm: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """Content-addressed config slice for the per-cell arm fingerprint."""
    return {
        "arm": arm["label"],
        "use_instrumental_avoidance": bool(arm["use_ia"]),
        "scaffold_avoidance_driver_enabled": bool(arm["driver"]),
        "use_pag_freeze_gate": True,
        "pag_theta_freeze": PAG_THETA_FREEZE,
        "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
        "avoidance_threat_ref": AVOIDANCE_THREAT_REF,
        "feed_harm_stream": True,
        "avoidance_scaffold_floor_start": AVOIDANCE_SCAFFOLD_FLOOR_START,
        "avoidance_scaffold_floor_end": AVOIDANCE_SCAFFOLD_FLOOR_END,
        "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, P2_BUDGET, TRAIN_STEPS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "dry_run": bool(dry_run),
    }


def _aborted_record(arm_label: str, seed: int, stage: str, reason: str,
                    s0_peak: float = 0.0) -> Dict[str, Any]:
    return {
        "arm": arm_label, "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "hazard_stage_survival_pass": False,
        "hazard_stage_median_last_window": 0.0,
        "p1_survival_pass": False,
        "p2_contact_rate": 0.0,
        "g0_stage0_zgoal": bool(s0_peak > STAGE0_ZGOAL_GATE),
        "g1_p1_survival": False,
        "g2_p2_contact": False,
        "g_h_hazard_survival": False,
        "avoidance_gate_state": {},
        "pag_n_commits": 0,
        "pag_n_releases": 0,
        "reached_hazard_stage": stage not in ("stage0", "stage0b", "p0"),
        "reached_p1": False,
        "reached_p2": False,
        "seed_pass": False,
    }


def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool,
                  total_eps: int) -> Dict[str, Any]:
    """Full curriculum for one (arm, seed) cell. arm_cell resets all RNG on
    enter (order-independent) and stamps the fingerprint on the returned row."""
    with arm_cell(
        seed,
        config_slice=_config_slice(arm, dry_run),
        script_path=Path(__file__),
        config_slice_declared=True,
    ) as cell:
        scaffold_cfg = _make_scaffold_cfg(dry_run, arm["driver"])
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env, arm["use_ia"])).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

        print(f"Seed {seed} Condition {arm['label']}", flush=True)

        def _gate_state() -> Dict[str, Any]:
            g = getattr(agent, "instrumental_avoidance", None)
            return g.get_state() if g is not None else {}

        def _pag_state() -> Dict[str, Any]:
            p = getattr(agent, "pag_freeze_gate", None)
            return dict(p.diagnostics) if p is not None else {}

        # Stage 0 -- forced-benefit nursery (goal-formation positive control).
        s0 = scheduler.run_stage0_nursery(agent, device)
        done = s0.n_episodes
        print(f"  [train] stage0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0", s0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0b", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0b", s0b.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
              flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=p0", flush=True)
            rec = _aborted_record(arm["label"], seed, "p0", p0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        # Stage-H -- ISOLATED HAZARD-AVOIDANCE (the SD-058/MECH-357 driver target).
        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        gate_after_h = _gate_state()
        pag_after_h = _pag_state()
        print(f"  [train] hazard_avoidance {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={hz.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
              f" pag_commits={pag_after_h.get('n_commits', 0)}"
              f" eff={gate_after_h.get('mech357_avoidance_efficacy', 0.0):.4f}"
              f" n_credit={gate_after_h.get('mech357_n_credit', 0)}"
              f" n_freeze_suppr={gate_after_h.get('mech357_n_freeze_suppressed', 0)}",
              flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=hazard", flush=True)
            rec = _aborted_record(arm["label"], seed, "hazard", hz.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            rec["avoidance_gate_state"] = gate_after_h
            cell.stamp(rec)
            return rec

        # P1 -- combined wean (the GAP-2 TRANSFER test: does Stage-H-acquired
        # avoidance carry into P1? In ARM_ON the gate is still active here).
        p1 = scheduler.run_p1(agent, device)
        done += p1.n_episodes
        print(f"  [train] p1 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={p1.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

        # P2 -- frozen-policy guarded measurement.
        p2 = scheduler.run_p2(agent, device)
        done += p2.n_episodes
        print(f"  [train] p2 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" contact_rate={p2.contact_rate:.4f}", flush=True)

        g0 = bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE)
        g1 = bool(p1.survival_gate_passed)
        g2 = bool(p2.contact_rate > CONTACT_GATE)
        g_h = bool(hz.survival_gate_passed)
        gate_final = _gate_state()
        seed_pass = bool(g_h)  # this run's per-seed pass is the Stage-H survival
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm['label']}"
              f" g_h={g_h} g0={g0} g1={g1} g2={g2}"
              f" eff_final={gate_final.get('mech357_avoidance_efficacy', 0.0):.4f}",
              flush=True)

        rec = {
            "arm": arm["label"],
            "seed": seed,
            "aborted_at": None,
            "abort_reason": "",
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "p0_mean_episode_length": float(p0.mean_episode_length),
            "hazard_stage_survival_pass": g_h,
            "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            "hazard_stage_n_episodes": int(hz.n_episodes),
            "hazard_avoidance_driver_enabled": bool(getattr(hz, "avoidance_driver_enabled", False)),
            "pag_n_commits": int(pag_after_h.get("n_commits", 0)),
            "pag_n_releases": int(pag_after_h.get("n_releases", 0)),
            "p1_survival_pass": g1,
            "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
            "p2_contact_rate": float(p2.contact_rate),
            "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
            "g0_stage0_zgoal": g0,
            "g1_p1_survival": g1,
            "g2_p2_contact": g2,
            "g_h_hazard_survival": g_h,
            "avoidance_gate_state": gate_final,
            "reached_hazard_stage": True,
            "reached_p1": True,
            "reached_p2": True,
            "seed_pass": seed_pass,
        }
        cell.stamp(rec)
        return rec


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

    arm_results: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    for arm in ARMS:
        rows = [_run_seed_arm(arm, s, dry_run, total_eps) for s in seeds]
        per_seed.extend(rows)
        g_h_flags = [bool(r.get("g_h_hazard_survival", False)) for r in rows]
        g1_flags = [bool(r.get("g1_p1_survival", False)) for r in rows]
        g0_flags = [bool(r.get("g0_stage0_zgoal", False)) for r in rows]
        g2_flags = [bool(r.get("g2_p2_contact", False)) for r in rows]
        # Gate-engagement (non-vacuity): did the gate register efficacy updates
        # under threat (credit + decay) AND suppress the PAG freeze at least once?
        engaged_flags, suppressed_flags = [], []
        for r in rows:
            gs = r.get("avoidance_gate_state", {}) or {}
            engaged_flags.append(
                (int(gs.get("mech357_n_credit", 0)) + int(gs.get("mech357_n_decay", 0))) > 0
            )
            suppressed_flags.append(int(gs.get("mech357_n_freeze_suppressed", 0)) > 0)
        # PAG-freeze (the Pavlovian reaction): did the freeze gate actually commit?
        pag_freeze_flags = [int(r.get("pag_n_commits", 0)) > 0 for r in rows]
        arm_results.append({
            "arm": arm["label"],
            "use_instrumental_avoidance": bool(arm["use_ia"]),
            "scaffold_avoidance_driver_enabled": bool(arm["driver"]),
            "g_h_frac": _frac(g_h_flags),
            "g0_frac": _frac(g0_flags),
            "g1_frac": _frac(g1_flags),
            "g2_frac": _frac(g2_flags),
            "gate_engaged_frac": _frac(engaged_flags),
            "gate_freeze_suppressed_frac": _frac(suppressed_flags),
            "pag_freeze_frac": _frac(pag_freeze_flags),
            "per_seed_g_h": g_h_flags,
            "per_seed_g1": g1_flags,
            "per_seed_pag_n_commits": [int(r.get("pag_n_commits", 0)) for r in rows],
            "per_seed_hazard_median_last_window": [
                r.get("hazard_stage_median_last_window", 0.0) for r in rows
            ],
            "per_seed_avoidance_efficacy": [
                (r.get("avoidance_gate_state", {}) or {}).get("mech357_avoidance_efficacy", 0.0)
                for r in rows
            ],
            "arm_fingerprint": [r.get("arm_fingerprint") for r in rows],
        })

    off = next(a for a in arm_results if not a["use_instrumental_avoidance"])
    on = next(a for a in arm_results if a["use_instrumental_avoidance"])

    # --- PRIMARY pre-registered acceptance (load-bearing) ---
    g_h_on_clears = bool(on["g_h_frac"] >= MIN_FRACTION)
    g_h_on_beats_off = bool(on["g_h_frac"] > off["g_h_frac"])
    primary_pass = bool(g_h_on_clears and g_h_on_beats_off)

    # --- READINESS PRECONDITIONS (non-vacuity; the literal lesion-vs-intact
    # design must actually instantiate). Two conditions, both keyed on the SAME
    # mechanisms the PRIMARY G_H lift routes on:
    #   (1) The PAVLOVIAN REACTION is present: PAG freezes on the LESION arm
    #       (pag_n_commits > 0 on >= 2/3 LESION seeds). If PAG never freezes,
    #       there is no freeze for the ilPFC to suppress -> the lesion-vs-intact
    #       contrast is vacuous -> substrate_not_ready_requeue (retune PAG /
    #       Stage-H threat), NOT an avoidance verdict.
    #   (2) The ilPFC gate ENGAGES + SUPPRESSES on the INTACT arm: it credits
    #       avoidance-efficacy under threat (n_credit+n_decay>0) AND suppresses
    #       the PAG freeze (n_freeze_suppressed>0) on >= 2/3 INTACT seeds. ---
    pavlovian_reaction_present = bool(off["pag_freeze_frac"] >= MIN_FRACTION)
    gate_engaged = bool(on["gate_engaged_frac"] >= MIN_FRACTION)
    gate_suppresses = bool(on["gate_freeze_suppressed_frac"] >= MIN_FRACTION)
    readiness_met = bool(pavlovian_reaction_present and gate_engaged and gate_suppresses)
    # G0 positive control (goal-formation; shared from 603g) on the INTACT arm.
    g0_on_ok = bool(on["g0_frac"] >= MIN_FRACTION)

    if not readiness_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
    elif primary_pass:
        outcome = "PASS"
        readiness_route = "avoidance_mechanism_trains_survival_leg"
    else:
        outcome = "FAIL"
        readiness_route = "avoidance_gate_insufficient"

    # Diagnostic adjudication structures.
    preconditions = [
        {
            "name": "pavlovian_freeze_reaction_present_on_lesion",
            "kind": "readiness",
            "description": "The MECH-279 PAG freeze must actually commit on the LESION arm "
                           "(pag_n_commits > 0 on >= 2/3 LESION seeds) -- the Pavlovian reaction "
                           "the ilPFC suppresses. Below-floor => no freeze to suppress => the "
                           "lesion-vs-intact contrast is vacuous => substrate_not_ready_requeue "
                           "(retune pag_theta_freeze / pag_duration_input_threshold to the "
                           "Stage-H z_harm_a magnitude), NOT an avoidance verdict.",
            "control": "ARM_LESION: PAG on (theta_freeze tuned to z_harm_a_norm ~0.34), harm "
                       "stream fed, midline hazard band -- the freeze reaction must fire.",
            "measured": float(off["pag_freeze_frac"]),
            "threshold": float(MIN_FRACTION),
            "met": bool(pavlovian_reaction_present),
        },
        {
            "name": "ilpfc_gate_engages_and_suppresses_freeze_on_intact",
            "kind": "readiness",
            "description": "On the INTACT arm the ilPFC gate must (a) register avoidance-efficacy "
                           "updates under threat (n_credit+n_decay>0) AND (b) suppress the PAG "
                           "freeze (n_freeze_suppressed>0) on >= 2/3 seeds -- the SAME mechanism "
                           "(learned suppression of the freeze enabling instrumental avoidance) "
                           "the PRIMARY G_H lift routes on. Below-floor => the gate never engaged "
                           "=> substrate_not_ready_requeue, NOT an avoidance verdict.",
            "control": "ARM_INTACT: gate on + protective-scaffold floor 0.8 bootstrapping "
                       "suppression so the agent acts (not freezes) and accrues avoidance experience.",
            "measured": float(min(on["gate_engaged_frac"], on["gate_freeze_suppressed_frac"])),
            "threshold": float(MIN_FRACTION),
            "met": bool(gate_engaged and gate_suppresses),
        },
        {
            "name": "stage0_forced_feed_lights_zgoal_on_intact",
            "kind": "readiness",
            "description": "Stage-0 forced supra-threshold benefit lights z_goal (>0.4) on "
                           ">=2/3 INTACT seeds -- the goal-FORMATION positive control (shared "
                           "from 603g). Confirms the curriculum is intact under the gate.",
            "control": "run_stage0_nursery forced-feed.",
            "measured": float(on["g0_frac"]),
            "threshold": float(MIN_FRACTION),
            "met": bool(g0_on_ok),
        },
    ]
    criteria_non_degenerate = {
        # Both arms reached Stage-H so the G_H comparison is non-degenerate.
        "both_arms_reached_hazard_stage": bool(
            _frac([r.get("reached_hazard_stage", False) for r in per_seed]) >= MIN_FRACTION
        ),
        # The Pavlovian reaction is present (PAG froze on the lesion).
        "pavlovian_freeze_present": bool(off["pag_freeze_frac"] > 0.0),
        # The arms genuinely differ (the gate engaged + suppressed on intact).
        "gate_active_on_intact": bool(on["gate_engaged_frac"] > 0.0),
    }
    criteria = [
        {"name": "G_H_INTACT_clears_2of3", "load_bearing": True, "passed": bool(g_h_on_clears)},
        {"name": "G_H_INTACT_beats_LESION", "load_bearing": True, "passed": bool(g_h_on_beats_off)},
    ]

    label = readiness_route

    print(
        f"[{EXPERIMENT_TYPE}] G_H_INTACT={on['g_h_frac']:.2f} G_H_LESION={off['g_h_frac']:.2f}"
        f" pag_freeze_LESION={off['pag_freeze_frac']:.2f}"
        f" gate_engaged_INTACT={on['gate_engaged_frac']:.2f}"
        f" gate_suppr_INTACT={on['gate_freeze_suppressed_frac']:.2f}"
        f" G1_INTACT(transfer)={on['g1_frac']:.2f} G1_LESION={off['g1_frac']:.2f}"
        f" -> outcome={outcome} route={readiness_route}",
        flush=True,
    )

    return {
        "outcome": outcome,
        "primary_pass": primary_pass,
        "g_h_intact_clears": g_h_on_clears,
        "g_h_intact_beats_lesion": g_h_on_beats_off,
        "readiness_met": readiness_met,
        "pavlovian_reaction_present": pavlovian_reaction_present,
        "gate_engaged": gate_engaged,
        "gate_suppresses": gate_suppresses,
        "arm_results": arm_results,
        "acceptance": {
            "pass_rule": "PASS = readiness_met (PAG freezes on LESION >=2/3 AND gate engages+"
                         "suppresses on INTACT >=2/3) AND G_H_INTACT >= 2/3 AND "
                         "G_H_INTACT_frac > G_H_LESION_frac",
            "min_fraction": MIN_FRACTION,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "g_h_intact_frac": on["g_h_frac"],
            "g_h_lesion_frac": off["g_h_frac"],
            "pag_freeze_lesion_frac": off["pag_freeze_frac"],
            "gate_engaged_intact_frac": on["gate_engaged_frac"],
            "gate_freeze_suppressed_intact_frac": on["gate_freeze_suppressed_frac"],
            "g1_transfer_intact_frac": on["g1_frac"],
            "g1_transfer_lesion_frac": off["g1_frac"],
        },
        "interpretation": {
            "label": label,
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
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",  # diagnostic; tags no claim
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "SD-058 / MECH-357 instrumental-avoidance acquisition "
                     "(ree_core/pfc/infralimbic_avoidance_gate.py) driven in the "
                     "scaffolded_sd054_onboarding Stage-H",
        "validates": "SD-058 (architecture) + MECH-357 (ilPFC-analog freeze-suppression + "
                     "instrumental-avoidance action pathway + eligibility-trace efficacy learning)",
        "design_note": "Literal Moscarello & LeDoux lesion-vs-intact, 2-arm, 3 seeds. BOTH arms "
                       "have MECH-279 PAG (the Pavlovian freeze reaction, tuned to z_harm_a_norm "
                       "~0.34) AND the FED harm stream (scaffold_feed_harm_stream=True -- the "
                       "load-bearing prerequisite: the legacy scaffold never fed harm, so z_harm_a "
                       "was None and every harm-driven substrate inert). The ONLY difference is the "
                       "ilPFC gate: ARM_LESION (PAG, no gate -> freezes, cannot acquire avoidance) "
                       "vs ARM_INTACT (PAG + ilPFC gate + protective-scaffold driver -> SUPPRESSES "
                       "the freeze + acquires instrumental avoidance). The gate is active across ALL "
                       "stages in ARM_INTACT, so Stage-H-acquired avoidance carries into P1 (GAP-2 "
                       "transfer). DISTINCT from the reflexive escape levers (SD-037 orexin raises "
                       "PAG exit threshold; MECH-281 lowers urgency-interrupt) -- the ilPFC "
                       "suppression is gated by LEARNED avoidance-efficacy (eligibility trace), not "
                       "instantaneous arousal. Routed by failure_autopsy_V3-EXQ-603g-624c-651a_2026-06-07 "
                       "+ targeted_review_hazard_avoidance_learning/SYNTHESIS.md.",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_pass_rule": "readiness_met (PAG freezes on LESION >=2/3 AND gate engages+"
                                 "suppresses on INTACT >=2/3) AND G_H_INTACT >= 2/3 AND "
                                 "G_H_INTACT_frac > G_H_LESION_frac",
            "g_h_hazard_stage_survival": "median episode length over last 10 Stage-H episodes >= 75",
            "g1_p1_survival_transfer": "median episode length over last 10 P1 episodes >= 75 (reported, GAP-2 prize)",
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "min_fraction": MIN_FRACTION,
        },
        "avoidance_driver": {
            "scaffold_floor_start": AVOIDANCE_SCAFFOLD_FLOOR_START,
            "scaffold_floor_end": AVOIDANCE_SCAFFOLD_FLOOR_END,
            "feed_harm_stream": True,
            "avoidance_threat_ref": AVOIDANCE_THREAT_REF,
            "pag_theta_freeze": PAG_THETA_FREEZE,
            "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
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
