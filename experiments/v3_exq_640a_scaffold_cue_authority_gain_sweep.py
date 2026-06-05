"""
V3-EXQ-640a: SD-057 cue-recall -- cue-AUTHORITY GAIN SWEEP. The measurement
successor to V3-EXQ-640. 640 measured the post-cue z_goal/approach trace at a
SINGLE operating point (cue_recall_gain=0.2) and settled (autopsy
failure_autopsy_V3-EXQ-640_2026-06-05): the cue fires 1557x across seeds but
moves z_goal only ~0.4% (cue_zgoal_pull_norm ~0.002 vs ||z_goal|| ~0.45) and
the post-cue selected-action approach rate EQUALS the cue-independent
background rate -- cue-to-action AUTHORITY is missing, displacement was
REFUTED. The autopsy's mechanistic root: cue_recall_gain 0.2 x weak stored
token (~0.2) -> sub-threshold cue_pull, ~3 orders of magnitude too small to
redirect the committed z_goal attractor. The autopsy explicitly routes the
next step as a cue-authority gain sweep on the EXISTING substrate (cue_pull +
MECH-295/E3 approach pathways already exist; recommended_substrate_queue_entry
.action = "none" -- this is a gain/parameter question, not a missing primitive).

WHAT THIS SWEEPS: a 2-axis factorial over
  - cue_recall_gain          in {0.2, 1.0, 5.0}  (640 baseline -> 5x -> 25x)
  - incentive_drive_kappa_weight in {2.0, 10.0}  (recall-time wanting amplitude
      = base_value[k] * (1 + kappa * per_axis_drive[k]); the operational
      "incentive-token strength" at the moment of the cue-pull)
The per-cue-fire pull strength is cue_recall_gain * min(1.0, wanting_amp), and
GoalState.cue_pull moves z_goal = (1-s)*z_goal + s*z_object. So both axes scale
the measured cue_zgoal_pull_norm. The high-gain cells (gain=5.0) drive the pull
strength to its clamp ceiling (z_goal effectively snaps onto z_object), which is
the decisive probe: if even a near-full snap produces NO approach lift over
background, the z_goal -> approach link (E3 goal_proximity / MECH-295) is the
gap, NOT the cue_pull magnitude.

BEHAVIOURALLY: identical to 640 except for the two swept config knobs. Both arms
set scaffold_post_cue_instrumentation=True so scaffolded_sd054_onboarding.
_eval_episode populates the read-only post_cue_diag accumulator. No new
substrate primitive, no new env knob, no tuning-to-pass.

PRE-REGISTERED QUESTION (the sweep MEASURES this relationship; it does not hunt
for a passing config -- experiment_purpose=diagnostic, claim_ids=[] => weights
no governance score):
  Does the per-cue-fire z_goal pull (cue_zgoal_pull_norm) AND the post-cue
  approach-rate lift over the within-run background rate rise MONOTONICALLY as
  cue_recall_gain (and incentive-token strength) increase?

ARMS (3 seeds 42/43/44 each; all instrumented):
  ARM_OFF                cue-recall bridge OFF. No cues fire; supplies the
                         cue-independent background baselines + the wild-seeded
                         ||z_goal|| attractor norm (displacement cross-check).
                         Reference arm; one per seed (cue-independent).
  ARM_CUE_g{G}_k{K}      cue-recall bridge ON + scaffold_stage0_bind_incentive_token
                         =True (the 638 formation fix), with cue_recall_gain=G
                         and incentive_drive_kappa_weight=K. 6 cells:
                         (0.2,2.0) (0.2,10.0) (1.0,2.0) (1.0,10.0) (5.0,2.0) (5.0,10.0).
  Note: each cue-on run ALSO records its OWN within-run background_approach_rate
  (computed over all P2 steps, cue-independent), which is the matched denominator
  for the approach-lift read. ARM_OFF is an additional cross-arm reference.

SELF-CONTAINED + DECOUPLED: every arm sets the landed 634c ARM_3 seeding regime
(drive_floor=0.9 + benefit_threshold=0.02) directly, exactly as 640 did. Does
NOT depend on any stalled run -- only the landed substrate code.

INTERPRETATION GRID (applied at REVIEW; this experiment MEASURES, it does not
adjudicate). Per failure_autopsy_V3-EXQ-640_2026-06-05 Section 7 routing grid:
  z_goal pull rises with gain AND post-cue approach lifts above background
    -> cue authority is real but was under-tuned. Pick the operating point and
       proceed to the V3-EXQ-638b interoceptive build.
  z_goal pull rises with gain BUT approach lift stays ~0
    -> the pull reaches z_goal but does NOT propagate to action selection
       (E3 goal_proximity / MECH-295 integration gap). Route to that
       integration layer -- do NOT build 638b.
  z_goal pull stays ~0 even at high gain
    -> the cue_pull mechanism itself is under-powered / clipped. A
       substrate_queue entry (cue_pull primitive revisit) becomes appropriate.

ACCEPTANCE (measurement-success ONLY, NOT a scientific pass; diagnostic weights
no claim):
  C1 (cue fires ON):  across the 6 cue-on cells, ARM P2 post_cue n_cue_fire_steps
                      > 0 on >= 2/3 seeds in >= a majority of cells -- there are
                      cue events to measure.
  C2 (sweep trace):   across the 6 cue-on cells, ARM P2 n_cue_windows > 0 AND a
                      measurable post_cue_approach_rate on >= 2/3 seeds in >= a
                      majority of cells -- the per-cell pull/approach trace the
                      interpretation grid reads now exists for every operating
                      point.
  overall_pass = C1 AND C2. PASS => the sweep measurement succeeded and the
  monotonicity read + interpretation grid can be applied at review. A FAIL means
  the instrumentation captured no usable post-cue trace across the sweep ->
  /diagnose-errors on the harness wiring, NOT a substrate finding.

claim_ids = []  (measurement-only behavioural diagnostic; NOT governance evidence).
experiment_purpose = "diagnostic".

HARD CONSTRAINT (carried from the 640 autopsy do-not-do-yet list): this sweep
GATES V3-EXQ-638b. Do NOT build the 638b interoceptive need-gating substrate
until this sweep routes.

References:
- REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-640_2026-06-05.{md,json}
  (the autopsy that routed this sweep; Section 7 interpretation grid)
- V3-EXQ-640 (single-point cue_recall_gain=0.2 measurement; the gain=0.2 cells
  of this sweep are its direct extension; 640's measurement stays valid as the
  baseline point)
- ree-v3/experiments/scaffolded_sd054_onboarding.py (post-cue instrumentation)
- ree-v3 goal.py GoalState.cue_pull + IncentiveTokenBank; agent.py
  cue_recall_wanting (MECH-347 L6); cue_recall_gain + incentive_drive_kappa_weight
  (from_dims knobs)
- V3-EXQ-634c (seeding-calibration regime reused)
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

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from experiments.scaffolded_sd054_onboarding import (
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
)
from experiment_protocol import emit_outcome

EXPERIMENT_TYPE = "v3_exq_640a_scaffold_cue_authority_gain_sweep"
QUEUE_ID = "V3-EXQ-640a"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Curriculum budgets (mirror 634c / 638 / 638a / 640 -- behaviourally unchanged).
STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, P1_BUDGET, P2_BUDGET = 20, 10, 120, 70, 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
STAGE0_ZGOAL_GATE = 0.4
STAGE0B_RETENTION_GATE = 0.75
MIN_FRACTION = 2.0 / 3.0
N_RESOURCE_TYPES = 3
POST_CUE_WINDOW_STEPS = 4

# Shared ARM_3 seeding regime (so wild contact seeds in BOTH arms).
SEED_DRIVE_FLOOR = 0.9
SEED_BENEFIT_THRESHOLD = 0.02

# --- The two sweep axes (the cue-authority question). ---
GAIN_SWEEP = [0.2, 1.0, 5.0]      # cue_recall_gain: 640 baseline -> 5x -> 25x.
KAPPA_SWEEP = [2.0, 10.0]         # incentive_drive_kappa_weight: token-strength axis.
# 640's single-point baseline = (gain=0.2, kappa=2.0).
GAIN_BASELINE = 0.2
KAPPA_BASELINE = 2.0


def _build_arms() -> List[Dict[str, Any]]:
    arms: List[Dict[str, Any]] = [
        {"name": "ARM_OFF", "cue": False,
         "gain": GAIN_BASELINE, "kappa": KAPPA_BASELINE}
    ]
    for kappa in KAPPA_SWEEP:
        for gain in GAIN_SWEEP:
            name = f"ARM_CUE_g{gain:g}_k{kappa:g}"
            arms.append({"name": name, "cue": True, "gain": gain, "kappa": kappa})
    return arms


ARMS: List[Dict[str, Any]] = _build_arms()


def _seeding_floor() -> float:
    # b* = benefit_threshold / (gain(1.0) * (1 + drive_weight * drive_floor))
    return SEED_BENEFIT_THRESHOLD / (1.0 * (1.0 + DRIVE_WEIGHT * SEED_DRIVE_FLOOR))


def _make_scaffold_cfg(dry_run: bool, cue_on: bool) -> ScaffoldedSD054OnboardingConfig:
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
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        # Shared ARM_3 seeding regime (every arm).
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_contact_gating_benefit_threshold=_seeding_floor(),
        # Measurement: instrument the post-cue window in EVERY arm. ARM_OFF
        # fires no cue but supplies the cue-independent background baselines.
        scaffold_post_cue_instrumentation=True,
        scaffold_post_cue_window_steps=POST_CUE_WINDOW_STEPS,
    )
    if cue_on:
        cfg.scaffold_cue_recall_bridge_enabled = True
        cfg.scaffold_cue_n_resource_types = N_RESOURCE_TYPES
        # Formation fix (same as 638a / 640): bind a per-type token at Stage-0
        # forced feeding so the bank is non-empty entering P1/P2.
        cfg.scaffold_stage0_bind_incentive_token = True
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, cue_on: bool, cue_recall_gain: float,
                 kappa: float) -> REEConfig:
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
        # SD-057: cue-recall bridge agent flags (ON arms only). The two SWEPT
        # knobs: cue_recall_gain (L6 pull strength) + incentive_drive_kappa_weight
        # (L3 recall-time wanting amplitude = incentive-token strength).
        use_incentive_token_bank=cue_on,
        use_cue_recall=cue_on,
        cue_recall_gain=cue_recall_gain,
        incentive_drive_kappa_weight=kappa,
    )
    if cue_on:
        cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    return cfg


def _derive_post_cue_metrics(pc: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the raw post_cue_diag sums/counts into the interpretable rates the
    autopsy interpretation grid reads. Safe on empty (ARM_OFF / no fires)."""
    def _safe_div(a: float, b: float) -> Optional[float]:
        return (float(a) / float(b)) if b else None

    n_fire = int(pc.get("n_cue_fire_steps", 0))
    n_win = int(pc.get("n_cue_windows", 0))
    n_improved = int(pc.get("n_windows_improved", 0))
    min_norm = pc.get("min_zgoal_norm_at_cue_fire", 0.0)
    if n_fire == 0 or min_norm == float("inf"):
        min_norm = 0.0
    return {
        "n_cue_fire_steps": n_fire,
        "n_cue_windows": n_win,
        "n_steps_total": int(pc.get("n_steps_total", 0)),
        # DISPLACEMENT test (mean < 0 => cue lowers ||z_goal||).
        "mean_post_cue_zgoal_norm_delta": _safe_div(
            pc.get("sum_post_cue_zgoal_norm_delta", 0.0),
            pc.get("n_post_cue_zgoal_norm_delta", 0)),
        # CUE-AUTHORITY load-bearing measure: per-cue-fire z_goal pull magnitude.
        "mean_cue_zgoal_pull_norm": _safe_div(
            pc.get("sum_cue_zgoal_pull_norm", 0.0), n_fire),
        "mean_zgoal_norm_at_cue_fire": _safe_div(
            pc.get("sum_zgoal_norm_at_cue_fire", 0.0), n_fire),
        "min_zgoal_norm_at_cue_fire": float(min_norm),
        "max_zgoal_norm_at_cue_fire": float(pc.get("max_zgoal_norm_at_cue_fire", 0.0)),
        "mean_zgoal_norm_all_steps": _safe_div(
            pc.get("sum_zgoal_norm_all_steps", 0.0),
            pc.get("n_zgoal_norm_all_steps", 0)),
        "mean_cue_action_bias_norm": _safe_div(
            pc.get("sum_cue_action_bias_norm", 0.0),
            pc.get("n_cue_action_bias_present", 0)),
        # AUTHORITY / gradient-following test.
        "post_cue_approach_rate": _safe_div(
            pc.get("sum_move_improved_postcue_steps", 0),
            pc.get("n_postcue_eval_steps", 0)),
        "background_approach_rate": _safe_div(
            pc.get("sum_move_improved_all_steps", 0),
            pc.get("n_move_eval_steps", 0)),
        "frac_windows_first_move_approach": _safe_div(
            pc.get("n_windows_first_move_approach", 0), n_win),
        "frac_windows_with_approach": _safe_div(
            pc.get("n_windows_with_approach_move", 0), n_win),
        "mean_first_improving_latency": _safe_div(
            pc.get("sum_first_improving_latency", 0), n_improved),
        # INTERRUPT test.
        "hazard_interrupt_rate": _safe_div(
            pc.get("n_windows_with_hazard_interrupt", 0), n_win),
        "mean_window_oscillations": _safe_div(
            pc.get("sum_window_oscillations", 0), n_win),
    }


def _sanitize_pc(pc: Dict[str, Any]) -> Dict[str, Any]:
    """Replace the inf sentinel with 0.0 so the raw diag is JSON-clean."""
    out = dict(pc)
    if out.get("min_zgoal_norm_at_cue_fire", 0.0) == float("inf"):
        out["min_zgoal_norm_at_cue_fire"] = 0.0
    return out


def _new_post_cue_diag_empty() -> Dict[str, Any]:
    from experiments.scaffolded_sd054_onboarding import _new_post_cue_diag
    return _sanitize_pc(_new_post_cue_diag(POST_CUE_WINDOW_STEPS))


def _abort(seed: int, arm: Dict[str, Any], stage: str) -> Dict[str, Any]:
    return {
        "seed": seed, "arm": arm["name"], "cue": bool(arm["cue"]),
        "gain": float(arm["gain"]), "kappa": float(arm["kappa"]),
        "aborted_at": stage,
        "p1_survival_pass": False, "p2_contact_rate": 0.0,
        "p2_num_contact_events": 0, "p2_n_cue_recall_fires": 0,
        "post_cue_metrics": _derive_post_cue_metrics(_new_post_cue_diag_empty()),
        "post_cue_diag_raw": _new_post_cue_diag_empty(),
    }


def _run_seed(seed: int, dry_run: bool, arm: Dict[str, Any],
              total_eps: int) -> Dict[str, Any]:
    arm_name, cue_on = arm["name"], bool(arm["cue"])
    gain, kappa = float(arm["gain"]), float(arm["kappa"])
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, cue_on)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env, cue_on, gain, kappa)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {arm_name}", flush=True)

    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    token_bank = int(getattr(s0, "token_bank_size_end", 0))
    print(f"  [train] stage0 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed} "
          f"token_bank_size_end={token_bank}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=stage0", flush=True)
        return _abort(seed, arm, "stage0")

    s0b = scheduler.run_stage0b_consolidation(
        agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"retention={s0b.retention_ratio:.3f} "
          f"gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=stage0b", flush=True)
        return _abort(seed, arm, "stage0b")

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=p0", flush=True)
        return _abort(seed, arm, "p0")

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"survival={'pass' if p1.survival_gate_passed else 'FAIL'} "
          f"refresh={p1.n_contact_refresh_updates}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    p2_cue_diag = dict(getattr(p2, "cue_diag", {}) or {})
    pc_raw = _sanitize_pc(dict(getattr(p2, "post_cue_diag", {}) or {}))
    pc_metrics = _derive_post_cue_metrics(pc_raw)
    cue_fires = int(p2_cue_diag.get("n_cue_recall_fires", 0))
    print(f"  [train] p2 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"gain={gain:g} kappa={kappa:g} contact_rate={p2.contact_rate:.4f} "
          f"cue_fires={cue_fires} pc_fire_steps={pc_metrics['n_cue_fire_steps']} "
          f"pc_windows={pc_metrics['n_cue_windows']} "
          f"pull_norm={pc_metrics['mean_cue_zgoal_pull_norm']} "
          f"post_cue_approach={pc_metrics['post_cue_approach_rate']} "
          f"bg_approach={pc_metrics['background_approach_rate']}", flush=True)

    print(f"verdict: seed={seed} arm={arm_name} "
          f"pc_fire_steps={pc_metrics['n_cue_fire_steps']} "
          f"pc_windows={pc_metrics['n_cue_windows']}", flush=True)
    return {
        "seed": seed, "arm": arm_name, "cue": cue_on,
        "gain": gain, "kappa": kappa, "aborted_at": None,
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_num_contact_events": int(p2.num_contact_events),
        "p2_n_cue_recall_fires": cue_fires,
        "post_cue_metrics": pc_metrics,
        "post_cue_diag_raw": pc_raw,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _mean(vals: List[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None]
    return (sum(xs) / len(xs)) if xs else None


def _is_monotonic_nondecr(vals: List[Optional[float]]) -> Optional[bool]:
    xs = [v for v in vals if v is not None]
    if len(xs) < 2:
        return None
    return all(xs[i] <= xs[i + 1] + 1e-9 for i in range(len(xs) - 1))


def _sweep_read(by_arm: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Assemble the per-cell aggregate + the monotonicity read the autopsy
    interpretation grid consumes (reported as DATA; not a pass/fail gate)."""
    cells: Dict[str, Any] = {}
    for kappa in KAPPA_SWEEP:
        for gain in GAIN_SWEEP:
            name = f"ARM_CUE_g{gain:g}_k{kappa:g}"
            runs = by_arm.get(name, [])
            pulls = [r["post_cue_metrics"]["mean_cue_zgoal_pull_norm"] for r in runs]
            post = [r["post_cue_metrics"]["post_cue_approach_rate"] for r in runs]
            bg = [r["post_cue_metrics"]["background_approach_rate"] for r in runs]
            lifts: List[Optional[float]] = []
            for r in runs:
                p = r["post_cue_metrics"]["post_cue_approach_rate"]
                b = r["post_cue_metrics"]["background_approach_rate"]
                lifts.append((float(p) - float(b)) if (p is not None and b is not None) else None)
            cells[name] = {
                "gain": gain, "kappa": kappa,
                "mean_cue_zgoal_pull_norm": _mean(pulls),
                "mean_post_cue_approach_rate": _mean(post),
                "mean_background_approach_rate": _mean(bg),
                "mean_approach_lift_over_background": _mean(lifts),
            }
    # Monotonicity in gain at each fixed kappa (the pre-registered question).
    gain_trend: Dict[str, Any] = {}
    for kappa in KAPPA_SWEEP:
        ordered = [f"ARM_CUE_g{gain:g}_k{kappa:g}" for gain in GAIN_SWEEP]
        pull_series = [cells[n]["mean_cue_zgoal_pull_norm"] for n in ordered]
        lift_series = [cells[n]["mean_approach_lift_over_background"] for n in ordered]
        gain_trend[f"kappa_{kappa:g}"] = {
            "gain_levels": list(GAIN_SWEEP),
            "pull_series": pull_series,
            "approach_lift_series": lift_series,
            "pull_monotonic_nondecreasing_in_gain": _is_monotonic_nondecr(pull_series),
            "approach_lift_monotonic_nondecreasing_in_gain": _is_monotonic_nondecr(lift_series),
        }
    off_runs = by_arm.get("ARM_OFF", [])
    off_ref = {
        "mean_background_approach_rate": _mean(
            [r["post_cue_metrics"]["background_approach_rate"] for r in off_runs]),
        "mean_zgoal_norm_all_steps": _mean(
            [r["post_cue_metrics"]["mean_zgoal_norm_all_steps"] for r in off_runs]),
        "mean_contact_rate": _mean([r["p2_contact_rate"] for r in off_runs]),
    }
    return {"cells": cells, "gain_trend": gain_trend, "arm_off_reference": off_ref}


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    total_eps = (2 + 2 + 5 + 5 + 2) if dry_run else (
        STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + P1_BUDGET + P2_BUDGET)
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        by_arm[arm["name"]] = [
            _run_seed(s, dry_run, arm, total_eps) for s in seeds]

    cue_on_names = [a["name"] for a in ARMS if a["cue"]]

    # C1: cue fires ON across a majority of cue-on cells (>= 2/3 seeds each).
    cell_c1 = []
    for name in cue_on_names:
        runs = by_arm[name]
        cell_c1.append(
            _frac([r["post_cue_metrics"]["n_cue_fire_steps"] > 0 for r in runs])
            >= MIN_FRACTION)
    c1 = _frac(cell_c1) > 0.5

    # C2: post-cue sweep trace exists across a majority of cue-on cells.
    cell_c2 = []
    for name in cue_on_names:
        runs = by_arm[name]
        cell_c2.append(_frac([
            (r["post_cue_metrics"]["n_cue_windows"] > 0
             and r["post_cue_metrics"]["post_cue_approach_rate"] is not None)
            for r in runs
        ]) >= MIN_FRACTION)
    c2 = _frac(cell_c2) > 0.5

    overall = bool(c1 and c2)
    outcome = "PASS" if overall else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] C1_cue_fires_on={c1} C2_sweep_trace_captured={c2} "
          f"outcome={outcome}", flush=True)

    sweep = _sweep_read(by_arm)
    for kappa_key, tr in sweep["gain_trend"].items():
        print(f"[{EXPERIMENT_TYPE}] {kappa_key}: pull_series={tr['pull_series']} "
              f"pull_mono={tr['pull_monotonic_nondecreasing_in_gain']} "
              f"lift_series={tr['approach_lift_series']} "
              f"lift_mono={tr['approach_lift_monotonic_nondecreasing_in_gain']}",
              flush=True)

    return {
        "outcome": outcome,
        "acceptance": {
            "C1_cue_fires_on": c1,
            "C2_sweep_trace_captured": c2,
            "overall_pass": overall,
        },
        "sweep_read": sweep,
        "per_arm": by_arm,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "experiment_type": EXPERIMENT_TYPE, "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS, "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1", "timestamp_utc": ts,
        "outcome": result["outcome"], "evidence_direction": "non_contributory",
        "gates": "V3-EXQ-638b (interoceptive build) -- do NOT build 638b until this sweep routes",
        "predecessor": "V3-EXQ-640 (single-point cue_recall_gain=0.2 measurement; this sweep extends it)",
        "measures": (
            "cue-authority gain sweep: per-cue-fire z_goal pull (cue_zgoal_pull_norm) "
            "and post-cue approach-rate lift over within-run background, across a 2-axis "
            "factorial cue_recall_gain {0.2,1.0,5.0} x incentive_drive_kappa_weight "
            "{2.0,10.0}; behaviourally unchanged 640 ablation + scaffold_post_cue_instrumentation"
        ),
        "substrate": "scaffolded_sd054_onboarding (post-cue instrumentation amend, 2026-06-05)",
        "sweep_axes": {
            "cue_recall_gain": GAIN_SWEEP,
            "incentive_drive_kappa_weight": KAPPA_SWEEP,
            "baseline_cell": {"cue_recall_gain": GAIN_BASELINE,
                              "incentive_drive_kappa_weight": KAPPA_BASELINE,
                              "note": "= the V3-EXQ-640 single-point measurement"},
        },
        "decoupling_note": (
            "Self-contained: every arm sets the landed 634c ARM_3 seeding regime "
            "(drive_floor=0.9 + benefit_threshold=0.02) directly. Every arm sets "
            "scaffold_post_cue_instrumentation=True; each cue-on run records its OWN "
            "within-run background_approach_rate (the matched approach-lift denominator). "
            "ARM_OFF is an additional cue-independent reference. Behaviourally identical "
            "to 640 except the two swept config knobs."
        ),
        "seeding_regime": {"drive_floor": SEED_DRIVE_FLOOR,
                           "benefit_threshold": SEED_BENEFIT_THRESHOLD,
                           "seeding_floor": _seeding_floor()},
        "post_cue_window_steps": POST_CUE_WINDOW_STEPS,
        "pre_registered_question": (
            "Do per-cue-fire cue_zgoal_pull_norm AND post-cue approach-rate lift over "
            "the within-run background rise monotonically as cue_recall_gain (and "
            "incentive-token strength) increase? Reported in sweep_read.gain_trend."
        ),
        "pre_registered_gates": {
            "C1_cue_fires_on": "majority of 6 cue-on cells have n_cue_fire_steps > 0 on >= 2/3 seeds",
            "C2_sweep_trace_captured": "majority of 6 cue-on cells have n_cue_windows > 0 AND post_cue_approach_rate measurable on >= 2/3 seeds",
            "min_fraction": MIN_FRACTION,
            "note": "measurement-success gates only; the autopsy interpretation grid is applied at review (diagnostic, claim_ids=[])",
        },
        "interpretation_grid": {
            "pull_rises_with_gain_AND_approach_lifts": "cue authority real but under-tuned -> pick operating point, proceed to V3-EXQ-638b interoceptive build",
            "pull_rises_with_gain_BUT_approach_flat": "pull reaches z_goal but does not propagate to action selection (E3 goal_proximity / MECH-295 integration gap) -> route to that integration layer, NOT 638b",
            "pull_stays_~0_even_at_high_gain": "cue_pull mechanism itself under-powered / clipped -> substrate_queue entry (cue_pull primitive revisit) becomes appropriate",
        },
        "hard_constraint": "GATES V3-EXQ-638b. Do NOT build the 638b interoceptive need-gating substrate until this sweep routes.",
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _o = str(_res["outcome"]).upper()
        emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                     manifest_path=_res["manifest_path"])
    sys.exit(0 if _res["outcome"] == "PASS" else 1)
