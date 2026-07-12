"""
V3-EXQ-640b: SD-057 cue-recall -- cue-AUTHORITY GAIN SWEEP, CLEAN EVIDENCE
RETEST on the VALIDATED modulatory-bias-selection-authority substrate.

WHY THIS RE-RUN (the substrate that was missing is now active):
  The prior sweep V3-EXQ-640a (ran 2026-06-06T01:36Z, ~21h BEFORE the
  substrate was validated) FAILed to show any post-cue approach lift: across
  the full cue_recall_gain{0.2,1.0,5.0} x kappa{2.0,10.0} sweep the post-cue
  approach-lift over within-run background was flat/negative
  (-0.006 .. -0.00001) despite z_goal forming at the cue (~0.5 vs 0.42 OFF)
  and the cue firing every step. The 640a autopsy routing grid read this as
  "z_goal pull reaches the attractor but does NOT propagate to action
  selection" -- i.e. the cue's MECH-295 approach bias had NO AUTHORITY over
  the committed E3.select argmin. That diagnosis is now mechanistically
  explained: the modulatory score-bias channels (incl. MECH-295) added a
  small fixed magnitude (~0.05) to primary E3 scores whose raw range was much
  larger, so the argmin never moved (the 604a/624a/614d "drowning" cluster).
  The modulatory-bias-selection-authority substrate (gap-relative E3.select
  authority) was implemented 2026-06-03 and VALIDATED by V3-EXQ-643a PASS
  (2026-06-06T22:29Z, after the float32 catastrophic-cancellation fix). With
  use_modulatory_selection_authority=True the combined modulatory contribution
  is rescaled so its range == modulatory_authority_gain * raw_score_range,
  giving the cue-driven approach bias GENUINE BUT BOUNDED authority over the
  argmin. The substrate target records the exact behavioural acceptance this
  experiment now tests: "post-cue approach-lift > 0 over within-run background
  on >= 2/3 seeds in a contact-making env once modulatory selection authority
  is active." The authority is now active.

WHAT CHANGED FROM 640a (and ONLY this):
  - use_modulatory_selection_authority=True + modulatory_authority_gain=0.5
    is set on EVERY arm (the validated substrate operating mode; the OFF arm
    is cue-OFF but authority-ON, so the only between-arm difference stays the
    cue-recall bridge -- the within-run background denominator already
    reflects the authority-amplified wild-goal approach baseline, so the lift
    isolates the cue contribution).
  - PROMOTED from a measurement-only diagnostic (640a: claim_ids=[],
    experiment_purpose="diagnostic") to EVIDENCE: claim_ids carry the real
    cue-authority lineage SD-057 / MECH-346 / MECH-347 and the acceptance is
    the scientific lift criterion, not just measurement-success.
  - NO SD-056 online contrastive training in this harness (the 643 score
    explosion that motivated the float32 fix + rollout clamp does NOT occur
    here: scaffold P2 is frozen-policy eval, primary scores stay
    normal-magnitude, the authority gate fires on the real ~0.05-0.2
    modulatory range). No rollout-norm clamp needed.

CLAIM TAGGING (all three v3_pending; the single cue-on arm legitimately
exercises a SERIAL DEPENDENCY chain whose behavioural readout is the post-cue
approach lift -- no single claim is separable from the others in this
readout, so they co-tag with a shared direction):
  - MECH-347 (L6 cue-recall): a perceived cue type retrieves its SD-057
    incentive token and cue_pulls z_goal toward the object's stored embedding
    BEFORE consumption (cfg.use_cue_recall=True; scaffold_cue_recall_bridge).
  - MECH-346 (L4 token pointer; amends MECH-230): z_goal is seeded FROM the
    most-wanted object's stored embedding via the incentive-token pointer
    (cfg.use_incentive_token_bank=True), so the cue's pull target IS the
    object-bound pointer.
  - SD-057 (the object-bound incentive-salience bank L2-L4 + L6): the bank
    binds + holds + serves the per-object token the cue retrieves.
  A positive post-cue approach lift supports all three jointly (the chain
  cue-recall -> token-pointer seed -> bank fired AND propagated to action);
  a null lift weakens the joint chain (cannot be attributed to one link).

PRE-REGISTERED ACCEPTANCE (this is now a SCIENTIFIC pass, not measurement-only):
  PRIMARY  C_LIFT_PRIMARY: in the PRE-REGISTERED decisive operating-point cell
    (cue_recall_gain=5.0, kappa=10.0 -- the maximal-authority cell where the
    cue_pull reaches its clamp ceiling so z_goal snaps onto z_object, the
    strongest test of whether the now-authorised approach bias bites), the
    per-cell post_cue_approach_lift = post_cue_approach_rate -
    within-run background_approach_rate is > 0 on >= 2/3 seeds.
  GUARDS (measurement-success, retained from 640a so a no-trace run is an
    INSTRUMENTATION failure -- routed to /diagnose-errors -- NOT evidence
    against the claim):
    C1 cue fires ON across a majority of the 6 cue-on cells (>= 2/3 seeds each).
    C2 a measurable post-cue approach trace exists across a majority of cells.
  overall_pass = C1 AND C2 AND C_LIFT_PRIMARY.
  evidence_direction: if NOT (C1 AND C2) -> "non_contributory" (instrumentation;
    per-claim "unknown"); elif C_LIFT_PRIMARY -> "supports"; else -> "weakens".
  SECONDARY (reported, does NOT drive the verdict; carries an explicit
    multiple-comparison caveat): C_LIFT_ANY = at least one cue-on cell clears
    the lift bar on >= 2/3 seeds, the full per-cell lift grid, and the
    monotonicity-in-gain read.

ARMS (3 seeds 42/43/44 each; all instrumented; authority ON on all):
  ARM_OFF                cue-recall bridge OFF (authority ON). No cues fire;
                         supplies the cue-independent background baselines +
                         the wild-seeded ||z_goal|| attractor norm. Reference.
  ARM_CUE_g{G}_k{K}      cue-recall bridge ON + scaffold_stage0_bind_incentive_token
                         =True (the 638 formation fix), with cue_recall_gain=G
                         and incentive_drive_kappa_weight=K. 6 cells over
                         G in {0.2,1.0,5.0} x K in {2.0,10.0}. Each cue-on run
                         records its OWN within-run background_approach_rate
                         (the matched approach-lift denominator).

SELF-CONTAINED + DECOUPLED: every arm sets the landed 634c ARM_3 seeding
regime (drive_floor=0.9 + benefit_threshold=0.02) directly, exactly as
640/640a did, so wild contact seeds in BOTH arms in the contact-making
scaffold P2 env (scaffold_p2_hazard_food_attraction_guard=0.3). Does NOT
depend on any stalled run -- only the landed substrate code.

References:
- REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (the validated substrate this retest activates; V3-EXQ-643a fix section)
- V3-EXQ-640a (the pre-substrate sweep this supersedes; flat-lift FAIL)
- failure_autopsy_V3-EXQ-640_2026-06-05.{md,json} (Section 7 routing grid)
- ree-v3/experiments/scaffolded_sd054_onboarding.py (post-cue instrumentation)
- ree-v3 goal.py GoalState.cue_pull + IncentiveTokenBank; agent.py
  cue_recall_wanting (MECH-347 L6); cue_recall_gain + incentive_drive_kappa_weight
- claims.yaml SD-057 / MECH-346 / MECH-347
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_640b_scaffold_cue_authority_gain_sweep_retest"
QUEUE_ID = "V3-EXQ-640b"
SUPERSEDES = "V3-EXQ-640a"
CLAIM_IDS: List[str] = ["SD-057", "MECH-346", "MECH-347"]
EXPERIMENT_PURPOSE = "evidence"
SEEDS = [42, 43, 44]

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Curriculum budgets (mirror 634c / 638 / 638a / 640 / 640a -- behaviourally unchanged).
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

# --- VALIDATED SUBSTRATE: modulatory-bias-selection-authority (V3-EXQ-643a PASS). ---
# ON for EVERY arm. This is the single change vs 640a that makes the cue's
# MECH-295 approach bias capable of moving the committed E3.select argmin.
USE_MODULATORY_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5  # substrate default; bounded (< 1.0) so a clearly-
                                 # harmful candidate stays rejected (primary gap dominates).

# --- The two sweep axes (the cue-authority question). ---
GAIN_SWEEP = [0.2, 1.0, 5.0]      # cue_recall_gain: 640 baseline -> 5x -> 25x.
KAPPA_SWEEP = [2.0, 10.0]         # incentive_drive_kappa_weight: token-strength axis.
GAIN_BASELINE = 0.2
KAPPA_BASELINE = 2.0
# Pre-registered decisive operating-point cell for the PRIMARY lift criterion.
PRIMARY_GAIN = 5.0
PRIMARY_KAPPA = 10.0


def _primary_cell_name() -> str:
    return f"ARM_CUE_g{PRIMARY_GAIN:g}_k{PRIMARY_KAPPA:g}"


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
        # Formation fix (same as 638a / 640 / 640a): bind a per-type token at
        # Stage-0 forced feeding so the bank is non-empty entering P1/P2.
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
        # VALIDATED SUBSTRATE (the single change vs 640a): give the composed
        # modulatory score-bias chain (incl. the MECH-295 cue->approach bias)
        # gap-relative authority over the committed E3.select argmin. ON for all
        # arms; the cue on/off + gain/kappa axis is the only between-arm difference.
        use_modulatory_selection_authority=USE_MODULATORY_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
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
    acceptance criteria + autopsy interpretation grid read. Safe on empty
    (ARM_OFF / no fires)."""
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
        # AUTHORITY / gradient-following test -- the LOAD-BEARING evidence metric.
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


def _seed_lifts(runs: List[Dict[str, Any]]) -> List[Optional[float]]:
    """Per-seed post_cue_approach_lift = post_cue_approach_rate - background, or
    None when either rate is unmeasurable for that seed."""
    lifts: List[Optional[float]] = []
    for r in runs:
        p = r["post_cue_metrics"]["post_cue_approach_rate"]
        b = r["post_cue_metrics"]["background_approach_rate"]
        lifts.append((float(p) - float(b)) if (p is not None and b is not None) else None)
    return lifts


def _cell_lift_pass(runs: List[Dict[str, Any]]) -> bool:
    """A cell passes the lift bar when post_cue_approach_lift > 0 on >= 2/3 seeds."""
    lifts = _seed_lifts(runs)
    return _frac([(L is not None and L > 0.0) for L in lifts]) >= MIN_FRACTION


def _sweep_read(by_arm: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Assemble the per-cell aggregate + monotonicity read + per-cell lift-pass
    table (reported as DATA + secondary acceptance context)."""
    cells: Dict[str, Any] = {}
    for kappa in KAPPA_SWEEP:
        for gain in GAIN_SWEEP:
            name = f"ARM_CUE_g{gain:g}_k{kappa:g}"
            runs = by_arm.get(name, [])
            pulls = [r["post_cue_metrics"]["mean_cue_zgoal_pull_norm"] for r in runs]
            post = [r["post_cue_metrics"]["post_cue_approach_rate"] for r in runs]
            bg = [r["post_cue_metrics"]["background_approach_rate"] for r in runs]
            lifts = _seed_lifts(runs)
            cells[name] = {
                "gain": gain, "kappa": kappa,
                "mean_cue_zgoal_pull_norm": _mean(pulls),
                "mean_post_cue_approach_rate": _mean(post),
                "mean_background_approach_rate": _mean(bg),
                "mean_approach_lift_over_background": _mean(lifts),
                "per_seed_approach_lift": lifts,
                "n_seeds_lift_positive": int(
                    sum(1 for L in lifts if L is not None and L > 0.0)),
                "n_seeds_measured": int(sum(1 for L in lifts if L is not None)),
                "lift_pass_ge_2of3": _cell_lift_pass(runs),
            }
    # Monotonicity in gain at each fixed kappa (the pre-registered secondary question).
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
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) "
          f"modulatory_authority={USE_MODULATORY_AUTHORITY} "
          f"gain={MODULATORY_AUTHORITY_GAIN}", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    total_eps = (2 + 2 + 5 + 5 + 2) if dry_run else (
        STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + P1_BUDGET + P2_BUDGET)
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        by_arm[arm["name"]] = [
            _run_seed(s, dry_run, arm, total_eps) for s in seeds]

    cue_on_names = [a["name"] for a in ARMS if a["cue"]]

    # GUARD C1: cue fires ON across a majority of cue-on cells (>= 2/3 seeds each).
    cell_c1 = []
    for name in cue_on_names:
        runs = by_arm[name]
        cell_c1.append(
            _frac([r["post_cue_metrics"]["n_cue_fire_steps"] > 0 for r in runs])
            >= MIN_FRACTION)
    c1 = _frac(cell_c1) > 0.5

    # GUARD C2: post-cue approach trace exists across a majority of cue-on cells.
    cell_c2 = []
    for name in cue_on_names:
        runs = by_arm[name]
        cell_c2.append(_frac([
            (r["post_cue_metrics"]["n_cue_windows"] > 0
             and r["post_cue_metrics"]["post_cue_approach_rate"] is not None)
            for r in runs
        ]) >= MIN_FRACTION)
    c2 = _frac(cell_c2) > 0.5

    sweep = _sweep_read(by_arm)

    # PRIMARY evidence criterion: pre-registered decisive cell lift > 0 on >= 2/3 seeds.
    primary_name = _primary_cell_name()
    primary_runs = by_arm.get(primary_name, [])
    c_lift_primary = _cell_lift_pass(primary_runs) if primary_runs else False

    # SECONDARY (reported, NOT verdict-driving; carries a multiple-comparison caveat):
    # existence across the 6 cue-on cells.
    c_lift_any = any(sweep["cells"][n]["lift_pass_ge_2of3"] for n in cue_on_names)
    cells_passing_lift = [n for n in cue_on_names
                          if sweep["cells"][n]["lift_pass_ge_2of3"]]

    measurement_ok = bool(c1 and c2)
    overall = bool(measurement_ok and c_lift_primary)
    outcome = "PASS" if overall else "FAIL"

    if not measurement_ok:
        evidence_direction = "non_contributory"   # instrumentation -> /diagnose-errors
        per_claim_dir = "unknown"
    elif c_lift_primary:
        evidence_direction = "supports"
        per_claim_dir = "supports"
    else:
        evidence_direction = "weakens"
        per_claim_dir = "weakens"
    evidence_direction_per_claim = {cid: per_claim_dir for cid in CLAIM_IDS}

    print(f"[{EXPERIMENT_TYPE}] C1_cue_fires_on={c1} C2_trace_captured={c2} "
          f"C_LIFT_PRIMARY({primary_name})={c_lift_primary} "
          f"C_LIFT_ANY={c_lift_any} cells_passing_lift={cells_passing_lift} "
          f"outcome={outcome} evidence_direction={evidence_direction}", flush=True)
    for kappa_key, tr in sweep["gain_trend"].items():
        print(f"[{EXPERIMENT_TYPE}] {kappa_key}: pull_series={tr['pull_series']} "
              f"lift_series={tr['approach_lift_series']} "
              f"lift_mono={tr['approach_lift_monotonic_nondecreasing_in_gain']}",
              flush=True)

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "acceptance": {
            "C1_cue_fires_on": c1,
            "C2_trace_captured": c2,
            "measurement_ok": measurement_ok,
            "C_LIFT_PRIMARY": c_lift_primary,
            "primary_cell": primary_name,
            "primary_cell_per_seed_lift": _seed_lifts(primary_runs),
            "overall_pass": overall,
            "secondary_C_LIFT_ANY": c_lift_any,
            "secondary_cells_passing_lift": cells_passing_lift,
            "secondary_caveat": (
                "C_LIFT_ANY is an existence-across-6-cells read with an inflated "
                "false-positive rate; it does NOT drive the verdict. The verdict "
                "is the single pre-registered C_LIFT_PRIMARY cell."
            ),
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
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "supersedes": SUPERSEDES,
        "predecessor": (
            "V3-EXQ-640a (pre-substrate cue-authority gain sweep; flat/negative "
            "post-cue approach-lift across the full grid because the modulatory "
            "approach bias had no authority over the E3.select argmin)"
        ),
        "substrate_validated": (
            "modulatory-bias-selection-authority (gap-relative E3.select authority); "
            "VALIDATED by V3-EXQ-643a PASS 2026-06-06T22:29Z after the float32 "
            "catastrophic-cancellation fix; substrate_queue status=implemented ready=true"
        ),
        "measures": (
            "cue-authority gain sweep, EVIDENCE retest on the validated modulatory "
            "selection-authority substrate: per-cue-fire z_goal pull and post-cue "
            "approach-rate lift over within-run background, across cue_recall_gain "
            "{0.2,1.0,5.0} x incentive_drive_kappa_weight {2.0,10.0}; "
            "use_modulatory_selection_authority=True on every arm"
        ),
        "substrate": "scaffolded_sd054_onboarding (post-cue instrumentation amend, 2026-06-05)",
        "modulatory_authority": {
            "use_modulatory_selection_authority": USE_MODULATORY_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "applied_to": "all arms (substrate operating mode)",
        },
        "sweep_axes": {
            "cue_recall_gain": GAIN_SWEEP,
            "incentive_drive_kappa_weight": KAPPA_SWEEP,
        },
        "claim_tagging_rationale": (
            "The single cue-on arm exercises a SERIAL dependency: MECH-347 L6 "
            "cue-recall retrieves the SD-057 incentive token and cue_pulls z_goal "
            "toward the MECH-346 L4 object-bound token pointer; the post-cue "
            "approach lift is the behavioural readout of the whole chain firing AND "
            "propagating to action under the now-authorised modulatory bias. No "
            "single link is separable in this readout, so all three co-tag with a "
            "shared evidence_direction."
        ),
        "decoupling_note": (
            "Self-contained: every arm sets the landed 634c ARM_3 seeding regime "
            "(drive_floor=0.9 + benefit_threshold=0.02) directly + "
            "scaffold_post_cue_instrumentation=True; each cue-on run records its OWN "
            "within-run background_approach_rate (the matched approach-lift denominator). "
            "ARM_OFF is an additional cue-independent reference. NO SD-056 online "
            "contrastive training (scaffold P2 frozen-policy eval) -> primary scores "
            "stay normal-magnitude -> the authority gate fires on the real modulatory "
            "range; no rollout-norm clamp needed."
        ),
        "seeding_regime": {"drive_floor": SEED_DRIVE_FLOOR,
                           "benefit_threshold": SEED_BENEFIT_THRESHOLD,
                           "seeding_floor": _seeding_floor()},
        "post_cue_window_steps": POST_CUE_WINDOW_STEPS,
        "pre_registered_acceptance": {
            "PRIMARY_C_LIFT_PRIMARY": (
                f"post_cue_approach_lift > 0 on >= 2/3 seeds in the pre-registered "
                f"decisive cell ({_primary_cell_name()}: cue_recall_gain={PRIMARY_GAIN:g}, "
                f"kappa={PRIMARY_KAPPA:g} -- maximal cue_pull / z_goal-snap)"
            ),
            "GUARD_C1_cue_fires_on": "majority of 6 cue-on cells have n_cue_fire_steps > 0 on >= 2/3 seeds",
            "GUARD_C2_trace_captured": "majority of 6 cue-on cells have n_cue_windows > 0 AND post_cue_approach_rate measurable on >= 2/3 seeds",
            "overall_pass": "C1 AND C2 AND C_LIFT_PRIMARY",
            "min_fraction": MIN_FRACTION,
            "evidence_direction_rule": (
                "NOT(C1 AND C2) -> non_contributory (instrumentation, route /diagnose-errors, "
                "per-claim unknown); elif C_LIFT_PRIMARY -> supports; else -> weakens"
            ),
            "secondary_note": "C_LIFT_ANY (existence across cells) + monotonicity reported as context only",
        },
        "interpretation_grid": {
            "lift_positive_at_primary": "cue->z_goal->approach now propagates with authority -> supports SD-057/MECH-346/MECH-347; pick operating point, proceed to V3-EXQ-638b interoceptive build",
            "lift_null_at_primary_but_trace_ok": "authority did not rescue propagation despite maximal cue_pull -> weakens the cue-authority chain; re-route to the z_goal->approach integration layer (MECH-295 / E3 goal_proximity) NOT a cue_pull fix",
            "no_trace": "instrumentation FAIL (C1/C2) -> /diagnose-errors on the harness wiring, NOT a substrate finding",
        },
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
    print(f"Result written to: {out_path}", flush=True)
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
