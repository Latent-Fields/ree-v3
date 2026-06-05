"""
V3-EXQ-640: SD-057 cue-recall -- post-cue ACTION / GRADIENT measurement-only
diagnostic. Re-runs the V3-EXQ-638a ARM_OFF / ARM_CUE_ON ablation BEHAVIOURALLY
UNCHANGED and adds per-cue-fire instrumentation only. Routed by
failure_autopsy_V3-EXQ-638a_2026-06-05 (Sections 7-8): 638a settled the cue
"fires vs lifts contact" question at C3 FAIL (ARM_CUE_ON contact <= ARM_OFF on
all 3 seeds) but CANNOT discriminate WHY -- it logged no post-cue action trace,
so it cannot tell cue-to-action AUTHORITY / displacement / gradient-following /
hazard-interrupt apart. 640 is the smallest next step the autopsy prescribes: a
measurement-only post-cue diagnostic that GATES the planned 638b interoceptive
build.

WHAT CHANGES FROM 638a: nothing behavioural. Both arms additionally set
scaffold_post_cue_instrumentation=True, so scaffolded_sd054_onboarding._eval_episode
populates a read-only post_cue_diag accumulator. No new substrate primitive, no
new env knob, no tuning. The agent senses / selects / steps identically; only the
measurement surface grows.

PER-CUE-FIRE MEASUREMENTS (P2, the measurement phase), windowed over the next
scaffold_post_cue_window_steps moves after each cue fire:
  - z_goal NORM delta around the cue (||z_goal||_after - ||z_goal||_before): the
    DISPLACEMENT test. mean < 0 => the cue pulls z_goal toward a WEAKER token
    (displacement), not authority.
  - z_goal PULL magnitude (||z_goal_after - z_goal_before||): does the cue move
    z_goal at all?
  - absolute ||z_goal|| at the cue fire vs ARM_OFF wild-seeded attractor norm
    (LOWER under cue => displacement signature, per the 638a autopsy Section 3).
  - post-cue selected-action APPROACH rate (fraction of post-cue moves that
    reduce manhattan distance to the nearest resource) vs the background rate.
  - first-gradient-improving-move latency + frac windows whose FIRST move
    approaches (immediate authority).
  - hazard-salience-interrupt count within the post-cue window.
  - oscillation (action-direction reversal) count within the post-cue window.

DISCRIMINATOR GRID (applied at review; this experiment MEASURES, it does not
adjudicate -- experiment_purpose=diagnostic, claim_ids=[] => weights no
governance score). See failure_autopsy_V3-EXQ-638a Section 8:
  cue fires, ~0 (or negative) z_goal/action delta -> cue-to-action authority
    missing / displacement -> wire/strengthen cue->E3 authority BEFORE 638b.
  cue fires, action delta but no gradient improvement -> gradient-following /
    4-dir axis-decomposition missing -> safe-gradient diagnostic.
  cue fires, gradient improves then hazard interrupt aborts -> interrupt-without-
    resume -> goal-persistence-across-salience-switch diagnostic.
  cue fires, gradient improves, no interrupt, still no contact -> persistence /
    reorientation -> orienting/surveying diagnostic.
  cue fires, gradient improves, contact lifts -> (contradicts 638a) -> promote
    the authority bridge; only THEN consider 638b interoceptive.

ARMS (3 seeds 42/43/44 each; identical to 638a; both instrumented):
  ARM_OFF      cue-recall bridge OFF. No cues fire (n_cue_fire_steps=0), but the
               post_cue_diag background stats (approach rate, mean ||z_goal||
               over all steps) ARE recorded -- they are the BASELINE the
               displacement / authority-lift comparisons need.
  ARM_CUE_ON   cue-recall bridge ON + scaffold_stage0_bind_incentive_token=True
               (the 638 formation fix). Cues fire; the post-cue windows are
               populated.

SELF-CONTAINED + DECOUPLED FROM the stalled V3-EXQ-634c RUN: both arms set the
landed 634c ARM_3 seeding regime (drive_floor=0.9 + benefit_threshold=0.02)
directly, exactly as 638a did.

ACCEPTANCE (measurement-success, NOT a scientific pass; diagnostic weights no
claim):
  C1 (cue fires ON):  ARM_CUE_ON P2 post_cue_diag n_cue_fire_steps > 0 on >= 2/3
                      seeds -- there are cue events to measure (expected; 638a
                      already showed C1 holds).
  C2 (action trace):  ARM_CUE_ON P2 n_cue_windows > 0 on >= 2/3 seeds AND the
                      post-cue move-eval denominator n_postcue_eval_steps > 0 --
                      the post-cue action/gradient trace the 638a autopsy said
                      was MISSING now exists, so the discriminator grid is
                      computable.
  overall_pass = C1 AND C2. PASS => the measurement succeeded and the 638b-gating
  read can be made at review. A FAIL means the instrumentation captured no usable
  post-cue trace -> /diagnose-errors on the harness wiring, NOT a substrate
  finding.

claim_ids = []  (measurement-only behavioural diagnostic; NOT governance evidence).
experiment_purpose = "diagnostic".

References:
- REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-638a_2026-06-05.{md,json}
- ree-v3/experiments/scaffolded_sd054_onboarding.py (post-cue instrumentation,
  2026-06-05: scaffold_post_cue_instrumentation flag + _new_post_cue_diag())
- V3-EXQ-638a (the FAIL whose un-discriminated branch this measures)
- V3-EXQ-634c (seeding-calibration regime reused)
- REE_assembly/evidence/planning/thought_intake_2026-06-04_cue_ecology_weaning_nursery_to_forager.md
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

EXPERIMENT_TYPE = "v3_exq_640_scaffold_cue_postcue_action_gradient_diagnostic"
QUEUE_ID = "V3-EXQ-640"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Curriculum budgets (mirror 634c / 638 / 638a -- behaviourally unchanged).
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

ARMS: List[Dict[str, Any]] = [
    {"name": "ARM_OFF", "cue": False},
    {"name": "ARM_CUE_ON", "cue": True},
]


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
        # Shared ARM_3 seeding regime (both arms).
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_contact_gating_benefit_threshold=_seeding_floor(),
        # V3-EXQ-640 measurement: instrument the post-cue window in BOTH arms.
        # ARM_OFF fires no cue but supplies the background baselines the
        # displacement / authority-lift comparisons need.
        scaffold_post_cue_instrumentation=True,
        scaffold_post_cue_window_steps=POST_CUE_WINDOW_STEPS,
    )
    if cue_on:
        cfg.scaffold_cue_recall_bridge_enabled = True
        cfg.scaffold_cue_n_resource_types = N_RESOURCE_TYPES
        # Formation fix (same as 638a): bind a per-type token at Stage-0 forced
        # feeding so the bank is non-empty entering P1/P2 and the cue can fire.
        cfg.scaffold_stage0_bind_incentive_token = True
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, cue_on: bool) -> REEConfig:
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
        # SD-057: cue-recall bridge agent flags (ON arm only).
        use_incentive_token_bank=cue_on,
        use_cue_recall=cue_on,
        cue_recall_gain=0.2,
    )
    if cue_on:
        cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    return cfg


def _derive_post_cue_metrics(pc: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the raw post_cue_diag sums/counts into the interpretable rates the
    638a-autopsy discriminator grid reads. Safe on empty (ARM_OFF / no fires)."""
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


def _run_seed(seed: int, dry_run: bool, arm: Dict[str, Any]) -> Dict[str, Any]:
    arm_name, cue_on = arm["name"], bool(arm["cue"])
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, cue_on)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env, cue_on)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {arm_name}", flush=True)
    total_eps = (2 + 2 + 5 + 5 + 2) if dry_run else (
        STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + P1_BUDGET + P2_BUDGET)

    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    token_bank = int(getattr(s0, "token_bank_size_end", 0))
    print(f"  [train] stage0 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed} "
          f"token_bank_size_end={token_bank}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=stage0", flush=True)
        return _abort(seed, arm_name, "stage0")

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"retention={s0b.retention_ratio:.3f} gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=stage0b", flush=True)
        return _abort(seed, arm_name, "stage0b")

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=p0", flush=True)
        return _abort(seed, arm_name, "p0")

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
          f"contact_rate={p2.contact_rate:.4f} cue_fires={cue_fires} "
          f"pc_fire_steps={pc_metrics['n_cue_fire_steps']} "
          f"pc_windows={pc_metrics['n_cue_windows']} "
          f"post_cue_approach={pc_metrics['post_cue_approach_rate']} "
          f"bg_approach={pc_metrics['background_approach_rate']} "
          f"zgoal_delta={pc_metrics['mean_post_cue_zgoal_norm_delta']}", flush=True)

    print(f"verdict: seed={seed} arm={arm_name} pc_fire_steps={pc_metrics['n_cue_fire_steps']} "
          f"pc_windows={pc_metrics['n_cue_windows']}", flush=True)
    return {
        "seed": seed, "arm": arm_name, "aborted_at": None,
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_num_contact_events": int(p2.num_contact_events),
        "p2_n_cue_recall_fires": cue_fires,
        "post_cue_metrics": pc_metrics,
        "post_cue_diag_raw": pc_raw,
    }


def _abort(seed: int, arm_name: str, stage: str) -> Dict[str, Any]:
    return {
        "seed": seed, "arm": arm_name, "aborted_at": stage,
        "p1_survival_pass": False, "p2_contact_rate": 0.0,
        "p2_num_contact_events": 0, "p2_n_cue_recall_fires": 0,
        "post_cue_metrics": _derive_post_cue_metrics(_new_post_cue_diag_empty()),
        "post_cue_diag_raw": _new_post_cue_diag_empty(),
    }


def _new_post_cue_diag_empty() -> Dict[str, Any]:
    from experiments.scaffolded_sd054_onboarding import _new_post_cue_diag
    return _sanitize_pc(_new_post_cue_diag(POST_CUE_WINDOW_STEPS))


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        by_arm[arm["name"]] = [_run_seed(s, dry_run, arm) for s in seeds]

    on = by_arm["ARM_CUE_ON"]

    # C1: ARM_CUE_ON cue fires (post_cue_diag saw cue events to measure).
    c1 = _frac([r["post_cue_metrics"]["n_cue_fire_steps"] > 0 for r in on]) >= MIN_FRACTION
    # C2: post-cue ACTION TRACE exists (the thing 638a was missing) -- windows AND
    # a non-empty post-cue move-eval denominator.
    c2 = _frac([
        (r["post_cue_metrics"]["n_cue_windows"] > 0
         and r["post_cue_metrics"]["post_cue_approach_rate"] is not None)
        for r in on
    ]) >= MIN_FRACTION

    overall = bool(c1 and c2)
    outcome = "PASS" if overall else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] C1_cue_fires_on={c1} C2_postcue_trace_captured={c2} "
          f"outcome={outcome}", flush=True)

    return {
        "outcome": outcome,
        "acceptance": {
            "C1_cue_fires_on": c1,
            "C2_postcue_trace_captured": c2,
            "overall_pass": overall,
        },
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
        "gates": "V3-EXQ-638b (interoceptive build) -- 640 measures which branch 638a could not discriminate",
        "measures": (
            "post-cue action/gradient trace (behaviourally unchanged 638a ablation "
            "+ scaffold_post_cue_instrumentation=True): per-cue-fire z_goal norm/pull "
            "delta, approach rate vs background, gradient-improving-move latency, "
            "hazard-interrupt rate, oscillation"
        ),
        "substrate": "scaffolded_sd054_onboarding (post-cue instrumentation amend, 2026-06-05)",
        "decoupling_note": (
            "Self-contained: both arms set the landed 634c ARM_3 seeding regime "
            "(drive_floor=0.9 + benefit_threshold=0.02) directly. Both arms also set "
            "scaffold_post_cue_instrumentation=True -- ARM_OFF fires no cue but supplies "
            "the background baselines (approach rate, mean ||z_goal||) the displacement / "
            "authority-lift comparisons read against. Behaviourally identical to 638a."
        ),
        "seeding_regime": {"drive_floor": SEED_DRIVE_FLOOR,
                           "benefit_threshold": SEED_BENEFIT_THRESHOLD,
                           "seeding_floor": _seeding_floor()},
        "post_cue_window_steps": POST_CUE_WINDOW_STEPS,
        "pre_registered_gates": {
            "C1_cue_fires_on": "ARM_CUE_ON P2 post_cue n_cue_fire_steps > 0 on >= 2/3 seeds",
            "C2_postcue_trace_captured": "ARM_CUE_ON P2 n_cue_windows > 0 AND post_cue_approach_rate measurable on >= 2/3 seeds",
            "min_fraction": MIN_FRACTION,
            "note": "measurement-success gates only; the 638a-autopsy discriminator grid is applied at review (diagnostic, claim_ids=[])",
        },
        "discriminator_grid": {
            "cue_fires_~0_or_negative_zgoal_action_delta": "cue-to-action authority missing / displacement -> wire/strengthen cue->E3 authority BEFORE 638b",
            "cue_fires_action_delta_no_gradient_improvement": "gradient-following / 4-dir axis-decomposition missing -> safe-gradient diagnostic",
            "cue_fires_gradient_improves_then_hazard_interrupt": "interrupt-without-resume -> goal-persistence-across-salience-switch diagnostic",
            "cue_fires_gradient_improves_no_interrupt_no_contact": "persistence / reorientation -> orienting/surveying diagnostic",
            "cue_fires_gradient_improves_contact_lifts": "(contradicts 638a) -> promote authority bridge; only THEN consider 638b interoceptive",
        },
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
