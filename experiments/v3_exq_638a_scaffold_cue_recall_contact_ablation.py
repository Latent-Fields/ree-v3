"""
V3-EXQ-638a: SD-057 cue-recall bridge -- foraging contact-rate ablation, on the
FORMATION-FIXED substrate (validation re-issue of V3-EXQ-638).

638 FAILed cue-silent: C1_cue_fires_on=false. Root cause (code-confirmed, see
scaffolded_sd054_onboarding amend 2026-06-04b, commit a9ef0be): the
IncentiveTokenBank was EMPTY entering P1/P2 because Stage-0 forced feeding bound
no token -- it passed rt=_contacted_resource_type(obs), which is almost always
None since forced feeding is decoupled from standing on a typed cell, so the L2
bank.update bind (gated resource_type>0) was never reached -> cue_recall_wanting
returns 0 at `k not in bank._base_value` -> cue_fires=0 (chicken-and-egg: the cue
needed a token from contact it was meant to bootstrap). A bare `except: pass` made
the zero undiagnosable.

638a is the validation of the fix. The ONLY change from 638: the cue-ON arm sets
scaffold_stage0_bind_incentive_token=True, so Stage-0 forced feeding binds the
token to the STRONGEST-PERCEIVED type each step (identical perception to the wild
cue) -> the bank is non-empty entering P1/P2 -> the cue can match a token.
Activation smoke (2026-06-04): Stage-0 token_bank_size_end 0->2; P1 cue_fires 0->34.

It also SURFACES the new cue diagnostics in the manifest (token_bank_size,
n_external_cues_seen / n_token_matches / n_cue_recall_attempts / n_cue_recall_fires,
best_prox_peak, drive_peak, matched_token_strength_peak, cue_nonfire_reason_counts)
so the result is fully ATTRIBUTABLE -- 638a must show WHY (bank populated, cue
fires), not just a number.

NOT a supersession of 638: 638's cue-silent evidence is VALID and is the failure
record this fixes. 638a re-runs the same ablation to test whether the now-firing
cue actually LIFTS foraging contact.

SELF-CONTAINED + DECOUPLED FROM the stalled V3-EXQ-634c RUN: both arms set the
landed 634c seeding-calibration knobs (drive_floor=0.9 + benefit_threshold=0.02,
the ARM_3 regime) directly, so wild contact actually seeds z_goal. The differences
between arms are the cue-recall bridge AND the Stage-0 token-binding formation fix
(both bundled into the cue-ON arm: a cue that cannot fire is not a cue).

ARMS (3 seeds 42/43/44 each; identical seeding regime; same curriculum as 634c):
  ARM_OFF      cue-recall bridge OFF (legacy scaffold). Empty bank, cue silent.
               Baseline contact rate.
  ARM_CUE_ON   cue-recall bridge ON + scaffold_stage0_bind_incentive_token=True:
               agent built with use_incentive_token_bank + use_cue_recall +
               use_resource_encoder; scaffold envs SD-049-enabled; Stage-0 binds
               per-type tokens; cue_recall_wanting fires on the strongest-perceived
               resource each P1/P2 step.

Pre-registered acceptance criteria:
  C1 (cue fires ON):   ARM_CUE_ON P2 n_cue_recall_fires > 0 on >= 2/3 seeds.
                       Expected to PASS now (was the 638 failure).
  C2 (cue silent OFF):  ARM_OFF P2 n_cue_recall_fires == 0 on ALL seeds (no bank).
  C3 (contact lift):    ARM_CUE_ON P2 contact_rate > ARM_OFF P2 contact_rate
                        (per matched seed) on >= 2/3 seeds -- the load-bearing
                        foraging-contact lever test. The OPEN question.
  C4 (no survival regression): ARM_CUE_ON P1 survival pass-rate >= ARM_OFF
                        survival pass-rate. Reported; informational guard.
  overall_pass = C1 AND C2 AND C3 (C4 reported, not gating).

Interpretation grid (routes the next move; see thought_intake_2026-06-04_cue_
ecology_weaning_nursery_to_forager.md for the two-layer framing):
  (a) C1 PASS + C3 PASS -> the FORMATION gap was the whole story: the cue bridge
      is a working foraging-contact lever. Route to a combined scaffold readiness
      run + the GAP-7 L9 retest.
  (b) C1 PASS + C3 FAIL -> the cue FIRES but does not lift contact: it has no
      behavioural AUTHORITY. This is the evidence for the LAYER-2 interoceptive
      need-gating bridge (bind body/drive state to external affordance). Route to
      the interoceptive-coupling /implement-substrate pass + V3-EXQ-638b
      (OFF / EXTERNAL_ONLY / INTEROCEPTIVE+EXTERNAL arms). NB: in smoke P1
      drive_peak was ~0.037 (agent well-fed) -- a low drive_peak in the manifest
      is itself the layer-2 signal.
  (c) C1 still FAIL -> the formation fix was insufficient: re-audit via the now-
      populated cue_nonfire_reason_counts (no_token / resource_field_absent /
      proximity_below_threshold / exception:<Type>) -> /diagnose-errors. The whole
      point of the instrumentation is that this branch is now attributable.
  C2 FAIL -> OFF arm leaking cue behaviour (flag-gating bug).

HONEST SCOPE (carry from 638): targets the CONTACT axis only; does NOT fix
survival (2/3 seeds may die in P1) and may even raise hazard exposure by
approaching food in hazard_food_attraction>0 envs -- the ablation measures
survival (C4) too. Do NOT tune to pass.

claim_ids = []  (substrate-readiness diagnostic; NOT governance evidence).
experiment_purpose = "diagnostic".

References:
- ree-v3/experiments/scaffolded_sd054_onboarding.py (cue-recall bridge + formation-fix amend)
- ree-v3/CLAUDE.md "scaffolded_sd054_onboarding AMEND: cue-recall FORMATION fix (2026-06-04b)"
- REE_assembly/evidence/planning/thought_intake_2026-06-04_cue_ecology_weaning_nursery_to_forager.md
- V3-EXQ-638 (the cue-silent FAIL this validates the fix for)
- V3-EXQ-634c (seeding-calibration regime this reuses)
- REE_assembly/evidence/planning/goal_pipeline_plan.md (GAP-2 / GAP-7)
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

EXPERIMENT_TYPE = "v3_exq_638a_scaffold_cue_recall_contact_ablation"
QUEUE_ID = "V3-EXQ-638a"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Curriculum budgets (mirror 634c / 638).
STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, P1_BUDGET, P2_BUDGET = 20, 10, 120, 70, 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
STAGE0_ZGOAL_GATE = 0.4
STAGE0B_RETENTION_GATE = 0.75
MIN_FRACTION = 2.0 / 3.0
N_RESOURCE_TYPES = 3

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
    )
    if cue_on:
        cfg.scaffold_cue_recall_bridge_enabled = True
        cfg.scaffold_cue_n_resource_types = N_RESOURCE_TYPES
        # 638a FORMATION FIX: bind a per-type token at Stage-0 forced feeding (to
        # the strongest-perceived type) so the bank is non-empty entering P1/P2.
        # Without this the cue is silent (the 638 C1 failure).
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


def _run_seed(seed: int, dry_run: bool, arm: Dict[str, Any]) -> Dict[str, Any]:
    arm_name, cue_on = arm["name"], bool(arm["cue"])
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, cue_on)
    device = torch.device("cpu")

    # Probe env via the scheduler's own builder so the agent dims match the
    # phase envs EXACTLY (ON arm = SD-049-enabled => larger world_obs_dim).
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
    p1_cue_diag = dict(getattr(p1, "cue_diag", {}) or {})
    p2_cue_diag = dict(getattr(p2, "cue_diag", {}) or {})
    # Source cue fires from the cue_diag accumulator. P2OnboardingMetrics has NO
    # aggregated n_cue_recall_fires field, so getattr(p2, "n_cue_recall_fires", 0)
    # returns 0 even when the cue fires -- a measurement bug in 638's C1 that would
    # mask the formation fix (the cue_diag counts it correctly). The OFF arm leaves
    # cue_diag at defaults (bridge-off short-circuits before any diag write), so
    # C2's `== 0` still holds.
    cue_fires = int(p2_cue_diag.get("n_cue_recall_fires", 0))
    print(f"  [train] p2 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events} "
          f"cue_fires={cue_fires} bank={p2_cue_diag.get('token_bank_size', 0)} "
          f"drive_peak={p2_cue_diag.get('drive_peak', 0.0):.4f} "
          f"z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    print(f"verdict: seed={seed} arm={arm_name} contact_rate={p2.contact_rate:.4f} cue_fires={cue_fires}", flush=True)
    return {
        "seed": seed, "arm": arm_name, "aborted_at": None,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "stage0_token_bank_size_end": token_bank,
        "stage0b_retention_gate_passed": bool(s0b.retention_gate_passed),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_contact_steps": int(p2.contact_steps),
        "p2_num_contact_events": int(p2.num_contact_events),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_n_cue_recall_fires": cue_fires,
        # SD-057 cue diagnostics -- the load-bearing attribution (638a addition).
        "p1_cue_diag": p1_cue_diag,
        "p2_cue_diag": p2_cue_diag,
    }


def _abort(seed: int, arm_name: str, stage: str) -> Dict[str, Any]:
    return {
        "seed": seed, "arm": arm_name, "aborted_at": stage,
        "stage0_z_goal_norm_peak": 0.0, "stage0_token_bank_size_end": 0,
        "stage0b_retention_gate_passed": False,
        "p1_survival_pass": False, "p2_contact_rate": 0.0, "p2_contact_steps": 0,
        "p2_num_contact_events": 0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_n_cue_recall_fires": 0,
        "p1_cue_diag": {}, "p2_cue_diag": {},
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        by_arm[arm["name"]] = [_run_seed(s, dry_run, arm) for s in seeds]

    off = by_arm["ARM_OFF"]
    on = by_arm["ARM_CUE_ON"]

    c1 = _frac([r["p2_n_cue_recall_fires"] > 0 for r in on]) >= MIN_FRACTION
    c2 = all(r["p2_n_cue_recall_fires"] == 0 for r in off)
    # C3: per matched seed, cue-ON contact_rate > OFF contact_rate.
    off_by_seed = {r["seed"]: r["p2_contact_rate"] for r in off}
    lift_flags = [r["p2_contact_rate"] > off_by_seed.get(r["seed"], 0.0) for r in on]
    c3 = _frac(lift_flags) >= MIN_FRACTION
    # C4 (informational): survival not regressed.
    c4 = _frac([r["p1_survival_pass"] for r in on]) >= _frac([r["p1_survival_pass"] for r in off])

    overall = bool(c1 and c2 and c3)
    outcome = "PASS" if overall else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] C1_cue_fires_on={c1} C2_cue_silent_off={c2} "
          f"C3_contact_lift={c3} C4_survival_ok={c4} outcome={outcome}", flush=True)

    return {
        "outcome": outcome,
        "acceptance": {
            "C1_cue_fires_on": c1, "C2_cue_silent_off": c2,
            "C3_contact_lift": c3, "C4_survival_not_regressed": c4,
            "overall_pass": overall,
        },
        "on_contact_rates": [r["p2_contact_rate"] for r in on],
        "off_contact_rates": [r["p2_contact_rate"] for r in off],
        "on_cue_fires": [r["p2_n_cue_recall_fires"] for r in on],
        "on_token_bank_sizes": [r["stage0_token_bank_size_end"] for r in on],
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
        "validates": "V3-EXQ-638 (cue-silent FAIL); formation fix scaffold_stage0_bind_incentive_token",
        "substrate": "scaffolded_sd054_onboarding (SD-057 cue-recall bridge + formation-fix amend, 2026-06-04b, a9ef0be)",
        "formation_fix_note": (
            "638 was cue-silent because the IncentiveTokenBank was empty entering "
            "P1/P2 (Stage-0 forced feed bound no token). 638a's ARM_CUE_ON sets "
            "scaffold_stage0_bind_incentive_token=True so Stage-0 binds per-type "
            "tokens to the strongest-perceived type -> bank non-empty -> cue can "
            "match a token. token_bank_size_end + cue_diag are surfaced per arm/seed "
            "so the result is attributable."
        ),
        "decoupling_note": (
            "Self-contained: both arms set the landed 634c ARM_3 seeding regime "
            "(drive_floor=0.9 + benefit_threshold=0.02) directly, so wild contact "
            "seeds in both. Does not depend on the stalled V3-EXQ-634c run."
        ),
        "seeding_regime": {"drive_floor": SEED_DRIVE_FLOOR,
                           "benefit_threshold": SEED_BENEFIT_THRESHOLD,
                           "seeding_floor": _seeding_floor()},
        "pre_registered_gates": {
            "C1_cue_fires_on": "ARM_CUE_ON P2 n_cue_recall_fires > 0 on >= 2/3 seeds",
            "C2_cue_silent_off": "ARM_OFF P2 n_cue_recall_fires == 0 on ALL seeds",
            "C3_contact_lift": "ARM_CUE_ON P2 contact_rate > ARM_OFF per matched seed on >= 2/3",
            "min_fraction": MIN_FRACTION,
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
