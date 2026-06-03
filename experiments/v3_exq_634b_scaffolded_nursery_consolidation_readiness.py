"""
V3-EXQ-634b -- scaffolded_sd054_onboarding NURSERY + PROTECTED-CONSOLIDATION
substrate-readiness diagnostic (developmental-window amend, 2026-06-03b).

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
Validate the 2026-06-03b developmental-window / consolidation amend to
scaffolded_sd054_onboarding. The predecessor V3-EXQ-634 added Stage-0 forced
feeding (z_goal lights when fed) but then exposed that fragile trace to P1/P2
phases that call update_z_goal every step -- and because GoalState.update()
ALWAYS decays the persistent attractor (ree_core/goal.py:173) and only refreshes
on supra-threshold benefit, every UNFED step is a decay-only washout. With the
agent's poor early foraging competence, P1 is mostly unfed, so the Stage-0 trace
is washed out before P2 measurement. 634 therefore tests "can the infant stay
goal-active while fed-then-starved under decay-only updates?" rather than
"form -> consolidate -> learn guided/autonomous contact."

This corrected diagnostic turns ON the developmental-window flags:
  - Stage-0b protected consolidation (run_stage0b_consolidation): a short window
    after Stage-0 where update_z_goal is NOT called, so the just-formed z_goal
    cannot be washed out by decay-only updating (E1/E2 keep training).
  - Contact-gated P1/P2 goal updates (scaffold_contact_gated_goal_updates): P1/P2
    only call update_z_goal on a VALIDATED contact step; unfed steps are skipped,
    so the attractor is never eroded by decay-only updates. decay_only is reserved
    for mature/autonomous tests, NOT the nursery gate.

Developmental sequence per seed (Stage-0b inserted between Stage-0 and P0):
  Stage 0   run_stage0_nursery        -- forced supra-threshold benefit (z_goal lights)
  Stage 0b  run_stage0b_consolidation -- PROTECTED window; retention measured
  Stage 1   run_p0                    -- guided low-conflict warm-up (goal frozen)
  Stage 2+3 run_p1                    -- easy->guarded foraging, CONTACT-GATED goal updates
  Stage 4   run_p2                    -- frozen-policy measurement, CONTACT-GATED

PRE-REGISTERED SUBSTRATE GATES (each requires >= 2/3 of seeds; do NOT retune):
  G0  stage0_positive_control : Stage-0 z_goal_norm_peak > 0.4
  G0b stage0b_retention       : Stage-0b retention_ratio >= 0.75 (NEW; consolidation
                                must protect the Stage-0 trace, not wash it out)
  G1  p1_survival             : P1 survival/foraging gate passed
  G2  p2_contact              : P2 contact_rate > 0 (infant actually fed at measurement)
  G3  p2_zgoal                : P2 z_goal_norm_peak > 0.4 (now a FAIR test: the trace
                                is not destroyed by decay-only updating before contact)
PASS = G0 AND G0b AND G1 AND G2 AND G3.

INTERPRETATION ON OUTCOME:
  PASS -> substrate ready: (a follow-on /governance + /queue-experiment action,
          NOT automatic, NOT in this script) flip substrate_queue
          scaffolded_sd054_onboarding ready=true + queue V3-EXQ-603f.
  FAIL G0b (retention < 0.75) -> consolidation/protection bug -> /failure-autopsy
          or /implement-substrate (the protected window is not protecting).
  FAIL G3 despite G2 contact -> "fed_but_no_goal" (goal-formation issue, not washout).
  FAIL Stage-0-lights / no contact -> "substrate_not_engaged".
  In every FAIL case this is diagnostic, weights no claim, and 603f STAYS BLOCKED.

DIAGNOSTICS (so goal loss is attributable, not conflated): with contact-gating ON,
P1/P2 n_decay_only_updates should be ~0 (no decay-only washout); n_skipped_protected
counts the unfed steps that were protected; n_contact_refresh counts real ecological
refreshes.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; not governance evidence)
predecessor (NOT supersedes): V3-EXQ-634 (the pre-consolidation nursery run;
  634 lacks Stage-0b + contact-gating -- this run exercises the developmental
  window the 634 design-error review identified as missing).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

EXPERIMENT_TYPE = "v3_exq_634b_scaffolded_nursery_consolidation_readiness"
QUEUE_ID = "V3-EXQ-634b"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror V3-EXQ-634 _make_config so this run is
# representative of the env+agent the 603f re-issue will use).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10

# Strengthened scaffold curriculum (mirrors V3-EXQ-634; set HERE, not as
# substrate defaults -- the amend is additive; the experiment owns its
# curriculum). Stage-0b consolidation budget is the new addition.
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10            # NEW: protected consolidation window
P0_BUDGET = 100
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3          # gentler staged-withdrawal P1 schedule
P0_NUM_HAZARDS = 1              # reduced early hazard pressure (lever; default 2)
P2_HFA_GUARD = 0.3             # P2 measurement guard (default hard env is 0.7)

# Pre-registered gates (constants; NOT derived from the run's own statistics).
STAGE0_ZGOAL_GATE = 0.4
STAGE0B_RETENTION_GATE = 0.75   # NEW: consolidation must retain the Stage-0 trace
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0


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
        # The two-part-fix precondition (603e): z_goal_enabled creates GoalState;
        # drive_weight=2.0 is the SD-012 amplification the reference V3-EXQ-622 uses.
        z_goal_enabled=True,
        drive_weight=2.0,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
    )


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
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
        # --- 2026-06-03b developmental-window / consolidation amend (ON) ---
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
    )
    # Dry-run: scale the P1 survival gate so short episodes can clear it
    # (default 75 is unreachable at steps<75); mirrors the 603e/634 dry-run scaling.
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
    return cfg


def _run_seed(seed: int, dry_run: bool) -> Dict[str, Any]:
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    total_eps = (
        scaffold_cfg.scaffold_stage0_episode_budget
        + scaffold_cfg.scaffold_stage0b_episode_budget
        + scaffold_cfg.scaffold_p0_episode_budget
        + scaffold_cfg.scaffold_p1_episode_budget
        + scaffold_cfg.scaffold_p2_episode_budget
    )

    # Build the P2 target env first so the agent's REEConfig has matching dims.
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

    print(f"Seed {seed} Condition nursery_consolidation_scaffold", flush=True)

    # Stage 0 -- forced-benefit nursery.
    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    print(
        f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
        f" forced_benefit={s0.mean_forced_benefit:.2f}"
        f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}",
        flush=True,
    )
    if s0.aborted:
        print(
            f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}",
            flush=True,
        )
        return {
            "seed": seed, "aborted_at": "stage0", "abort_reason": s0.abort_reason,
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "stage0_benefit_exposure": float(s0.mean_forced_benefit),
            "stage0b_retention_ratio": 0.0,
            "stage0b_retention_gate_passed": False,
            "p1_survival_pass": False,
            "p2_contact_rate": 0.0, "p2_z_goal_norm_peak": 0.0,
            "seed_pass": False,
        }

    # Stage 0b -- PROTECTED consolidation (developmental-window amend).
    s0b = scheduler.run_stage0b_consolidation(
        agent, device, stage0_baseline_norm=s0.z_goal_norm_peak
    )
    done += s0b.n_episodes
    print(
        f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
        f" start={s0b.z_goal_norm_start:.4f} end={s0b.z_goal_norm_end:.4f}"
        f" retention={s0b.retention_ratio:.3f}"
        f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}",
        flush=True,
    )
    if s0b.aborted:
        print(
            f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}",
            flush=True,
        )
        return {
            "seed": seed, "aborted_at": "stage0b", "abort_reason": s0b.abort_reason,
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "stage0_benefit_exposure": float(s0.mean_forced_benefit),
            "stage0b_retention_ratio": float(s0b.retention_ratio),
            "stage0b_retention_gate_passed": bool(s0b.retention_gate_passed),
            "p1_survival_pass": False,
            "p2_contact_rate": 0.0, "p2_z_goal_norm_peak": 0.0,
            "seed_pass": False,
        }

    # Stage 1 -- guided low-conflict warm-up (run_p0).
    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(
        f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
        f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
        flush=True,
    )
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return {
            "seed": seed, "aborted_at": "p0", "abort_reason": p0.abort_reason,
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "stage0_benefit_exposure": float(s0.mean_forced_benefit),
            "stage0b_retention_ratio": float(s0b.retention_ratio),
            "stage0b_retention_gate_passed": bool(s0b.retention_gate_passed),
            "p1_survival_pass": False,
            "p2_contact_rate": 0.0, "p2_z_goal_norm_peak": 0.0,
            "seed_pass": False,
        }

    # Stage 2+3 -- easy->guarded foraging (run_p1, CONTACT-GATED goal updates).
    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(
        f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
        f" median_last={p1.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
        f" final_hfa={p1.final_hazard_food_attraction:.2f}"
        f" gated={p1.contact_gated} decay_only={p1.n_decay_only_updates}"
        f" skipped={p1.n_skipped_protected_updates}"
        f" refresh={p1.n_contact_refresh_updates}",
        flush=True,
    )

    # Stage 4 -- frozen-policy guarded measurement (run_p2, CONTACT-GATED).
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(
        f"  [train] p2_measure seed={seed} ep {done}/{total_eps}"
        f" z_goal_peak={p2.z_goal_norm_peak_max:.4f}"
        f" contact_rate={p2.contact_rate:.4f}"
        f" hfa_used={p2.hazard_food_attraction_used:.2f}"
        f" gated={p2.contact_gated} decay_only={p2.n_decay_only_updates}"
        f" skipped={p2.n_skipped_protected_updates}"
        f" refresh={p2.n_contact_refresh_updates}",
        flush=True,
    )

    seed_pass = (
        s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE
        and s0b.retention_gate_passed
        and p1.survival_gate_passed
        and p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_peak_max > P2_ZGOAL_GATE
    )
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed}", flush=True)

    return {
        "seed": seed,
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
        "p2_z_goal_norm_peak": float(p2.z_goal_norm_peak_max),
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


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    per_seed: List[Dict[str, Any]] = [_run_seed(s, dry_run) for s in seeds]

    # G0 / G1 / G2 / G3 via the substrate's own gate evaluator (unchanged from 634).
    gate = evaluate_substrate_gate(
        stage0_z_goal_peaks_per_seed=[r["stage0_z_goal_norm_peak"] for r in per_seed],
        p1_survival_pass_per_seed=[r["p1_survival_pass"] for r in per_seed],
        p2_z_goal_peaks_per_seed=[r["p2_z_goal_norm_peak"] for r in per_seed],
        p2_contact_rates_per_seed=[r["p2_contact_rate"] for r in per_seed],
        z_goal_gate=STAGE0_ZGOAL_GATE,
        contact_gate=CONTACT_GATE,
        min_fraction=MIN_FRACTION,
    )
    # G0b is the developmental-window addition: Stage-0b must RETAIN the trace.
    g0b_retention = _fraction_passing(
        [bool(r["stage0b_retention_gate_passed"]) for r in per_seed]
    ) >= MIN_FRACTION
    gate["g0b_stage0b_retention"] = bool(g0b_retention)
    gate["stage0b_retention_gate"] = float(STAGE0B_RETENTION_GATE)

    # PASS = the 634 conjunction AND the new consolidation-retention gate.
    substrate_gate_passed = bool(gate["substrate_gate_passed"] and g0b_retention)
    gate["substrate_gate_passed"] = substrate_gate_passed

    branch = classify_interpretation_branch(gate)
    # Refine the branch for the consolidation-specific failure mode: if the only
    # thing that failed is retention, name it explicitly (the protection is the
    # thing under test here).
    if (not substrate_gate_passed and not g0b_retention
            and gate.get("stage0_positive_control")):
        branch = "consolidation_not_protected"
    outcome = "PASS" if substrate_gate_passed else "FAIL"

    print(f"[{EXPERIMENT_TYPE}] gate={gate}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] interpretation_branch={branch}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] outcome={outcome}", flush=True)

    return {
        "outcome": outcome,
        "substrate_gate": gate,
        "interpretation_branch": branch,
        "per_seed": per_seed,
        # Manifest readout fields the autopsy/governance asked for.
        "stage0_benefit_exposure_per_seed": [r["stage0_benefit_exposure"] for r in per_seed],
        "stage0_z_goal_norm_peak_per_seed": [r["stage0_z_goal_norm_peak"] for r in per_seed],
        "stage0b_retention_ratio_per_seed": [r["stage0b_retention_ratio"] for r in per_seed],
        "stage0b_retention_gate_passed_per_seed": [r["stage0b_retention_gate_passed"] for r in per_seed],
        "p1_survival_foraging_pass_per_seed": [r["p1_survival_pass"] for r in per_seed],
        "p1_n_decay_only_updates_per_seed": [r.get("p1_n_decay_only_updates", -1) for r in per_seed],
        "p1_n_skipped_protected_updates_per_seed": [r.get("p1_n_skipped_protected_updates", -1) for r in per_seed],
        "p2_contact_rate_per_seed": [r["p2_contact_rate"] for r in per_seed],
        "p2_z_goal_norm_peak_per_seed": [r["p2_z_goal_norm_peak"] for r in per_seed],
        "p2_n_decay_only_updates_per_seed": [r.get("p2_n_decay_only_updates", -1) for r in per_seed],
        "p2_n_skipped_protected_updates_per_seed": [r.get("p2_n_skipped_protected_updates", -1) for r in per_seed],
        "substrate_gate_passed": substrate_gate_passed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    """Run + write manifest. Returns {outcome, manifest_path} for the
    __main__ block to relay to emit_outcome (the runner-conformance sentinel)."""
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
        "substrate": "scaffolded_sd054_onboarding (developmental-window / consolidation amend, 2026-06-03b)",
        "predecessor": "V3-EXQ-634 (pre-consolidation nursery; NOT superseded -- adds Stage-0b + contact-gating)",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "stage0b_retention_gate": STAGE0B_RETENTION_GATE,
            "p2_z_goal_gate": P2_ZGOAL_GATE,
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
    sys.exit(0)
