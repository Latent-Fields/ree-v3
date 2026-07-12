"""
V3-EXQ-634 -- scaffolded_sd054_onboarding NURSERY/FEEDING substrate-readiness diagnostic.

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
Validate that the 2026-06-03 nursery/feeding amend (ree-v3 e718bf4) to
scaffolded_sd054_onboarding actually delivers, AT FULL BUDGET across >=3 seeds,
the runtime gates that the V3-EXQ-603e/626a/622 cluster autopsy said were
missing. This is the gate that decides whether the post-substrate re-issue
V3-EXQ-603f (Q-045 / MECH-313 / MECH-260 diversity ablation) is runnable. It is
NOT 603f and it tags NO claims -- a same-substrate retest cannot masquerade as
603f because the substrate gate requires the Stage-0 forced-feed positive
control.

Core principle (autopsy): infant REE must be fed before mature autonomous goal
formation can be fairly tested. This diagnostic runs the full developmental
sequence on ONE strengthened scaffold config (no ablation arms):

  Stage 0  run_stage0_nursery -- forced supra-threshold benefit (decoupled from
           survival); proves the goal stream lights when fed.
  Stage 1  run_p0 -- guided low-conflict reef-refuge warm-up (goal pipeline frozen).
  Stage 2+3 run_p1 -- easy->guarded foraging, goal pipeline annealed on, with a
           gentler staged-withdrawal schedule (hold_fraction).
  Stage 4  run_p2 -- frozen-policy measurement under a GUARDED hazard_food_attraction
           with a foraging-contact-rate readout.

PRE-REGISTERED SUBSTRATE GATES (each requires >= 2/3 of seeds):
  G0 stage0_positive_control : Stage-0 z_goal_norm_peak > 0.4
  G1 p1_survival             : P1 survival/foraging gate passed
  G2 p2_contact              : P2 contact_rate > 0 (infant actually fed at measurement)
  G3 p2_zgoal                : P2 z_goal_norm_peak > 0.4 (goal forms ecologically)
PASS = substrate_gate_passed (G0 AND G1 AND G2 AND G3). Computed by the
substrate's own experiments.scaffolded_sd054_onboarding.evaluate_substrate_gate.

INTERPRETATION ON OUTCOME (manifest interpretation_branch via
classify_interpretation_branch):
  PASS  -> substrate ready: flip substrate_queue scaffolded_sd054_onboarding
           ready=true + unblock EXP-603F-POSTSUBSTRATE + queue V3-EXQ-603f
           (a follow-on /governance + /queue-experiment action, NOT automatic).
  FAIL  -> "substrate_not_engaged" / "fed_but_no_goal": the nursery amend is
           necessary but still insufficient at scale -> route /failure-autopsy
           or back to /implement-substrate; DO NOT queue 603f, DO NOT silently
           re-tune. (This run is diagnostic; it weights no claim either way.)

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; not governance evidence)
predecessor (NOT supersedes): V3-EXQ-621a (the pre-amend substrate-readiness run;
  different design -- this one exercises the new Stage-0 nursery method).
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_634_scaffolded_nursery_substrate_readiness"
QUEUE_ID = "V3-EXQ-634"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror V3-EXQ-603e _make_config so this run is
# representative of the env+agent the 603f re-issue will use).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10

# Strengthened scaffold curriculum (set HERE, not as substrate defaults -- the
# amend is additive; the experiment owns its curriculum, as 603e owns its budget).
STAGE0_BUDGET = 20
P0_BUDGET = 100
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3          # gentler staged-withdrawal P1 schedule
P0_NUM_HAZARDS = 1              # reduced early hazard pressure (lever; default 2)
P2_HFA_GUARD = 0.3             # P2 measurement guard (default hard env is 0.7)

# Pre-registered gates (constants; NOT derived from the run's own statistics).
STAGE0_ZGOAL_GATE = 0.4
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
        stage0, p0, p1, p2, steps = 2, 5, 5, 2, 30
    else:
        stage0, p0, p1, p2, steps = STAGE0_BUDGET, P0_BUDGET, P1_BUDGET, P2_BUDGET, TRAIN_STEPS
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
    )
    # Dry-run: scale the P1 survival gate so short episodes can clear it
    # (default 75 is unreachable at steps<75); mirrors the 603e dry-run scaling.
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
    return cfg


def _run_seed(seed: int, dry_run: bool) -> Dict[str, Any]:
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    total_eps = (
        scaffold_cfg.scaffold_stage0_episode_budget
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

    print(f"Seed {seed} Condition nursery_scaffold", flush=True)

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
            "p1_survival_pass": False,
            "p2_contact_rate": 0.0, "p2_z_goal_norm_peak": 0.0,
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
            "p1_survival_pass": False,
            "p2_contact_rate": 0.0, "p2_z_goal_norm_peak": 0.0,
        }

    # Stage 2+3 -- easy->guarded foraging (run_p1).
    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(
        f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
        f" median_last={p1.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
        f" final_hfa={p1.final_hazard_food_attraction:.2f}",
        flush=True,
    )

    # Stage 4 -- frozen-policy guarded measurement (run_p2).
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(
        f"  [train] p2_measure seed={seed} ep {done}/{total_eps}"
        f" z_goal_peak={p2.z_goal_norm_peak_max:.4f}"
        f" contact_rate={p2.contact_rate:.4f}"
        f" hfa_used={p2.hazard_food_attraction_used:.2f}",
        flush=True,
    )

    seed_pass = (
        s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE
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
        "p0_mean_episode_length": float(p0.mean_episode_length),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_contact_steps": int(p2.contact_steps),
        "p2_z_goal_norm_peak": float(p2.z_goal_norm_peak_max),
        "p2_hazard_food_attraction_used": float(p2.hazard_food_attraction_used),
        "seed_pass": bool(seed_pass),
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    per_seed: List[Dict[str, Any]] = [_run_seed(s, dry_run) for s in seeds]

    gate = evaluate_substrate_gate(
        stage0_z_goal_peaks_per_seed=[r["stage0_z_goal_norm_peak"] for r in per_seed],
        p1_survival_pass_per_seed=[r["p1_survival_pass"] for r in per_seed],
        p2_z_goal_peaks_per_seed=[r["p2_z_goal_norm_peak"] for r in per_seed],
        p2_contact_rates_per_seed=[r["p2_contact_rate"] for r in per_seed],
        z_goal_gate=STAGE0_ZGOAL_GATE,
        contact_gate=CONTACT_GATE,
        min_fraction=MIN_FRACTION,
    )
    branch = classify_interpretation_branch(gate)
    outcome = "PASS" if gate["substrate_gate_passed"] else "FAIL"

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
        "p1_survival_foraging_pass_per_seed": [r["p1_survival_pass"] for r in per_seed],
        "p2_contact_rate_per_seed": [r["p2_contact_rate"] for r in per_seed],
        "p2_z_goal_norm_peak_per_seed": [r["p2_z_goal_norm_peak"] for r in per_seed],
        "substrate_gate_passed": gate["substrate_gate_passed"],
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
        "substrate": "scaffolded_sd054_onboarding (nursery/feeding amend, ree-v3 e718bf4)",
        "predecessor": "V3-EXQ-621a (pre-amend readiness; NOT superseded -- different design)",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "p2_z_goal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "p0_budget": P0_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "train_steps": TRAIN_STEPS, "p1_hold_fraction": P1_HOLD_FRACTION,
            "p0_num_hazards": P0_NUM_HAZARDS, "p2_hfa_guard": P2_HFA_GUARD,
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
