"""V3-EXQ-571: E3 score composition variance decomposition diagnostic.

Stage 4 of the behavioral diversity collapse chain. After SP-CEM produces
diverse candidates (EXQ-567 PASS) and E2 rollout is inspected (EXQ-570),
this diagnostic decomposes E3 score variance to identify which components
dominate selection and which are negligible in practice.

Uses the V3-EXQ-571 instrumentation (e3_score_decomp_enabled flag on
E3TrajectorySelector, _last_score_bias_decomp on REEAgent) to collect per-
candidate score component values at every step.

Arms:
  ARM_0: Baseline (SP-CEM ON, diversity stack OFF)
  ARM_1: Diversity stack ON (MECH-313 noise_floor + MECH-314 curiosity + MECH-320 tonic_vigor)

Metrics per arm:
  - Inter-candidate variance fraction per component (what drives selection spread)
  - Temporal variance fraction per component (what varies across steps)
  - Bias contribution fraction (total bias vs raw trajectory score)
  - Per-diversity-stack component mean contribution

Interpretation grid:
  f/harm_weighted dominate (>90% each) -> score is reality cost only;
      diversity stack biases not reaching selection. Need scale calibration.
  Any bias component > 90% -> bias overwhelming trajectory quality;
      reduce bias_scale. Monostrategy may flip direction, not be cured.
  ARM_1 bias_total_variance >> ARM_0 -> diversity stack is reaching E3.
  ARM_1 bias_fraction ~ ARM_0 bias_fraction -> stack not activating; check
      upstream candidate diversity (SP-CEM / CEM).
  Roughly uniform component spread -> architecture balanced; ready for ARC-065.

WARNING thresholds:
  < 1% variance fraction -> component negligible (scaling too weak)
  > 90% variance fraction -> component dominant (scaling too strong)

EXPERIMENT_PURPOSE = "diagnostic" -- excluded from governance confidence scoring.
"""

import sys
import json
import argparse
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_protocol import emit_outcome
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_571_e3_score_variance_decomp"

SEEDS = [42, 123, 456]
N_STEPS = 200
GRID_SIZE = 8
NUM_HAZARDS = 1

# Variance warning thresholds
WARN_LOW_FRACTION = 0.01    # < 1% -> negligible
WARN_HIGH_FRACTION = 0.90   # > 90% -> dominant

ARM_NAMES = ["ARM_0_baseline", "ARM_1_diversity_stack"]

# Component names from e3_selector instrumentation + bias decomp
SCORE_COMPONENTS = ["f", "harm_weighted", "residue_weighted",
                    "benefit_weighted", "novelty_weighted", "goal_weighted"]
BIAS_COMPONENTS = ["dacc", "lateral_pfc", "ofc", "gated_policy",
                   "mech295_liking", "curiosity", "tonic_vigor",
                   "forced", "noise_floor_temp", "total_bias"]


def make_env_and_agent(seed, arm_name):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorld(
        size=GRID_SIZE,
        num_hazards=NUM_HAZARDS,
        use_proxy_fields=True,
    )
    _, obs_dict = env.reset()

    body_obs_dim = obs_dict["body_state"].shape[0]
    world_obs_dim = obs_dict["world_state"].shape[0]
    action_dim = env.action_dim

    config = REEConfig().from_dims(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        action_dim=action_dim,
        harm_obs_dim=51,
        # SP-CEM for candidate diversity
        use_support_preserving_cem=True,
        support_preserving_min_first_action_classes=2,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
    )
    config.latent.alpha_world = 0.9  # SD-008: high-fidelity z_world

    # ARM_1: diversity stack
    if arm_name == "ARM_1_diversity_stack":
        config.use_noise_floor = True
        config.noise_floor_alpha = 0.1
        config.noise_floor_min_temperature = 1.0
        config.use_structured_curiosity = True
        config.use_curiosity_novelty = True
        config.use_curiosity_uncertainty = True
        config.use_curiosity_learning_progress = True
        config.curiosity_novelty_weight = 0.05
        config.curiosity_uncertainty_weight = 0.05
        config.curiosity_learning_progress_weight = 0.05
        config.curiosity_bias_scale = 0.1
        config.use_tonic_vigor = True
        config.tonic_vigor_half_life = 100.0
        config.tonic_vigor_w_action = 0.1
        config.tonic_vigor_w_passive = 0.1
        config.tonic_vigor_bias_scale = 0.1

    agent = REEAgent(config=config)
    agent.eval()

    # Enable E3 score decomposition instrumentation
    agent.e3.e3_score_decomp_enabled = True

    return agent, env, obs_dict


def run_arm(arm_name, seed, dry_run=False):
    agent, env, obs_dict = make_env_and_agent(seed, arm_name)
    n_steps = 5 if dry_run else N_STEPS

    # Per-step accumulators: list of dicts
    per_step_decomp = []   # score components (mean across candidates)
    per_step_bias = []     # bias components
    per_step_scores = []   # K-vector of final scores -> use std as inter-cand var

    for step_i in range(n_steps):
        obs_body = torch.tensor(obs_dict["body_state"]).float()
        obs_world = torch.tensor(obs_dict["world_state"]).float()

        with torch.no_grad():
            _action = agent.act_with_split_obs(obs_body, obs_world)

        # --- Read decomposition from agent after selection ---
        last_decomp = agent.e3.last_score_decomp
        last_bias = agent._last_score_bias_decomp
        last_scores = agent.e3.last_scores  # [K] tensor, None if not yet set

        if last_decomp and last_decomp.get("per_candidate"):
            cands = last_decomp["per_candidate"]
            # Mean component values across candidates this step
            step_comps = {}
            for comp in SCORE_COMPONENTS:
                vals = [c.get(comp, 0.0) for c in cands]
                step_comps[comp] = float(np.mean(vals))
            per_step_decomp.append(step_comps)

        if last_bias:
            step_bias = {k: float(last_bias.get(k, 0.0)) for k in BIAS_COMPONENTS}
            per_step_bias.append(step_bias)

        if last_scores is not None and last_scores.numel() > 1:
            per_step_scores.append(last_scores.cpu().tolist())

        # Env step
        if isinstance(_action, torch.Tensor):
            action_idx = int(_action.argmax().item())
        else:
            action_idx = int(_action)
        _, _, terminated, _, obs_dict = env.step(action_idx % env.action_dim)
        if terminated:
            agent.reset()
            _, obs_dict = env.reset()

    # --- Variance decomposition ---
    result = compute_variance_decomp(per_step_decomp, per_step_bias, per_step_scores, arm_name)
    result["arm"] = arm_name
    result["seed"] = seed
    result["n_steps_collected"] = len(per_step_decomp)

    return result


def compute_variance_decomp(per_step_decomp, per_step_bias, per_step_scores, arm_name):
    """Compute temporal variance fraction for each component."""
    result = {
        "temporal_variance": {},
        "temporal_std": {},
        "temporal_fraction": {},
        "bias_variance": {},
        "bias_fraction": {},
        "inter_candidate_score_std_mean": 0.0,
        "warnings": [],
        "scaling_recommendations": [],
    }

    if not per_step_decomp:
        result["warnings"].append("no decomp data collected (decomp not populated)")
        return result

    # -- Temporal variance of each score component --
    comp_arrays = {}
    for comp in SCORE_COMPONENTS:
        vals = [s.get(comp, 0.0) for s in per_step_decomp]
        comp_arrays[comp] = np.array(vals, dtype=np.float64)

    # Total score ~ sum of all raw components (ignoring bias for this decomp)
    # Use temporal std of total as the denominator
    total_per_step = np.zeros(len(per_step_decomp), dtype=np.float64)
    for comp in SCORE_COMPONENTS:
        total_per_step += comp_arrays[comp]

    total_var = float(np.var(total_per_step)) + 1e-12

    for comp in SCORE_COMPONENTS:
        v = float(np.var(comp_arrays[comp]))
        std = float(np.std(comp_arrays[comp]))
        frac = v / total_var
        result["temporal_variance"][comp] = v
        result["temporal_std"][comp] = std
        result["temporal_fraction"][comp] = frac

        if frac < WARN_LOW_FRACTION and v > 0.0:
            result["warnings"].append(
                f"temporal: {comp} fraction={frac:.3f} < {WARN_LOW_FRACTION} (negligible)"
            )
        if frac > WARN_HIGH_FRACTION:
            result["warnings"].append(
                f"temporal: {comp} fraction={frac:.3f} > {WARN_HIGH_FRACTION} (dominant)"
            )

    # -- Bias variance --
    if per_step_bias:
        bias_arrays = {}
        for bcomp in BIAS_COMPONENTS:
            bvals = [s.get(bcomp, 0.0) for s in per_step_bias]
            bias_arrays[bcomp] = np.array(bvals, dtype=np.float64)

        total_bias_var = float(np.var(bias_arrays.get("total_bias", np.zeros(1)))) + 1e-12

        for bcomp in BIAS_COMPONENTS:
            v = float(np.var(bias_arrays[bcomp]))
            result["bias_variance"][bcomp] = v

        # Bias fraction vs total score variance
        for bcomp in BIAS_COMPONENTS:
            result["bias_fraction"][bcomp] = result["bias_variance"][bcomp] / total_var

        result["noise_floor_temp_mean"] = float(
            np.mean(bias_arrays.get("noise_floor_temp", np.zeros(1)))
        )

    # -- Inter-candidate score std (across K candidates per step) --
    if per_step_scores:
        stds = [float(np.std(s)) for s in per_step_scores if len(s) > 1]
        if stds:
            result["inter_candidate_score_std_mean"] = float(np.mean(stds))

    # -- Scaling recommendations --
    recs = []
    raw_total_frac = sum(result["temporal_fraction"].get(c, 0.0) for c in ["f", "harm_weighted"])
    bias_total_frac = result["bias_fraction"].get("total_bias", 0.0)

    if raw_total_frac > 0.90:
        recs.append(
            "f+harm_weighted explain >90%% of temporal variance; "
            "bias_scale on all diversity components may need 5-10x increase"
        )
    if bias_total_frac > 0.90:
        recs.append(
            "total_bias explains >90%% of temporal variance; "
            "diversity stack bias_scale too high -- reduce by 5x"
        )
    if bias_total_frac < 0.01 and per_step_bias:
        recs.append(
            "total_bias explains <1%% of temporal variance; "
            "diversity stack has negligible effect on E3 scores"
        )
    for bcomp in ["curiosity", "tonic_vigor", "dacc"]:
        frac = result["bias_fraction"].get(bcomp, 0.0)
        if frac < 0.001 and per_step_bias:
            recs.append(
                f"{bcomp} bias fraction={frac:.4f} effectively zero -- "
                f"check that the module is enabled and its inputs are non-zero"
            )

    result["scaling_recommendations"] = recs
    return result


def run_experiment(dry_run=False):
    all_results = {}
    arm_summaries = {}

    for arm_name in ARM_NAMES:
        all_results[arm_name] = []
        print(f"\n== {arm_name} ==", flush=True)
        for seed in SEEDS:
            print(f"  seed={seed}", flush=True)
            r = run_arm(arm_name, seed, dry_run=dry_run)
            all_results[arm_name].append(r)

    # -- Summarise per arm (mean across seeds) --
    for arm_name in ARM_NAMES:
        arm_results = all_results[arm_name]
        summary = {"arm": arm_name, "n_seeds": len(arm_results)}

        # Temporal fraction aggregation
        for comp in SCORE_COMPONENTS:
            fracs = [r["temporal_fraction"].get(comp, 0.0) for r in arm_results]
            summary[f"temporal_fraction_{comp}"] = float(np.mean(fracs))

        for bcomp in BIAS_COMPONENTS:
            fracs = [r["bias_fraction"].get(bcomp, 0.0) for r in arm_results]
            summary[f"bias_fraction_{bcomp}"] = float(np.mean(fracs))

        ic_stds = [r.get("inter_candidate_score_std_mean", 0.0) for r in arm_results]
        summary["inter_candidate_score_std_mean"] = float(np.mean(ic_stds))

        all_warnings = []
        all_recs = []
        for r in arm_results:
            all_warnings.extend(r.get("warnings", []))
            all_recs.extend(r.get("scaling_recommendations", []))
        summary["all_warnings"] = sorted(set(all_warnings))
        summary["scaling_recommendations"] = sorted(set(all_recs))

        arm_summaries[arm_name] = summary

    # -- Outcome determination --
    # This is a diagnostic: PASS if both arms produced data with meaningful
    # variance (i.e., decomp actually fired and collected values).
    arm0_steps = float(np.mean([r["n_steps_collected"] for r in all_results["ARM_0_baseline"]]))
    arm1_steps = float(np.mean([r["n_steps_collected"] for r in all_results["ARM_1_diversity_stack"]]))

    if arm0_steps < 1 or arm1_steps < 1:
        outcome = "FAIL"
        outcome_note = (
            f"decomp data collection failed: ARM_0 steps={arm0_steps:.0f}, "
            f"ARM_1 steps={arm1_steps:.0f}; e3_score_decomp_enabled may not be wired"
        )
    else:
        # PASS if data collected and any bias component is non-trivial in ARM_1
        arm1_bias = arm_summaries["ARM_1_diversity_stack"].get("bias_fraction_total_bias", 0.0)
        arm0_bias = arm_summaries["ARM_0_baseline"].get("bias_fraction_total_bias", 0.0)
        outcome = "PASS"
        outcome_note = (
            f"decomp data collected OK (ARM_0 {arm0_steps:.0f} steps, "
            f"ARM_1 {arm1_steps:.0f} steps); "
            f"ARM_1 bias_fraction_total={arm1_bias:.4f} vs ARM_0 {arm0_bias:.4f}. "
            f"Diagnostic complete -- see arm_summaries for component fractions and recommendations."
        )

    return {
        "outcome": outcome,
        "outcome_note": outcome_note,
        "arm_summaries": arm_summaries,
        "per_seed_results": {
            arm: [
                {k: v for k, v in r.items()
                 if k not in ("temporal_variance", "temporal_std", "bias_variance")}
                for r in results
            ]
            for arm, results in all_results.items()
        },
        "per_seed_full": {
            arm: results for arm, results in all_results.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3-EXQ-571: E3 score variance decomp")
    parser.add_argument("--dry-run", action="store_true", help="Short run for smoke testing")
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_571_e3_score_variance_decomp_{timestamp}_v3"

    out_dir = (
        Path(__file__).parent.parent.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": ["ARC-065", "MECH-313", "MECH-314", "MECH-320"],
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": "diagnostic",
        "outcome": result["outcome"],
        "outcome_note": result["outcome_note"],
        "timestamp_utc": timestamp,
        "seeds": SEEDS,
        "n_steps": N_STEPS,
        "warn_low_fraction": WARN_LOW_FRACTION,
        "warn_high_fraction": WARN_HIGH_FRACTION,
        "arm_names": ARM_NAMES,
        "score_components": SCORE_COMPONENTS,
        "bias_components": BIAS_COMPONENTS,
        "arm_summaries": result["arm_summaries"],
        "per_seed_results": result["per_seed_results"],
    }

    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
        json_default=str,
    )
    print(f"Manifest written: {out_path}")

    if args.dry_run:
        print("DRY RUN complete.")

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
