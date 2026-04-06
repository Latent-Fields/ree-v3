"""
EXQ-255: MECH-203 Benefit-Salience Tagging Diagnostic

Claim: MECH-203 (serotonergic_replay_salience_tagging)
Design doc: REE_assembly/docs/architecture/sleep/serotonergic_cross_state_substrate.md

Mechanism under test: tonic_5ht benefit-salience tagging (SR-2).
When tonic_5ht_enabled=True, the SerotoninModule tags benefit-relevant
experiences via VALENCE_WANTING in the residue field. This experiment
verifies that the tagging mechanism works correctly: benefit locations
accumulate higher VALENCE_WANTING than non-benefit locations.

Pre-registered acceptance criteria:
  C1: mean VALENCE_WANTING at benefit-contact locations > 2x non-benefit locations
      (benefit_valence_ratio > 2.0)
  C2: tonic_5ht rises above baseline (0.5) during benefit-rich episodes
      (mean_peak_5ht > 0.55)
  C3: z_goal_seeding_gain dynamically changes with tonic_5ht
      (gain_std > 0.01 across episode)

Conditions:
  A (SEROTONIN):  tonic_5ht_enabled=True, z_goal_enabled=True
  B (NO_SEROTONIN): tonic_5ht_enabled=False, z_goal_enabled=True (control)

Decision: PASS if all C1-C3 met in condition A. FAIL otherwise.
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from ree_core.residue.field import VALENCE_WANTING


EXPERIMENT_TYPE = "v3_exq_255_mech203_benefit_tagging"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-203"]

# Pre-registered thresholds
THRESH_C1_RATIO = 2.0    # benefit/non-benefit VALENCE_WANTING ratio
THRESH_C2_PEAK  = 0.55   # mean peak tonic_5ht
THRESH_C3_STD   = 0.01   # gain variation

# Architecture dims (standard V3 CausalGridWorldV2)
BODY_OBS_DIM   = 12
WORLD_OBS_DIM  = 250
ACTION_DIM     = 4

# Training params
TRAIN_EPISODES     = 100
STEPS_PER_EPISODE  = 200
SEEDS              = [42, 137, 2026]
LR                 = 1e-3


def make_config(condition: str) -> REEConfig:
    """Build config for each condition."""
    serotonin_en = (condition == "SEROTONIN")

    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,
        z_goal_enabled=True,
        drive_weight=2.0,
        benefit_eval_enabled=True,
        goal_weight=1.0,
        tonic_5ht_enabled=serotonin_en,
    )
    return cfg


def _action_onehot(idx: int, device) -> torch.Tensor:
    v = torch.zeros(1, ACTION_DIM, device=device)
    v[0, idx] = 1.0
    return v


def run_condition(condition: str, seed: int, dry_run: bool = False) -> Dict:
    """Run one condition x seed."""
    print(f"Seed {seed} Condition {condition}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")
    cfg = make_config(condition)
    agent = REEAgent(cfg)
    agent.to(device)

    env = CausalGridWorldV2(
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.05,
        resource_benefit=0.05,
        use_proxy_fields=True,
    )

    optimizer = optim.Adam(agent.parameters(), lr=LR)

    n_episodes = 2 if dry_run else TRAIN_EPISODES
    n_steps = 10 if dry_run else STEPS_PER_EPISODE

    # Metrics tracking
    tonic_5ht_peaks = []
    seeding_gains = []
    benefit_contact_valences = []
    nonbenefit_contact_valences = []

    for ep in range(n_episodes):
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  [train] {condition} seed={seed} ep {ep+1}/{n_episodes}", flush=True)
        obs, info = env.reset()
        agent.reset()
        body_obs = torch.tensor(obs[:BODY_OBS_DIM], dtype=torch.float32).unsqueeze(0)
        world_obs = torch.tensor(obs[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM],
                                 dtype=torch.float32).unsqueeze(0)

        ep_5ht_peak = 0.5
        ep_gains = []

        for step in range(n_steps):
            latent = agent.sense(body_obs, world_obs)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, cfg.latent.world_dim, device=device)
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # Extract benefit_exposure from body_state
            benefit_exposure = float(body_obs[0, 11]) if body_obs.shape[-1] > 11 else 0.0
            drive_level = agent.compute_drive_level(body_obs)

            # Serotonin step + goal update + benefit tagging
            agent.serotonin_step(benefit_exposure)
            agent.update_z_goal(benefit_exposure, drive_level)
            agent.update_benefit_salience(benefit_exposure)

            # Track metrics
            if agent.serotonin.enabled:
                ep_5ht_peak = max(ep_5ht_peak, agent.serotonin.tonic_5ht)
                ep_gains.append(agent.serotonin.current_seeding_gain())

            # Sample VALENCE_WANTING at current z_world
            if agent._current_latent is not None:
                z_w = agent._current_latent.z_world
                if hasattr(agent.residue_field, 'evaluate_valence'):
                    val = agent.residue_field.evaluate_valence(z_w)
                    wanting_val = float(val[0, VALENCE_WANTING].item())
                    if benefit_exposure > 0.01:
                        benefit_contact_valences.append(wanting_val)
                    else:
                        nonbenefit_contact_valences.append(wanting_val)

            # Environment step
            action_idx = int(action.argmax(dim=-1).item())
            obs_next, reward, done, truncated, info_next = env.step(action_idx)

            # Update residue
            harm_signal = info_next.get("harm", 0.0)
            if harm_signal != 0:
                agent.update_residue(harm_signal)

            # Training
            optimizer.zero_grad()
            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            # Prepare next step
            body_obs = torch.tensor(obs_next[:BODY_OBS_DIM],
                                    dtype=torch.float32).unsqueeze(0)
            world_obs = torch.tensor(obs_next[BODY_OBS_DIM:BODY_OBS_DIM + WORLD_OBS_DIM],
                                     dtype=torch.float32).unsqueeze(0)

            if done or truncated:
                break

        tonic_5ht_peaks.append(ep_5ht_peak)
        seeding_gains.extend(ep_gains)

    # Compute metrics
    mean_benefit_val = float(np.mean(benefit_contact_valences)) if benefit_contact_valences else 0.0
    mean_nonbenefit_val = float(np.mean(nonbenefit_contact_valences)) if nonbenefit_contact_valences else 0.0
    benefit_valence_ratio = (mean_benefit_val / max(mean_nonbenefit_val, 1e-9))
    mean_peak_5ht = float(np.mean(tonic_5ht_peaks))
    gain_std = float(np.std(seeding_gains)) if seeding_gains else 0.0

    run_pass = (benefit_valence_ratio > THRESH_C1_RATIO
                and mean_peak_5ht > THRESH_C2_PEAK
                and gain_std > THRESH_C3_STD) if condition == "SEROTONIN" else True
    print(f"verdict: {'PASS' if run_pass else 'FAIL'}")

    return {
        "condition": condition,
        "seed": seed,
        "mean_benefit_valence": mean_benefit_val,
        "mean_nonbenefit_valence": mean_nonbenefit_val,
        "benefit_valence_ratio": benefit_valence_ratio,
        "mean_peak_5ht": mean_peak_5ht,
        "gain_std": gain_std,
        "n_benefit_samples": len(benefit_contact_valences),
        "n_nonbenefit_samples": len(nonbenefit_contact_valences),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=str,
                        default=str(Path(__file__).resolve().parents[2] /
                                    "REE_assembly" / "evidence" / "experiments"))
    args = parser.parse_args()

    print(f"EXQ-255: MECH-203 Benefit-Salience Tagging Diagnostic")
    print(f"  dry_run={args.dry_run}")

    conditions = ["SEROTONIN", "NO_SEROTONIN"]
    all_results = []

    for cond in conditions:
        for seed in SEEDS:
            print(f"  Running {cond} seed={seed}...")
            result = run_condition(cond, seed, dry_run=args.dry_run)
            all_results.append(result)
            print(f"    ratio={result['benefit_valence_ratio']:.3f} "
                  f"peak_5ht={result['mean_peak_5ht']:.3f} "
                  f"gain_std={result['gain_std']:.4f}")

    # Aggregate across seeds for SEROTONIN condition
    serotonin_results = [r for r in all_results if r["condition"] == "SEROTONIN"]
    agg_ratio = float(np.mean([r["benefit_valence_ratio"] for r in serotonin_results]))
    agg_peak = float(np.mean([r["mean_peak_5ht"] for r in serotonin_results]))
    agg_gain_std = float(np.mean([r["gain_std"] for r in serotonin_results]))

    c1 = agg_ratio > THRESH_C1_RATIO
    c2 = agg_peak > THRESH_C2_PEAK
    c3 = agg_gain_std > THRESH_C3_STD

    verdict = "PASS" if (c1 and c2 and c3) else "FAIL"

    print(f"\n  Aggregated SEROTONIN condition:")
    print(f"    C1 benefit_valence_ratio={agg_ratio:.3f} > {THRESH_C1_RATIO} -> {'PASS' if c1 else 'FAIL'}")
    print(f"    C2 mean_peak_5ht={agg_peak:.3f} > {THRESH_C2_PEAK} -> {'PASS' if c2 else 'FAIL'}")
    print(f"    C3 gain_std={agg_gain_std:.4f} > {THRESH_C3_STD} -> {'PASS' if c3 else 'FAIL'}")
    print(f"  VERDICT: {verdict}")

    # Write flat JSON output
    run_id = f"v3_exq_255_mech203_benefit_tagging_v3"
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "verdict": verdict,
        "evidence_direction": "supports" if verdict == "PASS" else "weakens",
        "conditions": conditions,
        "seeds": SEEDS,
        "metrics": {
            "benefit_valence_ratio": agg_ratio,
            "mean_peak_5ht": agg_peak,
            "gain_std": agg_gain_std,
        },
        "evidence": {
            "c1_benefit_ratio_passed": c1,
            "c2_peak_5ht_passed": c2,
            "c3_gain_variation_passed": c3,
        },
        "per_condition_results": all_results,
    }

    out_path = Path(args.output_dir) / f"{run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
