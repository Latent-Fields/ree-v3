#!/opt/local/bin/python3
"""
V3-EXQ-318: SD-022 Limb Damage Stream Separation Validation

experiment_purpose: diagnostic

Tests whether directional limb damage (SD-022) produces genuine causal independence
between the sensory-discriminative (z_harm_s) and affective-motivational (z_harm_a)
harm streams.

EXQ-241b established a structural ceiling: r2_s_to_a = 0.996 when both streams
derive from hazard/resource proximity fields. SD-022 re-sources harm_obs_a from
the body's limb damage state -- a causally independent signal that persists even
when the agent leaves the hazardous area. This test verifies the fix works.

Two conditions (matched seeds):
  LIMB_DAMAGE_ENABLED  -- limb_damage_enabled=True, harm_obs_a from damage state [7 dims]
  LIMB_DAMAGE_DISABLED -- limb_damage_enabled=False, harm_obs_a from EMA proximity [50 dims]

Three matched seeds: 42, 7, 13

Training structure:
  150 warmup episodes (encoders learn representations)
  50 eval episodes (collect stream dissociation metrics)

Metrics (eval phase):
  harm_obs_a_variance:  variance of harm_obs_a signal across eval steps
  cosine_sim_harm_streams: mean cosine similarity between z_harm_s and z_harm_a
  r2_s_to_a: linear regression R2 predicting z_harm_a targets from z_harm_s

PRE-REGISTERED PASS CRITERION (for LIMB_DAMAGE_ENABLED condition, majority of seeds):
  r2_s_to_a < 0.5 AND harm_obs_a_variance > 0.001

Expected:
  LIMB_DAMAGE_ENABLED:  r2_s_to_a << 0.5 (streams causally independent)
  LIMB_DAMAGE_DISABLED: r2_s_to_a > 0.9  (streams structurally coupled, replicates EXQ-241b)

Claims: SD-022, SD-011
"""

import json
import sys
import random
import datetime
import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_318_sd022_limb_damage_stream_separation"
CLAIM_IDS = ["SD-022", "SD-011"]

# Pre-registered thresholds
THRESH_R2_S_TO_A_MAX = 0.5          # r2_s_to_a must be BELOW this in ENABLED condition
THRESH_HARM_OBS_A_VAR_MIN = 0.001   # harm_obs_a_variance must EXCEED this in ENABLED condition
PASS_MIN_SEEDS = 2                  # majority of 3 seeds must satisfy both thresholds

# Environment + architecture dims
WORLD_OBS_DIM = 250
HARM_OBS_DIM = 51    # sensory stream (world proximity)
HARM_OBS_A_DIM_ENABLED = 7   # SD-022: body damage state [4 dims + max + mean + residual_pain]
HARM_OBS_A_DIM_DISABLED = 50  # SD-011 EMA proximity (50-dim)
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16
ACTION_DIM = 4

# Training params
WARMUP_EPISODES = 150
EVAL_EPISODES = 50
STEPS_PER_EPISODE = 200
SEEDS = [42, 7, 13]
CONDITIONS = ["LIMB_DAMAGE_ENABLED", "LIMB_DAMAGE_DISABLED"]
LR = 1e-3


def make_env(condition: str, seed: int) -> CausalGridWorldV2:
    """Create environment for given condition."""
    limb_damage_enabled = (condition == "LIMB_DAMAGE_ENABLED")
    return CausalGridWorldV2(
        size=10,
        num_hazards=4,
        num_resources=3,
        hazard_harm=0.1,
        resource_benefit=0.05,
        use_proxy_fields=True,
        limb_damage_enabled=limb_damage_enabled,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        seed=seed,
    )


def make_config(condition: str, env: CausalGridWorldV2) -> REEConfig:
    """Create config matching environment dims for given condition."""
    harm_obs_a_dim = (
        HARM_OBS_A_DIM_ENABLED if condition == "LIMB_DAMAGE_ENABLED"
        else HARM_OBS_A_DIM_DISABLED
    )
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,   # 17 when limb_damage_enabled, else 12
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=harm_obs_a_dim,
        z_harm_a_dim=Z_HARM_A_DIM,
        limb_damage_enabled=(condition == "LIMB_DAMAGE_ENABLED"),
    )


def _linreg_r2(X: np.ndarray, y: np.ndarray) -> float:
    """Linear regression R2: predict y from X (columns). Returns R2 in [0, 1]."""
    if X.shape[0] < 5:
        return 0.0
    try:
        # Least squares: X @ w = y
        w, residuals, rank, sv = np.linalg.lstsq(
            np.hstack([X, np.ones((X.shape[0], 1))]),
            y, rcond=None
        )
        y_pred = np.hstack([X, np.ones((X.shape[0], 1))]) @ w
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot < 1e-12:
            return 0.0
        r2 = float(1.0 - ss_res / ss_tot)
        return float(np.clip(r2, 0.0, 1.0))
    except Exception:
        return 0.0


def run_condition(condition: str, seed: int, dry_run: bool = False) -> Dict:
    """Run one condition x seed. Returns metrics dict."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(condition, seed)
    cfg = make_config(condition, env)
    agent = REEAgent(cfg)
    device = agent.device

    # Build harm encoders
    harm_enc = HarmEncoder(
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    ).to(device)

    harm_obs_a_dim = (
        HARM_OBS_A_DIM_ENABLED if condition == "LIMB_DAMAGE_ENABLED"
        else HARM_OBS_A_DIM_DISABLED
    )
    harm_enc_a = AffectiveHarmEncoder(
        harm_obs_a_dim=harm_obs_a_dim,
        z_harm_a_dim=Z_HARM_A_DIM,
    ).to(device)

    # Optimizer over all learnable parameters
    all_params = (
        list(agent.parameters())
        + list(harm_enc.parameters())
        + list(harm_enc_a.parameters())
    )
    optimizer = optim.Adam(all_params, lr=LR)

    # Harm prediction target head (for training harm_enc)
    harm_head = nn.Sequential(
        nn.Linear(Z_HARM_DIM, 1),
        nn.Sigmoid(),
    ).to(device)
    optimizer_harm = optim.Adam(harm_head.parameters(), lr=LR)

    total_eps = WARMUP_EPISODES + EVAL_EPISODES
    eval_start = WARMUP_EPISODES

    # Eval-phase buffers for metric computation
    z_harm_s_buf: List[np.ndarray] = []
    z_harm_a_buf: List[np.ndarray] = []
    harm_obs_a_buf: List[np.ndarray] = []

    n_eps = 3 if dry_run else total_eps

    for ep in range(n_eps):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            harm_obs_s = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")

            if harm_obs_s is not None:
                harm_obs_s = harm_obs_s.to(device)
            if harm_obs_a is not None:
                harm_obs_a = harm_obs_a.to(device)

            # Encode harm streams
            z_harm_s = None
            z_harm_a = None
            if harm_obs_s is not None:
                z_harm_s = harm_enc(harm_obs_s.unsqueeze(0))
            if harm_obs_a is not None:
                z_harm_a_tuple = harm_enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = z_harm_a_tuple[0] if isinstance(z_harm_a_tuple, tuple) else z_harm_a_tuple

            # Sense (agent latent state update)
            latent = agent.sense(
                obs_body,
                obs_world,
                obs_harm=harm_obs_s,
                obs_harm_a=harm_obs_a,
            )

            # Action selection
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, cfg.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            # Eval: collect z_harm_s, z_harm_a, harm_obs_a for metrics
            if ep >= eval_start and z_harm_s is not None and z_harm_a is not None:
                z_harm_s_buf.append(z_harm_s.detach().cpu().numpy().flatten())
                z_harm_a_buf.append(z_harm_a.detach().cpu().numpy().flatten())
                if harm_obs_a is not None:
                    harm_obs_a_buf.append(harm_obs_a.detach().cpu().numpy().flatten())

            # Step environment
            flat_next, r, done, info, obs_dict_next = env.step(action_idx)

            # Training: prediction loss + harm proximity regression
            optimizer.zero_grad()
            optimizer_harm.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            total_loss = pred_loss
            if z_harm_s is not None and harm_obs_s is not None:
                # Auxiliary: predict harm_exposure from z_harm_s
                harm_target = harm_obs_s[-1:].unsqueeze(0)  # last dim = harm_exposure
                harm_pred = harm_head(z_harm_s)
                harm_loss = nn.functional.mse_loss(harm_pred, harm_target)
                total_loss = total_loss + harm_loss
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                optimizer_harm.step()

            obs_dict = obs_dict_next
            if done:
                break

        if not dry_run and (ep + 1) % 50 == 0:
            phase = "WARMUP" if ep < eval_start else "EVAL"
            print(
                f"  [{condition}] seed={seed} {phase} ep {ep+1}/{total_eps}"
            )

    # --- Compute metrics from eval buffers ---
    harm_obs_a_variance = 0.0
    cosine_sim_harm_streams = 0.0
    r2_s_to_a = 0.0

    if len(harm_obs_a_buf) > 0:
        all_harm_obs_a = np.stack(harm_obs_a_buf)  # [N, harm_obs_a_dim]
        harm_obs_a_variance = float(np.var(all_harm_obs_a))

    if len(z_harm_s_buf) > 2 and len(z_harm_a_buf) > 2:
        Z_s = np.stack(z_harm_s_buf)   # [N, z_harm_dim]
        Z_a = np.stack(z_harm_a_buf)   # [N, z_harm_a_dim]

        # Cosine similarity: mean across all pairs
        norms_s = np.linalg.norm(Z_s, axis=1, keepdims=True) + 1e-8
        norms_a = np.linalg.norm(Z_a, axis=1, keepdims=True) + 1e-8
        # z_harm_s [N, 32] and z_harm_a [N, 16] have different dims; use first min(dims) for cos sim
        min_dim = min(Z_s.shape[1], Z_a.shape[1])
        cos_sims = np.sum(
            (Z_s[:, :min_dim] / norms_s) * (Z_a[:, :min_dim] / norms_a), axis=1
        )
        cosine_sim_harm_streams = float(np.mean(cos_sims))

        # R2: predict z_harm_a dimensions from z_harm_s
        # Use mean of per-dimension R2 across z_harm_a dims
        r2_vals = []
        for dim_idx in range(Z_a.shape[1]):
            r2 = _linreg_r2(Z_s, Z_a[:, dim_idx])
            r2_vals.append(r2)
        r2_s_to_a = float(np.mean(r2_vals))

    result = {
        "condition": condition,
        "seed": seed,
        "harm_obs_a_variance": harm_obs_a_variance,
        "cosine_sim_harm_streams": cosine_sim_harm_streams,
        "r2_s_to_a": r2_s_to_a,
        "n_eval_steps": len(z_harm_s_buf),
        "harm_obs_a_dim": harm_obs_a_dim,
    }

    print(
        f"  [{condition}] seed={seed}: "
        f"r2_s_to_a={r2_s_to_a:.4f} "
        f"harm_obs_a_var={harm_obs_a_variance:.6f} "
        f"cos_sim={cosine_sim_harm_streams:.4f}"
    )
    return result


def evaluate_criteria(results: List[Dict]) -> Dict:
    """Evaluate PASS/FAIL against pre-registered criteria."""
    enabled_results = [r for r in results if r["condition"] == "LIMB_DAMAGE_ENABLED"]
    disabled_results = [r for r in results if r["condition"] == "LIMB_DAMAGE_DISABLED"]

    # Primary criterion: ENABLED condition must have low r2 AND non-trivial variance
    enabled_pass_per_seed = []
    for r in sorted(enabled_results, key=lambda x: x["seed"]):
        c_r2 = r["r2_s_to_a"] < THRESH_R2_S_TO_A_MAX
        c_var = r["harm_obs_a_variance"] > THRESH_HARM_OBS_A_VAR_MIN
        per_seed_pass = c_r2 and c_var
        enabled_pass_per_seed.append({
            "seed": r["seed"],
            "r2_s_to_a": r["r2_s_to_a"],
            "harm_obs_a_variance": r["harm_obs_a_variance"],
            "c_r2_pass": c_r2,
            "c_var_pass": c_var,
            "overall_pass": per_seed_pass,
        })

    n_seeds_pass = sum(1 for x in enabled_pass_per_seed if x["overall_pass"])
    primary_pass = n_seeds_pass >= PASS_MIN_SEEDS

    # Diagnostic: disabled condition should have high r2 (replicates EXQ-241b ceiling)
    disabled_r2_vals = [r["r2_s_to_a"] for r in disabled_results]
    disabled_replicates_ceiling = (
        len(disabled_r2_vals) > 0
        and float(np.mean(disabled_r2_vals)) > 0.5
    )

    mean_r2_enabled = float(np.mean([r["r2_s_to_a"] for r in enabled_results])) if enabled_results else 0.0
    mean_r2_disabled = float(np.mean(disabled_r2_vals)) if disabled_r2_vals else 0.0
    r2_reduction = mean_r2_disabled - mean_r2_enabled

    return {
        "overall_pass": primary_pass,
        "n_seeds_pass": n_seeds_pass,
        "pass_min_seeds": PASS_MIN_SEEDS,
        "enabled_pass_per_seed": enabled_pass_per_seed,
        "mean_r2_enabled": mean_r2_enabled,
        "mean_r2_disabled": mean_r2_disabled,
        "r2_reduction": r2_reduction,
        "disabled_replicates_ceiling": disabled_replicates_ceiling,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = int(time.time())
    run_id = (
        f"v3_exq_318_sd022_limb_damage_stream_separation_dry"
        if args.dry_run
        else f"v3_exq_318_sd022_limb_damage_stream_separation_{ts}_v3"
    )
    print(f"EXQ-318 start: {run_id}")

    all_results = []
    for seed in SEEDS:
        for condition in CONDITIONS:
            print(f"\n--- seed={seed} condition={condition} ---")
            result = run_condition(condition, seed, dry_run=args.dry_run)
            all_results.append(result)

    criteria = evaluate_criteria(all_results)
    outcome = "PASS" if criteria["overall_pass"] else "FAIL"

    print(f"\n=== EXQ-318 {outcome} ===")
    print(f"Seeds pass: {criteria['n_seeds_pass']}/{len(SEEDS)} (need {criteria['pass_min_seeds']})")
    print(f"Mean r2_s_to_a: ENABLED={criteria['mean_r2_enabled']:.4f} DISABLED={criteria['mean_r2_disabled']:.4f}")
    print(f"r2 reduction: {criteria['r2_reduction']:.4f}")
    print(f"Disabled replicates ceiling (>0.5): {criteria['disabled_replicates_ceiling']}")
    for entry in criteria["enabled_pass_per_seed"]:
        status = "PASS" if entry["overall_pass"] else "FAIL"
        print(
            f"  seed={entry['seed']}: {status} "
            f"r2={entry['r2_s_to_a']:.4f} var={entry['harm_obs_a_variance']:.6f} "
            f"(r2<{THRESH_R2_S_TO_A_MAX}:{entry['c_r2_pass']} "
            f"var>{THRESH_HARM_OBS_A_VAR_MIN}:{entry['c_var_pass']})"
        )

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "experiment_purpose": "diagnostic",
        "claim_ids": CLAIM_IDS,
        "evidence_direction_per_claim": {
            "SD-022": "supports" if criteria["overall_pass"] else "does_not_support",
            "SD-011": "supports" if criteria["overall_pass"] else "does_not_support",
        },
        "evidence_direction": "supports" if criteria["overall_pass"] else "does_not_support",
        "outcome": outcome,
        "criteria": criteria,
        "results_per_condition": all_results,
        "config": {
            "seeds": SEEDS,
            "conditions": CONDITIONS,
            "warmup_episodes": WARMUP_EPISODES,
            "eval_episodes": EVAL_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "thresh_r2_s_to_a_max": THRESH_R2_S_TO_A_MAX,
            "thresh_harm_obs_a_var_min": THRESH_HARM_OBS_A_VAR_MIN,
            "pass_min_seeds": PASS_MIN_SEEDS,
            "harm_obs_a_dim_enabled": HARM_OBS_A_DIM_ENABLED,
            "harm_obs_a_dim_disabled": HARM_OBS_A_DIM_DISABLED,
        },
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments"
        / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results -> {out_path}")


if __name__ == "__main__":
    main()
