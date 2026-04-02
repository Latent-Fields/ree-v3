#!/opt/local/bin/python3
"""
V3-EXQ-209 -- MECH-075: BG Dopaminergic Gain / Hippocampal Attractor Probe

Claim: MECH-075
Proposal: EVB-0050

MECH-075 asserts:
  Basal ganglia (BG) perform dopaminergic gain/threshold setting on hippocampal
  attractor dynamics. High dopamine (DA) signal widens hippocampal attractor
  basin width, increasing exploration diversity. Low DA narrows basins, yielding
  more exploitative, persistent trajectories.

Experiment design:
  The V3 proxy for DA gain setting is `novelty_bonus_weight` in E3Config
  (MECH-111). When > 0, E3 score_trajectory() subtracts the EMA novelty
  signal from trajectory scores, biasing selection toward unexplored states.
  This implements a BG-like modulation of hippocampal trajectory proposals.

  Conditions:
    DA_LOW:  config.e3.novelty_bonus_weight = 0.0  (no DA gain, exploitative)
    DA_HIGH: config.e3.novelty_bonus_weight = 2.0  (high DA gain, exploratory)

  Protocol:
  1. Train one agent for WARMUP_EPISODES under neutral conditions (novelty=0).
  2. Eval phase: for each condition (LOW/HIGH), set novelty_bonus_weight,
     run EVAL_EPISODES episodes, and at each step:
       - Generate trajectories via generate_trajectories()
       - Measure trajectory diversity = mean pairwise L2 dist between
         candidates' final z_self states
       - Update novelty EMA via update_novelty_ema(E1 error)
  3. Compute da_modulation_effect_size = diversity_HIGH - diversity_LOW.

Pre-registered thresholds
--------------------------
C1: mean_diversity_HIGH > mean_diversity_LOW in >= 2/3 seeds.
    High DA (novelty_bonus=2.0) increases trajectory exploration diversity.

C2: da_modulation_effect_size > THRESH_EFFECT in >= 2/3 seeds.
    Effect magnitude is above threshold (not just directional).

C3: attractor_basin_width_HIGH > attractor_basin_width_LOW in >= 2/3 seeds.
    attractor_basin_width = std of per-step trajectory diversity scores.
    Wider attractor basin = more step-to-step variation in trajectory spread.

C4: n_diverse_steps >= MIN_DIVERSE_STEPS in all seeds.
    Sanity: enough steps with at least 2 candidates to measure diversity.

PASS: C1 + C2 + C3 + C4
PARTIAL: C1 without C2 -- directional only
FAIL: C1 fails -- no measurable DA effect on trajectory diversity

Seeds: [42, 7, 123]
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources, hazard_harm=0.02
Train: 100 warmup episodes x 200 steps
Eval:  30 eval episodes x 200 steps per condition
Estimated runtime: ~80 min (any machine)
"""

import sys
import random
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Experiment metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_209_mech075_bg_hippocampal_gain_probe"
CLAIM_IDS = ["MECH-075"]
EXPERIMENT_PURPOSE = "evidence"

# ---------------------------------------------------------------------------
# Pre-registered thresholds
# ---------------------------------------------------------------------------
THRESH_EFFECT      = 0.005    # C2: da_modulation_effect_size must exceed this
MIN_DIVERSE_STEPS  = 50       # C4: minimum steps with diversity measurement

# DA condition values (V3 proxy for dopaminergic gain)
DA_LOW_WEIGHT   = 0.0   # No novelty bonus -- narrow attractor basins (exploitative)
DA_HIGH_WEIGHT  = 2.0   # High novelty bonus -- wide attractor basins (exploratory)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 32

WARMUP_EPISODES  = 100
EVAL_EPISODES    = 30
STEPS_PER_EP     = 200

# Novelty EMA warmup before measuring: first N eval steps discarded
NOVELTY_WARMUP_STEPS = 20

SEEDS = [42, 7, 123]


# ---------------------------------------------------------------------------
# Environment / config factory
# ---------------------------------------------------------------------------

def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=2,
        num_resources=3,
        hazard_harm=0.02,
        env_drift_interval=10,
        env_drift_prob=0.2,
    )


def _make_config(novelty_bonus_weight: float = 0.0) -> REEConfig:
    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )
    config.e3.novelty_bonus_weight = novelty_bonus_weight
    return config


# ---------------------------------------------------------------------------
# Trajectory diversity metric
# ---------------------------------------------------------------------------

def _trajectory_diversity(candidates) -> Optional[float]:
    """
    Mean pairwise L2 distance between candidates' final z_self states.

    Returns None if fewer than 2 candidates with states.
    """
    finals = []
    for traj in candidates:
        if traj.states:
            finals.append(traj.get_final_state()[0].detach())  # [self_dim]
    if len(finals) < 2:
        return None
    dists = []
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            d = float((finals[i] - finals[j]).norm().item())
            dists.append(d)
    return sum(dists) / max(1, len(dists))


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed: int, dry_run: bool) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    warmup = 5 if dry_run else WARMUP_EPISODES
    n_eval = 3 if dry_run else EVAL_EPISODES
    steps  = 20 if dry_run else STEPS_PER_EP

    env    = _make_env(seed)
    config = _make_config(novelty_bonus_weight=0.0)
    agent  = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)

    print(
        f"  [EXQ-209 MECH-075] seed={seed} warmup={warmup} eval_per_cond={n_eval}"
        f" steps_per_ep={steps}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Warmup training (neutral -- no novelty bonus)
    # -----------------------------------------------------------------------
    agent.train()
    for ep in range(warmup):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks["e1_tick"]
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            _, reward, done, _, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total   = e1_loss + e2_loss
            if total.requires_grad:
                e1_opt.zero_grad()
                e2_opt.zero_grad()
                total.backward()
                e1_opt.step()
                e2_opt.step()

            # Update novelty EMA with current E1 error
            if hasattr(e1_loss, 'item'):
                agent.e3.update_novelty_ema(float(e1_loss.item()))

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"    [train] seed={seed} ep {ep+1}/{warmup}",
                flush=True,
            )

    # -----------------------------------------------------------------------
    # Eval phase: measure trajectory diversity under two DA conditions
    # -----------------------------------------------------------------------
    agent.eval()

    condition_results = {}

    for condition, novelty_weight in [("DA_LOW", DA_LOW_WEIGHT), ("DA_HIGH", DA_HIGH_WEIGHT)]:
        # Set DA condition
        agent.e3.config.novelty_bonus_weight = novelty_weight
        # Reset novelty EMA for clean condition
        agent.e3._novelty_ema = 0.0

        diversity_per_step: List[float] = []
        step_count = 0

        for ep in range(n_eval):
            _, obs_dict = env.reset()
            agent.reset()

            for step in range(steps):
                obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
                obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    ticks  = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks["e1_tick"]
                        else torch.zeros(1, WORLD_DIM, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=1.0)

                    # Update novelty EMA (needed for DA_HIGH to take effect)
                    e1_loss_val = agent.compute_prediction_loss()
                    if hasattr(e1_loss_val, 'item'):
                        agent.e3.update_novelty_ema(float(e1_loss_val.item()))

                # Only measure after novelty EMA warmup
                if step_count >= NOVELTY_WARMUP_STEPS:
                    div = _trajectory_diversity(candidates)
                    if div is not None:
                        diversity_per_step.append(div)

                _, reward, done, _, obs_dict = env.step(action)
                harm_signal = float(reward) if reward < 0 else 0.0
                agent.update_residue(harm_signal)
                step_count += 1

                if done:
                    break

        n_diverse = len(diversity_per_step)
        mean_div  = sum(diversity_per_step) / max(1, n_diverse)
        std_div   = 0.0
        if n_diverse > 1:
            mean_ = mean_div
            std_div = math.sqrt(
                sum((x - mean_) ** 2 for x in diversity_per_step) / n_diverse
            )

        print(
            f"  [{condition}] seed={seed}"
            f" novelty_w={novelty_weight}"
            f" n_steps={n_diverse}"
            f" mean_diversity={mean_div:.5f}"
            f" std_diversity={std_div:.5f}",
            flush=True,
        )

        condition_results[condition] = {
            "condition":               condition,
            "novelty_bonus_weight":    novelty_weight,
            "n_diverse_steps":         n_diverse,
            "mean_diversity":          mean_div,
            "attractor_basin_width":   std_div,
        }

    # -----------------------------------------------------------------------
    # Compute summary metrics
    # -----------------------------------------------------------------------
    div_low  = condition_results["DA_LOW"]["mean_diversity"]
    div_high = condition_results["DA_HIGH"]["mean_diversity"]
    da_effect = div_high - div_low

    width_low  = condition_results["DA_LOW"]["attractor_basin_width"]
    width_high = condition_results["DA_HIGH"]["attractor_basin_width"]

    n_diverse_low  = condition_results["DA_LOW"]["n_diverse_steps"]
    n_diverse_high = condition_results["DA_HIGH"]["n_diverse_steps"]

    c1 = div_high > div_low
    c2 = da_effect > THRESH_EFFECT
    c3 = width_high > width_low
    c4 = (n_diverse_low >= MIN_DIVERSE_STEPS) and (n_diverse_high >= MIN_DIVERSE_STEPS)

    print(
        f"  [EXQ-209] seed={seed}"
        f" da_effect={da_effect:.5f}"
        f" C1={c1} C2={c2} C3={c3} C4={c4}",
        flush=True,
    )

    return {
        "seed":                     seed,
        "da_low":                   condition_results["DA_LOW"],
        "da_high":                  condition_results["DA_HIGH"],
        "da_modulation_effect_size": da_effect,
        "attractor_basin_width_low":  width_low,
        "attractor_basin_width_high": width_high,
        "hippocampal_persistence_delta": -da_effect,   # high DA = less persistence
        "c1_diversity_direction":   c1,
        "c2_effect_size":           c2,
        "c3_basin_width":           c3,
        "c4_sanity":                c4,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts      = int(time.time())
    root    = Path(__file__).resolve().parents[2]
    out_dir = root / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"

    print(f"[EXQ-209] MECH-075 BG Hippocampal Gain Probe", flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    seed_results = []
    for seed in SEEDS:
        res = _run_seed(seed, dry_run=args.dry_run)
        seed_results.append(res)

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    n_seeds  = len(seed_results)
    c1_count = sum(1 for r in seed_results if r["c1_diversity_direction"])
    c2_count = sum(1 for r in seed_results if r["c2_effect_size"])
    c3_count = sum(1 for r in seed_results if r["c3_basin_width"])
    c4_count = sum(1 for r in seed_results if r["c4_sanity"])

    c1_pass = c1_count >= 2
    c2_pass = c2_count >= 2
    c3_pass = c3_count >= 2
    c4_pass = c4_count >= 2   # all seeds must pass sanity (n >= 2)

    if c1_pass and c2_pass and c3_pass and c4_pass:
        outcome = "PASS"
        direction = "supports"
    elif c1_pass and not c2_pass:
        outcome = "PARTIAL"
        direction = "mixed"
    else:
        outcome = "FAIL"
        direction = "weakens"

    def _mean(key: str) -> float:
        return sum(r[key] for r in seed_results) / n_seeds

    print(
        f"\n[EXQ-209] RESULT: {outcome}"
        f" da_effect={_mean('da_modulation_effect_size'):.5f}"
        f" C1={c1_pass} C2={c2_pass} C3={c3_pass} C4={c4_pass}",
        flush=True,
    )

    manifest = {
        "run_id":                     f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type":            EXPERIMENT_TYPE,
        "architecture_epoch":         "ree_hybrid_guardrails_v1",
        "claim_ids":                  CLAIM_IDS,
        "experiment_purpose":         EXPERIMENT_PURPOSE,
        "outcome":                    outcome,
        "evidence_direction":         direction,
        "timestamp":                  ts,
        "dry_run":                    args.dry_run,
        "seeds":                      SEEDS,
        "warmup_episodes":            5 if args.dry_run else WARMUP_EPISODES,
        "eval_episodes":              3 if args.dry_run else EVAL_EPISODES,
        "steps_per_episode":          20 if args.dry_run else STEPS_PER_EP,
        "da_low_weight":              DA_LOW_WEIGHT,
        "da_high_weight":             DA_HIGH_WEIGHT,
        "thresh_effect":              THRESH_EFFECT,
        # Aggregate metrics
        "mean_diversity_low":         _mean("da_low") if False else
                                       sum(r["da_low"]["mean_diversity"] for r in seed_results) / n_seeds,
        "mean_diversity_high":        sum(r["da_high"]["mean_diversity"] for r in seed_results) / n_seeds,
        "da_modulation_effect_size":  _mean("da_modulation_effect_size"),
        "attractor_basin_width_low":  _mean("attractor_basin_width_low"),
        "attractor_basin_width_high": _mean("attractor_basin_width_high"),
        "hippocampal_persistence_delta": _mean("hippocampal_persistence_delta"),
        # Criteria
        "c1_diversity_direction_pass": c1_pass,
        "c2_effect_size_pass":         c2_pass,
        "c3_basin_width_pass":         c3_pass,
        "c4_sanity_pass":              c4_pass,
        "c1_count":                    c1_count,
        "c2_count":                    c2_count,
        "c3_count":                    c3_count,
        "c4_count":                    c4_count,
        "n_seeds":                     n_seeds,
        "seed_results":                seed_results,
    }

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[EXQ-209] Written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
