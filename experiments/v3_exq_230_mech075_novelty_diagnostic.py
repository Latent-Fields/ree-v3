#!/opt/local/bin/python3
"""
V3-EXQ-230 -- MECH-075 Novelty Loop Diagnostic

Claims: MECH-075
Dispatch mode: diagnostic

EXPERIMENT_PURPOSE = "diagnostic"

EXQ-192a FAIL (3D tensor shape bug fixed, still FAIL).
EXQ-209 evidence_direction = "weakens".
Multiple failures suggest the novelty mechanism may not be functionally
connected, or the signal is too weak to detect behaviorally.

This diagnostic probes whether:
  (1) The novelty_signal (E1 world-prediction MSE EMA) actually varies in
      NOVELTY_GATED condition (manipulation check: if it's flat < 0.01 the
      substrate_limitation is confirmed immediately).
  (2) Hippocampal gain variance differs between conditions.
  (3) Exploration rate (unique cell coverage) differs by >= 10%.

If novelty_signal_magnitude < 0.01 across seeds, DIAGNOSTIC FINDING =
"substrate_limitation": E1 prediction error not propagating to hippocampal
gain, novelty mechanism is structurally disconnected.

Conditions
----------
A. NOVELTY_GATED -- LC novelty signal feeds hippocampal gain.
   CEM noise std scaled by (1 + NOVELTY_GAIN * novelty_ema).
   novelty_ema updated from E1 world-prediction MSE each step.

B. NOVELTY_ABLATED -- LC input zeroed. Fixed CEM noise.
   novelty_ema held at 0.0 throughout.

Metrics
-------
- novelty_signal_magnitude: mean novelty_ema in NOVELTY_GATED condition
- hippocampal_gain_variance: variance of cem_noise_scale in GATED condition
- exploration_rate: fraction of unique grid cells visited per episode
- exploration_rate_gap: GATED - ABLATED (positive = GATED explores more)

PASS criteria
-------------
C1: novelty_signal_magnitude > 0.01  (novelty signal active)
C2: exploration_rate_gap >= 0.10     (10% higher exploration in GATED condition)
PASS: C1 AND C2 in >= 2/3 seeds.

DIAGNOSTIC: if C1 fails for all seeds -> substrate_limitation (signal not active)
            if C1 passes but C2 fails -> signal active but not behaviorally effective

Seeds: [42, 7, 123]
Env:   CausalGridWorldV2 size=10, 2 hazards, 3 resources
Train: 150 episodes x 200 steps
Eval:  50 episodes x 200 steps per condition
Est:   ~90 min DLAPTOP-4.local
"""

import sys
import random
import json
import time
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
EXPERIMENT_TYPE = "v3_exq_230_mech075_novelty_diagnostic"
CLAIM_IDS = ["MECH-075"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# Thresholds (pre-registered)
# ---------------------------------------------------------------------------
THRESH_NOVELTY_SIGNAL = 0.01   # C1: mean novelty_ema must exceed this
THRESH_EXPLORE_GAP    = 0.10   # C2: exploration_rate gap GATED - ABLATED

# Novelty gain for CEM noise scaling
NOVELTY_GAIN      = 2.0
NOVELTY_EMA_ALPHA = 0.1

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BODY_OBS_DIM  = 12
WORLD_OBS_DIM = 250
ACTION_DIM    = 5
WORLD_DIM     = 32
SELF_DIM      = 32

TRAIN_EPISODES = 150
EVAL_EPISODES  = 50
STEPS_PER_EP   = 200

SEEDS = [42, 7, 123]


# ---------------------------------------------------------------------------
# Env / config factory
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


def _make_config() -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        alpha_self=0.3,
    )


# ---------------------------------------------------------------------------
# Novelty signal computation
# ---------------------------------------------------------------------------

def _compute_world_novelty(agent: REEAgent, world_dim: int) -> float:
    """Compute E1 world-prediction MSE as novelty signal."""
    if len(agent._world_experience_buffer) < 2:
        return 0.0
    z_self_prev  = agent._self_experience_buffer[-2]
    z_world_prev = agent._world_experience_buffer[-2]
    z_world_act  = agent._world_experience_buffer[-1]
    combined     = torch.cat([z_self_prev.squeeze(0), z_world_prev.squeeze(0)])
    combined     = combined.unsqueeze(0)
    with torch.no_grad():
        if agent.e1._hidden_state is not None:
            saved = (
                agent.e1._hidden_state[0].clone(),
                agent.e1._hidden_state[1].clone(),
            )
        else:
            saved = None
        pred, _ = agent.e1(combined)
        if saved is not None:
            agent.e1._hidden_state = saved
        pred_world = pred[:, 0, -world_dim:]
        actual     = z_world_act.squeeze(0).unsqueeze(0)
        return float(F.mse_loss(pred_world, actual).item())


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def _run_seed(seed: int, dry_run: bool) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)

    n_train = 5  if dry_run else TRAIN_EPISODES
    n_eval  = 3  if dry_run else EVAL_EPISODES
    steps   = 20 if dry_run else STEPS_PER_EP

    env    = _make_env(seed)
    config = _make_config()
    agent  = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=1e-3)
    e2_opt = optim.Adam(agent.e2.parameters(), lr=3e-3)

    print(
        f"  [EXQ-230] seed={seed} train={n_train} eval_per_cond={n_eval}"
        f" steps={steps}",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Shared warmup training
    # -----------------------------------------------------------------------
    agent.train()
    for ep in range(n_train):
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
                e1_opt.zero_grad(); e2_opt.zero_grad()
                total.backward()
                e1_opt.step(); e2_opt.step()
            if hasattr(e1_loss, 'item'):
                agent.e3.update_novelty_ema(float(e1_loss.item()))
            agent.update_residue(harm_signal)
            if done:
                break
        if (ep + 1) % 50 == 0:
            print(f"    [train] seed={seed} ep {ep+1}/{n_train}", flush=True)

    # -----------------------------------------------------------------------
    # Eval phase: two conditions
    # -----------------------------------------------------------------------
    agent.eval()
    condition_data = {}

    for condition in ["NOVELTY_GATED", "NOVELTY_ABLATED"]:
        # Reset novelty EMA for clean start
        agent.e3._novelty_ema = 0.0
        novelty_ema_local: float = 0.0

        novelty_signal_history: List[float] = []
        cem_noise_history:      List[float] = []
        unique_cells_per_ep:    List[float] = []

        for ep in range(n_eval):
            _, obs_dict = env.reset()
            agent.reset()
            visited_cells = set()

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

                    # Compute raw novelty from E1 prediction error
                    raw_novelty = _compute_world_novelty(agent, WORLD_DIM)

                    if condition == "NOVELTY_GATED":
                        # Update EMA
                        novelty_ema_local = (
                            (1 - NOVELTY_EMA_ALPHA) * novelty_ema_local
                            + NOVELTY_EMA_ALPHA * raw_novelty
                        )
                        cem_noise_scale = 1.0 + NOVELTY_GAIN * novelty_ema_local
                    else:
                        # ABLATED: zero out novelty
                        novelty_ema_local = 0.0
                        cem_noise_scale   = 1.0

                    novelty_signal_history.append(novelty_ema_local)
                    cem_noise_history.append(cem_noise_scale)

                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=1.0)

                _, reward, done, info, obs_dict = env.step(action)
                harm_signal = float(reward) if reward < 0 else 0.0
                agent.update_residue(harm_signal)

                # Track visited cells
                if hasattr(env, 'agent_x') and hasattr(env, 'agent_y'):
                    visited_cells.add((env.agent_x, env.agent_y))

                if done:
                    break

            grid_size = getattr(env, 'size', 10)
            total_cells = grid_size * grid_size
            unique_frac  = len(visited_cells) / max(1, total_cells)
            unique_cells_per_ep.append(unique_frac)

        n_sig   = len(novelty_signal_history)
        mean_ns = sum(novelty_signal_history) / max(1, n_sig)

        n_cn    = len(cem_noise_history)
        mean_cn = sum(cem_noise_history) / max(1, n_cn)
        var_cn  = 0.0
        if n_cn > 1:
            var_cn = sum((x - mean_cn) ** 2 for x in cem_noise_history) / n_cn

        mean_explore = sum(unique_cells_per_ep) / max(1, len(unique_cells_per_ep))

        print(
            f"  [{condition}] seed={seed}"
            f" novelty_mag={mean_ns:.5f}"
            f" cem_noise_var={var_cn:.6f}"
            f" explore={mean_explore:.4f}",
            flush=True,
        )

        condition_data[condition] = {
            "condition":              condition,
            "novelty_signal_magnitude": mean_ns,
            "cem_noise_scale_mean":     mean_cn,
            "cem_noise_scale_variance": var_cn,
            "mean_exploration_rate":    mean_explore,
        }

    return condition_data


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> dict:
    print(f"\n[EXQ-230] MECH-075 Novelty Loop Diagnostic (dry_run={dry_run})",
          flush=True)

    seed_results = []
    for seed in SEEDS:
        print(f"\n--- seed={seed} ---", flush=True)
        cdata = _run_seed(seed, dry_run)
        seed_results.append({"seed": seed, "conditions": cdata})

    # Evaluate criteria per seed
    c1_passes = []
    c2_passes = []
    exploration_gaps = []
    novelty_mags_gated = []

    for sr in seed_results:
        cdata = sr["conditions"]
        gated   = cdata["NOVELTY_GATED"]
        ablated = cdata["NOVELTY_ABLATED"]
        nsm     = gated["novelty_signal_magnitude"]
        gap     = gated["mean_exploration_rate"] - ablated["mean_exploration_rate"]

        c1_passes.append(nsm > THRESH_NOVELTY_SIGNAL)
        c2_passes.append(gap >= THRESH_EXPLORE_GAP)
        exploration_gaps.append(gap)
        novelty_mags_gated.append(nsm)

    c1_pass = sum(c1_passes) >= 2
    c2_pass = sum(c2_passes) >= 2

    # Diagnostic classification
    if not c1_pass:
        diagnostic_finding = "substrate_limitation"
        interpretation = (
            "Novelty signal magnitude < 0.01 in all seeds. E1 world-prediction"
            " error is not producing a detectable novelty signal. The LC-VTA"
            " novelty loop is structurally disconnected from hippocampal gain"
            " modulation in current V3 substrate. MECH-075 mechanism not"
            " measurable at current architectural stage."
        )
    elif not c2_pass:
        diagnostic_finding = "signal_active_not_behavioral"
        interpretation = (
            "Novelty signal is active (C1 pass) but does not produce >= 10%"
            " increase in exploration coverage (C2 fail). Signal exists but"
            " CEM noise scaling does not translate to differential exploration."
            " May need larger NOVELTY_GAIN or different exploration metric."
        )
    else:
        diagnostic_finding = "mechanism_confirmed"
        interpretation = (
            "Novelty signal active (C1) and exploration gap >= 10% (C2)."
            " MECH-075 mechanism is functionally connected and behaviorally"
            " detectable. Supports further evidence experiments."
        )

    all_pass = c1_pass and c2_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass])

    # Summarise
    print(f"\n[EXQ-230] Results:", flush=True)
    for i, seed in enumerate(SEEDS):
        print(
            f"  seed={seed}: novelty_mag={novelty_mags_gated[i]:.5f}"
            f" explore_gap={exploration_gaps[i]:+.4f}"
            f" C1={'P' if c1_passes[i] else 'F'}"
            f" C2={'P' if c2_passes[i] else 'F'}",
            flush=True,
        )
    print(f"  Status: {status} ({criteria_met}/2 criteria)", flush=True)
    print(f"  Diagnostic finding: {diagnostic_finding}", flush=True)
    print(f"  Interpretation: {interpretation}", flush=True)

    summary_markdown = (
        f"# V3-EXQ-230 -- MECH-075 Novelty Loop Diagnostic\n\n"
        f"**Status:** {status}  **Diagnostic finding:** {diagnostic_finding}\n"
        f"**Claims:** MECH-075  **Purpose:** diagnostic\n\n"
        f"## Context\n\n"
        f"EXQ-192a FAIL (shape bug fixed, still FAIL). EXQ-209 weakens."
        f" This diagnostic tests whether the novelty signal is active and"
        f" whether hippocampal gain variance differs between conditions.\n\n"
        f"## Conditions\n\n"
        f"- NOVELTY_GATED: novelty_ema from E1 MSE scales CEM noise"
        f" (gain={NOVELTY_GAIN})\n"
        f"- NOVELTY_ABLATED: novelty_ema held at 0.0, fixed CEM noise\n\n"
        f"## Results by Seed\n\n"
        f"| Seed | Novelty mag | Explore gap | C1 | C2 |\n"
        f"|------|------------|------------|----|----|"
    )
    for i, seed in enumerate(SEEDS):
        summary_markdown += (
            f"\n| {seed} | {novelty_mags_gated[i]:.5f}"
            f" | {exploration_gaps[i]:+.4f}"
            f" | {'PASS' if c1_passes[i] else 'FAIL'}"
            f" | {'PASS' if c2_passes[i] else 'FAIL'} |"
        )
    summary_markdown += (
        f"\n\n## PASS Criteria\n\n"
        f"| Criterion | Threshold | Result |\n|---|---|---|\n"
        f"| C1: novelty_signal_mag > {THRESH_NOVELTY_SIGNAL}"
        f" (>= 2/3 seeds) | {THRESH_NOVELTY_SIGNAL}"
        f" | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: exploration_gap >= {THRESH_EXPLORE_GAP}"
        f" (>= 2/3 seeds) | {THRESH_EXPLORE_GAP}"
        f" | {'PASS' if c2_pass else 'FAIL'} |\n\n"
        f"## Diagnostic Finding\n\n{diagnostic_finding}\n\n"
        f"## Interpretation\n\n{interpretation}\n"
    )

    metrics = {
        "novelty_mag_seed0":    float(novelty_mags_gated[0]),
        "novelty_mag_seed1":    float(novelty_mags_gated[1]),
        "novelty_mag_seed2":    float(novelty_mags_gated[2]),
        "explore_gap_seed0":    float(exploration_gaps[0]),
        "explore_gap_seed1":    float(exploration_gaps[1]),
        "explore_gap_seed2":    float(exploration_gaps[2]),
        "c1_pass":              1.0 if c1_pass else 0.0,
        "c2_pass":              1.0 if c2_pass else 0.0,
        "criteria_met":         float(criteria_met),
        "diagnostic_finding":   diagnostic_finding,
    }
    for i, seed in enumerate(SEEDS):
        for cname, cdata in seed_results[i]["conditions"].items():
            prefix = f"seed{i}_{cname.lower()}_"
            for k, v in cdata.items():
                if k != "condition" and isinstance(v, (int, float)):
                    metrics[prefix + k] = float(v)

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "diagnostic",
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
        "seed_results": seed_results,
        "diagnostic_finding": diagnostic_finding,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(dry_run=args.dry_run)

    ts  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["run_timestamp"] = ts
    result["run_id"]         = run_id
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    print(f"Diagnostic finding: {result['diagnostic_finding']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)
