#!/opt/local/bin/python3
"""
V3-EXQ-319: SD-022 Harm Stream Dissociation -- Matched Threat, Different Damage History

experiment_purpose: evidence

Successor to EXQ-318/EXQ-241 line. Tests that z_harm_a (C-fiber analog, body damage state)
stays elevated after hazard offset while z_harm_s (A-delta analog, world-derived) drops --
in a matched-threat design where the only difference is damage history.

Two conditions per seed:
  HIGH_DAMAGE  -- pre-load agent with ~50 forced hazard transits, then bring to hazard,
                  then offset to safe area. z_harm_a should stay elevated.
  FRESH        -- no prior damage (clean limb state), same threat-on/offset protocol.
                  z_harm_a should be near zero post-offset.

Key dissociation: matched current threat (same hazard proximity during threat-on),
different recent damage history. After offset:
  z_harm_s_drop:          should be large in both (world signal clears)
  z_harm_a_retention_high: should stay elevated (damage persists in body state)
  z_harm_a_retention_fresh: should be near zero (no damage to retain)

Protocol phases:
  1. Warmup: 100 normal episodes (encoder initialisation)
  2. Pre-damage (HIGH_DAMAGE only): 50 forced hazard transits to accumulate limb damage
  3. Threat-on: 30 steps adjacent to hazard (both conditions)
  4. Harm-offset: 60 steps in safe area (both conditions)

Pass criterion (pre-registered):
  C1: z_harm_s drops >= 0.3 after offset (world signal clears)
  C2: z_harm_a_retention in HIGH_DAMAGE >= 0.02 (damage persists)
  C3: dissociation_score (high - fresh) >= 0.01
  D:  damage quality gate: mean limb_damage after pre-damage phase > 0.05

Experiment PASS: >= 3/5 seeds satisfy all per-seed criteria.

Claims: SD-022 (limb damage sourcing works as designed), SD-011 (dual streams independent)
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


EXPERIMENT_TYPE = "v3_exq_319_sd022_harm_stream_dissociation"
CLAIM_IDS = ["SD-022", "SD-011"]

# Pre-registered thresholds
C1_threshold = 0.3    # z_harm_s must drop by at least this after offset
C2_threshold = 0.02   # z_harm_a must stay above this in HIGH_DAMAGE post-offset
C3_threshold = 0.01   # dissociation_score must exceed this (high > fresh)
D_threshold  = 0.05   # damage quality gate: mean limb_damage must exceed this
PASS_MIN_SEEDS = 3    # majority of 5 seeds must satisfy all criteria

# Architecture constants
WORLD_OBS_DIM = 250
HARM_OBS_DIM = 51     # hazard_field(25) + resource_field(25) + harm_exposure(1)
HARM_OBS_A_DIM = 7    # SD-022 limb damage: damage[4] + max_damage + mean_damage + residual_pain
Z_HARM_DIM = 32
Z_HARM_A_DIM = 16
ACTION_DIM = 4

# Protocol params
SEEDS = [42, 43, 44, 45, 46]
WARMUP_EPISODES = 100
STEPS_PER_EPISODE = 200
PRE_DAMAGE_TRANSITS = 50    # forced hazard transits to accumulate limb damage
THREAT_ON_STEPS = 30        # steps in hazard proximity during threat-on phase
OFFSET_STEPS = 60           # steps in safe area during harm-offset phase
LR = 1e-3


def make_env(seed: int) -> CausalGridWorldV2:
    """Create environment with limb_damage_enabled=True."""
    return CausalGridWorldV2(
        size=10,
        num_hazards=4,
        num_resources=3,
        hazard_harm=0.1,
        resource_benefit=0.05,
        use_proxy_fields=True,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
        seed=seed,
    )


def make_config(env: CausalGridWorldV2) -> REEConfig:
    """Create REEConfig matching SD-022 limb-damage-enabled environment."""
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,   # 17 (proxy + limb_damage_enabled)
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        limb_damage_enabled=True,
    )


def _find_hazard_cell(env: CausalGridWorldV2) -> Optional[Tuple[int, int]]:
    """Return (x, y) of the first hazard cell, or None if none found."""
    for hx, hy in env.hazards:
        return (int(hx), int(hy))
    return None


def _find_safe_cell(env: CausalGridWorldV2) -> Tuple[int, int]:
    """Find a safe cell far from all hazards (max manhattan distance)."""
    best_pos = None
    best_dist = -1
    for i in range(1, env.size - 1):
        for j in range(1, env.size - 1):
            cell = env.grid[i, j]
            if cell not in (
                env.ENTITY_TYPES["wall"],
                env.ENTITY_TYPES["hazard"],
                env.ENTITY_TYPES["agent"],
            ):
                # Min manhattan distance to any hazard
                min_d = min(
                    abs(i - hx) + abs(j - hy)
                    for (hx, hy) in env.hazards
                ) if env.hazards else 99
                if min_d > best_dist:
                    best_dist = min_d
                    best_pos = (i, j)
    return best_pos if best_pos is not None else (1, 1)


def _teleport_agent(env: CausalGridWorldV2, target_x: int, target_y: int) -> None:
    """Teleport agent to target cell by directly updating env state."""
    # Clear current agent cell
    if env.contamination_grid[env.agent_x, env.agent_y] >= env.contamination_threshold:
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["contaminated"]
    else:
        env.grid[env.agent_x, env.agent_y] = env.ENTITY_TYPES["empty"]
    # Place agent at target
    env.agent_x = target_x
    env.agent_y = target_y
    env.grid[target_x, target_y] = env.ENTITY_TYPES["agent"]


def _get_harm_latents(
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_enc: HarmEncoder,
    harm_enc_a: AffectiveHarmEncoder,
    device: torch.device,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Collect z_harm (sensory) and z_harm_a (affective) from current env state."""
    obs_dict = env._get_observation_dict()
    obs_body = obs_dict["body_state"].to(device)
    obs_world = obs_dict["world_state"].to(device)
    harm_obs_s = obs_dict.get("harm_obs")
    harm_obs_a = obs_dict.get("harm_obs_a")

    if harm_obs_s is not None:
        harm_obs_s = harm_obs_s.to(device)
    if harm_obs_a is not None:
        harm_obs_a = harm_obs_a.to(device)

    # Encode harm streams (no grad: measurement phase)
    with torch.no_grad():
        z_harm_s = None
        z_harm_a = None
        if harm_obs_s is not None:
            z_harm_s = harm_enc(harm_obs_s.unsqueeze(0))
        if harm_obs_a is not None:
            res = harm_enc_a(harm_obs_a.unsqueeze(0))
            z_harm_a = res[0] if isinstance(res, tuple) else res
        # Update agent latent (needed for sense() internal state consistency)
        agent.sense(obs_body, obs_world, obs_harm=harm_obs_s, obs_harm_a=harm_obs_a)
    return z_harm_s, z_harm_a


def _norm(t: Optional[torch.Tensor]) -> float:
    """Return L2 norm of a tensor, or 0.0 if None."""
    if t is None:
        return 0.0
    return float(t.norm().item())


def run_warmup(
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_enc: HarmEncoder,
    harm_enc_a: AffectiveHarmEncoder,
    harm_head: nn.Module,
    optimizer: optim.Optimizer,
    optimizer_harm: optim.Optimizer,
    all_params: List,
    seed: int,
    device: torch.device,
    dry_run: bool = False,
) -> None:
    """Run warmup training episodes to initialise encoder representations."""
    n_eps = 3 if dry_run else WARMUP_EPISODES
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
            if harm_obs_s is not None:
                z_harm_s = harm_enc(harm_obs_s.unsqueeze(0))
            z_harm_a = None
            if harm_obs_a is not None:
                res = harm_enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res

            # Agent sense
            latent = agent.sense(obs_body, obs_world,
                                 obs_harm=harm_obs_s, obs_harm_a=harm_obs_a)

            # Action selection (training loop pattern from EXQ-318)
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            # Step env
            flat_next, r, done, info, obs_dict_next = env.step(action_idx)

            # Training: prediction loss + harm proximity regression
            optimizer.zero_grad()
            optimizer_harm.zero_grad()
            pred_loss = agent.compute_prediction_loss()
            total_loss = pred_loss
            if z_harm_s is not None and harm_obs_s is not None:
                harm_target = harm_obs_s[-1:].unsqueeze(0)
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

        if (ep + 1) % 50 == 0:
            print(f"  [train] seed={seed} warmup ep {ep+1}/{WARMUP_EPISODES}", flush=True)


def run_pre_damage_phase(
    env: CausalGridWorldV2,
    agent: REEAgent,
    device: torch.device,
    dry_run: bool = False,
) -> float:
    """
    Force agent through hazard transits to accumulate limb damage (HIGH_DAMAGE only).
    Uses env.reset() to get a fresh episode, then repeatedly moves through hazard cells.
    Returns mean limb_damage after the phase (quality gate).
    """
    n_transits = 3 if dry_run else PRE_DAMAGE_TRANSITS
    flat_obs, obs_dict = env.reset()
    agent.reset()

    hazard_cell = _find_hazard_cell(env)
    if hazard_cell is None:
        # No hazard found; skip -- quality gate will catch this
        return 0.0

    hx, hy = hazard_cell

    for transit in range(n_transits):
        # Teleport agent onto hazard cell
        _teleport_agent(env, hx, hy)
        # Step inside hazard: use action 4 (stay) to register contact harm
        # and accumulate limb damage without moving
        env.step(4)
        env.step(0)   # step away (north) to simulate transit
        env.step(1)   # step back (south, back onto / near hazard)
        env.step(4)   # stay on hazard
        env.step(4)   # one more stay for additional damage accumulation

    damage_mean = float(np.mean(env.limb_damage))
    return damage_mean


def run_threat_on_phase(
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_enc: HarmEncoder,
    harm_enc_a: AffectiveHarmEncoder,
    device: torch.device,
    dry_run: bool = False,
) -> Tuple[float, float]:
    """
    Place agent adjacent to hazard for 30 steps, collect z_harm_s and z_harm_a norms.
    Returns (z_harm_s_on_mean, z_harm_a_on_mean) averaged over last 10 steps.
    """
    n_steps = 5 if dry_run else THREAT_ON_STEPS

    hazard_cell = _find_hazard_cell(env)
    if hazard_cell is None:
        return 0.0, 0.0
    hx, hy = hazard_cell

    # Place agent adjacent (one step north of hazard)
    target_x = max(1, hx - 1)
    target_y = hy
    _teleport_agent(env, target_x, target_y)

    z_harm_s_vals = []
    z_harm_a_vals = []

    for step in range(n_steps):
        obs_dict = env._get_observation_dict()
        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)
        harm_obs_s = obs_dict.get("harm_obs")
        harm_obs_a = obs_dict.get("harm_obs_a")
        if harm_obs_s is not None:
            harm_obs_s = harm_obs_s.to(device)
        if harm_obs_a is not None:
            harm_obs_a = harm_obs_a.to(device)

        with torch.no_grad():
            z_harm_s = None
            z_harm_a = None
            if harm_obs_s is not None:
                z_harm_s = harm_enc(harm_obs_s.unsqueeze(0))
            if harm_obs_a is not None:
                res = harm_enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res
            agent.sense(obs_body, obs_world,
                        obs_harm=harm_obs_s, obs_harm_a=harm_obs_a)

        z_harm_s_vals.append(_norm(z_harm_s))
        z_harm_a_vals.append(_norm(z_harm_a))

        # Stay adjacent (oscillate: north/south to avoid leaving hazard proximity)
        action = 4  # stay
        env.step(action)

    # Mean of last 10 steps (or last 3 in dry run)
    tail = max(1, len(z_harm_s_vals) - 10) if not dry_run else max(1, len(z_harm_s_vals) - 3)
    z_harm_s_on_mean = float(np.mean(z_harm_s_vals[tail:]))
    z_harm_a_on_mean = float(np.mean(z_harm_a_vals[tail:]))
    return z_harm_s_on_mean, z_harm_a_on_mean


def run_offset_phase(
    env: CausalGridWorldV2,
    agent: REEAgent,
    harm_enc: HarmEncoder,
    harm_enc_a: AffectiveHarmEncoder,
    device: torch.device,
    dry_run: bool = False,
) -> Tuple[float, float]:
    """
    Move agent to safe cell, hold for 60 steps, collect z_harm_s and z_harm_a norms.
    Returns (z_harm_s_post_mean, z_harm_a_post_mean) averaged over steps 10-40.
    """
    n_steps = 10 if dry_run else OFFSET_STEPS

    safe_x, safe_y = _find_safe_cell(env)
    _teleport_agent(env, safe_x, safe_y)

    z_harm_s_vals = []
    z_harm_a_vals = []

    for step in range(n_steps):
        obs_dict = env._get_observation_dict()
        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)
        harm_obs_s = obs_dict.get("harm_obs")
        harm_obs_a = obs_dict.get("harm_obs_a")
        if harm_obs_s is not None:
            harm_obs_s = harm_obs_s.to(device)
        if harm_obs_a is not None:
            harm_obs_a = harm_obs_a.to(device)

        with torch.no_grad():
            z_harm_s = None
            z_harm_a = None
            if harm_obs_s is not None:
                z_harm_s = harm_enc(harm_obs_s.unsqueeze(0))
            if harm_obs_a is not None:
                res = harm_enc_a(harm_obs_a.unsqueeze(0))
                z_harm_a = res[0] if isinstance(res, tuple) else res
            agent.sense(obs_body, obs_world,
                        obs_harm=harm_obs_s, obs_harm_a=harm_obs_a)

        z_harm_s_vals.append(_norm(z_harm_s))
        z_harm_a_vals.append(_norm(z_harm_a))

        # Stay in safe area
        env.step(4)

    # Use steps 10-40 (or available range in dry run)
    if dry_run:
        start, end = 0, len(z_harm_s_vals)
    else:
        start, end = 10, min(40, len(z_harm_s_vals))
    if start >= end:
        start = 0
        end = len(z_harm_s_vals)

    z_harm_s_post_mean = float(np.mean(z_harm_s_vals[start:end])) if z_harm_s_vals[start:end] else 0.0
    z_harm_a_post_mean = float(np.mean(z_harm_a_vals[start:end])) if z_harm_a_vals[start:end] else 0.0
    return z_harm_s_post_mean, z_harm_a_post_mean


def run_seed(seed: int, dry_run: bool = False) -> Dict:
    """
    Run both conditions (HIGH_DAMAGE and FRESH) for one seed.
    Returns per-seed result dict.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Seed {seed}")

    env = make_env(seed)
    cfg = make_config(env)
    agent = REEAgent(cfg)
    device = agent.device

    # Build harm encoders
    harm_enc = HarmEncoder(
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
    ).to(device)
    harm_enc_a = AffectiveHarmEncoder(
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
    ).to(device)

    # Training head and optimizers
    harm_head = nn.Sequential(
        nn.Linear(Z_HARM_DIM, 1),
        nn.Sigmoid(),
    ).to(device)
    all_params = (
        list(agent.parameters())
        + list(harm_enc.parameters())
        + list(harm_enc_a.parameters())
    )
    optimizer = optim.Adam(all_params, lr=LR)
    optimizer_harm = optim.Adam(harm_head.parameters(), lr=LR)

    # --- Phase 1: warmup training ---
    run_warmup(
        env, agent, harm_enc, harm_enc_a, harm_head,
        optimizer, optimizer_harm, all_params,
        seed, device, dry_run=dry_run,
    )

    condition_results = {}
    for condition in ["HIGH_DAMAGE", "FRESH"]:
        print(f"Seed {seed} Condition {condition}")

        # Re-seed for reproducibility between conditions
        random.seed(seed + (1 if condition == "FRESH" else 0))
        np.random.seed(seed + (1 if condition == "FRESH" else 0))
        torch.manual_seed(seed + (1 if condition == "FRESH" else 0))

        # Reset env for a fresh episode
        flat_obs, obs_dict = env.reset()
        agent.reset()

        # --- Phase 2: pre-damage (HIGH_DAMAGE only) ---
        damage_vector_mean = 0.0
        if condition == "HIGH_DAMAGE":
            damage_vector_mean = run_pre_damage_phase(env, agent, device, dry_run=dry_run)
        # Ensure damage is reset for FRESH (env.reset() already zeros it, but be explicit)
        if condition == "FRESH":
            env.limb_damage[:] = 0.0
            damage_vector_mean = float(np.mean(env.limb_damage))

        # --- Phase 3: threat-on ---
        z_harm_s_on_mean, z_harm_a_on_mean = run_threat_on_phase(
            env, agent, harm_enc, harm_enc_a, device, dry_run=dry_run
        )

        # --- Phase 4: harm-offset ---
        z_harm_s_post_mean, z_harm_a_post_mean = run_offset_phase(
            env, agent, harm_enc, harm_enc_a, device, dry_run=dry_run
        )

        condition_results[condition] = {
            "z_harm_s_on_mean": z_harm_s_on_mean,
            "z_harm_a_on_mean": z_harm_a_on_mean,
            "z_harm_s_post_mean": z_harm_s_post_mean,
            "z_harm_a_post_mean": z_harm_a_post_mean,
            "damage_vector_mean": damage_vector_mean,
        }

    # --- Per-seed metrics ---
    high = condition_results["HIGH_DAMAGE"]
    fresh = condition_results["FRESH"]

    z_harm_s_drop_high = high["z_harm_s_on_mean"] - high["z_harm_s_post_mean"]
    z_harm_a_retention_high = high["z_harm_a_post_mean"]
    z_harm_a_retention_fresh = fresh["z_harm_a_post_mean"]
    dissociation_score = z_harm_a_retention_high - z_harm_a_retention_fresh
    damage_quality_gate = high["damage_vector_mean"]

    # Per-seed pass criteria
    c1_pass = z_harm_s_drop_high >= C1_threshold
    c2_pass = z_harm_a_retention_high >= C2_threshold
    c3_pass = dissociation_score >= C3_threshold
    d_pass  = damage_quality_gate >= D_threshold
    seed_pass = c1_pass and c2_pass and c3_pass and d_pass

    print(f"verdict: {'PASS' if seed_pass else 'FAIL'}")

    return {
        "seed": seed,
        "seed_pass": seed_pass,
        "z_harm_s_drop_high": z_harm_s_drop_high,
        "z_harm_a_retention_high": z_harm_a_retention_high,
        "z_harm_a_retention_fresh": z_harm_a_retention_fresh,
        "dissociation_score": dissociation_score,
        "damage_vector_mean_high": damage_quality_gate,
        "c1_z_harm_s_drop_pass": c1_pass,
        "c2_z_harm_a_retention_pass": c2_pass,
        "c3_dissociation_pass": c3_pass,
        "d_damage_quality_pass": d_pass,
        "condition_results": condition_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Short smoke test: 3 warmup eps, abbreviated protocol")
    args = parser.parse_args()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"v3_exq_319_sd022_harm_stream_dissociation_dry"
        if args.dry_run
        else f"v3_exq_319_sd022_harm_stream_dissociation_{timestamp}_v3"
    )
    print(f"EXQ-319 start: {run_id}")

    per_seed_results = []
    for seed in SEEDS:
        result = run_seed(seed, dry_run=args.dry_run)
        per_seed_results.append(result)

    seeds_passing = sum(1 for r in per_seed_results if r["seed_pass"])
    experiment_passes = seeds_passing >= PASS_MIN_SEEDS
    outcome = "PASS" if experiment_passes else "FAIL"

    print(f"\n=== EXQ-319 {outcome} ===")
    print(f"Seeds pass: {seeds_passing}/{len(SEEDS)} (need {PASS_MIN_SEEDS})")
    for r in per_seed_results:
        status = "PASS" if r["seed_pass"] else "FAIL"
        print(
            f"  seed={r['seed']}: {status} "
            f"s_drop={r['z_harm_s_drop_high']:.4f}(C1>={C1_threshold}) "
            f"a_ret_high={r['z_harm_a_retention_high']:.4f}(C2>={C2_threshold}) "
            f"dissoc={r['dissociation_score']:.4f}(C3>={C3_threshold}) "
            f"dmg={r['damage_vector_mean_high']:.4f}(D>={D_threshold})"
        )

    # Evidence direction
    if experiment_passes:
        evidence_direction = "supports"
        ev_sd022 = "supports"
        ev_sd011 = "supports"
    else:
        evidence_direction = "does_not_support"
        ev_sd022 = "does_not_support"
        ev_sd011 = "does_not_support"

    output = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": "evidence",
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {
            "SD-022": ev_sd022,
            "SD-011": ev_sd011,
        },
        "outcome": outcome,
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "registered_thresholds": {
            "C1_z_harm_s_drop": C1_threshold,
            "C2_z_harm_a_retention": C2_threshold,
            "C3_dissociation_score": C3_threshold,
            "D_damage_quality": D_threshold,
            "seeds_needed": PASS_MIN_SEEDS,
        },
        "per_seed_results": per_seed_results,
        "seeds_passing": seeds_passing,
        "experiment_passes": experiment_passes,
        "config": {
            "seeds": SEEDS,
            "warmup_episodes": WARMUP_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "pre_damage_transits": PRE_DAMAGE_TRANSITS,
            "threat_on_steps": THREAT_ON_STEPS,
            "offset_steps": OFFSET_STEPS,
            "harm_obs_a_dim": HARM_OBS_A_DIM,
            "limb_damage_enabled": True,
        },
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
