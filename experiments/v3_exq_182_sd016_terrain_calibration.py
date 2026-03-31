#!/opt/local/bin/python3
"""
EXQ-182: Supervised Terrain Calibration for MECH-150 / ARC-041

Tests whether adding a supervised terrain_loss to E1's training step makes
extract_cue_context() produce differentiated cue_context outputs across
hazard-proximate vs hazard-distal sensory contexts.

Background:
  EXQ-181/181b showed that without supervised training, ContextMemory outputs
  are effectively identical regardless of z_world context (cosine_sim=0.9999).
  The extract_cue_context() circuit is wired (SD-016 complete) but the
  projection pathway (world_query_proj -> attention -> cue_terrain_proj) receives
  no gradient signal because agent.py calls it with .detach().

Fix:
  Add a supervised terrain_loss during E1 training:
    hazard_max = hazard_field_view.max()
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.1 else 0.3
    terrain_loss = MSE(terrain_weight[:, 0], w_harm_target)
                 + MSE(terrain_weight[:, 1], w_goal_target)
    total_loss = prediction_loss + e2_loss + lambda_terrain * terrain_loss

  Gradient flows: terrain_loss -> cue_terrain_proj -> output_proj -> attention
  weights -> world_query_proj. This trains the full cue-indexed read pathway
  end-to-end. Memory slots get diverse content from EMA writes; the projections
  learn to extract context-relevant features.

Pass criteria:
  C1: cosine_sim(mean_cue_context_A, mean_cue_context_B) < 0.85
      MECH-150 core: cue context differentiates hazard-proximate from distal.
  C2: Pearson_r(w_harm, hazard_proximity) > 0.5
      terrain_weight[0] tracks hazard proximity.
  C3: Pearson_r(w_goal, hazard_proximity) < -0.3
      terrain_weight[1] inversely tracks hazard (goal upweighted when safe).
  PASS if all three pass in >= 2/3 seeds.
  MIXED if 1-2 criteria pass globally; FAIL if none.

Context thresholds (per EXQ-181b calibration for use_proxy_fields=True):
  Context A: hazard_field_view.max() > 0.7  (strongly hazard-proximate)
  Context B: hazard_field_view.max() < 0.33 (hazard-distal)

Claim IDs: MECH-150, ARC-041
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_182_sd016_terrain_calibration"
CLAIM_IDS = ["MECH-150", "ARC-041"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def get_hazard_max(obs_dict: Dict, world_obs: Optional[torch.Tensor]) -> float:
    """Extract hazard_field_view.max() from observation dict."""
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, 'shape') and harm_obs.shape[-1] >= 26:
            return float(harm_obs[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, 'shape'):
            return float(hfv.max().item())
    if world_obs is not None and world_obs.shape[-1] >= 225:
        return float(world_obs[..., 200:225].max().item())
    return 0.0


def pearson_r(x: List[float], y: List[float]) -> float:
    """Pearson correlation between two lists."""
    if len(x) < 4:
        return 0.0
    xa = np.array(x, dtype=np.float64)
    ya = np.array(y, dtype=np.float64)
    xa_mean = xa - xa.mean()
    ya_mean = ya - ya.mean()
    num = float(np.dot(xa_mean, ya_mean))
    denom = float(np.sqrt(np.dot(xa_mean, xa_mean) * np.dot(ya_mean, ya_mean)))
    if denom < 1e-12:
        return 0.0
    return num / denom


def compute_terrain_loss(
    agent: REEAgent,
    z_world: torch.Tensor,
    hazard_max: float,
) -> torch.Tensor:
    """
    Supervised terrain_loss for extract_cue_context() projections.

    Calls extract_cue_context WITH gradients (unlike agent.py which detaches).
    Returns MSE loss on terrain_weight vs proxy labels.
    """
    _, terrain_weight = agent.e1.extract_cue_context(z_world)
    # terrain_weight: [batch, 2] in (0, 1) via sigmoid

    # Proxy labels from hazard_field_view.max()
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.1 else 0.3

    target = torch.tensor(
        [[w_harm_target, w_goal_target]],
        dtype=terrain_weight.dtype,
        device=terrain_weight.device,
    )

    return F.mse_loss(terrain_weight, target)


# ------------------------------------------------------------------ #
# Single-seed runner                                                   #
# ------------------------------------------------------------------ #

def _run_seed(
    seed: int,
    warmup_episodes: int,
    collect_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    num_hazards: int,
    lambda_terrain: float,
) -> Dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed,
        size=10,
        num_hazards=num_hazards,
        num_resources=3,
        hazard_harm=0.5,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        hazard_field_decay=0.5,
        energy_decay=0.005,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )

    action_dim = env.action_dim

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
        novelty_bonus_weight=0.0,
        benefit_eval_enabled=False,
        sd016_enabled=True,
    )

    agent = REEAgent(config)
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # Verify SD-016 projections are present
    assert hasattr(agent.e1, 'world_query_proj'), \
        "SD-016 projections missing -- sd016_enabled not wiring correctly"

    # ---- Phase 0: Training warmup with terrain_loss ----
    agent.train()
    terrain_loss_history = []

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_terrain_losses = []

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()

            # Supervised terrain_loss through extract_cue_context()
            # Forward pass WITH gradients (agent.py uses .detach())
            hazard_max = get_hazard_max(obs_dict, obs_world)
            t_loss = compute_terrain_loss(agent, latent.z_world, hazard_max)

            total_loss = e1_loss + e2_loss + lambda_terrain * t_loss

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            ep_terrain_losses.append(float(t_loss.item()))

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            if done:
                break

        avg_t_loss = sum(ep_terrain_losses) / max(1, len(ep_terrain_losses))
        terrain_loss_history.append(avg_t_loss)

        if (ep + 1) % 40 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} ep {ep+1}/{warmup_episodes}"
                f" terrain_loss={avg_t_loss:.4f}",
                flush=True,
            )

    # ---- Phase 1: Context collection ----
    agent.eval()

    cue_contexts_A: List[torch.Tensor] = []
    cue_contexts_B: List[torch.Tensor] = []
    w_harm_vals: List[float] = []
    w_goal_vals: List[float] = []
    hazard_prox_vals: List[float] = []
    n_skipped = 0

    for ep in range(collect_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            z_world_det = latent.z_world.detach()

            # Extract cue context and terrain weight
            with torch.no_grad():
                action_bias, terrain_weight = agent.e1.extract_cue_context(z_world_det)

            # Intermediate cue_context for C1 metric (before projection heads)
            # Re-extract the pre-projection cue_context vector
            with torch.no_grad():
                batch_size = z_world_det.shape[0]
                memory = agent.e1.context_memory.memory
                q = agent.e1.world_query_proj(z_world_det).unsqueeze(1)
                k = agent.e1.context_memory.key_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
                v = agent.e1.context_memory.value_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
                scores = torch.bmm(q, k.transpose(1, 2)) / (agent.e1.context_memory.memory_dim ** 0.5)
                weights = F.softmax(scores, dim=-1)
                context = torch.bmm(weights, v).squeeze(1)
                cue_context = agent.e1.context_memory.output_proj(context)

            hazard_max = get_hazard_max(obs_dict, obs_world)

            # Record terrain_weight and hazard_prox for C2/C3
            w_harm_vals.append(float(terrain_weight[0, 0].item()))
            w_goal_vals.append(float(terrain_weight[0, 1].item()))
            hazard_prox_vals.append(hazard_max)

            # Context classification for C1
            if hazard_max > 0.7:
                cue_contexts_A.append(cue_context.squeeze(0).cpu())
            elif hazard_max < 0.33:
                cue_contexts_B.append(cue_context.squeeze(0).cpu())
            else:
                n_skipped += 1

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)

            if done:
                break

    n_A = len(cue_contexts_A)
    n_B = len(cue_contexts_B)

    print(
        f"  [collect] seed={seed}"
        f" n_context_A={n_A}"
        f" n_context_B={n_B}"
        f" n_skipped={n_skipped}",
        flush=True,
    )

    # ---- Phase 2: Metrics ----
    crit1_pass = False
    crit2_pass = False
    crit3_pass = False
    cos_sim_val = float("nan")
    r_w_harm = 0.0
    r_w_goal = 0.0

    # C1: cosine_sim of mean cue_context vectors
    if n_A >= 10 and n_B >= 10:
        mean_A = torch.stack(cue_contexts_A).mean(dim=0)
        mean_B = torch.stack(cue_contexts_B).mean(dim=0)
        cos_sim_val = float(
            F.cosine_similarity(mean_A.unsqueeze(0), mean_B.unsqueeze(0)).item()
        )
        crit1_pass = cos_sim_val < 0.85
        print(
            f"  [C1] seed={seed}"
            f" cosine_sim_AB={cos_sim_val:.4f}"
            f" threshold=0.85"
            f" pass={'YES' if crit1_pass else 'NO'}",
            flush=True,
        )
    else:
        print(
            f"  [C1] seed={seed} SKIP -- insufficient context steps"
            f" (n_A={n_A}, n_B={n_B}, need >=10 each)",
            flush=True,
        )

    # C2: Pearson r(w_harm, hazard_prox) > 0.5
    r_w_harm = pearson_r(w_harm_vals, hazard_prox_vals)
    crit2_pass = r_w_harm > 0.5
    print(
        f"  [C2] seed={seed}"
        f" r_w_harm={r_w_harm:.4f}"
        f" threshold=0.5"
        f" pass={'YES' if crit2_pass else 'NO'}",
        flush=True,
    )

    # C3: Pearson r(w_goal, hazard_prox) < -0.3
    r_w_goal = pearson_r(w_goal_vals, hazard_prox_vals)
    crit3_pass = r_w_goal < -0.3
    print(
        f"  [C3] seed={seed}"
        f" r_w_goal={r_w_goal:.4f}"
        f" threshold=-0.3"
        f" pass={'YES' if crit3_pass else 'NO'}",
        flush=True,
    )

    final_terrain_loss = terrain_loss_history[-1] if terrain_loss_history else float("nan")

    return {
        "seed": seed,
        "n_context_A": n_A,
        "n_context_B": n_B,
        "n_skipped": n_skipped,
        "cos_sim_AB": cos_sim_val if cos_sim_val == cos_sim_val else None,
        "r_w_harm": float(r_w_harm),
        "r_w_goal": float(r_w_goal),
        "final_terrain_loss": float(final_terrain_loss),
        "crit1_pass": int(crit1_pass),
        "crit2_pass": int(crit2_pass),
        "crit3_pass": int(crit3_pass),
    }


# ------------------------------------------------------------------ #
# Main run                                                             #
# ------------------------------------------------------------------ #

def run(
    seeds: Optional[List[int]] = None,
    warmup_episodes: int = 200,
    collect_episodes: int = 100,
    dry_run: bool = False,
    num_hazards: int = 1,
    lambda_terrain: float = 0.1,
) -> Dict:
    if seeds is None:
        seeds = [42, 7, 11]

    steps_per_episode = 200
    self_dim  = 32
    world_dim = 32
    lr        = 1e-3
    alpha_world = 0.9
    alpha_self  = 0.3

    per_seed_results: List[Dict] = []

    for seed in seeds:
        print(
            f"\n[V3-EXQ-182] seed={seed}"
            f" warmup={warmup_episodes}"
            f" collect={collect_episodes}"
            f" num_hazards={num_hazards}"
            f" lambda_terrain={lambda_terrain}"
            f" alpha_world={alpha_world}",
            flush=True,
        )
        r = _run_seed(
            seed=seed,
            warmup_episodes=warmup_episodes,
            collect_episodes=collect_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            lr=lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            num_hazards=num_hazards,
            lambda_terrain=lambda_terrain,
        )
        per_seed_results.append(r)

    seeds_c1_pass = sum(r["crit1_pass"] for r in per_seed_results)
    seeds_c2_pass = sum(r["crit2_pass"] for r in per_seed_results)
    seeds_c3_pass = sum(r["crit3_pass"] for r in per_seed_results)
    n_seeds       = len(per_seed_results)

    cos_vals = [
        r["cos_sim_AB"] for r in per_seed_results
        if r["cos_sim_AB"] is not None
    ]
    avg_cos_sim = float(sum(cos_vals) / len(cos_vals)) if cos_vals else float("nan")
    avg_r_w_harm = float(
        sum(r["r_w_harm"] for r in per_seed_results) / max(1, n_seeds)
    )
    avg_r_w_goal = float(
        sum(r["r_w_goal"] for r in per_seed_results) / max(1, n_seeds)
    )
    total_n_A = sum(r["n_context_A"] for r in per_seed_results)
    total_n_B = sum(r["n_context_B"] for r in per_seed_results)

    crit1_pass_global = seeds_c1_pass >= (n_seeds + 1) // 2
    crit2_pass_global = seeds_c2_pass >= (n_seeds + 1) // 2
    crit3_pass_global = seeds_c3_pass >= (n_seeds + 1) // 2

    all_pass = crit1_pass_global and crit2_pass_global and crit3_pass_global
    any_pass = crit1_pass_global or crit2_pass_global or crit3_pass_global

    if all_pass:
        status = "PASS"
        evidence_direction = "supports"
    elif any_pass:
        status = "MIXED"
        evidence_direction = "inconclusive"
    else:
        status = "FAIL"
        evidence_direction = "does_not_support"

    cos_display = f"{avg_cos_sim:.4f}" if avg_cos_sim == avg_cos_sim else "N/A"

    # Interpretations
    if crit1_pass_global:
        c1_interp = (
            f"C1 PASS: cosine_sim={cos_display} < 0.85 in {seeds_c1_pass}/{n_seeds} seeds."
            " Cue context differentiates hazard-proximate from hazard-distal states."
            " MECH-150 core claim validated: supervised terrain_loss trains the"
            " extract_cue_context() pathway to produce context-specific outputs."
        )
    else:
        c1_interp = (
            f"C1 FAIL: cosine_sim={cos_display} >= 0.85 (passed {seeds_c1_pass}/{n_seeds} seeds)."
            " Cue context does NOT differentiate despite supervised training."
            " Possible causes: lambda_terrain too low, insufficient warmup,"
            " or ContextMemory EMA slots lack sufficient diversity."
        )

    if crit2_pass_global:
        c2_interp = (
            f"C2 PASS: r_w_harm={avg_r_w_harm:.4f} > 0.5 in {seeds_c2_pass}/{n_seeds} seeds."
            " terrain_weight[0] (w_harm) tracks hazard proximity."
        )
    else:
        c2_interp = (
            f"C2 FAIL: r_w_harm={avg_r_w_harm:.4f} <= 0.5 (passed {seeds_c2_pass}/{n_seeds} seeds)."
            " w_harm does not reliably track hazard proximity."
        )

    if crit3_pass_global:
        c3_interp = (
            f"C3 PASS: r_w_goal={avg_r_w_goal:.4f} < -0.3 in {seeds_c3_pass}/{n_seeds} seeds."
            " terrain_weight[1] (w_goal) inversely tracks hazard -- goal scoring"
            " upweighted in safe contexts."
        )
    else:
        c3_interp = (
            f"C3 FAIL: r_w_goal={avg_r_w_goal:.4f} >= -0.3 (passed {seeds_c3_pass}/{n_seeds} seeds)."
            " w_goal does not inversely track hazard proximity."
        )

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" n_A={r['n_context_A']}"
        f" n_B={r['n_context_B']}"
        f" cos_sim={r['cos_sim_AB'] if r['cos_sim_AB'] is not None else 'N/A'}"
        f" r_w_harm={r['r_w_harm']:.4f}"
        f" r_w_goal={r['r_w_goal']:.4f}"
        f" terrain_loss={r['final_terrain_loss']:.4f}"
        f" C1={'PASS' if r['crit1_pass'] else 'FAIL'}"
        f" C2={'PASS' if r['crit2_pass'] else 'FAIL'}"
        f" C3={'PASS' if r['crit3_pass'] else 'FAIL'}"
        for r in per_seed_results
    )

    summary_markdown = (
        f"# V3-EXQ-182 -- Supervised Terrain Calibration (MECH-150 / ARC-041)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-150, ARC-041\n"
        f"**Seeds:** {seeds}\n\n"
        f"## Context\n\n"
        f"EXQ-181/181b showed that ContextMemory does not spontaneously produce"
        f" differentiated cue content (cosine_sim=0.9999 without supervised training)."
        f" This experiment adds a supervised terrain_loss (lambda={lambda_terrain})"
        f" using hazard_field_view.max() as proxy label, training the full"
        f" extract_cue_context() projection pathway end-to-end.\n\n"
        f"## Design\n\n"
        f"**Phase 0 (warmup):** {warmup_episodes} episodes x 200 steps. E1 + E2 losses"
        f" + terrain_loss (lambda={lambda_terrain}). Adam lr=1e-3, alpha_world=0.9,"
        f" sd016_enabled=True, num_hazards={num_hazards}.\n\n"
        f"**Phase 1 (collection):** {collect_episodes} episodes x 200 steps. Random actions."
        f" At each step: extract_cue_context(z_world) -> record cue_context, terrain_weight."
        f" Context A: hazard_max > 0.7. Context B: hazard_max < 0.33.\n\n"
        f"## Key Results\n\n"
        f"| Metric | Value | Threshold |\n"
        f"|---|---|---|\n"
        f"| cosine_sim(mean_A, mean_B) | {cos_display} | < 0.85 |\n"
        f"| r(w_harm, hazard_prox) | {avg_r_w_harm:.4f} | > 0.5 |\n"
        f"| r(w_goal, hazard_prox) | {avg_r_w_goal:.4f} | < -0.3 |\n"
        f"| n_context_A steps (total) | {total_n_A} | -- |\n"
        f"| n_context_B steps (total) | {total_n_B} | -- |\n\n"
        f"## Pass Criteria\n\n"
        f"| Criterion | Result | Seeds passing |\n"
        f"|---|---|---|\n"
        f"| C1: cosine_sim < 0.85 | {'PASS' if crit1_pass_global else 'FAIL'}"
        f" | {seeds_c1_pass}/{n_seeds} |\n"
        f"| C2: r(w_harm, hazard_prox) > 0.5 | {'PASS' if crit2_pass_global else 'FAIL'}"
        f" | {seeds_c2_pass}/{n_seeds} |\n"
        f"| C3: r(w_goal, hazard_prox) < -0.3 | {'PASS' if crit3_pass_global else 'FAIL'}"
        f" | {seeds_c3_pass}/{n_seeds} |\n\n"
        f"PASS rule: all three criteria pass in >= 2/3 seeds -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{c1_interp}\n\n"
        f"{c2_interp}\n\n"
        f"{c3_interp}\n\n"
        f"## Per-Seed Results\n\n"
        f"{per_seed_rows}\n"
    )

    metrics = {
        "cosine_sim_AB":      float(avg_cos_sim) if avg_cos_sim == avg_cos_sim else -1.0,
        "r_w_harm":           float(avg_r_w_harm),
        "r_w_goal":           float(avg_r_w_goal),
        "n_context_A_steps":  int(total_n_A),
        "n_context_B_steps":  int(total_n_B),
        "crit1_pass":         int(crit1_pass_global),
        "crit2_pass":         int(crit2_pass_global),
        "crit3_pass":         int(crit3_pass_global),
        "seeds_c1_pass":      int(seeds_c1_pass),
        "seeds_c2_pass":      int(seeds_c2_pass),
        "seeds_c3_pass":      int(seeds_c3_pass),
        "n_seeds":            int(n_seeds),
        "num_hazards":        int(num_hazards),
        "lambda_terrain":     float(lambda_terrain),
    }

    print(f"\n[V3-EXQ-182] Final results:", flush=True)
    print(f"  cosine_sim_AB={cos_display}", flush=True)
    print(f"  r_w_harm={avg_r_w_harm:.4f}  r_w_goal={avg_r_w_goal:.4f}", flush=True)
    print(f"  n_A={total_n_A}  n_B={total_n_B}", flush=True)
    print(
        f"  C1: {seeds_c1_pass}/{n_seeds}"
        f"  C2: {seeds_c2_pass}/{n_seeds}"
        f"  C3: {seeds_c3_pass}/{n_seeds}",
        flush=True,
    )
    print(f"  {c1_interp}", flush=True)
    print(f"  {c2_interp}", flush=True)
    print(f"  {c3_interp}", flush=True)
    print(f"  status={status}", flush=True)

    return {
        "status":             status,
        "metrics":            metrics,
        "summary_markdown":   summary_markdown,
        "claim_ids":          CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type":    EXPERIMENT_TYPE,
        "fatal_error_count":  0,
    }


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="EXQ-182: Supervised terrain calibration (MECH-150 / ARC-041)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11])
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--collect", type=int, default=100)
    parser.add_argument("--num-hazards", type=int, default=1)
    parser.add_argument("--lambda-terrain", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 1 seed, 3 warmup, 5 collect. No output file.")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds   = [42]
        args.warmup  = 3
        args.collect = 5
        print(
            "[DRY-RUN] 1 seed, 3 warmup, 5 collect,"
            f" num_hazards={args.num_hazards},"
            f" lambda_terrain={args.lambda_terrain}",
            flush=True,
        )

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        collect_episodes=args.collect,
        dry_run=args.dry_run,
        num_hazards=args.num_hazards,
        lambda_terrain=args.lambda_terrain,
    )

    if args.dry_run:
        print("\n[DRY-RUN] DRY-RUN complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            print(f"  {k}: {v}", flush=True)
        sys.exit(0)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"]      = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
