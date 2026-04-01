#!/opt/local/bin/python3
"""
EXQ-194a: Direct z_world Terrain Pathway with Phased Training (MECH-152)

Supersedes EXQ-194 (MIXED: C1 PASS 3/3, C2 FAIL 1/3, C3 PASS 3/3).

Two fixes:
  (1) Phased training. EXQ-194 trained TerrainHead jointly with E1+E2 from step 0.
      The 166e series established that downstream heads chasing a moving latent target
      during encoder convergence produce degenerate or unstable mappings. Fix:
        Phase 0: E1+E2 warmup ONLY (200 eps). No terrain_loss. Let z_world converge.
        Phase 1: Freeze E1+E2. Train TerrainHead on z_world.detach() (100 eps).
                 The target representation is now stable.
        Phase 2: Collection and evaluation (100 eps).

  (2) w_goal label threshold. With num_hazards=1 and use_proxy_fields=True, the hazard
      field minimum is ~0.22. EXQ-194 used w_goal_target = 0.8 if hazard_max < 0.1,
      which nearly never fires -- the network learned w_goal ~ 0.3 as a constant.
      Fix: w_goal_target = 0.8 if hazard_max < 0.33 else 0.2 (matches context-B
      distribution from EXQ-181b calibration: ~25% of steps).

Phased training protocol (hard-won lessons from 166e, 085l, 085m series):
  - Downstream heads MUST train on stabilised latents, not on latents still being learned.
  - The encoder's loss landscape changes as E1/E2 learn; features discovered early may
    vanish or shift as training continues.
  - Freezing the encoder during downstream head training prevents catastrophic chasing.
  - This is now a STANDARD PROTOCOL for all experiments training heads on z_world/z_harm.

Pass criteria:
  C1: Pearson r(w_harm, hazard_proximity) > 0.5
  C2: Pearson r(w_goal, hazard_proximity) < -0.3
  C3: final terrain_loss < 0.05 (converged)
  PASS if all three pass in >= 2/3 seeds.

Claim IDs: MECH-152
Supersedes: v3_exq_194_direct_terrain_pathway
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_194a_direct_terrain_phased"
CLAIM_IDS = ["MECH-152"]


# ------------------------------------------------------------------ #
# Terrain head (standalone, bypasses ContextMemory)                    #
# ------------------------------------------------------------------ #

class TerrainHead(nn.Module):
    """
    Direct z_world -> terrain_weight pathway.
    Learns [w_harm, w_goal] from exteroceptive latent alone.
    """
    def __init__(self, world_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(world_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        """Returns [batch, 2] in (0, 1): [w_harm, w_goal]."""
        return self.net(z_world)


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
    terrain_head: TerrainHead,
    z_world: torch.Tensor,
    hazard_max: float,
) -> torch.Tensor:
    """
    Supervised terrain_loss with corrected w_goal threshold.
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2  (unchanged)
    w_goal_target = 0.8 if hazard_max < 0.33 else 0.2  (FIXED: was < 0.1)
    """
    terrain_weight = terrain_head(z_world)  # [batch, 2]

    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.33 else 0.2  # FIXED threshold

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
    p0_episodes: int,
    p1_episodes: int,
    p2_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    terrain_lr: float,
    alpha_world: float,
    alpha_self: float,
    num_hazards: int,
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
    )

    agent = REEAgent(config)
    terrain_head = TerrainHead(world_dim=world_dim)

    # Separate optimizers for phased training
    agent_optimizer = optim.Adam(agent.parameters(), lr=lr)
    terrain_optimizer = optim.Adam(terrain_head.parameters(), lr=terrain_lr)

    # ================================================================ #
    # Phase 0: E1+E2 warmup ONLY -- let z_world converge               #
    # ================================================================ #
    agent.train()
    terrain_head.eval()  # terrain_head NOT trained in P0

    for ep in range(p0_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss

            if total_loss.requires_grad:
                agent_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                agent_optimizer.step()

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            if done:
                break

        if (ep + 1) % 40 == 0 or ep == p0_episodes - 1:
            print(
                f"  [P0-warmup] seed={seed} ep {ep+1}/{p0_episodes}",
                flush=True,
            )

    # ================================================================ #
    # Phase 1: Freeze E1+E2, train TerrainHead on stable z_world       #
    # ================================================================ #
    agent.eval()  # Freeze encoder -- no gradient updates to E1/E2
    terrain_head.train()

    terrain_loss_history = []

    for ep in range(p1_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_terrain_losses = []

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            # Forward through encoder with no_grad -- z_world is fixed
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            # Train terrain_head on detached z_world
            z_world_stable = latent.z_world.detach()
            hazard_max = get_hazard_max(obs_dict, obs_world)
            t_loss = compute_terrain_loss(terrain_head, z_world_stable, hazard_max)

            terrain_optimizer.zero_grad()
            t_loss.backward()
            torch.nn.utils.clip_grad_norm_(terrain_head.parameters(), 1.0)
            terrain_optimizer.step()

            ep_terrain_losses.append(float(t_loss.item()))

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)

            if done:
                break

        avg_t_loss = sum(ep_terrain_losses) / max(1, len(ep_terrain_losses))
        terrain_loss_history.append(avg_t_loss)

        if (ep + 1) % 20 == 0 or ep == p1_episodes - 1:
            print(
                f"  [P1-terrain] seed={seed} ep {ep+1}/{p1_episodes}"
                f" terrain_loss={avg_t_loss:.4f}",
                flush=True,
            )

    # ================================================================ #
    # Phase 2: Collection and evaluation                                #
    # ================================================================ #
    agent.eval()
    terrain_head.eval()

    w_harm_vals: List[float] = []
    w_goal_vals: List[float] = []
    hazard_prox_vals: List[float] = []
    n_context_A = 0
    n_context_B = 0
    n_skipped = 0

    for ep in range(p2_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            with torch.no_grad():
                terrain_weight = terrain_head(latent.z_world.detach())

            hazard_max = get_hazard_max(obs_dict, obs_world)

            w_harm_vals.append(float(terrain_weight[0, 0].item()))
            w_goal_vals.append(float(terrain_weight[0, 1].item()))
            hazard_prox_vals.append(hazard_max)

            if hazard_max > 0.7:
                n_context_A += 1
            elif hazard_max < 0.33:
                n_context_B += 1
            else:
                n_skipped += 1

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)

            if done:
                break

    print(
        f"  [P2-collect] seed={seed}"
        f" n_context_A={n_context_A}"
        f" n_context_B={n_context_B}"
        f" n_skipped={n_skipped}"
        f" total_steps={len(w_harm_vals)}",
        flush=True,
    )

    # ---- Metrics ----
    # C1: r(w_harm, hazard_prox) > 0.5
    r_w_harm = pearson_r(w_harm_vals, hazard_prox_vals)
    crit1_pass = r_w_harm > 0.5
    print(
        f"  [C1] seed={seed}"
        f" r_w_harm={r_w_harm:.4f}"
        f" threshold=0.5"
        f" pass={'YES' if crit1_pass else 'NO'}",
        flush=True,
    )

    # C2: r(w_goal, hazard_prox) < -0.3
    r_w_goal = pearson_r(w_goal_vals, hazard_prox_vals)
    crit2_pass = r_w_goal < -0.3
    print(
        f"  [C2] seed={seed}"
        f" r_w_goal={r_w_goal:.4f}"
        f" threshold=-0.3"
        f" pass={'YES' if crit2_pass else 'NO'}",
        flush=True,
    )

    # C3: final terrain_loss < 0.05
    final_terrain_loss = terrain_loss_history[-1] if terrain_loss_history else float("nan")
    crit3_pass = final_terrain_loss < 0.05
    print(
        f"  [C3] seed={seed}"
        f" final_terrain_loss={final_terrain_loss:.4f}"
        f" threshold=0.05"
        f" pass={'YES' if crit3_pass else 'NO'}",
        flush=True,
    )

    # Per-context diagnostic means
    w_harm_A = [w for w, h in zip(w_harm_vals, hazard_prox_vals) if h > 0.7]
    w_harm_B = [w for w, h in zip(w_harm_vals, hazard_prox_vals) if h < 0.33]
    w_goal_A = [w for w, h in zip(w_goal_vals, hazard_prox_vals) if h > 0.7]
    w_goal_B = [w for w, h in zip(w_goal_vals, hazard_prox_vals) if h < 0.33]

    mean_w_harm_A = float(np.mean(w_harm_A)) if w_harm_A else float("nan")
    mean_w_harm_B = float(np.mean(w_harm_B)) if w_harm_B else float("nan")
    mean_w_goal_A = float(np.mean(w_goal_A)) if w_goal_A else float("nan")
    mean_w_goal_B = float(np.mean(w_goal_B)) if w_goal_B else float("nan")

    print(
        f"  [diag] seed={seed}"
        f" w_harm: A={mean_w_harm_A:.3f} B={mean_w_harm_B:.3f}"
        f" w_goal: A={mean_w_goal_A:.3f} B={mean_w_goal_B:.3f}",
        flush=True,
    )

    return {
        "seed": seed,
        "n_context_A": n_context_A,
        "n_context_B": n_context_B,
        "n_skipped": n_skipped,
        "r_w_harm": float(r_w_harm),
        "r_w_goal": float(r_w_goal),
        "final_terrain_loss": float(final_terrain_loss),
        "mean_w_harm_A": mean_w_harm_A,
        "mean_w_harm_B": mean_w_harm_B,
        "mean_w_goal_A": mean_w_goal_A,
        "mean_w_goal_B": mean_w_goal_B,
        "crit1_pass": int(crit1_pass),
        "crit2_pass": int(crit2_pass),
        "crit3_pass": int(crit3_pass),
    }


# ------------------------------------------------------------------ #
# Main run                                                             #
# ------------------------------------------------------------------ #

def run(
    seeds: Optional[List[int]] = None,
    p0_episodes: int = 200,
    p1_episodes: int = 100,
    p2_episodes: int = 100,
    dry_run: bool = False,
    num_hazards: int = 1,
) -> Dict:
    if seeds is None:
        seeds = [42, 7, 11]

    steps_per_episode = 200
    self_dim  = 32
    world_dim = 32
    lr        = 1e-3
    terrain_lr = 1e-3
    alpha_world = 0.9
    alpha_self  = 0.3

    per_seed_results: List[Dict] = []

    for seed in seeds:
        print(
            f"\n[V3-EXQ-194a] seed={seed}"
            f" P0={p0_episodes} P1={p1_episodes} P2={p2_episodes}"
            f" num_hazards={num_hazards}"
            f" alpha_world={alpha_world}",
            flush=True,
        )
        r = _run_seed(
            seed=seed,
            p0_episodes=p0_episodes,
            p1_episodes=p1_episodes,
            p2_episodes=p2_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            lr=lr,
            terrain_lr=terrain_lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            num_hazards=num_hazards,
        )
        per_seed_results.append(r)

    seeds_c1_pass = sum(r["crit1_pass"] for r in per_seed_results)
    seeds_c2_pass = sum(r["crit2_pass"] for r in per_seed_results)
    seeds_c3_pass = sum(r["crit3_pass"] for r in per_seed_results)
    n_seeds       = len(per_seed_results)

    avg_r_w_harm = float(
        sum(r["r_w_harm"] for r in per_seed_results) / max(1, n_seeds)
    )
    avg_r_w_goal = float(
        sum(r["r_w_goal"] for r in per_seed_results) / max(1, n_seeds)
    )
    avg_final_t_loss = float(
        sum(r["final_terrain_loss"] for r in per_seed_results) / max(1, n_seeds)
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

    # Interpretations
    if crit1_pass_global:
        c1_interp = (
            f"C1 PASS: r_w_harm={avg_r_w_harm:.4f} > 0.5 in {seeds_c1_pass}/{n_seeds} seeds."
            " Direct pathway w_harm tracks hazard proximity on stable z_world."
        )
    else:
        c1_interp = (
            f"C1 FAIL: r_w_harm={avg_r_w_harm:.4f} <= 0.5 (passed {seeds_c1_pass}/{n_seeds} seeds)."
            " Direct pathway w_harm does not track hazard proximity even on stable z_world."
        )

    if crit2_pass_global:
        c2_interp = (
            f"C2 PASS: r_w_goal={avg_r_w_goal:.4f} < -0.3 in {seeds_c2_pass}/{n_seeds} seeds."
            " Direct pathway w_goal inversely tracks hazard -- goal upweighted when safe."
            " Corrected threshold (< 0.33) provides sufficient goal-high label coverage."
        )
    else:
        c2_interp = (
            f"C2 FAIL: r_w_goal={avg_r_w_goal:.4f} >= -0.3 (passed {seeds_c2_pass}/{n_seeds} seeds)."
            " Direct pathway w_goal does not inversely track hazard proximity."
        )

    if crit3_pass_global:
        c3_interp = (
            f"C3 PASS: final_terrain_loss={avg_final_t_loss:.4f} < 0.05 in {seeds_c3_pass}/{n_seeds} seeds."
            " Terrain head converged on stable z_world."
        )
    else:
        c3_interp = (
            f"C3 FAIL: final_terrain_loss={avg_final_t_loss:.4f} >= 0.05 (passed {seeds_c3_pass}/{n_seeds} seeds)."
            " Terrain head did not converge."
        )

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" n_A={r['n_context_A']}"
        f" n_B={r['n_context_B']}"
        f" r_w_harm={r['r_w_harm']:.4f}"
        f" r_w_goal={r['r_w_goal']:.4f}"
        f" terrain_loss={r['final_terrain_loss']:.4f}"
        f" w_harm: A={r['mean_w_harm_A']:.3f} B={r['mean_w_harm_B']:.3f}"
        f" w_goal: A={r['mean_w_goal_A']:.3f} B={r['mean_w_goal_B']:.3f}"
        f" C1={'PASS' if r['crit1_pass'] else 'FAIL'}"
        f" C2={'PASS' if r['crit2_pass'] else 'FAIL'}"
        f" C3={'PASS' if r['crit3_pass'] else 'FAIL'}"
        for r in per_seed_results
    )

    summary_markdown = (
        f"# V3-EXQ-194a -- Direct Terrain Pathway, Phased Training (MECH-152)\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-152\n"
        f"**Supersedes:** V3-EXQ-194\n"
        f"**Seeds:** {seeds}\n\n"
        f"## Context\n\n"
        f"EXQ-194 MIXED: C1 PASS (r_w_harm=0.70) but C2 FAIL (r_w_goal=-0.007)."
        f" Two fixes: (1) phased training (P0 E1+E2 warmup, P1 frozen-encoder"
        f" TerrainHead) per 166e lesson; (2) corrected w_goal threshold"
        f" (< 0.33 not < 0.1) to produce non-trivial goal-high label distribution.\n\n"
        f"## Design\n\n"
        f"**Phase 0 (E1+E2 warmup):** {p0_episodes} episodes x 200 steps."
        f" E1 + E2 losses only. No terrain_loss. z_world converges.\n\n"
        f"**Phase 1 (TerrainHead training):** {p1_episodes} episodes x 200 steps."
        f" E1+E2 frozen. TerrainHead trained on z_world.detach() with terrain_loss."
        f" Labels: w_harm=0.8 if haz>0.3 else 0.2; w_goal=0.8 if haz<0.33 else 0.2.\n\n"
        f"**Phase 2 (collection):** {p2_episodes} episodes x 200 steps. Random actions."
        f" Record terrain_weight and hazard_prox.\n\n"
        f"## Key Results\n\n"
        f"| Metric | Value | Threshold |\n"
        f"|---|---|---|\n"
        f"| r(w_harm, hazard_prox) | {avg_r_w_harm:.4f} | > 0.5 |\n"
        f"| r(w_goal, hazard_prox) | {avg_r_w_goal:.4f} | < -0.3 |\n"
        f"| final terrain_loss | {avg_final_t_loss:.4f} | < 0.05 |\n"
        f"| n_context_A (total) | {total_n_A} | -- |\n"
        f"| n_context_B (total) | {total_n_B} | -- |\n\n"
        f"## Pass Criteria\n\n"
        f"| Criterion | Result | Seeds passing |\n"
        f"|---|---|---|\n"
        f"| C1: r(w_harm, hazard_prox) > 0.5 | {'PASS' if crit1_pass_global else 'FAIL'}"
        f" | {seeds_c1_pass}/{n_seeds} |\n"
        f"| C2: r(w_goal, hazard_prox) < -0.3 | {'PASS' if crit2_pass_global else 'FAIL'}"
        f" | {seeds_c2_pass}/{n_seeds} |\n"
        f"| C3: terrain_loss < 0.05 | {'PASS' if crit3_pass_global else 'FAIL'}"
        f" | {seeds_c3_pass}/{n_seeds} |\n\n"
        f"PASS rule: all three pass in >= 2/3 seeds -> **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{c1_interp}\n\n"
        f"{c2_interp}\n\n"
        f"{c3_interp}\n\n"
        f"## Per-Seed Results\n\n"
        f"{per_seed_rows}\n"
    )

    metrics = {
        "r_w_harm":           float(avg_r_w_harm),
        "r_w_goal":           float(avg_r_w_goal),
        "final_terrain_loss": float(avg_final_t_loss),
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
        "p0_episodes":        int(p0_episodes),
        "p1_episodes":        int(p1_episodes),
        "p2_episodes":        int(p2_episodes),
    }

    print(f"\n[V3-EXQ-194a] Final results:", flush=True)
    print(f"  r_w_harm={avg_r_w_harm:.4f}  r_w_goal={avg_r_w_goal:.4f}", flush=True)
    print(f"  final_terrain_loss={avg_final_t_loss:.4f}", flush=True)
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
        description="EXQ-194a: Direct terrain pathway, phased training (MECH-152)"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11])
    parser.add_argument("--p0", type=int, default=200, help="Phase 0 warmup episodes")
    parser.add_argument("--p1", type=int, default=100, help="Phase 1 terrain training episodes")
    parser.add_argument("--p2", type=int, default=100, help="Phase 2 collection episodes")
    parser.add_argument("--num-hazards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 1 seed, 3/3/5 episodes. No output file.")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds = [42]
        args.p0    = 3
        args.p1    = 3
        args.p2    = 5
        print(
            "[DRY-RUN] 1 seed, P0=3 P1=3 P2=5,"
            f" num_hazards={args.num_hazards}",
            flush=True,
        )

    result = run(
        seeds=args.seeds,
        p0_episodes=args.p0,
        p1_episodes=args.p1,
        p2_episodes=args.p2,
        dry_run=args.dry_run,
        num_hazards=args.num_hazards,
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
    result["supersedes"]         = "v3_exq_194_direct_terrain_pathway"

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
