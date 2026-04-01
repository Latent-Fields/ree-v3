#!/opt/local/bin/python3
"""
EXQ-187: MECH-153 Validation -- Supervised Context-Labeling Training for E1 ContextMemory

Hypothesis (MECH-153): E1 ContextMemory requires a supervised context-labeling
  training objective to produce differentiated cue-indexed representations.
  Without this objective, hazard-proximate and hazard-distal context vectors
  remain near-identical (EXQ-181b: cosine_sim=0.9999, 0/3 seeds).

Design:
  Phase 0 (warmup): Train E1 with standard prediction loss PLUS terrain_loss
    (supervised context-labeling signal from SD-016 spec). sd016_enabled=True.
    terrain_loss = MSE(terrain_weight, targets) where targets are derived from
    hazard_field_view.max() -- proxy-field-aware thresholds.
  Phase 1 (collection): Collect extract_cue_context() outputs (cue_context,
    terrain_weight) in Context A (hazard-proximate) and Context B (hazard-distal).
  Phase 2 (metrics): Measure cosine_sim of cue_context and terrain_weight accuracy.

Pass criteria:
  C1 PASS: cosine_sim(mean_cue_context_A, mean_cue_context_B) < 0.85 in >= 2/3 seeds
  C2 PASS: terrain_weight w_harm accuracy > 0.70 (correct classification of
           Context A as high-harm, Context B as low-harm) in >= 2/3 seeds
  C3 PASS: ridge R^2 of cue_context -> harm_scalar > 0.3 in >= 2/3 seeds

  Context thresholds (proxy-field-aware, same as EXQ-181b):
    Context A: hazard_field_view.max() > 0.7
    Context B: hazard_field_view.max() < 0.33
  Context coverage: n_A >= 10 AND n_B >= 10 per seed (otherwise SKIP seed)
  PASS if all three pass in >= 2/3 seeds; MIXED if 1-2 pass; FAIL if none.

  If C1 PASS: supervised training produces differentiated ContextMemory --
    MECH-153 validated, ARC-042 Phase 1 gate cleared.
  If C1 FAIL: supervised training alone insufficient -- need stronger signal,
    more training, or architectural change to ContextMemory.

Terrain loss targets (proxy-field-aware adaptation of SD-016 spec):
  With use_proxy_fields=True, hazard_field range is [0.22, 1.0].
  SD-016 spec thresholds (0.3/0.1) are for raw field values.
  Proxy-field-aware thresholds:
    w_harm_target = 0.8 if hazard_max > 0.5 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.35 else 0.3

Claim IDs: MECH-153, ARC-042
Evidence for: ARC-042 Phase 1 (E1 substrate development gate)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_187_mech153_supervised_context_labeling"
CLAIM_IDS = ["MECH-153", "ARC-042"]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def get_hazard_max(obs_dict: Dict, world_obs: Optional[torch.Tensor]) -> float:
    """Extract max hazard field value from observation dict (proxy-field aware)."""
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


def get_harm_scalar(obs_dict: Dict) -> float:
    """Extract harm scalar from observation dict."""
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, 'shape') and harm_obs.shape[-1] > 50:
            return float(harm_obs[..., 50].mean().item())
    return 0.0


def ridge_r2(
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float = 1e-3,
    test_frac: float = 0.2,
) -> float:
    """Ridge regression R^2 on held-out test split."""
    N = X.shape[0]
    if N < 8:
        return 0.0
    n_test = max(1, int(N * test_frac))
    X_tr, y_tr = X[n_test:], y[n_test:]
    X_te, y_te = X[:n_test], y[:n_test]
    if X_tr.shape[0] < 4:
        return 0.0
    Xb = torch.cat([X_tr, torch.ones(X_tr.shape[0], 1, dtype=X_tr.dtype)], dim=1)
    XtX = Xb.T @ Xb + lam * torch.eye(Xb.shape[1], dtype=Xb.dtype)
    Xty = Xb.T @ y_tr
    try:
        w = torch.linalg.solve(XtX, Xty)
    except Exception:
        return 0.0
    Xb_te = torch.cat([X_te, torch.ones(X_te.shape[0], 1, dtype=X_te.dtype)], dim=1)
    y_pred = Xb_te @ w
    ss_res = ((y_te - y_pred) ** 2).sum()
    ss_tot = ((y_te - y_te.mean()) ** 2).sum()
    return float(1.0 - ss_res / (ss_tot + 1e-9))


def compute_terrain_targets(hazard_max: float, device) -> torch.Tensor:
    """
    Compute supervised terrain_weight targets from hazard_field_view.max().

    Proxy-field-aware adaptation of SD-016 spec:
      w_harm_target = 0.8 if hazard_max > 0.5 else 0.2
      w_goal_target = 0.8 if hazard_max < 0.35 else 0.3

    Returns: [1, 2] tensor of [w_harm_target, w_goal_target]
    """
    w_harm = 0.8 if hazard_max > 0.5 else 0.2
    w_goal = 0.8 if hazard_max < 0.35 else 0.3
    return torch.tensor([[w_harm, w_goal]], dtype=torch.float32, device=device)


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

    # KEY DIFFERENCE vs EXQ-181b: sd016_enabled=True
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

    terrain_losses_log = []

    # ---- Phase 0: Training with supervised terrain_loss ----
    agent.train()
    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_terrain_loss_sum = 0.0
        ep_steps = 0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            # Compute standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()

            # Compute terrain_loss (SD-016 supervised context-labeling signal)
            z_world = latent.z_world.detach()  # detach to isolate terrain gradient path
            # Actually: we need gradient to flow through extract_cue_context and
            # ContextMemory. Use the live z_world (not detached) for the cue query,
            # but the terrain targets are from the environment (no gradient).
            z_world_live = latent.z_world
            action_bias, terrain_weight = agent.e1.extract_cue_context(z_world_live)

            hazard_max = get_hazard_max(obs_dict, obs_world if obs_world.dim() >= 1 else None)
            terrain_targets = compute_terrain_targets(hazard_max, agent.device)

            terrain_loss = (
                F.mse_loss(terrain_weight[:, 0:1], terrain_targets[:, 0:1])
                + F.mse_loss(terrain_weight[:, 1:2], terrain_targets[:, 1:2])
            )

            total_loss = e1_loss + e2_loss + lambda_terrain * terrain_loss

            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            ep_terrain_loss_sum += terrain_loss.item()
            ep_steps += 1

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            if done:
                break

        avg_terrain = ep_terrain_loss_sum / max(1, ep_steps)
        terrain_losses_log.append(avg_terrain)

        if (ep + 1) % 20 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [train] seed={seed} ep {ep+1}/{warmup_episodes}"
                f" terrain_loss={avg_terrain:.6f}",
                flush=True,
            )

    # ---- Phase 1: Context collection ----
    agent.eval()

    cue_contexts_A: List[torch.Tensor] = []
    cue_contexts_B: List[torch.Tensor] = []
    terrain_weights_A: List[torch.Tensor] = []
    terrain_weights_B: List[torch.Tensor] = []
    harm_A: List[float] = []
    harm_B: List[float] = []
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

            # Extract cue context (z_world-only query)
            z_world_det = latent.z_world.detach()
            with torch.no_grad():
                action_bias, terrain_weight = agent.e1.extract_cue_context(z_world_det)

            # Also get the cue_context vector (before projection heads) for cosine_sim
            # We need to manually compute the intermediate cue_context:
            batch_size = z_world_det.shape[0]
            memory_dim = agent.e1.context_memory.memory_dim
            q = agent.e1.world_query_proj(z_world_det).unsqueeze(1)
            memory = agent.e1.context_memory.memory
            k = agent.e1.context_memory.key_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
            v = agent.e1.context_memory.value_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
            scores = torch.bmm(q, k.transpose(1, 2)) / (memory_dim ** 0.5)
            weights = F.softmax(scores, dim=-1)
            context = torch.bmm(weights, v).squeeze(1)
            cue_context = agent.e1.context_memory.output_proj(context)  # [batch, latent_dim=64]

            hazard_max = get_hazard_max(obs_dict, obs_world if obs_world.dim() >= 1 else None)
            harm_val   = get_harm_scalar(obs_dict)

            # Context classification (same thresholds as EXQ-181b)
            if hazard_max > 0.7:
                cue_contexts_A.append(cue_context.squeeze(0).cpu())
                terrain_weights_A.append(terrain_weight.squeeze(0).cpu())
                harm_A.append(harm_val)
            elif hazard_max < 0.33:
                cue_contexts_B.append(cue_context.squeeze(0).cpu())
                terrain_weights_B.append(terrain_weight.squeeze(0).cpu())
                harm_B.append(harm_val)
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
    w_harm_acc = 0.0
    r2_val = 0.0

    if n_A >= 10 and n_B >= 10:
        # C1: cosine_sim of cue_context vectors
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

        # C2: terrain_weight w_harm accuracy
        # Context A (hazard-proximate) should have w_harm > 0.5
        # Context B (hazard-distal) should have w_harm < 0.5
        tw_A = torch.stack(terrain_weights_A)  # [n_A, 2]
        tw_B = torch.stack(terrain_weights_B)  # [n_B, 2]
        correct_A = (tw_A[:, 0] > 0.5).float().sum().item()
        correct_B = (tw_B[:, 0] < 0.5).float().sum().item()
        w_harm_acc = (correct_A + correct_B) / (n_A + n_B)
        crit2_pass = w_harm_acc > 0.70
        print(
            f"  [C2] seed={seed}"
            f" w_harm_accuracy={w_harm_acc:.4f}"
            f" (A_correct={correct_A}/{n_A}, B_correct={correct_B}/{n_B})"
            f" threshold=0.70"
            f" pass={'YES' if crit2_pass else 'NO'}",
            flush=True,
        )
        # Also log mean terrain weights per context
        print(
            f"       mean_w_harm_A={tw_A[:, 0].mean():.4f}"
            f" mean_w_harm_B={tw_B[:, 0].mean():.4f}"
            f" mean_w_goal_A={tw_A[:, 1].mean():.4f}"
            f" mean_w_goal_B={tw_B[:, 1].mean():.4f}",
            flush=True,
        )
    else:
        print(
            f"  [C1/C2] seed={seed} SKIP -- insufficient context steps"
            f" (n_A={n_A}, n_B={n_B}, need >=10 each)",
            flush=True,
        )

    # C3: ridge R^2 of cue_context -> harm_scalar
    all_contexts = cue_contexts_A + cue_contexts_B
    all_harms    = harm_A + harm_B
    if len(all_contexts) >= 8:
        X = torch.stack(all_contexts).detach().float()
        y = torch.tensor(all_harms, dtype=torch.float32)
        r2_val = ridge_r2(X, y)
        crit3_pass = r2_val > 0.3
        print(
            f"  [C3] seed={seed}"
            f" harm_r2={r2_val:.4f}"
            f" threshold=0.3"
            f" pass={'YES' if crit3_pass else 'NO'}",
            flush=True,
        )
    else:
        print(
            f"  [C3] seed={seed} SKIP -- insufficient samples (n={len(all_contexts)})",
            flush=True,
        )

    # Terrain loss trajectory (first/last 5 episodes)
    tl_first5 = terrain_losses_log[:5] if len(terrain_losses_log) >= 5 else terrain_losses_log
    tl_last5  = terrain_losses_log[-5:] if len(terrain_losses_log) >= 5 else terrain_losses_log

    return {
        "seed": seed,
        "n_context_A": n_A,
        "n_context_B": n_B,
        "n_skipped": n_skipped,
        "cos_sim_AB": cos_sim_val if cos_sim_val == cos_sim_val else None,
        "w_harm_accuracy": float(w_harm_acc),
        "harm_r2": float(r2_val),
        "crit1_pass": int(crit1_pass),
        "crit2_pass": int(crit2_pass),
        "crit3_pass": int(crit3_pass),
        "terrain_loss_first5": [round(x, 6) for x in tl_first5],
        "terrain_loss_last5": [round(x, 6) for x in tl_last5],
    }


# ------------------------------------------------------------------ #
# Main run                                                             #
# ------------------------------------------------------------------ #

def run(
    seeds: Optional[List[int]] = None,
    warmup_episodes: int = 150,
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
            f"\n[V3-EXQ-187] seed={seed}"
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
    avg_cos_sim  = float(sum(cos_vals) / len(cos_vals)) if cos_vals else float("nan")
    avg_w_harm   = float(sum(r["w_harm_accuracy"] for r in per_seed_results) / max(1, n_seeds))
    avg_harm_r2  = float(sum(r["harm_r2"] for r in per_seed_results) / max(1, n_seeds))
    total_n_A    = sum(r["n_context_A"] for r in per_seed_results)
    total_n_B    = sum(r["n_context_B"] for r in per_seed_results)

    crit1_pass_global = seeds_c1_pass >= (n_seeds + 1) // 2
    crit2_pass_global = seeds_c2_pass >= (n_seeds + 1) // 2
    crit3_pass_global = seeds_c3_pass >= (n_seeds + 1) // 2

    n_criteria_pass = sum([crit1_pass_global, crit2_pass_global, crit3_pass_global])
    if n_criteria_pass == 3:
        status = "PASS"
        evidence_direction = "supports"
    elif n_criteria_pass >= 1:
        status = "MIXED"
        evidence_direction = "inconclusive"
    else:
        status = "FAIL"
        evidence_direction = "does_not_support"

    # Interpretation strings
    if crit1_pass_global:
        c1_interp = (
            f"C1 PASS: cosine_sim={avg_cos_sim:.4f} < 0.85 in {seeds_c1_pass}/{n_seeds} seeds."
            " Supervised training produces differentiated ContextMemory cue representations."
            " Compare EXQ-181b baseline: cosine_sim=0.9999 without supervised training."
        )
    else:
        c1_interp = (
            f"C1 FAIL: cosine_sim={avg_cos_sim:.4f} >= 0.85 (passed {seeds_c1_pass}/{n_seeds} seeds)."
            " Supervised training did NOT produce sufficient differentiation."
            " ContextMemory may need architectural changes or stronger training signal."
        )

    if crit2_pass_global:
        c2_interp = (
            f"C2 PASS: w_harm_accuracy={avg_w_harm:.4f} > 0.70 in {seeds_c2_pass}/{n_seeds} seeds."
            " terrain_weight correctly classifies hazard-proximate vs hazard-distal contexts."
        )
    else:
        c2_interp = (
            f"C2 FAIL: w_harm_accuracy={avg_w_harm:.4f} <= 0.70 (passed {seeds_c2_pass}/{n_seeds} seeds)."
            " terrain_weight does NOT reliably distinguish context types."
        )

    if crit3_pass_global:
        c3_interp = (
            f"C3 PASS: harm_r2={avg_harm_r2:.4f} > 0.3 in {seeds_c3_pass}/{n_seeds} seeds."
            " Cue context encodes harm information linearly."
        )
    else:
        c3_interp = (
            f"C3 FAIL: harm_r2={avg_harm_r2:.4f} <= 0.3 (passed {seeds_c3_pass}/{n_seeds} seeds)."
            " Cue context does NOT reliably encode harm information."
        )

    per_seed_rows = "\n".join(
        f"  seed={r['seed']}:"
        f" n_A={r['n_context_A']}"
        f" n_B={r['n_context_B']}"
        f" cos_sim={r['cos_sim_AB'] if r['cos_sim_AB'] is not None else 'N/A'}"
        f" w_harm_acc={r['w_harm_accuracy']:.4f}"
        f" harm_r2={r['harm_r2']:.4f}"
        f" C1={'PASS' if r['crit1_pass'] else 'FAIL'}"
        f" C2={'PASS' if r['crit2_pass'] else 'FAIL'}"
        f" C3={'PASS' if r['crit3_pass'] else 'FAIL'}"
        for r in per_seed_results
    )

    cos_display = f"{avg_cos_sim:.4f}" if avg_cos_sim == avg_cos_sim else "N/A"

    summary_markdown = (
        f"# V3-EXQ-187 -- MECH-153 Supervised Context-Labeling Validation\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-153, ARC-042\n"
        f"**Seeds:** {seeds}\n"
        f"**Baseline:** EXQ-181b (cosine_sim=0.9999 without supervised training)\n\n"
        f"## Context\n\n"
        f"EXQ-181b demonstrated that E1 ContextMemory trained only via the world-model\n"
        f"prediction objective does NOT produce differentiated cue representations\n"
        f"(cosine_sim=0.9999 between hazard-proximate and hazard-distal contexts, 0/3 seeds).\n"
        f"MECH-153 claims a supervised context-labeling objective is required. This experiment\n"
        f"adds terrain_loss (supervised hazard-field label, lambda={lambda_terrain}) to E1\n"
        f"training and re-runs the EXQ-181b diagnostic on extract_cue_context() outputs.\n\n"
        f"## Design\n\n"
        f"**Phase 0 (warmup):** {warmup_episodes} episodes x 200 steps. REEAgent training\n"
        f"with sd016_enabled=True: E1 prediction loss + E2 loss + lambda_terrain * terrain_loss.\n"
        f"terrain_loss = MSE(terrain_weight, targets) where targets derived from\n"
        f"hazard_field_view.max() (proxy-field-aware thresholds: >0.5 -> w_harm=0.8, else 0.2;\n"
        f"<0.35 -> w_goal=0.8, else 0.3).\n\n"
        f"**Phase 1 (collection):** {collect_episodes} episodes x 200 steps. Random actions.\n"
        f"Collect extract_cue_context() outputs in Context A (max > 0.7) and Context B (< 0.33).\n\n"
        f"**Phase 2 (metrics):** C1 = cosine_sim of cue_context < 0.85. C2 = w_harm accuracy > 0.70.\n"
        f"C3 = ridge R^2 of cue_context -> harm > 0.3.\n\n"
        f"## Key Results\n\n"
        f"| Metric | Value | Threshold | EXQ-181b baseline |\n"
        f"|---|---|---|---|\n"
        f"| cosine_sim(cue_A, cue_B) | {cos_display} | < 0.85 | 0.9999 |\n"
        f"| w_harm accuracy | {avg_w_harm:.4f} | > 0.70 | N/A |\n"
        f"| harm_r2 (cue -> harm) | {avg_harm_r2:.4f} | > 0.3 | 0.23 |\n"
        f"| n_context_A steps (total) | {total_n_A} | -- | -- |\n"
        f"| n_context_B steps (total) | {total_n_B} | -- | -- |\n\n"
        f"## Pass Criteria\n\n"
        f"| Criterion | Result | Seeds passing |\n"
        f"|---|---|---|\n"
        f"| C1: cosine_sim < 0.85 | {'PASS' if crit1_pass_global else 'FAIL'}"
        f" | {seeds_c1_pass}/{n_seeds} |\n"
        f"| C2: w_harm accuracy > 0.70 | {'PASS' if crit2_pass_global else 'FAIL'}"
        f" | {seeds_c2_pass}/{n_seeds} |\n"
        f"| C3: harm_r2 > 0.3 | {'PASS' if crit3_pass_global else 'FAIL'}"
        f" | {seeds_c3_pass}/{n_seeds} |\n\n"
        f"PASS rule: all three criteria pass in >= 2/3 seeds -> PASS."
        f" 1-2 criteria -> MIXED. 0 -> FAIL. **{status}**\n\n"
        f"## Interpretation\n\n"
        f"{c1_interp}\n\n"
        f"{c2_interp}\n\n"
        f"{c3_interp}\n\n"
        f"## Per-Seed Results\n\n"
        f"{per_seed_rows}\n"
    )

    metrics = {
        "cosine_sim_AB":      float(avg_cos_sim) if avg_cos_sim == avg_cos_sim else -1.0,
        "w_harm_accuracy":    float(avg_w_harm),
        "harm_r2":            float(avg_harm_r2),
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
        "warmup_episodes":    int(warmup_episodes),
    }

    print(f"\n[V3-EXQ-187] Final results:", flush=True)
    print(f"  cosine_sim_AB={cos_display}  w_harm_acc={avg_w_harm:.4f}  harm_r2={avg_harm_r2:.4f}", flush=True)
    print(f"  n_A={total_n_A}  n_B={total_n_B}", flush=True)
    print(f"  C1: {seeds_c1_pass}/{n_seeds}  C2: {seeds_c2_pass}/{n_seeds}  C3: {seeds_c3_pass}/{n_seeds}", flush=True)
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
        description="EXQ-187: MECH-153 supervised context-labeling validation"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11])
    parser.add_argument("--warmup", type=int, default=150)
    parser.add_argument("--collect", type=int, default=100)
    parser.add_argument("--num-hazards", type=int, default=1)
    parser.add_argument("--lambda-terrain", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true",
                        help="Smoke test: 1 seed, 3 warmup, 5 collect.")
    args = parser.parse_args()

    if args.dry_run:
        args.seeds   = [42]
        args.warmup  = 3
        args.collect = 5
        print("[DRY-RUN] 1 seed, 3 warmup, 5 collect, sd016_enabled=True", flush=True)

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
