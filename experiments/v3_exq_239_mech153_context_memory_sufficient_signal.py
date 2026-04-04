#!/opt/local/bin/python3
"""
V3-EXQ-239: MECH-153 Context Memory Sufficient Signal Test

Redesign of EXQ-187a: stronger supervised signal (lambda_terrain=0.5 vs 0.1,
500 warmup episodes vs 150). EXQ-187a was diagnosed as inconclusive (not a code
bug): context_memory.write() was correctly called, but 150 warmup episodes with
lambda=0.1 was insufficient to produce differentiated context vectors
(cosine_sim=1.0).

MECH-153: "Supervised terrain-context supervision is required for ContextMemory
to develop discriminating context representations. Without terrain-type labels
flowing gradients through extract_cue_context(), cosine_sim of retrieved context
vectors approaches 1.0."

ARC-042: "Staged developmental phases for ContextMemory: Phase 0 (world model
without context differentiation), Phase 1 (supervised ContextMemory training),
Phase 2 (context-gated evaluation)."

Experiment design:
  Two conditions, both trained from scratch with 5 seeds each.

  SUPERVISED (MECH-153 test):
    lambda_terrain=0.5 (5x stronger than EXQ-187a), 500 warmup episodes (3x).
    context_memory.write() called each step (Phase 0 + P1).
    Supervised CE loss on terrain type (3 classes: OPEN=0, ROCKY=1, FOREST=2,
    mapped from hazard_max thresholds) flowing gradients through
    extract_cue_context() and ContextMemory.

  ABLATED:
    lambda_terrain=0.0 (no supervised loss). context_memory.write() still called.
    Same 500 warmup episodes. Tests whether unsupervised context representations
    differentiate without supervision.

Pass criteria:
  C1: SUPERVISED cosine_sim < 0.95 in 3/5 seeds (representations differentiated)
  C2: SUPERVISED cosine_sim significantly lower than ABLATED (mean diff > 0.02)
  C3: SUPERVISED terrain_accuracy (retrieved context -> terrain classifier) > 0.55
      in 3/5 seeds
  C4: ABLATED cosine_sim >= 0.98 in 3/5 seeds (confirming supervision is
      necessary, not just helpful)

PASS: C1 AND (C2 OR C3) AND C4
FAIL otherwise

Terrain class mapping (from hazard_max):
  OPEN   (class 0): hazard_max < 0.33  (low hazard)
  ROCKY  (class 1): 0.33 <= hazard_max <= 0.70  (moderate hazard)
  FOREST (class 2): hazard_max > 0.70  (high hazard)

Note: these terrain classes are a simplification of the CausalGridWorldV2
    hazard field structure. The env does not have explicit terrain type tags
    in obs_dict; terrain is inferred from hazard_field_view.max() as proxy.

claim_ids: ["MECH-153"]
experiment_purpose: "evidence"
supersedes: v3_exq_187a_mech153_supervised_context_labeling
"""

import sys
import random
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_239_mech153_context_memory_sufficient_signal"
CLAIM_IDS = ["MECH-153"]
EXPERIMENT_PURPOSE = "evidence"

# Terrain class labels (mapped from hazard_max)
TERRAIN_OPEN   = 0  # hazard_max < 0.33
TERRAIN_ROCKY  = 1  # 0.33 <= hazard_max <= 0.70
TERRAIN_FOREST = 2  # hazard_max > 0.70
NUM_TERRAIN_CLASSES = 3


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def get_hazard_max(obs_dict: Dict, obs_world: Optional[torch.Tensor]) -> float:
    """Extract max hazard field value from observation dict (proxy-field aware)."""
    if "harm_obs" in obs_dict:
        harm_obs = obs_dict["harm_obs"]
        if hasattr(harm_obs, 'shape') and harm_obs.shape[-1] >= 26:
            return float(harm_obs[..., :25].max().item())
    if "hazard_field_view" in obs_dict:
        hfv = obs_dict["hazard_field_view"]
        if hasattr(hfv, 'shape'):
            return float(hfv.max().item())
    if obs_world is not None and obs_world.shape[-1] >= 225:
        return float(obs_world[..., 200:225].max().item())
    return 0.0


def hazard_max_to_terrain_class(hazard_max: float) -> int:
    """Map hazard_max to terrain class label (0=OPEN, 1=ROCKY, 2=FOREST)."""
    if hazard_max > 0.70:
        return TERRAIN_FOREST
    elif hazard_max >= 0.33:
        return TERRAIN_ROCKY
    else:
        return TERRAIN_OPEN


def extract_cue_context_vector(agent: "REEAgent", z_world: torch.Tensor) -> torch.Tensor:
    """
    Extract the cue_context vector from ContextMemory using world_query_proj.
    Returns [batch, latent_dim=64] cue context before the two projection heads.
    Requires sd016_enabled=True (world_query_proj must exist).
    """
    batch_size = z_world.shape[0]
    memory_dim = agent.e1.context_memory.memory_dim

    q = agent.e1.world_query_proj(z_world).unsqueeze(1)  # [batch, 1, memory_dim]
    memory = agent.e1.context_memory.memory
    k = agent.e1.context_memory.key_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
    v = agent.e1.context_memory.value_proj(memory).unsqueeze(0).expand(batch_size, -1, -1)
    scores = torch.bmm(q, k.transpose(1, 2)) / (memory_dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    context = torch.bmm(weights, v).squeeze(1)  # [batch, memory_dim]
    cue_context = agent.e1.context_memory.output_proj(context)  # [batch, latent_dim]
    return cue_context


def cosine_sim_of_means(
    vecs_a: List[torch.Tensor],
    vecs_b: List[torch.Tensor],
) -> float:
    """Cosine similarity between mean of vecs_a and mean of vecs_b."""
    if not vecs_a or not vecs_b:
        return float("nan")
    mean_a = torch.stack(vecs_a).mean(dim=0)
    mean_b = torch.stack(vecs_b).mean(dim=0)
    return float(F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item())


# ------------------------------------------------------------------ #
# Terrain classifier (simple linear head for accuracy measurement)    #
# ------------------------------------------------------------------ #

class TerrainClassifierHead(nn.Module):
    """Simple 3-class linear probe for terrain classification from cue_context."""
    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, NUM_TERRAIN_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ------------------------------------------------------------------ #
# Single-seed runner for one condition                                 #
# ------------------------------------------------------------------ #

def _run_condition(
    seed: int,
    condition: str,  # "supervised" or "ablated"
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
    dry_run: bool = False,
) -> Dict:
    """Run one condition (supervised or ablated) for a single seed."""
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

    # sd016_enabled=True provides world_query_proj and extract_cue_context()
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

    # Terrain classifier head (trained jointly during warmup for terrain_accuracy)
    latent_dim = self_dim + world_dim
    terrain_clf = TerrainClassifierHead(latent_dim)
    clf_optimizer = optim.Adam(terrain_clf.parameters(), lr=lr)

    terrain_losses_log = []
    clf_losses_log = []

    # ---- Phase 0 (P0): Warmup training ----
    agent.train()
    terrain_clf.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        ep_terrain_loss_sum = 0.0
        ep_clf_loss_sum = 0.0
        ep_steps = 0

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            agent.clock.advance()

            # Populate ContextMemory with diverse environment states.
            obs_state = torch.cat([latent.z_self.detach(), latent.z_world.detach()], dim=-1)
            agent.e1.context_memory.write(obs_state)

            # Standard E1 + E2 losses
            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()

            terrain_loss = torch.tensor(0.0)
            clf_loss = torch.tensor(0.0)

            if lambda_terrain > 0.0:
                # SUPERVISED: CE loss on terrain class from cue_context
                z_world_live = latent.z_world
                cue_context = extract_cue_context_vector(agent, z_world_live)

                hazard_max = get_hazard_max(obs_dict, obs_world if obs_world.dim() >= 1 else None)
                terrain_class = hazard_max_to_terrain_class(hazard_max)
                terrain_label = torch.tensor([terrain_class], dtype=torch.long)

                # CE loss flowing gradients through extract_cue_context() and ContextMemory
                logits = terrain_clf(cue_context)  # [1, 3]
                clf_loss = F.cross_entropy(logits, terrain_label)

                # Also add terrain CE loss to the agent's loss (flows through ContextMemory)
                terrain_loss = clf_loss

            total_loss = e1_loss + e2_loss + lambda_terrain * terrain_loss

            if total_loss.requires_grad:
                optimizer.zero_grad()
                clf_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(terrain_clf.parameters(), 1.0)
                optimizer.step()
                clf_optimizer.step()

            ep_terrain_loss_sum += terrain_loss.item() if hasattr(terrain_loss, 'item') else float(terrain_loss)
            ep_clf_loss_sum += clf_loss.item() if hasattr(clf_loss, 'item') else float(clf_loss)
            ep_steps += 1

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh

            _, harm_signal, done, info, obs_dict = env.step(action_oh)

            if done:
                break

        avg_terrain = ep_terrain_loss_sum / max(1, ep_steps)
        avg_clf = ep_clf_loss_sum / max(1, ep_steps)
        terrain_losses_log.append(avg_terrain)
        clf_losses_log.append(avg_clf)

        if (ep + 1) % 50 == 0 or ep == warmup_episodes - 1:
            print(
                f"  [{condition}] seed={seed} ep {ep+1}/{warmup_episodes}"
                f" terrain_loss={avg_terrain:.6f}"
                f" clf_loss={avg_clf:.6f}",
                flush=True,
            )

    # ---- Phase 1 (P1): Context collection ----
    agent.eval()
    terrain_clf.eval()

    cue_contexts_proximate: List[torch.Tensor] = []  # Context A: FOREST (high hazard)
    cue_contexts_distal: List[torch.Tensor] = []     # Context B: OPEN (low hazard)
    terrain_labels_all: List[int] = []
    cue_contexts_all: List[torch.Tensor] = []
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
            with torch.no_grad():
                cue_context = extract_cue_context_vector(agent, z_world_det)

            hazard_max = get_hazard_max(obs_dict, obs_world if obs_world.dim() >= 1 else None)
            terrain_class = hazard_max_to_terrain_class(hazard_max)

            cue_contexts_all.append(cue_context.squeeze(0).cpu())
            terrain_labels_all.append(terrain_class)

            if hazard_max > 0.70:
                cue_contexts_proximate.append(cue_context.squeeze(0).cpu())
            elif hazard_max < 0.33:
                cue_contexts_distal.append(cue_context.squeeze(0).cpu())
            else:
                n_skipped += 1

            action_idx = random.randint(0, action_dim - 1)
            action_oh  = _action_to_onehot(action_idx, action_dim, agent.device)
            agent._last_action = action_oh
            _, _, done, _, obs_dict = env.step(action_oh)

            if done:
                break

    n_proximate = len(cue_contexts_proximate)
    n_distal = len(cue_contexts_distal)
    n_total = len(cue_contexts_all)

    print(
        f"  [{condition}] seed={seed} collect done:"
        f" n_proximate={n_proximate}"
        f" n_distal={n_distal}"
        f" n_skipped={n_skipped}"
        f" n_total={n_total}",
        flush=True,
    )

    # ---- Phase 2 (P2): Metrics ----
    cos_sim = float("nan")
    terrain_accuracy = 0.0
    has_sufficient_samples = (n_proximate >= 10 and n_distal >= 10)

    if has_sufficient_samples:
        cos_sim = cosine_sim_of_means(cue_contexts_proximate, cue_contexts_distal)
        print(
            f"  [{condition}] seed={seed} cosine_sim={cos_sim:.4f}",
            flush=True,
        )
    else:
        print(
            f"  [{condition}] seed={seed} SKIP cosine_sim -- insufficient"
            f" (n_proximate={n_proximate}, n_distal={n_distal}, need >=10 each)",
            flush=True,
        )

    # Terrain accuracy: train a fresh linear probe on the collected representations
    # Use 80% for training, 20% for test (if sufficient samples)
    if n_total >= 20 and len(set(terrain_labels_all)) > 1:
        X = torch.stack(cue_contexts_all).detach().float()       # [N, latent_dim]
        y = torch.tensor(terrain_labels_all, dtype=torch.long)   # [N]

        n_test = max(1, int(n_total * 0.2))
        n_train = n_total - n_test
        X_tr, y_tr = X[n_test:], y[n_test:]
        X_te, y_te = X[:n_test], y[:n_test]

        # Quick linear probe: 20 gradient steps
        probe = nn.Linear(X.shape[1], NUM_TERRAIN_CLASSES)
        probe_opt = optim.Adam(probe.parameters(), lr=1e-2)
        for _ in range(20):
            probe_opt.zero_grad()
            loss = F.cross_entropy(probe(X_tr), y_tr)
            loss.backward()
            probe_opt.step()

        with torch.no_grad():
            preds = probe(X_te).argmax(dim=1)
            terrain_accuracy = float((preds == y_te).float().mean().item())

        print(
            f"  [{condition}] seed={seed} terrain_accuracy={terrain_accuracy:.4f}"
            f" (probe on {n_train} train / {n_test} test samples)",
            flush=True,
        )
    else:
        print(
            f"  [{condition}] seed={seed} SKIP terrain_accuracy"
            f" -- insufficient samples (n_total={n_total})",
            flush=True,
        )

    # Terrain loss trajectory (first/last 5 episodes)
    tl_first5 = terrain_losses_log[:5] if len(terrain_losses_log) >= 5 else terrain_losses_log
    tl_last5  = terrain_losses_log[-5:] if len(terrain_losses_log) >= 5 else terrain_losses_log

    return {
        "seed": seed,
        "condition": condition,
        "cosine_sim": cos_sim if (cos_sim == cos_sim) else None,  # nan -> None
        "terrain_accuracy": float(terrain_accuracy),
        "n_proximate": n_proximate,
        "n_distal": n_distal,
        "n_total": n_total,
        "n_skipped": n_skipped,
        "has_sufficient_samples": int(has_sufficient_samples),
        "terrain_loss_first5": [round(x, 6) for x in tl_first5],
        "terrain_loss_last5": [round(x, 6) for x in tl_last5],
        "lambda_terrain": float(lambda_terrain),
        "warmup_episodes": int(warmup_episodes),
    }


# ------------------------------------------------------------------ #
# Main run                                                             #
# ------------------------------------------------------------------ #

def run(
    seeds: Optional[List[int]] = None,
    warmup_episodes: int = 500,
    collect_episodes: int = 100,
    dry_run: bool = False,
    num_hazards: int = 1,
    lambda_supervised: float = 0.5,
) -> Dict:
    if seeds is None:
        seeds = [42, 7, 11, 99, 31]

    steps_per_episode = 200
    self_dim     = 32
    world_dim    = 32
    lr           = 1e-3
    alpha_world  = 0.9
    alpha_self   = 0.3

    print(
        f"\n[V3-EXQ-239] MECH-153 Context Memory Sufficient Signal Test",
        flush=True,
    )
    print(
        f"  seeds={seeds}"
        f" warmup_episodes={warmup_episodes}"
        f" lambda_supervised={lambda_supervised}"
        f" num_hazards={num_hazards}",
        flush=True,
    )

    supervised_results: List[Dict] = []
    ablated_results: List[Dict] = []

    for seed in seeds:
        print(f"\n--- SUPERVISED condition, seed={seed} ---", flush=True)
        r_sup = _run_condition(
            seed=seed,
            condition="supervised",
            warmup_episodes=warmup_episodes,
            collect_episodes=collect_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            lr=lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            num_hazards=num_hazards,
            lambda_terrain=lambda_supervised,
            dry_run=dry_run,
        )
        supervised_results.append(r_sup)

        print(f"\n--- ABLATED condition, seed={seed} ---", flush=True)
        r_abl = _run_condition(
            seed=seed,
            condition="ablated",
            warmup_episodes=warmup_episodes,
            collect_episodes=collect_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            lr=lr,
            alpha_world=alpha_world,
            alpha_self=alpha_self,
            num_hazards=num_hazards,
            lambda_terrain=0.0,
            dry_run=dry_run,
        )
        ablated_results.append(r_abl)

    n_seeds = len(seeds)

    # ---- Aggregate metrics ----
    sup_cosine_vals = [
        r["cosine_sim"] for r in supervised_results
        if r["cosine_sim"] is not None
    ]
    abl_cosine_vals = [
        r["cosine_sim"] for r in ablated_results
        if r["cosine_sim"] is not None
    ]
    sup_accuracy_vals = [r["terrain_accuracy"] for r in supervised_results]
    abl_accuracy_vals = [r["terrain_accuracy"] for r in ablated_results]

    mean_sup_cosine = float(sum(sup_cosine_vals) / len(sup_cosine_vals)) if sup_cosine_vals else float("nan")
    mean_abl_cosine = float(sum(abl_cosine_vals) / len(abl_cosine_vals)) if abl_cosine_vals else float("nan")
    mean_sup_accuracy = float(sum(sup_accuracy_vals) / max(1, n_seeds))
    mean_abl_accuracy = float(sum(abl_accuracy_vals) / max(1, n_seeds))

    mean_cosine_diff = (
        float(mean_abl_cosine - mean_sup_cosine)
        if (mean_sup_cosine == mean_sup_cosine and mean_abl_cosine == mean_abl_cosine)
        else float("nan")
    )

    # ---- Per-seed criteria ----
    # C1: SUPERVISED cosine_sim < 0.95 per seed
    c1_seeds_pass = sum(
        1 for r in supervised_results
        if r["cosine_sim"] is not None and r["cosine_sim"] < 0.95
    )

    # C2: mean_cosine_diff > 0.02 (aggregate, not per seed)
    c2_pass = (mean_cosine_diff > 0.02) if (mean_cosine_diff == mean_cosine_diff) else False

    # C3: SUPERVISED terrain_accuracy > 0.55 per seed (3/5 seeds)
    c3_seeds_pass = sum(
        1 for r in supervised_results if r["terrain_accuracy"] > 0.55
    )

    # C4: ABLATED cosine_sim >= 0.98 per seed (3/5 seeds)
    c4_seeds_pass = sum(
        1 for r in ablated_results
        if r["cosine_sim"] is not None and r["cosine_sim"] >= 0.98
    )

    c1_pass = c1_seeds_pass >= 3  # 3/5 seeds
    c3_pass = c3_seeds_pass >= 3  # 3/5 seeds
    c4_pass = c4_seeds_pass >= 3  # 3/5 seeds

    # PASS: C1 AND (C2 OR C3) AND C4
    pass_condition = c1_pass and (c2_pass or c3_pass) and c4_pass

    if pass_condition:
        status = "PASS"
        evidence_direction = "supports"
    else:
        status = "FAIL"
        evidence_direction = "does_not_support"

    # ---- Print summary ----
    print(f"\n[V3-EXQ-239] Aggregate results:", flush=True)
    print(
        f"  SUPERVISED: mean_cosine_sim={mean_sup_cosine:.4f}"
        f"  ABLATED: mean_cosine_sim={mean_abl_cosine:.4f}"
        f"  diff={mean_cosine_diff:.4f}",
        flush=True,
    )
    print(
        f"  SUPERVISED terrain_accuracy={mean_sup_accuracy:.4f}"
        f"  ABLATED terrain_accuracy={mean_abl_accuracy:.4f}",
        flush=True,
    )
    print(
        f"  C1 (sup cosine_sim<0.95): {c1_seeds_pass}/{n_seeds} seeds -> {'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C2 (mean_cosine_diff>0.02): diff={mean_cosine_diff:.4f} -> {'PASS' if c2_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C3 (sup terrain_acc>0.55): {c3_seeds_pass}/{n_seeds} seeds -> {'PASS' if c3_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  C4 (abl cosine_sim>=0.98): {c4_seeds_pass}/{n_seeds} seeds -> {'PASS' if c4_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"  PASS rule: C1 AND (C2 OR C3) AND C4 -> {status}",
        flush=True,
    )

    # ---- Per-seed table ----
    per_seed_rows_sup = "\n".join(
        f"  SUP seed={r['seed']}:"
        f" cosine_sim={r['cosine_sim'] if r['cosine_sim'] is not None else 'N/A'}"
        f" terrain_acc={r['terrain_accuracy']:.4f}"
        f" C1={'PASS' if r['cosine_sim'] is not None and r['cosine_sim'] < 0.95 else 'FAIL'}"
        f" C3={'PASS' if r['terrain_accuracy'] > 0.55 else 'FAIL'}"
        for r in supervised_results
    )
    per_seed_rows_abl = "\n".join(
        f"  ABL seed={r['seed']}:"
        f" cosine_sim={r['cosine_sim'] if r['cosine_sim'] is not None else 'N/A'}"
        f" terrain_acc={r['terrain_accuracy']:.4f}"
        f" C4={'PASS' if r['cosine_sim'] is not None and r['cosine_sim'] >= 0.98 else 'FAIL'}"
        for r in ablated_results
    )

    sup_cos_display = f"{mean_sup_cosine:.4f}" if mean_sup_cosine == mean_sup_cosine else "N/A"
    abl_cos_display = f"{mean_abl_cosine:.4f}" if mean_abl_cosine == mean_abl_cosine else "N/A"
    diff_display = f"{mean_cosine_diff:.4f}" if mean_cosine_diff == mean_cosine_diff else "N/A"

    summary_markdown = (
        f"# V3-EXQ-239 -- MECH-153 Context Memory Sufficient Signal Test\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-153\n"
        f"**Seeds:** {seeds}\n"
        f"**Supersedes:** EXQ-187a (inconclusive: lambda=0.1, 150 warmup eps insufficient)\n\n"
        f"## Context\n\n"
        f"EXQ-187a was diagnosed as inconclusive (not a code bug): context_memory.write()\n"
        f"was correctly called, but lambda=0.1 with 150 warmup episodes was insufficient\n"
        f"to produce differentiated context vectors (cosine_sim=1.0). This experiment\n"
        f"uses 5x stronger supervised signal (lambda=0.5) and 3x longer warmup (500 eps).\n\n"
        f"## Design\n\n"
        f"**SUPERVISED:** lambda_terrain={lambda_supervised}, {warmup_episodes} warmup eps x 200 steps.\n"
        f"CE terrain-class loss (3 classes: OPEN/ROCKY/FOREST) flowing through extract_cue_context()\n"
        f"and ContextMemory. context_memory.write() called each step.\n\n"
        f"**ABLATED:** lambda_terrain=0.0 (no supervised loss). Same warmup setup.\n"
        f"Tests whether unsupervised context representations differentiate without supervision.\n\n"
        f"## Key Results\n\n"
        f"| Condition | mean cosine_sim | mean terrain_acc |\n"
        f"|---|---|---|\n"
        f"| SUPERVISED | {sup_cos_display} | {mean_sup_accuracy:.4f} |\n"
        f"| ABLATED | {abl_cos_display} | {mean_abl_accuracy:.4f} |\n"
        f"| Difference (ABLATED - SUP) | {diff_display} | -- |\n\n"
        f"## Pass Criteria\n\n"
        f"| Criterion | Result | Notes |\n"
        f"|---|---|---|\n"
        f"| C1: SUP cosine_sim<0.95 (3/5 seeds) | {'PASS' if c1_pass else 'FAIL'} | {c1_seeds_pass}/{n_seeds} seeds |\n"
        f"| C2: mean_cosine_diff>0.02 | {'PASS' if c2_pass else 'FAIL'} | diff={diff_display} |\n"
        f"| C3: SUP terrain_acc>0.55 (3/5 seeds) | {'PASS' if c3_pass else 'FAIL'} | {c3_seeds_pass}/{n_seeds} seeds |\n"
        f"| C4: ABL cosine_sim>=0.98 (3/5 seeds) | {'PASS' if c4_pass else 'FAIL'} | {c4_seeds_pass}/{n_seeds} seeds |\n\n"
        f"PASS rule: C1 AND (C2 OR C3) AND C4. **{status}**\n\n"
        f"## Per-Seed Results\n\n"
        f"{per_seed_rows_sup}\n\n"
        f"{per_seed_rows_abl}\n"
    )

    metrics = {
        "cosine_sim_supervised_mean":  float(mean_sup_cosine) if mean_sup_cosine == mean_sup_cosine else -1.0,
        "cosine_sim_ablated_mean":     float(mean_abl_cosine) if mean_abl_cosine == mean_abl_cosine else -1.0,
        "mean_cosine_diff":            float(mean_cosine_diff) if mean_cosine_diff == mean_cosine_diff else -1.0,
        "terrain_accuracy_supervised_mean": float(mean_sup_accuracy),
        "terrain_accuracy_ablated_mean":    float(mean_abl_accuracy),
        "c1_pass":                     int(c1_pass),
        "c2_pass":                     int(c2_pass),
        "c3_pass":                     int(c3_pass),
        "c4_pass":                     int(c4_pass),
        "c1_seeds_pass":               int(c1_seeds_pass),
        "c3_seeds_pass":               int(c3_seeds_pass),
        "c4_seeds_pass":               int(c4_seeds_pass),
        "n_seeds":                     int(n_seeds),
        "lambda_supervised":           float(lambda_supervised),
        "warmup_episodes":             int(warmup_episodes),
        "cosine_sim_supervised_per_seed": [
            float(r["cosine_sim"]) if r["cosine_sim"] is not None else -1.0
            for r in supervised_results
        ],
        "cosine_sim_ablated_per_seed": [
            float(r["cosine_sim"]) if r["cosine_sim"] is not None else -1.0
            for r in ablated_results
        ],
        "terrain_accuracy_supervised_per_seed": [
            float(r["terrain_accuracy"]) for r in supervised_results
        ],
        "terrain_accuracy_ablated_per_seed": [
            float(r["terrain_accuracy"]) for r in ablated_results
        ],
    }

    return {
        "status":              status,
        "metrics":             metrics,
        "summary_markdown":    summary_markdown,
        "claim_ids":           CLAIM_IDS,
        "evidence_direction":  evidence_direction,
        "experiment_type":     EXPERIMENT_TYPE,
        "experiment_purpose":  EXPERIMENT_PURPOSE,
        "fatal_error_count":   0,
        "per_seed_supervised": supervised_results,
        "per_seed_ablated":    ablated_results,
    }


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V3-EXQ-239: MECH-153 context memory sufficient signal"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 11, 99, 31])
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--collect", type=int, default=100)
    parser.add_argument("--num-hazards", type=int, default=1)
    parser.add_argument("--lambda-supervised", type=float, default=0.5)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Smoke test: 1 seed, 2 warmup eps, 5 collect steps."
    )
    args = parser.parse_args()

    if args.dry_run:
        args.seeds   = [42]
        args.warmup  = 2
        args.collect = 5
        print("[DRY-RUN] 1 seed, 2 warmup eps, 5 collect eps -- smoke test only.", flush=True)

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        collect_episodes=args.collect,
        dry_run=args.dry_run,
        num_hazards=args.num_hazards,
        lambda_supervised=args.lambda_supervised,
    )

    if args.dry_run:
        print("\n[DRY-RUN] DRY-RUN complete. Not writing output file.", flush=True)
        print(f"Status: {result['status']}", flush=True)
        for k, v in result["metrics"].items():
            if not isinstance(v, list):
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
        if not isinstance(v, list):
            print(f"  {k}: {v}", flush=True)
