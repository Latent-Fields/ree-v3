#!/opt/local/bin/python3
"""
V3-EXQ-418a -- SD-017 Context-Conditioned Action: Sleep + SD-016 Integration (Fix)

EXPERIMENT_PURPOSE: evidence

CHANGES vs EXQ-418 (two bugs fixed):
  1. SHY disabled in both conditions (shy_enabled=False).
     EXQ-418 used shy_decay_rate=0.85 with 15 sleep cycles:
     0.85^15 ~= 0.09 -- slot collapse. WITH_SLEEP showed slot_diversity ~= 0.12
     vs WITHOUT_SLEEP ~= 1.0 (reversed!). Removing SHY isolates the
     sleep -> slots -> action pathway under test.
  2. terrain_loss added to E1 training loop (LAMBDA_TERRAIN=0.1).
     EXQ-418's cue_terrain_proj had random weights -- SD-016's action_bias
     pathway was untrained, so action_bias_divergence=0.0 in all seeds.
     Pattern copied from EXQ-182 (v3_exq_182_sd016_terrain_calibration.py).

SCIENTIFIC QUESTION (unchanged from EXQ-418):
  Does periodic sleep consolidation (SD-017) produce ContextMemory slot
  differentiation that, when routed through SD-016's cue_action_proj, yields
  measurably distinct action biases across SAFE vs DANGEROUS contexts?

DESIGN (unchanged from EXQ-418):
  Two conditions, 3 seeds each:
    WITH_SLEEP_SD016:    sws_enabled=True, rem_enabled=True, sd016_enabled=True
    WITHOUT_SLEEP_SD016: sws_enabled=False, rem_enabled=False, sd016_enabled=True

  Training:
    P0 (P0_EPISODES=50): Encoder warmup. Both conditions. Alternating contexts.
    P1 (P1_EPISODES=150): WITH_SLEEP gets sleep every SLEEP_INTERVAL=10 eps.

  Evaluation: EVAL_EPISODES eval episodes; measure action_bias_divergence.

ACCEPTANCE CRITERIA (unchanged from EXQ-418):
  C1: action_bias_divergence(WITH_SLEEP) >= 0.05 in >= 2/3 seeds
  C2: action_bias_divergence(WITH_SLEEP) > action_bias_divergence(WITHOUT_SLEEP)
      in >= 2/3 seeds
  C3: slot_diversity(WITH_SLEEP) > slot_diversity(WITHOUT_SLEEP) in >= 2/3 seeds

PASS: C1 AND C2.

claim_ids: ["SD-017"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id: ends _v3
supersedes: V3-EXQ-418
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_418a_sd016_sd017_context_conditioned_action"
SUPERSEDES_ID   = "V3-EXQ-418"
CLAIM_IDS       = ["SD-017"]
EXPERIMENT_PURPOSE = "evidence"

P0_EPISODES       = 50          # encoder warmup (no sleep in either condition)
P1_EPISODES       = 150         # P1: WITH_SLEEP gets sleep cycles
EVAL_EPISODES     = 30          # evaluation episodes per seed per condition
STEPS_PER_EPISODE = 150
SLEEP_INTERVAL    = 10          # sleep cycle every N P1 episodes (WITH_SLEEP only)
CONTEXT_SWITCH_EVERY = 5        # alternate SAFE/DANGEROUS every N episodes
LAMBDA_TERRAIN    = 0.1         # terrain_loss weight for SD-016 cue_terrain_proj training

LR = 1e-4

SEEDS = [42, 49, 56]

MIN_CONTEXT_SAMPLES = 10


# ---------------------------------------------------------------------------
# Terrain loss helpers (copied from v3_exq_182_sd016_terrain_calibration.py)
# ---------------------------------------------------------------------------

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


def compute_terrain_loss(agent: REEAgent, z_world: torch.Tensor, hazard_max: float) -> torch.Tensor:
    """Supervised terrain_loss for cue_terrain_proj (extract_cue_context WITH gradients)."""
    _, terrain_weight = agent.e1.extract_cue_context(z_world)
    w_harm_target = 0.8 if hazard_max > 0.3 else 0.2
    w_goal_target = 0.8 if hazard_max < 0.1 else 0.3
    target = torch.tensor(
        [[w_harm_target, w_goal_target]],
        dtype=terrain_weight.dtype,
        device=terrain_weight.device,
    )
    return F.mse_loss(terrain_weight, target)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def _make_env_safe(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=8,
        num_hazards=1,
        num_resources=3,
        hazard_harm=0.02,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


def _make_env_dangerous(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed + 1000,
        size=8,
        num_hazards=5,
        num_resources=3,
        hazard_harm=0.04,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _make_agent(env: CausalGridWorldV2, with_sleep: bool) -> REEAgent:
    """Build REEAgent with SD-016 enabled in both conditions.

    Fix vs EXQ-418: shy_enabled=False in both conditions.
    SHY is orthogonal to the SD-016 test; removing it prevents
    slot collapse from rapid decay (0.85^15 ~= 0.09).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # SD-016: frontal cue-indexed context retrieval (BOTH conditions)
        sd016_enabled=True,
        # SD-017: sleep phase switches (only WITH_SLEEP)
        sws_enabled=with_sleep,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=with_sleep,
        rem_attribution_steps=6,
        # SHY disabled: isolates sleep -> slots -> action pathway (fix for EXQ-418)
        shy_enabled=False,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Slot diversity helper
# ---------------------------------------------------------------------------

def _compute_slot_diversity(agent: REEAgent) -> float:
    """Mean pairwise cosine distance between E1 ContextMemory slots.

    0.0 = all slots identical (undifferentiated)
    1.0 = all slots orthogonal (fully differentiated)
    """
    with torch.no_grad():
        mem = agent.e1.context_memory.memory  # [num_slots, memory_dim]
        n = mem.shape[0]
        if n < 2:
            return 0.0
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())  # [n, n]
        mask = ~torch.eye(n, dtype=torch.bool, device=mem.device)
        dist = 1.0 - sim[mask]
        return float(dist.mean().item())


# ---------------------------------------------------------------------------
# Action bias extraction
# ---------------------------------------------------------------------------

def _extract_action_bias(agent: REEAgent, z_world: torch.Tensor) -> Optional[torch.Tensor]:
    """Extract action_bias from E1 ContextMemory via SD-016 cue projection."""
    if not hasattr(agent.e1, 'world_query_proj'):
        return None
    with torch.no_grad():
        z_w = z_world.detach()
        if z_w.dim() == 1:
            z_w = z_w.unsqueeze(0)
        action_bias, _ = agent.e1.extract_cue_context(z_w)
        return action_bias.squeeze(0)


def _action_bias_divergence(
    safe_biases: List[torch.Tensor],
    dang_biases: List[torch.Tensor],
) -> float:
    """Mean cosine distance between action_bias vectors from SAFE vs DANGEROUS."""
    if len(safe_biases) < MIN_CONTEXT_SAMPLES or len(dang_biases) < MIN_CONTEXT_SAMPLES:
        return 0.0
    with torch.no_grad():
        safe_mat = torch.stack(safe_biases[:50])
        dang_mat = torch.stack(dang_biases[:50])
        safe_norm = F.normalize(safe_mat, dim=-1)
        dang_norm = F.normalize(dang_mat, dim=-1)
        sim_mat = torch.mm(safe_norm, dang_norm.t())
        mean_sim = float(sim_mat.mean().item())
        return max(0.0, 1.0 - mean_sim)


# ---------------------------------------------------------------------------
# One-hot action helper
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Training episode (random actions, world-model + terrain_loss update)
# ---------------------------------------------------------------------------

def _run_training_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    """Train for one episode with random actions. Returns (harm_rate, pred_loss).

    Fix vs EXQ-418: terrain_loss added (LAMBDA_TERRAIN=0.1).
    Applied in both P0 and P1 phases whenever sd016_enabled=True.
    Gradient flows through cue_terrain_proj -> output_proj -> world_query_proj.
    """
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    ep_harm = 0.0
    ep_steps = 0
    pred_losses: List[float] = []

    for _step in range(STEPS_PER_EPISODE):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if not torch.is_tensor(obs_body):
            obs_body = torch.tensor(obs_body, dtype=torch.float32, device=device)
        else:
            obs_body = obs_body.to(device)
        if obs_body.dim() == 1:
            obs_body = obs_body.unsqueeze(0)

        if not torch.is_tensor(obs_world):
            obs_world = torch.tensor(obs_world, dtype=torch.float32, device=device)
        else:
            obs_world = obs_world.to(device)
        if obs_world.dim() == 1:
            obs_world = obs_world.unsqueeze(0)

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        # Capture hazard_max BEFORE env.step (uses current obs)
        hazard_max = get_hazard_max(obs_dict, obs_world)

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, harm_signal, done, info, obs_dict = env.step(action)
        ep_harm += max(0.0, float(-harm_signal))
        ep_steps += 1

        # Combined loss: prediction + terrain_loss for SD-016 training
        pred_loss = agent.compute_prediction_loss()
        t_loss = compute_terrain_loss(agent, latent.z_world, hazard_max)
        total_loss = pred_loss + LAMBDA_TERRAIN * t_loss
        if total_loss.requires_grad:
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()
            pred_losses.append(float(pred_loss.item()))

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    harm_rate = ep_harm / max(1, ep_steps)
    mean_pred_loss = float(sum(pred_losses) / len(pred_losses)) if pred_losses else 0.0
    return harm_rate, mean_pred_loss


# ---------------------------------------------------------------------------
# Evaluation episode: collect action_bias vectors per context
# ---------------------------------------------------------------------------

def _run_eval_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    env_label: str,
    safe_biases: List[torch.Tensor],
    dang_biases: List[torch.Tensor],
) -> float:
    """Run one eval episode (no training). Collect action_bias by context."""
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    ep_harm = 0.0
    ep_steps = 0

    for _step in range(STEPS_PER_EPISODE):
        obs_body = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]
        if not torch.is_tensor(obs_body):
            obs_body = torch.tensor(obs_body, dtype=torch.float32, device=device)
        else:
            obs_body = obs_body.to(device)
        if obs_body.dim() == 1:
            obs_body = obs_body.unsqueeze(0)

        if not torch.is_tensor(obs_world):
            obs_world = torch.tensor(obs_world, dtype=torch.float32, device=device)
        else:
            obs_world = obs_world.to(device)
        if obs_world.dim() == 1:
            obs_world = obs_world.unsqueeze(0)

        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        if latent.z_world is not None:
            bias = _extract_action_bias(agent, latent.z_world)
            if bias is not None:
                if env_label == "SAFE":
                    safe_biases.append(bias.cpu())
                else:
                    dang_biases.append(bias.cpu())

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        _, harm_signal, done, _info, obs_dict = env.step(action)
        ep_harm += max(0.0, float(-harm_signal))
        ep_steps += 1

        if done:
            break

    return ep_harm / max(1, ep_steps)


# ---------------------------------------------------------------------------
# Main condition runner
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    with_sleep: bool,
    seed: int,
    dry_run: bool = False,
) -> Dict:
    """Run one condition x seed. Returns result dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    p0_eps   = P0_EPISODES   if not dry_run else 3
    p1_eps   = P1_EPISODES   if not dry_run else 5
    eval_eps = EVAL_EPISODES if not dry_run else 4
    total_eps = p0_eps + p1_eps

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)

    agent = _make_agent(env_safe, with_sleep)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    print(
        f"  seed boundary: cond={condition_name} seed={seed}",
        flush=True,
    )

    # ---- P0: encoder warmup (no sleep in either condition) ----
    for ep in range(p0_eps):
        use_dangerous = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_training_episode(agent, env, optimizer)

        report_interval = max(1, p0_eps // 2)
        if (ep + 1) % report_interval == 0 or ep == 0:
            print(
                f"  [train] cond={condition_name} seed={seed}"
                f" ep {ep + 1}/{total_eps} phase=P0"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    # ---- P1: WITH_SLEEP gets sleep cycles ----
    for ep in range(p1_eps):
        abs_ep = p0_eps + ep
        use_dangerous = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_training_episode(agent, env, optimizer)

        if with_sleep and (ep + 1) % SLEEP_INTERVAL == 0:
            sleep_m = agent.run_sleep_cycle()
            sws_writes = sleep_m.get("sws_n_writes", 0.0)
            sws_div    = sleep_m.get("sws_slot_diversity", 0.0)
            rem_rolls  = sleep_m.get("rem_n_rollouts", 0.0)
            print(
                f"  [sleep] cond={condition_name} seed={seed}"
                f" ep {abs_ep + 1}/{total_eps}"
                f" sws_writes={sws_writes:.0f}"
                f" slot_div={sws_div:.4f}"
                f" rem_rolls={rem_rolls:.0f}",
                flush=True,
            )

        report_interval = max(1, p1_eps // 3)
        if (ep + 1) % report_interval == 0 or ep == p1_eps - 1:
            print(
                f"  [train] cond={condition_name} seed={seed}"
                f" ep {abs_ep + 1}/{total_eps} phase=P1"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    slot_diversity = _compute_slot_diversity(agent)

    # ---- Evaluation ----
    safe_biases: List[torch.Tensor] = []
    dang_biases: List[torch.Tensor] = []
    eval_harm_rates: List[float] = []

    for ev_ep in range(eval_eps):
        if ev_ep % 2 == 0:
            h = _run_eval_episode(agent, env_safe, "SAFE", safe_biases, dang_biases)
        else:
            h = _run_eval_episode(agent, env_dang, "DANGEROUS", safe_biases, dang_biases)
        eval_harm_rates.append(h)

    eval_harm_rate = float(sum(eval_harm_rates) / len(eval_harm_rates)) if eval_harm_rates else 0.0
    action_bias_div = _action_bias_divergence(safe_biases, dang_biases)
    n_safe_samples  = len(safe_biases)
    n_dang_samples  = len(dang_biases)

    print(
        f"  verdict: cond={condition_name} seed={seed}"
        f" slot_div={slot_diversity:.4f}"
        f" action_bias_div={action_bias_div:.4f}"
        f" eval_harm_rate={eval_harm_rate:.4f}"
        f" n_safe={n_safe_samples} n_dang={n_dang_samples}",
        flush=True,
    )

    return {
        "condition": condition_name,
        "seed": seed,
        "slot_diversity": slot_diversity,
        "action_bias_divergence": action_bias_div,
        "eval_harm_rate": eval_harm_rate,
        "n_safe_bias_samples": n_safe_samples,
        "n_dang_bias_samples": n_dang_samples,
        "p0_episodes": p0_eps,
        "p1_episodes": p1_eps,
        "eval_episodes": eval_eps,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Dict:
    conditions = [
        ("WITH_SLEEP_SD016",    True),
        ("WITHOUT_SLEEP_SD016", False),
    ]
    all_results: Dict[str, List[Dict]] = {cname: [] for cname, _ in conditions}

    n_seeds   = len(SEEDS)
    total_runs = len(conditions) * n_seeds

    run_num = 0
    for cond_name, with_sleep in conditions:
        for seed in SEEDS:
            run_num += 1
            print(
                f"[train] ep {run_num}/{total_runs}"
                f" cond={cond_name} seed={seed}",
                flush=True,
            )
            res = run_condition(cond_name, with_sleep, seed, dry_run=dry_run)
            all_results[cond_name].append(res)

    c1_wins = 0
    c2_wins = 0
    c3_wins = 0

    per_seed_comparisons = []
    for with_r, wo_r in zip(
        all_results["WITH_SLEEP_SD016"], all_results["WITHOUT_SLEEP_SD016"]
    ):
        assert with_r["seed"] == wo_r["seed"], "Seed mismatch in comparison"
        s = with_r["seed"]

        bias_threshold_win   = with_r["action_bias_divergence"] >= 0.05
        bias_improvement_win = with_r["action_bias_divergence"] > wo_r["action_bias_divergence"]
        div_win              = with_r["slot_diversity"] > wo_r["slot_diversity"]

        c1_wins += int(bias_threshold_win)
        c2_wins += int(bias_improvement_win)
        c3_wins += int(div_win)

        per_seed_comparisons.append({
            "seed": s,
            "with_action_bias_div": with_r["action_bias_divergence"],
            "without_action_bias_div": wo_r["action_bias_divergence"],
            "c1_bias_threshold_win": bias_threshold_win,
            "c2_bias_improvement_win": bias_improvement_win,
            "with_slot_diversity": with_r["slot_diversity"],
            "without_slot_diversity": wo_r["slot_diversity"],
            "c3_diversity_win": div_win,
            "with_eval_harm_rate": with_r["eval_harm_rate"],
            "without_eval_harm_rate": wo_r["eval_harm_rate"],
            "with_n_safe_samples": with_r["n_safe_bias_samples"],
            "with_n_dang_samples": with_r["n_dang_bias_samples"],
        })

    threshold = 2
    c1_pass = c1_wins >= threshold
    c2_pass = c2_wins >= threshold
    c3_pass = c3_wins >= threshold

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    print(
        f"verdict: C1_bias_threshold={c1_pass} ({c1_wins}/{n_seeds} seeds),"
        f" C2_bias_improvement={c2_pass} ({c2_wins}/{n_seeds} seeds),"
        f" C3_slot_div={c3_pass} ({c3_wins}/{n_seeds} seeds)"
        f" => {outcome}",
        flush=True,
    )

    ts     = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": run_id,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "ablation_pair",
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "supersedes": SUPERSEDES_ID,
        "experiment_version": "a",
        "fix_description": (
            "Two bugs fixed vs EXQ-418: "
            "(1) shy_enabled=False to prevent slot collapse (0.85^15~=0.09 reversed diversity). "
            "(2) terrain_loss added (lambda=0.1) to train cue_terrain_proj -- "
            "EXQ-418 had random weights causing action_bias_divergence=0.0."
        ),
        "acceptance_checks": {
            "C1_action_bias_div_gte_0.05_in_2of3_seeds": c1_pass,
            "C1_wins": c1_wins,
            "C2_action_bias_div_improvement_2of3_seeds": c2_pass,
            "C2_wins": c2_wins,
            "C3_slot_diversity_improvement_2of3_seeds": c3_pass,
            "C3_wins": c3_wins,
            "primary_pass": c1_pass and c2_pass,
            "secondary_pass": c3_pass,
        },
        "per_seed_comparisons": per_seed_comparisons,
        "all_results": {
            cond: results for cond, results in all_results.items()
        },
        "params": {
            "p0_episodes": P0_EPISODES if not dry_run else 3,
            "p1_episodes": P1_EPISODES if not dry_run else 5,
            "eval_episodes": EVAL_EPISODES if not dry_run else 4,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "lambda_terrain": LAMBDA_TERRAIN,
            "seeds": SEEDS,
            "sd016_enabled": True,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
            "shy_enabled": False,
            "shy_decay_rate": "disabled",
            "min_context_samples": MIN_CONTEXT_SAMPLES,
            "dry_run": dry_run,
        },
    }

    if not dry_run:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "REE_assembly", "evidence", "experiments",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{run_id}.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {out_path}", flush=True)
    else:
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run minimal episodes to verify wiring",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
