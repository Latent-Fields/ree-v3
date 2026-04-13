#!/opt/local/bin/python3
"""
V3-EXQ-385 -- INV-049 Offline Consolidation Necessity: Sleep Ablation Probe

experiment_purpose: evidence
Claims: INV-049

SCIENTIFIC QUESTION (INV-049):
  Is offline consolidation (sleep-equivalent phases) a computational NECESSITY
  for stable world-model formation in a model-building agent?

  INV-049 states: any agent that builds a world model from continuous experience
  cannot safely update that model while simultaneously using it to navigate.
  It therefore requires periodic offline phases where action is suspended and
  the model is reorganised. This is not a biological contingency but a general
  computational necessity.

WHAT THIS EXPERIMENT TESTS:
  The necessity claim requires demonstrating that WITHOUT offline phases:
  (1) E1 ContextMemory slots fail to differentiate (low slot_diversity)
  (2) World-model prediction quality degrades or stagnates at a higher loss
  (3) Behavioral quality (harm avoidance) is worse in the absence of offline
      consolidation, because the agent cannot safely reorganise its model

  We compare:
    WITH_OFFLINE   -- agent receives periodic SWS+REM sleep cycles every
                      SLEEP_INTERVAL episodes (SD-017 infrastructure, SHY wiring)
    WITHOUT_OFFLINE -- same agent, same training loop, NO sleep cycles
                       (waking-only continuous update, no offline reorganisation)

DESIGN:
  Two-phase training:
    P0 (P0_EPISODES): Both conditions. Warm up encoder + E1 world model.
                      No sleep in either condition. Buffer fills.
    P1 (P1_EPISODES): WITH_OFFLINE gets a sleep cycle every SLEEP_INTERVAL eps.
                      WITHOUT_OFFLINE continues waking-only.
  Evaluation (EVAL_EPISODES per seed): Run each trained agent on the standard
    SAFE/DANGEROUS alternating environment. Measure harm avoidance.

  Training actions: random (no CEM trajectory generation during training).
    This isolates the world model quality effect from action selection quality.
    The E1 prediction loss directly measures how well the agent models the world.

  Environment: two contexts alternating SAFE (1 hazard) / DANGEROUS (5 hazards)
    every CONTEXT_SWITCH_EVERY episodes. This gives the consolidation mechanism
    real contextual variation to organise (essential for slot differentiation to
    be meaningful).

  N_SEEDS = 3 (seeds 42, 49, 56).

ACCEPTANCE CRITERIA (INV-049 necessity):
  C1: slot_diversity(WITH_OFFLINE) > slot_diversity(WITHOUT_OFFLINE)
      in >= 2 of 3 seeds after P1 training.
      (Offline consolidation improves context differentiation in ContextMemory.)
  C2: late_pred_loss(WITH_OFFLINE) < late_pred_loss(WITHOUT_OFFLINE)
      in >= 2 of 3 seeds.
      (Offline consolidation produces better world model prediction quality.)
  C3: eval_harm_rate(WITH_OFFLINE) < eval_harm_rate(WITHOUT_OFFLINE)
      in >= 2 of 3 seeds.
      (Better model organisation translates to better harm avoidance behaviour.)

PASS: C1 AND C2 (primary structural necessity checks).
  C3 is secondary (behavioural consequence of C1/C2).
  If C1 AND C2 pass but C3 fails: PASS with caveat (model quality improves
  but evaluation episode count may be insufficient to observe behaviour change).

EVIDENCE DIRECTION:
  PASS -> supports INV-049 (offline phases produce measurably better model organisation)
  FAIL -> does_not_support INV-049 (no difference detected at current episode count)

claim_ids: ["INV-049"]
architecture_epoch: "ree_hybrid_guardrails_v1"
run_id ends: _v3
"""

import os
import sys
import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE = "v3_exq_385_inv049_offline_consolidation_necessity"
CLAIM_IDS = ["INV-049"]
EXPERIMENT_PURPOSE = "evidence"

P0_EPISODES = 50          # encoder warmup (no sleep in either condition)
P1_EPISODES = 150         # P1: WITH_OFFLINE gets sleep; WITHOUT_OFFLINE does not
EVAL_EPISODES = 30        # evaluation episodes per seed per condition
STEPS_PER_EPISODE = 150
SLEEP_INTERVAL = 10       # sleep cycle every N P1 episodes (WITH_OFFLINE only)
CONTEXT_SWITCH_EVERY = 5  # alternate SAFE/DANGEROUS every N episodes

LR = 1e-4

SEEDS = [42, 49, 56]

# Loss window for late-training prediction loss (last N episodes of P1)
LATE_WINDOW = 30


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

def _make_agent(env: CausalGridWorldV2, with_offline: bool) -> REEAgent:
    """Build REEAgent. WITH_OFFLINE enables SD-017 + SHY wiring."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # SD-017: sleep phase switches (only active in WITH_OFFLINE)
        sws_enabled=with_offline,
        sws_consolidation_steps=8,
        sws_schema_weight=0.1,
        rem_enabled=with_offline,
        rem_attribution_steps=6,
        # MECH-120: SHY normalisation (only meaningful with sleep)
        shy_enabled=with_offline,
        shy_decay_rate=0.85,
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
# One-hot action helper
# ---------------------------------------------------------------------------

def _onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


# ---------------------------------------------------------------------------
# Single episode step (training: random actions; eval: random actions)
# Training uses random actions to isolate world model learning from
# action quality. The E1 prediction loss directly measures world model quality.
# ---------------------------------------------------------------------------

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    train: bool = True,
) -> Tuple[float, float]:
    """Run one episode. Returns (episode_harm_rate, mean_pred_loss).

    Uses random actions throughout to isolate E1 world-model quality
    from action selection quality. This is intentional: INV-049 is about
    world-model formation, not action policy.
    """
    device = agent.device
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()

    ep_harm = 0.0
    ep_steps = 0
    pred_losses: List[float] = []

    for _step in range(STEPS_PER_EPISODE):
        # Ensure obs are tensors on device
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

        # Build latent state (world model update)
        latent = agent.sense(obs_body, obs_world)
        agent.clock.advance()

        # Random action (avoids expensive CEM trajectory generation)
        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

        # Record transition for E2 training buffer
        if agent._current_latent is not None:
            z_self_prev = agent._current_latent.z_self.detach().clone()
            agent.record_transition(z_self_prev, action, latent.z_self.detach())

        _, harm_signal, done, info, obs_dict = env.step(action)
        ep_harm += max(0.0, float(-harm_signal))
        ep_steps += 1

        if train:
            pred_loss = agent.compute_prediction_loss()
            if pred_loss.requires_grad:
                optimizer.zero_grad()
                pred_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()
                pred_losses.append(float(pred_loss.item()))

        agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)

        if done:
            break

    harm_rate = ep_harm / max(1, ep_steps)
    mean_pred_loss = (
        float(sum(pred_losses) / len(pred_losses)) if pred_losses else 0.0
    )
    return harm_rate, mean_pred_loss


# ---------------------------------------------------------------------------
# Main condition runner
# ---------------------------------------------------------------------------

def run_condition(
    condition_name: str,
    with_offline: bool,
    seed: int,
    dry_run: bool = False,
) -> Dict:
    """Run one condition for one seed. Returns result dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    p0_eps = P0_EPISODES if not dry_run else 3
    p1_eps = P1_EPISODES if not dry_run else 5
    eval_eps = EVAL_EPISODES if not dry_run else 2
    total_eps = p0_eps + p1_eps

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)

    agent = _make_agent(env_safe, with_offline)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )
    device = agent.device

    all_pred_losses: List[float] = []
    episode_harm_rates: List[float] = []

    print(
        f"  seed boundary: cond={condition_name} seed={seed}",
        flush=True,
    )

    # ---- P0: encoder warmup (no sleep in either condition) ----
    for ep in range(p0_eps):
        use_dangerous = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_episode(agent, env, optimizer, train=True)
        episode_harm_rates.append(harm_rate)
        all_pred_losses.append(pred_loss)

        report_interval = max(1, p0_eps // 2)
        if (ep + 1) % report_interval == 0 or ep == 0:
            print(
                f"  [train] cond={condition_name} seed={seed}"
                f" ep {ep + 1}/{total_eps} phase=P0"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    # ---- P1: WITH_OFFLINE gets sleep; WITHOUT_OFFLINE continues waking ----
    for ep in range(p1_eps):
        abs_ep = p0_eps + ep
        use_dangerous = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_episode(agent, env, optimizer, train=True)
        episode_harm_rates.append(harm_rate)
        all_pred_losses.append(pred_loss)

        # WITH_OFFLINE: periodic sleep cycle after each interval
        if with_offline and (ep + 1) % SLEEP_INTERVAL == 0:
            sleep_m = agent.run_sleep_cycle()
            sws_writes = sleep_m.get("sws_n_writes", 0.0)
            sws_div = sleep_m.get("sws_slot_diversity", 0.0)
            rem_rolls = sleep_m.get("rem_n_rollouts", 0.0)
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

    # Final slot diversity
    slot_diversity = _compute_slot_diversity(agent)

    # Late-window prediction loss (last LATE_WINDOW P1 episodes)
    late_losses = all_pred_losses[-(min(LATE_WINDOW, len(all_pred_losses))):]
    late_pred_loss = (
        float(sum(late_losses) / len(late_losses)) if late_losses else 0.0
    )

    # ---- Evaluation: harm avoidance ----
    eval_harm_rates: List[float] = []
    for ev_ep in range(eval_eps):
        use_dangerous = ev_ep % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, _ = _run_episode(agent, env, optimizer=optimizer, train=False)
        eval_harm_rates.append(harm_rate)

    eval_harm_rate = (
        float(sum(eval_harm_rates) / len(eval_harm_rates))
        if eval_harm_rates else 0.0
    )

    print(
        f"  verdict: cond={condition_name} seed={seed}"
        f" slot_div={slot_diversity:.4f}"
        f" late_pred_loss={late_pred_loss:.4f}"
        f" eval_harm_rate={eval_harm_rate:.4f}",
        flush=True,
    )

    return {
        "condition": condition_name,
        "seed": seed,
        "slot_diversity": slot_diversity,
        "late_pred_loss": late_pred_loss,
        "eval_harm_rate": eval_harm_rate,
        "p0_episodes": p0_eps,
        "p1_episodes": p1_eps,
        "eval_episodes": eval_eps,
        "mean_harm_rate_training": (
            float(sum(episode_harm_rates) / len(episode_harm_rates))
            if episode_harm_rates else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Dict:
    conditions = [
        ("WITH_OFFLINE",    True),
        ("WITHOUT_OFFLINE", False),
    ]
    all_results: Dict[str, List[Dict]] = {cname: [] for cname, _ in conditions}

    n_seeds = len(SEEDS)
    total_runs = len(conditions) * n_seeds

    run_num = 0
    for cond_name, with_offline in conditions:
        for seed in SEEDS:
            run_num += 1
            print(
                f"[train] ep {run_num}/{total_runs}"
                f" cond={cond_name} seed={seed}",
                flush=True,
            )
            res = run_condition(cond_name, with_offline, seed, dry_run=dry_run)
            all_results[cond_name].append(res)

    # ---- Per-seed comparisons ----
    c1_wins = 0  # slot_diversity WITH > WITHOUT
    c2_wins = 0  # late_pred_loss WITH < WITHOUT
    c3_wins = 0  # eval_harm_rate WITH < WITHOUT

    per_seed_comparisons = []
    for with_r, wo_r in zip(
        all_results["WITH_OFFLINE"], all_results["WITHOUT_OFFLINE"]
    ):
        assert with_r["seed"] == wo_r["seed"], "Seed mismatch in comparison"
        s = with_r["seed"]
        div_win = with_r["slot_diversity"] > wo_r["slot_diversity"]
        loss_win = (
            with_r["late_pred_loss"] < wo_r["late_pred_loss"]
            if with_r["late_pred_loss"] > 0 and wo_r["late_pred_loss"] > 0
            else False
        )
        harm_win = with_r["eval_harm_rate"] < wo_r["eval_harm_rate"]

        c1_wins += int(div_win)
        c2_wins += int(loss_win)
        c3_wins += int(harm_win)

        per_seed_comparisons.append({
            "seed": s,
            "with_slot_diversity": with_r["slot_diversity"],
            "without_slot_diversity": wo_r["slot_diversity"],
            "c1_diversity_win": div_win,
            "with_late_pred_loss": with_r["late_pred_loss"],
            "without_late_pred_loss": wo_r["late_pred_loss"],
            "c2_loss_win": loss_win,
            "with_eval_harm_rate": with_r["eval_harm_rate"],
            "without_eval_harm_rate": wo_r["eval_harm_rate"],
            "c3_harm_win": harm_win,
        })

    # ---- Acceptance criteria ----
    threshold = 2  # require >= 2 of 3 seeds
    c1_pass = c1_wins >= threshold
    c2_pass = c2_wins >= threshold
    c3_pass = c3_wins >= threshold

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    # Print verdict lines for runner
    print(
        f"verdict: C1_slot_diversity={c1_pass} ({c1_wins}/{n_seeds} seeds),"
        f" C2_pred_loss={c2_pass} ({c2_wins}/{n_seeds} seeds),"
        f" C3_harm={c3_pass} ({c3_wins}/{n_seeds} seeds)"
        f" => {outcome}",
        flush=True,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
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
        "acceptance_checks": {
            "C1_slot_diversity_2of3_seeds": c1_pass,
            "C1_wins": c1_wins,
            "C2_pred_loss_2of3_seeds": c2_pass,
            "C2_wins": c2_wins,
            "C3_harm_rate_2of3_seeds": c3_pass,
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
            "eval_episodes": EVAL_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "seeds": SEEDS,
            "late_window": LATE_WINDOW,
            "sws_consolidation_steps": 8,
            "rem_attribution_steps": 6,
            "shy_enabled": True,
            "shy_decay_rate": 0.85,
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-385: INV-049 offline consolidation necessity probe"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run minimal episodes (3+5 per condition) to verify wiring",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override STEPS_PER_EPISODE for quick testing",
    )
    args = parser.parse_args()

    if args.steps is not None:
        STEPS_PER_EPISODE = args.steps

    main(dry_run=args.dry_run)
