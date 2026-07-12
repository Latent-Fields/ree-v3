#!/opt/local/bin/python3
"""
V3-EXQ-673 -- MECH-171 Vicious Cycle: Sleep Disruption Amplification
SLEEP DRIVER: manual-multi (run_sleep_cycle() called every SLEEP_INTERVAL episodes)

experiment_purpose: evidence
Claims: MECH-171

SCIENTIFIC QUESTION:
  Does early sleep phase disruption trigger a cascading vicious cycle through
  consolidation failure -> attribution failure -> behavioral disorientation
  -> increased arousal -> further sleep degradation?

  MECH-171 predicts: sleep disruption at different stages produces different
  cumulative effects. Early disruption (ARM_B) should show amplifying degradation
  because the cycle compounds over time, while late disruption (ARM_C) shows
  one-time drop without amplification.

THREE-ARM COMPARISON (N=3 seeds each):
  ARM_A (HEALTHY_SLEEP): Full sleep architecture throughout all phases.
    Baseline for comparison.

  ARM_B (EARLY_DISRUPTION): Sleep phase compression starts at episode 50.
    Reduced SWS steps (8->3) and REM steps (6->2) to simulate disruption.
    Tests whether early disruption triggers vicious cycle amplification.

  ARM_C (LATE_DISRUPTION): Sleep phase compression starts at episode 150.
    Same compression as ARM_B but applied later.
    Tests critical window hypothesis: equal total disruption applied late
    should produce less cumulative damage than early disruption.

TRAINING PHASES:
  P0 (0-50):     Baseline warmup, all arms identical
  P1 (50-150):   ARM_B disruption begins, ARM_A/C continue normal
  P2 (150-250):  ARM_C disruption begins, ARM_B continues disrupted, ARM_A normal
  P3 (250-300):  All continue their patterns, measure cumulative effects

CONTEXT SWITCHING:
  Alternate safe/dangerous contexts every 5 episodes to test attribution accuracy

SUCCESS CRITERIA (>= 2/3 seeds):
  C1: slot_diversity(ARM_A) > slot_diversity(ARM_B) + 0.05
      (early disruption impairs consolidation more than baseline)
  C2: slot_diversity(ARM_B) < slot_diversity(ARM_C) - 0.03
      (early disruption worse than late disruption, confirms critical window)
  C3: eval_harm_rate(ARM_A) < eval_harm_rate(ARM_B) + 0.01
      (early disruption increases behavioral harm)
  C4: eval_harm_rate(ARM_B) > eval_harm_rate(ARM_C) + 0.005
      (early disruption worse than late, confirms vicious cycle vs linear)
  C5: late_pred_loss(ARM_A) < late_pred_loss(ARM_B) + 0.05
      (early disruption impairs attribution)

PASS: (C1 AND C3 AND C5) OR (C2 AND C4)
  First disjunct: early disruption causes degradation vs baseline
  Second disjunct: early disruption worse than late, confirms vicious cycle

claim_ids: ["MECH-171"]
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXPERIMENT_TYPE    = "v3_exq_673_mech171_vicious_cycle_sleep_disruption"
CLAIM_IDS          = ["MECH-171"]
EXPERIMENT_PURPOSE = "evidence"

P0_EPISODES       = 50          # baseline warmup
P1_EPISODES       = 100         # P1: ARM_B disruption begins
P2_EPISODES       = 100         # P2: ARM_C disruption begins
P3_EPISODES       = 50          # P3: measure cumulative effects
EVAL_EPISODES     = 30          # evaluation episodes per seed per arm
STEPS_PER_EPISODE = 150
SLEEP_INTERVAL    = 10          # sleep cycle every N episodes
CONTEXT_SWITCH_EVERY = 5        # alternate SAFE/DANGEROUS every N episodes

LR = 1e-4

SEEDS = [42, 49, 56]

LATE_WINDOW = 30

# Sleep phase step counts
HEALTHY_SWS_STEPS = 8
HEALTHY_REM_STEPS = 6
DISRUPTED_SWS_STEPS = 3  # compressed during disruption
DISRUPTED_REM_STEPS = 2  # compressed during disruption


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

def _make_agent(env: CausalGridWorldV2, sws_steps: int, rem_steps: int) -> REEAgent:
    """Build REEAgent with configurable sleep phase step counts."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        alpha_self=0.3,
        # SD-017: sleep phase switches
        sws_enabled=True,
        sws_consolidation_steps=sws_steps,
        sws_schema_weight=0.1,
        rem_enabled=True,
        rem_attribution_steps=rem_steps,
        # MECH-120: SHY normalisation with gentle decay
        shy_enabled=True,
        shy_decay_rate=0.98,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Slot diversity helper
# ---------------------------------------------------------------------------

def _compute_slot_diversity(agent: REEAgent) -> float:
    """Mean pairwise cosine distance between E1 ContextMemory slots."""
    with torch.no_grad():
        mem = agent.e1.context_memory.memory
        n = mem.shape[0]
        if n < 2:
            return 0.0
        normed = F.normalize(mem, dim=-1)
        sim = torch.mm(normed, normed.t())
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
# Single episode step
# ---------------------------------------------------------------------------

def _run_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    train: bool = True,
) -> Tuple[float, float]:
    """Run one episode. Returns (episode_harm_rate, mean_pred_loss)."""
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

        action_idx = random.randint(0, env.action_dim - 1)
        action = _onehot(action_idx, env.action_dim, device)

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
    arm_name: str,
    early_disruption: bool,
    late_disruption: bool,
    seed: int,
    dry_run: bool = False,
) -> Dict:
    """Run one arm for one seed. Returns result dict."""
    torch.manual_seed(seed)
    random.seed(seed)

    p0_eps   = P0_EPISODES   if not dry_run else 5
    p1_eps   = P1_EPISODES   if not dry_run else 5
    p2_eps   = P2_EPISODES   if not dry_run else 5
    p3_eps   = P3_EPISODES   if not dry_run else 3
    eval_eps = EVAL_EPISODES if not dry_run else 2
    total_eps = p0_eps + p1_eps + p2_eps + p3_eps

    env_safe = _make_env_safe(seed)
    env_dang = _make_env_dangerous(seed)

    # Start with healthy sleep parameters
    agent = _make_agent(env_safe, HEALTHY_SWS_STEPS, HEALTHY_REM_STEPS)
    optimizer = optim.Adam(
        list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
        lr=LR,
    )

    all_pred_losses: List[float] = []
    episode_harm_rates: List[float] = []
    sleep_quality_metrics: List[Dict[str, float]] = []

    print(
        f"  seed boundary: arm={arm_name} seed={seed}",
        flush=True,
    )

    # Helper to run sleep cycle with current agent config
    def run_sleep_and_record(ep_num: int, phase_name: str):
        sleep_m = agent.run_sleep_cycle()
        sws_writes = sleep_m.get("sws_n_writes", 0.0)
        sws_div    = sleep_m.get("sws_slot_diversity", 0.0)
        rem_rolls  = sleep_m.get("rem_n_rollouts", 0.0)
        sleep_quality_metrics.append({
            "episode": ep_num,
            "phase": phase_name,
            "sws_writes": sws_writes,
            "rem_rollouts": rem_rolls,
            "slot_diversity": sws_div,
        })
        print(
            f"  [sleep] arm={arm_name} seed={seed}"
            f" ep {ep_num}/{total_eps} phase={phase_name}"
            f" sws_writes={sws_writes:.0f}"
            f" slot_div={sws_div:.4f}"
            f" rem_rolls={rem_rolls:.0f}",
            flush=True,
        )

    # ---- P0: baseline warmup (no disruption in any arm) ----
    for ep in range(p0_eps):
        use_dangerous = (ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_episode(agent, env, optimizer, train=True)
        episode_harm_rates.append(harm_rate)
        all_pred_losses.append(pred_loss)

        if (ep + 1) % SLEEP_INTERVAL == 0:
            run_sleep_and_record(ep + 1, "P0")

        report_interval = max(1, p0_eps // 2)
        if (ep + 1) % report_interval == 0 or ep == 0:
            print(
                f"  [train] arm={arm_name} seed={seed}"
                f" ep {ep + 1}/{total_eps} phase=P0"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    # ---- P1: ARM_B disruption begins ----
    for ep in range(p1_eps):
        abs_ep = p0_eps + ep

        # Apply early disruption if ARM_B
        if early_disruption and ep == 0:
            print(f"  [disruption] arm={arm_name} seed={seed} ep {abs_ep}: "
                  f"switching to disrupted sleep (SWS {DISRUPTED_SWS_STEPS}, "
                  f"REM {DISRUPTED_REM_STEPS})", flush=True)
            # Rebuild agent with disrupted sleep parameters
            new_agent = _make_agent(env_safe, DISRUPTED_SWS_STEPS, DISRUPTED_REM_STEPS)
            # Transfer learned parameters
            new_agent.e1.load_state_dict(agent.e1.state_dict())
            new_agent.latent_stack.load_state_dict(agent.latent_stack.state_dict())
            agent = new_agent
            optimizer = optim.Adam(
                list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
                lr=LR,
            )

        use_dangerous = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_episode(agent, env, optimizer, train=True)
        episode_harm_rates.append(harm_rate)
        all_pred_losses.append(pred_loss)

        if (abs_ep + 1) % SLEEP_INTERVAL == 0:
            run_sleep_and_record(abs_ep + 1, "P1")

        report_interval = max(1, p1_eps // 3)
        if (ep + 1) % report_interval == 0 or ep == p1_eps - 1:
            print(
                f"  [train] arm={arm_name} seed={seed}"
                f" ep {abs_ep + 1}/{total_eps} phase=P1"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    # ---- P2: ARM_C disruption begins ----
    for ep in range(p2_eps):
        abs_ep = p0_eps + p1_eps + ep

        # Apply late disruption if ARM_C
        if late_disruption and ep == 0:
            print(f"  [disruption] arm={arm_name} seed={seed} ep {abs_ep}: "
                  f"switching to disrupted sleep (SWS {DISRUPTED_SWS_STEPS}, "
                  f"REM {DISRUPTED_REM_STEPS})", flush=True)
            new_agent = _make_agent(env_safe, DISRUPTED_SWS_STEPS, DISRUPTED_REM_STEPS)
            new_agent.e1.load_state_dict(agent.e1.state_dict())
            new_agent.latent_stack.load_state_dict(agent.latent_stack.state_dict())
            agent = new_agent
            optimizer = optim.Adam(
                list(agent.e1.parameters()) + list(agent.latent_stack.parameters()),
                lr=LR,
            )

        use_dangerous = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_episode(agent, env, optimizer, train=True)
        episode_harm_rates.append(harm_rate)
        all_pred_losses.append(pred_loss)

        if (abs_ep + 1) % SLEEP_INTERVAL == 0:
            run_sleep_and_record(abs_ep + 1, "P2")

        report_interval = max(1, p2_eps // 3)
        if (ep + 1) % report_interval == 0 or ep == p2_eps - 1:
            print(
                f"  [train] arm={arm_name} seed={seed}"
                f" ep {abs_ep + 1}/{total_eps} phase=P2"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    # ---- P3: measure cumulative effects ----
    for ep in range(p3_eps):
        abs_ep = p0_eps + p1_eps + p2_eps + ep
        use_dangerous = (abs_ep // CONTEXT_SWITCH_EVERY) % 2 == 1
        env = env_dang if use_dangerous else env_safe
        harm_rate, pred_loss = _run_episode(agent, env, optimizer, train=True)
        episode_harm_rates.append(harm_rate)
        all_pred_losses.append(pred_loss)

        if (abs_ep + 1) % SLEEP_INTERVAL == 0:
            run_sleep_and_record(abs_ep + 1, "P3")

        if (ep + 1) % max(1, p3_eps // 2) == 0 or ep == p3_eps - 1:
            print(
                f"  [train] arm={arm_name} seed={seed}"
                f" ep {abs_ep + 1}/{total_eps} phase=P3"
                f" harm_rate={harm_rate:.4f} pred_loss={pred_loss:.4f}",
                flush=True,
            )

    slot_diversity = _compute_slot_diversity(agent)

    late_losses = all_pred_losses[-(min(LATE_WINDOW, len(all_pred_losses))):]
    late_pred_loss = (
        float(sum(late_losses) / len(late_losses)) if late_losses else 0.0
    )

    # ---- Evaluation ----
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
        f"  verdict: arm={arm_name} seed={seed}"
        f" slot_div={slot_diversity:.4f}"
        f" late_pred_loss={late_pred_loss:.4f}"
        f" eval_harm_rate={eval_harm_rate:.4f}",
        flush=True,
    )

    return {
        "arm": arm_name,
        "seed": seed,
        "slot_diversity": slot_diversity,
        "late_pred_loss": late_pred_loss,
        "eval_harm_rate": eval_harm_rate,
        "sleep_quality_metrics": sleep_quality_metrics,
        "mean_harm_rate_training": (
            float(sum(episode_harm_rates) / len(episode_harm_rates))
            if episode_harm_rates else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> Dict:
    arms = [
        ("ARM_A_HEALTHY", False, False),  # no disruption
        ("ARM_B_EARLY", True, False),     # early disruption (P1)
        ("ARM_C_LATE", False, True),      # late disruption (P2)
    ]
    all_results: Dict[str, List[Dict]] = {name: [] for name, _, _ in arms}

    n_seeds    = len(SEEDS)
    total_runs = len(arms) * n_seeds

    run_num = 0
    for arm_name, early_disrupt, late_disrupt in arms:
        for seed in SEEDS:
            run_num += 1
            print(
                f"[train] run {run_num}/{total_runs}"
                f" arm={arm_name} seed={seed}",
                flush=True,
            )
            res = run_condition(
                arm_name, early_disrupt, late_disrupt, seed, dry_run=dry_run
            )
            all_results[arm_name].append(res)

    # Compute success criteria
    c1_wins = 0  # ARM_A > ARM_B slot diversity
    c2_wins = 0  # ARM_B < ARM_C slot diversity (critical window)
    c3_wins = 0  # ARM_A < ARM_B harm rate
    c4_wins = 0  # ARM_B > ARM_C harm rate (vicious cycle)
    c5_wins = 0  # ARM_A < ARM_B pred loss

    per_seed_comparisons = []
    for a_r, b_r, c_r in zip(
        all_results["ARM_A_HEALTHY"],
        all_results["ARM_B_EARLY"],
        all_results["ARM_C_LATE"],
    ):
        assert a_r["seed"] == b_r["seed"] == c_r["seed"], "Seed mismatch"
        s = a_r["seed"]

        c1_win = (a_r["slot_diversity"] > b_r["slot_diversity"] + 0.05)
        c2_win = (b_r["slot_diversity"] < c_r["slot_diversity"] - 0.03)
        c3_win = (a_r["eval_harm_rate"] < b_r["eval_harm_rate"] + 0.01)
        c4_win = (b_r["eval_harm_rate"] > c_r["eval_harm_rate"] + 0.005)
        c5_win = (a_r["late_pred_loss"] < b_r["late_pred_loss"] + 0.05)

        c1_wins += int(c1_win)
        c2_wins += int(c2_win)
        c3_wins += int(c3_win)
        c4_wins += int(c4_win)
        c5_wins += int(c5_win)

        per_seed_comparisons.append({
            "seed": s,
            "arm_a_slot_diversity": a_r["slot_diversity"],
            "arm_b_slot_diversity": b_r["slot_diversity"],
            "arm_c_slot_diversity": c_r["slot_diversity"],
            "c1_slot_a_gt_b": c1_win,
            "c2_slot_b_lt_c": c2_win,
            "arm_a_eval_harm": a_r["eval_harm_rate"],
            "arm_b_eval_harm": b_r["eval_harm_rate"],
            "arm_c_eval_harm": c_r["eval_harm_rate"],
            "c3_harm_a_lt_b": c3_win,
            "c4_harm_b_gt_c": c4_win,
            "arm_a_late_pred_loss": a_r["late_pred_loss"],
            "arm_b_late_pred_loss": b_r["late_pred_loss"],
            "arm_c_late_pred_loss": c_r["late_pred_loss"],
            "c5_loss_a_lt_b": c5_win,
        })

    threshold = 2
    c1_pass = c1_wins >= threshold
    c2_pass = c2_wins >= threshold
    c3_pass = c3_wins >= threshold
    c4_pass = c4_wins >= threshold
    c5_pass = c5_wins >= threshold

    # PASS: (C1 AND C3 AND C5) OR (C2 AND C4)
    outcome = "PASS" if ((c1_pass and c3_pass and c5_pass) or (c2_pass and c4_pass)) else "FAIL"

    print(
        f"verdict: C1_slot_a_gt_b={c1_pass} ({c1_wins}/{n_seeds} seeds),"
        f" C2_slot_b_lt_c={c2_pass} ({c2_wins}/{n_seeds} seeds),"
        f" C3_harm_a_lt_b={c3_pass} ({c3_wins}/{n_seeds} seeds),"
        f" C4_harm_b_gt_c={c4_pass} ({c4_wins}/{n_seeds} seeds),"
        f" C5_loss_a_lt_b={c5_pass} ({c5_wins}/{n_seeds} seeds)"
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
        "evidence_class": "ablation_multi",
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "acceptance_checks": {
            "C1_slot_diversity_a_gt_b_2of3_seeds": c1_pass,
            "C1_wins": c1_wins,
            "C2_slot_diversity_b_lt_c_2of3_seeds": c2_pass,
            "C2_wins": c2_wins,
            "C3_harm_a_lt_b_2of3_seeds": c3_pass,
            "C3_wins": c3_wins,
            "C4_harm_b_gt_c_2of3_seeds": c4_pass,
            "C4_wins": c4_wins,
            "C5_loss_a_lt_b_2of3_seeds": c5_pass,
            "C5_wins": c5_wins,
            "baseline_degradation_pass": c1_pass and c3_pass and c5_pass,
            "vicious_cycle_pass": c2_pass and c4_pass,
        },
        "per_seed_comparisons": per_seed_comparisons,
        "all_results": {
            arm: results for arm, results in all_results.items()
        },
        "params": {
            "p0_episodes": P0_EPISODES if not dry_run else 5,
            "p1_episodes": P1_EPISODES if not dry_run else 5,
            "p2_episodes": P2_EPISODES if not dry_run else 5,
            "p3_episodes": P3_EPISODES if not dry_run else 3,
            "eval_episodes": EVAL_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "sleep_interval": SLEEP_INTERVAL,
            "context_switch_every": CONTEXT_SWITCH_EVERY,
            "seeds": SEEDS,
            "late_window": LATE_WINDOW,
            "healthy_sws_steps": HEALTHY_SWS_STEPS,
            "healthy_rem_steps": HEALTHY_REM_STEPS,
            "disrupted_sws_steps": DISRUPTED_SWS_STEPS,
            "disrupted_rem_steps": DISRUPTED_REM_STEPS,
            "dry_run": dry_run,
        },
    }

    if not dry_run:
        out_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "REE_assembly", "evidence", "experiments",
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = write_flat_manifest(
            output,
            out_dir,
            dry_run=False,
            config=output.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Results written to {out_path}", flush=True)
    else:
        print(f"[DRY RUN] run_id={run_id} outcome={outcome}", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-673: MECH-171 vicious cycle sleep disruption"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run minimal episodes to verify wiring",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override STEPS_PER_EPISODE for quick testing",
    )
    args = parser.parse_args()

    if args.steps is not None:
        STEPS_PER_EPISODE = args.steps

    main(dry_run=args.dry_run)
