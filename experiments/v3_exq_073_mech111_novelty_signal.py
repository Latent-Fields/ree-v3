"""
V3-EXQ-073 -- MECH-111 Novelty Signal

Claims: MECH-111

MECH-111 asserts that E1 prediction error variance constitutes a novelty signal
that, when fed back as a trajectory scoring bonus, increases exploratory behaviour
(higher policy entropy, more novel cells visited) without substantially degrading
harm-avoidance.

Two conditions (matched seeds):
  A. NoveltyOFF -- novelty_bonus_weight=0 (no novelty bonus, current default)
  B. NoveltyON  -- novelty_bonus_weight=0.1 (E1 error EMA as exploration bonus)

The novelty EMA is updated from actual E1 prediction error at each step.
E3.score_trajectory() subtracts novelty_bonus_weight * _novelty_ema from each
trajectory's score -- lower score = preferred -- so higher novelty = preferred.

PASS criteria (ALL required):
  C1: policy_entropy_novelty >= policy_entropy_baseline + 0.10
      (novelty bonus increases action diversity)
  C2: novel_cell_visits_novelty >= novel_cell_visits_baseline + 3
      (more unique cells visited in novelty condition)
  C3: harm_rate_novelty <= harm_rate_baseline + 0.02
      (novelty does not substantially increase harm exposure)
  C4: novelty_ema_nonzero (novelty signal is actually non-zero at eval time)

Informational:
  mean_e1_prediction_error per condition (should be higher in novelty condition
  early in training as agent visits more unpredictable states)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import math
from typing import Dict, List, Set

import torch
import torch.optim as optim
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_073_mech111_novelty_signal"
CLAIM_IDS = ["MECH-111"]

BODY_OBS_DIM = 10   # no proxy fields needed for this experiment
WORLD_OBS_DIM = 54
ACTION_DIM = 4


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=10,
        num_resources=5,
        num_hazards=3,
        use_proxy_fields=False,
        seed=seed,
    )


def _action_entropy(action_counts: List[int]) -> float:
    total = sum(action_counts) + 1e-8
    probs = [c / total for c in action_counts]
    return -sum(p * math.log(p + 1e-9) for p in probs if p > 0)


def _run_single(
    seed: int,
    novelty_enabled: bool,
    novelty_bonus_weight: float,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
) -> Dict:
    cond_label = f"NOVELTY_ON(w={novelty_bonus_weight})" if novelty_enabled else "NOVELTY_OFF"

    torch.manual_seed(seed)
    random.seed(seed)

    env = _make_env(seed)

    config = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        novelty_bonus_weight=novelty_bonus_weight if novelty_enabled else 0.0,
    )
    agent = REEAgent(config)

    e1_opt = optim.Adam(agent.e1.parameters(), lr=lr)
    e3_opt = optim.Adam(
        list(agent.e3.parameters()) + list(agent.latent_stack.parameters()),
        lr=lr,
    )
    e2_opt = optim.Adam(agent.e2.parameters(), lr=lr * 3)

    print(f"\n[EXQ-073] TRAIN {cond_label} seed={seed}", flush=True)
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()
        ep_harm = 0.0

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            z_self_t = None
            if agent._current_latent is not None:
                z_self_t = agent._current_latent.z_self.detach().clone()

            latent = agent.sense(obs_body, obs_world)
            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, world_dim, device=agent.device
            )

            if z_self_t is not None:
                agent.record_transition(z_self_t, latent.z_self.detach().clone(),
                                        latent.z_self.detach().clone())

            # E1 prediction loss -- also drives novelty EMA update
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                e1_opt.zero_grad()
                e1_loss.backward()
                e1_opt.step()
                # Update novelty EMA from E1 loss value
                if novelty_enabled:
                    agent.e3.update_novelty_ema(float(e1_loss.item()))

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            # E2 loss
            e2_loss = agent.compute_e2_loss()
            if e2_loss.requires_grad:
                e2_opt.zero_grad()
                e2_loss.backward()
                e2_opt.step()

            _, reward, done, info, obs_dict = env.step(action)
            harm_signal = float(reward) if reward < 0 else 0.0
            ep_harm += abs(harm_signal)

            # E3 harm supervision
            if agent._current_latent is not None:
                z_world = agent._current_latent.z_world.detach()
                harm_target = torch.tensor(
                    [[1.0 if harm_signal < 0 else 0.0]], device=agent.device
                )
                harm_loss = F.mse_loss(agent.e3.harm_eval(z_world), harm_target)
                e3_opt.zero_grad()
                harm_loss.backward()
                e3_opt.step()

            agent.update_residue(harm_signal)
            if done:
                break

        if (ep + 1) % 50 == 0:
            print(
                f"  [train] {cond_label} seed={seed} ep {ep+1}/{warmup_episodes}"
                f" harm={ep_harm:.3f} novelty_ema={agent.e3._novelty_ema:.5f}",
                flush=True,
            )

    # --- EVAL ---
    agent.eval()
    action_counts = [0] * ACTION_DIM
    harm_events = 0
    total_steps = 0
    visited_cells: Set[tuple] = set()
    e1_errors: List[float] = []
    novelty_ema_at_eval = agent.e3._novelty_ema

    for ep in range(eval_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = torch.tensor(obs_dict["body_state"],  dtype=torch.float32)
            obs_world = torch.tensor(obs_dict["world_state"], dtype=torch.float32)

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks = agent.clock.advance()
                e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                    1, world_dim, device=agent.device
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks, temperature=0.5)

                # Track E1 error for informational purposes
                e1_loss_val = agent.compute_prediction_loss()
                if not (e1_loss_val == 0.0):
                    e1_errors.append(float(e1_loss_val.item()))

            action_idx = int(action.squeeze().argmax().item())
            action_counts[action_idx] += 1

            _, reward, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in ("agent_caused_hazard", "hazard_approach"):
                harm_events += 1

            # Track visited grid cells (approximate from body_state position)
            pos_x = int(obs_dict["body_state"][0] * 10)
            pos_y = int(obs_dict["body_state"][1] * 10)
            visited_cells.add((pos_x, pos_y))
            total_steps += 1

            if done:
                break

    policy_entropy = _action_entropy(action_counts)
    harm_rate = harm_events / max(1, total_steps)
    novel_cell_visits = len(visited_cells)
    mean_e1_error = sum(e1_errors) / max(1, len(e1_errors))

    print(
        f"  [eval] {cond_label} seed={seed}"
        f" entropy={policy_entropy:.4f}"
        f" cells={novel_cell_visits}"
        f" harm_rate={harm_rate:.4f}"
        f" novelty_ema={novelty_ema_at_eval:.5f}"
        f" mean_e1_err={mean_e1_error:.4f}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "novelty_enabled": novelty_enabled,
        "policy_entropy": policy_entropy,
        "novel_cell_visits": novel_cell_visits,
        "harm_rate": harm_rate,
        "novelty_ema_at_eval": novelty_ema_at_eval,
        "mean_e1_prediction_error": mean_e1_error,
        "total_steps": total_steps,
    }


def run(
    seed: int = 42,
    novelty_bonus_weight: float = 0.1,
    warmup_episodes: int = 150,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    """MECH-111: novelty bonus OFF vs ON."""
    print(f"\n[EXQ-073] MECH-111 Novelty Signal", flush=True)

    r_off = _run_single(
        seed=seed, novelty_enabled=False, novelty_bonus_weight=0.0,
        warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        self_dim=self_dim, world_dim=world_dim, lr=lr, alpha_world=alpha_world,
    )
    r_on = _run_single(
        seed=seed, novelty_enabled=True, novelty_bonus_weight=novelty_bonus_weight,
        warmup_episodes=warmup_episodes, eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        self_dim=self_dim, world_dim=world_dim, lr=lr, alpha_world=alpha_world,
    )

    ent_off = r_off["policy_entropy"]
    ent_on  = r_on["policy_entropy"]
    cells_off = r_off["novel_cell_visits"]
    cells_on  = r_on["novel_cell_visits"]
    harm_off  = r_off["harm_rate"]
    harm_on   = r_on["harm_rate"]
    novelty_ema = r_on["novelty_ema_at_eval"]

    c1_pass = ent_on >= ent_off + 0.10
    c2_pass = cells_on >= cells_off + 3
    c3_pass = harm_on <= harm_off + 0.02
    c4_pass = novelty_ema > 1e-6

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])
    status = "PASS" if all_pass else "FAIL"

    print(f"\n[EXQ-073] Results:", flush=True)
    print(f"  entropy:       OFF={ent_off:.4f}  ON={ent_on:.4f}  gap={ent_on-ent_off:+.4f}", flush=True)
    print(f"  novel_cells:   OFF={cells_off}  ON={cells_on}  gap={cells_on-cells_off:+d}", flush=True)
    print(f"  harm_rate:     OFF={harm_off:.4f}  ON={harm_on:.4f}  delta={harm_on-harm_off:+.4f}", flush=True)
    print(f"  novelty_ema:   {novelty_ema:.6f}", flush=True)
    print(f"  Status: {status} ({criteria_met}/4)", flush=True)

    if all_pass:
        interpretation = (
            "MECH-111 SUPPORTED: E1 prediction-error EMA bonus increases exploration"
            " (entropy and cell coverage up, harm not substantially increased)."
            " Novelty signal drives the agent toward less-predicted, more novel states."
        )
    elif criteria_met >= 2:
        interpretation = (
            "MECH-111 PARTIAL: Some exploratory signal present but below threshold."
            " Novelty weight or training duration may need adjustment."
        )
    else:
        interpretation = (
            "MECH-111 NOT SUPPORTED: Novelty bonus does not produce measurable"
            " increase in exploration. E1 error may be too uniform, or the EMA"
            " is not feeding back into selection effectively."
        )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(f"C1 FAIL: entropy gap={ent_on-ent_off:.4f} < 0.10")
    if not c2_pass:
        failure_notes.append(f"C2 FAIL: cell coverage gap={cells_on-cells_off} < 3")
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: harm delta={harm_on-harm_off:.4f} > 0.02")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: novelty_ema={novelty_ema:.6f} = 0 (signal absent)")
    for n in failure_notes:
        print(f"  {n}", flush=True)

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = (
        f"# V3-EXQ-073 -- MECH-111 Novelty Signal\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-111\n"
        f"**Seed:** {seed}  **Warmup:** {warmup_episodes}  **Eval:** {eval_episodes}\n"
        f"**novelty_bonus_weight:** {novelty_bonus_weight}\n\n"
        f"## Results\n\n"
        f"| Metric | NoveltyOFF | NoveltyON | Delta |\n"
        f"|---|---|---|---|\n"
        f"| policy_entropy | {ent_off:.4f} | {ent_on:.4f} | {ent_on-ent_off:+.4f} |\n"
        f"| novel_cell_visits | {cells_off} | {cells_on} | {cells_on-cells_off:+d} |\n"
        f"| harm_rate | {harm_off:.4f} | {harm_on:.4f} | {harm_on-harm_off:+.4f} |\n"
        f"| novelty_ema_at_eval | -- | {novelty_ema:.6f} | -- |\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result |\n|---|---|\n"
        f"| C1: entropy gap >= 0.10 | {'PASS' if c1_pass else 'FAIL'} |\n"
        f"| C2: cell coverage gap >= 3 | {'PASS' if c2_pass else 'FAIL'} |\n"
        f"| C3: harm delta <= 0.02 | {'PASS' if c3_pass else 'FAIL'} |\n"
        f"| C4: novelty_ema > 0 | {'PASS' if c4_pass else 'FAIL'} |\n\n"
        f"Criteria met: {criteria_met}/4 -> **{status}**\n\n"
        f"## Interpretation\n\n{interpretation}\n{failure_section}\n"
    )

    metrics = {
        "policy_entropy_off":       float(ent_off),
        "policy_entropy_on":        float(ent_on),
        "entropy_gap":              float(ent_on - ent_off),
        "novel_cell_visits_off":    float(cells_off),
        "novel_cell_visits_on":     float(cells_on),
        "cell_gap":                 float(cells_on - cells_off),
        "harm_rate_off":            float(harm_off),
        "harm_rate_on":             float(harm_on),
        "harm_delta":               float(harm_on - harm_off),
        "novelty_ema_at_eval":      float(novelty_ema),
        "mean_e1_error_off":        float(r_off["mean_e1_prediction_error"]),
        "mean_e1_error_on":         float(r_on["mean_e1_prediction_error"]),
        "novelty_bonus_weight":     float(novelty_bonus_weight),
        "crit1_pass":              1.0 if c1_pass else 0.0,
        "crit2_pass":              1.0 if c2_pass else 0.0,
        "crit3_pass":              1.0 if c3_pass else 0.0,
        "crit4_pass":              1.0 if c4_pass else 0.0,
        "criteria_met":            float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 2 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--novelty-weight", type=float, default=0.1)
    parser.add_argument("--warmup",         type=int,   default=150)
    parser.add_argument("--eval-eps",       type=int,   default=50)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        novelty_bonus_weight=args.novelty_weight,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
