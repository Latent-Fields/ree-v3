"""
V3-EXQ-001 — SD-005 z_self/z_world Channel Separation Validation

Claim: SD-005 (z_self/z_world split) produces functionally distinct latent channels
after minimal training. Validates the substrate wiring before any claim experiments.

Two-phase design:
  TRAIN: gradient updates on E1 prediction loss + E2 motor-sensory loss for
         num_train_episodes episodes. Encoder learns to separate body/world dynamics.
  EVAL:  run num_episodes without gradients and measure channel selectivity.

PASS criteria (ALL must hold):
  C1: z_self/z_world mean absolute correlation < 0.8  (channels not collapsed)
  C2: world_selectivity_margin > 0.0  (z_world tracks world_delta better than z_self,
      after training — positive margin is achievable once encoder has gradient signal)
  C3: fatal_error_count == 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_001_z_separation"
CLAIM_IDS = ["SD-005"]


def _pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson r between two 1-D tensors."""
    if a.shape[0] < 2:
        return 0.0
    a_z = (a - a.mean()) / (a.std() + 1e-8)
    b_z = (b - b.mean()) / (b.std() + 1e-8)
    return float((a_z * b_z).mean().item())


def _train_episode(
    agent: REEAgent,
    env: CausalGridWorld,
    optimizer: optim.Optimizer,
    steps_per_episode: int,
) -> None:
    """Run one training episode: E1 + E2 motor-sensory gradient updates."""
    agent.train()
    flat_obs, obs_dict = env.reset()
    agent.reset()
    z_self_prev = None

    for step in range(steps_per_episode):
        obs_body  = obs_dict["body_state"]
        obs_world = obs_dict["world_state"]

        latent = agent.sense(obs_body, obs_world)
        ticks  = agent.clock.advance()

        if ticks["e1_tick"]:
            agent._e1_tick(latent)

        # Record transition for E2
        if z_self_prev is not None:
            last_a = agent._last_action if agent._last_action is not None else torch.zeros(1, env.action_dim)
            agent.record_transition(z_self_prev, last_a, latent.z_self.detach())
        z_self_prev = latent.z_self.detach()

        # Select action (no gradient needed here)
        with torch.no_grad():
            e1_prior = torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
            candidates = agent.hippocampal.propose_trajectories(
                z_world=latent.z_world.detach(),
                z_self=latent.z_self.detach(),
                num_candidates=4,
                e1_prior=e1_prior,
            )
            result = agent.e3.select(candidates, temperature=1.5)
            action = result.selected_action
            agent._last_action = action

        flat_obs, harm_signal, done, info, obs_dict = env.step(action)

        # Training loss
        e1_loss = agent.compute_prediction_loss()
        e2_loss = agent.compute_e2_loss()
        loss = e1_loss + e2_loss

        if loss.requires_grad:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        if done:
            break


def run(
    seed: int = 0,
    num_train_episodes: int = 10,
    num_episodes: int = 20,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    """
    Run V3-EXQ-001.

    Returns result dict compatible with experiments/run.py.
    """
    torch.manual_seed(seed)

    env = CausalGridWorld(seed=seed)

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )
    agent = REEAgent(config)

    # ------------------------------------------------------------------ #
    # Phase 1: Training                                                    #
    # ------------------------------------------------------------------ #
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    print(f"[V3-EXQ-001] Training phase: {num_train_episodes} episodes ...", flush=True)
    for ep in range(num_train_episodes):
        _train_episode(agent, env, optimizer, steps_per_episode)
        if (ep + 1) % 5 == 0:
            print(f"  Train ep {ep + 1}/{num_train_episodes}", flush=True)

    # ------------------------------------------------------------------ #
    # Phase 2: Evaluation                                                  #
    # ------------------------------------------------------------------ #
    agent.eval()

    z_self_records:  List[torch.Tensor] = []
    z_world_records: List[torch.Tensor] = []
    body_delta_records:  List[float] = []
    world_delta_records: List[float] = []
    harm_events = 0
    agent_caused_harm = 0
    survival_steps_all: List[int] = []
    fatal_errors = 0

    prev_body_state = None
    prev_world_state = None

    try:
        for episode in range(num_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()
            prev_body_state = obs_dict["body_state"].clone()
            prev_world_state = obs_dict["world_state"].clone()

            for step in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    ticks  = agent.clock.advance()

                    z_self  = latent.z_self.detach().squeeze(0)
                    z_world = latent.z_world.detach().squeeze(0)
                    z_self_records.append(z_self)
                    z_world_records.append(z_world)

                    body_delta  = float((obs_body  - prev_body_state).abs().mean().item())
                    world_delta = float((obs_world - prev_world_state).abs().mean().item())
                    body_delta_records.append(body_delta)
                    world_delta_records.append(world_delta)
                    prev_body_state  = obs_body.clone()
                    prev_world_state = obs_world.clone()

                    e1_prior = torch.zeros(1, world_dim)
                    candidates = agent.hippocampal.propose_trajectories(
                        z_world=z_world.unsqueeze(0),
                        z_self=z_self.unsqueeze(0),
                        num_candidates=4,
                        e1_prior=e1_prior,
                    )
                    result = agent.e3.select(candidates, temperature=1.5)
                    action = result.selected_action

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                if harm_signal < 0:
                    harm_events += 1
                    if info["transition_type"] == "agent_caused_hazard":
                        agent_caused_harm += 1
                    agent.update_residue(
                        harm_signal=harm_signal,
                        hypothesis_tag=False,
                        owned=(info["transition_type"] == "agent_caused_hazard"),
                    )

                if done:
                    break

            survival_steps_all.append(env.steps)

    except Exception as e:
        import traceback
        fatal_errors += 1
        tb = traceback.format_exc()
        return {
            "status": "FAIL",
            "metrics": {"fatal_error_count": float(fatal_errors)},
            "summary_markdown": f"# FAIL\n\nFatal exception:\n```\n{tb}\n```",
            "claim_ids": CLAIM_IDS,
            "evidence_direction": "unknown",
            "experiment_type": EXPERIMENT_TYPE,
        }

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    if len(z_self_records) < 10:
        return {
            "status": "FAIL",
            "metrics": {"fatal_error_count": 0.0, "record_count": float(len(z_self_records))},
            "summary_markdown": "# FAIL\n\nInsufficient records collected.",
            "claim_ids": CLAIM_IDS,
            "evidence_direction": "weakens",
            "experiment_type": EXPERIMENT_TYPE,
        }

    z_self_mat  = torch.stack(z_self_records)
    z_world_mat = torch.stack(z_world_records)

    z_self_mean_per_step  = z_self_mat.mean(dim=-1)
    z_world_mean_per_step = z_world_mat.mean(dim=-1)
    self_world_corr = abs(_pearson_correlation(z_self_mean_per_step, z_world_mean_per_step))

    body_delta_t  = torch.tensor(body_delta_records,  dtype=torch.float32)
    world_delta_t = torch.tensor(world_delta_records, dtype=torch.float32)

    z_self_world_delta_corr  = abs(_pearson_correlation(z_self_mean_per_step, world_delta_t))
    z_world_world_delta_corr = abs(_pearson_correlation(z_world_mean_per_step, world_delta_t))
    z_self_body_delta_corr   = abs(_pearson_correlation(z_self_mean_per_step, body_delta_t))
    z_world_body_delta_corr  = abs(_pearson_correlation(z_world_mean_per_step, body_delta_t))

    world_selectivity_margin = z_world_world_delta_corr - z_self_world_delta_corr
    body_selectivity_margin  = z_self_body_delta_corr   - z_world_body_delta_corr

    mean_survival = float(sum(survival_steps_all) / max(1, len(survival_steps_all)))
    residue_stats = agent.residue_field.get_statistics()

    # ------------------------------------------------------------------ #
    # PASS / FAIL decision                                                 #
    # ------------------------------------------------------------------ #
    # C1: channels not collapsed
    crit1_pass = self_world_corr < 0.8
    # C2: after training, z_world must be positively more selective for world_delta
    crit2_pass = world_selectivity_margin > 0.0
    # C3: no fatal errors
    crit3_pass = fatal_errors == 0

    all_pass = crit1_pass and crit2_pass and crit3_pass
    status = "PASS" if all_pass else "FAIL"

    failure_notes = []
    if not crit1_pass:
        failure_notes.append(
            f"C1 FAIL: z_self/z_world corr {self_world_corr:.3f} >= 0.8 (channels collapsing)"
        )
    if not crit2_pass:
        failure_notes.append(
            f"C2 FAIL: world_selectivity_margin {world_selectivity_margin:.3f} <= 0.0 "
            f"after {num_train_episodes} training episodes (z_world not selective for world_delta)"
        )
    if not crit3_pass:
        failure_notes.append(f"C3 FAIL: fatal_errors={fatal_errors}")

    metrics = {
        "fatal_error_count": float(fatal_errors),
        "self_world_correlation": float(self_world_corr),
        "world_selectivity_margin": float(world_selectivity_margin),
        "body_selectivity_margin": float(body_selectivity_margin),
        "z_world_world_delta_corr": float(z_world_world_delta_corr),
        "z_self_world_delta_corr": float(z_self_world_delta_corr),
        "z_self_body_delta_corr": float(z_self_body_delta_corr),
        "z_world_body_delta_corr": float(z_world_body_delta_corr),
        "mean_survival_steps": mean_survival,
        "harm_events": float(harm_events),
        "agent_caused_harm": float(agent_caused_harm),
        "total_residue": float(residue_stats["total_residue"].item()),
        "num_harm_events_residue": float(residue_stats["num_harm_events"].item()),
        "record_count": float(len(z_self_records)),
        "num_train_episodes": float(num_train_episodes),
        "crit1_pass": 1.0 if crit1_pass else 0.0,
        "crit2_pass": 1.0 if crit2_pass else 0.0,
    }

    evidence_direction = "supports" if all_pass else ("mixed" if (crit1_pass or crit2_pass) else "weakens")

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    print(f"\nV3-EXQ-001 verdict: {status}", flush=True)
    print(f"  C1 self/world corr: {self_world_corr:.4f} ({'PASS' if crit1_pass else 'FAIL'})", flush=True)
    print(f"  C2 world_selectivity_margin: {world_selectivity_margin:.4f} ({'PASS' if crit2_pass else 'FAIL'})", flush=True)
    print(f"  Train eps: {num_train_episodes}  Eval eps: {num_episodes}", flush=True)

    summary_markdown = f"""# V3-EXQ-001 — SD-005 z_self/z_world Channel Separation

**Status:** {status}
**Training episodes:** {num_train_episodes}
**Eval episodes:** {num_episodes} x {steps_per_episode} steps
**Seed:** {seed}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: z_self/z_world corr < 0.8 | {"PASS" if crit1_pass else "FAIL"} | {self_world_corr:.4f} |
| C2: world_selectivity_margin > 0.0 (trained) | {"PASS" if crit2_pass else "FAIL"} | {world_selectivity_margin:.4f} |
| C3: No fatal errors | {"PASS" if crit3_pass else "FAIL"} | {fatal_errors} |

## Selectivity Metrics

- z_world ↔ world_delta correlation: {z_world_world_delta_corr:.4f}
- z_self  ↔ world_delta correlation: {z_self_world_delta_corr:.4f}
- z_self  ↔ body_delta correlation:  {z_self_body_delta_corr:.4f}
- z_world ↔ body_delta correlation:  {z_world_body_delta_corr:.4f}

## Environment Metrics

- Mean survival: {mean_survival:.1f} steps
- Harm events: {harm_events} (agent-caused: {agent_caused_harm})
- Total residue: {residue_stats["total_residue"].item():.4f}
- Records: {len(z_self_records)}
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        num_train_episodes=args.train_episodes,
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0] if CLAIM_IDS else EXPERIMENT_TYPE
    result["verdict"] = result["status"]

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Result written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
