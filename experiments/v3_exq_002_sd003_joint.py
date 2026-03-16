"""
V3-EXQ-002 — Full SD-003 Self-Attribution: E2+E3 Joint Pipeline

Claim: SD-003 (self-attribution via counterfactual E2) correctly identifies
agent-caused harm when implemented as the full V3 joint pipeline:

    z_world_actual = E2.world_forward(z_world, a_actual)
    z_world_cf     = E2.world_forward(z_world, a_cf)      # random counterfactual
    harm_actual    = E3.harm_eval(z_world_actual)
    harm_cf        = E3.harm_eval(z_world_cf)
    causal_sig     = harm_actual - harm_cf

The V3 pipeline separates concerns that were conflated in V2:
  - V2 (EXQ-027 FAIL): E2.predict_harm(z_gamma) — harm prediction lives in E2,
    in a unified latent that conflates self and world.
  - V3 (this experiment): E2.world_forward predicts z_world transitions (motor-sensory
    domain). E3.harm_eval evaluates harm of any z_world state (harm/goal domain).
    Together they produce calibrated causal attribution.

Two conditions:
  TRAINED:  warmup_episodes of joint gradient training, then counterfactual evaluation.
            PASS: calibration_gap = mean(causal_sig | agent_caused) - mean(causal_sig | env_caused) > 0.05
  RANDOM:   no training (control). Expected: |calibration_gap| < 0.10 (random baseline).

PASS criteria (ALL must hold):
  1. TRAINED calibration_gap > 0.05
  2. RANDOM  |calibration_gap| < 0.10
  3. TRAINED agent_caused_harm_count > 0 (agent actually caused harm — env is non-trivial)
  4. fatal_error_count == 0
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_002_sd003_joint"
CLAIM_IDS = ["SD-003"]

CONDITION_TRAINED = "TRAINED"
CONDITION_RANDOM  = "RANDOM"


def _random_cf_action(actual_action_idx: int, num_actions: int) -> int:
    """Pick a different action index uniformly at random."""
    choices = [a for a in range(num_actions) if a != actual_action_idx]
    return random.choice(choices) if choices else 0


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _train_episodes(
    agent: REEAgent,
    env: CausalGridWorld,
    optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
) -> Dict[str, float]:
    """
    Training phase: jointly train E1, E2 (self + world), and E3.harm_eval.

    Losses:
      - E1: prediction loss over latent history
      - E2 self: motor-sensory forward model (z_self domain)
      - E2 world: world_forward model — ||E2.world_forward(z_world_t, a) - z_world_{t+1}||²
      - E3 harm: supervised BCE on actual harm signals
    """
    agent.train()
    world_transition_buffer: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_label_buffer: List[Tuple[torch.Tensor, float]] = []

    total_harm = 0.0
    total_agent_harm = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev  = None
        z_world_prev = None
        action_prev  = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            ticks  = agent.clock.advance()

            if ticks["e1_tick"]:
                agent._e1_tick(latent)

            # Record world transition for E2.world_forward training
            if z_world_prev is not None and action_prev is not None:
                world_transition_buffer.append((
                    z_world_prev.detach(),
                    action_prev.detach(),
                    latent.z_world.detach(),
                ))
                if len(world_transition_buffer) > 500:
                    world_transition_buffer = world_transition_buffer[-500:]

            # Record E2 self transition
            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            # Select action
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

            z_self_prev  = latent.z_self.detach()
            z_world_prev = latent.z_world.detach()
            action_prev  = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if harm_signal < 0:
                total_harm += abs(harm_signal)
                if info["transition_type"] == "agent_caused_hazard":
                    total_agent_harm += 1
                # Record harm label for E3.harm_eval training
                harm_label_buffer.append((latent.z_world.detach(), 1.0))
                if len(harm_label_buffer) > 200:
                    harm_label_buffer = harm_label_buffer[-200:]
            else:
                # Subsample no-harm states to avoid class imbalance domination
                if step % 10 == 0:
                    harm_label_buffer.append((latent.z_world.detach(), 0.0))
                    if len(harm_label_buffer) > 200:
                        harm_label_buffer = harm_label_buffer[-200:]

            # Training update
            e1_loss = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()

            # E2 world_forward loss
            e2_world_loss = e1_loss.new_zeros(())
            if len(world_transition_buffer) >= 4:
                n = min(16, len(world_transition_buffer))
                idxs = torch.randperm(len(world_transition_buffer))[:n].tolist()
                batch = [world_transition_buffer[i] for i in idxs]
                zw_t, acts, zw_t1 = zip(*batch)
                zw_t  = torch.cat(zw_t, dim=0)
                acts  = torch.cat(acts, dim=0)
                zw_t1 = torch.cat(zw_t1, dim=0)
                zw_t1_pred = agent.e2.world_forward(zw_t, acts)
                e2_world_loss = F.mse_loss(zw_t1_pred, zw_t1)

            # E3 harm_eval supervised loss
            e3_harm_loss = e1_loss.new_zeros(())
            if len(harm_label_buffer) >= 4:
                n = min(16, len(harm_label_buffer))
                idxs = torch.randperm(len(harm_label_buffer))[:n].tolist()
                batch = [harm_label_buffer[i] for i in idxs]
                zw_batch, labels = zip(*batch)
                zw_batch = torch.cat(zw_batch, dim=0)
                labels_t = torch.tensor(labels, dtype=torch.float32, device=agent.device).unsqueeze(1)
                harm_pred = torch.nan_to_num(
                    agent.e3.harm_eval(zw_batch), nan=0.5
                ).clamp(1e-6, 1 - 1e-6)
                e3_harm_loss = F.binary_cross_entropy(harm_pred, labels_t)

            loss = e1_loss + e2_self_loss + e2_world_loss + e3_harm_loss
            if loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if done:
                break

        if (ep + 1) % 20 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep + 1}/{num_episodes}  harm={total_harm:.2f}  "
                  f"agent_caused={total_agent_harm}", flush=True)

    return {"total_harm": total_harm, "agent_caused_harm": total_agent_harm}


def _eval_condition(
    agent: REEAgent,
    env: CausalGridWorld,
    num_episodes: int,
    steps_per_episode: int,
    seed: int,
    trained: bool,
) -> Dict:
    """
    Evaluation phase: run counterfactual attribution pipeline.

    For each step, compute causal_sig = harm_actual - harm_cf via
    E2.world_forward + E3.harm_eval. Aggregate by transition_type.
    """
    agent.eval()
    torch.manual_seed(seed + 1000)

    agent_caused_sigs: List[float] = []
    env_caused_sigs:   List[float] = []
    all_sigs:          List[float] = []
    harm_events = 0
    agent_caused_harm = 0
    fatal_errors = 0

    try:
        for ep in range(num_episodes):
            flat_obs, obs_dict = env.reset()
            agent.reset()

            for step in range(steps_per_episode):
                obs_body  = obs_dict["body_state"]
                obs_world = obs_dict["world_state"]

                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    ticks  = agent.clock.advance()

                    z_world = latent.z_world  # [1, world_dim]

                    # Action selection
                    e1_prior = torch.zeros(1, agent.config.latent.world_dim, device=agent.device)
                    candidates = agent.hippocampal.propose_trajectories(
                        z_world=z_world.detach(),
                        z_self=latent.z_self.detach(),
                        num_candidates=4,
                        e1_prior=e1_prior,
                    )
                    result = agent.e3.select(candidates, temperature=1.5)
                    action = result.selected_action  # [1, action_dim]
                    agent._last_action = action

                    # SD-003 V3 counterfactual pipeline
                    actual_idx = int(action.argmax(dim=-1).item())
                    cf_idx = _random_cf_action(actual_idx, env.action_dim)
                    a_cf = _action_to_onehot(cf_idx, env.action_dim, agent.device)

                    z_world_actual = agent.e2.world_forward(z_world, action)
                    z_world_cf     = agent.e2.world_forward(z_world, a_cf)

                    harm_actual = torch.nan_to_num(agent.e3.harm_eval(z_world_actual), nan=0.5)
                    harm_cf     = torch.nan_to_num(agent.e3.harm_eval(z_world_cf),     nan=0.5)
                    causal_sig  = float((harm_actual - harm_cf).item())

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                transition_type = info.get("transition_type", "none")
                all_sigs.append(causal_sig)

                if harm_signal < 0:
                    harm_events += 1
                    if transition_type == "agent_caused_hazard":
                        agent_caused_harm += 1
                        agent_caused_sigs.append(causal_sig)
                    elif transition_type == "env_caused_hazard":
                        env_caused_sigs.append(causal_sig)
                    agent.update_residue(
                        harm_signal=harm_signal,
                        hypothesis_tag=False,
                        owned=(transition_type == "agent_caused_hazard"),
                    )
                else:
                    # Attribute some non-harm steps to env baseline
                    if transition_type == "env_caused_hazard":
                        env_caused_sigs.append(causal_sig)

                if done:
                    break

    except Exception as e:
        import traceback
        fatal_errors += 1
        print(f"  FATAL: {traceback.format_exc()}", flush=True)

    mean_agent = float(sum(agent_caused_sigs) / max(1, len(agent_caused_sigs)))
    mean_env   = float(sum(env_caused_sigs)   / max(1, len(env_caused_sigs)))
    calibration_gap = mean_agent - mean_env

    cond_label = CONDITION_TRAINED if trained else CONDITION_RANDOM
    print(f"  [{cond_label}] harm={harm_events}  agent_caused={agent_caused_harm}  "
          f"calibration_gap={calibration_gap:.4f}  "
          f"(n_agent={len(agent_caused_sigs)} n_env={len(env_caused_sigs)})", flush=True)

    return {
        "condition": cond_label,
        "calibration_gap": calibration_gap,
        "mean_causal_sig_agent": mean_agent,
        "mean_causal_sig_env": mean_env,
        "harm_events": harm_events,
        "agent_caused_harm": agent_caused_harm,
        "n_agent_caused_sigs": len(agent_caused_sigs),
        "n_env_caused_sigs": len(env_caused_sigs),
        "fatal_errors": fatal_errors,
        "mean_all_sigs": float(sum(all_sigs) / max(1, len(all_sigs))),
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 100,
    eval_episodes: int = 30,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    **kwargs,
) -> dict:
    """
    Run V3-EXQ-002.

    Returns result dict compatible with experiments/run.py.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(seed=seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
    )

    results_by_condition = {}
    fatal_errors = 0

    # ------------------------------------------------------------------ #
    # CONDITION: TRAINED                                                   #
    # ------------------------------------------------------------------ #
    print(f"\n[V3-EXQ-002] Seed {seed} Condition TRAINED", flush=True)
    agent_trained = REEAgent(config)
    opt = optim.Adam(agent_trained.parameters(), lr=lr)
    print(f"  Training for {warmup_episodes} episodes ...", flush=True)
    train_metrics = _train_episodes(agent_trained, env, opt, warmup_episodes, steps_per_episode)
    print(f"  Evaluating for {eval_episodes} episodes ...", flush=True)
    r_trained = _eval_condition(agent_trained, env, eval_episodes, steps_per_episode, seed, trained=True)
    results_by_condition[CONDITION_TRAINED] = r_trained
    fatal_errors += r_trained["fatal_errors"]

    # ------------------------------------------------------------------ #
    # CONDITION: RANDOM                                                    #
    # ------------------------------------------------------------------ #
    print(f"\n[V3-EXQ-002] Seed {seed} Condition RANDOM", flush=True)
    torch.manual_seed(seed + 5000)
    agent_random = REEAgent(config)  # fresh, untrained
    print(f"  Evaluating for {eval_episodes} episodes (no training) ...", flush=True)
    r_random = _eval_condition(agent_random, env, eval_episodes, steps_per_episode, seed, trained=False)
    results_by_condition[CONDITION_RANDOM] = r_random
    fatal_errors += r_random["fatal_errors"]

    # ------------------------------------------------------------------ #
    # PASS / FAIL decision                                                 #
    # ------------------------------------------------------------------ #
    trained_gap  = results_by_condition[CONDITION_TRAINED]["calibration_gap"]
    random_gap   = results_by_condition[CONDITION_RANDOM]["calibration_gap"]
    agent_harm_n = results_by_condition[CONDITION_TRAINED]["agent_caused_harm"]

    crit1_pass = trained_gap > 0.05
    crit2_pass = abs(random_gap) < 0.10
    crit3_pass = agent_harm_n > 0
    crit4_pass = fatal_errors == 0

    all_pass = crit1_pass and crit2_pass and crit3_pass and crit4_pass
    status = "PASS" if all_pass else "FAIL"

    failure_notes = []
    if not crit1_pass:
        failure_notes.append(f"C1 FAIL: TRAINED calibration_gap {trained_gap:.4f} <= 0.05")
    if not crit2_pass:
        failure_notes.append(f"C2 FAIL: RANDOM |calibration_gap| {abs(random_gap):.4f} >= 0.10 (spurious attribution)")
    if not crit3_pass:
        failure_notes.append("C3 FAIL: agent_caused_harm_count == 0 (agent never caused harm — env non-trivial?)")
    if not crit4_pass:
        failure_notes.append(f"C4 FAIL: fatal_errors={fatal_errors}")

    criteria_met = sum([crit1_pass, crit2_pass, crit3_pass, crit4_pass])

    print(f"\nSD-003 / V3-EXQ-002 verdict: {status}", flush=True)
    print(f"  TRAINED calibration_gap: {trained_gap:.4f}  (>0.05? {crit1_pass})", flush=True)
    print(f"  RANDOM  |calibration_gap|: {abs(random_gap):.4f}  (<0.10? {crit2_pass})", flush=True)
    print(f"  Criteria met: {criteria_met}/4", flush=True)

    metrics = {
        "fatal_error_count": float(fatal_errors),
        "trained_calibration_gap": float(trained_gap),
        "random_calibration_gap": float(random_gap),
        "trained_mean_causal_sig_agent": float(results_by_condition[CONDITION_TRAINED]["mean_causal_sig_agent"]),
        "trained_mean_causal_sig_env":   float(results_by_condition[CONDITION_TRAINED]["mean_causal_sig_env"]),
        "random_mean_causal_sig_agent":  float(results_by_condition[CONDITION_RANDOM]["mean_causal_sig_agent"]),
        "random_mean_causal_sig_env":    float(results_by_condition[CONDITION_RANDOM]["mean_causal_sig_env"]),
        "trained_agent_caused_harm": float(results_by_condition[CONDITION_TRAINED]["agent_caused_harm"]),
        "random_agent_caused_harm":  float(results_by_condition[CONDITION_RANDOM]["agent_caused_harm"]),
        "trained_n_agent_sigs": float(results_by_condition[CONDITION_TRAINED]["n_agent_caused_sigs"]),
        "trained_n_env_sigs":   float(results_by_condition[CONDITION_TRAINED]["n_env_caused_sigs"]),
        "warmup_episodes": float(warmup_episodes),
        "crit1_pass": 1.0 if crit1_pass else 0.0,
        "crit2_pass": 1.0 if crit2_pass else 0.0,
        "crit3_pass": 1.0 if crit3_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    evidence_direction = "supports" if all_pass else ("mixed" if criteria_met >= 2 else "weakens")

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-002 — SD-003 Self-Attribution: E2+E3 Joint Pipeline

**Status:** {status}
**Warmup episodes:** {warmup_episodes}
**Eval episodes:** {eval_episodes} × {steps_per_episode} steps
**Seed:** {seed}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: TRAINED calibration_gap > 0.05 | {"PASS" if crit1_pass else "FAIL"} | {trained_gap:.4f} |
| C2: RANDOM abs(calibration_gap) < 0.10 | {"PASS" if crit2_pass else "FAIL"} | {abs(random_gap):.4f} |
| C3: Agent caused harm (env non-trivial) | {"PASS" if crit3_pass else "FAIL"} | {agent_harm_n} events |
| C4: No fatal errors | {"PASS" if crit4_pass else "FAIL"} | {fatal_errors} |

## Calibration Results

| Condition | mean_causal_sig(agent) | mean_causal_sig(env) | calibration_gap |
|---|---|---|---|
| TRAINED | {results_by_condition[CONDITION_TRAINED]["mean_causal_sig_agent"]:.4f} | {results_by_condition[CONDITION_TRAINED]["mean_causal_sig_env"]:.4f} | {trained_gap:.4f} |
| RANDOM  | {results_by_condition[CONDITION_RANDOM]["mean_causal_sig_agent"]:.4f} | {results_by_condition[CONDITION_RANDOM]["mean_causal_sig_env"]:.4f} | {random_gap:.4f} |

## Attribution Pipeline

```
z_world_actual = E2.world_forward(z_world, a_actual)
z_world_cf     = E2.world_forward(z_world, a_cf)
harm_actual    = E3.harm_eval(z_world_actual)
harm_cf        = E3.harm_eval(z_world_cf)
causal_sig     = harm_actual - harm_cf
```

Criteria met: {criteria_met}/4 → SD-003 / V3-EXQ-002 verdict: **{status}**
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
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_episodes,
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
