#!/opt/local/bin/python3
"""
V3-EXQ-065 -- MECH-029: Default Mode Probe

Claims: MECH-029

Motivation:
  MECH-029: when the agent is in uncommitted / idle state, it engages in
  internal simulation and reflective processing. This is the REE "default mode" --
  analogous to the brain's default mode network activating during low-demand,
  inwardly directed cognition.

  The behavioural proxy: during uncommitted steps, the agent should show higher
  z_world variance and higher E3 running_variance (less precision certainty) than
  during committed steps. This reflects active simulation/exploration of world
  states rather than locked-in execution.

  z_world variance captures: how much the latent world representation is varying
  across consecutive uncommitted steps -- more exploration = more variance.

  E3 running_variance captures: E3's own uncertainty estimate -- during committed
  execution, E3 has converged; during reflective/default mode, E3 is more uncertain.

Design:
  Single-condition run.
  Training: 500 episodes with mixed hazard environment.
  Eval: 100 episodes. At each step, label as committed or uncommitted (using
  agent.e3._running_variance < agent.e3.commit_threshold).

  Collect across eval:
    - z_world vectors at each step, grouped by committed/uncommitted label
    - E3 running_variance values at each step, grouped by label

  z_world_variance per group = variance of z_world tensors (mean of per-dim variances).
  e3_variance_mean per group = mean of running_variance values.

Pre-registered PASS criteria (ALL must hold):
  C1: uncommitted_z_world_var > committed_z_world_var * 1.1
      -- z_world more variable during uncommitted steps (reflective exploration)
  C2: uncommitted_e3_variance_mean > committed_e3_variance_mean * 1.05
      -- E3 less precise during uncommitted (exploratory/simulation mode)
  C3: committed_step_count >= 20 AND uncommitted_step_count >= 20
      -- both modes observed during eval
  C4: harm_pred_std > 0.01
      -- E3 not collapsed
  C5: No fatal errors
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_065_mech029_default_mode_probe"
CLAIM_IDS = ["MECH-029"]


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _tensor_variance(tensors: List[torch.Tensor]) -> float:
    """
    Compute mean per-dimension variance across a list of [1, d] tensors.
    Returns a scalar: mean_d(Var_steps(z_world[:, d])).
    """
    if len(tensors) < 2:
        return 0.0
    stacked = torch.cat(tensors, dim=0)   # [n, d]
    return float(stacked.var(dim=0).mean().item())


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Standard training: E1 + E2 world_forward + E3 harm_eval."""
    agent.train()

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    total_harm = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

                if len(wf_buf) >= 16:
                    k = min(32, len(wf_buf))
                    idxs = torch.randperm(len(wf_buf))[:k].tolist()
                    zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                    a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                    zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                    wf_pred = agent.e2.world_forward(zw_b, a_b)
                    wf_loss = F.mse_loss(wf_pred, zw1_b)
                    if wf_loss.requires_grad:
                        wf_optimizer.zero_grad()
                        wf_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            list(agent.e2.world_transition.parameters()) +
                            list(agent.e2.world_action_encoder.parameters()),
                            1.0,
                        )
                        wf_optimizer.step()
                    with torch.no_grad():
                        agent.e3.update_running_variance(
                            (wf_pred.detach() - zw1_b).detach()
                        )

            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] ep {ep+1}/{num_episodes}"
                f"  harm={total_harm}  running_var={rv:.6f}",
                flush=True,
            )

    return {
        "total_harm": total_harm,
        "final_running_variance": agent.e3._running_variance,
    }


def _eval_default_mode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Collect z_world and E3 running_variance for committed vs uncommitted steps.
    Committed = running_variance < commit_threshold (ARC-016 definition).
    """
    agent.eval()
    agent.beta_gate.reset()

    z_worlds_committed: List[torch.Tensor] = []
    z_worlds_uncommitted: List[torch.Tensor] = []
    e3_vars_committed: List[float] = []
    e3_vars_uncommitted: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            try:
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                    is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                    rv = agent.e3._running_variance
                    z_w = latent.z_world.detach().cpu()

                    harm_pred = float(agent.e3.harm_eval(latent.z_world).item())
                    all_harm_preds.append(harm_pred)

                    if is_committed:
                        z_worlds_committed.append(z_w)
                        e3_vars_committed.append(rv)
                    else:
                        z_worlds_uncommitted.append(z_w)
                        e3_vars_uncommitted.append(rv)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, _, obs_dict = env.step(action)
            if done:
                break

    committed_z_world_var = _tensor_variance(z_worlds_committed)
    uncommitted_z_world_var = _tensor_variance(z_worlds_uncommitted)
    committed_e3_var_mean = _mean_safe(e3_vars_committed)
    uncommitted_e3_var_mean = _mean_safe(e3_vars_uncommitted)
    harm_pred_std = float(
        torch.tensor(all_harm_preds).std().item()
    ) if len(all_harm_preds) > 1 else 0.0

    print(
        f"  [eval]"
        f"  committed_steps={len(z_worlds_committed)}"
        f"  uncommitted_steps={len(z_worlds_uncommitted)}"
        f"\n    committed_z_world_var={committed_z_world_var:.6f}"
        f"  uncommitted_z_world_var={uncommitted_z_world_var:.6f}"
        f"\n    committed_e3_var_mean={committed_e3_var_mean:.6f}"
        f"  uncommitted_e3_var_mean={uncommitted_e3_var_mean:.6f}"
        f"\n    harm_pred_std={harm_pred_std:.4f}",
        flush=True,
    )

    return {
        "committed_step_count": len(z_worlds_committed),
        "uncommitted_step_count": len(z_worlds_uncommitted),
        "committed_z_world_var": committed_z_world_var,
        "uncommitted_z_world_var": uncommitted_z_world_var,
        "committed_e3_variance_mean": committed_e3_var_mean,
        "uncommitted_e3_variance_mean": uncommitted_e3_var_mean,
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
    }


def run(
    seed: int = 0,
    train_episodes: int = 500,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    num_hazards: int = 3,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    world_dim = 32

    print(
        f"[V3-EXQ-065] MECH-029: Default Mode Probe\n"
        f"  train={train_episodes}  eval={eval_episodes}"
        f"  alpha_world={alpha_world}  seed={seed}"
        f"  num_hazards={num_hazards}",
        flush=True,
    )

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=num_hazards,
        num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    torch.manual_seed(seed)
    random.seed(seed)
    agent = REEAgent(config)

    print(f"\n[V3-EXQ-065] Training ({train_episodes} eps)...", flush=True)
    _train(agent, env, train_episodes, steps_per_episode, world_dim)

    print(f"\n[V3-EXQ-065] Eval -- probing default mode ({eval_episodes} eps)...", flush=True)
    eval_out = _eval_default_mode(agent, env, eval_episodes, steps_per_episode, world_dim)

    # -- PASS / FAIL ----------------------------------------------------------
    committed_var = eval_out["committed_z_world_var"]
    uncommitted_var = eval_out["uncommitted_z_world_var"]
    committed_e3_var = eval_out["committed_e3_variance_mean"]
    uncommitted_e3_var = eval_out["uncommitted_e3_variance_mean"]
    committed_count = eval_out["committed_step_count"]
    uncommitted_count = eval_out["uncommitted_step_count"]
    harm_pred_std = eval_out["harm_pred_std"]
    fatal = eval_out["fatal_errors"]

    c1_pass = uncommitted_var > committed_var * 1.1
    c2_pass = uncommitted_e3_var > committed_e3_var * 1.05
    c3_pass = committed_count >= 20 and uncommitted_count >= 20
    c4_pass = harm_pred_std > 0.01
    c5_pass = fatal == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: uncommitted_z_world_var={uncommitted_var:.6f}"
            f" <= committed_z_world_var*1.1={committed_var * 1.1:.6f}"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: uncommitted_e3_var_mean={uncommitted_e3_var:.6f}"
            f" <= committed_e3_var_mean*1.05={committed_e3_var * 1.05:.6f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: committed_steps={committed_count}"
            f"  uncommitted_steps={uncommitted_count}  (both need >= 20)"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: harm_pred_std={harm_pred_std:.4f} <= 0.01")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal}")

    print(f"\nV3-EXQ-065 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "committed_z_world_var":       float(committed_var),
        "uncommitted_z_world_var":     float(uncommitted_var),
        "z_world_var_ratio":           float(uncommitted_var / max(committed_var, 1e-8)),
        "committed_e3_variance_mean":  float(committed_e3_var),
        "uncommitted_e3_variance_mean":float(uncommitted_e3_var),
        "e3_var_ratio":                float(uncommitted_e3_var / max(committed_e3_var, 1e-8)),
        "committed_step_count":        float(committed_count),
        "uncommitted_step_count":      float(uncommitted_count),
        "harm_pred_std":               float(harm_pred_std),
        "fatal_error_count":           float(fatal),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-065 -- MECH-029: Default Mode Probe

**Status:** {status}
**Claim:** MECH-029 -- default mode (uncommitted/idle) shows higher z_world variance and E3 uncertainty than committed execution
**Design:** Single-condition; label each eval step as committed or uncommitted; compare latent variance distributions
**alpha_world:** {alpha_world}  |  **Train:** {train_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Seed:** {seed}

## Design Rationale

Committed execution is a convergent, low-variance state: E3 has selected a trajectory,
the world model is relatively stable, and E3 precision is high (running_variance low).
Uncommitted / default mode is divergent: E3 is exploring, z_world is varying across
candidate world models, and E3 remains uncertain. The ratio of uncommitted/committed
variance is the key readout.

Commitment criterion: running_variance < commit_threshold (ARC-016).

## z_world Variance by Mode

| Mode | z_world_var | E3 running_var_mean | step_count |
|------|------------|---------------------|------------|
| Committed | {committed_var:.6f} | {committed_e3_var:.6f} | {committed_count} |
| Uncommitted (default) | {uncommitted_var:.6f} | {uncommitted_e3_var:.6f} | {uncommitted_count} |

**z_world_var_ratio** (uncommitted/committed) = {uncommitted_var / max(committed_var, 1e-8):.4f}
**e3_var_ratio** (uncommitted/committed) = {uncommitted_e3_var / max(committed_e3_var, 1e-8):.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: uncommitted_z_world_var > committed * 1.1 | {"PASS" if c1_pass else "FAIL"} | {uncommitted_var:.6f} vs {committed_var * 1.1:.6f} |
| C2: uncommitted_e3_var > committed * 1.05 | {"PASS" if c2_pass else "FAIL"} | {uncommitted_e3_var:.6f} vs {committed_e3_var * 1.05:.6f} |
| C3: both modes >= 20 steps | {"PASS" if c3_pass else "FAIL"} | committed={committed_count} uncommitted={uncommitted_count} |
| C4: harm_pred_std > 0.01 (E3 not collapsed) | {"PASS" if c4_pass else "FAIL"} | {harm_pred_std:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal} |

Criteria met: {criteria_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": float(fatal),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--train-eps",    type=int,   default=500)
    parser.add_argument("--eval-eps",     type=int,   default=100)
    parser.add_argument("--steps",        type=int,   default=200)
    parser.add_argument("--alpha-world",  type=float, default=0.9)
    parser.add_argument("--num-hazards",  type=int,   default=3)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        train_episodes=args.train_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        num_hazards=args.num_hazards,
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
