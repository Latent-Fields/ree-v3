#!/opt/local/bin/python3
"""
V3-EXQ-063 -- MECH-026: Ready Vigilance Mode

Claims: MECH-026

Motivation:
  Ready Vigilance is a distinct agent state characterised by heightened
  sensitivity and priming *without* action commitment. When threats are
  present but not immediately triggering (distant hazards), the agent should
  show elevated beta_gate / E3 running_variance proxy relative to a
  threat-absent (neutral) baseline -- while remaining uncommitted.

  This tests that "vigilance" and "doing" are separable modes: threat presence
  raises preparedness (C1) without forcing constant commitment (C2).

Design:
  Two-condition within-episode comparison, matched seeds:
    Condition A: hazard_density=0.3  (threat present, distant hazards)
    Condition B: hazard_density=0.0  (neutral baseline, no hazards)

  Each condition: 500 training episodes, then 100 eval episodes.
  During eval, only *uncommitted* steps are analysed (agent not yet in doing mode).

  At each uncommitted step, record:
    - beta_gate.is_elevated  (commitment gate state proxy for vigilance)
    - agent.e3._running_variance  (E3 precision proxy: lower variance = higher confidence;
      during vigilance we expect elevated variance as E3 remains uncertain)

  readiness_gap = mean beta_gate elevation during threat uncommitted steps
                  - mean beta_gate elevation during neutral uncommitted steps

Pre-registered PASS criteria (ALL must hold):
  C1: readiness_gap > 0.005
      -- beta_gate elevated more during uncommitted threat steps vs neutral
         (vigilance elevates gate without committing)
  C2: threat_commitment_rate < neutral_commitment_rate * 1.5
      -- threat does NOT force constant commitment (ready-vigilance != doing mode)
  C3: uncommitted_threat_step_count >= 20
      -- enough threat-without-commitment steps to measure vigilance
  C4: harm_pred_std > 0.01
      -- E3 not collapsed (harm evaluator producing variance)
  C5: No fatal errors in either condition
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


EXPERIMENT_TYPE = "v3_exq_067_mech026_ready_vigilance"
CLAIM_IDS = ["MECH-026"]


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_env(seed: int, num_hazards: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=num_hazards,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _make_agent(env: CausalGridWorldV2, seed: int, alpha_world: float) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    return REEAgent(config)


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    label: str,
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
                        agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

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
                f"  [{label} train] ep {ep+1}/{num_episodes}"
                f"  harm={total_harm}  running_var={rv:.6f}",
                flush=True,
            )

    return {"total_harm": total_harm, "final_running_variance": agent.e3._running_variance}


def _eval_vigilance(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    label: str,
) -> Dict:
    """
    Eval vigilance: record beta_gate elevation and commitment rate per step.
    Focus on uncommitted steps (committed = running_variance < commit_threshold).
    """
    agent.eval()
    agent.beta_gate.reset()

    beta_uncommitted: List[float] = []   # gate elevation during uncommitted steps
    beta_committed: List[float] = []     # gate elevation during committed steps
    all_harm_preds: List[float] = []
    committed_step_count = 0
    uncommitted_step_count = 0
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
                    is_elevated = float(agent.beta_gate.is_elevated)

                    harm_pred = float(agent.e3.harm_eval(latent.z_world).item())
                    all_harm_preds.append(harm_pred)

                    if is_committed:
                        committed_step_count += 1
                        beta_committed.append(is_elevated)
                    else:
                        uncommitted_step_count += 1
                        beta_uncommitted.append(is_elevated)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, _, obs_dict = env.step(action)
            if done:
                break

    mean_beta_uncommitted = _mean_safe(beta_uncommitted)
    commitment_rate = committed_step_count / max(1, committed_step_count + uncommitted_step_count)
    harm_pred_std = float(
        torch.tensor(all_harm_preds).std().item()
    ) if len(all_harm_preds) > 1 else 0.0

    print(
        f"  [{label} eval]"
        f"  uncommitted_steps={uncommitted_step_count}"
        f"  committed_steps={committed_step_count}"
        f"  commitment_rate={commitment_rate:.3f}"
        f"  mean_beta_uncommitted={mean_beta_uncommitted:.4f}"
        f"  harm_pred_std={harm_pred_std:.4f}",
        flush=True,
    )

    return {
        "uncommitted_step_count": uncommitted_step_count,
        "committed_step_count": committed_step_count,
        "commitment_rate": commitment_rate,
        "mean_beta_uncommitted": mean_beta_uncommitted,
        "mean_beta_committed": _mean_safe(beta_committed),
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
    }


def run(
    seed: int = 0,
    train_episodes: int = 500,
    eval_episodes: int = 100,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    num_hazards_threat: int = 4,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    world_dim = 32

    print(
        f"[V3-EXQ-063] MECH-026: Ready Vigilance Mode\n"
        f"  train={train_episodes}  eval={eval_episodes}"
        f"  alpha_world={alpha_world}  seed={seed}\n"
        f"  Condition A: {num_hazards_threat} hazards (threat present)\n"
        f"  Condition B: 0 hazards (neutral baseline)",
        flush=True,
    )

    # -- Condition A: threat present (hazards) --------------------------------
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-063] Condition A -- threat present ({num_hazards_threat} hazards)", flush=True)
    print('='*60, flush=True)

    env_threat = _make_env(seed, num_hazards=num_hazards_threat)
    agent_threat = _make_agent(env_threat, seed, alpha_world)
    _train(agent_threat, env_threat, train_episodes, steps_per_episode, world_dim, "threat")
    eval_threat = _eval_vigilance(
        agent_threat, env_threat, eval_episodes, steps_per_episode, world_dim, "threat"
    )

    # -- Condition B: neutral (no hazards) ------------------------------------
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-063] Condition B -- neutral baseline (0 hazards)", flush=True)
    print('='*60, flush=True)

    env_neutral = _make_env(seed + 1000, num_hazards=0)
    agent_neutral = _make_agent(env_neutral, seed + 1000, alpha_world)
    _train(agent_neutral, env_neutral, train_episodes, steps_per_episode, world_dim, "neutral")
    eval_neutral = _eval_vigilance(
        agent_neutral, env_neutral, eval_episodes, steps_per_episode, world_dim, "neutral"
    )

    # -- Metrics & criteria ---------------------------------------------------
    readiness_gap = eval_threat["mean_beta_uncommitted"] - eval_neutral["mean_beta_uncommitted"]
    threat_commit_rate = eval_threat["commitment_rate"]
    neutral_commit_rate = eval_neutral["commitment_rate"]
    uncommitted_threat_count = eval_threat["uncommitted_step_count"]
    harm_pred_std = eval_threat["harm_pred_std"]
    total_fatal = eval_threat["fatal_errors"] + eval_neutral["fatal_errors"]

    print(
        f"\n  readiness_gap={readiness_gap:+.4f}"
        f"  threat_commit_rate={threat_commit_rate:.3f}"
        f"  neutral_commit_rate={neutral_commit_rate:.3f}"
        f"  uncommitted_threat_count={uncommitted_threat_count}"
        f"  harm_pred_std={harm_pred_std:.4f}",
        flush=True,
    )

    c1_pass = readiness_gap > 0.005
    c2_pass = threat_commit_rate < neutral_commit_rate * 1.5
    c3_pass = uncommitted_threat_count >= 20
    c4_pass = harm_pred_std > 0.01
    c5_pass = total_fatal == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: readiness_gap={readiness_gap:+.4f} <= 0.005"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: threat_commit_rate={threat_commit_rate:.3f}"
            f" >= neutral_commit_rate*1.5={neutral_commit_rate * 1.5:.3f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: uncommitted_threat_step_count={uncommitted_threat_count} < 20"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: harm_pred_std={harm_pred_std:.4f} <= 0.01")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={total_fatal}")

    print(f"\nV3-EXQ-063 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "readiness_gap":                 float(readiness_gap),
        "threat_commitment_rate":        float(threat_commit_rate),
        "neutral_commitment_rate":       float(neutral_commit_rate),
        "uncommitted_threat_step_count": float(uncommitted_threat_count),
        "uncommitted_neutral_step_count":float(eval_neutral["uncommitted_step_count"]),
        "committed_threat_step_count":   float(eval_threat["committed_step_count"]),
        "committed_neutral_step_count":  float(eval_neutral["committed_step_count"]),
        "mean_beta_uncommitted_threat":  float(eval_threat["mean_beta_uncommitted"]),
        "mean_beta_uncommitted_neutral": float(eval_neutral["mean_beta_uncommitted"]),
        "mean_beta_committed_threat":    float(eval_threat["mean_beta_committed"]),
        "mean_beta_committed_neutral":   float(eval_neutral["mean_beta_committed"]),
        "harm_pred_std":                 float(harm_pred_std),
        "fatal_error_count":             float(total_fatal),
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

    summary_markdown = f"""# V3-EXQ-063 -- MECH-026: Ready Vigilance Mode

**Status:** {status}
**Claim:** MECH-026 -- ready vigilance is a distinct mode from both neutral baseline and doing mode
**Design:** Two-condition (threat present vs neutral), matched training protocol
**alpha_world:** {alpha_world}  |  **Train:** {train_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Seed:** {seed}

## Design Rationale

Ready vigilance predicts that an agent facing distant-but-present threats shows elevated
preparedness (beta_gate elevation) *without* committing to action. If threat simply forced
constant commitment, doing mode and vigilance would be indistinguishable. C1 tests that
the gap exists; C2 tests that vigilance is distinct from doing.

## Condition Comparison

| Metric | Threat (A) | Neutral (B) |
|--------|-----------|-------------|
| mean_beta_uncommitted | {eval_threat['mean_beta_uncommitted']:.4f} | {eval_neutral['mean_beta_uncommitted']:.4f} |
| commitment_rate | {eval_threat['commitment_rate']:.3f} | {eval_neutral['commitment_rate']:.3f} |
| uncommitted_steps | {eval_threat['uncommitted_step_count']} | {eval_neutral['uncommitted_step_count']} |
| committed_steps | {eval_threat['committed_step_count']} | {eval_neutral['committed_step_count']} |
| harm_pred_std | {eval_threat['harm_pred_std']:.4f} | {eval_neutral['harm_pred_std']:.4f} |

**readiness_gap** = {readiness_gap:+.4f}  (threat.beta_uncommitted - neutral.beta_uncommitted)

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: readiness_gap > 0.005 (vigilance elevates gate) | {"PASS" if c1_pass else "FAIL"} | {readiness_gap:+.4f} |
| C2: threat_commit_rate < neutral_commit_rate * 1.5 (not forced-doing) | {"PASS" if c2_pass else "FAIL"} | {threat_commit_rate:.3f} vs {neutral_commit_rate * 1.5:.3f} |
| C3: uncommitted_threat_step_count >= 20 | {"PASS" if c3_pass else "FAIL"} | {uncommitted_threat_count} |
| C4: harm_pred_std > 0.01 (E3 not collapsed) | {"PASS" if c4_pass else "FAIL"} | {harm_pred_std:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {total_fatal} |

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
        "fatal_error_count": float(total_fatal),
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
    parser.add_argument("--num-hazards",  type=int,   default=4)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        train_episodes=args.train_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        num_hazards_threat=args.num_hazards,
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
