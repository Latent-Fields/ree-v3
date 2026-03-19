"""
V3-EXQ-050 — MECH-025: Action-Doing Mode Probe

Claims: MECH-025

Prerequisite: EXQ-030b PASS (SD-003 attribution works).

Motivation (2026-03-19):
  MECH-025: When an agent is executing a committed action sequence ("doing mode"),
  its internal state should be distinguishable from free exploration. V2 FAIL
  because SD-003 attribution and dynamic precision were not wired.

  With SD-003 validated (EXQ-030b) and ARC-016 functional (EXQ-018), the "doing
  mode" probe tests whether:
    1. Committed steps show higher causal signature than uncommitted steps.
       (During doing, the agent's action has larger causal effect on z_world.)
    2. The beta gate is elevated during committed mode (gate-level doing signature).
    3. E3 running_variance is lower during committed steps (higher confidence
       → less exploratory uncertainty → doing requires precision).

  Probes:
    causal_sig = E3(E2.world_forward(z_world, a_actual)) - E3(E2.world_forward(z_world, a_cf))
    committed_causal_sig: mean causal_sig during committed steps
    uncommitted_causal_sig: mean causal_sig during uncommitted steps
    doing_mode_delta = committed_causal_sig - uncommitted_causal_sig

PASS criteria (ALL must hold):
  C1: doing_mode_delta > 0.002    (committed steps have higher causal signature)
  C2: committed_step_count >= 10  (doing mode is actually entered)
  C3: world_forward_r2 > 0.05     (E2 world model functional for attribution)
  C4: harm_pred_std > 0.01        (E3 not collapsed)
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


EXPERIMENT_TYPE = "v3_exq_050_mech025_doing_mode_probe"
CLAIM_IDS = ["MECH-025"]


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _train(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Standard full-pipeline training to get a functional E3 + E2.world_forward."""
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    e3_tick_total = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
            else:
                action = agent._last_action
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

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

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
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  e3_ticks={e3_tick_total}",
                flush=True,
            )

    return {"total_harm": total_harm, "wf_buf": wf_buf, "e3_tick_total": e3_tick_total}


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
        a   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
        zw1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _eval_doing_mode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Probe action-doing mode by comparing causal signature during committed vs uncommitted steps.
    causal_sig = E3(E2(z_world, a_actual)) - E3(E2(z_world, a_cf))
    """
    agent.eval()
    causal_sigs_committed:   List[float] = []
    causal_sigs_uncommitted: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev:  Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        result = agent.e3.select(candidates, temperature=1.0)
                        action = result.selected_action.detach()
                        agent._last_action = action
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                is_committed = agent._committed_candidates is not None

                # Compute causal signature via SD-003 counterfactual
                with torch.no_grad():
                    z_world = latent.z_world  # [1, world_dim]
                    # Counterfactual: random different action
                    cf_idx = (
                        (random.randint(0, env.action_dim - 2) + 1 +
                         action.argmax(dim=-1).item()) % env.action_dim
                    )
                    a_cf = _action_to_onehot(int(cf_idx), env.action_dim, agent.device)

                    z_actual = agent.e2.world_forward(z_world, action)
                    z_cf     = agent.e2.world_forward(z_world, a_cf)
                    h_actual = float(agent.e3.harm_eval(z_actual).item())
                    h_cf     = float(agent.e3.harm_eval(z_cf).item())
                    causal_sig = h_actual - h_cf
                    all_harm_preds.append(h_actual)

                if is_committed:
                    causal_sigs_committed.append(causal_sig)
                else:
                    causal_sigs_uncommitted.append(causal_sig)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    mean_committed   = _mean_safe(causal_sigs_committed)
    mean_uncommitted = _mean_safe(causal_sigs_uncommitted)
    doing_mode_delta = mean_committed - mean_uncommitted
    harm_pred_std = float(
        torch.tensor(all_harm_preds).std().item()
    ) if len(all_harm_preds) > 1 else 0.0

    print(
        f"  causal_sig committed={mean_committed:.4f}  uncommitted={mean_uncommitted:.4f}"
        f"  doing_mode_delta={doing_mode_delta:+.4f}"
        f"  n_committed={len(causal_sigs_committed)}  n_uncommitted={len(causal_sigs_uncommitted)}",
        flush=True,
    )

    return {
        "mean_causal_sig_committed":   mean_committed,
        "mean_causal_sig_uncommitted": mean_uncommitted,
        "doing_mode_delta":            doing_mode_delta,
        "committed_step_count":        len(causal_sigs_committed),
        "uncommitted_step_count":      len(causal_sigs_uncommitted),
        "harm_pred_std":               harm_pred_std,
        "fatal_errors":                fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=env.action_dim,
    )
    agent = REEAgent(config)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    print(
        f"[V3-EXQ-050] MECH-025: Action-Doing Mode Probe\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}",
        flush=True,
    )

    train_out = _train(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_episodes, steps_per_episode, world_dim,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2: {world_forward_r2:.4f}", flush=True)

    print(f"\n[V3-EXQ-050] Eval — probing action-doing mode...", flush=True)
    eval_out = _eval_doing_mode(agent, env, eval_episodes, steps_per_episode, world_dim)

    # PASS / FAIL
    c1_pass = eval_out["doing_mode_delta"] > 0.002
    c2_pass = eval_out["committed_step_count"] >= 10
    c3_pass = world_forward_r2 > 0.05
    c4_pass = eval_out["harm_pred_std"] > 0.01
    c5_pass = eval_out["fatal_errors"] == 0

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: doing_mode_delta={eval_out['doing_mode_delta']:.4f} <= 0.002"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: committed_step_count={eval_out['committed_step_count']} < 10"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: world_forward_r2={world_forward_r2:.4f} <= 0.05")
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-050 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "mean_causal_sig_committed":   float(eval_out["mean_causal_sig_committed"]),
        "mean_causal_sig_uncommitted": float(eval_out["mean_causal_sig_uncommitted"]),
        "doing_mode_delta":            float(eval_out["doing_mode_delta"]),
        "committed_step_count":        float(eval_out["committed_step_count"]),
        "uncommitted_step_count":      float(eval_out["uncommitted_step_count"]),
        "harm_pred_std":               float(eval_out["harm_pred_std"]),
        "world_forward_r2":            float(world_forward_r2),
        "e3_tick_total":               float(train_out["e3_tick_total"]),
        "total_harm_train":            float(train_out["total_harm"]),
        "fatal_error_count":           float(eval_out["fatal_errors"]),
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

    summary_markdown = f"""# V3-EXQ-050 — MECH-025: Action-Doing Mode Probe

**Status:** {status}
**Claim:** MECH-025 — action-doing mode produces distinct internal signature
**Prerequisite:** EXQ-030b PASS (SD-003 attribution pipeline functional)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Motivation

MECH-025 (V2 FAIL): agent in doing mode should show a distinct internal signature.
V3 fix: SD-003 attribution now works (EXQ-030b PASS). During committed action
execution, the causal signature E3(E2(z,a_actual)) - E3(E2(z,a_cf)) should be
higher than during free exploration (uncommitted steps).

## Causal Signature by Mode

| Mode | mean causal_sig | n_steps |
|------|----------------|---------|
| Committed (doing) | {eval_out['mean_causal_sig_committed']:.4f} | {eval_out['committed_step_count']} |
| Uncommitted (exploring) | {eval_out['mean_causal_sig_uncommitted']:.4f} | {eval_out['uncommitted_step_count']} |

- **doing_mode_delta**: {eval_out['doing_mode_delta']:+.4f}  (committed - uncommitted)
- world_forward_r2: {world_forward_r2:.4f}
- harm_pred_std: {eval_out['harm_pred_std']:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: doing_mode_delta > 0.002 (committed has higher causal sig) | {"PASS" if c1_pass else "FAIL"} | {eval_out['doing_mode_delta']:+.4f} |
| C2: committed_step_count >= 10 (doing mode entered) | {"PASS" if c2_pass else "FAIL"} | {eval_out['committed_step_count']} |
| C3: world_forward_r2 > 0.05 (E2 functional) | {"PASS" if c3_pass else "FAIL"} | {world_forward_r2:.4f} |
| C4: harm_pred_std > 0.01 (E3 not collapsed) | {"PASS" if c4_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {eval_out['fatal_errors']} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": eval_out["fatal_errors"],
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=500)
    parser.add_argument("--eval-eps",    type=int,   default=50)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale",  type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
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
