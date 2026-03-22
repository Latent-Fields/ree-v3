#!/opt/local/bin/python3
"""
V3-EXQ-068 -- MECH-047: Mode Manager Hysteresis

Claims: MECH-047

Motivation:
  MECH-047: once an agent enters doing mode (committed), there is inertia against
  switching back. Post-commitment, the agent should take more steps to return to
  uncommitted state than a memoryless system would predict.

  This tests that commitment leaves a residual signature in E3's running_variance
  and beta_gate state: after the incentive to commit is removed, the agent doesn't
  immediately drop back to baseline but decays gradually (hysteresis).

Design:
  Two-agent comparison:
    Agent H (hysteresis probe): trained for two phases.
      Phase A: 200 high-hazard episodes to establish committed mode.
               (num_hazards=6, hazard_harm=0.05 -- forces commitment).
      Phase B: switch to low-hazard environment (num_hazards=1, hazard_harm=0.005).
               Run 100 more episodes; track how long commitment_rate stays elevated.

    Agent C (control): trained only in low-hazard environment from the start.
               100 training episodes, no prior commitment history.

  Hysteresis metric: count the number of CONSECUTIVE eval steps from the start of
  Phase B until commitment_rate (rolling 20-step window) first drops to <= baseline.
  baseline = Agent C mean commitment_rate.

Pre-registered PASS criteria (ALL must hold):
  C1: hysteresis_decay_steps > 20
      -- Agent H takes >20 steps to exit committed mode after incentive removal
  C2: beta_gate_post_commit_mean > beta_gate_control_mean * 1.1
      -- Agent H shows residual beta elevation vs control in low-hazard eval
  C3: commitment_rate_phase_a > 0.3
      -- Phase A actually established committed mode (validation gate)
  C4: No fatal errors in either agent
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


EXPERIMENT_TYPE = "v3_exq_068_mech047_mode_hysteresis"
CLAIM_IDS = ["MECH-047"]

# High-hazard environment parameters (Phase A)
ENV_HIGH_HAZARD = dict(
    size=12, num_hazards=6, num_resources=3,
    hazard_harm=0.05,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.02,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

# Low-hazard environment parameters (Phase B / control)
ENV_LOW_HAZARD = dict(
    size=12, num_hazards=1, num_resources=5,
    hazard_harm=0.005,
    env_drift_interval=5, env_drift_prob=0.05,
    proximity_harm_scale=0.01,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent(env: CausalGridWorldV2, seed: int, alpha_world: float) -> Tuple[REEAgent, optim.Adam, optim.Adam, optim.Adam]:
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
    agent = REEAgent(config)
    opt_e1 = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    opt_wf = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    opt_harm = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)
    return agent, opt_e1, opt_wf, opt_harm


def _run_episodes(
    agent: REEAgent,
    env: CausalGridWorldV2,
    opt_e1: optim.Adam,
    opt_wf: optim.Adam,
    opt_harm: optim.Adam,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    training: bool,
    label: str,
    wf_buf: Optional[List] = None,
    harm_buf_pos: Optional[List] = None,
    harm_buf_neg: Optional[List] = None,
) -> Dict:
    """
    Run episodes (training or eval). Returns per-step commitment flags and beta states.
    Buffers are passed in so phase A can hand off accumulated experience to phase B.
    """
    if training:
        agent.train()
    else:
        agent.eval()
        agent.beta_gate.reset()

    if wf_buf is None:
        wf_buf = []
    if harm_buf_pos is None:
        harm_buf_pos = []
    if harm_buf_neg is None:
        harm_buf_neg = []

    committed_flags: List[float] = []    # 1.0 per step if committed
    beta_flags: List[float] = []         # 1.0 per step if beta elevated
    fatal = 0
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
            z_world_curr: Optional[torch.Tensor] = None

            try:
                if training:
                    latent = agent.sense(obs_body, obs_world)
                else:
                    with torch.no_grad():
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

                if training:
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                else:
                    with torch.no_grad():
                        action = agent.select_action(candidates, ticks, temperature=1.0)

                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

                is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                committed_flags.append(1.0 if is_committed else 0.0)
                beta_flags.append(1.0 if agent.beta_gate.is_elevated else 0.0)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action
                committed_flags.append(0.0)
                beta_flags.append(0.0)

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if training:
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
                            opt_wf.zero_grad()
                            wf_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                list(agent.e2.world_transition.parameters()) +
                                list(agent.e2.world_action_encoder.parameters()),
                                1.0,
                            )
                            opt_wf.step()
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
                    opt_e1.zero_grad()
                    e1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                    opt_e1.step()

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
                        opt_harm.zero_grad()
                        harm_loss.backward()
                        torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                        opt_harm.step()

            z_world_prev = z_world_curr
            try:
                z_self_prev = latent.z_self.detach()
                action_prev = action.detach()
            except Exception:
                pass
            if done:
                break

        if training and ((ep + 1) % 50 == 0 or ep == num_episodes - 1):
            rv = agent.e3._running_variance
            cf = _mean_safe(committed_flags[-500:]) if committed_flags else 0.0
            print(
                f"  [{label} train] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.6f}  committed_frac={cf:.3f}",
                flush=True,
            )

    return {
        "committed_flags": committed_flags,
        "beta_flags": beta_flags,
        "fatal_errors": fatal,
        "total_harm": total_harm,
        "final_running_variance": agent.e3._running_variance,
        "wf_buf": wf_buf,
        "harm_buf_pos": harm_buf_pos,
        "harm_buf_neg": harm_buf_neg,
    }


def _hysteresis_decay_steps(
    committed_flags: List[float],
    baseline_rate: float,
    window: int = 20,
) -> int:
    """
    Count steps from start until rolling commitment_rate (window) first drops <= baseline.
    If it never drops, return len(committed_flags) as upper bound.
    """
    for i in range(len(committed_flags) - window + 1):
        window_rate = _mean_safe(committed_flags[i:i + window])
        if window_rate <= baseline_rate:
            return i + window
    return len(committed_flags)


def run(
    seed: int = 0,
    phase_a_episodes: int = 200,
    phase_b_episodes: int = 100,
    control_episodes: int = 100,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    world_dim = 32

    print(
        f"[V3-EXQ-068] MECH-047: Mode Manager Hysteresis\n"
        f"  Phase A: {phase_a_episodes} high-hazard eps (force commitment)\n"
        f"  Phase B: {phase_b_episodes} low-hazard eps (measure decay)\n"
        f"  Control: {control_episodes} low-hazard eps (no prior commitment)\n"
        f"  alpha_world={alpha_world}  seed={seed}",
        flush=True,
    )

    # -- Agent H: Phase A (high hazard, establish commitment) -----------------
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-068] Agent H -- Phase A: high-hazard training ({phase_a_episodes} eps)", flush=True)
    print('='*60, flush=True)

    env_high = CausalGridWorldV2(seed=seed, **ENV_HIGH_HAZARD)
    agent_h, opt_e1_h, opt_wf_h, opt_harm_h = _make_agent(env_high, seed, alpha_world)

    phase_a_out = _run_episodes(
        agent_h, env_high, opt_e1_h, opt_wf_h, opt_harm_h,
        phase_a_episodes, steps_per_episode, world_dim,
        training=True, label="H-phaseA",
    )
    phase_a_commit_rate = _mean_safe(phase_a_out["committed_flags"])
    print(
        f"\n  Phase A final: commit_rate={phase_a_commit_rate:.3f}"
        f"  running_var={phase_a_out['final_running_variance']:.6f}",
        flush=True,
    )

    # -- Agent H: Phase B (low hazard, measure hysteresis) --------------------
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-068] Agent H -- Phase B: low-hazard transition ({phase_b_episodes} eps)", flush=True)
    print('='*60, flush=True)

    env_low_h = CausalGridWorldV2(seed=seed + 500, **ENV_LOW_HAZARD)
    # Agent H continues from its Phase A state; hand off buffers
    phase_b_out = _run_episodes(
        agent_h, env_low_h, opt_e1_h, opt_wf_h, opt_harm_h,
        phase_b_episodes, steps_per_episode, world_dim,
        training=True, label="H-phaseB",
        wf_buf=phase_a_out["wf_buf"],
        harm_buf_pos=phase_a_out["harm_buf_pos"],
        harm_buf_neg=phase_a_out["harm_buf_neg"],
    )

    # Eval Agent H in low-hazard
    print(f"\n  Evaluating Agent H in low-hazard ({eval_episodes} eps)...", flush=True)
    eval_h_out = _run_episodes(
        agent_h, env_low_h, opt_e1_h, opt_wf_h, opt_harm_h,
        eval_episodes, steps_per_episode, world_dim,
        training=False, label="H-eval",
    )
    beta_h_mean = _mean_safe(eval_h_out["beta_flags"])
    commit_h_mean = _mean_safe(eval_h_out["committed_flags"])
    print(
        f"  Agent H eval: beta_mean={beta_h_mean:.4f}"
        f"  commit_rate={commit_h_mean:.3f}",
        flush=True,
    )

    # -- Agent C: control (low hazard only, no commitment history) ------------
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-068] Agent C -- control: low-hazard only ({control_episodes} eps)", flush=True)
    print('='*60, flush=True)

    env_low_c = CausalGridWorldV2(seed=seed + 2000, **ENV_LOW_HAZARD)
    agent_c, opt_e1_c, opt_wf_c, opt_harm_c = _make_agent(env_low_c, seed + 2000, alpha_world)

    _run_episodes(
        agent_c, env_low_c, opt_e1_c, opt_wf_c, opt_harm_c,
        control_episodes, steps_per_episode, world_dim,
        training=True, label="C-train",
    )

    # Eval Agent C in low-hazard
    print(f"\n  Evaluating Agent C in low-hazard ({eval_episodes} eps)...", flush=True)
    eval_c_out = _run_episodes(
        agent_c, env_low_c, opt_e1_c, opt_wf_c, opt_harm_c,
        eval_episodes, steps_per_episode, world_dim,
        training=False, label="C-eval",
    )
    beta_c_mean = _mean_safe(eval_c_out["beta_flags"])
    commit_c_mean = _mean_safe(eval_c_out["committed_flags"])
    print(
        f"  Agent C eval: beta_mean={beta_c_mean:.4f}"
        f"  commit_rate={commit_c_mean:.3f}",
        flush=True,
    )

    # -- Hysteresis decay measurement -----------------------------------------
    hysteresis_steps = _hysteresis_decay_steps(
        phase_b_out["committed_flags"],
        baseline_rate=commit_c_mean,
        window=20,
    )
    print(
        f"\n  hysteresis_decay_steps={hysteresis_steps}"
        f"  (steps until Phase B commitment_rate drops to control baseline={commit_c_mean:.3f})",
        flush=True,
    )

    # -- PASS / FAIL ----------------------------------------------------------
    c1_pass = hysteresis_steps > 20
    c2_pass = beta_h_mean > beta_c_mean * 1.1
    c3_pass = phase_a_commit_rate > 0.3
    c4_pass = (eval_h_out["fatal_errors"] + eval_c_out["fatal_errors"]) == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: hysteresis_decay_steps={hysteresis_steps} <= 20"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: beta_h_mean={beta_h_mean:.4f}"
            f" <= beta_c_mean*1.1={beta_c_mean * 1.1:.4f}"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: phase_a_commit_rate={phase_a_commit_rate:.3f} <= 0.3"
        )
    if not c4_pass:
        total_fatal = eval_h_out["fatal_errors"] + eval_c_out["fatal_errors"]
        failure_notes.append(f"C4 FAIL: fatal_errors={total_fatal}")

    print(f"\nV3-EXQ-068 verdict: {status}  ({criteria_met}/4)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    total_fatal = eval_h_out["fatal_errors"] + eval_c_out["fatal_errors"]
    metrics = {
        "hysteresis_decay_steps":      float(hysteresis_steps),
        "beta_gate_post_commit_mean":  float(beta_h_mean),
        "beta_gate_control_mean":      float(beta_c_mean),
        "commit_rate_post_commit":     float(commit_h_mean),
        "commit_rate_control":         float(commit_c_mean),
        "commitment_rate_phase_a":     float(phase_a_commit_rate),
        "phase_a_final_running_var":   float(phase_a_out["final_running_variance"]),
        "phase_b_final_running_var":   float(phase_b_out["final_running_variance"]),
        "fatal_error_count":           float(total_fatal),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-068 -- MECH-047: Mode Manager Hysteresis

**Status:** {status}
**Claim:** MECH-047 -- commitment mode persists after incentive removal (mode hysteresis)
**Design:** Agent H (Phase A: high-hazard -> Phase B: low-hazard) vs Agent C (low-hazard only)
**alpha_world:** {alpha_world}  |  **Phase A:** {phase_a_episodes} eps  |  **Phase B:** {phase_b_episodes} eps  |  **Seed:** {seed}

## Design Rationale

Memoryless commitment: as soon as hazards are reduced, commitment_rate should drop
immediately. Hysteresis predicts a decay lag. We measure this by counting steps until
a 20-step rolling commitment_rate window drops to the control baseline. If hysteresis
exists, this count exceeds 20.

## Results

| Metric | Value |
|--------|-------|
| Phase A commitment_rate | {phase_a_commit_rate:.3f} |
| Phase A final running_variance | {phase_a_out['final_running_variance']:.6f} |
| Phase B final running_variance | {phase_b_out['final_running_variance']:.6f} |
| Agent H beta_gate_mean (eval) | {beta_h_mean:.4f} |
| Agent C beta_gate_mean (eval) | {beta_c_mean:.4f} |
| Agent H commit_rate (eval) | {commit_h_mean:.3f} |
| Agent C commit_rate (eval) | {commit_c_mean:.3f} |
| hysteresis_decay_steps | {hysteresis_steps} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: hysteresis_decay_steps > 20 | {"PASS" if c1_pass else "FAIL"} | {hysteresis_steps} |
| C2: beta_post_commit > beta_control * 1.1 | {"PASS" if c2_pass else "FAIL"} | {beta_h_mean:.4f} vs {beta_c_mean * 1.1:.4f} |
| C3: phase_a_commit_rate > 0.3 (commitment established) | {"PASS" if c3_pass else "FAIL"} | {phase_a_commit_rate:.3f} |
| C4: No fatal errors | {"PASS" if c4_pass else "FAIL"} | {total_fatal} |

Criteria met: {criteria_met}/4 -> **{status}**
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
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--phase-a-eps",     type=int,   default=200)
    parser.add_argument("--phase-b-eps",     type=int,   default=100)
    parser.add_argument("--control-eps",     type=int,   default=100)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        phase_a_episodes=args.phase_a_eps,
        phase_b_episodes=args.phase_b_eps,
        control_episodes=args.control_eps,
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
