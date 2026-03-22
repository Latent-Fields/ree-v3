#!/opt/local/bin/python3
"""
V3-EXQ-063 -- ARC-029: Committed vs Ablated Operating Mode Harm Outcomes

Claims: ARC-029

Context:
  ARC-029: "Committed and uncommitted operating modes produce measurably distinct
  harm outcomes." The claim predicts that the BG beta commitment gate (MECH-090)
  reduces harm in stable environments by holding the agent to a well-evaluated
  policy; and that this advantage narrows or reverses in volatile environments
  where the environment changes faster than the committed trajectory remains valid.

  This is the behavioral consequence layer split from ARC-016 into ARC-029 during
  the 2026-03-22 governance session. ARC-016 tests WHEN the gate fires (precision
  regime separation); ARC-029 tests WHETHER firing the gate helps (outcome consequence).

  Dependencies:
    MECH-090 ACTIVE (EXQ-060 PASS): BG beta gate correctly controls policy propagation.
    ARC-016 provisional: dynamic precision drives commitment threshold.

Design -- 2x2 gate x environment:
  Two matched seeds. For each seed, train one agent on the standard environment
  until variance collapses (committed state). Then evaluate 4 conditions:

    Condition 1: GateActive  x Stable  (committed agent, low-drift env)
    Condition 2: GateActive  x Volatile (committed agent, high-drift env)
    Condition 3: GateAblated x Stable  (ablated agent, low-drift env)
    Condition 4: GateAblated x Volatile (ablated agent, high-drift env)

  Gate ablation: force agent.e3._running_variance = commit_threshold + 0.1 and
  agent.e3._committed_trajectory = None before each SELECT step. This keeps the
  agent permanently in the uncommitted branch (beta always released, policy always
  propagates freshly). The underlying agent is the same -- only the commitment gate
  is disabled.

  Stable environment:  env_drift_prob=0.0 (no layout changes during eval)
  Volatile environment: env_drift_prob=0.4, env_drift_interval=3 (frequent changes)
  Training environment: standard (env_drift_prob=0.1, env_drift_interval=5)

  Outcome metric: mean_harm_signal_per_step (more positive = better; harm is negative)

PASS criteria (ALL must hold):
  C1: harm_gap_stable > 0
      (committed outperforms ablated in stable: committed_harm_stable > ablated_harm_stable)
  C2: harm_gap_volatile > harm_gap_stable
      (advantage narrows or reverses in volatile env)
  C3: Condition 1 (committed, stable) has n_committed_steps > n_uncommitted_steps
      (gate is actually active -- sanity check)
  C4: Condition 3 (ablated, stable) has n_committed_steps == 0
      (ablation is complete -- gate never elevates)
  C5: No fatal errors across all conditions
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


EXPERIMENT_TYPE = "v3_exq_063_arc029_committed_mode_harm_outcomes"
CLAIM_IDS = ["ARC-029"]

TRAIN_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

STABLE_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=50, env_drift_prob=0.0,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

VOLATILE_ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=3, env_drift_prob=0.4,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    env: CausalGridWorldV2,
) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    return REEAgent(config)


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Train agent until running_variance collapses to committed state."""
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

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

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

            (harm_buf_pos if harm_signal < 0 else harm_buf_neg).append(theta_z.detach())
            if len(harm_buf_pos) > 1000: harm_buf_pos = harm_buf_pos[-1000:]
            if len(harm_buf_neg) > 1000: harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

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
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

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
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.6f}",
                flush=True,
            )

    return {"final_running_variance": float(agent.e3._running_variance)}


def _eval_condition(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    ablated: bool,
    label: str,
    train_variance: float,
) -> Dict:
    """Eval harm outcomes under gate_active or gate_ablated condition.

    Gate ablation: before each SELECT step, force running_variance above
    commit_threshold and clear committed_trajectory. This keeps the agent
    permanently in the uncommitted branch. Policy still runs -- only the
    commitment gate is disabled.
    """
    agent.eval()

    harm_signals: List[float] = []
    n_committed = 0
    n_uncommitted = 0
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        # Restore variance to post-training level for each episode
        agent.e3._running_variance = train_variance

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            # Ablation: force uncommitted state before SELECT
            if ablated:
                agent.e3._running_variance = agent.e3.commit_threshold + 0.1
                agent.e3._committed_trajectory = None

            try:
                with torch.no_grad():
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

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                # Record commitment state (post-select, before ablation of next step)
                is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                if is_committed:
                    n_committed += 1
                else:
                    n_uncommitted += 1

                harm_signals.append(float(harm_signal))

            except Exception:
                fatal += 1
                flat_obs, obs_dict = env.reset()
                done = True

            if done:
                break

    mean_harm = _mean_safe(harm_signals)
    print(
        f"\n  [{label}]  mean_harm_per_step={mean_harm:.5f}"
        f"  n_committed={n_committed}  n_uncommitted={n_uncommitted}"
        f"  fatal={fatal}",
        flush=True,
    )

    return {
        "mean_harm_per_step": mean_harm,
        "n_committed": n_committed,
        "n_uncommitted": n_uncommitted,
        "fatal_errors": fatal,
        "n_steps": len(harm_signals),
    }


def run(
    seeds: List[int] = None,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = [0, 1]

    torch.manual_seed(seeds[0])
    random.seed(seeds[0])

    print(
        f"[V3-EXQ-063] ARC-029: Committed vs Ablated Gate -- Harm Outcomes (2x2)\n"
        f"  2x2: [gate_active / gate_ablated] x [stable / volatile env]\n"
        f"  Gate ablation: force uncommitted state (running_var > commit_threshold)\n"
        f"  Stable: drift_prob=0.0 | Volatile: drift_prob=0.4, interval=3\n"
        f"  Training: standard env (drift_prob=0.1)  warmup={warmup_episodes} eps\n"
        f"  seeds={seeds}  alpha_world={alpha_world}",
        flush=True,
    )

    # Results aggregated across seeds
    results_by_seed = []

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-063] Seed {seed}", flush=True)
        print('='*60, flush=True)

        # Train on standard env
        train_env = CausalGridWorldV2(seed=seed, **TRAIN_ENV_KWARGS)
        agent = _make_agent(seed, self_dim, world_dim, alpha_world, train_env)
        train_out = _train_agent(agent, train_env, warmup_episodes, steps_per_episode, world_dim)
        train_variance = train_out["final_running_variance"]

        print(
            f"\n  Post-train running_variance={train_variance:.6f}"
            f"  commit_threshold={agent.e3.commit_threshold:.4f}"
            f"  committed={train_variance < agent.e3.commit_threshold}",
            flush=True,
        )

        if train_variance >= agent.e3.commit_threshold:
            print(
                "  WARNING: agent not collapsed to committed state. "
                "Gate-active / ablated comparison may not be meaningful.",
                flush=True,
            )

        # Build eval envs (new seeds to avoid overlap with training)
        stable_env = CausalGridWorldV2(seed=seed + 500, **STABLE_ENV_KWARGS)
        volatile_env = CausalGridWorldV2(seed=seed + 500, **VOLATILE_ENV_KWARGS)

        seed_results = {}
        for env_name, env_obj in [("stable", stable_env), ("volatile", volatile_env)]:
            for gate_label, ablated in [("gate_active", False), ("gate_ablated", True)]:
                label = f"seed{seed}_{env_name}_{gate_label}"
                print(f"\n{'='*60}", flush=True)
                print(f"[V3-EXQ-063] {label}", flush=True)
                print('='*60, flush=True)
                seed_results[f"{env_name}_{gate_label}"] = _eval_condition(
                    agent=agent,
                    env=env_obj,
                    num_episodes=eval_episodes,
                    steps_per_episode=steps_per_episode,
                    world_dim=world_dim,
                    ablated=ablated,
                    label=label,
                    train_variance=train_variance,
                )

        results_by_seed.append(seed_results)

    # Average metrics across seeds
    def _avg_metric(key: str, subkey: str) -> float:
        vals = [r[key][subkey] for r in results_by_seed]
        return float(sum(vals) / len(vals))

    harm_active_stable   = _avg_metric("stable_gate_active",   "mean_harm_per_step")
    harm_ablated_stable  = _avg_metric("stable_gate_ablated",  "mean_harm_per_step")
    harm_active_volatile = _avg_metric("volatile_gate_active", "mean_harm_per_step")
    harm_ablated_volatile= _avg_metric("volatile_gate_ablated","mean_harm_per_step")

    # harm_gap = committed_harm - ablated_harm
    # Positive gap = ablated is worse (committed helps)
    # Negative gap = ablated is better (committed hurts)
    harm_gap_stable   = harm_active_stable   - harm_ablated_stable
    harm_gap_volatile = harm_active_volatile - harm_ablated_volatile

    # Sanity: gate actually active in committed condition
    n_committed_active_stable = _avg_metric("stable_gate_active",  "n_committed")
    n_uncommitted_active_stable = _avg_metric("stable_gate_active", "n_uncommitted")
    n_committed_ablated_stable  = _avg_metric("stable_gate_ablated","n_committed")

    total_fatal = sum(
        sum(r[k]["fatal_errors"] for k in r)
        for r in results_by_seed
    )

    # -- PASS / FAIL ---------------------------------------------------------
    # C1: committed outperforms ablated in stable (harm_gap_stable > 0)
    c1_pass = harm_gap_stable > 0.0
    # C2: advantage narrows in volatile (gap_volatile < gap_stable)
    c2_pass = harm_gap_volatile < harm_gap_stable
    # C3: gate is active in committed-stable condition
    c3_pass = n_committed_active_stable > n_uncommitted_active_stable
    # C4: ablation is complete (no committed steps when ablated)
    c4_pass = n_committed_ablated_stable == 0
    # C5: no fatal errors
    c5_pass = total_fatal == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: harm_gap_stable={harm_gap_stable:.5f} <= 0 "
            f"(committed did not outperform ablated in stable env; "
            f"active={harm_active_stable:.5f} vs ablated={harm_ablated_stable:.5f})"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: harm_gap_volatile={harm_gap_volatile:.5f} >= harm_gap_stable={harm_gap_stable:.5f} "
            f"(advantage did not narrow in volatile env; "
            f"volatile_active={harm_active_volatile:.5f} vs volatile_ablated={harm_ablated_volatile:.5f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: n_committed_active_stable={n_committed_active_stable:.0f} "
            f"<= n_uncommitted={n_uncommitted_active_stable:.0f} "
            f"(gate not actually elevating in committed condition)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_committed_ablated_stable={n_committed_ablated_stable:.0f} != 0 "
            f"(ablation incomplete -- committed steps still occurring)"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={total_fatal:.0f}")

    print(f"\nV3-EXQ-063 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    print(f"\n  Summary:", flush=True)
    print(f"    harm_active_stable={harm_active_stable:.5f}  harm_ablated_stable={harm_ablated_stable:.5f}", flush=True)
    print(f"    harm_gap_stable={harm_gap_stable:.5f}  harm_gap_volatile={harm_gap_volatile:.5f}", flush=True)
    print(f"    harm_active_volatile={harm_active_volatile:.5f}  harm_ablated_volatile={harm_ablated_volatile:.5f}", flush=True)

    metrics = {
        "harm_active_stable":        float(harm_active_stable),
        "harm_ablated_stable":       float(harm_ablated_stable),
        "harm_gap_stable":           float(harm_gap_stable),
        "harm_active_volatile":      float(harm_active_volatile),
        "harm_ablated_volatile":     float(harm_ablated_volatile),
        "harm_gap_volatile":         float(harm_gap_volatile),
        "n_committed_active_stable": float(n_committed_active_stable),
        "n_uncommitted_active_stable":float(n_uncommitted_active_stable),
        "n_committed_ablated_stable":float(n_committed_ablated_stable),
        "gap_reduction_ratio": float(
            harm_gap_volatile / max(abs(harm_gap_stable), 1e-6)
            if harm_gap_stable != 0 else 0.0
        ),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
        "fatal_error_count": float(total_fatal),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-063 -- ARC-029: Committed vs Ablated Mode Harm Outcomes (2x2)

**Status:** {status}
**Claims:** ARC-029
**Design:** 2x2 [gate_active / gate_ablated] x [stable (drift=0.0) / volatile (drift=0.4, interval=3)]
**Seeds:** {seeds}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps

## Design Rationale

ARC-029 predicts that the BG beta commitment gate (MECH-090) reduces harm in stable
environments by holding the agent to a well-evaluated trajectory, and that this
advantage narrows or reverses in volatile environments where the committed trajectory
quickly becomes invalid as the layout changes.

Gate ablation: force `agent.e3._running_variance = commit_threshold + 0.1` and
`agent.e3._committed_trajectory = None` before each SELECT step. This keeps the
agent permanently uncommitted -- policy re-evaluates freshly every step. Same trained
weights; only the commitment gate is disabled.

harm_gap = harm_gate_active - harm_gate_ablated (positive = committed is better)

## Results

| Condition | Mean Harm/Step |
|---|---|
| Gate Active, Stable | {harm_active_stable:.5f} |
| Gate Ablated, Stable | {harm_ablated_stable:.5f} |
| Gate Active, Volatile | {harm_active_volatile:.5f} |
| Gate Ablated, Volatile | {harm_ablated_volatile:.5f} |

| Gap | Value |
|---|---|
| harm_gap_stable (committed advantage) | {harm_gap_stable:.5f} |
| harm_gap_volatile (committed advantage) | {harm_gap_volatile:.5f} |
| gap_reduction_ratio (volatile/stable) | {metrics['gap_reduction_ratio']:.3f} |

## Commitment Gate Sanity

| Metric | Value |
|---|---|
| n_committed_active_stable (avg) | {n_committed_active_stable:.0f} |
| n_uncommitted_active_stable (avg) | {n_uncommitted_active_stable:.0f} |
| n_committed_ablated_stable (avg) | {n_committed_ablated_stable:.0f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: harm_gap_stable > 0 (committed better in stable) | {"PASS" if c1_pass else "FAIL"} | {harm_gap_stable:.5f} |
| C2: harm_gap_volatile < harm_gap_stable (narrows in volatile) | {"PASS" if c2_pass else "FAIL"} | {harm_gap_volatile:.5f} vs {harm_gap_stable:.5f} |
| C3: committed cond has more committed than uncommitted steps | {"PASS" if c3_pass else "FAIL"} | {n_committed_active_stable:.0f} vs {n_uncommitted_active_stable:.0f} |
| C4: ablated cond has zero committed steps | {"PASS" if c4_pass else "FAIL"} | {n_committed_ablated_stable:.0f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {total_fatal:.0f} |

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
    parser.add_argument("--seeds",        type=int, nargs="+", default=[0, 1])
    parser.add_argument("--warmup",       type=int,   default=400)
    parser.add_argument("--eval-eps",     type=int,   default=50)
    parser.add_argument("--steps",        type=int,   default=200)
    parser.add_argument("--alpha-world",  type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seeds=args.seeds,
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
        if isinstance(v, float):
            print(f"  {k}: {v:.5f}", flush=True)
