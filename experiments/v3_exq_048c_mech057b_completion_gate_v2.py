"""
V3-EXQ-048c — MECH-057b: Trajectory Completion Gate (running_variance fix)

Claims: MECH-057b, MECH-090

Root cause of EXQ-048b FAIL:
  EXQ-048b fixed the select_action() bypass (gate wiring correct) but still
  never called agent.e3.update_after_execution(). Without this call, E3's
  _running_variance stays at precision_init=0.5 forever. Since
  commit_threshold=0.40, the condition `_running_variance < commit_threshold`
  (0.5 < 0.40) is never satisfied → agent never commits → beta_gate.elevate()
  never fires → C1/C2/C3 all FAIL regardless of gate wiring.

Fix:
  At the start of each step, after encoding the new observation with
  agent.sense(), call:
    agent.e3.update_after_execution(latent.z_world.detach(), harm_prev < 0)
  This feeds the actual z_world (observed outcome of the previous action) back
  into E3, updating _running_variance via EMA. Once training lowers variance
  below 0.40, the agent commits, elevates the gate, and MECH-057b can be tested.

  Sequence:
    step t:   encode obs_t → z_world_t
              call update_after_execution(z_world_t, harm_{t-1} < 0)   ← fix
              select action_t via select_action()
              env.step(action_t) → harm_t, obs_{t+1}

  z_world_t is the "actual outcome" of action_{t-1}, compared against
  _committed_trajectory.world_states[1] (E3's one-step prediction).

Prerequisite: EXQ-042 PASS (HippocampalModule terrain prior functional).
Prior: EXQ-048 FAIL (bypassed select_action), EXQ-048b FAIL (update_after_execution missing).

Motivation:
  MECH-057b: BetaGate fires at trajectory COMPLETION, not initiation.
  During committed sequence:
    - BetaGate elevated (is_elevated = True)
    - Policy output blocked (propagate() returns None)
    - E3 updates internally
  At completion (E3 tick boundary → new trajectory selected):
    - BetaGate released (is_elevated = False)
    - Policy propagates

PASS criteria (ALL must hold):
  C1: committed_step_count >= 10        (agent enters committed mode at all)
  C2: hold_rate_during_committed > 0.5  (gate mostly holds during committed steps)
  C3: propagation_count > 0             (gate does release at some completions)
  C4: calibration_gap_approach > 0.0    (E3 still making meaningful selections)
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


EXPERIMENT_TYPE = "v3_exq_048c_mech057b_completion_gate_v2"
CLAIM_IDS = ["MECH-057b", "MECH-090"]

APPROACH_TTYPES = {"hazard_approach"}


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
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    e3_tick_total = 0
    committed_fraction_log: List[float] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None
        harm_prev: float = 0.0  # FIX: track harm from previous step

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            # FIX: update E3 running variance using z_world from this step
            # (= actual outcome of the previous action) so _running_variance
            # evolves and the agent can eventually commit.
            if step > 0:
                agent.e3.update_after_execution(
                    latent.z_world.detach(), harm_prev < 0
                )

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

            if ticks.get("e3_tick", False):
                e3_tick_total += 1

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Track commitment fraction
            committed_fraction_log.append(
                1.0 if agent.e3._committed_trajectory is not None else 0.0
            )

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

            # E1 loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            # World-forward loss
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

            # Harm eval (balanced)
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
            harm_prev    = float(harm_signal)
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            ct = agent.e3.commit_threshold
            cf = _mean_safe(committed_fraction_log[-1000:])
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  e3_ticks={e3_tick_total}"
                f"  running_var={rv:.4f}  commit_thresh={ct:.3f}"
                f"  committed_frac={cf:.3f}",
                flush=True,
            )

    return {
        "total_harm": total_harm,
        "e3_tick_total": e3_tick_total,
        "final_running_variance": agent.e3._running_variance,
        "mean_committed_fraction": _mean_safe(committed_fraction_log),
    }


def _eval_beta_gate(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Track beta gate state relative to commitment status step-by-step."""
    agent.eval()

    committed_steps = 0
    uncommitted_steps = 0
    committed_and_elevated = 0
    committed_and_propagated = 0
    gate_release_events = 0
    approach_scores: List[float] = []
    none_scores: List[float] = []
    fatal = 0
    prev_elevated = False
    running_variances: List[float] = []

    agent.beta_gate.reset()

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        prev_elevated = False
        harm_prev: float = 0.0

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            # FIX: update running variance from actual z_world
            if step > 0:
                agent.e3.update_after_execution(
                    latent.z_world.detach(), harm_prev < 0
                )

            running_variances.append(agent.e3._running_variance)

            with torch.no_grad():
                ticks  = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                theta_z = agent.theta_buffer.summary()

            try:
                with torch.no_grad():
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                is_committed = agent.e3._committed_trajectory is not None
                is_elevated  = agent.beta_gate.is_elevated

                if is_committed:
                    committed_steps += 1
                    if is_elevated:
                        committed_and_elevated += 1
                    else:
                        committed_and_propagated += 1
                else:
                    uncommitted_steps += 1

                if prev_elevated and not is_elevated:
                    gate_release_events += 1
                prev_elevated = is_elevated

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(theta_z).item())
                if ttype in APPROACH_TTYPES:
                    approach_scores.append(score)
                elif ttype == "none":
                    none_scores.append(score)
            except Exception:
                pass

            harm_prev = float(harm_signal)
            if done:
                break

    beta_state = agent.beta_gate.get_state()
    hold_count  = beta_state["hold_count"]
    prop_count  = beta_state["propagation_count"]

    hold_rate_during_committed = (
        committed_and_elevated / max(1, committed_steps)
    )
    cal_gap = _mean_safe(approach_scores) - _mean_safe(none_scores)
    mean_rv = _mean_safe(running_variances)

    print(
        f"  committed_steps={committed_steps}  uncommitted={uncommitted_steps}\n"
        f"  hold_rate_during_committed={hold_rate_during_committed:.3f}"
        f"  gate_releases={gate_release_events}\n"
        f"  hold_count={hold_count}  propagation_count={prop_count}\n"
        f"  cal_gap_approach={cal_gap:.4f}"
        f"  mean_running_variance={mean_rv:.4f}",
        flush=True,
    )

    return {
        "committed_step_count":          committed_steps,
        "uncommitted_step_count":        uncommitted_steps,
        "committed_and_elevated":        committed_and_elevated,
        "committed_and_propagated":      committed_and_propagated,
        "hold_rate_during_committed":    hold_rate_during_committed,
        "gate_release_events":           gate_release_events,
        "hold_count_total":              hold_count,
        "propagation_count_total":       prop_count,
        "calibration_gap_approach":      cal_gap,
        "mean_running_variance":         mean_rv,
        "fatal_errors":                  fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
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
        f"[V3-EXQ-048c] MECH-057b: Trajectory Completion Gate (running_variance fix)\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}\n"
        f"  precision_init={agent.e3._running_variance:.3f}"
        f"  commit_threshold={agent.e3.commit_threshold:.3f}",
        flush=True,
    )

    train_out = _train(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_episodes, steps_per_episode, world_dim,
    )

    print(
        f"\n[V3-EXQ-048c] Post-train:"
        f"  running_var={train_out['final_running_variance']:.4f}"
        f"  committed_frac={train_out['mean_committed_fraction']:.3f}",
        flush=True,
    )
    print(f"\n[V3-EXQ-048c] Eval -- tracking beta gate state...", flush=True)
    eval_out = _eval_beta_gate(agent, env, eval_episodes, steps_per_episode, world_dim)

    # PASS / FAIL
    c1_pass = eval_out["committed_step_count"] >= 10
    c2_pass = eval_out["hold_rate_during_committed"] > 0.5
    c3_pass = eval_out["propagation_count_total"] > 0
    c4_pass = eval_out["calibration_gap_approach"] > 0.0
    c5_pass = eval_out["fatal_errors"] == 0

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: committed_step_count={eval_out['committed_step_count']} < 10"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: hold_rate_during_committed={eval_out['hold_rate_during_committed']:.3f} <= 0.5"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: propagation_count_total={eval_out['propagation_count_total']} == 0"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: calibration_gap_approach={eval_out['calibration_gap_approach']:.4f} <= 0"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-048c verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "committed_step_count":         float(eval_out["committed_step_count"]),
        "uncommitted_step_count":       float(eval_out["uncommitted_step_count"]),
        "committed_and_elevated":       float(eval_out["committed_and_elevated"]),
        "committed_and_propagated":     float(eval_out["committed_and_propagated"]),
        "hold_rate_during_committed":   float(eval_out["hold_rate_during_committed"]),
        "gate_release_events":          float(eval_out["gate_release_events"]),
        "hold_count_total":             float(eval_out["hold_count_total"]),
        "propagation_count_total":      float(eval_out["propagation_count_total"]),
        "calibration_gap_approach":     float(eval_out["calibration_gap_approach"]),
        "mean_running_variance":        float(eval_out["mean_running_variance"]),
        "final_running_variance":       float(train_out["final_running_variance"]),
        "mean_committed_fraction_train": float(train_out["mean_committed_fraction"]),
        "fatal_error_count":            float(eval_out["fatal_errors"]),
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

    summary_markdown = f"""# V3-EXQ-048c -- MECH-057b: Trajectory Completion Gate (running_variance fix)

**Status:** {status}
**Claims:** MECH-057b, MECH-090
**Fix:** Adds agent.e3.update_after_execution() call each step so _running_variance evolves
**Prior attempts:** EXQ-048 (bypassed select_action), EXQ-048b (missing update_after_execution)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Root Cause Chain

1. **EXQ-048**: `agent.e3.select()` called directly → beta_gate never wired
2. **EXQ-048b**: `agent.select_action()` used correctly → gate wired, BUT
   `update_after_execution()` never called → `_running_variance` frozen at
   `precision_init=0.5` → `0.5 < commit_threshold=0.40` always False →
   agent never commits → `beta_gate.elevate()` never fires
3. **EXQ-048c**: Adds `agent.e3.update_after_execution(z_world_t, harm_{t-1})` at
   each step start → running_variance evolves → commitment eventually fires

## Training Diagnostics

| Metric | Value |
|--------|-------|
| final_running_variance (post-train) | {train_out['final_running_variance']:.4f} |
| mean_committed_fraction (train) | {train_out['mean_committed_fraction']:.3f} |
| commit_threshold | {agent.e3.commit_threshold:.3f} |

## Beta Gate State During Eval

| Metric | Value |
|--------|-------|
| committed_step_count | {eval_out['committed_step_count']} |
| uncommitted_step_count | {eval_out['uncommitted_step_count']} |
| hold_rate_during_committed | {eval_out['hold_rate_during_committed']:.3f} |
| gate_release_events | {eval_out['gate_release_events']} |
| hold_count_total (all steps) | {eval_out['hold_count_total']} |
| propagation_count_total | {eval_out['propagation_count_total']} |
| calibration_gap_approach | {eval_out['calibration_gap_approach']:.4f} |
| mean_running_variance (eval) | {eval_out['mean_running_variance']:.4f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: committed_step_count >= 10 (agent commits) | {"PASS" if c1_pass else "FAIL"} | {eval_out['committed_step_count']} |
| C2: hold_rate_during_committed > 0.5 (gate holds) | {"PASS" if c2_pass else "FAIL"} | {eval_out['hold_rate_during_committed']:.3f} |
| C3: propagation_count > 0 (gate releases) | {"PASS" if c3_pass else "FAIL"} | {eval_out['propagation_count_total']} |
| C4: calibration_gap_approach > 0 (E3 functional) | {"PASS" if c4_pass else "FAIL"} | {eval_out['calibration_gap_approach']:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {eval_out['fatal_errors']} |

Criteria met: {criteria_met}/5 -> **{status}**
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
    parser.add_argument("--warmup",      type=int,   default=400)
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
