"""
V3-EXQ-048b — MECH-057b: Trajectory Completion Gate (Fixed)

Claims: MECH-057b, MECH-090

Root cause of EXQ-048 FAIL:
  EXQ-048 called agent.e3.select(candidates, temperature=1.0) directly and then
  manually set agent._last_action. This bypasses agent.select_action(), which is
  the only code path that calls beta_gate.elevate() and beta_gate.propagate().
  As a result, gate.is_elevated was always False and hold_count=0, regardless of
  commitment state. The bug was purely an experiment instrumentation error — not a
  mechanism failure.

Fix:
  Replace:
    result = agent.e3.select(candidates, temperature=1.0)
    action = result.selected_action.detach()
    agent._last_action = action
  With:
    action = agent.select_action(candidates, ticks, temperature=1.0)
  agent.select_action() calls e3.select(), reads result.committed, calls
  beta_gate.elevate() or beta_gate.release(), then beta_gate.propagate(action).
  Gate state is now updated on every e3_tick.

Prerequisite: EXQ-042 PASS (HippocampalModule terrain prior trains, E3 selects trajectories).

Motivation (2026-03-20):
  MECH-057b: The commitment gate fires at trajectory COMPLETION, not at initiation.
  During a committed sequence:
    - BetaGate is elevated (beta_gate.is_elevated = True)
    - Policy output is blocked (propagate() returns None)
    - E3 continues updating internally
  At completion (sequence finishes or stop-change signal):
    - BetaGate is released (beta_gate.is_elevated = False)
    - Policy output propagates (propagate() returns the action tensor)

  MECH-057b is distinct from MECH-090 in that it specifically tests the TIMING
  of gate opening: gate opens when the trajectory's action-object sequence reaches
  completion, not when commitment begins.

  This experiment probes the completion gate by:
  1. Running the full pipeline and tracking beta_gate state step-by-step.
  2. Measuring whether the beta_gate is elevated during committed execution steps.
  3. Measuring whether the beta_gate releases at E3 tick boundaries (when a new
     trajectory is selected, the previous one is "complete").
  4. Computing policy_output_blocked_ratio = hold_count / (hold_count + prop_count)
     during committed steps (should be high).

  The key question: when the agent is committed to a trajectory, are policy outputs
  being held by the beta gate? And are they released at completion?

Protocol:
  1. Train agent (400 eps) — full pipeline to get E3 making committed selections.
  2. Eval (50 eps): step-by-step tracking of:
     - is_committed (agent.e3._committed_trajectory is not None)
     - beta_gate.is_elevated at each step
     - propagated (beta_gate.propagate result)
  3. Measure:
     - committed_step_count vs uncommitted_step_count
     - hold_count and propagation_count from beta_gate.get_state()
     - policy_blocked_during_committed = step_count where committed AND gate elevated
     - gate_release_count = transitions from elevated to not-elevated

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


EXPERIMENT_TYPE = "v3_exq_048b_mech057b_completion_gate_fixed"
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
                # FIX: routes through agent.select_action() which calls beta_gate.elevate()/propagate()
                action = agent.select_action(candidates, ticks, temperature=1.0)
            else:
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
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  e3_ticks={e3_tick_total}",
                flush=True,
            )

    return {"total_harm": total_harm, "e3_tick_total": e3_tick_total}


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
    committed_and_elevated = 0    # committed AND gate elevated (C2)
    committed_and_propagated = 0  # committed AND gate open (unexpected)
    gate_release_events = 0       # transitions from elevated->not-elevated
    approach_scores: List[float] = []
    none_scores:     List[float] = []
    fatal = 0
    prev_elevated = False

    agent.beta_gate.reset()

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        prev_elevated = False

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                ticks  = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                theta_z = agent.theta_buffer.summary()

            try:
                with torch.no_grad():
                    # FIX: routes through agent.select_action() which calls beta_gate.elevate()/propagate()
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                # Record step-level gate state
                # committed state: agent.e3._committed_trajectory is not None
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

                # Count gate release events (elevated -> not-elevated)
                if prev_elevated and not is_elevated:
                    gate_release_events += 1
                prev_elevated = is_elevated

                # E3 calibration probe
                with torch.no_grad():
                    score = float(agent.e3.harm_eval(theta_z).item())

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
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

            if done:
                break

    beta_state = agent.beta_gate.get_state()
    hold_count  = beta_state["hold_count"]
    prop_count  = beta_state["propagation_count"]

    hold_rate_during_committed = (
        committed_and_elevated / max(1, committed_steps)
    )
    cal_gap = _mean_safe(approach_scores) - _mean_safe(none_scores)

    print(
        f"  committed_steps={committed_steps}  uncommitted={uncommitted_steps}\n"
        f"  hold_rate_during_committed={hold_rate_during_committed:.3f}"
        f"  gate_releases={gate_release_events}\n"
        f"  hold_count={hold_count}  propagation_count={prop_count}"
        f"  cal_gap_approach={cal_gap:.4f}",
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
        f"[V3-EXQ-048b] MECH-057b: Trajectory Completion Gate (Fixed)\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}",
        flush=True,
    )

    _train(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_episodes, steps_per_episode, world_dim,
    )

    print(f"\n[V3-EXQ-048b] Eval -- tracking beta gate state...", flush=True)
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

    print(f"\nV3-EXQ-048b verdict: {status}  ({criteria_met}/5)", flush=True)
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

    summary_markdown = f"""# V3-EXQ-048b -- MECH-057b: Trajectory Completion Gate (Fixed)

**Status:** {status}
**Claims:** MECH-057b, MECH-090
**Fix:** Routes through agent.select_action() instead of agent.e3.select() directly
**Prerequisite:** EXQ-042 PASS (HippocampalModule terrain prior functional)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Root Cause of EXQ-048 FAIL

EXQ-048 called `agent.e3.select()` directly and set `agent._last_action` manually,
bypassing `agent.select_action()`. The `select_action()` method is the only code
path that calls `beta_gate.elevate()` and `beta_gate.propagate()`. As a result,
`gate.is_elevated` was always False and `hold_count=0` in EXQ-048, regardless of
commitment state. This was an experiment instrumentation error, not a mechanism failure.

## Motivation

MECH-057b: BetaGate should hold policy output during committed action execution,
and release at completion (E3 tick boundary where a new trajectory is selected).
This tests whether the gate behaves correctly: elevated during sequences, released
when sequences complete.

Committed state read from: agent.e3._committed_trajectory is not None
Gate state read from: agent.beta_gate.is_elevated, agent.beta_gate.get_state()

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
