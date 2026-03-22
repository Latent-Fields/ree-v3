"""
V3-EXQ-049 -- MECH-090: Beta-Gated Policy Propagation

Claims: MECH-090

Motivation (2026-03-19):
  MECH-090: Beta oscillations gate E3→action_selection propagation.
  During committed action sequence: beta_gate.is_elevated = True -> E3 updates
  internally but policy output is held (propagate() returns None).
  At completion / stop-change signal: beta_gate releases -> E3 state propagates.

  BetaGate is implemented in ree_core/heartbeat/beta_gate.py.
  Agent sets beta state via:
    result.committed -> beta_gate.elevate() -> propagate() returns None
    not result.committed -> beta_gate.release() -> propagate() returns action

  This experiment measures the gate's behavior across a full eval run:
    - policy_hold_rate: fraction of steps where gate is elevated (holding)
    - committed_hold_concordance: when committed, gate is elevated
    - uncommitted_release_concordance: when uncommitted, gate is not elevated
    - hold_count and propagation_count from beta_gate.get_state()

  MECH-090 distinction from MECH-057b (EXQ-045):
    - EXQ-045 measured completion timing (gate fires at completion, not initiation)
    - EXQ-049 measures the POLICY CONTENT: is the held vs propagated content
      meaningfully different? Does propagation happen at the right moments?

  Specifically: beta gate should be elevated more during committed phases and
  less during non-committed phases (concordance test).

PASS criteria (ALL must hold):
  C1: committed_hold_concordance > 0.6   (gate elevated when committed ≥ 60% of committed steps)
  C2: uncommitted_release_concordance > 0.5  (gate not elevated when not committed ≥ 50%)
  C3: hold_count > 0                     (gate does hold at some point)
  C4: propagation_count > 0             (gate does propagate at some point)
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


EXPERIMENT_TYPE = "v3_exq_049_mech090_beta_gate"
CLAIM_IDS = ["MECH-090"]


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
) -> None:
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

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
            print(f"  [train] ep {ep+1}/{num_episodes}", flush=True)


def _eval_beta_concordance(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Measure concordance between commitment state and beta gate state.
    committed + elevated = correct hold (MECH-090 prediction)
    uncommitted + not elevated = correct release
    """
    agent.eval()
    agent.beta_gate.reset()

    committed_elevated_count   = 0  # committed AND gate elevated (correct)
    committed_not_elevated     = 0  # committed AND gate NOT elevated (unexpected)
    uncommitted_elevated       = 0  # NOT committed AND gate elevated (unexpected)
    uncommitted_not_elevated   = 0  # NOT committed AND gate NOT elevated (correct)
    total_steps = 0
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent   = agent.sense(obs_body, obs_world)
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
                is_elevated  = agent.beta_gate.is_elevated
                total_steps += 1

                if is_committed and is_elevated:
                    committed_elevated_count += 1
                elif is_committed and not is_elevated:
                    committed_not_elevated += 1
                elif not is_committed and is_elevated:
                    uncommitted_elevated += 1
                else:
                    uncommitted_not_elevated += 1

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            if done:
                break

    gate_state = agent.beta_gate.get_state()
    hold_count = gate_state["hold_count"]
    prop_count = gate_state["propagation_count"]

    n_committed   = committed_elevated_count + committed_not_elevated
    n_uncommitted = uncommitted_elevated + uncommitted_not_elevated

    committed_hold_concordance  = (
        committed_elevated_count / max(1, n_committed)
    )
    uncommitted_release_concordance = (
        uncommitted_not_elevated / max(1, n_uncommitted)
    )

    print(
        f"  committed_steps={n_committed}  uncommitted_steps={n_uncommitted}\n"
        f"  committed_hold_concordance={committed_hold_concordance:.3f}"
        f"  uncommitted_release_concordance={uncommitted_release_concordance:.3f}\n"
        f"  hold_count={hold_count}  propagation_count={prop_count}",
        flush=True,
    )

    return {
        "committed_elevated_count":          committed_elevated_count,
        "committed_not_elevated":            committed_not_elevated,
        "uncommitted_elevated":              uncommitted_elevated,
        "uncommitted_not_elevated":          uncommitted_not_elevated,
        "committed_hold_concordance":        committed_hold_concordance,
        "uncommitted_release_concordance":   uncommitted_release_concordance,
        "hold_count":                        hold_count,
        "propagation_count":                 prop_count,
        "n_committed_steps":                 n_committed,
        "n_uncommitted_steps":               n_uncommitted,
        "total_steps":                       total_steps,
        "fatal_errors":                      fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
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
        alpha_self=0.3,
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
        f"[V3-EXQ-049] MECH-090: Beta-Gated Policy Propagation\n"
        f"  warmup={warmup_episodes}  eval={eval_episodes}  alpha_world={alpha_world}",
        flush=True,
    )

    _train(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_episodes, steps_per_episode, world_dim,
    )

    print(f"\n[V3-EXQ-049] Eval -- beta gate concordance...", flush=True)
    eval_out = _eval_beta_concordance(agent, env, eval_episodes, steps_per_episode, world_dim)

    # PASS / FAIL
    c1_pass = eval_out["committed_hold_concordance"] > 0.6
    c2_pass = eval_out["uncommitted_release_concordance"] > 0.5
    c3_pass = eval_out["hold_count"] > 0
    c4_pass = eval_out["propagation_count"] > 0
    c5_pass = eval_out["fatal_errors"] == 0

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: committed_hold_concordance={eval_out['committed_hold_concordance']:.3f} <= 0.6"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: uncommitted_release_concordance={eval_out['uncommitted_release_concordance']:.3f} <= 0.5"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: hold_count={eval_out['hold_count']} == 0")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: propagation_count={eval_out['propagation_count']} == 0")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-049 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "committed_elevated_count":        float(eval_out["committed_elevated_count"]),
        "committed_not_elevated":          float(eval_out["committed_not_elevated"]),
        "uncommitted_elevated":            float(eval_out["uncommitted_elevated"]),
        "uncommitted_not_elevated":        float(eval_out["uncommitted_not_elevated"]),
        "committed_hold_concordance":      float(eval_out["committed_hold_concordance"]),
        "uncommitted_release_concordance": float(eval_out["uncommitted_release_concordance"]),
        "hold_count":                      float(eval_out["hold_count"]),
        "propagation_count":               float(eval_out["propagation_count"]),
        "n_committed_steps":               float(eval_out["n_committed_steps"]),
        "n_uncommitted_steps":             float(eval_out["n_uncommitted_steps"]),
        "fatal_error_count":               float(eval_out["fatal_errors"]),
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

    summary_markdown = f"""# V3-EXQ-049 -- MECH-090: Beta-Gated Policy Propagation

**Status:** {status}
**Claim:** MECH-090 -- beta gate holds E3 policy output during committed action
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Motivation

MECH-090: When the agent is committed to a trajectory, BetaGate.is_elevated = True
and propagate() returns None (policy output held). When uncommitted, gate releases.
This measures the concordance between commitment state and gate state.

## Beta Gate Concordance

| State | Count | Rate |
|-------|-------|------|
| committed + gate elevated (correct hold) | {eval_out['committed_elevated_count']} | {eval_out['committed_hold_concordance']:.3f} |
| committed + gate NOT elevated (unexpected) | {eval_out['committed_not_elevated']} | {1 - eval_out['committed_hold_concordance']:.3f} |
| uncommitted + NOT elevated (correct release) | {eval_out['uncommitted_not_elevated']} | {eval_out['uncommitted_release_concordance']:.3f} |
| uncommitted + gate elevated (unexpected) | {eval_out['uncommitted_elevated']} | {1 - eval_out['uncommitted_release_concordance']:.3f} |

- hold_count (total gate holds): {eval_out['hold_count']}
- propagation_count (total gate releases): {eval_out['propagation_count']}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: committed_hold_concordance > 0.6 | {"PASS" if c1_pass else "FAIL"} | {eval_out['committed_hold_concordance']:.3f} |
| C2: uncommitted_release_concordance > 0.5 | {"PASS" if c2_pass else "FAIL"} | {eval_out['uncommitted_release_concordance']:.3f} |
| C3: hold_count > 0 (gate holds) | {"PASS" if c3_pass else "FAIL"} | {eval_out['hold_count']} |
| C4: propagation_count > 0 (gate releases) | {"PASS" if c4_pass else "FAIL"} | {eval_out['propagation_count']} |
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
