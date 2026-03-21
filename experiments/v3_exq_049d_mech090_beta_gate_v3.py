"""
V3-EXQ-049d — MECH-090: Beta-Gated Policy Propagation (chicken-and-egg fix)

Claims: MECH-090

Root cause of EXQ-049c FAIL (predicted before results):
  EXQ-049c calls post_action_update() to evolve _running_variance, but
  post_action_update() has its own guard:
      if self._committed_trajectory is not None: ...update_running_variance()
  _committed_trajectory is set only when result.committed=True, which requires
  _running_variance < commit_threshold. Starting at precision_init=0.5 > 0.40,
  variance never updates → committed never fires. Identical chicken-and-egg to
  EXQ-048c, which failed for the same reason.

  Additionally, is_committed used agent.e3._committed_trajectory is not None,
  which post_action_update() resets to None at the start of every step. Even if
  commitment did fire at an e3_tick, the probe reads False on the next step.

Fixes:
  Fix 1 (training): After the world_forward loss, directly call
      agent.e3.update_running_variance(wf_err)
  where wf_err = wf_pred.detach() - zw1_b. This bypasses the _committed_trajectory
  guard and gets variance moving from the very first training batch.

  Fix 2 (eval): Replace
      is_committed = agent.e3._committed_trajectory is not None
  with
      is_committed = agent.e3._running_variance < agent.e3.commit_threshold
  This correctly reflects commitment state throughout the episode, not just on
  the single step when _committed_trajectory is transiently set.

Root cause chain:
  EXQ-049:  select() bypassed → gate never wired
  EXQ-049b: gate wired, post_action_update missing → variance frozen
  EXQ-049c: post_action_update called, but its own _committed_trajectory guard
            keeps variance frozen — chicken-and-egg still intact
  EXQ-049d: direct update_running_variance() from wf error — deadlock broken

PASS criteria (ALL must hold):
  C1: committed_hold_concordance > 0.6   (gate elevated when committed >= 60%)
  C2: uncommitted_release_concordance > 0.5  (gate not elevated when uncommitted >= 50%)
  C3: hold_count > 0                     (gate does hold at some point)
  C4: propagation_count > 0              (gate does propagate at some point)
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


EXPERIMENT_TYPE = "v3_exq_049d_mech090_beta_gate_v3"
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
) -> Dict:
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    committed_fraction_log: List[float] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None
        harm_prev: float = 0.0

        for step in range(steps_per_episode):
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

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Fix 2 probe (training fraction): use variance-based criterion
            committed_fraction_log.append(
                1.0 if agent.e3._running_variance < agent.e3.commit_threshold else 0.0
            )

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
                # Fix 1: direct variance update — breaks the _committed_trajectory
                # chicken-and-egg in post_action_update().
                with torch.no_grad():
                    wf_err = (wf_pred.detach() - zw1_b).detach()
                    agent.e3.update_running_variance(wf_err)

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
                f"  [train] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.4f}  commit_thresh={ct:.3f}"
                f"  committed_frac={cf:.3f}",
                flush=True,
            )

    return {
        "final_running_variance": agent.e3._running_variance,
        "mean_committed_fraction": _mean_safe(committed_fraction_log),
    }


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

    committed_elevated_count = 0
    committed_not_elevated   = 0
    uncommitted_elevated     = 0
    uncommitted_not_elevated = 0
    total_steps = 0
    fatal = 0
    running_variances: List[float] = []

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        harm_prev: float = 0.0

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            running_variances.append(agent.e3._running_variance)

            with torch.no_grad():
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                with torch.no_grad():
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action

                # Fix 2: use variance-based commitment criterion, not
                # _committed_trajectory (which post_action_update resets every step).
                is_committed = agent.e3._running_variance < agent.e3.commit_threshold
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

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            harm_prev = float(harm_signal)
            if done:
                break

    gate_state = agent.beta_gate.get_state()
    hold_count = gate_state["hold_count"]
    prop_count = gate_state["propagation_count"]

    n_committed   = committed_elevated_count + committed_not_elevated
    n_uncommitted = uncommitted_elevated + uncommitted_not_elevated

    committed_hold_concordance = (
        committed_elevated_count / max(1, n_committed)
    )
    uncommitted_release_concordance = (
        uncommitted_not_elevated / max(1, n_uncommitted)
    )
    mean_rv = _mean_safe(running_variances)

    print(
        f"  committed_steps={n_committed}  uncommitted_steps={n_uncommitted}\n"
        f"  committed_hold_concordance={committed_hold_concordance:.3f}"
        f"  uncommitted_release_concordance={uncommitted_release_concordance:.3f}\n"
        f"  hold_count={hold_count}  propagation_count={prop_count}"
        f"  mean_running_variance={mean_rv:.4f}",
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
        "mean_running_variance":             mean_rv,
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
        f"[V3-EXQ-049d] MECH-090: Beta-Gated Policy Propagation (chicken-and-egg fix)\n"
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
        f"\n[V3-EXQ-049d] Post-train:"
        f"  running_var={train_out['final_running_variance']:.4f}"
        f"  committed_frac={train_out['mean_committed_fraction']:.3f}",
        flush=True,
    )
    print(f"\n[V3-EXQ-049d] Eval -- beta gate concordance...", flush=True)
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

    print(f"\nV3-EXQ-049d verdict: {status}  ({criteria_met}/5)", flush=True)
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
        "mean_running_variance":           float(eval_out["mean_running_variance"]),
        "final_running_variance":          float(train_out["final_running_variance"]),
        "mean_committed_fraction_train":   float(train_out["mean_committed_fraction"]),
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

    summary_markdown = f"""# V3-EXQ-049d -- MECH-090: Beta-Gated Policy Propagation (chicken-and-egg fix)

**Status:** {status}
**Claim:** MECH-090 -- beta gate holds E3 policy output during committed action
**Fix 1:** Direct update_running_variance(wf_err) after world_forward loss — breaks _committed_trajectory deadlock
**Fix 2:** is_committed = _running_variance < commit_threshold (not _committed_trajectory probe)
**Prior attempts:** EXQ-049/049b (gate not wired), EXQ-049c (post_action_update still stuck in deadlock)
**alpha_world:** {alpha_world}
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## Root Cause Chain

1. **EXQ-049**: `agent.e3.select()` bypassed → gate never wired
2. **EXQ-049b**: gate wired, `post_action_update` missing → variance frozen
3. **EXQ-049c**: `post_action_update` called, but its own `_committed_trajectory` guard
   re-creates the deadlock — variance still never moves
4. **EXQ-049d**: `update_running_variance(wf_err)` called directly → deadlock broken

## Training Diagnostics

| Metric | Value |
|--------|-------|
| final_running_variance (post-train) | {train_out['final_running_variance']:.4f} |
| mean_committed_fraction (train) | {train_out['mean_committed_fraction']:.3f} |
| commit_threshold | {agent.e3.commit_threshold:.3f} |

## Beta Gate Concordance

| State | Count | Rate |
|-------|-------|------|
| committed + gate elevated (correct hold) | {eval_out['committed_elevated_count']} | {eval_out['committed_hold_concordance']:.3f} |
| committed + gate NOT elevated (unexpected) | {eval_out['committed_not_elevated']} | {1.0 - eval_out['committed_hold_concordance']:.3f} |
| uncommitted + NOT elevated (correct release) | {eval_out['uncommitted_not_elevated']} | {eval_out['uncommitted_release_concordance']:.3f} |
| uncommitted + gate elevated (unexpected) | {eval_out['uncommitted_elevated']} | {1.0 - eval_out['uncommitted_release_concordance']:.3f} |

- hold_count (total gate holds): {eval_out['hold_count']}
- propagation_count (total gate releases): {eval_out['propagation_count']}
- mean_running_variance (eval): {eval_out['mean_running_variance']:.4f}

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
